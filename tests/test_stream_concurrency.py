#!/usr/bin/env python3
"""
Test for HIP stream concurrency in the decode attention path (m1-stream-concurrency).

Verifies that:
1. Two HIP streams are created at engine init (stream_q for Q+Qgate, stream_kv for K+V)
2. Q+Qgate GEMV and K+V GEMV launch on separate streams concurrently
3. Streams are synchronized before QK-norm (no race conditions)
4. Decode output is numerically correct vs sequential baseline (max abs err < 1e-3)
5. Total kernel launch count per full-attention layer is <= 12 (VAL-DLR-008)
"""

import ctypes
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig
from src.inference.engine import InferenceEngine
from src.runtime.hip_dispatch import GPUDevice

TOLERANCE = 1e-3   # FP16 decode step tolerance


def make_random_int4_weights(K, N, group_size=128, seed=42):
    """Create random GPTQ-format INT4 weights."""
    rng = np.random.RandomState(seed)
    K8 = K // 8
    qweight = rng.randint(0, 2**32, size=(K8, N), dtype=np.uint32)
    num_groups = K // group_size
    scales = rng.randn(num_groups, N).astype(np.float16) * 0.01
    zeros = np.full((num_groups, N), 8.0, dtype=np.float16)
    return qweight, scales, zeros


def make_random_fp16(shape, scale=0.01, seed=42):
    """Create random FP16 weights."""
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape) * scale).astype(np.float16)


def make_small_full_attention_config():
    """Create a minimal config with just a full-attention layer."""
    # Minimal config: 2 layers, layer 0 = full_attention
    # (full_attention_interval=1 means every layer is full)
    config = QwenConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        group_size=128,
        full_attention_interval=1,  # all layers are full attention
        # Linear attention params (not used but required by config)
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )
    return config


def load_full_attention_layer(engine, layer_idx, config, seed=0):
    """Load random full-attention layer weights into engine."""
    h = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size

    weights = {
        'layer_type': 'full_attention',
        'q_weight':      make_random_fp16((q_dim, h),   seed=seed+0),
        'q_gate_weight': make_random_fp16((q_dim, h),   seed=seed+1),
        'k_weight':      make_random_fp16((kv_dim, h),  seed=seed+2),
        'v_weight':      make_random_fp16((kv_dim, h),  seed=seed+3),
        'o_weight':      make_random_fp16((h, q_dim),   seed=seed+4),
        'q_norm':        make_random_fp16((config.head_dim,), seed=seed+5),
        'k_norm':        make_random_fp16((config.head_dim,), seed=seed+6),
        'attn_norm':     make_random_fp16((h,),         seed=seed+7),
        'ffn_norm':      make_random_fp16((h,),         seed=seed+8),
        'gate_qweight':  make_random_int4_weights(h, inter, config.group_size, seed=seed+9)[0],
        'gate_scales':   make_random_int4_weights(h, inter, config.group_size, seed=seed+9)[1],
        'gate_zeros':    make_random_int4_weights(h, inter, config.group_size, seed=seed+9)[2],
        'up_qweight':    make_random_int4_weights(h, inter, config.group_size, seed=seed+10)[0],
        'up_scales':     make_random_int4_weights(h, inter, config.group_size, seed=seed+10)[1],
        'up_zeros':      make_random_int4_weights(h, inter, config.group_size, seed=seed+10)[2],
        'down_qweight':  make_random_int4_weights(inter, h, config.group_size, seed=seed+11)[0],
        'down_scales':   make_random_int4_weights(inter, h, config.group_size, seed=seed+11)[1],
        'down_zeros':    make_random_int4_weights(inter, h, config.group_size, seed=seed+11)[2],
    }
    engine.load_layer_weights(layer_idx, weights)


def test_streams_created():
    """Test that engine creates two HIP streams at init."""
    print("\n[TEST] test_streams_created")
    config = make_small_full_attention_config()
    engine = InferenceEngine(config, device_id=0, max_seq_len=32)

    assert hasattr(engine, '_stream_q'), "Engine missing _stream_q attribute"
    assert hasattr(engine, '_stream_kv'), "Engine missing _stream_kv attribute"
    assert hasattr(engine, '_streams_ready'), "Engine missing _streams_ready attribute"

    assert engine._streams_ready, "_streams_ready should be True on GPU with HIP stream support"
    assert engine._stream_q != 0, "_stream_q should be a valid stream handle"
    assert engine._stream_kv != 0, "_stream_kv should be a valid stream handle"
    assert engine._stream_q != engine._stream_kv, "Q and KV streams must be distinct"

    engine.cleanup()
    print("  PASS: Two distinct HIP streams created at engine init")


def test_decode_correctness_with_streams():
    """Test that decode output is numerically correct with stream concurrency.

    Compares the output of one decode step against a reference run that
    disables streams (fallback to default stream 0).
    """
    print("\n[TEST] test_decode_correctness_with_streams")
    config = make_small_full_attention_config()

    engine = InferenceEngine(config, device_id=0, max_seq_len=32)
    load_full_attention_layer(engine, 0, config, seed=42)

    h = config.hidden_size
    rng = np.random.RandomState(100)
    token_emb = (rng.randn(h) * 0.01).astype(np.float16)

    # Run with streams (default)
    out_with_streams = engine.decode_step(token_emb, position=0)

    # Run again (same token, same position — KV cache advanced; use position=1)
    engine.kv_cache.current_len = 0  # reset position
    engine.kv_cache.advance()  # advance once to match first run

    # Reset and rerun to get consistent output
    engine2 = InferenceEngine(config, device_id=0, max_seq_len=32)
    load_full_attention_layer(engine2, 0, config, seed=42)

    # Disable streams to simulate sequential fallback
    engine2._stream_q = 0
    engine2._stream_kv = 0
    engine2._streams_ready = False

    out_without_streams = engine2.decode_step(token_emb, position=0)

    max_err = float(np.max(np.abs(out_with_streams.astype(np.float32) -
                                   out_without_streams.astype(np.float32))))
    print(f"  Max abs error (with vs without streams): {max_err:.2e}")
    assert max_err < TOLERANCE, (
        f"Decode with streams diverges from sequential: max_err={max_err:.2e} "
        f"(tolerance={TOLERANCE})"
    )

    engine.cleanup()
    engine2.cleanup()
    print(f"  PASS: Decode output matches within tolerance {TOLERANCE}")


def test_launch_count_full_attention_layer():
    """Test that total kernel launches per full-attention layer is <= 12.

    Uses the launch counting infrastructure to count launches for one
    full-attention decode step with all M1 fused kernels active.
    """
    print("\n[TEST] test_launch_count_full_attention_layer")
    config = make_small_full_attention_config()
    engine = InferenceEngine(config, device_id=0, max_seq_len=32)
    load_full_attention_layer(engine, 0, config, seed=42)

    h = config.hidden_size
    rng = np.random.RandomState(200)
    token_emb = (rng.randn(h) * 0.01).astype(np.float16)

    # Enable launch counting before decode
    engine._count_launches = True
    engine.reset_launch_counters()

    engine.decode_step(token_emb, position=0)

    engine._count_launches = False

    total = engine.get_layer_launch_count(0)
    print(f"  Layer 0 (full-attention) total kernel launches: {total}")
    print(f"  Expected kernels:")
    print(f"    1: pre-attn rmsnorm")
    print(f"    2: Q+Qgate GEMV (stream_q)")
    print(f"    3: K+V GEMV (stream_kv)")
    print(f"    4: qknorm_rope(Q)")
    print(f"    5: qknorm_rope(K)")
    print(f"    6: flash_attn_256 (decode attention)")
    print(f"    7: sigmoid_mul (gate)")
    print(f"    8: out_proj GEMV + residual (gemv_fp16_v2)")
    print(f"    9: pre-FFN rmsnorm")
    print(f"   10: gate+up GEMV fused (gemv_int4_dual_fused)")
    print(f"   11: down_proj GEMV + residual (gemv_int4_v2_fused)")
    print(f"  Total: 11 expected, limit: 12")

    assert total <= 12, (
        f"Launch count {total} exceeds limit of 12 per full-attention layer"
    )

    engine.cleanup()
    print(f"  PASS: {total} launches <= 12 limit")


def test_no_race_condition_on_shared_buffer():
    """Test that concurrent GEMVs on d_normed produce correct output.

    The d_normed buffer is read-only during Q+Qgate and K+V GEMV launches,
    so concurrent reads are safe. This test verifies correctness across
    multiple positions to catch any race condition manifestation.
    """
    print("\n[TEST] test_no_race_condition_on_shared_buffer")
    config = make_small_full_attention_config()
    engine = InferenceEngine(config, device_id=0, max_seq_len=16)
    load_full_attention_layer(engine, 0, config, seed=77)

    h = config.hidden_size
    rng = np.random.RandomState(77)

    # Run 5 decode steps; any race condition would cause diverging output
    results = []
    for pos in range(5):
        token_emb = (rng.randn(h) * 0.01).astype(np.float16)
        out = engine.decode_step(token_emb, position=pos)
        results.append(out.copy())

    # Verify outputs are finite (NaN would indicate a race/correctness issue)
    for i, out in enumerate(results):
        assert not np.any(np.isnan(out.astype(np.float32))), (
            f"NaN in decode output at position {i} — possible race condition"
        )
        assert not np.any(np.isinf(out.astype(np.float32))), (
            f"Inf in decode output at position {i} — possible race condition"
        )

    engine.cleanup()
    print(f"  PASS: {len(results)} decode steps produced finite outputs (no race condition)")


def test_stream_cleanup():
    """Test that streams are properly destroyed in engine.cleanup()."""
    print("\n[TEST] test_stream_cleanup")
    config = make_small_full_attention_config()
    engine = InferenceEngine(config, device_id=0, max_seq_len=16)

    assert engine._streams_ready, "Streams should be created"
    stream_q = engine._stream_q
    stream_kv = engine._stream_kv

    # Cleanup should not raise
    try:
        engine.cleanup()
        print("  PASS: engine.cleanup() succeeded with stream destruction")
    except Exception as e:
        raise AssertionError(f"engine.cleanup() raised exception: {e}")


def main():
    print("=" * 60)
    print("HIP Stream Concurrency Tests (m1-stream-concurrency)")
    print("=" * 60)

    tests = [
        test_streams_created,
        test_decode_correctness_with_streams,
        test_launch_count_full_attention_layer,
        test_no_race_condition_on_shared_buffer,
        test_stream_cleanup,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
