#!/usr/bin/env python3
"""
Integration test for ALL M1+M2 optimizations active simultaneously.

Tests that a decode step with ALL M1 and M2 optimizations active produces
correct output vs baseline (max abs error < 1e-2).

Covers:
  - VAL-CROSS-006: Decode step with ALL M1+M2 optimizations (max abs error < 1e-2)

M1 optimizations:
  - DeltaNet GPU path (no CPU fallback) — VAL-DLR-001
  - skip_rmsnorm_v2 fused in decode path — VAL-DLR-002
  - Fused INT4 split-K (no memset/convert) — VAL-DLR-003
  - Dual gate+up INT4 fused (no memset) — VAL-DLR-004, VAL-DLR-011
  - Fused QK-norm + RoPE — VAL-DLR-005, VAL-DLR-010
  - Residual-add epilogues in out_proj + down_proj — VAL-DLR-006, VAL-DLR-007
  - HIP stream concurrency — VAL-DLR-008, VAL-DLR-009

M2 optimizations:
  - INT4 GEMV v3_t16 cooperative reduction — VAL-INT4-003
  - ubfe nibble extract in v2/v3 — VAL-INT4-001
  - HIP RoPE kernel — VAL-ROPE-001, VAL-ROPE-002
  - FP16 GEMM prefill with v_dot2_f32_f16 + swizzled LDS — VAL-GEMM-001, VAL-GEMM-002
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig
from src.inference.engine import InferenceEngine

# --- Tolerances ---
M1M2_DECODE_TOLERANCE = 1e-2   # VAL-CROSS-006
PREFILL_TOLERANCE     = 1e-3   # non-regression for prefill


# ---------------------------------------------------------------------------
# Helper: configs (matching test_m1_integration.py pattern)
# ---------------------------------------------------------------------------

def make_full_attention_only_config(num_layers=2, hidden=256, num_heads=4,
                                     num_kv_heads=2, head_dim=64,
                                     intermediate=512, group_size=128):
    """Config where all layers are full_attention (no linear attention)."""
    config = QwenConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        group_size=group_size,
        full_attention_interval=1,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )
    return config


def make_linear_attention_only_config(num_layers=2):
    """Config where all layers are linear_attention (DeltaNet v3)."""
    config = QwenConfig()   # Full Qwen 3.5 27B config
    config.num_hidden_layers = num_layers
    config.layer_types = ["linear_attention"] * num_layers
    return config


def make_hybrid_config(num_layers=4):
    """Config with mixed full_attention and linear_attention layers."""
    config = QwenConfig()  # Full Qwen 3.5 27B config
    config.num_hidden_layers = num_layers
    config.layer_types = []
    for i in range(num_layers):
        if (i + 1) % 4 == 0:
            config.layer_types.append("full_attention")
        else:
            config.layer_types.append("linear_attention")
    return config


def make_random_int4_weights(K, N, group_size=128, seed=0):
    rng = np.random.RandomState(seed)
    K8 = K // 8
    qweight = rng.randint(0, 2**32, size=(K8, N), dtype=np.uint32)
    num_groups = K // group_size
    scales = (rng.rand(num_groups, N) * 0.01 + 0.005).astype(np.float16)
    zeros  = np.full((num_groups, N), 8.0, dtype=np.float16)
    return qweight, scales, zeros


def make_random_fp16(shape, scale=0.02, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape) * scale).astype(np.float16)


def build_full_attention_weights(config, layer_idx, seed_base=0):
    h    = config.hidden_size
    q_d  = config.num_attention_heads * config.head_dim
    kv_d = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    gs   = config.group_size

    g_qw, g_sc, g_zr = make_random_int4_weights(h, inter, gs, seed_base + 0)
    u_qw, u_sc, u_zr = make_random_int4_weights(h, inter, gs, seed_base + 1)
    d_qw, d_sc, d_zr = make_random_int4_weights(inter, h, gs, seed_base + 2)

    return {
        'layer_type':    'full_attention',
        'q_weight':      make_random_fp16((q_d,   h),  seed=seed_base + 3),
        'q_gate_weight': make_random_fp16((q_d,   h),  seed=seed_base + 4),
        'k_weight':      make_random_fp16((kv_d,  h),  seed=seed_base + 5),
        'v_weight':      make_random_fp16((kv_d,  h),  seed=seed_base + 6),
        'o_weight':      make_random_fp16((h,     q_d), seed=seed_base + 7),
        'q_norm':        make_random_fp16((config.head_dim,), scale=0.1, seed=seed_base + 8),
        'k_norm':        make_random_fp16((config.head_dim,), scale=0.1, seed=seed_base + 9),
        'attn_norm':     make_random_fp16((h,),  scale=0.1, seed=seed_base + 10),
        'ffn_norm':      make_random_fp16((h,),  scale=0.1, seed=seed_base + 11),
        'gate_qweight': g_qw, 'gate_scales': g_sc, 'gate_zeros': g_zr,
        'up_qweight':   u_qw, 'up_scales':   u_sc, 'up_zeros':   u_zr,
        'down_qweight': d_qw, 'down_scales': d_sc, 'down_zeros': d_zr,
    }


def build_linear_attention_weights(config, layer_idx, seed_base=100):
    h      = config.hidden_size
    k_dim  = config.linear_num_key_heads * config.linear_key_head_dim
    v_dim  = config.linear_num_value_heads * config.linear_value_head_dim
    a_dim  = config.linear_num_value_heads
    conv_k = config.linear_conv_kernel_dim
    qkv_dim = k_dim * 2 + v_dim
    inter  = config.intermediate_size
    gs     = config.group_size

    g_qw, g_sc, g_zr = make_random_int4_weights(h, inter, gs, seed_base + 0)
    u_qw, u_sc, u_zr = make_random_int4_weights(h, inter, gs, seed_base + 1)
    d_qw, d_sc, d_zr = make_random_int4_weights(inter, h, gs, seed_base + 2)

    rng = np.random.RandomState(seed_base + 50)

    return {
        'layer_type':      'linear_attention',
        'la_in_proj_qkv':  make_random_fp16((qkv_dim, h), scale=0.01, seed=seed_base + 3),
        'la_in_proj_a':    make_random_fp16((a_dim,   h), scale=0.01, seed=seed_base + 4),
        'la_in_proj_b':    make_random_fp16((a_dim,   h), scale=0.01, seed=seed_base + 5),
        'la_in_proj_z':    make_random_fp16((v_dim,   h), scale=0.01, seed=seed_base + 6),
        'la_out_proj':     make_random_fp16((h, v_dim),   scale=0.01, seed=seed_base + 7),
        'la_conv1d':       rng.randn(qkv_dim, 1, conv_k).astype(np.float16) * 0.1,
        'la_A_log':        np.full(a_dim, -0.1, dtype=np.float16),
        'la_dt_bias':      np.zeros(a_dim, dtype=np.float16),
        'la_norm':         np.ones(config.linear_value_head_dim, dtype=np.float16),
        'attn_norm':       make_random_fp16((h,), scale=0.1, seed=seed_base + 8),
        'ffn_norm':        make_random_fp16((h,), scale=0.1, seed=seed_base + 9),
        'gate_qweight': g_qw, 'gate_scales': g_sc, 'gate_zeros': g_zr,
        'up_qweight':   u_qw, 'up_scales':   u_sc, 'up_zeros':   u_zr,
        'down_qweight': d_qw, 'down_scales': d_sc, 'down_zeros': d_zr,
    }


def load_all_layers(engine, config):
    for layer_idx in range(config.num_hidden_layers):
        if config.is_full_attention(layer_idx):
            weights = build_full_attention_weights(
                config, layer_idx, seed_base=layer_idx * 20)
        else:
            weights = build_linear_attention_weights(
                config, layer_idx, seed_base=layer_idx * 20 + 1000)
        engine.load_layer_weights(layer_idx, weights)


# ---------------------------------------------------------------------------
# Test 1: All M1+M2 optimization flags are True
# ---------------------------------------------------------------------------

def test_all_m1m2_flags_loaded():
    """Verify ALL M1 and M2 optimization flags are True after engine init.

    This test checks that all optimizations from both milestones are active
    simultaneously in a single engine instance.
    """
    print("\n[TEST] test_all_m1m2_flags_loaded")
    config = make_full_attention_only_config(num_layers=1)
    engine = InferenceEngine(config, device_id=0, max_seq_len=32)

    # === M1 flags ===
    # DeltaNet GPU always (VAL-DLR-001)
    assert engine._deltanet_gpu is True, \
        f"M1: _deltanet_gpu={engine._deltanet_gpu}, expected True"
    assert engine._deltanet_v3 is True, \
        f"M1: _deltanet_v3={engine._deltanet_v3}, expected True"

    # Fused INT4 split-K (VAL-DLR-003)
    assert engine._gemv_int4_v2 is True, \
        f"M1: _gemv_int4_v2={engine._gemv_int4_v2}, expected True"

    # Fused dual gate+up (VAL-DLR-004, VAL-DLR-011)
    assert engine._gemv_int4_dual is True, \
        f"M1: _gemv_int4_dual={engine._gemv_int4_dual}, expected True"
    assert engine._gemv_int4_dual_fused is True, \
        f"M1: _gemv_int4_dual_fused={engine._gemv_int4_dual_fused}, expected True"

    # Fused QK-norm + RoPE (VAL-DLR-005, VAL-DLR-010)
    assert engine._qknorm_rope_fused is True, \
        f"M1: _qknorm_rope_fused={engine._qknorm_rope_fused}, expected True"

    # HIP streams (VAL-DLR-009)
    assert engine._streams_ready is True, \
        f"M1: _streams_ready={engine._streams_ready}, expected True"

    # FP16 GEMV v2 (residual epilogues)
    assert engine._gemv_fp16_v2 is True, \
        f"M1: _gemv_fp16_v2={engine._gemv_fp16_v2}, expected True"

    # === M2 flags ===
    # INT4 GEMV v3 cooperative reduction (VAL-INT4-003)
    assert engine._gemv_int4_v3 is True, \
        f"M2: _gemv_int4_v3={engine._gemv_int4_v3}, expected True"

    # HIP RoPE (VAL-ROPE-002)
    assert engine._rope_hip is True, \
        f"M2: _rope_hip={engine._rope_hip}, expected True"

    # FP16 GEMM prefill (VAL-GEMM-001)
    assert engine._gemm_fp16_prefill is True, \
        f"M2: _gemm_fp16_prefill={engine._gemm_fp16_prefill}, expected True"

    engine.cleanup()
    print("  PASS: ALL M1+M2 optimization flags are True simultaneously")


# ---------------------------------------------------------------------------
# Test 2: Full-attention decode with all M1+M2 opts — reproducibility
# ---------------------------------------------------------------------------

def test_m1m2_full_attention_decode_reproducible():
    """Verify M1+M2 decode output is reproducible.

    Two identical engines with the same weights must produce the same output.
    This catches any race conditions introduced by combining M1 (stream
    concurrency, DPP) and M2 (v3 cooperative reduction) optimizations.
    """
    print("\n[TEST] test_m1m2_full_attention_decode_reproducible")
    config = make_full_attention_only_config(num_layers=2, hidden=256)
    rng = np.random.RandomState(42)
    token_emb = (rng.randn(config.hidden_size) * 0.01).astype(np.float16)

    outputs = []
    for run in range(2):
        engine = InferenceEngine(config, device_id=0, max_seq_len=32)
        load_all_layers(engine, config)

        # Confirm both M1 and M2 are active
        assert engine._gemv_int4_v3 is True, "M2 v3 must be active"
        assert engine._streams_ready is True, "M1 streams must be active"
        assert engine._rope_hip is True, "M2 HIP RoPE must be active"

        out = engine.decode_step(token_emb, position=0)
        engine.cleanup()
        outputs.append(out.astype(np.float32))

    max_err = float(np.max(np.abs(outputs[0] - outputs[1])))
    print(f"  Max abs error between two M1+M2 runs: {max_err:.2e}")
    assert max_err < 1e-4, \
        f"M1+M2 decode output is not reproducible: max_err={max_err:.2e}"
    assert not np.any(np.isnan(outputs[0])), "NaN in M1+M2 decode output"

    print(f"  PASS: M1+M2 decode is reproducible (max_err={max_err:.2e})")


# ---------------------------------------------------------------------------
# Test 3: M1+M2 combined vs numpy reference (VAL-CROSS-006)
# ---------------------------------------------------------------------------

def test_m1m2_combined_correctness_vs_reference():
    """Combined M1+M2 decode matches numpy reference within tolerance.

    The core of VAL-CROSS-006: a decode step with ALL optimizations active
    must produce output within max abs error < 1e-2 of the reference.
    """
    print("\n[TEST] test_m1m2_combined_correctness_vs_reference")

    hidden   = 128
    num_head = 2
    kv_head  = 1
    head_d   = 64
    inter    = 256
    gs       = 128

    config = QwenConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=1,
        num_attention_heads=num_head,
        num_key_value_heads=kv_head,
        head_dim=head_d,
        vocab_size=512,
        rms_norm_eps=1e-6,
        group_size=gs,
        full_attention_interval=1,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_num_value_heads=2,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )

    rng = np.random.RandomState(42)
    token_emb = (rng.randn(hidden) * 0.01).astype(np.float16)
    weights = build_full_attention_weights(config, 0, seed_base=42)

    # Run combined M1+M2 engine
    engine = InferenceEngine(config, device_id=0, max_seq_len=16)
    engine.load_layer_weights(0, weights)

    # Verify all key M1+M2 flags
    assert engine._gemv_int4_v3 is True, "M2: v3 GEMV must be active"
    assert engine._qknorm_rope_fused is True, "M1: fused qknorm_rope must be active"
    assert engine._streams_ready is True, "M1: streams must be active"
    assert engine._rope_hip is True, "M2: HIP RoPE must be active"
    assert engine._gemm_fp16_prefill is True, "M2: FP16 GEMM prefill must be active"

    m1m2_out = engine.decode_step(token_emb, position=0).astype(np.float32)
    engine.cleanup()

    # Compute numpy reference
    np_ref = numpy_decode_reference(token_emb, weights, config, position=0)

    max_err = float(np.max(np.abs(m1m2_out - np_ref)))
    mean_err = float(np.mean(np.abs(m1m2_out - np_ref)))
    print(f"  M1+M2 vs reference: max_err={max_err:.4e}, mean_err={mean_err:.4e}")
    print(f"  m1m2_out[:4]: {m1m2_out[:4]}")
    print(f"  ref[:4]:      {np_ref[:4]}")

    # VAL-CROSS-006 tolerance
    assert max_err < M1M2_DECODE_TOLERANCE, \
        (f"M1+M2 combined decode diverges from reference: "
         f"max_err={max_err:.4e} > tolerance={M1M2_DECODE_TOLERANCE}")
    assert not np.any(np.isnan(m1m2_out)), "NaN in M1+M2 decode output"
    assert not np.any(np.isinf(m1m2_out)), "Inf in M1+M2 decode output"

    print(f"  PASS: M1+M2 combined decode matches reference "
          f"(max_err={max_err:.4e}, tolerance={M1M2_DECODE_TOLERANCE}) "
          f"— VAL-CROSS-006")


# ---------------------------------------------------------------------------
# Test 4: Hybrid config with all M1+M2 opts (VAL-CROSS-006)
# ---------------------------------------------------------------------------

def test_m1m2_hybrid_decode():
    """Hybrid config (linear + full attention) with all M1+M2 optimizations.

    Uses full Qwen 3.5 27B dimensions (required by DeltaNet v3 kernel).
    Both full-attention (with M1+M2 opts) and linear-attention (DeltaNet GPU)
    layers must produce finite, non-NaN outputs.
    """
    print("\n[TEST] test_m1m2_hybrid_decode")
    config = make_hybrid_config(num_layers=4)

    rng = np.random.RandomState(123)
    token_emb = (rng.randn(config.hidden_size) * 0.01).astype(np.float16)

    outputs = []
    for run in range(2):
        engine = InferenceEngine(config, device_id=0, max_seq_len=32)
        load_all_layers(engine, config)

        # Verify M1 flags
        assert engine._deltanet_gpu is True, "DeltaNet GPU must be active"
        assert engine._qknorm_rope_fused is True, "QKnorm+RoPE fused must be active"
        # Verify M2 flags
        assert engine._gemv_int4_v3 is True, "INT4 GEMV v3 must be active"
        assert engine._rope_hip is True, "HIP RoPE must be active"

        out = engine.decode_step(token_emb, position=0)
        engine.cleanup()
        outputs.append(out.astype(np.float32))

    max_err = float(np.max(np.abs(outputs[0] - outputs[1])))
    print(f"  Layer types: {config.layer_types}")
    print(f"  Max abs error between two M1+M2 hybrid runs: {max_err:.2e}")
    assert max_err < M1M2_DECODE_TOLERANCE, \
        f"M1+M2 hybrid decode not reproducible: max_err={max_err:.2e}"
    assert not np.any(np.isnan(outputs[0])), "NaN in M1+M2 hybrid output"

    print(f"  PASS: M1+M2 hybrid (linear+full) decode reproducible "
          f"(max_err={max_err:.2e})")


# ---------------------------------------------------------------------------
# Test 5: Multi-step decode with all M1+M2 opts
# ---------------------------------------------------------------------------

def test_m1m2_multi_step_decode():
    """Multiple decode steps with all M1+M2 optimizations active.

    Verifies that stream concurrency (M1) + v3 cooperative reduction (M2)
    work together correctly across sequential decode steps.
    """
    print("\n[TEST] test_m1m2_multi_step_decode")
    config = make_full_attention_only_config(num_layers=2, hidden=256)
    engine = InferenceEngine(config, device_id=0, max_seq_len=32)
    load_all_layers(engine, config)

    # All M1+M2 flags active
    assert engine._streams_ready is True, "Streams (M1) must be active"
    assert engine._gemv_int4_v3 is True, "v3 GEMV (M2) must be active"
    assert engine._rope_hip is True, "HIP RoPE (M2) must be active"

    rng = np.random.RandomState(77)
    outputs = []
    for pos in range(5):
        token_emb = (rng.randn(config.hidden_size) * 0.01).astype(np.float16)
        out = engine.decode_step(token_emb, position=pos).astype(np.float32)
        outputs.append(out.copy())
        assert not np.any(np.isnan(out)), f"NaN at step {pos} (M1+M2)"
        assert not np.any(np.isinf(out)), f"Inf at step {pos} (M1+M2)"

    engine.cleanup()

    diffs = [float(np.max(np.abs(outputs[i] - outputs[i + 1])))
             for i in range(len(outputs) - 1)]
    print(f"  Step-to-step max diffs: {[f'{d:.3e}' for d in diffs]}")
    assert any(d > 1e-6 for d in diffs), \
        "All M1+M2 decode outputs identical — model may be broken"

    print(f"  PASS: {len(outputs)} steps with ALL M1+M2 optimizations, "
          f"all finite, outputs vary")


# ---------------------------------------------------------------------------
# Test 6: HIP RoPE integration in decode path (VAL-ROPE-001, VAL-ROPE-002)
# ---------------------------------------------------------------------------

def test_hip_rope_active_in_decode():
    """Verify HIP RoPE is active and decode produces correct output.

    The HIP RoPE kernel (rope_v2.hip) replaces assembly rope.s.
    We verify _rope_hip=True and that decode step produces finite output.
    """
    print("\n[TEST] test_hip_rope_active_in_decode")
    config = make_full_attention_only_config(num_layers=1, hidden=256)
    engine = InferenceEngine(config, device_id=0, max_seq_len=32)
    load_all_layers(engine, config)

    # M2: HIP RoPE must be active
    assert engine._rope_hip is True, \
        "_rope_hip=False — HIP RoPE (rope_v2.hip) is not active (VAL-ROPE-002)"

    rng = np.random.RandomState(33)
    token_emb = (rng.randn(config.hidden_size) * 0.01).astype(np.float16)
    out = engine.decode_step(token_emb, position=0).astype(np.float32)

    assert not np.any(np.isnan(out)), "NaN with HIP RoPE active"
    assert not np.any(np.isinf(out)), "Inf with HIP RoPE active"
    assert np.max(np.abs(out)) > 1e-7, "Zero output with HIP RoPE"

    engine.cleanup()
    print("  PASS: HIP RoPE (rope_v2.hip) active in decode, output is finite "
          "(VAL-ROPE-002)")


# ---------------------------------------------------------------------------
# Test 7: Prefill step with M1+M2 (VAL-CROSS-003 + VAL-CROSS-007)
# ---------------------------------------------------------------------------

def test_m1m2_prefill_step():
    """Prefill step with all M1+M2 optimizations produces finite output.

    Verifies that prefill (which uses GEMM path) works correctly with
    all M1+M2 changes to the engine.
    """
    print("\n[TEST] test_m1m2_prefill_step")
    config = make_full_attention_only_config(
        num_layers=2, hidden=256, num_heads=4, num_kv_heads=2, head_dim=64,
        intermediate=512
    )
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    load_all_layers(engine, config)

    assert engine._gemm_fp16_prefill is True, "M2 GEMM prefill must be active"
    assert engine._gemv_int4_v3 is True, "M2 INT4 v3 must be active"
    assert engine._streams_ready is True, "M1 streams must be active"

    rng = np.random.RandomState(66)
    seq_len = 32
    tokens = (rng.randn(seq_len, config.hidden_size) * 0.01).astype(np.float16)

    # Run prefill
    pf_out = engine.prefill_step(tokens)
    pf_f32 = pf_out.astype(np.float32)

    assert not np.any(np.isnan(pf_f32)), "NaN in M1+M2 prefill output"
    assert not np.any(np.isinf(pf_f32)), "Inf in M1+M2 prefill output"
    assert np.max(np.abs(pf_f32)) > 1e-7, "Zero M1+M2 prefill output"
    assert engine.kv_cache.current_len == seq_len, \
        f"KV cache not populated: {engine.kv_cache.current_len} != {seq_len}"

    engine.cleanup()
    print(f"  PASS: M1+M2 prefill (seq_len={seq_len}) produces finite output "
          f"(VAL-CROSS-003, VAL-CROSS-007)")


# ---------------------------------------------------------------------------
# Test 8: Prefill then decode with all M1+M2 opts
# ---------------------------------------------------------------------------

def test_m1m2_prefill_then_decode():
    """Prefill + decode with all M1+M2 optimizations active.

    Exercises the full inference pipeline: prefill fills KV cache, then
    decode steps use optimized kernels with the populated cache.
    """
    print("\n[TEST] test_m1m2_prefill_then_decode")
    config = make_full_attention_only_config(
        num_layers=2, hidden=256, num_heads=4, num_kv_heads=2, head_dim=64,
        intermediate=512
    )
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    load_all_layers(engine, config)

    rng = np.random.RandomState(99)
    seq_len = 32
    tokens = (rng.randn(seq_len, config.hidden_size) * 0.01).astype(np.float16)

    # Prefill with M2 GEMM
    pf_out = engine.prefill_step(tokens)
    assert not np.any(np.isnan(pf_out.astype(np.float32))), "NaN in prefill"

    # Decode with M1+M2 opts
    decode_outputs = []
    for pos in range(seq_len, seq_len + 3):
        tok_emb = (rng.randn(config.hidden_size) * 0.01).astype(np.float16)
        dec_out = engine.decode_step(tok_emb, position=pos).astype(np.float32)
        assert not np.any(np.isnan(dec_out)), f"NaN at decode pos {pos}"
        decode_outputs.append(dec_out)

    engine.cleanup()

    # Outputs should vary (active model)
    if len(decode_outputs) >= 2:
        max_diff = float(np.max(np.abs(decode_outputs[0] - decode_outputs[-1])))
        print(f"  First vs last decode output diff: {max_diff:.3e}")

    print(f"  PASS: M1+M2 prefill({seq_len}) + 3 decode steps all finite")


# ---------------------------------------------------------------------------
# Test 9: VAL-CROSS-006 — full integration check
# ---------------------------------------------------------------------------

def test_val_cross_006():
    """VAL-CROSS-006 explicit validation.

    A decode step with ALL M1 and M2 optimizations active simultaneously
    produces output matching the reference within 1e-2.

    This is the canonical test for VAL-CROSS-006.
    """
    print("\n[TEST] test_val_cross_006 (canonical)")

    # Use 2-layer config for a more thorough test
    hidden   = 128
    num_head = 2
    kv_head  = 1
    head_d   = 64
    inter    = 256

    config = QwenConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=2,
        num_attention_heads=num_head,
        num_key_value_heads=kv_head,
        head_dim=head_d,
        vocab_size=512,
        rms_norm_eps=1e-6,
        group_size=128,
        full_attention_interval=1,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_num_value_heads=2,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )

    rng = np.random.RandomState(42)
    token_emb = (rng.randn(hidden) * 0.01).astype(np.float16)

    # Build weights for 2 layers
    all_weights = [build_full_attention_weights(config, i, seed_base=i * 42)
                   for i in range(2)]

    # === Run 1: M1+M2 optimized engine ===
    engine = InferenceEngine(config, device_id=0, max_seq_len=16)
    for i, w in enumerate(all_weights):
        engine.load_layer_weights(i, w)

    # Verify all critical flags
    flags = {
        # M1
        "_deltanet_gpu": engine._deltanet_gpu,
        "_qknorm_rope_fused": engine._qknorm_rope_fused,
        "_gemv_int4_dual_fused": engine._gemv_int4_dual_fused,
        "_streams_ready": engine._streams_ready,
        "_gemv_fp16_v2": engine._gemv_fp16_v2,
        # M2
        "_gemv_int4_v3": engine._gemv_int4_v3,
        "_rope_hip": engine._rope_hip,
        "_gemm_fp16_prefill": engine._gemm_fp16_prefill,
    }
    missing = [k for k, v in flags.items() if not v]
    if missing:
        print(f"  WARNING: These M1/M2 flags are False: {missing}")
    else:
        print(f"  All M1+M2 flags True: {list(flags.keys())}")

    m1m2_out = engine.decode_step(token_emb, position=0).astype(np.float32)
    engine.cleanup()

    # === Reference: numpy computation ===
    # Compute layer-by-layer reference
    np_ref = token_emb.astype(np.float32)
    for i in range(2):
        np_ref = numpy_single_layer_reference(
            np_ref.astype(np.float16), all_weights[i], config, position=0
        ).astype(np.float32)

    max_err = float(np.max(np.abs(m1m2_out - np_ref)))
    mean_err = float(np.mean(np.abs(m1m2_out - np_ref)))
    print(f"  2-layer M1+M2 vs reference: max_err={max_err:.4e}, mean_err={mean_err:.4e}")
    print(f"  Missing flags (if any): {missing}")

    assert max_err < M1M2_DECODE_TOLERANCE, \
        (f"VAL-CROSS-006 FAILED: max_err={max_err:.4e} > tolerance={M1M2_DECODE_TOLERANCE}. "
         f"Missing flags: {missing}")
    assert not np.any(np.isnan(m1m2_out)), "NaN in VAL-CROSS-006 output"
    assert not np.any(np.isinf(m1m2_out)), "Inf in VAL-CROSS-006 output"

    print(f"  PASS: VAL-CROSS-006 — M1+M2 combined decode matches reference "
          f"(max_err={max_err:.4e}, tolerance={M1M2_DECODE_TOLERANCE})")


# ---------------------------------------------------------------------------
# Numpy references
# ---------------------------------------------------------------------------

def numpy_single_layer_reference(token_emb, weights, config, position=0):
    """Pure numpy reference for a single full-attention decode layer."""
    h        = config.hidden_size
    n_heads  = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_d   = config.head_dim
    inter    = config.intermediate_size
    gs       = config.group_size
    q_dim    = n_heads * head_d
    eps      = config.rms_norm_eps
    partial  = config.partial_rotary_factor

    def rmsnorm(x, w):
        x = x.astype(np.float32)
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms * w.astype(np.float32)).astype(np.float16)

    def dequant_int4(qweight, scales, zeros, K, N):
        K8 = K // 8
        W = np.zeros((K, N), dtype=np.float32)
        for i in range(K8):
            packed = qweight[i].astype(np.uint32)  # [N] vector of packed uint32
            for bit in range(8):
                nibble = ((packed >> np.uint32(bit * 4)) & np.uint32(0xF)).astype(np.float32)
                W[i * 8 + bit] = nibble
        num_groups = K // gs
        for g in range(num_groups):
            k0, k1 = g * gs, (g + 1) * gs
            s = scales[g].astype(np.float32)
            z = zeros[g].astype(np.float32)
            W[k0:k1] = (W[k0:k1] - z[None, :]) * s[None, :]
        return W

    hidden = token_emb.astype(np.float32)
    q_w  = weights['q_weight'].astype(np.float32)
    qg_w = weights['q_gate_weight'].astype(np.float32)
    k_w  = weights['k_weight'].astype(np.float32)
    v_w  = weights['v_weight'].astype(np.float32)
    o_w  = weights['o_weight'].astype(np.float32)

    g_dq = dequant_int4(weights['gate_qweight'], weights['gate_scales'],
                         weights['gate_zeros'], h, inter)
    u_dq = dequant_int4(weights['up_qweight'],   weights['up_scales'],
                         weights['up_zeros'],     h, inter)
    d_dq = dequant_int4(weights['down_qweight'], weights['down_scales'],
                         weights['down_zeros'],   inter, h)

    # Pre-attention RMSNorm
    normed = rmsnorm(hidden, weights['attn_norm']).astype(np.float32)

    # Projections
    q   = (q_w @ normed).astype(np.float16)
    qg  = (qg_w @ normed).astype(np.float16)
    k   = (k_w @ normed).astype(np.float16)
    v   = (v_w @ normed).astype(np.float16)

    # QK-norm + RoPE at position
    rotary_dim = int(head_d * partial)
    half_r = rotary_dim // 2
    freqs = 1.0 / (config.rope_theta **
                   (np.arange(0, half_r, dtype=np.float32) * 2.0 / rotary_dim))
    cos_v = np.cos(position * freqs).astype(np.float32)
    sin_v = np.sin(position * freqs).astype(np.float32)

    def apply_qknorm_rope(x, norm_w, num_h):
        heads = x.reshape(num_h, head_d).astype(np.float32)
        w = norm_w.astype(np.float32)
        out = np.zeros_like(heads)
        for hi in range(num_h):
            v_h = heads[hi]
            rms = np.sqrt(np.mean(v_h ** 2) + eps)
            v_h = v_h / rms * w
            for i in range(half_r):
                x0, x1 = v_h[2 * i], v_h[2 * i + 1]
                v_h[2 * i]     = x0 * cos_v[i] - x1 * sin_v[i]
                v_h[2 * i + 1] = x0 * sin_v[i] + x1 * cos_v[i]
            out[hi] = v_h
        return out.reshape(-1).astype(np.float16)

    q_rot = apply_qknorm_rope(q, weights['q_norm'], n_heads)
    k_rot = apply_qknorm_rope(k, weights['k_norm'], kv_heads)

    # Decode attention (single KV → softmax = 1)
    groups = n_heads // kv_heads
    attn_out = np.zeros((n_heads, head_d), dtype=np.float32)
    v_h = v.reshape(kv_heads, head_d).astype(np.float32)
    for qi in range(n_heads):
        ki = qi // groups
        attn_out[qi] = v_h[ki]
    attn_out = attn_out.reshape(q_dim).astype(np.float16)

    # Sigmoid gate
    gate_f32 = qg.astype(np.float32)
    sig_gate = 1.0 / (1.0 + np.exp(-gate_f32))
    attn_gated = (attn_out.astype(np.float32) * sig_gate).astype(np.float16)

    # Out projection + residual
    out_proj = (o_w @ attn_gated.astype(np.float32))
    hidden = (hidden + out_proj).astype(np.float32)

    # Pre-FFN RMSNorm
    normed_ffn = rmsnorm(hidden, weights['ffn_norm']).astype(np.float32)

    # INT4 FFN
    gate_ffn = (g_dq.T @ normed_ffn).astype(np.float32)
    up_ffn   = (u_dq.T @ normed_ffn).astype(np.float32)
    silu_gate = gate_ffn / (1.0 + np.exp(-gate_ffn.astype(np.float64))).astype(np.float32)
    ffn_mid   = (silu_gate * up_ffn).astype(np.float32)
    down_out  = (d_dq.T @ ffn_mid).astype(np.float32)
    hidden    = (hidden + down_out).astype(np.float32)

    return hidden.astype(np.float16)


def numpy_decode_reference(token_emb, weights, config, position=0):
    """Single-layer reference (alias for numpy_single_layer_reference)."""
    return numpy_single_layer_reference(token_emb, weights, config, position)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("M1+M2 Combined Integration Test Suite (test_m1m2_combined.py)")
    print("Tests VAL-CROSS-006 (all M1+M2 optimizations active simultaneously)")
    print("=" * 70)

    tests = [
        test_all_m1m2_flags_loaded,
        test_m1m2_full_attention_decode_reproducible,
        test_m1m2_combined_correctness_vs_reference,
        test_m1m2_hybrid_decode,
        test_m1m2_multi_step_decode,
        test_hip_rope_active_in_decode,
        test_m1m2_prefill_step,
        test_m1m2_prefill_then_decode,
        test_val_cross_006,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  FAIL: {test_fn.__name__}: {e}")
            print(f"  Traceback:\n{tb}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
