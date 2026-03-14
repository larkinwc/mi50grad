#!/usr/bin/env python3
"""Integration test for InferenceEngine: validates a single decode step
with random weights to ensure all kernels connect correctly.

Tests both full_attention and linear_attention layer types.
"""

import ctypes
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig
from src.inference.engine import InferenceEngine


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


def test_single_decode_step():
    """Test one decode step through a tiny 2-layer hybrid model."""
    print("Testing single decode step with hybrid attention...")

    # Small config: layer 0 = linear_attention, layer 1 = full_attention
    config = QwenConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        group_size=128,
        full_attention_interval=2,  # every 2nd layer is full attention
        # Linear attention params (scaled down)
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )

    # layer_types: [linear_attention, full_attention]
    assert config.is_linear_attention(0)
    assert config.is_full_attention(1)

    engine = InferenceEngine(config, device_id=0, max_seq_len=32)

    h = config.hidden_size
    inter = config.intermediate_size

    try:
        for layer_idx in range(config.num_hidden_layers):
            weights = {}

            if config.is_full_attention(layer_idx):
                weights['layer_type'] = 'full_attention'
                q_dim = config.num_attention_heads * config.head_dim  # 256
                kv_dim = config.num_key_value_heads * config.head_dim  # 128

                # FP16 attention weights
                weights['q_weight'] = make_random_fp16((q_dim, h), seed=layer_idx*10)
                weights['q_gate_weight'] = make_random_fp16((q_dim, h), seed=layer_idx*10+1)
                weights['k_weight'] = make_random_fp16((kv_dim, h), seed=layer_idx*10+2)
                weights['v_weight'] = make_random_fp16((kv_dim, h), seed=layer_idx*10+3)
                weights['o_weight'] = make_random_fp16((h, q_dim), seed=layer_idx*10+4)
                weights['q_norm'] = np.ones(config.head_dim, dtype=np.float16)
                weights['k_norm'] = np.ones(config.head_dim, dtype=np.float16)
            else:
                weights['layer_type'] = 'linear_attention'
                q_dim = config.linear_num_key_heads * config.linear_key_head_dim  # 64
                v_dim = config.linear_num_value_heads * config.linear_value_head_dim  # 128
                qkv_dim = q_dim + q_dim + v_dim  # 256

                weights['la_in_proj_qkv'] = make_random_fp16((qkv_dim, h), seed=layer_idx*10)
                weights['la_in_proj_a'] = make_random_fp16(
                    (config.linear_num_value_heads, h), seed=layer_idx*10+1)
                weights['la_in_proj_b'] = make_random_fp16(
                    (config.linear_num_value_heads, h), seed=layer_idx*10+2)
                weights['la_in_proj_z'] = make_random_fp16((v_dim, h), seed=layer_idx*10+3)
                weights['la_conv1d'] = make_random_fp16(
                    (qkv_dim, 1, config.linear_conv_kernel_dim), seed=layer_idx*10+4)
                weights['la_A_log'] = np.zeros(config.linear_num_value_heads,
                                                dtype=np.float32)
                weights['la_dt_bias'] = np.zeros(config.linear_num_value_heads,
                                                  dtype=np.float16)
                weights['la_norm'] = np.ones(config.linear_value_head_dim,
                                              dtype=np.float32)
                weights['la_out_proj'] = make_random_fp16((h, v_dim), seed=layer_idx*10+5)

            # INT4 FFN (same for all layer types)
            qw, sc, zr = make_random_int4_weights(h, inter, config.group_size,
                                                    seed=layer_idx*10+6)
            weights['gate_qweight'] = qw
            weights['gate_scales'] = sc
            weights['gate_zeros'] = zr

            qw, sc, zr = make_random_int4_weights(h, inter, config.group_size,
                                                    seed=layer_idx*10+7)
            weights['up_qweight'] = qw
            weights['up_scales'] = sc
            weights['up_zeros'] = zr

            qw, sc, zr = make_random_int4_weights(inter, h, config.group_size,
                                                    seed=layer_idx*10+8)
            weights['down_qweight'] = qw
            weights['down_scales'] = sc
            weights['down_zeros'] = zr

            weights['attn_norm'] = np.ones(h, dtype=np.float16)
            weights['ffn_norm'] = np.ones(h, dtype=np.float16)

            engine.load_layer_weights(layer_idx, weights)

        # Load final norm
        engine.load_final_norm(np.ones(h, dtype=np.float16))

        # Run decode step
        embedding = np.random.randn(h).astype(np.float16) * 0.1
        hidden = engine.decode_step(embedding, position=0)

        assert hidden.shape == (h,), f"Expected shape ({h},), got {hidden.shape}"
        assert hidden.dtype == np.float16, f"Expected FP16, got {hidden.dtype}"
        assert not np.any(np.isnan(hidden)), "Output contains NaN!"
        assert not np.any(np.isinf(hidden)), "Output contains Inf!"

        print(f"  Output shape: {hidden.shape}, dtype: {hidden.dtype}")
        print(f"  Output range: [{hidden.min():.4f}, {hidden.max():.4f}]")
        print(f"  KV cache len: {engine.kv_cache.current_len}")
        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.cleanup()


def test_multi_step_decode():
    """Test multiple sequential decode steps."""
    print("\nTesting multi-step decode (3 tokens)...")

    config = QwenConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        group_size=128,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
    )

    engine = InferenceEngine(config, device_id=0, max_seq_len=32)
    h = config.hidden_size
    inter = config.intermediate_size

    try:
        for layer_idx in range(config.num_hidden_layers):
            weights = {}

            if config.is_full_attention(layer_idx):
                weights['layer_type'] = 'full_attention'
                q_dim = config.num_attention_heads * config.head_dim
                kv_dim = config.num_key_value_heads * config.head_dim
                weights['q_weight'] = make_random_fp16((q_dim, h), seed=layer_idx)
                weights['q_gate_weight'] = make_random_fp16((q_dim, h), seed=layer_idx+100)
                weights['k_weight'] = make_random_fp16((kv_dim, h), seed=layer_idx+200)
                weights['v_weight'] = make_random_fp16((kv_dim, h), seed=layer_idx+300)
                weights['o_weight'] = make_random_fp16((h, q_dim), seed=layer_idx+400)
                weights['q_norm'] = np.ones(config.head_dim, dtype=np.float16)
                weights['k_norm'] = np.ones(config.head_dim, dtype=np.float16)
            else:
                weights['layer_type'] = 'linear_attention'
                q_dim = config.linear_num_key_heads * config.linear_key_head_dim
                v_dim = config.linear_num_value_heads * config.linear_value_head_dim
                qkv_dim = q_dim + q_dim + v_dim
                weights['la_in_proj_qkv'] = make_random_fp16((qkv_dim, h), seed=layer_idx)
                weights['la_in_proj_a'] = make_random_fp16(
                    (config.linear_num_value_heads, h), seed=layer_idx+100)
                weights['la_in_proj_b'] = make_random_fp16(
                    (config.linear_num_value_heads, h), seed=layer_idx+200)
                weights['la_in_proj_z'] = make_random_fp16((v_dim, h), seed=layer_idx+300)
                weights['la_conv1d'] = make_random_fp16(
                    (qkv_dim, 1, config.linear_conv_kernel_dim), seed=layer_idx+400)
                weights['la_A_log'] = np.zeros(config.linear_num_value_heads, dtype=np.float32)
                weights['la_dt_bias'] = np.zeros(config.linear_num_value_heads, dtype=np.float16)
                weights['la_norm'] = np.ones(config.linear_value_head_dim, dtype=np.float32)
                weights['la_out_proj'] = make_random_fp16((h, v_dim), seed=layer_idx+500)

            qw, sc, zr = make_random_int4_weights(h, inter, config.group_size, seed=layer_idx+600)
            weights['gate_qweight'] = qw
            weights['gate_scales'] = sc
            weights['gate_zeros'] = zr
            qw, sc, zr = make_random_int4_weights(h, inter, config.group_size, seed=layer_idx+700)
            weights['up_qweight'] = qw
            weights['up_scales'] = sc
            weights['up_zeros'] = zr
            qw, sc, zr = make_random_int4_weights(inter, h, config.group_size, seed=layer_idx+800)
            weights['down_qweight'] = qw
            weights['down_scales'] = sc
            weights['down_zeros'] = zr

            weights['attn_norm'] = np.ones(h, dtype=np.float16)
            weights['ffn_norm'] = np.ones(h, dtype=np.float16)
            engine.load_layer_weights(layer_idx, weights)

        engine.load_final_norm(np.ones(h, dtype=np.float16))

        # Run 3 decode steps
        for step in range(3):
            embedding = np.random.randn(h).astype(np.float16) * 0.1
            hidden = engine.decode_step(embedding, position=step)
            assert not np.any(np.isnan(hidden)), f"Step {step}: NaN in output"
            assert not np.any(np.isinf(hidden)), f"Step {step}: Inf in output"
            print(f"  Step {step}: range=[{hidden.min():.4f}, {hidden.max():.4f}], "
                  f"kv_len={engine.kv_cache.current_len}")

        assert engine.kv_cache.current_len == 3, \
            f"Expected kv_cache.current_len=3, got {engine.kv_cache.current_len}"

        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.cleanup()


if __name__ == "__main__":
    passed = 0
    failed = 0

    for test in [test_single_decode_step, test_multi_step_decode]:
        if test():
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
