#!/usr/bin/env python3
"""
Kernel isolation test for fused QKV GEMV.

Tests numerical correctness of gemv_int4_qkv_fused kernel by comparing
against separate Q, K, V GEMV launches.

Expected: cosine_sim >= 0.999 for each component (Q, K, V)
"""
import sys
import numpy as np
sys.path.insert(0, '/opt/mi50grad')

from src.runtime.hip_dispatch import GPUDevice
from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine

def test_fused_qkv_correctness():
    """Test fused QKV GEMV vs separate Q/K/V GEMVs."""
    print("=" * 70)
    print("Fused QKV GEMV Kernel Isolation Test")
    print("=" * 70)
    
    # Load config
    MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(MODEL_DIR)
    
    # Create engine for TP=4
    tp_size = 4
    hidden_size = config.hidden_size
    local_q_dim = (config.num_attention_heads // tp_size) * config.head_dim
    local_kv_dim = (config.num_key_value_heads // tp_size) * config.head_dim
    N_total = local_q_dim + 2 * local_kv_dim
    group_size = config.group_size
    
    print(f"\nConfiguration:")
    print(f"  TP size: {tp_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Local Q dim: {local_q_dim}")
    print(f"  Local KV dim: {local_kv_dim}")
    print(f"  Total output dim (N_total): {N_total}")
    print(f"  Group size: {group_size}")
    
    # Create engine
    engine = InferenceEngine(
        config, 
        device_id=0, 
        max_seq_len=64,
        tp_size=tp_size,
        tp_rank=0,
        use_int4_attention=True
    )
    
    # Create synthetic INT4 weights and scales
    rng = np.random.default_rng(42)
    
    # Q weights: [q_dim/8, hidden] INT32 packed
    q_qweight = rng.integers(0, 0xFFFFFFFF, size=(local_q_dim // 8, hidden_size), dtype=np.uint32)
    q_scales = rng.standard_normal((local_q_dim // group_size, hidden_size)).astype(np.float16)
    q_zeros = np.zeros((local_q_dim // group_size, hidden_size), dtype=np.float16)
    
    # K weights: [kv_dim/8, hidden]
    k_qweight = rng.integers(0, 0xFFFFFFFF, size=(local_kv_dim // 8, hidden_size), dtype=np.uint32)
    k_scales = rng.standard_normal((local_kv_dim // group_size, hidden_size)).astype(np.float16)
    k_zeros = np.zeros((local_kv_dim // group_size, hidden_size), dtype=np.float16)
    
    # V weights: [kv_dim/8, hidden]
    v_qweight = rng.integers(0, 0xFFFFFFFF, size=(local_kv_dim // 8, hidden_size), dtype=np.uint32)
    v_scales = rng.standard_normal((local_kv_dim // group_size, hidden_size)).astype(np.float16)
    v_zeros = np.zeros((local_kv_dim // group_size, hidden_size), dtype=np.float16)
    
    # Concatenate for fused kernel
    qkv_qweight = np.concatenate([q_qweight, k_qweight, v_qweight], axis=0)
    qkv_scales = np.concatenate([q_scales, k_scales, v_scales], axis=0)
    qkv_zeros = np.concatenate([q_zeros, k_zeros, v_zeros], axis=0)
    
    print(f"\nWeight shapes:")
    print(f"  Q: {q_qweight.shape}, Scales: {q_scales.shape}")
    print(f"  K: {k_qweight.shape}, Scales: {k_scales.shape}")
    print(f"  V: {v_qweight.shape}, Scales: {v_scales.shape}")
    print(f"  QKV fused: {qkv_qweight.shape}, Scales: {qkv_scales.shape}")
    
    # Create input activation (normalized hidden state)
    A = rng.standard_normal(hidden_size).astype(np.float16)
    
    # Upload weights and input to GPU
    device = engine.device
    d_A = device.upload(A.tobytes())
    d_q_qweight = device.upload(q_qweight.tobytes())
    d_q_scales = device.upload(q_scales.tobytes())
    d_q_zeros = device.upload(q_zeros.tobytes())
    d_k_qweight = device.upload(k_qweight.tobytes())
    d_k_scales = device.upload(k_scales.tobytes())
    d_k_zeros = device.upload(k_zeros.tobytes())
    d_v_qweight = device.upload(v_qweight.tobytes())
    d_v_scales = device.upload(v_scales.tobytes())
    d_v_zeros = device.upload(v_zeros.tobytes())
    d_qkv_qweight = device.upload(qkv_qweight.tobytes())
    d_qkv_scales = device.upload(qkv_scales.tobytes())
    d_qkv_zeros = device.upload(qkv_zeros.tobytes())
    
    # Allocate output buffers
    d_Q_sep = device.malloc(local_q_dim * 2)
    d_K_sep = device.malloc(local_kv_dim * 2)
    d_V_sep = device.malloc(local_kv_dim * 2)
    d_Q_fused = device.malloc(local_q_dim * 2)
    d_K_fused = device.malloc(local_kv_dim * 2)
    d_V_fused = device.malloc(local_kv_dim * 2)
    
    # Check if fused kernel is available
    if not engine._gemv_int4_qkv_fused:
        print("\nWARNING: Fused QKV kernel not available!")
        print("Test cannot proceed without gemv_int4_qkv_fused.hip")
        return False
    
    # Get kernel functions
    # Use v8-based separate kernels for comparison
    if engine._gemv_int4_v8:
        gemv_q_func = engine.kernels.get_hip("gemv_int4_v8_t16", "gemv_int4_v8")
        gemv_k_func = engine.kernels.get_hip("gemv_int4_v8_t16", "gemv_int4_v8")
        gemv_v_func = engine.kernels.get_hip("gemv_int4_v8_t16", "gemv_int4_v8")
    else:
        print("WARNING: v8 kernel not available, using v7")
        gemv_q_func = engine.kernels.get_hip("gemv_int4_v7_t16", "gemv_int4_v7")
        gemv_k_func = engine.kernels.get_hip("gemv_int4_v7_t16", "gemv_int4_v7")
        gemv_v_func = engine.kernels.get_hip("gemv_int4_v7_t16", "gemv_int4_v7")
    
    gemv_qkv_func = engine.kernels.get_hip("gemv_int4_qkv_fused_t16", "gemv_int4_qkv_fused", hsaco_suffix="_t16")
    
    # Launch separate Q, K, V GEMVs (AWQ mode: 8 params without zeros)
    q_grid = (2 * local_q_dim + 3) // 4
    k_grid = (2 * local_kv_dim + 3) // 4
    v_grid = (2 * local_kv_dim + 3) // 4
    
    print(f"\nLaunching separate GEMV kernels...")
    device.launch_kernel(
        gemv_q_func,
        (q_grid, 1, 1), (256, 1, 1),
        [d_A, d_q_qweight, d_q_scales, d_q_zeros, d_Q_sep,
         np.uint32(hidden_size), np.uint32(local_q_dim), np.uint32(group_size)]
    )
    device.launch_kernel(
        gemv_k_func,
        (k_grid, 1, 1), (256, 1, 1),
        [d_A, d_k_qweight, d_k_scales, d_k_zeros, d_K_sep,
         np.uint32(hidden_size), np.uint32(local_kv_dim), np.uint32(group_size)]
    )
    device.launch_kernel(
        gemv_v_func,
        (v_grid, 1, 1), (256, 1, 1),
        [d_A, d_v_qweight, d_v_scales, d_v_zeros, d_V_sep,
         np.uint32(hidden_size), np.uint32(local_kv_dim), np.uint32(group_size)]
    )
    device.synchronize()
    
    # Launch fused QKV GEMV
    qkv_grid = (2 * N_total + 3) // 4
    print(f"Launching fused QKV GEMV kernel...")
    device.launch_kernel(
        gemv_qkv_func,
        (qkv_grid, 1, 1), (256, 1, 1),
        [d_A, d_qkv_qweight, d_qkv_scales, d_qkv_zeros,
         d_Q_fused, d_K_fused, d_V_fused,
         np.uint32(hidden_size), np.uint32(local_q_dim), np.uint32(local_kv_dim),
         np.uint32(group_size), np.uint64(0)]  # v_cache_dst = null
    )
    device.synchronize()
    
    # Download results
    Q_sep = np.frombuffer(device.download(d_Q_sep, local_q_dim * 2), dtype=np.float16)
    K_sep = np.frombuffer(device.download(d_K_sep, local_kv_dim * 2), dtype=np.float16)
    V_sep = np.frombuffer(device.download(d_V_sep, local_kv_dim * 2), dtype=np.float16)
    Q_fused = np.frombuffer(device.download(d_Q_fused, local_q_dim * 2), dtype=np.float16)
    K_fused = np.frombuffer(device.download(d_K_fused, local_kv_dim * 2), dtype=np.float16)
    V_fused = np.frombuffer(device.download(d_V_fused, local_kv_dim * 2), dtype=np.float16)
    
    # Compute cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    q_sim = cosine_sim(Q_sep, Q_fused)
    k_sim = cosine_sim(K_sep, K_fused)
    v_sim = cosine_sim(V_sep, V_fused)
    
    # Compute max absolute error
    q_err = np.max(np.abs(Q_sep - Q_fused))
    k_err = np.max(np.abs(K_sep - K_fused))
    v_err = np.max(np.abs(V_sep - V_fused))
    
    print(f"\nResults:")
    print(f"  Q cosine_sim: {q_sim:.6f}, max_abs_error: {q_err:.6e}")
    print(f"  K cosine_sim: {k_sim:.6f}, max_abs_error: {k_err:.6e}")
    print(f"  V cosine_sim: {v_sim:.6f}, max_abs_error: {v_err:.6e}")
    
    # Cleanup
    device.free(d_A)
    device.free(d_q_qweight)
    device.free(d_q_scales)
    device.free(d_q_zeros)
    device.free(d_k_qweight)
    device.free(d_k_scales)
    device.free(d_k_zeros)
    device.free(d_v_qweight)
    device.free(d_v_scales)
    device.free(d_v_zeros)
    device.free(d_qkv_qweight)
    device.free(d_qkv_scales)
    device.free(d_qkv_zeros)
    device.free(d_Q_sep)
    device.free(d_K_sep)
    device.free(d_V_sep)
    device.free(d_Q_fused)
    device.free(d_K_fused)
    device.free(d_V_fused)
    
    # Check thresholds
    PASSED = True
    if q_sim < 0.999:
        print(f"\nFAIL: Q cosine_sim {q_sim:.6f} < 0.999")
        PASSED = False
    if k_sim < 0.999:
        print(f"\nFAIL: K cosine_sim {k_sim:.6f} < 0.999")
        PASSED = False
    if v_sim < 0.999:
        print(f"\nFAIL: V cosine_sim {v_sim:.6f} < 0.999")
        PASSED = False
    
    if PASSED:
        print("\n" + "=" * 70)
        print("TEST PASSED: Fused QKV GEMV matches separate Q/K/V GEMVs")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("TEST FAILED")
        print("=" * 70)
    
    return PASSED

if __name__ == "__main__":
    success = test_fused_qkv_correctness()
    sys.exit(0 if success else 1)
