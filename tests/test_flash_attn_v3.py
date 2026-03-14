#!/usr/bin/env python3
"""
test_flash_attn_v3.py - Test and benchmark flash_attn_256_v3.hip

Tests:
  1. Correctness of v3 prefill at seq_len=16,64,128,512, heads=4, kv_heads=2, causal=1
  2. Correctness of decode path (unchanged from tuned kernel, regression check)
  3. Benchmark v3 prefill vs current tuned prefill at seq_len=128,512 with heads=48, kv_heads=8
  4. Reports max abs error and latency comparison

Kernel v3 design summary:
  - BLOCK_M=16, BLOCK_N=16 (16 Q rows per WG, 4 per wavefront)
  - v_dot2_f32_f16 (fdot2) for score computation: 2x arithmetic throughput
  - 16 KB LDS (K+V blocks), FP32 accumulators throughout
  - GQA and causal masking supported
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco


# ============================================================================
# Reference implementation
# ============================================================================

def numpy_attention(Q, K, V, causal=False):
    """CPU reference: softmax(Q @ K^T / sqrt(d)) @ V
    Q: [num_q_rows, num_heads, head_dim]
    K: [kv_len, num_kv_heads, head_dim]
    V: [kv_len, num_kv_heads, head_dim]
    Returns: [num_q_rows, num_heads, head_dim] FP16
    """
    num_q_rows, num_heads, head_dim = Q.shape
    kv_len = K.shape[0]
    num_kv_heads = K.shape[1]
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros((num_q_rows, num_heads, head_dim), dtype=np.float32)

    for h in range(num_heads):
        kv_h = int(h * num_kv_heads // num_heads)
        q = Q[:, h, :].astype(np.float32) * scale     # [num_q_rows, d]
        k = K[:, kv_h, :].astype(np.float32)           # [kv_len, d]
        v = V[:, kv_h, :].astype(np.float32)           # [kv_len, d]

        scores = q @ k.T                                # [num_q_rows, kv_len]
        if causal:
            for i in range(num_q_rows):
                scores[i, i+1:] = -1e9

        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
        out[:, h, :] = weights @ v

    return out.astype(np.float16)


def numpy_decode_attention(Q, K, V):
    """Reference for decode (q_rows=1, no causal needed)."""
    return numpy_attention(Q, K, V, causal=False)


# ============================================================================
# Kernel launchers
# ============================================================================

BLOCK_M = 16   # must match flash_attn_256_v3.hip BLOCK_M constant

def load_v3_kernel(dev):
    """Build and load flash_attn_256_v3.hip."""
    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_path = PROJECT_ROOT / "src" / "kernels" / "flash_attn_256_v3.hip"
    hsaco_path = BUILD_DIR / "flash_attn_256_v3.hsaco"
    print("Building flash_attn_256_v3.hip ...", flush=True)
    build_hip_hsaco(str(hip_path), str(hsaco_path))
    print("  Done.", flush=True)
    module = dev.load_hsaco(str(hsaco_path))
    return module


def load_tuned_kernel(dev):
    """Build and load flash_attn_256_tuned.hip (baseline for benchmark)."""
    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_path = PROJECT_ROOT / "src" / "kernels" / "flash_attn_256_tuned.hip"
    hsaco_path = BUILD_DIR / "flash_attn_256_tuned.hsaco"
    print("Building flash_attn_256_tuned.hip ...", flush=True)
    build_hip_hsaco(str(hip_path), str(hsaco_path))
    print("  Done.", flush=True)
    module = dev.load_hsaco(str(hsaco_path))
    return module


def alloc_kv(dev, Q, K, V):
    d_Q = dev.malloc(Q.nbytes)
    d_K = dev.malloc(K.nbytes)
    d_V = dev.malloc(V.nbytes)
    dev.upload(d_Q, Q.tobytes())
    dev.upload(d_K, K.tobytes())
    dev.upload(d_V, V.tobytes())
    return d_Q, d_K, d_V


def free_kv(dev, d_Q, d_K, d_V):
    dev.free(d_Q)
    dev.free(d_K)
    dev.free(d_V)


def run_v3_prefill(dev, func, Q, K, V, num_q_rows, kv_seq_len, num_heads, num_kv_heads, causal=1):
    """Launch flash_attn_256_v3_prefill.
    Grid: (num_heads, ceil(num_q_rows / BLOCK_M), 1), Block: (256, 1, 1)
    """
    head_dim = 256
    d_Q, d_K, d_V = alloc_kv(dev, Q, K, V)
    out_bytes = num_q_rows * num_heads * head_dim * 2   # FP16
    d_Out = dev.malloc(out_bytes)
    dev.hip.memset(d_Out, 0, out_bytes)

    grid_y = (num_q_rows + BLOCK_M - 1) // BLOCK_M
    params = [
        ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
        ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
        ctypes.c_uint32(kv_seq_len),
        ctypes.c_uint32(num_q_rows),
        ctypes.c_uint32(num_heads),
        ctypes.c_uint32(num_kv_heads),
        ctypes.c_uint32(causal),
    ]
    dev.launch(func, (num_heads, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_Out, out_bytes), dtype=np.float16).copy()
    result = result.reshape(num_q_rows, num_heads, head_dim)

    dev.free(d_Out)
    free_kv(dev, d_Q, d_K, d_V)
    return result


def run_decode(dev, func, Q, K, V, kv_seq_len, num_heads, num_kv_heads):
    """Launch flash_attn_256_decode.
    Grid: (num_heads, 1, 1), Block: (256, 1, 1)
    """
    head_dim = 256
    d_Q, d_K, d_V = alloc_kv(dev, Q, K, V)
    out_bytes = 1 * num_heads * head_dim * 2
    d_Out = dev.malloc(out_bytes)
    dev.hip.memset(d_Out, 0, out_bytes)

    params = [
        ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
        ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
        ctypes.c_uint32(kv_seq_len),
        ctypes.c_uint32(num_heads),
        ctypes.c_uint32(num_kv_heads),
    ]
    dev.launch(func, (num_heads, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_Out, out_bytes), dtype=np.float16).copy()
    result = result.reshape(1, num_heads, head_dim)

    dev.free(d_Out)
    free_kv(dev, d_Q, d_K, d_V)
    return result


def run_tuned_prefill(dev, func, Q, K, V, num_q_rows, kv_seq_len, num_heads, num_kv_heads, causal=1):
    """Launch flash_attn_256_prefill (tuned baseline).
    Grid: (num_heads, ceil(num_q_rows/4), 1), Block: (256, 1, 1)
    """
    head_dim = 256
    d_Q, d_K, d_V = alloc_kv(dev, Q, K, V)
    out_bytes = num_q_rows * num_heads * head_dim * 2
    d_Out = dev.malloc(out_bytes)
    dev.hip.memset(d_Out, 0, out_bytes)

    grid_y = (num_q_rows + 3) // 4
    params = [
        ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
        ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
        ctypes.c_uint32(kv_seq_len),
        ctypes.c_uint32(num_q_rows),
        ctypes.c_uint32(num_heads),
        ctypes.c_uint32(num_kv_heads),
        ctypes.c_uint32(causal),
    ]
    dev.launch(func, (num_heads, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_Out, out_bytes), dtype=np.float16).copy()
    result = result.reshape(num_q_rows, num_heads, head_dim)

    dev.free(d_Out)
    free_kv(dev, d_Q, d_K, d_V)
    return result


# ============================================================================
# Test 1: V3 prefill correctness
# ============================================================================

def test_v3_prefill_correctness(dev, v3_prefill_func):
    """Test v3 prefill at seq_len=16,64,128,512, heads=4, kv_heads=2, causal=1."""
    print("=== Test 1: V3 Prefill Correctness (heads=4, kv_heads=2, causal=1) ===")
    all_pass = True
    np.random.seed(42)

    seq_lens = [16, 64, 128, 512]
    for seq_len in seq_lens:
        Q = np.random.randn(seq_len, 4, 256).astype(np.float16) * 0.1
        K = np.random.randn(seq_len, 2, 256).astype(np.float16) * 0.1
        V = np.random.randn(seq_len, 2, 256).astype(np.float16) * 0.1

        ref = numpy_attention(Q, K, V, causal=True)
        result = run_v3_prefill(dev, v3_prefill_func, Q, K, V,
                                 seq_len, seq_len, num_heads=4, num_kv_heads=2, causal=1)

        max_err = float(np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32))))
        ok = max_err < 1e-2
        status = "PASS" if ok else "FAIL"
        print(f"  seq_len={seq_len:4d}, heads=4, kv_heads=2, causal=1: "
              f"max_err={max_err:.6f} {status}")
        if not ok:
            all_pass = False

    # Also test non-causal
    for seq_len in [16, 128]:
        Q = np.random.randn(seq_len, 4, 256).astype(np.float16) * 0.1
        K = np.random.randn(seq_len, 2, 256).astype(np.float16) * 0.1
        V = np.random.randn(seq_len, 2, 256).astype(np.float16) * 0.1

        ref = numpy_attention(Q, K, V, causal=False)
        result = run_v3_prefill(dev, v3_prefill_func, Q, K, V,
                                 seq_len, seq_len, num_heads=4, num_kv_heads=2, causal=0)

        max_err = float(np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32))))
        ok = max_err < 1e-2
        status = "PASS" if ok else "FAIL"
        print(f"  seq_len={seq_len:4d}, heads=4, kv_heads=2, causal=0: "
              f"max_err={max_err:.6f} {status}")
        if not ok:
            all_pass = False

    # Test GQA with equal heads
    for seq_len in [16, 64]:
        Q = np.random.randn(seq_len, 4, 256).astype(np.float16) * 0.1
        K = np.random.randn(seq_len, 4, 256).astype(np.float16) * 0.1
        V = np.random.randn(seq_len, 4, 256).astype(np.float16) * 0.1

        ref = numpy_attention(Q, K, V, causal=True)
        result = run_v3_prefill(dev, v3_prefill_func, Q, K, V,
                                 seq_len, seq_len, num_heads=4, num_kv_heads=4, causal=1)

        max_err = float(np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32))))
        ok = max_err < 1e-2
        status = "PASS" if ok else "FAIL"
        print(f"  seq_len={seq_len:4d}, heads=4, kv_heads=4, causal=1: "
              f"max_err={max_err:.6f} {status}")
        if not ok:
            all_pass = False

    return all_pass


# ============================================================================
# Test 2: Decode regression (unchanged decode kernel)
# ============================================================================

def test_decode_regression(dev, decode_func):
    """Verify decode kernel (from v3 file) matches reference. Regression check."""
    print("\n=== Test 2: Decode Correctness Regression (unchanged kernel) ===")
    all_pass = True
    np.random.seed(7)

    configs = [
        (4,  2,  256),
        (4,  4,  256),
        (8,  2,  512),
        (48, 8,  256),
        (48, 8,  512),
        (48, 8, 1024),
    ]

    for (num_heads, num_kv_heads, kv_len) in configs:
        Q = np.random.randn(1, num_heads, 256).astype(np.float16) * 0.1
        K = np.random.randn(kv_len, num_kv_heads, 256).astype(np.float16) * 0.1
        V = np.random.randn(kv_len, num_kv_heads, 256).astype(np.float16) * 0.1

        ref = numpy_decode_attention(Q, K, V)
        result = run_decode(dev, decode_func, Q, K, V, kv_len, num_heads, num_kv_heads)

        max_err = float(np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32))))
        ok = max_err < 1e-2
        status = "PASS" if ok else "FAIL"
        print(f"  decode h={num_heads:2d} kv_h={num_kv_heads} kv_len={kv_len:5d}: "
              f"max_err={max_err:.6f} {status}")
        if not ok:
            all_pass = False

    return all_pass


# ============================================================================
# Test 3 & 4: Benchmark v3 vs tuned prefill
# ============================================================================

def bench_prefill(dev, func, num_heads, num_kv_heads, seq_len, causal=1,
                  grid_y_fn=None, iters=50, warmup=10):
    """Time prefill kernel over `iters` iterations after `warmup` warm-up runs."""
    head_dim = 256
    Q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float16) * 0.1
    K = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
    V = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
    out_bytes = Q.nbytes

    d_Q = dev.malloc(Q.nbytes);  dev.upload(d_Q, Q.tobytes())
    d_K = dev.malloc(K.nbytes);  dev.upload(d_K, K.tobytes())
    d_V = dev.malloc(V.nbytes);  dev.upload(d_V, V.tobytes())
    d_Out = dev.malloc(out_bytes)

    if grid_y_fn is None:
        grid_y = (seq_len + BLOCK_M - 1) // BLOCK_M
    else:
        grid_y = grid_y_fn(seq_len)

    params = [
        ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
        ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
        ctypes.c_uint32(seq_len), ctypes.c_uint32(seq_len),
        ctypes.c_uint32(num_heads), ctypes.c_uint32(num_kv_heads),
        ctypes.c_uint32(causal),
    ]

    for _ in range(warmup):
        dev.launch(func, (num_heads, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(func, (num_heads, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    dev.free(d_Q); dev.free(d_K); dev.free(d_V); dev.free(d_Out)
    return t_us


def run_benchmarks(dev, v3_module, tuned_module):
    """Benchmark v3 prefill vs tuned prefill at seq_len=128,512, heads=48, kv_heads=8."""
    print("\n=== Benchmark: V3 Prefill vs Tuned Prefill (heads=48, kv_heads=8, causal=1) ===")
    v3_func     = dev.get_kernel(v3_module,    "flash_attn_256_v3_prefill")
    tuned_func  = dev.get_kernel(tuned_module, "flash_attn_256_prefill")

    num_heads    = 48
    num_kv_heads = 8
    iters        = 30

    print(f"{'Seq len':>8} | {'Tuned (ms)':>12} | {'V3 (ms)':>12} | {'Speedup':>8} | Notes")
    print("-" * 65)

    results = {}
    for seq_len in [128, 512]:
        t_tuned = bench_prefill(dev, tuned_func, num_heads, num_kv_heads, seq_len,
                                causal=1, grid_y_fn=lambda s: (s + 3) // 4, iters=iters)
        t_v3    = bench_prefill(dev, v3_func, num_heads, num_kv_heads, seq_len,
                                causal=1, grid_y_fn=None, iters=iters)
        speedup = t_tuned / t_v3
        results[seq_len] = (t_tuned, t_v3, speedup)
        print(f"  {seq_len:6d}   | {t_tuned/1000:10.3f}   | {t_v3/1000:10.3f}   | {speedup:6.2f}x  | "
              f"{'faster' if speedup >= 1.0 else 'SLOWER'}")

    # Additional seq_lens for completeness
    for seq_len in [64, 256]:
        t_tuned = bench_prefill(dev, tuned_func, num_heads, num_kv_heads, seq_len,
                                causal=1, grid_y_fn=lambda s: (s + 3) // 4, iters=iters)
        t_v3    = bench_prefill(dev, v3_func, num_heads, num_kv_heads, seq_len,
                                causal=1, grid_y_fn=None, iters=iters)
        speedup = t_tuned / t_v3
        print(f"  {seq_len:6d}   | {t_tuned/1000:10.3f}   | {t_v3/1000:10.3f}   | {speedup:6.2f}x")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("FlashAttention v3 Test Suite")
    print("=" * 60)

    dev = GPUDevice(0)
    np.random.seed(42)

    # Load kernels
    v3_module    = load_v3_kernel(dev)
    tuned_module = load_tuned_kernel(dev)

    v3_prefill_func = dev.get_kernel(v3_module, "flash_attn_256_v3_prefill")
    decode_func     = dev.get_kernel(v3_module, "flash_attn_256_decode")

    # Run correctness tests
    prefill_ok = test_v3_prefill_correctness(dev, v3_prefill_func)
    decode_ok  = test_decode_regression(dev, decode_func)

    all_correct = prefill_ok and decode_ok
    print(f"\nCorrectness summary: {'ALL PASSED' if all_correct else 'SOME FAILED'}")
    if not prefill_ok:
        print("  FAIL: V3 prefill correctness")
    if not decode_ok:
        print("  FAIL: Decode regression")

    # Run benchmarks regardless of correctness (to capture performance data)
    bench_results = run_benchmarks(dev, v3_module, tuned_module)

    print("\n=== Summary ===")
    print(f"V3 Design: BLOCK_M=16 (4 Q rows/wavefront), BLOCK_N=16,")
    print(f"           v_dot2_f32_f16 for score computation (2x arithmetic throughput),")
    print(f"           16 KB LDS (K+V blocks), FP32 accumulators, vectorized float4 loads.")
    print(f"Tile size note: BLOCK_M=16 (not 64) chosen due to VGPR budget on gfx906.")
    print(f"  Br=64 (16 Q rows/wf × 4-thread reduction) would need 64 Q-dims/thread")
    print(f"  in registers = too many VGPRs. Br=16 (4 Q rows/wf × 16-thread reduction)")
    print(f"  uses 16 dims/thread = ~50 VGPRs total (healthy occupancy).")

    dev.cleanup()
    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
