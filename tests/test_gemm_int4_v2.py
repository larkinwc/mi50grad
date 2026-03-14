#!/usr/bin/env python3
"""
Test harness for INT4 GEMM v2 kernel (gemm_int4_prefill_v2) with on-the-fly
dequantization vs the original HIP INT4 GEMM (gemm_int4_prefill_hip).

Tests:
1. Correctness at (M=128, N=4096, K=4096, gs=128)
2. Correctness at (M=64, N=11008, K=4096, gs=128)
3. Benchmark v2 vs original at (M=128, N=4096, K=4096)

Reports max abs error and speedup.
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

# ---- Build both kernels ----
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

dev = GPUDevice(0)

# Original kernel
hip_orig = str(PROJECT_ROOT / "src" / "kernels" / "gemm_int4_prefill_hip.hip")
hsaco_orig = str(BUILD_DIR / "gemm_int4_prefill_hip.hsaco")
build_hip_hsaco(hip_orig, hsaco_orig)
module_orig = dev.load_hsaco(hsaco_orig)
func_orig = dev.get_kernel(module_orig, "gemm_int4_prefill_hip")

# v2 kernel (on-the-fly dequant)
hip_v2 = str(PROJECT_ROOT / "src" / "kernels" / "gemm_int4_prefill_v2.hip")
hsaco_v2 = str(BUILD_DIR / "gemm_int4_prefill_v2.hsaco")
build_hip_hsaco(hip_v2, hsaco_v2)
module_v2 = dev.load_hsaco(hsaco_v2)
func_v2 = dev.get_kernel(module_v2, "gemm_int4_prefill_v2")

print("Both kernels compiled and loaded.")


def quantize_weights_gptq(W_fp16, group_size=128):
    """
    Simulate GPTQ quantization of a weight matrix.

    W_fp16: [K, N] FP16 weight matrix
    Returns: B_q4 [K/8, N] uint32, scales [K/group_size, N] FP16, zeros [K/group_size, N] FP16

    Kernel formula: w = (q - zero) * scale
    So: zero = -w_min / scale,  scale = (w_max - w_min) / 15
    """
    K, N = W_fp16.shape
    W = W_fp16.astype(np.float32)
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float16)
    zeros = np.zeros((n_groups, N), dtype=np.float16)
    q4_matrix = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        group = W[k_start:k_end, :]  # [group_size, N]

        w_min = group.min(axis=0)  # [N]
        w_max = group.max(axis=0)  # [N]

        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0, 1.0, scale)

        zero = -w_min / scale

        q = np.round((group - w_min[np.newaxis, :]) / scale[np.newaxis, :])
        q = np.clip(q, 0, 15).astype(np.uint8)

        scales[g, :] = scale.astype(np.float16)
        zeros[g, :] = zero.astype(np.float16)
        q4_matrix[k_start:k_end, :] = q

    # Pack: B_q4[K/8, N], 8 consecutive K values per uint32
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_matrix[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales, zeros


def dequant_reference(B_q4, scales, zeros, K, N, group_size):
    """CPU reference dequantization: w = (q - zero) * scale. Returns [K, N] FP32."""
    K8 = K // 8
    W = np.zeros((K, N), dtype=np.float32)
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)

    for k8 in range(K8):
        packed = B_q4[k8, :]  # [N] uint32
        for i in range(8):
            k_idx = k8 * 8 + i
            q_val = ((packed >> (i * 4)) & 0xF).astype(np.float32)
            g_idx = k_idx // group_size
            W[k_idx, :] = (q_val - zeros_f32[g_idx, :]) * scales_f32[g_idx, :]
    return W


def run_kernel(func, A, B_q4, scales, zeros, dev, group_size):
    """Launch kernel and return result as numpy array [M, N] FP16."""
    M, K = A.shape
    K8, N = B_q4.shape

    d_A      = dev.malloc(A.nbytes)
    d_B      = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros  = dev.malloc(zeros.nbytes)
    d_C      = dev.malloc(M * N * 2)

    dev.upload(d_A,      A.tobytes())
    dev.upload(d_B,      B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros,  zeros.tobytes())
    dev.hip.memset(d_C, 0, M * N * 2)

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
        ctypes.c_uint32(group_size),
    ]

    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_C, M * N * 2), dtype=np.float16).copy()
    result = result.reshape(M, N)

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)
    return result


def benchmark_kernel(func, A, B_q4, scales, zeros, dev, group_size, n_warmup=10, n_iters=100):
    """Benchmark kernel: returns median latency in microseconds."""
    M, K = A.shape
    K8, N = B_q4.shape

    d_A      = dev.malloc(A.nbytes)
    d_B      = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros  = dev.malloc(zeros.nbytes)
    d_C      = dev.malloc(M * N * 2)

    dev.upload(d_A,      A.tobytes())
    dev.upload(d_B,      B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros,  zeros.tobytes())

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
        ctypes.c_uint32(group_size),
    ]

    # Warmup
    for _ in range(n_warmup):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    # Timed runs
    times = []
    for _ in range(n_iters):
        dev.synchronize()
        t0 = time.perf_counter()
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)

    return float(np.median(times))


def test_correctness(M, N, K, group_size=128):
    """Test correctness of v2 kernel vs CPU reference. Returns True on PASS."""
    print(f"\n--- Correctness: M={M}, N={N}, K={K}, gs={group_size} ---")

    # N must be multiple of 64
    assert N % 64 == 0, f"N={N} must be multiple of 64"
    assert M % 64 == 0, f"M={M} must be multiple of 64"
    assert K % group_size == 0, f"K={K} must be multiple of group_size={group_size}"

    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np.float16)
    W_fp16 = (np.random.randn(K, N) * 0.1).astype(np.float16)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp16, group_size)

    # CPU reference
    W_deq = dequant_reference(B_q4, scales, zeros, K, N, group_size)
    C_ref = (A.astype(np.float32) @ W_deq).astype(np.float16)

    # GPU v2
    C_v2 = run_kernel(func_v2, A, B_q4, scales, zeros, dev, group_size)

    max_err = float(np.abs(C_v2.astype(np.float32) - C_ref.astype(np.float32)).max())
    mean_err = float(np.abs(C_v2.astype(np.float32) - C_ref.astype(np.float32)).mean())

    threshold = 1e-2
    passed = max_err < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  v2 max_abs_err={max_err:.6f}, mean_err={mean_err:.6f} [{status}]")

    if not passed:
        print(f"  C_ref[0,:4] = {C_ref[0,:4]}")
        print(f"  C_v2[0,:4]  = {C_v2[0,:4]}")

    return passed


def test_benchmark(M, N, K, group_size=128):
    """Benchmark v2 vs original. Returns speedup ratio."""
    print(f"\n--- Benchmark: M={M}, N={N}, K={K}, gs={group_size} ---")

    assert N % 64 == 0
    assert M % 64 == 0

    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np.float16)
    W_fp16 = (np.random.randn(K, N) * 0.1).astype(np.float16)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp16, group_size)

    lat_orig = benchmark_kernel(func_orig, A, B_q4, scales, zeros, dev, group_size)
    lat_v2   = benchmark_kernel(func_v2,   A, B_q4, scales, zeros, dev, group_size)

    speedup = lat_orig / lat_v2
    print(f"  Original:   {lat_orig:.1f} us")
    print(f"  v2 (on-the-fly dequant): {lat_v2:.1f} us")
    print(f"  Speedup:    {speedup:.3f}x")

    return speedup


if __name__ == "__main__":
    print("=" * 60)
    print("INT4 GEMM v2 (on-the-fly dequant) — Correctness & Benchmark")
    print("=" * 60)

    all_pass = True

    # Correctness test 1: (M=128, N=4096, K=4096, gs=128)
    if not test_correctness(128, 4096, 4096, group_size=128):
        all_pass = False

    # Correctness test 2: (M=64, N=11008, K=4096, gs=128)
    # Note: N=11008 rounded up to multiple of 64 = 11008 (11008 % 64 = 0)
    if not test_correctness(64, 11008, 4096, group_size=128):
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("All correctness tests PASSED")
    else:
        print("Some correctness tests FAILED")
        sys.exit(1)

    # Benchmark: v2 vs original at (M=128, N=4096, K=4096)
    print()
    speedup = test_benchmark(128, 4096, 4096, group_size=128)

    print("\n" + "=" * 60)
    print(f"Final speedup (v2 vs original): {speedup:.3f}x")
    print("=" * 60)
    sys.exit(0)
