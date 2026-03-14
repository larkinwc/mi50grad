#!/usr/bin/env python3
"""
Test harness for INT4 GEMV v4 kernel (gemv_int4_v4) using v_dot2_f32_f16.

Approach: Extract weight nibbles into FP16 pairs, use __builtin_amdgcn_fdot2
(v_dot2_f32_f16) for 2 FMAs per instruction. Avoids activation quantization
error entirely (FP16 activations used directly).

Tests:
1. Correctness at (N=4096, K=4096, gs=128) — gate/up projection shape
2. Correctness at (N=11008, K=4096, gs=128) — FFN intermediate shape
3. Benchmark v4_t16 vs v3_t16 at both shapes (latency comparison)

Reports max abs error and PASS/FAIL for each correctness test.
Reports latency comparison for each shape.
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

# ---- Build kernels ----
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

print("Building gemv_int4_v3 kernel...")
hip_v3 = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v3.hip")
hsaco_v3 = str(BUILD_DIR / "gemv_int4_v3.hsaco")
build_hip_hsaco(hip_v3, hsaco_v3)

print("Building gemv_int4_v4 kernel...")
hip_v4 = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v4.hip")
hsaco_v4 = str(BUILD_DIR / "gemv_int4_v4.hsaco")
build_hip_hsaco(hip_v4, hsaco_v4)

print("Kernels compiled successfully.")

dev = GPUDevice(0)

module_v3 = dev.load_hsaco(hsaco_v3)
func_v3_t4  = dev.get_kernel(module_v3, "gemv_int4_v3_t4")
func_v3_t8  = dev.get_kernel(module_v3, "gemv_int4_v3_t8")
func_v3_t16 = dev.get_kernel(module_v3, "gemv_int4_v3_t16")

module_v4 = dev.load_hsaco(hsaco_v4)
func_v4_t4  = dev.get_kernel(module_v4, "gemv_int4_v4_t4")
func_v4_t8  = dev.get_kernel(module_v4, "gemv_int4_v4_t8")
func_v4_t16 = dev.get_kernel(module_v4, "gemv_int4_v4_t16")

print("Kernels loaded OK.\n")


def quantize_weights_gptq(W_fp32, group_size=128):
    """
    Simulate GPTQ quantization (unsigned INT4, 0-15 range).

    W_fp32: [K, N] float32 weight matrix
    Returns:
      B_q4: [K/8, N] uint32 — 8 nibbles per uint32, K-major
      scales: [K/group_size, N] float16
      zeros: [K/group_size, N] float16
    Dequant formula: w = (q - zero) * scale
    """
    K, N = W_fp32.shape
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float32)
    zeros  = np.zeros((n_groups, N), dtype=np.float32)
    q4_mat = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        ks = g * group_size
        ke = ks + group_size
        grp = W_fp32[ks:ke, :]  # [group_size, N]

        w_min = grp.min(axis=0)
        w_max = grp.max(axis=0)

        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0.0, 1.0, scale)
        zero  = -w_min / scale

        q = np.round((grp - w_min[np.newaxis, :]) / scale[np.newaxis, :])
        q = np.clip(q, 0, 15).astype(np.uint8)

        scales[g, :] = scale
        zeros[g, :]  = zero
        q4_mat[ks:ke, :] = q

    # Pack: B_q4[K/8, N] uint32 — bits[4*i+3:4*i] = nibble i
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_mat[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales.astype(np.float16), zeros.astype(np.float16)


def dequant_reference(B_q4, scales, zeros, K, N, group_size):
    """CPU reference dequantization: w = (q - zero) * scale. Returns [K, N] FP32."""
    K8 = K // 8
    W = np.zeros((K, N), dtype=np.float32)
    scales_f = scales.astype(np.float32)
    zeros_f  = zeros.astype(np.float32)

    for k8 in range(K8):
        packed = B_q4[k8, :]
        for i in range(8):
            k_idx = k8 * 8 + i
            q_val = ((packed >> (i * 4)) & 0xF).astype(np.float32)
            g_idx = k_idx // group_size
            W[k_idx, :] = (q_val - zeros_f[g_idx, :]) * scales_f[g_idx, :]
    return W


def run_gemv(func, threads_per_col, A_h16, B_q4, scales, zeros, N, K, group_size):
    """
    Launch a GEMV kernel and return result [N] FP16.
    A_h16: [K] FP16 activation vector
    B_q4: [K/8, N] uint32 weights
    scales, zeros: [K/group_size, N] FP16
    """
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

    d_A      = dev.malloc(A_h16.nbytes)
    d_B      = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros  = dev.malloc(zeros.nbytes)
    d_C      = dev.malloc(N * 2)

    dev.upload(d_A,      A_h16.tobytes())
    dev.upload(d_B,      B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros,  zeros.tobytes())

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]

    dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16).copy()

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)

    return result


def benchmark_gemv(func, threads_per_col, A_h16, B_q4, scales, zeros, N, K, group_size,
                   n_warmup=10, n_iters=100):
    """Benchmark kernel: returns median latency in microseconds."""
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

    d_A      = dev.malloc(A_h16.nbytes)
    d_B      = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros  = dev.malloc(zeros.nbytes)
    d_C      = dev.malloc(N * 2)

    dev.upload(d_A,      A_h16.tobytes())
    dev.upload(d_B,      B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros,  zeros.tobytes())

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]

    # Warmup
    for _ in range(n_warmup):
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    # Timed
    times = []
    for _ in range(n_iters):
        dev.synchronize()
        t0 = time.perf_counter()
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)

    return float(np.median(times))


def test_correctness(N, K, group_size=128, seed=42):
    """
    Test correctness of v4 kernel variants at given (N, K, group_size).
    Returns True if all variants pass (max abs error < 1e-2).
    """
    print(f"\n--- Correctness: N={N}, K={K}, gs={group_size} ---")

    np.random.seed(seed)
    A_f32 = (np.random.randn(K) * 0.1).astype(np.float32)
    A_h16 = A_f32.astype(np.float16)
    W_fp32 = (np.random.randn(K, N) * 0.1).astype(np.float32)

    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)
    W_deq = dequant_reference(B_q4, scales, zeros, K, N, group_size)

    # CPU reference: y = A @ W_deq
    ref = (A_f32 @ W_deq).astype(np.float16)

    threshold = 1e-2
    all_pass = True

    # Test v3_t16 for comparison
    result_v3 = run_gemv(func_v3_t16, 16, A_h16, B_q4, scales, zeros, N, K, group_size)
    err_v3 = float(np.abs(result_v3.astype(np.float32) - ref.astype(np.float32)).max())
    pass_v3 = err_v3 < threshold
    print(f"  v3_t16: max_abs_err={err_v3:.6f} [{'PASS' if pass_v3 else 'FAIL'}]")

    # Test all v4 variants
    for tpc, func_v4 in [(4, func_v4_t4), (8, func_v4_t8), (16, func_v4_t16)]:
        result = run_gemv(func_v4, tpc, A_h16, B_q4, scales, zeros, N, K, group_size)
        max_err = float(np.abs(result.astype(np.float32) - ref.astype(np.float32)).max())
        mean_err = float(np.abs(result.astype(np.float32) - ref.astype(np.float32)).mean())
        passed = max_err < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  v4_t{tpc}: max_abs_err={max_err:.6f}, mean_err={mean_err:.6f} [{status}]")
        if not passed:
            all_pass = False
            # Print some debug values
            print(f"    ref[0:4]    = {ref.astype(np.float32)[:4]}")
            print(f"    result[0:4] = {result.astype(np.float32)[:4]}")

    return all_pass


def test_performance(N, K, group_size=128, seed=42):
    """
    Benchmark v4_t16 vs v3_t16 at given (N, K, group_size).
    Prints latency comparison and speedup.
    """
    print(f"\n--- Performance: N={N}, K={K}, gs={group_size} ---")

    np.random.seed(seed)
    A_h16 = (np.random.randn(K) * 0.1).astype(np.float16)
    W_fp32 = (np.random.randn(K, N) * 0.1).astype(np.float32)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)

    lat_v3 = benchmark_gemv(func_v3_t16, 16, A_h16, B_q4, scales, zeros, N, K, group_size)
    lat_v4 = benchmark_gemv(func_v4_t16, 16, A_h16, B_q4, scales, zeros, N, K, group_size)

    # Also benchmark v4_t4 and v4_t8 for reference
    lat_v4_t4 = benchmark_gemv(func_v4_t4, 4, A_h16, B_q4, scales, zeros, N, K, group_size)
    lat_v4_t8 = benchmark_gemv(func_v4_t8, 8, A_h16, B_q4, scales, zeros, N, K, group_size)

    speedup = lat_v3 / lat_v4

    # Theoretical bandwidth
    weight_bytes = K * N // 2       # INT4 = K*N/2 bytes
    act_bytes    = K * 2            # FP16 activation
    out_bytes    = N * 2            # FP16 output
    total_bytes  = weight_bytes + act_bytes + out_bytes
    theoretical_us = total_bytes / 857e9 * 1e6  # ~857 GB/s MI60 HBM bandwidth

    print(f"  v3_t16:  {lat_v3:.1f} us")
    print(f"  v4_t4:   {lat_v4_t4:.1f} us")
    print(f"  v4_t8:   {lat_v4_t8:.1f} us")
    print(f"  v4_t16:  {lat_v4:.1f} us  (speedup vs v3: {speedup:.3f}x)")
    print(f"  theoretical min ({total_bytes/1e6:.1f}MB @ 857GB/s): {theoretical_us:.1f} us")

    if speedup >= 1.0:
        print(f"  Result: v4_t16 is {speedup:.3f}x FASTER than v3_t16")
    else:
        print(f"  Result: v4_t16 is {1/speedup:.3f}x SLOWER than v3_t16")
        print(f"  Note: fdot2 packs 2 FMAs per instruction but adds FP16 conversion overhead.")
        print(f"  The v3 ubfe approach may be compiler-optimized as well as fdot2 here.")

    return lat_v3, lat_v4, speedup


if __name__ == "__main__":
    print("=" * 70)
    print("INT4 GEMV v4 (fdot2 / v_dot2_f32_f16) — Correctness & Benchmark")
    print("=" * 70)

    all_pass = True

    # Test 1: N=4096, K=4096, gs=128 (gate/up projection shape)
    if not test_correctness(N=4096, K=4096, group_size=128):
        all_pass = False

    # Test 2: N=11008, K=4096, gs=128 (FFN intermediate shape)
    if not test_correctness(N=11008, K=4096, group_size=128):
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("All correctness tests PASSED")
    else:
        print("Some correctness tests FAILED")
        sys.exit(1)

    # Latency comparison
    print("\n" + "=" * 70)
    print("Latency Comparison: v4_t16 vs v3_t16")
    print("=" * 70)

    lat_v3_4096, lat_v4_4096, sp_4096 = test_performance(N=4096, K=4096, group_size=128)
    lat_v3_11008, lat_v4_11008, sp_11008 = test_performance(N=11008, K=4096, group_size=128)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  N=4096,  K=4096: v3={lat_v3_4096:.1f}us, v4={lat_v4_4096:.1f}us, speedup={sp_4096:.3f}x")
    print(f"  N=11008, K=4096: v3={lat_v3_11008:.1f}us, v4={lat_v4_11008:.1f}us, speedup={sp_11008:.3f}x")
    print("=" * 70)
    sys.exit(0)
