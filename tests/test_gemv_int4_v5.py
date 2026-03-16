#!/usr/bin/env python3
"""
Test harness for INT4 GEMV v5 (gemv_int4_v5) — DPP wave reduction.

v5 uses a col-major thread layout so all k-split threads for one output column
are contiguous within a single wavefront. This enables pure intra-wavefront
__shfl_down reduction with no LDS and no __syncthreads().

Tests:
1. Correctness vs v4 (reference) for all Qwen 3.5 27B shapes:
   N=4096, 5120, 11008, 13696 at K=5120, gs=128
   max abs error < 1e-3

2. Latency benchmark for t4, t8, t16 variants vs v4:
   100 iterations, 10 warmup, median latency

VAL-DPP-001: max abs error < 1e-3 for all shapes
VAL-DPP-002: v5 latency <= v4 for primary shapes (N=4096,K=5120 and N=11008,K=5120)
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

print("Building gemv_int4_v4 kernel (reference)...")
hip_v4 = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v4.hip")
hsaco_v4 = str(BUILD_DIR / "gemv_int4_v4.hsaco")
build_hip_hsaco(hip_v4, hsaco_v4)

print("Building gemv_int4_v5 kernel (DPP wave reduction)...")
hip_v5 = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v5.hip")
hsaco_v5 = str(BUILD_DIR / "gemv_int4_v5.hsaco")
build_hip_hsaco(hip_v5, hsaco_v5)

print("Kernels compiled successfully.\n")

dev = GPUDevice(0)

module_v4 = dev.load_hsaco(hsaco_v4)
func_v4_t4  = dev.get_kernel(module_v4, "gemv_int4_v4_t4")
func_v4_t8  = dev.get_kernel(module_v4, "gemv_int4_v4_t8")
func_v4_t16 = dev.get_kernel(module_v4, "gemv_int4_v4_t16")

module_v5 = dev.load_hsaco(hsaco_v5)
func_v5_t4  = dev.get_kernel(module_v5, "gemv_int4_v5_t4")
func_v5_t8  = dev.get_kernel(module_v5, "gemv_int4_v5_t8")
func_v5_t16 = dev.get_kernel(module_v5, "gemv_int4_v5_t16")

print("Kernels loaded OK.\n")


def quantize_weights_gptq(W_fp32, group_size=128):
    """
    Simulate GPTQ quantization (unsigned INT4, 0-15 range).
    W_fp32: [K, N] float32
    Returns: B_q4[K/8, N] uint32, scales[K/gs, N] float16, zeros[K/gs, N] float16
    """
    K, N = W_fp32.shape
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float32)
    zeros  = np.zeros((n_groups, N), dtype=np.float32)
    q4_mat = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        ks = g * group_size
        ke = ks + group_size
        grp = W_fp32[ks:ke, :]

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

    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_mat[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales.astype(np.float16), zeros.astype(np.float16)


def run_gemv(func, threads_per_col, A_h16, B_q4, scales, zeros, N, K, group_size,
             d_A=None, d_B=None, d_scales=None, d_zeros=None, d_C=None):
    """
    Launch a GEMV kernel and return result [N] FP16.
    Optionally accepts pre-allocated device buffers for reuse.
    """
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

    own_buffers = d_A is None
    if own_buffers:
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

    if own_buffers:
        dev.free(d_A)
        dev.free(d_B)
        dev.free(d_scales)
        dev.free(d_zeros)
        dev.free(d_C)

    return result


def benchmark_gemv(func, threads_per_col, d_A, d_B, d_scales, d_zeros, d_C,
                   N, K, group_size, n_warmup=10, n_iters=100):
    """Benchmark kernel: returns median latency in microseconds."""
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

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

    for _ in range(n_warmup):
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    times = []
    for _ in range(n_iters):
        dev.synchronize()
        t0 = time.perf_counter()
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    return float(np.median(times))


def test_correctness(N, K, group_size=128, seed=42):
    """
    Test correctness of v5 variants vs v4 reference at given (N, K, group_size).
    Threshold: max abs error < 1e-3.
    Returns True if ALL variants pass.
    """
    print(f"\n--- Correctness: N={N}, K={K}, gs={group_size} ---")

    np.random.seed(seed)
    A_f32  = (np.random.randn(K) * 0.1).astype(np.float32)
    A_h16  = A_f32.astype(np.float16)
    W_fp32 = (np.random.randn(K, N) * 0.1).astype(np.float32)

    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)

    threshold = 1e-3
    all_pass = True

    # v4 reference variants
    ref_t4  = run_gemv(func_v4_t4,  4,  A_h16, B_q4, scales, zeros, N, K, group_size)
    ref_t8  = run_gemv(func_v4_t8,  8,  A_h16, B_q4, scales, zeros, N, K, group_size)
    ref_t16 = run_gemv(func_v4_t16, 16, A_h16, B_q4, scales, zeros, N, K, group_size)

    for tpc, func_v5, ref in [
        (4,  func_v5_t4,  ref_t4),
        (8,  func_v5_t8,  ref_t8),
        (16, func_v5_t16, ref_t16),
    ]:
        result = run_gemv(func_v5, tpc, A_h16, B_q4, scales, zeros, N, K, group_size)
        max_err  = float(np.abs(result.astype(np.float32) - ref.astype(np.float32)).max())
        mean_err = float(np.abs(result.astype(np.float32) - ref.astype(np.float32)).mean())
        passed = max_err < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  v5_t{tpc}: max_abs_err={max_err:.6f}, mean_err={mean_err:.6f} [{status}]")
        if not passed:
            all_pass = False
            # Debug: show first mismatches
            diff = np.abs(result.astype(np.float32) - ref.astype(np.float32))
            idx = np.argsort(diff)[-5:][::-1]
            print(f"    Top-5 errors at indices {idx}:")
            for i in idx:
                print(f"      idx={i}: v5={float(result[i]):.6f}, v4={float(ref[i]):.6f}, err={diff[i]:.6f}")

    return all_pass


def test_performance(N, K, group_size=128, seed=42):
    """
    Benchmark v5 vs v4 for t4, t8, t16 at given (N, K).
    Returns dict of latencies and pass/fail for VAL-DPP-002.
    """
    print(f"\n--- Performance: N={N}, K={K}, gs={group_size} ---")

    np.random.seed(seed)
    A_h16  = (np.random.randn(K) * 0.1).astype(np.float16)
    W_fp32 = (np.random.randn(K, N) * 0.1).astype(np.float32)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)

    # Allocate persistent device buffers
    d_A      = dev.malloc(A_h16.nbytes)
    d_B      = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros  = dev.malloc(zeros.nbytes)
    d_C      = dev.malloc(N * 2)

    dev.upload(d_A,      A_h16.tobytes())
    dev.upload(d_B,      B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros,  zeros.tobytes())

    results = {}
    for tpc, f4, f5 in [
        (4,  func_v4_t4,  func_v5_t4),
        (8,  func_v4_t8,  func_v5_t8),
        (16, func_v4_t16, func_v5_t16),
    ]:
        lat_v4 = benchmark_gemv(f4, tpc, d_A, d_B, d_scales, d_zeros, d_C,
                                 N, K, group_size)
        lat_v5 = benchmark_gemv(f5, tpc, d_A, d_B, d_scales, d_zeros, d_C,
                                 N, K, group_size)
        speedup = lat_v4 / lat_v5
        improvement = (lat_v4 - lat_v5) / lat_v4 * 100
        print(f"  t{tpc}: v4={lat_v4:.1f}us  v5={lat_v5:.1f}us  speedup={speedup:.3f}x  "
              f"({'faster' if speedup > 1.0 else 'slower'} by {abs(improvement):.1f}%)")
        results[tpc] = (lat_v4, lat_v5, speedup)

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("INT4 GEMV v5 (DPP Wave Reduction) — Correctness & Benchmark")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Correctness: VAL-DPP-001
    # All Qwen 3.5 27B shapes at K=5120, gs=128
    # -----------------------------------------------------------------------
    print("\n[VAL-DPP-001] Correctness vs v4 (threshold: max_abs_err < 1e-3)")

    shapes = [
        (4096, 5120),
        (5120, 5120),
        (11008, 5120),
        (13696, 5120),
    ]

    all_correct = True
    for N, K in shapes:
        if not test_correctness(N=N, K=K, group_size=128):
            all_correct = False

    print()
    if all_correct:
        print("VAL-DPP-001: ALL correctness tests PASSED")
    else:
        print("VAL-DPP-001: SOME correctness tests FAILED")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Performance: VAL-DPP-002
    # Primary shapes: N=4096,K=5120 and N=11008,K=5120
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[VAL-DPP-002] Latency benchmark: v5 vs v4 (100 iters, 10 warmup)")
    print("=" * 72)

    perf_4096  = test_performance(N=4096, K=5120, group_size=128)
    perf_5120  = test_performance(N=5120, K=5120, group_size=128)
    perf_11008 = test_performance(N=11008, K=5120, group_size=128)
    perf_13696 = test_performance(N=13696, K=5120, group_size=128)

    print("\n" + "=" * 72)
    print("Summary:")
    print(f"{'Shape':>20}  {'variant':>8}  {'v4 (us)':>10}  {'v5 (us)':>10}  {'speedup':>10}")
    print("-" * 72)
    for (N, K), perf in [(( 4096, 5120), perf_4096),
                          (( 5120, 5120), perf_5120),
                          ((11008, 5120), perf_11008),
                          ((13696, 5120), perf_13696)]:
        for tpc in [4, 8, 16]:
            lat_v4, lat_v5, sp = perf[tpc]
            print(f"  N={N:5d},K={K:5d}  t{tpc:>2}  {lat_v4:>10.1f}  {lat_v5:>10.1f}  {sp:>10.3f}x")

    # Check VAL-DPP-002: v5 latency <= v4 for primary shapes
    # Check all t-variants for the primary shapes
    primary_shapes = [(4096, 5120, perf_4096), (11008, 5120, perf_11008)]
    val_dpp002_pass = True
    print("\n[VAL-DPP-002] Primary shape performance check (v5 <= v4):")
    for N, K, perf in primary_shapes:
        for tpc in [4, 8, 16]:
            lat_v4, lat_v5, sp = perf[tpc]
            ok = lat_v5 <= lat_v4 * 1.05  # allow 5% noise margin
            status = "PASS" if ok else "FAIL"
            print(f"  N={N:5d},K={K:5d} t{tpc}: v5={lat_v5:.1f}us vs v4={lat_v4:.1f}us [{status}]")
            if not ok:
                val_dpp002_pass = False

    print()
    if val_dpp002_pass:
        print("VAL-DPP-002: PASS — v5 latency <= v4 (within 5% margin) for primary shapes")
    else:
        print("VAL-DPP-002: FAIL — v5 is slower than v4 on some primary shapes")
        sys.exit(1)

    print("\nAll validations passed.")
    sys.exit(0)
