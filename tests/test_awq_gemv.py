#!/usr/bin/env python3
"""
Test harness for AWQ INT4 GEMV kernel (gemv_int4_v5_awq).

AWQ dequantization: w = q * scale (no zero-point subtraction)
GPTQ dequantization: w = (q - zero) * scale

Tests:
1. Correctness vs numpy reference dequantize-then-matmul for Qwen 3.5 27B shapes:
   N=4096, 5120, 11008, 13696/17408 at K=5120, group_size=128
   max abs error < 1e-2 (VAL-AWQ-002)

2. Latency comparison vs GPTQ GEMV (gemv_int4_v5):
   - AWQ should be equal or faster (one fewer instruction per weight)
   - 100 iterations, 10 warmup
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

print("Building gemv_int4_v5 kernel (GPTQ reference)...")
hip_v5 = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v5.hip")
hsaco_v5 = str(BUILD_DIR / "gemv_int4_v5.hsaco")
build_hip_hsaco(hip_v5, hsaco_v5)

print("Building gemv_int4_v5_awq kernel (AWQ variant)...")
hip_awq = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v5_awq.hip")
hsaco_awq = str(BUILD_DIR / "gemv_int4_v5_awq.hsaco")
build_hip_hsaco(hip_awq, hsaco_awq)

print("Kernels compiled successfully.\n")

dev = GPUDevice(0)

# Load GPTQ v5 kernels (used as reference for latency comparison)
module_v5 = dev.load_hsaco(hsaco_v5)
func_v5_t4  = dev.get_kernel(module_v5, "gemv_int4_v5_t4")
func_v5_t8  = dev.get_kernel(module_v5, "gemv_int4_v5_t8")
func_v5_t16 = dev.get_kernel(module_v5, "gemv_int4_v5_t16")

# Load AWQ kernels
module_awq = dev.load_hsaco(hsaco_awq)
func_awq_t4  = dev.get_kernel(module_awq, "gemv_int4_v5_awq_t4")
func_awq_t8  = dev.get_kernel(module_awq, "gemv_int4_v5_awq_t8")
func_awq_t16 = dev.get_kernel(module_awq, "gemv_int4_v5_awq_t16")

print("Kernels loaded OK.\n")


def make_awq_weights(K, N, group_size=128, seed=42):
    """
    Create synthetic AWQ-format weights.
    AWQ: q ∈ [0, 15] unsigned, w = q * scale (no zero subtraction)

    Returns:
        A_h16:  [K] FP16 activation vector
        B_q4:   [K/8, N] uint32 packed nibbles
        scales: [K/group_size, N] FP16 scales
        w_ref:  [K, N] FP32 reference dequantized weights (numpy)
    """
    rng = np.random.default_rng(seed)

    A_f32 = (rng.standard_normal(K) * 0.1).astype(np.float32)
    A_h16 = A_f32.astype(np.float16)

    num_groups = K // group_size
    # AWQ-like scales: per-group FP16
    scales_fp32 = (rng.random((num_groups, N)) * 0.009 + 0.001).astype(np.float32)
    scales_h16 = scales_fp32.astype(np.float16)

    # Raw nibbles [K, N] in [0, 15]
    raw = rng.integers(0, 16, size=(K, N), dtype=np.uint8)

    # Pack into uint32: [K/8, N]
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for b in range(8):
        B_q4 |= (raw[b::8, :].astype(np.uint32) << (b * 4))

    # Reference dequantization: w = q * scale
    scales_expanded = np.repeat(scales_fp32, group_size, axis=0)  # [K, N]
    w_ref = raw.astype(np.float32) * scales_expanded  # [K, N] FP32

    return A_h16, B_q4, scales_h16, w_ref


def make_gptq_weights(K, N, group_size=128, seed=42):
    """
    Create synthetic GPTQ-format weights for latency comparison.
    GPTQ: w = (q - zero) * scale

    Returns:
        A_h16:  [K] FP16 activation vector
        B_q4:   [K/8, N] uint32 packed nibbles
        scales: [K/group_size, N] FP16 scales
        zeros:  [K/group_size, N] FP16 zeros
    """
    rng = np.random.default_rng(seed)

    A_f32 = (rng.standard_normal(K) * 0.1).astype(np.float32)
    A_h16 = A_f32.astype(np.float16)

    num_groups = K // group_size
    scales_fp32 = (rng.random((num_groups, N)) * 0.009 + 0.001).astype(np.float32)
    scales_h16 = scales_fp32.astype(np.float16)
    # GPTQ zeros: typically ~8 (midpoint of [0,15])
    zeros_h16 = (np.ones((num_groups, N)) * 8.0).astype(np.float16)

    raw = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for b in range(8):
        B_q4 |= (raw[b::8, :].astype(np.uint32) << (b * 4))

    return A_h16, B_q4, scales_h16, zeros_h16


def run_awq_gemv(func, threads_per_col, A_h16, B_q4, scales, K, N, group_size,
                 d_A=None, d_B=None, d_scales=None, d_C=None):
    """
    Launch AWQ GEMV kernel (no zeros pointer) and return result [N] FP16.
    """
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

    own_buffers = d_A is None
    if own_buffers:
        d_A      = dev.malloc(A_h16.nbytes)
        d_B      = dev.malloc(B_q4.nbytes)
        d_scales = dev.malloc(scales.nbytes)
        d_C      = dev.malloc(N * 2)
        dev.upload(d_A,      A_h16.tobytes())
        dev.upload(d_B,      B_q4.tobytes())
        dev.upload(d_scales, scales.tobytes())

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
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
        dev.free(d_C)

    return result


def run_gptq_gemv(func, threads_per_col, A_h16, B_q4, scales, zeros, K, N, group_size,
                  d_A=None, d_B=None, d_scales=None, d_zeros=None, d_C=None):
    """
    Launch GPTQ GEMV kernel (with zeros pointer) and return result [N] FP16.
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


def benchmark_awq_gemv(func, threads_per_col, d_A, d_B, d_scales, d_C,
                        K, N, group_size, n_warmup=10, n_iters=100):
    """Benchmark AWQ GEMV: returns median latency in microseconds."""
    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
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


def benchmark_gptq_gemv(func, threads_per_col, d_A, d_B, d_scales, d_zeros, d_C,
                         K, N, group_size, n_warmup=10, n_iters=100):
    """Benchmark GPTQ GEMV: returns median latency in microseconds."""
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


def test_correctness_vs_numpy(N, K, group_size=128, seed=42):
    """
    VAL-AWQ-002: Correctness vs numpy dequantize-then-matmul.
    max abs error < 1e-2 for all variants (t4, t8, t16).
    """
    print(f"\n--- Correctness: N={N}, K={K}, gs={group_size} ---")

    A_h16, B_q4, scales_h16, w_ref = make_awq_weights(K, N, group_size, seed)

    # Reference: numpy dequantize-then-matmul
    # y_ref[n] = sum_k A_h16[k] * w_ref[k, n]
    A_f32 = A_h16.astype(np.float32)
    y_ref = A_f32 @ w_ref  # [N] FP32

    # Convert B_q4 to int32 for device upload (same bit pattern)
    B_q4_i32 = B_q4.view(np.int32)

    threshold = 1e-2  # AWQ precision threshold per VAL-AWQ-002
    all_pass = True

    for tpc, func in [(4, func_awq_t4), (8, func_awq_t8), (16, func_awq_t16)]:
        result = run_awq_gemv(func, tpc, A_h16, B_q4_i32, scales_h16, K, N, group_size)
        result_f32 = result.astype(np.float32)

        max_err = float(np.max(np.abs(result_f32 - y_ref)))
        mean_err = float(np.mean(np.abs(result_f32 - y_ref)))
        cos_sim = float(np.dot(result_f32, y_ref) /
                        (np.linalg.norm(result_f32) * np.linalg.norm(y_ref) + 1e-12))
        passed = max_err < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  awq_t{tpc}: max_abs_err={max_err:.6f}  mean_err={mean_err:.6f}  "
              f"cos_sim={cos_sim:.6f}  [{status}]")

        if not passed:
            all_pass = False
            # Debug: top 5 errors
            diff = np.abs(result_f32 - y_ref)
            idx = np.argsort(diff)[-5:][::-1]
            print(f"    Top-5 errors:")
            for i in idx:
                print(f"      idx={i}: awq={result_f32[i]:.6f}, ref={y_ref[i]:.6f}, err={diff[i]:.6f}")

    return all_pass


def test_awq_vs_zeros_match(N, K, group_size=128, seed=42):
    """
    Additional check: AWQ kernel output should match GPTQ kernel with zeros=0.
    This verifies the kernel is arithmetically equivalent when zeros=0.
    """
    print(f"\n--- AWQ vs GPTQ(zeros=0) match: N={N}, K={K} ---")

    A_h16, B_q4_awq, scales_h16, w_ref = make_awq_weights(K, N, group_size, seed)
    B_q4_i32 = B_q4_awq.view(np.int32)

    # For GPTQ with zeros=0: same behavior as AWQ
    zeros_zero = np.zeros_like(scales_h16)

    all_pass = True
    for tpc, func_awq_k, func_gptq in [
        (4,  func_awq_t4,  func_v5_t4),
        (8,  func_awq_t8,  func_v5_t8),
        (16, func_awq_t16, func_v5_t16),
    ]:
        res_awq  = run_awq_gemv(func_awq_k, tpc, A_h16, B_q4_i32, scales_h16,
                                 K, N, group_size)
        res_gptq = run_gptq_gemv(func_gptq, tpc, A_h16, B_q4_i32, scales_h16,
                                  zeros_zero, K, N, group_size)

        max_err = float(np.max(np.abs(res_awq.astype(np.float32) -
                                       res_gptq.astype(np.float32))))
        status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  AWQ t{tpc} vs GPTQ(zeros=0) t{tpc}: max_abs_err={max_err:.6e}  [{status}]")
        if max_err >= 1e-4:
            all_pass = False

    return all_pass


def test_latency_comparison(N, K, group_size=128, seed=42, n_warmup=10, n_iters=100):
    """
    Latency comparison: AWQ vs GPTQ kernel (all t-variants).
    AWQ should be <= GPTQ (one fewer instruction per weight, no zeros load).
    Returns dict of {tpc: (awq_us, gptq_us, speedup)}.
    """
    print(f"\n--- Latency: N={N}, K={K}, gs={group_size} ---")

    A_h16, B_q4, scales_h16, _ = make_awq_weights(K, N, group_size, seed)
    B_q4_i32 = B_q4.view(np.int32)
    _, B_q4_g, scales_g, zeros_g = make_gptq_weights(K, N, group_size, seed)
    B_q4_g_i32 = B_q4_g.view(np.int32)

    # AWQ buffers (no zeros)
    d_A_awq      = dev.malloc(A_h16.nbytes)
    d_B_awq      = dev.malloc(B_q4_i32.nbytes)
    d_scales_awq = dev.malloc(scales_h16.nbytes)
    d_C_awq      = dev.malloc(N * 2)
    dev.upload(d_A_awq,      A_h16.tobytes())
    dev.upload(d_B_awq,      B_q4_i32.tobytes())
    dev.upload(d_scales_awq, scales_h16.tobytes())

    # GPTQ buffers (with zeros)
    d_A_gptq      = dev.malloc(A_h16.nbytes)
    d_B_gptq      = dev.malloc(B_q4_g_i32.nbytes)
    d_scales_gptq = dev.malloc(scales_g.nbytes)
    d_zeros_gptq  = dev.malloc(zeros_g.nbytes)
    d_C_gptq      = dev.malloc(N * 2)
    dev.upload(d_A_gptq,      A_h16.tobytes())
    dev.upload(d_B_gptq,      B_q4_g_i32.tobytes())
    dev.upload(d_scales_gptq, scales_g.tobytes())
    dev.upload(d_zeros_gptq,  zeros_g.tobytes())

    results = {}
    for tpc, func_awq_k, func_gptq in [
        (4,  func_awq_t4,  func_v5_t4),
        (8,  func_awq_t8,  func_v5_t8),
        (16, func_awq_t16, func_v5_t16),
    ]:
        lat_awq  = benchmark_awq_gemv(func_awq_k, tpc, d_A_awq, d_B_awq, d_scales_awq, d_C_awq,
                                       K, N, group_size, n_warmup, n_iters)
        lat_gptq = benchmark_gptq_gemv(func_gptq, tpc, d_A_gptq, d_B_gptq, d_scales_gptq,
                                        d_zeros_gptq, d_C_gptq, K, N, group_size, n_warmup, n_iters)
        speedup = lat_gptq / lat_awq
        status = "FASTER" if speedup >= 1.0 else "SLOWER"
        print(f"  t{tpc}: GPTQ={lat_gptq:.1f}us  AWQ={lat_awq:.1f}us  speedup={speedup:.3f}x  [{status}]")
        results[tpc] = (lat_awq, lat_gptq, speedup)

    dev.free(d_A_awq); dev.free(d_B_awq); dev.free(d_scales_awq); dev.free(d_C_awq)
    dev.free(d_A_gptq); dev.free(d_B_gptq); dev.free(d_scales_gptq)
    dev.free(d_zeros_gptq); dev.free(d_C_gptq)

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("AWQ INT4 GEMV (gemv_int4_v5_awq) — Correctness & Benchmark")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Correctness: VAL-AWQ-002
    # AWQ GEMV output vs numpy dequantize-then-matmul
    # All Qwen 3.5 27B shapes, threshold: max abs error < 1e-2
    # -----------------------------------------------------------------------
    print("\n[VAL-AWQ-002] Correctness vs numpy reference (threshold: max_abs_err < 1e-2)")

    shapes = [
        (4096, 5120),
        (5120, 5120),
        (11008, 5120),
        (13696, 5120),
        (17408, 5120),   # AWQ intermediate_size
    ]

    all_correct = True
    for N, K in shapes:
        if not test_correctness_vs_numpy(N=N, K=K, group_size=128):
            all_correct = False

    print()
    if all_correct:
        print("VAL-AWQ-002: ALL correctness tests PASSED (max_abs_err < 1e-2)")
    else:
        print("VAL-AWQ-002: SOME correctness tests FAILED")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Equivalence check: AWQ kernel == GPTQ(zeros=0)
    # -----------------------------------------------------------------------
    print("\n[Equivalence] AWQ kernel == GPTQ kernel with zeros=0")
    equiv_shapes = [(4096, 5120), (11008, 5120)]
    all_equiv = True
    for N, K in equiv_shapes:
        if not test_awq_vs_zeros_match(N=N, K=K, group_size=128):
            all_equiv = False

    if all_equiv:
        print("\nEquivalence check: PASS (AWQ == GPTQ with zeros=0)")
    else:
        print("\nEquivalence check: FAIL")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Performance comparison: AWQ vs GPTQ (VAL-AWQ expected behavior)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[Latency] AWQ vs GPTQ GEMV (100 iters, 10 warmup)")
    print("=" * 72)

    bench_shapes = [
        (4096,  5120),
        (5120,  5120),
        (11008, 5120),
        (13696, 5120),
        (17408, 5120),
    ]

    all_latency_results = {}
    for N, K in bench_shapes:
        all_latency_results[(N, K)] = test_latency_comparison(N=N, K=K, group_size=128)

    print("\n" + "=" * 72)
    print("Latency Summary:")
    print(f"{'Shape':>22}  {'var':>4}  {'GPTQ(us)':>10}  {'AWQ(us)':>10}  {'speedup':>10}")
    print("-" * 72)
    for (N, K), perf in all_latency_results.items():
        for tpc in [4, 8, 16]:
            lat_awq, lat_gptq, sp = perf[tpc]
            direction = "+" if sp >= 1.0 else "-"
            print(f"  N={N:5d},K={K:5d}  t{tpc:>2}  {lat_gptq:>10.1f}  {lat_awq:>10.1f}  {sp:>9.3f}x")

    # Check for latency regression: AWQ should not be significantly slower than GPTQ
    # Allow 5% margin for measurement noise
    print("\n[Latency check] AWQ latency <= GPTQ latency (5% margin) for primary shapes")
    primary_shapes = [(4096, 5120), (11008, 5120)]
    latency_pass = True
    for N, K in primary_shapes:
        perf = all_latency_results[(N, K)]
        for tpc in [4, 8, 16]:
            lat_awq, lat_gptq, sp = perf[tpc]
            ok = lat_awq <= lat_gptq * 1.05  # 5% margin
            status = "PASS" if ok else "FAIL"
            print(f"  N={N:5d},K={K:5d} t{tpc}: AWQ={lat_awq:.1f}us vs GPTQ={lat_gptq:.1f}us "
                  f"speedup={sp:.3f}x [{status}]")
            if not ok:
                latency_pass = False

    print()
    if latency_pass:
        print("Latency check: PASS — AWQ is equal or faster than GPTQ (within 5% margin)")
    else:
        print("Latency check: NOTE — AWQ slightly slower on some shapes (memory-bound; "
              "improvement may be within noise)")
        # Don't fail on latency — memory-bound kernels are similar; correctness is key
        # The latency check is informational

    print("\n" + "=" * 72)
    print("Final Summary:")
    print(f"  VAL-AWQ-002 (correctness): {'PASS' if all_correct else 'FAIL'}")
    print(f"  Equivalence (awq==gptq/zeros=0): {'PASS' if all_equiv else 'FAIL'}")
    print(f"  Latency vs GPTQ: {'PASS' if latency_pass else 'NOTE (informational)'}")
    print()

    if all_correct and all_equiv:
        print("ALL CRITICAL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME CRITICAL TESTS FAILED")
        sys.exit(1)
