#!/usr/bin/env python3
"""
Test harness for fused skip-connection + RMSNorm + INT4 GEMV kernel.

Fuses the decode path:
  1. hidden += residual        (skip connection)
  2. norm_out = rmsnorm(hidden) (RMSNorm with weight)
  3. out = norm_out @ W_int4^T  (INT4 GEMV)

Into a single kernel launch, eliminating intermediate HBM write/read of norm_out.

Tests:
1. Correctness: compare fused output vs separate skip_rmsnorm_v2 + gemv_int4_v3_t16
   at dim=5120, N=4096, gs=128
2. Correctness: also test at dim=5120, N=11008, gs=128 (FFN shape)
3. Benchmark: fused vs separate (2 launches) at same dimensions
4. Reports max abs error and latency comparison

Validation contract: VAL-FUSE-001
  - Max abs error vs separate kernel results < 1e-2
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

# ---- Kernel paths ----
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

HIP_DIR = PROJECT_ROOT / "src" / "kernels"

print("Building elementwise_v2 (skip_rmsnorm_v2, rmsnorm_v2)...")
hip_elem = str(HIP_DIR / "elementwise_v2.hip")
hsaco_elem = str(BUILD_DIR / "elementwise_v2.hsaco")
build_hip_hsaco(hip_elem, hsaco_elem)

print("Building gemv_int4_v3 (gemv_int4_v3_t16)...")
hip_v3 = str(HIP_DIR / "gemv_int4_v3.hip")
hsaco_v3 = str(BUILD_DIR / "gemv_int4_v3.hsaco")
build_hip_hsaco(hip_v3, hsaco_v3)

print("Building fused_skip_rmsnorm_gemv...")
hip_fused = str(HIP_DIR / "fused_skip_rmsnorm_gemv.hip")
hsaco_fused = str(BUILD_DIR / "fused_skip_rmsnorm_gemv.hsaco")
build_hip_hsaco(hip_fused, hsaco_fused)

print("All kernels compiled successfully.\n")

dev = GPUDevice(0)

module_elem  = dev.load_hsaco(hsaco_elem)
func_skip_rms = dev.get_kernel(module_elem, "skip_rmsnorm_v2")
func_rmsnorm  = dev.get_kernel(module_elem, "rmsnorm_v2")

module_v3     = dev.load_hsaco(hsaco_v3)
func_v3_t4    = dev.get_kernel(module_v3, "gemv_int4_v3_t4")
func_v3_t8    = dev.get_kernel(module_v3, "gemv_int4_v3_t8")
func_v3_t16   = dev.get_kernel(module_v3, "gemv_int4_v3_t16")

module_fused  = dev.load_hsaco(hsaco_fused)
func_fused_t4  = dev.get_kernel(module_fused, "fused_skip_rmsnorm_gemv_t4")
func_fused_t8  = dev.get_kernel(module_fused, "fused_skip_rmsnorm_gemv_t8")
func_fused_t16 = dev.get_kernel(module_fused, "fused_skip_rmsnorm_gemv_t16")

print("All kernels loaded.\n")


# ---------------------------------------------------------------------------
# Helper: quantize weights in GPTQ format (same as other test files)
# ---------------------------------------------------------------------------
def quantize_weights_gptq(W_fp32, group_size=128):
    """
    Simulate GPTQ quantization (unsigned INT4, 0-15 range).

    W_fp32: [K, N] float32 weight matrix (K=dim=activation dim)
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

    # Pack: B_q4[K/8, N] uint32 — bits[4*i+3:4*i] = nibble i
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_mat[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales.astype(np.float16), zeros.astype(np.float16)


def dequant_reference(B_q4, scales, zeros, K, N, group_size):
    """CPU reference dequantization. Returns [K, N] FP32."""
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


# ---------------------------------------------------------------------------
# Run separate kernels: skip_rmsnorm_v2 + gemv_int4_v3_t16
# ---------------------------------------------------------------------------
def run_separate(hidden_np, residual_np, weight_np, B_q4, scales, zeros,
                 K, N, group_size=128, eps=1e-5):
    """
    Reference path:
      1. skip_rmsnorm_v2(norm_out, hidden, residual, weight, K, eps)
      2. gemv_int4_v3_t16(norm_out, B_q4, scales, zeros, out, K, N, gs)
    Returns: (out [N] FP16, hidden_updated [K] FP16)
    """
    h_bytes = hidden_np.astype(np.float16).tobytes()
    r_bytes = residual_np.astype(np.float16).tobytes()
    w_bytes = weight_np.astype(np.float16).tobytes()

    d_hidden   = dev.malloc(K * 2)
    d_residual = dev.malloc(K * 2)
    d_weight   = dev.malloc(K * 2)
    d_norm_out = dev.malloc(K * 2)
    d_B_q4     = dev.malloc(B_q4.nbytes)
    d_scales   = dev.malloc(scales.nbytes)
    d_zeros    = dev.malloc(zeros.nbytes)
    d_out      = dev.malloc(N * 2)

    dev.upload(d_hidden,   h_bytes)
    dev.upload(d_residual, r_bytes)
    dev.upload(d_weight,   w_bytes)
    dev.upload(d_B_q4,     B_q4.tobytes())
    dev.upload(d_scales,   scales.tobytes())
    dev.upload(d_zeros,    zeros.tobytes())

    # Step 1: skip_rmsnorm_v2
    params_norm = [
        ctypes.c_uint64(d_norm_out),
        ctypes.c_uint64(d_hidden),
        ctypes.c_uint64(d_residual),
        ctypes.c_uint64(d_weight),
        ctypes.c_uint32(K),
        ctypes.c_float(eps),
    ]
    dev.launch(func_skip_rms, (1, 1, 1), (256, 1, 1), params_norm)

    # Step 2: gemv_int4_v3_t16
    COLS_PER_WG = 16   # = 256 / 16 (THREADS_PER_COL=16)
    grid_x = (N + COLS_PER_WG - 1) // COLS_PER_WG
    params_gemv = [
        ctypes.c_uint64(d_norm_out),
        ctypes.c_uint64(d_B_q4),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]
    dev.launch(func_v3_t16, (grid_x, 1, 1), (256, 1, 1), params_gemv)
    dev.synchronize()

    out     = np.frombuffer(dev.download(d_out,    N * 2), dtype=np.float16).copy()
    h_upd   = np.frombuffer(dev.download(d_hidden, K * 2), dtype=np.float16).copy()

    dev.free(d_hidden); dev.free(d_residual); dev.free(d_weight)
    dev.free(d_norm_out); dev.free(d_B_q4); dev.free(d_scales)
    dev.free(d_zeros);  dev.free(d_out)

    return out, h_upd


# ---------------------------------------------------------------------------
# Run fused kernel
# ---------------------------------------------------------------------------
def run_fused(func_fused, threads_per_col,
              hidden_np, residual_np, weight_np, B_q4, scales, zeros,
              K, N, group_size=128, eps=1e-5):
    """
    Fused kernel: fused_skip_rmsnorm_gemv_t{threads_per_col}
    Uses separate hidden_out buffer to avoid race conditions.
    Returns: (out [N] FP16, hidden_updated [K] FP16)
    """
    h_bytes = hidden_np.astype(np.float16).tobytes()
    r_bytes = residual_np.astype(np.float16).tobytes()
    w_bytes = weight_np.astype(np.float16).tobytes()

    d_hidden     = dev.malloc(K * 2)   # input hidden (read-only)
    d_hidden_out = dev.malloc(K * 2)   # output hidden (h+r, written by block 0)
    d_residual   = dev.malloc(K * 2)
    d_weight     = dev.malloc(K * 2)
    d_B_q4       = dev.malloc(B_q4.nbytes)
    d_scales     = dev.malloc(scales.nbytes)
    d_zeros      = dev.malloc(zeros.nbytes)
    d_out        = dev.malloc(N * 2)

    dev.upload(d_hidden,   h_bytes)
    dev.upload(d_residual, r_bytes)
    dev.upload(d_weight,   w_bytes)
    dev.upload(d_B_q4,     B_q4.tobytes())
    dev.upload(d_scales,   scales.tobytes())
    dev.upload(d_zeros,    zeros.tobytes())

    COLS_PER_WG = 256 // threads_per_col
    grid_x = (N + COLS_PER_WG - 1) // COLS_PER_WG

    # Dynamic shared memory: K*2 (lds_hval) + K*2 (lds_A) + 16 (s_warp) + 256*4 (s_reduce)
    shared_mem_bytes = K * 4 + 16 + 256 * 4

    # New signature: out_gemv, hidden_out, hidden (const), residual, weight, eps, ...
    params = [
        ctypes.c_uint64(d_out),
        ctypes.c_uint64(d_hidden_out),
        ctypes.c_uint64(d_hidden),
        ctypes.c_uint64(d_residual),
        ctypes.c_uint64(d_weight),
        ctypes.c_float(eps),
        ctypes.c_uint64(d_B_q4),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]
    dev.launch(func_fused, (grid_x, 1, 1), (256, 1, 1), params, shared_mem=shared_mem_bytes)
    dev.synchronize()

    out   = np.frombuffer(dev.download(d_out,        N * 2), dtype=np.float16).copy()
    h_upd = np.frombuffer(dev.download(d_hidden_out, K * 2), dtype=np.float16).copy()

    dev.free(d_hidden);     dev.free(d_hidden_out);  dev.free(d_residual)
    dev.free(d_weight);     dev.free(d_B_q4);        dev.free(d_scales)
    dev.free(d_zeros);      dev.free(d_out)

    return out, h_upd


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------
def reference_cpu(hidden_np, residual_np, weight_np, W_deq, K, N, eps=1e-5):
    """
    CPU reference:
      hidden = hidden + residual
      rms = sqrt(mean(hidden^2) + eps)
      norm_out = (hidden / rms) * weight
      out = norm_out @ W_deq   (W_deq is [K, N] FP32)
    """
    hidden_f32 = hidden_np.astype(np.float32)
    residual_f32 = residual_np.astype(np.float32)
    weight_f32 = weight_np.astype(np.float32)

    h = hidden_f32 + residual_f32
    rms = np.sqrt(np.mean(h ** 2) + eps)
    norm_out = (h / rms) * weight_f32

    out = (norm_out @ W_deq).astype(np.float16)
    return out, h.astype(np.float16)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def bench_separate(hidden_np, residual_np, weight_np, B_q4, scales, zeros,
                   K, N, group_size=128, eps=1e-5, n_warmup=10, n_iters=100):
    """Benchmark separate skip_rmsnorm_v2 + gemv_int4_v3_t16 (2 launches)."""
    h_bytes = hidden_np.astype(np.float16).tobytes()
    r_bytes = residual_np.astype(np.float16).tobytes()
    w_bytes = weight_np.astype(np.float16).tobytes()

    d_hidden   = dev.malloc(K * 2)
    d_residual = dev.malloc(K * 2)
    d_weight   = dev.malloc(K * 2)
    d_norm_out = dev.malloc(K * 2)
    d_B_q4     = dev.malloc(B_q4.nbytes)
    d_scales   = dev.malloc(scales.nbytes)
    d_zeros    = dev.malloc(zeros.nbytes)
    d_out      = dev.malloc(N * 2)

    dev.upload(d_residual, r_bytes)
    dev.upload(d_weight,   w_bytes)
    dev.upload(d_B_q4,     B_q4.tobytes())
    dev.upload(d_scales,   scales.tobytes())
    dev.upload(d_zeros,    zeros.tobytes())

    COLS_PER_WG = 16
    grid_x = (N + COLS_PER_WG - 1) // COLS_PER_WG

    params_norm = [
        ctypes.c_uint64(d_norm_out),
        ctypes.c_uint64(d_hidden),
        ctypes.c_uint64(d_residual),
        ctypes.c_uint64(d_weight),
        ctypes.c_uint32(K),
        ctypes.c_float(eps),
    ]
    params_gemv = [
        ctypes.c_uint64(d_norm_out),
        ctypes.c_uint64(d_B_q4),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]

    # Warmup
    for _ in range(n_warmup):
        dev.upload(d_hidden, h_bytes)
        dev.launch(func_skip_rms, (1, 1, 1), (256, 1, 1), params_norm)
        dev.launch(func_v3_t16, (grid_x, 1, 1), (256, 1, 1), params_gemv)
    dev.synchronize()

    # Timed
    times = []
    for _ in range(n_iters):
        dev.upload(d_hidden, h_bytes)  # reset hidden each iteration
        dev.synchronize()
        t0 = time.perf_counter()
        dev.launch(func_skip_rms, (1, 1, 1), (256, 1, 1), params_norm)
        dev.launch(func_v3_t16, (grid_x, 1, 1), (256, 1, 1), params_gemv)
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    dev.free(d_hidden); dev.free(d_residual); dev.free(d_weight)
    dev.free(d_norm_out); dev.free(d_B_q4); dev.free(d_scales)
    dev.free(d_zeros);  dev.free(d_out)

    return float(np.median(times))


def bench_fused(func_fused, threads_per_col,
                hidden_np, residual_np, weight_np, B_q4, scales, zeros,
                K, N, group_size=128, eps=1e-5, n_warmup=10, n_iters=100):
    """Benchmark fused kernel."""
    h_bytes = hidden_np.astype(np.float16).tobytes()
    r_bytes = residual_np.astype(np.float16).tobytes()
    w_bytes = weight_np.astype(np.float16).tobytes()

    d_hidden     = dev.malloc(K * 2)
    d_hidden_out = dev.malloc(K * 2)
    d_residual   = dev.malloc(K * 2)
    d_weight     = dev.malloc(K * 2)
    d_B_q4       = dev.malloc(B_q4.nbytes)
    d_scales     = dev.malloc(scales.nbytes)
    d_zeros      = dev.malloc(zeros.nbytes)
    d_out        = dev.malloc(N * 2)

    dev.upload(d_residual, r_bytes)
    dev.upload(d_weight,   w_bytes)
    dev.upload(d_B_q4,     B_q4.tobytes())
    dev.upload(d_scales,   scales.tobytes())
    dev.upload(d_zeros,    zeros.tobytes())

    COLS_PER_WG = 256 // threads_per_col
    grid_x = (N + COLS_PER_WG - 1) // COLS_PER_WG
    shared_mem_bytes = K * 4 + 16 + 256 * 4

    # New signature: out_gemv, hidden_out, hidden (const), residual, weight, eps, ...
    params = [
        ctypes.c_uint64(d_out),
        ctypes.c_uint64(d_hidden_out),
        ctypes.c_uint64(d_hidden),
        ctypes.c_uint64(d_residual),
        ctypes.c_uint64(d_weight),
        ctypes.c_float(eps),
        ctypes.c_uint64(d_B_q4),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]

    # Warmup
    for _ in range(n_warmup):
        dev.upload(d_hidden, h_bytes)
        dev.launch(func_fused, (grid_x, 1, 1), (256, 1, 1), params, shared_mem=shared_mem_bytes)
    dev.synchronize()

    # Timed
    times = []
    for _ in range(n_iters):
        dev.upload(d_hidden, h_bytes)  # reset hidden each iteration (input only)
        dev.synchronize()
        t0 = time.perf_counter()
        dev.launch(func_fused, (grid_x, 1, 1), (256, 1, 1), params, shared_mem=shared_mem_bytes)
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    dev.free(d_hidden);     dev.free(d_hidden_out);  dev.free(d_residual)
    dev.free(d_weight);     dev.free(d_B_q4);        dev.free(d_scales)
    dev.free(d_zeros);      dev.free(d_out)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_correctness(K, N, group_size=128, eps=1e-5, seed=42, label=""):
    """
    Compare fused kernel output vs separate skip_rmsnorm_v2 + gemv_int4_v3_t16.
    Returns True if all fused variants pass (max abs error < 1e-2).
    """
    tag = label or f"K={K}, N={N}, gs={group_size}"
    print(f"\n--- Correctness: {tag} ---")

    rng = np.random.default_rng(seed)
    hidden   = (rng.standard_normal(K) * 0.5).astype(np.float16)
    residual = (rng.standard_normal(K) * 0.5).astype(np.float16)
    weight   = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float16)
    W_fp32   = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)
    W_deq = dequant_reference(B_q4, scales, zeros, K, N, group_size)

    # CPU reference
    ref_out, ref_hidden = reference_cpu(hidden, residual, weight, W_deq, K, N, eps)

    # Separate GPU path (reference baseline)
    sep_out, sep_hidden = run_separate(hidden, residual, weight, B_q4, scales, zeros,
                                       K, N, group_size, eps)

    sep_err = float(np.abs(sep_out.astype(np.float32) - ref_out.astype(np.float32)).max())
    print(f"  separate (v3_t16) vs CPU ref: max_abs_err={sep_err:.4e} "
          f"[{'PASS' if sep_err < 1e-2 else 'FAIL'}]")

    threshold = 1e-2
    all_pass = True

    for tpc, func_fused in [(4, func_fused_t4), (8, func_fused_t8), (16, func_fused_t16)]:
        fused_out, fused_hidden = run_fused(
            func_fused, tpc, hidden, residual, weight, B_q4, scales, zeros,
            K, N, group_size, eps)

        # Compare fused vs separate (primary correctness check)
        err_vs_sep = float(np.abs(
            fused_out.astype(np.float32) - sep_out.astype(np.float32)).max())
        # Also compare vs CPU ref
        err_vs_ref = float(np.abs(
            fused_out.astype(np.float32) - ref_out.astype(np.float32)).max())

        # Check hidden update
        err_hidden = float(np.abs(
            fused_hidden.astype(np.float32) - sep_hidden.astype(np.float32)).max())

        passed = err_vs_sep < threshold
        status = "PASS" if passed else "FAIL"
        print(f"  fused_t{tpc}: err_vs_sep={err_vs_sep:.4e} err_vs_ref={err_vs_ref:.4e} "
              f"hidden_err={err_hidden:.4e} [{status}]")

        if not passed:
            all_pass = False
            # Print some debug
            print(f"    ref_out[:4]   = {ref_out[:4]}")
            print(f"    sep_out[:4]   = {sep_out[:4]}")
            print(f"    fused_out[:4] = {fused_out[:4]}")

    return all_pass


def test_performance(K, N, group_size=128, eps=1e-5, seed=42, label=""):
    """
    Benchmark fused vs separate at given (K, N).
    Returns (lat_separate, lat_fused_t16, speedup).
    """
    tag = label or f"K={K}, N={N}, gs={group_size}"
    print(f"\n--- Performance: {tag} ---")

    rng = np.random.default_rng(seed)
    hidden   = (rng.standard_normal(K) * 0.5).astype(np.float16)
    residual = (rng.standard_normal(K) * 0.5).astype(np.float16)
    weight   = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float16)
    W_fp32   = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)

    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)

    lat_sep = bench_separate(hidden, residual, weight, B_q4, scales, zeros,
                             K, N, group_size, eps)

    lat_t4  = bench_fused(func_fused_t4,  4, hidden, residual, weight, B_q4, scales, zeros,
                          K, N, group_size, eps)
    lat_t8  = bench_fused(func_fused_t8,  8, hidden, residual, weight, B_q4, scales, zeros,
                          K, N, group_size, eps)
    lat_t16 = bench_fused(func_fused_t16, 16, hidden, residual, weight, B_q4, scales, zeros,
                          K, N, group_size, eps)

    speedup_t16 = lat_sep / lat_t16

    print(f"  separate (skip_norm + gemv_t16):  {lat_sep:.1f} us  (2 kernel launches)")
    print(f"  fused_t4  (1 launch):              {lat_t4:.1f} us  ({lat_sep/lat_t4:.3f}x)")
    print(f"  fused_t8  (1 launch):              {lat_t8:.1f} us  ({lat_sep/lat_t8:.3f}x)")
    print(f"  fused_t16 (1 launch):              {lat_t16:.1f} us  ({speedup_t16:.3f}x)")

    # Theoretical min: weight bandwidth dominated
    weight_bytes = K * N // 2    # INT4 = K*N/2 bytes
    act_bytes    = K * 2 * 2     # read hidden + residual (FP16 each)
    hidden_write = K * 2         # write updated hidden
    weight_norm  = K * 2         # read norm weight
    out_bytes    = N * 2         # write output
    total_bytes  = weight_bytes + act_bytes + hidden_write + weight_norm + out_bytes
    theoretical_us = total_bytes / 857e9 * 1e6  # ~857 GB/s MI60

    print(f"  theoretical min ({total_bytes/1e6:.1f}MB @ 857GB/s): {theoretical_us:.1f} us")

    if speedup_t16 >= 1.0:
        print(f"  Result: fused_t16 is {speedup_t16:.3f}x FASTER (saves 2 kernel launch overheads + norm_out HBM R/W)")
    else:
        print(f"  Result: fused_t16 is {1/speedup_t16:.3f}x SLOWER "
              f"(LDS staging cost may exceed HBM savings at this shape)")

    return lat_sep, lat_t16, speedup_t16


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("test_fused_norm_gemv.py")
    print("Fused Skip-Connection + RMSNorm + INT4 GEMV")
    print("=" * 70)

    all_pass = True

    # VAL-FUSE-001: dim=5120, N=4096, gs=128
    print("\n" + "=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)

    if not test_correctness(K=5120, N=4096, group_size=128,
                            label="dim=5120, N=4096 (gate/up projection)"):
        all_pass = False

    # Additional test shape: N=11008 (FFN intermediate)
    if not test_correctness(K=5120, N=11008, group_size=128,
                            label="dim=5120, N=11008 (FFN width)"):
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("All correctness tests PASSED ✓")
    else:
        print("Some correctness tests FAILED ✗")
        sys.exit(1)

    # Performance benchmarks
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)

    lat_sep_4096, lat_fused_4096, sp_4096 = test_performance(
        K=5120, N=4096, group_size=128,
        label="dim=5120, N=4096")

    lat_sep_11008, lat_fused_11008, sp_11008 = test_performance(
        K=5120, N=11008, group_size=128,
        label="dim=5120, N=11008")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  N=4096:  separate={lat_sep_4096:.1f}us, fused_t16={lat_fused_4096:.1f}us, speedup={sp_4096:.3f}x")
    print(f"  N=11008: separate={lat_sep_11008:.1f}us, fused_t16={lat_fused_11008:.1f}us, speedup={sp_11008:.3f}x")

    if sp_4096 >= 1.0 or sp_11008 >= 1.0:
        print("\nLatency comparison: at least one shape shows improvement ✓")
    else:
        print("\nLatency comparison: fused was slower on both shapes (LDS overhead)")
        print("  Note: correctness still passes; latency depends on occupancy and workload size.")

    print("=" * 70)
    print("DONE")
    sys.exit(0)
