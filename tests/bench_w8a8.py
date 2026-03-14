#!/usr/bin/env python3
"""Benchmark W8A8 GEMV vs W4A16 GEMV (gemv_int4_v3_t16) on representative Qwen dimensions.

Measures:
  - W8A8 GEMV (gemv_w8a8.hip): INT8 weight + INT8 activation via v_dot4_i32_i8
  - W4A16 GEMV (gemv_int4_v3_t16): INT4 weight + FP16 activation via ubfe dequant

Test dimensions (Qwen 3.5 27B representative shapes):
  - N=4096, K=4096  (hidden→hidden projections: q/k/v/o projections)
  - N=11008, K=4096 (FFN gate/up projections: hidden→intermediate)

Comparison analysis:
  - W8A8 has 2x weight size vs W4A16 (8-bit vs 4-bit per weight)
  - W8A8 uses v_dot4_i32_i8 (4 INT8 MACs per instruction, ~7.58 instr/cycle)
  - W4A16 uses ubfe dequant + FP32 FMA (more instruction overhead per weight)
  - Both are memory-bandwidth-bound for decode (M=1)

Memory bandwidth analysis:
  - W8A8: reads N*K bytes (weights) + K bytes (activations) + N*4 (scales) → writes N*2 (FP16 out)
  - W4A16: reads N*K/2 bytes (weights) + K*2 bytes (FP16 acts) + N*K/group_size*4 (scales+zeros) → writes N*2

Expected outcome: W8A8 reads 2x more weight data, so typically SLOWER for memory-bound decode.
However, simpler compute path (no dequant FP16 conversion) may partially offset the bandwidth cost.
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco

dev = GPUDevice(0)
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build W8A8 kernel
# ---------------------------------------------------------------------------
print("Building gemv_w8a8.hip ...")
w8a8_hip    = str(PROJECT_ROOT / "src" / "kernels" / "gemv_w8a8.hip")
w8a8_hsaco  = str(BUILD_DIR / "gemv_w8a8.hsaco")
build_hip_hsaco(w8a8_hip, w8a8_hsaco)
w8a8_mod    = dev.load_hsaco(w8a8_hsaco)
w8a8_kernel = dev.get_kernel(w8a8_mod, "gemv_w8a8")
print("  gemv_w8a8 OK")

# ---------------------------------------------------------------------------
# Build W4A16 kernels (v3_t16 for N=4096, v3_t16 also for N=11008)
# ---------------------------------------------------------------------------
print("Building gemv_int4_v3.hip ...")
v3_hip    = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v3.hip")
v3_hsaco  = str(BUILD_DIR / "gemv_int4_v3.hsaco")
build_hip_hsaco(v3_hip, v3_hsaco)
v3_mod        = dev.load_hsaco(v3_hsaco)
v3_t16_kernel = dev.get_kernel(v3_mod, "gemv_int4_v3_t16")
v3_t4_kernel  = dev.get_kernel(v3_mod, "gemv_int4_v3_t4")
print("  gemv_int4_v3 OK")

print()

# ---------------------------------------------------------------------------
# Helper: pack INT4 weights for W4A16 kernel
# Input W_int4 is [N, K] with values 0..15 (unsigned nibbles)
# Kernel expects packed uint32: [K_groups, N] where each uint32 = 8 nibbles (K-major)
# ---------------------------------------------------------------------------
def pack_w4a16(W_int4, K, N):
    """Pack [N, K] INT4 weights to [K//8, N] uint32 for gemv_int4_v3."""
    # Reorder to K-major: [K, N]
    W_km = W_int4.T  # [K, N]
    assert W_km.shape == (K, N)
    assert K % 8 == 0, f"K={K} must be multiple of 8"
    num_k_groups = K // 8
    W_packed = np.zeros((num_k_groups, N), dtype=np.uint32)
    for kg in range(num_k_groups):
        k_base = kg * 8
        for bit in range(8):
            W_packed[kg] |= (W_km[k_base + bit, :].astype(np.uint32) & 0xF) << (bit * 4)
    return W_packed


# ---------------------------------------------------------------------------
# Benchmark function for W8A8
# ---------------------------------------------------------------------------
def bench_w8a8(N, K, iters=500, warmup=50):
    """Benchmark W8A8 GEMV: INT8 weights + INT8 activations → FP16 output."""
    np.random.seed(42)
    W_int8  = np.random.randint(-64, 64, (N, K), dtype=np.int8)
    x_int8  = np.random.randint(-64, 64, (K,),   dtype=np.int8)
    scale_w = np.ones(N, dtype=np.float32) * 0.01
    scale_a = np.float32(0.008)

    d_x   = dev.malloc(K)           # K INT8 bytes
    d_W   = dev.malloc(N * K)       # N*K INT8 bytes
    d_sw  = dev.malloc(N * 4)       # N FP32 bytes
    d_out = dev.malloc(N * 2)       # N FP16 bytes

    dev.upload(d_x,  x_int8.tobytes())
    dev.upload(d_W,  W_int8.tobytes())
    dev.upload(d_sw, scale_w.tobytes())

    grid_x = (N + 3) // 4
    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_W),
        ctypes.c_uint64(d_sw),
        ctypes.c_float(float(scale_a)),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
    ]

    # Warmup
    for _ in range(warmup):
        dev.launch(w8a8_kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(w8a8_kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    # Memory: read W[N,K] INT8 + read x[K] INT8 + read scale_w[N] FP32 + write out[N] FP16
    bytes_rw = N * K + K + N * 4 + N * 2
    bw_gbps = bytes_rw / 1e9 / (t_us * 1e-6)
    # Ops: 2*N*K multiply-accumulate (INT8 dot products via v_dot4_i32_i8)
    tops = 2.0 * N * K / 1e12 / (t_us * 1e-6)

    dev.free(d_x)
    dev.free(d_W)
    dev.free(d_sw)
    dev.free(d_out)

    return t_us, bw_gbps, tops


# ---------------------------------------------------------------------------
# Benchmark function for W4A16 (v3_t16 and v3_t4)
# ---------------------------------------------------------------------------
def bench_w4a16_v3(N, K, kernel, threads_per_col, group_size=128, iters=500, warmup=50):
    """Benchmark W4A16 GEMV: INT4 weights + FP16 activations → FP16 output."""
    np.random.seed(42)
    W_int4  = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    x_fp16  = np.random.randn(K).astype(np.float16)
    num_groups = K // group_size
    # scales/zeros: [num_groups, N] FP16 (matches v3 kernel layout)
    scales  = np.ones((num_groups, N), dtype=np.float16) * np.float16(0.01)
    zeros   = np.ones((num_groups, N), dtype=np.float16) * np.float16(8.0)

    W_packed = pack_w4a16(W_int4, K, N)  # [K//8, N] uint32

    d_x  = dev.malloc(K * 2)                   # K FP16 bytes
    d_W  = dev.malloc(W_packed.nbytes)          # [K//8, N] uint32
    d_sc = dev.malloc(scales.nbytes)            # [num_groups, N] FP16
    d_zr = dev.malloc(zeros.nbytes)             # [num_groups, N] FP16
    d_out = dev.malloc(N * 2)                   # N FP16

    dev.upload(d_x,  x_fp16.tobytes())
    dev.upload(d_W,  W_packed.tobytes())
    dev.upload(d_sc, scales.tobytes())
    dev.upload(d_zr, zeros.tobytes())

    cols_per_wg = 256 // threads_per_col
    grid_x = (N + cols_per_wg - 1) // cols_per_wg
    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_W),
        ctypes.c_uint64(d_sc),
        ctypes.c_uint64(d_zr),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]

    # Warmup
    for _ in range(warmup):
        dev.launch(kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    # Memory: read W[K//8,N] uint32 + x[K] FP16 + scales[num_groups,N] FP16 + zeros + write out[N] FP16
    bytes_w4    = (K // 2) * N           # K*N/2 bytes for INT4 weights (packed)
    bytes_x     = K * 2                  # K FP16
    bytes_scales = num_groups * N * 2    # scales FP16
    bytes_zeros  = num_groups * N * 2    # zeros FP16
    bytes_out    = N * 2                 # output FP16
    bytes_rw = bytes_w4 + bytes_x + bytes_scales + bytes_zeros + bytes_out
    bw_gbps = bytes_rw / 1e9 / (t_us * 1e-6)
    # Ops: 2*N*K (same arithmetic as W8A8, just in FP32 via dequant)
    tops = 2.0 * N * K / 1e12 / (t_us * 1e-6)

    dev.free(d_x)
    dev.free(d_W)
    dev.free(d_sc)
    dev.free(d_zr)
    dev.free(d_out)

    return t_us, bw_gbps, tops


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
BENCH_SHAPES = [
    (4096,  4096, "N=4096,  K=4096  (hidden→hidden, e.g. q/k/v proj)"),
    (11008, 4096, "N=11008, K=4096  (FFN gate/up: hidden→intermediate)"),
]

GROUP_SIZE = 128  # Standard GPTQ group size

print("=" * 70)
print("W8A8 GEMV vs W4A16 GEMV Benchmark")
print("Hardware: gfx906 (MI50/MI60)")
print("=" * 70)
print()
print("W8A8:  INT8 weights (N*K bytes) + INT8 activations, v_dot4_i32_i8")
print("W4A16: INT4 weights (N*K/2 bytes) + FP16 activations, ubfe dequant")
print(f"       group_size={GROUP_SIZE} (per-group scale+zero in FP16)")
print()
print(f"{'Shape':<50} {'Kernel':<20} {'Latency':>10} {'BW':>10} {'TOPS':>10}")
print("-" * 105)

results = {}

for N, K, label in BENCH_SHAPES:
    # W8A8 benchmark
    t_w8a8, bw_w8a8, tops_w8a8 = bench_w8a8(N, K)
    print(f"{label:<50} {'W8A8 (v_dot4)':<20} {t_w8a8:>8.2f} us  {bw_w8a8:>7.1f} GB/s  {tops_w8a8*1e3:>8.3f} TOPS")

    # W4A16 v3_t16 benchmark (16 threads/col = best for these shapes per architecture.md)
    t_v3t16, bw_v3t16, tops_v3t16 = bench_w4a16_v3(N, K, v3_t16_kernel, threads_per_col=16)
    print(f"{'':<50} {'W4A16 (v3_t16)':<20} {t_v3t16:>8.2f} us  {bw_v3t16:>7.1f} GB/s  {tops_v3t16*1e3:>8.3f} TOPS")

    # W4A16 v3_t4 benchmark (4 threads/col = default for N>=4096 per architecture.md)
    t_v3t4, bw_v3t4, tops_v3t4 = bench_w4a16_v3(N, K, v3_t4_kernel, threads_per_col=4)
    print(f"{'':<50} {'W4A16 (v3_t4)':<20} {t_v3t4:>8.2f} us  {bw_v3t4:>7.1f} GB/s  {tops_v3t4*1e3:>8.3f} TOPS")

    # Best W4A16 variant
    best_w4a16_t = min(t_v3t16, t_v3t4)
    best_w4a16_label = "v3_t16" if t_v3t16 < t_v3t4 else "v3_t4"

    ratio = t_w8a8 / best_w4a16_t
    results[(N, K)] = {
        "w8a8_us": t_w8a8,
        "w4a16_us": best_w4a16_t,
        "w4a16_best": best_w4a16_label,
        "ratio": ratio,
        "w8a8_bw": bw_w8a8,
        "w4a16_bw": best_w4a16_t,
    }

    if ratio > 1.0:
        verdict = f"W4A16 ({best_w4a16_label}) is {ratio:.2f}x FASTER (W8A8 slower due to 2x weight bandwidth)"
    else:
        verdict = f"W8A8 is {1.0/ratio:.2f}x FASTER (v_dot4 arithmetic intensity wins)"
    print(f"{'':<50} {'Comparison':<20} → {verdict}")
    print()

# ---------------------------------------------------------------------------
# Summary analysis
# ---------------------------------------------------------------------------
print("=" * 70)
print("SUMMARY: Memory Bandwidth Analysis")
print("=" * 70)
print()
print("MI50/MI60 peak HBM bandwidth: ~860 GB/s (vectorized)")
print()
for N, K, label in BENCH_SHAPES:
    r = results[(N, K)]

    # Theoretical minimum times (bandwidth-bound)
    bytes_w8a8_weights   = N * K          # INT8: 1 byte/weight
    bytes_w4a16_weights  = (N * K) // 2   # INT4: 0.5 bytes/weight
    peak_bw = 860.0  # GB/s

    theory_w8a8_us  = bytes_w8a8_weights  / (peak_bw * 1e9) * 1e6
    theory_w4a16_us = bytes_w4a16_weights / (peak_bw * 1e9) * 1e6

    print(f"Shape N={N}, K={K}:")
    print(f"  W8A8:  weight bytes = {bytes_w8a8_weights/1e6:.2f} MB  →  theory min = {theory_w8a8_us:.2f} us  →  measured = {r['w8a8_us']:.2f} us")
    print(f"  W4A16: weight bytes = {bytes_w4a16_weights/1e6:.2f} MB  →  theory min = {theory_w4a16_us:.2f} us  →  measured = {r['w4a16_us']:.2f} us (best: {r['w4a16_best']})")
    print(f"  Ratio (W8A8/W4A16): {r['ratio']:.2f}x  ({'W4A16 faster' if r['ratio'] > 1.0 else 'W8A8 faster'})")
    print()

print("CONCLUSION:")
print("  For decode (M=1), both kernels are memory-bandwidth-bound.")
print("  W8A8 reads 2x the weight data vs W4A16, so W4A16 is expected to be")
print("  ~2x faster in the bandwidth-bound regime.")
print("  v_dot4_i32_i8 (W8A8) provides higher arithmetic intensity per byte")
print("  but the bandwidth penalty dominates for decode workloads.")
print("  W8A8 becomes competitive for prefill (M>1) where compute density increases.")
print()

dev.cleanup()
print("Benchmark complete.")
