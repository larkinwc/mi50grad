#!/usr/bin/env python3
"""Benchmark W4A8 GEMV vs W4A16 GEMV on representative Qwen dimensions.

Compares:
  - W4A8 (gemv_w4a8_dot4): INT4 weights (K/2 bytes) + INT8 activations, v_dot4_i32_i8
  - W4A8 (gemv_w4a8_dot8): INT4 weights (K/2 bytes) + INT8 activations, v_dot8_i32_i4
  - W4A16 (gemv_int4_v3_t4): INT4 weights + FP16 activations (existing baseline)

Test dimensions (Qwen 3.5 27B representative shapes):
  - N=4096, K=4096  (hidden→hidden projections)
  - N=11008, K=4096 (FFN gate/up: hidden→intermediate)

Memory bandwidth analysis:
  - W4A8:  reads K*N/2 bytes (INT4 weights) + K bytes (INT8 acts) → vs W8A8: 2x less weight bandwidth
  - W4A16: reads K*N/2 bytes (INT4 weights) + K*2 bytes (FP16 acts)
  - W4A8 and W4A16 have same weight bandwidth! W4A8 uses INT8 acts (1 byte) vs FP16 (2 bytes)
  - W4A8 reads K/2 fewer activation bytes vs W4A16 → slight bandwidth advantage
  - W4A8 advantage: INT32 accumulation via v_dot4 (faster than FP32 FMA in W4A16)
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
from src.kernels.repack_w4a8 import repack_simple_for_test

dev = GPUDevice(0)
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build W4A8 kernel
# ---------------------------------------------------------------------------
print("Building gemv_w4a8.hip ...")
w4a8_hip    = str(PROJECT_ROOT / "src" / "kernels" / "gemv_w4a8.hip")
w4a8_hsaco  = str(BUILD_DIR / "gemv_w4a8.hsaco")
build_hip_hsaco(w4a8_hip, w4a8_hsaco)
w4a8_mod     = dev.load_hsaco(w4a8_hsaco)
kernel_dot4  = dev.get_kernel(w4a8_mod, "gemv_w4a8_dot4")
kernel_dot8  = dev.get_kernel(w4a8_mod, "gemv_w4a8_dot8")
print("  gemv_w4a8 OK (dot4 + dot8)")

# ---------------------------------------------------------------------------
# Build W4A16 kernel (v3_t4 — default for large N)
# ---------------------------------------------------------------------------
print("Building gemv_int4_v3.hip ...")
v3_hip    = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v3.hip")
v3_hsaco  = str(BUILD_DIR / "gemv_int4_v3.hsaco")
build_hip_hsaco(v3_hip, v3_hsaco)
v3_mod    = dev.load_hsaco(v3_hsaco)
v3_t4_kernel  = dev.get_kernel(v3_mod, "gemv_int4_v3_t4")
v3_t16_kernel = dev.get_kernel(v3_mod, "gemv_int4_v3_t16")
print("  gemv_int4_v3 OK (t4 + t16)")

print()

# ---------------------------------------------------------------------------
# Helper: pack INT4 weights for W4A16 kernel (K-major)
# Input W_int4 is [N, K] with values 0..15 (unsigned nibbles)
# Kernel expects packed uint32: [K//8, N] where each uint32 = 8 nibbles (K-major)
# ---------------------------------------------------------------------------
def pack_w4a16(W_int4, K, N):
    """Pack [N, K] INT4 weights to [K//8, N] uint32 for gemv_int4_v3."""
    W_km = W_int4.T  # [K, N]
    assert K % 8 == 0
    K_groups = K // 8
    W_packed = np.zeros((K_groups, N), dtype=np.uint32)
    for b in range(8):
        W_packed |= (W_km[b::8, :].astype(np.uint32) & 0xF) << (b * 4)
    return W_packed


# ---------------------------------------------------------------------------
# Benchmark W4A8 dot4
# ---------------------------------------------------------------------------
def bench_w4a8_dot4(N, K, iters=500, warmup=50):
    np.random.seed(42)
    W_fp32  = np.random.randn(N, K).astype(np.float32) * 0.1
    W_packed, scale_w = repack_simple_for_test(W_fp32)
    x_int8   = np.random.randint(-64, 64, (K,), dtype=np.int8)
    scale_a  = np.float32(0.008)

    d_x  = dev.malloc(K)
    d_W  = dev.malloc(N * (K // 8) * 4)
    d_sw = dev.malloc(N * 4)
    d_out = dev.malloc(N * 2)

    dev.upload(d_x,  x_int8.tobytes())
    dev.upload(d_W,  W_packed.tobytes())
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

    for _ in range(warmup):
        dev.launch(kernel_dot4, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(kernel_dot4, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    dev.free(d_x); dev.free(d_W); dev.free(d_sw); dev.free(d_out)

    bytes_rw = (N * K // 2) + K + N * 4 + N * 2
    bw_gbps  = bytes_rw / 1e9 / (t_us * 1e-6)
    tops     = 2.0 * N * K / 1e12 / (t_us * 1e-6)
    return t_us, bw_gbps, tops


# ---------------------------------------------------------------------------
# Benchmark W4A8 dot8
# ---------------------------------------------------------------------------
def bench_w4a8_dot8(N, K, iters=500, warmup=50):
    np.random.seed(42)
    W_fp32  = np.random.randn(N, K).astype(np.float32) * 0.1
    W_packed, scale_w = repack_simple_for_test(W_fp32)
    x_int8   = np.random.randint(-64, 64, (K,), dtype=np.int8)
    scale_a  = np.float32(0.008)

    d_x  = dev.malloc(K)
    d_W  = dev.malloc(N * (K // 8) * 4)
    d_sw = dev.malloc(N * 4)
    d_out = dev.malloc(N * 2)

    dev.upload(d_x,  x_int8.tobytes())
    dev.upload(d_W,  W_packed.tobytes())
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

    for _ in range(warmup):
        dev.launch(kernel_dot8, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(kernel_dot8, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    dev.free(d_x); dev.free(d_W); dev.free(d_sw); dev.free(d_out)

    bytes_rw = (N * K // 2) + K + N * 4 + N * 2
    bw_gbps  = bytes_rw / 1e9 / (t_us * 1e-6)
    tops     = 2.0 * N * K / 1e12 / (t_us * 1e-6)
    return t_us, bw_gbps, tops


# ---------------------------------------------------------------------------
# Benchmark W4A16 v3
# ---------------------------------------------------------------------------
def bench_w4a16_v3(N, K, kernel, threads_per_col, group_size=128, iters=500, warmup=50):
    np.random.seed(42)
    W_int4  = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    x_fp16  = np.random.randn(K).astype(np.float16)
    num_groups = K // group_size
    scales   = np.ones((num_groups, N), dtype=np.float16) * np.float16(0.01)
    zeros    = np.ones((num_groups, N), dtype=np.float16) * np.float16(8.0)
    W_packed = pack_w4a16(W_int4, K, N)

    d_x   = dev.malloc(K * 2)
    d_W   = dev.malloc(W_packed.nbytes)
    d_sc  = dev.malloc(scales.nbytes)
    d_zr  = dev.malloc(zeros.nbytes)
    d_out = dev.malloc(N * 2)

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

    for _ in range(warmup):
        dev.launch(kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(kernel, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    dev.free(d_x); dev.free(d_W); dev.free(d_sc); dev.free(d_zr); dev.free(d_out)

    bytes_w4    = (K // 2) * N
    bytes_x     = K * 2
    bytes_scales = num_groups * N * 2
    bytes_zeros  = num_groups * N * 2
    bytes_out    = N * 2
    bytes_rw = bytes_w4 + bytes_x + bytes_scales + bytes_zeros + bytes_out
    bw_gbps  = bytes_rw / 1e9 / (t_us * 1e-6)
    tops     = 2.0 * N * K / 1e12 / (t_us * 1e-6)
    return t_us, bw_gbps, tops


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
BENCH_SHAPES = [
    (4096,  4096,  "N=4096,  K=4096  (hidden→hidden)"),
    (11008, 4096,  "N=11008, K=4096  (FFN gate/up)"),
]

GROUP_SIZE = 128

print("=" * 75)
print("W4A8 GEMV vs W4A16 GEMV Benchmark")
print("Hardware: gfx906 (MI50/MI60)")
print("=" * 75)
print()
print("W4A8 dot4:  INT4 weights (K/2 bytes) + INT8 activations, v_dot4_i32_i8")
print("W4A8 dot8:  INT4 weights (K/2 bytes) + INT8 activations, v_dot8_i32_i4")
print("W4A16 v3_t4: INT4 weights (K/2 bytes) + FP16 activations, ubfe+FP32 FMA")
print(f"group_size={GROUP_SIZE}")
print()
print(f"{'Shape':<45} {'Kernel':<22} {'Latency':>10} {'BW':>10} {'TOPS':>10}")
print("-" * 102)

results = {}

for N, K, label in BENCH_SHAPES:
    # W4A8 dot4
    t_dot4, bw_dot4, tops_dot4 = bench_w4a8_dot4(N, K)
    print(f"{label:<45} {'W4A8 dot4 (v_dot4)':<22} {t_dot4:>8.2f} us  {bw_dot4:>7.1f} GB/s  {tops_dot4*1e3:>8.3f} TOPS")

    # W4A8 dot8
    t_dot8, bw_dot8, tops_dot8 = bench_w4a8_dot8(N, K)
    print(f"{'':<45} {'W4A8 dot8 (v_dot8)':<22} {t_dot8:>8.2f} us  {bw_dot8:>7.1f} GB/s  {tops_dot8*1e3:>8.3f} TOPS")

    # W4A16 v3_t4 (default for N>=4096 per architecture.md)
    t_t4, bw_t4, tops_t4 = bench_w4a16_v3(N, K, v3_t4_kernel, threads_per_col=4)
    print(f"{'':<45} {'W4A16 v3_t4':<22} {t_t4:>8.2f} us  {bw_t4:>7.1f} GB/s  {tops_t4*1e3:>8.3f} TOPS")

    # W4A16 v3_t16 (sometimes better for large N)
    t_t16, bw_t16, tops_t16 = bench_w4a16_v3(N, K, v3_t16_kernel, threads_per_col=16)
    print(f"{'':<45} {'W4A16 v3_t16':<22} {t_t16:>8.2f} us  {bw_t16:>7.1f} GB/s  {tops_t16*1e3:>8.3f} TOPS")

    best_w4a16 = min(t_t4, t_t16)
    best_w4a16_label = "v3_t4" if t_t4 <= t_t16 else "v3_t16"
    best_w4a8 = min(t_dot4, t_dot8)
    best_w4a8_label = "dot4" if t_dot4 <= t_dot8 else "dot8"

    ratio = best_w4a8 / best_w4a16
    if ratio > 1.0:
        verdict = f"W4A16 ({best_w4a16_label}) is {ratio:.2f}x FASTER"
    else:
        verdict = f"W4A8 ({best_w4a8_label}) is {1.0/ratio:.2f}x FASTER"
    print(f"{'':<45} {'Comparison':<22} → {verdict}")
    print()

    results[(N, K)] = {
        "w4a8_dot4_us": t_dot4,
        "w4a8_dot8_us": t_dot8,
        "w4a16_best_us": best_w4a16,
        "w4a16_best": best_w4a16_label,
        "w4a8_best_us": best_w4a8,
        "w4a8_best": best_w4a8_label,
        "ratio": ratio,
    }

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 75)
print("SUMMARY: W4A8 vs W4A16 Analysis")
print("=" * 75)
print()
print("MI50/MI60 peak HBM bandwidth: ~860 GB/s")
print()
print("Weight memory footprint:")
print("  W4A8:  K*N/2 bytes (same as W4A16 — INT4 weights in both)")
print("  W4A16: K*N/2 bytes")
print()
print("Activation footprint:")
print("  W4A8:  K bytes (INT8 activations)")
print("  W4A16: K*2 bytes (FP16 activations) — W4A8 saves K bytes of activation bandwidth")
print()
print("Compute path:")
print("  W4A8 dot4:  unpack INT4→INT8 + v_dot4_i32_i8 (INT8×INT8→INT32)")
print("  W4A8 dot8:  split INT8→2×INT4 + 2×v_dot8_i32_i4 (INT4×INT4→INT32)")
print("  W4A16:      ubfe dequant + FP32 FMA (INT4→FP32 + FP16×FP32→FP32)")
print()
for N, K, label in BENCH_SHAPES:
    r = results[(N, K)]
    print(f"Shape {label}:")
    print(f"  W4A8 dot4:   {r['w4a8_dot4_us']:.2f} us")
    print(f"  W4A8 dot8:   {r['w4a8_dot8_us']:.2f} us")
    print(f"  W4A16 best ({r['w4a16_best']}): {r['w4a16_best_us']:.2f} us")
    print(f"  W4A8 ({r['w4a8_best']}) vs W4A16: ratio={r['ratio']:.2f}x "
          f"({'W4A16 faster' if r['ratio'] > 1.0 else 'W4A8 faster'})")
    print()

print("CONCLUSION:")
print("  Both W4A8 and W4A16 use INT4 weight storage (same memory bandwidth).")
print("  W4A8 has slightly less activation bandwidth (INT8 vs FP16).")
print("  The performance difference comes from the compute path:")
print("    - W4A8 dot4 avoids FP32 dequant overhead (uses INT32 accumulation)")
print("    - W4A8 dot8 uses v_dot8 (8 INT4 products/instr vs v_dot4's 4 INT8 products/instr)")
print("  For decode (M=1), kernels are memory-bandwidth-bound; compute difference is small.")

dev.cleanup()
print("\nBenchmark complete.")
