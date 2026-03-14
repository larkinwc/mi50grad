#!/usr/bin/env python3
"""Test W8A8 tiled GEMM kernel (gemm_w8a8.hip): INT8xINT8 GEMM for prefill.

VAL-W8A8-002: W8A8 tiled GEMM kernel for prefill produces correct output for
representative shapes (4096x4096x4096, 128x4096x4096).

Tests:
1. Correctness vs INT32 accumulation reference for multiple matrix shapes
2. Verifies v_dot4_i32_i8 (sdot4) is used in kernel source
3. Edge cases: non-power-of-two M, padded tiles
4. Performance benchmark for representative Qwen 3.5 shapes

Kernel spec:
  C[M, N] = FP16(scale_a * diag(scale_w) * (A_int8[M,K] @ B_int8^T[N,K]))
  A_int8: [M, K] INT8 activations (row-major)
  B_int8: [N, K] INT8 weights (row-major, weights in row dimension)
  scale_w: [N] FP32 per-channel weight scales
  scale_a: FP32 per-tensor activation scale
  C: [M, N] FP16 output
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco

dev = GPUDevice(0)
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

hip_path   = str(PROJECT_ROOT / "src" / "kernels" / "gemm_w8a8.hip")
hsaco_path = str(BUILD_DIR / "gemm_w8a8.hsaco")

print("Building gemm_w8a8.hip ...")
build_hip_hsaco(hip_path, hsaco_path)
module = dev.load_hsaco(hsaco_path)
kernel = dev.get_kernel(module, "gemm_w8a8")
print("Build OK")

# Verify v_dot4_i32_i8 / sdot4 intrinsic is used in kernel source
with open(hip_path) as f:
    src = f.read()
assert "__builtin_amdgcn_sdot4" in src, "v_dot4_i32_i8 intrinsic not found in kernel source!"
print("v_dot4_i32_i8 intrinsic verified in source")


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a):
    """INT32 accumulation reference: matches kernel computation exactly.

    C[m, n] = scale_w[n] * scale_a * INT32_dot(A_int8[m], B_int8[n])

    Uses INT64 for exact accumulation, then scales in FP64 precision.
    Returns FP32 array (to compare with FP16 GPU output).
    """
    M, K = A_int8.shape
    N = B_int8.shape[0]
    assert B_int8.shape == (N, K), f"B shape mismatch: {B_int8.shape}"
    assert scale_w.shape == (N,), f"scale_w shape mismatch: {scale_w.shape}"

    # Exact INT64 accumulation: C_int64[M, N] = A[M, K] @ B^T[K, N]
    A_i64 = A_int8.astype(np.int64)
    B_i64 = B_int8.astype(np.int64)
    C_int64 = A_i64 @ B_i64.T  # [M, N]

    # Scale epilogue: per-channel weight scale × per-tensor activation scale
    # Broadcast scale_w [N] over rows [M]
    C_fp64 = C_int64.astype(np.float64) * scale_w.astype(np.float64)[None, :] * float(scale_a)
    return C_fp64.astype(np.float32)


# ---------------------------------------------------------------------------
# GPU launcher
# ---------------------------------------------------------------------------
def launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a):
    """Launch gemm_w8a8 kernel and return FP16 output as numpy float32.

    Args:
        A_int8:   [M, K] INT8 numpy array (activations)
        B_int8:   [N, K] INT8 numpy array (weights, rows = output channels)
        scale_w:  [N] FP32 numpy array (per-channel weight scales)
        scale_a:  FP32 scalar (per-tensor activation scale)

    Returns:
        C:  [M, N] FP32 numpy array (converted from FP16 output)
    """
    M, K = A_int8.shape
    N = B_int8.shape[0]
    assert B_int8.shape[1] == K, f"K mismatch: A has K={K}, B has K={B_int8.shape[1]}"
    assert scale_w.shape == (N,), f"scale_w shape mismatch: {scale_w.shape}"

    # Prepare byte buffers
    A_bytes  = A_int8.astype(np.int8).tobytes()
    B_bytes  = B_int8.astype(np.int8).tobytes()
    sw_bytes = scale_w.astype(np.float32).tobytes()
    out_size = M * N * 2  # FP16

    # Allocate device buffers
    d_A   = dev.malloc(len(A_bytes))
    d_B   = dev.malloc(len(B_bytes))
    d_sw  = dev.malloc(len(sw_bytes))
    d_C   = dev.malloc(out_size)

    dev.upload(d_A, A_bytes)
    dev.upload(d_B, B_bytes)
    dev.upload(d_sw, sw_bytes)

    # Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1)
    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_sw),
        ctypes.c_float(float(scale_a)),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]
    dev.launch(kernel, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    out_bytes = dev.download(d_C, out_size)
    C_fp16 = np.frombuffer(out_bytes, dtype=np.float16).reshape(M, N).copy()

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_sw)
    dev.free(d_C)

    return C_fp16.astype(np.float32)


# ---------------------------------------------------------------------------
# Correctness Tests
# ---------------------------------------------------------------------------
print("\n=== Correctness Tests ===")
all_pass = True

TEST_SHAPES = [
    # (M, N, K, label)
    (128, 4096, 4096, "128x4096x4096 (prefill)"),
    (4096, 4096, 4096, "4096x4096x4096 (large)"),
    (64, 64, 64, "64x64x64 (small, aligned)"),
    (128, 128, 128, "128x128x128 (medium, aligned)"),
    (7, 64, 128, "7x64x128 (non-aligned M)"),
    (128, 4096, 128, "128x4096x128 (narrow K)"),
]

for M, N, K, label in TEST_SHAPES:
    np.random.seed(42)

    # Random INT8 activations and weights (realistic range)
    A_int8  = np.random.randint(-64, 64, (M, K), dtype=np.int8)
    B_int8  = np.random.randint(-64, 64, (N, K), dtype=np.int8)

    # Per-channel weight scales (SmoothQuant-style, positive)
    scale_w = (np.random.rand(N).astype(np.float32) * 0.01 + 0.001)
    # Per-tensor activation scale
    scale_a = np.float32(0.008)

    # Reference: INT64 accumulation → FP64 scale → FP32
    C_ref = ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

    # GPU computation
    C_gpu = launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

    # Compute max absolute error and relative error
    abs_err = np.abs(C_gpu - C_ref)
    max_abs_err = float(np.max(abs_err))

    # Relative error (skip near-zero elements)
    ref_max = float(np.max(np.abs(C_ref)))
    thresh = ref_max * 0.01  # skip < 1% of max
    mask = np.abs(C_ref) > thresh
    if mask.sum() > 0:
        rel_err = abs_err[mask] / (np.abs(C_ref[mask]) + 1e-12)
        max_rel_err = float(np.max(rel_err))
    else:
        max_rel_err = 0.0

    # Pass criteria:
    # GPU FP16 output vs INT64-reference:
    # - FP16 has ~3 decimal digits precision, so 1% relative error for values within FP16 range
    # - The main source of error is FP16 rounding in the output (INT32 acc → scale → FP16)
    # For large K (K=4096), INT32 accumulation can reach ~4096*127*127 ~ 66M before scaling.
    # After scaling (scale_w~0.005, scale_a~0.008): ~66M * 5e-3 * 8e-3 ~ 2640, within FP16.
    # FP16 precision: ~0.1% relative error for values < 65504.
    PASS = (max_rel_err < 0.02)  # 2% tolerance for FP16 output rounding

    status = "PASS" if PASS else "FAIL"
    print(f"  {label}:")
    print(f"    max_abs={max_abs_err:.5f}  max_rel={max_rel_err:.4f}({max_rel_err*100:.2f}%)  {status}")

    if not PASS:
        all_pass = False
        # Diagnose: find worst mismatch
        worst_flat = int(np.argmax(abs_err))
        wr, wc = np.unravel_index(worst_flat, C_gpu.shape)
        print(f"    worst at [{wr},{wc}]: gpu={C_gpu[wr,wc]:.6f} ref={C_ref[wr,wc]:.6f}")


# ---------------------------------------------------------------------------
# Test: Exact correctness with unit scales
# ---------------------------------------------------------------------------
print("\n=== Unit Scale Exactness Test ===")
M, N, K = 64, 64, 64
np.random.seed(13)
A_int8 = np.random.randint(-5, 6, (M, K), dtype=np.int8)   # small values → no FP16 overflow
B_int8 = np.random.randint(-5, 6, (N, K), dtype=np.int8)
scale_w = np.ones(N, dtype=np.float32)
scale_a = np.float32(1.0)

C_ref = ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)
C_gpu = launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

abs_err = np.abs(C_gpu - C_ref)
max_abs = float(np.max(abs_err))
# With small INT8 values and unit scale, results should be exact integers in FP16
# FP16 can represent integers exactly up to 2048
PASS = max_abs < 1.0  # should be 0 for exact integer results
print(f"  Unit scale M={M},N={N},K={K}: max_abs={max_abs:.4f} {'PASS' if PASS else 'FAIL'}")
if not PASS:
    all_pass = False


# ---------------------------------------------------------------------------
# Test: Scale correctness (verify per-channel scale is applied correctly)
# ---------------------------------------------------------------------------
print("\n=== Per-Channel Scale Test ===")
M, N, K = 4, 8, 16
# A = all ones, B = identity-like, so A @ B^T is known
A_int8 = np.ones((M, K), dtype=np.int8)
B_int8 = np.zeros((N, K), dtype=np.int8)
# Each row of B has exactly one non-zero element = 1 at position j
for n in range(N):
    B_int8[n, n % K] = 1
scale_w = np.arange(1, N + 1, dtype=np.float32) * 0.1  # [0.1, 0.2, ..., 0.8]
scale_a = np.float32(2.0)

C_ref = ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)
C_gpu = launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

abs_err = np.abs(C_gpu - C_ref)
max_abs = float(np.max(abs_err))
PASS = max_abs < 0.01  # near-exact for simple values
print(f"  Per-channel scale: max_abs={max_abs:.6f} {'PASS' if PASS else 'FAIL'}")
if not PASS:
    all_pass = False
    print(f"  C_ref[0]: {C_ref[0].tolist()}")
    print(f"  C_gpu[0]: {C_gpu[0].tolist()}")


# ---------------------------------------------------------------------------
# Test: Tile boundary handling (M/N not multiples of 64)
# ---------------------------------------------------------------------------
print("\n=== Tile Boundary Tests ===")
BOUNDARY_SHAPES = [
    (65, 65, 128, "M=65,N=65,K=128 (one element past tile boundary)"),
    (1, 4096, 4096, "M=1,N=4096,K=4096 (GEMV via GEMM)"),
    (63, 128, 64, "M=63,N=128,K=64 (just under tile boundary)"),
]
for M, N, K, label in BOUNDARY_SHAPES:
    np.random.seed(99)
    A_int8  = np.random.randint(-32, 32, (M, K), dtype=np.int8)
    B_int8  = np.random.randint(-32, 32, (N, K), dtype=np.int8)
    scale_w = np.random.rand(N).astype(np.float32) * 0.005 + 0.001
    scale_a = np.float32(0.005)

    C_ref = ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)
    C_gpu = launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

    abs_err = np.abs(C_gpu - C_ref)
    max_abs = float(np.max(abs_err))
    ref_max = float(np.max(np.abs(C_ref)))
    thresh = ref_max * 0.01
    mask = np.abs(C_ref) > thresh
    max_rel = float(np.max(np.abs(C_gpu[mask] - C_ref[mask]) / (np.abs(C_ref[mask]) + 1e-12))) if mask.sum() > 0 else 0.0
    PASS = max_rel < 0.02
    print(f"  {label}:")
    print(f"    max_abs={max_abs:.5f}  max_rel={max_rel:.4f}({max_rel*100:.2f}%)  {'PASS' if PASS else 'FAIL'}")
    if not PASS:
        all_pass = False


# ---------------------------------------------------------------------------
# Test: K not multiple of 16 (TILE_K=16 boundary handling)
# ---------------------------------------------------------------------------
print("\n=== K Alignment Tests ===")
K_SHAPES = [
    (64, 64, 32, "K=32 (multiple of 16)"),
    (64, 64, 48, "K=48 (multiple of 16)"),
    (64, 64, 64, "K=64 (power of 2)"),
]
for M, N, K, label in K_SHAPES:
    np.random.seed(55)
    A_int8  = np.random.randint(-20, 20, (M, K), dtype=np.int8)
    B_int8  = np.random.randint(-20, 20, (N, K), dtype=np.int8)
    scale_w = np.ones(N, dtype=np.float32) * 0.01
    scale_a = np.float32(0.01)

    C_ref = ref_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)
    C_gpu = launch_w8a8_gemm(A_int8, B_int8, scale_w, scale_a)

    abs_err = np.abs(C_gpu - C_ref)
    max_abs = float(np.max(abs_err))
    ref_max = float(np.max(np.abs(C_ref)))
    thresh = ref_max * 0.01
    mask = np.abs(C_ref) > thresh
    max_rel = float(np.max(np.abs(C_gpu[mask] - C_ref[mask]) / (np.abs(C_ref[mask]) + 1e-12))) if mask.sum() > 0 else 0.0
    PASS = max_rel < 0.02
    print(f"  {label}: max_abs={max_abs:.5f} max_rel={max_rel:.4f}({max_rel*100:.2f}%) {'PASS' if PASS else 'FAIL'}")
    if not PASS:
        all_pass = False


# ---------------------------------------------------------------------------
# Performance Benchmark
# ---------------------------------------------------------------------------
print("\n=== Performance Benchmark ===")
import time

BENCH_SHAPES = [
    (128, 4096, 4096, "128x4096x4096 (prefill token)"),
    (4096, 4096, 4096, "4096x4096x4096 (large prefill)"),
]

for M, N, K, label in BENCH_SHAPES:
    np.random.seed(42)
    A_int8  = np.random.randint(-64, 64, (M, K), dtype=np.int8)
    B_int8  = np.random.randint(-64, 64, (N, K), dtype=np.int8)
    scale_w = np.ones(N, dtype=np.float32) * 0.01
    scale_a = np.float32(0.008)

    A_bytes  = A_int8.tobytes()
    B_bytes  = B_int8.tobytes()
    sw_bytes = scale_w.tobytes()

    d_A  = dev.malloc(len(A_bytes))
    d_B  = dev.malloc(len(B_bytes))
    d_sw = dev.malloc(len(sw_bytes))
    d_C  = dev.malloc(M * N * 2)

    dev.upload(d_A, A_bytes)
    dev.upload(d_B, B_bytes)
    dev.upload(d_sw, sw_bytes)

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_sw),
        ctypes.c_float(float(scale_a)),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]

    # Warmup
    for _ in range(10):
        dev.launch(kernel, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    # Benchmark
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(kernel, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    # Compute: 2*M*N*K INT8 multiply-add ops
    flops = 2.0 * M * N * K
    tflops = flops / 1e12 / (t_us * 1e-6)

    # Bandwidth: read A[M,K] INT8 + B[N,K] INT8 + scale_w[N] FP32 + write C[M,N] FP16
    bytes_rw = M * K + N * K + N * 4 + M * N * 2
    bw_gbps = bytes_rw / 1e9 / (t_us * 1e-6)

    print(f"  {label}: {t_us:.1f} us  {tflops:.3f} TFLOPS  {bw_gbps:.1f} GB/s")

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_sw)
    dev.free(d_C)

dev.cleanup()
print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
if not all_pass:
    sys.exit(1)
