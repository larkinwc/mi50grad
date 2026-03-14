#!/usr/bin/env python3
"""Test double-buffered FP16 GEMM for prefill projections.

Tests the gemm_fp16_prefill_db kernel (double-buffered K-loop) alongside
the existing gemm_fp16_prefill kernel.

Correctness: max abs error < 1e-2 at all Qwen shapes
Performance: speedup ratio between db and single-buffered versions
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

dev = GPUDevice(0)
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

hip_path = str(PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill.hip")
hsaco_path = str(BUILD_DIR / "gemm_fp16_prefill.hsaco")
build_hip_hsaco(hip_path, hsaco_path)
module = dev.load_hsaco(hsaco_path)

# Load both kernels from the same .so
func_orig = dev.get_kernel(module, "gemm_fp16_prefill")
func_db   = dev.get_kernel(module, "gemm_fp16_prefill_db")


def run_gemm(func, A, B, dev):
    """Run a GEMM kernel and return result as numpy array."""
    M, K = A.shape
    N = B.shape[0]  # B is [N, K]

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B.nbytes)
    d_C = dev.malloc(M * N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B.tobytes())
    dev.hip.memset(d_C, 0, M * N * 2)

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]
    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_C, M * N * 2), dtype=np.float16).copy()
    result = result.reshape(M, N)

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_C)
    return result


def benchmark_gemm(func, A, B, dev, iters=100, warmup=10):
    """Benchmark a GEMM kernel, returns median latency in microseconds."""
    M, K = A.shape
    N = B.shape[0]

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B.nbytes)
    d_C = dev.malloc(M * N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B.tobytes())

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]

    # Warmup
    for _ in range(warmup):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)  # us

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_C)

    return float(np.median(times))


print("=== Double-Buffered FP16 GEMM Correctness Tests ===")

# VAL-GEMM-001: Qwen-sized shapes + small shapes
test_shapes = [
    # Small shapes for quick sanity check
    (4,   64,   64),
    (16, 256,  128),
    (64, 1024,  512),
    # Qwen shapes (required by spec)
    (128, 6144, 5120),   # Q projection: seq=128, out=6144, in=5120
    (128, 1024, 5120),   # K/V projection
    (128, 5120, 6144),   # Output projection
]

all_pass = True
for M, N, K in test_shapes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(N, K).astype(np.float16) * 0.1  # [N, K] row-major

    # NumPy reference: C = A @ B^T
    ref = (A.astype(np.float32) @ B.astype(np.float32).T)

    result_db = run_gemm(func_db, A, B, dev)

    max_err = float(np.max(np.abs(ref - result_db.astype(np.float32))))
    ok = max_err < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  M={M:4d} N={N:5d} K={K:5d}: max_abs_err={max_err:.6f} {status}")

    if not ok:
        all_pass = False
        # Extra debug for small shapes
        if M <= 16:
            print(f"    ref[0,:4]: {ref[0,:4]}")
            print(f"    res[0,:4]: {result_db[0,:4]}")

# ============================================================
# Performance benchmark: db vs. single-buffered
# VAL-GEMM-002: shape (128, 6144, 5120) over 100 iterations
# ============================================================
print("\n=== Performance Benchmark: db vs. single-buffered ===")
print("  Shape: M=128, N=6144, K=5120 — 100 iterations, 10 warmup")

M, N, K = 128, 6144, 5120
np.random.seed(7)
A = np.random.randn(M, K).astype(np.float16) * 0.1
B = np.random.randn(N, K).astype(np.float16) * 0.1

lat_orig = benchmark_gemm(func_orig, A, B, dev, iters=100, warmup=10)
lat_db   = benchmark_gemm(func_db,   A, B, dev, iters=100, warmup=10)
speedup  = lat_orig / lat_db

flops = 2 * M * N * K
tflops_orig = flops / (lat_orig * 1e-6) / 1e12
tflops_db   = flops / (lat_db   * 1e-6) / 1e12

print(f"  original (single-buffer): {lat_orig:.1f} us  ({tflops_orig:.2f} TFLOPS)")
print(f"  double-buffered:          {lat_db:.1f} us  ({tflops_db:.2f} TFLOPS)")
print(f"  speedup: {speedup:.2f}x")

if speedup >= 1.0:
    print("  Performance: PASS (measurable speedup)")
else:
    print(f"  Performance: NOTE — speedup={speedup:.3f}x (regression, check hardware state)")

dev.cleanup()

print()
if all_pass:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
    sys.exit(1)
