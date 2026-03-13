#!/usr/bin/env python3
"""Test FP16 GEMM for prefill projections."""

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
func = dev.get_kernel(module, "gemm_fp16_prefill")

print("=== Correctness Tests ===")
test_shapes = [
    # (M, N, K) - typical prefill projection shapes
    (4, 64, 64),
    (16, 256, 128),
    (64, 1024, 512),
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
    ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B.nbytes)
    d_C = dev.malloc(ref.nbytes)
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B.tobytes())
    dev.hip.memset(d_C, 0, ref.nbytes)

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

    result = np.frombuffer(dev.download(d_C, ref.nbytes), dtype=np.float16).copy()
    result = result.reshape(M, N)

    max_err = np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32)))
    cos_sim = np.dot(ref.ravel().astype(np.float32), result.ravel().astype(np.float32)) / (
        np.linalg.norm(ref.ravel().astype(np.float32)) *
        np.linalg.norm(result.ravel().astype(np.float32)) + 1e-10)

    ok = cos_sim > 0.999 and max_err < 1.0
    print(f"  M={M:4d} N={N:5d} K={K:5d}: maxerr={max_err:.4f} cos={cos_sim:.6f} "
          f"grid=({grid_x}x{grid_y}) {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        if M <= 16 and N <= 256:
            print(f"    ref[0,:8]: {ref[0,:8]}")
            print(f"    res[0,:8]: {result[0,:8]}")

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_C)

# Performance benchmark
print(f"\n=== Performance ===")
# Q projection shape: M=128, N=6144, K=5120
for M in [16, 64, 128]:
    N, K = 6144, 5120
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(N, K).astype(np.float16) * 0.1

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B.nbytes)
    d_C = dev.malloc(M * N * 2)
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B.tobytes())

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B), ctypes.c_uint64(d_C),
        ctypes.c_uint32(M), ctypes.c_uint32(N), ctypes.c_uint32(K),
    ]

    # Warmup
    for _ in range(3):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    # Benchmark
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()
    t_ms = (time.perf_counter() - t0) / iters * 1000

    # Compare with M individual GEMV calls (what prefill currently does)
    from src.kernels.launcher import build_hip_hsaco as bh
    gemv_hip = str(PROJECT_ROOT / "src" / "kernels" / "gemv_fp16_v2.hip")
    gemv_hsaco = str(BUILD_DIR / "gemv_fp16_v2.hsaco")
    bh(gemv_hip, gemv_hsaco)
    gemv_mod = dev.load_hsaco(gemv_hsaco)
    gemv_func = dev.get_kernel(gemv_mod, "gemv_fp16_v2")

    d_x = dev.malloc(K * 2)
    d_out = dev.malloc(N * 2)

    for _ in range(3):
        for t in range(M):
            gemv_params = [
                ctypes.c_uint64(d_x), ctypes.c_uint64(d_B),
                ctypes.c_uint64(d_out), ctypes.c_uint32(K), ctypes.c_uint32(N),
                ctypes.c_uint64(0),  # residual=null
            ]
            dev.launch(gemv_func, ((N + 3) // 4, 1, 1), (256, 1, 1), gemv_params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for t in range(M):
            gemv_params = [
                ctypes.c_uint64(d_x), ctypes.c_uint64(d_B),
                ctypes.c_uint64(d_out), ctypes.c_uint32(K), ctypes.c_uint32(N),
                ctypes.c_uint64(0),  # residual=null
            ]
            dev.launch(gemv_func, ((N + 3) // 4, 1, 1), (256, 1, 1), gemv_params)
    dev.synchronize()
    t_gemv = (time.perf_counter() - t0) / iters * 1000

    # Theoretical: FLOPS = 2*M*N*K, bandwidth = (M*K + N*K + M*N) * 2 bytes
    flops = 2 * M * N * K
    bytes_accessed = (M * K + N * K + M * N) * 2
    peak_tflops = 12.29  # gfx906 peak FP32
    peak_bw = 857  # GB/s
    t_compute = flops / (peak_tflops * 1e9)  # ms
    t_bandwidth = bytes_accessed / (peak_bw * 1e6)  # ms

    print(f"  M={M:4d} N={N} K={K}: GEMM={t_ms:.2f}ms, {M}xGEMV={t_gemv:.2f}ms, "
          f"speedup={t_gemv/t_ms:.1f}x, "
          f"theo_compute={t_compute:.2f}ms, theo_bw={t_bandwidth:.2f}ms")

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_C)
    dev.free(d_x)
    dev.free(d_out)

dev.cleanup()
print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
