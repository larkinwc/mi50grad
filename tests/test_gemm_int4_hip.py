#!/usr/bin/env python3
"""Test INT4 GEMM HIP kernel for prefill FFN projections."""

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

hip_path = str(PROJECT_ROOT / "src" / "kernels" / "gemm_int4_prefill_hip.hip")
hsaco_path = str(BUILD_DIR / "gemm_int4_prefill_hip.hsaco")
build_hip_hsaco(hip_path, hsaco_path)
module = dev.load_hsaco(hsaco_path)
func = dev.get_kernel(module, "gemm_int4_prefill_hip")

print("=== Correctness Tests ===")
group_size = 128

test_shapes = [
    (4, 64, 128),
    (16, 256, 256),
    (64, 1024, 512),
    (128, 17408, 5120),   # FFN gate/up
    (128, 5120, 17408),   # FFN down
]

all_pass = True
for M, N, K in test_shapes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1

    # Generate quantized weights matching GPTQ format
    n_groups = K // group_size

    # Random quantized values
    qweight = np.random.randint(0, 2**32, size=(K // 8, N), dtype=np.uint32)
    scales_np = (np.random.randn(n_groups, N) * 0.01).astype(np.float16)
    # Zeros as FP16 [n_groups, N] (matching weight_loader's unpacked format)
    zeros_np = np.random.randint(0, 15, size=(n_groups, N)).astype(np.float16)

    # NumPy reference: dequantize and multiply
    W_deq = np.zeros((N, K), dtype=np.float32)
    for ki in range(K):
        pack_idx = ki // 8
        bit_pos = (ki % 8) * 4
        g = ki // group_size
        for n in range(N):
            q = int((qweight[pack_idx, n] >> bit_pos) & 0xF)
            z = float(zeros_np[g, n])
            s = float(scales_np[g, n])
            W_deq[n, ki] = (q - z) * s
    ref = (A.astype(np.float32) @ W_deq.T).astype(np.float16)

    # Upload to GPU
    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(qweight.nbytes)
    d_scales = dev.malloc(scales_np.nbytes)
    d_zeros = dev.malloc(zeros_np.nbytes)
    d_C = dev.malloc(ref.nbytes)
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, qweight.tobytes())
    dev.upload(d_scales, scales_np.tobytes())
    dev.upload(d_zeros, zeros_np.tobytes())
    dev.hip.memset(d_C, 0, ref.nbytes)

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M), ctypes.c_uint32(N),
        ctypes.c_uint32(K), ctypes.c_uint32(group_size),
    ]
    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_C, ref.nbytes), dtype=np.float16).copy()
    result = result.reshape(M, N)

    max_err = np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32)))
    # Use relative error for large values
    ref_norm = np.linalg.norm(ref.ravel().astype(np.float32))
    res_norm = np.linalg.norm(result.ravel().astype(np.float32))
    if ref_norm > 0 and res_norm > 0:
        cos_sim = np.dot(ref.ravel().astype(np.float32), result.ravel().astype(np.float32)) / (
            ref_norm * res_norm)
    else:
        cos_sim = 1.0 if ref_norm == res_norm else 0.0

    ok = cos_sim > 0.99 and max_err < 2.0
    print(f"  M={M:4d} N={N:5d} K={K:5d}: maxerr={max_err:.4f} cos={cos_sim:.6f} "
          f"grid=({grid_x}x{grid_y}) {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_C)

dev.cleanup()
print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
