#!/usr/bin/env python3
"""Quick test to verify partitioned GEMV works correctly."""
import numpy as np
import sys
sys.path.insert(0, "/opt/mi50grad")
from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco
from pathlib import Path
import ctypes

BUILD_DIR = Path("/opt/mi50grad/build/kernels")

# Build reference kernel
hip_v6 = str(Path("/opt/mi50grad/src/kernels/gemv_int4_v6.hip"))
hsaco_v6 = str(BUILD_DIR / "gemv_int4_v6.hsaco")
build_hip_hsaco(hip_v6, hsaco_v6)

dev = GPUDevice(0)
module_v6 = dev.load_hsaco(hsaco_v6)
func_v6_t16 = dev.get_kernel(module_v6, "gemv_int4_v6_t16")

# Test parameters
K = 17408
N = 5120
cols_per_gpu = N // 4
group_size = 128

np.random.seed(42)
A_f32 = (np.random.randn(K) * 0.1).astype(np.float32)
A_h16 = A_f32.astype(np.float16)
W_fp32 = (np.random.randn(K, N) * 0.1).astype(np.float32)

# Quantize
scales = np.zeros((K // group_size, N), dtype=np.float32)
zeros  = np.zeros((K // group_size, N), dtype=np.float32)
q4_mat = np.zeros((K, N), dtype=np.uint8)

for g in range(K // group_size):
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

def run_gemv(B_q4_part, scales_part, zeros_part, N_part):
    cols_per_wg = 16
    grid_x = (N_part + cols_per_wg - 1) // cols_per_wg
    dev = GPUDevice(0)
    d_A = dev.malloc(A_h16.nbytes)
    d_B = dev.malloc(B_q4_part.nbytes)
    d_s = dev.malloc(scales_part.nbytes)
    d_z = dev.malloc(zeros_part.nbytes)
    d_C = dev.malloc(N_part * 2)
    dev.upload(d_A, A_h16.tobytes())
    dev.upload(d_B, B_q4_part.tobytes())
    dev.upload(d_s, scales_part.tobytes())
    dev.upload(d_z, zeros_part.tobytes())
    params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B), ctypes.c_uint64(d_s), ctypes.c_uint64(d_z),
        ctypes.c_uint64(d_C), ctypes.c_uint32(K), ctypes.c_uint32(N_part), ctypes.c_uint32(group_size),
    ]
    dev.launch(func_v6_t16, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    result = np.frombuffer(dev.download(d_C, N_part * 2), dtype=np.float16).copy()
    dev.free(d_A); dev.free(d_B); dev.free(d_s); dev.free(d_z); dev.free(d_C)
    return result

# Test all TP ranks
results = []
for tp_rank in range(4):
    col_start = tp_rank * cols_per_gpu
    col_end = col_start + cols_per_gpu
    B_q4_p = B_q4[:, col_start:col_end].copy()
    scales_p = scales[:, col_start:col_end].copy().astype(np.float16)
    zeros_p = zeros[:, col_start:col_end].copy().astype(np.float16)
    r = run_gemv(B_q4_p, scales_p, zeros_p, cols_per_gpu)
    results.append(r)
    print(f"TP{tp_rank}: range=[{r.min():.4f}, {r.max():.4f}], sum={r.sum():.4f}")

full = np.concatenate(results)
print(f"\nFull result: min={full.min():.4f}, max={full.max():.4f}")
print(f"Per-partition sums: TP0={results[0].sum():.2f}, TP1={results[1].sum():.2f}, TP2={results[2].sum():.2f}, TP3={results[3].sum():.2f}")
