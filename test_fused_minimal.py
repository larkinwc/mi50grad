#!/usr/bin/env python3
"""Minimal test of fused kernel GEMV computation."""
import numpy as np
import sys
import ctypes
sys.path.insert(0, "/opt/mi50grad")
from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco
from pathlib import Path
import subprocess

BUILD_DIR = Path("/opt/mi50grad/build/kernels")

# Build fused kernel
hip_fused = str(Path("/opt/mi50grad/src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip"))
so_fused = str(BUILD_DIR / "gemv_int4_p2p_allreduce_rmsnorm.so")
subprocess.run(f"hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o {so_fused} {hip_fused}", shell=True, check=True)

dev = GPUDevice(0)
fused_lib = ctypes.CDLL(str(so_fused))
fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
]

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

def run_fused_single_tp(tp_rank, B_q4_part, scales_part, zeros_part):
    """Run fused kernel for single TP rank and return GEMV results from partial_local."""
    hip = dev.hip
    d_A = dev.malloc(A_h16.nbytes)
    d_B = dev.malloc(B_q4_part.nbytes)
    d_s = dev.malloc(scales_part.nbytes)
    d_z = dev.malloc(zeros_part.nbytes)
    d_partial = dev.malloc(N * 2)  # Full size
    d_weight = dev.malloc(N * 2)
    d_output = dev.malloc(cols_per_gpu * 2)
    d_wg_sum_sq = dev.malloc(80 * 4)  # num_wgs * 4
    d_wg_counter = dev.malloc(4)
    
    dev.upload(d_A, A_h16.tobytes())
    dev.upload(d_B, B_q4_part.tobytes())
    dev.upload(d_s, scales_part.tobytes())
    dev.upload(d_z, zeros_part.tobytes())
    dev.upload(d_partial, np.zeros(N, dtype=np.float16).tobytes())
    dev.upload(d_weight, np.ones(N, dtype=np.float16).tobytes())
    dev.upload(d_wg_counter, np.array([0], dtype=np.uint32).tobytes())
    
    # Zero peer buffers (not used for GEMV computation)
    d_peer0 = dev.malloc(N * 2)
    d_peer1 = dev.malloc(N * 2)
    d_peer2 = dev.malloc(N * 2)
    dev.upload(d_peer0, np.zeros(N, dtype=np.float16).tobytes())
    dev.upload(d_peer1, np.zeros(N, dtype=np.float16).tobytes())
    dev.upload(d_peer2, np.zeros(N, dtype=np.float16).tobytes())
    
    err = fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4(
        d_output, d_A, d_B, d_s, d_z, d_partial, d_peer0, d_peer1, d_peer2, d_weight,
        d_wg_sum_sq, d_wg_counter,
        K, N, N, group_size, 1e-6, tp_rank, 4, 0
    )
    
    if err != 0:
        raise RuntimeError(f"Kernel error {err}")
    
    hip.synchronize()
    
    # Download partial_local to see GEMV results
    partial_data = np.frombuffer(dev.download(d_partial, N * 2), dtype=np.float16).copy()
    output_data = np.frombuffer(dev.download(d_output, cols_per_gpu * 2), dtype=np.float16).copy()
    
    dev.free(d_A); dev.free(d_B); dev.free(d_s); dev.free(d_z)
    dev.free(d_partial); dev.free(d_peer0); dev.free(d_peer1); dev.free(d_peer2)
    dev.free(d_weight); dev.free(d_output); dev.free(d_wg_sum_sq); dev.free(d_wg_counter)
    
    return partial_data, output_data

# Test all TP ranks
print("Testing fused kernel GEMV for each TP rank:")
for tp_rank in range(4):
    col_start = tp_rank * cols_per_gpu
    col_end = col_start + cols_per_gpu
    B_q4_p = B_q4[:, col_start:col_end].copy()
    scales_p = scales[:, col_start:col_end].copy().astype(np.float16)
    zeros_p = zeros[:, col_start:col_end].copy().astype(np.float16)
    
    partial, output = run_fused_single_tp(tp_rank, B_q4_p, scales_p, zeros_p)
    
    # Extract this GPU's partition from partial (stored at global indices)
    gemv_partial = partial[col_start:col_end].copy()
    
    print(f"TP{tp_rank}:")
    print(f"  Output:     range=[{output.min():.4f}, {output.max():.4f}], sum={output.sum():.4f}")
    print(f"  Partial:    range=[{gemv_partial.min():.4f}, {gemv_partial.max():.4f}], sum={gemv_partial.sum():.4f}")
    print(f"  Partial[0]: {gemv_partial[0]:.4f}, Partial[-1]: {gemv_partial[-1]:.4f}")
