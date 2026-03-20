#!/usr/bin/env python3
"""Debug test to check peer buffer setup and reading."""
import numpy as np
import sys
import ctypes
sys.path.insert(0, "/opt/mi50grad")
from src.runtime.hip_dispatch import GPUDevice

dev = GPUDevice(0)

# Create test data
N = 5120
cols_per_gpu = N // 4
gemv_full = np.arange(N, dtype=np.float16)  # Simple pattern: 0, 1, 2, ..., 5119

print("gemv_full:", gemv_full[:10], "...", gemv_full[-10:])
print("gemv_full sum of squares:", np.sum(gemv_full.astype(np.float32)**2))

# Test TP rank 0
tp_rank = 0
other_gpus = [g for g in range(4) if g != tp_rank]
print(f"\nTP{tp_rank}: other_gpus = {other_gpus}")

partial_peer0 = gemv_full[other_gpus[0]*cols_per_gpu:(other_gpus[0]+1)*cols_per_gpu].copy()
partial_peer1 = gemv_full[other_gpus[1]*cols_per_gpu:(other_gpus[1]+1)*cols_per_gpu].copy()
partial_peer2 = gemv_full[other_gpus[2]*cols_per_gpu:(other_gpus[2]+1)*cols_per_gpu].copy()

print(f"partial_peer0: {partial_peer0[:5]} ... {partial_peer0[-5:]} (from GPU{other_gpus[0]})")
print(f"partial_peer1: {partial_peer1[:5]} ... {partial_peer1[-5:]} (from GPU{other_gpus[1]})")
print(f"partial_peer2: {partial_peer2[:5]} ... {partial_peer2[-5:]} (from GPU{other_gpus[2]})")

# Simulate what the kernel does
sum_sq = 0.0
for i in range(N):
    col_gpu = i // cols_per_gpu
    if col_gpu == tp_rank:
        # Read from partial_local (simulated as gemv_full for this GPU's columns)
        val = float(gemv_full[i])
    else:
        col_in_peer = i % cols_per_gpu
        if col_gpu < tp_rank:
            peer_idx = col_gpu
        else:
            peer_idx = col_gpu - 1
        
        if peer_idx == 0:
            val = float(partial_peer0[col_in_peer])
        elif peer_idx == 1:
            val = float(partial_peer1[col_in_peer])
        else:
            val = float(partial_peer2[col_in_peer])
    
    sum_sq += val * val

print(f"\nSimulated sum_sq for TP{tp_rank}: {sum_sq}")
print(f"Expected sum_sq (all columns): {np.sum(gemv_full.astype(np.float32)**2)}")

# Do the same for all TP ranks
print("\n\nAll TP ranks:")
for tp_rank in range(4):
    other_gpus = [g for g in range(4) if g != tp_rank]
    pp0 = gemv_full[other_gpus[0]*cols_per_gpu:(other_gpus[0]+1)*cols_per_gpu].copy()
    pp1 = gemv_full[other_gpus[1]*cols_per_gpu:(other_gpus[1]+1)*cols_per_gpu].copy()
    pp2 = gemv_full[other_gpus[2]*cols_per_gpu:(other_gpus[2]+1)*cols_per_gpu].copy()
    
    sum_sq = 0.0
    for i in range(N):
        col_gpu = i // cols_per_gpu
        if col_gpu == tp_rank:
            val = float(gemv_full[i])
        else:
            col_in_peer = i % cols_per_gpu
            peer_idx = col_gpu if col_gpu < tp_rank else col_gpu - 1
            if peer_idx == 0:
                val = float(pp0[col_in_peer])
            elif peer_idx == 1:
                val = float(pp1[col_in_peer])
            else:
                val = float(pp2[col_in_peer])
        sum_sq += val * val
    
    print(f"TP{tp_rank}: sum_sq = {sum_sq:.2f}")
