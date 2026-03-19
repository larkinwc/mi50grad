#!/usr/bin/env python3
"""Multi-GPU test for v2 fused kernel using single HIPRuntime."""
import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import HIPRuntime

print("Multi-GPU v2 fused kernel test (single HIPRuntime)...")

# Single HIP runtime for all GPUs
hip = HIPRuntime()
hip.init()
print(f"GPUs: {hip.device_count()}")

if hip.device_count() < 4:
    print("Need 4 GPUs")
    sys.exit(1)

# Enable P2P access between all GPUs
print("Enabling P2P access...")
for i in range(4):
    hip.set_device(i)
    for j in range(4):
        if i != j:
            if hip.device_can_access_peer(i, j):
                try:
                    hip.device_enable_peer_access(j)
                    print(f"  GPU{i} -> GPU{j}: enabled")
                except Exception as e:
                    print(f"  GPU{i} -> GPU{j}: already enabled or error: {e}")

# Load v2 kernel
build_dir = Path(__file__).parent.parent / "build" / "kernels"
so_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
print(f"Loading v2 kernel: {so_path}")

if not so_path.exists():
    print(f"v2 Kernel not found")
    sys.exit(1)

lib = ctypes.CDLL(str(so_path))
print("v2 Kernel loaded")

# Create streams for all GPUs
streams = []
for i in range(4):
    hip.set_device(i)
    streams.append(hip.stream_create())
print("Streams created")

# Test data
dim = 5120
rng = np.random.default_rng(42)
partials = [rng.random(dim).astype(np.float16) * 2 - 1 for _ in range(4)]
hidden = rng.random(dim).astype(np.float16) * 2 - 1
weight = rng.random(dim).astype(np.float16) + 0.5

# Allocate memory on each GPU
print("Allocating memory...")
partial_ptrs = []
hidden_ptrs = []
output_ptrs = []
weight_ptrs = []

for i in range(4):
    hip.set_device(i)
    
    p_ptr = hip.malloc(dim * 2)
    hip.memcpy_h2d(p_ptr, partials[i].tobytes(), dim * 2)
    partial_ptrs.append(p_ptr)
    
    h_ptr = hip.malloc(dim * 2)
    hip.memcpy_h2d(h_ptr, hidden.tobytes(), dim * 2)
    hidden_ptrs.append(h_ptr)
    
    output_ptrs.append(hip.malloc(dim * 2))
    
    w_ptr = hip.malloc(dim * 2)
    hip.memcpy_h2d(w_ptr, weight.tobytes(), dim * 2)
    weight_ptrs.append(w_ptr)

print("Memory allocated")

# Launch kernel on all GPUs simultaneously
print("Launching v2 kernels on all 4 GPUs...")
for i in range(4):
    hip.set_device(i)
    stream_ptr = ctypes.c_void_p(streams[i])
    peers = [j for j in range(4) if j != i]
    
    err = lib.kernel_p2p_allreduce_rmsnorm_tp4_v2(
        ctypes.c_void_p(output_ptrs[i]),
        ctypes.c_void_p(hidden_ptrs[i]),
        ctypes.c_void_p(partial_ptrs[i]),
        ctypes.c_void_p(partial_ptrs[peers[0]]),
        ctypes.c_void_p(partial_ptrs[peers[1]]),
        ctypes.c_void_p(partial_ptrs[peers[2]]),
        ctypes.c_void_p(weight_ptrs[i]),
        ctypes.c_uint(dim),
        ctypes.c_uint(1),
        ctypes.c_float(1e-6),
        stream_ptr
    )
    
    if err != 0:
        print(f"GPU{i} kernel launch failed with error {err}")
        sys.exit(1)
    print(f"  GPU{i}: launched")

# Sync all GPUs
print("Synchronizing...")
for i in range(4):
    hip.set_device(i)
    hip.stream_synchronize(streams[i])
    print(f"  GPU{i}: synced")

# Download results
print("Downloading results...")
results = []
for i in range(4):
    hip.set_device(i)
    buf = ctypes.create_string_buffer(dim * 2)
    hip.memcpy_d2h(buf, output_ptrs[i], dim * 2)
    result = np.frombuffer(buf, dtype=np.float16).copy()
    results.append(result)
    print(f"  GPU{i}: min={result.min():.4f}, max={result.max():.4f}")

# Check consistency
print("Checking GPU consistency...")
for i in range(1, 4):
    diff = float(np.max(np.abs(results[0].astype(np.float32) - results[i].astype(np.float32))))
    print(f"  GPU{i} vs GPU0: max diff = {diff:.4e}")
    if diff > 1e-3:
        print(f"  WARNING: GPUs differ!")

print("\nv2 Multi-GPU test PASSED!")

# Cleanup
for i in range(4):
    hip.set_device(i)
    hip.free(partial_ptrs[i])
    hip.free(hidden_ptrs[i])
    hip.free(output_ptrs[i])
    hip.free(weight_ptrs[i])
    hip.stream_destroy(streams[i])

print("Done")
