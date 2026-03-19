#!/usr/bin/env python3
"""Minimal test for v2 kernel to debug crash."""
import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime

print("Starting minimal v2 kernel test...")

# Initialize HIP
hip = HIPRuntime()
hip.init()
print(f"GPUs: {hip.device_count()}")

if hip.device_count() < 4:
    print("Need 4 GPUs")
    sys.exit(1)

# Load v2 kernel
build_dir = Path(__file__).parent.parent / "build" / "kernels"
so_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
print(f"Loading kernel: {so_path}")

if not so_path.exists():
    print(f"Kernel not found: {so_path}")
    sys.exit(1)

lib = ctypes.CDLL(str(so_path))
print("Kernel loaded successfully")

# Setup devices - use single HIP runtime for all
devices = [GPUDevice(i) for i in range(4)]
print("Devices created")

# Create streams using the same HIP runtime
streams = []
for i in range(4):
    hip.set_device(i)
    streams.append(hip.stream_create())
print("Streams created")

# Test data
rng = np.random.default_rng(42)
dim = 5120
partials_np = [rng.random(dim).astype(np.float16) * 2 - 1 for _ in range(4)]
hidden_np = rng.random(dim).astype(np.float16) * 2 - 1
weight_np = rng.random(dim).astype(np.float16) + 0.5

print("Test data generated")

# Allocate on each GPU
partial_ptrs = []
hidden_ptrs = []
output_ptrs = []
weight_ptrs = []

for i in range(4):
    print(f"Allocating on GPU {i}...")
    dev = devices[i]
    dev.hip.set_device(i)
    
    # Allocate and upload partial
    p_ptr = dev.malloc(dim * 2)
    dev.upload(p_ptr, partials_np[i].tobytes())
    partial_ptrs.append(p_ptr)
    
    # Allocate and upload hidden
    h_ptr = dev.malloc(dim * 2)
    dev.upload(h_ptr, hidden_np.tobytes())
    hidden_ptrs.append(h_ptr)
    
    # Allocate output
    output_ptrs.append(dev.malloc(dim * 2))
    
    # Allocate and upload weight
    w_ptr = dev.malloc(dim * 2)
    dev.upload(w_ptr, weight_np.tobytes())
    weight_ptrs.append(w_ptr)

print("Memory allocated")

# Launch kernel on GPU 0 only first (to isolate the issue)
print("\nLaunching kernel on GPU 0 only...")
hip.set_device(0)
stream_ptr = ctypes.c_void_p(streams[0])
peers = [1, 2, 3]

err = lib.kernel_p2p_allreduce_rmsnorm_tp4_v2(
    ctypes.c_void_p(output_ptrs[0]),
    ctypes.c_void_p(hidden_ptrs[0]),
    ctypes.c_void_p(partial_ptrs[0]),
    ctypes.c_void_p(partial_ptrs[peers[0]]),
    ctypes.c_void_p(partial_ptrs[peers[1]]),
    ctypes.c_void_p(partial_ptrs[peers[2]]),
    ctypes.c_void_p(weight_ptrs[0]),
    ctypes.c_uint(dim),
    ctypes.c_uint(1),
    ctypes.c_float(1e-6),
    stream_ptr
)

print(f"Kernel launch returned: {err}")

if err != 0:
    print(f"Kernel launch failed with error {err}")
    sys.exit(1)

# Sync
print("Synchronizing...")
hip.set_device(0)
hip.stream_synchronize(streams[0])
print("GPU 0 sync complete")

# Download result
print("Downloading result...")
raw = devices[0].download(output_ptrs[0], dim * 2)
result = np.frombuffer(raw, dtype=np.float16).copy()
print(f"Result: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")

print("\nMinimal test PASSED!")

# Cleanup
for i in range(4):
    hip.set_device(i)
    hip.free(partial_ptrs[i])
    hip.free(hidden_ptrs[i])
    hip.free(output_ptrs[i])
    hip.free(weight_ptrs[i])
    hip.stream_destroy(streams[i])
for d in devices:
    d.cleanup()

print("Cleanup complete")
