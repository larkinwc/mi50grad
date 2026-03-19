#!/usr/bin/env python3
"""Test v1 fused kernel only."""
import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime

print("Testing v1 fused kernel...")

hip = HIPRuntime()
hip.init()
print(f"GPUs: {hip.device_count()}")

if hip.device_count() < 4:
    print("Need 4 GPUs")
    sys.exit(1)

# Load v1 kernel
build_dir = Path(__file__).parent.parent / "build" / "kernels"
so_path = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
print(f"Loading v1 kernel: {so_path}")

if not so_path.exists():
    print(f"Kernel not found: {so_path}")
    sys.exit(1)

lib = ctypes.CDLL(str(so_path))
print("v1 Kernel loaded")

# Simple test - single GPU first
dev = GPUDevice(0)
print("Device 0 created")

# Allocate memory on GPU 0 only
dim = 5120
rng = np.random.default_rng(42)
partial0 = rng.random(dim).astype(np.float16) * 2 - 1
hidden = rng.random(dim).astype(np.float16) * 2 - 1
weight = rng.random(dim).astype(np.float16) + 0.5

# For single GPU test, we'll use the same pointer for all peers (not realistic but tests kernel launch)
p_ptr = dev.malloc(dim * 2)
dev.upload(p_ptr, partial0.tobytes())
h_ptr = dev.malloc(dim * 2)
dev.upload(h_ptr, hidden.tobytes())
w_ptr = dev.malloc(dim * 2)
dev.upload(w_ptr, weight.tobytes())
out_ptr = dev.malloc(dim * 2)

print("Memory allocated")

# Launch kernel (self-referencing for single GPU test)
stream = hip.stream_create()
stream_ptr = ctypes.c_void_p(stream)

print("Launching kernel...")
err = lib.kernel_p2p_allreduce_rmsnorm_tp4(
    ctypes.c_void_p(out_ptr),
    ctypes.c_void_p(h_ptr),
    ctypes.c_void_p(p_ptr),
    ctypes.c_void_p(p_ptr),  # Using same pointer (not realistic)
    ctypes.c_void_p(p_ptr),
    ctypes.c_void_p(p_ptr),
    ctypes.c_void_p(w_ptr),
    ctypes.c_uint(dim),
    ctypes.c_uint(1),
    ctypes.c_float(1e-6),
    stream_ptr
)

print(f"Kernel returned: {err}")

if err != 0:
    print(f"Kernel launch failed!")
    sys.exit(1)

# Sync
hip.set_device(0)
hip.stream_synchronize(stream)
print("Sync complete")

# Download result
raw = dev.download(out_ptr, dim * 2)
result = np.frombuffer(raw, dtype=np.float16).copy()
print(f"Result: min={result.min():.4f}, max={result.max():.4f}")

print("v1 single-GPU test PASSED!")

# Cleanup
hip.free(p_ptr)
hip.free(h_ptr)
hip.free(w_ptr)
hip.free(out_ptr)
hip.stream_destroy(stream)
dev.cleanup()

print("Done")
