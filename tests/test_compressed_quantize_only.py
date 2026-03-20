#!/usr/bin/env python3
"""Test just the quantize kernel on a single GPU."""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import HIPRuntime

t0 = time.time()
print("Initializing HIP...")
hip = HIPRuntime()
hip.init()
print(f"  HIP init took {time.time()-t0:.2f}s")

n_gpus = hip.device_count()
print(f"  GPUs: {n_gpus}")

if n_gpus < 1:
    sys.exit(1)

t0 = time.time()
print("Loading kernel...")
build_path = Path(__file__).parent.parent / "build" / "kernels" / "kernel_p2p_allreduce_compressed.so"
lib = ctypes.CDLL(str(build_path))

lib.kernel_p2p_allreduce_compressed_quantize.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p
]
lib.kernel_p2p_allreduce_compressed_quantize.restype = ctypes.c_int
print(f"  Kernel load took {time.time()-t0:.2f}s")

t0 = time.time()
print("Creating stream...")
stream = hip.stream_create()
print(f"  Stream create took {time.time()-t0:.2f}s")

t0 = time.time()
print("Allocating buffers...")
rng = np.random.default_rng(42)
n = 1024
partial = rng.random(n).astype(np.float16) * 2 - 1
num_blocks = (n + 31) // 32
compressed_size = n + num_blocks * 2

hip.set_device(0)
partial_ptr = hip.malloc(n * 2)
hip.memcpy_h2d(partial_ptr, partial.tobytes(), n * 2)

compressed_ptr = hip.malloc(compressed_size)
print(f"  Alloc took {time.time()-t0:.2f}s")

t0 = time.time()
print("Launching quantize kernel...")
err = lib.kernel_p2p_allreduce_compressed_quantize(
    partial_ptr,
    compressed_ptr,
    n,
    num_blocks,
    ctypes.c_void_p(stream)
)
print(f"  Kernel launch took {time.time()-t0:.2f}s, err={err}")

if err != 0:
    print(f"ERROR: Kernel launch failed")
    sys.exit(1)

t0 = time.time()
print("Syncing...")
hip.stream_synchronize(stream)
print(f"  Sync took {time.time()-t0:.2f}s")

t0 = time.time()
print("Downloading result...")
buf = ctypes.create_string_buffer(compressed_size)
hip.memcpy_d2h(buf, compressed_ptr, compressed_size)
print(f"  Download took {time.time()-t0:.2f}s")

# Verify result
int8_data = np.frombuffer(buf[:n], dtype=np.int8)
scales = np.frombuffer(buf[n:n+num_blocks*2], dtype=np.float16)
print(f"\nResult:")
print(f"  INT8 data: {int8_data[:10]}... (showing first 10)")
print(f"  Scales: {scales[:5]}... (showing first 5)")

hip.free(partial_ptr)
hip.free(compressed_ptr)
hip.stream_destroy(stream)

print("\nSUCCESS: Quantize kernel works")
