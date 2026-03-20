#!/usr/bin/env python3
"""
Compressed allreduce kernel correctness test.
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing HIPRuntime...")
from src.runtime.hip_dispatch import HIPRuntime
print("HIPRuntime imported")

print("Initializing HIP...")
hip = HIPRuntime()
hip.init()
n_gpus = hip.device_count()
print(f"HIP initialized, GPUs: {n_gpus}")

if n_gpus < 4:
    print(f"ERROR: Need 4 GPUs, have {n_gpus}")
    sys.exit(1)

print("Loading kernel...")
build_path = Path(__file__).parent.parent / "build" / "kernels" / "kernel_p2p_allreduce_compressed.so"
lib = ctypes.CDLL(str(build_path))
print("Kernel loaded")

# Check for split functions
has_split_functions = hasattr(lib, 'kernel_p2p_allreduce_compressed_quantize') and \
                      hasattr(lib, 'kernel_p2p_allreduce_compressed_residual_tp4_read')

if has_split_functions:
    print("Using split kernel functions (quantize + read)")
    lib.kernel_p2p_allreduce_compressed_quantize.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p
    ]
    lib.kernel_p2p_allreduce_compressed_quantize.restype = ctypes.c_int
    
    lib.kernel_p2p_allreduce_compressed_residual_tp4_read.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p
    ]
    lib.kernel_p2p_allreduce_compressed_residual_tp4_read.restype = ctypes.c_int
else:
    print("WARNING: Using combined kernel function (may hang without threading.Barrier)")
    lib.kernel_p2p_allreduce_compressed_residual_tp4.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p
    ]
    lib.kernel_p2p_allreduce_compressed_residual_tp4.restype = ctypes.c_int

print("Enabling P2P...")
for i in range(4):
    hip.set_device(i)
    for j in range(4):
        if i != j:
            try:
                hip.device_enable_peer_access(j)
            except:
                pass
print("P2P enabled")

print("Creating streams...")
streams = [hip.stream_create() for _ in range(4)]
print("Streams created")

print("Allocating test data...")
rng = np.random.default_rng(42)
n = 5120
partials = [rng.random(n).astype(np.float16) * 2 - 1 for _ in range(4)]
hidden0 = rng.random(n).astype(np.float16) * 2 - 1
num_blocks = (n + 31) // 32
compressed_size = n + num_blocks * 2
print(f"  Data size: {n} elements, compressed: {compressed_size} bytes")

print("Allocating GPU buffers...")
partial_ptrs = []
compressed_ptrs = []
hidden_ptrs = []

for i in range(4):
    hip.set_device(i)
    
    partial_ptr = hip.malloc(n * 2)
    hip.memcpy_h2d(partial_ptr, partials[i].tobytes(), n * 2)
    partial_ptrs.append(partial_ptr)
    
    compressed_ptr = hip.malloc(compressed_size)
    compressed_ptrs.append(compressed_ptr)
    
    hidden_ptr = hip.malloc(n * 2)
    hip.memcpy_h2d(hidden_ptr, hidden0.tobytes(), n * 2)
    hidden_ptrs.append(hidden_ptr)

print("  All buffers allocated")

# Get peer pointers
compressed_peer_ptrs = []
for i in range(4):
    peers = [(i + j) % 4 for j in range(1, 4)]
    compressed_peer_ptrs.append([compressed_ptrs[p] for p in peers])

print("Launching kernel...")
t0 = time.perf_counter()

if has_split_functions:
    # Two-phase approach: quantize all, sync, read all
    print("  Phase 1: Quantizing partials...")
    for i in range(4):
        hip.set_device(i)
        err = lib.kernel_p2p_allreduce_compressed_quantize(
            partial_ptrs[i],
            compressed_ptrs[i],
            n,
            num_blocks,
            ctypes.c_void_p(streams[i])
        )
        if err != 0:
            print(f"  ERROR: GPU{i} quantize kernel failed with error {err}")
            sys.exit(1)
        print(f"  GPU{i}: quantize launched")
    
    print("  Synchronizing all GPUs after quantize...")
    for i in range(4):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
    
    print("  Phase 2: Reading peer compressed data...")
    for i in range(4):
        hip.set_device(i)
        err = lib.kernel_p2p_allreduce_compressed_residual_tp4_read(
            hidden_ptrs[i],
            compressed_ptrs[i],
            compressed_peer_ptrs[i][0],
            compressed_peer_ptrs[i][1],
            compressed_peer_ptrs[i][2],
            n,
            num_blocks,
            ctypes.c_void_p(streams[i])
        )
        if err != 0:
            print(f"  ERROR: GPU{i} read kernel failed with error {err}")
            sys.exit(1)
        print(f"  GPU{i}: read launched")
else:
    # Combined kernel (buggy without threading.Barrier)
    for i in range(4):
        hip.set_device(i)
        err = lib.kernel_p2p_allreduce_compressed_residual_tp4(
            hidden_ptrs[i],
            partial_ptrs[i],
            compressed_ptrs[i],
            compressed_peer_ptrs[i][0],
            compressed_peer_ptrs[i][1],
            compressed_peer_ptrs[i][2],
            n,
            num_blocks,
            ctypes.c_void_p(streams[i])
        )
        if err != 0:
            print(f"  ERROR: GPU{i} kernel launch failed with error {err}")
            sys.exit(1)
        print(f"  GPU{i}: kernel launched")

print("Synchronizing...")
for i in range(4):
    hip.set_device(i)
    hip.stream_synchronize(streams[i])

t1 = time.perf_counter()
print(f"Kernel execution time: {(t1 - t0) * 1e6:.1f} us")

print("Downloading results...")
results = []
for i in range(4):
    hip.set_device(i)
    buf = ctypes.create_string_buffer(n * 2)
    hip.memcpy_d2h(buf, hidden_ptrs[i], n * 2)
    result = np.frombuffer(buf, dtype=np.float16).copy()
    results.append(result)

print("  Results downloaded")

# Check consistency
print("\nResults consistency:")
for i in range(1, 4):
    diff = float(np.max(np.abs(
        results[0].astype(np.float32) - results[i].astype(np.float32))))
    print(f"  GPU{i} vs GPU0 max diff: {diff:.4e}")

# Reference calculation
ref = hidden0.astype(np.float32).copy()
for p in partials:
    ref += p.astype(np.float32)
ref = ref.astype(np.float16)

# Cosine similarity
cos_sim = float(np.dot(results[0].astype(np.float32), ref.astype(np.float32)) / \
          (np.linalg.norm(results[0].astype(np.float32)) * np.linalg.norm(ref.astype(np.float32))))

print(f"\nCosine similarity vs reference: {cos_sim:.6f}")

# Cleanup
print("\nCleaning up...")
for i in range(4):
    hip.set_device(i)
    hip.free(partial_ptrs[i])
    hip.free(compressed_ptrs[i])
    hip.free(hidden_ptrs[i])
    hip.stream_destroy(streams[i])

if cos_sim >= 0.99:
    print("\nPASS: cos_sim >= 0.99")
    sys.exit(0)
else:
    print(f"\nFAIL: cos_sim = {cos_sim:.6f} < 0.99")
    sys.exit(1)
