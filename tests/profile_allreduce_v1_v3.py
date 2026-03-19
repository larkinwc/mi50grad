#!/usr/bin/env python3
"""
Profile v1 vs v3 allreduce kernels using ROCm profiler.

Usage:
    python3 tests/profile_allreduce_v1_v3.py
"""
import sys
import ctypes
import time
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime

def alloc_fill_gpu(dev: GPUDevice, data: np.ndarray) -> int:
    ptr = dev.malloc(data.nbytes)
    dev.upload(ptr, data.tobytes())
    return ptr

def download_fp16(dev: GPUDevice, ptr: int, num_elems: int) -> np.ndarray:
    raw = dev.download(ptr, num_elems * 2)
    return np.frombuffer(raw, dtype=np.float16).copy()

def load_kernel(build_dir, version):
    """Load kernel by version."""
    if version == "v3":
        path = build_dir / "kernel_p2p_allreduce_rmsnorm_v3.so"
        func_name = "kernel_p2p_allreduce_rmsnorm_tp4_v3"
    else:  # v1
        path = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
        func_name = "kernel_p2p_allreduce_rmsnorm_tp4"
    
    if not path.exists():
        return None, None, None
    
    try:
        lib = ctypes.CDLL(str(path))
        func = getattr(lib, func_name)
        func.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_float, ctypes.c_void_p,
        ]
        func.restype = ctypes.c_int
        return lib, func, path
    except Exception as e:
        print(f"  Failed to load {version}: {e}")
        return None, None, None

def profile_kernel(version, lib, func, devices, streams, partial_ptrs, hidden_ptrs, output_ptrs, weight,
                  num_elems, n_iters=100):
    """Profile kernel using Python timing."""
    hip = devices[0].hip
    
    # Allocate weight on each GPU
    weight_ptrs = []
    for i in range(4):
        hip.set_device(i)
        weight_ptrs.append(alloc_fill_gpu(devices[i], weight))
    
    # Warmup on each GPU
    for _ in range(10):
        for gpu_idx in range(4):
            hip.set_device(gpu_idx)
            # Get peer indices
            peers = [j for j in range(4) if j != gpu_idx]
            func(
                output_ptrs[gpu_idx],
                hidden_ptrs[gpu_idx],
                partial_ptrs[gpu_idx],  # local
                partial_ptrs[peers[0]],  # peer0
                partial_ptrs[peers[1]],  # peer1
                partial_ptrs[peers[2]],  # peer2
                weight_ptrs[gpu_idx],
                num_elems,
                1,
                1e-6,
                streams[gpu_idx]
            )
        # Sync all streams
        for i in range(4):
            hip.stream_synchronize(streams[i])
    
    # Measure - launch on all GPUs simultaneously
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        for gpu_idx in range(4):
            hip.set_device(gpu_idx)
            peers = [j for j in range(4) if j != gpu_idx]
            func(
                output_ptrs[gpu_idx],
                hidden_ptrs[gpu_idx],
                partial_ptrs[gpu_idx],
                partial_ptrs[peers[0]],
                partial_ptrs[peers[1]],
                partial_ptrs[peers[2]],
                weight_ptrs[gpu_idx],
                num_elems,
                1,
                1e-6,
                streams[gpu_idx]
            )
        # Sync all
        for i in range(4):
            hip.stream_synchronize(streams[i])
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)
    
    # Cleanup weight pointers
    for i in range(4):
        hip.set_device(i)
        hip.free(weight_ptrs[i])
    
    median = np.median(times)
    p10 = np.percentile(times, 10)
    p90 = np.percentile(times, 90)
    
    return median, p10, p90

def main():
    print("=" * 70)
    print("Profile v1 vs v3 Allreduce Kernels")
    print("=" * 70)
    
    tp_size = 4
    num_elems = 5120
    n_iters = 100
    
    print(f"\nConfiguration:")
    print(f"  TP size: {tp_size}")
    print(f"  Hidden size: {num_elems}")
    print(f"  Iterations: {n_iters}")
    
    # Setup
    devices = [GPUDevice(i) for i in range(tp_size)]
    hip = devices[0].hip
    
    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())
    
    rng = np.random.default_rng(0)
    partials = [rng.random(num_elems).astype(np.float16) * 2 - 1
                for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16) * 2 - 1
    weight = rng.random(num_elems).astype(np.float16) * 0.1 + 0.9
    
    partial_ptrs = []
    hidden_ptrs = []
    output_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        hidden_ptrs.append(alloc_fill_gpu(devices[i], hidden0))
        output_ptrs.append(alloc_fill_gpu(devices[i], np.zeros(num_elems, dtype=np.float16)))
    
    # Load kernels
    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    
    results = {}
    for version in ["v1", "v3"]:
        print(f"\nProfiling {version.upper()}...")
        lib, func, path = load_kernel(build_dir, version)
        if lib:
            median, p10, p90 = profile_kernel(
                version, lib, func, devices, streams,
                partial_ptrs, hidden_ptrs, output_ptrs, weight,
                num_elems, n_iters
            )
            results[version] = (median, p10, p90)
            print(f"  {version.upper()}: median={median:.2f}us, p10={p10:.2f}us, p90={p90:.2f}us")
            print(f"  Path: {path}")
            print(f"  Size: {path.stat().st_size if path.exists() else 'N/A'} bytes")
        else:
            print(f"  {version.upper()}: NOT AVAILABLE")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    v1_lat = results.get("v1", (None, None, None))[0]
    v3_lat = results.get("v3", (None, None, None))[0]
    
    if v1_lat and v3_lat:
        print(f"\nLatency comparison:")
        print(f"  v1: {v1_lat:.2f} us")
        print(f"  v3: {v3_lat:.2f} us")
        print(f"  v3 vs v1: {v3_lat/v1_lat:.2f}x ({'SLOWER' if v3_lat > v1_lat else 'FASTER'})")
        
        # Compute expected throughput impact
        # 64 allreduces per token with deferred AR
        v1_ar_time = v1_lat * 64 / 1000  # ms/token
        v3_ar_time = v3_lat * 64 / 1000  # ms/token
        print(f"\n  Expected allreduce time per token:")
        print(f"    v1: {v1_ar_time:.2f} ms")
        print(f"    v3: {v3_ar_time:.2f} ms")
        print(f"    Difference: {abs(v3_ar_time - v1_ar_time):.2f} ms")
    
    # Cleanup
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.free(output_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
