#!/usr/bin/env python3
"""
Microbenchmark comparing v1, v2, and v3 allreduce kernels.

Tests all three kernel versions (if available) and reports:
- Latency per call (us)
- Speedup relative to v1
- Speedup v3 vs v2

Usage:
    python3 tests/bench_allreduce_v3.py
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice


def alloc_fill_gpu(dev: GPUDevice, data: np.ndarray) -> int:
    """Allocate GPU buffer and upload numpy array."""
    ptr = dev.malloc(data.nbytes)
    dev.upload(ptr, data.tobytes())
    return ptr


def load_kernel(build_dir, version):
    """Load kernel by version. Returns (lib, func) or (None, None)."""
    if version == "v3":
        path = build_dir / "kernel_p2p_allreduce_rmsnorm_v3.so"
        func_name = "kernel_p2p_allreduce_rmsnorm_tp4_v3"
    elif version == "v2":
        path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
        func_name = "kernel_p2p_allreduce_rmsnorm_tp4_v2"
    else:  # v1
        path = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
        func_name = "kernel_p2p_allreduce_rmsnorm_tp4"
    
    if not path.exists():
        return None, None
    
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
        return lib, func
    except Exception as e:
        print(f"  Failed to load {version}: {e}")
        return None, None


def benchmark_kernel(version, lib, func, devices, streams, p2p_info,
                    partial_ptrs, hidden_ptrs, output_ptrs, weight,
                    num_elems, n_warmup=10, n_iters=100):
    """Benchmark a specific kernel version."""
    hip = devices[0].hip
    gpu_idx = 0
    
    # Build peer pointers
    partial_ptrs_peer = []
    for peer_idx in range(4):
        if peer_idx == gpu_idx:
            partial_ptrs_peer.append(partial_ptrs[peer_idx])
        else:
            peer_info = p2p_info[gpu_idx][peer_idx]
            peer_base = peer_info['base']
            offset = partial_ptrs[peer_idx] - p2p_info[peer_idx][peer_idx]['base']
            partial_ptrs_peer.append(peer_base + offset)
    
    weight_ptr = alloc_fill_gpu(devices[gpu_idx], weight)
    
    # Warmup
    for _ in range(n_warmup):
        func(
            output_ptrs[gpu_idx],
            hidden_ptrs[gpu_idx],
            partial_ptrs_peer[0],
            partial_ptrs_peer[1],
            partial_ptrs_peer[2],
            partial_ptrs_peer[3],
            weight_ptr,
            num_elems,
            1,
            1e-6,
            streams[gpu_idx]
        )
        hip.stream_synchronize(streams[gpu_idx])
    
    # Measure
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        func(
            output_ptrs[gpu_idx],
            hidden_ptrs[gpu_idx],
            partial_ptrs_peer[0],
            partial_ptrs_peer[1],
            partial_ptrs_peer[2],
            partial_ptrs_peer[3],
            weight_ptr,
            num_elems,
            1,
            1e-6,
            streams[gpu_idx]
        )
        hip.stream_synchronize(streams[gpu_idx])
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)
    
    median = np.median(times)
    p10 = np.percentile(times, 10)
    p90 = np.percentile(times, 90)
    
    return median, p10, p90


def main():
    print("=" * 70)
    print("Allreduce Kernel Microbenchmark: v1 vs v2 vs v3")
    print("=" * 70)
    
    tp_size = 4
    num_elems = 5120
    n_warmup = 10
    n_iters = 100
    
    print(f"\nConfiguration:")
    print(f"  TP size: {tp_size}")
    print(f"  Hidden size: {num_elems}")
    print(f"  Payload: {num_elems * 2} bytes (FP16)")
    print(f"  Iterations: {n_warmup} warmup + {n_iters} measured")
    
    # Setup
    devices = [GPUDevice(i) for i in range(tp_size)]
    hip = devices[0].hip
    device_ids = list(range(tp_size))
    
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
    
    # Setup P2P
    from src.runtime.p2p_allreduce import setup_p2p_access
    p2p_info = setup_p2p_access(hip, device_ids, streams)
    
    # Load kernels
    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    
    kernels = {}
    for version in ["v3", "v2", "v1"]:
        lib, func = load_kernel(build_dir, version)
        if lib:
            kernels[version] = (lib, func)
            print(f"\n  {version.upper()}: loaded")
        else:
            print(f"\n  {version.upper()}: NOT AVAILABLE")
    
    if not kernels:
        print("\n  ERROR: No kernels available!")
        return 1
    
    # Benchmark each
    results = {}
    for version in ["v3", "v2", "v1"]:
        if version in kernels:
            print(f"\nBenchmarking {version.upper()}...")
            lib, func = kernels[version]
            median, p10, p90 = benchmark_kernel(
                version, lib, func, devices, streams, p2p_info,
                partial_ptrs, hidden_ptrs, output_ptrs, weight,
                num_elems, n_warmup, n_iters
            )
            results[version] = (median, p10, p90)
            print(f"  {version.upper()}: median={median:.2f}us, p10={p10:.2f}us, p90={p90:.2f}us")
    
    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    v1_lat = results.get("v1", (None, None, None))[0]
    v2_lat = results.get("v2", (None, None, None))[0]
    v3_lat = results.get("v3", (None, None, None))[0]
    
    print(f"\nLatency (us):")
    if v1_lat:
        print(f"  v1 (baseline): {v1_lat:.2f}")
    if v2_lat:
        print(f"  v2 (optimized): {v2_lat:.2f}")
    if v3_lat:
        print(f"  v3 (single-wavefront): {v3_lat:.2f}")
    
    print(f"\nSpeedup:")
    if v3_lat and v1_lat:
        print(f"  v3 vs v1: {v1_lat/v3_lat:.2f}x")
    if v3_lat and v2_lat:
        print(f"  v3 vs v2: {v2_lat/v3_lat:.2f}x")
    if v2_lat and v1_lat:
        print(f"  v2 vs v1: {v1_lat/v2_lat:.2f}x")
    
    # Target check
    print(f"\nTarget: v3 <= 40us")
    if v3_lat:
        if v3_lat <= 40:
            print(f"  ✓ PASS: {v3_lat:.2f}us <= 40us")
        else:
            print(f"  ✗ PARTIAL: {v3_lat:.2f}us > 40us target")
    
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
