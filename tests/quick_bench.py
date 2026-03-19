#!/usr/bin/env python3
"""Quick benchmark for TP=4 throughput."""
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.runtime.hip_dispatch import GPUDevice, HIPRuntime
from src.runtime.p2p_allreduce import P2PAllreduce

def main():
    """Quick allreduce latency test."""
    devices = [GPUDevice(i) for i in range(4)]
    hip = devices[0].hip
    
    hidden_size = 5120
    rng = np.random.default_rng(42)
    
    # Allocate partials
    partials = [rng.random(hidden_size).astype(np.float16) * 2 - 1 for _ in range(4)]
    hidden = rng.random(hidden_size).astype(np.float16) * 2 - 1
    
    partial_ptrs = []
    hidden_ptrs = []
    for i in range(4):
        hip.set_device(i)
        ptr = hip.malloc(hidden_size * 2)
        hip.memcpy_h2d(ptr, partials[i].tobytes(), hidden_size * 2)
        partial_ptrs.append(ptr)
        
        hptr = hip.malloc(hidden_size * 2)
        hip.memcpy_h2d(hptr, hidden.tobytes(), hidden_size * 2)
        hidden_ptrs.append(hptr)
    
    # Create P2P allreduce
    streams = [hip.stream_create() for _ in range(4)]
    p2p_ar = P2PAllreduce(hip, [0, 1, 2, 3], hidden_size, streams=streams)
    
    # Warmup
    for _ in range(5):
        p2p_ar.allreduce_residual_kernel(partial_ptrs, hidden_ptrs, hidden_size)
    
    # Benchmark
    n_iters = 50
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        p2p_ar.allreduce_residual_kernel(partial_ptrs, hidden_ptrs, hidden_size)
        elapsed = (time.perf_counter() - start) * 1e6
        times.append(elapsed)
    
    median = np.median(times)
    print(f"Allreduce latency (kernel P2P): median={median:.1f}us, p10={np.percentile(times, 10):.1f}us, p90={np.percentile(times, 90):.1f}us")
    
    # Cleanup
    for i in range(4):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

if __name__ == '__main__':
    main()
