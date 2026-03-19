#!/usr/bin/env python3
"""
Microbenchmark for P2P allreduce kernel per-call latency.

Measures the latency of kernel_p2p_allreduce_rmsnorm.hip for TP=4
with hidden_size=5120 FP16 elements (10KB payload).

This is a focused microbenchmark that isolates the allreduce kernel
latency without other inference overhead.

Usage:
    python3 tests/bench_allreduce_micro.py

Validates:
  - Current latency measurement (baseline ~79us per call)
  - Optimized kernel latency (target <= 50us per call)
  - Correctness: max_abs_error < 1e-3 vs reference
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError


def alloc_fill_gpu(hip: HIPRuntime, device_id: int, data: np.ndarray) -> int:
    """Allocate GPU buffer and upload numpy array. Returns device pointer."""
    hip.set_device(device_id)
    ptr = hip.malloc(data.nbytes)
    hip.memcpy_h2d(ptr, data.tobytes(), data.nbytes)
    return ptr


def download_fp16(hip: HIPRuntime, device_id: int, ptr: int, num_elems: int) -> np.ndarray:
    """Download FP16 buffer from GPU."""
    hip.set_device(device_id)
    buf = ctypes.create_string_buffer(num_elems * 2)
    hip.memcpy_d2h(buf, ptr, num_elems * 2)
    return np.frombuffer(buf, dtype=np.float16).copy()


def reference_allreduce_rmsnorm(
    partials: list, hidden: np.ndarray, weight: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Reference implementation: allreduce + RMSNorm in numpy.
    
    Args:
        partials: list of FP16 numpy arrays (one per GPU)
        hidden: FP16 hidden residual (same on all GPUs)
        weight: FP16 RMSNorm weight
        eps: epsilon for numerical stability
    
    Returns:
        FP16 result after allreduce + RMSNorm
    """
    # FP32 accumulation
    result_f32 = hidden.astype(np.float32)
    for p in partials:
        result_f32 += p.astype(np.float32)
    
    # RMSNorm
    sum_sq = np.sum(result_f32 ** 2)
    rms = np.sqrt(sum_sq / len(result_f32) + eps)
    result_normed = result_f32 / rms
    
    # Apply weight
    result_final = (result_normed * weight.astype(np.float32)).astype(np.float16)
    return result_final


class MicrobenchmarkState:
    """Manages GPU resources for microbenchmark.
    
    Uses a single HIPRuntime instance for all GPUs to ensure proper P2P coordination.
    """
    
    def __init__(self, tp_size: int = 4, hidden_size: int = 5120, use_v2: bool = False):
        self.tp_size = tp_size
        self.hidden_size = hidden_size
        self.use_v2 = use_v2
        self.device_ids = list(range(tp_size))
        
        # Single HIP runtime for all GPUs (critical for P2P access)
        self.hip = HIPRuntime()
        self.hip.init()
        
        # Enable P2P access between all GPU pairs
        for i in range(tp_size):
            self.hip.set_device(i)
            for j in range(tp_size):
                if i != j:
                    if self.hip.device_can_access_peer(i, j):
                        try:
                            self.hip.device_enable_peer_access(j)
                        except Exception:
                            pass  # Already enabled
        
        # Create streams
        self.streams = []
        for i in range(tp_size):
            self.hip.set_device(i)
            self.streams.append(self.hip.stream_create())
        
        # Load kernel library
        self.lib = self._load_kernel_lib(use_v2=use_v2)
        
    def _load_kernel_lib(self, use_v2: bool = False) -> Optional[ctypes.CDLL]:
        """Load the kernel_p2p_allreduce_rmsnorm.so library."""
        build_dir = Path(__file__).parent.parent / "build" / "kernels"
        so_name = "kernel_p2p_allreduce_rmsnorm_v2.so" if use_v2 else "kernel_p2p_allreduce_rmsnorm.so"
        so_path = build_dir / so_name
        
        if not so_path.exists():
            print(f"  WARNING: Kernel library not found: {so_path}")
            print(f"  Build with: make kernels")
            return None
        
        lib = ctypes.CDLL(str(so_path))
        
        # kernel_p2p_allreduce_rmsnorm_tp4 (or _v2):
        # (output, hidden, partial_local, peer0, peer1, peer2, weight, dim, batch_size, eps, stream)
        fn_name = "kernel_p2p_allreduce_rmsnorm_tp4_v2" if use_v2 else "kernel_p2p_allreduce_rmsnorm_tp4"
        if not hasattr(lib, fn_name):
            print(f"  WARNING: Function {fn_name} not found in library")
            return None
        
        getattr(lib, fn_name).argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # hidden
            ctypes.c_void_p,  # partial_local
            ctypes.c_void_p,  # peer0
            ctypes.c_void_p,  # peer1
            ctypes.c_void_p,  # peer2
            ctypes.c_void_p,  # weight
            ctypes.c_uint,    # dim
            ctypes.c_uint,    # batch_size
            ctypes.c_float,   # eps
            ctypes.c_void_p,  # stream
        ]
        getattr(lib, fn_name).restype = ctypes.c_int
        
        return lib
    
    def cleanup(self):
        """Free all resources."""
        for i in range(self.tp_size):
            self.hip.set_device(self.device_ids[i])
            self.hip.stream_destroy(self.streams[i])


def run_microbenchmark(
    state: MicrobenchmarkState,
    partial_ptrs: list,
    hidden_ptrs: list,
    output_ptrs: list,
    weight_ptrs: list,
    n_warmup: int = 20,
    n_iters: int = 100
) -> Tuple[float, float, list]:
    """Run the microbenchmark and measure per-call latency.
    
    Args:
        state: MicrobenchmarkState with GPU resources
        partial_ptrs: per-GPU partial buffer pointers
        hidden_ptrs: per-GPU hidden buffer pointers
        output_ptrs: per-GPU output buffer pointers
        weight_ptrs: per-GPU weight buffer pointers
        n_warmup: number of warmup iterations
        n_iters: number of measurement iterations
    
    Returns:
        (median_latency_us, mean_latency_us, latency_list)
    """
    if state.lib is None:
        print("  ERROR: Kernel library not loaded")
        return None, None, None
    
    lib = state.lib
    hip = state.hip
    dim = state.hidden_size
    batch_size = 1
    eps = 1e-6
    
    # Get the kernel function
    fn_name = "kernel_p2p_allreduce_rmsnorm_tp4_v2" if state.use_v2 else "kernel_p2p_allreduce_rmsnorm_tp4"
    kernel_fn = getattr(lib, fn_name)
    
    # Sync all GPUs first
    for i in range(state.tp_size):
        hip.set_device(state.device_ids[i])
        hip.synchronize()
    
    # Warmup
    for _ in range(n_warmup):
        for i in range(state.tp_size):
            hip.set_device(state.device_ids[i])
            stream_ptr = ctypes.c_void_p(state.streams[i])
            peers = [j for j in range(state.tp_size) if j != i]
            err = kernel_fn(
                ctypes.c_void_p(output_ptrs[i]),
                ctypes.c_void_p(hidden_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[peers[0]]),
                ctypes.c_void_p(partial_ptrs[peers[1]]),
                ctypes.c_void_p(partial_ptrs[peers[2]]),
                ctypes.c_void_p(weight_ptrs[i]),
                ctypes.c_uint(dim),
                ctypes.c_uint(batch_size),
                ctypes.c_float(eps),
                stream_ptr
            )
            if err != 0:
                raise HIPError(f"Kernel launch failed on GPU {i}: HIP error {err}")
        
        # Sync all GPUs
        for i in range(state.tp_size):
            hip.set_device(state.device_ids[i])
            hip.stream_synchronize(state.streams[i])
    
    # Benchmark
    latencies = []
    for _ in range(n_iters):
        # Record start time
        hip.set_device(0)
        hip.synchronize()  # Ensure all prior work is done
        t0 = time.perf_counter()
        
        # Launch kernel on all GPUs simultaneously
        for i in range(state.tp_size):
            hip.set_device(state.device_ids[i])
            stream_ptr = ctypes.c_void_p(state.streams[i])
            peers = [j for j in range(state.tp_size) if j != i]
            err = kernel_fn(
                ctypes.c_void_p(output_ptrs[i]),
                ctypes.c_void_p(hidden_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[peers[0]]),
                ctypes.c_void_p(partial_ptrs[peers[1]]),
                ctypes.c_void_p(partial_ptrs[peers[2]]),
                ctypes.c_void_p(weight_ptrs[i]),
                ctypes.c_uint(dim),
                ctypes.c_uint(batch_size),
                ctypes.c_float(eps),
                stream_ptr
            )
            if err != 0:
                raise HIPError(f"Kernel launch failed on GPU {i}: HIP error {err}")
        
        # Wait for all GPUs to complete
        for i in range(state.tp_size):
            hip.set_device(state.device_ids[i])
            hip.stream_synchronize(state.streams[i])
        
        # Record end time
        hip.set_device(0)
        hip.synchronize()
        t1 = time.perf_counter()
        
        latency_us = (t1 - t0) * 1e6
        latencies.append(latency_us)
    
    median_us = float(np.median(latencies))
    mean_us = float(np.mean(latencies))
    std_us = float(np.std(latencies))
    
    return median_us, mean_us, latencies


def test_correctness(
    state: MicrobenchmarkState,
    partial_ptrs: list,
    hidden_ptrs: list,
    output_ptrs: list,
    weight_ptrs: list,
    partials_np: list,
    hidden_np: np.ndarray,
    weight_np: np.ndarray
) -> Tuple[float, bool]:
    """Test correctness vs reference implementation.
    
    Returns:
        (max_abs_error, passed)
    """
    if state.lib is None:
        print("  SKIP: Kernel library not loaded")
        return None, False
    
    lib = state.lib
    hip = state.hip
    dim = state.hidden_size
    batch_size = 1
    eps = 1e-6
    
    # Run kernel on all GPUs
    fn_name = "kernel_p2p_allreduce_rmsnorm_tp4_v2" if state.use_v2 else "kernel_p2p_allreduce_rmsnorm_tp4"
    kernel_fn = getattr(lib, fn_name)
    for i in range(state.tp_size):
        hip.set_device(state.device_ids[i])
        stream_ptr = ctypes.c_void_p(state.streams[i])
        peers = [j for j in range(state.tp_size) if j != i]
        err = kernel_fn(
            ctypes.c_void_p(output_ptrs[i]),
            ctypes.c_void_p(hidden_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[peers[0]]),
            ctypes.c_void_p(partial_ptrs[peers[1]]),
            ctypes.c_void_p(partial_ptrs[peers[2]]),
            ctypes.c_void_p(weight_ptrs[i]),
            ctypes.c_uint(dim),
            ctypes.c_uint(batch_size),
            ctypes.c_float(eps),
            stream_ptr
        )
        if err != 0:
            raise HIPError(f"Kernel launch failed on GPU {i}: HIP error {err}")
    
    # Sync all GPUs
    for i in range(state.tp_size):
        hip.set_device(state.device_ids[i])
        hip.stream_synchronize(state.streams[i])
    
    # Download results from all GPUs
    results = []
    for i in range(state.tp_size):
        results.append(download_fp16(hip, state.device_ids[i], output_ptrs[i], dim))
    
    # Compute reference
    ref_result = reference_allreduce_rmsnorm(partials_np, hidden_np, weight_np, eps)
    
    # Check correctness
    max_abs_err = 0.0
    for i in range(state.tp_size):
        err = float(np.max(np.abs(
            ref_result.astype(np.float32) - results[i].astype(np.float32)
        )))
        max_abs_err = max(max_abs_err, err)
    
    max_ref = float(np.max(np.abs(ref_result.astype(np.float32))))
    passed = max_abs_err < 1e-3
    
    print(f"  Max abs error (GPU kernel vs reference): {max_abs_err:.4e}")
    print(f"  Max reference value: {max_ref:.4f}")
    print(f"  Correctness: {'PASS' if passed else 'FAIL'} (threshold: 1e-3)")
    
    # Check GPU consistency
    print(f"  GPU consistency check:")
    all_consistent = True
    for i in range(1, state.tp_size):
        diff = float(np.max(np.abs(
            results[0].astype(np.float32) - results[i].astype(np.float32)
        )))
        if diff > 1e-3:
            print(f"    GPU{i} vs GPU0: FAIL (max diff={diff:.4e})")
            all_consistent = False
        else:
            print(f"    GPU{i} vs GPU0: OK (max diff={diff:.4e})")
    
    return max_abs_err, passed and all_consistent


def benchmark_kernel(use_v2: bool) -> Tuple[float, float, float]:
    """Run benchmark for a single kernel version.
    
    Returns:
        (median_us, mean_us, max_err)
    """
    print(f"\nTesting {'v2 (optimized)' if use_v2 else 'v1 (baseline)'} kernel...")
    
    state = MicrobenchmarkState(tp_size=4, hidden_size=5120, use_v2=use_v2)
    
    if state.lib is None:
        print(f"  ERROR: Failed to load kernel library")
        state.cleanup()
        return None, None, None
    
    # Allocate test data
    rng = np.random.default_rng(42)
    partials_np = [rng.random(5120).astype(np.float16) * 2 - 1 for _ in range(4)]
    hidden_np = rng.random(5120).astype(np.float16) * 2 - 1
    weight_np = rng.random(5120).astype(np.float16) + 0.5
    
    partial_ptrs = []
    hidden_ptrs = []
    output_ptrs = []
    weight_ptrs = []
    
    for i in range(4):
        partial_ptrs.append(alloc_fill_gpu(state.hip, i, partials_np[i]))
        hidden_ptrs.append(alloc_fill_gpu(state.hip, i, hidden_np))
        state.hip.set_device(i)
        output_ptrs.append(state.hip.malloc(5120 * 2))
        weight_ptrs.append(alloc_fill_gpu(state.hip, i, weight_np))
    
    # Correctness test
    max_err, passed = test_correctness(
        state, partial_ptrs, hidden_ptrs, output_ptrs, weight_ptrs,
        partials_np, hidden_np, weight_np
    )
    
    # Benchmark
    median_us, mean_us, _ = run_microbenchmark(
        state, partial_ptrs, hidden_ptrs, output_ptrs, weight_ptrs,
        n_warmup=20, n_iters=100
    )
    
    # Cleanup
    for i in range(4):
        state.hip.set_device(i)
        state.hip.free(partial_ptrs[i])
        state.hip.free(hidden_ptrs[i])
        state.hip.free(output_ptrs[i])
        state.hip.free(weight_ptrs[i])
    state.cleanup()
    
    return median_us, mean_us, max_err


def main():
    import argparse
    parser = argparse.ArgumentParser(description='P2P Allreduce Kernel Microbenchmark')
    parser.add_argument('--v2', action='store_true', help='Test only v2 optimized kernel')
    parser.add_argument('--compare', action='store_true', help='Compare v1 and v2 kernels')
    args = parser.parse_args()
    
    print("=" * 72)
    print("  P2P Allreduce Kernel Microbenchmark")
    print("  Target: hidden_size=5120 FP16, TP=4, latency <= 50us")
    print("=" * 72)
    
    # Check GPU availability
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"\nGPUs available: {n_gpus}")
    
    if n_gpus < 4:
        print(f"ERROR: Need 4 GPUs for TP=4 benchmark (have {n_gpus})")
        sys.exit(1)
    
    baseline_us = 79.0  # Current baseline
    target_us = 50.0    # Target
    
    if args.compare:
        # Compare v1 and v2 side-by-side
        print("\n" + "=" * 72)
        print("  Comparative Benchmark: v1 vs v2")
        print("=" * 72)
        
        v1_median, v1_mean, v1_err = benchmark_kernel(use_v2=False)
        v2_median, v2_mean, v2_err = benchmark_kernel(use_v2=True)
        
        print("\n" + "=" * 72)
        print("  Comparative Results")
        print("=" * 72)
        print(f"  {'Kernel':<20} {'Median (us)':<15} {'Mean (us)':<15} {'Max Err':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        if v1_median:
            print(f"  {'v1 (baseline)':<20} {v1_median:>10.1f}     {v1_mean:>10.1f}     {v1_err:>10.4e}")
        if v2_median:
            print(f"  {'v2 (optimized)':<20} {v2_median:>10.1f}     {v2_mean:>10.1f}     {v2_err:>10.4e}")
        
        if v1_median and v2_median:
            speedup = v1_median / v2_median
            improvement = (v1_median - v2_median) / v1_median * 100
            print(f"\n  Speedup: {speedup:.2f}x ({improvement:.1f}% improvement)")
            
            if v2_median <= target_us:
                print(f"\n  ✓ v2 meets target ({v2_median:.1f} us <= {target_us:.0f} us)")
            else:
                print(f"\n  ✗ v2 misses target ({v2_median:.1f} us > {target_us:.0f} us)")
        
        print(f"\n  Summary metrics:")
        print(f"    v1_latency_us={v1_median:.1f}")
        print(f"    v2_latency_us={v2_median:.1f}")
        print(f"    target_latency_us={target_us:.1f}")
        print(f"    speedup={v1_median/v2_median if v1_median and v2_median else 0:.2f}x")
        print(f"    v2_correctness_max_err={v2_err:.4e}")
        
    elif args.v2:
        # Test only v2
        median_us, mean_us, max_err = benchmark_kernel(use_v2=True)
        
        if median_us:
            print(f"\n  Results for v2 (optimized):")
            print(f"    Median latency: {median_us:.1f} us")
            print(f"    Mean latency:   {mean_us:.1f} us")
            print(f"    Max error:      {max_err:.4e}")
            
            if median_us <= target_us:
                print(f"\n  ✓ PASS - meets target ({median_us:.1f} us <= {target_us:.0f} us)")
            else:
                print(f"\n  ✗ MISS - {median_us - target_us:.1f} us above target")
    else:
        # Default: test v1 (baseline)
        median_us, mean_us, max_err = benchmark_kernel(use_v2=False)
        
        if median_us:
            print(f"\n  Results for v1 (baseline):")
            print(f"    Median latency: {median_us:.1f} us")
            print(f"    Mean latency:   {mean_us:.1f} us")
            print(f"    Max error:      {max_err:.4e}")
    
    print("\nDone.")


def std(values):
    """Compute standard deviation."""
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


if __name__ == "__main__":
    main()
