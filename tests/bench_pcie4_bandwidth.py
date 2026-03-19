#!/usr/bin/env python3
"""
PCIe 4.0 P2P Bandwidth Microbenchmark for MI50 (gfx906).

This benchmark measures the actual PCIe 4.0 bandwidth utilization during P2P allreduce
operations on 4x MI50 GPUs connected via PCIe 4.0 x16 (2-hop through switch).

Tests:
1. Raw BAR1 P2P read bandwidth between GPU pairs (memcpy peer vs kernel-based reads)
2. Multiple payload sizes: 1KB, 5KB, 10KB, 30KB, 100KB, 1MB
3. Different access patterns: sequential coalesced, strided, random
4. Kernel launch overhead vs actual data transfer time
5. Vectorized load widths: float (4B), float2 (8B), float4 (16B), dwordx4 (16B)

Hardware: PCIe 4.0 x16, 2-hop via switch, all pairs weight=40
Theoretical PCIe 4.0 x16 bandwidth: ~25.6 GB/s (raw), ~23-24 GB/s practical
Measured P2P bandwidth (2-hop): ~12 GB/s effective

Returns: bytes/sec achieved, % of theoretical PCIe 4.0 x16, and latency/bandwidth analysis
"""

import ctypes
import time
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import HIPRuntime, HIPError


class PCIebandwidthBenchmark:
    """PCIe 4.0 P2P bandwidth microbenchmark."""

    def __init__(self, device_ids=None):
        """Initialize benchmark.

        Args:
            device_ids: list of GPU device IDs (default: [0, 1, 2, 3])
        """
        self.hip = HIPRuntime()
        self.device_ids = device_ids or [0, 1, 2, 3]
        self.tp_size = len(self.device_ids)
        
        # Enable P2P access
        self._enable_p2p()
        
        # Results storage
        self.results = {}
        
    def _enable_p2p(self):
        """Enable P2P access between all GPU pairs."""
        for i, dev_i in enumerate(self.device_ids):
            self.hip.set_device(dev_i)
            for j, dev_j in enumerate(self.device_ids):
                if i == j:
                    continue
                if self.hip.device_can_access_peer(dev_i, dev_j):
                    try:
                        self.hip.device_enable_peer_access(dev_j)
                    except HIPError:
                        pass  # Already enabled
    
    def _get_p2p_ptr(self, ptr, src_dev, dst_dev):
        """Get P2P pointer for accessing src_dev's memory from dst_dev.
        
        Args:
            ptr: Device pointer on src_dev
            src_dev: Source device ID where ptr is allocated
            dst_dev: Destination device ID that will access the memory
            
        Returns:
            P2P-accessible pointer for dst_dev
        """
        self.hip.set_device(dst_dev)
        return self.hip.get_p2p_peer_ptr(ptr, src_dev)
    
    def bench_memcpy_peer(self, payload_sizes, num_iterations=100):
        """Benchmark hipMemcpyPeerAsync bandwidth.
        
        Measures raw P2P memory copy bandwidth between GPU pairs.
        
        Args:
            payload_sizes: list of payload sizes in bytes
            num_iterations: number of iterations per measurement
        """
        print("\n" + "="*70)
        print("  hipMemcpyPeerAsync P2P Bandwidth Benchmark")
        print("="*70)
        
        results = {}
        
        # Allocate buffers on all GPUs
        max_size = max(payload_sizes)
        buffers = []
        for dev_id in self.device_ids:
            self.hip.set_device(dev_id)
            buffers.append(self.hip.malloc(max_size))
        
        for size in payload_sizes:
            latencies = []
            bandwidths = []
            
            # Test all GPU pairs
            for i, src_dev in enumerate(self.device_ids):
                for j, dst_dev in enumerate(self.device_ids):
                    if i == j:
                        continue
                    
                    # Warmup
                    self.hip.set_device(src_dev)
                    stream = self.hip.stream_create()
                    
                    for _ in range(10):
                        self.hip.memcpy_peer_async(
                            buffers[dst_dev], dst_dev,
                            buffers[src_dev], src_dev,
                            size, stream)
                    
                    self.hip.stream_synchronize(stream)
                    
                    # Measure latency
                    times = []
                    for _ in range(num_iterations):
                        start = time.perf_counter()
                        self.hip.memcpy_peer_async(
                            buffers[dst_dev], dst_dev,
                            buffers[src_dev], src_dev,
                            size, stream)
                        self.hip.stream_synchronize(stream)
                        end = time.perf_counter()
                        times.append((end - start) * 1e6)  # microseconds
                    
                    latencies.extend(times)
                    bandwidth = (size / 1e9) / (np.mean(times) * 1e-6)  # GB/s
                    bandwidths.append(bandwidth)
                    
                    self.hip.stream_destroy(stream)
            
            results[size] = {
                'memcpy_avg_latency_us': np.mean(latencies),
                'memcpy_std_latency_us': np.std(latencies),
                'memcpy_min_latency_us': np.min(latencies),
                'memcpy_max_latency_us': np.max(latencies),
                'memcpy_avg_bandwidth_gbps': np.mean(bandwidths),
                'memcpy_std_bandwidth_gbps': np.std(bandwidths),
            }
            
            print(f"  Payload: {size / 1e3:6.1f} KB | "
                  f"Avg Latency: {np.mean(latencies):7.2f} µs | "
                  f"Bandwidth: {np.mean(bandwidths):6.2f} ± {np.std(bandwidths):.2f} GB/s")
        
        # Cleanup
        for buf in buffers:
            self.hip.set_device(self.device_ids[0])
            self.hip.free(buf)
        
        return results
    
    def bench_kernel_p2p_read(self, payload_sizes, num_iterations=100,
                              access_pattern='sequential', vector_width='float4'):
        """Benchmark kernel-based P2P reads with different access patterns.
        
        Tests different kernel configurations:
        - Sequential coalesced access (optimal)
        - Strided access (tests coalescing requirements)
        - Random access (worst case, tests latency)
        
        Vector widths:
        - float (4 bytes)
        - float2 (8 bytes)
        - float4 (16 bytes)
        - dwordx4 (16 bytes, 4x dword)
        
        Args:
            payload_sizes: list of payload sizes in bytes
            num_iterations: number of iterations
            access_pattern: 'sequential', 'strided', or 'random'
            vector_width: 'float', 'float2', 'float4', or 'dwordx4'
        """
        print("\n" + "="*70)
        print(f"  Kernel P2P Read Benchmark ({access_pattern}, {vector_width})")
        print("="*70)
        
        results = {}
        
        # Load kernel library
        kernel_path = Path(__file__).parent.parent / "build" / "kernels" / "p2p_bandwidth_kernels.so"
        
        if not kernel_path.exists():
            print(f"  WARNING: Kernel library not found at {kernel_path}")
            print(f"  Building kernel library...")
            self._build_p2p_bandwidth_kernel(kernel_path)
        
        if kernel_path.exists():
            lib = ctypes.CDLL(str(kernel_path))
            
            # Get the appropriate kernel function
            kernel_func_name = f"p2p_read_{access_pattern}_{vector_width}"
            if not hasattr(lib, kernel_func_name):
                print(f"  WARNING: Kernel function {kernel_func_name} not found")
                return results
            
            kernel_func = getattr(lib, kernel_func_name)
            kernel_func.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p
            ]
            kernel_func.restype = ctypes.c_int
            
            # Allocate buffers and P2P pointers
            max_size = max(payload_sizes)
            buffers = []
            p2p_ptrs = []  # p2p_ptrs[dst][src] = P2P pointer to src's buffer accessible from dst
            
            for dev_id in self.device_ids:
                self.hip.set_device(dev_id)
                buffers.append(self.hip.malloc(max_size))
                p2p_ptrs.append([])
            
            # Get P2P pointers
            for i, dst_dev in enumerate(self.device_ids):
                self.hip.set_device(dst_dev)
                for j, src_dev in enumerate(self.device_ids):
                    if i != j:
                        try:
                            p2p_ptr = self.hip.get_p2p_peer_ptr(buffers[src_dev], src_dev)
                            p2p_ptrs[i].append((src_dev, p2p_ptr))
                        except Exception:
                            p2p_ptrs[i].append((src_dev, buffers[src_dev]))
            
            # Determine element size based on vector_width
            element_bytes = {'float': 4, 'float2': 8, 'float4': 16, 'dwordx4': 16}.get(vector_width, 4)
            
            for size in payload_sizes:
                latencies = []
                bandwidths = []
                num_elements = size // element_bytes
                
                # Test kernel launch on each GPU reading from peer 0
                for i, dev_id in enumerate(self.device_ids):
                    self.hip.set_device(dev_id)
                    stream = self.hip.stream_create()
                    
                    # Warmup
                    peer_ptr = p2p_ptrs[i][0][1] if p2p_ptrs[i] else buffers[dev_id]
                    for _ in range(10):
                        kernel_func(
                            ctypes.c_void_p(0),  # output (dummy, read-only test)
                            ctypes.c_void_p(peer_ptr),
                            ctypes.c_uint(num_elements),
                            stream)
                    self.hip.stream_synchronize(stream)
                    
                    # Measure
                    times = []
                    for _ in range(num_iterations):
                        start = time.perf_counter()
                        kernel_func(
                            ctypes.c_void_p(0),
                            ctypes.c_void_p(peer_ptr),
                            ctypes.c_uint(num_elements),
                            stream)
                        self.hip.stream_synchronize(stream)
                        end = time.perf_counter()
                        times.append((end - start) * 1e6)
                    
                    latencies.extend(times)
                    bandwidth = (size / 1e9) / (np.mean(times) * 1e-6)
                    bandwidths.append(bandwidth)
                    
                    self.hip.stream_destroy(stream)
                
                results[size] = {
                    'kernel_avg_latency_us': np.mean(latencies),
                    'kernel_std_latency_us': np.std(latencies),
                    'kernel_bandwidth_gbps': np.mean(bandwidths),
                }
                
                print(f"  Payload: {size / 1e3:6.1f} KB | "
                      f"Avg Latency: {np.mean(latencies):7.2f} µs | "
                      f"Bandwidth: {np.mean(bandwidths):6.2f} GB/s")
            
            # Cleanup
            for buf in buffers:
                self.hip.set_device(self.device_ids[0])
                self.hip.free(buf)
        else:
            print(f"  WARNING: Failed to build kernel library")
            print(f"  Skipping kernel P2P read benchmark.")
        
        return results
    
    def _build_p2p_bandwidth_kernel(self, kernel_path):
        """Build P2P bandwidth benchmark kernel.
        
        Args:
            kernel_path: Path to output .so file
        """
        import subprocess
        
        src_path = Path(__file__).parent.parent / "src" / "kernels" / "p2p_bandwidth_kernels.hip"
        
        if not src_path.exists():
            print(f"  ERROR: Kernel source not found at {src_path}")
            return False
        
        build_dir = kernel_path.parent
        build_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.check_call([
                "/opt/rocm/bin/hipcc",
                "-O3", "--offload-arch=gfx906", "-std=c++17",
                "-shared", "-fPIC",
                "-o", str(kernel_path),
                str(src_path),
            ])
            print(f"  Built kernel: {kernel_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  WARNING: Failed to build kernel: {e}")
            return False
    
    def bench_vector_width_comparison(self, payload_sizes, num_iterations=100):
        """Compare different vectorized load widths.
        
        Args:
            payload_sizes: list of payload sizes
            num_iterations: number of iterations
        """
        print("\n" + "="*70)
        print("  Vector Width Comparison Benchmark")
        print("="*70)
        
        vector_widths = ['float', 'float2', 'float4', 'dwordx4']
        results = {}
        
        for vw in vector_widths:
            print(f"\n  Testing {vw} ({[4, 8, 16, 16][vector_widths.index(vw)]} bytes)...")
            results[vw] = self.bench_kernel_p2p_read(
                payload_sizes, num_iterations,
                access_pattern='sequential', vector_width=vw)
        
        # Print comparison
        print("\n  Vector Width Comparison Summary:")
        print(f"  {'Payload':<10} | {'float':<8} | {'float2':<8} | {'float4':<8} | {'dwordx4':<8}")
        print(f"  {'(KB)':<10} | {'GB/s':<8} | {'GB/s':<8} | {'GB/s':<8} | {'GB/s':<8}")
        print("-" * 55)
        
        for size in payload_sizes[:4]:  # Show first 4 sizes
            size_kb = size / 1e3
            row = f"  {size_kb:<10.1f} | "
            for vw in vector_widths:
                if size in results.get(vw, {}):
                    bw = results[vw][size]['kernel_bandwidth_gbps']
                    row += f"{bw:<8.2f} | "
            print(row)
        
        return results
        
        # Determine vector type
        if vector_width == 'float':
            vec_type = 'float'
            vec_size = 1
            vec_load = ''
        elif vector_width == 'float2':
            vec_type = 'float2'
            vec_size = 2
            vec_load = 'make_float2'
        elif vector_width == 'float4':
            vec_type = 'float4'
            vec_size = 4
            vec_load = 'make_float4'
        elif vector_width == 'dwordx4':
            vec_type = 'uint4'
            vec_size = 4
            vec_load = ''
        else:
            vec_type = 'float4'
            vec_size = 4
        
        kernel_code = f'''
/**
 * P2P Bandwidth Benchmark Kernels
 * 
 * Access pattern: {access_pattern}
 * Vector width: {vector_width} ({vec_size}x float)
 */

#include <hip/hip_runtime.h>
#include <cstdint>

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void p2p_read_kernel(
    {vec_type}* __restrict__ output,
    const {vec_type}* __restrict__ local_data,
    const {vec_type}* __restrict__ peer_data,
    unsigned int num_elements
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    // Access pattern
'''
        
        if access_pattern == 'sequential':
            kernel_code += '''
    // Sequential coalesced access
    for (unsigned int i = idx; i < num_elements; i += stride) {
        {vec_type} val = peer_data[i];
        if (output) output[i] = val;
    }}
'''.format(vec_type=vec_type)
        
        elif access_pattern == 'strided':
            kernel_code += '''
    // Strided access (stride = 64 elements)
    unsigned int stride_access = 64;
    for (unsigned int i = idx * stride_access; i < num_elements; i += stride * stride_access) {
        {vec_type} val = peer_data[i];
        if (output) output[i] = val;
    }}
'''.format(vec_type=vec_type)
        
        elif access_pattern == 'random':
            kernel_code += '''
    // Random access (pseudo-random via LCG)
    unsigned int state = idx + blockIdx.x * blockDim.x;
    for (unsigned int i = 0; i < num_elements / stride; i++) {{
        state = state * 1664525u + 1013904223u;
        unsigned int rand_idx = state % num_elements;
        {vec_type} val = peer_data[rand_idx];
        if (output) output[i] = val;
    }}
'''.format(vec_type=vec_type)
        
        kernel_code += '''
extern "C" int p2p_read_kernel(
    void* output,
    const void* local_data,
    const void* peer_data,
    unsigned int num_elements,
    hipStream_t stream)
{
    (void)hipGetLastError();
    unsigned int blocks = 256;
    hipLaunchKernelGGL(p2p_read_kernel, dim3(blocks), dim3(256), 0, stream,
                       ({vec_type}*)output,
                       (const {vec_type}*)local_data,
                       (const {vec_type}*)peer_data,
                       num_elements);
    return (int)hipGetLastError();
}}
'''.format(vec_type=vec_type)
        
        src_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, 'w') as f:
            f.write(kernel_code)
    
    def bench_kernel_launch_overhead(self, num_iterations=1000):
        """Measure kernel launch overhead without data transfer.
        
        Args:
            num_iterations: number of iterations
        """
        print("\n" + "="*70)
        print("  Kernel Launch Overhead Benchmark")
        print("="*70)
        
        # Simple empty kernel
        kernel_code = '''
#include <hip/hip_runtime.h>

__global__ void empty_kernel() {}

extern "C" int empty_kernel_launch(hipStream_t stream)
{
    (void)hipGetLastError();
    hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    return (int)hipGetLastError();
}
'''
        # Write and compile
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "empty_kernel.hip"
            so_path = Path(tmpdir) / "empty_kernel.so"
            
            with open(src_path, 'w') as f:
                f.write(kernel_code)
            
            try:
                subprocess.check_call([
                    "/opt/rocm/bin/hipcc",
                    "-O3", "--offload-arch=gfx906", "-std=c++17",
                    "-shared", "-fPIC",
                    "-o", str(so_path),
                    str(src_path),
                ])
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  WARNING: Failed to compile empty kernel")
                return {'launch_overhead_us': 0}
            
            lib = ctypes.CDLL(str(so_path))
            
            times = []
            for dev_id in self.device_ids:
                self.hip.set_device(dev_id)
                stream = self.hip.stream_create()
                
                # Warmup
                for _ in range(100):
                    lib.empty_kernel_launch(stream)
                self.hip.stream_synchronize(stream)
                
                # Measure
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    lib.empty_kernel_launch(stream)
                    self.hip.stream_synchronize(stream)
                    end = time.perf_counter()
                    times.append((end - start) * 1e6)
                
                self.hip.stream_destroy(stream)
            
            avg_overhead = np.mean(times)
            print(f"  Average kernel launch overhead: {avg_overhead:.2f} µs")
            
            return {'launch_overhead_us': avg_overhead}
    
    def analyze_latency_bandwidth_crossover(self, results):
        """Analyze whether 10KB payload is latency-bound or bandwidth-bound.
        
        Args:
            results: dictionary of benchmark results
        """
        print("\n" + "="*70)
        print("  Latency vs Bandwidth Analysis")
        print("="*70)
        
        theoretical_bandwidth_gbps = 25.6  # PCIe 4.0 x16 theoretical
        measured_p2p_bandwidth_gbps = 12.0  # Typical 2-hop P2P
        
        # Find 10KB results
        size_10kb = 10 * 1024
        if size_10kb in results:
            r = results[size_10kb]
            
            latency_us = r.get('memcpy_avg_latency_us', 0)
            bandwidth_gbps = r.get('memcpy_avg_bandwidth_gbps', 0)
            
            # Calculate time components
            transfer_time_us = (size_10kb / 1e9) / (bandwidth_gbps * 1e9) * 1e6 if bandwidth_gbps > 0 else 0
            fixed_overhead_us = latency_us - transfer_time_us
            
            # Determine if latency-bound
            is_latency_bound = transfer_time_us < fixed_overhead_us
            
            print(f"  10KB Payload Analysis:")
            print(f"    Measured latency: {latency_us:.2f} µs")
            print(f"    Measured bandwidth: {bandwidth_gbps:.2f} GB/s")
            print(f"    Transfer time (at measured BW): {transfer_time_us:.2f} µs")
            print(f"    Fixed overhead: {fixed_overhead_us:.2f} µs")
            print(f"    Status: {'LATENCY-BOUND' if is_latency_bound else 'BANDWIDTH-BOUND'}")
            
            return {
                'size': size_10kb,
                'latency_us': latency_us,
                'bandwidth_gbps': bandwidth_gbps,
                'transfer_time_us': transfer_time_us,
                'fixed_overhead_us': fixed_overhead_us,
                'is_latency_bound': is_latency_bound,
            }
        
        return None
    
    def run_full_benchmark(self, payload_sizes=None):
        """Run complete PCIe 4.0 bandwidth benchmark suite.
        
        Args:
            payload_sizes: list of payload sizes in bytes (default: standard set)
        """
        if payload_sizes is None:
            payload_sizes = [
                1 * 1024,      # 1 KB
                5 * 1024,      # 5 KB
                10 * 1024,     # 10 KB (typical allreduce)
                30 * 1024,     # 30 KB
                100 * 1024,    # 100 KB
                1 * 1024 * 1024,  # 1 MB
            ]
        
        print("="*70)
        print("  PCIe 4.0 P2P Bandwidth Benchmark Suite")
        print("  Hardware: 4x MI50 (gfx906), PCIe 4.0 x16, 2-hop")
        print("  Theoretical PCIe 4.0 x16: ~25.6 GB/s")
        print("="*70)
        
        self.results = {}
        
        # 1. Raw memcpy P2P bandwidth
        self.results['memcpy'] = self.bench_memcpy_peer(payload_sizes)
        
        # 2. Kernel launch overhead
        self.results['overhead'] = self.bench_kernel_launch_overhead()
        
        # 3. Build kernel library
        kernel_path = Path(__file__).parent.parent / "build" / "kernels" / "p2p_bandwidth_kernels.so"
        self._build_p2p_bandwidth_kernel(kernel_path)
        
        # 4. Kernel P2P read (sequential, different vector widths)
        for vw in ['float4', 'float', 'float2', 'dwordx4']:
            self.results[f'kernel_seq_{vw}'] = self.bench_kernel_p2p_read(
                payload_sizes, access_pattern='sequential', vector_width=vw)
        
        # 5. Kernel P2P read (strided, float4)
        self.results['kernel_strided'] = self.bench_kernel_p2p_read(
            payload_sizes, access_pattern='strided', vector_width='float4')
        
        # 6. Kernel P2P read (random, float4)
        self.results['kernel_random'] = self.bench_kernel_p2p_read(
            payload_sizes, access_pattern='random', vector_width='float4')
        
        # 7. Analysis
        self.results['analysis'] = self.analyze_latency_bandwidth_crossover(
            self.results['memcpy'])
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary report."""
        print("\n" + "="*70)
        print("  BENCHMARK SUMMARY REPORT")
        print("="*70)
        
        print("\n  Hardware Configuration:")
        print("    GPUs: 4x AMD Instinct MI50 (gfx906)")
        print("    Interconnect: PCIe 4.0 x16, 2-hop via switch")
        print(f"    Theoretical PCIe 4.0 x16: 25.6 GB/s")
        print(f"    Expected P2P (2-hop): ~12.0 GB/s")
        
        if 'memcpy' in self.results:
            print("\n  hipMemcpyPeerAsync Raw P2P Bandwidth:")
            print(f"  {'Payload':<10} | {'Bandwidth':<10} | {'% Theoretical':<12} | {'Latency':<10}")
            print(f"  {'(KB)':<10} | {'(GB/s)':<10} | {'(of 25.6 GB/s)':<12} | {'(µs)':<10}")
            print("  " + "-"*55)
            for size, r in self.results['memcpy'].items():
                bw = r['memcpy_avg_bandwidth_gbps']
                pct = (bw / 25.6) * 100
                lat = r['memcpy_avg_latency_us']
                print(f"  {size / 1e3:<10.1f} | {bw:<10.2f} | {pct:<12.1f} | {lat:<10.2f}")
        
        if 'overhead' in self.results:
            print("\n  Kernel Launch Overhead:")
            print(f"    Average: {self.results['overhead']['launch_overhead_us']:.2f} µs")
        
        # Vector width comparison
        print("\n  Vector Width Comparison (Sequential Access):")
        print(f"  {'Payload':<8} | {'float':<8} | {'float2':<8} | {'float4':<8} | {'dwordx4':<8}")
        print(f"  {'(KB)':<8} | {'(GB/s)':<8} | {'(GB/s)':<8} | {'(GB/s)':<8} | {'(GB/s)':<8}")
        print("  " + "-"*50)
        for size in [1*1024, 10*1024, 100*1024]:
            if size in self.results.get('memcpy', {}):
                row = f"  {size / 1e3:<8.1f} | "
                for vw in ['float', 'float2', 'float4', 'dwordx4']:
                    key = f'kernel_seq_{vw}'
                    if key in self.results and size in self.results[key]:
                        bw = self.results[key][size]['kernel_bandwidth_gbps']
                        row += f"{bw:<8.2f} | "
                    else:
                        row += f"{'N/A':<8} | "
                print(row)
        
        # Access pattern comparison
        print("\n  Access Pattern Comparison (10KB payload, float4):")
        patterns = {
            'kernel_seq_float4': 'Sequential',
            'kernel_strided': 'Strided',
            'kernel_random': 'Random'
        }
        size_10kb = 10 * 1024
        for key, name in patterns.items():
            if key in self.results and size_10kb in self.results[key]:
                bw = self.results[key][size_10kb]['kernel_bandwidth_gbps']
                lat = self.results[key][size_10kb]['kernel_avg_latency_us']
                print(f"    {name:<12}: {bw:6.2f} GB/s, {lat:7.2f} µs")
        
        if 'analysis' in self.results and self.results['analysis']:
            a = self.results['analysis']
            print("\n  10KB Allreduce Characterization (VAL-M2-PCIe-001):")
            print(f"    Status: {'LATENCY-BOUND' if a['is_latency_bound'] else 'BANDWIDTH-BOUND'}")
            print(f"    Measured latency: {a['latency_us']:.2f} µs")
            print(f"    Fixed overhead:   {a['fixed_overhead_us']:.2f} µs ({a['fixed_overhead_us']/a['latency_us']*100:.1f}%)")
            print(f"    Transfer time:    {a['transfer_time_us']:.2f} µs ({a['transfer_time_us']/a['latency_us']*100:.1f}%)")
            print(f"    Bandwidth utilization: {a['bandwidth_gbps']:.2f} GB/s ({a['bandwidth_gbps']/25.6*100:.1f}% of theoretical)")
        
        print("\n  Key Findings:")
        if 'analysis' in self.results and self.results['analysis']:
            a = self.results['analysis']
            if a['is_latency_bound']:
                print("    • 10KB allreduce is LATENCY-BOUND")
                print("    • Kernel optimization focus: reduce fixed overhead (launch, sync)")
                print("    • Vectorization provides marginal benefit for small payloads")
            else:
                print("    • 10KB allreduce is BANDWIDTH-BOUND")
                print("    • Kernel optimization focus: improve memory coalescing")
                print("    • Vectorized loads (float4, dwordx4) provide significant benefit")


def main():
    parser = argparse.ArgumentParser(description='PCIe 4.0 P2P Bandwidth Benchmark')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='GPU device IDs (comma-separated)')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Payload sizes in KB (comma-separated, e.g., "1,5,10,30,100,1024")')
    args = parser.parse_args()
    
    device_ids = [int(x) for x in args.gpus.split(',')]
    
    if args.sizes:
        payload_sizes = [int(x) * 1024 for x in args.sizes.split(',')]
    else:
        payload_sizes = None
    
    benchmark = PCIebandwidthBenchmark(device_ids)
    results = benchmark.run_full_benchmark(payload_sizes)
    benchmark.print_summary()
    
    return results


if __name__ == '__main__':
    main()
