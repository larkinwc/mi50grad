#!/usr/bin/env python3
"""
Test INT8-compressed fused GEMV+P2P allreduce+RMSNorm kernel.

This test validates the compressed fused kernel against the uncompressed version:
  - gemv_int4_p2p_allreduce_rmsnorm_compressed.hip (INT8-compressed P2P)
  - gemv_int4_p2p_allreduce_rmsnorm.hip (uncompressed FP16 P2P)

Validates:
  VAL-COMP-FUSED-001: Compressed fused kernel compiles and loads successfully
  VAL-COMP-FUSED-002: Cosine similarity >= 0.99 vs uncompressed fused kernel
  VAL-COMP-FUSED-003: Max absolute error < 5e-3 (INT8 quantization noise budget)

Usage:
    python3 tests/test_compressed_fused_kernel.py
    
Deployment:
    rsync -avz --delete . root@192.168.1.198:/opt/mi50grad/
    ssh root@192.168.1.198 'docker stop vllm-mobydick'
    ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/test_compressed_fused_kernel.py"'
"""

import sys
import os
import ctypes
import multiprocessing as mp
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime
from src.kernels.launcher import build_hip_hsaco


# ============================================================================
# Build kernels
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("Compressed Fused Kernel Test")
print("=" * 72)

# Build uncompressed reference kernel
print("\nBuilding uncompressed fused kernel (reference)...")
hip_uncompressed = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_p2p_allreduce_rmsnorm.hip")
so_uncompressed = str(BUILD_DIR / "gemv_int4_p2p_allreduce_rmsnorm.so")
try:
    import subprocess
    result = subprocess.run(
        f"hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o {so_uncompressed} {hip_uncompressed}",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    print("  Uncompressed kernel SO built successfully")
except Exception as e:
    print(f"  ERROR building uncompressed kernel SO: {e}")
    sys.exit(1)

# Build compressed kernel
print("\nBuilding compressed fused kernel...")
hip_compressed = str(PROJECT_ROOT / "src" / "kernels" / "gemv_int4_p2p_allreduce_rmsnorm_compressed.hip")
so_compressed = str(BUILD_DIR / "gemv_int4_p2p_allreduce_rmsnorm_compressed.so")
try:
    result = subprocess.run(
        f"hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o {so_compressed} {hip_compressed}",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    print("  Compressed kernel SO built successfully")
except Exception as e:
    print(f"  ERROR building compressed kernel SO: {e}")
    sys.exit(1)

# ============================================================================
# Load kernels
# ============================================================================
print("\nLoading kernels...")

# Load uncompressed kernel
uncompressed_lib = ctypes.CDLL(so_uncompressed)
uncompressed_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
]
uncompressed_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4.restype = ctypes.c_int

# Load compressed kernel
compressed_lib = ctypes.CDLL(so_compressed)
compressed_lib.gemv_int4_p2p_allreduce_rmsnorm_compressed_tp4.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_float, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_void_p,
]
compressed_lib.gemv_int4_p2p_allreduce_rmsnorm_compressed_tp4.restype = ctypes.c_int

print("  Kernels loaded successfully")


# ============================================================================
# Helper functions
# ============================================================================

def quantize_weights_gptq(W_fp32, group_size=128):
    """
    Simulate GPTQ quantization (unsigned INT4, 0-15 range).
    W_fp32: [K, N] float32
    Returns: B_q4[K/8, N] uint32, scales[K/gs, N] float16, zeros[K/gs, N] float16
    """
    K, N = W_fp32.shape
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float32)
    zeros  = np.zeros((n_groups, N), dtype=np.float32)
    q4_mat = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        ks = g * group_size
        ke = ks + group_size
        grp = W_fp32[ks:ke, :]

        w_min = grp.min(axis=0)
        w_max = grp.max(axis=0)

        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0.0, 1.0, scale)
        zero  = -w_min / scale

        q = np.round((grp - w_min[np.newaxis, :]) / scale[np.newaxis, :])
        q = np.clip(q, 0, 15).astype(np.uint8)

        scales[g, :] = scale
        zeros[g, :]  = zero
        q4_mat[ks:ke, :] = q

    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_mat[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales.astype(np.float16), zeros.astype(np.float16)


def run_uncompressed_kernel(tp_rank, K, N, group_size, buffers, results_dict, launch_barrier):
    """Run uncompressed fused kernel on one GPU."""
    try:
        dev = GPUDevice(tp_rank)
        stream = dev.get_stream()
        
        # Allocate P2P pointers
        partial_ptrs = (ctypes.c_uint64 * 4)()
        for i in range(4):
            partial_ptrs[i] = buffers[f'd_partial_{i}']
        
        # Zero counters
        dev.upload(buffers[f'd_wg_write_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        dev.upload(buffers[f'd_wg_done_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        
        # Wait for all GPUs to be ready
        launch_barrier.wait()
        
        # Launch uncompressed kernel
        err = uncompressed_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4(
            buffers[f'd_output_{tp_rank}'],
            buffers[f'd_A_{tp_rank}'],
            buffers[f'd_B_{tp_rank}'],
            buffers[f'd_scales_{tp_rank}'],
            buffers[f'd_zeros_{tp_rank}'],
            buffers[f'd_partial_{tp_rank}'],
            partial_ptrs[(tp_rank + 1) % 4],
            partial_ptrs[(tp_rank + 2) % 4],
            partial_ptrs[(tp_rank + 3) % 4],
            buffers[f'd_weight_{tp_rank}'],
            buffers[f'd_wg_sum_sq_{tp_rank}'],
            buffers[f'd_wg_write_{tp_rank}'],
            buffers[f'd_wg_done_{tp_rank}'],
            K, N, N, group_size, 1e-6,
            tp_rank, 4,
            stream
        )
        
        if err != 0:
            results_dict[f'error_{tp_rank}'] = f"HIP error {err}"
            return
        
        # Sync and retrieve output
        dev.synchronize()
        output_bytes = dev.download(buffers[f'd_output_{tp_rank}'], (N // 4) * 2)
        output = np.frombuffer(output_bytes, dtype=np.float16)
        
        results_dict[f'output_{tp_rank}'] = output
        results_dict[f'tp_rank_{tp_rank}'] = tp_rank
        
    except Exception as e:
        results_dict[f'error_{tp_rank}'] = str(e)


def run_compressed_kernel(tp_rank, K, N, group_size, buffers, results_dict, launch_barrier):
    """Run compressed fused kernel on one GPU."""
    try:
        dev = GPUDevice(tp_rank)
        stream = dev.get_stream()
        
        cols_per_gpu = (N + 3) // 4
        num_blocks = (cols_per_gpu + 31) // 32
        compressed_size = cols_per_gpu + num_blocks * 2
        
        # Allocate P2P pointers for compressed buffers
        compressed_ptrs = (ctypes.c_uint64 * 4)()
        for i in range(4):
            compressed_ptrs[i] = buffers[f'd_compressed_{i}']
        
        # Zero counters
        dev.upload(buffers[f'd_wg_write_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        dev.upload(buffers[f'd_wg_done_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        
        # Wait for all GPUs to be ready
        launch_barrier.wait()
        
        # Launch compressed kernel
        err = compressed_lib.gemv_int4_p2p_allreduce_rmsnorm_compressed_tp4(
            buffers[f'd_output_{tp_rank}'],
            buffers[f'd_A_{tp_rank}'],
            buffers[f'd_B_{tp_rank}'],
            buffers[f'd_scales_{tp_rank}'],
            buffers[f'd_zeros_{tp_rank}'],
            buffers[f'd_partial_{tp_rank}'],
            buffers[f'd_compressed_{tp_rank}'],
            compressed_ptrs[(tp_rank + 1) % 4],
            compressed_ptrs[(tp_rank + 2) % 4],
            compressed_ptrs[(tp_rank + 3) % 4],
            buffers[f'd_weight_{tp_rank}'],
            buffers[f'd_wg_sum_sq_{tp_rank}'],
            buffers[f'd_wg_write_{tp_rank}'],
            buffers[f'd_wg_done_{tp_rank}'],
            K, N, N, group_size, 1e-6,
            tp_rank, 4,
            stream
        )
        
        if err != 0:
            results_dict[f'error_{tp_rank}'] = f"HIP error {err}"
            return
        
        # Sync and retrieve output
        dev.synchronize()
        output_bytes = dev.download(buffers[f'd_output_{tp_rank}'], (N // 4) * 2)
        output = np.frombuffer(output_bytes, dtype=np.float16)
        
        results_dict[f'output_compressed_{tp_rank}'] = output
        results_dict[f'tp_rank_{tp_rank}'] = tp_rank
        
    except Exception as e:
        results_dict[f'error_{tp_rank}'] = str(e)


def run_test(tp_rank, K, N, group_size, buffers, results_dict, launch_barrier):
    """Run both uncompressed and compressed kernels."""
    try:
        dev = GPUDevice(tp_rank)
        stream = dev.get_stream()
        
        cols_per_gpu = (N + 3) // 4
        num_blocks = (cols_per_gpu + 31) // 32
        compressed_size = cols_per_gpu + num_blocks * 2
        
        # Allocate P2P pointers
        partial_ptrs = (ctypes.c_uint64 * 4)()
        compressed_ptrs = (ctypes.c_uint64 * 4)()
        for i in range(4):
            partial_ptrs[i] = buffers[f'd_partial_{i}']
            compressed_ptrs[i] = buffers[f'd_compressed_{i}']
        
        # Zero counters for uncompressed
        dev.upload(buffers[f'd_wg_write_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        dev.upload(buffers[f'd_wg_done_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        
        # Wait for all GPUs to be ready
        launch_barrier.wait()
        
        # Launch uncompressed kernel first
        err = uncompressed_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4(
            buffers[f'd_output_{tp_rank}'],
            buffers[f'd_A_{tp_rank}'],
            buffers[f'd_B_{tp_rank}'],
            buffers[f'd_scales_{tp_rank}'],
            buffers[f'd_zeros_{tp_rank}'],
            buffers[f'd_partial_{tp_rank}'],
            partial_ptrs[(tp_rank + 1) % 4],
            partial_ptrs[(tp_rank + 2) % 4],
            partial_ptrs[(tp_rank + 3) % 4],
            buffers[f'd_weight_{tp_rank}'],
            buffers[f'd_wg_sum_sq_{tp_rank}'],
            buffers[f'd_wg_write_{tp_rank}'],
            buffers[f'd_wg_done_{tp_rank}'],
            K, N, N, group_size, 1e-6,
            tp_rank, 4,
            stream
        )
        
        if err != 0:
            results_dict[f'error_uncompressed_{tp_rank}'] = f"Uncompressed HIP error {err}"
            return
        
        dev.synchronize()
        output_bytes = dev.download(buffers[f'd_output_{tp_rank}'], cols_per_gpu * 2)
        output_uncompressed = np.frombuffer(output_bytes, dtype=np.float16)
        
        # Zero counters for compressed
        dev.upload(buffers[f'd_wg_write_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        dev.upload(buffers[f'd_wg_done_{tp_rank}'], np.array([0], dtype=np.uint32).tobytes())
        
        # Second barrier before compressed launch
        launch_barrier.wait()
        
        # Launch compressed kernel
        err = compressed_lib.gemv_int4_p2p_allreduce_rmsnorm_compressed_tp4(
            buffers[f'd_output_compressed_{tp_rank}'],
            buffers[f'd_A_{tp_rank}'],
            buffers[f'd_B_{tp_rank}'],
            buffers[f'd_scales_{tp_rank}'],
            buffers[f'd_zeros_{tp_rank}'],
            buffers[f'd_partial_{tp_rank}'],
            buffers[f'd_compressed_{tp_rank}'],
            compressed_ptrs[(tp_rank + 1) % 4],
            compressed_ptrs[(tp_rank + 2) % 4],
            compressed_ptrs[(tp_rank + 3) % 4],
            buffers[f'd_weight_{tp_rank}'],
            buffers[f'd_wg_sum_sq_{tp_rank}'],
            buffers[f'd_wg_write_{tp_rank}'],
            buffers[f'd_wg_done_{tp_rank}'],
            K, N, N, group_size, 1e-6,
            tp_rank, 4,
            stream
        )
        
        if err != 0:
            results_dict[f'error_compressed_{tp_rank}'] = f"Compressed HIP error {err}"
            return
        
        dev.synchronize()
        output_bytes = dev.download(buffers[f'd_output_compressed_{tp_rank}'], cols_per_gpu * 2)
        output_compressed = np.frombuffer(output_bytes, dtype=np.float16)
        
        results_dict[f'output_uncompressed_{tp_rank}'] = output_uncompressed
        results_dict[f'output_compressed_{tp_rank}'] = output_compressed
        results_dict[f'tp_rank_{tp_rank}'] = tp_rank
        
    except Exception as e:
        results_dict[f'error_{tp_rank}'] = str(e)


def allocate_buffers(devices, K, N, group_size):
    """Allocate buffers for all 4 GPUs."""
    buffers = {}
    
    # Generate random test data
    np.random.seed(42)
    A_h16 = np.random.randn(K).astype(np.float16) * 0.1
    W_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1
    B_q4, scales, zeros = quantize_weights_gptq(W_fp32, group_size)
    weight_h16 = np.ones(N, dtype=np.float16)
    
    cols_per_gpu = (N + 3) // 4
    num_blocks = (cols_per_gpu + 31) // 32
    compressed_size = cols_per_gpu + num_blocks * 2
    
    # Partition weights
    B_q4_parts = np.array_split(B_q4, 4, axis=1)
    scales_parts = np.array_split(scales, 4, axis=1)
    zeros_parts = np.array_split(zeros, 4, axis=1)
    weight_parts = np.array_split(weight_h16, 4)
    
    for tp_rank, dev in enumerate(devices):
        d_A = dev.malloc(A_h16.nbytes)
        dev.upload(d_A, A_h16.tobytes())
        buffers[f'd_A_{tp_rank}'] = d_A
        
        d_B = dev.malloc(B_q4_parts[tp_rank].nbytes)
        dev.upload(d_B, B_q4_parts[tp_rank].tobytes())
        buffers[f'd_B_{tp_rank}'] = d_B
        
        d_scales = dev.malloc(scales_parts[tp_rank].nbytes)
        dev.upload(d_scales, scales_parts[tp_rank].tobytes())
        buffers[f'd_scales_{tp_rank}'] = d_scales
        
        d_zeros = dev.malloc(zeros_parts[tp_rank].nbytes)
        dev.upload(d_zeros, zeros_parts[tp_rank].tobytes())
        buffers[f'd_zeros_{tp_rank}'] = d_zeros
        
        # Allocate partial_local (full N size for P2P)
        d_partial = dev.malloc(N * 2)
        dev.upload(d_partial, np.zeros(N, dtype=np.float16).tobytes())
        buffers[f'd_partial_{tp_rank}'] = d_partial
        
        # Allocate compressed buffer
        d_compressed = dev.malloc(compressed_size)
        dev.upload(d_compressed, np.zeros(compressed_size, dtype=np.uint8).tobytes())
        buffers[f'd_compressed_{tp_rank}'] = d_compressed
        
        d_weight = dev.malloc(N * 2)
        dev.upload(d_weight, weight_h16.tobytes())
        buffers[f'd_weight_{tp_rank}'] = d_weight
        
        d_output = dev.malloc(cols_per_gpu * 2)
        buffers[f'd_output_{tp_rank}'] = d_output
        
        d_output_compressed = dev.malloc(cols_per_gpu * 2)
        buffers[f'd_output_compressed_{tp_rank}'] = d_output_compressed
        
        num_wgs = (cols_per_gpu + 16 - 1) // 16
        d_sum_sq = dev.malloc(num_wgs * 4)
        buffers[f'd_wg_sum_sq_{tp_rank}'] = d_sum_sq
        
        d_write = dev.malloc(4)
        dev.upload(d_write, np.array([0], dtype=np.uint32).tobytes())
        buffers[f'd_wg_write_{tp_rank}'] = d_write
        
        d_done = dev.malloc(4)
        dev.upload(d_done, np.array([0], dtype=np.uint32).tobytes())
        buffers[f'd_wg_done_{tp_rank}'] = d_done
    
    return buffers


# ============================================================================
# Main test
# ============================================================================
print("\n" + "=" * 72)
print("TEST: Compressed vs Uncompressed Fused Kernel")
print("=" * 72)

# Test parameters (FFN down projection shape for Qwen 27B)
K = 17408  # intermediate_size
N = 5120   # hidden_size
group_size = 128

print(f"\nTest parameters:")
print(f"  K (input dim): {K}")
print(f"  N (output dim): {N}")
print(f"  group_size: {group_size}")
print(f"  TP size: 4")

# Initialize devices
print("\nInitializing 4 GPUs...")
try:
    devices = [GPUDevice(i) for i in range(4)]
    print("  GPUs initialized successfully")
except Exception as e:
    print(f"  ERROR initializing GPUs: {e}")
    sys.exit(1)

# Allocate buffers
print("\nAllocating buffers...")
buffers = allocate_buffers(devices, K, N, group_size)
print(f"  Buffers allocated")

# Create threading barrier and results dict
results_dict = {}
launch_barrier = threading.Barrier(4, timeout=30)

# Create and start threads
print(f"\nLaunching 4 threads (GPU 0-3)...")
threads = []
for tp_rank in range(4):
    args = (tp_rank, K, N, group_size, buffers, results_dict, launch_barrier)
    t = threading.Thread(target=run_test, args=args)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join(timeout=60)

# Check for errors
errors = []
for tp_rank in range(4):
    for key in [f'error_{tp_rank}', f'error_uncompressed_{tp_rank}', f'error_compressed_{tp_rank}']:
        error = results_dict.get(key)
        if error:
            errors.append({'tp_rank': tp_rank, 'key': key, 'error': error})

if errors:
    print(f"\n  ERROR: {len(errors)} thread(s) failed:")
    for err in errors:
        print(f"    GPU{err['tp_rank']} ({err['key']}): {err['error']}")
    sys.exit(1)

# Collect results
print("\nCollecting results...")
all_pass = True
for tp_rank in range(4):
    if f'output_uncompressed_{tp_rank}' in results_dict and f'output_compressed_{tp_rank}' in results_dict:
        output_uncompressed = results_dict[f'output_uncompressed_{tp_rank}']
        output_compressed = results_dict[f'output_compressed_{tp_rank}']
        
        # Compute cosine similarity
        dot_product = np.dot(output_uncompressed.astype(np.float32), output_compressed.astype(np.float32))
        norm_uncompressed = np.linalg.norm(output_uncompressed.astype(np.float32))
        norm_compressed = np.linalg.norm(output_compressed.astype(np.float32))
        cosine_sim = dot_product / (norm_uncompressed * norm_compressed)
        
        # Compute max absolute error
        max_abs_error = np.max(np.abs(output_uncompressed.astype(np.float32) - output_compressed.astype(np.float32)))
        
        print(f"  GPU{tp_rank}: cosine_sim = {cosine_sim:.6f}, max_abs_error = {max_abs_error:.6e}")
        
        if cosine_sim < 0.99:
            print(f"    FAIL: Cosine similarity < 0.99")
            all_pass = False
        
        if max_abs_error > 5e-3:
            print(f"    FAIL: Max absolute error > 5e-3")
            all_pass = False
    else:
        print(f"  GPU{tp_rank}: Missing results")
        all_pass = False

# Cleanup
print(f"\nCleaning up...")
for tp_rank in range(4):
    dev = devices[tp_rank]
    for key in buffers:
        if key.endswith(f'_{tp_rank}'):
            try:
                dev.free(buffers[key])
            except:
                pass

# Summary
print("\n" + "=" * 72)
if all_pass:
    print("RESULT: ALL TESTS PASSED")
    print("  - Cosine similarity >= 0.99 for all GPUs")
    print("  - Max absolute error < 5e-3 for all GPUs")
    print("  VAL-COMP-FUSED-001: PASSED (kernel compiles and loads)")
    print("  VAL-COMP-FUSED-002: PASSED (cosine_sim >= 0.99)")
    print("  VAL-COMP-FUSED-003: PASSED (max_abs_error < 5e-3)")
else:
    print("RESULT: TESTS FAILED")
    print("  See errors above")
    sys.exit(1)
print("=" * 72)
