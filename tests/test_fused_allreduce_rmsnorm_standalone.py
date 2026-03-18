#!/usr/bin/env python3
"""
Standalone test for fused P2P allreduce + RMSNorm kernel.

This test runs the fused kernel directly without relying on the reference
implementation (kernel_p2p_allreduce.so) for comparison.

Tests:
1. Fused kernel library loads and exports correct functions (VAL-FUSE-005)
2. Fused kernel runs on 4 GPUs without errors
3. Multi-GPU output consistency (VAL-FUSE-007)
4. Numerical correctness via host-side reference implementation

Usage:
    python3 tests/test_fused_allreduce_rmsnorm_standalone.py
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError


def host_reference_fused(partial0, partial1, partial2, partial3, weight, eps=1e-6):
    """Reference implementation: sum partials + RMSNorm."""
    # Sum all partials (FP32 accumulation)
    summed = (partial0.astype(np.float32) + 
              partial1.astype(np.float32) + 
              partial2.astype(np.float32) + 
              partial3.astype(np.float32))
    
    # RMSNorm
    sum_sq = np.sum(summed ** 2)
    rms = np.sqrt(sum_sq / len(summed) + eps)
    normalized = summed / rms
    
    # Apply weight
    result = (normalized * weight.astype(np.float32)).astype(np.float16)
    return result


def test_fused_kernel_standalone(tp_size=4, dim=5120):
    """Test fused kernel runs correctly on all GPUs.
    
    Validates:
    - VAL-FUSE-005: C dispatch path integration (library loads)
    - VAL-FUSE-007: Multi-GPU output consistency
    
    Returns: (passed, results_dict)
    """
    print(f"\n--- Standalone fused kernel test: TP={tp_size}, dim={dim} ---")
    
    results = {
        'library_loaded': False,
        'kernel_runs': False,
        'multi_gpu_consistent': False,
        'numerical_correct': False,
        'max_abs_error': None
    }
    
    # Load fused kernel library
    build_dir = Path("/opt/mi50grad/build/kernels")
    fused_so = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
    
    if not fused_so.exists():
        print(f"  FAIL: {fused_so} not found")
        return False, results
    
    try:
        fused_lib = ctypes.CDLL(str(fused_so))
        
        # Check functions exist
        if not hasattr(fused_lib, 'kernel_p2p_allreduce_rmsnorm_tp4'):
            print(f"  FAIL: kernel_p2p_allreduce_rmsnorm_tp4 not exported")
            return False, results
            
        # Set function signature
        fused_lib.kernel_p2p_allreduce_rmsnorm_tp4.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # partial_local
            ctypes.c_void_p,  # partial_peer0
            ctypes.c_void_p,  # partial_peer1
            ctypes.c_void_p,  # partial_peer2
            ctypes.c_void_p,  # weight
            ctypes.c_uint,    # dim
            ctypes.c_uint,    # batch_size
            ctypes.c_float,   # eps
            ctypes.c_void_p,  # stream
        ]
        fused_lib.kernel_p2p_allreduce_rmsnorm_tp4.restype = ctypes.c_int
        
        results['library_loaded'] = True
        print(f"  PASS: Fused kernel library loaded")
        
    except Exception as e:
        print(f"  FAIL: Library load error: {e}")
        return False, results
    
    # Initialize GPUs
    devices = [GPUDevice(i) for i in range(tp_size)]
    hip = devices[0].hip
    
    # Enable P2P access between all GPUs
    for i in range(tp_size):
        hip.set_device(i)
        for j in range(tp_size):
            if i != j:
                try:
                    hip.device_enable_peer_access(j)
                except Exception as e:
                    print(f"  WARNING: Could not enable P2P from GPU {i} to GPU {j}: {e}")
    
    # Create streams
    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())
    
    # Generate test data
    rng = np.random.default_rng(42)
    partials = [rng.random(dim).astype(np.float16) * 2 - 1 for _ in range(tp_size)]
    weight = rng.random(dim).astype(np.float16)
    
    # Allocate GPU buffers
    partial_ptrs = []
    weight_ptrs = []
    output_ptrs = []
    
    for i in range(tp_size):
        hip.set_device(i)
        partial_ptrs.append(devices[i].malloc(dim * 2))
        devices[i].upload(partial_ptrs[i], partials[i].tobytes())
        weight_ptrs.append(devices[i].malloc(dim * 2))
        devices[i].upload(weight_ptrs[i], weight.tobytes())
        output_ptrs.append(devices[i].malloc(dim * 2))
    
    # Compute host reference
    expected = host_reference_fused(partials[0], partials[1], partials[2], partials[3], weight)
    
    # Launch fused kernel on each GPU
    try:
        for i in range(tp_size):
            hip.set_device(i)
            peer_indices = [j for j in range(tp_size) if j != i]
            
            err = fused_lib.kernel_p2p_allreduce_rmsnorm_tp4(
                ctypes.c_void_p(output_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[peer_indices[0]]),
                ctypes.c_void_p(partial_ptrs[peer_indices[1]]),
                ctypes.c_void_p(partial_ptrs[peer_indices[2]]),
                ctypes.c_void_p(weight_ptrs[i]),
                ctypes.c_uint(dim),
                ctypes.c_uint(1),  # batch_size
                ctypes.c_float(1e-6),
                ctypes.c_void_p(streams[i])
            )
            
            if err != 0:
                print(f"  FAIL: Kernel returned error {err} on GPU {i}")
                raise HIPError(f"Kernel error: {err}")
        
        results['kernel_runs'] = True
        print(f"  PASS: Kernel executed on all {tp_size} GPUs")
        
    except Exception as e:
        print(f"  FAIL: Kernel execution error: {e}")
        # Cleanup
        for i in range(tp_size):
            hip.set_device(i)
            hip.stream_destroy(streams[i])
            hip.free(partial_ptrs[i])
            hip.free(weight_ptrs[i])
            hip.free(output_ptrs[i])
        for d in devices:
            d.cleanup()
        return False, results
    
    # Sync and download results
    gpu_outputs = []
    for i in range(tp_size):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
        raw = devices[i].download(output_ptrs[i], dim * 2)
        gpu_outputs.append(np.frombuffer(raw, dtype=np.float16).copy())
    
    # Check multi-GPU consistency (VAL-FUSE-007)
    max_diff = 0.0
    for i in range(1, tp_size):
        diff = float(np.max(np.abs(
            gpu_outputs[0].astype(np.float32) - gpu_outputs[i].astype(np.float32))))
        max_diff = max(max_diff, diff)
    
    results['multi_gpu_consistent'] = max_diff < 1e-3
    if results['multi_gpu_consistent']:
        print(f"  PASS (VAL-FUSE-007): Multi-GPU consistency OK (max diff: {max_diff:.4e})")
    else:
        print(f"  FAIL (VAL-FUSE-007): Multi-GPU inconsistent (max diff: {max_diff:.4e})")
    
    # Check numerical correctness vs host reference
    max_abs_error = float(np.max(np.abs(
        expected.astype(np.float32) - gpu_outputs[0].astype(np.float32))))
    results['max_abs_error'] = max_abs_error
    
    results['numerical_correct'] = max_abs_error < 5e-3
    if results['numerical_correct']:
        print(f"  PASS (VAL-FUSE-001): Numerical correctness OK (max abs error: {max_abs_error:.4e})")
    else:
        print(f"  FAIL (VAL-FUSE-001): Numerical error too large (max abs error: {max_abs_error:.4e})")
    
    # Cleanup
    for i in range(tp_size):
        hip.set_device(i)
        hip.stream_destroy(streams[i])
        hip.free(partial_ptrs[i])
        hip.free(weight_ptrs[i])
        hip.free(output_ptrs[i])
    for d in devices:
        d.cleanup()
    
    all_passed = (results['library_loaded'] and 
                  results['kernel_runs'] and 
                  results['multi_gpu_consistent'] and 
                  results['numerical_correct'])
    
    return all_passed, results


def test_dimension_alignment():
    """Test non-aligned dimensions (VAL-FUSE-004)."""
    print(f"\n--- Dimension alignment test ---")
    
    test_dims = [5100, 5122, 5125]
    all_passed = True
    
    for dim in test_dims:
        print(f"\n  Testing dim={dim}...")
        passed, results = test_fused_kernel_standalone(tp_size=4, dim=dim)
        if passed:
            print(f"    PASS: dim={dim} handled correctly")
        else:
            print(f"    FAIL: dim={dim} failed")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 70)
    print("Standalone Fused P2P Allreduce + RMSNorm Kernel Test")
    print("Validates: VAL-FUSE-001, VAL-FUSE-004, VAL-FUSE-005, VAL-FUSE-007")
    print("=" * 70)
    
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs available: {n_gpus}")
    
    if n_gpus < 4:
        print("ERROR: Need 4 GPUs for TP=4 fused kernel test")
        sys.exit(1)
    
    all_pass = True
    
    # Test 1: Standard dimension (5120)
    passed, results = test_fused_kernel_standalone(tp_size=4, dim=5120)
    if not passed:
        all_pass = False
    
    # Test 2: Non-aligned dimensions
    if not test_dimension_alignment():
        all_pass = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_pass:
        print("All tests PASSED")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
