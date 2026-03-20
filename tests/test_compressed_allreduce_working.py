#!/usr/bin/env python3
"""
Working test for INT8-compressed kernel P2P allreduce.
Uses threading to launch all 4 GPU kernels simultaneously.
"""

import sys
import ctypes
import threading
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime

def test_compressed_correctness():
    """Test compressed allreduce correctness with simultaneous multi-GPU launch."""
    print("=" * 70)
    print("INT8-Compressed P2P Allreduce Correctness Test")
    print("=" * 70)
    
    print("\nInitializing HIP...")
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"  GPUs: {n_gpus}")
    
    if n_gpus < 4:
        print(f"  ERROR: Need 4 GPUs")
        return False
    
    print("\nLoading kernel...")
    build_path = Path(__file__).parent.parent / "build" / "kernels" / "kernel_p2p_allreduce_compressed.so"
    if not build_path.exists():
        print(f"  ERROR: {build_path} not found")
        return False
    
    lib = ctypes.CDLL(str(build_path))
    
    # Set up function signatures
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
    
    print("  Kernel loaded")
    
    # Test parameters
    num_elems = 1024  # Small size for quick test
    num_blocks = (num_elems + 31) // 32
    compressed_size = num_elems + num_blocks * 2
    
    print(f"\nTest parameters:")
    print(f"  num_elems: {num_elems}")
    print(f"  num_blocks: {num_blocks}")
    print(f"  compressed_size: {compressed_size} bytes")
    
    # Generate test data
    print(f"\nGenerating test data...")
    rng = np.random.default_rng(42)
    partials = [rng.random(num_elems).astype(np.float16) * 2 - 1 for _ in range(4)]
    hidden0 = rng.random(num_elems).astype(np.float16) * 2 - 1
    
    # Reference: uncompressed allreduce
    print(f"\nComputing reference (uncompressed allreduce)...")
    ref = hidden0.astype(np.float32).copy()
    for p in partials:
        ref += p.astype(np.float32)
    ref = ref.astype(np.float16)
    
    # Create devices and allocate buffers
    print(f"\nAllocating GPU buffers...")
    devices = [GPUDevice(i) for i in range(4)]
    
    partial_ptrs = []
    compressed_ptrs = []
    hidden_ptrs = []
    
    for i in range(4):
        hip.set_device(i)
        
        # Allocate partial (FP16)
        partial_ptr = hip.malloc(num_elems * 2)
        hip.memcpy_h2d(partial_ptr, partials[i].tobytes(), num_elems * 2)
        partial_ptrs.append(partial_ptr)
        
        # Allocate compressed buffer (INT8 + scales)
        compressed_ptr = hip.malloc(compressed_size)
        compressed_ptrs.append(compressed_ptr)
        
        # Allocate hidden (FP16)
        hidden_ptr = hip.malloc(num_elems * 2)
        hip.memcpy_h2d(hidden_ptr, hidden0.tobytes(), num_elems * 2)
        hidden_ptrs.append(hidden_ptr)
    
    print(f"  Buffers allocated")
    
    # Create streams
    print(f"\nCreating streams...")
    streams = []
    for i in range(4):
        hip.set_device(i)
        streams.append(hip.stream_create())
    
    # Enable P2P
    print(f"Enabling P2P access...")
    for i in range(4):
        hip.set_device(i)
        for j in range(4):
            if i != j:
                try:
                    hip.device_enable_peer_access(j)
                except:
                    pass
    print(f"  P2P enabled")
    
    # Launch kernels using two-phase approach
    print(f"\nPhase 1: Quantize (sequential)...")
    for i in range(4):
        hip.set_device(i)
        err = lib.kernel_p2p_allreduce_compressed_quantize(
            partial_ptrs[i],
            compressed_ptrs[i],
            num_elems,
            num_blocks,
            ctypes.c_void_p(streams[i])
        )
        if err != 0:
            print(f"  ERROR: GPU{i} quantize failed: {err}")
            return False
        print(f"  GPU{i}: quantize launched")
    
    # Sync all streams after quantize
    print(f"\nSyncing all GPUs after quantize...")
    for i in range(4):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
    print(f"  All GPUs synced")
    
    # Get peer compressed pointers
    # These are the actual device pointers from peer GPUs
    compressed_peer_ptrs = []
    for i in range(4):
        peers = [(i + j) % 4 for j in range(1, 4)]
        compressed_peer_ptrs.append([compressed_ptrs[p] for p in peers])
    
    # Phase 2: Read (all GPUs simultaneously via threading)
    print(f"\nPhase 2: Read (simultaneous launch via threading)...")
    results = [None] * 4
    errors = [None] * 4
    
    def run_read_kernel(gpu_id):
        try:
            hip.set_device(gpu_id)
            err = lib.kernel_p2p_allreduce_compressed_residual_tp4_read(
                hidden_ptrs[gpu_id],
                compressed_ptrs[gpu_id],
                compressed_peer_ptrs[gpu_id][0],
                compressed_peer_ptrs[gpu_id][1],
                compressed_peer_ptrs[gpu_id][2],
                num_elems,
                num_blocks,
                ctypes.c_void_p(streams[gpu_id])
            )
            errors[gpu_id] = err
            
            if err == 0:
                hip.stream_synchronize(streams[gpu_id])
                # Download result
                buf = ctypes.create_string_buffer(num_elems * 2)
                hip.memcpy_d2h(buf, hidden_ptrs[gpu_id], num_elems * 2)
                results[gpu_id] = np.frombuffer(buf, dtype=np.float16).copy()
        except Exception as e:
            errors[gpu_id] = f"Exception: {e}"
    
    # Launch all 4 GPUs simultaneously
    threads = []
    t0 = time.time()
    for i in range(4):
        t = threading.Thread(target=run_read_kernel, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join(timeout=30)
    
    elapsed = time.time() - t0
    print(f"  Read phase completed in {elapsed*1000:.1f}ms")
    
    # Check for errors
    has_error = False
    for i in range(4):
        if errors[i]:
            print(f"  ERROR: GPU{i}: {errors[i]}")
            has_error = True
    
    if has_error:
        print(f"\nFAILED: Errors during read phase")
        # Cleanup
        for i in range(4):
            hip.set_device(i)
            hip.free(partial_ptrs[i])
            hip.free(compressed_ptrs[i])
            hip.free(hidden_ptrs[i])
            hip.stream_destroy(streams[i])
        for d in devices:
            d.cleanup()
        return False
    
    # Check results consistency
    print(f"\nResults consistency:")
    for i in range(1, 4):
        diff = float(np.max(np.abs(
            results[0].astype(np.float32) - results[i].astype(np.float32))))
        print(f"  GPU{i} vs GPU0: {diff:.4e}")
    
    # Compute cosine similarity vs reference
    cos_sim = float(np.dot(results[0].astype(np.float32), ref.astype(np.float32)) / \
              (np.linalg.norm(results[0].astype(np.float32)) * np.linalg.norm(ref.astype(np.float32))))
    
    print(f"\nCosine similarity vs reference: {cos_sim:.6f}")
    
    # Cleanup
    print(f"\nCleaning up...")
    for i in range(4):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(compressed_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()
    
    # Report result
    if cos_sim >= 0.99:
        print(f"\n[SUCCESS] PASS: cos_sim={cos_sim:.6f} >= 0.99")
        return True
    else:
        print(f"\n[FAILURE] FAIL: cos_sim={cos_sim:.6f} < 0.99")
        return False

if __name__ == "__main__":
    success = test_compressed_correctness()
    sys.exit(0 if success else 1)
