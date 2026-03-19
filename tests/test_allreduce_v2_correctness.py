#!/usr/bin/env python3
"""
Standalone correctness test for optimized v2 allreduce kernel.

This test validates that the v2 kernel produces numerically equivalent
results to the reference implementation (numpy) and to the v1 kernel.

Usage:
    python3 tests/test_allreduce_v2_correctness.py

Validates:
  - VAL-AR-001: v2 kernel correctness (max_abs_error < 1e-3 vs reference)
  - VAL-AR-002: v2 kernel consistency (all GPUs produce same result)
  - VAL-AR-003: v2 vs v1 equivalence (max_diff < 1e-3)
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError


def alloc_fill_gpu(hip: HIPRuntime, device_id: int, data: np.ndarray) -> int:
    """Allocate GPU buffer and upload numpy array."""
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
    """Reference implementation: allreduce + RMSNorm in numpy."""
    result_f32 = hidden.astype(np.float32)
    for p in partials:
        result_f32 += p.astype(np.float32)
    
    sum_sq = np.sum(result_f32 ** 2)
    rms = np.sqrt(sum_sq / len(result_f32) + eps)
    result_normed = result_f32 / rms
    result_final = (result_normed * weight.astype(np.float32)).astype(np.float16)
    return result_final


def test_v2_correctness():
    """Test v2 kernel correctness vs reference."""
    print("=" * 72)
    print("  Test: v2 Kernel Correctness")
    print("=" * 72)
    
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    
    if n_gpus < 4:
        print(f"SKIP: Need 4 GPUs (have {n_gpus})")
        return True
    
    # Load v2 kernel
    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    so_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
    
    if not so_path.exists():
        print(f"SKIP: v2 kernel not found: {so_path}")
        print("  Build with: make kernels")
        return True
    
    lib = ctypes.CDLL(str(so_path))
    lib.kernel_p2p_allreduce_rmsnorm_tp4_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_float, ctypes.c_void_p
    ]
    lib.kernel_p2p_allreduce_rmsnorm_tp4_v2.restype = ctypes.c_int
    
    # Enable P2P access between all GPU pairs (critical for kernel to access peer memory)
    for i in range(4):
        hip.set_device(i)
        for j in range(4):
            if i != j:
                if hip.device_can_access_peer(i, j):
                    try:
                        hip.device_enable_peer_access(j)
                    except Exception:
                        pass  # Already enabled
    
    streams = []
    for i in range(4):
        hip.set_device(i)
        streams.append(hip.stream_create())
    
    # Test data
    rng = np.random.default_rng(42)
    partials_np = [rng.random(5120).astype(np.float16) * 2 - 1 for _ in range(4)]
    hidden_np = rng.random(5120).astype(np.float16) * 2 - 1
    weight_np = rng.random(5120).astype(np.float16) + 0.5
    
    # Allocate
    partial_ptrs = [alloc_fill_gpu(hip, i, partials_np[i]) for i in range(4)]
    hidden_ptrs = [alloc_fill_gpu(hip, i, hidden_np) for i in range(4)]
    output_ptrs = [hip.malloc(5120 * 2) for i in range(4)]
    weight_ptrs = [alloc_fill_gpu(hip, i, weight_np) for i in range(4)]
    
    # Run kernel
    for i in range(4):
        hip.set_device(i)
        peers = [j for j in range(4) if j != i]
        err = lib.kernel_p2p_allreduce_rmsnorm_tp4_v2(
            ctypes.c_void_p(output_ptrs[i]),
            ctypes.c_void_p(hidden_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[peers[0]]),
            ctypes.c_void_p(partial_ptrs[peers[1]]),
            ctypes.c_void_p(partial_ptrs[peers[2]]),
            ctypes.c_void_p(weight_ptrs[i]),
            ctypes.c_uint(5120),
            ctypes.c_uint(1),
            ctypes.c_float(1e-6),
            ctypes.c_void_p(streams[i])
        )
        if err != 0:
            print(f"FAIL: Kernel launch failed on GPU {i}: HIP error {err}")
            return False
    
    # Sync
    for i in range(4):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
    
    # Download results
    results = [download_fp16(hip, i, output_ptrs[i], 5120) for i in range(4)]
    
    # Reference
    ref = reference_allreduce_rmsnorm(partials_np, hidden_np, weight_np, 1e-6)
    
    # Check correctness
    max_err = max(float(np.max(np.abs(ref.astype(np.float32) - results[i].astype(np.float32))))
                  for i in range(4))
    
    print(f"  Max abs error vs reference: {max_err:.4e}")
    
    if max_err >= 1e-3:
        print(f"  FAIL: VAL-AR-001 - max_err={max_err:.4e} >= 1e-3")
        return False
    print(f"  PASS: VAL-AR-001 - max_err={max_err:.4e} < 1e-3")
    
    # Check GPU consistency
    max_diff = max(float(np.max(np.abs(results[0].astype(np.float32) - results[i].astype(np.float32))))
                   for i in range(1, 4))
    
    print(f"  Max diff between GPUs: {max_diff:.4e}")
    
    if max_diff >= 1e-3:
        print(f"  FAIL: VAL-AR-002 - max_diff={max_diff:.4e} >= 1e-3")
        return False
    print(f"  PASS: VAL-AR-002 - max_diff={max_diff:.4e} < 1e-3")
    
    # Cleanup
    for i in range(4):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.free(output_ptrs[i])
        hip.free(weight_ptrs[i])
        hip.stream_destroy(streams[i])
    
    print("\n  All correctness tests PASSED")
    return True


def test_v1_v2_equivalence():
    """Test that v1 and v2 kernels produce equivalent results."""
    print("\n" + "=" * 72)
    print("  Test: v1 vs v2 Equivalence")
    print("=" * 72)
    
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    
    if n_gpus < 4:
        print(f"SKIP: Need 4 GPUs (have {n_gpus})")
        return True
    
    # Load both kernels
    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    v1_path = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
    v2_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
    
    if not v1_path.exists():
        print(f"SKIP: v1 kernel not found: {v1_path}")
        return True
    if not v2_path.exists():
        print(f"SKIP: v2 kernel not found: {v2_path}")
        return True
    
    lib_v1 = ctypes.CDLL(str(v1_path))
    lib_v2 = ctypes.CDLL(str(v2_path))
    
    lib_v1.kernel_p2p_allreduce_rmsnorm_tp4.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_float, ctypes.c_void_p
    ]
    lib_v1.kernel_p2p_allreduce_rmsnorm_tp4.restype = ctypes.c_int
    
    lib_v2.kernel_p2p_allreduce_rmsnorm_tp4_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_float, ctypes.c_void_p
    ]
    lib_v2.kernel_p2p_allreduce_rmsnorm_tp4_v2.restype = ctypes.c_int
    
    # Enable P2P access
    for i in range(4):
        hip.set_device(i)
        for j in range(4):
            if i != j:
                if hip.device_can_access_peer(i, j):
                    try:
                        hip.device_enable_peer_access(j)
                    except Exception:
                        pass
    
    streams = []
    for i in range(4):
        hip.set_device(i)
        streams.append(hip.stream_create())
    
    # Test data
    rng = np.random.default_rng(42)
    partials_np = [rng.random(5120).astype(np.float16) * 2 - 1 for _ in range(4)]
    hidden_np = rng.random(5120).astype(np.float16) * 2 - 1
    weight_np = rng.random(5120).astype(np.float16) + 0.5
    
    # Allocate (run v1 and v2 on same data)
    partial_ptrs = [alloc_fill_gpu(hip, i, partials_np[i]) for i in range(4)]
    hidden_ptrs = [alloc_fill_gpu(hip, i, hidden_np) for i in range(4)]
    output_v1_ptrs = [hip.malloc(5120 * 2) for i in range(4)]
    output_v2_ptrs = [hip.malloc(5120 * 2) for i in range(4)]
    weight_ptrs = [alloc_fill_gpu(hip, i, weight_np) for i in range(4)]
    
    # Run v1
    for i in range(4):
        hip.set_device(i)
        peers = [j for j in range(4) if j != i]
        lib_v1.kernel_p2p_allreduce_rmsnorm_tp4(
            ctypes.c_void_p(output_v1_ptrs[i]),
            ctypes.c_void_p(hidden_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[peers[0]]),
            ctypes.c_void_p(partial_ptrs[peers[1]]),
            ctypes.c_void_p(partial_ptrs[peers[2]]),
            ctypes.c_void_p(weight_ptrs[i]),
            5120, 1, 1e-6, ctypes.c_void_p(streams[i]))
    
    for i in range(4):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
    
    # Reset hidden_ptrs for v2 (need to re-upload since v1 modified them)
    for i in range(4):
        hip.set_device(i)
        hip.memcpy_h2d(hidden_ptrs[i], hidden_np.tobytes(), hidden_np.nbytes)
    
    # Run v2
    for i in range(4):
        hip.set_device(i)
        peers = [j for j in range(4) if j != i]
        lib_v2.kernel_p2p_allreduce_rmsnorm_tp4_v2(
            ctypes.c_void_p(output_v2_ptrs[i]),
            ctypes.c_void_p(hidden_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[i]),
            ctypes.c_void_p(partial_ptrs[peers[0]]),
            ctypes.c_void_p(partial_ptrs[peers[1]]),
            ctypes.c_void_p(partial_ptrs[peers[2]]),
            ctypes.c_void_p(weight_ptrs[i]),
            5120, 1, 1e-6, ctypes.c_void_p(streams[i]))
    
    for i in range(4):
        hip.set_device(i)
        hip.stream_synchronize(streams[i])
    
    # Download and compare
    results_v1 = [download_fp16(hip, i, output_v1_ptrs[i], 5120) for i in range(4)]
    results_v2 = [download_fp16(hip, i, output_v2_ptrs[i], 5120) for i in range(4)]
    
    max_diff = max(float(np.max(np.abs(results_v1[i].astype(np.float32) - results_v2[i].astype(np.float32))))
                   for i in range(4))
    
    print(f"  Max diff v1 vs v2: {max_diff:.4e}")
    
    if max_diff >= 1e-3:
        print(f"  FAIL: VAL-AR-003 - max_diff={max_diff:.4e} >= 1e-3")
        return False
    print(f"  PASS: VAL-AR-003 - max_diff={max_diff:.4e} < 1e-3")
    
    # Cleanup
    for i in range(4):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.free(output_v1_ptrs[i])
        hip.free(output_v2_ptrs[i])
        hip.free(weight_ptrs[i])
        hip.stream_destroy(streams[i])
    
    print("\n  v1/v2 equivalence test PASSED")
    return True


def main():
    print("=" * 72)
    print("  Allreduce v2 Kernel Correctness Validation")
    print("=" * 72)
    
    all_pass = True
    
    # Test 1: v2 correctness
    if not test_v2_correctness():
        all_pass = False
    
    # Test 2: v1/v2 equivalence
    if not test_v1_v2_equivalence():
        all_pass = False
    
    print("\n" + "=" * 72)
    if all_pass:
        print("  ALL TESTS PASSED")
        print("=" * 72)
        sys.exit(0)
    else:
        print("  SOME TESTS FAILED")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
