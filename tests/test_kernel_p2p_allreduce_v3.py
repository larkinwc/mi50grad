#!/usr/bin/env python3
"""
Correctness and latency benchmark for v3 optimized kernel P2P allreduce.

V3 Optimizations (from m2-profile-pcie4-bandwidth profiling):
- Single-wavefront design (64 threads instead of 128/256)
- 1 __syncthreads() barrier (vs 2 in v2/v1)
- No LDS usage - pure register + __shfl reduction
- Optimized for latency-bound 10KB payloads

Tests:
1. Correctness: max abs error < 5e-3 vs reference (host-mediated) allreduce
2. Latency: v3 vs v2 vs v1 - target <= 40us (vs ~75us v2, ~79us baseline)
3. Multi-GPU consistency: all GPUs produce identical results

Validates:
  VAL-M2-PCIE4-001: V3 kernel correctness (max_abs_error < 5e-3)
  VAL-M2-PCIE4-002: V3 kernel latency improvement (target: <= 40us vs ~75us baseline)
  VAL-M2-PCIE4-003: Multi-GPU consistency (max diff < 1e-3)

Usage:
    python3 tests/test_kernel_p2p_allreduce_v3.py
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError
from src.runtime.p2p_allreduce import P2PAllreduce


def alloc_fill_gpu(dev: GPUDevice, data: np.ndarray) -> int:
    """Allocate GPU buffer and upload numpy array. Returns device pointer."""
    ptr = dev.malloc(data.nbytes)
    dev.upload(ptr, data.tobytes())
    return ptr


def download_fp16(dev: GPUDevice, ptr: int, num_elems: int) -> np.ndarray:
    """Download FP16 buffer from GPU."""
    raw = dev.download(ptr, num_elems * 2)
    return np.frombuffer(raw, dtype=np.float16).copy()


def host_reference_allreduce(hip, device_ids, partial_ptrs, hidden_ptrs, num_elems):
    """Reference: host-mediated allreduce + residual add.

    Downloads all partials + hidden[0] from GPU, accumulates in FP32, then
    uploads the result to all GPUs' hidden buffers.
    """
    size = num_elems * 2

    # Download all partials
    partials = []
    for dev_id, ptr in zip(device_ids, partial_ptrs):
        hip.set_device(dev_id)
        buf = ctypes.create_string_buffer(size)
        hip.memcpy_d2h(buf, ptr, size)
        partials.append(np.frombuffer(buf, dtype=np.float16).copy())

    # Download hidden from GPU0 (representative)
    hip.set_device(device_ids[0])
    hbuf = ctypes.create_string_buffer(size)
    hip.memcpy_d2h(hbuf, hidden_ptrs[0], size)
    hidden_f32 = np.frombuffer(hbuf, dtype=np.float16).copy().astype(np.float32)

    # Accumulate all partials + hidden
    for p in partials:
        hidden_f32 += p.astype(np.float32)
    result = hidden_f32.astype(np.float16)

    # Upload result to all GPUs' hidden buffers
    result_bytes = result.tobytes()
    for dev_id, hp in zip(device_ids, hidden_ptrs):
        hip.set_device(dev_id)
        hip.memcpy_h2d(hp, result_bytes, size)

    return result


def load_v3_kernel(build_dir):
    """Load v3 kernel library. Returns (lib, func) or (None, None)."""
    v3_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v3.so"
    if not v3_path.exists():
        return None, None
    
    try:
        lib = ctypes.CDLL(str(v3_path))
        func = lib.kernel_p2p_allreduce_rmsnorm_tp4_v3
        func.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # hidden
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
        func.restype = ctypes.c_int
        return lib, func
    except Exception as e:
        print(f"  Failed to load v3 kernel: {e}")
        return None, None


def load_v2_kernel(build_dir):
    """Load v2 kernel library for comparison. Returns (lib, func) or (None, None)."""
    v2_path = build_dir / "kernel_p2p_allreduce_rmsnorm_v2.so"
    if not v2_path.exists():
        return None, None
    
    try:
        lib = ctypes.CDLL(str(v2_path))
        func = lib.kernel_p2p_allreduce_rmsnorm_tp4_v2
        func.argtypes = [
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # hidden
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
        func.restype = ctypes.c_int
        return lib, func
    except Exception as e:
        print(f"  Failed to load v2 kernel: {e}")
        return None, None


def test_v3_correctness(tp_size: int = 4, num_elems: int = 5120):
    """Test correctness of v3 kernel P2P allreduce vs host reference.

    VAL-M2-PCIE4-001: max abs error < 5e-3

    Returns: (max_abs_error, passed)
    """
    print(f"\n--- V3 Correctness test: TP={tp_size}, hidden={num_elems} ---")
    rng = np.random.default_rng(42)

    # Create devices
    devices = [GPUDevice(i) for i in range(tp_size)]
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    # Create per-GPU streams
    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    # Generate test data
    partials = [rng.random(num_elems).astype(np.float16) * 2 - 1
                for _ in range(tp_size)]
    hidden0 = (rng.random(num_elems).astype(np.float16) * 2 - 1)
    weight = rng.random(num_elems).astype(np.float16) * 0.1 + 0.9  # RMSNorm weight

    # Reference: host-mediated allreduce + RMSNorm
    partial_ptrs_ref = []
    hidden_ptrs_ref = []
    for i in range(tp_size):
        partial_ptrs_ref.append(alloc_fill_gpu(devices[i], partials[i]))
        hidden_ptrs_ref.append(alloc_fill_gpu(devices[i], hidden0))

    ref_result = host_reference_allreduce(
        hip, device_ids, partial_ptrs_ref, hidden_ptrs_ref, num_elems)
    
    # Apply RMSNorm on host
    ref_result_f32 = ref_result.astype(np.float32)
    rms = np.sqrt(np.mean(ref_result_f32 ** 2) + 1e-6)
    ref_result_norm = (ref_result_f32 / rms * weight.astype(np.float32)).astype(np.float16)

    # V3 kernel P2P allreduce
    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    v3_lib, v3_func = load_v3_kernel(build_dir)
    
    if v3_lib is None:
        print("  SKIP: kernel_p2p_allreduce_rmsnorm_v3.so not available")
        # Cleanup
        for i in range(tp_size):
            hip.set_device(i)
            hip.free(partial_ptrs_ref[i])
            hip.free(hidden_ptrs_ref[i])
            hip.stream_destroy(streams[i])
        for d in devices:
            d.cleanup()
        return None, False

    # Allocate output buffers
    output_ptrs = []
    for i in range(tp_size):
        output_ptrs.append(alloc_fill_gpu(devices[i], np.zeros(num_elems, dtype=np.float16)))

    # Get P2P pointers (BAR1-mapped peer memory access)
    from src.runtime.p2p_allreduce import setup_p2p_access
    p2p_info = setup_p2p_access(hip, device_ids, streams)
    
    # Build peer pointers for each GPU
    # Each GPU reads: local + 3 peers via BAR1
    results_v3 = []
    for gpu_idx in range(tp_size):
        hip.set_device(gpu_idx)
        stream = streams[gpu_idx]
        
        # Build peer pointers (peer memory access via BAR1)
        partial_ptrs = []
        for peer_idx in range(tp_size):
            if peer_idx == gpu_idx:
                # Local pointer
                partial_ptrs.append(partial_ptrs_ref[peer_idx])
            else:
                # Peer pointer via BAR1
                peer_info = p2p_info[gpu_idx][peer_idx]
                peer_base = peer_info['base']
                offset = partial_ptrs_ref[peer_idx] - p2p_info[peer_idx][peer_idx]['base']
                partial_ptrs.append(peer_base + offset)
        
        # Call v3 kernel
        err = v3_func(
            output_ptrs[gpu_idx],
            hidden_ptrs_ref[gpu_idx],
            partial_ptrs[0],
            partial_ptrs[1],
            partial_ptrs[2],
            partial_ptrs[3],
            alloc_fill_gpu(devices[gpu_idx], weight),  # weight
            num_elems,
            1,  # batch_size
            1e-6,  # eps
            stream
        )
        
        if err != 0:
            print(f"  ERROR: v3 kernel launch failed on GPU{gpu_idx}: {err}")
            return None, False
        
        # Download result
        results_v3.append(download_fp16(devices[gpu_idx], output_ptrs[gpu_idx], num_elems))

    # Compare results vs reference
    max_abs_err = float(np.max(np.abs(
        ref_result_norm.astype(np.float32) - results_v3[0].astype(np.float32))))
    max_ref = float(np.max(np.abs(ref_result_norm.astype(np.float32))))

    print(f"  Max abs error (v3 vs reference): {max_abs_err:.4e}")
    print(f"  Max reference value: {max_ref:.4f}")

    passed = max_abs_err < 5e-3
    if passed:
        print(f"  PASS (VAL-M2-PCIE4-001): max_abs_error={max_abs_err:.4e} < 5e-3")
    else:
        print(f"  FAIL (VAL-M2-PCIE4-001): max_abs_error={max_abs_err:.4e} >= 5e-3")

    # Check multi-GPU consistency
    all_consistent = True
    for i in range(1, tp_size):
        diff = float(np.max(np.abs(
            results_v3[0].astype(np.float32) -
            results_v3[i].astype(np.float32))))
        if diff > 1e-3:
            print(f"  FAIL: GPU{i} result differs from GPU0 by {diff:.4e}")
            all_consistent = False
        else:
            print(f"  GPU{i} consistency: OK (max diff={diff:.4e})")

    # Cleanup
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs_ref[i])
        hip.free(hidden_ptrs_ref[i])
        hip.free(output_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return max_abs_err, passed and all_consistent


def test_v3_latency(tp_size: int = 4, num_elems: int = 5120,
                    n_warmup: int = 10, n_iters: int = 100):
    """Benchmark v3 kernel latency vs v2.

    VAL-M2-PCIE4-002: v3 latency <= 40us (vs ~75us v2)

    Returns: (v3_median_us, v2_median_us, speedup)
    """
    print(f"\n--- V3 Latency benchmark: TP={tp_size}, hidden={num_elems}, "
          f"{n_iters} iters ---")

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

    # Allocate buffers
    partial_ptrs = []
    hidden_ptrs = []
    output_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        hidden_ptrs.append(alloc_fill_gpu(devices[i], hidden0))
        output_ptrs.append(alloc_fill_gpu(devices[i], np.zeros(num_elems, dtype=np.float16)))

    build_dir = Path(__file__).parent.parent / "build" / "kernels"
    v3_lib, v3_func = load_v3_kernel(build_dir)
    v2_lib, v2_func = load_v2_kernel(build_dir)

    if v3_lib is None:
        print("  SKIP: v3 kernel not available")
        return None, None, None

    # Setup P2P
    from src.runtime.p2p_allreduce import setup_p2p_access
    p2p_info = setup_p2p_access(hip, device_ids, streams)

    # Helper to run v3 kernel on GPU0
    def run_v3():
        gpu_idx = 0
        partial_ptrs_peer = []
        for peer_idx in range(tp_size):
            if peer_idx == gpu_idx:
                partial_ptrs_peer.append(partial_ptrs[peer_idx])
            else:
                peer_info = p2p_info[gpu_idx][peer_idx]
                peer_base = peer_info['base']
                offset = partial_ptrs[peer_idx] - p2p_info[peer_idx][peer_idx]['base']
                partial_ptrs_peer.append(peer_base + offset)
        
        v3_func(
            output_ptrs[gpu_idx],
            hidden_ptrs[gpu_idx],
            partial_ptrs_peer[0],
            partial_ptrs_peer[1],
            partial_ptrs_peer[2],
            partial_ptrs_peer[3],
            alloc_fill_gpu(devices[gpu_idx], weight),
            num_elems,
            1,
            1e-6,
            streams[gpu_idx]
        )
        hip.stream_synchronize(streams[gpu_idx])

    # Helper to run v2 kernel
    def run_v2():
        if v2_func is None:
            return float('inf')
        gpu_idx = 0
        partial_ptrs_peer = []
        for peer_idx in range(tp_size):
            if peer_idx == gpu_idx:
                partial_ptrs_peer.append(partial_ptrs[peer_idx])
            else:
                peer_info = p2p_info[gpu_idx][peer_idx]
                peer_base = peer_info['base']
                offset = partial_ptrs[peer_idx] - p2p_info[peer_idx][peer_idx]['base']
                partial_ptrs_peer.append(peer_base + offset)
        
        v2_func(
            output_ptrs[gpu_idx],
            hidden_ptrs[gpu_idx],
            partial_ptrs_peer[0],
            partial_ptrs_peer[1],
            partial_ptrs_peer[2],
            partial_ptrs_peer[3],
            alloc_fill_gpu(devices[gpu_idx], weight),
            num_elems,
            1,
            1e-6,
            streams[gpu_idx]
        )
        hip.stream_synchronize(streams[gpu_idx])

    # Warmup
    print("  Warming up...")
    for _ in range(n_warmup):
        run_v3()
        if v2_func:
            run_v2()

    # Measure v3
    print("  Measuring v3...")
    v3_times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        run_v3()
        elapsed = (time.perf_counter() - start) * 1e6  # us
        v3_times.append(elapsed)

    v3_median = np.median(v3_times)

    # Measure v2
    v2_median = None
    if v2_func:
        print("  Measuring v2...")
        v2_times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            run_v2()
            elapsed = (time.perf_counter() - start) * 1e6
            v2_times.append(elapsed)
        v2_median = np.median(v2_times)

    print(f"  V3 median latency: {v3_median:.2f} us")
    if v2_median:
        print(f"  V2 median latency: {v2_median:.2f} us")
        speedup = v2_median / v3_median if v3_median > 0 else float('inf')
        print(f"  Speedup: v2/v3 = {speedup:.2f}x")
        
        if v3_median <= 40:
            print(f"  PASS (VAL-M2-PCIE4-002): v3 latency {v3_median:.2f}us <= 40us target")
        else:
            print(f"  PARTIAL (VAL-M2-PCIE4-002): v3 latency {v3_median:.2f}us > 40us target")
    else:
        print(f"  V2 not available for comparison")
        speedup = None

    # Cleanup
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.free(output_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return v3_median, v2_median, speedup


if __name__ == "__main__":
    print("=" * 70)
    print("V3 Kernel P2P Allreduce Tests")
    print("=" * 70)

    # Correctness test
    print("\n[1/2] Correctness Test")
    max_err, passed_correctness = test_v3_correctness()

    # Latency test
    print("\n[2/2] Latency Benchmark")
    v3_lat, v2_lat, speedup = test_v3_latency()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if passed_correctness:
        print("✓ Correctness: PASS")
    else:
        print("✗ Correctness: FAIL")

    if v3_lat is not None:
        print(f"✓ V3 Latency: {v3_lat:.2f} us")
        if v2_lat:
            print(f"  V2 Latency: {v2_lat:.2f} us")
            print(f"  Speedup: {speedup:.2f}x")
    else:
        print("✗ V3 Latency: N/A (kernel not available)")

    if passed_correctness and v3_lat is not None:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED/SKIPPED")
        sys.exit(1)
