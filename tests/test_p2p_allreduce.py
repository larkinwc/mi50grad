#!/usr/bin/env python3
"""
Microbenchmark and correctness test for P2P allreduce vs host-mediated allreduce.

Tests:
- Correctness: max abs error < 1e-3 vs reference (host-mediated)
- Microbenchmark: old (fast_allreduce.c) vs new (P2P HIP kernel) for TP=2 and TP=4
- Reports latency in us/call and speedup ratio
- Tests TP=2, TP=3, and TP=4 configurations

Usage:
    python3 tests/test_p2p_allreduce.py
"""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime, HIPError
from src.runtime.p2p_allreduce import P2PAllreduce


def alloc_fill_gpu(dev: GPUDevice, data: np.ndarray):
    """Allocate GPU buffer and upload numpy array."""
    ptr = dev.malloc(data.nbytes)
    dev.upload(ptr, data.tobytes())
    return ptr


def download_fp16(dev: GPUDevice, ptr: int, num_elems: int) -> np.ndarray:
    """Download FP16 buffer from GPU."""
    raw = dev.download(ptr, num_elems * 2)
    return np.frombuffer(raw, dtype=np.float16).copy()


def host_allreduce_residual(hip, device_ids, partial_ptrs, hidden_ptrs,
                            num_elems):
    """Reference: host-mediated allreduce + residual add."""
    size = num_elems * 2
    # Download all partials + hidden from GPU0
    partials = []
    for i, (dev_id, ptr) in enumerate(zip(device_ids, partial_ptrs)):
        hip.set_device(dev_id)
        buf = ctypes.create_string_buffer(size)
        hip.memcpy_d2h(buf, ptr, size)
        partials.append(np.frombuffer(buf, dtype=np.float16).copy())

    hip.set_device(device_ids[0])
    hbuf = ctypes.create_string_buffer(size)
    hip.memcpy_d2h(hbuf, hidden_ptrs[0], size)
    hidden = np.frombuffer(hbuf, dtype=np.float16).copy().astype(np.float32)

    # Accumulate
    for p in partials:
        hidden += p.astype(np.float32)
    result = hidden.astype(np.float16)

    # Upload result to all GPUs
    result_bytes = result.tobytes()
    for dev_id, hp in zip(device_ids, hidden_ptrs):
        hip.set_device(dev_id)
        hip.memcpy_h2d(hp, result_bytes, size)

    return result


def test_correctness(tp_size: int, num_elems: int = 5120):
    """Test correctness of P2P allreduce vs host-mediated reference.

    Args:
        tp_size: number of GPUs (2, 3, or 4)
        num_elems: number of FP16 elements (hidden_size)

    Returns: max abs error between P2P and reference
    """
    print(f"\n--- Correctness test: TP={tp_size}, hidden={num_elems} ---")
    rng = np.random.default_rng(42)

    # Create devices and initialize HIP
    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    # Create per-GPU streams
    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    # Generate test data
    # partials: what each GPU computed (random FP16 values)
    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16)  # original hidden (GPU0)

    # Allocate GPU buffers
    partial_ptrs_ref = []
    hidden_ptrs_ref = []
    for i in range(tp_size):
        partial_ptrs_ref.append(alloc_fill_gpu(devices[i], partials[i]))
        h_data = hidden0 if i == 0 else np.zeros(num_elems, dtype=np.float16)
        hidden_ptrs_ref.append(alloc_fill_gpu(devices[i], h_data))

    # Compute reference with host-mediated allreduce
    ref_result = host_allreduce_residual(
        hip, device_ids, partial_ptrs_ref, hidden_ptrs_ref, num_elems)

    # Reset partial buffers for P2P test
    partial_ptrs_p2p = []
    hidden_ptrs_p2p = []
    for i in range(tp_size):
        partial_ptrs_p2p.append(alloc_fill_gpu(devices[i], partials[i]))
        h_data = hidden0 if i == 0 else np.zeros(num_elems, dtype=np.float16)
        hidden_ptrs_p2p.append(alloc_fill_gpu(devices[i], h_data))

    # Run P2P allreduce
    p2p_ar = P2PAllreduce(hip, device_ids, num_elems, streams=streams)
    p2p_ar.allreduce_residual(partial_ptrs_p2p, hidden_ptrs_p2p, num_elems)

    # Download P2P result from GPU0
    p2p_result = download_fp16(devices[0], hidden_ptrs_p2p[0], num_elems)

    # Compare
    max_abs_err = float(np.max(np.abs(ref_result.astype(np.float32) -
                                      p2p_result.astype(np.float32))))
    max_ref = float(np.max(np.abs(ref_result.astype(np.float32))))
    rel_err = max_abs_err / (max_ref + 1e-8)

    print(f"  Max abs error: {max_abs_err:.4e}")
    print(f"  Max relative error: {rel_err:.4e}")
    print(f"  Max ref value: {max_ref:.4f}")

    if max_abs_err < 1e-3:
        print(f"  PASS: max_abs_err={max_abs_err:.4e} < 1e-3")
    else:
        print(f"  FAIL: max_abs_err={max_abs_err:.4e} >= 1e-3")

    # Also verify all GPUs got the same result
    for i in range(1, tp_size):
        result_i = download_fp16(devices[i], hidden_ptrs_p2p[i], num_elems)
        diff = float(np.max(np.abs(p2p_result.astype(np.float32) -
                                   result_i.astype(np.float32))))
        if diff > 1e-6:
            print(f"  FAIL: GPU{i} result differs from GPU0 by {diff:.4e}")
        else:
            print(f"  GPU{i} consistency: OK (max diff={diff:.4e})")

    # Cleanup
    p2p_ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs_ref[i])
        hip.free(hidden_ptrs_ref[i])
        hip.free(partial_ptrs_p2p[i])
        hip.free(hidden_ptrs_p2p[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return max_abs_err


def test_allreduce_sum_correctness(tp_size: int, num_elems: int = 5120):
    """Test correctness of P2P allreduce_sum (no residual)."""
    print(f"\n--- Correctness test (sum only): TP={tp_size}, hidden={num_elems} ---")
    rng = np.random.default_rng(123)

    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]

    # Reference: CPU sum
    ref_sum = np.zeros(num_elems, dtype=np.float32)
    for p in partials:
        ref_sum += p.astype(np.float32)
    ref_result = ref_sum.astype(np.float16)

    # Allocate GPU buffers
    partial_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))

    # Run P2P allreduce_sum
    p2p_ar = P2PAllreduce(hip, device_ids, num_elems, streams=streams)
    p2p_ar.allreduce_sum(partial_ptrs, num_elems)

    # Download result from GPU0
    result = download_fp16(devices[0], partial_ptrs[0], num_elems)

    max_abs_err = float(np.max(np.abs(ref_result.astype(np.float32) -
                                      result.astype(np.float32))))
    print(f"  Max abs error: {max_abs_err:.4e}")
    if max_abs_err < 1e-3:
        print(f"  PASS: max_abs_err={max_abs_err:.4e} < 1e-3")
    else:
        print(f"  FAIL: max_abs_err={max_abs_err:.4e} >= 1e-3")

    # Cleanup
    p2p_ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    return max_abs_err


def bench_old_allreduce(tp_size: int, num_elems: int = 5120,
                        n_warmup: int = 10, n_iters: int = 200):
    """Benchmark host-mediated allreduce (old fast_allreduce.c path).

    Returns: median latency in microseconds
    """
    import subprocess
    import os

    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    # Try to load fast_allreduce.c
    src_dir = Path(__file__).parent.parent / "src" / "runtime"
    so_path = src_dir / "fast_allreduce.so"
    c_path = src_dir / "fast_allreduce.c"

    if not so_path.exists() or (c_path.exists() and
            os.path.getmtime(c_path) > os.path.getmtime(so_path)):
        subprocess.check_call([
            "gcc", "-O3", "-mf16c", "-mavx", "-shared", "-fPIC",
            "-o", str(so_path), str(c_path)
        ], stderr=subprocess.DEVNULL)

    lib = ctypes.CDLL(str(so_path))

    hip_set_device_fn = ctypes.cast(hip._lib.hipSetDevice, ctypes.c_void_p).value
    hip_sync_fn = ctypes.cast(hip._lib.hipDeviceSynchronize, ctypes.c_void_p).value
    hip_memcpy_fn = ctypes.cast(hip._lib.hipMemcpy, ctypes.c_void_p).value

    lib.fast_ar_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.fast_ar_init.restype = ctypes.c_int
    lib.fast_ar_init(hip_set_device_fn, hip_sync_fn, hip_memcpy_fn)

    rng = np.random.default_rng(0)
    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16)

    partial_ptrs = []
    hidden_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        h_data = hidden0 if i == 0 else np.zeros(num_elems, dtype=np.float16)
        hidden_ptrs.append(alloc_fill_gpu(devices[i], h_data))

    def run_fast_ar():
        if tp_size == 2:
            lib.fast_ar_fused_tp2.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_int,
            ]
            lib.fast_ar_fused_tp2.restype = ctypes.c_int
            lib.fast_ar_fused_tp2(
                device_ids[0], device_ids[1],
                partial_ptrs[0], partial_ptrs[1],
                hidden_ptrs[0], hidden_ptrs[1],
                num_elems)
        elif tp_size == 4:
            lib.fast_ar_fused_tp4.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                ctypes.c_int,
            ]
            lib.fast_ar_fused_tp4.restype = ctypes.c_int
            lib.fast_ar_fused_tp4(
                device_ids[0], device_ids[1], device_ids[2], device_ids[3],
                partial_ptrs[0], partial_ptrs[1], partial_ptrs[2], partial_ptrs[3],
                hidden_ptrs[0], hidden_ptrs[1], hidden_ptrs[2], hidden_ptrs[3],
                num_elems)

    # Warmup
    for _ in range(n_warmup):
        run_fast_ar()

    # Timed
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        run_fast_ar()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    # Cleanup
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
    for d in devices:
        d.cleanup()

    latencies.sort()
    return float(np.median(latencies)), float(np.mean(latencies))


def bench_p2p_allreduce(tp_size: int, num_elems: int = 5120,
                         n_warmup: int = 10, n_iters: int = 200):
    """Benchmark P2P GPU allreduce.

    Returns: (median_us, mean_us)
    """
    devices = []
    for i in range(tp_size):
        d = GPUDevice(i)
        devices.append(d)
    hip = devices[0].hip
    device_ids = list(range(tp_size))

    streams = []
    for i in range(tp_size):
        hip.set_device(i)
        streams.append(hip.stream_create())

    rng = np.random.default_rng(0)
    partials = [rng.random(num_elems).astype(np.float16) for _ in range(tp_size)]
    hidden0 = rng.random(num_elems).astype(np.float16)

    partial_ptrs = []
    hidden_ptrs = []
    for i in range(tp_size):
        partial_ptrs.append(alloc_fill_gpu(devices[i], partials[i]))
        h_data = hidden0 if i == 0 else np.zeros(num_elems, dtype=np.float16)
        hidden_ptrs.append(alloc_fill_gpu(devices[i], h_data))

    p2p_ar = P2PAllreduce(hip, device_ids, num_elems, streams=streams)

    def run_p2p():
        p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, num_elems)

    # Warmup
    for _ in range(n_warmup):
        run_p2p()

    # Timed
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        run_p2p()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    # Cleanup
    p2p_ar.cleanup()
    for i in range(tp_size):
        hip.set_device(i)
        hip.free(partial_ptrs[i])
        hip.free(hidden_ptrs[i])
        hip.stream_destroy(streams[i])
    for d in devices:
        d.cleanup()

    latencies.sort()
    return float(np.median(latencies)), float(np.mean(latencies))


def main():
    print("=" * 60)
    print("P2P Allreduce Test: Correctness + Microbenchmark")
    print("=" * 60)

    # Check available GPUs
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs available: {n_gpus}")

    if n_gpus < 2:
        print("ERROR: Need at least 2 GPUs for TP allreduce tests")
        sys.exit(1)

    # --- Correctness Tests ---
    print("\n" + "=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    all_pass = True

    # TP=2 correctness
    if n_gpus >= 2:
        err = test_correctness(2, 5120)
        if err >= 1e-3:
            all_pass = False

    # TP=3 correctness
    if n_gpus >= 3:
        err = test_correctness(3, 5120)
        if err >= 1e-3:
            all_pass = False

    # TP=4 correctness
    if n_gpus >= 4:
        err = test_correctness(4, 5120)
        if err >= 1e-3:
            all_pass = False

    # TP=4 allreduce_sum correctness
    if n_gpus >= 4:
        err = test_allreduce_sum_correctness(4, 5120)
        if err >= 1e-3:
            all_pass = False

    if all_pass:
        print("\nAll correctness tests PASSED")
    else:
        print("\nSome correctness tests FAILED")
        sys.exit(1)

    # --- Benchmarks ---
    print("\n" + "=" * 60)
    print("MICROBENCHMARKS (hidden_size=5120, FP16, 200 iters)")
    print("=" * 60)

    for tp in [2, 4]:
        if n_gpus < tp:
            print(f"\nSkipping TP={tp}: only {n_gpus} GPUs available")
            continue

        print(f"\n--- TP={tp} allreduce_residual benchmark ---")

        # Old host-mediated
        print("  Running host-mediated (old)...")
        try:
            old_med, old_mean = bench_old_allreduce(tp, 5120)
            print(f"  Host-mediated (fast_allreduce.c): "
                  f"median={old_med:.1f} us, mean={old_mean:.1f} us/call")
        except Exception as e:
            print(f"  Host-mediated benchmark failed: {e}")
            old_med = None

        # New P2P
        print("  Running P2P GPU allreduce (new)...")
        p2p_med, p2p_mean = bench_p2p_allreduce(tp, 5120)
        print(f"  P2P GPU allreduce (new):         "
              f"median={p2p_med:.1f} us, mean={p2p_mean:.1f} us/call")

        if old_med is not None:
            speedup = old_med / p2p_med
            print(f"  Speedup (P2P vs host): {speedup:.2f}x")
            if speedup >= 1.5:
                print(f"  PASS: {speedup:.2f}x >= 1.5x target")
            else:
                print(f"  NOTE: {speedup:.2f}x < 1.5x target (P2P may have 2-hop overhead)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("All tests complete. See above for pass/fail status.")


if __name__ == "__main__":
    main()
