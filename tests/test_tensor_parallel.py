#!/usr/bin/env python3
"""Test harness for tensor parallelism: all_reduce, scatter, gather, broadcast."""

import ctypes
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime
from src.runtime.tensor_parallel import TensorParallelGroup


DEVICE_IDS = [0, 1]
NUM_ELEMS = 2048  # FP16 elements per buffer (4096 bytes, well under 32KB threshold)


def _alloc_and_upload(tp: TensorParallelGroup, arrays: list[np.ndarray]) -> list[int]:
    """Allocate device memory on each GPU and upload host arrays. Returns device ptrs."""
    ptrs = []
    for i, arr in enumerate(arrays):
        dev = tp.devices[i]
        dev.hip.set_device(dev.device_id)
        ptr = dev.malloc(arr.nbytes)
        dev.upload(ptr, arr.tobytes())
        ptrs.append(ptr)
    return ptrs


def _download(tp: TensorParallelGroup, ptrs: list[int], nbytes: int) -> list[np.ndarray]:
    """Download device buffers back to host as FP16 numpy arrays."""
    results = []
    for i, ptr in enumerate(ptrs):
        dev = tp.devices[i]
        dev.hip.set_device(dev.device_id)
        data = dev.download(ptr, nbytes)
        results.append(np.frombuffer(data, dtype=np.float16).copy())
    return results


def _free_ptrs(tp: TensorParallelGroup, ptrs: list[int]):
    """Free device pointers."""
    for i, ptr in enumerate(ptrs):
        dev = tp.devices[i]
        dev.hip.set_device(dev.device_id)
        dev.free(ptr)


def test_all_reduce_sum(tp: TensorParallelGroup):
    print(f"\n{'='*60}")
    print("Testing all_reduce_sum")
    print(f"{'='*60}")

    np.random.seed(42)
    n = tp.world_size

    # Each GPU gets a distinct array
    host_arrays = [
        np.random.randn(NUM_ELEMS).astype(np.float16) for _ in range(n)
    ]

    # Reference: element-wise sum of all arrays
    ref = np.zeros(NUM_ELEMS, dtype=np.float32)
    for arr in host_arrays:
        ref += arr.astype(np.float32)
    ref = ref.astype(np.float16)

    ptrs = _alloc_and_upload(tp, host_arrays)
    try:
        size = NUM_ELEMS * 2  # FP16 = 2 bytes
        tp.all_reduce_sum(ptrs, size)

        results = _download(tp, ptrs, size)

        for i in range(n):
            max_err = np.max(np.abs(results[i].astype(np.float32) - ref.astype(np.float32)))
            # FP16 accumulation tolerance -- allow small drift
            ok = max_err < 0.5
            status = "PASS" if ok else "FAIL"
            print(f"  GPU {DEVICE_IDS[i]}: max_err={max_err:.4f} [{status}]")
            if not ok:
                return False
    finally:
        _free_ptrs(tp, ptrs)

    print("  all_reduce_sum: PASS")
    return True


def test_scatter_gather(tp: TensorParallelGroup):
    print(f"\n{'='*60}")
    print("Testing scatter + gather")
    print(f"{'='*60}")

    np.random.seed(123)
    n = tp.world_size
    chunk_elems = NUM_ELEMS
    total_elems = chunk_elems * n
    chunk_bytes = chunk_elems * 2

    # Create full source array on device 0
    src_host = np.random.randn(total_elems).astype(np.float16)
    dev0 = tp.devices[0]
    dev0.hip.set_device(dev0.device_id)
    src_ptr = dev0.malloc(src_host.nbytes)
    dev0.upload(src_ptr, src_host.tobytes())

    # Allocate chunk buffers on each device
    dst_ptrs = []
    for i in range(n):
        dev = tp.devices[i]
        dev.hip.set_device(dev.device_id)
        dst_ptrs.append(dev.malloc(chunk_bytes))

    try:
        # Scatter
        tp.scatter(src_ptr, DEVICE_IDS[0], dst_ptrs, chunk_bytes)

        # Verify each chunk
        scatter_ok = True
        for i in range(n):
            dev = tp.devices[i]
            dev.hip.set_device(dev.device_id)
            data = dev.download(dst_ptrs[i], chunk_bytes)
            got = np.frombuffer(data, dtype=np.float16)
            expected = src_host[i * chunk_elems:(i + 1) * chunk_elems]
            if not np.array_equal(got, expected):
                print(f"  Scatter GPU {DEVICE_IDS[i]}: FAIL (mismatch)")
                scatter_ok = False
            else:
                print(f"  Scatter GPU {DEVICE_IDS[i]}: PASS")

        if not scatter_ok:
            return False

        # Gather back to device 0
        dev0.hip.set_device(dev0.device_id)
        gather_ptr = dev0.malloc(src_host.nbytes)
        try:
            tp.gather(dst_ptrs, chunk_bytes, gather_ptr, DEVICE_IDS[0])

            gathered_data = dev0.download(gather_ptr, src_host.nbytes)
            gathered = np.frombuffer(gathered_data, dtype=np.float16)

            if np.array_equal(gathered, src_host):
                print("  Gather: PASS")
            else:
                max_err = np.max(np.abs(gathered.astype(np.float32) - src_host.astype(np.float32)))
                print(f"  Gather: FAIL (max_err={max_err:.6f})")
                return False
        finally:
            dev0.hip.set_device(dev0.device_id)
            dev0.free(gather_ptr)

    finally:
        dev0.hip.set_device(dev0.device_id)
        dev0.free(src_ptr)
        _free_ptrs(tp, dst_ptrs)

    print("  scatter+gather: PASS")
    return True


def test_broadcast(tp: TensorParallelGroup):
    print(f"\n{'='*60}")
    print("Testing broadcast")
    print(f"{'='*60}")

    np.random.seed(999)
    n = tp.world_size
    size = NUM_ELEMS * 2  # bytes

    # Source data on device 0
    src_host = np.random.randn(NUM_ELEMS).astype(np.float16)
    dev0 = tp.devices[0]
    dev0.hip.set_device(dev0.device_id)
    src_ptr = dev0.malloc(size)
    dev0.upload(src_ptr, src_host.tobytes())

    # Allocate destination buffers on all devices
    dst_ptrs = []
    for i in range(n):
        dev = tp.devices[i]
        dev.hip.set_device(dev.device_id)
        if i == 0:
            # For source device, dst can be same as src
            dst_ptrs.append(src_ptr)
        else:
            dst_ptrs.append(dev.malloc(size))

    try:
        tp.broadcast(src_ptr, DEVICE_IDS[0], dst_ptrs, size)

        for i in range(n):
            dev = tp.devices[i]
            dev.hip.set_device(dev.device_id)
            data = dev.download(dst_ptrs[i], size)
            got = np.frombuffer(data, dtype=np.float16)
            if np.array_equal(got, src_host):
                print(f"  GPU {DEVICE_IDS[i]}: PASS")
            else:
                print(f"  GPU {DEVICE_IDS[i]}: FAIL")
                return False
    finally:
        # Free src_ptr (same as dst_ptrs[0]) and other dst_ptrs
        dev0.hip.set_device(dev0.device_id)
        dev0.free(src_ptr)
        for i in range(1, n):
            dev = tp.devices[i]
            dev.hip.set_device(dev.device_id)
            dev.free(dst_ptrs[i])

    print("  broadcast: PASS")
    return True


def main():
    print(f"Tensor Parallel tests using devices {DEVICE_IDS}")

    # Verify we have enough GPUs
    hip = HIPRuntime()
    hip.init()
    count = hip.device_count()
    print(f"Found {count} GPU(s)")
    if count < len(DEVICE_IDS):
        print(f"ERROR: Need at least {len(DEVICE_IDS)} GPUs, found {count}")
        sys.exit(1)

    tp = TensorParallelGroup(DEVICE_IDS)
    try:
        passed = 0
        failed = 0

        for test_fn in [test_all_reduce_sum, test_scatter_gather, test_broadcast]:
            try:
                if test_fn(tp):
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                failed += 1

        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")
        sys.exit(0 if failed == 0 else 1)

    finally:
        tp.cleanup()


if __name__ == "__main__":
    main()
