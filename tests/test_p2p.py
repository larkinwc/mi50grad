#!/usr/bin/env python3
"""Test P2P connectivity and measure latency/bandwidth between GPUs."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runtime.hip_dispatch import GPUDevice


def main():
    # Create devices
    devices = []
    for i in range(4):
        try:
            d = GPUDevice(i)
            devices.append(d)
            print(f"GPU {i}: OK")
        except Exception as e:
            print(f"GPU {i}: {e}")

    n_gpus = len(devices)
    print(f"\n{n_gpus} GPUs available")

    # Test P2P access
    print("\nP2P connectivity:")
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i == j:
                continue
            try:
                can = devices[i].hip.device_can_access_peer(i, j)
                print(f"  GPU {i} -> GPU {j}: {'OK' if can else 'NO'}")
                if can:
                    devices[i].hip.set_device(i)
                    devices[i].hip.device_enable_peer_access(j)
            except Exception as e:
                print(f"  GPU {i} -> GPU {j}: {e}")

    # Measure P2P latency for small messages (10 KB = hidden state)
    print("\nP2P latency (10 KB = hidden state):")
    size = 10240  # 5120 * 2 bytes (hidden_dim FP16)

    for i in range(min(n_gpus, 2)):
        for j in range(min(n_gpus, 2)):
            if i == j:
                continue
            devices[i].hip.set_device(i)
            src = devices[i].malloc(size)
            devices[j].hip.set_device(j)
            dst = devices[j].malloc(size)

            # Warm up
            devices[i].hip.set_device(i)
            for _ in range(5):
                devices[i].hip.memcpy_peer_async(dst, j, src, i, size, 0)
            devices[i].synchronize()

            # Measure
            iters = 100
            devices[i].hip.set_device(i)
            t0 = time.perf_counter()
            for _ in range(iters):
                devices[i].hip.memcpy_peer_async(dst, j, src, i, size, 0)
            devices[i].synchronize()
            t1 = time.perf_counter()

            latency_us = (t1 - t0) / iters * 1e6
            bw_gbs = size / ((t1 - t0) / iters) / 1e9
            print(f"  GPU {i} -> GPU {j}: {latency_us:.1f} us ({bw_gbs:.2f} GB/s)")

    # Measure allreduce simulation: sum 10 KB across 2 GPUs
    if n_gpus >= 2:
        print("\nSimulated allreduce (10 KB, 2 GPUs):")
        import numpy as np

        devices[0].hip.set_device(0)
        buf0 = devices[0].malloc(size)
        devices[1].hip.set_device(1)
        buf1 = devices[1].malloc(size)

        # Host-side allreduce
        devices[0].hip.set_device(0)
        t0 = time.perf_counter()
        for _ in range(100):
            # Download both
            data0 = devices[0].download(buf0, size)
            data1 = devices[1].download(buf1, size)
            # Sum
            arr0 = np.frombuffer(data0, dtype=np.float16)
            arr1 = np.frombuffer(data1, dtype=np.float16)
            result = (arr0 + arr1).tobytes()
            # Upload to both
            devices[0].upload(buf0, result)
            devices[1].upload(buf1, result)
        t1 = time.perf_counter()
        print(f"  Host allreduce: {(t1-t0)/100*1e6:.0f} us")

        # P2P allreduce: copy buf1 to GPU0, sum on GPU0, copy back
        # (Would need a kernel for the sum step - just measure the copies)
        devices[0].hip.set_device(0)
        tmp = devices[0].malloc(size)
        t0 = time.perf_counter()
        for _ in range(100):
            devices[0].hip.memcpy_peer_async(tmp, 0, buf1, 1, size, 0)
            devices[0].synchronize()
            # GPU sum would go here (~2 us for 10 KB)
            devices[0].hip.memcpy_peer_async(buf1, 1, buf0, 0, size, 0)
            devices[0].synchronize()
        t1 = time.perf_counter()
        print(f"  P2P copy-only: {(t1-t0)/100*1e6:.0f} us (+ ~2 us GPU sum)")

    for d in devices:
        d.cleanup()


if __name__ == "__main__":
    main()
