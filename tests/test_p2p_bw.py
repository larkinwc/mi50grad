"""Test P2P access and measure inter-GPU bandwidth."""
import sys, time, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.path.insert(0, '/opt/mi50grad')

import ctypes
from src.runtime.hip_dispatch import GPUDevice

# Check P2P capability between all GPU pairs
num_gpus = 4
print(f"Testing {num_gpus} GPUs\n")

# P2P access check
print("=== P2P Access Matrix ===")
devs = []
for i in range(num_gpus):
    devs.append(GPUDevice(device_id=i))

hip = devs[0].hip

for i in range(num_gpus):
    for j in range(num_gpus):
        if i == j:
            continue
        can = ctypes.c_int(0)
        err = hip._lib.hipDeviceCanAccessPeer(ctypes.byref(can), i, j)
        print(f"  GPU{i} -> GPU{j}: canAccessPeer={can.value} (err={err})")

# Try enabling P2P
print("\n=== Enabling P2P ===")
for i in range(num_gpus):
    for j in range(num_gpus):
        if i == j:
            continue
        hip._lib.hipSetDevice(i)
        err = hip._lib.hipDeviceEnablePeerAccess(j, 0)
        if err == 0:
            print(f"  GPU{i} -> GPU{j}: P2P enabled OK")
        elif err == 704:  # hipErrorPeerAccessAlreadyEnabled
            print(f"  GPU{i} -> GPU{j}: P2P already enabled")
        else:
            print(f"  GPU{i} -> GPU{j}: P2P enable FAILED (err={err})")

# Bandwidth test: GPU-to-GPU memcpy
print("\n=== Inter-GPU Bandwidth (hipMemcpyPeer) ===")
sizes = [10240, 102400, 1048576, 10485760]  # 10KB, 100KB, 1MB, 10MB

for sz in sizes:
    # Allocate on GPU0 and GPU1
    hip._lib.hipSetDevice(0)
    d_src = devs[0].malloc(sz)
    hip._lib.hipSetDevice(1)
    d_dst = devs[1].malloc(sz)

    # Warmup
    hip._lib.hipSetDevice(0)
    for _ in range(3):
        err = hip._lib.hipMemcpyPeer(
            ctypes.c_void_p(d_dst), 1,
            ctypes.c_void_p(d_src), 0,
            sz)
    hip._lib.hipDeviceSynchronize()

    if err != 0:
        print(f"  {sz//1024:6d} KB: hipMemcpyPeer FAILED (err={err})")
        continue

    # Timed
    N = 100
    hip._lib.hipDeviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        hip._lib.hipMemcpyPeer(
            ctypes.c_void_p(d_dst), 1,
            ctypes.c_void_p(d_src), 0,
            sz)
    hip._lib.hipDeviceSynchronize()
    elapsed = time.perf_counter() - t0

    bw = (sz * N) / elapsed / 1e9
    lat = elapsed / N * 1e6
    print(f"  {sz//1024:6d} KB: {bw:6.2f} GB/s  ({lat:7.1f} us/copy)")

    devs[0].free(d_src)
    devs[1].free(d_dst)

# Also test H2D and D2H for comparison
print("\n=== Host <-> GPU0 Bandwidth ===")
for sz in sizes:
    hip._lib.hipSetDevice(0)
    d_buf = devs[0].malloc(sz)
    h_buf = ctypes.create_string_buffer(sz)

    # D2H
    for _ in range(3):
        hip._lib.hipMemcpy(h_buf, ctypes.c_void_p(d_buf), sz, 2)
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        hip._lib.hipMemcpy(h_buf, ctypes.c_void_p(d_buf), sz, 2)
    elapsed = time.perf_counter() - t0
    bw_d2h = (sz * N) / elapsed / 1e9
    lat_d2h = elapsed / N * 1e6

    # H2D
    for _ in range(3):
        hip._lib.hipMemcpy(ctypes.c_void_p(d_buf), h_buf, sz, 1)
    t0 = time.perf_counter()
    for _ in range(N):
        hip._lib.hipMemcpy(ctypes.c_void_p(d_buf), h_buf, sz, 1)
    elapsed = time.perf_counter() - t0
    bw_h2d = (sz * N) / elapsed / 1e9
    lat_h2d = elapsed / N * 1e6

    print(f"  {sz//1024:6d} KB: D2H {bw_d2h:6.2f} GB/s ({lat_d2h:7.1f} us)  "
          f"H2D {bw_h2d:6.2f} GB/s ({lat_h2d:7.1f} us)")

    devs[0].free(d_buf)
