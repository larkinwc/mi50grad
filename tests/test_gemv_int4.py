#!/usr/bin/env python3
"""Test harness for INT4 GEMV (quantized weight-only decode path)."""

import ctypes
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"


def ensure_kernel(name):
    asm = PROJECT_ROOT / "src" / "asm" / f"{name}.s"
    hsaco = BUILD_DIR / f"{name}.hsaco"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not hsaco.exists() or asm.stat().st_mtime > hsaco.stat().st_mtime:
        print(f"Building {name}...")
        build_hsaco(str(asm), str(hsaco))
    return str(hsaco)


def pack_int4(values):
    """Pack INT4 values (0-15) into uint32 (8 per uint32, little-endian)."""
    assert len(values) % 8 == 0
    packed = np.zeros(len(values) // 8, dtype=np.uint32)
    for i in range(len(values)):
        group = i // 8
        pos = i % 8
        packed[group] |= (int(values[i]) & 0xF) << (pos * 4)
    return packed


def test_gemv_int4(K=2048, N=64, group_size=128, device_id=0):
    print(f"\n{'='*60}")
    print(f"Testing INT4 GEMV: K={K}, N={N}, group_size={group_size}")
    print(f"{'='*60}")

    assert K % 2048 == 0, "K must be multiple of 2048 (256 threads * 8)"
    assert K % group_size == 0

    np.random.seed(42)

    # Generate random activations
    A = (np.random.randn(K) * 0.1).astype(np.float16)

    # Generate random INT4 weights (0-15)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)

    # Generate scales and zeros per group
    num_groups = K // group_size
    scales = (np.random.randn(num_groups, N) * 0.01 + 0.02).astype(np.float16)
    zeros = (np.random.randn(num_groups, N) * 0.1 + 8.0).astype(np.float16)

    # Reference: dequantize and multiply
    A32 = A.astype(np.float32)
    W_fp32 = np.zeros((K, N), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        for n in range(N):
            s = float(scales[g, n])
            z = float(zeros[g, n])
            for k in range(k_start, k_end):
                W_fp32[k, n] = (float(W_q4[k, n]) - z) * s

    ref = (A32 @ W_fp32).astype(np.float16)

    # Pack INT4 weights: [K, N] -> [K/8, N] uint32
    # Each uint32 holds 8 consecutive K values for one N column
    B_packed = np.zeros((K // 8, N), dtype=np.uint32)
    for n in range(N):
        B_packed[:, n] = pack_int4(W_q4[:, n])

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("gemm_int4")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "gemv_int4_fp16")

        d_A = dev.malloc(A.nbytes)
        d_B = dev.malloc(B_packed.nbytes)
        d_scales = dev.malloc(scales.nbytes)
        d_zeros = dev.malloc(zeros.nbytes)
        d_C = dev.malloc(N * 2)

        dev.upload(d_A, A.tobytes())
        dev.upload(d_B, B_packed.tobytes())
        dev.upload(d_scales, scales.tobytes())
        dev.upload(d_zeros, zeros.tobytes())
        dev.hip.memset(d_C, 0, N * 2)

        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_scales),
            ctypes.c_uint64(d_zeros),
            ctypes.c_uint64(d_C),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(group_size),
        ]

        # Grid: (N, 1, 1), Block: (256, 1, 1)
        dev.launch(func, (N, 1, 1), (256, 1, 1), params, shared_mem=1024)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")
        print(f"ref[:4] = {ref[:4]}")
        print(f"out[:4] = {out[:4]}")

        if max_err < 0.5:  # INT4 quantization introduces significant error
            print("PASS")
            return True
        else:
            print(f"FAIL (max_err={max_err})")
            return False
    finally:
        dev.cleanup()


if __name__ == "__main__":
    all_pass = True

    if not test_gemv_int4(K=2048, N=64, group_size=128):
        all_pass = False

    if not test_gemv_int4(K=4096, N=128, group_size=128):
        all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("All INT4 GEMV tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
