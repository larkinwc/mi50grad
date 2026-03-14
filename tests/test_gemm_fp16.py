#!/usr/bin/env python3
"""
Test harness for FP16 GEMM kernel (gemm_fp16_64x64).

Compiles the assembly kernel, runs on MI50, validates against numpy reference.
Must be run on the dev server (or in the ROCm container).
"""

import ctypes
import struct
import sys
import os
import subprocess
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice, HIPError
from src.kernels.launcher import build_hsaco


ASM_PATH = PROJECT_ROOT / "src" / "asm" / "gemm_fp16.s"
HSACO_PATH = PROJECT_ROOT / "build" / "kernels" / "gemm_fp16.hsaco"


def ensure_hsaco():
    """Build the HSACO if it doesn't exist or is stale."""
    HSACO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HSACO_PATH.exists() or ASM_PATH.stat().st_mtime > HSACO_PATH.stat().st_mtime:
        print(f"Building {HSACO_PATH}...")
        build_hsaco(str(ASM_PATH), str(HSACO_PATH))
        print("Build OK")
    return str(HSACO_PATH)


def fp16_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy FP16 array to bytes."""
    return arr.astype(np.float16).tobytes()


def bytes_to_fp16(data: bytes, shape: tuple) -> np.ndarray:
    """Convert bytes to numpy FP16 array."""
    return np.frombuffer(data, dtype=np.float16).reshape(shape)


def run_gemm_test(M, N, K, device_id=0):
    """Run GEMM test for given dimensions and compare to numpy."""
    print(f"\n{'='*60}")
    print(f"Testing GEMM: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    assert M % 64 == 0, f"M must be multiple of 64, got {M}"
    assert N % 64 == 0, f"N must be multiple of 64, got {N}"
    assert K % 16 == 0, f"K must be multiple of 16, got {K}"

    # Generate random FP16 inputs
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)

    # Scale down to avoid FP16 overflow
    A = (A * 0.1).astype(np.float16)
    B = (B * 0.1).astype(np.float16)

    # Numpy reference (compute in FP32 for accuracy)
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)

    # Setup GPU
    hsaco_path = ensure_hsaco()
    dev = GPUDevice(device_id)

    try:
        # Allocate device memory
        A_bytes = fp16_to_bytes(A)
        B_bytes = fp16_to_bytes(B)
        C_size = M * N * 2  # FP16

        d_A = dev.malloc(len(A_bytes))
        d_B = dev.malloc(len(B_bytes))
        d_C = dev.malloc(C_size)

        # Upload
        dev.upload(d_A, A_bytes)
        dev.upload(d_B, B_bytes)
        dev.hip.memset(d_C, 0, C_size)

        # Load kernel
        module = dev.load_hsaco(hsaco_path)
        func = dev.get_kernel(module, "gemm_fp16_64x64")

        # Pack kernel params
        params = [
            ctypes.c_uint64(d_A),    # A ptr
            ctypes.c_uint64(d_B),    # B ptr
            ctypes.c_uint64(d_C),    # C ptr
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
            ctypes.c_uint32(0),      # padding
        ]

        grid = (N // 64, M // 64, 1)
        block = (256, 1, 1)

        print(f"Grid: {grid}, Block: {block}")
        print(f"LDS: 8192 bytes")

        # Launch
        dev.launch(func, grid, block, params, shared_mem=8192)
        dev.synchronize()

        # Download result
        C_data = dev.download(d_C, C_size)
        C_gpu = bytes_to_fp16(C_data, (M, N))

        # Compare
        abs_err = np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()
        rel_err = abs_err / (np.abs(C_ref.astype(np.float32)) + 1e-7)
        max_rel_err = rel_err.max()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")
        print(f"Max relative error: {max_rel_err:.6f}")

        # FP16 GEMM should be accurate to ~1e-3 for reasonable sizes
        threshold = 0.1 if K > 256 else 0.01
        if max_err < threshold:
            print(f"PASS (threshold={threshold})")
            return True
        else:
            print(f"FAIL (max_err={max_err} >= {threshold})")
            # Print some values for debugging
            print(f"C_ref[0,:4] = {C_ref[0,:4]}")
            print(f"C_gpu[0,:4] = {C_gpu[0,:4]}")
            print(f"C_ref[-1,-4:] = {C_ref[-1,-4:]}")
            print(f"C_gpu[-1,-4:] = {C_gpu[-1,-4:]}")
            return False

    finally:
        dev.cleanup()


def run_perf_test(M, N, K, n_warmup=5, n_iters=20, device_id=0):
    """Run performance measurement."""
    print(f"\n{'='*60}")
    print(f"Perf GEMM: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(K, N).astype(np.float16) * 0.1

    hsaco_path = ensure_hsaco()
    dev = GPUDevice(device_id)

    try:
        d_A = dev.malloc(M * K * 2)
        d_B = dev.malloc(K * N * 2)
        d_C = dev.malloc(M * N * 2)

        dev.upload(d_A, fp16_to_bytes(A))
        dev.upload(d_B, fp16_to_bytes(B))

        module = dev.load_hsaco(hsaco_path)
        func = dev.get_kernel(module, "gemm_fp16_64x64")

        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_C),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
            ctypes.c_uint32(0),
        ]

        grid = (N // 64, M // 64, 1)
        block = (256, 1, 1)

        # Warmup
        for _ in range(n_warmup):
            dev.launch(func, grid, block, params, shared_mem=8192)
        dev.synchronize()

        # Timed runs
        import time
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            dev.launch(func, grid, block, params, shared_mem=8192)
        dev.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) / n_iters * 1000.0
        # FLOPS = 2 * M * N * K (multiply-add)
        flops = 2.0 * M * N * K
        tflops = flops / (elapsed_ms / 1000.0) / 1e12

        print(f"Time: {elapsed_ms:.3f} ms")
        print(f"TFLOPS: {tflops:.2f}")
        print(f"Efficiency: {tflops / 26.8 * 100:.1f}% of peak (26.8 TFLOPS FP16)")

    finally:
        dev.cleanup()


if __name__ == "__main__":
    print("mi50grad FP16 GEMM kernel test")
    print("=" * 60)

    # Correctness tests
    all_pass = True
    for M, N, K in [(64, 64, 16), (64, 64, 32), (128, 128, 64),
                     (256, 256, 128), (512, 512, 256)]:
        if not run_gemm_test(M, N, K):
            all_pass = False

    if all_pass:
        print("\n\nAll correctness tests PASSED!")

        # Performance tests (Qwen 3.5 27B shapes)
        # QKV projection: M=seqlen, N=3*4096=12288, K=4096
        # Output projection: M=seqlen, N=4096, K=4096
        # FFN up: M=seqlen, N=11008, K=4096
        # FFN down: M=seqlen, N=4096, K=11008
        print("\n\nPerformance benchmarks:")
        for M, N, K in [(256, 256, 256), (1024, 1024, 1024),
                         (4096, 4096, 4096),
                         (128, 4096, 4096), (128, 11008, 4096)]:
            # Round to tile multiples
            M = ((M + 63) // 64) * 64
            N = ((N + 63) // 64) * 64
            K = ((K + 15) // 16) * 16
            run_perf_test(M, N, K)
    else:
        print("\n\nSome correctness tests FAILED!")
        sys.exit(1)
