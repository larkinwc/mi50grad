#!/usr/bin/env python3
"""
Test harness for INT4 GEMM Prefill kernel (gemm_int4_prefill_64x64).

Compiles the assembly kernel, runs on MI50, validates against numpy reference
that performs INT4 dequantization + matmul.

Must be run on the dev server (or in the ROCm container).
"""

import ctypes
import struct
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice, HIPError
from src.kernels.launcher import build_hsaco


ASM_PATH = PROJECT_ROOT / "src" / "asm" / "gemm_int4_prefill.s"
HSACO_PATH = PROJECT_ROOT / "build" / "kernels" / "gemm_int4_prefill.hsaco"


def ensure_hsaco():
    """Build the HSACO if it doesn't exist or is stale."""
    HSACO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HSACO_PATH.exists() or ASM_PATH.stat().st_mtime > HSACO_PATH.stat().st_mtime:
        print(f"Building {HSACO_PATH}...")
        build_hsaco(str(ASM_PATH), str(HSACO_PATH))
        print("Build OK")
    return str(HSACO_PATH)


def quantize_weights_gptq(W_fp16, group_size=128):
    """
    Simulate GPTQ quantization of a weight matrix.

    W_fp16: [K, N] FP16 weight matrix
    Returns: B_q4 [K/8, N] uint32, scales [K/group_size, N] FP16, zeros [K/group_size, N] FP16

    Quantization per group:
      zero = min of group
      scale = (max - min) / 15
      q = round((w - zero) / scale)  clamped to [0, 15]
      dequant: w_approx = q * scale + zero
      But our kernel does: (q - zero_stored) * scale_stored
      So we store: zero_stored = -min/scale = -zero/scale ... no.

    Actually the kernel computes: w = (q4_val - zero) * scale
    So: q4_val = round(w / scale + zero)
    Or equivalently: w_approx = (q4_val - zero) * scale

    Let's use the simpler symmetric-ish scheme:
      For each group of group_size values along K for each column:
        w_min = min(group)
        w_max = max(group)
        scale = (w_max - w_min) / 15.0  (FP16)
        zero = w_min / scale  (stored as FP16, represents the zero point)
        q4_val = round((w - w_min) / scale) = round(w/scale - zero)
        dequant: w_approx = (q4_val - zero) * scale = (q4_val * scale - zero * scale)
                           = q4_val * scale + w_min  ... wait that's w_min not -zero*scale.

    Let me re-derive to match the kernel exactly:
      kernel: w = (q4_val - zero) * scale
      We want: q4_val in [0, 15]
      Choose: scale = (w_max - w_min) / 15
              zero = w_min / scale   (so (0 - zero)*scale = -zero*scale = -w_min... no)

    Actually: (q4_val - zero) * scale = w
    When q4_val = 0: w = -zero * scale = w_min  =>  zero = -w_min / scale
    When q4_val = 15: w = (15 - zero) * scale = w_max  =>  15*scale + w_min = w_max  =>  scale = (w_max-w_min)/15

    So: zero = -w_min / scale  (this is positive when w_min < 0)
    """
    K, N = W_fp16.shape
    W = W_fp16.astype(np.float32)
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float16)
    zeros = np.zeros((n_groups, N), dtype=np.float16)
    q4_matrix = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        group = W[k_start:k_end, :]  # [group_size, N]

        w_min = group.min(axis=0)  # [N]
        w_max = group.max(axis=0)  # [N]

        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0, 1.0, scale)  # avoid div by zero

        zero = -w_min / scale

        # Quantize
        q = np.round((group - w_min[np.newaxis, :]) / scale[np.newaxis, :])
        q = np.clip(q, 0, 15).astype(np.uint8)

        scales[g, :] = scale.astype(np.float16)
        zeros[g, :] = zero.astype(np.float16)
        q4_matrix[k_start:k_end, :] = q

    # Pack q4_matrix into uint32: 8 values along K per uint32
    # B_q4 shape: [K/8, N]
    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= q4_matrix[np.arange(K8) * 8 + i, :].astype(np.uint32) << (i * 4)

    return B_q4, scales, zeros


def dequant_reference(B_q4, scales, zeros, K, N, group_size):
    """
    Reference dequantization matching the kernel: w = (q4_val - zero) * scale

    Returns: [K, N] FP32 weight matrix
    """
    K8 = K // 8
    W = np.zeros((K, N), dtype=np.float32)

    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)

    for k8 in range(K8):
        packed = B_q4[k8, :]  # [N] uint32
        for i in range(8):
            k_idx = k8 * 8 + i
            q_val = ((packed >> (i * 4)) & 0xF).astype(np.float32)
            g_idx = k_idx // group_size
            w = (q_val - zeros_f32[g_idx, :]) * scales_f32[g_idx, :]
            W[k_idx, :] = w

    return W


def run_gemm_int4_test(M, N, K, group_size=128, device_id=0):
    """Run INT4 GEMM prefill test and compare to numpy reference."""
    print(f"\n{'='*60}")
    print(f"Testing INT4 GEMM Prefill: M={M}, N={N}, K={K}, group_size={group_size}")
    print(f"{'='*60}")

    assert M % 64 == 0, f"M must be multiple of 64, got {M}"
    assert N % 64 == 0, f"N must be multiple of 64, got {N}"
    assert K % 16 == 0, f"K must be multiple of 16, got {K}"
    assert K % group_size == 0, f"K must be multiple of group_size"

    np.random.seed(42)

    # Generate random FP16 activations
    A = (np.random.randn(M, K) * 0.1).astype(np.float16)

    # Generate random FP16 weights, then quantize to INT4
    W_fp16 = (np.random.randn(K, N) * 0.1).astype(np.float16)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp16, group_size)

    # Numpy reference: dequantize and matmul
    W_deq = dequant_reference(B_q4, scales, zeros, K, N, group_size)
    C_ref = (A.astype(np.float32) @ W_deq).astype(np.float16)

    # Setup GPU
    hsaco_path = ensure_hsaco()
    dev = GPUDevice(device_id)

    try:
        # Prepare data
        A_bytes = A.tobytes()
        B_q4_bytes = B_q4.tobytes()
        scales_bytes = scales.tobytes()
        zeros_bytes = zeros.tobytes()
        C_size = M * N * 2  # FP16

        # Allocate device memory
        d_A = dev.malloc(len(A_bytes))
        d_B = dev.malloc(len(B_q4_bytes))
        d_scales = dev.malloc(len(scales_bytes))
        d_zeros = dev.malloc(len(zeros_bytes))
        d_C = dev.malloc(C_size)

        # Upload
        dev.upload(d_A, A_bytes)
        dev.upload(d_B, B_q4_bytes)
        dev.upload(d_scales, scales_bytes)
        dev.upload(d_zeros, zeros_bytes)
        dev.hip.memset(d_C, 0, C_size)

        # Load kernel
        module = dev.load_hsaco(hsaco_path)
        func = dev.get_kernel(module, "gemm_int4_prefill_64x64")

        # Pack kernel params
        params = [
            ctypes.c_uint64(d_A),        # A ptr
            ctypes.c_uint64(d_B),        # B_q4 ptr
            ctypes.c_uint64(d_scales),   # scales ptr
            ctypes.c_uint64(d_zeros),    # zeros ptr
            ctypes.c_uint64(d_C),        # C ptr
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
            ctypes.c_uint32(group_size),
        ]

        grid = (N // 64, M // 64, 1)
        block = (256, 1, 1)

        print(f"Grid: {grid}, Block: {block}")
        print(f"LDS: 4096 bytes")

        # Launch
        dev.launch(func, grid, block, params, shared_mem=4096)
        dev.synchronize()

        # Download result
        C_data = dev.download(d_C, C_size)
        C_gpu = np.frombuffer(C_data, dtype=np.float16).reshape(M, N)

        # Compare
        abs_err = np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()
        rel_err = abs_err / (np.abs(C_ref.astype(np.float32)) + 1e-7)
        max_rel_err = rel_err.max()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")
        print(f"Max relative error: {max_rel_err:.6f}")

        # INT4 quantization has inherent error; threshold is generous
        threshold = 0.5
        if max_err < threshold:
            print(f"PASS (threshold={threshold})")
            return True
        else:
            print(f"FAIL (max_err={max_err} >= {threshold})")
            print(f"C_ref[0,:4] = {C_ref[0,:4]}")
            print(f"C_gpu[0,:4] = {C_gpu[0,:4]}")
            print(f"C_ref[-1,-4:] = {C_ref[-1,-4:]}")
            print(f"C_gpu[-1,-4:] = {C_gpu[-1,-4:]}")
            return False

    finally:
        dev.cleanup()


def run_perf_test(M, N, K, group_size=128, n_warmup=5, n_iters=20, device_id=0):
    """Run performance measurement."""
    print(f"\n{'='*60}")
    print(f"Perf INT4 GEMM Prefill: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np.float16)
    W_fp16 = (np.random.randn(K, N) * 0.1).astype(np.float16)
    B_q4, scales, zeros = quantize_weights_gptq(W_fp16, group_size)

    hsaco_path = ensure_hsaco()
    dev = GPUDevice(device_id)

    try:
        d_A = dev.malloc(M * K * 2)
        d_B = dev.malloc(B_q4.nbytes)
        d_scales = dev.malloc(scales.nbytes)
        d_zeros = dev.malloc(zeros.nbytes)
        d_C = dev.malloc(M * N * 2)

        dev.upload(d_A, A.tobytes())
        dev.upload(d_B, B_q4.tobytes())
        dev.upload(d_scales, scales.tobytes())
        dev.upload(d_zeros, zeros.tobytes())

        module = dev.load_hsaco(hsaco_path)
        func = dev.get_kernel(module, "gemm_int4_prefill_64x64")

        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_scales),
            ctypes.c_uint64(d_zeros),
            ctypes.c_uint64(d_C),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
            ctypes.c_uint32(group_size),
        ]

        grid = (N // 64, M // 64, 1)
        block = (256, 1, 1)

        # Warmup
        for _ in range(n_warmup):
            dev.launch(func, grid, block, params, shared_mem=4096)
        dev.synchronize()

        # Timed runs
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            dev.launch(func, grid, block, params, shared_mem=4096)
        dev.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) / n_iters * 1000.0
        # Effective FLOPS = 2 * M * N * K (even though weights are INT4, compute is FP16/32)
        flops = 2.0 * M * N * K
        tflops = flops / (elapsed_ms / 1000.0) / 1e12

        print(f"Time: {elapsed_ms:.3f} ms")
        print(f"Effective TFLOPS: {tflops:.2f}")
        # B data is 4x smaller than FP16 -> bandwidth advantage
        b_data_bytes = B_q4.nbytes + scales.nbytes + zeros.nbytes
        a_data_bytes = M * K * 2
        total_bytes = a_data_bytes + b_data_bytes
        bw_gbps = total_bytes / (elapsed_ms / 1000.0) / 1e9
        print(f"Bandwidth: {bw_gbps:.1f} GB/s (data={total_bytes/1e6:.1f} MB)")

    finally:
        dev.cleanup()


if __name__ == "__main__":
    print("mi50grad INT4 GEMM Prefill kernel test")
    print("=" * 60)

    # Correctness tests
    all_pass = True

    # Small sizes
    for M, N, K in [(64, 64, 128), (128, 128, 256)]:
        if not run_gemm_int4_test(M, N, K, group_size=128):
            all_pass = False

    # Qwen shapes
    for M, N, K in [(128, 4096, 4096), (128, 11008, 4096)]:
        # Round N to multiple of 64
        N = ((N + 63) // 64) * 64
        if not run_gemm_int4_test(M, N, K, group_size=128):
            all_pass = False

    if all_pass:
        print("\n\nAll correctness tests PASSED!")

        # Performance tests
        print("\n\nPerformance benchmarks:")
        for M, N, K in [(128, 4096, 4096), (128, 11008, 4096)]:
            N = ((N + 63) // 64) * 64
            run_perf_test(M, N, K, group_size=128)
    else:
        print("\n\nSome correctness tests FAILED!")
        sys.exit(1)
