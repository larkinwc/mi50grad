#!/usr/bin/env python3
"""Verify GEMV kernels produce correct results vs numpy reference."""

import sys
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco


def test_gemv_fp16(device, func, K, N, name=""):
    """Test FP16 GEMV: out[N] = W[N,K] * x[K]."""
    np.random.seed(42)
    W = np.random.randn(N, K).astype(np.float16)
    x = np.random.randn(K).astype(np.float16)

    # CPU reference
    ref = W.astype(np.float32) @ x.astype(np.float32)

    # GPU
    d_x = device.malloc(K * 2)
    d_W = device.malloc(N * K * 2)
    d_out = device.malloc(N * 2)
    device.upload(d_x, x.tobytes())
    device.upload(d_W, W.tobytes())

    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_W),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
    ]
    device.launch(func, (N, 1, 1), (256, 1, 1), params, shared_mem=1024)

    gpu_out = np.frombuffer(device.download(d_out, N * 2), dtype=np.float16)
    gpu_f32 = gpu_out.astype(np.float32)

    max_err = np.max(np.abs(ref - gpu_f32))
    rel_err = max_err / (np.max(np.abs(ref)) + 1e-10)
    matches = np.allclose(ref, gpu_f32, atol=1.0, rtol=0.05)

    print(f"GEMV FP16 {name} K={K} N={N}: max_err={max_err:.4f}, "
          f"rel_err={rel_err:.6f}, ref_range=[{ref.min():.4f},{ref.max():.4f}], "
          f"gpu_range=[{gpu_f32.min():.4f},{gpu_f32.max():.4f}] "
          f"{'PASS' if matches else 'FAIL'}")

    device.free(d_x)
    device.free(d_W)
    device.free(d_out)
    return matches


def test_gemv_int4(device, func, K, N, group_size=128, name=""):
    """Test INT4 GEMV: out[N] = dequant(W_q4) * x."""
    np.random.seed(42)
    x = np.random.randn(K).astype(np.float16)

    # Generate random INT4 weights
    num_groups = K // group_size
    qweight = np.random.randint(0, 256, size=(K // 8, N), dtype=np.int32)
    # For each pair of nibbles, this gives random 4-bit values
    for i in range(4):
        qweight |= np.random.randint(0, 16, size=(K // 8, N), dtype=np.int32) << (i * 8)

    scales = (np.random.randn(num_groups, N) * 0.01).astype(np.float16)
    zeros = np.full((num_groups, N), 8, dtype=np.float16)  # symmetric: zero=8

    # CPU reference dequant
    ref_out = np.zeros(N, dtype=np.float32)
    x_f32 = x.astype(np.float32)
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)

    for col in range(min(N, 4)):  # Just check first 4 columns for speed
        acc = 0.0
        for k_row in range(K // 8):
            packed = int(qweight[k_row, col])
            for bit in range(8):
                k_idx = k_row * 8 + bit
                int4_val = (packed >> (bit * 4)) & 0xF
                g = k_idx // group_size
                w = (int4_val - zeros_f32[g, col]) * scales_f32[g, col]
                acc += w * x_f32[k_idx]
        ref_out[col] = acc

    # GPU
    d_x = device.malloc(K * 2)
    d_qw = device.malloc(qweight.nbytes)
    d_sc = device.malloc(scales.nbytes)
    d_zr = device.malloc(zeros.nbytes)
    d_out = device.malloc(N * 2)

    device.upload(d_x, x.tobytes())
    device.upload(d_qw, qweight.tobytes())
    device.upload(d_sc, scales.tobytes())
    device.upload(d_zr, zeros.tobytes())

    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_qw),
        ctypes.c_uint64(d_sc),
        ctypes.c_uint64(d_zr),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]
    device.launch(func, (N, 1, 1), (256, 1, 1), params, shared_mem=1024)

    gpu_out = np.frombuffer(device.download(d_out, N * 2), dtype=np.float16)
    gpu_f32 = gpu_out.astype(np.float32)

    # Compare first 4 columns
    errs = []
    for col in range(min(N, 4)):
        err = abs(ref_out[col] - gpu_f32[col])
        rel = err / (abs(ref_out[col]) + 1e-10)
        errs.append(err)
        print(f"  col {col}: ref={ref_out[col]:.6f}, gpu={gpu_f32[col]:.6f}, "
              f"err={err:.6f}, rel={rel:.6f}")

    max_err = max(errs)
    print(f"INT4 GEMV {name} K={K} N={N}: max_err={max_err:.6f} "
          f"{'PASS' if max_err < 1.0 else 'FAIL'}")

    device.free(d_x)
    device.free(d_qw)
    device.free(d_sc)
    device.free(d_zr)
    device.free(d_out)
    return max_err < 1.0


def main():
    device = GPUDevice(0)

    # Build kernels
    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    fp16_asm = PROJECT_ROOT / "src" / "asm" / "gemv_fp16.s"
    fp16_hsaco = BUILD_DIR / "gemv_fp16.hsaco"
    build_hsaco(str(fp16_asm), str(fp16_hsaco))
    mod_fp16 = device.load_hsaco(str(fp16_hsaco))
    func_fp16 = device.get_kernel(mod_fp16, "gemv_fp16")

    int4_asm = PROJECT_ROOT / "src" / "asm" / "gemm_int4.s"
    int4_hsaco = BUILD_DIR / "gemm_int4.hsaco"
    build_hsaco(str(int4_asm), str(int4_hsaco))
    mod_int4 = device.load_hsaco(str(int4_hsaco))
    func_int4 = device.get_kernel(mod_int4, "gemv_int4_fp16")

    print("=== FP16 GEMV Tests ===")
    # Test shapes from Qwen
    test_gemv_fp16(device, func_fp16, 5120, 6144, "q_proj")
    test_gemv_fp16(device, func_fp16, 5120, 1024, "k_proj")
    test_gemv_fp16(device, func_fp16, 6144, 5120, "o_proj")
    test_gemv_fp16(device, func_fp16, 5120, 10240, "in_proj_qkv")

    print("\n=== INT4 GEMV Tests ===")
    # Test with K that IS multiple of 2048
    test_gemv_int4(device, func_int4, 2048, 64, name="small_mult")
    # Test with K=5120 (NOT multiple of 2048 — the bug case)
    test_gemv_int4(device, func_int4, 5120, 64, name="gate_proj_K5120")
    # Test with Qwen shapes
    test_gemv_int4(device, func_int4, 5120, 256, name="gate_proj")
    test_gemv_int4(device, func_int4, 17408, 64, name="down_proj_K17408")

    device.cleanup()


if __name__ == "__main__":
    main()
