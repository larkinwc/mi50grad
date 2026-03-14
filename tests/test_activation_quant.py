#!/usr/bin/env python3
"""Test activation quantization kernel: dynamic per-tensor INT8 quantization of FP16 activations.

Tests:
1. Correctness: scale = max_abs / 127.0, output = round(x / scale), clamp(-128, 127)
2. Round-trip error: quantize → dequantize should be within expected INT8 noise
3. Typical activation sizes: 4096 and 11008 elements
4. Edge cases: minimum dimensions, all-same values, near-zero
"""

import sys
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco

dev = GPUDevice(0)
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

hip_path = str(PROJECT_ROOT / "src" / "kernels" / "activation_quant.hip")
hsaco_path = str(BUILD_DIR / "activation_quant.hsaco")
print("Building activation_quant.hip ...")
build_hip_hsaco(hip_path, hsaco_path)
module = dev.load_hsaco(hsaco_path)

reduce_func = dev.get_kernel(module, "activation_quant_reduce")
quant_func = dev.get_kernel(module, "activation_quant_quant")

print("Build OK")


def ref_activation_quant(x_fp16):
    """Reference implementation of dynamic per-tensor INT8 quantization."""
    x = x_fp16.astype(np.float32)
    max_abs = np.max(np.abs(x))
    # Avoid division by zero
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, np.float32(scale)


def launch_activation_quant(x_fp16):
    """Launch kernel and return (q_int8, scale_fp32)."""
    n = x_fp16.size
    x_bytes = x_fp16.astype(np.float16).tobytes()

    d_x = dev.malloc(len(x_bytes))
    d_out = dev.malloc(n)           # INT8 output
    d_scale = dev.malloc(4)         # FP32 scale
    d_max = dev.malloc(4)           # FP32 max_abs (intermediate)

    dev.upload(d_x, x_bytes)
    dev.hip.memset(d_max, 0, 4)
    dev.hip.memset(d_scale, 0, 4)
    dev.hip.memset(d_out, 0, n)

    # Pass 1: reduction to find max_abs
    # Grid: (ceil(n / 1024), 1, 1), Block: (256, 1, 1) — each WG reduces 4*256=1024 elems
    # For very large tensors, we do two-level reduction:
    # - Each WG of 256 threads reduces n/NUM_WG elements down to one partial max
    # - Then we pass partial max buffer through a second kernel call
    # Simple approach: single kernel with atomicMax on a global FP32 max buffer
    # (reset to 0 before call)
    num_blocks = (n + 1023) // 1024
    params_reduce = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_max),
        ctypes.c_uint32(n),
    ]
    dev.launch(reduce_func, (num_blocks, 1, 1), (256, 1, 1), params_reduce)
    dev.synchronize()

    # Pass 2: quantize elements using scale = max_abs / 127.0
    num_blocks_q = (n + 255) // 256
    params_quant = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_out),
        ctypes.c_uint64(d_max),    # in: max_abs, out: scale (reuse buffer, but separate)
        ctypes.c_uint64(d_scale),  # out: scale
        ctypes.c_uint32(n),
    ]
    dev.launch(quant_func, (num_blocks_q, 1, 1), (256, 1, 1), params_quant)
    dev.synchronize()

    q_bytes = dev.download(d_out, n)
    scale_bytes = dev.download(d_scale, 4)

    q_int8 = np.frombuffer(q_bytes, dtype=np.int8).copy()
    scale_fp32 = np.frombuffer(scale_bytes, dtype=np.float32)[0]

    dev.free(d_x)
    dev.free(d_out)
    dev.free(d_scale)
    dev.free(d_max)

    return q_int8, scale_fp32


print("\n=== Correctness Tests ===")
all_pass = True

# --- Test 1: basic correctness for typical sizes ---
for n in [64, 256, 1024, 4096, 11008]:
    np.random.seed(42 + n)
    x = np.random.randn(n).astype(np.float16)

    q_ref, scale_ref = ref_activation_quant(x)
    q_gpu, scale_gpu = launch_activation_quant(x)

    scale_ok = abs(float(scale_gpu) - float(scale_ref)) / (float(scale_ref) + 1e-8) < 1e-4
    q_ok = np.all(q_gpu == q_ref)

    # Round-trip: dequantize and check error
    dequant_ref = q_ref.astype(np.float32) * float(scale_ref)
    dequant_gpu = q_gpu.astype(np.float32) * float(scale_gpu)
    x_fp32 = x.astype(np.float32)

    rt_err_ref = np.max(np.abs(x_fp32 - dequant_ref))
    rt_err_gpu = np.max(np.abs(x_fp32 - dequant_gpu))

    # Expected INT8 noise: scale/2 (half of quantization step)
    expected_noise = float(scale_ref) * 0.6

    ok = scale_ok and q_ok and rt_err_gpu <= expected_noise
    print(f"  n={n:6d}: scale_ref={scale_ref:.6f} scale_gpu={scale_gpu:.6f} "
          f"q_match={'YES' if q_ok else 'NO'} "
          f"rt_err={rt_err_gpu:.5f} noise_budget={expected_noise:.5f} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        if not scale_ok:
            print(f"    scale mismatch: ref={scale_ref} gpu={scale_gpu}")
        if not q_ok:
            diff = np.where(q_gpu != q_ref)[0]
            print(f"    {len(diff)} q elements differ, first 5: {diff[:5]}")
            print(f"    q_ref[0:8]: {q_ref[:8]}, q_gpu[0:8]: {q_gpu[:8]}")

# --- Test 2: all-zero input ---
x_zeros = np.zeros(4096, dtype=np.float16)
q_gpu, scale_gpu = launch_activation_quant(x_zeros)
ok = np.all(q_gpu == 0)
print(f"\n  all-zeros n=4096: scale={scale_gpu:.6f} all_q_zero={'YES' if ok else 'NO'} "
      f"{'PASS' if ok else 'FAIL'}")
if not ok:
    all_pass = False

# --- Test 3: single large value + many near-zeros ---
x_spike = np.zeros(4096, dtype=np.float32)
x_spike[0] = 10.0
x_spike[1:] = 0.001
x_spike = x_spike.astype(np.float16)
q_ref, scale_ref = ref_activation_quant(x_spike)
q_gpu, scale_gpu = launch_activation_quant(x_spike)
scale_ok = abs(float(scale_gpu) - float(scale_ref)) / (float(scale_ref) + 1e-8) < 1e-4
q0_ok = q_gpu[0] == q_ref[0]  # The max element should be exactly 127
print(f"\n  spike test n=4096: scale_ref={scale_ref:.6f} scale_gpu={scale_gpu:.6f} "
      f"q[0]_ref={q_ref[0]} q[0]_gpu={q_gpu[0]} "
      f"{'PASS' if (scale_ok and q0_ok) else 'FAIL'}")
if not (scale_ok and q0_ok):
    all_pass = False

# --- Test 4: negative values ---
np.random.seed(100)
x_neg = -np.abs(np.random.randn(4096)).astype(np.float16)
q_ref, scale_ref = ref_activation_quant(x_neg)
q_gpu, scale_gpu = launch_activation_quant(x_neg)
scale_ok = abs(float(scale_gpu) - float(scale_ref)) / (float(scale_ref) + 1e-8) < 1e-4
q_ok = np.all(q_gpu == q_ref)
print(f"\n  neg-only n=4096: scale_ref={scale_ref:.6f} scale_gpu={scale_gpu:.6f} "
      f"q_match={'YES' if q_ok else 'NO'} "
      f"{'PASS' if (scale_ok and q_ok) else 'FAIL'}")
if not (scale_ok and q_ok):
    all_pass = False

# --- Test 5: round-trip error distribution for typical activations ---
print("\n=== Round-Trip Error Distribution ===")
for n in [4096, 11008]:
    np.random.seed(77)
    x = (np.random.randn(n) * 2.0).astype(np.float16)  # typical activation range ~2-3
    q_ref, scale_ref = ref_activation_quant(x)
    q_gpu, scale_gpu = launch_activation_quant(x)

    dequant = q_gpu.astype(np.float32) * float(scale_gpu)
    x_fp32 = x.astype(np.float32)
    abs_err = np.abs(x_fp32 - dequant)
    max_err = np.max(abs_err)
    mean_err = np.mean(abs_err)

    # Expected: max error ≤ 0.5 * scale (half quantization step), typically much less
    noise_budget = float(scale_ref) * 0.6
    ok = max_err <= noise_budget
    print(f"  n={n:6d}: scale={scale_gpu:.5f} max_err={max_err:.6f} mean_err={mean_err:.6f} "
          f"noise_budget={noise_budget:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

# --- Test 6: performance ---
print("\n=== Performance ===")
import time

for n in [4096, 11008, 16384]:
    np.random.seed(42)
    x = np.random.randn(n).astype(np.float16)
    x_bytes = x.tobytes()

    d_x = dev.malloc(len(x_bytes))
    d_out = dev.malloc(n)
    d_scale = dev.malloc(4)
    d_max = dev.malloc(4)
    dev.upload(d_x, x_bytes)

    num_blocks = (n + 1023) // 1024
    num_blocks_q = (n + 255) // 256
    params_reduce = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_max), ctypes.c_uint32(n)]
    params_quant = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_out),
                    ctypes.c_uint64(d_max), ctypes.c_uint64(d_scale), ctypes.c_uint32(n)]

    # Warmup
    for _ in range(5):
        dev.hip.memset(d_max, 0, 4)
        dev.launch(reduce_func, (num_blocks, 1, 1), (256, 1, 1), params_reduce)
        dev.launch(quant_func, (num_blocks_q, 1, 1), (256, 1, 1), params_quant)
    dev.synchronize()

    # Benchmark
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.hip.memset(d_max, 0, 4)
        dev.launch(reduce_func, (num_blocks, 1, 1), (256, 1, 1), params_reduce)
        dev.launch(quant_func, (num_blocks_q, 1, 1), (256, 1, 1), params_quant)
    dev.synchronize()
    t_us = (time.perf_counter() - t0) / iters * 1e6

    bw_gbps = (n * 2 + n) / 1e9 / (t_us * 1e-6)  # input FP16 + output INT8
    print(f"  n={n:6d}: {t_us:.2f} us, {bw_gbps:.1f} GB/s (2-kernel)")

    dev.free(d_x)
    dev.free(d_out)
    dev.free(d_scale)
    dev.free(d_max)

dev.cleanup()
print(f"\n{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
if not all_pass:
    sys.exit(1)
