#!/usr/bin/env python3
"""
Test for fused INT4 split-K GEMV kernel (m1-fused-int4-splitk).

Verifies that the fused kernel:
1. Produces correct FP16 output in a single launch (no separate memset/convert)
2. Has max absolute error < 1e-2 vs reference (per INT4 quantized tolerance)
3. Works for key model dimensions: N=4096,K=4096 and N=11008,K=4096
"""

import ctypes
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"

TOLERANCE = 1e-2  # INT4 quantized tolerance


def build_fused_kernel():
    """Build the fused INT4 GEMV kernel as HSACO and return the path."""
    import subprocess

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_src = PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v2.hip"
    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"

    # Build using --genco to produce HSACO code object (same as engine does)
    cmd = [
        "/opt/rocm/bin/hipcc",
        "--genco",
        "--offload-arch=gfx906",
        "-O3",
        "-o", str(hsaco_path),
        str(hip_src),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build stderr: {result.stderr}")
        raise RuntimeError(f"hipcc build failed: {result.stdout}\n{result.stderr}")
    return hsaco_path


def pack_int4_weights(W_q4):
    """Pack INT4 weight matrix [K, N] into [K/8, N] uint32.

    8 consecutive K values for one N column packed into one uint32.
    """
    K, N = W_q4.shape
    assert K % 8 == 0
    B_packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(K // 8):
        for bit in range(8):
            B_packed[i] |= (W_q4[i * 8 + bit].astype(np.uint32) & 0xF) << (bit * 4)
    return B_packed


def compute_reference(A, W_q4, scales, zeros, K, N, group_size):
    """Compute reference FP32 output: dequantize W then multiply."""
    A32 = A.astype(np.float32)
    W_fp32 = np.zeros((K, N), dtype=np.float32)
    num_groups = K // group_size
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        for n in range(N):
            s = float(scales[g, n])
            z = float(zeros[g, n])
            for k in range(k_start, k_end):
                W_fp32[k, n] = (float(W_q4[k, n]) - z) * s
    return A32 @ W_fp32  # [N] FP32


def compute_reference_fast(A, W_q4, scales, zeros, K, N, group_size):
    """Vectorized reference computation (faster for large matrices)."""
    A32 = A.astype(np.float32)
    num_groups = K // group_size
    ref = np.zeros(N, dtype=np.float32)
    W_q4_f32 = W_q4.astype(np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        s = scales[g].astype(np.float32)   # [N]
        z = zeros[g].astype(np.float32)    # [N]
        W_block = (W_q4_f32[k_start:k_end] - z[None, :]) * s[None, :]  # [group_size, N]
        ref += A32[k_start:k_end] @ W_block  # [N]
    return ref


def run_fused_splitk_test(K, N, group_size, k_splits, dev, description):
    """
    Run the fused INT4 split-K GEMV test.

    The fused kernel (gemv_int4_v2_fused) uses a persistent FP32 accumulator and
    done-counter buffer that are reset by the kernel itself. No separate memset
    or fp32_to_fp16 launch is needed.
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"K={K}, N={N}, group_size={group_size}, k_splits={k_splits}")
    print(f"{'='*60}")

    assert K % 8 == 0, "K must be divisible by 8"
    assert K % group_size == 0, "K must be divisible by group_size"

    np.random.seed(42)

    # Generate random activations
    A = (np.random.randn(K) * 0.1).astype(np.float16)

    # Generate random INT4 weights (0-15)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)

    # Generate scales and zeros per group
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)

    # Pack weights
    B_packed = pack_int4_weights(W_q4)

    # Compute reference
    ref_fp32 = compute_reference_fast(A, W_q4, scales, zeros, K, N, group_size)
    ref_fp16 = ref_fp32.astype(np.float16)

    # Load fused kernel
    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_fused = dev.get_kernel(module, "gemv_int4_v2_fused")

    # Allocate GPU buffers
    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    # Persistent FP32 accumulator buffer — zeroed ONCE at allocation, reset by kernel
    d_fp32 = dev.malloc(N * 4)
    dev.memset(d_fp32, 0, N * 4)
    # Persistent done-counter buffer — zeroed ONCE at allocation, reset by kernel
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)
    # FP16 output buffer
    d_C = dev.malloc(N * 2)

    # Upload inputs
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())

    # Launch the fused kernel — single launch, no pre-memset, no post-convert
    grid_x = (N + 255) // 256
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32),    # persistent FP32 accumulator
        ctypes.c_uint64(d_done),    # persistent done counter
        ctypes.c_uint64(d_C),       # FP16 output directly
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits),
    ]
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()

    # Download and compare
    out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)

    abs_err = np.abs(out.astype(np.float32) - ref_fp16.astype(np.float32))
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())

    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")
    print(f"ref[:4]  = {ref_fp16[:4]}")
    print(f"out[:4]  = {out[:4]}")

    # Verify that buffers are zeroed after the call (kernel reset them)
    fp32_after = np.frombuffer(dev.download(d_fp32, N * 4), dtype=np.float32)
    done_after = np.frombuffer(dev.download(d_done, N * 4), dtype=np.uint32)
    if np.any(fp32_after != 0.0) or np.any(done_after != 0):
        print(f"  WARNING: Persistent buffers not fully reset after kernel call!")
        print(f"  fp32 non-zero: {np.sum(fp32_after != 0)}, done non-zero: {np.sum(done_after != 0)}")

    # Test that a second call also works (verifies buffers reset correctly)
    np.random.seed(123)
    A2 = (np.random.randn(K) * 0.1).astype(np.float16)
    dev.upload(d_A, A2.tobytes())
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()
    out2 = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
    ref2 = compute_reference_fast(A2, W_q4, scales, zeros, K, N, group_size).astype(np.float16)
    max_err2 = float(np.abs(out2.astype(np.float32) - ref2.astype(np.float32)).max())
    print(f"Second call max error: {max_err2:.6f}")

    # Cleanup
    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_fp32)
    dev.free(d_done)
    dev.free(d_C)

    ok = max_err < TOLERANCE and max_err2 < TOLERANCE
    if ok:
        print(f"PASS (max_err={max_err:.6f}, max_err2={max_err2:.6f})")
        return True, max_err
    else:
        print(f"FAIL (max_err={max_err:.6f}, max_err2={max_err2:.6f})")
        return False, max(max_err, max_err2)


def run_perf_comparison(K, N, group_size, k_splits, dev, description):
    """Compare latency: old 3-launch vs new 1-launch fused path."""
    print(f"\n{'='*60}")
    print(f"Performance: {description}")
    print(f"{'='*60}")

    np.random.seed(99)
    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_packed = pack_int4_weights(W_q4)

    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_splitk = dev.get_kernel(module, "gemv_int4_v2_splitk")
    func_convert = dev.get_kernel(module, "fp32_to_fp16")
    func_fused = dev.get_kernel(module, "gemv_int4_v2_fused")

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_fp32 = dev.malloc(N * 4)
    d_fp32_fused = dev.malloc(N * 4)
    d_done = dev.malloc(N * 4)
    d_C_old = dev.malloc(N * 2)
    d_C_new = dev.malloc(N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())
    dev.memset(d_fp32_fused, 0, N * 4)
    dev.memset(d_done, 0, N * 4)

    grid_x = (N + 255) // 256

    old_params_splitk = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32), ctypes.c_uint32(K), ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
    ]
    old_params_convert = [
        ctypes.c_uint64(d_fp32), ctypes.c_uint64(d_C_old), ctypes.c_uint32(N),
    ]
    new_params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32_fused), ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_C_new), ctypes.c_uint32(K), ctypes.c_uint32(N),
        ctypes.c_uint32(group_size), ctypes.c_uint32(k_splits),
    ]

    # Warmup
    for _ in range(5):
        dev.memset(d_fp32, 0, N * 4)
        dev.launch(func_splitk, (grid_x, k_splits, 1), (256, 1, 1), old_params_splitk)
        dev.launch(func_convert, (grid_x, 1, 1), (256, 1, 1), old_params_convert)
        dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), new_params)
    dev.synchronize()

    N_ITERS = 100

    # Benchmark old path (3 launches: memset + splitk + convert)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.memset(d_fp32, 0, N * 4)
        dev.launch(func_splitk, (grid_x, k_splits, 1), (256, 1, 1), old_params_splitk)
        dev.launch(func_convert, (grid_x, 1, 1), (256, 1, 1), old_params_convert)
    dev.synchronize()
    t1 = time.perf_counter()
    old_us = (t1 - t0) / N_ITERS * 1e6
    print(f"Old path (memset+splitk+convert): {old_us:.1f} us/iter")

    # Benchmark new path (1 launch: fused)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), new_params)
    dev.synchronize()
    t1 = time.perf_counter()
    new_us = (t1 - t0) / N_ITERS * 1e6
    print(f"New path (fused):                 {new_us:.1f} us/iter")
    speedup = old_us / new_us
    print(f"Speedup: {speedup:.2f}x")

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_fp32)
    dev.free(d_fp32_fused)
    dev.free(d_done)
    dev.free(d_C_old)
    dev.free(d_C_new)

    return old_us, new_us


def test_engine_single_launch():
    """Verify that _launch_gemv_int4 does NOT call memset or fp32_to_fp16 separately.

    Parses the engine.py code path by inspecting the _launch_gemv_int4 method directly.
    """
    print(f"\n{'='*60}")
    print("Test: Engine uses single-launch path (no separate memset/convert)")
    print(f"{'='*60}")

    import inspect
    from src.inference.engine import InferenceEngine

    src = inspect.getsource(InferenceEngine._launch_gemv_int4)

    # Check that the method uses the fused kernel
    has_fused = "gemv_int4_v2_fused" in src

    # Check that the method does NOT call device.memset (actual call, not in comments)
    # Look for actual API call patterns, not just the word "memset"
    import re
    # Find lines that are not comments and contain device.memset or .memset(
    non_comment_lines = [
        line for line in src.split('\n')
        if not line.strip().startswith('#')
    ]
    non_comment_src = '\n'.join(non_comment_lines)
    has_memset_call = bool(re.search(r'\.memset\s*\(', non_comment_src))
    has_fp32_to_fp16_call = "fp32_to_fp16" in non_comment_src

    print(f"  Uses gemv_int4_v2_fused:                {has_fused}")
    print(f"  Contains device.memset() call:          {has_memset_call}")
    print(f"  Contains fp32_to_fp16 kernel call:      {has_fp32_to_fp16_call}")

    passed = not has_memset_call and not has_fp32_to_fp16_call and has_fused
    if passed:
        print("PASS")
    else:
        print("FAIL: Engine still uses 3-launch pattern or missing fused kernel")
    return passed


def main():
    print("Building fused INT4 GEMV kernel...")
    build_fused_kernel()
    print("Build successful.\n")

    dev = GPUDevice(0)
    all_pass = True
    errors = {}

    try:
        # Test 1: N=4096, K=4096, group_size=128 (feature required)
        passed, max_err = run_fused_splitk_test(
            K=4096, N=4096, group_size=128, k_splits=16, dev=dev,
            description="Fused INT4 GEMV: N=4096, K=4096"
        )
        errors["N=4096,K=4096"] = max_err
        if not passed:
            all_pass = False

        # Test 2: N=11008, K=4096, group_size=128 (FFN gate/up dimensions)
        passed, max_err = run_fused_splitk_test(
            K=4096, N=11008, group_size=128, k_splits=16, dev=dev,
            description="Fused INT4 GEMV: N=11008, K=4096"
        )
        errors["N=11008,K=4096"] = max_err
        if not passed:
            all_pass = False

        # Test 3: N=5120, K=5120, group_size=128 (hidden size dims)
        passed, max_err = run_fused_splitk_test(
            K=5120, N=5120, group_size=128, k_splits=16, dev=dev,
            description="Fused INT4 GEMV: N=5120, K=5120"
        )
        errors["N=5120,K=5120"] = max_err
        if not passed:
            all_pass = False

        # Test 4: N=4096, K=4096, with k_splits=8
        passed, max_err = run_fused_splitk_test(
            K=4096, N=4096, group_size=128, k_splits=8, dev=dev,
            description="Fused INT4 GEMV: N=4096, K=4096, k_splits=8"
        )
        errors["N=4096,K=4096,k_splits=8"] = max_err
        if not passed:
            all_pass = False

        # Test 5: Engine structural check (single-launch path)
        if not test_engine_single_launch():
            all_pass = False

        # Performance comparison
        print("\n" + "="*60)
        print("Performance Benchmarks")
        run_perf_comparison(
            K=4096, N=4096, group_size=128, k_splits=16, dev=dev,
            description="N=4096, K=4096"
        )
        run_perf_comparison(
            K=4096, N=11008, group_size=128, k_splits=16, dev=dev,
            description="N=11008, K=4096"
        )

    finally:
        dev.cleanup()

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, err in errors.items():
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"  {name}: max_err={err:.6f} [{status}]")

    if all_pass:
        print("\nAll fused INT4 GEMV tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
