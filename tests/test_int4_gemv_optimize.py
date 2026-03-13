#!/usr/bin/env python3
"""
Test for optimized INT4 GEMV kernels (m2-int4-gemv-optimize).

Verifies:
1. Optimized v2 (ubfe + power-of-2 scale lookup) correctness for key dimensions
2. v3 variants (t4, t8, t16, dpp) correctness vs reference
3. Benchmark comparison: optimized v2 vs v3 variants
4. DPP wave reduction variant (v3_dpp) correctness
5. Best kernel wired into engine as default

Reference: dequantize-then-multiply on CPU (float32)
Tolerance: max abs error < 1e-2 (INT4 quantized)
"""

import ctypes
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
TOLERANCE = 1e-2


def build_kernel(hip_src_name, hsaco_name=None):
    """Build a HIP kernel as HSACO code object."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if hsaco_name is None:
        hsaco_name = hip_src_name
    hip_src = PROJECT_ROOT / "src" / "kernels" / f"{hip_src_name}.hip"
    hsaco_path = BUILD_DIR / f"{hsaco_name}.hsaco"

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
        raise RuntimeError(f"hipcc build failed for {hip_src_name}: {result.stdout}\n{result.stderr}")
    return hsaco_path


def pack_int4_weights(W_q4):
    """Pack INT4 weight matrix [K, N] into [K/8, N] uint32."""
    K, N = W_q4.shape
    assert K % 8 == 0
    B_packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(K // 8):
        for bit in range(8):
            B_packed[i] |= (W_q4[i * 8 + bit].astype(np.uint32) & 0xF) << (bit * 4)
    return B_packed


def compute_reference(A, W_q4, scales, zeros, K, N, group_size):
    """Vectorized reference: dequantize weights then multiply."""
    A32 = A.astype(np.float32)
    num_groups = K // group_size
    ref = np.zeros(N, dtype=np.float32)
    W_q4_f32 = W_q4.astype(np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        s = scales[g].astype(np.float32)
        z = zeros[g].astype(np.float32)
        W_block = (W_q4_f32[k_start:k_end] - z[None, :]) * s[None, :]
        ref += A32[k_start:k_end] @ W_block
    return ref


def test_v2_optimized(dev, K, N, group_size=128, k_splits=16, description=""):
    """Test optimized v2 fused kernel (ubfe + power-of-2 shift)."""
    print(f"\n{'='*60}")
    print(f"Test v2_fused (optimized): {description}")
    print(f"K={K}, N={N}, group_size={group_size}, k_splits={k_splits}")

    np.random.seed(42)
    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_packed = pack_int4_weights(W_q4)
    ref = compute_reference(A, W_q4, scales, zeros, K, N, group_size)

    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_fused = dev.get_kernel(module, "gemv_int4_v2_fused")

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_fp32 = dev.malloc(N * 4)
    dev.memset(d_fp32, 0, N * 4)
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)
    d_C = dev.malloc(N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())

    grid_x = (N + 255) // 256
    params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32), ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits), ctypes.c_uint64(0),
    ]
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()

    out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
    abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())

    dev.free(d_A); dev.free(d_B); dev.free(d_scales); dev.free(d_zeros)
    dev.free(d_fp32); dev.free(d_done); dev.free(d_C)

    passed = max_err < TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f} [{status}]")
    return passed, max_err


def test_v3_variants(dev, K, N, group_size=128, description=""):
    """Test v3 cooperative variants and new DPP variant."""
    print(f"\n{'='*60}")
    print(f"Test v3 variants: {description}")
    print(f"K={K}, N={N}, group_size={group_size}")

    np.random.seed(42)
    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_packed = pack_int4_weights(W_q4)
    ref = compute_reference(A, W_q4, scales, zeros, K, N, group_size)

    hsaco_path = BUILD_DIR / "gemv_int4_v3.hsaco"
    module = dev.load_hsaco(str(hsaco_path))

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_C = dev.malloc(N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())

    base_params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
    ]

    results = {}
    variants = [
        ("gemv_int4_v3_t4",  4,  64),
        ("gemv_int4_v3_t8",  8,  32),
        ("gemv_int4_v3_t16", 16, 16),
        ("gemv_int4_v3_dpp", 64,  4),
    ]

    for kname, tpc, cols_per_wg in variants:
        try:
            func = dev.get_kernel(module, kname)
            grid_x = (N + cols_per_wg - 1) // cols_per_wg
            dev.launch(func, (grid_x, 1, 1), (256, 1, 1), base_params)
            dev.synchronize()
            out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
            abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
            max_err = float(abs_err.max())
            passed = max_err < TOLERANCE
            status = "PASS" if passed else "FAIL"
            print(f"  {kname}: max_err={max_err:.6f} [{status}], grid=({grid_x},1,1)")
            results[kname] = (passed, max_err)
        except Exception as e:
            print(f"  {kname}: ERROR - {e}")
            results[kname] = (False, 99.0)

    dev.free(d_A); dev.free(d_B); dev.free(d_scales); dev.free(d_zeros); dev.free(d_C)
    return results


def benchmark_kernels(dev, K, N, group_size=128, description=""):
    """Benchmark all INT4 GEMV variants and report latency."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {description}")
    print(f"K={K}, N={N}, group_size={group_size}")

    np.random.seed(99)
    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_packed = pack_int4_weights(W_q4)

    hsaco_v2 = BUILD_DIR / "gemv_int4_v2.hsaco"
    hsaco_v3 = BUILD_DIR / "gemv_int4_v3.hsaco"
    mod_v2 = dev.load_hsaco(str(hsaco_v2))
    mod_v3 = dev.load_hsaco(str(hsaco_v3))

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_fp32 = dev.malloc(N * 4)
    dev.memset(d_fp32, 0, N * 4)
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)
    d_C_v2 = dev.malloc(N * 2)
    d_C_v3 = dev.malloc(N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())

    k_splits = 16
    grid_x_v2 = (N + 255) // 256

    func_fused = dev.get_kernel(mod_v2, "gemv_int4_v2_fused")
    v2_params = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32), ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_C_v2),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits), ctypes.c_uint64(0),
    ]

    base_params_v3 = [
        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_C_v3),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
    ]

    N_ITERS = 100
    timings = {}

    # Warmup
    for _ in range(10):
        dev.launch(func_fused, (grid_x_v2, k_splits, 1), (256, 1, 1), v2_params)
    dev.synchronize()

    # Benchmark v2_fused (optimized)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.launch(func_fused, (grid_x_v2, k_splits, 1), (256, 1, 1), v2_params)
    dev.synchronize()
    timings["v2_fused"] = (time.perf_counter() - t0) / N_ITERS * 1e6

    # Benchmark v3 variants
    variants = [
        ("gemv_int4_v3_t4",  4,  64, mod_v3),
        ("gemv_int4_v3_t8",  8,  32, mod_v3),
        ("gemv_int4_v3_t16", 16, 16, mod_v3),
        ("gemv_int4_v3_dpp", 64,  4, mod_v3),
    ]
    for kname, tpc, cols, mod in variants:
        try:
            func = dev.get_kernel(mod, kname)
            grid_x = (N + cols - 1) // cols
            # Warmup
            for _ in range(10):
                dev.launch(func, (grid_x, 1, 1), (256, 1, 1), base_params_v3)
            dev.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITERS):
                dev.launch(func, (grid_x, 1, 1), (256, 1, 1), base_params_v3)
            dev.synchronize()
            timings[kname] = (time.perf_counter() - t0) / N_ITERS * 1e6
        except Exception as e:
            print(f"  {kname} benchmark error: {e}")

    # Theoretical
    weight_bytes = K * N // 2
    total_bytes = weight_bytes + K * 2 + N * 2
    theoretical_us = total_bytes / 857e9 * 1e6

    print(f"  Theoretical (bandwidth-limited @857GB/s): {theoretical_us:.1f} us")
    for name, us in sorted(timings.items(), key=lambda x: x[1]):
        pct = theoretical_us / us * 100
        print(f"  {name:30s}: {us:6.1f} us  ({pct:.0f}% of BW limit)")

    best_name = min(timings, key=timings.get)
    best_us = timings[best_name]
    print(f"\n  WINNER: {best_name} @ {best_us:.1f} us")
    print(f"  v2_fused vs best: {timings['v2_fused']:.1f} vs {best_us:.1f} us"
          f" ({timings['v2_fused']/best_us:.2f}x)")

    dev.free(d_A); dev.free(d_B); dev.free(d_scales); dev.free(d_zeros)
    dev.free(d_fp32); dev.free(d_done); dev.free(d_C_v2); dev.free(d_C_v3)

    return timings, best_name


def test_engine_uses_correct_kernel():
    """Verify which INT4 GEMV kernel is wired into the engine.
    
    After m2-int4-gemv-optimize:
    - v3_t16 is used for non-residual GEMV (faster path)
    - v2_fused is used for GEMV with residual (down_proj) 
    - No separate memset or fp32_to_fp16 calls
    """
    print(f"\n{'='*60}")
    print("Test: Engine kernel wiring")

    import inspect
    from src.inference.engine import InferenceEngine

    src = inspect.getsource(InferenceEngine._launch_gemv_int4)
    has_fused = "gemv_int4_v2_fused" in src
    has_v3 = "gemv_int4_v3_t16" in src or "_gemv_int4_v3" in src
    import re
    non_comment_lines = [l for l in src.split('\n') if not l.strip().startswith('#')]
    non_comment_src = '\n'.join(non_comment_lines)
    has_memset = bool(re.search(r'\.memset\s*\(', non_comment_src))
    has_convert = "fp32_to_fp16" in non_comment_src

    print(f"  Uses gemv_int4_v2_fused (for residual): {has_fused}")
    print(f"  Uses gemv_int4_v3 (for non-residual):   {has_v3}")
    print(f"  Has device.memset call:                  {has_memset}")
    print(f"  Has fp32_to_fp16 call:                   {has_convert}")

    # Check init loads v3
    init_src = inspect.getsource(InferenceEngine._init_gemv_v2)
    has_v3_init = "gemv_int4_v3" in init_src and "_gemv_int4_v3" in init_src

    print(f"  v3 initialized in _init_gemv_v2:         {has_v3_init}")

    ok = has_fused and has_v3 and not has_memset and not has_convert and has_v3_init
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Building optimized INT4 GEMV kernels...")
    build_kernel("gemv_int4_v2")
    print("  gemv_int4_v2.hsaco built OK")
    build_kernel("gemv_int4_v3")
    print("  gemv_int4_v3.hsaco built OK")
    print()

    dev = GPUDevice(0)
    all_pass = True
    benchmark_results = {}

    try:
        # --- Correctness tests ---
        dims_to_test = [
            (4096, 4096,  128, 16, "N=4096 K=4096 (Qwen hidden)"),
            (4096, 11008, 128, 16, "N=11008 K=4096 (Qwen FFN gate/up)"),
            (5120, 5120,  128, 16, "N=5120 K=5120 (Qwen 27B hidden)"),
            (4096, 4096,  128,  8, "N=4096 K=4096 k_splits=8"),
        ]

        print("\n=== V2 Fused (Optimized with ubfe + pow2 shift) ===")
        for K, N, gs, ks, desc in dims_to_test:
            passed, max_err = test_v2_optimized(dev, K, N, gs, ks, desc)
            if not passed:
                all_pass = False

        print("\n=== V3 Variants (ubfe + DPP reduction) ===")
        v3_dims = [
            (4096,  4096,  128, "N=4096 K=4096"),
            (4096,  11008, 128, "N=11008 K=4096"),
        ]
        for K, N, gs, desc in v3_dims:
            results = test_v3_variants(dev, K, N, gs, desc)
            for kname, (passed, err) in results.items():
                if not passed:
                    all_pass = False

        # --- Performance benchmarks ---
        print("\n=== Performance Benchmarks ===")
        bench_dims = [
            (4096, 4096,  128, "N=4096 K=4096"),
            (4096, 11008, 128, "N=11008 K=4096"),
        ]
        for K, N, gs, desc in bench_dims:
            timings, best = benchmark_kernels(dev, K, N, gs, desc)
            benchmark_results[desc] = (timings, best)

        # --- Engine check ---
        if not test_engine_uses_correct_kernel():
            all_pass = False

    finally:
        dev.cleanup()

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY:")
    for desc, (timings, best) in benchmark_results.items():
        print(f"  {desc}: best={best} @ {timings.get(best, 0):.1f}us,"
              f" v2_fused={timings.get('v2_fused', 0):.1f}us")

    if all_pass:
        print("\nAll INT4 GEMV optimization tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
