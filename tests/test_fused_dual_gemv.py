#!/usr/bin/env python3
"""
Test for fused dual gate+up INT4 GEMV kernel (m1-fused-dual-gemv).

Verifies that the fused kernel (gemv_int4_dual_fused):
1. Produces correct FP16 output with SiLU in a single launch (no memset/convert)
2. Has max absolute error < 1e-2 vs reference (per INT4 quantized tolerance)
3. Works for key FFN dimensions: N=11008 and N=14336, K=4096
4. Correctly resets persistent buffers after each call (multi-call correctness)
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


def build_dual_kernel():
    """Build the dual INT4 GEMV kernel as HSACO and return the path."""
    import subprocess

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_src = PROJECT_ROOT / "src" / "kernels" / "gemv_int4_dual.hip"
    hsaco_path = BUILD_DIR / "gemv_int4_dual.hsaco"

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
    """Pack INT4 weight matrix [K, N] into [K/8, N] uint32."""
    K, N = W_q4.shape
    assert K % 8 == 0
    B_packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(K // 8):
        for bit in range(8):
            B_packed[i] |= (W_q4[i * 8 + bit].astype(np.uint32) & 0xF) << (bit * 4)
    return B_packed


def compute_reference_fast(A, W_q4, scales, zeros, K, N, group_size):
    """Vectorized reference computation for a single GEMV."""
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


def silu(x):
    """Reference SiLU (sigmoid-weighted linear unit)."""
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def compute_dual_reference(A, W_gate_q4, gate_scales, gate_zeros,
                            W_up_q4, up_scales, up_zeros, K, N, group_size):
    """Compute reference for fused dual GEMV + SiLU: silu(gate(A)) * up(A)."""
    gate = compute_reference_fast(A, W_gate_q4, gate_scales, gate_zeros, K, N, group_size)
    up   = compute_reference_fast(A, W_up_q4,   up_scales,   up_zeros,   K, N, group_size)
    return silu(gate) * up


def run_fused_dual_test(K, N, group_size, k_splits, dev, description, seed=42):
    """
    Run the fused dual INT4 GEMV test.

    The fused kernel (gemv_int4_dual_fused) uses persistent FP32 accumulator
    buffers and a done-counter that are reset by the kernel itself. No separate
    memset or fp32_to_silu_fp16 launch is needed.
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"K={K}, N={N}, group_size={group_size}, k_splits={k_splits}")
    print(f"{'='*60}")

    assert K % 8 == 0, "K must be divisible by 8"
    assert K % group_size == 0, "K must be divisible by group_size"

    np.random.seed(seed)

    A = (np.random.randn(K) * 0.1).astype(np.float16)

    W_gate_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    W_up_q4   = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)

    num_groups = K // group_size
    gate_scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    gate_zeros  = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    up_scales   = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    up_zeros    = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)

    B_gate_packed = pack_int4_weights(W_gate_q4)
    B_up_packed   = pack_int4_weights(W_up_q4)

    # Compute reference: silu(gate(A)) * up(A)
    ref_fp32 = compute_dual_reference(
        A, W_gate_q4, gate_scales, gate_zeros,
        W_up_q4, up_scales, up_zeros, K, N, group_size
    )
    ref_fp16 = ref_fp32.astype(np.float16)

    # Load fused kernel
    hsaco_path = BUILD_DIR / "gemv_int4_dual.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_fused = dev.get_kernel(module, "gemv_int4_dual_fused")

    # Allocate GPU buffers
    d_A          = dev.malloc(A.nbytes)
    d_B_gate     = dev.malloc(B_gate_packed.nbytes)
    d_gate_scales = dev.malloc(gate_scales.nbytes)
    d_gate_zeros  = dev.malloc(gate_zeros.nbytes)
    d_B_up       = dev.malloc(B_up_packed.nbytes)
    d_up_scales   = dev.malloc(up_scales.nbytes)
    d_up_zeros    = dev.malloc(up_zeros.nbytes)

    # Persistent FP32 accumulators — zeroed ONCE, reset by kernel
    d_gate_fp32 = dev.malloc(N * 4)
    dev.memset(d_gate_fp32, 0, N * 4)
    d_up_fp32   = dev.malloc(N * 4)
    dev.memset(d_up_fp32, 0, N * 4)
    # Persistent done-counter — zeroed ONCE, reset by kernel
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)

    d_out = dev.malloc(N * 2)  # FP16 output

    # Upload inputs
    dev.upload(d_A,          A.tobytes())
    dev.upload(d_B_gate,     B_gate_packed.tobytes())
    dev.upload(d_gate_scales, gate_scales.tobytes())
    dev.upload(d_gate_zeros,  gate_zeros.tobytes())
    dev.upload(d_B_up,       B_up_packed.tobytes())
    dev.upload(d_up_scales,   up_scales.tobytes())
    dev.upload(d_up_zeros,    up_zeros.tobytes())

    grid_x = (N + 255) // 256

    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B_gate),
        ctypes.c_uint64(d_gate_scales),
        ctypes.c_uint64(d_gate_zeros),
        ctypes.c_uint64(d_B_up),
        ctypes.c_uint64(d_up_scales),
        ctypes.c_uint64(d_up_zeros),
        ctypes.c_uint64(d_gate_fp32),
        ctypes.c_uint64(d_up_fp32),
        ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits),
    ]

    # Launch fused kernel — single launch, no pre-memset, no post-convert
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()

    out = np.frombuffer(dev.download(d_out, N * 2), dtype=np.float16)

    abs_err = np.abs(out.astype(np.float32) - ref_fp16.astype(np.float32))
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())

    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")
    print(f"ref[:4]  = {ref_fp16[:4]}")
    print(f"out[:4]  = {out[:4]}")

    # Verify that persistent buffers were reset by the kernel
    gate_fp32_after = np.frombuffer(dev.download(d_gate_fp32, N * 4), dtype=np.float32)
    up_fp32_after   = np.frombuffer(dev.download(d_up_fp32,   N * 4), dtype=np.float32)
    done_after      = np.frombuffer(dev.download(d_done,      N * 4), dtype=np.uint32)
    buffers_clean = (not np.any(gate_fp32_after != 0.0) and
                     not np.any(up_fp32_after   != 0.0) and
                     not np.any(done_after       != 0))
    if not buffers_clean:
        print(f"  WARNING: Persistent buffers not reset after kernel call!")
        print(f"  gate_fp32 non-zero: {np.sum(gate_fp32_after != 0)}, "
              f"up_fp32 non-zero: {np.sum(up_fp32_after != 0)}, "
              f"done non-zero: {np.sum(done_after != 0)}")

    # Second call with different input (verifies buffer reset correctness)
    np.random.seed(seed + 100)
    A2 = (np.random.randn(K) * 0.1).astype(np.float16)
    dev.upload(d_A, A2.tobytes())
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()
    out2 = np.frombuffer(dev.download(d_out, N * 2), dtype=np.float16)
    ref2_fp32 = compute_dual_reference(
        A2, W_gate_q4, gate_scales, gate_zeros,
        W_up_q4, up_scales, up_zeros, K, N, group_size
    ).astype(np.float16)
    max_err2 = float(np.abs(out2.astype(np.float32) - ref2_fp32.astype(np.float32)).max())
    print(f"Second call max error: {max_err2:.6f}")

    # Cleanup
    dev.free(d_A)
    dev.free(d_B_gate)
    dev.free(d_gate_scales)
    dev.free(d_gate_zeros)
    dev.free(d_B_up)
    dev.free(d_up_scales)
    dev.free(d_up_zeros)
    dev.free(d_gate_fp32)
    dev.free(d_up_fp32)
    dev.free(d_done)
    dev.free(d_out)

    ok = max_err < TOLERANCE and max_err2 < TOLERANCE and buffers_clean
    if ok:
        print(f"PASS (max_err={max_err:.6f}, max_err2={max_err2:.6f}, buffers_clean={buffers_clean})")
    else:
        print(f"FAIL (max_err={max_err:.6f}, max_err2={max_err2:.6f}, buffers_clean={buffers_clean})")
    return ok, max(max_err, max_err2)


def run_perf_comparison(K, N, group_size, k_splits, dev, description):
    """Compare latency: old 4-launch vs new 1-launch fused path."""
    print(f"\n{'='*60}")
    print(f"Performance: {description}")
    print(f"{'='*60}")

    np.random.seed(77)
    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_gate_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    W_up_q4   = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    gate_scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    gate_zeros  = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    up_scales   = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    up_zeros    = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_gate_packed = pack_int4_weights(W_gate_q4)
    B_up_packed   = pack_int4_weights(W_up_q4)

    hsaco_path = BUILD_DIR / "gemv_int4_dual.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_dual_splitk = dev.get_kernel(module, "gemv_int4_dual_splitk")
    func_silu        = dev.get_kernel(module, "dual_fp32_to_silu_fp16")
    func_fused       = dev.get_kernel(module, "gemv_int4_dual_fused")

    d_A          = dev.malloc(A.nbytes)
    d_B_gate     = dev.malloc(B_gate_packed.nbytes)
    d_gate_scales = dev.malloc(gate_scales.nbytes)
    d_gate_zeros  = dev.malloc(gate_zeros.nbytes)
    d_B_up       = dev.malloc(B_up_packed.nbytes)
    d_up_scales   = dev.malloc(up_scales.nbytes)
    d_up_zeros    = dev.malloc(up_zeros.nbytes)
    d_gate_fp32_old = dev.malloc(N * 4)
    d_up_fp32_old   = dev.malloc(N * 4)
    d_gate_fp32_new = dev.malloc(N * 4)
    d_up_fp32_new   = dev.malloc(N * 4)
    d_done_new   = dev.malloc(N * 4)
    dev.memset(d_gate_fp32_new, 0, N * 4)
    dev.memset(d_up_fp32_new,   0, N * 4)
    dev.memset(d_done_new,      0, N * 4)
    d_out_old = dev.malloc(N * 2)
    d_out_new = dev.malloc(N * 2)

    dev.upload(d_A,          A.tobytes())
    dev.upload(d_B_gate,     B_gate_packed.tobytes())
    dev.upload(d_gate_scales, gate_scales.tobytes())
    dev.upload(d_gate_zeros,  gate_zeros.tobytes())
    dev.upload(d_B_up,       B_up_packed.tobytes())
    dev.upload(d_up_scales,   up_scales.tobytes())
    dev.upload(d_up_zeros,    up_zeros.tobytes())

    grid_x = (N + 255) // 256

    old_params_splitk = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B_gate), ctypes.c_uint64(d_gate_scales), ctypes.c_uint64(d_gate_zeros),
        ctypes.c_uint64(d_B_up),   ctypes.c_uint64(d_up_scales),   ctypes.c_uint64(d_up_zeros),
        ctypes.c_uint64(d_gate_fp32_old), ctypes.c_uint64(d_up_fp32_old),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
    ]
    old_params_silu = [
        ctypes.c_uint64(d_gate_fp32_old), ctypes.c_uint64(d_up_fp32_old),
        ctypes.c_uint64(d_out_old), ctypes.c_uint32(N),
    ]
    new_params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B_gate), ctypes.c_uint64(d_gate_scales), ctypes.c_uint64(d_gate_zeros),
        ctypes.c_uint64(d_B_up),   ctypes.c_uint64(d_up_scales),   ctypes.c_uint64(d_up_zeros),
        ctypes.c_uint64(d_gate_fp32_new), ctypes.c_uint64(d_up_fp32_new),
        ctypes.c_uint64(d_done_new), ctypes.c_uint64(d_out_new),
        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits),
    ]

    # Warmup
    for _ in range(5):
        dev.memset(d_gate_fp32_old, 0, N * 4)
        dev.memset(d_up_fp32_old,   0, N * 4)
        dev.launch(func_dual_splitk, (grid_x, k_splits, 1), (256, 1, 1), old_params_splitk)
        dev.launch(func_silu, (grid_x, 1, 1), (256, 1, 1), old_params_silu)
        dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), new_params)
    dev.synchronize()

    N_ITERS = 100

    # Benchmark old path (4 launches: 2 memsets + splitk + convert+silu)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.memset(d_gate_fp32_old, 0, N * 4)
        dev.memset(d_up_fp32_old,   0, N * 4)
        dev.launch(func_dual_splitk, (grid_x, k_splits, 1), (256, 1, 1), old_params_splitk)
        dev.launch(func_silu, (grid_x, 1, 1), (256, 1, 1), old_params_silu)
    dev.synchronize()
    t1 = time.perf_counter()
    old_us = (t1 - t0) / N_ITERS * 1e6
    print(f"Old path (2×memset + dual_splitk + silu): {old_us:.1f} us/iter")

    # Benchmark new path (1 launch: fused)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), new_params)
    dev.synchronize()
    t1 = time.perf_counter()
    new_us = (t1 - t0) / N_ITERS * 1e6
    print(f"New path (fused dual):                    {new_us:.1f} us/iter")
    speedup = old_us / new_us
    print(f"Speedup: {speedup:.2f}x")

    dev.free(d_A)
    dev.free(d_B_gate)
    dev.free(d_gate_scales)
    dev.free(d_gate_zeros)
    dev.free(d_B_up)
    dev.free(d_up_scales)
    dev.free(d_up_zeros)
    dev.free(d_gate_fp32_old)
    dev.free(d_up_fp32_old)
    dev.free(d_gate_fp32_new)
    dev.free(d_up_fp32_new)
    dev.free(d_done_new)
    dev.free(d_out_old)
    dev.free(d_out_new)

    return old_us, new_us


def test_engine_no_memset():
    """Verify that _launch_ffn_gate_up_silu does NOT call device.memset in the fused path.

    Inspects engine.py source to confirm no per-call memset remains.
    """
    print(f"\n{'='*60}")
    print("Test: Engine uses fused dual path (no per-call memset)")
    print(f"{'='*60}")

    import re
    import inspect
    from src.inference.engine import InferenceEngine

    src = inspect.getsource(InferenceEngine._launch_ffn_gate_up_silu)

    # Check for the fused kernel reference
    has_fused = "gemv_int4_dual_fused" in src

    # Check that the PRIMARY path (if self._gemv_int4_dual_fused) does NOT call device.memset
    # Extract only the "if self._gemv_int4_dual_fused:" block
    lines = src.split('\n')
    in_fused_block = False
    fused_block_lines = []
    indent_level = None
    for line in lines:
        stripped = line.lstrip()
        if 'if self._gemv_int4_dual_fused:' in line:
            in_fused_block = True
            indent_level = len(line) - len(stripped)
            continue
        if in_fused_block:
            curr_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
            if line.strip() and curr_indent <= indent_level:
                in_fused_block = False
            else:
                fused_block_lines.append(line)

    fused_block = '\n'.join(fused_block_lines)
    # Remove comment lines
    non_comment_fused = '\n'.join(
        l for l in fused_block.split('\n') if not l.strip().startswith('#')
    )
    has_memset_in_fused = bool(re.search(r'\.memset\s*\(', non_comment_fused))

    print(f"  Uses gemv_int4_dual_fused kernel:       {has_fused}")
    print(f"  Fused path contains device.memset():    {has_memset_in_fused}")

    passed = has_fused and not has_memset_in_fused
    if passed:
        print("PASS")
    else:
        print("FAIL: Engine fused path still uses memset or missing fused kernel")
    return passed


def main():
    print("Building dual fused INT4 GEMV kernel...")
    build_dual_kernel()
    print("Build successful.\n")

    dev = GPUDevice(0)
    all_pass = True
    errors = {}

    try:
        # Test 1: N=11008, K=4096 (Qwen 2.5 27B FFN gate/up dimensions)
        passed, max_err = run_fused_dual_test(
            K=4096, N=11008, group_size=128, k_splits=16, dev=dev,
            description="Fused Dual GEMV: N=11008, K=4096 (FFN gate/up dims)"
        )
        errors["N=11008,K=4096"] = max_err
        if not passed:
            all_pass = False

        # Test 2: N=4096, K=4096 (hidden size)
        passed, max_err = run_fused_dual_test(
            K=4096, N=4096, group_size=128, k_splits=16, dev=dev,
            description="Fused Dual GEMV: N=4096, K=4096"
        )
        errors["N=4096,K=4096"] = max_err
        if not passed:
            all_pass = False

        # Test 3: N=14336, K=4096 (Qwen 3.5 27B FFN gate/up dimensions)
        passed, max_err = run_fused_dual_test(
            K=4096, N=14336, group_size=128, k_splits=16, dev=dev,
            description="Fused Dual GEMV: N=14336, K=4096 (Qwen 3.5 27B FFN)"
        )
        errors["N=14336,K=4096"] = max_err
        if not passed:
            all_pass = False

        # Test 4: k_splits=8 (different split count)
        passed, max_err = run_fused_dual_test(
            K=4096, N=11008, group_size=128, k_splits=8, dev=dev,
            description="Fused Dual GEMV: N=11008, K=4096, k_splits=8"
        )
        errors["N=11008,K=4096,k_splits=8"] = max_err
        if not passed:
            all_pass = False

        # Test 5: Engine structural check (no per-call memset in fused path)
        if not test_engine_no_memset():
            all_pass = False

        # Performance benchmarks
        print("\n" + "="*60)
        print("Performance Benchmarks")
        run_perf_comparison(
            K=4096, N=11008, group_size=128, k_splits=16, dev=dev,
            description="N=11008, K=4096"
        )
        run_perf_comparison(
            K=4096, N=14336, group_size=128, k_splits=16, dev=dev,
            description="N=14336, K=4096"
        )

    finally:
        dev.cleanup()

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, err in errors.items():
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"  {name}: max_err={err:.6f} [{status}]")

    if all_pass:
        print("\nAll fused dual GEMV tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
