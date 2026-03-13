#!/usr/bin/env python3
"""
Test for residual-add epilogues in projection GEMV kernels (m1-residual-epilogues).

Verifies that:
1. gemv_fp16_v2 with residual epilogue: out[i] = gemv(x, W) + residual[i]
   Max abs error < 1e-3 vs separate GEMV + residual_add (VAL-DLR-006)
2. gemv_int4_v2_fused with residual epilogue: C_fp16[col] = fp16(total) + residual[col]
   Max abs error < 1e-2 vs separate INT4 GEMV + residual_add (VAL-DLR-007)
3. Engine removes 2 residual_add launches per layer (1 for out_proj, 1 for down_proj)
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

FP16_TOLERANCE = 1e-3
INT4_TOLERANCE = 1e-2


def build_kernel(kernel_name):
    """Build a HIP kernel as HSACO."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_src = PROJECT_ROOT / "src" / "kernels" / f"{kernel_name}.hip"
    hsaco_path = BUILD_DIR / f"{kernel_name}.hsaco"

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
        raise RuntimeError(f"hipcc build failed for {kernel_name}:\n{result.stdout}\n{result.stderr}")
    print(f"  Built {kernel_name}.hsaco")
    return hsaco_path


def test_fp16_gemv_residual(K, N, dev, description, seed=42):
    """
    Test FP16 GEMV with residual epilogue.

    Verifies: out[i] = W[i,:] @ x + residual[i]
    vs reference: out = W @ x + residual (separately)
    """
    print(f"\n{'='*60}")
    print(f"Test: FP16 GEMV + Residual — {description}")
    print(f"  K={K}, N={N}")
    print(f"{'='*60}")

    np.random.seed(seed)

    x = (np.random.randn(K) * 0.1).astype(np.float16)
    W = (np.random.randn(N, K) * 0.02).astype(np.float16)
    residual = (np.random.randn(N) * 0.3).astype(np.float16)

    # Reference: gemv + residual_add
    x32 = x.astype(np.float32)
    W32 = W.astype(np.float32)
    ref32 = W32 @ x32 + residual.astype(np.float32)
    ref = ref32.astype(np.float16)

    # Load gemv_fp16_v2 kernel
    hsaco_path = BUILD_DIR / "gemv_fp16_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func = dev.get_kernel(module, "gemv_fp16_v2")

    # Allocate GPU buffers
    d_x = dev.malloc(x.nbytes)
    d_W = dev.malloc(W.nbytes)
    d_out = dev.malloc(N * 2)
    d_residual = dev.malloc(residual.nbytes)

    dev.upload(d_x, x.tobytes())
    dev.upload(d_W, W.tobytes())
    dev.upload(d_residual, residual.tobytes())

    # Launch with residual (non-null residual pointer)
    grid_x = (N + 3) // 4  # 4 rows per WG
    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_W),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint64(d_residual),  # residual ptr (non-null)
    ]
    dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    out = np.frombuffer(dev.download(d_out, N * 2), dtype=np.float16)
    abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    max_err = float(abs_err.max())

    print(f"Max absolute error: {max_err:.6f}")
    print(f"ref[:4]  = {ref[:4]}")
    print(f"out[:4]  = {out[:4]}")

    # Also test without residual (null pointer = 0)
    params_no_res = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_W),
        ctypes.c_uint64(d_out),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint64(0),  # null residual ptr
    ]
    dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params_no_res)
    dev.synchronize()
    out_no_res = np.frombuffer(dev.download(d_out, N * 2), dtype=np.float16)
    ref_no_res = (W32 @ x32).astype(np.float16)
    err_no_res = float(np.abs(out_no_res.astype(np.float32) - ref_no_res.astype(np.float32)).max())
    print(f"Without residual max error: {err_no_res:.6f}")

    dev.free(d_x)
    dev.free(d_W)
    dev.free(d_out)
    dev.free(d_residual)

    passed = max_err < FP16_TOLERANCE and err_no_res < FP16_TOLERANCE
    if passed:
        print(f"PASS (with_residual={max_err:.6f}, no_residual={err_no_res:.6f})")
    else:
        print(f"FAIL (with_residual={max_err:.6f}, no_residual={err_no_res:.6f})")
    return passed, max_err


def pack_int4_weights(W_q4):
    """Pack INT4 weight matrix [K, N] into [K/8, N] uint32."""
    K, N = W_q4.shape
    assert K % 8 == 0
    B_packed = np.zeros((K // 8, N), dtype=np.uint32)
    for i in range(K // 8):
        for bit in range(8):
            B_packed[i] |= (W_q4[i * 8 + bit].astype(np.uint32) & 0xF) << (bit * 4)
    return B_packed


def compute_int4_reference(A, W_q4, scales, zeros, K, N, group_size):
    """Vectorized reference for INT4 GEMV."""
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


def test_int4_gemv_residual(K, N, group_size, k_splits, dev, description, seed=42):
    """
    Test INT4 GEMV fused kernel with residual epilogue.

    Verifies: C_fp16[col] = fp16(accumulated_fp32) + residual[col]
    vs reference: int4_gemv + residual_add separately
    """
    print(f"\n{'='*60}")
    print(f"Test: INT4 GEMV + Residual — {description}")
    print(f"  K={K}, N={N}, group_size={group_size}, k_splits={k_splits}")
    print(f"{'='*60}")

    np.random.seed(seed)

    A = (np.random.randn(K) * 0.1).astype(np.float16)
    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    residual = (np.random.randn(N) * 0.5).astype(np.float16)

    B_packed = pack_int4_weights(W_q4)

    # Reference: int4_gemv + residual_add
    ref32 = compute_int4_reference(A, W_q4, scales, zeros, K, N, group_size)
    ref = (ref32 + residual.astype(np.float32)).astype(np.float16)

    # Load fused kernel
    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_fused = dev.get_kernel(module, "gemv_int4_v2_fused")

    # Allocate GPU buffers
    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_fp32 = dev.malloc(N * 4)
    dev.memset(d_fp32, 0, N * 4)
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)
    d_C = dev.malloc(N * 2)
    d_residual = dev.malloc(residual.nbytes)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())
    dev.upload(d_residual, residual.tobytes())

    # Launch with residual (non-null residual pointer)
    grid_x = (N + 255) // 256
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32),
        ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits),
        ctypes.c_uint64(d_residual),  # residual ptr (non-null)
    ]
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
    dev.synchronize()

    out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
    abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    max_err = float(abs_err.max())

    print(f"Max absolute error: {max_err:.6f}")
    print(f"ref[:4]  = {ref[:4]}")
    print(f"out[:4]  = {out[:4]}")

    # Also test without residual (null pointer = 0)
    # Need to reset persistent buffers since kernel resets them after use
    # (they should already be reset, but re-check)
    params_no_res = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_scales),
        ctypes.c_uint64(d_zeros),
        ctypes.c_uint64(d_fp32),
        ctypes.c_uint64(d_done),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N),
        ctypes.c_uint32(group_size),
        ctypes.c_uint32(k_splits),
        ctypes.c_uint64(0),  # null residual ptr
    ]
    dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params_no_res)
    dev.synchronize()
    out_no_res = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
    ref_no_res = ref32.astype(np.float16)
    err_no_res = float(np.abs(out_no_res.astype(np.float32) - ref_no_res.astype(np.float32)).max())
    print(f"Without residual max error: {err_no_res:.6f}")

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_fp32)
    dev.free(d_done)
    dev.free(d_C)
    dev.free(d_residual)

    passed = max_err < INT4_TOLERANCE and err_no_res < INT4_TOLERANCE
    if passed:
        print(f"PASS (with_residual={max_err:.6f}, no_residual={err_no_res:.6f})")
    else:
        print(f"FAIL (with_residual={max_err:.6f}, no_residual={err_no_res:.6f})")
    return passed, max_err


def test_engine_residual_epilogues():
    """
    Verify engine.py wiring for residual epilogues.

    Checks:
    - out_proj GEMV passes residual ptr to gemv_fp16_v2
    - down_proj INT4 GEMV passes residual ptr to gemv_int4_v2_fused
    - No separate _launch_residual_add after down_proj in decode_step
    """
    print(f"\n{'='*60}")
    print("Test: Engine wiring for residual epilogues (VAL-DLR-006, VAL-DLR-007)")
    print(f"{'='*60}")

    import inspect
    import re
    from src.inference.engine import InferenceEngine

    # Check _launch_gemv_fp16: should accept and pass residual parameter
    src_fp16 = inspect.getsource(InferenceEngine._launch_gemv_fp16)
    has_residual_param_fp16 = "residual" in src_fp16
    print(f"  _launch_gemv_fp16 has residual param: {has_residual_param_fp16}")

    # Check _launch_gemv_int4: should accept and pass residual parameter
    src_int4 = inspect.getsource(InferenceEngine._launch_gemv_int4)
    has_residual_param_int4 = "residual" in src_int4
    print(f"  _launch_gemv_int4 has residual param: {has_residual_param_int4}")

    # Check decode_step: no separate _launch_residual_add for down_proj
    # (look at the code around down_proj launch)
    src_decode = inspect.getsource(InferenceEngine.decode_step)
    # Count _launch_residual_add calls in decode_step
    residual_add_calls = re.findall(r'_launch_residual_add', src_decode)
    print(f"  decode_step _launch_residual_add calls: {len(residual_add_calls)}")
    print(f"  (expected: 0 for single-GPU tp_size=1 path after epilogue fusion)")

    # Check _decode_full_attention: out_proj should pass residual
    src_full_attn = inspect.getsource(InferenceEngine._decode_full_attention)
    # The out_proj gemv should have residual=d_hidden
    has_out_proj_residual = "d_hidden" in src_full_attn and "residual" in src_full_attn
    print(f"  _decode_full_attention out_proj passes residual: {has_out_proj_residual}")

    # Check that skip_rmsnorm is replaced by plain rmsnorm for the pre-FFN position
    # (since out_proj now adds residual directly)
    has_skip_rmsnorm_in_decode = "_launch_skip_rmsnorm" in src_decode
    has_rmsnorm_for_ffn = "_launch_rmsnorm" in src_decode
    print(f"  decode_step still uses skip_rmsnorm: {has_skip_rmsnorm_in_decode}")
    print(f"  decode_step uses rmsnorm: {has_rmsnorm_for_ffn}")

    passed = (has_residual_param_fp16 and
              has_residual_param_int4 and
              len(residual_add_calls) == 0)

    if passed:
        print("PASS")
    else:
        print("FAIL: Engine not properly wired for residual epilogues")
    return passed


def test_multiple_calls_int4_residual(K, N, group_size, k_splits, dev, description):
    """Verify that INT4 GEMV with residual works correctly on multiple sequential calls."""
    print(f"\n{'='*60}")
    print(f"Multi-call test: INT4 + Residual — {description}")
    print(f"{'='*60}")

    np.random.seed(99)
    hsaco_path = BUILD_DIR / "gemv_int4_v2.hsaco"
    module = dev.load_hsaco(str(hsaco_path))
    func_fused = dev.get_kernel(module, "gemv_int4_v2_fused")

    W_q4 = np.random.randint(0, 16, size=(K, N)).astype(np.uint8)
    num_groups = K // group_size
    scales = (np.random.rand(num_groups, N) * 0.01 + 0.01).astype(np.float16)
    zeros = (np.random.rand(num_groups, N) * 2.0 + 7.0).astype(np.float16)
    B_packed = pack_int4_weights(W_q4)

    d_B = dev.malloc(B_packed.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_fp32 = dev.malloc(N * 4)
    dev.memset(d_fp32, 0, N * 4)
    d_done = dev.malloc(N * 4)
    dev.memset(d_done, 0, N * 4)
    d_C = dev.malloc(N * 2)

    dev.upload(d_B, B_packed.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())

    grid_x = (N + 255) // 256
    all_pass = True

    for call_idx in range(3):
        A = (np.random.randn(K) * 0.1).astype(np.float16)
        residual = (np.random.randn(N) * 0.5).astype(np.float16)

        d_A = dev.malloc(A.nbytes)
        d_residual = dev.malloc(residual.nbytes)
        dev.upload(d_A, A.tobytes())
        dev.upload(d_residual, residual.tobytes())

        ref32 = compute_int4_reference(A, W_q4, scales, zeros, K, N, group_size)
        ref = (ref32 + residual.astype(np.float32)).astype(np.float16)

        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_scales),
            ctypes.c_uint64(d_zeros),
            ctypes.c_uint64(d_fp32),
            ctypes.c_uint64(d_done),
            ctypes.c_uint64(d_C),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(group_size),
            ctypes.c_uint32(k_splits),
            ctypes.c_uint64(d_residual),
        ]
        dev.launch(func_fused, (grid_x, k_splits, 1), (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16)
        max_err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())
        ok = max_err < INT4_TOLERANCE
        print(f"  Call {call_idx+1}: max_err={max_err:.6f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

        dev.free(d_A)
        dev.free(d_residual)

    dev.free(d_B)
    dev.free(d_scales)
    dev.free(d_zeros)
    dev.free(d_fp32)
    dev.free(d_done)
    dev.free(d_C)

    return all_pass


def main():
    print("=" * 70)
    print("Residual-Add Epilogue Kernel Correctness Tests (m1-residual-epilogues)")
    print("=" * 70)

    print("\nBuilding kernels...")
    build_kernel("gemv_fp16_v2")
    build_kernel("gemv_int4_v2")
    print("Build successful.\n")

    dev = GPUDevice(0)
    all_pass = True
    errors = {}

    try:
        # FP16 GEMV + Residual tests
        print("\n--- FP16 GEMV + Residual (VAL-DLR-006) ---")

        # Test 1: out_proj size (q_dim → hidden, K=q_dim, N=hidden)
        # Typical: K=5120 (q_heads*head_dim, local), N=5120 (hidden)
        passed, max_err = test_fp16_gemv_residual(
            K=5120, N=5120, dev=dev,
            description="out_proj K=5120 N=5120"
        )
        errors["fp16_5120x5120"] = max_err
        if not passed:
            all_pass = False

        # Test 2: smaller size for quick validation
        passed, max_err = test_fp16_gemv_residual(
            K=512, N=512, dev=dev,
            description="K=512 N=512 (small)"
        )
        errors["fp16_512x512"] = max_err
        if not passed:
            all_pass = False

        # Test 3: la_out_proj size (la_z_dim → hidden)
        # la_z_dim = 6144, hidden = 5120
        passed, max_err = test_fp16_gemv_residual(
            K=6144, N=5120, dev=dev,
            description="la_out_proj K=6144 N=5120"
        )
        errors["fp16_6144x5120"] = max_err
        if not passed:
            all_pass = False

        # INT4 GEMV + Residual tests
        print("\n--- INT4 GEMV + Residual (VAL-DLR-007) ---")

        # Test 4: down_proj size (intermediate → hidden, K=local_inter, N=hidden)
        passed, max_err = test_int4_gemv_residual(
            K=4096, N=4096, group_size=128, k_splits=16, dev=dev,
            description="down_proj K=4096 N=4096"
        )
        errors["int4_4096x4096"] = max_err
        if not passed:
            all_pass = False

        # Test 5: wider FFN
        passed, max_err = test_int4_gemv_residual(
            K=11008, N=4096, group_size=128, k_splits=16, dev=dev,
            description="down_proj K=11008 N=4096"
        )
        errors["int4_11008x4096"] = max_err
        if not passed:
            all_pass = False

        # Test 6: Multi-call correctness for INT4 + residual
        multi_ok = test_multiple_calls_int4_residual(
            K=4096, N=4096, group_size=128, k_splits=16, dev=dev,
            description="K=4096 N=4096 multi-call"
        )
        if not multi_ok:
            all_pass = False

        # Test 7: Engine wiring check
        engine_ok = test_engine_residual_epilogues()
        if not engine_ok:
            all_pass = False

    finally:
        dev.cleanup()

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, err in errors.items():
        tol = FP16_TOLERANCE if "fp16" in name else INT4_TOLERANCE
        status = "PASS" if err < tol else "FAIL"
        print(f"  {name}: max_err={err:.6f} [{status}]")

    if all_pass:
        print(f"\nAll residual epilogue tests PASSED!")
        sys.exit(0)
    else:
        print(f"\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
