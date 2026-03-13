#!/usr/bin/env python3
"""
Test for fused QK-norm + RoPE kernel (m1-fused-qknorm-rope).

Verifies that the fused qknorm_rope_fused kernel:
1. Produces output matching separate batched_rmsnorm + rope within tolerance
2. Works for head_dim=256 (Qwen 3.5 27B) and head_dim=128
3. Handles multiple heads (typical Q: 8+ heads, K: 1-8 heads for GQA)
4. Max abs error < 2e-3 (VAL-DLR-005)

Also verifies engine wiring:
5. Engine uses fused kernel (no try/except fallback in _launch_qk_norm)
6. Batched path always taken (VAL-DLR-010)
"""

import ctypes
import subprocess
import sys
import time
import struct
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
TOLERANCE = 2e-3  # per VAL-DLR-005


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


def compute_reference_qknorm_rope(x_heads, norm_weight, cos_vals, sin_vals,
                                   num_heads, head_dim, half_rotary, eps=1e-6):
    """
    Reference implementation: per-head RMSNorm then partial RoPE.

    x_heads: [num_heads, head_dim] FP16 array
    norm_weight: [head_dim] FP16
    cos_vals: [half_rotary] FP16
    sin_vals: [half_rotary] FP16

    Returns: [num_heads, head_dim] FP16
    """
    result = x_heads.astype(np.float32).copy()
    norm_w = norm_weight.astype(np.float32)
    cos_f = cos_vals.astype(np.float32)
    sin_f = sin_vals.astype(np.float32)

    for h in range(num_heads):
        vec = result[h]

        # RMSNorm
        rms = np.sqrt(np.mean(vec ** 2) + eps)
        vec = vec / rms * norm_w

        # Partial RoPE: rotate pairs (2i, 2i+1) for i in [0, half_rotary)
        for i in range(half_rotary):
            x0 = vec[2 * i]
            x1 = vec[2 * i + 1]
            c = cos_f[i]
            s = sin_f[i]
            vec[2 * i]     = x0 * c - x1 * s
            vec[2 * i + 1] = x0 * s + x1 * c

        result[h] = vec

    return result.astype(np.float16)


def run_qknorm_rope_test(num_heads, head_dim, half_rotary, dev, description, seed=42):
    """
    Test fused qknorm_rope_fused vs sequential reference.

    Returns (passed, max_error).
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"  num_heads={num_heads}, head_dim={head_dim}, half_rotary={half_rotary}")
    print(f"{'='*60}")

    assert half_rotary * 2 <= head_dim, "rotary_dim must be <= head_dim"

    np.random.seed(seed)

    # Random Q/K heads and weights
    x_heads = (np.random.randn(num_heads, head_dim) * 0.3).astype(np.float16)
    norm_weight = (np.random.rand(head_dim) * 0.5 + 0.5).astype(np.float16)  # [0.5, 1.0]

    # RoPE cos/sin for a fixed position
    cos_vals = np.cos(np.arange(half_rotary, dtype=np.float32) * 0.1).astype(np.float16)
    sin_vals = np.sin(np.arange(half_rotary, dtype=np.float32) * 0.1).astype(np.float16)

    # Compute reference
    ref = compute_reference_qknorm_rope(
        x_heads, norm_weight, cos_vals, sin_vals,
        num_heads, head_dim, half_rotary
    )

    # Load fused kernel
    fused_hsaco = BUILD_DIR / "qknorm_rope.hsaco"
    module = dev.load_hsaco(str(fused_hsaco))
    func = dev.get_kernel(module, "qknorm_rope_fused")

    # Allocate GPU buffers
    x_flat = x_heads.flatten()
    d_x = dev.malloc(x_flat.nbytes)
    d_weight = dev.malloc(norm_weight.nbytes)
    d_cos = dev.malloc(cos_vals.nbytes)
    d_sin = dev.malloc(sin_vals.nbytes)

    dev.upload(d_x, x_flat.tobytes())
    dev.upload(d_weight, norm_weight.tobytes())
    dev.upload(d_cos, cos_vals.tobytes())
    dev.upload(d_sin, sin_vals.tobytes())

    # Encode eps as float bits (kernel takes float eps directly as float arg)
    eps = 1e-6
    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_weight),
        ctypes.c_uint64(d_cos),
        ctypes.c_uint64(d_sin),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(half_rotary),
        ctypes.c_float(eps),
    ]

    # Launch: one block per head
    dev.launch(func, (num_heads, 1, 1), (256, 1, 1), params)
    dev.synchronize()

    # Download result
    out_bytes = dev.download(d_x, x_flat.nbytes)
    out = np.frombuffer(out_bytes, dtype=np.float16).reshape(num_heads, head_dim)

    # Compare
    abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())

    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")
    print(f"ref[0,:4]  = {ref[0,:4]}")
    print(f"out[0,:4]  = {out[0,:4]}")

    # Check non-RoPE region is just normalized (no rotation)
    rope_end = 2 * half_rotary
    if rope_end < head_dim:
        # Compute just-rmsnorm reference for non-rope region
        norm_only_ref = np.zeros_like(ref)
        for h in range(num_heads):
            vec = x_heads[h].astype(np.float32)
            rms = np.sqrt(np.mean(vec ** 2) + eps)
            vec_normed = vec / rms * norm_weight.astype(np.float32)
            norm_only_ref[h] = vec_normed.astype(np.float16)

        nonrope_err = float(np.abs(
            out[:, rope_end:].astype(np.float32) -
            norm_only_ref[:, rope_end:].astype(np.float32)
        ).max())
        print(f"Non-RoPE region max error: {nonrope_err:.6f}")

    # Cleanup
    dev.free(d_x)
    dev.free(d_weight)
    dev.free(d_cos)
    dev.free(d_sin)

    passed = max_err < TOLERANCE
    if passed:
        print(f"PASS (max_err={max_err:.6f} < {TOLERANCE})")
    else:
        print(f"FAIL (max_err={max_err:.6f} >= {TOLERANCE})")
    return passed, max_err


def run_multi_call_test(num_heads, head_dim, half_rotary, dev, description, seed=99):
    """Test that fused kernel produces correct results across multiple sequential calls."""
    print(f"\n{'='*60}")
    print(f"Multi-call test: {description}")
    print(f"{'='*60}")

    np.random.seed(seed)
    fused_hsaco = BUILD_DIR / "qknorm_rope.hsaco"
    module = dev.load_hsaco(str(fused_hsaco))
    func = dev.get_kernel(module, "qknorm_rope_fused")

    all_pass = True
    for call_idx in range(3):
        x_heads = (np.random.randn(num_heads, head_dim) * 0.3).astype(np.float16)
        norm_weight = (np.random.rand(head_dim) * 0.5 + 0.5).astype(np.float16)
        pos = call_idx * 7 + 1
        cos_vals = np.cos(pos * np.arange(half_rotary, dtype=np.float32) * 0.01).astype(np.float16)
        sin_vals = np.sin(pos * np.arange(half_rotary, dtype=np.float32) * 0.01).astype(np.float16)

        ref = compute_reference_qknorm_rope(
            x_heads, norm_weight, cos_vals, sin_vals, num_heads, head_dim, half_rotary
        )

        x_flat = x_heads.flatten()
        d_x = dev.malloc(x_flat.nbytes)
        d_weight = dev.malloc(norm_weight.nbytes)
        d_cos = dev.malloc(cos_vals.nbytes)
        d_sin = dev.malloc(sin_vals.nbytes)

        dev.upload(d_x, x_flat.tobytes())
        dev.upload(d_weight, norm_weight.tobytes())
        dev.upload(d_cos, cos_vals.tobytes())
        dev.upload(d_sin, sin_vals.tobytes())

        params = [
            ctypes.c_uint64(d_x),
            ctypes.c_uint64(d_weight),
            ctypes.c_uint64(d_cos),
            ctypes.c_uint64(d_sin),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(half_rotary),
            ctypes.c_float(1e-6),
        ]
        dev.launch(func, (num_heads, 1, 1), (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_x, x_flat.nbytes), dtype=np.float16).reshape(num_heads, head_dim)
        max_err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())

        dev.free(d_x); dev.free(d_weight); dev.free(d_cos); dev.free(d_sin)

        ok = max_err < TOLERANCE
        print(f"  Call {call_idx+1}: max_err={max_err:.6f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    return all_pass


def run_perf_test(num_heads, head_dim, half_rotary, dev, description):
    """Measure latency of fused vs reference (2 batched_rmsnorm + 2 rope) path."""
    print(f"\n{'='*60}")
    print(f"Performance: {description}")
    print(f"{'='*60}")

    np.random.seed(123)
    fused_hsaco = BUILD_DIR / "qknorm_rope.hsaco"
    module = dev.load_hsaco(str(fused_hsaco))
    func_fused = dev.get_kernel(module, "qknorm_rope_fused")

    # Load separate kernels for baseline comparison
    batched_norm_hsaco = BUILD_DIR / "batched_rmsnorm.hsaco"
    bn_module = dev.load_hsaco(str(batched_norm_hsaco))
    func_batched_norm = dev.get_kernel(bn_module, "batched_rmsnorm_fp16")

    # Try to load assembly rope kernel
    rope_hsaco = BUILD_DIR / "rope.hsaco"
    if not rope_hsaco.exists():
        print("  rope.hsaco not found, skipping performance comparison")
        return None, None

    rope_module = dev.load_hsaco(str(rope_hsaco))
    func_rope = dev.get_kernel(rope_module, "rope_fp16")

    x_heads = (np.random.randn(num_heads, head_dim) * 0.3).astype(np.float16)
    norm_weight = (np.random.rand(head_dim) * 0.5 + 0.5).astype(np.float16)
    cos_vals = np.cos(np.arange(half_rotary, dtype=np.float32) * 0.1).astype(np.float16)
    sin_vals = np.sin(np.arange(half_rotary, dtype=np.float32) * 0.1).astype(np.float16)

    # Allocate GPU buffers
    x_flat = x_heads.flatten()
    d_x_fused = dev.malloc(x_flat.nbytes)
    d_x_sep = dev.malloc(x_flat.nbytes)
    d_weight = dev.malloc(norm_weight.nbytes)
    d_cos = dev.malloc(cos_vals.nbytes)
    d_sin = dev.malloc(sin_vals.nbytes)

    dev.upload(d_weight, norm_weight.tobytes())
    dev.upload(d_cos, cos_vals.tobytes())
    dev.upload(d_sin, sin_vals.tobytes())

    # Fused params
    fused_params = [
        ctypes.c_uint64(d_x_fused),
        ctypes.c_uint64(d_weight),
        ctypes.c_uint64(d_cos),
        ctypes.c_uint64(d_sin),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(half_rotary),
        ctypes.c_float(1e-6),
    ]

    # Batched rmsnorm params
    eps_bits = struct.unpack('<I', struct.pack('<f', 1e-6))[0]
    bn_params = [
        ctypes.c_uint64(d_x_sep),
        ctypes.c_uint64(d_weight),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(eps_bits),
    ]

    # Rope params (Grid: (num_tokens=1, num_heads), Block: (head_dim/2))
    # rope_fp16 takes: x, cos_tab, sin_tab, head_dim, num_heads
    # but it's currently unused in this path - we just measure fused launch
    N_ITERS = 200

    # Warmup fused
    for _ in range(20):
        dev.upload(d_x_fused, x_flat.tobytes())
        dev.launch(func_fused, (num_heads, 1, 1), (256, 1, 1), fused_params)
    dev.synchronize()

    # Benchmark fused
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.upload(d_x_fused, x_flat.tobytes())
        dev.launch(func_fused, (num_heads, 1, 1), (256, 1, 1), fused_params)
    dev.synchronize()
    t1 = time.perf_counter()
    fused_us = (t1 - t0) / N_ITERS * 1e6

    # Benchmark separate: batched_rmsnorm only (no rope, just the norm part)
    for _ in range(20):
        dev.upload(d_x_sep, x_flat.tobytes())
        dev.launch(func_batched_norm, (num_heads, 1, 1), (256, 1, 1), bn_params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        dev.upload(d_x_sep, x_flat.tobytes())
        dev.launch(func_batched_norm, (num_heads, 1, 1), (256, 1, 1), bn_params)
    dev.synchronize()
    t1 = time.perf_counter()
    norm_only_us = (t1 - t0) / N_ITERS * 1e6

    print(f"  Fused qknorm+rope: {fused_us:.1f} us/call")
    print(f"  Separate norm (1 launch): {norm_only_us:.1f} us/call")
    print(f"  (Fused replaces 2 norm + 2 rope = 4 launches total per decode step)")

    dev.free(d_x_fused)
    dev.free(d_x_sep)
    dev.free(d_weight)
    dev.free(d_cos)
    dev.free(d_sin)

    return fused_us, norm_only_us


def test_engine_uses_fused_kernel():
    """Verify engine.py uses the fused qknorm_rope kernel without try/except fallback."""
    print(f"\n{'='*60}")
    print("Test: Engine uses fused QK-norm+RoPE kernel (VAL-DLR-010)")
    print(f"{'='*60}")

    import inspect
    import re
    from src.inference.engine import InferenceEngine

    # Check _launch_qk_norm has no silent try/except fallback
    # (Look for actual "try:" statement, not just the word "try" in comments)
    src_qk = inspect.getsource(InferenceEngine._launch_qk_norm)
    has_try_except = bool(re.search(r'^\s+try\s*:', src_qk, re.MULTILINE))
    uses_fused = "qknorm_rope" in inspect.getsource(InferenceEngine._decode_full_attention)

    # Check _decode_full_attention no longer has separate _launch_qk_norm + _launch_rope
    src_decode = inspect.getsource(InferenceEngine._decode_full_attention)
    has_separate_qk_norm = "_launch_qk_norm" in src_decode
    has_separate_rope = "_launch_rope" in src_decode
    has_fused_qknorm_rope = "_launch_qknorm_rope" in src_decode

    print(f"  _launch_qk_norm has try/except fallback:   {has_try_except}")
    print(f"  _decode_full_attention uses _launch_qknorm_rope: {has_fused_qknorm_rope}")
    print(f"  _decode_full_attention has _launch_qk_norm: {has_separate_qk_norm}")
    print(f"  _decode_full_attention has _launch_rope:    {has_separate_rope}")

    # The fused path should be present, and the separate pair should NOT be present
    passed = (not has_try_except and
              has_fused_qknorm_rope and
              not has_separate_qk_norm and
              not has_separate_rope)

    if passed:
        print("PASS")
    else:
        print("FAIL: Engine not properly wired to use fused kernel")
    return passed


def main():
    print("=" * 70)
    print("Fused QK-norm + RoPE Kernel Correctness Tests (m1-fused-qknorm-rope)")
    print("=" * 70)

    print("\nBuilding kernels...")
    build_kernel("qknorm_rope")
    build_kernel("batched_rmsnorm")  # For baseline comparison
    print("Build successful.\n")

    dev = GPUDevice(0)
    all_pass = True
    errors = {}

    try:
        # Test 1: head_dim=256 (Qwen 3.5 27B), typical Q heads (48 heads local)
        # For TP=1: num_q_heads=8, num_kv_heads=1 (GQA)
        # Actual Qwen config: 8 attention heads, 1 kv head for GQA
        passed, max_err = run_qknorm_rope_test(
            num_heads=8, head_dim=256, half_rotary=32,
            dev=dev,
            description="head_dim=256, 8 Q heads (Qwen 3.5 GQA Q)"
        )
        errors["256_8heads"] = max_err
        if not passed: all_pass = False

        # Test 2: head_dim=256, KV heads
        passed, max_err = run_qknorm_rope_test(
            num_heads=1, head_dim=256, half_rotary=32,
            dev=dev,
            description="head_dim=256, 1 KV head (Qwen 3.5 GQA K)"
        )
        errors["256_1head"] = max_err
        if not passed: all_pass = False

        # Test 3: head_dim=128 (alternate config)
        passed, max_err = run_qknorm_rope_test(
            num_heads=8, head_dim=128, half_rotary=16,
            dev=dev,
            description="head_dim=128, 8 heads"
        )
        errors["128_8heads"] = max_err
        if not passed: all_pass = False

        # Test 4: head_dim=256, large num_heads (full non-TP model: 48 Q heads)
        passed, max_err = run_qknorm_rope_test(
            num_heads=48, head_dim=256, half_rotary=32,
            dev=dev,
            description="head_dim=256, 48 Q heads (full model)"
        )
        errors["256_48heads"] = max_err
        if not passed: all_pass = False

        # Test 5: head_dim=256, 8 KV heads (full model)
        passed, max_err = run_qknorm_rope_test(
            num_heads=8, head_dim=256, half_rotary=32,
            dev=dev,
            description="head_dim=256, 8 KV heads (full model)"
        )
        errors["256_8kv"] = max_err
        if not passed: all_pass = False

        # Test 6: Multi-call correctness (verify no state leakage)
        multi_ok = run_multi_call_test(
            num_heads=8, head_dim=256, half_rotary=32,
            dev=dev,
            description="Multi-call correctness head_dim=256"
        )
        if not multi_ok: all_pass = False

        # Test 7: Engine structural check
        engine_ok = test_engine_uses_fused_kernel()
        if not engine_ok: all_pass = False

        # Performance (informational)
        run_perf_test(
            num_heads=8, head_dim=256, half_rotary=32,
            dev=dev,
            description="Q heads: 8 heads, head_dim=256"
        )

    finally:
        dev.cleanup()

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, err in errors.items():
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"  {name}: max_err={err:.6f} [{status}]")

    if all_pass:
        print(f"\nAll fused QK-norm+RoPE tests PASSED!")
        sys.exit(0)
    else:
        print(f"\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
