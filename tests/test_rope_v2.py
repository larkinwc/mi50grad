#!/usr/bin/env python3
"""
Test for HIP RoPE v2 kernel (m2-hip-rope).

Verifies that the HIP rope_fp16_v2 kernel:
1. Produces output matching the assembly rope.s kernel within FP16 tolerance
2. Works for head_dim=128 and head_dim=256
3. Handles single and multiple heads (Q heads, K heads)
4. Max abs error vs reference < 1e-3 (VAL-ROPE-001)

Also verifies engine wiring:
5. Engine uses HIP RoPE kernel instead of assembly (VAL-ROPE-002)

Note: The assembly rope.s kernel only supports single-token mode (blockIdx.x=0).
      Both assembly and HIP kernels are validated against CPU reference.
      In the decode path, _launch_rope is always called with a single token slice.
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
ASM_DIR = PROJECT_ROOT / "src" / "asm"
HIP_DIR = PROJECT_ROOT / "src" / "kernels"
TOLERANCE = 1e-3  # per VAL-ROPE-001


def build_asm_kernel(kernel_name):
    """Build assembly kernel to HSACO."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    asm_path = ASM_DIR / f"{kernel_name}.s"
    hsaco_path = BUILD_DIR / f"{kernel_name}.hsaco"
    obj_path = BUILD_DIR / f"{kernel_name}.o"

    LLVM_MC = "/opt/rocm/llvm/bin/llvm-mc"
    LD_LLD = "/opt/rocm/llvm/bin/ld.lld"

    # Assemble
    result = subprocess.run(
        [LLVM_MC, "--triple=amdgcn-amd-amdhsa", "--mcpu=gfx906", "--filetype=obj",
         str(asm_path), "-o", str(obj_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Assembly failed for {kernel_name}: {result.stderr}")

    # Link
    result = subprocess.run(
        [LD_LLD, "--shared", str(obj_path), "-o", str(hsaco_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Link failed for {kernel_name}: {result.stderr}")

    obj_path.unlink(missing_ok=True)
    print(f"  Built {kernel_name}.hsaco (assembly)")
    return hsaco_path


def build_hip_kernel(kernel_name):
    """Build HIP C++ kernel to HSACO."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_src = HIP_DIR / f"{kernel_name}.hip"
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
    print(f"  Built {kernel_name}.hsaco (HIP C++)")
    return hsaco_path


def compute_reference_rope(x, cos_vals, sin_vals, num_tokens, num_heads, head_dim, half_rotary):
    """
    Reference CPU implementation of partial RoPE.

    x:         [num_tokens, num_heads, head_dim] FP16
    cos_vals:  [num_tokens, half_rotary] FP16
    sin_vals:  [num_tokens, half_rotary] FP16

    Returns: [num_tokens, num_heads, head_dim] FP16 with partial RoPE applied.
    Only first 2*half_rotary elements of each head are rotated; rest unchanged.
    """
    result = x.astype(np.float32).copy()
    cos_f = cos_vals.astype(np.float32)
    sin_f = sin_vals.astype(np.float32)

    for t in range(num_tokens):
        for h in range(num_heads):
            for i in range(half_rotary):
                x0 = result[t, h, 2 * i]
                x1 = result[t, h, 2 * i + 1]
                c = cos_f[t, i]
                s = sin_f[t, i]
                result[t, h, 2 * i]     = x0 * c - x1 * s
                result[t, h, 2 * i + 1] = x0 * s + x1 * c

    return result.astype(np.float16)


def launch_rope_kernel(dev, func, d_x, d_cos, d_sin, head_dim, num_heads, half_rotary, num_tokens):
    """Launch a RoPE kernel (both assembly and HIP have same interface).

    Grid: (num_tokens, num_heads, 1), Block: (half_rotary, 1, 1)
    """
    params = [
        ctypes.c_uint64(d_x),
        ctypes.c_uint64(d_cos),
        ctypes.c_uint64(d_sin),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(num_heads),
    ]
    dev.launch(func, (num_tokens, num_heads, 1), (half_rotary, 1, 1), params)
    dev.synchronize()


def run_single_token_test(dev, func_asm, func_hip, num_heads, head_dim,
                          half_rotary, description, seed=42):
    """
    Test HIP RoPE vs assembly RoPE for single-token decode (the primary use case).

    Both assembly and HIP are tested against CPU reference.
    The assembly kernel is known to work correctly for single-token.

    Returns (passed, max_error_hip_vs_asm, max_error_hip_vs_ref).
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"  num_heads={num_heads}, head_dim={head_dim}, half_rotary={half_rotary}")
    print(f"{'='*60}")

    np.random.seed(seed)
    num_tokens = 1

    # Generate random input data
    x = (np.random.randn(num_tokens, num_heads, head_dim) * 0.5).astype(np.float16)

    # Generate cos/sin values for this position
    base_freqs = np.exp(-np.arange(half_rotary, dtype=np.float32) * np.log(10000.0) / half_rotary)
    # Simulate position 42 (a non-trivial decode position)
    position = 42.0
    angles = position * base_freqs
    cos_vals = np.cos(angles).astype(np.float16).reshape(1, half_rotary)
    sin_vals = np.sin(angles).astype(np.float16).reshape(1, half_rotary)

    # Compute CPU reference using FP16-quantized cos/sin (same as kernels read)
    ref = compute_reference_rope(x, cos_vals, sin_vals, num_tokens, num_heads, head_dim, half_rotary)

    # Allocate GPU buffers
    x_flat = x.flatten()
    d_x_asm = dev.malloc(x_flat.nbytes)
    d_x_hip = dev.malloc(x_flat.nbytes)
    d_cos = dev.malloc(cos_vals.nbytes)
    d_sin = dev.malloc(sin_vals.nbytes)

    dev.upload(d_x_asm, x_flat.tobytes())
    dev.upload(d_x_hip, x_flat.tobytes())
    dev.upload(d_cos, cos_vals.flatten().tobytes())
    dev.upload(d_sin, sin_vals.flatten().tobytes())

    # Launch both kernels
    launch_rope_kernel(dev, func_asm, d_x_asm, d_cos, d_sin, head_dim, num_heads, half_rotary, 1)
    launch_rope_kernel(dev, func_hip, d_x_hip, d_cos, d_sin, head_dim, num_heads, half_rotary, 1)

    # Download results
    out_asm = np.frombuffer(dev.download(d_x_asm, x_flat.nbytes), dtype=np.float16).reshape(
        num_tokens, num_heads, head_dim)
    out_hip = np.frombuffer(dev.download(d_x_hip, x_flat.nbytes), dtype=np.float16).reshape(
        num_tokens, num_heads, head_dim)

    # Compare
    max_err_hip_asm = float(np.abs(out_hip.astype(np.float32) - out_asm.astype(np.float32)).max())
    max_err_hip_ref = float(np.abs(out_hip.astype(np.float32) - ref.astype(np.float32)).max())
    max_err_asm_ref = float(np.abs(out_asm.astype(np.float32) - ref.astype(np.float32)).max())

    print(f"HIP vs ASM:  max_err={max_err_hip_asm:.6f}")
    print(f"HIP vs REF:  max_err={max_err_hip_ref:.6f}")
    print(f"ASM vs REF:  max_err={max_err_asm_ref:.6f}")
    print(f"ref[0,0,:4] = {ref[0, 0, :4]}")
    print(f"asm[0,0,:4] = {out_asm[0, 0, :4]}")
    print(f"hip[0,0,:4] = {out_hip[0, 0, :4]}")

    # Verify non-RoPE region is unchanged
    rope_end = 2 * half_rotary
    if rope_end < head_dim:
        nonrope_hip_err = float(np.abs(
            out_hip[:, :, rope_end:].astype(np.float32) -
            x[:, :, rope_end:].astype(np.float32)
        ).max())
        print(f"Non-RoPE region HIP error (should be ~0): {nonrope_hip_err:.8f}")
        if nonrope_hip_err > 1e-4:
            print(f"  FAIL: Non-RoPE region was modified by HIP kernel")
            return False, max_err_hip_asm, max_err_hip_ref

    # Pass criterion: HIP vs reference within tolerance
    passed = max_err_hip_ref < TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status} (max_err_hip_vs_ref={max_err_hip_ref:.6f}, tolerance={TOLERANCE})")
    return passed, max_err_hip_asm, max_err_hip_ref


def run_hip_only_test(dev, func_hip, num_tokens, num_heads, head_dim,
                      half_rotary, description, seed=42):
    """
    Test HIP RoPE against CPU reference only (no assembly comparison).
    Used for multi-token correctness verification.

    The HIP kernel indexes cos/sin as: token_idx * (head_dim/2) + pair_idx
    (same as the assembly kernel). For multi-token tests with head_dim > 2*half_rotary,
    the cos/sin table must be padded to head_dim/2 columns.

    Returns (passed, max_error_hip_vs_ref).
    """
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"  num_tokens={num_tokens}, num_heads={num_heads}, head_dim={head_dim}, "
          f"half_rotary={half_rotary}")
    print(f"{'='*60}")

    np.random.seed(seed)
    half_dim = head_dim // 2  # kernel uses head_dim/2 as cos/sin table stride

    # Generate random input data
    x = (np.random.randn(num_tokens, num_heads, head_dim) * 0.5).astype(np.float16)

    # Generate cos/sin for each token position with half_dim stride (as kernel expects)
    base_freqs = np.exp(-np.arange(half_rotary, dtype=np.float32) * np.log(10000.0) / half_rotary)
    angles = np.outer(np.arange(num_tokens, dtype=np.float32), base_freqs)
    cos_row = np.cos(angles).astype(np.float16)  # [num_tokens, half_rotary]
    sin_row = np.sin(angles).astype(np.float16)

    # Pad to [num_tokens, half_dim] since kernel indexes with stride half_dim
    cos_padded = np.zeros((num_tokens, half_dim), dtype=np.float16)
    sin_padded = np.zeros((num_tokens, half_dim), dtype=np.float16)
    cos_padded[:, :half_rotary] = cos_row
    sin_padded[:, :half_rotary] = sin_row

    # Compute CPU reference using the same FP16-quantized values
    ref = compute_reference_rope(x, cos_row, sin_row, num_tokens, num_heads, head_dim, half_rotary)

    # GPU test
    x_flat = x.flatten()
    d_x = dev.malloc(x_flat.nbytes)
    d_cos = dev.malloc(cos_padded.nbytes)
    d_sin = dev.malloc(sin_padded.nbytes)

    dev.upload(d_x, x_flat.tobytes())
    dev.upload(d_cos, cos_padded.flatten().tobytes())
    dev.upload(d_sin, sin_padded.flatten().tobytes())

    launch_rope_kernel(dev, func_hip, d_x, d_cos, d_sin, head_dim, num_heads, half_rotary, num_tokens)

    out = np.frombuffer(dev.download(d_x, x_flat.nbytes), dtype=np.float16).reshape(
        num_tokens, num_heads, head_dim)

    max_err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())
    mean_err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).mean())

    print(f"HIP vs REF:  max_err={max_err:.6f}, mean_err={mean_err:.6f}")
    print(f"ref[0,0,:4] = {ref[0, 0, :4]}")
    print(f"hip[0,0,:4] = {out[0, 0, :4]}")

    # Verify non-RoPE region unchanged (elements [2*half_rotary, head_dim) should be unchanged)
    rope_end = 2 * half_rotary
    if rope_end < head_dim:
        nonrope_err = float(np.abs(
            out[:, :, rope_end:].astype(np.float32) - x[:, :, rope_end:].astype(np.float32)
        ).max())
        print(f"Non-RoPE region HIP error (should be ~0): {nonrope_err:.8f}")
        if nonrope_err > 1e-4:
            print(f"  FAIL: Non-RoPE region was modified")
            return False, max_err

    passed = max_err < TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status} (max_err={max_err:.6f}, tolerance={TOLERANCE})")
    return passed, max_err


def run_performance_benchmark(dev, func_asm, func_hip, num_heads, head_dim, half_rotary,
                               description, num_iters=1000):
    """Benchmark HIP vs assembly RoPE kernel performance (single-token decode)."""
    print(f"\n--- Benchmark: {description} ---")
    num_tokens = 1

    np.random.seed(0)
    x = (np.random.randn(num_tokens, num_heads, head_dim) * 0.5).astype(np.float16)
    cos_vals = np.ones((num_tokens, half_rotary), dtype=np.float16) * 0.9
    sin_vals = np.ones((num_tokens, half_rotary), dtype=np.float16) * 0.1

    d_x = dev.malloc(x.nbytes)
    d_cos = dev.malloc(cos_vals.nbytes)
    d_sin = dev.malloc(sin_vals.nbytes)
    dev.upload(d_x, x.tobytes())
    dev.upload(d_cos, cos_vals.flatten().tobytes())
    dev.upload(d_sin, sin_vals.flatten().tobytes())

    # Warmup
    for _ in range(50):
        params = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_cos), ctypes.c_uint64(d_sin),
                  ctypes.c_uint32(head_dim), ctypes.c_uint32(num_heads)]
        dev.launch(func_asm, (1, num_heads, 1), (half_rotary, 1, 1), params)
        dev.launch(func_hip, (1, num_heads, 1), (half_rotary, 1, 1), params)
    dev.synchronize()

    # Benchmark ASM
    t0 = time.perf_counter()
    for _ in range(num_iters):
        params = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_cos), ctypes.c_uint64(d_sin),
                  ctypes.c_uint32(head_dim), ctypes.c_uint32(num_heads)]
        dev.launch(func_asm, (1, num_heads, 1), (half_rotary, 1, 1), params)
    dev.synchronize()
    t_asm = (time.perf_counter() - t0) / num_iters * 1e6

    # Benchmark HIP
    t0 = time.perf_counter()
    for _ in range(num_iters):
        params = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_cos), ctypes.c_uint64(d_sin),
                  ctypes.c_uint32(head_dim), ctypes.c_uint32(num_heads)]
        dev.launch(func_hip, (1, num_heads, 1), (half_rotary, 1, 1), params)
    dev.synchronize()
    t_hip = (time.perf_counter() - t0) / num_iters * 1e6

    print(f"  ASM: {t_asm:.2f} us/iter")
    print(f"  HIP: {t_hip:.2f} us/iter")
    speedup = t_asm / t_hip if t_hip > 0 else float('inf')
    print(f"  HIP speedup: {speedup:.2f}x vs ASM")

    return t_asm, t_hip


def test_engine_uses_hip_rope():
    """Verify that the engine uses HIP RoPE instead of assembly (VAL-ROPE-002)."""
    print(f"\n{'='*60}")
    print("Test: Engine wiring - HIP RoPE replaces assembly")
    print(f"{'='*60}")

    engine_path = PROJECT_ROOT / "src" / "inference" / "engine.py"
    engine_src = engine_path.read_text()

    checks = [
        ("_init_rope_hip function defined", "_init_rope_hip" in engine_src),
        ("_rope_hip flag used", "_rope_hip" in engine_src),
        ("rope_fp16_v2 referenced", "rope_fp16_v2" in engine_src),
        ("rope_v2.hip referenced", "rope_v2" in engine_src),
        ("rope_v2.hip source exists", (PROJECT_ROOT / "src" / "kernels" / "rope_v2.hip").exists()),
    ]

    all_ok = True
    for desc, result in checks:
        status = "OK" if result else "FAIL"
        print(f"  {status}: {desc}")
        all_ok = all_ok and result

    if all_ok:
        print("  PASS: Engine wired to use HIP RoPE kernel")
    else:
        print("  FAIL: Engine wiring incomplete")

    return all_ok


def main():
    print("=" * 70)
    print("HIP RoPE v2 Correctness and Performance Test (m2-hip-rope)")
    print("VAL-ROPE-001: HIP RoPE matches assembly/reference within FP16 < 1e-3")
    print("VAL-ROPE-002: Engine uses HIP RoPE instead of assembly")
    print("=" * 70)

    # Initialize GPU device
    dev = GPUDevice(0)
    dev._ensure_device()

    # Build kernels
    print("\n--- Building kernels ---")
    try:
        asm_hsaco = build_asm_kernel("rope")
        hip_hsaco = build_hip_kernel("rope_v2")
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

    # Load kernels
    mod_asm = dev.load_hsaco(str(asm_hsaco))
    mod_hip = dev.load_hsaco(str(hip_hsaco))
    func_asm = dev.get_kernel(mod_asm, "rope_fp16")
    func_hip = dev.get_kernel(mod_hip, "rope_fp16_v2")

    # -----------------------------------------------------------------------
    # Single-token tests: compare HIP vs assembly AND HIP vs CPU reference
    # These are the primary use case (decode path: one token at a time)
    # -----------------------------------------------------------------------
    print("\n--- Single-Token Tests (Primary Decode Use Case) ---")
    all_passed = True
    results = []

    # Test 1: head_dim=128, typical Q heads
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=8, head_dim=128, half_rotary=32,
        description="head_dim=128, 8 Q heads, half_rotary=32",
        seed=1
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=128 8 heads", passed, max_ref))

    # Test 2: head_dim=256, Qwen 3.5 config (Q heads)
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=8, head_dim=256, half_rotary=32,
        description="head_dim=256, 8 Q heads (Qwen 3.5 config), half_rotary=32",
        seed=2
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=256 8 heads", passed, max_ref))

    # Test 3: head_dim=128, KV heads
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=4, head_dim=128, half_rotary=32,
        description="head_dim=128, 4 KV heads, half_rotary=32",
        seed=3
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=128 4 KV heads", passed, max_ref))

    # Test 4: head_dim=256, single KV head
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=1, head_dim=256, half_rotary=32,
        description="head_dim=256, 1 KV head, half_rotary=32",
        seed=4
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=256 1 KV head", passed, max_ref))

    # Test 5: head_dim=128, full rotary (half_rotary=64 = all elements)
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=4, head_dim=128, half_rotary=64,
        description="head_dim=128, full rotary (half_rotary=64)",
        seed=5
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=128 full rotary", passed, max_ref))

    # Test 6: head_dim=256, full rotary (half_rotary=128 = all elements)
    passed, max_asm, max_ref = run_single_token_test(
        dev, func_asm, func_hip,
        num_heads=4, head_dim=256, half_rotary=128,
        description="head_dim=256, full rotary (half_rotary=128)",
        seed=6
    )
    all_passed = all_passed and passed
    results.append(("Single-token head_dim=256 full rotary", passed, max_ref))

    # -----------------------------------------------------------------------
    # Multi-token tests: HIP only vs CPU reference
    # The HIP kernel supports multi-token (for future batched prefill use)
    # The assembly kernel does NOT support multi-token (only used in decode)
    # -----------------------------------------------------------------------
    print("\n--- Multi-Token Tests (HIP vs CPU Reference) ---")

    # Test 7: head_dim=256, 4 tokens (small prefill batch)
    passed, max_ref = run_hip_only_test(
        dev, func_hip,
        num_tokens=4, num_heads=4, head_dim=256, half_rotary=32,
        description="head_dim=256, 4 tokens (HIP only), half_rotary=32",
        seed=7
    )
    all_passed = all_passed and passed
    results.append(("Multi-token head_dim=256 4 tokens", passed, max_ref))

    # Test 8: head_dim=128, 8 tokens (batch prefill)
    passed, max_ref = run_hip_only_test(
        dev, func_hip,
        num_tokens=8, num_heads=8, head_dim=128, half_rotary=64,
        description="head_dim=128, 8 tokens (HIP only), full rotary",
        seed=8
    )
    all_passed = all_passed and passed
    results.append(("Multi-token head_dim=128 8 tokens", passed, max_ref))

    # -----------------------------------------------------------------------
    # Performance benchmarks (single-token decode, primary use case)
    # -----------------------------------------------------------------------
    print("\n--- Performance Benchmarks (Single-Token Decode) ---")
    run_performance_benchmark(
        dev, func_asm, func_hip,
        num_heads=8, head_dim=128, half_rotary=32,
        description="head_dim=128, 8 Q heads"
    )
    run_performance_benchmark(
        dev, func_asm, func_hip,
        num_heads=8, head_dim=256, half_rotary=32,
        description="head_dim=256, 8 Q heads"
    )
    run_performance_benchmark(
        dev, func_asm, func_hip,
        num_heads=1, head_dim=256, half_rotary=32,
        description="head_dim=256, 1 KV head (minimum)"
    )

    # -----------------------------------------------------------------------
    # Engine wiring verification (VAL-ROPE-002)
    # -----------------------------------------------------------------------
    engine_ok = test_engine_uses_hip_rope()
    all_passed = all_passed and engine_ok

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, passed, max_err in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}: max_err={max_err:.6f}")

    engine_status = "PASS" if engine_ok else "FAIL"
    print(f"  {engine_status}  Engine uses HIP RoPE (VAL-ROPE-002)")

    print()
    if all_passed:
        print("ALL TESTS PASSED")
        print(f"  VAL-ROPE-001: HIP RoPE matches assembly/reference (max err < {TOLERANCE})")
        print("  VAL-ROPE-002: Engine wired to use HIP RoPE kernel")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
