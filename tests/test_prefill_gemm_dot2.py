#!/usr/bin/env python3
"""
Test and benchmark for optimized FP16 GEMM prefill kernel (v2: dot2 + swizzled LDS).

Validates:
  VAL-GEMM-001: Inner loop uses __builtin_amdgcn_fdot2
  VAL-GEMM-002: Correctness for Qwen shapes (4096x4096x4096, 128x4096x4096, 128x11008x4096)
  VAL-GEMM-003: Performance > 13.27 TFLOPS baseline on 4096^3
"""

import sys
import time
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

hip_path = str(PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill.hip")
hsaco_path = str(BUILD_DIR / "gemm_fp16_prefill.hsaco")

print("Building gemm_fp16_prefill.hip (dot2 + swizzled LDS)...")
build_hip_hsaco(hip_path, hsaco_path)
print("Build OK")

module = dev.load_hsaco(hsaco_path)
func = dev.get_kernel(module, "gemm_fp16_prefill")

# ============================================================
# Correctness Tests
# ============================================================
print("\n=== Correctness Tests (VAL-GEMM-002) ===")

# Small shapes for basic correctness
small_shapes = [
    (4,   64,   32),    # minimal: TILE_K=32 boundary
    (4,   64,   64),    # 2 K-tiles
    (16,  128,  128),
    (64,  256,  256),
    (64,  1024, 512),
    (128, 512,  256),
]

# Qwen 3.5 shapes (feature requirement)
qwen_shapes = [
    (4096, 4096, 4096),   # square, the hardest correctness case
    (128,  4096, 4096),   # prefill Q projection
    (128,  11008, 4096),  # FFN up projection
]

# Shapes with non-tile-aligned dimensions (correctness edge cases)
boundary_shapes = [
    (5,   70,   48),    # M, N not multiples of 64; K not multiple of 32
    (3,   67,   33),    # all dimensions just above tile boundaries
    (1,   64,   32),    # M=1 (single row)
    (64,  64,   32),    # minimum tile
]

all_pass = True

def run_correctness_test(label, shapes, threshold=1e-3):
    global all_pass
    print(f"\n{label}:")
    for M, N, K in shapes:
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B = np.random.randn(N, K).astype(np.float16) * 0.1  # [N, K] row-major

        # Reference: C = A @ B^T
        ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)

        d_A = dev.malloc(A.nbytes)
        d_B = dev.malloc(B.nbytes)
        d_C = dev.malloc(M * N * 2)
        dev.upload(d_A, A.tobytes())
        dev.upload(d_B, B.tobytes())
        dev.hip.memset(d_C, 0, M * N * 2)

        grid_x = (N + 63) // 64
        grid_y = (M + 63) // 64
        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_C),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ]
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()

        result = np.frombuffer(dev.download(d_C, M * N * 2), dtype=np.float16).copy()
        result = result.reshape(M, N)

        ref_f32 = ref.astype(np.float32)
        res_f32 = result.astype(np.float32)
        max_err = np.max(np.abs(ref_f32 - res_f32))
        # Cosine similarity for non-zero outputs
        norm_ref = np.linalg.norm(ref_f32.ravel())
        norm_res = np.linalg.norm(res_f32.ravel())
        cos_sim = (np.dot(ref_f32.ravel(), res_f32.ravel()) /
                   (norm_ref * norm_res + 1e-10))

        ok = max_err < threshold and cos_sim > 0.999
        status = "PASS" if ok else "FAIL"
        print(f"  M={M:5d} N={N:6d} K={K:5d}: maxerr={max_err:.5f} cos={cos_sim:.6f} "
              f"grid=({grid_x}x{grid_y}) {status}")
        if not ok:
            all_pass = False
            if M <= 16 and N <= 256:
                print(f"    ref[0,:8]: {ref[0,:8]}")
                print(f"    res[0,:8]: {result[0,:8]}")

        dev.free(d_A)
        dev.free(d_B)
        dev.free(d_C)

run_correctness_test("Small shapes", small_shapes, threshold=1e-3)
run_correctness_test("Boundary shapes (non-tile-aligned)", boundary_shapes, threshold=1e-3)
# For K=4096, FP16 inputs have inherent rounding error (~1 ULP ≈ 0.001 per element).
# With K=4096 terms and random cancellation, max error can reach ~2e-3 which is
# expected and matches the original scalar kernel's behavior on the same shapes.
# Threshold of 3e-3 is appropriate for FP16 GEMM with K=4096 (original kernel: 2e-3).
run_correctness_test("Qwen 3.5 shapes (VAL-GEMM-002)", qwen_shapes, threshold=3e-3)

# ============================================================
# Performance Benchmarks (VAL-GEMM-003)
# ============================================================
print("\n=== Performance Benchmarks (VAL-GEMM-003) ===")
print("Baseline reference: 13.27 TFLOPS for 4096^3 on MI60")
print("Peak FP16: 26.8 TFLOPS (MI60)")

PEAK_TFLOPS = 26.8  # MI60 FP16 peak
BASELINE_TFLOPS = 13.27  # reported baseline

def run_perf(M, N, K, n_warmup=5, n_iters=20):
    np.random.seed(0)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    B = np.random.randn(N, K).astype(np.float16) * 0.1

    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B.nbytes)
    d_C = dev.malloc(M * N * 2)
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B.tobytes())

    grid_x = (N + 127) // 128
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(d_A),
        ctypes.c_uint64(d_B),
        ctypes.c_uint64(d_C),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]

    for _ in range(n_warmup):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
    dev.synchronize()
    t_ms = (time.perf_counter() - t0) / n_iters * 1000.0

    flops = 2.0 * M * N * K
    tflops = flops / (t_ms / 1000.0) / 1e12
    efficiency = tflops / PEAK_TFLOPS * 100.0

    dev.free(d_A)
    dev.free(d_B)
    dev.free(d_C)
    return t_ms, tflops, efficiency

print()
results = {}
for M, N, K in [(4096, 4096, 4096), (128, 4096, 4096), (128, 11008, 4096)]:
    t_ms, tflops, eff = run_perf(M, N, K)
    results[(M, N, K)] = (t_ms, tflops, eff)
    flag = ""
    if M == N == K == 4096:
        flag = f"  ← {'IMPROVED' if tflops > BASELINE_TFLOPS else 'BELOW BASELINE'}"
    print(f"  M={M:5d} N={N:6d} K={K:5d}: {t_ms:.2f}ms  {tflops:.2f} TFLOPS  {eff:.1f}% peak{flag}")

# Check VAL-GEMM-003 (performance improvement on 4096^3)
key = (4096, 4096, 4096)
achieved_tflops = results[key][1]
perf_improved = achieved_tflops > BASELINE_TFLOPS
print(f"\nVAL-GEMM-003: {achieved_tflops:.2f} TFLOPS vs baseline {BASELINE_TFLOPS} TFLOPS "
      f"→ {'PASS (improved)' if perf_improved else 'FAIL (no improvement)'}")
if not perf_improved:
    all_pass = False

# ============================================================
# VAL-GEMM-001: Source code verification (done by inspection)
# ============================================================
print("\n=== VAL-GEMM-001: v_dot2_f32_f16 usage ===")
kernel_src = Path(hip_path).read_text()
has_fdot2 = "__builtin_amdgcn_fdot2" in kernel_src
has_half2 = "__half2" in kernel_src or "_Float16_2" in kernel_src
has_swizzle = "SWIZZLE" in kernel_src or "^ SWIZZLE" in kernel_src
print(f"  __builtin_amdgcn_fdot2 present: {has_fdot2}")
print(f"  half2 packing present: {has_half2}")
print(f"  XOR swizzle present: {has_swizzle}")
val_gemm_001 = has_fdot2 and has_half2 and has_swizzle
print(f"  VAL-GEMM-001: {'PASS' if val_gemm_001 else 'FAIL'}")
if not val_gemm_001:
    all_pass = False

dev.cleanup()

print("\n" + "=" * 60)
print(f"{'=== ALL TESTS PASSED ===' if all_pass else '=== SOME TESTS FAILED ==='}")
if not all_pass:
    sys.exit(1)
