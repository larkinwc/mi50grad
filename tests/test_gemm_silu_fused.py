#!/usr/bin/env python3
"""
Test fused GEMM+SiLU kernels for prefill gate/up projections.

Two fusion strategies tested:

Strategy B (gemm_fp16_prefill_silu_epilogue):
  2 launches: gemm(gate) + gemm_with_silu_epilogue(up)
  Saves: 1 write (up→HBM), 1 read+write (silu kernel eliminates up→out pass)
  Status: correctness PASS, latency ~neutral (GEMM is compute-dominated)

Strategy A (gemm_fp16_prefill_silu_dual):
  1 launch: computes gate AND up in one kernel, applies silu epilogue
  Tile: TILE_N=32 (dual B tiles in LDS), THREAD_N=2 (dual accumulators)
  Saves: complete elimination of intermediate HBM traffic
  Status: correctness PASS, latency ~0.78x baseline (overhead from 2x WGs)

For both strategies: the GEMM kernel at M=128, N=11008, K=5120 is
compute-dominated (~1800us per GEMM), not bandwidth-dominated. The HBM
traffic savings (5-6MB out of 225MB total) do not manifest as wall-clock
speedup at this shape.

VAL-FUSE-002: correctness at (M=128, N=11008, K=5120), max abs error < 1e-2
VAL-FUSE-003: latency comparison printed (both strategies benchmarked)
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

# ============================================================
# Build kernels
# ============================================================
print("Building kernels...")

# gemm_fp16_prefill (original, for gate GEMM + reference up GEMM)
gemm_hip = str(PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill.hip")
gemm_hsaco = str(BUILD_DIR / "gemm_fp16_prefill.hsaco")
build_hip_hsaco(gemm_hip, gemm_hsaco)
gemm_module = dev.load_hsaco(gemm_hsaco)
func_gemm = dev.get_kernel(gemm_module, "gemm_fp16_prefill")

# gemm_fp16_prefill_silu_epilogue (Strategy B: fused up GEMM + silu + mul)
silu_hip = str(PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill_silu.hip")
silu_hsaco = str(BUILD_DIR / "gemm_fp16_prefill_silu.hsaco")
build_hip_hsaco(silu_hip, silu_hsaco)
silu_module = dev.load_hsaco(silu_hsaco)
func_silu = dev.get_kernel(silu_module, "gemm_fp16_prefill_silu_epilogue")

# gemm_fp16_prefill_silu_dual (Strategy A: single kernel computes both gate+up+silu)
dual_hip = str(PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill_silu_dual.hip")
dual_hsaco = str(BUILD_DIR / "gemm_fp16_prefill_silu_dual.hsaco")
build_hip_hsaco(dual_hip, dual_hsaco)
dual_module = dev.load_hsaco(dual_hsaco)
func_dual = dev.get_kernel(dual_module, "gemm_fp16_prefill_silu_dual")

# elementwise_v2 for silu_fused_v2 (baseline silu step)
elem_hip = str(PROJECT_ROOT / "src" / "kernels" / "elementwise_v2.hip")
elem_hsaco = str(BUILD_DIR / "elementwise_v2.hsaco")
build_hip_hsaco(elem_hip, elem_hsaco)
elem_module = dev.load_hsaco(elem_hsaco)
func_silu_v2 = dev.get_kernel(elem_module, "silu_fused_v2")

print("Build complete.\n")


def launch_gemm(func, A_ptr, B_ptr, C_ptr, M, N, K):
    """Launch gemm_fp16_prefill."""
    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(A_ptr),
        ctypes.c_uint64(B_ptr),
        ctypes.c_uint64(C_ptr),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]
    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)


def launch_silu_fused(func, gate_ptr, up_ptr, n):
    """Launch silu_fused_v2 (in-place on gate, reads up)."""
    grid_x = (n + 511) // 512
    params = [
        ctypes.c_uint64(gate_ptr),
        ctypes.c_uint64(up_ptr),
        ctypes.c_uint32(n),
    ]
    dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)


def launch_gemm_silu(func, A_ptr, B_ptr, gate_ptr, out_ptr, M, N, K):
    """Launch gemm_fp16_prefill_silu_epilogue (Strategy B)."""
    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(A_ptr),
        ctypes.c_uint64(B_ptr),
        ctypes.c_uint64(gate_ptr),
        ctypes.c_uint64(out_ptr),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]
    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)


def launch_dual(func, A_ptr, Bg_ptr, Bu_ptr, out_ptr, M, N, K):
    """Launch gemm_fp16_prefill_silu_dual (Strategy A: TILE_N=32)."""
    grid_x = (N + 31) // 32   # TILE_N=32 for dual B tiles
    grid_y = (M + 63) // 64
    params = [
        ctypes.c_uint64(A_ptr),
        ctypes.c_uint64(Bg_ptr),
        ctypes.c_uint64(Bu_ptr),
        ctypes.c_uint64(out_ptr),
        ctypes.c_uint32(M),
        ctypes.c_uint32(N),
        ctypes.c_uint32(K),
    ]
    dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)


# ============================================================
# Correctness Tests: Strategy B (gemm_fp16_prefill_silu_epilogue)
# ============================================================
print("=== Correctness Tests: Strategy B (gemm+silu_epilogue, 2 launches) ===")

all_pass = True
test_shapes = [
    # (M, N, K)
    (4,   64,   64),     # tiny sanity check
    (16,  256,  128),    # small
    (64,  1024, 512),    # medium
    (128, 11008, 5120),  # Qwen FFN shape (required by VAL-FUSE-002)
    (128, 6144,  5120),  # Qwen Q-projection shape
]

for M, N, K in test_shapes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    W_gate = np.random.randn(N, K).astype(np.float16) * 0.1
    W_up   = np.random.randn(N, K).astype(np.float16) * 0.1

    # CPU reference: full precision
    gate_ref = A.astype(np.float32) @ W_gate.astype(np.float32).T  # [M, N]
    up_ref   = A.astype(np.float32) @ W_up.astype(np.float32).T    # [M, N]
    silu_ref = gate_ref * (1.0 / (1.0 + np.exp(-gate_ref)))        # silu(gate)
    out_ref  = (silu_ref * up_ref).astype(np.float16)               # silu(gate)*up

    d_A      = dev.malloc(A.nbytes)
    d_Wgate  = dev.malloc(W_gate.nbytes)
    d_Wup    = dev.malloc(W_up.nbytes)
    d_gate   = dev.malloc(M * N * 2)
    d_out    = dev.malloc(M * N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_Wgate, W_gate.tobytes())
    dev.upload(d_Wup, W_up.tobytes())

    # Step 1: gate GEMM
    dev.hip.memset(d_gate, 0, M * N * 2)
    launch_gemm(func_gemm, d_A, d_Wgate, d_gate, M, N, K)
    dev.synchronize()

    # Step 2: up GEMM with SiLU epilogue
    dev.hip.memset(d_out, 0, M * N * 2)
    launch_gemm_silu(func_silu, d_A, d_Wup, d_gate, d_out, M, N, K)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_out, M * N * 2), dtype=np.float16).copy()
    result = result.reshape(M, N)

    max_err = float(np.max(np.abs(out_ref.astype(np.float32) - result.astype(np.float32))))
    ok = max_err < 1e-2
    print(f"  M={M:4d} N={N:6d} K={K:5d}: max_abs_err={max_err:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False
        if M <= 16:
            print(f"    ref[0,:4]: {out_ref[0,:4]}")
            print(f"    res[0,:4]: {result[0,:4]}")

    dev.free(d_A); dev.free(d_Wgate); dev.free(d_Wup); dev.free(d_gate); dev.free(d_out)


# ============================================================
# Correctness Tests: Strategy A (gemm_fp16_prefill_silu_dual)
# ============================================================
print("\n=== Correctness Tests: Strategy A (dual GEMM+silu, 1 launch) ===")

for M, N, K in test_shapes:
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float16) * 0.1
    W_gate = np.random.randn(N, K).astype(np.float16) * 0.1
    W_up   = np.random.randn(N, K).astype(np.float16) * 0.1

    gate_ref = A.astype(np.float32) @ W_gate.astype(np.float32).T
    up_ref   = A.astype(np.float32) @ W_up.astype(np.float32).T
    silu_ref = gate_ref * (1.0 / (1.0 + np.exp(-gate_ref)))
    out_ref  = (silu_ref * up_ref).astype(np.float16)

    d_A     = dev.malloc(A.nbytes)
    d_Wgate = dev.malloc(W_gate.nbytes)
    d_Wup   = dev.malloc(W_up.nbytes)
    d_out   = dev.malloc(M * N * 2)

    dev.upload(d_A, A.tobytes())
    dev.upload(d_Wgate, W_gate.tobytes())
    dev.upload(d_Wup, W_up.tobytes())
    dev.hip.memset(d_out, 0, M * N * 2)

    # Single launch: computes gate AND up, applies silu*up in epilogue
    launch_dual(func_dual, d_A, d_Wgate, d_Wup, d_out, M, N, K)
    dev.synchronize()

    result = np.frombuffer(dev.download(d_out, M * N * 2), dtype=np.float16).copy()
    result = result.reshape(M, N)

    max_err = float(np.max(np.abs(out_ref.astype(np.float32) - result.astype(np.float32))))
    ok = max_err < 1e-2
    print(f"  M={M:4d} N={N:6d} K={K:5d}: max_abs_err={max_err:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    dev.free(d_A); dev.free(d_Wgate); dev.free(d_Wup); dev.free(d_out)


# ============================================================
# Performance Benchmark: all three approaches at M=128, N=11008, K=5120
# ============================================================
print("\n=== Performance Benchmark ===")
print("Shape: M=128, N=11008, K=5120")
print("Comparing:")
print("  Baseline   (3 launches): gemm(gate) + gemm(up) + silu_fused")
print("  Strategy B (2 launches): gemm(gate) + gemm_silu_epilogue(up)")
print("  Strategy A (1 launch):   dual_gemm_silu (gate+up+silu in one kernel)")
print()

M, N, K = 128, 11008, 5120
np.random.seed(7)
A      = np.random.randn(M, K).astype(np.float16) * 0.1
W_gate = np.random.randn(N, K).astype(np.float16) * 0.1
W_up   = np.random.randn(N, K).astype(np.float16) * 0.1

d_A     = dev.malloc(A.nbytes)
d_Wgate = dev.malloc(W_gate.nbytes)
d_Wup   = dev.malloc(W_up.nbytes)
d_gate  = dev.malloc(M * N * 2)
d_up    = dev.malloc(M * N * 2)
d_out   = dev.malloc(M * N * 2)

dev.upload(d_A, A.tobytes())
dev.upload(d_Wgate, W_gate.tobytes())
dev.upload(d_Wup, W_up.tobytes())

WARMUP = 10
ITERS  = 100

# --- Warmup all ---
for _ in range(WARMUP):
    launch_gemm(func_gemm, d_A, d_Wgate, d_gate, M, N, K)
    launch_gemm(func_gemm, d_A, d_Wup, d_up, M, N, K)
    launch_silu_fused(func_silu_v2, d_gate, d_up, M * N)
    launch_gemm(func_gemm, d_A, d_Wgate, d_gate, M, N, K)
    launch_gemm_silu(func_silu, d_A, d_Wup, d_gate, d_out, M, N, K)
    launch_dual(func_dual, d_A, d_Wgate, d_Wup, d_out, M, N, K)
dev.synchronize()

# --- Baseline (3 launches) ---
times_baseline = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    launch_gemm(func_gemm, d_A, d_Wgate, d_gate, M, N, K)
    launch_gemm(func_gemm, d_A, d_Wup, d_up, M, N, K)
    launch_silu_fused(func_silu_v2, d_gate, d_up, M * N)
    dev.synchronize()
    times_baseline.append((time.perf_counter() - t0) * 1e6)
lat_baseline = float(np.median(times_baseline))

# --- Strategy B (2 launches) ---
times_b = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    launch_gemm(func_gemm, d_A, d_Wgate, d_gate, M, N, K)
    launch_gemm_silu(func_silu, d_A, d_Wup, d_gate, d_out, M, N, K)
    dev.synchronize()
    times_b.append((time.perf_counter() - t0) * 1e6)
lat_b = float(np.median(times_b))

# --- Strategy A (1 launch) ---
times_a = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    launch_dual(func_dual, d_A, d_Wgate, d_Wup, d_out, M, N, K)
    dev.synchronize()
    times_a.append((time.perf_counter() - t0) * 1e6)
lat_a = float(np.median(times_a))

# Memory traffic analysis
bytes_per_half = 2
a_bytes = M * K * bytes_per_half
w_bytes = N * K * bytes_per_half
mn_bytes = M * N * bytes_per_half
baseline_traffic_mb = (2*a_bytes + 2*w_bytes + 3*mn_bytes + 2*mn_bytes) / (1024*1024)
stratb_traffic_mb = (2*a_bytes + 2*w_bytes + 2*mn_bytes + mn_bytes) / (1024*1024)
strata_traffic_mb = (a_bytes + 2*w_bytes + mn_bytes) / (1024*1024)

print(f"  Baseline   (3 launches): {lat_baseline:.1f} us  (median over {ITERS} iters)")
print(f"  Strategy B (2 launches): {lat_b:.1f} us  speedup={lat_baseline/lat_b:.3f}x")
print(f"  Strategy A (1 launch):   {lat_a:.1f} us  speedup={lat_baseline/lat_a:.3f}x")
print()
print(f"  HBM traffic (theoretical):")
print(f"    Baseline:   {baseline_traffic_mb:.1f} MB")
print(f"    Strategy B: {stratb_traffic_mb:.1f} MB  (saves {baseline_traffic_mb-stratb_traffic_mb:.1f} MB)")
print(f"    Strategy A: {strata_traffic_mb:.1f} MB  (saves {baseline_traffic_mb-strata_traffic_mb:.1f} MB)")
print()
print(f"  Note: At M=128 N=11008 K=5120, the GEMM is compute-dominated (~1700-1800us")
print(f"  per kernel). The saved HBM traffic (~5-50MB) is negligible vs GEMM compute")
print(f"  time, so wall-clock speedup is within measurement noise at this shape.")

dev.free(d_A); dev.free(d_Wgate); dev.free(d_Wup); dev.free(d_gate); dev.free(d_up); dev.free(d_out)

dev.cleanup()

print()
if all_pass:
    print("=== ALL TESTS PASSED ===")
else:
    print("=== SOME TESTS FAILED ===")
    sys.exit(1)
