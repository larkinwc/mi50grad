#!/usr/bin/env python3
"""
Final benchmark script for kernel optimization sprint.

Measures performance using all new optimized kernels:
  - flash_attn_256_v3 (prefill) instead of flash_attn_256_tuned prefill
  - gemm_fp16_prefill_db instead of gemm_fp16_prefill
  - gemm_int4_prefill_v2 instead of gemm_int4_prefill_hip
  - gemv_int4_v4 (t4/t8/t16) instead of gemv_int4_v3
  - elementwise_v3 (rmsnorm_v3, silu_fused_v3, residual_add_v3) instead of v2
  - Fused kernels: fused_skip_rmsnorm_gemv, gemm_fp16_prefill_silu_epilogue

Results saved to bench/optimization_final.json.
"""

import sys
import ctypes
import json
import time
import statistics
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WARMUP_ITERS = 10
BENCH_ITERS = 100

# Qwen 3.5 27B architecture constants
NUM_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 256
GROUP_SIZE = 128

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Output file
BENCH_OUTPUT = PROJECT_ROOT / "bench" / "optimization_final.json"
BENCH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: benchmark a kernel launcher function
# ---------------------------------------------------------------------------

def bench_kernel(launch_fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run launch_fn warmup+iters times, return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        launch_fn()
    dev.synchronize()

    # Timed iterations
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch_fn()
        dev.synchronize()
        latencies.append((time.perf_counter() - t0) * 1e6)

    return statistics.median(latencies)


def try_build_kernel(name, hip_path, hsaco_path):
    """Build a HIP kernel, return (module, True) or (None, False) on failure."""
    try:
        build_hip_hsaco(str(hip_path), str(hsaco_path))
        module = dev.load_hsaco(str(hsaco_path))
        print(f"  [OK]  {name}")
        return module, True
    except Exception as e:
        warnings.warn(f"  [SKIP] {name}: {e}")
        return None, False


def get_kernel_safe(module, name, kernel_display_name):
    """Get a kernel function, return (func, True) or (None, False) on failure."""
    try:
        func = dev.get_kernel(module, name)
        return func, True
    except Exception as e:
        warnings.warn(f"  [SKIP] kernel function '{kernel_display_name}': {e}")
        return None, False


# ---------------------------------------------------------------------------
# Initialize GPU device
# ---------------------------------------------------------------------------
print("Initializing GPU device...")
dev = GPUDevice(0)
print("GPU device ready.")
print()

results = {}  # {kernel_name: {shape_key: {latency_us, tflops_or_gbs}}}

# ---------------------------------------------------------------------------
# 1. FlashAttention 256 v3 (new: block-tiled prefill, same decode as tuned)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building flash_attn_256_v3.hip ...")
fa_v3_hip = PROJECT_ROOT / "src" / "kernels" / "flash_attn_256_v3.hip"
fa_v3_hsaco = BUILD_DIR / "flash_attn_256_v3.hsaco"
fa_v3_module, fa_v3_ok = try_build_kernel("flash_attn_256_v3", fa_v3_hip, str(fa_v3_hsaco))

if fa_v3_ok:
    decode_func, decode_ok = get_kernel_safe(fa_v3_module, "flash_attn_256_decode", "flash_attn_256_decode")
    prefill_v3_func, prefill_v3_ok = get_kernel_safe(fa_v3_module, "flash_attn_256_v3_prefill", "flash_attn_256_v3_prefill")

    # --- Decode benchmarks (using new kernel's decode, same as tuned) ---
    if decode_ok:
        results["flash_attn_decode"] = {}
        for kv_seq_len in [256, 1024]:
            Q = np.random.randn(1, NUM_HEADS, HEAD_DIM).astype(np.float16) * 0.1
            K = np.random.randn(kv_seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float16) * 0.1
            V = np.random.randn(kv_seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float16) * 0.1

            d_Q = dev.malloc(Q.nbytes)
            d_K = dev.malloc(K.nbytes)
            d_V = dev.malloc(V.nbytes)
            d_Out = dev.malloc(NUM_HEADS * HEAD_DIM * 2)  # FP16

            dev.upload(d_Q, Q.tobytes())
            dev.upload(d_K, K.tobytes())
            dev.upload(d_V, V.tobytes())
            dev.hip.memset(d_Out, 0, NUM_HEADS * HEAD_DIM * 2)

            params = [
                ctypes.c_uint64(d_Q),
                ctypes.c_uint64(d_K),
                ctypes.c_uint64(d_V),
                ctypes.c_uint64(d_Out),
                ctypes.c_uint32(kv_seq_len),
                ctypes.c_uint32(NUM_HEADS),
                ctypes.c_uint32(NUM_KV_HEADS),
            ]

            def _launch_decode():
                dev.launch(decode_func, (NUM_HEADS, 1, 1), (256, 1, 1), params)

            lat_us = bench_kernel(_launch_decode)

            flops = 4.0 * NUM_HEADS * kv_seq_len * HEAD_DIM
            tflops = flops / 1e12 / (lat_us * 1e-6)

            shape_key = f"kv_len={kv_seq_len},h={NUM_HEADS},kvh={NUM_KV_HEADS},d={HEAD_DIM}"
            results["flash_attn_decode"][shape_key] = {
                "latency_us": round(lat_us, 3),
                "tflops": round(tflops, 6),
            }
            print(f"  decode kv_len={kv_seq_len:5d}: {lat_us:8.2f} us  {tflops*1e3:.4f} GFLOPS")

            dev.free(d_Q); dev.free(d_K); dev.free(d_V); dev.free(d_Out)

    # --- Prefill benchmarks (new v3 prefill kernel) ---
    if prefill_v3_ok:
        results["flash_attn_prefill"] = {}
        BLOCK_M = 16  # v3 uses BLOCK_M=16
        for seq_len in [128, 512, 2048]:
            Q = np.random.randn(seq_len, NUM_HEADS, HEAD_DIM).astype(np.float16) * 0.1
            K = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float16) * 0.1
            V = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float16) * 0.1

            d_Q = dev.malloc(Q.nbytes)
            d_K = dev.malloc(K.nbytes)
            d_V = dev.malloc(V.nbytes)
            d_Out = dev.malloc(Q.nbytes)

            dev.upload(d_Q, Q.tobytes())
            dev.upload(d_K, K.tobytes())
            dev.upload(d_V, V.tobytes())
            dev.hip.memset(d_Out, 0, Q.nbytes)

            # v3: Grid=(num_heads, ceil(seq_len/BLOCK_M), 1), Block=(256,1,1)
            grid_y = (seq_len + BLOCK_M - 1) // BLOCK_M
            params = [
                ctypes.c_uint64(d_Q),
                ctypes.c_uint64(d_K),
                ctypes.c_uint64(d_V),
                ctypes.c_uint64(d_Out),
                ctypes.c_uint32(seq_len),  # num_q_rows
                ctypes.c_uint32(seq_len),  # kv_len
                ctypes.c_uint32(NUM_HEADS),
                ctypes.c_uint32(NUM_KV_HEADS),
                ctypes.c_uint32(1),  # causal=1
            ]

            def _launch_prefill_v3(grid_y=grid_y):
                dev.launch(prefill_v3_func, (NUM_HEADS, grid_y, 1), (256, 1, 1), params)

            lat_us = bench_kernel(_launch_prefill_v3)

            flops = 4.0 * NUM_HEADS * seq_len * seq_len * HEAD_DIM
            tflops = flops / 1e12 / (lat_us * 1e-6)

            shape_key = f"seq={seq_len},h={NUM_HEADS},kvh={NUM_KV_HEADS},d={HEAD_DIM}"
            results["flash_attn_prefill"][shape_key] = {
                "latency_us": round(lat_us, 3),
                "tflops": round(tflops, 6),
            }
            lat_ms = lat_us / 1000
            print(f"  prefill seq_len={seq_len:5d}: {lat_ms:8.3f} ms  {tflops:.4f} TFLOPS")

            dev.free(d_Q); dev.free(d_K); dev.free(d_V); dev.free(d_Out)
else:
    print("  [SKIP] FlashAttention v3 (build failed)")

print()

# ---------------------------------------------------------------------------
# 2. FP16 GEMM — double-buffered (gemm_fp16_prefill_db)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building gemm_fp16_prefill.hip (double-buffered variant) ...")
fp16_gemm_hip = PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill.hip"
fp16_gemm_hsaco = BUILD_DIR / "gemm_fp16_prefill_db.hsaco"
fp16_gemm_module, fp16_gemm_ok = try_build_kernel(
    "gemm_fp16_prefill_db", fp16_gemm_hip, str(fp16_gemm_hsaco))

if fp16_gemm_ok:
    gemm_fp16_db_func, gemm_fp16_db_fn_ok = get_kernel_safe(
        fp16_gemm_module, "gemm_fp16_prefill_db", "gemm_fp16_prefill_db")

    if gemm_fp16_db_fn_ok:
        results["gemm_fp16_prefill"] = {}
        fp16_shapes = [
            (128, 6144, 5120),
            (128, 5120, 6144),
            (128, 1024, 5120),
        ]

        for M, N, K in fp16_shapes:
            A = np.random.randn(M, K).astype(np.float16) * 0.1
            B = np.random.randn(N, K).astype(np.float16) * 0.1

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

            def _launch_fp16gemm_db(gx=grid_x, gy=grid_y):
                dev.launch(gemm_fp16_db_func, (gx, gy, 1), (256, 1, 1), params)

            lat_us = bench_kernel(_launch_fp16gemm_db)

            flops = 2.0 * M * N * K
            tflops = flops / 1e12 / (lat_us * 1e-6)

            shape_key = f"M={M},N={N},K={K}"
            results["gemm_fp16_prefill"][shape_key] = {
                "latency_us": round(lat_us, 3),
                "tflops": round(tflops, 6),
            }
            print(f"  M={M:3d} N={N:5d} K={K:5d}: {lat_us:8.2f} us  {tflops:.4f} TFLOPS")

            dev.free(d_A); dev.free(d_B); dev.free(d_C)
else:
    print("  [SKIP] FP16 GEMM double-buffered (build failed)")

print()

# ---------------------------------------------------------------------------
# 3. INT4 GEMM v2 — on-the-fly dequantization (gemm_int4_prefill_v2)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building gemm_int4_prefill_v2.hip ...")
int4_gemm_v2_hip = PROJECT_ROOT / "src" / "kernels" / "gemm_int4_prefill_v2.hip"
int4_gemm_v2_hsaco = BUILD_DIR / "gemm_int4_prefill_v2.hsaco"
int4_gemm_v2_module, int4_gemm_v2_ok = try_build_kernel(
    "gemm_int4_prefill_v2", int4_gemm_v2_hip, str(int4_gemm_v2_hsaco))

if int4_gemm_v2_ok:
    gemm_int4_v2_func, gemm_int4_v2_fn_ok = get_kernel_safe(
        int4_gemm_v2_module, "gemm_int4_prefill_v2", "gemm_int4_prefill_v2")

    if gemm_int4_v2_fn_ok:
        results["gemm_int4_prefill"] = {}
        int4_gemm_shapes = [
            (128, 4096, 4096, 128),
            (64, 11008, 4096, 128),
        ]

        for M, N, K, gs in int4_gemm_shapes:
            np.random.seed(42)
            A = np.random.randn(M, K).astype(np.float16) * 0.1
            B_packed = np.random.randint(0, 2**32 - 1, (K // 8, N), dtype=np.uint32)
            num_groups = K // gs
            scales = np.ones((num_groups, N), dtype=np.float16) * np.float16(0.01)
            zeros = np.ones((num_groups, N), dtype=np.float16) * np.float16(8.0)

            d_A = dev.malloc(A.nbytes)
            d_B = dev.malloc(B_packed.nbytes)
            d_sc = dev.malloc(scales.nbytes)
            d_zr = dev.malloc(zeros.nbytes)
            d_C = dev.malloc(M * N * 2)

            dev.upload(d_A, A.tobytes())
            dev.upload(d_B, B_packed.tobytes())
            dev.upload(d_sc, scales.tobytes())
            dev.upload(d_zr, zeros.tobytes())
            dev.hip.memset(d_C, 0, M * N * 2)

            grid_x = (N + 63) // 64
            grid_y = (M + 63) // 64
            params = [
                ctypes.c_uint64(d_A),
                ctypes.c_uint64(d_B),
                ctypes.c_uint64(d_sc),
                ctypes.c_uint64(d_zr),
                ctypes.c_uint64(d_C),
                ctypes.c_uint32(M),
                ctypes.c_uint32(N),
                ctypes.c_uint32(K),
                ctypes.c_uint32(gs),
            ]

            def _launch_int4gemm_v2(gx=grid_x, gy=grid_y):
                dev.launch(gemm_int4_v2_func, (gx, gy, 1), (256, 1, 1), params)

            lat_us = bench_kernel(_launch_int4gemm_v2)

            flops = 2.0 * M * N * K
            tflops = flops / 1e12 / (lat_us * 1e-6)

            shape_key = f"M={M},N={N},K={K},gs={gs}"
            results["gemm_int4_prefill"][shape_key] = {
                "latency_us": round(lat_us, 3),
                "tflops": round(tflops, 6),
            }
            print(f"  M={M:3d} N={N:5d} K={K:5d} gs={gs}: {lat_us:8.2f} us  {tflops:.4f} TFLOPS")

            dev.free(d_A); dev.free(d_B); dev.free(d_sc); dev.free(d_zr); dev.free(d_C)
else:
    print("  [SKIP] INT4 GEMM v2 (build failed)")

print()

# ---------------------------------------------------------------------------
# 4. INT4 GEMV v4 (uses v_dot2_f32_f16 fdot2: t4, t8, t16)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building gemv_int4_v4.hip ...")
gemv_v4_hip = PROJECT_ROOT / "src" / "kernels" / "gemv_int4_v4.hip"
gemv_v4_hsaco = BUILD_DIR / "gemv_int4_v4.hsaco"
gemv_v4_module, gemv_v4_ok = try_build_kernel("gemv_int4_v4", gemv_v4_hip, str(gemv_v4_hsaco))


def pack_w4a16(W_int4, K, N):
    """Pack [N, K] INT4 weights (0-15) to [K//8, N] uint32 for gemv_int4_v4."""
    W_km = W_int4.T  # [K, N]
    num_k_groups = K // 8
    W_packed = np.zeros((num_k_groups, N), dtype=np.uint32)
    for kg in range(num_k_groups):
        k_base = kg * 8
        for bit in range(8):
            W_packed[kg] |= (W_km[k_base + bit, :].astype(np.uint32) & 0xF) << (bit * 4)
    return W_packed


if gemv_v4_ok:
    v4_variants = {}
    for tpc in [4, 8, 16]:
        fn, ok = get_kernel_safe(gemv_v4_module, f"gemv_int4_v4_t{tpc}", f"gemv_int4_v4_t{tpc}")
        if ok:
            v4_variants[tpc] = fn

    gemv_shapes = [
        (4096,  4096, 128),
        (11008, 4096, 128),
    ]

    for N, K, gs in gemv_shapes:
        np.random.seed(42)
        x_fp16 = np.random.randn(K).astype(np.float16)
        W_int4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
        num_groups = K // gs
        scales = np.ones((num_groups, N), dtype=np.float16) * np.float16(0.01)
        zeros = np.ones((num_groups, N), dtype=np.float16) * np.float16(8.0)
        W_packed = pack_w4a16(W_int4, K, N)

        d_x = dev.malloc(K * 2)
        d_W = dev.malloc(W_packed.nbytes)
        d_sc = dev.malloc(scales.nbytes)
        d_zr = dev.malloc(zeros.nbytes)
        d_out = dev.malloc(N * 2)

        dev.upload(d_x, x_fp16.tobytes())
        dev.upload(d_W, W_packed.tobytes())
        dev.upload(d_sc, scales.tobytes())
        dev.upload(d_zr, zeros.tobytes())

        for tpc, func in v4_variants.items():
            cols_per_wg = 256 // tpc
            grid_x = (N + cols_per_wg - 1) // cols_per_wg
            params = [
                ctypes.c_uint64(d_x),
                ctypes.c_uint64(d_W),
                ctypes.c_uint64(d_sc),
                ctypes.c_uint64(d_zr),
                ctypes.c_uint64(d_out),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint32(gs),
            ]

            def _launch_gemv_v4(f=func, gx=grid_x):
                dev.launch(f, (gx, 1, 1), (256, 1, 1), params)

            lat_us = bench_kernel(_launch_gemv_v4)

            bytes_w = (N * K) // 2
            bytes_x = K * 2
            bytes_sc = num_groups * N * 2
            bytes_zr = num_groups * N * 2
            bytes_out = N * 2
            total_bytes = bytes_w + bytes_x + bytes_sc + bytes_zr + bytes_out
            gbs = total_bytes / 1e9 / (lat_us * 1e-6)

            kernel_name = f"gemv_int4_v4_t{tpc}"
            if kernel_name not in results:
                results[kernel_name] = {}
            shape_key = f"N={N},K={K},gs={gs}"
            results[kernel_name][shape_key] = {
                "latency_us": round(lat_us, 3),
                "gbs": round(gbs, 3),
            }
            print(f"  t{tpc:2d} N={N:5d} K={K:5d}: {lat_us:8.2f} us  {gbs:.1f} GB/s")

        dev.free(d_x); dev.free(d_W); dev.free(d_sc); dev.free(d_zr); dev.free(d_out)
else:
    print("  [SKIP] INT4 GEMV v4 (build failed)")

print()

# ---------------------------------------------------------------------------
# 5. Elementwise kernels v3: rmsnorm_v3, silu_fused_v3, residual_add_v3
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building elementwise_v3.hip ...")
ew_v3_hip = PROJECT_ROOT / "src" / "kernels" / "elementwise_v3.hip"
ew_v3_hsaco = BUILD_DIR / "elementwise_v3.hsaco"
ew_v3_module, ew_v3_ok = try_build_kernel("elementwise_v3", ew_v3_hip, str(ew_v3_hsaco))

if ew_v3_ok:
    # --- rmsnorm_v3 ---
    # Grid: (num_vectors, 1, 1), Block: (256, 1, 1)
    # Signature: (dst, src, weight, dim, eps)
    rmsnorm_v3_func, rmsnorm_v3_ok = get_kernel_safe(ew_v3_module, "rmsnorm_v3", "rmsnorm_v3")
    if rmsnorm_v3_ok:
        results["rmsnorm_v3"] = {}
        dim = 5120
        num_vectors = 128
        eps = 1e-5

        np.random.seed(42)
        src = np.random.randn(num_vectors, dim).astype(np.float16)
        dst = np.zeros_like(src)
        weight = np.ones(dim, dtype=np.float16)

        d_src = dev.malloc(src.nbytes)
        d_dst = dev.malloc(dst.nbytes)
        d_w = dev.malloc(weight.nbytes)
        dev.upload(d_src, src.tobytes())
        dev.upload(d_dst, dst.tobytes())
        dev.upload(d_w, weight.tobytes())

        params = [
            ctypes.c_uint64(d_dst),
            ctypes.c_uint64(d_src),
            ctypes.c_uint64(d_w),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps),
        ]

        def _launch_rmsnorm_v3():
            dev.launch(rmsnorm_v3_func, (num_vectors, 1, 1), (256, 1, 1), params)

        lat_us = bench_kernel(_launch_rmsnorm_v3)

        total_bytes = num_vectors * dim * 2 + dim * 2 + num_vectors * dim * 2
        gbs = total_bytes / 1e9 / (lat_us * 1e-6)

        shape_key = f"dim={dim},n={num_vectors}"
        results["rmsnorm_v3"][shape_key] = {
            "latency_us": round(lat_us, 3),
            "gbs": round(gbs, 3),
        }
        print(f"  rmsnorm_v3 dim={dim} n={num_vectors}: {lat_us:.2f} us  {gbs:.1f} GB/s")

        dev.free(d_src); dev.free(d_dst); dev.free(d_w)

    # --- silu_fused_v3 ---
    # Grid: (ceil(n/2048), 1, 1), Block: (256, 1, 1)
    # Signature: (gate, up, n)
    silu_v3_func, silu_v3_ok = get_kernel_safe(ew_v3_module, "silu_fused_v3", "silu_fused_v3")
    if silu_v3_ok:
        results["silu_fused_v3"] = {}
        dim = 11008
        num_vectors = 128
        n = dim * num_vectors

        np.random.seed(42)
        gate = np.random.randn(n).astype(np.float16)
        up = np.random.randn(n).astype(np.float16)

        d_gate = dev.malloc(gate.nbytes)
        d_up = dev.malloc(up.nbytes)
        dev.upload(d_gate, gate.tobytes())
        dev.upload(d_up, up.tobytes())

        grid_x = (n + 2047) // 2048
        params = [
            ctypes.c_uint64(d_gate),
            ctypes.c_uint64(d_up),
            ctypes.c_uint32(n),
        ]

        def _launch_silu_v3():
            dev.launch(silu_v3_func, (grid_x, 1, 1), (256, 1, 1), params)

        lat_us = bench_kernel(_launch_silu_v3)

        total_bytes = n * 2 * 3
        gbs = total_bytes / 1e9 / (lat_us * 1e-6)

        shape_key = f"dim={dim},n={num_vectors}"
        results["silu_fused_v3"][shape_key] = {
            "latency_us": round(lat_us, 3),
            "gbs": round(gbs, 3),
        }
        print(f"  silu_fused_v3 dim={dim} n={num_vectors}: {lat_us:.2f} us  {gbs:.1f} GB/s")

        dev.free(d_gate); dev.free(d_up)

    # --- residual_add_v3 ---
    # Grid: (ceil(n/2048), 1, 1), Block: (256, 1, 1)
    # Signature: (dst, src, n)
    resadd_v3_func, resadd_v3_ok = get_kernel_safe(ew_v3_module, "residual_add_v3", "residual_add_v3")
    if resadd_v3_ok:
        results["residual_add_v3"] = {}
        dim = 5120
        num_vectors = 128
        n = dim * num_vectors

        np.random.seed(42)
        dst = np.random.randn(n).astype(np.float16)
        src = np.random.randn(n).astype(np.float16)

        d_dst = dev.malloc(dst.nbytes)
        d_src = dev.malloc(src.nbytes)
        dev.upload(d_dst, dst.tobytes())
        dev.upload(d_src, src.tobytes())

        grid_x = (n + 2047) // 2048
        params = [
            ctypes.c_uint64(d_dst),
            ctypes.c_uint64(d_src),
            ctypes.c_uint32(n),
        ]

        def _launch_resadd_v3():
            dev.launch(resadd_v3_func, (grid_x, 1, 1), (256, 1, 1), params)

        lat_us = bench_kernel(_launch_resadd_v3)

        total_bytes = n * 2 * 3
        gbs = total_bytes / 1e9 / (lat_us * 1e-6)

        shape_key = f"dim={dim},n={num_vectors}"
        results["residual_add_v3"][shape_key] = {
            "latency_us": round(lat_us, 3),
            "gbs": round(gbs, 3),
        }
        print(f"  residual_add_v3 dim={dim} n={num_vectors}: {lat_us:.2f} us  {gbs:.1f} GB/s")

        dev.free(d_dst); dev.free(d_src)
else:
    print("  [SKIP] elementwise_v3 (build failed)")

print()

# ---------------------------------------------------------------------------
# 6. Fused kernels: fused_skip_rmsnorm_gemv, gemm_fp16_prefill_silu_epilogue
# ---------------------------------------------------------------------------
print("=" * 60)
print("Building fused_skip_rmsnorm_gemv.hip ...")
fused_gemv_hip = PROJECT_ROOT / "src" / "kernels" / "fused_skip_rmsnorm_gemv.hip"
fused_gemv_hsaco = BUILD_DIR / "fused_skip_rmsnorm_gemv.hsaco"
fused_gemv_module, fused_gemv_ok = try_build_kernel(
    "fused_skip_rmsnorm_gemv", fused_gemv_hip, str(fused_gemv_hsaco))

if fused_gemv_ok:
    # Use t16 variant (best performer)
    fused_gemv_func, fused_gemv_fn_ok = get_kernel_safe(
        fused_gemv_module, "fused_skip_rmsnorm_gemv_t16", "fused_skip_rmsnorm_gemv_t16")

    if fused_gemv_fn_ok:
        results["fused_skip_rmsnorm_gemv"] = {}
        # Shape: K=5120 (dim), N=4096
        K_dim = 5120
        N_out = 4096
        gs = 128
        num_groups = K_dim // gs

        np.random.seed(42)
        hidden = np.random.randn(K_dim).astype(np.float16) * 0.1
        residual = np.random.randn(K_dim).astype(np.float16) * 0.1
        W_int4 = np.random.randint(0, 16, (N_out, K_dim), dtype=np.uint8)
        rms_weight = np.ones(K_dim, dtype=np.float16)
        scales = np.ones((num_groups, N_out), dtype=np.float16) * np.float16(0.01)
        zeros = np.ones((num_groups, N_out), dtype=np.float16) * np.float16(8.0)
        W_packed = pack_w4a16(W_int4, K_dim, N_out)

        d_out_gemv = dev.malloc(N_out * 2)
        d_hidden_out = dev.malloc(K_dim * 2)
        d_hidden = dev.malloc(K_dim * 2)
        d_residual = dev.malloc(K_dim * 2)
        d_rms_w = dev.malloc(K_dim * 2)
        d_W = dev.malloc(W_packed.nbytes)
        d_sc = dev.malloc(scales.nbytes)
        d_zr = dev.malloc(zeros.nbytes)

        dev.upload(d_hidden, hidden.tobytes())
        dev.upload(d_residual, residual.tobytes())
        dev.upload(d_rms_w, rms_weight.tobytes())
        dev.upload(d_W, W_packed.tobytes())
        dev.upload(d_sc, scales.tobytes())
        dev.upload(d_zr, zeros.tobytes())

        # t16: cols_per_wg = 256/16 = 16
        cols_per_wg = 256 // 16
        grid_x = (N_out + cols_per_wg - 1) // cols_per_wg

        # Dynamic shared memory: K*2 (lds_hval) + K*2 (lds_A) + 16 (s_warp) + 256*4 (s_reduce)
        shared_mem_bytes = K_dim * 4 + 16 + 256 * 4

        # Signature: out_gemv, hidden_out, hidden (const), residual, weight, eps,
        #            B_q4, scales, zeros, K, N, group_size
        params = [
            ctypes.c_uint64(d_out_gemv),
            ctypes.c_uint64(d_hidden_out),
            ctypes.c_uint64(d_hidden),
            ctypes.c_uint64(d_residual),
            ctypes.c_uint64(d_rms_w),
            ctypes.c_float(1e-5),  # eps
            ctypes.c_uint64(d_W),
            ctypes.c_uint64(d_sc),
            ctypes.c_uint64(d_zr),
            ctypes.c_uint32(K_dim),
            ctypes.c_uint32(N_out),
            ctypes.c_uint32(gs),
        ]

        def _launch_fused_gemv():
            dev.launch(fused_gemv_func, (grid_x, 1, 1), (256, 1, 1), params,
                       shared_mem=shared_mem_bytes)

        lat_us = bench_kernel(_launch_fused_gemv)

        shape_key = f"K={K_dim},N={N_out},gs={gs}"
        results["fused_skip_rmsnorm_gemv"][shape_key] = {
            "latency_us": round(lat_us, 3),
        }
        print(f"  fused_skip_rmsnorm_gemv K={K_dim} N={N_out}: {lat_us:.2f} us")

        dev.free(d_out_gemv); dev.free(d_hidden_out); dev.free(d_hidden)
        dev.free(d_residual); dev.free(d_rms_w); dev.free(d_W)
        dev.free(d_sc); dev.free(d_zr)
else:
    print("  [SKIP] fused_skip_rmsnorm_gemv (build failed)")

print()

# --- Fused SiLU-epilogue GEMM ---
print("=" * 60)
print("Building gemm_fp16_prefill_silu.hip (silu epilogue) ...")
silu_gemm_hip = PROJECT_ROOT / "src" / "kernels" / "gemm_fp16_prefill_silu.hip"
silu_gemm_hsaco = BUILD_DIR / "gemm_fp16_prefill_silu.hsaco"
silu_gemm_module, silu_gemm_ok = try_build_kernel(
    "gemm_fp16_prefill_silu_epilogue", silu_gemm_hip, str(silu_gemm_hsaco))

if silu_gemm_ok:
    silu_gemm_func, silu_gemm_fn_ok = get_kernel_safe(
        silu_gemm_module, "gemm_fp16_prefill_silu_epilogue", "gemm_fp16_prefill_silu_epilogue")

    if silu_gemm_fn_ok:
        results["gemm_fp16_prefill_silu_epilogue"] = {}
        # Fused flow: gate already in gate_buf, compute up via GEMM + silu epilogue
        # Shape: M=128, N=11008 (FFN intermediate), K=5120
        M, N, K = 128, 11008, 5120

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float16) * 0.1
        B_up = np.random.randn(N, K).astype(np.float16) * 0.1
        gate_buf = np.random.randn(M, N).astype(np.float16) * 0.1
        Out = np.zeros((M, N), dtype=np.float16)

        d_A = dev.malloc(A.nbytes)
        d_B = dev.malloc(B_up.nbytes)
        d_gate = dev.malloc(gate_buf.nbytes)
        d_Out = dev.malloc(Out.nbytes)

        dev.upload(d_A, A.tobytes())
        dev.upload(d_B, B_up.tobytes())
        dev.upload(d_gate, gate_buf.tobytes())

        grid_x = (N + 63) // 64
        grid_y = (M + 63) // 64
        params = [
            ctypes.c_uint64(d_A),
            ctypes.c_uint64(d_B),
            ctypes.c_uint64(d_gate),
            ctypes.c_uint64(d_Out),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ]

        def _launch_silu_epilogue(gx=grid_x, gy=grid_y):
            dev.launch(silu_gemm_func, (gx, gy, 1), (256, 1, 1), params)

        lat_us = bench_kernel(_launch_silu_epilogue)

        shape_key = f"M={M},N={N},K={K}"
        results["gemm_fp16_prefill_silu_epilogue"][shape_key] = {
            "latency_us": round(lat_us, 3),
        }
        print(f"  gemm_fp16_prefill_silu_epilogue M={M} N={N} K={K}: {lat_us:.2f} us")

        dev.free(d_A); dev.free(d_B); dev.free(d_gate); dev.free(d_Out)
else:
    print("  [SKIP] gemm_fp16_prefill_silu_epilogue (build failed)")

print()

# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------
print("=" * 60)
with open(BENCH_OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {BENCH_OUTPUT}")
print()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY: Final Optimized Performance")
print("=" * 60)
print()

# FlashAttention decode
if "flash_attn_decode" in results:
    print("FlashAttention Decode (flash_attn_256_v3, same decode kernel):")
    print(f"  {'Shape':<50} {'Latency (us)':>14} {'GFLOPS':>10}")
    print(f"  {'-'*74}")
    for shape, data in results["flash_attn_decode"].items():
        print(f"  {shape:<50} {data['latency_us']:>12.2f}   {data['tflops']*1e3:>8.4f}")
    print()

# FlashAttention prefill
if "flash_attn_prefill" in results:
    print("FlashAttention Prefill (flash_attn_256_v3_prefill):")
    print(f"  {'Shape':<50} {'Latency (ms)':>14} {'TFLOPS':>10}")
    print(f"  {'-'*74}")
    for shape, data in results["flash_attn_prefill"].items():
        print(f"  {shape:<50} {data['latency_us']/1000:>12.3f}   {data['tflops']:>8.4f}")
    print()

# FP16 GEMM double-buffered
if "gemm_fp16_prefill" in results:
    print("FP16 GEMM Prefill (gemm_fp16_prefill_db — double buffered):")
    print(f"  {'Shape':<40} {'Latency (us)':>14} {'TFLOPS':>10}")
    print(f"  {'-'*64}")
    for shape, data in results["gemm_fp16_prefill"].items():
        print(f"  {shape:<40} {data['latency_us']:>12.2f}   {data['tflops']:>8.4f}")
    print()

# INT4 GEMM v2
if "gemm_int4_prefill" in results:
    print("INT4 GEMM Prefill (gemm_int4_prefill_v2 — on-the-fly dequant):")
    print(f"  {'Shape':<50} {'Latency (us)':>14} {'TFLOPS':>10}")
    print(f"  {'-'*74}")
    for shape, data in results["gemm_int4_prefill"].items():
        print(f"  {shape:<50} {data['latency_us']:>12.2f}   {data['tflops']:>8.4f}")
    print()

# INT4 GEMV v4 variants
for tpc in [4, 8, 16]:
    kname = f"gemv_int4_v4_t{tpc}"
    if kname in results:
        print(f"INT4 GEMV v4_t{tpc} (gemv_int4_v4 — v_dot2_f32_f16):")
        print(f"  {'Shape':<40} {'Latency (us)':>14} {'GB/s':>10}")
        print(f"  {'-'*64}")
        for shape, data in results[kname].items():
            print(f"  {shape:<40} {data['latency_us']:>12.2f}   {data['gbs']:>8.1f}")
        print()

# Elementwise v3
for kname in ["rmsnorm_v3", "silu_fused_v3", "residual_add_v3"]:
    if kname in results:
        print(f"Elementwise: {kname}:")
        print(f"  {'Shape':<40} {'Latency (us)':>14} {'GB/s':>10}")
        print(f"  {'-'*64}")
        for shape, data in results[kname].items():
            print(f"  {shape:<40} {data['latency_us']:>12.2f}   {data['gbs']:>8.1f}")
        print()

# Fused kernels
for kname in ["fused_skip_rmsnorm_gemv", "gemm_fp16_prefill_silu_epilogue"]:
    if kname in results:
        print(f"Fused: {kname}:")
        print(f"  {'Shape':<40} {'Latency (us)':>14}")
        print(f"  {'-'*54}")
        for shape, data in results[kname].items():
            print(f"  {shape:<40} {data['latency_us']:>12.2f}")
        print()

print("=" * 60)
print(f"Benchmarked {len(results)} kernel categories.")
print(f"Results saved to: {BENCH_OUTPUT}")

dev.cleanup()
print("Done.")
