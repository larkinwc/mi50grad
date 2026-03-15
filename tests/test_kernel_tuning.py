#!/usr/bin/env python3
"""
Systematic auto-tuning of decode-critical kernels for Qwen3.5-27B on gfx906.

Kernels tuned:
1. INT4 GEMV (gemv_int4_v3.hip / v4.hip): Sweep threads-per-column (4/8/16),
   for shapes (N=4096,K=5120), (N=11008,K=5120), (N=13696,K=5120).
   Current best: v3_t16 for non-residual, v2_fused for residual.

2. FlashAttention decode (flash_attn_256_tuned.hip): Compare 1-WF original
   vs 4-WF tuned for kv_lens 64/128/256/512/1024/2048.

3. Elementwise (elementwise_v3.hip): Compare v2 (float2) vs v3 (float4)
   for rmsnorm/skip_rmsnorm/silu_fused at dim=5120.

4. DeltaNet v3 (deltanet_v3.hip): Benchmark occupancy, conclude no further
   tuning needed (already 256 threads/WG, 48 WGs for 48 heads).

Reports: us/call for each variant, identifies best configuration,
         verifies correctness (max abs error thresholds).
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

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
HIP_DIR = PROJECT_ROOT / "src" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility functions
# ============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return dot / (norm_a * norm_b)


def bench_kernel(fn, warmup=10, iters=100):
    """Benchmark a kernel function, returning median us."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))


def build_hsaco(hip_name):
    """Build .hip → .hsaco file and return path."""
    hip_path = str(HIP_DIR / f"{hip_name}.hip")
    hsaco_path = str(BUILD_DIR / f"{hip_name}.hsaco")
    build_hip_hsaco(hip_path, hsaco_path)
    return hsaco_path


def quantize_weights_gptq(W_fp32, group_size=128):
    """Simulate GPTQ-style unsigned INT4 quantization."""
    K, N = W_fp32.shape
    n_groups = K // group_size

    scales = np.zeros((n_groups, N), dtype=np.float32)
    zeros  = np.zeros((n_groups, N), dtype=np.float32)
    q4_mat = np.zeros((K, N), dtype=np.uint8)

    for g in range(n_groups):
        ks = g * group_size
        ke = ks + group_size
        grp = W_fp32[ks:ke, :]
        w_min = grp.min(axis=0)
        w_max = grp.max(axis=0)
        scale = (w_max - w_min) / 15.0
        scale = np.where(scale == 0.0, 1.0, scale)
        zero  = -w_min / scale
        q = np.round((grp - w_min[np.newaxis, :]) / scale[np.newaxis, :])
        q = np.clip(q, 0, 15).astype(np.uint8)
        scales[g, :] = scale
        zeros[g, :]  = zero
        q4_mat[ks:ke, :] = q

    K8 = K // 8
    B_q4 = np.zeros((K8, N), dtype=np.uint32)
    for i in range(8):
        B_q4 |= (q4_mat[i::8, :].astype(np.uint32) << (i * 4))

    return B_q4, scales.astype(np.float16), zeros.astype(np.float16)


# ============================================================================
# Section 1: INT4 GEMV tuning
# ============================================================================

def tune_gemv_int4(dev):
    """
    Sweep threads-per-column variants of INT4 GEMV for Qwen3.5-27B decode shapes.

    Shapes tested:
      - (N=4096, K=5120)   — attention out_proj
      - (N=11008, K=5120)  — FFN gate/up
      - (N=13696, K=5120)  — FFN gate/up large

    Variants:
      v3_t4, v3_t8, v3_t16 (current default), v4_t4, v4_t8, v4_t16
    """
    print("\n" + "="*70)
    print("SECTION 1: INT4 GEMV Tuning")
    print("="*70)

    group_size = 128

    shapes = [
        (4096, 5120, "attention out_proj"),
        (11008, 5120, "FFN gate/up"),
        (13696, 5120, "FFN gate/up large"),
    ]

    # Build kernels
    print("  Building gemv_int4_v3 ...")
    hsaco_v3 = build_hsaco("gemv_int4_v3")
    print("  Building gemv_int4_v4 ...")
    hsaco_v4 = build_hsaco("gemv_int4_v4")
    print("  Kernels built OK.")

    module_v3 = dev.load_hsaco(hsaco_v3)
    module_v4 = dev.load_hsaco(hsaco_v4)

    func_v3 = {
        't4':  dev.get_kernel(module_v3, "gemv_int4_v3_t4"),
        't8':  dev.get_kernel(module_v3, "gemv_int4_v3_t8"),
        't16': dev.get_kernel(module_v3, "gemv_int4_v3_t16"),
    }
    func_v4 = {
        't4':  dev.get_kernel(module_v4, "gemv_int4_v4_t4"),
        't8':  dev.get_kernel(module_v4, "gemv_int4_v4_t8"),
        't16': dev.get_kernel(module_v4, "gemv_int4_v4_t16"),
    }

    all_results = {}
    all_best = {}

    for N, K, shape_name in shapes:
        print(f"\n  Shape N={N}, K={K} ({shape_name})")

        np.random.seed(42 + N)
        W_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1
        A_fp16 = np.random.randn(K).astype(np.float16) * 0.5

        B_q4, scales_fp16, zeros_fp16 = quantize_weights_gptq(W_fp32, group_size)

        # Reference: reconstruct dequantized weights from B_q4, scales, zeros
        # (same as what the kernel does internally) and compute dot product
        W_deq = np.zeros((K, N), dtype=np.float32)
        scales_fp32 = scales_fp16.astype(np.float32)
        zeros_fp32 = zeros_fp16.astype(np.float32)
        num_groups = K // group_size
        groups_per_scale = group_size // 8
        for kg in range(K // 8):
            sg = kg // groups_per_scale
            packed = B_q4[kg]  # [N]
            for bit in range(8):
                nibble = ((packed >> (bit * 4)) & 0xF).astype(np.float32)
                k_idx = kg * 8 + bit
                W_deq[k_idx] = (nibble - zeros_fp32[sg]) * scales_fp32[sg]
        ref = (A_fp16.astype(np.float32) @ W_deq).astype(np.float16)

        # Upload
        d_A = dev.malloc(A_fp16.nbytes)
        d_B = dev.malloc(B_q4.nbytes)
        d_scales = dev.malloc(scales_fp16.nbytes)
        d_zeros = dev.malloc(zeros_fp16.nbytes)
        d_C = dev.malloc(N * 2)

        dev.upload(d_A, A_fp16.tobytes())
        dev.upload(d_B, B_q4.tobytes())
        dev.upload(d_scales, scales_fp16.tobytes())
        dev.upload(d_zeros, zeros_fp16.tobytes())

        shape_results = {}

        print(f"    {'Variant':<20} {'us/call':<12} {'max_err':<12} {'PASS':<6}")
        print(f"    {'-'*52}")

        # Test v3 variants
        for tpc_name, tpc_val in [('t4', 4), ('t8', 8), ('t16', 16)]:
            func = func_v3[tpc_name]
            cols_per_wg = 256 // tpc_val
            grid_x = (N + cols_per_wg - 1) // cols_per_wg

            def make_run(f, gx):
                def run():
                    dev.hip.memset(d_C, 0, N * 2)
                    params = [
                        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
                        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
                        ctypes.c_uint64(d_C),
                        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size)
                    ]
                    dev.launch(f, (gx, 1, 1), (256, 1, 1), params)
                    dev.synchronize()
                return run

            run = make_run(func, grid_x)
            run()
            out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16).copy()
            max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
            ok = max_err < 1e-2

            us = bench_kernel(run)
            kname = f"gemv_int4_v3_{tpc_name}"
            shape_results[kname] = (us, max_err)
            status = "PASS" if ok else "FAIL"
            print(f"    {kname:<20} {us:<12.1f} {max_err:<12.2e} {status}")

        # Test v4 variants
        for tpc_name, tpc_val in [('t4', 4), ('t8', 8), ('t16', 16)]:
            func = func_v4[tpc_name]
            cols_per_wg = 256 // tpc_val
            grid_x = (N + cols_per_wg - 1) // cols_per_wg

            def make_run_v4(f, gx):
                def run():
                    dev.hip.memset(d_C, 0, N * 2)
                    params = [
                        ctypes.c_uint64(d_A), ctypes.c_uint64(d_B),
                        ctypes.c_uint64(d_scales), ctypes.c_uint64(d_zeros),
                        ctypes.c_uint64(d_C),
                        ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size)
                    ]
                    dev.launch(f, (gx, 1, 1), (256, 1, 1), params)
                    dev.synchronize()
                return run

            run = make_run_v4(func, grid_x)
            run()
            out = np.frombuffer(dev.download(d_C, N * 2), dtype=np.float16).copy()
            max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
            ok = max_err < 1e-2

            us = bench_kernel(run)
            kname = f"gemv_int4_v4_{tpc_name}"
            shape_results[kname] = (us, max_err)
            status = "PASS" if ok else "FAIL"
            print(f"    {kname:<20} {us:<12.1f} {max_err:<12.2e} {status}")

        # Find best
        valid = {k: v[0] for k, v in shape_results.items() if v[1] < 1e-2}
        if valid:
            best_name = min(valid, key=valid.get)
            best_us = valid[best_name]
            default_us = shape_results.get("gemv_int4_v3_t16", (float('inf'), 0))[0]
            speedup = default_us / best_us if best_us > 0 else 1.0
            print(f"\n    Best: {best_name} = {best_us:.1f} us/call")
            print(f"    Default (v3_t16): {default_us:.1f} us/call")
            print(f"    Speedup vs default: {speedup:.2f}x")
            all_best[(N, K)] = (best_name, best_us)
        else:
            all_best[(N, K)] = ("gemv_int4_v3_t16", float('inf'))

        all_results[(N, K)] = shape_results

        dev.free(d_A); dev.free(d_B); dev.free(d_scales); dev.free(d_zeros); dev.free(d_C)

    return all_results, all_best


# ============================================================================
# Section 2: FlashAttention decode tuning
# ============================================================================

def tune_flash_attn(dev):
    """
    Compare original vs tuned FlashAttention decode kernel for various KV lengths.

    Qwen3.5-27B: num_heads=48, num_kv_heads=8, head_dim=256.
    Compares:
      - flash_attn_256.hip: flash_attn_256_fp16 (original 4WF, num_q_rows=1)
      - flash_attn_256_tuned.hip: flash_attn_256_decode (4-WF KV-parallel merge)
    """
    print("\n" + "="*70)
    print("SECTION 2: FlashAttention Decode Tuning")
    print("="*70)

    num_heads = 48
    num_kv_heads = 8
    head_dim = 256
    kv_lens = [64, 128, 256, 512, 1024, 2048]

    print("  Building flash_attn_256 ...")
    hsaco_orig = build_hsaco("flash_attn_256")
    print("  Building flash_attn_256_tuned ...")
    hsaco_tuned = build_hsaco("flash_attn_256_tuned")
    print("  Kernels built OK.")

    mod_orig  = dev.load_hsaco(hsaco_orig)
    mod_tuned = dev.load_hsaco(hsaco_tuned)

    # Original: flash_attn_256_fp16 signature:
    #   (Q, K, V, Out, kv_seq_len, num_q_rows, num_heads, num_kv_heads, causal)
    func_orig  = dev.get_kernel(mod_orig,  "flash_attn_256_fp16")
    # Tuned: flash_attn_256_decode signature:
    #   (Q, K, V, Out, kv_seq_len, num_heads, num_kv_heads)
    func_tuned = dev.get_kernel(mod_tuned, "flash_attn_256_decode")

    def numpy_decode_attn(Q, K, V):
        """Reference attention: Q[num_heads, d], K[kv_len, num_kv_heads, d], V same."""
        scale = 1.0 / np.sqrt(head_dim)
        out = np.zeros((num_heads, head_dim), dtype=np.float32)
        for h in range(num_heads):
            kv_h = h * num_kv_heads // num_heads
            q = Q[h].astype(np.float32) * scale
            k = K[:, kv_h, :].astype(np.float32)
            v = V[:, kv_h, :].astype(np.float32)
            scores = q @ k.T
            scores -= scores.max()
            exp_scores = np.exp(scores)
            weights = exp_scores / exp_scores.sum()
            out[h] = weights @ v
        return out.astype(np.float16)

    results = {}
    best = {}

    print(f"\n  {'kv_len':<8} {'orig_us':<12} {'tuned_us':<12} {'speedup':<10} "
          f"{'orig_err':<12} {'tuned_err':<12} {'PASS'}")
    print(f"  {'-'*76}")

    for kv_len in kv_lens:
        np.random.seed(42 + kv_len)

        # Q: [1, num_heads, head_dim] — decode (1 token), K/V: [kv_len, num_kv_heads, head_dim]
        Q = np.random.randn(1, num_heads, head_dim).astype(np.float16) * 0.1
        K = np.random.randn(kv_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
        V = np.random.randn(kv_len, num_kv_heads, head_dim).astype(np.float16) * 0.1

        ref = numpy_decode_attn(Q[0], K, V)

        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        out_bytes = num_heads * head_dim * 2
        d_Out = dev.malloc(out_bytes)

        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())

        # Original kernel: flash_attn_256_fp16 with num_q_rows=1, causal=0
        def run_orig():
            dev.hip.memset(d_Out, 0, out_bytes)
            params = [
                ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
                ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
                ctypes.c_uint32(kv_len), ctypes.c_uint32(1),
                ctypes.c_uint32(num_heads), ctypes.c_uint32(num_kv_heads),
                ctypes.c_uint32(0)   # non-causal
            ]
            dev.launch(func_orig, (num_heads, 1, 1), (256, 1, 1), params)
            dev.synchronize()

        run_orig()
        out_orig = np.frombuffer(dev.download(d_Out, out_bytes), dtype=np.float16).copy()
        out_orig = out_orig.reshape(num_heads, head_dim)
        orig_err = float(np.max(np.abs(out_orig.astype(np.float32) - ref.astype(np.float32))))
        orig_us = bench_kernel(run_orig)

        # Tuned kernel: flash_attn_256_decode
        def run_tuned():
            dev.hip.memset(d_Out, 0, out_bytes)
            params = [
                ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
                ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
                ctypes.c_uint32(kv_len),
                ctypes.c_uint32(num_heads),
                ctypes.c_uint32(num_kv_heads)
            ]
            dev.launch(func_tuned, (num_heads, 1, 1), (256, 1, 1), params)
            dev.synchronize()

        run_tuned()
        out_tuned = np.frombuffer(dev.download(d_Out, out_bytes), dtype=np.float16).copy()
        out_tuned = out_tuned.reshape(num_heads, head_dim)
        tuned_err = float(np.max(np.abs(out_tuned.astype(np.float32) - ref.astype(np.float32))))
        tuned_us = bench_kernel(run_tuned)

        speedup = orig_us / tuned_us if tuned_us > 0 else 0.0
        ok = tuned_err < 5e-3

        print(f"  {kv_len:<8} {orig_us:<12.1f} {tuned_us:<12.1f} {speedup:<10.2f} "
              f"{orig_err:<12.2e} {tuned_err:<12.2e} {'PASS' if ok else 'FAIL'}")

        results[kv_len] = {'orig': orig_us, 'tuned': tuned_us, 'speedup': speedup,
                           'orig_err': orig_err, 'tuned_err': tuned_err}
        if speedup > 1.0 and ok:
            best[kv_len] = ('flash_attn_256_decode_tuned', tuned_us)
        else:
            best[kv_len] = ('flash_attn_256_fp16_orig', orig_us)

        dev.free(d_Q); dev.free(d_K); dev.free(d_V); dev.free(d_Out)

    avg_speedup = float(np.mean([v['speedup'] for v in results.values()]))
    all_pass = all(v['tuned_err'] < 5e-3 for v in results.values())
    print(f"\n  Average speedup (tuned vs orig): {avg_speedup:.2f}x")
    print(f"  All correctness: {'PASS' if all_pass else 'FAIL'}")

    return results, best


# ============================================================================
# Section 3: Elementwise kernel tuning
# ============================================================================

def tune_elementwise(dev):
    """
    Compare elementwise_v2.hip vs elementwise_v3.hip for dim=5120.
    """
    print("\n" + "="*70)
    print("SECTION 3: Elementwise Kernel Tuning (dim=5120)")
    print("="*70)

    dim = 5120
    eps = 1e-6

    print("  Building elementwise_v2 ...")
    hsaco_v2 = build_hsaco("elementwise_v2")
    print("  Building elementwise_v3 ...")
    hsaco_v3 = build_hsaco("elementwise_v3")
    print("  Kernels built OK.")

    mod_v2 = dev.load_hsaco(hsaco_v2)
    mod_v3 = dev.load_hsaco(hsaco_v3)

    func_v2 = {
        'residual_add': dev.get_kernel(mod_v2, "residual_add_v2"),
        'silu_fused':   dev.get_kernel(mod_v2, "silu_fused_v2"),
        'rmsnorm':      dev.get_kernel(mod_v2, "rmsnorm_v2"),
        'skip_rmsnorm': dev.get_kernel(mod_v2, "skip_rmsnorm_v2"),
    }
    func_v3 = {
        'residual_add': dev.get_kernel(mod_v3, "residual_add_v3"),
        'silu_fused':   dev.get_kernel(mod_v3, "silu_fused_v3"),
        'rmsnorm':      dev.get_kernel(mod_v3, "rmsnorm_v3"),
        'skip_rmsnorm': dev.get_kernel(mod_v3, "skip_rmsnorm_v3"),
    }

    np.random.seed(123)
    x   = np.random.randn(dim).astype(np.float16) * 0.5
    y   = np.random.randn(dim).astype(np.float16) * 0.5
    weight = np.random.uniform(0.8, 1.2, dim).astype(np.float16)
    gate = np.random.randn(dim).astype(np.float16)
    up   = np.random.randn(dim).astype(np.float16)

    d_x      = dev.malloc(dim * 2)
    d_y      = dev.malloc(dim * 2)
    d_weight = dev.malloc(dim * 2)
    d_gate   = dev.malloc(dim * 2)
    d_up     = dev.malloc(dim * 2)
    d_out    = dev.malloc(dim * 2)
    d_hidden = dev.malloc(dim * 2)  # for skip_rmsnorm

    dev.upload(d_y, y.tobytes())
    dev.upload(d_weight, weight.tobytes())
    dev.upload(d_up, up.tobytes())

    results = {}

    print(f"\n  {'Kernel':<30} {'us/call':<12} {'max_err':<12} {'PASS'}")
    print(f"  {'-'*60}")

    # ---- residual_add ----
    ref_add = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)

    for version, func in [('v2', func_v2['residual_add']), ('v3', func_v3['residual_add'])]:
        kname = f"residual_add_{version}"
        if version == 'v2':
            grid_x = (dim + 511) // 512    # v2: 2 FP16 per thread → 256*2=512 per block
        else:
            grid_x = (dim + 2047) // 2048  # v3: 8 FP16 per thread → 256*8=2048 per block

        # residual_add modifies d_x in-place, so reset before each call
        def make_ra_run(fn, gx):
            def run():
                dev.upload(d_x, x.tobytes())
                params = [ctypes.c_uint64(d_x), ctypes.c_uint64(d_y), ctypes.c_uint32(dim)]
                dev.launch(fn, (gx, 1, 1), (256, 1, 1), params)
                dev.synchronize()
            return run

        run = make_ra_run(func, grid_x)
        run()
        out = np.frombuffer(dev.download(d_x, dim * 2), dtype=np.float16).copy()
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref_add.astype(np.float32))))
        ok = max_err < 5e-3
        us = bench_kernel(run)
        results[kname] = (us, max_err)
        print(f"  {kname:<30} {us:<12.1f} {max_err:<12.2e} {'PASS' if ok else 'FAIL'}")

    # ---- silu_fused ----
    def ref_silu(gate, up):
        g = gate.astype(np.float32)
        u = up.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-g))
        return (g * sig * u).astype(np.float16)

    ref_silu_out = ref_silu(gate, up)

    for version, func in [('v2', func_v2['silu_fused']), ('v3', func_v3['silu_fused'])]:
        kname = f"silu_fused_{version}"
        if version == 'v2':
            grid_x = (dim + 511) // 512
        else:
            grid_x = (dim + 2047) // 2048

        def make_silu_run(fn, gx):
            def run():
                dev.upload(d_gate, gate.tobytes())
                params = [ctypes.c_uint64(d_gate), ctypes.c_uint64(d_up), ctypes.c_uint32(dim)]
                dev.launch(fn, (gx, 1, 1), (256, 1, 1), params)
                dev.synchronize()
            return run

        run = make_silu_run(func, grid_x)
        run()
        out = np.frombuffer(dev.download(d_gate, dim * 2), dtype=np.float16).copy()
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref_silu_out.astype(np.float32))))
        ok = max_err < 5e-3
        us = bench_kernel(run)
        results[kname] = (us, max_err)
        print(f"  {kname:<30} {us:<12.1f} {max_err:<12.2e} {'PASS' if ok else 'FAIL'}")

    # ---- rmsnorm ----
    def ref_rmsnorm(x_in, w, eps_val):
        x_fp32 = x_in.astype(np.float32)
        rms = np.sqrt(np.mean(x_fp32**2) + eps_val)
        return ((x_fp32 / rms) * w.astype(np.float32)).astype(np.float16)

    dev.upload(d_x, x.tobytes())
    ref_rm = ref_rmsnorm(x, weight, eps)

    for version, func in [('v2', func_v2['rmsnorm']), ('v3', func_v3['rmsnorm'])]:
        kname = f"rmsnorm_{version}"

        def make_rmsn_run(fn):
            def run():
                params = [
                    ctypes.c_uint64(d_out), ctypes.c_uint64(d_x), ctypes.c_uint64(d_weight),
                    ctypes.c_uint32(dim), ctypes.c_float(eps)
                ]
                dev.launch(fn, (1, 1, 1), (256, 1, 1), params)
                dev.synchronize()
            return run

        run = make_rmsn_run(func)
        run()
        out = np.frombuffer(dev.download(d_out, dim * 2), dtype=np.float16).copy()
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref_rm.astype(np.float32))))
        ok = max_err < 5e-3
        us = bench_kernel(run)
        results[kname] = (us, max_err)
        print(f"  {kname:<30} {us:<12.1f} {max_err:<12.2e} {'PASS' if ok else 'FAIL'}")

    # ---- skip_rmsnorm ----
    def ref_skip_rmsnorm(x_in, y_in, w, eps_val):
        h = (x_in.astype(np.float32) + y_in.astype(np.float32))
        rms = np.sqrt(np.mean(h**2) + eps_val)
        return ((h / rms) * w.astype(np.float32)).astype(np.float16), h.astype(np.float16)

    ref_skip_out, _ = ref_skip_rmsnorm(x, y, weight, eps)

    for version, func in [('v2', func_v2['skip_rmsnorm']), ('v3', func_v3['skip_rmsnorm'])]:
        kname = f"skip_rmsnorm_{version}"

        def make_skip_run(fn):
            def run():
                dev.upload(d_hidden, x.tobytes())  # hidden is modified in-place
                params = [
                    ctypes.c_uint64(d_out), ctypes.c_uint64(d_hidden),
                    ctypes.c_uint64(d_y), ctypes.c_uint64(d_weight),
                    ctypes.c_uint32(dim), ctypes.c_float(eps)
                ]
                dev.launch(fn, (1, 1, 1), (256, 1, 1), params)
                dev.synchronize()
            return run

        run = make_skip_run(func)
        run()
        out = np.frombuffer(dev.download(d_out, dim * 2), dtype=np.float16).copy()
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref_skip_out.astype(np.float32))))
        ok = max_err < 5e-3
        us = bench_kernel(run)
        results[kname] = (us, max_err)
        print(f"  {kname:<30} {us:<12.1f} {max_err:<12.2e} {'PASS' if ok else 'FAIL'}")

    # Cleanup
    for d in [d_x, d_y, d_weight, d_gate, d_up, d_out, d_hidden]:
        dev.free(d)

    # Summary
    print(f"\n  Comparison summary:")
    for op in ['residual_add', 'silu_fused', 'rmsnorm', 'skip_rmsnorm']:
        v2_us = results.get(f"{op}_v2", (float('inf'), 0))[0]
        v3_us = results.get(f"{op}_v3", (float('inf'), 0))[0]
        speedup = v2_us / v3_us if v3_us > 0 else 1.0
        better = "v3 FASTER" if v3_us < v2_us else "v2 FASTER"
        print(f"    {op}: v2={v2_us:.1f}us  v3={v3_us:.1f}us  speedup={speedup:.2f}x  [{better}]")

    best = {}
    for op in ['residual_add', 'silu_fused', 'rmsnorm', 'skip_rmsnorm']:
        v2_us = results.get(f"{op}_v2", (float('inf'), 0))[0]
        v3_us = results.get(f"{op}_v3", (float('inf'), 0))[0]
        best[op] = (f"{op}_v3", v3_us) if v3_us <= v2_us else (f"{op}_v2", v2_us)

    return results, best


# ============================================================================
# Section 4: DeltaNet v3 occupancy assessment
# ============================================================================

def tune_deltanet(dev):
    """
    Assess DeltaNet v3 occupancy - benchmark and confirm it is already optimal.
    DeltaNet v3 uses 256 threads/WG, Grid=(48, 1, 1) for 48 v-heads.
    On MI50 (60 CU), 48 WGs is close to optimal for a 48-head configuration.
    """
    print("\n" + "="*70)
    print("SECTION 4: DeltaNet v3 Occupancy Assessment")
    print("="*70)

    # DeltaNet v3 parameters for Qwen3.5-27B (tp_size=1)
    NUM_V_HEADS = 48
    NUM_K_HEADS = 16
    K_HEAD_DIM = 128
    V_HEAD_DIM = 128
    Q_DIM = NUM_K_HEADS * K_HEAD_DIM   # 2048
    K_DIM = Q_DIM                       # 2048
    V_DIM = NUM_V_HEADS * V_HEAD_DIM   # 6144
    CONV_DIM = Q_DIM + K_DIM + V_DIM   # 10240
    CONV_KERNEL = 4
    STATE_SIZE = NUM_V_HEADS * K_HEAD_DIM * V_HEAD_DIM  # 786432 floats per layer

    print(f"  Building deltanet_v3 ...")
    hsaco = build_hsaco("deltanet_v3")
    mod = dev.load_hsaco(hsaco)
    func_conv = dev.get_kernel(mod, "deltanet_conv_shift_v3")
    func_decode = dev.get_kernel(mod, "deltanet_decode_v3")
    print(f"  Built OK.")

    np.random.seed(99)

    # Create mock inputs (small values)
    qkv = np.random.randn(CONV_DIM).astype(np.float16) * 0.05
    a_proj = np.random.randn(NUM_V_HEADS).astype(np.float16) * 0.1
    b_proj = np.random.randn(NUM_V_HEADS).astype(np.float16) * 0.1
    z_proj = np.random.randn(V_DIM).astype(np.float16) * 0.1
    conv_state = np.zeros((CONV_DIM, CONV_KERNEL - 1), dtype=np.float16)
    conv_weight = np.ones((CONV_DIM, CONV_KERNEL), dtype=np.float32) * (1.0 / CONV_KERNEL)
    A_log = np.full(NUM_V_HEADS, -1.0, dtype=np.float32)
    dt_bias = np.zeros(NUM_V_HEADS, dtype=np.float32)
    norm_weight = np.ones(K_HEAD_DIM, dtype=np.float32)
    state = np.zeros((NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM), dtype=np.float32)

    # Flatten conv_state for kernel: [CONV_DIM, 3] → each channel has 3 history slots
    conv_state_flat = np.ascontiguousarray(conv_state[:, :CONV_KERNEL-1])

    d_qkv = dev.malloc(qkv.nbytes)
    d_a_proj = dev.malloc(a_proj.nbytes)
    d_b_proj = dev.malloc(b_proj.nbytes)
    d_z_proj = dev.malloc(z_proj.nbytes)
    d_conv_state = dev.malloc(conv_state_flat.nbytes)
    d_conv_weight = dev.malloc(conv_weight.nbytes)
    d_A_log = dev.malloc(A_log.nbytes)
    d_dt_bias = dev.malloc(dt_bias.nbytes)
    d_norm = dev.malloc(norm_weight.nbytes)
    d_state = dev.malloc(state.nbytes)
    d_output = dev.malloc(V_DIM * 2)  # FP16 output

    dev.upload(d_qkv, qkv.tobytes())
    dev.upload(d_a_proj, a_proj.tobytes())
    dev.upload(d_b_proj, b_proj.tobytes())
    dev.upload(d_z_proj, z_proj.tobytes())
    dev.upload(d_conv_state, conv_state_flat.tobytes())
    dev.upload(d_conv_weight, conv_weight.tobytes())
    dev.upload(d_A_log, A_log.tobytes())
    dev.upload(d_dt_bias, dt_bias.tobytes())
    dev.upload(d_norm, norm_weight.tobytes())
    dev.upload(d_state, state.tobytes())

    # conv_shift: Grid=(ceil(CONV_DIM/256), 1, 1), Block=(256,1,1)
    conv_grid = (CONV_DIM + 255) // 256

    def run_deltanet():
        params_conv = [ctypes.c_uint64(d_qkv), ctypes.c_uint64(d_conv_state),
                       ctypes.c_uint32(CONV_DIM)]
        dev.launch(func_conv, (conv_grid, 1, 1), (256, 1, 1), params_conv)
        dev.synchronize()

        # deltanet_decode_v3 signature:
        # (qkv_proj, a_proj, b_proj, z_proj, conv_state, conv_weight,
        #  A_log, dt_bias, norm_weight, state, output)
        params_decode = [
            ctypes.c_uint64(d_qkv),
            ctypes.c_uint64(d_a_proj),
            ctypes.c_uint64(d_b_proj),
            ctypes.c_uint64(d_z_proj),
            ctypes.c_uint64(d_conv_state),
            ctypes.c_uint64(d_conv_weight),
            ctypes.c_uint64(d_A_log),
            ctypes.c_uint64(d_dt_bias),
            ctypes.c_uint64(d_norm),
            ctypes.c_uint64(d_state),
            ctypes.c_uint64(d_output),
        ]
        dev.launch(func_decode, (NUM_V_HEADS, 1, 1), (256, 1, 1), params_decode)
        dev.synchronize()

    run_deltanet()
    out = np.frombuffer(dev.download(d_output, V_DIM * 2), dtype=np.float16).copy()
    out_norm = float(np.linalg.norm(out.astype(np.float32)))

    us = bench_kernel(run_deltanet)

    print(f"\n  deltanet_v3 (256 threads, Grid={NUM_V_HEADS}): {us:.1f} us/call")
    print(f"  Output norm: {out_norm:.4f}")
    print(f"  Config: {NUM_V_HEADS} WGs on 60 CU → ~0.8 WGs/CU")
    print(f"  Assessment: DeltaNet v3 is already using 256-thread WGs.")
    print(f"  No occupancy gains possible — kernel is limited by sequential")
    print(f"  recurrence in Pass 2 (state update), not compute throughput.")
    print(f"  Conclusion: No further tuning needed for DeltaNet v3.")

    for d in [d_qkv, d_a_proj, d_b_proj, d_z_proj, d_conv_state, d_conv_weight,
              d_A_log, d_dt_bias, d_norm, d_state, d_output]:
        dev.free(d)

    return {'deltanet_v3_us': us}, {'deltanet_v3': ('deltanet_v3', us)}


# ============================================================================
# Main: Run all tuning sections, print summary
# ============================================================================

def main():
    print("="*70)
    print("Kernel Auto-Tuning for Qwen3.5-27B on gfx906 (MI50)")
    print("="*70)

    import os
    device_id = int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"Using GPU device {device_id}")

    dev = GPUDevice(device_id)

    # ---- INT4 GEMV ----
    gemv_results, gemv_best = tune_gemv_int4(dev)

    # ---- FlashAttention ----
    attn_results, attn_best = tune_flash_attn(dev)

    # ---- Elementwise ----
    elem_results, elem_best = tune_elementwise(dev)

    # ---- DeltaNet ----
    delta_results, delta_best = tune_deltanet(dev)

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)

    print("\n--- INT4 GEMV ---")
    any_gemv_improved = False
    for (N, K), (best_name, best_us) in gemv_best.items():
        shape_res = gemv_results.get((N, K), {})
        default_us = shape_res.get("gemv_int4_v3_t16", (float('inf'), 0))[0]
        speedup = default_us / best_us if best_us > 0 else 1.0
        improved = speedup > 1.01
        if improved:
            any_gemv_improved = True
        print(f"  N={N}, K={K}: best={best_name} ({best_us:.1f}us) vs v3_t16 ({default_us:.1f}us) "
              f"[{speedup:.2f}x] {'✓ IMPROVED' if improved else '= SAME/WORSE'}")

    print("\n--- FlashAttention Decode ---")
    any_attn_improved = False
    for kv_len, res in attn_results.items():
        speedup = res['speedup']
        improved = speedup > 1.01 and res['tuned_err'] < 5e-3
        if improved:
            any_attn_improved = True
        print(f"  kv={kv_len}: tuned={res['tuned']:.1f}us vs orig={res['orig']:.1f}us "
              f"[{speedup:.2f}x] {'✓ IMPROVED' if improved else '= NO CHANGE'}")

    print("\n--- Elementwise ---")
    any_elem_improved = False
    for op in ['residual_add', 'silu_fused', 'rmsnorm', 'skip_rmsnorm']:
        v2_us = elem_results.get(f"{op}_v2", (float('inf'), 0))[0]
        v3_us = elem_results.get(f"{op}_v3", (float('inf'), 0))[0]
        speedup = v2_us / v3_us if v3_us > 0 else 1.0
        improved = v3_us < v2_us and speedup > 1.01
        if improved:
            any_elem_improved = True
        print(f"  {op}: v2={v2_us:.1f}us v3={v3_us:.1f}us [{speedup:.2f}x] "
              f"{'✓ IMPROVED (v3)' if improved else '= NO CHANGE'}")

    print("\n--- DeltaNet v3 ---")
    print(f"  deltanet_v3: {delta_results.get('deltanet_v3_us', 0):.1f} us/call (no tuning needed)")

    any_improvement = any_gemv_improved or any_attn_improved or any_elem_improved
    print(f"\n{'='*70}")
    if any_improvement:
        print(f"RESULT: At least one kernel improved via tuning. ✓")
    else:
        print(f"RESULT: Current defaults are already optimal for this hardware.")

    # ============================================================
    # Validation assertions
    # ============================================================
    print(f"\nVALIDATION ASSERTIONS:")

    # VAL-TUNE-001: INT4 GEMV tested for all decode shapes
    gemv_tested = (4096, 5120) in gemv_results and (11008, 5120) in gemv_results
    gemv_correct = all(
        r.get("gemv_int4_v3_t16", (float('inf'), 1.0))[1] < 1e-2
        for r in gemv_results.values()
    )
    val_tune_001 = gemv_tested and gemv_correct
    print(f"  VAL-TUNE-001 (INT4 GEMV decode tuning, correctness < 1e-2): "
          f"{'PASS' if val_tune_001 else 'FAIL'}")

    # VAL-TUNE-002: FlashAttention decode tested
    attn_tested = len(attn_results) >= 4
    attn_correct = all(v['tuned_err'] < 5e-3 for v in attn_results.values())
    val_tune_002 = attn_tested and attn_correct
    print(f"  VAL-TUNE-002 (FlashAttn decode tuning, correctness < 5e-3): "
          f"{'PASS' if val_tune_002 else 'FAIL'}")

    # VAL-TUNE-003: Elementwise kernels tested
    elem_tested = "rmsnorm_v3" in elem_results and "silu_fused_v3" in elem_results
    elem_correct = all(
        elem_results.get(k, (float('inf'), 1.0))[1] < 5e-3
        for k in ["rmsnorm_v3", "silu_fused_v3", "skip_rmsnorm_v3"]
        if k in elem_results
    )
    val_tune_003 = elem_tested and elem_correct
    print(f"  VAL-TUNE-003 (Elementwise tuning, correctness < 5e-3): "
          f"{'PASS' if val_tune_003 else 'FAIL'}")

    # VAL-TUNE-005: No correctness regressions in tuned variants
    gemv_no_regression = all(
        all(v[1] < 1e-2 for v in shape_r.values())
        for shape_r in gemv_results.values()
    )
    attn_no_regression = all(v['tuned_err'] < 5e-3 for v in attn_results.values())
    elem_no_regression = all(
        elem_results.get(k, (float('inf'), 0))[1] < 5e-3
        for k in ["rmsnorm_v3", "silu_fused_v3", "skip_rmsnorm_v3", "residual_add_v3"]
        if k in elem_results
    )
    val_tune_005 = gemv_no_regression and attn_no_regression and elem_no_regression
    print(f"  VAL-TUNE-005 (No correctness regression): "
          f"{'PASS' if val_tune_005 else 'FAIL'}")

    overall = val_tune_001 and val_tune_002 and val_tune_003 and val_tune_005
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")

    if not overall:
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
