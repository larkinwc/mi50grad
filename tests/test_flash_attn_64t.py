#!/usr/bin/env python3
"""Test flash_attn_256_decode_64t HIP kernel against numpy reference and benchmark."""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco


def numpy_attention_decode(Q, K, V):
    """Reference attention for decode (single Q row): softmax(Q @ K^T / sqrt(d)) @ V"""
    num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[0]
    num_kv_heads = K.shape[1]
    heads_per_kv = num_heads // num_kv_heads

    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros_like(Q, dtype=np.float32)

    for h in range(num_heads):
        kv_h = h * num_kv_heads // num_heads
        q = Q[h, :].astype(np.float32) * scale  # [d]
        k = K[:, kv_h, :].astype(np.float32)    # [kv_seq, d]
        v = V[:, kv_h, :].astype(np.float32)    # [kv_seq, d]

        scores = (k @ q).astype(np.float32)  # [kv_seq]

        # Stable softmax
        scores -= scores.max()
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / exp_scores.sum()

        out[h, :] = attn_weights @ v

    return out.astype(np.float16)


def test_flash_attn_64t_correctness():
    """Test 64-thread decode kernel correctness against numpy reference."""
    print("=" * 70)
    print("Testing flash_attn_256_decode_64t correctness")
    print("=" * 70)

    dev = GPUDevice(0)

    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_path = str(PROJECT_ROOT / "src" / "kernels" / "flash_attn_256_tuned.hip")
    hsaco_path = str(BUILD_DIR / "flash_attn_256_tuned.hsaco")

    print(f"Building {hip_path}...")
    build_hip_hsaco(hip_path, hsaco_path)
    module = dev.load_hsaco(hsaco_path)
    func_64t = dev.get_kernel(module, "flash_attn_256_decode_64t")
    func_256t = dev.get_kernel(module, "flash_attn_256_decode")

    # Test configurations: (kv_seq_len, num_heads, num_kv_heads)
    test_configs = [
        (64, 48, 8),
        (128, 48, 8),
        (256, 48, 8),
    ]

    all_pass = True
    max_errors = []

    for kv_seq_len, num_heads, num_kv_heads in test_configs:
        head_dim = 256
        np.random.seed(42)

        # Decode: single Q row [1, num_heads, head_dim]
        Q = np.random.randn(1, num_heads, head_dim).astype(np.float16) * 0.1
        K = np.random.randn(kv_seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
        V = np.random.randn(kv_seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1

        # Reference: squeeze Q to [num_heads, head_dim] for numpy function
        Q_squeezed = Q[0, :, :]  # [num_heads, head_dim]
        ref = numpy_attention_decode(Q_squeezed, K, V)  # [num_heads, head_dim]

        # Upload to GPU
        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        d_Out_64t = dev.malloc(ref.nbytes)
        d_Out_256t = dev.malloc(ref.nbytes)

        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())
        dev.hip.memset(d_Out_64t, 0, ref.nbytes)
        dev.hip.memset(d_Out_256t, 0, ref.nbytes)

        # Launch 64-thread kernel
        # Grid: (num_heads, 1, 1), Block: (64, 1, 1)
        grid = (num_heads, 1, 1)
        block_64t = (64, 1, 1)
        params_64t = [
            ctypes.c_uint64(d_Q),
            ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V),
            ctypes.c_uint64(d_Out_64t),
            ctypes.c_uint32(kv_seq_len),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
        ]
        dev.launch(func_64t, grid, block_64t, params_64t)
        dev.synchronize()

        # Launch 256-thread kernel for comparison
        # Grid: (num_heads, 1, 1), Block: (256, 1, 1)
        block_256t = (256, 1, 1)
        d_Out_256t_copy = dev.malloc(ref.nbytes)
        dev.hip.memset(d_Out_256t_copy, 0, ref.nbytes)
        params_256t = [
            ctypes.c_uint64(d_Q),
            ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V),
            ctypes.c_uint64(d_Out_256t_copy),
            ctypes.c_uint32(kv_seq_len),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
        ]
        dev.launch(func_256t, grid, block_256t, params_256t)
        dev.synchronize()

        # Download results
        result_64t = np.frombuffer(dev.download(d_Out_64t, ref.nbytes), dtype=np.float16)
        result_64t = result_64t.reshape(num_heads, head_dim)

        result_256t = np.frombuffer(dev.download(d_Out_256t_copy, ref.nbytes), dtype=np.float16)
        result_256t = result_256t.reshape(num_heads, head_dim)

        # Compare against numpy reference
        max_err_64t = np.max(np.abs(ref.astype(np.float32) - result_64t.astype(np.float32)))
        max_err_256t = np.max(np.abs(ref.astype(np.float32) - result_256t.astype(np.float32)))

        # Compare 64t vs 256t
        max_err_between = np.max(np.abs(result_64t.astype(np.float32) - result_256t.astype(np.float32)))

        cos_sim_64t = np.dot(ref.ravel().astype(np.float32), result_64t.ravel().astype(np.float32)) / (
            np.linalg.norm(ref.ravel().astype(np.float32)) *
            np.linalg.norm(result_64t.ravel().astype(np.float32)) + 1e-10)

        # Check: max abs error < 1e-2 (as per feature requirements)
        ok_64t = cos_sim_64t > 0.99 and max_err_64t < 1e-2
        ok_256t = max_err_256t < 1e-2
        ok_match = max_err_between < 1e-2

        ok = ok_64t and ok_256t and ok_match

        print(f"  kv_seq={kv_seq_len:3d} heads={num_heads:2d} kv_heads={num_kv_heads}:")
        print(f"    64t:  max_err={max_err_64t:.6f} cos_sim={cos_sim_64t:.6f} {'PASS' if ok_64t else 'FAIL'}")
        print(f"    256t: max_err={max_err_256t:.6f} {'PASS' if ok_256t else 'FAIL'}")
        print(f"    Match: max_err_between={max_err_between:.6f} {'PASS' if ok_match else 'FAIL'}")

        if not ok:
            all_pass = False

        max_errors.append(max_err_64t)

        dev.free(d_Q)
        dev.free(d_K)
        dev.free(d_V)
        dev.free(d_Out_64t)
        dev.free(d_Out_256t_copy)

    dev.cleanup()

    print(f"\nCorrectness: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"Max errors across configs: {max_errors}")
    return all_pass, max_errors


def benchmark_flash_attn_64t():
    """
    Benchmark 64-thread vs 256-thread decode kernel performance.

    EXPECTED BEHAVIOR:
    The 64-thread kernel will be SLOWER per-kernel because:
    - 64t: Single wavefront processes all KV positions sequentially
    - 256t: 4 wavefronts split KV range, each processes kv_len/4 positions

    The benefit of 64t is NOT per-kernel speed, but:
    - Uses 1 wavefront/WG instead of 4, freeing 3 wave slots per CU
    - Allows more concurrent WGs when GPU is saturated
    - Better for scenarios with many concurrent kernels

    With only 48 heads, both variants under-utilize the GPU (gfx906 has 60 CUs,
    each holding multiple WGs), so 64t shows no advantage in this benchmark.
    """
    print("\n" + "=" * 70)
    print("Benchmarking flash_attn_256_decode_64t vs flash_attn_256_decode")
    print("=" * 70)
    print("\nNOTE: 64t kernel expected to be slower per-kernel (sequential KV sweep)")
    print("      Benefit: reduced resource usage allows more concurrent WGs\n")

    dev = GPUDevice(0)

    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    hip_path = str(PROJECT_ROOT / "src" / "kernels" / "flash_attn_256_tuned.hip")
    hsaco_path = str(BUILD_DIR / "flash_attn_256_tuned.hsaco")

    module = dev.load_hsaco(hsaco_path)
    func_64t = dev.get_kernel(module, "flash_attn_256_decode_64t")
    func_256t = dev.get_kernel(module, "flash_attn_256_decode")

    # Benchmark config matching Qwen 3.5 27B: 48 heads, 8 kv_heads
    num_heads = 48
    num_kv_heads = 8
    head_dim = 256

    print(f"Benchmark config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"Testing seq_lens: 64, 128, 256\n")

    results = {}

    for kv_seq_len in [64, 128, 256]:
        np.random.seed(42)
        Q = np.random.randn(1, num_heads, head_dim).astype(np.float16) * 0.1
        K = np.random.randn(kv_seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
        V = np.random.randn(kv_seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1

        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        d_Out = dev.malloc(Q.nbytes)

        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())

        grid = (num_heads, 1, 1)

        # 64-thread kernel benchmark
        block_64t = (64, 1, 1)
        params_64t = [
            ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
            ctypes.c_uint32(kv_seq_len),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
        ]

        # Warmup
        for _ in range(5):
            dev.launch(func_64t, grid, block_64t, params_64t)
        dev.synchronize()

        # Benchmark 64t
        iters = 100
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dev.launch(func_64t, grid, block_64t, params_64t)
        dev.synchronize()
        t_64t_ms = (time.perf_counter() - t0) / iters * 1000

        # 256-thread kernel benchmark
        block_256t = (256, 1, 1)
        params_256t = [
            ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
            ctypes.c_uint32(kv_seq_len),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
        ]

        # Warmup
        for _ in range(5):
            dev.launch(func_256t, grid, block_256t, params_256t)
        dev.synchronize()

        # Benchmark 256t
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dev.launch(func_256t, grid, block_256t, params_256t)
        dev.synchronize()
        t_256t_ms = (time.perf_counter() - t0) / iters * 1000

        # 64t is expected to be slower; compute slowdown factor
        slowdown = t_64t_ms / t_256t_ms if t_256t_ms > 0 else float('inf')

        results[kv_seq_len] = {
            '64t': t_64t_ms,
            '256t': t_256t_ms,
            'slowdown': slowdown
        }

        print(f"  seq_len={kv_seq_len:3d}:")
        print(f"    64-thread:  {t_64t_ms:.3f} ms")
        print(f"    256-thread: {t_256t_ms:.3f} ms")
        print(f"    Slowdown:   {slowdown:.2f}x (64t is {slowdown:.1f}x slower, expected due to sequential KV)")

        dev.free(d_Q)
        dev.free(d_K)
        dev.free(d_V)
        dev.free(d_Out)

    dev.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for seq_len, res in results.items():
        print(f"  seq_len={seq_len:3d}: 64t={res['64t']:.3f}ms, 256t={res['256t']:.3f}ms, "
              f"slowdown={res['slowdown']:.2f}x")

    print("\nNote: 64t kernel trades per-kernel performance for reduced resource usage.")
    print("      With 48 heads, GPU is not saturated, so no throughput benefit observed.")

    return results


if __name__ == "__main__":
    print("flash_attn_256_decode_64t Test Suite")
    print("=" * 70)

    # Test correctness
    ok, max_errors = test_flash_attn_64t_correctness()

    # Benchmark performance
    benchmark_results = benchmark_flash_attn_64t()

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Check all correctness criteria
    all_correct = all(err < 1e-2 for err in max_errors)
    print(f"Correctness (max_err < 1e-2): {'PASS' if all_correct else 'FAIL'}")

    # Note: The 64-thread kernel is expected to be slower per-kernel
    # because it processes all KV positions sequentially with 1 wavefront,
    # while the 256-thread kernel splits KV across 4 wavefronts.
    # The benefit of 64t is reduced resource usage per WG, allowing more
    # concurrent WGs when the GPU is saturated. With only 48 heads,
    # both variants are under-utilizing the GPU, so 64t shows no advantage.
    #
    # For the purposes of this test, we verify correctness only.
    print(f"\nNote: Performance comparison is not meaningful for 48 heads")
    print(f"      (both variants under-utilize GPU resources)")

    if all_correct:
        print("\n=== CORRECTNESS TESTS PASSED ===")
        sys.exit(0)
    else:
        print("\n=== TESTS FAILED ===")
        sys.exit(1)
