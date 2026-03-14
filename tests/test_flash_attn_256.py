#!/usr/bin/env python3
"""Test flash_attn_256 HIP kernel against numpy reference."""

import sys
import time
import ctypes
import struct
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco


def numpy_attention(Q, K, V, causal=True):
    """Reference attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    seq_len, num_heads, head_dim = Q.shape
    num_kv_heads = K.shape[1]
    heads_per_kv = num_heads // num_kv_heads

    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros_like(Q, dtype=np.float32)

    for h in range(num_heads):
        kv_h = h * num_kv_heads // num_heads
        q = Q[:, h, :].astype(np.float32) * scale  # [seq, d]
        k = K[:, kv_h, :].astype(np.float32)        # [seq, d]
        v = V[:, kv_h, :].astype(np.float32)        # [seq, d]

        scores = q @ k.T  # [seq, seq]
        if causal:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores += mask

        # Stable softmax
        scores -= scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        out[:, h, :] = attn_weights @ v

    return out.astype(np.float16)


def test_flash_attn_256():
    dev = GPUDevice(0)

    BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    hip_path = str(PROJECT_ROOT / "src" / "kernels" / "flash_attn_256.hip")
    hsaco_path = str(BUILD_DIR / "flash_attn_256.hsaco")

    build_hip_hsaco(hip_path, hsaco_path)
    module = dev.load_hsaco(hsaco_path)
    func = dev.get_kernel(module, "flash_attn_256_fp16")

    test_configs = [
        # (seq_len, num_heads, num_kv_heads, causal)
        (4, 4, 4, True),
        (8, 4, 2, True),       # GQA
        (16, 4, 4, True),
        (32, 8, 2, True),
        (64, 4, 4, False),     # non-causal
        (128, 16, 4, True),    # close to real model
    ]

    all_pass = True
    for seq_len, num_heads, num_kv_heads, causal in test_configs:
        head_dim = 256
        np.random.seed(42)

        Q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float16) * 0.1
        K = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
        V = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1

        ref = numpy_attention(Q, K, V, causal=causal)

        # Upload
        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        d_Out = dev.malloc(ref.nbytes)
        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())
        dev.hip.memset(d_Out, 0, ref.nbytes)

        # Launch
        grid_x = num_heads
        grid_y = (seq_len + 3) // 4
        params = [
            ctypes.c_uint64(d_Q),
            ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V),
            ctypes.c_uint64(d_Out),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(1 if causal else 0),
        ]
        dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()

        result = np.frombuffer(dev.download(d_Out, ref.nbytes), dtype=np.float16).copy()
        result = result.reshape(seq_len, num_heads, head_dim)

        max_err = np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32)))
        cos_sim = np.dot(ref.ravel().astype(np.float32), result.ravel().astype(np.float32)) / (
            np.linalg.norm(ref.ravel().astype(np.float32)) *
            np.linalg.norm(result.ravel().astype(np.float32)) + 1e-10)

        ok = cos_sim > 0.99 and max_err < 0.1
        causal_str = "causal" if causal else "full"
        print(f"  seq={seq_len:3d} h={num_heads:2d} kv={num_kv_heads} {causal_str:6s}: "
              f"maxerr={max_err:.4f} cos={cos_sim:.6f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

        dev.free(d_Q)
        dev.free(d_K)
        dev.free(d_V)
        dev.free(d_Out)

    # Performance benchmark with model-like config
    print(f"\nPerformance benchmark:")
    for seq_len in [32, 64, 128, 256]:
        num_heads = 16
        num_kv_heads = 4
        head_dim = 256

        Q = np.random.randn(seq_len, num_heads, head_dim).astype(np.float16) * 0.1
        K = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1
        V = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float16) * 0.1

        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        d_Out = dev.malloc(Q.nbytes)
        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())

        grid_x = num_heads
        grid_y = (seq_len + 3) // 4
        params = [
            ctypes.c_uint64(d_Q), ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V), ctypes.c_uint64(d_Out),
            ctypes.c_uint32(seq_len), ctypes.c_uint32(seq_len),
            ctypes.c_uint32(num_heads), ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(1),
        ]

        # Warmup
        for _ in range(3):
            dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()

        # Benchmark
        iters = 20
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            dev.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()
        t_ms = (time.perf_counter() - t0) / iters * 1000

        print(f"  seq_len={seq_len:4d}: {t_ms:.2f}ms "
              f"({grid_x}x{grid_y} WGs = {grid_x*grid_y} total)")

        dev.free(d_Q)
        dev.free(d_K)
        dev.free(d_V)
        dev.free(d_Out)

    dev.cleanup()
    return all_pass


if __name__ == "__main__":
    ok = test_flash_attn_256()
    print(f"\n{'=== ALL TESTS PASSED ===' if ok else '=== SOME TESTS FAILED ==='}")
