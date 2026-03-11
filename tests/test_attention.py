#!/usr/bin/env python3
"""Test harness for attention kernels."""

import ctypes
import sys
import numpy as np
from pathlib import Path


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"


def ensure_kernel(name):
    asm = PROJECT_ROOT / "src" / "asm" / f"{name}.s"
    hsaco = BUILD_DIR / f"{name}.hsaco"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not hsaco.exists() or asm.stat().st_mtime > hsaco.stat().st_mtime:
        print(f"Building {name}...")
        build_hsaco(str(asm), str(hsaco))
    return str(hsaco)


def test_decode_attn(batch=1, seq_len=16, num_heads=4, num_kv_heads=4,
                     head_dim=128, device_id=0):
    print(f"\n{'='*60}")
    print(f"Testing Decode Attention: batch={batch}, seq={seq_len}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, dim={head_dim}")
    print(f"{'='*60}")

    assert head_dim <= 128  # max 64 threads * 2 dims

    np.random.seed(42)
    Q = (np.random.randn(batch, num_heads, head_dim) * 0.1).astype(np.float16)
    K = (np.random.randn(batch, seq_len, num_kv_heads, head_dim) * 0.1).astype(np.float16)
    V = (np.random.randn(batch, seq_len, num_kv_heads, head_dim) * 0.1).astype(np.float16)

    # Reference (FP32)
    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    V32 = V.astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)

    ref = np.zeros((batch, num_heads, head_dim), dtype=np.float32)
    gqa_ratio = num_heads // num_kv_heads

    for b in range(batch):
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            q = Q32[b, h]  # [head_dim]
            k = K32[b, :, kv_h]  # [seq_len, head_dim]
            v = V32[b, :, kv_h]  # [seq_len, head_dim]

            scores = (q @ k.T) * scale  # [seq_len]
            weights = softmax(scores)  # [seq_len]
            ref[b, h] = weights @ v  # [head_dim]

    ref = ref.astype(np.float16)

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("decode_attn")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "decode_attn_fp16")

        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        out_size = batch * num_heads * head_dim * 2
        d_out = dev.malloc(out_size)

        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())
        dev.hip.memset(d_out, 0, out_size)

        params = [
            ctypes.c_uint64(d_Q),
            ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V),
            ctypes.c_uint64(d_out),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
        ]

        # Grid: (num_heads, batch, 1), Block: (256, 1, 1)
        # (only first 64 threads do work)
        dev.launch(func, (num_heads, batch, 1), (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_out, out_size),
                            dtype=np.float16).reshape(batch, num_heads, head_dim)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")

        # FP16 precision: shorter sequences have larger relative errors
        threshold = 0.05
        if max_err < threshold:
            print(f"PASS (threshold={threshold})")
            return True
        else:
            print(f"FAIL (max_err={max_err} >= {threshold})")
            print(f"ref[0,0,:4] = {ref[0,0,:4]}")
            print(f"out[0,0,:4] = {out[0,0,:4]}")
            return False
    finally:
        dev.cleanup()


if __name__ == "__main__":
    all_pass = True

    # Basic tests
    if not test_decode_attn(batch=1, seq_len=4, num_heads=4, num_kv_heads=4, head_dim=128):
        all_pass = False
    if not test_decode_attn(batch=1, seq_len=16, num_heads=4, num_kv_heads=4, head_dim=128):
        all_pass = False
    if not test_decode_attn(batch=1, seq_len=64, num_heads=4, num_kv_heads=4, head_dim=128):
        all_pass = False

    # GQA test (4:1 ratio)
    if not test_decode_attn(batch=1, seq_len=16, num_heads=8, num_kv_heads=2, head_dim=128):
        all_pass = False

    # Longer sequence
    if not test_decode_attn(batch=1, seq_len=256, num_heads=4, num_kv_heads=4, head_dim=128):
        all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("All attention tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
