#!/usr/bin/env python3
"""Test harness for FlashAttention prefill kernel."""

import ctypes
import struct
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco

BUILD_DIR = PROJECT_ROOT / "build" / "kernels"


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def ensure_kernel(name):
    asm = PROJECT_ROOT / "src" / "asm" / f"{name}.s"
    hsaco = BUILD_DIR / f"{name}.hsaco"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not hsaco.exists() or asm.stat().st_mtime > hsaco.stat().st_mtime:
        print(f"Building {name}...")
        build_hsaco(str(asm), str(hsaco))
    return str(hsaco)


def reference_attention(Q, K, V, scale, num_heads, num_kv_heads, causal=False):
    """Compute reference attention in FP32.

    Q: [seq_len, num_heads, head_dim]
    K: [seq_len, num_kv_heads, head_dim]
    V: [seq_len, num_kv_heads, head_dim]

    Returns: [seq_len, num_heads, head_dim]
    """
    seq_len, _, head_dim = Q.shape
    gqa_ratio = num_heads // num_kv_heads

    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    V32 = V.astype(np.float32)

    out = np.zeros((seq_len, num_heads, head_dim), dtype=np.float32)

    for h in range(num_heads):
        kv_h = h // gqa_ratio
        q = Q32[:, h, :]          # [seq_len, head_dim]
        k = K32[:, kv_h, :]      # [seq_len, head_dim]
        v = V32[:, kv_h, :]      # [seq_len, head_dim]

        # scores[i,j] = q[i] dot k[j] * scale
        scores = (q @ k.T) * scale  # [seq_len, seq_len]

        if causal:
            # Mask out future positions: scores[i,j] = -inf if j > i
            mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
            scores = scores - mask * 1e9

        weights = softmax(scores, axis=-1)  # [seq_len, seq_len]
        out[:, h, :] = weights @ v           # [seq_len, head_dim]

    return out.astype(np.float16)


def test_flash_attn(seq_len, num_heads=4, num_kv_heads=4, head_dim=128,
                    causal=False, device_id=0):
    causal_str = "causal" if causal else "non-causal"
    print(f"\n{'='*60}")
    print(f"Testing FlashAttention Prefill ({causal_str}): seq={seq_len}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, dim={head_dim}")
    print(f"{'='*60}")

    assert head_dim == 128, "Only head_dim=128 supported"

    np.random.seed(42)
    Q = (np.random.randn(seq_len, num_heads, head_dim) * 0.1).astype(np.float16)
    K = (np.random.randn(seq_len, num_kv_heads, head_dim) * 0.1).astype(np.float16)
    V = (np.random.randn(seq_len, num_kv_heads, head_dim) * 0.1).astype(np.float16)

    scale = 1.0 / np.sqrt(head_dim)
    ref = reference_attention(Q, K, V, scale, num_heads, num_kv_heads, causal=causal)

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("flash_attn")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "flash_attn_fp16")

        d_Q = dev.malloc(Q.nbytes)
        d_K = dev.malloc(K.nbytes)
        d_V = dev.malloc(V.nbytes)
        out_size = seq_len * num_heads * head_dim * 2  # FP16
        d_out = dev.malloc(out_size)

        dev.upload(d_Q, Q.tobytes())
        dev.upload(d_K, K.tobytes())
        dev.upload(d_V, V.tobytes())
        dev.hip.memset(d_out, 0, out_size)

        # Pack scale as float32 bits
        scale_bits = struct.unpack('I', struct.pack('f', scale))[0]

        params = [
            ctypes.c_uint64(d_Q),
            ctypes.c_uint64(d_K),
            ctypes.c_uint64(d_V),
            ctypes.c_uint64(d_out),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(num_heads),
            ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(scale_bits),       # float as uint32 bits
            ctypes.c_uint32(1 if causal else 0),
        ]

        # Grid: (num_heads, ceil(seq_len/4), 1), Block: (256, 1, 1)
        grid_y = (seq_len + 3) // 4
        dev.launch(func, (num_heads, grid_y, 1), (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_out, out_size),
                            dtype=np.float16).reshape(seq_len, num_heads, head_dim)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")

        threshold = 1e-2
        if max_err < threshold:
            print(f"PASS (threshold={threshold})")
            return True
        else:
            print(f"FAIL (max_err={max_err} >= {threshold})")
            # Debug: show first row, first head
            print(f"ref[0,0,:4] = {ref[0,0,:4]}")
            print(f"out[0,0,:4] = {out[0,0,:4]}")
            # Show worst position
            idx = np.unravel_index(abs_err.argmax(), abs_err.shape)
            print(f"Worst error at position {idx}: ref={ref[idx]}, out={out[idx]}")
            return False
    finally:
        dev.cleanup()


if __name__ == "__main__":
    all_pass = True

    # Non-causal tests
    if not test_flash_attn(seq_len=128, num_heads=4, num_kv_heads=4, causal=False):
        all_pass = False

    if not test_flash_attn(seq_len=512, num_heads=4, num_kv_heads=4, causal=False):
        all_pass = False

    # Causal masking tests
    if not test_flash_attn(seq_len=128, num_heads=4, num_kv_heads=4, causal=True):
        all_pass = False

    if not test_flash_attn(seq_len=512, num_heads=4, num_kv_heads=4, causal=True):
        all_pass = False

    # GQA test (4:1 ratio) with causal
    if not test_flash_attn(seq_len=128, num_heads=8, num_kv_heads=2, causal=True):
        all_pass = False

    # Short sequence edge case
    if not test_flash_attn(seq_len=4, num_heads=4, num_kv_heads=4, causal=True):
        all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("All FlashAttention prefill tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
