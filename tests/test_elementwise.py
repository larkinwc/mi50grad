#!/usr/bin/env python3
"""Test harness for elementwise kernels: RMSNorm, SiLU, RoPE."""

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


def ensure_kernel(name):
    """Build kernel if needed."""
    asm = PROJECT_ROOT / "src" / "asm" / f"{name}.s"
    hsaco = BUILD_DIR / f"{name}.hsaco"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not hsaco.exists() or asm.stat().st_mtime > hsaco.stat().st_mtime:
        print(f"Building {name}...")
        build_hsaco(str(asm), str(hsaco))
    return str(hsaco)


def test_rmsnorm(hidden_dim=4096, num_rows=4, eps=1e-6, device_id=0):
    print(f"\n{'='*60}")
    print(f"Testing RMSNorm: rows={num_rows}, hidden_dim={hidden_dim}")
    print(f"{'='*60}")

    assert hidden_dim % 256 == 0

    np.random.seed(42)
    x = (np.random.randn(num_rows, hidden_dim) * 0.5).astype(np.float16)
    w = (np.random.randn(hidden_dim) * 0.1 + 1.0).astype(np.float16)

    # Reference (FP32)
    x32 = x.astype(np.float32)
    w32 = w.astype(np.float32)
    rms = np.sqrt(np.mean(x32 ** 2, axis=1, keepdims=True) + eps)
    ref = ((x32 / rms) * w32).astype(np.float16)

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("rmsnorm")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "rmsnorm_fp16")

        d_in = dev.malloc(x.nbytes)
        d_w = dev.malloc(w.nbytes)
        d_out = dev.malloc(x.nbytes)

        dev.upload(d_in, x.tobytes())
        dev.upload(d_w, w.tobytes())
        dev.hip.memset(d_out, 0, x.nbytes)

        eps_bits = struct.unpack('<I', struct.pack('<f', eps))[0]

        params = [
            ctypes.c_uint64(d_in),
            ctypes.c_uint64(d_w),
            ctypes.c_uint64(d_out),
            ctypes.c_uint32(hidden_dim),
            ctypes.c_uint32(eps_bits),  # pass float as uint32 bits
        ]

        grid = (num_rows, 1, 1)
        block = (256, 1, 1)

        dev.launch(func, grid, block, params, shared_mem=16)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_out, x.nbytes), dtype=np.float16).reshape(x.shape)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")

        if max_err < 0.05:
            print("PASS")
            return True
        else:
            print(f"FAIL (max_err={max_err})")
            print(f"ref[0,:4] = {ref[0,:4]}")
            print(f"out[0,:4] = {out[0,:4]}")
            return False
    finally:
        dev.cleanup()


def test_silu(n=4096, fused=False, device_id=0):
    mode = "fused SiLU*up" if fused else "SiLU"
    print(f"\n{'='*60}")
    print(f"Testing {mode}: n={n}")
    print(f"{'='*60}")

    np.random.seed(42)
    gate = (np.random.randn(n) * 0.5).astype(np.float16)
    up = (np.random.randn(n) * 0.5).astype(np.float16) if fused else None

    # Reference
    g32 = gate.astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-g32))
    silu_out = g32 * sigmoid
    if fused:
        silu_out = silu_out * up.astype(np.float32)
    ref = silu_out.astype(np.float16)

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("silu")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "silu_fp16")

        d_gate = dev.malloc(gate.nbytes)
        d_up = dev.malloc(up.nbytes) if fused else 0
        d_out = dev.malloc(gate.nbytes)

        dev.upload(d_gate, gate.tobytes())
        if fused:
            dev.upload(d_up, up.tobytes())

        params = [
            ctypes.c_uint64(d_gate),
            ctypes.c_uint64(d_up),
            ctypes.c_uint64(d_out),
            ctypes.c_uint32(n),
        ]

        grid_x = (n + 255) // 256
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_out, gate.nbytes), dtype=np.float16)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")

        if max_err < 0.01:
            print("PASS")
            return True
        else:
            print(f"FAIL (max_err={max_err})")
            print(f"ref[:4] = {ref[:4]}")
            print(f"out[:4] = {out[:4]}")
            return False
    finally:
        dev.cleanup()


def test_rope(num_tokens=4, num_heads=8, head_dim=128, base=1e6, device_id=0):
    print(f"\n{'='*60}")
    print(f"Testing RoPE: tokens={num_tokens}, heads={num_heads}, dim={head_dim}")
    print(f"{'='*60}")

    assert head_dim % 2 == 0
    half_dim = head_dim // 2

    np.random.seed(42)
    x = (np.random.randn(num_tokens, num_heads, head_dim) * 0.5).astype(np.float16)

    # Compute cos/sin tables
    freqs = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) * 2.0 / head_dim))
    positions = np.arange(num_tokens, dtype=np.float32)
    angles = np.outer(positions, freqs)  # [num_tokens, half_dim]
    cos_tab = np.cos(angles).astype(np.float16)
    sin_tab = np.sin(angles).astype(np.float16)

    # Reference
    x32 = x.astype(np.float32)
    cos32 = cos_tab.astype(np.float32)
    sin32 = sin_tab.astype(np.float32)
    ref = x32.copy()
    for t in range(num_tokens):
        for h in range(num_heads):
            for i in range(half_dim):
                x0 = x32[t, h, 2*i]
                x1 = x32[t, h, 2*i+1]
                c = cos32[t, i]
                s = sin32[t, i]
                ref[t, h, 2*i]   = x0 * c - x1 * s
                ref[t, h, 2*i+1] = x0 * s + x1 * c
    ref = ref.astype(np.float16)

    dev = GPUDevice(device_id)
    try:
        hsaco = ensure_kernel("rope")
        module = dev.load_hsaco(hsaco)
        func = dev.get_kernel(module, "rope_fp16")

        d_x = dev.malloc(x.nbytes)
        d_cos = dev.malloc(cos_tab.nbytes)
        d_sin = dev.malloc(sin_tab.nbytes)

        dev.upload(d_x, x.tobytes())
        dev.upload(d_cos, cos_tab.tobytes())
        dev.upload(d_sin, sin_tab.tobytes())

        params = [
            ctypes.c_uint64(d_x),
            ctypes.c_uint64(d_cos),
            ctypes.c_uint64(d_sin),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(num_heads),
        ]

        # Grid: (num_tokens, num_heads, 1), Block: (head_dim/2, 1, 1)
        dev.launch(func, (num_tokens, num_heads, 1), (half_dim, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_x, x.nbytes), dtype=np.float16).reshape(x.shape)

        abs_err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        print(f"Max absolute error: {max_err:.6f}")
        print(f"Mean absolute error: {mean_err:.6f}")

        if max_err < 0.01:
            print("PASS")
            return True
        else:
            print(f"FAIL (max_err={max_err})")
            print(f"ref[0,0,:4] = {ref[0,0,:4]}")
            print(f"out[0,0,:4] = {out[0,0,:4]}")
            return False
    finally:
        dev.cleanup()


if __name__ == "__main__":
    all_pass = True

    # RMSNorm tests
    for dim in [256, 1024, 4096]:
        if not test_rmsnorm(hidden_dim=dim, num_rows=2):
            all_pass = False

    # SiLU tests
    if not test_silu(n=1024, fused=False):
        all_pass = False
    if not test_silu(n=4096, fused=True):
        all_pass = False

    # RoPE tests
    if not test_rope(num_tokens=4, num_heads=8, head_dim=128):
        all_pass = False
    if not test_rope(num_tokens=1, num_heads=32, head_dim=128):
        all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("All elementwise tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
