#!/usr/bin/env python3
"""
Test harness for elementwise_v3.hip kernels.

Tests:
1. Correctness of all 4 kernels at dim=5120 (Qwen hidden_size)
2. Bandwidth comparison (GB/s) of v3 vs v2 for residual_add, silu_fused, rmsnorm
3. Reports max abs error and bandwidth comparison

Kernels tested:
  residual_add_v3  — float4 loads, 8 FP16/thread
  silu_fused_v3    — float4 loads + polynomial sigmoid (no exp)
  rmsnorm_v3       — float4 loads for sum-sq and normalize phases
  skip_rmsnorm_v3  — float4 loads for add + sum-sq + normalize phases

Correctness thresholds:
  residual_add: max_err < 1e-4
  silu_fused:   max_err < 5e-3  (polynomial sigmoid approximation)
  rmsnorm:      max_err < 5e-3
  skip_rmsnorm: max_err < 5e-3
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

# ---- Build both kernels ----
BUILD_DIR = PROJECT_ROOT / "build" / "kernels"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

print("Building elementwise_v2 kernel...")
hip_v2 = str(PROJECT_ROOT / "src" / "kernels" / "elementwise_v2.hip")
hsaco_v2 = str(BUILD_DIR / "elementwise_v2.hsaco")
build_hip_hsaco(hip_v2, hsaco_v2)

print("Building elementwise_v3 kernel...")
hip_v3 = str(PROJECT_ROOT / "src" / "kernels" / "elementwise_v3.hip")
hsaco_v3 = str(BUILD_DIR / "elementwise_v3.hsaco")
build_hip_hsaco(hip_v3, hsaco_v3)

print("Kernels compiled successfully.\n")

dev = GPUDevice(0)
module_v2 = dev.load_hsaco(hsaco_v2)
module_v3 = dev.load_hsaco(hsaco_v3)

func_resadd_v2    = dev.get_kernel(module_v2, "residual_add_v2")
func_silu_v2      = dev.get_kernel(module_v2, "silu_fused_v2")
func_rms_v2       = dev.get_kernel(module_v2, "rmsnorm_v2")
func_skiprms_v2   = dev.get_kernel(module_v2, "skip_rmsnorm_v2")

func_resadd_v3    = dev.get_kernel(module_v3, "residual_add_v3")
func_silu_v3      = dev.get_kernel(module_v3, "silu_fused_v3")
func_rms_v3       = dev.get_kernel(module_v3, "rmsnorm_v3")
func_skiprms_v3   = dev.get_kernel(module_v3, "skip_rmsnorm_v3")

print("All kernels loaded.\n")

all_pass = True
WARMUP = 20
ITERS = 200


# ============================================================
# Helper: time a no-arg launch thunk
# ============================================================
def benchmark(launch_fn, warmup=WARMUP, iters=ITERS):
    """Run launch_fn warmup+iters times, return median latency in us."""
    latencies = []
    for _ in range(warmup):
        launch_fn()
    dev.synchronize()
    for _ in range(iters):
        t0 = time.perf_counter()
        launch_fn()
        dev.synchronize()
        latencies.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(latencies))


# ============================================================
# TEST 1: residual_add_v3 correctness + bandwidth
# ============================================================
def test_residual_add(dim=5120):
    print(f"{'='*60}")
    print(f"TEST: residual_add — dim={dim}")
    print(f"{'='*60}")

    np.random.seed(42)
    dst_np = (np.random.randn(dim) * 0.5).astype(np.float16)
    src_np = (np.random.randn(dim) * 0.5).astype(np.float16)

    # Reference (FP32 accumulation, then cast to FP16)
    ref = (dst_np.astype(np.float32) + src_np.astype(np.float32)).astype(np.float16)

    # Allocate GPU buffers
    d_dst_v2 = dev.malloc(dst_np.nbytes)
    d_dst_v3 = dev.malloc(dst_np.nbytes)
    d_src    = dev.malloc(src_np.nbytes)

    dev.upload(d_dst_v2, dst_np.tobytes())
    dev.upload(d_dst_v3, dst_np.tobytes())
    dev.upload(d_src,    src_np.tobytes())

    n = ctypes.c_uint32(dim)

    # v2: grid = (ceil(dim/512), 1, 1), block = (256, 1, 1)
    grid_v2 = ((dim + 511) // 512, 1, 1)
    params_v2 = [ctypes.c_uint64(d_dst_v2), ctypes.c_uint64(d_src), n]
    dev.launch(func_resadd_v2, grid_v2, (256, 1, 1), params_v2)
    dev.synchronize()

    # v3: grid = (ceil(dim/2048), 1, 1), block = (256, 1, 1)
    grid_v3 = ((dim + 2047) // 2048, 1, 1)
    params_v3 = [ctypes.c_uint64(d_dst_v3), ctypes.c_uint64(d_src), n]
    dev.launch(func_resadd_v3, grid_v3, (256, 1, 1), params_v3)
    dev.synchronize()

    out_v3 = np.frombuffer(dev.download(d_dst_v3, dst_np.nbytes), dtype=np.float16)
    max_err = float(np.abs(out_v3.astype(np.float32) - ref.astype(np.float32)).max())
    print(f"  Max abs error: {max_err:.6f}  (threshold: 1e-4)")

    global all_pass
    if max_err < 1e-4:
        print("  Correctness: PASS")
    else:
        print("  Correctness: FAIL")
        all_pass = False

    # Reset dst buffers to initial state for timing
    dev.upload(d_dst_v2, dst_np.tobytes())
    dev.upload(d_dst_v3, dst_np.tobytes())

    # Bandwidth benchmark
    # Bandwidth = 3 * dim * 2 bytes (read dst, read src, write dst)
    bytes_accessed = 3 * dim * 2

    def run_v2():
        dev.upload(d_dst_v2, dst_np.tobytes())
        dev.launch(func_resadd_v2, grid_v2, (256, 1, 1), params_v2)

    def run_v3():
        dev.upload(d_dst_v3, dst_np.tobytes())
        dev.launch(func_resadd_v3, grid_v3, (256, 1, 1), params_v3)

    lat_v2 = benchmark(run_v2)
    lat_v3 = benchmark(run_v3)

    bw_v2 = bytes_accessed / lat_v2 / 1e3  # GB/s
    bw_v3 = bytes_accessed / lat_v3 / 1e3  # GB/s
    speedup = lat_v2 / lat_v3

    print(f"  v2 latency: {lat_v2:.1f} us  ({bw_v2:.1f} GB/s)")
    print(f"  v3 latency: {lat_v3:.1f} us  ({bw_v3:.1f} GB/s)  speedup={speedup:.2f}x")
    print()
    return max_err < 1e-4


# ============================================================
# TEST 2: silu_fused_v3 correctness + bandwidth
# ============================================================
def test_silu_fused(dim=5120):
    print(f"{'='*60}")
    print(f"TEST: silu_fused — dim={dim}")
    print(f"{'='*60}")

    np.random.seed(7)
    # Use std=2.0 to stress-test the sigmoid across a range of values
    gate_np = (np.random.randn(dim) * 2.0).astype(np.float16)
    up_np   = (np.random.randn(dim) * 0.5).astype(np.float16)

    # Reference: exact sigmoid
    g32 = gate_np.astype(np.float32)
    u32 = up_np.astype(np.float32)
    ref = (g32 * (1.0 / (1.0 + np.exp(-g32))) * u32).astype(np.float16)

    d_gate_v2 = dev.malloc(gate_np.nbytes)
    d_gate_v3 = dev.malloc(gate_np.nbytes)
    d_up      = dev.malloc(up_np.nbytes)

    dev.upload(d_gate_v2, gate_np.tobytes())
    dev.upload(d_gate_v3, gate_np.tobytes())
    dev.upload(d_up,      up_np.tobytes())

    n = ctypes.c_uint32(dim)

    # v2: grid = (ceil(dim/512), 1, 1)
    grid_v2 = ((dim + 511) // 512, 1, 1)
    params_v2 = [ctypes.c_uint64(d_gate_v2), ctypes.c_uint64(d_up), n]
    dev.launch(func_silu_v2, grid_v2, (256, 1, 1), params_v2)
    dev.synchronize()

    # v3: grid = (ceil(dim/2048), 1, 1)
    grid_v3 = ((dim + 2047) // 2048, 1, 1)
    params_v3 = [ctypes.c_uint64(d_gate_v3), ctypes.c_uint64(d_up), n]
    dev.launch(func_silu_v3, grid_v3, (256, 1, 1), params_v3)
    dev.synchronize()

    out_v3 = np.frombuffer(dev.download(d_gate_v3, gate_np.nbytes), dtype=np.float16)
    max_err = float(np.abs(out_v3.astype(np.float32) - ref.astype(np.float32)).max())
    print(f"  Max abs error vs exact sigmoid: {max_err:.6f}  (threshold: 5e-3)")

    global all_pass
    if max_err < 5e-3:
        print("  Correctness: PASS")
    else:
        print(f"  Correctness: FAIL  (max_err={max_err:.6f})")
        all_pass = False

    # Bandwidth benchmark
    # Bandwidth = 3 * dim * 2 bytes (read gate, read up, write gate)
    bytes_accessed = 3 * dim * 2

    def run_v2():
        dev.upload(d_gate_v2, gate_np.tobytes())
        dev.launch(func_silu_v2, grid_v2, (256, 1, 1), params_v2)

    def run_v3():
        dev.upload(d_gate_v3, gate_np.tobytes())
        dev.launch(func_silu_v3, grid_v3, (256, 1, 1), params_v3)

    lat_v2 = benchmark(run_v2)
    lat_v3 = benchmark(run_v3)

    bw_v2 = bytes_accessed / lat_v2 / 1e3
    bw_v3 = bytes_accessed / lat_v3 / 1e3
    speedup = lat_v2 / lat_v3

    print(f"  v2 latency: {lat_v2:.1f} us  ({bw_v2:.1f} GB/s)")
    print(f"  v3 latency: {lat_v3:.1f} us  ({bw_v3:.1f} GB/s)  speedup={speedup:.2f}x")
    print()
    return max_err < 5e-3


# ============================================================
# TEST 3: rmsnorm_v3 correctness + bandwidth
# ============================================================
def test_rmsnorm(dim=5120, num_vecs=4, eps=1e-5):
    print(f"{'='*60}")
    print(f"TEST: rmsnorm — dim={dim}, num_vecs={num_vecs}")
    print(f"{'='*60}")

    np.random.seed(13)
    src_np = (np.random.randn(num_vecs, dim) * 0.5).astype(np.float16)
    w_np   = (np.random.randn(dim) * 0.1 + 1.0).astype(np.float16)

    # Reference (FP32)
    s32 = src_np.astype(np.float32)
    w32 = w_np.astype(np.float32)
    rms = np.sqrt(np.mean(s32 ** 2, axis=1, keepdims=True) + eps)
    ref = ((s32 / rms) * w32).astype(np.float16)

    d_src_v2 = dev.malloc(src_np.nbytes)
    d_src_v3 = dev.malloc(src_np.nbytes)
    d_w      = dev.malloc(w_np.nbytes)
    d_out_v2 = dev.malloc(src_np.nbytes)
    d_out_v3 = dev.malloc(src_np.nbytes)

    dev.upload(d_src_v2, src_np.tobytes())
    dev.upload(d_src_v3, src_np.tobytes())
    dev.upload(d_w,      w_np.tobytes())

    import struct
    eps_bits = ctypes.c_uint32(struct.unpack('<I', struct.pack('<f', eps))[0])
    _dim = ctypes.c_uint32(dim)

    # v2 launch
    params_v2 = [ctypes.c_uint64(d_out_v2), ctypes.c_uint64(d_src_v2),
                 ctypes.c_uint64(d_w), _dim, eps_bits]
    dev.hip.memset(d_out_v2, 0, src_np.nbytes)
    dev.launch(func_rms_v2, (num_vecs, 1, 1), (256, 1, 1), params_v2)
    dev.synchronize()

    # v3 launch (same interface)
    params_v3 = [ctypes.c_uint64(d_out_v3), ctypes.c_uint64(d_src_v3),
                 ctypes.c_uint64(d_w), _dim, eps_bits]
    dev.hip.memset(d_out_v3, 0, src_np.nbytes)
    dev.launch(func_rms_v3, (num_vecs, 1, 1), (256, 1, 1), params_v3)
    dev.synchronize()

    out_v3 = np.frombuffer(dev.download(d_out_v3, src_np.nbytes),
                           dtype=np.float16).reshape(src_np.shape)
    max_err = float(np.abs(out_v3.astype(np.float32) - ref.astype(np.float32)).max())
    print(f"  Max abs error: {max_err:.6f}  (threshold: 5e-3)")

    global all_pass
    if max_err < 5e-3:
        print("  Correctness: PASS")
    else:
        print(f"  Correctness: FAIL  (max_err={max_err:.6f})")
        print(f"  ref[0,:8] = {ref[0,:8]}")
        print(f"  out[0,:8] = {out_v3[0,:8]}")
        all_pass = False

    # Bandwidth benchmark (single vector = dim=5120 for timing clarity)
    single_src = src_np[0:1]
    d_ssrc_v2 = dev.malloc(single_src.nbytes)
    d_ssrc_v3 = dev.malloc(single_src.nbytes)
    d_sout_v2 = dev.malloc(single_src.nbytes)
    d_sout_v3 = dev.malloc(single_src.nbytes)
    dev.upload(d_ssrc_v2, single_src.tobytes())
    dev.upload(d_ssrc_v3, single_src.tobytes())

    # Bandwidth = read src + read weight + write dst = 3 * dim * 2 bytes
    bytes_accessed = 3 * dim * 2

    sp_v2 = [ctypes.c_uint64(d_sout_v2), ctypes.c_uint64(d_ssrc_v2),
             ctypes.c_uint64(d_w), _dim, eps_bits]
    sp_v3 = [ctypes.c_uint64(d_sout_v3), ctypes.c_uint64(d_ssrc_v3),
             ctypes.c_uint64(d_w), _dim, eps_bits]

    def run_v2():
        dev.launch(func_rms_v2, (1, 1, 1), (256, 1, 1), sp_v2)

    def run_v3():
        dev.launch(func_rms_v3, (1, 1, 1), (256, 1, 1), sp_v3)

    lat_v2 = benchmark(run_v2)
    lat_v3 = benchmark(run_v3)

    bw_v2 = bytes_accessed / lat_v2 / 1e3
    bw_v3 = bytes_accessed / lat_v3 / 1e3
    speedup = lat_v2 / lat_v3

    print(f"  v2 latency: {lat_v2:.1f} us  ({bw_v2:.1f} GB/s)")
    print(f"  v3 latency: {lat_v3:.1f} us  ({bw_v3:.1f} GB/s)  speedup={speedup:.2f}x")
    print()
    return max_err < 5e-3


# ============================================================
# TEST 4: skip_rmsnorm_v3 correctness
# ============================================================
def test_skip_rmsnorm(dim=5120, eps=1e-5):
    print(f"{'='*60}")
    print(f"TEST: skip_rmsnorm — dim={dim}")
    print(f"{'='*60}")

    np.random.seed(99)
    hidden_np   = (np.random.randn(dim) * 0.5).astype(np.float16)
    residual_np = (np.random.randn(dim) * 0.5).astype(np.float16)
    weight_np   = (np.random.randn(dim) * 0.1 + 1.0).astype(np.float16)

    # Reference
    combined = (hidden_np.astype(np.float32) + residual_np.astype(np.float32))
    rms = np.sqrt(np.mean(combined ** 2) + eps)
    ref_dst = ((combined / rms) * weight_np.astype(np.float32)).astype(np.float16)
    ref_hidden = combined.astype(np.float16)

    d_dst_v2    = dev.malloc(hidden_np.nbytes)
    d_dst_v3    = dev.malloc(hidden_np.nbytes)
    d_hidden_v2 = dev.malloc(hidden_np.nbytes)
    d_hidden_v3 = dev.malloc(hidden_np.nbytes)
    d_residual  = dev.malloc(residual_np.nbytes)
    d_weight    = dev.malloc(weight_np.nbytes)

    dev.upload(d_hidden_v2, hidden_np.tobytes())
    dev.upload(d_hidden_v3, hidden_np.tobytes())
    dev.upload(d_residual,  residual_np.tobytes())
    dev.upload(d_weight,    weight_np.tobytes())

    import struct
    eps_bits = ctypes.c_uint32(struct.unpack('<I', struct.pack('<f', eps))[0])
    _dim = ctypes.c_uint32(dim)

    # v2 launch
    params_v2 = [ctypes.c_uint64(d_dst_v2), ctypes.c_uint64(d_hidden_v2),
                 ctypes.c_uint64(d_residual), ctypes.c_uint64(d_weight), _dim, eps_bits]
    dev.launch(func_skiprms_v2, (1, 1, 1), (256, 1, 1), params_v2)
    dev.synchronize()

    # v3 launch
    params_v3 = [ctypes.c_uint64(d_dst_v3), ctypes.c_uint64(d_hidden_v3),
                 ctypes.c_uint64(d_residual), ctypes.c_uint64(d_weight), _dim, eps_bits]
    dev.launch(func_skiprms_v3, (1, 1, 1), (256, 1, 1), params_v3)
    dev.synchronize()

    # Check dst output
    out_dst_v3    = np.frombuffer(dev.download(d_dst_v3,    hidden_np.nbytes), dtype=np.float16)
    out_hidden_v3 = np.frombuffer(dev.download(d_hidden_v3, hidden_np.nbytes), dtype=np.float16)

    max_err_dst    = float(np.abs(out_dst_v3.astype(np.float32) -
                                  ref_dst.astype(np.float32)).max())
    max_err_hidden = float(np.abs(out_hidden_v3.astype(np.float32) -
                                   ref_hidden.astype(np.float32)).max())

    print(f"  Max abs error (normalized dst): {max_err_dst:.6f}    (threshold: 5e-3)")
    print(f"  Max abs error (updated hidden): {max_err_hidden:.6f}  (threshold: 1e-4)")

    global all_pass
    passed = (max_err_dst < 5e-3) and (max_err_hidden < 1e-4)
    if passed:
        print("  Correctness: PASS")
    else:
        if max_err_dst >= 5e-3:
            print(f"  FAIL: dst max_err={max_err_dst:.6f} >= 5e-3")
        if max_err_hidden >= 1e-4:
            print(f"  FAIL: hidden max_err={max_err_hidden:.6f} >= 1e-4")
        all_pass = False
    print()
    return passed


# ============================================================
# TEST 5: Additional edge cases (smaller/non-power-of-2 dims)
# ============================================================
def test_residual_add_edge_cases():
    print(f"{'='*60}")
    print("TEST: residual_add — edge cases (dim=7, dim=16, dim=1024)")
    print(f"{'='*60}")

    global all_pass
    for dim in [7, 16, 256, 1024, 4096]:
        np.random.seed(dim)
        dst_np = (np.random.randn(dim) * 0.5).astype(np.float16)
        src_np = (np.random.randn(dim) * 0.5).astype(np.float16)
        ref = (dst_np.astype(np.float32) + src_np.astype(np.float32)).astype(np.float16)

        d_dst = dev.malloc(dst_np.nbytes)
        d_src = dev.malloc(src_np.nbytes)
        dev.upload(d_dst, dst_np.tobytes())
        dev.upload(d_src, src_np.tobytes())

        grid_v3 = ((dim + 2047) // 2048, 1, 1)
        params = [ctypes.c_uint64(d_dst), ctypes.c_uint64(d_src), ctypes.c_uint32(dim)]
        dev.launch(func_resadd_v3, grid_v3, (256, 1, 1), params)
        dev.synchronize()

        out = np.frombuffer(dev.download(d_dst, dst_np.nbytes), dtype=np.float16)
        max_err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())

        status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  dim={dim:5d}: max_err={max_err:.2e}  {status}")
        if max_err >= 1e-4:
            all_pass = False
    print()


# ============================================================
# Main: run all tests
# ============================================================
if __name__ == "__main__":
    DIM = 5120  # Qwen hidden_size

    p1 = test_residual_add(dim=DIM)
    p2 = test_silu_fused(dim=DIM)
    p3 = test_rmsnorm(dim=DIM, num_vecs=4)
    p4 = test_skip_rmsnorm(dim=DIM)
    test_residual_add_edge_cases()

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
