#!/usr/bin/env python3
"""Test and benchmark GEMV INT4 v7 vs v6."""
import sys, os, time, ctypes as ct
import numpy as np
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.runtime.hip_dispatch import GPUDevice

dev = GPUDevice(0)

# Build v7
print("Building v7 kernel...", flush=True)
ret = os.system("cd /opt/mi50grad && hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC "
                "-o build/kernels/gemv_int4_v7.so src/kernels/gemv_int4_v7.hip 2>&1")
if ret != 0:
    print("BUILD FAILED", flush=True)
    sys.exit(1)
print("Build done.", flush=True)

from src.inference.engine import KernelCache
kernels = KernelCache(dev)

v6_func = kernels.get_hip("gemv_int4_v6_t16", "gemv_int4_v6")
v7_func = kernels.get_hip("gemv_int4_v7_t16", "gemv_int4_v7")
print("Kernels loaded.", flush=True)

def run_gemv(func, d_A, d_B, d_scales, d_zeros, d_C, K, N, group_size, warmup=5, iters=200):
    cols_per_wg = 16
    grid_x = (N + cols_per_wg - 1) // cols_per_wg
    params = [
        ct.c_uint64(d_A), ct.c_uint64(d_B),
        ct.c_uint64(d_scales), ct.c_uint64(d_zeros),
        ct.c_uint64(d_C),
        ct.c_uint32(K), ct.c_uint32(N), ct.c_uint32(group_size),
    ]
    for _ in range(warmup):
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
    dev.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1e6
    return elapsed

def test_shape(K, N, group_size=128):
    print(f"\n--- K={K}, N={N}, group_size={group_size} ---")
    rng = np.random.default_rng(42)
    
    A = rng.standard_normal(K).astype(np.float16)
    n_groups = (K + group_size - 1) // group_size
    B_q4 = rng.integers(0, 0xFFFFFFFF, size=(K // 8, N), dtype=np.uint32)
    scales = (rng.standard_normal((n_groups, N)) * 0.01).astype(np.float16)
    zeros = (rng.standard_normal((n_groups, N)) * 0.1 + 8.0).astype(np.float16)
    
    d_A = dev.malloc(A.nbytes)
    d_B = dev.malloc(B_q4.nbytes)
    d_scales = dev.malloc(scales.nbytes)
    d_zeros = dev.malloc(zeros.nbytes)
    d_C_v6 = dev.malloc(N * 2)
    d_C_v7 = dev.malloc(N * 2)
    
    dev.upload(d_A, A.tobytes())
    dev.upload(d_B, B_q4.tobytes())
    dev.upload(d_scales, scales.tobytes())
    dev.upload(d_zeros, zeros.tobytes())
    
    cols_per_wg = 16
    grid_x = (N + cols_per_wg - 1) // cols_per_wg
    params_v6 = [
        ct.c_uint64(d_A), ct.c_uint64(d_B),
        ct.c_uint64(d_scales), ct.c_uint64(d_zeros),
        ct.c_uint64(d_C_v6),
        ct.c_uint32(K), ct.c_uint32(N), ct.c_uint32(group_size),
    ]
    params_v7 = [
        ct.c_uint64(d_A), ct.c_uint64(d_B),
        ct.c_uint64(d_scales), ct.c_uint64(d_zeros),
        ct.c_uint64(d_C_v7),
        ct.c_uint32(K), ct.c_uint32(N), ct.c_uint32(group_size),
    ]
    
    dev.launch(v6_func, (grid_x, 1, 1), (256, 1, 1), params_v6)
    dev.launch(v7_func, (grid_x, 1, 1), (256, 1, 1), params_v7)
    dev.synchronize()
    
    raw_v6 = dev.download(d_C_v6, N * 2)
    raw_v7 = dev.download(d_C_v7, N * 2)
    out_v6 = np.frombuffer(raw_v6, dtype=np.float16).copy()
    out_v7 = np.frombuffer(raw_v7, dtype=np.float16).copy()
    
    max_err = np.max(np.abs(out_v6.astype(np.float32) - out_v7.astype(np.float32)))
    cos_sim = np.dot(out_v6.astype(np.float32), out_v7.astype(np.float32)) / (
        np.linalg.norm(out_v6.astype(np.float32)) * np.linalg.norm(out_v7.astype(np.float32)) + 1e-10)
    
    print(f"  Correctness: max_abs_err={max_err:.6f}, cos_sim={cos_sim:.6f}")
    ok = max_err < 0.1 and cos_sim > 0.999
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    
    # Benchmark
    t_v6 = run_gemv(v6_func, d_A, d_B, d_scales, d_zeros, d_C_v6, K, N, group_size)
    t_v7 = run_gemv(v7_func, d_A, d_B, d_scales, d_zeros, d_C_v7, K, N, group_size)
    speedup = t_v6 / t_v7
    print(f"  v6: {t_v6:.1f} us, v7: {t_v7:.1f} us, speedup: {speedup:.3f}x")
    
    dev.free(d_A); dev.free(d_B); dev.free(d_scales); dev.free(d_zeros)
    dev.free(d_C_v6); dev.free(d_C_v7)
    return ok, speedup

# Test key dimensions
results = []
# Q/K/V projection (small N)
results.append(test_shape(K=5120, N=640))
# FFN gate/up (large N)
results.append(test_shape(K=5120, N=4352))
# FFN down (medium N)  
results.append(test_shape(K=4352, N=1280))
# Full hidden (attention input)
results.append(test_shape(K=5120, N=5120))

print("\n=== Summary ===")
all_pass = True
for i, (ok, spd) in enumerate(results):
    status = "PASS" if ok else "FAIL"
    print(f"  Test {i}: {status}, speedup={spd:.3f}x")
    if not ok:
        all_pass = False

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
