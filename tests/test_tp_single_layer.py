#!/usr/bin/env python3
"""Stage 5 verification: single-layer TP on 2 GPUs matches single-GPU output."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.runtime.hip_dispatch import GPUDevice


def gemv_fp16_numpy(x, W):
    """Reference FP16 GEMV: out = W @ x"""
    return (W.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)


def test_column_parallel_gemv():
    """Test column-parallel GEMV: split N dimension across 2 GPUs.

    Each GPU computes half the output, results concatenated.
    """
    print("Test: Column-parallel FP16 GEMV (2 GPUs)")

    K, N = 5120, 6144
    x = np.random.randn(K).astype(np.float16) * 0.1
    W = np.random.randn(N, K).astype(np.float16) * 0.01

    # Reference: single GPU
    ref = gemv_fp16_numpy(x, W)

    # TP=2: split W along rows (N dimension)
    N_half = N // 2
    W0 = W[:N_half, :].copy()  # GPU 0 gets first half
    W1 = W[N_half:, :].copy()  # GPU 1 gets second half

    dev0 = GPUDevice(0)
    dev1 = GPUDevice(1)

    # Upload x to both GPUs (replicated)
    d_x0 = dev0.malloc(x.nbytes)
    dev0.upload(d_x0, x.tobytes())
    d_x1 = dev1.malloc(x.nbytes)
    dev1.upload(d_x1, x.tobytes())

    # Upload sharded W
    d_W0 = dev0.malloc(W0.nbytes)
    dev0.upload(d_W0, W0.tobytes())
    d_W1 = dev1.malloc(W1.nbytes)
    dev1.upload(d_W1, W1.tobytes())

    # Allocate outputs
    d_out0 = dev0.malloc(N_half * 2)
    d_out1 = dev1.malloc(N_half * 2)

    # Build and launch GEMV on each GPU
    from src.kernels.launcher import build_hip_hsaco, Kernel
    import ctypes

    BUILD_DIR = Path(__file__).parent.parent / "build" / "kernels"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    hip_path = str(Path(__file__).parent.parent / "src" / "kernels" / "gemv_fp16_v2.hip")
    hsaco0 = str(BUILD_DIR / "gemv_fp16_v2_tp0.hsaco")
    hsaco1 = str(BUILD_DIR / "gemv_fp16_v2_tp1.hsaco")

    build_hip_hsaco(hip_path, hsaco0)
    build_hip_hsaco(hip_path, hsaco1)

    mod0 = dev0.load_hsaco(hsaco0)
    func0 = dev0.get_kernel(mod0, "gemv_fp16_v2")
    mod1 = dev1.load_hsaco(hsaco1)
    func1 = dev1.get_kernel(mod1, "gemv_fp16_v2")

    # Launch on GPU 0
    params0 = [
        ctypes.c_uint64(d_x0),
        ctypes.c_uint64(d_W0),
        ctypes.c_uint64(d_out0),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N_half),
        ctypes.c_uint64(0),  # residual=null
    ]
    grid0 = ((N_half + 3) // 4, 1, 1)
    dev0.launch(func0, grid0, (256, 1, 1), params0)
    dev0.synchronize()

    # Launch on GPU 1
    params1 = [
        ctypes.c_uint64(d_x1),
        ctypes.c_uint64(d_W1),
        ctypes.c_uint64(d_out1),
        ctypes.c_uint32(K),
        ctypes.c_uint32(N_half),
        ctypes.c_uint64(0),  # residual=null
    ]
    grid1 = ((N_half + 3) // 4, 1, 1)
    dev1.launch(func1, grid1, (256, 1, 1), params1)
    dev1.synchronize()

    # Download and concatenate
    out0 = np.frombuffer(dev0.download(d_out0, N_half * 2), dtype=np.float16).copy()
    out1 = np.frombuffer(dev1.download(d_out1, N_half * 2), dtype=np.float16).copy()
    tp_result = np.concatenate([out0, out1])

    # Compare
    max_err = np.max(np.abs(ref.astype(np.float32) - tp_result.astype(np.float32)))
    cos_sim = np.dot(ref.astype(np.float32), tp_result.astype(np.float32)) / (
        np.linalg.norm(ref.astype(np.float32)) * np.linalg.norm(tp_result.astype(np.float32)) + 1e-10)

    print(f"  Max error: {max_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    ok = cos_sim > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")

    dev0.cleanup()
    dev1.cleanup()
    return ok


def test_row_parallel_gemv_with_allreduce():
    """Test row-parallel GEMV: split K dimension, then allreduce.

    Each GPU has half the input dimension. After GEMV, allreduce sums partial results.
    """
    print("\nTest: Row-parallel FP16 GEMV + allreduce (2 GPUs)")

    K, N = 5120, 5120
    x = np.random.randn(K).astype(np.float16) * 0.1
    W = np.random.randn(N, K).astype(np.float16) * 0.01

    # Reference
    ref = gemv_fp16_numpy(x, W)

    # TP=2: split W and x along K dimension
    K_half = K // 2
    W0 = W[:, :K_half].copy()  # GPU 0: W[:, :K/2]
    W1 = W[:, K_half:].copy()  # GPU 1: W[:, K/2:]
    x0 = x[:K_half].copy()
    x1 = x[K_half:].copy()

    dev0 = GPUDevice(0)
    dev1 = GPUDevice(1)

    # Upload
    d_x0 = dev0.malloc(x0.nbytes)
    dev0.upload(d_x0, x0.tobytes())
    d_x1 = dev1.malloc(x1.nbytes)
    dev1.upload(d_x1, x1.tobytes())

    d_W0 = dev0.malloc(W0.nbytes)
    dev0.upload(d_W0, W0.tobytes())
    d_W1 = dev1.malloc(W1.nbytes)
    dev1.upload(d_W1, W1.tobytes())

    d_out0 = dev0.malloc(N * 2)
    d_out1 = dev1.malloc(N * 2)

    # Build kernels
    from src.kernels.launcher import build_hip_hsaco
    import ctypes

    BUILD_DIR = Path(__file__).parent.parent / "build" / "kernels"
    hip_path = str(Path(__file__).parent.parent / "src" / "kernels" / "gemv_fp16_v2.hip")
    hsaco0 = str(BUILD_DIR / "gemv_fp16_v2_tp0.hsaco")

    mod0 = dev0.load_hsaco(hsaco0)
    func0 = dev0.get_kernel(mod0, "gemv_fp16_v2")
    mod1 = dev1.load_hsaco(hsaco0)  # Same binary, different device
    func1 = dev1.get_kernel(mod1, "gemv_fp16_v2")

    # Launch on both GPUs (each computes partial sum)
    params0 = [ctypes.c_uint64(d_x0), ctypes.c_uint64(d_W0), ctypes.c_uint64(d_out0),
               ctypes.c_uint32(K_half), ctypes.c_uint32(N), ctypes.c_uint64(0)]
    params1 = [ctypes.c_uint64(d_x1), ctypes.c_uint64(d_W1), ctypes.c_uint64(d_out1),
               ctypes.c_uint32(K_half), ctypes.c_uint32(N), ctypes.c_uint64(0)]

    grid = ((N + 3) // 4, 1, 1)
    dev0.launch(func0, grid, (256, 1, 1), params0)
    dev1.launch(func1, grid, (256, 1, 1), params1)
    dev0.synchronize()
    dev1.synchronize()

    # Allreduce: download, sum, upload
    out0_data = np.frombuffer(dev0.download(d_out0, N * 2), dtype=np.float16).copy()
    out1_data = np.frombuffer(dev1.download(d_out1, N * 2), dtype=np.float16).copy()
    summed = (out0_data.astype(np.float32) + out1_data.astype(np.float32)).astype(np.float16)
    dev0.upload(d_out0, summed.tobytes())
    dev1.upload(d_out1, summed.tobytes())

    # Verify
    tp_result = np.frombuffer(dev0.download(d_out0, N * 2), dtype=np.float16).copy()

    max_err = np.max(np.abs(ref.astype(np.float32) - tp_result.astype(np.float32)))
    cos_sim = np.dot(ref.astype(np.float32), tp_result.astype(np.float32)) / (
        np.linalg.norm(ref.astype(np.float32)) * np.linalg.norm(tp_result.astype(np.float32)) + 1e-10)

    print(f"  Max error: {max_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    ok = cos_sim > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")

    dev0.cleanup()
    dev1.cleanup()
    return ok


def main():
    ok1 = test_column_parallel_gemv()
    ok2 = test_row_parallel_gemv_with_allreduce()

    if ok1 and ok2:
        print("\n=== Stage 5 Verification: PASSED ===")
        print("Column-parallel and row-parallel GEMV with allreduce work correctly on 2 GPUs.")
    else:
        print("\n=== Stage 5 Verification: FAILED ===")


if __name__ == "__main__":
    main()
