#!/usr/bin/env python3
"""Test INT4 GEMV v3 (cooperative reduction) vs v2 (split-K)."""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine, KernelCache
from src.runtime.hip_dispatch import GPUDevice


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)

    loader = QwenWeightLoader(model_dir, config)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    embed = loader.load_embedding()
    engine.load_lm_head(loader.load_lm_head())

    # Warm up
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)
    engine.device.synchronize()

    # Load v3 kernels
    kernels = engine.kernels
    try:
        kernels.get_hip("gemv_int4_v3_t4", "gemv_int4_v3")
        kernels.get_hip("gemv_int4_v3_t8", "gemv_int4_v3")
        kernels.get_hip("gemv_int4_v3_t16", "gemv_int4_v3")
        print("INT4 GEMV v3 kernels compiled OK")
    except Exception as e:
        print(f"Failed to compile v3 kernels: {e}")
        engine.cleanup()
        return

    h = config.hidden_size
    N_inter = config.intermediate_size
    lw = engine.layers[0]
    group_size = config.group_size

    shapes = [
        ("gate/up", h, N_inter, lw.gate_qweight, lw.gate_scales, lw.gate_zeros),
        ("down", N_inter, h, lw.down_qweight, lw.down_scales, lw.down_zeros),
    ]

    iters = 100
    d_out = engine.d_ffn_gate  # Reuse output buffer

    for name, K, N, qw, sc, zr in shapes:
        print(f"\n{'='*60}")
        print(f"{name}: K={K}, N={N}")

        # Correctness: get reference from split-K
        engine.device.hip.memset(engine.d_gemv_fp32, 0, N * 4)
        func_sk = kernels.get_hip("gemv_int4_v2_splitk", "gemv_int4_v2")
        func_cv = kernels.get_hip("fp32_to_fp16", "gemv_int4_v2")
        grid_x = (N + 255) // 256
        engine.device.launch(func_sk, (grid_x, 16, 1), (256, 1, 1), [
            ctypes.c_uint64(engine.d_normed), ctypes.c_uint64(qw),
            ctypes.c_uint64(sc), ctypes.c_uint64(zr),
            ctypes.c_uint64(engine.d_gemv_fp32),
            ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
        ])
        engine.device.launch(func_cv, (grid_x, 1, 1), (256, 1, 1), [
            ctypes.c_uint64(engine.d_gemv_fp32), ctypes.c_uint64(engine.d_ffn_up),
            ctypes.c_uint32(N),
        ])
        engine.device.synchronize()
        ref = np.frombuffer(engine.device.download(engine.d_ffn_up, N * 2), dtype=np.float16).copy()

        # Test each v3 variant
        for tpc, kernel_name in [(4, "gemv_int4_v3_t4"), (8, "gemv_int4_v3_t8"),
                                  (16, "gemv_int4_v3_t16")]:
            cols_per_wg = 256 // tpc
            grid = (N + cols_per_wg - 1) // cols_per_wg
            func = kernels.get_hip(kernel_name, "gemv_int4_v3")

            params = [
                ctypes.c_uint64(engine.d_normed), ctypes.c_uint64(qw),
                ctypes.c_uint64(sc), ctypes.c_uint64(zr),
                ctypes.c_uint64(d_out),
                ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
            ]

            # Correctness
            engine.device.launch(func, (grid, 1, 1), (256, 1, 1), params)
            engine.device.synchronize()
            result = np.frombuffer(engine.device.download(d_out, N * 2), dtype=np.float16).copy()

            max_err = np.max(np.abs(ref.astype(np.float32) - result.astype(np.float32)))
            cos_sim = np.dot(ref.astype(np.float32), result.astype(np.float32)) / (
                np.linalg.norm(ref.astype(np.float32)) * np.linalg.norm(result.astype(np.float32)) + 1e-10)

            # Warmup
            for _ in range(5):
                engine.device.launch(func, (grid, 1, 1), (256, 1, 1), params)
            engine.device.synchronize()

            # Benchmark
            engine.device.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                engine.device.launch(func, (grid, 1, 1), (256, 1, 1), params)
            engine.device.synchronize()
            t_v3 = (time.perf_counter() - t0) / iters

            print(f"  v3_t{tpc}: {t_v3*1e6:.0f} us, WGs={grid}, cos={cos_sim:.6f}, maxerr={max_err:.4f}")

        # Split-K=16 reference timing
        for _ in range(5):
            engine.device.hip.memset(engine.d_gemv_fp32, 0, N * 4)
            engine.device.launch(func_sk, (grid_x, 16, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_normed), ctypes.c_uint64(qw),
                ctypes.c_uint64(sc), ctypes.c_uint64(zr),
                ctypes.c_uint64(engine.d_gemv_fp32),
                ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
            ])
            engine.device.launch(func_cv, (grid_x, 1, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_gemv_fp32), ctypes.c_uint64(engine.d_ffn_up),
                ctypes.c_uint32(N),
            ])
        engine.device.synchronize()

        engine.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            engine.device.hip.memset(engine.d_gemv_fp32, 0, N * 4)
            engine.device.launch(func_sk, (grid_x, 16, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_normed), ctypes.c_uint64(qw),
                ctypes.c_uint64(sc), ctypes.c_uint64(zr),
                ctypes.c_uint64(engine.d_gemv_fp32),
                ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size),
            ])
            engine.device.launch(func_cv, (grid_x, 1, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_gemv_fp32), ctypes.c_uint64(engine.d_ffn_up),
                ctypes.c_uint32(N),
            ])
        engine.device.synchronize()
        t_sk = (time.perf_counter() - t0) / iters

        # Theoretical
        weight_bytes = K * N // 2
        total_bytes = weight_bytes + K * 2 + N * 2
        theoretical = total_bytes / 857e9 * 1e6
        print(f"  split-K=16: {t_sk*1e6:.0f} us")
        print(f"  theoretical: {theoretical:.0f} us ({total_bytes/1e6:.1f}MB)")

    engine.cleanup()


if __name__ == "__main__":
    main()
