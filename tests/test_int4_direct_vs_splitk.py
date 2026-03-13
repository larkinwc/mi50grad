#!/usr/bin/env python3
"""Compare INT4 GEMV direct vs split-K for different shapes."""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


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

    h = config.hidden_size
    N_inter = config.intermediate_size

    # Get a linear attention layer for FP16 GEMV test
    lw_lin = engine.layers[0]  # layer 0 should be linear attention
    # Get any layer for INT4
    lw = engine.layers[0]

    print("INT4 GEMV: Direct vs Split-K comparison")
    print("=" * 60)

    # Test shapes
    shapes = [
        ("gate/up", h, N_inter, lw.gate_qweight, lw.gate_scales, lw.gate_zeros),
        ("down", N_inter, h, lw.down_qweight, lw.down_scales, lw.down_zeros),
    ]

    for name, K, N, qw, sc, zr in shapes:
        print(f"\n{name}: K={K}, N={N}")

        grid_x = (N + 255) // 256
        print(f"  WGs (direct): {grid_x}")

        # Direct (no split-K): single kernel, FP16 output
        func_direct = engine.kernels.get_hip("gemv_int4_v2_direct", "gemv_int4_v2")
        params_direct = [
            ctypes.c_uint64(engine.d_normed),
            ctypes.c_uint64(qw),
            ctypes.c_uint64(sc),
            ctypes.c_uint64(zr),
            ctypes.c_uint64(engine.d_ffn_gate),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(config.group_size),
        ]

        # Warm up direct
        for _ in range(5):
            engine.device.launch(func_direct, (grid_x, 1, 1), (256, 1, 1), params_direct)
        engine.device.synchronize()

        # Benchmark direct
        engine.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            engine.device.launch(func_direct, (grid_x, 1, 1), (256, 1, 1), params_direct)
        engine.device.synchronize()
        t_direct = (time.perf_counter() - t0) / 100

        # Split-K
        for k_splits in [4, 8, 16]:
            func_splitk = engine.kernels.get_hip("gemv_int4_v2_splitk", "gemv_int4_v2")
            func_convert = engine.kernels.get_hip("fp32_to_fp16", "gemv_int4_v2")

            # Warm up
            for _ in range(5):
                engine.device.hip.memset(engine.d_gemv_fp32, 0, N * 4)
                engine.device.launch(func_splitk, (grid_x, k_splits, 1), (256, 1, 1), [
                    ctypes.c_uint64(engine.d_normed),
                    ctypes.c_uint64(qw),
                    ctypes.c_uint64(sc),
                    ctypes.c_uint64(zr),
                    ctypes.c_uint64(engine.d_gemv_fp32),
                    ctypes.c_uint32(K),
                    ctypes.c_uint32(N),
                    ctypes.c_uint32(config.group_size),
                ])
                engine.device.launch(func_convert, (grid_x, 1, 1), (256, 1, 1), [
                    ctypes.c_uint64(engine.d_gemv_fp32),
                    ctypes.c_uint64(engine.d_ffn_up),
                    ctypes.c_uint32(N),
                ])
            engine.device.synchronize()

            # Benchmark split-K
            engine.device.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                engine.device.hip.memset(engine.d_gemv_fp32, 0, N * 4)
                engine.device.launch(func_splitk, (grid_x, k_splits, 1), (256, 1, 1), [
                    ctypes.c_uint64(engine.d_normed),
                    ctypes.c_uint64(qw),
                    ctypes.c_uint64(sc),
                    ctypes.c_uint64(zr),
                    ctypes.c_uint64(engine.d_gemv_fp32),
                    ctypes.c_uint32(K),
                    ctypes.c_uint32(N),
                    ctypes.c_uint32(config.group_size),
                ])
                engine.device.launch(func_convert, (grid_x, 1, 1), (256, 1, 1), [
                    ctypes.c_uint64(engine.d_gemv_fp32),
                    ctypes.c_uint64(engine.d_ffn_up),
                    ctypes.c_uint32(N),
                ])
            engine.device.synchronize()
            t_splitk = (time.perf_counter() - t0) / 100

            print(f"  Split-K={k_splits}: {t_splitk*1e6:.0f} us (WGs={grid_x*k_splits})")

        print(f"  Direct:    {t_direct*1e6:.0f} us (WGs={grid_x})")

        # Theoretical
        weight_bytes = K * N // 2  # INT4 packed
        scale_bytes = (K // config.group_size) * N * 4  # FP16 scale + zero
        total_bytes = weight_bytes + scale_bytes + K * 2 + N * 2  # + input + output
        theoretical = total_bytes / 857e9 * 1e6
        print(f"  Theoretical: {theoretical:.0f} us ({total_bytes/1e6:.1f}MB @ 857 GB/s)")

    # Also test FP16 GEMV shapes used by linear attention
    print(f"\n\nFP16 GEMV shapes (linear attention):")
    fp16_shapes = [
        ("in_proj_qkv", h, 10240, lw_lin.la_in_proj_qkv),
        ("in_proj_z", h, 6144, lw_lin.la_in_proj_z),
        ("out_proj", 6144, h, lw_lin.la_out_proj),
    ]

    func_fp16 = engine.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2")
    for name, K, N, weight in fp16_shapes:
        grid_x = (N + 3) // 4

        # Warm up
        for _ in range(5):
            engine.device.launch(func_fp16, (grid_x, 1, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_normed),
                ctypes.c_uint64(weight),
                ctypes.c_uint64(engine.d_la_qkv),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint64(0),  # residual=null
            ])
        engine.device.synchronize()

        engine.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            engine.device.launch(func_fp16, (grid_x, 1, 1), (256, 1, 1), [
                ctypes.c_uint64(engine.d_normed),
                ctypes.c_uint64(weight),
                ctypes.c_uint64(engine.d_la_qkv),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint64(0),  # residual=null
            ])
        engine.device.synchronize()
        t_val = (time.perf_counter() - t0) / 100

        theoretical = K * N * 2 / 857e9 * 1e6
        bw = K * N * 2 / t_val / 1e9
        print(f"  {name} ({K}x{N}): {t_val*1e6:.0f} us (theoretical {theoretical:.0f} us, {bw:.0f} GB/s = {bw/857*100:.0f}%)")

    engine.cleanup()


if __name__ == "__main__":
    main()
