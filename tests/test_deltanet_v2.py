#!/usr/bin/env python3
"""Test DeltaNet v2 kernel: correctness vs CPU reference + performance."""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def test_correctness():
    """Compare GPU DeltaNet v2 output against CPU reference for several layers."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)

    # Create two engines: one forced CPU, one GPU
    engine_gpu = InferenceEngine(config, device_id=0, max_seq_len=64)
    print(f"GPU DeltaNet: v2={engine_gpu._deltanet_v2}, gpu={engine_gpu._deltanet_gpu}")

    loader = QwenWeightLoader(model_dir, config)
    for i in range(config.num_hidden_layers):
        engine_gpu.load_layer_weights(i, loader.load_layer(i))
    engine_gpu.load_final_norm(loader.load_final_norm())
    embed = loader.load_embedding()
    engine_gpu.load_lm_head(loader.load_lm_head())

    # Run 3 tokens to build up some conv/recurrent state
    for i in range(3):
        engine_gpu.decode_step(embed[760].copy(), position=i)
    engine_gpu.device.synchronize()

    # Now compare GPU vs CPU for token 4
    # Save current state
    engine_gpu.device.synchronize()

    # Get GPU result
    gpu_result = engine_gpu.decode_step(embed[1234].copy(), position=3)
    engine_gpu.device.synchronize()
    gpu_logits = engine_gpu.compute_logits()

    # Reset and run with CPU fallback
    engine_gpu.kv_cache.current_len = 0
    engine_gpu.deltanet_state.reset()

    saved_gpu = engine_gpu._deltanet_gpu
    engine_gpu._deltanet_gpu = False
    for i in range(3):
        engine_gpu.decode_step(embed[760].copy(), position=i)
    cpu_result = engine_gpu.decode_step(embed[1234].copy(), position=3)
    engine_gpu.device.synchronize()
    cpu_logits = engine_gpu.compute_logits()
    engine_gpu._deltanet_gpu = saved_gpu

    # Compare hidden states
    max_err = np.max(np.abs(gpu_result.astype(np.float32) - cpu_result.astype(np.float32)))
    cos_sim = np.dot(gpu_result.astype(np.float32).ravel(),
                     cpu_result.astype(np.float32).ravel()) / (
        np.linalg.norm(gpu_result.astype(np.float32)) *
        np.linalg.norm(cpu_result.astype(np.float32)) + 1e-10)

    print(f"\nHidden state comparison (GPU vs CPU):")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # Compare logits
    if gpu_logits is not None and cpu_logits is not None:
        gpu_top = np.argsort(gpu_logits.astype(np.float32))[-5:][::-1]
        cpu_top = np.argsort(cpu_logits.astype(np.float32))[-5:][::-1]
        print(f"\nTop-5 logit tokens:")
        print(f"  GPU: {gpu_top}")
        print(f"  CPU: {cpu_top}")
        logit_cos = np.dot(gpu_logits.astype(np.float32), cpu_logits.astype(np.float32)) / (
            np.linalg.norm(gpu_logits.astype(np.float32)) *
            np.linalg.norm(cpu_logits.astype(np.float32)) + 1e-10)
        print(f"  Logit cosine similarity: {logit_cos:.8f}")

    ok = cos_sim > 0.99
    print(f"\n  {'PASS' if ok else 'FAIL'}")

    engine_gpu.cleanup()
    return ok


def test_performance():
    """Benchmark DeltaNet GPU vs CPU per-token speed."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)

    loader = QwenWeightLoader(model_dir, config)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    embed = loader.load_embedding()
    engine.load_lm_head(loader.load_lm_head())

    print(f"\nDeltaNet mode: gpu={engine._deltanet_gpu}, v2={engine._deltanet_v2}")

    # Warm up
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)
    engine.device.synchronize()

    # Benchmark GPU path
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()

    n_tokens = 10
    engine.device.synchronize()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        engine.decode_step(embed[760 + i].copy(), position=i)
    engine.device.synchronize()
    t_gpu = time.perf_counter() - t0
    print(f"\nGPU path: {t_gpu*1000:.1f}ms for {n_tokens} tokens = {t_gpu/n_tokens*1000:.1f}ms/tok ({n_tokens/t_gpu:.1f} tok/s)")

    # Benchmark CPU fallback
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()
    engine._deltanet_gpu = False

    engine.device.synchronize()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        engine.decode_step(embed[760 + i].copy(), position=i)
    engine.device.synchronize()
    t_cpu = time.perf_counter() - t0
    print(f"CPU path: {t_cpu*1000:.1f}ms for {n_tokens} tokens = {t_cpu/n_tokens*1000:.1f}ms/tok ({n_tokens/t_cpu:.1f} tok/s)")

    speedup = t_cpu / t_gpu
    print(f"\nSpeedup: {speedup:.2f}x")

    engine.cleanup()


def test_timing_breakdown():
    """Per-component timing to isolate DeltaNet kernel time."""
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

    print(f"\nDeltaNet mode: gpu={engine._deltanet_gpu}, v2={engine._deltanet_v2}")

    # Time just linear attention layers
    import ctypes
    h = config.hidden_size
    cfg = config

    t_gemv = 0.0
    t_deltanet = 0.0
    t_conv_shift = 0.0

    emb = embed[760].copy()
    engine.device.upload(engine.d_hidden, emb.tobytes())

    for layer_idx in range(cfg.num_hidden_layers):
        lw = engine.layers[layer_idx]
        if lw.layer_type != 'linear_attention':
            continue

        slot = engine.deltanet_state.get_slot(layer_idx, cfg)
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)

        # Time input GEMVs
        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_gemv_fp16(engine.d_la_qkv, engine.d_normed, lw.la_in_proj_qkv,
                                  h, engine.la_qkv_dim)
        engine._launch_gemv_fp16(engine.d_la_dt, engine.d_normed, lw.la_in_proj_a,
                                  h, engine.la_dt_dim)
        engine._launch_gemv_fp16(engine.d_la_b, engine.d_normed, lw.la_in_proj_b,
                                  h, engine.la_dt_dim)
        engine._launch_gemv_fp16(engine.d_la_z, engine.d_normed, lw.la_in_proj_z,
                                  h, engine.la_z_dim)
        engine.device.synchronize()
        t_gemv += time.perf_counter() - t

        d_conv = engine.deltanet_state.d_conv_states[slot]
        d_state = engine.deltanet_state.d_states[slot]

        if engine._deltanet_v2:
            # Time main kernel
            engine.device.synchronize()
            t = time.perf_counter()
            func = engine.kernels.get_hip("deltanet_decode_v2", "deltanet_v2")
            params = [
                ctypes.c_uint64(engine.d_la_qkv),
                ctypes.c_uint64(engine.d_la_dt),
                ctypes.c_uint64(engine.d_la_b),
                ctypes.c_uint64(engine.d_la_z),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint64(lw.d_la_conv_weight),
                ctypes.c_uint64(lw.d_la_A_log),
                ctypes.c_uint64(lw.d_la_dt_bias),
                ctypes.c_uint64(lw.d_la_norm),
                ctypes.c_uint64(d_state),
                ctypes.c_uint64(engine.d_la_out),
            ]
            engine.device.launch(func, (cfg.linear_num_value_heads, 1, 1),
                                 (256, 1, 1), params, shared_mem=4096)
            engine.device.synchronize()
            t_deltanet += time.perf_counter() - t

            # Time conv shift
            engine.device.synchronize()
            t = time.perf_counter()
            conv_dim = engine.deltanet_state.conv_dim
            shift_func = engine.kernels.get_hip("deltanet_conv_shift", "deltanet_v2")
            shift_params = [
                ctypes.c_uint64(engine.d_la_qkv),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint32(conv_dim),
            ]
            grid_x = (conv_dim + 255) // 256
            engine.device.launch(shift_func, (grid_x, 1, 1), (256, 1, 1), shift_params)
            engine.device.synchronize()
            t_conv_shift += time.perf_counter() - t

        # Output GEMV
        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_gemv_fp16(engine.d_proj_out, engine.d_la_out, lw.la_out_proj,
                                  engine.la_z_dim, h)
        engine.device.synchronize()
        t_gemv += time.perf_counter() - t

    num_lin = cfg.num_linear_attention_layers
    print(f"\nLinear attention breakdown ({num_lin} layers):")
    print(f"  Input+Output GEMVs: {t_gemv*1000:.2f}ms ({t_gemv/num_lin*1e6:.0f} us/layer)")
    print(f"  DeltaNet recurrence: {t_deltanet*1000:.2f}ms ({t_deltanet/num_lin*1e6:.0f} us/layer)")
    print(f"  Conv shift: {t_conv_shift*1000:.2f}ms ({t_conv_shift/num_lin*1e6:.0f} us/layer)")
    print(f"  Total: {(t_gemv+t_deltanet+t_conv_shift)*1000:.2f}ms")

    # Theoretical GEMV minimum
    gemv_bytes = (engine.la_qkv_dim * h + engine.la_dt_dim * h * 2 +
                  engine.la_z_dim * h + h * engine.la_z_dim) * 2
    print(f"\n  GEMV weight data: {gemv_bytes/1e6:.1f}MB/layer")
    print(f"  Theoretical GEMV time: {gemv_bytes/857e9*1e6:.0f} us/layer")

    engine.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["correctness", "perf", "breakdown", "all"],
                        default="all")
    args = parser.parse_args()

    if args.test in ("correctness", "all"):
        test_correctness()
    if args.test in ("perf", "all"):
        test_performance()
    if args.test in ("breakdown", "all"):
        test_timing_breakdown()
