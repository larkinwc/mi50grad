#!/usr/bin/env python3
"""Diagnose GEMV v2 correctness by comparing against v1."""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine
import ctypes


def test_gemv_int4(engine, lw, config):
    """Compare INT4 GEMV v1 vs v2 on a single FFN gate projection."""
    h = config.hidden_size
    inter = config.intermediate_size

    # Create random input
    x = np.random.randn(h).astype(np.float16)
    engine.device.upload(engine.d_normed, x.tobytes())

    # Run v2 (current)
    engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                              lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                              h, inter)
    engine.device.synchronize()
    out_v2 = np.frombuffer(engine.device.download(engine.d_ffn_gate, inter * 2),
                            dtype=np.float16).copy()

    # Run v1 (force old kernel)
    old_flag = engine._gemv_int4_v2
    engine._gemv_int4_v2 = False
    engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                              lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                              h, inter)
    engine.device.synchronize()
    out_v1 = np.frombuffer(engine.device.download(engine.d_ffn_gate, inter * 2),
                            dtype=np.float16).copy()
    engine._gemv_int4_v2 = old_flag

    # Compare
    max_err = np.max(np.abs(out_v1.astype(np.float32) - out_v2.astype(np.float32)))
    mean_err = np.mean(np.abs(out_v1.astype(np.float32) - out_v2.astype(np.float32)))
    cos_sim = np.dot(out_v1.astype(np.float32), out_v2.astype(np.float32)) / (
        np.linalg.norm(out_v1.astype(np.float32)) * np.linalg.norm(out_v2.astype(np.float32)) + 1e-10)

    print(f"INT4 GEMV comparison (gate proj, layer 0):")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  v1 stats: mean={np.mean(out_v1):.4f} std={np.std(out_v1):.4f} min={np.min(out_v1):.4f} max={np.max(out_v1):.4f}")
    print(f"  v2 stats: mean={np.mean(out_v2):.4f} std={np.std(out_v2):.4f} min={np.min(out_v2):.4f} max={np.max(out_v2):.4f}")
    print(f"  First 10 v1: {out_v1[:10]}")
    print(f"  First 10 v2: {out_v2[:10]}")
    return cos_sim > 0.99


def test_gemv_fp16(engine, lw, config):
    """Compare FP16 GEMV v1 vs v2."""
    h = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim

    x = np.random.randn(h).astype(np.float16)
    engine.device.upload(engine.d_normed, x.tobytes())

    # Run v2
    engine._launch_gemv_fp16(engine.d_q, engine.d_normed, lw.q_weight, h, q_dim)
    engine.device.synchronize()
    out_v2 = np.frombuffer(engine.device.download(engine.d_q, q_dim * 2),
                            dtype=np.float16).copy()

    # Run v1
    old_flag = engine._gemv_fp16_v2
    engine._gemv_fp16_v2 = False
    engine._launch_gemv_fp16(engine.d_q, engine.d_normed, lw.q_weight, h, q_dim)
    engine.device.synchronize()
    out_v1 = np.frombuffer(engine.device.download(engine.d_q, q_dim * 2),
                            dtype=np.float16).copy()
    engine._gemv_fp16_v2 = old_flag

    max_err = np.max(np.abs(out_v1.astype(np.float32) - out_v2.astype(np.float32)))
    mean_err = np.mean(np.abs(out_v1.astype(np.float32) - out_v2.astype(np.float32)))
    cos_sim = np.dot(out_v1.astype(np.float32), out_v2.astype(np.float32)) / (
        np.linalg.norm(out_v1.astype(np.float32)) * np.linalg.norm(out_v2.astype(np.float32)) + 1e-10)

    print(f"\nFP16 GEMV comparison (q_proj, layer 0):")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  v1 stats: mean={np.mean(out_v1):.4f} std={np.std(out_v1):.4f}")
    print(f"  v2 stats: mean={np.mean(out_v2):.4f} std={np.std(out_v2):.4f}")
    print(f"  First 10 v1: {out_v1[:10]}")
    print(f"  First 10 v2: {out_v2[:10]}")
    return cos_sim > 0.99


def test_speed(engine, lw, config):
    """Benchmark INT4 GEMV v1 vs v2."""
    h = config.hidden_size
    inter = config.intermediate_size

    x = np.random.randn(h).astype(np.float16)
    engine.device.upload(engine.d_normed, x.tobytes())

    # Warm up
    for _ in range(5):
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, inter)
    engine.device.synchronize()

    # v2 speed
    t0 = time.perf_counter()
    for _ in range(20):
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, inter)
    engine.device.synchronize()
    t_v2 = (time.perf_counter() - t0) / 20

    # v1 speed
    old_flag = engine._gemv_int4_v2
    engine._gemv_int4_v2 = False
    for _ in range(5):
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, inter)
    engine.device.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, inter)
    engine.device.synchronize()
    t_v1 = (time.perf_counter() - t0) / 20
    engine._gemv_int4_v2 = old_flag

    # Theoretical minimum
    data_bytes = inter * h / 2  # INT4 weights
    theoretical_us = data_bytes / 857e9 * 1e6

    print(f"\nINT4 GEMV speed (gate proj K={h} N={inter}):")
    print(f"  v1: {t_v1*1e6:.0f} us")
    print(f"  v2: {t_v2*1e6:.0f} us")
    print(f"  Speedup: {t_v1/t_v2:.1f}x")
    print(f"  Theoretical min: {theoretical_us:.0f} us")
    print(f"  v2 BW util: {data_bytes / t_v2 / 857e9 * 100:.0f}%")

    # FP16 speed
    q_dim = config.num_attention_heads * config.head_dim
    # Find a full attention layer
    fa_layer = None
    for i, layer in enumerate(engine.layers):
        if layer.layer_type == 'full_attention':
            fa_layer = layer
            break
    if fa_layer:
        # v2
        engine._gemv_fp16_v2 = True
        for _ in range(5):
            engine._launch_gemv_fp16(engine.d_q, engine.d_normed, fa_layer.q_weight, h, q_dim)
        engine.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            engine._launch_gemv_fp16(engine.d_q, engine.d_normed, fa_layer.q_weight, h, q_dim)
        engine.device.synchronize()
        t_fp16_v2 = (time.perf_counter() - t0) / 20

        # v1
        engine._gemv_fp16_v2 = False
        for _ in range(5):
            engine._launch_gemv_fp16(engine.d_q, engine.d_normed, fa_layer.q_weight, h, q_dim)
        engine.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            engine._launch_gemv_fp16(engine.d_q, engine.d_normed, fa_layer.q_weight, h, q_dim)
        engine.device.synchronize()
        t_fp16_v1 = (time.perf_counter() - t0) / 20
        engine._gemv_fp16_v2 = True

        fp16_bytes = q_dim * h * 2
        theoretical_fp16 = fp16_bytes / 857e9 * 1e6
        print(f"\nFP16 GEMV speed (q_proj K={h} N={q_dim}):")
        print(f"  v1: {t_fp16_v1*1e6:.0f} us")
        print(f"  v2: {t_fp16_v2*1e6:.0f} us")
        print(f"  Speedup: {t_fp16_v1/t_fp16_v2:.1f}x")
        print(f"  Theoretical min: {theoretical_fp16:.0f} us")


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    # Find a full attention layer for FP16 test
    fa_idx = None
    for i in range(config.num_hidden_layers):
        if config.is_full_attention(i):
            fa_idx = i
            break
    print(f"First full attention layer: {fa_idx}")
    print(f"Layer 0 type: {'full' if config.is_full_attention(0) else 'linear'}")

    # Load layers we need
    layers_to_load = {0}
    if fa_idx is not None:
        layers_to_load.add(fa_idx)

    for i in range(max(layers_to_load) + 1):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)

    lw_int4 = engine.layers[0]  # Any layer has FFN weights
    int4_ok = test_gemv_int4(engine, lw_int4, config)

    fp16_ok = True
    if fa_idx is not None:
        lw_fp16 = engine.layers[fa_idx]
        fp16_ok = test_gemv_fp16(engine, lw_fp16, config)
    else:
        print("\nNo full attention layers found, skipping FP16 test")

    if int4_ok and fp16_ok:
        print("\n=== CORRECTNESS PASSED ===")
    else:
        print(f"\n=== CORRECTNESS FAILED (INT4: {int4_ok}, FP16: {fp16_ok}) ===")

    # Load all layers for speed test
    print("\nLoading remaining layers for speed test...")
    for i in range(max(layers_to_load) + 1, config.num_hidden_layers):
        w = loader.load_layer(i)
        engine.load_layer_weights(i, w)
    del w

    test_speed(engine, lw_int4, config)

    engine.cleanup()


if __name__ == "__main__":
    main()
