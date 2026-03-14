#!/usr/bin/env python3
"""Profile decode step using actual engine methods with instrumentation."""

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
import struct


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    print(f"DeltaNet GPU: {engine._deltanet_gpu}")

    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights
    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()
    engine.load_lm_head(loader.load_lm_head())

    # Warm up
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)

    # Now instrument one decode step
    h = config.hidden_size
    cfg = config
    emb = embed[760].copy()
    engine.device.upload(engine.d_hidden, emb.tobytes())

    times = {}

    for layer_idx in range(cfg.num_hidden_layers):
        lw = engine.layers[layer_idx]
        layer_type = 'full' if lw.layer_type == 'full_attention' else 'linear'

        # Pre-attn norm
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
        times.setdefault(f'{layer_type}_rmsnorm_attn', 0)
        times[f'{layer_type}_rmsnorm_attn'] += time.perf_counter() - t

        # Attention
        t = time.perf_counter()
        if lw.layer_type == 'full_attention':
            engine._decode_full_attention(layer_idx, lw, 3)
        else:
            if engine._deltanet_gpu:
                engine._decode_linear_attention_gpu(layer_idx, lw, 3)
            else:
                engine._decode_linear_attention(layer_idx, lw, 3)
        times.setdefault(f'{layer_type}_attn', 0)
        times[f'{layer_type}_attn'] += time.perf_counter() - t

        # FFN
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, cfg.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_up, engine.d_normed,
                                  lw.up_qweight, lw.up_scales, lw.up_zeros,
                                  h, cfg.intermediate_size)
        engine._launch_silu_fused(engine.d_ffn_gate, engine.d_ffn_up,
                                   engine.d_ffn_gate, cfg.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_out, engine.d_ffn_gate,
                                  lw.down_qweight, lw.down_scales, lw.down_zeros,
                                  cfg.intermediate_size, h)
        engine._launch_residual_add(engine.d_hidden, engine.d_ffn_out, h)
        times.setdefault(f'{layer_type}_ffn', 0)
        times[f'{layer_type}_ffn'] += time.perf_counter() - t

    engine.kv_cache.advance()

    # Final norm + LM head
    t = time.perf_counter()
    engine._launch_rmsnorm(engine.d_hidden2, engine.d_hidden, engine.d_final_norm, h)
    engine.compute_logits()
    times['final_norm_lmhead'] = time.perf_counter() - t

    total = sum(times.values())
    print(f"\n{'Component':<30} {'Time (ms)':>10} {'%':>6}")
    print("-" * 50)
    for name, t_val in sorted(times.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {t_val*1000:>8.1f}ms {t_val/total*100:>5.1f}%")
    print(f"\n  {'TOTAL':<28} {total*1000:>8.1f}ms ({1000/total:.1f} tok/s)")

    engine.cleanup()


if __name__ == "__main__":
    main()
