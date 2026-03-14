#!/usr/bin/env python3
"""Accurate per-component timing with GPU sync between measurements."""

import sys
import time
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

    h = config.hidden_size
    cfg = config
    emb = embed[760].copy()
    engine.device.upload(engine.d_hidden, emb.tobytes())

    times = {}

    for layer_idx in range(cfg.num_hidden_layers):
        lw = engine.layers[layer_idx]
        layer_type = 'full' if lw.layer_type == 'full_attention' else 'linear'

        # Pre-attn norm
        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_rmsnorm', 0)
        times[f'{layer_type}_rmsnorm'] += time.perf_counter() - t

        # Attention
        engine.device.synchronize()
        t = time.perf_counter()
        if lw.layer_type == 'full_attention':
            engine._decode_full_attention(layer_idx, lw, 3)
        else:
            if engine._deltanet_gpu:
                engine._decode_linear_attention_gpu(layer_idx, lw, 3)
            else:
                engine._decode_linear_attention(layer_idx, lw, 3)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_attn', 0)
        times[f'{layer_type}_attn'] += time.perf_counter() - t

        # FFN norm
        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_ffn_norm', 0)
        times[f'{layer_type}_ffn_norm'] += time.perf_counter() - t

        # FFN GEMVs
        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, cfg.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_up, engine.d_normed,
                                  lw.up_qweight, lw.up_scales, lw.up_zeros,
                                  h, cfg.intermediate_size)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_ffn_gate_up', 0)
        times[f'{layer_type}_ffn_gate_up'] += time.perf_counter() - t

        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_silu_fused(engine.d_ffn_gate, engine.d_ffn_up,
                                   engine.d_ffn_gate, cfg.intermediate_size)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_ffn_silu', 0)
        times[f'{layer_type}_ffn_silu'] += time.perf_counter() - t

        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_gemv_int4(engine.d_ffn_out, engine.d_ffn_gate,
                                  lw.down_qweight, lw.down_scales, lw.down_zeros,
                                  cfg.intermediate_size, h)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_ffn_down', 0)
        times[f'{layer_type}_ffn_down'] += time.perf_counter() - t

        engine.device.synchronize()
        t = time.perf_counter()
        engine._launch_residual_add(engine.d_hidden, engine.d_ffn_out, h)
        engine.device.synchronize()
        times.setdefault(f'{layer_type}_residual', 0)
        times[f'{layer_type}_residual'] += time.perf_counter() - t

    engine.kv_cache.advance()

    # Final norm
    engine.device.synchronize()
    t = time.perf_counter()
    engine._launch_rmsnorm(engine.d_hidden2, engine.d_hidden, engine.d_final_norm, h)
    engine.device.synchronize()
    times['final_norm'] = time.perf_counter() - t

    # LM head
    engine.device.synchronize()
    t = time.perf_counter()
    engine._launch_gemv_fp16(engine.d_logits, engine.d_hidden2, engine.d_lm_head,
                              h, engine.lm_head_vocab)
    engine.device.synchronize()
    times['lm_head'] = time.perf_counter() - t

    total = sum(times.values())
    print(f"\n{'Component':<30} {'Time (ms)':>10} {'%':>6}")
    print("-" * 50)
    for name, t_val in sorted(times.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {t_val*1000:>8.2f}ms {t_val/total*100:>5.1f}%")
    print(f"\n  {'TOTAL':<28} {total*1000:>8.2f}ms ({1000/total:.1f} tok/s)")

    # Theoretical minimums
    print(f"\nTheoretical minimums (857 GB/s peak):")
    int4_gate = cfg.intermediate_size * h / 2
    int4_down = h * cfg.intermediate_size / 2
    fp16_lmhead = engine.lm_head_vocab * h * 2
    print(f"  INT4 gate/up GEMV: {int4_gate / 857e9 * 1e6:.0f} us each")
    print(f"  INT4 down GEMV: {int4_down / 857e9 * 1e6:.0f} us")
    print(f"  FP16 LM head: {fp16_lmhead / 857e9 * 1e6:.0f} us")

    engine.cleanup()


if __name__ == "__main__":
    main()
