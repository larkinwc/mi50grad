#!/usr/bin/env python3
"""Profile decode step timing breakdown."""

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

    print("Loading all 64 layers...")
    t0 = time.time()
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Run one warmup step
    emb = embed[100].copy()
    hidden = engine.decode_step(emb, position=0)

    # Now profile second step with timing
    print("\nProfiling decode step...")
    emb = embed[200].copy()
    h = config.hidden_size
    cfg = config

    engine.device.upload(engine.d_hidden, emb.tobytes())

    t_total = time.time()
    t_lin_attn = 0
    t_full_attn = 0
    t_ffn = 0
    t_norm = 0

    for layer_idx in range(cfg.num_hidden_layers):
        lw = engine.layers[layer_idx]

        t0 = time.time()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
        t_norm += time.time() - t0

        t0 = time.time()
        if lw.layer_type == 'full_attention':
            engine._decode_full_attention(layer_idx, lw, 1)
            t_full_attn += time.time() - t0
        else:
            engine._decode_linear_attention(layer_idx, lw, 1)
            t_lin_attn += time.time() - t0

        t0 = time.time()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)
        t_norm += time.time() - t0

        t0 = time.time()
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
        t_ffn += time.time() - t0

    engine.kv_cache.advance()
    t_total = time.time() - t_total

    print(f"\nDecode step breakdown:")
    print(f"  Total:         {t_total*1000:.0f}ms")
    print(f"  Linear attn:   {t_lin_attn*1000:.0f}ms ({48} layers)")
    print(f"  Full attn:     {t_full_attn*1000:.0f}ms ({16} layers)")
    print(f"  FFN:           {t_ffn*1000:.0f}ms ({64} layers)")
    print(f"  RMSNorm:       {t_norm*1000:.0f}ms")
    print(f"  Per lin layer: {t_lin_attn/48*1000:.1f}ms")
    print(f"  Per full layer:{t_full_attn/16*1000:.1f}ms")
    print(f"  Per FFN:       {t_ffn/64*1000:.1f}ms")

    engine.cleanup()


if __name__ == "__main__":
    main()
