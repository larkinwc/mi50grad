#!/usr/bin/env python3
"""Test optimized GEMV kernels (INT4 v2 coalesced + FP16 v2 vectorized)."""

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

    print(f"GEMV INT4 v2: {engine._gemv_int4_v2}")
    print(f"GEMV FP16 v2: {engine._gemv_fp16_v2}")
    print(f"DeltaNet GPU: {engine._deltanet_gpu}")

    # Load all layers
    print("Loading layers...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights
    engine.load_final_norm(loader.load_final_norm())
    embed = loader.load_embedding()
    engine.load_lm_head(loader.load_lm_head())

    # Warm up
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)

    # Correctness check: "The capital of France is"
    import struct
    engine.device.upload(engine.d_hidden, embed[760].copy().tobytes())
    for layer_idx in range(config.num_hidden_layers):
        lw = engine.layers[layer_idx]
        h = config.hidden_size
        cfg = config

        # Pre-attn norm
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)

        # Attention
        if lw.layer_type == 'full_attention':
            engine._decode_full_attention(layer_idx, lw, 3)
        else:
            if engine._deltanet_gpu:
                engine._decode_linear_attention_gpu(layer_idx, lw, 3)
            else:
                engine._decode_linear_attention(layer_idx, lw, 3)

        # FFN
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

    engine.kv_cache.advance()

    # Final norm + logits
    engine._launch_rmsnorm(engine.d_hidden2, engine.d_hidden, engine.d_final_norm, h)
    logits_fp16 = engine.compute_logits()
    logits = logits_fp16.astype(np.float32)

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\nCorrectness check (position for 'The capital of France is'):")
    print(f"  Top 5 token IDs: {top5}")
    print(f"  Top 5 logits: {[f'{logits[i]:.2f}' for i in top5]}")

    # Try to decode token names
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"  Top 5 tokens: {[tok.decode([i]) for i in top5]}")
    except:
        pass

    # Speed test
    print("\nSpeed test (10 decode steps):")
    engine.kv_cache.reset()
    engine.deltanet_state.reset()
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)

    times = []
    for i in range(10):
        emb = embed[760 + i].copy()
        t0 = time.perf_counter()
        engine.decode_step(emb, position=3 + i)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Step {i}: {(t1-t0)*1000:.1f}ms")

    avg = np.mean(times) * 1000
    print(f"\n  Avg: {avg:.1f}ms/tok = {1000/avg:.2f} tok/s")

    engine.cleanup()


if __name__ == "__main__":
    main()
