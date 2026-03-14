#!/usr/bin/env python3
"""Test: skip attention (use identity) and only run FFN to verify FFN path."""

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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading layers...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()
    lm_head = loader.load_lm_head()

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    h = config.hidden_size
    cfg = config

    # Run prefill with SKIPPED attention (identity) — only FFN runs
    print(f"\n=== FFN-only mode (skip attention) ===")
    for i, tid in enumerate(tokens):
        emb = embed[tid].copy()
        engine.device.upload(engine.d_hidden, emb.tobytes())

        for layer_idx in range(cfg.num_hidden_layers):
            lw = engine.layers[layer_idx]

            # SKIP attention — don't call _decode_full/linear_attention
            # Just run FFN
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

    # Final norm
    engine._launch_rmsnorm(engine.d_hidden2, engine.d_hidden,
                            engine.d_final_norm, h)
    hidden = np.frombuffer(engine.device.download(engine.d_hidden2, h * 2),
                            dtype=np.float16).copy()

    logits = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    top = np.argsort(logits)[-10:][::-1]
    print(f"Hidden: range=[{hidden.min():.4f}, {hidden.max():.4f}], "
          f"mean={hidden.astype(np.float32).mean():.4f}")
    print(f"Logits: range=[{logits.min():.4f}, {logits.max():.4f}], "
          f"std={logits.std():.4f}")
    print("Top 10:")
    for idx in top:
        print(f"  {idx}: {repr(tokenizer.decode([idx]))} = {logits[idx]:.4f}")

    # Now run WITH attention
    print(f"\n=== Full mode (with attention) ===")
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()

    for i, tid in enumerate(tokens):
        emb = embed[tid].copy()
        hidden = engine.decode_step(emb, position=i)

    logits2 = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    top2 = np.argsort(logits2)[-10:][::-1]
    print(f"Hidden: range=[{hidden.min():.4f}, {hidden.max():.4f}], "
          f"mean={hidden.astype(np.float32).mean():.4f}")
    print(f"Logits: range=[{logits2.min():.4f}, {logits2.max():.4f}], "
          f"std={logits2.std():.4f}")
    print("Top 10:")
    for idx in top2:
        print(f"  {idx}: {repr(tokenizer.decode([idx]))} = {logits2[idx]:.4f}")

    engine.cleanup()


if __name__ == "__main__":
    main()
