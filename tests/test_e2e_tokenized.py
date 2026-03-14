#!/usr/bin/env python3
"""E2E generation test with proper tokenized prompt."""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine
from src.inference.sampler import SamplingParams, sample_token


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    max_tokens = 30

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt: {repr(prompt)}")
    print(f"Token IDs: {prompt_tokens}")
    print(f"Decoded back: {repr(tokenizer.decode(prompt_tokens))}")

    # Load model
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    loader = QwenWeightLoader(model_dir, config)

    print(f"\nLoading {config.num_hidden_layers} layers...")
    t0 = time.time()
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
        if i % 16 == 0:
            print(f"  Layer {i}/{config.num_hidden_layers}")
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed_weight = loader.load_embedding()
    lm_head_weight = loader.load_lm_head()

    # Upload LM head to GPU
    print("Uploading LM head to GPU...")
    engine.load_lm_head(lm_head_weight)

    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s")

    # Prefill
    print(f"\nPrefill ({len(prompt_tokens)} tokens)...")
    h = config.hidden_size
    t0 = time.time()
    for i, tid in enumerate(prompt_tokens):
        emb = embed_weight[tid].copy()
        hidden = engine.decode_step(emb, position=i)
    t_prefill = time.time() - t0
    print(f"Prefill: {t_prefill:.1f}s ({len(prompt_tokens)/t_prefill:.2f} tok/s)")

    # Generate
    print(f"\nGenerating {max_tokens} tokens...")
    generated_ids = []
    params = SamplingParams(temperature=0.0)  # greedy for reproducibility
    position = len(prompt_tokens)

    t_decode_start = time.time()
    for step in range(max_tokens):
        t_step_start = time.time()

        # LM head via GPU
        logits = engine.compute_logits()

        t_logits = time.time() - t_step_start

        if np.any(np.isnan(logits)):
            print(f"NaN in logits at step {step}")
            break

        token_id = sample_token(logits, params)
        generated_ids.append(token_id)

        # Print incremental
        text_so_far = tokenizer.decode(generated_ids)
        print(f"  Step {step}: token={token_id} ({repr(tokenizer.decode([token_id]))})"
              f" logits={t_logits*1000:.0f}ms => {repr(text_so_far)}")

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            print("  [EOS]")
            break

        # Next step
        emb = embed_weight[token_id].copy()
        t_dec = time.time()
        hidden = engine.decode_step(emb, position)
        t_dec_step = time.time() - t_dec
        print(f"         decode={t_dec_step*1000:.0f}ms")
        position += 1

    t_decode = time.time() - t_decode_start
    print(f"\nResults:")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Generated: {repr(tokenizer.decode(generated_ids))}")
    print(f"  Decode: {t_decode:.1f}s ({len(generated_ids)} tokens)")
    if generated_ids:
        print(f"  Speed: {len(generated_ids)/t_decode:.3f} tok/s")

    engine.cleanup()


if __name__ == "__main__":
    main()
