#!/usr/bin/env python3
"""End-to-end text generation test."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def generate(engine, embed, tokenizer, prompt, max_tokens=50, temperature=0.7):
    """Generate text autoregressively."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt ({len(tokens)} tokens): {prompt}")

    # Feed prompt tokens
    for i, t in enumerate(tokens):
        engine.decode_step(embed[t].copy(), position=i)

    # Generate
    generated = []
    t_start = time.perf_counter()
    for step in range(max_tokens):
        logits = engine.compute_logits()

        # Temperature sampling
        if temperature > 0:
            logits = logits / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            # Top-k filter
            top_k = 40
            top_k_idx = np.argpartition(probs, -top_k)[-top_k:]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[top_k_idx] = probs[top_k_idx]
            filtered_probs = filtered_probs / filtered_probs.sum()
            next_token = np.random.choice(len(filtered_probs), p=filtered_probs)
        else:
            next_token = np.argmax(logits)

        generated.append(next_token)

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Feed generated token
        pos = len(tokens) + step
        engine.decode_step(embed[next_token].copy(), position=pos)

    t_elapsed = time.perf_counter() - t_start

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    tok_per_sec = len(generated) / t_elapsed

    print(f"\nGenerated ({len(generated)} tokens, {tok_per_sec:.1f} tok/s):")
    print(output_text)
    print(f"\nTime: {t_elapsed:.2f}s")
    return output_text


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading model...")
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    embed = loader.load_embedding()
    engine.load_lm_head(loader.load_lm_head())
    print("Model loaded.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Test 1: Simple factual
    print("\n" + "="*60)
    print("TEST 1: Factual completion")
    print("="*60)
    generate(engine, embed, tok, "The capital of France is", max_tokens=20, temperature=0)

    # Reset state for next prompt
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()

    # Test 2: Greedy completion
    print("\n" + "="*60)
    print("TEST 2: Reasoning")
    print("="*60)
    generate(engine, embed, tok, "1 + 1 = ", max_tokens=30, temperature=0)

    engine.cleanup()


if __name__ == "__main__":
    main()
