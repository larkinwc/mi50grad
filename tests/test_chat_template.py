#!/usr/bin/env python3
"""Test E2E with chat template format and check Paris ranking."""

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
    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading all layers...")
    t0 = time.time()
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()
    lm_head = loader.load_lm_head()
    engine.load_lm_head(lm_head)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Test 1: Raw text (baseline)
    prompt_raw = "The capital of France is"
    tokens_raw = tokenizer.encode(prompt_raw)
    print(f"\n=== Test 1: Raw text ===")
    print(f"Prompt: {repr(prompt_raw)}")
    print(f"Tokens ({len(tokens_raw)}): {tokens_raw}")

    for i, tid in enumerate(tokens_raw):
        emb = embed[tid].copy()
        hidden = engine.decode_step(emb, position=i)

    logits = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    paris_id = tokenizer.encode(" Paris")[-1]
    paris_rank = np.sum(logits > logits[paris_id]) + 1
    print(f"Paris token={paris_id}, logit={logits[paris_id]:.4f}, rank={paris_rank}")
    top5 = np.argsort(logits)[-5:][::-1]
    for idx in top5:
        print(f"  {repr(tokenizer.decode([idx]))}: {logits[idx]:.4f}")

    # Test 2: Chat template
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()

    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    prompt_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens_chat = tokenizer.encode(prompt_chat)
    print(f"\n=== Test 2: Chat template ===")
    print(f"Prompt: {repr(prompt_chat[:200])}")
    print(f"Tokens ({len(tokens_chat)}): {tokens_chat[:10]}...")

    t1 = time.time()
    for i, tid in enumerate(tokens_chat):
        emb = embed[tid].copy()
        hidden = engine.decode_step(emb, position=i)
    t2 = time.time()
    print(f"Prefill: {t2-t1:.1f}s ({len(tokens_chat)/(t2-t1):.1f} tok/s)")

    logits2 = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    paris_rank2 = np.sum(logits2 > logits2[paris_id]) + 1
    print(f"Paris token={paris_id}, logit={logits2[paris_id]:.4f}, rank={paris_rank2}")
    top10 = np.argsort(logits2)[-10:][::-1]
    print("Top 10:")
    for idx in top10:
        print(f"  {repr(tokenizer.decode([idx]))}: {logits2[idx]:.4f}")

    # Greedy decode a few tokens
    print("\nGreedy generation:")
    generated = []
    pos = len(tokens_chat)
    for step in range(20):
        top_id = np.argmax(logits2)
        generated.append(top_id)
        tok_text = tokenizer.decode([top_id])
        print(f"  step {step}: {repr(tok_text)} (id={top_id}, logit={logits2[top_id]:.4f})")

        if top_id == tokenizer.eos_token_id:
            break

        emb = embed[top_id].copy()
        hidden = engine.decode_step(emb, position=pos)
        pos += 1
        logits2 = lm_head.astype(np.float32) @ hidden.astype(np.float32)

    print(f"\nGenerated text: {tokenizer.decode(generated)}")

    engine.cleanup()


if __name__ == "__main__":
    main()
