#!/usr/bin/env python3
"""Generate a full response with chat template to verify E2E quality."""

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
    engine = InferenceEngine(config, device_id=0, max_seq_len=512)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading layers...")
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
    t_load = time.time() - t0
    print(f"Loaded in {t_load:.1f}s")

    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"Prompt ({len(tokens)} tokens): {repr(prompt[:120])}")

    # Prefill
    t1 = time.time()
    for i, tid in enumerate(tokens):
        emb = embed[tid].copy()
        hidden = engine.decode_step(emb, position=i)
    logits = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    t_prefill = time.time() - t1
    print(f"Prefill: {t_prefill:.1f}s ({len(tokens)/t_prefill:.1f} tok/s)")

    # Generate
    generated = []
    pos = len(tokens)
    t2 = time.time()
    for step in range(200):
        top_id = int(np.argmax(logits))
        generated.append(top_id)
        tok_text = tokenizer.decode([top_id])
        if top_id in (tokenizer.eos_token_id, 248044):  # EOS
            break
        emb = embed[top_id].copy()
        hidden = engine.decode_step(emb, position=pos)
        pos += 1
        logits = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    t_gen = time.time() - t2
    decode_toks = len(generated) - 1  # exclude first which doesn't need decode
    tok_per_s = decode_toks / t_gen if t_gen > 0 else 0

    text = tokenizer.decode(generated)
    print(f"\nGenerated {len(generated)} tokens in {t_gen:.1f}s ({tok_per_s:.2f} tok/s)")
    print(f"--- Output ---")
    print(text)
    print(f"--- End ---")

    engine.cleanup()


if __name__ == "__main__":
    main()
