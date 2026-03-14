#!/usr/bin/env python3
"""Quick correctness check: verify 'Paris' is top prediction."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    # Tokenize "The capital of France is"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokens = tok.encode("The capital of France is", add_special_tokens=False)
    print(f"Tokens: {tokens} = {[tok.decode([t]) for t in tokens]}")

    # Feed tokens sequentially (decode mode)
    for i, t in enumerate(tokens):
        engine.decode_step(embed[t].copy(), position=i)

    # Compute logits from final hidden state
    logits = engine.compute_logits()
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\nTop 5 predictions:")
    for i, tid in enumerate(top5):
        print(f"  #{i+1}: '{tok.decode([tid])}' (id={tid}, logit={logits[tid]:.2f})")

    # Check Paris is #1
    paris_tokens = tok.encode("Paris", add_special_tokens=False)
    if top5[0] in paris_tokens or "Paris" in tok.decode([top5[0]]):
        print("\n=== CORRECTNESS PASSED ===")
    else:
        print(f"\n=== UNEXPECTED TOP TOKEN (expected Paris) ===")

    engine.cleanup()


if __name__ == "__main__":
    main()
