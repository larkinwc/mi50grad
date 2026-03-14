#!/usr/bin/env python3
"""Debug logit quality by examining top tokens and comparing with numpy reference."""

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

    # Simple prompt
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt: {repr(prompt)}")
    print(f"Tokens: {tokens}")

    # Prefill
    h = config.hidden_size
    for i, tid in enumerate(tokens):
        emb = embed[tid].copy()
        hidden = engine.decode_step(emb, position=i)

    # Check hidden state
    print(f"\nHidden after prefill:")
    print(f"  range: [{hidden.min():.4f}, {hidden.max():.4f}]")
    print(f"  mean: {hidden.astype(np.float32).mean():.6f}")
    print(f"  std: {hidden.astype(np.float32).std():.6f}")
    print(f"  NaN: {np.sum(np.isnan(hidden))}")
    print(f"  Inf: {np.sum(np.isinf(hidden))}")

    # Compute logits via CPU (reference)
    logits_cpu = lm_head.astype(np.float32) @ hidden.astype(np.float32)
    print(f"\nCPU logits:")
    print(f"  range: [{logits_cpu.min():.4f}, {logits_cpu.max():.4f}]")
    print(f"  mean: {logits_cpu.mean():.6f}")
    print(f"  std: {logits_cpu.std():.6f}")

    # Top tokens (CPU)
    top_indices = np.argsort(logits_cpu)[-20:][::-1]
    print(f"\nTop 20 tokens (CPU logits):")
    for i, idx in enumerate(top_indices):
        tok_text = tokenizer.decode([idx])
        print(f"  {i+1}. token={idx} ({repr(tok_text)}): logit={logits_cpu[idx]:.4f}")

    # Compute logits via GPU
    logits_gpu = engine.compute_logits()
    print(f"\nGPU logits:")
    print(f"  range: [{logits_gpu.min():.4f}, {logits_gpu.max():.4f}]")
    print(f"  mean: {logits_gpu.mean():.6f}")
    print(f"  std: {logits_gpu.std():.6f}")

    # Top tokens (GPU)
    top_indices_gpu = np.argsort(logits_gpu)[-20:][::-1]
    print(f"\nTop 20 tokens (GPU logits):")
    for i, idx in enumerate(top_indices_gpu):
        tok_text = tokenizer.decode([idx])
        print(f"  {i+1}. token={idx} ({repr(tok_text)}): logit={logits_gpu[idx]:.4f}")

    # Compare
    diff = np.abs(logits_cpu - logits_gpu)
    print(f"\nCPU vs GPU logit diff: max={diff.max():.4f}, mean={diff.mean():.6f}")

    engine.cleanup()


if __name__ == "__main__":
    main()
