#!/usr/bin/env python3
"""E2E generation test: load model, generate text, benchmark."""
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine
from src.inference.generate import TextGenerator
from src.inference.sampler import SamplingParams


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading model...")
    t0 = time.perf_counter()
    embed = loader.load_embedding()
    lm_head = loader.load_lm_head()

    engine = InferenceEngine(config, device_id=0, max_seq_len=2048)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print("Model loaded in %.1fs" % (time.perf_counter() - t0))

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    gen = TextGenerator(engine, embed, lm_head, tokenizer)

    # Generation quality tests
    print("\n=== Generation Quality ===")
    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n",
        "In quantum mechanics, the uncertainty principle states that",
    ]

    params = SamplingParams(temperature=0, max_tokens=50)
    for prompt in prompts:
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        text = gen.generate(prompt, params)
        print("Prompt: %r" % prompt)
        print("Output: %s" % text[:200])
        print()

    # Benchmark
    print("=== Benchmark ===")
    for prompt_len_target in [16, 64, 128]:
        base = "The quick brown fox jumps over the lazy dog. "
        prompt = base * (prompt_len_target // 10 + 1)
        input_ids = tokenizer.encode(prompt)[:prompt_len_target]
        prompt = tokenizer.decode(input_ids)

        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        result = gen.benchmark(prompt, max_tokens=50)

        print("  prompt=%d tokens:" % result['prefill_tokens'])
        print("    Prefill: %.1fms (%.1f tok/s)" % (
            result['prefill_time_ms'], result['prefill_tok_s']))
        print("    Decode:  %.1fms / %d tokens (%.1f tok/s)" % (
            result['decode_time_ms'], result['decode_tokens'], result['decode_tok_s']))
        print("    TTFT:    %.1fms" % result['ttft_ms'])

    gen.cleanup()
    engine.cleanup()


if __name__ == "__main__":
    main()
