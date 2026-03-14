#!/usr/bin/env python3
"""Perplexity evaluation on WikiText-2 test set.

Downloads WikiText-2 if not present, evaluates using sliding window.
"""
import sys
import time
import math
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def load_wikitext2(model_dir, max_tokens=2048):
    """Load WikiText-2 test set, tokenize, return token IDs."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Try loading from datasets
    data_path = Path("/opt/data/wikitext-2-raw/wiki.test.raw")
    if data_path.exists():
        text = data_path.read_text()
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n".join(ds["text"])
            # Cache it
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(text)
        except ImportError:
            # Fallback: use a fixed text sample
            print("WARNING: datasets not available, using small test sample")
            text = ("The tower is 324 metres (1,063 ft) tall, about the same height "
                    "as an 81-storey building, and the tallest structure in Paris. "
                    "Its base is square, measuring 125 metres (410 ft) on each side. "
                    "During its construction, the Eiffel Tower surpassed the Washington "
                    "Monument to become the tallest man-made structure in the world. ")

    token_ids = tokenizer.encode(text)
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]
    return token_ids, tokenizer


def evaluate_perplexity(engine, embed, token_ids, stride=512, max_len=512):
    """Calculate perplexity using sliding window approach.

    For each window of max_len tokens, run prefill and compute logits for
    each position, then calculate cross-entropy loss.
    """
    vocab_size = engine.lm_head_vocab
    total_loss = 0.0
    total_tokens = 0

    n_windows = max(1, (len(token_ids) - 1) // stride)
    print("Evaluating %d tokens in %d windows (stride=%d, window=%d)" % (
        len(token_ids), n_windows, stride, max_len))

    for i in range(0, len(token_ids) - 1, stride):
        window_ids = token_ids[i:i + max_len]
        if len(window_ids) < 2:
            break

        embeddings = embed[window_ids].copy()

        # Reset state
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()

        # Process all tokens
        seq_len = len(window_ids)
        if seq_len <= 1:
            break

        # Use prefill for the whole window
        hidden = engine.prefill_step(embeddings)
        engine.device.synchronize()

        # For perplexity, we need logits at each position
        # But prefill only returns the last hidden state
        # So we do sequential decode instead
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()

        window_loss = 0.0
        window_count = 0

        for pos in range(len(window_ids) - 1):
            emb = embeddings[pos]
            hidden = engine.decode_step(emb, pos)
            engine.device.synchronize()

            # Get logits
            logits = engine.compute_logits()

            # Cross-entropy for next token
            target = window_ids[pos + 1]
            # Numerically stable log-softmax
            logits_f64 = logits.astype(np.float64)
            max_logit = np.max(logits_f64)
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits_f64 - max_logit)))
            log_prob = logits_f64[target] - log_sum_exp

            window_loss -= log_prob
            window_count += 1

        total_loss += window_loss
        total_tokens += window_count

        ppl_so_far = math.exp(total_loss / total_tokens)
        print("  Window %d/%d: loss=%.4f, running_ppl=%.2f (%d tokens)" % (
            i // stride + 1, n_windows, window_loss / window_count,
            ppl_so_far, total_tokens))

        if total_tokens >= 500:  # Enough for a reasonable estimate
            break

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity, total_tokens


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading model...")
    embed = loader.load_embedding()

    engine = InferenceEngine(config, device_id=0, max_seq_len=2048)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print("Model loaded.")

    # Load test data
    print("\nLoading WikiText-2...")
    token_ids, tokenizer = load_wikitext2(model_dir, max_tokens=2048)
    print("Loaded %d tokens" % len(token_ids))

    # Evaluate
    print("\n=== Perplexity Evaluation ===")
    t0 = time.perf_counter()
    ppl, n_tokens = evaluate_perplexity(engine, embed, token_ids,
                                         stride=256, max_len=256)
    elapsed = time.perf_counter() - t0

    print("\n=== Results ===")
    print("Perplexity: %.2f" % ppl)
    print("Tokens evaluated: %d" % n_tokens)
    print("Time: %.1fs (%.1f tok/s)" % (elapsed, n_tokens / elapsed))
    print("\nExpected range for Qwen 3.5 27B INT4: 5-8 PPL on WikiText-2")

    engine.cleanup()


if __name__ == "__main__":
    main()
