#!/usr/bin/env python3
"""
Test tensor-parallel inference across multiple MI50 GPUs.

Validates that TP=2 produces the same output as single-GPU (TP=1).
Then benchmarks decode throughput.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import QwenConfig, load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine


MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"


def load_model_single(config, device_id=0):
    """Load model on a single GPU."""
    loader = QwenWeightLoader(MODEL_DIR, config=config,
                               bits=config.bits,
                               group_size=config.group_size)
    engine = InferenceEngine(config, device_id=device_id)

    print(f"Loading {config.num_hidden_layers} layers on GPU {device_id}...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
        if (i + 1) % 16 == 0:
            print(f"  Layer {i+1}/{config.num_hidden_layers}")

    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    return engine, loader


def load_model_tp(config, device_ids):
    """Load model with tensor parallelism."""
    loader = QwenWeightLoader(MODEL_DIR, config=config,
                               bits=config.bits,
                               group_size=config.group_size)
    engine = TPInferenceEngine(config, device_ids=device_ids)

    print(f"Loading {config.num_hidden_layers} layers on GPUs {device_ids} (TP={len(device_ids)})...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
        if (i + 1) % 16 == 0:
            print(f"  Layer {i+1}/{config.num_hidden_layers}")

    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    return engine, loader


def test_correctness(config, single_engine, tp_engine, loader):
    """Compare single-GPU vs TP outputs for correctness."""
    print("\n=== Correctness Test ===")

    embed_weight = loader.load_embedding()

    # Use a deterministic embedding
    np.random.seed(42)
    token_ids = [1234, 5678, 9012]  # 3 tokens

    # Process through single GPU
    single_engine.deltanet_state.reset()
    single_engine.kv_cache.current_len = 0

    for i, tid in enumerate(token_ids[:-1]):
        emb = embed_weight[tid].copy()
        single_engine.decode_step(emb, i)

    # Last token — get output
    last_emb = embed_weight[token_ids[-1]].copy()
    single_out = single_engine.decode_step(last_emb, len(token_ids) - 1)

    # Process through TP
    for eng in tp_engine.engines:
        eng.deltanet_state.reset()
        eng.kv_cache.current_len = 0

    for i, tid in enumerate(token_ids[:-1]):
        emb = embed_weight[tid].copy()
        tp_engine.decode_step(emb, i)

    tp_out = tp_engine.decode_step(last_emb, len(token_ids) - 1)

    # Compare
    cos_sim = np.dot(single_out.astype(np.float32), tp_out.astype(np.float32)) / (
        np.linalg.norm(single_out.astype(np.float32)) *
        np.linalg.norm(tp_out.astype(np.float32)) + 1e-10)
    max_diff = np.max(np.abs(single_out.astype(np.float32) - tp_out.astype(np.float32)))
    mean_diff = np.mean(np.abs(single_out.astype(np.float32) - tp_out.astype(np.float32)))

    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max abs diff:      {max_diff:.6f}")
    print(f"Mean abs diff:     {mean_diff:.6f}")

    if cos_sim > 0.99:
        print("PASS: TP output matches single-GPU")
    else:
        print("FAIL: TP output diverges from single-GPU")

    # Also compare logits
    single_logits = single_engine.compute_logits(single_out)
    tp_logits = tp_engine.compute_logits(tp_out)

    top_single = np.argsort(single_logits)[-5:][::-1]
    top_tp = np.argsort(tp_logits)[-5:][::-1]
    print(f"\nTop-5 tokens (single): {top_single}")
    print(f"Top-5 tokens (TP):     {top_tp}")

    match = sum(1 for a, b in zip(top_single, top_tp) if a == b)
    print(f"Top-5 overlap: {match}/5")

    return cos_sim > 0.99


def benchmark_tp(tp_engine, loader, num_steps=20):
    """Benchmark TP decode throughput."""
    print(f"\n=== TP={tp_engine.tp_size} Decode Benchmark ({num_steps} steps) ===")

    embed_weight = loader.load_embedding()

    # Reset state
    for eng in tp_engine.engines:
        eng.deltanet_state.reset()
        eng.kv_cache.current_len = 0

    # Warmup
    emb = embed_weight[1234].copy()
    for i in range(3):
        tp_engine.decode_step(emb, i)

    # Benchmark
    for eng in tp_engine.engines:
        eng.deltanet_state.reset()
        eng.kv_cache.current_len = 0

    t0 = time.perf_counter()
    for i in range(num_steps):
        tp_engine.decode_step(emb, i)
    tp_engine.synchronize()
    elapsed = time.perf_counter() - t0

    tok_s = num_steps / elapsed
    ms_per_tok = elapsed / num_steps * 1000
    print(f"Time: {elapsed*1000:.1f}ms for {num_steps} steps")
    print(f"Throughput: {tok_s:.1f} tok/s ({ms_per_tok:.1f} ms/tok)")
    return tok_s


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="0,1",
                        help="Comma-separated GPU device IDs for TP")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness test (single-GPU comparison)")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    args = parser.parse_args()

    device_ids = [int(x) for x in args.devices.split(",")]
    tp_size = len(device_ids)

    print(f"Loading config from {MODEL_DIR}")
    config = load_config_from_json(MODEL_DIR)
    print(f"Model: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, inter={config.intermediate_size}")
    print(f"TP size: {tp_size}, devices: {device_ids}")

    # Load TP model
    tp_engine, loader = load_model_tp(config, device_ids)

    if not args.skip_correctness:
        # Load single-GPU model for comparison
        # Use a device NOT in the TP group if possible
        single_dev = 2 if 2 not in device_ids else device_ids[0]
        print(f"\nLoading single-GPU reference on device {single_dev}...")
        single_engine, _ = load_model_single(config, device_id=single_dev)

        passed = test_correctness(config, single_engine, tp_engine, loader)
        single_engine.cleanup()

        if not passed:
            print("\nCorrectness test failed. Skipping benchmark.")
            tp_engine.cleanup()
            return

    # Benchmark
    tok_s = benchmark_tp(tp_engine, loader, args.benchmark_steps)

    tp_engine.cleanup()
    print(f"\nDone. TP={tp_size} decode: {tok_s:.1f} tok/s")


if __name__ == "__main__":
    main()
