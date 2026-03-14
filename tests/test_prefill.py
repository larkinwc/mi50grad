#!/usr/bin/env python3
"""Test prefill_step (flash attention) vs sequential decode_step.

Validates correctness and performance of the prefill path.
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def test_prefill():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())

    print("=== Correctness: prefill vs sequential decode ===")
    all_pass = True

    # Use real tokenizer tokens: "The capital of France is" = [760, 6511, 314, 9338, 369]
    base_tokens = [760, 6511, 314, 9338, 369]

    for seq_len in [2, 4, 5, 8, 16]:
        # Repeat tokens to fill sequence
        token_ids = (base_tokens * ((seq_len // len(base_tokens)) + 1))[:seq_len]
        embeddings = embed[token_ids].copy()

        # Sequential decode (reference)
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        for i in range(seq_len):
            hidden_ref = engine.decode_step(embeddings[i], i)
        engine.device.synchronize()

        # Prefill (flash attention)
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        hidden_pf = engine.prefill_step(embeddings)
        engine.device.synchronize()

        max_err = np.max(np.abs(hidden_ref.astype(np.float32) -
                                 hidden_pf.astype(np.float32)))
        cos_sim = np.dot(hidden_ref.astype(np.float32).ravel(),
                         hidden_pf.astype(np.float32).ravel()) / (
            np.linalg.norm(hidden_ref.astype(np.float32)) *
            np.linalg.norm(hidden_pf.astype(np.float32)) + 1e-10)

        ok = cos_sim > 0.999 and max_err < 0.5
        print(f"  seq_len={seq_len:3d}: maxerr={max_err:.4f} cos={cos_sim:.6f} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    # Generation quality check with prefill
    print("\n=== Generation quality: prefill path ===")
    tokens = [760, 6511, 314, 9338, 369]  # "The capital of France is"
    embeddings = embed[tokens].copy()

    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()
    hidden = engine.prefill_step(embeddings)
    engine.device.synchronize()
    logits = engine.compute_logits()
    top5 = np.argsort(logits)[-5:][::-1]
    paris_rank = np.sum(logits > logits[11751]) + 1
    print(f"  Top-5 logits: {logits[top5]}")
    print(f"  ' Paris' (11751): logit={logits[11751]:.2f}, rank={paris_rank}")
    if paris_rank > 5:
        print(f"  WARNING: Paris not in top-5 (rank {paris_rank})")
        all_pass = False
    else:
        print(f"  PASS: Paris is rank {paris_rank}")

    print("\n=== Performance: prefill vs sequential decode ===")
    for seq_len in [4, 16, 64, 128]:
        token_ids = (base_tokens * ((seq_len // len(base_tokens)) + 1))[:seq_len]
        embeddings = embed[token_ids].copy()

        # Warmup prefill
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        engine.prefill_step(embeddings)
        engine.device.synchronize()

        # Time prefill
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        t0 = time.perf_counter()
        engine.prefill_step(embeddings)
        engine.device.synchronize()
        t_pf = time.perf_counter() - t0

        # Warmup sequential
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        for i in range(seq_len):
            engine.decode_step(embeddings[i], i)
        engine.device.synchronize()

        # Time sequential
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        t0 = time.perf_counter()
        for i in range(seq_len):
            engine.decode_step(embeddings[i], i)
        engine.device.synchronize()
        t_seq = time.perf_counter() - t0

        speedup = t_seq / t_pf if t_pf > 0 else 0
        pf_tok_s = seq_len / t_pf if t_pf > 0 else 0
        seq_tok_s = seq_len / t_seq if t_seq > 0 else 0
        print(f"  seq_len={seq_len:4d}: prefill={t_pf*1000:.1f}ms ({pf_tok_s:.1f} tok/s) "
              f"sequential={t_seq*1000:.1f}ms ({seq_tok_s:.1f} tok/s) "
              f"speedup={speedup:.2f}x")

    engine.cleanup()
    return all_pass


if __name__ == "__main__":
    ok = test_prefill()
    print(f"\n{'=== ALL TESTS PASSED ===' if ok else '=== SOME TESTS FAILED ==='}")
