#!/usr/bin/env python3
"""
TP=4 decode correctness test.

Validates that TP=4 decode with P2P allreduce produces output that is
numerically equivalent to single-GPU decode.

Tests:
1. Loads Qwen3.5-27B-GPTQ-Int4 on 4 GPUs (TPInferenceEngine, device_ids=[0,1,2,3])
2. Loads same model on single GPU (device 0, InferenceEngine)
3. Runs 10 decode steps with same input on both
4. Compares outputs: cosine similarity > 0.99 after each step
5. Reports per-step cosine similarity to detect numerical drift

USAGE:
    python3 tests/test_tp4_correctness.py

IMPORTANT: Must stop vLLM first (docker stop vllm-mobydick) and use
    -e HIP_VISIBLE_DEVICES=0,1,2,3 in Docker run command.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
NUM_DECODE_STEPS = 10
COSINE_SIM_THRESHOLD = 0.99
DEVICE_IDS_TP4 = [0, 1, 2, 3]
DEVICE_ID_SINGLE = 0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two FP16 vectors.
    Returns nan if either vector contains NaN values.
    Returns 0.0 if either vector has zero norm.
    """
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    # Check for NaN in either vector
    if np.any(np.isnan(a32)) or np.any(np.isnan(b32)):
        return float('nan')
    dot = float(np.dot(a32, b32))
    norm_a = float(np.linalg.norm(a32))
    norm_b = float(np.linalg.norm(b32))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def load_tp4_engine(config, loader):
    """Load TPInferenceEngine with 4 GPUs."""
    print(f"\nLoading TP=4 engine on GPUs {DEVICE_IDS_TP4}...")
    t0 = time.perf_counter()
    tp_engine = TPInferenceEngine(config, DEVICE_IDS_TP4)
    for i in range(config.num_hidden_layers):
        if i % 8 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        tp_engine.load_layer_weights(i, loader.load_layer(i))
    tp_engine.load_final_norm(loader.load_final_norm())
    tp_engine.load_lm_head(loader.load_lm_head())
    elapsed = time.perf_counter() - t0
    print(f"TP=4 engine loaded in {elapsed:.1f}s")
    return tp_engine


def load_single_gpu_engine(config, loader):
    """Load single-GPU InferenceEngine."""
    print(f"\nLoading single-GPU engine on device {DEVICE_ID_SINGLE}...")
    t0 = time.perf_counter()
    engine = InferenceEngine(config, device_id=DEVICE_ID_SINGLE)
    for i in range(config.num_hidden_layers):
        if i % 8 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    elapsed = time.perf_counter() - t0
    print(f"Single-GPU engine loaded in {elapsed:.1f}s")
    return engine


def reset_tp_engine(tp_engine):
    """Reset TP engine state for fresh decode."""
    for e in tp_engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


def reset_single_engine(engine):
    """Reset single-GPU engine state for fresh decode."""
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()


def main():
    print("=" * 70)
    print("TP=4 Decode Correctness Test")
    print("=" * 70)
    print(f"Model: {MODEL_DIR}")
    print(f"TP=4 GPUs: {DEVICE_IDS_TP4}")
    print(f"Single-GPU device: {DEVICE_ID_SINGLE}")
    print(f"Decode steps: {NUM_DECODE_STEPS}")
    print(f"Cosine similarity threshold: {COSINE_SIM_THRESHOLD}")

    # Check GPU count
    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"\nGPUs visible: {n_gpus}")
    if n_gpus < 4:
        print(f"ERROR: Need 4 GPUs for TP=4, only {n_gpus} visible.")
        print("Make sure to use: -e HIP_VISIBLE_DEVICES=0,1,2,3")
        sys.exit(1)

    # Load config
    print(f"\nLoading config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"Config: {config.num_hidden_layers} layers, "
          f"hidden_size={config.hidden_size}, "
          f"num_attention_heads={config.num_attention_heads}, "
          f"num_kv_heads={config.num_key_value_heads}")
    print(f"  TP=4 sharding: {config.num_attention_heads}→"
          f"{config.num_attention_heads//4} attn heads per GPU")
    print(f"  TP=4 sharding: {config.num_key_value_heads}→"
          f"{config.num_key_value_heads//4} kv heads per GPU")
    print(f"  TP=4 sharding: {config.linear_num_key_heads}→"
          f"{config.linear_num_key_heads//4} linear k heads per GPU")
    print(f"  TP=4 sharding: {config.linear_num_value_heads}→"
          f"{config.linear_num_value_heads//4} linear v heads per GPU")
    print(f"  TP=4 sharding: {config.intermediate_size}→"
          f"{config.intermediate_size//4} intermediate size per GPU")

    # Create weight loader
    loader = QwenWeightLoader(MODEL_DIR, config)

    # Load single-GPU engine FIRST (uses GPU 0, will be freed before TP engine)
    print("\n--- Phase 1: Single-GPU Reference Decode ---")
    single_engine = load_single_gpu_engine(config, loader)

    # Fixed random input for reproducibility
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # Warm up single GPU engine
    print("\nWarming up single-GPU engine (1 step)...")
    reset_single_engine(single_engine)
    _ = single_engine.decode_step(emb, 0)
    reset_single_engine(single_engine)

    # Run 10 decode steps on single GPU and collect outputs
    print(f"\nRunning {NUM_DECODE_STEPS} decode steps on single GPU...")
    single_outputs = []
    reset_single_engine(single_engine)
    for step in range(NUM_DECODE_STEPS):
        out = single_engine.decode_step(emb, step)
        single_outputs.append(out.copy())
    single_engine.device.synchronize()
    print(f"Single-GPU decode complete.")

    # Free single-GPU engine to reduce VRAM usage during TP=4 load
    print("\nFreeing single-GPU engine (to free VRAM for TP=4)...")
    single_engine.cleanup()
    del single_engine

    # Load TP=4 engine
    print("\n--- Phase 2: TP=4 Decode ---")
    tp_engine = load_tp4_engine(config, loader)

    # Warm up TP=4 engine
    print("\nWarming up TP=4 engine (1 step)...")
    reset_tp_engine(tp_engine)
    _ = tp_engine.decode_step(emb, 0)
    reset_tp_engine(tp_engine)

    # Run 10 decode steps on TP=4 and collect outputs
    print(f"\nRunning {NUM_DECODE_STEPS} decode steps on TP=4...")
    tp_outputs = []
    reset_tp_engine(tp_engine)
    for step in range(NUM_DECODE_STEPS):
        out = tp_engine.decode_step(emb, step)
        tp_outputs.append(out.copy())
    tp_engine.synchronize()
    print(f"TP=4 decode complete.")

    # --- Correctness Comparison ---
    print("\n" + "=" * 70)
    print("CORRECTNESS COMPARISON: TP=4 vs Single-GPU")
    print("=" * 70)
    print(f"{'Step':>4}  {'Cosine Sim':>12}  {'Status':>10}  {'Max|diff|':>12}")
    print("-" * 54)

    all_pass = True
    min_cosine = 1.0
    max_cosine = 0.0

    for step in range(NUM_DECODE_STEPS):
        ref = single_outputs[step]
        tp = tp_outputs[step]
        cos_sim = cosine_similarity(ref, tp)
        ref_nan = int(np.sum(np.isnan(ref.astype(np.float32))))
        tp_nan = int(np.sum(np.isnan(tp.astype(np.float32))))
        if np.isnan(cos_sim):
            max_diff = float('nan')
            status = "FAIL(NaN)"
            all_pass = False
        else:
            max_diff = float(np.max(np.abs(ref.astype(np.float32) - tp.astype(np.float32))))
            if cos_sim >= COSINE_SIM_THRESHOLD:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
        min_cosine = min(min_cosine, cos_sim) if not np.isnan(cos_sim) else min_cosine
        max_cosine = max(max_cosine, cos_sim) if not np.isnan(cos_sim) else max_cosine
        if np.isnan(cos_sim):
            print(f"{step:>4}  {'nan':>12}  {status:>10}  {'nan':>12}  ref_nan={ref_nan} tp_nan={tp_nan}")
        else:
            print(f"{step:>4}  {cos_sim:>12.6f}  {status:>10}  {max_diff:>12.4e}")

    print("-" * 54)
    print(f"Min cosine similarity: {min_cosine:.6f}")
    print(f"Max cosine similarity: {max_cosine:.6f}")
    print(f"Threshold: {COSINE_SIM_THRESHOLD}")

    print("\n" + "=" * 70)
    if all_pass:
        print(f"RESULT: PASS — All {NUM_DECODE_STEPS} steps have cosine sim >= {COSINE_SIM_THRESHOLD}")
        print(f"  TP=4 decode is correct with P2P allreduce!")
    else:
        print(f"RESULT: FAIL — Some steps have cosine sim < {COSINE_SIM_THRESHOLD}")
        print(f"  Min cosine sim: {min_cosine:.6f} (threshold: {COSINE_SIM_THRESHOLD})")
    print("=" * 70)

    # Cleanup
    tp_engine.cleanup()

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
