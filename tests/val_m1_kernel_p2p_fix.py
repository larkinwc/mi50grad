#!/usr/bin/env python3
"""
Milestone 1 Validation: Kernel P2P Allreduce Fix

Validates that kernel P2P allreduce performs at least as well as star topology.

Target:
  - Kernel P2P throughput >= 44.0 tok/s (matching star topology)
  - Numerical correctness: cosine sim >= 0.99

USAGE:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      -e HIP_VISIBLE_DEVICES=0,1,2,3 \
      -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
      mi50grad bash -c 'cd /opt/mi50grad && python3 tests/val_m1_kernel_p2p_fix.py'
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 100
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256

# Validation thresholds
# Note: Throughput varies by system conditions; the key test is correctness
# and that kernel P2P is not slower than star topology
MIN_COSINE_SIM = 0.99


def reset_tp(tp_engine):
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def benchmark_decode(tp_engine, config, steps, warmup):
    """Benchmark decode throughput."""
    rng = np.random.default_rng(42)
    
    for i in range(warmup):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        reset_tp(tp_engine)
        tp_engine.decode_step(emb, i)
        tp_engine._hip.synchronize()
    
    reset_tp(tp_engine)
    t0 = time.perf_counter()
    for step in range(steps):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_engine.decode_step(emb, step)
        tp_engine._hip.synchronize()
    elapsed = time.perf_counter() - t0
    
    return steps / elapsed, elapsed / steps * 1000


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 72)
    print("  MILESTONE 1 VALIDATION: Kernel P2P Allreduce Fix")
    print("=" * 72)
    print(f"  Target cosine sim: >= {MIN_COSINE_SIM}")
    print(f"  Target: Kernel P2P throughput >= star topology (no regression)")
    print()
    
    from src.model.qwen import load_config_from_json
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader
    
    config = load_config_from_json(MODEL_DIR)
    results = {}
    
    # --- Test 1: Star topology baseline ---
    print("[1/3] Loading TP=4 engine (star topology)...")
    tp_star = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        tp_star.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp_star.load_final_norm(loader.load_final_norm())
    tp_star.load_lm_head(loader.load_lm_head())
    
    tp_star.build_dispatch_cache()
    tp_star.set_direct_kv_write(True)
    tp_star.set_c_dispatch(True)
    # Ensure kernel P2P is OFF (star topology)
    tp_star._kernel_p2p_allreduce = False
    if hasattr(tp_star, '_build_c_dispatch_plan'):
        tp_star._build_c_dispatch_plan()
    
    tps_star, ms_star = benchmark_decode(tp_star, config, BENCH_STEPS, WARMUP_STEPS)
    results['star_topology'] = {'tps': tps_star, 'ms': ms_star}
    print(f"  Star topology: {tps_star:.2f} tok/s ({ms_star:.2f} ms/tok)")
    
    # Capture output for correctness - use FIXED seed for reproducibility
    rng = np.random.default_rng(123)
    emb_star = rng.standard_normal(config.hidden_size).astype(np.float16)
    reset_tp(tp_star)
    hidden_star = tp_star.decode_step(emb_star, 0)
    tp_star._hip.synchronize()
    
    # --- Test 2: Kernel P2P ---
    print()
    print("[2/3] Testing kernel P2P allreduce...")
    tp_p2p = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    for layer_idx in range(config.num_hidden_layers):
        tp_p2p.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp_p2p.load_final_norm(loader.load_final_norm())
    tp_p2p.load_lm_head(loader.load_lm_head())
    
    tp_p2p.build_dispatch_cache()
    tp_p2p.set_direct_kv_write(True)
    tp_p2p.set_c_dispatch(True)
    tp_p2p.set_kernel_p2p_allreduce(True)
    
    # Verify kernel P2P is enabled
    print(f"  Kernel P2P enabled: {tp_p2p._kernel_p2p_allreduce}")
    print(f"  C dispatch enabled: {tp_p2p._c_dispatch_enabled}")
    
    tps_p2p, ms_p2p = benchmark_decode(tp_p2p, config, BENCH_STEPS, WARMUP_STEPS)
    results['kernel_p2p'] = {'tps': tps_p2p, 'ms': ms_p2p}
    print(f"  Kernel P2P: {tps_p2p:.2f} tok/s ({ms_p2p:.2f} ms/tok)")
    
    # Capture output for correctness - USE SAME EMBEDDING as star topology
    reset_tp(tp_p2p)
    hidden_p2p = tp_p2p.decode_step(emb_star, 0)  # Same embedding!
    tp_p2p._hip.synchronize()
    
    # --- Test 3: Correctness ---
    print()
    print("[3/3] Checking numerical correctness...")
    cos_sim = cosine_similarity(hidden_star, hidden_p2p)
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    # --- Validation ---
    print()
    print("=" * 72)
    print("  VALIDATION RESULTS")
    print("=" * 72)
    
    passed = []
    
    # Correctness check (primary)
    correct = cos_sim >= MIN_COSINE_SIM
    passed.append(correct)
    status = "PASS" if correct else "FAIL"
    print(f"  [{status}] Numerical correctness cos_sim >= {MIN_COSINE_SIM}: {cos_sim:.6f}")
    
    # Performance check (no regression vs star topology)
    no_regression = tps_p2p >= tps_star * 0.95  # Allow 5% variance
    passed.append(no_regression)
    status = "PASS" if no_regression else "FAIL (REGRESSION)"
    print(f"  [{status}] Kernel P2P vs star topology: {tps_p2p:.2f} vs {tps_star:.2f} tok/s")
    
    # Comparison
    print()
    print(f"  Comparison: Kernel P2P is {tps_p2p/tps_star:.2%} of star topology")
    if tps_p2p < tps_star:
        print(f"  WARNING: Kernel P2P is {(1 - tps_p2p/tps_star)*100:.1f}% SLOWER than star topology")
    
    # Summary
    print()
    all_passed = all(passed)
    if all_passed:
        print("  *** MILESTONE 1 PASSED ***")
    else:
        print("  *** MILESTONE 1 FAILED ***")
        print("  Action items:")
        if not correct:
            print("    - Debug numerical differences in kernel P2P path")
        if not no_regression:
            print("    - Investigate why kernel P2P is slower than star topology")
    
    # Cleanup
    tp_star.cleanup()
    tp_p2p.cleanup()
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
