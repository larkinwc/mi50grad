#!/usr/bin/env python3
"""
Validate VAL-AR-004 and VAL-AR-005: Double-buffer correctness and throughput.

Tests:
- VAL-AR-004: Double-buffer produces cosine sim >= 0.99 vs standard path
- VAL-AR-005: Double-buffer achieves >= 1.05x speedup vs standard path
"""

import sys
sys.path.insert(0, "/opt/mi50grad")

import numpy as np
import time

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
NUM_WARMUP = 5
NUM_STEPS = 10
MAX_SEQ_LEN = 64

def main():
    print("=" * 70)
    print("Double-Buffer Validation: VAL-AR-004 and VAL-AR-005")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Load weights once
    print("\nLoading model weights...")
    loader = QwenWeightLoader(MODEL_DIR, config)
    
    # Load all layers
    layers_weights = []
    for layer_idx in range(config.num_hidden_layers):
        layer_weights = loader.load_layer(layer_idx)
        layers_weights.append(layer_weights)
    final_norm = loader.load_final_norm()
    lm_head = loader.load_lm_head()
    print(f"  Loaded {len(layers_weights)} layers")
    
    # Test 1: Standard path
    print("\n" + "-" * 70)
    print("Creating TP=4 engine (standard mode)...")
    print("-" * 70)
    
    tp_std = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    # Check double-buffer support
    if not hasattr(tp_std, 'set_double_buffer_enabled'):
        print("ERROR: set_double_buffer_enabled not found")
        return False
    
    # Use cached+stream dispatch (not C dispatch, which doesn't support double-buffer)
    tp_std.set_cached_dispatch(True)
    tp_std.set_stream_overlap_dispatch(True)
    
    # Load weights
    for layer_idx, layer_weights in enumerate(layers_weights):
        tp_std.load_layer_weights(layer_idx, layer_weights)
    tp_std.load_final_norm(final_norm)
    tp_std.load_lm_head(lm_head)
    tp_std.build_dispatch_cache()
    
    print(f"  Double buffer enabled: {tp_std._double_buffer_enabled}")
    
    # Warmup
    print(f"\nWarming up ({NUM_WARMUP} steps)...")
    rng = np.random.default_rng(42)
    for i in range(NUM_WARMUP):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_std.decode_step(emb, i)
    tp_std._hip.synchronize()
    
    # Benchmark standard path
    print(f"Benchmarking standard path ({NUM_STEPS} steps)...")
    times_std = []
    outputs_std = []
    for i in range(NUM_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        t0 = time.perf_counter()
        out = tp_std.decode_step(emb, NUM_WARMUP + i)
        tp_std._hip.synchronize()
        t1 = time.perf_counter()
        times_std.append((t1 - t0) * 1000)
        outputs_std.append(out.copy())
    
    median_std = np.median(times_std)
    print(f"  Standard path: {median_std:.2f} ms/step (median)")
    
    # Cleanup
    tp_std.cleanup()
    
    # Test 2: Double-buffer path
    print("\n" + "-" * 70)
    print("Creating TP=4 engine (double-buffer mode)...")
    print("-" * 70)
    
    tp_db = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_db.set_double_buffer_enabled(True)
    tp_db.set_cached_dispatch(True)
    tp_db.set_stream_overlap_dispatch(True)
    
    for layer_idx, layer_weights in enumerate(layers_weights):
        tp_db.load_layer_weights(layer_idx, layer_weights)
    tp_db.load_final_norm(final_norm)
    tp_db.load_lm_head(lm_head)
    tp_db.build_dispatch_cache()
    
    print(f"  Double buffer enabled: {tp_db._double_buffer_enabled}")
    
    # Warmup
    print(f"\nWarming up ({NUM_WARMUP} steps)...")
    rng = np.random.default_rng(42)  # Same seed
    for i in range(NUM_WARMUP):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_db.decode_step(emb, i)
    tp_db._hip.synchronize()
    
    # Benchmark double-buffer path
    print(f"Benchmarking double-buffer path ({NUM_STEPS} steps)...")
    times_db = []
    outputs_db = []
    cos_sims = []
    for i in range(NUM_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        t0 = time.perf_counter()
        out = tp_db.decode_step(emb, NUM_WARMUP + i)
        tp_db._hip.synchronize()
        t1 = time.perf_counter()
        times_db.append((t1 - t0) * 1000)
        outputs_db.append(out.copy())
        
        # Compute cosine similarity with standard output
        out_std = outputs_std[i].astype(np.float32)
        out_db_fp = out.astype(np.float32)
        dot = np.dot(out_std, out_db_fp)
        norm_std = np.linalg.norm(out_std)
        norm_db = np.linalg.norm(out_db_fp)
        cos_sim = dot / (norm_std * norm_db + 1e-8)
        cos_sims.append(cos_sim)
    
    median_db = np.median(times_db)
    print(f"  Double-buffer path: {median_db:.2f} ms/step (median)")
    
    # Cleanup
    tp_db.cleanup()
    
    # Results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    
    # VAL-AR-004: Correctness
    min_cos_sim = min(cos_sims)
    avg_cos_sim = np.mean(cos_sims)
    print(f"\nVAL-AR-004 (Correctness):")
    print(f"  Min cosine similarity: {min_cos_sim:.6f}")
    print(f"  Avg cosine similarity: {avg_cos_sim:.6f}")
    print(f"  Threshold: >= 0.99")
    val_ar_004_pass = min_cos_sim >= 0.99
    print(f"  Result: {'PASS' if val_ar_004_pass else 'FAIL'}")
    
    # VAL-AR-005: Throughput
    speedup = median_std / median_db if median_db > 0 else 1.0
    print(f"\nVAL-AR-005 (Throughput):")
    print(f"  Standard: {median_std:.2f} ms/step")
    print(f"  Double-buffer: {median_db:.2f} ms/step")
    print(f"  Speedup: {speedup:.3f}x")
    print(f"  Threshold: >= 1.05x")
    val_ar_005_pass = speedup >= 1.05
    print(f"  Result: {'PASS' if val_ar_005_pass else 'FAIL'}")
    
    # Summary
    print("\n" + "=" * 70)
    all_pass = val_ar_004_pass and val_ar_005_pass
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 70)
    
    return all_pass

if __name__ == "__main__":
    try:
        passed = main()
        sys.exit(0 if passed else 1)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
