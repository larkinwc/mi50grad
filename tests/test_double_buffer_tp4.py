#!/usr/bin/env python3
"""
Comprehensive double-buffer validation for TP=4 on 4x MI50.

Validates all VAL-DB assertions:
- VAL-DB-001: Buffer swap alternation (even layers write to B, odd to A)
- VAL-DB-002: Numerical correctness (cosine similarity >= 0.99 vs standard path)
- VAL-DB-003: Throughput with stream overlap (>= 5% improvement)
- VAL-DB-004: Long-run stability (1000+ tokens without NaN/Inf)
- VAL-DB-005: C dispatch interaction (C dispatch takes precedence)

Usage:
  python3 tests/test_double_buffer_tp4.py --all
  python3 tests/test_double_buffer_tp4.py --correctness
  python3 tests/test_double_buffer_tp4.py --benchmark
  python3 tests/test_double_buffer_tp4.py --stability
  python3 tests/test_double_buffer_tp4.py --buffer-swap
  python3 tests/test_double_buffer_tp4.py --c-dispatch
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
MAX_SEQ_LEN = 64


def test_buffer_swap_alternation():
    """VAL-DB-001: Verify buffer swap alternation.
    
    Even layers read from d_hidden_A and write to d_hidden_B.
    Odd layers read from d_hidden_B and write to d_hidden_A.
    After 64 layers, d_hidden should point to d_hidden_A (started with A).
    """
    print("=" * 70)
    print("VAL-DB-001: Buffer Swap Alternation Test")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Create TP engine with double-buffer enabled
    print("\nCreating TP=4 engine with double-buffer...")
    tp_engine = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_engine.set_double_buffer_enabled(True)
    
    # Load minimal weights for testing
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        layer_weights = loader.load_layer(layer_idx)
        tp_engine.load_layer_weights(layer_idx, layer_weights)
    tp_engine.load_final_norm(loader.load_final_norm())
    tp_engine.load_lm_head(loader.load_lm_head())
    
    # Build dispatch cache with cached+stream overlap (needed for double-buffer path)
    tp_engine.set_cached_dispatch(True)
    tp_engine.set_stream_overlap_dispatch(True)
    tp_engine.build_dispatch_cache()
    
    # Manually verify buffer alternation by simulating the decode loop
    print("\nSimulating buffer alternation for 64 layers...")
    
    # Get initial buffer addresses
    e0 = tp_engine.engines[0]
    addr_A = e0.d_hidden_A
    addr_B = e0.d_hidden_B
    
    print(f"  d_hidden_A address: 0x{addr_A:x}")
    print(f"  d_hidden_B address: 0x{addr_B:x}")
    
    # Track buffer usage per layer
    buffer_pattern = []
    
    # Simulate what _decode_step_cached_stream does
    for layer_idx in range(config.num_hidden_layers):
        # At start of layer (after swap from previous layer)
        read_buf = e0.d_hidden
        write_buf = e0.d_hidden_write
        
        buffer_pattern.append({
            'layer': layer_idx,
            'read': 'A' if read_buf == addr_A else 'B',
            'write': 'A' if write_buf == addr_A else 'B'
        })
        
        # Swap buffers (as done at end of each layer)
        tp_engine.engines[0]._swap_hidden_buffers()
        tp_engine.engines[1]._swap_hidden_buffers()
        tp_engine.engines[2]._swap_hidden_buffers()
        tp_engine.engines[3]._swap_hidden_buffers()
    
    # Verify pattern
    print("\nBuffer usage pattern (first 8 layers):")
    for i in range(min(8, len(buffer_pattern))):
        bp = buffer_pattern[i]
        expected_read = 'A' if bp['layer'] % 2 == 0 else 'B'
        expected_write = 'B' if bp['layer'] % 2 == 0 else 'A'
        match = (bp['read'] == expected_read and bp['write'] == expected_write)
        status = "✓" if match else "✗"
        print(f"  Layer {bp['layer']:2d}: read={bp['read']}, write={bp['write']} {status}")
    
    # Check all layers
    all_correct = True
    for bp in buffer_pattern:
        expected_read = 'A' if bp['layer'] % 2 == 0 else 'B'
        expected_write = 'B' if bp['layer'] % 2 == 0 else 'A'
        if bp['read'] != expected_read or bp['write'] != expected_write:
            print(f"  ERROR at layer {bp['layer']}: expected read={expected_read}, write={expected_write}")
            all_correct = False
    
    # After 64 layers (even number), should be back to A
    final_read = 'A' if e0.d_hidden == addr_A else 'B'
    print(f"\nAfter 64 layers:")
    print(f"  Final d_hidden: {final_read} (expected: A)")
    
    if final_read != 'A':
        print(f"  ERROR: Expected d_hidden to be A after even number of layers")
        all_correct = False
    
    # Cleanup (don't fully cleanup, we'll reuse the engine)
    tp_engine.cleanup()
    
    print("\n" + "=" * 70)
    if all_correct:
        print("VAL-DB-001: PASS - Buffer alternation correct")
    else:
        print("VAL-DB-001: FAIL - Buffer alternation incorrect")
    print("=" * 70)
    
    return all_correct


def test_numerical_correctness(num_steps: int = 20):
    """VAL-DB-002: Test numerical equivalence vs standard path.
    
    Double-buffer decode must produce numerically equivalent output
    to standard single-buffer path with cosine similarity >= 0.99.
    """
    print("=" * 70)
    print("VAL-DB-002: Numerical Correctness Test")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Load weights once
    print("\nLoading model weights...")
    loader = QwenWeightLoader(MODEL_DIR, config)
    layers_weights = []
    for layer_idx in range(config.num_hidden_layers):
        layers_weights.append(loader.load_layer(layer_idx))
    final_norm = loader.load_final_norm()
    lm_head = loader.load_lm_head()
    print(f"  Loaded {len(layers_weights)} layers")
    
    # Create standard engine
    print("\nCreating standard TP=4 engine...")
    tp_std = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    # Load weights FIRST before setting dispatch modes
    print(f"  Loading {len(layers_weights)} layers...")
    for layer_idx, lw in enumerate(layers_weights):
        tp_std.load_layer_weights(layer_idx, lw)
    tp_std.load_final_norm(final_norm)
    tp_std.load_lm_head(lm_head)
    # Now set dispatch modes (will build cache with loaded layers)
    tp_std.set_cached_dispatch(True)
    tp_std.set_stream_overlap_dispatch(True)
    # Explicitly build cache if not already built
    if not tp_std._engine_layer_caches:
        tp_std.build_dispatch_cache()
    print(f"  Standard engine ready: {len(tp_std.engines[0].layers)} layers")
    
    # Create double-buffer engine
    print("Creating double-buffer TP=4 engine...")
    tp_db = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    # Load weights FIRST
    print(f"  Loading {len(layers_weights)} layers...")
    for layer_idx, lw in enumerate(layers_weights):
        tp_db.load_layer_weights(layer_idx, lw)
    tp_db.load_final_norm(final_norm)
    tp_db.load_lm_head(lm_head)
    # Now set dispatch modes
    tp_db.set_double_buffer_enabled(True)
    tp_db.set_cached_dispatch(True)
    tp_db.set_stream_overlap_dispatch(True)
    if not tp_db._engine_layer_caches:
        tp_db.build_dispatch_cache()
    print(f"  Double-buffer engine ready: {len(tp_db.engines[0].layers)} layers")
    
    print(f"\nRunning {num_steps} decode steps...")
    
    cos_sims = []
    max_diffs = []
    rng = np.random.default_rng(42)
    
    for step in range(num_steps):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        
        # Run standard path
        out_std = tp_std.decode_step(emb, step)
        
        # Run double-buffer path
        out_db = tp_db.decode_step(emb, step)
        
        # Compute cosine similarity
        out_std_f32 = out_std.astype(np.float32)
        out_db_f32 = out_db.astype(np.float32)
        
        dot = np.dot(out_std_f32, out_db_f32)
        norm_std = np.linalg.norm(out_std_f32)
        norm_db = np.linalg.norm(out_db_f32)
        cos_sim = dot / (norm_std * norm_db + 1e-8)
        
        max_diff = np.max(np.abs(out_std_f32 - out_db_f32))
        
        cos_sims.append(cos_sim)
        max_diffs.append(max_diff)
        
        if step % 5 == 0 or step == num_steps - 1:
            print(f"  Step {step+1}/{num_steps}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6e}")
        
        if cos_sim < 0.99:
            print(f"    WARNING: Low cosine similarity at step {step+1}")
    
    # Cleanup
    tp_std.cleanup()
    tp_db.cleanup()
    
    # Results
    min_cos_sim = min(cos_sims)
    avg_cos_sim = np.mean(cos_sims)
    max_max_diff = max(max_diffs)
    
    print("\n" + "=" * 70)
    print(f"Results:")
    print(f"  Min cosine similarity:  {min_cos_sim:.6f}")
    print(f"  Avg cosine similarity:  {avg_cos_sim:.6f}")
    print(f"  Max absolute difference: {max_max_diff:.6e}")
    print(f"  Threshold: >= 0.99")
    
    passed = min_cos_sim >= 0.99
    print(f"\nVAL-DB-002: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    
    return passed


def benchmark_throughput(num_warmup: int = 10, num_iters: int = 100):
    """VAL-DB-003: Benchmark throughput with stream overlap.
    
    Double-buffer combined with stream overlap should achieve
    >= 5% improvement (median latency <= 0.95x vs standard).
    """
    print("=" * 70)
    print("VAL-DB-003: Throughput Benchmark with Stream Overlap")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Load weights once
    print("\nLoading model weights...")
    loader = QwenWeightLoader(MODEL_DIR, config)
    layers_weights = []
    for layer_idx in range(config.num_hidden_layers):
        layers_weights.append(loader.load_layer(layer_idx))
    final_norm = loader.load_final_norm()
    lm_head = loader.load_lm_head()
    print(f"  Loaded {len(layers_weights)} layers")
    
    # Standard engine (cached + stream overlap, NO double-buffer)
    print("\nCreating standard engine (cached + stream_overlap)...")
    tp_std = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_std.set_cached_dispatch(True)
    tp_std.set_stream_overlap_dispatch(True)
    for layer_idx, lw in enumerate(layers_weights):
        tp_std.load_layer_weights(layer_idx, lw)
    tp_std.load_final_norm(final_norm)
    tp_std.load_lm_head(lm_head)
    tp_std.build_dispatch_cache()
    
    # Double-buffer engine (cached + stream overlap + double-buffer)
    print("Creating double-buffer engine (cached + stream_overlap + double_buffer)...")
    tp_db = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_db.set_double_buffer_enabled(True)
    tp_db.set_cached_dispatch(True)
    tp_db.set_stream_overlap_dispatch(True)
    for layer_idx, lw in enumerate(layers_weights):
        tp_db.load_layer_weights(layer_idx, lw)
    tp_db.load_final_norm(final_norm)
    tp_db.load_lm_head(lm_head)
    tp_db.build_dispatch_cache()
    
    # Warmup
    print(f"\nWarming up ({num_warmup} steps)...")
    rng = np.random.default_rng(42)
    for i in range(num_warmup):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_std.decode_step(emb, i)
        tp_db.decode_step(emb, i)
    
    # Benchmark standard
    print(f"Benchmarking standard path ({num_iters} iterations)...")
    times_std = []
    for i in range(num_iters):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        t0 = time.perf_counter()
        tp_std.decode_step(emb, num_warmup + i)
        tp_std._hip.synchronize()
        t1 = time.perf_counter()
        times_std.append((t1 - t0) * 1000)
    
    # Reset DB engine state
    for eng in tp_db.engines:
        eng.kv_cache.current_len = 0
    
    # Benchmark double-buffer
    print(f"Benchmarking double-buffer path ({num_iters} iterations)...")
    times_db = []
    for i in range(num_iters):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        t0 = time.perf_counter()
        tp_db.decode_step(emb, num_warmup + i)
        tp_db._hip.synchronize()
        t1 = time.perf_counter()
        times_db.append((t1 - t0) * 1000)
    
    # Cleanup
    tp_std.cleanup()
    tp_db.cleanup()
    
    # Statistics
    median_std = np.median(times_std)
    median_db = np.median(times_db)
    speedup = median_std / median_db if median_db > 0 else 1.0
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Standard path:       {median_std:.2f} ms/step")
    print(f"  Double-buffer path:  {median_db:.2f} ms/step")
    print(f"  Speedup:             {speedup:.3f}x")
    
    if speedup > 1.0:
        improvement = (speedup - 1.0) * 100
        print(f"  Improvement:         +{improvement:.1f}%")
    else:
        degradation = (1.0 - speedup) * 100
        print(f"  Degradation:         -{degradation:.1f}%")
    
    print(f"  Threshold: >= 1.05x (5% improvement)")
    
    passed = speedup >= 1.05
    print(f"\nVAL-DB-003: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    
    return passed


def test_long_run_stability(num_tokens: int = 1000):
    """VAL-DB-004: Long-run stability test.
    
    Run 1000+ sequential decode steps and verify:
    - No NaN/Inf in outputs
    - Coefficient of variation < 10%
    - Final output passes cosine check vs standard path
    """
    print("=" * 70)
    print(f"VAL-DB-004: Long-Run Stability Test ({num_tokens} tokens)")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Load weights once
    print("\nLoading model weights...")
    loader = QwenWeightLoader(MODEL_DIR, config)
    layers_weights = []
    for layer_idx in range(config.num_hidden_layers):
        layers_weights.append(loader.load_layer(layer_idx))
    final_norm = loader.load_final_norm()
    lm_head = loader.load_lm_head()
    print(f"  Loaded {len(layers_weights)} layers")
    
    # Create double-buffer engine
    print("\nCreating double-buffer TP=4 engine...")
    tp_db = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=num_tokens)
    tp_db.set_double_buffer_enabled(True)
    tp_db.set_cached_dispatch(True)
    tp_db.set_stream_overlap_dispatch(True)
    for layer_idx, lw in enumerate(layers_weights):
        tp_db.load_layer_weights(layer_idx, lw)
    tp_db.load_final_norm(final_norm)
    tp_db.load_lm_head(lm_head)
    tp_db.build_dispatch_cache()
    
    # Run long sequence
    print(f"\nRunning {num_tokens} sequential decode steps...")
    
    times_per_step = []
    has_nan_inf = False
    rng = np.random.default_rng(42)
    
    # Also run standard path for final comparison
    tp_std = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=num_tokens)
    tp_std.set_cached_dispatch(True)
    tp_std.set_stream_overlap_dispatch(True)
    for layer_idx, lw in enumerate(layers_weights):
        tp_std.load_layer_weights(layer_idx, lw)
    tp_std.load_final_norm(final_norm)
    tp_std.load_lm_head(lm_head)
    tp_std.build_dispatch_cache()
    
    last_out_std = None
    last_out_db = None
    
    start_time = time.perf_counter()
    
    for step in range(num_tokens):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        
        # Run both paths
        t0 = time.perf_counter()
        out_std = tp_std.decode_step(emb, step)
        out_db = tp_db.decode_step(emb, step)
        tp_std._hip.synchronize()
        tp_db._hip.synchronize()
        t1 = time.perf_counter()
        
        times_per_step.append((t1 - t0) * 1000)
        
        # Check for NaN/Inf
        if np.any(np.isnan(out_db)) or np.any(np.isinf(out_db)):
            print(f"  ERROR at step {step+1}: NaN/Inf detected in double-buffer output")
            has_nan_inf = True
            break
        
        if step % 100 == 0 or step == num_tokens - 1:
            print(f"  Step {step+1}/{num_tokens}: time={times_per_step[-1]:.2f}ms")
        
        last_out_std = out_std
        last_out_db = out_db
    
    total_time = time.perf_counter() - start_time
    
    # Cleanup
    tp_std.cleanup()
    tp_db.cleanup()
    
    if has_nan_inf:
        print("\n" + "=" * 70)
        print("VAL-DB-004: FAIL - NaN/Inf detected")
        print("=" * 70)
        return False
    
    # Check coefficient of variation
    median_time = np.median(times_per_step)
    std_time = np.std(times_per_step)
    cv = (std_time / median_time) * 100 if median_time > 0 else 0
    
    # Final cosine similarity
    dot = np.dot(last_out_std.astype(np.float32), last_out_db.astype(np.float32))
    norm_std = np.linalg.norm(last_out_std.astype(np.float32))
    norm_db = np.linalg.norm(last_out_db.astype(np.float32))
    final_cos_sim = dot / (norm_std * norm_db + 1e-8)
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Total time:            {total_time:.2f}s")
    print(f"  Throughput:            {num_tokens / total_time:.2f} tok/s")
    print(f"  Median latency:        {median_time:.2f} ms/step")
    print(f"  Std deviation:         {std_time:.2f} ms")
    print(f"  Coefficient of var:    {cv:.1f}%")
    print(f"  Final cosine sim:      {final_cos_sim:.6f}")
    print(f"  NaN/Inf detected:      {has_nan_inf}")
    print(f"  Thresholds: CV < 10%, cos_sim >= 0.99")
    
    passed = (cv < 10) and (final_cos_sim >= 0.99) and (not has_nan_inf)
    print(f"\nVAL-DB-004: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    
    return passed


def test_c_dispatch_interaction():
    """VAL-DB-005: Test interaction with C dispatch.
    
    When both _double_buffer_enabled=True and _c_dispatch_enabled=True,
    C dispatch takes precedence and double-buffer is ignored.
    Output should be correct (matches C dispatch path).
    """
    print("=" * 70)
    print("VAL-DB-005: C Dispatch Interaction Test")
    print("=" * 70)
    
    config = load_config_from_json(MODEL_DIR)
    
    # Load weights once
    print("\nLoading model weights...")
    loader = QwenWeightLoader(MODEL_DIR, config)
    layers_weights = []
    for layer_idx in range(config.num_hidden_layers):
        layers_weights.append(loader.load_layer(layer_idx))
    final_norm = loader.load_final_norm()
    lm_head = loader.load_lm_head()
    print(f"  Loaded {len(layers_weights)} layers")
    
    # Create C dispatch engine (no double-buffer)
    print("\nCreating C dispatch engine (no double-buffer)...")
    tp_c = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_c.set_c_dispatch_enabled(True)
    for layer_idx, lw in enumerate(layers_weights):
        tp_c.load_layer_weights(layer_idx, lw)
    tp_c.load_final_norm(final_norm)
    tp_c.load_lm_head(lm_head)
    tp_c.build_dispatch_cache()
    
    # Create C dispatch + double-buffer engine
    # Double-buffer should be ignored when C dispatch is enabled
    print("Creating C dispatch + double-buffer engine...")
    tp_c_db = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    tp_c_db.set_c_dispatch_enabled(True)
    tp_c_db.set_double_buffer_enabled(True)  # Should be ignored
    for layer_idx, lw in enumerate(layers_weights):
        tp_c_db.load_layer_weights(layer_idx, lw)
    tp_c_db.load_final_norm(final_norm)
    tp_c_db.load_lm_head(lm_head)
    tp_c_db.build_dispatch_cache()
    
    print(f"\nRunning 10 decode steps...")
    
    cos_sims = []
    rng = np.random.default_rng(42)
    
    for step in range(10):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        
        # Run C dispatch path
        out_c = tp_c.decode_step(emb, step)
        
        # Run C dispatch + double-buffer path
        out_c_db = tp_c_db.decode_step(emb, step)
        
        # Compute cosine similarity
        out_c_f32 = out_c.astype(np.float32)
        out_c_db_f32 = out_c_db.astype(np.float32)
        
        dot = np.dot(out_c_f32, out_c_db_f32)
        norm_c = np.linalg.norm(out_c_f32)
        norm_c_db = np.linalg.norm(out_c_db_f32)
        cos_sim = dot / (norm_c * norm_c_db + 1e-8)
        
        cos_sims.append(cos_sim)
        
        if step % 3 == 0 or step == 9:
            print(f"  Step {step+1}/10: cos_sim={cos_sim:.6f}")
    
    # Cleanup
    tp_c.cleanup()
    tp_c_db.cleanup()
    
    # Results
    min_cos_sim = min(cos_sims)
    avg_cos_sim = np.mean(cos_sims)
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Min cosine similarity: {min_cos_sim:.6f}")
    print(f"  Avg cosine similarity: {avg_cos_sim:.6f}")
    print(f"  Threshold: >= 0.99")
    
    # When C dispatch is enabled, both should produce identical results
    # (double-buffer is ignored)
    passed = min_cos_sim >= 0.99
    print(f"\nVAL-DB-005: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    
    return passed


def run_all_tests():
    """Run all VAL-DB validation tests."""
    print("\n" + "=" * 70)
    print("DOUBLE-BUFFER TP=4 COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    results = {}
    
    # VAL-DB-001: Buffer swap
    try:
        results['VAL-DB-001'] = test_buffer_swap_alternation()
    except Exception as e:
        print(f"VAL-DB-001 ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['VAL-DB-001'] = False
    
    # VAL-DB-002: Numerical correctness
    try:
        results['VAL-DB-002'] = test_numerical_correctness(num_steps=20)
    except Exception as e:
        print(f"VAL-DB-002 ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['VAL-DB-002'] = False
    
    # VAL-DB-003: Throughput
    try:
        results['VAL-DB-003'] = benchmark_throughput(num_warmup=5, num_iters=50)
    except Exception as e:
        print(f"VAL-DB-003 ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['VAL-DB-003'] = False
    
    # VAL-DB-004: Long-run stability (quick version for initial testing)
    try:
        results['VAL-DB-004'] = test_long_run_stability(num_tokens=100)
    except Exception as e:
        print(f"VAL-DB-004 ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['VAL-DB-004'] = False
    
    # VAL-DB-005: C dispatch interaction
    try:
        results['VAL-DB-005'] = test_c_dispatch_interaction()
    except Exception as e:
        print(f"VAL-DB-005 ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['VAL-DB-005'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double-buffer TP=4 validation")
    parser.add_argument("--all", action="store_true", help="Run all validation tests")
    parser.add_argument("--buffer-swap", action="store_true", help="Run VAL-DB-001 only")
    parser.add_argument("--correctness", action="store_true", help="Run VAL-DB-002 only")
    parser.add_argument("--benchmark", action="store_true", help="Run VAL-DB-003 only")
    parser.add_argument("--stability", action="store_true", help="Run VAL-DB-004 only")
    parser.add_argument("--c-dispatch", action="store_true", help="Run VAL-DB-005 only")
    parser.add_argument("--steps", type=int, default=20, help="Number of correctness test steps")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--tokens", type=int, default=1000, help="Number of stability test tokens")
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific test is selected
    if not any([args.buffer_swap, args.correctness, args.benchmark, 
                args.stability, args.c_dispatch, args.all]):
        args.all = True
    
    # Check GPU availability
    try:
        from src.runtime.hip_dispatch import HIPRuntime
        hip = HIPRuntime()
        hip.init()
        import ctypes
        device_count = ctypes.c_int()
        hip._lib.hipGetDeviceCount(ctypes.byref(device_count))
        if device_count.value < 4:
            print(f"ERROR: Need 4 GPUs for TP=4 testing, found {device_count.value}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize HIP or access GPUs: {e}")
        sys.exit(1)
    
    # Run requested tests
    passed = True
    
    if args.all:
        passed = run_all_tests()
    elif args.buffer_swap:
        passed = test_buffer_swap_alternation()
    elif args.correctness:
        passed = test_numerical_correctness(num_steps=args.steps)
    elif args.benchmark:
        passed = benchmark_throughput(num_iters=args.iters)
    elif args.stability:
        passed = test_long_run_stability(num_tokens=args.tokens)
    elif args.c_dispatch:
        passed = test_c_dispatch_interaction()
    
    sys.exit(0 if passed else 1)
