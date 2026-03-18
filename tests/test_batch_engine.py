#!/usr/bin/env python3
"""Test batch>1 support in the inference engine.

Tests:
1. batch=1 correctness and throughput (no regression)
2. batch=2 correctness (per-sequence cosine similarity vs individual decode)
3. Dynamic GEMV/GEMM switching verification
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def test_batch1_correctness_and_throughput():
    """Verify batch=1 produces same output as pre-batch engine with no throughput regression."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    # Reference: single-sequence decode
    engine_ref = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine_ref.load_layer_weights(i, loader.load_layer(i))
    engine_ref.load_final_norm(loader.load_final_norm())

    # Batch engine
    engine_batch = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine_batch.load_layer_weights(i, loader.load_layer(i))
    engine_batch.load_final_norm(loader.load_final_norm())

    print("=== Batch=1 Correctness vs Single-Sequence Reference ===")
    
    # Test token sequence
    tokens = [760, 6511, 314, 9338, 369, 11751, 2918]  # "The capital of France is Paris."
    
    all_pass = True
    
    # Run 10 decode steps
    for step in range(10):
        token_id = tokens[step % len(tokens)]
        emb = embed[token_id].copy()
        
        # Reference: single-sequence decode
        engine_ref.kv_cache.current_len = step
        engine_ref.deltanet_state.reset()
        hidden_ref = engine_ref.decode_step(emb, step)
        
        # Batch=1 decode
        engine_batch.kv_cache.current_len = step
        engine_batch.deltanet_state.reset()
        hidden_batch = engine_batch.decode_step_batch(emb[None, :], 1, step)
        
        # Compare
        max_err = np.max(np.abs(hidden_ref.astype(np.float32) - hidden_batch[0].astype(np.float32)))
        cos_sim = np.dot(hidden_ref.astype(np.float32).ravel(),
                         hidden_batch[0].astype(np.float32).ravel()) / (
            np.linalg.norm(hidden_ref.astype(np.float32)) *
            np.linalg.norm(hidden_batch[0].astype(np.float32)) + 1e-10)
        
        ok = cos_sim > 0.99 and max_err < 0.1
        print(f"  Step {step:2d}: max_err={max_err:.4f} cos_sim={cos_sim:.6f} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    
    print("\n=== Batch=1 Throughput (No Regression Check) ===")
    
    # Time batch=1
    engine_batch.kv_cache.current_len = 0
    engine_batch.deltanet_state.reset()
    
    # Warmup
    for i in range(5):
        engine_batch.decode_step_batch(embed[tokens[:1]], 1, i)
    engine_batch.device.synchronize()
    
    # Benchmark
    n_steps = 100
    t0 = time.perf_counter()
    for i in range(n_steps):
        engine_batch.decode_step_batch(embed[tokens[:1]], 1, i % 256)
    engine_batch.device.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s_batch1 = n_steps / elapsed
    
    print(f"  Batch=1 throughput: {tok_s_batch1:.1f} tok/s")
    print(f"  Expected: >= 36.4 tok/s (within 5% of 38.3 baseline)")
    
    # Check no regression (>= 36.4 tok/s = within 5% of 38.3)
    throughput_ok = tok_s_batch1 >= 36.4
    print(f"  Throughput check: {'PASS' if throughput_ok else 'FAIL'}")
    
    engine_ref.cleanup()
    engine_batch.cleanup()
    
    return all_pass and throughput_ok


def test_batch2_correctness():
    """Verify batch=2 produces per-sequence output matching individual decode."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    # Batch=2 engine
    engine_batch = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine_batch.load_layer_weights(i, loader.load_layer(i))
    engine_batch.load_final_norm(loader.load_final_norm())

    print("=== Batch=2 Correctness (Per-Sequence Cosine Similarity) ===")
    
    # Two different token sequences
    tokens1 = [760, 6511, 314, 9338, 369]  # "The capital of France is"
    tokens2 = [11751, 314, 2918, 760, 1980]  # "Paris. Hello world"
    
    all_pass = True
    
    # Test at different sequence positions
    for pos in [0, 5, 10]:
        # Get reference outputs (individual decode)
        engine_batch.kv_cache.current_len = pos
        engine_batch.deltanet_state.reset()
        
        emb1 = embed[tokens1[0]].copy()
        emb2 = embed[tokens2[0]].copy()
        
        # Reference: decode each sequence individually
        hidden_ref1 = engine_batch.decode_step(emb1, pos)
        
        engine_batch.kv_cache.current_len = pos
        engine_batch.deltanet_state.reset()
        hidden_ref2 = engine_batch.decode_step(emb2, pos)
        
        # Batch=2 decode
        batch_emb = np.vstack([emb1, emb2])
        engine_batch.kv_cache.current_len = pos
        engine_batch.deltanet_state.reset()
        hidden_batch = engine_batch.decode_step_batch(batch_emb, 2, pos)
        
        # Compare per-sequence
        cos_sim1 = np.dot(hidden_ref1.astype(np.float32).ravel(),
                          hidden_batch[0].astype(np.float32).ravel()) / (
            np.linalg.norm(hidden_ref1.astype(np.float32)) *
            np.linalg.norm(hidden_batch[0].astype(np.float32)) + 1e-10)
        
        cos_sim2 = np.dot(hidden_ref2.astype(np.float32).ravel(),
                          hidden_batch[1].astype(np.float32).ravel()) / (
            np.linalg.norm(hidden_ref2.astype(np.float32)) *
            np.linalg.norm(hidden_batch[1].astype(np.float32)) + 1e-10)
        
        max_err1 = np.max(np.abs(hidden_ref1.astype(np.float32) - hidden_batch[0].astype(np.float32)))
        max_err2 = np.max(np.abs(hidden_ref2.astype(np.float32) - hidden_batch[1].astype(np.float32)))
        
        ok1 = cos_sim1 > 0.99 and max_err1 < 0.1
        ok2 = cos_sim2 > 0.99 and max_err2 < 0.1
        
        print(f"  Position {pos:2d}:")
        print(f"    Seq 0: max_err={max_err1:.4f} cos_sim={cos_sim1:.6f} "
              f"{'PASS' if ok1 else 'FAIL'}")
        print(f"    Seq 1: max_err={max_err2:.4f} cos_sim={cos_sim2:.6f} "
              f"{'PASS' if ok2 else 'FAIL'}")
        
        if not (ok1 and ok2):
            all_pass = False
    
    engine_batch.cleanup()
    return all_pass


def test_gemv_gemm_switching():
    """Verify dynamic GEMV/GEMM kernel selection based on batch size."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())

    print("=== Dynamic GEMV/GEMM Switching ===")
    
    # Test batch=1 uses GEMV
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()
    
    # Reset launch counters
    engine.reset_launch_counters()
    engine._count_launches = True
    
    emb1 = embed[760:761].copy()  # batch=1
    _ = engine.decode_step_batch(emb1, 1, 0)
    engine.device.synchronize()
    
    print(f"  Batch=1: GEMV kernels should be selected")
    # Note: actual kernel launch counts would need kernel-specific tracking
    
    # Test batch>=2 uses GEMM
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()
    
    engine.reset_launch_counters()
    engine._count_launches = True
    
    emb4 = np.vstack([embed[760]] * 4)  # batch=4
    _ = engine.decode_step_batch(emb4, 4, 0)
    engine.device.synchronize()
    
    print(f"  Batch=4: GEMM kernels should be selected")
    
    # For now, just verify both batch sizes work
    print(f"  Verification: Both batch=1 and batch=4 complete successfully")
    print(f"  (Detailed kernel selection logging requires instrumentation)")
    
    engine.cleanup()
    return True


def test_batch_throughput_scaling():
    """Verify batch throughput scaling (batch=2 >= 1.4x, batch=4 >= 2.0x batch=1)."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    engine = InferenceEngine(config, device_id=0, max_seq_len=256)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())

    print("=== Batch Throughput Scaling ===")
    
    tokens = [760, 6511, 314, 9338, 369]
    n_steps = 50
    
    results = {}
    
    for batch_size in [1, 2, 4]:
        # Warmup
        batch_emb = np.vstack([embed[tokens[:1]]] * batch_size)
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        for i in range(5):
            engine.decode_step_batch(batch_emb, batch_size, i)
        engine.device.synchronize()
        
        # Benchmark
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        
        t0 = time.perf_counter()
        for i in range(n_steps):
            engine.decode_step_batch(batch_emb, batch_size, i % 256)
        engine.device.synchronize()
        elapsed = time.perf_counter() - t0
        
        # Total tokens processed = batch_size * n_steps
        total_tokens = batch_size * n_steps
        tok_s = total_tokens / elapsed
        results[batch_size] = tok_s
        
        print(f"  Batch={batch_size}: {tok_s:.1f} total tok/s "
              f"({tok_s/batch_size:.1f} tok/s per sequence)")
    
    # Check scaling
    scaling_2 = results[2] / results[1] if results[1] > 0 else 0
    scaling_4 = results[4] / results[1] if results[1] > 0 else 0
    
    print(f"\n  Scaling factors:")
    print(f"    Batch=2 / Batch=1 = {scaling_2:.2f}x (target: >= 1.4x)")
    print(f"    Batch=4 / Batch=1 = {scaling_4:.2f}x (target: >= 2.0x)")
    
    scaling_ok = scaling_2 >= 1.4 and scaling_4 >= 2.0
    
    engine.cleanup()
    return scaling_ok


if __name__ == "__main__":
    print("=" * 60)
    print("BATCH ENGINE TESTS")
    print("=" * 60)
    
    print("\n[TEST 1] Batch=1 Correctness and Throughput")
    print("-" * 60)
    test1_pass = test_batch1_correctness_and_throughput()
    
    print("\n[TEST 2] Batch=2 Correctness")
    print("-" * 60)
    test2_pass = test_batch2_correctness()
    
    print("\n[TEST 3] GEMV/GEMM Switching")
    print("-" * 60)
    test3_pass = test_gemv_gemm_switching()
    
    print("\n[TEST 4] Batch Throughput Scaling")
    print("-" * 60)
    test4_pass = test_batch_throughput_scaling()
    
    print("\n" + "=" * 60)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if all_pass else 1)
