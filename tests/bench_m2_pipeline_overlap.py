#!/usr/bin/env python3
"""
M2 Pipeline Overlap Benchmark — VAL-AR-003

Benchmark to measure the throughput benefit of pipelining allreduce with next-layer compute.

This test compares:
1. Standard stream overlap (wait for allreduce before next-layer compute)
2. Aggressive pipeline overlap (double-buffer: start next-layer compute before allreduce completes)

The aggressive overlap is implemented via _decode_step_cached_stream() with double-buffer enabled.
This allows Layer N+1's RMSNorm and attention GEMV to run concurrently with Layer N's FFN allreduce.

VAL-AR-003: Pipelining allreduce with next-layer compute produces measurable throughput 
improvement (>= 2% over non-overlapped path) while maintaining correctness (cosine sim >= 0.99).

Usage on dev server:
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
        mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_m2_pipeline_overlap.py'
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
MAX_SEQ_LEN = 64

WARMUP_STEPS = 3
BENCH_STEPS = 50
CORRECTNESS_STEPS = 10
COSINE_THRESHOLD = 0.99


def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    dot = float(np.dot(a32, b32))
    den = float(np.linalg.norm(a32) * np.linalg.norm(b32)) + 1e-12
    return dot / den


def create_tp_engine(config, loader, enable_double_buffer: bool):
    """Create TP=4 engine with specified overlap mode."""
    tp_engine = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    # Enable flags BEFORE loading weights (required for proper initialization)
    tp_engine.set_double_buffer_enabled(enable_double_buffer)
    tp_engine.set_cached_dispatch(True)
    tp_engine.set_stream_overlap_dispatch(True)
    
    # Load weights
    for layer_idx in range(config.num_hidden_layers):
        tp_engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp_engine.load_final_norm(loader.load_final_norm())
    tp_engine.load_lm_head(loader.load_lm_head())
    
    # Build dispatch cache (after weights loaded)
    tp_engine.build_dispatch_cache()
    
    return tp_engine


def benchmark_mode(tp_engine, embs, num_warmup: int, num_iters: int, mode_name: str, position_offset: int = 0):
    """Run benchmark for a specific overlap mode."""
    print(f"\nBenchmarking {mode_name} ({num_iters} iterations)...")
    
    # Reset KV cache
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0
        eng.deltanet_state.reset()
    
    # Warmup
    for i in range(num_warmup):
        tp_engine.decode_step(embs[i % len(embs)], i)
    tp_engine.synchronize()
    
    # Benchmark
    times = []
    for i in range(num_iters):
        t0 = time.perf_counter()
        tp_engine.decode_step(embs[i % len(embs)], position_offset + i)
        tp_engine.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    median_ms = np.median(times)
    tok_s = 1.0 / (median_ms / 1000)
    
    print(f"  {mode_name}: {tok_s:.1f} tok/s  ({median_ms:.2f} ms/tok)")
    
    return tok_s, median_ms


def test_correctness(tp_std, tp_db, embs, num_steps: int = 10):
    """Verify that double-buffer path produces correct results."""
    print(f"\nCorrectness check ({num_steps} steps)...")
    
    cos_sims = []
    all_pass = True
    for step in range(num_steps):
        # Reset both engines to same KV cache state
        for eng in tp_std.engines:
            eng.kv_cache.current_len = 0
            eng.deltanet_state.reset()
        for eng in tp_db.engines:
            eng.kv_cache.current_len = 0
            eng.deltanet_state.reset()
        
        try:
            # Run both paths on SAME step with SAME embedding
            out_std = tp_std.decode_step(embs[step], 0)  # position 0 for both
            out_db = tp_db.decode_step(embs[step], 0)
            tp_std.synchronize()
            tp_db.synchronize()
            
            # Compute cosine similarity
            cos = cosine_sim(out_std, out_db)
            cos_sims.append(cos)
            
            max_diff = np.max(np.abs(out_std.astype(np.float32) - out_db.astype(np.float32)))
            
            status = "✓" if cos >= COSINE_THRESHOLD else "✗"
            print(f"  Step {step+1}/{num_steps}: cos_sim={cos:.6f}, max_diff={max_diff:.6e} {status}")
            
            if cos < COSINE_THRESHOLD:
                print(f"    WARNING: Low cosine similarity at step {step+1}")
                all_pass = False
        except Exception as e:
            print(f"  Step {step+1}/{num_steps}: ERROR - {e}")
            all_pass = False
            break
    
    if cos_sims:
        min_cos = min(cos_sims)
        avg_cos = np.mean(cos_sims)
        
        print(f"\n  Min cosine similarity: {min_cos:.6f}")
        print(f"  Avg cosine similarity: {avg_cos:.6f}")
        print(f"  Threshold: >= {COSINE_THRESHOLD}")
        
        return all_pass and min_cos >= COSINE_THRESHOLD
    else:
        print("\n  No successful steps completed")
        return False


def main():
    print("=" * 70)
    print("M2 Pipeline Overlap Benchmark (VAL-AR-003)")
    print("=" * 70)
    print("\nComparing:")
    print("  1. Standard overlap: cached + stream_overlap (NO double-buffer)")
    print("  2. Aggressive overlap: cached + stream_overlap + double-buffer")
    print("\nAggressive overlap allows Layer N+1 attention to run concurrently")
    print("with Layer N's FFN allreduce, hiding allreduce latency.")
    
    # Check GPU count
    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    import ctypes
    device_count = ctypes.c_int()
    hip._lib.hipGetDeviceCount(ctypes.byref(device_count))
    
    if device_count.value < 4:
        print(f"\nERROR: Need 4 GPUs for TP=4, found {device_count.value}")
        sys.exit(1)
    
    print(f"\nGPUs: {device_count.value} (TP=4 enabled)")
    
    # Load config and weights
    print("\nLoading model...")
    config = load_config_from_json(MODEL_DIR)
    loader = QwenWeightLoader(MODEL_DIR, config)
    
    # Create engines
    print("\nCreating TP=4 engines...")
    
    # Standard overlap (NO double-buffer)
    print("  Creating standard overlap engine...")
    tp_std = create_tp_engine(config, loader, enable_double_buffer=False)
    
    # Aggressive overlap (WITH double-buffer)
    print("  Creating aggressive overlap engine (double-buffer)...")
    tp_db = create_tp_engine(config, loader, enable_double_buffer=True)
    
    # Generate random embeddings for benchmark
    print("\nGenerating test embeddings...")
    np.random.seed(42)
    embs = [np.random.randn(config.hidden_size).astype(np.float16) * 0.02 
            for _ in range(WARMUP_STEPS + BENCH_STEPS)]
    
    # Run benchmarks
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    
    # Generate separate embeddings for benchmarking to avoid KV cache conflicts
    np.random.seed(123)
    bench_embs = [np.random.randn(config.hidden_size).astype(np.float16) * 0.02 
                  for _ in range(WARMUP_STEPS + BENCH_STEPS)]
    
    tok_std, ms_std = benchmark_mode(tp_std, bench_embs, WARMUP_STEPS, BENCH_STEPS, 
                                      "Standard overlap", position_offset=0)
    
    # Reset and benchmark double-buffer
    tok_db, ms_db = benchmark_mode(tp_db, bench_embs, WARMUP_STEPS, BENCH_STEPS, 
                                    "Aggressive overlap (double-buffer)", position_offset=0)
    
    # Compute speedup
    speedup = tok_db / tok_std if tok_std > 0 else 1.0
    improvement = (speedup - 1.0) * 100
    
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"  Standard overlap:      {tok_std:.1f} tok/s  ({ms_std:.2f} ms/tok)")
    print(f"  Aggressive overlap:    {tok_db:.1f} tok/s  ({ms_db:.2f} ms/tok)")
    print(f"  Speedup:               {speedup:.3f}x")
    print(f"  Improvement:           {improvement:+.1f}%")
    print(f"  Target:                >= +2.0%")
    
    # Correctness check (use first CORRECTNESS_STEPS from embs)
    correctness_passed = test_correctness(tp_std, tp_db, embs[:CORRECTNESS_STEPS], num_steps=CORRECTNESS_STEPS)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("VAL-AR-003 Verdict")
    print("=" * 70)
    
    overlap_benefit = improvement >= 2.0
    print(f"  Throughput improvement >= 2%:  {'✓ PASS' if overlap_benefit else '✗ FAIL'} ({improvement:+.1f}%)")
    print(f"  Correctness (cos sim >= 0.99): {'✓ PASS' if correctness_passed else '✗ FAIL'}")
    
    overall_pass = overlap_benefit and correctness_passed
    
    if overall_pass:
        print("\n✓ VAL-AR-003: PASS")
        print("\nConclusion: Aggressive pipeline overlap (double-buffer) provides")
        print(f"measurable throughput improvement ({improvement:+.1f}%) while maintaining")
        print("numerical correctness. The overlap is WORTH the complexity.")
    else:
        print("\n✗ VAL-AR-003: FAIL")
        if not overlap_benefit:
            print(f"\nNote: Overlap benefit ({improvement:+.1f}%) below 2% target.")
            print("The complexity may not be justified for this workload.")
        if not correctness_passed:
            print("\nNote: Correctness check failed - investigate numerical issues.")
    
    # Cleanup
    print("\nCleaning up...")
    tp_std.cleanup()
    tp_db.cleanup()
    
    print("=" * 70)
    
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
