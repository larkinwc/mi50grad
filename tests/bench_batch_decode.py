#!/usr/bin/env python3
"""
Batch decode throughput benchmark.

Measures throughput for batch=1, batch=2, and batch=4.
Expected results:
- batch=1: >= 53.0 tok/s (no regression)
- batch=2: > batch=1 by at least 5%
- batch=4: > batch=2

Validation contract assertions:
- VAL-BD-009: Batch=2 throughput improvement (>5% over batch=1)
- VAL-BD-010: Batch=4 throughput improvement (>batch=2)
- VAL-BD-011: No batch=1 regression (>= 53.0 tok/s)
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.tp_engine import TPInferenceEngine
from src.model.qwen import QwenConfig


def benchmark_batch_throughput():
    """Benchmark batch decode throughput for batch=1,2,4."""
    print("=" * 80)
    print("BATCH DECODE THROUGHPUT BENCHMARK")
    print("=" * 80)
    
    config = QwenConfig()
    hidden_size = config.hidden_size
    
    print("\nInitializing TP engine (tp_size=4)...")
    try:
        engine = TPInferenceEngine(
            config=config,
            device_ids=list(range(4)),
            max_seq_len=512,
        )
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        print("Benchmark requires 4 GPUs")
        return None
    
    print("Engine initialized")
    
    # Enable cached dispatch for optimal throughput (required for batch decode)
    from src.model.weight_loader import QwenWeightLoader
    MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    loader = QwenWeightLoader(MODEL_DIR, config)
    print("Loading weights...")
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    
    print("Building dispatch cache...")
    engine.build_dispatch_cache()
    engine.set_c_dispatch(True)
    engine.set_direct_kv_write(True)
    engine.set_kernel_p2p_allreduce(True)
    engine.set_deferred_attention_ar(True)
    print("Cached dispatch enabled")
    
    # Generate test embeddings
    np.random.seed(42)
    embeddings = {
        1: [np.random.randn(hidden_size).astype(np.float16) * 0.1],
        2: [np.random.randn(hidden_size).astype(np.float16) * 0.1 for _ in range(2)],
        4: [np.random.randn(hidden_size).astype(np.float16) * 0.1 for _ in range(4)],
    }
    
    results = {}
    n_steps = 100
    
    for batch_size in [1, 2, 4]:
        print(f"\n--- Benchmarking batch={batch_size} ---")
        
        # Warmup
        engine.engines[0].kv_cache.current_len = 0
        for i in range(5):
            positions = list(range(batch_size))
            engine.decode_step_batch(embeddings[batch_size][:batch_size], positions)
        engine.engines[0].device.synchronize()
        
        # Benchmark
        engine.engines[0].kv_cache.current_len = 0
        t0 = time.perf_counter()
        for i in range(n_steps):
            positions = [(i * batch_size + j) % 256 for j in range(batch_size)]
            engine.decode_step_batch(embeddings[batch_size][:batch_size], positions)
        engine.engines[0].device.synchronize()
        elapsed = time.perf_counter() - t0
        
        # Calculate throughput
        total_tokens = batch_size * n_steps
        tok_s = total_tokens / elapsed
        ms_per_tok = (elapsed / n_steps) * 1000  # ms per batch
        
        results[batch_size] = {
            'tok_s': tok_s,
            'ms_per_tok': ms_per_tok,
            'tok_s_per_seq': tok_s / batch_size,
        }
        
        print(f"  Total throughput: {tok_s:.1f} tok/s")
        print(f"  Per-sequence:     {tok_s/batch_size:.1f} tok/s")
        print(f"  Ms per batch:     {ms_per_tok:.2f} ms")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Batch':<8} {'Total tok/s':<15} {'Per-seq tok/s':<15} {'ms/batch':<10}")
    print("-" * 80)
    for batch_size in [1, 2, 4]:
        r = results[batch_size]
        print(f"{batch_size:<8} {r['tok_s']:<15.1f} {r['tok_s_per_seq']:<15.1f} {r['ms_per_tok']:<10.2f}")
    
    # Analyze scaling
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    
    scaling_2_vs_1 = results[2]['tok_s'] / results[1]['tok_s']
    scaling_4_vs_2 = results[4]['tok_s'] / results[2]['tok_s']
    scaling_4_vs_1 = results[4]['tok_s'] / results[1]['tok_s']
    
    print(f"  Batch=2 / Batch=1: {scaling_2_vs_1:.2f}x (target: >1.05x)")
    print(f"  Batch=4 / Batch=2: {scaling_4_vs_2:.2f}x (target: >1.0x)")
    print(f"  Batch=4 / Batch=1: {scaling_4_vs_1:.2f}x")
    
    # Check targets
    print("\n" + "=" * 80)
    print("TARGET VERIFICATION")
    print("=" * 80)
    
    batch1_ok = results[1]['tok_s'] >= 53.0
    batch2_ok = scaling_2_vs_1 > 1.05
    batch4_ok = scaling_4_vs_2 > 1.0
    
    print(f"  Batch=1 >= 53.0 tok/s:           {'✓ PASS' if batch1_ok else '✗ FAIL'} ({results[1]['tok_s']:.1f})")
    print(f"  Batch=2 > Batch=1 by >5%:        {'✓ PASS' if batch2_ok else '✗ FAIL'} ({scaling_2_vs_1:.2f}x)")
    print(f"  Batch=4 > Batch=2:               {'✓ PASS' if batch4_ok else '✗ FAIL'} ({scaling_4_vs_2:.2f}x)")
    
    del engine
    
    all_ok = batch1_ok and batch2_ok and batch4_ok
    print("\n" + "=" * 80)
    if all_ok:
        print("ALL TARGETS MET")
    else:
        print("SOME TARGETS NOT MET")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = benchmark_batch_throughput()
    sys.exit(0 if results else 1)
