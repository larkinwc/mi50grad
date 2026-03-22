#!/usr/bin/env python3
"""
Benchmark for fused QKV GEMV kernel.

Measures throughput (tok/s) with fused QKV kernel enabled.
Compares against baseline (if available).
"""
import sys
import time
import numpy as np
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import QwenConfig
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

def benchmark_fused_qkv():
    """Benchmark throughput with fused QKV kernel."""
    print("=" * 70)
    print("Fused QKV GEMV Benchmark")
    print("=" * 70)
    
    MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = QwenConfig.from_pretrained(MODEL_DIR)
    
    # Create TP engine
    print("\nInitializing TP=4 engine...")
    tp = TPInferenceEngine(config, device_ids=[0,1,2,3], max_seq_len=512)
    
    # Load weights
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 16 == 0:
            print(f"    Loading layer {layer_idx}...")
        tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp.load_final_norm(loader.load_final_norm())
    tp.load_lm_head(loader.load_lm_head())
    
    # Build dispatch cache
    print("Building dispatch cache...")
    tp.build_dispatch_cache()
    
    # Enable all optimizations
    tp.set_direct_kv_write(True)
    tp.set_c_dispatch(True)
    tp.set_kernel_p2p_allreduce(True)
    tp.set_deferred_attention_ar(True)
    
    # Check fused kernel availability
    fused_available = False
    for e in tp.engines:
        if getattr(e, '_gemv_int4_qkv_fused', False):
            fused_available = True
            break
    
    print(f"\nFused QKV kernel: {'AVAILABLE' if fused_available else 'NOT AVAILABLE'}")
    
    # Check other optimizations
    for e in tp.engines:
        print(f"GPU{e.device.device_id}: "
              f"v8={getattr(e, '_gemv_int4_v8', False)}, "
              f"v7={getattr(e, '_gemv_int4_v7', False)}, "
              f"qkv_fused={getattr(e, '_gemv_int4_qkv_fused', False)}")
        break
    
    rng = np.random.default_rng(42)
    
    # Warmup
    print("\nWarmup (10 steps)...")
    for i in range(10):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        for e in tp.engines:
            e.kv_cache.current_len = 0
        tp.decode_step(emb, i)
        tp._hip.synchronize()
    
    # Benchmark
    num_steps = 50
    print(f"\nBenchmarking {num_steps} steps...")
    
    for e in tp.engines:
        e.kv_cache.current_len = 0
    
    t0 = time.perf_counter()
    for step in range(num_steps):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp.decode_step(emb, step)
        tp._hip.synchronize()
    elapsed = time.perf_counter() - t0
    
    tps = num_steps / elapsed
    ms_per_tok = elapsed / num_steps * 1000
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {tps:.2f} tok/s")
    print(f"  Latency: {ms_per_tok:.2f} ms/tok")
    
    # Cleanup
    tp.cleanup()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return tps

if __name__ == "__main__":
    tps = benchmark_fused_qkv()
    print(f"\nFinal throughput: {tps:.2f} tok/s")
    sys.exit(0)
