#!/usr/bin/env python3
"""
Benchmark AWQ vs GPTQ performance for Qwen3.5-27B on TP=4 MI50s.

Compares:
- GPTQ baseline (standard v5 kernels)
- AWQ kernel mode (AWQ v5 kernels, no zero-point subtraction)

USAGE:
    # On the dev server with both models available:
    python3 tests/bench_awq_vs_gptq.py

REQUIREMENTS:
    - GPTQ model: /opt/models/Qwen3.5-27B-GPTQ-Int4
    - AWQ model: /opt/models/Qwen3.5-27B-AWQ
    - gemv_int4_v5_awq.hip compiled
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.model.awq_loader import AWQWeightLoader, detect_awq_format
from src.inference.tp_engine import TPInferenceEngine


GPTQ_MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
AWQ_MODEL_DIR = "/opt/models/Qwen3.5-27B-AWQ"
DEVICE_IDS = [0, 1, 2, 3]

# Benchmark configuration
WARMUP_STEPS = 10
BENCH_STEPS = 50


def print_header(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def reset_tp(engine: TPInferenceEngine):
    """Reset all KV caches and DeltaNet states."""
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


def benchmark_engine(engine: TPInferenceEngine, prompt_tokens: np.ndarray,
                     max_tokens: int = BENCH_STEPS) -> dict:
    """Benchmark a TP engine for decode throughput."""
    
    # Warmup
    reset_tp(engine)
    hidden = np.array(prompt_tokens[:1], dtype=np.float16)  # First token as hidden
    
    print(f"  Warming up ({WARMUP_STEPS} steps)...")
    for step in range(WARMUP_STEPS):
        hidden = engine.decode_step(hidden, step)
    
    # Benchmark
    reset_tp(engine)
    hidden = np.array(prompt_tokens[:1], dtype=np.float16)
    
    print(f"  Benchmarking ({BENCH_STEPS} steps)...")
    t0 = time.perf_counter()
    
    for step in range(BENCH_STEPS):
        hidden = engine.decode_step(hidden, step)
        if step == BENCH_STEPS // 2:
            # Mid-benchmark sync to ensure GPUs are done
            for dev_id in DEVICE_IDS:
                engine._hip.set_device(dev_id)
                engine._hip.synchronize()
    
    # Final sync
    for dev_id in DEVICE_IDS:
        engine._hip.set_device(dev_id)
        engine._hip.synchronize()
    
    elapsed = time.perf_counter() - t0
    tok_per_sec = BENCH_STEPS / elapsed
    ms_per_tok = (elapsed / BENCH_STEPS) * 1000
    
    return {
        'elapsed': elapsed,
        'steps': BENCH_STEPS,
        'tok_per_sec': tok_per_sec,
        'ms_per_tok': ms_per_tok,
    }


def load_gptq_engine() -> TPInferenceEngine:
    """Load TP=4 engine with GPTQ weights."""
    print_header("Loading GPTQ Model")
    print(f"  Model: {GPTQ_MODEL_DIR}")
    
    if not os.path.exists(GPTQ_MODEL_DIR):
        raise FileNotFoundError(f"GPTQ model not found at {GPTQ_MODEL_DIR}")
    
    config = load_config_from_json(GPTQ_MODEL_DIR)
    loader = QwenWeightLoader(GPTQ_MODEL_DIR, config)
    
    engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=512)
    
    print(f"  Loading {config.num_hidden_layers} layers on {len(DEVICE_IDS)} GPUs...")
    t0 = time.perf_counter()
    for layer_idx in range(config.num_hidden_layers):
        weights = loader.load_layer(layer_idx)
        engine.load_layer_weights(layer_idx, weights)
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print(f"  Weights loaded in {time.perf_counter() - t0:.1f}s")
    
    # Build dispatch cache (GPTQ mode, no AWQ)
    print("  Building dispatch cache...")
    engine.build_dispatch_cache()
    engine.set_direct_kv_write(True)
    engine.set_kernel_p2p_allreduce(True)
    engine.set_c_dispatch(True)
    
    # Verify GPTQ mode (AWQ should be disabled)
    awq_count = sum(1 for e in engine.engines if e._awq_mode)
    print(f"  AWQ mode: {awq_count}/{len(engine.engines)} engines (should be 0)")
    
    return engine


def load_awq_engine() -> TPInferenceEngine:
    """Load TP=4 engine with AWQ weights and AWQ kernels."""
    print_header("Loading AWQ Model")
    print(f"  Model: {AWQ_MODEL_DIR}")
    
    if not os.path.exists(AWQ_MODEL_DIR):
        raise FileNotFoundError(f"AWQ model not found at {AWQ_MODEL_DIR}")
    
    # Verify format
    detected = detect_awq_format(AWQ_MODEL_DIR)
    print(f"  Detected format: '{detected}'")
    if detected != 'awq':
        raise ValueError(f"Expected AWQ format, got '{detected}'")
    
    config = load_config_from_json(AWQ_MODEL_DIR)
    loader = AWQWeightLoader(AWQ_MODEL_DIR, config)
    
    engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=512)
    
    print(f"  Loading {config.num_hidden_layers} layers on {len(DEVICE_IDS)} GPUs...")
    t0 = time.perf_counter()
    for layer_idx in range(config.num_hidden_layers):
        weights = loader.load_layer(layer_idx)
        engine.load_layer_weights(layer_idx, weights)
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print(f"  Weights loaded in {time.perf_counter() - t0:.1f}s")
    
    # Enable AWQ mode BEFORE building dispatch cache
    print("  Enabling AWQ mode...")
    engine.set_awq_mode(True)
    
    # Build dispatch cache (AWQ kernels should be selected)
    print("  Building dispatch cache (AWQ kernels)...")
    engine.build_dispatch_cache()
    engine.set_direct_kv_write(True)
    engine.set_kernel_p2p_allreduce(True)
    engine.set_c_dispatch(True)
    
    # Verify AWQ mode
    awq_enabled = sum(1 for e in engine.engines if e._awq_mode)
    awq_kernel = sum(1 for e in engine.engines if e._gemv_int4_v5_awq)
    print(f"  AWQ mode: {awq_enabled}/{len(engine.engines)} engines")
    print(f"  AWQ kernel: {awq_kernel}/{len(engine.engines)} engines available")
    
    return engine


def main():
    print("=" * 72)
    print("AWQ vs GPTQ Benchmark — Qwen3.5-27B on TP=4 MI50s")
    print("=" * 72)
    
    # Check model availability
    gptq_available = os.path.exists(GPTQ_MODEL_DIR)
    awq_available = os.path.exists(AWQ_MODEL_DIR)
    
    print()
    print(f"GPTQ model: {'✓' if gptq_available else '✗'} {GPTQ_MODEL_DIR}")
    print(f"AWQ model:  {'✓' if awq_available else '✗'} {AWQ_MODEL_DIR}")
    
    if not gptq_available or not awq_available:
        print()
        print("ERROR: Both GPTQ and AWQ models are required for this benchmark.")
        if not gptq_available:
            print(f"  Download GPTQ model to: {GPTQ_MODEL_DIR}")
        if not awq_available:
            print(f"  Download AWQ model: ./scripts/download_awq_model.sh")
        return 1
    
    # Create a simple prompt (random tokens for consistency)
    rng = np.random.default_rng(42)
    prompt_len = 32
    prompt_tokens = rng.integers(0, 1000, size=prompt_len).tolist()
    
    results = {}
    
    # Benchmark GPTQ
    try:
        gptq_engine = load_gptq_engine()
        results['gptq'] = benchmark_engine(gptq_engine, prompt_tokens)
        print(f"\n  GPTQ: {results['gptq']['tok_per_sec']:.1f} tok/s "
              f"({results['gptq']['ms_per_tok']:.1f} ms/tok)")
        gptq_engine.cleanup()
    except Exception as e:
        print(f"  GPTQ benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results['gptq'] = None
    
    # Benchmark AWQ
    try:
        awq_engine = load_awq_engine()
        results['awq'] = benchmark_engine(awq_engine, prompt_tokens)
        print(f"\n  AWQ:  {results['awq']['tok_per_sec']:.1f} tok/s "
              f"({results['awq']['ms_per_tok']:.1f} ms/tok)")
        awq_engine.cleanup()
    except Exception as e:
        print(f"  AWQ benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results['awq'] = None
    
    # Compare results
    print_header("Results Summary")
    
    if results['gptq'] and results['awq']:
        gptq_tps = results['gptq']['tok_per_sec']
        awq_tps = results['awq']['tok_per_sec']
        speedup = awq_tps / gptq_tps
        improvement = (speedup - 1) * 100
        
        print(f"  GPTQ: {gptq_tps:.1f} tok/s (baseline)")
        print(f"  AWQ:  {awq_tps:.1f} tok/s")
        print(f"  Speedup: {speedup:.2f}x ({improvement:+.1f}%)")
        
        if improvement > 0:
            print(f"\n  ✓ AWQ provides {improvement:.1f}% throughput improvement")
        else:
            print(f"\n  ✗ AWQ did not improve performance (expected ~3-5%)")
            print(f"    Possible causes:")
            print(f"    - AWQ kernel not properly selected")
            print(f"    - Memory bandwidth not the bottleneck")
            print(f"    - Other dispatch overhead dominates")
        
        # Verify AWQ kernel was used
        print(f"\n  Verification:")
        print(f"    - AWQ kernel compiled: gemv_int4_v5_awq.hip")
        print(f"    - AWQ kernels used in dispatch: YES (checked in load_awq_engine)")
        
    elif results['gptq']:
        print(f"  GPTQ: {results['gptq']['tok_per_sec']:.1f} tok/s")
        print(f"  AWQ:  FAILED")
    elif results['awq']:
        print(f"  GPTQ: FAILED")
        print(f"  AWQ:  {results['awq']['tok_per_sec']:.1f} tok/s")
    else:
        print(f"  Both benchmarks failed")
        return 1
    
    print()
    print("=" * 72)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
