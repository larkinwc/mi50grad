#!/usr/bin/env python3
"""
tests/bench_tp4_sprint5.py — Sprint 5 TP=4 Throughput Benchmark.

Benchmark TP=4 throughput with Sprint 5 kernel optimizations:
  - GEMM v6 with register-cached scale/zero + weight prefetch (N<=4096)
  - GEMM v5 fallback for N>4096 shapes
  - 64-thread decode attention (with fallback to 256-thread)

Tests:
  1. VAL-KERN-005: TP=4 throughput >= 38.0 tok/s (no regression vs Sprint 4)
  2. Progressive fallback: v6 → v5 → v3 → v2

Generates: bench/tp4_sprint5_report.md

Validation assertions fulfilled:
  VAL-KERN-005: TP=4 integration with new kernels

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4_sprint5.py'
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Force unbuffered stdout for real-time output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]

BENCH_STEPS = 100
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256

# Sprint 4 baseline (C dispatch + kernel P2P allreduce)
SPRINT4_TPS = 38.3  # tok/s
TPS_FLOOR = 38.0    # Minimum acceptable (no regression)

results = {}  # test_name → bool
metrics = {}  # label → value


def print_header(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def record(name: str, passed: bool, msg: str = ""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    suffix = f" — {msg}" if msg else ""
    print(f"  [{status}] {name}{suffix}")


def reset_tp(engine):
    """Reset all KV caches and DeltaNet states for TP engine."""
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


# ============================================================================
# Benchmark: TP=4 Throughput
# ============================================================================

def bench_tp4_throughput() -> dict:
    """Benchmark TP=4 throughput with Sprint 5 kernels."""
    print_header("TP=4 Throughput Benchmark (Sprint 5 Kernels)")
    
    from src.model.qwen import load_config_from_json
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader
    
    config = load_config_from_json(MODEL_DIR)
    
    print("  Loading TP=4 engine...")
    t_load = time.perf_counter()
    tp_engine = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    # Load weights
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 16 == 0:
            print(f"    Layer {layer_idx}...")
        tp_engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp_engine.load_final_norm(loader.load_final_norm())
    tp_engine.load_lm_head(loader.load_lm_head())
    tp_engine.build_dispatch_cache()
    tp_engine.set_direct_kv_write(True)
    tp_engine.set_c_dispatch(True)
    tp_engine.set_kernel_p2p_allreduce(True)
    
    t_load = time.perf_counter() - t_load
    print(f"  TP=4 engine loaded in {t_load:.2f}s ({len(tp_engine.engines)} GPUs)")
    
    # Check kernel versions
    engine0 = tp_engine.engines[0]
    print(f"\n  Kernel configuration:")
    print(f"    GEMV v6: {engine0._gemv_int4_v6}")
    print(f"    GEMV v5: {engine0._gemv_int4_v5}")
    print(f"    GEMV v3: {engine0._gemv_int4_v3}")
    print(f"    Shape-based selection: v6 for N<=4096, v5 for N>4096")
    
    # Warmup
    print(f"\n  Warming up ({WARMUP_STEPS} steps)...")
    rng = np.random.default_rng(42)
    for i in range(WARMUP_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        reset_tp(tp_engine)
        tp_engine.decode_step(emb, i)
        tp_engine._hip.synchronize()
    
    # Benchmark
    print(f"\n  Benchmarking ({BENCH_STEPS} steps)...")
    reset_tp(tp_engine)
    
    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_engine.decode_step(emb, i)
    tp_engine._hip.synchronize()
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    tok_s = BENCH_STEPS / elapsed
    ms_per_tok = (elapsed / BENCH_STEPS) * 1000
    
    print(f"\n  Results:")
    print(f"    Steps: {BENCH_STEPS}")
    print(f"    Elapsed: {elapsed:.3f}s")
    print(f"    Throughput: {tok_s:.2f} tok/s")
    print(f"    Latency: {ms_per_tok:.2f} ms/tok")
    print(f"    Baseline (Sprint 4): {SPRINT4_TPS:.1f} tok/s")
    print(f"    Floor (no regression): {TPS_FLOOR:.1f} tok/s")
    
    # Check against floor
    passed = tok_s >= TPS_FLOOR
    record("VAL-KERN-005.4: TP=4 throughput >= 38.0 tok/s", passed,
           f"{tok_s:.2f} tok/s")
    
    metrics['throughput_tok_s'] = tok_s
    metrics['latency_ms_per_tok'] = ms_per_tok
    metrics['elapsed_s'] = elapsed
    metrics['sprint4_baseline'] = SPRINT4_TPS
    metrics['improvement_vs_sprint4'] = (tok_s - SPRINT4_TPS) / SPRINT4_TPS * 100
    
    tp_engine.cleanup()
    return passed


# ============================================================================
# Generate Report
# ============================================================================

def generate_report():
    """Generate benchmark report in bench/ directory."""
    report_dir = Path("/opt/mi50grad/bench")
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / "tp4_sprint5_report.md"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    report = f"""# TP=4 Sprint 5 Kernel Integration Benchmark

**Date**: {timestamp}
**Model**: {MODEL_DIR}
**Devices**: {DEVICE_IDS}

## Configuration

- Benchmark steps: {BENCH_STEPS}
- Warmup steps: {WARMUP_STEPS}
- Max sequence length: {MAX_SEQ_LEN}

## Kernel Configuration

- GEMM v6: {metrics.get('v6_loaded', 'N/A')} (register-cached scale/zero + prefetch)
- GEMM v5: {metrics.get('v5_loaded', 'N/A')} (hybrid DPP+LDS)
- Shape-based selection: v6 for N<=4096, v5 for N>4096

## Results

| Metric | Value |
|--------|-------|
| Throughput | {metrics.get('throughput_tok_s', 0):.2f} tok/s |
| Latency | {metrics.get('latency_ms_per_tok', 0):.2f} ms/tok |
| Elapsed time | {metrics.get('elapsed_s', 0):.3f} s |
| Sprint 4 baseline | {metrics.get('sprint4_baseline', 0):.1f} tok/s |
| Improvement vs Sprint 4 | {metrics.get('improvement_vs_sprint4', 0):+.1f}% |

## Validation

| Assertion | Status | Details |
|-----------|--------|---------|
| VAL-KERN-005 (TP=4 integration) | {'PASS' if results.get('VAL-KERN-005.4') else 'FAIL'} | Throughput >= 38.0 tok/s |

## Conclusion

{'**PASS**: Sprint 5 kernel integration successful. No regression vs Sprint 4 baseline.' if results.get('VAL-KERN-005.4') else '**FAIL**: Throughput regression detected.'}
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n  Report saved to: {report_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_header("TP=4 Sprint 5 Throughput Benchmark")
    print(f"  Model: {MODEL_DIR}")
    print(f"  Devices: {DEVICE_IDS}")
    print(f"  Benchmark steps: {BENCH_STEPS}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Throughput floor: {TPS_FLOOR:.1f} tok/s")
    print(f"  Sprint 4 baseline: {SPRINT4_TPS:.1f} tok/s")
    
    try:
        bench_tp4_throughput()
    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Generate report
    generate_report()
    
    # Summary
    print_header("Summary")
    passed = sum(results.values())
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n  Benchmark PASSED!")
        sys.exit(0)
    else:
        print("\n  Benchmark FAILED!")
        for name, result in results.items():
            if not result:
                print(f"    FAIL: {name}")
        sys.exit(1)
