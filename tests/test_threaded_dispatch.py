#!/usr/bin/env python3
"""
Dispatch optimization correctness and performance test.

Tests:
1. Correctness: cached dispatch output == serial dispatch output (cosine sim > 0.999)
2. Performance: cached dispatch is faster than serial dispatch
3. Reports speedup ratio

NOTE: Python threading for GPU dispatch is counter-productive (see architecture.md).
      hipDeviceSynchronize takes ~0.6μs on idle GPU; hipModuleLaunchKernel is async,
      so threading adds 490μs/round overhead × 128 rounds = 63ms penalty per step.
      The optimized path is CACHED dispatch (pre-built ctypes parameter arrays).

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_threaded_dispatch.py'
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
WARMUP_STEPS = 3
BENCH_STEPS = 10
COSINE_SIM_THRESHOLD = 0.999


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two FP16 vectors."""
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    if np.any(np.isnan(a32)) or np.any(np.isnan(b32)):
        return float('nan')
    dot = float(np.dot(a32, b32))
    norm_a = float(np.linalg.norm(a32))
    norm_b = float(np.linalg.norm(b32))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def reset_engine(engine: TPInferenceEngine):
    """Reset KV cache and DeltaNet state for a fresh decode sequence."""
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


def bench_mode(engine: TPInferenceEngine, emb: np.ndarray,
               mode: str, label: str, steps: int = BENCH_STEPS):
    """Run decode steps in specified mode, return (outputs, tok_per_sec, mean_ms).

    mode: 'serial' or 'cached'
    """
    print(f"\n[{label}] Warming up {WARMUP_STEPS} steps (mode={mode})...")
    reset_engine(engine)

    if mode == 'serial':
        engine.set_cached_dispatch(False)
        engine.set_threaded_dispatch(False)
    elif mode == 'cached':
        engine.set_cached_dispatch(True)
        engine.set_threaded_dispatch(False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.synchronize()

    print(f"[{label}] Running {steps} timed decode steps (total timing)...")
    reset_engine(engine)
    outputs = []

    # Total timing (like bench_tp4.py) - avoids per-step sync overhead
    t_total = time.perf_counter()
    for i in range(steps):
        out = engine.decode_step(emb, WARMUP_STEPS + i)
        outputs.append(out.copy())
    engine.synchronize()
    total_elapsed = time.perf_counter() - t_total

    tok_per_sec = steps / total_elapsed
    mean_ms = total_elapsed / steps * 1000.0

    print(f"[{label}] Throughput: {tok_per_sec:.2f} tok/s")
    print(f"[{label}] Mean latency: {mean_ms:.2f} ms/tok")
    return outputs, tok_per_sec, mean_ms


def main():
    print("=" * 70)
    print("Dispatch Optimization: Correctness and Performance Test")
    print("=" * 70)
    print(f"Model:        {MODEL_DIR}")
    print(f"GPUs:         {DEVICE_IDS}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Bench steps:  {BENCH_STEPS}")
    print(f"Cosine sim threshold: {COSINE_SIM_THRESHOLD}")
    print()
    print("Testing CACHED dispatch (pre-built ctypes parameter arrays)")
    print("vs SERIAL dispatch (rebuilds ctypes params each launch)")

    # Verify GPU count
    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"\nGPUs visible: {n_gpus}")
    if n_gpus < 4:
        print(f"ERROR: Need 4 GPUs, only {n_gpus} visible.")
        print("Make sure to use: -e HIP_VISIBLE_DEVICES=0,1,2,3")
        sys.exit(1)

    # Load config
    print(f"\nLoading config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"Config: {config.num_hidden_layers} layers, "
          f"hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}")

    # Load engine
    print(f"\nLoading TP=4 engine on GPUs {DEVICE_IDS}...")
    t_load = time.perf_counter()
    engine = TPInferenceEngine(config, DEVICE_IDS)
    loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    t_load_elapsed = time.perf_counter() - t_load
    print(f"Weights loaded in {t_load_elapsed:.1f}s")

    # Build dispatch cache
    print("\nBuilding dispatch cache...")
    engine.build_dispatch_cache()

    # Fixed input for reproducibility
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # --- Serial benchmark ---
    serial_outputs, serial_tps, serial_mean_ms = bench_mode(
        engine, emb, mode='serial', label="SERIAL")

    # --- Cached dispatch benchmark ---
    cached_outputs, cached_tps, cached_mean_ms = bench_mode(
        engine, emb, mode='cached', label="CACHED")

    # --- Correctness comparison ---
    print("\n" + "=" * 70)
    print("CORRECTNESS: CACHED vs SERIAL")
    print("=" * 70)
    print(f"{'Step':>4}  {'Cosine Sim':>12}  {'Status':>10}  {'Max|diff|':>12}")
    print("-" * 54)

    all_pass = True
    min_cosine = 1.0

    for step in range(BENCH_STEPS):
        ref = serial_outputs[step]
        cached = cached_outputs[step]
        cos_sim = cosine_similarity(ref, cached)
        if np.isnan(cos_sim):
            max_diff = float('nan')
            status = "FAIL(NaN)"
            all_pass = False
        else:
            max_diff = float(np.max(np.abs(ref.astype(np.float32) - cached.astype(np.float32))))
            if cos_sim >= COSINE_SIM_THRESHOLD:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
            min_cosine = min(min_cosine, cos_sim)
        if np.isnan(cos_sim):
            print(f"{step:>4}  {'nan':>12}  {status:>10}  {'nan':>12}")
        else:
            print(f"{step:>4}  {cos_sim:>12.6f}  {status:>10}  {max_diff:>12.4e}")

    print("-" * 54)
    print(f"Min cosine similarity: {min_cosine:.6f}")
    print(f"Threshold: {COSINE_SIM_THRESHOLD}")

    # --- Performance summary ---
    speedup = serial_mean_ms / cached_mean_ms if cached_mean_ms > 0 else float('nan')
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Serial dispatch:   {serial_tps:.2f} tok/s  ({serial_mean_ms:.2f} ms/tok)")
    print(f"  Cached dispatch:   {cached_tps:.2f} tok/s  ({cached_mean_ms:.2f} ms/tok)")
    print(f"  Speedup:           {speedup:.3f}x")
    print()
    print("BASELINES FOR COMPARISON:")
    print(f"  Single-GPU:  20.3 tok/s  (49.3 ms/tok)")
    print(f"  vLLM TP=4:   46.9 tok/s")
    print("=" * 70)

    print("\n" + "=" * 70)
    correctness_ok = all_pass
    speedup_ok = speedup > 1.0

    if correctness_ok and speedup_ok:
        print(f"RESULT: PASS")
        print(f"  Correctness: cosine sim >= {COSINE_SIM_THRESHOLD} for all steps ✓")
        print(f"  Performance: cached {speedup:.3f}x faster than serial ✓")
    else:
        print(f"RESULT: FAIL")
        if not correctness_ok:
            print(f"  FAIL: Correctness — some steps have cosine sim < {COSINE_SIM_THRESHOLD}")
        if not speedup_ok:
            print(f"  FAIL: Performance — cached not faster than serial ({speedup:.3f}x)")
    print("=" * 70)

    engine.cleanup()

    if not (correctness_ok and speedup_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
