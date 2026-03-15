#!/usr/bin/env python3
"""
Dispatch overhead test: measures per-launch Python overhead before and after
parameter pre-caching optimization.

Tests:
1. Per-launch Python overhead: uncached vs cached ctypes construction
2. Total dispatch time for 64 layers on 4 GPUs (simulated, no GPU)
3. Correctness: cached dispatch produces same output as serial dispatch
   (cosine similarity > 0.999)
4. Speedup report

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_dispatch_overhead.py'
"""

import sys
import time
import ctypes
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
BENCH_STEPS = 20
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
    """Reset KV cache and DeltaNet state."""
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


def bench_ctypes_overhead():
    """Benchmark Python overhead of ctypes parameter construction vs pre-cached.

    Measures the per-launch overhead of building ctypes arrays vs reusing them.
    This quantifies the bottleneck that pre-caching addresses.
    """
    print("\n" + "=" * 60)
    print("CTYPES CONSTRUCTION OVERHEAD BENCHMARK")
    print("=" * 60)
    print("Measuring per-launch Python overhead: uncached vs cached")

    ITERS = 10000
    N_PARAMS = 8  # typical kernel param count

    # --- Uncached: build ctypes objects from scratch each iteration ---
    params_data = [
        0x1234567890, 0xABCDEF01, 0xDEADBEEF00, 0xCAFEBABE00,
        0x12345678, 1024, 4096, 128
    ]

    t0 = time.perf_counter()
    for _ in range(ITERS):
        params = [
            ctypes.c_uint64(params_data[0]),
            ctypes.c_uint64(params_data[1]),
            ctypes.c_uint64(params_data[2]),
            ctypes.c_uint64(params_data[3]),
            ctypes.c_uint32(params_data[4]),
            ctypes.c_uint32(params_data[5]),
            ctypes.c_uint32(params_data[6]),
            ctypes.c_uint32(params_data[7]),
        ]
        n = len(params)
        arr = (ctypes.c_void_p * n)()
        for i, p in enumerate(params):
            arr[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)
    uncached_us = (time.perf_counter() - t0) / ITERS * 1e6

    # --- Cached: pre-build once, then update only mutable values ---
    from src.runtime.hip_dispatch import LaunchSpec

    # This simulates a kernel with 2 mutable params (e.g., cos/sin ptr)
    cached_params = [
        ctypes.c_uint64(params_data[0]),
        ctypes.c_uint64(params_data[1]),
        ctypes.c_uint64(params_data[2]),  # mutable: cos_ptr
        ctypes.c_uint64(params_data[3]),  # mutable: sin_ptr
        ctypes.c_uint32(params_data[4]),
        ctypes.c_uint32(params_data[5]),
        ctypes.c_uint32(params_data[6]),
        ctypes.c_uint32(params_data[7]),
    ]
    # Simulate LaunchSpec pre-build (done once at init)
    n = len(cached_params)
    pre_arr = (ctypes.c_void_p * n)()
    for i, p in enumerate(cached_params):
        pre_arr[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

    cos_ptr_val = params_data[2]
    sin_ptr_val = params_data[3]

    t0 = time.perf_counter()
    for _ in range(ITERS):
        # Only update position-dependent values in-place
        cached_params[2].value = cos_ptr_val
        cached_params[3].value = sin_ptr_val
        # pre_arr already points to cached_params[2] and [3] — no re-construction needed
    cached_us = (time.perf_counter() - t0) / ITERS * 1e6

    speedup = uncached_us / cached_us if cached_us > 0 else float('inf')
    print(f"  Uncached (rebuild ctypes+array):  {uncached_us:.2f} μs/launch")
    print(f"  Cached   (update 2 values only):  {cached_us:.2f} μs/launch")
    print(f"  Speedup:                          {speedup:.1f}x")

    # Estimate total savings for TP=4 decode step
    # Typical: ~10 launches/layer × 64 layers × 4 GPUs = 2560 launches
    total_launches = 10 * 64 * 4
    uncached_total = uncached_us * total_launches / 1000  # ms
    cached_total = cached_us * total_launches / 1000      # ms
    savings = uncached_total - cached_total
    print(f"\n  Extrapolated for TP=4 decode step ({total_launches} launches):")
    print(f"    Uncached: {uncached_total:.1f} ms")
    print(f"    Cached:   {cached_total:.1f} ms")
    print(f"    Savings:  {savings:.1f} ms")

    return uncached_us, cached_us, speedup


def bench_decode_mode(engine: TPInferenceEngine, emb: np.ndarray,
                       mode: str, label: str, steps: int = BENCH_STEPS):
    """Run decode steps in specified mode, return (outputs, tok_per_sec, mean_ms)."""
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

    print(f"[{label}] Running {steps} timed decode steps...")
    reset_engine(engine)
    outputs = []

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
    print("Dispatch Overhead Test: Parameter Pre-Caching Optimization")
    print("=" * 70)
    print(f"Model:        {MODEL_DIR}")
    print(f"GPUs:         {DEVICE_IDS}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Bench steps:  {BENCH_STEPS}")
    print(f"Cosine sim threshold: {COSINE_SIM_THRESHOLD}")

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

    # --- Micro-benchmark: ctypes construction overhead ---
    uncached_us, cached_us, ctypes_speedup = bench_ctypes_overhead()

    # --- Load model ---
    print(f"\nLoading config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"Config: {config.num_hidden_layers} layers, "
          f"hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}")

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
    print("\nBuilding dispatch cache (pre-caching ctypes parameters)...")
    t_cache = time.perf_counter()
    engine.build_dispatch_cache()
    t_cache_elapsed = time.perf_counter() - t_cache
    print(f"Dispatch cache built in {t_cache_elapsed:.2f}s")

    # Fixed input for reproducibility
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # --- Serial baseline ---
    serial_outputs, serial_tps, serial_mean_ms = bench_decode_mode(
        engine, emb, mode='serial', label="SERIAL")

    # --- Cached dispatch ---
    cached_outputs, cached_tps, cached_mean_ms = bench_decode_mode(
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
            print(f"{step:>4}  {'nan':>12}  {'FAIL(NaN)':>10}  {'nan':>12}")
            all_pass = False
        else:
            max_diff = float(np.max(np.abs(ref.astype(np.float32) -
                                            cached.astype(np.float32))))
            status = "PASS" if cos_sim >= COSINE_SIM_THRESHOLD else "FAIL"
            if cos_sim < COSINE_SIM_THRESHOLD:
                all_pass = False
            min_cosine = min(min_cosine, cos_sim)
            print(f"{step:>4}  {cos_sim:>12.6f}  {status:>10}  {max_diff:>12.4e}")

    print("-" * 54)
    print(f"Min cosine similarity: {min_cosine:.6f}")
    print(f"Threshold: {COSINE_SIM_THRESHOLD}")

    # --- Performance summary ---
    speedup = serial_mean_ms / cached_mean_ms if cached_mean_ms > 0 else float('nan')

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} {'tok/s':>10} {'ms/tok':>10}")
    print("-" * 45)
    print(f"{'Serial':<20} {serial_tps:>10.2f} {serial_mean_ms:>10.2f}")
    print(f"{'Cached':<20} {cached_tps:>10.2f} {cached_mean_ms:>10.2f}")
    print("-" * 45)
    print(f"  Speedup (cached/serial): {speedup:.3f}x")
    print()
    print("PER-LAUNCH OVERHEAD BENCHMARK:")
    print(f"  Uncached ctypes: {uncached_us:.2f} μs/launch")
    print(f"  Cached ctypes:   {cached_us:.2f} μs/launch")
    print(f"  ctypes speedup:  {ctypes_speedup:.1f}x")
    print()
    print("BASELINES FOR COMPARISON:")
    print(f"  Single-GPU:   20.3 tok/s  (49.3 ms/tok)")
    print(f"  vLLM TP=4:    46.9 tok/s")
    print("=" * 70)

    print("\n" + "=" * 70)
    correctness_ok = all_pass
    speedup_ok = speedup > 1.0

    if correctness_ok and speedup_ok:
        print("RESULT: PASS")
        print(f"  Correctness: cosine sim >= {COSINE_SIM_THRESHOLD} for all steps ✓")
        print(f"  Performance: cached {speedup:.3f}x faster than serial ✓")
    else:
        print("RESULT: FAIL")
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
