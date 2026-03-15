#!/usr/bin/env python3
"""
Combined cached dispatch + stream overlap correctness and benchmark test.

Tests the combined mode that merges:
- Cached dispatch (pre-built ctypes parameter arrays, 23.7 tok/s)
- Stream overlap (async allreduce on dedicated HIP streams, 14.6 tok/s)

Expected: combined mode should be faster than either mode alone (~28-33 tok/s).

Verifies:
1. Correctness: combined mode output matches serial output (cosine sim > 0.99)
2. Benchmark: combined vs cached-only vs serial tok/s
3. Profile breakdown showing reduced effective allreduce overhead

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_combined_dispatch.py'
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
CORRECTNESS_STEPS = 10
BENCH_STEPS = 100
COSINE_SIM_THRESHOLD = 0.99


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


def run_mode_correctness(engine: TPInferenceEngine, emb: np.ndarray,
                         mode: str, steps: int):
    """Run decode steps in specified mode, collect outputs.

    mode: 'serial', 'cached', 'stream_overlap', 'combined'
    """
    engine.set_cached_dispatch(False)
    engine.set_stream_overlap_dispatch(False)
    engine.set_threaded_dispatch(False)

    if mode == 'serial':
        pass
    elif mode == 'cached':
        engine.set_cached_dispatch(True)
    elif mode == 'stream_overlap':
        engine.set_stream_overlap_dispatch(True)
    elif mode == 'combined':
        engine.set_cached_dispatch(True)
        engine.set_stream_overlap_dispatch(True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    reset_engine(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.synchronize()

    # Timed correctness run
    reset_engine(engine)
    outputs = []
    for i in range(steps):
        out = engine.decode_step(emb, WARMUP_STEPS + i)
        outputs.append(out.copy())
    engine.synchronize()

    return outputs


def run_mode_benchmark(engine: TPInferenceEngine, emb: np.ndarray,
                       mode: str, warmup: int, steps: int):
    """Run benchmark for a given mode, return (tok_per_sec, ms_per_tok)."""
    engine.set_cached_dispatch(False)
    engine.set_stream_overlap_dispatch(False)
    engine.set_threaded_dispatch(False)

    if mode == 'serial':
        pass
    elif mode == 'cached':
        engine.set_cached_dispatch(True)
    elif mode == 'stream_overlap':
        engine.set_stream_overlap_dispatch(True)
    elif mode == 'combined':
        engine.set_cached_dispatch(True)
        engine.set_stream_overlap_dispatch(True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    reset_engine(engine)
    for i in range(warmup):
        engine.decode_step(emb, i)
    engine.synchronize()

    # Timed run
    reset_engine(engine)
    t0 = time.perf_counter()
    for i in range(steps):
        engine.decode_step(emb, warmup + i)
    engine.synchronize()
    total_elapsed = time.perf_counter() - t0

    tok_per_sec = steps / total_elapsed
    ms_per_tok = total_elapsed / steps * 1000.0
    return tok_per_sec, ms_per_tok


def main():
    print("=" * 70)
    print("Combined Cached Dispatch + Stream Overlap: Correctness and Benchmark")
    print("=" * 70)
    print(f"Model:             {MODEL_DIR}")
    print(f"GPUs:              {DEVICE_IDS}")
    print(f"Warmup steps:      {WARMUP_STEPS}")
    print(f"Correctness steps: {CORRECTNESS_STEPS}")
    print(f"Bench steps:       {BENCH_STEPS}")
    print(f"Cosine threshold:  {COSINE_SIM_THRESHOLD}")
    print()

    # Verify GPU count
    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs visible: {n_gpus}")
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

    # Load TP=4 engine
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

    # Build dispatch cache (required for cached and combined modes)
    print("\nBuilding dispatch cache...")
    engine.build_dispatch_cache()

    # Verify P2P allreduce is available (required for combined and stream_overlap modes)
    if engine._p2p_ar is None:
        print("ERROR: P2P allreduce not available. Combined mode requires P2P AR.")
        sys.exit(1)
    print(f"P2P allreduce: available (TP={engine.tp_size})")

    # Fixed input for reproducibility
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # =========================================================================
    # CORRECTNESS TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST: Combined mode vs Serial reference")
    print("=" * 70)
    print(f"Running {CORRECTNESS_STEPS} decode steps in each mode...")

    # Serial reference
    print("\n[SERIAL] Running reference decode steps...")
    serial_outputs = run_mode_correctness(engine, emb, 'serial', CORRECTNESS_STEPS)
    print(f"[SERIAL] Done ({CORRECTNESS_STEPS} steps)")

    # Cached mode outputs
    print("\n[CACHED] Running cached dispatch decode steps...")
    cached_outputs = run_mode_correctness(engine, emb, 'cached', CORRECTNESS_STEPS)
    print(f"[CACHED] Done ({CORRECTNESS_STEPS} steps)")

    # Stream overlap mode outputs
    print("\n[STREAM OVERLAP] Running stream overlap decode steps...")
    overlap_outputs = run_mode_correctness(engine, emb, 'stream_overlap', CORRECTNESS_STEPS)
    print(f"[STREAM OVERLAP] Done ({CORRECTNESS_STEPS} steps)")

    # Combined mode outputs
    print("\n[COMBINED] Running combined cached+stream decode steps...")
    combined_outputs = run_mode_correctness(engine, emb, 'combined', CORRECTNESS_STEPS)
    print(f"[COMBINED] Done ({CORRECTNESS_STEPS} steps)")

    # Compare outputs
    print(f"\n{'Step':>4}  {'Cached':>12}  {'Overlap':>12}  {'Combined':>12}  {'Status':>10}")
    print("-" * 60)

    all_pass = True
    min_cosine_cached = 1.0
    min_cosine_overlap = 1.0
    min_cosine_combined = 1.0

    for step in range(CORRECTNESS_STEPS):
        ref = serial_outputs[step]
        cos_cached = cosine_similarity(ref, cached_outputs[step])
        cos_overlap = cosine_similarity(ref, overlap_outputs[step])
        cos_combined = cosine_similarity(ref, combined_outputs[step])

        combined_ok = (not np.isnan(cos_combined) and
                       cos_combined >= COSINE_SIM_THRESHOLD)
        if not combined_ok:
            all_pass = False

        if not np.isnan(cos_cached):
            min_cosine_cached = min(min_cosine_cached, cos_cached)
        if not np.isnan(cos_overlap):
            min_cosine_overlap = min(min_cosine_overlap, cos_overlap)
        if not np.isnan(cos_combined):
            min_cosine_combined = min(min_cosine_combined, cos_combined)

        status = "PASS" if combined_ok else "FAIL"
        print(f"{step:>4}  {cos_cached:>12.6f}  {cos_overlap:>12.6f}  "
              f"{cos_combined:>12.6f}  {status:>10}")

    print("-" * 60)
    print(f"Min cosine (cached):   {min_cosine_cached:.6f}")
    print(f"Min cosine (overlap):  {min_cosine_overlap:.6f}")
    print(f"Min cosine (combined): {min_cosine_combined:.6f}")
    print(f"Threshold:             {COSINE_SIM_THRESHOLD}")

    correctness_pass = all_pass
    print(f"\nCorrectness result: {'PASS' if correctness_pass else 'FAIL'}")

    # =========================================================================
    # BENCHMARK TEST
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {BENCH_STEPS} decode steps, {WARMUP_STEPS} warmup")
    print("=" * 70)

    print("\nRunning serial benchmark...")
    serial_tps, serial_ms = run_mode_benchmark(
        engine, emb, 'serial', WARMUP_STEPS, BENCH_STEPS)
    print(f"[SERIAL]        {serial_tps:.2f} tok/s  ({serial_ms:.2f} ms/tok)")

    print("Running cached benchmark...")
    cached_tps, cached_ms = run_mode_benchmark(
        engine, emb, 'cached', WARMUP_STEPS, BENCH_STEPS)
    print(f"[CACHED]        {cached_tps:.2f} tok/s  ({cached_ms:.2f} ms/tok)")

    print("Running stream overlap benchmark...")
    overlap_tps, overlap_ms = run_mode_benchmark(
        engine, emb, 'stream_overlap', WARMUP_STEPS, BENCH_STEPS)
    print(f"[STREAM OVERLAP]{overlap_tps:10.2f} tok/s  ({overlap_ms:.2f} ms/tok)")

    print("Running combined benchmark...")
    combined_tps, combined_ms = run_mode_benchmark(
        engine, emb, 'combined', WARMUP_STEPS, BENCH_STEPS)
    print(f"[COMBINED]      {combined_tps:.2f} tok/s  ({combined_ms:.2f} ms/tok)")

    # Speedup calculations
    speedup_cached = serial_ms / cached_ms if cached_ms > 0 else float('nan')
    speedup_overlap = serial_ms / overlap_ms if overlap_ms > 0 else float('nan')
    speedup_combined = serial_ms / combined_ms if combined_ms > 0 else float('nan')
    speedup_combined_vs_cached = cached_ms / combined_ms if combined_ms > 0 else float('nan')

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} {'tok/s':>10} {'ms/tok':>10} {'vs serial':>12}")
    print("-" * 56)
    print(f"{'Serial':<20} {serial_tps:>10.2f} {serial_ms:>10.2f} {'1.000x':>12}")
    print(f"{'Cached':<20} {cached_tps:>10.2f} {cached_ms:>10.2f} {speedup_cached:>11.3f}x")
    print(f"{'Stream Overlap':<20} {overlap_tps:>10.2f} {overlap_ms:>10.2f} {speedup_overlap:>11.3f}x")
    print(f"{'Combined':<20} {combined_tps:>10.2f} {combined_ms:>10.2f} {speedup_combined:>11.3f}x")
    print("-" * 56)
    print(f"  Combined vs cached: {speedup_combined_vs_cached:.3f}x")
    print()
    print("BASELINES FOR COMPARISON:")
    print(f"  Single-GPU:        20.3 tok/s  (49.3 ms/tok)")
    print(f"  vLLM TP=4:         46.9 tok/s")
    print(f"  Cached baseline:   23.7 tok/s  (42.6 ms/tok)")
    print(f"  Stream overlap:    14.6 tok/s  (68.5 ms/tok)")
    print("=" * 70)

    # =========================================================================
    # PROFILE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROFILE BREAKDOWN: Allreduce overlap with Python dispatch")
    print("=" * 70)
    print("In combined mode, allreduce runs on dedicated HIP streams while")
    print("Python dispatches next-layer cached kernel params.")
    print()
    print("Key metrics:")
    print(f"  Cached dispatch overhead:        ~14 ms/tok (vs 42ms serial)")
    print(f"  Allreduce overhead (synchronous): ~28 ms/tok (in cached mode)")
    print(f"  If fully overlapped:              max(14, 28) = 28 ms → ~35 tok/s")
    print(f"  Combined actual:                  {combined_ms:.2f} ms/tok → {combined_tps:.1f} tok/s")
    overlap_hidden = max(0, 28.0 - max(0, combined_ms - 14.0))
    print(f"  Estimated allreduce hidden:       {overlap_hidden:.1f} ms/tok")
    print("=" * 70)

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    print("\n" + "=" * 70)

    # Performance assertions
    combined_beats_cached = combined_tps > cached_tps * 0.95  # at least 95% of cached
    combined_beats_serial = combined_tps > serial_tps

    print(f"CORRECTNESS:  {'PASS' if correctness_pass else 'FAIL'}")
    print(f"  Combined cosine similarity >= {COSINE_SIM_THRESHOLD} "
          f"for all {CORRECTNESS_STEPS} steps: "
          f"{'✓' if correctness_pass else '✗'}")
    print(f"  Min combined cosine sim: {min_cosine_combined:.6f}")
    print()
    print(f"PERFORMANCE:")
    print(f"  Combined ({combined_tps:.2f} tok/s) vs Serial ({serial_tps:.2f} tok/s): "
          f"{'✓' if combined_beats_serial else '✗'}")
    print(f"  Combined ({combined_tps:.2f} tok/s) vs Cached ({cached_tps:.2f} tok/s): "
          f"{'✓ (faster)' if combined_tps >= cached_tps else '~ (within 5%)' if combined_beats_cached else '✗ (slower)'}")
    print()

    final_pass = correctness_pass and combined_beats_serial
    print(f"FINAL RESULT: {'PASS' if final_pass else 'FAIL'}")
    print("=" * 70)

    # Cleanup
    engine.cleanup()

    if not final_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
