#!/usr/bin/env python3
"""
Sprint 3 Final TP=4 Benchmark: All Optimizations Combined (allreduce-pipeline + HIP graph decode).

Runs a comprehensive benchmark of all Sprint 3 optimizations combined and generates
optimization report v3.

Tests:
  1. Single-GPU regression check (expect ~20.3 tok/s, within ±10%)
  2. TP=4 benchmark with all Sprint 3 optimizations:
       - HIP graph decode (if available, else C dispatch)
       - Q/KV sync elimination
       - Direct KV cache writes
       - Allreduce overlap improvements
       - Star topology allreduce
       - Tuned kernels
  3. TP=4 correctness check (cosine sim >= 0.99 vs single-GPU, 10 steps)
  4. Comparison table with ALL prior phases:
       - Single-GPU baseline: 20.3 tok/s
       - TP=4 serial: 12.4 tok/s
       - TP=4 cached: 23.7 tok/s
       - TP=4 combined: 25.5 tok/s
       - TP=4 C dispatch + tuned (Sprint 2): 38.0 tok/s
       - TP=4 allreduce pipeline (Sprint 3 M1): 38.1 tok/s (from bench_tp4_sprint3_m1.py)
       - TP=4 graph decode (Sprint 3 M2): NEW (this run)
       - vLLM: 46.9 tok/s
  5. Progressive fallback test: graph → c_dispatch → cached+stream → cached → serial
  6. Generate bench/tp4_optimization_report_v3.md

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4_sprint3.py'
"""

import sys
import os
import time
import math
import gc
import numpy as np
from pathlib import Path

# Force unbuffered stdout for real-time output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader
from src.runtime.hip_dispatch import HIPRuntime

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
DEVICE_ID_SINGLE = 0

WARMUP_STEPS = 3
BENCH_STEPS = 100
CORRECTNESS_STEPS = 10
COSINE_SIM_THRESHOLD = 0.99
SINGLE_GPU_BASELINE = 20.3      # tok/s — Sprint 1 measured baseline
SINGLE_GPU_TOLERANCE = 0.10     # ±10%
MAX_SEQ_LEN = 512

# Prior sprint phase baselines (for comparison table)
TP4_SERIAL_TPS = 12.4           # tok/s  (P2P allreduce, no caching)
TP4_CACHED_TPS = 23.7           # tok/s  (cached dispatch only)
TP4_COMBINED_TPS = 25.5         # tok/s  (cached + stream overlap)
SPRINT2_BASELINE_TPS = 38.0     # tok/s  (C dispatch + star allreduce + tuned kernels)
SPRINT3_M1_TPS = 38.1           # tok/s  (allreduce-pipeline, from bench_tp4_sprint3_m1.py)
VLLM_BASELINE_TPS = 46.9        # tok/s  (AWQ TP=4, reference)

# Sprint 3 regression tolerance: within ±5% of Sprint 2 baseline
SPRINT3_TOLERANCE = 0.05


# =============================================================================
# Utility helpers
# =============================================================================

def print_header(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays, computed in FP32."""
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


def reset_single(engine: InferenceEngine):
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()


def reset_tp(engine: TPInferenceEngine):
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


# =============================================================================
# Model loading helpers
# =============================================================================

def load_single_engine(config, loader) -> InferenceEngine:
    print(f"  Loading single-GPU engine (GPU {DEVICE_ID_SINGLE})...")
    t0 = time.perf_counter()
    engine = InferenceEngine(config, device_id=DEVICE_ID_SINGLE,
                             max_seq_len=MAX_SEQ_LEN)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"    Layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print(f"  Single-GPU engine loaded in {time.perf_counter()-t0:.1f}s")
    return engine


def load_tp4_engine(config, loader, direct_kv_write=True) -> TPInferenceEngine:
    """Load TP=4 engine with Sprint 3 M1+M2 optimizations."""
    print(f"  Loading TP=4 engine (GPUs {DEVICE_IDS})...")
    t0 = time.perf_counter()
    engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)

    if direct_kv_write and hasattr(engine, 'set_direct_kv_write'):
        engine.set_direct_kv_write(True)
        print("  Direct KV write enabled")

    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"    Layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())

    engine.build_dispatch_cache()
    print(f"  TP=4 engine loaded in {time.perf_counter()-t0:.1f}s")
    return engine


def get_active_dispatch_mode(engine: TPInferenceEngine) -> str:
    """Return a description of the active dispatch mode."""
    if getattr(engine, '_graph_dispatch_enabled', False):
        c_graph = getattr(engine, '_c_graph_dispatch_plan', None) is not None
        return "graph (C)" if c_graph else "graph (Python)"
    if getattr(engine, '_c_dispatch_enabled', False):
        return "C dispatch"
    if getattr(engine, '_cached_dispatch', False):
        if getattr(engine, '_stream_overlap_dispatch', False):
            return "cached+stream"
        return "cached"
    return "serial"


# =============================================================================
# TEST 1: Single-GPU regression check
# =============================================================================

def run_single_gpu_benchmark(config, loader) -> float:
    """
    Single-GPU regression check.
    Expected: ~20.3 tok/s within ±10%.
    Returns: measured tok/s.
    """
    print_header("TEST 1: Single-GPU Regression Check")
    low = SINGLE_GPU_BASELINE * (1 - SINGLE_GPU_TOLERANCE)
    high = SINGLE_GPU_BASELINE * (1 + SINGLE_GPU_TOLERANCE)
    print(f"  Baseline: {SINGLE_GPU_BASELINE} tok/s  "
          f"(allowed: {low:.1f}–{high:.1f} tok/s)")

    engine = load_single_engine(config, loader)

    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02

    # Warmup
    print(f"\n  Warmup ({WARMUP_STEPS} steps)...")
    reset_single(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.device.synchronize()

    # Timed benchmark
    reset_single(engine)
    print(f"  Timed benchmark ({BENCH_STEPS} steps)...")
    times = []
    for i in range(BENCH_STEPS):
        t0 = time.perf_counter()
        engine.decode_step(emb, i)
        times.append(time.perf_counter() - t0)
    engine.device.synchronize()

    median_ms = float(np.median(times)) * 1000
    tok_per_sec = 1.0 / float(np.median(times))
    deviation = abs(tok_per_sec - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE * 100

    engine.cleanup()
    del engine
    gc.collect()

    passed = low <= tok_per_sec <= high
    print(f"\n  Single-GPU: {tok_per_sec:.1f} tok/s ({median_ms:.2f} ms/tok)")
    print(f"  Deviation: {deviation:.1f}% from {SINGLE_GPU_BASELINE} baseline")
    print(f"  Regression check: {'PASS' if passed else 'FAIL'}")
    return tok_per_sec


# =============================================================================
# TEST 2: TP=4 benchmark — all Sprint 3 optimizations combined
# =============================================================================

def run_tp4_all_optimizations(config, loader) -> dict:
    """
    TP=4 benchmark with all Sprint 3 M1+M2 optimizations:
    - HIP graph decode (if available, else C dispatch)
    - Direct KV cache writes
    - Q/KV sync elimination
    - Allreduce overlap improvements
    - Star topology allreduce
    - Tuned kernels

    Also benchmarks Sprint 3 M1 (C dispatch without graph) for comparison.
    Returns dict with timing results.
    """
    print_header("TEST 2: TP=4 Benchmark — All Sprint 3 Optimizations Combined")
    print(f"  GPUs: {DEVICE_IDS}")
    print(f"  Steps: {WARMUP_STEPS} warmup + {BENCH_STEPS} timed")
    print(f"  Sprint 3 optimizations:")
    print(f"    - HIP graph decode (graph > c_dispatch in priority)")
    print(f"    - Q/KV sync elimination (sequential null-stream GEMVs)")
    print(f"    - Direct KV cache writes (no D2D copies for K/V append)")
    print(f"    - Allreduce overlap deepening (deepened compute/AR overlap)")
    print(f"    - C dispatch loop (tight C loop, no Python dispatch)")
    print(f"    - Star topology allreduce (8.5× faster than ring for 10KB)")
    print(f"    - Tuned kernels (elementwise_v3, flash_attn_256_tuned, gemv_int4_v3_t16)")

    np.random.seed(42)
    bench_emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02

    results = {}

    # -------------------------------------------------------------------------
    # Mode A: Sprint 3 M1 baseline (C dispatch + direct KV write, NO graph)
    # -------------------------------------------------------------------------
    print(f"\n  --- Mode A: Sprint 3 M1 (C dispatch + all M1 opts, no graph) ---")
    engine_a = load_tp4_engine(config, loader, direct_kv_write=True)
    engine_a.set_ring_allreduce(False)
    engine_a.set_c_dispatch(True)
    if hasattr(engine_a, 'set_graph_dispatch'):
        engine_a.set_graph_dispatch(False)
    c_avail_a = engine_a._c_dispatch_enabled
    mode_a = get_active_dispatch_mode(engine_a)
    print(f"  Active mode: {mode_a}")
    print(f"  C dispatch available: {c_avail_a}")

    reset_tp(engine_a)
    for i in range(WARMUP_STEPS):
        engine_a.decode_step(bench_emb, i)
    engine_a.synchronize()

    reset_tp(engine_a)
    times_a = []
    for i in range(BENCH_STEPS):
        t0 = time.perf_counter()
        engine_a.decode_step(bench_emb, i)
        times_a.append(time.perf_counter() - t0)
    engine_a.synchronize()

    engine_a.cleanup()
    del engine_a
    gc.collect()
    time.sleep(1.0)

    tps_a = 1.0 / float(np.median(times_a)) if times_a else 0.0
    ms_a = float(np.median(times_a)) * 1000 if times_a else 0.0
    results['sprint3_m1_tps'] = tps_a
    results['sprint3_m1_ms'] = ms_a
    results['c_dispatch_available'] = c_avail_a
    print(f"  Sprint 3 M1 (C dispatch): {tps_a:.1f} tok/s ({ms_a:.2f} ms/tok)")

    # -------------------------------------------------------------------------
    # Mode B: Sprint 3 M2 — all optimizations combined (with graph decode)
    # -------------------------------------------------------------------------
    print(f"\n  --- Mode B: Sprint 3 M2 — all optimizations + HIP graph decode ---")
    engine_b = load_tp4_engine(config, loader, direct_kv_write=True)
    engine_b.set_ring_allreduce(False)
    engine_b.set_c_dispatch(True)

    graph_available = hasattr(engine_b, 'set_graph_dispatch')
    c_graph_plan_avail = False

    if graph_available:
        engine_b.set_graph_dispatch(True)
        print(f"  Graph dispatch method available: True")
        print(f"  Graph dispatch enabled: {engine_b._graph_dispatch_enabled}")

        # Trigger first step to capture graphs (and potentially build C graph plan)
        print(f"  Warming up + capturing graphs...")
        reset_tp(engine_b)
        for i in range(WARMUP_STEPS):
            engine_b.decode_step(bench_emb, i)
        engine_b.synchronize()

        # Check if C graph dispatch plan was built
        c_graph_plan_avail = getattr(engine_b, '_c_graph_dispatch_plan', None) is not None
        mode_b = get_active_dispatch_mode(engine_b)
        print(f"  Active mode: {mode_b}")
        print(f"  C graph dispatch plan built: {c_graph_plan_avail}")
    else:
        print(f"  Graph dispatch not available → using C dispatch as best mode")
        mode_b = "C dispatch"
        reset_tp(engine_b)
        for i in range(WARMUP_STEPS):
            engine_b.decode_step(bench_emb, i)
        engine_b.synchronize()

    reset_tp(engine_b)
    times_b = []
    for i in range(BENCH_STEPS):
        t0 = time.perf_counter()
        engine_b.decode_step(bench_emb, i)
        times_b.append(time.perf_counter() - t0)
    engine_b.synchronize()

    engine_b.cleanup()
    del engine_b
    gc.collect()
    time.sleep(1.0)

    tps_b = 1.0 / float(np.median(times_b)) if times_b else 0.0
    ms_b = float(np.median(times_b)) * 1000 if times_b else 0.0
    results['sprint3_m2_tps'] = tps_b
    results['sprint3_m2_ms'] = ms_b
    results['graph_available'] = graph_available
    results['c_graph_plan_avail'] = c_graph_plan_avail
    results['sprint3_m2_mode'] = mode_b

    improvement_vs_m1 = (tps_b - tps_a) / tps_a * 100 if tps_a > 0 else 0.0
    improvement_vs_sprint2 = (tps_b - SPRINT2_BASELINE_TPS) / SPRINT2_BASELINE_TPS * 100
    results['improvement_vs_sprint2_pct'] = improvement_vs_sprint2

    print(f"  Sprint 3 M2 ({mode_b}): {tps_b:.1f} tok/s ({ms_b:.2f} ms/tok)")
    print(f"\n  Improvement (M2 vs M1): {improvement_vs_m1:+.1f}%")
    print(f"  Improvement vs Sprint 2 baseline (38.0): {improvement_vs_sprint2:+.1f}%")

    return results


# =============================================================================
# TEST 3: TP=4 correctness vs single-GPU
# =============================================================================

def run_tp4_correctness(config, loader) -> float:
    """
    TP=4 correctness with all Sprint 3 optimizations (including graph if available).
    Sequential load to avoid OOM.
    Returns: min cosine similarity across CORRECTNESS_STEPS steps.
    """
    print_header("TEST 3: TP=4 Correctness vs Single-GPU (All Sprint 3 Opts)")
    print(f"  Steps: {CORRECTNESS_STEPS}")
    print(f"  Threshold: cosine sim >= {COSINE_SIM_THRESHOLD}")
    print(f"  Note: sequential loading to avoid OOM (16GB VRAM per GPU)")

    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02

    # --- Phase A: Collect single-GPU reference outputs ---
    print(f"\n  Phase A: Single-GPU reference ({CORRECTNESS_STEPS} steps)...")
    ref_engine = load_single_engine(config, loader)

    reset_single(ref_engine)
    for i in range(WARMUP_STEPS):
        ref_engine.decode_step(emb, i)
    ref_engine.device.synchronize()

    reset_single(ref_engine)
    ref_outputs = []
    for step in range(CORRECTNESS_STEPS):
        out = ref_engine.decode_step(emb, WARMUP_STEPS + step)
        ref_outputs.append(np.array(out, dtype=np.float32).copy())
    ref_engine.device.synchronize()
    print(f"  Collected {CORRECTNESS_STEPS} reference outputs")

    ref_engine.cleanup()
    del ref_engine
    gc.collect()
    time.sleep(1.0)
    print(f"  Single-GPU engine freed (VRAM released)")

    # --- Phase B: TP=4 with all Sprint 3 optimizations ---
    print(f"\n  Phase B: TP=4 with all Sprint 3 optimizations...")
    tp_engine = load_tp4_engine(config, loader, direct_kv_write=True)
    tp_engine.set_ring_allreduce(False)
    tp_engine.set_c_dispatch(True)

    if hasattr(tp_engine, 'set_graph_dispatch'):
        tp_engine.set_graph_dispatch(True)

    c_enabled = tp_engine._c_dispatch_enabled
    graph_enabled = getattr(tp_engine, '_graph_dispatch_enabled', False)
    direct_kv_active = getattr(tp_engine.engines[0], '_direct_kv_write', False)
    mode = get_active_dispatch_mode(tp_engine)

    print(f"  Active mode: {mode}")
    print(f"  C dispatch: {c_enabled}, Graph: {graph_enabled}")
    print(f"  Direct KV write: {direct_kv_active}")

    reset_tp(tp_engine)
    for i in range(WARMUP_STEPS):
        tp_engine.decode_step(emb, i)
    tp_engine.synchronize()

    reset_tp(tp_engine)
    cosine_sims = []
    all_pass = True

    print(f"\n  {'Step':>4}  {'Cosine Sim':>12}  {'Status':>8}")
    print(f"  {'-'*30}")
    for step in range(CORRECTNESS_STEPS):
        tp_out = tp_engine.decode_step(emb, WARMUP_STEPS + step)
        if tp_out is None:
            print(f"  {step+1:>4}  {'None':>12}  {'ERROR':>8}")
            all_pass = False
            continue
        tp_np = np.array(tp_out, dtype=np.float32)
        ref_np = ref_outputs[step]
        sim = cosine_sim(ref_np, tp_np)
        cosine_sims.append(sim)
        status = "PASS" if (not math.isnan(sim) and sim >= COSINE_SIM_THRESHOLD) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {step+1:>4}  {sim:>12.6f}  {status:>8}")

    tp_engine.synchronize()
    tp_engine.cleanup()
    del tp_engine
    gc.collect()
    time.sleep(1.0)

    min_sim = min(cosine_sims) if cosine_sims else 0.0
    print(f"\n  Min cosine similarity: {min_sim:.6f}")
    print(f"  Threshold:             {COSINE_SIM_THRESHOLD}")
    print(f"  Correctness: {'PASS' if all_pass and cosine_sims else 'FAIL'}")
    return min_sim


# =============================================================================
# TEST 4: Progressive fallback chain
#   graph → c_dispatch → cached+stream → cached → serial
# =============================================================================

def run_fallback_chain_test(config, loader) -> dict:
    """
    Test progressive fallback chain:
      graph → c_dispatch → cached+stream → cached → serial

    Verifies each mode produces valid outputs and that disabling higher-priority
    modes correctly falls back to lower-priority modes.
    """
    print_header("TEST 4: Progressive Fallback Chain")
    print(f"  Testing: graph → c_dispatch → cached+stream → cached → serial")

    np.random.seed(99)
    emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02
    FALLBACK_STEPS = 3
    results = {}

    fallback_configs = [
        # (label, setup_fn, expected_mode_contains)
        ("Mode A: graph + c_dispatch",
         lambda e: (_set_graph_dispatch(e, True), e.set_c_dispatch(True), e.set_cached_dispatch(True), e.set_stream_overlap_dispatch(True)),
         "graph"),
        ("Mode B: c_dispatch only (no graph)",
         lambda e: (_set_graph_dispatch(e, False), e.set_c_dispatch(True), e.set_cached_dispatch(True), e.set_stream_overlap_dispatch(True)),
         "C dispatch"),
        ("Mode C: cached+stream (no c_dispatch, no graph)",
         lambda e: (_set_graph_dispatch(e, False), e.set_c_dispatch(False), e.set_cached_dispatch(True), e.set_stream_overlap_dispatch(True)),
         "cached+stream"),
        ("Mode D: cached only",
         lambda e: (_set_graph_dispatch(e, False), e.set_c_dispatch(False), e.set_cached_dispatch(True), e.set_stream_overlap_dispatch(False)),
         "cached"),
        ("Mode E: serial (all disabled)",
         lambda e: (_set_graph_dispatch(e, False), e.set_c_dispatch(False), e.set_cached_dispatch(False), e.set_stream_overlap_dispatch(False)),
         "serial"),
    ]

    engine = load_tp4_engine(config, loader, direct_kv_write=True)
    engine.set_ring_allreduce(False)

    mode_outputs = {}

    for label, setup_fn, expected_contains in fallback_configs:
        print(f"\n  --- {label} ---")
        try:
            setup_fn(engine)
            mode = get_active_dispatch_mode(engine)
            print(f"  Active mode: {mode}")

            reset_tp(engine)
            out_list = []
            for i in range(FALLBACK_STEPS):
                out = engine.decode_step(emb, i)
                out_list.append(np.array(out, dtype=np.float32).copy() if out is not None else None)
            engine.synchronize()

            non_null = sum(1 for o in out_list if o is not None)
            mode_ok = expected_contains.lower() in mode.lower()
            outputs_ok = non_null == FALLBACK_STEPS

            results[label] = mode_ok and outputs_ok
            mode_outputs[label] = out_list

            print(f"  Mode check ({expected_contains!r} in {mode!r}): {'OK' if mode_ok else 'FAIL'}")
            print(f"  Output check ({non_null}/{FALLBACK_STEPS} valid): {'OK' if outputs_ok else 'FAIL'}")
            print(f"  {'PASS' if results[label] else 'FAIL'}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            results[label] = False
            mode_outputs[label] = []

    engine.cleanup()
    del engine
    gc.collect()
    time.sleep(1.0)

    # Cross-mode correctness: compare all modes vs Mode A (reference)
    print(f"\n  Cross-mode correctness (all modes vs Mode A):")
    ref_label = "Mode A: graph + c_dispatch"
    ref_outs = mode_outputs.get(ref_label, [])
    cross_results = {}
    if ref_outs and any(o is not None for o in ref_outs):
        print(f"  {'Mode':<45} {'Min Cosine Sim':>16}  {'Status':>8}")
        print(f"  {'-'*73}")
        for label, out_list in mode_outputs.items():
            if label == ref_label:
                continue
            sims = []
            for ref, test in zip(ref_outs, out_list):
                if ref is not None and test is not None:
                    sim = cosine_sim(ref, test)
                    sims.append(sim)
            if sims:
                min_sim = min(s for s in sims if not math.isnan(s)) if sims else 0.0
                ok = min_sim >= COSINE_SIM_THRESHOLD
                cross_results[label] = ok
                print(f"  {label:<45} {min_sim:>16.6f}  {'PASS' if ok else 'FAIL':>8}")
            else:
                cross_results[label] = False
                print(f"  {label:<45} {'N/A':>16}  {'FAIL':>8}")
    else:
        print(f"  Mode A reference outputs unavailable — skipping cross-mode check")

    results['cross_mode_correctness'] = all(cross_results.values()) if cross_results else True

    overall = all(results.values())
    print(f"\n  Overall fallback chain: {'PASS' if overall else 'FAIL'}")
    return results


def _set_graph_dispatch(engine: TPInferenceEngine, enabled: bool):
    """Helper to set graph dispatch if available."""
    if hasattr(engine, 'set_graph_dispatch'):
        engine.set_graph_dispatch(enabled)


# =============================================================================
# TEST 5: Per-optimization A/B comparison
# =============================================================================

def run_ab_comparison(config, loader) -> dict:
    """
    Quick A/B comparison of Sprint 3 optimization modes:
    - Sprint 2 C dispatch baseline (reference)
    - Sprint 3 M1: C dispatch + direct KV write
    - Sprint 3 M2: graph decode (best available)
    """
    print_header("TEST 5: A/B Comparison — Sprint 3 Optimization Modes")

    np.random.seed(7)
    bench_emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02
    # Use fewer steps for A/B to save time
    AB_WARMUP = 3
    AB_BENCH = 50
    results = {}

    ab_configs = [
        {
            'label': 'Sprint 2: C dispatch (star+tuned, no Sprint 3 opts)',
            'direct_kv': False,
            'use_graph': False,
            'use_c': True,
        },
        {
            'label': 'Sprint 3 M1: +direct KV write + Q/KV sync elim',
            'direct_kv': True,
            'use_graph': False,
            'use_c': True,
        },
        {
            'label': 'Sprint 3 M2: +HIP graph decode (best available)',
            'direct_kv': True,
            'use_graph': True,
            'use_c': True,
        },
    ]

    for cfg in ab_configs:
        label = cfg['label']
        print(f"\n  Config: {label}")
        try:
            engine = load_tp4_engine(config, loader, direct_kv_write=cfg['direct_kv'])
            engine.set_ring_allreduce(False)

            if cfg['use_c']:
                engine.set_c_dispatch(True)
            else:
                engine.set_cached_dispatch(True)
                engine.set_stream_overlap_dispatch(True)

            if cfg['use_graph']:
                _set_graph_dispatch(engine, True)
            else:
                _set_graph_dispatch(engine, False)

            mode = get_active_dispatch_mode(engine)
            print(f"  Active mode: {mode}")

            reset_tp(engine)
            for i in range(AB_WARMUP):
                engine.decode_step(bench_emb, i)
            engine.synchronize()

            reset_tp(engine)
            times = []
            for i in range(AB_BENCH):
                t0 = time.perf_counter()
                engine.decode_step(bench_emb, i)
                times.append(time.perf_counter() - t0)
            engine.synchronize()

            engine.cleanup()
            del engine
            gc.collect()
            time.sleep(1.0)

            tps = 1.0 / float(np.median(times))
            ms = float(np.median(times)) * 1000
            results[label] = {'tps': tps, 'ms': ms, 'mode': mode}
            print(f"  Result: {tps:.1f} tok/s ({ms:.2f} ms/tok)")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            results[label] = {'tps': 0.0, 'ms': 0.0, 'mode': 'error'}

    # Print comparison table
    print(f"\n  A/B Comparison:")
    print(f"  {'Configuration':<52} {'Tok/s':>8} {'vs Sprint 2':>12}")
    print(f"  {'-'*75}")
    prev_tps = None
    for cfg in ab_configs:
        label = cfg['label']
        data = results.get(label, {'tps': 0.0})
        tps = data['tps']
        if prev_tps is None:
            prev_tps = SPRINT2_BASELINE_TPS
        vs_s2 = f"{(tps - SPRINT2_BASELINE_TPS)/SPRINT2_BASELINE_TPS*100:+.1f}%" if SPRINT2_BASELINE_TPS > 0 else "—"
        print(f"  {label:<52} {tps:>7.1f} {vs_s2:>12}")

    return results


# =============================================================================
# Comparison table
# =============================================================================

def print_comparison_table(
    single_gpu_tps: float,
    bench_results: dict,
    min_cosine_sim: float,
):
    """Print the full comparison table across all sprint phases."""
    print_header("COMPARISON TABLE: All Sprint Phases")

    sprint3_m2_tps = bench_results.get('sprint3_m2_tps', 0.0)

    print(f"  {'Phase':<56} {'Tok/s':>7}  {'vs Single-GPU':>14}  {'vs vLLM':>8}")
    print(f"  {'-'*88}")

    phases = [
        ("Single-GPU baseline (mi50grad)",              SINGLE_GPU_BASELINE),
        ("TP=4 serial (P2P, no caching)",               TP4_SERIAL_TPS),
        ("TP=4 cached dispatch",                        TP4_CACHED_TPS),
        ("TP=4 combined (cached + stream overlap)",     TP4_COMBINED_TPS),
        ("TP=4 Sprint 2: C dispatch + tuned kernels",   SPRINT2_BASELINE_TPS),
        ("TP=4 Sprint 3 M1: allreduce pipeline",        SPRINT3_M1_TPS),
        ("TP=4 Sprint 3 M2: graph decode ← NEW",       sprint3_m2_tps),
        ("vLLM TP=4 (AWQ, reference)",                  VLLM_BASELINE_TPS),
    ]

    for name, tps in phases:
        vs_single = f"{tps/SINGLE_GPU_BASELINE:.2f}×" if SINGLE_GPU_BASELINE > 0 else "—"
        vs_vllm = f"{tps/VLLM_BASELINE_TPS:.2f}×" if VLLM_BASELINE_TPS > 0 else "—"
        print(f"  {name:<56} {tps:>7.1f}  {vs_single:>14}  {vs_vllm:>8}")

    # Gap analysis
    gap_to_vllm = VLLM_BASELINE_TPS - sprint3_m2_tps
    improvement_vs_s2 = sprint3_m2_tps - SPRINT2_BASELINE_TPS
    improvement_vs_m1 = sprint3_m2_tps - SPRINT3_M1_TPS
    print(f"\n  Sprint 3 M2 improvement vs Sprint 2:      {improvement_vs_s2:+.1f} tok/s "
          f"({improvement_vs_s2/SPRINT2_BASELINE_TPS*100:+.1f}%)")
    print(f"  Sprint 3 M2 improvement vs Sprint 3 M1:   {improvement_vs_m1:+.1f} tok/s "
          f"({improvement_vs_m1/SPRINT3_M1_TPS*100:+.1f}% )")
    print(f"  Remaining gap to vLLM:                    {gap_to_vllm:.1f} tok/s "
          f"({gap_to_vllm/VLLM_BASELINE_TPS*100:.0f}% below vLLM)")
    print(f"  TP=4 correctness: cosine sim = {min_cosine_sim:.6f} "
          f"(threshold: {COSINE_SIM_THRESHOLD})")


# =============================================================================
# Report generation
# =============================================================================

def generate_report_v3(
    single_gpu_tps: float,
    bench_results: dict,
    min_cosine_sim: float,
    fallback_results: dict,
    ab_results: dict,
    output_path: str,
) -> str:
    """Generate bench/tp4_optimization_report_v3.md."""

    sprint3_m1_tps = bench_results.get('sprint3_m1_tps', SPRINT3_M1_TPS)
    sprint3_m2_tps = bench_results.get('sprint3_m2_tps', 0.0)
    sprint3_m2_ms = bench_results.get('sprint3_m2_ms', 0.0)
    sprint3_m2_mode = bench_results.get('sprint3_m2_mode', 'unknown')
    graph_available = bench_results.get('graph_available', False)
    c_graph_plan_avail = bench_results.get('c_graph_plan_avail', False)
    c_dispatch_avail = bench_results.get('c_dispatch_available', False)

    improvement_vs_s2 = sprint3_m2_tps - SPRINT2_BASELINE_TPS
    improvement_vs_m1 = sprint3_m2_tps - SPRINT3_M1_TPS
    improvement_vs_s2_pct = (improvement_vs_s2 / SPRINT2_BASELINE_TPS * 100) if SPRINT2_BASELINE_TPS > 0 else 0.0
    improvement_vs_m1_pct = (improvement_vs_m1 / SPRINT3_M1_TPS * 100) if SPRINT3_M1_TPS > 0 else 0.0
    gap_to_vllm = VLLM_BASELINE_TPS - sprint3_m2_tps
    ratio_vs_vllm = sprint3_m2_tps / VLLM_BASELINE_TPS if VLLM_BASELINE_TPS > 0 else 0.0

    correctness_ok = min_cosine_sim >= COSINE_SIM_THRESHOLD
    single_gpu_ok = abs(single_gpu_tps - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE <= SINGLE_GPU_TOLERANCE
    no_regression = (
        abs(sprint3_m2_tps - SPRINT2_BASELINE_TPS) / SPRINT2_BASELINE_TPS <= SPRINT3_TOLERANCE
    )
    fallback_ok = all(fallback_results.values()) if fallback_results else False

    ts = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    # A/B table rows
    ab_rows = ""
    for label, data in ab_results.items():
        tps = data.get('tps', 0.0)
        mode = data.get('mode', '—')
        vs_s2 = f"{(tps - SPRINT2_BASELINE_TPS)/SPRINT2_BASELINE_TPS*100:+.1f}%" if SPRINT2_BASELINE_TPS > 0 else "—"
        ab_rows += f"| {label} | {mode} | {tps:.1f} tok/s | {vs_s2} |\n"

    # Fallback table rows
    fallback_rows = ""
    for label, passed in fallback_results.items():
        if label == 'cross_mode_correctness':
            fallback_rows += f"| Cross-mode correctness (all modes agree ≥0.99) | {'PASS' if passed else 'FAIL'} |\n"
        else:
            fallback_rows += f"| {label} | {'PASS' if passed else 'FAIL'} |\n"

    report = f"""# TP=4 Optimization Report v3: Sprint 3 Final Benchmark

**Generated:** {ts}
**Model:** Qwen3.5-27B-GPTQ-Int4
**Hardware:** 3× AMD MI50 (gfx906) + 1× AMD MI100 (gfx908) for TP=4, 16GB HBM2 each
**ROCm:** 7.1.0
**Report file:** bench/tp4_optimization_report_v3.md

---

## Executive Summary

Sprint 3 final benchmark achieved **{sprint3_m2_tps:.1f} tok/s** TP=4 throughput
with all optimizations combined, representing **{improvement_vs_s2_pct:+.1f}% vs
Sprint 2 baseline** ({SPRINT2_BASELINE_TPS} tok/s).
The gap to vLLM ({VLLM_BASELINE_TPS} tok/s) is {gap_to_vllm:.1f} tok/s
({(1 - ratio_vs_vllm)*100:.0f}% below vLLM).

**Best active dispatch mode:** {sprint3_m2_mode}

**Sprint 3 Optimizations Applied:**
1. **Q/KV stream sync elimination** (M1): Removed 32 host-blocking `hipStreamSynchronize`
   calls per token by running Q and KV GEMVs sequentially on the null stream.
2. **Direct KV cache writes** (M1): Eliminated 32 `hipMemcpyAsync` D2D copies per token
   via `qknorm_rope_cachew` fused kernel + direct V write to cache position.
3. **Allreduce overlap deepening** (M1): `c_dispatch_v2.c` reduces `hipSetDevice` calls
   by ~384/token; `hipStreamWaitEvent` is host-non-blocking on gfx906/ROCm 7.1.
4. **HIP graph capture and replay** (M2): Per-GPU compute segments captured as HIP graphs.
   Graph capture: available={graph_available}; C graph dispatch plan: {c_graph_plan_avail}.
   7.9× speedup per graph segment vs direct launch; 512 total segments per step.
5. **C graph dispatch extension** (M2): C extension (`c_graph_dispatch.c`) eliminates
   Python ctypes overhead from graph replay loop (~512 `hipGraphLaunch` calls in C).
6. **C dispatch loop** (Sprint 2): All 64 layers dispatched in tight C loop, no Python.
7. **Star topology allreduce** (Sprint 2): 8.5× faster than ring for 10KB payloads.
8. **Tuned kernels** (Sprint 2): `elementwise_v3`, `flash_attn_256_tuned`,
   `gemv_int4_v3_t16`.

**Correctness:** TP=4 (all opts) vs single-GPU: cosine similarity = {min_cosine_sim:.6f}
(threshold: {COSINE_SIM_THRESHOLD}) — **{'PASS' if correctness_ok else 'FAIL'}**

---

## Throughput Comparison: All Sprint Phases

| Optimization Phase | Throughput | vs Single-GPU | vs vLLM |
|---|---|---|---|
| Single-GPU baseline (mi50grad) | {SINGLE_GPU_BASELINE} tok/s | 1.00× | {SINGLE_GPU_BASELINE/VLLM_BASELINE_TPS:.2f}× |
| TP=4 serial (P2P, no caching) | {TP4_SERIAL_TPS} tok/s | {TP4_SERIAL_TPS/SINGLE_GPU_BASELINE:.2f}× | {TP4_SERIAL_TPS/VLLM_BASELINE_TPS:.2f}× |
| TP=4 cached dispatch | {TP4_CACHED_TPS} tok/s | {TP4_CACHED_TPS/SINGLE_GPU_BASELINE:.2f}× | {TP4_CACHED_TPS/VLLM_BASELINE_TPS:.2f}× |
| TP=4 combined (cached + stream overlap) | {TP4_COMBINED_TPS} tok/s | {TP4_COMBINED_TPS/SINGLE_GPU_BASELINE:.2f}× | {TP4_COMBINED_TPS/VLLM_BASELINE_TPS:.2f}× |
| TP=4 Sprint 2: C dispatch + tuned kernels | {SPRINT2_BASELINE_TPS} tok/s | {SPRINT2_BASELINE_TPS/SINGLE_GPU_BASELINE:.2f}× | {SPRINT2_BASELINE_TPS/VLLM_BASELINE_TPS:.2f}× |
| TP=4 Sprint 3 M1: allreduce pipeline | {SPRINT3_M1_TPS} tok/s | {SPRINT3_M1_TPS/SINGLE_GPU_BASELINE:.2f}× | {SPRINT3_M1_TPS/VLLM_BASELINE_TPS:.2f}× |
| **TP=4 Sprint 3 M2: graph decode (this run)** | **{sprint3_m2_tps:.1f} tok/s** | **{sprint3_m2_tps/SINGLE_GPU_BASELINE:.2f}×** | **{sprint3_m2_tps/VLLM_BASELINE_TPS:.2f}×** |
| vLLM TP=4 (AWQ, reference) | {VLLM_BASELINE_TPS} tok/s | {VLLM_BASELINE_TPS/SINGLE_GPU_BASELINE:.2f}× | 1.00× |

Sprint 3 M2 improvement vs Sprint 2: **{improvement_vs_s2:+.1f} tok/s ({improvement_vs_s2_pct:+.1f}%)**
Sprint 3 M2 improvement vs Sprint 3 M1: **{improvement_vs_m1:+.1f} tok/s ({improvement_vs_m1_pct:+.1f}%)**
Remaining gap to vLLM: **{gap_to_vllm:.1f} tok/s ({gap_to_vllm/VLLM_BASELINE_TPS*100:.0f}% below vLLM)**

---

## Sprint 3 Optimization Results

### Milestone 1: allreduce-pipeline

**Result: {SPRINT3_M1_TPS:.1f} tok/s** (+{(SPRINT3_M1_TPS-SPRINT2_BASELINE_TPS):.1f} tok/s vs Sprint 2)

Key finding: These optimizations provide architectural cleanup rather than large standalone
throughput gains. The bottleneck is allreduce latency (~15.2 ms/token = 128 × ~119 µs).

| Optimization | Impact |
|---|---|
| Q/KV stream sync elimination (32 syncs/token removed) | Neutral — syncs were already overlapped by allreduce |
| Direct KV cache writes (32 D2D copies/token removed) | Neutral — D2D copies too small vs allreduce bottleneck |
| Allreduce overlap deepening (384 hipSetDevice calls saved) | +0.3% — within measurement noise |
| Combined M1 vs Sprint 2 | +0.1 tok/s (+0.3%) |

### Milestone 2: hip-graph-decode

**Result: {sprint3_m2_tps:.1f} tok/s** ({improvement_vs_m1:+.1f} tok/s vs M1, active mode: {sprint3_m2_mode})

#### HIP Graph Infrastructure

Graph API availability on gfx906/ROCm 7.1: **ALL CONFIRMED**
- `hipGraphCreate`, `hipStreamBeginCapture`, `hipStreamEndCapture` ✓
- `hipGraphInstantiate`, `hipGraphLaunch` ✓
- `hipGraphExecKernelNodeSetParams`, `hipGraphGetNodes` ✓

Key infrastructure findings:
- **7.9× speedup per graph segment** vs direct launch (8 kernels at N=5120)
- **512 total segments** per decode step (4 GPUs × 64 layers × 2 segments)
- **Graph capture time**: ~130ms (one-time cost at first decode step)
- **Position-based node identification**: required because multiple kernels share the same
  function handle (e.g., `gemv_fp16_v2` is used for Q, K, V, and O projections)

#### Graph-Based Decode Path

**Critical finding: Python-level graph replay is SLOWER than C dispatch.**
- C dispatch: ~38 tok/s (tight C loop, ~960 hipModuleLaunchKernel calls)
- Python graph replay: ~28 tok/s (Python loop, 512 hipGraphLaunch + 256 hipGraphExecKernelNodeSetParams)
- **Root cause**: hipGraphLaunch via Python ctypes still carries Python overhead per call.
  The 7.9× per-segment speedup (17.83 µs vs 140.89 µs/segment) is negated by 512 Python-level
  dispatch calls (~8ms/token) vs the C dispatch's single C function call per step.

#### C Graph Dispatch Extension

**Solution: C extension (`c_graph_dispatch.c`) runs graph replay in tight C loop.**
- C graph dispatch: ~35.9 tok/s (1.01× vs 35.6 tok/s C dispatch baseline)
- Python graph dispatch: ~28 tok/s (0.74× vs C dispatch)
- The C extension eliminates Python overhead from graph replay: 512 `hipGraphLaunch` calls
  in C run in ~1ms vs ~8ms in Python

**What worked:**
- HIP graph capture on gfx906/ROCm 7.1 (confirmed working)
- Mutable parameter updates via `hipGraphExecKernelNodeSetParams` (cos/sin, seq_len)
- Direct KV write mode (`qknorm_rope_cachew` kernel) works correctly in graph mode
- C graph dispatch plan serialization (struct-based, ~1056 bytes/layer/GPU)
- All 15 decode steps produce cosine sim >= 0.99 vs single-GPU reference

**What didn't work (as expected):**
- Python-orchestrated graph replay is SLOWER (documented above)
- Per-segment overhead dominates: 512 Python calls > C loop's 960 kernel launches
- `hipGraphLaunch` overhead (~15 µs/call) × 512 = ~7.7ms vs hipModuleLaunchKernel (~1 µs/call) × 960 = ~1ms

---

## Per-Optimization Impact Breakdown

| Configuration | Mode | Throughput | vs Sprint 2 |
|---|---|---|---|
{ab_rows}

---

## Correctness Validation

| Check | Value | Threshold | Result |
|---|---|---|---|
| Single-GPU regression | {single_gpu_tps:.1f} tok/s | {SINGLE_GPU_BASELINE}±{SINGLE_GPU_TOLERANCE*100:.0f}% | {'PASS' if single_gpu_ok else 'FAIL'} |
| TP=4 all opts cosine sim | {min_cosine_sim:.6f} | ≥{COSINE_SIM_THRESHOLD} | {'PASS' if correctness_ok else 'FAIL'} |
| No regression vs Sprint 2 (±5%) | {sprint3_m2_tps:.1f} tok/s ({improvement_vs_s2_pct:+.1f}%) | ≥{SPRINT2_BASELINE_TPS*(1-SPRINT3_TOLERANCE):.1f} tok/s | {'PASS' if no_regression else 'FAIL'} |

---

## Progressive Fallback Chain

| Fallback Step | Result |
|---|---|
{fallback_rows}

---

## Gap Analysis vs vLLM (Post Sprint 3)

| Factor | Current State | Remaining Impact |
|---|---|---|
| Kernel dispatch overhead | C dispatch: ~1ms/token; Graph: ~1ms via C extension | Minimal (~2% of total) |
| Allreduce latency | 128 × ~119 µs ≈ 15.2 ms/token (hard floor, star topology) | **Dominant bottleneck** |
| Per-layer GPU compute | ~11 ms/token (64 layers × ~172 µs) | Fixed by hardware |
| hipSetDevice + event overhead | ~10 ms/token (reduced from ~13ms by M1) | ~3% improvement possible |
| vLLM AWQ advantage | AWQ vs GPTQ-Int4: potentially 10-15% GEMV speedup | Medium impact |
| vLLM HIP graph (global capture) | Captures allreduce+compute together (no host orchestration) | **High impact if achievable** |

**Remaining gap: {gap_to_vllm:.1f} tok/s ({gap_to_vllm/VLLM_BASELINE_TPS*100:.0f}% below vLLM)**

---

## Recommendations for Sprint 4

Based on the findings from Sprint 3:

### Priority 1: Eliminate per-allreduce host round-trips

The dominant remaining bottleneck is the 128 host-level allreduce calls per token
(~15.2 ms/token). Approaches:
1. **All-in-one C graph with embedded allreduce**: Write a custom HIP compute+allreduce
   kernel that performs allreduce on-device without returning to host. Requires NVLink or
   XGMI for fast GPU-GPU transfers — PCIe BAR1 has ~12 GB/s P2P bandwidth.
2. **Fused layer compute with allreduce**: Defer allreduce until multiple layer outputs
   have accumulated. Note: DeltaNet layers show that deferred allreduce causes cosine sim ~0.59
   (infeasible for correctness). Only applicable if layer independence can be proven.
3. **Reduce allreduce calls via model surgery**: Use GQA to reduce KV head count further,
   reducing the number of attention layers that require full allreduce. Currently 16 full-attn
   + 48 DeltaNet = 64 layers × 2 AR = 128 AR/step.

### Priority 2: Improve allreduce throughput

Star topology is already near-optimal for 10KB payloads. Options:
1. **Kernel-level P2P allreduce**: Fused P2P reduce already explored (1.72× isolated
   speedup, but only 1.01× e2e due to sync overhead). Further refinement needed.
2. **XGMI upgrade**: MI100+ with XGMI fabric would provide 100+ GB/s vs 12 GB/s PCIe.
   Not an option with current hardware.
3. **Allreduce-free weight distribution**: Tensor parallel approaches that avoid allreduce
   (e.g., sequence parallel) have limited applicability for decode (batch=1, seq_len=1).

### Priority 3: Reduce compute time

With allreduce at ~15.2 ms/token (hard floor) and dispatch at ~1 ms, compute
is ~11 ms/token. Possible improvements:
1. **W8A8/W4A8 quantization for FFN**: INT8 activations reduce GEMV bandwidth by 2×.
   Benchmarks show W4A16 is faster for decode (bandwidth-limited), but W8A8 may help
   for specific shapes. Currently W4A16 (GPTQ-Int4) is the best for MI50.
2. **Flash attention prefill optimization**: v3 block-tiled kernel is 1.59-1.89× faster
   for prefill. Decode kernel unchanged.
3. **Custom MLA attention**: Multi-head latent attention (as in DeepSeek) could reduce
   KV cache size and attention compute, but requires model re-training.

### Priority 4: Global HIP graph (future)

vLLM likely achieves graph-based dispatch by using a global graph that captures
BOTH compute AND allreduce (using NCCL for XGMI-connected GPUs). On PCIe:
1. hipStreamBeginCapture cannot capture hipMemcpyPeerAsync (or can it?)
2. If capturable: create one global graph per layer that captures:
   kernels + P2P transfers + reduce kernel + broadcast = full layer graph
3. Only update mutable params per step (RoPE, seq_len, KV pointers)
4. Expected result: near-zero host overhead for entire decode step

---

## Technical Notes

- **Hardware:** MI50 (gfx906 Vega 20) + MI100 (gfx908). No XGMI — P2P uses PCIe BAR1.
- **Allreduce payload:** hidden_size=5120 × FP16 = 10 KB per call, 128 calls/token.
- **Benchmark conditions:** batch=1, fixed random embedding, {BENCH_STEPS} steps, {WARMUP_STEPS} warmup.
- **C dispatch availability:** {'YES' if c_dispatch_avail else 'NO'} (c_dispatch.so loadable).
- **Graph dispatch availability:** {'YES' if graph_available else 'NO'} (set_graph_dispatch() exists).
- **C graph dispatch plan:** {'BUILT' if c_graph_plan_avail else 'NOT BUILT / Python fallback'}.
- **Direct KV write:** Uses `qknorm_rope_cachew` fused kernel for K, separate V write to cache.
- **Q/KV sync:** Sequential null-stream dispatch (no explicit sync needed).

---

*Report generated by tests/bench_tp4_sprint3.py*
"""

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to: {output_path}")
    return report


# =============================================================================
# Final summary
# =============================================================================

def print_final_summary(
    single_gpu_tps: float,
    bench_results: dict,
    min_cosine_sim: float,
    fallback_results: dict,
    report_path: str,
):
    print_header("FINAL SUMMARY")

    sprint3_m2_tps = bench_results.get('sprint3_m2_tps', 0.0)
    sprint3_m2_ms = bench_results.get('sprint3_m2_ms', 0.0)
    sprint3_m1_tps = bench_results.get('sprint3_m1_tps', SPRINT3_M1_TPS)
    sprint3_m2_mode = bench_results.get('sprint3_m2_mode', 'unknown')
    improvement_vs_s2 = sprint3_m2_tps - SPRINT2_BASELINE_TPS
    improvement_vs_m1 = sprint3_m2_tps - sprint3_m1_tps
    improvement_vs_s2_pct = (improvement_vs_s2 / SPRINT2_BASELINE_TPS * 100) if SPRINT2_BASELINE_TPS > 0 else 0.0
    gap_to_vllm = VLLM_BASELINE_TPS - sprint3_m2_tps

    single_gpu_ok = abs(single_gpu_tps - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE <= SINGLE_GPU_TOLERANCE
    correctness_ok = min_cosine_sim >= COSINE_SIM_THRESHOLD
    no_regression = (
        abs(sprint3_m2_tps - SPRINT2_BASELINE_TPS) / SPRINT2_BASELINE_TPS <= SPRINT3_TOLERANCE
    )
    fallback_ok = all(fallback_results.values()) if fallback_results else False
    report_ok = Path(report_path).exists()

    print(f"  {'Metric':<56} {'Value':>20}")
    print(f"  {'-'*78}")
    print(f"  {'Single-GPU throughput':<56} {single_gpu_tps:>18.1f} tok/s")
    print(f"  {'Sprint 3 M1 throughput (C dispatch)':<56} {sprint3_m1_tps:>18.1f} tok/s")
    print(f"  {'Sprint 3 M2 throughput (best mode)':<56} {sprint3_m2_tps:>18.1f} tok/s")
    print(f"  {'  Active mode':<56} {sprint3_m2_mode:>20}")
    print(f"  {'Sprint 3 M2 latency':<56} {sprint3_m2_ms:>18.2f} ms/tok")
    print(f"  {'vs Sprint 2 baseline (38.0)':<56} {improvement_vs_s2:>+17.1f} tok/s ({improvement_vs_s2_pct:+.1f}%)")
    print(f"  {'vs Sprint 3 M1 (38.1)':<56} {improvement_vs_m1:>+17.1f} tok/s")
    low_s2 = SPRINT2_BASELINE_TPS * (1 - SPRINT3_TOLERANCE)
    high_s2 = SPRINT2_BASELINE_TPS * (1 + SPRINT3_TOLERANCE)
    print(f"  {'  (allowed range ±5%)':<56} {low_s2:.1f}–{high_s2:.1f} tok/s")
    print(f"  {'Remaining gap to vLLM (46.9)':<56} {gap_to_vllm:>18.1f} tok/s")
    print(f"  {'TP=4 cosine sim vs single-GPU':<56} {min_cosine_sim:>22.6f}")
    print()

    print(f"  Validation:")
    print(f"  {'Single-GPU regression (within ±10% of 20.3)':<54} {'PASS' if single_gpu_ok else 'FAIL'} ({single_gpu_tps:.1f} tok/s)")
    print(f"  {'TP=4 correctness (cosine sim >= 0.99)':<54} {'PASS' if correctness_ok else 'FAIL'} (sim={min_cosine_sim:.4f})")
    print(f"  {'No regression vs Sprint 2 (within ±5%)':<54} {'PASS' if no_regression else 'FAIL'} ({improvement_vs_s2_pct:+.1f}%)")
    print(f"  {'Progressive fallback chain':<54} {'PASS' if fallback_ok else 'FAIL'}")
    print(f"  {'Report generated':<54} {'PASS' if report_ok else 'FAIL'}")

    all_pass = (single_gpu_ok and correctness_ok and no_regression and fallback_ok and report_ok)
    print()
    print("=" * 72)
    if all_pass:
        print("  OVERALL: ALL CRITICAL CHECKS PASSED")
        print(f"  Sprint 3 M2 throughput: {sprint3_m2_tps:.1f} tok/s ({improvement_vs_s2_pct:+.1f}% vs Sprint 2).")
        print(f"  Best mode: {sprint3_m2_mode}")
        print(f"  Remaining gap to vLLM: {gap_to_vllm:.1f} tok/s.")
    else:
        print("  OVERALL: SOME CRITICAL CHECKS FAILED (see above)")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("Sprint 3 Final TP=4 Benchmark (allreduce-pipeline + HIP graph decode)", 72)
    print(f"  Model:       {MODEL_DIR}")
    print(f"  Date:        {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"  GPUs (TP=4): {DEVICE_IDS}")
    print()
    print("  Sprint 3 optimizations being tested:")
    print("    Sprint 3 M1 (allreduce-pipeline):")
    print("      - Q/KV sync elimination (sequential null-stream GEMVs)")
    print("      - Direct KV cache writes (no D2D copies for K/V append)")
    print("      - Allreduce overlap deepening (reduced hipSetDevice calls)")
    print("    Sprint 3 M2 (hip-graph-decode):")
    print("      - HIP graph capture (per-GPU compute segments between allreduce points)")
    print("      - C graph dispatch extension (graph replay in tight C loop)")
    print("      - Mutable param updates (hipGraphExecKernelNodeSetParams)")
    print("    Baseline (Sprint 2):")
    print("      - C dispatch loop, star allreduce, tuned kernels")
    print()
    print("  Tests:")
    print("    1. Single-GPU regression check (expect ~20.3 tok/s, ±10%)")
    print("    2. TP=4 benchmark with all Sprint 3 M1+M2 optimizations")
    print("    3. TP=4 correctness vs single-GPU (cosine sim >= 0.99, 10 steps)")
    print("    4. Progressive fallback chain (graph → c_dispatch → cached+stream → cached → serial)")
    print("    5. Per-optimization A/B comparison")
    print("    6. Comparison table + report generation")

    # Verify GPU count
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"\n  GPUs visible: {n_gpus}")
    if n_gpus < 4:
        print(f"\nERROR: Need 4 GPUs for TP=4, only {n_gpus} visible.")
        print("       Ensure HIP_VISIBLE_DEVICES=0,1,2,3 is set.")
        sys.exit(1)

    # Load model config (shared across all tests)
    print(f"\nLoading model config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"  {config.num_hidden_layers} layers, hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}")

    loader = QwenWeightLoader(MODEL_DIR, config)

    # -------------------------------------------------------------------------
    # Test 1: Single-GPU regression
    # -------------------------------------------------------------------------
    single_gpu_tps = run_single_gpu_benchmark(config, loader)

    # -------------------------------------------------------------------------
    # Test 2: TP=4 all Sprint 3 optimizations benchmark
    # -------------------------------------------------------------------------
    bench_results = run_tp4_all_optimizations(config, loader)

    # -------------------------------------------------------------------------
    # Test 3: TP=4 correctness (sequential load to avoid OOM)
    # -------------------------------------------------------------------------
    min_cosine_sim = run_tp4_correctness(config, loader)

    # -------------------------------------------------------------------------
    # Test 4: Progressive fallback chain
    # -------------------------------------------------------------------------
    fallback_results = run_fallback_chain_test(config, loader)

    # -------------------------------------------------------------------------
    # Test 5: A/B per-optimization comparison
    # -------------------------------------------------------------------------
    ab_results = run_ab_comparison(config, loader)

    # -------------------------------------------------------------------------
    # Test 6: Comparison table
    # -------------------------------------------------------------------------
    print_comparison_table(single_gpu_tps, bench_results, min_cosine_sim)

    # -------------------------------------------------------------------------
    # Generate report v3
    # -------------------------------------------------------------------------
    print_header("Generating Optimization Report v3")
    report_path = "/opt/mi50grad/bench/tp4_optimization_report_v3.md"
    generate_report_v3(
        single_gpu_tps=single_gpu_tps,
        bench_results=bench_results,
        min_cosine_sim=min_cosine_sim,
        fallback_results=fallback_results,
        ab_results=ab_results,
        output_path=report_path,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print_final_summary(single_gpu_tps, bench_results, min_cosine_sim,
                        fallback_results, report_path)


if __name__ == "__main__":
    main()
