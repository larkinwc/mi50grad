#!/usr/bin/env python3
"""
Final TP=4 benchmark: comprehensive comparison of all optimization phases.

Runs the best available mode (combined: cached dispatch + stream overlap) and
generates a comparison report in bench/tp4_optimization_report.md.

Phase baselines (from prior milestone benchmarks):
  - Single-GPU baseline:             20.3 tok/s
  - TP=4 serial baseline:            12.4 tok/s (P2P allreduce, no cached dispatch)
  - TP=4 cached dispatch:            23.7 tok/s
  - TP=4 combined (cached + stream): ~33.5 tok/s
  - TP=4 fused P2P + combined:       ~33.6 tok/s
  - vLLM TP=4 baseline:              46.9 tok/s

Steps:
  1. Single-GPU regression check (expect ~20 tok/s)
  2. TP=4 combined mode benchmark (best available mode)
  3. TP=4 correctness check (cosine sim > 0.99 vs single-GPU reference)
  4. Generate bench/tp4_optimization_report.md

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4_final.py'
"""

import sys
import time
import os
import math
import numpy as np
from pathlib import Path

# Force unbuffered stdout for real-time output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
WARMUP_STEPS = 3
BENCH_STEPS = 100
CORRECTNESS_STEPS = 10
COSINE_SIM_THRESHOLD = 0.99

# Known baselines from prior milestone measurements
SINGLE_GPU_BASELINE = 20.3   # tok/s
TP4_SERIAL_BASELINE = 12.4   # tok/s (P2P allreduce, no cached dispatch)
TP4_CACHED_BASELINE = 23.7   # tok/s
TP4_COMBINED_BASELINE = 33.5  # tok/s (approximate)
TP4_FUSED_COMBINED_BASELINE = 33.6  # tok/s (approximate)
VLLM_BASELINE = 46.9          # tok/s


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


def print_header(title: str):
    width = 70
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def run_single_gpu_benchmark(config) -> float:
    """Run single-GPU decode benchmark. Returns tok/s."""
    print_header("STEP 1: Single-GPU Regression Check")
    print(f"Model: {MODEL_DIR}")
    print(f"Device: GPU 0")
    print(f"Steps: {WARMUP_STEPS} warmup + {BENCH_STEPS} timed")
    print()

    engine = InferenceEngine(config, device_id=0)
    loader = QwenWeightLoader(MODEL_DIR, config)

    print("Loading weights onto single GPU...")
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print("Weights loaded.")

    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # Warmup
    print(f"\nRunning {WARMUP_STEPS} warmup steps...")
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.device.synchronize()

    # Benchmark
    print(f"Running {BENCH_STEPS} timed steps...")
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()

    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        engine.decode_step(emb, i)
    engine.device.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = BENCH_STEPS / elapsed
    ms_per_tok = elapsed / BENCH_STEPS * 1000

    print(f"\nSingle-GPU Result:")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"  Latency:    {ms_per_tok:.1f} ms/tok")

    # Regression check
    deviation_pct = abs(tok_per_sec - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE * 100
    if deviation_pct <= 10.0:
        print(f"  Regression check: PASS ({deviation_pct:.1f}% deviation from {SINGLE_GPU_BASELINE} tok/s baseline)")
    else:
        print(f"  Regression check: WARN ({deviation_pct:.1f}% deviation from {SINGLE_GPU_BASELINE} tok/s baseline)")

    engine.cleanup()
    return tok_per_sec


def run_tp4_correctness_check(config) -> float:
    """Check TP=4 correctness vs single-GPU reference.

    To avoid OOM (loading both models simultaneously uses too much VRAM),
    this runs in two sequential phases:
    1. Load single-GPU engine, run CORRECTNESS_STEPS, save outputs, cleanup
    2. Load TP=4 engine, run same CORRECTNESS_STEPS, compare outputs

    Returns min cosine similarity across all steps.
    """
    print_header("STEP 2: TP=4 Correctness Check (vs Single-GPU Reference)")
    print(f"Steps: {CORRECTNESS_STEPS} decode steps")
    print(f"Threshold: cosine sim > {COSINE_SIM_THRESHOLD}")
    print(f"Note: Running single-GPU first, then TP=4 (sequential to avoid OOM)")
    print()

    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # --- Phase A: Single-GPU reference outputs ---
    print("Phase A: Loading single-GPU reference engine...")
    ref_engine = InferenceEngine(config, device_id=0)
    ref_loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading ref layer {i}/{config.num_hidden_layers}...")
        ref_engine.load_layer_weights(i, ref_loader.load_layer(i))
    ref_engine.load_final_norm(ref_loader.load_final_norm())
    ref_engine.load_lm_head(ref_loader.load_lm_head())
    print("Reference engine loaded.")

    print(f"\nRunning {CORRECTNESS_STEPS} reference decode steps...")
    ref_outputs = []
    for step in range(CORRECTNESS_STEPS):
        out = ref_engine.decode_step(emb, step)
        ref_outputs.append(np.array(out, dtype=np.float32).copy())
    ref_engine.device.synchronize()
    print(f"Reference outputs collected ({len(ref_outputs)} steps).")

    ref_engine.cleanup()
    del ref_engine
    print("Reference engine cleaned up (VRAM freed).")

    # --- Phase B: TP=4 outputs ---
    print(f"\nPhase B: Loading TP=4 engine on GPUs {DEVICE_IDS}...")
    tp_engine = TPInferenceEngine(config, DEVICE_IDS)
    tp_loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading TP layer {i}/{config.num_hidden_layers}...")
        tp_engine.load_layer_weights(i, tp_loader.load_layer(i))
    tp_engine.load_final_norm(tp_loader.load_final_norm())
    tp_engine.load_lm_head(tp_loader.load_lm_head())
    print("TP=4 engine loaded.")

    # Enable best mode (combined: cached dispatch + stream overlap)
    tp_engine.build_dispatch_cache()
    tp_engine.set_cached_dispatch(True)
    tp_engine.set_stream_overlap_dispatch(True)

    print(f"\nRunning {CORRECTNESS_STEPS} TP=4 decode steps (combined mode)...")
    cosine_sims = []
    all_pass = True

    for step in range(CORRECTNESS_STEPS):
        tp_output = tp_engine.decode_step(emb, step)
        if tp_output is None:
            print(f"  Step {step:2d}: ERROR - None output from TP=4")
            all_pass = False
            continue

        tp_np = np.array(tp_output, dtype=np.float32)
        ref_np = ref_outputs[step]
        sim = cosine_similarity(ref_np, tp_np)
        cosine_sims.append(sim)
        status = "PASS" if sim >= COSINE_SIM_THRESHOLD else "FAIL"
        if sim < COSINE_SIM_THRESHOLD:
            all_pass = False
        print(f"  Step {step:2d}: cosine sim = {sim:.6f}  [{status}]")

    tp_engine.synchronize()
    min_sim = min(cosine_sims) if cosine_sims else 0.0
    print(f"\nCorrectness Summary:")
    print(f"  Steps tested:    {len(cosine_sims)}")
    print(f"  Min cosine sim:  {min_sim:.6f}")
    print(f"  Threshold:       {COSINE_SIM_THRESHOLD}")
    print(f"  Result:          {'PASS' if all_pass else 'FAIL'}")

    tp_engine.cleanup()

    return min_sim


def run_tp4_benchmark(config) -> tuple:
    """Run TP=4 combined mode benchmark. Returns (tok_per_sec, ms_per_tok)."""
    print_header("STEP 3: TP=4 Final Benchmark (Combined Mode)")
    print(f"Model: {MODEL_DIR}")
    print(f"GPUs: {DEVICE_IDS}")
    print(f"Mode: cached dispatch + stream overlap (best available)")
    print(f"Steps: {WARMUP_STEPS} warmup + {BENCH_STEPS} timed")
    print()

    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs visible: {n_gpus}")
    if n_gpus < 4:
        print(f"ERROR: Need 4 GPUs, only {n_gpus} visible.")
        print("Make sure to use: -e HIP_VISIBLE_DEVICES=0,1,2,3")
        sys.exit(1)

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

    # Enable best mode: cached dispatch + stream overlap
    engine.set_cached_dispatch(True)
    engine.set_stream_overlap_dispatch(True)

    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # Reset state
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()

    # Warmup
    print(f"\nRunning {WARMUP_STEPS} warmup steps...")
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.synchronize()

    # Reset state for timed run
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()

    # Timed benchmark
    print(f"Running {BENCH_STEPS} timed steps...")
    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        engine.decode_step(emb, WARMUP_STEPS + i)
    engine.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = BENCH_STEPS / elapsed
    ms_per_tok = elapsed / BENCH_STEPS * 1000

    print(f"\nTP=4 Combined Mode Result:")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"  Latency:    {ms_per_tok:.1f} ms/tok")
    print(f"  Elapsed:    {elapsed:.2f}s ({BENCH_STEPS} steps)")

    engine.cleanup()
    return tok_per_sec, ms_per_tok


def generate_report(single_gpu_tps: float, tp4_combined_tps: float,
                    tp4_combined_ms: float, min_cosine_sim: float,
                    output_path: str):
    """Generate the tp4_optimization_report.md comparison report."""

    speedup_vs_single = tp4_combined_tps / SINGLE_GPU_BASELINE
    speedup_vs_serial = tp4_combined_tps / TP4_SERIAL_BASELINE
    speedup_vs_vllm_ratio = tp4_combined_tps / VLLM_BASELINE

    # Per-optimization improvements
    cached_vs_serial_pct = (TP4_CACHED_BASELINE - TP4_SERIAL_BASELINE) / TP4_SERIAL_BASELINE * 100
    combined_vs_cached_pct = (TP4_COMBINED_BASELINE - TP4_CACHED_BASELINE) / TP4_CACHED_BASELINE * 100
    fused_vs_combined_pct = (TP4_FUSED_COMBINED_BASELINE - TP4_COMBINED_BASELINE) / TP4_COMBINED_BASELINE * 100
    measured_vs_cached_pct = (tp4_combined_tps - TP4_CACHED_BASELINE) / TP4_CACHED_BASELINE * 100
    measured_vs_serial_pct = (tp4_combined_tps - TP4_SERIAL_BASELINE) / TP4_SERIAL_BASELINE * 100

    vllm_gap_tps = VLLM_BASELINE - tp4_combined_tps
    vllm_gap_pct = vllm_gap_tps / VLLM_BASELINE * 100

    theoretical_tp4_ceiling = SINGLE_GPU_BASELINE * 4

    report = f"""# TP=4 Optimization Report: Final Benchmark Comparison

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Model:** Qwen3.5-27B-GPTQ-Int4  
**Hardware:** 4× AMD MI50 (gfx906, Vega 20), PCIe x16 Gen4, 32GB HBM2 each  
**ROCm:** 7.1.0  

---

## Executive Summary

mi50grad's TP=4 inference achieves **{tp4_combined_tps:.1f} tok/s** with the combined
cached dispatch + async stream overlap optimization, representing a **{speedup_vs_single:.2f}×
speedup over single-GPU** decode ({SINGLE_GPU_BASELINE} tok/s). The gap to vLLM's
{VLLM_BASELINE} tok/s ({vllm_gap_tps:.1f} tok/s, {vllm_gap_pct:.0f}% faster) is attributable
to optimizations not yet implemented in mi50grad (CUDA graphs, torch.compile, chunked prefill,
optimized attention kernels).

**Correctness:** TP=4 final output vs single-GPU reference: cosine similarity = {min_cosine_sim:.6f} (threshold: {COSINE_SIM_THRESHOLD}) — **{'PASS' if min_cosine_sim >= COSINE_SIM_THRESHOLD else 'FAIL'}**

---

## Throughput Comparison

| Optimization Phase | Throughput | vs Prior Phase | vs Single-GPU | vs vLLM |
|---|---|---|---|---|
| Single-GPU baseline (mi50grad) | {SINGLE_GPU_BASELINE} tok/s | — | 1.00× | {SINGLE_GPU_BASELINE/VLLM_BASELINE:.2f}× |
| TP=4 serial (P2P allreduce, no caching) | {TP4_SERIAL_BASELINE} tok/s | — | {TP4_SERIAL_BASELINE/SINGLE_GPU_BASELINE:.2f}× | {TP4_SERIAL_BASELINE/VLLM_BASELINE:.2f}× |
| TP=4 cached dispatch | {TP4_CACHED_BASELINE} tok/s | +{cached_vs_serial_pct:.0f}% | {TP4_CACHED_BASELINE/SINGLE_GPU_BASELINE:.2f}× | {TP4_CACHED_BASELINE/VLLM_BASELINE:.2f}× |
| TP=4 combined (cached + stream overlap) | ~{TP4_COMBINED_BASELINE} tok/s | +{combined_vs_cached_pct:.0f}% | {TP4_COMBINED_BASELINE/SINGLE_GPU_BASELINE:.2f}× | {TP4_COMBINED_BASELINE/VLLM_BASELINE:.2f}× |
| TP=4 fused P2P + combined | ~{TP4_FUSED_COMBINED_BASELINE} tok/s | +{fused_vs_combined_pct:.1f}% | {TP4_FUSED_COMBINED_BASELINE/SINGLE_GPU_BASELINE:.2f}× | {TP4_FUSED_COMBINED_BASELINE/VLLM_BASELINE:.2f}× |
| **TP=4 measured (this run)** | **{tp4_combined_tps:.1f} tok/s** | — | **{speedup_vs_single:.2f}×** | **{speedup_vs_vllm_ratio:.2f}×** |
| vLLM TP=4 (AWQ, reference) | {VLLM_BASELINE} tok/s | — | {VLLM_BASELINE/SINGLE_GPU_BASELINE:.2f}× | 1.00× |
| Theoretical TP=4 ceiling | ~{theoretical_tp4_ceiling:.0f} tok/s | — | 4.00× | {theoretical_tp4_ceiling/VLLM_BASELINE:.2f}× |

---

## Per-Optimization Breakdown

### Phase 0 → Phase 1: GPU P2P Allreduce

| Metric | Before | After | Improvement |
|---|---|---|---|
| Allreduce mechanism | CPU-mediated (9× hipMemcpy) | GPU P2P (hipMemcpyPeerAsync + reduce kernel) | — |
| Allreduce latency | ~187 µs/call | ~122 µs/call | 1.53× faster |
| TP=4 throughput | ~11 tok/s (CPU allreduce) | 12.4 tok/s | ~1.13× |
| Allreduce share of step time | ~6–13 ms (est.) | 23.5 ms/tok | 29% of step |

**Key insight:** GPU P2P allreduce eliminated the host-roundtrip overhead (9 synchronous PCIe 
memcpy calls per allreduce). The 23.5 ms/tok allreduce time in the P2P baseline reflects that 
128 allreduces × 122 µs/call ≈ 15.6 ms of pure allreduce, plus synchronization overhead.

### Phase 1 → Phase 2: Cached Kernel Dispatch

| Metric | Before | After | Improvement |
|---|---|---|---|
| TP=4 throughput | 12.4 tok/s | {TP4_CACHED_BASELINE} tok/s | **+{cached_vs_serial_pct:.0f}%** |
| ms/tok | 80.5 ms | 42.2 ms | 1.91× |
| Python dispatch overhead | ~44 ms/tok | ~14 ms/tok | 3.1× reduction |

**Key insight:** The primary bottleneck was not GPU compute or allreduce — it was Python ctypes 
parameter construction (~640 launches × 8–10 µs/launch = 5–6 ms Python overhead × 8 per token ≈ 
44 ms/tok). Pre-caching the ctypes parameter arrays at engine init eliminated this overhead, 
reducing Python dispatch from ~44 ms to ~14 ms per token.

**Note:** Python threading for dispatch was evaluated and found COUNTER-PRODUCTIVE (+490 µs/round 
Python threading overhead × 128 rounds = 63 ms penalty — 2.3× slower than serial dispatch).

### Phase 2 → Phase 3: Async Stream Overlap

| Metric | Before | After | Improvement |
|---|---|---|---|
| TP=4 throughput (combined) | {TP4_CACHED_BASELINE} tok/s | ~{TP4_COMBINED_BASELINE} tok/s | **+{combined_vs_cached_pct:.0f}%** |
| ms/tok | 42.2 ms | ~29.9 ms | 1.41× |
| Allreduce overlap | Sequential (CPU blocks) | Async (GPU-side event ordering) | — |

**Key insight:** Cached dispatch reduced Python overhead to ~14 ms/tok. Allreduce 
(128 × 122 µs ≈ 15.6 ms base) remained partially visible. With async stream overlap, 
allreduce on a dedicated HIP stream runs concurrently with Python dispatching the next 
layer's kernels. Since Python dispatch takes ~14 ms and allreduce takes ~15.6 ms, the 
overlap hides most of the allreduce latency behind Python dispatch time, achieving 
29–35 ms/tok.

### Phase 3: Fused P2P GEMV Epilogue

| Metric | Before | After | Improvement |
|---|---|---|---|
| Raw allreduce latency | 101.7 µs/call | 59.3 µs/call | **1.72× faster** |
| TP=4 serial throughput | 13.2 tok/s | 14.4 tok/s | +9% |
| TP=4 combined throughput | {TP4_COMBINED_BASELINE} tok/s | ~{TP4_FUSED_COMBINED_BASELINE} tok/s | +{fused_vs_combined_pct:.1f}% |

**Key insight:** The fused P2P GEMV kernel (where all 4 GPUs simultaneously read peer partials 
via BAR1 P2P pointers in a single kernel, eliminating sequential gather→reduce→broadcast) reduces 
raw allreduce latency 1.72× in isolation. However, in combined mode, the async fused allreduce 
requires all 4 GPUs to wait on all 4 compute events (increasing synchronization overhead), 
offsetting the raw latency benefit. The fused kernel is best for serial decode paths (+9%).

### Phase 4: Deferred DeltaNet Allreduce (INFEASIBLE)

**Finding:** The proposed optimization of combining attention+FFN allreduces for DeltaNet 
layers (48 of 64 layers) to reduce allreduce count from 128 to 80 per token was found 
**numerically infeasible**.

| Metric | Value |
|---|---|
| Proposed allreduce count | 80/step (37.5% reduction from 128) |
| Cosine similarity | 0.59 (far below 0.99 threshold) |
| Result | INFEASIBLE |

**Reason:** The attention residual contribution is significant — skipping the intermediate 
hidden state update causes the pre-FFN RMSNorm to see a fundamentally different input, 
leading to catastrophic output divergence across 48 DeltaNet layers. This is a fundamental 
property of the Qwen3.5 27B architecture.

---

## Gap Analysis: mi50grad vs vLLM

| Factor | vLLM Advantage | Estimated Impact |
|---|---|---|
| CUDA graphs | Eliminates Python dispatch overhead entirely (~0 ms dispatch) | ~14 ms/tok → ~0 ms |
| torch.compile | Kernel fusion, optimized memory layout, kernel auto-tuning | 10–20% |
| Chunked prefill | Better GPU utilization, faster KV cache warming | Prefill-focused |
| Optimized attention | FlashAttention-2/3, hardware-specific tuning for decode | 10–15% |
| INT8/FP8 activations | Reduced allreduce payload, faster GEMV | 5–10% |
| Continuous batching | Improved GPU utilization across requests | Multi-request |

**Primary gap:** With Python dispatch overhead down to ~14 ms/tok (from 44 ms) but still 
significant, the next major optimization is eliminating Python dispatch entirely — either via 
CUDA/HIP graph capture or a compiled C loop dispatching all 64 layers' kernels without 
returning to Python. This would reduce step latency from ~30 ms toward the theoretical 
minimum of ~15–18 ms (allreduce + GPU compute time).

**Estimated potential:** Eliminating Python dispatch overhead entirely could push throughput 
toward 50–60 tok/s, approaching or exceeding vLLM's {VLLM_BASELINE} tok/s.

---

## What's Left on the Table

| Optimization | Estimated Gain | Complexity |
|---|---|---|
| HIP graph capture (eliminate Python dispatch entirely) | ~1.5–2× vs combined | High |
| Ring allreduce (eliminate GPU0 bottleneck) | ~10–20% | Medium |
| Fused attention+FFN INT8 quantization (reduce allreduce payload) | ~10% | Medium |
| Better kernel auto-tuning (tile sizes, occupancy) | 5–10% | Low |
| Multi-request batching (amortize fixed overheads) | Large at scale | High |

---

## Correctness Validation

| Check | Value | Threshold | Result |
|---|---|---|---|
| Single-GPU regression | {single_gpu_tps:.1f} tok/s | ±10% of {SINGLE_GPU_BASELINE} tok/s | {'PASS' if abs(single_gpu_tps - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE <= 0.10 else 'WARN'} |
| TP=4 vs single-GPU cosine sim | {min_cosine_sim:.6f} | >{COSINE_SIM_THRESHOLD} | {'PASS' if min_cosine_sim >= COSINE_SIM_THRESHOLD else 'FAIL'} |

---

## Technical Notes

- **Hardware:** MI50 uses gfx906 (Vega 20). No XGMI fabric — P2P uses BAR1 PCIe aperture.
  All GPU pairs are 2 PCIe hops apart, limiting P2P bandwidth vs NVLink/XGMI.
- **vLLM comparison:** vLLM uses AWQ quantization (possibly higher throughput than GPTQ-Int4 
  due to different kernel tuning). The comparison is directionally valid but not perfectly 
  apples-to-apples.
- **Benchmark conditions:** Single decode request (batch=1), fixed random embedding input,
  100 steps, 3 warmup steps. Real inference with variable-length prompts and KV cache 
  growth would show different characteristics.
- **DeltaNet layers:** 48 of 64 layers use DeltaNet linear attention (no quadratic attention), 
  which uses a state matrix updated recurrently. The 16 full GQA layers use standard 
  FlashAttention decode.

---

*Report generated by tests/bench_tp4_final.py*
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to: {output_path}")
    return report


def main():
    print_header("FINAL TP=4 BENCHMARK: Comprehensive Optimization Comparison")
    print(f"Model:    {MODEL_DIR}")
    print(f"Date:     {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()
    print("This benchmark:")
    print("  1. Runs single-GPU regression check (~20 tok/s expected)")
    print("  2. Runs TP=4 correctness check (cosine sim > 0.99 vs single-GPU)")
    print("  3. Runs TP=4 combined mode benchmark (best available mode)")
    print("  4. Generates bench/tp4_optimization_report.md")

    # Load config once
    print(f"\nLoading config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"Config: {config.num_hidden_layers} layers, hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}")
    print(f"TP=4 sharding: "
          f"{config.num_attention_heads}→{config.num_attention_heads // 4} attn heads/GPU, "
          f"{config.intermediate_size}→{config.intermediate_size // 4} FFN intermediate/GPU")

    # -------------------------
    # Step 1: Single-GPU check
    # -------------------------
    single_gpu_tps = run_single_gpu_benchmark(config)

    # -------------------------
    # Step 2: TP=4 correctness
    # -------------------------
    min_cosine_sim = run_tp4_correctness_check(config)

    # -------------------------
    # Step 3: TP=4 combined benchmark
    # -------------------------
    tp4_combined_tps, tp4_combined_ms = run_tp4_benchmark(config)

    # -------------------------
    # Step 4: Generate report
    # -------------------------
    print_header("STEP 4: Generating Comparison Report")
    report_path = "/opt/mi50grad/bench/tp4_optimization_report.md"
    generate_report(
        single_gpu_tps=single_gpu_tps,
        tp4_combined_tps=tp4_combined_tps,
        tp4_combined_ms=tp4_combined_ms,
        min_cosine_sim=min_cosine_sim,
        output_path=report_path,
    )

    # -------------------------
    # Final summary
    # -------------------------
    print_header("FINAL SUMMARY")
    speedup_vs_single = tp4_combined_tps / SINGLE_GPU_BASELINE
    speedup_vs_vllm = tp4_combined_tps / VLLM_BASELINE
    gap_to_vllm = VLLM_BASELINE - tp4_combined_tps

    print(f"{'Metric':<40} {'Value':>20}")
    print("-" * 62)
    print(f"{'Single-GPU regression':<40} {single_gpu_tps:>18.1f} tok/s")
    print(f"{'TP=4 combined throughput':<40} {tp4_combined_tps:>18.1f} tok/s")
    print(f"{'TP=4 combined latency':<40} {tp4_combined_ms:>18.1f} ms/tok")
    print(f"{'Speedup vs single-GPU':<40} {speedup_vs_single:>19.2f}x")
    print(f"{'vLLM TP=4 baseline':<40} {VLLM_BASELINE:>18.1f} tok/s")
    print(f"{'Gap to vLLM':<40} {gap_to_vllm:>18.1f} tok/s")
    print(f"{'Ratio vs vLLM':<40} {speedup_vs_vllm:>19.2f}x")
    print(f"{'Cosine sim (TP=4 vs single-GPU)':<40} {min_cosine_sim:>20.6f}")
    print()

    # VAL assertions
    print("Validation Assertions:")
    single_gpu_regression_ok = abs(single_gpu_tps - SINGLE_GPU_BASELINE) / SINGLE_GPU_BASELINE <= 0.10
    correctness_ok = min_cosine_sim >= COSINE_SIM_THRESHOLD
    throughput_ok = tp4_combined_tps > SINGLE_GPU_BASELINE  # significantly higher
    report_generated = Path(report_path).exists()

    print(f"  VAL-FINAL-001 (TP=4 > single-GPU):         {'PASS' if throughput_ok else 'FAIL'}  "
          f"({tp4_combined_tps:.1f} tok/s > {SINGLE_GPU_BASELINE} tok/s)")
    print(f"  VAL-FINAL-002 (vLLM comparison reported):  {'PASS' if report_generated else 'FAIL'}  "
          f"(report generated with gap analysis)")
    print(f"  VAL-FINAL-003 (single-GPU regression):     {'PASS' if single_gpu_regression_ok else 'WARN'}  "
          f"({single_gpu_tps:.1f} tok/s, {abs(single_gpu_tps - SINGLE_GPU_BASELINE)/SINGLE_GPU_BASELINE*100:.1f}% deviation)")
    print(f"  VAL-FINAL-004 (report generated):          {'PASS' if report_generated else 'FAIL'}  "
          f"({report_path})")
    print(f"  CORRECTNESS   (cosine sim > {COSINE_SIM_THRESHOLD}):      {'PASS' if correctness_ok else 'FAIL'}  "
          f"(sim={min_cosine_sim:.6f})")
    print()

    all_pass = throughput_ok and correctness_ok and report_generated
    if all_pass:
        print("OVERALL RESULT: PASS — All validation assertions satisfied.")
    else:
        print("OVERALL RESULT: PARTIAL — Some assertions may not be met (see above).")

    print("=" * 70)


if __name__ == "__main__":
    main()
