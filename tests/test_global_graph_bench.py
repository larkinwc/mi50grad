#!/usr/bin/env python3
"""
tests/test_global_graph_bench.py — 100-step TP=4 throughput benchmark for global graph dispatch.

Tests:
  1. Throughput benchmark: 100 steps each for:
       - Global graph dispatch (highest priority)
       - C dispatch + kernel P2P allreduce (baseline, ~21.1 tok/s)
       - Cached+stream (fallback, ~15 tok/s)
  2. Fallback chain correctness:
       - global graph → C dispatch → cached+stream
       - Each mode: cosine sim >= 0.99 vs single-GPU reference
  3. set_global_graph_dispatch() toggle test

Validation assertions:
  VAL-GGC-001: Full-layer graph capture succeeds
  VAL-GGC-002: Graph decode correctness (cosine_sim >= 0.99)
  VAL-GGC-003: Mutable parameter updates work correctly
  VAL-GGC-004: E2E throughput measurement
  VAL-GGC-005: C graph dispatch integration
  VAL-GGC-006: Graph fallback chain
  VAL-CROSS-002: Progressive fallback chain

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_global_graph_bench.py'
"""

import sys
import os
import time
import math
import subprocess
import json
import numpy as np
from pathlib import Path

# Force unbuffered stdout for real-time output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
DEVICE_ID_SINGLE = 0

BENCH_STEPS = 100
WARMUP_STEPS = 3
CORRECTNESS_STEPS = 10
COSINE_SIM_THRESHOLD = 0.99
MAX_SEQ_LEN = 256

# Baselines from kernel-p2p-tp4-integration milestone
C_DISPATCH_BASELINE_TPS = 21.1    # tok/s (4x MI50, kernel P2P + C dispatch)
CACHED_STREAM_BASELINE_TPS = 15.3 # tok/s (4x MI50, star allreduce, cached+stream)

# ============================================================================
# Utilities
# ============================================================================

def print_header(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
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


results = {}  # test_name → bool
tps_values = {}  # label → tok/s


def record(name: str, passed: bool, msg: str = ""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    suffix = f" — {msg}" if msg else ""
    print(f"  [{status}] {name}{suffix}")


# ============================================================================
# Core benchmark subprocess: runs all modes in one engine load
# ============================================================================

def run_all_modes_subprocess(bench_steps: int, correctness_steps: int,
                              ref_seed: int = 42) -> dict:
    """Run all dispatch modes in one subprocess to avoid repeated engine loads.

    Returns dict with keys: capture_ok, c_plan_ok, sims_global, sims_c_dispatch,
    sims_cached_stream, tps_global, tps_c_dispatch, tps_cached_stream,
    toggle_ok, c_plan_built.
    """
    script = f"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
import time
import json
from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.tp_engine import TPInferenceEngine

MODEL_DIR = '{MODEL_DIR}'
DEVICE_IDS = {DEVICE_IDS}
MAX_SEQ_LEN = {MAX_SEQ_LEN}
BENCH_STEPS = {bench_steps}
WARMUP_STEPS = {WARMUP_STEPS}
CORRECTNESS_STEPS = {correctness_steps}
REF_SEED = {ref_seed}

# ---- Collect single-GPU reference (subprocess within subprocess) ----
import subprocess
ref_script = f'''
import sys
sys.path.insert(0, "/opt/mi50grad")
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.model.weight_loader import QwenWeightLoader
import json

config = load_config_from_json("{MODEL_DIR}")
loader = QwenWeightLoader("{MODEL_DIR}", config)
engine = InferenceEngine(config, device_id={DEVICE_ID_SINGLE}, max_seq_len={MAX_SEQ_LEN})
for i in range(config.num_hidden_layers):
    engine.load_layer_weights(i, loader.load_layer(i))
engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())

rng = np.random.default_rng({ref_seed})
outputs = []
for step in range({correctness_steps}):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    outputs.append(out.tolist())
engine.cleanup()
del engine
print("OUTPUTS:" + json.dumps(outputs))
'''
ref_result = subprocess.run([sys.executable, '-c', ref_script],
                            capture_output=True, text=True, timeout=600)
ref_outputs = []
if ref_result.returncode == 0:
    for line in ref_result.stdout.split('\\n'):
        if line.startswith('OUTPUTS:'):
            ref_outputs = [np.array(x, dtype=np.float16) for x in json.loads(line[8:])]
            break
print(f"REF_OK={{len(ref_outputs)}}")

# ---- Load ONE TP=4 engine (reused across all test modes) ----
print("Loading TP=4 engine...")
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)
engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)

for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print(f"  Layer {{layer_idx}}...")
    engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())

engine.build_dispatch_cache()
engine.set_direct_kv_write(True)

def reset_tp():
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()

def cosine_sim(a, b):
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    dot = float(np.dot(a32, b32))
    na = float(np.linalg.norm(a32))
    nb = float(np.linalg.norm(b32))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0

# =========================================================
# MODE 1: Cached + stream overlap (Sprint 3 fallback)
# =========================================================
print("\\n=== MODE: cached+stream ===")
engine.set_global_graph_dispatch(False)
engine.set_c_dispatch(False)
engine.set_cached_dispatch(True)
engine.set_stream_overlap_dispatch(True)

rng = np.random.default_rng(REF_SEED)
sims_cs = []
for step in range(CORRECTNESS_STEPS):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    if step < len(ref_outputs):
        cs = cosine_sim(out, ref_outputs[step])
        sims_cs.append(cs)
        print(f"  cs_step{{step}}: {{cs:.6f}}")

reset_tp()
for i in range(WARMUP_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
reset_tp()

t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
t1 = time.perf_counter()
tps_cs = BENCH_STEPS / (t1 - t0)
print(f"TPS_CACHED_STREAM={{tps_cs:.2f}}")
reset_tp()

# =========================================================
# MODE 2: C dispatch + kernel P2P allreduce
# =========================================================
print("\\n=== MODE: c_dispatch ===")
engine.set_stream_overlap_dispatch(False)
engine.set_cached_dispatch(False)
engine.set_kernel_p2p_allreduce(True)
engine.set_c_dispatch(True)

rng = np.random.default_rng(REF_SEED)
sims_c = []
for step in range(CORRECTNESS_STEPS):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    if step < len(ref_outputs):
        cs = cosine_sim(out, ref_outputs[step])
        sims_c.append(cs)
        print(f"  c_step{{step}}: {{cs:.6f}}")

reset_tp()
for i in range(WARMUP_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
reset_tp()

t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
t1 = time.perf_counter()
tps_c = BENCH_STEPS / (t1 - t0)
print(f"TPS_C_DISPATCH={{tps_c:.2f}}")
reset_tp()

# =========================================================
# MODE 3: Global graph dispatch (highest priority)
# =========================================================
print("\\n=== MODE: global_graph ===")
engine.set_global_graph_dispatch(True)

# Correctness: first step triggers capture
rng = np.random.default_rng(REF_SEED)
sims_gg = []
for step in range(CORRECTNESS_STEPS):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    if step < len(ref_outputs):
        cs = cosine_sim(out, ref_outputs[step])
        sims_gg.append(cs)
        print(f"  gg_step{{step}}: {{cs:.6f}}")

# Check C graph plan
gds = engine._global_graph_decode_state
capture_ok = gds is not None and gds.captured
c_plan_ok = (capture_ok and
              hasattr(gds, '_c_graph_plan_ptr') and
              gds._c_graph_plan_ptr != 0)
print(f"CAPTURE_OK={{capture_ok}}")
print(f"C_PLAN_OK={{c_plan_ok}}")

# Layer node counts
if capture_ok and hasattr(gds, '_attn_segs') and gds._attn_segs:
    for li in range(min(3, config.num_hidden_layers)):
        a_n = gds._attn_segs[0][li].num_kernel_nodes()
        f_n = gds._ffn_segs[0][li].num_kernel_nodes()
        lw = engine.engines[0].layers[li]
        print(f"LAYER{{li}}_TYPE={{lw.layer_type}}_ATTN={{a_n}}_FFN={{f_n}}")

reset_tp()
for i in range(WARMUP_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
reset_tp()

t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
t1 = time.perf_counter()
tps_gg = BENCH_STEPS / (t1 - t0)
print(f"TPS_GLOBAL_GRAPH={{tps_gg:.2f}}")

# =========================================================
# Toggle test: disable and re-enable
# =========================================================
reset_tp()
engine.set_global_graph_dispatch(False)
toggle_disabled_ok = not engine._global_graph_dispatch_enabled
toggle_cleaned_ok = engine._global_graph_decode_state is None
# Verify C dispatch fallback works after disable
emb = np.random.randn(config.hidden_size).astype(np.float16)
try:
    out_fallback = engine.decode_step(emb, 0)
    toggle_fallback_ok = out_fallback is not None
except Exception as ex:
    toggle_fallback_ok = False
    print(f"Toggle fallback failed: {{ex}}")

toggle_ok = toggle_disabled_ok and toggle_cleaned_ok and toggle_fallback_ok
print(f"TOGGLE_OK={{toggle_ok}}")
print(f"TOGGLE_DISABLED={{toggle_disabled_ok}}")
print(f"TOGGLE_CLEANED={{toggle_cleaned_ok}}")

# =========================================================
# Output all results
# =========================================================
print("SIMS_GLOBAL=" + json.dumps([float(x) for x in sims_gg]))
print("SIMS_C_DISPATCH=" + json.dumps([float(x) for x in sims_c]))
print("SIMS_CACHED_STREAM=" + json.dumps([float(x) for x in sims_cs]))
print("ALL_DONE")
engine.cleanup()
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=900
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0 and "ALL_DONE" not in stdout:
        print("STDERR:", stderr[-3000:])
        print("STDOUT:", stdout[-2000:])
        return {'error': f"Subprocess failed (exit {result.returncode})"}

    # Parse results
    out = {}
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('TPS_GLOBAL_GRAPH='):
            out['tps_global'] = float(line.split('=')[1])
        elif line.startswith('TPS_C_DISPATCH='):
            out['tps_c_dispatch'] = float(line.split('=')[1])
        elif line.startswith('TPS_CACHED_STREAM='):
            out['tps_cached_stream'] = float(line.split('=')[1])
        elif line.startswith('CAPTURE_OK='):
            out['capture_ok'] = line.split('=')[1] == 'True'
        elif line.startswith('C_PLAN_OK='):
            out['c_plan_ok'] = line.split('=')[1] == 'True'
        elif line.startswith('TOGGLE_OK='):
            out['toggle_ok'] = line.split('=')[1] == 'True'
        elif line.startswith('SIMS_GLOBAL='):
            out['sims_global'] = json.loads(line[12:])
        elif line.startswith('SIMS_C_DISPATCH='):
            out['sims_c_dispatch'] = json.loads(line[16:])
        elif line.startswith('SIMS_CACHED_STREAM='):
            out['sims_cached_stream'] = json.loads(line[19:])
        elif line.startswith('REF_OK='):
            out['ref_steps'] = int(line.split('=')[1])
        elif line.startswith('LAYER') and '_TYPE=' in line:
            # Print layer info
            print(f"    {line}")

    # Print relevant output lines
    print_lines = []
    for line in stdout.split('\n'):
        if any(x in line for x in ['tok/s', 'cosine', 'CAPTURE', 'C_PLAN', 'TOGGLE',
                                     'TPS_', 'gg_step', 'c_step', 'cs_step',
                                     'Loading', 'Capturing', 'Layer', 'kernel',
                                     'global graph', 'C dispatch plan', 'REF_OK',
                                     'ALL_DONE']):
            print_lines.append(line)
    for line in print_lines[:80]:  # cap output
        print(f"    {line}")

    out['stdout'] = stdout
    return out


# ============================================================================
# Main entry
# ============================================================================

def main():
    print("=" * 72)
    print("  Global HIP Graph Bench — 100-step TP=4 (4x MI50 gfx906)")
    print("=" * 72)
    print(f"  Model: {MODEL_DIR}")
    print(f"  GPUs: {DEVICE_IDS}")
    print(f"  Benchmark steps: {BENCH_STEPS}")
    print(f"  Correctness steps: {CORRECTNESS_STEPS}")
    print(f"  Cosine sim threshold: {COSINE_SIM_THRESHOLD}")
    print(f"  C dispatch baseline: {C_DISPATCH_BASELINE_TPS:.1f} tok/s")

    print_header("Running All Dispatch Modes (one engine load)")
    t_start = time.perf_counter()
    r = run_all_modes_subprocess(BENCH_STEPS, CORRECTNESS_STEPS, ref_seed=42)
    t_elapsed = time.perf_counter() - t_start
    print(f"\n  Total subprocess time: {t_elapsed:.0f}s")

    if 'error' in r:
        print(f"  FATAL: {r['error']}")
        sys.exit(1)

    # ---- Parse and report VAL-GGC-001: Capture ----
    print_header("Results: VAL-GGC-001 — Full-Layer Graph Capture")
    capture_ok = r.get('capture_ok', False)
    c_plan_ok = r.get('c_plan_ok', False)
    record("capture_global_graph", capture_ok,
           "All layers captured" if capture_ok else "Capture FAILED")
    record("c_graph_dispatch_active", c_plan_ok,
           "C graph dispatch plan built" if c_plan_ok else "Python fallback used")

    # ---- VAL-GGC-002 & VAL-GGC-003: Correctness ----
    print_header("Results: VAL-GGC-002/003 — Correctness (Global Graph)")
    sims_gg = r.get('sims_global', [])
    if sims_gg:
        min_gg = min(sims_gg)
        all_ok_gg = all(s >= COSINE_SIM_THRESHOLD for s in sims_gg)
        print(f"  Global graph: min_cosine_sim={min_gg:.6f} "
              f"({'PASS' if all_ok_gg else 'FAIL'})")
        for i, s in enumerate(sims_gg):
            status = "OK" if s >= COSINE_SIM_THRESHOLD else "FAIL"
            print(f"    Step {i:2d}: cosine_sim={s:.6f}  {status}")
        record("global_graph_correctness", all_ok_gg,
               f"min={min_gg:.6f}")
        record("global_graph_mutable_params", all_ok_gg,
               f"all {len(sims_gg)} steps pass >= {COSINE_SIM_THRESHOLD}")
    else:
        record("global_graph_correctness", False, "No cosine_sim data")
        record("global_graph_mutable_params", False, "No cosine_sim data")
        all_ok_gg = False

    # ---- Correctness for other modes (VAL-GGC-006) ----
    print_header("Results: VAL-GGC-006 / VAL-CROSS-002 — Fallback Chain Correctness")
    sims_c = r.get('sims_c_dispatch', [])
    sims_cs = r.get('sims_cached_stream', [])

    if sims_c:
        min_c = min(sims_c)
        ok_c = all(s >= COSINE_SIM_THRESHOLD for s in sims_c)
        print(f"  C dispatch:    min_cosine_sim={min_c:.6f} ({'PASS' if ok_c else 'FAIL'})")
        record("fallback_c_dispatch_correctness", ok_c, f"min={min_c:.6f}")
    else:
        record("fallback_c_dispatch_correctness", False, "No data")
        ok_c = False

    if sims_cs:
        min_cs = min(sims_cs)
        ok_cs = all(s >= COSINE_SIM_THRESHOLD for s in sims_cs)
        print(f"  Cached+stream: min_cosine_sim={min_cs:.6f} ({'PASS' if ok_cs else 'FAIL'})")
        record("fallback_cached_stream_correctness", ok_cs, f"min={min_cs:.6f}")
    else:
        record("fallback_cached_stream_correctness", False, "No data")
        ok_cs = False

    all_fallback_ok = all_ok_gg and ok_c and ok_cs
    record("fallback_chain_all_modes", all_fallback_ok,
           "All 3 modes produce cosine_sim >= 0.99")

    # ---- VAL-GGC-004: Throughput ----
    print_header("Results: VAL-GGC-004 — 100-Step Throughput Benchmark")
    tps_gg = r.get('tps_global', 0.0)
    tps_c = r.get('tps_c_dispatch', 0.0)
    tps_cs = r.get('tps_cached_stream', 0.0)
    tps_values['global_graph'] = tps_gg
    tps_values['c_dispatch_kernel_p2p'] = tps_c
    tps_values['cached_stream'] = tps_cs

    speedup_vs_c  = tps_gg / tps_c  if tps_c  > 0 else float('nan')
    speedup_vs_cs = tps_gg / tps_cs if tps_cs > 0 else float('nan')

    print(f"  Cached + stream:         {tps_cs:.1f} tok/s")
    print(f"  C dispatch + kernel P2P: {tps_c:.1f} tok/s (baseline ~{C_DISPATCH_BASELINE_TPS} tok/s)")
    print(f"  Global graph dispatch:   {tps_gg:.1f} tok/s")
    print(f"  Speedup vs C dispatch:   {speedup_vs_c:.2f}x")
    print(f"  Speedup vs cached+stream:{speedup_vs_cs:.2f}x")

    # VAL-GGC-004: Global graph should be >= 95% of C dispatch (no regression)
    throughput_ok = tps_gg >= (tps_c * 0.95) if tps_c > 0 else False
    record("global_graph_throughput",
           throughput_ok,
           f"global={tps_gg:.1f} tok/s vs c_dispatch={tps_c:.1f} tok/s ({speedup_vs_c:.2f}x)")

    # At least one mode should hit the C dispatch baseline (~21 tok/s)
    best_tps = max(tps_gg, tps_c, tps_cs)
    baseline_ok = best_tps >= C_DISPATCH_BASELINE_TPS * 0.95
    record("c_dispatch_baseline_tps",
           baseline_ok,
           f"best={best_tps:.1f} tok/s >= {C_DISPATCH_BASELINE_TPS:.1f} tok/s")

    # ---- VAL-GGC-005: Toggle ----
    print_header("Results: VAL-GGC-005 — Global Graph Toggle")
    toggle_ok = r.get('toggle_ok', False)
    record("global_graph_toggle", toggle_ok,
           "enable/disable works correctly" if toggle_ok else "Toggle FAILED")

    # ============================================================================
    # Summary
    # ============================================================================
    print_header("Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for name, ok in sorted(results.items()):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print("  ---- Throughput Results ----")
    print(f"  Cached + stream:         {tps_values.get('cached_stream', 0):.1f} tok/s")
    print(f"  C dispatch + kernel P2P: {tps_values.get('c_dispatch_kernel_p2p', 0):.1f} tok/s")
    print(f"  Global graph dispatch:   {tps_values.get('global_graph', 0):.1f} tok/s")
    print(f"  Baseline (C dispatch):   {C_DISPATCH_BASELINE_TPS:.1f} tok/s (reference)")

    print()
    print(f"  Results: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print("\n  *** ALL TESTS PASSED ***")
        sys.exit(0)
    else:
        print("\n  *** SOME TESTS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
