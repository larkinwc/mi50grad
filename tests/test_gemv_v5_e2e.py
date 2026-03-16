#!/usr/bin/env python3
"""
E2E benchmark: gemv_int4_v5 wired as default GEMV in TP=4 decode engine.

Verifies:
  VAL-DPP-003: TP=4 decode throughput with v5 shows no regression vs current
               best TP=4 (kernel P2P + C dispatch, ~21.1 tok/s on 4x MI50).
  VAL-CROSS-003: Single-GPU throughput >= 18.3 tok/s.

Structure (avoids OOM from running multiple engines simultaneously):
  Phase 1: Single-GPU benchmark + reference output collection (GPU 0 only)
  Phase 2: TP=4 correctness check + throughput benchmark (all 4 GPUs)

USAGE:
    # Stop vLLM first, then run with all 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_gemv_v5_e2e.py'
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
WARMUP_STEPS = 5
BENCH_STEPS = 100
COSINE_STEPS = 10

TP4_MIN_TOKS = 20.0        # tok/s (current best ~21.1, allow noise margin)
SINGLE_GPU_MIN_TOKS = 18.3  # tok/s (20.3 baseline - 10% = 18.3)
COSINE_SIM_THRESHOLD = 0.99


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    dot = float(np.dot(a32, b32))
    norm_a = float(np.linalg.norm(a32))
    norm_b = float(np.linalg.norm(b32))
    return dot / (norm_a * norm_b) if norm_a > 1e-12 and norm_b > 1e-12 else 0.0


def print_header(title: str):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


config = load_config_from_json(MODEL_DIR)

# ============================================================
# PHASE 1: Single-GPU — reference collection + benchmark
# ============================================================
print_header("PHASE 1: Single-GPU (reference output + regression benchmark)")

sg_engine = InferenceEngine(config, device_id=0)
sg_loader = QwenWeightLoader(MODEL_DIR, config)
print("  Loading weights on GPU 0...")
for i in range(config.num_hidden_layers):
    sg_engine.load_layer_weights(i, sg_loader.load_layer(i))
sg_engine.load_final_norm(sg_loader.load_final_norm())
sg_engine.load_lm_head(sg_loader.load_lm_head())
print(f"  Weights loaded. v5={sg_engine._gemv_int4_v5}, v3={sg_engine._gemv_int4_v3}")

# Collect reference outputs for correctness
np.random.seed(42)
emb0 = np.random.randn(config.hidden_size).astype(np.float16)
ref_outputs = []
print(f"  Collecting {COSINE_STEPS} reference decode outputs...")
for step in range(COSINE_STEPS):
    out = sg_engine.decode_step(emb0, step)
    ref_outputs.append(out.copy())
sg_engine.device.synchronize()

# Reset and benchmark single GPU
sg_engine.kv_cache.current_len = 0
sg_engine.deltanet_state.reset()
emb_sg = np.random.randn(config.hidden_size).astype(np.float16)

print(f"  Warmup ({WARMUP_STEPS} steps)...")
for i in range(WARMUP_STEPS):
    sg_engine.decode_step(emb_sg, i)
sg_engine.device.synchronize()

sg_engine.kv_cache.current_len = 0
sg_engine.deltanet_state.reset()

print(f"  Benchmarking ({BENCH_STEPS} steps)...")
t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    sg_engine.decode_step(emb_sg, i)
sg_engine.device.synchronize()
elapsed_sg = time.perf_counter() - t0

single_gpu_toks = BENCH_STEPS / elapsed_sg
single_gpu_ms = elapsed_sg / BENCH_STEPS * 1000
sg_pass = single_gpu_toks >= SINGLE_GPU_MIN_TOKS
sg_status = "PASS" if sg_pass else "FAIL"
print(f"  Single-GPU: {single_gpu_toks:.1f} tok/s ({single_gpu_ms:.1f} ms/tok)")
print(f"  Kernel path: {'v5_t16 (DPP+LDS)' if sg_engine._gemv_int4_v5 else 'v3_t16 (fallback)'}")
print(f"  Threshold >= {SINGLE_GPU_MIN_TOKS} tok/s: [{sg_status}]")

sg_engine.cleanup()
del sg_engine, sg_loader


# ============================================================
# PHASE 2: TP=4 — correctness + throughput
# ============================================================
print_header("PHASE 2: TP=4 (correctness + throughput)")

tp_engine = TPInferenceEngine(config, device_ids=DEVICE_IDS)
tp_loader = QwenWeightLoader(MODEL_DIR, config)
print("  Loading weights on 4 GPUs...")
for i in range(config.num_hidden_layers):
    tp_engine.load_layer_weights(i, tp_loader.load_layer(i))
tp_engine.load_final_norm(tp_loader.load_final_norm())
tp_engine.load_lm_head(tp_loader.load_lm_head())
print("  Weights loaded.")

for idx, eng in enumerate(tp_engine.engines):
    print(f"  Engine[{idx}]: v5={eng._gemv_int4_v5}, v3={eng._gemv_int4_v3}")

# Enable best dispatch mode: C dispatch + kernel P2P allreduce (~21.1 tok/s on 4x MI50)
tp_engine.build_dispatch_cache()
tp_engine.set_kernel_p2p_allreduce(True)
tp_engine.set_c_dispatch(True)
print(f"  C dispatch={tp_engine._c_dispatch_enabled}, kernel P2P={tp_engine._kernel_p2p_allreduce}")

# --- TP=4 Correctness ---
print_header("TEST: TP=4 Correctness (cosine sim >= 0.99 vs single-GPU)")

for eng in tp_engine.engines:
    eng.kv_cache.current_len = 0
    eng.deltanet_state.reset()

cosine_sims = []
print(f"  Running {COSINE_STEPS} decode steps...")
for step in range(COSINE_STEPS):
    out_tp = tp_engine.decode_step(emb0, step)
    cs = cosine_similarity(np.array(out_tp, dtype=np.float16), ref_outputs[step])
    cosine_sims.append(cs)
    print(f"    Step {step+1:2d}: cosine_sim = {cs:.6f}")

min_cosine = min(cosine_sims)
correctness_pass = min_cosine >= COSINE_SIM_THRESHOLD
correctness_status = "PASS" if correctness_pass else "FAIL"
print(f"  Min cosine_sim = {min_cosine:.6f}, threshold >= {COSINE_SIM_THRESHOLD}: [{correctness_status}]")

# --- TP=4 Throughput ---
print_header("TEST: TP=4 Throughput (v5 as default)")

emb_bench = np.random.randn(config.hidden_size).astype(np.float16)

for eng in tp_engine.engines:
    eng.kv_cache.current_len = 0
    eng.deltanet_state.reset()

print(f"  Warmup ({WARMUP_STEPS} steps)...")
for step in range(WARMUP_STEPS):
    tp_engine.decode_step(emb_bench, step)
for eng in tp_engine.engines:
    eng.device.synchronize()

for eng in tp_engine.engines:
    eng.kv_cache.current_len = 0
    eng.deltanet_state.reset()

print(f"  Benchmarking ({BENCH_STEPS} steps)...")
t0 = time.perf_counter()
for step in range(BENCH_STEPS):
    tp_engine.decode_step(emb_bench, step)
for eng in tp_engine.engines:
    eng.device.synchronize()
elapsed_tp = time.perf_counter() - t0

tp4_toks = BENCH_STEPS / elapsed_tp
tp4_ms = elapsed_tp / BENCH_STEPS * 1000
tp4_pass = tp4_toks >= TP4_MIN_TOKS
tp4_status = "PASS" if tp4_pass else "FAIL"
print(f"  TP=4 result: {tp4_toks:.1f} tok/s ({tp4_ms:.1f} ms/tok)")
print(f"  Reference baseline (kernel P2P + C dispatch): ~21.1 tok/s")
print(f"  Threshold >= {TP4_MIN_TOKS} tok/s: [{tp4_status}]")

tp_engine.cleanup()
del tp_engine, tp_loader


# ============================================================
# Summary
# ============================================================
print_header("SUMMARY")
print(f"  Single-GPU (v5 default): {single_gpu_toks:.1f} tok/s  [{sg_status}]")
print(f"  TP=4 cosine sim min:     {min_cosine:.6f}  [{correctness_status}]")
print(f"  TP=4 throughput (v5):    {tp4_toks:.1f} tok/s  [{tp4_status}]")
print()

all_pass = sg_pass and correctness_pass and tp4_pass
if all_pass:
    print("All tests PASSED.")
    sys.exit(0)
else:
    print("SOME tests FAILED:")
    if not sg_pass:
        print(f"  FAIL: Single-GPU {single_gpu_toks:.1f} tok/s < {SINGLE_GPU_MIN_TOKS} threshold")
    if not correctness_pass:
        print(f"  FAIL: TP=4 cosine sim {min_cosine:.6f} < {COSINE_SIM_THRESHOLD} threshold")
    if not tp4_pass:
        print(f"  FAIL: TP=4 throughput {tp4_toks:.1f} tok/s < {TP4_MIN_TOKS} threshold")
    sys.exit(1)
