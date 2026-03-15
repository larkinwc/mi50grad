"""
tests/test_qkv_sync_removal.py — Q/KV stream sync elimination test.

Tests:
1. Stream object preservation (streams kept alive, not destroyed)
2. Single-GPU regression (within ±10% of 20.3 tok/s baseline)
3. TP=4 correctness (cosine sim >= 0.99 vs single-GPU, 10 steps)
4. TP=4 latency benchmark (measures effect of eliminating 32 sync calls/token)

Usage:
    # Stop vLLM first, then:
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
        -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
        mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_qkv_sync_removal.py'
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Flush output immediately
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
BENCH_STEPS = 50
CORRECTNESS_STEPS = 10
COSINE_THRESHOLD = 0.99
SINGLE_GPU_BASELINE = 20.3     # tok/s
REGRESSION_MARGIN = 0.10       # ±10%
MAX_SEQ_LEN = 512
SPRINT2_TP4_BASELINE = 38.0    # tok/s


def cosine_sim(a, b):
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    dot = float(np.dot(a32, b32))
    den = float(np.linalg.norm(a32) * np.linalg.norm(b32)) + 1e-12
    return dot / den


def reset_tp(engine):
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


def reset_single(engine):
    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def load_single_gpu(config, loader):
    print(f"\nLoading single-GPU engine (device {DEVICE_ID_SINGLE})...")
    t0 = time.perf_counter()
    engine = InferenceEngine(config, device_id=DEVICE_ID_SINGLE,
                              max_seq_len=MAX_SEQ_LEN)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print(f"Single-GPU engine loaded in {time.perf_counter()-t0:.1f}s")
    return engine


def load_tp4(config, loader):
    print(f"\nLoading TP=4 engine (devices {DEVICE_IDS})...")
    t0 = time.perf_counter()
    engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    engine.build_dispatch_cache()
    print(f"TP=4 engine loaded in {time.perf_counter()-t0:.1f}s")
    return engine


# -----------------------------------------------------------------------
# Test 1: Stream object preservation
# -----------------------------------------------------------------------

def test_stream_preservation(engine):
    """Verify stream objects are kept alive (not destroyed) after sync removal."""
    print("\n=== Test 1: Stream Object Preservation ===")
    has_q  = hasattr(engine, '_stream_q')
    has_kv = hasattr(engine, '_stream_kv')
    has_rdy = hasattr(engine, '_streams_ready')
    print(f"  _stream_q exists:   {has_q}")
    print(f"  _stream_kv exists:  {has_kv}")
    print(f"  _streams_ready:     {engine._streams_ready if has_rdy else 'missing'}")
    # Streams should still exist (not destroyed) so prefill can use them if needed
    passed = has_q and has_kv
    print(f"  {'PASS' if passed else 'FAIL'}: streams preserved for other paths")
    return passed


# -----------------------------------------------------------------------
# Test 2: Single-GPU regression check
# -----------------------------------------------------------------------

def test_single_gpu_regression(config, loader):
    """Verify single-GPU throughput within ±10% of 20.3 tok/s baseline."""
    print("\n=== Test 2: Single-GPU Regression Check ===")

    engine = load_single_gpu(config, loader)
    h = config.hidden_size
    np.random.seed(0)
    emb = np.random.randn(h).astype(np.float16) * 0.02

    # Test stream preservation here (while engine is loaded)
    stream_ok = test_stream_preservation(engine)

    # Warmup
    reset_single(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.device.synchronize()

    # Bench
    reset_single(engine)
    times = []
    for i in range(BENCH_STEPS):
        t0 = time.perf_counter()
        engine.decode_step(emb, i)
        times.append(time.perf_counter() - t0)
    engine.device.synchronize()

    median_ms = np.median(times) * 1000
    tok_s = 1.0 / np.median(times)

    low  = SINGLE_GPU_BASELINE * (1 - REGRESSION_MARGIN)
    high = SINGLE_GPU_BASELINE * (1 + REGRESSION_MARGIN)
    passed = low <= tok_s <= high

    print(f"\n  Result:   {tok_s:.1f} tok/s  ({median_ms:.2f} ms/tok)")
    print(f"  Expected: [{low:.1f}, {high:.1f}] tok/s  (20.3 ± 10%)")
    print(f"  {'PASS' if passed else 'FAIL'}")

    engine.cleanup()
    del engine

    return stream_ok, passed, tok_s


# -----------------------------------------------------------------------
# Test 3 & 4: TP=4 correctness and latency
# -----------------------------------------------------------------------

def collect_single_gpu_reference(config, loader, embs):
    """Collect single-GPU reference outputs for comparison."""
    print(f"\nCollecting {len(embs)} single-GPU reference outputs...")
    engine = load_single_gpu(config, loader)

    # Warmup
    reset_single(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(embs[i % len(embs)], i)
    engine.device.synchronize()

    reset_single(engine)
    outs = []
    for step, emb in enumerate(embs):
        out = engine.decode_step(emb, step)
        outs.append(out.copy())
    engine.device.synchronize()

    print(f"  Reference outputs collected")
    engine.cleanup()
    del engine
    return outs


def test_tp4_correctness_and_latency(config, loader, single_outs, embs):
    """Test TP=4 correctness and measure latency after sync removal."""
    tp_engine = load_tp4(config, loader)

    # Enable C dispatch + cached + stream overlap (best-mode path)
    tp_engine.set_c_dispatch(True)
    print(f"\n  C dispatch enabled: {tp_engine._c_dispatch_enabled}")

    # ---- Correctness test ----
    print(f"\n=== Test 3: TP=4 Correctness ({CORRECTNESS_STEPS} steps, Q/KV sync removed) ===")

    # Warmup
    reset_tp(tp_engine)
    for i in range(WARMUP_STEPS):
        tp_engine.decode_step(embs[i % len(embs)], i)
    tp_engine.synchronize()

    # Collect TP=4 outputs
    reset_tp(tp_engine)
    min_cos = 1.0
    all_pass = True
    tp_outs = []

    for step in range(CORRECTNESS_STEPS):
        out = tp_engine.decode_step(embs[step], step)
        tp_outs.append(out.copy())
        cos = cosine_sim(single_outs[step], out)
        min_cos = min(min_cos, cos)
        ok = cos >= COSINE_THRESHOLD
        if not ok:
            all_pass = False
        print(f"  Step {step+1:2d}: cosine_sim={cos:.6f}  {'OK' if ok else 'FAIL'}")

    tp_engine.synchronize()
    print(f"  Min cosine sim: {min_cos:.6f}  (threshold {COSINE_THRESHOLD})")
    print(f"  {'PASS' if all_pass else 'FAIL'}")

    # ---- Latency benchmark ----
    print("\n=== Test 4: TP=4 Latency Benchmark (no Q/KV stream sync) ===")
    np.random.seed(1)
    bench_emb = np.random.randn(config.hidden_size).astype(np.float16) * 0.02

    # Warmup
    reset_tp(tp_engine)
    for i in range(WARMUP_STEPS):
        tp_engine.decode_step(bench_emb, i)
    tp_engine.synchronize()

    # Bench
    reset_tp(tp_engine)
    times = []
    for i in range(BENCH_STEPS):
        t0 = time.perf_counter()
        tp_engine.decode_step(bench_emb, i)
        times.append(time.perf_counter() - t0)
    tp_engine.synchronize()

    median_ms = np.median(times) * 1000
    tok_s = 1.0 / np.median(times)
    improvement = ((tok_s - SPRINT2_TP4_BASELINE) / SPRINT2_TP4_BASELINE) * 100

    print(f"  TP=4 (no Q/KV sync): {tok_s:.1f} tok/s  ({median_ms:.2f} ms/tok)")
    print(f"  Sprint 2 baseline:   {SPRINT2_TP4_BASELINE:.1f} tok/s")
    print(f"  Change vs Sprint 2:  {improvement:+.1f}%")
    print(f"  [Sync calls removed: 16 full-attn layers x 2 syncs = 32 per token]")

    tp_engine.cleanup()
    del tp_engine

    return all_pass, min_cos, tok_s, median_ms


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Q/KV Stream Sync Elimination Tests")
    print("=" * 70)

    # Check GPU count
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"\nGPUs visible: {n_gpus}")

    config = load_config_from_json(MODEL_DIR)
    loader = QwenWeightLoader(MODEL_DIR, config)

    results = {}

    # ---- Phase 1: Single-GPU (includes Test 1 stream preservation + Test 2 regression) ----
    print("\n" + "=" * 70)
    print("Phase 1: Single-GPU Tests")
    print("=" * 70)

    try:
        stream_ok, reg_ok, tok_s = test_single_gpu_regression(config, loader)
        results['stream_preservation'] = stream_ok
        results['single_gpu_regression'] = reg_ok
        results['single_gpu_tok_s'] = tok_s
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results['stream_preservation'] = False
        results['single_gpu_regression'] = False
        results['single_gpu_tok_s'] = 0.0

    # ---- Phase 2: TP=4 (Tests 3 + 4) ----
    if n_gpus < 4:
        print(f"\nWARNING: {n_gpus} GPU(s) visible, need 4 for TP=4 tests. Skipping.")
        print("To run: HIP_VISIBLE_DEVICES=0,1,2,3 python3 tests/test_qkv_sync_removal.py")
        results['tp4_correctness'] = None
        results['tp4_latency'] = None
    else:
        print("\n" + "=" * 70)
        print("Phase 2: TP=4 Tests")
        print("=" * 70)

        # Collect single-GPU reference (separate load, then free)
        np.random.seed(42)
        embs = [np.random.randn(config.hidden_size).astype(np.float16) * 0.02
                for _ in range(CORRECTNESS_STEPS)]

        try:
            single_outs = collect_single_gpu_reference(config, loader, embs)
        except Exception as e:
            print(f"ERROR collecting reference: {e}")
            import traceback; traceback.print_exc()
            results['tp4_correctness'] = False
            results['tp4_latency'] = None
            single_outs = None

        if single_outs is not None:
            try:
                passed, min_cos, tp4_tok_s, tp4_ms = test_tp4_correctness_and_latency(
                    config, loader, single_outs, embs)
                results['tp4_correctness'] = passed
                results['tp4_min_cos'] = min_cos
                results['tp4_latency'] = tp4_tok_s
                results['tp4_ms'] = tp4_ms
            except Exception as e:
                print(f"  ERROR in TP=4 test: {e}")
                import traceback; traceback.print_exc()
                results['tp4_correctness'] = False
                results['tp4_latency'] = None

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True

    s1 = results.get('stream_preservation', False)
    print(f"  [1] Stream preservation:     {'PASS' if s1 else 'FAIL'}")

    s2 = results.get('single_gpu_regression')
    if s2 is None:
        print(f"  [2] Single-GPU regression:   SKIP")
    elif s2:
        print(f"  [2] Single-GPU regression:   PASS  ({results.get('single_gpu_tok_s',0):.1f} tok/s)")
    else:
        print(f"  [2] Single-GPU regression:   FAIL  ({results.get('single_gpu_tok_s',0):.1f} tok/s)")
        all_passed = False

    s3 = results.get('tp4_correctness')
    if s3 is None:
        print(f"  [3] TP=4 correctness:        SKIP (< 4 GPUs)")
    elif s3:
        print(f"  [3] TP=4 correctness:        PASS  (min_cos={results.get('tp4_min_cos',0):.6f})")
    else:
        print(f"  [3] TP=4 correctness:        FAIL  (min_cos={results.get('tp4_min_cos',0):.6f})")
        all_passed = False

    s4 = results.get('tp4_latency')
    if s4 is None:
        print(f"  [4] TP=4 latency:            SKIP")
    else:
        print(f"  [4] TP=4 latency:            {s4:.1f} tok/s  ({results.get('tp4_ms',0):.2f} ms/tok)")
        print(f"      [32 stream_synchronize calls/token eliminated]")

    print()
    if all_passed:
        print("OVERALL: PASS")
        sys.exit(0)
    else:
        print("OVERALL: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
