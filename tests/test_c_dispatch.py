#!/usr/bin/env python3
"""
C dispatch extension: correctness and benchmark tests.

Tests:
1. Compilation and loading: c_dispatch.so compiles with gcc and loads via ctypes
2. Correctness: C dispatch produces cosine sim >= 0.99 vs Python cached+stream
   for 10 decode steps
3. Benchmark: C dispatch vs cached+stream tok/s over 100 steps
4. Position-dependent params: RoPE and seq_len update correctly
5. Both attention types: full GQA (16 layers) and DeltaNet (48 layers)

USAGE:
    # Stop vLLM first, then:
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
        -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
        mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_c_dispatch.py'
"""

import sys
import time
import os
import ctypes
import subprocess
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
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()


# -------------------------------------------------------------------------
# Test 1: Compilation and loading
# -------------------------------------------------------------------------

def test_compilation():
    print("\n=== Test 1: C Extension Compilation ===")
    src_dir = Path('/opt/mi50grad/src/runtime')
    c_path = src_dir / 'c_dispatch.c'
    so_path = src_dir / 'c_dispatch.so'

    assert c_path.exists(), f"c_dispatch.c not found at {c_path}"
    print(f"  Source: {c_path}")

    # Compile
    cmd = [
        'gcc', '-O3', '-shared', '-fPIC',
        '-I/opt/rocm/include',
        '-L/opt/rocm/lib', '-lamdhip64',
        '-o', str(so_path), str(c_path),
    ]
    print(f"  Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
        raise RuntimeError(f"Compilation failed with exit code {result.returncode}")
    print(f"  Compiled: {so_path}")

    # Load
    lib = ctypes.CDLL(str(so_path))
    lib.c_dispatch_get_spec_size.restype = ctypes.c_int
    lib.c_dispatch_get_kernel_spec_size.restype = ctypes.c_int
    lib.c_dispatch_get_allreduce_spec_size.restype = ctypes.c_int
    lib.c_dispatch_get_plan_size.restype = ctypes.c_int
    spec_size      = lib.c_dispatch_get_spec_size()
    kernel_size    = lib.c_dispatch_get_kernel_spec_size()
    ar_spec_size   = lib.c_dispatch_get_allreduce_spec_size()
    plan_size      = lib.c_dispatch_get_plan_size()
    print(f"  CKernelSpec:     {kernel_size} bytes")
    print(f"  CEngineLayerSpec: {spec_size} bytes")
    print(f"  CAllreduceSpec:  {ar_spec_size} bytes")
    print(f"  CDispatchPlan:   {plan_size} bytes")
    print("  PASS: C extension compiled and loaded successfully")
    return lib


# -------------------------------------------------------------------------
# Model loading (shared across tests)
# -------------------------------------------------------------------------

def load_model():
    print("\n=== Loading Model ===")
    config = load_config_from_json(MODEL_DIR)
    engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=2048)

    loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    print("  Weights loaded")

    # Build dispatch cache (required for C dispatch)
    engine.build_dispatch_cache()
    print("  Dispatch cache built")

    return config, engine


# -------------------------------------------------------------------------
# Test 2 & 3: Correctness and benchmark
# -------------------------------------------------------------------------

def run_decode_steps(engine: TPInferenceEngine, emb: np.ndarray,
                     mode: str, steps: int, warmup: int = 3):
    """Run decode steps in specified mode.

    mode: 'cached_stream' or 'c_dispatch'
    Returns list of output arrays.
    """
    engine.set_c_dispatch(False)
    engine.set_cached_dispatch(False)
    engine.set_stream_overlap_dispatch(False)

    if mode == 'cached_stream':
        engine.set_cached_dispatch(True)
        engine.set_stream_overlap_dispatch(True)
        label = "cached+stream"
    elif mode == 'c_dispatch':
        engine.set_c_dispatch(True)
        label = "C dispatch"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    reset_engine(engine)
    for i in range(warmup):
        engine.decode_step(emb, i)
    engine.synchronize()

    # Collect outputs
    reset_engine(engine)
    outputs = []
    for i in range(steps):
        out = engine.decode_step(emb, warmup + i)
        outputs.append(out.copy())
    engine.synchronize()

    print(f"  Completed {steps} {label} steps")
    return outputs


def test_correctness(engine: TPInferenceEngine, emb: np.ndarray):
    print(f"\n=== Test 2: Correctness (C dispatch vs cached+stream, {CORRECTNESS_STEPS} steps) ===")

    ref_outputs = run_decode_steps(engine, emb, 'cached_stream', CORRECTNESS_STEPS)
    c_outputs   = run_decode_steps(engine, emb, 'c_dispatch',    CORRECTNESS_STEPS)

    all_pass = True
    min_cos = 1.0
    for i, (ref, c) in enumerate(zip(ref_outputs, c_outputs)):
        cos = cosine_similarity(ref, c)
        min_cos = min(min_cos, cos)
        status = "PASS" if cos >= COSINE_SIM_THRESHOLD else "FAIL"
        if cos < COSINE_SIM_THRESHOLD:
            all_pass = False
        print(f"  Step {i:2d}: cosine_sim = {cos:.6f}  [{status}]")

    print(f"  Min cosine similarity: {min_cos:.6f} (threshold: {COSINE_SIM_THRESHOLD})")
    if all_pass:
        print("  PASS: All steps meet correctness threshold")
    else:
        print("  FAIL: Some steps below threshold!")
    return all_pass, min_cos


def test_benchmark(engine: TPInferenceEngine, emb: np.ndarray):
    print(f"\n=== Test 3: Benchmark ({BENCH_STEPS} steps) ===")

    # Benchmark cached+stream
    engine.set_c_dispatch(False)
    engine.set_cached_dispatch(True)
    engine.set_stream_overlap_dispatch(True)
    reset_engine(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.synchronize()

    reset_engine(engine)
    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        engine.decode_step(emb, WARMUP_STEPS + i)
    engine.synchronize()
    t_cached_stream = time.perf_counter() - t0
    toks_cached_stream = BENCH_STEPS / t_cached_stream
    ms_cached_stream = t_cached_stream / BENCH_STEPS * 1000

    # Benchmark C dispatch
    engine.set_c_dispatch(True)
    reset_engine(engine)
    for i in range(WARMUP_STEPS):
        engine.decode_step(emb, i)
    engine.synchronize()

    reset_engine(engine)
    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        engine.decode_step(emb, WARMUP_STEPS + i)
    engine.synchronize()
    t_c_dispatch = time.perf_counter() - t0
    toks_c_dispatch = BENCH_STEPS / t_c_dispatch
    ms_c_dispatch = t_c_dispatch / BENCH_STEPS * 1000

    speedup = toks_c_dispatch / toks_cached_stream

    print(f"  Cached+stream: {toks_cached_stream:.1f} tok/s ({ms_cached_stream:.1f} ms/tok)")
    print(f"  C dispatch:    {toks_c_dispatch:.1f} tok/s ({ms_c_dispatch:.1f} ms/tok)")
    print(f"  Speedup:       {speedup:.2f}x")

    improved = toks_c_dispatch > toks_cached_stream
    if improved:
        print(f"  PASS: C dispatch is faster than cached+stream")
    else:
        print(f"  INFO: C dispatch did not improve over cached+stream "
              f"(may be due to C function call overhead at small scales)")

    return toks_c_dispatch, toks_cached_stream, speedup


# -------------------------------------------------------------------------
# Test 4: Position-dependent params across multiple positions
# -------------------------------------------------------------------------

def test_position_dependent(engine: TPInferenceEngine, emb: np.ndarray):
    """Verify correctness at positions 0, 10, 50, 100."""
    print(f"\n=== Test 4: Position-Dependent Params ===")

    test_positions = [0, 10, 50, 100]
    all_pass = True

    for start_pos in test_positions:
        # Run cached+stream for 1 step from start_pos
        engine.set_c_dispatch(False)
        engine.set_cached_dispatch(True)
        engine.set_stream_overlap_dispatch(True)
        reset_engine(engine)
        # Set KV cache to simulate being at start_pos
        for e in engine.engines:
            e.kv_cache.current_len = start_pos

        ref = engine.decode_step(emb, start_pos)
        engine.synchronize()

        # Run C dispatch for 1 step from start_pos
        engine.set_c_dispatch(True)
        reset_engine(engine)
        for e in engine.engines:
            e.kv_cache.current_len = start_pos

        c_out = engine.decode_step(emb, start_pos)
        engine.synchronize()

        cos = cosine_similarity(ref, c_out)
        status = "PASS" if cos >= COSINE_SIM_THRESHOLD else "FAIL"
        if cos < COSINE_SIM_THRESHOLD:
            all_pass = False
        print(f"  Position {start_pos:3d}: cosine_sim = {cos:.6f}  [{status}]")

    if all_pass:
        print("  PASS: Position-dependent params correct at all tested positions")
    else:
        print("  FAIL: Some positions failed!")
    return all_pass


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("C Dispatch Extension Test Suite")
    print("=" * 60)

    # Test 1: Compilation
    lib = test_compilation()

    # Load model (shared across remaining tests)
    config, engine = load_model()

    # Random embedding
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # Test 2: Correctness
    correct, min_cos = test_correctness(engine, emb)

    # Test 3: Benchmark
    toks_c, toks_ref, speedup = test_benchmark(engine, emb)

    # Test 4: Position-dependent params
    pos_correct = test_position_dependent(engine, emb)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (compilation):       PASS")
    print(f"  Test 2 (correctness):       {'PASS' if correct else 'FAIL'} (min_cos={min_cos:.6f})")
    print(f"  Test 3 (benchmark):         C={toks_c:.1f} tok/s, ref={toks_ref:.1f} tok/s, {speedup:.2f}x")
    print(f"  Test 4 (position params):   {'PASS' if pos_correct else 'FAIL'}")

    all_pass = correct and pos_correct
    if all_pass:
        print("\nALL CRITICAL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
