#!/usr/bin/env python3
"""
TP=4 throughput benchmark for INT8-compressed allreduce.

Tests:
1. Throughput: compressed vs uncompressed (target: compressed > 53.74 tok/s)
2. Correctness: cosine similarity between compressed and uncompressed (target: >= 0.99)
3. Single-GPU regression: ensure no regression in single-GPU mode (target: >= 21.0 tok/s)
4. Batch=1 uncompressed: ensure no regression (target: >= 53.0 tok/s)

Usage:
    python3 tests/bench_compressed_ar_tp4.py
"""

import sys, os, time, ctypes
import numpy as np
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 100
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256


def reset_tp(tp_engine):
    """Reset KV cache position for all TP engines."""
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_f32 = a.astype(np.float32).flatten()
    b_f32 = b.astype(np.float32).flatten()
    
    dot = np.dot(a_f32, b_f32)
    norm_a = np.linalg.norm(a_f32)
    norm_b = np.linalg.norm(b_f32)
    
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0 if np.allclose(a_f32, b_f32, atol=1e-6) else 0.0
    
    return float(dot / (norm_a * norm_b))


def bench_decode_collect_output(label, tp_engine, config, n_steps=10, collect_outputs=False):
    """
    Benchmark TP=4 decode throughput.
    
    If collect_outputs=True, returns the last hidden state for correctness comparison.
    """
    rng = np.random.default_rng(42)
    hip = tp_engine._hip
    
    # Use fixed random seed for reproducibility
    test_embs = [rng.standard_normal(config.hidden_size).astype(np.float16) 
                 for _ in range(WARMUP_STEPS + n_steps)]
    
    # Warmup
    for i in range(WARMUP_STEPS):
        reset_tp(tp_engine)
        tp_engine.decode_step(test_embs[i], i)
        hip.synchronize()
    
    # Benchmark
    reset_tp(tp_engine)
    t0 = time.perf_counter()
    last_hidden = None
    
    for step in range(n_steps):
        hidden = tp_engine.decode_step(test_embs[WARMUP_STEPS + step], step)
        if collect_outputs and step == n_steps - 1:
            # Get hidden state from GPU0
            hip.set_device(tp_engine.device_ids[0])
            last_hidden = np.frombuffer(
                hip.download(tp_engine.engines[0].d_hidden, config.hidden_size * 2),
                dtype=np.float16
            ).copy()
        tp_engine._hip.synchronize()
    
    elapsed = time.perf_counter() - t0
    tps = n_steps / elapsed
    ms_per_tok = elapsed / n_steps * 1000
    print(f"  {label}: {tps:.2f} tok/s ({ms_per_tok:.2f} ms/tok)")
    
    if collect_outputs:
        return tps, ms_per_tok, last_hidden
    return tps, ms_per_tok


def load_tp_engine(config, enable_compressed=False):
    """Load TP engine with specified allreduce mode."""
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader
    
    print(f"Loading TP=4 engine (compressed={enable_compressed})...")
    tp = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
    
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 16 == 0:
            print(f"    Layer {layer_idx}...")
        tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp.load_final_norm(loader.load_final_norm())
    tp.load_lm_head(loader.load_lm_head())
    
    # Enable optimizations
    tp.build_dispatch_cache()
    tp.set_direct_kv_write(True)
    tp.set_c_dispatch(True)
    tp.set_kernel_p2p_allreduce(True)
    
    # Enable compressed allreduce if requested
    if enable_compressed:
        tp.set_compressed_allreduce(True)
        print(f"  INT8-compressed allreduce ENABLED")
    else:
        print(f"  Using standard FP16 allreduce (uncompressed)")
    
    return tp


def main():
    print("=" * 72)
    print("  INT8-Compressed Allreduce TP=4 Benchmark")
    print("=" * 72)
    print(f"  Model: {MODEL_DIR}")
    print(f"  Devices: {DEVICE_IDS}")
    print(f"  Steps: {BENCH_STEPS}, Warmup: {WARMUP_STEPS}")
    print(f"  MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print()

    from src.model.qwen import load_config_from_json
    config = load_config_from_json(MODEL_DIR)
    results = {}
    
    # ====================================================================
    # Test 1: Uncompressed baseline
    # ====================================================================
    print("=" * 72)
    print("  TEST 1: Uncompressed Baseline")
    print("=" * 72)
    
    tp_uncompressed = None
    try:
        tp_uncompressed = load_tp_engine(config, enable_compressed=False)
        tps_uncomp, ms_uncomp, hidden_uncomp = bench_decode_collect_output(
            "Uncompressed (FP16 AR)", tp_uncompressed, config, 
            n_steps=BENCH_STEPS, collect_outputs=True
        )
        results["uncompressed"] = {"tps": tps_uncomp, "ms": ms_uncomp, "hidden": hidden_uncomp}
    except Exception as e:
        print(f"ERROR: Uncompressed benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["uncompressed"] = None
    
    if tp_uncompressed:
        tp_uncompressed.cleanup()
        del tp_uncompressed
    
    # Force GC to free GPU memory
    import gc; gc.collect()
    time.sleep(2)
    
    # ====================================================================
    # Test 2: Compressed allreduce
    # ====================================================================
    print()
    print("=" * 72)
    print("  TEST 2: Compressed Allreduce")
    print("=" * 72)
    
    tp_compressed = None
    try:
        tp_compressed = load_tp_engine(config, enable_compressed=True)
        tps_comp, ms_comp, hidden_comp = bench_decode_collect_output(
            "Compressed (INT8 AR)", tp_compressed, config, 
            n_steps=BENCH_STEPS, collect_outputs=True
        )
        results["compressed"] = {"tps": tps_comp, "ms": ms_comp, "hidden": hidden_comp}
    except Exception as e:
        print(f"ERROR: Compressed benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["compressed"] = None
    
    if tp_compressed:
        tp_compressed.cleanup()
        del tp_compressed
    
    # Force GC to free GPU memory
    import gc; gc.collect()
    time.sleep(2)
    
    # ====================================================================
    # Test 3: Correctness comparison (cosine similarity)
    # ====================================================================
    print()
    print("=" * 72)
    print("  TEST 3: Correctness (Cosine Similarity)")
    print("=" * 72)
    
    if results["uncompressed"] and results["compressed"]:
        hidden_uncomp = results["uncompressed"]["hidden"]
        hidden_comp = results["compressed"]["hidden"]
        
        cos_sim = cosine_similarity(hidden_uncomp, hidden_comp)
        max_abs_err = np.max(np.abs(hidden_uncomp.astype(np.float32) - hidden_comp.astype(np.float32)))
        
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max absolute error: {max_abs_err:.6e}")
        results["correctness"] = {"cosine_sim": cos_sim, "max_abs_err": max_abs_err}
    else:
        print("  SKIPPED: One or both benchmarks failed")
        results["correctness"] = None
    
    # ====================================================================
    # Test 4: Single-GPU baseline (no regression check)
    # ====================================================================
    print()
    print("=" * 72)
    print("  TEST 4: Single-GPU Baseline (Regression Check)")
    print("=" * 72)
    
    try:
        from src.inference.engine import InferenceEngine
        from src.model.weight_loader import QwenWeightLoader
        
        print("Loading single-GPU engine...")
        single_eng = InferenceEngine(config, device_id=0, max_seq_len=MAX_SEQ_LEN)
        
        loader = QwenWeightLoader(MODEL_DIR, config)
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx % 16 == 0:
                print(f"    Layer {layer_idx}...")
            single_eng.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
        single_eng.load_final_norm(loader.load_final_norm())
        single_eng.load_lm_head(loader.load_lm_head())
        
        rng = np.random.default_rng(42)
        
        # Warmup
        for i in range(WARMUP_STEPS):
            emb = rng.standard_normal(config.hidden_size).astype(np.float16)
            single_eng.kv_cache.current_len = 0
            single_eng.decode_step(emb, i)
            single_eng.device.synchronize()
        
        # Benchmark
        single_eng.kv_cache.current_len = 0
        t0 = time.perf_counter()
        for step in range(BENCH_STEPS):
            emb = rng.standard_normal(config.hidden_size).astype(np.float16)
            single_eng.decode_step(emb, step)
            single_eng.device.synchronize()
        elapsed = time.perf_counter() - t0
        
        tps_single = BENCH_STEPS / elapsed
        ms_single = elapsed / BENCH_STEPS * 1000
        print(f"  Single-GPU: {tps_single:.2f} tok/s ({ms_single:.2f} ms/tok)")
        results["single_gpu"] = {"tps": tps_single, "ms": ms_single}
        
        single_eng.cleanup()
        del single_eng
    except Exception as e:
        print(f"ERROR: Single-GPU benchmark failed: {e}")
        import traceback; traceback.print_exc()
        results["single_gpu"] = None
    
    # ====================================================================
    # Summary and Validation
    # ====================================================================
    print()
    print("=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Mode':<35} {'tok/s':>10} {'ms/tok':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    
    for mode in ["uncompressed", "compressed", "single_gpu"]:
        if results.get(mode):
            tps = results[mode]["tps"]
            ms = results[mode]["ms"]
            label = mode.replace("_", " ").title()
            print(f"  {label:<35} {tps:>10.2f} {ms:>10.2f}")
        else:
            print(f"  {mode.title():<35} {'N/A':>10} {'N/A':>10}")
    
    if results.get("correctness"):
        print(f"\n  Correctness:")
        print(f"    Cosine similarity: {results['correctness']['cosine_sim']:.6f}")
        print(f"    Max abs error: {results['correctness']['max_abs_err']:.6e}")
    
    # ====================================================================
    # Validation Assertions
    # ====================================================================
    print()
    print("=" * 72)
    print("  VALIDATION ASSERTIONS")
    print("=" * 72)
    
    # Target values from feature description
    TARGET_COMPRESSED_TPS = 53.74
    TARGET_UNCOMPRESSED_B1_TPS = 53.0
    TARGET_SINGLE_GPU_TPS = 21.0
    TARGET_COSINE_SIM = 0.99
    
    # Check compressed throughput
    if results.get("compressed"):
        tps_comp = results["compressed"]["tps"]
        comp_ok = tps_comp >= TARGET_COMPRESSED_TPS
        print(f"  Compressed throughput >= {TARGET_COMPRESSED_TPS:.2f} tok/s: "
              f"{tps_comp:.2f} - {'PASS' if comp_ok else 'FAIL'}")
    else:
        print(f"  Compressed throughput >= {TARGET_COMPRESSED_TPS:.2f} tok/s: FAIL (benchmark failed)")
    
    # Check uncompressed batch=1 (same as compressed test but FP16)
    if results.get("uncompressed"):
        tps_uncomp = results["uncompressed"]["tps"]
        uncomp_ok = tps_uncomp >= TARGET_UNCOMPRESSED_B1_TPS
        print(f"  Uncompressed (batch=1) >= {TARGET_UNCOMPRESSED_B1_TPS:.2f} tok/s: "
              f"{tps_uncomp:.2f} - {'PASS' if uncomp_ok else 'FAIL'}")
    else:
        print(f"  Uncompressed (batch=1) >= {TARGET_UNCOMPRESSED_B1_TPS:.2f} tok/s: FAIL (benchmark failed)")
    
    # Check single-GPU
    if results.get("single_gpu"):
        tps_single = results["single_gpu"]["tps"]
        single_ok = tps_single >= TARGET_SINGLE_GPU_TPS
        print(f"  Single-GPU >= {TARGET_SINGLE_GPU_TPS:.2f} tok/s: "
              f"{tps_single:.2f} - {'PASS' if single_ok else 'FAIL'}")
    else:
        print(f"  Single-GPU >= {TARGET_SINGLE_GPU_TPS:.2f} tok/s: FAIL (benchmark failed)")
    
    # Check correctness
    if results.get("correctness"):
        cos_sim = results["correctness"]["cosine_sim"]
        correct_ok = cos_sim >= TARGET_COSINE_SIM
        print(f"  Cosine similarity >= {TARGET_COSINE_SIM:.2f}: "
              f"{cos_sim:.6f} - {'PASS' if correct_ok else 'FAIL'}")
    else:
        print(f"  Cosine similarity >= {TARGET_COSINE_SIM:.2f}: FAIL (no data)")
    
    # Speedup calculation
    if results.get("compressed") and results.get("uncompressed"):
        speedup = results["compressed"]["tps"] / results["uncompressed"]["tps"]
        print(f"\n  Compressed vs Uncompressed speedup: {speedup:.3f}x")
    
    print("=" * 72)
    print("\nDone.")


if __name__ == "__main__":
    main()
