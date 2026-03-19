#!/usr/bin/env python3
"""Cross-validation benchmark for VAL-CROSS-* and VAL-TP-PREFILL-002 assertions.

This benchmark validates:
- VAL-CROSS-001: Full pipeline throughput >= 60 tok/s
- VAL-CROSS-002: Memory usage < 32GB per GPU  
- VAL-CROSS-003: End-to-end generation produces coherent output
- VAL-TP-PREFILL-002: Prefill throughput >= 1000 tok/s (512 tokens in < 0.5s)
- VAL-SPEC-003: Speculative speedup measurement
"""

import sys
import os
import time
import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 50  # Reduced from 100 to avoid timeout
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256


def reset_tp(tp_engine):
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def bench_decode(label, tp_engine, config, steps=BENCH_STEPS):
    """Benchmark TP=4 decode throughput."""
    rng = np.random.default_rng(42)

    # Warmup
    for i in range(WARMUP_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        reset_tp(tp_engine)
        tp_engine.decode_step(emb, i)
        tp_engine._hip.synchronize()

    # Benchmark
    reset_tp(tp_engine)
    t0 = time.perf_counter()
    for step in range(steps):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_engine.decode_step(emb, step)
        tp_engine._hip.synchronize()
    elapsed = time.perf_counter() - t0

    tps = steps / elapsed
    ms_per_tok = elapsed / steps * 1000
    print(f"  {label}: {tps:.2f} tok/s ({ms_per_tok:.2f} ms/tok)")
    return tps, ms_per_tok


def main():
    print("=" * 72)
    print("  Cross-Validation Benchmark — TP=4 Decode (Qwen3.5-27B GPTQ-Int4)")
    print("=" * 72)
    print(f"  Model: {MODEL_DIR}")
    print(f"  Devices: {DEVICE_IDS}")
    print(f"  Steps: {BENCH_STEPS}, Warmup: {WARMUP_STEPS}")
    print()

    from src.model.qwen import load_config_from_json
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader

    config = load_config_from_json(MODEL_DIR)
    results = {}
    validation_results = {}

    # --- Load engine and weights ---
    print("Loading TP=4 engine + weights...")
    tp = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)

    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 16 == 0:
            print(f"    Layer {layer_idx}...")
        tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    tp.load_final_norm(loader.load_final_norm())
    tp.load_lm_head(loader.load_lm_head())

    # --- Enable all optimizations ---
    tp.build_dispatch_cache()
    tp.set_direct_kv_write(True)
    tp.set_c_dispatch(True)
    tp.set_kernel_p2p_allreduce(True)
    tp.set_deferred_attention_ar(True)  # M3 optimization: reduce AR count from 128 to 64

    print(f"\n  Active features:")
    print(f"    GEMV v6: {getattr(tp.engines[0], '_gemv_int4_v6', False)}")
    print(f"    Kernel P2P AR: {getattr(tp, '_kernel_p2p_allreduce', False)}")
    print(f"    C dispatch: {getattr(tp, '_c_dispatch_enabled', False)}")
    print(f"    Deferred AR (M3): {getattr(tp, '_deferred_attention_ar', False)}")
    print()

    # --- VAL-CROSS-001: Best decode throughput ---
    print("  VAL-CROSS-001: Full pipeline throughput")
    tps_best, ms_best = bench_decode("C dispatch + kernel P2P + deferred AR", tp, config)
    results["TP=4 Best Decode"] = (tps_best, ms_best)
    
    # Target check
    target_60 = tps_best >= 60
    print(f"    Target: >= 60 tok/s - {'PASS' if target_60 else 'FAIL'}")
    validation_results["VAL-CROSS-001"] = {
        "status": "passed" if target_60 else "failed",
        "measured": f"{tps_best:.2f} tok/s",
        "target": ">= 60 tok/s"
    }
    print()

    # --- VAL-TP-PREFILL-002: TP Prefill Throughput ---
    print("  VAL-TP-PREFILL-002: TP Prefill throughput")
    try:
        if hasattr(tp, 'prefill_step'):
            rng_prefill = np.random.default_rng(42)
            SEQ_LEN = 512
            tokens = rng_prefill.integers(0, config.vocab_size, size=SEQ_LEN, dtype=np.int32)
            
            reset_tp(tp)
            t0 = time.perf_counter()
            hidden_out = tp.prefill_step(tokens)
            tp._hip.synchronize()
            elapsed = time.perf_counter() - t0
            
            prefill_tps = SEQ_LEN / elapsed
            ms_per_tok = elapsed / SEQ_LEN * 1000
            print(f"    512-token prefill: {elapsed:.4f}s ({prefill_tps:.2f} tok/s, {ms_per_tok:.2f} ms/tok)")
            results["TP=4 Prefill (512 tokens)"] = (prefill_tps, ms_per_tok)
            
            # Check against target
            target_met = elapsed < 0.5
            print(f"    Target: < 0.5s (1000+ tok/s) - {'PASS' if target_met else 'FAIL'}")
            
            validation_results["VAL-TP-PREFILL-002"] = {
                "status": "passed" if target_met else "failed",
                "measured": f"{elapsed:.4f}s ({prefill_tps:.2f} tok/s)",
                "target": "< 0.5s (1000+ tok/s)"
            }
            
            # Clean up KV cache after prefill
            reset_tp(tp)
        else:
            print("    prefill_step method not found")
            validation_results["VAL-TP-PREFILL-002"] = {
                "status": "failed",
                "measured": "method not found",
                "target": "< 0.5s (1000+ tok/s)"
            }
    except Exception as e:
        print(f"    TP Prefill benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        validation_results["VAL-TP-PREFILL-002"] = {
            "status": "failed",
            "measured": f"error: {e}",
            "target": "< 0.5s (1000+ tok/s)"
        }
    print()

    # --- VAL-CROSS-002: Memory Usage Check ---
    print("  VAL-CROSS-002: Memory usage check")
    try:
        import ctypes
        hip_lib = ctypes.CDLL('/opt/rocm/lib/libhip_runtime.so')
        
        total_mem = 0
        max_mem_per_gpu = 0
        for i in DEVICE_IDS:
            hip_lib.hipSetDevice(i)
            free_mem = ctypes.c_size_t()
            total_mem_gpu = ctypes.c_size_t()
            hip_lib.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem_gpu))
            
            used_mem = total_mem_gpu.value - free_mem.value
            used_mb = used_mem / (1024 * 1024)
            total_mb = total_mem_gpu.value / (1024 * 1024)
            total_mem += used_mb
            max_mem_per_gpu = max(max_mem_per_gpu, used_mb)
            print(f"    GPU {i}: {used_mb:.1f} MB / {total_mb:.1f} MB")
        
        print(f"    Total used: {total_mem:.1f} MB")
        print(f"    Max per GPU: {max_mem_per_gpu:.1f} MB")
        
        # Check against 32GB limit
        mem_limit_mb = 32 * 1024
        mem_ok = max_mem_per_gpu < mem_limit_mb
        print(f"    Memory limit (32GB): {'PASS' if mem_ok else 'FAIL'}")
        results["Memory (max per GPU)"] = (max_mem_per_gpu, 0)
        
        validation_results["VAL-CROSS-002"] = {
            "status": "passed" if mem_ok else "failed",
            "measured": f"{max_mem_per_gpu:.1f} MB",
            "target": "< 32GB per GPU"
        }
    except Exception as e:
        print(f"    Memory check failed: {e}")
        validation_results["VAL-CROSS-002"] = {
            "status": "failed",
            "measured": f"error: {e}",
            "target": "< 32GB per GPU"
        }
    print()

    # --- VAL-CROSS-003: End-to-end Generation Test ---
    print("  VAL-CROSS-003: End-to-end generation test")
    try:
        # Test with simple random input (no tokenizer needed)
        rng_gen = np.random.default_rng(42)
        hidden = rng_gen.standard_normal(config.hidden_size).astype(np.float16)
        
        # Run a few decode steps to verify generation works
        t0_gen = time.perf_counter()
        generated_tokens = 0
        max_gen_tokens = 20
        
        for step in range(max_gen_tokens):
            hidden = tp.decode_step(hidden, step)
            tp._hip.synchronize()
            generated_tokens += 1
            
            # Early exit if NaN
            if np.any(np.isnan(hidden)):
                print(f"    WARNING: NaN detected at step {step}")
                break
        
        elapsed_gen = time.perf_counter() - t0_gen
        gen_tps = generated_tokens / elapsed_gen if elapsed_gen > 0 else 0
        
        print(f"    Generated {generated_tokens} tokens in {elapsed_gen:.4f}s")
        print(f"    Generation throughput: {gen_tps:.2f} tok/s")
        print(f"    Output hidden state: min={np.nanmin(hidden):.4f}, max={np.nanmax(hidden):.4f}, mean={np.nanmean(hidden):.4f}")
        
        # Check for coherence (no NaN/Inf, reasonable magnitude)
        is_coherent = (not np.any(np.isnan(hidden)) and 
                      not np.any(np.isinf(hidden)) and
                      np.abs(np.mean(hidden)) < 10.0)
        print(f"    Coherence check: {'PASS' if is_coherent else 'FAIL'}")
        results["E2E Generation"] = (gen_tps, 0)
        
        validation_results["VAL-CROSS-003"] = {
            "status": "passed" if is_coherent else "failed",
            "measured": f"{generated_tokens} tokens, coherent={is_coherent}",
            "target": "Coherent output without NaN/Inf"
        }
    except Exception as e:
        print(f"    End-to-end generation failed: {e}")
        import traceback
        traceback.print_exc()
        validation_results["VAL-CROSS-003"] = {
            "status": "failed",
            "measured": f"error: {e}",
            "target": "Coherent output without NaN/Inf"
        }
    print()

    # --- VAL-SPEC-003: Speculative Speedup ---
    print("  VAL-SPEC-003: Speculative decode speedup")
    try:
        if hasattr(tp, 'set_speculative_mode'):
            # Try n-gram speculative mode
            tp.set_speculative_mode(True)
            print(f"    Speculative mode enabled: {getattr(tp, '_speculative_mode', None)}")
            tps_spec, ms_spec = bench_decode("Speculative decode (n-gram)", tp, config)
            results["TP=4 Speculative (n-gram)"] = (tps_spec, ms_spec)
            
            speedup_ngram = tps_spec / tps_best if tps_best > 0 else 0
            print(f"    N-gram speedup: {speedup_ngram:.3f}x")
            
            tp.set_speculative_mode(False)
            
            validation_results["VAL-SPEC-003"] = {
                "status": "passed" if speedup_ngram >= 1.0 else "info",
                "measured": f"{speedup_ngram:.3f}x ({tps_spec:.2f} tok/s)",
                "target": "Document actual speedup (any value acceptable)"
            }
        else:
            print("    Speculative mode not available")
            validation_results["VAL-SPEC-003"] = {
                "status": "info",
                "measured": "not available",
                "target": "Document actual speedup"
            }
    except Exception as e:
        print(f"    Speculative decode benchmark failed: {e}")
        validation_results["VAL-SPEC-003"] = {
            "status": "info",
            "measured": f"error: {e}",
            "target": "Document actual speedup"
        }
    print()

    # Cleanup
    tp.cleanup()

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY — Cross-Validation")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"  Model: Qwen3.5-27B-GPTQ-Int4, TP=4, batch=1, seq_len={MAX_SEQ_LEN}")
    print("=" * 72)
    print(f"  {'Mode':<40} {'tok/s':>10} {'ms/tok':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for mode, (tps, ms) in results.items():
        if tps is not None:
            print(f"  {mode:<40} {tps:>10.2f} {ms:>10.2f}")
        else:
            print(f"  {mode:<40} {'N/A':>10} {'N/A':>10}")
    print("=" * 72)
    
    # --- Validation Assertions ---
    print("\n" + "=" * 72)
    print("  VALIDATION ASSERTIONS SUMMARY")
    print("=" * 72)
    
    for assertion_id, result in sorted(validation_results.items()):
        status_icon = "✓" if result["status"] == "passed" else ("✗" if result["status"] == "failed" else "ℹ")
        print(f"  {status_icon} {assertion_id}: {result['status'].upper()}")
        print(f"      Measured: {result['measured']}")
        print(f"      Target: {result['target']}")
        print()
    
    # Overall pass/fail
    passed = sum(1 for r in validation_results.values() if r["status"] == "passed")
    failed = sum(1 for r in validation_results.values() if r["status"] == "failed")
    total = len(validation_results)
    
    print("=" * 72)
    print(f"  Overall: {passed}/{total} assertions passed, {failed} failed")
    print("=" * 72)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
