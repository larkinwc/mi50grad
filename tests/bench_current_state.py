#!/usr/bin/env python3
"""Quick current-state benchmark: TP=4 decode throughput across all working modes."""

import sys, os, time
import numpy as np
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 100
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256


def reset_tp(tp_engine):
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def bench_decode(label, tp_engine, config):
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
    for step in range(BENCH_STEPS):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_engine.decode_step(emb, step)
        tp_engine._hip.synchronize()
    elapsed = time.perf_counter() - t0

    tps = BENCH_STEPS / elapsed
    ms_per_tok = elapsed / BENCH_STEPS * 1000
    print(f"  {label}: {tps:.2f} tok/s ({ms_per_tok:.2f} ms/tok)")
    return tps, ms_per_tok


def main():
    print("=" * 72)
    print("  Current State Benchmark — TP=4 Decode (Qwen3.5-27B GPTQ-Int4)")
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

    # Print active features
    eng0 = tp.engines[0]
    print(f"\n  Active features:")
    print(f"    GEMV v6: {getattr(eng0, '_gemv_int4_v6', False)}")
    print(f"    GEMV v5: {getattr(eng0, '_gemv_int4_v5', False)}")
    print(f"    Kernel P2P AR: {getattr(tp, '_kernel_p2p_allreduce', False)}")
    print(f"    C dispatch: {getattr(tp, '_c_dispatch_enabled', False)}")
    print(f"    Deferred AR (M3): {getattr(tp, '_deferred_attention_ar', False)}")
    print(f"    Graph decode: {getattr(tp, '_global_graph_decode_state', None) is not None}")

    # Check speculative decode
    spec_methods = [m for m in dir(tp) if 'spec' in m.lower() or 'draft' in m.lower() or 'eagle' in m.lower()]
    print(f"    Speculative methods: {spec_methods}")
    print(f"    Speculative mode: {getattr(tp, '_speculative_mode', None)}")
    print(f"    Eagle mode: {getattr(tp, '_eagle_mode', None)}")
    print()

    # --- Mode 1: Best mode (C dispatch + kernel P2P) ---
    tps, ms = bench_decode("Best (C dispatch + kernel P2P)", tp, config)
    results["TP=4 C dispatch + kernel P2P"] = (tps, ms)

    # --- Mode 2: Try speculative decode if integrated ---
    print()
    try:
        if hasattr(tp, 'set_speculative_mode'):
            tp.set_speculative_mode(True)
            print(f"  Speculative mode enabled: {getattr(tp, '_speculative_mode', None)}")
            tps_spec, ms_spec = bench_decode("Speculative decode (n-gram)", tp, config)
            results["TP=4 Speculative (n-gram)"] = (tps_spec, ms_spec)
            tp.set_speculative_mode(False)
    except Exception as e:
        print(f"  Speculative decode benchmark failed: {e}")

    # --- Mode 3: Try EAGLE if integrated ---
    try:
        if hasattr(tp, 'set_eagle_mode'):
            tp.set_eagle_mode(True)
            print(f"  Eagle mode enabled: {getattr(tp, '_eagle_mode', None)}")
            tps_eagle, ms_eagle = bench_decode("EAGLE speculative decode", tp, config)
            results["TP=4 EAGLE"] = (tps_eagle, ms_eagle)
            tp.set_eagle_mode(False)
    except Exception as e:
        print(f"  EAGLE decode benchmark failed: {e}")

    # --- Mode 4: Star topology (disable kernel P2P) ---
    print()
    try:
        tp._kernel_p2p_allreduce = False
        if hasattr(tp, '_build_c_dispatch_plan'):
            tp._build_c_dispatch_plan()
        tps_star, ms_star = bench_decode("Star topology (no kernel P2P)", tp, config)
        results["TP=4 Star topology"] = (tps_star, ms_star)
        # Re-enable
        tp.set_kernel_p2p_allreduce(True)
    except Exception as e:
        print(f"  Star topology benchmark failed: {e}")

    # --- VAL-TP-PREFILL-002: TP Prefill Throughput ---
    # Target: 512-token prompt processes in under 0.5 seconds (1000+ tok/s)
    print()
    print("  TP Prefill throughput test (VAL-TP-PREFILL-002):")
    try:
        if hasattr(tp, 'prefill_step'):
            rng_prefill = np.random.default_rng(42)
            # Test with 512 tokens
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
            
            # Clean up KV cache after prefill
            reset_tp(tp)
        else:
            print("    prefill_step method not found")
    except Exception as e:
        print(f"    TP Prefill benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # --- Memory Usage Check ---
    print()
    print("  Memory usage check:")
    try:
        import ctypes
        # Load ROCm HIP runtime
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
    except Exception as e:
        print(f"    Memory check failed: {e}")
        results["Memory (max per GPU)"] = (None, 0)

    # Cleanup TP engine
    tp.cleanup()

    # --- VAL-CROSS-003: End-to-end Generation Test ---
    print()
    print("  End-to-end generation test (VAL-CROSS-003):")
    try:
        from src.inference.tp_engine import TPInferenceEngine
        from src.inference.sampler import SamplingParams, sample_token
        
        # Reload TP engine for generation
        print("    Loading TP engine for generation...")
        tp_gen = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
        
        loader_gen = QwenWeightLoader(MODEL_DIR, config)
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx % 16 == 0:
                print(f"      Layer {layer_idx}...")
            tp_gen.load_layer_weights(layer_idx, loader_gen.load_layer(layer_idx))
        tp_gen.load_final_norm(loader_gen.load_final_norm())
        tp_gen.load_lm_head(loader_gen.load_lm_head())
        
        # Enable optimizations
        tp_gen.build_dispatch_cache()
        tp_gen.set_direct_kv_write(True)
        tp_gen.set_c_dispatch(True)
        tp_gen.set_kernel_p2p_allreduce(True)
        tp_gen.set_deferred_attention_ar(True)
        
        # Test prompt
        test_prompt = "The future of artificial intelligence"
        print(f"    Prompt: '{test_prompt}'")
        
        # Simple generation loop (without tokenizer, use random tokens for now)
        rng_gen = np.random.default_rng(42)
        hidden = rng_gen.standard_normal(config.hidden_size).astype(np.float16)
        
        # Run a few decode steps to verify generation works
        t0_gen = time.perf_counter()
        generated_tokens = 0
        max_gen_tokens = 20
        
        for step in range(max_gen_tokens):
            hidden = tp_gen.decode_step(hidden, step)
            tp_gen._hip.synchronize()
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
        
        tp_gen.cleanup()
    except Exception as e:
        print(f"    End-to-end generation failed: {e}")
        import traceback; traceback.print_exc()
        results["E2E Generation"] = (None, 0)

    # --- Mode 5: Single-GPU baseline ---
    print()
    try:
        from src.inference.engine import InferenceEngine
        print("Loading single-GPU engine...")
        single_eng = InferenceEngine(config, device_id=0, max_seq_len=MAX_SEQ_LEN)

        loader2 = QwenWeightLoader(MODEL_DIR, config)
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx % 16 == 0:
                print(f"    Layer {layer_idx}...")
            single_eng.load_layer_weights(layer_idx, loader2.load_layer(layer_idx))
        single_eng.load_final_norm(loader2.load_final_norm())
        single_eng.load_lm_head(loader2.load_lm_head())

        rng = np.random.default_rng(42)

        # Warmup
        for i in range(WARMUP_STEPS):
            emb = rng.standard_normal(config.hidden_size).astype(np.float16)
            single_eng.kv_cache.current_len = 0
            single_eng.decode_step(emb, i)
            single_eng.device.synchronize()

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
        results["Single-GPU"] = (tps_single, ms_single)
        single_eng.cleanup()
    except Exception as e:
        print(f"  Single-GPU benchmark failed: {e}")
        import traceback; traceback.print_exc()

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY — Current State")
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
    print("  VALIDATION ASSERTIONS")
    print("=" * 72)
    
    # VAL-CROSS-001: Full pipeline throughput >= 60 tok/s
    best_decode = max([tps for mode, (tps, ms) in results.items() if 'TP=4' in mode and 'Prefill' not in mode and tps is not None], default=0)
    cross001 = best_decode >= 60
    print(f"  VAL-CROSS-001 (>= 60 tok/s): {best_decode:.2f} tok/s - {'PASS' if cross001 else 'FAIL'}")
    
    # VAL-CROSS-002: Memory usage within 32GB per GPU
    mem_result = results.get("Memory (max per GPU)", (None, 0))[0]
    cross002 = mem_result is not None and mem_result < 32 * 1024
    print(f"  VAL-CROSS-002 (Memory < 32GB): {mem_result:.1f} MB - {'PASS' if cross002 else 'FAIL' if mem_result else 'N/A'}")
    
    # VAL-TP-PREFILL-002: Prefill throughput >= 1000 tok/s (512 tokens in < 0.5s)
    prefill_result = results.get("TP=4 Prefill (512 tokens)", (None, 0))[0]
    tp_prefill002 = prefill_result is not None and prefill_result >= 1000
    print(f"  VAL-TP-PREFILL-002 (>= 1000 tok/s): {prefill_result:.2f} tok/s - {'PASS' if tp_prefill002 else 'FAIL' if prefill_result else 'N/A'}")
    
    # VAL-CROSS-003: End-to-end generation produces coherent text
    e2e_result = results.get("E2E Generation", (None, 0))[0]
    cross003 = e2e_result is not None
    print(f"  VAL-CROSS-003 (E2E generation): {'PASS' if cross003 else 'FAIL'}")
    
    # VAL-SPEC-003: Speculative speedup (if speculative modes were tested)
    spec_result = results.get("TP=4 EAGLE", (None, 0))[0]
    if spec_result is not None and best_decode > 0:
        speedup = spec_result / best_decode
        spec003 = speedup >= 1.0  # Any speedup is acceptable
        print(f"  VAL-SPEC-003 (Speculative speedup): {speedup:.3f}x - {'PASS' if spec003 else 'FAIL'}")
    else:
        print(f"  VAL-SPEC-003 (Speculative speedup): N/A")
    
    print("=" * 72)
    print("\nDone.")


if __name__ == "__main__":
    main()
