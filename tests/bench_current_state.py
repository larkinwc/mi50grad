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

    # Print active features
    eng0 = tp.engines[0]
    print(f"\n  Active features:")
    print(f"    GEMV v6: {getattr(eng0, '_gemv_int4_v6', False)}")
    print(f"    GEMV v5: {getattr(eng0, '_gemv_int4_v5', False)}")
    print(f"    Kernel P2P AR: {getattr(tp, '_kernel_p2p_allreduce', False)}")
    print(f"    C dispatch: {getattr(tp, '_c_dispatch_enabled', False)}")
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

    # Cleanup TP engine
    tp.cleanup()

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
        print(f"  {mode:<40} {tps:>10.2f} {ms:>10.2f}")
    print("=" * 72)
    print("\nDone.")


if __name__ == "__main__":
    main()
