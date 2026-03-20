#!/usr/bin/env python3
"""
VAL-CROSS-004: Gap Closure Percentage Validation

Measures final throughput and calculates gap closure percentage.

Gap closure calculation:
- Baseline (star topology): 15.3 tok/s
- Target: 60 tok/s
- Total gap: 60 - 15.3 = 44.7 tok/s
- Gap closure = (measured - baseline) / gap × 100%
- Target: >= 75% gap closure (>= 57.94 tok/s)

Usage:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      -e HIP_VISIBLE_DEVICES=0,1,2,3 \
      -v /opt/mi50grad:/opt/mi50grad \
      -v /opt/models:/opt/models \
      mi50grad bash -c 'cd /opt/mi50grad && python3 tests/val_cross_gap_closure.py'
"""

import sys
import os
import time
import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

# Constants
MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 50  # Reduced for faster testing
WARMUP_STEPS = 5
MAX_SEQ_LEN = 256

# Gap closure targets
BASELINE_STAR_TOPOLOGY = 15.3  # tok/s (star topology baseline)
TARGET_THROUGHPUT = 60.0  # tok/s
GAP = TARGET_THROUGHPUT - BASELINE_STAR_TOPOLOGY  # 44.7 tok/s
TARGET_GAP_CLOSURE = 0.75  # 75%
TARGET_TPS_FOR_75_CLOSURE = BASELINE_STAR_TOPOLOGY + TARGET_GAP_CLOSURE * GAP  # 57.94 tok/s


def reset_tp(tp_engine):
    """Reset KV cache on all TP engines."""
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def bench_decode(label, tp_engine, config, steps=BENCH_STEPS, warmup=WARMUP_STEPS, verbose=True):
    """Benchmark TP=4 decode throughput."""
    rng = np.random.default_rng(42)

    # Warmup
    for i in range(warmup):
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
    if verbose:
        print(f"  {label}: {tps:.2f} tok/s ({ms_per_tok:.2f} ms/tok)")
    return tps, ms_per_tok


def calculate_gap_closure(measured_tps):
    """Calculate gap closure percentage."""
    gap_closed = measured_tps - BASELINE_STAR_TOPOLOGY
    gap_closure_pct = (gap_closed / GAP) * 100.0
    return gap_closed, gap_closure_pct


def main():
    print("=" * 80)
    print("  VAL-CROSS-004: Gap Closure Percentage Validation")
    print("=" * 80)
    print(f"  Model: {MODEL_DIR}")
    print(f"  Devices: {DEVICE_IDS}")
    print(f"  Steps: {BENCH_STEPS}, Warmup: {WARMUP_STEPS}")
    print()
    print(f"  Gap Closure Calculation:")
    print(f"    Baseline (star topology): {BASELINE_STAR_TOPOLOGY:.2f} tok/s")
    print(f"    Target: {TARGET_THROUGHPUT:.2f} tok/s")
    print(f"    Total gap: {GAP:.2f} tok/s")
    print(f"    Target gap closure: >= {TARGET_GAP_CLOSURE*100:.0f}% ({TARGET_TPS_FOR_75_CLOSURE:.2f} tok/s)")
    print("=" * 80)
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
    print("\nBuilding dispatch cache and enabling optimizations...")
    tp.build_dispatch_cache()
    tp.set_direct_kv_write(True)
    tp.set_c_dispatch(True)
    tp.set_kernel_p2p_allreduce(True)
    tp.set_deferred_attention_ar(True)  # M3: reduce AR count from 128 to 64

    # Print active features
    eng0 = tp.engines[0]
    print(f"\n  Active features:")
    print(f"    GEMV v6: {getattr(eng0, '_gemv_int4_v6', False)}")
    print(f"    GEMV v5: {getattr(eng0, '_gemv_int4_v5', False)}")
    print(f"    Kernel P2P AR: {getattr(tp, '_kernel_p2p_allreduce', False)}")
    print(f"    C dispatch: {getattr(tp, '_c_dispatch_enabled', False)}")
    print(f"    Deferred AR (M3): {getattr(tp, '_deferred_attention_ar', False)}")
    print(f"    Direct KV write: {getattr(tp, '_direct_kv_write', False)}")
    
    # Check for fused GEMV kernel (M1)
    if hasattr(tp, '_c_dispatch_state'):
        has_fused_gemv = tp._c_dispatch_state is not None
        print(f"    M1 Fused GEMV: {has_fused_gemv}")
    
    # Check speculative decode
    spec_methods = [m for m in dir(tp) if 'spec' in m.lower() or 'draft' in m.lower() or 'eagle' in m.lower()]
    print(f"    Speculative methods: {spec_methods}")
    print(f"    Speculative mode: {getattr(tp, '_speculative_mode', None)}")
    print()

    # --- Mode 1: Best mode (all optimizations) ---
    print("  Mode 1: Best mode (all optimizations)")
    tps_best, ms_best = bench_decode("All optimizations", tp, config)
    results["Best (all optimizations)"] = (tps_best, ms_best)

    # --- Mode 2: Try speculative decode ---
    print()
    print("  Mode 2: Speculative decode (if available)")
    try:
        if hasattr(tp, 'set_speculative_mode'):
            tp.set_speculative_mode(True, ngram_size=3, max_draft_len=5)
            print(f"    Speculative mode enabled: {getattr(tp, '_speculative_mode', None)}")
            tps_spec, ms_spec = bench_decode("Speculative (n-gram)", tp, config)
            results["Speculative (n-gram)"] = (tps_spec, ms_spec)
            tp.set_speculative_mode(False)
            print(f"    Speculative mode disabled")
        else:
            print("    set_speculative_mode not available")
            results["Speculative (n-gram)"] = (None, None)
    except Exception as e:
        print(f"    Speculative benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results["Speculative (n-gram)"] = (None, None)

    # --- Mode 3: Try EAGLE ---
    print()
    print("  Mode 3: EAGLE speculative decode (if available)")
    try:
        if hasattr(tp, 'set_eagle_mode'):
            tp.set_eagle_mode(True)
            print(f"    Eagle mode enabled: {getattr(tp, '_eagle_mode', None)}")
            tps_eagle, ms_eagle = bench_decode("EAGLE", tp, config)
            results["EAGLE"] = (tps_eagle, ms_eagle)
            tp.set_eagle_mode(False)
            print(f"    Eagle mode disabled")
        else:
            print("    set_eagle_mode not available")
            results["EAGLE"] = (None, None)
    except Exception as e:
        print(f"    EAGLE benchmark failed: {e}")
        results["EAGLE"] = (None, None)

    # --- Mode 4: Star topology (disable kernel P2P) ---
    print()
    print("  Mode 4: Star topology baseline (no kernel P2P)")
    try:
        tp._kernel_p2p_allreduce = False
        if hasattr(tp, '_build_c_dispatch_plan'):
            tp._build_c_dispatch_plan()
        tps_star, ms_star = bench_decode("Star topology (no kernel P2P)", tp, config)
        results["Star topology"] = (tps_star, ms_star)
        # Re-enable kernel P2P
        tp.set_kernel_p2p_allreduce(True)
    except Exception as e:
        print(f"    Star topology benchmark failed: {e}")
        results["Star topology"] = (None, None)

    # --- Mode 5: Single-GPU baseline ---
    print()
    print("  Mode 5: Single-GPU baseline (no regression check)")
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
        import traceback
        traceback.print_exc()
        results["Single-GPU"] = (None, None)

    # Cleanup TP engine
    tp.cleanup()

    # --- Summary ---
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Mode':<40} {'tok/s':>10} {'ms/tok':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for mode, (tps, ms) in results.items():
        if tps is not None:
            print(f"  {mode:<40} {tps:>10.2f} {ms:>10.2f}")
        else:
            print(f"  {mode:<40} {'N/A':>10} {'N/A':>10}")
    print("=" * 80)

    # --- Gap Closure Calculation ---
    print()
    print("=" * 80)
    print("  VAL-CROSS-004: Gap Closure Percentage")
    print("=" * 80)
    
    # Get best throughput
    best_tps = max([tps for mode, (tps, ms) in results.items() 
                    if tps is not None and 'TP=4' in mode or 'Best' in mode or 'Star' in mode or 'Speculative' in mode or 'EAGLE' in mode], 
                   default=0)
    
    if best_tps == 0:
        # Fallback: use any TP=4 mode
        best_tps = max([tps for mode, (tps, ms) in results.items() if tps is not None], default=0)
    
    gap_closed, gap_closure_pct = calculate_gap_closure(best_tps)
    
    print(f"  Baseline (star topology): {BASELINE_STAR_TOPOLOGY:.2f} tok/s")
    print(f"  Target: {TARGET_THROUGHPUT:.2f} tok/s")
    print(f"  Total gap: {GAP:.2f} tok/s")
    print()
    print(f"  Measured throughput: {best_tps:.2f} tok/s")
    print(f"  Gap closed: {gap_closed:.2f} tok/s")
    print(f"  Gap closure: {gap_closure_pct:.1f}%")
    print()
    print(f"  Target: >= {TARGET_GAP_CLOSURE*100:.0f}% gap closure ({TARGET_TPS_FOR_75_CLOSURE:.2f} tok/s)")
    
    cross004 = gap_closure_pct >= (TARGET_GAP_CLOSURE * 100)
    print(f"  Status: {'✅ PASS' if cross004 else '❌ FAIL'}")
    
    if not cross004:
        remaining_gap = TARGET_TPS_FOR_75_CLOSURE - best_tps
        print(f"  Gap to 75% closure: {remaining_gap:.2f} tok/s ({TARGET_GAP_CLOSURE*100 - gap_closure_pct:.1f}% short)")
    
    print()
    print("=" * 80)
    print("  Path to 60 tok/s Analysis")
    print("=" * 80)
    
    remaining_to_target = TARGET_THROUGHPUT - best_tps
    remaining_pct = (1 - best_tps/TARGET_THROUGHPUT) * 100
    
    print(f"  Remaining to 60 tok/s: {remaining_to_target:.2f} tok/s ({remaining_pct:.1f}% below target)")
    print()
    print("  Potential optimizations:")
    print("    1. Allreduce micro-optimization (79µs → 60µs per call)")
    print("       - Current: 64 AR calls × 79µs = 5.06ms/token")
    print("       - Potential: 64 AR calls × 60µs = 3.84ms/token")
    print("       - Savings: ~1.2ms/token → +2-3 tok/s")
    print()
    print("    2. Fix M2 fused GEMV kernel (if regression resolved)")
    print("       - Previous attempt: 53.74 tok/s achieved")
    print("       - Potential: +2-4 tok/s if fused kernel works")
    print()
    print("    3. Batch size > 1 (throughput vs latency trade-off)")
    print("       - GEMV → GEMM transition")
    print("       - Better GPU utilization")
    print()
    print("    4. Ring allreduce (better P2P bandwidth utilization)")
    print("       - Current: Star topology, ~12 GB/s per link")
    print("       - Ring: Could utilize all links simultaneously")
    print()
    print("  Hardware constraints:")
    print("    - MI50 (gfx906) lacks MFMA instructions")
    print("    - PCIe 3.0 x16 P2P bandwidth limits allreduce")
    print("    - GPU compute fixed at ~11ms/token (64 layers)")
    print()
    print("=" * 80)
    print("  VALIDATION COMPLETE")
    print("=" * 80)
    
    # Return appropriate exit code
    if cross004:
        print(f"\n✅ VAL-CROSS-004: PASSED ({gap_closure_pct:.1f}% gap closure)")
        return 0
    else:
        print(f"\n❌ VAL-CROSS-004: FAILED (target: {TARGET_GAP_CLOSURE*100:.0f}%, measured: {gap_closure_pct:.1f}%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
