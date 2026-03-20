#!/usr/bin/env python3
"""
VAL-CROSS-001: Stacked M1+M2 Throughput Validation

Tests that M1 (fused GEMV) and M2 (speculative decode) optimizations 
stack correctly and achieve >= 55 tok/s throughput.

M1 (fused GEMV): gemv_int4_p2p_allreduce_rmsnorm kernel
M2 (speculative decode): n-gram lookahead speculation

Usage:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
      -e HIP_VISIBLE_DEVICES=0,1,2,3 \
      -v /opt/mi50grad:/opt/mi50grad \
      -v /opt/models:/opt/models \
      mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_cross_stacked_optim.py'
"""

import sys
import time
import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
BENCH_STEPS = 50  # Reduced for faster testing
WARMUP_STEPS = 3
MAX_SEQ_LEN = 256


def reset_tp(tp_engine):
    for eng in tp_engine.engines:
        eng.kv_cache.current_len = 0


def bench_decode(label, tp_engine, config, steps=BENCH_STEPS, warmup=WARMUP_STEPS):
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
    print(f"  {label}: {tps:.2f} tok/s ({ms_per_tok:.2f} ms/tok)")
    return tps, ms_per_tok


def main():
    print("=" * 80)
    print("  VAL-CROSS-001: Stacked M1+M2 Throughput Validation")
    print("=" * 80)
    print(f"  Model: {MODEL_DIR}")
    print(f"  Devices: {DEVICE_IDS}")
    print(f"  Steps: {BENCH_STEPS}, Warmup: {WARMUP_STEPS}")
    print()
    
    print("  M1 (fused GEMV): gemv_int4_p2p_allreduce_rmsnorm kernel")
    print("  M2 (speculative decode): n-gram lookahead")
    print("  Target: >= 55 tok/s with both enabled")
    print("=" * 80)
    print()

    from src.model.qwen import load_config_from_json
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader

    config = load_config_from_json(MODEL_DIR)

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
    tp.set_deferred_attention_ar(True)  # M3 optimization

    # Check which optimizations are active
    print(f"\n  Active features after setup:")
    print(f"    C dispatch: {getattr(tp, '_c_dispatch_enabled', False)}")
    print(f"    Kernel P2P AR: {getattr(tp, '_kernel_p2p_allreduce', False)}")
    print(f"    Deferred AR (M3): {getattr(tp, '_deferred_attention_ar', False)}")
    print(f"    Direct KV write: {getattr(tp, '_direct_kv_write', False)}")
    
    # Check for fused GEMV kernel
    print(f"\n  M1 (fused GEMV) status:")
    if hasattr(tp, '_c_dispatch_state'):
        # Check if gemv_fused_tp4_fn is set in the dispatch plan
        has_fused_gemv = tp._c_dispatch_state is not None
        print(f"    C dispatch state available: {has_fused_gemv}")
    
    # Check speculative decode methods
    print(f"\n  M2 (speculative decode) status:")
    spec_methods = [m for m in dir(tp) if 'spec' in m.lower() or 'draft' in m.lower()]
    print(f"    Speculative methods available: {spec_methods}")
    print(f"    set_speculative_mode: {hasattr(tp, 'set_speculative_mode')}")
    print(f"    set_eagle_mode: {hasattr(tp, 'set_eagle_mode')}")
    print()

    results = {}
    
    # --- Mode 1: Baseline with M3 deferred AR only ---
    print("  Mode 1: Baseline (C dispatch + kernel P2P + deferred AR)")
    tps_baseline, ms_baseline = bench_decode("Baseline (M3 deferred AR)", tp, config)
    results["Baseline (M3)"] = (tps_baseline, ms_baseline)
    print()

    # --- Mode 2: Enable speculative decode (M2) ---
    print("  Mode 2: M2 Speculative Decode (n-gram)")
    try:
        if hasattr(tp, 'set_speculative_mode'):
            tp.set_speculative_mode(True, ngram_size=3, max_draft_len=5)
            print(f"    Speculative mode enabled: {getattr(tp, '_speculative_mode', None)}")
            tps_spec, ms_spec = bench_decode("M2 Speculative (n-gram)", tp, config)
            results["M2 Speculative"] = (tps_spec, ms_spec)
            tp.set_speculative_mode(False)
            print(f"    Speculative mode disabled")
        else:
            print("    set_speculative_mode not available")
            results["M2 Speculative"] = (None, None)
    except Exception as e:
        print(f"    Speculative benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results["M2 Speculative"] = (None, None)
    print()

    # --- Mode 3: Check if fused GEMV is enabled in C dispatch ---
    print("  Mode 3: Checking M1 (fused GEMV) kernel status...")
    # The fused GEMV kernel should be automatically enabled if:
    # 1. The .so file exists (we verified it does)
    # 2. TP=4 is used
    # 3. C dispatch is built
    # Check the C dispatch plan
    if hasattr(tp, '_c_dispatch_plan'):
        # Try to inspect the plan structure
        plan = tp._c_dispatch_plan
        print(f"    C dispatch plan exists: {plan is not None}")
        if plan:
            # Check ffn_allreduce_specs for gemv_fused usage
            # This is internal, but we can check if the library was loaded
            if hasattr(tp, '_c_dispatch_objects') and 'gemv_fused_lib' in tp._c_dispatch_objects:
                print(f"    M1 fused GEMV kernel library loaded: True")
                print(f"    M1 fused GEMV should be active in FFN down-projection")
            else:
                print(f"    M1 fused GEMV kernel library loaded: False")
    else:
        print(f"    C dispatch plan not directly accessible")
    
    # Run another baseline to ensure consistency
    print()
    print("  Mode 4: Second baseline run (verify consistency)")
    tps_baseline2, ms_baseline2 = bench_decode("Baseline (re-run)", tp, config)
    results["Baseline (re-run)"] = (tps_baseline2, ms_baseline2)
    print()

    # --- Summary ---
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
    
    # --- VAL-CROSS-001 Validation ---
    print()
    print("=" * 80)
    print("  VAL-CROSS-001: Stacked M1+M2 Throughput Validation")
    print("=" * 80)
    
    # Get the best throughput from speculative mode
    spec_result = results.get("M2 Speculative", (None, None))[0]
    baseline_result = results.get("Baseline (M3)", (None, None))[0]
    
    target_tps = 55.0
    
    if spec_result is not None:
        # Speculative mode represents M1+M2 stacked (baseline has M1, speculative adds M2)
        cross001 = spec_result >= target_tps
        print(f"\n  Target: >= {target_tps:.1f} tok/s")
        print(f"  Measured (M2 Speculative): {spec_result:.2f} tok/s")
        print(f"  Status: {'✅ PASS' if cross001 else '❌ FAIL'}")
        print(f"  Gap to target: {target_tps - spec_result:.2f} tok/s")
        
        # Also report baseline
        print(f"\n  Baseline (M3 only): {baseline_result:.2f} tok/s")
        print(f"  Speculative overhead: {baseline_result - spec_result:.2f} tok/s ({(1 - spec_result/baseline_result)*100:.1f}%)")
    else:
        print(f"\n  ❌ FAIL: Speculative mode benchmark failed")
        print(f"  Using baseline: {baseline_result:.2f} tok/s (target: {target_tps:.2f} tok/s)")
    
    print()
    print("=" * 80)
    print("  VALIDATION COMPLETE")
    print("=" * 80)
    
    # Cleanup
    tp.cleanup()
    
    # Return appropriate exit code
    if spec_result is not None and spec_result >= target_tps:
        print("\n✅ VAL-CROSS-001: PASSED")
        return 0
    else:
        print(f"\n❌ VAL-CROSS-001: FAILED (target: {target_tps:.1f} tok/s, measured: {spec_result if spec_result else 'N/A'})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
