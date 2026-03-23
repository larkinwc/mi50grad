#!/usr/bin/env python3
"""
Benchmark prompt processing (pp512) and text generation (tg128) across TP=1,2,4.

Model: Qwen 3.5 27B GPTQ-Int4
Hardware: AMD MI50 (gfx906) x4

Measures:
  - pp512: prefill throughput (tok/s) for 512-token prompt
  - tg128: decode throughput (tok/s) for 128 token generation

Usage:
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
        mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_pp_tg.py'
"""

import sys, os, time, json
import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
PP_LEN = 512
TG_LEN = 128
WARMUP_ITERS = 2
TG_WARMUP = 3
# TP=1: max_seq_len=256 (27B model tight on single 32GB GPU)
# TP>1: max_seq_len=768 (pp512+tg128=640 needs headroom)
MAX_SEQ_LEN_TP1 = 256
MAX_SEQ_LEN_TP = 768

results = {}


def load_single_gpu_engine(config, device_id=0):
    from src.inference.engine import InferenceEngine
    from src.model.weight_loader import QwenWeightLoader

    engine = InferenceEngine(config, device_id=device_id, max_seq_len=MAX_SEQ_LEN_TP1)
    loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"    Layer {i}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    return engine


def load_tp_engine(config, device_ids):
    from src.inference.tp_engine import TPInferenceEngine
    from src.model.weight_loader import QwenWeightLoader

    tp = TPInferenceEngine(config, device_ids=device_ids, max_seq_len=MAX_SEQ_LEN_TP)
    loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"    Layer {i}...")
        tp.load_layer_weights(i, loader.load_layer(i))
    tp.load_final_norm(loader.load_final_norm())
    tp.load_lm_head(loader.load_lm_head())
    tp.build_dispatch_cache()
    tp.set_direct_kv_write(True)
    tp.set_c_dispatch(True)
    tp.set_kernel_p2p_allreduce(True)
    tp.set_deferred_attention_ar(True)
    return tp


def bench_prefill_single(engine, config, pp_len=None):
    """Benchmark prefill on single GPU."""
    if pp_len is None:
        pp_len = PP_LEN
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((pp_len, config.hidden_size)).astype(np.float16)

    for _ in range(WARMUP_ITERS):
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        engine.prefill_step(embeddings)
        engine.device.synchronize()

    times = []
    for _ in range(3):
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        t0 = time.perf_counter()
        engine.prefill_step(embeddings)
        engine.device.synchronize()
        times.append(time.perf_counter() - t0)

    best = min(times)
    return pp_len / best


def bench_decode_single(engine, config):
    """Benchmark tg128 on single GPU."""
    rng = np.random.default_rng(42)

    for i in range(TG_WARMUP):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        engine.kv_cache.current_len = 0
        engine.deltanet_state.reset()
        engine.decode_step(emb, i)
        engine.device.synchronize()

    engine.kv_cache.current_len = 0
    engine.deltanet_state.reset()
    t0 = time.perf_counter()
    for step in range(TG_LEN):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        engine.decode_step(emb, step)
        engine.device.synchronize()
    elapsed = time.perf_counter() - t0
    return TG_LEN / elapsed


def bench_prefill_tp(tp_engine, config):
    """Benchmark pp512 on TP engine using batched prefill_step."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((PP_LEN, config.hidden_size)).astype(np.float16)

    def reset():
        for e in tp_engine.engines:
            e.kv_cache.current_len = 0
            e.deltanet_state.reset()

    # Warmup
    for _ in range(WARMUP_ITERS):
        reset()
        tp_engine.prefill_step(embeddings)
        tp_engine._hip.synchronize()

    # Benchmark
    times = []
    for _ in range(3):
        reset()
        t0 = time.perf_counter()
        tp_engine.prefill_step(embeddings)
        tp_engine._hip.synchronize()
        times.append(time.perf_counter() - t0)

    best = min(times)
    return PP_LEN / best


def bench_decode_tp(tp_engine, config):
    """Benchmark tg128 on TP engine."""
    rng = np.random.default_rng(42)

    for i in range(TG_WARMUP):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        for e in tp_engine.engines:
            e.kv_cache.current_len = 0
            e.deltanet_state.reset()
        tp_engine.decode_step(emb, i)
        tp_engine._hip.synchronize()

    for e in tp_engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()

    t0 = time.perf_counter()
    for step in range(TG_LEN):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp_engine.decode_step(emb, step)
        tp_engine._hip.synchronize()
    elapsed = time.perf_counter() - t0
    return TG_LEN / elapsed


def main():
    print("=" * 72)
    print("  Qwen 3.5 27B GPTQ-Int4 — PP/TG Benchmark")
    print("  pp512 (prompt processing 512 tokens)")
    print("  tg128 (text generation 128 tokens)")
    print("=" * 72)
    print(f"  Model: {MODEL_DIR}")
    print(f"  Max seq len: TP1={MAX_SEQ_LEN_TP1}, TP>1={MAX_SEQ_LEN_TP}")
    print()

    from src.model.qwen import load_config_from_json
    config = load_config_from_json(MODEL_DIR)

    # --- TP=1 ---
    print("=" * 72)
    print("  TP=1 (Single GPU)")
    print("  NOTE: max_seq_len=256 due to 32GB VRAM constraint on 27B model")
    print("  Using pp256 (pp512 requires TP>=2)")
    print("=" * 72)
    print("  Loading engine...")
    engine = load_single_gpu_engine(config, device_id=0)

    tp1_pp_len = min(PP_LEN, MAX_SEQ_LEN_TP1)
    print(f"  Benchmarking pp{tp1_pp_len}...")
    pp1 = bench_prefill_single(engine, config, pp_len=tp1_pp_len)
    print(f"    pp{tp1_pp_len}: {pp1:.2f} tok/s")

    print("  Benchmarking tg128...")
    tg1 = bench_decode_single(engine, config)
    print(f"    tg128: {tg1:.2f} tok/s")

    results['tp1'] = {'pp_len': tp1_pp_len, 'pp_tok_s': pp1, 'tg128': tg1}
    engine.cleanup()
    del engine

    # --- TP=2 ---
    print()
    print("=" * 72)
    print("  TP=2 (2 GPUs)")
    print("=" * 72)
    try:
        print("  Loading TP=2 engine...")
        tp2 = load_tp_engine(config, device_ids=[0, 1])

        print("  Benchmarking pp512...")
        pp2 = bench_prefill_tp(tp2, config)
        print(f"    pp512: {pp2:.2f} tok/s")

        print("  Benchmarking tg128...")
        tg2 = bench_decode_tp(tp2, config)
        print(f"    tg128: {tg2:.2f} tok/s")

        results['tp2'] = {'pp512': pp2, 'tg128': tg2}
        tp2.cleanup()
        del tp2
    except Exception as e:
        print(f"  TP=2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results['tp2'] = {'pp512': None, 'tg128': None, 'error': str(e)}

    # --- TP=4 ---
    print()
    print("=" * 72)
    print("  TP=4 (4 GPUs)")
    print("=" * 72)
    print("  Loading TP=4 engine...")
    tp4 = load_tp_engine(config, device_ids=[0, 1, 2, 3])

    print("  Benchmarking pp512...")
    pp4 = bench_prefill_tp(tp4, config)
    print(f"    pp512: {pp4:.2f} tok/s")

    print("  Benchmarking tg128...")
    tg4 = bench_decode_tp(tp4, config)
    print(f"    tg128: {tg4:.2f} tok/s")

    results['tp4'] = {'pp512': pp4, 'tg128': tg4}
    tp4.cleanup()
    del tp4

    # --- Summary ---
    print()
    print("=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Config':<12} {'Test':>8} {'PP (tok/s)':>12} {'tg128 (tok/s)':>15}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*15}")

    for tp_key in ['tp1', 'tp2', 'tp4']:
        r = results.get(tp_key, {})
        label = tp_key.upper().replace('TP', 'TP=')

        if tp_key == 'tp1':
            pp_len = r.get('pp_len', 256)
            pp_val = r.get('pp_tok_s')
        else:
            pp_len = PP_LEN
            pp_val = r.get('pp512')

        tg_val = r.get('tg128')

        pp_label = f"pp{pp_len}"
        pp_str = f"{pp_val:.2f}" if pp_val is not None else "FAILED"
        tg_str = f"{tg_val:.2f}" if tg_val is not None else "FAILED"
        print(f"  {label:<12} {pp_label:>8} {pp_str:>12} {tg_str:>15}")

    print("=" * 72)

    # Write JSON results
    out_path = "/opt/mi50grad/bench/pp_tg_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'model': 'Qwen3.5-27B-GPTQ-Int4',
            'hardware': '4x AMD MI50 (gfx906, 32GB HBM2)',
            'pp_len': PP_LEN,
            'tg_len': TG_LEN,
            'max_seq_len_tp1': MAX_SEQ_LEN_TP1,
            'max_seq_len_tp': MAX_SEQ_LEN_TP,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'results': results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
