#!/usr/bin/env python3
"""End-to-end benchmark: v7 GEMV + updated fused kernel"""
import sys, time, numpy as np
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
config = load_config_from_json(MODEL_DIR)
tp = TPInferenceEngine(config, device_ids=[0,1,2,3], max_seq_len=256)

loader = QwenWeightLoader(MODEL_DIR, config)
for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print("    Layer %d..." % layer_idx)
    tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
tp.load_final_norm(loader.load_final_norm())
tp.load_lm_head(loader.load_lm_head())

tp.build_dispatch_cache()
tp.set_direct_kv_write(True)
tp.set_c_dispatch(True)
tp.set_kernel_p2p_allreduce(True)
tp.set_deferred_attention_ar(True)

# Check which GEMV kernel is loaded
for e in tp.engines:
    v7 = getattr(e, '_gemv_int4_v7', None)
    v6 = getattr(e, '_gemv_int4_v6', None)
    print("GPU%d: GEMV v7=%s v6=%s" % (e.device.device_id, v7 is not None, v6 is not None))
    break

# Check fused kernel
fused = getattr(tp, '_gemv_fused_tp4_fn', None)
print("Fused GEMV+AR+RMSNorm: %s" % (fused is not None))

rng = np.random.default_rng(42)

# Warmup
print("Warmup (10 steps)...")
for i in range(10):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    for e in tp.engines:
        e.kv_cache.current_len = 0
    tp.decode_step(emb, i)
    tp._hip.synchronize()

# Benchmark
for num_steps in [50, 100, 128]:
    for e in tp.engines:
        e.kv_cache.current_len = 0
    t0 = time.perf_counter()
    for step in range(num_steps):
        emb = rng.standard_normal(config.hidden_size).astype(np.float16)
        tp.decode_step(emb, step)
        tp._hip.synchronize()
    elapsed = time.perf_counter() - t0
    tps = num_steps / elapsed
    ms = elapsed / num_steps * 1000
    print("Bench %d steps: %.2f tok/s (%.2f ms/tok)" % (num_steps, tps, ms))

print("DONE")
