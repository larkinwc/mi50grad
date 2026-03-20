#!/usr/bin/env python3
"""Test fused GEMV kernel throughput."""
import sys, time
sys.path.insert(0, "/opt/mi50grad")
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader
import numpy as np

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
config = load_config_from_json(MODEL_DIR)
print("Loading TP=4 engine...")
tp = TPInferenceEngine(config, device_ids=[0,1,2,3], max_seq_len=256)
loader = QwenWeightLoader(MODEL_DIR, config)
for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print(f"  Layer {layer_idx}...")
    tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
tp.load_final_norm(loader.load_final_norm())
tp.load_lm_head(loader.load_lm_head())
tp.build_dispatch_cache()
tp.set_direct_kv_write(True)
tp.set_c_dispatch(True)
tp.set_kernel_p2p_allreduce(True)
tp.set_deferred_attention_ar(True)
print(f"Fused GEMV enabled: {tp._c_dispatch_objects.get('gemv_fused_lib') is not None}")

# Run a few tokens to test
rng = np.random.default_rng(42)
print("Running 10 warmup tokens...")
for i in range(10):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    tp.decode_step(emb, i)
    tp._hip.synchronize()

print("Running 50 benchmark tokens...")
t0 = time.perf_counter()
for i in range(50):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    tp.decode_step(emb, 10 + i)
    tp._hip.synchronize()
elapsed = time.perf_counter() - t0
tps = 50 / elapsed
print(f"Throughput: {tps:.2f} tok/s ({elapsed/50*1000:.2f} ms/tok)")
tp.cleanup()
