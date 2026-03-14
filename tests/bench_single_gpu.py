"""Benchmark single GPU decode to see fused kernel improvement."""
import sys, time, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.path.insert(0, '/opt/mi50grad')

import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
from src.model.weight_loader import QwenWeightLoader

config = load_config_from_json("/opt/models/Qwen3.5-27B-GPTQ-Int4")
engine = InferenceEngine(config, device_id=0)

loader = QwenWeightLoader("/opt/models/Qwen3.5-27B-GPTQ-Int4", config)
for i in range(config.num_hidden_layers):
    engine.load_layer_weights(i, loader.load_layer(i))
engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())
print("Weights loaded")

emb = np.random.randn(config.hidden_size).astype(np.float16)

# Warmup
for i in range(3):
    engine.decode_step(emb, i)
engine.device.synchronize()

# Benchmark
N = 100
engine.kv_cache.current_len = 0
engine.deltanet_state.reset()

t0 = time.perf_counter()
for i in range(N):
    engine.decode_step(emb, i)
engine.device.synchronize()
elapsed = time.perf_counter() - t0
print(f"Single GPU: {N/elapsed:.1f} tok/s ({elapsed/N*1000:.1f} ms/tok)")

engine.cleanup()
