"""100-step TP benchmark with correctness check."""
import sys, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

import numpy as np
from src.model.qwen import QwenConfig
from src.inference.tp_engine import TPInferenceEngine
from src.inference.engine import InferenceEngine
from src.model.weight_loader import load_qwen_weights

config = QwenConfig()
print(f"Config: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

# TP=2 engine
tp_engine = TPInferenceEngine(config, [0, 1])

# Single GPU reference
ref_engine = InferenceEngine(config, device_id=0)

# Load weights
load_qwen_weights(tp_engine, config, "/opt/models/Qwen3.5-27B-GPTQ-Int4")
load_qwen_weights(ref_engine, config, "/opt/models/Qwen3.5-27B-GPTQ-Int4")

print("Weights loaded")

# Correctness: compare first token output
np.random.seed(42)
emb = np.random.randn(config.hidden_size).astype(np.float16)

tp_out = tp_engine.decode_step(emb, 0)
ref_out = ref_engine.decode_step(emb, 0)
cos = np.dot(tp_out.astype(np.float32), ref_out.astype(np.float32)) / (
    np.linalg.norm(tp_out.astype(np.float32)) * np.linalg.norm(ref_out.astype(np.float32)))
print(f"Correctness: cosine similarity = {cos:.6f}")

# Reset both
for e in tp_engine.engines:
    e.kv_cache.current_len = 0
    e.deltanet_state.reset()
ref_engine.kv_cache.current_len = 0
ref_engine.deltanet_state.reset()

# Benchmark: 100 steps
N = 100
emb = np.random.randn(config.hidden_size).astype(np.float16)

# Warmup
for i in range(3):
    tp_engine.decode_step(emb, i)
tp_engine.synchronize()

# Timed
start = time.perf_counter()
for i in range(N):
    tp_engine.decode_step(emb, 3 + i)
tp_engine.synchronize()
elapsed = time.perf_counter() - start

print(f"TP=2: {N/elapsed:.1f} tok/s ({elapsed/N*1000:.1f} ms/tok) [{N} steps]")

tp_engine.cleanup()
ref_engine.cleanup()
