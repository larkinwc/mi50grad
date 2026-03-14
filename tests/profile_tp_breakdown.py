"""Profile TP decode: measure allreduce vs compute time."""
import sys, time, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.path.insert(0, '/opt/mi50grad')

import numpy as np
from pathlib import Path
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

config = load_config_from_json("/opt/models/Qwen3.5-27B-GPTQ-Int4")
tp_engine = TPInferenceEngine(config, [0, 1])

loader = QwenWeightLoader("/opt/models/Qwen3.5-27B-GPTQ-Int4", config)
for i in range(config.num_hidden_layers):
    weights = loader.load_layer(i)
    tp_engine.load_layer_weights(i, weights)
    del weights
tp_engine.load_final_norm(loader.load_final_norm())
tp_engine.load_lm_head(loader.load_lm_head())
print("Weights loaded")

# Warmup
emb = np.random.randn(config.hidden_size).astype(np.float16)
for i in range(3):
    tp_engine.decode_step(emb, i)
tp_engine.synchronize()

# Instrument allreduce timing
import ctypes
orig_ar = tp_engine._allreduce_residual
ar_times = []

def timed_ar(buffer_name, hidden_size):
    t0 = time.perf_counter()
    orig_ar(buffer_name, hidden_size)
    ar_times.append(time.perf_counter() - t0)

tp_engine._allreduce_residual = timed_ar

# Run 10 profiled steps
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0

N = 10
ar_times.clear()
t0 = time.perf_counter()
for i in range(N):
    tp_engine.decode_step(emb, i)
tp_engine.synchronize()
total = time.perf_counter() - t0

total_ar = sum(ar_times)
total_compute = total - total_ar
per_step = total / N * 1000
ar_per_step = total_ar / N * 1000
compute_per_step = total_compute / N * 1000
ar_per_call = total_ar / len(ar_times) * 1000000  # microseconds

print(f"\nTotal: {per_step:.1f} ms/tok ({N/total:.1f} tok/s)")
print(f"Allreduce: {ar_per_step:.1f} ms/tok ({ar_per_step/per_step*100:.0f}%)")
print(f"Compute:   {compute_per_step:.1f} ms/tok ({compute_per_step/per_step*100:.0f}%)")
print(f"Allreduce calls: {len(ar_times)//N}/step, {ar_per_call:.0f} us/call")

tp_engine.cleanup()
