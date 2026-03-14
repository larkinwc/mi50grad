#!/usr/bin/env python3
"""Profile TP overhead: measure allreduce cost vs compute cost."""
import sys
import os
import time
import numpy as np
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine
from src.inference.tp_engine import TPInferenceEngine
from src.runtime.tensor_parallel import TensorParallelGroup

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"

config = load_config_from_json(MODEL_DIR)
print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

# 1. Measure raw allreduce cost
print("\n=== Allreduce Microbenchmark ===")
tp_group = TensorParallelGroup([0, 1])
from src.runtime.hip_dispatch import GPUDevice
dev0 = GPUDevice(0)
dev1 = GPUDevice(1)

size = config.hidden_size * 2  # FP16
ptr0 = dev0.malloc(size)
ptr1 = dev1.malloc(size)

# Fill with some data
data = np.random.randn(config.hidden_size).astype(np.float16).tobytes()
dev0.upload(ptr0, data)
dev1.upload(ptr1, data)

# Warmup
for _ in range(5):
    tp_group.all_reduce_sum([ptr0, ptr1], size)

# Benchmark allreduce
N = 100
t0 = time.perf_counter()
for _ in range(N):
    tp_group.all_reduce_sum([ptr0, ptr1], size)
elapsed = time.perf_counter() - t0
print(f"Allreduce ({size} bytes): {elapsed/N*1e6:.1f} us/call")
print(f"Per token (128 allreduces): {elapsed/N*128*1000:.1f} ms")

# 2. Measure synchronize cost
print("\n=== Synchronize Microbenchmark ===")
t0 = time.perf_counter()
for _ in range(N):
    dev0.synchronize()
    dev1.synchronize()
elapsed = time.perf_counter() - t0
print(f"Sync both GPUs: {elapsed/N*1e6:.1f} us/call")
print(f"Per token (256 syncs): {elapsed/N*256*1000:.1f} ms")

dev0.free(ptr0)
dev1.free(ptr1)
dev0.cleanup()
dev1.cleanup()
tp_group.cleanup()

# 3. Full TP engine profile
print("\n=== Full TP Decode Profile ===")
tp_engine = TPInferenceEngine(config, device_ids=[0, 1])

loader = QwenWeightLoader(MODEL_DIR, config=config,
                           bits=config.bits, group_size=config.group_size)

for i in range(config.num_hidden_layers):
    weights = loader.load_layer(i)
    tp_engine.load_layer_weights(i, weights)
    if (i + 1) % 32 == 0:
        print(f"  Loaded {i+1}/{config.num_hidden_layers}")

tp_engine.load_final_norm(loader.load_final_norm())
tp_engine.load_lm_head(loader.load_lm_head())
embed = loader.load_embedding()

emb = embed[1234].copy()

# Warmup
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0

for i in range(3):
    tp_engine.decode_step(emb, i)

# Time a single decode step in detail by monkey-patching
original_sync_ar = tp_engine._sync_allreduce_residual_broadcast
ar_times = []

def timed_sync_ar(buffer_name, hidden_size):
    t = time.perf_counter()
    original_sync_ar(buffer_name, hidden_size)
    ar_times.append(time.perf_counter() - t)

tp_engine._sync_allreduce_residual_broadcast = timed_sync_ar

# Reset and time one step
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0

ar_times.clear()
t0 = time.perf_counter()
tp_engine.decode_step(emb, 0)
total = time.perf_counter() - t0

ar_total = sum(ar_times)
compute_total = total - ar_total

print(f"\nSingle decode step breakdown:")
print(f"  Total:        {total*1000:.1f} ms")
print(f"  Allreduce:    {ar_total*1000:.1f} ms ({ar_total/total*100:.0f}%)")
print(f"  Compute:      {compute_total*1000:.1f} ms ({compute_total/total*100:.0f}%)")
print(f"  Allreduce calls: {len(ar_times)}")
print(f"  Avg per call: {ar_total/len(ar_times)*1e6:.0f} us")

tp_engine.cleanup()
print("\nDone")
