#!/usr/bin/env python3
"""Debug TP loading step by step."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.tp_engine import TPInferenceEngine

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"

config = load_config_from_json(MODEL_DIR)
print(f"Config: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

print("Creating TP engine with devices [0, 1]...")
tp_engine = TPInferenceEngine(config, device_ids=[0, 1])
print("TP engine created")

loader = QwenWeightLoader(MODEL_DIR, config=config,
                           bits=config.bits, group_size=config.group_size)

# Load first linear attention layer (layer 0 is linear)
print("\nLoading layer 0 (linear attention)...")
weights = loader.load_layer(0)
print(f"  Type: {weights.get('layer_type')}")
for k, v in weights.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape} {v.dtype}")

print("\nUploading to both engines...")
tp_engine.load_layer_weights(0, weights)
print("Layer 0 loaded OK")

# Load first full attention layer
fa_idx = None
for i in range(config.num_hidden_layers):
    if config.is_full_attention(i):
        fa_idx = i
        break

if fa_idx is not None:
    print(f"\nLoading layer {fa_idx} (full attention)...")
    weights = loader.load_layer(fa_idx)
    print(f"  Type: {weights.get('layer_type')}")
    for k, v in weights.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")

    print("\nUploading to both engines...")
    tp_engine.load_layer_weights(fa_idx, weights)
    print(f"Layer {fa_idx} loaded OK")

# Quick decode test
print("\nTesting 1-step decode...")
embed = loader.load_embedding()
tp_engine.load_final_norm(loader.load_final_norm())
tp_engine.load_lm_head(loader.load_lm_head())

# Only run if we loaded enough layers (just 2 layers)
# We need all layers. Let's load them all.
print("\nLoading remaining layers...")
for i in range(config.num_hidden_layers):
    if i == 0 or i == fa_idx:
        continue  # Already loaded
    weights = loader.load_layer(i)
    tp_engine.load_layer_weights(i, weights)
    if (i + 1) % 16 == 0:
        print(f"  Layer {i+1}/{config.num_hidden_layers}")
print("All layers loaded")

emb = embed[1234].copy()
print(f"\nRunning decode step 0 (emb shape={emb.shape})...")
out = tp_engine.decode_step(emb, 0)
print(f"Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")

print("\nRunning decode step 1...")
out = tp_engine.decode_step(emb, 1)
print(f"Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")

print("\nRunning decode step 2...")
out = tp_engine.decode_step(emb, 2)
print(f"Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")

print("\nResetting state...")
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0
print("State reset OK")

print("\nRunning decode after reset...")
out = tp_engine.decode_step(emb, 0)
print(f"Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")

# Benchmark
import time
print("\nBenchmark: 10 steps...")
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0

t0 = time.perf_counter()
for i in range(10):
    tp_engine.decode_step(emb, i)
tp_engine.synchronize()
elapsed = time.perf_counter() - t0
print(f"  {10/elapsed:.1f} tok/s ({elapsed*1000/10:.1f} ms/tok)")

print("\nSUCCESS")
