#!/usr/bin/env python3
"""Minimal TP debug - isolate segfault."""
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line-buffered
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

print("Step 1: imports starting")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Step 2: importing config")
from src.model.qwen import load_config_from_json

print("Step 3: importing engine")
from src.inference.engine import InferenceEngine

print("Step 4: importing tp_engine")
from src.inference.tp_engine import TPInferenceEngine

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"

print("Step 5: loading config")
config = load_config_from_json(MODEL_DIR)
print(f"  Config: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tp', type=int, default=2)
args, _ = parser.parse_known_args()
device_ids = list(range(args.tp))
print(f"Step 6: creating TP engine with {device_ids}")
try:
    tp_engine = TPInferenceEngine(config, device_ids=device_ids)
    print("Step 6: TP engine created OK")
except Exception as e:
    print(f"Step 6 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 7: importing weight loader")
from src.model.weight_loader import QwenWeightLoader
loader = QwenWeightLoader(MODEL_DIR, config=config,
                           bits=config.bits, group_size=config.group_size)

print("Step 8: loading layer 0")
import numpy as np
weights = loader.load_layer(0)
print(f"  Type: {weights.get('layer_type')}")
for k, v in weights.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape} {v.dtype}")

print("Step 9: uploading layer 0 to TP engine")
try:
    tp_engine.load_layer_weights(0, weights)
    print("  Layer 0 loaded OK")
except Exception as e:
    print(f"  Layer 0 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 10: loading all remaining layers")
for i in range(1, config.num_hidden_layers):
    weights = loader.load_layer(i)
    tp_engine.load_layer_weights(i, weights)
    if (i + 1) % 16 == 0:
        print(f"  Layer {i+1}/{config.num_hidden_layers}")
print("  All layers loaded")

print("Step 11: loading final norm + lm head")
tp_engine.load_final_norm(loader.load_final_norm())
tp_engine.load_lm_head(loader.load_lm_head())
embed = loader.load_embedding()
print("  Done")

print("Step 12: single decode step")
emb = embed[1234].copy()
try:
    out = tp_engine.decode_step(emb, 0)
    print(f"  Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 13: second decode step")
try:
    out = tp_engine.decode_step(emb, 1)
    print(f"  Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 14: third decode step")
try:
    out = tp_engine.decode_step(emb, 2)
    print(f"  Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 15: state reset")
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0
print("  Reset OK")

print("Step 16: decode after reset")
try:
    out = tp_engine.decode_step(emb, 0)
    print(f"  Output: shape={out.shape}, norm={np.linalg.norm(out.astype(np.float32)):.4f}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 17: 100-step benchmark")
import time
for eng in tp_engine.engines:
    eng.deltanet_state.reset()
    eng.kv_cache.current_len = 0

# Warmup 3 steps
for i in range(3):
    tp_engine.decode_step(emb, i)
tp_engine.synchronize()

N_BENCH = 100
t0 = time.perf_counter()
for i in range(N_BENCH):
    tp_engine.decode_step(emb, 3 + i)
tp_engine.synchronize()
elapsed = time.perf_counter() - t0
print(f"  {N_BENCH/elapsed:.1f} tok/s ({elapsed*1000/N_BENCH:.1f} ms/tok)")

print("\nSUCCESS - all steps passed")
