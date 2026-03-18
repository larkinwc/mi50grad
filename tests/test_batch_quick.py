#!/usr/bin/env python3
"""Quick test for batch decode functionality."""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine

model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
print(f"Loading config from {model_dir}...")
config = load_config_from_json(model_dir)
print(f"Config loaded: hidden={config.hidden_size}, layers={config.num_hidden_layers}")

print("Loading weight loader...")
loader = QwenWeightLoader(model_dir, config)
print("Loading embedding...")
embed = loader.load_embedding()
print(f"Embedding shape: {embed.shape}")

print("Creating engine...")
engine = InferenceEngine(config, device_id=0, max_seq_len=256)
print(f"Loading all {config.num_hidden_layers} layers...")
for i in range(config.num_hidden_layers):
    engine.load_layer_weights(i, loader.load_layer(i))
    if i % 16 == 0:
        print(f"  Loaded {i+1}/{config.num_hidden_layers} layers...")
engine.load_final_norm(loader.load_final_norm())
print("All layers loaded")

print("\nTesting batch=1 decode...")
engine.kv_cache.current_len = 0
engine.deltanet_state.reset()
emb1 = embed[760:761].copy()
try:
    result1 = engine.decode_step_batch(emb1, 1, 0)
    print(f"Batch=1 result shape: {result1.shape}")
    print("Batch=1 SUCCESS")
except Exception as e:
    print(f"Batch=1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting batch=2 decode...")
engine.kv_cache.current_len = 0
engine.deltanet_state.reset()
emb2 = np.vstack([embed[760]] * 2)
print(f"Batch=2 embeddings shape: {emb2.shape}")
try:
    result2 = engine.decode_step_batch(emb2, 2, 0)
    print(f"Batch=2 result shape: {result2.shape}")
    print("Batch=2 SUCCESS")
except Exception as e:
    print(f"Batch=2 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Correctness: Batch=2 vs Individual Decode ===")
# Test correctness: each sequence in batch should match individual decode
token_ids = [760, 6511, 314]  # Use a few different tokens
for pos in range(3):
    engine.kv_cache.current_len = pos
    engine.deltanet_state.reset()
    
    # Individual decode for sequence 0
    emb_indiv0 = embed[token_ids[0]].copy()
    engine.deltanet_state.reset()
    result_indiv0 = engine.decode_step(emb_indiv0, pos)
    
    # Individual decode for sequence 1
    engine.kv_cache.current_len = pos
    engine.deltanet_state.reset()
    emb_indiv1 = embed[token_ids[1]].copy()
    result_indiv1 = engine.decode_step(emb_indiv1, pos)
    
    # Batch decode
    engine.kv_cache.current_len = pos
    engine.deltanet_state.reset()
    emb_batch = np.vstack([embed[token_ids[0]], embed[token_ids[1]]])
    result_batch = engine.decode_step_batch(emb_batch, 2, pos)
    
    # Compare
    cos_sim0 = np.dot(result_indiv0.astype(np.float32).ravel(),
                      result_batch[0].astype(np.float32).ravel()) / (
        np.linalg.norm(result_indiv0.astype(np.float32)) *
        np.linalg.norm(result_batch[0].astype(np.float32)) + 1e-10)
    cos_sim1 = np.dot(result_indiv1.astype(np.float32).ravel(),
                      result_batch[1].astype(np.float32).ravel()) / (
        np.linalg.norm(result_indiv1.astype(np.float32)) *
        np.linalg.norm(result_batch[1].astype(np.float32)) + 1e-10)
    
    max_err0 = np.max(np.abs(result_indiv0.astype(np.float32) - result_batch[0].astype(np.float32)))
    max_err1 = np.max(np.abs(result_indiv1.astype(np.float32) - result_batch[1].astype(np.float32)))
    
    ok0 = cos_sim0 > 0.99 and max_err0 < 0.1
    ok1 = cos_sim1 > 0.99 and max_err1 < 0.1
    
    print(f"  Position {pos}: Seq0 cos={cos_sim0:.4f} err={max_err0:.4f} {'PASS' if ok0 else 'FAIL'} | "
          f"Seq1 cos={cos_sim1:.4f} err={max_err1:.4f} {'PASS' if ok1 else 'FAIL'}")

engine.cleanup()
print("\nTest complete.")
