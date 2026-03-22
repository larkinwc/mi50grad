#!/usr/bin/env python3
"""Minimal test for tree decode to isolate kernel launch issues."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader
from src.inference.tree_attention import TreeTopology, TreeAttentionMask

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"

print("Loading model...")
config = load_config_from_json(MODEL_DIR)
tp_engine = TPInferenceEngine(config, device_ids=[0,1,2,3], max_seq_len=512)

loader = QwenWeightLoader(MODEL_DIR, config)
for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print(f"  Loading layer {layer_idx}...")
    tp_engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))

print("Loading final norm...")
tp_engine.load_final_norm(loader.load_final_norm())

print("Building dispatch cache...")
print(f"TP engine has {len(tp_engine.engines)} engines")
tp_engine.set_direct_kv_write(True)  # Must be set BEFORE building cache
tp_engine.build_dispatch_cache()
tp_engine.set_cached_dispatch(True)
print(f"Cached dispatch enabled: {tp_engine._cached_dispatch}")
print(f"Engine layer caches: {len(tp_engine._engine_layer_caches)} engines")
if len(tp_engine._engine_layer_caches) > 0:
    print(f"  First engine has {len(tp_engine._engine_layer_caches[0])} layer caches")

print("Testing tree decode with 3 tokens...")
tree_size = 3
h = config.hidden_size

# Create fake embeddings
embeddings = [np.random.randn(h).astype(np.float16) for _ in range(tree_size)]

# Create a simple tree
tree = TreeTopology(0)
tree.add_child(tree.root, 1)
tree.add_child(tree.root, 2)

tree_mask = TreeAttentionMask(tree, kv_len=0)

print(f"Calling decode_step_tree with tree_size={tree_size}...")
try:
    outputs = tp_engine.decode_step_tree(embeddings, tree_mask, None)
    print(f"SUCCESS! Got {len(outputs)} outputs, first output shape: {outputs[0].shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
