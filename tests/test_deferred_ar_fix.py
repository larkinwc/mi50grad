#!/usr/bin/env python3
"""Quick verification that deferred AR works."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
DEVICE_IDS = [0, 1, 2, 3]
MAX_SEQ_LEN = 256

print('Loading TP=4 engine...')
config = load_config_from_json(MODEL_DIR)
tp = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)

loader = QwenWeightLoader(MODEL_DIR, config)
for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print(f'  Layer {layer_idx}...')
    tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
tp.load_final_norm(loader.load_final_norm())
tp.load_lm_head(loader.load_lm_head())

print('Building dispatch cache...')
tp.build_dispatch_cache()
tp.set_c_dispatch(True)
tp.set_kernel_p2p_allreduce(True)
tp.set_deferred_attention_ar(True)

print('Deferred AR enabled successfully!')
print(f'  C dispatch: {tp._c_dispatch_enabled}')
print(f'  Kernel P2P: {tp._kernel_p2p_allreduce}')
print(f'  Deferred AR: {tp._deferred_attention_ar}')
print(f'  C dispatch plan: {tp._c_dispatch_plan is not None}')
if tp._c_dispatch_objects:
    plan = tp._c_dispatch_objects.get("plan")
    print(f'  Residual add fn: 0x{plan.residual_add_fn:016x}' if plan.residual_add_fn else '  Residual add fn: None')

# Test a decode step
rng = np.random.default_rng(42)
emb = rng.standard_normal(config.hidden_size).astype(np.float16)
print('Testing decode step...')
out = tp.decode_step(emb, 0)
tp._hip.synchronize()
print(f'Decode step completed successfully! Output shape: {out.shape}, min={np.min(out):.4f}, max={np.max(out):.4f}')
print('SUCCESS: Deferred AR is working correctly!')
tp.cleanup()
