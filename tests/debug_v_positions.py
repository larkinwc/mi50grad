"""Debug: check if V cache positions are written correctly at each step."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
import ctypes
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

emb = np.random.default_rng(42).standard_normal(config.hidden_size).astype(np.float16)
STEPS = 12

print("Loading direct-KV engine...")
eng = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng.load_layer_weights(i, loader.load_layer(i))
eng.load_final_norm(loader.load_final_norm())
eng.load_lm_head(loader.load_lm_head())
eng.set_direct_kv_write(True)
eng.build_dispatch_cache()
eng.set_cached_dispatch(True)
eng.set_stream_overlap_dispatch(True)
eng.set_c_dispatch(True)

e0 = eng.engines[0]
fa_layers = [i for i in range(config.num_hidden_layers) 
             if e0.layers[i].layer_type == 'full_attention']
fa_layer = fa_layers[0]
kv_stride = e0.local_num_kv_heads * config.head_dim * 2

print(f"FA layer for testing: {fa_layer}, kv_stride={kv_stride}")

# Run steps and check V cache positions
prev_v = {}
print(f"\n{'Step':>4}  {'cur_len':>8}  {'pos_checked':>12}  {'zero_frac':>10}  {'v_norm':>8}")
print(f"  {'-'*52}")

for step in range(STEPS):
    out = eng.decode_step(emb, step)
    eng.synchronize()
    
    # After decode, kv_cache.advance() was called, so current_len = step+1
    cur_len = e0.kv_cache.current_len
    pos = cur_len - 1  # position just written
    
    v_base = e0.kv_cache.layer_v_ptr(fa_layer)
    v_bytes = e0.device.download(v_base + pos * kv_stride, kv_stride)
    v = np.frombuffer(v_bytes, dtype=np.float16)
    
    v_norm = float(np.linalg.norm(v.astype(np.float32)))
    zero_frac = float(np.sum(v == 0)) / len(v)
    
    # Check if this position was written (non-zero expected)
    print(f"  {step:>4}  {cur_len:>8}  {pos:>12}  {zero_frac:>10.4f}  {v_norm:>8.4f}")
    
    # Also check if position matches what was expected
    if step > 0:
        # prev_v should still be there (not overwritten)
        prev_bytes = e0.device.download(v_base + (pos-1) * kv_stride, kv_stride)
        prev_v_now = np.frombuffer(prev_bytes, dtype=np.float16)
        if step > 1 and step - 1 in prev_v:
            diff = np.max(np.abs(prev_v_now.astype(np.float32) - prev_v[step-1].astype(np.float32)))
            if diff > 0.001:
                print(f"  ERROR: Position {pos-1} was modified! diff={diff:.4f}")
    
    prev_v[step] = v.copy()

eng.cleanup()
