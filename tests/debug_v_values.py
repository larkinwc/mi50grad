"""Debug V values in KV cache - examine actual magnitudes."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

emb = np.random.default_rng(42).standard_normal(config.hidden_size).astype(np.float16)

print("Loading copy engine...")
eng_c = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_c.load_layer_weights(i, loader.load_layer(i))
eng_c.load_final_norm(loader.load_final_norm())
eng_c.load_lm_head(loader.load_lm_head())
eng_c.build_dispatch_cache()
eng_c.set_cached_dispatch(True)
eng_c.set_stream_overlap_dispatch(True)
eng_c.set_c_dispatch(True)

print("Loading direct engine...")
eng_d = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_d.load_layer_weights(i, loader.load_layer(i))
eng_d.load_final_norm(loader.load_final_norm())
eng_d.load_lm_head(loader.load_lm_head())
eng_d.set_direct_kv_write(True)
eng_d.build_dispatch_cache()
eng_d.set_cached_dispatch(True)
eng_d.set_stream_overlap_dispatch(True)
eng_d.set_c_dispatch(True)

# Run 3 steps
for i in range(3):
    eng_c.decode_step(emb, i)
    eng_d.decode_step(emb, i)

e0c = eng_c.engines[0]
e0d = eng_d.engines[0]
kv_stride = e0c.local_num_kv_heads * config.head_dim * 2
fa_layer = [i for i in range(config.num_hidden_layers) if e0c.layers[i].layer_type == 'full_attention'][0]

# Examine V at position 2 (last warmup step)
v_base_c = e0c.kv_cache.layer_v_ptr(fa_layer)
v_base_d = e0d.kv_cache.layer_v_ptr(fa_layer)
v_off = 2 * kv_stride

v_c = np.frombuffer(e0c.device.download(v_base_c + v_off, kv_stride), dtype=np.float16)
v_d = np.frombuffer(e0d.device.download(v_base_d + v_off, kv_stride), dtype=np.float16)
v_abs_c = np.abs(v_c.astype(np.float32))
v_abs_d = np.abs(v_d.astype(np.float32))
v_diff = np.abs(v_c.astype(np.float32) - v_d.astype(np.float32))

print(f"\nV cache at position 2 (layer {fa_layer}, engine 0):")
print(f"  Copy path   - mean_abs={v_abs_c.mean():.4f}, max_abs={v_abs_c.max():.4f}")
print(f"  Direct path - mean_abs={v_abs_d.mean():.4f}, max_abs={v_abs_d.max():.4f}")
print(f"  Max diff: {v_diff.max():.4f}")
print(f"  Mean diff: {v_diff.mean():.6f}")

# Check if values match within FP16 precision
# FP16 relative precision: ~1e-3
# Absolute tolerance at max_abs: max_abs * 1e-3
abs_tol = v_abs_c.max() * 1e-3 * 2  # 2x ulp tolerance
print(f"  FP16 abs tolerance at max value: {abs_tol:.4f}")
print(f"  Max diff within 2 FP16 ulps: {v_diff.max() <= abs_tol}")

# Now run step 3 and check
eng_c.decode_step(emb, 3)
eng_d.decode_step(emb, 3)
v_off3 = 3 * kv_stride
v3_c = np.frombuffer(e0c.device.download(v_base_c + v_off3, kv_stride), dtype=np.float16)
v3_d = np.frombuffer(e0d.device.download(v_base_d + v_off3, kv_stride), dtype=np.float16)
v3_diff = np.abs(v3_c.astype(np.float32) - v3_d.astype(np.float32))
v3_abs = np.abs(v3_c.astype(np.float32))
abs_tol3 = v3_abs.max() * 2e-3
print(f"\nV cache at position 3:")
print(f"  Copy: max_abs={v3_abs.max():.4f}")
print(f"  Max diff: {v3_diff.max():.4f}")
print(f"  FP16 abs tol (2 ulps at max): {abs_tol3:.4f}")
print(f"  Within tolerance: {v3_diff.max() <= abs_tol3}")

eng_c.cleanup()
eng_d.cleanup()
