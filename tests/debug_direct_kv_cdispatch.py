"""Debug C dispatch with direct KV write."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

def cosine_sim(a, b):
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    return float(np.dot(a32, b32)) / (float(np.linalg.norm(a32)) * float(np.linalg.norm(b32)))

emb = np.random.default_rng(42).standard_normal(config.hidden_size).astype(np.float16)

print("Loading direct-kv engine (C dispatch)...")
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
print(f"C dispatch enabled: {eng._c_dispatch_enabled}")
print("Direct-KV engine ready")

# Load reference engine (no C dispatch, standard)
print("Loading copy engine (Python cached only)...")
eng_ref = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_ref.load_layer_weights(i, loader.load_layer(i))
eng_ref.load_final_norm(loader.load_final_norm())
eng_ref.load_lm_head(loader.load_lm_head())
eng_ref.build_dispatch_cache()
eng_ref.set_cached_dispatch(True)
print("Ref engine ready")

print("\nComparing C dispatch vs Python cached reference:")
for step in range(5):
    out_ref = eng_ref.decode_step(emb, step)
    out_c = eng.decode_step(emb, step)
    cos = cosine_sim(out_ref, out_c)
    
    e0 = eng.engines[0]
    e0_ref = eng_ref.engines[0]
    kv_stride = e0.local_num_kv_heads * config.head_dim * 2
    fa_layer = [i for i in range(config.num_hidden_layers) if e0.layers[i].layer_type == 'full_attention'][0]
    
    v_base = e0.kv_cache.layer_v_ptr(fa_layer)
    v_base_ref = e0_ref.kv_cache.layer_v_ptr(fa_layer)
    v_off = step * kv_stride
    
    v = np.frombuffer(e0.device.download(v_base + v_off, kv_stride), dtype=np.float16)
    v_ref = np.frombuffer(e0_ref.device.download(v_base_ref + v_off, kv_stride), dtype=np.float16)
    v_err = float(np.max(np.abs(v.astype(np.float32) - v_ref.astype(np.float32))))
    
    k_base = e0.kv_cache.layer_k_ptr(fa_layer)
    k_base_ref = e0_ref.kv_cache.layer_k_ptr(fa_layer)
    k = np.frombuffer(e0.device.download(k_base + v_off, kv_stride), dtype=np.float16)
    k_ref = np.frombuffer(e0_ref.device.download(k_base_ref + v_off, kv_stride), dtype=np.float16)
    k_err = float(np.max(np.abs(k.astype(np.float32) - k_ref.astype(np.float32))))
    
    print(f"  Step {step}: cos={cos:.6f}, K_err={k_err:.2e}, V_err={v_err:.2e}")

eng.cleanup()
eng_ref.cleanup()
print("Done")
