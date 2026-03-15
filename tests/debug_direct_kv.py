"""Quick debug test for direct KV write correctness using Python cached path only (no C dispatch)."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
import time
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.inference.engine import InferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

def cosine_sim(a, b):
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    return float(np.dot(a32, b32)) / (float(np.linalg.norm(a32)) * float(np.linalg.norm(b32)))

emb = np.random.default_rng(42).standard_normal(config.hidden_size).astype(np.float16)

print("Loading copy engine (Python cached, no C dispatch)...")
eng_copy = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_copy.load_layer_weights(i, loader.load_layer(i))
eng_copy.load_final_norm(loader.load_final_norm())
eng_copy.load_lm_head(loader.load_lm_head())
eng_copy.build_dispatch_cache()
eng_copy.set_cached_dispatch(True)
print("Copy engine ready")

print("Loading direct-kv engine (Python cached, no C dispatch)...")
eng_direct = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_direct.load_layer_weights(i, loader.load_layer(i))
eng_direct.load_final_norm(loader.load_final_norm())
eng_direct.load_lm_head(loader.load_lm_head())
eng_direct.set_direct_kv_write(True)
eng_direct.build_dispatch_cache()
eng_direct.set_cached_dispatch(True)
print("Direct-KV engine ready")

# Run 5 steps and compare
print("\nComparing outputs (Python cached, no C dispatch):")
for step in range(5):
    out_c = eng_copy.decode_step(emb, step)
    out_d = eng_direct.decode_step(emb, step)
    cos = cosine_sim(out_c, out_d)
    
    # Check V cache at position `step` for layer 0
    eng_c0 = eng_copy.engines[0]
    eng_d0 = eng_direct.engines[0]
    kv_stride = eng_c0.local_num_kv_heads * config.head_dim * 2
    
    # Find first full attention layer
    fa_layer = [i for i in range(config.num_hidden_layers) if eng_c0.layers[i].layer_type == 'full_attention'][0]
    v_base_c = eng_c0.kv_cache.layer_v_ptr(fa_layer)
    v_base_d = eng_d0.kv_cache.layer_v_ptr(fa_layer)
    v_off = step * kv_stride
    
    v_c = np.frombuffer(eng_c0.device.download(v_base_c + v_off, kv_stride), dtype=np.float16)
    v_d = np.frombuffer(eng_d0.device.download(v_base_d + v_off, kv_stride), dtype=np.float16)
    v_err = float(np.max(np.abs(v_c.astype(np.float32) - v_d.astype(np.float32))))
    
    k_base_c = eng_c0.kv_cache.layer_k_ptr(fa_layer)
    k_base_d = eng_d0.kv_cache.layer_k_ptr(fa_layer)
    k_c = np.frombuffer(eng_c0.device.download(k_base_c + v_off, kv_stride), dtype=np.float16)
    k_d = np.frombuffer(eng_d0.device.download(k_base_d + v_off, kv_stride), dtype=np.float16)
    k_err = float(np.max(np.abs(k_c.astype(np.float32) - k_d.astype(np.float32))))
    
    print(f"  Step {step}: cos={cos:.6f}, K_err={k_err:.2e}, V_err={v_err:.2e}")

eng_copy.cleanup()
eng_direct.cleanup()
print("Done")
