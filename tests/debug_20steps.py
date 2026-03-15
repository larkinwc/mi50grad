"""Debug direct KV with more steps."""
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
    d = float(np.dot(a32, b32))
    n = float(np.linalg.norm(a32) * np.linalg.norm(b32))
    return d / n if n > 1e-12 else 0.0

emb = np.random.default_rng(42).standard_normal(config.hidden_size).astype(np.float16)

print("Loading reference (Python cached, standard path)...")
eng_ref = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    eng_ref.load_layer_weights(i, loader.load_layer(i))
eng_ref.load_final_norm(loader.load_final_norm())
eng_ref.load_lm_head(loader.load_lm_head())
eng_ref.build_dispatch_cache()
eng_ref.set_cached_dispatch(True)
eng_ref.set_stream_overlap_dispatch(True)
eng_ref.set_c_dispatch(True)

print("Loading direct KV (C dispatch)...")
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

print("\n20 steps, C dispatch ref vs direct:")
for i in range(20):
    out_ref = eng_ref.decode_step(emb, i)
    out_d = eng_d.decode_step(emb, i)
    cos = cosine_sim(out_ref, out_d)
    print(f"  Step {i:2d}: cos={cos:.6f} {'FAIL' if cos < 0.99 else ''}")

eng_ref.cleanup()
eng_d.cleanup()
