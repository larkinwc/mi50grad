"""Compare TP=4 standard vs TP=4 direct over 15 steps, same reference."""
import sys
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.engine import InferenceEngine
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
WARMUP = 3
STEPS = 15

print("Loading single-GPU reference...")
sg = InferenceEngine(config, device_id=0, max_seq_len=2048)
for i in range(config.num_hidden_layers):
    sg.load_layer_weights(i, loader.load_layer(i))
sg.load_final_norm(loader.load_final_norm())
sg.load_lm_head(loader.load_lm_head())
sg.kv_cache.current_len = 0
sg.deltanet_state.reset()
for i in range(WARMUP):
    sg.decode_step(emb, i)
sg.device.synchronize()
sg.kv_cache.current_len = 0
sg.deltanet_state.reset()
ref_outs = []
for i in range(STEPS):
    out = sg.decode_step(emb, i)
    ref_outs.append(out.copy())
sg.device.synchronize()
sg.cleanup()
del sg
print(f"  {STEPS} reference outputs collected")

print("\nLoading TP=4 standard (copy) engine...")
tp_std = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    tp_std.load_layer_weights(i, loader.load_layer(i))
tp_std.load_final_norm(loader.load_final_norm())
tp_std.load_lm_head(loader.load_lm_head())
tp_std.build_dispatch_cache()
tp_std.set_cached_dispatch(True)
tp_std.set_stream_overlap_dispatch(True)
tp_std.set_c_dispatch(True)

print("Loading TP=4 direct-KV engine...")
tp_dir = TPInferenceEngine(config, [0,1,2,3], max_seq_len=2048)
for i in range(config.num_hidden_layers):
    tp_dir.load_layer_weights(i, loader.load_layer(i))
tp_dir.load_final_norm(loader.load_final_norm())
tp_dir.load_lm_head(loader.load_lm_head())
tp_dir.set_direct_kv_write(True)
tp_dir.build_dispatch_cache()
tp_dir.set_cached_dispatch(True)
tp_dir.set_stream_overlap_dispatch(True)
tp_dir.set_c_dispatch(True)

def run_engine(eng, steps, warmup=WARMUP, label=""):
    for e in eng.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()
    for i in range(warmup):
        eng.decode_step(emb, i)
    eng.synchronize()
    for e in eng.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()
    outs = []
    for i in range(steps):
        out = eng.decode_step(emb, i)
        outs.append(out.copy())
    eng.synchronize()
    return outs

print("\nRunning standard TP=4...")
std_outs = run_engine(tp_std, STEPS)
print("Running direct-KV TP=4...")
dir_outs = run_engine(tp_dir, STEPS)

print(f"\n{'Step':>4}  {'std vs ref':>12}  {'dir vs ref':>12}  {'std vs dir':>12}")
print(f"  {'-'*45}")
for i in range(STEPS):
    c_sr = cosine_sim(ref_outs[i], std_outs[i])
    c_dr = cosine_sim(ref_outs[i], dir_outs[i])
    c_sd = cosine_sim(std_outs[i], dir_outs[i])
    f1 = 'FAIL' if c_sr < 0.99 else ''
    f2 = 'FAIL' if c_dr < 0.99 else ''
    print(f"  {i:>4}  {c_sr:>12.6f}{f1:>6}  {c_dr:>12.6f}{f2:>6}  {c_sd:>12.6f}")

tp_std.cleanup()
tp_dir.cleanup()
