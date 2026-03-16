#!/usr/bin/env python3
"""Quick correctness test for global graph dispatch."""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')
import numpy as np
import time

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.tp_engine import TPInferenceEngine

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]

config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

print("Loading engine...")
engine = TPInferenceEngine(config, DEVICE_IDS, max_seq_len=512)
for layer_idx in range(config.num_hidden_layers):
    if layer_idx % 16 == 0:
        print(f"  Layer {layer_idx}...")
    engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())

engine.build_dispatch_cache()
engine.set_direct_kv_write(True)
engine.set_kernel_p2p_allreduce(True)
engine.set_c_dispatch(True)

print("Testing C dispatch (baseline)...")
rng = np.random.default_rng(42)
c_dispatch_outs = []
for step in range(5):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    c_dispatch_outs.append(out.copy())
    print(f"  C dispatch step {step}: out[0]={out[0]:.4f}")

# Reset
for e in engine.engines:
    e.kv_cache.current_len = 0
    e.deltanet_state.reset()

print("\nEnabling global graph dispatch...")
engine.set_global_graph_dispatch(True)

rng = np.random.default_rng(42)
print("Testing global graph (5 steps)...")
global_outs = []
for step in range(5):
    emb = rng.standard_normal(config.hidden_size).astype(np.float16)
    out = engine.decode_step(emb, step)
    global_outs.append(out.copy())
    ref = c_dispatch_outs[step]
    a32 = out.astype(np.float32)
    b32 = ref.astype(np.float32)
    dot = float(np.dot(a32, b32))
    na = float(np.linalg.norm(a32))
    nb = float(np.linalg.norm(b32))
    cs = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0
    status = "OK" if cs >= 0.99 else "*** FAIL ***"
    print(f"  Global graph step {step}: cosine_sim = {cs:.6f}  {status}")

# Check graph state
gds = engine._global_graph_decode_state
if gds and gds.captured:
    if hasattr(gds, '_attn_segs') and gds._attn_segs:
        num_layers_captured = len(gds._attn_segs[0])
        print(f"\nGraphs captured: {num_layers_captured} layers per GPU (2 segments each)")
        for li in [0, 16]:
            if li < num_layers_captured:
                a_nodes = gds._attn_segs[0][li].num_kernel_nodes()
                f_nodes = gds._ffn_segs[0][li].num_kernel_nodes()
                lw = engine.engines[0].layers[li]
                print(f"  Layer {li} ({lw.layer_type}): attn={a_nodes}, ffn={f_nodes} kernel nodes")
    elif hasattr(gds, '_full_segs') and gds._full_segs:
        print(f"\nGraphs captured: {len(gds._full_segs[0])} layers per GPU")
        for li in [0, 16]:
            if li < len(gds._full_segs[0]):
                seg = gds._full_segs[0][li]
                lw = engine.engines[0].layers[li]
                print(f"  Layer {li} ({lw.layer_type}): {seg.num_kernel_nodes()} kernel nodes")
else:
    print("WARNING: Graph not captured or in fallback mode!")

# Benchmark
print("\nBenchmarking (10 steps each)...")
BENCH_STEPS = 10

# C dispatch baseline
for e in engine.engines:
    e.kv_cache.current_len = 0
    e.deltanet_state.reset()
engine.set_global_graph_dispatch(False)

t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
t1 = time.perf_counter()
c_tps = BENCH_STEPS / (t1 - t0)
print(f"C dispatch: {c_tps:.1f} tok/s")

# Global graph
for e in engine.engines:
    e.kv_cache.current_len = 0
    e.deltanet_state.reset()
engine.set_global_graph_dispatch(True)

# Warmup (first call triggers capture or re-use)
for i in range(3):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)

for e in engine.engines:
    e.kv_cache.current_len = 0
    e.deltanet_state.reset()

t0 = time.perf_counter()
for i in range(BENCH_STEPS):
    emb = np.random.randn(config.hidden_size).astype(np.float16)
    engine.decode_step(emb, i)
t1 = time.perf_counter()
global_tps = BENCH_STEPS / (t1 - t0)
speedup = global_tps / c_tps
print(f"Global graph: {global_tps:.1f} tok/s ({speedup:.2f}x vs C dispatch)")

engine.cleanup()
del engine
print("\nDone!")
