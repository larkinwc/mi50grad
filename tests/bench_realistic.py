"""Benchmark decode at realistic sequence lengths.

Real-world agentic coding: contexts of 1K-8K tokens.
Full attention (16 layers) scales with seq_len, DeltaNet (48 layers) doesn't.
"""
import sys, time, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.path.insert(0, '/opt/mi50grad')

import argparse
import numpy as np
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.inference.engine import InferenceEngine
from src.model.weight_loader import QwenWeightLoader

parser = argparse.ArgumentParser()
parser.add_argument('--tp', type=int, default=1)
parser.add_argument('--steps', type=int, default=50)
args = parser.parse_args()

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
config = load_config_from_json(MODEL_DIR)

if args.tp > 1:
    device_ids = list(range(args.tp))
    engine = TPInferenceEngine(config, device_ids)
else:
    engine = InferenceEngine(config, device_id=0)

loader = QwenWeightLoader(MODEL_DIR, config)
for i in range(config.num_hidden_layers):
    w = loader.load_layer(i)
    if args.tp > 1:
        engine.load_layer_weights(i, w)
    else:
        engine.load_layer_weights(i, w)
    del w
engine.load_final_norm(loader.load_final_norm())
engine.load_lm_head(loader.load_lm_head())
print(f"Weights loaded. TP={args.tp}, steps={args.steps}")

emb = np.random.randn(config.hidden_size).astype(np.float16)
N = args.steps

def sync():
    if args.tp > 1:
        engine.synchronize()
    else:
        engine.device.synchronize()

def reset():
    if args.tp > 1:
        for eng in engine.engines:
            eng.deltanet_state.reset()
            eng.kv_cache.current_len = 0
    else:
        engine.deltanet_state.reset()
        engine.kv_cache.current_len = 0

# Test at different context lengths
# First fill KV cache to target position, then measure decode throughput
for ctx_start in [0, 128, 512, 1024, 2048]:
    reset()

    # Fill cache to ctx_start (untimed)
    if ctx_start > 0:
        for i in range(ctx_start):
            if args.tp > 1:
                engine.decode_step(emb, i)
            else:
                engine.decode_step(emb, i)
        sync()

    # Timed: N steps starting from ctx_start
    t0 = time.perf_counter()
    for i in range(N):
        if args.tp > 1:
            engine.decode_step(emb, ctx_start + i)
        else:
            engine.decode_step(emb, ctx_start + i)
    sync()
    elapsed = time.perf_counter() - t0

    avg_pos = ctx_start + N // 2
    print(f"  ctx={ctx_start:5d}-{ctx_start+N:5d} (avg pos {avg_pos:5d}): "
          f"{N/elapsed:5.1f} tok/s  ({elapsed/N*1000:5.1f} ms/tok)")

if args.tp > 1:
    engine.cleanup()
else:
    engine.cleanup()
