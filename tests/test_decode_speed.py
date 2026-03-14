#!/usr/bin/env python3
"""Measure decode step speed."""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading layers...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights
    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()

    # Warm up
    for i in range(3):
        engine.decode_step(embed[760].copy(), position=i)

    # Time 10 decode steps
    times = []
    for i in range(10):
        emb = embed[760 + i].copy()
        t0 = time.perf_counter()
        engine.decode_step(emb, position=3 + i)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"Step {i}: {(t1-t0)*1000:.1f}ms")

    avg = np.mean(times) * 1000
    print(f"\nAvg: {avg:.1f}ms/tok = {1000/avg:.2f} tok/s")
    print(f"GPU DeltaNet: {engine._deltanet_gpu}")
    engine.cleanup()


if __name__ == "__main__":
    main()
