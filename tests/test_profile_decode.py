#!/usr/bin/env python3
"""Profile decode step to find remaining bottlenecks."""
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
    loader = QwenWeightLoader(model_dir, config)
    embed = loader.load_embedding()

    engine = InferenceEngine(config, device_id=0, max_seq_len=2048)
    for i in range(config.num_hidden_layers):
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())

    # Warm up
    tokens = [760, 6511, 314, 9338, 369]
    for i, tid in enumerate(tokens):
        engine.decode_step(embed[tid], i)
    engine.device.synchronize()

    # Instrument key methods
    timers = {}

    def make_wrapper(name, orig_fn):
        timers[name] = 0.0
        def wrapper(*args, **kwargs):
            engine.device.synchronize()
            t0 = time.perf_counter()
            result = orig_fn(*args, **kwargs)
            engine.device.synchronize()
            timers[name] += time.perf_counter() - t0
            return result
        return wrapper

    engine._launch_rmsnorm = make_wrapper('rmsnorm', engine._launch_rmsnorm)
    engine._launch_gemv_fp16 = make_wrapper('gemv_fp16', engine._launch_gemv_fp16)
    engine._launch_gemv_int4 = make_wrapper('gemv_int4', engine._launch_gemv_int4)
    engine._launch_qk_norm = make_wrapper('qk_norm', engine._launch_qk_norm)
    engine._launch_rope = make_wrapper('rope', engine._launch_rope)
    engine._launch_decode_attn_256 = make_wrapper('attention', engine._launch_decode_attn_256)
    engine._launch_sigmoid_mul = make_wrapper('sigmoid_mul', engine._launch_sigmoid_mul)
    engine._launch_silu_fused = make_wrapper('silu', engine._launch_silu_fused)
    engine._launch_residual_add = make_wrapper('residual_add', engine._launch_residual_add)
    engine._decode_linear_attention_gpu = make_wrapper('deltanet', engine._decode_linear_attention_gpu)

    # Profile decode steps
    n_iters = 10
    t_total_start = time.perf_counter()
    for _ in range(n_iters):
        engine.kv_cache.current_len = 5
        engine.deltanet_state.reset()
        hidden = engine.decode_step(embed[760], 5)
        engine.device.synchronize()
        logits = engine.compute_logits()
        engine.device.synchronize()
    t_total = time.perf_counter() - t_total_start

    # Print results
    print("\nOperation breakdown (avg of %d decode steps):" % n_iters)
    instrumented_total = sum(timers.values())
    for name, t in sorted(timers.items(), key=lambda x: -x[1]):
        t_avg = t / n_iters * 1000
        pct = t / instrumented_total * 100
        print("  %-16s %6.2f ms  %5.1f%%" % (name, t_avg, pct))
    print("  %-16s %6.2f ms" % ("instrumented", instrumented_total / n_iters * 1000))
    print("  %-16s %6.2f ms" % ("wall_total", t_total / n_iters * 1000))
    print("  %-16s %6.2f ms" % ("overhead", (t_total - instrumented_total) / n_iters * 1000))
    print("\n  Decode rate: %.1f tok/s" % (n_iters / t_total))

    engine.cleanup()


if __name__ == "__main__":
    main()
