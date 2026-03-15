"""
TP=4 decode benchmark with P2P allreduce and threaded dispatch profiling.

Establishes the TP=4 baseline with P2P allreduce (milestone 1 result)
and measures improvement from multi-threaded kernel dispatch (milestone 2).

Measures:
  - Overall tok/s and ms/tok (100 decode steps)
  - Allreduce time vs compute time breakdown
  - Comparison: serial dispatch vs threaded dispatch
  - Total elapsed time

USAGE:
    # Stop vLLM first:
    # docker stop vllm-mobydick
    # Run with 4 GPUs:
    # docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    #     -e HIP_VISIBLE_DEVICES=0,1,2,3 \\
    #     -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \\
    #     mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4.py'
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
WARMUP_STEPS = 3
BENCH_STEPS = 100


# ---------------------------------------------------------------------------
# Allreduce profiling instrumentation
# ---------------------------------------------------------------------------

class AllreduceProfiler:
    """Wraps TPInferenceEngine._allreduce_residual to measure allreduce time."""

    def __init__(self, engine: TPInferenceEngine):
        self.engine = engine
        self._original_allreduce_residual = engine._allreduce_residual
        self._original_decode_step = engine.decode_step
        self.allreduce_times = []
        self._step_ar_total = 0.0
        self._in_step = False

    def _timed_allreduce_residual(self, buffer_name: str, hidden_size: int):
        t0 = time.perf_counter()
        self._original_allreduce_residual(buffer_name, hidden_size)
        elapsed = time.perf_counter() - t0
        if self._in_step:
            self._step_ar_total += elapsed

    def timed_decode_step(self, token_embedding, position):
        self._step_ar_total = 0.0
        self._in_step = True
        t0 = time.perf_counter()
        result = self._original_decode_step(token_embedding, position)
        total = time.perf_counter() - t0
        self._in_step = False
        self.allreduce_times.append((total, self._step_ar_total))
        return result

    def install(self):
        """Monkey-patch the engine methods."""
        self.engine._allreduce_residual = self._timed_allreduce_residual
        self.engine.decode_step = self.timed_decode_step

    def uninstall(self):
        """Restore original methods."""
        self.engine._allreduce_residual = self._original_allreduce_residual
        self.engine.decode_step = self._original_decode_step

    def get_stats(self):
        """Return (mean_total_ms, mean_ar_ms, mean_compute_ms)."""
        if not self.allreduce_times:
            return 0.0, 0.0, 0.0
        totals = [t * 1000 for t, _ in self.allreduce_times]
        ars = [a * 1000 for _, a in self.allreduce_times]
        computes = [t - a for t, a in zip(totals, ars)]
        mean_total = sum(totals) / len(totals)
        mean_ar = sum(ars) / len(ars)
        mean_compute = sum(computes) / len(computes)
        return mean_total, mean_ar, mean_compute


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(engine, emb, warmup_steps, bench_steps, threaded: bool, label: str):
    """Run a benchmark with the given mode (serial or threaded).

    Returns (tok_per_sec, ms_per_tok, mean_ar_ms, mean_compute_ms).
    """
    engine.set_threaded_dispatch(threaded)

    profiler = AllreduceProfiler(engine)
    profiler.install()

    # Warmup
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()
    for i in range(warmup_steps):
        engine.decode_step(emb, i)
    engine.synchronize()
    profiler.allreduce_times.clear()

    # Timed run
    for e in engine.engines:
        e.kv_cache.current_len = 0
        e.deltanet_state.reset()

    t0 = time.perf_counter()
    for i in range(bench_steps):
        engine.decode_step(emb, warmup_steps + i)
    engine.synchronize()
    total_elapsed = time.perf_counter() - t0

    profiler.uninstall()

    tok_per_sec = bench_steps / total_elapsed
    ms_per_tok = total_elapsed / bench_steps * 1000
    mean_total_ms, mean_ar_ms, mean_compute_ms = profiler.get_stats()
    ar_pct = (mean_ar_ms / mean_total_ms * 100) if mean_total_ms > 0 else 0.0
    compute_pct = (mean_compute_ms / mean_total_ms * 100) if mean_total_ms > 0 else 0.0

    print(f"\n[{label}] Results:")
    print(f"  Throughput:         {tok_per_sec:.1f} tok/s")
    print(f"  Latency:            {ms_per_tok:.1f} ms/tok")
    print(f"  Total elapsed:      {total_elapsed:.2f}s ({bench_steps} steps)")
    print(f"  Allreduce time:     {mean_ar_ms:.2f} ms/tok  ({ar_pct:.1f}%)")
    print(f"  Compute time:       {mean_compute_ms:.2f} ms/tok  ({compute_pct:.1f}%)")

    return tok_per_sec, ms_per_tok, mean_ar_ms, mean_compute_ms


def main():
    print("=" * 70)
    print("TP=4 Decode Benchmark — P2P Allreduce + Threaded Dispatch")
    print("=" * 70)
    print(f"Model:        {MODEL_DIR}")
    print(f"GPUs:         {DEVICE_IDS}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Bench steps:  {BENCH_STEPS}")
    print()

    # Verify GPU count
    from src.runtime.hip_dispatch import HIPRuntime
    hip = HIPRuntime()
    hip.init()
    n_gpus = hip.device_count()
    print(f"GPUs visible: {n_gpus}")
    if n_gpus < 4:
        print(f"ERROR: Need 4 GPUs, only {n_gpus} visible.")
        print("Make sure to use: -e HIP_VISIBLE_DEVICES=0,1,2,3")
        sys.exit(1)

    # Load config
    print(f"\nLoading config from {MODEL_DIR}...")
    config = load_config_from_json(MODEL_DIR)
    print(f"Config: {config.num_hidden_layers} layers, "
          f"hidden_size={config.hidden_size}, "
          f"intermediate_size={config.intermediate_size}")
    print(f"TP=4 sharding: "
          f"{config.num_attention_heads}→{config.num_attention_heads // 4} attn heads/GPU, "
          f"{config.intermediate_size}→{config.intermediate_size // 4} FFN intermediate/GPU")

    # Load TP=4 engine
    print(f"\nLoading TP=4 engine on GPUs {DEVICE_IDS}...")
    t_load = time.perf_counter()
    engine = TPInferenceEngine(config, DEVICE_IDS)

    loader = QwenWeightLoader(MODEL_DIR, config)
    for i in range(config.num_hidden_layers):
        if i % 16 == 0:
            print(f"  Loading layer {i}/{config.num_hidden_layers}...")
        engine.load_layer_weights(i, loader.load_layer(i))
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    t_load_elapsed = time.perf_counter() - t_load
    print(f"Weights loaded in {t_load_elapsed:.1f}s")

    # Fixed input embedding
    np.random.seed(42)
    emb = np.random.randn(config.hidden_size).astype(np.float16)

    # ----------- Serial benchmark -----------
    print(f"\n{'=' * 70}")
    print("SERIAL DISPATCH (baseline)")
    print("=" * 70)
    serial_tps, serial_ms, serial_ar_ms, serial_compute_ms = run_benchmark(
        engine, emb, WARMUP_STEPS, BENCH_STEPS, threaded=False,
        label="SERIAL")

    # ----------- Threaded benchmark -----------
    print(f"\n{'=' * 70}")
    print("THREADED DISPATCH (multi-threaded, one thread per GPU)")
    print("=" * 70)
    threaded_tps, threaded_ms, threaded_ar_ms, threaded_compute_ms = run_benchmark(
        engine, emb, WARMUP_STEPS, BENCH_STEPS, threaded=True,
        label="THREADED")

    # ----------- Summary -----------
    speedup = serial_ms / threaded_ms if threaded_ms > 0 else float('nan')

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} {'tok/s':>10} {'ms/tok':>10} {'AR ms':>10} {'Compute ms':>12}")
    print("-" * 65)
    print(f"{'Serial':<20} {serial_tps:>10.1f} {serial_ms:>10.1f} "
          f"{serial_ar_ms:>10.2f} {serial_compute_ms:>12.2f}")
    print(f"{'Threaded':<20} {threaded_tps:>10.1f} {threaded_ms:>10.1f} "
          f"{threaded_ar_ms:>10.2f} {threaded_compute_ms:>12.2f}")
    print("-" * 65)
    print(f"  Speedup (threaded/serial): {speedup:.3f}x")
    print()
    print("BASELINES FOR COMPARISON:")
    print(f"  Single-GPU:         20.3 tok/s  (49.3 ms/tok)")
    print(f"  vLLM TP=4:          46.9 tok/s")
    print(f"  Theoretical TP=4:  ~81.2 tok/s  (20.3 * 4)")
    if threaded_tps > 20.3:
        vs_single = threaded_tps / 20.3
        print(f"  Threaded vs single: {vs_single:.2f}x")
    print("=" * 70)

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()

