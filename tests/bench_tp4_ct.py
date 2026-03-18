#!/usr/bin/env python3
"""Benchmark TP=4 throughput for compressed-tensors model.

Compressed-tensors format:
  - Attention projections (q/k/v/o): INT4 quantized (group_size=32)
  - MLP projections (gate/up/down): INT4 quantized (group_size=32)
  - Symmetric quantization (no zero-point)

Benchmark configuration:
  - TP=4 across 4 MI50 GPUs (gfx906)
  - 100 decode steps, 5 warmup steps
  - Report tok/s and cosine similarity vs single-GPU

Expected throughput:
  - Baseline GPTQ model: 44.42 tok/s
  - CT model target: >= 38.0 tok/s (INT4 attention may be slower due to 4x more scale groups)
  - Cosine sim target: >= 0.99 vs single-GPU over 10 steps
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig, load_config_from_json
from src.model.compressed_tensors_loader import CompressedTensorsLoader, detect_compressed_tensors_format
from src.inference.tp_engine import TPInferenceEngine
from src.inference.engine import InferenceEngine


# Model path on dev server
MODEL_DIR = "/opt/models/Qwen3.5-27B-CT-4bit"

# Benchmark configuration
DEVICE_IDS = [0, 1, 2, 3]
NUM_STEPS = 100
WARMUP_STEPS = 5
MAX_SEQ_LEN = 512


def benchmark_single_gpu(config, loader, num_steps=10):
    """Run single-GPU benchmark for baseline comparison."""
    print("\n" + "=" * 60)
    print("Single-GPU Baseline Benchmark")
    print("=" * 60)
    
    engine = InferenceEngine(
        config, device_id=0, max_seq_len=MAX_SEQ_LEN,
        tp_size=1, tp_rank=0,
        quant_format='w4a16',
        use_int4_attention=True
    )
    
    # Load weights
    print("  Loading weights...")
    for layer_idx in range(config.num_hidden_layers):
        weights = loader.load_layer(layer_idx, tp_size=1, tp_rank=0)
        engine.load_layer_weights(layer_idx, weights)
    engine.load_final_norm(loader.load_final_norm())
    engine.load_lm_head(loader.load_lm_head())
    engine.build_dispatch_cache()
    engine.set_direct_kv_write(True)
    print("  Weights loaded, dispatch cache built")
    
    # Initialize with random embedding
    np.random.seed(42)
    hidden = np.random.randn(config.hidden_size).astype(np.float16) * 0.1
    
    # Warmup
    print(f"  Running {WARMUP_STEPS} warmup steps...")
    for step in range(WARMUP_STEPS):
        hidden = engine.decode_step(hidden, position=step)
    
    # Benchmark
    print(f"  Running {num_steps} benchmark steps...")
    start = time.perf_counter()
    hidden_states = []
    for step in range(num_steps):
        hidden = engine.decode_step(hidden, position=WARMUP_STEPS + step)
        hidden_states.append(hidden.copy())
    elapsed = time.perf_counter() - start
    
    tps = num_steps / elapsed
    print(f"\n  Single-GPU Results:")
    print(f"    Throughput: {tps:.2f} tok/s")
    print(f"    Total time: {elapsed*1000:.1f} ms for {num_steps} steps")
    print(f"    Per-step time: {elapsed/num_steps*1000:.2f} ms")
    
    engine.cleanup()
    return hidden_states, tps


def benchmark_tp4(config, loader, single_gpu_states=None, num_steps=10):
    """Run TP=4 benchmark and compare to single-GPU."""
    print("\n" + "=" * 60)
    print("TP=4 Benchmark")
    print("=" * 60)
    
    tp_engine = TPInferenceEngine(
        config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN,
        quant_format='w4a16',
        use_int4_attention=True
    )
    
    # Load weights
    print("  Loading weights...")
    for layer_idx in range(config.num_hidden_layers):
        weights = loader.load_layer(layer_idx, tp_size=4, tp_rank=0)
        tp_engine.load_layer_weights(layer_idx, weights)
    tp_engine.load_final_norm(loader.load_final_norm())
    tp_engine.load_lm_head(loader.load_lm_head())
    tp_engine.build_dispatch_cache()
    tp_engine.set_kernel_p2p_allreduce(True)
    tp_engine.set_direct_kv_write(True)
    print("  Weights loaded, dispatch cache built")
    print(f"  Kernel P2P allreduce enabled")
    print(f"  Direct KV cache write enabled")
    
    # Initialize with same random embedding as single-GPU
    np.random.seed(42)
    hidden = np.random.randn(config.hidden_size).astype(np.float16) * 0.1
    
    # Warmup
    print(f"  Running {WARMUP_STEPS} warmup steps...")
    for step in range(WARMUP_STEPS):
        hidden = tp_engine.decode_step(hidden, position=step)
    
    # Benchmark
    print(f"  Running {num_steps} benchmark steps...")
    start = time.perf_counter()
    tp_hidden_states = []
    for step in range(num_steps):
        hidden = tp_engine.decode_step(hidden, position=WARMUP_STEPS + step)
        tp_hidden_states.append(hidden.copy())
    elapsed = time.perf_counter() - start
    
    tps = num_steps / elapsed
    print(f"\n  TP=4 Results:")
    print(f"    Throughput: {tps:.2f} tok/s")
    print(f"    Total time: {elapsed*1000:.1f} ms for {num_steps} steps")
    print(f"    Per-step time: {elapsed/num_steps*1000:.2f} ms")
    
    # Compute cosine similarity vs single-GPU
    if single_gpu_states is not None:
        print(f"\n  Cosine Similarity vs Single-GPU:")
        cos_sims = []
        for i, (sg, tp) in enumerate(zip(single_gpu_states, tp_hidden_states)):
            cos_sim = np.dot(sg, tp) / (np.linalg.norm(sg) * np.linalg.norm(tp) + 1e-8)
            cos_sims.append(cos_sim)
            if (i + 1) % 5 == 0:
                print(f"    Step {WARMUP_STEPS + i + 1}: {cos_sim:.6f}")
        
        avg_cos_sim = np.mean(cos_sims)
        min_cos_sim = np.min(cos_sims)
        print(f"\n    Average cosine similarity: {avg_cos_sim:.6f}")
        print(f"    Minimum cosine similarity: {min_cos_sim:.6f}")
        
        if avg_cos_sim >= 0.99:
            print(f"    Cosine similarity target (>= 0.99): PASS")
        else:
            print(f"    Cosine similarity target (>= 0.99): FAIL ({avg_cos_sim:.4f})")
    
    tp_engine.cleanup()
    return tp_hidden_states, tps


def main():
    print("=" * 70)
    print("TP=4 Compressed-Tensors Model Benchmark")
    print("=" * 70)
    print(f"\nModel directory: {MODEL_DIR}")
    print(f"Device IDs: {DEVICE_IDS}")
    print(f"Configuration: {NUM_STEPS} steps ({WARMUP_STEPS} warmup), max_seq_len={MAX_SEQ_LEN}")
    
    # Check if model exists
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        print(f"\nERROR: Model directory {MODEL_DIR} does not exist.")
        print("Run the model download script first.")
        return 1
    
    # Check format detection
    is_ct = detect_compressed_tensors_format(MODEL_DIR)
    if not is_ct:
        print(f"\nERROR: Model {MODEL_DIR} is not in compressed-tensors format.")
        return 1
    
    print(f"  Detected compressed-tensors format: PASS")
    
    # Load config and weights
    print("\nLoading model configuration and weights...")
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    print(f"  Config loaded: {config.num_hidden_layers} layers, hidden_size={config.hidden_size}")
    
    try:
        # Run single-GPU baseline
        single_gpu_states, single_gpu_tps = benchmark_single_gpu(
            config, loader, num_steps=NUM_STEPS // 10)
        
        # Run TP=4 benchmark
        tp4_states, tp4_tps = benchmark_tp4(
            config, loader, single_gpu_states=single_gpu_states,
            num_steps=NUM_STEPS // 10)
        
        # Summary
        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(f"  Single-GPU: {single_gpu_tps:.2f} tok/s")
        print(f"  TP=4:       {tp4_tps:.2f} tok/s")
        print(f"  Speedup:    {tp4_tps / single_gpu_tps:.2f}x")
        
        # Compare to GPTQ baseline
        gptq_baseline = 44.42  # tok/s from GPTQ model
        print(f"\n  Comparison to GPTQ baseline ({gptq_baseline:.2f} tok/s):")
        print(f"    CT model TP=4: {tp4_tps:.2f} tok/s ({tp4_tps / gptq_baseline * 100:.1f}% of baseline)")
        
        # Check throughput target
        target_tps = 38.0
        if tp4_tps >= target_tps:
            print(f"\n  Throughput target (>= {target_tps:.1f} tok/s): PASS")
        else:
            print(f"\n  Throughput target (>= {target_tps:.1f} tok/s): NEEDS IMPROVEMENT ({tp4_tps:.2f} tok/s)")
        
        print("\n" + "=" * 70)
        print("Benchmark complete")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\nBENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
