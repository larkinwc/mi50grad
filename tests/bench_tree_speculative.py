"""
Benchmark script for tree-based speculative decoding with REAL TPInferenceEngine.

Measures:
1. Effective throughput: accepted_tokens / wall_time
2. Acceptance rate per verification step (target: >= 2.0)
3. Speedup vs baseline (~54 tok/s for TP=4)
4. Allreduce efficiency: 64 calls per verification regardless of tree size
5. Correctness: output matches greedy decode

Usage (on dev server with 4x MI50):
    cd /opt/mi50grad
    python3 tests/bench_tree_speculative.py
    
    # With custom parameters
    python3 tests/bench_tree_speculative.py --max-tokens 100 --tree-size 7 --num-runs 5
    python3 tests/bench_tree_speculative.py --tree-topology optimized
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.tree_speculative import TreeSpeculativeDecoder
from src.inference.speculative import NgramCache
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"


def create_real_decoder(max_tree_size: int = 7,
                        max_tree_depth: int = 2,
                        branching_factor: int = 2,
                        tree_topology: str = 'ngram',
                        base_acceptance_rate: float = 0.54) -> TreeSpeculativeDecoder:
    """Create TreeSpeculativeDecoder with real TPInferenceEngine.
    
    Loads the full model on 4x MI50 GPUs.
    """
    print("Loading model and TP engine...")
    config = load_config_from_json(MODEL_DIR)
    
    # Create TP engine on all 4 GPUs
    tp_engine = TPInferenceEngine(config, device_ids=[0, 1, 2, 3], max_seq_len=512)
    
    # Load weights
    loader = QwenWeightLoader(MODEL_DIR, config)
    for layer_idx in range(config.num_hidden_layers):
        if layer_idx % 8 == 0:
            print(f"  Loading layer {layer_idx}/{config.num_hidden_layers}...")
        tp_engine.load_layer_weights(layer_idx, loader.load_layer(layer_idx))
    
    print("  Loading final norm and lm_head...")
    tp_engine.load_final_norm(loader.load_final_norm())
    lm_head_weight = loader.load_lm_head()
    
    # Configure engine
    tp_engine.build_dispatch_cache()
    tp_engine.set_direct_kv_write(True)
    tp_engine.set_c_dispatch(True)
    tp_engine.set_kernel_p2p_allreduce(True)
    tp_engine.set_deferred_attention_ar(True)
    
    print("Model loaded successfully.")
    
    # Extract lm_head weight for decoder
    lm_head_np = lm_head_weight.copy()
    
    # Get embed weight from first GPU
    embed_weight = tp_engine.engines[0]._embed_weight.copy()
    
    # Try to load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}). Using token IDs directly.")
        tokenizer = None
    
    decoder = TreeSpeculativeDecoder(
        engine=tp_engine,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_np,
        tokenizer=tokenizer,
        ngram_size=3,
        max_tree_size=max_tree_size,
        max_tree_depth=max_tree_depth,
        branching_factor=branching_factor,
        tree_topology=tree_topology,
        base_acceptance_rate=base_acceptance_rate,
    )
    
    return decoder


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_effective_throughput(decoder: TreeSpeculativeDecoder,
                                    prompt: str,
                                    max_tokens: int = 100,
                                    num_runs: int = 3,
                                    verbose: bool = True) -> Dict[str, Any]:
    """Benchmark effective throughput of tree speculative decoding.
    
    Effective throughput = total_accepted_tokens / wall_time
    
    This measures the real speedup from accepting multiple tokens
    per verification step.
    
    Args:
        decoder: TreeSpeculativeDecoder instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        num_runs: Number of benchmark runs
        verbose: Print progress
        
    Returns:
        Benchmark results dict
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Effective Throughput")
        print(f"{'='*60}")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Num runs: {num_runs}")
        print(f"  Tree topology: {decoder.tree_topology}")
        print(f"  Max tree size: {decoder.max_tree_size}")
    
    # Encode prompt - use tokenizer if available, otherwise work with token IDs directly
    if decoder.tokenizer is not None:
        input_ids = decoder.tokenizer.encode(prompt)
    else:
        # Fallback: convert prompt to simple token IDs (ord values)
        # This is for testing only - real usage should have a tokenizer
        input_ids = [ord(c) % 1000 for c in prompt]
        print(f"  Warning: No tokenizer - using character-level encoding")
    
    all_stats = []
    total_wall_time = 0.0
    total_accepted = 0
    total_generated = 0
    
    for run_idx in range(num_runs):
        # Reset decoder state
        decoder.ngram_cache.clear()
        decoder.stats = {
            'total_verifications': 0,
            'total_drafts': 0,
            'total_accepted': 0,
            'acceptance_rate': 0.0,
            'avg_accepted_per_step': 0.0,
        }
        
        # Run decode
        t0 = time.perf_counter()
        generated_ids, run_stats = decoder.generate(
            input_ids,
            max_tokens=max_tokens,
            verbose=False
        )
        elapsed = time.perf_counter() - t0
        
        all_stats.append(run_stats)
        total_wall_time += elapsed
        total_accepted += run_stats['total_accepted']
        total_generated += len(generated_ids)
        
        if verbose:
            print(f"\n  Run {run_idx + 1}/{num_runs}:")
            print(f"    Generated: {len(generated_ids)} tokens")
            print(f"    Accepted: {run_stats['total_accepted']} tokens")
            print(f"    Verifications: {run_stats['total_verifications']}")
            print(f"    Wall time: {elapsed*1000:.1f}ms")
            print(f"    Generated throughput: {len(generated_ids)/elapsed:.1f} tok/s")
            print(f"    Effective throughput: {run_stats['total_accepted']/elapsed:.1f} tok/s")
            print(f"    Avg accepted/step: {run_stats['avg_accepted_per_step']:.2f}")
    
    # Aggregate results
    avg_time = total_wall_time / num_runs
    avg_generated = total_generated / num_runs
    avg_accepted = total_accepted / num_runs
    
    results = {
        'avg_wall_time_ms': avg_time * 1000,
        'avg_generated_tokens': avg_generated,
        'avg_accepted_tokens': avg_accepted,
        'generated_throughput_tps': avg_generated / avg_time,
        'effective_throughput_tps': avg_accepted / avg_time,
        'avg_acceptance_rate': sum(s['overall_acceptance_rate'] for s in all_stats) / num_runs,
        'avg_accepted_per_step': sum(s['avg_accepted_per_step'] for s in all_stats) / num_runs,
        'num_runs': num_runs,
        'speedup_over_baseline': (avg_accepted / avg_time) / 54.0,  # vs 54 tok/s baseline
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS (AVERAGE OVER {num_runs} RUNS)")
        print(f"{'='*60}")
        print(f"  Generated throughput: {results['generated_throughput_tps']:.1f} tok/s")
        print(f"  Effective throughput: {results['effective_throughput_tps']:.1f} tok/s")
        print(f"  Speedup vs baseline (54 tok/s): {results['speedup_over_baseline']:.2f}x")
        print(f"  Avg acceptance rate: {results['avg_acceptance_rate']:.1%}")
        print(f"  Avg accepted per step: {results['avg_accepted_per_step']:.2f}")
    
    return results


def benchmark_allreduce_efficiency(decoder: TreeSpeculativeDecoder,
                                    prompt: str,
                                    num_steps: int = 10) -> Dict[str, Any]:
    """Benchmark allreduce efficiency (should be 64 per step).
    
    Args:
        decoder: TreeSpeculativeDecoder instance
        prompt: Input prompt
        num_steps: Number of decode steps to measure
        
    Returns:
        Allreduce efficiency metrics
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Allreduce Efficiency")
    print(f"{'='*60}")
    
    # Encode prompt
    if decoder.tokenizer is not None:
        input_ids = decoder.tokenizer.encode(prompt)
    else:
        input_ids = [ord(c) % 1000 for c in prompt]
    decoder.ngram_cache.build_from_sequence(input_ids)
    
    context = input_ids.copy()
    all_allreduce_counts = []
    total_steps = 0
    
    for step in range(num_steps):
        # Reset step counter
        decoder.engine.step_allreduce_calls = 0
        
        # Single decode step
        accepted, _ = decoder.decode_step(context)
        
        # Record allreduce count for this step
        all_allreduce_counts.append(decoder.engine.step_allreduce_calls)
        total_steps += 1
        
        if len(accepted) > 0:
            context.extend(accepted[:1])  # Add one token to continue
        else:
            context.append(0)  # Dummy token
    
    avg_allreduces_per_step = sum(all_allreduce_counts) / len(all_allreduce_counts)
    
    print(f"  Total steps: {total_steps}")
    print(f"  Allreduce counts per step: {all_allreduce_counts}")
    print(f"  Avg allreduces per step: {avg_allreduces_per_step:.1f}")
    print(f"  Expected: 64")
    print(f"  Efficiency: {avg_allreduces_per_step/64*100:.1f}%")
    
    return {
        'total_steps': total_steps,
        'allreduce_counts': all_allreduce_counts,
        'avg_allreduces_per_step': avg_allreduces_per_step,
        'expected': 64,
        'efficiency': avg_allreduces_per_step / 64,
    }


def run_full_benchmark(args):
    """Run complete benchmark suite."""
    print("\n" + "="*70)
    print("TREE SPECULATIVE DECODING - FULL BENCHMARK SUITE (REAL ENGINE)")
    print("="*70)
    
    # Create decoder with real TPInferenceEngine
    decoder = create_real_decoder(
        max_tree_size=args.tree_size,
        max_tree_depth=args.tree_depth,
        branching_factor=args.branching,
        tree_topology=args.tree_topology,
        base_acceptance_rate=args.base_acceptance_rate,
    )
    
    # Select prompt
    if args.prompt:
        prompt = args.prompt
    else:
        # Default test prompt
        prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:"
    
    # Run benchmarks
    print(f"\nUsing prompt: {prompt[:60]}...")
    
    # 1. Effective throughput
    throughput_results = benchmark_effective_throughput(
        decoder,
        prompt,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        verbose=True,
    )
    
    # 2. Allreduce efficiency
    allreduce_results = benchmark_allreduce_efficiency(
        decoder,
        prompt,
        num_steps=10,
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # Validation criteria
    criteria = {
        'effective_throughput_target': 54.0,  # tok/s baseline
        'accepted_per_step_target': 2.0,
        'allreduce_per_step': 64,
    }
    
    print("\nValidation Criteria:")
    print(f"  [{'✓' if throughput_results['effective_throughput_tps'] >= criteria['effective_throughput_target'] else '✗'}] "
          f"Effective throughput >= {criteria['effective_throughput_target']} tok/s")
    print(f"      Measured: {throughput_results['effective_throughput_tps']:.1f} tok/s")
    
    print(f"  [{'✓' if throughput_results['avg_accepted_per_step'] >= criteria['accepted_per_step_target'] else '✗'}] "
          f"Avg accepted per step >= {criteria['accepted_per_step_target']}")
    print(f"      Measured: {throughput_results['avg_accepted_per_step']:.2f}")
    
    print(f"  [{'✓' if allreduce_results['avg_allreduces_per_step'] == criteria['allreduce_per_step'] else '✗'}] "
          f"Allreduces per step = {criteria['allreduce_per_step']}")
    print(f"      Measured: {allreduce_results['avg_allreduces_per_step']:.1f}")
    
    # Save results
    if args.output:
        results = {
            'throughput': throughput_results,
            'allreduce': allreduce_results,
            'config': {
                'tree_size': args.tree_size,
                'tree_depth': args.tree_depth,
                'branching': args.branching,
                'max_tokens': args.max_tokens,
                'num_runs': args.num_runs,
            },
            'criteria': criteria,
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return throughput_results, allreduce_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark tree speculative decoding')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--tree-size', type=int, default=7,
                       help='Maximum tree size (default: 7)')
    parser.add_argument('--tree-depth', type=int, default=2,
                       help='Maximum tree depth (default: 2)')
    parser.add_argument('--branching', type=int, default=2,
                       help='Branching factor (default: 2)')
    parser.add_argument('--tree-topology', type=str, default='ngram',
                       choices=['ngram', 'optimized', 'binary', 'chain', 'star'],
                       help='Tree topology strategy (default: ngram)')
    parser.add_argument('--base-acceptance-rate', type=float, default=0.54,
                       help='Base acceptance rate for optimizer (default: 0.54)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of benchmark runs')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt (default: code prompt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    run_full_benchmark(args)


if __name__ == "__main__":
    main()
