"""
Benchmark script for tree-based speculative decoding.

Measures:
1. Effective throughput: accepted_tokens / wall_time
2. Acceptance rate per verification step (target: >= 2.0)
3. Speedup vs baseline (~54 tok/s for TP=4)
4. Allreduce efficiency: 64 calls per verification regardless of tree size

Usage:
    python3 tests/bench_tree_speculative.py
    
    # With custom parameters
    python3 tests/bench_tree_speculative.py --max-tokens 100 --tree-size 7 --num-runs 5
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.tree_speculative import TreeSpeculativeDecoder
from src.inference.speculative import NgramCache


# ============================================================================
# Mock Components for Infrastructure Testing
# ============================================================================

class MockTokenizer:
    """Simple character-level tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 0
    
    def encode(self, text: str):
        """Encode text to token IDs (simple char-level)."""
        return [ord(c) % 256 for c in text]
    
    def decode(self, tokens):
        """Decode token IDs to text."""
        return ''.join(chr(t) for t in tokens)


class MockEngine:
    """Mock TP engine for benchmarking tree speculative infrastructure."""
    
    def __init__(self, hidden_size: int = 5120, num_layers: int = 64):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
        })()
        
        # Track allreduce calls
        self.total_allreduce_calls = 0
        self.decode_step_tree_calls = 0
        self.step_allreduce_calls = 0  # Reset per step
    
    def decode_step_tree(self, 
                         token_embeddings,
                         tree_mask,
                         kv_embeddings=None):
        """Mock tree decode step with realistic timing."""
        self.decode_step_tree_calls += 1
        self.step_allreduce_calls = self.num_layers  # 64 per call (reset for this step)
        self.total_allreduce_calls += self.num_layers
        
        tree_size = len(token_embeddings)
        
        # Simulate compute time (proportional to tree size)
        # Base: 18.5ms for single token, +0.5ms per additional tree token
        compute_time_ms = 18.5 + (tree_size - 1) * 0.5
        
        # Simulate allreduce time: 64 calls × 79us = 5.06ms
        allreduce_time_ms = 64 * 0.079
        
        # Total simulated time
        time.sleep((compute_time_ms + allreduce_time_ms) / 1000.0)
        
        # Return mock outputs
        import numpy as np
        outputs = []
        for i in range(tree_size):
            # Make output correlated with input embedding for deterministic behavior
            out = token_embeddings[i].copy().astype(np.float16)
            outputs.append(out)
        
        return outputs


def create_mock_decoder(max_tree_size: int = 7,
                        max_tree_depth: int = 2,
                        branching_factor: int = 2) -> TreeSpeculativeDecoder:
    """Create TreeSpeculativeDecoder with mock components."""
    import numpy as np
    
    engine = MockEngine(hidden_size=128, num_layers=64)
    vocab_size = 1000
    embed_weight = np.random.randn(vocab_size, 128).astype(np.float16) * 0.01
    lm_head_weight = np.random.randn(vocab_size, 128).astype(np.float16) * 0.01
    tokenizer = MockTokenizer()
    
    decoder = TreeSpeculativeDecoder(
        engine=engine,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        tokenizer=tokenizer,
        ngram_size=3,
        max_tree_size=max_tree_size,
        max_tree_depth=max_tree_depth,
        branching_factor=branching_factor,
        tree_topology='ngram',
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
    
    input_ids = decoder.tokenizer.encode(prompt)
    
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
    
    input_ids = decoder.tokenizer.encode(prompt)
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
    print("TREE SPECULATIVE DECODING - FULL BENCHMARK SUITE")
    print("="*70)
    
    # Create decoder
    decoder = create_mock_decoder(
        max_tree_size=args.tree_size,
        max_tree_depth=args.tree_depth,
        branching_factor=args.branching,
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
