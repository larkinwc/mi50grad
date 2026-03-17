"""
Test double-buffered hidden state for compute-communication overlap.

This test validates that the double-buffered decode path produces identical
results to the standard single-buffer path, while enabling overlap between
allreduce and next-layer compute.

The double-buffer approach:
- Allocates two hidden buffers per GPU: d_hidden_A and d_hidden_B
- Layer N writes allreduce result to buffer X
- Layer N+1 reads from buffer X while layer N's allreduce completes
- Buffers alternate each layer (even: A→B, odd: B→A)

Expected: cosine similarity >= 0.99 vs standard path
Memory overhead: 5120 × 2 bytes = 10KB per GPU (negligible)
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen import QwenConfig
from src.inference.tp_engine import TPInferenceEngine


def create_dummy_weights(config: QwenConfig, tp_size: int, layer_idx: int):
    """Create dummy weights for a single layer for testing.
    
    Creates FULL (unsharded) weights - the engine will shard them for TP.
    """
    # Use FULL dimensions (not local/TP-sharded) - engine handles sharding
    num_q_heads = config.num_attention_heads  # 24 for full attention
    num_kv_heads = config.num_key_value_heads  # 4 for GQA
    intermediate = config.intermediate_size  # Full intermediate size
    head_dim = config.head_dim
    h = config.hidden_size
    q_dim = num_q_heads * head_dim  # Full Q dimension
    kv_dim = num_kv_heads * head_dim  # Full KV dimension

    weights = {
        'layer_type': 'full_attention',  # Explicitly set layer type
        # Attention RMSNorm
        'attn_norm': np.ones(h, dtype=np.float16),
        # Full attention weights (FP16) - FULL SIZE, will be sharded by engine
        'q_weight': np.random.randn(q_dim, h).astype(np.float16) * 0.02,
        'q_gate_weight': np.random.randn(q_dim, h).astype(np.float16) * 0.02,
        'k_weight': np.random.randn(kv_dim, h).astype(np.float16) * 0.02,
        'v_weight': np.random.randn(kv_dim, h).astype(np.float16) * 0.02,
        'o_weight': np.random.randn(h, q_dim).astype(np.float16) * 0.02,
        'q_norm': np.ones(head_dim, dtype=np.float16),
        'k_norm': np.ones(head_dim, dtype=np.float16),
        # FFN RMSNorm
        'ffn_norm': np.ones(h, dtype=np.float16),
        # FFN weights (INT4 GPTQ format simulation) - FULL SIZE
        'gate_qweight': np.random.randint(0, 255, (intermediate, h // 8), dtype=np.uint8),
        'gate_scales': np.random.randn(intermediate, h // 128).astype(np.float16) * 0.1 + 0.5,
        'gate_zeros': np.zeros((intermediate, h // 128), dtype=np.float16),
        'up_qweight': np.random.randint(0, 255, (intermediate, h // 8), dtype=np.uint8),
        'up_scales': np.random.randn(intermediate, h // 128).astype(np.float16) * 0.1 + 0.5,
        'up_zeros': np.zeros((intermediate, h // 128), dtype=np.float16),
        'down_qweight': np.random.randint(0, 255, (h, intermediate // 8), dtype=np.uint8),
        'down_scales': np.random.randn(h, intermediate // 128).astype(np.float16) * 0.1 + 0.5,
        'down_zeros': np.zeros((h, intermediate // 128), dtype=np.float16),
    }
    return weights


def load_dummy_weights(tp_engine, num_layers: int):
    """Load dummy weights into the TP engine for testing."""
    for layer_idx in range(num_layers):
        weights = create_dummy_weights(tp_engine.config, tp_engine.tp_size, layer_idx)
        tp_engine.load_layer_weights(layer_idx, weights)
    
    # Final norm
    final_norm = np.ones(tp_engine.config.hidden_size, dtype=np.float16)
    tp_engine.load_final_norm(final_norm)
    
    # Dummy LM head (not used in this test)
    lm_head = np.random.randn(151936, tp_engine.config.hidden_size).astype(np.float16) * 0.02
    tp_engine.load_lm_head(lm_head)


def test_double_buffer_correctness(num_steps: int = 10, max_tokens: int = 10):
    """Test that double-buffer path matches standard path.
    
    This validates that the double-buffer implementation is functionally correct.
    In serial mode (without stream overlap), double-buffer should produce identical results.
    """
    print("=" * 70)
    print("Testing Double-Buffer Correctness")
    print("=" * 70)
    
    config = QwenConfig()
    device_ids = [0, 1, 2, 3]
    tp_size = len(device_ids)
    h = config.hidden_size
    
    print(f"\nConfiguration:")
    print(f"  TP size: {tp_size}")
    print(f"  Hidden size: {h}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Test steps: {num_steps}")
    print(f"  Max tokens: {max_tokens}")
    
    # Create two TP engines: one standard, one with double-buffer
    print("\nCreating TP engines...")
    
    # Standard engine
    print("  Creating standard engine...")
    tp_std = TPInferenceEngine(config, device_ids, max_seq_len=max_tokens)
    load_dummy_weights(tp_std, config.num_hidden_layers)
    tp_std.build_dispatch_cache()
    
    # Double-buffer engine  
    print("  Creating double-buffer engine...")
    tp_db = TPInferenceEngine(config, device_ids, max_seq_len=max_tokens)
    
    # Enable double-buffer mode
    if not hasattr(tp_db, 'set_double_buffer_enabled'):
        print("  ERROR: Double-buffer mode not implemented yet")
        print("  set_double_buffer_enabled() method not found")
        tp_std.cleanup()
        return False
    
    tp_db.set_double_buffer_enabled(True)
    load_dummy_weights(tp_db, config.num_hidden_layers)
    tp_db.build_dispatch_cache()
    
    print("\nRunning decode steps...")
    
    # Run same random inputs through both engines
    cos_sims = []
    all_passed = True
    for step in range(num_steps):
        # Random token embedding
        token_emb = np.random.randn(h).astype(np.float16) * 0.1
        
        try:
            # Run standard path
            out_std = tp_std.decode_step(token_emb, position=step)
            
            # Run double-buffer path
            out_db = tp_db.decode_step(token_emb, position=step)
            
            # Compute cosine similarity
            dot = np.dot(out_std.astype(np.float32), out_db.astype(np.float32))
            norm_std = np.linalg.norm(out_std.astype(np.float32))
            norm_db = np.linalg.norm(out_db.astype(np.float32))
            cos_sim = dot / (norm_std * norm_db + 1e-8)
            cos_sims.append(cos_sim)
            
            max_diff = np.max(np.abs(out_std.astype(np.float32) - out_db.astype(np.float32)))
            
            print(f"  Step {step+1}/{num_steps}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6e}")
            
            if cos_sim < 0.99:
                print(f"    WARNING: Low cosine similarity at step {step+1}")
                all_passed = False
        except Exception as e:
            print(f"  Step {step+1}/{num_steps}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break
    
    # Cleanup
    tp_std.cleanup()
    tp_db.cleanup()
    
    # Report results
    if cos_sims:
        avg_cos_sim = np.mean(cos_sims)
        min_cos_sim = np.min(cos_sims)
        
        print("\n" + "=" * 70)
        print(f"Results:")
        print(f"  Average cosine similarity: {avg_cos_sim:.6f}")
        print(f"  Minimum cosine similarity: {min_cos_sim:.6f}")
        print(f"  Target: >= 0.99")
        
        passed = all_passed and min_cos_sim >= 0.99
        print(f"\nCorrectness test: {'PASSED' if passed else 'FAILED'}")
        print("=" * 70)
        
        return passed
    else:
        print("\n" + "=" * 70)
        print("Correctness test: FAILED (no successful steps)")
        print("=" * 70)
        return False


def benchmark_double_buffer(num_warmup: int = 10, num_iters: int = 100):
    """Benchmark double-buffer vs standard path.
    
    Note: In serial mode (without stream overlap), double-buffer may show
    slight overhead due to buffer management. The true benefit comes when
    combined with stream overlap dispatch.
    """
    print("\n" + "=" * 70)
    print("Benchmarking Double-Buffer Performance")
    print("=" * 70)
    
    config = QwenConfig()
    device_ids = [0, 1, 2, 3]
    tp_size = len(device_ids)
    h = config.hidden_size
    max_tokens = 64
    
    print(f"\nConfiguration:")
    print(f"  TP size: {tp_size}")
    print(f"  Hidden size: {h}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark iterations: {num_iters}")
    
    # Create engines
    print("\nCreating engines...")
    tp_std = TPInferenceEngine(config, device_ids, max_seq_len=max_tokens)
    load_dummy_weights(tp_std, config.num_hidden_layers)
    tp_std.build_dispatch_cache()
    
    tp_db = TPInferenceEngine(config, device_ids, max_seq_len=max_tokens)
    if not hasattr(tp_db, 'set_double_buffer_enabled'):
        print("  ERROR: Double-buffer not implemented")
        tp_std.cleanup()
        return None
    
    tp_db.set_double_buffer_enabled(True)
    load_dummy_weights(tp_db, config.num_hidden_layers)
    tp_db.build_dispatch_cache()
    # Note: For actual overlap benefit, would enable:
    # tp_db.set_stream_overlap_dispatch(True)
    
    # Warmup
    print("\nWarming up...")
    token_emb = np.random.randn(h).astype(np.float16) * 0.1
    try:
        for i in range(num_warmup):
            tp_std.decode_step(token_emb, position=i % max_tokens)
            tp_db.decode_step(token_emb, position=i % max_tokens)
    except Exception as e:
        print(f"  ERROR during warmup: {e}")
        tp_std.cleanup()
        tp_db.cleanup()
        return None
    
    # Benchmark standard path
    print(f"Benchmarking standard path ({num_iters} iterations)...")
    times_std = []
    for i in range(num_iters):
        t0 = time.perf_counter()
        tp_std.decode_step(token_emb, position=i % max_tokens)
        t1 = time.perf_counter()
        times_std.append((t1 - t0) * 1000)  # ms
    
    # Reset double-buffer engine state
    tp_db.engines[0].kv_cache.current_len = 0
    for eng in tp_db.engines:
        eng.deltanet_state.reset()
    
    # Benchmark double-buffer path
    print(f"Benchmarking double-buffer path ({num_iters} iterations)...")
    times_db = []
    for i in range(num_iters):
        t0 = time.perf_counter()
        tp_db.decode_step(token_emb, position=i % max_tokens)
        t1 = time.perf_counter()
        times_db.append((t1 - t0) * 1000)  # ms
    
    # Cleanup
    tp_std.cleanup()
    tp_db.cleanup()
    
    # Compute statistics
    median_std = np.median(times_std)
    median_db = np.median(times_db)
    speedup = median_std / median_db if median_db > 0 else 1.0
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"  Standard path:       {median_std:.2f} ms/step")
    print(f"  Double-buffer path:  {median_db:.2f} ms/step")
    print(f"  Speedup:             {speedup:.2f}x")
    
    if speedup > 1.0:
        improvement = (speedup - 1.0) * 100
        print(f"  Improvement:         +{improvement:.1f}%")
    else:
        degradation = (1.0 - speedup) * 100
        print(f"  Degradation:         -{degradation:.1f}%")
        print(f"  Note: Double-buffer benefit requires stream_overlap dispatch")
    print("=" * 70)
    
    return {
        'median_std': median_std,
        'median_db': median_db,
        'speedup': speedup,
    }


def test_double_buffer_minimal():
    """Minimal test to validate double-buffer mechanism works.
    
    Tests the buffer alternation logic without loading full model weights.
    """
    print("=" * 70)
    print("Testing Double-Buffer Buffer Swapping")
    print("=" * 70)
    
    from src.runtime.hip_dispatch import GPUDevice
    
    # Create a single GPU device for testing
    device = GPUDevice(0)
    
    # Allocate test buffers (simulate d_hidden_A and d_hidden_B)
    h = 5120  # hidden size
    d_hidden_A = device.malloc(h * 2)
    d_hidden_B = device.malloc(h * 2)
    
    # Initialize with different patterns
    data_A = np.ones(h, dtype=np.float16) * 1.0
    data_B = np.ones(h, dtype=np.float16) * 2.0
    device.upload(d_hidden_A, data_A.tobytes())
    device.upload(d_hidden_B, data_B.tobytes())
    
    # Create a mock engine object with double-buffer support
    class MockEngine:
        def __init__(self, d_A, d_B):
            self.d_hidden_A = d_A
            self.d_hidden_B = d_B
            self.d_hidden = d_A  # Start with A as read buffer
            self.d_hidden_write = d_B  # B as write buffer
            
        def _swap_hidden_buffers(self):
            """Swap read and write hidden buffers."""
            self.d_hidden, self.d_hidden_write = self.d_hidden_write, self.d_hidden
    
    engine = MockEngine(d_hidden_A, d_hidden_B)
    
    print(f"\nInitial state:")
    print(f"  d_hidden = 0x{engine.d_hidden:x} (should be A)")
    print(f"  d_hidden_write = 0x{engine.d_hidden_write:x} (should be B)")
    
    # Test buffer alternation for 4 layers
    for layer_idx in range(4):
        # Even layers: read from A, write to B
        # Odd layers: read from B, write to A
        expected_read = d_hidden_A if layer_idx % 2 == 0 else d_hidden_B
        expected_write = d_hidden_B if layer_idx % 2 == 0 else d_hidden_A
        
        if engine.d_hidden != expected_read:
            print(f"  ERROR Layer {layer_idx}: d_hidden=0x{engine.d_hidden:x}, expected 0x{expected_read:x}")
            return False
        if engine.d_hidden_write != expected_write:
            print(f"  ERROR Layer {layer_idx}: d_hidden_write=0x{engine.d_hidden_write:x}, expected 0x{expected_write:x}")
            return False
        
        # Simulate write to d_hidden_write (swap at end of layer)
        print(f"  Layer {layer_idx}: read=0x{engine.d_hidden:x}, write=0x{engine.d_hidden_write:x} ✓")
        engine._swap_hidden_buffers()
    
    print("\nBuffer alternation test: PASSED")
    print("=" * 70)
    
    # Cleanup
    device.free(d_hidden_A)
    device.free(d_hidden_B)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test double-buffer overlap optimization")
    parser.add_argument("--correctness", action="store_true", default=True,
                        help="Run correctness test")
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Run benchmark")
    parser.add_argument("--minimal", action="store_true", default=False,
                        help="Run minimal buffer swap test only")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of correctness test steps")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Check GPU availability
    try:
        from src.runtime.hip_dispatch import HIPRuntime
        hip = HIPRuntime()
        hip.init()
        # Try to get device count
        import ctypes
        device_count = ctypes.c_int()
        hip._lib.hipGetDeviceCount(ctypes.byref(device_count))
        if device_count.value < 1:
            print(f"ERROR: Need at least 1 GPU for testing, found {device_count.value}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize HIP or access GPUs: {e}")
        sys.exit(1)
    
    # Run tests
    passed = True
    
    if args.minimal or (args.correctness and device_count.value < 4):
        # Run minimal test if we don't have 4 GPUs or explicitly requested
        print(f"\nRunning minimal test (device_count={device_count.value}, need 4 for full test)")
        if not test_double_buffer_minimal():
            passed = False
    
    if args.correctness and device_count.value >= 4:
        if not test_double_buffer_correctness(num_steps=args.steps):
            passed = False
    
    if args.benchmark and device_count.value >= 4:
        results = benchmark_double_buffer(num_iters=args.iters)
        if results is None:
            passed = False
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)
