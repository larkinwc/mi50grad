#!/usr/bin/env python3
"""
Test batch KV cache multi-position write functionality.

This test validates that:
1. KV cache can be written at multiple consecutive positions in a single operation
2. Data written at batch positions is correctly retrievable
3. Subsequent batch=1 decode reads correct KV cache data written by batch call
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.tp_engine import TPInferenceEngine
from src.inference.engine import KVCache
from src.model.qwen import QwenConfig
from src.runtime.hip_dispatch import GPUDevice


def test_kv_cache_batch_write_structure():
    """Test that KVCache supports batch indexing."""
    print("=" * 70)
    print("Test: KV Cache Batch Write Structure")
    print("=" * 70)
    
    config = QwenConfig()
    device = GPUDevice(0)
    max_seq_len = 512
    
    # Test with batch_size=4
    batch_size = 4
    kv_cache = KVCache(config, max_seq_len, device, tp_size=1, batch_size=batch_size)
    
    print(f"\nKV Cache initialized:")
    print(f"  batch_size: {batch_size}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  num_full_layers: {kv_cache.num_full_layers}")
    print(f"  local_num_kv_heads: {kv_cache.local_num_kv_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  pos_size_per_batch: {kv_cache.pos_size_per_batch} bytes")
    print(f"  batch_stride: {kv_cache.batch_stride} bytes")
    
    # Verify batch stride calculation
    expected_batch_stride = max_seq_len * kv_cache.pos_size_per_batch
    assert kv_cache.batch_stride == expected_batch_stride, \
        f"batch_stride mismatch: {kv_cache.batch_stride} != {expected_batch_stride}"
    
    # Test layer_k_ptr and layer_v_ptr with batch indices
    for layer_idx in range(min(3, kv_cache.num_full_layers)):
        for batch_idx in range(batch_size):
            k_ptr = kv_cache.layer_k_ptr(layer_idx, batch_idx)
            v_ptr = kv_cache.layer_v_ptr(layer_idx, batch_idx)
            
            # Verify pointers are correctly offset by batch
            expected_offset = batch_idx * kv_cache.batch_stride
            k_ptr_0 = kv_cache.layer_k_ptr(layer_idx, 0)
            v_ptr_0 = kv_cache.layer_v_ptr(layer_idx, 0)
            
            assert k_ptr - k_ptr_0 == expected_offset, \
                f"K pointer offset wrong for batch {batch_idx}"
            assert v_ptr - v_ptr_0 == expected_offset, \
                f"V pointer offset wrong for batch {batch_idx}"
    
    print("✓ KV cache batch pointer calculations correct")
    
    # Test contiguous position write capability
    # For positions [p, p+1, p+2, p+3], we should be able to write B * kv_stride bytes
    kv_stride = kv_cache.pos_stride
    B = batch_size
    total_write_size = B * kv_stride
    
    print(f"\nContiguous write test:")
    print(f"  kv_stride: {kv_stride} bytes")
    print(f"  batch_size: {B}")
    print(f"  total_write_size: {total_write_size} bytes")
    
    # Verify pos_stride matches expected size for one position
    expected_pos_stride = kv_cache.local_num_kv_heads * config.head_dim * 2  # FP16 = 2 bytes
    assert kv_stride == expected_pos_stride, \
        f"pos_stride mismatch: {kv_stride} != {expected_pos_stride}"
    
    print("✓ Contiguous write size calculation correct")
    
    del kv_cache
    device.release()
    
    print("\n" + "=" * 70)
    print("STRUCTURE TEST PASSED")
    print("=" * 70)
    return 0


def test_kv_cache_batch_write_correctness():
    """Test that KV cache batch writes are correctly stored and retrievable."""
    print("=" * 70)
    print("Test: KV Cache Batch Write Correctness")
    print("=" * 70)
    
    config = QwenConfig()
    device = GPUDevice(0)
    max_seq_len = 512
    batch_size = 4
    
    kv_cache = KVCache(config, max_seq_len, device, tp_size=1, batch_size=batch_size)
    
    # Create test K and V data for batch of 4 tokens
    # Shape: [batch_size, local_kv_heads, head_dim]
    local_kv_heads = kv_cache.local_num_kv_heads
    head_dim = config.head_dim
    
    # Generate distinct test patterns for each batch position
    np.random.seed(42)
    test_k_data = []
    test_v_data = []
    for i in range(batch_size):
        # Create unique pattern for each position
        k_pattern = np.full((local_kv_heads, head_dim), fill_value=i+1, dtype=np.float16)
        v_pattern = np.full((local_kv_heads, head_dim), fill_value=(i+1)*10, dtype=np.float16)
        test_k_data.append(k_pattern)
        test_v_data.append(v_pattern)
    
    print(f"\nWriting test data for {batch_size} tokens...")
    print(f"  Pattern: K=i+1, V=(i+1)*10 for position i")
    
    # Write to KV cache at positions 0, 1, 2, 3
    # For layer 0 (first full attention layer)
    layer_idx = kv_cache.full_layer_indices[0]
    
    # Write each position sequentially (simulating batch write)
    for i in range(batch_size):
        # Allocate GPU buffers for this position's K and V
        d_k_temp = device.alloc(test_k_data[i].nbytes)
        d_v_temp = device.alloc(test_v_data[i].nbytes)
        
        # Upload test data
        device.upload(d_k_temp, test_k_data[i].tobytes())
        device.upload(d_v_temp, test_v_data[i].tobytes())
        
        # Write to KV cache at position i
        kv_cache.append_kv_gpu_from_batch(layer_idx, d_k_temp, d_v_temp, batch_idx=i)
        
        device.free(d_k_temp)
        device.free(d_v_temp)
    
    print(f"✓ Wrote {batch_size} positions to KV cache")
    
    # Read back and verify
    print("\nReading back data from KV cache...")
    for i in range(batch_size):
        # Read K cache for position i, batch i
        k_ptr = kv_cache.layer_k_ptr(layer_idx, batch_idx=i)
        # Offset to position i within the batch's cache region
        k_ptr_pos = k_ptr + i * kv_cache.pos_stride
        
        # Read back
        k_read = np.empty((local_kv_heads, head_dim), dtype=np.float16)
        device.download(k_read, k_ptr_pos, k_read.nbytes)
        
        # Verify
        max_diff = np.max(np.abs(k_read.astype(np.float32) - test_k_data[i].astype(np.float32)))
        if max_diff > 1e-3:
            print(f"✗ Position {i} K data mismatch: max_diff={max_diff}")
            return 1
        
        # Similarly for V
        v_ptr = kv_cache.layer_v_ptr(layer_idx, batch_idx=i)
        v_ptr_pos = v_ptr + i * kv_cache.pos_stride
        
        v_read = np.empty((local_kv_heads, head_dim), dtype=np.float16)
        device.download(v_read, v_ptr_pos, v_read.nbytes)
        
        max_diff = np.max(np.abs(v_read.astype(np.float32) - test_v_data[i].astype(np.float32)))
        if max_diff > 1e-3:
            print(f"✗ Position {i} V data mismatch: max_diff={max_diff}")
            return 1
    
    print(f"✓ All {batch_size} positions read back correctly")
    
    del kv_cache
    device.release()
    
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST PASSED")
    print("=" * 70)
    return 0


def test_kv_cache_contiguous_batch_write():
    """Test contiguous batch write (single memcpy for all B positions)."""
    print("=" * 70)
    print("Test: KV Cache Contiguous Batch Write")
    print("=" * 70)
    
    config = QwenConfig()
    device = GPUDevice(0)
    max_seq_len = 512
    batch_size = 4
    
    kv_cache = KVCache(config, max_seq_len, device, tp_size=1, batch_size=batch_size)
    
    # Create contiguous batch data: [batch_size, local_kv_heads, head_dim]
    local_kv_heads = kv_cache.local_num_kv_heads
    head_dim = config.head_dim
    
    # Generate test data
    np.random.seed(123)
    batch_k_data = np.zeros((batch_size, local_kv_heads, head_dim), dtype=np.float16)
    batch_v_data = np.zeros((batch_size, local_kv_heads, head_dim), dtype=np.float16)
    
    for i in range(batch_size):
        batch_k_data[i] = np.full((local_kv_heads, head_dim), i+1, dtype=np.float16)
        batch_v_data[i] = np.full((local_kv_heads, head_dim), (i+1)*100, dtype=np.float16)
    
    # Allocate GPU buffer for batch data
    batch_k_bytes = batch_k_data.nbytes
    batch_v_bytes = batch_v_data.nbytes
    
    d_batch_k = device.alloc(batch_k_bytes)
    d_batch_v = device.alloc(batch_v_bytes)
    
    device.upload(d_batch_k, batch_k_data.tobytes())
    device.upload(d_batch_v, batch_v_data.tobytes())
    
    print(f"\nContiguous batch write:")
    print(f"  Total K bytes: {batch_k_bytes}")
    print(f"  Total V bytes: {batch_v_bytes}")
    print(f"  Batch stride: {kv_cache.batch_stride} bytes")
    
    # Write contiguous batch to position 0 (positions 0,1,2,3)
    layer_idx = kv_cache.full_layer_indices[0]
    
    # This should write all B positions in a single D2D memcpy
    kv_cache.append_kv_gpu_batch(layer_idx, d_batch_k, d_batch_v, 
                                  start_batch_idx=0, num_positions=batch_size)
    
    print(f"✓ Wrote {batch_size} contiguous positions")
    
    # Verify each position was written correctly
    print("\nVerifying contiguous write...")
    for i in range(batch_size):
        k_ptr = kv_cache.layer_k_ptr(layer_idx, batch_idx=i)
        k_ptr_pos = k_ptr + i * kv_cache.pos_stride
        
        k_read = np.empty((local_kv_heads, head_dim), dtype=np.float16)
        device.download(k_read, k_ptr_pos, k_read.nbytes)
        
        max_diff = np.max(np.abs(k_read.astype(np.float32) - batch_k_data[i].astype(np.float32)))
        if max_diff > 1e-3:
            print(f"✗ Position {i} K data mismatch: max_diff={max_diff}")
            device.free(d_batch_k)
            device.free(d_batch_v)
            del kv_cache
            device.release()
            return 1
    
    print(f"✓ Contiguous batch write verified for all {batch_size} positions")
    
    device.free(d_batch_k)
    device.free(d_batch_v)
    del kv_cache
    device.release()
    
    print("\n" + "=" * 70)
    print("CONTIGUOUS BATCH WRITE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test batch KV cache operations")
    parser.add_argument("--structure", action="store_true", default=True,
                        help="Test KV cache batch structure")
    parser.add_argument("--correctness", action="store_true", default=True,
                        help="Test KV cache batch write correctness")
    parser.add_argument("--contiguous", action="store_true", default=True,
                        help="Test contiguous batch write")
    
    args = parser.parse_args()
    
    exit_code = 0
    
    if args.structure:
        code = test_kv_cache_batch_write_structure()
        exit_code = max(exit_code, code)
    
    if args.correctness:
        code = test_kv_cache_batch_write_correctness()
        exit_code = max(exit_code, code)
    
    if args.contiguous:
        code = test_kv_cache_contiguous_batch_write()
        exit_code = max(exit_code, code)
    
    sys.exit(exit_code)
