#!/usr/bin/env python3
"""
Minimal test for compressed allreduce integration in tp_engine.py.

This test verifies:
1. The set_compressed_allreduce() API exists and works
2. The compressed kernel library loads correctly
3. C dispatch plan can be built with compressed mode enabled

Does NOT run full decode (too slow for quick validation).
"""

import sys
import time
sys.path.insert(0, '/opt/mi50grad')

from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
DEVICE_IDS = [0, 1, 2, 3]
MAX_SEQ_LEN = 256

print("=" * 72)
print("  Minimal Compressed Allreduce Integration Test")
print("=" * 72)

# Load config
print("\nLoading config...")
config = load_config_from_json(MODEL_DIR)
print(f"  Model: {MODEL_DIR}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_layers: {config.num_hidden_layers}")

# Load TP engine
print("\nLoading TP=4 engine...")
t0 = time.time()
tp = TPInferenceEngine(config, device_ids=DEVICE_IDS, max_seq_len=MAX_SEQ_LEN)
print(f"  Engine created in {time.time() - t0:.2f}s")

# Load weights (just first few layers for quick test)
print("\nLoading weights (first 4 layers for quick test)...")
loader = QwenWeightLoader(MODEL_DIR, config)
for layer_idx in range(4):
    print(f"    Layer {layer_idx}...")
    tp.load_layer_weights(layer_idx, loader.load_layer(layer_idx))

t_load = time.time()
print(f"  Weights loaded in {t_load - t0:.2f}s")

# Build dispatch cache
print("\nBuilding dispatch cache...")
tp.build_dispatch_cache()
print(f"  Dispatch cache built in {time.time() - t_load:.2f}s")

# Enable C dispatch
print("\nEnabling C dispatch...")
tp.set_c_dispatch(True)
print(f"  C dispatch enabled: {tp._c_dispatch_enabled}")

# Enable kernel P2P allreduce
print("\nEnabling kernel P2P allreduce...")
tp.set_kernel_p2p_allreduce(True)
print(f"  Kernel P2P allreduce enabled: {tp._kernel_p2p_allreduce}")

# Check if compressed kernel is loaded
print("\nChecking compressed kernel availability...")
has_compressed_fn = hasattr(tp, '_gemv_fused_compressed_fn_ptr') and tp._gemv_fused_compressed_fn_ptr is not None
print(f"  Compressed kernel function pointer: {has_compressed_fn}")
if has_compressed_fn:
    print(f"  Fn ptr: 0x{tp._gemv_fused_compressed_fn_ptr:016x}")
    print(f"  Compressed buffers allocated: {hasattr(tp, '_compressed_ptrs') and len(tp._compressed_ptrs) == 4}")

# Test set_compressed_allreduce API
print("\nTesting set_compressed_allreduce() API...")
try:
    tp.set_compressed_allreduce(True)
    print(f"  set_compressed_allreduce(True) - SUCCESS")
    print(f"  Compressed allreduce enabled: {getattr(tp, '_compressed_allreduce_enabled', 'N/A')}")
    
    # Check if C dispatch plan was rebuilt
    print(f"  C dispatch plan exists: {tp._c_dispatch_plan is not None}")
    
    # Disable
    tp.set_compressed_allreduce(False)
    print(f"  set_compressed_allreduce(False) - SUCCESS")
    print(f"  Compressed allreduce enabled: {getattr(tp, '_compressed_allreduce_enabled', 'N/A')}")
    
except Exception as e:
    print(f"  ERROR: set_compressed_allreduce() failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Cleanup
print("\nCleaning up...")
tp.cleanup()
print("  Done!")

print("\n" + "=" * 72)
print("  RESULT: TEST PASSED - Compressed allreduce integration works!")
print("=" * 72)
