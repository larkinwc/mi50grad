#!/usr/bin/env python3
"""Test compressed-tensors weight loader for Qwen 3.5 27B.

Compressed-tensors format:
  - weight_packed:  [N, K/8] INT32 (N-major packing)
  - weight_scale:   [N, K/group_size] BF16/FP16
  - weight_shape:   [2] INT64 = [N, K]
  - group_size=32 (from config.json)
  - NO zero-point tensor (symmetric quantization)

Test assertions:
  VAL-CT-001: Model files exist on server
  VAL-CT-002: Loader reads all 64 layers without errors
  VAL-CT-003: qweight shapes match expected [K/8, N] for each projection
  VAL-CT-004: scales shapes match [K/32, N] (group_size=32)
  VAL-CT-005: zeros are all zero (symmetric quant)
  VAL-CT-006: Embedding and lm_head load correctly
  VAL-CT-007: Vision layer tensors are skipped (model.visual.*, mtp.*)
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig, load_config_from_json
from src.model.compressed_tensors_loader import (
    CompressedTensorsLoader,
    detect_compressed_tensors_format,
    detect_quant_format,
    ct_dequantize,
    ct_to_engine_format,
)


# Model path on dev server
MODEL_DIR = "/opt/models/Qwen3.5-27B-CT-4bit"


def test_model_files_exist():
    """VAL-CT-001: Verify model files exist on server."""
    print("\n=== Test 1: Model files exist (VAL-CT-001) ===")
    
    model_path = Path(MODEL_DIR)
    assert model_path.exists(), f"Model directory {MODEL_DIR} does not exist"
    
    # Check for safetensors files
    sf_files = list(model_path.glob("*.safetensors"))
    assert len(sf_files) > 0, "No safetensors files found"
    print(f"  Found {len(sf_files)} safetensors files: {[f.name for f in sf_files]}")
    
    # Check for index
    index_path = model_path / "model.safetensors.index.json"
    assert index_path.exists(), "model.safetensors.index.json not found"
    print("  model.safetensors.index.json: EXISTS")
    
    # Check for config
    config_path = model_path / "config.json"
    assert config_path.exists(), "config.json not found"
    print("  config.json: EXISTS")
    
    # Verify config has quantization info
    with open(config_path) as f:
        config_data = json.load(f)
    assert "quantization_config" in config_data, "No quantization_config in config.json"
    quant_config = config_data["quantization_config"]
    assert quant_config.get("format") == "pack-quantized", "Not compressed-tensors format"
    
    # Verify group_size=32
    if "config_groups" in quant_config:
        group_0 = quant_config["config_groups"].get("group_0", {})
        weights_config = group_0.get("weights", {})
        group_size = weights_config.get("group_size", 32)
        print(f"  group_size from config: {group_size}")
        assert group_size == 32, f"Expected group_size=32, got {group_size}"
    
    print("  Model files exist: PASS")
    return True


def test_detect_format():
    """Test format detection."""
    print("\n=== Test 2: Format detection ===")
    
    is_ct = detect_compressed_tensors_format(MODEL_DIR)
    assert is_ct, "Should detect compressed-tensors format"
    print(f"  detect_compressed_tensors_format() -> {is_ct}: PASS")
    
    fmt = detect_quant_format(MODEL_DIR)
    assert fmt == "compressed-tensors", f"Expected 'compressed-tensors', got '{fmt}'"
    print(f"  detect_quant_format() -> '{fmt}': PASS")
    
    return True


def test_config_loading():
    """Test loading Qwen config from compressed-tensors model."""
    print("\n=== Test 3: Config loading ===")
    
    config = load_config_from_json(MODEL_DIR)
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    
    assert config.num_hidden_layers == 64, f"Expected 64 layers, got {config.num_hidden_layers}"
    assert config.hidden_size == 5120, f"Expected hidden_size=5120, got {config.hidden_size}"
    assert config.intermediate_size == 17408, f"Expected intermediate_size=17408, got {config.intermediate_size}"
    
    print("  Config loading: PASS")
    return True


def test_loader_initialization():
    """Test loader initialization."""
    print("\n=== Test 4: Loader initialization ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    assert loader.num_layers == 64, f"Expected 64 layers, got {loader.num_layers}"
    assert loader.group_size == 32, f"Expected group_size=32, got {loader.group_size}"
    print(f"  loader.num_layers: {loader.num_layers}")
    print(f"  loader.group_size: {loader.group_size}")
    
    # Test layer counting
    layer_count = loader.count_layers()
    print(f"  Loader counted {layer_count} text layers")
    assert layer_count == 64, f"Expected 64 text layers, got {layer_count}"
    
    print("  Loader initialization: PASS")
    return True


def test_layer_loading_shapes():
    """VAL-CT-002, VAL-CT-003, VAL-CT-004: Load layers and verify shapes."""
    print("\n=== Test 5: Layer loading and shapes (VAL-CT-002, 003, 004) ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    h = config.hidden_size       # 5120
    inter = config.intermediate_size  # 17408
    g = loader.group_size        # 32
    
    # Test layer 0 (linear attention)
    print("\n  Loading layer 0 (linear attention)...")
    w0 = loader.load_layer(0)
    assert 'layer_type' in w0
    assert w0['layer_type'] == 'linear_attention', f"Layer 0 should be linear_attention, got {w0['layer_type']}"
    print(f"  Layer 0 type: {w0['layer_type']}")
    
    # MLP shapes for layer 0
    # gate_proj: K=5120, N=17408 -> qweight=[640, 17408], scales=[160, 17408]
    assert 'gate_qweight' in w0, "Missing gate_qweight"
    assert w0['gate_qweight'].shape == (640, 17408), \
        f"gate_qweight shape mismatch: {w0['gate_qweight'].shape}, expected (640, 17408)"
    assert w0['gate_qweight'].dtype in (np.int32, np.uint32), \
        f"gate_qweight dtype should be int32/uint32, got {w0['gate_qweight'].dtype}"
    print(f"  gate_qweight: {w0['gate_qweight'].shape} {w0['gate_qweight'].dtype} PASS")
    
    assert 'gate_scales' in w0, "Missing gate_scales"
    assert w0['gate_scales'].shape == (160, 17408), \
        f"gate_scales shape mismatch: {w0['gate_scales'].shape}, expected (160, 17408)"
    assert w0['gate_scales'].dtype == np.float16, \
        f"gate_scales dtype should be float16, got {w0['gate_scales'].dtype}"
    print(f"  gate_scales: {w0['gate_scales'].shape} {w0['gate_scales'].dtype} PASS")
    
    # down_proj: K=17408, N=5120 -> qweight=[2176, 5120], scales=[544, 5120]
    assert 'down_qweight' in w0
    assert w0['down_qweight'].shape == (2176, 5120), \
        f"down_qweight shape mismatch: {w0['down_qweight'].shape}, expected (2176, 5120)"
    assert w0['down_scales'].shape == (544, 5120), \
        f"down_scales shape mismatch: {w0['down_scales'].shape}, expected (544, 5120)"
    print(f"  down_qweight: {w0['down_qweight'].shape} PASS")
    print(f"  down_scales: {w0['down_scales'].shape} PASS")
    
    # up_proj
    assert 'up_qweight' in w0
    assert w0['up_qweight'].shape == (640, 17408), \
        f"up_qweight shape mismatch: {w0['up_qweight'].shape}, expected (640, 17408)"
    print(f"  up_qweight: {w0['up_qweight'].shape} PASS")
    
    # Norm weights
    assert 'attn_norm' in w0
    assert w0['attn_norm'].shape == (h,), f"attn_norm shape mismatch: {w0['attn_norm'].shape}"
    assert w0['attn_norm'].dtype == np.float16
    print(f"  attn_norm: {w0['attn_norm'].shape} {w0['attn_norm'].dtype} PASS")
    
    # Test layer 11 (full attention)
    print("\n  Loading layer 11 (full attention)...")
    w11 = loader.load_layer(11)
    assert w11['layer_type'] == 'full_attention', f"Layer 11 should be full_attention, got {w11['layer_type']}"
    print(f"  Layer 11 type: {w11['layer_type']}")
    
    # Check attention weights exist
    assert 'q_weight' in w11, "Missing q_weight in full attention layer"
    assert 'k_weight' in w11, "Missing k_weight"
    assert 'v_weight' in w11, "Missing v_weight"
    assert 'o_weight' in w11, "Missing o_weight"
    print(f"  Full attention weights loaded: q, k, v, o projections PASS")
    
    print("  Layer loading and shapes: PASS")
    return True


def test_zeros_all_zero():
    """VAL-CT-005: Verify zeros are all zero (symmetric quantization)."""
    print("\n=== Test 6: Zeros are all zero (VAL-CT-005) ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    # Test multiple layers
    for layer_idx in [0, 1, 10, 11, 63]:
        w = loader.load_layer(layer_idx)
        for proj in ['gate', 'up', 'down']:
            zeros_key = f"{proj}_zeros"
            if zeros_key in w:
                assert np.all(w[zeros_key] == 0), \
                    f"Layer {layer_idx} {proj}_zeros should be all 0"
    
    print(f"  All zeros tensors are zero (symmetric quant): PASS")
    return True


def test_embedding_and_lm_head():
    """VAL-CT-006: Test embedding and lm_head loading."""
    print("\n=== Test 7: Embedding and LM head (VAL-CT-006) ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    # Load embedding
    embed = loader.load_embedding()
    assert embed.shape == (config.vocab_size, config.hidden_size), \
        f"Embedding shape mismatch: {embed.shape}, expected ({config.vocab_size}, {config.hidden_size})"
    assert embed.dtype == np.float16, f"Embedding dtype should be float16, got {embed.dtype}"
    print(f"  Embedding: {embed.shape} {embed.dtype} PASS")
    
    # Load LM head
    lm_head = loader.load_lm_head()
    assert lm_head.shape == (config.vocab_size, config.hidden_size), \
        f"LM head shape mismatch: {lm_head.shape}"
    assert lm_head.dtype == np.float16
    print(f"  LM head: {lm_head.shape} {lm_head.dtype} PASS")
    
    # Load final norm
    final_norm = loader.load_final_norm()
    assert final_norm.shape == (config.hidden_size,), \
        f"Final norm shape mismatch: {final_norm.shape}"
    assert final_norm.dtype == np.float16
    print(f"  Final norm: {final_norm.shape} {final_norm.dtype} PASS")
    
    print("  Embedding and LM head: PASS")
    return True


def test_vision_layers_skipped():
    """VAL-CT-007: Verify vision layers are skipped."""
    print("\n=== Test 8: Vision layers skipped (VAL-CT-007) ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    # Get all tensor keys
    all_keys = loader.get_all_tensor_keys()
    
    # Check that vision patterns are NOT in the keys
    vision_patterns = ["model.visual.", "model.merger.", "mtp."]
    vision_keys_found = []
    for key in all_keys:
        for pattern in vision_patterns:
            if key.startswith(pattern):
                vision_keys_found.append(key)
    
    assert len(vision_keys_found) == 0, \
        f"Found vision keys that should be skipped: {vision_keys_found[:10]}"
    print(f"  No vision layer tensors in loaded keys: PASS")
    
    # Verify that the loader's skip logic works
    test_vision_keys = [
        "model.visual.blocks.0.attn.qkv.weight_packed",
        "model.merger.proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight_packed",
    ]
    for key in test_vision_keys:
        assert loader._should_skip_tensor(key), f"Should skip {key}"
    print(f"  Skip patterns correctly identify vision tensors: PASS")
    
    # Count total keys vs loaded keys
    loader._load_index()
    total_keys = len(loader._index)
    loaded_keys = len(all_keys)
    skipped_keys = total_keys - loaded_keys
    print(f"  Total keys: {total_keys}, Loaded keys: {loaded_keys}, Skipped: {skipped_keys}")
    
    print("  Vision layers skipped: PASS")
    return True


def test_all_layers_load():
    """VAL-CT-002: Load all 64 layers without errors."""
    print("\n=== Test 9: Load all 64 layers (VAL-CT-002) ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    layers_loaded = 0
    for i in range(64):
        try:
            w = loader.load_layer(i)
            assert 'layer_type' in w
            assert 'gate_qweight' in w
            assert 'up_qweight' in w
            assert 'down_qweight' in w
            layers_loaded += 1
            if (i + 1) % 16 == 0:
                print(f"  Loaded layers 0-{i}: {layers_loaded} layers")
        except Exception as e:
            print(f"  ERROR loading layer {i}: {e}")
            raise
    
    assert layers_loaded == 64, f"Expected 64 layers loaded, got {layers_loaded}"
    print(f"  All 64 layers loaded successfully: PASS")
    return True


def test_dequantize_correctness():
    """Test CT dequantization produces correct output."""
    print("\n=== Test 10: Dequantization correctness ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    # Load a small weight to test dequantization
    # Load gate_proj weight_packed and weight_scale directly
    gate_packed = loader.load_tensor("model.language_model.layers.0.mlp.gate_proj.weight_packed")
    gate_scale = loader.load_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale")
    gate_shape = loader.load_tensor("model.language_model.layers.0.mlp.gate_proj.weight_shape")
    
    original_shape = (int(gate_shape[0]), int(gate_shape[1]))
    
    # Dequantize
    w_dequant = ct_dequantize(gate_packed, gate_scale, original_shape, group_size=32)
    
    # Check shape
    assert w_dequant.shape == tuple(original_shape), \
        f"Dequantized shape {w_dequant.shape} != original {original_shape}"
    print(f"  Dequantized shape: {w_dequant.shape} (matches original)")
    
    # Check dtype
    assert w_dequant.dtype == np.float32, f"Dequantized dtype should be float32, got {w_dequant.dtype}"
    print(f"  Dequantized dtype: {w_dequant.dtype}")
    
    # Check reasonable value range (weights should be small after quantization)
    max_abs = np.max(np.abs(w_dequant))
    print(f"  Max abs value: {max_abs:.6f}")
    assert max_abs < 1.0, f"Dequantized values seem too large: {max_abs}"
    
    print("  Dequantization correctness: PASS")
    return True


def test_engine_format_conversion():
    """Test conversion to engine format (K-major packing)."""
    print("\n=== Test 11: Engine format conversion ===")
    
    config = load_config_from_json(MODEL_DIR)
    loader = CompressedTensorsLoader(MODEL_DIR, config, group_size=32)
    
    # Load layer and check engine format
    w = loader.load_layer(0)
    
    # Check that qweight is K-major [K/8, N]
    assert w['gate_qweight'].shape[0] == 640, f"Expected K/8=640 rows, got {w['gate_qweight'].shape[0]}"
    assert w['gate_qweight'].shape[1] == 17408, f"Expected N=17408 cols, got {w['gate_qweight'].shape[1]}"
    print(f"  gate_qweight: K-major [K/8, N] = {w['gate_qweight'].shape} PASS")
    
    # Check that scales are [K/group_size, N]
    assert w['gate_scales'].shape[0] == 160, f"Expected K/group_size=160 rows, got {w['gate_scales'].shape[0]}"
    assert w['gate_scales'].shape[1] == 17408, f"Expected N=17408 cols, got {w['gate_scales'].shape[1]}"
    print(f"  gate_scales: [K/group_size, N] = {w['gate_scales'].shape} PASS")
    
    print("  Engine format conversion: PASS")
    return True


def main():
    print("=" * 70)
    print("Compressed-Tensors Weight Loader Tests")
    print("=" * 70)
    print(f"\nModel directory: {MODEL_DIR}")
    print()
    
    tests = [
        ("Model files exist (VAL-CT-001)", test_model_files_exist),
        ("Format detection", test_detect_format),
        ("Config loading", test_config_loading),
        ("Loader initialization", test_loader_initialization),
        ("Layer loading and shapes (VAL-CT-002, 003, 004)", test_layer_loading_shapes),
        ("Zeros all zero (VAL-CT-005)", test_zeros_all_zero),
        ("Embedding and LM head (VAL-CT-006)", test_embedding_and_lm_head),
        ("Vision layers skipped (VAL-CT-007)", test_vision_layers_skipped),
        ("Load all 64 layers (VAL-CT-002)", test_all_layers_load),
        ("Dequantization correctness", test_dequantize_correctness),
        ("Engine format conversion", test_engine_format_conversion),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                failed_tests.append(name)
                print(f"  FAIL: {name}")
        except Exception as e:
            failed += 1
            failed_tests.append(name)
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    if failed == 0:
        print("\nALL TESTS PASSED")
        print()
        print("VAL-CT-001 PASS: Model files exist")
        print("VAL-CT-002 PASS: All 64 layers loaded")
        print("VAL-CT-003 PASS: qweight shapes match [K/8, N]")
        print("VAL-CT-004 PASS: scales shapes match [K/32, N]")
        print("VAL-CT-005 PASS: zeros all zero (symmetric)")
        print("VAL-CT-006 PASS: Embedding and lm_head loaded")
        print("VAL-CT-007 PASS: Vision layers skipped")
        return 0
    else:
        print(f"\nFAILED: {failed} test(s) failed")
        print(f"Failed tests: {failed_tests}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
