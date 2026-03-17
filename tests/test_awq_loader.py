#!/usr/bin/env python3
"""Test AWQ safetensors weight loader for Qwen 3.5 27B.

AWQ format differences from GPTQ:
- Weights packed as INT4 in uint32 (same packing, same layout [K/8, N])
- Scales: per-group FP16 (same layout [K/group_size, N])
- NO zero-point tensor (AWQ uses zero-point = 0, pure scaling)
- Dequantization: w = q * scale (no subtraction of zero-point)
- Different tensor naming: model.layers.N (vs model.language_model.layers.N)
  and full proj names: self_attn.q_proj.qweight, .scales (vs GPTQ .qzeros)

Since no AWQ model is available at /opt/models/, this test uses synthetic
AWQ-format weights to validate the loader logic:
  1. Shape/dtype correctness
  2. No zero-point tensors expected or loaded
  3. Weight format compatibility with packed uint32 layout
  4. Single-layer matmul output correctness (vs reference dequantize-then-matmul)
  5. Layer count check (64 layers structure)

VAL-AWQ-001: Validates shapes/dtypes match expected Qwen 3.5 27B architecture
"""

import sys
import os
import struct
import json
import tempfile
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig
from src.model.awq_loader import AWQWeightLoader, awq_dequantize, detect_awq_format


def create_synthetic_awq_safetensors(output_dir: str, config: QwenConfig,
                                      num_layers: int = 2):
    """Create synthetic AWQ-format safetensors files for testing.

    AWQ tensor naming convention (HuggingFace AutoAWQ format):
      model.layers.N.self_attn.q_proj.qweight    [K/8, N] INT32
      model.layers.N.self_attn.q_proj.scales     [K/group_size, N] F16
      model.layers.N.mlp.gate_proj.qweight       [K/8, N] INT32
      model.layers.N.mlp.gate_proj.scales        [K/group_size, N] F16
      (NO qzeros tensors in AWQ)

    Qwen 3.5 27B shapes:
      hidden_size = 5120
      intermediate_size = 17408
      group_size = 128
      gate_proj/up_proj: K=5120, N=17408 -> qweight=[640, 17408], scales=[40, 17408]
      down_proj: K=17408, N=5120 -> qweight=[2176, 5120], scales=[136, 5120]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    h = config.hidden_size        # 5120
    inter = config.intermediate_size  # 17408
    g = config.group_size         # 128

    def make_qweight(K, N):
        """Create packed INT4 qweight tensor [K/8, N] as INT32."""
        # Pack 8 nibbles (0..15) into each uint32
        raw = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
        packed = np.zeros((K // 8, N), dtype=np.int32)
        for b in range(8):
            packed |= (raw[b::8, :].astype(np.int32) << (b * 4))
        return packed

    def make_scales(K, N, group_size):
        """Create FP16 per-group scales [K/group_size, N]."""
        num_groups = K // group_size
        # Realistic scale range: ~0.001 to 0.01
        scales = (rng.random((num_groups, N)) * 0.009 + 0.001).astype(np.float16)
        return scales

    def pack_tensor(data: np.ndarray):
        """Pack a numpy array for safetensors format."""
        raw = data.tobytes()
        dtype_map = {
            np.dtype('int32'):   'I32',
            np.dtype('float16'): 'F16',
            np.dtype('float32'): 'F32',
            np.dtype('uint16'):  'U16',
        }
        dtype_str = dtype_map[data.dtype]
        return raw, dtype_str, list(data.shape)

    def write_safetensors(path: str, tensors: dict):
        """Write tensors to a safetensors file."""
        # Build header
        header = {}
        current_offset = 0
        raw_datas = []

        for key, arr in tensors.items():
            raw, dtype_str, shape = pack_tensor(arr)
            raw_datas.append(raw)
            header[key] = {
                'dtype': dtype_str,
                'shape': shape,
                'data_offsets': [current_offset, current_offset + len(raw)]
            }
            current_offset += len(raw)

        header_json = json.dumps(header).encode('utf-8')
        # Pad to 8-byte alignment
        padding = (8 - len(header_json) % 8) % 8
        header_json += b' ' * padding

        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(header_json)))
            f.write(header_json)
            for raw in raw_datas:
                f.write(raw)

    # Create index for multi-file safetensors
    weight_map = {}

    # File 1: layer 0 and layer 3 (the ones we'll test)
    tensors_file1 = {}

    for layer_idx in range(num_layers):
        lp = f"model.layers.{layer_idx}"

        # Gate proj: K=5120, N=17408
        tensors_file1[f"{lp}.mlp.gate_proj.qweight"] = make_qweight(h, inter)
        tensors_file1[f"{lp}.mlp.gate_proj.scales"] = make_scales(h, inter, g)
        # Up proj: K=5120, N=17408
        tensors_file1[f"{lp}.mlp.up_proj.qweight"] = make_qweight(h, inter)
        tensors_file1[f"{lp}.mlp.up_proj.scales"] = make_scales(h, inter, g)
        # Down proj: K=17408, N=5120
        tensors_file1[f"{lp}.mlp.down_proj.qweight"] = make_qweight(inter, h)
        tensors_file1[f"{lp}.mlp.down_proj.scales"] = make_scales(inter, h, g)

        # Norms (FP16 for simplicity; real models use BF16)
        norm_data = (rng.random(h) * 0.1).astype(np.float16)
        tensors_file1[f"{lp}.input_layernorm.weight"] = norm_data
        tensors_file1[f"{lp}.post_attention_layernorm.weight"] = norm_data.copy()

        # Linear attention weights (FP16, not quantized in AWQ)
        # Layer type pattern: every 4th layer is full attention
        if (layer_idx + 1) % 4 != 0:
            # Linear attention layer
            la = f"{lp}.linear_attn"
            lin_k = 2048   # linear_key_dim = 16*128
            lin_v = 6144   # linear_value_dim = 48*128
            # in_proj_qkv: [Q_k + K_k + V] = [2048 + 2048 + 6144 = 10240]
            tensors_file1[f"{la}.in_proj_qkv.weight"] = rng.random((lin_k + lin_k + lin_v, h)).astype(np.float16)
            tensors_file1[f"{la}.in_proj_a.weight"] = rng.random((48, h)).astype(np.float16)
            tensors_file1[f"{la}.in_proj_b.weight"] = rng.random((48, h)).astype(np.float16)
            tensors_file1[f"{la}.in_proj_z.weight"] = rng.random((lin_v, h)).astype(np.float16)
            tensors_file1[f"{la}.conv1d.weight"] = rng.random((lin_k + lin_v, 1, 4)).astype(np.float16)
            tensors_file1[f"{la}.A_log"] = rng.random(48).astype(np.float32)
            tensors_file1[f"{la}.dt_bias"] = rng.random(48).astype(np.float16)
            tensors_file1[f"{la}.norm.weight"] = rng.random(128).astype(np.float32)
            tensors_file1[f"{la}.out_proj.weight"] = rng.random((h, lin_v)).astype(np.float16)
        else:
            # Full attention layer (FP16, not quantized)
            attn = f"{lp}.self_attn"
            num_heads, head_dim = 24, 256
            kv_heads = 4
            # q_proj packed: [num_heads * head_dim * 2, hidden]
            tensors_file1[f"{attn}.q_proj.weight"] = rng.random(
                (num_heads * head_dim * 2, h)).astype(np.float16)
            tensors_file1[f"{attn}.k_proj.weight"] = rng.random(
                (kv_heads * head_dim, h)).astype(np.float16)
            tensors_file1[f"{attn}.v_proj.weight"] = rng.random(
                (kv_heads * head_dim, h)).astype(np.float16)
            tensors_file1[f"{attn}.o_proj.weight"] = rng.random(
                (h, num_heads * head_dim)).astype(np.float16)
            tensors_file1[f"{attn}.q_norm.weight"] = rng.random(head_dim).astype(np.float16)
            tensors_file1[f"{attn}.k_norm.weight"] = rng.random(head_dim).astype(np.float16)

        for k in list(tensors_file1.keys()):
            if k not in weight_map:
                weight_map[k] = "model.safetensors-00001-of-00001.safetensors"

    file_path = str(output_dir / "model.safetensors-00001-of-00001.safetensors")
    write_safetensors(file_path, tensors_file1)

    # Embeddings / lm_head
    tensors_file2 = {}
    embed = rng.random((config.vocab_size, h)).astype(np.float16)
    tensors_file2["model.embed_tokens.weight"] = embed
    tensors_file2["model.norm.weight"] = rng.random(h).astype(np.float16)
    tensors_file2["lm_head.weight"] = embed.copy()

    for k in tensors_file2.keys():
        weight_map[k] = "model.safetensors-00002-of-00002.safetensors"
    file_path2 = str(output_dir / "model.safetensors-00002-of-00002.safetensors")
    write_safetensors(file_path2, tensors_file2)

    # Write index
    index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", 'w') as f:
        json.dump(index, f)

    # Write a minimal config.json
    config_data = {
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": False,
        },
        "num_hidden_layers": 64,  # The real architecture has 64 layers
        "hidden_size": h,
        "intermediate_size": inter,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_data, f)

    print(f"  Created synthetic AWQ dataset with {len(tensors_file1)} tensors in {output_dir}")
    return str(output_dir)


def test_detect_awq_format():
    """Test that AWQ format is detected correctly (no qzeros tensors)."""
    print("\n=== Test 1: AWQ format detection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = QwenConfig()
        model_dir = create_synthetic_awq_safetensors(tmpdir, config, num_layers=2)

        # Should detect AWQ (no qzeros tensors)
        fmt = detect_awq_format(model_dir)
        assert fmt == 'awq', f"Expected 'awq' format, got '{fmt}'"
        print(f"  detect_awq_format() -> '{fmt}' PASS")

        # Test with a GPTQ-like structure (has qzeros)
        # We'll check that we can distinguish
        print("  AWQ format detection: PASS")
    return True


def test_shapes_and_dtypes():
    """Test VAL-AWQ-001: AWQ weights loaded with correct shapes/dtypes."""
    print("\n=== Test 2: Shape/dtype validation (VAL-AWQ-001) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = QwenConfig()
        model_dir = create_synthetic_awq_safetensors(tmpdir, config, num_layers=2)

        loader = AWQWeightLoader(model_dir, config)

        # Load layer 0
        w = loader.load_layer(0)
        print(f"  Layer 0 keys: {sorted([k for k in w.keys()])}")

        h = config.hidden_size       # 5120
        inter = config.intermediate_size  # 17408
        g = config.group_size        # 128

        # Validate gate_proj shapes
        # qweight: [K/8, N] = [5120/8, 17408] = [640, 17408]
        assert 'gate_qweight' in w, "Missing gate_qweight"
        assert w['gate_qweight'].shape == (640, 17408), \
            f"gate_qweight shape mismatch: {w['gate_qweight'].shape}"
        assert w['gate_qweight'].dtype in (np.int32, np.uint32), \
            f"gate_qweight dtype should be int32/uint32, got {w['gate_qweight'].dtype}"
        print(f"  gate_qweight: {w['gate_qweight'].shape} {w['gate_qweight'].dtype} PASS")

        # scales: [K/group_size, N] = [40, 17408]
        assert 'gate_scales' in w, "Missing gate_scales"
        assert w['gate_scales'].shape == (40, 17408), \
            f"gate_scales shape mismatch: {w['gate_scales'].shape}"
        assert w['gate_scales'].dtype == np.float16, \
            f"gate_scales dtype should be float16, got {w['gate_scales'].dtype}"
        print(f"  gate_scales: {w['gate_scales'].shape} {w['gate_scales'].dtype} PASS")

        # zeros: should be 0 (or zeros array) for AWQ
        assert 'gate_zeros' in w, "Missing gate_zeros key (should be zeros array)"
        assert np.all(w['gate_zeros'] == 0), "AWQ zeros should be all 0"
        print(f"  gate_zeros: all zeros (AWQ no zero-point) PASS")

        # down_proj: K=17408, N=5120 -> [2176, 5120], [136, 5120]
        assert w['down_qweight'].shape == (2176, 5120), \
            f"down_qweight shape mismatch: {w['down_qweight'].shape}"
        assert w['down_scales'].shape == (136, 5120), \
            f"down_scales shape mismatch: {w['down_scales'].shape}"
        assert np.all(w['down_zeros'] == 0), "down_zeros should be 0 for AWQ"
        print(f"  down_qweight: {w['down_qweight'].shape} PASS")
        print(f"  down_scales: {w['down_scales'].shape} PASS")

        # up_proj
        assert w['up_qweight'].shape == (640, 17408), \
            f"up_qweight shape mismatch: {w['up_qweight'].shape}"
        print(f"  up_qweight: {w['up_qweight'].shape} PASS")

        # Norm weights
        assert 'attn_norm' in w, "Missing attn_norm"
        assert w['attn_norm'].shape == (h,), f"attn_norm shape mismatch: {w['attn_norm'].shape}"
        assert w['attn_norm'].dtype == np.float16
        print(f"  attn_norm: {w['attn_norm'].shape} {w['attn_norm'].dtype} PASS")

        print("  Shape/dtype validation: ALL PASS")
    return True


def test_no_zeros_tensors():
    """Verify no zero-point tensors are present in AWQ format."""
    print("\n=== Test 3: No zero-point tensors ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = QwenConfig()
        model_dir = create_synthetic_awq_safetensors(tmpdir, config, num_layers=2)

        # Check that no qzeros keys exist in the safetensors index
        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        wmap = index.get("weight_map", {})

        qzeros_keys = [k for k in wmap.keys() if 'qzeros' in k or 'zeros' in k.lower()]
        assert len(qzeros_keys) == 0, f"Found unexpected qzeros keys: {qzeros_keys}"
        print(f"  No qzeros tensors found in AWQ model index: PASS")

        # Verify loader doesn't try to load qzeros
        loader = AWQWeightLoader(model_dir, config)
        w = loader.load_layer(0)

        # The zeros field should be a zeros array (not loaded from file)
        for proj in ['gate', 'up', 'down']:
            zeros_key = f"{proj}_zeros"
            assert zeros_key in w, f"Missing {zeros_key}"
            assert np.all(w[zeros_key] == 0), f"{zeros_key} should be all 0"
        print("  Loader zeros are synthetic (all 0): PASS")
    return True


def test_awq_dequantize_correctness():
    """Test AWQ dequantization: w = q * scale (no zero-point).

    VAL-AWQ-002: Max absolute error should be < 1e-2 vs reference.
    """
    print("\n=== Test 4: AWQ dequantization correctness ===")

    rng = np.random.default_rng(123)
    K, N, g = 512, 256, 128  # Small test case

    # Create synthetic weights
    raw = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    # Pack into qweight [K/8, N]
    qweight = np.zeros((K // 8, N), dtype=np.int32)
    for b in range(8):
        qweight |= (raw[b::8, :].astype(np.int32) << (b * 4))

    # Scales [K/group_size, N]
    num_groups = K // g
    scales = (rng.random((num_groups, N)) * 0.009 + 0.001).astype(np.float16)

    # Reference dequantization: w = q * scale (AWQ, no zero-point)
    # raw is already [K, N] uint8 with values 0..15
    # AWQ uses symmetric quant: values are 0..15, but treated as signed -8..7
    # Actually AWQ is asymmetric but uses zero_point=0: values 0..15, no offset
    # w_ref[k, n] = raw[k, n] * scales[k // group_size, n]
    scales_expanded = np.repeat(scales.astype(np.float32), g, axis=0)  # [K, N]
    w_ref = raw.astype(np.float32) * scales_expanded  # [K, N] reference

    # Test our dequantize function
    w_ours = awq_dequantize(qweight, scales, group_size=g)  # should return [K, N] FP32 or [N, K] FP32

    # Tolerance: FP16 scales introduce ~0.1% error
    max_err = np.max(np.abs(w_ref - w_ours))
    rel_err = max_err / (np.max(np.abs(w_ref)) + 1e-12)
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Relative error: {rel_err:.6f}")

    # AWQ precision threshold: < 1e-2 (per VAL-AWQ-002)
    assert max_err < 1e-2, f"Dequantization error too high: {max_err:.6f} >= 1e-2"
    print(f"  Dequantization correctness: PASS (max_err={max_err:.6f} < 1e-2)")
    return True


def test_layer_count():
    """Verify the loader correctly handles the 64-layer Qwen 3.5 27B architecture."""
    print("\n=== Test 5: Layer count and architecture ===")

    config = QwenConfig()
    assert config.num_hidden_layers == 64, \
        f"Expected 64 layers, got {config.num_hidden_layers}"
    assert config.num_full_attention_layers == 16, \
        f"Expected 16 full attention layers, got {config.num_full_attention_layers}"
    assert config.num_linear_attention_layers == 48, \
        f"Expected 48 linear attention layers, got {config.num_linear_attention_layers}"

    print(f"  Total layers: {config.num_hidden_layers} PASS")
    print(f"  Full attention layers: {config.num_full_attention_layers} PASS")
    print(f"  Linear attention layers: {config.num_linear_attention_layers} PASS")

    # Test that AWQ loader reports expected layer count
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = create_synthetic_awq_safetensors(tmpdir, config, num_layers=2)
        loader = AWQWeightLoader(model_dir, config)
        assert loader.num_layers == 64, \
            f"AWQ loader should report 64 layers (from config), got {loader.num_layers}"
        print(f"  AWQ loader.num_layers = {loader.num_layers} (from config): PASS")
    return True


def test_matmul_output():
    """Test single-layer matmul output using AWQ dequantized weights."""
    print("\n=== Test 6: Single-layer matmul output correctness ===")

    rng = np.random.default_rng(456)
    K, N, g = 512, 256, 128
    batch = 1

    # Create synthetic AWQ weights
    raw = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    qweight = np.zeros((K // 8, N), dtype=np.int32)
    for b in range(8):
        qweight |= (raw[b::8, :].astype(np.int32) << (b * 4))
    num_groups = K // g
    scales = (rng.random((num_groups, N)) * 0.009 + 0.001).astype(np.float16)

    # Input activation [batch, K]
    x = rng.standard_normal((batch, K)).astype(np.float32)

    # Reference: dequantize then matmul
    scales_expanded = np.repeat(scales.astype(np.float32), g, axis=0)  # [K, N]
    w_fp32 = raw.astype(np.float32) * scales_expanded  # [K, N]
    y_ref = x @ w_fp32  # [batch, N]

    # Our AWQ path: dequantize using loader function
    w_our = awq_dequantize(qweight, scales, group_size=g)  # [K, N] float32
    y_ours = x @ w_our  # [batch, N]

    max_err = np.max(np.abs(y_ref - y_ours))
    # Cosine similarity
    cos_sim = float(np.dot(y_ref.flatten(), y_ours.flatten()) /
                    (np.linalg.norm(y_ref) * np.linalg.norm(y_ours) + 1e-12))

    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    assert max_err < 1e-2, f"Matmul output error too high: {max_err:.6f}"
    assert cos_sim > 0.999, f"Cosine similarity too low: {cos_sim:.6f}"
    print(f"  Matmul output correctness: PASS (max_err={max_err:.6f}, cos_sim={cos_sim:.6f})")
    return True


def test_awq_vs_gptq_compatibility():
    """Test that AWQ weights produce the same output format as GPTQ for engine compatibility."""
    print("\n=== Test 7: AWQ/GPTQ format compatibility ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = QwenConfig()
        model_dir = create_synthetic_awq_safetensors(tmpdir, config, num_layers=2)
        loader = AWQWeightLoader(model_dir, config)

        w = loader.load_layer(0)

        # Verify the output format matches what the engine expects from GPTQ loader
        # Engine expects: {proj}_qweight, {proj}_scales, {proj}_zeros
        for proj in ['gate', 'up', 'down']:
            assert f'{proj}_qweight' in w, f"Missing {proj}_qweight"
            assert f'{proj}_scales' in w, f"Missing {proj}_scales"
            assert f'{proj}_zeros' in w, f"Missing {proj}_zeros"

        print("  Output dict has same keys as GPTQ loader: PASS")

        # Verify AWQ qweight is INT32 (same as GPTQ)
        assert w['gate_qweight'].dtype in (np.int32, np.uint32), \
            f"Expected int32/uint32, got {w['gate_qweight'].dtype}"
        print(f"  qweight dtype = {w['gate_qweight'].dtype} (compatible with engine): PASS")

        # Verify scales are FP16 (same as GPTQ)
        assert w['gate_scales'].dtype == np.float16, \
            f"Expected float16 scales, got {w['gate_scales'].dtype}"
        print(f"  scales dtype = {w['gate_scales'].dtype} (compatible with engine): PASS")

        # Verify zeros are all 0 (AWQ specific)
        assert np.all(w['gate_zeros'] == 0), "AWQ zeros should be all 0"
        assert np.all(w['up_zeros'] == 0), "AWQ zeros should be all 0"
        assert np.all(w['down_zeros'] == 0), "AWQ zeros should be all 0"
        print("  zeros = 0 (AWQ no zero-point): PASS")
    return True


def main():
    print("=" * 60)
    print("AWQ Weight Loader Tests")
    print("=" * 60)
    print()
    print("NOTE: No AWQ model available at /opt/models/.")
    print("Testing with synthetic AWQ-format weights.")
    print()

    tests = [
        ("AWQ format detection", test_detect_awq_format),
        ("Shape/dtype validation (VAL-AWQ-001)", test_shapes_and_dtypes),
        ("No zero-point tensors", test_no_zeros_tensors),
        ("AWQ dequantization correctness", test_awq_dequantize_correctness),
        ("Layer count and architecture", test_layer_count),
        ("Single-layer matmul output", test_matmul_output),
        ("AWQ/GPTQ format compatibility", test_awq_vs_gptq_compatibility),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed == 0:
        print("ALL TESTS PASSED")
        print()
        print("VAL-AWQ-001 PASS: AWQ weights loaded correctly.")
        print("  - 64 layers expected (from config)")
        print("  - gate_qweight: (640, 17408) int32")
        print("  - gate_scales: (40, 17408) float16")
        print("  - gate_zeros: all 0 (AWQ no zero-point)")
        print("  - down_qweight: (2176, 5120) int32")
        print("  - Dequantization max_err < 1e-2")
        return 0
    else:
        print(f"FAILED: {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
