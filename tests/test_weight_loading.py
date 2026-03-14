#!/usr/bin/env python3
"""Test weight loading from real Qwen 3.5 27B GPTQ model."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig, load_config_from_json
from src.model.weight_loader import QwenWeightLoader


def test_weight_loading():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"

    print("Loading config...")
    config = load_config_from_json(model_dir)
    print(f"  hidden={config.hidden_size}, head_dim={config.head_dim}")
    print(f"  layers={config.num_hidden_layers}, full={config.num_full_attention_layers}, linear={config.num_linear_attention_layers}")

    print("Creating weight loader...")
    loader = QwenWeightLoader(model_dir, config)

    # Test layer 0 (linear attention)
    print("\nLoading layer 0 (linear attention)...")
    t0 = time.time()
    w0 = loader.load_layer(0)
    t1 = time.time()
    print(f"  Loaded in {t1-t0:.2f}s")
    print(f"  layer_type: {w0['layer_type']}")
    for k in sorted(w0.keys()):
        v = w0[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")
        elif k == 'layer_type':
            pass
        else:
            print(f"  {k}: {v}")

    # Validate linear attention shapes
    assert w0['la_in_proj_qkv'].shape == (10240, 5120), f"Bad shape: {w0['la_in_proj_qkv'].shape}"
    assert w0['la_in_proj_a'].shape == (48, 5120)
    assert w0['la_conv1d'].shape == (10240, 1, 4)
    assert w0['gate_qweight'].shape[1] == 17408

    # Test layer 3 (full attention)
    print("\nLoading layer 3 (full attention)...")
    t0 = time.time()
    w3 = loader.load_layer(3)
    t1 = time.time()
    print(f"  Loaded in {t1-t0:.2f}s")
    print(f"  layer_type: {w3['layer_type']}")
    for k in sorted(w3.keys()):
        v = w3[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")

    # Validate full attention shapes
    assert w3['q_weight'].shape == (6144, 5120), f"Bad q_weight shape: {w3['q_weight'].shape}"
    assert w3['q_gate_weight'].shape == (6144, 5120)
    assert w3['k_weight'].shape == (1024, 5120)
    assert w3['v_weight'].shape == (1024, 5120)
    assert w3['o_weight'].shape == (5120, 6144)
    assert w3['q_norm'].shape == (256,)
    assert w3['k_norm'].shape == (256,)

    # Test embedding
    print("\nLoading embedding...")
    t0 = time.time()
    embed = loader.load_embedding()
    t1 = time.time()
    print(f"  embed: {embed.shape} {embed.dtype}, loaded in {t1-t0:.2f}s")
    assert embed.shape == (248320, 5120)

    # Test final norm
    print("\nLoading final norm...")
    fn = loader.load_final_norm()
    print(f"  final_norm: {fn.shape} {fn.dtype}")
    assert fn.shape == (5120,)

    # Test lm_head
    print("\nLoading lm_head...")
    t0 = time.time()
    lm = loader.load_lm_head()
    t1 = time.time()
    print(f"  lm_head: {lm.shape} {lm.dtype}, loaded in {t1-t0:.2f}s")
    assert lm.shape == (248320, 5120)

    print("\nALL WEIGHT LOADING TESTS PASSED")
    return True


if __name__ == "__main__":
    try:
        success = test_weight_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
