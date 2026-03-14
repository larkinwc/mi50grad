#!/usr/bin/env python3
"""Test loading and running real Qwen 3.5 27B GPTQ weights through the engine.

Loads first 4 layers (3 linear + 1 full attention) and runs a decode step
to verify the complete pipeline works with real weights.
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import QwenConfig, load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def test_real_4_layers():
    """Load first 4 layers from real model and run a decode step."""
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    num_test_layers = 4

    print(f"Loading config from {model_dir}...")
    config = load_config_from_json(model_dir)

    # Override to only use first N layers for testing
    config.num_hidden_layers = num_test_layers
    config.layer_types = config.layer_types[:num_test_layers]
    print(f"  Testing with {num_test_layers} layers: {config.layer_types}")

    print("Creating engine...")
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)

    print("Creating weight loader...")
    loader = QwenWeightLoader(model_dir, config)

    try:
        # Load layers
        for i in range(num_test_layers):
            t0 = time.time()
            weights = loader.load_layer(i)
            t_load = time.time() - t0

            t0 = time.time()
            engine.load_layer_weights(i, weights)
            t_upload = time.time() - t0

            lt = weights['layer_type']
            print(f"  Layer {i} ({lt}): load={t_load:.2f}s, upload={t_upload:.2f}s")

        # Load final norm
        final_norm = loader.load_final_norm()
        engine.load_final_norm(final_norm)
        print("  Final norm loaded")

        # Load embedding for token lookup
        print("Loading embedding...")
        t0 = time.time()
        embed = loader.load_embedding()
        print(f"  Embedding loaded in {time.time()-t0:.2f}s: {embed.shape}")

        # Use a real token embedding (e.g., token 0)
        token_id = 100  # arbitrary token
        embedding = embed[token_id].copy()
        print(f"  Token {token_id} embedding: range=[{embedding.min():.4f}, {embedding.max():.4f}]")

        # Run decode step
        print("\nRunning decode step...")
        t0 = time.time()
        hidden = engine.decode_step(embedding, position=0)
        t_decode = time.time() - t0

        print(f"  Decode time: {t_decode*1000:.1f}ms")
        print(f"  Output shape: {hidden.shape}, dtype: {hidden.dtype}")
        print(f"  Output range: [{hidden.min():.4f}, {hidden.max():.4f}]")
        print(f"  Output mean: {hidden.astype(np.float32).mean():.4f}")
        print(f"  Output std: {hidden.astype(np.float32).std():.4f}")

        assert hidden.shape == (config.hidden_size,)
        assert hidden.dtype == np.float16
        assert not np.any(np.isnan(hidden)), "Output contains NaN!"
        assert not np.any(np.isinf(hidden)), "Output contains Inf!"

        # Run another decode step
        print("\nRunning second decode step...")
        embedding2 = embed[200].copy()
        t0 = time.time()
        hidden2 = engine.decode_step(embedding2, position=1)
        t_decode2 = time.time() - t0

        print(f"  Decode time: {t_decode2*1000:.1f}ms")
        print(f"  Output range: [{hidden2.min():.4f}, {hidden2.max():.4f}]")
        assert not np.any(np.isnan(hidden2)), "Output 2 contains NaN!"

        # Test LM head projection (CPU)
        print("\nTesting LM head projection...")
        t0 = time.time()
        lm_head = loader.load_lm_head()
        t_lm = time.time() - t0
        print(f"  LM head loaded in {t_lm:.2f}s")

        t0 = time.time()
        logits = lm_head.astype(np.float32) @ hidden.astype(np.float32)
        t_proj = time.time() - t0
        print(f"  Logits shape: {logits.shape}, projection time: {t_proj*1000:.1f}ms")
        print(f"  Top-5 tokens: {np.argsort(logits)[-5:][::-1]}")
        print(f"  Max logit: {logits.max():.4f}")

        print("\nPASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.cleanup()


if __name__ == "__main__":
    success = test_real_4_layers()
    sys.exit(0 if success else 1)
