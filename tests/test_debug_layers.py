#!/usr/bin/env python3
"""Debug per-layer output ranges to find where NaN/inf starts."""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def main():
    config = load_config_from_json("/opt/models/Qwen3.5-27B-GPTQ-Int4")
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader("/opt/models/Qwen3.5-27B-GPTQ-Int4", config)

    h = config.hidden_size
    num_layers = 8  # Just test first 8

    for i in range(num_layers):
        w = loader.load_layer(i)
        engine.load_layer_weights(i, w)
        lt = w.get("layer_type", "?")
        print(f"Loaded layer {i} ({lt})")

    embed = loader.load_embedding()
    emb = embed[100].copy()
    print(f"\nInput embedding range: [{emb.min():.4f}, {emb.max():.4f}]")

    engine.device.upload(engine.d_hidden, emb.tobytes())

    for layer_idx in range(num_layers):
        lw = engine.layers[layer_idx]

        # Attention
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
        normed = np.frombuffer(engine.device.download(engine.d_normed, h*2),
                                dtype=np.float16)
        print(f"\nL{layer_idx} ({lw.layer_type})")
        print(f"  normed: [{normed.min():.4f}, {normed.max():.4f}]")

        if lw.layer_type == "full_attention":
            engine._decode_full_attention(layer_idx, lw, 0)
        else:
            engine._decode_linear_attention(layer_idx, lw, 0)

        hc = np.frombuffer(engine.device.download(engine.d_hidden, h*2),
                            dtype=np.float16)
        print(f"  after attn+res: [{hc.min():.4f}, {hc.max():.4f}]")

        # FFN
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, config.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_up, engine.d_normed,
                                  lw.up_qweight, lw.up_scales, lw.up_zeros,
                                  h, config.intermediate_size)
        engine._launch_silu_fused(engine.d_ffn_gate, engine.d_ffn_up,
                                   engine.d_ffn_gate, config.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_out, engine.d_ffn_gate,
                                  lw.down_qweight, lw.down_scales, lw.down_zeros,
                                  config.intermediate_size, h)

        ffn_out = np.frombuffer(engine.device.download(engine.d_ffn_out, h*2),
                                 dtype=np.float16)
        print(f"  FFN out: [{ffn_out.min():.4f}, {ffn_out.max():.4f}]")

        engine._launch_residual_add(engine.d_hidden, engine.d_ffn_out, h)
        hc = np.frombuffer(engine.device.download(engine.d_hidden, h*2),
                            dtype=np.float16)
        print(f"  after FFN+res: [{hc.min():.4f}, {hc.max():.4f}]")

    engine.kv_cache.advance()
    engine.cleanup()


if __name__ == "__main__":
    main()
