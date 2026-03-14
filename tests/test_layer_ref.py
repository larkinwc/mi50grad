#!/usr/bin/env python3
"""Compare one FFN layer output: GPU kernel vs numpy dequant reference."""

import sys
import ctypes
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader, unpack_gptq_qzeros
from src.inference.engine import InferenceEngine


def dequant_matmul_ref(x_f32, qweight, scales, zeros, K, N, group_size=128):
    """CPU reference: dequantize INT4 and multiply."""
    out = np.zeros(N, dtype=np.float32)
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)

    K_packed = K // 8
    for col in range(N):
        acc = 0.0
        for k_row in range(K_packed):
            packed = int(qweight[k_row, col])
            for bit in range(8):
                k_idx = k_row * 8 + bit
                if k_idx >= K:
                    break
                int4_val = (packed >> (bit * 4)) & 0xF
                g = k_idx // group_size
                w = (int4_val - zeros_f32[g, col]) * scales_f32[g, col]
                acc += w * x_f32[k_idx]
        out[col] = acc
    return out


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    # Load just first 2 layers
    for i in range(2):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()

    # Run one token through layer 0
    emb = embed[760].copy()  # "The" token
    h = config.hidden_size

    engine.device.upload(engine.d_hidden, emb.tobytes())

    lw = engine.layers[0]
    # Run pre-attention RMSNorm
    engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)

    # Run linear attention
    engine._decode_linear_attention(0, lw, 0)

    # Now test FFN
    engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)

    # Get normed hidden for CPU reference
    normed = np.frombuffer(engine.device.download(engine.d_normed, h * 2),
                            dtype=np.float16).copy()
    normed_f32 = normed.astype(np.float32)

    # GPU gate_proj
    engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                              lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                              h, config.intermediate_size)
    gpu_gate = np.frombuffer(engine.device.download(engine.d_ffn_gate,
                              config.intermediate_size * 2),
                              dtype=np.float16).copy().astype(np.float32)

    # CPU reference gate_proj (just first 16 columns for speed)
    test_cols = 16
    prefix = "model.language_model.layers.0"
    qweight = loader.gptq.load_tensor(f"{prefix}.mlp.gate_proj.qweight")
    scales = loader.gptq.load_tensor(f"{prefix}.mlp.gate_proj.scales")
    qzeros = loader.gptq.load_tensor(f"{prefix}.mlp.gate_proj.qzeros")
    zeros = unpack_gptq_qzeros(qzeros.astype(np.int32), 4, sym=config.quant_sym)

    print(f"gate_proj: qweight={qweight.shape}, scales={scales.shape}, zeros={zeros.shape}")
    print(f"normed range: [{normed.min():.4f}, {normed.max():.4f}]")

    print(f"\nComparing gate_proj first {test_cols} columns:")
    for col in range(test_cols):
        # CPU reference for this column
        acc = 0.0
        for k_row in range(h // 8):
            packed = int(qweight[k_row, col])
            for bit in range(8):
                k_idx = k_row * 8 + bit
                int4_val = (packed >> (bit * 4)) & 0xF
                g = k_idx // config.group_size
                w = (int4_val - float(zeros[g, col])) * float(scales[g, col])
                acc += w * normed_f32[k_idx]
        print(f"  col {col}: ref={acc:.6f}, gpu={gpu_gate[col]:.6f}, "
              f"diff={abs(acc - gpu_gate[col]):.6f}")

    # Also compare up_proj
    print("\nComparing up_proj first 4 columns:")
    qweight_up = loader.gptq.load_tensor(f"{prefix}.mlp.up_proj.qweight")
    scales_up = loader.gptq.load_tensor(f"{prefix}.mlp.up_proj.scales")
    qzeros_up = loader.gptq.load_tensor(f"{prefix}.mlp.up_proj.qzeros")
    zeros_up = unpack_gptq_qzeros(qzeros_up.astype(np.int32), 4, sym=config.quant_sym)

    engine._launch_gemv_int4(engine.d_ffn_up, engine.d_normed,
                              lw.up_qweight, lw.up_scales, lw.up_zeros,
                              h, config.intermediate_size)
    gpu_up = np.frombuffer(engine.device.download(engine.d_ffn_up,
                            config.intermediate_size * 2),
                            dtype=np.float16).copy().astype(np.float32)

    for col in range(4):
        acc = 0.0
        for k_row in range(h // 8):
            packed = int(qweight_up[k_row, col])
            for bit in range(8):
                k_idx = k_row * 8 + bit
                int4_val = (packed >> (bit * 4)) & 0xF
                g = k_idx // config.group_size
                w = (int4_val - float(zeros_up[g, col])) * float(scales_up[g, col])
                acc += w * normed_f32[k_idx]
        print(f"  col {col}: ref={acc:.6f}, gpu={gpu_up[col]:.6f}, "
              f"diff={abs(acc - gpu_up[col]):.6f}")

    # Overall stats
    print(f"\nGPU gate_proj full output: range=[{gpu_gate.min():.4f}, {gpu_gate.max():.4f}], "
          f"mean={gpu_gate.mean():.6f}")
    print(f"GPU up_proj full output: range=[{gpu_up.min():.4f}, {gpu_up.max():.4f}], "
          f"mean={gpu_up.mean():.6f}")

    engine.cleanup()


if __name__ == "__main__":
    main()
