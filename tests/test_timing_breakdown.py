#!/usr/bin/env python3
"""Profile decode step timing breakdown by component."""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.qwen import load_config_from_json
from src.model.weight_loader import QwenWeightLoader
from src.inference.engine import InferenceEngine


def main():
    model_dir = "/opt/models/Qwen3.5-27B-GPTQ-Int4"
    config = load_config_from_json(model_dir)
    engine = InferenceEngine(config, device_id=0, max_seq_len=64)
    loader = QwenWeightLoader(model_dir, config)

    print("Loading layers...")
    for i in range(config.num_hidden_layers):
        weights = loader.load_layer(i)
        engine.load_layer_weights(i, weights)
    del weights

    final_norm = loader.load_final_norm()
    engine.load_final_norm(final_norm)
    embed = loader.load_embedding()
    lm_head = loader.load_lm_head()
    engine.load_lm_head(lm_head)
    print("Loaded")

    h = config.hidden_size
    cfg = config

    # Warm up with a few tokens
    for i in range(3):
        emb = embed[760].copy()
        engine.decode_step(emb, position=i)

    # Time individual components for one decode step
    emb = embed[760].copy()
    engine.device.upload(engine.d_hidden, emb.tobytes())

    # Profile one full decode step
    times = {
        'rmsnorm_attn': 0, 'rmsnorm_ffn': 0, 'rmsnorm_final': 0,
        'full_attn_proj': 0, 'full_attn_qknorm': 0, 'full_attn_rope': 0,
        'full_attn_kv_upload': 0, 'full_attn_decode': 0,
        'full_attn_gate': 0, 'full_attn_out_proj': 0,
        'linear_attn_gpu_proj': 0, 'linear_attn_download': 0,
        'linear_attn_conv': 0, 'linear_attn_deltanet': 0,
        'linear_attn_norm_gate': 0, 'linear_attn_upload_out': 0,
        'linear_attn_out_proj': 0,
        'ffn_int4': 0, 'ffn_silu': 0, 'residual': 0,
        'lm_head': 0,
    }

    n_full = 0
    n_linear = 0

    for layer_idx in range(cfg.num_hidden_layers):
        lw = engine.layers[layer_idx]

        # Pre-attn norm
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
        times['rmsnorm_attn'] += time.perf_counter() - t

        if lw.layer_type == 'full_attention':
            n_full += 1
            # Q, K, V, gate projections
            t = time.perf_counter()
            engine._launch_gemv_fp16(engine.d_q, engine.d_normed, lw.q_weight, h, engine.q_dim)
            engine._launch_gemv_fp16(engine.d_q_gate, engine.d_normed, lw.q_gate_weight, h, engine.q_dim)
            engine._launch_gemv_fp16(engine.d_k, engine.d_normed, lw.k_weight, h, engine.kv_dim)
            engine._launch_gemv_fp16(engine.d_v, engine.d_normed, lw.v_weight, h, engine.kv_dim)
            times['full_attn_proj'] += time.perf_counter() - t

            # Q/K norm
            t = time.perf_counter()
            engine._launch_qk_norm(engine.d_q, lw.q_norm, cfg.num_attention_heads, cfg.head_dim)
            engine._launch_qk_norm(engine.d_k, lw.k_norm, cfg.num_key_value_heads, cfg.head_dim)
            times['full_attn_qknorm'] += time.perf_counter() - t

            # RoPE
            t = time.perf_counter()
            engine._launch_rope(engine.d_q, 3, cfg.num_attention_heads, cfg.head_dim)
            engine._launch_rope(engine.d_k, 3, cfg.num_key_value_heads, cfg.head_dim)
            times['full_attn_rope'] += time.perf_counter() - t

            # KV cache upload
            t = time.perf_counter()
            kv_bytes = engine.kv_dim * 2
            k_data = engine.device.download(engine.d_k, kv_bytes)
            v_data = engine.device.download(engine.d_v, kv_bytes)
            engine.kv_cache.append_kv(layer_idx, k_data, v_data)
            times['full_attn_kv_upload'] += time.perf_counter() - t

            # Decode attention
            t = time.perf_counter()
            engine._launch_decode_attn_256(
                engine.d_attn_out, engine.d_q,
                engine.kv_cache.layer_k_ptr(layer_idx),
                engine.kv_cache.layer_v_ptr(layer_idx),
                engine.kv_cache.current_len + 1)
            times['full_attn_decode'] += time.perf_counter() - t

            # Sigmoid gate
            t = time.perf_counter()
            engine._apply_sigmoid_gate(engine.d_attn_out, engine.d_q_gate, engine.q_dim)
            times['full_attn_gate'] += time.perf_counter() - t

            # Output projection
            t = time.perf_counter()
            engine._launch_gemv_fp16(engine.d_proj_out, engine.d_attn_out, lw.o_weight,
                                      engine.q_dim, h)
            times['full_attn_out_proj'] += time.perf_counter() - t

            # Residual
            t = time.perf_counter()
            engine._launch_residual_add(engine.d_hidden, engine.d_proj_out, h)
            times['residual'] += time.perf_counter() - t

        else:
            n_linear += 1
            # GPU projections
            t = time.perf_counter()
            engine._launch_gemv_fp16(engine.d_la_qkv, engine.d_normed, lw.la_in_proj_qkv,
                                      h, engine.la_qkv_dim)
            engine._launch_gemv_fp16(engine.d_la_dt, engine.d_normed, lw.la_in_proj_a,
                                      h, engine.la_dt_dim)
            engine._launch_gemv_fp16(engine.d_la_b, engine.d_normed, lw.la_in_proj_b,
                                      h, engine.la_dt_dim)
            engine._launch_gemv_fp16(engine.d_la_z, engine.d_normed, lw.la_in_proj_z,
                                      h, engine.la_z_dim)
            times['linear_attn_gpu_proj'] += time.perf_counter() - t

            # Download to CPU
            t = time.perf_counter()
            qkv = np.frombuffer(engine.device.download(engine.d_la_qkv, engine.la_qkv_dim * 2),
                                 dtype=np.float16).copy().astype(np.float32)
            a_input = np.frombuffer(engine.device.download(engine.d_la_dt, engine.la_dt_dim * 2),
                                     dtype=np.float16).copy().astype(np.float32)
            b_input = np.frombuffer(engine.device.download(engine.d_la_b, engine.la_dt_dim * 2),
                                     dtype=np.float16).copy().astype(np.float32)
            z = np.frombuffer(engine.device.download(engine.d_la_z, engine.la_z_dim * 2),
                               dtype=np.float16).copy().astype(np.float32)
            times['linear_attn_download'] += time.perf_counter() - t

            # Conv1d
            slot = engine.deltanet_state.get_slot(layer_idx, cfg)
            t = time.perf_counter()
            conv_state = engine.deltanet_state.conv_states[slot]
            qkv_f16 = qkv.astype(np.float16)
            conv_input = np.concatenate([conv_state, qkv_f16[:, None]], axis=1)
            if conv_state.shape[1] > 0:
                conv_state[:, :-1] = conv_state[:, 1:]
                conv_state[:, -1] = qkv_f16
            conv_weight = lw.la_conv1d.astype(np.float32).squeeze(1)
            qkv_conv = np.sum(conv_input.astype(np.float32) * conv_weight, axis=1)
            qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv)))
            times['linear_attn_conv'] += time.perf_counter() - t

            # DeltaNet recurrence
            t = time.perf_counter()
            q_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            k_dim = q_dim
            v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            q = qkv_conv[:q_dim].reshape(cfg.linear_num_key_heads, cfg.linear_key_head_dim)
            k = qkv_conv[q_dim:q_dim + k_dim].reshape(cfg.linear_num_key_heads, cfg.linear_key_head_dim)
            v = qkv_conv[q_dim + k_dim:].reshape(cfg.linear_num_value_heads, cfg.linear_value_head_dim)

            def l2norm(x, eps=1e-6):
                norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
                return x / norm
            q = l2norm(q) / np.sqrt(cfg.linear_key_head_dim)
            k = l2norm(k)
            heads_per_q = cfg.linear_num_value_heads // cfg.linear_num_key_heads
            q = np.repeat(q, heads_per_q, axis=0)
            k = np.repeat(k, heads_per_q, axis=0)

            A = np.exp(lw.la_A_log.astype(np.float32))
            softplus_a = np.log1p(np.exp(a_input + lw.la_dt_bias.astype(np.float32)))
            g = -A * softplus_a
            decay = np.exp(np.clip(g, -20, 0))
            beta = 1.0 / (1.0 + np.exp(-b_input))

            state = engine.deltanet_state.states[slot]
            output = np.zeros((cfg.linear_num_value_heads, cfg.linear_value_head_dim), dtype=np.float32)
            for head in range(cfg.linear_num_value_heads):
                state[head] *= decay[head]
                kv_mem = state[head].T @ k[head]
                delta = (v[head] - kv_mem) * beta[head]
                state[head] += np.outer(k[head], delta)
                output[head] = state[head].T @ q[head]
            times['linear_attn_deltanet'] += time.perf_counter() - t

            # Gated RMSNorm + SiLU
            t = time.perf_counter()
            norm_weight = lw.la_norm.astype(np.float32)
            z_reshaped = z.reshape(cfg.linear_num_value_heads, cfg.linear_value_head_dim)
            for head in range(cfg.linear_num_value_heads):
                rms = np.sqrt(np.mean(output[head] ** 2) + 1e-6)
                output[head] = (output[head] / rms) * norm_weight
                z_silu = z_reshaped[head] / (1.0 + np.exp(-z_reshaped[head]))
                output[head] = output[head] * z_silu
            times['linear_attn_norm_gate'] += time.perf_counter() - t

            # Upload + output projection
            t = time.perf_counter()
            y_f16 = output.reshape(-1).astype(np.float16)
            engine.device.upload(engine.d_la_out, y_f16.tobytes())
            times['linear_attn_upload_out'] += time.perf_counter() - t

            t = time.perf_counter()
            engine._launch_gemv_fp16(engine.d_proj_out, engine.d_la_out, lw.la_out_proj,
                                      engine.la_z_dim, h)
            times['linear_attn_out_proj'] += time.perf_counter() - t

            # Residual
            t = time.perf_counter()
            engine._launch_residual_add(engine.d_hidden, engine.d_proj_out, h)
            times['residual'] += time.perf_counter() - t

        # FFN
        t = time.perf_counter()
        engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.ffn_norm, h)
        times['rmsnorm_ffn'] += time.perf_counter() - t

        t = time.perf_counter()
        engine._launch_gemv_int4(engine.d_ffn_gate, engine.d_normed,
                                  lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                  h, cfg.intermediate_size)
        engine._launch_gemv_int4(engine.d_ffn_up, engine.d_normed,
                                  lw.up_qweight, lw.up_scales, lw.up_zeros,
                                  h, cfg.intermediate_size)
        times['ffn_int4'] += time.perf_counter() - t

        t = time.perf_counter()
        engine._launch_silu_fused(engine.d_ffn_gate, engine.d_ffn_up,
                                   engine.d_ffn_gate, cfg.intermediate_size)
        times['ffn_silu'] += time.perf_counter() - t

        t = time.perf_counter()
        engine._launch_gemv_int4(engine.d_ffn_out, engine.d_ffn_gate,
                                  lw.down_qweight, lw.down_scales, lw.down_zeros,
                                  cfg.intermediate_size, h)
        times['ffn_int4'] += time.perf_counter() - t

        t = time.perf_counter()
        engine._launch_residual_add(engine.d_hidden, engine.d_ffn_out, h)
        times['residual'] += time.perf_counter() - t

    engine.kv_cache.advance()

    # Final norm
    t = time.perf_counter()
    engine._launch_rmsnorm(engine.d_hidden2, engine.d_hidden, engine.d_final_norm, h)
    times['rmsnorm_final'] = time.perf_counter() - t

    # LM head
    t = time.perf_counter()
    logits = engine.compute_logits()
    times['lm_head'] = time.perf_counter() - t

    total = sum(times.values())
    print(f"\n{'Component':<30} {'Time (ms)':>10} {'%':>6} {'Per-layer':>12}")
    print("-" * 65)
    for name, t_val in sorted(times.items(), key=lambda x: -x[1]):
        if t_val < 0.0001:
            continue
        n = n_full if 'full_attn' in name else n_linear if 'linear_attn' in name else cfg.num_hidden_layers
        per_layer = t_val / n * 1000 if n > 0 else 0
        print(f"  {name:<28} {t_val*1000:>8.1f}ms {t_val/total*100:>5.1f}% {per_layer:>8.3f}ms/L")

    print(f"\n  {'TOTAL':<28} {total*1000:>8.1f}ms")
    print(f"\n  Full attention layers: {n_full}")
    print(f"  Linear attention layers: {n_linear}")

    engine.cleanup()


if __name__ == "__main__":
    main()
