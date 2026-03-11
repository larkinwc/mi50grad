"""
Inference engine for Qwen 3.5 27B on MI50.

Ties together all assembly kernels (GEMM, attention, elementwise)
into a complete transformer forward pass.

Supports:
- Single-token decode (GEMV + decode attention)
- Multi-token prefill (GEMM + flash attention) [future]
- INT4 quantized weights (GPTQ format)
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco, Kernel
from src.model.qwen import QwenConfig


ASM_DIR = Path(__file__).parent.parent / "asm"
BUILD_DIR = Path(__file__).parent.parent.parent / "build" / "kernels"


class KernelCache:
    """Lazily builds and caches HSACO kernels."""

    def __init__(self, device: GPUDevice):
        self.device = device
        self._modules = {}
        self._funcs = {}

    def get(self, kernel_name: str, asm_name: str) -> int:
        """Get kernel function handle, building if needed."""
        if kernel_name not in self._funcs:
            asm_path = ASM_DIR / f"{asm_name}.s"
            hsaco_path = BUILD_DIR / f"{asm_name}.hsaco"
            BUILD_DIR.mkdir(parents=True, exist_ok=True)

            if not hsaco_path.exists() or asm_path.stat().st_mtime > hsaco_path.stat().st_mtime:
                build_hsaco(str(asm_path), str(hsaco_path))

            module = self.device.load_hsaco(str(hsaco_path))
            self._funcs[kernel_name] = self.device.get_kernel(module, kernel_name)

        return self._funcs[kernel_name]


class KVCache:
    """Contiguous KV cache for all layers."""

    def __init__(self, config: QwenConfig, max_seq_len: int, device: GPUDevice):
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        self.current_len = 0

        # K cache: [num_layers, max_seq, num_kv_heads, head_dim] in FP16
        # V cache: same
        self.k_size = (config.num_hidden_layers * max_seq_len *
                       config.num_key_value_heads * config.head_dim * 2)
        self.v_size = self.k_size

        self.d_k = device.malloc(self.k_size)
        self.d_v = device.malloc(self.v_size)

        # Zero out
        device.hip.memset(self.d_k, 0, self.k_size)
        device.hip.memset(self.d_v, 0, self.v_size)

    def layer_k_ptr(self, layer_idx: int) -> int:
        """Get device pointer for K cache of a specific layer."""
        offset = (layer_idx * self.max_seq_len *
                  self.config.num_key_value_heads * self.config.head_dim * 2)
        return self.d_k + offset

    def layer_v_ptr(self, layer_idx: int) -> int:
        offset = (layer_idx * self.max_seq_len *
                  self.config.num_key_value_heads * self.config.head_dim * 2)
        return self.d_v + offset

    def append_kv(self, layer_idx: int, k_data: bytes, v_data: bytes):
        """Append one position's K and V to the cache."""
        # k_data shape: [num_kv_heads, head_dim] FP16
        kv_size = self.config.num_key_value_heads * self.config.head_dim * 2

        k_offset = (layer_idx * self.max_seq_len *
                     self.config.num_key_value_heads * self.config.head_dim +
                     self.current_len *
                     self.config.num_key_value_heads * self.config.head_dim) * 2
        v_offset = k_offset

        self.device.hip.memcpy_h2d(self.d_k + k_offset, k_data, kv_size)
        self.device.hip.memcpy_h2d(self.d_v + v_offset, v_data, kv_size)

    def advance(self):
        """Mark that one more position has been written."""
        self.current_len += 1

    def cleanup(self):
        self.device.free(self.d_k)
        self.device.free(self.d_v)


class LayerWeights:
    """Device-side weights for one transformer layer (INT4 quantized)."""

    def __init__(self):
        # QKV projection (INT4)
        self.qkv_qweight = 0  # device ptr
        self.qkv_scales = 0
        self.qkv_zeros = 0

        # Output projection (INT4)
        self.out_qweight = 0
        self.out_scales = 0
        self.out_zeros = 0

        # FFN up + gate (INT4)
        self.up_qweight = 0
        self.up_scales = 0
        self.up_zeros = 0
        self.gate_qweight = 0
        self.gate_scales = 0
        self.gate_zeros = 0

        # FFN down (INT4)
        self.down_qweight = 0
        self.down_scales = 0
        self.down_zeros = 0

        # RMSNorm weights (FP16)
        self.attn_norm = 0  # [hidden_dim] FP16
        self.ffn_norm = 0   # [hidden_dim] FP16


class InferenceEngine:
    """Single-GPU inference engine for Qwen 3.5 27B (INT4)."""

    def __init__(self, config: QwenConfig, device_id: int = 0,
                 max_seq_len: int = 2048):
        self.config = config
        self.device = GPUDevice(device_id)
        self.kernels = KernelCache(self.device)
        self.kv_cache = KVCache(config, max_seq_len, self.device)
        self.layers: List[LayerWeights] = []

        # Precompute RoPE cos/sin tables
        self._init_rope_tables(max_seq_len)

        # Allocate scratch buffers
        self._alloc_scratch()

    def _init_rope_tables(self, max_seq_len: int):
        """Precompute RoPE cos/sin tables."""
        half_dim = self.config.head_dim // 2
        freqs = 1.0 / (self.config.rope_theta **
                        (np.arange(0, half_dim, dtype=np.float32) * 2.0 / self.config.head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        cos_tab = np.cos(angles).astype(np.float16)
        sin_tab = np.sin(angles).astype(np.float16)

        self.d_cos = self.device.malloc(cos_tab.nbytes)
        self.d_sin = self.device.malloc(sin_tab.nbytes)
        self.device.upload(self.d_cos, cos_tab.tobytes())
        self.device.upload(self.d_sin, sin_tab.tobytes())

    def _alloc_scratch(self):
        """Allocate reusable scratch buffers for intermediate activations."""
        h = self.config.hidden_size
        inter = self.config.intermediate_size

        # Scratch buffers (sized for single-token decode)
        self.d_hidden = self.device.malloc(h * 2)       # [hidden_dim] FP16
        self.d_hidden2 = self.device.malloc(h * 2)      # residual
        self.d_qkv = self.device.malloc((h + 1024) * 2) # QKV output (Q: 4096 + KV: 1024)
        self.d_attn_out = self.device.malloc(h * 2)     # attention output
        self.d_ffn_up = self.device.malloc(inter * 2)   # FFN up
        self.d_ffn_gate = self.device.malloc(inter * 2) # FFN gate
        self.d_ffn_out = self.device.malloc(h * 2)      # FFN down output

    def load_layer_weights(self, layer_idx: int, weights: dict):
        """Upload one layer's INT4 weights to GPU.

        weights dict should contain numpy arrays for:
        qkv.qweight, qkv.scales, qkv.zeros, etc.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(LayerWeights())

        lw = self.layers[layer_idx]

        def upload(data: np.ndarray) -> int:
            ptr = self.device.malloc(data.nbytes)
            self.device.upload(ptr, data.tobytes())
            return ptr

        for key, arr in weights.items():
            setattr(lw, key, upload(arr))

    def decode_step(self, token_embedding: np.ndarray, position: int) -> np.ndarray:
        """Run one decode step through all layers.

        Args:
            token_embedding: [hidden_dim] FP16 array
            position: current sequence position

        Returns:
            [hidden_dim] FP16 logits-ready hidden state
        """
        h = self.config.hidden_size
        cfg = self.config

        # Upload embedding to d_hidden
        self.device.upload(self.d_hidden, token_embedding.tobytes())

        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.layers[layer_idx]

            # --- Attention block ---
            # 1. RMSNorm
            self._launch_rmsnorm(self.d_hidden2, self.d_hidden, lw.attn_norm, h)

            # 2. QKV projection (INT4 GEMV)
            self._launch_gemv_int4(self.d_qkv, self.d_hidden2,
                                    lw.qkv_qweight, lw.qkv_scales, lw.qkv_zeros,
                                    h, h + 1024)  # Q=4096 + KV=1024

            # 3. RoPE on Q and K
            self._launch_rope(self.d_qkv, position, cfg.num_attention_heads, cfg.head_dim)
            # K is at offset Q_size in qkv buffer
            q_size = cfg.num_attention_heads * cfg.head_dim * 2
            self._launch_rope_kv(self.d_qkv + q_size, position,
                                  cfg.num_key_value_heads, cfg.head_dim)

            # 4. Update KV cache
            k_data = self.device.download(self.d_qkv + q_size,
                                           cfg.num_key_value_heads * cfg.head_dim * 2)
            kv_head_dim = cfg.num_key_value_heads * cfg.head_dim * 2
            v_offset = q_size + kv_head_dim
            v_data = self.device.download(self.d_qkv + v_offset, kv_head_dim)
            self.kv_cache.append_kv(layer_idx, k_data, v_data)

            # 5. Decode attention
            self._launch_decode_attn(
                self.d_attn_out, self.d_qkv,  # Q part
                self.kv_cache.layer_k_ptr(layer_idx),
                self.kv_cache.layer_v_ptr(layer_idx),
                self.kv_cache.current_len + 1,
                layer_idx
            )

            # 6. Output projection (INT4 GEMV)
            self._launch_gemv_int4(self.d_hidden2, self.d_attn_out,
                                    lw.out_qweight, lw.out_scales, lw.out_zeros,
                                    h, h)

            # 7. Residual add: d_hidden += d_hidden2
            self._launch_residual_add(self.d_hidden, self.d_hidden2, h)

            # --- FFN block ---
            # 8. RMSNorm
            self._launch_rmsnorm(self.d_hidden2, self.d_hidden, lw.ffn_norm, h)

            # 9. FFN up + gate (two INT4 GEMVs)
            self._launch_gemv_int4(self.d_ffn_up, self.d_hidden2,
                                    lw.up_qweight, lw.up_scales, lw.up_zeros,
                                    h, cfg.intermediate_size)
            self._launch_gemv_int4(self.d_ffn_gate, self.d_hidden2,
                                    lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                                    h, cfg.intermediate_size)

            # 10. SiLU(gate) * up
            self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                     self.d_ffn_gate, cfg.intermediate_size)

            # 11. FFN down (INT4 GEMV)
            self._launch_gemv_int4(self.d_ffn_out, self.d_ffn_gate,
                                    lw.down_qweight, lw.down_scales, lw.down_zeros,
                                    cfg.intermediate_size, h)

            # 12. Residual add
            self._launch_residual_add(self.d_hidden, self.d_ffn_out, h)

        self.kv_cache.advance()

        # Download final hidden state
        return np.frombuffer(self.device.download(self.d_hidden, h * 2), dtype=np.float16)

    def _launch_rmsnorm(self, dst, src, weight, dim):
        func = self.kernels.get("rmsnorm_fp16", "rmsnorm")
        eps_bits = struct.unpack('<I', struct.pack('<f', self.config.rms_norm_eps))[0]
        params = [
            ctypes.c_uint64(src),
            ctypes.c_uint64(weight),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(dim),
            ctypes.c_uint32(eps_bits),
        ]
        self.device.launch(func, (1, 1, 1), (256, 1, 1), params, shared_mem=1024)

    def _launch_gemv_int4(self, dst, src, qweight, scales, zeros, K, N):
        func = self.kernels.get("gemv_int4_fp16", "gemm_int4")
        params = [
            ctypes.c_uint64(src),
            ctypes.c_uint64(qweight),
            ctypes.c_uint64(scales),
            ctypes.c_uint64(zeros),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(self.config.group_size),
        ]
        self.device.launch(func, (N, 1, 1), (256, 1, 1), params, shared_mem=1024)

    def _launch_rope(self, x_ptr, position, num_heads, head_dim):
        func = self.kernels.get("rope_fp16", "rope")
        half_dim = head_dim // 2
        # cos/sin for this position
        cos_offset = position * half_dim * 2  # FP16 bytes
        sin_offset = cos_offset
        params = [
            ctypes.c_uint64(x_ptr),
            ctypes.c_uint64(self.d_cos + cos_offset),
            ctypes.c_uint64(self.d_sin + sin_offset),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(num_heads),
        ]
        self.device.launch(func, (1, num_heads, 1), (half_dim, 1, 1), params)

    def _launch_rope_kv(self, x_ptr, position, num_kv_heads, head_dim):
        self._launch_rope(x_ptr, position, num_kv_heads, head_dim)

    def _launch_decode_attn(self, out, q, k_cache, v_cache, seq_len, layer_idx):
        func = self.kernels.get("decode_attn_fp16", "decode_attn")
        cfg = self.config
        params = [
            ctypes.c_uint64(q),
            ctypes.c_uint64(k_cache),
            ctypes.c_uint64(v_cache),
            ctypes.c_uint64(out),
            ctypes.c_uint32(seq_len),
            ctypes.c_uint32(cfg.head_dim),
            ctypes.c_uint32(cfg.num_attention_heads),
            ctypes.c_uint32(cfg.num_key_value_heads),
        ]
        self.device.launch(func, (cfg.num_attention_heads, 1, 1), (256, 1, 1), params)

    def _launch_silu_fused(self, gate, up, out, n):
        func = self.kernels.get("silu_fp16", "silu")
        params = [
            ctypes.c_uint64(gate),
            ctypes.c_uint64(up),
            ctypes.c_uint64(out),
            ctypes.c_uint32(n),
        ]
        grid_x = (n + 255) // 256
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)

    def _launch_residual_add(self, dst, src, n):
        """dst[i] += src[i]. Simple elementwise add.
        For now, download, add on host, re-upload. TODO: write kernel."""
        dst_data = np.frombuffer(self.device.download(dst, n * 2), dtype=np.float16).copy()
        src_data = np.frombuffer(self.device.download(src, n * 2), dtype=np.float16)
        dst_data += src_data
        self.device.upload(dst, dst_data.tobytes())

    def cleanup(self):
        self.kv_cache.cleanup()
        self.device.cleanup()
