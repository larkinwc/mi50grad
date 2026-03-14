"""
GPTQ safetensors weight loader for Qwen 3.5 27B INT4.

Loads quantized weights from HuggingFace GPTQ safetensors format,
repacks them into our kernel's expected layout, and uploads to GPU.

Qwen 3.5 weight structure:
  - Full attention layers: self_attn.{q,k,v,o}_proj.weight (BF16)
  - Linear attention layers: linear_attn.{in_proj_qkv, in_proj_a/b/z, conv1d, ...} (BF16)
  - MLP: mlp.{gate,up,down}_proj.{qweight,scales,qzeros} (INT4 GPTQ)
  - All norm weights: BF16
  - Embeddings/lm_head: BF16

Our kernels expect FP16, so BF16 is converted at load time.

W8A8 packed format (for gemv_w8a8):
  - W_int8:   [N, K] signed INT8 (row-major)
  - scale_w:  [N] FP32 per-channel weight scales

W4A8 packed format (for gemv_w4a8_grouped):
  - W_packed:      [N, K/8] uint32 (nibble-packed signed INT4, row-major)
  - scale_w_grp:   [K/group_size, N] FP16 per-group scales
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.model.qwen import QwenConfig


def bf16_to_fp16(data: bytes, shape: tuple) -> np.ndarray:
    """Convert BF16 raw bytes to FP16 numpy array.

    BF16 and FP16 are both 16-bit but with different exponent/mantissa splits.
    We convert via FP32 intermediate.
    """
    # BF16: 1 sign + 8 exponent + 7 mantissa
    # Read as uint16, pad to uint32 (shift left 16), reinterpret as float32
    bf16_vals = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    # Pad mantissa with zeros to get float32
    fp32_vals = bf16_vals.astype(np.uint32) << 16
    fp32_array = fp32_vals.view(np.float32)
    return fp32_array.astype(np.float16)


def load_safetensors_metadata(path: str) -> Tuple[dict, int]:
    """Load safetensors header metadata without reading tensor data."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size)
        metadata = json.loads(header_json)
    return metadata, 8 + header_size


def load_safetensors_tensor(path: str, key: str, metadata: dict,
                            data_offset: int) -> np.ndarray:
    """Load a specific tensor from a safetensors file.

    Handles BF16 by converting to FP16.
    """
    if key not in metadata:
        raise KeyError(f"Tensor '{key}' not found in safetensors")

    info = metadata[key]
    dtype_str = info['dtype']
    shape = tuple(info['shape'])
    offsets = info['data_offsets']
    start, end = offsets

    with open(path, 'rb') as f:
        f.seek(data_offset + start)
        data = f.read(end - start)

    if dtype_str == 'BF16':
        return bf16_to_fp16(data, shape)

    dtype_map = {
        'F16': np.float16,
        'F32': np.float32,
        'I32': np.int32,
        'I64': np.int64,
        'U8': np.uint8,
        'I8': np.int8,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return np.frombuffer(data, dtype=dtype).reshape(shape)


def unpack_gptq_qzeros(qzeros_packed: np.ndarray, bits: int = 4,
                        sym: bool = True) -> np.ndarray:
    """Unpack GPTQ qzeros from packed uint32 to individual values.

    qzeros shape: [num_groups, N_packed] where N_packed = N / (32/bits)
    Returns: [num_groups, N] as FP16 zeros

    For symmetric quantization (sym=True), zero point is the raw value (8 for 4-bit).
    For asymmetric (sym=False), stored values need +1 offset.
    """
    vals_per_u32 = 32 // bits
    num_groups, n_packed = qzeros_packed.shape
    N = n_packed * vals_per_u32

    zeros = np.zeros((num_groups, N), dtype=np.float16)
    mask = (1 << bits) - 1

    for i in range(vals_per_u32):
        vals = (qzeros_packed >> (i * bits)) & mask
        if sym:
            zeros[:, i::vals_per_u32] = vals.astype(np.float16)
        else:
            zeros[:, i::vals_per_u32] = vals.astype(np.float16) + 1

    return zeros


def gptq_to_w8a8(qweight: np.ndarray, scales: np.ndarray,
                  qzeros: Optional[np.ndarray] = None,
                  group_size: int = 128,
                  sym: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convert GPTQ INT4 weights to W8A8 INT8 format.

    This dequantizes the INT4 GPTQ weights to FP32 (simulating the original
    FP16 weights), then requantizes to INT8 with per-channel scaling.

    The resulting INT8 weights are suitable for the gemv_w8a8 kernel:
      y[n] = scale_w[n] * scale_a * dot_int32(W_int8[n], x_int8)

    Args:
        qweight:    [K/8, N] uint32 GPTQ packed INT4 weights
        scales:     [K/group_size, N] FP16 per-group scales
        qzeros:     [K/group_size, N/8] uint32 GPTQ zeros (None = symmetric, zp=8)
        group_size: GPTQ group size
        sym:        True for symmetric quantization (default)

    Returns:
        W_int8:    [N, K] signed INT8 (row-major)
        scale_w:   [N] FP32 per-channel weight scales for W8A8
    """
    K_packed, N = qweight.shape
    K = K_packed * 8

    # Unpack GPTQ weights: [K, N] uint8 (0..15)
    W_raw = np.zeros((K, N), dtype=np.uint8)
    for b in range(8):
        W_raw[b::8, :] = (qweight >> (b * 4)) & 0xF

    # Get zero points
    if qzeros is not None:
        num_groups, n_packed_z = qzeros.shape
        zeros = np.zeros((num_groups, N), dtype=np.uint8)
        vals_per_u32 = 8  # 32 / 4 bits
        for b in range(vals_per_u32):
            zeros[:, b::vals_per_u32] = (qzeros >> (b * 4)) & 0xF
    else:
        num_groups = K // group_size
        zeros = np.full((num_groups, N), 8, dtype=np.uint8)

    # Expand zeros and scales to [K, N]
    num_groups = K // group_size
    zeros_expanded = np.repeat(zeros, group_size, axis=0)  # [K, N]
    scales_expanded = np.repeat(scales.astype(np.float32), group_size, axis=0)  # [K, N]

    # Dequantize to FP32: w_fp32 = (w_raw - zero) * scale
    W_signed = W_raw.astype(np.int16) - zeros_expanded.astype(np.int16)  # [K, N]
    W_fp32 = W_signed.astype(np.float32) * scales_expanded  # [K, N]

    # Transpose to N-major: [N, K]
    W_fp32_T = W_fp32.T  # [N, K]

    # Per-channel INT8 requantization: scale = max_abs / 127.0
    max_abs = np.abs(W_fp32_T).max(axis=1) + 1e-12  # [N]
    scale_w = (max_abs / 127.0).astype(np.float32)  # [N]

    # Quantize to INT8: round(w / scale), clamp to [-128, 127]
    W_int8 = np.clip(
        np.round(W_fp32_T / scale_w[:, np.newaxis]),
        -128, 127
    ).astype(np.int8)  # [N, K]

    return W_int8, scale_w


def gptq_to_w4a8(qweight: np.ndarray, scales: np.ndarray,
                  qzeros: Optional[np.ndarray] = None,
                  group_size: int = 128,
                  sym: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convert GPTQ INT4 weights to W4A8 packed format.

    Repacks GPTQ weights into the N-major nibble format expected by
    gemv_w4a8_grouped. The per-group scales are preserved as-is.

    Args:
        qweight:    [K/8, N] uint32 GPTQ packed INT4 weights
        scales:     [K/group_size, N] FP16 per-group scales
        qzeros:     [K/group_size, N/8] uint32 GPTQ zeros (None = symmetric, zp=8)
        group_size: GPTQ group size
        sym:        True for symmetric quantization

    Returns:
        W_packed:    [N, K/8] uint32 (N-major, signed nibbles, zero-subtracted)
        scale_w_grp: [K/group_size, N] FP16 per-group scales
    """
    K_packed, N = qweight.shape
    K = K_packed * 8
    num_groups = K // group_size

    # Unpack GPTQ weights: [K, N] uint8 (0..15)
    W_raw = np.zeros((K, N), dtype=np.uint8)
    for b in range(8):
        W_raw[b::8, :] = (qweight >> (b * 4)) & 0xF

    # Get zero points
    if qzeros is not None:
        num_groups_z, n_packed_z = qzeros.shape
        zeros = np.zeros((num_groups_z, N), dtype=np.uint8)
        for b in range(8):  # 8 nibbles per uint32
            zeros[:, b::8] = (qzeros >> (b * 4)) & 0xF
    else:
        zeros = np.full((num_groups, N), 8, dtype=np.uint8)

    # Expand zeros to [K, N]
    zeros_expanded = np.repeat(zeros, group_size, axis=0)  # [K, N]

    # Subtract zero point: w_signed = w_raw - zp, ∈ [-8, 7]
    W_signed = np.clip(
        W_raw.astype(np.int16) - zeros_expanded.astype(np.int16),
        -8, 7
    ).astype(np.int8)  # [K, N]

    # Transpose to N-major: [N, K]
    W_Nmajor = W_signed.T  # [N, K]

    # Pack [N, K] int8 → [N, K/8] uint32 (nibble-packed, signed)
    K_groups = K // 8
    W_packed = np.zeros((N, K_groups), dtype=np.uint32)
    for b in range(8):
        nibbles = (W_Nmajor[:, b::8].astype(np.int32)) & 0xF  # [N, K/8]
        W_packed |= (nibbles.astype(np.uint32) << (b * 4))

    # Per-group scales: keep as [K/group_size, N] FP16
    scale_w_grp = scales.astype(np.float16)  # [K/group_size, N]

    return W_packed, scale_w_grp


class GPTQWeightLoader:
    """Loads GPTQ-quantized model weights from safetensors files."""

    def __init__(self, model_dir: str, bits: int = 4, group_size: int = 128,
                 sym: bool = True):
        self.model_dir = Path(model_dir)
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self._file_cache = {}
        self._index = None

    def _load_index(self):
        if self._index is not None:
            return
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self._index = json.load(f).get("weight_map", {})
        else:
            self._index = {}

    def _get_file_info(self, path: str) -> Tuple[dict, int]:
        if path not in self._file_cache:
            metadata, offset = load_safetensors_metadata(path)
            self._file_cache[path] = (metadata, offset)
        return self._file_cache[path]

    def find_tensor_file(self, key: str) -> Optional[str]:
        """Find which safetensors file contains a given tensor key."""
        self._load_index()
        if key in self._index:
            return str(self.model_dir / self._index[key])

        for sf in sorted(self.model_dir.glob("*.safetensors")):
            metadata, _ = self._get_file_info(str(sf))
            if key in metadata and key != "__metadata__":
                return str(sf)
        return None

    def load_tensor(self, key: str) -> np.ndarray:
        """Load a tensor by key, handling BF16→FP16 conversion."""
        file_path = self.find_tensor_file(key)
        if file_path is None:
            raise FileNotFoundError(f"Cannot find tensor: {key}")
        metadata, offset = self._get_file_info(file_path)
        return load_safetensors_tensor(file_path, key, metadata, offset)

    def load_linear_weights(self, layer_prefix: str) -> Dict[str, np.ndarray]:
        """Load qweight, scales, and zeros for a GPTQ linear layer."""
        keys = {
            'qweight': f"{layer_prefix}.qweight",
            'scales': f"{layer_prefix}.scales",
            'qzeros': f"{layer_prefix}.qzeros",
        }

        result = {}
        for name, key in keys.items():
            result[name] = self.load_tensor(key)

        result['zeros'] = unpack_gptq_qzeros(result['qzeros'], self.bits,
                                               sym=self.sym)
        del result['qzeros']

        return result

    def load_linear_weights_w8a8(self, layer_prefix: str) -> Dict[str, np.ndarray]:
        """Load GPTQ INT4 weights and convert to W8A8 INT8 format.

        Returns dict with:
          'w8a8':     [N, K] INT8 weights
          'scale_w':  [N] FP32 per-channel weight scales
        """
        qweight = self.load_tensor(f"{layer_prefix}.qweight")
        scales = self.load_tensor(f"{layer_prefix}.scales")
        try:
            qzeros_raw = self.load_tensor(f"{layer_prefix}.qzeros")
        except (KeyError, FileNotFoundError):
            qzeros_raw = None

        W_int8, scale_w = gptq_to_w8a8(
            qweight, scales, qzeros_raw, self.group_size, self.sym)
        return {'w8a8': W_int8, 'scale_w': scale_w}

    def load_linear_weights_w4a8(self, layer_prefix: str) -> Dict[str, np.ndarray]:
        """Load GPTQ INT4 weights and repack to W4A8 format.

        Returns dict with:
          'w4a8':       [N, K/8] uint32 packed INT4 (N-major, signed)
          'scale_grp':  [K/group_size, N] FP16 per-group scales
        """
        qweight = self.load_tensor(f"{layer_prefix}.qweight")
        scales = self.load_tensor(f"{layer_prefix}.scales")
        try:
            qzeros_raw = self.load_tensor(f"{layer_prefix}.qzeros")
        except (KeyError, FileNotFoundError):
            qzeros_raw = None

        W_packed, scale_w_grp = gptq_to_w4a8(
            qweight, scales, qzeros_raw, self.group_size, self.sym)
        return {'w4a8': W_packed, 'scale_grp': scale_w_grp}

    def load_linear_weights_sharded(self, layer_prefix: str,
                                     shard_dim: int,
                                     num_shards: int,
                                     shard_id: int) -> Dict[str, np.ndarray]:
        """Load a shard of quantized linear weights."""
        full = self.load_linear_weights(layer_prefix)
        qweight = full['qweight']
        scales = full['scales']
        zeros = full['zeros']

        if shard_dim == 1:
            N = qweight.shape[1]
            shard_n = N // num_shards
            start = shard_id * shard_n
            end = start + shard_n
            return {
                'qweight': qweight[:, start:end].copy(),
                'scales': scales[:, start:end].copy(),
                'zeros': zeros[:, start:end].copy(),
            }
        elif shard_dim == 0:
            K8 = qweight.shape[0]
            Kg = scales.shape[0]
            qw_shard = K8 // num_shards
            sg_shard = Kg // num_shards
            qw_start = shard_id * qw_shard
            sg_start = shard_id * sg_shard
            return {
                'qweight': qweight[qw_start:qw_start + qw_shard, :].copy(),
                'scales': scales[sg_start:sg_start + sg_shard, :].copy(),
                'zeros': zeros[sg_start:sg_start + sg_shard, :].copy(),
            }
        else:
            raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}")


class QwenWeightLoader:
    """Qwen 3.5-specific weight loading with HF→mi50grad layer name mapping.

    Qwen 3.5 uses 'model.language_model.layers.{i}' prefix.
    Full attention layers have self_attn.{q,k,v,o}_proj.weight (BF16).
    Linear attention layers have linear_attn.{in_proj_qkv,...} (BF16).
    MLP is INT4 GPTQ for all layers.

    quant_format controls FFN weight format:
      'w4a16' (default): raw GPTQ INT4 (qweight, scales, zeros)
      'w8a8':            INT8 repacked weights + FP32 per-channel scales
      'w4a8':            Nibble-packed INT4 (N-major) + FP16 per-group scales
    """

    LAYER_PREFIX = "model.language_model.layers.{layer_idx}"

    def __init__(self, model_dir: str, config: QwenConfig,
                 bits: int = 4, group_size: int = 128,
                 quant_format: str = 'w4a16'):
        self.gptq = GPTQWeightLoader(model_dir, bits=bits, group_size=group_size,
                                      sym=getattr(config, 'quant_sym', True))
        self.model_dir = Path(model_dir)
        self.config = config
        self.quant_format = quant_format

    def load_layer(self, layer_idx: int,
                   tp_size: int = 1, tp_rank: int = 0) -> dict:
        """Load all weights for one transformer layer.

        Returns dict with keys matching LayerWeights attributes.
        Dispatches to full_attention or linear_attention loader based on config.
        """
        prefix = self.LAYER_PREFIX.format(layer_idx=layer_idx)
        weights = {}

        if self.config.is_full_attention(layer_idx):
            self._load_full_attn_weights(prefix, weights, tp_size, tp_rank)
        else:
            self._load_linear_attn_weights(prefix, weights)

        # MLP weights — format depends on quant_format
        self._load_mlp_weights(prefix, weights, tp_size, tp_rank)

        # Layer norms (BF16 → FP16)
        # Qwen 3.5 uses (1 + weight) * (x / rms) normalization
        attn_norm = self.gptq.load_tensor(
            f"{prefix}.input_layernorm.weight")
        ffn_norm = self.gptq.load_tensor(
            f"{prefix}.post_attention_layernorm.weight")
        weights['attn_norm'] = (1.0 + attn_norm.astype(np.float32)).astype(np.float16)
        weights['ffn_norm'] = (1.0 + ffn_norm.astype(np.float32)).astype(np.float16)

        return weights

    def _load_full_attn_weights(self, prefix: str, weights: dict,
                                 tp_size: int, tp_rank: int):
        """Load full attention weights (FP16, not quantized).

        q_proj: [12288, 5120] = [Q(6144) + gate(6144), hidden]
        k_proj: [1024, 5120]
        v_proj: [1024, 5120]
        o_proj: [5120, 6144]
        q_norm: [256]
        k_norm: [256]
        """
        weights['layer_type'] = 'full_attention'

        # Load attention projection weights (FP16)
        q_weight = self.gptq.load_tensor(f"{prefix}.self_attn.q_proj.weight")
        k_weight = self.gptq.load_tensor(f"{prefix}.self_attn.k_proj.weight")
        v_weight = self.gptq.load_tensor(f"{prefix}.self_attn.v_proj.weight")
        o_weight = self.gptq.load_tensor(f"{prefix}.self_attn.o_proj.weight")

        # q_proj is packed: [num_heads, head_dim*2, hidden] = [12288, 5120]
        # HF splits as view(num_heads, head_dim*2).chunk(2, dim=-1)
        # So layout is interleaved: [Q_h0(256), gate_h0(256), Q_h1(256), gate_h1(256), ...]
        num_heads = self.config.num_attention_heads  # 24
        head_dim = self.config.head_dim  # 256
        hidden = self.config.hidden_size  # 5120
        q_reshaped = q_weight.reshape(num_heads, head_dim * 2, hidden)
        weights['q_weight'] = q_reshaped[:, :head_dim, :].reshape(num_heads * head_dim, hidden).copy()
        weights['q_gate_weight'] = q_reshaped[:, head_dim:, :].reshape(num_heads * head_dim, hidden).copy()
        weights['k_weight'] = k_weight  # [1024, 5120]
        weights['v_weight'] = v_weight  # [1024, 5120]
        weights['o_weight'] = o_weight  # [5120, 6144]

        # Q/K norms — also use (1+weight) Gemma-style
        q_norm = self.gptq.load_tensor(f"{prefix}.self_attn.q_norm.weight")
        k_norm = self.gptq.load_tensor(f"{prefix}.self_attn.k_norm.weight")
        weights['q_norm'] = (1.0 + q_norm.astype(np.float32)).astype(np.float16)
        weights['k_norm'] = (1.0 + k_norm.astype(np.float32)).astype(np.float16)

    def _load_linear_attn_weights(self, prefix: str, weights: dict):
        """Load linear (Mamba-style) attention weights (all FP16).

        in_proj_qkv: [10240, 5120] = [Q(2048) + K(2048) + V(6144), hidden]
        in_proj_a: [48, 5120]
        in_proj_b: [48, 5120]
        in_proj_z: [6144, 5120]  (gate)
        conv1d: [10240, 1, 4]
        A_log: [48] (F32)
        dt_bias: [48]
        norm: [128] (F32, group norm on value_head_dim)
        out_proj: [5120, 6144]
        """
        weights['layer_type'] = 'linear_attention'

        la = f"{prefix}.linear_attn"

        weights['la_in_proj_qkv'] = self.gptq.load_tensor(f"{la}.in_proj_qkv.weight")
        weights['la_in_proj_a'] = self.gptq.load_tensor(f"{la}.in_proj_a.weight")
        weights['la_in_proj_b'] = self.gptq.load_tensor(f"{la}.in_proj_b.weight")
        weights['la_in_proj_z'] = self.gptq.load_tensor(f"{la}.in_proj_z.weight")
        weights['la_conv1d'] = self.gptq.load_tensor(f"{la}.conv1d.weight")
        weights['la_A_log'] = self.gptq.load_tensor(f"{la}.A_log")
        weights['la_dt_bias'] = self.gptq.load_tensor(f"{la}.dt_bias")
        weights['la_norm'] = self.gptq.load_tensor(f"{la}.norm.weight")
        weights['la_out_proj'] = self.gptq.load_tensor(f"{la}.out_proj.weight")

    def _load_mlp_weights(self, prefix: str, weights: dict,
                           tp_size: int, tp_rank: int):
        """Load MLP weights in the configured quantization format.

        w4a16 (default): raw GPTQ INT4 (qweight, scales, zeros) for existing INT4 kernels
        w8a8:            INT8 repacked weights + FP32 per-channel scales
        w4a8:            nibble-packed INT4 (N-major) + FP16 per-group scales
        """
        for proj, shard_dim in [('gate_proj', 1), ('up_proj', 1), ('down_proj', 0)]:
            short = proj.replace('_proj', '')
            mlp_prefix = f"{prefix}.mlp.{proj}"

            if self.quant_format == 'w4a16':
                if tp_size > 1:
                    w = self.gptq.load_linear_weights_sharded(
                        mlp_prefix, shard_dim, tp_size, tp_rank)
                else:
                    w = self.gptq.load_linear_weights(mlp_prefix)
                weights[f'{short}_qweight'] = w['qweight']
                weights[f'{short}_scales'] = w['scales']
                weights[f'{short}_zeros'] = w['zeros']

            elif self.quant_format == 'w8a8':
                w = self.gptq.load_linear_weights_w8a8(mlp_prefix)
                weights[f'{short}_w8a8'] = w['w8a8']
                weights[f'{short}_scale_w8a8'] = w['scale_w']

            elif self.quant_format == 'w4a8':
                w = self.gptq.load_linear_weights_w4a8(mlp_prefix)
                weights[f'{short}_w4a8'] = w['w4a8']
                weights[f'{short}_scale_w4a8'] = w['scale_grp']

    def load_embedding(self) -> np.ndarray:
        """Load embedding weight [vocab_size, hidden_dim] FP16."""
        return self.gptq.load_tensor("model.language_model.embed_tokens.weight")

    def load_lm_head(self) -> np.ndarray:
        """Load LM head weight [vocab_size, hidden_dim] FP16.
        Falls back to tied embedding if lm_head.weight not found."""
        try:
            return self.gptq.load_tensor("lm_head.weight")
        except FileNotFoundError:
            return self.load_embedding()

    def load_final_norm(self) -> np.ndarray:
        """Load final RMSNorm weight [hidden_dim] FP16.

        Qwen 3.5 uses (1+weight)*(x/rms) normalization.
        """
        w = self.gptq.load_tensor("model.language_model.norm.weight")
        return (1.0 + w.astype(np.float32)).astype(np.float16)

