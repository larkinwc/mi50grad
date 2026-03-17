"""
AWQ safetensors weight loader for Qwen 3.5 27B INT4.

AWQ (Activation-aware Weight Quantization) format differences from GPTQ:
  - Weights packed as INT4 in INT32 (same packing: 8 nibbles per uint32)
  - Same layout: qweight [K/8, N] INT32, scales [K/group_size, N] FP16
  - NO zero-point tensor (AWQ uses zero_point=0, so dequant = q * scale)
  - Different tensor naming: "model.layers.N" (vs GPTQ "model.language_model.layers.N")
  - No qzeros key in safetensors

Dequantization comparison:
  GPTQ: w = (q - zero) * scale
  AWQ:  w = q * scale  (zero = 0, skip zero-point subtraction)

The AWQWeightLoader produces output in the SAME format as GPTQWeightLoader:
  {proj}_qweight: [K/8, N] INT32 (packed nibbles, compatible with existing GEMV kernels)
  {proj}_scales:  [K/group_size, N] FP16
  {proj}_zeros:   [K/group_size, N] FP16, all zeros (signals no zero-point to GEMV kernel)

This allows the existing engine to use AWQ weights with minimal changes.

NOTE: AWQ model not available at /opt/models/ on the dev server.
If an AWQ Qwen 3.5 27B model is added to /opt/models/, set model_dir
to that path. The loader handles both the standard HuggingFace AutoAWQ
naming ("model.layers.N") and the Qwen-specific variant.
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.model.qwen import QwenConfig
from src.model.weight_loader import load_safetensors_metadata, load_safetensors_tensor


def detect_awq_format(model_dir: str) -> str:
    """Detect quantization format from safetensors index.

    Returns:
        'awq'  - if no qzeros tensors found (AWQ format)
        'gptq' - if qzeros tensors found (GPTQ format)
        'fp16' - if no quantized weights found

    AWQ models have qweight and scales but NO qzeros.
    GPTQ models have qweight, scales, AND qzeros.
    """
    model_dir = Path(model_dir)

    # Try to find tensors from index or single file
    all_keys = set()

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        all_keys = set(index.get("weight_map", {}).keys())
    else:
        # Single file
        for sf in sorted(model_dir.glob("*.safetensors")):
            meta, _ = load_safetensors_metadata(str(sf))
            all_keys.update(k for k in meta.keys() if k != "__metadata__")

    if not all_keys:
        return 'fp16'

    has_qweight = any('qweight' in k for k in all_keys)
    has_qzeros = any('qzeros' in k for k in all_keys)

    if not has_qweight:
        return 'fp16'
    if has_qzeros:
        return 'gptq'
    return 'awq'


def awq_dequantize(qweight: np.ndarray, scales: np.ndarray,
                    group_size: int = 128) -> np.ndarray:
    """Dequantize AWQ packed INT4 weights to FP32.

    AWQ dequantization: w = q * scale (no zero-point subtraction)

    Args:
        qweight:    [K/8, N] INT32 packed nibbles (unsigned 0..15)
        scales:     [K/group_size, N] FP16 per-group scales
        group_size: quantization group size (default: 128)

    Returns:
        w_fp32: [K, N] FP32 dequantized weights

    The nibble values 0..15 are treated as unsigned integers.
    AWQ weights are stored with zero_point=0, so w = q * scale directly.
    """
    K_packed, N = qweight.shape
    K = K_packed * 8

    # Unpack nibbles: [K, N] uint8 (0..15)
    w_raw = np.zeros((K, N), dtype=np.uint8)
    mask = 0xF
    for b in range(8):
        w_raw[b::8, :] = (qweight >> (b * 4)) & mask

    # Expand scales from [K/group_size, N] to [K, N]
    scales_fp32 = scales.astype(np.float32)
    scales_expanded = np.repeat(scales_fp32, group_size, axis=0)  # [K, N]

    # AWQ dequant: w = q * scale (no zero subtraction)
    w_fp32 = w_raw.astype(np.float32) * scales_expanded  # [K, N]

    return w_fp32


class AWQWeightLoader:
    """Loads AWQ-quantized model weights from safetensors files.

    AWQ tensor naming convention (HuggingFace AutoAWQ):
      model.layers.N.mlp.gate_proj.qweight    [K/8, N] INT32
      model.layers.N.mlp.gate_proj.scales     [K/group_size, N] FP16
      model.layers.N.self_attn.q_proj.qweight [K/8, N] INT32  (if attn is quantized)
      (NO .qzeros tensors)

    Note: Qwen 3.5 27B AWQ typically quantizes only FFN layers (same as GPTQ).
    Attention projections remain FP16/BF16.

    The loader produces output compatible with the GPTQWeightLoader format:
      {proj}_qweight, {proj}_scales, {proj}_zeros (all zeros for AWQ)
    """

    # AWQ uses "model.layers.N" prefix (vs GPTQ "model.language_model.layers.N")
    LAYER_PREFIX = "model.layers.{layer_idx}"

    # Some Qwen AWQ models use the language_model prefix; we try both
    ALT_LAYER_PREFIX = "model.language_model.layers.{layer_idx}"

    def __init__(self, model_dir: str, config: QwenConfig,
                 group_size: int = 128):
        self.model_dir = Path(model_dir)
        self.config = config
        self.group_size = group_size
        self.num_layers = config.num_hidden_layers
        self._file_cache = {}
        self._index = None
        self._layer_prefix = None  # Detected at first use

    def _load_index(self):
        """Load the safetensors index."""
        if self._index is not None:
            return
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self._index = json.load(f).get("weight_map", {})
        else:
            self._index = {}

    def _get_file_info(self, path: str) -> Tuple[dict, int]:
        """Get metadata and data offset for a safetensors file (cached)."""
        if path not in self._file_cache:
            metadata, offset = load_safetensors_metadata(path)
            self._file_cache[path] = (metadata, offset)
        return self._file_cache[path]

    def _detect_layer_prefix(self) -> str:
        """Detect whether the model uses 'model.layers' or 'model.language_model.layers'."""
        if self._layer_prefix is not None:
            return self._layer_prefix

        self._load_index()

        # Check for layer 0 with both prefixes
        for prefix_template in [self.LAYER_PREFIX, self.ALT_LAYER_PREFIX]:
            prefix = prefix_template.format(layer_idx=0)
            test_key = f"{prefix}.mlp.gate_proj.qweight"
            if test_key in self._index:
                self._layer_prefix = prefix_template
                return self._layer_prefix

        # Fall back to default
        self._layer_prefix = self.LAYER_PREFIX
        return self._layer_prefix

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
        """Load a tensor by key from safetensors files."""
        file_path = self.find_tensor_file(key)
        if file_path is None:
            raise FileNotFoundError(f"Cannot find tensor: {key}")
        metadata, offset = self._get_file_info(file_path)
        return load_safetensors_tensor(file_path, key, metadata, offset)

    def load_awq_linear_weights(self, layer_prefix: str,
                                  proj_name: str) -> Dict[str, np.ndarray]:
        """Load AWQ linear weights for a single projection.

        Returns:
            dict with:
              'qweight': [K/8, N] INT32 packed nibbles
              'scales':  [K/group_size, N] FP16 per-group scales
              'zeros':   [K/group_size, N] FP16 all zeros (AWQ: no zero-point)
        """
        qweight_key = f"{layer_prefix}.{proj_name}.qweight"
        scales_key = f"{layer_prefix}.{proj_name}.scales"

        qweight = self.load_tensor(qweight_key)
        scales = self.load_tensor(scales_key)

        # AWQ has no zero-point: create zeros array matching scales shape
        zeros = np.zeros_like(scales, dtype=np.float16)

        # Ensure INT32 dtype for qweight
        if qweight.dtype != np.int32:
            qweight = qweight.view(np.int32).reshape(qweight.shape)

        return {
            'qweight': qweight,
            'scales': scales.astype(np.float16),
            'zeros': zeros,
        }

    def load_layer(self, layer_idx: int,
                   tp_size: int = 1, tp_rank: int = 0) -> dict:
        """Load all weights for one transformer layer.

        Returns dict with keys matching GPTQ loader output:
          {gate,up,down}_{qweight,scales,zeros}
          attn_norm, ffn_norm
          layer_type
          (attention weights same as GPTQ if present)

        NOTE: AWQ typically only quantizes FFN layers.
        Attention weights (q,k,v,o projections) are FP16.
        """
        prefix_template = self._detect_layer_prefix()
        prefix = prefix_template.format(layer_idx=layer_idx)
        mlp_prefix = f"{prefix}.mlp"
        weights = {}

        # Determine layer type
        if self.config.is_full_attention(layer_idx):
            weights['layer_type'] = 'full_attention'
            self._load_full_attn_weights(prefix, weights, tp_size, tp_rank)
        else:
            weights['layer_type'] = 'linear_attention'
            self._load_linear_attn_weights(prefix, weights)

        # MLP weights (INT4 AWQ)
        for proj, short in [('gate_proj', 'gate'), ('up_proj', 'up'),
                             ('down_proj', 'down')]:
            w = self.load_awq_linear_weights(mlp_prefix, proj)
            if tp_size > 1:
                w = self._shard_mlp_weights(w, proj, tp_size, tp_rank)
            weights[f'{short}_qweight'] = w['qweight']
            weights[f'{short}_scales'] = w['scales']
            weights[f'{short}_zeros'] = w['zeros']

        # Layer norms
        attn_norm = self.load_tensor(f"{prefix}.input_layernorm.weight")
        ffn_norm = self.load_tensor(f"{prefix}.post_attention_layernorm.weight")
        weights['attn_norm'] = (1.0 + attn_norm.astype(np.float32)).astype(np.float16)
        weights['ffn_norm'] = (1.0 + ffn_norm.astype(np.float32)).astype(np.float16)

        return weights

    def _shard_mlp_weights(self, w: dict, proj_name: str,
                            num_shards: int, shard_id: int) -> dict:
        """Shard MLP weights for tensor parallelism."""
        qweight = w['qweight']
        scales = w['scales']
        zeros = w['zeros']

        # gate_proj and up_proj: shard on N (output) dimension
        # down_proj: shard on K (input) dimension
        if 'down' not in proj_name:
            # Shard along N (column) dimension
            N = qweight.shape[1]
            shard_n = N // num_shards
            start = shard_id * shard_n
            end = start + shard_n
            return {
                'qweight': qweight[:, start:end].copy(),
                'scales': scales[:, start:end].copy(),
                'zeros': zeros[:, start:end].copy(),
            }
        else:
            # Shard along K (row) dimension
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

    def _load_full_attn_weights(self, prefix: str, weights: dict,
                                 tp_size: int, tp_rank: int):
        """Load full attention projection weights (FP16, typically not quantized in AWQ).

        AWQ usually leaves attention projections in FP16/BF16.
        Same shapes as GPTQ loader full attention.
        """
        q_weight = self.load_tensor(f"{prefix}.self_attn.q_proj.weight")
        k_weight = self.load_tensor(f"{prefix}.self_attn.k_proj.weight")
        v_weight = self.load_tensor(f"{prefix}.self_attn.v_proj.weight")
        o_weight = self.load_tensor(f"{prefix}.self_attn.o_proj.weight")

        # q_proj is packed: [num_heads, head_dim*2, hidden]
        num_heads = self.config.num_attention_heads  # 24
        head_dim = self.config.head_dim  # 256
        hidden = self.config.hidden_size  # 5120
        q_reshaped = q_weight.reshape(num_heads, head_dim * 2, hidden)
        weights['q_weight'] = q_reshaped[:, :head_dim, :].reshape(
            num_heads * head_dim, hidden).copy()
        weights['q_gate_weight'] = q_reshaped[:, head_dim:, :].reshape(
            num_heads * head_dim, hidden).copy()
        weights['k_weight'] = k_weight
        weights['v_weight'] = v_weight
        weights['o_weight'] = o_weight

        q_norm = self.load_tensor(f"{prefix}.self_attn.q_norm.weight")
        k_norm = self.load_tensor(f"{prefix}.self_attn.k_norm.weight")
        weights['q_norm'] = (1.0 + q_norm.astype(np.float32)).astype(np.float16)
        weights['k_norm'] = (1.0 + k_norm.astype(np.float32)).astype(np.float16)

    def _load_linear_attn_weights(self, prefix: str, weights: dict):
        """Load linear attention weights (FP16, not quantized in AWQ)."""
        la = f"{prefix}.linear_attn"

        weights['la_in_proj_qkv'] = self.load_tensor(f"{la}.in_proj_qkv.weight")
        weights['la_in_proj_a'] = self.load_tensor(f"{la}.in_proj_a.weight")
        weights['la_in_proj_b'] = self.load_tensor(f"{la}.in_proj_b.weight")
        weights['la_in_proj_z'] = self.load_tensor(f"{la}.in_proj_z.weight")
        weights['la_conv1d'] = self.load_tensor(f"{la}.conv1d.weight")
        weights['la_A_log'] = self.load_tensor(f"{la}.A_log")
        weights['la_dt_bias'] = self.load_tensor(f"{la}.dt_bias")
        weights['la_norm'] = self.load_tensor(f"{la}.norm.weight")
        weights['la_out_proj'] = self.load_tensor(f"{la}.out_proj.weight")

    def load_embedding(self) -> np.ndarray:
        """Load embedding weight [vocab_size, hidden_dim] FP16.

        AWQ uses 'model.embed_tokens.weight' (vs GPTQ 'model.language_model.embed_tokens.weight')
        Falls back to GPTQ naming if not found.
        """
        for key in ["model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight"]:
            try:
                return self.load_tensor(key)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("Cannot find embed_tokens.weight in model")

    def load_lm_head(self) -> np.ndarray:
        """Load LM head weight [vocab_size, hidden_dim] FP16."""
        try:
            return self.load_tensor("lm_head.weight")
        except FileNotFoundError:
            return self.load_embedding()

    def load_final_norm(self) -> np.ndarray:
        """Load final RMSNorm weight [hidden_dim] FP16.

        AWQ uses 'model.norm.weight' (vs GPTQ 'model.language_model.norm.weight')
        """
        for key in ["model.norm.weight", "model.language_model.norm.weight"]:
            try:
                w = self.load_tensor(key)
                return (1.0 + w.astype(np.float32)).astype(np.float16)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("Cannot find norm.weight in model")


def load_awq_or_gptq(model_dir: str, config: QwenConfig,
                      quant_format: str = 'auto',
                      **kwargs):
    """Auto-detect and load AWQ or GPTQ weights.

    Args:
        model_dir:    Path to model directory
        config:       QwenConfig instance
        quant_format: 'auto' to detect, 'awq' or 'gptq' to force

    Returns:
        (loader, detected_format) tuple
    """
    from src.model.weight_loader import QwenWeightLoader

    if quant_format == 'auto':
        quant_format = detect_awq_format(model_dir)

    if quant_format == 'awq':
        return AWQWeightLoader(model_dir, config, **kwargs), 'awq'
    else:
        return QwenWeightLoader(model_dir, config, **kwargs), quant_format
