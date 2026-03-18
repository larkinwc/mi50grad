"""
Compressed-tensors format weight loader for Qwen 3.5 27B INT4.

Compressed-tensors format (HuggingFace compressed-tensors quantization):
  - weight_packed:  [N, K/8] INT32 (N-major, 8 INT4 nibbles packed per uint32)
  - weight_scale:   [N, K/group_size] BF16 (per-row, per-group scales)
  - weight_shape:   [2] INT64 = [out_features (N), in_features (K)]
  - NO zero-point tensor (symmetric quantization, zero_point=0)
  - group_size=32 (from config.json quantization_config.config_groups.group_0.weights.group_size)
  - Dequantization: w = q * scale (no zero-point subtraction)

Compressed-tensors vs AWQ/GPTQ:
  - AWQ/GPTQ: qweight [K/8, N] (K-major packing)
  - CT:       weight_packed [N, K/8] (N-major packing)
  - AWQ/GPTQ: scales [K/group_size, N]
  - CT:       weight_scale [N, K/group_size]
  - Same nibble packing: 8 INT4 values (0..15) per uint32, low-to-high

Layer naming:
  - Text layers: model.language_model.layers.N (same as GPTQ)
  - Embeddings: model.language_model.embed_tokens.weight (FP16/BF16)
  - LM head: lm_head.weight (FP16/BF16)
  - Final norm: model.language_model.norm.weight

Skip vision layers:
  - model.visual.* (all vision encoder layers)
  - model.merger.* (vision-language merger)
  - mtp.* (multi-token prediction heads, vision-related)

DeltaNet linear attention:
  - in_proj_qkv, in_proj_z, out_proj: INT4 quantized (weight_packed)
  - in_proj_a, in_proj_b: FP16/BF16 (not quantized, in ignore list)
  - Other tensors (conv1d, A_log, dt_bias, norm): FP16/BF16

The loader produces output compatible with the existing GPTQ/AWQ loader format:
  {proj}_qweight: [K/8, N] INT32 (transposed from CT N-major to K-major)
  {proj}_scales:  [K/group_size, N] FP16 (transposed)
  {proj}_zeros:   [K/group_size, N] FP16 all zeros (symmetric quant)
"""

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from src.model.qwen import QwenConfig
from src.model.weight_loader import load_safetensors_metadata, load_safetensors_tensor


def detect_compressed_tensors_format(model_dir: str) -> bool:
    """Detect if model uses compressed-tensors format.
    
    Returns:
        True if weight_packed tensors found (compressed-tensors format)
        False otherwise
    """
    model_dir = Path(model_dir)
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
    
    # Check for compressed-tensors signature
    has_weight_packed = any('weight_packed' in k for k in all_keys)
    has_weight_scale = any('weight_scale' in k for k in all_keys)
    has_weight_shape = any('weight_shape' in k for k in all_keys)
    
    return has_weight_packed and has_weight_scale and has_weight_shape


def detect_quant_format(model_dir: str) -> str:
    """Detect quantization format from safetensors index.
    
    Returns:
        'compressed-tensors' - if weight_packed tensors found
        'awq'  - if qweight but no qzeros (AWQ format)
        'gptq' - if qweight and qzeros tensors found
        'fp16' - if no quantized weights found
    """
    # Check for compressed-tensors first
    if detect_compressed_tensors_format(model_dir):
        return 'compressed-tensors'
    
    # Fall back to AWQ/GPTQ detection
    from src.model.awq_loader import detect_awq_format
    return detect_awq_format(model_dir)


def ct_dequantize(weight_packed: np.ndarray, weight_scale: np.ndarray,
                   original_shape: Tuple[int, int],
                   group_size: int = 32) -> np.ndarray:
    """Dequantize compressed-tensors packed INT4 weights to FP32.
    
    CT format uses N-major packing: weight_packed [N, K/8], scales [N, num_groups]
    
    Args:
        weight_packed:   [N, K/8] INT32 packed nibbles (unsigned 0..15)
        weight_scale:    [N, K/group_size] BF16/FP16 per-group scales
        original_shape:  (N, K) original weight shape
        group_size:      quantization group size (default: 32)
    
    Returns:
        w_fp32: [N, K] FP32 dequantized weights
    
    The nibble values 0..15 are treated as unsigned integers.
    CT uses symmetric quantization with zero_point=0, so w = q * scale.
    """
    N, K_packed = weight_packed.shape
    K = original_shape[1]  # in_features
    
    # Unpack nibbles: [N, K] uint8 (0..15)
    w_raw = np.zeros((N, K), dtype=np.uint8)
    mask = 0xF
    for b in range(8):
        w_raw[:, b::8] = (weight_packed >> (b * 4)) & mask
    
    # Expand scales from [N, K/group_size] to [N, K]
    scales_fp32 = weight_scale.astype(np.float32)
    scales_expanded = np.repeat(scales_fp32, group_size, axis=1)  # [N, K]
    
    # CT dequant: w = q * scale (no zero subtraction, symmetric)
    w_fp32 = w_raw.astype(np.float32) * scales_expanded  # [N, K]
    
    return w_fp32


def ct_to_engine_format(weight_packed: np.ndarray, weight_scale: np.ndarray,
                         original_shape: Tuple[int, int],
                         group_size: int = 32) -> Dict[str, np.ndarray]:
    """Convert compressed-tensors format to engine format (GPTQ-compatible).
    
    Engine format expects K-major packing like GPTQ:
      qweight: [K/8, N] INT32
      scales:  [K/group_size, N] FP16
      zeros:   [K/group_size, N] FP16 (all zeros for symmetric quant)
    
    Args:
        weight_packed:   [N, K/8] INT32 (CT N-major)
        weight_scale:    [N, K/group_size] FP16/BF16
        original_shape:  (N, K)
        group_size:      quantization group size
    
    Returns:
        dict with:
          'qweight': [K/8, N] INT32 (transposed and repacked)
          'scales':  [K/group_size, N] FP16 (transposed)
          'zeros':   [K/group_size, N] FP16 (all zeros)
    """
    N, K = original_shape
    K_packed = K // 8
    num_groups = K // group_size
    mask = 0xF
    
    # Step 1: Unpack from N-major [N, K/8] to [N, K] uint8
    w_raw = np.zeros((N, K), dtype=np.uint8)
    for b in range(8):
        w_raw[:, b::8] = (weight_packed >> (b * 4)) & mask
    
    # Step 2: Transpose to K-major: [K, N]
    w_raw_T = w_raw.T  # [K, N]
    
    # Step 3: Repack into K-major format [K/8, N]
    qweight = np.zeros((K_packed, N), dtype=np.int32)
    for b in range(8):
        qweight |= (w_raw_T[b::8, :].astype(np.int32) << (b * 4))
    
    # Step 4: Transpose scales from [N, K/group_size] to [K/group_size, N]
    scales = weight_scale.astype(np.float16).T  # [K/group_size, N]
    
    # Step 5: Create zeros array (all zeros for symmetric quant)
    zeros = np.zeros((num_groups, N), dtype=np.float16)
    
    return {
        'qweight': qweight,
        'scales': scales,
        'zeros': zeros,
    }


class CompressedTensorsLoader:
    """Loads compressed-tensors quantized model weights from safetensors files.
    
    Compressed-tensors naming convention:
      model.language_model.layers.N.mlp.gate_proj.weight_packed    [N, K/8] INT32
      model.language_model.layers.N.mlp.gate_proj.weight_scale     [N, K/group_size] BF16
      model.language_model.layers.N.mlp.gate_proj.weight_shape     [2] INT64 = [N, K]
      (NO weight_zero_point tensors - symmetric quantization)
    
    Skip patterns (vision-related):
      model.visual.*       - vision encoder
      model.merger.*       - vision-language merger
      mtp.*                - multi-token prediction (vision-related)
    
    The loader produces output compatible with the GPTQWeightLoader format:
      {proj}_qweight, {proj}_scales, {proj}_zeros
    """
    
    LAYER_PREFIX = "model.language_model.layers.{layer_idx}"
    
    # Skip patterns for vision layers
    SKIP_PATTERNS = [
        "model.visual.",
        "model.merger.",
        "mtp.",
    ]
    
    # Linear attention projections that are quantized
    QUANTIZED_LINEAR_ATTN = [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    ]
    
    # Linear attention projections that are FP16 (not quantized)
    FP16_LINEAR_ATTN = [
        "in_proj_a",
        "in_proj_b",
    ]
    
    def __init__(self, model_dir: str, config: QwenConfig,
                 group_size: int = 32):
        self.model_dir = Path(model_dir)
        self.config = config
        self.group_size = group_size
        self.num_layers = config.num_hidden_layers
        self._file_cache = {}
        self._index = None
    
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
    
    def _should_skip_tensor(self, key: str) -> bool:
        """Check if tensor should be skipped (vision-related)."""
        for pattern in self.SKIP_PATTERNS:
            if key.startswith(pattern):
                return True
        return False
    
    def _get_file_info(self, path: str) -> Tuple[dict, int]:
        """Get metadata and data offset for a safetensors file (cached)."""
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
        """Load a tensor by key from safetensors files."""
        file_path = self.find_tensor_file(key)
        if file_path is None:
            raise FileNotFoundError(f"Cannot find tensor: {key}")
        metadata, offset = self._get_file_info(file_path)
        return load_safetensors_tensor(file_path, key, metadata, offset)
    
    def load_ct_linear_weights(self, layer_prefix: str,
                                 proj_name: str) -> Dict[str, np.ndarray]:
        """Load compressed-tensors linear weights for a single projection.
        
        Returns:
            dict with:
              'qweight': [K/8, N] INT32 (converted to K-major)
              'scales':  [K/group_size, N] FP16
              'zeros':   [K/group_size, N] FP16 (all zeros)
        """
        packed_key = f"{layer_prefix}.{proj_name}.weight_packed"
        scale_key = f"{layer_prefix}.{proj_name}.weight_scale"
        shape_key = f"{layer_prefix}.{proj_name}.weight_shape"
        
        weight_packed = self.load_tensor(packed_key)
        weight_scale = self.load_tensor(scale_key)
        weight_shape = self.load_tensor(shape_key)
        
        # weight_shape is [N, K]
        original_shape = (int(weight_shape[0]), int(weight_shape[1]))
        
        # Convert to engine format
        return ct_to_engine_format(
            weight_packed, weight_scale, original_shape, self.group_size)
    
    def load_layer(self, layer_idx: int,
                   tp_size: int = 1, tp_rank: int = 0) -> dict:
        """Load all weights for one transformer layer.
        
        Returns dict with keys matching GPTQ loader output:
          {gate,up,down}_{qweight,scales,zeros}
          attn_norm, ffn_norm
          layer_type
          (attention weights same as GPTQ if present)
        """
        prefix = self.LAYER_PREFIX.format(layer_idx=layer_idx)
        mlp_prefix = f"{prefix}.mlp"
        weights = {}
        
        # Determine layer type
        if self.config.is_full_attention(layer_idx):
            weights['layer_type'] = 'full_attention'
            self._load_full_attn_weights(prefix, weights, tp_size, tp_rank)
        else:
            weights['layer_type'] = 'linear_attention'
            self._load_linear_attn_weights(prefix, weights)
        
        # MLP weights (INT4 compressed-tensors)
        for proj, short in [('gate_proj', 'gate'), ('up_proj', 'up'),
                             ('down_proj', 'down')]:
            w = self.load_ct_linear_weights(mlp_prefix, proj)
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
        """Load full attention projection weights.
        
        In compressed-tensors format, attention projections are also INT4 quantized:
          q_proj, k_proj, v_proj, o_proj: weight_packed, weight_scale, weight_shape
        Norm weights remain FP16/BF16.
        
        Output format matches GPTQ loader:
          q_weight, q_gate_weight: split from q_proj
          k_weight, v_weight, o_weight: from respective projections
        """
        # Load quantized attention projections (compressed-tensors format)
        q_proj = self.load_ct_linear_weights(f"{prefix}.self_attn", "q_proj")
        k_proj = self.load_ct_linear_weights(f"{prefix}.self_attn", "k_proj")
        v_proj = self.load_ct_linear_weights(f"{prefix}.self_attn", "v_proj")
        o_proj = self.load_ct_linear_weights(f"{prefix}.self_attn", "o_proj")
        
        # q_proj is packed: qweight shape [K/8, N] where K=12288 (Q+gate), N=5120
        # Split into q_weight and q_gate_weight
        # Original q_proj shape: [12288, 5120] = [num_heads * head_dim * 2, hidden]
        # qweight: [12288/8, 5120] = [1536, 5120]
        # After unpacking: [12288, 5120] -> reshape to [24, 512, 5120] -> split on dim 1
        num_heads = self.config.num_attention_heads  # 24
        head_dim = self.config.head_dim  # 256
        hidden = self.config.hidden_size  # 5120
        
        # Unpack q_proj from [K/8, N] to [K, N]
        K_q, N_q = q_proj['qweight'].shape
        q_unpacked = np.zeros((K_q * 8, N_q), dtype=np.uint8)
        mask = 0xF
        for b in range(8):
            q_unpacked[b::8, :] = (q_proj['qweight'] >> (b * 4)) & mask
        
        # Reshape and split: [12288, 5120] -> [24, 512, 5120] -> [24, 256, 5120] each
        q_reshaped = q_unpacked.reshape(num_heads, head_dim * 2, hidden)
        weights['q_weight'] = q_reshaped[:, :head_dim, :].reshape(
            num_heads * head_dim, hidden).copy()
        weights['q_gate_weight'] = q_reshaped[:, head_dim:, :].reshape(
            num_heads * head_dim, hidden).copy()
        
        # k_proj, v_proj, o_proj: keep as unpacked FP32 for now (simplified)
        # Unpack k_proj
        K_k, N_k = k_proj['qweight'].shape
        k_unpacked = np.zeros((K_k * 8, N_k), dtype=np.uint8)
        for b in range(8):
            k_unpacked[b::8, :] = (k_proj['qweight'] >> (b * 4)) & mask
        weights['k_weight'] = k_unpacked.astype(np.float16)
        
        # Unpack v_proj
        K_v, N_v = v_proj['qweight'].shape
        v_unpacked = np.zeros((K_v * 8, N_v), dtype=np.uint8)
        for b in range(8):
            v_unpacked[b::8, :] = (v_proj['qweight'] >> (b * 4)) & mask
        weights['v_weight'] = v_unpacked.astype(np.float16)
        
        # Unpack o_proj
        K_o, N_o = o_proj['qweight'].shape
        o_unpacked = np.zeros((K_o * 8, N_o), dtype=np.uint8)
        for b in range(8):
            o_unpacked[b::8, :] = (o_proj['qweight'] >> (b * 4)) & mask
        weights['o_weight'] = o_unpacked.astype(np.float16)
        
        # Load norm weights (FP16/BF16)
        q_norm = self.load_tensor(f"{prefix}.self_attn.q_norm.weight")
        k_norm = self.load_tensor(f"{prefix}.self_attn.k_norm.weight")
        weights['q_norm'] = (1.0 + q_norm.astype(np.float32)).astype(np.float16)
        weights['k_norm'] = (1.0 + k_norm.astype(np.float32)).astype(np.float16)
    
    def _load_linear_attn_weights(self, prefix: str, weights: dict):
        """Load linear attention weights.
        
        Quantized projections (INT4): in_proj_qkv, in_proj_z, out_proj
        FP16 projections: in_proj_a, in_proj_b
        Other FP16 tensors: conv1d, A_log, dt_bias, norm
        """
        la = f"{prefix}.linear_attn"
        
        # Quantized projections (compressed-tensors format)
        for proj in self.QUANTIZED_LINEAR_ATTN:
            w = self.load_ct_linear_weights(la, proj)
            weights[f'la_{proj}'] = w['qweight']
            weights[f'la_{proj}_scales'] = w['scales']
        
        # FP16 projections (not quantized)
        weights['la_in_proj_a'] = self.load_tensor(f"{la}.in_proj_a.weight")
        weights['la_in_proj_b'] = self.load_tensor(f"{la}.in_proj_b.weight")
        
        # Other FP16/BF16 tensors
        weights['la_conv1d'] = self.load_tensor(f"{la}.conv1d.weight")
        weights['la_A_log'] = self.load_tensor(f"{la}.A_log")
        weights['la_dt_bias'] = self.load_tensor(f"{la}.dt_bias")
        weights['la_norm'] = self.load_tensor(f"{la}.norm.weight")
        # out_proj is quantized, already loaded above as la_out_proj
    
    def load_embedding(self) -> np.ndarray:
        """Load embedding weight [vocab_size, hidden_dim] FP16."""
        return self.load_tensor("model.language_model.embed_tokens.weight")
    
    def load_lm_head(self) -> np.ndarray:
        """Load LM head weight [vocab_size, hidden_dim] FP16."""
        try:
            return self.load_tensor("lm_head.weight")
        except FileNotFoundError:
            return self.load_embedding()
    
    def load_final_norm(self) -> np.ndarray:
        """Load final RMSNorm weight [hidden_dim] FP16.
        
        Qwen 3.5 uses (1+weight)*(x/rms) normalization.
        """
        w = self.load_tensor("model.language_model.norm.weight")
        return (1.0 + w.astype(np.float32)).astype(np.float16)
    
    def get_all_tensor_keys(self) -> List[str]:
        """Get all tensor keys, excluding skipped (vision) layers."""
        self._load_index()
        keys = []
        for key in self._index.keys():
            if not self._should_skip_tensor(key):
                keys.append(key)
        return keys
    
    def count_layers(self) -> int:
        """Count number of text layers (excluding vision)."""
        self._load_index()
        layer_indices = set()
        for key in self._index.keys():
            if self._should_skip_tensor(key):
                continue
            # Extract layer index from key like "model.language_model.layers.0..."
            if "model.language_model.layers." in key:
                parts = key.split("model.language_model.layers.")
                if len(parts) > 1:
                    layer_num = parts[1].split(".")[0]
                    try:
                        layer_indices.add(int(layer_num))
                    except ValueError:
                        pass
        return len(layer_indices)


def load_compressed_tensors(model_dir: str, config: QwenConfig,
                             group_size: int = 32, **kwargs):
    """Load compressed-tensors format weights.
    
    Args:
        model_dir:   Path to model directory
        config:      QwenConfig instance
        group_size:  Quantization group size (default: 32)
    
    Returns:
        CompressedTensorsLoader instance
    """
    return CompressedTensorsLoader(model_dir, config, group_size)
