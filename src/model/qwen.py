"""
Qwen 3.5 27B model definition.

Architecture:
- 64 transformer layers with hybrid attention pattern
- 48 linear attention layers + 16 full attention layers (3 linear → 1 full)
- Hidden dim: 5120
- Full attention: 24 heads × 256 head_dim, 4 KV heads (GQA 6:1)
- Linear attention: 16 key heads × 128 dim, 48 value heads × 128 dim, conv_kernel=4
- FFN hidden: 17408 (SwiGLU)
- Vocab: 248320
- M-RoPE: base 10M, partial_rotary_factor=0.25, sections=[11,11,10]
- Attention weights: FP16 (NOT quantized). Only FFN is INT4.

At INT4 (GPTQ, group_size=128):
- ~30GB total (mixed FP16 attn + INT4 FFN)
- Needs 2-3x MI50 (16GB each)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QwenConfig:
    """Qwen 3.5 27B configuration."""
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    num_attention_heads: int = 24      # for full attention layers
    num_key_value_heads: int = 4       # for full attention layers (GQA)
    head_dim: int = 256                # for full attention layers
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rms_norm_eps: float = 1e-6

    # Hybrid attention
    full_attention_interval: int = 4   # every 4th layer uses full attention
    layer_types: list = field(default_factory=list)  # populated by load_config

    # Linear attention params
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # RoPE
    partial_rotary_factor: float = 0.25
    mrope_sections: list = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True

    # Quantization (only applies to FFN, NOT attention)
    quant_method: str = "gptq"
    bits: int = 4
    group_size: int = 128
    quant_sym: bool = True  # symmetric quantization (zero point = midpoint)
    attn_quantized: bool = False  # attention weights are FP16

    # Output gate
    attn_output_gate: bool = True

    def __post_init__(self):
        if not self.layer_types:
            # Generate default pattern: 3 linear + 1 full, repeating
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % self.full_attention_interval == 0:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("linear_attention")

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    def is_linear_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def full_attn_q_dim(self) -> int:
        """Q projection output dim for full attention layers."""
        return self.num_attention_heads * self.head_dim  # 24 * 256 = 6144

    @property
    def full_attn_kv_dim(self) -> int:
        """K or V projection output dim for full attention layers."""
        return self.num_key_value_heads * self.head_dim  # 4 * 256 = 1024

    @property
    def linear_attn_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim  # 16 * 128 = 2048

    @property
    def linear_attn_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim  # 48 * 128 = 6144

    @property
    def kv_heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


# Qwen 3.5 27B GEMM shapes for tuning
# h=5120, inter=17408
# Full attention: Q=6144, K=1024, V=1024
# Linear attention: K=2048, V=6144
QWEN_GEMM_SHAPES = [
    # Full attention QKV: FP16 GEMM (NOT quantized)
    (1, 6144 + 1024 + 1024, 5120, "full_attn_qkv"),
    (128, 6144 + 1024 + 1024, 5120, "full_attn_qkv_128"),

    # Linear attention QKV: FP16 GEMM
    (1, 2048 + 6144, 5120, "linear_attn_qkv"),
    (128, 2048 + 6144, 5120, "linear_attn_qkv_128"),

    # Attention output projection: FP16 GEMM
    (1, 5120, 6144, "attn_out_proj"),
    (128, 5120, 6144, "attn_out_proj_128"),

    # FFN up + gate: INT4 GEMM (quantized)
    (1, 17408, 5120, "ffn_up"),
    (128, 17408, 5120, "ffn_up_128"),

    # FFN down: INT4 GEMM (quantized)
    (1, 5120, 17408, "ffn_down"),
    (128, 5120, 17408, "ffn_down_128"),
]


def load_config_from_json(model_dir: str) -> QwenConfig:
    """Load QwenConfig from a HuggingFace config.json file."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path) as f:
        hf = json.load(f)

    # Qwen 3.5 nests text config under "text_config"
    text_cfg = hf.get("text_config", hf)

    # RoPE config
    rope_params = text_cfg.get("rope_parameters", {})

    config = QwenConfig(
        hidden_size=text_cfg.get("hidden_size", 5120),
        intermediate_size=text_cfg.get("intermediate_size", 17408),
        num_hidden_layers=text_cfg.get("num_hidden_layers", 64),
        num_attention_heads=text_cfg.get("num_attention_heads", 24),
        num_key_value_heads=text_cfg.get("num_key_value_heads", 4),
        head_dim=text_cfg.get("head_dim", 256),
        vocab_size=text_cfg.get("vocab_size", 248320),
        max_position_embeddings=text_cfg.get("max_position_embeddings", 262144),
        rope_theta=rope_params.get("rope_theta",
                                    text_cfg.get("rope_theta", 10000000.0)),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        full_attention_interval=text_cfg.get("full_attention_interval", 4),
        layer_types=text_cfg.get("layer_types", []),
        linear_key_head_dim=text_cfg.get("linear_key_head_dim", 128),
        linear_num_key_heads=text_cfg.get("linear_num_key_heads", 16),
        linear_num_value_heads=text_cfg.get("linear_num_value_heads", 48),
        linear_value_head_dim=text_cfg.get("linear_value_head_dim", 128),
        linear_conv_kernel_dim=text_cfg.get("linear_conv_kernel_dim", 4),
        partial_rotary_factor=rope_params.get("partial_rotary_factor",
                                               text_cfg.get("partial_rotary_factor", 0.25)),
        mrope_sections=rope_params.get("mrope_section", [11, 11, 10]),
        mrope_interleaved=rope_params.get("mrope_interleaved", True),
        attn_output_gate=text_cfg.get("attn_output_gate", True),
    )

    # Quantization config
    quant_config = hf.get("quantization_config", {})
    if quant_config:
        config.quant_method = quant_config.get("quant_method", "gptq")
        config.bits = quant_config.get("bits", 4)
        config.group_size = quant_config.get("group_size", 128)
        config.quant_sym = quant_config.get("sym", True)
        # Check if attention is excluded from quantization
        dynamic = quant_config.get("dynamic", {})
        config.attn_quantized = "-:.*attn.*" not in dynamic

    return config


def kv_cache_bytes(config: QwenConfig, max_seq_len: int) -> int:
    """Calculate total KV cache size in bytes for all layers.

    Only full attention layers need KV cache. Linear attention
    layers use recurrent state (much smaller).
    """
    full_attn_layers = config.num_full_attention_layers
    per_layer = max_seq_len * config.num_key_value_heads * config.head_dim * 2
    return 2 * full_attn_layers * per_layer  # K + V


def memory_budget(config: QwenConfig, max_seq_len: int = 2048,
                  num_gpus: int = 1) -> dict:
    """Estimate memory usage for the model."""
    # Rough estimate: attention FP16, FFN INT4
    h = config.hidden_size
    inter = config.intermediate_size
    n_layers = config.num_hidden_layers

    # Attention weights (FP16): Q, K, V, O projections per layer
    full_q_dim = config.full_attn_q_dim
    full_kv_dim = config.full_attn_kv_dim
    lin_k_dim = config.linear_attn_key_dim
    lin_v_dim = config.linear_attn_value_dim

    n_full = config.num_full_attention_layers
    n_lin = config.num_linear_attention_layers

    attn_params_full = n_full * (h * (full_q_dim + 2 * full_kv_dim + h))
    attn_params_lin = n_lin * (h * (lin_k_dim + lin_v_dim + h))
    attn_bytes = (attn_params_full + attn_params_lin) * 2  # FP16

    # FFN weights (INT4): up, gate, down per layer
    ffn_params = n_layers * 3 * h * inter
    ffn_bytes = ffn_params // 2  # 4 bits
    ffn_scale_bytes = (ffn_params // config.group_size) * 2  # FP16 scales

    # Embeddings
    embed_bytes = config.vocab_size * h * 2 * 2  # embed + lm_head, FP16
    norm_bytes = n_layers * 2 * h * 2  # 2 norms per layer

    model = attn_bytes + ffn_bytes + ffn_scale_bytes + embed_bytes + norm_bytes
    kv = kv_cache_bytes(config, max_seq_len)
    scratch = max_seq_len * h * 2 * 8
    total = model + kv + scratch
    per_gpu = total / num_gpus
    mi50_vram = 16 * 1024 * 1024 * 1024

    return {
        "model_gb": model / 1e9,
        "attn_fp16_gb": attn_bytes / 1e9,
        "ffn_int4_gb": (ffn_bytes + ffn_scale_bytes) / 1e9,
        "kv_cache_gb": kv / 1e9,
        "scratch_gb": scratch / 1e9,
        "total_gb": total / 1e9,
        "per_gpu_gb": per_gpu / 1e9,
        "fits_per_gpu": per_gpu < mi50_vram,
        "min_gpus": max(1, -(-total // mi50_vram)),
    }
