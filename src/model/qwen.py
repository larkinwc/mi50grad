"""
Qwen 3.5 27B model definition.

Architecture (approximate):
- 64 transformer layers
- Hidden dim: 4096
- Num heads: 32 (128 dim each)
- Num KV heads: 8 (GQA, 4:1 ratio)
- FFN hidden: 11008 (SwiGLU: up_proj * gate_proj -> SiLU -> down_proj)
- Vocab: 152064
- RoPE: base 1000000, max_position_embeddings 131072

At INT4 (GPTQ/AWQ, group_size=128):
- ~14GB model weights
- Fits on 2x MI50 (16GB each) with room for KV cache

This is a Phase 5 component — defines the model structure for graph construction.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QwenConfig:
    """Qwen 3.5 27B configuration."""
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 64
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    # Quantization
    quant_method: str = "gptq"  # "gptq" or "awq"
    bits: int = 4
    group_size: int = 128

    @property
    def kv_heads_per_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def model_bytes_fp16(self) -> int:
        """Approximate model size in FP16."""
        params = (
            self.vocab_size * self.hidden_size +  # embed
            self.num_hidden_layers * (
                3 * self.hidden_size * self.hidden_size +  # QKV (approximate)
                self.hidden_size * self.hidden_size +  # output proj
                3 * self.hidden_size * self.intermediate_size +  # FFN (up, gate, down)
                2 * self.hidden_size  # norms
            ) +
            self.vocab_size * self.hidden_size  # lm_head
        )
        return params * 2  # FP16

    @property
    def model_bytes_int4(self) -> int:
        """Approximate model size at INT4 (weights only, scales in FP16)."""
        # Weights: 4 bits each
        # Scales: FP16 per group
        weight_params = self.model_bytes_fp16 // 2  # param count
        weight_bytes = weight_params // 2  # 4 bits each
        scale_bytes = (weight_params // self.group_size) * 2  # FP16 scales
        return weight_bytes + scale_bytes


# Qwen 3.5 27B GEMM shapes for tuning
# Format: (M_range, N, K, description)
QWEN_GEMM_SHAPES = [
    # Attention QKV projection: [batch*seq, hidden] @ [hidden, 3*hidden]
    # (variable M, but key sizes during decode are M=1 or small)
    (1, 4096 + 1024, 4096, "qkv_proj"),       # Q + KV (GQA)
    (128, 4096 + 1024, 4096, "qkv_proj_128"),

    # Attention output projection: [batch*seq, hidden] @ [hidden, hidden]
    (1, 4096, 4096, "out_proj"),
    (128, 4096, 4096, "out_proj_128"),

    # FFN up + gate: [batch*seq, hidden] @ [hidden, intermediate]
    (1, 11008, 4096, "ffn_up"),
    (128, 11008, 4096, "ffn_up_128"),

    # FFN down: [batch*seq, intermediate] @ [intermediate, hidden]
    (1, 4096, 11008, "ffn_down"),
    (128, 4096, 11008, "ffn_down_128"),
]
