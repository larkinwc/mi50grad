"""
Inference engine for Qwen 3.5 27B on MI50.

Supports:
- Hybrid architecture: full attention (GQA, head_dim=256) + linear attention (Mamba SSM)
- FP16 attention weights (gemv_fp16) + INT4 FFN weights (gemv_int4)
- Single-token decode with online softmax decode attention
- Multi-token prefill (sequential fallback)
- Q/K RMSNorm, partial RoPE, sigmoid output gate
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hsaco, build_hip_hsaco, Kernel
from src.model.qwen import QwenConfig


ASM_DIR = Path(__file__).parent.parent / "asm"
HIP_DIR = Path(__file__).parent.parent / "kernels"
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

    def get_hip(self, kernel_name: str, hip_name: str,
                extra_flags: list = None, hsaco_suffix: str = "") -> int:
        """Get kernel from HIP C++ source, building if needed.

        extra_flags: additional compiler flags (e.g. ['-DNUM_V_HEADS=24'])
        hsaco_suffix: appended to .hsaco filename for variant builds
        """
        cache_key = kernel_name + hsaco_suffix
        if cache_key not in self._funcs:
            hip_path = HIP_DIR / f"{hip_name}.hip"
            hsaco_path = BUILD_DIR / f"{hip_name}{hsaco_suffix}.hsaco"
            BUILD_DIR.mkdir(parents=True, exist_ok=True)

            if not hsaco_path.exists() or hip_path.stat().st_mtime > hsaco_path.stat().st_mtime:
                build_hip_hsaco(str(hip_path), str(hsaco_path),
                                extra_flags=extra_flags)

            if str(hsaco_path) not in self._modules:
                self._modules[str(hsaco_path)] = self.device.load_hsaco(str(hsaco_path))
            module = self._modules[str(hsaco_path)]
            self._funcs[cache_key] = self.device.get_kernel(module, kernel_name)

        return self._funcs[cache_key]


class KVCache:
    """Contiguous KV cache for full attention layers only."""

    def __init__(self, config: QwenConfig, max_seq_len: int, device: GPUDevice,
                 tp_size: int = 1):
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        self.current_len = 0

        # Only full attention layers need KV cache
        self.num_full_layers = config.num_full_attention_layers
        self.full_layer_indices = [i for i in range(config.num_hidden_layers)
                                    if config.is_full_attention(i)]

        # K cache: [num_full_layers, max_seq, local_kv_heads, head_dim] FP16
        self.local_num_kv_heads = config.num_key_value_heads // tp_size
        per_layer = max_seq_len * self.local_num_kv_heads * config.head_dim * 2
        self.k_size = self.num_full_layers * per_layer
        self.v_size = self.k_size

        self.d_k = device.malloc(self.k_size)
        self.d_v = device.malloc(self.v_size)
        device.memset(self.d_k, 0, self.k_size)
        device.memset(self.d_v, 0, self.v_size)

    def _full_layer_slot(self, layer_idx: int) -> int:
        """Map global layer_idx to slot in KV cache (only full attention layers)."""
        return self.full_layer_indices.index(layer_idx)

    def layer_k_ptr(self, layer_idx: int) -> int:
        slot = self._full_layer_slot(layer_idx)
        offset = (slot * self.max_seq_len *
                  self.local_num_kv_heads * self.config.head_dim * 2)
        return self.d_k + offset

    def layer_v_ptr(self, layer_idx: int) -> int:
        slot = self._full_layer_slot(layer_idx)
        offset = (slot * self.max_seq_len *
                  self.local_num_kv_heads * self.config.head_dim * 2)
        return self.d_v + offset

    def append_kv(self, layer_idx: int, k_data: bytes, v_data: bytes):
        """Append one position's K and V to the cache (from host bytes)."""
        slot = self._full_layer_slot(layer_idx)
        kv_size = self.local_num_kv_heads * self.config.head_dim * 2

        base_offset = (slot * self.max_seq_len *
                       self.local_num_kv_heads * self.config.head_dim +
                       self.current_len *
                       self.local_num_kv_heads * self.config.head_dim) * 2

        self.device._ensure_device()
        self.device.hip.memcpy_h2d(self.d_k + base_offset, k_data, kv_size)
        self.device.hip.memcpy_h2d(self.d_v + base_offset, v_data, kv_size)

    def append_kv_gpu(self, layer_idx: int, d_k_src: int, d_v_src: int):
        """Append one position's K and V from GPU buffers (device-to-device)."""
        slot = self._full_layer_slot(layer_idx)
        kv_size = self.local_num_kv_heads * self.config.head_dim * 2

        base_offset = (slot * self.max_seq_len *
                       self.local_num_kv_heads * self.config.head_dim +
                       self.current_len *
                       self.local_num_kv_heads * self.config.head_dim) * 2

        self.device._ensure_device()
        self.device.hip.memcpy_d2d(self.d_k + base_offset, d_k_src, kv_size)
        self.device.hip.memcpy_d2d(self.d_v + base_offset, d_v_src, kv_size)

    def advance(self):
        self.current_len += 1

    def cleanup(self):
        self.device.free(self.d_k)
        self.device.free(self.d_v)


class DeltaNetState:
    """Recurrent state for linear attention (Gated DeltaNet) layers.

    State lives on GPU for fast kernel access. CPU copies for fallback.
    Supports tensor parallelism: each GPU holds state for its local heads.
    """

    def __init__(self, config: QwenConfig, device: GPUDevice = None,
                 tp_size: int = 1):
        self.config = config
        self.device = device
        num_lin = config.num_linear_attention_layers

        # Local head counts for TP
        local_v_heads = config.linear_num_value_heads // tp_size
        local_k_heads = config.linear_num_key_heads // tp_size

        # DeltaNet state: [local_v_heads, k_head_dim, v_head_dim] per layer
        state_size = (local_v_heads *
                      config.linear_key_head_dim *
                      config.linear_value_head_dim)
        self.state_size = state_size

        # Conv1d state: [local_conv_dim, kernel_dim - 1] per layer
        conv_dim = (local_k_heads * config.linear_key_head_dim * 2 +
                    local_v_heads * config.linear_value_head_dim)
        self.conv_dim = conv_dim
        conv_state_size = conv_dim * (config.linear_conv_kernel_dim - 1)
        self.conv_state_size = conv_state_size

        # GPU state buffers
        self.d_states = []
        self.d_conv_states = []
        if device is not None:
            for _ in range(num_lin):
                d_s = device.malloc(state_size * 4)  # FP32
                device.memset(d_s, 0, state_size * 4)
                self.d_states.append(d_s)

                d_c = device.malloc(conv_state_size * 2)  # FP16
                device.memset(d_c, 0, conv_state_size * 2)
                self.d_conv_states.append(d_c)

        # CPU fallback state
        self.states = [
            np.zeros((local_v_heads,
                       config.linear_key_head_dim,
                       config.linear_value_head_dim),
                     dtype=np.float32)
            for _ in range(num_lin)
        ]
        self.conv_states = [
            np.zeros((conv_dim, config.linear_conv_kernel_dim - 1), dtype=np.float16)
            for _ in range(num_lin)
        ]

        # Precompute linear layer slot mapping
        self._slot_map = {}
        slot = 0
        for i in range(config.num_hidden_layers):
            if config.is_linear_attention(i):
                self._slot_map[i] = slot
                slot += 1

    def get_slot(self, layer_idx: int, config: QwenConfig) -> int:
        return self._slot_map[layer_idx]

    def reset(self):
        for s in self.states:
            s[:] = 0
        for c in self.conv_states:
            c[:] = 0
        if self.device is not None:
            for d_s in self.d_states:
                self.device.memset(d_s, 0, self.state_size * 4)
            for d_c in self.d_conv_states:
                self.device.memset(d_c, 0, self.conv_state_size * 2)


class LayerWeights:
    """Device-side weights for one transformer layer."""

    def __init__(self):
        self.layer_type = "full_attention"

        # Full attention weights (FP16, device ptrs)
        self.q_weight = 0       # [Q_dim, hidden]
        self.q_gate_weight = 0  # [Q_dim, hidden] (output gate)
        self.k_weight = 0       # [KV_dim, hidden]
        self.v_weight = 0       # [KV_dim, hidden]
        self.o_weight = 0       # [hidden, Q_dim]
        self.q_norm = 0         # [head_dim]
        self.k_norm = 0         # [head_dim]

        # Linear attention weights — large projections on GPU, small ops on CPU/GPU
        self.la_in_proj_qkv = 0     # GPU ptr [10240, 5120]
        self.la_in_proj_a = 0       # GPU ptr [48, 5120]
        self.la_in_proj_b = 0       # GPU ptr [48, 5120]
        self.la_in_proj_z = 0       # GPU ptr [6144, 5120]
        self.la_conv1d = None       # CPU [10240, 1, 4]
        self.la_A_log = None        # CPU [48]
        self.la_dt_bias = None      # CPU [48]
        self.la_norm = None         # CPU [128]
        self.la_out_proj = 0        # GPU ptr [5120, 6144]
        # GPU versions of small weights (for DeltaNet kernel)
        self.d_la_conv_weight = 0   # GPU ptr [10240, 4] FP32
        self.d_la_A_log = 0         # GPU ptr [48] FP32
        self.d_la_dt_bias = 0       # GPU ptr [48] FP32
        self.d_la_norm = 0          # GPU ptr [128] FP32

        # FFN weights (INT4 GPTQ, device ptrs)
        self.gate_qweight = 0
        self.gate_scales = 0
        self.gate_zeros = 0
        self.up_qweight = 0
        self.up_scales = 0
        self.up_zeros = 0
        self.down_qweight = 0
        self.down_scales = 0
        self.down_zeros = 0

        # RMSNorm (FP16, device ptrs)
        self.attn_norm = 0
        self.ffn_norm = 0


class InferenceEngine:
    """Inference engine for Qwen 3.5 27B on MI50.

    Supports tensor parallelism: pass tp_size > 1 to shard heads and FFN.
    """

    def __init__(self, config: QwenConfig, device_id: int = 0,
                 max_seq_len: int = 2048,
                 tp_size: int = 1, tp_rank: int = 0):
        self.config = config
        self.device = GPUDevice(device_id)
        self.kernels = KernelCache(self.device)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self._tp_group = None  # Set by TPInferenceEngine
        self._tp_peers = None  # Set by register_tp_peers()

        # Compute local (per-GPU) dimensions for TP
        self.local_num_attention_heads = config.num_attention_heads // tp_size
        self.local_num_kv_heads = config.num_key_value_heads // tp_size
        self.local_linear_num_v_heads = config.linear_num_value_heads // tp_size
        self.local_linear_num_k_heads = config.linear_num_key_heads // tp_size
        self.local_intermediate_size = config.intermediate_size // tp_size

        self.kv_cache = KVCache(config, max_seq_len, self.device,
                                tp_size=tp_size)
        self.deltanet_state = DeltaNetState(config, self.device,
                                            tp_size=tp_size)
        self._deltanet_gpu = False
        self.layers: List[LayerWeights] = []

        self.d_final_norm = 0
        self.d_lm_head = 0
        self.lm_head_vocab = 0

        self._init_rope_tables(max_seq_len)
        self._alloc_scratch()
        self._init_deltanet_gpu()
        self._init_qknorm_rope()
        self._init_gemv_v2()
        self._init_gemm_prefill()
        self._init_streams()

        # Launch counting infrastructure: tracks kernel launches per layer.
        # Call reset_launch_counters() before a decode step, then read
        # self._layer_launch_counts[layer_idx] for per-layer counts.
        self._layer_launch_counts: dict = {}  # layer_idx -> int
        self._count_launches: bool = False    # enable via count_launches ctx mgr

    def register_tp_peers(self, peers: list, tp_group):
        """Register peer engines and TP group for allreduce coordination.

        peers: list of InferenceEngine instances (index = tp_rank, includes self)
        tp_group: TensorParallelGroup instance
        """
        self._tp_peers = peers
        self._tp_group = tp_group

    def _init_deltanet_gpu(self):
        """Compile and load the DeltaNet v3 GPU kernel (required — no CPU fallback).

        For TP, compiles with adjusted NUM_V_HEADS/NUM_K_HEADS defines.
        Raises RuntimeError if the kernel cannot be loaded.
        """
        self._deltanet_v3 = False
        self._deltanet_v2 = False
        # Compile flags for TP-aware head counts
        dn_flags = None
        dn_suffix = ""
        if self.tp_size > 1:
            dn_flags = [
                f"-DNUM_V_HEADS={self.local_linear_num_v_heads}",
                f"-DNUM_K_HEADS={self.local_linear_num_k_heads}",
            ]
            dn_suffix = f"_tp{self.tp_size}"

        hip_v3 = HIP_DIR / "deltanet_v3.hip"
        if not hip_v3.exists():
            raise RuntimeError(
                f"DeltaNet v3 kernel source not found at {hip_v3}. "
                "This kernel is required — the CPU fallback has been removed."
            )

        # Raises RuntimeError on compilation failure — intentional, no silent fallback
        self.kernels.get_hip("deltanet_decode_v3", "deltanet_v3",
                             extra_flags=dn_flags, hsaco_suffix=dn_suffix)
        self.kernels.get_hip("deltanet_conv_shift_v3", "deltanet_v3",
                             extra_flags=dn_flags, hsaco_suffix=dn_suffix)
        self._deltanet_gpu = True
        self._deltanet_v2 = True
        self._deltanet_v3 = True
        print(f"DeltaNet v3 GPU kernel loaded (v_heads={self.local_linear_num_v_heads})")

    def _init_qknorm_rope(self):
        """Load the fused QK-norm + RoPE kernel.

        The fused kernel replaces 4 separate launches:
          batched_rmsnorm(Q) + batched_rmsnorm(K) + rope(Q) + rope(K)
        with 2 launches (one for Q heads, one for K heads), each performing
        per-head RMSNorm and partial RoPE in a single pass.

        Raises RuntimeError if the kernel cannot be loaded — no silent fallback.
        """
        self._qknorm_rope_fused = False
        hip_path = HIP_DIR / "qknorm_rope.hip"
        if not hip_path.exists():
            raise RuntimeError(
                f"Fused QK-norm+RoPE kernel source not found at {hip_path}. "
                "This kernel is required — no silent fallback."
            )
        # Raises RuntimeError on compilation failure — no silent fallback
        self.kernels.get_hip("qknorm_rope_fused", "qknorm_rope")
        self._qknorm_rope_fused = True
        print("Fused QK-norm+RoPE kernel loaded")

    def _init_gemv_v2(self):
        """Try to compile optimized GEMV kernels (coalesced INT4, vectorized FP16)."""
        self._gemv_int4_v2 = False
        self._gemv_fp16_v2 = False
        self._gemv_int4_k_splits = 16  # Split K for more parallelism
        try:
            hip_path = HIP_DIR / "gemv_int4_v2.hip"
            if hip_path.exists():
                self.kernels.get_hip("gemv_int4_v2_direct", "gemv_int4_v2")
                self.kernels.get_hip("gemv_int4_v2_fused", "gemv_int4_v2")
                self._gemv_int4_v2 = True
                max_n = max(self.local_intermediate_size, self.config.hidden_size)
                # Persistent FP32 accumulator buffer (zeroed once at init, reset by kernel)
                self.d_gemv_fp32 = self.device.malloc(max_n * 4)
                self.device.memset(self.d_gemv_fp32, 0, max_n * 4)
                # Persistent done-counter buffer (uint32, zeroed once at init, reset by kernel)
                self.d_gemv_done = self.device.malloc(max_n * 4)
                self.device.memset(self.d_gemv_done, 0, max_n * 4)
                # Second FP32 accumulator for dual (gate+up) path
                # Also uses d_gemv_done for the done-counter; both persistent + init-zeroed
                self.d_gemv_fp32_2 = self.device.malloc(max_n * 4)
                self.device.memset(self.d_gemv_fp32_2, 0, max_n * 4)
                print("GEMV INT4 v2 (coalesced + fused split-K) loaded")
        except Exception as e:
            print(f"GEMV INT4 v2 failed: {e}")
        self._gemv_int4_dual = False
        self._gemv_int4_dual_fused = False
        try:
            hip_path = HIP_DIR / "gemv_int4_dual.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_dual_splitk", "gemv_int4_dual")
                self.kernels.get_hip("dual_fp32_to_silu_fp16", "gemv_int4_dual")
                self.kernels.get_hip("gemv_int4_dual_fused", "gemv_int4_dual")
                self._gemv_int4_dual = True
                self._gemv_int4_dual_fused = True
                print("GEMV INT4 dual (fused gate+up+silu, no-memset) loaded")
        except Exception as e:
            print(f"GEMV INT4 dual failed: {e}")
        try:
            hip_path = HIP_DIR / "gemv_fp16_v2.hip"
            if hip_path.exists():
                self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2")
                self._gemv_fp16_v2 = True
                print("GEMV FP16 v2 (vectorized) loaded")
        except Exception as e:
            print(f"GEMV FP16 v2 failed: {e}")

    def _init_gemm_prefill(self):
        """Try to compile GEMM kernels for batched prefill projections."""
        self._gemm_fp16_prefill = False
        self._gemm_int4_prefill = False
        try:
            hip_path = HIP_DIR / "gemm_fp16_prefill.hip"
            if hip_path.exists():
                self.kernels.get_hip("gemm_fp16_prefill", "gemm_fp16_prefill")
                self._gemm_fp16_prefill = True
                print("GEMM FP16 prefill kernel loaded")
        except Exception as e:
            print(f"GEMM FP16 prefill failed: {e}, using per-token GEMV fallback")
        try:
            hip_path = HIP_DIR / "gemm_int4_prefill_hip.hip"
            if hip_path.exists():
                self.kernels.get_hip("gemm_int4_prefill_hip", "gemm_int4_prefill_hip")
                self._gemm_int4_prefill = True
                print("GEMM INT4 prefill kernel loaded")
        except Exception as e:
            print(f"GEMM INT4 prefill failed: {e}, using per-token GEMV fallback")

    def _init_streams(self):
        """Create two HIP streams for concurrent Q+Qgate and K+V GEMV projections.

        Stream 1 (_stream_q):  Q+Qgate fused GEMV (independent of K/V)
        Stream 2 (_stream_kv): K+V fused GEMV (independent of Q/Qgate)

        Both read d_normed which is read-only during these GEMVs — no race.
        Both are synchronized before the downstream QK-norm (which reads d_q, d_k).
        """
        try:
            self._stream_q = self.device.create_stream()
            self._stream_kv = self.device.create_stream()
            self._streams_ready = True
            print("HIP streams created for Q/KV concurrency")
        except Exception as e:
            print(f"Stream creation failed: {e}, using default stream")
            self._stream_q = 0
            self._stream_kv = 0
            self._streams_ready = False

    def reset_launch_counters(self):
        """Reset per-layer kernel launch counters.

        Call before a decode step to measure launches per layer.
        Enable counting with: self._count_launches = True
        """
        self._layer_launch_counts = {}
        self._current_count_layer = -1

    def _record_launch(self, layer_idx: int):
        """Increment launch count for a layer (only when counting is active)."""
        if self._count_launches:
            if layer_idx not in self._layer_launch_counts:
                self._layer_launch_counts[layer_idx] = 0
            self._layer_launch_counts[layer_idx] += 1

    def get_layer_launch_count(self, layer_idx: int) -> int:
        """Return total kernel launches recorded for a given layer."""
        return self._layer_launch_counts.get(layer_idx, 0)

    def _init_rope_tables(self, max_seq_len: int):
        """Precompute RoPE cos/sin tables.

        Qwen 3.5 uses partial_rotary_factor=0.25, so only 64 of 256 dims get RoPE.
        """
        cfg = self.config
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)  # 64
        half_rotary = rotary_dim // 2  # 32

        freqs = 1.0 / (cfg.rope_theta **
                        (np.arange(0, half_rotary, dtype=np.float32) * 2.0 / rotary_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        cos_tab = np.cos(angles).astype(np.float16)
        sin_tab = np.sin(angles).astype(np.float16)

        self.rotary_dim = rotary_dim
        self.d_cos = self.device.malloc(cos_tab.nbytes)
        self.d_sin = self.device.malloc(sin_tab.nbytes)
        self.device.upload(self.d_cos, cos_tab.tobytes())
        self.device.upload(self.d_sin, sin_tab.tobytes())

    def _alloc_scratch(self):
        """Allocate reusable scratch buffers (sizes adjusted for TP)."""
        h = self.config.hidden_size
        inter = self.local_intermediate_size
        cfg = self.config

        # Local full attention dims (per-GPU)
        self.q_dim = self.local_num_attention_heads * cfg.head_dim
        self.kv_dim = self.local_num_kv_heads * cfg.head_dim

        # Scratch for decode (single token)
        self.d_hidden = self.device.malloc(h * 2)
        self.d_hidden2 = self.device.malloc(h * 2)
        self.d_normed = self.device.malloc(h * 2)

        # Full attention scratch — fused Q+Q_gate and K+V for fewer GEMV launches
        self.d_q_fused = self.device.malloc(self.q_dim * 2 * 2)  # [Q, Q_gate] contiguous
        self.d_q = self.d_q_fused
        self.d_q_gate = self.d_q_fused + self.q_dim * 2
        self.d_kv_fused = self.device.malloc(self.kv_dim * 2 * 2)  # [K, V] contiguous
        self.d_k = self.d_kv_fused
        self.d_v = self.d_kv_fused + self.kv_dim * 2
        self.d_attn_out = self.device.malloc(self.q_dim * 2)
        self.d_attn_gated = self.device.malloc(self.q_dim * 2)
        self.d_proj_out = self.device.malloc(h * 2)

        # Linear attention scratch — local dims for TP
        la_qkv_dim = (self.local_linear_num_k_heads * cfg.linear_key_head_dim * 2 +
                       self.local_linear_num_v_heads * cfg.linear_value_head_dim)
        la_z_dim = self.local_linear_num_v_heads * cfg.linear_value_head_dim
        la_dt_dim = self.local_linear_num_v_heads
        self.la_qkv_dim = la_qkv_dim
        self.la_z_dim = la_z_dim
        self.la_dt_dim = la_dt_dim

        # Pack into one contiguous buffer: [qkv | a | b | z]
        la_total = la_qkv_dim + la_dt_dim + la_dt_dim + la_z_dim
        self.la_total_dim = la_total
        self.d_la_packed = self.device.malloc(la_total * 2)
        self.d_la_qkv = self.d_la_packed
        self.d_la_dt = self.d_la_packed + la_qkv_dim * 2
        self.d_la_b = self.d_la_dt + la_dt_dim * 2
        self.d_la_z = self.d_la_b + la_dt_dim * 2
        self.d_la_out = self.device.malloc(la_z_dim * 2)

        # FFN scratch (local intermediate size for TP)
        self.d_ffn_gate = self.device.malloc(inter * 2)
        self.d_ffn_up = self.device.malloc(inter * 2)
        self.d_ffn_out = self.device.malloc(h * 2)

    def load_final_norm(self, weight: np.ndarray):
        """Upload final RMSNorm weight to GPU."""
        if weight.dtype != np.float16:
            weight = weight.astype(np.float16)
        self.d_final_norm = self.device.malloc(weight.nbytes)
        self.device.upload(self.d_final_norm, weight.tobytes())

    def load_lm_head(self, weight: np.ndarray):
        """Upload LM head weight to GPU for fast logit computation.

        weight: [vocab_size, hidden_dim] FP16
        """
        if weight.dtype != np.float16:
            weight = weight.astype(np.float16)
        self.lm_head_vocab = weight.shape[0]
        self.d_lm_head = self.device.malloc(weight.nbytes)
        self.device.upload(self.d_lm_head, weight.tobytes())
        # Allocate logits buffer
        self.d_logits = self.device.malloc(self.lm_head_vocab * 2)

    def compute_logits(self, hidden: np.ndarray = None) -> np.ndarray:
        """Compute logits using GPU GEMV.

        If hidden is provided, uploads it first. Otherwise uses d_hidden2
        (output of final norm).

        Returns: [vocab_size] FP32 logits
        """
        if hidden is not None:
            src = self.d_hidden2
            self.device.upload(src, hidden.tobytes())
        else:
            src = self.d_hidden2

        self._launch_gemv_fp16(self.d_logits, src, self.d_lm_head,
                                self.config.hidden_size, self.lm_head_vocab)

        logits_f16 = np.frombuffer(
            self.device.download(self.d_logits, self.lm_head_vocab * 2),
            dtype=np.float16)
        return logits_f16.astype(np.float32)

    def _shard_weights_for_tp(self, weights: dict):
        """Shard weight arrays in-place for tensor parallelism.

        Called before the upload loop so only sharded data hits GPU memory.
        Modifies the weights dict in-place.
        """
        cfg = self.config
        tp = self.tp_size
        rank = self.tp_rank
        h = cfg.hidden_size
        layer_type = weights.get('layer_type', 'full_attention')

        if layer_type == 'full_attention':
            head_dim = cfg.head_dim

            # Column-parallel: shard Q, Q_gate by heads
            for key, num_heads in [('q_weight', cfg.num_attention_heads),
                                    ('q_gate_weight', cfg.num_attention_heads)]:
                if key in weights and isinstance(weights[key], np.ndarray):
                    w = weights[key].reshape(num_heads, head_dim, h)
                    local_heads = num_heads // tp
                    weights[key] = w[rank * local_heads:(rank + 1) * local_heads].reshape(-1, h).copy()

            # Column-parallel: shard K, V by KV heads
            for key in ['k_weight', 'v_weight']:
                if key in weights and isinstance(weights[key], np.ndarray):
                    num_heads = cfg.num_key_value_heads
                    w = weights[key].reshape(num_heads, head_dim, h)
                    local_heads = num_heads // tp
                    weights[key] = w[rank * local_heads:(rank + 1) * local_heads].reshape(-1, h).copy()

            # Row-parallel: shard O projection input dim by heads
            if 'o_weight' in weights and isinstance(weights['o_weight'], np.ndarray):
                full_q_dim = cfg.num_attention_heads * head_dim
                local_q_dim = self.q_dim
                start = rank * local_q_dim
                weights['o_weight'] = weights['o_weight'][:, start:start + local_q_dim].copy()

        elif layer_type == 'linear_attention':
            k_head_dim = cfg.linear_key_head_dim
            v_head_dim = cfg.linear_value_head_dim
            local_k_heads = self.local_linear_num_k_heads
            local_v_heads = self.local_linear_num_v_heads
            q_dim_full = cfg.linear_num_key_heads * k_head_dim
            k_dim_full = q_dim_full

            # Shard la_in_proj_qkv by heads
            if 'la_in_proj_qkv' in weights and isinstance(weights['la_in_proj_qkv'], np.ndarray):
                full_qkv = weights['la_in_proj_qkv']
                q_part = full_qkv[:q_dim_full].reshape(cfg.linear_num_key_heads, k_head_dim, h)
                k_part = full_qkv[q_dim_full:q_dim_full + k_dim_full].reshape(
                    cfg.linear_num_key_heads, k_head_dim, h)
                v_part = full_qkv[q_dim_full + k_dim_full:].reshape(
                    cfg.linear_num_value_heads, v_head_dim, h)
                weights['la_in_proj_qkv'] = np.vstack([
                    q_part[rank * local_k_heads:(rank + 1) * local_k_heads].reshape(-1, h),
                    k_part[rank * local_k_heads:(rank + 1) * local_k_heads].reshape(-1, h),
                    v_part[rank * local_v_heads:(rank + 1) * local_v_heads].reshape(-1, h),
                ]).copy()

            # Shard a/b projections by v_heads
            for key in ['la_in_proj_a', 'la_in_proj_b']:
                if key in weights and isinstance(weights[key], np.ndarray):
                    weights[key] = weights[key][rank * local_v_heads:
                                                 (rank + 1) * local_v_heads].copy()

            # Shard z projection by v_heads
            if 'la_in_proj_z' in weights and isinstance(weights['la_in_proj_z'], np.ndarray):
                z = weights['la_in_proj_z']
                z_reshaped = z.reshape(cfg.linear_num_value_heads, v_head_dim, h)
                weights['la_in_proj_z'] = z_reshaped[
                    rank * local_v_heads:(rank + 1) * local_v_heads].reshape(-1, h).copy()

            # Row-parallel: shard out_proj input dim
            if 'la_out_proj' in weights and isinstance(weights['la_out_proj'], np.ndarray):
                local_v_dim = local_v_heads * v_head_dim
                start = rank * local_v_dim
                weights['la_out_proj'] = weights['la_out_proj'][:, start:start + local_v_dim].copy()

            # Shard conv1d by heads
            if 'la_conv1d' in weights and isinstance(weights['la_conv1d'], np.ndarray):
                full_conv = weights['la_conv1d']
                conv_q = full_conv[:q_dim_full].reshape(cfg.linear_num_key_heads, k_head_dim, 1, 4)
                conv_k = full_conv[q_dim_full:q_dim_full + k_dim_full].reshape(
                    cfg.linear_num_key_heads, k_head_dim, 1, 4)
                conv_v = full_conv[q_dim_full + k_dim_full:].reshape(
                    cfg.linear_num_value_heads, v_head_dim, 1, 4)
                weights['la_conv1d'] = np.concatenate([
                    conv_q[rank * local_k_heads:(rank + 1) * local_k_heads].reshape(-1, 1, 4),
                    conv_k[rank * local_k_heads:(rank + 1) * local_k_heads].reshape(-1, 1, 4),
                    conv_v[rank * local_v_heads:(rank + 1) * local_v_heads].reshape(-1, 1, 4),
                ], axis=0)

            # Shard A_log, dt_bias by v_heads
            for key in ['la_A_log', 'la_dt_bias']:
                if key in weights and isinstance(weights[key], np.ndarray):
                    weights[key] = weights[key][rank * local_v_heads:
                                                 (rank + 1) * local_v_heads].copy()

        # FFN: sharding handled by weight_loader (tp_size/tp_rank params)
        # But if weights come unsharded, shard here
        for proj, shard_dim in [('gate', 1), ('up', 1), ('down', 0)]:
            qw_key = f'{proj}_qweight'
            sc_key = f'{proj}_scales'
            zr_key = f'{proj}_zeros'
            if qw_key in weights and isinstance(weights[qw_key], np.ndarray):
                qw = weights[qw_key]
                sc = weights[sc_key]
                zr = weights[zr_key]
                if shard_dim == 1:
                    N = qw.shape[1]
                    shard_n = N // tp
                    start = rank * shard_n
                    weights[qw_key] = qw[:, start:start + shard_n].copy()
                    weights[sc_key] = sc[:, start:start + shard_n].copy()
                    weights[zr_key] = zr[:, start:start + shard_n].copy()
                else:  # shard_dim == 0
                    K8 = qw.shape[0]
                    Kg = sc.shape[0]
                    qw_shard = K8 // tp
                    sg_shard = Kg // tp
                    qw_start = rank * qw_shard
                    sg_start = rank * sg_shard
                    weights[qw_key] = qw[qw_start:qw_start + qw_shard, :].copy()
                    weights[sc_key] = sc[sg_start:sg_start + sg_shard, :].copy()
                    weights[zr_key] = zr[sg_start:sg_start + sg_shard, :].copy()

    def load_layer_weights(self, layer_idx: int, weights: dict):
        """Upload one layer's weights to GPU.

        Handles both full attention (FP16 → GPU) and linear attention (keep on CPU).
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(LayerWeights())

        lw = self.layers[layer_idx]
        lw.layer_type = weights.get('layer_type', 'full_attention')

        def upload(data: np.ndarray) -> int:
            if data.dtype != np.float16 and data.dtype not in (np.int32, np.uint32, np.float32):
                data = data.astype(np.float16)
            ptr = self.device.malloc(data.nbytes)
            self.device.upload(ptr, data.tobytes())
            return ptr

        # TP: shard numpy arrays BEFORE uploading to avoid peak memory doubling
        if self.tp_size > 1:
            self._shard_weights_for_tp(weights)

        # Linear attention: large projections go to GPU, small ops stay on CPU
        la_cpu_keys = ['la_conv1d', 'la_A_log', 'la_dt_bias', 'la_norm']

        for key, arr in weights.items():
            if key == 'layer_type':
                continue
            if isinstance(arr, np.ndarray):
                if key in la_cpu_keys:
                    setattr(lw, key, arr)  # Keep on CPU
                else:
                    setattr(lw, key, upload(arr))  # Upload to GPU
            elif isinstance(arr, (int, float)):
                setattr(lw, key, arr)

        # Fuse Q+Q_gate and K+V weights for full attention (fewer GEMV launches)
        if lw.layer_type == 'full_attention':
            h = self.config.hidden_size
            q_dim = self.q_dim
            kv_dim = self.kv_dim
            if 'q_weight' in weights and 'q_gate_weight' in weights:
                q_fused = np.vstack([weights['q_weight'], weights['q_gate_weight']])
                lw.q_fused_weight = upload(q_fused)
                if lw.q_weight:
                    self.device.free(lw.q_weight)
                if lw.q_gate_weight:
                    self.device.free(lw.q_gate_weight)
                lw.q_weight = lw.q_fused_weight
                lw.q_gate_weight = lw.q_fused_weight + q_dim * h * 2
            if 'k_weight' in weights and 'v_weight' in weights:
                kv_fused = np.vstack([weights['k_weight'], weights['v_weight']])
                lw.kv_fused_weight = upload(kv_fused)
                if lw.k_weight:
                    self.device.free(lw.k_weight)
                if lw.v_weight:
                    self.device.free(lw.v_weight)
                lw.k_weight = lw.kv_fused_weight
                lw.v_weight = lw.kv_fused_weight + kv_dim * h * 2

        # Fuse linear attention input projections (4 GEMV → 1)
        if lw.layer_type == 'linear_attention':
            la_keys = ['la_in_proj_qkv', 'la_in_proj_a', 'la_in_proj_b', 'la_in_proj_z']
            if all(k in weights for k in la_keys):
                la_fused = np.vstack([weights[k] for k in la_keys])
                lw.la_in_proj_fused = upload(la_fused)
                for k in la_keys:
                    ptr = getattr(lw, k, 0)
                    if ptr and isinstance(ptr, int):
                        self.device.free(ptr)
                off = 0
                lw.la_in_proj_qkv = lw.la_in_proj_fused + off
                off += self.la_qkv_dim * self.config.hidden_size * 2
                lw.la_in_proj_a = lw.la_in_proj_fused + off
                off += self.la_dt_dim * self.config.hidden_size * 2
                lw.la_in_proj_b = lw.la_in_proj_fused + off
                off += self.la_dt_dim * self.config.hidden_size * 2
                lw.la_in_proj_z = lw.la_in_proj_fused + off

        # Upload small DeltaNet weights to GPU for the GPU kernel path
        if lw.layer_type == 'linear_attention':
            if lw.la_conv1d is not None:
                conv_w = lw.la_conv1d.astype(np.float32).squeeze(1)
                lw.d_la_conv_weight = upload(conv_w)
            if lw.la_A_log is not None:
                lw.d_la_A_log = upload(lw.la_A_log.astype(np.float32))
            if lw.la_dt_bias is not None:
                lw.d_la_dt_bias = upload(lw.la_dt_bias.astype(np.float32))
            if lw.la_norm is not None:
                lw.d_la_norm = upload(lw.la_norm.astype(np.float32))

    def decode_step(self, token_embedding: np.ndarray, position: int) -> np.ndarray:
        """Run one decode step through all layers (single-GPU only).

        For multi-GPU TP, use TPInferenceEngine.decode_step() instead.

        Args:
            token_embedding: [hidden_dim] FP16 array
            position: current sequence position

        Returns:
            [hidden_dim] FP16 hidden state (after final RMSNorm)
        """
        if self.tp_size > 1:
            raise RuntimeError(
                "decode_step() not supported with tp_size > 1. "
                "Use TPInferenceEngine.decode_step() instead.")

        h = self.config.hidden_size
        cfg = self.config

        self.device.upload(self.d_hidden, token_embedding.tobytes())
        self._active_layer_idx = -1  # Track current layer for launch counting

        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.layers[layer_idx]
            self._active_layer_idx = layer_idx

            # Pre-attention RMSNorm
            self._launch_rmsnorm(self.d_normed, self.d_hidden, lw.attn_norm, h)

            if lw.layer_type == 'full_attention':
                self._decode_full_attention(layer_idx, lw, position)
            else:
                # Always use GPU path — CPU fallback has been removed.
                # _init_deltanet_gpu() raises RuntimeError at init if kernel unavailable.
                self._decode_linear_attention_gpu(layer_idx, lw, position)

            # FFN block: fused residual-add + RMSNorm at pre-FFN position.
            # With residual-epilogue in out_proj GEMV (single-GPU path):
            #   - out_proj already added proj_out + hidden → d_hidden is up-to-date
            #   - Just run plain RMSNorm for the pre-FFN norm
            # With TP (tp_size > 1):
            #   - TPInferenceEngine allreduces d_proj_out first, then residual+norm
            #   - Fallback: skip_rmsnorm handles residual+norm together
            if self.tp_size <= 1:
                self._launch_rmsnorm(self.d_normed, self.d_hidden, lw.ffn_norm, h)
            else:
                # TP path: residual add + norm is handled by TPInferenceEngine
                # but decode_step single-call is not used for tp_size > 1
                self._launch_skip_rmsnorm(self.d_normed, self.d_hidden, self.d_proj_out,
                                           lw.ffn_norm, h)

            # Gate + up INT4 GEMV projections (column-parallel: local N)
            if self._gemv_int4_dual:
                self._launch_ffn_gate_up_silu(self.d_ffn_gate, self.d_normed,
                                               lw, h, self.local_intermediate_size)
            else:
                self._launch_gemv_int4(self.d_ffn_gate, self.d_normed,
                                        lw.gate_qweight, lw.gate_scales,
                                        lw.gate_zeros, h, self.local_intermediate_size)
                self._launch_gemv_int4(self.d_ffn_up, self.d_normed,
                                        lw.up_qweight, lw.up_scales,
                                        lw.up_zeros, h, self.local_intermediate_size)
                self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                         self.d_ffn_gate, self.local_intermediate_size)

            # Down projection (row-parallel with TP: local K, full N=hidden).
            # For single-GPU (tp_size <= 1): use residual epilogue to fuse
            # residual-add into the GEMV kernel, writing directly to d_hidden.
            # This eliminates the separate residual-add launch after down_proj.
            # For TP: TPInferenceEngine handles allreduce+residual externally.
            if self.tp_size <= 1:
                self._launch_gemv_int4(self.d_hidden, self.d_ffn_gate,
                                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                                        self.local_intermediate_size, h,
                                        residual=self.d_hidden)
            else:
                self._launch_gemv_int4(self.d_ffn_out, self.d_ffn_gate,
                                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                                        self.local_intermediate_size, h)

        self.kv_cache.advance()

        # Final RMSNorm
        if self.d_final_norm:
            self._launch_rmsnorm(self.d_hidden2, self.d_hidden, self.d_final_norm, h)
            return np.frombuffer(self.device.download(self.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(self.device.download(self.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_full_attention(self, layer_idx: int, lw: LayerWeights,
                                position: int):
        """Full attention decode step: Q/K/V projections, RoPE, decode_attn_256, gate.

        With TP: each GPU handles local_num_attention_heads Q heads and
        local_num_kv_heads KV heads. O projection is row-parallel with allreduce.
        """
        cfg = self.config
        h = cfg.hidden_size

        # Concurrent Q+Qgate and K+V projections on independent streams.
        # Both GEMVs read d_normed (read-only) — no race condition.
        # Stream 1: Q+Qgate fused GEMV → d_q_fused ([Q, Q_gate])
        # Stream 2: K+V fused GEMV   → d_kv_fused ([K, V])
        self._launch_gemv_fp16(self.d_q_fused, self.d_normed, lw.q_fused_weight,
                                h, 2 * self.q_dim, stream=self._stream_q)

        # Fused K+V projection: normed_hidden → [K, V] [2*kv_dim]
        self._launch_gemv_fp16(self.d_kv_fused, self.d_normed, lw.kv_fused_weight,
                                h, 2 * self.kv_dim, stream=self._stream_kv)

        # Synchronize both streams before QK-norm (which reads d_q and d_k).
        if self._streams_ready:
            self.device.hip.stream_synchronize(self._stream_q)
            self.device.hip.stream_synchronize(self._stream_kv)

        # Fused QK-norm + RoPE (per-head RMSNorm + partial RoPE in one launch each)
        # Replaces 4 separate launches: qk_norm(Q) + qk_norm(K) + rope(Q) + rope(K)
        self._launch_qknorm_rope(self.d_q, lw.q_norm, position,
                                  self.local_num_attention_heads, cfg.head_dim)
        self._launch_qknorm_rope(self.d_k, lw.k_norm, position,
                                  self.local_num_kv_heads, cfg.head_dim)

        # Update KV cache (GPU-to-GPU copy, no host roundtrip)
        self.kv_cache.append_kv_gpu(layer_idx, self.d_k, self.d_v)

        # Decode attention (head_dim=256 variant, local heads only)
        self._launch_decode_attn_256(
            self.d_attn_out, self.d_q,
            self.kv_cache.layer_k_ptr(layer_idx),
            self.kv_cache.layer_v_ptr(layer_idx),
            self.kv_cache.current_len + 1)

        # Apply sigmoid gate: attn_out *= sigmoid(gate)
        self._launch_sigmoid_mul(self.d_attn_out, self.d_q_gate, self.q_dim)

        # Output projection (row-parallel with TP: local K=q_dim, full N=hidden).
        # For single-GPU (tp_size <= 1): use residual epilogue to fuse residual-add
        # into the GEMV kernel, writing d_hidden += proj_result directly.
        # This eliminates the need for a separate residual_add; the subsequent
        # pre-FFN norm in decode_step becomes plain rmsnorm (not skip_rmsnorm).
        # For TP: write to d_proj_out; TPInferenceEngine allreduces first, then
        # decode_step falls through to skip_rmsnorm to handle residual+norm.
        if self.tp_size <= 1:
            self._launch_gemv_fp16(self.d_hidden, self.d_attn_out, lw.o_weight,
                                    self.q_dim, h, residual=self.d_hidden)
        else:
            self._launch_gemv_fp16(self.d_proj_out, self.d_attn_out, lw.o_weight,
                                    self.q_dim, h)

        # Note: for tp_size <= 1, d_hidden now contains (old_hidden + out_proj_result).
        # decode_step will run plain rmsnorm on d_hidden for the pre-FFN position.
        # For TP (tp_size > 1): TPInferenceEngine handles allreduce + residual + norm.

    def _decode_linear_attention(self, layer_idx: int, lw: LayerWeights,
                                  position: int):
        """Linear attention (Gated DeltaNet) decode step.

        GPU GEMV for large projections, CPU for DeltaNet recurrence.
        """
        cfg = self.config
        h = cfg.hidden_size

        slot = self.deltanet_state.get_slot(layer_idx, cfg)

        # 1. Fused input projections via single GPU GEMV
        self._launch_gemv_fp16(self.d_la_packed, self.d_normed,
                                lw.la_in_proj_fused, h, self.la_total_dim)

        # Single download of all linear attention outputs (contiguous buffer)
        packed = np.frombuffer(self.device.download(self.d_la_packed, self.la_total_dim * 2),
                                dtype=np.float16).copy()
        off = 0
        qkv = packed[off:off + self.la_qkv_dim].astype(np.float32); off += self.la_qkv_dim
        a_input = packed[off:off + self.la_dt_dim].astype(np.float32); off += self.la_dt_dim
        b_input = packed[off:off + self.la_dt_dim].astype(np.float32); off += self.la_dt_dim
        z = packed[off:off + self.la_z_dim].astype(np.float32)

        # 2. Causal conv1d on qkv (z, b, a bypass conv)
        conv_state = self.deltanet_state.conv_states[slot]
        qkv_f16 = qkv.astype(np.float16)

        # Form conv input BEFORE updating state (state has history, append current)
        conv_input = np.concatenate([conv_state, qkv_f16[:, None]], axis=1)
        # Update state for next step
        if conv_state.shape[1] > 0:
            conv_state[:, :-1] = conv_state[:, 1:]
            conv_state[:, -1] = qkv_f16

        conv_weight = lw.la_conv1d.astype(np.float32).squeeze(1)  # [10240, 4]
        qkv_conv = np.sum(conv_input.astype(np.float32) * conv_weight, axis=1)

        # SiLU after conv
        qkv_conv = qkv_conv * (1.0 / (1.0 + np.exp(-qkv_conv)))

        # 3. Split into Q, K, V
        q_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim  # 2048
        k_dim = q_dim  # 2048
        v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim  # 6144

        q = qkv_conv[:q_dim].reshape(cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = qkv_conv[q_dim:q_dim + k_dim].reshape(cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = qkv_conv[q_dim + k_dim:].reshape(cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        # 4. L2 normalize Q and K, scale Q
        def l2norm(x, eps=1e-6):
            norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
            return x / norm

        q = l2norm(q) / np.sqrt(cfg.linear_key_head_dim)  # scale by 1/sqrt(d_k)
        k = l2norm(k)

        # 5. Expand Q, K from 16 to 48 heads (repeat_interleave)
        heads_per_q = cfg.linear_num_value_heads // cfg.linear_num_key_heads  # 3
        q = np.repeat(q, heads_per_q, axis=0)  # [48, 128]
        k = np.repeat(k, heads_per_q, axis=0)  # [48, 128]

        # 6. Compute decay g and beta
        # g = -exp(A_log) * softplus(a + dt_bias)
        A = np.exp(lw.la_A_log.astype(np.float32))  # [48]
        softplus_a = np.log1p(np.exp(a_input + lw.la_dt_bias.astype(np.float32)))  # [48]
        g = -A * softplus_a  # [48], negative values
        decay = np.exp(np.clip(g, -20, 0))  # [48], in (0, 1]

        # beta = sigmoid(b)
        beta = 1.0 / (1.0 + np.exp(-b_input))  # [48]

        # 7. Gated DeltaNet recurrence — vectorized across heads where possible
        # state: [48, k_dim, v_dim]
        state = self.deltanet_state.states[slot]

        # Decay all heads at once: state[h] *= decay[h]
        state *= decay[:, None, None]

        # Read memory: kv_mem[h] = state[h].T @ k[h] for all heads
        # state: [48, 128, 128], k: [48, 128] → kv_mem: [48, 128]
        kv_mem = np.einsum('hkv,hk->hv', state, k)

        # Delta with beta gate
        delta = (v - kv_mem) * beta[:, None]  # [48, 128]

        # Write: state += outer(k, delta) for all heads
        state += np.einsum('hk,hv->hkv', k, delta)

        # Read output: output[h] = state[h].T @ q[h]
        output = np.einsum('hkv,hk->hv', state, q)  # [48, 128]

        # 8. Gated RMSNorm: RMSNorm(output) * SiLU(z) — vectorized
        norm_weight = lw.la_norm.astype(np.float32)  # [v_head_dim=128]
        z_reshaped = z.reshape(cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        rms = np.sqrt(np.mean(output ** 2, axis=-1, keepdims=True) + 1e-6)
        output = (output / rms) * norm_weight[None, :]
        z_silu = z_reshaped * (1.0 / (1.0 + np.exp(-z_reshaped)))
        output = output * z_silu

        y = output.reshape(-1)

        # 9. Output projection via GPU GEMV
        y_f16 = y.astype(np.float16)
        self.device.upload(self.d_la_out, y_f16.tobytes())
        self._launch_gemv_fp16(self.d_proj_out, self.d_la_out, lw.la_out_proj,
                                self.la_z_dim, h)

        # Residual (TP allreduce handled externally by TPInferenceEngine)
        if self.tp_size <= 1:
            self._launch_residual_add(self.d_hidden, self.d_proj_out, h)

    def _decode_linear_attention_gpu(self, layer_idx: int, lw: LayerWeights,
                                      position: int):
        """GPU-accelerated DeltaNet decode step.

        With TP: each GPU handles local heads. out_proj is row-parallel with allreduce.
        """
        cfg = self.config
        h = cfg.hidden_size
        slot = self.deltanet_state.get_slot(layer_idx, cfg)

        # 1. Fused input projections via single GPU GEMV (column-parallel: local N)
        self._launch_gemv_fp16(self.d_la_packed, self.d_normed,
                                lw.la_in_proj_fused, h, self.la_total_dim)

        d_conv = self.deltanet_state.d_conv_states[slot]
        d_state = self.deltanet_state.d_states[slot]

        if self._deltanet_v3:
            # 2a. DeltaNet v3: parallel kq_dot, 256-thread pass1
            dn_suffix = f"_tp{self.tp_size}" if self.tp_size > 1 else ""
            func = self.kernels.get_hip("deltanet_decode_v3", "deltanet_v3",
                                        hsaco_suffix=dn_suffix)
            params = [
                ctypes.c_uint64(self.d_la_qkv),
                ctypes.c_uint64(self.d_la_dt),
                ctypes.c_uint64(self.d_la_b),
                ctypes.c_uint64(self.d_la_z),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint64(lw.d_la_conv_weight),
                ctypes.c_uint64(lw.d_la_A_log),
                ctypes.c_uint64(lw.d_la_dt_bias),
                ctypes.c_uint64(lw.d_la_norm),
                ctypes.c_uint64(d_state),
                ctypes.c_uint64(self.d_la_out),
            ]
            self.device.launch(func, (self.local_linear_num_v_heads, 1, 1),
                               (256, 1, 1), params, shared_mem=8192)

            conv_dim = self.deltanet_state.conv_dim
            shift_func = self.kernels.get_hip("deltanet_conv_shift_v3", "deltanet_v3",
                                              hsaco_suffix=dn_suffix)
            shift_params = [
                ctypes.c_uint64(self.d_la_qkv),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint32(conv_dim),
            ]
            grid_x = (conv_dim + 255) // 256
            self.device.launch(shift_func, (grid_x, 1, 1), (256, 1, 1), shift_params)
        elif self._deltanet_v2:
            # 2a. DeltaNet v2: main kernel (conv_state read-only)
            func = self.kernels.get_hip("deltanet_decode_v2", "deltanet_v2")
            params = [
                ctypes.c_uint64(self.d_la_qkv),
                ctypes.c_uint64(self.d_la_dt),
                ctypes.c_uint64(self.d_la_b),
                ctypes.c_uint64(self.d_la_z),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint64(lw.d_la_conv_weight),
                ctypes.c_uint64(lw.d_la_A_log),
                ctypes.c_uint64(lw.d_la_dt_bias),
                ctypes.c_uint64(lw.d_la_norm),
                ctypes.c_uint64(d_state),
                ctypes.c_uint64(self.d_la_out),
            ]
            self.device.launch(func, (self.local_linear_num_v_heads, 1, 1),
                               (256, 1, 1), params, shared_mem=4096)

            conv_dim = self.deltanet_state.conv_dim
            shift_func = self.kernels.get_hip("deltanet_conv_shift", "deltanet_v2")
            shift_params = [
                ctypes.c_uint64(self.d_la_qkv),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint32(conv_dim),
            ]
            grid_x = (conv_dim + 255) // 256
            self.device.launch(shift_func, (grid_x, 1, 1), (256, 1, 1), shift_params)
        else:
            # 2. DeltaNet v1 fallback
            func = self.kernels.get_hip("deltanet_decode", "deltanet")
            params = [
                ctypes.c_uint64(self.d_la_qkv),
                ctypes.c_uint64(self.d_la_dt),
                ctypes.c_uint64(self.d_la_b),
                ctypes.c_uint64(self.d_la_z),
                ctypes.c_uint64(d_conv),
                ctypes.c_uint64(lw.d_la_conv_weight),
                ctypes.c_uint64(lw.d_la_A_log),
                ctypes.c_uint64(lw.d_la_dt_bias),
                ctypes.c_uint64(lw.d_la_norm),
                ctypes.c_uint64(d_state),
                ctypes.c_uint64(self.d_la_out),
            ]
            self.device.launch(func, (self.local_linear_num_v_heads, 1, 1),
                               (256, 1, 1), params, shared_mem=4096)

        # 3. Output projection (row-parallel with TP: local K=la_z_dim, full N=hidden).
        # For single-GPU (tp_size <= 1): use residual epilogue to fuse residual-add
        # into the GEMV kernel, writing d_hidden += la_out_proj result directly.
        # For TP: write to d_proj_out; TPInferenceEngine allreduces first.
        if self.tp_size <= 1:
            self._launch_gemv_fp16(self.d_hidden, self.d_la_out, lw.la_out_proj,
                                    self.la_z_dim, h, residual=self.d_hidden)
        else:
            self._launch_gemv_fp16(self.d_proj_out, self.d_la_out, lw.la_out_proj,
                                    self.la_z_dim, h)

        # Note: for tp_size <= 1, d_hidden now contains (old_hidden + la_out_proj_result).
        # decode_step will run plain rmsnorm on d_hidden for the pre-FFN position.
        # For TP (tp_size > 1): TPInferenceEngine handles allreduce + residual + norm.

    # --- Kernel launchers ---

    def _launch_rmsnorm(self, dst, src, weight, dim):
        func = self.kernels.get_hip("rmsnorm_v2", "elementwise_v2")
        params = [
            ctypes.c_uint64(dst),
            ctypes.c_uint64(src),
            ctypes.c_uint64(weight),
            ctypes.c_uint32(dim),
            ctypes.c_float(self.config.rms_norm_eps),
        ]
        self.device.launch(func, (1, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_skip_rmsnorm(self, dst, hidden, residual, weight, dim):
        """Fused: hidden += residual; dst = rmsnorm(hidden) * weight."""
        func = self.kernels.get_hip("skip_rmsnorm_v2", "elementwise_v2")
        params = [
            ctypes.c_uint64(dst),
            ctypes.c_uint64(hidden),
            ctypes.c_uint64(residual),
            ctypes.c_uint64(weight),
            ctypes.c_uint32(dim),
            ctypes.c_float(self.config.rms_norm_eps),
        ]
        self.device.launch(func, (1, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_qk_norm(self, x_ptr, norm_weight, num_heads, head_dim):
        """Apply RMSNorm per-head to Q or K vector using batched GPU kernel.

        Always uses the batched kernel (1 launch for all heads).
        The silent try/except fallback to per-head launches has been removed
        to ensure batched path is always taken (VAL-DLR-010).
        """
        eps_bits = struct.unpack('<I', struct.pack('<f', self.config.rms_norm_eps))[0]
        func = self.kernels.get_hip("batched_rmsnorm_fp16", "batched_rmsnorm")
        params = [
            ctypes.c_uint64(x_ptr),
            ctypes.c_uint64(norm_weight),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(eps_bits),
        ]
        self.device.launch(func, (num_heads, 1, 1), (256, 1, 1), params)

    def _launch_qknorm_rope(self, x_ptr, norm_weight, position, num_heads, head_dim):
        """Fused per-head RMSNorm + partial RoPE in a single kernel launch.

        Replaces two separate calls: _launch_qk_norm + _launch_rope.
        The fused kernel computes RMSNorm then applies partial RoPE to the
        first rotary_dim elements of each head in one pass.

        Args:
            x_ptr:       GPU pointer to [num_heads, head_dim] FP16 (modified in-place)
            norm_weight: GPU pointer to [head_dim] FP16 norm weights
            position:    Sequence position (for RoPE cos/sin table lookup)
            num_heads:   Number of heads (Q or K)
            head_dim:    Dimension per head (e.g. 256)
        """
        func = self.kernels.get_hip("qknorm_rope_fused", "qknorm_rope")
        half_rotary = self.rotary_dim // 2
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables
        params = [
            ctypes.c_uint64(x_ptr),
            ctypes.c_uint64(norm_weight),
            ctypes.c_uint64(self.d_cos + cos_offset),
            ctypes.c_uint64(self.d_sin + cos_offset),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(half_rotary),
            ctypes.c_float(self.config.rms_norm_eps),
        ]
        # One block per head; 256 threads per block
        self.device.launch(func, (num_heads, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_gemv_fp16(self, dst, src, weight, K, N, residual=0, stream=0):
        """FP16 GEMV: dst[N] = weight[N, K] * src[K].

        Optional residual epilogue: if residual != 0, dst[i] += residual[i]
        This fuses the residual-add into the GEMV kernel, eliminating a
        separate residual_add launch (used for out_proj with residual=d_hidden).

        Optional stream: if non-zero, launch on that HIP stream (for concurrency).
        """
        if self._gemv_fp16_v2:
            func = self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2")
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(weight),
                ctypes.c_uint64(dst),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint64(residual),  # optional residual ptr (0 = null = no residual)
            ]
            # 4 rows per WG (one per wavefront)
            grid_x = (N + 3) // 4
            self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params, stream=stream)
        else:
            func = self.kernels.get("gemv_fp16", "gemv_fp16")
            params = [
                ctypes.c_uint64(src),      # x ptr
                ctypes.c_uint64(weight),   # W ptr
                ctypes.c_uint64(dst),      # out ptr
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
            ]
            self.device.launch(func, (N, 1, 1), (256, 1, 1), params, shared_mem=1024)
            # Fallback: apply residual separately if requested
            if residual:
                self._launch_residual_add(dst, residual, N)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_gemm_fp16(self, dst, src, weight, M, N, K):
        """FP16 GEMM: dst[M, N] = src[M, K] @ weight[N, K]^T.

        Used for batched projections during prefill.
        """
        func = self.kernels.get_hip("gemm_fp16_prefill", "gemm_fp16_prefill")
        params = [
            ctypes.c_uint64(src),
            ctypes.c_uint64(weight),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
        ]
        grid_x = (N + 63) // 64
        grid_y = (M + 63) // 64
        self.device.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)

    def _launch_gemm_int4(self, dst, src, qweight, scales, zeros, M, N, K):
        """INT4 GEMM: dst[M, N] = src[M, K] @ dequant(qweight[K/8, N])."""
        func = self.kernels.get_hip("gemm_int4_prefill_hip", "gemm_int4_prefill_hip")
        params = [
            ctypes.c_uint64(src),
            ctypes.c_uint64(qweight),
            ctypes.c_uint64(scales),
            ctypes.c_uint64(zeros),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(M),
            ctypes.c_uint32(N),
            ctypes.c_uint32(K),
            ctypes.c_uint32(self.config.group_size),
        ]
        grid_x = (N + 63) // 64
        grid_y = (M + 63) // 64
        self.device.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)

    def _launch_gemv_int4(self, dst, src, qweight, scales, zeros, K, N, residual=0):
        if self._gemv_int4_v2:
            grid_x = (N + 255) // 256
            k_splits = self._gemv_int4_k_splits
            # Single fused launch: no separate memset or fp32_to_fp16 needed.
            # The kernel initializes C_fp32 internally and writes FP16 output directly.
            # Optional residual epilogue: if residual != 0, dst[i] += residual[i]
            func = self.kernels.get_hip("gemv_int4_v2_fused", "gemv_int4_v2")
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(qweight),
                ctypes.c_uint64(scales),
                ctypes.c_uint64(zeros),
                ctypes.c_uint64(self.d_gemv_fp32),
                ctypes.c_uint64(self.d_gemv_done),
                ctypes.c_uint64(dst),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint32(self.config.group_size),
                ctypes.c_uint32(k_splits),
                ctypes.c_uint64(residual),  # optional residual ptr (0 = null = no residual)
            ]
            self.device.launch(func, (grid_x, k_splits, 1), (256, 1, 1), params)
        else:
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
            # Fallback: apply residual separately if requested
            if residual:
                self._launch_residual_add(dst, residual, N)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_ffn_gate_up_silu(self, dst, src, lw, K, N):
        """Fused gate+up INT4 GEMV + SiLU: dst = silu(gate(x)) * up(x).

        Uses the fully fused kernel (gemv_int4_dual_fused) when available:
        no separate memset or fp32_to_fp16 calls needed. The kernel initializes
        both FP32 accumulator buffers internally on the first-completing split tile,
        and writes FP16 output with SiLU in the last-completing split tile.

        Saves 2 memset launches + 1 convert+silu launch vs the 3-kernel path.
        """
        grid_x = (N + 255) // 256
        k_splits = self._gemv_int4_k_splits

        if self._gemv_int4_dual_fused:
            # Fully fused path: single launch, no memset needed.
            # Persistent buffers d_gemv_fp32, d_gemv_fp32_2, d_gemv_done are
            # zeroed once at init and reset by the kernel after each call.
            func = self.kernels.get_hip("gemv_int4_dual_fused", "gemv_int4_dual")
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(lw.gate_qweight),
                ctypes.c_uint64(lw.gate_scales),
                ctypes.c_uint64(lw.gate_zeros),
                ctypes.c_uint64(lw.up_qweight),
                ctypes.c_uint64(lw.up_scales),
                ctypes.c_uint64(lw.up_zeros),
                ctypes.c_uint64(self.d_gemv_fp32),
                ctypes.c_uint64(self.d_gemv_fp32_2),
                ctypes.c_uint64(self.d_gemv_done),
                ctypes.c_uint64(dst),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint32(self.config.group_size),
                ctypes.c_uint32(k_splits),
            ]
            self.device.launch(func, (grid_x, k_splits, 1), (256, 1, 1), params)
        else:
            # Fallback 3-launch path: memset + splitk + convert+silu
            # Zero both FP32 accumulators
            self.device.memset(self.d_gemv_fp32, 0, N * 4)
            self.device.memset(self.d_gemv_fp32_2, 0, N * 4)

            # Fused gate+up split-K
            func = self.kernels.get_hip("gemv_int4_dual_splitk", "gemv_int4_dual")
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(lw.gate_qweight),
                ctypes.c_uint64(lw.gate_scales),
                ctypes.c_uint64(lw.gate_zeros),
                ctypes.c_uint64(lw.up_qweight),
                ctypes.c_uint64(lw.up_scales),
                ctypes.c_uint64(lw.up_zeros),
                ctypes.c_uint64(self.d_gemv_fp32),
                ctypes.c_uint64(self.d_gemv_fp32_2),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint32(self.config.group_size),
            ]
            self.device.launch(func, (grid_x, k_splits, 1), (256, 1, 1), params)
            self._record_launch(getattr(self, '_active_layer_idx', -1))  # splitk launch

            # Convert FP32 → FP16 with fused SiLU
            func2 = self.kernels.get_hip("dual_fp32_to_silu_fp16", "gemv_int4_dual")
            params2 = [
                ctypes.c_uint64(self.d_gemv_fp32),
                ctypes.c_uint64(self.d_gemv_fp32_2),
                ctypes.c_uint64(dst),
                ctypes.c_uint32(N),
            ]
            self.device.launch(func2, (grid_x, 1, 1), (256, 1, 1), params2)
            self._record_launch(getattr(self, '_active_layer_idx', -1))  # convert+silu
            return  # fallback path: 2 launches recorded (splitk + convert)

        self._record_launch(getattr(self, '_active_layer_idx', -1))  # fused single launch

    def _launch_rope(self, x_ptr, position, num_heads, head_dim):
        """Apply partial RoPE (only first rotary_dim dims of each head)."""
        func = self.kernels.get("rope_fp16", "rope")
        half_rotary = self.rotary_dim // 2
        cos_offset = position * half_rotary * 2
        params = [
            ctypes.c_uint64(x_ptr),
            ctypes.c_uint64(self.d_cos + cos_offset),
            ctypes.c_uint64(self.d_sin + cos_offset),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(num_heads),
        ]
        self.device.launch(func, (1, num_heads, 1), (half_rotary, 1, 1), params)

    def _launch_decode_attn_256(self, out, q, k_cache, v_cache, seq_len):
        """Launch head_dim=256 decode attention using flash attention kernel.

        Uses flash_attn_256 with num_q_rows=1 and causal=0 (single Q row
        attends to all KV positions). TP: uses local head counts.
        """
        func = self.kernels.get_hip("flash_attn_256_fp16", "flash_attn_256")
        params = [
            ctypes.c_uint64(q),
            ctypes.c_uint64(k_cache),
            ctypes.c_uint64(v_cache),
            ctypes.c_uint64(out),
            ctypes.c_uint32(seq_len),       # kv_seq_len
            ctypes.c_uint32(1),              # num_q_rows = 1 for decode
            ctypes.c_uint32(self.local_num_attention_heads),
            ctypes.c_uint32(self.local_num_kv_heads),
            ctypes.c_uint32(0),              # non-causal (single Q row)
        ]
        self.device.launch(func, (self.local_num_attention_heads, 1, 1),
                           (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))


    def _launch_sigmoid_mul(self, attn_out, gate, n):
        """Apply sigmoid gate on GPU: attn_out[i] *= sigmoid(gate[i])."""
        func = self.kernels.get_hip("sigmoid_mul_fp16", "sigmoid_mul")
        params = [
            ctypes.c_uint64(attn_out),
            ctypes.c_uint64(gate),
            ctypes.c_uint32(n),
        ]
        grid_x = (n + 255) // 256
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_silu_fused(self, gate, up, out, n):
        func = self.kernels.get_hip("silu_fused_v2", "elementwise_v2")
        params = [
            ctypes.c_uint64(gate),
            ctypes.c_uint64(up),
            ctypes.c_uint32(n),
        ]
        grid_x = (n + 511) // 512  # each thread handles 2 elements
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)

    def _launch_residual_add(self, dst, src, n):
        func = self.kernels.get_hip("residual_add_v2", "elementwise_v2")
        params = [
            ctypes.c_uint64(dst),
            ctypes.c_uint64(src),
            ctypes.c_uint32(n),
        ]
        grid_x = (n + 511) // 512  # each thread handles 2 elements
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)

    def _alloc_prefill_scratch(self, seq_len: int):
        """Allocate GPU scratch buffers for prefill."""
        if hasattr(self, '_pf_seq_len') and self._pf_seq_len >= seq_len:
            return

        # Free old buffers if reallocating
        if hasattr(self, '_pf_seq_len'):
            for attr in ['d_pf_hidden', 'd_pf_normed', 'd_pf_q', 'd_pf_q_gate',
                          'd_pf_k', 'd_pf_v', 'd_pf_attn_out',
                          'd_pf_ffn_gate', 'd_pf_ffn_up', 'd_pf_ffn_out']:
                ptr = getattr(self, attr, 0)
                if ptr:
                    self.device.free(ptr)

        h = self.config.hidden_size
        q_dim = self.q_dim
        kv_dim = self.kv_dim

        self.d_pf_hidden = self.device.malloc(seq_len * h * 2)
        self.d_pf_normed = self.device.malloc(seq_len * h * 2)
        self.d_pf_q = self.device.malloc(seq_len * q_dim * 2)
        self.d_pf_q_gate = self.device.malloc(seq_len * q_dim * 2)
        self.d_pf_k = self.device.malloc(seq_len * kv_dim * 2)
        self.d_pf_v = self.device.malloc(seq_len * kv_dim * 2)
        self.d_pf_attn_out = self.device.malloc(seq_len * q_dim * 2)
        inter = self.config.intermediate_size
        self.d_pf_ffn_gate = self.device.malloc(seq_len * inter * 2)
        self.d_pf_ffn_up = self.device.malloc(seq_len * inter * 2)
        self.d_pf_ffn_out = self.device.malloc(seq_len * h * 2)
        self._pf_seq_len = seq_len

    def _prefill_full_attention(self, layer_idx: int, lw: LayerWeights,
                                 seq_len: int):
        """Full attention prefill using flash_attn_256 + batched GEMM."""
        cfg = self.config
        h = cfg.hidden_size
        q_dim = self.q_dim
        kv_dim = self.kv_dim

        if self._gemm_fp16_prefill and seq_len >= 32:
            # Batched path: RMSNorm per-token into contiguous buffer, then GEMM

            # Per-token RMSNorm into d_pf_normed
            for t in range(seq_len):
                self._launch_rmsnorm(self.d_pf_normed + t * h * 2,
                                      self.d_pf_hidden + t * h * 2,
                                      lw.attn_norm, h)

            # Batched GEMM projections: normed[M, K] @ weight[N, K]^T
            self._launch_gemm_fp16(self.d_pf_q, self.d_pf_normed,
                                    lw.q_weight, seq_len, q_dim, h)
            self._launch_gemm_fp16(self.d_pf_q_gate, self.d_pf_normed,
                                    lw.q_gate_weight, seq_len, q_dim, h)
            self._launch_gemm_fp16(self.d_pf_k, self.d_pf_normed,
                                    lw.k_weight, seq_len, kv_dim, h)
            self._launch_gemm_fp16(self.d_pf_v, self.d_pf_normed,
                                    lw.v_weight, seq_len, kv_dim, h)
        else:
            # Fallback: per-token GEMV
            for t in range(seq_len):
                t_h = t * h * 2
                t_q = t * q_dim * 2
                t_kv = t * kv_dim * 2

                self._launch_rmsnorm(self.d_normed, self.d_pf_hidden + t_h,
                                      lw.attn_norm, h)
                self._launch_gemv_fp16(self.d_pf_q + t_q, self.d_normed,
                                        lw.q_weight, h, q_dim)
                self._launch_gemv_fp16(self.d_pf_q_gate + t_q, self.d_normed,
                                        lw.q_gate_weight, h, q_dim)
                self._launch_gemv_fp16(self.d_pf_k + t_kv, self.d_normed,
                                        lw.k_weight, h, kv_dim)
                self._launch_gemv_fp16(self.d_pf_v + t_kv, self.d_normed,
                                        lw.v_weight, h, kv_dim)

        # Per-token fused QK-norm + RoPE: RMSNorm + RoPE in one launch per token per Q/K
        # Treat each token×head as one unit; _launch_qknorm_rope handles the RoPE table lookup
        for t in range(seq_len):
            self._launch_qknorm_rope(self.d_pf_q + t * q_dim * 2, lw.q_norm, t,
                                      cfg.num_attention_heads, cfg.head_dim)
            self._launch_qknorm_rope(self.d_pf_k + t * kv_dim * 2, lw.k_norm, t,
                                      cfg.num_key_value_heads, cfg.head_dim)

        # Bulk write K/V to cache (contiguous D2D copy)
        kv_layer_k = self.kv_cache.layer_k_ptr(layer_idx)
        kv_layer_v = self.kv_cache.layer_v_ptr(layer_idx)
        kv_bytes = seq_len * kv_dim * 2
        self.device.memcpy_d2d(kv_layer_k, self.d_pf_k, kv_bytes)
        self.device.memcpy_d2d(kv_layer_v, self.d_pf_v, kv_bytes)

        # Flash attention (causal)
        func = self.kernels.get_hip("flash_attn_256_fp16", "flash_attn_256")
        params = [
            ctypes.c_uint64(self.d_pf_q),
            ctypes.c_uint64(kv_layer_k),
            ctypes.c_uint64(kv_layer_v),
            ctypes.c_uint64(self.d_pf_attn_out),
            ctypes.c_uint32(seq_len),     # kv_seq_len
            ctypes.c_uint32(seq_len),     # num_q_rows
            ctypes.c_uint32(cfg.num_attention_heads),
            ctypes.c_uint32(cfg.num_key_value_heads),
            ctypes.c_uint32(1),           # causal
        ]
        grid_x = cfg.num_attention_heads
        grid_y = (seq_len + 3) // 4
        self.device.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)

        # Sigmoid gate on full buffer (elementwise, can batch)
        self._launch_sigmoid_mul(self.d_pf_attn_out, self.d_pf_q_gate,
                                  seq_len * q_dim)

        if self._gemm_fp16_prefill and seq_len >= 32:
            # Batched output projection + per-token residual
            # d_pf_normed reused as temporary for O projection output
            self._launch_gemm_fp16(self.d_pf_normed, self.d_pf_attn_out,
                                    lw.o_weight, seq_len, h, q_dim)
            # Batch residual add
            self._launch_residual_add(self.d_pf_hidden, self.d_pf_normed,
                                       seq_len * h)
        else:
            # Per-token output projection + residual
            for t in range(seq_len):
                t_q = t * q_dim * 2
                t_h = t * h * 2
                self._launch_gemv_fp16(self.d_proj_out, self.d_pf_attn_out + t_q,
                                        lw.o_weight, q_dim, h)
                self._launch_residual_add(self.d_pf_hidden + t_h,
                                           self.d_proj_out, h)

    def _prefill_linear_attention(self, layer_idx: int, lw: LayerWeights,
                                   seq_len: int):
        """Linear attention prefill (sequential — DeltaNet is recurrent)."""
        cfg = self.config
        h = cfg.hidden_size

        for t in range(seq_len):
            t_off = t * h * 2

            # Copy this token's hidden to d_hidden for existing methods
            self.device.memcpy_d2d(self.d_hidden, self.d_pf_hidden + t_off,
                                    h * 2)

            self._launch_rmsnorm(self.d_normed, self.d_hidden, lw.attn_norm, h)

            # Always use GPU path — CPU fallback has been removed.
            self._decode_linear_attention_gpu(layer_idx, lw, t)

            # Copy updated hidden back to prefill buffer
            self.device.memcpy_d2d(self.d_pf_hidden + t_off, self.d_hidden,
                                    h * 2)

    def prefill_step(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Process multiple tokens using flash attention for prefill.

        Uses flash_attn_256 for full attention layers (single kernel launch
        instead of O(n) decode attention launches). Linear attention layers
        are processed sequentially (DeltaNet recurrence is inherently serial).

        Note: TP prefill not yet implemented. Use sequential decode for TP.

        Args:
            token_embeddings: [seq_len, hidden_dim] FP16

        Returns:
            [hidden_dim] FP16 hidden state after final norm
        """
        seq_len = token_embeddings.shape[0]
        h = self.config.hidden_size
        cfg = self.config

        self._alloc_prefill_scratch(seq_len)
        self.device.upload(self.d_pf_hidden, token_embeddings.tobytes())

        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.layers[layer_idx]

            if lw.layer_type == 'full_attention':
                self._prefill_full_attention(layer_idx, lw, seq_len)
            else:
                self._prefill_linear_attention(layer_idx, lw, seq_len)

            # FFN block
            if self._gemm_int4_prefill and seq_len >= 32:
                # Batched FFN: RMSNorm all tokens, GEMM gate/up, SiLU, GEMM down
                inter = cfg.intermediate_size

                # Per-token RMSNorm into d_pf_normed
                for t in range(seq_len):
                    self._launch_rmsnorm(self.d_pf_normed + t * h * 2,
                                          self.d_pf_hidden + t * h * 2,
                                          lw.ffn_norm, h)

                # Allocate batched FFN scratch on first use
                if not hasattr(self, 'd_pf_ffn_gate'):
                    self.d_pf_ffn_gate = self.device.malloc(seq_len * inter * 2)
                    self.d_pf_ffn_up = self.device.malloc(seq_len * inter * 2)
                    self.d_pf_ffn_out = self.device.malloc(seq_len * h * 2)

                # Batched gate/up projections
                self._launch_gemm_int4(self.d_pf_ffn_gate, self.d_pf_normed,
                                        lw.gate_qweight, lw.gate_scales,
                                        lw.gate_zeros, seq_len, inter, h)
                self._launch_gemm_int4(self.d_pf_ffn_up, self.d_pf_normed,
                                        lw.up_qweight, lw.up_scales,
                                        lw.up_zeros, seq_len, inter, h)

                # Batched SiLU: gate = silu(gate) * up for all tokens
                self._launch_silu_fused(self.d_pf_ffn_gate, self.d_pf_ffn_up,
                                         self.d_pf_ffn_gate, seq_len * inter)

                # Batched down projection
                self._launch_gemm_int4(self.d_pf_ffn_out, self.d_pf_ffn_gate,
                                        lw.down_qweight, lw.down_scales,
                                        lw.down_zeros, seq_len, h, inter)

                # Batched residual add
                self._launch_residual_add(self.d_pf_hidden, self.d_pf_ffn_out,
                                           seq_len * h)
            else:
                # Per-token FFN fallback (uses fused gate+up GEMV)
                for t in range(seq_len):
                    t_off = t * h * 2
                    self._launch_rmsnorm(self.d_normed, self.d_pf_hidden + t_off,
                                          lw.ffn_norm, h)
                    self._launch_gemv_int4(self.d_ffn_gate, self.d_normed,
                                            lw.gate_qweight, lw.gate_scales,
                                            lw.gate_zeros, h, cfg.intermediate_size)
                    self._launch_gemv_int4(self.d_ffn_up, self.d_normed,
                                            lw.up_qweight, lw.up_scales,
                                            lw.up_zeros, h, cfg.intermediate_size)
                    self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                             self.d_ffn_gate, cfg.intermediate_size)
                    self._launch_gemv_int4(self.d_ffn_out, self.d_ffn_gate,
                                            lw.down_qweight, lw.down_scales,
                                            lw.down_zeros, cfg.intermediate_size, h)
                    self._launch_residual_add(self.d_pf_hidden + t_off,
                                               self.d_ffn_out, h)

        # Set KV cache length
        self.kv_cache.current_len = seq_len

        # Final RMSNorm on last token
        last_off = (seq_len - 1) * h * 2
        if self.d_final_norm:
            self._launch_rmsnorm(self.d_hidden2, self.d_pf_hidden + last_off,
                                  self.d_final_norm, h)
            return np.frombuffer(self.device.download(self.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(
            self.device.download(self.d_pf_hidden + last_off, h * 2),
            dtype=np.float16)

    def cleanup(self):
        self.kv_cache.cleanup()
        # Destroy streams if they were created
        if getattr(self, '_streams_ready', False):
            try:
                self.device.hip.stream_destroy(self._stream_q)
                self.device.hip.stream_destroy(self._stream_kv)
            except Exception:
                pass
        self.device.cleanup()
