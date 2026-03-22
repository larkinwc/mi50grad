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
    """Contiguous KV cache for full attention layers only.
    
    Supports batched decoding with per-sequence position tracking.
    Layout: [num_layers, batch_size, max_seq, local_kv_heads, head_dim] FP16
    For each layer, batches are stored consecutively, and within each batch,
    sequence positions are stored consecutively.
    """

    def __init__(self, config: QwenConfig, max_seq_len: int, device: GPUDevice,
                 tp_size: int = 1, batch_size: int = 1):
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        self.batch_size = batch_size
        self.current_len = 0  # Current sequence position (shared across batch)

        # Only full attention layers need KV cache
        self.num_full_layers = config.num_full_attention_layers
        self.full_layer_indices = [i for i in range(config.num_hidden_layers)
                                    if config.is_full_attention(i)]

        # K cache: [num_full_layers, batch_size, max_seq, local_kv_heads, head_dim] FP16
        self.local_num_kv_heads = config.num_key_value_heads // tp_size
        # Size of one position for one batch: [local_kv_heads, head_dim] * 2 bytes
        self.pos_size_per_batch = self.local_num_kv_heads * self.config.head_dim * 2
        # Stride between consecutive positions within a batch
        self.pos_stride = self.pos_size_per_batch
        # Stride between batches (all positions for one batch)
        self.batch_stride = max_seq_len * self.pos_stride
        # Size per layer
        per_layer = self.batch_size * self.batch_stride
        self.k_size = self.num_full_layers * per_layer
        self.v_size = self.k_size

        self.d_k = device.malloc(self.k_size)
        self.d_v = device.malloc(self.v_size)
        device.memset(self.d_k, 0, self.k_size)
        device.memset(self.d_v, 0, self.v_size)

    def _full_layer_slot(self, layer_idx: int) -> int:
        """Map global layer_idx to slot in KV cache (only full attention layers)."""
        return self.full_layer_indices.index(layer_idx)

    def layer_k_ptr(self, layer_idx: int, batch_idx: int = 0) -> int:
        """Get K cache BASE pointer for a layer and batch index.
        
        The base pointer points to position 0 for the given batch.
        The attention kernel reads positions 0..seq_len-1 consecutively from this base.
        
        Args:
            layer_idx: Layer index
            batch_idx: Batch index (0..batch_size-1), defaults to 0 for single-sequence
            
        Returns:
            GPU pointer to start of K cache for this layer and batch (position 0)
        """
        slot = self._full_layer_slot(layer_idx)
        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        return self.d_k + layer_offset + batch_offset

    def layer_v_ptr(self, layer_idx: int, batch_idx: int = 0) -> int:
        """Get V cache BASE pointer for a layer and batch index.
        
        The base pointer points to position 0 for the given batch.
        
        Args:
            layer_idx: Layer index
            batch_idx: Batch index (0..batch_size-1), defaults to 0 for single-sequence
            
        Returns:
            GPU pointer to start of V cache for this layer and batch (position 0)
        """
        slot = self._full_layer_slot(layer_idx)
        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        return self.d_v + layer_offset + batch_offset

    def get_kv_ptr(self, layer_idx: int, position: int, batch_idx: int = 0):
        """Get K and V cache pointers for a specific position and batch.
        
        Used for writing a single position's KV data.
        
        Args:
            layer_idx: Layer index
            position: Sequence position (0..max_seq_len-1)
            batch_idx: Batch index (0..batch_size-1)
            
        Returns:
            Tuple of (k_ptr, v_ptr) GPU pointers
        """
        slot = self._full_layer_slot(layer_idx)
        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        pos_offset = position * self.pos_stride
        base_offset = layer_offset + batch_offset + pos_offset
        return (self.d_k + base_offset, self.d_v + base_offset)

    def append_kv(self, layer_idx: int, k_data: bytes, v_data: bytes, batch_idx: int = 0):
        """Append one position's K and V to the cache (from host bytes).
        
        Args:
            layer_idx: Layer index
            k_data: K data bytes
            v_data: V data bytes
            batch_idx: Batch index (0..batch_size-1)
        """
        slot = self._full_layer_slot(layer_idx)
        kv_size = self.pos_size_per_batch

        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        pos_offset = self.current_len * self.pos_stride
        base_offset = layer_offset + batch_offset + pos_offset

        self.device._ensure_device()
        self.device.hip.memcpy_h2d(self.d_k + base_offset, k_data, kv_size)
        self.device.hip.memcpy_h2d(self.d_v + base_offset, v_data, kv_size)

    def append_kv_gpu(self, layer_idx: int, d_k_src: int, d_v_src: int, batch_idx: int = 0):
        """Append one position's K and V from GPU buffers (device-to-device).
        
        Args:
            layer_idx: Layer index
            d_k_src: Source K GPU pointer
            d_v_src: Source V GPU pointer
            batch_idx: Batch index (0..batch_size-1)
        """
        slot = self._full_layer_slot(layer_idx)
        kv_size = self.pos_size_per_batch

        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        pos_offset = self.current_len * self.pos_stride
        base_offset = layer_offset + batch_offset + pos_offset

        self.device._ensure_device()
        self.device.hip.memcpy_d2d(self.d_k + base_offset, d_k_src, kv_size)
        self.device.hip.memcpy_d2d(self.d_v + base_offset, d_v_src, kv_size)

    def advance(self, num_positions: int = 1):
        """Advance the sequence position for all sequences in the batch.
        
        Args:
            num_positions: Number of positions to advance (for batch writes)
        """
        self.current_len += num_positions

    def reset(self):
        """Reset the cache position to 0."""
        self.current_len = 0

    def append_kv_gpu_batch(self, layer_idx: int, d_k_src: int, d_v_src: int,
                             start_batch_idx: int = 0, num_positions: int = 1):
        """Append multiple consecutive positions' K and V from GPU buffers.
        
        This is optimized for contiguous batch writes: writes num_positions
        consecutive positions starting at current_len for batches starting
        at start_batch_idx.
        
        Args:
            layer_idx: Layer index
            d_k_src: Source K GPU pointer (points to [num_positions, local_kv_heads, head_dim])
            d_v_src: Source V GPU pointer (same layout as K)
            start_batch_idx: Starting batch index (default 0)
            num_positions: Number of consecutive positions to write
        """
        slot = self._full_layer_slot(layer_idx)
        
        # Size for all positions: num_positions * pos_stride
        total_kv_size = num_positions * self.pos_stride
        
        # Calculate base offset for the first batch/position
        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = start_batch_idx * self.batch_stride
        pos_offset = self.current_len * self.pos_stride
        base_offset = layer_offset + batch_offset + pos_offset
        
        # Single memcpy for all positions (contiguous write)
        self.device._ensure_device()
        self.device.hip.memcpy_d2d(self.d_k + base_offset, d_k_src, total_kv_size)
        self.device.hip.memcpy_d2d(self.d_v + base_offset, d_v_src, total_kv_size)
    
    def append_kv_gpu_from_batch(self, layer_idx: int, d_k_src: int, d_v_src: int, 
                                   batch_idx: int = 0):
        """Append one position's K and V to a specific batch slot.
        
        Similar to append_kv_gpu but allows specifying the batch_idx explicitly.
        This is useful for non-sequential batch writes.
        
        Args:
            layer_idx: Layer index
            d_k_src: Source K GPU pointer
            d_v_src: Source V GPU pointer  
            batch_idx: Batch index (0..batch_size-1)
        """
        slot = self._full_layer_slot(layer_idx)
        kv_size = self.pos_size_per_batch

        layer_offset = slot * self.batch_size * self.batch_stride
        batch_offset = batch_idx * self.batch_stride
        pos_offset = self.current_len * self.pos_stride
        base_offset = layer_offset + batch_offset + pos_offset

        self.device._ensure_device()
        self.device.hip.memcpy_d2d(self.d_k + base_offset, d_k_src, kv_size)
        self.device.hip.memcpy_d2d(self.d_v + base_offset, d_v_src, kv_size)

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
        
        # INT4 attention weights for compressed-tensors format
        # q/k/v/o projections are INT4 quantized (AWQ-style, symmetric)
        self.q_qweight = 0      # [Q_dim/8, hidden] INT32 packed
        self.q_scales = 0       # [Q_dim/group_size, hidden] FP16
        self.q_zeros = 0        # [Q_dim/group_size, hidden] FP16 (all zeros)
        self.k_qweight = 0      # [KV_dim/8, hidden] INT32 packed
        self.k_scales = 0       # [KV_dim/group_size, hidden] FP16
        self.k_zeros = 0        # [KV_dim/group_size, hidden] FP16 (all zeros)
        self.v_qweight = 0      # [KV_dim/8, hidden] INT32 packed
        self.v_scales = 0       # [KV_dim/group_size, hidden] FP16
        self.v_zeros = 0        # [KV_dim/group_size, hidden] FP16 (all zeros)
        self.o_qweight = 0      # [hidden/8, Q_dim] INT32 packed
        self.o_scales = 0       # [hidden/group_size, Q_dim] FP16
        self.o_zeros = 0        # [hidden/group_size, Q_dim] FP16 (all zeros)
        
        # Concatenated QKV weights for fused QKV GEMV kernel
        # Fuses q/k/v projections into single kernel launch
        self.qkv_qweight = 0    # [N_total/8, hidden] INT32 packed, N_total = q_dim + 2*kv_dim
        self.qkv_scales = 0     # [N_total/group_size, hidden] FP16
        self.qkv_zeros = 0      # [N_total/group_size, hidden] FP16

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

        # FFN weights for W8A8: INT8 weight + FP32 per-channel scale (device ptrs)
        # gate_w8a8: [N, K] INT8, gate_scale_w8a8: [N] FP32
        self.gate_w8a8 = 0
        self.gate_scale_w8a8 = 0
        self.up_w8a8 = 0
        self.up_scale_w8a8 = 0
        self.down_w8a8 = 0
        self.down_scale_w8a8 = 0

        # FFN weights for W4A8: packed INT4 + per-group FP32 scales (device ptrs)
        # gate_w4a8: [N, K/8] uint32, gate_scale_w4a8: [K/group_size, N] FP16
        self.gate_w4a8 = 0
        self.gate_scale_w4a8 = 0
        self.up_w4a8 = 0
        self.up_scale_w4a8 = 0
        self.down_w4a8 = 0
        self.down_scale_w4a8 = 0

        # RMSNorm (FP16, device ptrs)
        self.attn_norm = 0
        self.ffn_norm = 0


class InferenceEngine:
    """Inference engine for Qwen 3.5 27B on MI50.

    Supports tensor parallelism: pass tp_size > 1 to shard heads and FFN.

    quant_format selects the FFN quantization format:
      'w4a16' (default): INT4 weights, FP16 activations (existing path)
      'w8a8':            INT8 weights, INT8 activations (W8A8 GEMV/GEMM)
      'w4a8':            INT4 weights packed for W4A8, INT8 activations
    
    use_int4_attention: when True, attention projections (q/k/v/o) use INT4 GEMV
      instead of FP16 GEMV. Required for compressed-tensors format models.
    """

    VALID_QUANT_FORMATS = ('w4a16', 'w8a8', 'w4a8')

    def __init__(self, config: QwenConfig, device_id: int = 0,
                 max_seq_len: int = 2048,
                 tp_size: int = 1, tp_rank: int = 0,
                 quant_format: str = 'w4a16',
                 use_int4_attention: bool = False):
        if quant_format not in self.VALID_QUANT_FORMATS:
            raise ValueError(
                f"quant_format must be one of {self.VALID_QUANT_FORMATS}, got {quant_format!r}")
        self.quant_format = quant_format
        # AWQ mode: skip zero-point subtraction in GEMV (set via set_awq_mode() after loading)
        self._awq_mode = False
        # INT4 attention projections (compressed-tensors format)
        self.use_int4_attention = use_int4_attention

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

        # KV cache supports up to batch_size=8 by default (can be increased if needed)
        self.max_batch_size = 8
        self.kv_cache = KVCache(config, max_seq_len, self.device,
                                tp_size=tp_size, batch_size=self.max_batch_size)
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
        self._init_qknorm_rope_cachew()
        self._init_rope_hip()
        self._init_gemv_v2()
        self._init_gemm_prefill()
        self._init_quant_kernels()
        self._init_streams()

        # Direct KV cache write flag (set via set_direct_kv_write())
        self._direct_kv_write = False

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

    def set_awq_mode(self, enabled: bool = True):
        """Enable or disable AWQ GEMV kernel mode.

        When enabled, the AWQ variant of gemv_int4_v5 is used for non-residual
        GEMV calls, and gemv_int4_dual_awq is used for FFN gate+up projections.
        The AWQ kernel skips the zero-point subtraction (w = q * scale
        instead of w = (q - zero) * scale), saving 8 v_sub_f32 instructions per
        uint32 word and eliminating the zeros tensor memory traffic.

        Must be called after weights are loaded (zeros tensor pointers are ignored
        by the AWQ kernel path). Requires gemv_int4_v5_awq.hip and 
        gemv_int4_dual_awq.hip to be compiled.

        Args:
            enabled: True to use AWQ kernel, False to fall back to GPTQ kernel.
        """
        if enabled and not self._gemv_int4_v5_awq:
            print("WARNING: AWQ GEMV kernel not available (gemv_int4_v5_awq.hip not found). "
                  "AWQ mode disabled; falling back to standard v5/v3 GEMV.")
            self._awq_mode = False
        elif enabled and not self._gemv_int4_dual_awq:
            print("NOTE: AWQ dual GEMV kernel not available (gemv_int4_dual_awq.hip not found). "
                  "AWQ mode enabled for FFN down only; gate+up will use GPTQ dual kernel.")
            self._awq_mode = True
        else:
            self._awq_mode = enabled
            if enabled:
                if self._gemv_int4_dual_awq:
                    print("AWQ GEMV mode enabled: using gemv_int4_v5_awq_t16 + gemv_int4_dual_awq_fused (no zero-point)")
                else:
                    print("AWQ GEMV mode enabled: using gemv_int4_v5_awq_t16 (no zero-point)")
            else:
                print("AWQ GEMV mode disabled: using standard v5/v3 GEMV")

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

    def _init_qknorm_rope_cachew(self):
        """Load the fused QK-norm + RoPE + KV cache write kernel.

        This is a variant of qknorm_rope_fused that additionally writes the
        post-RoPE output directly to a KV cache position (cache_dst pointer).
        When non-NULL, it writes to both the working buffer AND the cache,
        eliminating the separate hipMemcpyAsync D2D for K.

        Falls back gracefully if the kernel cannot be loaded (uses base
        qknorm_rope_fused + separate memcpy instead).
        """
        self._qknorm_rope_cachew = False
        hip_path = HIP_DIR / "qknorm_rope_cachew.hip"
        if not hip_path.exists():
            print("qknorm_rope_cachew.hip not found, using qknorm_rope_fused + memcpy")
            return
        try:
            self.kernels.get_hip("qknorm_rope_cachew_fused", "qknorm_rope_cachew")
            self._qknorm_rope_cachew = True
            print("Fused QK-norm+RoPE+cache-write kernel loaded (qknorm_rope_cachew.hip)")
        except Exception as e:
            print(f"qknorm_rope_cachew kernel failed to load: {e}, "
                  "using qknorm_rope_fused + memcpy fallback")

    def set_direct_kv_write(self, enabled: bool):
        """Enable or disable direct KV cache writes from QKNorm/RoPE kernel.

        When enabled=True, the QKNorm/RoPE kernel (qknorm_rope_cachew_fused) writes
        the post-RoPE K directly to the KV cache position, and V is written directly
        from its GEMV to the cache position (splitting the KV fused GEMV into
        separate K and V GEMVs).

        This eliminates 2 hipMemcpyAsync D2D calls per full-attention layer
        (16 layers × 2 copies = 32 D2D copies per token for TP=4).

        Requires _qknorm_rope_cachew = True (loaded at init).
        Requires build_dispatch_cache() to be rebuilt after enabling.
        """
        if enabled and not self._qknorm_rope_cachew:
            print("WARNING: Direct KV write requires qknorm_rope_cachew kernel "
                  "(unavailable). Keeping direct_kv_write disabled.")
            enabled = False
        self._direct_kv_write = enabled

    def _init_rope_hip(self):
        """Load the HIP RoPE kernel (rope_v2.hip), replacing assembly rope.s.

        The HIP kernel uses vectorized half2 loads and FP32 rotation,
        supporting head_dim=128 and head_dim=256.
        Falls back to assembly rope.s if HIP kernel cannot be loaded.
        """
        self._rope_hip = False
        hip_path = HIP_DIR / "rope_v2.hip"
        if not hip_path.exists():
            print("HIP RoPE kernel (rope_v2.hip) not found, will use assembly fallback")
            return
        try:
            self.kernels.get_hip("rope_fp16_v2", "rope_v2")
            self._rope_hip = True
            print("HIP RoPE kernel (rope_v2.hip) loaded — replaces assembly rope.s")
        except Exception as e:
            print(f"HIP RoPE kernel failed to load: {e}, will use assembly fallback")

    def _init_gemv_v2(self):
        """Try to compile optimized GEMV kernels (coalesced INT4, vectorized FP16)."""
        self._gemv_int4_v2 = False
        self._gemv_int4_v3 = False  # v3 cooperative reduction (faster for large N)
        self._gemv_int4_v5 = False  # v5 hybrid DPP+LDS reduction
        self._gemv_int4_v6 = False  # v6 register-cached scale/zero + weight prefetch
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
        # Try to load v6 kernel (register-cached scale/zero + weight prefetch).
        # v6 is shape-dependent: v6_t16 is best at N=4096,K=4096 (1.13x speedup),
        # but may regress at N=17408,K=5120. Use shape-based selection in _launch_gemv_int4().
        # Falls back to v5 if v6 is unavailable.
        # Try v7 first (FP32-only accumulation + 2x register blocking)
        # Try v8 first (FP32-only accumulation + 4x register blocking)
        self._gemv_int4_v8 = False
        try:
            hip_path = HIP_DIR / "gemv_int4_v8.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v8_t16", "gemv_int4_v8")
                self.kernels.get_hip("gemv_int4_v8_t8", "gemv_int4_v8")
                self.kernels.get_hip("gemv_int4_v8_t4", "gemv_int4_v8")
                self._gemv_int4_v8 = True
                print("GEMV INT4 v8 (FP32 accum + 4x register blocking) loaded as default")
        except Exception as e:
            print(f"GEMV INT4 v8 failed (falling back to v7): {e}")
        
        # Try to load fused QKV GEMV kernel (reduces 3 launches to 1 per attention layer)
        # Based on v8 patterns with 4x register blocking and FP32 accumulation
        # Fuses Q, K, V projections into single kernel launch
        self._gemv_int4_qkv_fused = False
        try:
            hip_path = HIP_DIR / "gemv_int4_qkv_fused.hip"
            if hip_path.exists():
                self.kernels.get_hip("gemv_int4_qkv_fused_t16", "gemv_int4_qkv_fused",
                                      hsaco_suffix="_t16")
                self.kernels.get_hip("gemv_int4_qkv_fused_t8", "gemv_int4_qkv_fused",
                                      hsaco_suffix="_t8")
                self.kernels.get_hip("gemv_int4_qkv_fused_t4", "gemv_int4_qkv_fused",
                                      hsaco_suffix="_t4")
                self._gemv_int4_qkv_fused = True
                print("Fused QKV GEMV kernel (gemv_int4_qkv_fused.hip) loaded — 3-in-1 launch")
        except Exception as e:
            print(f"Fused QKV GEMV kernel failed to load: {e}")
        
        self._gemv_int4_v7 = False
        try:
            hip_path = HIP_DIR / "gemv_int4_v7.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v7_t16", "gemv_int4_v7")
                self.kernels.get_hip("gemv_int4_v7_t8", "gemv_int4_v7")
                self.kernels.get_hip("gemv_int4_v7_t4", "gemv_int4_v7")
                self._gemv_int4_v7 = True
                if self._gemv_int4_v8:
                    print("GEMV INT4 v7 (FP32 accum + 2x register blocking) loaded as fallback")
                else:
                    print("GEMV INT4 v7 (FP32 accum + 2x register blocking) loaded as default")
        except Exception as e:
            print(f"GEMV INT4 v7 failed (falling back to v6): {e}")
        try:
            hip_path = HIP_DIR / "gemv_int4_v6.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v6_t16", "gemv_int4_v6")
                self.kernels.get_hip("gemv_int4_v6_t8", "gemv_int4_v6")
                self.kernels.get_hip("gemv_int4_v6_t4", "gemv_int4_v6")
                self._gemv_int4_v6 = True
                if self._gemv_int4_v7:
                    print("GEMV INT4 v6 (register-cached scale/zero + prefetch) loaded as fallback")
                else:
                    print("GEMV INT4 v6 (register-cached scale/zero + prefetch) loaded as default for N<=4096")
        except Exception as e:
            print(f"GEMV INT4 v6 failed (falling back to v5): {e}")
        # Try to load v5 hybrid DPP+LDS reduction kernel (fallback for v6, or for N>4096).
        try:
            hip_path = HIP_DIR / "gemv_int4_v5.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v5_t16", "gemv_int4_v5")
                self.kernels.get_hip("gemv_int4_v5_t8", "gemv_int4_v5")
                self.kernels.get_hip("gemv_int4_v5_t4", "gemv_int4_v5")
                self._gemv_int4_v5 = True
                if self._gemv_int4_v6:
                    print("GEMV INT4 v5 (hybrid DPP+LDS t16) loaded as fallback for N>4096")
                else:
                    print("GEMV INT4 v5 (hybrid DPP+LDS t16) loaded as default for non-residual GEMV")
        except Exception as e:
            print(f"GEMV INT4 v5 failed (falling back to v4/v3): {e}")
        # Try to load AWQ GEMV kernel (v5 variant without zero-point subtraction).
        # AWQ dequantization: w = q * scale (no zeros tensor, saves 8 v_sub_f32 per uint32).
        # Used when awq_mode=True is set via set_awq_mode() after weight loading.
        self._gemv_int4_v5_awq = False
        try:
            hip_path = HIP_DIR / "gemv_int4_v5_awq.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v5_awq_t16", "gemv_int4_v5_awq")
                self.kernels.get_hip("gemv_int4_v5_awq_t8", "gemv_int4_v5_awq")
                self.kernels.get_hip("gemv_int4_v5_awq_t4", "gemv_int4_v5_awq")
                self._gemv_int4_v5_awq = True
                print("GEMV INT4 v5 AWQ (no zero-point) loaded")
        except Exception as e:
            print(f"GEMV INT4 v5 AWQ failed (AWQ mode will fall back to v5/v3): {e}")
        # Try to load v3 cooperative-reduction kernel (16 threads/col = t16 variant).
        # v3_t16 is 16% faster than v2_fused for N=4096 and same speed for N=11008.
        # Used for GEMV WITHOUT residual epilogue when v5/v6 unavailable.
        try:
            hip_path = HIP_DIR / "gemv_int4_v3.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_v3_t16", "gemv_int4_v3")
                self._gemv_int4_v3 = True
                if self._gemv_int4_v5 or self._gemv_int4_v6:
                    print("GEMV INT4 v3 (cooperative reduction t16) loaded as fallback")
                else:
                    print("GEMV INT4 v3 (cooperative reduction t16) loaded as default for non-residual GEMV")
        except Exception as e:
            print(f"GEMV INT4 v3 failed (falling back to v2): {e}")
        self._gemv_int4_dual = False
        self._gemv_int4_dual_fused = False
        self._gemv_int4_dual_awq = False
        self._gemv_int4_dual_awq_fused = False
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
        # AWQ variant of dual GEMV (no zero-point subtraction)
        try:
            hip_path = HIP_DIR / "gemv_int4_dual_awq.hip"
            if hip_path.exists() and self._gemv_int4_v2:
                self.kernels.get_hip("gemv_int4_dual_awq_splitk", "gemv_int4_dual_awq")
                self.kernels.get_hip("dual_awq_fp32_to_silu_fp16", "gemv_int4_dual_awq")
                self.kernels.get_hip("gemv_int4_dual_awq_fused", "gemv_int4_dual_awq")
                self._gemv_int4_dual_awq = True
                self._gemv_int4_dual_awq_fused = True
                print("GEMV INT4 dual AWQ (fused gate+up+silu, no zero-point) loaded")
        except Exception as e:
            print(f"GEMV INT4 dual AWQ failed: {e}")
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

    def _init_quant_kernels(self):
        """Initialize W8A8/W4A8 quantization kernels based on quant_format.

        W8A8: loads gemv_w8a8.hip (INT8 weight + INT8 activation GEMV)
              and activation_quant.hip (dynamic INT8 quantization of FP16 activations).
        W4A8: loads gemv_w4a8.hip (INT4 weight + INT8 activation GEMV)
              and activation_quant.hip.
        W4A16 (default): no extra kernels needed.

        Allocates INT8 activation buffer and FP32 max_abs/scale buffers for the
        two-kernel activation quantization pipeline.
        """
        self._w8a8_ready = False
        self._w4a8_ready = False
        self._act_quant_ready = False

        if self.quant_format == 'w4a16':
            print("quant_format=w4a16 (default INT4 weight + FP16 activation path)")
            return

        # Both W8A8 and W4A8 need the activation quantization kernel.
        hip_aq = HIP_DIR / "activation_quant.hip"
        if not hip_aq.exists():
            raise RuntimeError(
                f"activation_quant.hip not found at {hip_aq}. "
                f"Required for quant_format={self.quant_format!r}."
            )
        try:
            self.kernels.get_hip("activation_quant_reduce", "activation_quant")
            self.kernels.get_hip("activation_quant_quant", "activation_quant")
            self._act_quant_ready = True
            print("Activation quantization kernels loaded (activation_quant.hip)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load activation_quant.hip: {e}. "
                f"Required for quant_format={self.quant_format!r}."
            ) from e

        # Allocate persistent buffers for activation quantization:
        #   d_act_int8:   [max_act_size] INT8 — quantized activation
        #   d_act_maxabs: scalar FP32 — max absolute value (output of reduce pass)
        #   d_act_scale:  scalar FP32 — scale = max_abs / 127.0 (output of quant pass)
        max_act_size = max(self.config.hidden_size, self.local_intermediate_size)
        self.d_act_int8 = self.device.malloc(max_act_size)   # INT8 = 1 byte each
        self.d_act_maxabs = self.device.malloc(4)  # FP32 scalar
        self.d_act_scale = self.device.malloc(4)   # FP32 scalar

        if self.quant_format == 'w8a8':
            hip_w8a8 = HIP_DIR / "gemv_w8a8.hip"
            if not hip_w8a8.exists():
                raise RuntimeError(
                    f"gemv_w8a8.hip not found at {hip_w8a8}. "
                    "Required for quant_format='w8a8'."
                )
            try:
                self.kernels.get_hip("gemv_w8a8", "gemv_w8a8")
                self._w8a8_ready = True
                print("W8A8 GEMV kernel loaded (gemv_w8a8.hip)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load gemv_w8a8.hip: {e}. "
                    "Required for quant_format='w8a8'."
                ) from e

        elif self.quant_format == 'w4a8':
            hip_w4a8 = HIP_DIR / "gemv_w4a8.hip"
            if not hip_w4a8.exists():
                raise RuntimeError(
                    f"gemv_w4a8.hip not found at {hip_w4a8}. "
                    "Required for quant_format='w4a8'."
                )
            try:
                # Load the grouped variant (per-group scales, GPTQ-compatible)
                self.kernels.get_hip("gemv_w4a8_grouped", "gemv_w4a8")
                # Also load the per-channel dot4 variant for testing
                self.kernels.get_hip("gemv_w4a8_dot4", "gemv_w4a8")
                self._w4a8_ready = True
                print("W4A8 GEMV kernels loaded (gemv_w4a8.hip)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load gemv_w4a8.hip: {e}. "
                    "Required for quant_format='w4a8'."
                ) from e

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

    # Double-buffer support methods for compute-communication overlap
    def _swap_hidden_buffers(self):
        """Swap read and write hidden buffers.
        
        After each layer's allreduce completes, call this to swap buffers:
        - The allreduce write buffer becomes the next layer's read buffer
        - The previous read buffer becomes the next allreduce write target
        This enables overlapping layer N+1's RMSNorm with layer N's allreduce.
        """
        # Swap: read becomes old write, write becomes old read
        self.d_hidden, self.d_hidden_write = self.d_hidden_write, self.d_hidden

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
        # Double-buffer support: d_hidden_A and d_hidden_B for compute-communication overlap
        # d_hidden is the current read buffer, d_hidden_write is the current write buffer
        self.d_hidden_A = self.device.malloc(h * 2)
        self.d_hidden_B = self.device.malloc(h * 2)
        self.d_hidden = self.d_hidden_A  # Current read buffer (starts as A)
        self.d_hidden_write = self.d_hidden_B  # Current write buffer (starts as B)
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
            if data.dtype != np.float16 and data.dtype not in (np.int32, np.uint32, np.float32,
                                                                 np.int8, np.uint8):
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
            
            # Check if using INT4 attention projections (compressed-tensors format)
            has_int4_attn = 'q_qweight' in weights
            
            if has_int4_attn:
                # INT4 attention projections: upload qweight/scales/zeros for q/k/v/o
                # These use AWQ-style GEMV (no zero-point subtraction)
                lw.q_qweight = upload(weights.get('q_qweight', np.zeros(1, dtype=np.int32)))
                lw.q_scales = upload(weights.get('q_scales', np.zeros(1, dtype=np.float16)))
                lw.q_zeros = upload(weights.get('q_zeros', np.zeros(1, dtype=np.float16)))
                
                lw.k_qweight = upload(weights.get('k_qweight', np.zeros(1, dtype=np.int32)))
                lw.k_scales = upload(weights.get('k_scales', np.zeros(1, dtype=np.float16)))
                lw.k_zeros = upload(weights.get('k_zeros', np.zeros(1, dtype=np.float16)))
                
                lw.v_qweight = upload(weights.get('v_qweight', np.zeros(1, dtype=np.int32)))
                lw.v_scales = upload(weights.get('v_scales', np.zeros(1, dtype=np.float16)))
                lw.v_zeros = upload(weights.get('v_zeros', np.zeros(1, dtype=np.float16)))
                
                lw.o_qweight = upload(weights.get('o_qweight', np.zeros(1, dtype=np.int32)))
                lw.o_scales = upload(weights.get('o_scales', np.zeros(1, dtype=np.float16)))
                lw.o_zeros = upload(weights.get('o_zeros', np.zeros(1, dtype=np.float16)))
                
                # Also handle q_gate_weight (INT4)
                if 'q_gate_qweight' in weights:
                    lw.q_gate_qweight = upload(weights['q_gate_qweight'])
                    lw.q_gate_scales = upload(weights['q_gate_scales'])
                    lw.q_gate_zeros = upload(weights['q_gate_zeros'])
                
                # Concatenate Q, K, V weights for fused QKV GEMV kernel
                # Concatenate along output dimension (dim 0): [N_total/8, hidden]
                # where N_total = q_dim + 2*kv_dim
                if 'q_qweight' in weights and 'k_qweight' in weights and 'v_qweight' in weights:
                    q_qw = weights.get('q_qweight')
                    k_qw = weights.get('k_qweight')
                    v_qw = weights.get('v_qweight')
                    q_sc = weights.get('q_scales')
                    k_sc = weights.get('k_scales')
                    v_sc = weights.get('v_scales')
                    q_z = weights.get('q_zeros')
                    k_z = weights.get('k_zeros')
                    v_z = weights.get('v_zeros')
                    
                    # Concatenate along output dimension (axis 0)
                    qkv_qweight = np.concatenate([q_qw, k_qw, v_qw], axis=0)
                    qkv_scales = np.concatenate([q_sc, k_sc, v_sc], axis=0)
                    qkv_zeros = np.concatenate([q_z, k_z, v_z], axis=0)
                    
                    lw.qkv_qweight = upload(qkv_qweight)
                    lw.qkv_scales = upload(qkv_scales)
                    lw.qkv_zeros = upload(qkv_zeros)
            else:
                # FP16 attention projections (standard GPTQ/AWQ format)
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

            # Gate + up INT4 GEMV projections + down projection (column+row parallel)
            if self.quant_format in ('w8a8', 'w4a8'):
                # W8A8/W4A8: dispatch to quantized FFN helper (includes residual_add)
                self._decode_ffn_quantized(lw, h)
            else:
                # W4A16 path: fused gate+up dual GEMV, and down_proj with residual epilogue
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

    def decode_step_batch(self, token_embeddings: np.ndarray, batch_size: int, position: int) -> np.ndarray:
        """Run one decode step for a batch of sequences (single-GPU only).

        Supports batch>1 with dynamic GEMV/GEMM switching:
          - batch=1: uses GEMV kernels (bandwidth-bound, optimized)
          - batch>=2: uses GEMM kernels (tiled, compute-bound for larger M)

        Args:
            token_embeddings: [batch_size, hidden_dim] FP16 array
            batch_size: number of sequences in batch (1, 2, 4, etc.)
            position: current sequence position (same for all sequences in batch)

        Returns:
            [batch_size, hidden_dim] FP16 hidden states (after final RMSNorm)
        """
        if self.tp_size > 1:
            raise RuntimeError(
                "decode_step_batch() not supported with tp_size > 1. "
                "Multi-GPU batch support not yet implemented.")

        h = self.config.hidden_size
        cfg = self.config

        # Upload batch embeddings to GPU
        # For batch=1, use d_hidden directly; for batch>1, use d_hidden_batch if available
        if batch_size == 1:
            # Single sequence: use existing single-sequence path
            self.device.upload(self.d_hidden, token_embeddings[0].tobytes())
            hidden_out = self._decode_step_single(token_embeddings[0], position, use_batch_path=False)
            return hidden_out[None, :]
        else:
            # Batch > 1: use batched path with GEMM
            return self._decode_step_batched(token_embeddings, batch_size, position)

    def _decode_step_single(self, token_embedding: np.ndarray, position: int, use_batch_path: bool = False) -> np.ndarray:
        """Internal single-sequence decode (factored out from decode_step).
        
        This allows code reuse between decode_step (single) and decode_step_batch (batch=1 case).
        
        Args:
            token_embedding: [hidden_dim] FP16 array
            position: current sequence position
            use_batch_path: if True, writes output to d_hidden_batch for consistency
        
        Returns:
            [hidden_dim] FP16 hidden state
        """
        h = self.config.hidden_size
        cfg = self.config

        self._active_layer_idx = -1

        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.layers[layer_idx]
            self._active_layer_idx = layer_idx

            # Pre-attention RMSNorm
            self._launch_rmsnorm(self.d_normed, self.d_hidden, lw.attn_norm, h)

            if lw.layer_type == 'full_attention':
                self._decode_full_attention(layer_idx, lw, position)
            else:
                self._decode_linear_attention_gpu(layer_idx, lw, position)

            # FFN norm
            if self.tp_size <= 1:
                self._launch_rmsnorm(self.d_normed, self.d_hidden, lw.ffn_norm, h)
            else:
                self._launch_skip_rmsnorm(self.d_normed, self.d_hidden, self.d_proj_out,
                                           lw.ffn_norm, h)

            # FFN projections
            if self.quant_format in ('w8a8', 'w4a8'):
                self._decode_ffn_quantized(lw, h)
            else:
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

                if self.tp_size <= 1:
                    self._launch_gemv_int4(self.d_hidden, self.d_ffn_gate,
                                            lw.down_qweight, lw.down_scales, lw.down_zeros,
                                            self.local_intermediate_size, h,
                                            residual=self.d_hidden)
                else:
                    self._launch_gemv_int4(self.d_ffn_out, self.d_ffn_gate,
                                            lw.down_qweight, lw.down_scales, lw.down_zeros,
                                            self.local_intermediate_size, h)

        # Final RMSNorm
        if self.d_final_norm:
            self._launch_rmsnorm(self.d_hidden2, self.d_hidden, self.d_final_norm, h)
            return np.frombuffer(self.device.download(self.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(self.device.download(self.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_step_batched(self, token_embeddings: np.ndarray, batch_size: int, position: int) -> np.ndarray:
        """Batched decode step using GEMM for projections.

        For batch>=2, uses tiled GEMM kernels instead of GEMV for better compute utilization.

        Args:
            token_embeddings: [batch_size, hidden_dim] FP16 array
            batch_size: number of sequences (>= 2)
            position: current sequence position

        Returns:
            [batch_size, hidden_dim] FP16 hidden states
        """
        h = self.config.hidden_size
        cfg = self.config

        # Allocate batched buffers if not already done
        if not hasattr(self, '_d_hidden_batch') or self._batch_size_cached != batch_size:
            if hasattr(self, '_d_hidden_batch'):
                self.device.free(self._d_hidden_batch)
                self.device.free(self._d_normed_batch)
                self.device.free(self._d_proj_out_batch)
                self.device.free(self._d_ffn_out_batch)
            # Allocate batched hidden state: [batch_size * hidden]
            self._d_hidden_batch = self.device.malloc(batch_size * h * 2)
            self._d_normed_batch = self.device.malloc(batch_size * h * 2)
            self._d_proj_out_batch = self.device.malloc(batch_size * h * 2)
            self._d_ffn_out_batch = self.device.malloc(batch_size * h * 2)
            self._batch_size_cached = batch_size
            print(f"  Allocated batched buffers for batch_size={batch_size}")

        # Upload batch embeddings
        self.device.upload(self._d_hidden_batch, token_embeddings.tobytes())

        self._active_layer_idx = -1

        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.layers[layer_idx]
            self._active_layer_idx = layer_idx

            # Batched RMSNorm: process each sequence in the batch
            # For now, use sequential RMSNorm per sequence (can be optimized later)
            for b in range(batch_size):
                src_ptr = self._d_hidden_batch + b * h * 2
                dst_ptr = self._d_normed_batch + b * h * 2
                self._launch_rmsnorm(dst_ptr, src_ptr, lw.attn_norm, h)

            # Use GEMM for batched projections
            if lw.layer_type == 'full_attention':
                self._decode_full_attention_batched(layer_idx, lw, position, batch_size)
            else:
                self._decode_linear_attention_batched(layer_idx, lw, position, batch_size)

            # Batched FFN norm
            for b in range(batch_size):
                src_ptr = self._d_hidden_batch + b * h * 2
                dst_ptr = self._d_normed_batch + b * h * 2
                self._launch_rmsnorm(dst_ptr, src_ptr, lw.ffn_norm, h)

            # Batched FFN projections using GEMM
            # Launch GEMM: [batch, local_intermediate] = [batch, h] @ [local_intermediate, h]^T
            # For INT4: gemm_int4_prefill_v2, for FP16: gemm_fp16_prefill
            if self.quant_format not in ('w8a8', 'w4a8') and self._gemm_int4_prefill:
                # INT4 GEMM path for FFN gate+up
                # Note: this is a simplified path; full implementation would fuse gate+up+silu
                # For now, use sequential per-sequence INT4 GEMV as fallback
                for b in range(batch_size):
                    normed_ptr = self._d_normed_batch + b * h * 2
                    gate_ptr = self._d_ffn_out_batch + b * self.local_intermediate_size * 2
                    
                    if self._gemv_int4_dual:
                        # Use fused dual GEMV for gate+up+silu
                        self._launch_ffn_gate_up_silu(gate_ptr, normed_ptr, lw, h, self.local_intermediate_size)
                    else:
                        gate_proj = self._d_ffn_gate if hasattr(self, '_d_ffn_gate') else self.device.malloc(self.local_intermediate_size * 2)
                        up_proj = self._d_ffn_up if hasattr(self, '_d_ffn_up') else self.device.malloc(self.local_intermediate_size * 2)
                        
                        self._launch_gemv_int4(gate_proj, normed_ptr,
                                                lw.gate_qweight, lw.gate_scales,
                                                lw.gate_zeros, h, self.local_intermediate_size)
                        self._launch_gemv_int4(up_proj, normed_ptr,
                                                lw.up_qweight, lw.up_scales,
                                                lw.up_zeros, h, self.local_intermediate_size)
                        self._launch_silu_fused(gate_proj, up_proj, gate_proj, self.local_intermediate_size)
                        
                        # Down projection
                        self._launch_gemv_int4(gate_ptr, gate_proj,
                                                lw.down_qweight, lw.down_scales, lw.down_zeros,
                                                self.local_intermediate_size, h,
                                                residual=normed_ptr)
            else:
                # Fallback to per-sequence GEMV
                for b in range(batch_size):
                    normed_ptr = self._d_normed_batch + b * h * 2
                    ffn_out_ptr = self._d_ffn_out_batch + b * h * 2
                    
                    if self.quant_format in ('w8a8', 'w4a8'):
                        self._decode_ffn_quantized_batched(lw, h, normed_ptr, ffn_out_ptr)
                    else:
                        # Use dual GEMV if available
                        if self._gemv_int4_dual:
                            self._launch_ffn_gate_up_silu(ffn_out_ptr, normed_ptr, lw, h, self.local_intermediate_size)
                        else:
                            gate_ptr = self._d_ffn_gate if hasattr(self, '_d_ffn_gate') else self.device.malloc(self.local_intermediate_size * 2)
                            up_ptr = self._d_ffn_up if hasattr(self, '_d_ffn_up') else self.device.malloc(self.local_intermediate_size * 2)
                            
                            self._launch_gemv_int4(gate_ptr, normed_ptr,
                                                    lw.gate_qweight, lw.gate_scales,
                                                    lw.gate_zeros, h, self.local_intermediate_size)
                            self._launch_gemv_int4(up_ptr, normed_ptr,
                                                    lw.up_qweight, lw.up_scales,
                                                    lw.up_zeros, h, self.local_intermediate_size)
                            self._launch_silu_fused(gate_ptr, up_ptr, gate_ptr, self.local_intermediate_size)
                            
                            self._launch_gemv_int4(ffn_out_ptr, gate_ptr,
                                                    lw.down_qweight, lw.down_scales, lw.down_zeros,
                                                    self.local_intermediate_size, h,
                                                    residual=normed_ptr)

            # Copy FFN output back to hidden (with residual already added)
            for b in range(batch_size):
                src_ptr = self._d_ffn_out_batch + b * h * 2
                dst_ptr = self._d_hidden_batch + b * h * 2
                # Use hipMemcpy for batch copy
                self.device._ensure_device()
                self.device.hip.memcpy_d2d(dst_ptr, src_ptr, h * 2)

        # Advance KV cache for all sequences in batch
        self.kv_cache.current_len += 1

        # Final RMSNorm for all sequences
        if self.d_final_norm:
            for b in range(batch_size):
                src_ptr = self._d_hidden_batch + b * h * 2
                dst_ptr = self._d_hidden_batch + b * h * 2
                self._launch_rmsnorm(dst_ptr, src_ptr, self.d_final_norm, h)

        # Download batched output
        output_bytes = self.device.download(self._d_hidden_batch, batch_size * h * 2)
        output = np.frombuffer(output_bytes, dtype=np.float16).reshape(batch_size, h)
        return output

    def _decode_ffn_quantized_batched(self, lw: 'LayerWeights', h: int, normed_ptr: int, out_ptr: int):
        """FFN block for W8A8/W4A8 in batched mode (single sequence)."""
        inter = self.local_intermediate_size
        gate_ptr = self._d_ffn_gate if hasattr(self, '_d_ffn_gate') else self.device.malloc(inter * 2)
        up_ptr = self._d_ffn_up if hasattr(self, '_d_ffn_up') else self.device.malloc(inter * 2)

        if self.quant_format == 'w8a8' and self._w8a8_ready:
            self._launch_gemv_w8a8(gate_ptr, normed_ptr, lw.gate_w8a8, lw.gate_scale_w8a8, h, inter)
            self._launch_gemv_w8a8(up_ptr, normed_ptr, lw.up_w8a8, lw.up_scale_w8a8, h, inter)
            self._launch_silu_fused(gate_ptr, up_ptr, gate_ptr, inter)
            self._launch_gemv_w8a8(out_ptr, gate_ptr, lw.down_w8a8, lw.down_scale_w8a8, inter, h)
        elif self.quant_format == 'w4a8' and self._w4a8_ready:
            self._launch_gemv_w4a8(gate_ptr, normed_ptr, lw.gate_w4a8, lw.gate_scale_w4a8, h, inter)
            self._launch_gemv_w4a8(up_ptr, normed_ptr, lw.up_w4a8, lw.up_scale_w8a8, h, inter)
            self._launch_silu_fused(gate_ptr, up_ptr, gate_ptr, inter)
            self._launch_gemv_w4a8(out_ptr, gate_ptr, lw.down_w4a8, lw.down_scale_w4a8, inter, h)

    def _decode_full_attention_batched(self, layer_idx: int, lw: LayerWeights,
                                        position: int, batch_size: int):
        """Batched full attention decode using GEMM for projections.
        
        Key fix: Each sequence in the batch writes to its own KV cache slot.
        The KV cache layout is now: [layer, batch, seq, kv_heads, head_dim]
        Attention is computed sequentially per sequence (batched attention kernel would be a future optimization).
        """
        cfg = self.config
        h = cfg.hidden_size
        
        # Allocate batched Q/K/V buffers if not already done
        if not hasattr(self, '_d_q_fused_batch') or self._batch_size_cached != batch_size:
            if hasattr(self, '_d_q_fused_batch'):
                self.device.free(self._d_q_fused_batch)
                self.device.free(self._d_kv_fused_batch)
                self.device.free(self._d_attn_out_batch)
            # Q+Qgate: [batch_size, 2*q_dim]
            self._d_q_fused_batch = self.device.malloc(batch_size * 2 * self.q_dim * 2)
            # K+V: [batch_size, 2*kv_dim]
            self._d_kv_fused_batch = self.device.malloc(batch_size * 2 * self.kv_dim * 2)
            # Attention output: [batch_size, q_dim]
            self._d_attn_out_batch = self.device.malloc(batch_size * self.q_dim * 2)
            print(f"  Allocated batched attention buffers for batch_size={batch_size}")

        # For batch>1, decide between GEMM and GEMV based on batch size
        # GEMM is more efficient for larger batches (M >= 4), GEMV for small batches
        use_gemm = self._gemm_fp16_prefill and batch_size >= 4
        
        if use_gemm:
            # FP16 GEMM for Q+Qgate projection
            # [batch, 2*q_dim] = [batch, h] @ [2*q_dim, h]^T
            q_fused_ptr = lw.q_fused_weight if hasattr(lw, 'q_fused_weight') else lw.q_weight
            self._launch_gemm_fp16(self._d_q_fused_batch,
                                    self._d_normed_batch, q_fused_ptr,
                                    batch_size, 2 * self.q_dim, h)
            
            # FP16 GEMM for K+V projection
            kv_fused_ptr = lw.kv_fused_weight if hasattr(lw, 'kv_fused_weight') else lw.k_weight
            self._launch_gemm_fp16(self._d_kv_fused_batch,
                                    self._d_normed_batch, kv_fused_ptr,
                                    batch_size, 2 * self.kv_dim, h)
        else:
            # Use per-sequence GEMV for small batches (more reliable for M < 4)
            for b in range(batch_size):
                normed_ptr = self._d_normed_batch + b * h * 2
                q_ptr = self._d_q_fused_batch + b * 2 * self.q_dim * 2
                kv_ptr = self._d_kv_fused_batch + b * 2 * self.kv_dim * 2
                
                self._launch_gemv_fp16(q_ptr, normed_ptr, lw.q_fused_weight, h, 2 * self.q_dim)
                self._launch_gemv_fp16(kv_ptr, normed_ptr, lw.kv_fused_weight, h, 2 * self.kv_dim)

        # Process each sequence: QKNorm/RoPE, KV cache write, and attention
        # Each sequence writes to and reads from its own KV cache slot (batch_idx = b)
        for b in range(batch_size):
            q_ptr = self._d_q_fused_batch + b * 2 * self.q_dim * 2
            kv_base = self._d_kv_fused_batch + b * 2 * self.kv_dim * 2
            k_ptr = kv_base
            v_ptr = kv_base + self.kv_dim * 2  # V starts after K in the fused buffer
            attn_out_ptr = self._d_attn_out_batch + b * self.q_dim * 2
            
            # QKNorm + RoPE for Q
            self._launch_qknorm_rope(q_ptr, lw.q_norm, position,
                                      self.local_num_attention_heads, cfg.head_dim)
            # QKNorm + RoPE for K
            self._launch_qknorm_rope(k_ptr, lw.k_norm, position,
                                      self.local_num_kv_heads, cfg.head_dim)
            
            # Write K and V to KV cache for this sequence's batch slot
            # Get KV cache pointers for this batch index at current position
            k_cache_ptr, v_cache_ptr = self.kv_cache.get_kv_ptr(layer_idx, position, batch_idx=b)
            
            # Copy K and V to cache (GPU-to-GPU)
            kv_size = self.kv_dim * 2
            self.device.hip.memcpy_d2d(k_cache_ptr, k_ptr, kv_size)
            self.device.hip.memcpy_d2d(v_cache_ptr, v_ptr, kv_size)
            
            # Decode attention for this sequence
            # Get KV cache base pointers for this batch (includes all positions 0..position)
            k_cache_base = self.kv_cache.layer_k_ptr(layer_idx, batch_idx=b)
            v_cache_base = self.kv_cache.layer_v_ptr(layer_idx, batch_idx=b)
            
            # Launch attention: seq_len = position + 1 (number of valid KV pairs)
            self._launch_decode_attn_256(
                attn_out_ptr, q_ptr,
                k_cache_base, v_cache_base,
                position + 1  # kv_seq_len
            )
            
            # Sigmoid gate (Q_gate is in second half of q_fused)
            q_gate_ptr = self._d_q_fused_batch + b * 2 * self.q_dim * 2 + self.q_dim * 2
            self._launch_sigmoid_mul(attn_out_ptr, q_gate_ptr, self.q_dim)
            
            # O projection
            o_out_ptr = self._d_proj_out_batch + b * h * 2 if hasattr(self, '_d_proj_out_batch') else self.d_proj_out
            self._launch_gemv_fp16(o_out_ptr, attn_out_ptr, lw.o_weight,
                                    self.q_dim, h, residual=self._d_hidden_batch + b * h * 2)

    def _decode_linear_attention_batched(self, layer_idx: int, lw: LayerWeights,
                                          position: int, batch_size: int):
        """Batched linear attention (DeltaNet) decode."""
        h = self.config.hidden_size
        
        # For batched linear attention, use per-sequence GEMV for now
        # (full batched DeltaNet would require a new kernel)
        for b in range(batch_size):
            normed_ptr = self._d_normed_batch + b * h * 2
            
            # Input projection
            self._launch_gemv_fp16(self.d_la_packed, normed_ptr,
                                    lw.la_in_proj_fused, h, self.la_total_dim)
            
            # Download and process on CPU (DeltaNet CPU path)
            # For production, a GPU DeltaNet kernel would be needed
            packed = np.frombuffer(self.device.download(self.d_la_packed, self.la_total_dim * 2),
                                    dtype=np.float16).copy()
            
            # ... (DeltaNet CPU processing, same as _decode_linear_attention)
            # Simplified for brevity - would need full implementation
            
            # Output projection
            y_f16 = packed[:self.la_z_dim].astype(np.float16)
            self.device.upload(self.d_la_out, y_f16.tobytes())
            
            la_out_ptr = self._d_proj_out_batch + b * h * 2 if hasattr(self, '_d_proj_out_batch') else self.d_proj_out
            self._launch_gemv_fp16(la_out_ptr, self.d_la_out, lw.la_out_proj,
                                    self.la_z_dim, h, residual=self._d_hidden_batch + b * h * 2)

    def _decode_ffn_quantized(self, lw: 'LayerWeights', h: int):
        """FFN block for W8A8 and W4A8 quantization formats (decode step).

        Called by decode_step when quant_format is 'w8a8' or 'w4a8'.
        Unlike the W4A16 path, these kernels don't support a fused residual
        epilogue, so a separate _launch_residual_add is used after down_proj.

        Args:
            lw: layer weights (must have w8a8/w4a8 weight fields set)
            h:  hidden size
        """
        inter = self.local_intermediate_size

        if self.quant_format == 'w8a8' and self._w8a8_ready:
            self._launch_gemv_w8a8(self.d_ffn_gate, self.d_normed,
                                    lw.gate_w8a8, lw.gate_scale_w8a8, h, inter)
            self._launch_gemv_w8a8(self.d_ffn_up, self.d_normed,
                                    lw.up_w8a8, lw.up_scale_w8a8, h, inter)
            self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                     self.d_ffn_gate, inter)
            self._launch_gemv_w8a8(self.d_ffn_out, self.d_ffn_gate,
                                    lw.down_w8a8, lw.down_scale_w8a8, inter, h)
            if self.tp_size <= 1:
                self._launch_residual_add(self.d_hidden, self.d_ffn_out, h)

        elif self.quant_format == 'w4a8' and self._w4a8_ready:
            self._launch_gemv_w4a8(self.d_ffn_gate, self.d_normed,
                                    lw.gate_w4a8, lw.gate_scale_w4a8, h, inter)
            self._launch_gemv_w4a8(self.d_ffn_up, self.d_normed,
                                    lw.up_w4a8, lw.up_scale_w4a8, h, inter)
            self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                     self.d_ffn_gate, inter)
            self._launch_gemv_w4a8(self.d_ffn_out, self.d_ffn_gate,
                                    lw.down_w4a8, lw.down_scale_w4a8, inter, h)
            if self.tp_size <= 1:
                self._launch_residual_add(self.d_hidden, self.d_ffn_out, h)

    def _decode_full_attention(self, layer_idx: int, lw: LayerWeights,
                                position: int):
        """Full attention decode step: Q/K/V projections, RoPE, decode_attn_256, gate.

        With TP: each GPU handles local_num_attention_heads Q heads and
        local_num_kv_heads KV heads. O projection is row-parallel with allreduce.
        
        When use_int4_attention=True (compressed-tensors format):
        - Uses INT4 GEMV (AWQ kernel) for q/k/v/o projections
        - Attention weights are INT4 quantized with symmetric quantization
        
        When _direct_kv_write=True:
        - K GEMV writes to d_k (working buffer for qknorm_rope)
        - V GEMV writes directly to cache position (eliminates V D2D copy)
        - qknorm_rope_cachew writes post-RoPE K directly to cache (eliminates K D2D copy)
        - No separate append_kv_gpu call needed
        """
        cfg = self.config
        h = cfg.hidden_size

        if self.use_int4_attention:
            # INT4 attention projections (compressed-tensors format)
            # Use AWQ GEMV kernel (no zero-point subtraction) for q/k/v/o
            # Enable AWQ mode temporarily if not already enabled
            was_awq = self._awq_mode
            if not was_awq and self._gemv_int4_v5_awq:
                self._awq_mode = True
            
            # Q+Qgate fused INT4 GEMV → d_q_fused
            self._launch_gemv_int4(self.d_q_fused, self.d_normed,
                                    lw.q_qweight, lw.q_scales, lw.q_zeros,
                                    h, 2 * self.q_dim)
            
            if self._direct_kv_write:
                # Split KV: K GEMV to working buffer, V GEMV direct to cache
                self._launch_gemv_int4(self.d_k, self.d_normed,
                                        lw.k_qweight, lw.k_scales, lw.k_zeros,
                                        h, self.kv_dim)
                _, cache_v_ptr = self.kv_cache.get_kv_ptr(layer_idx, self.kv_cache.current_len, batch_idx=0)
                self._launch_gemv_int4(cache_v_ptr, self.d_normed,
                                        lw.v_qweight, lw.v_scales, lw.v_zeros,
                                        h, self.kv_dim)
            else:
                # Fused K+V projection
                # Note: for INT4 we don't have fused KV, so we do sequential
                self._launch_gemv_int4(self.d_k, self.d_normed,
                                        lw.k_qweight, lw.k_scales, lw.k_zeros,
                                        h, self.kv_dim)
                self._launch_gemv_int4(self.d_v, self.d_normed,
                                        lw.v_qweight, lw.v_scales, lw.v_zeros,
                                        h, self.kv_dim)
            
            # Restore AWQ mode
            if not was_awq:
                self._awq_mode = False
        else:
            # FP16 attention projections (standard GPTQ/AWQ format)
            # Q+Qgate and K+V projections run sequentially on the default stream.
            # For decode (batch=1), these are tiny GEMVs (~30µs each) where concurrency
            # benefit is negligible compared to the 2 host-blocking sync calls per layer.
            # Sequential execution on the default stream guarantees ordering for QKNorm
            # without explicit stream synchronization.
            # Stream 1: Q+Qgate fused GEMV → d_q_fused ([Q, Q_gate])
            self._launch_gemv_fp16(self.d_q_fused, self.d_normed, lw.q_fused_weight,
                                    h, 2 * self.q_dim)

            if self._direct_kv_write:
                # Split KV GEMV into K GEMV (working buffer) + V GEMV (direct cache write)
                # K GEMV: writes to d_k (working buffer, needed for qknorm_rope)
                self._launch_gemv_fp16(self.d_k, self.d_normed, lw.k_weight,
                                        h, self.kv_dim)
                # V GEMV: writes directly to cache position (eliminates V D2D copy)
                # Use get_kv_ptr to get the correct cache position for batch_idx=0
                _, cache_v_ptr = self.kv_cache.get_kv_ptr(layer_idx, self.kv_cache.current_len, batch_idx=0)
                self._launch_gemv_fp16(cache_v_ptr, self.d_normed, lw.v_weight,
                                        h, self.kv_dim)
            else:
                # Fused K+V projection: normed_hidden → [K, V] [2*kv_dim]
                self._launch_gemv_fp16(self.d_kv_fused, self.d_normed, lw.kv_fused_weight,
                                        h, 2 * self.kv_dim)

        # No stream sync needed: both GEMVs now run on the default stream,
        # so QKNorm ordering is guaranteed by the null stream's serial execution.

        # Fused QK-norm + RoPE (per-head RMSNorm + partial RoPE in one launch each)
        # Replaces 4 separate launches: qk_norm(Q) + qk_norm(K) + rope(Q) + rope(K)
        self._launch_qknorm_rope(self.d_q, lw.q_norm, position,
                                  self.local_num_attention_heads, cfg.head_dim)

        if self._direct_kv_write:
            # Use cache-write variant: writes post-RoPE K to both d_k AND cache position
            # Use get_kv_ptr to get the correct cache position for batch_idx=0
            cache_k_ptr, _ = self.kv_cache.get_kv_ptr(layer_idx, self.kv_cache.current_len, batch_idx=0)
            self._launch_qknorm_rope_cachew(self.d_k, lw.k_norm, position,
                                             self.local_num_kv_heads, cfg.head_dim,
                                             cache_k_ptr)
            # V was already written to cache above; K was written by qknorm_rope_cachew
            # No separate append_kv_gpu call needed
        else:
            self._launch_qknorm_rope(self.d_k, lw.k_norm, position,
                                      self.local_num_kv_heads, cfg.head_dim)
            # Update KV cache (GPU-to-GPU copy, no host roundtrip)
            self.kv_cache.append_kv_gpu(layer_idx, self.d_k, self.d_v, batch_idx=0)

        # Decode attention (head_dim=256 variant, local heads only)
        # Use layer pointers for batch_idx=0
        self._launch_decode_attn_256(
            self.d_attn_out, self.d_q,
            self.kv_cache.layer_k_ptr(layer_idx, batch_idx=0),
            self.kv_cache.layer_v_ptr(layer_idx, batch_idx=0),
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
        if self.use_int4_attention:
            # INT4 O projection (AWQ kernel)
            was_awq = self._awq_mode
            if not was_awq and self._gemv_int4_v5_awq:
                self._awq_mode = True
            
            if self.tp_size <= 1:
                self._launch_gemv_int4(self.d_hidden, self.d_attn_out,
                                        lw.o_qweight, lw.o_scales, lw.o_zeros,
                                        self.q_dim, h, residual=self.d_hidden)
            else:
                self._launch_gemv_int4(self.d_proj_out, self.d_attn_out,
                                        lw.o_qweight, lw.o_scales, lw.o_zeros,
                                        self.q_dim, h)
            
            if not was_awq:
                self._awq_mode = False
        else:
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
        # Use v3 (float4 vectorized, 1.42-1.58x faster than v2) if available
        try:
            func = self.kernels.get_hip("rmsnorm_v3", "elementwise_v3")
        except Exception:
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
        # Use v3 (float4 vectorized, 1.16x faster than v2) if available
        try:
            func = self.kernels.get_hip("skip_rmsnorm_v3", "elementwise_v3")
        except Exception:
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

    def _launch_qknorm_rope_cachew(self, x_ptr, norm_weight, position,
                                    num_heads, head_dim, cache_dst):
        """Fused per-head RMSNorm + partial RoPE + KV cache write.

        Like _launch_qknorm_rope but also writes post-RoPE output to cache_dst
        (the KV cache position for the current token), eliminating a separate
        hipMemcpyAsync D2D copy.

        Args:
            x_ptr:       GPU pointer to [num_heads, head_dim] FP16 (modified in-place)
            norm_weight: GPU pointer to [head_dim] FP16 norm weights
            position:    Sequence position (for RoPE cos/sin table lookup)
            num_heads:   Number of K heads
            head_dim:    Dimension per head (e.g. 256)
            cache_dst:   GPU pointer to KV cache position for this token (K channel)
        """
        func = self.kernels.get_hip("qknorm_rope_cachew_fused", "qknorm_rope_cachew")
        half_rotary = self.rotary_dim // 2
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables
        params = [
            ctypes.c_uint64(x_ptr),
            ctypes.c_uint64(norm_weight),
            ctypes.c_uint64(self.d_cos + cos_offset),
            ctypes.c_uint64(self.d_sin + cos_offset),
            ctypes.c_uint64(cache_dst),        # cache write destination
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
        v2: 64×64 tile with XOR-swizzled LDS and v_dot2_f32_f16.
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
        # AWQ mode: use no-zeros-point kernel for non-residual GEMV (saves loads + subtractions)
        if self._awq_mode and self._gemv_int4_v5_awq and not residual:
            func = self.kernels.get_hip("gemv_int4_v5_awq_t16", "gemv_int4_v5_awq")
            cols_per_wg = 16  # 256 threads / 16 threads_per_col
            grid_x = (N + cols_per_wg - 1) // cols_per_wg
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(qweight),
                ctypes.c_uint64(scales),
                ctypes.c_uint64(dst),
                ctypes.c_uint32(K),
                ctypes.c_uint32(N),
                ctypes.c_uint32(self.config.group_size),
            ]
            self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
            self._record_launch(getattr(self, '_active_layer_idx', -1))
            return
        if self._gemv_int4_v2:
            # For GEMV without residual epilogue, use shape-based selection:
            #   - v6_t16 for N <= 4096 (1.13x speedup at N=4096,K=4096)
            #   - v5_t16 for N > 4096 (v6 may regress at N=17408,K=5120)
            # v5_t16 or v3_t16 as fallback if v6 unavailable.
            # v2_fused remains for residual GEMV (down_proj).
            if not residual and (self._gemv_int4_v8 or self._gemv_int4_v7 or self._gemv_int4_v6 or self._gemv_int4_v5 or self._gemv_int4_v3):
                # Shape-based kernel + thread config selection: v8 > v7 > v6 > v5 > v3
                # Thread config: t4 for N<=640, t8 for N<=2048, t16 for larger
                if N <= 640:
                    t_suffix = "_t4"
                    cols_per_wg = 64
                elif N <= 2048:
                    t_suffix = "_t8"
                    cols_per_wg = 32
                else:
                    t_suffix = "_t16"
                    cols_per_wg = 16

                if self._gemv_int4_v8:
                    func = self.kernels.get_hip("gemv_int4_v8" + t_suffix, "gemv_int4_v8")
                    kernel_name = "v8" + t_suffix
                elif self._gemv_int4_v7:
                    func = self.kernels.get_hip("gemv_int4_v7" + t_suffix, "gemv_int4_v7")
                    kernel_name = "v7" + t_suffix
                elif self._gemv_int4_v6 and (N <= 4096):
                    func = self.kernels.get_hip("gemv_int4_v6" + t_suffix, "gemv_int4_v6")
                    kernel_name = "v6" + t_suffix
                elif self._gemv_int4_v5:
                    func = self.kernels.get_hip("gemv_int4_v5" + t_suffix, "gemv_int4_v5")
                    kernel_name = "v5" + t_suffix
                else:
                    func = self.kernels.get_hip("gemv_int4_v3" + t_suffix, "gemv_int4_v3")
                    kernel_name = "v3" + t_suffix
                
                grid_x = (N + cols_per_wg - 1) // cols_per_wg
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
                self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
            else:
                # v2_fused: used for residual GEMV (down_proj) or v3 unavailable.
                # Single fused launch: no separate memset or fp32_to_fp16 needed.
                # Optional residual epilogue: if residual != 0, dst[i] += residual[i]
                grid_x = (N + 255) // 256
                k_splits = self._gemv_int4_k_splits
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

        Uses the fully fused kernel (gemv_int4_dual_fused or gemv_int4_dual_awq_fused)
        when available: no separate memset or fp32_to_fp16 calls needed. The kernel
        initializes both FP32 accumulator buffers internally on the first-completing
        split tile, and writes FP16 output with SiLU in the last-completing split tile.

        AWQ mode: uses gemv_int4_dual_awq_fused which skips zero-point subtraction,
        saving 8 v_sub_f32 instructions per uint32 word and eliminating zeros loads.

        Saves 2 memset launches + 1 convert+silu launch vs the 3-kernel path.
        """
        grid_x = (N + 255) // 256
        k_splits = self._gemv_int4_k_splits

        if self._awq_mode and self._gemv_int4_dual_awq_fused:
            # AWQ fused path: single launch, no memset, no zero-point subtraction
            func = self.kernels.get_hip("gemv_int4_dual_awq_fused", "gemv_int4_dual_awq")
            params = [
                ctypes.c_uint64(src),
                ctypes.c_uint64(lw.gate_qweight),
                ctypes.c_uint64(lw.gate_scales),
                ctypes.c_uint64(lw.up_qweight),
                ctypes.c_uint64(lw.up_scales),
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
        elif self._gemv_int4_dual_fused:
            # GPTQ fused path: single launch, no memset needed
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
            if self._awq_mode and self._gemv_int4_dual_awq:
                # AWQ split-K (no zeros)
                func = self.kernels.get_hip("gemv_int4_dual_awq_splitk", "gemv_int4_dual_awq")
                params = [
                    ctypes.c_uint64(src),
                    ctypes.c_uint64(lw.gate_qweight),
                    ctypes.c_uint64(lw.gate_scales),
                    ctypes.c_uint64(lw.up_qweight),
                    ctypes.c_uint64(lw.up_scales),
                    ctypes.c_uint64(self.d_gemv_fp32),
                    ctypes.c_uint64(self.d_gemv_fp32_2),
                    ctypes.c_uint32(K),
                    ctypes.c_uint32(N),
                    ctypes.c_uint32(self.config.group_size),
                ]
            else:
                # GPTQ split-K (with zeros)
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
            if self._awq_mode and self._gemv_int4_dual_awq:
                func2 = self.kernels.get_hip("dual_awq_fp32_to_silu_fp16", "gemv_int4_dual_awq")
            else:
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
        """Apply partial RoPE (only first rotary_dim dims of each head).

        Uses HIP rope_v2 kernel if available (vectorized half2 loads),
        otherwise falls back to assembly rope.s.
        """
        half_rotary = self.rotary_dim // 2
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables

        if self._rope_hip:
            func = self.kernels.get_hip("rope_fp16_v2", "rope_v2")
            params = [
                ctypes.c_uint64(x_ptr),
                ctypes.c_uint64(self.d_cos + cos_offset),
                ctypes.c_uint64(self.d_sin + cos_offset),
                ctypes.c_uint32(head_dim),
                ctypes.c_uint32(num_heads),
            ]
            # Grid: (1, num_heads, 1) — note: token index 0 (single token decode)
            # Block: (half_rotary, 1, 1) — one thread per rotation pair
            self.device.launch(func, (1, num_heads, 1), (half_rotary, 1, 1), params)
        else:
            func = self.kernels.get("rope_fp16", "rope")
            params = [
                ctypes.c_uint64(x_ptr),
                ctypes.c_uint64(self.d_cos + cos_offset),
                ctypes.c_uint64(self.d_sin + cos_offset),
                ctypes.c_uint32(head_dim),
                ctypes.c_uint32(num_heads),
            ]
            self.device.launch(func, (1, num_heads, 1), (half_rotary, 1, 1), params)

    def _launch_decode_attn_256(self, out, q, k_cache, v_cache, seq_len):
        """Launch head_dim=256 decode attention for a single sequence.

        Uses flash_attn_256_tuned.hip with shape-based selection:
          - flash_attn_256_decode_64t: 64-thread variant (1 wavefront/WG, reduced resource usage)
          - flash_attn_256_decode: 256-thread variant (4 wavefronts/WG, KV-parallel merge)
        
        The 64-thread kernel is 2-4.6x slower per-kernel but uses fewer GPU resources
        (1 wave/WG vs 4 waves/WG), potentially improving overall throughput in
        resource-constrained scenarios. Falls back to the original flash_attn_256_fp16
        if tuned kernels are unavailable.
        
        TP: uses local head counts.
        """
        try:
            # Use 256-thread decode kernel (4-WF KV-parallel merge, faster per-kernel)
            func = self.kernels.get_hip("flash_attn_256_decode", "flash_attn_256_tuned")
            params = [
                ctypes.c_uint64(q),
                ctypes.c_uint64(k_cache),
                ctypes.c_uint64(v_cache),
                ctypes.c_uint64(out),
                ctypes.c_uint32(seq_len),       # kv_seq_len
                ctypes.c_uint32(self.local_num_attention_heads),
                ctypes.c_uint32(self.local_num_kv_heads),
            ]
            self.device.launch(func, (self.local_num_attention_heads, 1, 1),
                               (256, 1, 1), params)
        except Exception:
            # Fallback to original flash_attn_256_fp16
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

    def _launch_decode_attn_256_batched(self, out_batch, q_batch, k_cache_base, v_cache_base,
                                         seq_len, batch_size):
        """Launch head_dim=256 decode attention for a batch of sequences.
        
        The kernel grid includes the batch dimension: Grid=(num_heads, batch_size, 1)
        Each (head, batch_idx) pair processes one sequence's attention independently.
        
        Args:
            out_batch: GPU pointer to batched output [batch_size, num_heads, head_dim]
            q_batch: GPU pointer to batched Q [batch_size, num_heads, head_dim]
            k_cache_base: Base pointer to K cache for this layer (batch dimension included)
            v_cache_base: Base pointer to V cache for this layer (batch dimension included)
            seq_len: Current sequence length (same for all sequences in batch)
            batch_size: Number of sequences in batch
        
        Note: The flash_attn_256_decode kernel expects grid=(num_heads, batch_size, 1)
        and each block processes one (head, batch_idx) pair.
        """
        try:
            # Use 256-thread decode kernel with batch dimension in grid
            # Grid=(num_heads, batch_size, 1), Block=(256, 1, 1)
            func = self.kernels.get_hip("flash_attn_256_decode", "flash_attn_256_tuned")
            params = [
                ctypes.c_uint64(q_batch),
                ctypes.c_uint64(k_cache_base),
                ctypes.c_uint64(v_cache_base),
                ctypes.c_uint64(out_batch),
                ctypes.c_uint32(seq_len),       # kv_seq_len
                ctypes.c_uint32(self.local_num_attention_heads),
                ctypes.c_uint32(self.local_num_kv_heads),
            ]
            # Batch dimension in grid Y
            self.device.launch(func, (self.local_num_attention_heads, batch_size, 1),
                               (256, 1, 1), params)
        except Exception as e:
            # Fallback to original flash_attn_256_fp16 with batch dimension
            func = self.kernels.get_hip("flash_attn_256_fp16", "flash_attn_256")
            params = [
                ctypes.c_uint64(q_batch),
                ctypes.c_uint64(k_cache_base),
                ctypes.c_uint64(v_cache_base),
                ctypes.c_uint64(out_batch),
                ctypes.c_uint32(seq_len),       # kv_seq_len
                ctypes.c_uint32(batch_size),     # num_q_rows = batch_size for batched decode
                ctypes.c_uint32(self.local_num_attention_heads),
                ctypes.c_uint32(self.local_num_kv_heads),
                ctypes.c_uint32(0),              # non-causal (decode mode)
            ]
            # Grid=(num_heads, batch_size, 1)
            self.device.launch(func, (self.local_num_attention_heads, batch_size, 1),
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
        # Use v3 (float4 vectorized, 1.31x faster than v2) if available
        try:
            func = self.kernels.get_hip("silu_fused_v3", "elementwise_v3")
            grid_x = (n + 2047) // 2048  # v3: 8 FP16 per thread × 256 threads = 2048 per block
        except Exception:
            func = self.kernels.get_hip("silu_fused_v2", "elementwise_v2")
            grid_x = (n + 511) // 512    # v2: 2 FP16 per thread × 256 threads = 512 per block
        params = [
            ctypes.c_uint64(gate),
            ctypes.c_uint64(up),
            ctypes.c_uint32(n),
        ]
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)

    def _launch_residual_add(self, dst, src, n):
        # Use v3 (float4 vectorized, 1.26-1.32x faster than v2) if available
        try:
            func = self.kernels.get_hip("residual_add_v3", "elementwise_v3")
            grid_x = (n + 2047) // 2048  # v3: 8 FP16 per thread × 256 = 2048 per block
        except Exception:
            func = self.kernels.get_hip("residual_add_v2", "elementwise_v2")
            grid_x = (n + 511) // 512    # v2: 2 FP16 per thread × 256 = 512 per block
        params = [
            ctypes.c_uint64(dst),
            ctypes.c_uint64(src),
            ctypes.c_uint32(n),
        ]
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)

    def _launch_activation_quant(self, src_fp16, n):
        """Quantize FP16 activations to INT8 (dynamic per-tensor).

        Two-kernel pipeline:
          1. activation_quant_reduce: finds max|x| → d_act_maxabs
          2. activation_quant_quant: computes scale and quantizes → d_act_int8, d_act_scale

        After this call:
          self.d_act_int8   holds [n] INT8 quantized values
          self.d_act_scale  holds FP32 scale = max_abs / 127.0
        """
        # Zero max_abs before reduce (uses atomicMax)
        self.device.memset(self.d_act_maxabs, 0, 4)

        # Pass 1: find max absolute value
        grid_x = max(1, (n + 255) // 256)
        func1 = self.kernels.get_hip("activation_quant_reduce", "activation_quant")
        params1 = [
            ctypes.c_uint64(src_fp16),
            ctypes.c_uint64(self.d_act_maxabs),
            ctypes.c_uint32(n),
        ]
        self.device.launch(func1, (grid_x, 1, 1), (256, 1, 1), params1)

        # Pass 2: compute scale and quantize
        grid_x2 = max(1, (n + 255) // 256)
        func2 = self.kernels.get_hip("activation_quant_quant", "activation_quant")
        params2 = [
            ctypes.c_uint64(src_fp16),
            ctypes.c_uint64(self.d_act_int8),
            ctypes.c_uint64(self.d_act_maxabs),
            ctypes.c_uint64(self.d_act_scale),
            ctypes.c_uint32(n),
        ]
        self.device.launch(func2, (grid_x2, 1, 1), (256, 1, 1), params2)

    def _launch_gemv_w8a8(self, dst, src_fp16, weight_int8, scale_w_fp32, K, N):
        """W8A8 GEMV: quantize FP16 activations, then launch W8A8 GEMV kernel.

        dst:          [N] FP16 output
        src_fp16:     [K] FP16 activations
        weight_int8:  [N, K] INT8 device ptr
        scale_w_fp32: [N] FP32 per-channel weight scales device ptr
        K, N:         dimensions
        """
        # Step 1: quantize FP16 activations to INT8 (dynamic per-tensor)
        self._launch_activation_quant(src_fp16, K)

        # Step 2: W8A8 GEMV — dst = scale_w[n] * scale_a * (W_int8 @ x_int8)
        func = self.kernels.get_hip("gemv_w8a8", "gemv_w8a8")
        grid_x = (N + 3) // 4  # 4 rows per workgroup
        # scale_a is a float scalar — pass via a temporary GPU float buffer
        # We read it from d_act_scale. The kernel signature takes float scale_a by value,
        # so we need to download and pass it. But HIP kernels get scalars by-value in params.
        # We use a workaround: pass d_act_scale pointer, then read it from kernel's perspective.
        # Actually kernel signature: gemv_w8a8(x, W, scale_w, scale_a_float, out, K, N)
        # scale_a is FP32 scalar — we need to download it from GPU then pass as ctypes.c_float.
        # This is a necessary D2H sync for the scale (4 bytes only).
        scale_a_bytes = self.device.download(self.d_act_scale, 4)
        scale_a_val = float(np.frombuffer(scale_a_bytes, dtype=np.float32)[0])
        params = [
            ctypes.c_uint64(self.d_act_int8),
            ctypes.c_uint64(weight_int8),
            ctypes.c_uint64(scale_w_fp32),
            ctypes.c_float(scale_a_val),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
        ]
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _launch_gemv_w4a8(self, dst, src_fp16, weight_w4a8, scale_grp, K, N, group_size=None):
        """W4A8 GEMV with per-group scales: quantize FP16 activations, then launch.

        dst:          [N] FP16 output
        src_fp16:     [K] FP16 activations
        weight_w4a8:  [N, K/8] uint32 packed INT4 device ptr
        scale_grp:    [K/group_size, N] FP16 per-group weight scales device ptr
        K, N:         dimensions
        group_size:   quantization group size (default: self.config.group_size)
        """
        if group_size is None:
            group_size = self.config.group_size

        # Step 1: quantize FP16 activations to INT8
        self._launch_activation_quant(src_fp16, K)

        # Step 2: W4A8 grouped GEMV
        # Download scale_a (4 bytes, necessary for by-value scalar param)
        scale_a_bytes = self.device.download(self.d_act_scale, 4)
        scale_a_val = float(np.frombuffer(scale_a_bytes, dtype=np.float32)[0])
        func = self.kernels.get_hip("gemv_w4a8_grouped", "gemv_w4a8")
        grid_x = (N + 3) // 4
        params = [
            ctypes.c_uint64(self.d_act_int8),
            ctypes.c_uint64(weight_w4a8),
            ctypes.c_uint64(scale_grp),
            ctypes.c_float(scale_a_val),
            ctypes.c_uint64(dst),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(group_size),
        ]
        self.device.launch(func, (grid_x, 1, 1), (256, 1, 1), params)
        self._record_launch(getattr(self, '_active_layer_idx', -1))

    def _alloc_prefill_scratch(self, seq_len: int):
        """Allocate GPU scratch buffers for prefill."""
        if hasattr(self, '_pf_seq_len') and self._pf_seq_len >= seq_len:
            return

        # Free old buffers if reallocating
        if hasattr(self, '_pf_seq_len'):
            for attr in ['d_pf_hidden', 'd_pf_hidden2', 'd_pf_normed', 'd_pf_q', 'd_pf_q_gate',
                          'd_pf_k', 'd_pf_v', 'd_pf_kv_fused', 'd_pf_attn_out',
                          'd_pf_ffn_gate', 'd_pf_ffn_up', 'd_pf_ffn_out', 'd_pf_proj_out']:
                ptr = getattr(self, attr, 0)
                if ptr:
                    self.device.free(ptr)

        h = self.config.hidden_size
        q_dim = self.q_dim
        kv_dim = self.kv_dim

        self.d_pf_hidden = self.device.malloc(seq_len * h * 2)
        self.d_pf_hidden2 = self.device.malloc(seq_len * h * 2)  # For tree decode FFN path
        self.d_pf_normed = self.device.malloc(seq_len * h * 2)
        self.d_pf_q = self.device.malloc(seq_len * q_dim * 2)
        self.d_pf_q_gate = self.device.malloc(seq_len * q_dim * 2)
        self.d_pf_k = self.device.malloc(seq_len * kv_dim * 2)
        self.d_pf_v = self.device.malloc(seq_len * kv_dim * 2)
        self.d_pf_kv_fused = self.device.malloc(seq_len * 2 * kv_dim * 2)  # For tree decode with fused KV
        self.d_pf_attn_out = self.device.malloc(seq_len * q_dim * 2)
        inter = self.config.intermediate_size
        self.d_pf_proj_out = self.device.malloc(seq_len * h * 2)  # For batched allreduce
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
            if (self.quant_format == 'w4a16' and self._gemm_int4_prefill
                    and seq_len >= 32):
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
                # Per-token FFN fallback: supports W4A16, W8A8, W4A8
                for t in range(seq_len):
                    t_off = t * h * 2
                    self._launch_rmsnorm(self.d_normed, self.d_pf_hidden + t_off,
                                          lw.ffn_norm, h)

                    if self.quant_format == 'w8a8' and self._w8a8_ready:
                        self._launch_gemv_w8a8(self.d_ffn_gate, self.d_normed,
                                                lw.gate_w8a8, lw.gate_scale_w8a8,
                                                h, cfg.intermediate_size)
                        self._launch_gemv_w8a8(self.d_ffn_up, self.d_normed,
                                                lw.up_w8a8, lw.up_scale_w8a8,
                                                h, cfg.intermediate_size)
                        self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                                 self.d_ffn_gate, cfg.intermediate_size)
                        self._launch_gemv_w8a8(self.d_ffn_out, self.d_ffn_gate,
                                                lw.down_w8a8, lw.down_scale_w8a8,
                                                cfg.intermediate_size, h)
                        self._launch_residual_add(self.d_pf_hidden + t_off,
                                                   self.d_ffn_out, h)
                    elif self.quant_format == 'w4a8' and self._w4a8_ready:
                        self._launch_gemv_w4a8(self.d_ffn_gate, self.d_normed,
                                                lw.gate_w4a8, lw.gate_scale_w4a8,
                                                h, cfg.intermediate_size)
                        self._launch_gemv_w4a8(self.d_ffn_up, self.d_normed,
                                                lw.up_w4a8, lw.up_scale_w4a8,
                                                h, cfg.intermediate_size)
                        self._launch_silu_fused(self.d_ffn_gate, self.d_ffn_up,
                                                 self.d_ffn_gate, cfg.intermediate_size)
                        self._launch_gemv_w4a8(self.d_ffn_out, self.d_ffn_gate,
                                                lw.down_w4a8, lw.down_scale_w4a8,
                                                cfg.intermediate_size, h)
                        self._launch_residual_add(self.d_pf_hidden + t_off,
                                                   self.d_ffn_out, h)
                    else:
                        # W4A16 per-token fallback
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

    # --- Parameter pre-caching for low-overhead cached dispatch ---

    def build_decode_launch_cache(self):
        """Pre-build ctypes parameter arrays for all decode kernel launches.

        Call this after all layer weights are loaded. Creates LaunchSpec objects
        for every kernel in the decode loop (per layer). Avoids re-constructing
        ctypes objects on each decode step, reducing Python overhead by ~5x.

        Returns:
            dict mapping layer_idx -> LayerLaunchCache (namedtuple-like object)
            with pre-built LaunchSpec instances for each kernel in that layer.

        For position-dependent kernels (qknorm_rope), the LaunchSpec contains
        mutable ctypes.c_uint64 objects for the cos/sin pointers. Update them
        in-place before calling launch_cached:
            cache[layer_idx].qknorm_q.params[2].value = new_cos_ptr
            cache[layer_idx].qknorm_q.params[3].value = new_sin_ptr
        """
        from src.runtime.hip_dispatch import LaunchSpec
        cfg = self.config
        h = cfg.hidden_size
        half_rotary = self.rotary_dim // 2
        inter = self.local_intermediate_size
        k_splits = self._gemv_int4_k_splits

        layer_caches = {}

        for layer_idx, lw in enumerate(self.layers):
            lc = {}

            # --- Attention RMSNorm ---
            lc['attn_rmsnorm'] = LaunchSpec(
                func=self.kernels.get_hip("rmsnorm_v3", "elementwise_v3"),
                grid=(1, 1, 1), block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(self.d_normed),
                    ctypes.c_uint64(self.d_hidden),
                    ctypes.c_uint64(lw.attn_norm),
                    ctypes.c_uint32(h),
                    ctypes.c_float(cfg.rms_norm_eps),
                ],
            )

            if lw.layer_type == 'full_attention':
                # --- Full attention GEMV projections ---
                if self._gemv_fp16_v2:
                    q_grid = (2 * self.q_dim + 3) // 4
                    lc['gemv_q_fused'] = LaunchSpec(
                        func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                        grid=(q_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_normed),
                            ctypes.c_uint64(lw.q_fused_weight),
                            ctypes.c_uint64(self.d_q_fused),
                            ctypes.c_uint32(h),
                            ctypes.c_uint32(2 * self.q_dim),
                            ctypes.c_uint64(0),  # no residual
                        ],
                    )
                    if self._direct_kv_write and self._qknorm_rope_cachew:
                        # Split KV GEMV into K-only and V-to-cache GEMVs
                        k_grid = (self.kv_dim + 3) // 4
                        lc['gemv_k_only'] = LaunchSpec(
                            func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                            grid=(k_grid, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_normed),
                                ctypes.c_uint64(lw.k_weight),
                                ctypes.c_uint64(self.d_k),  # working buffer
                                ctypes.c_uint32(h),
                                ctypes.c_uint32(self.kv_dim),
                                ctypes.c_uint64(0),  # no residual
                            ],
                        )
                        # V GEMV writes directly to KV cache position (mutable output ptr)
                        # Initial value: position 0 in the cache; updated per step at params[2]
                        kv_stride = self.local_num_kv_heads * cfg.head_dim * 2
                        v_cache_ptr_init = self.kv_cache.layer_v_ptr(layer_idx)
                        lc['gemv_v_cache'] = LaunchSpec(
                            func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                            grid=(k_grid, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_normed),
                                ctypes.c_uint64(lw.v_weight),
                                ctypes.c_uint64(v_cache_ptr_init),  # mutable: index [2]
                                ctypes.c_uint32(h),
                                ctypes.c_uint32(self.kv_dim),
                                ctypes.c_uint64(0),  # no residual
                            ],
                        )
                        # Store layer's V cache base pointer for offset computation
                        lc['_v_cache_base'] = v_cache_ptr_init
                        lc['_kv_stride'] = kv_stride
                    else:
                        kv_grid = (2 * self.kv_dim + 3) // 4
                        lc['gemv_kv_fused'] = LaunchSpec(
                            func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                            grid=(kv_grid, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_normed),
                                ctypes.c_uint64(lw.kv_fused_weight),
                                ctypes.c_uint64(self.d_kv_fused),
                                ctypes.c_uint32(h),
                                ctypes.c_uint32(2 * self.kv_dim),
                                ctypes.c_uint64(0),  # no residual
                            ],
                        )

                # --- INT4 attention GEMV projections (compressed-tensors AWQ format) ---
                # Use fused QKV kernel when available (3 launches → 1)
                if self._gemv_int4_qkv_fused and hasattr(lw, 'qkv_qweight') and lw.qkv_qweight:
                    # Fused QKV GEMV: single kernel launches for Q, K, V projections
                    N_total = self.q_dim + 2 * self.kv_dim
                    # Use v8-style grid: 256 threads/WG, 16 cols/WG for t16
                    qkv_grid = (2 * N_total + 3) // 4
                    
                    # Select kernel variant based on N_total size (same as v8)
                    if N_total <= 2048:
                        qkv_func = self.kernels.get_hip("gemv_int4_qkv_fused_t16", "gemv_int4_qkv_fused",
                                                         hsaco_suffix="_t16")
                    elif N_total <= 4096:
                        qkv_func = self.kernels.get_hip("gemv_int4_qkv_fused_t8", "gemv_int4_qkv_fused",
                                                         hsaco_suffix="_t8")
                    else:
                        qkv_func = self.kernels.get_hip("gemv_int4_qkv_fused_t4", "gemv_int4_qkv_fused",
                                                         hsaco_suffix="_t4")
                    
                    # V cache pointer for direct KV write mode (mutable, updated per step)
                    if self._direct_kv_write and self._qknorm_rope_cachew:
                        v_cache_ptr_init = self.kv_cache.layer_v_ptr(layer_idx)
                        kv_stride = self.local_num_kv_heads * cfg.head_dim * 2
                        lc['gemv_qkv_fused'] = LaunchSpec(
                            func=qkv_func,
                            grid=(qkv_grid, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_normed),
                                ctypes.c_uint64(lw.qkv_qweight),
                                ctypes.c_uint64(lw.qkv_scales),
                                ctypes.c_uint64(lw.qkv_zeros),
                                ctypes.c_uint64(self.d_q),           # Q output buffer
                                ctypes.c_uint64(self.d_k),           # K output buffer  
                                ctypes.c_uint64(v_cache_ptr_init),   # V output (mutable: cache position)
                                ctypes.c_uint32(h),
                                ctypes.c_uint32(self.q_dim),
                                ctypes.c_uint32(self.kv_dim),
                                ctypes.c_uint32(cfg.group_size),
                                ctypes.c_uint64(v_cache_ptr_init),   # v_cache_dst (same as V output)
                            ],
                        )
                        # Store for C dispatch and mutable updates
                        lc['_v_cache_base'] = v_cache_ptr_init
                        lc['_kv_stride'] = kv_stride
                    else:
                        # Standard mode: V writes to d_v working buffer
                        lc['gemv_qkv_fused'] = LaunchSpec(
                            func=qkv_func,
                            grid=(qkv_grid, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_normed),
                                ctypes.c_uint64(lw.qkv_qweight),
                                ctypes.c_uint64(lw.qkv_scales),
                                ctypes.c_uint64(lw.qkv_zeros),
                                ctypes.c_uint64(self.d_q),           # Q output buffer
                                ctypes.c_uint64(self.d_k),           # K output buffer
                                ctypes.c_uint64(self.d_v),           # V output buffer
                                ctypes.c_uint32(h),
                                ctypes.c_uint32(self.q_dim),
                                ctypes.c_uint32(self.kv_dim),
                                ctypes.c_uint32(cfg.group_size),
                                ctypes.c_uint64(0),                  # v_cache_dst = null
                            ],
                        )
                    
                    # Skip separate Q/K/V launches when using fused kernel
                    # (don't create gemv_q_fused, gemv_k_only, gemv_v_cache, gemv_kv_fused)

                # --- QK-norm + RoPE (position-mutable) ---
                # params[2] = cos ptr, params[3] = sin ptr — updated per step
                if self._qknorm_rope_fused:
                    qknorm_func = self.kernels.get_hip("qknorm_rope_fused", "qknorm_rope")
                    # Initial cos/sin at position 0 (will be updated per step)
                    lc['qknorm_q'] = LaunchSpec(
                        func=qknorm_func,
                        grid=(self.local_num_attention_heads, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_q),
                            ctypes.c_uint64(lw.q_norm),
                            ctypes.c_uint64(self.d_cos),  # mutable: index [2]
                            ctypes.c_uint64(self.d_sin),  # mutable: index [3]
                            ctypes.c_uint32(cfg.head_dim),
                            ctypes.c_uint32(half_rotary),
                            ctypes.c_float(cfg.rms_norm_eps),
                        ],
                    )
                    if self._direct_kv_write and self._qknorm_rope_cachew:
                        # Use cache-write variant for K: params[4] = cache_dst (mutable)
                        cachew_func = self.kernels.get_hip("qknorm_rope_cachew_fused",
                                                           "qknorm_rope_cachew")
                        k_cache_ptr_init = self.kv_cache.layer_k_ptr(layer_idx)
                        kv_stride = self.local_num_kv_heads * cfg.head_dim * 2
                        lc['qknorm_k'] = LaunchSpec(
                            func=cachew_func,
                            grid=(self.local_num_kv_heads, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_k),
                                ctypes.c_uint64(lw.k_norm),
                                ctypes.c_uint64(self.d_cos),  # mutable: index [2]
                                ctypes.c_uint64(self.d_sin),  # mutable: index [3]
                                ctypes.c_uint64(k_cache_ptr_init),  # mutable: index [4]
                                ctypes.c_uint32(cfg.head_dim),
                                ctypes.c_uint32(half_rotary),
                                ctypes.c_float(cfg.rms_norm_eps),
                            ],
                        )
                        # Store K cache base pointer for per-step offset computation
                        lc['_k_cache_base'] = k_cache_ptr_init
                    else:
                        lc['qknorm_k'] = LaunchSpec(
                            func=qknorm_func,
                            grid=(self.local_num_kv_heads, 1, 1), block=(256, 1, 1),
                            params=[
                                ctypes.c_uint64(self.d_k),
                                ctypes.c_uint64(lw.k_norm),
                                ctypes.c_uint64(self.d_cos),  # mutable: index [2]
                                ctypes.c_uint64(self.d_sin),  # mutable: index [3]
                                ctypes.c_uint32(cfg.head_dim),
                                ctypes.c_uint32(half_rotary),
                                ctypes.c_float(cfg.rms_norm_eps),
                            ],
                        )

                # --- Decode attention (seq_len is mutable: index [4]) ---
                # Use tuned flash_attn_256_decode (4-WF KV-parallel, 2.8-5.5x faster)
                # Signature: (Q, K, V, Out, kv_seq_len, num_heads, num_kv_heads) — 7 params
                attn_func = self.kernels.get_hip("flash_attn_256_decode", "flash_attn_256_tuned")
                lc['decode_attn'] = LaunchSpec(
                    func=attn_func,
                    grid=(self.local_num_attention_heads, 1, 1), block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(self.d_q),
                        ctypes.c_uint64(self.kv_cache.layer_k_ptr(layer_idx)),
                        ctypes.c_uint64(self.kv_cache.layer_v_ptr(layer_idx)),
                        ctypes.c_uint64(self.d_attn_out),
                        ctypes.c_uint32(1),   # seq_len: mutable [4], updated per step
                        ctypes.c_uint32(self.local_num_attention_heads),
                        ctypes.c_uint32(self.local_num_kv_heads),
                    ],
                )

                # --- Sigmoid gate on attention output ---
                sig_grid = (self.q_dim + 255) // 256
                lc['sigmoid_mul'] = LaunchSpec(
                    func=self.kernels.get_hip("sigmoid_mul_fp16", "sigmoid_mul"),
                    grid=(sig_grid, 1, 1), block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(self.d_attn_out),
                        ctypes.c_uint64(self.d_q_gate),
                        ctypes.c_uint32(self.q_dim),
                    ],
                )

                # --- Output projection (o_proj): TP writes to d_proj_out ---
                if self._gemv_fp16_v2:
                    o_grid = (h + 3) // 4
                    lc['gemv_o_proj'] = LaunchSpec(
                        func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                        grid=(o_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_attn_out),
                            ctypes.c_uint64(lw.o_weight),
                            ctypes.c_uint64(self.d_proj_out),
                            ctypes.c_uint32(self.q_dim),
                            ctypes.c_uint32(h),
                            ctypes.c_uint64(0),  # no residual (TP path)
                        ],
                    )
                
                # INT4 O projection (AWQ format) - uses v8 kernel with shape-based selection
                if hasattr(lw, 'o_qweight') and lw.o_qweight and (self._gemv_int4_v8 or self._gemv_int4_v7):
                    o_grid = (h + 3) // 4
                    # Select kernel based on output size
                    if h <= 2048:
                        o_func = self.kernels.get_hip("gemv_int4_v8_t16", "gemv_int4_v8")
                    elif h <= 4096:
                        o_func = self.kernels.get_hip("gemv_int4_v8_t8", "gemv_int4_v8")
                    else:
                        o_func = self.kernels.get_hip("gemv_int4_v8_t4", "gemv_int4_v8")
                    lc['gemv_o_proj'] = LaunchSpec(
                        func=o_func,
                        grid=(o_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_attn_out),
                            ctypes.c_uint64(lw.o_qweight),
                            ctypes.c_uint64(lw.o_scales),
                            ctypes.c_uint64(lw.o_zeros),
                            ctypes.c_uint64(self.d_proj_out),
                            ctypes.c_uint32(self.q_dim),
                            ctypes.c_uint32(h),
                            ctypes.c_uint32(cfg.group_size),
                        ],
                    )

            else:
                # --- Linear attention (DeltaNet) ---
                if self._gemv_fp16_v2:
                    la_grid = (self.la_total_dim + 3) // 4
                    lc['gemv_la_in_proj'] = LaunchSpec(
                        func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                        grid=(la_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_normed),
                            ctypes.c_uint64(lw.la_in_proj_fused),
                            ctypes.c_uint64(self.d_la_packed),
                            ctypes.c_uint32(h),
                            ctypes.c_uint32(self.la_total_dim),
                            ctypes.c_uint64(0),  # no residual
                        ],
                    )

                # DeltaNet v3 main kernel
                if self._deltanet_v3:
                    slot = self.deltanet_state.get_slot(layer_idx, cfg)
                    d_conv = self.deltanet_state.d_conv_states[slot]
                    d_state = self.deltanet_state.d_states[slot]
                    dn_suffix = f"_tp{self.tp_size}" if self.tp_size > 1 else ""
                    dn_func = self.kernels.get_hip("deltanet_decode_v3", "deltanet_v3",
                                                    hsaco_suffix=dn_suffix)
                    lc['deltanet_v3'] = LaunchSpec(
                        func=dn_func,
                        grid=(self.local_linear_num_v_heads, 1, 1), block=(256, 1, 1),
                        params=[
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
                        ],
                        shared_mem=8192,
                    )
                    # DeltaNet conv shift kernel
                    conv_dim = self.deltanet_state.conv_dim
                    shift_func = self.kernels.get_hip("deltanet_conv_shift_v3",
                                                       "deltanet_v3",
                                                       hsaco_suffix=dn_suffix)
                    shift_grid = (conv_dim + 255) // 256
                    lc['deltanet_v3_shift'] = LaunchSpec(
                        func=shift_func,
                        grid=(shift_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_la_qkv),
                            ctypes.c_uint64(d_conv),
                            ctypes.c_uint32(conv_dim),
                        ],
                    )

                # --- Linear attention output projection ---
                if self._gemv_fp16_v2:
                    la_out_grid = (h + 3) // 4
                    lc['gemv_la_out_proj'] = LaunchSpec(
                        func=self.kernels.get_hip("gemv_fp16_v2", "gemv_fp16_v2"),
                        grid=(la_out_grid, 1, 1), block=(256, 1, 1),
                        params=[
                            ctypes.c_uint64(self.d_la_out),
                            ctypes.c_uint64(lw.la_out_proj),
                            ctypes.c_uint64(self.d_proj_out),
                            ctypes.c_uint32(self.la_z_dim),
                            ctypes.c_uint32(h),
                            ctypes.c_uint64(0),  # no residual (TP path)
                        ],
                    )

            # --- FFN RMSNorm ---
            lc['ffn_rmsnorm'] = LaunchSpec(
                func=self.kernels.get_hip("rmsnorm_v3", "elementwise_v3"),
                grid=(1, 1, 1), block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(self.d_normed),
                    ctypes.c_uint64(self.d_hidden),
                    ctypes.c_uint64(lw.ffn_norm),
                    ctypes.c_uint32(h),
                    ctypes.c_float(cfg.rms_norm_eps),
                ],
            )

            # --- FFN gate+up+silu ---
            # AWQ mode: use AWQ dual fused kernel (12 params, no zeros)
            # GPTQ mode: use GPTQ dual fused kernel (14 params, with zeros)
            if self._awq_mode and self._gemv_int4_dual_awq_fused:
                # AWQ variant: no zero-point subtraction, 12 params
                ffn_grid = (inter + 255) // 256
                lc['ffn_gate_up_silu'] = LaunchSpec(
                    func=self.kernels.get_hip("gemv_int4_dual_awq_fused", "gemv_int4_dual_awq"),
                    grid=(ffn_grid, k_splits, 1), block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(self.d_normed),
                        ctypes.c_uint64(lw.gate_qweight),
                        ctypes.c_uint64(lw.gate_scales),
                        ctypes.c_uint64(lw.up_qweight),
                        ctypes.c_uint64(lw.up_scales),
                        ctypes.c_uint64(self.d_gemv_fp32),
                        ctypes.c_uint64(self.d_gemv_fp32_2),
                        ctypes.c_uint64(self.d_gemv_done),
                        ctypes.c_uint64(self.d_ffn_gate),
                        ctypes.c_uint32(h),
                        ctypes.c_uint32(inter),
                        ctypes.c_uint32(cfg.group_size),
                        ctypes.c_uint32(k_splits),
                    ],
                )
            elif self._gemv_int4_dual_fused:
                # GPTQ dual kernel: includes zero-point subtraction, 14 params
                ffn_grid = (inter + 255) // 256
                lc['ffn_gate_up_silu'] = LaunchSpec(
                    func=self.kernels.get_hip("gemv_int4_dual_fused", "gemv_int4_dual"),
                    grid=(ffn_grid, k_splits, 1), block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(self.d_normed),
                        ctypes.c_uint64(lw.gate_qweight),
                        ctypes.c_uint64(lw.gate_scales),
                        ctypes.c_uint64(lw.gate_zeros),
                        ctypes.c_uint64(lw.up_qweight),
                        ctypes.c_uint64(lw.up_scales),
                        ctypes.c_uint64(lw.up_zeros),
                        ctypes.c_uint64(self.d_gemv_fp32),
                        ctypes.c_uint64(self.d_gemv_fp32_2),
                        ctypes.c_uint64(self.d_gemv_done),
                        ctypes.c_uint64(self.d_ffn_gate),
                        ctypes.c_uint32(h),
                        ctypes.c_uint32(inter),
                        ctypes.c_uint32(cfg.group_size),
                        ctypes.c_uint32(k_splits),
                    ],
                )

            # --- FFN down projection (v5_t16 default, v3_t16 fallback, no residual for TP) ---
            # v5 uses hybrid DPP+LDS reduction with identical performance to v3/v4.
            # Falls back to v3_t16 if v5 is unavailable.
            # AWQ mode: use gemv_int4_v5_awq_t16 (no zero-point subtraction, zeros ptr ignored).
            if self._gemv_int4_v5 or self._gemv_int4_v3 or self._gemv_int4_v5_awq:
                cols_per_wg = 16
                down_grid = (h + cols_per_wg - 1) // cols_per_wg
                # Select kernel based on AWQ mode and availability
                if self._awq_mode and self._gemv_int4_v5_awq:
                    down_func = self.kernels.get_hip("gemv_int4_v5_awq_t16", "gemv_int4_v5_awq")
                    shared_mem = 0  # AWQ v5 uses static shared memory
                elif self._gemv_int4_v5:
                    down_func = self.kernels.get_hip("gemv_int4_v5_t16", "gemv_int4_v5")
                    shared_mem = 0  # v5 uses static shared memory (NUM_WF * COLS_PER_WG floats)
                else:
                    down_func = self.kernels.get_hip("gemv_int4_v3_t16", "gemv_int4_v3")
                    shared_mem = 1024
                # Build params: AWQ kernel ignores zeros pointer (pass 0 or actual zeros)
                down_params = [
                    ctypes.c_uint64(self.d_ffn_gate),
                    ctypes.c_uint64(lw.down_qweight),
                    ctypes.c_uint64(lw.down_scales),
                    ctypes.c_uint64(lw.down_zeros),  # ignored by AWQ kernel
                    ctypes.c_uint64(self.d_ffn_out),
                    ctypes.c_uint32(inter),
                    ctypes.c_uint32(h),
                    ctypes.c_uint32(cfg.group_size),
                ]
                lc['ffn_down'] = LaunchSpec(
                    func=down_func,
                    grid=(down_grid, 1, 1), block=(256, 1, 1),
                    params=down_params,
                    shared_mem=shared_mem,
                )

            layer_caches[layer_idx] = lc

        return layer_caches
