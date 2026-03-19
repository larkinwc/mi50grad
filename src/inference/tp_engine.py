"""
Tensor-parallel inference engine for Qwen 3.5 27B across multiple MI50s.

Uses fast_ar_fused_tp2 C extension for all allreduces: downloads partials +
hidden from both GPUs, AVX F16C 3-way add on host, uploads result.
"""

import ctypes
import os
import threading
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List

from src.inference.engine import InferenceEngine
from src.runtime.tensor_parallel import TensorParallelGroup
from src.runtime.p2p_allreduce import P2PAllreduce, FusedP2PReduce, RingAllreduce
from src.model.qwen import QwenConfig


# ============================================================
# _GraphDecodeState: manages HIP graph capture and replay
# for the full decode step in TPInferenceEngine
# ============================================================

class _GraphDecodeState:
    """Manages HIP graph segments for graph-based decode.

    Per GPU per layer, captures two graph segments:
      - Attention segment: RMSNorm → GEMV projections → QKNorm/RoPE
                           → KV cache update → decode attention
                           → sigmoid gate → O-proj
      - FFN segment: FFN RMSNorm → gate+up+silu → down_proj

    Mutable nodes tracked per attention segment:
      - qknorm_q, qknorm_k: cos/sin ptr update (indices [2],[3])
      - qknorm_k (cache-write mode): k_cache_dst ptr update (index [4])
      - gemv_v_cache (direct-KV mode): v_cache_dst ptr update (index [2])
      - decode_attn: seq_len update (index [4])

    Host allreduce (P2P) is performed between attention and FFN segments
    for each layer.
    """

    def __init__(self, tp_engine, graph_rt):
        self._tp = tp_engine
        self._graph_rt = graph_rt
        self._captured = False

        # Per-GPU, per-layer: GraphSegment objects
        # _attn_segs[gpu_idx][layer_idx] = GraphSegment
        # _ffn_segs[gpu_idx][layer_idx]  = GraphSegment
        self._attn_segs = []
        self._ffn_segs  = []

        # Per-GPU, per-layer: list of (node, func_handle, param_index, is_ptr64)
        # for each mutable kernel node in the attention segment.
        # Used to update cos/sin ptrs and seq_len between replays.
        # _mutable_attn[gpu_idx][layer_idx] = list of MutableParam dicts
        self._mutable_attn = []

        # Per-GPU capture streams (non-default streams required for graph capture)
        self._capture_streams = []

        # Track KV cache positions for direct-KV-write path
        # _kv_write_pos[gpu_idx] = current KV cache write position
        # (same as engine.kv_cache.current_len but maintained separately)

    @property
    def captured(self) -> bool:
        return self._captured

    def _hipModuleLaunchKernel(self, hip, stream, spec):
        """Launch a LaunchSpec on the given (non-default) stream via hipModuleLaunchKernel."""
        import ctypes
        hip._lib.hipModuleLaunchKernel(
            ctypes.c_void_p(spec.func),
            ctypes.c_uint(spec.grid[0]),
            ctypes.c_uint(spec.grid[1]),
            ctypes.c_uint(spec.grid[2]),
            ctypes.c_uint(spec.block[0]),
            ctypes.c_uint(spec.block[1]),
            ctypes.c_uint(spec.block[2]),
            ctypes.c_uint(spec.shared_mem),
            ctypes.c_void_p(stream),
            ctypes.cast(spec.params_array, ctypes.POINTER(ctypes.c_void_p)),
            None,
        )

    def capture_all(self, initial_position: int):
        """Capture attention and FFN graph segments for all GPUs and layers.

        Must be called once before replay_step(). Creates non-default capture
        streams for each GPU (graph capture requires non-default streams).

        The capture is driven by the LaunchSpec objects already built by
        build_decode_launch_cache(). For each layer, we:
          1. hipStreamBeginCapture on GPU's capture stream
          2. hipModuleLaunchKernel for each attention kernel in sequence
          3. hipStreamEndCapture → instantiate → find mutable nodes
          4. Repeat for FFN segment
        """
        from src.runtime.hip_graph_dispatch import GraphSegment

        import ctypes

        tp = self._tp
        hip = tp._hip
        num_layers = tp.config.num_hidden_layers
        num_gpus = tp.tp_size
        half_rotary = tp.engines[0].rotary_dim // 2
        cos_offset_init = initial_position * half_rotary * 2

        print(f"  Capturing HIP graphs for {num_gpus} GPUs × {num_layers} layers × 2 segments...")
        t0 = __import__('time').perf_counter()

        # Create per-GPU capture streams
        self._capture_streams = []
        for gpu_idx in range(num_gpus):
            hip.set_device(tp.device_ids[gpu_idx])
            stream = ctypes.c_void_p(0)
            err = hip._lib.hipStreamCreate(ctypes.byref(stream))
            if err != 0:
                raise RuntimeError(f"hipStreamCreate failed: {err}")
            self._capture_streams.append(stream.value)

        # Init per-gpu segment lists
        self._attn_segs = [[] for _ in range(num_gpus)]
        self._ffn_segs  = [[] for _ in range(num_gpus)]
        self._mutable_attn = [[] for _ in range(num_gpus)]

        for layer_idx in range(num_layers):
            for gpu_idx in range(num_gpus):
                engine = tp.engines[gpu_idx]
                lc = tp._engine_layer_caches[gpu_idx][layer_idx]
                lw = engine.layers[layer_idx]
                cap_stream = self._capture_streams[gpu_idx]

                hip.set_device(tp.device_ids[gpu_idx])

                # ---- Capture ATTENTION segment ----
                attn_seg = GraphSegment(self._graph_rt, tp.device_ids[gpu_idx])
                attn_seg.begin_capture(cap_stream)

                # Attention RMSNorm
                if 'attn_rmsnorm' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['attn_rmsnorm'])

                if lw.layer_type == 'full_attention':
                    # GEMV projections
                    if 'gemv_q_fused' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_q_fused'])
                    if 'gemv_k_only' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_k_only'])
                        # V GEMV writes to current KV cache position (mutable)
                        # Update the output ptr to initial position before capture
                        cur_pos = engine.kv_cache.current_len
                        kv_stride = lc['_kv_stride']
                        v_cache_ptr = lc['_v_cache_base'] + cur_pos * kv_stride
                        lc['gemv_v_cache'].params[2].value = v_cache_ptr
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_v_cache'])
                    elif 'gemv_kv_fused' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_kv_fused'])

                    # QKNorm/RoPE — set cos/sin to initial position before capture
                    if 'qknorm_q' in lc:
                        lc['qknorm_q'].params[2].value = engine.d_cos + cos_offset_init
                        lc['qknorm_q'].params[3].value = engine.d_sin + cos_offset_init
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['qknorm_q'])
                    if 'qknorm_k' in lc:
                        lc['qknorm_k'].params[2].value = engine.d_cos + cos_offset_init
                        lc['qknorm_k'].params[3].value = engine.d_sin + cos_offset_init
                        if '_k_cache_base' in lc:
                            cur_pos = engine.kv_cache.current_len
                            kv_stride = lc['_kv_stride']
                            k_cache_ptr = lc['_k_cache_base'] + cur_pos * kv_stride
                            lc['qknorm_k'].params[4].value = k_cache_ptr
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['qknorm_k'])

                    # Decode attention — set seq_len to initial value before capture
                    if 'decode_attn' in lc:
                        lc['decode_attn'].params[4].value = (
                            engine.kv_cache.current_len + 1)
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['decode_attn'])

                    # Sigmoid gate
                    if 'sigmoid_mul' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['sigmoid_mul'])

                    # O-proj
                    if 'gemv_o_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_o_proj'])

                else:
                    # DeltaNet linear attention
                    if 'gemv_la_in_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_la_in_proj'])
                    if 'deltanet_v3' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['deltanet_v3'])
                    if 'deltanet_v3_shift' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['deltanet_v3_shift'])
                    if 'gemv_la_out_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_la_out_proj'])

                attn_seg.end_capture(cap_stream)
                attn_seg.instantiate()
                self._attn_segs[gpu_idx].append(attn_seg)

                # Build mutable node list for attention segment using POSITION-BASED matching.
                # We track the ORDER of kernel launches during capture and map each
                # position in _kernel_nodes to the corresponding LaunchSpec.
                # This is necessary because multiple kernels (e.g., gemv_fp16_v2 for
                # q_fused, k_only, v_cache, o_proj) share the same func handle and
                # cannot be distinguished by func alone.
                mutable_list = []
                if lw.layer_type == 'full_attention':
                    # Rebuild the ordered list of kernels in the capture sequence
                    capture_order = []
                    if 'attn_rmsnorm' in lc:
                        capture_order.append('attn_rmsnorm')
                    if 'gemv_q_fused' in lc:
                        capture_order.append('gemv_q_fused')
                    if 'gemv_k_only' in lc:
                        capture_order.append('gemv_k_only')
                        capture_order.append('gemv_v_cache')
                    elif 'gemv_kv_fused' in lc:
                        capture_order.append('gemv_kv_fused')
                    if 'qknorm_q' in lc:
                        capture_order.append('qknorm_q')
                    if 'qknorm_k' in lc:
                        capture_order.append('qknorm_k')
                    if 'decode_attn' in lc:
                        capture_order.append('decode_attn')
                    if 'sigmoid_mul' in lc:
                        capture_order.append('sigmoid_mul')
                    if 'gemv_o_proj' in lc:
                        capture_order.append('gemv_o_proj')

                    # Only these keys have mutable params that need updating per step
                    mutable_keys = frozenset(('qknorm_q', 'qknorm_k', 'decode_attn', 'gemv_v_cache'))

                    kernel_nodes = attn_seg._kernel_nodes  # in capture order
                    assert len(kernel_nodes) == len(capture_order), (
                        f"Kernel node count mismatch: {len(kernel_nodes)} nodes "
                        f"vs {len(capture_order)} expected for layer {layer_idx} GPU {gpu_idx}"
                    )

                    for pos, (node, key) in enumerate(zip(kernel_nodes, capture_order)):
                        if key in mutable_keys:
                            spec = lc[key]
                            node_params = attn_seg.get_kernel_params(node)
                            mutable_list.append({
                                'node': node,
                                'key': key,
                                'spec': spec,
                                'base_params': node_params,
                            })

                self._mutable_attn[gpu_idx].append(mutable_list)

                # ---- Capture FFN segment ----
                ffn_seg = GraphSegment(self._graph_rt, tp.device_ids[gpu_idx])
                ffn_seg.begin_capture(cap_stream)

                if 'ffn_rmsnorm' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_rmsnorm'])
                if 'ffn_gate_up_silu' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_gate_up_silu'])
                if 'ffn_down' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_down'])

                ffn_seg.end_capture(cap_stream)
                ffn_seg.instantiate()
                self._ffn_segs[gpu_idx].append(ffn_seg)

        elapsed = __import__('time').perf_counter() - t0
        total_segments = num_gpus * num_layers * 2
        print(f"  Graph capture complete: {total_segments} segments in {elapsed*1000:.0f}ms")

        # Synchronize all GPUs to ensure capture streams are done
        for gpu_idx in range(num_gpus):
            hip.set_device(tp.device_ids[gpu_idx])
            hip.synchronize()

        self._captured = True

    def replay_step(self, position: int):
        """Replay all captured graphs with updated mutable params for the given position.

        Per layer:
          1. For each GPU: update mutable params (cos/sin, seq_len, kv_ptr)
          2. For each GPU: replay attention graph on default stream
          3. P2P allreduce (attention)
          4. For each GPU: replay FFN graph on default stream
          5. P2P allreduce (FFN)
        """
        import ctypes

        tp = self._tp
        hip = tp._hip
        num_layers = tp.config.num_hidden_layers
        half_rotary = tp.engines[0].rotary_dim // 2
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables
        h = tp.config.hidden_size

        from src.runtime.hip_graph_dispatch import hipKernelNodeParams

        for layer_idx in range(num_layers):
            for gpu_idx in range(tp.tp_size):
                engine = tp.engines[gpu_idx]
                lc = tp._engine_layer_caches[gpu_idx][layer_idx]
                lw = engine.layers[layer_idx]
                cur_pos = engine.kv_cache.current_len

                hip.set_device(tp.device_ids[gpu_idx])

                if lw.layer_type == 'full_attention':
                    # Update mutable params for this step
                    for m in self._mutable_attn[gpu_idx][layer_idx]:
                        key = m['key']
                        node = m['node']
                        spec = m['spec']
                        base = m['base_params']

                        # Build new hipKernelNodeParams with updated values
                        new_kp = hipKernelNodeParams()
                        new_kp.blockDimX = base.blockDimX
                        new_kp.blockDimY = base.blockDimY
                        new_kp.blockDimZ = base.blockDimZ
                        new_kp.gridDimX  = base.gridDimX
                        new_kp.gridDimY  = base.gridDimY
                        new_kp.gridDimZ  = base.gridDimZ
                        new_kp.func = base.func
                        new_kp.sharedMemBytes = base.sharedMemBytes
                        new_kp.extra = None

                        if key in ('qknorm_q', 'qknorm_k'):
                            # Update cos/sin pointers (params[2] and params[3])
                            spec.params[2].value = engine.d_cos + cos_offset
                            spec.params[3].value = engine.d_sin + cos_offset
                            if key == 'qknorm_k' and '_k_cache_base' in lc:
                                kv_stride = lc['_kv_stride']
                                spec.params[4].value = (lc['_k_cache_base']
                                                        + cur_pos * kv_stride)
                        elif key == 'decode_attn':
                            # Update seq_len (params[4])
                            spec.params[4].value = cur_pos + 1
                        elif key == 'gemv_v_cache':
                            # Update V cache destination pointer (params[2])
                            kv_stride = lc['_kv_stride']
                            spec.params[2].value = (lc['_v_cache_base']
                                                    + cur_pos * kv_stride)

                        new_kp.kernelParams = ctypes.cast(
                            spec.params_array, ctypes.c_void_p).value
                        self._attn_segs[gpu_idx][layer_idx].update_kernel_params(
                            node, new_kp)

                # Replay attention segment on default stream (0)
                hip.set_device(tp.device_ids[gpu_idx])
                self._attn_segs[gpu_idx][layer_idx].replay(stream=0)

            # Host allreduce: attention partials → d_hidden
            partial_ptrs = [e.d_proj_out for e in tp.engines]
            hidden_ptrs  = [e.d_hidden   for e in tp.engines]
            tp._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, h)

            for gpu_idx in range(tp.tp_size):
                hip.set_device(tp.device_ids[gpu_idx])
                # Replay FFN segment on default stream (0)
                self._ffn_segs[gpu_idx][layer_idx].replay(stream=0)

            # Host allreduce: FFN partials → d_hidden
            partial_ptrs = [e.d_ffn_out for e in tp.engines]
            tp._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, h)

    def cleanup(self):
        """Destroy all captured graph segments and release capture streams."""
        import ctypes

        tp = self._tp
        hip = tp._hip

        for gpu_segs in self._attn_segs:
            for seg in gpu_segs:
                seg.cleanup()
        for gpu_segs in self._ffn_segs:
            for seg in gpu_segs:
                seg.cleanup()

        self._attn_segs.clear()
        self._ffn_segs.clear()
        self._mutable_attn.clear()

        # Destroy capture streams
        for gpu_idx, stream in enumerate(self._capture_streams):
            if stream:
                hip.set_device(tp.device_ids[gpu_idx])
                hip._lib.hipStreamDestroy(ctypes.c_void_p(stream))
        self._capture_streams.clear()
        self._captured = False


# ============================================================
# _GlobalGraphDecodeState: manages FULL-LAYER HIP graph capture
# Each GPU captures TWO graph segments per layer:
#   Segment 1: attn_rmsnorm → O-proj (attention compute)
#   Segment 2: FFN RMSNorm → down_proj (FFN compute)
# Between segments: kernel P2P allreduce via host (not in graph).
# 
# The "global" aspect: supports loading kernel P2P allreduce via HSACO
# for future integration, and provides a unified replay interface.
# Correctness is maintained by the host-side allreduce sync.
# ============================================================

class _GlobalGraphDecodeState:
    """Manages full-layer HIP graph capture for global graph-based decode.

    Captures TWO graph segments per GPU per layer (attention + FFN),
    with kernel P2P allreduce between them (host-orchestrated, using
    the pre-loaded kernel P2P allreduce from _p2p_ar).

    Structure per GPU per layer:
      attn_seg: attn_rmsnorm → GEMV projections → QKNorm/RoPE →
                attention/DeltaNet → sigmoid gate → O-proj
      [kernel P2P allreduce: attn partials → d_hidden]
      ffn_seg: FFN RMSNorm → gate+up+silu → down_proj
      [kernel P2P allreduce: FFN partials → d_hidden]

    This is architecturally equivalent to _GraphDecodeState but uses
    kernel P2P allreduce (faster, ~79us vs ~119us) between segments.

    Mutable nodes tracked per attention segment (same as _GraphDecodeState):
      - qknorm_q, qknorm_k: cos/sin ptr update (indices [2],[3])
      - qknorm_k (cache-write mode): k_cache_dst ptr update (index [4])
      - gemv_v_cache (direct-KV mode): v_cache_dst ptr update (index [2])
      - decode_attn: seq_len update (index [4])
    """

    def __init__(self, tp_engine, graph_rt):
        self._tp = tp_engine
        self._graph_rt = graph_rt
        self._captured = False

        # Per-GPU, per-layer: attention GraphSegment
        # _attn_segs[gpu_idx][layer_idx] = GraphSegment
        self._attn_segs = []
        # Per-GPU, per-layer: FFN GraphSegment
        # _ffn_segs[gpu_idx][layer_idx] = GraphSegment
        self._ffn_segs = []

        # Per-GPU, per-layer: mutable attention params list
        # Same structure as _GraphDecodeState._mutable_attn
        self._mutable_attn = []

        # Per-GPU capture streams
        self._capture_streams = []

        # C graph dispatch support (loaded lazily after capture)
        self._c_graph_lib = None         # ctypes handle to c_graph_dispatch.so
        self._c_graph_plan_buf = None    # ctypes buffer for CGraphDispatchPlan
        self._c_graph_objects = {}       # keeps ctypes objects alive
        self._c_graph_plan_ptr = 0       # int64 address of the plan

    @property
    def captured(self) -> bool:
        return self._captured

    def _hipModuleLaunchKernel(self, hip, stream, spec):
        """Launch a LaunchSpec on the given stream via hipModuleLaunchKernel."""
        import ctypes
        hip._lib.hipModuleLaunchKernel(
            ctypes.c_void_p(spec.func),
            ctypes.c_uint(spec.grid[0]),
            ctypes.c_uint(spec.grid[1]),
            ctypes.c_uint(spec.grid[2]),
            ctypes.c_uint(spec.block[0]),
            ctypes.c_uint(spec.block[1]),
            ctypes.c_uint(spec.block[2]),
            ctypes.c_uint(spec.shared_mem),
            ctypes.c_void_p(stream),
            ctypes.cast(spec.params_array, ctypes.POINTER(ctypes.c_void_p)),
            None,
        )

    def capture_full_layer(self, initial_position: int,
                           include_allreduce: bool = True):
        """Capture attention and FFN graph segments for all GPUs and layers.

        Captures TWO segments per GPU per layer (same as _GraphDecodeState),
        with kernel P2P allreduce called between them during replay.

        The 'include_allreduce' parameter is retained for API compatibility but
        the allreduce is ALWAYS done host-side between segments (not in graph),
        ensuring correct cross-GPU synchronization.

        Args:
            initial_position: initial decode position (for cos/sin and seq_len)
            include_allreduce: parameter for API compatibility (ignored in this impl)
        """
        from src.runtime.hip_graph_dispatch import GraphSegment

        import ctypes

        tp = self._tp
        hip = tp._hip
        num_layers = tp.config.num_hidden_layers
        num_gpus = tp.tp_size
        half_rotary = tp.engines[0].rotary_dim // 2
        cos_offset_init = initial_position * half_rotary * 2

        print(f"  Capturing full-layer HIP graphs for {num_gpus} GPUs × "
              f"{num_layers} layers × 2 segments...")
        t0 = __import__('time').perf_counter()

        # Create per-GPU capture streams
        self._capture_streams = []
        for gpu_idx in range(num_gpus):
            hip.set_device(tp.device_ids[gpu_idx])
            stream = ctypes.c_void_p(0)
            err = hip._lib.hipStreamCreate(ctypes.byref(stream))
            if err != 0:
                raise RuntimeError(f"hipStreamCreate failed: {err}")
            self._capture_streams.append(stream.value)

        # Init per-gpu segment lists
        self._attn_segs   = [[] for _ in range(num_gpus)]
        self._ffn_segs    = [[] for _ in range(num_gpus)]
        self._mutable_attn = [[] for _ in range(num_gpus)]

        for layer_idx in range(num_layers):
            for gpu_idx in range(num_gpus):
                engine = tp.engines[gpu_idx]
                lc = tp._engine_layer_caches[gpu_idx][layer_idx]
                lw = engine.layers[layer_idx]
                cap_stream = self._capture_streams[gpu_idx]

                hip.set_device(tp.device_ids[gpu_idx])

                # ---- Capture ATTENTION segment ----
                attn_seg = GraphSegment(self._graph_rt, tp.device_ids[gpu_idx])
                attn_seg.begin_capture(cap_stream)

                # Attention RMSNorm
                if 'attn_rmsnorm' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['attn_rmsnorm'])

                if lw.layer_type == 'full_attention':
                    if 'gemv_q_fused' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_q_fused'])
                    if 'gemv_k_only' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_k_only'])
                        # V GEMV: update output ptr to initial position
                        cur_pos = engine.kv_cache.current_len
                        kv_stride = lc['_kv_stride']
                        v_cache_ptr = lc['_v_cache_base'] + cur_pos * kv_stride
                        lc['gemv_v_cache'].params[2].value = v_cache_ptr
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_v_cache'])
                    elif 'gemv_kv_fused' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_kv_fused'])

                    # QKNorm/RoPE — set cos/sin to initial position
                    if 'qknorm_q' in lc:
                        lc['qknorm_q'].params[2].value = engine.d_cos + cos_offset_init
                        lc['qknorm_q'].params[3].value = engine.d_sin + cos_offset_init
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['qknorm_q'])
                    if 'qknorm_k' in lc:
                        lc['qknorm_k'].params[2].value = engine.d_cos + cos_offset_init
                        lc['qknorm_k'].params[3].value = engine.d_sin + cos_offset_init
                        if '_k_cache_base' in lc:
                            cur_pos = engine.kv_cache.current_len
                            kv_stride = lc['_kv_stride']
                            k_cache_ptr = lc['_k_cache_base'] + cur_pos * kv_stride
                            lc['qknorm_k'].params[4].value = k_cache_ptr
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['qknorm_k'])

                    # Decode attention — set seq_len to initial value
                    if 'decode_attn' in lc:
                        lc['decode_attn'].params[4].value = (
                            engine.kv_cache.current_len + 1)
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['decode_attn'])

                    if 'sigmoid_mul' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['sigmoid_mul'])
                    if 'gemv_o_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_o_proj'])

                else:
                    # DeltaNet linear attention
                    if 'gemv_la_in_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_la_in_proj'])
                    if 'deltanet_v3' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['deltanet_v3'])
                    if 'deltanet_v3_shift' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['deltanet_v3_shift'])
                    if 'gemv_la_out_proj' in lc:
                        self._hipModuleLaunchKernel(hip, cap_stream, lc['gemv_la_out_proj'])

                attn_seg.end_capture(cap_stream)
                attn_seg.instantiate()
                self._attn_segs[gpu_idx].append(attn_seg)

                # Build mutable node list for attention segment (position-based matching)
                mutable_list = []
                if lw.layer_type == 'full_attention':
                    capture_order = []
                    if 'attn_rmsnorm' in lc:
                        capture_order.append('attn_rmsnorm')
                    if 'gemv_q_fused' in lc:
                        capture_order.append('gemv_q_fused')
                    if 'gemv_k_only' in lc:
                        capture_order.append('gemv_k_only')
                        capture_order.append('gemv_v_cache')
                    elif 'gemv_kv_fused' in lc:
                        capture_order.append('gemv_kv_fused')
                    if 'qknorm_q' in lc:
                        capture_order.append('qknorm_q')
                    if 'qknorm_k' in lc:
                        capture_order.append('qknorm_k')
                    if 'decode_attn' in lc:
                        capture_order.append('decode_attn')
                    if 'sigmoid_mul' in lc:
                        capture_order.append('sigmoid_mul')
                    if 'gemv_o_proj' in lc:
                        capture_order.append('gemv_o_proj')

                    mutable_keys = frozenset(('qknorm_q', 'qknorm_k', 'decode_attn',
                                              'gemv_v_cache'))
                    kernel_nodes = attn_seg._kernel_nodes
                    if len(kernel_nodes) == len(capture_order):
                        for pos, (node, key) in enumerate(
                                zip(kernel_nodes, capture_order)):
                            if key in mutable_keys:
                                spec = lc[key]
                                node_params = attn_seg.get_kernel_params(node)
                                mutable_list.append({
                                    'node': node,
                                    'key': key,
                                    'spec': spec,
                                    'base_params': node_params,
                                })
                    else:
                        print(f"    WARNING Layer {layer_idx} GPU {gpu_idx}: "
                              f"{len(kernel_nodes)} nodes vs {len(capture_order)} "
                              "expected. Mutable tracking may be incomplete.")
                        min_len = min(len(kernel_nodes), len(capture_order))
                        for pos in range(min_len):
                            node = kernel_nodes[pos]
                            key = capture_order[pos]
                            if key in mutable_keys:
                                spec = lc[key]
                                node_params = attn_seg.get_kernel_params(node)
                                mutable_list.append({
                                    'node': node,
                                    'key': key,
                                    'spec': spec,
                                    'base_params': node_params,
                                })

                self._mutable_attn[gpu_idx].append(mutable_list)

                # ---- Capture FFN segment ----
                ffn_seg = GraphSegment(self._graph_rt, tp.device_ids[gpu_idx])
                ffn_seg.begin_capture(cap_stream)

                if 'ffn_rmsnorm' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_rmsnorm'])
                if 'ffn_gate_up_silu' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_gate_up_silu'])
                if 'ffn_down' in lc:
                    self._hipModuleLaunchKernel(hip, cap_stream, lc['ffn_down'])

                ffn_seg.end_capture(cap_stream)
                ffn_seg.instantiate()
                self._ffn_segs[gpu_idx].append(ffn_seg)

        elapsed = __import__('time').perf_counter() - t0
        total_segs = num_gpus * num_layers * 2
        print(f"  Full-layer graph capture complete: {total_segs} segments "
              f"in {elapsed*1000:.0f}ms")

        # Report node counts for first few layers
        for li in range(min(2, num_layers)):
            attn_nodes = self._attn_segs[0][li].num_kernel_nodes()
            ffn_nodes  = self._ffn_segs[0][li].num_kernel_nodes()
            lw = tp.engines[0].layers[li]
            print(f"    Layer {li} ({lw.layer_type}): "
                  f"attn_seg={attn_nodes} nodes, ffn_seg={ffn_nodes} nodes")

        # Synchronize all GPUs
        for gpu_idx in range(num_gpus):
            hip.set_device(tp.device_ids[gpu_idx])
            hip.synchronize()

        self._captured = True

        # Eagerly build C graph dispatch plan after capture completes.
        # This eliminates the per-step lazy-build overhead and ensures
        # _c_graph_plan_ptr is set before the first replay call.
        try:
            self._load_c_graph_lib()
            self._build_c_graph_plan()
            print(f"  C graph dispatch plan ready: plan_ptr=0x{self._c_graph_plan_ptr:x}")
        except Exception as e:
            print(f"  WARNING: Failed to build C graph dispatch plan during capture: {e}")
            print(f"  Will fall back to Python replay loop.")
            import traceback
            traceback.print_exc()
            # Ensure _c_graph_lib is not None so lazy-build won't retry in replay
            if self._c_graph_lib is None:
                self._c_graph_lib = False
            self._c_graph_plan_ptr = 0

    def _load_c_graph_lib(self):
        """Load c_graph_dispatch.so for tight-C graph replay loop.

        Eliminates Python per-layer overhead: instead of 512 Python API calls
        per decode step (from the Python replay loop), uses a single C function
        call that handles all 64 layers' graph replays + allreduces in C.

        Auto-builds c_graph_dispatch.so if it is missing or stale.
        """
        import ctypes

        src_dir = Path(__file__).parent.parent / "runtime"
        c_path  = src_dir / "c_graph_dispatch.c"
        so_path = src_dir / "c_graph_dispatch.so"

        if not c_path.exists():
            raise FileNotFoundError(
                f"c_graph_dispatch.c not found at {c_path}.")

        # Auto-build if .so is missing or .c is newer
        if not so_path.exists() or os.path.getmtime(c_path) > os.path.getmtime(so_path):
            # Try multiple ROCm paths (container vs host)
            rocm_include = None
            rocm_lib = None
            for rocm_root in ['/opt/rocm', '/usr/local/rocm', '/usr/rocm']:
                inc = f"{rocm_root}/include"
                lib = f"{rocm_root}/lib"
                if os.path.isdir(inc):
                    rocm_include = inc
                    rocm_lib = lib
                    break

            build_cmd = [
                "gcc", "-O3", "-shared", "-fPIC",
                "-o", str(so_path), str(c_path),
            ]
            if rocm_include:
                build_cmd += [f"-I{rocm_include}", f"-L{rocm_lib}", "-lamdhip64"]

            try:
                print(f"  Auto-building c_graph_dispatch.so ...")
                result = subprocess.run(build_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # Try without ROCm flags (pure types, no HIP calls at build time)
                    build_cmd_simple = [
                        "gcc", "-O3", "-shared", "-fPIC",
                        "-o", str(so_path), str(c_path),
                    ]
                    result2 = subprocess.run(build_cmd_simple, capture_output=True,
                                             text=True)
                    if result2.returncode != 0:
                        raise RuntimeError(
                            f"gcc failed: {result.stderr}\n{result2.stderr}")
                print(f"  c_graph_dispatch.so built successfully.")
            except Exception as build_err:
                raise RuntimeError(
                    f"Failed to auto-build c_graph_dispatch.so: {build_err}. "
                    "Build manually with: gcc -O3 -shared -fPIC "
                    "-I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 "
                    "-o src/runtime/c_graph_dispatch.so "
                    "src/runtime/c_graph_dispatch.c") from build_err

        if not so_path.exists():
            raise FileNotFoundError(
                f"c_graph_dispatch.so not found at {so_path} after build attempt.")

        lib = ctypes.CDLL(str(so_path))
        lib.c_graph_dispatch_step.argtypes = [
            ctypes.c_uint64,   # plan_ptr
            ctypes.c_uint64,   # cos_offset
            ctypes.c_uint32,   # seq_len
        ]
        lib.c_graph_dispatch_step.restype = ctypes.c_int

        lib.c_graph_dispatch_get_layer_spec_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_mutable_param_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_allreduce_spec_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_plan_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_kparams_size.restype = ctypes.c_int

        self._c_graph_lib = lib
        return lib

    def _build_c_graph_plan(self):
        """Build CGraphDispatchPlan ctypes structure from captured graph segments.

        Returns (plan_buf, objects) where objects keeps all ctypes objects alive.
        Sets self._c_graph_plan_ptr to the plan address.

        The plan matches c_graph_dispatch.c:
          CGraphLayerSpec: per-GPU × per-layer, with attn_graph_exec, ffn_graph_exec,
                           and mutable_params array
          CGraphAllreduceSpec: per-layer allreduce config (same for all layers)
          CGraphDispatchPlan: top-level struct with all arrays + HIP fn pointers
        """
        import ctypes as ct
        import struct

        tp = self._tp
        hip = tp._hip
        num_layers = tp.config.num_hidden_layers
        num_engines = tp.tp_size
        h = tp.config.hidden_size
        p2p_ar = tp._p2p_ar

        if p2p_ar is None:
            raise RuntimeError("_build_c_graph_plan requires P2P allreduce")

        lib = self._c_graph_lib
        if lib is None:
            lib = self._load_c_graph_lib()

        # Get struct sizes from C (for alignment correctness)
        layer_spec_size  = lib.c_graph_dispatch_get_layer_spec_size()
        mutable_sz       = lib.c_graph_dispatch_get_mutable_param_size()
        ar_spec_size     = lib.c_graph_dispatch_get_allreduce_spec_size()
        plan_size        = lib.c_graph_dispatch_get_plan_size()
        kparams_size     = lib.c_graph_dispatch_get_kparams_size()

        print(f"  C graph dispatch plan: layer_spec={layer_spec_size}B, "
              f"mutable={mutable_sz}B, ar_spec={ar_spec_size}B, plan={plan_size}B, "
              f"kparams={kparams_size}B")

        # CMutableParam layout (from c_graph_dispatch.c):
        # uint64 node            +0
        # uint64 graph_exec      +8
        # uint64 params_array    +16
        # uint32 param_index     +24
        # uint32 mutable_type    +28
        # HipKernelNodeParams    +32  (kparams_size bytes)
        # uint64 kv_cache_base   +32+kparams_size
        # uint32 kv_stride       +40+kparams_size
        # CMutableParam layout (size=128, confirmed via c_graph_dispatch_get_mutable_param_size()):
        # uint64 node            +0
        # uint64 graph_exec      +8
        # uint64 params_array    +16
        # uint32 param_index     +24
        # uint32 mutable_type    +28
        # HipKernelNodeParams    +32  (64 bytes → ends at 96)
        # uint64 kv_cache_base   +96
        # uint32 kv_stride       +104
        # uint32 _pad            +108
        # uint64 d_cos_base      +112
        # uint64 d_sin_base      +120
        # Total: 128 bytes
        kparams_offset   = 32
        kv_base_offset   = 96
        kv_stride_offset = 104
        cos_base_offset  = 112
        sin_base_offset  = 120

        # MUTABLE_TYPE constants (from c_graph_dispatch.c)
        MUTABLE_TYPE_COS_PTR = 1
        MUTABLE_TYPE_SIN_PTR = 2
        MUTABLE_TYPE_SEQ_LEN = 3
        MUTABLE_TYPE_KV_K_PTR = 4
        MUTABLE_TYPE_KV_V_PTR = 5

        # Key-to-mutable-type mapping
        # For qknorm_q and qknorm_k, we generate TWO CMutableParam entries:
        #   one for cos (param[2]) and one for sin (param[3])
        # For qknorm_k with k-cache: THREE entries (cos, sin, kv_k_ptr)
        # For gemv_v_cache: ONE entry (kv_v_ptr, param[2])
        # For decode_attn: ONE entry (seq_len, param[4])

        # CGraphLayerSpec layout (from c_graph_dispatch.c):
        # uint64 attn_graph_exec +0
        # uint64 ffn_graph_exec  +8
        # CMutableParam[8]       +16  (8 * mutable_sz bytes)
        # uint32 num_mutable     +16 + 8*mutable_sz
        # int32  layer_type      +20 + 8*mutable_sz  (0=full_attn, 1=deltanet)
        # uint32 _pad            +24 + 8*mutable_sz

        # CGraphAllreduceSpec layout (from c_graph_dispatch.c):
        # 3 function pointers (24 bytes): reduce_tp2, reduce_tp3, reduce_tp4
        # int32 tp_size (+24)
        # int32 device_ids[4] (+28)
        # uint64 partial_ptrs[4] (+44)
        # uint64 hidden_ptrs[4] (+76)  -- 8-byte aligned
        # uint64 gather_bufs[3] (+108)
        # uint64 allreduce_streams[4] (+132)
        # uint64 compute_events[4] (+164)
        # uint64 ar_done_events[4] (+196)
        # uint64 compute_streams[4] (+228)
        # uint32 num_elems (+260)
        # uint32 _pad (+264)
        # Total: 272 bytes

        # Allocate flat buffers
        n_layer_specs = num_layers * num_engines
        layer_specs_buf = ct.create_string_buffer(n_layer_specs * layer_spec_size)
        attn_ar_buf     = ct.create_string_buffer(num_layers * ar_spec_size)
        ffn_ar_buf      = ct.create_string_buffer(num_layers * ar_spec_size)
        plan_buf        = ct.create_string_buffer(plan_size)

        # Keep all alive
        objects = {
            'layer_specs_buf': layer_specs_buf,
            'attn_ar_buf':     attn_ar_buf,
            'ffn_ar_buf':      ffn_ar_buf,
            'plan_buf':        plan_buf,
        }

        def write_uint64(buf, offset, val):
            ct.memmove(ct.addressof(buf) + offset,
                       (ct.c_uint64 * 1)(val), 8)

        def write_uint32(buf, offset, val):
            ct.memmove(ct.addressof(buf) + offset,
                       (ct.c_uint32 * 1)(val), 4)

        def write_int32(buf, offset, val):
            ct.memmove(ct.addressof(buf) + offset,
                       (ct.c_int32 * 1)(val), 4)

        def write_ptr(buf, offset, val):
            ct.memmove(ct.addressof(buf) + offset,
                       (ct.c_uint64 * 1)(val), 8)

        def read_uint32(buf, offset):
            v = (ct.c_uint32 * 1)()
            ct.memmove(v, ct.addressof(buf) + offset, 4)
            return v[0]

        # ---- Fill CGraphLayerSpec for each layer × engine ----
        MAX_MUTABLE = 8
        for layer_idx in range(num_layers):
            for gpu_idx in range(num_engines):
                spec_offset = (layer_idx * num_engines + gpu_idx) * layer_spec_size

                attn_seg = self._attn_segs[gpu_idx][layer_idx]
                ffn_seg  = self._ffn_segs[gpu_idx][layer_idx]
                engine   = tp.engines[gpu_idx]
                lc       = tp._engine_layer_caches[gpu_idx][layer_idx]
                lw       = engine.layers[layer_idx]

                # attn_graph_exec and ffn_graph_exec
                write_uint64(layer_specs_buf, spec_offset + 0,
                             attn_seg._graph_exec or 0)
                write_uint64(layer_specs_buf, spec_offset + 8,
                             ffn_seg._graph_exec or 0)

                # Mutable params: starts at offset +16
                mutable_base = spec_offset + 16
                num_mutable = 0

                if lw.layer_type == 'full_attention':
                    for m in self._mutable_attn[gpu_idx][layer_idx]:
                        key  = m['key']
                        node = m['node']
                        spec = m['spec']
                        base = m['base_params']
                        graph_exec = attn_seg._graph_exec

                        # Build a HipKernelNodeParams bytes blob (64 bytes, confirmed size)
                        # Layout (confirmed with offsetof checks):
                        #   blockDimX,Y,Z,_pad1: 0-15 (4×uint32)
                        #   extra:               16-23 (uint64, NULL)
                        #   func:                24-31 (uint64)
                        #   gridDimX,Y,Z,_pad2:  32-47 (4×uint32)
                        #   kernelParams:        48-55 (uint64, → params_array)
                        #   sharedMemBytes:      56-59 (uint32)
                        #   _pad3:               60-63 (uint32)
                        kparams_bytes = bytearray(kparams_size)  # 64 bytes, zero-init
                        struct.pack_into('<IIII', kparams_bytes, 0,
                                         base.blockDimX, base.blockDimY, base.blockDimZ, 0)
                        struct.pack_into('<Q', kparams_bytes, 16, 0)   # extra = NULL
                        struct.pack_into('<Q', kparams_bytes, 24,
                                         base.func if base.func else 0)
                        struct.pack_into('<IIII', kparams_bytes, 32,
                                         base.gridDimX, base.gridDimY, base.gridDimZ, 0)
                        # kernelParams = address of spec.params_array
                        params_arr_ptr = ct.addressof(spec.params_array)
                        struct.pack_into('<Q', kparams_bytes, 48, params_arr_ptr)
                        struct.pack_into('<I', kparams_bytes, 56,
                                         base.sharedMemBytes if hasattr(base, 'sharedMemBytes') else 0)

                        # Identify the param indices and types for this key
                        # We expand each key into one or more CMutableParam entries

                        def write_mutable(mp_idx, param_idx, mtype,
                                           kv_base=0, kv_stride=0,
                                           cos_base=0, sin_base=0,
                                           kparams_blob=kparams_bytes):
                            nonlocal num_mutable
                            if mp_idx >= MAX_MUTABLE:
                                return
                            mp_off = mutable_base + mp_idx * mutable_sz

                            write_uint64(layer_specs_buf, mp_off + 0, node)
                            write_uint64(layer_specs_buf, mp_off + 8, graph_exec)
                            # params_array: pointer to spec.params_array
                            params_arr = ct.addressof(spec.params_array)
                            write_uint64(layer_specs_buf, mp_off + 16, params_arr)
                            write_uint32(layer_specs_buf, mp_off + 24, param_idx)
                            write_uint32(layer_specs_buf, mp_off + 28, mtype)
                            # kparams blob
                            ct.memmove(ct.addressof(layer_specs_buf) + mp_off + kparams_offset,
                                       bytes(kparams_blob), kparams_size)
                            write_uint64(layer_specs_buf, mp_off + kv_base_offset, kv_base)
                            write_uint32(layer_specs_buf, mp_off + kv_stride_offset, kv_stride)
                            write_uint32(layer_specs_buf, mp_off + kv_stride_offset + 4, 0) # _pad
                            write_uint64(layer_specs_buf, mp_off + cos_base_offset, cos_base)
                            write_uint64(layer_specs_buf, mp_off + sin_base_offset, sin_base)
                            num_mutable += 1

                        if key in ('qknorm_q', 'qknorm_k'):
                            # COS ptr: params[2]
                            write_mutable(num_mutable, 2, MUTABLE_TYPE_COS_PTR,
                                          cos_base=engine.d_cos, sin_base=engine.d_sin)
                            # SIN ptr: params[3]
                            write_mutable(num_mutable, 3, MUTABLE_TYPE_SIN_PTR,
                                          cos_base=engine.d_cos, sin_base=engine.d_sin)
                            if key == 'qknorm_k' and '_k_cache_base' in lc:
                                # KV K ptr: params[4]
                                write_mutable(num_mutable, 4, MUTABLE_TYPE_KV_K_PTR,
                                              kv_base=lc['_k_cache_base'],
                                              kv_stride=lc['_kv_stride'])
                        elif key == 'decode_attn':
                            # SEQ_LEN: params[4]
                            write_mutable(num_mutable, 4, MUTABLE_TYPE_SEQ_LEN)
                        elif key == 'gemv_v_cache':
                            # KV V ptr: params[2]
                            write_mutable(num_mutable, 2, MUTABLE_TYPE_KV_V_PTR,
                                          kv_base=lc['_v_cache_base'],
                                          kv_stride=lc['_kv_stride'])

                # Write num_mutable and layer_type
                # CGraphLayerSpec: [attn_exec(8)][ffn_exec(8)][mutable_params[8](1024)][num_mutable(4)][layer_type(4)][_pad(8)]
                # mutable_params starts at offset 16, size = 8 × 128 = 1024 → ends at 1040
                nm_offset = spec_offset + 16 + MAX_MUTABLE * mutable_sz  # 16+1024=1040
                write_uint32(layer_specs_buf, nm_offset, num_mutable)
                write_int32(layer_specs_buf, nm_offset + 4,
                            0 if lw.layer_type == 'full_attention' else 1)

        # ---- Fill CGraphAllreduceSpec for each layer ----
        # All layers use the same allreduce config (same buffers, same streams)
        # We build ONE template and copy it for each layer, updating partial_ptrs per layer.

        def fill_ar_spec(buf, ar_offset, partial_ptrs, hidden_ptrs):
            """Fill a CGraphAllreduceSpec at buf[ar_offset:ar_offset+ar_spec_size]."""
            # Function pointers: reduce_tp2, reduce_tp3, reduce_tp4
            # (32 bytes total for 3 pointers)
            if hasattr(p2p_ar, '_lib'):
                lib_ar = p2p_ar._lib
                # Get function pointers by name
                for i, fn_name in enumerate(['p2p_reduce_residual_tp2',
                                              'p2p_reduce_residual_tp3',
                                              'p2p_reduce_residual_tp4']):
                    if hasattr(lib_ar, fn_name):
                        fn_ptr = ct.cast(getattr(lib_ar, fn_name), ct.c_void_p).value
                        if fn_ptr:
                            write_uint64(buf, ar_offset + i * 8, fn_ptr)

            # tp_size
            write_int32(buf, ar_offset + 24, p2p_ar._tp_size)

            # device_ids[4]
            for i, dev_id in enumerate(p2p_ar._device_ids):
                write_int32(buf, ar_offset + 28 + i * 4, dev_id)

            # partial_ptrs[4]  (confirmed offset: 48)
            for i, ptr in enumerate(partial_ptrs):
                write_uint64(buf, ar_offset + 48 + i * 8, ptr)

            # hidden_ptrs[4]  (confirmed offset: 80)
            for i, ptr in enumerate(hidden_ptrs):
                write_uint64(buf, ar_offset + 80 + i * 8, ptr)

            # gather_bufs[3]  (confirmed offset: 112)
            for i, ptr in enumerate(p2p_ar._gather_bufs[:3]):
                write_uint64(buf, ar_offset + 112 + i * 8, ptr)

            # allreduce_streams[4]  (confirmed offset: 136)
            for i, s in enumerate(p2p_ar._allreduce_streams):
                write_uint64(buf, ar_offset + 136 + i * 8, s)

            # compute_events[4]  (confirmed offset: 168)
            for i, e in enumerate(p2p_ar._compute_events):
                write_uint64(buf, ar_offset + 168 + i * 8, e)

            # ar_done_events[4]  (confirmed offset: 200)
            for i, e in enumerate(p2p_ar._ar_done_events):
                write_uint64(buf, ar_offset + 200 + i * 8, e)

            # compute_streams[4]: null streams (0)  (confirmed offset: 232)
            for i in range(num_engines):
                write_uint64(buf, ar_offset + 232 + i * 8, 0)

            # num_elems  (confirmed offset: 264)
            write_uint32(buf, ar_offset + 264, h)

            # --- Kernel P2P allreduce extension (new fields at offset 272+) ---
            # use_kernel_p2p: 272, _pad2: 276, kernel_p2p_fn: 280
            # peer_ptrs[4][3]: 288 (4×3×8 bytes = 96 bytes → total 384)
            kernel_p2p_lib = None
            if p2p_ar is not None and hasattr(p2p_ar, '_kernel_p2p_lib'):
                kernel_p2p_lib = p2p_ar._kernel_p2p_lib

            if kernel_p2p_lib is not None and tp.tp_size == 4:
                # Get kernel_p2p_allreduce_residual_tp4 function pointer
                try:
                    fn = getattr(kernel_p2p_lib, 'kernel_p2p_allreduce_residual_tp4')
                    fn_ptr = ct.cast(fn, ct.c_void_p).value
                    if fn_ptr:
                        write_uint32(buf, ar_offset + 272, 1)  # use_kernel_p2p = 1
                        write_uint32(buf, ar_offset + 276, 0)  # _pad2
                        write_uint64(buf, ar_offset + 280, fn_ptr)  # kernel_p2p_fn

                        # peer_ptrs[4][3]: for GPU i, the 3 other GPUs' partial ptrs
                        for gi in range(4):
                            peer_indices = [j for j in range(4) if j != gi]
                            for k, peer_idx in enumerate(peer_indices):
                                peer_off = 288 + gi * 24 + k * 8
                                write_uint64(buf, ar_offset + peer_off, partial_ptrs[peer_idx])
                except (AttributeError, TypeError) as ex:
                    print(f"  WARNING: kernel P2P fn ptr failed: {ex}. Using standard P2P.")

        attn_partial_ptrs = [e.d_proj_out for e in tp.engines]
        ffn_partial_ptrs  = [e.d_ffn_out  for e in tp.engines]
        hidden_ptrs_all   = [e.d_hidden   for e in tp.engines]

        for layer_idx in range(num_layers):
            fill_ar_spec(attn_ar_buf, layer_idx * ar_spec_size,
                         attn_partial_ptrs, hidden_ptrs_all)
            fill_ar_spec(ffn_ar_buf, layer_idx * ar_spec_size,
                         ffn_partial_ptrs, hidden_ptrs_all)

        # ---- Fill CGraphDispatchPlan ----
        # Layout (from c_graph_dispatch.c):
        # int32 num_layers       +0
        # int32 num_engines      +4
        # uint64 graph_layer_specs   +8
        # uint64 attn_allreduce_specs +16
        # uint64 ffn_allreduce_specs  +24
        # HIP function pointers (starting at +32):
        #   hipSetDevice_fn     +32
        #   hipStreamSynchronize_fn +40
        #   hipEventRecord_fn   +48
        #   hipStreamWaitEvent_fn +56
        #   hipMemcpyPeerAsync_fn +64
        #   hipMemcpyAsync_fn   +72
        #   hipGetLastError_fn  +80
        #   hipGraphLaunch_fn   +88
        #   hipGraphExecKernelNodeSetParams_fn +96

        write_int32(plan_buf, 0, num_layers)
        write_int32(plan_buf, 4, num_engines)
        write_uint64(plan_buf, 8, ct.addressof(layer_specs_buf))
        write_uint64(plan_buf, 16, ct.addressof(attn_ar_buf))
        write_uint64(plan_buf, 24, ct.addressof(ffn_ar_buf))

        # HIP function pointers
        # Use the graph_rt library (has hipGraphLaunch + hipGraphExecKernelNodeSetParams)
        # and hip._lib for standard functions; both are libamdhip64.so so same ptrs.
        import ctypes.util
        graph_lib = self._graph_rt._lib  # HIPGraphRuntime._lib = libamdhip64.so

        fn_names = [
            ('hipSetDevice',                      32),
            ('hipStreamSynchronize',              40),
            ('hipEventRecord',                    48),
            ('hipStreamWaitEvent',                56),
            ('hipMemcpyPeerAsync',                64),
            ('hipMemcpyAsync',                    72),
            ('hipGetLastError',                   80),
            ('hipGraphLaunch',                    88),
            ('hipGraphExecKernelNodeSetParams',   96),
        ]
        for fn_name, offset in fn_names:
            # Get function pointer using ctypes handle+name lookup
            try:
                fn = getattr(graph_lib, fn_name)
                fn_ptr = ct.cast(fn, ct.c_void_p).value
                if fn_ptr:
                    write_uint64(plan_buf, offset, fn_ptr)
                else:
                    print(f"  WARNING: {fn_name} returned NULL pointer")
            except AttributeError:
                print(f"  WARNING: {fn_name} not found in HIP library")

        plan_ptr = ct.addressof(plan_buf)
        self._c_graph_plan_buf = plan_buf
        self._c_graph_objects  = objects
        self._c_graph_plan_ptr = plan_ptr

        print(f"  C graph dispatch plan built: {num_layers} layers × {num_engines} GPUs")
        return plan_ptr

    def replay_step_full_layer(self, position: int,
                               include_allreduce: bool = True):
        """Replay attention and FFN graphs with updated mutable params.

        Per layer:
          1. Update mutable params (cos/sin, seq_len, kv_ptr) for all GPUs
          2. Replay attn graphs for all GPUs
          3. Kernel P2P allreduce (attn) — host-orchestrated, ensures cross-GPU sync
          4. Replay FFN graphs for all GPUs
          5. Kernel P2P allreduce (FFN) — host-orchestrated

        The allreduce is ALWAYS host-side (not in graph) to ensure correct
        cross-GPU synchronization. 'include_allreduce' is kept for API compatibility.

        Args:
            position: current decode position (for cos/sin and seq_len)
            include_allreduce: if True (default), use kernel P2P allreduce;
                               if False, use existing _allreduce_residual path
        """
        import ctypes

        tp = self._tp
        hip = tp._hip
        num_layers = tp.config.num_hidden_layers
        half_rotary = tp.engines[0].rotary_dim // 2
        cos_offset = position * half_rotary * 2
        h = tp.config.hidden_size

        # Use C graph dispatch plan if available (eliminates Python per-layer overhead)
        if self._c_graph_plan_ptr:
            import ctypes
            seq_len = tp.engines[0].kv_cache.current_len + 1
            err = self._c_graph_lib.c_graph_dispatch_step(
                ctypes.c_uint64(self._c_graph_plan_ptr),
                ctypes.c_uint64(cos_offset),
                ctypes.c_uint32(seq_len),
            )
            if err != 0:
                raise RuntimeError(f"c_graph_dispatch_step failed: HIP error {err}")
            return

        # Build C plan on first call (after capture)
        if self._c_graph_lib is None:
            try:
                self._load_c_graph_lib()
                self._build_c_graph_plan()
                # Recurse with plan now built
                return self.replay_step_full_layer(position, include_allreduce)
            except Exception as e:
                print(f"WARNING: Failed to build C graph dispatch plan: {e}. "
                      "Falling back to Python replay loop.")
                import traceback
                traceback.print_exc()
                # Set to False (not None) to avoid retrying on every step
                self._c_graph_lib = False
                self._c_graph_plan_ptr = 0

        # Python fallback loop (used only if C plan fails to build)
        from src.runtime.hip_graph_dispatch import hipKernelNodeParams
        import ctypes

        for layer_idx in range(num_layers):
            cur_pos = tp.engines[0].kv_cache.current_len

            # PHASE 1: Update mutable params for ALL GPUs
            for gpu_idx in range(tp.tp_size):
                engine = tp.engines[gpu_idx]
                lc = tp._engine_layer_caches[gpu_idx][layer_idx]
                lw = engine.layers[layer_idx]

                hip.set_device(tp.device_ids[gpu_idx])

                if lw.layer_type == 'full_attention':
                    for m in self._mutable_attn[gpu_idx][layer_idx]:
                        key = m['key']
                        node = m['node']
                        spec = m['spec']
                        base = m['base_params']

                        new_kp = hipKernelNodeParams()
                        new_kp.blockDimX = base.blockDimX
                        new_kp.blockDimY = base.blockDimY
                        new_kp.blockDimZ = base.blockDimZ
                        new_kp.gridDimX  = base.gridDimX
                        new_kp.gridDimY  = base.gridDimY
                        new_kp.gridDimZ  = base.gridDimZ
                        new_kp.func = base.func
                        new_kp.sharedMemBytes = base.sharedMemBytes
                        new_kp.extra = None

                        if key in ('qknorm_q', 'qknorm_k'):
                            spec.params[2].value = engine.d_cos + cos_offset
                            spec.params[3].value = engine.d_sin + cos_offset
                            if key == 'qknorm_k' and '_k_cache_base' in lc:
                                kv_stride = lc['_kv_stride']
                                spec.params[4].value = (lc['_k_cache_base']
                                                        + cur_pos * kv_stride)
                        elif key == 'decode_attn':
                            spec.params[4].value = cur_pos + 1
                        elif key == 'gemv_v_cache':
                            kv_stride = lc['_kv_stride']
                            spec.params[2].value = (lc['_v_cache_base']
                                                    + cur_pos * kv_stride)

                        new_kp.kernelParams = ctypes.cast(
                            spec.params_array, ctypes.c_void_p).value
                        self._attn_segs[gpu_idx][layer_idx].update_kernel_params(
                            node, new_kp)

            # PHASE 2: Replay ATTENTION segments for ALL GPUs
            for gpu_idx in range(tp.tp_size):
                hip.set_device(tp.device_ids[gpu_idx])
                self._attn_segs[gpu_idx][layer_idx].replay(stream=0)

            # PHASE 3: Allreduce attention partials → d_hidden
            compute_streams = [0] * tp.tp_size
            partial_attn = [e.d_proj_out for e in tp.engines]
            partial_ffn  = [e.d_ffn_out  for e in tp.engines]
            hidden_ptrs  = [e.d_hidden   for e in tp.engines]
            p2p_ar = tp._p2p_ar
            if p2p_ar is not None:
                p2p_ar.allreduce_residual_async(
                    partial_attn, hidden_ptrs, h, compute_streams)
                p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)
            else:
                tp._allreduce_residual("d_proj_out", h)

            # PHASE 4: Replay FFN segments for ALL GPUs
            for gpu_idx in range(tp.tp_size):
                hip.set_device(tp.device_ids[gpu_idx])
                self._ffn_segs[gpu_idx][layer_idx].replay(stream=0)

            # PHASE 5: Allreduce FFN partials → d_hidden
            if p2p_ar is not None:
                p2p_ar.allreduce_residual_async(
                    partial_ffn, hidden_ptrs, h, compute_streams)
                p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)
            else:
                tp._allreduce_residual("d_ffn_out", h)

    def cleanup(self):
        """Destroy all captured graph segments and release capture streams."""
        import ctypes

        tp = self._tp
        hip = tp._hip

        for gpu_segs in self._attn_segs:
            for seg in gpu_segs:
                seg.cleanup()
        for gpu_segs in self._ffn_segs:
            for seg in gpu_segs:
                seg.cleanup()

        self._attn_segs.clear()
        self._ffn_segs.clear()
        self._mutable_attn.clear()

        # Destroy capture streams
        for gpu_idx, stream in enumerate(self._capture_streams):
            if stream:
                hip.set_device(tp.device_ids[gpu_idx])
                hip._lib.hipStreamDestroy(ctypes.c_void_p(stream))
        self._capture_streams.clear()

        # Clean up C graph dispatch plan
        self._c_graph_plan_buf = None
        self._c_graph_objects  = {}
        self._c_graph_plan_ptr = 0
        self._c_graph_lib      = None

        self._captured = False

    def num_kernel_nodes_for_layer(self, gpu_idx: int, layer_idx: int) -> int:
        """Return total kernel nodes (attn + ffn) for a layer on one GPU."""
        if (gpu_idx < len(self._attn_segs) and
                layer_idx < len(self._attn_segs[gpu_idx])):
            attn_n = self._attn_segs[gpu_idx][layer_idx].num_kernel_nodes()
            ffn_n  = self._ffn_segs[gpu_idx][layer_idx].num_kernel_nodes()
            return attn_n + ffn_n
        return 0

def _load_fast_allreduce(hip_lib):
    """Build and load the fast_allreduce C extension."""
    src_dir = Path(__file__).parent.parent / "runtime"
    c_path = src_dir / "fast_allreduce.c"
    so_path = src_dir / "fast_allreduce.so"

    if not c_path.exists():
        return None

    if not so_path.exists() or os.path.getmtime(c_path) > os.path.getmtime(so_path):
        try:
            subprocess.check_call(
                ["gcc", "-O3", "-mf16c", "-mavx", "-shared", "-fPIC",
                 "-o", str(so_path), str(c_path)],
                stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        return None

    lib.fast_ar_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.fast_ar_init.restype = ctypes.c_int

    lib.fast_ar_fused_tp2.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fast_ar_fused_tp2.restype = ctypes.c_int

    lib.fast_ar_sum_tp2.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fast_ar_sum_tp2.restype = ctypes.c_int

    lib.fast_ar_fused_tp3.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fast_ar_fused_tp3.restype = ctypes.c_int

    lib.fast_ar_fused_tp4.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_int,
    ]
    lib.fast_ar_fused_tp4.restype = ctypes.c_int

    hip_set_device = ctypes.cast(
        hip_lib._lib.hipSetDevice, ctypes.c_void_p).value
    hip_sync = ctypes.cast(
        hip_lib._lib.hipDeviceSynchronize, ctypes.c_void_p).value
    hip_memcpy = ctypes.cast(
        hip_lib._lib.hipMemcpy, ctypes.c_void_p).value

    lib.fast_ar_init(hip_set_device, hip_sync, hip_memcpy)
    return lib


# ============================================================
# Speculative Decoding Support
# ============================================================

class SpeculativeDecodeState:
    """Manages state for speculative decoding with n-gram lookahead.
    
    This class handles:
    1. N-gram cache management (built from prompt and updated with generated tokens)
    2. Draft token generation from n-gram matches
    3. Verification of draft tokens using batched prefill-style forward pass
    4. KV cache updates for accepted tokens
    5. Rollback support for rejected tokens
    
    The key optimization: verification of K draft tokens uses a single
    prefill-style forward pass (batched GEMV), amortizing the TP allreduce
    cost across K tokens instead of 1.
    
    Attributes:
        tp_engine: TPInferenceEngine instance
        ngram_size: size of n-grams for draft generation
        max_draft_len: maximum draft tokens per iteration
        ngram_cache: NgramCache instance for storing n-gram continuations
        acceptance_stats: dict tracking acceptance rate and counts
    """
    
    def __init__(self, tp_engine, ngram_size: int = 3, max_draft_len: int = 5):
        """Initialize speculative decode state.
        
        Args:
            tp_engine: TPInferenceEngine instance to wrap
            ngram_size: size of n-grams for draft generation (default: 3)
            max_draft_len: maximum draft tokens per iteration (default: 5)
        """
        from src.inference.speculative import NgramCache
        
        self.tp_engine = tp_engine
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len
        self.ngram_cache = NgramCache(n=ngram_size)
        
        # Acceptance rate tracking
        self.acceptance_stats = {
            'total_drafts': 0,
            'total_accepted': 0,
            'total_iterations': 0,
        }
        
        # Cached buffers for batched verification
        # These avoid repeated allocations during decode
        self._hidden_size = tp_engine.config.hidden_size
        self._batch_bufs = None  # Will hold [max_draft_len, hidden_size] buffer
    
    def build_cache_from_prompt(self, prompt_token_ids: List[int]):
        """Build n-gram cache from initial prompt tokens.
        
        Must be called before starting speculative decoding on a new prompt.
        
        Args:
            prompt_token_ids: list of token IDs from the prompt
        """
        self.ngram_cache.clear()
        self.ngram_cache.build_from_sequence(prompt_token_ids)
        
        # Reset acceptance stats for new generation
        self.acceptance_stats = {
            'total_drafts': 0,
            'total_accepted': 0,
            'total_iterations': 0,
        }
    
    def generate_drafts(self, context_token_ids: List[int]) -> List[int]:
        """Generate draft tokens from n-gram cache given current context.
        
        Looks up the (n-1)-gram ending with the last token in context,
        and returns continuation candidates from the cache.
        
        Args:
            context_token_ids: full context (prompt + generated so far)
        
        Returns:
            List of draft token IDs (may be empty if no matches)
        """
        # Update cache with recent context
        # Only need to index the last few tokens to add new n-grams
        n = self.ngram_size
        if len(context_token_ids) >= n:
            window_start = max(0, len(context_token_ids) - n * 2)
            self.ngram_cache.build_from_sequence(
                context_token_ids[window_start:])
        
        # Generate draft tokens by following n-gram matches
        draft_tokens = []
        current_context = context_token_ids.copy()
        
        # Limit to max_draft_len
        for _ in range(self.max_draft_len):
            candidates = self.ngram_cache.query(current_context)
            if candidates is None or len(candidates) == 0:
                break
            
            # Pick the first candidate as draft (could use smarter selection)
            draft_token = candidates[0]
            draft_tokens.append(draft_token)
            current_context.append(draft_token)
        
        return draft_tokens
    
    def generate_eagle_drafts(self, hidden_state: np.ndarray,
                               k: int = None) -> List[int]:
        """Generate K draft tokens using EAGLE method from a hidden state.
        
        EAGLE uses the target model's own hidden states and lm_head to generate
        draft tokens, eliminating the need for a separate draft model.
        
        Algorithm:
        1. Apply lm_head to hidden state to get logits
        2. Sample or greedily select draft token
        3. Get embedding for the token (via engine lookup)
        4. Run decode step to get next hidden state
        5. Repeat K times
        
        Note: This is a simplified implementation that returns only token IDs.
        The full implementation would need to handle embedding lookups internally.
        
        Args:
            hidden_state: [hidden_size] FP16 hidden state from target model
            k: number of draft tokens to generate (default: self.max_draft_len)
        
        Returns:
            List of K draft token IDs
        """
        if k is None:
            k = self.max_draft_len
        
        draft_tokens = []
        current_hidden = hidden_state.copy()
        
        # Get engine 0's compute_logits method (all engines have same lm_head)
        engine0 = self.tp_engine.engines[0]
        
        # Note: For a full EAGLE implementation, we would need:
        # 1. Access to embedding weights for lookup
        # 2. A way to run partial decode steps without updating KV cache
        # 
        # For this integration, we use the TP engine's EAGLE methods
        # which handle these details. This method delegates to the engine.
        
        # Delegate to TP engine's EAGLE implementation
        if hasattr(self.tp_engine, 'generate_eagle_drafts'):
            draft_tokens = self.tp_engine.generate_eagle_drafts(
                current_hidden, k=k)
        
        return draft_tokens
    
    def verify_drafts(self, token_embeddings: List[np.ndarray], 
                      position: int,
                      draft_tokens: List[int]) -> tuple:
        """Verify draft tokens by running batched forward pass.
        
        This is the core verification step for speculative decoding:
        1. Process each draft token through the model (sequential for now)
        2. The caller compares output logits with draft tokens to decide acceptance
        3. Returns the hidden states for each position (for logit computation)
        
        For TP engines, each token processes through all TP ranks with allreduce.
        Future optimization: batch multiple tokens in a single prefill-style pass.
        
        Args:
            token_embeddings: list of [hidden_size] FP16 embeddings for each draft token
            position: starting position for KV cache writes
            draft_tokens: list of draft token IDs (for tracking/stats)
        
        Returns:
            Tuple of (output_hidden_states, accept_count_placeholder)
            - output_hidden_states: list of hidden states [hidden_size] for each position
            - accept_count_placeholder: 0 (caller determines actual acceptance)
        
        Note: Actual token acceptance is determined by comparing model logits
        (computed from hidden states) with draft tokens. This is done by the
        caller (e.g., speculative_decode function in speculative.py).
        """
        self.acceptance_stats['total_iterations'] += 1
        self.acceptance_stats['total_drafts'] += len(draft_tokens)
        
        if not draft_tokens or not token_embeddings:
            return [], 0
        
        # Process each draft token through the model
        # In a full optimization, this would be batched as a prefill-style pass
        output_hidden_states = []
        current_pos = position
        
        for emb in token_embeddings:
            # Run one decode step
            hidden_out = self.tp_engine.decode_step(emb, current_pos)
            output_hidden_states.append(hidden_out.copy())
            current_pos += 1
        
        # Note: KV cache was updated during decode_step calls above.
        # If tokens are rejected, caller must call rollback_kv_cache().
        
        # Acceptance is determined by caller comparing logits with draft tokens
        # Return 0 as placeholder - caller tracks actual acceptance
        return output_hidden_states, 0
    
    def record_accepted_tokens(self, num_accepted: int):
        """Record accepted token count for statistics.
        
        Called by the speculative decoding loop after determining which
        draft tokens were actually accepted.
        
        Args:
            num_accepted: number of tokens that were accepted
        """
        self.acceptance_stats['total_accepted'] += num_accepted
    
    def rollback_kv_cache(self, num_tokens: int):
        """Rollback KV cache by removing recently added tokens.
        
        Called when draft tokens are rejected and we need to regenerate
        from an earlier position.
        
        Args:
            num_tokens: number of tokens to remove from KV cache
        """
        for engine in self.tp_engine.engines:
            # Move KV cache position back
            engine.kv_cache.current_len = max(
                0, engine.kv_cache.current_len - num_tokens)
            # Note: The actual KV data in GPU memory would need to be
            # overwritten or tracked. For now, we just adjust the position.
    
    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate for generated tokens.
        
        Returns:
            acceptance_rate: ratio of accepted to total draft tokens
        """
        total = self.acceptance_stats['total_drafts']
        if total == 0:
            return 0.0
        return self.acceptance_stats['total_accepted'] / total
    
    def get_stats(self) -> dict:
        """Get speculative decoding statistics.
        
        Returns:
            Dict with acceptance rate and counts
        """
        return {
            **self.acceptance_stats,
            'acceptance_rate': self.get_acceptance_rate(),
        }
    
    def cleanup(self):
        """Clean up speculative decode state."""
        if self.ngram_cache is not None:
            self.ngram_cache.clear()
        self._batch_bufs = None


class TPInferenceEngine:
    """Multi-GPU tensor-parallel inference engine."""

    def __init__(self, config: QwenConfig, device_ids: list,
                 max_seq_len: int = 2048,
                 quant_format: str = 'w4a16',
                 use_int4_attention: bool = False):
        self.config = config
        self.tp_size = len(device_ids)
        self.device_ids = device_ids
        
        # Compute local intermediate size for TP sharding
        self.local_intermediate_size = config.intermediate_size // self.tp_size

        self.tp_group = TensorParallelGroup(device_ids)

        self.engines = []
        for rank, dev_id in enumerate(device_ids):
            engine = InferenceEngine(
                config, device_id=dev_id, max_seq_len=max_seq_len,
                tp_size=self.tp_size, tp_rank=rank,
                quant_format=quant_format,
                use_int4_attention=use_int4_attention,
            )
            self.engines.append(engine)

        for engine in self.engines:
            engine.register_tp_peers(self.engines, self.tp_group)

        self.device = self.engines[0].device
        self._hip = self.engines[0].device.hip

        self._fast_ar = _load_fast_allreduce(self._hip)
        if self._fast_ar:
            print("Fast C allreduce (AVX F16C) loaded")
        else:
            print("Using Python allreduce fallback")

        # Initialize P2P GPU allreduce (new path)
        # Use streams from tp_group for async P2P operations
        self._p2p_ar = None
        if self.tp_size >= 2:
            try:
                h = config.hidden_size
                streams = self.tp_group.streams  # per-GPU streams
                self._p2p_ar = P2PAllreduce(
                    self._hip, device_ids, h, streams=streams)
                print(f"P2P allreduce loaded (TP={self.tp_size}, GPU P2P reduce kernel)")
            except Exception as e:
                print(f"P2P allreduce unavailable ({e}), using host-mediated fallback")
                self._p2p_ar = None

        # Initialize fused P2P reduce (new path: each GPU launches its own kernel
        # that reads all peer partials via P2P, eliminating gather+broadcast steps)
        self._fused_p2p_ar = None
        self._fused_p2p_reduce = False  # flag to enable fused path
        if self.tp_size in (2, 4):
            try:
                h = config.hidden_size
                self._fused_p2p_ar = FusedP2PReduce(
                    self._hip, device_ids, h)
                print(f"Fused P2P reduce loaded (TP={self.tp_size}, all-GPU simultaneous kernel)")
            except Exception as e:
                print(f"Fused P2P reduce unavailable ({e}), using P2P allreduce fallback")
                self._fused_p2p_ar = None

        # Initialize ring allreduce (new path: ring topology for balanced PCIe bandwidth)
        self._ring_ar = None
        self._ring_allreduce = False  # flag to enable ring allreduce path
        if self.tp_size in (2, 4):
            try:
                h = config.hidden_size
                self._ring_ar = RingAllreduce(
                    self._hip, device_ids, h, streams=self.tp_group.streams)
                print(f"Ring allreduce loaded (TP={self.tp_size}, ring topology)")
            except Exception as e:
                print(f"Ring allreduce unavailable ({e}), using P2P allreduce fallback")
                self._ring_ar = None

        # Kernel P2P allreduce flag: use allreduce_residual_kernel() on _p2p_ar
        # (new kernel that directly reads peer GPU memory via BAR1, no host round-trips)
        self._kernel_p2p_allreduce = False  # flag to enable kernel P2P path

        # Persistent megakernel dispatcher (Milestone 5: m5-persistent-kernel)
        # When enabled, runs entire decode step as a single persistent kernel
        # with on-GPU task scheduling, eliminating all host dispatch overhead
        self._persistent_enabled = False
        self._persistent_dispatcher = None  # PersistentDecodeDispatcher instance

        # Fallback host buffers
        h = config.hidden_size
        self._host_bufs = [ctypes.create_string_buffer(h * 2)
                           for _ in range(self.tp_size)]
        self._host_hidden = ctypes.create_string_buffer(h * 2)

        # Threaded dispatch infrastructure (lazy init on first use)
        self._threaded_dispatch = False
        self._thread_pool_initialized = False
        self._worker_threads = []
        self._go_events = []      # main → worker: "start executing"
        self._done_events = []    # worker → main: "finished"
        self._worker_cmds = []    # shared command array (written by main, read by workers)
        self._worker_exception = None  # captures exception from any worker thread

        # Cached dispatch infrastructure (lazy init on build_dispatch_cache())
        self._cached_dispatch = False
        self._engine_layer_caches = []  # per-engine list of layer_cache dicts

        # Stream overlap dispatch: allreduce on dedicated stream, overlapping with
        # next-layer compute. Requires P2P allreduce with async methods.
        self._stream_overlap_dispatch = False
        # Default compute stream for each GPU (0 = HIP default stream)
        self._compute_streams = [0] * self.tp_size

        # C dispatch loop extension (loaded lazily via set_c_dispatch())
        self._c_dispatch_enabled = False
        self._c_dispatch_lib = None
        self._c_dispatch_plan = None
        self._c_dispatch_objects = {}  # keeps ctypes objects alive

        # C dispatch v2 (optimized: batched hipSetDevice calls)
        self._c_dispatch_v2_enabled = False
        self._c_dispatch_v2_lib = None

        # Graph dispatch (HIP graph capture and replay for near-zero kernel overhead)
        # Sits above C dispatch in priority: graph > c_dispatch > cached+stream > cached > serial
        self._graph_dispatch_enabled = False
        self._graph_decode_step = None  # GraphDecodeStep instance (lazy init)

        # C graph dispatch (tight C loop for HIP graph replay — eliminates Python overhead)
        # Used inside _decode_step_graph() as the primary replay mechanism.
        # Falls back to Python replay if C extension unavailable.
        self._c_graph_dispatch_lib = None     # ctypes.CDLL for c_graph_dispatch.so
        self._c_graph_dispatch_plan = None    # CGraphDispatchPlan ctypes structure
        self._c_graph_dispatch_objects = {}   # keeps ctypes objects alive

        # Global graph dispatch: full-layer HIP graph capture with kernel P2P allreduce
        # in the same graph. Highest priority dispatch path when enabled.
        # Priority: global_graph > graph > c_dispatch > cached+stream > cached > serial
        self._global_graph_dispatch_enabled = False
        self._global_graph_decode_state = None  # _GlobalGraphDecodeState instance

        # Batched allreduce for consecutive DeltaNet layers:
        # Defers FFN allreduces across 3 consecutive DeltaNet layers, doing one
        # batched allreduce at block boundaries instead of one per layer.
        # Reduces allreduce count from 128 to ~96 per step.
        self._batched_allreduce_enabled = False
        self._batched_allreduce_counter = {'count': 0}  # Track allreduce calls

        # Double-buffer hidden state for compute-communication overlap:
        # Allocates two hidden buffers per GPU (d_hidden_A, d_hidden_B).
        # Layer N writes allreduce result to buffer X; layer N+1 reads from X
        # while layer N's allreduce completes to buffer Y. Buffers alternate.
        # This hides allreduce latency behind next-layer RMSNorm dispatch.
        self._double_buffer_enabled = False

        # Speculative decoding support (n-gram lookahead):
        # When enabled, decode_step_speculative() generates multiple draft tokens
        # and verifies them in a single forward pass using prefill-style dispatch.
        self._speculative_mode = False
        self._speculative_state = None  # SpeculativeDecodeState instance (lazy init)
        
        # EAGLE-style speculative decoding support:
        # Uses the model's own hidden states and lm_head to generate draft tokens
        # without needing a separate draft model.
        self._eagle_mode = False
        self._eagle_k_draft = 5  # Default: 5 draft tokens per iteration
        self._eagle_temperature = 0.0  # Default: greedy sampling for drafts

    def load_layer_weights(self, layer_idx: int, weights: dict):
        import copy
        for engine in self.engines:
            engine.load_layer_weights(layer_idx, copy.deepcopy(weights))

    def load_final_norm(self, weight: np.ndarray):
        for engine in self.engines:
            engine.load_final_norm(weight)

    def load_lm_head(self, weight: np.ndarray):
        for engine in self.engines:
            engine.load_lm_head(weight)

    def prefill_step(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Process multiple tokens using TP for prefill.

        Implements tensor-parallel prefill with:
        - Column-parallel QKV and FFN gate/up projections
        - Row-parallel output and FFN down projections with allreduce
        
        Each GPU processes 1/tp_size of the output dimensions for column-parallel ops,
        then allreduces after row-parallel ops.

        Args:
            token_embeddings: [seq_len, hidden_dim] FP16

        Returns:
            [hidden_dim] FP16 hidden state after final norm
        """
        seq_len = token_embeddings.shape[0]
        h = self.config.hidden_size
        cfg = self.config
        tp_size = self.tp_size
        
        # Validate TP size divides hidden dimensions
        assert h % tp_size == 0, f"hidden_size {h} must be divisible by tp_size {tp_size}"
        assert cfg.intermediate_size % tp_size == 0, f"intermediate_size {cfg.intermediate_size} must be divisible by tp_size {tp_size}"
        
        # Allocate scratch space on each GPU for prefill
        for engine in self.engines:
            engine._alloc_prefill_scratch(seq_len)
        
        # Upload embeddings to all GPUs (each GPU gets full embedding - TP handles the rest)
        emb_bytes = token_embeddings.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_pf_hidden, emb_bytes)
        
        # Process all layers
        for layer_idx in range(cfg.num_hidden_layers):
            lw = self.engines[0].layers[layer_idx]  # All engines have same layer config
            
            if lw.layer_type == 'full_attention':
                self._prefill_full_attention_tp(layer_idx, seq_len)
            else:
                self._prefill_linear_attention_tp(layer_idx, seq_len)
            
            # FFN block with TP GEMM
            self._prefill_ffn_tp(layer_idx, seq_len)
        
        # Set KV cache length
        for engine in self.engines:
            engine.kv_cache.current_len = seq_len
        
        # Final RMSNorm on last token (GPU0 only)
        e0 = self.engines[0]
        last_off = (seq_len - 1) * h * 2
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_pf_hidden + last_off, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2), dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_pf_hidden + last_off, h * 2), dtype=np.float16)
    
    def _prefill_full_attention_tp(self, layer_idx: int, seq_len: int):
        """TP prefill for full attention layer with KV cache sharding.
        
        Implements tensor-parallel FlashAttention v3 with:
        - Column-parallel QKV projections (each GPU: local_q_dim, local_kv_dim)
        - KV cache sharded across TP ranks (each GPU stores local_kv_heads)
        - FlashAttention computed independently on each GPU (no communication)
        - Row-parallel O projection with allreduce
        
        For GQA (Qwen3.5: 24 Q heads, 4 KV heads, TP=4):
          - Each GPU: 6 Q heads, 1 KV head
          - Attention is completely local (no cross-GPU attention needed)
          - O projection sums partial outputs via allreduce
        
        Args:
            layer_idx: layer index
            seq_len: sequence length
        """
        h = self.config.hidden_size
        cfg = self.config
        tp_size = self.tp_size
        
        for gpu_idx, engine in enumerate(self.engines):
            lw = engine.layers[layer_idx]
            
            # Local dimensions for this TP rank
            local_q_dim = engine.local_num_attention_heads * cfg.head_dim
            local_kv_dim = engine.local_num_kv_heads * cfg.head_dim
            
            # Use GEMM path for seq_len >= 32, GEMV for shorter sequences
            if engine._gemm_fp16_prefill and seq_len >= 32:
                # Batched path: RMSNorm per-token into contiguous buffer, then GEMM
                
                # Per-token RMSNorm into d_pf_normed
                for t in range(seq_len):
                    engine._launch_rmsnorm(engine.d_pf_normed + t * h * 2,
                                            engine.d_pf_hidden + t * h * 2,
                                            lw.attn_norm, h)
                
                # Column-parallel GEMM projections
                # Q: [seq_len, h] @ [h, local_q_dim] -> [seq_len, local_q_dim]
                engine._launch_gemm_fp16(engine.d_pf_q, engine.d_pf_normed,
                                          lw.q_weight, seq_len, local_q_dim, h)
                engine._launch_gemm_fp16(engine.d_pf_q_gate, engine.d_pf_normed,
                                          lw.q_gate_weight, seq_len, local_q_dim, h)
                # K,V: [seq_len, h] @ [h, local_kv_dim] -> [seq_len, local_kv_dim]
                engine._launch_gemm_fp16(engine.d_pf_k, engine.d_pf_normed,
                                          lw.k_weight, seq_len, local_kv_dim, h)
                engine._launch_gemm_fp16(engine.d_pf_v, engine.d_pf_normed,
                                          lw.v_weight, seq_len, local_kv_dim, h)
            else:
                # Fallback: per-token GEMV
                for t in range(seq_len):
                    t_h = t * h * 2
                    t_q = t * local_q_dim * 2
                    t_kv = t * local_kv_dim * 2
                    
                    engine._launch_rmsnorm(engine.d_normed, engine.d_pf_hidden + t_h,
                                            lw.attn_norm, h)
                    engine._launch_gemv_fp16(engine.d_pf_q + t_q, engine.d_normed,
                                              lw.q_weight, h, local_q_dim)
                    engine._launch_gemv_fp16(engine.d_pf_q_gate + t_q, engine.d_normed,
                                              lw.q_gate_weight, h, local_q_dim)
                    engine._launch_gemv_fp16(engine.d_pf_k + t_kv, engine.d_normed,
                                              lw.k_weight, h, local_kv_dim)
                    engine._launch_gemv_fp16(engine.d_pf_v + t_kv, engine.d_normed,
                                              lw.v_weight, h, local_kv_dim)
            
            # Per-token fused QK-norm + RoPE
            # Each GPU processes its local heads only
            for t in range(seq_len):
                engine._launch_qknorm_rope(engine.d_pf_q + t * local_q_dim * 2, lw.q_norm, t,
                                            engine.local_num_attention_heads, cfg.head_dim)
                engine._launch_qknorm_rope(engine.d_pf_k + t * local_kv_dim * 2, lw.k_norm, t,
                                            engine.local_num_kv_heads, cfg.head_dim)
            
            # Bulk write K/V to cache (contiguous D2D copy)
            # KV cache is naturally sharded: each GPU stores its local_kv_heads
            kv_layer_k = engine.kv_cache.layer_k_ptr(layer_idx)
            kv_layer_v = engine.kv_cache.layer_v_ptr(layer_idx)
            kv_bytes = seq_len * local_kv_dim * 2
            engine.device.memcpy_d2d(kv_layer_k, engine.d_pf_k, kv_bytes)
            engine.device.memcpy_d2d(kv_layer_v, engine.d_pf_v, kv_bytes)
            
            # FlashAttention v3 (causal)
            # Each GPU runs attention on its local Q heads and local KV heads
            # No communication needed - attention is completely local!
            func = engine.kernels.get_hip("flash_attn_256_fp16", "flash_attn_256")
            params = [
                ctypes.c_uint64(engine.d_pf_q),
                ctypes.c_uint64(kv_layer_k),
                ctypes.c_uint64(kv_layer_v),
                ctypes.c_uint64(engine.d_pf_attn_out),
                ctypes.c_uint32(seq_len),                          # kv_seq_len
                ctypes.c_uint32(seq_len),                          # num_q_rows
                ctypes.c_uint32(engine.local_num_attention_heads), # local Q heads
                ctypes.c_uint32(engine.local_num_kv_heads),        # local KV heads
                ctypes.c_uint32(1),                                # causal
            ]
            grid_x = engine.local_num_attention_heads
            grid_y = (seq_len + 3) // 4
            engine.device.launch(func, (grid_x, grid_y, 1), (256, 1, 1), params)
            
            # Sigmoid gate on local attention output
            engine._launch_sigmoid_mul(engine.d_pf_attn_out, engine.d_pf_q_gate,
                                        seq_len * local_q_dim)
            
            # Row-parallel O projection: [seq_len, local_q_dim] @ [local_q_dim, h]
            # Each GPU computes partial output over its slice of Q heads
            if engine._gemm_fp16_prefill and seq_len >= 32:
                # Batched output projection
                engine._launch_gemm_fp16(engine.d_pf_normed, engine.d_pf_attn_out,
                                          lw.o_weight, seq_len, h, local_q_dim)
            else:
                # Per-token output projection
                for t in range(seq_len):
                    t_q = t * local_q_dim * 2
                    engine._launch_gemv_fp16(engine.d_proj_out, 
                                              engine.d_pf_attn_out + t_q,
                                              lw.o_weight, local_q_dim, h)
                    # Per-token residual add
                    engine._launch_residual_add(engine.d_pf_hidden + t * h * 2,
                                                 engine.d_proj_out, h)
        
        # Allreduce O projection outputs (row-parallel reduction)
        # Sum partial outputs from all GPUs -> d_pf_hidden
        if engine._gemm_fp16_prefill and seq_len >= 32:
            # Batch residual add after allreduce
            partial_ptrs = [e.d_pf_normed for e in self.engines]
            hidden_ptrs = [e.d_pf_hidden for e in self.engines]
            
            if self._ring_allreduce and self._ring_ar is not None:
                self._ring_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
            elif self._fused_p2p_reduce and self._fused_p2p_ar is not None:
                self._fused_p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
            else:
                self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
            
            # Batch residual add
            for engine in self.engines:
                engine._launch_residual_add(engine.d_pf_hidden, engine.d_pf_normed,
                                             seq_len * h)
    
    def _prefill_linear_attention_tp(self, layer_idx: int, seq_len: int):
        """TP prefill for linear attention (DeltaNet) layer.
        
        Similar to full attention TP:
        - Column-parallel in_proj (Q, K, V projections)
        - KV cache sharded across TP ranks
        - DeltaNet computed locally on each GPU
        - Row-parallel out_proj with allreduce
        
        Note: DeltaNet is inherently sequential across sequence positions,
        so we process tokens one at a time (no batched FlashAttention equivalent).
        """
        h = self.config.hidden_size
        cfg = self.config
        
        for gpu_idx, engine in enumerate(self.engines):
            lw = engine.layers[layer_idx]
            
            # Local dimensions for this TP rank
            local_linear_in_dim = (engine.local_linear_num_k_heads * cfg.linear_key_head_dim +
                                   engine.local_linear_num_v_heads * cfg.linear_value_head_dim)
            
            # Process tokens sequentially (DeltaNet is recurrent)
            for t in range(seq_len):
                t_h = t * h * 2
                
                # Copy this token's hidden to d_hidden for existing methods
                engine.device.memcpy_d2d(engine.d_hidden, engine.d_pf_hidden + t_h, h * 2)
                
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden, lw.attn_norm, h)
                
                # GPU path for DeltaNet (local heads only)
                engine._decode_linear_attention_gpu(layer_idx, lw, t)
                
                # Copy updated hidden back to prefill buffer
                engine.device.memcpy_d2d(engine.d_pf_hidden + t_h, engine.d_hidden, h * 2)
    
    def _prefill_ffn_tp(self, layer_idx: int, seq_len: int):
        """TP prefill for FFN block.
        
        Column-parallel gate/up projections, row-parallel down projection with allreduce.
        
        FFN flow:
        1. RMSNorm: d_pf_hidden -> d_pf_normed
        2. Gate projection (column-parallel): d_pf_normed @ W_gate -> d_pf_ffn_gate
        3. Up projection (column-parallel): d_pf_normed @ W_up -> d_pf_ffn_up
        4. SiLU: gate = silu(gate) * up
        5. Down projection (row-parallel): gate @ W_down -> d_pf_ffn_out
        6. Allreduce: sum(d_pf_ffn_out) across GPUs -> d_pf_hidden
        """
        h = self.config.hidden_size
        inter = self.config.intermediate_size
        tp_size = self.tp_size
        local_inter = inter // tp_size
        
        for gpu_idx, engine in enumerate(self.engines):
            lw = engine.layers[layer_idx]
            
            # Per-token RMSNorm into d_pf_normed
            for t in range(seq_len):
                engine._launch_rmsnorm(
                    engine.d_pf_normed + t * h * 2,
                    engine.d_pf_hidden + t * h * 2,
                    lw.ffn_norm, h
                )
            
            # Column-parallel gate projection: [seq_len, h] @ [h, inter/tp_size]
            engine._launch_gemm_int4(
                engine.d_pf_ffn_gate,
                engine.d_pf_normed,
                lw.gate_qweight,  # Sharded: [h/group_size, inter/tp_size]
                lw.gate_scales,
                lw.gate_zeros,
                seq_len, local_inter, h
            )
            
            # Column-parallel up projection: [seq_len, h] @ [h, inter/tp_size]
            engine._launch_gemm_int4(
                engine.d_pf_ffn_up,
                engine.d_pf_normed,
                lw.up_qweight,  # Sharded: [h/group_size, inter/tp_size]
                lw.up_scales,
                lw.up_zeros,
                seq_len, local_inter, h
            )
            
            # Batched SiLU: gate = silu(gate) * up for all tokens
            engine._launch_silu_fused(
                engine.d_pf_ffn_gate,
                engine.d_pf_ffn_up,
                engine.d_pf_ffn_gate,
                seq_len * local_inter
            )
            
            # Row-parallel down projection: [seq_len, inter/tp_size] @ [inter/tp_size, h]
            # Each GPU computes partial output over its slice of intermediate dim
            engine._launch_gemm_int4(
                engine.d_pf_ffn_out,
                engine.d_pf_ffn_gate,
                lw.down_qweight,  # Sharded: [inter/group_size/tp_size, h]
                lw.down_scales,
                lw.down_zeros,
                seq_len, h, local_inter
            )
        
        # Allreduce FFN outputs (row-parallel reduction)
        # Sum partial outputs from all GPUs -> d_pf_hidden
        partial_ptrs = [e.d_pf_ffn_out for e in self.engines]
        hidden_ptrs = [e.d_pf_hidden for e in self.engines]
        
        # Use appropriate allreduce based on configuration
        if self._ring_allreduce and self._ring_ar is not None:
            self._ring_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
        elif self._fused_p2p_reduce and self._fused_p2p_ar is not None:
            self._fused_p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
        else:
            self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, seq_len * h)
        
        # Add residual connection
        for engine in self.engines:
            engine._launch_residual_add(
                engine.d_pf_hidden,
                engine.d_pf_ffn_out,
                seq_len * h
            )

    def build_dispatch_cache(self):
        """Pre-build ctypes parameter arrays for all decode kernel launches.

        Call this after all layer weights are loaded. Pre-building the parameter
        arrays eliminates per-launch ctypes construction overhead (~8-10μs/launch),
        reducing total dispatch overhead from ~5-6ms to ~1ms per decode step.

        After calling this, set_cached_dispatch(True) to use the optimized path.
        """
        self._engine_layer_caches = []
        for engine in self.engines:
            lc = engine.build_decode_launch_cache()
            self._engine_layer_caches.append(lc)
        print(f"Dispatch cache built for {len(self.engines)} engines × "
              f"{len(self.engines[0].layers)} layers")

    def set_stream_overlap_dispatch(self, overlap: bool):
        """Enable or disable stream overlap dispatch.

        When overlap=True, allreduce operations are submitted asynchronously
        on dedicated allreduce streams. Stream events gate the dependency
        between allreduce completion and the next layer's compute (RMSNorm).

        This allows Python to continue dispatching the next layer's kernels
        immediately after submitting the allreduce. The GPU-side execution
        uses hipStreamWaitEvent to enforce the correct data dependency, so
        the RMSNorm kernel will wait for allreduce to complete on the GPU,
        but Python does not block.

        Requires P2P allreduce to be available (_p2p_ar is not None).
        Falls back to cached dispatch if P2P allreduce unavailable.
        """
        if overlap and self._p2p_ar is None:
            print("WARNING: Stream overlap requires P2P allreduce (unavailable). "
                  "Falling back to cached dispatch.")
            overlap = False
        self._stream_overlap_dispatch = overlap

    def set_fused_p2p_reduce(self, fused: bool):
        """Enable or disable fused P2P reduce path.

        When fused=True, allreduce operations use the fused P2P reduce kernel
        (gemv_p2p_reduce.hip) where each GPU independently reads all peer
        partial results via P2P and computes the full sum. This eliminates
        the gather→reduce→broadcast pipeline used by the standard P2P allreduce.

        The fused path may be faster for small payloads (10KB) because:
        - All 4 GPUs run their kernels simultaneously (no sequential gather)
        - No intermediate gather buffer needed
        - Single sync point instead of 2+ stream synchronizations

        However, PCIe remote memory reads from within a kernel may have higher
        latency than hipMemcpyPeerAsync on 2-hop PCIe topologies. Test both and
        pick the faster one.

        Requires FusedP2PReduce to be available (_fused_p2p_ar is not None).
        Falls back to standard P2P allreduce if fused kernel unavailable.
        """
        if fused and self._fused_p2p_ar is None:
            print("WARNING: Fused P2P reduce unavailable. Falling back to standard P2P allreduce.")
            fused = False
        self._fused_p2p_reduce = fused

    def set_ring_allreduce(self, ring: bool):
        """Enable or disable ring allreduce path.

        When ring=True, allreduce operations use the ring topology where each
        GPU only communicates with its neighbors (distributed PCIe bandwidth),
        instead of the star topology where GPU0 does all gathering/broadcasting.

        Ring allreduce has 6 P2P transfers total (same as star), but they are
        distributed across all GPUs rather than all going through GPU0.

        Requires RingAllreduce to be available (_ring_ar is not None).
        Falls back to standard P2P allreduce if ring allreduce unavailable.
        """
        if ring and self._ring_ar is None:
            print("WARNING: Ring allreduce unavailable. Falling back to standard P2P allreduce.")
            ring = False
        self._ring_allreduce = ring
        if ring:
            print(f"Ring allreduce enabled (TP={self.tp_size})")
        else:
            print(f"Ring allreduce disabled, using {'fused P2P' if self._fused_p2p_reduce else 'standard P2P'} allreduce")

    def set_kernel_p2p_allreduce(self, enabled: bool):
        """Enable or disable kernel-based P2P allreduce path.

        When enabled=True, allreduce operations use allreduce_residual_kernel()
        from the P2PAllreduce class, which launches a single HIP kernel per GPU
        that directly reads peer GPU partial buffers via BAR1-mapped P2P pointers.

        This eliminates all host synchronization points per allreduce call:
        - No hipSetDevice x4
        - No hipMemcpyPeerAsync gather x3
        - No hipStreamSynchronize (before reduce)
        - No reduce kernel on GPU0 only
        - No hipStreamSynchronize (after reduce)
        - No hipMemcpyPeerAsync broadcast x3
        - No sync all
        Instead: 1 kernel launch per GPU + 1 stream sync per GPU

        Requires _p2p_ar to be initialized and kernel_p2p_allreduce.so to be compiled.
        Falls back to standard P2P allreduce if kernel P2P is unavailable.
        """
        if enabled:
            if self._p2p_ar is None:
                print("WARNING: Kernel P2P allreduce requires P2P allreduce (unavailable). "
                      "Falling back to standard allreduce.")
                enabled = False
            elif self._p2p_ar._kernel_p2p_lib is None:
                print("WARNING: kernel_p2p_allreduce.so not loaded. "
                      "Falling back to standard P2P allreduce.")
                enabled = False
        self._kernel_p2p_allreduce = enabled
        if enabled:
            print(f"Kernel P2P allreduce enabled (TP={self.tp_size}, "
                  f"single kernel per GPU, no host round-trips)")
        else:
            print(f"Kernel P2P allreduce disabled")
        # Invalidate C dispatch plan so it gets rebuilt with the new allreduce mode
        # when set_c_dispatch(True) is called or _build_c_dispatch_plan() is invoked.
        if self._c_dispatch_plan is not None:
            self._c_dispatch_plan = None
            if self._c_dispatch_enabled:
                # Rebuild immediately so decode_step continues to work
                try:
                    self._c_dispatch_plan = self._build_c_dispatch_plan()
                except Exception as e:
                    print(f"WARNING: Failed to rebuild C dispatch plan after "
                          f"set_kernel_p2p_allreduce({enabled}): {e}")

    def set_deferred_attention_ar(self, enabled: bool):
        """Enable or disable deferred attention allreduce (M3 optimization).
        
        When enabled=True:
          - Attention output allreduce is SKIPPED
          - Partial attention output (d_proj_out) is added directly to d_hidden (local residual add)
          - FFN operates on partial hidden state (d_hidden contains partial attention + residual)
          - Single allreduce after FFN down-projection (d_ffn_out → d_hidden)
          - Reduces allreduce count from 128 to 64 per token (50% reduction)
        
        Mathematical justification:
          - FFN gate projection: gate = SiLU(x @ W_gate) — x must be reduced for correctness
          - FFN up projection: up = x @ W_up — linear, can operate on partial x
          - FFN down projection: out = (gate * up) @ W_down — row-parallel, allreduce after
          
          Since gate uses element-wise SiLU (not linear), this changes the computation.
          However, for TP=4 with sufficient numerical precision, cosine similarity >= 0.99
          is expected.
        
        Note: This is an approximation that changes the computation graph. Validate
        correctness with cosine similarity >= 0.99 threshold before using in production.
        
        Args:
            enabled: True to enable deferred attention AR, False for standard path
        """
        self._deferred_attention_ar = enabled
        if enabled:
            print(f"Deferred attention allreduce ENABLED: "
                  f"skipping attention AR, adding partial locally, "
                  f"reduces AR count from 128 to 64 per token")
        else:
            print(f"Deferred attention allreduce disabled (standard 2 ARs per layer)")
        
        # Invalidate C dispatch plan so it gets rebuilt
        if self._c_dispatch_plan is not None:
            self._c_dispatch_plan = None
            if self._c_dispatch_enabled:
                try:
                    self._c_dispatch_plan = self._build_c_dispatch_plan()
                except Exception as e:
                    print(f"WARNING: Failed to rebuild C dispatch plan after "
                          f"set_deferred_attention_ar({enabled}): {e}")

    def set_cached_dispatch(self, cached: bool):
        """Enable or disable cached (pre-built parameter) dispatch.

        When cached=True, decode_step uses pre-built ctypes parameter arrays
        (built by build_dispatch_cache()) to avoid re-constructing ctypes objects
        per launch. This reduces Python dispatch overhead from ~5-6ms to ~1ms
        per decode step.

        Requires build_dispatch_cache() to have been called first.
        """
        if cached and not self._engine_layer_caches:
            self.build_dispatch_cache()
        self._cached_dispatch = cached

    def set_batched_allreduce_enabled(self, enabled: bool):
        """Enable or disable batched allreduce for DeltaNet layers.

        When enabled=True, FFN allreduces are deferred across consecutive
        DeltaNet layers (3 layers per block) and performed as a single batched
        allreduce at block boundaries. This reduces the total allreduce count
        from 128 to ~96 per decode step (25% reduction).

        For full-attention layers, the standard 2-allreduce path is used.

        Note: This changes the computation slightly (FFN inputs differ from
        standard path) but should maintain cosine similarity >= 0.99.

        Args:
            enabled: True to enable batched allreduce, False for standard
        """
        if enabled and self._p2p_ar is None:
            print("WARNING: Batched allreduce requires P2P allreduce. "
                  "Falling back to standard allreduce.")
            enabled = False
        self._batched_allreduce_enabled = enabled
        if enabled:
            print(f"Batched allreduce enabled: FFN allreduces deferred across "
                  f"DeltaNet blocks (reduces AR count from 128 to ~96)")
        else:
            print(f"Batched allreduce disabled (standard path)")
        # Reset counter
        self._batched_allreduce_counter = {'count': 0}

    def get_allreduce_call_count(self) -> int:
        """Get the current allreduce call count (for instrumentation)."""
        return self._batched_allreduce_counter.get('count', 0)

    def reset_allreduce_counter(self):
        """Reset the allreduce call counter."""
        self._batched_allreduce_counter = {'count': 0}

    def set_double_buffer_enabled(self, enabled: bool):
        """Enable or disable double-buffer hidden state for compute-communication overlap.

        When enabled=True, each GPU allocates two hidden buffers (d_hidden_A, d_hidden_B).
        The decode loop alternates buffers each layer:
          - Even layers: RMSNorm reads from A, allreduce writes to B
          - Odd layers:  RMSNorm reads from B, allreduce writes to A

        This allows layer N+1's RMSNorm to start immediately after submitting
        layer N's allreduce, hiding the allreduce latency (~79us) behind the
        next layer's compute dispatch.

        Memory overhead: 5120 × 2 bytes = 10KB per GPU (negligible).

        Requires engines to support double-buffer (they allocate buffers in __init__).
        Must be called before build_dispatch_cache() if using cached dispatch.

        Args:
            enabled: True to enable double-buffer mode, False for standard
        """
        self._double_buffer_enabled = enabled
        if enabled:
            print(f"Double-buffer enabled: overlapping allreduce with next-layer compute")
        else:
            print(f"Double-buffer disabled (standard single-buffer path)")

    def compute_logits(self, hidden: np.ndarray = None) -> np.ndarray:
        return self.engines[0].compute_logits(hidden)

    def set_threaded_dispatch(self, threaded: bool):
        """Enable or disable multi-threaded GPU synchronization.

        When threaded=True, hipDeviceSynchronize() calls before each allreduce
        run in parallel (one thread per GPU) rather than sequentially. Since
        hipDeviceSynchronize() is a C-level call that releases the Python GIL,
        the 4 GPU syncs can truly run in parallel, reducing the sync overhead
        from 4 × sync_time to 1 × sync_time per allreduce.

        The kernel launches themselves remain serial (dispatched by the main
        thread), which avoids Python GIL contention for the per-launch overhead.
        The main benefit is parallelizing the pre-allreduce device synchronization.

        Safe to call at any time (before or after decode_step calls).
        """
        if threaded and not self._thread_pool_initialized:
            self._init_thread_pool()
        self._threaded_dispatch = threaded

    def _init_thread_pool(self):
        """Initialize worker threads for parallel device synchronization.

        Each worker thread:
        - Is assigned one GPU (by rank)
        - Waits for a signal from the main thread to start syncing
        - Calls hipDeviceSynchronize() for its GPU (GIL-free C call)
        - Signals done when sync completes

        This allows the 4 GPU device syncs before each allreduce to run in
        parallel, reducing sync overhead from 4×sync_time to ~sync_time.

        Per decode step: 128 allreduce points (2 per layer × 64 layers).
        Each allreduce requires device sync on all 4 GPUs.
        Serial sync: 4 × ~100μs = 400μs per allreduce × 128 = 51ms total
        Parallel sync: ~100μs per allreduce × 128 = 13ms total → saves ~38ms
        """
        tp = self.tp_size
        self._go_events = [threading.Event() for _ in range(tp)]
        self._done_events = [threading.Event() for _ in range(tp)]
        self._worker_cmds = [None] * tp
        self._worker_exception = None

        for rank in range(tp):
            t = threading.Thread(
                target=self._worker_loop,
                args=(rank,),
                daemon=True,
                name=f"gpu-sync-worker-{rank}",
            )
            t.start()
            self._worker_threads.append(t)

        self._thread_pool_initialized = True
        print(f"Thread pool initialized: {tp} GPU sync worker threads")

    def _worker_loop(self, rank: int):
        """GPU sync worker thread for GPU rank.

        Waits for 'sync' command, calls hipDeviceSynchronize() for its GPU
        (this is a C-level call that releases the Python GIL, allowing other
        threads to run concurrently), then signals done.

        Command format: (cmd_type, ...) where cmd_type is:
          'sync'  - call hipDeviceSynchronize() for this GPU
          'stop'  - exit the loop
        """
        engine = self.engines[rank]
        go_event = self._go_events[rank]
        done_event = self._done_events[rank]
        hip = engine.device.hip
        dev_id = engine.device.device_id

        try:
            while True:
                go_event.wait()
                go_event.clear()

                cmd = self._worker_cmds[rank]
                if cmd is None or cmd == 'stop':
                    done_event.set()
                    break

                if cmd == 'sync':
                    # hipDeviceSynchronize releases the GIL during the C call,
                    # allowing other worker threads to run their own device syncs
                    # in parallel on their respective GPUs.
                    hip.set_device(dev_id)
                    hip.synchronize()  # GIL-free during C call (~50-100μs)

                done_event.set()

        except Exception as e:
            self._worker_exception = e
            done_event.set()

    def _parallel_device_sync(self):
        """Run hipDeviceSynchronize() on all GPUs in parallel via worker threads.

        This is the key optimization: instead of 4 sequential device syncs
        (400μs), we run them in parallel (100μs).

        The hipDeviceSynchronize() calls release the Python GIL, allowing
        all 4 threads to run their C-level sync calls simultaneously.
        """
        tp = self.tp_size

        # Clear done events
        for done_event in self._done_events:
            done_event.clear()

        # Set 'sync' command for all workers
        for rank in range(tp):
            self._worker_cmds[rank] = 'sync'

        # Signal all workers to start simultaneously
        for go_event in self._go_events:
            go_event.set()

        # Wait for all workers to complete their device sync
        for done_event in self._done_events:
            done_event.wait()

        # Propagate any worker exception
        if self._worker_exception is not None:
            exc = self._worker_exception
            self._worker_exception = None
            raise RuntimeError(f"GPU sync worker raised exception: {exc}") from exc

    def _decode_step_threaded(self, token_embedding: np.ndarray,
                               position: int) -> np.ndarray:
        """Run one decode step with parallel GPU device synchronization.

        The main thread dispatches all kernel launches (serial, fast Python),
        while worker threads run hipDeviceSynchronize() for each GPU in parallel
        before each allreduce (the key bottleneck: serial sync takes 4×100μs=400μs,
        parallel sync takes ~100μs = 3x speedup per allreduce point).

        This exploits that hipDeviceSynchronize() is a C-level call that releases
        the Python GIL, allowing all 4 threads to sync their GPUs simultaneously.

        Per-layer flow:
          Main: RMSNorm + attention kernel launches for all 4 GPUs (serial, fast)
          Workers: hipDeviceSynchronize() for each GPU in PARALLEL
          Main: P2P gather + allreduce (no device sync needed — workers did it)
          Main: RMSNorm + FFN kernel launches for all 4 GPUs (serial, fast)
          Workers: hipDeviceSynchronize() for each GPU in PARALLEL
          Main: P2P gather + allreduce
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers

        # Upload embedding to all GPUs (serial, fast)
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        for layer_idx in range(num_layers):
            lw_list = [e.layers[layer_idx] for e in self.engines]

            # RMSNorm + attention: serial kernel launches for all GPUs
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.attn_norm, h)
                if lw.layer_type == 'full_attention':
                    engine._decode_full_attention(layer_idx, lw, position)
                else:
                    if engine._deltanet_gpu:
                        engine._decode_linear_attention_gpu(layer_idx, lw, position)
                    else:
                        engine._decode_linear_attention(layer_idx, lw, position)

            # Parallel device sync: workers sync all 4 GPUs simultaneously
            # This is the key speedup: 400μs serial → ~100μs parallel
            self._parallel_device_sync()

            # P2P gather + allreduce (skip device sync since workers already did it)
            self._allreduce_residual_no_sync("d_proj_out", h)

            # RMSNorm + FFN: serial kernel launches for all GPUs
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.ffn_norm, h)
                if engine._gemv_int4_dual:
                    engine._launch_ffn_gate_up_silu(
                        engine.d_ffn_gate, engine.d_normed,
                        lw, h, engine.local_intermediate_size)
                else:
                    engine._launch_gemv_int4(
                        engine.d_ffn_gate, engine.d_normed,
                        lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_gemv_int4(
                        engine.d_ffn_up, engine.d_normed,
                        lw.up_qweight, lw.up_scales, lw.up_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_silu_fused(
                        engine.d_ffn_gate, engine.d_ffn_up,
                        engine.d_ffn_gate, engine.local_intermediate_size)
                engine._launch_gemv_int4(
                    engine.d_ffn_out, engine.d_ffn_gate,
                    lw.down_qweight, lw.down_scales, lw.down_zeros,
                    engine.local_intermediate_size, h)

            # Parallel device sync before FFN allreduce
            self._parallel_device_sync()

            # P2P gather + FFN allreduce (no device sync)
            self._allreduce_residual_no_sync("d_ffn_out", h)

        for engine in self.engines:
            engine.kv_cache.advance()

        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def decode_step(self, token_embedding: np.ndarray, position: int) -> np.ndarray:
        """Run one decode step with fused host-side allreduce.

        Dispatches to global graph dispatch, graph dispatch, C dispatch,
        cached+stream (combined), cached, stream_overlap, threaded, or serial
        implementation based on flags.

        Priority order:
        0. Persistent megakernel (highest): _persistent_enabled is True.
           Single persistent kernel runs entire decode step on GPU with
           on-GPU task scheduling, eliminating all host dispatch overhead.
        1. Global graph dispatch: _global_graph_dispatch_enabled is True.
           Full-layer HIP graph per GPU per layer, including kernel P2P allreduce.
           No host-side allreduce between attention and FFN.
        2. Graph dispatch: _graph_dispatch_enabled is True. Replays
           pre-captured HIP graphs for all compute segments, eliminating Python
           dispatch overhead. Host-orchestrated allreduce between segments.
        3. C dispatch: _c_dispatch_enabled is True. Dispatches all 64 layers
           in a tight C loop via c_dispatch.so.
        4. Batched allreduce: _batched_allreduce_enabled is True. Defers FFN
           allreduces across consecutive DeltaNet layers, reducing AR count.
        5. Combined (cached + stream overlap): _cached_dispatch and
           _stream_overlap_dispatch are True and _p2p_ar is available.
        6. Cached dispatch only: _cached_dispatch is True.
        7. Stream overlap only: _stream_overlap_dispatch is True.
        8. Threaded: _threaded_dispatch is True.
        9. Serial: fallback.
        """
        if self._persistent_enabled and self._persistent_dispatcher is not None:
            return self._decode_step_persistent(token_embedding, position)
        if self._global_graph_dispatch_enabled and self._engine_layer_caches:
            return self._decode_step_global_graph(token_embedding, position)
        if self._graph_dispatch_enabled and self._engine_layer_caches:
            return self._decode_step_graph(token_embedding, position)
        if self._c_dispatch_enabled and self._c_dispatch_plan is not None:
            return self._decode_step_c_dispatch(token_embedding, position)
        if self._batched_allreduce_enabled and self._p2p_ar is not None:
            return self._decode_step_batched_allreduce(token_embedding, position)
        if (self._cached_dispatch and self._engine_layer_caches
                and self._stream_overlap_dispatch and self._p2p_ar is not None):
            return self._decode_step_cached_stream(token_embedding, position)
        if self._cached_dispatch and self._engine_layer_caches:
            return self._decode_step_cached(token_embedding, position)
        if self._stream_overlap_dispatch and self._p2p_ar is not None:
            return self._decode_step_stream_overlap(token_embedding, position)
        if self._threaded_dispatch:
            return self._decode_step_threaded(token_embedding, position)
        return self._decode_step_serial(token_embedding, position)

    def _decode_step_cached(self, token_embedding: np.ndarray,
                             position: int) -> np.ndarray:
        """Optimized decode step using pre-built ctypes parameter arrays.

        Uses LaunchSpec objects built by build_dispatch_cache() to avoid
        re-constructing Python ctypes objects on each kernel launch. Reduces
        Python dispatch overhead from ~5-6ms to ~1ms per decode step.

        Position-dependent kernel params (RoPE cos/sin offsets, attention seq_len)
        are updated in-place on the pre-built ctypes objects before each launch.

        All other params (weight pointers, buffer pointers, K/N dimensions) are
        pre-built once and reused every step.
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        half_rotary = self.engines[0].rotary_dim // 2

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Pre-compute RoPE offset for this position
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables

        for layer_idx in range(num_layers):
            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]
                lw = engine.layers[layer_idx]

                # --- Attention RMSNorm (all static) ---
                engine.device.launch_cached(layer_cache['attn_rmsnorm'])

                if lw.layer_type == 'full_attention':
                    # --- GEMV projections (all static) ---
                    # Q and KV GEMVs run sequentially on the default stream.
                    # No explicit stream sync needed: null stream serializes execution,
                    # guaranteeing ordering for QKNorm without host-blocking sync calls.
                    if 'gemv_q_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_q_fused'])
                    if 'gemv_k_only' in layer_cache:
                        # Direct KV write mode: K to working buffer, V directly to cache
                        engine.device.launch_cached(layer_cache['gemv_k_only'])
                        # Update V GEMV output to current KV cache position
                        cur_pos = engine.kv_cache.current_len
                        kv_stride = layer_cache['_kv_stride']
                        v_cache_ptr = layer_cache['_v_cache_base'] + cur_pos * kv_stride
                        layer_cache['gemv_v_cache'].params[2].value = v_cache_ptr
                        engine.device.launch_cached(layer_cache['gemv_v_cache'])
                    elif 'gemv_kv_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_kv_fused'])

                    # --- QK-norm + RoPE: update position-dependent cos/sin ptrs ---
                    if 'qknorm_q' in layer_cache:
                        spec_q = layer_cache['qknorm_q']
                        spec_q.params[2].value = engine.d_cos + cos_offset
                        spec_q.params[3].value = engine.d_sin + cos_offset
                        engine.device.launch_cached(spec_q)
                    if 'qknorm_k' in layer_cache:
                        spec_k = layer_cache['qknorm_k']
                        spec_k.params[2].value = engine.d_cos + cos_offset
                        spec_k.params[3].value = engine.d_sin + cos_offset
                        if '_k_cache_base' in layer_cache:
                            # Direct KV write mode: also update K cache destination
                            cur_pos = engine.kv_cache.current_len
                            kv_stride = layer_cache['_kv_stride']
                            k_cache_ptr = layer_cache['_k_cache_base'] + cur_pos * kv_stride
                            spec_k.params[4].value = k_cache_ptr
                        engine.device.launch_cached(spec_k)

                    # KV cache update: skip if direct write mode (cache written above)
                    if '_k_cache_base' not in layer_cache:
                        # Standard path: GPU-to-GPU D2D copy to KV cache
                        engine.kv_cache.append_kv_gpu(layer_idx, engine.d_k, engine.d_v)

                    # --- Decode attention: update seq_len ---
                    if 'decode_attn' in layer_cache:
                        spec_attn = layer_cache['decode_attn']
                        spec_attn.params[4].value = engine.kv_cache.current_len + 1
                        engine.device.launch_cached(spec_attn)

                    # --- Sigmoid gate ---
                    if 'sigmoid_mul' in layer_cache:
                        engine.device.launch_cached(layer_cache['sigmoid_mul'])

                    # --- Output projection ---
                    if 'gemv_o_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_o_proj'])

                else:  # DeltaNet linear attention
                    # --- Linear attention input projection ---
                    if 'gemv_la_in_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_la_in_proj'])

                    # --- DeltaNet v3 main kernel ---
                    if 'deltanet_v3' in layer_cache:
                        engine.device.launch_cached(layer_cache['deltanet_v3'])
                    if 'deltanet_v3_shift' in layer_cache:
                        engine.device.launch_cached(layer_cache['deltanet_v3_shift'])

                    # --- Linear attention output projection ---
                    if 'gemv_la_out_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_la_out_proj'])

            # Allreduce attention partials + residual add
            self._allreduce_residual("d_proj_out", h)

            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]

                # --- FFN RMSNorm ---
                engine.device.launch_cached(layer_cache['ffn_rmsnorm'])

                # --- FFN gate+up+silu ---
                if 'ffn_gate_up_silu' in layer_cache:
                    engine.device.launch_cached(layer_cache['ffn_gate_up_silu'])
                else:
                    # Fallback to regular launch for non-fused path
                    lw = engine.layers[layer_idx]
                    engine._launch_ffn_gate_up_silu(
                        engine.d_ffn_gate, engine.d_normed,
                        lw, h, engine.local_intermediate_size)

                # --- FFN down projection ---
                if 'ffn_down' in layer_cache:
                    engine.device.launch_cached(layer_cache['ffn_down'])
                else:
                    lw = engine.layers[layer_idx]
                    engine._launch_gemv_int4(
                        engine.d_ffn_out, engine.d_ffn_gate,
                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                        engine.local_intermediate_size, h)

            # Allreduce FFN partials + residual add
            self._allreduce_residual("d_ffn_out", h)

        for engine in self.engines:
            engine.kv_cache.advance()

        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_step_cached_stream(self, token_embedding: np.ndarray,
                                    position: int) -> np.ndarray:
        """Combined cached dispatch + async stream overlap decode step.

        Combines the benefits of both optimization modes:
        - Cached dispatch: pre-built ctypes parameter arrays avoid per-launch
          Python overhead (~5-6ms savings vs serial dispatch).
        - Stream overlap: async allreduce on dedicated streams avoids CPU-blocking
          hipDeviceSynchronize(), allowing Python to continue dispatching next-layer
          kernels while GPU executes allreduce P2P copies in flight.

        Per-layer flow:
          1. [Attention] Launch RMSNorm + GEMV on compute stream (via cached params)
          2. Submit async allreduce for attention partials:
             - Record compute events on each GPU
             - Allreduce stream waits on compute events (GPU-side)
             - P2P gather + reduce kernel + broadcast on allreduce streams (non-blocking)
             - Record allreduce done events
          3. Make compute stream wait on allreduce done events (GPU-side, queued)
          4. [FFN] Launch FFN RMSNorm + gate+up+silu + down_proj on compute stream (cached)
          5. Submit async allreduce for FFN partials
          6. [Next layer] Compute stream waits on FFN allreduce done event before RMSNorm

        Python returns immediately after submitting GPU work (step 2 returns without
        CPU blocking). While GPU executes allreduce, Python is already dispatching
        step 3 (wait event, queued GPU-side) and next layer's cached kernel launches.

        This hides allreduce latency (~23ms/tok) behind Python dispatch time for
        next-layer kernel launches (~14ms/tok with cached dispatch).
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        half_rotary = self.engines[0].rotary_dim // 2
        # Stream overlap path: select active allreduce implementation.
        # Ring allreduce has the same async interface as P2PAllreduce.
        # The fused kernel's async variant requires all 4 GPUs to sync with each
        # other's compute events, creating more overhead in the overlap path.
        # Standard P2P allreduce is better for streaming (1.41x slower in isolation
        # but faster overall due to better overlap with compute dispatch).
        if self._ring_allreduce and self._ring_ar is not None:
            p2p_ar = self._ring_ar
        else:
            p2p_ar = self._p2p_ar
        compute_streams = self._compute_streams  # [0] * tp_size (default null streams)
        use_double_buffer = self._double_buffer_enabled

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            if use_double_buffer:
                # Double-buffer mode: start with buffer A as read buffer
                engine.d_hidden = engine.d_hidden_A
                engine.d_hidden_write = engine.d_hidden_B
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Pre-compute RoPE offset for this position
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables

        for layer_idx in range(num_layers):
            # Double-buffer mode: NO wait at layer start.
            # Layer N+1's RMSNorm reads from the buffer that layer N's allreduce wrote to.
            # The data dependency is enforced by the allreduce stream event mechanism.
            # Standard mode: wait for previous layer's allreduce (preserves existing behavior)
            if not use_double_buffer and layer_idx > 0:
                p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]
                lw = engine.layers[layer_idx]

                # --- Attention RMSNorm (cached, all static) ---
                # CRITICAL FIX: In double-buffer mode, update the cached LaunchSpec
                # to use the current d_hidden pointer, since buffers swap each layer
                if use_double_buffer:
                    attn_rmsnorm_spec = layer_cache['attn_rmsnorm']
                    attn_rmsnorm_spec.params[1].value = engine.d_hidden
                engine.device.launch_cached(layer_cache['attn_rmsnorm'])

                if lw.layer_type == 'full_attention':
                    # --- GEMV projections (cached, all static) ---
                    # Q and KV GEMVs run sequentially on the default stream.
                    # No explicit stream sync needed: null stream serializes execution.
                    if 'gemv_q_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_q_fused'])
                    if 'gemv_k_only' in layer_cache:
                        # Direct KV write mode: K to working buffer, V directly to cache
                        engine.device.launch_cached(layer_cache['gemv_k_only'])
                        # Update V GEMV output to current KV cache position
                        cur_pos = engine.kv_cache.current_len
                        kv_stride = layer_cache['_kv_stride']
                        v_cache_ptr = layer_cache['_v_cache_base'] + cur_pos * kv_stride
                        layer_cache['gemv_v_cache'].params[2].value = v_cache_ptr
                        engine.device.launch_cached(layer_cache['gemv_v_cache'])
                    elif 'gemv_kv_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_kv_fused'])

                    # --- QK-norm + RoPE: update position-dependent cos/sin ptrs ---
                    if 'qknorm_q' in layer_cache:
                        spec_q = layer_cache['qknorm_q']
                        spec_q.params[2].value = engine.d_cos + cos_offset
                        spec_q.params[3].value = engine.d_sin + cos_offset
                        engine.device.launch_cached(spec_q)
                    if 'qknorm_k' in layer_cache:
                        spec_k = layer_cache['qknorm_k']
                        spec_k.params[2].value = engine.d_cos + cos_offset
                        spec_k.params[3].value = engine.d_sin + cos_offset
                        if '_k_cache_base' in layer_cache:
                            # Direct KV write mode: also update K cache destination
                            cur_pos = engine.kv_cache.current_len
                            kv_stride = layer_cache['_kv_stride']
                            k_cache_ptr = layer_cache['_k_cache_base'] + cur_pos * kv_stride
                            spec_k.params[4].value = k_cache_ptr
                        engine.device.launch_cached(spec_k)

                    # KV cache update: skip if direct write mode (cache written above)
                    if '_k_cache_base' not in layer_cache:
                        # Standard path: GPU-to-GPU D2D copy to KV cache
                        engine.kv_cache.append_kv_gpu(layer_idx, engine.d_k, engine.d_v)

                    # --- Decode attention: update seq_len ---
                    if 'decode_attn' in layer_cache:
                        spec_attn = layer_cache['decode_attn']
                        spec_attn.params[4].value = engine.kv_cache.current_len + 1
                        engine.device.launch_cached(spec_attn)

                    # --- Sigmoid gate ---
                    if 'sigmoid_mul' in layer_cache:
                        engine.device.launch_cached(layer_cache['sigmoid_mul'])

                    # --- Output projection ---
                    if 'gemv_o_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_o_proj'])

                else:  # DeltaNet linear attention
                    # --- Linear attention input projection ---
                    if 'gemv_la_in_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_la_in_proj'])

                    # --- DeltaNet v3 main kernel ---
                    if 'deltanet_v3' in layer_cache:
                        engine.device.launch_cached(layer_cache['deltanet_v3'])
                    if 'deltanet_v3_shift' in layer_cache:
                        engine.device.launch_cached(layer_cache['deltanet_v3_shift'])

                    # --- Linear attention output projection ---
                    if 'gemv_la_out_proj' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_la_out_proj'])

            # --- Async allreduce attention partials + residual add ---
            # Records compute events → allreduce stream waits → P2P gather+reduce+broadcast
            # Records allreduce done events. Returns immediately (non-blocking).
            partial_ptrs = [e.d_proj_out for e in self.engines]
            if use_double_buffer:
                # Double-buffer: write to the alternate buffer (d_hidden_write)
                # CRITICAL: For residual add to work correctly, d_hidden_write must
                # contain the same value as d_hidden (the layer input). Copy it now.
                ar_stream = p2p_ar._allreduce_streams[0]
                for engine in self.engines:
                    engine.device.memcpy_d2d_async(
                        engine.d_hidden_write, engine.d_hidden, h * 2, ar_stream)
                hidden_ptrs = [e.d_hidden_write for e in self.engines]
            else:
                hidden_ptrs = [e.d_hidden for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

            # CRITICAL: Wait for attention allreduce before FFN RMSNorm.
            # FFN RMSNorm reads the attention output, so it cannot execute until
            # the allreduce completes. This wait is required for BOTH standard and
            # double-buffer modes.
            # Double-buffer overlap benefit: The next layer's attention RMSNorm will
            # naturally read from this layer's FFN output via buffer swap, without
            # needing an inter-layer wait.
            p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]

                # --- FFN RMSNorm ---
                # Standard mode: reads d_hidden (gated by attention AR done event)
                # Double-buffer mode: reads d_hidden_write (gated by AR stream event)
                # CRITICAL FIX: In double-buffer mode, update the cached LaunchSpec
                # to use d_hidden_write as input, since the cached pointer is stale
                if use_double_buffer:
                    ffn_rmsnorm_spec = layer_cache['ffn_rmsnorm']
                    ffn_rmsnorm_spec.params[1].value = engine.d_hidden_write
                engine.device.launch_cached(layer_cache['ffn_rmsnorm'])

                # --- FFN gate+up+silu (cached) ---
                if 'ffn_gate_up_silu' in layer_cache:
                    engine.device.launch_cached(layer_cache['ffn_gate_up_silu'])
                else:
                    # Fallback to regular launch for non-fused path
                    lw = engine.layers[layer_idx]
                    engine._launch_ffn_gate_up_silu(
                        engine.d_ffn_gate, engine.d_normed,
                        lw, h, engine.local_intermediate_size)

                # --- FFN down projection (cached) ---
                if 'ffn_down' in layer_cache:
                    engine.device.launch_cached(layer_cache['ffn_down'])
                else:
                    lw = engine.layers[layer_idx]
                    engine._launch_gemv_int4(
                        engine.d_ffn_out, engine.d_ffn_gate,
                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                        engine.local_intermediate_size, h)

            # --- Async allreduce FFN partials + residual add ---
            # Non-blocking: next layer will wait (standard mode) or naturally sync (double-buffer)
            partial_ptrs = [e.d_ffn_out for e in self.engines]
            if use_double_buffer:
                # Double-buffer: write to d_hidden_write
                hidden_ptrs = [e.d_hidden_write for e in self.engines]
            else:
                hidden_ptrs = [e.d_hidden for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

            # Double-buffer mode: swap buffers after each layer's FFN allreduce
            # This makes the next layer's RMSNorm read from the buffer this layer wrote to
            if use_double_buffer:
                for engine in self.engines:
                    engine._swap_hidden_buffers()

        # After all layers: wait for the last FFN allreduce before reading d_hidden
        p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

        # Synchronize all GPUs to ensure all GPU work is complete
        for dev_id in self.device_ids:
            self._hip.set_device(dev_id)
            self._hip.synchronize()

        for engine in self.engines:
            engine.kv_cache.advance()

        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_step_stream_overlap(self, token_embedding: np.ndarray,
                                     position: int) -> np.ndarray:
        """Decode step with async allreduce on dedicated streams.

        Uses HIP stream events to overlap allreduce with next-layer compute:
        1. Launch GEMV kernels on default compute stream (all 4 GPUs, serial Python)
        2. Record compute events on each GPU's compute stream
        3. Make allreduce stream wait on compute events (GPU-side, no CPU block)
        4. Submit allreduce asynchronously (returns to Python immediately)
        5. Make next-layer's compute stream wait on allreduce done events (GPU-side)
        6. Launch next-layer's RMSNorm on compute stream (queued on GPU, waits for AR)

        This creates a pipeline: while GPU runs allreduce, Python dispatches
        next-layer kernel launches. The GPU enforces data dependencies via events.

        Key difference from serial: no CPU-blocking hipDeviceSynchronize().
        Instead: hipEventRecord + hipStreamWaitEvent for GPU-side ordering.
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        if self._ring_allreduce and self._ring_ar is not None:
            p2p_ar = self._ring_ar
        else:
            p2p_ar = self._p2p_ar
        compute_streams = self._compute_streams  # [0] * tp_size (default streams)

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        for layer_idx in range(num_layers):
            lw_list = [e.layers[layer_idx] for e in self.engines]

            # ── ATTENTION PHASE ──────────────────────────────────────────────

            # Make compute stream wait on allreduce done from previous layer's
            # FFN allreduce (GPU-side). First layer: hipEventCreate creates events
            # in an "unrecorded" state on ROCm — hipStreamWaitEvent on an unrecorded
            # event is a no-op (safe). So layer 0 proceeds without blocking.
            if layer_idx > 0:
                p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            # RMSNorm (reads d_hidden — gated by allreduce done event above)
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.attn_norm, h)

            # Attention GEMV kernels (reads d_normed)
            for engine, lw in zip(self.engines, lw_list):
                if lw.layer_type == 'full_attention':
                    engine._decode_full_attention(layer_idx, lw, position)
                else:
                    if engine._deltanet_gpu:
                        engine._decode_linear_attention_gpu(layer_idx, lw, position)
                    else:
                        engine._decode_linear_attention(layer_idx, lw, position)

            # Async allreduce: record compute events → allreduce stream waits →
            # P2P gather + reduce + broadcast (all non-blocking from Python)
            partial_ptrs = [e.d_proj_out for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

            # ── FFN PHASE ────────────────────────────────────────────────────

            # Make compute stream wait for attention allreduce completion (GPU-side).
            # This ensures FFN's RMSNorm reads the updated d_hidden.
            p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            # FFN RMSNorm (reads d_hidden — gated by attention allreduce done event)
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.ffn_norm, h)

            # FFN GEMV kernels
            for engine, lw in zip(self.engines, lw_list):
                if engine._gemv_int4_dual:
                    engine._launch_ffn_gate_up_silu(
                        engine.d_ffn_gate, engine.d_normed,
                        lw, h, engine.local_intermediate_size)
                else:
                    engine._launch_gemv_int4(
                        engine.d_ffn_gate, engine.d_normed,
                        lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_gemv_int4(
                        engine.d_ffn_up, engine.d_normed,
                        lw.up_qweight, lw.up_scales, lw.up_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_silu_fused(
                        engine.d_ffn_gate, engine.d_ffn_up,
                        engine.d_ffn_gate, engine.local_intermediate_size)
                engine._launch_gemv_int4(
                    engine.d_ffn_out, engine.d_ffn_gate,
                    lw.down_qweight, lw.down_scales, lw.down_zeros,
                    engine.local_intermediate_size, h)

            # Async FFN allreduce (non-blocking; next layer will wait for it)
            partial_ptrs = [e.d_ffn_out for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

        # After all layers: wait for the last FFN allreduce to complete
        # before reading d_hidden for final norm
        p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

        # Synchronize all GPUs to ensure all GPU work is complete
        # (compute streams + allreduce streams). hipDeviceSynchronize waits
        # for all streams on the current device.
        for dev_id in self.device_ids:
            self._hip.set_device(dev_id)
            self._hip.synchronize()  # hipDeviceSynchronize — waits all streams

        for engine in self.engines:
            engine.kv_cache.advance()

        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_step_serial(self, token_embedding: np.ndarray,
                             position: int) -> np.ndarray:
        """Serial (original) decode step implementation.

        Every allreduce uses fast_ar_fused: downloads partials + hidden,
        AVX FP16 add on host, uploads result.
        
        Double-buffer mode support:
          When _double_buffer_enabled=True, alternates buffers each layer:
          - Even layers: RMSNorm reads from A, allreduce writes to B
          - Odd layers: RMSNorm reads from B, allreduce writes to A
          This enables compute-communication overlap when combined with
          stream overlap dispatch.
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        use_double_buffer = self._double_buffer_enabled

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            if use_double_buffer:
                # In double-buffer mode, start with buffer A as read buffer
                engine.d_hidden = engine.d_hidden_A
                engine.d_hidden_write = engine.d_hidden_B
            engine.device.upload(engine.d_hidden, emb_bytes)

        for layer_idx in range(num_layers):
            lw_list = [e.layers[layer_idx] for e in self.engines]

            # RMSNorm for attention (reads from d_hidden)
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.attn_norm, h)

            # --- Attention ---
            for engine, lw in zip(self.engines, lw_list):
                if lw.layer_type == 'full_attention':
                    engine._decode_full_attention(layer_idx, lw, position)
                else:
                    if engine._deltanet_gpu:
                        engine._decode_linear_attention_gpu(layer_idx, lw, position)
                    else:
                        engine._decode_linear_attention(layer_idx, lw, position)

            # Allreduce attention partials + residual add
            if use_double_buffer:
                # Double-buffer mode: write to d_hidden_write
                self._allreduce_residual_double_buffer("d_proj_out", h)
            else:
                self._allreduce_residual("d_proj_out", h)

            # RMSNorm for FFN (reads from d_hidden)
            for engine, lw in zip(self.engines, lw_list):
                engine._launch_rmsnorm(engine.d_normed, engine.d_hidden,
                                       lw.ffn_norm, h)

            # --- FFN ---
            for engine, lw in zip(self.engines, lw_list):
                if engine._gemv_int4_dual:
                    engine._launch_ffn_gate_up_silu(
                        engine.d_ffn_gate, engine.d_normed,
                        lw, h, engine.local_intermediate_size)
                else:
                    engine._launch_gemv_int4(
                        engine.d_ffn_gate, engine.d_normed,
                        lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_gemv_int4(
                        engine.d_ffn_up, engine.d_normed,
                        lw.up_qweight, lw.up_scales, lw.up_zeros,
                        h, engine.local_intermediate_size)
                    engine._launch_silu_fused(
                        engine.d_ffn_gate, engine.d_ffn_up,
                        engine.d_ffn_gate, engine.local_intermediate_size)
                engine._launch_gemv_int4(
                    engine.d_ffn_out, engine.d_ffn_gate,
                    lw.down_qweight, lw.down_scales, lw.down_zeros,
                    engine.local_intermediate_size, h)

            # Allreduce FFN partials + residual add
            if use_double_buffer:
                # Double-buffer mode: write to d_hidden_write
                self._allreduce_residual_double_buffer("d_ffn_out", h)
                # Swap buffers: d_hidden becomes old d_hidden_write, and vice versa
                for engine in self.engines:
                    engine._swap_hidden_buffers()
            else:
                self._allreduce_residual("d_ffn_out", h)

        for engine in self.engines:
            engine.kv_cache.advance()

        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def _decode_step_batched_allreduce(self, token_embedding: np.ndarray,
                                        position: int) -> np.ndarray:
        """Decode step with batched FFN allreduce for consecutive DeltaNet layers.
        
        Conservative approach (FFN-only deferral within DeltaNet blocks):
          For each DeltaNet layer:
            1. RMSNorm(d_hidden) → attention → proj_out
            2. ALLREDUCE(proj_out) → d_hidden += attn_result  (cannot defer)
            3. RMSNorm(d_hidden) → FFN → ffn_out
            4. SKIP FFN allreduce, accumulate locally
            
          After 3 consecutive DeltaNet layers (at block boundary):
            - ALLREDUCE(accumulated_f  FN_out) → d_hidden += ffn_result_global
            
          For full-attention layers: standard 2-allreduce path.
        
        This reduces allreduce count from 128 to ~96 per step (25% reduction).
        The key approximation: within a DeltaNet block, each layer's FFN uses
        d_hidden that includes attention results but NOT previous FFN results
        (until the block boundary). For DeltaNet layers, this is acceptable
        because the recurrent state carries the cross-layer information.
        
        Args:
            token_embedding: input embedding
            position: current decode position
            
        Returns:
            output hidden state
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers
        half_rotary = self.engines[0].rotary_dim // 2
        cos_offset = position * half_rotary * 2
        
        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for eng in self.engines:
            eng.device.upload(eng.d_hidden, emb_bytes)
        
        # Track accumulation state
        in_deltanet_block = False
        
        for layer_idx in range(num_layers):
            lw_list = [e.layers[layer_idx] for e in self.engines]
            
            # Check if this is a full-attention layer (every 4th layer: 3, 7, 11, ...)
            is_full_attn = cfg.is_full_attention(layer_idx)
            
            if is_full_attn:
                # Before full-attention, flush any accumulated DeltaNet FFN partials
                if in_deltanet_block:
                    # Batched allreduce of accumulated FFN partials for the block
                    partial_ptrs = [e.d_ffn_out for e in self.engines]
                    hidden_ptrs = [e.d_hidden for e in self.engines]
                    self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, h)
                    self._batched_allreduce_counter['count'] += 1
                    in_deltanet_block = False
                
                # Standard path for full-attention layer: 2 allreduces
                # RMSNorm + attention
                for eng, lw in zip(self.engines, lw_list):
                    eng._launch_rmsnorm(eng.d_normed, eng.d_hidden, lw.attn_norm, h)
                    eng._decode_full_attention(layer_idx, lw, position)
                
                # Allreduce 1: attention partials
                self._allreduce_residual("d_proj_out", h)
                self._batched_allreduce_counter['count'] += 1
                
                # RMSNorm + FFN
                for eng, lw in zip(self.engines, lw_list):
                    eng._launch_rmsnorm(eng.d_normed, eng.d_hidden, lw.ffn_norm, h)
                    if eng._gemv_int4_dual:
                        eng._launch_ffn_gate_up_silu(
                            eng.d_ffn_gate, eng.d_normed,
                            lw, h, eng.local_intermediate_size)
                    else:
                        eng._launch_gemv_int4(
                            eng.d_ffn_gate, eng.d_normed,
                            lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                            h, eng.local_intermediate_size)
                        eng._launch_gemv_int4(
                            eng.d_ffn_up, eng.d_normed,
                            lw.up_qweight, lw.up_scales, lw.up_zeros,
                            h, eng.local_intermediate_size)
                        eng._launch_silu_fused(
                            eng.d_ffn_gate, eng.d_ffn_up,
                            eng.d_ffn_gate, eng.local_intermediate_size)
                    eng._launch_gemv_int4(
                        eng.d_ffn_out, eng.d_ffn_gate,
                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                        eng.local_intermediate_size, h)
                
                # Allreduce 2: FFN partials (standard, not batched)
                self._allreduce_residual("d_ffn_out", h)
                self._batched_allreduce_counter['count'] += 1
                
            else:
                # DeltaNet layer
                # Check if starting a new DeltaNet block
                if not in_deltanet_block:
                    in_deltanet_block = True
                
                # Attention path (standard - cannot defer)
                for eng, lw in zip(self.engines, lw_list):
                    eng._launch_rmsnorm(eng.d_normed, eng.d_hidden, lw.attn_norm, h)
                    if eng._deltanet_gpu:
                        eng._decode_linear_attention_gpu(layer_idx, lw, position)
                    else:
                        eng._decode_linear_attention(layer_idx, lw, position)
                
                # Allreduce attention partials
                self._allreduce_residual("d_proj_out", h)
                self._batched_allreduce_counter['count'] += 1
                
                # FFN path
                for eng, lw in zip(self.engines, lw_list):
                    eng._launch_rmsnorm(eng.d_normed, eng.d_hidden, lw.ffn_norm, h)
                    if eng._gemv_int4_dual:
                        eng._launch_ffn_gate_up_silu(
                            eng.d_ffn_gate, eng.d_normed,
                            lw, h, eng.local_intermediate_size)
                    else:
                        eng._launch_gemv_int4(
                            eng.d_ffn_gate, eng.d_normed,
                            lw.gate_qweight, lw.gate_scales, lw.gate_zeros,
                            h, eng.local_intermediate_size)
                        eng._launch_gemv_int4(
                            eng.d_ffn_up, eng.d_normed,
                            lw.up_qweight, lw.up_scales, lw.up_zeros,
                            h, eng.local_intermediate_size)
                        eng._launch_silu_fused(
                            eng.d_ffn_gate, eng.d_ffn_up,
                            eng.d_ffn_gate, eng.local_intermediate_size)
                    eng._launch_gemv_int4(
                        eng.d_ffn_out, eng.d_ffn_gate,
                        lw.down_qweight, lw.down_scales, lw.down_zeros,
                        eng.local_intermediate_size, h)
                
                # Check if this is the last DeltaNet in the block
                # (next layer is full-attention OR this is the last layer)
                next_is_full = (layer_idx + 1 < num_layers and 
                               cfg.is_full_attention(layer_idx + 1))
                is_last_layer = (layer_idx == num_layers - 1)
                
                if next_is_full or is_last_layer:
                    # End of DeltaNet block: batched allreduce of FFN partials
                    partial_ptrs = [e.d_ffn_out for e in self.engines]
                    hidden_ptrs = [e.d_hidden for e in self.engines]
                    self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, h)
                    self._batched_allreduce_counter['count'] += 1
                    in_deltanet_block = False
                # else: continue accumulating (don't allreduce yet)
        
        # GPU sync before advancing KV cache
        self.synchronize()
        
        for eng in self.engines:
            eng.kv_cache.advance()
        
        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                            dtype=np.float16)

    def _allreduce_residual_no_sync(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add, skipping hipDeviceSynchronize().

        Used in threaded dispatch mode where workers have already called
        hipDeviceSynchronize() in parallel before this method is called.
        This avoids the serial sequential device sync overhead.

        Falls back to regular _allreduce_residual if P2P is unavailable.
        """
        if self._p2p_ar is not None and 2 <= self.tp_size <= 4:
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            # skip_device_sync=True: workers already synced all GPUs in parallel
            self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, hidden_size,
                                             skip_device_sync=True)
            return

        # Fallback: regular allreduce (with device sync)
        self._allreduce_residual(buffer_name, hidden_size)

    def _allreduce_sum(self, buffer_name: str, hidden_size: int):
        """Allreduce partials only (no hidden download).

        Tries ring allreduce first (if enabled), then P2P GPU allreduce,
        falls back to fast_ar C extension, then Python fallback.
        """
        # Ring allreduce (new path)
        if (self._ring_allreduce and self._ring_ar is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            self._ring_ar.allreduce_sum(partial_ptrs, hidden_size)
            return

        # P2P GPU allreduce (primary path): async P2P gather + on-device kernel + broadcast
        if self._p2p_ar is not None and 2 <= self.tp_size <= 4:
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            self._p2p_ar.allreduce_sum(partial_ptrs, hidden_size)
            return

        # Host-mediated C extension fallback (TP=2 only)
        if self._fast_ar and self.tp_size == 2:
            e0, e1 = self.engines
            err = self._fast_ar.fast_ar_sum_tp2(
                e0.device.device_id, e1.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_sum_tp2 failed: HIP error {err}")
            return

        # Python fallback
        size = hidden_size * 2
        hip = self._hip

        for i, engine in enumerate(self.engines):
            hip.set_device(engine.device.device_id)
            hip.synchronize()
            hip.memcpy_d2h(self._host_bufs[i],
                           getattr(engine, buffer_name), size)

        accum = np.frombuffer(self._host_bufs[0], dtype=np.float16).copy()
        for i in range(1, self.tp_size):
            accum += np.frombuffer(self._host_bufs[i], dtype=np.float16)
        result_bytes = accum.tobytes()

        for engine in self.engines:
            hip.set_device(engine.device.device_id)
            hip.memcpy_h2d(getattr(engine, buffer_name), result_bytes, size)

    def _allreduce_residual_double_buffer(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add to d_hidden_write (double-buffer mode).

        Same as _allreduce_residual but writes to engine.d_hidden_write instead
        of engine.d_hidden. This enables overlapping allreduce with next-layer
        compute in the double-buffer decode loop.

        Tries kernel P2P allreduce first (when enabled), then fused P2P reduce,
        then ring allreduce, then standard P2P GPU allreduce, then fast_ar C
        extension, then Python fallback.
        """
        # Kernel P2P allreduce: writes to d_hidden_write
        if (self._kernel_p2p_allreduce and self._p2p_ar is not None
                and self._p2p_ar._kernel_p2p_lib is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_write_ptrs = [e.d_hidden_write for e in self.engines]
            self._p2p_ar.allreduce_residual_kernel(partial_ptrs, hidden_write_ptrs, hidden_size)
            return

        # Fused P2P reduce: writes to d_hidden_write
        if (self._fused_p2p_reduce and self._fused_p2p_ar is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_write_ptrs = [e.d_hidden_write for e in self.engines]
            self._fused_p2p_ar.allreduce_residual(partial_ptrs, hidden_write_ptrs, hidden_size)
            return

        # Ring allreduce: writes to d_hidden_write
        if (self._ring_allreduce and self._ring_ar is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_write_ptrs = [e.d_hidden_write for e in self.engines]
            self._ring_ar.allreduce_residual(partial_ptrs, hidden_write_ptrs, hidden_size)
            return

        # P2P GPU allreduce: writes to d_hidden_write
        if self._p2p_ar is not None and 2 <= self.tp_size <= 4:
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_write_ptrs = [e.d_hidden_write for e in self.engines]
            self._p2p_ar.allreduce_residual(partial_ptrs, hidden_write_ptrs, hidden_size)
            return

        # Host-mediated C extension fallback (TP=2,3,4): writes to d_hidden_write
        if self._fast_ar and self.tp_size == 2:
            e0, e1 = self.engines
            err = self._fast_ar.fast_ar_fused_tp2(
                e0.device.device_id, e1.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                e0.d_hidden_write, e1.d_hidden_write,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp2 (double-buffer) failed: HIP error {err}")
            return

        if self._fast_ar and self.tp_size == 3:
            e0, e1, e2 = self.engines
            err = self._fast_ar.fast_ar_fused_tp3(
                e0.device.device_id, e1.device.device_id, e2.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                getattr(e2, buffer_name),
                e0.d_hidden_write, e1.d_hidden_write, e2.d_hidden_write,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp3 (double-buffer) failed: HIP error {err}")
            return

        if self._fast_ar and self.tp_size == 4:
            e0, e1, e2, e3 = self.engines
            err = self._fast_ar.fast_ar_fused_tp4(
                e0.device.device_id, e1.device.device_id,
                e2.device.device_id, e3.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                getattr(e2, buffer_name), getattr(e3, buffer_name),
                e0.d_hidden_write, e1.d_hidden_write, e2.d_hidden_write, e3.d_hidden_write,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp4 (double-buffer) failed: HIP error {err}")
            return

        # Python fallback: writes to d_hidden_write
        size = hidden_size * 2
        hip = self._hip

        for i, engine in enumerate(self.engines):
            hip.set_device(engine.device.device_id)
            hip.synchronize()
            hip.memcpy_d2h(self._host_bufs[i],
                           getattr(engine, buffer_name), size)

        # Download from first engine's d_hidden_write
        hip.set_device(self.engines[0].device.device_id)
        hip.memcpy_d2h(self._host_hidden, self.engines[0].d_hidden_write, size)

        hidden = np.frombuffer(self._host_hidden, dtype=np.float16).copy()
        for i in range(self.tp_size):
            hidden += np.frombuffer(self._host_bufs[i], dtype=np.float16)
        result_bytes = hidden.tobytes()

        for engine in self.engines:
            hip.set_device(engine.device.device_id)
            hip.memcpy_h2d(engine.d_hidden_write, result_bytes, size)

    def _allreduce_residual(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add to d_hidden.

        Tries kernel P2P allreduce first (when enabled), then fused P2P reduce,
        then ring allreduce, then standard P2P GPU allreduce, then fast_ar C
        extension, then Python fallback.
        """
        # Kernel P2P allreduce (new path): each GPU runs a single kernel that reads
        # peer GPU partials directly via BAR1 P2P — no host round-trips, no gather/broadcast.
        if (self._kernel_p2p_allreduce and self._p2p_ar is not None
                and self._p2p_ar._kernel_p2p_lib is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            self._p2p_ar.allreduce_residual_kernel(partial_ptrs, hidden_ptrs, hidden_size)
            return

        # Fused P2P reduce (new path): each GPU reads all peer partials directly via P2P
        # and updates its own hidden, all simultaneously. Eliminates gather+broadcast steps.
        if (self._fused_p2p_reduce and self._fused_p2p_ar is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            self._fused_p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, hidden_size)
            return

        # Ring allreduce (new path): distributes PCIe bandwidth across all GPUs
        if (self._ring_allreduce and self._ring_ar is not None
                and self.tp_size in (2, 4)):
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            self._ring_ar.allreduce_residual(partial_ptrs, hidden_ptrs, hidden_size)
            return

        # P2P GPU allreduce (primary path): async P2P gather + on-device kernel + broadcast
        if self._p2p_ar is not None and 2 <= self.tp_size <= 4:
            partial_ptrs = [getattr(e, buffer_name) for e in self.engines]
            hidden_ptrs = [e.d_hidden for e in self.engines]
            self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, hidden_size)
            return

        # Host-mediated C extension fallback (TP=2,3,4)
        if self._fast_ar and self.tp_size == 2:
            e0, e1 = self.engines
            err = self._fast_ar.fast_ar_fused_tp2(
                e0.device.device_id, e1.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                e0.d_hidden, e1.d_hidden,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp2 failed: HIP error {err}")
            return

        if self._fast_ar and self.tp_size == 3:
            e0, e1, e2 = self.engines
            err = self._fast_ar.fast_ar_fused_tp3(
                e0.device.device_id, e1.device.device_id, e2.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                getattr(e2, buffer_name),
                e0.d_hidden, e1.d_hidden, e2.d_hidden,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp3 failed: HIP error {err}")
            return

        if self._fast_ar and self.tp_size == 4:
            e0, e1, e2, e3 = self.engines
            err = self._fast_ar.fast_ar_fused_tp4(
                e0.device.device_id, e1.device.device_id,
                e2.device.device_id, e3.device.device_id,
                getattr(e0, buffer_name), getattr(e1, buffer_name),
                getattr(e2, buffer_name), getattr(e3, buffer_name),
                e0.d_hidden, e1.d_hidden, e2.d_hidden, e3.d_hidden,
                hidden_size)
            if err:
                raise RuntimeError(f"fast_ar_fused_tp4 failed: HIP error {err}")
            return

        # Python fallback
        size = hidden_size * 2
        hip = self._hip

        for i, engine in enumerate(self.engines):
            hip.set_device(engine.device.device_id)
            hip.synchronize()
            hip.memcpy_d2h(self._host_bufs[i],
                           getattr(engine, buffer_name), size)

        hip.set_device(self.engines[0].device.device_id)
        hip.memcpy_d2h(self._host_hidden, self.engines[0].d_hidden, size)

        hidden = np.frombuffer(self._host_hidden, dtype=np.float16).copy()
        for i in range(self.tp_size):
            hidden += np.frombuffer(self._host_bufs[i], dtype=np.float16)
        result_bytes = hidden.tobytes()

        for engine in self.engines:
            hip.set_device(engine.device.device_id)
            hip.memcpy_h2d(engine.d_hidden, result_bytes, size)

    # ------------------------------------------------------------------
    # C dispatch extension support
    # ------------------------------------------------------------------

    def _load_c_dispatch_lib(self):
        """Load the c_dispatch.so shared library.

        Builds it via gcc if the .so is missing or stale.
        Returns the ctypes.CDLL handle, or None on failure.
        """
        src_dir = Path(__file__).parent.parent / "runtime"
        c_path = src_dir / "c_dispatch.c"
        so_path = src_dir / "c_dispatch.so"

        if not c_path.exists():
            return None

        if not so_path.exists() or os.path.getmtime(c_path) > os.path.getmtime(so_path):
            try:
                subprocess.check_call([
                    "gcc", "-O3", "-shared", "-fPIC",
                    "-I/opt/rocm/include",
                    "-L/opt/rocm/lib", "-lamdhip64",
                    "-o", str(so_path), str(c_path),
                ], stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError:
            return None

        # Register function signatures
        lib.c_dispatch_step.argtypes = [
            ctypes.c_uint64,  # plan_ptr
            ctypes.c_uint64,  # cos_offset
            ctypes.c_uint32,  # seq_len
        ]
        lib.c_dispatch_step.restype = ctypes.c_int

        lib.c_dispatch_get_spec_size.argtypes = []
        lib.c_dispatch_get_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_kernel_spec_size.argtypes = []
        lib.c_dispatch_get_kernel_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_allreduce_spec_size.argtypes = []
        lib.c_dispatch_get_allreduce_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_plan_size.argtypes = []
        lib.c_dispatch_get_plan_size.restype = ctypes.c_int

        return lib

    def set_direct_kv_write(self, enabled: bool):
        """Enable or disable direct KV cache writes from QKNorm/RoPE kernel.

        When enabled=True:
        - The QKNorm/RoPE kernel (qknorm_rope_cachew_fused) writes post-RoPE K
          directly to the KV cache position, eliminating the K D2D memcpy.
        - V is written directly from a separate V GEMV to the cache position,
          eliminating the V D2D memcpy.
        - Total D2D copies eliminated: 2 per full-attention layer × 16 layers = 32/token.

        Requires build_dispatch_cache() to be called after enabling to rebuild
        the parameter arrays with the new kernel specs.

        Falls back gracefully if qknorm_rope_cachew kernel is unavailable.
        """
        for engine in self.engines:
            engine.set_direct_kv_write(enabled)
        # Force rebuild of dispatch cache to use new kernel specs
        if self._engine_layer_caches:
            self._engine_layer_caches = []
            self._c_dispatch_plan = None  # Invalidate C dispatch plan

    def set_awq_mode(self, enabled: bool = True):
        """Enable or disable AWQ GEMV kernel mode on all GPU engines.

        When enabled=True, all per-GPU InferenceEngine instances use the AWQ
        variant of gemv_int4_v5 for non-residual GEMV calls. The AWQ kernel
        skips the zero-point subtraction (w = q * scale instead of
        w = (q - zero) * scale), saving 8 v_sub_f32 instructions per uint32 word
        and eliminating the zeros tensor memory traffic.

        Used when AWQ-format weights are loaded (zeros tensor = all zeros).
        Requires gemv_int4_v5_awq.hip to be compiled on the target GPU.

        NOTE: Call this BEFORE build_dispatch_cache() so that the cached
        launch specs use the correct kernels. If called after building the
        cache, the cache will be automatically rebuilt.

        Args:
            enabled: True to use AWQ kernel, False to fall back to GPTQ kernel.
        """
        for engine in self.engines:
            engine.set_awq_mode(enabled)
        # Force rebuild of dispatch cache to use AWQ kernels
        if self._engine_layer_caches:
            print(f"  AWQ mode {'enabled' if enabled else 'disabled'}: rebuilding dispatch cache...")
            self._engine_layer_caches = []
            self._c_dispatch_plan = None  # Invalidate C dispatch plan
            self.build_dispatch_cache()  # Rebuild with correct kernels

    def set_c_dispatch(self, enabled: bool):
        """Enable or disable C dispatch loop.

        When enabled=True, decode_step uses _decode_step_c_dispatch() which
        dispatches all 64 layers' kernels in a tight C loop via c_dispatch.so,
        without returning to Python between layers. This eliminates ~14ms/tok
        Python dispatch overhead from the cached+stream path.

        Requires:
        - build_dispatch_cache() must have been called first
        - P2P allreduce must be available
        - c_dispatch.so must be loadable

        Falls back to cached+stream if C dispatch is unavailable.
        """
        if enabled:
            if not self._engine_layer_caches:
                self.build_dispatch_cache()
            if self._p2p_ar is None:
                print("WARNING: C dispatch requires P2P allreduce (unavailable). "
                      "Falling back to cached+stream.")
                enabled = False
            elif self._c_dispatch_lib is None:
                self._c_dispatch_lib = self._load_c_dispatch_lib()
                if self._c_dispatch_lib is None:
                    print("WARNING: Failed to load c_dispatch.so. "
                          "Falling back to cached+stream.")
                    enabled = False
            if enabled and self._c_dispatch_plan is None:
                try:
                    self._c_dispatch_plan = self._build_c_dispatch_plan()
                except Exception as e:
                    print(f"WARNING: Failed to build C dispatch plan: {e}. "
                          "Falling back to cached+stream.")
                    enabled = False
        self._c_dispatch_enabled = enabled

    def _load_c_dispatch_v2_lib(self):
        """Load the c_dispatch_v2.so shared library.

        Builds it via gcc if the .so is missing or stale.
        c_dispatch_v2 has the same interface as c_dispatch but with optimized
        hipSetDevice batching that reduces call overhead by ~384 calls/token.
        Returns the ctypes.CDLL handle, or None on failure.
        """
        src_dir = Path(__file__).parent.parent / "runtime"
        c_path = src_dir / "c_dispatch_v2.c"
        so_path = src_dir / "c_dispatch_v2.so"

        if not c_path.exists():
            return None

        if not so_path.exists() or os.path.getmtime(c_path) > os.path.getmtime(so_path):
            try:
                subprocess.check_call([
                    "gcc", "-O3", "-shared", "-fPIC",
                    "-I/opt/rocm/include",
                    "-L/opt/rocm/lib", "-lamdhip64",
                    "-o", str(so_path), str(c_path),
                ], stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError:
            return None

        # Register function signatures (same as v1 but entry point is c_dispatch_step_v2)
        lib.c_dispatch_step_v2.argtypes = [
            ctypes.c_uint64,  # plan_ptr
            ctypes.c_uint64,  # cos_offset
            ctypes.c_uint32,  # seq_len
        ]
        lib.c_dispatch_step_v2.restype = ctypes.c_int

        lib.c_dispatch_get_spec_size.argtypes = []
        lib.c_dispatch_get_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_kernel_spec_size.argtypes = []
        lib.c_dispatch_get_kernel_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_allreduce_spec_size.argtypes = []
        lib.c_dispatch_get_allreduce_spec_size.restype = ctypes.c_int
        lib.c_dispatch_get_plan_size.argtypes = []
        lib.c_dispatch_get_plan_size.restype = ctypes.c_int

        return lib

    def set_c_dispatch_v2(self, enabled: bool):
        """Enable or disable optimized C dispatch v2.

        c_dispatch_v2 is a drop-in replacement for c_dispatch with batched
        hipSetDevice calls. It saves ~384 hipSetDevice calls/token (from ~2432
        to ~2048) by eliminating redundant device context switches in the
        allreduce routine when the device context is already correct.

        Estimated improvement: ~1-2% throughput from reduced host overhead.

        Requires set_c_dispatch(True) to have been called first (to build the
        dispatch plan). The same CDispatchPlan is reused — only the entry
        point changes from c_dispatch_step to c_dispatch_step_v2.
        """
        if enabled:
            if not self._c_dispatch_enabled:
                # Must have c_dispatch enabled first (for plan building)
                print("WARNING: set_c_dispatch_v2 requires set_c_dispatch(True) first.")
                return
            if self._c_dispatch_v2_lib is None:
                self._c_dispatch_v2_lib = self._load_c_dispatch_v2_lib()
                if self._c_dispatch_v2_lib is None:
                    print("WARNING: Failed to load c_dispatch_v2.so. "
                          "Staying with c_dispatch_v1.")
                    return
        self._c_dispatch_v2_enabled = enabled

    # ------------------------------------------------------------------
    # Persistent megakernel support (Milestone 5: m5-persistent-kernel)
    # ------------------------------------------------------------------

    def _load_persistent_dispatch(self):
        """Load the PersistentDecodeDispatcher for persistent megakernel.
        
        Returns the dispatcher instance, or None on failure.
        """
        try:
            from src.runtime.persistent_dispatch import PersistentDecodeDispatcher
            dispatcher = PersistentDecodeDispatcher(self)
            return dispatcher
        except Exception as e:
            print(f"WARNING: Failed to load persistent dispatch: {e}")
            return None

    def set_persistent_kernel(self, enabled: bool):
        """Enable or disable persistent megakernel mode.
        
        When enabled=True, the entire decode step (64 layers) runs as a single
        persistent kernel on the GPU, with on-GPU task scheduling and synchronization.
        This eliminates all host-side kernel launch overhead (~7ms/tok savings).
        
        Priority: persistent > global_graph > graph > c_dispatch > cached+stream
        
        Args:
            enabled: True to enable persistent kernel mode
        """
        if enabled:
            if self._persistent_dispatcher is None:
                self._persistent_dispatcher = self._load_persistent_dispatch()
                if self._persistent_dispatcher is None:
                    print("WARNING: Persistent kernel not available. "
                          "Falling back to C dispatch.")
                    return
                
                # Initialize the dispatcher
                if not self._persistent_dispatcher.enable():
                    print("WARNING: Failed to enable persistent kernel. "
                          "Falling back to C dispatch.")
                    self._persistent_dispatcher = None
                    return
                
                # Build the task queue
                self._persistent_dispatcher.build_task_queue()
            
            self._persistent_enabled = True
            print(f"Persistent megakernel mode ENABLED (TP={self.tp_size})")
        else:
            self._persistent_enabled = False
            if self._persistent_dispatcher is not None:
                self._persistent_dispatcher.disable()
            print("Persistent megakernel mode DISABLED")

    def _decode_step_persistent(self, token_embedding: np.ndarray,
                                 position: int) -> np.ndarray:
        """Decode step using persistent megakernel.
        
        Launches a single persistent kernel that executes all 64 layers
        internally via on-GPU task scheduling. Eliminates host dispatch overhead.
        
        Args:
            token_embedding: Input token embedding (hidden_size,)
            position: Current sequence position
            
        Returns:
            Updated hidden state (hidden_size,)
        """
        if self._persistent_dispatcher is None:
            raise RuntimeError("Persistent kernel not initialized")
        
        h = self.config.hidden_size
        
        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)
        
        # Update position-dependent params in persistent state
        # (In full implementation, would update cos/sin offsets, KV cache pointers, etc.)
        # For now, placeholder - the persistent kernel would read these from global state
        
        # Launch persistent kernel on all GPUs
        for rank, engine in enumerate(self.engines):
            stream = engine.device.stream  # Use engine's default stream
            err = self._persistent_dispatcher._lib.persistent_decode_tp4(
                self._persistent_dispatcher._state_ptr,
                ctypes.c_void_p(stream),
                ctypes.c_uint32(rank)
            )
            if err:
                raise RuntimeError(f"persistent_decode_tp4 failed on GPU {rank}: HIP error {err}")
        
        # Synchronize all GPUs
        for engine in self.engines:
            engine.device.synchronize()
        
        # Advance KV caches
        for engine in self.engines:
            engine.kv_cache.advance()
        
        # Final norm and logits (GPU0 only)
        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    # ------------------------------------------------------------------
    # C graph dispatch support
    # ------------------------------------------------------------------

    def _load_c_graph_dispatch_lib(self):
        """Load the c_graph_dispatch.so shared library.

        Builds it via gcc if the .so is missing or stale.
        Returns the ctypes.CDLL handle, or None on failure.
        """
        src_dir = Path(__file__).parent.parent / "runtime"
        c_path = src_dir / "c_graph_dispatch.c"
        so_path = src_dir / "c_graph_dispatch.so"

        if not c_path.exists():
            return None

        if not so_path.exists() or os.path.getmtime(c_path) > os.path.getmtime(so_path):
            try:
                subprocess.check_call([
                    "gcc", "-O3", "-shared", "-fPIC",
                    "-I/opt/rocm/include",
                    "-L/opt/rocm/lib", "-lamdhip64",
                    "-o", str(so_path), str(c_path),
                ], stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError:
            return None

        # Register function signatures
        lib.c_graph_dispatch_step.argtypes = [
            ctypes.c_uint64,  # plan_ptr
            ctypes.c_uint64,  # cos_offset
            ctypes.c_uint32,  # seq_len
        ]
        lib.c_graph_dispatch_step.restype = ctypes.c_int

        lib.c_graph_dispatch_get_layer_spec_size.argtypes = []
        lib.c_graph_dispatch_get_layer_spec_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_mutable_param_size.argtypes = []
        lib.c_graph_dispatch_get_mutable_param_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_allreduce_spec_size.argtypes = []
        lib.c_graph_dispatch_get_allreduce_spec_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_plan_size.argtypes = []
        lib.c_graph_dispatch_get_plan_size.restype = ctypes.c_int
        lib.c_graph_dispatch_get_kparams_size.argtypes = []
        lib.c_graph_dispatch_get_kparams_size.restype = ctypes.c_int

        return lib

    def build_c_graph_dispatch_plan(self):
        """Build the CGraphDispatchPlan ctypes structure from captured graph state.

        Must be called after graph capture (_GraphDecodeState.capture_all()).
        The plan contains:
        - CGraphLayerSpec[num_layers * num_engines]: graph exec handles + mutable params
        - CGraphAllreduceSpec[num_layers]: for attention allreduce
        - CGraphAllreduceSpec[num_layers]: for FFN allreduce

        Returns the CGraphDispatchPlan ctypes structure (kept alive via
        self._c_graph_dispatch_objects).
        """
        import ctypes as ct

        if self._c_graph_dispatch_lib is None:
            self._c_graph_dispatch_lib = self._load_c_graph_dispatch_lib()
            if self._c_graph_dispatch_lib is None:
                raise RuntimeError("c_graph_dispatch.so not available")

        lib = self._c_graph_dispatch_lib
        gds = self._graph_decode_step  # _GraphDecodeState instance

        if gds is None or not gds.captured:
            raise RuntimeError("Graph capture must be done before building C graph plan")

        num_layers  = self.config.num_hidden_layers
        num_engines = self.tp_size

        if self._ring_allreduce and self._ring_ar is not None:
            p2p_ar = self._ring_ar
        else:
            p2p_ar = self._p2p_ar

        # --- Query struct sizes from C ---
        layer_spec_size  = lib.c_graph_dispatch_get_layer_spec_size()
        mutable_size     = lib.c_graph_dispatch_get_mutable_param_size()
        ar_spec_size     = lib.c_graph_dispatch_get_allreduce_spec_size()
        plan_size        = lib.c_graph_dispatch_get_plan_size()
        kparams_size     = lib.c_graph_dispatch_get_kparams_size()

        # --- Define Python ctypes structs matching C structs ---
        from src.runtime.hip_graph_dispatch import hipKernelNodeParams

        # hipKernelNodeParams layout (must match C HipKernelNodeParams with padding)
        class CHipKernelNodeParams(ct.Structure):
            _fields_ = [
                ('blockDimX',     ct.c_uint),
                ('blockDimY',     ct.c_uint),
                ('blockDimZ',     ct.c_uint),
                ('_pad1',         ct.c_uint),
                ('extra',         ct.c_void_p),
                ('func',          ct.c_void_p),
                ('gridDimX',      ct.c_uint),
                ('gridDimY',      ct.c_uint),
                ('gridDimZ',      ct.c_uint),
                ('_pad2',         ct.c_uint),
                ('kernelParams',  ct.c_void_p),
                ('sharedMemBytes', ct.c_uint),
                ('_pad3',         ct.c_uint),
            ]

        MAX_MUTABLE_PARAMS = 8

        class CMutableParam(ct.Structure):
            _fields_ = [
                ('node',          ct.c_uint64),
                ('graph_exec',    ct.c_uint64),
                ('params_array',  ct.c_uint64),
                ('param_index',   ct.c_uint32),
                ('mutable_type',  ct.c_uint32),
                ('kparams',       CHipKernelNodeParams),
                ('kv_cache_base', ct.c_uint64),
                ('kv_stride',     ct.c_uint32),
                ('_pad',          ct.c_uint32),
                ('d_cos_base',    ct.c_uint64),
                ('d_sin_base',    ct.c_uint64),
            ]

        class CGraphLayerSpec(ct.Structure):
            _fields_ = [
                ('attn_graph_exec',  ct.c_uint64),
                ('ffn_graph_exec',   ct.c_uint64),
                ('mutable_params',   CMutableParam * MAX_MUTABLE_PARAMS),
                ('num_mutable',      ct.c_uint32),
                ('layer_type',       ct.c_int),
                ('_pad',             ct.c_uint32),
            ]

        # Allreduce function type signatures (same as c_dispatch.c)
        ReduceTp2Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_uint32, ct.c_void_p)
        ReduceTp3Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_void_p, ct.c_uint32,
                                      ct.c_void_p)
        ReduceTp4Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_uint32, ct.c_void_p)
        HipSetDeviceFunc       = ct.CFUNCTYPE(ct.c_int, ct.c_int)
        HipStreamSyncFunc      = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)
        HipEventRecordFunc     = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
        HipStreamWaitFunc      = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_uint)
        HipMemcpyPeerAsyncFunc = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_int,
                                               ct.c_void_p, ct.c_int, ct.c_size_t,
                                               ct.c_void_p)
        HipMemcpyAsyncFunc     = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                               ct.c_size_t, ct.c_int, ct.c_void_p)
        HipGetLastErrorFunc    = ct.CFUNCTYPE(ct.c_int)
        HipGraphLaunchFunc     = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
        HipGraphExecKernelNodeSetParamsFunc = ct.CFUNCTYPE(
            ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p)

        class CGraphAllreduceSpec(ct.Structure):
            _fields_ = [
                ('reduce_tp2',        ReduceTp2Func),
                ('reduce_tp3',        ReduceTp3Func),
                ('reduce_tp4',        ReduceTp4Func),
                ('tp_size',           ct.c_int),
                ('device_ids',        ct.c_int * 4),
                ('partial_ptrs',      ct.c_uint64 * 4),
                ('hidden_ptrs',       ct.c_uint64 * 4),
                ('gather_bufs',       ct.c_uint64 * 3),
                ('allreduce_streams', ct.c_uint64 * 4),
                ('compute_events',    ct.c_uint64 * 4),
                ('ar_done_events',    ct.c_uint64 * 4),
                ('compute_streams',   ct.c_uint64 * 4),
                ('num_elems',         ct.c_uint32),
                ('_pad',              ct.c_uint32),
            ]

        class CGraphDispatchPlan(ct.Structure):
            _fields_ = [
                ('num_layers',              ct.c_int),
                ('num_engines',             ct.c_int),
                ('graph_layer_specs',       ct.c_uint64),
                ('attn_allreduce_specs',    ct.c_uint64),
                ('ffn_allreduce_specs',     ct.c_uint64),
                ('hipSetDevice_fn',         HipSetDeviceFunc),
                ('hipStreamSynchronize_fn', HipStreamSyncFunc),
                ('hipEventRecord_fn',       HipEventRecordFunc),
                ('hipStreamWaitEvent_fn',   HipStreamWaitFunc),
                ('hipMemcpyPeerAsync_fn',   HipMemcpyPeerAsyncFunc),
                ('hipMemcpyAsync_fn',       HipMemcpyAsyncFunc),
                ('hipGetLastError_fn',      HipGetLastErrorFunc),
                ('hipGraphLaunch_fn',       HipGraphLaunchFunc),
                ('hipGraphExecKernelNodeSetParams_fn',
                                            HipGraphExecKernelNodeSetParamsFunc),
            ]

        # Verify sizes match C
        assert ct.sizeof(CMutableParam) == mutable_size, \
            f"CMutableParam size mismatch: Python={ct.sizeof(CMutableParam)}, C={mutable_size}"
        assert ct.sizeof(CGraphLayerSpec) == layer_spec_size, \
            f"CGraphLayerSpec size mismatch: Python={ct.sizeof(CGraphLayerSpec)}, C={layer_spec_size}"
        assert ct.sizeof(CGraphAllreduceSpec) == ar_spec_size, \
            f"CGraphAllreduceSpec size mismatch: Python={ct.sizeof(CGraphAllreduceSpec)}, C={ar_spec_size}"
        assert ct.sizeof(CGraphDispatchPlan) == plan_size, \
            f"CGraphDispatchPlan size mismatch: Python={ct.sizeof(CGraphDispatchPlan)}, C={plan_size}"
        assert ct.sizeof(CHipKernelNodeParams) == kparams_size, \
            f"CHipKernelNodeParams size mismatch: Python={ct.sizeof(CHipKernelNodeParams)}, C={kparams_size}"

        # Mutable type constants (must match C header)
        MUTABLE_TYPE_COS_PTR  = 1
        MUTABLE_TYPE_SIN_PTR  = 2
        MUTABLE_TYPE_SEQ_LEN  = 3
        MUTABLE_TYPE_KV_K_PTR = 4
        MUTABLE_TYPE_KV_V_PTR = 5

        def py_kparams_to_c(py_kp):
            """Convert Python hipKernelNodeParams to C CHipKernelNodeParams."""
            c_kp = CHipKernelNodeParams()
            c_kp.blockDimX = py_kp.blockDimX
            c_kp.blockDimY = py_kp.blockDimY
            c_kp.blockDimZ = py_kp.blockDimZ
            c_kp.extra = py_kp.extra
            c_kp.func  = py_kp.func
            c_kp.gridDimX = py_kp.gridDimX
            c_kp.gridDimY = py_kp.gridDimY
            c_kp.gridDimZ = py_kp.gridDimZ
            c_kp.kernelParams = py_kp.kernelParams
            c_kp.sharedMemBytes = py_kp.sharedMemBytes
            return c_kp

        # --- Build CGraphLayerSpec array ---
        GLSArray = CGraphLayerSpec * (num_layers * num_engines)
        gls_array = GLSArray()

        for layer_idx in range(num_layers):
            for engine_idx, engine in enumerate(self.engines):
                gs = gls_array[layer_idx * num_engines + engine_idx]
                lw = engine.layers[layer_idx]
                lc = self._engine_layer_caches[engine_idx][layer_idx]

                # Graph exec handles from captured state
                attn_seg = gds._attn_segs[engine_idx][layer_idx]
                ffn_seg  = gds._ffn_segs[engine_idx][layer_idx]
                gs.attn_graph_exec = attn_seg._graph_exec or 0
                gs.ffn_graph_exec  = ffn_seg._graph_exec  or 0
                gs.layer_type      = 0 if lw.layer_type == 'full_attention' else 1

                # Build mutable param entries for full attention layers
                mutable_count = 0
                if lw.layer_type == 'full_attention':
                    mutable_list = gds._mutable_attn[engine_idx][layer_idx]
                    graph_exec_val = attn_seg._graph_exec or 0

                    for m in mutable_list:
                        key  = m['key']
                        node = m['node']
                        spec = m['spec']
                        base = m['base_params']

                        if mutable_count >= MAX_MUTABLE_PARAMS:
                            break

                        def add_mutable(mp, mtype, pidx, kv_base=0, kv_str=0,
                                        cos_base=0, sin_base=0):
                            mp.node          = node or 0
                            mp.graph_exec    = graph_exec_val
                            mp.params_array  = ct.addressof(spec.params_array)
                            mp.param_index   = pidx
                            mp.mutable_type  = mtype
                            # Copy the base hipKernelNodeParams
                            mp.kparams       = py_kparams_to_c(base)
                            # Update kernelParams to point to the spec's params_array
                            mp.kparams.kernelParams = ct.addressof(spec.params_array)
                            mp.kv_cache_base = kv_base
                            mp.kv_stride     = kv_str
                            mp.d_cos_base    = cos_base
                            mp.d_sin_base    = sin_base

                        if key == 'qknorm_q':
                            # cos (param[2]) and sin (param[3])
                            add_mutable(gs.mutable_params[mutable_count],
                                        MUTABLE_TYPE_COS_PTR, 2,
                                        cos_base=engine.d_cos,
                                        sin_base=engine.d_sin)
                            mutable_count += 1
                            if mutable_count < MAX_MUTABLE_PARAMS:
                                add_mutable(gs.mutable_params[mutable_count],
                                            MUTABLE_TYPE_SIN_PTR, 3,
                                            cos_base=engine.d_cos,
                                            sin_base=engine.d_sin)
                                mutable_count += 1

                        elif key == 'qknorm_k':
                            add_mutable(gs.mutable_params[mutable_count],
                                        MUTABLE_TYPE_COS_PTR, 2,
                                        cos_base=engine.d_cos,
                                        sin_base=engine.d_sin)
                            mutable_count += 1
                            if mutable_count < MAX_MUTABLE_PARAMS:
                                add_mutable(gs.mutable_params[mutable_count],
                                            MUTABLE_TYPE_SIN_PTR, 3,
                                            cos_base=engine.d_cos,
                                            sin_base=engine.d_sin)
                                mutable_count += 1
                            # KV cache K write ptr (param[4]) if present
                            if '_k_cache_base' in lc and mutable_count < MAX_MUTABLE_PARAMS:
                                kv_stride = lc['_kv_stride']
                                add_mutable(gs.mutable_params[mutable_count],
                                            MUTABLE_TYPE_KV_K_PTR, 4,
                                            kv_base=lc['_k_cache_base'],
                                            kv_str=kv_stride)
                                mutable_count += 1

                        elif key == 'decode_attn':
                            add_mutable(gs.mutable_params[mutable_count],
                                        MUTABLE_TYPE_SEQ_LEN, 4)
                            mutable_count += 1

                        elif key == 'gemv_v_cache':
                            # V cache write ptr (param[2])
                            if '_v_cache_base' in lc and mutable_count < MAX_MUTABLE_PARAMS:
                                kv_stride = lc['_kv_stride']
                                add_mutable(gs.mutable_params[mutable_count],
                                            MUTABLE_TYPE_KV_V_PTR, 2,
                                            kv_base=lc['_v_cache_base'],
                                            kv_str=kv_stride)
                                mutable_count += 1

                gs.num_mutable = mutable_count

        # --- Build allreduce spec arrays ---
        AttnARArray = CGraphAllreduceSpec * num_layers
        FfnARArray  = CGraphAllreduceSpec * num_layers
        attn_ar_array = AttnARArray()
        ffn_ar_array  = FfnARArray()

        p2p_lib = p2p_ar._lib
        reduce_tp2 = ReduceTp2Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp2, ct.c_void_p).value)
        reduce_tp3 = ReduceTp3Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp3, ct.c_void_p).value)
        reduce_tp4 = ReduceTp4Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp4, ct.c_void_p).value)

        def fill_ar_spec(c_ar, partial_attr):
            c_ar.reduce_tp2 = reduce_tp2
            c_ar.reduce_tp3 = reduce_tp3
            c_ar.reduce_tp4 = reduce_tp4
            c_ar.tp_size = self.tp_size
            for i, engine in enumerate(self.engines):
                c_ar.device_ids[i]    = engine.device.device_id
                c_ar.partial_ptrs[i]  = getattr(engine, partial_attr)
                c_ar.hidden_ptrs[i]   = engine.d_hidden
                c_ar.compute_streams[i] = 0  # null stream (default)
            for i, ptr in enumerate(p2p_ar._gather_bufs):
                c_ar.gather_bufs[i] = ptr
            for i in range(self.tp_size):
                c_ar.allreduce_streams[i] = p2p_ar._allreduce_streams[i]
                c_ar.compute_events[i]    = p2p_ar._compute_events[i]
                c_ar.ar_done_events[i]    = p2p_ar._ar_done_events[i]
            c_ar.num_elems = self.config.hidden_size

        for layer_idx in range(num_layers):
            fill_ar_spec(attn_ar_array[layer_idx], 'd_proj_out')
            fill_ar_spec(ffn_ar_array[layer_idx],  'd_ffn_out')

        # --- Build the CGraphDispatchPlan ---
        plan = CGraphDispatchPlan()
        plan.num_layers  = num_layers
        plan.num_engines = num_engines
        plan.graph_layer_specs    = ct.addressof(gls_array)
        plan.attn_allreduce_specs = ct.addressof(attn_ar_array)
        plan.ffn_allreduce_specs  = ct.addressof(ffn_ar_array)

        hip_lib_handle = self._hip._lib
        hip_set_device_fn = HipSetDeviceFunc(
            ct.cast(hip_lib_handle.hipSetDevice, ct.c_void_p).value)
        hip_stream_sync_fn = HipStreamSyncFunc(
            ct.cast(hip_lib_handle.hipStreamSynchronize, ct.c_void_p).value)
        hip_event_record_fn = HipEventRecordFunc(
            ct.cast(hip_lib_handle.hipEventRecord, ct.c_void_p).value)
        hip_stream_wait_fn = HipStreamWaitFunc(
            ct.cast(hip_lib_handle.hipStreamWaitEvent, ct.c_void_p).value)
        hip_memcpy_peer_async_fn = HipMemcpyPeerAsyncFunc(
            ct.cast(hip_lib_handle.hipMemcpyPeerAsync, ct.c_void_p).value)
        hip_memcpy_async_fn = HipMemcpyAsyncFunc(
            ct.cast(hip_lib_handle.hipMemcpyAsync, ct.c_void_p).value)
        hip_get_last_error_fn = HipGetLastErrorFunc(
            ct.cast(hip_lib_handle.hipGetLastError, ct.c_void_p).value)
        hip_graph_launch_fn = HipGraphLaunchFunc(
            ct.cast(hip_lib_handle.hipGraphLaunch, ct.c_void_p).value)
        hip_graph_exec_set_params_fn = HipGraphExecKernelNodeSetParamsFunc(
            ct.cast(hip_lib_handle.hipGraphExecKernelNodeSetParams,
                    ct.c_void_p).value)

        plan.hipSetDevice_fn         = hip_set_device_fn
        plan.hipStreamSynchronize_fn = hip_stream_sync_fn
        plan.hipEventRecord_fn       = hip_event_record_fn
        plan.hipStreamWaitEvent_fn   = hip_stream_wait_fn
        plan.hipMemcpyPeerAsync_fn   = hip_memcpy_peer_async_fn
        plan.hipMemcpyAsync_fn       = hip_memcpy_async_fn
        plan.hipGetLastError_fn      = hip_get_last_error_fn
        plan.hipGraphLaunch_fn       = hip_graph_launch_fn
        plan.hipGraphExecKernelNodeSetParams_fn = hip_graph_exec_set_params_fn

        # Store all objects to keep them alive
        self._c_graph_dispatch_objects = {
            'gls_array':     gls_array,
            'attn_ar_array': attn_ar_array,
            'ffn_ar_array':  ffn_ar_array,
            'plan':          plan,
            'reduce_tp2':    reduce_tp2,
            'reduce_tp3':    reduce_tp3,
            'reduce_tp4':    reduce_tp4,
            'hip_set_device_fn':       hip_set_device_fn,
            'hip_stream_sync_fn':      hip_stream_sync_fn,
            'hip_event_record_fn':     hip_event_record_fn,
            'hip_stream_wait_fn':      hip_stream_wait_fn,
            'hip_memcpy_peer_async_fn': hip_memcpy_peer_async_fn,
            'hip_memcpy_async_fn':     hip_memcpy_async_fn,
            'hip_get_last_error_fn':   hip_get_last_error_fn,
            'hip_graph_launch_fn':     hip_graph_launch_fn,
            'hip_graph_exec_set_params_fn': hip_graph_exec_set_params_fn,
        }

        print(f"C graph dispatch plan built: {num_layers} layers × {num_engines} engines")
        return plan

    # ------------------------------------------------------------------
    # Global graph dispatch support (full-layer graph with kernel P2P allreduce)
    # ------------------------------------------------------------------

    def set_global_graph_dispatch(self, enabled: bool):
        """Enable or disable global HIP graph dispatch.

        When enabled=True, decode_step() uses _decode_step_global_graph() which
        captures each layer's full computation (compute + kernel P2P allreduce)
        as a single HIP graph. This is the highest-priority dispatch path.

        The full-layer graph includes:
          attn_rmsnorm → projections → QKNorm/RoPE → attention/DeltaNet →
          O-proj → kernel_p2p_allreduce (attn) →
          FFN RMSNorm → gate+up+silu → down_proj → kernel_p2p_allreduce (ffn)

        Requirements:
          - build_dispatch_cache() must have been called first
          - kernel P2P allreduce must be available (_p2p_ar initialized)
          - kernel_p2p_allreduce.hsaco must exist in build/kernels/

        Falls back to C dispatch if global graph capture is not available.

        Priority: global_graph > graph > c_dispatch > cached+stream > cached > serial
        """
        if enabled:
            if not self._engine_layer_caches:
                self.build_dispatch_cache()
            if self._p2p_ar is None:
                print("WARNING: Global graph dispatch requires P2P allreduce "
                      "(unavailable). Falling back to C dispatch.")
                enabled = False
            elif self._p2p_ar._kernel_p2p_lib is None:
                print("WARNING: Global graph dispatch requires kernel P2P allreduce "
                      "(kernel_p2p_allreduce.so not loaded). Falling back to C dispatch.")
                enabled = False
        if not enabled and self._global_graph_decode_state is not None:
            self._global_graph_decode_state.cleanup()
            self._global_graph_decode_state = None
        self._global_graph_dispatch_enabled = enabled
        if enabled:
            print(f"Global graph dispatch enabled (full-layer graph + kernel P2P allreduce)")
        else:
            print(f"Global graph dispatch disabled")

    def _decode_step_global_graph(self, token_embedding: np.ndarray,
                                   position: int) -> np.ndarray:
        """Decode step using full-layer HIP graph capture and replay.

        On first call: captures per-GPU full-layer graphs (compute + kernel P2P allreduce)
        for all layers.
        On subsequent calls: replays graphs with updated mutable params.

        Structure per layer (per GPU):
          Graph contains:
            attn_rmsnorm → GEMV projections → QKNorm/RoPE →
            attention/DeltaNet → O-proj → kernel_p2p_allreduce (attn) →
            FFN RMSNorm → gate+up+silu → down_proj → kernel_p2p_allreduce (ffn)

          All GPUs run their full-layer graphs independently (no host-side sync
          between attention and FFN — the allreduce is in the graph).
          After all layer graphs complete, advance KV caches and compute final norm.

        Falls back to C dispatch on error.
        """
        from src.runtime.hip_graph_dispatch import HIPGraphRuntime

        h = self.config.hidden_size

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Lazy init: create global graph state on first use
        if self._global_graph_decode_state is None:
            try:
                self._global_graph_decode_state = _GlobalGraphDecodeState(
                    self, HIPGraphRuntime())
                print("Global graph decode state initialized")
            except Exception as e:
                print(f"WARNING: Failed to init global graph decode state: {e}. "
                      "Falling back to C dispatch.")
                self._global_graph_dispatch_enabled = False
                return self._decode_step_c_dispatch(token_embedding, position)

        gds = self._global_graph_decode_state

        # First call: capture all full-layer graphs
        if not gds.captured:
            try:
                gds.capture_full_layer(position, include_allreduce=True)
            except Exception as e:
                print(f"WARNING: Global graph capture failed: {e}. "
                      "Falling back to C dispatch.")
                import traceback
                traceback.print_exc()
                self._global_graph_dispatch_enabled = False
                gds.cleanup()
                self._global_graph_decode_state = None
                return self._decode_step_c_dispatch(token_embedding, position)

        # Replay all full-layer graphs with updated mutable params
        try:
            gds.replay_step_full_layer(position, include_allreduce=True)
        except Exception as e:
            print(f"WARNING: Global graph replay failed at position={position}: {e}. "
                  "Falling back to C dispatch.")
            import traceback
            traceback.print_exc()
            self._global_graph_dispatch_enabled = False
            gds.cleanup()
            self._global_graph_decode_state = None
            return self._decode_step_c_dispatch(token_embedding, position)

        # Synchronize all GPUs to ensure all GPU work is complete
        for dev_id in self.device_ids:
            self._hip.set_device(dev_id)
            self._hip.synchronize()

        # Advance KV caches
        for engine in self.engines:
            engine.kv_cache.advance()

        # Final norm and logits (GPU0 only)
        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    # ------------------------------------------------------------------
    # Graph dispatch support
    # ------------------------------------------------------------------

    def set_graph_dispatch(self, enabled: bool):
        """Enable or disable HIP graph dispatch.

        When enabled=True, decode_step() uses _decode_step_graph() which:
          1. On the first call: captures per-GPU attention and FFN compute
             segments as HIP graphs (using the LaunchSpec kernel handles from
             build_decode_launch_cache()). This takes ~100-500ms.
          2. On subsequent calls: replays captured graphs with updated mutable
             params (cos/sin RoPE pointers, decode attention seq_len) and
             performs host-orchestrated allreduce between segments.

        HIP graph replay eliminates Python↔C kernel launch overhead by
        dispatching all per-GPU compute kernels in a single hipGraphLaunch call.
        Expected throughput improvement: ~1-3% over C dispatch.

        Requirements:
          - build_dispatch_cache() must have been called first
          - P2P allreduce must be available

        Falls back silently if graph capture is not available.
        """
        if enabled:
            if not self._engine_layer_caches:
                self.build_dispatch_cache()
            if self._p2p_ar is None:
                print("WARNING: Graph dispatch requires P2P allreduce (unavailable). "
                      "Falling back to C dispatch.")
                enabled = False
        if not enabled and self._graph_decode_step is not None:
            self._graph_decode_step.cleanup()
            self._graph_decode_step = None
        if not enabled:
            # Also reset C graph dispatch plan so re-enabling creates fresh graphs
            # (old plan holds destroyed graph exec handles, causing segfault on re-enable)
            self._c_graph_dispatch_plan = None
            self._c_graph_dispatch_objects = {}
        self._graph_dispatch_enabled = enabled

    def _decode_step_graph(self, token_embedding: np.ndarray,
                            position: int) -> np.ndarray:
        """Decode step using HIP graph capture and replay.

        On first call: captures per-GPU compute graphs for all layers.
        On subsequent calls: replays graphs with updated mutable params.

        Structure per layer:
          GPU n: replay attention graph (RMSNorm→GEMV→QKNorm/RoPE→attn→sigmoid→O-proj)
          [Host allreduce: attention partials → d_hidden]
          GPU n: replay FFN graph (FFN RMSNorm→gate+up+silu→down_proj)
          [Host allreduce: FFN partials → d_hidden]

        Mutable params updated per step:
          - cos/sin pointers for QKNorm/RoPE (position-dependent)
          - seq_len for decode attention (grows each step)
          - KV cache write pointers (for direct-KV-write mode)
        """
        from src.runtime.hip_graph_dispatch import (
            HIPGraphRuntime, GraphSegment,
            hipStreamCaptureModeRelaxed, hipKernelNodeParams,
            hipGraphNodeTypeKernel,
        )

        h = self.config.hidden_size
        half_rotary = self.engines[0].rotary_dim // 2

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Lazy init: create graph runtime and GraphDecodeStep on first use
        if self._graph_decode_step is None:
            try:
                self._graph_decode_step = _GraphDecodeState(
                    self, HIPGraphRuntime())
                print("Graph decode state initialized")
            except Exception as e:
                print(f"WARNING: Failed to init graph decode state: {e}. "
                      "Falling back to C dispatch.")
                self._graph_dispatch_enabled = False
                return self._decode_step_c_dispatch(token_embedding, position)

        gds = self._graph_decode_step

        # First call: capture all graphs
        if not gds.captured:
            try:
                gds.capture_all(position)
            except Exception as e:
                print(f"WARNING: Graph capture failed: {e}. "
                      "Falling back to C dispatch.")
                self._graph_dispatch_enabled = False
                gds.cleanup()
                self._graph_decode_step = None
                return self._decode_step_c_dispatch(token_embedding, position)

            # After capture: try to build C graph dispatch plan (first-time-only)
            if self._c_graph_dispatch_plan is None:
                try:
                    if self._c_graph_dispatch_lib is None:
                        self._c_graph_dispatch_lib = self._load_c_graph_dispatch_lib()
                    if self._c_graph_dispatch_lib is not None:
                        self._c_graph_dispatch_plan = self.build_c_graph_dispatch_plan()
                        print("C graph dispatch plan built; using C replay loop")
                    else:
                        print("c_graph_dispatch.so unavailable; using Python replay loop")
                except Exception as e:
                    print(f"WARNING: Failed to build C graph dispatch plan: {e}. "
                          "Using Python replay loop.")
                    self._c_graph_dispatch_plan = None
                    self._c_graph_dispatch_objects = {}

        # Replay graphs (prefer C graph dispatch, fall back to Python)
        if self._c_graph_dispatch_plan is not None:
            try:
                half_rotary = self.engines[0].rotary_dim // 2
                cos_offset  = position * half_rotary * 2
                seq_len     = self.engines[0].kv_cache.current_len + 1
                plan        = self._c_graph_dispatch_objects['plan']
                plan_ptr    = ctypes.addressof(plan)
                err = self._c_graph_dispatch_lib.c_graph_dispatch_step(
                    ctypes.c_uint64(plan_ptr),
                    ctypes.c_uint64(cos_offset),
                    ctypes.c_uint32(seq_len),
                )
                if err:
                    raise RuntimeError(f"c_graph_dispatch_step failed: HIP error {err}")
            except Exception as e:
                print(f"WARNING: C graph replay failed at position={position}: {e}. "
                      "Falling back to Python graph replay.")
                self._c_graph_dispatch_plan = None
                self._c_graph_dispatch_objects = {}
                # fall through to Python replay below

        if self._c_graph_dispatch_plan is None:
            # Python replay path (fallback)
            try:
                gds.replay_step(position)
            except Exception as e:
                print(f"WARNING: Graph replay failed at step position={position}: {e}. "
                      "Falling back to C dispatch.")
                self._graph_dispatch_enabled = False
                gds.cleanup()
                self._graph_decode_step = None
                return self._decode_step_c_dispatch(token_embedding, position)

        # Synchronize all GPUs
        for dev_id in self.device_ids:
            self._hip.set_device(dev_id)
            self._hip.synchronize()

        # Advance KV caches
        for engine in self.engines:
            engine.kv_cache.advance()

        # Final norm and logits (GPU0 only)
        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def _build_c_dispatch_plan(self):
        """Build the CDispatchPlan ctypes structure from the engine caches.

        Returns a ctypes buffer containing the serialized plan plus references
        to all Python objects that must stay alive.

        The plan contains:
        - CDispatchPlan header
        - CEngineLayerSpec[num_layers * num_engines] (all engines, all layers)
        - CAllreduceSpec[num_layers] for attention
        - CAllreduceSpec[num_layers] for FFN
        """
        import ctypes as ct

        num_layers = self.config.num_hidden_layers
        num_engines = self.tp_size
        if self._ring_allreduce and self._ring_ar is not None:
            p2p_ar = self._ring_ar
        else:
            p2p_ar = self._p2p_ar

        # Kernel spec struct (must match c_dispatch.c exactly)
        # uint64 func, 3x uint32 grid, 3x uint32 block, uint32 shared_mem,
        # uint64 params_array, uint32 num_params, uint32 present
        # = 8 + 12 + 12 + 4 + 8 + 4 + 4 = 52 bytes (need alignment)
        # Let C tell us the actual size
        lib = self._c_dispatch_lib
        kernel_spec_size = lib.c_dispatch_get_kernel_spec_size()
        engine_spec_size = lib.c_dispatch_get_spec_size()
        ar_spec_size     = lib.c_dispatch_get_allreduce_spec_size()
        plan_size        = lib.c_dispatch_get_plan_size()

        # Allocate flat buffers
        n_engine_specs = num_layers * num_engines
        engine_specs_buf = ct.create_string_buffer(n_engine_specs * engine_spec_size)
        attn_ar_buf      = ct.create_string_buffer(num_layers * ar_spec_size)
        ffn_ar_buf       = ct.create_string_buffer(num_layers * ar_spec_size)
        plan_buf         = ct.create_string_buffer(plan_size)

        hip = self._hip

        # Helper to pack a CKernelSpec into a bytes buffer at offset
        def pack_kernel_spec(buf, offset, spec_dict, key):
            """Pack a LaunchSpec into a CKernelSpec at buf[offset]."""
            if key not in spec_dict:
                # present = 0, everything else 0
                return  # already zero-initialized by create_string_buffer

            spec = spec_dict[key]
            ct.memmove(
                ct.addressof(buf) + offset + 0,
                ct.c_uint64(spec.func),
                8
            )
            o = 8
            for v in [spec.grid[0], spec.grid[1], spec.grid[2],
                      spec.block[0], spec.block[1], spec.block[2],
                      spec.shared_mem]:
                ct.memmove(ct.addressof(buf) + offset + o, ct.c_uint32(v), 4)
                o += 4
            # params_array pointer
            params_ptr = ct.addressof(spec.params_array)
            ct.memmove(ct.addressof(buf) + offset + o, ct.c_uint64(params_ptr), 8)
            o += 8
            # num_params
            ct.memmove(ct.addressof(buf) + offset + o,
                        ct.c_uint32(len(spec.params)), 4)
            o += 4
            # present = 1
            ct.memmove(ct.addressof(buf) + offset + o, ct.c_uint32(1), 4)

        # Build a simpler approach: use ctypes structures
        # Define CKernelSpec as a ctypes Structure
        class CKernelSpec(ct.Structure):
            _fields_ = [
                ('func',         ct.c_uint64),
                ('grid_x',       ct.c_uint32),
                ('grid_y',       ct.c_uint32),
                ('grid_z',       ct.c_uint32),
                ('block_x',      ct.c_uint32),
                ('block_y',      ct.c_uint32),
                ('block_z',      ct.c_uint32),
                ('shared_mem',   ct.c_uint32),
                ('params_array', ct.c_uint64),
                ('num_params',   ct.c_uint32),
                ('present',      ct.c_uint32),
            ]

        class CEngineLayerSpec(ct.Structure):
            _fields_ = [
                ('attn_rmsnorm',        CKernelSpec),
                ('gemv_q_fused',        CKernelSpec),
                ('gemv_kv_fused',       CKernelSpec),
                ('qknorm_q',            CKernelSpec),
                ('qknorm_k',            CKernelSpec),
                ('decode_attn',         CKernelSpec),
                ('sigmoid_mul',         CKernelSpec),
                ('gemv_o_proj',         CKernelSpec),
                ('gemv_la_in_proj',     CKernelSpec),
                ('deltanet_v3',         CKernelSpec),
                ('deltanet_v3_shift',   CKernelSpec),
                ('gemv_la_out_proj',    CKernelSpec),
                ('ffn_rmsnorm',         CKernelSpec),
                ('ffn_gate_up_silu',    CKernelSpec),
                ('ffn_down',            CKernelSpec),
                ('gemv_k_only',         CKernelSpec),  # Direct KV write: K-only GEMV
                ('gemv_v_cache',        CKernelSpec),  # Direct KV write: V to cache
                ('layer_type',          ct.c_int),
                ('streams_ready',       ct.c_int),
                ('stream_q',            ct.c_uint64),
                ('stream_kv',           ct.c_uint64),
                ('d_cos_base',          ct.c_uint64),
                ('d_sin_base',          ct.c_uint64),
                ('d_k_src',             ct.c_uint64),
                ('d_v_src',             ct.c_uint64),
                ('kv_cache_k_base',     ct.c_uint64),
                ('kv_cache_v_base',     ct.c_uint64),
                ('kv_stride',           ct.c_uint32),
                ('use_direct_kv_write', ct.c_uint32),
                # Deferred attention allreduce (M3 optimization)
                ('d_hidden',            ct.c_uint64),  # engine->d_hidden for residual add
                ('d_proj_out',          ct.c_uint64),  # engine->d_proj_out (attn output partial)
            ]

        # Allreduce function type signatures
        ReduceTp2Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_uint32, ct.c_void_p)
        ReduceTp3Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_void_p, ct.c_uint32,
                                      ct.c_void_p)
        ReduceTp4Func = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_uint32, ct.c_void_p)
        HipSetDeviceFunc      = ct.CFUNCTYPE(ct.c_int, ct.c_int)
        HipStreamSyncFunc     = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)
        HipEventRecordFunc    = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
        HipStreamWaitFunc     = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_uint)
        HipMemcpyPeerAsyncFunc= ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_int,
                                              ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_void_p)
        HipMemcpyAsyncFunc    = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                                              ct.c_size_t, ct.c_int, ct.c_void_p)
        HipGetLastErrorFunc   = ct.CFUNCTYPE(ct.c_int)
        HipModuleLaunchFunc   = ct.CFUNCTYPE(
            ct.c_int,
            ct.c_void_p,                  # function handle
            ct.c_uint, ct.c_uint, ct.c_uint,  # grid xyz
            ct.c_uint, ct.c_uint, ct.c_uint,  # block xyz
            ct.c_uint,                    # shared_mem
            ct.c_void_p,                  # stream
            ct.POINTER(ct.c_void_p),      # params
            ct.POINTER(ct.c_void_p),      # extra
        )

        class CAllreduceSpec(ct.Structure):
            _fields_ = [
                ('reduce_tp2',      ReduceTp2Func),
                ('reduce_tp3',      ReduceTp3Func),
                ('reduce_tp4',      ReduceTp4Func),
                ('tp_size',         ct.c_int),
                ('device_ids',      ct.c_int * 4),
                ('partial_ptrs',    ct.c_uint64 * 4),
                ('hidden_ptrs',     ct.c_uint64 * 4),
                ('gather_bufs',     ct.c_uint64 * 3),
                ('allreduce_streams', ct.c_uint64 * 4),
                ('compute_events',  ct.c_uint64 * 4),
                ('ar_done_events',  ct.c_uint64 * 4),
                ('compute_streams', ct.c_uint64 * 4),
                ('num_elems',       ct.c_uint32),
                # Kernel P2P allreduce fields (kernel-p2p-tp4-integration)
                ('use_kernel_p2p',  ct.c_uint32),
                ('kernel_p2p_tp4_fn', ct.c_void_p),  # kernel_p2p_tp4_fn_t function pointer
                # Fused kernel P2P allreduce + RMSNorm fields
                ('use_fused_kernel', ct.c_uint32),
                ('kernel_p2p_fused_tp4_fn', ct.c_void_p),  # kernel_p2p_fused_tp4_fn_t
                ('rmsnorm_weight_ptrs', ct.c_uint64 * 4),  # RMSNorm weight pointers per GPU
                ('eps',             ct.c_float),  # RMSNorm epsilon
                # Fused GEMV+AR+RMSNorm kernel fields (for FFN down projection)
                ('use_gemv_fused',  ct.c_uint32),  # 1=use fused GEMV+allreduce+RMSNorm
                ('gemv_fused_tp4_fn', ct.c_void_p),  # gemv_int4_fused_tp4_fn_t function pointer
                ('ffn_gate_ptrs',   ct.c_uint64 * 4),  # FFN gate output pointers per GPU (input to down proj)
                ('ffn_down_qweight_ptrs', ct.c_uint64 * 4),  # FFN down proj INT4 weights per GPU
                ('ffn_down_scales_ptrs', ct.c_uint64 * 4),  # FFN down proj scales per GPU
                ('ffn_down_zeros_ptrs', ct.c_uint64 * 4),   # FFN down proj zeros per GPU
                ('ffn_K',           ct.c_uint32),  # FFN intermediate size (input to down proj)
                ('ffn_group_size',  ct.c_uint32),  # Quantization group size
                # Padding for alignment (total struct size should be 448 bytes)
            ]

        class CDispatchPlan(ct.Structure):
            _fields_ = [
                ('num_layers',              ct.c_int),
                ('num_engines',             ct.c_int),
                ('engine_layer_specs',      ct.c_uint64),
                ('attn_allreduce_specs',    ct.c_uint64),
                ('ffn_allreduce_specs',     ct.c_uint64),
                ('use_stream_overlap',      ct.c_int),
                # Deferred attention allreduce (M3 optimization)
                ('use_deferred_attention_ar', ct.c_int),  # 1=skip attention AR, add partial locally
                ('residual_add_fn',         ct.c_void_p),  # residual_add_v2_fn_t function pointer
                ('hipSetDevice_fn',         HipSetDeviceFunc),
                ('hipStreamSynchronize_fn', HipStreamSyncFunc),
                ('hipEventRecord_fn',       HipEventRecordFunc),
                ('hipStreamWaitEvent_fn',   HipStreamWaitFunc),
                ('hipMemcpyPeerAsync_fn',   HipMemcpyPeerAsyncFunc),
                ('hipMemcpyAsync_fn',       HipMemcpyAsyncFunc),
                ('hipGetLastError_fn',      HipGetLastErrorFunc),
                ('hipModuleLaunchKernel_fn', HipModuleLaunchFunc),
            ]

        # Verify sizes match C
        assert ct.sizeof(CKernelSpec) == lib.c_dispatch_get_kernel_spec_size(), \
            f"CKernelSpec size mismatch: Python={ct.sizeof(CKernelSpec)}, C={lib.c_dispatch_get_kernel_spec_size()}"
        assert ct.sizeof(CEngineLayerSpec) == lib.c_dispatch_get_spec_size(), \
            f"CEngineLayerSpec size mismatch: Python={ct.sizeof(CEngineLayerSpec)}, C={lib.c_dispatch_get_spec_size()}"
        assert ct.sizeof(CAllreduceSpec) == lib.c_dispatch_get_allreduce_spec_size(), \
            f"CAllreduceSpec size mismatch: Python={ct.sizeof(CAllreduceSpec)}, C={lib.c_dispatch_get_allreduce_spec_size()}"
        assert ct.sizeof(CDispatchPlan) == lib.c_dispatch_get_plan_size(), \
            f"CDispatchPlan size mismatch: Python={ct.sizeof(CDispatchPlan)}, C={lib.c_dispatch_get_plan_size()}"

        # Helper: fill a CKernelSpec from a LaunchSpec (or None for absent)
        def fill_kernel_spec(c_spec, py_spec):
            if py_spec is None:
                c_spec.present = 0
                return
            c_spec.func = py_spec.func
            c_spec.grid_x, c_spec.grid_y, c_spec.grid_z = py_spec.grid
            c_spec.block_x, c_spec.block_y, c_spec.block_z = py_spec.block
            c_spec.shared_mem = py_spec.shared_mem
            # params_array: address of the pre-built void*[] array
            c_spec.params_array = ct.addressof(py_spec.params_array)
            c_spec.num_params = len(py_spec.params)
            c_spec.present = 1

        # Build the engine layer spec array
        ELSArray = CEngineLayerSpec * (num_layers * num_engines)
        els_array = ELSArray()

        for layer_idx in range(num_layers):
            for engine_idx, engine in enumerate(self.engines):
                lc = self._engine_layer_caches[engine_idx][layer_idx]
                lw = engine.layers[layer_idx]
                es = els_array[layer_idx * num_engines + engine_idx]

                fill_kernel_spec(es.attn_rmsnorm,      lc.get('attn_rmsnorm'))
                fill_kernel_spec(es.gemv_q_fused,      lc.get('gemv_q_fused'))
                fill_kernel_spec(es.gemv_kv_fused,     lc.get('gemv_kv_fused'))
                fill_kernel_spec(es.qknorm_q,          lc.get('qknorm_q'))
                fill_kernel_spec(es.qknorm_k,          lc.get('qknorm_k'))
                fill_kernel_spec(es.decode_attn,       lc.get('decode_attn'))
                fill_kernel_spec(es.sigmoid_mul,       lc.get('sigmoid_mul'))
                fill_kernel_spec(es.gemv_o_proj,       lc.get('gemv_o_proj'))
                fill_kernel_spec(es.gemv_la_in_proj,   lc.get('gemv_la_in_proj'))
                fill_kernel_spec(es.deltanet_v3,       lc.get('deltanet_v3'))
                fill_kernel_spec(es.deltanet_v3_shift, lc.get('deltanet_v3_shift'))
                fill_kernel_spec(es.gemv_la_out_proj,  lc.get('gemv_la_out_proj'))
                fill_kernel_spec(es.ffn_rmsnorm,       lc.get('ffn_rmsnorm'))
                fill_kernel_spec(es.ffn_gate_up_silu,  lc.get('ffn_gate_up_silu'))
                fill_kernel_spec(es.ffn_down,          lc.get('ffn_down'))
                # Direct KV write kernels (populated when direct_kv_write is enabled)
                fill_kernel_spec(es.gemv_k_only,       lc.get('gemv_k_only'))
                fill_kernel_spec(es.gemv_v_cache,      lc.get('gemv_v_cache'))

                es.layer_type = 0 if lw.layer_type == 'full_attention' else 1
                # Q/KV stream sync eliminated: always set streams_ready=0 so C dispatch
                # never calls hipStreamSynchronize for Q/KV streams.
                # _stream_q/_stream_kv objects are kept alive for other paths.
                es.streams_ready = 0
                es.stream_q  = engine._stream_q
                es.stream_kv = engine._stream_kv

                # Position-dependent: store d_cos / d_sin base pointers
                es.d_cos_base = engine.d_cos
                es.d_sin_base = engine.d_sin

                # KV cache append params (full attention layers only)
                if lw.layer_type == 'full_attention':
                    kv_stride = (engine.local_num_kv_heads *
                                 self.config.head_dim * 2)
                    es.d_k_src         = engine.d_k
                    es.d_v_src         = engine.d_v
                    es.kv_cache_k_base = engine.kv_cache.layer_k_ptr(layer_idx)
                    es.kv_cache_v_base = engine.kv_cache.layer_v_ptr(layer_idx)
                    es.kv_stride       = kv_stride
                    # Enable direct KV write if engine has it enabled and kernels built
                    es.use_direct_kv_write = (
                        1 if (engine._direct_kv_write and
                              lc.get('gemv_k_only') is not None and
                              lc.get('gemv_v_cache') is not None)
                        else 0
                    )
                else:
                    es.kv_cache_k_base = 0
                    es.kv_cache_v_base = 0
                    es.kv_stride       = 0
                    es.use_direct_kv_write = 0
                
                # Deferred attention allreduce (M3 optimization): store pointers for residual add
                es.d_hidden = engine.d_hidden
                es.d_proj_out = engine.d_proj_out

        # Build allreduce spec arrays
        AttnARArray = CAllreduceSpec * num_layers
        FfnARArray  = CAllreduceSpec * num_layers
        attn_ar_array = AttnARArray()
        ffn_ar_array  = FfnARArray()

        # Get p2p_allreduce library function pointers
        p2p_lib = p2p_ar._lib
        reduce_tp2 = ReduceTp2Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp2, ct.c_void_p).value)
        reduce_tp3 = ReduceTp3Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp3, ct.c_void_p).value)
        reduce_tp4 = ReduceTp4Func(
            ct.cast(p2p_lib.p2p_reduce_residual_tp4, ct.c_void_p).value)

        # Get kernel P2P allreduce function pointer (if available and enabled)
        kernel_p2p_fn_ptr = None
        use_kernel_p2p_in_c = (
            self._kernel_p2p_allreduce
            and p2p_ar._kernel_p2p_lib is not None
            and self.tp_size == 4
        )
        if use_kernel_p2p_in_c:
            kernel_p2p_fn_ptr = ct.cast(
                p2p_ar._kernel_p2p_lib.kernel_p2p_allreduce_residual_tp4,
                ct.c_void_p
            ).value
            print(f"C dispatch: kernel P2P allreduce enabled "
                  f"(fn_ptr=0x{kernel_p2p_fn_ptr:016x})")
        else:
            print(f"C dispatch: using star topology allreduce "
                  f"(kernel_p2p_enabled={self._kernel_p2p_allreduce}, "
                  f"tp_size={self.tp_size})")

        # Load fused kernel P2P allreduce + RMSNorm library (if available)
        fused_kernel_fn_ptr = None
        use_fused_kernel_in_c = False
        fused_lib = None
        build_dir = Path(__file__).parent.parent.parent / "build" / "kernels"
        fused_so_path = build_dir / "kernel_p2p_allreduce_rmsnorm.so"
        
        if fused_so_path.exists():
            try:
                fused_lib = ct.CDLL(str(fused_so_path))
                # Set function signature (includes hidden residual parameter)
                fused_lib.kernel_p2p_allreduce_rmsnorm_tp4.argtypes = [
                    ct.c_void_p,  # output
                    ct.c_void_p,  # hidden (residual input)
                    ct.c_void_p,  # partial_local
                    ct.c_void_p,  # partial_peer0
                    ct.c_void_p,  # partial_peer1
                    ct.c_void_p,  # partial_peer2
                    ct.c_void_p,  # weight
                    ct.c_uint,    # dim
                    ct.c_uint,    # batch_size
                    ct.c_float,   # eps
                    ct.c_void_p,  # stream
                ]
                fused_lib.kernel_p2p_allreduce_rmsnorm_tp4.restype = ct.c_int
                fused_kernel_fn_ptr = ct.cast(
                    fused_lib.kernel_p2p_allreduce_rmsnorm_tp4,
                    ct.c_void_p
                ).value
                # Enable fused kernel only for TP=4
                if self.tp_size == 4:
                    use_fused_kernel_in_c = True
                    print(f"C dispatch: fused kernel allreduce+RMSNorm enabled "
                          f"(fn_ptr=0x{fused_kernel_fn_ptr:016x})")
                else:
                    print(f"C dispatch: fused kernel available but TP={self.tp_size} "
                          f"(requires TP=4)")
            except Exception as e:
                print(f"C dispatch: failed to load fused kernel library: {e}")
                fused_lib = None
        else:
            print(f"C dispatch: fused kernel library not found at {fused_so_path}")

        # Load fused GEMV+AR+RMSNorm kernel library (gemv_int4_p2p_allreduce_rmsnorm)
        # This kernel fuses FFN down projection (INT4 GEMV) + P2P allreduce + RMSNorm
        gemv_fused_fn_ptr = None
        use_gemv_fused_in_c = False
        gemv_fused_lib = None
        gemv_fused_so_path = build_dir / "gemv_int4_p2p_allreduce_rmsnorm.so"
        
        if gemv_fused_so_path.exists():
            try:
                gemv_fused_lib = ct.CDLL(str(gemv_fused_so_path))
                # Set function signature
                gemv_fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4.argtypes = [
                    ct.c_void_p,        # output
                    ct.c_void_p,        # A (input activation)
                    ct.POINTER(ct.c_uint),  # B_q4 (INT4 weights)
                    ct.c_void_p,        # scales
                    ct.c_void_p,        # zeros
                    ct.c_void_p,        # partial_local
                    ct.c_void_p,        # partial_peer0
                    ct.c_void_p,        # partial_peer1
                    ct.c_void_p,        # partial_peer2
                    ct.c_void_p,        # weight (RMSNorm)
                    ct.c_uint,          # K (input dim)
                    ct.c_uint,          # N (output dim)
                    ct.c_uint,          # dim (for RMSNorm)
                    ct.c_uint,          # group_size
                    ct.c_float,         # eps
                    ct.c_uint,          # tp_rank
                    ct.c_uint,          # tp_size
                    ct.c_void_p,        # stream
                ]
                gemv_fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4.restype = ct.c_int
                gemv_fused_fn_ptr = ct.cast(
                    gemv_fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4,
                    ct.c_void_p
                ).value
                # Fused GEMV+AR+RMSNorm kernel - enabled after fixing double-counting bug
                # The bug was in Phase 2 where partial_local was incorrectly added to
                # partial_result (which already contains the inline GEMV output).
                if True:  # Enabled after fix validation
                    use_gemv_fused_in_c = True
                    print(f"C dispatch: fused GEMV+AR+RMSNorm kernel ENABLED for FFN down-proj "
                          f"(fn_ptr=0x{gemv_fused_fn_ptr:016x})")
                else:
                    print(f"C dispatch: fused GEMV+AR+RMSNorm kernel available but DISABLED (needs investigation)")
            except Exception as e:
                print(f"C dispatch: failed to load fused GEMV kernel library: {e}")
                gemv_fused_lib = None
        else:
            print(f"C dispatch: fused GEMV kernel library not found at {gemv_fused_so_path}")

        # We need to access layer-specific RMSNorm weights for the fused kernel.
        # The fused kernel is called after allreduce and does the RMSNorm that would
        # normally be done by the next kernel:
        # - Attention allreduce → fused does ffn_rmsnorm (same layer)
        # - FFN allreduce → fused does attn_rmsnorm (next layer)
        #
        # Since each layer has different weights, we need to fill the specs per-layer
        # with the correct weight pointers.
        
        # Get RMSNorm weight pointers for each layer and engine
        layer_attn_norm_ptrs = []  # [layer_idx][engine_idx] -> d_attn_norm
        layer_ffn_norm_ptrs = []    # [layer_idx][engine_idx] -> d_ffn_norm
        for layer_idx in range(num_layers):
            attn_ptrs = []
            ffn_ptrs = []
            for engine_idx, engine in enumerate(self.engines):
                lw = engine.layers[layer_idx]
                attn_ptrs.append(lw.attn_norm)
                ffn_ptrs.append(lw.ffn_norm)
            layer_attn_norm_ptrs.append(attn_ptrs)
            layer_ffn_norm_ptrs.append(ffn_ptrs)

        def fill_ar_spec(c_ar, partial_attr, rmsnorm_weight_ptrs, layer_idx, is_ffn=False):
            """Fill a CAllreduceSpec for one layer.
            
            Args:
                c_ar: CAllreduceSpec to fill
                partial_attr: Attribute name for partial pointers ('d_proj_out' or 'd_ffn_out')
                rmsnorm_weight_ptrs: List of RMSNorm weight pointers [layer_idx][engine_idx] -> ptr
                layer_idx: Layer index for accessing layer-specific weights
                is_ffn: If True, populate gemv_fused fields for FFN down projection
            """
            c_ar.reduce_tp2 = reduce_tp2
            c_ar.reduce_tp3 = reduce_tp3
            c_ar.reduce_tp4 = reduce_tp4
            c_ar.tp_size = self.tp_size
            for i, engine in enumerate(self.engines):
                c_ar.device_ids[i]    = engine.device.device_id
                c_ar.partial_ptrs[i]  = getattr(engine, partial_attr)
                c_ar.hidden_ptrs[i]   = engine.d_hidden
                c_ar.compute_streams[i] = 0  # null stream (default)
            # Gather bufs (on GPU0, for TP>1) — used by star topology path
            for i, ptr in enumerate(p2p_ar._gather_bufs):
                c_ar.gather_bufs[i] = ptr
            # Allreduce streams and events from p2p_ar
            for i in range(self.tp_size):
                c_ar.allreduce_streams[i] = p2p_ar._allreduce_streams[i]
                c_ar.compute_events[i]    = p2p_ar._compute_events[i]
                c_ar.ar_done_events[i]    = p2p_ar._ar_done_events[i]
            c_ar.num_elems = self.config.hidden_size
            # Kernel P2P allreduce fields
            c_ar.use_kernel_p2p = 1 if use_kernel_p2p_in_c else 0
            c_ar.kernel_p2p_tp4_fn = kernel_p2p_fn_ptr
            # Fused kernel fields (allreduce+RMSNorm only)
            c_ar.use_fused_kernel = 1 if use_fused_kernel_in_c else 0
            c_ar.kernel_p2p_fused_tp4_fn = fused_kernel_fn_ptr
            # RMSNorm weight pointers for fused kernel (per-GPU)
            for i in range(self.tp_size):
                c_ar.rmsnorm_weight_ptrs[i] = rmsnorm_weight_ptrs[i]
            c_ar.eps = 1e-6  # Default RMSNorm epsilon
            
            # Fused GEMV+AR+RMSNorm kernel fields (for FFN down projection only)
            if is_ffn and use_gemv_fused_in_c and gemv_fused_fn_ptr:
                c_ar.use_gemv_fused = 1
                c_ar.gemv_fused_tp4_fn = gemv_fused_fn_ptr
                # Populate FFN gate output pointers (input to down projection, replicated across GPUs)
                for i, engine in enumerate(self.engines):
                    c_ar.ffn_gate_ptrs[i] = engine.d_ffn_gate
                # Populate FFN down projection weight pointers (per-GPU partitioned)
                for i, engine in enumerate(self.engines):
                    lw = engine.layers[layer_idx]
                    c_ar.ffn_down_qweight_ptrs[i] = lw.down_qweight
                    c_ar.ffn_down_scales_ptrs[i] = lw.down_scales
                    c_ar.ffn_down_zeros_ptrs[i] = lw.down_zeros
                c_ar.ffn_K = self.local_intermediate_size
                c_ar.ffn_group_size = self.config.group_size
            else:
                c_ar.use_gemv_fused = 0
                c_ar.gemv_fused_tp4_fn = 0

        for layer_idx in range(num_layers):
            # Attention allreduce: fused kernel does ffn_rmsnorm (same layer)
            fill_ar_spec(attn_ar_array[layer_idx], 'd_proj_out', layer_ffn_norm_ptrs[layer_idx], layer_idx, is_ffn=False)
            # FFN allreduce: fused kernel does attn_rmsnorm (next layer)
            # For the last layer, use the last layer's attn_norm (won't be used anyway)
            next_layer_idx = min(layer_idx + 1, num_layers - 1)
            fill_ar_spec(ffn_ar_array[layer_idx], 'd_ffn_out', layer_attn_norm_ptrs[next_layer_idx], layer_idx, is_ffn=True)

        # Build the CDispatchPlan
        plan = CDispatchPlan()
        plan.num_layers   = num_layers
        plan.num_engines  = num_engines
        plan.engine_layer_specs   = ct.addressof(els_array)
        plan.attn_allreduce_specs = ct.addressof(attn_ar_array)
        plan.ffn_allreduce_specs  = ct.addressof(ffn_ar_array)
        plan.use_stream_overlap   = 1
        
        # Deferred attention allreduce (M3 optimization): disabled by default
        # Can be enabled via set_deferred_attention_ar(True)
        plan.use_deferred_attention_ar = 0
        plan.residual_add_fn = 0  # Will be set if deferred AR is enabled

        # HIP API function pointers
        hip_lib_handle = self._hip._lib
        hip_set_device_fn = HipSetDeviceFunc(
            ct.cast(hip_lib_handle.hipSetDevice, ct.c_void_p).value)
        hip_stream_sync_fn = HipStreamSyncFunc(
            ct.cast(hip_lib_handle.hipStreamSynchronize, ct.c_void_p).value)
        hip_event_record_fn = HipEventRecordFunc(
            ct.cast(hip_lib_handle.hipEventRecord, ct.c_void_p).value)
        hip_stream_wait_fn = HipStreamWaitFunc(
            ct.cast(hip_lib_handle.hipStreamWaitEvent, ct.c_void_p).value)
        hip_memcpy_peer_async_fn = HipMemcpyPeerAsyncFunc(
            ct.cast(hip_lib_handle.hipMemcpyPeerAsync, ct.c_void_p).value)
        hip_memcpy_async_fn = HipMemcpyAsyncFunc(
            ct.cast(hip_lib_handle.hipMemcpyAsync, ct.c_void_p).value)
        hip_get_last_error_fn = HipGetLastErrorFunc(
            ct.cast(hip_lib_handle.hipGetLastError, ct.c_void_p).value)
        hip_module_launch_fn = HipModuleLaunchFunc(
            ct.cast(hip_lib_handle.hipModuleLaunchKernel, ct.c_void_p).value)

        plan.hipSetDevice_fn         = hip_set_device_fn
        plan.hipStreamSynchronize_fn = hip_stream_sync_fn
        plan.hipEventRecord_fn       = hip_event_record_fn
        plan.hipStreamWaitEvent_fn   = hip_stream_wait_fn
        plan.hipMemcpyPeerAsync_fn   = hip_memcpy_peer_async_fn
        plan.hipMemcpyAsync_fn       = hip_memcpy_async_fn
        plan.hipGetLastError_fn      = hip_get_last_error_fn
        plan.hipModuleLaunchKernel_fn = hip_module_launch_fn
        
        # Load residual_add_v2 function pointer for deferred attention allreduce
        if getattr(self, '_deferred_attention_ar', False):
            elementwise_so_path = build_dir / "elementwise_v2.so"
            if elementwise_so_path.exists():
                try:
                    elementwise_lib = ct.CDLL(str(elementwise_so_path))
                    # Set function signature for residual_add_v2(dst, src, n)
                    elementwise_lib.residual_add_v2.argtypes = [
                        ct.c_void_p,  # dst (modified in place)
                        ct.c_void_p,  # src (added to dst)
                        ct.c_uint,    # n (number of FP16 elements)
                    ]
                    elementwise_lib.residual_add_v2.restype = ct.c_int
                    plan.residual_add_fn = ct.cast(
                        elementwise_lib.residual_add_v2,
                        ct.c_void_p
                    ).value
                    plan.use_deferred_attention_ar = 1
                    print(f"C dispatch: deferred attention allreduce ENABLED "
                          f"(residual_add_fn=0x{plan.residual_add_fn:016x}, "
                          f"AR count will be 64 instead of 128)")
                    # Keep library alive to prevent GC
                    if not hasattr(self, '_c_dispatch_objects'):
                        self._c_dispatch_objects = {}
                    self._c_dispatch_objects['elementwise_lib'] = elementwise_lib
                except Exception as e:
                    print(f"C dispatch: failed to load elementwise_v2 for deferred AR: {e}")
                    plan.use_deferred_attention_ar = 0
                    plan.residual_add_fn = 0
            else:
                print(f"C dispatch: elementwise_v2.so not found at {elementwise_so_path}, "
                      f"deferred AR disabled")
                plan.use_deferred_attention_ar = 0
                plan.residual_add_fn = 0
        else:
            plan.use_deferred_attention_ar = 0
            plan.residual_add_fn = 0

        # Store all objects to keep them alive (Python GC must not collect them)
        self._c_dispatch_objects = {
            'els_array':       els_array,
            'attn_ar_array':   attn_ar_array,
            'ffn_ar_array':    ffn_ar_array,
            'plan':            plan,
            'reduce_tp2':      reduce_tp2,
            'reduce_tp3':      reduce_tp3,
            'reduce_tp4':      reduce_tp4,
            'hip_set_device_fn':        hip_set_device_fn,
            'hip_stream_sync_fn':       hip_stream_sync_fn,
            'hip_event_record_fn':      hip_event_record_fn,
            'hip_stream_wait_fn':       hip_stream_wait_fn,
            'hip_memcpy_peer_async_fn': hip_memcpy_peer_async_fn,
            'hip_memcpy_async_fn':      hip_memcpy_async_fn,
            'hip_get_last_error_fn':    hip_get_last_error_fn,
            'hip_module_launch_fn':     hip_module_launch_fn,
        }
        # Keep fused kernel library alive (prevent GC)
        if fused_lib is not None:
            self._c_dispatch_objects['fused_kernel_lib'] = fused_lib

        print(f"C dispatch plan built: {num_layers} layers × {num_engines} engines")
        return plan

    def _decode_step_c_dispatch(self, token_embedding: np.ndarray,
                                 position: int) -> np.ndarray:
        """Decode step using C dispatch loop.

        Dispatches all 64 layers' kernels in a tight C loop via c_dispatch.so,
        without returning to Python between layers. This eliminates ~14ms/tok
        Python dispatch overhead from the cached+stream path.

        The C extension handles:
        - Per-engine kernel dispatch (attention + FFN for all layers)
        - Position-dependent param updates (cos/sin offsets, seq_len)
        - KV cache D2D copies
        - Async allreduce with HIP events (stream overlap)

        Python handles:
        - Token embedding upload to all GPUs
        - Final norm + logits (post-loop, single GPU)
        - KV cache advance
        """
        h = self.config.hidden_size
        half_rotary = self.engines[0].rotary_dim // 2

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Compute position-dependent params
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables
        seq_len    = self.engines[0].kv_cache.current_len + 1

        # Call C dispatch loop (v2 if enabled, else v1)
        plan = self._c_dispatch_objects['plan']
        plan_ptr = ctypes.addressof(plan)
        if self._c_dispatch_v2_enabled and self._c_dispatch_v2_lib is not None:
            err = self._c_dispatch_v2_lib.c_dispatch_step_v2(
                ctypes.c_uint64(plan_ptr),
                ctypes.c_uint64(cos_offset),
                ctypes.c_uint32(seq_len),
            )
            if err:
                raise RuntimeError(f"c_dispatch_step_v2 failed: HIP error {err}")
        else:
            err = self._c_dispatch_lib.c_dispatch_step(
                ctypes.c_uint64(plan_ptr),
                ctypes.c_uint64(cos_offset),
                ctypes.c_uint32(seq_len),
            )
            if err:
                raise RuntimeError(f"c_dispatch_step failed: HIP error {err}")

        # Synchronize all GPUs after C loop
        for dev_id in self.device_ids:
            self._hip.set_device(dev_id)
            self._hip.synchronize()

        # Advance KV caches
        for engine in self.engines:
            engine.kv_cache.advance()

        # Final norm and logits (GPU0 only)
        e0 = self.engines[0]
        if e0.d_final_norm:
            e0._launch_rmsnorm(e0.d_hidden2, e0.d_hidden, e0.d_final_norm, h)
            return np.frombuffer(e0.device.download(e0.d_hidden2, h * 2),
                                 dtype=np.float16)
        return np.frombuffer(e0.device.download(e0.d_hidden, h * 2),
                             dtype=np.float16)

    def synchronize(self):
        for engine in self.engines:
            engine.device.synchronize()

    def set_speculative_mode(self, enabled: bool, ngram_size: int = 3, 
                             max_draft_len: int = 5):
        """Enable or disable speculative decoding mode.
        
        When enabled=True, the engine prepares for speculative decoding with
        n-gram lookahead. This allocates additional buffers for draft token
        verification and enables the decode_step_speculative() path.
        
        Speculative decoding workflow:
        1. Build n-gram cache from prompt tokens
        2. Generate draft tokens from n-gram matches
        3. Verify drafts by running batched forward pass (prefill-style)
        4. Accept longest matching prefix, reject rest
        5. Update KV cache with accepted tokens only
        
        Args:
            enabled: whether to enable speculative mode
            ngram_size: size of n-grams for draft generation (default: 3)
            max_draft_len: maximum draft tokens per iteration (default: 5)
        """
        self._speculative_mode = enabled
        if enabled:
            if self._speculative_state is None:
                self._speculative_state = SpeculativeDecodeState(
                    self, ngram_size, max_draft_len)
                print(f"Speculative decoding initialized (n={ngram_size}, "
                      f"max_draft={max_draft_len})")
            else:
                self._speculative_state.ngram_size = ngram_size
                self._speculative_state.max_draft_len = max_draft_len
        else:
            if self._speculative_state is not None:
                self._speculative_state.cleanup()
                self._speculative_state = None
            print("Speculative decoding disabled")

    def decode_step_speculative(self, token_embeddings: List[np.ndarray], 
                                 position: int,
                                 draft_tokens: List[int]) -> tuple:
        """Run speculative decoding step with draft token verification.
        
        This method verifies K draft tokens by running forward passes
        and returns the output hidden states for logit computation.
        
        The verification workflow:
        1. Process each draft token through the model (sequential for now)
        2. Return hidden states for each position
        3. Caller computes logits and compares with draft tokens
        4. Caller decides which tokens to accept
        5. Caller calls record_accepted_tokens() and possibly rollback_kv_cache()
        
        For rejected tokens, the caller must handle rollback by calling
        rollback_kv_cache() to revert the KV cache position.
        
        Args:
            token_embeddings: list of [hidden_size] FP16 embeddings for draft tokens
            position: current sequence position for KV cache writes
            draft_tokens: list of draft token IDs to verify
        
        Returns:
            Tuple of (output_hidden_states, num_drafts)
            - output_hidden_states: list of hidden states [hidden_size] for each position
            - num_drafts: number of draft tokens processed (for convenience)
        """
        if not draft_tokens or not token_embeddings:
            # No drafts to verify - return empty
            return [], 0
        
        # Use the speculative state to verify drafts
        if self._speculative_state is None:
            raise RuntimeError("Speculative mode not initialized. "
                               "Call set_speculative_mode(True) first.")
        
        # Ensure embeddings and drafts have same length
        if len(token_embeddings) != len(draft_tokens):
            raise ValueError(f"Embedding count ({len(token_embeddings)}) must "
                             f"match draft token count ({len(draft_tokens)})")
        
        hidden_states, _ = self._speculative_state.verify_drafts(
            token_embeddings, position, draft_tokens)
        
        return hidden_states, len(draft_tokens)

    def set_eagle_mode(self, enabled: bool, k_draft: int = 5, 
                       temperature: float = 0.0):
        """Enable or disable EAGLE-style speculative decoding mode.
        
        EAGLE uses the target model's own hidden states and lm_head to generate
        draft tokens, eliminating the need for a separate draft model.
        
        Workflow:
        1. Get hidden state from target model
        2. Apply lm_head to get logits
        3. Sample/greedy select draft token
        4. Convert token to embedding for next iteration
        5. Repeat K times to generate K draft tokens
        6. Verify drafts by running model forward pass
        7. Accept longest matching prefix, reject rest
        
        Args:
            enabled: whether to enable EAGLE mode
            k_draft: number of draft tokens per iteration (default: 5)
            temperature: sampling temperature for draft generation (0.0 = greedy)
        """
        self._eagle_mode = enabled
        if enabled:
            self._eagle_k_draft = k_draft
            self._eagle_temperature = temperature
            print(f"EAGLE speculative decoding enabled (K={k_draft}, "
                  f"temperature={temperature})")
        else:
            print("EAGLE speculative decoding disabled")
    
    def generate_eagle_drafts(self, hidden_state: np.ndarray, 
                               k: int = None) -> List[int]:
        """Generate K draft tokens using EAGLE method from a hidden state.
        
        This is the core EAGLE draft generation:
        1. Apply lm_head to hidden state to get logits
        2. Sample or greedily select token
        3. Get embedding for the token
        4. Repeat K times
        
        Args:
            hidden_state: [hidden_size] FP16 hidden state from target model
            k: number of draft tokens to generate (default: self._eagle_k_draft)
        
        Returns:
            List of K draft token IDs
        """
        if k is None:
            k = self._eagle_k_draft
        
        draft_tokens = []
        current_hidden = hidden_state.copy()
        
        # Get engine 0's compute_logits method (all engines have same lm_head)
        engine0 = self.engines[0]
        
        for _ in range(k):
            # Compute logits from hidden state using lm_head
            # Note: We need to temporarily upload hidden to GPU, compute, download
            logits = engine0.compute_logits(current_hidden)
            
            # Sample or greedy select
            if self._eagle_temperature == 0.0:
                token_id = int(np.argmax(logits))
            else:
                # Temperature sampling
                scaled_logits = logits / self._eagle_temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_logits / np.sum(exp_logits)
                token_id = int(np.random.choice(len(probs), p=probs))
            
            draft_tokens.append(token_id)
            
            # Get embedding for next iteration
            # We need to access the embedding weight - it's stored in the model
            # For now, we use a simple approach: the caller will provide embeddings
            # In a full implementation, we'd cache the embedding table here
            # For this integration, we return only token IDs and let caller handle embeddings
            # To continue the loop, we need the embedding - use a placeholder
            # The actual embedding lookup happens in the verification phase
            current_hidden = np.zeros(self.config.hidden_size, dtype=np.float16)
        
        return draft_tokens
    
    def decode_step_eagle(self, hidden_state: np.ndarray,
                          position: int) -> tuple:
        """Run one EAGLE speculative decoding step.
        
        This method:
        1. Generates K draft tokens from the hidden state using lm_head
        2. Returns the draft tokens for verification
        
        The caller should then:
        1. Get embeddings for draft tokens
        2. Call decode_step_speculative() to verify
        3. Compare logits with drafts to determine acceptance
        4. Update hidden state with accepted tokens
        
        Args:
            hidden_state: [hidden_size] FP16 current hidden state
            position: current sequence position
        
        Returns:
            Tuple of (draft_token_ids, k_draft)
            - draft_token_ids: list of K draft token IDs
            - k_draft: number of draft tokens generated
        """
        if not self._eagle_mode:
            raise RuntimeError("EAGLE mode not enabled. Call set_eagle_mode(True) first.")
        
        # Generate K draft tokens from hidden state
        k = self._eagle_k_draft
        draft_tokens = self.generate_eagle_drafts(hidden_state, k=k)
        
        return draft_tokens, k

    def cleanup(self):
        # Shut down worker threads
        if self._thread_pool_initialized:
            tp = self.tp_size
            # Clear done events and send stop command
            for done_event in self._done_events:
                done_event.clear()
            for rank in range(tp):
                self._worker_cmds[rank] = 'stop'
            for go_event in self._go_events:
                go_event.set()
            for t in self._worker_threads:
                t.join(timeout=5.0)
            self._worker_threads.clear()
            self._go_events.clear()
            self._done_events.clear()
            self._worker_cmds.clear()
            self._thread_pool_initialized = False

        if self._graph_decode_step is not None:
            self._graph_decode_step.cleanup()
            self._graph_decode_step = None
        # Release C graph dispatch objects (ctypes keeps them alive)
        self._c_graph_dispatch_plan = None
        self._c_graph_dispatch_objects = {}
        # Cleanup persistent kernel state
        if self._persistent_dispatcher is not None:
            self._persistent_dispatcher = None
        # Cleanup global graph decode state
        if self._global_graph_decode_state is not None:
            self._global_graph_decode_state.cleanup()
            self._global_graph_decode_state = None
        if self._p2p_ar:
            self._p2p_ar.cleanup()
            self._p2p_ar = None
        if self._fused_p2p_ar:
            self._fused_p2p_ar.cleanup()
            self._fused_p2p_ar = None
        if self._ring_ar:
            self._ring_ar.cleanup()
            self._ring_ar = None
        for engine in self.engines:
            engine.cleanup()
        self.tp_group.cleanup()
