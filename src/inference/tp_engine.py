"""
Tensor-parallel inference engine for Qwen 3.5 27B across multiple MI50s.

Uses fast_ar_fused_tp2 C extension for all allreduces: downloads partials +
hidden from both GPUs, AVX F16C 3-way add on host, uploads result.
"""

import ctypes
import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

from src.inference.engine import InferenceEngine
from src.runtime.tensor_parallel import TensorParallelGroup
from src.model.qwen import QwenConfig


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


class TPInferenceEngine:
    """Multi-GPU tensor-parallel inference engine."""

    def __init__(self, config: QwenConfig, device_ids: list,
                 max_seq_len: int = 2048):
        self.config = config
        self.tp_size = len(device_ids)
        self.device_ids = device_ids

        self.tp_group = TensorParallelGroup(device_ids)

        self.engines = []
        for rank, dev_id in enumerate(device_ids):
            engine = InferenceEngine(
                config, device_id=dev_id, max_seq_len=max_seq_len,
                tp_size=self.tp_size, tp_rank=rank,
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

        # Fallback host buffers
        h = config.hidden_size
        self._host_bufs = [ctypes.create_string_buffer(h * 2)
                           for _ in range(self.tp_size)]
        self._host_hidden = ctypes.create_string_buffer(h * 2)

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

    def compute_logits(self, hidden: np.ndarray = None) -> np.ndarray:
        return self.engines[0].compute_logits(hidden)

    def decode_step(self, token_embedding: np.ndarray, position: int) -> np.ndarray:
        """Run one decode step with fused host-side allreduce.

        Every allreduce uses fast_ar_fused: downloads partials + hidden,
        AVX FP16 add on host, uploads result.
        """
        h = self.config.hidden_size
        cfg = self.config
        num_layers = cfg.num_hidden_layers

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        for layer_idx in range(num_layers):
            lw_list = [e.layers[layer_idx] for e in self.engines]

            # RMSNorm for attention
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
            self._allreduce_residual("d_proj_out", h)

            # RMSNorm for FFN
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

    def _allreduce_sum(self, buffer_name: str, hidden_size: int):
        """Allreduce partials only (no hidden download). 2 D2H + 2 H2D."""
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

    def _allreduce_residual(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add to d_hidden."""
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

    def synchronize(self):
        for engine in self.engines:
            engine.device.synchronize()

    def cleanup(self):
        for engine in self.engines:
            engine.cleanup()
        self.tp_group.cleanup()
