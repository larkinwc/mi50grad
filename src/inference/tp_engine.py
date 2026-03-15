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
from typing import Optional

from src.inference.engine import InferenceEngine
from src.runtime.tensor_parallel import TensorParallelGroup
from src.runtime.p2p_allreduce import P2PAllreduce
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

        Dispatches to cached, threaded, or serial implementation based on flags.
        Cached dispatch (set_cached_dispatch(True)) uses pre-built ctypes parameter
        arrays for lower Python overhead. Set threading with set_threaded_dispatch().
        """
        if self._cached_dispatch and self._engine_layer_caches:
            return self._decode_step_cached(token_embedding, position)
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
                    if 'gemv_q_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_q_fused'])
                    if 'gemv_kv_fused' in layer_cache:
                        engine.device.launch_cached(layer_cache['gemv_kv_fused'])

                    # Sync Q+KV streams if needed
                    if engine._streams_ready:
                        engine.device.hip.stream_synchronize(engine._stream_q)
                        engine.device.hip.stream_synchronize(engine._stream_kv)

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
                        engine.device.launch_cached(spec_k)

                    # GPU-to-GPU KV cache update
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

    def _decode_step_serial(self, token_embedding: np.ndarray,
                             position: int) -> np.ndarray:
        """Serial (original) decode step implementation.

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

        Tries P2P GPU allreduce first, falls back to fast_ar C extension,
        then Python fallback.
        """
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

    def _allreduce_residual(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add to d_hidden.

        Tries P2P GPU allreduce first, falls back to fast_ar C extension,
        then Python fallback.
        """
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

    def synchronize(self):
        for engine in self.engines:
            engine.device.synchronize()

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

        if self._p2p_ar:
            self._p2p_ar.cleanup()
            self._p2p_ar = None
        for engine in self.engines:
            engine.cleanup()
        self.tp_group.cleanup()
