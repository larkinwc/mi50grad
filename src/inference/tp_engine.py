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
from src.runtime.p2p_allreduce import P2PAllreduce, FusedP2PReduce, RingAllreduce
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

        Dispatches to C dispatch, cached+stream (combined), cached,
        stream_overlap, threaded, or serial implementation based on flags.

        Priority order:
        0. C dispatch (highest): _c_dispatch_enabled is True. Dispatches all
           64 layers in a tight C loop via c_dispatch.so, eliminating Python
           dispatch overhead.
        1. Combined (cached + stream overlap): _cached_dispatch and
           _stream_overlap_dispatch are True and _p2p_ar is available.
           NOTE: stream overlap always uses standard P2P allreduce (not fused),
           because the fused kernel's async variant has more cross-GPU event overhead.
           The fused P2P kernel is faster in serial/synchronous mode but the
           standard P2P allreduce overlaps better with cached dispatch.
        2. Cached dispatch only: _cached_dispatch is True.
        3. Stream overlap only: _stream_overlap_dispatch is True.
        4. Threaded: _threaded_dispatch is True.
        5. Serial: fallback.
        """
        if self._c_dispatch_enabled and self._c_dispatch_plan is not None:
            return self._decode_step_c_dispatch(token_embedding, position)
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

        # Upload embedding to all GPUs
        emb_bytes = token_embedding.tobytes()
        for engine in self.engines:
            engine.device.upload(engine.d_hidden, emb_bytes)

        # Pre-compute RoPE offset for this position
        cos_offset = position * half_rotary * 2  # byte offset into cos/sin tables

        for layer_idx in range(num_layers):
            # --- Compute stream waits for previous layer's FFN allreduce ---
            # GPU-side wait: hipStreamWaitEvent on allreduce_done events for all GPUs.
            # (Layer 0: allreduce events are unrecorded on ROCm →
            #  hipStreamWaitEvent on unrecorded event is a no-op, safe.)
            if layer_idx > 0:
                p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]
                lw = engine.layers[layer_idx]

                # --- Attention RMSNorm (cached, all static) ---
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
            hidden_ptrs = [e.d_hidden for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

            # --- Make compute stream wait for attention allreduce before FFN RMSNorm ---
            p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)

            for engine_idx, (engine, lc) in enumerate(
                    zip(self.engines, self._engine_layer_caches)):
                layer_cache = lc[layer_idx]

                # --- FFN RMSNorm (cached, reads d_hidden — gated by attention AR done) ---
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
            # Non-blocking: next layer will call wait_for_allreduce_on_compute_stream()
            # before launching its RMSNorm.
            partial_ptrs = [e.d_ffn_out for e in self.engines]
            p2p_ar.allreduce_residual_async(
                partial_ptrs, hidden_ptrs, h, compute_streams)

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

    def _allreduce_residual(self, buffer_name: str, hidden_size: int):
        """Allreduce + residual add to d_hidden.

        Tries fused P2P reduce first (when enabled), then standard P2P GPU allreduce,
        then fast_ar C extension, then Python fallback.
        """
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
                ('_pad',            ct.c_uint32),
            ]

        class CDispatchPlan(ct.Structure):
            _fields_ = [
                ('num_layers',              ct.c_int),
                ('num_engines',             ct.c_int),
                ('engine_layer_specs',      ct.c_uint64),
                ('attn_allreduce_specs',    ct.c_uint64),
                ('ffn_allreduce_specs',     ct.c_uint64),
                ('use_stream_overlap',      ct.c_int),
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

        def fill_ar_spec(c_ar, partial_attr):
            """Fill a CAllreduceSpec for one layer."""
            c_ar.reduce_tp2 = reduce_tp2
            c_ar.reduce_tp3 = reduce_tp3
            c_ar.reduce_tp4 = reduce_tp4
            c_ar.tp_size = self.tp_size
            for i, engine in enumerate(self.engines):
                c_ar.device_ids[i]    = engine.device.device_id
                c_ar.partial_ptrs[i]  = getattr(engine, partial_attr)
                c_ar.hidden_ptrs[i]   = engine.d_hidden
                c_ar.compute_streams[i] = 0  # null stream (default)
            # Gather bufs (on GPU0, for TP>1)
            for i, ptr in enumerate(p2p_ar._gather_bufs):
                c_ar.gather_bufs[i] = ptr
            # Allreduce streams and events from p2p_ar
            for i in range(self.tp_size):
                c_ar.allreduce_streams[i] = p2p_ar._allreduce_streams[i]
                c_ar.compute_events[i]    = p2p_ar._compute_events[i]
                c_ar.ar_done_events[i]    = p2p_ar._ar_done_events[i]
            c_ar.num_elems = self.config.hidden_size

        for layer_idx in range(num_layers):
            fill_ar_spec(attn_ar_array[layer_idx], 'd_proj_out')
            fill_ar_spec(ffn_ar_array[layer_idx],  'd_ffn_out')

        # Build the CDispatchPlan
        plan = CDispatchPlan()
        plan.num_layers   = num_layers
        plan.num_engines  = num_engines
        plan.engine_layer_specs   = ct.addressof(els_array)
        plan.attn_allreduce_specs = ct.addressof(attn_ar_array)
        plan.ffn_allreduce_specs  = ct.addressof(ffn_ar_array)
        plan.use_stream_overlap   = 1

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

        # Call C dispatch loop
        plan = self._c_dispatch_objects['plan']
        plan_ptr = ctypes.addressof(plan)
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
        if self._fused_p2p_ar:
            self._fused_p2p_ar.cleanup()
            self._fused_p2p_ar = None
        if self._ring_ar:
            self._ring_ar.cleanup()
            self._ring_ar = None
        for engine in self.engines:
            engine.cleanup()
        self.tp_group.cleanup()
