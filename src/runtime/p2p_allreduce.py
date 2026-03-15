"""
GPU P2P allreduce module for TP inference.

Replaces the CPU-mediated fast_allreduce.c path with:
1. hipMemcpyPeerAsync gather partials from all GPUs to GPU0 (async, non-blocking)
2. On-device HIP reduce kernel on GPU0 (sum FP16 partials + optional hidden for residual)
3. hipMemcpyPeerAsync broadcast result from GPU0 to all GPUs (async)
4. Single synchronization per GPU stream

The HIP reduce kernels are compiled as a shared library (.so) and the
host-callable C wrappers use hipLaunchKernelGGL (HIP C++ runtime launch),
which is reliable from ctypes without requiring hipModuleLaunchKernel.

Alternative (pinned_async mode): Use pinned host memory buffers with async H2D/D2H.
This avoids host roundtrip overhead while keeping host-side FP32 accumulation.
Useful if P2P is not faster than the pinned path for 2-hop PCIe topology.

For hidden_size=5120 FP16 payload (10 KB):
- Old path: ~9 synchronous hipMemcpy calls, ~5-10us each = ~50-90us/allreduce
- P2P path: 2 async gather + 1 kernel + 1 async broadcast + 1 sync = ~15-30us target
"""

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Optional

from .hip_dispatch import HIPRuntime, HIPError


class P2PAllreduce:
    """GPU P2P allreduce for TP inference.

    Uses hipMemcpyPeerAsync for gather/broadcast and a HIP C++ kernel
    (compiled as shared library) for on-device FP16 reduction.

    Usage:
        ar = P2PAllreduce(hip, device_ids, hidden_size)
        ar.allreduce_residual(partial_ptrs, hidden_ptrs, num_elems)
        ar.allreduce_sum(partial_ptrs, num_elems)
        ar.cleanup()
    """

    def __init__(self, hip: HIPRuntime, device_ids: list, hidden_size: int,
                 streams: Optional[list] = None):
        """Initialize P2P allreduce.

        Args:
            hip: HIPRuntime instance
            device_ids: list of GPU device IDs (GPU0 is the reduce root)
            hidden_size: max number of FP16 elements (buffer size)
            streams: optional per-GPU streams for async operations.
                     If None, uses default stream (0).
        """
        self._hip = hip
        self._device_ids = list(device_ids)
        self._tp_size = len(device_ids)
        self._hidden_size = hidden_size
        self._streams = streams if streams else [0] * len(device_ids)

        # Load kernel shared library
        self._lib = None
        self._load_lib()

        # Allocate temporary gather buffers on GPU0 for remote partials
        # (one per remote GPU: tp_size - 1 buffers)
        size = hidden_size * 2  # FP16 = 2 bytes
        self._gather_bufs = []  # pointers on GPU0
        dev0_id = self._device_ids[0]
        hip.set_device(dev0_id)
        for i in range(1, self._tp_size):
            ptr = hip.malloc(size)
            self._gather_bufs.append(ptr)

        # Dedicated allreduce streams (non-blocking) for async allreduce overlap.
        # Using non-blocking streams avoids the null stream's implicit serialization,
        # allowing allreduce to truly run concurrently with compute (null stream)
        # when using allreduce_residual_async().
        self._allreduce_streams = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._allreduce_streams.append(hip.stream_create_nonblocking())

        # Compute completion events (one per GPU): recorded after GEMV on compute stream,
        # waited on by allreduce stream before P2P gather.
        self._compute_events = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._compute_events.append(hip.event_create())

        # Allreduce completion events (one per GPU): recorded on allreduce stream after
        # broadcast, waited on by compute stream before next RMSNorm reads d_hidden.
        self._ar_done_events = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._ar_done_events.append(hip.event_create())

        # Enable P2P access between all pairs (idempotent)
        self._enable_p2p()

    def _enable_p2p(self):
        """Enable P2P access between all GPU pairs."""
        hip = self._hip
        for i, dev_i in enumerate(self._device_ids):
            hip.set_device(dev_i)
            for j, dev_j in enumerate(self._device_ids):
                if i == j:
                    continue
                if hip.device_can_access_peer(dev_i, dev_j):
                    try:
                        hip.device_enable_peer_access(dev_j)
                    except HIPError:
                        pass  # Already enabled

    def _load_lib(self):
        """Build and load the p2p_allreduce.hip kernel as shared library (.so).

        Uses hipcc -shared -fPIC to compile a shared library with host-callable
        C wrappers (p2p_reduce_residual_tp2, etc.) that use hipLaunchKernelGGL
        internally. This is more reliable than hipModuleLaunchKernel from ctypes.
        """
        src_dir = Path(__file__).parent.parent / "kernels"
        hip_path = src_dir / "p2p_allreduce.hip"
        build_dir = Path(__file__).parent.parent.parent / "build" / "kernels"
        so_path = build_dir / "p2p_allreduce.so"

        if not hip_path.exists():
            raise FileNotFoundError(f"Kernel not found: {hip_path}")

        # Build if missing or stale
        build_dir.mkdir(parents=True, exist_ok=True)
        if not so_path.exists() or (
                os.path.getmtime(hip_path) > os.path.getmtime(so_path)):
            try:
                subprocess.check_call([
                    "/opt/rocm/bin/hipcc",
                    "-O3", "--offload-arch=gfx906", "-std=c++17",
                    "-shared", "-fPIC",
                    "-o", str(so_path),
                    str(hip_path),
                ])
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(f"Failed to build p2p_allreduce.hip: {e}")

        # Load the shared library via ctypes
        lib = ctypes.CDLL(str(so_path))

        # Register function signatures
        # p2p_reduce_residual_tp2(hidden, partial0, partial1, n, stream)
        lib.p2p_reduce_residual_tp2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p  # stream
        ]
        lib.p2p_reduce_residual_tp2.restype = ctypes.c_int

        lib.p2p_reduce_residual_tp3.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.p2p_reduce_residual_tp3.restype = ctypes.c_int

        lib.p2p_reduce_residual_tp4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.p2p_reduce_residual_tp4.restype = ctypes.c_int

        lib.p2p_reduce_sum_tp2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.p2p_reduce_sum_tp2.restype = ctypes.c_int

        lib.p2p_reduce_sum_tp3.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.p2p_reduce_sum_tp3.restype = ctypes.c_int

        lib.p2p_reduce_sum_tp4.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.p2p_reduce_sum_tp4.restype = ctypes.c_int

        self._lib = lib

    def _stream_ptr(self, stream: int):
        """Convert Python int stream handle to ctypes c_void_p."""
        return ctypes.c_void_p(stream) if stream else None

    def allreduce_residual(self, partial_ptrs: list, hidden_ptrs: list,
                           num_elems: int, skip_device_sync: bool = False):
        """P2P allreduce + residual add.

        Computes: hidden[i] = hidden[0] + partial[0] + partial[1] + ...
        and broadcasts result to all GPUs' hidden buffers.

        Args:
            partial_ptrs: list of device ptrs, partial_ptrs[i] on device i
            hidden_ptrs:  list of device ptrs, hidden_ptrs[i] on device i
            num_elems:    number of FP16 elements
            skip_device_sync: if True, skip hipDeviceSynchronize() (caller
                              guarantees all GPU work has already completed).
                              Used with threaded dispatch where workers call
                              hipDeviceSynchronize() in parallel before allreduce.
        """
        tp = self._tp_size
        if tp == 1:
            return

        hip = self._hip
        size = num_elems * 2
        dev0_id = self._device_ids[0]
        stream0 = self._streams[0]
        stream0_ptr = ctypes.c_void_p(stream0)

        # Step 0: Synchronize all GPUs to ensure compute kernels have completed
        # and their partial results are ready for P2P gather.
        # Without this, the P2P copy might read stale/unfinished data.
        # skip_device_sync=True: caller has already synchronized (e.g., via
        # parallel hipDeviceSynchronize() calls in worker threads).
        if not skip_device_sync:
            for i in range(tp):
                hip.set_device(self._device_ids[i])
                hip.synchronize()

        # Step 1: Gather remote partials to GPU0 (async P2P)
        hip.set_device(dev0_id)
        for i in range(1, tp):
            hip.memcpy_peer_async(
                self._gather_bufs[i - 1], dev0_id,
                partial_ptrs[i], self._device_ids[i],
                size, stream0)

        # Wait for all P2P copies to complete on stream0
        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)

        # Step 2: Run on-device reduce kernel on GPU0
        # hidden[0] = hidden[0] + partial[0] + gathered[1] + ... gathered[tp-1]
        hip.set_device(dev0_id)
        if tp == 2:
            err = self._lib.p2p_reduce_residual_tp2(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        elif tp == 3:
            err = self._lib.p2p_reduce_residual_tp3(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        elif tp == 4:
            err = self._lib.p2p_reduce_residual_tp4(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_void_p(self._gather_bufs[2]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        else:
            raise ValueError(f"tp_size={tp} not supported (2-4 only)")

        if err != 0:
            raise HIPError(f"p2p_reduce_residual_tp{tp} kernel failed: HIP error {err}")

        # Step 3: Broadcast result from GPU0's hidden to all other GPUs (async P2P)
        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)  # Wait for kernel to finish
        for i in range(1, tp):
            hip.memcpy_peer_async(
                hidden_ptrs[i], self._device_ids[i],
                hidden_ptrs[0], dev0_id,
                size, self._streams[i])

        # Step 4: Sync all streams
        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)
        for i in range(1, tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

    def allreduce_sum(self, partial_ptrs: list, num_elems: int):
        """P2P allreduce sum only (no hidden residual add).

        Computes: partial[i] = partial[0] + partial[1] + ...
        and broadcasts result to all GPUs.

        Args:
            partial_ptrs: list of device ptrs, partial_ptrs[i] on device i
            num_elems:    number of FP16 elements
        """
        tp = self._tp_size
        if tp == 1:
            return

        hip = self._hip
        size = num_elems * 2
        dev0_id = self._device_ids[0]
        stream0 = self._streams[0]
        stream0_ptr = ctypes.c_void_p(stream0)

        # Synchronize all GPUs to ensure compute kernels have completed
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.synchronize()

        # Gather remote partials to GPU0
        hip.set_device(dev0_id)
        for i in range(1, tp):
            hip.memcpy_peer_async(
                self._gather_bufs[i - 1], dev0_id,
                partial_ptrs[i], self._device_ids[i],
                size, stream0)
        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)

        # Run reduce kernel on GPU0 (writes into partial_ptrs[0])
        hip.set_device(dev0_id)
        if tp == 2:
            err = self._lib.p2p_reduce_sum_tp2(
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        elif tp == 3:
            err = self._lib.p2p_reduce_sum_tp3(
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        elif tp == 4:
            err = self._lib.p2p_reduce_sum_tp4(
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_void_p(self._gather_bufs[2]),
                ctypes.c_uint(num_elems),
                stream0_ptr)
        else:
            raise ValueError(f"tp_size={tp} not supported (2-4 only)")

        if err != 0:
            raise HIPError(f"p2p_reduce_sum_tp{tp} kernel failed: HIP error {err}")

        # Broadcast result from GPU0 to all other GPUs
        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)
        for i in range(1, tp):
            hip.memcpy_peer_async(
                partial_ptrs[i], self._device_ids[i],
                partial_ptrs[0], dev0_id,
                size, self._streams[i])

        hip.set_device(dev0_id)
        hip.stream_synchronize(stream0)
        for i in range(1, tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

    def record_compute_events(self, compute_streams: list):
        """Record compute completion events on each GPU's compute stream.

        Call this after GEMV kernel launches (before allreduce). The events
        are then waited on by the allreduce stream, creating a GPU-side
        dependency that replaces hipDeviceSynchronize().

        Args:
            compute_streams: per-GPU compute stream handles (list of int)
        """
        hip = self._hip
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            hip.event_record(self._compute_events[i], compute_streams[i])

    def allreduce_residual_async(self, partial_ptrs: list, hidden_ptrs: list,
                                 num_elems: int, compute_streams: list):
        """P2P allreduce + residual add, submitted asynchronously.

        This is a non-blocking version of allreduce_residual() that:
        1. Records compute events on each GPU's compute stream
        2. Makes the allreduce stream wait on all compute events (GPU-side wait)
        3. Runs P2P gather + reduce kernel + broadcast on dedicated allreduce streams
        4. Records allreduce completion events (for next-layer compute to wait on)

        The Python call returns immediately after submitting GPU work; the GPU
        executes the allreduce on its dedicated stream concurrently with any
        subsequent Python-submitted work on the compute stream.

        Next-layer's first kernel (RMSNorm reading d_hidden) must call
        wait_for_allreduce_on_compute_stream() before it can safely read d_hidden.

        Args:
            partial_ptrs: list of device ptrs, partial_ptrs[i] on device i
            hidden_ptrs:  list of device ptrs, hidden_ptrs[i] on device i
            num_elems:    number of FP16 elements
            compute_streams: per-GPU compute stream handles (for event recording)
        """
        tp = self._tp_size
        if tp == 1:
            return

        hip = self._hip
        size = num_elems * 2
        dev0_id = self._device_ids[0]
        ar_stream0 = self._allreduce_streams[0]
        ar_stream0_ptr = ctypes.c_void_p(ar_stream0)

        # Step 1: Record compute completion events on each GPU's compute stream.
        # The allreduce stream will wait on these before reading partial results.
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            hip.event_record(self._compute_events[i], compute_streams[i])

        # Step 2: Make allreduce stream on GPU0 wait for ALL compute events.
        # This is a GPU-side wait (no CPU blocking). The allreduce stream
        # queues the dependency and won't start P2P copies until all GEMV
        # kernels have completed on all GPUs.
        hip.set_device(dev0_id)
        for i in range(tp):
            hip.stream_wait_event(ar_stream0, self._compute_events[i])

        # Step 3: Async P2P gather partials to GPU0 on allreduce stream
        for i in range(1, tp):
            hip.memcpy_peer_async(
                self._gather_bufs[i - 1], dev0_id,
                partial_ptrs[i], self._device_ids[i],
                size, ar_stream0)

        # Step 4: Launch reduce kernel on GPU0 allreduce stream
        # (P2P copies above complete before kernel runs, since same stream)
        if tp == 2:
            err = self._lib.p2p_reduce_residual_tp2(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_uint(num_elems),
                ar_stream0_ptr)
        elif tp == 3:
            err = self._lib.p2p_reduce_residual_tp3(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_uint(num_elems),
                ar_stream0_ptr)
        elif tp == 4:
            err = self._lib.p2p_reduce_residual_tp4(
                ctypes.c_void_p(hidden_ptrs[0]),
                ctypes.c_void_p(partial_ptrs[0]),
                ctypes.c_void_p(self._gather_bufs[0]),
                ctypes.c_void_p(self._gather_bufs[1]),
                ctypes.c_void_p(self._gather_bufs[2]),
                ctypes.c_uint(num_elems),
                ar_stream0_ptr)
        else:
            raise ValueError(f"tp_size={tp} not supported (2-4 only)")

        if err != 0:
            raise HIPError(f"p2p_reduce_residual_tp{tp} kernel failed: HIP error {err}")

        # Step 5: Record event on GPU0's allreduce stream after reduce kernel.
        # Broadcast streams for other GPUs must wait on this event.
        hip.set_device(dev0_id)
        hip.event_record(self._ar_done_events[0], ar_stream0)

        # Broadcast to all other GPUs, each on their own allreduce stream
        for i in range(1, tp):
            hip.set_device(self._device_ids[i])
            # Make GPU[i]'s allreduce stream wait for GPU0's reduce to complete
            hip.stream_wait_event(self._allreduce_streams[i], self._ar_done_events[0])
            # Async P2P broadcast to GPU[i]
            hip.memcpy_peer_async(
                hidden_ptrs[i], self._device_ids[i],
                hidden_ptrs[0], dev0_id,
                size, self._allreduce_streams[i])
            # Record completion event for GPU[i]
            hip.event_record(self._ar_done_events[i], self._allreduce_streams[i])

        # Update GPU0's completion event after all broadcast dispatches are queued
        hip.set_device(dev0_id)
        hip.event_record(self._ar_done_events[0], ar_stream0)
        # Python call returns here. GPU continues asynchronously.

    def wait_for_allreduce_on_compute_stream(self, compute_streams: list):
        """Make each GPU's compute stream wait for allreduce completion (GPU-side wait).

        Call this before launching the next-layer's first kernel (RMSNorm) that
        reads d_hidden. This creates a GPU-side dependency without CPU blocking.

        The kernel launch itself can happen immediately (HIP queues it on the
        compute stream, which will wait on the allreduce completion event
        before executing the kernel).

        Args:
            compute_streams: per-GPU compute stream handles
        """
        hip = self._hip
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            # Make compute stream wait on allreduce done event for THIS GPU
            hip.stream_wait_event(compute_streams[i], self._ar_done_events[i])

    def cleanup(self):
        """Free temporary buffers."""
        if self._gather_bufs:
            self._hip.set_device(self._device_ids[0])
            for ptr in self._gather_bufs:
                try:
                    self._hip.free(ptr)
                except HIPError:
                    pass
            self._gather_bufs.clear()

        # Destroy dedicated allreduce streams
        for i, dev_id in enumerate(self._device_ids):
            try:
                self._hip.set_device(dev_id)
                if i < len(self._allreduce_streams):
                    self._hip.stream_destroy(self._allreduce_streams[i])
            except HIPError:
                pass
        self._allreduce_streams.clear()

        # Destroy compute events
        for i, dev_id in enumerate(self._device_ids):
            try:
                self._hip.set_device(dev_id)
                if i < len(self._compute_events):
                    self._hip.event_destroy(self._compute_events[i])
            except HIPError:
                pass
        self._compute_events.clear()

        # Destroy allreduce done events
        for i, dev_id in enumerate(self._device_ids):
            try:
                self._hip.set_device(dev_id)
                if i < len(self._ar_done_events):
                    self._hip.event_destroy(self._ar_done_events[i])
            except HIPError:
                pass
        self._ar_done_events.clear()

        self._lib = None


class PinnedAsyncAllreduce:
    """Pinned-memory async allreduce for TP inference.

    Alternative to P2P allreduce for PCIe topologies where P2P has high
    latency (2 hops). Uses hipMallocHost (pinned) + hipMemcpyAsync for
    non-blocking D2H/H2D, with host-side FP32 accumulation.

    Reduces sync points from 9 (old path) to 1 (sync at the end).
    """

    def __init__(self, hip: HIPRuntime, device_ids: list, hidden_size: int,
                 streams: Optional[list] = None):
        self._hip = hip
        self._device_ids = list(device_ids)
        self._tp_size = len(device_ids)
        self._streams = streams if streams else [0] * len(device_ids)

        # Allocate pinned host buffers (one per GPU for D2H, one for result H2D)
        size = hidden_size * 2
        self._host_partials = []
        self._host_hidden = hip.host_malloc(size)
        self._host_result = hip.host_malloc(size)
        for i in range(self._tp_size):
            self._host_partials.append(hip.host_malloc(size))

    def allreduce_residual(self, partial_ptrs: list, hidden_ptrs: list,
                           num_elems: int):
        """Pinned-async allreduce + residual add."""
        import numpy as np
        import ctypes as ct
        tp = self._tp_size
        size = num_elems * 2
        hip = self._hip

        # Async D2H partials from all GPUs
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.memcpy_d2h_async(
                self._host_partials[i], partial_ptrs[i], size, self._streams[i])

        # Also async D2H hidden from GPU0
        hip.set_device(self._device_ids[0])
        hip.memcpy_d2h_async(
            self._host_hidden, hidden_ptrs[0], size, self._streams[0])

        # Wait for all D2H to complete
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

        # CPU FP32 accumulate
        hidden_arr = np.frombuffer(
            (ct.c_char * size).from_address(self._host_hidden),
            dtype=np.float16).copy().astype(np.float32)
        for i in range(tp):
            partial_arr = np.frombuffer(
                (ct.c_char * size).from_address(self._host_partials[i]),
                dtype=np.float16)
            hidden_arr += partial_arr.astype(np.float32)
        result = hidden_arr.astype(np.float16)

        # Copy result to pinned buffer
        ct.memmove(self._host_result, result.ctypes.data, size)

        # Async H2D result to all GPUs
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.memcpy_h2d_async(
                hidden_ptrs[i], self._host_result, size, self._streams[i])

        # Sync all streams once
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

    def allreduce_sum(self, partial_ptrs: list, num_elems: int):
        """Pinned-async allreduce sum only (no hidden residual)."""
        import numpy as np
        import ctypes as ct
        tp = self._tp_size
        size = num_elems * 2
        hip = self._hip

        # Async D2H all partials
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.memcpy_d2h_async(
                self._host_partials[i], partial_ptrs[i], size, self._streams[i])
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

        # CPU FP32 accumulate
        accum = np.frombuffer(
            (ct.c_char * size).from_address(self._host_partials[0]),
            dtype=np.float16).copy().astype(np.float32)
        for i in range(1, tp):
            accum += np.frombuffer(
                (ct.c_char * size).from_address(self._host_partials[i]),
                dtype=np.float16).astype(np.float32)
        result = accum.astype(np.float16)
        ct.memmove(self._host_result, result.ctypes.data, size)

        # Async H2D result to all GPUs
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.memcpy_h2d_async(
                partial_ptrs[i], self._host_result, size, self._streams[i])
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._streams[i])

    def cleanup(self):
        """Free pinned host buffers."""
        hip = self._hip
        if self._host_hidden:
            try:
                hip.host_free(self._host_hidden)
            except HIPError:
                pass
            self._host_hidden = 0
        if self._host_result:
            try:
                hip.host_free(self._host_result)
            except HIPError:
                pass
            self._host_result = 0
        for ptr in self._host_partials:
            try:
                hip.host_free(ptr)
            except HIPError:
                pass
        self._host_partials.clear()


class FusedP2PReduce:
    """Fused P2P allreduce for TP inference using gemv_p2p_reduce.hip.

    Key difference from P2PAllreduce:
    - Each GPU launches its OWN fused kernel that reads all other GPUs' partial
      results directly via P2P pointers (remote memory access in GPU kernel).
    - All GPUs compute the full sum simultaneously — no sequential gather+reduce+broadcast.
    - Eliminates the separate gather and broadcast phases (no intermediate buffers needed).

    For TP=4, hidden_size=5120 (10KB payload):
      Old P2PAllreduce path per allreduce:
        1. GPU0: 3x async P2P gathers (30KB transfers, sequential batching)
        2. GPU0: sync gather stream
        3. GPU0: reduce kernel (40KB reads + 10KB writes)
        4. GPU0: sync
        5. GPU0: 3x async P2P broadcasts (30KB transfers)
        6. All GPUs: sync broadcast streams
        Total: ~2 stream syncs + 6 P2P transfers + 1 kernel on 1 GPU

      New FusedP2PReduce path per allreduce:
        1. All 4 GPUs: sync (ensure GEMV is done on all GPUs)
        2. All 4 GPUs simultaneously: fused kernel (reads local + 3 remote partials,
           reads local hidden, writes updated hidden)
        3. All 4 GPUs: sync kernels
        Total: 1 sync + 4 simultaneous kernels (each reads 50KB, writes 10KB)

    On PCIe (no XGMI), remote GPU memory reads within a kernel use BAR1 aperture
    mapping. Latency per remote read may be higher than hipMemcpyPeerAsync for large
    transfers, but for 10KB payloads the fixed latency dominates over bandwidth.

    Usage:
        fpr = FusedP2PReduce(hip, device_ids, hidden_size, streams)
        fpr.allreduce_residual(partial_ptrs, hidden_ptrs, num_elems)
        fpr.allreduce_residual_async(partial_ptrs, hidden_ptrs, num_elems, compute_streams)
        fpr.cleanup()
    """

    def __init__(self, hip: HIPRuntime, device_ids: list, hidden_size: int,
                 streams: Optional[list] = None):
        """Initialize fused P2P reduce.

        Args:
            hip: HIPRuntime instance
            device_ids: list of GPU device IDs
            hidden_size: number of FP16 elements per allreduce
            streams: optional per-GPU compute streams
        """
        self._hip = hip
        self._device_ids = list(device_ids)
        self._tp_size = len(device_ids)
        self._hidden_size = hidden_size
        self._streams = streams if streams else [0] * len(device_ids)

        # Load kernel shared library
        self._lib = None
        self._load_lib()

        # Dedicated allreduce streams (non-blocking) for async allreduce overlap
        self._allreduce_streams = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._allreduce_streams.append(hip.stream_create_nonblocking())

        # Compute completion events: recorded after GEMV, waited on by allreduce stream
        self._compute_events = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._compute_events.append(hip.event_create())

        # Allreduce completion events: recorded after fused kernel, waited on by compute
        self._ar_done_events = []
        for dev_id in self._device_ids:
            hip.set_device(dev_id)
            self._ar_done_events.append(hip.event_create())

        # Enable P2P access between all pairs
        self._enable_p2p()

    def _enable_p2p(self):
        """Enable P2P access between all GPU pairs."""
        hip = self._hip
        for i, dev_i in enumerate(self._device_ids):
            hip.set_device(dev_i)
            for j, dev_j in enumerate(self._device_ids):
                if i == j:
                    continue
                if hip.device_can_access_peer(dev_i, dev_j):
                    try:
                        hip.device_enable_peer_access(dev_j)
                    except HIPError:
                        pass  # Already enabled

    def _load_lib(self):
        """Build and load the gemv_p2p_reduce.hip kernel as shared library."""
        src_dir = Path(__file__).parent.parent / "kernels"
        hip_path = src_dir / "gemv_p2p_reduce.hip"
        build_dir = Path(__file__).parent.parent.parent / "build" / "kernels"
        so_path = build_dir / "gemv_p2p_reduce.so"

        if not hip_path.exists():
            raise FileNotFoundError(f"Kernel not found: {hip_path}")

        # Build if missing or stale
        build_dir.mkdir(parents=True, exist_ok=True)
        if not so_path.exists() or (
                os.path.getmtime(hip_path) > os.path.getmtime(so_path)):
            try:
                subprocess.check_call([
                    "/opt/rocm/bin/hipcc",
                    "-O3", "--offload-arch=gfx906", "-std=c++17",
                    "-shared", "-fPIC",
                    "-o", str(so_path),
                    str(hip_path),
                ])
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(f"Failed to build gemv_p2p_reduce.hip: {e}")

        lib = ctypes.CDLL(str(so_path))

        # fused_p2p_reduce_residual_tp2(hidden, partial_local, partial_peer0, n, stream)
        lib.fused_p2p_reduce_residual_tp2.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.fused_p2p_reduce_residual_tp2.restype = ctypes.c_int

        # fused_p2p_reduce_residual_tp4(hidden, partial_local, peer0, peer1, peer2, n, stream)
        lib.fused_p2p_reduce_residual_tp4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.fused_p2p_reduce_residual_tp4.restype = ctypes.c_int

        # fused_p2p_reduce_only_tp4(partial_local, peer0, peer1, peer2, n, stream)
        lib.fused_p2p_reduce_only_tp4.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_void_p
        ]
        lib.fused_p2p_reduce_only_tp4.restype = ctypes.c_int

        self._lib = lib

    def allreduce_residual(self, partial_ptrs: list, hidden_ptrs: list,
                           num_elems: int):
        """Fused P2P allreduce + residual add (synchronous).

        Each GPU launches its own fused kernel that reads all other GPUs'
        partial results via P2P pointers and updates its own hidden buffer.
        All 4 kernels run simultaneously on their respective GPUs.

        Args:
            partial_ptrs: per-GPU GEMV partial result pointers
            hidden_ptrs:  per-GPU hidden state pointers (read-modify-write)
            num_elems:    number of FP16 elements
        """
        tp = self._tp_size
        if tp == 1:
            return

        hip = self._hip

        # Sync all GPUs to ensure GEMV kernels have completed
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.synchronize()

        # Launch fused kernel on each GPU simultaneously
        # Each GPU reads its own partial + peer partials via P2P pointers
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            ar_stream = self._allreduce_streams[i]
            ar_stream_ptr = ctypes.c_void_p(ar_stream)

            if tp == 2:
                # peer index for GPU i = 1 - i
                peer_i = 1 - i
                err = self._lib.fused_p2p_reduce_residual_tp2(
                    ctypes.c_void_p(hidden_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[peer_i]),
                    ctypes.c_uint(num_elems),
                    ar_stream_ptr)
            elif tp == 4:
                # For GPU i, the 3 peer indices are all j != i
                peer_indices = [j for j in range(tp) if j != i]
                err = self._lib.fused_p2p_reduce_residual_tp4(
                    ctypes.c_void_p(hidden_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[0]]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[1]]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[2]]),
                    ctypes.c_uint(num_elems),
                    ar_stream_ptr)
            else:
                raise ValueError(f"tp_size={tp} not supported (2 or 4 only)")

            if err != 0:
                raise HIPError(
                    f"fused_p2p_reduce_residual_tp{tp} kernel failed on GPU {i}: "
                    f"HIP error {err}")

        # Sync all allreduce streams
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._allreduce_streams[i])

    def allreduce_residual_async(self, partial_ptrs: list, hidden_ptrs: list,
                                  num_elems: int, compute_streams: list):
        """Fused P2P allreduce + residual add (asynchronous, stream-overlapping).

        Records compute events → each GPU's allreduce stream waits on ALL compute
        events (since we need all GPUs to finish GEMV before reading remote partials)
        → launches fused kernel simultaneously on all GPUs → records completion events.

        The Python call returns immediately; GPU work continues asynchronously.
        Next-layer's first kernel must call wait_for_allreduce_on_compute_stream()
        before reading d_hidden.

        Args:
            partial_ptrs: per-GPU GEMV partial result pointers
            hidden_ptrs:  per-GPU hidden state pointers
            num_elems:    number of FP16 elements
            compute_streams: per-GPU compute stream handles (for event recording)
        """
        tp = self._tp_size
        if tp == 1:
            return

        hip = self._hip

        # Step 1: Record compute completion events on each GPU's compute stream
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            hip.event_record(self._compute_events[i], compute_streams[i])

        # Step 2: Each GPU's allreduce stream waits for ALL compute events
        # (needed because kernel reads remote GPU data — all GEMVs must complete)
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            ar_stream = self._allreduce_streams[i]
            for j in range(tp):
                hip.stream_wait_event(ar_stream, self._compute_events[j])

        # Step 3: Launch fused kernel on each GPU (all on their allreduce streams)
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            ar_stream = self._allreduce_streams[i]
            ar_stream_ptr = ctypes.c_void_p(ar_stream)

            if tp == 2:
                peer_i = 1 - i
                err = self._lib.fused_p2p_reduce_residual_tp2(
                    ctypes.c_void_p(hidden_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[peer_i]),
                    ctypes.c_uint(num_elems),
                    ar_stream_ptr)
            elif tp == 4:
                peer_indices = [j for j in range(tp) if j != i]
                err = self._lib.fused_p2p_reduce_residual_tp4(
                    ctypes.c_void_p(hidden_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[i]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[0]]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[1]]),
                    ctypes.c_void_p(partial_ptrs[peer_indices[2]]),
                    ctypes.c_uint(num_elems),
                    ar_stream_ptr)
            else:
                raise ValueError(f"tp_size={tp} not supported (2 or 4 only)")

            if err != 0:
                raise HIPError(
                    f"fused_p2p_reduce_residual_tp{tp} failed on GPU {i}: "
                    f"HIP error {err}")

        # Step 4: Record allreduce completion events on each GPU's allreduce stream
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            hip.event_record(self._ar_done_events[i], self._allreduce_streams[i])

        # Python returns immediately — GPU work continues asynchronously

    def wait_for_allreduce_on_compute_stream(self, compute_streams: list):
        """Make each GPU's compute stream wait for allreduce completion (GPU-side).

        Same interface as P2PAllreduce.wait_for_allreduce_on_compute_stream.
        """
        hip = self._hip
        for i, dev_id in enumerate(self._device_ids):
            hip.set_device(dev_id)
            hip.stream_wait_event(compute_streams[i], self._ar_done_events[i])

    def allreduce_sum(self, partial_ptrs: list, num_elems: int):
        """Fused P2P allreduce sum only (no hidden residual add).

        Each GPU computes the full sum into its own partial buffer.
        Only supported for TP=4 currently.
        """
        tp = self._tp_size
        if tp == 1:
            return
        if tp != 4:
            raise ValueError(f"allreduce_sum only supported for TP=4, got TP={tp}")

        hip = self._hip

        # Sync all GPUs
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.synchronize()

        # Launch fused reduce-only kernel on each GPU
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            ar_stream = self._allreduce_streams[i]
            peer_indices = [j for j in range(tp) if j != i]
            err = self._lib.fused_p2p_reduce_only_tp4(
                ctypes.c_void_p(partial_ptrs[i]),
                ctypes.c_void_p(partial_ptrs[peer_indices[0]]),
                ctypes.c_void_p(partial_ptrs[peer_indices[1]]),
                ctypes.c_void_p(partial_ptrs[peer_indices[2]]),
                ctypes.c_uint(num_elems),
                ctypes.c_void_p(ar_stream))
            if err != 0:
                raise HIPError(
                    f"fused_p2p_reduce_only_tp4 failed on GPU {i}: HIP error {err}")

        # Sync all allreduce streams
        for i in range(tp):
            hip.set_device(self._device_ids[i])
            hip.stream_synchronize(self._allreduce_streams[i])

    def cleanup(self):
        """Destroy streams and events."""
        hip = self._hip
        for i, dev_id in enumerate(self._device_ids):
            try:
                hip.set_device(dev_id)
                if i < len(self._allreduce_streams):
                    hip.stream_destroy(self._allreduce_streams[i])
            except HIPError:
                pass
        self._allreduce_streams.clear()

        for i, dev_id in enumerate(self._device_ids):
            try:
                hip.set_device(dev_id)
                if i < len(self._compute_events):
                    hip.event_destroy(self._compute_events[i])
            except HIPError:
                pass
        self._compute_events.clear()

        for i, dev_id in enumerate(self._device_ids):
            try:
                hip.set_device(dev_id)
                if i < len(self._ar_done_events):
                    hip.event_destroy(self._ar_done_events[i])
            except HIPError:
                pass
        self._ar_done_events.clear()

        self._lib = None
