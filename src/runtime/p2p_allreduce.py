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
                           num_elems: int):
        """P2P allreduce + residual add.

        Computes: hidden[i] = hidden[0] + partial[0] + partial[1] + ...
        and broadcasts result to all GPUs' hidden buffers.

        Args:
            partial_ptrs: list of device ptrs, partial_ptrs[i] on device i
            hidden_ptrs:  list of device ptrs, hidden_ptrs[i] on device i
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

        # Step 0: Synchronize all GPUs to ensure compute kernels have completed
        # and their partial results are ready for P2P gather.
        # Without this, the P2P copy might read stale/unfinished data.
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
