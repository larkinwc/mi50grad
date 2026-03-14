"""
Tensor parallelism for multi-GPU inference on MI50.

Manages a group of GPUs for parallel execution, providing collective
operations (all-reduce, scatter, gather, broadcast) and weight sharding.

For Qwen 3.5 27B with hidden_dim=4096, the all-reduce payload is 8KB (FP16),
so we use simple reduce-broadcast rather than ring allreduce -- lower latency
for small messages over PCIe.
"""

import ctypes
import numpy as np
from typing import Optional

from .hip_dispatch import GPUDevice, HIPRuntime, HIPError


# Threshold below which we use reduce-broadcast instead of ring allreduce.
# 32KB covers hidden_dim=4096 in FP16 (8KB) with margin.
SMALL_MESSAGE_THRESHOLD = 32 * 1024


class TensorParallelGroup:
    """Manages a group of GPUs for tensor parallelism."""

    def __init__(self, device_ids: list[int]):
        """Initialize TP group with given GPU devices.

        Creates a GPUDevice per device, enables peer access between all pairs,
        and creates one stream per device for async operations.
        """
        if len(device_ids) < 2:
            raise ValueError("Tensor parallelism requires at least 2 devices")

        self._device_ids = list(device_ids)
        self._devices: list[GPUDevice] = []
        self._streams: list[int] = []
        self._hip = HIPRuntime()
        self._hip.init()

        # Create GPUDevice for each device
        for dev_id in self._device_ids:
            device = GPUDevice(dev_id)
            self._devices.append(device)

        # Enable peer access between all pairs
        for i, dev_i in enumerate(self._device_ids):
            self._hip.set_device(dev_i)
            for j, dev_j in enumerate(self._device_ids):
                if i == j:
                    continue
                if self._hip.device_can_access_peer(dev_i, dev_j):
                    try:
                        self._hip.device_enable_peer_access(dev_j)
                    except HIPError:
                        # Already enabled -- not an error
                        pass

        # Create one stream per device for async collective ops
        for dev in self._devices:
            self._streams.append(dev.create_stream())

    @property
    def world_size(self) -> int:
        return len(self._device_ids)

    @property
    def devices(self) -> list[GPUDevice]:
        return self._devices

    @property
    def streams(self) -> list[int]:
        return self._streams

    def all_reduce_sum(self, ptrs: list[int], size: int):
        """In-place all-reduce sum across all devices.

        ptrs[i] is a device pointer on device i, all with the same byte size.

        For small messages (<32KB, typical for 4096-dim hidden states in FP16):
            Simple reduce-broadcast -- copy everything to device 0, accumulate,
            then broadcast back. Lower latency than ring for small payloads.

        For larger messages:
            Ring allreduce with scatter-reduce + all-gather phases.
        """
        if len(ptrs) != self.world_size:
            raise ValueError(
                f"Expected {self.world_size} pointers, got {len(ptrs)}")

        if size <= SMALL_MESSAGE_THRESHOLD:
            self._all_reduce_sum_small(ptrs, size)
        else:
            self._all_reduce_sum_ring(ptrs, size)

    def _all_reduce_sum_small(self, ptrs: list[int], size: int):
        """Reduce-broadcast allreduce for small messages.

        1. Copy all remote buffers to device 0 temp buffer
        2. Accumulate on device 0 (host-side FP16 add -- size is tiny)
        3. Broadcast result back to all devices
        """
        n = self.world_size
        dev0 = self._devices[0]
        dev0_id = self._device_ids[0]
        stream0 = self._streams[0]

        # Allocate temp buffer on device 0 for receiving remote data
        self._hip.set_device(dev0_id)
        tmp_ptr = self._hip.malloc(size)

        try:
            # Start with device 0's data -- download to host
            host_accum = ctypes.create_string_buffer(size)
            self._hip.set_device(dev0_id)
            self._hip.memcpy_d2h(host_accum, ptrs[0], size)

            accum = np.frombuffer(host_accum, dtype=np.float16).copy()

            # Copy each remote buffer to device 0 temp, then download and add
            for i in range(1, n):
                src_dev_id = self._device_ids[i]
                self._hip.memcpy_peer_async(
                    tmp_ptr, dev0_id, ptrs[i], src_dev_id, size, stream0)
                self._hip.stream_synchronize(stream0)

                host_buf = ctypes.create_string_buffer(size)
                self._hip.set_device(dev0_id)
                self._hip.memcpy_d2h(host_buf, tmp_ptr, size)
                remote = np.frombuffer(host_buf, dtype=np.float16)
                accum += remote

            # Upload result to device 0
            result_bytes = accum.tobytes()
            self._hip.set_device(dev0_id)
            self._hip.memcpy_h2d(ptrs[0], result_bytes, size)

            # Broadcast result to all other devices
            for i in range(1, n):
                dst_dev_id = self._device_ids[i]
                self._hip.memcpy_peer_async(
                    ptrs[i], dst_dev_id, ptrs[0], dev0_id, size,
                    self._streams[i])

            # Synchronize all streams
            for i in range(1, n):
                self._hip.set_device(self._device_ids[i])
                self._hip.stream_synchronize(self._streams[i])

        finally:
            self._hip.set_device(dev0_id)
            self._hip.free(tmp_ptr)

    def _all_reduce_sum_ring(self, ptrs: list[int], size: int):
        """Ring allreduce for larger messages.

        Phase 1 -- Scatter-reduce: divide each buffer into world_size chunks.
            In each of (world_size - 1) steps, GPU[i] sends chunk to GPU[(i+1)%n]
            which accumulates it into its own buffer.

        Phase 2 -- All-gather: each GPU now has one fully-reduced chunk.
            Rotate the reduced chunks around the ring so everyone gets all of them.
        """
        n = self.world_size
        num_elems = size // 2  # FP16 = 2 bytes per element
        chunk_elems = num_elems // n
        chunk_bytes = chunk_elems * 2
        # Handle remainder by giving extra to the last chunk
        last_chunk_elems = num_elems - chunk_elems * (n - 1)
        last_chunk_bytes = last_chunk_elems * 2

        def chunk_size(idx):
            return last_chunk_bytes if idx == n - 1 else chunk_bytes

        def chunk_offset(idx):
            return idx * chunk_bytes

        # Allocate one temp receive buffer per device
        tmp_ptrs = []
        for i, dev in enumerate(self._devices):
            self._hip.set_device(self._device_ids[i])
            tmp_ptrs.append(self._hip.malloc(max(chunk_bytes, last_chunk_bytes)))

        try:
            # Phase 1: scatter-reduce
            for step in range(n - 1):
                for i in range(n):
                    send_chunk_idx = (i - step) % n
                    recv_rank = (i + 1) % n
                    src_dev_id = self._device_ids[i]
                    dst_dev_id = self._device_ids[recv_rank]
                    c_off = chunk_offset(send_chunk_idx)
                    c_size = chunk_size(send_chunk_idx)

                    # Async copy chunk from GPU[i] to tmp on GPU[recv_rank]
                    self._hip.memcpy_peer_async(
                        tmp_ptrs[recv_rank], dst_dev_id,
                        ptrs[i] + c_off, src_dev_id,
                        c_size, self._streams[i])

                # Sync all transfers
                for i in range(n):
                    self._hip.set_device(self._device_ids[i])
                    self._hip.stream_synchronize(self._streams[i])

                # Each device accumulates the received chunk into its buffer
                for i in range(n):
                    recv_rank = (i + 1) % n
                    recv_chunk_idx = (i - step) % n
                    c_off = chunk_offset(recv_chunk_idx)
                    c_size = chunk_size(recv_chunk_idx)
                    c_elems = c_size // 2

                    # Download both, add on host, upload back
                    self._hip.set_device(self._device_ids[recv_rank])
                    host_local = ctypes.create_string_buffer(c_size)
                    host_remote = ctypes.create_string_buffer(c_size)
                    self._hip.memcpy_d2h(
                        host_local, ptrs[recv_rank] + c_off, c_size)
                    self._hip.memcpy_d2h(
                        host_remote, tmp_ptrs[recv_rank], c_size)

                    local_arr = np.frombuffer(host_local, dtype=np.float16).copy()
                    remote_arr = np.frombuffer(host_remote, dtype=np.float16)
                    local_arr += remote_arr

                    self._hip.memcpy_h2d(
                        ptrs[recv_rank] + c_off, local_arr.tobytes(), c_size)

            # Phase 2: all-gather
            for step in range(n - 1):
                for i in range(n):
                    send_chunk_idx = (i - step + 1) % n
                    recv_rank = (i + 1) % n
                    src_dev_id = self._device_ids[i]
                    dst_dev_id = self._device_ids[recv_rank]
                    c_off = chunk_offset(send_chunk_idx)
                    c_size = chunk_size(send_chunk_idx)

                    self._hip.memcpy_peer_async(
                        ptrs[recv_rank] + c_off, dst_dev_id,
                        ptrs[i] + c_off, src_dev_id,
                        c_size, self._streams[i])

                for i in range(n):
                    self._hip.set_device(self._device_ids[i])
                    self._hip.stream_synchronize(self._streams[i])

        finally:
            for i, tmp in enumerate(tmp_ptrs):
                self._hip.set_device(self._device_ids[i])
                self._hip.free(tmp)

    def scatter(self, src_ptr: int, src_device: int,
                dst_ptrs: list[int], chunk_size: int):
        """Scatter src into equal chunks across devices.

        src_ptr on src_device is split into world_size chunks of chunk_size bytes.
        dst_ptrs[i] receives the i-th chunk on device i.
        """
        src_dev_idx = self._device_ids.index(src_device)

        for i in range(self.world_size):
            dst_dev_id = self._device_ids[i]
            offset = i * chunk_size

            if dst_dev_id == src_device:
                # Same device -- use D2D copy
                self._hip.set_device(src_device)
                self._hip.memcpy_d2d(dst_ptrs[i], src_ptr + offset, chunk_size)
            else:
                self._hip.memcpy_peer_async(
                    dst_ptrs[i], dst_dev_id,
                    src_ptr + offset, src_device,
                    chunk_size, self._streams[i])

        # Sync all streams
        for i in range(self.world_size):
            if self._device_ids[i] != src_device:
                self._hip.set_device(self._device_ids[i])
                self._hip.stream_synchronize(self._streams[i])

    def gather(self, src_ptrs: list[int], chunk_size: int,
               dst_ptr: int, dst_device: int):
        """Gather chunks from all devices into dst on one device.

        src_ptrs[i] is a chunk_size buffer on device i.
        Result is concatenated into dst_ptr on dst_device.
        """
        for i in range(self.world_size):
            src_dev_id = self._device_ids[i]
            offset = i * chunk_size

            if src_dev_id == dst_device:
                self._hip.set_device(dst_device)
                self._hip.memcpy_d2d(dst_ptr + offset, src_ptrs[i], chunk_size)
            else:
                self._hip.memcpy_peer_async(
                    dst_ptr + offset, dst_device,
                    src_ptrs[i], src_dev_id,
                    chunk_size, self._streams[i])

        # Sync all streams
        for i in range(self.world_size):
            if self._device_ids[i] != dst_device:
                self._hip.set_device(self._device_ids[i])
                self._hip.stream_synchronize(self._streams[i])

    def broadcast(self, src_ptr: int, src_device: int,
                  dst_ptrs: list[int], size: int):
        """Broadcast from one device to all others.

        dst_ptrs[i] is a buffer on device i. The src_device entry in dst_ptrs
        is ignored (or can be the same as src_ptr for in-place).
        """
        for i in range(self.world_size):
            dst_dev_id = self._device_ids[i]
            if dst_dev_id == src_device:
                # If dst_ptrs[i] != src_ptr, do a local copy
                if dst_ptrs[i] != src_ptr:
                    self._hip.set_device(src_device)
                    self._hip.memcpy_d2d(dst_ptrs[i], src_ptr, size)
                continue

            self._hip.memcpy_peer_async(
                dst_ptrs[i], dst_dev_id,
                src_ptr, src_device,
                size, self._streams[i])

        for i in range(self.world_size):
            if self._device_ids[i] != src_device:
                self._hip.set_device(self._device_ids[i])
                self._hip.stream_synchronize(self._streams[i])

    def shard_weight(self, weight_data: bytes, dim: int,
                     shape: tuple) -> list[int]:
        """Shard a weight tensor along a given dimension across devices.

        Returns list of device pointers, one per device.

        dim=0: split rows (for row-parallel, e.g., FFN down, output proj)
        dim=1: split columns (for column-parallel, e.g., QKV, FFN up/gate)

        The weight_data is interpreted as a contiguous C-order array of the
        given shape. Element size is inferred from len(weight_data) / product(shape).
        """
        n = self.world_size
        total_elems = 1
        for s in shape:
            total_elems *= s
        elem_size = len(weight_data) // total_elems

        arr = np.frombuffer(weight_data, dtype=np.uint8).reshape(
            *shape, elem_size)

        if dim == 0:
            # Split along first dimension
            if shape[0] % n != 0:
                raise ValueError(
                    f"Cannot split dim 0 of size {shape[0]} into {n} shards")
            shard_size_0 = shape[0] // n
            shards = []
            for i in range(n):
                shard = arr[i * shard_size_0:(i + 1) * shard_size_0]
                shards.append(shard.tobytes())
        elif dim == 1:
            if len(shape) < 2:
                raise ValueError("Cannot split dim 1 of a 1D tensor")
            if shape[1] % n != 0:
                raise ValueError(
                    f"Cannot split dim 1 of size {shape[1]} into {n} shards")
            shard_size_1 = shape[1] // n
            shards = []
            for i in range(n):
                shard = arr[:, i * shard_size_1:(i + 1) * shard_size_1]
                # Need contiguous copy for upload
                shards.append(np.ascontiguousarray(shard).tobytes())
        else:
            raise ValueError(f"Unsupported shard dim: {dim}")

        # Upload each shard to its device
        ptrs = []
        for i, shard_bytes in enumerate(shards):
            dev = self._devices[i]
            dev.hip.set_device(dev.device_id)
            ptr = dev.malloc(len(shard_bytes))
            dev.upload(ptr, shard_bytes)
            ptrs.append(ptr)

        return ptrs

    def cleanup(self):
        """Destroy streams and clean up all devices."""
        for i, stream in enumerate(self._streams):
            try:
                self._hip.set_device(self._device_ids[i])
                self._hip.stream_destroy(stream)
            except HIPError:
                pass
        self._streams.clear()

        for dev in self._devices:
            dev.cleanup()
        self._devices.clear()
