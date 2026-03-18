"""
mi50grad runtime: HIP-based kernel dispatch from Python via ctypes.

This module provides:
- GPU device management (init, select, query)
- Memory allocation/free on device
- HSACO module loading and kernel launching
- Host<->device memory transfers
"""

import ctypes
import ctypes.util
import os
import subprocess
from pathlib import Path
from typing import Optional


# ============================================================
# Pre-built launch specification for cached kernel dispatch
# ============================================================

class LaunchSpec:
    """Pre-built kernel launch specification for cached (low-overhead) dispatch.

    Holds a pre-built C array of void pointers to ctypes values. The ctypes
    values themselves are stored in `params` and remain alive as long as the
    LaunchSpec exists. Mutable values (e.g., position-dependent pointers) can
    be updated in-place:

        spec.params[2].value = new_ptr  # update ctypes value in-place
        hip.launch_spec(spec)           # params_array pointer automatically updated

    This avoids re-allocating Python ctypes objects and the params_array on
    every kernel launch, reducing per-launch overhead from ~10μs to ~2μs.

    Usage:
        from src.runtime.hip_dispatch import LaunchSpec
        spec = LaunchSpec(
            func=kernel_func_handle,
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            params=[ctypes.c_uint64(ptr0), ctypes.c_uint32(K), ...],
            shared_mem=0,
        )
        # Later: update mutable param in-place
        spec.params[0].value = new_ptr
        # Launch using pre-built params array
        hip_runtime.launch_spec(spec)
    """

    def __init__(self, func: int, grid: tuple, block: tuple,
                 params: list, shared_mem: int = 0):
        """
        Args:
            func: HIP kernel function handle (int)
            grid: (gx, gy, gz) grid dimensions
            block: (bx, by, bz) block dimensions
            params: list of ctypes objects (c_uint64, c_uint32, c_float, etc.)
            shared_mem: shared memory bytes per block
        """
        self.func = func
        self.grid = grid
        self.block = block
        self.params = params  # Keep alive: params_array holds pointers into these
        self.shared_mem = shared_mem

        # Pre-build the C array of void* pointers once
        n = len(params)
        self.params_array = (ctypes.c_void_p * n)()
        for i, p in enumerate(params):
            self.params_array[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)


# ============================================================
# HIP library binding via ctypes
# ============================================================

def _find_hip_library():
    """Find libamdhip64.so on the system."""
    search_paths = [
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/hip/lib/libamdhip64.so",
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    # Try system search
    found = ctypes.util.find_library("amdhip64")
    if found:
        return found
    raise RuntimeError("Cannot find libamdhip64.so. Is ROCm installed?")


class HIPError(Exception):
    """HIP runtime error."""
    pass


class HIPRuntime:
    """Low-level HIP runtime wrapper via ctypes."""

    def __init__(self):
        self._lib = ctypes.CDLL(_find_hip_library())
        self._setup_functions()

    def _setup_functions(self):
        lib = self._lib

        # hipInit
        lib.hipInit.argtypes = [ctypes.c_uint]
        lib.hipInit.restype = ctypes.c_int

        # hipGetDeviceCount
        lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDeviceCount.restype = ctypes.c_int

        # hipSetDevice
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int

        # hipGetDevice
        lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDevice.restype = ctypes.c_int

        # hipMalloc
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.hipMalloc.restype = ctypes.c_int

        # hipFree
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipFree.restype = ctypes.c_int

        # hipMemcpy
        lib.hipMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        lib.hipMemcpy.restype = ctypes.c_int

        # hipMemset
        lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.hipMemset.restype = ctypes.c_int

        # hipDeviceSynchronize
        lib.hipDeviceSynchronize.argtypes = []
        lib.hipDeviceSynchronize.restype = ctypes.c_int

        # hipModuleLoad
        lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.hipModuleLoad.restype = ctypes.c_int

        # hipModuleUnload
        lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        lib.hipModuleUnload.restype = ctypes.c_int

        # hipModuleGetFunction
        lib.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        lib.hipModuleGetFunction.restype = ctypes.c_int

        # hipModuleLaunchKernel
        lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,                        # function
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block
            ctypes.c_uint,                          # shared_mem
            ctypes.c_void_p,                        # stream
            ctypes.POINTER(ctypes.c_void_p),        # kernelParams
            ctypes.POINTER(ctypes.c_void_p),        # extra
        ]
        lib.hipModuleLaunchKernel.restype = ctypes.c_int

        # --- Peer-to-peer ---

        # hipDeviceCanAccessPeer
        lib.hipDeviceCanAccessPeer.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]
        lib.hipDeviceCanAccessPeer.restype = ctypes.c_int

        # hipDeviceEnablePeerAccess
        lib.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
        lib.hipDeviceEnablePeerAccess.restype = ctypes.c_int

        # hipMemcpyPeerAsync
        lib.hipMemcpyPeerAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_int,       # dst, dst_device
            ctypes.c_void_p, ctypes.c_int,       # src, src_device
            ctypes.c_size_t, ctypes.c_void_p     # size, stream
        ]
        lib.hipMemcpyPeerAsync.restype = ctypes.c_int

        # --- Events ---

        # hipEventCreate
        lib.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.hipEventCreate.restype = ctypes.c_int

        # hipEventRecord
        lib.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.hipEventRecord.restype = ctypes.c_int

        # hipEventSynchronize
        lib.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.hipEventSynchronize.restype = ctypes.c_int

        # hipEventElapsedTime
        lib.hipEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.hipEventElapsedTime.restype = ctypes.c_int

        # hipEventDestroy
        lib.hipEventDestroy.argtypes = [ctypes.c_void_p]
        lib.hipEventDestroy.restype = ctypes.c_int

        # --- Streams ---

        # hipStreamCreate
        lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.hipStreamCreate.restype = ctypes.c_int

        # hipStreamCreateWithFlags
        lib.hipStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint
        ]
        lib.hipStreamCreateWithFlags.restype = ctypes.c_int

        # hipStreamSynchronize
        lib.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.hipStreamSynchronize.restype = ctypes.c_int

        # hipGetLastError
        lib.hipGetLastError.argtypes = []
        lib.hipGetLastError.restype = ctypes.c_int

        # hipStreamDestroy
        lib.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.hipStreamDestroy.restype = ctypes.c_int

        # hipStreamWaitEvent
        lib.hipStreamWaitEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint  # stream, event, flags
        ]
        lib.hipStreamWaitEvent.restype = ctypes.c_int

        # --- Pinned host memory ---

        # hipHostMalloc
        lib.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
        ]
        lib.hipHostMalloc.restype = ctypes.c_int

        # hipHostFree
        lib.hipHostFree.argtypes = [ctypes.c_void_p]
        lib.hipHostFree.restype = ctypes.c_int

        # hipMemcpyAsync (async host<->device transfer on stream)
        lib.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p  # dst, src, size, kind, stream
        ]
        lib.hipMemcpyAsync.restype = ctypes.c_int

    def _check(self, err: int, msg: str = ""):
        if err != 0:
            raise HIPError(f"HIP error {err}: {msg}")

    def init(self):
        self._check(self._lib.hipInit(0), "hipInit")

    def device_count(self) -> int:
        count = ctypes.c_int(0)
        self._check(self._lib.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
        return count.value

    def set_device(self, device_id: int):
        self._check(self._lib.hipSetDevice(device_id), f"hipSetDevice({device_id})")

    def get_device(self) -> int:
        dev = ctypes.c_int(0)
        self._check(self._lib.hipGetDevice(ctypes.byref(dev)), "hipGetDevice")
        return dev.value

    def malloc(self, size: int) -> int:
        ptr = ctypes.c_void_p(0)
        self._check(self._lib.hipMalloc(ctypes.byref(ptr), size), f"hipMalloc({size})")
        return ptr.value

    def free(self, ptr: int):
        self._check(self._lib.hipFree(ctypes.c_void_p(ptr)), "hipFree")

    def memcpy_h2d(self, dst: int, src: bytes, size: int):
        """Host to device copy."""
        self._check(
            self._lib.hipMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_char_p(src) if isinstance(src, bytes) else src,
                size, 1  # hipMemcpyHostToDevice = 1
            ),
            "hipMemcpy H2D"
        )

    def memcpy_d2h(self, dst, src: int, size: int):
        """Device to host copy."""
        self._check(
            self._lib.hipMemcpy(dst, ctypes.c_void_p(src), size, 2),  # D2H = 2
            "hipMemcpy D2H"
        )

    def memcpy_d2d(self, dst: int, src: int, size: int):
        """Device to device copy."""
        self._check(
            self._lib.hipMemcpy(
                ctypes.c_void_p(dst), ctypes.c_void_p(src), size, 3  # D2D = 3
            ),
            "hipMemcpy D2D"
        )

    def host_malloc(self, size: int) -> int:
        """Allocate pinned host memory. Returns host pointer."""
        ptr = ctypes.c_void_p(0)
        # hipHostMallocDefault = 0
        self._check(self._lib.hipHostMalloc(ctypes.byref(ptr), size, 0),
                     f"hipHostMalloc({size})")
        return ptr.value

    def host_free(self, ptr: int):
        """Free pinned host memory."""
        self._check(self._lib.hipHostFree(ctypes.c_void_p(ptr)), "hipHostFree")

    def memcpy_h2d_async(self, dst: int, src: int, size: int, stream: int = 0):
        """Async host to device copy on stream (pinned host memory)."""
        self._check(
            self._lib.hipMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                size, 1, ctypes.c_void_p(stream)  # hipMemcpyHostToDevice = 1
            ),
            "hipMemcpyAsync H2D"
        )

    def memcpy_d2h_async(self, dst: int, src: int, size: int, stream: int = 0):
        """Async device to host copy on stream (pinned host memory)."""
        self._check(
            self._lib.hipMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                size, 2, ctypes.c_void_p(stream)  # hipMemcpyDeviceToHost = 2
            ),
            "hipMemcpyAsync D2H"
        )

    def memcpy_d2d_async(self, dst: int, src: int, size: int, stream: int = 0):
        """Async device to device copy on stream."""
        self._check(
            self._lib.hipMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                size, 1, ctypes.c_void_p(stream)  # hipMemcpyDeviceToDevice = 1
            ),
            "hipMemcpyAsync D2D"
        )

    def memset(self, ptr: int, value: int, size: int):
        self._check(
            self._lib.hipMemset(ctypes.c_void_p(ptr), value, size),
            "hipMemset"
        )

    def synchronize(self):
        self._check(self._lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def module_load(self, hsaco_path: str) -> int:
        """Load an HSACO code object. Returns module handle."""
        module = ctypes.c_void_p(0)
        self._check(
            self._lib.hipModuleLoad(
                ctypes.byref(module),
                hsaco_path.encode('utf-8')
            ),
            f"hipModuleLoad({hsaco_path})"
        )
        return module.value

    def module_unload(self, module: int):
        self._check(self._lib.hipModuleUnload(ctypes.c_void_p(module)), "hipModuleUnload")

    def module_get_function(self, module: int, name: str) -> int:
        """Get kernel function from module. Returns function handle."""
        func = ctypes.c_void_p(0)
        self._check(
            self._lib.hipModuleGetFunction(
                ctypes.byref(func),
                ctypes.c_void_p(module),
                name.encode('utf-8')
            ),
            f"hipModuleGetFunction({name})"
        )
        return func.value

    def launch_kernel(self, func: int,
                      grid: tuple, block: tuple,
                      kernel_params: list,
                      shared_mem: int = 0, stream: int = 0):
        """Launch kernel with kernelParams (list of ctypes values).

        kernel_params: list of ctypes objects (e.g., ctypes.c_uint32(42)).
        Each element's address is passed to hipModuleLaunchKernel.
        """
        n = len(kernel_params)
        params_array = (ctypes.c_void_p * n)()
        for i, p in enumerate(kernel_params):
            params_array[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

        gx, gy, gz = grid
        bx, by, bz = block

        self._check(
            self._lib.hipModuleLaunchKernel(
                ctypes.c_void_p(func),
                gx, gy, gz, bx, by, bz,
                shared_mem,
                ctypes.c_void_p(stream),
                ctypes.cast(params_array, ctypes.POINTER(ctypes.c_void_p)),
                None,
            ),
            "hipModuleLaunchKernel"
        )

    def launch_spec(self, spec: 'LaunchSpec', stream: int = 0):
        """Launch a pre-built LaunchSpec (cached parameter array).

        This avoids re-allocating Python ctypes objects on each launch,
        reducing per-launch Python overhead significantly.

        The LaunchSpec holds a pre-built C array of void pointers to ctypes
        values. Mutable ctypes values (e.g., position-dependent pointers) can
        be updated in-place via spec.update_value(idx, new_val) before calling
        this method — the pre-built pointer array automatically reflects updates.
        """
        gx, gy, gz = spec.grid
        bx, by, bz = spec.block
        self._check(
            self._lib.hipModuleLaunchKernel(
                ctypes.c_void_p(spec.func),
                gx, gy, gz, bx, by, bz,
                spec.shared_mem,
                ctypes.c_void_p(stream),
                ctypes.cast(spec.params_array, ctypes.POINTER(ctypes.c_void_p)),
                None,
            ),
            "hipModuleLaunchKernel (cached)"
        )

    # --- Peer-to-peer methods ---

    def device_can_access_peer(self, device_id: int, peer_device_id: int) -> bool:
        """Check if device_id can access memory on peer_device_id."""
        can_access = ctypes.c_int(0)
        self._check(
            self._lib.hipDeviceCanAccessPeer(
                ctypes.byref(can_access), device_id, peer_device_id
            ),
            f"hipDeviceCanAccessPeer({device_id}, {peer_device_id})"
        )
        return can_access.value != 0

    def device_enable_peer_access(self, peer_device_id: int):
        """Enable peer access from the current device to peer_device_id."""
        self._check(
            self._lib.hipDeviceEnablePeerAccess(peer_device_id, 0),
            f"hipDeviceEnablePeerAccess({peer_device_id})"
        )

    def memcpy_peer_async(self, dst: int, dst_device: int,
                          src: int, src_device: int,
                          size: int, stream: int = 0):
        """Async peer-to-peer memory copy between devices."""
        self._check(
            self._lib.hipMemcpyPeerAsync(
                ctypes.c_void_p(dst), dst_device,
                ctypes.c_void_p(src), src_device,
                size, ctypes.c_void_p(stream)
            ),
            "hipMemcpyPeerAsync"
        )

    # --- Event methods ---

    def event_create(self) -> int:
        """Create a HIP event. Returns event handle."""
        event = ctypes.c_void_p(0)
        self._check(
            self._lib.hipEventCreate(ctypes.byref(event)),
            "hipEventCreate"
        )
        return event.value

    def event_record(self, event: int, stream: int = 0):
        """Record an event on a stream."""
        self._check(
            self._lib.hipEventRecord(
                ctypes.c_void_p(event), ctypes.c_void_p(stream)
            ),
            "hipEventRecord"
        )

    def event_synchronize(self, event: int):
        """Wait for an event to complete."""
        self._check(
            self._lib.hipEventSynchronize(ctypes.c_void_p(event)),
            "hipEventSynchronize"
        )

    def event_elapsed_time(self, start: int, end: int) -> float:
        """Get elapsed time in milliseconds between two events."""
        ms = ctypes.c_float(0.0)
        self._check(
            self._lib.hipEventElapsedTime(
                ctypes.byref(ms),
                ctypes.c_void_p(start), ctypes.c_void_p(end)
            ),
            "hipEventElapsedTime"
        )
        return ms.value

    def event_destroy(self, event: int):
        """Destroy a HIP event."""
        self._check(
            self._lib.hipEventDestroy(ctypes.c_void_p(event)),
            "hipEventDestroy"
        )

    # --- Stream methods ---

    def stream_create(self) -> int:
        """Create a HIP stream (blocking — synchronizes with null stream). Returns stream handle."""
        stream = ctypes.c_void_p(0)
        self._check(
            self._lib.hipStreamCreate(ctypes.byref(stream)),
            "hipStreamCreate"
        )
        return stream.value

    def stream_create_nonblocking(self) -> int:
        """Create a non-blocking HIP stream (does NOT synchronize with null stream).

        Use for streams that should run independently of the default stream.
        hipStreamNonBlocking = 1. This avoids the null stream's implicit
        serialization, enabling true overlap between compute (null stream)
        and allreduce (non-blocking stream).
        """
        stream = ctypes.c_void_p(0)
        HIP_STREAM_NON_BLOCKING = 1
        self._check(
            self._lib.hipStreamCreateWithFlags(
                ctypes.byref(stream), ctypes.c_uint(HIP_STREAM_NON_BLOCKING)
            ),
            "hipStreamCreateWithFlags(NonBlocking)"
        )
        return stream.value

    def stream_synchronize(self, stream: int):
        """Wait for all operations on a stream to complete."""
        self._check(
            self._lib.hipStreamSynchronize(ctypes.c_void_p(stream)),
            "hipStreamSynchronize"
        )

    def stream_destroy(self, stream: int):
        """Destroy a HIP stream."""
        self._check(
            self._lib.hipStreamDestroy(ctypes.c_void_p(stream)),
            "hipStreamDestroy"
        )

    def stream_wait_event(self, stream: int, event: int):
        """Make a stream wait for an event to complete (GPU-side wait, non-blocking on CPU)."""
        self._check(
            self._lib.hipStreamWaitEvent(
                ctypes.c_void_p(stream), ctypes.c_void_p(event), ctypes.c_uint(0)
            ),
            "hipStreamWaitEvent"
        )


# ============================================================
# High-level device abstraction
# ============================================================

class GPUDevice:
    """High-level GPU device wrapper."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.hip = HIPRuntime()
        self.hip.init()
        self.hip.set_device(device_id)
        self._modules = {}  # path -> module handle
        self._allocations = []  # track for cleanup

    def _ensure_device(self):
        """Set current HIP device to this device (for multi-GPU safety)."""
        self.hip.set_device(self.device_id)

    def malloc(self, size: int) -> int:
        self._ensure_device()
        ptr = self.hip.malloc(size)
        self._allocations.append(ptr)
        return ptr

    def free(self, ptr: int):
        self._ensure_device()
        self.hip.free(ptr)
        if ptr in self._allocations:
            self._allocations.remove(ptr)

    def upload(self, device_ptr: int, host_data: bytes):
        self._ensure_device()
        self.hip.memcpy_h2d(device_ptr, host_data, len(host_data))

    def download(self, device_ptr: int, size: int) -> bytes:
        self._ensure_device()
        buf = ctypes.create_string_buffer(size)
        self.hip.memcpy_d2h(buf, device_ptr, size)
        return buf.raw

    def memset(self, ptr: int, value: int, size: int):
        self._ensure_device()
        self.hip.memset(ptr, value, size)

    def memcpy_d2d(self, dst: int, src: int, size: int):
        self._ensure_device()
        self.hip.memcpy_d2d(dst, src, size)

    def memcpy_d2d_async(self, dst: int, src: int, size: int, stream: int = 0):
        self._ensure_device()
        self.hip.memcpy_d2d_async(dst, src, size, stream)

    def synchronize(self):
        self._ensure_device()
        self.hip.synchronize()

    def load_hsaco(self, path: str) -> int:
        """Load HSACO and cache the module."""
        self._ensure_device()
        if path not in self._modules:
            self._modules[path] = self.hip.module_load(path)
        return self._modules[path]

    def get_kernel(self, module: int, name: str) -> int:
        self._ensure_device()
        return self.hip.module_get_function(module, name)

    def create_stream(self) -> int:
        """Create a HIP stream on this device."""
        self.hip.set_device(self.device_id)
        return self.hip.stream_create()

    def create_stream_nonblocking(self) -> int:
        """Create a non-blocking HIP stream on this device (no null stream sync)."""
        self.hip.set_device(self.device_id)
        return self.hip.stream_create_nonblocking()

    def create_event(self) -> int:
        """Create a HIP event on this device."""
        self.hip.set_device(self.device_id)
        return self.hip.event_create()

    def launch(self, func: int, grid: tuple, block: tuple,
               kernel_params: list, shared_mem: int = 0,
               stream: int = 0):
        """Launch kernel. kernel_params is a list of ctypes values."""
        self.hip.set_device(self.device_id)
        self.hip.launch_kernel(func, grid, block, kernel_params,
                               shared_mem, stream)

    def launch_cached(self, spec: 'LaunchSpec', stream: int = 0):
        """Launch a pre-built LaunchSpec (cached parameter array).

        Lower overhead than launch() since no Python ctypes objects are
        created. The caller must update spec.params[i].value in-place for
        any parameters that change between calls (e.g., position pointers).
        """
        self.hip.set_device(self.device_id)
        self.hip.launch_spec(spec, stream)

    def cleanup(self):
        for ptr in self._allocations:
            try:
                self.hip.free(ptr)
            except HIPError:
                pass
        self._allocations.clear()
        for mod in self._modules.values():
            try:
                self.hip.module_unload(mod)
            except HIPError:
                pass
        self._modules.clear()
