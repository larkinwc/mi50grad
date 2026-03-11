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

    def malloc(self, size: int) -> int:
        ptr = self.hip.malloc(size)
        self._allocations.append(ptr)
        return ptr

    def free(self, ptr: int):
        self.hip.free(ptr)
        if ptr in self._allocations:
            self._allocations.remove(ptr)

    def upload(self, device_ptr: int, host_data: bytes):
        self.hip.memcpy_h2d(device_ptr, host_data, len(host_data))

    def download(self, device_ptr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        self.hip.memcpy_d2h(buf, device_ptr, size)
        return buf.raw

    def synchronize(self):
        self.hip.synchronize()

    def load_hsaco(self, path: str) -> int:
        """Load HSACO and cache the module."""
        if path not in self._modules:
            self._modules[path] = self.hip.module_load(path)
        return self._modules[path]

    def get_kernel(self, module: int, name: str) -> int:
        return self.hip.module_get_function(module, name)

    def launch(self, func: int, grid: tuple, block: tuple,
               kernel_params: list, shared_mem: int = 0):
        """Launch kernel. kernel_params is a list of ctypes values."""
        self.hip.launch_kernel(func, grid, block, kernel_params, shared_mem)

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
