"""
Kernel launcher: builds .s -> .hsaco and dispatches via HIP runtime.

This module provides a clean Python interface for:
1. Compiling GCN assembly kernels
2. Loading HSACO modules
3. Launching kernels with typed arguments
"""

import ctypes
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..runtime.hip_dispatch import GPUDevice


LLVM_MC = "/opt/rocm/llvm/bin/llvm-mc"
LD_LLD = "/opt/rocm/llvm/bin/ld.lld"
HIPCC = "/opt/rocm/bin/hipcc"
MCPU = "gfx906"


def build_hip_hsaco(hip_path: str, hsaco_path: Optional[str] = None,
                    mcpu: str = MCPU, extra_flags: Optional[list] = None) -> str:
    """Compile .hip C++ source to .hsaco code object using hipcc."""
    hip_path = Path(hip_path)
    if hsaco_path is None:
        hsaco_path = hip_path.with_suffix(".hsaco")
    else:
        hsaco_path = Path(hsaco_path)

    cmd = [HIPCC, "--genco", f"--offload-arch={mcpu}",
           "-O3", "-o", str(hsaco_path), str(hip_path)]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HIP compilation failed: {result.stderr}")

    return str(hsaco_path)


def build_hsaco(asm_path: str, hsaco_path: Optional[str] = None, mcpu: str = MCPU) -> str:
    """Compile .s assembly file to .hsaco code object.

    Returns path to the .hsaco file.
    """
    asm_path = Path(asm_path)
    if hsaco_path is None:
        hsaco_path = asm_path.with_suffix(".hsaco")
    else:
        hsaco_path = Path(hsaco_path)

    obj_path = hsaco_path.with_suffix(".o")

    # Assemble
    result = subprocess.run(
        [LLVM_MC, "--triple=amdgcn-amd-amdhsa", f"--mcpu={mcpu}", "--filetype=obj",
         str(asm_path), "-o", str(obj_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Assembly failed: {result.stderr}")

    # Link
    result = subprocess.run(
        [LD_LLD, "--shared", str(obj_path), "-o", str(hsaco_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Link failed: {result.stderr}")

    # Cleanup
    obj_path.unlink(missing_ok=True)

    return str(hsaco_path)


class KernelArg:
    """Helper to pack kernel arguments into a flat buffer for assembly kernels."""

    def __init__(self):
        self._parts = []

    def u32(self, val: int) -> 'KernelArg':
        self._parts.append(struct.pack("<I", val & 0xFFFFFFFF))
        return self

    def u64(self, val: int) -> 'KernelArg':
        self._parts.append(struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))
        return self

    def f32(self, val: float) -> 'KernelArg':
        self._parts.append(struct.pack("<f", val))
        return self

    def ptr(self, device_ptr: int) -> 'KernelArg':
        """Add a 64-bit device pointer."""
        return self.u64(device_ptr)

    def pad(self, nbytes: int) -> 'KernelArg':
        self._parts.append(b'\x00' * nbytes)
        return self

    def build(self) -> bytes:
        return b''.join(self._parts)


class Kernel:
    """A loaded, ready-to-launch kernel."""

    def __init__(self, device: GPUDevice, hsaco_path: str, kernel_name: str):
        self.device = device
        self.module = device.load_hsaco(hsaco_path)
        self.func = device.get_kernel(self.module, kernel_name)

    def launch(self, grid: tuple, block: tuple, kernel_params: list,
               shared_mem: int = 0):
        """Launch kernel. kernel_params is a list of ctypes values."""
        self.device.launch(self.func, grid, block, kernel_params, shared_mem)

    def launch_sync(self, grid: tuple, block: tuple, kernel_params: list,
                    shared_mem: int = 0):
        """Launch and synchronize."""
        self.launch(grid, block, kernel_params, shared_mem)
        self.device.synchronize()
