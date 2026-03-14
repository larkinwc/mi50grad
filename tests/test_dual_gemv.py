"""Quick test: verify fused gate+up+silu GEMV matches separate kernels."""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.path.insert(0, '/opt/mi50grad')

import numpy as np
import ctypes
from src.runtime.hip_dispatch import GPUDevice
from src.kernels.launcher import build_hip_hsaco
from pathlib import Path

BUILD_DIR = Path("/opt/mi50grad/build/kernels")
HIP_DIR = Path("/opt/mi50grad/src/kernels")

dev = GPUDevice(device_id=0)
print(f"GPU initialized")

# Build kernels
for name in ["gemv_int4_v2", "gemv_int4_dual", "elementwise_v2"]:
    build_hip_hsaco(str(HIP_DIR / f"{name}.hip"), str(BUILD_DIR / f"{name}.hsaco"))

# Load kernels
mod_v2 = dev.load_hsaco(str(BUILD_DIR / "gemv_int4_v2.hsaco"))
mod_dual = dev.load_hsaco(str(BUILD_DIR / "gemv_int4_dual.hsaco"))
mod_elem = dev.load_hsaco(str(BUILD_DIR / "elementwise_v2.hsaco"))

splitk_fn = dev.get_kernel(mod_v2, "gemv_int4_v2_splitk")
convert_fn = dev.get_kernel(mod_v2, "fp32_to_fp16")
dual_fn = dev.get_kernel(mod_dual, "gemv_int4_dual_splitk")
dual_silu_fn = dev.get_kernel(mod_dual, "dual_fp32_to_silu_fp16")
silu_fn = dev.get_kernel(mod_elem, "silu_fused_v2")

# Test dimensions (Qwen 3.5 27B FFN, TP=2)
K = 5120
N = 9472  # local_intermediate_size for TP=2
group_size = 128
k_splits = 16

# Random test data
np.random.seed(42)
x = np.random.randn(K).astype(np.float16)
gate_w = np.random.randint(0, 2**32, size=(K // 8, N), dtype=np.uint32)
up_w = np.random.randint(0, 2**32, size=(K // 8, N), dtype=np.uint32)
num_groups = K // group_size
gate_scales = np.random.randn(num_groups, N).astype(np.float16) * 0.01
gate_zeros = np.random.randn(num_groups, N).astype(np.float16) * 0.01
up_scales = np.random.randn(num_groups, N).astype(np.float16) * 0.01
up_zeros = np.random.randn(num_groups, N).astype(np.float16) * 0.01

# Upload to GPU
d_x = dev.malloc(K * 2); dev.upload(d_x, x.tobytes())
d_gate_w = dev.malloc(gate_w.nbytes); dev.upload(d_gate_w, gate_w.tobytes())
d_up_w = dev.malloc(up_w.nbytes); dev.upload(d_up_w, up_w.tobytes())
d_gate_s = dev.malloc(gate_scales.nbytes); dev.upload(d_gate_s, gate_scales.tobytes())
d_gate_z = dev.malloc(gate_zeros.nbytes); dev.upload(d_gate_z, gate_zeros.tobytes())
d_up_s = dev.malloc(up_scales.nbytes); dev.upload(d_up_s, up_scales.tobytes())
d_up_z = dev.malloc(up_zeros.nbytes); dev.upload(d_up_z, up_zeros.tobytes())

# Separate path: gate GEMV + up GEMV + silu
d_fp32_1 = dev.malloc(N * 4)
d_fp32_2 = dev.malloc(N * 4)
d_gate_out = dev.malloc(N * 2)
d_up_out = dev.malloc(N * 2)

grid_x = (N + 255) // 256

# Gate
dev.memset(d_fp32_1, 0, N * 4)
dev.launch(splitk_fn, (grid_x, k_splits, 1), (256, 1, 1), [
    ctypes.c_uint64(d_x), ctypes.c_uint64(d_gate_w),
    ctypes.c_uint64(d_gate_s), ctypes.c_uint64(d_gate_z),
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint32(K), ctypes.c_uint32(N),
    ctypes.c_uint32(group_size)])
dev.launch(convert_fn, (grid_x, 1, 1), (256, 1, 1), [
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint64(d_gate_out), ctypes.c_uint32(N)])

# Up
dev.memset(d_fp32_1, 0, N * 4)
dev.launch(splitk_fn, (grid_x, k_splits, 1), (256, 1, 1), [
    ctypes.c_uint64(d_x), ctypes.c_uint64(d_up_w),
    ctypes.c_uint64(d_up_s), ctypes.c_uint64(d_up_z),
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint32(K), ctypes.c_uint32(N),
    ctypes.c_uint32(group_size)])
dev.launch(convert_fn, (grid_x, 1, 1), (256, 1, 1), [
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint64(d_up_out), ctypes.c_uint32(N)])

# SiLU fused: gate = silu(gate) * up
dev.launch(silu_fn, ((N + 511) // 512, 1, 1), (256, 1, 1), [
    ctypes.c_uint64(d_gate_out), ctypes.c_uint64(d_up_out), ctypes.c_uint32(N)])

dev.synchronize()
ref = np.frombuffer(dev.download(d_gate_out, N * 2), dtype=np.float16).copy()
print(f"Reference (separate): first 5 = {ref[:5]}")

# Fused path: dual GEMV + silu convert
dev.memset(d_fp32_1, 0, N * 4)
dev.memset(d_fp32_2, 0, N * 4)
d_fused_out = dev.malloc(N * 2)

dev.launch(dual_fn, (grid_x, k_splits, 1), (256, 1, 1), [
    ctypes.c_uint64(d_x),
    ctypes.c_uint64(d_gate_w), ctypes.c_uint64(d_gate_s), ctypes.c_uint64(d_gate_z),
    ctypes.c_uint64(d_up_w), ctypes.c_uint64(d_up_s), ctypes.c_uint64(d_up_z),
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint64(d_fp32_2),
    ctypes.c_uint32(K), ctypes.c_uint32(N), ctypes.c_uint32(group_size)])

dev.launch(dual_silu_fn, (grid_x, 1, 1), (256, 1, 1), [
    ctypes.c_uint64(d_fp32_1), ctypes.c_uint64(d_fp32_2),
    ctypes.c_uint64(d_fused_out), ctypes.c_uint32(N)])

dev.synchronize()
fused = np.frombuffer(dev.download(d_fused_out, N * 2), dtype=np.float16).copy()
print(f"Fused:                first 5 = {fused[:5]}")

# Compare
max_err = np.max(np.abs(ref.astype(np.float32) - fused.astype(np.float32)))
cos_sim = np.dot(ref.astype(np.float32), fused.astype(np.float32)) / (
    np.linalg.norm(ref.astype(np.float32)) * np.linalg.norm(fused.astype(np.float32)))
print(f"Max abs error: {max_err}")
print(f"Cosine similarity: {cos_sim}")
print(f"PASS: {max_err < 0.01 and cos_sim > 0.9999}")
