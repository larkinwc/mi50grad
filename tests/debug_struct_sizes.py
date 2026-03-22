#!/usr/bin/env python3
"""Debug struct size mismatch between Python and C."""
import ctypes as ct
import sys
sys.path.insert(0, "/opt/mi50grad")

# Load the C library
lib = ct.CDLL("src/runtime/c_dispatch.so")
lib.c_dispatch_get_spec_size.restype = ct.c_int
lib.c_dispatch_get_kernel_spec_size.restype = ct.c_int

c_spec_size = lib.c_dispatch_get_spec_size()
c_kernel_size = lib.c_dispatch_get_kernel_spec_size()

print(f"C CEngineLayerSpec size: {c_spec_size}")
print(f"C CKernelSpec size: {c_kernel_size}")

# Python definition
class CKernelSpec(ct.Structure):
    _fields_ = [
        ("func",         ct.c_uint64),
        ("grid_x",       ct.c_uint32),
        ("grid_y",       ct.c_uint32),
        ("grid_z",       ct.c_uint32),
        ("block_x",      ct.c_uint32),
        ("block_y",      ct.c_uint32),
        ("block_z",      ct.c_uint32),
        ("shared_mem",   ct.c_uint32),
        ("params_array", ct.c_uint64),
        ("num_params",   ct.c_uint32),
        ("present",      ct.c_uint32),
    ]

print(f"Python CKernelSpec size: {ct.sizeof(CKernelSpec)}")

# Python CEngineLayerSpec from tp_engine.py
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
        ('gemv_k_only',         CKernelSpec),
        ('gemv_v_cache',        CKernelSpec),
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
        ('d_hidden',            ct.c_uint64),
        ('d_proj_out',          ct.c_uint64),
    ]

print(f"Python CEngineLayerSpec size: {ct.sizeof(CEngineLayerSpec)}")
print(f"Size mismatch: C={c_spec_size}, Python={ct.sizeof(CEngineLayerSpec)}")
print(f"Difference: {c_spec_size - ct.sizeof(CEngineLayerSpec)}")

# Count kernel specs
kernel_count = 17  # from _fields_ above
trailer = 8 + 8 + 8 + 8 + 8 + 8 + 4 + 4 + 8 + 8  # layer_type to d_proj_out
expected = kernel_count * ct.sizeof(CKernelSpec) + trailer
print(f"Expected size with 17 kernels + trailer: {expected}")
