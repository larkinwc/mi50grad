#!/usr/bin/env python3
"""Debug struct size mismatch between Python and C."""
import ctypes as ct
import sys
sys.path.insert(0, "/opt/mi50grad")

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

# Try to load C library for comparison (only works on dev server)
try:
    lib = ct.CDLL("/opt/mi50grad/src/runtime/c_dispatch.so")
    lib.c_dispatch_get_spec_size.restype = ct.c_int
    lib.c_dispatch_get_kernel_spec_size.restype = ct.c_int
    
    c_spec_size = lib.c_dispatch_get_spec_size()
    c_kernel_size = lib.c_dispatch_get_kernel_spec_size()
    
    print(f"\nC CEngineLayerSpec size: {c_spec_size}")
    print(f"C CKernelSpec size: {c_kernel_size}")
    c_available = True
except Exception as e:
    print(f"\nC library not available (expected on macOS): {e}")
    c_spec_size = None
    c_available = False

# Python CEngineLayerSpec from tp_engine.py (FIXED VERSION WITH gemv_qkv_fused)
class CEngineLayerSpec(ct.Structure):
    _fields_ = [
        ('attn_rmsnorm',        CKernelSpec),
        ('gemv_q_fused',        CKernelSpec),
        ('gemv_kv_fused',       CKernelSpec),
        ('gemv_qkv_fused',      CKernelSpec),  # Fused QKV GEMV (3-in-1, INT4 only) - ADDED
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

print(f"\nPython CEngineLayerSpec size: {ct.sizeof(CEngineLayerSpec)}")

# Count kernel specs
kernel_count = 18  # 17 + gemv_qkv_fused
trailer = 8 + 8 + 8 + 8 + 8 + 8 + 4 + 4 + 8 + 8  # layer_type to d_proj_out
expected = kernel_count * ct.sizeof(CKernelSpec) + trailer
print(f"Expected size with {kernel_count} kernels + trailer: {expected}")

if c_available and c_spec_size is not None:
    print(f"\nSize comparison: C={c_spec_size}, Python={ct.sizeof(CEngineLayerSpec)}")
    if c_spec_size == ct.sizeof(CEngineLayerSpec):
        print("✓ SUCCESS: Struct sizes match!")
        sys.exit(0)
    else:
        print(f"✗ MISMATCH: Difference = {c_spec_size - ct.sizeof(CEngineLayerSpec)} bytes")
        sys.exit(1)
else:
    print(f"\nC struct expected size: 1104 bytes (18 CKernelSpecs * 48 bytes + 240 bytes trailer)")
    print(f"Python struct calculated size: {ct.sizeof(CEngineLayerSpec)} bytes")
    if ct.sizeof(CEngineLayerSpec) == 1104:
        print("✓ Python struct size matches expected C struct size (1104 bytes)")
        sys.exit(0)
    else:
        print(f"✗ Python struct size does not match expected (1104 bytes)")
        sys.exit(1)
