#!/usr/bin/env python3
"""Simple test to verify fused kernel library loads and exports correct functions."""
import ctypes
from pathlib import Path

build_dir = Path("/opt/mi50grad/build/kernels")
fused_so = build_dir / "kernel_p2p_allreduce_rmsnorm.so"

if not fused_so.exists():
    print(f"FAIL: {fused_so} not found")
    exit(1)

try:
    lib = ctypes.CDLL(str(fused_so))
    has_tp4 = hasattr(lib, "kernel_p2p_allreduce_rmsnorm_tp4")
    has_tp2 = hasattr(lib, "kernel_p2p_allreduce_rmsnorm_tp2")
    
    print(f"PASS: kernel_p2p_allreduce_rmsnorm.so loaded successfully")
    print(f"  kernel_p2p_allreduce_rmsnorm_tp4: {'OK' if has_tp4 else 'MISSING'}")
    print(f"  kernel_p2p_allreduce_rmsnorm_tp2: {'OK' if has_tp2 else 'MISSING'}")
    
    if has_tp4 and has_tp2:
        exit(0)
    else:
        exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    exit(1)
