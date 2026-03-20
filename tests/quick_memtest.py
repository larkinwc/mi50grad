#!/usr/bin/env python3
"""Quick memory test to diagnose GPU allocation issues."""
import sys
sys.path.insert(0, '/opt/mi50grad')

from src.runtime.hip_dispatch import GPUDevice, HIPRuntime
import numpy as np

hip = HIPRuntime()
hip.init()
dev = GPUDevice(0)

print("Testing memory allocation...")
ptrs = []
total_alloc = 0

for size in [10*1024*1024, 50*1024*1024, 100*1024*1024, 500*1024*1024, 1000*1024*1024, 5000*1024*1024]:
    try:
        ptr = dev.malloc(size)
        ptrs.append((ptr, size))
        total_alloc += size
        print(f"  Allocated {size/(1024**2):.0f}MB, total={total_alloc/(1024**2):.0f}MB")
    except Exception as e:
        print(f"  FAILED to allocate {size/(1024**2):.0f}MB: {e}")
        break

print(f"Total allocated: {total_alloc/(1024**2):.0f}MB")
dev.cleanup()
print("SUCCESS")
