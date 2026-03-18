#!/usr/bin/env python3
"""Check P2P access between GPUs."""
import sys
sys.path.insert(0, "/opt/mi50grad")
from src.runtime.hip_dispatch import HIPRuntime

hip = HIPRuntime()
hip.init()
print(f"GPUs: {hip.device_count()}")

# Check P2P access
for i in range(4):
    for j in range(4):
        if i != j:
            can_access = hip.device_can_access_peer(i, j)
            print(f"  GPU {i} can access GPU {j}: {can_access}")
