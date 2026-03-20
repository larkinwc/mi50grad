---
name: gpu-kernel-worker
description: Implements GPU kernels (HIP C++) and TP engine optimizations with remote build/test on dev server
---

# GPU Kernel Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

GPU kernel implementation and optimization features:
- HIP kernel development (GEMV, allreduce, attention, etc.)
- Cross-WG coordination mechanisms (atomic counters, barriers)
- Kernel fusion optimizations
- Performance profiling and tuning
- TP engine integration for GPU compute paths

## Work Procedure

### 1. Read Feature Description and Library Files
Before anything else:
- Read the feature description carefully
- Read `.factory/library/compressed-allreduce.md` (for compressed AR features)
- Read `.factory/library/batch-decode.md` (for batch decode features)
- Read `.factory/library/architecture.md` for existing kernel patterns
- Read `.factory/library/fused-gemv-patterns.md` for fused kernel design

### 2. TDD - Write Tests First (RED)
Before implementing:
- Identify existing test files for similar kernels
- Write/update test file with failing test cases covering:
  - Numerical correctness (cosine_sim >= 0.99, max_abs_error)
  - Edge cases (different dimensions, batch sizes)
  - Performance targets (latency, throughput)
- Run tests to confirm they FAIL (red phase)

### 3. Implement Kernel Changes
- Edit HIP kernel files in `src/kernels/`
- For cross-WG coordination: use `atomicAdd()` on global memory counters
- Maintain FP32 accumulation for numerical precision
- Add detailed comments explaining synchronization mechanism
- For compressed allreduce: see `.factory/library/compressed-allreduce.md` for INT8 quantization design
- For batch decode: see `.factory/library/batch-decode.md` for GEMV/GEMM switching design
- **CRITICAL: Create NEW kernel files for A/B testing, do NOT modify existing working kernels**

### 4. Build on Dev Server
```bash
# Deploy code
rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' --exclude='notes/' --exclude='plans/' --exclude='.factory' /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/

# Build kernels AND C extensions
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c "cd /opt/mi50grad && make hip_kernels c_extensions"'
```

### 5. Run Tests to Verify (GREEN)
```bash
# Run kernel-specific tests
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/test_<kernel>.py"'
```

### 6. Benchmark and Profile
```bash
# Full throughput benchmark
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/bench_current_state.py"'
```

### 7. Integration Verification
- Update `src/inference/tp_engine.py` if kernel interface changes
- Update `src/runtime/c_dispatch.c` if C dispatch integration changes
- Verify end-to-end decode produces correct output
- **Always verify no regression in batch=1 mode (>= 53.0 tok/s) and single-GPU (>= 21.0 tok/s)**

## Example Handoff

```json
{
  "salientSummary": "Fixed cross-WG coordination in fused GEMV kernel by adding atomic completion counter. Kernel now correctly synchronizes before RMSNorm reduction. Throughput improved from 51.75 to 56.2 tok/s (8.5% improvement). All tests pass with cosine_sim >= 0.99.",
  "whatWasImplemented": "Added `__shared__ bool s_last_wg` and atomic counter to gemv_int4_p2p_allreduce_rmsnorm_tp4 kernel. Last workgroup to complete GEMV performs global sum-of-squares reduction and broadcasts rms_inv via LDS. All WGs apply RMSNorm using broadcast value. Updated c_dispatch.c to enable fused path.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "make hip_kernels",
        "exitCode": 0,
        "observation": "Kernel compiled successfully, gemv_int4_p2p_allreduce_rmsnorm.so created"
      },
      {
        "command": "python3 tests/test_gemv_fused_isolate.py",
        "exitCode": 0,
        "observation": "All 8 tests passed, cosine_sim=0.9987, max_abs_error=2.3e-4"
      },
      {
        "command": "python3 tests/bench_current_state.py",
        "exitCode": 0,
        "observation": "Best throughput: 56.2 tok/s with fused kernel enabled"
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_gemv_fused_cross_wg.py",
        "cases": [
          {"name": "test_cross_wg_correctness", "verifies": "VAL-M1-003"},
          {"name": "test_atomic_counter_stress", "verifies": "VAL-M1-003"}
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on a kernel interface that doesn't exist yet
- Cross-WG synchronization requires changes to kernel launch parameters
- Throughput regression detected after implementation
- HIP compiler errors that require architectural changes
- P2P access issues that cannot be resolved within the kernel
