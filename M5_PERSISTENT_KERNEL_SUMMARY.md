# Milestone 5: Persistent Megakernel Implementation

**Status:** IMPLEMENTED (ready for GPU server deployment and testing)

**Target:** Achieve 48+ tok/s TP=4 decode throughput by eliminating all host-side kernel launch overhead.

---

## Overview

The persistent megakernel (`persistent_decode.hip`) runs the **entire decode step** (64 transformer layers) as a **single GPU kernel** with internal task scheduling. This eliminates:
- ~960 `hipModuleLaunchKernel` calls per decode step
- ~14ms/tok Python dispatch overhead
- ~7ms/tok C dispatch loop overhead
- All host-side synchronization between layers

**Expected improvement:** 38-45 tok/s (C dispatch) → 48-52 tok/s (persistent kernel)

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Host (CPU)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PersistentDecodeDispatcher (Python)                  │   │
│  │  - Manages global state structure                     │   │
│  │  - Builds task queue (64 layers × ~12 tasks/layer)    │   │
│  │  - Single kernel launch per decode step               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1x hipLaunchKernelGGL per GPU
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU (All 4 run same kernel)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PersistentDecodeState (global memory)                │   │
│  │  - Task queue (2048 entries)                          │   │
│  │  - Synchronization counters (atomics)                 │   │
│  │  - P2P pointers (partial buffers from all 4 GPUs)     │   │
│  │  - KV cache pointers, RoPE tables                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────────────────────────┐     │
│  │ Scheduler WG │  │ Worker WGs (1..60)                 │     │
│  │ (WG 0)       │  │ - Pull tasks from queue            │     │
│  │ - Manages    │  │ - Execute GEMV, attention,         │     │
│  │   task queue │  │   RMSNorm, allreduce               │     │
│  │ - Tracks     │  │ - Wait for dependencies            │     │
│  │   layers     │  │ - Report completion                │     │
│  │ - Syncs      │  │                                    │     │
│  │   boundaries │  │                                    │     │
│  └──────────────┘  └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. TaskDescriptor (64 bytes)
```c
typedef struct {
    uint32_t type;              // Task type (GEMV, RMSNorm, attention, etc.)
    uint32_t layer_id;          // Layer index (0-63)
    uint64_t input_ptr;         // Input buffer device pointer
    uint64_t output_ptr;        // Output buffer device pointer
    uint64_t weight_ptr;        // Weight buffer (if applicable)
    uint64_t extra_ptr;         // Extra (scales, zeros, etc.)
    uint32_t input_size;        // Input dimension
    uint32_t output_size;       // Output dimension
    float    eps;               // RMSNorm epsilon
    uint32_t group_size;        // Quantization group size
    uint32_t dep_count;         // Number of dependencies (0-4)
    uint32_t dep_task_ids[4];   // Dependency task IDs
} TaskDescriptor;
```

#### 2. PersistentDecodeState (~132 KB)
Global state structure shared by all workgroups:
- Task queue (2048 entries × 64 bytes = 128 KB)
- Queue head/tail counters (atomics)
- Task completion counter
- Current layer tracker
- P2P allreduce state (partial pointers, hidden pointer)
- KV cache pointers per layer
- RoPE table pointers

#### 3. Scheduler Workgroup (WG 0)
- Manages task queue population
- Tracks layer boundaries
- Synchronizes attention + FFN phases
- Monitors task completion via atomic counters

#### 4. Worker Workgroups (WG 1..60)
- Pull tasks dynamically from queue
- Check dependencies before execution
- Execute kernel logic (GEMV, RMSNorm, attention, allreduce)
- Increment completion counter

---

## Implementation Files

### 1. `src/kernels/persistent_decode.hip`
Main persistent kernel implementation:
- `TaskDescriptor` and `PersistentDecodeState` structs
- `persistent_decode_scheduler()` - scheduler WG kernel
- `persistent_decode_worker()` - worker WG kernel
- `persistent_decode_tp4()` - host-callable entry point

**Key kernels:**
- `gemv_fp16_worker()` - FP16 GEMV (Q, K, V, O projections)
- `gemv_int4_worker()` - INT4 GEMV (FFN gate, up, down)
- `rmsnorm_worker()` - RMSNorm
- `p2p_allreduce_worker()` - P2P allreduce via BAR1

### 2. `src/runtime/persistent_dispatch.py`
Python wrapper and dispatcher:
- `TaskDescriptor` - ctypes structure (matches HIP)
- `PersistentDecodeState` - ctypes structure (matches HIP)
- `PersistentDecodeDispatcher` - main dispatcher class
  - `enable()` - initialize state, load kernel
  - `disable()` - cleanup
  - `decode_step()` - execute single decode step
  - `build_task_queue()` - pre-populate task queue

### 3. `src/inference/tp_engine.py` (updated)
Integration into TP engine:
- `self._persistent_enabled` - flag
- `self._persistent_dispatcher` - dispatcher instance
- `set_persistent_kernel(enabled)` - enable/disable
- `_decode_step_persistent()` - persistent kernel decode path
- Updated `decode_step()` priority chain (persistent is highest)

### 4. `tests/val_m5_persistent_kernel.py`
Validation test:
- Compilation check
- Kernel structure verification
- Correctness test (cosine sim >= 0.99)
- Throughput benchmark (target: 48+ tok/s)
- Integration check

### 5. `Makefile` (updated)
Added `persistent_decode.hip` to `KERNEL_HIP_SRCS` for automatic compilation.

---

## Task Graph (per layer)

Each transformer layer has ~12 tasks:

### Full Attention Layer (layers 3, 7, 11, ..., 63)
1. Attention RMSNorm
2. GEMV Q projection
3. GEMV K projection
4. GEMV V projection
5. QKNorm + RoPE
6. Decode attention (flash_attn_256)
7. O projection (GEMV FP16)
8. **Attention allreduce (P2P)**
9. FFN RMSNorm
10. Gate projection (GEMV INT4)
11. Up projection (GEMV INT4)
12. SiLU activation
13. Down projection (GEMV INT4)
14. **FFN allreduce (P2P)**

### DeltaNet Layer (layers 0-2, 4-6, ..., 60-62)
Same as above, but replace steps 5-7 with:
5. GEMV linear attention input proj
6. DeltaNet recurrence
7. Shift operation
8. Output projection

**Total tasks:** 64 layers × ~14 tasks = **~896 tasks per decode step**

---

## Synchronization

### On-GPU Barrier
The persistent kernel uses atomic counters for synchronization:
- `tasks_completed` - incremented by workers after each task
- `wg_ready_count` - barrier at layer boundaries
- `ar_phase` - allreduce phase tracking

### Dependency Tracking
Each `TaskDescriptor` includes `dep_count` and `dep_task_ids[]`:
- Before executing a task, worker waits for all dependencies
- Dependencies are checked via completion counter
- Example: QKNorm depends on Q and K GEMV completion

### P2P Allreduce
The allreduce task reads peer GPU partials via BAR1-mapped pointers:
```c
__device__ void p2p_allreduce_worker(
    __half* hidden,
    const __half* partial_local,
    const __half* partial_peer0,  // P2P pointer
    const __half* partial_peer1,  // P2P pointer
    const __half* partial_peer2,  // P2P pointer
    unsigned int n
) {
    // Read peer memory directly from kernel
    float p0 = __half2float(partial_peer0[idx]);
    // ...
}
```

---

## Deployment

### Build on GPU Server
```bash
# Deploy to server
./scripts/deploy.sh

# Or build manually inside Docker
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad \
    mi50grad bash -c 'cd /opt/mi50grad && make hip_kernels'
```

### Run Validation
```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/opt/mi50grad \
    -v /opt/models:/opt/models \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/val_m5_persistent_kernel.py'
```

### Enable in Engine
```python
from src.inference.tp_engine import TPInferenceEngine

engine = TPInferenceEngine(model_path, tp_size=4)
engine.set_persistent_kernel(True)  # Enable persistent mode

# Run inference
hidden = engine.decode_step(embedding, position)
```

---

## Performance Analysis

### Expected Speedup Breakdown

| Component | C Dispatch | Persistent | Savings |
|-----------|------------|------------|---------|
| Kernel launches | 960 × 1µs = 1ms | 1 × 0.1ms = 0.1ms | 0.9ms |
| C loop overhead | ~6ms | 0ms | 6ms |
| Python overhead | 0ms (already in C) | 0ms | 0ms |
| Allreduce | 128 × 79µs = 10ms | 10ms (unchanged) | 0ms |
| Compute | 11ms | 11ms (unchanged) | 0ms |
| **Total** | **~28ms** | **~21ms** | **~7ms** |

**Tok/s improvement:** 1000ms / 28ms = 35.7 tok/s → 1000ms / 21ms = 47.6 tok/s

### Bottleneck After Persistent Kernel

Once dispatch overhead is eliminated, the new bottleneck is:
1. **Allreduce (48%):** 10ms of 21ms total
2. **Compute (52%):** 11ms of 21ms total

**Future optimization target:** Fuse allreduce into GEMV epilogue (Milestone 2) to further reduce allreduce overhead.

---

## Correctness Guarantees

### Validation Tests
1. **Compilation:** Kernel must compile without errors
2. **Structure:** All required components present (scheduler, workers, task queue, P2P)
3. **Correctness:** Cosine similarity >= 0.99 vs single-GPU reference
4. **Throughput:** >= 48 tok/s (vs C dispatch ~38 tok/s)
5. **Integration:** Python wrapper integrates with TPInferenceEngine

### Numerical Precision
- FP32 accumulators in all kernels (mandatory on gfx906)
- P2P allreduce uses FP32 accumulation
- RMSNorm uses FP32 for sum-sq and normalization

### Edge Cases Handled
- Odd-sized tensors (tail handling in kernels)
- Quantization group boundaries (scale/zero reload)
- Dependency ordering (task queue ensures correct execution)

---

## Known Limitations (v1 Implementation)

1. **Simplified scheduler:** Current implementation uses busy-wait loops for dependency tracking. Production version would use more efficient barrier primitives.

2. **Static task queue:** Task queue is pre-populated at initialization. Dynamic task generation (e.g., for speculative decoding) requires queue rebuild.

3. **No double-buffering:** Current implementation doesn't overlap task execution with queue updates. Future optimization could use ping-pong buffers.

4. **Worker kernel simplifications:** The worker GEMV/RMSNorm kernels are simplified versions. Production would use the full v5/v6 optimized kernels.

5. **Single precision accumulation:** While FP32 accumulators are used, the final output is FP16. Some precision loss is expected (within acceptable threshold).

---

## Future Enhancements

### M5.1: Optimized Barrier Primitives
Replace atomic counter busy-wait with:
- Wavefront-level barriers (`__builtin_amdgcn_barrier`)
- WG-level signals (global memory + memory fences)
- Layer completion interrupts (if hardware supports)

### M5.2: Dynamic Task Generation
Allow runtime task generation for:
- Speculative decoding (variable draft length)
- Adaptive batch sizing
- Conditional computation (MoE routing)

### M5.3: Multi-Kernel Specialization
Instead of generic `gemv_worker()`, use specialized kernels:
- `gemv_fp16_v5_worker()` - DPP reduction
- `gemv_int4_v6_worker()` - register-cached scales
- `flash_attn_decode_worker()` - tuned attention

### M5.4: Persistent State Caching
Cache frequently-used data in LDS/shared memory:
- RoPE cos/sin tables (per layer)
- RMSNorm weights (per layer)
- Task queue prefetching

---

## References

- **Mirage MPK architecture:** Persistent kernel design inspiration
- **LLVM gfx906 ISA:** https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html
- **ROCm HIP programming:** https://rocm.docs.amd.com/projects/HIP/en/latest/
- **MI50 architecture:** Vega 7nm Shader ISA (AMD, 2019)

---

## Troubleshooting

### Build Errors
```bash
# Check ROCm installation
/opt/rocm/bin/hipcc --version

# Check gfx906 support
/opt/rocm/bin/rocminfo | grep -i "gfx906"
```

### Runtime Errors
```python
# Check P2P access
python3 -c "
import hip
hip.hipSetDevice(0)
for i in range(1, 4):
    hip.hipDeviceEnablePeerAccess(i, 0)
    print(f'P2P 0->{i}: enabled')
"
```

### Performance Issues
1. Check task queue occupancy (should be high)
2. Verify WG count (60 for MI50)
3. Profile kernel with ROCm profiler (`rocprof`)

---

*Last updated: 2026-03-18*
*Implementation status: COMPLETE (ready for GPU validation)*
