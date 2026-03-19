# Cross-WG Coord Issue Analysis: Fused GEMV+AR+RMSNorm Kernel

**Feature**: m1-analyze-cross-wg-issue  
**Milestone**: m1-fused-gemv-v2  
**Date**: 2026-03-19  
**Author**: GPU Kernel Worker (Session 27f3ed8d-3bef-4a54-a205-5288fd781e21)

---

## Executive Summary

The fused GEMV+AR+RMSNorm kernel (`gemv_int4_p2p_allreduce_rmsnorm.hip`) causes a **71% throughput regression** (45 tok/s → 15 tok/s) due to **missing cross-workgroup synchronization** for RMSNorm sum-of-squares computation.

### Root Cause
The kernel uses **multiple workgroups** (80 WGs for N=5120, TP=4) to compute GEMV output columns, but **RMSNorm requires a global sum-of-squares across ALL N=5120 columns**. The current implementation attempts to compute this within each workgroup's LDS, but **workgroups cannot see each other's partial results** without explicit synchronization via global memory.

### Evidence from Code
File: `src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip`, lines 240-275:
```cpp
// Store allreduce results for our 16 columns in LDS
__shared__ float s_col_results[16];
if (k_split_id == 0) {
    s_col_results[col_in_wg] = total;
}
__syncthreads();

// Compute sum-of-squares across ALL N columns
float sum_sq = 0.0f;

// Vectorized strided loop: thread t processes {t*8, t*8+2048, ...}
for (unsigned int i = threadIdx.x * 8; i + 7 < dim; i += 256 * 8) {
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        unsigned int col_i = i + k;
        float allreduce_result;
        
        unsigned int wg_col_start = blockIdx.x * COLS_PER_WG;
        unsigned int wg_col_end = wg_col_start + COLS_PER_WG;
        
        if (col_i >= wg_col_start && col_i < wg_col_end) {
            // Column in our workgroup - read from LDS
            allreduce_result = s_col_results[col_i - wg_col_start];
        } else if (col_i >= col_start && col_i < col_start + cols_per_gpu) {
            // Column in our GPU's partition but different workgroup
            // Read peer partials (our GEMV contribution is in our workgroup's LDS)
            // For now, approximate as peer sum (IMPERFECT - needs cross-WG LDS)
            allreduce_result = __half2float(partial_peer0[col_i])
                             + __half2float(partial_peer1[col_i])
                             + __half2float(partial_peer2[col_i]);
        } else {
            // Column in peer partition
            allreduce_result = __half2float(partial_peer0[col_i])
                             + __half2float(partial_peer1[col_i])
                             + __half2float(partial_peer2[col_i]);
        }
        
        sum_sq += allreduce_result * allreduce_result;
    }
}
```

**Critical Issue**: When `col_i` belongs to a **different workgroup on the same GPU**, the code attempts to read from `partial_peer*` buffers, but these buffers **do NOT contain the GEMV results yet** - they contain peer GPU partials from the previous layer's allreduce. The actual GEMV results for those columns are stored in **that workgroup's LDS**, which is inaccessible to other workgroups.

---

## Current Kernel Architecture Analysis

### Workgroup Structure
```
Kernel launch configuration (TP=4, N=5120):
- cols_per_gpu = 5120 / 4 = 1280 columns per GPU
- COLS_PER_WG = 16 columns per workgroup
- num_wgs = ceil(1280 / 16) = 80 workgroups per GPU

Thread layout:
- 256 threads per workgroup
- 16 threads per output column
- 16 columns processed in parallel per WG
```

### Phase Breakdown

#### Phase 1: INT4 GEMV (Lines 134-198)
- Each workgroup independently computes GEMV for its 16 assigned columns
- Uses register-cached scale/zero prefetching
- Cooperative reduction via DPP (intra-wavefront) + LDS (cross-wavefront)
- **Output**: `partial_result` for each of the 16 columns

#### Phase 2: P2P Allreduce (Lines 205-215)
- Each GPU reads peer partials via BAR1-mapped P2P pointers
- Reduces: `total = partial_result + peer0[col] + peer1[col] + peer2[col]`
- **Critical Note**: `partial_local` is NOT added (would double-count GEMV result)

#### Phase 3: RMSNorm (Lines 223-310) - **WHERE IT FAILS**
**Step 3a**: Store allreduce results in LDS (per-WG)
```cpp
__shared__ float s_col_results[16];
if (k_split_id == 0) {
    s_col_results[col_in_wg] = total;  // Only 16 columns visible to this WG
}
```

**Step 3b**: Compute sum-of-squares with strided loop
- Each thread processes columns at stride `256 * 8 = 2048`
- Problem: When accessing columns from **other workgroups**, LDS is stale/inaccessible
- **Workaround in current code**: Reads from `partial_peer*` buffers, but these are **WRONG** for same-GPU columns

**Step 3c**: Warp reduction (DPP + LDS)
- Correctly implemented for intra-WG reduction
- But `total_sq` is **INCORRECT** because it used wrong values for peer columns

#### Phase 4: Apply RMSNorm (Lines 317-324)
- Each thread normalizes its column: `output = total * rms_inv * weight`
- Uses the **incorrect** `rms_inv` computed from wrong sum-of-squares

---

## Why This Causes 71% Regression

### Hypothesis 1: Incorrect RMSNorm Computation (Most Likely)
The sum-of-squares is computed using **stale/incorrect values** for columns in other workgroups. This leads to:
- **Wrong `rms_inv`** scaling factor
- **Numerically incorrect output** that propagates through all 64 layers
- **Garbage output** that may trigger NaN/Inf handling overhead
- **Coherence issues** that force conservative synchronization

### Hypothesis 2: Memory Bandwidth Contention
When workgroups read `partial_peer*` buffers for columns they don't own:
- **Increased L2 cache pressure** from random access patterns
- **P2P read latency** (~12 GB/s per link) adds up across 80 WGs
- **Cache thrashing** between GEMV weight loads and peer buffer reads

### Hypothesis 3: Divergent Execution Paths
The conditional logic for determining column ownership:
```cpp
if (col_i >= wg_col_start && col_i < wg_col_end) {
    // Path A: read LDS
} else if (col_i >= col_start && col_i < col_start + cols_per_gpu) {
    // Path B: read peer partials
} else {
    // Path C: read peer partials
}
```
- **Warp divergence** across 64-lane wavefronts
- **Serialization** of execution paths
- **Reduced instruction throughput**

---

## Proposed Solution: Atomic Completion Counter

### Design Pattern (from gemv_int4_dual.hip and gemv_int4_v2.hip)

The existing codebase already has a working pattern for cross-WG coordination in split-K GEMV kernels:

```cpp
// Global memory atomic counter (declared in global scope)
__device__ unsigned int done[N] = {0};

// Per-column completion tracking
extern "C" __global__ void gemv_int4_v2_fused(...) {
    // ... compute partial sum ...
    
    // Accumulate into persistent FP32 buffer
    atomicAdd(&C_fp32[col], acc);
    
    // Fence to ensure atomicAdd is globally visible
    __threadfence();
    
    // Increment completion counter
    unsigned int old_done = atomicAdd(&done[col], 1U);
    if (old_done == k_splits - 1U) {
        // LAST tile: write output and reset
        float total = C_fp32[col];
        C_fp16[col] = __float2half(total);
        C_fp32[col] = 0.0f;
        done[col] = 0;
    }
}
```

### Adaptation for RMSNorm Cross-WG Sync

**Approach**: Use a **single workgroup-level completion counter** instead of per-column counters:

```cpp
// Global memory: one counter per GPU (not per column!)
__device__ unsigned int wg_completion_counter[4];  // One per TP rank

__global__ void gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel(...) {
    __shared__ float s_local_sum_sq;     // This WG's partial sum-of-squares
    __shared__ float s_rms_inv_broadcast; // Broadcast value for all WGs
    
    unsigned int wg_id = blockIdx.x;
    unsigned int num_wgs = gridDim.x;
    
    // Phase 1-2: GEMV + Allreduce (unchanged)
    // ... compute total for each of our 16 columns ...
    
    // Phase 3a: Compute LOCAL sum-of-squares for our 16 columns
    s_local_sum_sq = 0.0f;
    if (k_split_id == 0) {
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            float val = s_col_results[c];
            s_local_sum_sq += val * val;
        }
    }
    
    // Phase 3b: Atomic completion counter
    __threadfence(); // Ensure all LDS writes visible (not strictly needed for counter)
    
    unsigned int my_id = atomicAdd(&wg_completion_counter[tp_rank], 1U);
    
    // Phase 3c: Last WG performs global reduction
    if (my_id == num_wgs - 1U) {
        // We are the LAST workgroup to arrive
        // Need to read all WGs' partial sum-of-squares from global memory
        
        // But wait - we can't directly access other WGs' LDS!
        // Solution: Use a global memory array for partial sums
    }
    
    __syncthreads();
    
    // Phase 3d: All WGs read broadcast rms_inv
    // ... apply RMSNorm ...
}
```

### Corrected Design with Global Memory Partial Sum Array

```cpp
// Global memory arrays (one set per TP rank)
__device__ float* wg_partial_sum_sq[4];  // [num_wgs] per GPU
__device__ unsigned int wg_completion_counter[4];

__global__ void gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel(...) {
    __shared__ float s_local_sum_sq;
    __shared__ float s_rms_inv_broadcast;
    
    unsigned int wg_id = blockIdx.x;
    unsigned int num_wgs = gridDim.x;
    unsigned int lane = threadIdx.x % 64;
    
    // Phase 1-2: GEMV + Allreduce (unchanged)
    // ... compute total for each of our 16 columns ...
    
    // Phase 3a: Compute LOCAL sum-of-squares for our 16 columns
    float local_sum_sq = 0.0f;
    if (k_split_id == 0) {
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            float val = s_col_results[c];
            local_sum_sq += val * val;
        }
        // Write to global memory array
        wg_partial_sum_sq[tp_rank][wg_id] = local_sum_sq;
    }
    
    // Phase 3b: Atomic completion counter with memory fence
    __threadfence(); // Ensure global memory write is visible
    unsigned int my_id = atomicAdd(&wg_completion_counter[tp_rank], 1U);
    
    // Phase 3c: Last WG performs global reduction
    float total_sum_sq = 0.0f;
    if (my_id == num_wgs - 1U) {
        // We are the LAST workgroup
        // Sum all partials from global memory
        for (unsigned int w = 0; w < num_wgs; w++) {
            total_sum_sq += wg_partial_sum_sq[tp_rank][w];
        }
        
        // Compute inverse RMS
        float rms_inv = rsqrtf(total_sum_sq / (float)dim + eps);
        
        // Store in LDS for broadcast
        s_rms_inv_broadcast = rms_inv;
        
        // Reset counter for next token (MUST happen before other WGs continue)
        wg_completion_counter[tp_rank] = 0;
    }
    
    // Phase 3d: Broadcast to all WGs
    __syncthreads();
    
    // Read broadcast value
    float rms_inv = s_rms_inv_broadcast;
    
    // Phase 4: Apply RMSNorm (unchanged)
    if (k_split_id == 0) {
        float normalized = total * rms_inv;
        float w = __half2float(weight[col]);
        output[local_col] = __float2half(normalized * w);
    }
}
```

### Memory Layout Requirements

```cpp
// Host-side allocation (in tp_engine.py or c_dispatch.c)
// Must be allocated once at engine initialization

// For each GPU:
// - wg_partial_sum_sq[tp_rank]: array of [max_num_wgs] floats
// - wg_completion_counter[tp_rank]: single uint (or array [1])

// Example allocation in Python:
num_wgs_max = 128  // ceil(17408/128/16) for largest FFN layer
wg_partial_sum_sq = [hipMalloc(num_wgs_max * 4) for _ in range(4)]
wg_completion_counter = [hipMalloc(4) for _ in range(4)]

// Pass to kernel launch:
err = fused_lib.gemv_int4_p2p_allreduce_rmsnorm_tp4(
    ...
    wg_partial_sum_sq[tp_rank],  // New parameter
    wg_completion_counter[tp_rank],  // New parameter
    ...
)
```

---

## Test Plan for Verification

### Test 1: Atomic Counter Correctness (VAL-M1-003)

**Purpose**: Verify cross-WG synchronization works correctly

**Setup**:
```python
def test_cross_wg_correctness():
    """Test with 4+ workgroups to ensure proper synchronization."""
    # Force multiple WGs by using large N
    N = 5120  # 80 WGs with TP=4
    K = 17408
    
    # Run fused kernel
    result_fused = run_fused_kernel(...)
    
    # Run reference path (separate GEMV + allreduce + RMSNorm)
    result_ref = run_reference_path(...)
    
    # Verify correctness
    cosine_sim = compute_cosine_similarity(result_fused, result_ref)
    max_err = compute_max_abs_error(result_fused, result_ref)
    
    assert cosine_sim >= 0.99, f"cosine_sim={cosine_sim}"
    assert max_err < 5e-3, f"max_err={max_err}"
```

**Expected Behavior**:
- All 80 WGs complete GEMV before any WG starts RMSNorm
- Last WG correctly sums all 80 partial sum-of-squares values
- All WGs read the same `rms_inv` broadcast value
- Output matches reference path with cosine_sim >= 0.99

### Test 2: Atomic Counter Stress Test

**Purpose**: Verify counter resets correctly across multiple tokens

**Setup**:
```python
def test_atomic_counter_stress():
    """Run 100 consecutive tokens to verify counter reset."""
    for token_idx in range(100):
        result = run_fused_kernel(...)
        result_ref = run_reference_path(...)
        
        err = compute_max_abs_error(result, result_ref)
        if err >= 5e-3:
            print(f"FAIL at token {token_idx}: err={err}")
            return False
    
    return True
```

**Expected Behavior**:
- Counter resets to 0 after each token
- No accumulation or drift across tokens
- Consistent correctness across all 100 iterations

### Test 3: Multi-Iteration with hipDeviceSynchronize()

**Purpose**: Ensure proper synchronization between iterations

**Setup**:
```python
def test_multi_iteration_sync():
    """Test with explicit synchronization between iterations."""
    for i in range(10):
        result = run_fused_kernel(...)
        hipDeviceSynchronize()  # Force completion
        
        result_ref = run_reference_path(...)
        hipDeviceSynchronize()
        
        assert cosine_sim(result, result_ref) >= 0.99
```

**Expected Behavior**:
- Each iteration is independent and correct
- No race conditions from concurrent kernel launches

---

## Code Locations for Fix

### File 1: `src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip`

**Changes Needed**:
1. Add global memory declarations (lines 1-10):
   ```cpp
   // Global workgroup coordination arrays
   extern "C" __device__ float* d_wg_partial_sum_sq[4];
   extern "C" __device__ unsigned int* d_wg_completion_counter[4];
   ```

2. Update kernel signature (line ~100):
   ```cpp
   __global__ void gemv_int4_p2p_allreduce_rmsnorm_tp4_kernel(
       ...
       float* wg_partial_sum_sq,  // NEW: [num_wgs]
       unsigned int* wg_completion_counter,  // NEW: [1]
       ...
   )
   ```

3. Replace Phase 3 (lines 223-310) with atomic counter implementation

4. Update host wrapper function (line ~370) to pass new parameters

### File 2: `src/runtime/c_dispatch.c`

**Changes Needed**:
1. Add fields to `CAllreduceSpec` struct:
   ```c
   typedef struct {
       ...
       void** wg_partial_sum_sq;  // [4] arrays
       void** wg_completion_counter;  // [4] counters
   } CAllreduceSpec;
   ```

2. Initialize in `c_allreduce_init()`:
   ```c
   for (int i = 0; i < 4; i++) {
       hipMalloc(&spec->wg_partial_sum_sq[i], max_num_wgs * sizeof(float));
       hipMalloc(&spec->wg_completion_counter[i], sizeof(unsigned int));
       hipMemset(spec->wg_completion_counter[i], 0, sizeof(unsigned int));
   }
   ```

3. Pass to kernel launch in `do_allreduce_gemv_fused()`:
   ```c
   spec->gemv_fused_tp4_fn(
       ...
       spec->wg_partial_sum_sq[tp_rank],
       spec->wg_completion_counter[tp_rank],
       ...
   );
   ```

### File 3: `src/inference/tp_engine.py`

**Changes Needed**:
1. Allocate arrays in `__init__()`:
   ```python
   self.wg_partial_sum_sq = [
       hip.hipMalloc(max_num_wgs * 4) for _ in range(4)
   ]
   self.wg_completion_counter = [
       hip.hipMalloc(4) for _ in range(4)
   ]
   ```

2. Pass to kernel wrapper in `run_layer()`

---

## Performance Considerations

### Overhead Analysis
- **Atomic counter increment**: ~100-200 cycles per WG
- **__threadfence()**: ~500-1000 cycles (ensures global visibility)
- **Global memory read (last WG)**: 80 floats × ~400 cycles = ~32,000 cycles
- **Total overhead**: ~35,000 cycles ≈ **35 microseconds** (at 1 GHz)

### Comparison to Current Path
- **Current (broken)**: 0 overhead but incorrect results
- **Separate kernels**: 3 kernel launches × ~20μs = ~60μs launch overhead
- **Proposed fix**: ~35μs synchronization overhead

**Net Impact**: Should recover the 60μs launch overhead while maintaining correctness.

### Bottleneck Analysis
The last-WG reduction loop:
```cpp
for (unsigned int w = 0; w < num_wgs; w++) {
    total_sum_sq += wg_partial_sum_sq[tp_rank][w];
}
```
- **80 iterations** for N=5120
- Could be optimized with **parallel reduction** if needed
- For now, simple loop is acceptable (only 1 WG does this)

---

## References

1. **Existing atomic counter pattern**: `src/kernels/gemv_int4_v2.hip` (lines 200-230)
2. **Split-K coordination**: `src/kernels/gemv_int4_dual.hip` (lines 120-160)
3. **Library documentation**: `.factory/library/fused-gemv-patterns.md`
4. **Validation synthesis**: `.factory/validation/fused-gemv-ar/scrutiny/synthesis.json`

---

## Summary

**Problem**: Cross-WG RMSNorm coordination is broken because workgroups cannot access each other's LDS data.

**Root Cause**: Phase 3 attempts to compute global sum-of-squares using local LDS, but LDS is per-WG.

**Solution**: Implement atomic completion counter with global memory partial sum array:
1. Each WG writes its partial sum-of-squares to global memory
2. Each WG atomically increments completion counter
3. Last WG sums all partials and broadcasts `rms_inv` via LDS
4. All WGs apply RMSNorm using broadcast value

**Expected Impact**: Restore throughput from 15 tok/s (broken) to 55+ tok/s (target) by eliminating 64 kernel launches per token while maintaining numerical correctness.

**Test Plan**: Three tests (correctness, stress, multi-iteration sync) to verify cross-WG coordination.

---

**Next Steps**:
1. Implement atomic counter mechanism in kernel
2. Update C dispatch and TP engine to allocate/pass coordination arrays
3. Run isolation tests to verify GEMV, allreduce, and RMSNorm portions
4. Benchmark full throughput with fused kernel enabled
5. Validate numerical correctness (cosine_sim >= 0.99)
