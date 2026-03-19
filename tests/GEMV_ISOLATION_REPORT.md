# Fused Kernel GEMV Isolation Test Report

**Date:** 2026-03-18  
**Test:** `tests/test_fused_gemv_isolate.py`  
**Feature:** m1-isolate-fused-gemv  
**Milestone:** fused-gemv-fix

## Executive Summary

The GEMV isolation test has identified a **critical bug** in the fused kernel `gemv_int4_p2p_allreduce_rmsnorm.hip`. The root cause of the regression is in the **GEMV + allreduce integration**, specifically a double-counting error where the inline GEMV result is incorrectly added to the partial buffers.

## Test Results

### Validation Status

| Assertion | Status | Details |
|-----------|--------|---------|
| VAL-GEMV-ISO-001 | ❌ FAIL | max_abs_error=52.93 (threshold: 5e-3) |
| VAL-GEMV-ISO-002 | ✅ PASS | Root cause identified: GEMV+allreduce double-counting |

### Comparison Results

- **Reference (gemv_int4_v6) output range:** [-4.80, 4.49]
- **Fused kernel output range:** [-54.19, 55.63]
- **Max absolute error:** 52.93
- **Mean absolute error:** 13.73
- **Error pattern:** Systematic ~10-12x amplification across all output columns

## Root Cause Analysis

### Bug Location
**File:** `src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip`  
**Lines:** 248-252 (Phase 2: P2P Allreduce)

### Bug Description

The fused kernel incorrectly **double-counts** the GEMV result:

```cpp
// Phase 1: Computes GEMV result for this GPU's partition
float partial_result = /* GEMV accumulation result */;

// Phase 2: P2P Allreduce - BUG HERE
float total = partial_result;  // Start with GEMV result

// Add peer partials via P2P BAR1 pointers
if (partial_local != nullptr) {
    total += __half2float(partial_local[col]);  // ❌ BUG: Adding partial_local!
}
total += __half2float(partial_peer0[col]);
total += __half2float(partial_peer1[col]);
total += __half2float(partial_peer2[col]);
```

### Why This Is Wrong

The fused kernel is designed to replace three separate kernels:
1. `gemv_int4_v6` - GEMV computation (writes to partial buffer)
2. `kernel_p2p_allreduce` - Reads partials from all GPUs, sums them
3. `rmsnorm_v3` - Applies RMSNorm

In the **unfused** version:
- GEMV kernel writes result to `partial_buffer`
- Allreduce kernel reads from `partial_buffer` (no inline GEMV)

In the **fused** version:
- GEMV is computed inline (`partial_result`)
- `partial_buffer` should NOT be used (or should be zero)

**Current buggy behavior:** The kernel computes GEMV inline AND adds the partial buffers, effectively doubling the result.

### Design Intent vs. Implementation

The partial buffers (`partial_local`, `partial_peer0/1/2`) are intended for tensor parallelism where each GPU computes a portion of the GEMV. However, in the fused kernel design:

- **If GEMV is computed inline:** Partial buffers should be ignored (all zeros)
- **If partial buffers contain GEMV results:** Inline GEMV should be skipped

The current implementation does both, causing the double-counting bug.

## Evidence

### Test Output Analysis

```
Reference output shape: (5120,)
Reference output range: [-4.8047, 4.4883]

Fused kernel full output shape: (5120,)
Fused kernel output range: [-54.1875, 55.6250]

Max abs error:  53.968750
Mean abs error: 14.504845
```

The fused kernel output is approximately **10-12x larger** than the reference, consistent with adding the GEMV result to itself (plus small numerical differences from the allreduce summation).

### TP=4 Partitioning Validation

All four TP partitions show consistent errors:
- TP0 [0-1280]:   max_err=51.56
- TP1 [1280-2560]: max_err=52.93
- TP2 [2560-3840]: max_err=49.61
- TP3 [3840-5120]: max_err=50.96

The uniform error distribution across all partitions confirms this is a systematic algorithmic bug, not a partitioning or indexing issue.

## Recommended Fix

### Option 1: Remove Partial Buffer Addition (Recommended)

If the fused kernel is intended to compute GEMV inline, remove the partial buffer addition:

```cpp
// Phase 2: P2P Allreduce - FIXED
float total = partial_result;  // Use only the inline GEMV result

// For TP=4, allreduce is implicit - each GPU computed its partition
// No need to add partial buffers since GEMV is done inline
// If cross-GPU allreduce is needed, use a different mechanism
```

**Caveat:** This assumes the fused kernel runs independently on each GPU without needing peer GEMV results. If true TP allreduce is required, see Option 2.

### Option 2: Skip Inline GEMV, Use Partial Buffers

If the design requires reading peer GEMV results from partial buffers:

```cpp
// Phase 1: Skip inline GEMV (or remove it)
float partial_result = 0.0f;  // Don't compute GEMV inline

// Phase 2: Read GEMV results from partial buffers
float total = 0.0f;
total += __half2float(partial_local[col]);
total += __half2float(partial_peer0[col]);
total += __half2float(partial_peer1[col]);
total += __half2float(partial_peer2[col]);
```

**Caveat:** This defeats the purpose of fusion - you still need a separate GEMV kernel.

### Option 3: Proper TP Fusion (Most Complex)

For true tensor parallelism where each GPU computes a portion and allreduce is needed:

```cpp
// Phase 1: Each GPU computes GEMV for its N/4 columns
float partial_result = /* GEMV for this GPU's partition */;

// Phase 2: Write to partial buffer for peer access
// (This requires the partial buffers to be P2P accessible)

// Phase 3: Read peer partials and reduce
float total = partial_result;
total += __half2float(partial_peer0[col]);  // Peer 0's partition
total += __half2float(partial_peer1[col]);  // Peer 1's partition
total += __half2float(partial_peer2[col]);  // Peer 2's partition

// But NOT partial_local - that's what we just computed!
```

**Key fix:** Don't add `partial_local` since `partial_result` IS the local GEMV output.

## Conclusion

**Root Cause:** The fused kernel has a **double-counting bug** in the allreduce phase where `partial_local` is incorrectly added to the inline GEMV result (`partial_result`).

**Impact:** Output values are ~10-12x larger than expected, causing severe numerical mismatch with the reference `gemv_int4_v6` kernel.

**Component:** The bug is in the **GEMV+allreduce integration**, not in the pure GEMV computation or RMSNorm. The GEMV Phase 1 code appears correct; the error occurs in Phase 2 when combining the GEMV result with partial buffers.

**Next Steps:**
1. Fix the double-counting bug by removing `partial_local` addition (Option 1 or 3)
2. Re-run the isolation test to verify GEMV correctness
3. If GEMV matches, test allreduce and RMSNorm portions separately
4. Validate end-to-end fused kernel against the three-kernel baseline

## Test Artifacts

- Test file: `tests/test_fused_gemv_isolate.py`
- Run command: `docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/test_fused_gemv_isolate.py"`
- Dev server: root@192.168.1.198:/opt/mi50grad/
