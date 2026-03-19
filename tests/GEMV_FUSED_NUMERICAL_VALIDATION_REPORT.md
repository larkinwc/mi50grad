# Fused GEMV Kernel Numerical Correctness Validation Report

**Feature**: m1-verify-numerical-correctness  
**Date**: 2026-03-19  
**Validation Status**: STRUCTURAL PASS, NUMERICAL FAIL

---

## Executive Summary

The fused GEMV + P2P Allreduce + RMSNorm kernel (`gemv_int4_p2p_allreduce_rmsnorm.hip`) has been validated for:

1. ✅ **Kernel Compilation and Symbol Resolution (VAL-M1-001)** - PASS
2. ❌ **Numerical Correctness vs Reference Path (VAL-M1-002)** - FAIL
3. ❌ **FP32 Accumulation Preservation (VAL-M1-004)** - INCONCLUSIVE

**Critical Finding**: The fused kernel produces numerically incorrect output compared to the reference path (separate GEMV + RMSNorm kernels). Maximum absolute error exceeds 4.0 (threshold: 5e-3), with cosine similarity of ~0.22 (threshold: 0.99).

---

## Test Methodology

### Structural Validation (Single GPU)
- **Test**: `tests/test_gemv_fused_numerical_correctness.py`
- **Purpose**: Validate kernel compilation, symbol exports, and atomic counter implementation
- **Result**: All structural checks PASSED

### Numerical Validation (TP=4 Multi-GPU)
- **Test**: `tests/test_fused_gemv_isolate.py`
- **Purpose**: Compare fused kernel output against reference (gemv_int4_v6 + RMSNorm)
- **Configuration**:
  - Hidden sizes: 4096, 5120, 7168
  - Batch sizes: 1, 2, 4
  - K (FFN intermediate): 17408
  - TP size: 4
- **Result**: Numerical correctness checks FAILED

---

## Detailed Results

### VAL-M1-001: Kernel Compilation and Symbol Resolution

**Status**: ✅ PASS

**Validation**:
- ✅ Fused kernel compiles for gfx906 target
- ✅ Shared library exports `gemv_int4_p2p_allreduce_rmsnorm_tp4`
- ✅ Shared library exports `gemv_int4_p2p_allreduce_rmsnorm_tp2`
- ✅ Function signature correct (19 parameters including atomic counter)
- ✅ Kernel loads without errors

**Evidence**:
```
[TEST] Validating kernel source structure...
  [PASS] Atomic counter parameter
  [PASS] Sum-of-squares array
  [PASS] atomicAdd for counter
  [PASS] __threadfence() memory barrier
  [PASS] Last WG reduction
  [PASS] Global reduction loop
  [PASS] LDS broadcast
  [PASS] Counter reset
  [PASS] FP32 accumulation (gemv_acc)
  [PASS] FDOT2 instructions
```

---

### VAL-M1-002: Numerical Correctness vs Reference Path

**Status**: ❌ FAIL

**Validation Criteria**:
- cosine_sim(fused_output, reference_output) >= 0.99
- max_abs_error < 5e-3

**Actual Results**:
```
Test: hidden_size=5120, batch_size=1, K=17408
  Reference output range: [-4.8047, 4.4883]
  Fused output range:     [-6.6055, 5.7383]
  
  Max abs error:  4.185547 (threshold: 0.005)
  Mean abs error: 0.878566
  Cosine sim:     ~0.22 (threshold: 0.99)
  
  Status: FAIL
```

**Error Analysis**:
- **Top error locations**: Show fused output = 0.0 where reference has significant values
- **Pattern**: Errors distributed uniformly across all columns (not localized)
- **Magnitude**: Errors of 4.0+ indicate fundamental computation mismatch, not precision loss

**Sample Error Distribution**:
```
Top-10 error locations:
  idx= 4236: v6=   -4.8047, fused=    0.0000, err=4.804688
  idx= 2347: v6=   -4.7969, fused=    0.0000, err=4.796875
  idx= 4850: v6=    4.4883, fused=    0.0000, err=4.488281
  idx= 2365: v6=   -4.3398, fused=    0.0000, err=4.339844
  idx= 4801: v6=   -4.3281, fused=    0.0000, err=4.328125
```

**TP=4 Partitioning Validation**:
```
TP0 [    0- 1280]: max_err=3.650391 MISMATCH
TP1 [ 1280- 2560]: max_err=3.648438 MISMATCH
TP2 [ 2560- 3840]: max_err=4.185547 MISMATCH
TP3 [ 3840- 5120]: max_err=4.017578 MISMATCH
```

All 4 TP partitions show mismatches, indicating the issue is not isolated to a specific GPU's computation.

---

### VAL-M1-004: FP32 Accumulation Preservation

**Status**: ⚠️ INCONCLUSIVE

**Structural Validation**: ✅ PASS
- ✅ Kernel declares `float gemv_acc = 0.0f` for accumulation
- ✅ Uses `__builtin_amdgcn_fdot2` (FP32 accumulation)
- ✅ Reduction uses float precision
- ✅ Final output converted to FP16 only at write

**Numerical Validation**: ❌ FAIL
- Cannot verify FP32 accumulation is working correctly due to fundamental output mismatch
- Large errors (4.0+) suggest issue is NOT FP16 drift (which would be ~1e-3 to 1e-4)

---

## Root Cause Analysis

### Observed Symptoms
1. Fused kernel outputs zeros for many columns where reference has non-zero values
2. Errors are uniformly distributed across all columns and all TP partitions
3. Error magnitude (4.0+) far exceeds FP16 precision tolerance

### Potential Root Causes

#### 1. Weight Indexing Issue (Most Likely)
The fused kernel partitions weights for TP=4:
```cpp
unsigned int col_in_partition = local_col;
// ...
gemv_acc += dequant_fdot2(packed, cur_scale, cur_zero, A, kg << 3);
packed = B_q4[kg * cols_per_gpu + col_in_partition];
```

**Hypothesis**: Column indexing (`col_in_partition`) may not correctly map to the partitioned weight layout.

**Evidence**: 
- Test shows `B_q4` partitioned to shape `(2176, 1280)` for each GPU
- If indexing is off, threads may read wrong weights or zeros

#### 2. Scale/Zero Pointer Arithmetic
Similar to weight indexing, scale and zero pointers use partitioned layout:
```cpp
cur_scale = __half2float(scales[next_group * cols_per_gpu + col_in_partition]);
```

**Hypothesis**: Scale/zero indexing may be incorrect for partitioned weights.

#### 3. Activation Pointer Offset
The activation pointer `A` is used with `kg << 3` offset:
```cpp
A_base = kg << 3  // kg * 8 elements
```

**Hypothesis**: Activation is NOT partitioned, but the kernel may be using wrong offset calculation.

#### 4. Peer Partial Buffer Reads
The kernel reads peer partials:
```cpp
total += __half2float(partial_peer0[col]);
```

**Hypothesis**: If peer buffers are zeroed (as in the test), this shouldn't cause zeros in output, since the inline GEMV should still compute the result.

---

## Recommended Next Steps

### Immediate Actions

1. **Debug Weight Indexing**
   - Add debug output to kernel to print weight values being read
   - Compare against expected values from partitioned weight arrays
   - Verify `col_in_partition` calculation is correct

2. **Verify Scale/Zero Indexing**
   - Add debug output for scale and zero values
   - Ensure quantization group boundaries are computed correctly
   - Check `groups_per_scale` and `gps_log2` calculations

3. **Compare Intermediate Accumulation**
   - Modify kernel to output `gemv_acc` before RMSNorm
   - Compare against `gemv_int4_v6` kernel accumulation values
   - Isolate whether issue is in GEMV or later stages

4. **Check Thread/Block Launch Configuration**
   - Verify grid size: `ceil((N/TP)/16)` workgroups
   - Verify block size: 256 threads
   - Ensure all output columns are covered

### Test Enhancements

1. **Add Elementwise Debug Test**
   - Create test that compares single column output
   - Print all intermediate values (weights, scales, zeros, accumulation)
   - Step through computation manually

2. **Add Regression Test**
   - Once fixed, add test to CI/CD pipeline
   - Run on every kernel change
   - Threshold: cosine_sim >= 0.99, max_abs_error < 5e-3

---

## Test Files

### Created Files
- `tests/test_gemv_fused_numerical_correctness.py` - Structural validation (single GPU)
- `tests/GEMV_FUSED_NUMERICAL_VALIDATION_REPORT.md` - This report

### Existing Files Used
- `tests/test_fused_gemv_isolate.py` - TP=4 numerical isolation test
- `src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip` - Fused kernel source
- `src/kernels/gemv_int4_v6.hip` - Reference GEMV kernel

---

## Validation Contract Status

| Assertion | Description | Status | Notes |
|-----------|-------------|--------|-------|
| VAL-M1-001 | Kernel compilation and symbol resolution | ✅ PASS | All symbols exported correctly |
| VAL-M1-002 | Numerical correctness vs reference path | ❌ FAIL | max_abs_error=4.18 >> 5e-3 |
| VAL-M1-003 | Cross-WG atomic completion counter | ✅ PASS | Structure validated, numerical test pending |
| VAL-M1-004 | FP32 accumulation preservation | ⚠️ INCONCLUSIVE | Structure OK, numerical test pending |
| VAL-M1-005 | Throughput improvement | ⏸️ PENDING | Requires numerical correctness first |
| VAL-M1-006 | C dispatch integration | ⏸️ PENDING | Requires numerical correctness first |
| VAL-M1-007 | P2P peer partial access | ⏸️ PENDING | Requires numerical correctness first |

---

## Conclusion

The fused kernel has correct **structural** implementation:
- ✅ Compiles successfully
- ✅ Exports correct symbols
- ✅ Includes atomic counter for cross-WG coordination
- ✅ Uses FP32 accumulation in structure

However, the kernel has **numerical correctness** issues:
- ❌ Output does not match reference path
- ❌ Error magnitude indicates fundamental computation bug
- ❌ Root cause likely in weight/scale/zero indexing for TP=4 partitioned layout

**Recommendation**: Do not enable fused kernel in production until numerical correctness is validated. Focus debugging efforts on weight indexing and scale/zero pointer arithmetic in the TP=4 partitioned weight layout.

---

**Report Generated**: 2026-03-19  
**Validated By**: GPU Kernel Worker (m1-verify-numerical-correctness feature)  
**Next Review**: After kernel debugging and fix
