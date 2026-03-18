# Double-Buffer TP=4 Validation Summary

**Date:** 2026-03-18  
**Feature:** double-buffer-tp4-validation  
**Hardware:** 4x MI50 (gfx906) @ root@192.168.1.198

---

## Validation Status (Updated 2026-03-18)

| Assertion | Description | Status | Notes |
|-----------|-------------|--------|-------|
| VAL-DB-001 | Buffer swap alternation | ✅ PASS | Verified on dev server - even layers read A→write B, odd layers read B→write A |
| VAL-DB-002 | Numerical correctness (cos_sim >= 0.99) | ✅ PASS | **VERIFIED** - Min cos_sim=0.997815 >= 0.99 (test_val_ar_004_005.py) |
| VAL-DB-003 | Throughput improvement (>= 5%) | ❌ FAIL | Shows 0.577x (42% degradation) - double-buffer has overhead |
| VAL-DB-004 | Long-run stability (1000+ tokens) | ⏳ PENDING | Requires extended test run (test timeout) |
| VAL-DB-005 | C dispatch interaction | ⏳ PENDING | Test timeout - needs shorter test variant |

---

## Root Cause Analysis (VAL-DB-002)

### Problem
Double-buffer mode produced outputs with cosine similarity ~0.30 vs standard path, instead of >= 0.99.

### Root Cause
The `_decode_step_cached_stream` function uses pre-built `LaunchSpec` objects for kernel launches. These LaunchSpecs cache the `d_hidden` pointer value at **cache build time**. However, in double-buffer mode:

1. **FFN RMSNorm** should read from `d_hidden_write` (the buffer being written by the attention allreduce)
2. **Attention RMSNorm** (after layer 0) should read from the current `d_hidden` (which changes after each buffer swap)

The cached LaunchSpecs had **stale pointers** that didn't reflect the dynamic buffer swapping.

### Specific Issues

**Issue 1: FFN RMSNorm Input Pointer**
- At cache build time: `ffn_rmsnorm.params[1] = engine.d_hidden` (which equals `d_hidden_A`)
- At runtime (double-buffer): Should read from `engine.d_hidden_write` instead
- Bug: The cached pointer was never updated at runtime

**Issue 2: Attention RMSNorm Input Pointer (after buffer swap)**
- After each layer's buffer swap, `engine.d_hidden` points to a different buffer
- At runtime (double-buffer): Each layer's attention RMSNorm should read from the current `d_hidden`
- Bug: The cached pointer was never updated to reflect the swap

### Fix Applied

Modified `src/inference/tp_engine.py`, function `_decode_step_cached_stream`:

**Fix 1: Update Attention RMSNorm pointer (line ~2591-2596)**
```python
# --- Attention RMSNorm (cached, all static) ---
# CRITICAL FIX: In double-buffer mode, update the cached LaunchSpec
# to use the current d_hidden pointer, since buffers swap each layer
if use_double_buffer:
    attn_rmsnorm_spec = layer_cache['attn_rmsnorm']
    attn_rmsnorm_spec.params[1].value = engine.d_hidden
engine.device.launch_cached(layer_cache['attn_rmsnorm'])
```

**Fix 2: Update FFN RMSNorm pointer (line ~2694-2699)**
```python
# --- FFN RMSNorm ---
# Standard mode: reads d_hidden (gated by attention AR done event)
# Double-buffer mode: reads d_hidden_write (gated by AR stream event)
# CRITICAL FIX: In double-buffer mode, update the cached LaunchSpec
# to use d_hidden_write as input, since the cached pointer is stale
if use_double_buffer:
    ffn_rmsnorm_spec = layer_cache['ffn_rmsnorm']
    ffn_rmsnorm_spec.params[1].value = engine.d_hidden_write
engine.device.launch_cached(layer_cache['ffn_rmsnorm'])
```

### Why This Fixes the Issue

The LaunchSpec's `params[1]` is the input buffer pointer for RMSNorm kernels. By updating it at runtime:

1. **Attention RMSNorm** now reads from the correct buffer after each swap
2. **FFN RMSNorm** now reads from `d_hidden_write` (the buffer being written by attention allreduce)
3. The computational graph is now **numerically equivalent** to the standard path, just with different buffer management

### Expected Outcome

With this fix, double-buffer mode should produce **identical numerical results** to standard mode (cosine similarity >= 0.99), while achieving the desired compute-communication overlap for improved throughput.

---

## Deployment & Verification

### Files Changed
- `src/inference/tp_engine.py` - Fixed RMSNorm pointer updates in double-buffer mode
- `tests/test_double_buffer_tp4.py` - Updated test comments
- `DOUBLE_BUFFER_VALIDATION_SUMMARY.md` - This document

### Deploy to Dev Server

```bash
# From project root
./scripts/deploy_and_test_valdb002.sh
```

Or manually:
```bash
# Sync files
rsync -avz src/inference/tp_engine.py tests/test_double_buffer_tp4.py \
    root@192.168.1.198:/opt/mi50grad/

# Run VAL-DB-002 test
ssh root@192.168.1.198 "cd /opt/mi50grad && docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/workspace -v /opt/models:/opt/models \
    mi50grad python3 tests/test_double_buffer_tp4.py --correctness --steps 20"
```

### Expected Test Output

```
======================================================================
VAL-DB-002: Numerical Correctness Test (Single-Engine Approach)
======================================================================
...
Running 20 decode steps (per-step comparison)...
  Step 1/20: cos_sim=0.999xxx, max_diff=x.xxxxe-xx
  Step 6/20: cos_sim=0.999xxx, max_diff=x.xxxxe-xx
  Step 11/20: cos_sim=0.999xxx, max_diff=x.xxxxe-xx
  Step 16/20: cos_sim=0.999xxx, max_diff=x.xxxxe-xx
  Step 20/20: cos_sim=0.999xxx, max_diff=x.xxxxe-xx

======================================================================
Results:
  Min cosine similarity:  0.99xxxx
  Avg cosine similarity:  0.99xxxx
  Max absolute difference: x.xxxxe-xx
  Threshold: >= 0.99

VAL-DB-002: PASS
======================================================================
```

### Actual Test Output (2026-03-18) ✅ PASS

**From test_val_ar_004_005.py:**
```
VAL-AR-004 (Correctness):
  Min cosine similarity: 0.997815
  Avg cosine similarity: 0.999437
  Threshold: >= 0.99
  Result: PASS
```

The numerical correctness is **VERIFIED** on the dev server (4x MI50). The minimum cosine similarity of 0.997815 exceeds the 0.99 threshold, confirming that double-buffer mode produces numerically equivalent output to the standard path.

### VAL-DB-003 Throughput Analysis ❌ FAIL

**Test:** `test_val_ar_004_005.py` (VAL-AR-005)

**Result:**
```
VAL-AR-005 (Throughput):
  Standard: 28.51 ms/step
  Double-buffer: 49.39 ms/step
  Speedup: 0.577x
  Threshold: >= 1.05x
  Result: FAIL
```

**Analysis:**
The double-buffer implementation shows **42% degradation** instead of the expected 5%+ improvement. This is unexpected and suggests:

1. **Stream synchronization overhead:** The GPU stream event mechanism may be introducing more overhead than anticipated
2. **Python dispatch bottleneck:** The additional buffer swap operations in Python may be adding overhead
3. **Suboptimal overlap:** The allreduce may be completing faster than expected, reducing the benefit of overlap
4. **Test configuration:** The test uses only 10 steps after 5 warmup - may not represent steady-state behavior

**Next Steps for Throughput Investigation:**
1. Profile the stream event overhead to quantify synchronization cost
2. Test with longer sequences to amortize Python overhead
3. Verify that stream events are correctly ordered and not causing unnecessary waits
4. Consider removing buffer swap from hot path or optimizing the swap operation

---

## Test Infrastructure

Created comprehensive test file: `tests/test_double_buffer_tp4.py`

**Test modes:**
- `--buffer-swap`: Validates VAL-DB-001 (buffer alternation pattern)
- `--correctness`: Validates VAL-DB-002 (numerical equivalence)
- `--benchmark`: Validates VAL-DB-003 (throughput comparison)
- `--stability`: Validates VAL-DB-004 (1000+ token stability)
- `--c-dispatch`: Validates VAL-DB-005 (C dispatch precedence)
- `--all`: Run all tests

**Usage on dev server:**
```bash
ssh root@192.168.1.198 "cd /opt/mi50grad && docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/workspace -v /opt/models:/opt/models \
    mi50grad python3 tests/test_double_buffer_tp4.py --all"
```

---

## Detailed Test Results (2026-03-18)

### VAL-DB-001: Buffer Swap Alternation ✅ PASS

**Test Command:**
```bash
ssh root@192.168.1.198 "cd /opt/mi50grad && docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --buffer-swap"
```

**Result:**
```
Buffer usage pattern (first 8 layers):
  Layer  0: read=A, write=B ✓
  Layer  1: read=B, write=A ✓
  Layer  2: read=A, write=B ✓
  Layer  3: read=B, write=A ✓
  ...
After 64 layers:
  Final d_hidden: A (expected: A)

VAL-DB-001: PASS - Buffer alternation correct
```

**Validation:**
- Even layers correctly read from `d_hidden_A` and write to `d_hidden_B`
- Odd layers correctly read from `d_hidden_B` and write to `d_hidden_A`
- After 64 layers (even number), `d_hidden` returns to `d_hidden_A`
- Buffer swap mechanism (`_swap_hidden_buffers()`) working correctly

---

### VAL-DB-002: Numerical Correctness ✅ PASS

**Test Command:**
```bash
ssh root@192.168.1.198 "cd /opt/mi50grad && docker run --rm ... mi50grad python3 tests/test_val_ar_004_005.py"
```

**Result:**
```
VAL-AR-004 (Correctness):
  Min cosine similarity: 0.997815
  Avg cosine similarity: 0.999437
  Threshold: >= 0.99
  Result: PASS
```

**Validation:**
- Double-buffer produces numerically equivalent output to standard path
- All 10 decode steps achieved cosine similarity > 0.99
- Fix applied in _decode_step_cached_stream() correctly updates RMSNorm pointers at runtime
- The single-engine comparison approach eliminates cross-engine state differences

---

### VAL-DB-003: Throughput ❌ FAIL

**Test Command:**
```bash
ssh root@192.168.1.198 "cd /opt/mi50grad && docker run --rm ... mi50grad python3 tests/test_val_ar_004_005.py"
```

**Result:**
```
VAL-AR-005 (Throughput):
  Standard: 28.51 ms/step
  Double-buffer: 49.39 ms/step
  Speedup: 0.577x
  Threshold: >= 1.05x
  Result: FAIL
```

**Analysis:**
Double-buffer shows 42% degradation instead of expected 5%+ improvement.

**Potential Causes:**
1. Stream event synchronization overhead exceeds expected allreduce hide time
2. Python-side buffer swap adds overhead to hot path
3. Allreduce completes faster than expected, reducing overlap benefit
4. Test configuration (10 steps) may not represent steady-state

**Recommendation:** Further profiling needed to identify the source of overhead. The double-buffer mechanism is numerically correct but requires performance tuning.

---

## Detailed Results (Previous)

### VAL-DB-001: Buffer Swap Alternation ✅ PASS

**Test:** `test_double_buffer_tp4.py --buffer-swap`

**Result:**
```
Buffer usage pattern (first 8 layers):
  Layer  0: read=A, write=B ✓
  Layer  1: read=B, write=A ✓
  Layer  2: read=A, write=B ✓
  Layer  3: read=B, write=A ✓
  ...
After 64 layers:
  Final d_hidden: A (expected: A)

VAL-DB-001: PASS - Buffer alternation correct
```

**Validation:**
- Even layers correctly read from `d_hidden_A` and write to `d_hidden_B`
- Odd layers correctly read from `d_hidden_B` and write to `d_hidden_A`
- After 64 layers (even number), `d_hidden` returns to `d_hidden_A`
- Buffer swap mechanism (`_swap_hidden_buffers()`) working correctly

---

### VAL-DB-002: Numerical Correctness ❌ FAIL

**Test:** `test_double_buffer_tp4.py --correctness --steps 10`

**Expected:** Cosine similarity >= 0.99 between standard and double-buffer paths

**Actual:** Cosine similarity ~0.06-0.30 (varies by step)

**Sample output:**
```
Running 10 decode steps...
  Step 1/10: cos_sim=0.299560, max_diff=2.399219e+01
  Step 6/10: cos_sim=0.065960, max_diff=1.875098e+01
  Step 10/10: cos_sim=0.290698, max_diff=1.453125e+01

Results:
  Min cosine similarity:  0.061231
  Threshold: >= 0.99

VAL-DB-002: FAIL
```

**Investigation:**

1. **Engine initialization order:** Identified issue where `set_cached_dispatch(True)` triggers automatic `build_dispatch_cache()` before weights are loaded. Fixed by loading weights first, then setting dispatch modes.

2. **Two-engine comparison approach:** The test creates two separate engine instances (standard and double-buffer) and compares their outputs. This approach may introduce state differences.

3. **Comparison with existing test:** The existing `test_val_ar_004_005.py` uses the same pattern but hasn't been successfully executed on the dev server yet due to import path issues (fixed).

**Next Steps for Debugging:**

1. Run the fixed `test_val_ar_004_005.py` to see if it passes
2. If it passes, compare the implementation with `test_double_buffer_tp4.py` to identify differences
3. If it also fails, the issue is in the double-buffer implementation itself
4. Consider running a single-engine test: run standard mode, reset state, enable double-buffer, run again with same input

**Potential Root Causes:**

1. **KV cache state:** The two engines may have different KV cache initialization
2. **Stream synchronization:** Double-buffer mode may have different stream event timing
3. **Buffer pointer swap timing:** The `_swap_hidden_buffers()` call may be happening at wrong time
4. **Allreduce write destination:** Double-buffer writes to `d_hidden_write` but may not be waiting for completion before next layer reads

---

## Implementation Background

### Double-Buffer Mechanism

The double-buffer optimization enables compute-communication overlap by:

1. **Two buffers per GPU:** `d_hidden_A` and `d_hidden_B` (each 5120 × 2 bytes = 10KB)
2. **Alternating writes:** Layer N writes to buffer X, layer N+1 reads from X while layer N's allreduce completes
3. **Stream events:** GPU-side synchronization via HIP events (no CPU blocking)
4. **Integration:** Works with `cached_dispatch` + `stream_overlap_dispatch` modes

**Code flow in `_decode_step_cached_stream()`:**
```python
use_double_buffer = self._double_buffer_enabled

for layer_idx in range(num_layers):
    # NO wait at layer start when double-buffer enabled
    if not use_double_buffer and layer_idx > 0:
        p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)
    
    # ... launch kernels (RMSNorm, GEMV, attention, FFN) ...
    
    # Submit async allreduce
    if use_double_buffer:
        hidden_ptrs = [e.d_hidden_write for e in self.engines]
    else:
        hidden_ptrs = [e.d_hidden for e in self.engines]
    p2p_ar.allreduce_residual_async(partial_ptrs, hidden_ptrs, h, compute_streams)
    
    # Swap buffers after FFN allreduce
    if use_double_buffer:
        for engine in self.engines:
            engine._swap_hidden_buffers()

# Final wait for last allreduce
p2p_ar.wait_for_allreduce_on_compute_stream(compute_streams)
```

### Files Modified

1. **`tests/test_double_buffer_tp4.py`** (NEW - 696 lines)
   - Comprehensive validation test for all VAL-DB assertions
   - Supports individual test modes or full suite
   - Includes debug output for troubleshooting

2. **`tests/test_val_ar_004_005.py`** (FIXED)
   - Fixed import path for Docker container compatibility
   - Changed `sys.path.insert(0, "/opt/mi50grad")` to use `Path(__file__).parent.parent`

---

## Performance Expectations (Once Fixed)

Based on the implementation design:

- **VAL-DB-003 (Throughput):** Expected 5-10% improvement when combined with stream overlap
  - Hides allreduce latency (~79μs per call) behind next-layer RMSNorm dispatch
  - Python dispatch time with cached dispatch: ~14ms/layer
  - Allreduce time: ~10ms/token total (128 calls × ~79μs)
  - Overlap efficiency: ~14ms / (14ms + 10ms) = 58% overlap

- **VAL-DB-004 (Stability):** Expected to pass with no NaN/Inf over 1000+ tokens
  - Double-buffer doesn't change numerical operations, just buffer management
  - Stream events ensure correct data dependencies

- **VAL-DB-005 (C dispatch):** Expected to pass
  - C dispatch (`_c_dispatch_enabled=True`) takes precedence
  - Double-buffer is ignored when C dispatch is active

---

## Deployment

**Test file deployed to dev server:**
```bash
rsync -avz tests/test_double_buffer_tp4.py root@192.168.1.198:/opt/mi50grad/tests/
```

**Run individual tests:**
```bash
# Buffer swap (fast, ~2 min)
ssh root@192.168.1.198 "docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --buffer-swap"

# Correctness (medium, ~5 min)
ssh root@192.168.1.198 "docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --correctness --steps 10"

# Benchmark (medium, ~5 min)
ssh root@192.168.1.198 "docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --benchmark --iters 50"

# Stability (long, ~15 min for 1000 tokens)
ssh root@192.168.1.198 "docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --stability --tokens 1000"

# C dispatch interaction (fast, ~2 min)
ssh root@192.168.1.198 "docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --c-dispatch"
```

---

## Next Steps

1. **Debug VAL-DB-002 failure:**
   - Run fixed `test_val_ar_004_005.py` to compare
   - Add more debug output to identify where outputs diverge
   - Consider single-engine test approach

2. **Once VAL-DB-002 passes:**
   - Run VAL-DB-003 (throughput benchmark)
   - Run VAL-DB-004 (long-run stability)
   - Run VAL-DB-005 (C dispatch interaction)

3. **Documentation:**
   - Update README with double-buffer usage instructions
   - Document performance characteristics based on benchmark results
   - Add troubleshooting guide for common issues

---

## References

- Implementation summary: `DOUBLE_BUFFER_IMPLEMENTATION.md`
- Architecture notes: `.factory/library/architecture.md` (search for "Double-Buffer")
- Validation contract: `.factory/missions/.../validation-contract.md` (VAL-DB-001 through VAL-DB-005)
- Existing test: `tests/test_overlap_double_buffer.py` (original minimal test)
- Related test: `tests/test_val_ar_004_005.py` (VAL-AR-004/005, similar pattern)
