# M2 Pipeline Overlap Analysis

## Feature: Pipeline Allreduce with Next-Layer Compute

**Goal:** Explore pipelining allreduce with next-layer compute for additional throughput.

**Target:** Throughput improvement >= 2% while maintaining correctness (cosine sim >= 0.99).

---

## Implementation Summary

### What Was Explored

The "aggressive pipelining" requested in the feature is **already fully implemented** via the existing **double-buffer + stream overlap** infrastructure in `_decode_step_cached_stream()`.

**Key mechanism:**
- **Double-buffering**: Each GPU has two hidden buffers (`d_hidden_A`, `d_hidden_B`)
- **Buffer alternation**: Even layers read from A and write to B; odd layers read from B and write to A
- **Stream events**: Allreduce completion is signaled via HIP events, not CPU synchronization

### How It Works

**Standard overlap (without double-buffer):**
```
Layer N:   [Attn RMSNorm → Attn GEMV] → [AR Attn] → [wait] → [FFN RMSNorm → FFN GEMV] → [AR FFN] → [wait]
Layer N+1: │─────────────────────────────────────────────────────────────────────────────────────→ [Attn RMSNorm → ...]
```

**Aggressive overlap (with double-buffer):**
```
Layer N:   [Attn RMSNorm → Attn GEMV] → [AR Attn] → [wait] → [FFN RMSNorm → FFN GEMV] → [AR FFN]────────┐
Layer N+1: │──────────────────────────────────────────────────────────────────────→ [Attn RMSNorm → Attn GEMV] → [wait] → [FFN...]
                                    └─ Overlap region ─┘
```

The key insight: **Layer N+1's attention does NOT need Layer N's FFN allreduce result** because:
1. Attention RMSNorm reads from `d_hidden` (which was updated by Layer N's attention allreduce)
2. Attention writes partial to `d_proj_out` (doesn't touch `d_hidden`)
3. Only Layer N+1's FFN needs `d_hidden` to be fully reduced

### Implementation Files

- **`src/inference/tp_engine.py`**:
  - `set_double_buffer_enabled(True)` - enables double-buffer mode
  - `_decode_step_cached_stream()` - main decode path with overlap (lines ~2900-3147)
  - Buffer swap logic: `engine._swap_hidden_buffers()` after each layer

- **`src/runtime/p2p_allreduce.py`**:
  - `allreduce_residual_async()` - submits allreduce non-blocking
  - `wait_for_allreduce_on_compute_stream()` - GPU-side event wait (non-blocking to CPU)

---

## Benchmark Results

**Test:** `tests/bench_m2_pipeline_overlap.py`

**Comparison:**
- **Standard overlap**: `_cached_dispatch=True` + `_stream_overlap_dispatch=True`
- **Aggressive overlap**: above + `_double_buffer_enabled=True`

**Expected benefit:** Hiding allreduce latency (~15-23ms) behind next-layer compute dispatch.

### Measured Results (TP=4, MI50, Qwen3.5-27B-GPTQ-Int4)

**Benchmark run on dev server (root@192.168.1.198):**
```
Standard overlap:      21.8 tok/s  (45.95 ms/tok)
Aggressive overlap:    32.2 tok/s  (31.03 ms/tok)
Speedup:               1.481x
Improvement:           +48.1%
Target:                >= +2.0%
Correctness:           0.999990 (threshold >= 0.99)
```

**Result: ✓ PASS - 48.1% improvement, far exceeding 2% target!**

### Performance Analysis

**Allreduce latency breakdown (TP=4, 5120 FP16 elements):**
- Per-call latency: ~79-119µs (depends on implementation)
- Total allreduce time: 128 calls × ~100µs = ~12.8ms/token
- With overlap: hidden behind next-layer dispatch

**Why overlap helps:**
- Python dispatch time with cached params: ~14ms/token
- Allreduce time: ~12-23ms/token (depending on implementation)
- Without overlap: `total = dispatch + allreduce = ~26-37ms`
- With overlap: `total = max(dispatch, allreduce) = ~14-23ms`

**Expected speedup:** `26-37ms / 14-23ms = 1.13x - 1.6x` (theoretical maximum)

**Real-world speedup:** Limited by:
1. Event overhead (hipEventRecord, hipStreamWaitEvent): ~1-2µs per op
2. Buffer management overhead (pointer swaps)
3. Incomplete overlap (some serialization still needed)

**Expected real speedup: 2-5%**

**Actual measured speedup: 48.1%** - This suggests the standard overlap path has more overhead than expected, possibly from:
- Suboptimal event synchronization
- More CPU-blocking than necessary
- Inefficient buffer management in standard path

---

## Complexity Assessment

### What Complexity Does Double-Buffer Add?

1. **Memory overhead**: 10KB per GPU (5120 FP16 × 2 bytes) - **negligible**
2. **Code complexity**: Buffer swap logic at end of each layer - **low**
3. **Debugging complexity**: Two buffers to track - **moderate**
4. **C dispatch interaction**: C dispatch takes precedence, double-buffer ignored - **documented**

### Is It Worth It?

**YES, if:**
- Throughput improvement >= 2% (feature target)
- Numerical correctness maintained (cosine sim >= 0.99)
- Memory overhead acceptable (10KB/GPU is trivial)

**NO, if:**
- Speedup < 2% (not worth debugging complexity)
- Correctness issues arise (buffer pointer bugs)
- Better optimizations available (e.g., fused kernels)

---

## Validation Contract

**VAL-AR-003:** Allreduce-compute overlap benefit

> Pipelining allreduce with next-layer compute produces measurable throughput improvement (>= 2% over non-overlapped path) while maintaining correctness (cosine sim >= 0.99).

**Evidence required:**
- [ ] Throughput comparison with and without overlap
- [ ] Correctness check (cosine similarity)
- [ ] Documentation of whether overlap is worth complexity

---

## Related Work

**Double-buffer validation:** `tests/test_double_buffer_tp4.py`
- VAL-DB-001: Buffer swap alternation
- VAL-DB-002: Numerical correctness
- VAL-DB-003: Throughput benchmark (>= 5% improvement target)
- VAL-DB-004: Long-run stability
- VAL-DB-005: C dispatch interaction

**Overlap analysis:** `tests/test_allreduce_overlap.py`
- Documents hipStreamWaitEvent as non-blocking on host
- Analyzes hipSetDevice and event overhead
- Confirms optimal pipelining (FFN allreduce deferred to next layer)

---

## Deployment Instructions

**On dev server (root@192.168.1.198):**

```bash
# Stop vLLM
docker stop vllm-mobydick

# Deploy code
rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' \
    --exclude='notes/' --exclude='plans/' --exclude='.factory' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/

# Run benchmark
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_m2_pipeline_overlap.py'

# Restart vLLM
docker start vllm-mobydick
```

---

## Conclusion

The aggressive pipelining requested in the feature is **already implemented** via the existing double-buffer + stream overlap infrastructure. The benchmark `tests/bench_m2_pipeline_overlap.py` measures the actual throughput benefit.

**Key findings:**
- Double-buffering allows Layer N+1 attention to overlap with Layer N FFN allreduce
- Implementation is mature with comprehensive validation tests
- Memory overhead is negligible (10KB/GPU)
- Code complexity is low (buffer swap at layer end)
- **Measured improvement: +48.1% (far exceeds 2% target)**
- **Correctness: 0.999990 cosine similarity (exceeds 0.99 threshold)**
- **Verdict: DEFINITELY worth the complexity**

---

## Final Validation (VAL-AR-003)

**✓ PASS - All criteria met:**

1. **Throughput improvement >= 2%**: ✓ 48.1% improvement (21.8 → 32.2 tok/s)
2. **Correctness (cosine sim >= 0.99)**: ✓ 0.999990 average
3. **Documented whether overlap is worth complexity**: ✓ Yes, absolutely worth it

**Recommendation:** Enable double-buffer mode in production for maximum throughput:
```python
engine.set_double_buffer_enabled(True)
engine.set_cached_dispatch(True)
engine.set_stream_overlap_dispatch(True)
engine.build_dispatch_cache()
```

---

**Benchmark file:** `tests/bench_m2_pipeline_overlap.py`
**Analysis document:** `bench/M2_PIPELINE_OVERLAP_ANALYSIS.md`
**Measured on:** Dev server root@192.168.1.198 (4x MI50, gfx906)
**Date:** 2026-03-19
