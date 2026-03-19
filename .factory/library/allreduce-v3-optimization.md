# Allreduce V3 Kernel Optimization

**Date:** 2026-03-19  
**Feature:** m2-optimize-allreduce-pcie4  
**Source:** `M2_PCIE4_V3_KERNEL_SUMMARY.md`

---

## Key Insight: 10KB Allreduce is LATENCY-BOUND

From PCIe4 bandwidth profiling (`bench/PCIe4_BANDWIDTH_REPORT.md`):

- **Kernel launch overhead:** 12.20 µs (fixed cost)
- **10KB payload latency:** 20.37 µs (all overhead, zero transfer time)
- **Vectorization provides NO benefit** for small payloads
- **Access pattern is irrelevant** (sequential == strided == random)

## V3 Optimizations

1. **Single-wavefront design** (64 threads instead of 128/256)
2. **Zero `__syncthreads()` barriers** (vs 2 in v2/v1)
3. **No LDS usage** - pure register + `__shfl` reduction
4. **80 elements/thread** (perfect fit: 64 × 80 = 5120)
5. **Streamlined code path** - minimal instruction overhead

## Files

- **Kernel:** `src/kernels/kernel_p2p_allreduce_rmsnorm_v3.hip`
- **Tests:** `tests/test_kernel_p2p_allreduce_v3.py`
- **Benchmark:** `tests/bench_allreduce_v3.py`
- **Summary:** `M2_PCIE4_V3_KERNEL_SUMMARY.md`

## Performance Targets

- **v1 baseline:** ~79 µs
- **v2 optimized:** ~75 µs
- **v3 target:** ≤ 40 µs
- **Expected speedup:** 1.9-2.0×

## Expected E2E Impact

- **Allreduce time:** 4.8 ms → 2.6 ms/token
- **Savings:** 2.2 ms/token
- **Throughput gain:** 51.72 → 57.7 tok/s (11.5%)

## How to Build

```bash
# On dev server
make hip_kernels
# Builds: build/kernels/kernel_p2p_allreduce_rmsnorm_v3.so
```

## How to Test

```bash
# Correctness + latency
python3 tests/test_kernel_p2p_allreduce_v3.py

# Microbenchmark comparison
python3 tests/bench_allreduce_v3.py
```

## Integration

The v3 kernel is automatically loaded by `tp_engine.py` with priority:
1. v3 (single-wavefront)
2. v2 (optimized)
3. v1 (baseline)

No manual configuration needed - fallback ensures compatibility.

## Lessons Learned

### What Doesn't Help (Latency-Bound)
❌ Vectorized loads (float4, dwordx4)  
❌ Memory coalescing optimization  
❌ Access pattern optimization  
❌ Kernel micro-optimizations (thread count, LDS usage)

### What Does Help
✅ Reduce kernel launch overhead  
✅ Minimize synchronization barriers  
✅ Single-wavefront design  
✅ Kernel fusion  
✅ Persistent kernels  
✅ Async operations  
✅ Reduced synchronization

---

**References:**
- `bench/PCIe4_BANDWIDTH_REPORT.md` - Profiling data
- `src/kernels/kernel_p2p_allreduce_rmsnorm_v3.hip` - Implementation
- `M2_PCIE4_V3_KERNEL_SUMMARY.md` - Full documentation
