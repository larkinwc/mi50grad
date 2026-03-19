# Fused GEMV+AR+RMSNorm Kernel Patterns

Patterns for fused kernel implementation and cross-WG coordination.

## What belongs here

Fused kernel design patterns, cross-WG synchronization, atomic counter mechanisms, and performance considerations.

---

## Cross-WG Coordination Pattern

**Problem**: GEMV uses multiple workgroups (16 columns per WG), but RMSNorm needs global sum-of-squares across ALL columns.

**Solution**: Atomic completion counter with last-WG reduction.

```cpp
// Global memory atomic counter
__device__ unsigned int completion_counter = 0;

// Each workgroup:
// 1. Compute GEMV partial results
// 2. Compute local sum-of-squares
// 3. Atomically increment counter
// 4. Last WG performs global reduction

__shared__ float s_local_sum_sq;
s_local_sum_sq = local_sum_sq;

// Atomic increment
__threadfence(); // Ensure writes visible
unsigned int my_id = atomicInc(&completion_counter, num_wgs - 1);

if (my_id == num_wgs - 1) {
    // Last WG: sum all partial sums
    // Use LDS to accumulate, broadcast result
}

__syncthreads();
// All WGs read broadcast result
```

## Performance Targets

- Baseline (separate kernels): 51.75 tok/s
- Target (fused kernel): 55+ tok/s
- Kernel launch reduction: 192 -> 64 per token

## Numerical Requirements

- FP32 accumulation mandatory on gfx906
- cosine_sim >= 0.99 vs reference path
- max_abs_error < 5e-3

---

## Known Issues

1. **71% regression** when cross-WG sync missing
2. **Double-counting** if hidden input not correctly handled
3. **Precision drift** if FP16 accumulation used
