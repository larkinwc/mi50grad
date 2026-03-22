# Weight Prefetch Patterns

Weight prefetch during allreduce window for HBM memory access optimization.

**What belongs here:** hipMemcpyAsync patterns, L2 cache constraints, prefetch field conventions.

---

## hipMemcpyAsync dst=src Pattern

To trigger memory controller activity without actual data movement, use `hipMemcpyAsync` with `dst=src`:

```c
// Forces a read from src address, warming memory controller
hipMemcpyAsync_fn(src, src, bytes, hipMemcpyDeviceToDevice, compute_stream);
```

**Effect:** Triggers memory reads from the source address which can warm the memory controller but may not effectively fill L2 cache since no actual data is written anywhere.

**Production alternative:** Allocate a ~4MB scratch buffer (fits in MI50 L2) and copy first tile of largest weights to actually capture prefetched data in L2.

## MI50 L2 Cache Constraints

- **L2 cache size:** ~4MB per GPU
- **Typical weight sizes per layer:**
  - QKV fused: ~3.6MB INT4
  - FFN gate: ~11MB INT4
  - FFN up: ~11MB INT4
  - Total: ~25MB per layer

**Implication:** Only partial prefetch is feasible. Full weight prefetch exceeds L2 capacity.

**Current implementation:** Prefetches entire weight matrices (~25MB total). May still provide benefit from memory controller warming even if L2 overflow occurs.

## Adding C Struct Fields Pattern

When adding new fields to CEngineLayerSpec or CAllreduceSpec:

1. **C struct:** Add field to struct in `c_dispatch.c`
2. **Python struct:** Add matching ctypes field with same order and type
3. **Initialization:** Set to 0/NULL in fill function
4. **Enable/disable method:** Create method that invalidates and rebuilds dispatch plan

Example from weight prefetch:
```python
# In TPInferenceEngine
def set_weight_prefetch(self, enabled: bool):
    self._weight_prefetch_enabled = enabled
    self._build_dispatch_plan()  # Rebuild with prefetch fields
```

## Prefetch Timing

- **FFN allreduce window:** ~46us per layer (64 layers = 2.94ms per token)
- **Compute stream idle:** During allreduce on AR stream
- **Prefetch target:** Next layer's QKV + FFN gate+up weights

The compute stream waits for AR done event at the next layer start, allowing prefetch to overlap.
