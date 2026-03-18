# Deferred Attention Allreduce (M3 Optimization)

## Overview

Deferred attention allreduce is an optimization that reduces the allreduce count from 128 to 64 per decode step in TP=4 inference.

## Technique

**Standard flow (2 allreduces per layer):**
1. RMSNorm(d_hidden) → attention → proj_out (partial)
2. ALLREDUCE(proj_out) → d_hidden += attn_result_global
3. RMSNorm(d_hidden) → FFN → ffn_out (partial)
4. ALLREDUCE(ffn_out) → d_hidden += ffn_out_global

**Deferred flow (1 allreduce per layer):**
1. RMSNorm(d_hidden) → attention → proj_out (partial)
2. d_hidden += proj_out (LOCAL residual add, no allreduce)
3. RMSNorm(d_hidden) → FFN → ffn_out (partial)
4. ALLREDUCE(ffn_out) → d_hidden += ffn_out_global

## Mathematical Considerations

The FFN gate projection uses SiLU activation: `gate = SiLU(x @ W_gate)`

Since SiLU is non-linear, operating on partial x produces different results than operating on fully reduced x. This is an **approximation** that changes the computation graph.

For TP=4 with FP16 precision, cosine similarity >= 0.99 is expected. Always validate correctness before using in production.

## Usage

```python
# Enable deferred attention allreduce
engine.set_deferred_attention_ar(True)
engine.set_cached_dispatch(True)
engine.set_stream_overlap_dispatch(True)

# Run decode step (will use 64 allreduces instead of 128)
output = engine.decode_step(embedding, position)
```

## Requirements

- `elementwise_v2.so` must be built (provides `residual_add_v2` kernel)
- C dispatch must be enabled (`set_cached_dispatch(True)`)
- Works with both full attention and DeltaNet layer types

## Performance

- Allreduce reduction: 50% (128 → 64)
- Expected throughput improvement: 10-30%
- Measured: 1.27x speedup (34.06 vs 26.82 tok/s)

## Files

- `src/runtime/c_dispatch.c`: C dispatch loop with deferred AR logic
- `src/inference/tp_engine.py`: Python integration and API
- `src/kernels/elementwise_v2.hip`: residual_add_v2 kernel
- `tests/val_m3_reduced_ar_count.py`: Validation test

## Limitations

- Suitable for inference only (not training)
- Cosine similarity should be validated for each model/architecture combination
- Requires TP=2 or TP=4 configuration
