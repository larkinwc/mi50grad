# Batch Decode

Design notes for batch=1-4 decode with dynamic GEMV/GEMM switching.

## GEMV vs GEMM Transition

- batch=1: GEMV (matrix-vector multiply). Existing kernels: gemv_int4_v6, gemv_int4_v5_awq
- batch>=2: GEMM (matrix-matrix multiply). Existing kernels: gemm_int4_prefill_v2, gemm_fp16_prefill

### Kernel Mapping for Batch Decode

| Operation | batch=1 (GEMV) | batch>=2 (GEMM) |
|-----------|----------------|-----------------|
| Q proj (INT4) | gemv_int4_v6 (or dual) | gemm_int4_prefill_v2 |
| K proj (INT4) | gemv_int4_v6 (or dual) | gemm_int4_prefill_v2 |
| V proj (INT4) | gemv_int4_v6 (or dual) | gemm_int4_prefill_v2 |
| O proj (FP16) | gemv_fp16_v2 | gemm_fp16_prefill |
| FFN gate (INT4) | gemv_int4_dual | gemm_int4_prefill_v2 |
| FFN up (INT4) | (fused with gate) | gemm_int4_prefill_v2 |
| FFN down (INT4) | gemv_int4_v6 / fused | gemm_int4_prefill_v2 |

## Dimension Mapping

For Qwen3.5-27B with TP=4:
- hidden_size = 5120
- intermediate_size = 17408
- num_attention_heads = 40 (10 per GPU)
- num_kv_heads = 8 (2 per GPU)
- head_dim = 128

GEMM call for Q-proj batch=B:
- M = B (batch size)
- K = 5120 (hidden_size, but each GPU does K=5120, N=5120/4=1280 for TP)
- N = 1280 (local Q output: num_heads_per_gpu * head_dim = 10 * 128)

## Allreduce Scaling

- batch=1: allreduce 5120 FP16 elements = 10KB
- batch=2: allreduce 10240 FP16 elements = 20KB
- batch=4: allreduce 20480 FP16 elements = 40KB

With compressed allreduce (INT8):
- batch=4 compressed: ~21.8KB (vs 40KB uncompressed)

## KV Cache Multi-Position Write

KV cache layout: [num_layers, max_seq, local_kv_heads, head_dim]
For batch=B tokens at positions [p0, p1, ..., pB-1]:
- Write K[layer, pi, :, :] for each i in [0, B)
- Write V[layer, pi, :, :] for each i in [0, B)

For contiguous positions (common case: p, p+1, ..., p+B-1):
- Single memcpy of B * kv_stride bytes to cache[layer, p, :, :]

## Multi-Query Attention

For batch decode, we have B query positions and kv_len KV positions.
- Q shape: [B, num_heads_per_gpu, head_dim]
- K cache: [kv_len + B, num_kv_heads_per_gpu, head_dim]  (after appending)
- V cache: [kv_len + B, num_kv_heads_per_gpu, head_dim]

Option A: Call FlashAttention B times with seq_len=1 each (simplest but no batching benefit)
Option B: Use prefill attention kernel with M=B (multi-query positions)
Option C: New batch-decode attention kernel

The existing flash_attn_256_tuned.hip handles variable seq_len for prefill.
For batch decode, each query position has a DIFFERENT effective kv_len:
- Query at position p sees kv_len = p+1
- Query at position p+1 sees kv_len = p+2

This complicates batching -- the simplest correct approach is Option A (loop B times).
For throughput, attention is a small fraction of total time, so this is acceptable initially.

## C Dispatch Integration

CDispatchPlan needs batch_size field.
The C loop logic:
- if batch_size == 1: use existing GEMV path (no change)
- if batch_size >= 2: use GEMM kernels, adjust grid dimensions, adjust allreduce num_elems

## Expected Performance

- GEMV at batch=1: ~172us per layer (compute portion)
- GEMM at batch=4: ~250us per layer (4 tokens, ~62us amortized per token)
- Allreduce at batch=4 uncompressed: 64 * 79us * (40KB/10KB scaling factor) ~ maybe 64 * 120us = 7.7ms
- Allreduce at batch=4 compressed: ~64 * 65us = 4.2ms (reduced by compression)
- Expected batch=4 compressed throughput: 4 tokens / (total_time) ≈ 60+ tok/s aggregate
