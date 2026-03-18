# TP Prefill Implementation Patterns

## Overview

Tensor-parallel prefill for Qwen3.5 27B uses column-parallel and row-parallel projections with P2P allreduce for aggregation.

## Column-Parallel Projections

Used for QKV, FFN gate, and FFN up projections:
- Input: full hidden dimension [seq_len, hidden_size]
- Output: sharded across GPUs [seq_len, hidden_size/tp_size]
- Each GPU computes its slice of output dimensions independently

## Row-Parallel Projections

Used for O projection and FFN down projection:
- Input: sharded across GPUs [seq_len, hidden_size/tp_size]
- Output: full hidden dimension [seq_len, hidden_size]
- Each GPU computes partial output, then allreduce sums across GPUs

## Batched vs Per-Token

TP prefill uses two paths based on sequence length:
- `seq_len >= 32`: Batched GEMM path (more efficient)
- `seq_len < 32`: Per-token GEMV fallback (lower latency for short sequences)

## Allreduce Fallback Chain

```python
if self._ring_allreduce and self._ring_ar is not None:
    self._ring_ar.allreduce_residual(partial_ptrs, hidden_ptrs, size)
elif self._fused_p2p_reduce and self._fused_p2p_ar is not None:
    self._fused_p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, size)
else:
    self._p2p_ar.allreduce_residual(partial_ptrs, hidden_ptrs, size)
```

## GQA with KV Cache Sharding

For Qwen3.5 with GQA (32 Q heads, 4 KV heads, TP=4):
- Each GPU: 8 Q heads, 1 KV head
- KV cache naturally sharded - each GPU stores its local KV heads
- FlashAttention is completely local (no cross-GPU attention needed)
- O projection allreduce aggregates partial outputs

This is a key insight: GQA with TP=4 means attention requires no cross-GPU communication!

## Validation

Run on dev server:
```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/opt/mi50grad \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/val_m4_tp_prefill.py && python3 tests/val_m4_tp_flash_attention.py'
```

## Files

- `src/inference/tp_engine.py`: TP prefill implementation
  - `prefill_step()`: Main entry point
  - `_prefill_full_attention_tp()`: Column-parallel QKV, row-parallel O
  - `_prefill_ffn_tp()`: Column-parallel gate/up, row-parallel down
- `tests/val_m4_tp_prefill.py`: GEMM validation
- `tests/val_m4_tp_flash_attention.py`: FlashAttention validation
