# Dispatch Optimization

C dispatch loop overhead reduction techniques for TP4 decode.

**What belongs here:** hipSetDevice optimization, kernel launch reduction, dispatch restructuring.

---

## hipSetDevice Overhead Analysis

Current c_dispatch_step() with deferred AR + gemv_fused (optimized path):

**Per layer (64 layers), 4 engines:**
- Attention kernel loop: 4x hipSetDevice (one per engine_idx)
- FFN kernel loop: 4x hipSetDevice (one per engine_idx)
- wait_for_allreduce (prev layer): 4x hipSetDevice
- do_allreduce_gemv_fused:
  - Event record loop (reverse): 4x hipSetDevice
  - Stream wait GPU0: 0 (already on last GPU from reverse loop, but then needs GPU0)
  - Stream wait GPUs 1-3: 3x hipSetDevice
  - Kernel launch loop: 4x hipSetDevice
  - Done event record (reverse): 4x hipSetDevice
- Total per layer: ~27 hipSetDevice calls
- Total per token: ~27 x 64 = ~1728 calls

**Optimization strategies:**
1. Cache `current_device` in a local int, skip hipSetDevice when unchanged
2. Restructure allreduce inner loops to minimize device switching (batch all GPU0 ops, then GPU1, etc.)
3. The reverse iteration pattern (for i = tp-1; i >= 0; i--) in event record means we end on GPU0, avoiding one extra switch for GPU0's stream wait

## Fused QKV GEMV Design

**Current attention GEMV launches per engine per layer:**
- gemv_q_fused: [5120, 1280] -> Q buffer (10 heads x 128 head_dim)
- gemv_k_only: [5120, 256] -> K working buffer (2 kv_heads x 128)
- gemv_v_cache: [5120, 256] -> V cache position directly

**Fused approach:**
- Concatenate Q, K, V weights: [5120, 1792] (1280 + 256 + 256)
- Single GEMV launch writes 1792 output columns
- Thread mapping: same as existing v8 kernel but wider output
- Output routing: threads 0-1279 write to Q buffer, 1280-1535 to K buffer, 1536-1791 to V buffer (or cache position)
- Scales/zeros also concatenated with matching group structure
- **V direct cache write:** The V portion of the output must write to kv_cache position (not intermediate buffer). This requires the kernel to know the V output offset (1536) and redirect those writes to the cache pointer.

**Weight concatenation in engine.py:**
During weight loading, concatenate q_proj, k_proj, v_proj qweight/scales/zeros:
```python
qkv_qweight = torch.cat([q_qweight, k_qweight, v_qweight], dim=1)  # [K/8, N_total]
qkv_scales = torch.cat([q_scales, k_scales, v_scales], dim=1)
qkv_zeros = torch.cat([q_zeros, k_zeros, v_zeros], dim=1)
```

**C dispatch changes:**
- Add `CKernelSpec gemv_qkv_fused` to CEngineLayerSpec
- In attention block: if gemv_qkv_fused.present, launch it instead of separate Q/K/V launches
- Still need to update V output pointer for cache position (same mechanism as current gemv_v_cache)
