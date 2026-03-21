# Future Research: Performance Improvement Opportunities

**Current:** ~54 tok/s (TP=4, 4x MI50 gfx906, Qwen3.5-27B-GPTQ-Int4)
**Current gap:** ~6 tok/s to 60 tok/s target (~11% improvement needed)

---

## Tried and Ruled Out (see RESEARCH.md for details)

| Approach | Result | Why |
|---|---|---|
| INT8-compressed allreduce | −15.8% regression | Two-phase kernel overhead > bandwidth savings for 10KB payloads |
| Batch decode (M=2-4) | No improvement / OOM | GEMM tiles underutilized at small M, allreduce scales linearly |
| Persistent megakernel | Not viable | Skeleton code, would need full rewrite for ~0.5-1ms savings |
| KV Cache INT8 | Not impactful | Attention is compute-bound at kv_len<256; KV reads complete in ~3µs |
| Assembly-optimized GEMV | Diminishing returns | v7/v8 already eliminated FP16 conversion overhead; remaining bottleneck is HBM2 bandwidth |
| GEMV v7/v8 register blocking | Marginal e2e (+0.5 tok/s) | Standalone GEMV not used in C dispatch hot path; hot-path kernels already optimized |
| Allreduce v2 kernel optimization | No improvement | BAR1 P2P latency dominates, not kernel efficiency |

---

## Remaining Viable Approaches

### 1. Batched Speculative Verification with GEMM (Highest potential)
**Estimated gain:** ~1.5x effective throughput → ~80 tok/s

Verify K draft tokens in a single GEMM call with a single allreduce for all K tokens, rather than K separate decode steps each with 64 allreduces. N-gram acceptance already validated at 54% overall (59% code, 87% repetitive). The key insight: amortize allreduce across multiple tokens.

**Why this is promising:** The profiled time breakdown shows the remaining 15.75ms is dominated by per-token GEMV compute + 2.94ms allreduce. Speculative verification would amortize the allreduce cost across K accepted tokens, effectively reducing the ~46µs-per-call × 64-calls overhead.

**Implementation path:**
- Extend batch decode to handle speculative token sequences (K draft tokens)
- Use GEMM (not GEMV) for K×1 verification pass
- Single allreduce after GEMM verification (not K separate allreduces)
- Replace character-level tokenization with BPE/sentencepiece for better JSON/conversational acceptance

**Prerequisites:** Working batch decode with GEMM (batch=2-3 tested but showed no gain; speculative case is different because all K tokens share the same KV cache prefix).

**Complexity:** High

---

### 2. Layer Pruning / Early Exit (LayerSkip)
**Estimated gain:** ~1.3 ms/tok allreduce savings if skipping 16 layers → ~57 tok/s

Meta's LayerSkip shows many tokens can be generated using only 50-70% of layers with full-model verification. Skipping 16 of 64 layers saves 16 x ~46µs = 0.74ms allreduce + proportional compute (~4ms).

**Caveat:** Requires model fine-tuning with layer dropout. Qwen3.5-27B was NOT trained with LayerSkip.

**Reference:** arxiv.org/abs/2404.16710

**Complexity:** Very High (requires model fine-tuning)

---

### 3. Mixed-Precision Per-Layer Quantization
**Estimated gain:** ~1-2 tok/s (+2-4%)

Not all layers are equally sensitive. Later transformer layers tolerate more aggressive quantization (INT2/INT3). Use sensitivity analysis to assign per-layer bit widths, with INT4 for critical layers and INT2-3 for tolerant ones. Reduced weight size directly reduces HBM2 bandwidth consumption (the primary bottleneck).

**Complexity:** High (requires sensitivity analysis + per-layer kernel variants)

**Reference:** SqueezeLLM, AQLM (arxiv.org/abs/2401.06118)

---

### 4. Activation Quantization (W4A8/W8A8)
**Estimated gain:** ~1-2 tok/s (+2-4%)

Quantize activations to INT8 for GEMV/GEMM. `activation_quant.hip` already exists as starting point. W8A8 GEMM kernels can use `v_dot2_i32_i16` instruction on gfx906 for 2x throughput on INT8×INT8 dot products.

**Implementation path:**
- Dynamic per-tensor activation quantization (already implemented in `activation_quant.hip`)
- INT8 GEMV kernel variant
- Calibration pass to verify acceptable accuracy loss

**Complexity:** Medium-High

---

*Document updated: 2026-03-21*
*Previous Tier 1 approaches (compressed allreduce, batch decode) have been tried and ruled out.*
*GEMV kernel optimizations (v7, v8, dual 4x blocking) have been applied with marginal e2e gain.*
*Primary remaining bottleneck: HBM2 memory bandwidth for INT4 GEMV weight streaming.*
