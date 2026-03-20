# Future Research: Performance Improvement Opportunities

**Baseline:** 53.74 tok/s (TP=4, 4x MI50 gfx906, Qwen3.5-27B-GPTQ-Int4)
**Current gap:** 6.26 tok/s to 60 tok/s target (11.6% improvement needed)

---

## Tier 2: Medium ROI (1-3 tok/s each)

### 1. KV Cache INT8 Quantization
**Estimated gain:** ~0.5-1.0 ms/tok → ~55-56 tok/s (+2-4%)

Store KV cache in INT8 (per-head, per-token asymmetric quantization) instead of FP16, halving cache memory bandwidth during attention decode. FlashAttention decode kernel reads KV cache from HBM; at kv_len=256+, cache reads are a meaningful fraction of attention time.

**Implementation path:**
- Add quantize-on-write to KV cache append in `c_dispatch.c` (after qknorm_rope)
- Modify `flash_attn_256_tuned.hip` to accept INT8 KV inputs with per-head scale/zero
- `activation_quant.hip` already exists as starting point
- Well-studied technique: PyTorch, vLLM, LMDeploy all support it

**Reference:** arxiv.org/abs/2601.04719 (GPU-Accelerated INT8 KV Cache Compression)

**Complexity:** Medium

---

### 2. v_dot2_f32_f16 Assembly-Optimized GEMV
**Estimated gain:** ~0.5-1.0 ms/tok → ~55-56 tok/s (+2-5%)

Hand-tuned GCN5.1 assembly GEMV kernel. Current GEMV v6 uses `__builtin_amdgcn_fdot2` but the INT4 dequantization path (8x `ubfe` + 8x float mul + 8x `float2half` + half2 pack) is suboptimal. Assembly version can:
- Use `v_cvt_f32_ubyte*` for faster INT4→FP32 conversion
- Explicit register scheduling to hide HBM latency
- 8x register block loading (ILP improvement per llama.cpp-gfx906 fork approach)
- Strategic LDS padding (48B KV, 32B Q) for bank conflict elimination

**Reference:** github.com/eslowney/llama.cpp-gfx906 (gfx906-asm-kernels.s)

**Complexity:** High (requires GCN5.1 ISA expertise)

---

### 3. Persistent Megakernel Validation
**Estimated gain:** ~0.5-1.0 ms/tok → ~55-56 tok/s (+1-2%)

`persistent_decode.hip` and `persistent_dispatch.py` already exist but have never been validated on real hardware. Eliminates all remaining host-side kernel launch overhead. Less impactful now that fused kernel reduced launches from 192→64, but could still save ~0.5-1.0ms from eliminating the remaining 64 launches + C dispatch loop overhead.

**Implementation path:** Validate existing code on dev server, debug as needed.

**Complexity:** Low to validate, High to debug

---

## Tier 3: Speculative / Longer-term

### 4. Batched Speculative Verification with GEMM
**Estimated gain:** ~1.5x effective throughput → ~80 tok/s (requires batch decode as prerequisite)

Verify K draft tokens in a single GEMM call with a single allreduce for all K tokens, rather than K separate decode steps each with 64 allreduces. N-gram acceptance already validated at 54% overall (59% code, 87% repetitive). The key insight: amortize allreduce across multiple tokens.

**Prerequisites:** Batch decode (GEMV→GEMM transition) must be implemented first.

**Complexity:** High

---

### 5. Layer Pruning / Early Exit (LayerSkip)
**Estimated gain:** ~1.3 ms/tok allreduce savings if skipping 16 layers → ~57 tok/s

Meta's LayerSkip shows many tokens can be generated using only 50-70% of layers with full-model verification. Skipping 16 of 64 layers saves 16 x 79µs = 1.3ms allreduce + proportional compute.

**Caveat:** Requires model fine-tuning with layer dropout. Qwen3.5-27B was NOT trained with LayerSkip.

**Reference:** arxiv.org/abs/2404.16710

**Complexity:** Very High (requires model fine-tuning)

---

### 6. Mixed-Precision Per-Layer Quantization
**Estimated gain:** ~1-2 tok/s (+2-4%)

Not all layers are equally sensitive. Later transformer layers tolerate more aggressive quantization (INT2/INT3). Use sensitivity analysis to assign per-layer bit widths, with INT4 for critical layers and INT2-3 for tolerant ones.

**Complexity:** High (requires sensitivity analysis + per-layer kernel variants)

**Reference:** SqueezeLLM, AQLM (arxiv.org/abs/2401.06118)

---

*Document generated: 2026-03-20*
*To be revisited after Tier 1 optimizations (inline-compressed allreduce + batch decode) are complete.*
