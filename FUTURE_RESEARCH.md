# Future Research: Performance Improvement Opportunities

**Current:** ~54 tok/s decode (TP=4, 4x MI50 gfx906, Qwen3.5-27B-GPTQ-Int4)
**Current gap:** ~6 tok/s to 60 tok/s target (~11% improvement needed)
**Theoretical bandwidth floor:** ~3.1ms/tok (2.63GB weights / 860 GB/s peak HBM2)
**Actual:** ~18.5ms/tok → **6x gap** between theoretical and actual (significant room)

---

## Tried and Ruled Out (see RESEARCH.md for details)

| Approach | Result | Why |
|---|---|---|
| INT8-compressed allreduce | -15.8% regression | Two-phase kernel overhead > bandwidth savings for 10KB payloads |
| Batch decode (M=2-4) | No improvement / OOM | GEMM tiles underutilized at small M, allreduce scales linearly |
| Persistent megakernel | Not viable | Skeleton code, would need full rewrite for ~0.5-1ms savings |
| KV Cache INT8 | Not impactful | Attention is compute-bound at kv_len<256; KV reads complete in ~3us |
| Assembly-optimized GEMV | Diminishing returns | v7/v8 already eliminated FP16 conversion overhead; remaining bottleneck is HBM2 bandwidth |
| GEMV v7/v8 register blocking | Marginal e2e (+0.5 tok/s) | Standalone GEMV not used in C dispatch hot path; hot-path kernels already optimized |
| Allreduce v2 kernel optimization | No improvement | BAR1 P2P latency dominates, not kernel efficiency |
| Double-buffer pipeline overlap | -9.3% degradation | Buffer copy + swap overhead > overlap benefit |
| Flat speculative decode (n-gram/EAGLE) | ~0% throughput gain | High acceptance (54%) but allreduce still dominates per-token cost |

---

## Tier 1: Highest Potential (Recommended Next)

### 1. Sequoia Tree-Based Speculative Verification (Enhancement to batched spec decode)
**Estimated gain:** 20-40% effective throughput -> ~65-75 tok/s
**Complexity:** High

Sequoia (NeurIPS 2024) uses dynamic programming to find the optimal verification tree structure for specific hardware. Unlike our flat n-gram approach, a tree structure maximizes E[accepted_tokens] / verification_cost, where cost is dominated by allreduce latency.

**Why this is different from what we tried:** Our current speculative decode runs K separate decode steps each with 64 allreduces. Sequoia verifies an entire tree of draft tokens in a SINGLE forward pass with a single set of allreduces. The DP algorithm accounts for:
- Hardware-specific allreduce latency (46us per call for us)
- Per-position acceptance probability (our measured 54% n-gram acceptance)
- Tree depth/breadth tradeoff to maximize tokens accepted per verification step

**Implementation path:**
- Implement Sequoia's DP tree optimizer parameterized with our hardware costs
- Build tree attention mask for single-pass verification (GEMM not GEMV, M=tree_size)
- Single allreduce set for entire tree verification (64 calls total, not 64*K)
- Combine with improved n-gram or EAGLE draft generation
- Consider BPE/sentencepiece tokenization for better JSON/conversational acceptance

**Key insight:** With 54% per-token acceptance, optimal tree of depth 4-5 with branching factor 2-3 can yield ~3 accepted tokens per verification step on average, giving ~3x effective throughput for the same allreduce cost.

**Reference:** arxiv.org/abs/2402.12374, github.com/Infini-AI-Lab/Sequoia

---

### 2. Weight Prefetch Overlap with Allreduce (PRESERVE-style)
**Estimated gain:** 0.5-1.5 ms/tok -> ~56-58 tok/s
**Complexity:** Medium

PRESERVE (2025) demonstrates overlapping weight prefetch from HBM to L2 cache with collective communication, achieving near-theoretical bandwidth utilization. During our ~2.94ms allreduce window per token, GPU compute units and the HBM memory controller are largely idle.

**Approach:**
- During allreduce (on AR stream), issue async prefetch of next layer's weights on compute stream
- Use `__builtin_nontemporal_load` hints or explicit L2 prefetch intrinsics
- Next layer's GEMV kernel then hits L2 instead of HBM for weight reads
- MI50 has ~4MB L2 per GPU; per-layer weight data is ~42MB, so prefetch the hottest portion (e.g., first tile of gate+up weights = ~11MB, won't fit entirely but partial prefetch still helps)

**Key analysis:** The 6x gap between theoretical bandwidth floor (3.1ms) and actual (18.5ms) suggests significant memory access inefficiency. Even modest L2 hit rate improvement from prefetching could recover 0.5-1.5ms.

**Alternative framing:** Even without L2 prefetch, the idle compute stream during allreduce could "warm up" the HBM memory controller for next-layer addresses, reducing first-access latency spikes.

**Reference:** arxiv.org/abs/2501.08192

---

### 3. Fused QKV Attention Projection
**Estimated gain:** 0.3-0.5 ms/tok -> ~55-56 tok/s
**Complexity:** Medium

Current C dispatch launches 2-3 separate attention GEMV kernels per engine per layer (gemv_q_fused + gemv_kv_fused, or gemv_q + gemv_k_only + gemv_v_cache). Concatenating Q/K/V projection weights into a single matrix and using one fused GEMV launch would:
- Reduce kernel launches by 64-128 per token (on top of existing 64)
- Improve weight streaming efficiency: one large sequential HBM read vs 2-3 smaller reads with launch gaps
- Eliminate per-kernel launch overhead (~10us per launch × 64-128 saved launches)

**Implementation:**
- Concatenate Q, K, V weight matrices along output dimension: [5120, 1280+256+256=1792] per GPU
- Single fused GEMV kernel that writes Q, K, V outputs to separate buffers
- Update C dispatch to skip separate Q/KV launches

**Why this wasn't done before:** The fused GEMV+AR+RMSNorm work was focused on FFN down-proj. The attention projections were left as separate kernels. This is a lower-risk fusion (no cross-WG coordination needed, just a wider output).

---

### 4. Dispatch Restructuring: hipSetDevice Reduction
**Estimated gain:** 0.2-0.4 ms/tok -> ~55 tok/s
**Complexity:** Low

The C dispatch loop currently calls hipSetDevice ~20 times per layer x 64 layers = ~1280 calls per token. At ~0.6us each (measured), that's ~0.77ms/tok of pure overhead.

**Optimizations:**
a) **Track current device:** Cache last-set device, skip hipSetDevice when unchanged. Saves calls within allreduce functions where the same GPU is set multiple times.
b) **Batch operations per GPU:** Instead of interleaving GPU0/1/2/3 for each kernel, launch all GPU0 attention kernels, then all GPU1 attention kernels, etc. This is valid because attention kernels are independent across GPUs within a phase (all run on NULL stream, async launch).
c) **Reduce allreduce hipSetDevice calls:** The allreduce functions (do_allreduce_gemv_fused etc.) have heavy hipSetDevice cycling through 4 GPUs multiple times. The event record + stream wait loops can be restructured.

---

## Tier 2: Medium Potential

### 5. Batched Speculative Verification with GEMM (original approach)
**Estimated gain:** ~1.5x effective throughput -> ~80 tok/s
**Complexity:** High

Verify K draft tokens in a single GEMM call with a single allreduce for all K tokens. Subsumed by approach #1 (Sequoia) which adds optimal tree structure on top of this idea.

**Note:** Approach #1 (Sequoia) is the better version of this. Consider implementing #1 directly rather than flat batched verification.

---

### 6. Compute-Communication Overlap for GEMV (TokenWeave-style)
**Estimated gain:** 0.5-1.0 ms/tok -> ~56-57 tok/s
**Complexity:** High

TokenWeave (MLSys 2026, Microsoft) splits work into subsets and overlaps computation of one subset with communication of another. For batch=1 decode, adapt by splitting GEMV output columns:

- Split FFN down GEMV (N/TP=1280 columns) into two halves (640 each)
- Launch first-half GEMV, start its partial allreduce
- Overlap: launch second-half GEMV while first-half allreduce runs
- Then allreduce second half

This partially hides allreduce latency behind GEMV compute. Different from our failed double-buffer attempt (which was about hidden state buffers, not splitting the GEMV itself).

**Caveat:** Requires split GEMV kernel + partial allreduce + careful synchronization. The allreduce for 640 elements (1.25KB) might have similar latency to 1280 elements due to fixed overhead, reducing benefit.

**Reference:** arxiv.org/abs/2505.11329, github.com/microsoft/tokenweave

---

### 7. Native v_dot8_i32_i4 Instruction Exploitation
**Estimated gain:** 1-5% decode, potentially 10-20% prefill
**Complexity:** Medium-High

gfx906 has `v_dot8_i32_i4` which processes 8 pairs of signed INT4 values per cycle -> INT32 accumulator. Our current GEMV dequantizes INT4->FP32 and uses scalar FP32 multiply-add (1 multiply per instruction).

**For decode GEMV (bandwidth-bound):**
- Using v_dot8_i32_i4 for INT4xINT4 dot product (8 ops/instruction vs ~1 op/instruction)
- Requires quantizing activations to INT4 (per-token dynamic quantization)
- Fewer instructions -> less register pressure -> more in-flight memory ops -> better memory pipeline utilization
- Modest decode gain since bottleneck is HBM bandwidth, not compute

**For prefill GEMM (compute-bound, higher potential):**
- Prefill GEMM is compute-bound (M=seq_len >> 1)
- v_dot8_i32_i4 gives 8x arithmetic throughput in the GEMM inner loop
- Combined with INT4 activation quantization, could significantly accelerate prefill
- INT4 GEMM v2 already achieved 2.07x; v_dot8_i32_i4 could stack on top

**Also available:** `v_dot4_i32_i8` (4 pairs INT8 per cycle), `v_dot8_u32_u4` (8 pairs unsigned INT4)

**Reference:** llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html

---

### 8. Mixed-Precision Per-Layer Quantization (INT2/INT3 for tolerant layers)
**Estimated gain:** ~1-2 tok/s (+2-4%)
**Complexity:** High

Not all layers are equally sensitive. Later transformer layers tolerate more aggressive quantization (INT2/INT3). Use sensitivity analysis to assign per-layer bit widths: INT4 for critical layers, INT2-3 for tolerant ones. Reduced weight size directly reduces HBM2 bandwidth consumption (the primary bottleneck).

Sub-4-bit quantization approaches: AQLM uses additive codebook quantization, QuIP# uses incoherence processing, SpQR uses sparse+quantized representation. Recent work (2025) shows 2-bit on ~30-40% of layers with codebook approach can maintain >95% of full-precision quality.

**Implementation path:**
- Run sensitivity analysis per-layer (measure perplexity impact of INT2/INT3 per layer)
- Create INT2/INT3 GEMV kernel variants (or modify v_dot8 kernel for sub-4-bit)
- Repack weights with per-layer bit widths
- Estimated bandwidth reduction: ~15-25% if 30-40% of layers use INT2

**Reference:** AQLM (arxiv.org/abs/2401.06118), QuIP# (arxiv.org/abs/2307.13304)

---

### 9. Activation Quantization (W4A8)
**Estimated gain:** ~1-2 tok/s (+2-4%)
**Complexity:** Medium-High

Quantize activations to INT8 for GEMV. `activation_quant.hip` already exists. gfx906 has `v_dot4_i32_i8` for native 4-way INT8 dot product per cycle.

**Implementation path:**
- SmoothQuant-style per-channel scaling to balance activation/weight quantization difficulty
- Dynamic per-tensor activation quantization (already implemented in `activation_quant.hip`)
- INT8 GEMV kernel variant using v_dot4_i32_i8 (4 INT8 pairs per instruction)
- Calibration pass to verify acceptable accuracy loss
- AMD has validated SmoothQuant on MI300X with Composable Kernel (2024)

**Note:** For decode GEMV, activation vector is only 5120 elements (10KB FP16 -> 5KB INT8). The bandwidth savings from smaller activations is negligible. The real benefit is compute efficiency from v_dot4_i32_i8, but decode GEMV is bandwidth-bound not compute-bound. **This optimization has more impact on prefill (compute-bound GEMM) than decode.**

---

## Tier 3: Lower Potential / Higher Risk

### 10. Layer Pruning / Early Exit (LayerSkip)
**Estimated gain:** ~4.7 ms/tok if skipping 16/64 layers -> ~64 tok/s
**Complexity:** Very High (requires model fine-tuning)

Meta's LayerSkip shows many tokens can be generated using only 50-70% of layers with full-model verification. Skipping 16 of 64 layers saves 16 x ~46us allreduce + proportional compute.

**Caveat:** Requires model fine-tuning with layer dropout. Qwen3.5-27B was NOT trained with LayerSkip. This is a model-level change, not an inference optimization.

**Reference:** arxiv.org/abs/2404.16710

---

### 11. Prefill-Specific: Chunked Prefill with GEMM Tiling
**Estimated gain:** Better prompt processing latency and throughput
**Complexity:** Medium

For long prompts, chunk the input sequence and process chunks with optimized GEMM tiling. Current prefill uses INT4 GEMM v2 (2.07x speedup). Additional gains possible from:
- Chunk sizes tuned to LDS capacity (64KB per CU on gfx906)
- Better tile scheduling for non-MFMA GEMM (current tiling may not be optimal)
- Overlapping chunk GEMM with KV cache writes for next chunk

---

## Recommended Priority Order

For reaching 60 tok/s decode on current hardware:

1. **#4 hipSetDevice reduction** (Low effort, ~0.3ms savings, implement first as quick win)
2. **#3 Fused QKV projection** (Medium effort, ~0.4ms savings, clean architectural improvement)
3. **#2 Weight prefetch overlap** (Medium effort, ~1.0ms savings, addresses the 6x bandwidth gap)
4. **#1 Sequoia tree verification** (High effort, ~20-40% effective throughput, the big swing)

Combined #4 + #3 + #2 could yield ~1.7ms savings (18.5 -> 16.8ms = ~60 tok/s). If those alone don't reach 60, #1 (Sequoia) is the algorithmic game-changer that amortizes allreduce across multiple tokens.

---

*Document updated: 2026-03-21*
*Added 6 new approaches from research (PRESERVE, TokenWeave, Sequoia, v_dot8_i32_i4, fused QKV, hipSetDevice)*
*Reorganized into tiers by estimated impact and implementation effort*
*Key finding: 6x gap between theoretical bandwidth floor and actual suggests significant headroom*
*Primary remaining bottleneck: HBM2 memory bandwidth for INT4 GEMV weight streaming + dispatch overhead*
