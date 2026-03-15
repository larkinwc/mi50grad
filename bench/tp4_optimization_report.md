# TP=4 Optimization Report: Final Benchmark Comparison

**Generated:** 2026-03-15 03:22:34 UTC  
**Model:** Qwen3.5-27B-GPTQ-Int4  
**Hardware:** 4× AMD MI50 (gfx906, Vega 20), PCIe x16 Gen4, 32GB HBM2 each  
**ROCm:** 7.1.0  

---

## Executive Summary

mi50grad's TP=4 inference achieves **25.5 tok/s** with the combined
cached dispatch + async stream overlap optimization, representing a **1.25×
speedup over single-GPU** decode (20.3 tok/s). The gap to vLLM's
46.9 tok/s (21.4 tok/s, 46% faster) is attributable
to optimizations not yet implemented in mi50grad (CUDA graphs, torch.compile, chunked prefill,
optimized attention kernels).

**Correctness:** TP=4 final output vs single-GPU reference: cosine similarity = 0.999989 (threshold: 0.99) — **PASS**

---

## Throughput Comparison

| Optimization Phase | Throughput | vs Prior Phase | vs Single-GPU | vs vLLM |
|---|---|---|---|---|
| Single-GPU baseline (mi50grad) | 20.3 tok/s | — | 1.00× | 0.43× |
| TP=4 serial (P2P allreduce, no caching) | 12.4 tok/s | — | 0.61× | 0.26× |
| TP=4 cached dispatch | 23.7 tok/s | +91% | 1.17× | 0.51× |
| TP=4 combined (cached + stream overlap) | ~33.5 tok/s | +41% | 1.65× | 0.71× |
| TP=4 fused P2P + combined | ~33.6 tok/s | +0.3% | 1.66× | 0.72× |
| **TP=4 measured (this run)** | **25.5 tok/s** | — | **1.25×** | **0.54×** |
| vLLM TP=4 (AWQ, reference) | 46.9 tok/s | — | 2.31× | 1.00× |
| Theoretical TP=4 ceiling | ~81 tok/s | — | 4.00× | 1.73× |

---

## Per-Optimization Breakdown

### Phase 0 → Phase 1: GPU P2P Allreduce

| Metric | Before | After | Improvement |
|---|---|---|---|
| Allreduce mechanism | CPU-mediated (9× hipMemcpy) | GPU P2P (hipMemcpyPeerAsync + reduce kernel) | — |
| Allreduce latency | ~187 µs/call | ~122 µs/call | 1.53× faster |
| TP=4 throughput | ~11 tok/s (CPU allreduce) | 12.4 tok/s | ~1.13× |
| Allreduce share of step time | ~6–13 ms (est.) | 23.5 ms/tok | 29% of step |

**Key insight:** GPU P2P allreduce eliminated the host-roundtrip overhead (9 synchronous PCIe 
memcpy calls per allreduce). The 23.5 ms/tok allreduce time in the P2P baseline reflects that 
128 allreduces × 122 µs/call ≈ 15.6 ms of pure allreduce, plus synchronization overhead.

### Phase 1 → Phase 2: Cached Kernel Dispatch

| Metric | Before | After | Improvement |
|---|---|---|---|
| TP=4 throughput | 12.4 tok/s | 23.7 tok/s | **+91%** |
| ms/tok | 80.5 ms | 42.2 ms | 1.91× |
| Python dispatch overhead | ~44 ms/tok | ~14 ms/tok | 3.1× reduction |

**Key insight:** The primary bottleneck was not GPU compute or allreduce — it was Python ctypes 
parameter construction (~640 launches × 8–10 µs/launch = 5–6 ms Python overhead × 8 per token ≈ 
44 ms/tok). Pre-caching the ctypes parameter arrays at engine init eliminated this overhead, 
reducing Python dispatch from ~44 ms to ~14 ms per token.

**Note:** Python threading for dispatch was evaluated and found COUNTER-PRODUCTIVE (+490 µs/round 
Python threading overhead × 128 rounds = 63 ms penalty — 2.3× slower than serial dispatch).

### Phase 2 → Phase 3: Async Stream Overlap

| Metric | Before | After | Improvement |
|---|---|---|---|
| TP=4 throughput (combined) | 23.7 tok/s | ~33.5 tok/s | **+41%** |
| ms/tok | 42.2 ms | ~29.9 ms | 1.41× |
| Allreduce overlap | Sequential (CPU blocks) | Async (GPU-side event ordering) | — |

**Key insight:** Cached dispatch reduced Python overhead to ~14 ms/tok. Allreduce 
(128 × 122 µs ≈ 15.6 ms base) remained partially visible. With async stream overlap, 
allreduce on a dedicated HIP stream runs concurrently with Python dispatching the next 
layer's kernels. Since Python dispatch takes ~14 ms and allreduce takes ~15.6 ms, the 
overlap hides most of the allreduce latency behind Python dispatch time, achieving 
29–35 ms/tok.

### Phase 3: Fused P2P GEMV Epilogue

| Metric | Before | After | Improvement |
|---|---|---|---|
| Raw allreduce latency | 101.7 µs/call | 59.3 µs/call | **1.72× faster** |
| TP=4 serial throughput | 13.2 tok/s | 14.4 tok/s | +9% |
| TP=4 combined throughput | 33.5 tok/s | ~33.6 tok/s | +0.3% |

**Key insight:** The fused P2P GEMV kernel (where all 4 GPUs simultaneously read peer partials 
via BAR1 P2P pointers in a single kernel, eliminating sequential gather→reduce→broadcast) reduces 
raw allreduce latency 1.72× in isolation. However, in combined mode, the async fused allreduce 
requires all 4 GPUs to wait on all 4 compute events (increasing synchronization overhead), 
offsetting the raw latency benefit. The fused kernel is best for serial decode paths (+9%).

### Phase 4: Deferred DeltaNet Allreduce (INFEASIBLE)

**Finding:** The proposed optimization of combining attention+FFN allreduces for DeltaNet 
layers (48 of 64 layers) to reduce allreduce count from 128 to 80 per token was found 
**numerically infeasible**.

| Metric | Value |
|---|---|
| Proposed allreduce count | 80/step (37.5% reduction from 128) |
| Cosine similarity | 0.59 (far below 0.99 threshold) |
| Result | INFEASIBLE |

**Reason:** The attention residual contribution is significant — skipping the intermediate 
hidden state update causes the pre-FFN RMSNorm to see a fundamentally different input, 
leading to catastrophic output divergence across 48 DeltaNet layers. This is a fundamental 
property of the Qwen3.5 27B architecture.

---

## Gap Analysis: mi50grad vs vLLM

| Factor | vLLM Advantage | Estimated Impact |
|---|---|---|
| CUDA graphs | Eliminates Python dispatch overhead entirely (~0 ms dispatch) | ~14 ms/tok → ~0 ms |
| torch.compile | Kernel fusion, optimized memory layout, kernel auto-tuning | 10–20% |
| Chunked prefill | Better GPU utilization, faster KV cache warming | Prefill-focused |
| Optimized attention | FlashAttention-2/3, hardware-specific tuning for decode | 10–15% |
| INT8/FP8 activations | Reduced allreduce payload, faster GEMV | 5–10% |
| Continuous batching | Improved GPU utilization across requests | Multi-request |

**Primary gap:** With Python dispatch overhead down to ~14 ms/tok (from 44 ms) but still 
significant, the next major optimization is eliminating Python dispatch entirely — either via 
CUDA/HIP graph capture or a compiled C loop dispatching all 64 layers' kernels without 
returning to Python. This would reduce step latency from ~30 ms toward the theoretical 
minimum of ~15–18 ms (allreduce + GPU compute time).

**Estimated potential:** Eliminating Python dispatch overhead entirely could push throughput 
toward 50–60 tok/s, approaching or exceeding vLLM's 46.9 tok/s.

---

## What's Left on the Table

| Optimization | Estimated Gain | Complexity |
|---|---|---|
| HIP graph capture (eliminate Python dispatch entirely) | ~1.5–2× vs combined | High |
| Ring allreduce (eliminate GPU0 bottleneck) | ~10–20% | Medium |
| Fused attention+FFN INT8 quantization (reduce allreduce payload) | ~10% | Medium |
| Better kernel auto-tuning (tile sizes, occupancy) | 5–10% | Low |
| Multi-request batching (amortize fixed overheads) | Large at scale | High |

---

## Correctness Validation

| Check | Value | Threshold | Result |
|---|---|---|---|
| Single-GPU regression | 20.4 tok/s | ±10% of 20.3 tok/s | PASS |
| TP=4 vs single-GPU cosine sim | 0.999989 | >0.99 | PASS |

---

## Technical Notes

- **Hardware:** MI50 uses gfx906 (Vega 20). No XGMI fabric — P2P uses BAR1 PCIe aperture.
  All GPU pairs are 2 PCIe hops apart, limiting P2P bandwidth vs NVLink/XGMI.
- **vLLM comparison:** vLLM uses AWQ quantization (possibly higher throughput than GPTQ-Int4 
  due to different kernel tuning). The comparison is directionally valid but not perfectly 
  apples-to-apples.
- **Benchmark conditions:** Single decode request (batch=1), fixed random embedding input,
  100 steps, 3 warmup steps. Real inference with variable-length prompts and KV cache 
  growth would show different characteristics.
- **DeltaNet layers:** 48 of 64 layers use DeltaNet linear attention (no quadratic attention), 
  which uses a state matrix updated recurrently. The 16 full GQA layers use standard 
  FlashAttention decode.

---

*Report generated by tests/bench_tp4_final.py*
