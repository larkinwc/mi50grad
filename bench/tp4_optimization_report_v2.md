# TP=4 Optimization Report v2: Sprint 2 Final Benchmark

**Generated:** 2026-03-15 07:32:12 UTC
**Model:** Qwen3.5-27B-GPTQ-Int4
**Hardware:** 4× AMD MI50 (gfx906, Vega 20), PCIe x16 Gen4, 16GB HBM2 each
**ROCm:** 7.1.0
**Report file:** bench/tp4_optimization_report_v2.md

---

## Executive Summary

Sprint 2 achieved **38.0 tok/s** TP=4 throughput with all optimizations
combined (C dispatch loop + star topology allreduce + tuned kernels), representing a
**1.87× speedup over single-GPU** (20.3 tok/s
baseline). The gap to vLLM (46.9 tok/s) is 8.9 tok/s
(19% slower than vLLM).

**Key findings of this sprint:**
1. **C dispatch loop**: Eliminates Python kernel dispatch overhead by running all 64
   layers in a compiled C loop. Provides additional throughput improvement over cached+stream.
2. **Star topology wins for 10KB allreduce**: Ring allreduce is **8.5×
   slower** (1015 µs vs 119 µs/call) for hidden_size=5120
   (10KB) on PCIe. Star topology remains the production default.
3. **INT8 allreduce not beneficial**: Star topology already handles 10KB efficiently at ~119 µs.
   Ring allreduce (where smaller INT8 payloads would matter) is fundamentally slower due to 6 sequential
   P2P rounds on PCIe. INT8 quantization of allreduce payloads was assessed and not pursued.
4. **Tuned kernels**: elementwise_v3 (float4 vectorization), flash_attn_256_tuned (4-wavefront
   parallelism, ~5× decode speedup), and gemv_int4_v3_t16 (cooperative reduction, 1.29× faster
   than v2_fused) are already wired as defaults in the engine.

**Correctness:** TP=4 (all opts) vs single-GPU: cosine similarity = 0.999988
(threshold: 0.99) — **PASS**

---

## Throughput Comparison: All Phases

| Optimization Phase | Throughput | vs Prior Phase | vs Single-GPU | vs vLLM |
|---|---|---|---|---|
| Single-GPU baseline (mi50grad) | 20.3 tok/s | — | 1.00× | 0.43× |
| TP=4 serial (P2P allreduce, no caching) | 12.4 tok/s | — | 0.61× | 0.26× |
| TP=4 cached dispatch | 23.7 tok/s | +91% | 1.17× | 0.51× |
| TP=4 combined (cached + stream overlap) | 25.5 tok/s | +8% | 1.26× | 0.54× |
| TP=4 C dispatch + tuned (this run) | **38.0 tok/s** | +49% | **1.87×** | **0.81×** |
| vLLM TP=4 (AWQ, reference) | 46.9 tok/s | — | 2.31× | 1.00× |
| Theoretical TP=4 ceiling | ~81 tok/s | — | 4.00× | 1.73× |

---

## Sprint 2 Optimization Details

### Optimization 1: C Dispatch Loop

| Metric | Cached+Stream (Python) | C Dispatch (this run) | Improvement |
|---|---|---|---|
| TP=4 throughput | 34.0 tok/s | 38.0 tok/s | 1.12× |
| Latency (ms/tok) | 29.38 ms | 26.31 ms | — |
| vs 25.5 baseline | — | — | 1.49× |
| C dispatch available | — | YES | — |

**What the C dispatch loop does:**
- Pre-serializes all 64 layers' kernel parameters into a C-accessible plan at init
- Dispatches all kernels in a tight C loop (`c_dispatch_step()`), bypassing Python entirely
- Handles position-dependent parameter updates (RoPE cos/sin, attention seq_len) in C
- Integrates HIP event-based async allreduce within the C loop
- Falls back to Python cached+stream if c_dispatch.so is unavailable

**Dispatch priority in `decode_step()`:**
1. C dispatch (highest priority, `_c_dispatch_enabled=True`)
2. Python cached+stream (`_cached_dispatch` + `_stream_overlap_dispatch`)
3. Python cached-only
4. Python serial (lowest priority, fallback)

### Optimization 2: Allreduce Topology — Star vs Ring Analysis

| Metric | Star (P2PAllreduce) | Ring (RingAllreduce) | Winner |
|---|---|---|---|
| Allreduce latency | ~119 µs/call | ~1015 µs/call | **Star 8.5× faster** |
| TP=4 tok/s | ~25.5 tok/s | ~10.1 tok/s | **Star** |
| P2P rounds | 2 (gather + broadcast) | 6 (3 reduce-scatter + 3 all-gather) | Star |
| Transfer type | FP16 (10KB per call) | FP32 (20KB per round) | Star (less data) |
| Precision | FP16 throughout | FP32 accumulators | Ring (higher precision) |
| Async overlap | YES (non-blocking GPU events) | NO (CPU-blocking sync per round) | Star |

**Why ring is slower for 10KB payloads on PCIe:**
- Ring requires 6 sequential P2P rounds with CPU-level synchronization between each
- For 5120 FP16 elements (10KB), PCIe latency (not bandwidth) dominates:
  6 rounds × ~170 µs/round ≈ 1015 µs vs star's 2 rounds ≈ 119 µs
- Ring's bandwidth advantage only materializes at hidden_size ≥ ~32768 (64KB+ FP16)
  where: `6 × latency < 2 × (payload / P2P_bandwidth)`
  For 12 GB/s PCIe: break-even at ~65536 elements (128KB)
- **Recommendation:** Star topology for all Qwen3.5-27B paths (hidden_size=5120)

**Ring allreduce is available** via `set_ring_allreduce(True)` for future models with
larger hidden dimensions where ring topology becomes beneficial.

### Optimization 3: INT8 Allreduce Payload Assessment

**Assessment: INT8 partial quantization of allreduce payload is NOT beneficial.**

Rationale:
- **Star topology** (production path) already handles 10KB allreduce efficiently at ~119 µs/call
  - INT8 would halve payload (10KB → 5KB) but star is already PCIe-latency-bound, not bandwidth-bound
  - Adding quantize/dequantize kernels (~38-49 µs each for hidden_size=5120) exceeds any bandwidth savings
- **Ring topology** (where smaller payloads matter for bandwidth) is fundamentally slower due to
  6 sequential P2P rounds on PCIe — adding INT8 compression does not fix the latency problem
- **Conclusion:** INT8 allreduce quantization would add correctness risk and implementation complexity
  for no measurable throughput gain at hidden_size=5120

### Optimization 4: Kernel Tuning Results

Decode-critical kernels on gfx906 (MI50), measured for Qwen3.5-27B shapes:

**INT4 GEMV** (primary FFN kernel):
| Shape | Kernel | us/call | vs Prior |
|---|---|---|---|
| N=4096, K=5120 | gemv_int4_v3_t16 (default) | ~30 µs | 1.29× vs v2_fused |
| N=11008, K=5120 | gemv_int4_v3_t16 (default) | ~64 µs | ~tied vs v2_fused |
| N=13696, K=5120 | gemv_int4_v3_t16 (default) | ~80 µs | — |

**FlashAttention Decode** (GQA, head_dim=256):
| kv_len | Kernel | us/call | vs Original |
|---|---|---|---|
| 256 | flash_attn_256_tuned (default) | ~62 µs | 3.57× faster |
| 512 | flash_attn_256_tuned | ~113 µs | 5.21× faster |
| 1024 | flash_attn_256_tuned | ~223 µs | 5.56× faster |
| 2048 | flash_attn_256_tuned | ~435 µs | 5.68× faster |

**Elementwise** (RMSNorm, SiLU, residual add):
| Kernel | us/call | vs v2 |
|---|---|---|
| rmsnorm_v3 (dim=5120) | ~35 µs | 1.43× faster |
| silu_fused_v3 (dim=5120) | ~54 µs | ~tied |
| residual_add_v3 (dim=5120) | ~53 µs | ~tied |

All tuned variants are **already wired as the default** in `engine.py` decode path.
No additional wiring changes needed; tuning confirmed that current defaults are optimal.

---

## Correctness Validation

| Check | Value | Threshold | Result |
|---|---|---|---|
| Single-GPU regression | 21.5 tok/s | 20.3±10% | PASS |
| TP=4 vs single-GPU cosine sim (all opts) | 0.999988 | ≥0.99 | PASS |
| Fallback path integrity | — | C dispatch off → cached+stream | PASS |

---

## Gap Analysis: mi50grad vs vLLM

| Factor | vLLM Advantage | Estimated Impact |
|---|---|---|
| HIP graph capture | Eliminates Python dispatch entirely (~0 ms dispatch) | ~10–15 ms/tok → ~0 ms |
| torch.compile | Kernel fusion, optimized memory layout, kernel auto-tuning | 10–20% |
| Chunked prefill | Better GPU utilization, faster KV cache warming | Prefill-focused |
| Optimized attention kernels | FlashAttention-2/3, hardware-specific decode tuning | 5–10% |
| INT8/FP8 activations | Reduced allreduce payload, faster GEMV | 5–10% |
| Continuous batching | Improved GPU utilization across requests | Multi-request |

**Current status:** With C dispatch, Python dispatch overhead is near-zero. The remaining bottleneck
is allreduce (128 × ~119 µs ≈ 15 ms/tok) and per-layer
GPU compute. The gap to vLLM (8.9 tok/s) is primarily due to:
- vLLM uses AWQ (higher throughput than GPTQ-Int4 for some shapes)
- vLLM uses HIP graph capture (eliminates all Python dispatch overhead)
- vLLM continuous batching (amortizes overhead across multiple requests)

---

## Technical Notes

- **Hardware:** MI50 uses gfx906 (Vega 20). No XGMI fabric — P2P uses BAR1 PCIe aperture.
  All GPU pairs are 2 PCIe hops apart, limiting P2P bandwidth vs NVLink/XGMI.
- **Allreduce payload:** hidden_size=5120 × 2 bytes = 10 KB per allreduce call.
  128 allreduces per decode step (2 per layer × 64 layers).
- **Benchmark conditions:** Single decode request (batch=1), fixed random embedding,
  100 decode steps, 3 warmup steps. Real inference with growing KV cache would vary.
- **DeltaNet layers:** 48 of 64 layers use DeltaNet linear attention (recurrent state,
  no KV cache). 16 layers use full GQA FlashAttention decode with KV cache.
- **vLLM comparison:** vLLM uses AWQ quantization. Not perfectly apples-to-apples with
  our GPTQ-Int4 model, but directionally valid for gap analysis.

---

*Report generated by tests/bench_tp4_sprint2.py*
