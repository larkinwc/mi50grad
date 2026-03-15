# TP=4 Optimization Report v3: Sprint 3 Final Benchmark

**Generated:** 2026-03-15 21:35:43 UTC
**Model:** Qwen3.5-27B-GPTQ-Int4
**Hardware:** 3× AMD MI50 (gfx906) + 1× AMD MI100 (gfx908) for TP=4, 16GB HBM2 each
**ROCm:** 7.1.0
**Report file:** bench/tp4_optimization_report_v3.md

---

## Executive Summary

Sprint 3 final benchmark achieved **36.6 tok/s** TP=4 throughput
with all optimizations combined, representing **-3.7% vs
Sprint 2 baseline** (38.0 tok/s).
The gap to vLLM (46.9 tok/s) is 10.3 tok/s
(22% below vLLM).

**Best active dispatch mode:** graph (C)

**Sprint 3 Optimizations Applied:**
1. **Q/KV stream sync elimination** (M1): Removed 32 host-blocking `hipStreamSynchronize`
   calls per token by running Q and KV GEMVs sequentially on the null stream.
2. **Direct KV cache writes** (M1): Eliminated 32 `hipMemcpyAsync` D2D copies per token
   via `qknorm_rope_cachew` fused kernel + direct V write to cache position.
3. **Allreduce overlap deepening** (M1): `c_dispatch_v2.c` reduces `hipSetDevice` calls
   by ~384/token; `hipStreamWaitEvent` is host-non-blocking on gfx906/ROCm 7.1.
4. **HIP graph capture and replay** (M2): Per-GPU compute segments captured as HIP graphs.
   Graph capture: available=True; C graph dispatch plan: True.
   7.9× speedup per graph segment vs direct launch; 512 total segments per step.
5. **C graph dispatch extension** (M2): C extension (`c_graph_dispatch.c`) eliminates
   Python ctypes overhead from graph replay loop (~512 `hipGraphLaunch` calls in C).
6. **C dispatch loop** (Sprint 2): All 64 layers dispatched in tight C loop, no Python.
7. **Star topology allreduce** (Sprint 2): 8.5× faster than ring for 10KB payloads.
8. **Tuned kernels** (Sprint 2): `elementwise_v3`, `flash_attn_256_tuned`,
   `gemv_int4_v3_t16`.

**Correctness:** TP=4 (all opts) vs single-GPU: cosine similarity = 0.999760
(threshold: 0.99) — **PASS**

---

## Throughput Comparison: All Sprint Phases

| Optimization Phase | Throughput | vs Single-GPU | vs vLLM |
|---|---|---|---|
| Single-GPU baseline (mi50grad) | 20.3 tok/s | 1.00× | 0.43× |
| TP=4 serial (P2P, no caching) | 12.4 tok/s | 0.61× | 0.26× |
| TP=4 cached dispatch | 23.7 tok/s | 1.17× | 0.51× |
| TP=4 combined (cached + stream overlap) | 25.5 tok/s | 1.26× | 0.54× |
| TP=4 Sprint 2: C dispatch + tuned kernels | 38.0 tok/s | 1.87× | 0.81× |
| TP=4 Sprint 3 M1: allreduce pipeline | 38.1 tok/s | 1.88× | 0.81× |
| **TP=4 Sprint 3 M2: graph decode (this run)** | **36.6 tok/s** | **1.80×** | **0.78×** |
| vLLM TP=4 (AWQ, reference) | 46.9 tok/s | 2.31× | 1.00× |

Sprint 3 M2 improvement vs Sprint 2: **-1.4 tok/s (-3.7%)**
Sprint 3 M2 improvement vs Sprint 3 M1: **-1.5 tok/s (-4.0%)**
Remaining gap to vLLM: **10.3 tok/s (22% below vLLM)**

---

## Sprint 3 Optimization Results

### Milestone 1: allreduce-pipeline

**Result: 38.1 tok/s** (+0.1 tok/s vs Sprint 2)

Key finding: These optimizations provide architectural cleanup rather than large standalone
throughput gains. The bottleneck is allreduce latency (~15.2 ms/token = 128 × ~119 µs).

| Optimization | Impact |
|---|---|
| Q/KV stream sync elimination (32 syncs/token removed) | Neutral — syncs were already overlapped by allreduce |
| Direct KV cache writes (32 D2D copies/token removed) | Neutral — D2D copies too small vs allreduce bottleneck |
| Allreduce overlap deepening (384 hipSetDevice calls saved) | +0.3% — within measurement noise |
| Combined M1 vs Sprint 2 | +0.1 tok/s (+0.3%) |

### Milestone 2: hip-graph-decode

**Result: 36.6 tok/s** (-1.5 tok/s vs M1, active mode: graph (C))

#### HIP Graph Infrastructure

Graph API availability on gfx906/ROCm 7.1: **ALL CONFIRMED**
- `hipGraphCreate`, `hipStreamBeginCapture`, `hipStreamEndCapture` ✓
- `hipGraphInstantiate`, `hipGraphLaunch` ✓
- `hipGraphExecKernelNodeSetParams`, `hipGraphGetNodes` ✓

Key infrastructure findings:
- **7.9× speedup per graph segment** vs direct launch (8 kernels at N=5120)
- **512 total segments** per decode step (4 GPUs × 64 layers × 2 segments)
- **Graph capture time**: ~130ms (one-time cost at first decode step)
- **Position-based node identification**: required because multiple kernels share the same
  function handle (e.g., `gemv_fp16_v2` is used for Q, K, V, and O projections)

#### Graph-Based Decode Path

**Critical finding: Python-level graph replay is SLOWER than C dispatch.**
- C dispatch: ~38 tok/s (tight C loop, ~960 hipModuleLaunchKernel calls)
- Python graph replay: ~28 tok/s (Python loop, 512 hipGraphLaunch + 256 hipGraphExecKernelNodeSetParams)
- **Root cause**: hipGraphLaunch via Python ctypes still carries Python overhead per call.
  The 7.9× per-segment speedup (17.83 µs vs 140.89 µs/segment) is negated by 512 Python-level
  dispatch calls (~8ms/token) vs the C dispatch's single C function call per step.

#### C Graph Dispatch Extension

**Solution: C extension (`c_graph_dispatch.c`) runs graph replay in tight C loop.**
- C graph dispatch: ~35.9 tok/s (1.01× vs 35.6 tok/s C dispatch baseline)
- Python graph dispatch: ~28 tok/s (0.74× vs C dispatch)
- The C extension eliminates Python overhead from graph replay: 512 `hipGraphLaunch` calls
  in C run in ~1ms vs ~8ms in Python

**What worked:**
- HIP graph capture on gfx906/ROCm 7.1 (confirmed working)
- Mutable parameter updates via `hipGraphExecKernelNodeSetParams` (cos/sin, seq_len)
- Direct KV write mode (`qknorm_rope_cachew` kernel) works correctly in graph mode
- C graph dispatch plan serialization (struct-based, ~1056 bytes/layer/GPU)
- All 15 decode steps produce cosine sim >= 0.99 vs single-GPU reference

**What didn't work (as expected):**
- Python-orchestrated graph replay is SLOWER (documented above)
- Per-segment overhead dominates: 512 Python calls > C loop's 960 kernel launches
- `hipGraphLaunch` overhead (~15 µs/call) × 512 = ~7.7ms vs hipModuleLaunchKernel (~1 µs/call) × 960 = ~1ms

---

## Per-Optimization Impact Breakdown

| Configuration | Mode | Throughput | vs Sprint 2 |
|---|---|---|---|
| Sprint 2: C dispatch (star+tuned, no Sprint 3 opts) | C dispatch | 38.0 tok/s | -0.0% |
| Sprint 3 M1: +direct KV write + Q/KV sync elim | C dispatch | 37.9 tok/s | -0.2% |
| Sprint 3 M2: +HIP graph decode (best available) | graph (Python) | 37.5 tok/s | -1.4% |


---

## Correctness Validation

| Check | Value | Threshold | Result |
|---|---|---|---|
| Single-GPU regression | 22.1 tok/s | 20.3±10% | PASS |
| TP=4 all opts cosine sim | 0.999760 | ≥0.99 | PASS |
| No regression vs Sprint 2 (±5%) | 36.6 tok/s (-3.7%) | ≥36.1 tok/s | PASS |

---

## Progressive Fallback Chain

| Fallback Step | Result |
|---|---|
| Mode A: graph + c_dispatch | PASS |
| Mode B: c_dispatch only (no graph) | PASS |
| Mode C: cached+stream (no c_dispatch, no graph) | PASS |
| Mode D: cached only | PASS |
| Mode E: serial (all disabled) | PASS |
| Cross-mode correctness (all modes agree ≥0.99) | PASS |


---

## Gap Analysis vs vLLM (Post Sprint 3)

| Factor | Current State | Remaining Impact |
|---|---|---|
| Kernel dispatch overhead | C dispatch: ~1ms/token; Graph: ~1ms via C extension | Minimal (~2% of total) |
| Allreduce latency | 128 × ~119 µs ≈ 15.2 ms/token (hard floor, star topology) | **Dominant bottleneck** |
| Per-layer GPU compute | ~11 ms/token (64 layers × ~172 µs) | Fixed by hardware |
| hipSetDevice + event overhead | ~10 ms/token (reduced from ~13ms by M1) | ~3% improvement possible |
| vLLM AWQ advantage | AWQ vs GPTQ-Int4: potentially 10-15% GEMV speedup | Medium impact |
| vLLM HIP graph (global capture) | Captures allreduce+compute together (no host orchestration) | **High impact if achievable** |

**Remaining gap: 10.3 tok/s (22% below vLLM)**

---

## Recommendations for Sprint 4

Based on the findings from Sprint 3:

### Priority 1: Eliminate per-allreduce host round-trips

The dominant remaining bottleneck is the 128 host-level allreduce calls per token
(~15.2 ms/token). Approaches:
1. **All-in-one C graph with embedded allreduce**: Write a custom HIP compute+allreduce
   kernel that performs allreduce on-device without returning to host. Requires NVLink or
   XGMI for fast GPU-GPU transfers — PCIe BAR1 has ~12 GB/s P2P bandwidth.
2. **Fused layer compute with allreduce**: Defer allreduce until multiple layer outputs
   have accumulated. Note: DeltaNet layers show that deferred allreduce causes cosine sim ~0.59
   (infeasible for correctness). Only applicable if layer independence can be proven.
3. **Reduce allreduce calls via model surgery**: Use GQA to reduce KV head count further,
   reducing the number of attention layers that require full allreduce. Currently 16 full-attn
   + 48 DeltaNet = 64 layers × 2 AR = 128 AR/step.

### Priority 2: Improve allreduce throughput

Star topology is already near-optimal for 10KB payloads. Options:
1. **Kernel-level P2P allreduce**: Fused P2P reduce already explored (1.72× isolated
   speedup, but only 1.01× e2e due to sync overhead). Further refinement needed.
2. **XGMI upgrade**: MI100+ with XGMI fabric would provide 100+ GB/s vs 12 GB/s PCIe.
   Not an option with current hardware.
3. **Allreduce-free weight distribution**: Tensor parallel approaches that avoid allreduce
   (e.g., sequence parallel) have limited applicability for decode (batch=1, seq_len=1).

### Priority 3: Reduce compute time

With allreduce at ~15.2 ms/token (hard floor) and dispatch at ~1 ms, compute
is ~11 ms/token. Possible improvements:
1. **W8A8/W4A8 quantization for FFN**: INT8 activations reduce GEMV bandwidth by 2×.
   Benchmarks show W4A16 is faster for decode (bandwidth-limited), but W8A8 may help
   for specific shapes. Currently W4A16 (GPTQ-Int4) is the best for MI50.
2. **Flash attention prefill optimization**: v3 block-tiled kernel is 1.59-1.89× faster
   for prefill. Decode kernel unchanged.
3. **Custom MLA attention**: Multi-head latent attention (as in DeepSeek) could reduce
   KV cache size and attention compute, but requires model re-training.

### Priority 4: Global HIP graph (future)

vLLM likely achieves graph-based dispatch by using a global graph that captures
BOTH compute AND allreduce (using NCCL for XGMI-connected GPUs). On PCIe:
1. hipStreamBeginCapture cannot capture hipMemcpyPeerAsync (or can it?)
2. If capturable: create one global graph per layer that captures:
   kernels + P2P transfers + reduce kernel + broadcast = full layer graph
3. Only update mutable params per step (RoPE, seq_len, KV pointers)
4. Expected result: near-zero host overhead for entire decode step

---

## Technical Notes

- **Hardware:** MI50 (gfx906 Vega 20) + MI100 (gfx908). No XGMI — P2P uses PCIe BAR1.
- **Allreduce payload:** hidden_size=5120 × FP16 = 10 KB per call, 128 calls/token.
- **Benchmark conditions:** batch=1, fixed random embedding, 100 steps, 3 warmup.
- **C dispatch availability:** YES (c_dispatch.so loadable).
- **Graph dispatch availability:** YES (set_graph_dispatch() exists).
- **C graph dispatch plan:** BUILT.
- **Direct KV write:** Uses `qknorm_rope_cachew` fused kernel for K, separate V write to cache.
- **Q/KV sync:** Sequential null-stream dispatch (no explicit sync needed).

---

*Report generated by tests/bench_tp4_sprint3.py*
