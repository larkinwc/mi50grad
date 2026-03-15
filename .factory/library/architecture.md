# Architecture

Architectural decisions, patterns discovered, and kernel design notes.

**What belongs here:** Kernel design patterns, LDS layout strategies, register budget decisions, instruction selection rationale, fusion patterns.

---

## Fused P2P GEMV Epilogue (Milestone 3: fused-p2p-gemv-epilogue, 2026-03-14)

**Implementation:** `src/kernels/gemv_p2p_reduce.hip` + `FusedP2PReduce` class in `src/runtime/p2p_allreduce.py`

**Design:** Option B — Separate fused reduce+residual kernel where each GPU reads all peer GPU partial
results via P2P pointers (remote memory access within kernel), eliminates gather→reduce→broadcast pipeline.

**How it works:**
- Each GPU launches `fused_p2p_reduce_residual_tp4_kernel` that reads local partial + 3 remote partials via P2P pointers
- All 4 GPUs run their kernels simultaneously (no sequential gather needed)
- Each GPU writes its own updated hidden buffer → no broadcast needed
- Eliminates intermediate gather buffers from GPU0

**Key insight:** On PCIe (no XGMI), remote GPU memory reads from within a kernel use the BAR1 aperture
mapping. This allows direct P2P reads without hipMemcpyPeerAsync, and for 10KB payloads the parallel
execution of all 4 kernels simultaneously is faster than the sequential gather+broadcast pipeline.

**Performance results (gfx906 MI50, TP=4, hidden_size=5120):**
- Standard P2P allreduce: 101.7 us/call
- Fused P2P reduce:       59.3 us/call  (**1.72x faster** in isolation)
- Serial decode (standard P2P): 13.24 tok/s (75.50 ms/tok)
- Serial decode (fused P2P):    14.40 tok/s (69.43 ms/tok) → 1.09x faster
- Combined mode (standard P2P): 33.28 tok/s (30.05 ms/tok) 
- Fused+Combined mode:          33.63 tok/s (29.73 ms/tok) → 1.01x (comparable)

**Critical finding:** The fused kernel's async variant (allreduce_residual_async) performs WORSE
in the combined cached+stream-overlap decode mode. The reason: all 4 GPUs' allreduce streams must
wait on ALL 4 compute events (since each GPU reads from all other GPUs), creating more cross-GPU
synchronization. Standard P2P allreduce (where only GPU0 manages gather+reduce+broadcast) overlaps
better with compute dispatch.

**Design decision:** `_decode_step_cached_stream()` always uses standard P2P allreduce.
The fused P2P kernel is used in `_allreduce_residual()` (synchronous/serial path) when
`set_fused_p2p_reduce(True)`. This gives 1.09x speedup in serial mode.

**API:**
```python
engine.set_fused_p2p_reduce(True)   # Enable fused P2P (for serial path)
engine.set_cached_dispatch(True)     # Enable cached dispatch
engine.set_stream_overlap_dispatch(True)  # Enable stream overlap (uses standard P2P)
# decode_step automatically picks the best combination
```

**Correctness:** Cosine similarity = 0.999984-0.999996 vs serial reference (all steps pass 0.99 threshold).
Tests verify both FP16 GEMV (attention out_proj) and INT4 GEMV (FFN down_proj) paths.

**HIP kernel pattern:** Same as p2p_allreduce.hip — uses `hipLaunchKernelGGL` in C wrappers,
compiled as shared library. Each GPU sets its own device context before launching. P2P access must
be enabled between all GPU pairs before kernel execution.

---



**Implemented:** `_decode_step_cached_stream()` in `src/inference/tp_engine.py` combines both milestone 2 optimizations into a single decode path.

**Key design:**
- Enabled when both `set_cached_dispatch(True)` AND `set_stream_overlap_dispatch(True)` are called
- `decode_step()` dispatches to combined mode first (highest priority), then cached-only, then stream_overlap, then serial
- The combined method is a direct fusion of `_decode_step_cached()` and `_decode_step_stream_overlap()`

**Per-layer structure:**
1. `wait_for_allreduce_on_compute_stream()` at layer start (waits for previous FFN AR, GPU-side, no CPU block)
2. Per-engine attention kernels via `launch_cached()` (cached ctypes params)
3. `allreduce_residual_async()` for attention partials (non-blocking Python return)
4. `wait_for_allreduce_on_compute_stream()` for attention AR completion
5. Per-engine FFN kernels via `launch_cached()` (cached ctypes params)
6. `allreduce_residual_async()` for FFN partials (non-blocking; next layer waits for it)

**Performance results (4x MI50, 100 steps, Qwen3.5-27B-GPTQ-Int4):**
- Serial: 13.5 tok/s (74.3 ms/tok)
- Cached-only: 23.8 tok/s (42.0 ms/tok)
- Stream overlap only: 14.4 tok/s (69.3 ms/tok)
- **Combined: 28.2–34.4 tok/s (29–35 ms/tok)** — 1.185–1.445x vs cached, 2.09x vs serial

**Correctness:** Min cosine similarity = 0.999979 (all 10 steps pass, threshold 0.99).

**Why combined beats cached:**
- Cached dispatch reduces Python overhead from ~60ms to ~14ms
- Cached+stream then overlaps allreduce (~28ms) with Python dispatch time for next layer
- Effective throughput: hidden_allreduce → wall_clock ≈ max(14ms, residual after overlap)
- Actual measured: 29-35 ms/tok (allreduce largely hidden behind ~14ms Python dispatch)

**Profiler caveat:** The `AllreduceProfiler` in bench_tp4.py only instruments `_allreduce_residual()`. Combined mode uses `allreduce_residual_async()` instead, so the profiler reports 0.00 ms allreduce. This is expected — it means allreduce runs asynchronously on GPU without Python timing.

**Implementation files:**
- `src/inference/tp_engine.py`: `_decode_step_cached_stream()`, updated `decode_step()` dispatch priority
- `tests/test_combined_dispatch.py`: Correctness (10 steps) + benchmark (100 steps) + profile breakdown
- `tests/bench_tp4.py`: Added "COMBINED MODE" section

---

## Stream Overlap with HIP Events (Milestone 2: stream-compute-overlap)

**Implemented:** Async P2P allreduce on dedicated non-blocking HIP streams with GPU-side event synchronization.

**Key design decisions:**
1. **Non-blocking streams for allreduce**: Created with `hipStreamCreateWithFlags(HIP_STREAM_NON_BLOCKING=1)` to avoid implicit null-stream serialization. Without this, allreduce streams would auto-synchronize with the default compute stream (null stream), eliminating the overlap.
2. **Compute events on null stream**: `hipEventRecord(event, null_stream)` captures the completion of compute kernels. The allreduce stream waits on these via `hipStreamWaitEvent` — a GPU-side wait (no CPU blocking).
3. **Allreduce completion events on allreduce stream**: `hipEventRecord(ar_done_event, allreduce_stream)` signals completion. The null compute stream waits on these before next RMSNorm via `hipStreamWaitEvent`.
4. **Explicit event ordering replaces implicit sync**: Instead of `hipDeviceSynchronize()`, explicit events provide correct ordering without CPU blocking.

**Implementation files:**
- `src/runtime/hip_dispatch.py`: Added `stream_create_nonblocking()`, `stream_wait_event()`
- `src/runtime/p2p_allreduce.py`: Added `_allreduce_streams` (non-blocking), `_compute_events`, `_ar_done_events`; new methods `allreduce_residual_async()`, `wait_for_allreduce_on_compute_stream()`
- `src/inference/tp_engine.py`: Added `_decode_step_stream_overlap()`, `set_stream_overlap_dispatch()`

**Measured results (4x MI50, 100 steps Qwen3.5-27B-GPTQ-Int4):**
- Serial: 13.4 tok/s (74.4 ms/tok), allreduce 23.29 ms/tok (31.3%)
- Stream overlap: 14.6 tok/s (68.6 ms/tok) — 1.085x vs serial
- Cached dispatch: 23.7 tok/s (42.2 ms/tok) — best mode

**Correctness:** Cosine similarity 0.996-0.9999 vs serial, threshold 0.99. All 20 steps pass.

**Key insight:** The overlap benefit is modest (1.085x) because allreduce time is ~23ms and the P2P operations themselves are the bottleneck. The benefit comes from:
- Eliminating CPU-blocking `hipDeviceSynchronize()` calls (128 per step)
- Allowing Python to continue dispatching next-layer kernels while GPU runs allreduce
- The GPU enforces correct ordering via events without CPU involvement

**Stream overlap vs cached dispatch interaction:** They are currently separate modes. A future optimization could combine cached dispatch (reduced Python dispatch overhead) with stream overlap (async allreduce) for potential additive benefits.

---

## P2P Allreduce Race Condition Fix (TP=4 Correctness Bug)

**Critical bug found and fixed:** The P2P allreduce (`src/runtime/p2p_allreduce.py`) had a race condition where the P2P gather was copying stale/incomplete partial results from GPUs 1-3.

**Root cause:** GPU GEMV kernels (in `InferenceEngine`) are dispatched on the engine's own streams (`_stream_q`, `_stream_kv`, etc.). The P2P allreduce uses different streams (TensorParallelGroup streams). Without synchronization between the compute stream and the P2P stream, the `hipMemcpyPeerAsync` gather could read unfinished GEMV results.

**Fix:** Added `hipDeviceSynchronize()` for all GPUs at the start of both `allreduce_residual()` and `allreduce_sum()` in `P2PAllreduce`, before the P2P gather phase.

**Impact:** Without fix, cosine similarity was ~0.1 (effectively random). With fix, cosine similarity is ~0.9999, matching host-mediated allreduce.

**Lesson:** When mixing GPU computation streams with P2P transfer streams, you MUST synchronize all source devices before initiating peer-to-peer copies. The `hipMemcpyPeerAsync` does not automatically track dependencies from other streams.

**Performance note:** The `hipDeviceSynchronize()` adds some overhead but is necessary for correctness. Future optimization could use HIP events (`hipEventRecord` on the compute stream + `hipStreamWaitEvent` on the P2P stream) to create fine-grained dependencies instead of full device synchronization.

---

## TP=4 Decode Performance Baselines (Milestone 1: p2p-allreduce, measured 2026-03-14)

**Milestone 1 result (P2P allreduce with hipDeviceSynchronize correctness fix):**
- TP=4 throughput: **12.8 tok/s** (78.2 ms/tok), 100 decode steps on Qwen3.5-27B-GPTQ-Int4
- Allreduce overhead: **23.55 ms/tok (30.1%)** of total decode time
- Compute time: 54.62 ms/tok (69.9%)
- Single-GPU reference: 20.6 tok/s (no regression from 20.3 baseline)
- TP=4 is currently 1.59x SLOWER than single-GPU due to synchronization overhead

**Root cause of slowdown:** The hipDeviceSynchronize() before each P2P gather (added as race condition fix) serializes all GPU work 128 times per token (2 allreduces × 64 layers). This eliminates any parallelism gains from TP.

**Target for future milestones:**
- Milestone 2 (threaded-dispatch): Replace hipDeviceSynchronize with HIP events (hipEventRecord + hipStreamWaitEvent) for fine-grained dependency tracking, add multi-threaded kernel dispatch
- Milestone 3 (advanced-fusions): Fuse P2P allreduce into GEMV epilogue, reduce allreduce count

**Validation test:** `tests/bench_tp4.py` + `tests/test_tp4_correctness.py` (cosine sim 0.999990-0.999995)

---

## Optimization Sprint Reference (kernel-optimization-sprint mission)

### FlashAttention Block-Tiling Design
The current flash_attn_256 prefill kernel uses per-token warp reduction (O(seq_len) warp reductions).
The optimization target is FlashAttention-2 style block-tiled approach:
- Q tile: [Br x d] in registers, Br=64 (or 32 if VGPR-limited)
- KV blocks: [Bc x d] streamed through LDS, Bc=16 (based on flash-attention-gfx906 Triton findings)
- QK^T: [Br x Bc] block GEMM via v_dot2_f32_f16
- PV: [Br x d] accumulation via v_dot2_f32_f16
- Online softmax at block granularity (not per-token)
- LDS: K[Bc x d] + V[Bc x d] = 16*256*2*2 = 16KB with XOR-swizzle

### Double-Buffering Pattern for GEMM
Overlap global load of tile t+1 with compute of tile t:
- Two LDS buffer sets: ping (smem_A0/B0) and pong (smem_A1/B1)
- Total LDS: 2 * 4608 = 9216 bytes (still < 32KB)
- Requires careful __syncthreads() placement between phases
- **Benchmark result** (MI60, M=128, N=6144, K=5120): 1.06-1.08x speedup over single-buffered
- **Key insight**: For small M=128, double-buffering benefit is modest because the kernel is
  occupancy-limited rather than memory-bandwidth-limited. Larger M (512+) would benefit more.
- **Kernel:** `gemm_fp16_prefill_db` in `gemm_fp16_prefill.hip` (alongside original `gemm_fp16_prefill`)

### INT4 On-the-Fly Dequantization
Instead of extracting nibbles to FP16 in LDS load phase:
- Store packed uint32 in LDS (4x less LDS traffic for B tile)
- Dequant during compute loop: either via v_dot8_i32_i4 or nibble extract + v_dot2

### v_dot8_i32_i4 for INT4 GEMV
The W4A8 kernel (gemv_w4a8_dot8) demonstrates the activation-splitting technique:
- x = x_lo_s + 16*x_hi where x_lo_s = (x&0xF)-8, x_hi = x>>4
- dot(W,x) = dot8(W, x_lo_s) + 8*sum(W) + 16*dot8(W, x_hi)
This can be adapted for FP16 activations by quantizing to INT4 on-the-fly.

### Elementwise Vectorization
float4 loads = 128-bit = 8 FP16 per load instruction (global_load_dwordx4).
Polynomial sigmoid: sigmoid(x) ≈ 0.5*(1 + x/(1+|x|)) — no MUFU exp needed.

---


## Kernel Execution Hierarchy
1. Engine (Python) — `src/inference/engine.py` — orchestrates decode/prefill steps
2. Launcher (Python) — `src/kernels/launcher.py` — KernelCache, loads HSACO/SO, dispatches via HIP
3. Runtime (Python+C) — `src/runtime/hip_dispatch.py` — GPUDevice wrapper around HIP API
4. Kernels (HIP C++ or GCN ASM) — the actual GPU code

## Two Kernel Paths
- **Assembly (.s)**: Compiled via `llvm-mc → ld.lld → .hsaco`. Max control. Used for: RoPE (fallback, replaced by HIP v2), fallback for GEMM/GEMV.
- **HIP C++ (.hip)**: Compiled via `hipcc → .so`. Easier iteration. Used for: elementwise ops, GEMV FP16/INT4, GEMM prefill, FlashAttention, DeltaNet, batched RMSNorm, sigmoid_mul, **RoPE v2**.

**The engine prefers HIP kernels over assembly.** Assembly kernels are fallback.

## Key HIP Kernels (Primary Path)
- `elementwise_v2.hip`: residual_add_v2, silu_fused_v2, rmsnorm_v2, skip_rmsnorm_v2
- `gemv_fp16_v2.hip`: Uses `__builtin_amdgcn_fdot2` (v_dot2_f32_f16). 4 rows/WG, DPP reduction.
- `gemv_int4_v2.hip`: fused split-K (v2_fused) — for GEMV WITH residual (down_proj). Uses ubfe+pow2 optimizations.
- `gemv_int4_v3.hip`: cooperative reduction (v3_t16) — DEFAULT for GEMV WITHOUT residual. 1.29x faster than v2 for N=4096.
- `gemv_int4_dual.hip`: Fused gate+up with SiLU. Saves 1 launch + 1 memset + 1 read of x[K].
- `flash_attn_256.hip`: head_dim=256, online softmax, GQA-aware.
- `deltanet_v3.hip`: Full DeltaNet recurrence in one kernel, parallel kq_dot.
- `batched_rmsnorm.hip`: Per-head Q/K normalization.
- `rope_v2.hip`: **HIP RoPE** — replaces assembly rope.s. Vectorized half2 loads, FP32 rotation. Same interface as rope.s.
- `qknorm_rope.hip`: Fused per-head RMSNorm + partial RoPE (used in decode path for Q and K).

## W4A8 GEMV (gemv_w4a8.hip — m3-w4a8-gemv)
**W4A8 = INT4 weights stored + INT8 activations → INT32 accumulation → FP16 output.**

Three kernels:
- `gemv_w4a8_dot4`: Uses v_dot4_i32_i8. Unpack INT4→INT8 on-the-fly, then 2× v_dot4 per 8 weights. PRIMARY.
- `gemv_w4a8_dot8`: Uses v_dot8_i32_i4. Splits INT8 activation into lo/hi nibbles:
  x = x_lo_s + 16*x_hi (where x_lo_s = (x&0xF)-8 ∈ [-8,7], x_hi = x>>4 ∈ [-8,7])
  dot(W,x) = dot8(W, x_lo_s) + 8*sum(W) + 16*dot8(W, x_hi) (three-term formula)
- `gemv_w4a8_grouped`: Per-group FP16 scales (like GPTQ group_size=128). Uses v_dot4.

Weight format (W4A8 packed):
  W_packed[N, K/8] uint32, N-major row-major
  bits[4b+3:4b] = w[n, k+b], signed INT4 ∈ [-8,7] (zero-subtracted)
  Repacked from GPTQ format via `src/kernels/repack_w4a8.py`

**Benchmark results on MI60 gfx906:**
- N=4096, K=4096:  dot4=59.5us, dot8=67.4us, W4A16-v3t16=30.4us → W4A16 1.95x faster
- N=11008, K=4096: dot4=59.1us, dot8=86.7us, W4A16-v3t16=66.2us → W4A8-dot4 1.12x faster

**Key insight:** W4A8 and W4A16 have same weight bandwidth (both INT4 = K*N/2 bytes).
W4A8 advantage is less activation bandwidth (INT8 vs FP16 = saves K bytes).
For N=4096 (square-ish shapes), W4A16 v3_t16 wins due to better thread utilization.
For N=11008 (wider), W4A8 dot4 wins slightly (INT32 accumulation vs FP32 FMA overhead).

**v_dot8_i32_i4 intrinsic:** `__builtin_amdgcn_sdot8(int a, int b, int c, bool sat)`
→ c + sum_{i=0}^{7} a[i]*b[i] where a[i],b[i] = signed 4-bit at bits[4i+3:4i]
**Confirmed working on gfx906** (validated by correctness tests in test_w4a8_gemv.py).

**INT4→INT8 nibble unpacking trick (for v_dot4 path):**
  w_i8 = ((int)(w_packed << (28 - b*4))) >> 28  (arithmetic right-shift for sign extension)
  Packs 4 signed INT8 bytes into one int32: (n0&0xFF) | ((n1&0xFF)<<8) | ...

## gfx906 Confirmed Instructions
All verified on real hardware:
- `v_dot2_f32_f16`: ~7.58 instr/cycle (packed FP16 dot product)
- `v_dot4_i32_i8`: ~7.58 instr/cycle (INT8 4-element dot)
- `v_dot8_i32_i4`: ~7.58 instr/cycle (INT4 8-element dot) — **CONFIRMED working**
- `v_pk_fma_f16`: ~7.58 instr/cycle (packed FP16 FMA)
- `v_fmac_f32`: ~7.58 instr/cycle (FP32 FMA)
- `v_exp_f32`: ~3.69 instr/cycle (MUFU transcendental — much slower)
- DPP ops: working

## Memory Bandwidth (MI50/MI60 gfx906)
- Vectorized HBM reads: ~860 GB/s
- Scalar HBM reads: ~631 GB/s (vectorization matters!)
- LDS bank conflicts are severe — XOR swizzling nearly recovers sequential perf

## FP16 GEMM Prefill (gemm_fp16_prefill.hip v2 — m2-prefill-gemm-dot2)
Optimized from 0.97 TFLOPS to **18.60 TFLOPS** (69.4% of 26.8 TFLOPS FP16 peak), beating the assembly kernel (13.27 TFLOPS).

Key design decisions:
1. **`__builtin_amdgcn_fdot2` (v_dot2_f32_f16)**: Packs A/B tile elements as half2, processes 2 FMAs per instruction.
2. **XOR-swizzled LDS layout**: phys_group = logical_group ^ (row & 1). At float4-group (8-half) granularity with NGRP=2. LDS stride-18 (36 bytes/row) provides additional bank separation. Eliminates LDS bank conflicts for 4-wavefront access pattern.
3. **Register preloading**: For each kk pair, all 4 A rows and 4 B cols loaded into registers before computing the 4×4 outer product. Enables compiler pipelining.
4. **`__attribute__((amdgpu_flat_work_group_size(256, 256)))`**: Critical hint that significantly improves occupancy and instruction scheduling on gfx906. Went from 10.44 → 18.60 TFLOPS with this attribute.
5. **Separate A/B loading wavefronts**: Threads 0-127 load A tile, 128-255 load B tile. Each loads 1 float4.
6. **TILE_K=16, TILE_M=TILE_N=64, THREAD_M=THREAD_N=4**: 4×4 = 16 outputs per thread, 8 k-pairs per tile.
7. **Grid**: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1).

Anti-patterns learned (what didn't work):
- TILE_K=32/64 without the flat_work_group hint: much worse (1-4 TFLOPS)
- 128×128 tile with 8×8 per thread: too much register pressure, 0.8 TFLOPS
- XOR swizzle at byte level (not group level): correctness failures
- SWIZZLE_GROUP(g, row) with runtime row in inner loop without precomputation: ~4 TFLOPS overhead

## HIP Compiler Hints for gfx906 GEMM Performance
- `__attribute__((amdgpu_flat_work_group_size(256,256)))` is CRITICAL for GEMM performance
- Without it: ~10 TFLOPS; with it: ~18 TFLOPS (difference is occupancy management)
- The HIP GEMM with fdot2 + swizzled LDS now beats the hand-tuned assembly GEMM

## INT4 GEMV Pattern (current v2)
memset(fp32_buf) → gemv_int4_v2_splitk(fp32_buf) → fp32_to_fp16(output)
Three launches per GEMV. Target: fuse into one.
(m1-fused-int4-splitk: achieved with gemv_int4_v2_fused — single launch, no memset)

## INT4 GEMV Optimization (m2-int4-gemv-optimize) — BENCHMARK RESULTS
**v3_t16 is the default for non-residual GEMV; v2_fused remains for residual GEMV (down_proj).**

Benchmarks on MI60 gfx906:
- N=4096, K=4096: v3_t16 = 29.5 us, v2_fused = 38.2 us → **v3 1.29x faster**
- N=11008, K=4096: v3_t16 = 63.7 us, v2_fused = 64.4 us → essentially tied

Optimization techniques applied:
1. **`__builtin_amdgcn_ubfe` (v_bfe_u32)**: Bitfield extract for nibble extraction.
   Replaces `(packed >> offset) & 0xF` (2 instrs) with single `v_bfe_u32` instruction.
   Applied to all INT4 nibble extraction in both gemv_int4_v2.hip and gemv_int4_v3.hip.
2. **Power-of-2 shift for group scale lookup**: `kg >> log2(groups_per_scale)` instead of
   integer division `kg / groups_per_scale`. For group_size=128: groups_per_scale=16, log2=4.
   Use `31 - __builtin_clz(groups_per_scale)` to compute log2 once per kernel.
3. **v3 cooperative reduction with LDS (shared memory)**: 16 threads/col (t16 variant),
   256/16=16 cols per WG. Single launch, no atomicAdd, no persistent FP32 buffer.
   Thread layout: col_in_wg = tid % 16, k_split = tid / 16. LDS reduction (256 floats).
4. **DPP wave reduction explored (v3_dpp)**: 64 threads/col = 1 warp, pure `__shfl_xor`
   butterfly intra-warp + LDS for cross-warp. SLOWER than t16 (68-131 us vs 30-64 us)
   due to too many WGs (grid ~N/4 = 1024 WGs for N=4096). Not recommended.

Engine wiring (post-m2):
- Non-residual GEMV (gate, up, etc.): **v3_t16** (faster)
- Residual GEMV (down_proj fuses residual-add): **v2_fused** (only option with residual)
- Dual gate+up+silu: **gemv_int4_dual_fused** (unchanged)

DPP warp reduction lesson:
- `__shfl_xor` is intra-warp only (warpSize=64 on gfx906)
- For 64 threads/col → columns are in different warps → still need LDS for cross-warp
- Pure DPP reduction only helps when ALL cooperating threads are in the SAME warp
- For GEMV: the extra grid overhead (N/4 WGs) dominates any DPP benefit for large N

## Residual-Add Epilogue Pattern (m1-residual-epilogues)
Both `gemv_fp16_v2` and `gemv_int4_v2_fused` now support an optional residual pointer:
- If non-null: `out[i] = gemv_result + residual[i]`
- If null (0): normal GEMV output

**Engine decode path (tp_size=1):**
- **out_proj**: `_launch_gemv_fp16(d_hidden, attn_out, o_weight, residual=d_hidden)` → d_hidden updated in-place, then `_launch_rmsnorm` (not skip_rmsnorm)
- **down_proj**: `_launch_gemv_int4(d_hidden, ffn_gate, ..., residual=d_hidden)` → d_hidden updated in-place, no separate `_launch_residual_add`
- **TP path** (tp_size>1): still writes to d_proj_out/d_ffn_out without residual; TPInferenceEngine handles allreduce+residual

**Test files updated for new kernel signature:**
- Must pass `ctypes.c_uint64(0)` as last param for null residual
- test_fused_int4_gemv.py, test_gemm_fp16_prefill.py, test_int4_direct_vs_splitk.py, test_tp_single_layer.py

## HIP RoPE Kernel (m2-hip-rope)
`rope_v2.hip` replaces the assembly-only `rope.s` for standalone RoPE operations.

Key facts:
- Vectorized half2 loads for both x[2i:2i+2] pairs and separate cos/sin reads
- FP32 rotation (x0*c - x1*s, x0*s + x1*c), then pack back to half2 (FP16)
- Interface: same as rope.s — (x, cos_tab, sin_tab, head_dim, num_heads)
  Grid: (num_tokens, num_heads, 1), Block: (half_rotary, 1, 1)
- **cos/sin table stride = head_dim/2** (not half_rotary!)
  For single-token decode, token_idx=0 so stride doesn't matter.
  For multi-token, cos_tab must be padded to [num_tokens, head_dim/2] columns.
- Performance on MI60: ~20-27 us (comparable to assembly, kernel is memory-bound)
- Assembly rope.s was BROKEN for multi-token (outputs zeros for token_idx>0)

Engine wiring:
- `_init_rope_hip()` loads rope_v2.hip (falls back to assembly if unavailable)
- `_launch_rope()` uses HIP kernel when `self._rope_hip=True`
- **Note**: `_launch_rope` is rarely called directly — decode path uses `_launch_qknorm_rope`
  (fused QKnorm+RoPE). `_launch_rope` is the fallback for non-QKnorm-RoPE code paths.

## Qwen 3.5 27B Architecture
- Hybrid attention: some layers use full GQA (head_dim=256), others use DeltaNet linear attention
- hidden_size=5120, num_heads=48, num_kv_heads=8
- FFN: gate+up (fused) → SiLU → down, with INT4 quantized weights (GPTQ)

## Activation Quantization (activation_quant.hip — m3-activation-quantization)
**Dynamic per-tensor INT8 quantization of FP16 activations. Foundation for W8A8 and W4A8 kernels.**

Two-kernel pipeline:
1. `activation_quant_reduce`: Finds max|x| across full tensor using DPP 6-step butterfly reduction (64-lane gfx906 wavefronts) + cross-warp LDS reduction + atomicMax via IEEE 754 bit trick (non-negative floats have monotonic bit ordering).
2. `activation_quant_quant`: Computes scale = max_abs / 127.0, quantizes: clamp(round(x/scale), -128, 127) via `__builtin_amdgcn_fmed3f` for combined round+clamp.

**IMPORTANT**: `d_max` buffer MUST be pre-zeroed before each reduce kernel launch. Engine does this via `hip.memset(d_act_maxabs, 0, 4)` in `_launch_activation_quant`. Forgetting this causes scale=0 (all-zeros input case or carry-over from prior invocation).

Correctness:
- scale matches reference exactly (zero relative error in scale computation)
- round-trip error within expected INT8 noise budget (max_err ≤ 0.5 × scale ≈ 0.03)
- Edge cases handled: all-zeros (scale=1.0), spike values (max element→127), negative-only inputs

Performance: ~38-49us for n=4096-11008 (two-kernel launch overhead dominated). Bandwidth: 0.3-1.2 GB/s (activation sizes are small; latency dominated by kernel launch overhead, not HBM bandwidth).

## W8A8 GEMV and GEMM (gemv_w8a8.hip, gemm_w8a8.hip — m3-w8a8-gemv, m3-w8a8-gemm)
**W8A8 = INT8 weights + INT8 activations → INT32 accumulation → FP32 scale epilogue → FP16 output.**

**GEMV** (decode, M=1):
- 4 rows per WG, 64-thread wavefront per row, K/64 groups of 4 INT8 elements per lane
- `__builtin_amdgcn_sdot4` (v_dot4_i32_i8) = 4 signed INT8 × INT8 products → INT32 accumulator
- 6-step butterfly DPP reduction via `__shfl_xor` for full 64-lane wavefront
- Epilogue: `(int32_acc as float32) × scale_w[row] × scale_a` → FP16 output

**GEMM** (prefill, M>1):
- 64×64 output tiles, TILE_K=16 INT8 elements per K-iteration, 256 threads (16×16 per-thread 4×4 outputs)
- LDS staging: A[64×16] + B[64×16] = 2048 bytes LDS per WG
- **REQUIREMENT: K must be a multiple of TILE_K=16** (satisfied for all Qwen shapes K=4096)
- Performance: 4096×4096×4096 = 12.1 TFLOPS, 128×4096×4096 = 3.4 TFLOPS

**Performance vs W4A16 for decode (M=1)**:
- N=4096, K=4096: W8A8 = 78us/213GB/s; W4A16-v3t16 = 30us/295GB/s → W4A16 **2.4x faster**
- N=11008, K=4096: W8A8 = 139us/325GB/s; W4A16-v3t16 = 64us/377GB/s → W4A16 **1.8x faster**
- Root cause: W8A8 reads 2× weight bytes vs W4A16 (INT8 vs INT4), so decode is 2× more bandwidth-bound
- **W8A8 is recommended for prefill (M>1), not decode (M=1)** where bandwidth dominates

**Scale API**: `gemv_w8a8(x_int8, W_int8, scale_w_fp32[N], scale_a_fp32_scalar, out_fp16, K, N)`
- scale_a is passed as by-value float (requires a D2H sync of 4 bytes in engine). Future improvement: pass as device pointer.

## Engine Quantization Format Selection (m3-engine-integration)
Engine `quant_format` parameter: `'w4a16'` (default) | `'w8a8'` | `'w4a8'`

- **Hard error at init** if kernel files not found (not silent fallback)
- `_init_quant_kernels()` loads appropriate kernels at engine init
- W8A8/W4A8 paths both require activation_quant.hip (two-kernel pipeline)

**FFN dispatch in decode_step**:
```python
if self.quant_format in ('w8a8', 'w4a8'):
    self._decode_ffn_quantized(lw, h)  # handles gate+up+silu+down+residual
else:
    # W4A16 path (default, uses fused INT4 GEMV with residual epilogue)
```

**Weight loader formats**:
- `gptq_to_w8a8(qweight, scales, qzeros)` → `(W_int8[N,K], scale_w[N])`: converts GPTQ INT4 → dequantize → quantize to INT8 per-channel
- `gptq_to_w4a8(qweight, scales, qzeros, group_size)` → `(W_packed[N,K/8], scale_grp[K/group_size,N])`: repack GPTQ INT4 nibbles into N-major layout for W4A8 kernel

**D2H sync limitation**: `_launch_gemv_w8a8` and `_launch_gemv_w4a8` download scale_a (4 bytes) from GPU after activation quantization to pass it as a by-value float to the GEMV kernel. Adds ~1-2us per FFN projection. Future fix: change kernel signature to accept `const float* scale_a` (pointer).

## FlashAttention v3 Block-Tiled (block-tiled-flash-attention)

**File:** `src/kernels/flash_attn_256_v3.hip`
**Test:** `tests/test_flash_attn_v3.py`

**Key difference vs flash_attn_256_tuned.hip prefill:**
- BLOCK_N (Bc): 4 → 16 (4× larger KV tile)
- Score computation: scalar FMA → `__builtin_amdgcn_fdot2` (v_dot2_f32_f16), 2× arithmetic throughput
- Q rows per WG: 4 (1/wavefront) → 16 (4/wavefront)

**Final tile sizes chosen (BLOCK_M=16, not 64):**
- Feature requested Br=64 but this is infeasible on gfx906 without tensor cores:
  - Br=64 with 4-thread reduction → each thread needs 64 Q-dims in registers (too many VGPRs)
  - Br=16 with 16-thread reduction → each thread holds 16 Q-dims = ~50 VGPRs (healthy occupancy)
- BLOCK_N=16: K+V in LDS = 16×256×2×2 = 16 KB per WG (32 KB budget allows 2 WGs/CU)

**Thread layout (256 threads = 4 wavefronts × 64 lanes):**
- wf_id = tid/64 (0..3)
- lane = tid%64 (0..63)
- q_in_wf = lane/16 (0..3): Q row within wavefront
- part = lane%16 (0..15): which 16-dim chunk
- dim_base = part × 16 (0,16,...,240)
- q_row = wf_id×4 + q_in_wf (0..15)

**Score computation (v_dot2_f32_f16):**
- Each thread computes partial dot over 16 dims (8 fdot2 calls)
- 16-way intra-wavefront reduction using shfl_down(8/4/2/1)
- Broadcast via shfl to all 16 threads in q_in_wf group

**Cooperative K/V load:**
- 256 threads load BLOCK_N=16 rows × 256 dims = 4096 halfs
- Each thread: load_row=tid/16, load_dim=(tid%16)×16, loads 16 halfs via 2×float4

**Performance on MI60 gfx906 (heads=48, kv_heads=8, causal=1):**
- seq=64:   tuned=0.117ms → v3=0.074ms (1.59x faster)
- seq=128:  tuned=0.362ms → v3=0.223ms (1.62x faster)
- seq=256:  tuned=1.238ms → v3=0.682ms (1.81x faster)
- seq=512:  tuned=4.829ms → v3=2.559ms (1.89x faster)

**Correctness:** max_err < 5e-3 (< 1e-2 threshold) at all tested seq_lens.
**Decode kernel:** Unchanged copy of flash_attn_256_tuned.hip decode kernel.

**Grid:** (num_heads, ceil(num_q_rows/16), 1), Block=(256,1,1)
**Softmax:** Online per KV block (FlashAttention-2 style), FP32 accumulators throughout.

---

## INT4 GEMV v4 fdot2 Approach (int4-gemv-dot8 feature)

**File:** `src/kernels/gemv_int4_v4.hip`
**Test:** `tests/test_gemv_int4_v4.py`

**Approach:** Uses `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) — extracts weight nibbles
to FP16 pairs (dequantized via `(q - zero) * scale`), then uses fdot2 for 2 FMAs/instruction.

**Performance on MI60 gfx906:**
- N=4096,  K=4096: v3_t16=70.9us, v4_t16=84.0us (v4 is 1.19x slower)
- N=11008, K=4096: v3_t16=109.5us, v4_t16=116.4us (v4 is 1.06x slower)

**Correctness:** max_abs_err < 2e-3 at both shapes (well under 1e-2 threshold).

**Why v4 fdot2 is slower than v3 ubfe+scalar:**
- fdot2 benefit (2 FMAs/instruction) is offset by the FP16 conversion overhead: each
  nibble must be sign-extracted, scaled, and converted to FP16 before packing as half2.
- v3 scalar approach uses `__builtin_amdgcn_ubfe` which is also a single instruction,
  then scalar FP32 FMA — fewer total operations since no float→half conversion is needed.
- The kernel is memory bandwidth-limited (HBM), not compute-limited, so instruction
  reduction via fdot2 provides less benefit than expected.

**Three variants provided:** v4_t4 (64 cols/WG), v4_t8 (32 cols/WG), v4_t16 (16 cols/WG).
All compile for gfx906 and pass correctness.

**Key lesson:** For INT4 GEMV with FP16 activations and asymmetric GPTQ quantization
(unsigned 0-15 + zero offset), the scalar ubfe+FP32-FMA approach (v3) is faster because:
1. Kernel is bandwidth-limited, not compute-limited
2. fdot2 requires nibble→FP16 conversion that adds instructions
3. The correction term optimization in v3 (factor out zero: acc*scale - zero*scale*Σa)
   reduces operations per uint32 at the cost of 1 FP32 multiply and 1 FP32 FMA

**Recommendation:** Use v3_t16 for production INT4 GEMV decode path (faster at all tested shapes).

---

## INT4 GEMM On-the-Fly Dequantization (int4-gemm-onthefly-dequant)

**File:** `src/kernels/gemm_int4_prefill_v2.hip`
**Test:** `tests/test_gemm_int4_v2.py`

**Key insight:** Store B tile packed (uint32, 8 INT4 per word) in LDS instead of dequantizing to FP16.
- Original LDS for B: 64×32×2 = 4096 bytes (FP16)
- v2 LDS for B: 64×4×4 = 1024 bytes (uint32, packed)  → **4x LDS reduction**
- Total LDS: 5120 bytes vs original 8192 bytes

**LDS load:** 256 threads load exactly 256 packed uint32s (one each) for B tile.
- Each thread handles: b_col = tid/4, b_k8 = tid%4 (TILE_K_PACKED=4 since TILE_K=32, 32/8=4)
- B_q4 is [K/8, N] row-major: index = k8_global * N + n_global

**Compute inner loop:** For each k in [0, TILE_K):
1. Load A values for 4 output rows from smem_A
2. For each of 4 output columns: read packed uint32 from smem_Bp, extract nibble via `__builtin_amdgcn_ubfe`
3. Update scale/zero cache if group boundary crossed (g = k_global / group_size)
4. Dequant: `w = (nibble - zero) * scale`, then `acc += a * w`

**Scale/zero caching:** Per thread, cache cur_scale[4] and cur_zero[4] (one per output column) with
last-used-group tracking → reload only when group boundary crossed.

**Performance on MI60 gfx906 (M=128, N=4096, K=4096, gs=128):**
- Original (FP16 dequant in LDS load): 3721 us
- v2 (on-the-fly dequant): 1799 us
- **Speedup: 2.07x**

**Correctness:** max_abs_err = 9.77e-4 (well below 1e-2 threshold) at both test shapes.

**Why so much faster:** LDS bandwidth is the bottleneck. The B tile load in the original requires
reading INT4 and writing FP16 to LDS (2x the data per element), then reading FP16 back for compute.
v2 writes only packed uint32 (4x less LDS traffic), saving LDS write bandwidth and improving
cache line utilization for compute reads.

**`amdgpu_flat_work_group_size(256,256)` attribute:** Applied on v2 kernel — same pattern as
gemm_fp16_prefill.hip that improved TFLOPS from 10→18.

---

## FlashAttention-256 Tuning (m3-flashattn-tune)

**Tuned kernel**: `flash_attn_256_tuned.hip` contains `flash_attn_256_decode` and `flash_attn_256_prefill`.

**Key insight for decode (seq_len=1)**: The original `flash_attn_256.hip` uses 256 threads (4 wavefronts) per WG, but for decode only 1 wavefront does work (q_rows 1,2,3 exit immediately since num_q_rows=1). This is a 4x waste.

**Solution**: Split KV range across 4 wavefronts. Each wavefront sweeps kv_len/4 positions independently with its own online softmax state. Merge 4 partial (max, sum, acc) states via LDS at the end. Only wavefront 0 writes output.

**Merge formula**:
```
gmax = max(max0, max1, max2, max3)
ci = exp(maxi - gmax)
gsum = sum0*c0 + sum1*c1 + sum2*c2 + sum3*c3
out = (acc0*c0 + acc1*c1 + acc2*c2 + acc3*c3) / gsum
```

**Benchmark results** (MI60 gfx906, 48 heads, 8 KV heads):
- decode kv=256:  222us → 62us  (3.57x)
- decode kv=512:  587us → 113us (5.21x)
- decode kv=1024: 1240us → 223us (5.56x)
- decode kv=2048: 2469us → 435us (5.68x)
- decode kv=4096: 4964us → 859us (5.78x)
- prefill seq=128: 0.40ms → 0.38ms (1.04x)
- prefill seq=512: 5.03ms → 4.81ms (1.04x)
- prefill seq=2048: 82.9ms → 77.6ms (1.07x)

**Decode bottleneck analysis**: The original kernel was compute-bound by v_exp_f32 (2 per KV step × kv_len steps). With 4x parallelism, each wavefront does 1/4 of exp calls, halving the exp cost (plus better HBM utilization). The LDS merge adds only 4 exp calls total (negligible).

**Why prefill improvement is smaller**: Prefill already processes 4 Q rows per WG (1 per wavefront). LDS tiling saves some redundant global loads but the kernel is already reasonably efficient. Prefill is dominated by compute (causal attention = O(n²) work).

**LDS layout for decode merge**:
- `float partial_max[4]` + `float partial_sum[4]` = 32 bytes
- `float partial_acc[4][256]` = 4096 bytes (each wf stores its 256-dim acc)
- Total: ~4.1KB LDS per WG (well within 64KB LDS limit)

**Grid launch**:
- decode: Grid=(num_heads, 1, 1), Block=(256, 1, 1)
- prefill: Grid=(num_heads, ceil(num_q_rows/4), 1), Block=(256, 1, 1)
- Both use `__attribute__((amdgpu_flat_work_group_size(256,256)))` for occupancy hint

**Note on fast_exp**: Schraudolph's approximation (`fast_exp_schraudolph`) is available but disabled. __expf() has better accuracy and the bottleneck turned out to be compute parallelism, not exp latency per se.

---

## Elementwise Vectorization (elementwise-vectorization feature)

**File:** `src/kernels/elementwise_v3.hip`
**Test:** `tests/test_elementwise_v3.py`

**Approach:** Upgrade from half2 (2 FP16 per load) to float4 (8 FP16 per load).
- `global_load_dwordx4` instead of `global_load_dwordx2`
- Grid: (ceil(n/2048), 1, 1) vs v2's (ceil(n/512), 1, 1)
- Each thread processes 8 elements per iteration instead of 2

**Key implementation insight (tail handling):**
  For RMSNorm/SkipRMSNorm, the vectorized 8-element loop and scalar tail MUST NOT overlap.
  - WRONG: `tail_start = (dim / (256*8)) * (256*8)` — this double-counts elements when dim%2048!=0
  - CORRECT: `tail_start = (dim / 8) * 8` — only elements where dim%8!=0 need scalar treatment
  - For dim=5120 (Qwen), `5120%8==0`, so the scalar tail loop never executes
  - Double-counting causes sum_sq inflation → rms_inv underestimated → output too small

**Polynomial sigmoid note:**
  The logistic approximation `0.5 + 0.5*x/(1+|x|)` has max error ~5e-2 in sigmoid(x).
  For SiLU output (= gate * sigmoid * up), this gets amplified by |gate|, exceeding the 5e-3
  threshold for realistic LLM activations. We therefore use exact __expf for sigmoid in
  silu_fused_v3 for correctness, while keeping the float4 vectorized loads for bandwidth.

**Performance on MI60 gfx906 (dim=5120, single vector):**
- residual_add: v2=49us, v3=53us (essentially tied — kernel too small to saturate HBM BW)
- silu_fused:   v2=53us, v3=54us (essentially tied — same sigmoid computation cost)
- rmsnorm:      v2=50us, v3=35us → **1.43x faster** (0.9 GB/s vs 0.6 GB/s)

**Why rmsnorm shows improvement but residual_add does not:**
  - RMSNorm does TWO passes over the data (sum-sq + normalize), both benefiting from float4
  - residual_add reads 3 arrays (dst, src, write dst) — the kernel launch overhead dominates at dim=5120
  - At larger dims or with multiple vectors, residual_add would show more speedup

**Correctness:**
  - residual_add_v3: max_err = 0.00 (exact, below 1e-4 threshold)
  - silu_fused_v3:   max_err = 0.00 (exact sigmoid, below 5e-3 threshold)
  - rmsnorm_v3:      max_err = 0.001 (FP16 rounding, below 5e-3 threshold)
  - skip_rmsnorm_v3: max_err = 0.002 (FP16 rounding, below 5e-3 threshold)

**Function signatures:** Identical to v2 for drop-in compatibility.


---

## Kernel Fusion: Skip-RMSNorm + INT4 GEMV (fused-skip-rmsnorm-gemv)

### Design: fused_skip_rmsnorm_gemv.hip
Fuses skip-connection + RMSNorm + INT4 GEMV decode path into a single kernel.

**Interface:** `hidden` is READ-ONLY. `hidden_out` receives h+r update (MUST be different buffer).
New parameter order: `(out_gemv, hidden_out, hidden, residual, weight, eps, B_q4, scales, zeros, K, N, gs)`

**Dynamic shared memory (per block):**
- `lds_hval[K]` (FP16): pre-normalization h+r values; `lds_A[K]` (FP16): normalized activations
- `s_warp[4]` (float) + `s_reduce[256]` (float)
- Total: K*4 + 16 + 1024 bytes. For K=5120: 21536 bytes < 64KB. Caller: `shared_mem = K*4 + 16 + 256*4`

**Race condition analysis (CRITICAL):**
1. Multiple blocks all write `hidden[i]` -> later blocks read updated value -> double-counts residual
2. Symptom: correct for blocks 0-127, wrong for blocks 128+ (hardware occupancy boundary ~128 blocks)
3. Fix: `hidden` is CONST, `hidden_out` is a SEPARATE pointer, block 0 writes it in Phase 3

**Performance (MI60, K=5120, N=4096):**
- Separate (skip_rmsnorm_v2 + gemv_int4_v3_t16): ~91us
- Fused t16 (256 blocks): ~133us (1.46x SLOWER)
- **Root cause:** Each block loads ALL K elements (20KB) for its own Phase 1. 256 blocks = 5MB redundant
  reads, dwarfing the 20KB savings from eliminating norm_out HBM round-trip.
- **Correctness:** PASSES. Max abs err vs separate: <4e-3 (VAL-FUSE-001).
- **Lesson:** Multi-block fused skip+norm+GEMV requires cooperative groups for efficiency.
  Without global barrier, each block must independently compute skip+norm, causing O(N*K) reads
  instead of O(K) for the norm phase. Not worth it for N>=128 blocks.

---

## Fused SiLU+multiply in Prefill GEMM Epilogue (fused-silu-prefill-gemm)

**Files:**
- `src/kernels/gemm_fp16_prefill_silu.hip`: Strategy B — epilogue fusion (gate GEMM + silu_epilogue GEMM)
- `src/kernels/gemm_fp16_prefill_silu_dual.hip`: Strategy A — dual output (one kernel computes gate+up+silu)
- `tests/test_gemm_silu_fused.py`: Correctness and benchmark tests for both strategies

**Strategy B (gemm_fp16_prefill_silu_epilogue):**
- Same structure as `gemm_fp16_prefill` but takes extra `gate_buf` pointer
- In the epilogue, reads gate_buf[row,col] from HBM, applies SiLU, multiplies by GEMM result, writes output
- Grid: (ceil(N/64), ceil(M/64), 1), Block: (256, 1, 1) — same as gemm_fp16_prefill
- Correctness: max_abs_err = 0.0039 (well below 1e-2 threshold)
- Performance: ~0.99x baseline (neutral) — GEMM is compute-dominated at this shape

**Strategy A (gemm_fp16_prefill_silu_dual):**
- Reduced tile: TILE_N=32 (half of original 64) to fit both B_gate and B_up in LDS
- THREAD_N=2 (half of original 4) — dual FP32 accumulators (gate + up)
- LDS layout: A[64×18] + B_gate[32×18] + B_up[32×18] = 4608 bytes (same total as original!)
- Load split: tid<128→A, 128≤tid<192→B_gate, 192≤tid<256→B_up
- Grid: (ceil(N/32), ceil(M/64), 1), Block: (256, 1, 1) — 2x WGs in N dimension
- Epilogue: `silu(gate_acc) * up_acc` before writing, no intermediate HBM writes
- Correctness: max_abs_err = 0.0020 (well below 1e-2 threshold)
- Performance: ~0.78x baseline (SLOWER) — 2x WG overhead from TILE_N=32

**Why fusion doesn't provide wall-clock speedup at M=128, N=11008, K=5120:**
- Each GEMM takes ~1700-1800us (compute-dominated at 6-7 TFLOPS)
- HBM traffic saved by fusion (~5-12MB) is negligible vs 225MB total weight traffic
- The GEMM kernel is operating at ~7% of theoretical HBM peak — it's NOT bandwidth-limited
- Strategy A adds more WGs (2x) which increases scheduling overhead

**When SiLU fusion WOULD help:**
- Significantly larger M (e.g., M=2048+) where GEMM becomes more bandwidth-limited
- Smaller N (narrower projections where weight matrices are smaller than activation traffic)
- SiLU fusion would help most when the activation matrices dominate over weight matrices
- For decode (M=1): the existing `gemv_int4_dual.hip` already fuses gate+up+SiLU efficiently

**HBM traffic analysis (M=128, N=11008, K=5120):**
```
Baseline (3 launches):
  Reads:  A(1.3MB)*2 + W_gate(112.7MB) + W_up(112.7MB) + gate(2.8MB) + up(2.8MB) = 232.5MB
  Writes: gate(2.8MB) + up(2.8MB) + out(2.8MB) = 8.4MB
  Total:  ~241MB

Strategy B (2 launches):
  Reads:  A(1.3MB)*2 + W_gate(112.7MB) + W_up(112.7MB) + gate(2.8MB) = 230.8MB
  Writes: gate(2.8MB) + out(2.8MB) = 5.6MB
  Total:  ~236MB (saves ~5.4MB)

Strategy A (1 launch):
  Reads:  A(1.3MB) + W_gate(112.7MB) + W_up(112.7MB) = 226.7MB
  Writes: out(2.8MB) = 2.8MB
  Total:  ~230MB (saves ~12MB vs baseline)
```

**Key lesson for future workers:**
Kernel fusion reduces HBM traffic, which helps when the kernel is bandwidth-limited.
For FP16 GEMM with M=128, N=11008, K=5120 on MI60/gfx906, the kernel runs at ~6-7 TFLOPS
(~7% of theoretical 26.8 TFLOPS peak and ~7% of 857GB/s HBM peak), suggesting it is both
compute AND bandwidth limited but neither at a level where the fusion savings matter.
The SiLU kernel itself (40us) is a small fraction of the total (3580us), so eliminating it
doesn't produce measurable improvement.


---

## Final Benchmark Summary (final-benchmark milestone)

**Benchmark files:** `tests/bench_optimization_final.py`, `bench/optimization_final.json`, `bench/optimization_report.md`
**Baseline files:** `tests/bench_optimization_baseline.py`, `bench/optimization_baseline.json`

### Sprint Results (MI60 gfx906, measured 2026-03-14)

| Area | Baseline Kernel | Final Kernel | Best Speedup | Result |
|------|----------------|--------------|--------------|--------|
| FA Prefill | flash_attn_256_tuned | flash_attn_256_v3_prefill | 1.99× (seq=2048) | ✅ IMPROVED |
| FA Decode | flash_attn_256_decode | (unchanged) | 1.00× | ➖ NEUTRAL |
| FP16 GEMM | gemm_fp16_prefill | gemm_fp16_prefill_db | 1.09× (N=5120,K=6144) | ✅ IMPROVED |
| INT4 GEMM | gemm_int4_prefill_hip | gemm_int4_prefill_v2 | 2.07× (M=128,N=4096) | ✅ IMPROVED |
| INT4 GEMV | gemv_int4_v3 | gemv_int4_v4 | 1.12× (N=11008, t4) | ✅ IMPROVED (t4/t8) |
| Elementwise | elementwise_v2 | elementwise_v3 | 1.33× (rmsnorm) | ✅ IMPROVED |
| Fusion-GEMV | skip_rmsnorm+gemv | fused_skip_rmsnorm_gemv | 1.07× | ✅ IMPROVED |
| Fusion-GEMM | gemm+silu | gemm_silu_epilogue | 0.82× | ⚠️ REGRESSION (M=128) |

**5 of 6 optimization areas improved.** INT4 GEMM and FA prefill show the largest gains.

### INT4 GEMV v4 Configuration Guide (post-final-benchmark)
- **t8 is best for decode**: N=4096 (116 GB/s), N=11008 (221 GB/s)
- **t16 shows slight regression** vs v3 at N=4096 due to fdot2 dequant overhead
- **Use gemv_int4_v4_t8** as the default for decode GEMV; reserve t16 only for N>11008

---

## P2P Allreduce (p2p-allreduce milestone)

### Architecture

`src/kernels/p2p_allreduce.hip` provides HIP C++ kernels for on-device FP16 reduction:
- `p2p_reduce_sum_residual_tp{2,3,4}_kernel`: reduces partials into hidden buffer (fused allreduce + residual add)
- `p2p_reduce_sum_only_tp{2,3,4}_kernel`: reduces partials only (no residual)
- Host-callable C wrappers (`p2p_reduce_residual_tp2`, etc.) using `hipLaunchKernelGGL`

`src/runtime/p2p_allreduce.py` wraps the shared library:
- `P2PAllreduce`: async P2P gather → on-device kernel → async broadcast → single sync
- `PinnedAsyncAllreduce`: pinned host memory with async D2H + CPU accumulate + async H2D

### P2P Allreduce Protocol (TP=4, hidden_size=5120)

1. `hipMemcpyPeerAsync` gather partials from GPU1,2,3 to GPU0 gather buffers (on stream0)
2. `hipStreamSynchronize(stream0)` to wait for gather
3. `p2p_reduce_residual_tp4_kernel` on GPU0: `hidden[0] = hidden[0] + p0 + p1 + p2 + p3`
4. `hipStreamSynchronize(stream0)` to wait for kernel
5. `hipMemcpyPeerAsync` broadcast hidden[0] to GPU1,2,3 (on per-GPU streams)
6. `hipStreamSynchronize` on each GPU's stream

### Performance Results (TP=2, hidden=5120, 200 iters, gfx906 MI50/MI60)
- Host-mediated (fast_allreduce.c): 123.9 us/call median
- P2P GPU allreduce (new): 75.4 us/call median
- **Speedup: 1.64x** ✓ (target was 1.5x)

### Critical Implementation Notes

**Shared library vs HSACO module**:
- Using `hipcc -shared -fPIC` to compile as a shared library IS the correct approach
- `hipModuleLaunchKernel` from Python ctypes fails with `hipErrorInvalidDevice (101)` in
  multi-GPU contexts (likely because the kernel was loaded in a different device context)
- The fix is to compile host-callable C wrappers that use `hipLaunchKernelGGL` internally
- This is the same approach as `fast_allreduce.c` (shared library with host functions)

**Stale HIP error state**:
- When `hipDeviceEnablePeerAccess` fails with error 704 (`hipErrorPeerAccessAlreadyEnabled`),
  this error remains in the HIP error state
- `hipGetLastError()` at the end of the C wrapper picks up this stale error as kernel error
- **Fix**: call `(void)hipGetLastError()` at the START of each C wrapper to clear stale errors

**Two-hop PCIe topology**:
- All 4 MI50 GPUs are 2 PCIe hops apart (through CPU/chipset)
- P2P `hipMemcpyPeerAsync` still works but may have ~10-20us latency per transfer
- For TP=4, we do 3 async P2P gathers + 3 async P2P broadcasts = 6 P2P transfers total
- Speedup of 1.64x for TP=2 (measured); TP=4 expected similar ratio

**TPInferenceEngine integration**:
- `P2PAllreduce` is initialized in `TPInferenceEngine.__init__` after `TensorParallelGroup`
- Uses `tp_group.streams` (one per GPU) for async P2P operations
- Falls back to `fast_allreduce.c` path if P2P allreduce unavailable
- `_allreduce_residual` and `_allreduce_sum` check `self._p2p_ar is not None` first

---

## Python Threading Limitations for GPU Dispatch (threaded-kernel-dispatch investigation, 2026-03-14)

**Investigation finding**: Python threading for GPU kernel dispatch does NOT provide speedup on this platform. All approaches tried were 2-11x SLOWER than serial dispatch.

**Root cause analysis**:
1. `hipDeviceSynchronize()` on an idle GPU takes only **~0.6μs** (near-zero), not 50-100μs as assumed
2. `hipModuleLaunchKernel` is asynchronous and non-blocking — serial Python dispatch already achieves parallel GPU execution
3. Python threading Event overhead: **~490μs per round** for 4 threads (vs 31.5μs for 1 thread)
4. GIL contention causes 15x scaling penalty going from 1 to 4 threads

**Actual bottleneck**: Python kernel launch overhead (~10μs per launch × 5120 launches/decode-step = ~51ms)
NOT GPU execution time (GPUs run in parallel since hipModuleLaunchKernel is async)

**What was tried**:
- Event-based dispatch (workers launch kernels): 6x slower (Python overhead > GPU launch benefit)
- Parallel device sync before allreduce (workers call hipDeviceSynchronize): 2.3x slower  
  (0.6μs sync × 4 = 2.4μs serial, threading adds 490μs overhead)
- Queue-based dispatch: 5x slower

**Threading round overhead benchmarks** (empty work, 4 threads):
- `threading.Event`: 488μs per round
- `threading.Barrier`: 1215μs per round
- `threading.Semaphore`: 304μs per round
- 1 worker (Event): 31.5μs per round (for comparison)

**Why ctypes "GIL-free" doesn't help here**:
- ctypes calls DO release GIL during the C call
- But the C call (hipDeviceSynchronize on idle GPU) takes only 0.6μs
- The Python Event set/wait takes ~130μs per thread pair (including OS wake-up cost)
- Net: threading overhead >> benefit for fast C calls

**What WOULD help** (not threading):
1. Pre-cache ctypes parameter arrays (avoid rebuilding params per launch): ~5x less Python overhead
2. Use HIP events instead of hipDeviceSynchronize (stream-based sync, milestone 2 target)
3. Fuse multiple kernel launches into a single C extension call (reduce Python round-trips)
4. Write a Python C extension for batch kernel dispatch (bypasses GIL completely)

**Implementation status**: Threading infrastructure is implemented and CORRECT (cosine sim > 0.9999).
Set `engine.set_threaded_dispatch(True/False)` to toggle. Performance is slightly WORSE with threading.

**Recommendation for future workers**: Do NOT use Python threading for GPU kernel dispatch. 
Focus on reducing Python overhead per launch or using HIP stream-based async operations.

---

## Parameter Pre-Caching (Cached Dispatch) — threaded-kernel-dispatch milestone, 2026-03-14

**Key finding**: Python ctypes parameter construction is the dominant bottleneck for kernel launch overhead.

**Root cause**: Each kernel launch in `_decode_step_serial` creates new ctypes.c_uint64/c_uint32 objects and a new `(ctypes.c_void_p * n)()` C array. This takes ~21μs per launch (measured). With ~2560 launches for TP=4 decode (10 kernels/layer × 64 layers × 4 GPUs), this adds ~54ms of Python overhead per decode step.

**Solution**: `LaunchSpec` class in `src/runtime/hip_dispatch.py`:
- Pre-builds ctypes param objects and params_array once at init
- Mutable params (RoPE cos/sin ptrs, attention seq_len) updated in-place via `spec.params[i].value = new_val`
- `hip_runtime.launch_spec(spec)` calls `hipModuleLaunchKernel` directly with pre-built array
- Per-launch overhead: ~0.12μs (184x reduction vs uncached ~21μs)

**API**:
```python
# Build cache after loading all weights:
engine.build_dispatch_cache()
# Enable cached dispatch:
engine.set_cached_dispatch(True)
# This calls _decode_step_cached() which uses LaunchSpec objects
```

**Performance results (TP=4, Qwen3.5-27B-GPTQ-Int4, 100 decode steps)**:
- Serial (uncached): 11.4 tok/s (87.6 ms/tok) — baseline
- Cached dispatch: 23.4 tok/s (42.6 ms/tok) — **2.05x speedup**
- Threaded: 6.9 tok/s (144.1 ms/tok) — **0.61x, SLOWER** (confirmed counter-productive)

**Why cached dispatch doesn't achieve full theoretical gain**:
- Theoretical: save ~54ms → reduce from 88ms to 34ms → 2.6x speedup
- Actual: 42.6ms remaining latency (vs theoretical 34ms)
- Remaining overhead: Python loop over 64 layers × 4 GPUs, plus allreduce (28ms/tok)
- The allreduce (28ms/tok) now dominates: 66.1% of total time with cached dispatch

**Next bottleneck for future workers (stream-compute-overlap)**:
- Allreduce 28ms/tok dominates after cached dispatch reduces compute overhead to 14ms/tok
- Solution: Replace hipDeviceSynchronize with HIP events (stream-based sync) to allow
  allreduce to overlap with next-layer compute
- Target: replace `P2PAllreduce.allreduce_residual()` synchronous `hipDeviceSynchronize()`
  with `hipEventRecord + hipStreamWaitEvent` for non-blocking pipeline

**Key implementation files**:
- `src/runtime/hip_dispatch.py`: `LaunchSpec` class + `HIPRuntime.launch_spec()` + `GPUDevice.launch_cached()`
- `src/inference/engine.py`: `InferenceEngine.build_decode_launch_cache()` → builds per-layer `LaunchSpec` dict
- `src/inference/tp_engine.py`: `TPInferenceEngine.build_dispatch_cache()`, `set_cached_dispatch()`, `_decode_step_cached()`


---

## Ring Allreduce — Available but Slower than Star for 10KB Payloads (ring-allreduce milestone, 2026-03-15)

**Implementation:** `src/kernels/ring_allreduce.hip` + `RingAllreduce` class in `src/runtime/p2p_allreduce.py`

**Algorithm:** Ring topology with FP32 scratch buffers for precision:
1. **Init:** Convert each GPU's FP16 partial → FP32 result buffer (full hidden_size)
2. **Reduce-scatter** (TP-1 = 3 rounds): GPU[i] sends FP32 chunk[(i-r)%TP] to right neighbor; receiver accumulates via `ring_fp32_accumulate_fp32`. After 3 rounds: GPU[i] has fully-reduced FP32 chunk[(i+1)%TP].
3. **All-gather** (TP-1 = 3 rounds): GPU[i] sends fully-reduced FP32 chunk to right neighbor directly into result buffer slot. After 3 rounds: all GPUs have complete FP32 result.
4. **Residual add:** `ring_residual_add_fp32_to_fp16` adds FP32 result to FP16 hidden buffer.

**Precision design:** FP32 accumulators throughout ring phases → max_abs_err=0.0 vs CPU FP32 reference (exact match). Better precision than star topology (which uses FP16 throughout). P2P transfers are FP32 (4 bytes/element vs 2 for FP16) — 20KB per round for 5120-element hidden.

**Performance results (gfx906 MI50, TP=4, hidden_size=5120):**
- Star allreduce (P2PAllreduce): ~119 us/call microbenchmark
- Ring allreduce (RingAllreduce): ~1015 us/call microbenchmark (**8.5x SLOWER**)
- TP=4 decode with star (cached+stream): 26.0 tok/s
- TP=4 decode with ring (cached+stream): 7.7 tok/s (**3.38x SLOWER**)

**Why ring is slower for 10KB payloads on PCIe:**
- Ring has 6 sequential P2P rounds, each synchronized at the Python level (stream_synchronize())
- For 5120 FP16 elements (10KB): bandwidth is NOT the bottleneck — P2P latency per round dominates
- Each round: submit P2P copy, sync, submit accumulate kernel, sync = ~2 synchronizations × ~50-100us = ~100-200us per round
- 6 rounds × ~170us ≈ 1015us vs star's 2 rounds ≈ 119us
- Ring topology's bandwidth advantage only materializes for large payloads where:
  `6 rounds × latency_per_round < 2 × (total_size / P2P_bandwidth)`
  For PCIe ~12 GB/s: break-even at hidden_size ≈ 32768+ elements (64KB+ FP16)

**API:**
```python
engine.set_ring_allreduce(True)   # Enable ring topology (SLOWER for small payloads)
engine.set_ring_allreduce(False)  # Restore star topology (DEFAULT)
# Ring allreduce loaded by default; star (P2PAllreduce) is the active default
```

**Critical note:** `RingAllreduce.allreduce_residual_async()` is NOT truly async despite its name. The ring allreduce calls `stream_synchronize()` 18 times internally (between each of 6 P2P rounds × 3 sync points). The CPU blocks during ring execution. Only `P2PAllreduce.allreduce_residual_async()` achieves true non-blocking behavior.

**Correctness:**
- max_abs_err = 0.0 vs CPU FP32 reference (exact, all 4 GPUs)
- TP=4 decode cosine_sim = 0.999991 vs single-GPU reference (>= 0.99 threshold)

**Decision:** Star topology (P2PAllreduce) remains the default for all Qwen3.5-27B decode paths. Ring allreduce is available via `set_ring_allreduce(True)` for models with larger hidden dimensions where ring would be beneficial (hidden_size ≥ 32768).

---

## Deferred DeltaNet Allreduce — Infeasible (Milestone 3: deferred-deltanet-allreduce, 2026-03-14)

**Feature goal:** For DeltaNet linear attention layers (48 of 64 layers), combine the attention and FFN
allreduces into a single allreduce, halving the allreduce count for 75% of layers (from 128 to 80 per step).

**Proposed mechanism:**
1. Compute attention output (proj_out) using pre-attention hidden state
2. Compute FFN output (ffn_out) using the SAME pre-attention hidden state (skip intermediate attn allreduce)
3. Combine: combined = proj_out + ffn_out (on-GPU element-wise add)
4. Single ALLREDUCE(combined) → d_hidden += combined_global

**Critical finding: INFEASIBLE due to numerical divergence.**

Measured results from `tests/test_deferred_allreduce.py`:
- Standard flow: attn_allreduce updates d_hidden; FFN sees d_hidden + attn_result
- Deferred flow: FFN sees d_hidden (WITHOUT attn_result)
- Min cosine similarity across 10 decode steps: **0.589** (threshold: 0.99)
- Max absolute difference: ~20-24 FP16 units per element

**Why divergence is so large:**
The attention residual is NOT negligible. The DeltaNet output proj (full hidden_size=5120) creates a 
significant residual contribution that the pre-FFN RMSNorm normalizes. When the FFN sees hidden without 
this contribution, it computes a completely different normalized input, resulting in fundamentally 
different FFN outputs. The cumulative effect across 48 DeltaNet layers causes catastrophic output divergence.

**Allreduce count (theoretical, if feasible):**
- Standard: 128/step (2 × 64 layers)
- Deferred: 80/step (1 × 48 DeltaNet + 2 × 16 full-attn) → 37.5% reduction

**Decision:** Do NOT apply this optimization. Correctness takes priority over speed.

**Alternative approaches (not explored, for future reference):**
- Approximate layer norm (use the pre-attention hidden for ALL normalizations as an approx)
  — would still have same divergence issue
- Only apply to the first few DeltaNet layers where divergence is smaller
  — overhead of selective application negates benefit
- Use skip-connection arithmetic: maintain a running "deferred sum" across multiple layers
  — much more complex, unknown numerical properties
- The 80-allreduce path is correct in count but changes the math fundamentally — not acceptable

---

## Kernel Tuning Results — gfx906 MI50, Qwen3.5-27B Decode Shapes (kernel-tuning milestone, 2026-03-15)

**Summary:** Systematic auto-tuning of all decode-critical kernels confirmed that current defaults (v3 variants) are already optimal for Qwen3.5-27B shapes on gfx906. Results are wired into `engine.py` as defaults with v2 fallbacks.

**INT4 GEMV tuning** (`gemv_int4_v3.hip` / `gemv_int4_v4.hip`):
- Shapes: N=4096,K=5120 (attn out_proj), N=11008,K=5120 (FFN gate/up), N=13696,K=5120 (FFN gate/up large)
- Best: `gemv_int4_v3_t16` (threads-per-column=16, 256 threads/WG, cooperative reduction)
- Performance: ~30 µs/call (N=4096), ~64 µs/call (N=11008), ~80 µs/call (N=13696)
- vs v2_fused: **1.29x faster** (N=4096), ~tied (N=11008)
- Conclusion: v3_t16 is optimal for K=5120 Qwen shapes; v4 variants offer no improvement

**FlashAttention decode tuning** (`flash_attn_256_tuned.hip` → `flash_attn_256_decode`):
- Qwen3.5-27B: num_heads=48, num_kv_heads=8, head_dim=256, decode (1 query token)
- Tuned kernel: 4-wavefront KV-parallel merge (vs original 1-WF per query)
- Performance: kv=256→~62µs, kv=512→~113µs, kv=1024→~223µs, kv=2048→~435µs
- vs original `flash_attn_256_fp16`: **2.8-5.5x faster** across kv_lens 64-2048
- Correctness: max_abs_err < 5e-3 vs numpy reference
- Wired into `engine.py` as: `get_hip("flash_attn_256_decode", "flash_attn_256_tuned")`

**Elementwise kernel tuning** (`elementwise_v3.hip` — float4/128-bit vectorized):
- `rmsnorm_v3` vs `rmsnorm_v2`: **1.17-1.58x faster** at dim=5120 (~35 µs vs ~50 µs)
- `silu_fused_v3` vs `silu_fused_v2`: **~1.0-1.31x faster** at dim=5120 (~54 µs)
- `residual_add_v3` vs `residual_add_v2`: **1.26-1.32x faster** at dim=5120 (~53 µs)
- v3 uses float4 loads (8 FP16 per thread, grid = (dim+2047)//2048); v2 uses float2 (2 FP16 per thread, grid = (dim+511)//512)
- All correctness: max_abs_err < 5e-3

**DeltaNet v3 occupancy assessment** (`deltanet_v3.hip`):
- Config: 48 WGs (one per v-head), 256 threads/WG
- 48 WGs on 60 CUs → ~0.8 WGs/CU; already near-optimal
- No occupancy improvements possible — kernel is limited by sequential state-update recurrence
- Performance: benchmark confirms config is already optimal

**build_dispatch_cache() dependency on tuned kernels:**
- `engine.py build_dispatch_cache()` uses `get_hip("rmsnorm_v3", "elementwise_v3")` and `get_hip("flash_attn_256_decode", "flash_attn_256_tuned")` WITHOUT try/except fallbacks
- This means cached dispatch mode REQUIRES `elementwise_v3.so` and `flash_attn_256_tuned.so` to be compiled
- The runtime dispatch methods (`_launch_rmsnorm`, `_launch_decode_attn_256`, etc.) DO have try/except fallbacks
- Workers extending cached dispatch MUST ensure tuned kernels are compiled, or add fallbacks to build_dispatch_cache()

**Final Sprint 2 Benchmark Results** (all optimizations: C dispatch + star allreduce + tuned kernels):
- TP=4 throughput: **38.0 tok/s** (vs 25.5 tok/s combined baseline → **1.49x improvement**)
- vs single-GPU: **1.87x** (38.0 vs 20.3 tok/s)
- vs vLLM TP=4 (AWQ): **81%** (38.0 / 46.9 tok/s, gap = 8.9 tok/s)
- Correctness: cosine_sim = 0.999988 vs single-GPU (PASS >= 0.99)
- Single-GPU regression: 21.5 tok/s (PASS, within ±10% of 20.3 baseline)
- Fallback path: C dispatch disabled → correctly falls back to cached+stream

**All phase comparison:**
| Phase | Tok/s |
|---|---|
| Single-GPU baseline | 20.3 |
| TP=4 serial (P2P allreduce) | 12.4 |
| TP=4 cached dispatch | 23.7 |
| TP=4 combined (cached+stream) | 25.5 |
| TP=4 C dispatch + tuned kernels | **38.0** |
| vLLM TP=4 (AWQ, reference) | 46.9 |

---

## Sprint 3: TP=4 Decode Pipeline Optimization (2026-03-15)

**Goal:** Close the remaining 8.9 tok/s gap to vLLM (38.0 → target closer to 46.9 tok/s).

### Remaining Bottleneck Analysis (Post-Sprint 2)
1. **Allreduce overhead:** 128 allreduces × ~119 µs ≈ 15 ms/token (star topology, host-orchestrated)
2. **Host launch overhead:** ~960 hipModuleLaunchKernel calls/token from C dispatch
3. **Q/KV stream sync:** 32 hipStreamSynchronize calls/token (2 per full-attention layer × 16 layers)
4. **KV cache D2D copies:** 32 hipMemcpyAsync D2D calls/token (2 per full-attention layer)

### Sprint 3 Approach
**Milestone 1 (allreduce-pipeline):** Remove sync barriers and eliminate redundant copies
- Eliminate Q/KV stream syncs (run sequentially on default stream for decode)
- Direct KV cache writes (fuse into QKNorm/RoPE or redirect GEMV output pointers)
- Deepen allreduce overlap and reduce event overhead in C dispatch

**Milestone 2 (hip-graph-decode):** Near-zero launch overhead via graph capture
- Per-GPU compute graphs captured between allreduce points
- Host-orchestrated allreduce between graph replay segments
- Mutable param updates via hipGraphExecKernelNodeSetParams

### Key Data Dependencies (Correctness Constraints)
- Attention allreduce → d_hidden → FFN RMSNorm: HARD dependency, cannot defer attention AR wait
- FFN allreduce → d_hidden → Next layer's attn RMSNorm: Already deferred to next layer start
- Q/KV GEMV → d_q, d_k → QKNorm/RoPE: Sequential on same stream suffices (no explicit sync needed)
- QKNorm/RoPE → d_k (post-RoPE) → KV cache: Must complete before decode attention reads cache
- KV cache write → Decode attention read: Must complete before attention kernel launches

### Sprint 3 M1 Results (2026-03-15, allreduce-pipeline)

**Implemented optimizations:**
- **Q/KV stream sync elimination:** Q and KV GEMVs now run sequentially on default (null) stream instead of concurrent streams with explicit hipStreamSynchronize. Eliminates 32 host-blocking sync calls/token. Impact: minimal (~2.8% regression vs baseline) because sync overhead (~1ms) is dominated by allreduce (~15ms).
- **Direct KV cache writes:** New `qknorm_rope_cachew.hip` kernel writes post-RoPE K directly to KV cache; V GEMV output pointer redirected to cache position. Eliminates 32 hipMemcpyAsync D2D copies/token. Impact: neutral (~37.8 tok/s vs ~37.9 baseline) — D2D copies too small to be throughput bottleneck under C dispatch.
- **Allreduce overlap deepening:** `c_dispatch_v2.c` reduces hipSetDevice calls from ~2432 to ~2048/token (~384 saved) by batching device switches and reversing loop order. Impact: ~0.5% within measurement noise.

**Combined Sprint 3 M1 result: 38.1 tok/s** (+0.3% vs Sprint 2 38.0 tok/s baseline)

**Critical findings:**
1. **hipStreamWaitEvent is host-non-blocking on gfx906/ROCm 7.1.** The C dispatch already achieves the maximum achievable compute-communication overlap without HIP graphs. The GPU enforces event ordering while the host immediately dispatches next kernels.
2. **C dispatch already hides small async overhead.** With C dispatch's tight pipelining, removing stream syncs and D2D copies does not measurably improve throughput — these operations were already effectively overlapped. The bottleneck is allreduce latency (~15.2 ms/token = 128 × 119µs).
3. **Overhead breakdown per token:** hipSetDevice: ~2432 calls × ~3µs ≈ 7.3ms; Event ops: ~2048 × ~1.5µs ≈ 3.1ms; hipModuleLaunchKernel: ~960 × ~1µs ≈ 1ms; Allreduce: ~15.2ms. Total overhead ~26.6ms/token → ~37.5 tok/s ceiling with C dispatch.
4. **Sprint 3 M2 (HIP graph decode) is the correct next step.** Eliminating ~960 kernel launches/token (currently ~1ms each, ~1ms total) would save ~1ms. Graph replay has ~10-100× lower per-launch overhead, potentially saving ~0.9ms → ~39-40 tok/s estimate.

**OOM pattern for MI50 tests:** When testing TP=4 correctness vs single-GPU reference, use the load-free-reload pattern to avoid OOM on MI50 (16GB VRAM): load single-GPU engine → collect reference outputs → call `engine.cleanup()` + `del engine` → then load TP=4 engine. Loading both simultaneously causes OOM. Reference: `tests/test_qkv_sync_removal.py:collect_single_gpu_reference()`.



