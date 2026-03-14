# Architecture

Architectural decisions, patterns discovered, and kernel design notes.

**What belongs here:** Kernel design patterns, LDS layout strategies, register budget decisions, instruction selection rationale, fusion patterns.

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
