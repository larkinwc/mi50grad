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
- **Assembly (.s)**: Compiled via `llvm-mc → ld.lld → .hsaco`. Max control. Used for: RoPE (only ASM), fallback for GEMM/GEMV.
- **HIP C++ (.hip)**: Compiled via `hipcc → .so`. Easier iteration. Used for: elementwise ops, GEMV FP16/INT4, GEMM prefill, FlashAttention, DeltaNet, batched RMSNorm, sigmoid_mul.

**The engine prefers HIP kernels over assembly.** Assembly kernels are fallback.

## Key HIP Kernels (Primary Path)
- `elementwise_v2.hip`: residual_add_v2, silu_fused_v2, rmsnorm_v2, skip_rmsnorm_v2
- `gemv_fp16_v2.hip`: Uses `__builtin_amdgcn_fdot2` (v_dot2_f32_f16). 4 rows/WG, DPP reduction.
- `gemv_int4_v2.hip`: Split-K with atomicAdd, factored zero subtraction. 
- `gemv_int4_dual.hip`: Fused gate+up with SiLU. Saves 1 launch + 1 memset + 1 read of x[K].
- `flash_attn_256.hip`: head_dim=256, online softmax, GQA-aware.
- `deltanet_v3.hip`: Full DeltaNet recurrence in one kernel, parallel kq_dot.
- `batched_rmsnorm.hip`: Per-head Q/K normalization.

## gfx906 Confirmed Instructions
All verified on real hardware:
- `v_dot2_f32_f16`: ~7.58 instr/cycle (packed FP16 dot product)
- `v_dot4_i32_i8`: ~7.58 instr/cycle (INT8 4-element dot)
- `v_dot8_i32_i4`: ~7.58 instr/cycle (INT4 8-element dot)
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

## INT4 GEMV Pattern (current v2)
memset(fp32_buf) → gemv_int4_v2_splitk(fp32_buf) → fp32_to_fp16(output)
Three launches per GEMV. Target: fuse into one.

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

## Qwen 3.5 27B Architecture
- Hybrid attention: some layers use full GQA (head_dim=256), others use DeltaNet linear attention
- hidden_size=5120, num_heads=48, num_kv_heads=8
- FFN: gate+up (fused) → SiLU → down, with INT4 quantized weights (GPTQ)
