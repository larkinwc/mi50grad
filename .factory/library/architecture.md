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

## INT4 GEMV Pattern (current v2)
memset(fp32_buf) → gemv_int4_v2_splitk(fp32_buf) → fp32_to_fp16(output)
Three launches per GEMV. Target: fuse into one.

## Qwen 3.5 27B Architecture
- Hybrid attention: some layers use full GQA (head_dim=256), others use DeltaNet linear attention
- hidden_size=5120, num_heads=48, num_kv_heads=8
- FFN: gate+up (fused) → SiLU → down, with INT4 quantized weights (GPTQ)
