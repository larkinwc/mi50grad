# mi50grad: Purpose-Built Inference Stack for gfx906

## Context

AMD MI50 (gfx906) has been dropped from ROCm support (removed in 7.0.1), leaving modern frameworks with missing Tensile kernel DBs, no SDPA kernels, and broken attention paths. Yet the hardware — 60 CUs, 1.0 TB/s HBM2, packed FP16/INT8/INT4 dot instructions — remains viable for LLM inference. Community forks (vllm-gfx906, llama.cpp-gfx906) prove this works but are fragile patches on top of general-purpose stacks.

mi50grad is a tinygrad-style, vertically-integrated inference stack built exclusively for gfx906 to maximize token throughput for Qwen 3.5 27B (INT4, ~14GB, across 2-3 MI50s on our dev server at root@192.168.1.198).

## Unresolved: ISA Dot Instruction Availability

The research reports contradict each other. chat.md and gemini.md say gfx906 has V_DOT2_F32_F16, V_DOT4_I32_I8, V_DOT8_I32_I4. perplexity.md says these are NOT available per LLVM docs. LLVM tablegen source (AMDGPU.td) confirms gfx906 has FeatureDot1Insts/Dot2Insts/Dot7Insts/Dot10Insts — but this **must be empirically verified on real hardware** before committing to kernel designs. This is Phase 1, Task 1.

## What to Study vs Build Fresh

| Source | Study | Adapt | Build Fresh |
|--------|-------|-------|-------------|
| tinygrad | AM driver, HCQ dispatch, BEAM search autotuner | HCQ dispatch pattern (later) | Our own kernel library |
| nlzy/vllm-gfx906 | FP32 accumulator fix for quant GEMV, doubled thread-blocks | FP32 accumulator pattern | Our own runtime |
| iacopPBK/llama.cpp-gfx906 | "Poor Man's FlashAttention", DPP warp reductions, MoE sub-warp shuffle | DPP reduction patterns, FA tiling strategy | Our own attention kernels |
| AMD ISA docs / LLVM | VOP3P encoding, SDWA, wait counters, LDS banking | Reference only | All assembly kernels |

---

## Phase 1: Foundation (Weeks 1-3)

### 1.1 ISA Verification (Week 1, first 2 days)
Write tight-loop HIP assembly probes for each instruction, compile with `clang++ --offload-arch=gfx906`, run on dev server MI50s:
- `v_dot2_f32_f16` throughput
- `v_dot4_i32_i8` / `v_dot8_i32_i4` throughput
- `v_pk_fma_f16` throughput
- `v_fmac_f32` throughput
- MUFU `v_exp_f32` throughput vs polynomial approximation on VALU
- DPP instructions (`ds_swizzle_b32`, `v_mov_b32` with DPP modifiers)

Each reports: instructions/cycle/CU, and whether it executes or traps.

**If dot instructions are broken**: fallback to `v_pk_fma_f16` and `v_fma_mix_f32` for all kernel designs.

**Status**: Probes written at `tests/isa_probes/probe_*.s` with HIP runner at `tests/isa_probes/run_probes.hip.cpp`.

### 1.2 Minimal Runtime (Weeks 1-2)
- Python orchestration + C/HIP host dispatch + GCN assembly device kernels
- Build: Python-driven, compiles `.s` -> HSACO via `llvm-mc --arch=amdgcn --mcpu=gfx906` + `ld.lld`
- Runtime: HIP (`hipModuleLoad`, `hipModuleLaunchKernel`) — simple, avoids user-space driver complexity initially
- Memory: `hipMalloc`/`hipFree` with explicit device selection

**Status**: Runtime at `src/runtime/hip_dispatch.py`, kernel launcher at `src/kernels/launcher.py`, build tool at `tools/build_kernels.py`.

### 1.3 Microbenchmarks (Weeks 2-3)
- HBM bandwidth (sequential R/W/copy, various vector widths)
- LDS bandwidth + bank conflict sensitivity (with/without XOR swizzle)
- L2 working-set characterization
- Occupancy sweep (same kernel, varying VGPR count -> waves/CU vs throughput)
- Kernel dispatch latency

**Status**: Benchmarks at `bench/micro/*.hip.cpp`.

**Verify on dev server**: All probes run on MI50. Compare against theoretical peaks (26.8 TFLOPS FP16, 1.0 TB/s HBM).

---

## Phase 2: Core Kernels (Weeks 3-8)

### 2.1 FP16 GEMM (Weeks 3-5)
- Inner loop: `v_dot2_f32_f16` with FP32 accumulators
- Tile decomposition: workgroup computes 64x64 or 128x64 of C
- Double-buffered A/B tiles in LDS (~32KB to allow 2 workgroups/CU)
- XOR-swizzled LDS layout: `phys_addr = (row * stride) + (col XOR (row % 32))`
- Global load pipelining: overlap next-tile loads with current-tile compute via VMCNT
- Register budget: 64-80 VGPRs/thread -> 4-5 waves/SIMD
- Start with fixed tile sizes for Qwen 3.5 27B shapes (QKV proj, FFN up/down)

**Target**: 60-70% of 26.8 TFLOPS peak for large GEMMs.

### 2.2 FlashAttention Forward (Weeks 5-7)
Study iacopPBK's "Poor Man's FlashAttention" heavily, then reimplement for our runtime:
- Outer loop over K/V blocks, Q tile resident in LDS
- QK^T via `v_dot2_f32_f16`, online softmax, V accumulation
- **MUFU avoidance**: polynomial `exp()` on VALU if Phase 1 confirms MUFU is the bottleneck
- **DPP warp reductions**: `ds_swizzle_b32` with `shuffle_xor` for softmax max/sum — bypasses LDS entirely
- Block sizes: Br=64, Bc=64, head_dim=128 (Qwen 3.5)
- Two variants: prefill (throughput, large seq) and decode (latency, single token vs full KV)

### 2.3 Elementwise / Normalization (Weeks 6-8)
Bandwidth-bound kernels — maximize GB/s:
- RMSNorm (DPP reduction for variance)
- SiLU (`x * sigmoid(x)`, polynomial sigmoid on VALU)
- RoPE (polynomial sin/cos + rotation)
- Residual add (fuse into other kernels where possible)

**Verify**: GEMM correctness vs numpy reference. Attention vs naive implementation (max abs error < 1e-3). Elementwise: 80%+ of 1.0 TB/s.

---

## Phase 3: Quantization (Weeks 8-12)

### 3.1 INT4 Weight-Only GEMM (Weeks 8-10) — **Highest priority**
Qwen 3.5 27B at INT4 is ~14GB -> fits on 2 MI50s.
- `v_dot8_i32_i4`: 8 INT4 dots/instruction -> INT32 accumulate -> FP32 scale/zero-point
- GPTQ/AWQ group format (128 weights/group, FP16 scale+zp)
- SDWA for sub-byte extraction, interleave dequant with dot instructions
- **FP32 accumulation is non-negotiable** (per vllm-gfx906 overflow fix)
- Two variants: GEMV (decode, bandwidth-bound) and GEMM (prefill, tiled + on-the-fly dequant)

### 3.2 INT8 W8A8 GEMM (Weeks 10-11)
- `v_dot4_i32_i8`: 4 INT8 dots/instruction -> INT32 accumulate
- SmoothQuant-style per-tensor/per-channel scales in epilogue

### 3.3 Model Loaders (Weeks 11-12)
- GPTQ safetensors loader + weight repacking to our packed layout
- AWQ loader (secondary)

**Verify**: Quantized GEMM vs dequantize-then-multiply reference. Single-layer test vs HuggingFace.

---

## Phase 4: System Integration (Weeks 12-18)

### 4.1 Graph IR + Fusion (Weeks 12-14)
Minimal DAG of ops for transformer inference:
- Fusion rules: Linear+bias+activation, Linear+residual, RMSNorm+Linear, QKV concat
- Static memory planning: compute buffer lifetimes, reuse memory, no runtime allocation

**Status**: Stub at `src/graph/ir.py`.

### 4.2 Autotuning (Weeks 14-16)
- Search over tile sizes, workgroup sizes, unroll factors, LDS allocation per GEMM shape
- Exhaustive grid search (feasible for one architecture)
- SQLite DB keyed by (op, M, N, K, dtype). Ship pre-tuned configs for Qwen shapes.

**Status**: Stub at `src/tune/tuner.py`.

### 4.3 Multi-GPU Tensor Parallelism (Weeks 16-18)
- Column-parallel QKV/up, row-parallel output/down, AllReduce after row-parallel
- `hipMemcpyPeerAsync` for inter-GPU transfers (PCIe ~12-15 GB/s)
- 2-GPU primary target (14GB INT4 model + KV cache fits easily)
- Check PCIe topology with `rocm-smi --showtopo` on dev server

**Verify**: Fused vs unfused numerical equivalence. Tuned vs default performance. Multi-GPU vs single-GPU correctness.

---

## Phase 5: Model Integration (Weeks 18-24)

### 5.1 Qwen 3.5 27B Model Definition (Weeks 18-19)
- Python model -> graph IR mapping
- HF safetensors / GPTQ weight loading
- Contiguous KV cache (no PagedAttention initially)

**Status**: Config at `src/model/qwen.py`.

### 5.2 Inference Engine (Weeks 19-21)
- Prefill: GEMM projections + FlashAttention + fused elementwise
- Decode: GEMV + decode attention against growing KV cache
- Sampling: top-k/top-p/temperature on GPU
- Tokenizer: HuggingFace tokenizers (Python)

### 5.3 Optimization + Benchmarking (Weeks 21-24)
- Profile with `rocprof` or custom timing
- Target benchmarks:
  - Prefill tok/s at 128, 512, 2048, 8192 context
  - Decode tok/s (single user)
  - TTFT
  - Compare vs llama.cpp-gfx906 and vllm-gfx906 (~20 tok/s for 72B INT4 on 2xMI50, so 27B should be faster)
- If dispatch latency matters: consider tinygrad-style HCQ/AM driver approach

### 5.4 Stability
- 10,000+ token generation without crash or numerical divergence
- Perplexity comparison vs HF reference on WikiText-2

---

## Project Structure
```
mi50grad/
  src/
    runtime/          # HIP dispatch, device abstraction, multi-GPU
    asm/              # GCN assembly kernels (.s) + shared macros
    kernels/          # Python kernel launchers
    graph/            # IR, fusion passes, memory planning
    tune/             # Autotuner + SQLite DB
    model/            # Qwen definition, weight loaders, KV cache, TP
    inference/        # Engine, sampler
  tests/              # ISA probes, kernel correctness, model tests
  bench/              # Microbenchmarks, GEMM benchmarks, E2E benchmarks
  tools/              # build_kernels.py, profiling utilities
  scripts/            # deploy.sh, run_probes.sh, run_bench.sh
  plans/              # Research reports (existing)
```

## Critical Path
1. **Week 1**: ISA verification — unblocks everything
2. **Weeks 3-5**: FP16 GEMM — the most important kernel
3. **Weeks 5-7**: FlashAttention — without it, attention is the bottleneck
4. **Weeks 8-10**: INT4 GEMM — without it, model doesn't fit on 2 MI50s
5. **Weeks 16-18**: Multi-GPU — without it, can't run Qwen 3.5 27B
6. **Weeks 18-24**: End-to-end integration

Each phase produces testable artifacts on the dev server.

## Dev Environment
- Dev server: root@192.168.1.198
- 3x MI50 (gfx906, GPU 0-2) + 1x MI100 (gfx908, GPU 3)
- ROCm container: `mixa3607/rocm-gfx906:7.1.0-complete`
- LLVM tools: `/opt/rocm/llvm/bin/{clang++, llvm-mc, ld.lld}`
- Docker for builds: `docker build -t mi50grad .`
- Deploy: `./scripts/deploy.sh`
