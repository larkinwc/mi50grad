Today MI50/gfx906 can run modern LLM inference reasonably well via ROCm and community forks, but it is clearly under‑optimized compared to newer AMD and NVIDIA parts; a tinygrad‑style, gfx906‑specific stack could still unlock large gains via custom GEMM, attention, quantization, and graph‑fusion kernels tailored directly to the ISA and memory hierarchy.

Below I’ll sketch the current state, then the big levers a purpose‑built stack could pull.

Where MI50 stands today
MI50 (gfx906, Vega 20) is officially a “GCN5 / GFX9” part; compiler backends like LLVM and GCC treat it as one of several gfx9-generic targets, not something with its own deeply tuned, ML‑first kernel ecosystem.

ROCm and MIOpen do have support for gfx906, but AMD’s newer ROCm releases focus optimization effort on CDNA (MI100/200/300), and users report that support for gfx906 is being phased out or requires hacks like copying old Tensile kernel blobs from ROCm 6.4 into newer installs.

Many of the modern “secret sauce” pieces for PyTorch‑era LLMs either don’t officially support gfx906 or are only available via best‑effort forks:

Triton has no planned support for gfx906; a community fork exists solely to make projects like vLLM run at all.

FlashAttention2 on ROCm is supported and documented, but real‑world reports show that gfx906 is not an official target and requires manual hacking to even compile, often resulting in failing tests or corrupt outputs.

Despite this, a vLLM fork targeting gfx906 shows that 4‑bit/8‑bit LLM inference can be quite fast (e.g., ~20 tok/s on Qwen2.5‑72B int4 with 2×MI50), which is impressive given the lack of vendor love.
​

So: the “stack” today is ROCm + rocBLAS/MIOpen + PyTorch/Transformers, with scattered community hacks for vLLM, flash‑attn, and Triton, but essentially no cohesive, gfx906‑first runtime.

Hardware traits of gfx906 that shape optimization
gfx906 is a GCN5 GPU with 64‑lane wavefronts, large vector register files, and fast LDS (shared memory), but no dedicated matrix‑core or dot‑product instructions that later CDNA parts use for AI.

LLVM’s AMDGPU docs explicitly list many matrix/dot ops as not available on gfx906 – including v_fmac_f32, multiple integer dot‑product opcodes (v_dot4_i32_i8, v_dot8_i32_i4, etc.), and mixed‑precision f16→f32 dot ops.

Marketing and community measurements still quote sizeable theoretical INT4/INT8 TOPS for MI50/60, but those must be realized via more “scalar” fusion of FMA/ALU ops rather than true tensor cores, which raises pressure on scheduling, tiling, and memory reuse.

Bottom line: gfx906 is a very wide SIMD engine with strong HBM bandwidth but lacks the newer hardware shortcuts, so performance depends heavily on ISA‑aware kernel design and memory‑traffic minimization rather than relying on magic matrix cores.

Gaps in today’s software stack
In broad strokes, here’s where current ROCm‑based stacks typically leave performance on the table for MI50:

Generic, not gfx906‑specific kernels

Many libraries target gfx9-generic, meaning tile sizes, unroll factors, vector widths, and LDS usage are tuned as compromises across gfx900/902/904/906/909/90c, not specifically for MI50’s exact CU count, clock, and memory behavior.
​

ROCm 6+ tuning work is focused on MI200/MI300, so older Tensile/MIOpen kernels for gfx906 are basically “frozen in time.”

Limited fused / modern attention kernels

Official ROCm docs highlight FlashAttention2 as a key optimization for LLMs, but the examples and testing focus on recent GPUs; gfx906 is not a first‑class target, and community attempts show instability or test failures when forcing FA2 to run there.

Weak Triton / custom‑kernel ecosystem

Triton explicitly does not plan to support gfx906; the community fork exists but is fragile, which blocks a lot of modern PyTorch‑style fusion/tiling work from being easily ported.

Quantization support is ad‑hoc

Some projects (e.g., vllm‑gfx906) support GPTQ/AWQ int4 on MI50/60 and achieve strong throughput, but there is no general, reusable library of INT8/INT4 GEMM/conv kernels for gfx906 the way CUDA has CUTLASS and friends.
​

So in practice, you typically run MI50 with “good enough” rocBLAS/MIOpen plus a few hacked‑in attention/quant kernels, rather than a ground‑up, ISA‑tuned path.

Big levers for a tinygrad‑style gfx906 stack
If you built a for‑purpose training/inference stack like tinygrad, but only for gfx906, you’d have several high‑impact levers:

1. Hand‑tuned GEMM / matmul microkernels
Design GEMM kernels specifically for:

64‑lane wavefronts,

MI50’s register file size and LDS capacity,

and the absence of dot/matrix instructions (so everything is built from v_fma_f32 / v_mul_f32 etc.).

Use ISA‑level microkernels with:

wave‑synchronous tiling (e.g., 64×K tiles mapped 1:1 onto a wavefront),

explicit LDS double‑buffering of A/B tiles,

aggressive unrolling in K to maximize FMA issue density.

Build a tinygrad‑like autotuner around those microkernels: search over tile shapes (M×N×K), vector widths, LDS usage, and wave/per‑block layouts specifically for MI50’s CU count and HBM2 behavior, instead of relying on generic Tensile heuristics.

Result: a “gfx906‑only CUTLASS” can close a large portion of the gap vs. NVIDIA on raw matmul without matrix cores.

2. Gfx906‑specific FlashAttention/FlashAttention2
Implement FlashAttention/FA2 concepts with gfx906 constraints:

compute QKᵀ and softmax in tiles that fit in LDS, streaming through sequence length to avoid O(L²) HBM traffic, exactly as AMD’s ROCm docs describe conceptually for newer GPUs.

fuse QKᵀ, softmax, scaling, masking, and V·softmax into a single kernel to minimize HBM reads/writes.

Tune tile sizes and parallelization so that:

each block fits fully into LDS and registers without spilling,

wavefronts are fully occupied (e.g., 64‑element wide stripes along sequence length or heads).

Because gfx906 lacks dot/matrix ops, the kernel must carefully balance FMA throughput vs LDS bandwidth, but you can still get big gains just by eliminating redundant HBM traffic, as the AMD FlashAttention blog shows is the main source of speedup.

Realistically, a clean FA/FA2 implementation tuned for gfx906 could be one of the highest ROI items for LLMs and Transformers.

3. Systematic quantized kernels (INT8/INT4)
Despite lacking the dot opcodes listed for newer architectures, MI50/60 are known to have solid theoretical INT4/INT8 capabilities and community projects report strong 4‑bit inference throughput.

A gfx906‑centric stack could:

define a canonical packing layout for int4/int8 in registers and LDS (e.g., pack 8×int4 per 32‑bit lane, or 4×int8),

provide hand‑rolled microkernels that do bit‑unpacking and multiply‑accumulate in F32/F16 registers, with careful scheduling to hide that unpack cost behind FMA latency.

Add an autotuner to trade:

heavier packing/unpacking vs. bigger tiles vs. more reuse, depending on model shape and batch size.

Paired with a simple graph‑level quantization pipeline (GPTQ/AWQ‑like), this would give MI50 a real, reusable quantized kernel library instead of one‑off engine‑specific implementations.
​

4. Graph‑level fusion tailored to GCN
Implement a small graph compiler (or just a pass pipeline) that fuses:

Linear + bias + residual add,

RMSNorm/LayerNorm + linear,

SiLU/GELU + linear,
into single kernels whose launch shapes are chosen for gfx906 occupancy and LDS usage.

Because LLVM’s AMDGPU backend exposes detailed target properties for gfx9-generic vs gfx906 (missing instructions, LDS restrictions, xnack behavior), fusion passes can be written to avoid patterns that cause spilling or pathologically low occupancy on MI50 specifically.

Compared to generic PyTorch fusion, this would be much more aggressive but also more architecture‑aware.

5. Persistent and streaming decoder kernels
For decoder‑only LLM inference, a gfx906‑only stack could implement persistent kernels that:

keep key/value blocks resident in LDS or registers for as long as possible,

step through new tokens without relaunching every layer for every token.

The AMD FlashAttention/ROCm inference‑optimization docs emphasize reducing kernel launches and memory traffic for attention; extending that idea across the whole transformer block yields large latency and throughput wins.

This is especially valuable for multi‑MI50 rigs where PCIe / InfinityFabric transfers of KV cache are expensive.

6. MI50‑specific autotuning and offline profiling
ROCm and MIOpen already support tunable ops and offline tuning, but the experience is generic and often painful; users report that naive online tuning can even cause OOM or corruption issues if used blindly.
​

A tinygrad‑like stack could:

bake in an automated, MI50‑only autotuning phase that sweeps kernels once per model (or per batch/seq regime),

save the tuned configs in a small DB keyed by shapes and dtypes,

avoid runtime online tuning entirely in production.

Because the search space is much smaller when you only support gfx906, you can be more exhaustive without exploding complexity.

7. Cleaner multi‑GPU MI50 semantics
Many current MI50 users run 2–4 cards; a gfx906‑first stack could provide:

explicit tensor‑parallel and pipeline‑parallel layouts chosen to match MI50’s PCIe/IF interconnect and 32 GB HBM size (for the common 32 GB variant),
​

overlap of communication and compute using HIP streams and events, tuned specifically to the latency/bandwidth curves seen on MI50 clusters.

Nothing stops ROCm/PyTorch from doing this today, but a single‑arch stack has the freedom to hard‑code assumptions and simplify the runtime.

Training‑specific opportunities
For training (not just inference), the same gfx906‑aware stack could add:

Mixed‑precision training tuned to the absence of fancy dot ops

Use FP16/BF16 where possible but accept that dot‑product ops aren’t available; you’d schedule FP16 FMAs with accumulation into F32 and rely on your custom GEMM kernels’ tiling to extract throughput.

Gradient checkpointing + FlashAttention

Combine blockwise attention (FlashAttention‑style) with recomputation to keep activation memory within MI50’s HBM while maximizing GPU utilization.

Optimizer fusion

Implement fused Adam/Lion kernels that update weights and optimizer state in a single pass over memory, again tuned to gfx906’s memory and instruction mix.

How far could this reasonably go?
Evidence from community work suggests that even with the current patchwork stack, MI50 can deliver attractive price/perf for LLM inference when carefully configured:

Benchmarks with llama.cpp and specialized builds for gfx906 show substantial token/s for mid‑size and even very large models.
​
​

A hacked‑together FA2 + vLLM + Triton‑fork environment on MI60 already hints at up to an order‑of‑magnitude forward‑pass speedups for attention when things actually work, even though tests fail and outputs can be unstable.
​

That strongly implies that a cohesive, gfx906‑only, tinygrad‑style stack — with custom GEMM, proper FlashAttention, INT4/INT8 kernels, and graph fusion – could move MI50 from “works but awkward” into “legit workhorse” territory for inference, and make small‑scale training viable, even without matrix cores.

If you want, we can next sketch a concrete design (component list, minimal compiler, kernel library layout, and autotuner strategy) for a gfx906‑only stack you could realistically build and iterate on in a homelab.

