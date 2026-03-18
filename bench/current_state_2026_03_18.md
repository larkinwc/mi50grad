# Current State Benchmark — 2026-03-18

## Hardware
- 4x AMD Instinct MI50 32GB (gfx906, Vega 20)
- PCIe BAR1 P2P (~12 GB/s per link, no XGMI)
- ROCm 7.1.0 via Docker (`mi50grad:latest`)

## Model
- Qwen3.5-27B-GPTQ-Int4 (64 layers, hidden=5120, intermediate=17408)
- INT4 weight-only quantization, group_size=128
- Tensor parallelism: TP=4 (all 4 GPUs)

## Benchmark Config
- Batch size: 1
- Max sequence length: 256
- Benchmark steps: 100
- Warmup steps: 5

## Results

| Mode | tok/s | ms/tok | Notes |
|------|------:|-------:|-------|
| **TP=4 EAGLE speculative** | **45.19** | **22.13** | Best overall; EAGLE K=5 |
| TP=4 N-gram speculative | 45.14 | 22.15 | N-gram n=3, max_draft=5 |
| TP=4 Star topology (C dispatch) | 44.80 | 22.32 | Kernel P2P disabled |
| TP=4 C dispatch + kernel P2P | 39.71 | 25.18 | Kernel P2P SLOWER than star?! |
| Single-GPU | 21.97 | 45.51 | Baseline, no TP overhead |

## Active Features
- GEMV v6 (register-cached scale/zero + prefetch): active for N<=4096
- GEMV v5 (hybrid DPP+LDS t16): fallback for N>4096
- GEMV dual (fused gate+up+silu): active
- Flash attention 256 (tuned): active
- Fused QK-norm+RoPE+cache-write: active
- DeltaNet v3 GPU kernel: active (12 virtual heads per GPU in TP=4)
- C dispatch: active
- Kernel P2P allreduce: loaded but SLOWER than star topology
- Fused allreduce+RMSNorm kernel: active
- N-gram speculative decode: integrated into TPInferenceEngine
- EAGLE speculative decode: integrated into TPInferenceEngine
- GEMM INT4 prefill kernel: loaded
- GEMM FP16 prefill kernel: loaded

## Observations

### 1. Kernel P2P allreduce regression
The kernel P2P path (39.71 tok/s) is **slower** than star topology (44.80 tok/s) by 11%.
This is a regression from the Sprint 5 report which showed kernel P2P at 44.42 tok/s.
Possible causes:
- BAR1 P2P read latency increased due to different GPU topology or driver state
- The star topology now benefits from fused allreduce+RMSNorm kernel
- Kernel P2P may have a bug in the current integration

### 2. Speculative decode shows marginal gain
N-gram and EAGLE both show ~45.1-45.2 tok/s vs 44.8 tok/s star topology (0.7% gain).
This suggests speculative decode overhead nearly cancels out the allreduce amortization
benefit, or the draft acceptance rate is very low with random embeddings (not real text).
Real-world text with repetitive patterns should show higher acceptance rates.

### 3. Star topology is currently the fastest standard decode
44.80 tok/s with C dispatch + star topology + fused allreduce+RMSNorm is the current best
non-speculative mode.

### 4. TP scaling
- Single-GPU: 21.97 tok/s (45.51 ms/tok)
- TP=4 best: 45.19 tok/s (22.13 ms/tok)
- Scaling efficiency: 45.19 / (21.97 * 4) = 51.4%
- ~48.6% lost to allreduce overhead (128 calls/token x 64 layers)

## Comparison to Previous Reports
| Date | Best TP=4 | Notes |
|------|----------:|-------|
| Sprint 4 | 38.3 tok/s | C dispatch + kernel P2P baseline |
| Sprint 5 (2026-03-17) | 44.42 tok/s | GEMV v6 + kernel P2P |
| **Current (2026-03-18)** | **45.19 tok/s** | EAGLE speculative (star topo) |
