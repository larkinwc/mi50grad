# mi50grad

High-throughput inference optimization for **Qwen 3.5 27B** (GPTQ-Int4) on **AMD Instinct MI50** (gfx906) GPUs with tensor parallelism across 4 devices.

## Performance

### Optimization Modes (TP=4, decode)

| Mode | Throughput | Speedup |
|---|---|---|
| Star topology baseline (TP=4) | 15.3 tok/s | 1.00x |
| C dispatch + kernel P2P + deferred AR | 51.75 tok/s | 3.38x |
| **Fused GEMV+AR+RMSNorm + deferred AR** | **53.74 tok/s** | **3.51x** |
| Single-GPU | 22.0 tok/s | -- |

**Gap closure:** 81.5% toward 60 tok/s target. Kernel launches reduced from 192 to 64 per token.

### Prompt Processing & Text Generation (Qwen 3.5 27B GPTQ-Int4)

Benchmarked on 4x MI50 32GB (gfx906), ROCm 7.1.0. All optimizations enabled (C dispatch, kernel P2P allreduce, deferred attention AR, fused GEMV+AR+RMSNorm).

| Config | Test | Throughput (tok/s) |
|---|---|---|
| TP=1 (1x MI50) | pp256 | 28.27 |
| TP=1 (1x MI50) | tg128 | 20.39 |
| **TP=2 (2x MI50)** | **pp512** | **38.88** |
| TP=2 (2x MI50) | tg128 | 32.34 |
| TP=4 (4x MI50) | pp512 | 32.12 |
| **TP=4 (4x MI50)** | **tg128** | **56.33** |

Notes:
- **pp** = prompt processing (prefill), **tg** = text generation (autoregressive decode)
- TP=1 limited to pp256 (27B model uses ~25GB of 32GB VRAM, insufficient for pp512 KV cache)
- TP prefill uses batched GEMM + FlashAttention with FusedP2PReduce allreduce (no buffer size limitation)
- TP=2 achieves best pp512 throughput (38.88 tok/s) -- TP=4 pp is bottlenecked by PCIe P2P allreduce of 5MB payloads per layer (128 allreduces x 512*5120 FP16 elements)
- TP=4 tg128 achieves 56.33 tok/s with all optimizations (C dispatch + kernel P2P + deferred AR + fused kernels)

## Research & Optimization History

See [`RESEARCH.md`](RESEARCH.md) for full optimization history, failed attempts, and bottleneck analysis.

**Current Performance:** 53.74 tok/s on 4× MI50 (3.51× over baseline).

Key optimizations:
- **Fused GEMV+AR+RMSNorm:** Cross-WG atomic barrier, 66% fewer kernel launches (+3.8%)
- **Deferred Attention Allreduce (M3):** Halves allreduce count 128→64 (+35%)
- **Kernel P2P Allreduce (M1):** BAR1 direct reads, 119→79µs per call (1.50×)
- **Kernel micro-opts:** GEMV v6, FlashAttention-256 v3, INT4 GEMM v2 (2.07×)
- **Speculative decode:** N-gram 54% acceptance, EAGLE infrastructure validated

## Hardware Requirements

- 4x AMD Instinct MI50 32GB (gfx906, Vega 20)
- ROCm 7.1.0 (via patched Docker image `mixa3607/rocm-gfx906:7.1.0-complete`)
- P2P access between all GPUs (PCIe BAR1)

## Project Structure

```
src/
  inference/       # Engine and TP engine (tp_engine.py: ~200KB, full TP=4 decode pipeline)
    engine.py      # Single-GPU inference engine
    tp_engine.py   # Tensor-parallel engine (C dispatch, graph capture, kernel P2P)
    generate.py    # Text generation loop
    sampler.py     # Token sampling
  kernels/         # HIP C++ GPU kernels
    gemv_int4_v5.hip         # GEMV with hybrid DPP + minimal LDS reduction
    gemv_int4_v5_awq.hip     # AWQ variant (no zero-point subtraction)
    kernel_p2p_allreduce.hip # Kernel-based P2P allreduce via BAR1 mapping
    flash_attn_256_v3.hip    # Flash attention (max seq_len 256)
    deltanet_v3.hip          # DeltaNet recurrent layer
    ...                      # ~30 kernel variants
  runtime/         # Host-side dispatch infrastructure
    c_dispatch.c             # C dispatch loop (tight kernel launch from C)
    c_graph_dispatch.c       # HIP graph capture + C replay
    fast_allreduce.c         # Star topology allreduce (baseline)
    p2p_allreduce.py         # Python P2P allreduce orchestration
    hip_dispatch.py          # Python HIP dispatch
    tensor_parallel.py       # TP weight sharding
  model/           # Model loading
    weight_loader.py         # GPTQ-Int4 weight loader
    awq_loader.py            # AWQ format weight loader
    qwen.py                  # Qwen model config
  asm/             # GCN assembly kernels and ISA probes
  graph/           # Graph capture utilities
  tune/            # Kernel autotuning
tests/             # Tests, benchmarks, and probes
bench/             # Benchmark reports
scripts/           # Deployment and run scripts
```

## Setup

### Docker Build

```bash
# On the GPU server
docker build -t mi50grad .
```

### Deploy from Development Machine

```bash
# Syncs project, builds Docker image, and compiles all targets
./scripts/deploy.sh

# Build specific target
./scripts/deploy.sh probes    # ISA probes only
./scripts/deploy.sh kernels   # Assembly kernels only
```

### Build Inside Container

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /opt/mi50grad:/opt/mi50grad \
  mi50grad make all
```

Build targets: `probes`, `bench`, `kernels`, `c_extensions`, `all`, `clean`

## Running Inference

### Single-GPU

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 \
  -v /opt/mi50grad:/opt/mi50grad \
  -v /opt/models:/opt/models \
  mi50grad bash -c 'cd /opt/mi50grad && python3 -c "
from src.inference.generate import generate
generate(\"/opt/models/Qwen3.5-27B-GPTQ-Int4\", prompt=\"Hello\", max_tokens=50)
"'
```

### TP=4 (Tensor Parallel across 4 GPUs)

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /opt/mi50grad:/opt/mi50grad \
  -v /opt/models:/opt/models \
  mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4_sprint4.py'
```

## Running Tests

Tests require GPU access and run inside the Docker container:

```bash
# Sprint 4 combined benchmark (all optimizations)
python3 tests/bench_tp4_sprint4.py

# Individual component tests
python3 tests/test_kernel_p2p_allreduce.py  # Kernel P2P allreduce
python3 tests/test_gemv_int4_v5.py          # GEMV v5 (DPP reduction)
python3 tests/test_awq_gemv.py              # AWQ kernel
python3 tests/test_awq_loader.py            # AWQ weight loader (no GPU needed)
python3 tests/test_global_graph.py          # Global graph capture
python3 tests/test_c_dispatch.py            # C dispatch infrastructure
```

## Key Optimizations

### Fused GEMV+Allreduce+RMSNorm (+3.8%)
Fuses INT4 GEMV, P2P allreduce, and RMSNorm into a single kernel launch per FFN down-projection. Uses cross-WG atomic barrier with two counters (write + done) and `__threadfence()` for global RMSNorm sum-of-squares across all N columns via P2P reads. Reduces kernel launches from 192 to 64 per token.

### Deferred Attention Allreduce (+35%)
Defers attention output allreduce, operating FFN on partial activations. Halves allreduce count from 128 to 64 per token (dominant optimization).

### Kernel P2P Allreduce (1.50x allreduce speedup)
On-device kernel reads all 4 partial buffers directly via BAR1-mapped P2P. Eliminates host round-trips. Latency: ~79us vs ~119us star topology per call.

### C Dispatch
Moves per-layer kernel dispatch from Python into compiled C. Eliminates Python interpreter overhead from decode path.

### Speculative Decoding
N-gram (n=3) and EAGLE draft-token generation. N-gram acceptance: 54% overall (59% code, 87% repetitive, 39% JSON, 33% conversational). No throughput regression. Gains limited by allreduce bottleneck.

## Architecture Notes

- **No MFMA**: MI50 (gfx906) lacks matrix fused multiply-add instructions (available on gfx908+). All matrix operations use scalar VALU instructions.
- **P2P via PCIe BAR1**: No XGMI interconnect. Inter-GPU communication at ~12 GB/s per link.
- **Allreduce is the bottleneck**: 64 allreduce calls/token (with deferred AR) x ~79us = ~5.1ms of ~18.6ms total decode time (27%).
- **GPU compute is the ceiling**: ~11ms/token (59%) fixed by MI50 hardware — no MFMA means scalar VALU only.
- **Dispatch modes**: Progressive fallback chain: global_graph -> c_dispatch -> cached+stream. All modes maintain cosine similarity >= 0.99.

## Model

- **Qwen 3.5 27B GPTQ-Int4**: 64 transformer layers, hidden_size=5120, intermediate_size=17408
- **Model path**: `/opt/models/Qwen3.5-27B-GPTQ-Int4` (on GPU server)
- **Quantization**: 4-bit weight quantization, group_size=128
