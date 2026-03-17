# mi50grad

High-throughput inference optimization for **Qwen 3.5 27B** (GPTQ-Int4) on **AMD Instinct MI50** (gfx906) GPUs with tensor parallelism across 4 devices.

## Performance

| Mode | Throughput | Speedup |
|---|---|---|
| Star topology baseline (TP=4) | 15.3 tok/s | 1.00x |
| C dispatch + kernel P2P allreduce | **38.3 tok/s** | **2.50x** |
| Global graph capture | 36.5 tok/s | 2.39x |
| AWQ kernel mode | 44.7 tok/s | 2.92x |
| Single-GPU | 22.0 tok/s | -- |

Full benchmark report: [`bench/tp4_sprint4_report.md`](bench/tp4_sprint4_report.md)

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

### Kernel P2P Allreduce (2.50x speedup)
Replaces host-orchestrated star topology allreduce with on-device kernel that reads all 4 partial buffers directly via BAR1-mapped P2P. Eliminates `hipSetDevice`, `hipMemcpyPeerAsync`, and `hipStreamSynchronize` host round-trips. Latency: ~79us vs ~119us star topology per call (128 calls/token).

### C Dispatch
Moves the per-layer kernel dispatch loop from Python into a compiled C shared library. Eliminates Python interpreter overhead (~1ms/token) from the critical decode path.

### Global Graph Capture
Captures full-layer HIP graphs (4 GPUs x 64 layers x 2 segments) with a C-level replay plan. Provides infrastructure for future optimizations where graph overhead can be amortized.

### GEMV v5 (DPP Reduction)
INT4 dequantization GEMV with hybrid DPP (Data Parallel Primitives) intra-wavefront reduction + minimal LDS cross-wavefront reduction. The kernel is memory-bandwidth bound (~130-160 GB/s of 857 GB/s peak), so the reduction optimization has minimal E2E impact.

### AWQ Support
Zero-point-free GEMV kernel variant for AWQ quantization format. 1.17-1.27x faster than GPTQ GEMV in isolation (skips zero-point subtraction arithmetic).

## Architecture Notes

- **No MFMA**: MI50 (gfx906) lacks matrix fused multiply-add instructions (available on gfx908+). All matrix operations use scalar VALU instructions.
- **P2P via PCIe BAR1**: No XGMI interconnect. Inter-GPU communication at ~12 GB/s per link.
- **Allreduce is the bottleneck**: 128 allreduce calls/token x ~79us = ~10.1ms of ~26ms total decode time (39%).
- **Dispatch modes**: Progressive fallback chain: global_graph -> c_dispatch -> cached+stream. All modes maintain cosine similarity >= 0.99.

## Model

- **Qwen 3.5 27B GPTQ-Int4**: 64 transformer layers, hidden_size=5120, intermediate_size=17408
- **Model path**: `/opt/models/Qwen3.5-27B-GPTQ-Int4` (on GPU server)
- **Quantization**: 4-bit weight quantization, group_size=128
