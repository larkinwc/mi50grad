# Environment

Environment variables, external dependencies, and setup notes for MI50Grad.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.

---

## Hardware

- **Dev server:** root@192.168.1.198 (SSH key auth)
- **GPUs:** 4× AMD Instinct MI50 32GB HBM2 (gfx906, Vega 20)
- **ROCm:** 7.1.0 via Docker image `mixa3607/rocm-gfx906:7.1.0-complete`
- **P2P:** PCIe BAR1, ~12 GB/s inter-GPU bandwidth
- **No XGMI, No MFMA** (gfx906 limitations)

## Docker Commands

```bash
# Build image
docker build -t mi50grad .

# Run inference
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/opt/mi50grad \
    -v /opt/models:/opt/models \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_tp4_sprint5_final.py'
```

## Models

- `/opt/models/Qwen3.5-27B-GPTQ-Int4` - Primary model (GPTQ INT4)
- `/opt/models/Qwen3.5-27B-AWQ` - AWQ model for AWQ validation

## Build

- `make all` - Build kernels, C extensions
- `make kernels` - Build HIP kernels only
- `make c_extensions` - Build C dispatch extensions
- `make clean` - Clean build artifacts

## Key Environment Variables

- `HIP_VISIBLE_DEVICES` - GPU selection (0,1,2,3 for TP=4). **Required** when running tests inside Docker - without this, only 3 GPUs may be detected.
- `ROCM_PATH` - ROCm installation path (default: /opt/rocm)

## Validation Constraints

- **Model Loading Time:** ~10+ minutes for Qwen3.5-27B-GPTQ-Int4 on MI50 hardware. Validation tests that require fresh model loading may timeout. Historical evidence from existing benchmarks is acceptable for validation.
- **Single-GPU Memory:** 27B Int4 model requires ~14-15GB for weights. With Docker overhead, single-GPU benchmarks may OOM on MI50 (32GB total). Use subprocess isolation pattern from `bench_tp4_sprint4.py` for single-GPU tests on large models.

## HIP Kernel Shared Libraries

Kernel shared libraries (.so) are built to `build/kernels/` for ctypes-based dispatch:

```bash
make hip_kernels  # Build all HIP kernel shared libraries
```

Key libraries:
- `kernel_p2p_allreduce.so` - P2P allreduce kernel
- `kernel_p2p_allreduce_rmsnorm.so` - Fused P2P allreduce + RMSNorm kernel

## Fused Kernel Pattern

The fused P2P allreduce + RMSNorm kernel follows the same vectorized float4 load pattern as `rmsnorm_v3`:
- 8-wide float4 loads with half2 packing
- Thread layout: t*8, t*8+2048, t*8+4096, ...
- FP32 accumulation for numerical correctness on gfx906

This pattern can be reused for other fused kernel optimizations.
