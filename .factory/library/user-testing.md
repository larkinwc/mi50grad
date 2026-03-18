# User Testing Guide for MI50Grad

## Overview

This project optimizes GPU kernels for AMD MI50 (gfx906) running Qwen 3.5 27B INT4 with tensor parallelism across 4 GPUs.

## Testing Surface

### Primary: CLI/Python API inside Docker container

All tests run inside Docker on the dev server (root@192.168.1.198) using ROCm 7.1.0.

**Docker image:** `mixa3607/rocm-gfx906:7.1.0-complete`

**Test execution:**
```bash
ssh root@192.168.1.198
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad \
    -w /opt/mi50grad \
    mixa3607/rocm-gfx906:7.1.0-complete \
    python3 tests/test_<name>.py
```

### Validation Tools

1. **Direct Python execution** - Primary tool for kernel correctness tests
2. **agent-browser** - Not applicable (no web UI)
3. **tuistory** - Not applicable (not a TUI app)

## Resource Classification

### GPU Resources

- Each test uses all 4 GPUs fully for TP=4 tests
- Memory: ~14GB model weights + KV cache per GPU
- **Max concurrent validators: 1** (all 4 GPUs required for TP=4 tests)

### Isolation Requirements

- Only ONE validator can run at a time (requires all 4 GPUs)
- Tests must be run sequentially
- No user account isolation needed (single-user system)

## Setup Commands

### Build kernels
```bash
cd /opt/mi50grad && make all
```

### Build HIP kernel shared libraries
```bash
cd /opt/mi50grad && make hip_kernels
```

### Build C extensions
```bash
cd /opt/mi50grad && make c_extensions
```

## Test Commands by Milestone

### Fused Kernel Milestone

```bash
# Test fused P2P allreduce + RMSNorm numerical correctness
python3 tests/test_fused_allreduce_rmsnorm.py

# Test C dispatch integration
python3 tests/test_fused_kernel_c_dispatch.py
```

### Key Assertions

| Assertion | Description | Test File |
|-----------|-------------|-----------|
| VAL-FUSE-001 | Numerical equivalence vs separate kernels (max_abs_error < 5e-3) | test_fused_allreduce_rmsnorm.py |
| VAL-FUSE-002 | Kernel launch count reduction (128 -> 64 per token) | Implementation verification |
| VAL-FUSE-003 | Per-layer latency improvement >= 10% | Requires benchmark |
| VAL-FUSE-004 | Dimension alignment edge cases | test_fused_allreduce_rmsnorm.py |
| VAL-FUSE-005 | C dispatch path integration | test_fused_kernel_c_dispatch.py |
| VAL-FUSE-006 | Fallback to separate kernels | test_fused_kernel_c_dispatch.py |
| VAL-FUSE-007 | Multi-GPU output consistency | test_fused_allreduce_rmsnorm.py |

## Validation Concurrency

### Max Concurrent Validators: 1

All tests require 4 GPUs simultaneously. Run validators sequentially.

### Machine State Check
```bash
# Check GPU memory
rocminfo | grep "GPU Memory"

# Check ROCm version
rocminfo | grep "Name:" | head -1
```

---

## Flow Validator Guidance: GPU Kernel Testing

### Isolation Rules

1. **GPU Isolation:** Only one test can use all 4 GPUs at a time
2. **Build First:** Run `make all && make hip_kernels` before tests
3. **No Parallelism:** Tests must run one at a time

### Test Boundaries

- Do NOT modify model weights (`/opt/models/` is read-only)
- Do NOT modify kernel source code during testing
- Output evidence to `.factory/validation/<milestone>/user-testing/flows/`

### Evidence Collection

For each assertion tested:
1. Capture console output showing pass/fail
2. Record numerical values (max_abs_error, throughput, etc.)
3. Note any errors or warnings

---

## Known Issues

1. **SSH to dev server:** Use `ssh root@192.168.1.198` with default SSH key auth
2. **Docker device access:** Must include `--device=/dev/kfd --device=/dev/dri --group-add video`
3. **Build dependency:** HIP kernel shared libraries must be built before Python tests can load them
4. **Model mount:** Docker container must mount `/opt/models:/opt/models` for model weight access
5. **numpy in Docker:** Install with `pip3 install numpy --break-system-packages --quiet` in same command as test execution
6. **C dispatch setup order:** Must follow strict order: load weights -> build_dispatch_cache() -> set_c_dispatch(True). Setting C dispatch before loading weights causes dispatch cache to build with 0 layers.
7. **AWQ model format detection:** The model at `/opt/models/Qwen3.5-27B-AWQ` is detected as 'gptq' format by `detect_awq_format()`. AWQ kernel mode (`set_awq_mode(True)`) skips zero-point subtraction and produces NaN/Inf when used with GPTQ weights (which have non-zero zeros). To validate AWQ kernels E2E, need true AWQ model with zeros=0.
