# User Testing

Testing surface: tools, URLs, setup steps, isolation notes, known quirks.

**What belongs here:** How to test kernels, how to run benchmarks, what test scripts exist, setup for validation.

---

## Testing Surface
This is a GPU kernel project. All testing happens via SSH to the LXC, running Python test scripts that compile+load+launch kernels on the MI60 GPU.

**No browser, no TUI, no HTTP endpoints.** Testing is terminal-only.

## How to Test

### 1. Deploy Code
```bash
rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' --exclude='.factory' \
    -e 'ssh -J root@wittymantis.netbird.selfhosted' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.189:/root/mi50grad/
```

### 2. Build Kernels
```bash
ssh -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && export ROCM_PATH=/opt/rocm && make kernels'
```

For HIP kernels:
```bash
ssh -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o build/kernels/KERNEL.so src/kernels/KERNEL.hip'
```

### 3. Run Tests
```bash
ssh -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && PYTHONPATH=/root/mi50grad python3 tests/TEST_FILE.py'
```

## Existing Test Files
- `test_gemm_fp16.py` — FP16 GEMM correctness + perf
- `test_gemv_int4.py` — INT4 GEMV correctness
- `test_flash_attn.py`, `test_flash_attn_256.py` — FlashAttention
- `test_elementwise.py` — RMSNorm, SiLU, residual add
- `test_engine_integration.py` — Engine decode step
- `test_e2e_generate.py`, `test_e2e_tokenized.py` — End-to-end generation
- Many more in tests/

## Test Pattern
Tests typically:
1. Create random input tensors (numpy/torch on CPU)
2. Upload to GPU via GPUDevice.upload()
3. Launch kernel
4. Download result via GPUDevice.download()
5. Compare vs numpy/torch reference
6. Report max abs error and pass/fail

## Known Issues
- ~20GB of VRAM already in use on the MI60 — check before large allocations
- Some tests may fail due to missing model weights (e.g., test_real_model.py)
- The engine is large (~73KB) — tests that instantiate it take time to initialize
