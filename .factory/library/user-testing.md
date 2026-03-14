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

## Validation Concurrency
Max concurrent validators: **3** (GPU is a singleton resource; tests that allocate large VRAM tensors should not run in parallel)
Note: Individual kernel tests allocate modest VRAM and can run sequentially within a single subagent. Integration tests (engine.py) are heavy and must be serialized.

## Flow Validator Guidance: GPU Kernel (SSH/Python)

**Surface**: SSH to LXC 108, run Python test scripts on MI60 GPU

**SSH Setup** (required for every command):
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
ssh -o ConnectTimeout=20 -J root@wittymantis.netbird.selfhosted root@192.168.1.189 'COMMAND'
```

**Isolation rules**:
- Only one subagent should run a given test file at a time (no parallel execution of the same test)
- Tests are run sequentially within a subagent
- VRAM is shared — no more than 2 subagents allocating large tensors simultaneously
- Do NOT kill existing GPU processes

**Test execution pattern**:
```bash
ssh -o ConnectTimeout=20 -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && PYTHONPATH=/root/mi50grad python3 tests/TEST_FILE.py'
```

**Evidence collection**: Capture stdout/stderr from each test. Look for PASS/FAIL lines and max abs error values.

**Key test files for decode-launch-reduction milestone**:
- `tests/test_deltanet_gpu_always.py` — VAL-DLR-001
- `tests/test_skip_rmsnorm_decode.py` — VAL-DLR-002
- `tests/test_fused_int4_gemv.py` — VAL-DLR-003
- `tests/test_fused_dual_gemv.py` — VAL-DLR-004, VAL-DLR-011
- `tests/test_qknorm_rope.py` — VAL-DLR-005, VAL-DLR-010
- `tests/test_m1_integration.py` — VAL-DLR-006, VAL-DLR-007, VAL-DLR-008, VAL-DLR-009, VAL-CROSS-001, VAL-CROSS-007

## Flow Validator Guidance: GPU Kernel - w8a8-w4a8-flashattn

**Surface**: SSH to LXC (mi60-jupyter at 192.168.1.189), run Python test scripts on MI60 GPU

**SSH Setup** (required for every command):
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 'COMMAND'
```

**Isolation rules**:
- Run tests sequentially within a subagent (GPU is single, VRAM shared)
- ~58% VRAM (~18GB of 32GB) already in use — individual kernel tests allocate modest VRAM and are fine
- Integration tests (engine.py) may need more VRAM — run in separate batch
- Do NOT kill existing GPU processes

**Kernels are pre-built** in `/root/mi50grad/build/kernels/`:
- `activation_quant.so` — for VAL-W8A8-003
- `gemv_w8a8.so` — for VAL-W8A8-001, VAL-W8A8-004
- `gemm_w8a8.so` — for VAL-W8A8-002
- `gemv_w4a8.so` — for VAL-W4A8-001, VAL-W4A8-002, VAL-W4A8-003
- `flash_attn_256.so` and `flash_attn_256_tuned.hsaco` — for VAL-FA-001, VAL-FA-002

**Key test files for w8a8-w4a8-flashattn milestone**:
- `tests/test_activation_quant.py` — VAL-W8A8-003
- `tests/test_w8a8_gemv.py` — VAL-W8A8-001, VAL-W8A8-004
- `tests/test_w8a8_gemm.py` — VAL-W8A8-002
- `tests/bench_w8a8.py` — VAL-W8A8-004
- `tests/test_w4a8_gemv.py` — VAL-W4A8-001, VAL-W4A8-003
- `tests/test_w4a8_repack.py` — VAL-W4A8-002
- `tests/bench_w4a8.py` — VAL-W4A8-003
- `tests/test_flashattn_tune.py` — VAL-FA-001, VAL-FA-002
- `tests/test_engine_w8a8_w4a8.py` — VAL-ENG-001, VAL-ENG-002

---

## Flow Validator Guidance: GPU Kernel - core-kernel-optimization

**Surface**: SSH to LXC (mi60-jupyter at 192.168.1.189), run Python test scripts on MI60 GPU

**CRITICAL SSH Note**: The LXC host key changed (mi60-jupyter, not mi60-lxc). Remove known_hosts entry if needed:
```bash
ssh-keygen -R 192.168.1.189
```

**SSH Setup** (required for every command):
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 'COMMAND'
```

**Isolation rules**:
- Run tests sequentially within a subagent (GPU is single, VRAM shared)
- ~20GB VRAM already in use — individual kernel tests allocate modest VRAM and are fine
- Integration tests (engine.py) may need more VRAM — run in separate batch
- Do NOT kill existing GPU processes

**Kernels are pre-built** in `/root/mi50grad/build/kernels/` — no rebuild needed unless code changed.

**Key test files for core-kernel-optimization milestone**:
- `tests/test_prefill_gemm_dot2.py` — VAL-GEMM-001, VAL-GEMM-002, VAL-GEMM-003
- `tests/test_int4_gemv_optimize.py` — VAL-INT4-001, VAL-INT4-002
- `tests/test_gemv_int4_v3.py` — VAL-INT4-003
- `tests/test_gemv_int4.py` — VAL-INT4-004 (DPP wave reduction) 
- `tests/test_rope_v2.py` — VAL-ROPE-001, VAL-ROPE-002
- `tests/test_m2_decode.py` — VAL-CROSS-002
- `tests/test_m2_prefill.py` — VAL-CROSS-003
- `tests/test_gemm_fp16.py` + `tests/test_gemv_int4.py` + others — VAL-CROSS-004, VAL-CROSS-005
- `tests/test_m1m2_combined.py` — VAL-CROSS-006

---

## Flow Validator Guidance: GPU Kernel - isa-vectorization

**Surface**: SSH to LXC (mi60-jupyter at 192.168.1.189), run Python test scripts on MI60 GPU

**SSH Setup** (required for every command):
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 'COMMAND'
```

**Isolation rules**:
- Run tests sequentially within a subagent (GPU is single, VRAM shared)
- VRAM currently at 0% — kernel tests allocate modest VRAM and are fine
- Do NOT kill existing GPU processes

**Deploy before testing** (if needed):
```bash
rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' --exclude='.factory' \
    -e 'ssh -J root@wittymantis.netbird.selfhosted' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.189:/root/mi50grad/
```

**Key test files for isa-vectorization milestone**:
- `tests/test_gemv_int4_v4.py` — VAL-I4GV-001 (correctness at N=4096,K=4096 and N=11008,K=4096), VAL-I4GV-002 (latency comparison v4 vs v3)
- `tests/test_elementwise_v3.py` — VAL-ELEM-001 (correctness: residual_add, silu_fused, rmsnorm, skip_rmsnorm at dim=5120), VAL-ELEM-002 (bandwidth comparison v3 vs v2 GB/s)

**Evidence format**: Capture full stdout/stderr from each test command. Look for PASS/FAIL lines and max abs error values and bandwidth (GB/s) numbers.

---

## Flow Validator Guidance: GPU Kernel - compute-kernels

**Surface**: SSH to LXC (mi60-jupyter at 192.168.1.189), run Python test scripts on MI60 GPU

**SSH Setup** (required for every command):
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 'COMMAND'
```

**Isolation rules**:
- Run tests sequentially within a subagent (GPU is single, VRAM shared)
- ~20GB VRAM already in use — individual kernel tests allocate modest VRAM and are fine
- Integration tests (engine.py) may need more VRAM — run in separate batch
- Do NOT kill existing GPU processes
- Max 2 subagents running GPU tests concurrently

**Deploy before testing** (if needed):
```bash
rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' --exclude='.factory' \
    -e 'ssh -J root@wittymantis.netbird.selfhosted' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.189:/root/mi50grad/
```

**Key test files for compute-kernels milestone**:
- `tests/test_flash_attn_v3.py` — VAL-FA-001, VAL-FA-002, VAL-FA-003 (block-tiled FlashAttention v3: correctness, perf, decode regression)
- `tests/test_gemm_fp16_db.py` — VAL-GEMM-001, VAL-GEMM-002 (double-buffered FP16 GEMM: correctness, perf)
- `tests/test_gemm_int4_v2.py` — VAL-I4GM-001, VAL-I4GM-002 (on-the-fly dequant INT4 GEMM: correctness, perf)

**Evidence format**: Capture full stdout/stderr from each test command. Look for PASS/FAIL lines and error values.
