# User Testing

Testing surface, resource cost classification, and validation strategy.

## Validation Surface

All validation runs on the remote GPU server (root@192.168.1.198) via SSH + Docker.

**Surface type:** CLI / SSH command output
**Tools:** SSH + Docker run commands, Python test scripts
**No browser or UI testing needed.**

### Test execution pattern:
1. Stop vLLM: `ssh root@192.168.1.198 "docker stop vllm-mobydick 2>/dev/null || true"`
2. Deploy code: `rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' --exclude='notes/' --exclude='plans/' --exclude='.factory' /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/`
3. Build (if needed): HIP kernels or C extensions
4. Run test: `ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c 'cd /opt/mi50grad && python3 tests/TEST.py'"`
5. Parse output for PASS/FAIL metrics
6. Restart vLLM: `ssh root@192.168.1.198 "docker start vllm-mobydick 2>/dev/null || true"`

### Key metrics in test output:
- `cosine sim = X.XXXXXX` — correctness (threshold: >= 0.99)
- `X.X tok/s` — throughput
- `X.X ms/tok` — latency
- `max_err = X.XXe-X` — kernel-level correctness
- `PASS` / `FAIL` — overall test result

## Validation Concurrency

**Max concurrent validators: 1**

Rationale: All tests run on the shared GPU server with 4 MI50s. TP=4 tests consume ALL 4 GPUs. Only one TP=4 test can run at a time. The GPU server has 128GB system RAM and 4x 32GB HBM2 GPUs — sufficient for one test at a time but not concurrent tests. vLLM must also be stopped before each test run, which is a global state change.

## Known Constraints

- vLLM container (vllm-mobydick) must be stopped before tests and restarted after
- Docker container uses `mi50grad` image with ROCm 7.1.0
- Model weights at `/opt/models/Qwen3.5-27B-GPTQ-Int4` on the server
- TP=4 tests need `-e HIP_VISIBLE_DEVICES=0,1,2,3` (Dockerfile defaults to 3 GPUs)
- Each TP=4 benchmark run takes ~2-5 minutes (weight loading + 100 decode steps)
- Correctness checks add ~1-2 minutes (sequential single-GPU then TP=4 runs)

## Flow Validator Guidance: SSH / CLI (GPU Server)

All validation for this project happens via SSH to root@192.168.1.198 running Python test scripts inside the `mi50grad` Docker container.

**Isolation rules:**
- ONLY ONE validator can run at a time (max concurrency = 1)
- ALL tests require GPU access — no concurrent GPU tests
- vLLM must be stopped before tests and restarted after (global state)
- Do NOT touch vllm-mobydick container except to stop before and start after

**Resource boundaries:**
- GPU server: root@192.168.1.198
- Docker container: `mi50grad`
- Model path: `/opt/models/Qwen3.5-27B-GPTQ-Int4`
- Code path: `/opt/mi50grad/`
- GPUs for TP=4: `-e HIP_VISIBLE_DEVICES=0,1,2,3`
- GPUs for single-GPU: no HIP_VISIBLE_DEVICES override needed (defaults to GPU 0)

**Test execution pattern:**
```bash
# Stop vLLM
ssh root@192.168.1.198 "docker stop vllm-mobydick 2>/dev/null || true"
# Run test
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c 'cd /opt/mi50grad && python3 tests/TEST.py'"
# Restart vLLM after ALL tests are done
ssh root@192.168.1.198 "docker start vllm-mobydick 2>/dev/null || true"
```

**Test scripts for kernel-tuning milestone:**
- `tests/test_kernel_tuning.py` — kernel tuning sweep (VAL-TUNE-001, 002, 003, 005)
- `tests/bench_tp4_sprint2.py` — final TP=4 benchmark (VAL-TUNE-004, VAL-CROSS-001, 002, 003, 004, VAL-RING-006)

**Timeout guidance:**
- test_kernel_tuning.py: ~15-20 minutes (kernel build + sweep)
- bench_tp4_sprint2.py: ~15-25 minutes (two model loads + 100-step benchmarks + correctness)

**PASS indicators in output:**
- `PASS` or `ALL PASS` in the last few lines
- `OVERALL: ALL CRITICAL CHECKS PASSED`
- Individual assertion lines: `VAL-XXX-NNN (...): PASS`

**FAIL indicators:**
- `FAIL` in output
- `sys.exit(1)` exit code (non-zero from docker run)
- Missing output or Python exception traceback
