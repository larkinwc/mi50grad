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
