# User Testing

Testing surface, resource cost classification, and validation approach for this mission.

## Validation Surface

**Surface type:** Command-line (remote SSH + Docker)

All validation is done by running benchmark scripts on the remote GPU server (root@192.168.1.198) via SSH. Tests execute inside the `mi50grad` Docker container with GPU access. There is no web UI, API server, or interactive application to test.

**Testing tools:** SSH + Docker exec (no agent-browser or tuistory needed)

**Setup required before testing:**
1. Stop vLLM container: `ssh root@192.168.1.198 "docker stop vllm-mobydick"`
2. Deploy latest code: rsync to /opt/mi50grad/
3. Build kernels inside Docker

**Cleanup after testing:**
1. Restart vLLM: `ssh root@192.168.1.198 "docker start vllm-mobydick"`

**Verification pattern:**
- Run test/benchmark script via Docker
- Parse terminal output for pass/fail + numeric results
- Compare against baselines:
  - Single-GPU: 20.3 tok/s
  - vLLM TP=4: 46.9 tok/s
  - Target: >50 tok/s

## Validation Concurrency

**Max concurrent validators: 1**

Rationale: All tests require exclusive GPU access on a single remote server. The 4 GPUs have shared VRAM (vLLM must be stopped to free memory). Only one test can run at a time to avoid GPU memory contention. Multiple concurrent tests would OOM or interfere with each other's GPU state.

## Known Constraints

- vLLM must be stopped before any TP test (uses 93% VRAM on all GPUs)
- Model weight loading takes ~30-60 seconds per test run (48GB+ across 4 GPUs)
- Each TP=4 benchmark run takes ~2-5 minutes (weight loading + warmup + 100 steps)
- SSH connection timeouts may occur for long-running tests (use `-o ServerAliveInterval=30`)
- bench_tp4_final.py runs 3 sequential phases (single-GPU, TP=4 correctness, TP=4 combined benchmark) — total runtime ~20-30 minutes

## Flow Validator Guidance: CLI

**Surface:** Remote SSH + Docker command-line benchmarks on root@192.168.1.198

**Isolation rules:**
- Only one validator at a time (max concurrent = 1)
- Always stop vLLM before running GPU tests: `ssh root@192.168.1.198 "docker stop vllm-mobydick"`
- Always restart vLLM after tests: `ssh root@192.168.1.198 "docker start vllm-mobydick"`
- Use `-e HIP_VISIBLE_DEVICES=0,1,2,3` for TP=4 tests (Dockerfile defaults to 0,1,2 only)
- SSH with ServerAliveInterval to prevent timeouts on long-running tests

**Deploy pattern (required before running tests):**
```bash
rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' --exclude='notes/' --exclude='plans/' --exclude='.factory' /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/
```

**Build kernels pattern:**
```bash
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c 'cd /opt/mi50grad && mkdir -p build/kernels && for f in src/kernels/*.hip; do /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o build/kernels/\$(basename \$f .hip).so \$f 2>&1; done'"
```

**Run test pattern:**
```bash
ssh -o ServerAliveInterval=30 root@192.168.1.198 "docker stop vllm-mobydick 2>/dev/null; docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c 'cd /opt/mi50grad && python3 tests/TEST.py 2>&1'; docker start vllm-mobydick"
```

**Evidence to capture:**
- Terminal output showing tok/s, cosine similarity values, pass/fail per VAL assertion
- Bench report file at bench/tp4_optimization_report.md

**Test files for advanced-fusions assertions:**
- `tests/bench_tp4_final.py` — covers VAL-FINAL-001, VAL-FINAL-002, VAL-FINAL-003, VAL-FINAL-004
- `tests/test_fused_p2p_gemv.py` — covers VAL-ADV-001 (fused GEMV correctness + performance)
