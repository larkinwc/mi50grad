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
