---
name: kernel-optimization-worker
description: Implements GPU kernel optimizations (HIP C++) and TP engine changes with remote build/test on Docker dev server
---

# Kernel Optimization Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for any feature that involves:
- Writing or modifying HIP C++ kernels (.hip files)
- Modifying the Python inference engine (engine.py, tp_engine.py)
- Modifying runtime components (tensor_parallel.py, hip_dispatch.py, fast_allreduce.c)
- Creating correctness tests and performance benchmarks
- Integrating new allreduce or dispatch mechanisms

## Work Procedure

### Step 1: Understand the Feature
Read the feature description thoroughly. Identify:
- Which files need to be created/modified
- What existing code to study (read the relevant files)
- What the correctness criteria are (tolerance, reference implementation)
- What the performance target is (if applicable)

Read AGENTS.md for mission boundaries and coding conventions. Read `.factory/library/architecture.md` for architectural context.

### Step 2: Study Existing Code
Before writing ANY code, read:
- The existing code being modified/replaced (understand current approach)
- Related files that interact with the code being changed
- Any existing tests for the functionality being modified
- The `src/inference/tp_engine.py` decode_step loop (understand the allreduce pattern)

### Step 3: Write Correctness Tests FIRST (TDD)
Create a test file `tests/test_<feature>.py` that:
1. Generates test inputs or loads model weights as needed
2. Computes reference output (e.g., single-GPU decode, or separate allreduce path)
3. Runs the optimized code path
4. Compares outputs (cosine similarity > 0.99 for decode, max abs error for kernels)
5. Reports PASS/FAIL with specific metrics
6. For performance features: include a benchmark section comparing old vs new

The test should FAIL initially (new code doesn't exist yet or produces wrong output).

### Step 4: Implement the Change
Write the code following existing patterns:
- For HIP C++: Create/modify `src/kernels/<name>.hip`
- For C extensions: Create/modify `src/runtime/<name>.c`
- For Python: Modify `src/inference/tp_engine.py`, `engine.py`, etc.

**IMPORTANT: Create NEW files (versioned names) rather than overwriting working existing ones.** This allows A/B comparison.

**HIP C++ kernel conventions:**
- Use `extern "C" __global__` for kernel functions
- Use `__attribute__((amdgpu_flat_work_group_size(256, 256)))` for occupancy hints
- Use FP32 accumulators (mandatory on gfx906)
- Target 64-80 VGPRs for 4-5 waves/SIMD occupancy

### Step 5: Deploy and Build

**Deploy to server:**
```bash
rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' --exclude='notes/' --exclude='plans/' --exclude='.factory' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/
```

**Build HIP kernel:**
```bash
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c \
    'cd /opt/mi50grad && mkdir -p build/kernels && \
     /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC \
     -o build/kernels/KERNEL.so src/kernels/KERNEL.hip'"
```

**Build C extension:**
```bash
ssh root@192.168.1.198 "docker run --rm -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c \
    'cd /opt/mi50grad && gcc -O3 -mf16c -mavx -shared -fPIC \
     -o src/runtime/EXTENSION.so src/runtime/EXTENSION.c'"
```

**Build assembly kernels (if needed):**
```bash
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c \
    'cd /opt/mi50grad && export ROCM_PATH=/opt/rocm && make kernels'"
```

If build fails, fix compilation errors and retry. Do NOT proceed to testing with a broken build.

### Step 6: Run Correctness Tests on GPU Server

**IMPORTANT: Stop vLLM first to free GPU VRAM (it uses 93% on all 4 GPUs):**
```bash
ssh root@192.168.1.198 "docker stop vllm-mobydick 2>/dev/null || true"
```

**Run test:**
```bash
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/test_<feature>.py'"
```

Tests must PASS. If they fail:
- Check if the error is in the test or the implementation
- Fix and re-deploy/re-build/re-test
- Do NOT move to benchmarks with failing correctness

### Step 7: Run Performance Benchmarks (if applicable)
If the feature has a performance target:
- Build BOTH old and new paths in the same test script
- Run 100+ iterations with 10 warmup for each
- Report median latency (us), tok/s, and speedup ratio
- Print results clearly

### Step 8: Run Regression Check
Run at least one existing related test to verify no regressions:
```bash
ssh root@192.168.1.198 "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models \
    mi50grad bash -c 'cd /opt/mi50grad && python3 tests/bench_single_gpu.py'"
```

### Step 9: Restart vLLM
```bash
ssh root@192.168.1.198 "docker start vllm-mobydick 2>/dev/null || true"
```

### Step 10: Update Library Knowledge
If you discovered important patterns, quirks, or constraints during implementation, update the relevant file in `.factory/library/`.

## Example Handoff

```json
{
  "salientSummary": "Implemented P2P allreduce using hipMemcpyPeerAsync gather + on-device reduce kernel for TP=4. Replaces host-mediated AVX path. Allreduce latency reduced from 85us to 28us per call (3.0x speedup). TP=4 correctness validated: cosine sim = 0.9987 vs single-GPU.",
  "whatWasImplemented": "Created src/runtime/p2p_allreduce.hip with reduce_sum_fp16 kernel. Updated src/inference/tp_engine.py _allreduce_residual to use async P2P gather (hipMemcpyPeerAsync) + device-side reduce + async broadcast, replacing the synchronous host-mediated fast_allreduce.c path. Created tests/test_p2p_allreduce.py with microbenchmark and correctness tests.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "rsync ... && ssh root@192.168.1.198 docker run ... hipcc ... p2p_allreduce.hip",
        "exitCode": 0,
        "observation": "Kernel compiled successfully for gfx906"
      },
      {
        "command": "ssh root@192.168.1.198 docker run ... python3 tests/test_p2p_allreduce.py",
        "exitCode": 0,
        "observation": "Old allreduce: 85us/call, New P2P: 28us/call. Speedup: 3.0x. Correctness: max_err=2.1e-4"
      },
      {
        "command": "ssh root@192.168.1.198 docker run ... python3 tests/bench_single_gpu.py",
        "exitCode": 0,
        "observation": "Single GPU: 20.1 tok/s (no regression)"
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_p2p_allreduce.py",
        "cases": [
          {"name": "test_p2p_vs_host_correctness", "verifies": "P2P allreduce produces same result as host allreduce"},
          {"name": "test_p2p_tp4_correctness", "verifies": "TP=4 allreduce correctness with all 4 GPUs"},
          {"name": "test_p2p_latency_improvement", "verifies": "P2P is faster than host-mediated path"},
          {"name": "bench_p2p_vs_host", "verifies": "Latency comparison over 100 iterations"}
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- SSH connectivity to dev server is broken
- Docker container fails to start or access GPUs
- GPU is out of memory even after stopping vLLM
- The feature depends on another feature's code that doesn't exist yet
- Correctness tests reveal the approach won't work (report data)
- Performance measurements show the optimization makes things worse (report data)
- P2P peer access doesn't work despite being reported as available
- Build fails due to missing ROCm components
