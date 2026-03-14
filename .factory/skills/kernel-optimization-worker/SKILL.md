---
name: kernel-optimization-worker
description: Implements GPU kernel optimizations (assembly + HIP C++) with remote build/test on MI60 LXC
---

# Kernel Optimization Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for any feature that involves:
- Writing or modifying GCN assembly kernels (.s files)
- Writing or modifying HIP C++ kernels (.hip files)
- Modifying the Python kernel launcher or inference engine
- Creating kernel correctness tests
- Running performance benchmarks
- Integrating kernels into the inference engine
- Creating benchmark/comparison scripts

## Work Procedure

### Step 1: Understand the Feature
Read the feature description thoroughly. Identify:
- Which kernel(s) need to be created/modified
- What existing code to study (read the relevant files)
- What the correctness criteria are (tolerance, reference implementation)
- What the performance target is (if applicable)

Read AGENTS.md for mission boundaries and coding conventions. Read `.factory/library/architecture.md` for architectural context.

### Step 2: Study Existing Code
Before writing ANY code, read:
- The existing kernel being modified/replaced (understand current approach)
- The launcher (`src/kernels/launcher.py` — understand how kernels are loaded)
- Any existing tests for the kernel being modified
- Related kernels that use similar patterns (e.g., study gemv_w4a8_dot8 when implementing v_dot8 for INT4 GEMV)

### Step 3: Write Correctness Tests FIRST (TDD)
Create a test file `tests/test_<feature>.py` that:
1. Generates random input data (numpy/torch)
2. Computes reference output on CPU
3. Will upload inputs, launch kernel, download outputs, compare
4. Reports max absolute error and PASS/FAIL
5. Include edge cases (minimum dimensions, boundary conditions)
6. For optimization features: include a performance comparison section that benchmarks old vs new

The test should FAIL initially (kernel doesn't exist yet or produces wrong output).

### Step 4: Implement the Kernel
Write the kernel code following existing patterns:
- For HIP C++: Create/modify `src/kernels/<name>.hip`
- For assembly: Create/modify `src/asm/<name>.s`

**IMPORTANT: Create NEW kernel files (versioned names like _v3, _v4) rather than overwriting existing ones.** This allows A/B comparison.

**HIP C++ kernel conventions:**
- Use `extern "C" __global__` for kernel functions
- Use `__builtin_amdgcn_fdot2` for v_dot2_f32_f16
- Use `__builtin_amdgcn_sdot4` for v_dot4_i32_i8
- Use `__builtin_amdgcn_sdot8` for v_dot8_i32_i4
- Use `__shfl_xor` for DPP butterfly reductions
- Use `__half2` packed types for vectorized loads
- Use `__attribute__((amdgpu_flat_work_group_size(N, N)))` for workgroup size hints
- FP32/INT32 accumulators are MANDATORY
- Target 64-80 VGPRs for 4-5 waves/SIMD occupancy

### Step 5: Deploy and Build on LXC

**CRITICAL: Fix SSH auth first if needed:**
```bash
export SSH_AUTH_SOCK=$(ls /private/tmp/com.apple.launchd.*/Listeners 2>/dev/null | head -1)
```

**Deploy:**
```bash
rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' --exclude='.factory' \
    -e 'ssh -J root@wittymantis.netbird.selfhosted' \
    /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.189:/root/mi50grad/
```

**Build specific HIP kernel:**
```bash
ssh -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && mkdir -p build/kernels && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o build/kernels/KERNEL.so src/kernels/KERNEL.hip'
```

**Build assembly kernels (if needed):**
```bash
ssh -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && export ROCM_PATH=/opt/rocm && make kernels'
```

If build fails, fix compilation errors and retry. Do NOT proceed to testing with a broken build.

### Step 6: Run Correctness Tests on LXC GPU
```bash
ssh -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && PYTHONPATH=/root/mi50grad python3 tests/test_<feature>.py'
```

Tests must PASS. If they fail:
- Check if the error is in the test or the kernel
- Fix and re-deploy/re-build/re-test
- Do NOT move to performance benchmarks with failing correctness

### Step 7: Run Performance Benchmarks (if applicable)
If the feature has a performance target, measure before and after:
- Build BOTH old and new kernels in the same test script
- Run 100+ iterations with 10 warmup for each
- Report median latency (us), compute TFLOPS or GB/s where applicable
- Print speedup ratio clearly

### Step 8: Run Non-Regression Check
Run at least one existing related test to verify no regressions:
```bash
ssh -o StrictHostKeyChecking=no -J root@wittymantis.netbird.selfhosted root@192.168.1.189 \
    'cd /root/mi50grad && PYTHONPATH=/root/mi50grad python3 tests/test_<related>.py'
```

### Step 9: Update Library Knowledge
If you discovered important patterns, quirks, or constraints during implementation, update the relevant file in `.factory/library/`.

## Example Handoff

```json
{
  "salientSummary": "Implemented fused INT4 split-K GEMV with built-in zeroing and FP16 epilogue. Modified gemv_int4_v2.hip to add zero-init in first split tile and fp32→fp16 convert in final reduction. Test shows max error 3.2e-3 vs reference. Latency improved from 47us (3 launches) to 31us (1 launch) for N=4096 K=4096.",
  "whatWasImplemented": "Modified src/kernels/gemv_int4_v2.hip: added conditional zero-init in gemv_int4_v2_splitk (first split_id zeros accumulator), added fp16 epilogue in final atomicAdd pass. Removed fp32_to_fp16 kernel. Updated src/inference/engine.py _launch_gemv_int4 to skip memset and convert calls. Created tests/test_fused_int4_gemv.py with 5 test cases.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "rsync ... && ssh ... make kernels && hipcc ... gemv_int4_v2.hip",
        "exitCode": 0,
        "observation": "All kernels compiled successfully"
      },
      {
        "command": "ssh ... python3 tests/test_fused_int4_gemv.py",
        "exitCode": 0,
        "observation": "5/5 tests passed. Max error: N=4096,K=4096 → 3.2e-3, N=11008,K=4096 → 2.8e-3"
      },
      {
        "command": "ssh ... python3 tests/test_gemv_int4.py",
        "exitCode": 0,
        "observation": "Existing INT4 GEMV tests still pass (non-regression)"
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_fused_int4_gemv.py",
        "cases": [
          {"name": "test_fused_splitk_4096x4096", "verifies": "Correctness for N=4096 K=4096 group_size=128"},
          {"name": "test_fused_splitk_11008x4096", "verifies": "Correctness for FFN dimensions"},
          {"name": "test_fused_no_separate_memset", "verifies": "No memset call before launch"},
          {"name": "test_fused_fp16_output", "verifies": "Output is FP16 without separate convert"},
          {"name": "test_fused_perf_vs_baseline", "verifies": "Latency improvement over 3-launch baseline"}
        ]
      }
    ]
  },
  "discoveredIssues": [
    {
      "severity": "low",
      "description": "atomicAdd on FP32 has ~1 ULP non-determinism with different split counts",
      "suggestedFix": "Document as known behavior; does not affect practical accuracy"
    }
  ]
}
```

## When to Return to Orchestrator

- The kernel requires architectural changes not described in the feature (e.g., new memory allocation patterns in the engine)
- Build fails due to missing ROCm components or toolchain issues
- GPU is out of memory and tests cannot run
- SSH connectivity to LXC is broken
- The feature depends on another kernel or engine change that doesn't exist yet
- Correctness tests reveal that the approach described in the feature won't work
- Performance measurements show the optimization makes things worse (report data)
