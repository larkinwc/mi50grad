---
name: gpu-kernel-worker
description: Implements GPU kernels and TP engine optimizations for MI50 gfx906
---

# GPU Kernel Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for:
- Implementing HIP GPU kernels for MI50 (gfx906)
- Integrating kernels into TPInferenceEngine dispatch paths
- Creating C dispatch extensions
- Validating numerical correctness and throughput

## Work Procedure

### 1. Understand the Feature
- Read the feature description and expected behavior carefully
- Identify the kernel file(s) to create/modify
- Understand the existing patterns (read similar kernels for reference)

### 2. Write Tests First (RED)
- Create test file `tests/test_<feature>.py`
- Write failing tests for:
  - Numerical correctness vs reference implementation
  - Edge cases (dimension alignment, empty inputs)
  - Integration with dispatch path
- Run tests to confirm they FAIL (pytest -xvs)

### 3. Implement (GREEN)
- Create/modify kernel file(s) in `src/kernels/`
- Use FP32 accumulation for all FP16 arithmetic
- Follow naming convention: `kernel_<function>_<topology>`
- Export `extern "C"` host-callable wrapper for ctypes
- Integrate into C dispatch or Python engine as needed

### 4. Verify Numerical Correctness
- Compare kernel output vs reference (separate kernels or numpy)
- max_abs_error < 5e-3 for FP16 equivalence
- Test with non-aligned dimensions (5100, 5122, etc.)

### 5. Benchmark Performance
- Create benchmark if throughput is expected to improve
- Use 5+ warmup steps, 50+ measurement steps
- Report throughput (tok/s), latency (ms), speedup ratio

### 6. Run Validation
- `python3 tests/test_<feature>.py` - all tests pass
- `make all` - build succeeds
- For TP=4 features: test on dev server

## Example Handoff

```json
{
  "salientSummary": "Implemented kernel_p2p_allreduce_rmsnorm_tp4 fused kernel with numerical equivalence (max_abs_error=2.1e-4) and 12% latency improvement. Integrated into C dispatch path with fallback.",
  "whatWasImplemented": "Created src/kernels/kernel_p2p_allreduce_rmsnorm.hip with fused P2P allreduce + RMSNorm kernel for TP=4. Added host-callable wrapper function. Updated c_dispatch.c to load and call fused kernel. Added fallback to separate kernels when library not found.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o kernel_p2p_allreduce_rmsnorm.so kernel_p2p_allreduce_rmsnorm.hip", "exitCode": 0, "observation": "Kernel compiled successfully"},
      {"command": "python3 tests/test_fused_allreduce_rmsnorm.py", "exitCode": 0, "observation": "All 8 tests passed. max_abs_error=2.1e-4 < 5e-3 threshold."},
      {"command": "python3 tests/bench_fused_vs_separate.py", "exitCode": 0, "observation": "Fused: 68.3us, Separate: 77.5us, Speedup: 1.13x"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_fused_allreduce_rmsnorm.py", "cases": [
        {"name": "test_numerical_equivalence", "verifies": "VAL-FUSE-001"},
        {"name": "test_dimension_alignment", "verifies": "VAL-FUSE-004"},
        {"name": "test_c_dispatch_integration", "verifies": "VAL-FUSE-005"},
        {"name": "test_fallback_path", "verifies": "VAL-FUSE-006"},
        {"name": "test_multi_gpu_consistency", "verifies": "VAL-FUSE-007"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on missing kernel infrastructure
- Numerical correctness cannot be achieved (investigate root cause first)
- Performance regression unexplained after investigation
- Dev server inaccessible after retry
