# User Testing

Testing surface, resource cost classification, and validation approach.

## Validation Surface

**Surface:** GPU benchmark output via SSH + Docker on dev server (root@192.168.1.198)
**Tools:** Python scripts executed in Docker container with GPU access
**No browser/TUI/CLI surface** - all validation is automated

### Test Execution Pattern

1. Deploy code: `rsync -avz ... root@192.168.1.198:/opt/mi50grad/`
2. Stop vLLM: `ssh root@192.168.1.198 'docker stop vllm-mobydick'`
3. Run test: `ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/TEST.py"'`
4. Restart vLLM: `ssh root@192.168.1.198 'docker start vllm-mobydick'`

### Key Test Scripts

- `tests/bench_current_state.py` - Main benchmark (all modes)
- `tests/bench_allreduce_micro.py` - Allreduce microbenchmark
- `tests/test_fused_gemv_isolate.py` - Fused GEMV correctness
- `tests/bench_tp4_sprint4.py` - TP=4 benchmark (alternative)

## Validation Concurrency

**Max concurrent validators: 1**

Rationale: Only one Docker container can hold all 4 GPUs at a time. Sequential validation only.

## Known Constraints

- vLLM container uses ~93% VRAM on all 4 GPUs - must be stopped
- Docker defaults to 3 GPUs - must override with HIP_VISIBLE_DEVICES=0,1,2,3
- PCIe 4.0 x16, 2-hop via switch, ~25.6 GB/s theoretical
- **Fused GEMV+P2P+RMSNorm kernels produce NaN in single-GPU or isolated Python threading tests** -- these kernels require all 4 GPUs to launch simultaneously with cross-WG atomic counter coordination. Validation MUST go through the full TPInferenceEngine pipeline (C dispatch), NOT isolated kernel tests.

## Flow Validator Guidance: GPU Benchmark via SSH+Docker

### Isolation Rules
- Only ONE benchmark can run at a time (single Docker container with all 4 GPUs)
- No parallel validation - must serialize all tests
- Each benchmark run takes 10-15 minutes (model loading + 100 steps)

### Test Execution
1. Stop vLLM: `ssh root@192.168.1.198 'docker stop vllm-mobydick'`
2. Deploy code: `rsync -avz --delete --exclude='.git' --exclude='build/' --exclude='__pycache__' --exclude='notes/' --exclude='plans/' --exclude='.factory' /Users/larkinwc/personal/ml/mi50grad/ root@192.168.1.198:/opt/mi50grad/`
3. Build: `ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v /opt/mi50grad:/opt/mi50grad mi50grad bash -c "cd /opt/mi50grad && make hip_kernels c_extensions"'`
4. Run benchmark: `ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/bench_current_state.py"'`

### Compressed Allreduce Testing
- Run bench_current_state.py to get baseline throughput
- Modify engine to call `tp.set_compressed_allreduce(True)` and benchmark compressed mode
- Compare throughput: compressed vs uncompressed
- Target: compressed throughput > 53.74 tok/s baseline
- Correctness: compare logit outputs (cosine_sim >= 0.99)
- Single-GPU regression check: >= 21.0 tok/s

### Validation Assertions for compressed-allreduce Milestone
- VAL-CA-003: Fused GEMV+compressed AR+RMSNorm kernel correctness (cos_sim >= 0.99 vs uncompressed)
- VAL-CA-004: Fused compressed kernel compiles and is A/B testable (build exits 0, flag toggle works)
- VAL-CA-005: C dispatch integration with compressed allreduce (flag toggle produces correct results)
- VAL-CA-006: TP engine mode selection for compressed allreduce (set_compressed_allreduce API works)
- VAL-CA-007: TP4 throughput improvement with compressed allreduce (throughput > 53.74 tok/s)
- VAL-CA-008: No single-GPU regression (>= 21.0 tok/s)

NOTE: VAL-CA-001 and VAL-CA-002 were for the cancelled standalone kernel approach. The fused kernel implementation replaces them.
