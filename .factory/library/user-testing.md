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
