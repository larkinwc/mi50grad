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
- `tests/bench_tp4_sprint4.py` - TP=4 benchmark (working alternative)
- `tests/val_m*.py` - Milestone-specific validation tests
- `tests/test_*.py` - Feature-specific correctness tests

## Validation Concurrency

**Max concurrent validators: 1**

Rationale: Only one Docker container can hold all 4 GPUs at a time. vLLM must be stopped to free VRAM. Tests require exclusive access to all 4 MI50 GPUs. Sequential validation only.

## Known Constraints

- vLLM container (vllm-mobydick) uses ~93% VRAM on all 4 GPUs - must be stopped before any GPU test
- Docker image `mi50grad` defaults to 3 GPUs in Dockerfile - must override with `HIP_VISIBLE_DEVICES=0,1,2,3`
- bench_current_state.py crashes with segfault when deferred AR is enabled (to be fixed in M3)
- PCIe 3.0 x16 links between GPUs (no XGMI/Infinity Fabric) - limits peer read bandwidth to ~12.8 GB/s
