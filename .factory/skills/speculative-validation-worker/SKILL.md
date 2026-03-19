---
name: speculative-validation-worker
description: Validates speculative decoding with real text prompts across multiple domains
---

# Speculative Validation Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Speculative decoding validation and testing:
- N-gram acceptance rate measurement with real text
- EAGLE acceptance rate measurement with real text
- Throughput benchmarking with speculative enabled
- Text coherence and quality verification
- E2E generation quality testing

## Work Procedure

### 1. Prepare Test Prompts
- Use existing test prompts from `tests/val_m6_speculative.py`:
  - Code completion (Python functions, class definitions)
  - JSON completion (nested structures, key-value patterns)
  - Conversational (multi-turn dialogue)
  - Repetitive text (high n-gram match rate)
- Ensure train/test split (60%/40%) for n-gram cache building

### 2. Run Acceptance Rate Tests
```bash
# N-gram acceptance tests
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/val_m6_speculative.py --mode ngram"'
```

### 3. Run EAGLE Tests
```bash
# EAGLE acceptance tests
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/val_m6_speculative.py --mode eagle"'
```

### 4. Benchmark Throughput with Speculative
```bash
# Full benchmark with speculative enabled
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/bench_current_state.py" 2>&1 | grep -E "tok/s|acceptance"'
```

### 5. E2E Generation Quality Test
```bash
# Generate text with speculative decoding and verify quality
ssh root@192.168.1.198 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0,1,2,3 -v /opt/mi50grad:/opt/mi50grad -v /opt/models:/opt/models mi50grad bash -c "cd /opt/mi50grad && python3 tests/e2e_speculative_generation.py"'
```

### 6. Collect and Report Metrics
- Acceptance rates per domain (code, JSON, conversational)
- Throughput comparison (speculative vs baseline)
- Text quality metrics (coherence, syntax validity)
- Draft statistics (total drafts, accepted, rejected)

## Example Handoff

```json
{
  "salientSummary": "Validated speculative decoding with real text across 4 domains. N-gram acceptance: code 58%, JSON 52%, conversational 44%, repetitive 72%. EAGLE acceptance: 47% average. Throughput with speculative: 53.1 tok/s (vs 51.75 baseline, 2.6% improvement). All E2E generation tests passed.",
  "whatWasImplemented": "Ran full validation suite for speculative decoding with real text prompts. Used train/test split methodology for n-gram cache. Measured acceptance rates across code, JSON, conversational, and repetitive text domains. Benchmarked throughput with speculative enabled. Verified E2E generation quality for all task types.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "python3 tests/val_m6_speculative.py --mode ngram",
        "exitCode": 0,
        "observation": "N-gram acceptance rates: code=58%, JSON=52%, conv=44%, repetitive=72%. Overall avg: 56.5%"
      },
      {
        "command": "python3 tests/val_m6_speculative.py --mode eagle",
        "exitCode": 0,
        "observation": "EAGLE acceptance rate: 47% average across domains"
      },
      {
        "command": "python3 tests/bench_current_state.py",
        "exitCode": 0,
        "observation": "Speculative decode (n-gram): 53.1 tok/s. EAGLE: 52.8 tok/s. Baseline: 51.75 tok/s"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Verified E2E code completion produces valid Python syntax",
        "observed": "Generated function completed successfully, syntax valid"
      },
      {
        "action": "Verified E2E JSON completion produces valid JSON",
        "observed": "Generated JSON object parsed successfully"
      }
    ]
  },
  "tests": {
    "added": []
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Speculative acceptance rates significantly below target (<40%)
- Throughput regression when speculative is enabled
- Text quality issues (garbage output, NaN values)
- Integration issues with TP engine speculative mode
- Missing tokenizer or model weights for real text testing
