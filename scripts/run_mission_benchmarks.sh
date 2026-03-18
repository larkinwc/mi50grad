#!/bin/bash
# run_mission_benchmarks.sh
# Execute all milestone validation tests

set -e

echo "========================================"
echo " MISSION: TP4 Optimization Benchmarks"
echo " Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================"
echo ""

# Configuration
MODEL_DIR="/opt/models/Qwen3.5-27B-GPTQ-Int4"
MI50GRAD_DIR="/opt/mi50grad"

# Stop any running vLLM instance
docker stop vllm-mobydick 2>/dev/null || true

run_test() {
    local test_name="$1"
    local test_script="$2"
    
    echo ""
    echo "========================================"
    echo " Running: $test_name"
    echo "========================================"
    
    docker run --rm \
        --device=/dev/kfd --device=/dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -v "$MI50GRAD_DIR":/opt/mi50grad \
        -v /opt/models:/opt/models \
        mi50grad bash -c "cd /opt/mi50grad && python3 $test_script" || echo "FAILED: $test_name"
}

# M1: Kernel P2P Fix
run_test "M1: Kernel P2P Fix" "tests/val_m1_kernel_p2p_fix.py"

# M6: Speculative Decode (placeholder - will be created)
# run_test "M6: Speculative Decode" "tests/val_m6_speculative.py"

echo ""
echo "========================================"
echo " Current State Summary"
echo "========================================"
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v "$MI50GRAD_DIR":/opt/mi50grad \
    -v /opt/models:/opt/models \
    mi50grad bash -c "cd /opt/mi50grad && python3 tests/bench_current_state.py"

echo ""
echo "========================================"
echo " Mission Benchmarks Complete"
echo "========================================"
