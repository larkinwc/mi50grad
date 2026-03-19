#!/bin/bash
# init.sh - Environment setup for TP=4 Phase 2 mission
# Idempotent

set -e

echo "Checking SSH connectivity to dev server..."
ssh -o ConnectTimeout=5 root@192.168.1.198 "echo 'Dev server reachable'" 2>/dev/null || {
    echo "WARNING: Cannot reach dev server at 192.168.1.198"
    echo "GPU tests will not work without dev server access"
}

echo "Checking local project structure..."
for f in src/kernels/gemv_int4_p2p_allreduce_rmsnorm.hip \
         src/kernels/kernel_p2p_allreduce_rmsnorm.hip \
         src/kernels/kernel_p2p_allreduce_rmsnorm_v2.hip \
         src/runtime/c_dispatch.c \
         src/inference/tp_engine.py \
         tests/bench_current_state.py \
         tests/bench_allreduce_micro.py; do
    if [ ! -f "$f" ]; then
        echo "WARNING: Expected file missing: $f"
    fi
done

echo "Init complete."
