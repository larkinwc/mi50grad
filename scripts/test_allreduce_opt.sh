#!/bin/bash
# Automated test script for allreduce optimization
# Runs non-interactively on dev server
#
# Usage:
#   ./scripts/test_allreduce_opt.sh

set -e

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"

echo "============================================================"
echo "  Allreduce Optimization - Automated Test"
echo "============================================================"

# Sync files
echo ""
echo "Syncing files..."
rsync -avz \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='notes/' \
    --exclude='plans/' \
    --exclude='.factory' \
    --exclude='*.pyc' \
    /Users/larkinwc/personal/ml/mi50grad/ $DEV_SERVER:$REMOTE_DIR/ >/dev/null
echo "  ✓ Synced"

# Stop vLLM
echo ""
echo "Stopping vLLM..."
ssh $DEV_SERVER "docker stop vllm-mobydick 2>/dev/null || true"
echo "  ✓ Stopped"

# Build kernels
echo ""
echo "Building kernels..."
ssh $DEV_SERVER "cd $REMOTE_DIR && make kernels" 2>&1 | tail -5
echo "  ✓ Built"

# Run benchmark
echo ""
echo "Running comparative benchmark..."
echo ""
ssh $DEV_SERVER "cd $REMOTE_DIR && python3 tests/bench_allreduce_micro.py --compare"

# Restart vLLM
echo ""
echo "Restarting vLLM..."
ssh $DEV_SERVER "docker start vllm-mobydick" || true
echo "  ✓ Restarted"

echo ""
echo "============================================================"
echo "  Test Complete"
echo "============================================================"
