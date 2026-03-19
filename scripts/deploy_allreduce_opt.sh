#!/bin/bash
# Deploy allreduce optimization to dev server and run benchmarks
#
# Usage:
#   ./scripts/deploy_allreduce_opt.sh [--compare|--v2|--v1]
#
# Options:
#   --compare  Compare v1 and v2 kernels (default)
#   --v2       Test only v2 optimized kernel
#   --v1       Test only v1 baseline kernel

set -e

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"
BENCHMARK_ARG="${1:---compare}"

echo "============================================================"
echo "  Allreduce Optimization Deployment Script"
echo "============================================================"

# Step 1: Sync files to dev server
echo ""
echo "Step 1: Syncing files to dev server..."
rsync -avz \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='notes/' \
    --exclude='plans/' \
    --exclude='.factory' \
    --exclude='*.pyc' \
    /Users/larkinwc/personal/ml/mi50grad/ $DEV_SERVER:$REMOTE_DIR/

echo "  ✓ Files synced"

# Step 2: Stop vLLM if running
echo ""
echo "Step 2: Stopping vLLM container (if running)..."
ssh $DEV_SERVER "docker ps --filter 'name=vllm-mobydick' --format '{{.Names}}' | grep -q vllm && docker stop vllm-mobydick || echo '  vLLM not running'"
echo "  ✓ vLLM stopped"

# Step 3: Build kernels
echo ""
echo "Step 3: Building kernels on dev server..."
ssh $DEV_SERVER "cd $REMOTE_DIR && make kernels"
echo "  ✓ Kernels built"

# Step 4: Run microbenchmark
echo ""
echo "Step 4: Running microbenchmark ($BENCHMARK_ARG)..."
ssh $DEV_SERVER "cd $REMOTE_DIR && python3 tests/bench_allreduce_micro.py $BENCHMARK_ARG"

echo ""
echo "============================================================"
echo "  Deployment Complete"
echo "============================================================"

# Step 5: Optionally restart vLLM
echo ""
read -p "Restart vLLM container? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Restarting vLLM..."
    ssh $DEV_SERVER "docker start vllm-mobydick"
    echo "  ✓ vLLM restarted"
fi

echo ""
echo "Done."
