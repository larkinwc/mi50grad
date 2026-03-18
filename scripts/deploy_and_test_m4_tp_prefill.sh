#!/bin/bash
# Deploy M4 TP Prefill implementation to dev server and run validation

set -e

DEV_SERVER="root@192.168.1.198"
REMOTE_PATH="/opt/mi50grad"

echo "========================================"
echo "M4 TP Prefill Deployment & Test"
echo "========================================"

# Step 1: Stop vLLM container (if running)
echo ""
echo "[1/5] Stopping vLLM container..."
ssh $DEV_SERVER "docker stop vllm-mobydick 2>/dev/null || true"

# Step 2: Sync code to dev server
echo ""
echo "[2/5] Syncing code to dev server..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='notes/' \
    --exclude='plans/' \
    --exclude='.factory' \
    /Users/larkinwc/personal/ml/mi50grad/ \
    $DEV_SERVER:$REMOTE_PATH/

# Step 3: Build kernels (if needed)
echo ""
echo "[3/5] Building kernels..."
ssh $DEV_SERVER "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $REMOTE_PATH:$REMOTE_PATH mi50grad bash -c \
    'cd $REMOTE_PATH && make hip_kernels'"

# Step 4: Run validation test
echo ""
echo "[4/5] Running M4 TP Prefill validation..."
ssh $DEV_SERVER "docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v $REMOTE_PATH:$REMOTE_PATH \
    -v /opt/models:/opt/models \
    mi50grad bash -c \
    'cd $REMOTE_PATH && python3 tests/val_m4_tp_prefill.py'"

TEST_EXIT_CODE=$?

# Step 5: Restart vLLM container
echo ""
echo "[5/5] Restarting vLLM container..."
ssh $DEV_SERVER "docker start vllm-mobydick 2>/dev/null || true"

# Report results
echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "VALIDATION PASSED ✓"
else
    echo "VALIDATION FAILED ✗ (exit code: $TEST_EXIT_CODE)"
fi
echo "========================================"

exit $TEST_EXIT_CODE
