#!/bin/bash
# Deploy and test VAL-DB-002 fix on dev server
# Run this from the project root on your local machine

set -e

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"

echo "==================================================================="
echo "Deploying VAL-DB-002 fix to dev server"
echo "==================================================================="

# Sync updated files to dev server
echo "Syncing files to dev server..."
rsync -avz \
    src/inference/tp_engine.py \
    tests/test_double_buffer_tp4.py \
    DOUBLE_BUFFER_VALIDATION_SUMMARY.md \
    ${DEV_SERVER}:${REMOTE_DIR}/

echo ""
echo "==================================================================="
echo "Running VAL-DB-002 numerical correctness test on dev server"
echo "==================================================================="

# Run the test on dev server
ssh ${DEV_SERVER} "cd ${REMOTE_DIR} && docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v ${REMOTE_DIR}:/workspace -v /opt/models:/opt/models \
    mi50grad python3 tests/test_double_buffer_tp4.py --correctness --steps 20"

echo ""
echo "==================================================================="
echo "Test complete. Check output above for results."
echo "==================================================================="
echo ""
echo "Expected: Min cosine similarity >= 0.99"
echo "If test passes, proceed with full validation suite:"
echo "  ssh root@192.168.1.198 'cd /opt/mi50grad && docker run --rm ... mi50grad python3 tests/test_double_buffer_tp4.py --all'"
