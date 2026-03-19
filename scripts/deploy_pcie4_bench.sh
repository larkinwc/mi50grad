#!/bin/bash
# Deploy PCIe 4.0 bandwidth benchmark to dev server

set -e

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"

echo "Deploying PCIe 4.0 bandwidth benchmark to $DEV_SERVER..."

# Sync files
rsync -avz --delete \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='notes/' \
    --exclude='plans/' \
    --exclude='.factory' \
    --exclude='.pytest_cache' \
    /Users/larkinwc/personal/ml/mi50grad/ $DEV_SERVER:$REMOTE_DIR/

echo "Deployment complete!"
echo ""
echo "To run the benchmark on the dev server:"
echo "  ssh $DEV_SERVER"
echo "  cd $REMOTE_DIR"
echo "  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\"
echo "    -e HIP_VISIBLE_DEVICES=0,1,2,3 \\"
echo "    -v $REMOTE_DIR:$REMOTE_DIR -v /opt/models:/opt/models \\"
echo "    mi50grad bash -c \"cd $REMOTE_DIR && python3 tests/bench_pcie4_bandwidth.py\""
