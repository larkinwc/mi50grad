#!/bin/bash
# Deploy mi50grad to the dev server and build inside Docker
#
# Usage: ./scripts/deploy.sh [build_target]
# Examples:
#   ./scripts/deploy.sh          # sync + build all
#   ./scripts/deploy.sh probes   # sync + build ISA probes only
#   ./scripts/deploy.sh bench    # sync + build benchmarks only

set -euo pipefail

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"
DOCKER_IMAGE="mi50grad"
BUILD_TARGET="${1:-all}"

echo "=== mi50grad deploy ==="
echo "Server: $DEV_SERVER"
echo "Target: $BUILD_TARGET"
echo ""

# Sync project to dev server
echo "[1/4] Syncing to $DEV_SERVER:$REMOTE_DIR ..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='notes/' \
    --exclude='plans/' \
    . "$DEV_SERVER:$REMOTE_DIR/"

# Build Docker image if needed
echo "[2/4] Building Docker image..."
ssh "$DEV_SERVER" "cd $REMOTE_DIR && docker build -t $DOCKER_IMAGE . 2>&1 | tail -5"

# Build inside container
echo "[3/4] Building ($BUILD_TARGET)..."
ssh "$DEV_SERVER" "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $REMOTE_DIR:/workspace \
    $DOCKER_IMAGE \
    make $BUILD_TARGET"

echo "[4/4] Build complete."
echo ""
echo "To run ISA probes:"
echo "  ssh $DEV_SERVER 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v $REMOTE_DIR:/workspace $DOCKER_IMAGE bash -c \"cd /workspace && build/probes/run_probes 100000 0\"'"
echo ""
echo "To run benchmarks:"
echo "  ssh $DEV_SERVER 'docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -v $REMOTE_DIR:/workspace $DOCKER_IMAGE build/bench/hbm_bandwidth 0'"
