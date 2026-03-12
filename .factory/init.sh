#!/bin/bash
# Environment setup for MI50/MI60 kernel optimization mission
# Idempotent - safe to run multiple times

set -e

PROJECT_DIR="/Users/larkinwc/personal/ml/mi50grad"
REMOTE_HOST="root@192.168.1.189"
PROXY_HOST="root@wittymantis.netbird.selfhosted"
SSH_CMD="ssh -J ${PROXY_HOST} ${REMOTE_HOST}"

echo "=== Verifying local project structure ==="
cd "$PROJECT_DIR"
mkdir -p build/kernels

echo "=== Testing SSH connectivity to LXC ==="
${SSH_CMD} "echo 'SSH OK: $(hostname)'" || {
    echo "ERROR: Cannot reach LXC 108. Ensure wittymantis.netbird.selfhosted is reachable."
    exit 1
}

echo "=== Deploying code to LXC ==="
rsync -avz --exclude='.git' --exclude='build' --exclude='__pycache__' --exclude='.factory' \
    -e "ssh -J ${PROXY_HOST}" \
    "${PROJECT_DIR}/" "${REMOTE_HOST}:/root/mi50grad/"

echo "=== Building kernels on LXC ==="
${SSH_CMD} "cd /root/mi50grad && export ROCM_PATH=/opt/rocm && mkdir -p build/kernels && make kernels 2>&1"

echo "=== Building HIP kernels on LXC ==="
${SSH_CMD} "cd /root/mi50grad && export ROCM_PATH=/opt/rocm && for f in src/kernels/*.hip; do
    out=build/kernels/\$(basename \$f .hip).so
    echo \"Building \$f -> \$out\"
    /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -std=c++17 -shared -fPIC -o \$out \$f 2>&1 || echo \"WARN: Failed to build \$f\"
done"

echo "=== Verifying GPU access ==="
${SSH_CMD} "cd /root/mi50grad && PYTHONPATH=/root/mi50grad python3 -c '
from src.runtime.hip_dispatch import GPUDevice
d = GPUDevice(0)
buf = d.malloc(1024)
d.free(buf)
print(\"GPU verification OK\")
'"

echo "=== Init complete ==="
