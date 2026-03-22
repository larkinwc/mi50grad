#!/bin/bash
# Deploy and test fused QKV kernel integration
set -e

echo "=============================================="
echo "Deploying Fused QKV Kernel Integration"
echo "=============================================="

# Sync code to dev server
echo ""
echo "[1/4] Syncing code to dev server..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='build/' \
    --exclude='__pycache__' \
    --exclude='notes/' \
    --exclude='plans/' \
    --exclude='.factory' \
    /Users/larkinwc/personal/ml/mi50grad/ \
    root@192.168.1.198:/opt/mi50grad/

echo "Sync complete!"

# Build kernels and C extensions
echo ""
echo "[2/4] Building kernels and C extensions..."
ssh root@192.168.1.198 'docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v /opt/mi50grad:/opt/mi50grad \
    mi50grad bash -c "cd /opt/mi50grad && make hip_kernels c_extensions"'

echo "Build complete!"

# Run kernel isolation test
echo ""
echo "[3/4] Running kernel isolation test..."
ssh root@192.168.1.198 'docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -e HIP_VISIBLE_DEVICES=0 \
    -v /opt/mi50grad:/opt/mi50grad \
    -v /opt/models:/opt/models \
    mi50grad bash -c "cd /opt/mi50grad && python3 tests/test_gemv_qkv_fused_isolate.py"'

ISOLATION_RESULT=$?

if [ $ISOLATION_RESULT -eq 0 ]; then
    echo "Isolation test PASSED!"
else
    echo "Isolation test FAILED!"
    exit 1
fi

# Run benchmark
echo ""
echo "[4/4] Running benchmark..."
ssh root@192.168.1.198 'docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    -v /opt/mi50grad:/opt/mi50grad \
    -v /opt/models:/opt/models \
    mi50grad bash -c "cd /opt/mi50grad && python3 tests/bench_qkv_fused.py"'

echo ""
echo "=============================================="
echo "Deployment and Testing Complete!"
echo "=============================================="
