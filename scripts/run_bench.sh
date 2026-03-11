#!/bin/bash
# Run microbenchmark suite on the dev server
#
# Usage: ./scripts/run_bench.sh [gpu_id=0]

set -euo pipefail

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"
DOCKER_IMAGE="mi50grad"
GPU_ID="${1:-0}"

echo "=== Running Microbenchmarks ==="
echo "GPU: $GPU_ID"
echo ""

ssh "$DEV_SERVER" "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $REMOTE_DIR:/workspace \
    $DOCKER_IMAGE \
    bash -c '
        cd /workspace
        if [ ! -d build/bench ]; then make bench; fi

        echo \"\"
        echo \"========================================\"
        echo \"  HBM Bandwidth\"
        echo \"========================================\"
        build/bench/hbm_bandwidth $GPU_ID 256

        echo \"\"
        echo \"========================================\"
        echo \"  LDS Bandwidth\"
        echo \"========================================\"
        build/bench/lds_bandwidth $GPU_ID

        echo \"\"
        echo \"========================================\"
        echo \"  Dispatch Latency\"
        echo \"========================================\"
        build/bench/dispatch_latency $GPU_ID 10000

        echo \"\"
        echo \"========================================\"
        echo \"  Occupancy Sweep\"
        echo \"========================================\"
        build/bench/occupancy_sweep $GPU_ID
    '"
