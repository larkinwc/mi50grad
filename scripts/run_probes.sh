#!/bin/bash
# Run ISA probe suite on the dev server
#
# Usage: ./scripts/run_probes.sh [num_iters=100000] [gpu_id=0]

set -euo pipefail

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/mi50grad"
DOCKER_IMAGE="mi50grad"
NUM_ITERS="${1:-100000}"
GPU_ID="${2:-0}"

echo "=== Running ISA Probes ==="
echo "Iterations: $NUM_ITERS, GPU: $GPU_ID"
echo ""

# The probe runner needs to be in the same directory as the .s files
# (it builds them on-the-fly), OR we use pre-built HSACO files.
# Strategy: run the HIP binary which builds+runs the probes.
ssh "$DEV_SERVER" "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $REMOTE_DIR:/workspace \
    -w /workspace/tests/isa_probes \
    $DOCKER_IMAGE \
    bash -c '
        # Build the probe runner if not already built
        if [ ! -f /workspace/build/probes/run_probes ]; then
            cd /workspace && make probes
        fi
        # Run from the isa_probes dir so it can find .s files
        cd /workspace/tests/isa_probes
        /workspace/build/probes/run_probes $NUM_ITERS $GPU_ID
    '"
