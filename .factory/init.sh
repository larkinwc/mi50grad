#!/bin/bash
# Environment setup for MI50/MI60 kernel optimization mission
# Idempotent - safe to run multiple times
# NOTE: Heavy operations (rsync, build) are done by workers, not init

set -e

PROJECT_DIR="/Users/larkinwc/personal/ml/mi50grad"

echo "=== Verifying local project structure ==="
cd "$PROJECT_DIR"
mkdir -p build/kernels

echo "=== Init complete ==="
