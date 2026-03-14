#!/bin/bash
# Environment setup for TP=4 optimization mission
# Idempotent - safe to run multiple times

set -euo pipefail

# Verify SSH connectivity to dev server
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes root@192.168.1.198 "echo ok" &>/dev/null; then
    echo "WARNING: Cannot connect to dev server root@192.168.1.198"
    echo "SSH key auth must be configured for this mission to work"
fi

# Ensure deploy directory exists on remote
ssh -o ConnectTimeout=5 root@192.168.1.198 "mkdir -p /opt/mi50grad/build/kernels" 2>/dev/null || true

echo "Init complete"
