#!/bin/bash
# Download AWQ-quantized Qwen3.5-27B model from HuggingFace to dev server
#
# Model: QuantTrio/Qwen3.5-27B-AWQ
# URL: https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ
# Size: ~21GB (8 safetensors files)
#
# Usage: ./scripts/download_awq_model.sh
# The model will be downloaded to /opt/models/Qwen3.5-27B-AWQ on the dev server

set -euo pipefail

DEV_SERVER="root@192.168.1.198"
REMOTE_DIR="/opt/models"
MODEL_NAME="Qwen3.5-27B-AWQ"
HF_REPO="QuantTrio/Qwen3.5-27B-AWQ"
TARGET_PATH="$REMOTE_DIR/$MODEL_NAME"

echo "=== Download AWQ Model ==="
echo "Model: $HF_REPO"
echo "Destination: $DEV_SERVER:$TARGET_PATH"
echo ""

# Check if huggingface-hub is installed on dev server, if not install it
echo "[1/4] Checking huggingface-hub installation..."
ssh "$DEV_SERVER" "pip3 show huggingface-hub > /dev/null 2>&1 || pip3 install huggingface-hub -q"

# Create directory
echo "[2/4] Creating directory $TARGET_PATH..."
ssh "$DEV_SERVER" "mkdir -p $TARGET_PATH"

# Download the model using huggingface-cli
echo "[3/4] Downloading model (this may take 10-30 minutes depending on network)..."
ssh "$DEV_SERVER" "cd $TARGET_PATH && \
    python3 -c \"from huggingface_hub import snapshot_download; \
    snapshot_download(
        repo_id='$HF_REPO',
        local_dir='$TARGET_PATH',
        local_dir_use_symlinks=False,
        ignore_patterns=['*.git*', '*.md', '*.jpg', '*.png']
    )\""

# Verify download
echo "[4/4] Verifying download..."
ssh "$DEV_SERVER" "ls -lh $TARGET_PATH/*.safetensors 2>/dev/null | head -10 && \
    echo '' && \
    echo 'Download complete!' && \
    echo 'Model files:' && \
    ls -1 $TARGET_PATH/*.safetensors 2>/dev/null | wc -l && \
    echo 'Total size:' && \
    du -sh $TARGET_PATH"

echo ""
echo "Model downloaded successfully to $TARGET_PATH"
echo ""
echo "To verify the model format:"
echo "  ssh $DEV_SERVER 'cd /opt/mi50grad && python3 -c \"from src.model.awq_loader import detect_awq_format; print(detect_awq_format(\\\"$TARGET_PATH\\\"))\"'"
