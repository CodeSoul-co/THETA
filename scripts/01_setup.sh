#!/bin/bash
# THETA Setup Script
# Install dependencies and download pre-trained data from HuggingFace

set -e

echo "=========================================="
echo "THETA Setup Script"
echo "=========================================="

# Install Python dependencies
echo "[1/3] Installing Python dependencies..."
cd /root/autodl-tmp/ETM
pip install -r requirements.txt -q

# Install openai for agent
echo "[2/3] Installing agent dependencies..."
pip install openai python-dotenv -q

# Download pre-trained data from HuggingFace (if not exists locally)
echo "[3/3] Checking pre-trained data..."
if [ ! -d "/root/autodl-tmp/result/0.6B" ] || [ -z "$(ls -A /root/autodl-tmp/result/0.6B 2>/dev/null)" ]; then
    echo "Downloading pre-trained data from HuggingFace..."
    echo "Repository: https://huggingface.co/CodeSoulco/THETA"
    
    # Install huggingface_hub if not installed
    pip install huggingface_hub -q
    
    # Download using huggingface-cli
    python -c "
from huggingface_hub import snapshot_download
import os

# Download pre-trained embeddings and results
try:
    snapshot_download(
        repo_id='CodeSoulco/THETA',
        local_dir='/root/autodl-tmp/hf_cache/THETA',
        repo_type='model'
    )
    print('Downloaded successfully!')
except Exception as e:
    print(f'Download failed: {e}')
    print('You can manually download from: https://huggingface.co/CodeSoulco/THETA')
"
else
    echo "Pre-trained data already exists, skipping download."
fi

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
