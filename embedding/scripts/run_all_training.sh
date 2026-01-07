#!/bin/bash
# Run all training (supervised + unsupervised) sequentially
# Use with screen for background execution

set -e

source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding

mkdir -p logs

echo "=========================================="
echo "Full Training Pipeline"
echo "Time: $(date)"
echo "=========================================="

# Run unsupervised first (smaller datasets)
echo ""
echo "[1/2] Running Unsupervised Training..."
bash scripts/run_unsupervised_training.sh

# Run supervised
echo ""
echo "[2/2] Running Supervised Training..."
bash scripts/run_supervised_training.sh

echo ""
echo "=========================================="
echo "All training completed!"
echo "Time: $(date)"
echo "=========================================="
