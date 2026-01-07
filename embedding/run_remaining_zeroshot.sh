#!/bin/bash
source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding

echo "=========================================="
echo "Starting remaining zero-shot embeddings"
echo "Time: $(date)"
echo "=========================================="

# hatespeech - 436725 samples (needs re-run, previous was only 100 samples)
echo ""
echo "[1/3] Processing hatespeech (436725 samples)..."
python main.py --mode zero_shot --dataset hatespeech --batch_size 8 --max_length 512 2>&1 | tee logs/zero_shot_hatespeech_full.log

# mental_health - 1023524 samples (largest)
echo ""
echo "[2/3] Processing mental_health (1023524 samples)..."
python main.py --mode zero_shot --dataset mental_health --batch_size 8 --max_length 512 2>&1 | tee logs/zero_shot_mental_health.log

# socialTwitter - 39659 samples
echo ""
echo "[3/3] Processing socialTwitter (39659 samples)..."
python main.py --mode zero_shot --dataset socialTwitter --batch_size 8 --max_length 512 2>&1 | tee logs/zero_shot_socialTwitter.log

echo ""
echo "=========================================="
echo "All zero-shot embeddings completed!"
echo "Time: $(date)"
echo "=========================================="

# List results
ls -lh /root/autodl-tmp/embedding/outputs/zero_shot/
