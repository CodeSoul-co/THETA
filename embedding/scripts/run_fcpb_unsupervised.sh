#!/bin/bash
# FCPB Unsupervised Training (Autoregressive LM)
# Run in detached screen session

set -e

source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding
mkdir -p logs

echo "=========================================="
echo "FCPB Unsupervised Training"
echo "Time: $(date)"
echo "=========================================="

python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/embedding')

import torch
from data_loader import DatasetLoader
from trainer_v2 import EmbeddingTrainerV2, TrainingConfig

print('='*70)
print('Unsupervised Training: FCPB (208955 samples)')
print('='*70)

loader = DatasetLoader(dev_mode=False)
texts, labels, info = loader.load_dataset('FCPB')
print(f'Samples: {len(texts)}')

config = TrainingConfig(
    model_path='/root/autodl-tmp/qwen3_embedding_0.6B',
    max_length=256,
    epochs=5,  # Fewer epochs for large dataset
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    use_lora=True
)

trainer = EmbeddingTrainerV2(config=config, dev_mode=False)
result = trainer.train_unsupervised(texts, 'FCPB')

print(f'Final Loss: {result[\"final_loss\"]:.4f}')
print(f'Final Perplexity: {result[\"final_perplexity\"]:.2f}')

print('='*70)
print('FCPB Training completed!')
print('='*70)
" 2>&1 | tee logs/FCPB_unsupervised_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed at: $(date)"
