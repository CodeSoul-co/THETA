#!/bin/bash
# Quick test: Run training on smallest datasets only
# - Unsupervised: germanCoal (9136 samples)
# - Supervised: socialTwitter (39659 samples)

set -e

source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding

mkdir -p logs

echo "=========================================="
echo "Training on Small Datasets"
echo "Time: $(date)"
echo "=========================================="

python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/embedding')

import numpy as np
import torch
from data_loader import DatasetLoader
from trainer_v2 import EmbeddingTrainerV2, TrainingConfig

loader = DatasetLoader(dev_mode=False)

# ========== Unsupervised: germanCoal ==========
print('\n' + '='*70)
print('Unsupervised Training: germanCoal (9136 samples)')
print('='*70)

texts, labels, info = loader.load_dataset('germanCoal')
print(f'Samples: {len(texts)}')

config = TrainingConfig(
    model_path='/root/autodl-tmp/qwen3_embedding_0.6B',
    max_length=256,
    epochs=10,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    use_lora=True
)

trainer = EmbeddingTrainerV2(config=config, dev_mode=False)
result = trainer.train_unsupervised(texts, 'germanCoal')

print(f'Final Loss: {result[\"final_loss\"]:.4f}')
print(f'Final Perplexity: {result[\"final_perplexity\"]:.2f}')

del trainer
torch.cuda.empty_cache()

# ========== Supervised: socialTwitter ==========
print('\n' + '='*70)
print('Supervised Training: socialTwitter (39659 samples)')
print('='*70)

texts, labels, info = loader.load_dataset('socialTwitter')

# Convert labels to numeric
if not np.issubdtype(labels.dtype, np.number):
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

print(f'Samples: {len(texts)}')
print(f'Classes: {len(np.unique(labels))}')

config = TrainingConfig(
    model_path='/root/autodl-tmp/qwen3_embedding_0.6B',
    max_length=256,
    epochs=10,
    batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    classifier_lr=1e-3,
    use_lora=True
)

trainer = EmbeddingTrainerV2(config=config, dev_mode=False)
result = trainer.train_supervised(texts, labels, 'socialTwitter')

print(f'Final Loss: {result[\"final_loss\"]:.4f}')
print(f'Final Accuracy: {result[\"final_accuracy\"]:.4f}')

# Generate and save embeddings
print('Generating embeddings...')
embeddings = trainer.generate_embeddings(texts, batch_size=16)
trainer.save_embeddings(embeddings, 'socialTwitter', 'supervised', labels)

print('\n' + '='*70)
print('Small datasets training completed!')
print('='*70)
" 2>&1 | tee logs/small_datasets_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed at: $(date)"
