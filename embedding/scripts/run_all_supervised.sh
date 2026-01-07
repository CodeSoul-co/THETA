#!/bin/bash
# Supervised Training (MLP + Cross-Entropy) - All labeled datasets
# Order: socialTwitter (39659) -> hatespeech (436725) -> mental_health (1023524)

set -e

source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding
mkdir -p logs

echo "=========================================="
echo "Supervised Training - All Labeled Datasets"
echo "Time: $(date)"
echo "=========================================="

python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/embedding')

import numpy as np
import torch
from data_loader import DatasetLoader
from trainer_v2 import EmbeddingTrainerV2, TrainingConfig

# Datasets ordered by size (smallest first)
DATASETS = [
    ('socialTwitter', 10),   # 39659 samples, 10 epochs
    ('hatespeech', 5),       # 436725 samples, 5 epochs
    ('mental_health', 3),    # 1023524 samples, 3 epochs
]

loader = DatasetLoader(dev_mode=False)

for dataset_name, epochs in DATASETS:
    print(f'\n{\"=\"*70}')
    print(f'Supervised Training: {dataset_name}')
    print(f'{\"=\"*70}')
    
    # Load data
    texts, labels, info = loader.load_dataset(dataset_name)
    
    if labels is None:
        print(f'[SKIP] {dataset_name} has no labels')
        continue
    
    # Convert labels to numeric
    if not np.issubdtype(labels.dtype, np.number):
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
    
    num_classes = len(np.unique(labels))
    print(f'Samples: {len(texts)}')
    print(f'Classes: {num_classes}')
    
    # Initialize trainer
    config = TrainingConfig(
        model_path='/root/autodl-tmp/qwen3_embedding_0.6B',
        max_length=256,
        epochs=epochs,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        classifier_lr=1e-3,
        use_lora=True
    )
    
    trainer = EmbeddingTrainerV2(config=config, dev_mode=False)
    
    # Train
    result = trainer.train_supervised(texts, labels, dataset_name)
    
    print(f'Final Loss: {result[\"final_loss\"]:.4f}')
    print(f'Final Accuracy: {result[\"final_accuracy\"]:.4f}')
    
    # Generate and save embeddings
    print('Generating embeddings...')
    embeddings = trainer.generate_embeddings(texts, batch_size=16)
    trainer.save_embeddings(embeddings, dataset_name, 'supervised', labels)
    
    print(f'{dataset_name} completed!')
    
    # Clear memory
    del trainer, embeddings
    torch.cuda.empty_cache()

print('\n' + '='*70)
print('All supervised training completed!')
print('Time: $(date)')
print('='*70)
" 2>&1 | tee logs/supervised_all_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed at: $(date)"
