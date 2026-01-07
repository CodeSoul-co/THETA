#!/bin/bash
# Unsupervised Training Script (Autoregressive LM + Cross-Entropy)
# Datasets: germanCoal -> FCPB (ordered by size)

set -e

source activate jiqun
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/embedding

# Create log directory
mkdir -p logs

echo "=========================================="
echo "Unsupervised Training (Autoregressive LM)"
echo "Time: $(date)"
echo "=========================================="

# Training script
python -c "
import sys
sys.path.insert(0, '/root/autodl-tmp/embedding')

import torch
from data_loader import DatasetLoader
from trainer_v2 import EmbeddingTrainerV2, TrainingConfig

# Datasets ordered by size (smallest first)
DATASETS = [
    ('germanCoal', 10),   # 9136 samples, 10 epochs
    ('FCPB', 5),          # 208955 samples, 5 epochs
]

loader = DatasetLoader(dev_mode=False)

for dataset_name, epochs in DATASETS:
    print(f'\n{\"=\"*70}')
    print(f'Processing: {dataset_name}')
    print(f'{\"=\"*70}')
    
    # Load data
    texts, labels, info = loader.load_dataset(dataset_name)
    
    print(f'Samples: {len(texts)}')
    
    # Initialize trainer
    config = TrainingConfig(
        model_path='/root/autodl-tmp/qwen3_embedding_0.6B',
        max_length=256,
        epochs=epochs,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        use_lora=True
    )
    
    trainer = EmbeddingTrainerV2(config=config, dev_mode=False)
    
    # Train
    result = trainer.train_unsupervised(texts, dataset_name)
    
    print(f'Final Loss: {result[\"final_loss\"]:.4f}')
    print(f'Final Perplexity: {result[\"final_perplexity\"]:.2f}')
    
    print(f'Completed: {dataset_name}')
    
    # Clear memory
    del trainer
    torch.cuda.empty_cache()

print('\n' + '='*70)
print('All unsupervised training completed!')
print('='*70)
" 2>&1 | tee logs/unsupervised_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed at: $(date)"
