"""
Test TrainerV2 with redesigned training methods:
1. Supervised: MLP classifier + Cross-Entropy loss
2. Unsupervised: Autoregressive LM + Cross-Entropy loss
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/embedding')

from data_loader import DatasetLoader
from trainer_v2 import EmbeddingTrainerV2, TrainingConfig

def test_supervised():
    """Test supervised training with socialTwitter (smallest labeled)"""
    print("="*70)
    print("Test Supervised Training (MLP + Cross-Entropy)")
    print("Dataset: socialTwitter (smallest labeled)")
    print("="*70)
    
    MAX_SAMPLES = 500
    
    # Load data
    print("\n[Step 1] Loading data...")
    loader = DatasetLoader(dev_mode=True)
    texts, labels, info = loader.load_dataset(
        "socialTwitter",
        max_samples=MAX_SAMPLES,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Loaded {len(texts)} samples")
    
    # Convert labels to numeric
    if not np.issubdtype(labels.dtype, np.number):
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        print(f"Classes: {len(unique_labels)} -> {unique_labels}")
    
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Initialize trainer
    print("\n[Step 2] Initializing trainer...")
    config = TrainingConfig(
        model_path="/root/autodl-tmp/qwen3_embedding_0.6B",
        max_length=256,
        epochs=5,  # More epochs
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        classifier_lr=1e-3,
        use_lora=True
    )
    
    trainer = EmbeddingTrainerV2(config=config, dev_mode=True)
    
    # Train
    print("\n[Step 3] Training...")
    result = trainer.train_supervised(texts, labels, "socialTwitter_test_v2")
    
    print(f"\n--- Training Results ---")
    print(f"Final Loss: {result['final_loss']:.4f}")
    print(f"Final Accuracy: {result['final_accuracy']:.4f}")
    
    # Generate embeddings
    print("\n[Step 4] Generating embeddings...")
    embeddings = trainer.generate_embeddings(texts, batch_size=16)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    print("\n[Step 5] Saving embeddings...")
    paths = trainer.save_embeddings(embeddings, "socialTwitter_test_v2", "supervised", labels)
    
    print("\n" + "="*70)
    print("✓ Supervised Training Test PASSED!")
    print("="*70)
    
    return result


def test_unsupervised():
    """Test unsupervised training with germanCoal (smallest unlabeled)"""
    print("\n" + "="*70)
    print("Test Unsupervised Training (Autoregressive LM)")
    print("Dataset: germanCoal (smallest unlabeled)")
    print("="*70)
    
    MAX_SAMPLES = 500
    
    # Load data
    print("\n[Step 1] Loading data...")
    loader = DatasetLoader(dev_mode=True)
    texts, labels, info = loader.load_dataset(
        "germanCoal",
        max_samples=MAX_SAMPLES,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Loaded {len(texts)} samples")
    print(f"Sample: {texts[0][:100]}...")
    
    # Initialize trainer
    print("\n[Step 2] Initializing trainer...")
    config = TrainingConfig(
        model_path="/root/autodl-tmp/qwen3_embedding_0.6B",
        max_length=256,
        epochs=5,  # More epochs
        batch_size=4,  # Smaller for causal LM
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        use_lora=True
    )
    
    trainer = EmbeddingTrainerV2(config=config, dev_mode=True)
    
    # Train
    print("\n[Step 3] Training...")
    result = trainer.train_unsupervised(texts, "germanCoal_test_v2")
    
    print(f"\n--- Training Results ---")
    print(f"Final Loss: {result['final_loss']:.4f}")
    print(f"Final Perplexity: {result['final_perplexity']:.2f}")
    
    print("\n" + "="*70)
    print("✓ Unsupervised Training Test PASSED!")
    print("="*70)
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["supervised", "unsupervised", "both"])
    args = parser.parse_args()
    
    if args.mode in ["supervised", "both"]:
        test_supervised()
    
    if args.mode in ["unsupervised", "both"]:
        test_unsupervised()
