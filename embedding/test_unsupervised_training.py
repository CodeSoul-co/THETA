"""
Test Unsupervised Training with germanCoal (smallest unlabeled dataset)
Verify code works correctly before running on larger datasets
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/embedding')

from data_loader import DatasetLoader
from trainer import EmbeddingTrainer, TrainingConfig

def main():
    print("="*70)
    print("Test Unsupervised Training - germanCoal")
    print("="*70)
    
    # Use small sample for testing
    MAX_SAMPLES = 500  # Small sample for quick test
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    loader = DatasetLoader(dev_mode=True)
    texts, labels, info = loader.load_dataset(
        "germanCoal",
        max_samples=MAX_SAMPLES,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Loaded {len(texts)} samples")
    print(f"Has labels: {labels is not None}")
    print(f"Sample text: {texts[0][:100]}...")
    
    # Step 2: Initialize trainer
    print("\n[Step 2] Initializing trainer...")
    config = TrainingConfig(
        model_path="/root/autodl-tmp/qwen3_embedding_0.6B",
        max_length=256,  # Shorter for faster testing
        epochs=2,        # Fewer epochs for testing
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        use_lora=True,
        output_dir="/root/autodl-tmp/embedding/outputs",
        checkpoint_dir="/root/autodl-tmp/embedding/checkpoints",
        result_dir="/root/autodl-tmp/result"
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=True)
    
    # Step 3: Train
    print("\n[Step 3] Training (SimCSE)...")
    train_result = trainer.train_unsupervised(texts, "germanCoal_test")
    
    print(f"\nTraining completed!")
    print(f"Final loss: {train_result['final_loss']:.4f}")
    print(f"History: {train_result['history']}")
    
    # Step 4: Generate embeddings
    print("\n[Step 4] Generating embeddings with trained model...")
    embeddings = trainer.generate_embeddings(texts, batch_size=16)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
    
    # Step 5: Save embeddings
    print("\n[Step 5] Saving embeddings...")
    saved_paths = trainer.save_embeddings(
        embeddings, "germanCoal_test", "unsupervised", labels
    )
    
    print(f"\nSaved paths:")
    for key, path in saved_paths.items():
        if path:
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            print(f"  {key}: {path} (exists={exists}, size={size/1024:.1f}KB)")
    
    # Step 6: Verify saved files
    print("\n[Step 6] Verifying saved files...")
    
    # Load and verify embeddings
    emb_path = saved_paths['embeddings']
    loaded_emb = np.load(emb_path)
    print(f"Loaded embeddings shape: {loaded_emb.shape}")
    assert loaded_emb.shape == embeddings.shape, "Shape mismatch!"
    assert np.allclose(loaded_emb, embeddings), "Values mismatch!"
    print("✓ Embeddings verified")
    
    # Check checkpoint
    print("\n[Step 7] Checking checkpoint...")
    ckpt_dir = os.path.join(config.checkpoint_dir, "germanCoal_test")
    if os.path.exists(ckpt_dir):
        ckpt_files = os.listdir(ckpt_dir)
        print(f"Checkpoint directory: {ckpt_dir}")
        print(f"Files: {ckpt_files}")
    else:
        print(f"[WARNING] Checkpoint directory not found: {ckpt_dir}")
    
    print("\n" + "="*70)
    print("✓ Unsupervised Training Test PASSED!")
    print("="*70)

if __name__ == "__main__":
    main()
