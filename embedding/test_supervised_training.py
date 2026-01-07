"""
Test Supervised Training with socialTwitter (smallest labeled dataset)
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
    print("Test Supervised Training - socialTwitter")
    print("="*70)
    
    # Use small sample for testing
    MAX_SAMPLES = 500  # Small sample for quick test
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    loader = DatasetLoader(dev_mode=True)
    texts, labels, info = loader.load_dataset(
        "socialTwitter",
        max_samples=MAX_SAMPLES,
        shuffle=True,
        random_seed=42
    )
    
    print(f"Loaded {len(texts)} samples")
    print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
    
    if labels is None:
        print("[ERROR] No labels found!")
        return
    
    # Convert labels to numeric if needed
    if not np.issubdtype(labels.dtype, np.number):
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        print(f"Converted labels to numeric: {len(unique_labels)} unique labels")
    
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
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
    print("\n[Step 3] Training...")
    train_result = trainer.train_supervised(texts, labels, "socialTwitter_test")
    
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
        embeddings, "socialTwitter_test", "supervised", labels
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
    
    # Load and verify labels
    label_path = saved_paths['labels']
    if label_path:
        loaded_labels = np.load(label_path)
        print(f"Loaded labels shape: {loaded_labels.shape}")
        assert np.array_equal(loaded_labels, labels), "Labels mismatch!"
        print("✓ Labels verified")
    
    # Check checkpoint
    print("\n[Step 7] Checking checkpoint...")
    ckpt_dir = os.path.join(config.checkpoint_dir, "socialTwitter_test")
    if os.path.exists(ckpt_dir):
        ckpt_files = os.listdir(ckpt_dir)
        print(f"Checkpoint directory: {ckpt_dir}")
        print(f"Files: {ckpt_files}")
    else:
        print(f"[WARNING] Checkpoint directory not found: {ckpt_dir}")
    
    print("\n" + "="*70)
    print("✓ Supervised Training Test PASSED!")
    print("="*70)

if __name__ == "__main__":
    main()
