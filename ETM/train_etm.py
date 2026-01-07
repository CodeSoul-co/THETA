#!/usr/bin/env python
"""
Train ETM model with Qwen embeddings.

This script trains an ETM model using document embeddings from Qwen
and BOW matrices from Engine A.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import sparse
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from engine_c.etm import ETM
from data.dataloader import ETMDataLoader, create_dataloaders
from trainer.trainer import ETMTrainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser(description="Train ETM model")
    
    # Data paths
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., 'hatespeech', 'mental_health')")
    parser.add_argument("--embedding_mode", type=str, default="zero_shot",
                        choices=["zero_shot", "supervised", "unsupervised"],
                        help="Embedding mode to use")
    parser.add_argument("--embedding_dir", type=str, 
                        default="/root/autodl-tmp/embedding/outputs",
                        help="Directory containing document embeddings")
    parser.add_argument("--bow_dir", type=str, 
                        default="/root/autodl-tmp/ETM/outputs/engine_a",
                        help="Directory containing BOW matrices")
    parser.add_argument("--vocab_embeddings", type=str,
                        default="/root/autodl-tmp/ETM/outputs/engine_c/vocab_embeddings.npy",
                        help="Path to vocabulary embeddings")
    
    # Model parameters
    parser.add_argument("--num_topics", type=int, default=50,
                        help="Number of topics")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for encoder")
    parser.add_argument("--train_word_embeddings", action="store_true",
                        help="Whether to train word embeddings")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--kl_weight_start", type=float, default=0.1,
                        help="Initial KL weight")
    parser.add_argument("--kl_weight_end", type=float, default=1.0,
                        help="Final KL weight")
    parser.add_argument("--kl_weight_epochs", type=int, default=30,
                        help="Epochs to anneal KL weight")
    
    # Output paths
    parser.add_argument("--output_dir", type=str, 
                        default="/root/autodl-tmp/ETM/outputs/engine_c",
                        help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="/root/autodl-tmp/ETM/checkpoints",
                        help="Checkpoint directory")
    
    # Other options
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dev_mode", action="store_true",
                        help="Print debug information")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data for dataset: {args.dataset}")
    data_loader = ETMDataLoader(dev_mode=args.dev_mode)
    
    # Load document embeddings
    doc_embeddings, labels, emb_meta = data_loader.load_embeddings(
        dataset_name=args.dataset,
        mode=args.embedding_mode
    )
    
    # Load BOW matrix
    bow_matrix, bow_meta = data_loader.load_bow(
        dataset_name=args.dataset
    )
    
    # Load vocabulary
    word2idx, vocab_list = data_loader.load_vocab()
    
    logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
    logger.info(f"BOW matrix shape: {bow_matrix.shape}")
    logger.info(f"Vocabulary size: {len(vocab_list)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix,
        labels=labels,
        batch_size=args.batch_size,
        shuffle=True,
        random_seed=args.seed,
        dev_mode=args.dev_mode
    )
    
    logger.info(f"Created data loaders: train={len(train_loader.dataset)}, "
                f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    
    # Load vocabulary embeddings
    if os.path.exists(args.vocab_embeddings):
        logger.info(f"Loading vocabulary embeddings from {args.vocab_embeddings}")
        vocab_embeddings = torch.tensor(np.load(args.vocab_embeddings), dtype=torch.float32)
        
        # Check dimensions
        if vocab_embeddings.shape[0] != len(vocab_list):
            logger.warning(f"Vocabulary embedding size mismatch: {vocab_embeddings.shape[0]} vs {len(vocab_list)}")
            logger.warning("Using random initialization instead")
            vocab_embeddings = None
    else:
        logger.warning(f"Vocabulary embeddings not found at {args.vocab_embeddings}")
        logger.warning("Using random initialization")
        vocab_embeddings = None
    
    # Create model
    logger.info(f"Creating ETM model with {args.num_topics} topics")
    model = ETM(
        vocab_size=len(vocab_list),
        num_topics=args.num_topics,
        doc_embedding_dim=doc_embeddings.shape[1],
        word_embedding_dim=1024,  # Qwen embedding dimension
        hidden_dim=args.hidden_dim,
        word_embeddings=vocab_embeddings,
        train_word_embeddings=args.train_word_embeddings,
        kl_weight=args.kl_weight_start,
        dev_mode=args.dev_mode
    )
    
    # Create trainer config
    trainer_config = TrainerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        kl_weight_start=args.kl_weight_start,
        kl_weight_end=args.kl_weight_end,
        kl_weight_epochs=args.kl_weight_epochs
    )
    
    # Create trainer
    trainer = ETMTrainer(
        model=model,
        config=trainer_config,
        device=device,
        result_dir=os.path.join(args.output_dir, "results"),
        checkpoint_dir=args.checkpoint_dir,
        dev_mode=args.dev_mode
    )
    
    # Train model
    logger.info("Starting training")
    train_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=args.dataset
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = trainer.evaluate(test_loader)
    
    logger.info(f"Test metrics: loss={test_metrics['loss']:.4f}, "
                f"recon_loss={test_metrics['recon_loss']:.4f}, "
                f"kl_loss={test_metrics['kl_loss']:.4f}")
    
    # Calculate perplexity
    perplexity = model.compute_perplexity(
        doc_embeddings=torch.tensor(doc_embeddings, dtype=torch.float32).to(device),
        bow_targets=torch.tensor(bow_matrix.toarray(), dtype=torch.float32).to(device)
    )
    
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    # Get document-topic distributions
    logger.info("Generating document-topic distributions")
    theta = trainer.get_document_topics(test_loader)
    
    # Get topic-word distributions
    logger.info("Generating topic-word distributions")
    beta = model.get_beta().cpu().numpy()
    
    # Get topic words
    logger.info("Generating topic words")
    topic_words = model.get_topic_words(top_k=20, vocab=vocab_list)
    
    # Save results
    results_dir = os.path.join(args.output_dir, "results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save theta matrix
    theta_path = os.path.join(results_dir, f"theta_{timestamp}.npy")
    np.save(theta_path, theta)
    
    # Save beta matrix
    beta_path = os.path.join(results_dir, f"beta_{timestamp}.npy")
    np.save(beta_path, beta)
    
    # Save topic words
    topic_words_path = os.path.join(results_dir, f"topic_words_{timestamp}.json")
    with open(topic_words_path, 'w') as f:
        json.dump([(k, [(w, float(p)) for w, p in words]) for k, words in topic_words], f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
            "perplexity": perplexity,
            "train_history": train_results["history"][-10:],  # Last 10 epochs
            "best_val_loss": train_results["best_val_loss"],
            "final_epoch": train_results["final_epoch"],
            "total_time": train_results["total_time"]
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    logger.info(f"Theta matrix: {theta_path}")
    logger.info(f"Beta matrix: {beta_path}")
    logger.info(f"Topic words: {topic_words_path}")
    logger.info(f"Metrics: {metrics_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
