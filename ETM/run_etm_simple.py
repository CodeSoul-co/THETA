#!/usr/bin/env python
"""
Simple ETM Pipeline Script

This script runs the complete ETM pipeline for a single dataset:
1. Generate BOW matrix from text data
2. Generate vocabulary embeddings (rho) using Qwen
3. Train ETM model with Qwen document embeddings and BOW targets
4. Save results (theta, beta, topic words)

Usage:
    python run_etm_simple.py --dataset <dataset_name> --embedding_mode <mode>
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import sparse
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from engine_a.vocab_builder import VocabBuilder, VocabConfig
from engine_a.bow_generator import BOWGenerator
from engine_c.vocab_embedder import VocabEmbedder, generate_vocab_embeddings
from engine_c.etm import ETM
from data.dataloader import ETMDataset, ETMDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[str]:
    """
    Load dataset from file.
    
    Args:
        dataset_path: Path to dataset file (JSON or TXT)
        
    Returns:
        List of text documents
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            if isinstance(data[0], str):
                texts = data
            elif isinstance(data[0], dict) and 'text' in data[0]:
                texts = [item['text'] for item in data]
            else:
                raise ValueError("Unsupported JSON format")
        else:
            raise ValueError("JSON file must contain a list")
            
    elif dataset_path.endswith('.txt'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    else:
        raise ValueError("Unsupported file format")
    
    logger.info(f"Loaded {len(texts)} documents")
    return texts


def generate_bow(
    texts: List[str],
    dataset_name: str,
    output_dir: str,
    min_df: int = 5,
    max_df_ratio: float = 0.7,
    vocab_size: int = 10000,
    dev_mode: bool = False
) -> Tuple[sparse.csr_matrix, Dict[str, int], str]:
    """
    Generate BOW matrix and vocabulary.
    
    Args:
        texts: List of text documents
        dataset_name: Name of the dataset
        output_dir: Output directory
        min_df: Minimum document frequency
        max_df_ratio: Maximum document frequency ratio
        vocab_size: Maximum vocabulary size
        dev_mode: Print debug information
        
    Returns:
        (bow_matrix, word2idx, vocab_path)
    """
    logger.info(f"Generating BOW for {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create vocabulary config
    vocab_config = VocabConfig(
        min_df=min_df,
        max_df_ratio=max_df_ratio,
        max_vocab_size=vocab_size,
        lowercase=True,
        remove_stopwords=True
    )
    
    # Build vocabulary
    vocab_builder = VocabBuilder(config=vocab_config, dev_mode=dev_mode)
    vocab_builder.add_texts(texts, dataset_name=dataset_name)
    vocab_builder.build_vocab()
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, f"{dataset_name}_vocab.json")
    vocab_builder.save_vocab(vocab_path)
    
    # Get word2idx
    word2idx = vocab_builder.get_word2idx()
    
    # Generate BOW matrix
    bow_generator = BOWGenerator(vocab_builder=vocab_builder, dev_mode=dev_mode)
    bow_output = bow_generator.generate_bow(
        texts=texts,
        dataset_name=dataset_name,
        show_progress=True
    )
    
    # Save BOW matrix
    bow_generator.save_bow(bow_output, output_dir)
    
    logger.info(f"Generated BOW matrix: {bow_output.bow_matrix.shape}")
    logger.info(f"Vocabulary size: {len(word2idx)}")
    
    return bow_output.bow_matrix, word2idx, vocab_path


def load_document_embeddings(
    embedding_path: str,
    dataset_name: str,
    embedding_mode: str
) -> np.ndarray:
    """
    Load document embeddings.
    
    Args:
        embedding_path: Path to embedding directory
        dataset_name: Name of the dataset
        embedding_mode: Embedding mode (zero_shot, supervised, unsupervised)
        
    Returns:
        Document embeddings matrix
    """
    # Construct embedding file path
    file_path = os.path.join(
        embedding_path, 
        embedding_mode, 
        f"{dataset_name}_{embedding_mode}_embeddings.npy"
    )
    
    logger.info(f"Loading document embeddings from {file_path}")
    
    # Load embeddings
    embeddings = np.load(file_path)
    
    logger.info(f"Loaded document embeddings: {embeddings.shape}")
    
    return embeddings


def train_etm_model(
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    vocab_size: int,
    num_topics: int = 50,
    hidden_dim: int = 512,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 0.002,
    weight_decay: float = 1e-5,
    kl_weight_start: float = 0.1,
    kl_weight_end: float = 1.0,
    kl_weight_epochs: int = 30,
    train_word_embeddings: bool = False,
    device: Optional[str] = None,
    dev_mode: bool = False
) -> Tuple[ETM, Dict]:
    """
    Train ETM model.
    
    Args:
        doc_embeddings: Document embeddings
        bow_matrix: BOW matrix
        vocab_embeddings: Vocabulary embeddings (rho)
        vocab_size: Size of vocabulary
        num_topics: Number of topics
        hidden_dim: Hidden dimension for encoder
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        kl_weight_start: Initial KL weight
        kl_weight_end: Final KL weight
        kl_weight_epochs: Epochs to anneal KL weight
        train_word_embeddings: Whether to fine-tune word embeddings
        device: Device to use (cuda or cpu)
        dev_mode: Print debug information
        
    Returns:
        (trained_model, training_history)
    """
    logger.info(f"Training ETM model with {num_topics} topics")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = ETMDataset(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix
    )
    
    # Split into train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    
    # Convert vocab embeddings to tensor
    vocab_embeddings_tensor = torch.tensor(
        vocab_embeddings, dtype=torch.float32
    ).to(device)
    
    # Create model
    model = ETM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        doc_embedding_dim=doc_embeddings.shape[1],
        word_embedding_dim=vocab_embeddings.shape[1],
        hidden_dim=hidden_dim,
        word_embeddings=vocab_embeddings_tensor,
        train_word_embeddings=train_word_embeddings,
        dev_mode=dev_mode
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        # Calculate KL weight for annealing
        if epoch < kl_weight_epochs:
            kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * \
                        (epoch / kl_weight_epochs)
        else:
            kl_weight = kl_weight_end
        
        for batch_idx, (doc_emb, bow) in enumerate(train_loader):
            # Move to device
            doc_emb = doc_emb.to(device)
            bow = bow.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(doc_emb, bow, compute_loss=True, kl_weight=kl_weight)
            
            # Backward pass
            loss = output['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            
            # Update parameters
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            train_recon_loss += output['recon_loss'].item()
            train_kl_loss += output['kl_loss'].item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for doc_emb, bow in val_loader:
                # Move to device
                doc_emb = doc_emb.to(device)
                bow = bow.to(device)
                
                # Forward pass
                output = model(doc_emb, bow, compute_loss=True, kl_weight=kl_weight)
                
                # Track losses
                val_loss += output['total_loss'].item()
                val_recon_loss += output['recon_loss'].item()
                val_kl_loss += output['kl_loss'].item()
        
        # Average losses
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(val_recon_loss)
        history['kl_loss'].append(val_kl_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f} | "
                   f"KL Weight: {kl_weight:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test model
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    
    with torch.no_grad():
        for doc_emb, bow in test_loader:
            # Move to device
            doc_emb = doc_emb.to(device)
            bow = bow.to(device)
            
            # Forward pass
            output = model(doc_emb, bow, compute_loss=True)
            
            # Track losses
            test_loss += output['total_loss'].item()
            test_recon_loss += output['recon_loss'].item()
            test_kl_loss += output['kl_loss'].item()
    
    # Average losses
    test_loss /= len(test_loader)
    test_recon_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)
    
    logger.info(f"Test Loss: {test_loss:.4f} | "
               f"Recon Loss: {test_recon_loss:.4f} | "
               f"KL Loss: {test_kl_loss:.4f}")
    
    # Add test metrics to history
    history['test_loss'] = test_loss
    history['test_recon_loss'] = test_recon_loss
    history['test_kl_loss'] = test_kl_loss
    history['best_val_loss'] = best_val_loss
    
    return model, history


def save_results(
    model: ETM,
    doc_embeddings: np.ndarray,
    vocab_list: List[str],
    output_dir: str,
    dataset_name: str,
    embedding_mode: str,
    history: Dict,
    device: torch.device
) -> Dict[str, str]:
    """
    Save model results.
    
    Args:
        model: Trained ETM model
        doc_embeddings: Document embeddings
        vocab_list: Vocabulary list
        output_dir: Output directory
        dataset_name: Name of the dataset
        embedding_mode: Embedding mode
        history: Training history
        device: Device
        
    Returns:
        Dictionary with saved file paths
    """
    logger.info("Saving model results")
    
    # Create output directory
    results_dir = os.path.join(output_dir, "results", dataset_name, embedding_mode)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get all outputs
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        doc_emb_tensor = torch.tensor(doc_embeddings, dtype=torch.float32).to(device)
        
        # Get theta, beta, topic_embeddings
        theta = model.get_theta(doc_emb_tensor).cpu().numpy()
        beta = model.get_beta().cpu().numpy()
        topic_embeddings = model.get_topic_embeddings().cpu().numpy()
        
        # Get topic words
        topic_words = model.get_topic_words(top_k=20, vocab=vocab_list)
    
    # Save theta matrix
    theta_path = os.path.join(results_dir, f"theta_{timestamp}.npy")
    np.save(theta_path, theta)
    
    # Save beta matrix
    beta_path = os.path.join(results_dir, f"beta_{timestamp}.npy")
    np.save(beta_path, beta)
    
    # Save topic embeddings
    topic_emb_path = os.path.join(results_dir, f"topic_embeddings_{timestamp}.npy")
    np.save(topic_emb_path, topic_embeddings)
    
    # Save topic words
    topic_words_path = os.path.join(results_dir, f"topic_words_{timestamp}.json")
    with open(topic_words_path, 'w') as f:
        json.dump([(k, [(w, float(p)) for w, p in words]) for k, words in topic_words], f, indent=2)
    
    # Save model
    model_path = os.path.join(results_dir, f"model_{timestamp}.pt")
    model.save_model(model_path)
    
    # Save training history
    history_path = os.path.join(results_dir, f"history_{timestamp}.json")
    with open(history_path, 'w') as f:
        # Convert numpy values to float
        history_json = {k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                        for k, vals in history.items()}
        json.dump(history_json, f, indent=2)
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'embedding_mode': embedding_mode,
        'num_docs': len(doc_embeddings),
        'vocab_size': len(vocab_list),
        'num_topics': model.num_topics,
        'timestamp': timestamp,
        'files': {
            'theta': os.path.basename(theta_path),
            'beta': os.path.basename(beta_path),
            'topic_embeddings': os.path.basename(topic_emb_path),
            'topic_words': os.path.basename(topic_words_path),
            'model': os.path.basename(model_path),
            'history': os.path.basename(history_path)
        }
    }
    
    metadata_path = os.path.join(results_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    
    return {
        'theta': theta_path,
        'beta': beta_path,
        'topic_embeddings': topic_emb_path,
        'topic_words': topic_words_path,
        'model': model_path,
        'history': history_path,
        'metadata': metadata_path
    }


def main():
    parser = argparse.ArgumentParser(description="Run ETM pipeline")
    
    # Dataset and embedding mode
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file")
    parser.add_argument("--embedding_mode", type=str, default="zero_shot",
                        choices=["zero_shot", "supervised", "unsupervised"],
                        help="Embedding mode")
    
    # Paths
    parser.add_argument("--embedding_dir", type=str, 
                        default="/root/autodl-tmp/embedding/outputs",
                        help="Directory containing document embeddings")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/autodl-tmp/ETM/outputs",
                        help="Output directory")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/qwen3_embedding_0.6B",
                        help="Path to Qwen model")
    
    # BOW generation
    parser.add_argument("--min_df", type=int, default=5,
                        help="Minimum document frequency")
    parser.add_argument("--max_df_ratio", type=float, default=0.7,
                        help="Maximum document frequency ratio")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Maximum vocabulary size")
    
    # ETM model parameters
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
    
    # Other options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--dev_mode", action="store_true",
                        help="Print debug information")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create output directories
    bow_dir = os.path.join(args.output_dir, "engine_a")
    vocab_emb_dir = os.path.join(args.output_dir, "engine_c")
    os.makedirs(bow_dir, exist_ok=True)
    os.makedirs(vocab_emb_dir, exist_ok=True)
    
    # Step 1: Load dataset
    texts = load_dataset(args.dataset_path)
    
    # Step 2: Generate BOW matrix
    bow_matrix, word2idx, vocab_path = generate_bow(
        texts=texts,
        dataset_name=args.dataset,
        output_dir=bow_dir,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio,
        vocab_size=args.vocab_size,
        dev_mode=args.dev_mode
    )
    
    # Step 3: Generate vocabulary embeddings (rho)
    vocab_emb_path = os.path.join(vocab_emb_dir, f"{args.dataset}_vocab_embeddings.npy")
    
    # Convert word2idx to vocab_list
    vocab_list = [''] * len(word2idx)
    for word, idx in word2idx.items():
        vocab_list[idx] = word
    
    # Generate embeddings if not exists
    if not os.path.exists(vocab_emb_path):
        logger.info("Generating vocabulary embeddings")
        
        # Create embedder
        embedder = VocabEmbedder(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            dev_mode=args.dev_mode
        )
        
        # Generate embeddings
        vocab_embeddings = embedder.embed_vocab(
            vocab_list=vocab_list,
            output_path=vocab_emb_path,
            show_progress=True
        )
    else:
        logger.info(f"Loading vocabulary embeddings from {vocab_emb_path}")
        vocab_embeddings = np.load(vocab_emb_path)
    
    logger.info(f"Vocabulary embeddings shape: {vocab_embeddings.shape}")
    
    # Step 4: Load document embeddings
    doc_embeddings = load_document_embeddings(
        embedding_path=args.embedding_dir,
        dataset_name=args.dataset,
        embedding_mode=args.embedding_mode
    )
    
    # Step 5: Train ETM model
    model, history = train_etm_model(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix,
        vocab_embeddings=vocab_embeddings,
        vocab_size=len(word2idx),
        num_topics=args.num_topics,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        kl_weight_start=args.kl_weight_start,
        kl_weight_end=args.kl_weight_end,
        kl_weight_epochs=args.kl_weight_epochs,
        train_word_embeddings=args.train_word_embeddings,
        device=device,
        dev_mode=args.dev_mode
    )
    
    # Step 6: Save results
    save_results(
        model=model,
        doc_embeddings=doc_embeddings,
        vocab_list=vocab_list,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        embedding_mode=args.embedding_mode,
        history=history,
        device=device
    )
    
    logger.info("ETM pipeline completed successfully!")


if __name__ == "__main__":
    main()
