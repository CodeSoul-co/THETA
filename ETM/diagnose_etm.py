#!/usr/bin/env python
"""
Diagnostic script to identify ETM training issues.
"""

import os
import sys
import torch
import numpy as np
from scipy import sparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def diagnose_bow():
    """Check BOW matrix quality"""
    print("\n" + "=" * 60)
    print("DIAGNOSING BOW MATRIX")
    print("=" * 60)
    
    bow_path = "/root/autodl-tmp/result/socialTwitter/bow/bow_matrix.npz"
    vocab_path = "/root/autodl-tmp/result/socialTwitter/bow/vocab.txt"
    
    if not os.path.exists(bow_path):
        print("BOW matrix not found!")
        return None, None
    
    bow_matrix = sparse.load_npz(bow_path)
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f]
    
    print(f"BOW shape: {bow_matrix.shape}")
    print(f"Vocab size: {len(vocab)}")
    
    # Document length statistics
    doc_lengths = np.array(bow_matrix.sum(axis=1)).flatten()
    print(f"\nDocument length statistics:")
    print(f"  Mean: {doc_lengths.mean():.1f}")
    print(f"  Std: {doc_lengths.std():.1f}")
    print(f"  Min: {doc_lengths.min():.1f}")
    print(f"  Max: {doc_lengths.max():.1f}")
    print(f"  Median: {np.median(doc_lengths):.1f}")
    
    # Sparsity
    sparsity = 1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])
    print(f"\nSparsity: {sparsity:.4f}")
    
    # Top words
    word_counts = np.array(bow_matrix.sum(axis=0)).flatten()
    top_indices = word_counts.argsort()[-20:][::-1]
    print(f"\nTop 20 words:")
    for idx in top_indices:
        print(f"  {vocab[idx]}: {word_counts[idx]:.0f}")
    
    return bow_matrix, vocab


def diagnose_embeddings():
    """Check embedding quality"""
    print("\n" + "=" * 60)
    print("DIAGNOSING EMBEDDINGS")
    print("=" * 60)
    
    doc_emb_path = "/root/autodl-tmp/result/socialTwitter/zero_shot/embeddings/socialTwitter_zero_shot_embeddings.npy"
    vocab_emb_path = "/root/autodl-tmp/result/socialTwitter/bow/vocab_embeddings.npy"
    
    if os.path.exists(doc_emb_path):
        doc_emb = np.load(doc_emb_path)
        print(f"Doc embeddings shape: {doc_emb.shape}")
        print(f"Doc embeddings stats:")
        print(f"  Mean: {doc_emb.mean():.4f}")
        print(f"  Std: {doc_emb.std():.4f}")
        print(f"  Min: {doc_emb.min():.4f}")
        print(f"  Max: {doc_emb.max():.4f}")
        
        # Check if normalized
        norms = np.linalg.norm(doc_emb, axis=1)
        print(f"  L2 norm mean: {norms.mean():.4f}")
        print(f"  L2 norm std: {norms.std():.4f}")
    
    if os.path.exists(vocab_emb_path):
        vocab_emb = np.load(vocab_emb_path)
        print(f"\nVocab embeddings shape: {vocab_emb.shape}")
        print(f"Vocab embeddings stats:")
        print(f"  Mean: {vocab_emb.mean():.4f}")
        print(f"  Std: {vocab_emb.std():.4f}")
        print(f"  Min: {vocab_emb.min():.4f}")
        print(f"  Max: {vocab_emb.max():.4f}")
        
        norms = np.linalg.norm(vocab_emb, axis=1)
        print(f"  L2 norm mean: {norms.mean():.4f}")
        print(f"  L2 norm std: {norms.std():.4f}")
        
        return vocab_emb
    
    return None


def diagnose_model():
    """Check model forward pass"""
    print("\n" + "=" * 60)
    print("DIAGNOSING MODEL FORWARD PASS")
    print("=" * 60)
    
    from model.etm import ETM
    
    # Load data
    bow_matrix = sparse.load_npz("/root/autodl-tmp/result/socialTwitter/bow/bow_matrix.npz")
    doc_emb = np.load("/root/autodl-tmp/result/socialTwitter/zero_shot/embeddings/socialTwitter_zero_shot_embeddings.npy")
    vocab_emb = np.load("/root/autodl-tmp/result/socialTwitter/bow/vocab_embeddings.npy")
    
    # Create model
    vocab_size = bow_matrix.shape[1]
    model = ETM(
        vocab_size=vocab_size,
        num_topics=20,
        doc_embedding_dim=1024,
        word_embedding_dim=1024,
        hidden_dim=512,
        word_embeddings=torch.tensor(vocab_emb, dtype=torch.float32),
        dev_mode=True
    )
    
    # Sample batch
    batch_size = 32
    indices = np.random.choice(len(doc_emb), batch_size, replace=False)
    
    doc_batch = torch.tensor(doc_emb[indices], dtype=torch.float32)
    bow_batch = torch.tensor(bow_matrix[indices].toarray(), dtype=torch.float32)
    
    print(f"Input shapes:")
    print(f"  doc_batch: {doc_batch.shape}")
    print(f"  bow_batch: {bow_batch.shape}")
    print(f"  bow_batch sum per doc: {bow_batch.sum(dim=1).mean():.1f}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(doc_batch, bow_batch, kl_weight=0.5)
    
    print(f"\nOutput shapes:")
    print(f"  theta: {output['theta'].shape}")
    print(f"  beta: {output['beta'].shape}")
    print(f"  word_dist: {output['word_dist'].shape}")
    
    print(f"\nTheta statistics:")
    theta = output['theta']
    print(f"  Sum per doc (should be 1): {theta.sum(dim=1).mean():.4f}")
    print(f"  Mean: {theta.mean():.4f}")
    print(f"  Std: {theta.std():.4f}")
    print(f"  Max: {theta.max():.4f}")
    print(f"  Min: {theta.min():.4f}")
    
    print(f"\nBeta statistics:")
    beta = output['beta']
    print(f"  Sum per topic (should be 1): {beta.sum(dim=1).mean():.4f}")
    print(f"  Mean: {beta.mean():.6f}")
    print(f"  Std: {beta.std():.6f}")
    print(f"  Max: {beta.max():.6f}")
    print(f"  Min: {beta.min():.6f}")
    
    # Check if beta is uniform
    uniform_prob = 1.0 / vocab_size
    print(f"  Uniform prob would be: {uniform_prob:.6f}")
    print(f"  Beta max/uniform ratio: {beta.max().item() / uniform_prob:.2f}x")
    
    print(f"\nWord distribution statistics:")
    word_dist = output['word_dist']
    print(f"  Sum per doc (should be 1): {word_dist.sum(dim=1).mean():.4f}")
    print(f"  Mean: {word_dist.mean():.6f}")
    print(f"  Max: {word_dist.max():.6f}")
    
    print(f"\nLoss values:")
    print(f"  Recon loss: {output['recon_loss'].item():.4f}")
    print(f"  KL theta loss: {output['kl_theta_loss'].item():.4f}")
    print(f"  KL loss: {output['kl_loss'].item():.4f}")
    print(f"  Total loss: {output['total_loss'].item():.4f}")
    
    # Expected loss for uniform distribution
    expected_uniform_loss = np.log(vocab_size)
    print(f"\nExpected loss for uniform distribution: {expected_uniform_loss:.4f}")
    print(f"Your recon loss: {output['recon_loss'].item():.4f}")
    
    if output['recon_loss'].item() > expected_uniform_loss * 0.95:
        print("\n*** WARNING: Recon loss is close to uniform distribution! ***")
        print("This means the model is not learning meaningful patterns.")
    
    return model, output


def diagnose_loss_calculation():
    """Check loss calculation in detail"""
    print("\n" + "=" * 60)
    print("DIAGNOSING LOSS CALCULATION")
    print("=" * 60)
    
    # Load data
    bow_matrix = sparse.load_npz("/root/autodl-tmp/result/socialTwitter/bow/bow_matrix.npz")
    
    # Sample
    batch_size = 32
    indices = np.random.choice(bow_matrix.shape[0], batch_size, replace=False)
    bow_batch = torch.tensor(bow_matrix[indices].toarray(), dtype=torch.float32)
    
    # Normalize BOW
    bow_sum = bow_batch.sum(dim=1, keepdim=True)
    bow_prob = bow_batch / (bow_sum + 1e-10)
    
    print(f"BOW batch shape: {bow_batch.shape}")
    print(f"BOW sum per doc: {bow_sum.mean():.1f}")
    print(f"BOW prob sum per doc: {bow_prob.sum(dim=1).mean():.4f}")
    
    # Simulate uniform prediction
    vocab_size = bow_batch.shape[1]
    uniform_pred = torch.ones(batch_size, vocab_size) / vocab_size
    log_uniform = torch.log(uniform_pred)
    
    # Calculate loss with uniform prediction
    loss_uniform = -torch.sum(bow_prob * log_uniform, dim=-1).mean()
    print(f"\nLoss with uniform prediction: {loss_uniform.item():.4f}")
    print(f"Expected (log V): {np.log(vocab_size):.4f}")
    
    # Simulate perfect prediction
    perfect_pred = bow_prob + 1e-10
    log_perfect = torch.log(perfect_pred)
    loss_perfect = -torch.sum(bow_prob * log_perfect, dim=-1).mean()
    print(f"Loss with perfect prediction: {loss_perfect.item():.4f}")
    
    # Simulate slightly better prediction
    better_pred = uniform_pred * 0.5 + bow_prob * 0.5
    better_pred = better_pred / better_pred.sum(dim=1, keepdim=True)
    log_better = torch.log(better_pred + 1e-10)
    loss_better = -torch.sum(bow_prob * log_better, dim=-1).mean()
    print(f"Loss with 50% better prediction: {loss_better.item():.4f}")


def main():
    print("=" * 60)
    print("ETM DIAGNOSTIC REPORT")
    print("=" * 60)
    
    diagnose_bow()
    diagnose_embeddings()
    diagnose_loss_calculation()
    diagnose_model()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
