"""
Test ETM with germanCoal dataset - Print intermediate results in matrix form
"""

import os
import sys
import json
import numpy as np
import torch
from scipy import sparse

# Add paths
sys.path.insert(0, '/root/autodl-tmp/ETM')
sys.path.insert(0, '/root/autodl-tmp')

from engine_a import VocabBuilder, BOWGenerator
from engine_a.vocab_builder import VocabConfig
from engine_c import ETM
from embedding.data_loader import DatasetLoader

def print_matrix(name, matrix, max_rows=10, max_cols=10):
    """Print matrix in readable format"""
    print(f"\n{'='*70}")
    print(f"Matrix: {name}")
    print(f"Shape: {matrix.shape}, dtype: {matrix.dtype}")
    print(f"{'='*70}")
    
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    
    rows, cols = matrix.shape if len(matrix.shape) == 2 else (matrix.shape[0], 1)
    
    # Show subset
    show_rows = min(rows, max_rows)
    show_cols = min(cols, max_cols) if len(matrix.shape) == 2 else 1
    
    print(f"Showing [{show_rows}x{show_cols}] of [{rows}x{cols}]:")
    print("-" * 70)
    
    if len(matrix.shape) == 1:
        print(matrix[:show_rows])
    else:
        subset = matrix[:show_rows, :show_cols]
        # Format nicely
        for i in range(show_rows):
            row_str = " ".join([f"{v:8.4f}" for v in subset[i]])
            print(f"[{i:4d}] {row_str}")
    
    print("-" * 70)
    print(f"Stats: min={matrix.min():.4f}, max={matrix.max():.4f}, mean={matrix.mean():.4f}, std={matrix.std():.4f}")
    if len(matrix.shape) == 2:
        print(f"Row sums (first 5): {matrix[:5].sum(axis=1)}")

def main():
    print("="*70)
    print("ETM Test with germanCoal Dataset")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ========== Step 1: Load Data ==========
    print("\n" + "="*70)
    print("STEP 1: Loading Data")
    print("="*70)
    
    data_loader = DatasetLoader(base_path="/root/autodl-tmp", dev_mode=True)
    texts, labels, info = data_loader.load_dataset("germanCoal", max_samples=1000)
    
    print(f"Loaded {len(texts)} texts")
    print(f"Sample text: {texts[0][:100]}...")
    
    # ========== Step 2: Build Vocabulary & BOW ==========
    print("\n" + "="*70)
    print("STEP 2: Building Vocabulary and BOW Matrix")
    print("="*70)
    
    vocab_config = VocabConfig(
        min_df=3,           # Lower threshold for small dataset
        max_df_ratio=0.8,
        max_vocab_size=5000,
        language="de"       # German
    )
    
    vocab_builder = VocabBuilder(config=vocab_config, dev_mode=True)
    vocab_builder.add_documents(texts, "germanCoal")
    vocab_stats = vocab_builder.build_vocab()
    
    print(f"\nVocabulary size: {vocab_stats.vocab_size}")
    print(f"Total tokens: {vocab_stats.total_tokens}")
    
    # Get vocabulary
    word2idx = vocab_builder.get_word2idx()
    idx2word = vocab_builder.get_idx2word()
    vocab_list = vocab_builder.get_vocab_list()
    
    print(f"\nFirst 20 words in vocabulary:")
    print(vocab_list[:20])
    
    # Generate BOW
    bow_generator = BOWGenerator(vocab_builder, dev_mode=True)
    bow_output = bow_generator.generate_bow(texts, "germanCoal")
    bow_matrix = bow_output.bow_matrix
    
    print_matrix("BOW Matrix (D x V)", bow_matrix)
    
    # ========== Step 3: Load Document Embeddings ==========
    print("\n" + "="*70)
    print("STEP 3: Loading Document Embeddings from Qwen")
    print("="*70)
    
    emb_path = "/root/autodl-tmp/embedding/outputs/zero_shot/germanCoal_zero_shot_embeddings.npy"
    doc_embeddings_full = np.load(emb_path)
    
    # Use same number of samples
    doc_embeddings = doc_embeddings_full[:len(texts)]
    
    print_matrix("Document Embeddings (D x E)", doc_embeddings)
    
    # ========== Step 4: Initialize ETM Model ==========
    print("\n" + "="*70)
    print("STEP 4: Initializing ETM Model")
    print("="*70)
    
    num_topics = 10
    vocab_size = len(vocab_list)
    doc_emb_dim = doc_embeddings.shape[1]
    
    print(f"Configuration:")
    print(f"  num_topics (K): {num_topics}")
    print(f"  vocab_size (V): {vocab_size}")
    print(f"  doc_embedding_dim (E): {doc_emb_dim}")
    
    model = ETM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        doc_embedding_dim=doc_emb_dim,
        word_embedding_dim=doc_emb_dim,  # Same as doc embedding
        hidden_dim=256,
        encoder_dropout=0.2,
        encoder_activation='relu',
        word_embeddings=None,  # Will be randomly initialized
        train_word_embeddings=True,
        dev_mode=True
    )
    
    model = model.to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # ========== Step 5: Forward Pass (Before Training) ==========
    print("\n" + "="*70)
    print("STEP 5: Forward Pass - Before Training")
    print("="*70)
    
    # Prepare batch
    batch_size = min(32, len(texts))
    batch_emb = torch.tensor(doc_embeddings[:batch_size], dtype=torch.float32).to(device)
    batch_bow = torch.tensor(bow_matrix[:batch_size].toarray(), dtype=torch.float32).to(device)
    
    print(f"Batch doc_embeddings shape: {batch_emb.shape}")
    print(f"Batch BOW shape: {batch_bow.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_emb, batch_bow, compute_loss=True)
    
    print("\n--- Intermediate Results ---")
    
    # Theta: Document-Topic Distribution (D x K)
    theta = output['theta']
    print_matrix("Theta (Document-Topic Distribution, D x K)", theta)
    
    # Beta: Topic-Word Distribution (K x V)
    beta = output['beta']
    print_matrix("Beta (Topic-Word Distribution, K x V)", beta)
    
    # Z: Latent representation
    z = output['z']
    print_matrix("Z (Latent Representation)", z)
    
    # Log word distribution
    log_word_dist = output['log_word_dist']
    print_matrix("Log Word Distribution (D x V)", log_word_dist)
    
    print(f"\n--- Loss Values (Before Training) ---")
    print(f"Reconstruction Loss: {output['recon_loss'].item():.4f}")
    print(f"KL Divergence Loss: {output['kl_loss'].item():.4f}")
    print(f"Total Loss: {output['total_loss'].item():.4f}")
    
    # ========== Step 6: Training Loop ==========
    print("\n" + "="*70)
    print("STEP 6: Training ETM (10 epochs)")
    print("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Convert full data to tensors
    all_emb = torch.tensor(doc_embeddings, dtype=torch.float32).to(device)
    all_bow = torch.tensor(bow_matrix.toarray(), dtype=torch.float32).to(device)
    
    num_epochs = 10
    batch_size = 64
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        # Shuffle indices
        indices = torch.randperm(len(texts))
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_indices = indices[start:end]
            
            batch_emb = all_emb[batch_indices]
            batch_bow = all_bow[batch_indices]
            
            optimizer.zero_grad()
            output = model(batch_emb, batch_bow, compute_loss=True)
            
            loss = output['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += output['recon_loss'].item()
            epoch_kl += output['kl_loss'].item()
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KL={avg_kl:.4f})")
    
    # ========== Step 7: Results After Training ==========
    print("\n" + "="*70)
    print("STEP 7: Results After Training")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        output = model(all_emb[:batch_size], all_bow[:batch_size], compute_loss=True)
    
    theta = output['theta']
    beta = output['beta']
    
    print_matrix("Theta After Training (D x K)", theta)
    print_matrix("Beta After Training (K x V)", beta)
    
    print(f"\n--- Loss Values (After Training) ---")
    print(f"Reconstruction Loss: {output['recon_loss'].item():.4f}")
    print(f"KL Divergence Loss: {output['kl_loss'].item():.4f}")
    print(f"Total Loss: {output['total_loss'].item():.4f}")
    
    # ========== Step 8: Topic Interpretation ==========
    print("\n" + "="*70)
    print("STEP 8: Topic Interpretation")
    print("="*70)
    
    beta_np = beta.cpu().numpy()
    
    print("\nTop 10 words per topic:")
    print("-" * 70)
    
    for topic_idx in range(num_topics):
        topic_dist = beta_np[topic_idx]
        top_indices = topic_dist.argsort()[-10:][::-1]
        top_words = [vocab_list[i] for i in top_indices]
        top_probs = [topic_dist[i] for i in top_indices]
        
        print(f"\nTopic {topic_idx}:")
        for word, prob in zip(top_words, top_probs):
            print(f"  {word:20s} {prob:.4f}")
    
    # ========== Step 9: Document-Topic Assignment ==========
    print("\n" + "="*70)
    print("STEP 9: Document-Topic Assignment")
    print("="*70)
    
    # Get theta for all documents
    with torch.no_grad():
        all_theta = []
        for i in range(0, len(texts), batch_size):
            batch_emb = all_emb[i:i+batch_size]
            batch_bow = all_bow[i:i+batch_size]
            output = model(batch_emb, batch_bow, compute_loss=False)
            all_theta.append(output['theta'].cpu().numpy())
        
        all_theta = np.vstack(all_theta)
    
    print_matrix("All Document-Topic Distributions (D x K)", all_theta)
    
    # Topic assignment
    topic_assignments = all_theta.argmax(axis=1)
    print(f"\nTopic assignments (first 20 docs): {topic_assignments[:20]}")
    
    # Topic distribution
    unique, counts = np.unique(topic_assignments, return_counts=True)
    print(f"\nTopic distribution:")
    for t, c in zip(unique, counts):
        print(f"  Topic {t}: {c} documents ({100*c/len(texts):.1f}%)")
    
    print("\n" + "="*70)
    print("ETM Test Completed Successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
