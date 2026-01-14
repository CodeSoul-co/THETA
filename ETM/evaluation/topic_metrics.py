"""
Topic Quality Metrics for ETM

This module provides metrics to evaluate topic quality:
- Topic coherence (NPMI, PMI, UCI, UMass)
- Topic diversity
- Topic significance
"""

import numpy as np
import scipy.sparse as sparse
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_topic_diversity(
    beta: np.ndarray,
    top_k: int = 25
) -> float:
    """
    Compute topic diversity as the percentage of unique words in top-k words across all topics.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        top_k: Number of top words per topic
        
    Returns:
        Topic diversity score (0-1)
    """
    num_topics = beta.shape[0]
    
    # Get top-k words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Count unique words
    unique_words = set()
    for topic_idx in range(num_topics):
        unique_words.update(top_words_indices[topic_idx])
    
    # Calculate diversity
    total_words = num_topics * top_k
    diversity = len(unique_words) / total_words
    
    return diversity


def compute_topic_coherence_npmi(
    beta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    top_k: int = 10,
    eps: float = 1e-12
) -> Tuple[float, List[float]]:
    """
    Compute topic coherence using Normalized Pointwise Mutual Information (NPMI).
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        doc_term_matrix: Document-term matrix (D x V)
        top_k: Number of top words per topic
        eps: Small constant to avoid log(0)
        
    Returns:
        (Average coherence score, List of coherence scores per topic)
    """
    num_topics, vocab_size = beta.shape
    num_docs = doc_term_matrix.shape[0]
    
    # Convert to binary occurrence matrix if not already
    if sparse.issparse(doc_term_matrix):
        doc_term_binary = doc_term_matrix.copy()
        doc_term_binary.data = np.ones_like(doc_term_binary.data)
    else:
        doc_term_binary = (doc_term_matrix > 0).astype(np.float32)
    
    # Get top words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Compute document frequency for each word
    if sparse.issparse(doc_term_binary):
        word_doc_freq = np.array(doc_term_binary.sum(axis=0)).flatten()
    else:
        word_doc_freq = doc_term_binary.sum(axis=0)
    
    # Compute NPMI for each topic
    coherence_scores = []
    
    for topic_idx in range(num_topics):
        topic_words = top_words_indices[topic_idx]
        
        # Compute pairwise NPMI
        npmi_scores = []
        
        for i in range(len(topic_words)):
            for j in range(i+1, len(topic_words)):
                word_i, word_j = topic_words[i], topic_words[j]
                
                # Get document frequencies
                doc_freq_i = word_doc_freq[word_i]
                doc_freq_j = word_doc_freq[word_j]
                
                # Get co-occurrence frequency
                if sparse.issparse(doc_term_binary):
                    # For sparse matrices
                    docs_with_i = doc_term_binary[:, word_i].nonzero()[0]
                    docs_with_j = doc_term_binary[:, word_j].nonzero()[0]
                    co_occur = len(set(docs_with_i).intersection(set(docs_with_j)))
                else:
                    # For dense matrices
                    co_occur = np.sum(
                        np.logical_and(
                            doc_term_binary[:, word_i] > 0,
                            doc_term_binary[:, word_j] > 0
                        )
                    )
                
                # Compute probabilities
                p_i = (doc_freq_i + eps) / num_docs
                p_j = (doc_freq_j + eps) / num_docs
                p_ij = (co_occur + eps) / num_docs
                
                # Compute PMI and NPMI
                pmi = np.log(p_ij / (p_i * p_j))
                npmi = pmi / (-np.log(p_ij))
                
                npmi_scores.append(npmi)
        
        # Average NPMI for this topic
        if npmi_scores:
            topic_coherence = np.mean(npmi_scores)
            coherence_scores.append(topic_coherence)
    
    # Average coherence across all topics
    avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
    
    return avg_coherence, coherence_scores


def compute_topic_diversity_td(
    beta: np.ndarray,
    top_k: int = 25
) -> float:
    """
    Compute Topic Diversity (TD) score.
    
    TD measures the proportion of unique words in the top-k words of all topics.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        top_k: Number of top words per topic
        
    Returns:
        TD score (0-1)
    """
    num_topics = beta.shape[0]
    
    # Get top-k words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Count unique words
    unique_words = set()
    for topic_idx in range(num_topics):
        unique_words.update(top_words_indices[topic_idx])
    
    # Calculate diversity
    td_score = len(unique_words) / (num_topics * top_k)
    
    return td_score


def compute_topic_diversity_inverted_rbo(
    beta: np.ndarray,
    top_k: int = 25,
    p: float = 0.9
) -> float:
    """
    Compute Topic Diversity using inverted Rank-Biased Overlap (RBO).
    
    This measures how different the top word rankings are between topics.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        top_k: Number of top words per topic
        p: Persistence parameter for RBO (higher values give more weight to higher ranks)
        
    Returns:
        Inverted RBO score (0-1), higher means more diverse
    """
    num_topics = beta.shape[0]
    
    # Get top-k words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Compute pairwise RBO
    rbo_sum = 0.0
    pair_count = 0
    
    for i in range(num_topics):
        for j in range(i+1, num_topics):
            # Get top words for topics i and j
            words_i = set(top_words_indices[i])
            words_j = set(top_words_indices[j])
            
            # Compute overlap at each depth
            overlap_sum = 0.0
            weight_sum = 0.0
            
            for d in range(1, top_k + 1):
                # Overlap at depth d
                overlap_d = len(set(top_words_indices[i][:d]).intersection(set(top_words_indices[j][:d]))) / d
                # Weight for depth d
                weight_d = (1 - p) * (p ** (d - 1))
                
                overlap_sum += weight_d * overlap_d
                weight_sum += weight_d
            
            # Normalize by weights
            rbo = overlap_sum / weight_sum
            rbo_sum += rbo
            pair_count += 1
    
    # Average RBO across all pairs
    avg_rbo = rbo_sum / pair_count if pair_count > 0 else 0.0
    
    # Invert so higher means more diverse
    return 1.0 - avg_rbo


def compute_topic_significance(
    theta: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Compute topic significance as the proportion of documents where the topic is dominant.
    
    Args:
        theta: Document-topic distribution matrix (D x K)
        threshold: Threshold for considering a topic significant in a document
        
    Returns:
        Array of significance scores per topic
    """
    num_docs, num_topics = theta.shape
    
    # Count documents where each topic is significant
    significant_counts = np.sum(theta > threshold, axis=0)
    
    # Normalize by number of documents
    significance = significant_counts / num_topics
    
    return significance


def compute_all_metrics(
    beta: np.ndarray,
    theta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    top_k_coherence: int = 10,
    top_k_diversity: int = 25
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute all topic quality metrics.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        theta: Document-topic distribution matrix (D x K)
        doc_term_matrix: Document-term matrix (D x V)
        top_k_coherence: Number of top words for coherence calculation
        top_k_diversity: Number of top words for diversity calculation
        
    Returns:
        Dictionary with all metrics
    """
    logger.info("Computing topic diversity (TD)...")
    td_score = compute_topic_diversity_td(beta, top_k_diversity)
    
    logger.info("Computing topic diversity (inverted RBO)...")
    irbo_score = compute_topic_diversity_inverted_rbo(beta, top_k_diversity)
    
    logger.info("Computing topic coherence (NPMI)...")
    avg_coherence, topic_coherences = compute_topic_coherence_npmi(
        beta, doc_term_matrix, top_k_coherence
    )
    
    logger.info("Computing topic significance...")
    significance = compute_topic_significance(theta)
    
    return {
        'topic_diversity_td': td_score,
        'topic_diversity_irbo': irbo_score,
        'topic_coherence_avg': avg_coherence,
        'topic_coherence_per_topic': topic_coherences,
        'topic_significance': significance.tolist()
    }


if __name__ == "__main__":
    import argparse
    import os
    import json
    from scipy import sparse
    
    parser = argparse.ArgumentParser(description="Compute topic quality metrics")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing ETM results")
    parser.add_argument("--bow_path", type=str, required=True,
                        help="Path to BOW matrix (.npz)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save metrics")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific timestamp to load")
    
    args = parser.parse_args()
    
    # Load ETM results
    if args.timestamp:
        theta_path = os.path.join(args.results_dir, f"theta_{args.timestamp}.npy")
        beta_path = os.path.join(args.results_dir, f"beta_{args.timestamp}.npy")
    else:
        # Find latest files
        import glob
        from pathlib import Path
        
        theta_files = sorted(Path(args.results_dir).glob("theta_*.npy"), reverse=True)
        beta_files = sorted(Path(args.results_dir).glob("beta_*.npy"), reverse=True)
        
        if not theta_files or not beta_files:
            raise FileNotFoundError(f"Could not find ETM result files in {args.results_dir}")
        
        theta_path = str(theta_files[0])
        beta_path = str(beta_files[0])
    
    # Load matrices
    logger.info(f"Loading theta from {theta_path}")
    theta = np.load(theta_path)
    
    logger.info(f"Loading beta from {beta_path}")
    beta = np.load(beta_path)
    
    logger.info(f"Loading BOW matrix from {args.bow_path}")
    bow_matrix = sparse.load_npz(args.bow_path)
    
    # Compute metrics
    metrics = compute_all_metrics(beta, theta, bow_matrix)
    
    # Save or print metrics
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {args.output_path}")
    else:
        print(json.dumps(metrics, indent=2))
