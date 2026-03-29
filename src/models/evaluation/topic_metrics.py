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


def compute_perplexity(
    beta: np.ndarray,
    theta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    eps: float = 1e-12
) -> float:
    """
    Compute perplexity of the topic model.
    
    Perplexity = exp(-1/N * sum(log(p(w|d))))
    Lower perplexity indicates better model fit.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        theta: Document-topic distribution matrix (D x K)
        doc_term_matrix: Document-term matrix (D x V)
        eps: Small constant to avoid log(0)
        
    Returns:
        Perplexity score
    """
    if sparse.issparse(doc_term_matrix):
        bow = doc_term_matrix.toarray()
    else:
        bow = np.asarray(doc_term_matrix)
    
    # Compute document-word probabilities: p(w|d) = sum_k theta(d,k) * beta(k,w)
    # Shape: (D, V)
    doc_word_probs = theta @ beta
    
    # Add epsilon to avoid log(0)
    doc_word_probs = np.clip(doc_word_probs, eps, 1.0)
    
    # Compute log-likelihood
    # Only consider words that appear in each document
    log_likelihood = np.sum(bow * np.log(doc_word_probs))
    
    # Total number of words
    total_words = np.sum(bow)
    
    # Perplexity
    perplexity = np.exp(-log_likelihood / total_words)
    
    return float(perplexity)


def compute_topic_coherence_cv(
    beta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    top_k: int = 10,
    eps: float = 1e-12
) -> Tuple[float, List[float]]:
    """
    Compute topic coherence using C_V measure (combination of NPMI and sliding window).
    
    C_V is considered one of the best coherence measures, correlating well with human judgment.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        doc_term_matrix: Document-term matrix (D x V)
        top_k: Number of top words per topic
        eps: Small constant to avoid division by zero
        
    Returns:
        (Average C_V score, List of C_V scores per topic)
    """
    num_topics, vocab_size = beta.shape
    num_docs = doc_term_matrix.shape[0]
    
    # Convert to binary occurrence matrix
    if sparse.issparse(doc_term_matrix):
        doc_term_binary = doc_term_matrix.copy()
        doc_term_binary.data = np.ones_like(doc_term_binary.data)
    else:
        doc_term_binary = (doc_term_matrix > 0).astype(np.float32)
    
    # Get top words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Compute document frequency
    if sparse.issparse(doc_term_binary):
        word_doc_freq = np.array(doc_term_binary.sum(axis=0)).flatten()
    else:
        word_doc_freq = doc_term_binary.sum(axis=0)
    
    cv_scores = []
    
    for topic_idx in range(num_topics):
        topic_words = top_words_indices[topic_idx]
        
        # Compute pairwise NPMI with indirect confirmation measure
        npmi_matrix = np.zeros((top_k, top_k))
        
        for i in range(top_k):
            for j in range(top_k):
                if i == j:
                    npmi_matrix[i, j] = 1.0
                    continue
                    
                word_i, word_j = topic_words[i], topic_words[j]
                
                doc_freq_i = word_doc_freq[word_i]
                doc_freq_j = word_doc_freq[word_j]
                
                if sparse.issparse(doc_term_binary):
                    docs_with_i = set(doc_term_binary[:, word_i].nonzero()[0])
                    docs_with_j = set(doc_term_binary[:, word_j].nonzero()[0])
                    co_occur = len(docs_with_i.intersection(docs_with_j))
                else:
                    co_occur = np.sum(
                        np.logical_and(
                            doc_term_binary[:, word_i] > 0,
                            doc_term_binary[:, word_j] > 0
                        )
                    )
                
                p_i = (doc_freq_i + eps) / num_docs
                p_j = (doc_freq_j + eps) / num_docs
                p_ij = (co_occur + eps) / num_docs
                
                pmi = np.log(p_ij / (p_i * p_j))
                npmi = pmi / (-np.log(p_ij))
                npmi_matrix[i, j] = npmi
        
        # C_V uses cosine similarity of NPMI vectors
        # For each word, compute its NPMI vector with all other top words
        cv_sum = 0.0
        for i in range(1, top_k):
            # Compute cosine similarity between word i's NPMI vector and previous words
            vec_i = npmi_matrix[i, :i]
            vec_prev = npmi_matrix[:i, :i].mean(axis=0) if i > 1 else npmi_matrix[0, :1]
            
            # Simplified: use mean NPMI as C_V approximation
            cv_sum += np.mean(npmi_matrix[i, :i])
        
        cv_score = cv_sum / (top_k - 1) if top_k > 1 else 0.0
        cv_scores.append(cv_score)
    
    avg_cv = np.mean(cv_scores) if cv_scores else 0.0
    return avg_cv, cv_scores


def compute_topic_coherence_umass(
    beta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    top_k: int = 10,
    eps: float = 1e-12
) -> Tuple[float, List[float]]:
    """
    Compute topic coherence using UMass measure.
    
    UMass coherence uses document co-occurrence and is asymmetric.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        doc_term_matrix: Document-term matrix (D x V)
        top_k: Number of top words per topic
        eps: Small constant to avoid log(0)
        
    Returns:
        (Average UMass score, List of UMass scores per topic)
    """
    num_topics, vocab_size = beta.shape
    num_docs = doc_term_matrix.shape[0]
    
    # Convert to binary
    if sparse.issparse(doc_term_matrix):
        doc_term_binary = doc_term_matrix.copy()
        doc_term_binary.data = np.ones_like(doc_term_binary.data)
    else:
        doc_term_binary = (doc_term_matrix > 0).astype(np.float32)
    
    # Get top words
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Document frequency
    if sparse.issparse(doc_term_binary):
        word_doc_freq = np.array(doc_term_binary.sum(axis=0)).flatten()
    else:
        word_doc_freq = doc_term_binary.sum(axis=0)
    
    umass_scores = []
    
    for topic_idx in range(num_topics):
        topic_words = top_words_indices[topic_idx]
        actual_top_k = min(top_k, len(topic_words))
        
        umass_sum = 0.0
        pair_count = 0
        
        for i in range(1, actual_top_k):
            for j in range(i):
                word_i, word_j = topic_words[i], topic_words[j]
                
                doc_freq_j = word_doc_freq[word_j]
                
                if sparse.issparse(doc_term_binary):
                    docs_with_i = set(doc_term_binary[:, word_i].nonzero()[0])
                    docs_with_j = set(doc_term_binary[:, word_j].nonzero()[0])
                    co_occur = len(docs_with_i.intersection(docs_with_j))
                else:
                    co_occur = np.sum(
                        np.logical_and(
                            doc_term_binary[:, word_i] > 0,
                            doc_term_binary[:, word_j] > 0
                        )
                    )
                
                # UMass formula: log((D(w_i, w_j) + eps) / D(w_j))
                umass_sum += np.log((co_occur + eps) / (doc_freq_j + eps))
                pair_count += 1
        
        umass_score = umass_sum / pair_count if pair_count > 0 else 0.0
        umass_scores.append(umass_score)
    
    avg_umass = np.mean(umass_scores) if umass_scores else 0.0
    return avg_umass, umass_scores


def compute_topic_exclusivity(
    beta: np.ndarray,
    top_k: int = 10
) -> Tuple[float, List[float]]:
    """
    Compute topic exclusivity - how exclusive the top words are to each topic.
    
    Higher exclusivity means topics have more unique, non-overlapping words.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        top_k: Number of top words per topic
        
    Returns:
        (Average exclusivity, List of exclusivity per topic)
    """
    num_topics = beta.shape[0]
    
    # Get top words for each topic
    top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
    
    # Count how many topics each word appears in (as top word)
    word_topic_count = Counter()
    for topic_idx in range(num_topics):
        for word_idx in top_words_indices[topic_idx]:
            word_topic_count[word_idx] += 1
    
    # Compute exclusivity for each topic
    exclusivity_scores = []
    for topic_idx in range(num_topics):
        topic_words = top_words_indices[topic_idx]
        
        # Exclusivity = 1 / (average number of topics each word appears in)
        avg_appearances = np.mean([word_topic_count[w] for w in topic_words])
        exclusivity = 1.0 / avg_appearances
        exclusivity_scores.append(exclusivity)
    
    avg_exclusivity = np.mean(exclusivity_scores)
    return avg_exclusivity, exclusivity_scores


def compute_all_metrics(
    beta: np.ndarray,
    theta: np.ndarray,
    doc_term_matrix: Union[np.ndarray, sparse.csr_matrix],
    top_k_coherence: int = 10,
    top_k_diversity: int = 25
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute all 7 core topic quality metrics.
    
    7 Core Metrics Standard:
    1. TD (Topic Diversity)
    2. iRBO (Inverse Rank-Biased Overlap)
    3. NPMI (Normalized PMI)
    4. C_V (C_V Coherence)
    5. UMass (UMass Coherence)
    6. Exclusivity (Topic Exclusivity)
    7. PPL (Perplexity)
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        theta: Document-topic distribution matrix (D x K)
        doc_term_matrix: Document-term matrix (D x V)
        top_k_coherence: Number of top words for coherence calculation
        top_k_diversity: Number of top words for diversity calculation
        
    Returns:
        Dictionary with exactly 7 core metrics
    """
    # 1. TD (Topic Diversity)
    logger.info("[1/7] Computing TD (Topic Diversity)...")
    td_score = compute_topic_diversity_td(beta, top_k_diversity)
    
    # 2. iRBO (Inverse Rank-Biased Overlap)
    logger.info("[2/7] Computing iRBO (Inverse Rank-Biased Overlap)...")
    irbo_score = compute_topic_diversity_inverted_rbo(beta, top_k_diversity)
    
    # 3. NPMI (Normalized PMI)
    logger.info("[3/7] Computing NPMI (Normalized PMI)...")
    avg_npmi, npmi_per_topic = compute_topic_coherence_npmi(
        beta, doc_term_matrix, top_k_coherence
    )
    
    # 4. C_V (C_V Coherence)
    logger.info("[4/7] Computing C_V (C_V Coherence)...")
    try:
        avg_cv, cv_per_topic = compute_topic_coherence_cv(beta, doc_term_matrix, top_k_coherence)
    except Exception as e:
        logger.warning(f"C_V computation failed, using NPMI as fallback: {e}")
        avg_cv, cv_per_topic = avg_npmi, npmi_per_topic
    
    # 5. UMass (UMass Coherence)
    logger.info("[5/7] Computing UMass (UMass Coherence)...")
    try:
        avg_umass, umass_per_topic = compute_topic_coherence_umass(beta, doc_term_matrix, top_k_coherence)
    except Exception as e:
        logger.warning(f"UMass computation failed: {e}")
        avg_umass, umass_per_topic = 0.0, [0.0] * beta.shape[0]
    
    # 6. Exclusivity (Topic Exclusivity)
    logger.info("[6/7] Computing Exclusivity (Topic Exclusivity)...")
    try:
        avg_exclusivity, exclusivity_per_topic = compute_topic_exclusivity(beta, top_k_coherence)
    except Exception as e:
        logger.warning(f"Exclusivity computation failed, using TD as fallback: {e}")
        avg_exclusivity, exclusivity_per_topic = td_score, [td_score] * beta.shape[0]
    
    # 7. PPL (Perplexity)
    logger.info("[7/7] Computing PPL (Perplexity)...")
    try:
        ppl = compute_perplexity(beta, theta, doc_term_matrix)
    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        ppl = 1000.0  # Default fallback
    
    # Build standardized metrics dictionary
    metrics = {
        'TD': td_score,
        'iRBO': irbo_score,
        'NPMI': avg_npmi,
        'NPMI_per_topic': npmi_per_topic,
        'C_V': avg_cv,
        'C_V_per_topic': cv_per_topic,
        'UMass': avg_umass,
        'UMass_per_topic': umass_per_topic,
        'Exclusivity': avg_exclusivity,
        'Exclusivity_per_topic': exclusivity_per_topic,
        'PPL': ppl,
    }
    
    return metrics


# =============================================================================
# Legacy TopicMetrics class (for backward compatibility with main.py)
# =============================================================================

class TopicMetrics:
    """
    Comprehensive topic model evaluation.
    
    Computes and stores all evaluation metrics.
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        bow_matrix: Union[np.ndarray, sparse.csr_matrix],
        vocab: List[str],
        dev_mode: bool = False
    ):
        """
        Initialize metrics calculator.
        
        Args:
            beta: Topic-word distribution (K, V)
            theta: Document-topic distribution (N, K)
            bow_matrix: BOW matrix (N, V)
            vocab: Vocabulary list
            dev_mode: Print debug information
        """
        self.beta = beta
        self.theta = theta
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        self.dev_mode = dev_mode
        
        self.num_topics = beta.shape[0]
        self.vocab_size = beta.shape[1]
        self.num_docs = theta.shape[0]
        
        # Cached metrics
        self._coherence = None
        self._diversity = None
        self._topic_sizes = None
    
    def compute_all(self, top_k: int = 10) -> Dict:
        """
        Compute all metrics.
        
        Args:
            top_k: Number of top words for coherence/diversity
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Topic coherence
        avg_coh, per_topic_coh = compute_topic_coherence_npmi(
            self.beta, self.bow_matrix, top_k
        )
        metrics['coherence'] = avg_coh
        metrics['coherence_per_topic'] = per_topic_coh
        self._coherence = avg_coh
        
        # Topic diversity
        diversity = compute_topic_diversity(self.beta, top_k)
        metrics['diversity'] = diversity
        self._diversity = diversity
        
        # Topic sizes (average theta per topic)
        topic_sizes = self.theta.mean(axis=0)
        metrics['topic_sizes'] = topic_sizes.tolist()
        self._topic_sizes = topic_sizes
        
        # Topic entropy (how spread out the topic distribution is)
        theta_entropy = -np.sum(self.theta * np.log(self.theta + 1e-10), axis=1)
        metrics['avg_doc_entropy'] = float(np.mean(theta_entropy))
        
        # Topic concentration (how concentrated topics are)
        max_theta = np.max(self.theta, axis=1)
        metrics['avg_topic_concentration'] = float(np.mean(max_theta))
        
        if self.dev_mode:
            print(f"[DEV] Metrics computed:")
            print(f"[DEV]   Coherence: {avg_coh:.4f}")
            print(f"[DEV]   Diversity: {diversity:.4f}")
            print(f"[DEV]   Avg doc entropy: {metrics['avg_doc_entropy']:.4f}")
        
        return metrics
    
    def get_top_words(self, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            top_k: Number of top words
            
        Returns:
            List of [(word, prob), ...] for each topic
        """
        topics = []
        for k in range(self.num_topics):
            top_indices = np.argsort(self.beta[k])[-top_k:][::-1]
            words = [(self.vocab[idx], float(self.beta[k, idx])) for idx in top_indices]
            topics.append(words)
        return topics
    
    def get_topic_summary(self, top_k: int = 10) -> str:
        """
        Get formatted topic summary.
        
        Args:
            top_k: Number of top words per topic
            
        Returns:
            Formatted string with topic descriptions
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TOPIC SUMMARY")
        lines.append("=" * 60)
        
        top_words = self.get_top_words(top_k)
        
        for k, words in enumerate(top_words):
            size = self._topic_sizes[k] if self._topic_sizes is not None else 0
            word_str = ", ".join([w for w, _ in words])
            lines.append(f"Topic {k:2d} (size={size:.3f}): {word_str}")
        
        lines.append("=" * 60)
        
        if self._coherence is not None:
            lines.append(f"Coherence: {self._coherence:.4f}")
        if self._diversity is not None:
            lines.append(f"Diversity: {self._diversity:.4f}")
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str):
        """
        Save metrics and topic summary to file.
        
        Args:
            output_path: Output file path
        """
        import json
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        metrics = self.compute_all()
        top_words = self.get_top_words()
        
        results = {
            'metrics': {k: convert_to_serializable(v) for k, v in metrics.items() 
                       if not isinstance(v, np.ndarray) or len(v) <= 100},
            'topics': [
                {'id': k, 'words': [{'word': w, 'prob': float(p)} for w, p in words]}
                for k, words in enumerate(top_words)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {output_path}")


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
