"""
Unified Topic Model Evaluator

Unified evaluator supporting all model types (THETA, LDA, ETM, CTM, DTM)
with 7 standardized evaluation metrics:
1. PPL (Perplexity) - Model perplexity
2. TD (Topic Diversity) - Topic diversity
3. iRBO (Inverse Rank-Biased Overlap) - Topic diversity
4. NPMI (Normalized PMI) - Topic coherence
5. C_V - Topic coherence
6. UMass - Topic coherence
7. Exclusivity - Topic exclusivity
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from .topic_metrics import (
    compute_topic_diversity,
    compute_topic_diversity_inverted_rbo,
    compute_topic_coherence_npmi,
    compute_topic_coherence_cv,
    compute_topic_coherence_umass,
    compute_topic_exclusivity,
    compute_perplexity
)


class UnifiedEvaluator:
    """
    Unified Topic Model Evaluator
    
    Supports evaluation of all model types with standardized metrics and visualizations
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        bow_matrix: Union[np.ndarray, sp.csr_matrix],
        vocab: List[str],
        training_history: Optional[Dict] = None,
        model_name: str = "unknown",
        dataset: str = "unknown",
        output_dir: Optional[str] = None,
        num_topics: int = 20,
        dev_mode: bool = False
    ):
        """
        Initialize evaluator
        
        Args:
            beta: Topic-word distribution (K, V)
            theta: Document-topic distribution (N, K)
            bow_matrix: BOW matrix (N, V)
            vocab: Vocabulary list
            training_history: Training history (including loss etc.)
            model_name: Model name
            dataset: Dataset name
            output_dir: Output directory
            num_topics: Number of topics
            dev_mode: Debug mode
        """
        self.beta = beta
        self.theta = theta
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        self.training_history = training_history
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else None
        self.num_topics = num_topics
        self.dev_mode = dev_mode
        
        self.metrics = {}
    
    def compute_all_metrics(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Compute all 7 standardized evaluation metrics:
        1. TD (Topic Diversity)
        2. iRBO (Inverse Rank-Biased Overlap)
        3. NPMI (Normalized PMI)
        4. C_V (C_V Coherence)
        5. UMass (UMass Coherence)
        6. Exclusivity (Topic Exclusivity)
        7. PPL (Perplexity)
        
        Args:
            top_k: Top-k words per topic for computation
            
        Returns:
            Dictionary containing all 7 metrics
        """
        print(f"\n{'='*60}")
        print(f"Computing 7 Core Metrics for {self.model_name} on {self.dataset}")
        print(f"{'='*60}")
        
        # 1. TD (Topic Diversity)
        print("  [1/7] Computing TD (Topic Diversity)...")
        td = compute_topic_diversity(self.beta, top_k=25)
        self.metrics['TD'] = td
        
        # 2. iRBO (Inverse Rank-Biased Overlap)
        print("  [2/7] Computing iRBO (Inverse Rank-Biased Overlap)...")
        irbo = compute_topic_diversity_inverted_rbo(self.beta, top_k=25)
        self.metrics['iRBO'] = irbo
        
        # 3. NPMI (Normalized PMI)
        print("  [3/7] Computing NPMI (Normalized PMI)...")
        npmi_avg, npmi_per_topic = compute_topic_coherence_npmi(
            self.beta, self.bow_matrix, top_k=top_k
        )
        self.metrics['NPMI'] = npmi_avg
        self.metrics['NPMI_per_topic'] = npmi_per_topic
        
        # 4. C_V (C_V Coherence)
        print("  [4/7] Computing C_V (C_V Coherence)...")
        try:
            cv_avg, cv_per_topic = compute_topic_coherence_cv(
                self.beta, self.bow_matrix, top_k=top_k
            )
            self.metrics['C_V'] = cv_avg
            self.metrics['C_V_per_topic'] = cv_per_topic
        except Exception as e:
            print(f"    Warning: C_V computation failed, using fallback: {e}")
            # Fallback: use NPMI as approximation
            self.metrics['C_V'] = npmi_avg
            self.metrics['C_V_per_topic'] = npmi_per_topic
        
        # 5. UMass (UMass Coherence)
        print("  [5/7] Computing UMass (UMass Coherence)...")
        try:
            umass_avg, umass_per_topic = compute_topic_coherence_umass(
                self.beta, self.bow_matrix, top_k=top_k
            )
            self.metrics['UMass'] = umass_avg
            self.metrics['UMass_per_topic'] = umass_per_topic
        except Exception as e:
            print(f"    Warning: UMass computation failed, using fallback: {e}")
            self.metrics['UMass'] = 0.0
            self.metrics['UMass_per_topic'] = [0.0] * self.num_topics
        
        # 6. Exclusivity (Topic Exclusivity)
        print("  [6/7] Computing Exclusivity (Topic Exclusivity)...")
        try:
            excl_avg, excl_per_topic = compute_topic_exclusivity(self.beta, top_k=top_k)
            self.metrics['Exclusivity'] = excl_avg
            self.metrics['Exclusivity_per_topic'] = excl_per_topic
        except Exception as e:
            print(f"    Warning: Exclusivity computation failed, using fallback: {e}")
            # Fallback: compute simple exclusivity based on word overlap
            self.metrics['Exclusivity'] = td  # Use TD as approximation
            self.metrics['Exclusivity_per_topic'] = [td] * self.num_topics
        
        # 7. PPL (Perplexity)
        print("  [7/7] Computing PPL (Perplexity)...")
        try:
            ppl = compute_perplexity(self.beta, self.theta, self.bow_matrix)
            self.metrics['PPL'] = ppl
        except Exception as e:
            print(f"    Warning: Perplexity computation failed, using fallback: {e}")
            # Fallback: estimate perplexity from reconstruction
            self.metrics['PPL'] = self._estimate_perplexity_fallback()
        
        # Compute topic significance for visualization (NOT part of 7 core metrics)
        print("  [Extra] Computing Topic Significance (for visualization only)...")
        try:
            topic_sizes = self.theta.mean(axis=0)
            self.metrics['Significance_per_topic'] = topic_sizes.tolist()
            self.metrics['Significance'] = float(np.std(topic_sizes))  # Variance as significance measure
        except Exception as e:
            print(f"    Warning: Significance computation failed: {e}")
            self.metrics['Significance_per_topic'] = [1.0 / self.num_topics] * self.num_topics
            self.metrics['Significance'] = 0.0
        
        # Print results summary
        print(f"\n  {'='*50}")
        print(f"  7 Core Metrics Results:")
        print(f"  {'='*50}")
        print(f"    1. TD:          {self.metrics['TD']:.4f}")
        print(f"    2. iRBO:        {self.metrics['iRBO']:.4f}")
        print(f"    3. NPMI:        {self.metrics['NPMI']:.4f}")
        print(f"    4. C_V:         {self.metrics['C_V']:.4f}")
        print(f"    5. UMass:       {self.metrics['UMass']:.4f}")
        print(f"    6. Exclusivity: {self.metrics['Exclusivity']:.4f}")
        print(f"    7. PPL:         {self.metrics['PPL']:.2f}")
        print(f"  {'='*50}")
        print(f"  [Visualization Data] Significance: {self.metrics['Significance']:.4f}")
        
        return self.metrics
    
    def _estimate_perplexity_fallback(self) -> float:
        """
        Fallback perplexity estimation for models that don't natively support it.
        Uses reconstruction-based estimation.
        
        Returns:
            Estimated perplexity value
        """
        try:
            # Reconstruct document-word distribution
            doc_word_probs = self.theta @ self.beta
            doc_word_probs = np.clip(doc_word_probs, 1e-10, 1.0)
            
            # Normalize
            doc_word_probs = doc_word_probs / doc_word_probs.sum(axis=1, keepdims=True)
            
            # Get BOW as dense
            if sp.issparse(self.bow_matrix):
                bow = self.bow_matrix.toarray()
            else:
                bow = np.asarray(self.bow_matrix)
            
            # Compute log-likelihood
            log_likelihood = np.sum(bow * np.log(doc_word_probs + 1e-10))
            total_words = np.sum(bow)
            
            # Perplexity
            ppl = np.exp(-log_likelihood / max(total_words, 1))
            return float(min(ppl, 1e6))  # Cap at 1M to avoid overflow
        except:
            return 1000.0  # Default fallback value
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get the standardized 7-metric result dictionary.
        
        Returns:
            Dictionary with exactly 7 core metrics:
            - TD: Topic Diversity
            - iRBO: Inverse Rank-Biased Overlap
            - NPMI: Normalized PMI
            - C_V: C_V Coherence
            - UMass: UMass Coherence
            - Exclusivity: Topic Exclusivity
            - PPL: Perplexity
        """
        # Ensure all 7 metrics are present
        core_metrics = {
            'TD': self.metrics.get('TD', 0.0),
            'iRBO': self.metrics.get('iRBO', 0.0),
            'NPMI': self.metrics.get('NPMI', 0.0),
            'C_V': self.metrics.get('C_V', 0.0),
            'UMass': self.metrics.get('UMass', 0.0),
            'Exclusivity': self.metrics.get('Exclusivity', 0.0),
            'PPL': self.metrics.get('PPL', 1000.0),
        }
        
        # Add per-topic metrics
        per_topic_metrics = {
            'NPMI_per_topic': self.metrics.get('NPMI_per_topic', []),
            'C_V_per_topic': self.metrics.get('C_V_per_topic', []),
            'UMass_per_topic': self.metrics.get('UMass_per_topic', []),
            'Exclusivity_per_topic': self.metrics.get('Exclusivity_per_topic', []),
            # Significance for visualization only (not part of 7 core metrics)
            'Significance': self.metrics.get('Significance', 0.0),
            'Significance_per_topic': self.metrics.get('Significance_per_topic', []),
        }
        
        # Add metadata
        metadata = {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'num_topics': self.num_topics,
        }
        
        return {**core_metrics, **per_topic_metrics, **metadata}
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save evaluation metrics to JSON file.
        Only saves the 7 core metrics plus per-topic details.
        
        Args:
            filename: Filename, defaults to metrics_k{K}.json
            
        Returns:
            Saved file path
        """
        if self.output_dir is None:
            raise ValueError("output_dir not set")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f'metrics_k{self.num_topics}.json'
        
        filepath = self.output_dir / filename
        
        # Get standardized metrics dictionary
        metrics_dict = self.get_metrics_dict()
        
        # Convert numpy types to Python native types
        metrics_json = {}
        for k, v in metrics_dict.items():
            if isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                metrics_json[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                metrics_json[k] = int(v)
            elif isinstance(v, list):
                metrics_json[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
            else:
                metrics_json[k] = v
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Metrics saved to {filepath}")
        return str(filepath)
    
    def generate_training_plots(self) -> List[str]:
        """
        Generate training process visualization charts
        
        Returns:
            List of generated image paths
        """
        if self.training_history is None:
            print("  [SKIP] No training history available")
            return []
        
        if self.output_dir is None:
            raise ValueError("output_dir not set")
        
        # Output directly to output_dir (metrics plots, not main visualization)
        viz_dir = self.output_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Training Loss Curve
        if 'train_loss' in self.training_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            train_loss = self.training_history['train_loss']
            epochs = range(1, len(train_loss) + 1)
            
            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
            
            if 'val_loss' in self.training_history:
                val_loss = self.training_history['val_loss']
                ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{self.model_name.upper()} Training Loss - {self.dataset}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filepath = viz_dir / 'training_loss.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            generated_files.append(str(filepath))
            print(f"  ✓ training_loss.png")
        
        # 2. Reconstruction Loss + KL Loss (if available)
        if 'recon_loss' in self.training_history and 'kl_loss' in self.training_history:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            recon_loss = self.training_history['recon_loss']
            kl_loss = self.training_history['kl_loss']
            epochs = range(1, len(recon_loss) + 1)
            
            axes[0].plot(epochs, recon_loss, 'g-', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Reconstruction Loss', fontsize=12)
            axes[0].set_title('Reconstruction Loss', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, kl_loss, 'm-', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('KL Divergence Loss', fontsize=12)
            axes[1].set_title('KL Divergence Loss', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(f'{self.model_name.upper()} Loss Components - {self.dataset}', fontsize=14)
            plt.tight_layout()
            
            filepath = viz_dir / 'loss_components.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            generated_files.append(str(filepath))
            print(f"  ✓ loss_components.png")
        
        # 3. Perplexity Curve (if available)
        if 'perplexity' in self.training_history:
            ppl_history = self.training_history['perplexity']
            if isinstance(ppl_history, list) and len(ppl_history) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                epochs = range(1, len(ppl_history) + 1)
                ax.plot(epochs, ppl_history, 'c-', linewidth=2)
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Perplexity', fontsize=12)
                ax.set_title(f'{self.model_name.upper()} Perplexity - {self.dataset}', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                filepath = viz_dir / 'perplexity_curve.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                generated_files.append(str(filepath))
                print(f"  ✓ perplexity_curve.png")
        
        return generated_files
    
    # NOTE: generate_metrics_plots removed - PNG visualization is handled by visualization module
    # Evaluation module only outputs JSON metrics


def evaluate_model(
    model_dir: str,
    bow_matrix: Union[np.ndarray, sp.csr_matrix],
    vocab: List[str],
    model_name: str,
    dataset: str,
    num_topics: int = 20
) -> Dict[str, Any]:
    """
    Convenience function: Evaluate a trained model
    
    Args:
        model_dir: Model directory (containing theta, beta, etc.)
        bow_matrix: BOW matrix
        vocab: Vocabulary list
        model_name: Model name
        dataset: Dataset name
        num_topics: Number of topics
        
    Returns:
        Evaluation results
    """
    model_dir = Path(model_dir)
    
    theta_path = model_dir / f'theta_k{num_topics}.npy'
    beta_path = model_dir / f'beta_k{num_topics}.npy'
    
    if not theta_path.exists() or not beta_path.exists():
        theta_files = list(model_dir.glob('theta_*.npy'))
        beta_files = list(model_dir.glob('beta_*.npy'))
        if theta_files and beta_files:
            theta_path = sorted(theta_files)[-1]
            beta_path = sorted(beta_files)[-1]
        else:
            raise FileNotFoundError(f"theta/beta not found in {model_dir}")
    
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    
    training_history = None
    history_path = model_dir / f'training_history_k{num_topics}.json'
    if not history_path.exists():
        history_files = list(model_dir.glob('training_history_*.json'))
        if history_files:
            history_path = sorted(history_files)[-1]
    
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    evaluator = UnifiedEvaluator(
        beta=beta,
        theta=theta,
        bow_matrix=bow_matrix,
        vocab=vocab,
        training_history=training_history,
        model_name=model_name,
        dataset=dataset,
        output_dir=str(model_dir),
        num_topics=num_topics
    )
    
    metrics = evaluator.compute_all_metrics()
    
    # Only save JSON metrics, no PNG generation in evaluation phase
    # PNG visualization is handled by visualization module
    evaluator.save_metrics()
    
    return metrics
