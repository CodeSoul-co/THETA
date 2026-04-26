"""
ETM (Embedded Topic Model) - Modified for Qwen Embedding Input

Key modifications from original ETM:
1. Encoder input: Qwen document embeddings (1024-dim) instead of BOW
2. Decoder: Uses Qwen word embeddings as semantic basis
3. Loss: BOW reconstruction + KL divergence

This separates:
- Semantic understanding (Qwen embeddings)
- Structure modeling (ETM topic distribution)
- Interpretability (word-based topic descriptions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from .encoder import ETMEncoder
from .decoder import ETMDecoder


class ETM(nn.Module):
    """
    Embedded Topic Model with Qwen embedding input.
    
    Architecture:
        doc_embedding (Qwen) -> Encoder -> theta (topic dist)
        theta -> Decoder (with word embeddings) -> word_dist
        Loss = Reconstruction(word_dist, BOW) + KL(theta)
    
    Key outputs:
        - theta: Document-topic distribution (D x K)
        - beta: Topic-word distribution (K x V)
        - topic_embeddings: Topic vectors in embedding space (K x E)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        doc_embedding_dim: int = 1024,      # Qwen embedding dimension
        word_embedding_dim: int = 1024,     # Word embedding dimension
        hidden_dim: int = 512,              # Encoder hidden dimension
        encoder_dropout: float = 0.2,
        encoder_activation: str = 'relu',
        word_embeddings: Optional[torch.Tensor] = None,
        train_word_embeddings: bool = False,
        kl_weight: float = 0.5,            # Weight for KL divergence loss
        num_classes: int = 0,              # Number of classes for supervised mode (0 = disabled)
        contrastive_weight: float = 0.1,   # Weight for contrastive loss (unsupervised mode)
        contrastive_temp: float = 0.07,    # Temperature for contrastive loss
        dev_mode: bool = False
    ):
        """
        Initialize ETM model.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics (K)
            doc_embedding_dim: Dimension of document embeddings from Qwen
            word_embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for encoder
            encoder_dropout: Dropout rate for encoder
            encoder_activation: Activation function for encoder
            word_embeddings: Pre-trained word embeddings (V x E)
            train_word_embeddings: Whether to fine-tune word embeddings
            num_classes: Number of classes for supervised learning (0 = disabled)
            contrastive_weight: Weight for contrastive loss in unsupervised mode
            contrastive_temp: Temperature for contrastive loss (lower = sharper)
            dev_mode: Print debug information
            
        Note:
            - Supervised mode: CE + Recon + KL (with AdaptiveLossWeighter)
            - Unsupervised mode: NLL + Contrastive + KL
        """
        super(ETM, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.doc_embedding_dim = doc_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.dev_mode = dev_mode
        
        # Encoder: doc_embedding -> theta
        self.encoder = ETMEncoder(
            input_dim=doc_embedding_dim,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            dropout=encoder_dropout,
            activation=encoder_activation
        )
        
        # Decoder: theta -> word_dist
        self.decoder = ETMDecoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embedding_dim=word_embedding_dim,
            word_embeddings=word_embeddings,
            train_embeddings=train_word_embeddings
        )
        
        # Classification head for supervised mode
        if num_classes > 0:
            self.classifier = nn.Linear(num_topics, num_classes)
            self.ce_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classifier = None
            self.ce_loss_fn = None
        
        if self.dev_mode:
            print(f"[DEV] ETM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   doc_embedding_dim={doc_embedding_dim}")
            print(f"[DEV]   word_embedding_dim={word_embedding_dim}")
            print(f"[DEV]   hidden_dim={hidden_dim}")
            print(f"[DEV]   num_classes={num_classes} (supervised={'enabled' if num_classes > 0 else 'disabled'})")
            print(f"[DEV]   contrastive_weight={contrastive_weight}, contrastive_temp={contrastive_temp}")
    
    def _compute_contrastive_loss(
        self,
        theta: torch.Tensor,
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between topic distributions and document embeddings.
        
        This loss encourages documents with similar semantic content (similar Qwen embeddings)
        to have similar topic distributions, and vice versa.
        
        Uses InfoNCE loss:
            L = -log(exp(sim(z_i, z_i+) / τ) / Σ_j exp(sim(z_i, z_j) / τ))
        
        Args:
            theta: Topic distributions, shape (batch_size, num_topics)
            doc_embeddings: Document embeddings, shape (batch_size, doc_embedding_dim)
            
        Returns:
            Contrastive loss (scalar)
        """
        batch_size = theta.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=theta.device)
        
        # L2 normalize both representations
        theta_norm = F.normalize(theta, p=2, dim=1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=1)
        
        # Compute similarity matrix between theta and doc_embeddings
        # sim[i, j] = similarity between theta[i] and doc_embedding[j]
        # We want diagonal elements (positive pairs) to be high
        
        # Project doc_embeddings to topic space for fair comparison
        # Use a simple linear projection (reuse encoder's first layer weights conceptually)
        # Actually, we compute cross-modal similarity directly
        
        # Compute cosine similarity within each modality
        theta_sim = torch.mm(theta_norm, theta_norm.t()) / self.contrastive_temp  # (B, B)
        doc_sim = torch.mm(doc_norm, doc_norm.t()) / self.contrastive_temp  # (B, B)
        
        # Soft labels: documents with similar embeddings should have similar topics
        # Use doc_sim as soft targets for theta_sim
        doc_soft_labels = F.softmax(doc_sim, dim=1)
        
        # Cross-entropy between theta similarity and doc similarity (soft labels)
        # This encourages theta to preserve the similarity structure of doc_embeddings
        log_theta_sim = F.log_softmax(theta_sim, dim=1)
        contrastive_loss = -torch.sum(doc_soft_labels * log_theta_sim, dim=1).mean()
        
        return contrastive_loss
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'unsupervised',
        compute_loss: bool = True,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ETM.
        
        Args:
            doc_embeddings: Document embeddings from Qwen, shape (batch, doc_emb_dim)
            bow_targets: BOW target for reconstruction, shape (batch, vocab_size)
            labels: Optional labels for supervised mode, shape (batch,)
            mode: Training mode - 'zero_shot', 'supervised', or 'unsupervised'
            compute_loss: Whether to compute loss
            kl_weight: Weight for KL divergence loss
            
        Returns:
            Dictionary containing:
                - theta: Topic distribution (batch, K)
                - recon_loss: Reconstruction loss (scalar)
                - kl_loss: KL divergence loss (scalar)
                - total_loss: Total loss (scalar)
        """
        # Encode: doc_embedding -> theta
        theta, z, kl_theta_loss = self.encoder(doc_embeddings, compute_kl=True)
        
        # Get topic-word distribution
        beta = self.decoder.get_beta()
        
        # Decode: theta -> word distribution
        log_word_dist = self.decoder(theta, beta)
        
        output = {
            'theta': theta,
            'z': z,
            'beta': beta,
            'log_word_dist': log_word_dist
        }
        
        if compute_loss:
            # Mode-based loss routing
            if mode == 'zero_shot':
                # Zero-shot: no training loss, pure feature extraction
                output['total_loss'] = torch.tensor(0.0, device=doc_embeddings.device)
                output['recon_loss'] = torch.tensor(0.0, device=doc_embeddings.device)
                output['kl_loss'] = torch.tensor(0.0, device=doc_embeddings.device)
                
            elif mode == 'supervised':
                # Supervised: classification loss + reconstruction loss + KL divergence
                # This joint loss preserves topic model structure while learning classification
                # 
                # NOTE: Loss weighting is handled externally by AdaptiveLossWeighter
                # This method returns raw losses; the training loop computes:
                #   total_loss = w_ce * ce_loss + w_recon * recon_loss + kl_loss
                # where w_ce and w_recon are adaptive weights, and kl_loss uses fixed kl_weight
                # to avoid posterior collapse.
                if self.classifier is None:
                    raise ValueError("Supervised mode requires num_classes > 0 during model initialization")
                if labels is None:
                    raise ValueError("Supervised mode requires labels")
                
                # 1. Classification loss (raw, unweighted)
                logits = self.classifier(theta)
                ce_loss = self.ce_loss_fn(logits, labels)
                
                # 2. Reconstruction loss (raw, unweighted)
                recon_loss = -torch.sum(bow_targets * log_word_dist, dim=-1).mean()
                
                # 3. KL divergence with free bits (uses fixed kl_weight, NOT adaptive)
                free_bits = 0.5
                kl_loss = kl_weight * torch.clamp(kl_theta_loss, min=free_bits)
                
                output['logits'] = logits
                output['ce_loss'] = ce_loss
                output['recon_loss'] = recon_loss
                output['kl_theta_loss'] = kl_theta_loss
                output['kl_loss'] = kl_loss
                # Note: total_loss is computed by training loop with adaptive weights
                # Default fallback (if not using adaptive weighting):
                output['total_loss'] = ce_loss + recon_loss + kl_loss
                
            else:  # mode == 'unsupervised' (default, original behavior)
                # Unsupervised: NLL + Contrastive + KL
                # Loss = NLL(reconstruction) + λ_c * Contrastive + λ_kl * KL
                #
                # Components:
                # 1. NLL (Negative Log-Likelihood): BOW reconstruction loss
                # 2. Contrastive: Preserves semantic similarity structure from Qwen embeddings
                # 3. KL: Regularizes latent space to standard normal
                
                # 1. NLL / Reconstruction loss: negative log likelihood
                # recon_loss = -sum(p_true * log(p_pred)) where p_true is normalized BOW
                nll_loss = -torch.sum(bow_targets * log_word_dist, dim=-1).mean()
                
                # 2. Contrastive loss: preserve similarity structure
                # Encourages similar documents (by Qwen embedding) to have similar topic distributions
                contrastive_loss = self._compute_contrastive_loss(theta, doc_embeddings)
                
                # 3. KL divergence for variational inference
                # Use free bits strategy: ensure minimum KL per dimension to prevent posterior collapse
                free_bits = 0.5  # Minimum KL per topic dimension
                kl_per_dim = kl_theta_loss  # Already averaged over batch
                kl_loss = kl_weight * torch.clamp(kl_per_dim, min=free_bits)
                
                # Total loss: NLL + weighted contrastive + KL
                total_loss = nll_loss + self.contrastive_weight * contrastive_loss + kl_loss
                
                output['nll_loss'] = nll_loss
                output['recon_loss'] = nll_loss  # Alias for backward compatibility
                output['contrastive_loss'] = contrastive_loss
                output['kl_theta_loss'] = kl_theta_loss
                output['kl_loss'] = kl_loss
                output['total_loss'] = total_loss
        
        if self.dev_mode and not hasattr(self, '_forward_logged'):
            print(f"[DEV] Forward pass:")
            print(f"[DEV]   doc_embeddings shape: {doc_embeddings.shape}")
            print(f"[DEV]   bow_targets shape: {bow_targets.shape}")
            print(f"[DEV]   theta shape: {theta.shape}")
            print(f"[DEV]   beta shape: {beta.shape}")
            print(f"[DEV]   word_dist shape: {word_dist.shape}")
            self._forward_logged = True
        
        return output
    
    def get_theta(
        self,
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Get topic distribution for documents.
        
        Args:
            doc_embeddings: Document embeddings
            
        Returns:
            theta: Topic distribution (batch, K)
        """
        return self.encoder.get_theta(doc_embeddings)
    
    def get_beta(self) -> torch.Tensor:
        """
        Get topic-word distribution.
        
        Returns:
            beta: Topic-word distribution (K, V)
        """
        return self.decoder.get_beta()
    
    def get_topic_embeddings(self) -> torch.Tensor:
        """
        Get topic embedding vectors.
        
        Returns:
            Topic embeddings (K, E)
        """
        return self.decoder.get_topic_embeddings()
    
    def get_topic_words(
        self,
        top_k: int = 10,
        vocab: Optional[List[str]] = None
    ) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """
        Get top words for each topic.
        
        Args:
            top_k: Number of top words per topic
            vocab: Vocabulary list
            
        Returns:
            List of (topic_idx, [(word, prob), ...])
        """
        return self.decoder.get_topic_words(top_k, vocab)
    
    def get_all_outputs(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get all important output matrices for analysis.
        
        Args:
            doc_embeddings: All document embeddings
            bow_targets: BOW targets (optional, for loss computation)
            
        Returns:
            Dictionary with numpy arrays:
                - theta: Document-topic distribution (D x K)
                - beta: Topic-word distribution (K x V)
                - topic_embeddings: Topic vectors (K x E)
        """
        self.eval()
        with torch.no_grad():
            theta = self.get_theta(doc_embeddings)
            beta = self.get_beta()
            topic_emb = self.get_topic_embeddings()
        
        return {
            'theta': theta.cpu().numpy(),
            'beta': beta.cpu().numpy(),
            'topic_embeddings': topic_emb.cpu().numpy()
        }
    
    def compute_perplexity(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: torch.Tensor
    ) -> float:
        """
        Compute perplexity on given data.
        
        Perplexity = exp(avg negative log likelihood per word)
        
        Args:
            doc_embeddings: Document embeddings
            bow_targets: BOW targets
            
        Returns:
            Perplexity value
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(doc_embeddings, bow_targets, compute_loss=True)
            
            # 计算每个文档的NLL，然后除以每个文档的词数
            log_word_dist = output['log_word_dist']
            nll_per_doc = -torch.sum(bow_targets * log_word_dist, dim=-1)
            word_counts = bow_targets.sum(dim=-1)
            # 计算每个文档的per-word NLL，然后取平均
            perplexity = torch.exp((nll_per_doc / (word_counts + 1e-10)).mean()).item()
        
        return perplexity
        
    def compute_topic_coherence(
        self,
        bow_matrix: torch.Tensor,
        top_k: int = 10
    ) -> float:
        """
        Compute topic coherence using normalized pointwise mutual information (NPMI).
        
        Args:
            bow_matrix: BOW matrix for corpus, shape (num_docs, vocab_size)
            top_k: Number of top words per topic to consider
            
        Returns:
            Average topic coherence score
        """
        self.eval()
        with torch.no_grad():
            # Get topic-word distribution
            beta = self.get_beta().cpu().numpy()
            
            # Get top words for each topic
            top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
            
            # Convert BOW to document-term co-occurrence
            bow_np = bow_matrix.cpu().numpy()
            doc_term = (bow_np > 0).astype(np.float32)
            doc_count = doc_term.shape[0]
            
            # Compute coherence for each topic
            coherence_scores = []
            for topic_idx in range(self.num_topics):
                topic_words = top_words_indices[topic_idx]
                score = 0.0
                num_pairs = 0
                
                # Compute pairwise NPMI for top words
                for i in range(len(topic_words)):
                    for j in range(i+1, len(topic_words)):
                        word_i, word_j = topic_words[i], topic_words[j]
                        
                        # Count documents containing word i
                        count_i = doc_term[:, word_i].sum()
                        # Count documents containing word j
                        count_j = doc_term[:, word_j].sum()
                        # Count documents containing both words
                        count_ij = np.logical_and(doc_term[:, word_i], doc_term[:, word_j]).sum()
                        
                        # Avoid division by zero
                        if count_ij > 0:
                            # Compute PMI: log(p(i,j) / (p(i) * p(j)))
                            pmi = np.log((count_ij * doc_count) / (count_i * count_j))
                            # Normalize to [-1, 1]
                            npmi = pmi / (-np.log(count_ij / doc_count))
                            score += npmi
                            num_pairs += 1
                
                # Average coherence for this topic
                if num_pairs > 0:
                    coherence_scores.append(score / num_pairs)
            
            # Average coherence across all topics
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return avg_coherence
    
    def save_model(self, path: str) -> None:
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'num_topics': self.num_topics,
                'doc_embedding_dim': self.doc_embedding_dim,
                'word_embedding_dim': self.word_embedding_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> 'ETM':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            vocab_size=config['vocab_size'],
            num_topics=config['num_topics'],
            doc_embedding_dim=config['doc_embedding_dim'],
            word_embedding_dim=config['word_embedding_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
