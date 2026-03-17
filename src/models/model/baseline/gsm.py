"""
Gaussian Softmax Model (GSM)

Improvement over NVDM: applies softmax to z to get proper probability distribution.
This makes topic proportions interpretable (non-negative, sum to 1).

Reference:
- Miao et al., "Discovering Discrete Latent Topics with Neural Variational Inference", ICML 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Dict, Optional, List, Any, Tuple
import numpy as np

from ..base import NeuralTopicModel


def kld_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence to standard normal distribution.
    
    Args:
        mu: Mean, shape (batch_size, dim)
        log_sigma: Log standard deviation, shape (batch_size, dim)
    
    Returns:
        KL divergence, shape (batch_size,)
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class NormalParameter(nn.Module):
    """Parameterizes a normal distribution with mean and log_sigma."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self._reset_parameters()
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.log_sigma(h)
    
    def _reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


class TopicDecoder(nn.Module):
    """Decodes topic distribution to word distribution."""
    
    def __init__(self, num_topics: int, vocab_size: int, bias: bool = True):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.topic_word = nn.Linear(num_topics, vocab_size, bias=bias)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: Topic distribution, shape (batch_size, num_topics)
        
        Returns:
            Log probability over vocabulary, shape (batch_size, vocab_size)
        """
        return F.log_softmax(self.topic_word(theta), dim=-1)
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution (beta matrix)."""
        return F.softmax(self.topic_word.weight.data.T, dim=-1)


class GSM(NeuralTopicModel):
    """
    Gaussian Softmax Model (GSM)
    
    Improvement over NVDM: applies softmax to z to get proper probability distribution.
    This makes topic proportions interpretable (non-negative, sum to 1).
    
    Architecture:
        BOW -> Encoder -> (mu, log_sigma) -> z -> softmax(z) -> Decoder -> word_dist
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Normal parameters
        self.normal = NormalParameter(hidden_dim, num_topics)
        
        # Decoder
        self.decoder = TopicDecoder(num_topics, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode BOW to latent distribution parameters."""
        h = self.encoder(bow)
        mu, log_sigma = self.normal(h)
        return mu, log_sigma
    
    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        bow: torch.Tensor,
        doc_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Normalize BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        mu, log_sigma = self.encode(bow_norm)
        
        # Sample z and apply softmax to get theta
        z = self.reparameterize(mu, log_sigma)
        theta = F.softmax(z, dim=-1)
        theta = self.dropout(theta)
        
        # Decode
        log_prob = self.decoder(theta)
        
        # Losses
        recon_loss = -(bow_norm * log_prob).sum(dim=-1)
        kl_loss = kld_normal(mu, log_sigma)
        
        return {
            'theta': theta,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'log_sigma': log_sigma
        }
    
    def get_theta(self, bow: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get topic distribution for documents."""
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        mu, _ = self.encode(bow_norm)
        return F.softmax(mu, dim=-1)
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution."""
        return self.decoder.get_beta()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate
        }
    
    def get_topic_words(self, vocab: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """Get top words for each topic."""
        beta = self.get_beta().cpu().numpy()
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        return topic_words


def create_gsm(vocab_size: int, num_topics: int = 20, **kwargs) -> GSM:
    """Create GSM model."""
    return GSM(vocab_size=vocab_size, num_topics=num_topics, **kwargs)
