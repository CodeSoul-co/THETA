"""
LDA (Latent Dirichlet Allocation) - 经典概率主题模型

基于神经网络的LDA实现，与ETM接口保持一致。

TODO: 这是伪代码框架，需要后续完善实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np


class LDAEncoder(nn.Module):
    """
    LDA编码器 - 从BOW推断主题分布
    
    与ETM不同，LDA直接使用BOW作为输入（不需要预训练embedding）
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_topics: int = 20,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        
        # BOW编码器
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VAE参数
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.logvar = nn.Linear(hidden_dim, num_topics)
    
    def forward(self, bow: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            bow: BOW矩阵 (batch, vocab_size)
            
        Returns:
            theta: 主题分布 (batch, num_topics)
            mu: 均值
            logvar: 对数方差
        """
        # 归一化BOW
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # 编码
        hidden = self.encoder(bow_normalized)
        
        # VAE参数
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        
        # 重参数化
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # Softmax得到主题分布
        theta = F.softmax(z, dim=-1)
        
        return theta, mu, logvar


class LDADecoder(nn.Module):
    """
    LDA解码器 - 从主题分布生成词分布
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        
        # 主题-词分布 (可学习参数)
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
    
    def get_beta(self) -> torch.Tensor:
        """获取归一化的主题-词分布"""
        return F.softmax(self.beta, dim=-1)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            theta: 主题分布 (batch, num_topics)
            
        Returns:
            word_dist: 词分布 (batch, vocab_size)
        """
        beta = self.get_beta()  # (num_topics, vocab_size)
        word_dist = torch.mm(theta, beta)  # (batch, vocab_size)
        return word_dist


class LDA(nn.Module):
    """
    Neural LDA - 基于神经网络的LDA实现
    
    主要特点:
    1. 不需要预训练embedding
    2. 直接使用BOW作为输入
    3. 经典的主题模型，适合基线对比
    
    与ETM的接口保持一致，方便统一调用
    
    TODO: 完善以下功能
    - [ ] Dirichlet先验
    - [ ] 在线学习
    - [ ] 层次LDA
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        encoder_dropout: float = 0.2,
        alpha: float = 0.1,  # Dirichlet先验参数
        kl_weight: float = 1.0,
        dev_mode: bool = False,
        # 以下参数为了接口兼容，LDA不使用
        doc_embedding_dim: int = None,
        word_embedding_dim: int = None,
        word_embeddings: torch.Tensor = None,
        train_word_embeddings: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        
        # 编码器
        self.encoder = LDAEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            dropout=encoder_dropout
        )
        
        # 解码器
        self.decoder = LDADecoder(
            vocab_size=vocab_size,
            num_topics=num_topics
        )
        
        if self.dev_mode:
            print(f"[DEV] LDA initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   Note: LDA does not use embeddings")
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,  # 为了接口兼容，但LDA不使用
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            doc_embeddings: 文档embedding (LDA不使用，仅为接口兼容)
            bow: BOW矩阵 (batch, vocab_size)
            
        Returns:
            Dict containing:
                - loss: 总损失
                - recon_loss: 重建损失
                - kl_loss: KL散度损失
                - theta: 主题分布
        """
        # LDA直接使用BOW，忽略doc_embeddings
        theta, mu, logvar = self.encoder(bow)
        
        # 解码
        word_dist = self.decoder(theta)
        
        # 重建损失
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        recon_loss = -torch.sum(bow_normalized * torch.log(word_dist + 1e-10), dim=1).mean()
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # 总损失
        loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'theta': theta
        }
    
    def get_beta(self) -> torch.Tensor:
        """获取主题-词分布"""
        return self.decoder.get_beta()
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        获取主题词
        
        Args:
            vocab: 词汇表
            top_k: 每个主题返回的词数
            
        Returns:
            主题词字典 {topic_id: [word1, word2, ...]}
        """
        beta = self.get_beta()  # (num_topics, vocab_size)
        
        topic_words = {}
        for k in range(self.num_topics):
            top_indices = torch.topk(beta[k], top_k).indices.cpu().numpy()
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words


# ============================================================================
# 工厂函数
# ============================================================================

def create_lda(
    vocab_size: int,
    num_topics: int = 20,
    **kwargs
) -> LDA:
    """
    创建LDA模型的工厂函数
    """
    return LDA(
        vocab_size=vocab_size,
        num_topics=num_topics,
        **kwargs
    )
