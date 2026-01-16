"""
DTM (Dynamic Topic Model) - 动态主题模型

支持时间序列的主题模型，可以追踪主题随时间的演化。

TODO: 这是伪代码框架，需要后续完善实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np


class DTMEncoder(nn.Module):
    """
    DTM编码器 - 将文档embedding映射到主题分布
    
    与ETM编码器类似，但增加了时间信息的处理
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_topics: int = 20,
        time_slices: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        self.time_slices = time_slices
        
        # 文档编码器
        self.doc_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 时间编码 (可学习的时间embedding)
        self.time_embedding = nn.Embedding(time_slices, hidden_dim)
        
        # 融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 输出层 (VAE风格)
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.logvar = nn.Linear(hidden_dim, num_topics)
    
    def forward(
        self, 
        doc_embedding: torch.Tensor,
        time_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            doc_embedding: 文档embedding (batch, input_dim)
            time_index: 时间索引 (batch,)
            
        Returns:
            theta: 主题分布 (batch, num_topics)
            mu: 均值
            logvar: 对数方差
        """
        # 编码文档
        doc_hidden = self.doc_encoder(doc_embedding)
        
        # 获取时间embedding
        time_hidden = self.time_embedding(time_index)
        
        # 融合
        combined = torch.cat([doc_hidden, time_hidden], dim=-1)
        hidden = F.relu(self.fusion(combined))
        
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


class DTMDecoder(nn.Module):
    """
    DTM解码器 - 从主题分布重建词分布
    
    主题-词分布随时间变化
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        time_slices: int = 10,
        embedding_dim: int = 1024,
        word_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.time_slices = time_slices
        self.embedding_dim = embedding_dim
        
        # 词向量
        if word_embeddings is not None:
            self.word_embeddings = nn.Parameter(word_embeddings, requires_grad=False)
        else:
            self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        
        # 时间相关的主题embedding
        # 每个时间片有自己的主题向量
        self.topic_embeddings = nn.Parameter(
            torch.randn(time_slices, num_topics, embedding_dim)
        )
        
        # 主题演化网络 (可选: 建模主题随时间的平滑变化)
        self.topic_evolution = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
    
    def get_beta(self, time_index: int = None) -> torch.Tensor:
        """
        获取主题-词分布
        
        Args:
            time_index: 时间索引，None则返回所有时间的beta
            
        Returns:
            beta: (num_topics, vocab_size) 或 (time_slices, num_topics, vocab_size)
        """
        if time_index is not None:
            # 特定时间的主题embedding
            topic_emb = self.topic_embeddings[time_index]  # (num_topics, embedding_dim)
            # 计算与词向量的相似度
            beta = torch.mm(topic_emb, self.word_embeddings.t())  # (num_topics, vocab_size)
            beta = F.softmax(beta, dim=-1)
            return beta
        else:
            # 所有时间的beta
            betas = []
            for t in range(self.time_slices):
                topic_emb = self.topic_embeddings[t]
                beta = torch.mm(topic_emb, self.word_embeddings.t())
                beta = F.softmax(beta, dim=-1)
                betas.append(beta)
            return torch.stack(betas, dim=0)  # (time_slices, num_topics, vocab_size)
    
    def forward(
        self,
        theta: torch.Tensor,
        time_index: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            theta: 主题分布 (batch, num_topics)
            time_index: 时间索引 (batch,)
            
        Returns:
            word_dist: 词分布 (batch, vocab_size)
        """
        batch_size = theta.size(0)
        word_dists = []
        
        for i in range(batch_size):
            t = time_index[i].item()
            beta = self.get_beta(t)  # (num_topics, vocab_size)
            word_dist = torch.mm(theta[i:i+1], beta)  # (1, vocab_size)
            word_dists.append(word_dist)
        
        return torch.cat(word_dists, dim=0)


class DTM(nn.Module):
    """
    Dynamic Topic Model - 动态主题模型
    
    主要特点:
    1. 支持时间序列数据
    2. 主题随时间演化
    3. 可以追踪主题的变化趋势
    
    与ETM的接口保持一致，方便统一调用
    
    TODO: 完善以下功能
    - [ ] 主题演化的平滑约束
    - [ ] 时间序列预测
    - [ ] 主题生命周期分析
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        time_slices: int = 10,
        doc_embedding_dim: int = 1024,
        word_embedding_dim: int = 1024,
        hidden_dim: int = 512,
        encoder_dropout: float = 0.2,
        word_embeddings: Optional[torch.Tensor] = None,
        train_word_embeddings: bool = False,
        kl_weight: float = 0.5,
        evolution_weight: float = 0.1,  # 主题演化平滑约束权重
        dev_mode: bool = False,
        **kwargs  # 接收额外参数，保持接口兼容
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.time_slices = time_slices
        self.doc_embedding_dim = doc_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.evolution_weight = evolution_weight
        self.dev_mode = dev_mode
        
        # 编码器
        self.encoder = DTMEncoder(
            input_dim=doc_embedding_dim,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            time_slices=time_slices,
            dropout=encoder_dropout
        )
        
        # 解码器
        self.decoder = DTMDecoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            time_slices=time_slices,
            embedding_dim=word_embedding_dim,
            word_embeddings=word_embeddings
        )
        
        if self.dev_mode:
            print(f"[DEV] DTM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   time_slices={time_slices}")
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            doc_embeddings: 文档embedding (batch, doc_embedding_dim)
            bow: BOW矩阵 (batch, vocab_size)
            time_indices: 时间索引 (batch,)，如果为None则假设所有文档在同一时间
            
        Returns:
            Dict containing:
                - loss: 总损失
                - recon_loss: 重建损失
                - kl_loss: KL散度损失
                - evolution_loss: 主题演化平滑损失
                - theta: 主题分布
        """
        batch_size = doc_embeddings.size(0)
        
        # 如果没有时间索引，默认为0
        if time_indices is None:
            time_indices = torch.zeros(batch_size, dtype=torch.long, device=doc_embeddings.device)
        
        # 编码
        theta, mu, logvar = self.encoder(doc_embeddings, time_indices)
        
        # 解码
        word_dist = self.decoder(theta, time_indices)
        
        # 重建损失 (负对数似然)
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        recon_loss = -torch.sum(bow_normalized * torch.log(word_dist + 1e-10), dim=1).mean()
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # 主题演化平滑损失 (相邻时间片的主题应该相似)
        evolution_loss = self._compute_evolution_loss()
        
        # 总损失
        loss = recon_loss + self.kl_weight * kl_loss + self.evolution_weight * evolution_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'evolution_loss': evolution_loss,
            'theta': theta
        }
    
    def _compute_evolution_loss(self) -> torch.Tensor:
        """
        计算主题演化平滑损失
        
        鼓励相邻时间片的主题embedding相似
        """
        topic_emb = self.decoder.topic_embeddings  # (time_slices, num_topics, embedding_dim)
        
        if self.time_slices < 2:
            return torch.tensor(0.0, device=topic_emb.device)
        
        # 计算相邻时间片的差异
        diff = topic_emb[1:] - topic_emb[:-1]  # (time_slices-1, num_topics, embedding_dim)
        evolution_loss = torch.mean(diff.pow(2))
        
        return evolution_loss
    
    def get_beta(self, time_index: int = None) -> torch.Tensor:
        """获取主题-词分布"""
        return self.decoder.get_beta(time_index)
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10,
        time_index: int = None
    ) -> Dict[str, List[str]]:
        """
        获取主题词
        
        Args:
            vocab: 词汇表
            top_k: 每个主题返回的词数
            time_index: 时间索引，None则返回最后一个时间片
            
        Returns:
            主题词字典 {topic_id: [word1, word2, ...]}
        """
        if time_index is None:
            time_index = self.time_slices - 1
        
        beta = self.get_beta(time_index)  # (num_topics, vocab_size)
        
        topic_words = {}
        for k in range(self.num_topics):
            top_indices = torch.topk(beta[k], top_k).indices.cpu().numpy()
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words
    
    def get_topic_evolution(
        self,
        vocab: List[str],
        topic_id: int,
        top_k: int = 10
    ) -> Dict[int, List[str]]:
        """
        获取特定主题随时间的演化
        
        Args:
            vocab: 词汇表
            topic_id: 主题ID
            top_k: 每个时间片返回的词数
            
        Returns:
            {time_index: [word1, word2, ...]}
        """
        evolution = {}
        
        for t in range(self.time_slices):
            beta = self.get_beta(t)  # (num_topics, vocab_size)
            top_indices = torch.topk(beta[topic_id], top_k).indices.cpu().numpy()
            evolution[t] = [vocab[i] for i in top_indices]
        
        return evolution


# ============================================================================
# 工厂函数 - 方便从registry调用
# ============================================================================

def create_dtm(
    vocab_size: int,
    num_topics: int = 20,
    word_embeddings: Optional[torch.Tensor] = None,
    **kwargs
) -> DTM:
    """
    创建DTM模型的工厂函数
    
    Args:
        vocab_size: 词表大小
        num_topics: 主题数
        word_embeddings: 预训练词向量
        **kwargs: 其他参数
        
    Returns:
        DTM模型实例
    """
    return DTM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        word_embeddings=word_embeddings,
        **kwargs
    )
