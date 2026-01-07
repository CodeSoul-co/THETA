"""
主题感知模块 (Topic-Aware Module)

利用ETM的主题建模能力，实现主题识别、追踪和扩展功能。
该模块是Agent的核心组件，负责将用户输入映射到主题空间。
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))

from engine_c.etm import ETM
from embedding.embedder import QwenEmbedder


class TopicAwareModule:
    """
    主题感知模块，利用ETM模型将文本映射到主题空间。
    
    功能：
    1. 主题识别：将用户输入映射到主题空间(theta)
    2. 主题追踪：跟踪对话主题变化
    3. 主题扩展：基于主题相似性扩展相关知识
    4. 主题过滤：根据主题相关性过滤信息
    """
    
    def __init__(
        self,
        etm_model_path: str,
        vocab_path: str,
        embedding_model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B",
        device: str = None,
        threshold: float = 0.1,  # 主题显著性阈值
        dev_mode: bool = False
    ):
        """
        初始化主题感知模块。
        
        Args:
            etm_model_path: ETM模型路径
            vocab_path: 词汇表路径
            embedding_model_path: Qwen嵌入模型路径
            device: 设备 ('cuda', 'cpu', 或 None 表示自动选择)
            threshold: 主题显著性阈值
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.dev_mode = dev_mode
        self.threshold = threshold
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.dev_mode:
            print(f"[TopicAwareModule] Using device: {self.device}")
            print(f"[TopicAwareModule] Loading ETM model from {etm_model_path}")
        
        # 加载ETM模型
        self.etm = self._load_etm_model(etm_model_path)
        
        # 加载词汇表
        self.vocab = self._load_vocab(vocab_path)
        
        # 初始化Qwen嵌入模型
        self.embedder = self._init_embedder(embedding_model_path)
        
        if self.dev_mode:
            print(f"[TopicAwareModule] Initialized successfully")
            print(f"[TopicAwareModule] Vocabulary size: {len(self.vocab)}")
            print(f"[TopicAwareModule] Number of topics: {self.etm.num_topics}")
    
    def _load_etm_model(self, model_path: str) -> ETM:
        """加载ETM模型"""
        try:
            etm = ETM.load_model(model_path, self.device)
            etm.eval()  # 设置为评估模式
            return etm
        except Exception as e:
            raise RuntimeError(f"Failed to load ETM model: {e}")
    
    def _load_vocab(self, vocab_path: str) -> List[str]:
        """加载词汇表"""
        try:
            if vocab_path.endswith('_list.json'):
                # 直接列表格式
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_list = json.load(f)
                return vocab_list
            else:
                # word2idx格式
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    word2idx = json.load(f)
                
                # 转换为有序列表
                vocab_size = len(word2idx)
                vocab_list = [''] * vocab_size
                for word, idx in word2idx.items():
                    vocab_list[int(idx)] = word
                
                return vocab_list
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary: {e}")
    
    def _init_embedder(self, model_path: str) -> QwenEmbedder:
        """初始化Qwen嵌入模型"""
        try:
            # 这里假设QwenEmbedder已经实现
            # 如果没有，可以使用transformers库直接实现
            return QwenEmbedder(
                model_path=model_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder: {e}")
    
    def get_topic_distribution(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        获取文本的主题分布。
        
        Args:
            text: 输入文本
            normalize: 是否归一化主题分布
            
        Returns:
            主题分布向量 (num_topics,)
        """
        # 获取文本的Qwen嵌入
        embedding = self.embedder.embed_text(text)
        
        # 转换为张量并添加批次维度
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 使用ETM编码器获取主题分布
        with torch.no_grad():
            theta = self.etm.get_theta(embedding_tensor)
        
        # 转换为numpy数组
        theta_np = theta.squeeze().cpu().numpy()
        
        return theta_np
    
    def get_dominant_topics(
        self,
        topic_dist: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        获取主导主题。
        
        Args:
            topic_dist: 主题分布向量
            top_k: 返回的主题数量
            
        Returns:
            主题索引和权重的列表 [(topic_idx, weight), ...]
        """
        # 获取前k个主题
        top_indices = np.argsort(-topic_dist)[:top_k]
        
        # 过滤掉权重低于阈值的主题
        dominant_topics = [
            (idx, topic_dist[idx]) 
            for idx in top_indices 
            if topic_dist[idx] > self.threshold
        ]
        
        return dominant_topics
    
    def get_topic_words(
        self,
        topic_idx: int,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取主题的关键词。
        
        Args:
            topic_idx: 主题索引
            top_k: 返回的关键词数量
            
        Returns:
            关键词和权重的列表 [(word, weight), ...]
        """
        # 使用ETM获取主题词
        topic_words = self.etm.get_topic_words(top_k=top_k, vocab=self.vocab)
        
        # 返回指定主题的词
        return topic_words[topic_idx][1]
    
    def get_topic_similarity(
        self,
        topic_dist1: np.ndarray,
        topic_dist2: np.ndarray
    ) -> float:
        """
        计算两个主题分布的相似度。
        
        Args:
            topic_dist1: 第一个主题分布
            topic_dist2: 第二个主题分布
            
        Returns:
            余弦相似度 (0-1)
        """
        # 计算余弦相似度
        dot_product = np.dot(topic_dist1, topic_dist2)
        norm1 = np.linalg.norm(topic_dist1)
        norm2 = np.linalg.norm(topic_dist2)
        
        # 避免除零
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def detect_topic_shift(
        self,
        prev_topic_dist: np.ndarray,
        curr_topic_dist: np.ndarray,
        threshold: float = 0.7
    ) -> bool:
        """
        检测主题是否发生显著变化。
        
        Args:
            prev_topic_dist: 前一个主题分布
            curr_topic_dist: 当前主题分布
            threshold: 相似度阈值，低于此值视为主题变化
            
        Returns:
            是否发生主题变化
        """
        similarity = self.get_topic_similarity(prev_topic_dist, curr_topic_dist)
        return similarity < threshold
    
    def get_topic_context(
        self,
        topic_indices: List[int],
        words_per_topic: int = 5
    ) -> Dict[str, Any]:
        """
        获取主题上下文信息，用于增强提示。
        
        Args:
            topic_indices: 主题索引列表
            words_per_topic: 每个主题返回的关键词数量
            
        Returns:
            主题上下文信息
        """
        context = {
            "topics": []
        }
        
        for topic_idx in topic_indices:
            topic_words = self.get_topic_words(topic_idx, top_k=words_per_topic)
            
            context["topics"].append({
                "id": topic_idx,
                "keywords": [word for word, _ in topic_words],
                "weights": [float(weight) for _, weight in topic_words]
            })
        
        return context
    
    def enrich_prompt(
        self,
        prompt: str,
        topic_dist: np.ndarray,
        top_k_topics: int = 2,
        words_per_topic: int = 5
    ) -> str:
        """
        使用主题信息增强提示。
        
        Args:
            prompt: 原始提示
            topic_dist: 主题分布
            top_k_topics: 使用的主题数量
            words_per_topic: 每个主题的关键词数量
            
        Returns:
            增强后的提示
        """
        # 获取主导主题
        dominant_topics = self.get_dominant_topics(topic_dist, top_k=top_k_topics)
        
        if not dominant_topics:
            return prompt
        
        # 构建主题上下文
        topic_context = "相关主题上下文:\n"
        
        for topic_idx, weight in dominant_topics:
            topic_words = self.get_topic_words(topic_idx, top_k=words_per_topic)
            words_str = ", ".join([word for word, _ in topic_words])
            topic_context += f"- 主题 {topic_idx} (权重: {weight:.2f}): {words_str}\n"
        
        # 增强提示
        enhanced_prompt = f"{prompt}\n\n{topic_context}"
        
        return enhanced_prompt


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试主题感知模块")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM模型路径")
    parser.add_argument("--vocab", type=str, required=True, help="词汇表路径")
    parser.add_argument("--text", type=str, required=True, help="测试文本")
    parser.add_argument("--dev_mode", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    # 初始化模块
    topic_module = TopicAwareModule(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        dev_mode=args.dev_mode
    )
    
    # 获取主题分布
    topic_dist = topic_module.get_topic_distribution(args.text)
    print(f"Topic distribution: {topic_dist}")
    
    # 获取主导主题
    dominant_topics = topic_module.get_dominant_topics(topic_dist)
    print(f"Dominant topics: {dominant_topics}")
    
    # 获取主题词
    for topic_idx, weight in dominant_topics:
        topic_words = topic_module.get_topic_words(topic_idx)
        print(f"Topic {topic_idx} (weight: {weight:.2f}): {topic_words}")
    
    # 增强提示
    enhanced_prompt = topic_module.enrich_prompt(args.text, topic_dist)
    print(f"\nEnhanced prompt:\n{enhanced_prompt}")
