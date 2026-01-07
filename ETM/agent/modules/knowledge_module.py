"""
知识表示模块 (Knowledge Representation Module)

多模态知识表示，包括主题空间、语义向量和符号知识。
该模块负责知识的存储、检索和组织。
"""

import os
import sys
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))

# 导入其他模块
from agent.modules.topic_aware import TopicAwareModule
from embedding.embedder import QwenEmbedder


class KnowledgeModule:
    """
    知识表示模块，负责知识的存储、检索和组织。
    
    功能：
    1. 主题空间：ETM生成的主题分布(theta)
    2. 语义向量：Qwen生成的文档嵌入
    3. 符号知识：结构化知识和规则
    """
    
    def __init__(
        self,
        topic_module: TopicAwareModule,
        embedder: QwenEmbedder,
        vector_dim: int = 1024,
        use_faiss: bool = True,
        dev_mode: bool = False
    ):
        """
        初始化知识表示模块。
        
        Args:
            topic_module: 主题感知模块
            embedder: 嵌入模型
            vector_dim: 向量维度
            use_faiss: 是否使用FAISS进行向量检索
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.topic_module = topic_module
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.use_faiss = use_faiss
        self.dev_mode = dev_mode
        
        # 文档存储
        self.documents = []
        self.document_embeddings = []
        self.document_topic_dists = []
        
        # 初始化向量索引
        if use_faiss:
            self.index = faiss.IndexFlatIP(vector_dim)  # 内积相似度（余弦相似度）
        else:
            self.index = None
        
        if self.dev_mode:
            print(f"[KnowledgeModule] Initialized successfully")
    
    def add_document(
        self,
        document: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
        topic_dist: Optional[np.ndarray] = None
    ) -> int:
        """
        添加文档到知识库。
        
        Args:
            document: 文档内容
            embedding: 文档嵌入向量（可选）
            topic_dist: 文档主题分布（可选）
            
        Returns:
            文档ID
        """
        # 生成嵌入（如果未提供）
        if embedding is None and "text" in document:
            embedding = self.embedder.embed_text(document["text"])
        
        # 生成主题分布（如果未提供）
        if topic_dist is None and embedding is not None:
            topic_dist = self.topic_module.get_topic_distribution(
                document["text"] if "text" in document else ""
            )
        
        # 添加到存储
        doc_id = len(self.documents)
        self.documents.append(document)
        
        if embedding is not None:
            self.document_embeddings.append(embedding)
            
            # 添加到FAISS索引
            if self.use_faiss:
                # 归一化向量
                norm_embedding = embedding / np.linalg.norm(embedding)
                self.index.add(np.array([norm_embedding], dtype=np.float32))
        
        if topic_dist is not None:
            self.document_topic_dists.append(topic_dist)
        
        return doc_id
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None,
        topic_dists: Optional[List[np.ndarray]] = None
    ) -> List[int]:
        """
        批量添加文档到知识库。
        
        Args:
            documents: 文档列表
            embeddings: 嵌入向量列表（可选）
            topic_dists: 主题分布列表（可选）
            
        Returns:
            文档ID列表
        """
        doc_ids = []
        
        # 批量生成嵌入（如果未提供）
        if embeddings is None:
            texts = [doc.get("text", "") for doc in documents]
            embeddings = self.embedder.embed_texts(texts)
        
        # 批量生成主题分布（如果未提供）
        if topic_dists is None:
            topic_dists = []
            for i, doc in enumerate(documents):
                topic_dist = self.topic_module.get_topic_distribution(
                    doc.get("text", "")
                )
                topic_dists.append(topic_dist)
        
        # 批量添加到存储
        for i, doc in enumerate(documents):
            embedding = embeddings[i] if i < len(embeddings) else None
            topic_dist = topic_dists[i] if i < len(topic_dists) else None
            
            doc_id = self.add_document(doc, embedding, topic_dist)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def query_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        根据向量查询相关文档。
        
        Args:
            query_vector: 查询向量
            top_k: 返回的文档数量
            
        Returns:
            文档ID、相似度和文档内容的列表
        """
        if len(self.documents) == 0:
            return []
        
        if self.use_faiss and self.index.ntotal > 0:
            # 归一化查询向量
            norm_query = query_vector / np.linalg.norm(query_vector)
            
            # 使用FAISS查询
            scores, indices = self.index.search(
                np.array([norm_query], dtype=np.float32),
                min(top_k, len(self.documents))
            )
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append((
                        idx,
                        float(scores[0][i]),
                        self.documents[idx]
                    ))
            
            return results
        else:
            # 手动计算相似度
            similarities = []
            
            for i, emb in enumerate(self.document_embeddings):
                # 计算余弦相似度
                norm_query = query_vector / np.linalg.norm(query_vector)
                norm_emb = emb / np.linalg.norm(emb)
                similarity = np.dot(norm_query, norm_emb)
                
                similarities.append((i, similarity))
            
            # 排序并返回前k个
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = [
                (idx, sim, self.documents[idx])
                for idx, sim in similarities[:top_k]
            ]
            
            return results
    
    def query_by_text(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        根据文本查询相关文档。
        
        Args:
            query_text: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            文档ID、相似度和文档内容的列表
        """
        # 生成查询向量
        query_vector = self.embedder.embed_text(query_text)
        
        # 使用向量查询
        return self.query_by_vector(query_vector, top_k)
    
    def query_by_topic(
        self,
        topic_dist: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        根据主题分布查询相关文档。
        
        Args:
            topic_dist: 主题分布
            top_k: 返回的文档数量
            
        Returns:
            文档ID、相似度和文档内容的列表
        """
        if len(self.documents) == 0 or len(self.document_topic_dists) == 0:
            return []
        
        # 计算主题相似度
        similarities = []
        
        for i, doc_topic in enumerate(self.document_topic_dists):
            # 计算余弦相似度
            similarity = self.topic_module.get_topic_similarity(topic_dist, doc_topic)
            similarities.append((i, similarity))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            (idx, sim, self.documents[idx])
            for idx, sim in similarities[:top_k]
        ]
        
        return results
    
    def hybrid_query(
        self,
        query_text: str,
        topic_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        混合查询（结合主题和语义）。
        
        Args:
            query_text: 查询文本
            topic_weight: 主题相似度权重
            semantic_weight: 语义相似度权重
            top_k: 返回的文档数量
            
        Returns:
            文档ID、相似度和文档内容的列表
        """
        if len(self.documents) == 0:
            return []
        
        # 生成查询向量
        query_vector = self.embedder.embed_text(query_text)
        
        # 生成主题分布
        topic_dist = self.topic_module.get_topic_distribution(query_text)
        
        # 计算混合相似度
        hybrid_scores = {}
        
        # 语义相似度
        semantic_results = self.query_by_vector(query_vector, len(self.documents))
        for idx, score, _ in semantic_results:
            hybrid_scores[idx] = semantic_weight * score
        
        # 主题相似度
        topic_results = self.query_by_topic(topic_dist, len(self.documents))
        for idx, score, _ in topic_results:
            if idx in hybrid_scores:
                hybrid_scores[idx] += topic_weight * score
            else:
                hybrid_scores[idx] = topic_weight * score
        
        # 排序并返回前k个
        sorted_scores = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = [
            (idx, score, self.documents[idx])
            for idx, score in sorted_scores[:top_k]
        ]
        
        return results
    
    def save(self, path: str) -> None:
        """
        保存知识库到文件。
        
        Args:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存文档
        data = {
            "documents": self.documents,
            "document_embeddings": [emb.tolist() for emb in self.document_embeddings],
            "document_topic_dists": [dist.tolist() for dist in self.document_topic_dists]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 如果使用FAISS，保存索引
        if self.use_faiss and self.index is not None:
            index_path = f"{path}.index"
            faiss.write_index(self.index, index_path)
    
    @classmethod
    def load(
        cls,
        path: str,
        topic_module: TopicAwareModule,
        embedder: QwenEmbedder,
        use_faiss: bool = True,
        dev_mode: bool = False
    ) -> 'KnowledgeModule':
        """
        从文件加载知识库。
        
        Args:
            path: 加载路径
            topic_module: 主题感知模块
            embedder: 嵌入模型
            use_faiss: 是否使用FAISS
            dev_mode: 是否开启开发模式
            
        Returns:
            知识表示模块实例
        """
        # 创建实例
        instance = cls(
            topic_module=topic_module,
            embedder=embedder,
            use_faiss=use_faiss,
            dev_mode=dev_mode
        )
        
        # 加载数据
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        instance.documents = data["documents"]
        instance.document_embeddings = [np.array(emb) for emb in data["document_embeddings"]]
        instance.document_topic_dists = [np.array(dist) for dist in data["document_topic_dists"]]
        
        # 如果使用FAISS，加载索引
        if use_faiss:
            index_path = f"{path}.index"
            if os.path.exists(index_path):
                instance.index = faiss.read_index(index_path)
            else:
                # 重建索引
                instance.index = faiss.IndexFlatIP(instance.vector_dim)
                
                for emb in instance.document_embeddings:
                    norm_emb = emb / np.linalg.norm(emb)
                    instance.index.add(np.array([norm_emb], dtype=np.float32))
        
        return instance


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试知识表示模块")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM模型路径")
    parser.add_argument("--vocab", type=str, required=True, help="词汇表路径")
    parser.add_argument("--query", type=str, required=True, help="测试查询")
    parser.add_argument("--dev_mode", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    # 初始化组件
    from agent.modules.topic_aware import TopicAwareModule
    
    topic_module = TopicAwareModule(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        dev_mode=args.dev_mode
    )
    
    embedder = QwenEmbedder()
    
    # 初始化知识表示模块
    knowledge_module = KnowledgeModule(
        topic_module=topic_module,
        embedder=embedder,
        dev_mode=args.dev_mode
    )
    
    # 添加测试文档
    test_docs = [
        {"id": "doc1", "text": "气候变化是当今世界面临的最大环境挑战之一。"},
        {"id": "doc2", "text": "可再生能源包括太阳能、风能和水力发电。"},
        {"id": "doc3", "text": "碳排放交易是减少温室气体排放的市场机制。"},
        {"id": "doc4", "text": "能源转型是指从化石燃料向清洁能源的转变过程。"},
        {"id": "doc5", "text": "煤炭是一种非可再生的化石燃料，燃烧时会释放二氧化碳。"}
    ]
    
    knowledge_module.add_documents(test_docs)
    
    # 测试查询
    print(f"Query: {args.query}")
    
    # 语义查询
    semantic_results = knowledge_module.query_by_text(args.query)
    print("\nSemantic search results:")
    for idx, score, doc in semantic_results:
        print(f"  {doc['id']} (score: {score:.4f}): {doc['text']}")
    
    # 主题查询
    topic_dist = topic_module.get_topic_distribution(args.query)
    topic_results = knowledge_module.query_by_topic(topic_dist)
    print("\nTopic search results:")
    for idx, score, doc in topic_results:
        print(f"  {doc['id']} (score: {score:.4f}): {doc['text']}")
    
    # 混合查询
    hybrid_results = knowledge_module.hybrid_query(args.query)
    print("\nHybrid search results:")
    for idx, score, doc in hybrid_results:
        print(f"  {doc['id']} (score: {score:.4f}): {doc['text']}")
