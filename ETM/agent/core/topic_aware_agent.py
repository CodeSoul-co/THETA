"""
主题感知Agent (Topic-Aware Agent)

整合主题感知模块、认知控制模块、知识表示模块和记忆系统，
实现基于ETM的智能Agent。
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

# 导入其他模块
from agent.modules.topic_aware import TopicAwareModule
from agent.modules.cognitive_controller import CognitiveController
from agent.modules.knowledge_module import KnowledgeModule
from agent.memory.memory_system import MemorySystem
from agent.utils.llm_client import LLMClient
from agent.utils.tool_registry import ToolRegistry
from agent.utils.config import AgentConfig

from engine_c.etm import ETM
from embedding.embedder import QwenEmbedder


class TopicAwareAgent:
    """
    主题感知Agent，整合ETM模型与大语言模型，实现智能交互。
    
    架构：
    1. 主题感知模块：将文本映射到主题空间
    2. 认知控制模块：整合主题信息与大语言模型
    3. 知识表示模块：存储和检索知识
    4. 记忆系统：管理对话历史和主题演化
    """
    
    def __init__(
        self,
        config: AgentConfig,
        dev_mode: bool = False
    ):
        """
        初始化主题感知Agent。
        
        Args:
            config: Agent配置
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.config = config
        self.dev_mode = dev_mode
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Initializing with config: {config}")
        
        # 设置设备
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Using device: {self.device}")
        
        # 初始化组件
        self.etm = self._load_etm_model()
        self.embedder = self._init_embedder()
        self.topic_module = self._init_topic_module()
        self.memory = self._init_memory_system()
        self.knowledge_module = self._init_knowledge_module()
        self.llm_client = self._init_llm_client()
        self.tool_registry = self._init_tool_registry()
        self.cognitive_controller = self._init_cognitive_controller()
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Initialized successfully")
    
    def _load_etm_model(self) -> ETM:
        """加载ETM模型"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Loading ETM model from {self.config.etm_model_path}")
            
            etm = ETM.load_model(self.config.etm_model_path, self.device)
            etm.eval()  # 设置为评估模式
            return etm
        except Exception as e:
            raise RuntimeError(f"Failed to load ETM model: {e}")
    
    def _init_embedder(self) -> QwenEmbedder:
        """初始化Qwen嵌入模型"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing embedder with model: {self.config.embedding_model_path}")
            
            return QwenEmbedder(
                model_path=self.config.embedding_model_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder: {e}")
    
    def _init_topic_module(self) -> TopicAwareModule:
        """初始化主题感知模块"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing topic module")
            
            return TopicAwareModule(
                etm_model_path=self.config.etm_model_path,
                vocab_path=self.config.vocab_path,
                embedding_model_path=self.config.embedding_model_path,
                device=self.device,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize topic module: {e}")
    
    def _init_memory_system(self) -> MemorySystem:
        """初始化记忆系统"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing memory system")
            
            return MemorySystem(
                max_history_length=self.config.max_history_length,
                max_topic_history_length=self.config.max_topic_history_length,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory system: {e}")
    
    def _init_knowledge_module(self) -> KnowledgeModule:
        """初始化知识表示模块"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing knowledge module")
            
            return KnowledgeModule(
                topic_module=self.topic_module,
                embedder=self.embedder,
                vector_dim=self.config.embedding_dim,
                use_faiss=self.config.use_faiss,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize knowledge module: {e}")
    
    def _init_llm_client(self) -> LLMClient:
        """初始化大语言模型客户端"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing LLM client")
            
            return LLMClient(
                model_name=self.config.llm_model_name,
                api_key=self.config.llm_api_key,
                api_base=self.config.llm_api_base,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    def _init_tool_registry(self) -> ToolRegistry:
        """初始化工具注册表"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing tool registry")
            
            registry = ToolRegistry()
            
            # 注册默认工具
            if self.config.register_default_tools:
                registry.register_default_tools()
            
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tool registry: {e}")
    
    def _init_cognitive_controller(self) -> CognitiveController:
        """初始化认知控制模块"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing cognitive controller")
            
            return CognitiveController(
                topic_module=self.topic_module,
                memory_system=self.memory,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize cognitive controller: {e}")
    
    def process(
        self,
        user_input: str,
        session_id: str,
        use_tools: bool = True,
        use_topic_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        处理用户输入，生成响应。
        
        Args:
            user_input: 用户输入文本
            session_id: 会话ID
            use_tools: 是否使用工具
            use_topic_enhancement: 是否使用主题增强
            
        Returns:
            包含响应和元数据的字典
        """
        return self.cognitive_controller.process_input(
            user_input=user_input,
            session_id=session_id,
            use_tools=use_tools,
            use_topic_enhancement=use_topic_enhancement
        )
    
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
        return self.knowledge_module.add_document(document, embedding, topic_dist)
    
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
        return self.knowledge_module.add_documents(documents, embeddings, topic_dists)
    
    def query_knowledge(
        self,
        query_text: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        查询知识库。
        
        Args:
            query_text: 查询文本
            top_k: 返回的文档数量
            use_hybrid: 是否使用混合查询
            
        Returns:
            文档ID、相似度和文档内容的列表
        """
        if use_hybrid:
            return self.knowledge_module.hybrid_query(query_text, top_k=top_k)
        else:
            return self.knowledge_module.query_by_text(query_text, top_k=top_k)
    
    def get_topic_distribution(
        self,
        text: str
    ) -> np.ndarray:
        """
        获取文本的主题分布。
        
        Args:
            text: 输入文本
            
        Returns:
            主题分布向量
        """
        return self.topic_module.get_topic_distribution(text)
    
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
            主题索引和权重的列表
        """
        return self.topic_module.get_dominant_topics(topic_dist, top_k)
    
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
            关键词和权重的列表
        """
        return self.topic_module.get_topic_words(topic_idx, top_k)
    
    def register_tool(
        self,
        name: str,
        func: callable,
        description: str
    ) -> None:
        """
        注册工具。
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        self.tool_registry.register_tool(name, func, description)
    
    def save(
        self,
        path: str,
        save_knowledge: bool = True,
        save_memory: bool = True
    ) -> None:
        """
        保存Agent状态到文件。
        
        Args:
            path: 保存路径
            save_knowledge: 是否保存知识库
            save_memory: 是否保存记忆系统
        """
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存知识库
        if save_knowledge:
            knowledge_path = os.path.join(path, "knowledge.json")
            self.knowledge_module.save(knowledge_path)
        
        # 保存记忆系统
        if save_memory:
            memory_path = os.path.join(path, "memory.json")
            self.memory.save(memory_path)
    
    @classmethod
    def load(
        cls,
        path: str,
        dev_mode: bool = False
    ) -> 'TopicAwareAgent':
        """
        从文件加载Agent状态。
        
        Args:
            path: 加载路径
            dev_mode: 是否开启开发模式
            
        Returns:
            Agent实例
        """
        # 加载配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = AgentConfig.from_dict(config_dict)
        
        # 创建实例
        instance = cls(config, dev_mode)
        
        # 加载知识库
        knowledge_path = os.path.join(path, "knowledge.json")
        if os.path.exists(knowledge_path):
            instance.knowledge_module = KnowledgeModule.load(
                path=knowledge_path,
                topic_module=instance.topic_module,
                embedder=instance.embedder,
                use_faiss=config.use_faiss,
                dev_mode=dev_mode
            )
        
        # 加载记忆系统
        memory_path = os.path.join(path, "memory.json")
        if os.path.exists(memory_path):
            instance.memory = MemorySystem.load(
                path=memory_path,
                dev_mode=dev_mode
            )
            
            # 重新初始化认知控制模块
            instance.cognitive_controller = CognitiveController(
                topic_module=instance.topic_module,
                memory_system=instance.memory,
                llm_client=instance.llm_client,
                tool_registry=instance.tool_registry,
                dev_mode=dev_mode
            )
        
        return instance


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试主题感知Agent")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM模型路径")
    parser.add_argument("--vocab", type=str, required=True, help="词汇表路径")
    parser.add_argument("--input", type=str, required=True, help="测试输入")
    parser.add_argument("--dev_mode", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    # 创建配置
    from agent.utils.config import AgentConfig
    
    config = AgentConfig(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        embedding_model_path="/root/autodl-tmp/qwen3_embedding_0.6B",
        llm_model_name="gpt-3.5-turbo"  # 示例，实际使用时需要替换
    )
    
    # 初始化Agent
    agent = TopicAwareAgent(config, dev_mode=args.dev_mode)
    
    # 处理输入
    session_id = "test_session"
    result = agent.process(args.input, session_id)
    
    print(f"Response: {result['content']}")
    print(f"Dominant topics: {result['dominant_topics']}")
    print(f"Topic shifted: {result['topic_shifted']}")
