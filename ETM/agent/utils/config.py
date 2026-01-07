"""
Agent配置类 (Agent Configuration)

定义Agent的配置参数，包括模型路径、参数设置等。
"""

import os
from typing import Dict, Any, Optional


class AgentConfig:
    """
    Agent配置类，定义Agent的配置参数。
    """
    
    def __init__(
        self,
        etm_model_path: str,
        vocab_path: str,
        embedding_model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B",
        embedding_dim: int = 1024,
        device: Optional[str] = None,
        max_history_length: int = 10,
        max_topic_history_length: int = 5,
        use_faiss: bool = True,
        llm_model_name: str = "gpt-3.5-turbo",
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        register_default_tools: bool = True
    ):
        """
        初始化Agent配置。
        
        Args:
            etm_model_path: ETM模型路径
            vocab_path: 词汇表路径
            embedding_model_path: Qwen嵌入模型路径
            embedding_dim: 嵌入维度
            device: 设备 ('cuda', 'cpu', 或 None 表示自动选择)
            max_history_length: 最大对话历史长度
            max_topic_history_length: 最大主题历史长度
            use_faiss: 是否使用FAISS进行向量检索
            llm_model_name: 大语言模型名称
            llm_api_key: 大语言模型API密钥
            llm_api_base: 大语言模型API基础URL
            register_default_tools: 是否注册默认工具
        """
        self.etm_model_path = etm_model_path
        self.vocab_path = vocab_path
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.device = device
        self.max_history_length = max_history_length
        self.max_topic_history_length = max_topic_history_length
        self.use_faiss = use_faiss
        self.llm_model_name = llm_model_name
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.register_default_tools = register_default_tools
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典。
        
        Returns:
            配置字典
        """
        return {
            "etm_model_path": self.etm_model_path,
            "vocab_path": self.vocab_path,
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_history_length": self.max_history_length,
            "max_topic_history_length": self.max_topic_history_length,
            "use_faiss": self.use_faiss,
            "llm_model_name": self.llm_model_name,
            "llm_api_key": self.llm_api_key,
            "llm_api_base": self.llm_api_base,
            "register_default_tools": self.register_default_tools
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        从字典创建配置。
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置实例
        """
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """
        从环境变量创建配置。
        
        Returns:
            配置实例
        """
        return cls(
            etm_model_path=os.environ.get("ETM_MODEL_PATH", ""),
            vocab_path=os.environ.get("VOCAB_PATH", ""),
            embedding_model_path=os.environ.get("EMBEDDING_MODEL_PATH", "/root/autodl-tmp/qwen3_embedding_0.6B"),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1024")),
            device=os.environ.get("DEVICE", None),
            max_history_length=int(os.environ.get("MAX_HISTORY_LENGTH", "10")),
            max_topic_history_length=int(os.environ.get("MAX_TOPIC_HISTORY_LENGTH", "5")),
            use_faiss=os.environ.get("USE_FAISS", "True").lower() == "true",
            llm_model_name=os.environ.get("LLM_MODEL_NAME", "gpt-3.5-turbo"),
            llm_api_key=os.environ.get("LLM_API_KEY", None),
            llm_api_base=os.environ.get("LLM_API_BASE", None),
            register_default_tools=os.environ.get("REGISTER_DEFAULT_TOOLS", "True").lower() == "true"
        )
