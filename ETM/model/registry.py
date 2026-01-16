"""
Topic Model Registry - 主题模型注册表

方便前后端人员选择不同的主题模型。
未来可以在这里添加更多模型（如LDA、NTM等）。

Usage:
    from model.registry import get_topic_model_options, get_model_class
    
    # 获取所有可用模型（用于前端下拉框）
    options = get_topic_model_options()
    
    # 获取模型类
    ModelClass = get_model_class("etm")
"""

from typing import Dict, Any, Optional, Type
from dataclasses import dataclass


@dataclass
class TopicModelInfo:
    """主题模型信息"""
    name: str                           # 显示名称
    description: str                    # 模型描述
    module_path: str                    # 模块路径
    class_name: str                     # 类名
    supports_embeddings: bool           # 是否支持预训练embedding
    supports_pretrained_words: bool     # 是否支持预训练词向量
    default_params: Dict[str, Any]      # 默认参数
    param_options: Dict[str, list]      # 参数可选值


# ============================================================================
# 模型注册表 - 在这里添加新模型
# ============================================================================

TOPIC_MODEL_REGISTRY: Dict[str, TopicModelInfo] = {
    "etm": TopicModelInfo(
        name="ETM (Embedded Topic Model)",
        description="基于VAE的主题模型，使用Qwen词向量作为语义基础，"
                   "通过编码器将文档embedding映射到主题分布，"
                   "解码器使用词向量重建BOW。",
        module_path="model.etm",
        class_name="ETM",
        supports_embeddings=True,
        supports_pretrained_words=True,
        default_params={
            "num_topics": 20,
            "hidden_dim": 512,
            "doc_embedding_dim": 1024,
            "word_embedding_dim": 1024,
            "encoder_dropout": 0.2,
            "train_word_embeddings": True,
        },
        param_options={
            "num_topics": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
            "hidden_dim": [256, 512, 768, 1024],
            "encoder_dropout": [0.1, 0.2, 0.3, 0.5],
        }
    ),
    
    # DTM - 动态主题模型
    "dtm": TopicModelInfo(
        name="DTM (Dynamic Topic Model)",
        description="动态主题模型，支持时间序列分析，可追踪主题随时间的演化。"
                   "适用于有时间戳的文档集合。",
        module_path="model.dtm",
        class_name="DTM",
        supports_embeddings=True,
        supports_pretrained_words=True,
        default_params={
            "num_topics": 20,
            "time_slices": 10,
            "hidden_dim": 512,
            "doc_embedding_dim": 1024,
            "word_embedding_dim": 1024,
            "evolution_weight": 0.1,
        },
        param_options={
            "num_topics": [5, 10, 15, 20, 30, 50],
            "time_slices": [5, 10, 20, 50],
            "hidden_dim": [256, 512, 1024],
        }
    ),
    
    # LDA - 经典概率主题模型
    "lda": TopicModelInfo(
        name="LDA (Latent Dirichlet Allocation)",
        description="经典概率主题模型，基于神经网络实现。"
                   "不需要预训练embedding，直接使用BOW作为输入。",
        module_path="model.lda",
        class_name="LDA",
        supports_embeddings=False,
        supports_pretrained_words=False,
        default_params={
            "num_topics": 20,
            "hidden_dim": 256,
            "alpha": 0.1,
        },
        param_options={
            "num_topics": [5, 10, 20, 50, 100],
            "hidden_dim": [128, 256, 512],
        }
    ),
}


# ============================================================================
# API函数
# ============================================================================

def get_topic_model_options() -> Dict[str, Dict[str, Any]]:
    """
    获取所有可用的主题模型选项 - 供前端下拉框使用
    
    Returns:
        {
            "etm": {
                "name": "ETM (Embedded Topic Model)",
                "description": "...",
                "supports_embeddings": True,
                "default_params": {...},
                "param_options": {...}
            },
            ...
        }
    """
    return {
        model_id: {
            "name": info.name,
            "description": info.description,
            "supports_embeddings": info.supports_embeddings,
            "supports_pretrained_words": info.supports_pretrained_words,
            "default_params": info.default_params,
            "param_options": info.param_options,
        }
        for model_id, info in TOPIC_MODEL_REGISTRY.items()
    }


def get_model_info(model_id: str) -> Optional[TopicModelInfo]:
    """
    获取指定模型的详细信息
    
    Args:
        model_id: 模型ID (如 "etm")
        
    Returns:
        TopicModelInfo 或 None
    """
    return TOPIC_MODEL_REGISTRY.get(model_id)


def get_model_class(model_id: str) -> Type:
    """
    获取模型类
    
    Args:
        model_id: 模型ID (如 "etm")
        
    Returns:
        模型类
        
    Raises:
        ValueError: 如果模型不存在
        ImportError: 如果模块导入失败
    """
    if model_id not in TOPIC_MODEL_REGISTRY:
        available = list(TOPIC_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")
    
    info = TOPIC_MODEL_REGISTRY[model_id]
    
    # 动态导入模块
    import importlib
    module = importlib.import_module(info.module_path)
    model_class = getattr(module, info.class_name)
    
    return model_class


def get_default_params(model_id: str) -> Dict[str, Any]:
    """
    获取模型的默认参数
    
    Args:
        model_id: 模型ID
        
    Returns:
        默认参数字典
    """
    info = TOPIC_MODEL_REGISTRY.get(model_id)
    if info is None:
        return {}
    return info.default_params.copy()


def list_available_models() -> list:
    """
    列出所有可用的模型ID
    
    Returns:
        模型ID列表
    """
    return list(TOPIC_MODEL_REGISTRY.keys())


def register_model(
    model_id: str,
    name: str,
    description: str,
    module_path: str,
    class_name: str,
    supports_embeddings: bool = False,
    supports_pretrained_words: bool = False,
    default_params: Optional[Dict[str, Any]] = None,
    param_options: Optional[Dict[str, list]] = None
) -> None:
    """
    注册新模型 - 用于动态添加模型
    
    Args:
        model_id: 模型唯一标识
        name: 显示名称
        description: 模型描述
        module_path: 模块路径
        class_name: 类名
        supports_embeddings: 是否支持预训练embedding
        supports_pretrained_words: 是否支持预训练词向量
        default_params: 默认参数
        param_options: 参数可选值
    """
    TOPIC_MODEL_REGISTRY[model_id] = TopicModelInfo(
        name=name,
        description=description,
        module_path=module_path,
        class_name=class_name,
        supports_embeddings=supports_embeddings,
        supports_pretrained_words=supports_pretrained_words,
        default_params=default_params or {},
        param_options=param_options or {}
    )


# ============================================================================
# CLI测试
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("Available Topic Models:")
    print("=" * 60)
    
    for model_id, info in TOPIC_MODEL_REGISTRY.items():
        print(f"\n[{model_id}] {info.name}")
        print(f"  Description: {info.description[:80]}...")
        print(f"  Supports Embeddings: {info.supports_embeddings}")
        print(f"  Default params: {info.default_params}")
    
    print("\n" + "=" * 60)
    print("\nAPI Output (get_topic_model_options):")
    print(json.dumps(get_topic_model_options(), indent=2))
