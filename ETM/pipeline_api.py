"""
Pipeline API - 供前后端人员调用的统一接口

所有功能通过这个文件暴露，前后端人员无需了解内部实现。
使用方式：
    from ETM.pipeline_api import get_available_options, run_pipeline, PipelineRequest

Example:
    # 1. 获取所有可选参数（用于生成UI下拉框）
    options = get_available_options()
    
    # 2. 运行Pipeline
    request = PipelineRequest(
        dataset="hatespeech",
        num_topics=20,
        vocab_size=5000,
        embedding_model="qwen3_0.6B",
    )
    response = run_pipeline(request)
"""

import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PipelineConfig, DATASET_CONFIGS, PRESET_CONFIGS,
    DATA_DIR, RESULT_DIR, QWEN_MODEL_PATH
)


# ============================================================================
# 1. 枚举和常量定义 - 前端用于生成下拉框选项
# ============================================================================

class EmbeddingMode(Enum):
    """Embedding模式"""
    ZERO_SHOT = "zero_shot"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"


class TopicModel(Enum):
    """主题模型类型"""
    ETM = "etm"
    # 未来可扩展
    # LDA = "lda"
    # NTM = "ntm"


# 主题数可选值
NUM_TOPICS_OPTIONS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

# 词表大小可选值
VOCAB_SIZE_OPTIONS = [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]

# 隐藏层维度可选值
HIDDEN_DIM_OPTIONS = [256, 512, 768, 1024]

# 批次大小可选值
BATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256]

# 训练轮数可选值
EPOCHS_OPTIONS = [20, 30, 50, 100, 150, 200]

# 学习率可选值
LEARNING_RATE_OPTIONS = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]


# ============================================================================
# 2. 参数约束定义 - 用于验证前端传入的参数
# ============================================================================

PARAM_CONSTRAINTS = {
    "num_topics": {
        "type": "int",
        "min": 5,
        "max": 100,
        "default": 20,
        "options": NUM_TOPICS_OPTIONS,
        "description": "主题数量，建议根据数据集大小选择"
    },
    "vocab_size": {
        "type": "int",
        "min": 1000,
        "max": 20000,
        "default": 5000,
        "options": VOCAB_SIZE_OPTIONS,
        "description": "词表大小，建议 sqrt(文档数) * 50"
    },
    "hidden_dim": {
        "type": "int",
        "min": 128,
        "max": 1024,
        "default": 512,
        "options": HIDDEN_DIM_OPTIONS,
        "description": "编码器隐藏层维度"
    },
    "epochs": {
        "type": "int",
        "min": 10,
        "max": 500,
        "default": 50,
        "options": EPOCHS_OPTIONS,
        "description": "训练轮数"
    },
    "batch_size": {
        "type": "int",
        "min": 8,
        "max": 512,
        "default": 64,
        "options": BATCH_SIZE_OPTIONS,
        "description": "批次大小"
    },
    "learning_rate": {
        "type": "float",
        "min": 0.00001,
        "max": 0.1,
        "default": 0.002,
        "options": LEARNING_RATE_OPTIONS,
        "description": "学习率"
    },
}


# ============================================================================
# 3. 数据结构定义
# ============================================================================

@dataclass
class PipelineRequest:
    """
    Pipeline请求参数 - 前端传入
    
    所有参数都有默认值，前端可以只传必要的参数
    """
    # 必填参数
    dataset: str                                    # 数据集名称
    
    # 模式选择
    embedding_mode: str = "zero_shot"               # embedding模式: zero_shot/supervised/unsupervised
    embedding_model: str = "qwen3_0.6B"             # embedding模型名称
    topic_model: str = "etm"                        # 主题模型: etm (未来可扩展)
    
    # 核心超参数
    num_topics: int = 20                            # 主题数量
    vocab_size: int = 5000                          # 词表大小
    
    # 模型架构参数
    hidden_dim: int = 512                           # 隐藏层维度
    
    # 训练参数
    epochs: int = 50                                # 训练轮数
    batch_size: int = 64                            # 批次大小
    learning_rate: float = 0.002                    # 学习率
    
    # KL退火参数
    kl_start: float = 0.0                           # KL权重起始值
    kl_end: float = 1.0                             # KL权重终止值
    kl_warmup_epochs: int = 30                      # KL预热轮数
    
    # 早停参数
    early_stopping: bool = True                     # 是否启用早停
    patience: int = 15                              # 早停耐心值
    
    # 其他
    dev_mode: bool = False                          # 调试模式
    preset: Optional[str] = None                    # 预设配置: small/medium/large
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineRequest":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineResponse:
    """
    Pipeline响应 - 返回给前端
    """
    success: bool                                   # 是否成功
    task_id: str                                    # 任务ID
    dataset: str                                    # 数据集名称
    mode: str                                       # embedding模式
    
    # 结果路径
    result_dir: str = ""                            # 结果目录
    model_path: Optional[str] = None                # 模型文件路径
    
    # 评估指标
    metrics: Optional[Dict[str, float]] = None      # 评估指标
    
    # 主题词
    topic_words: Optional[Dict[str, List[str]]] = None  # 主题词列表
    
    # 可视化
    visualization_paths: Optional[List[str]] = None # 可视化文件路径
    
    # 错误信息
    error_message: Optional[str] = None             # 错误信息
    
    # 时间戳
    timestamp: str = ""                             # 结果时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str                                       # 数据集名称
    path: str                                       # 数据集路径
    size: Optional[int] = None                      # 文档数量
    has_labels: bool = False                        # 是否有标签
    language: str = "english"                       # 语言
    columns: Optional[List[str]] = None             # CSV列名
    recommended_vocab_size: int = 5000              # 推荐词表大小
    recommended_num_topics: int = 20                # 推荐主题数


@dataclass 
class ResultInfo:
    """训练结果信息"""
    dataset: str                                    # 数据集名称
    mode: str                                       # embedding模式
    timestamp: str                                  # 时间戳
    path: str                                       # 结果路径
    
    # 配置信息
    num_topics: Optional[int] = None
    vocab_size: Optional[int] = None
    epochs_trained: Optional[int] = None
    
    # 评估指标
    metrics: Optional[Dict[str, float]] = None
    
    # 文件存在性
    has_model: bool = False
    has_theta: bool = False
    has_beta: bool = False
    has_topic_words: bool = False
    has_visualizations: bool = False


# ============================================================================
# 4. 核心API函数
# ============================================================================

def get_available_options() -> Dict[str, Any]:
    """
    获取所有可配置选项 - 前端用于生成下拉框
    
    Returns:
        包含所有可选参数的字典，前端可直接用于渲染UI
        
    Example:
        options = get_available_options()
        # options["num_topics"]["options"] = [5, 10, 15, 20, ...]
        # options["embedding_modes"] = ["zero_shot", "supervised", "unsupervised"]
    """
    # 导入模型注册表
    try:
        from model.registry import get_topic_model_options
        topic_models = get_topic_model_options()
    except ImportError:
        topic_models = {
            "etm": {
                "name": "ETM (Embedded Topic Model)",
                "description": "基于VAE的主题模型，使用Qwen词向量"
            }
        }
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "embedding"))
        from registry import get_embedding_model_options
        embedding_models = get_embedding_model_options()
    except ImportError:
        embedding_models = {
            "qwen3_0.6B": {
                "name": "Qwen3-Embedding-0.6B",
                "path": str(QWEN_MODEL_PATH),
                "embedding_dim": 1024,
                "description": "轻量级中英文embedding模型"
            }
        }
    
    # 获取可用数据集
    datasets = list_datasets()
    
    return {
        # 参数选项（带约束信息）
        "parameters": PARAM_CONSTRAINTS,
        
        # 快捷选项列表
        "num_topics": NUM_TOPICS_OPTIONS,
        "vocab_size": VOCAB_SIZE_OPTIONS,
        "hidden_dim": HIDDEN_DIM_OPTIONS,
        "batch_size": BATCH_SIZE_OPTIONS,
        "epochs": EPOCHS_OPTIONS,
        "learning_rate": LEARNING_RATE_OPTIONS,
        
        # 模式选项
        "embedding_modes": [e.value for e in EmbeddingMode],
        
        # 模型选项
        "embedding_models": embedding_models,
        "topic_models": topic_models,
        
        # 数据集
        "datasets": {d.name: asdict(d) for d in datasets},
        
        # 预设配置
        "presets": PRESET_CONFIGS,
        
        # 数据集特定推荐配置
        "dataset_configs": DATASET_CONFIGS,
    }


def validate_request(request: PipelineRequest) -> tuple[bool, str]:
    """
    验证Pipeline请求参数
    
    Args:
        request: Pipeline请求
        
    Returns:
        (is_valid, error_message)
    """
    # 检查数据集是否存在
    dataset_dir = DATA_DIR / request.dataset
    if not dataset_dir.exists():
        return False, f"Dataset '{request.dataset}' not found at {dataset_dir}"
    
    # 检查embedding模式
    valid_modes = [e.value for e in EmbeddingMode]
    if request.embedding_mode not in valid_modes:
        return False, f"Invalid embedding_mode '{request.embedding_mode}'. Valid: {valid_modes}"
    
    # 检查数值参数范围
    for param_name, constraints in PARAM_CONSTRAINTS.items():
        value = getattr(request, param_name, None)
        if value is not None:
            if "min" in constraints and value < constraints["min"]:
                return False, f"{param_name}={value} is below minimum {constraints['min']}"
            if "max" in constraints and value > constraints["max"]:
                return False, f"{param_name}={value} is above maximum {constraints['max']}"
    
    return True, "OK"


def list_datasets() -> List[DatasetInfo]:
    """
    列出所有可用数据集
    
    Returns:
        数据集信息列表
    """
    datasets = []
    
    if not DATA_DIR.exists():
        return datasets
    
    for dataset_dir in DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # 查找CSV文件
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            continue
        
        # 获取数据集配置
        ds_config = DATASET_CONFIGS.get(dataset_dir.name, {})
        
        info = DatasetInfo(
            name=dataset_dir.name,
            path=str(dataset_dir),
            language=ds_config.get("language", "english"),
            recommended_vocab_size=ds_config.get("vocab_size", 5000),
            recommended_num_topics=ds_config.get("num_topics", 20),
        )
        
        # 尝试读取CSV获取更多信息
        for csv_file in csv_files:
            if "text_only" in csv_file.name or "cleaned" in csv_file.name:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, nrows=5)
                    info.columns = df.columns.tolist()
                    info.has_labels = any(col in df.columns for col in ['label', 'Label', 'labels'])
                    # 获取完整行数
                    info.size = sum(1 for _ in open(csv_file)) - 1
                except Exception:
                    pass
                break
        
        datasets.append(info)
    
    return datasets


def get_dataset_info(dataset: str) -> Optional[DatasetInfo]:
    """
    获取单个数据集的详细信息
    
    Args:
        dataset: 数据集名称
        
    Returns:
        数据集信息，不存在则返回None
    """
    datasets = list_datasets()
    for ds in datasets:
        if ds.name == dataset:
            return ds
    return None


def list_results(dataset: Optional[str] = None, mode: Optional[str] = None) -> List[ResultInfo]:
    """
    列出训练结果
    
    Args:
        dataset: 筛选特定数据集（可选）
        mode: 筛选特定模式（可选）
        
    Returns:
        结果信息列表
    """
    results = []
    
    if not RESULT_DIR.exists():
        return results
    
    for dataset_dir in RESULT_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        if dataset and dataset_dir.name != dataset:
            continue
        
        for mode_dir in dataset_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            if mode and mode_dir.name != mode:
                continue
            
            model_dir = mode_dir / "model"
            eval_dir = mode_dir / "evaluation"
            viz_dir = mode_dir / "visualization"
            
            info = ResultInfo(
                dataset=dataset_dir.name,
                mode=mode_dir.name,
                timestamp="",
                path=str(mode_dir),
                has_model=model_dir.exists() and any(model_dir.glob("*.pt")),
                has_theta=model_dir.exists() and any(model_dir.glob("theta_*.npy")),
                has_beta=model_dir.exists() and any(model_dir.glob("beta_*.npy")),
                has_topic_words=model_dir.exists() and any(model_dir.glob("topic_words_*.json")),
                has_visualizations=viz_dir.exists() and any(viz_dir.glob("*.png")),
            )
            
            # 获取时间戳和配置
            if model_dir.exists():
                theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
                if theta_files:
                    info.timestamp = theta_files[0].stem.replace("theta_", "")
                
                config_files = sorted(model_dir.glob("config_*.json"), reverse=True)
                if config_files:
                    try:
                        with open(config_files[0]) as f:
                            config = json.load(f)
                            info.num_topics = config.get("model", {}).get("num_topics")
                            info.vocab_size = config.get("bow", {}).get("vocab_size")
                    except Exception:
                        pass
            
            # 获取评估指标
            if eval_dir.exists():
                metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
                if metrics_files:
                    try:
                        with open(metrics_files[0]) as f:
                            info.metrics = json.load(f)
                    except Exception:
                        pass
            
            results.append(info)
    
    return results


def get_result_info(dataset: str, mode: str) -> Optional[ResultInfo]:
    """
    获取特定训练结果的详细信息
    
    Args:
        dataset: 数据集名称
        mode: embedding模式
        
    Returns:
        结果信息，不存在则返回None
    """
    results = list_results(dataset=dataset, mode=mode)
    return results[0] if results else None


def get_topic_words(dataset: str, mode: str, top_k: int = 10) -> Optional[Dict[str, List[str]]]:
    """
    获取主题词
    
    Args:
        dataset: 数据集名称
        mode: embedding模式
        top_k: 每个主题返回的词数
        
    Returns:
        主题词字典 {topic_id: [word1, word2, ...]}
    """
    result_path = RESULT_DIR / dataset / mode / "model"
    if not result_path.exists():
        return None
    
    topic_files = sorted(result_path.glob("topic_words_*.json"), reverse=True)
    if not topic_files:
        return None
    
    try:
        with open(topic_files[0]) as f:
            topic_words = json.load(f)
        
        # 截取top_k个词
        if top_k < 20:
            topic_words = {k: v[:top_k] for k, v in topic_words.items()}
        
        return topic_words
    except Exception:
        return None


def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """
    运行完整Pipeline
    
    Args:
        request: Pipeline请求参数
        
    Returns:
        PipelineResponse: 包含结果路径、指标、主题词等
        
    Example:
        request = PipelineRequest(
            dataset="hatespeech",
            num_topics=20,
            vocab_size=5000,
        )
        response = run_pipeline(request)
        if response.success:
            print(f"Results saved to: {response.result_dir}")
            print(f"Topic coherence: {response.metrics.get('topic_coherence_avg')}")
    """
    # 生成任务ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 验证请求
    is_valid, error_msg = validate_request(request)
    if not is_valid:
        return PipelineResponse(
            success=False,
            task_id=task_id,
            dataset=request.dataset,
            mode=request.embedding_mode,
            error_message=error_msg
        )
    
    try:
        # 创建配置
        config = _create_config_from_request(request)
        
        # 运行pipeline
        from main import (
            setup_logging, setup_device, load_texts,
            generate_bow, generate_vocab_embeddings, load_doc_embeddings,
            train_etm, save_results, run_evaluation, run_visualization
        )
        
        # 设置日志和设备
        logger = setup_logging(config)
        device = setup_device(config)
        
        logger.info(f"Starting pipeline for {request.dataset}/{request.embedding_mode}")
        logger.info(f"Config: num_topics={request.num_topics}, vocab_size={request.vocab_size}")
        
        # 1. 加载文本
        texts, labels = load_texts(config, logger)
        
        # 2. 生成BOW
        bow_matrix, vocab = generate_bow(texts, config, logger)
        
        # 3. 生成词汇嵌入
        vocab_embeddings = generate_vocab_embeddings(vocab, config, logger)
        
        # 4. 加载文档嵌入
        doc_embeddings, _ = load_doc_embeddings(config, logger)
        
        # 5. 训练ETM
        train_result = train_etm(
            doc_embeddings, bow_matrix, vocab_embeddings,
            config, logger, device
        )
        
        # 6. 保存结果
        timestamp = save_results(
            train_result['model'], train_result['history'],
            vocab, doc_embeddings, bow_matrix, vocab_embeddings,
            config, logger, device
        )
        
        # 7. 评估
        metrics = run_evaluation(config, logger, timestamp)
        
        # 8. 可视化
        viz_paths = run_visualization(config, logger, timestamp)
        
        # 获取主题词
        topic_words = get_topic_words(request.dataset, request.embedding_mode)
        
        return PipelineResponse(
            success=True,
            task_id=task_id,
            dataset=request.dataset,
            mode=request.embedding_mode,
            result_dir=config.result_dir,
            model_path=os.path.join(config.model_dir, f"etm_model_{timestamp}.pt"),
            metrics=metrics,
            topic_words=topic_words,
            visualization_paths=viz_paths,
            timestamp=timestamp
        )
        
    except FileNotFoundError as e:
        return PipelineResponse(
            success=False,
            task_id=task_id,
            dataset=request.dataset,
            mode=request.embedding_mode,
            error_message=f"File not found: {str(e)}. Please ensure embeddings are generated first."
        )
    except Exception as e:
        import traceback
        return PipelineResponse(
            success=False,
            task_id=task_id,
            dataset=request.dataset,
            mode=request.embedding_mode,
            error_message=f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
        )


def _create_config_from_request(request: PipelineRequest) -> PipelineConfig:
    """从请求创建配置对象"""
    # 如果指定了预设，先加载预设
    if request.preset and request.preset in PRESET_CONFIGS:
        from config import get_preset_config
        config = get_preset_config(request.preset)
    else:
        config = PipelineConfig()
    
    # 设置数据集
    config.data.dataset = request.dataset
    
    # 设置embedding模式
    config.embedding.mode = request.embedding_mode
    
    # 设置embedding模型路径
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "embedding"))
        from registry import get_embedding_model_path
        config.embedding.model_path = get_embedding_model_path(request.embedding_model)
    except ImportError:
        config.embedding.model_path = str(QWEN_MODEL_PATH)
    
    # 设置模型参数
    config.model.num_topics = request.num_topics
    config.bow.vocab_size = request.vocab_size
    config.model.hidden_dim = request.hidden_dim
    config.model.epochs = request.epochs
    config.model.batch_size = request.batch_size
    config.model.learning_rate = request.learning_rate
    config.model.kl_start = request.kl_start
    config.model.kl_end = request.kl_end
    config.model.kl_warmup_epochs = request.kl_warmup_epochs
    config.model.early_stopping = request.early_stopping
    config.model.patience = request.patience
    
    # 设置调试模式
    config.dev_mode = request.dev_mode
    
    return config


# ============================================================================
# 5. 便捷函数
# ============================================================================

def quick_train(
    dataset: str,
    num_topics: int = 20,
    vocab_size: int = 5000,
    mode: str = "zero_shot",
    preset: Optional[str] = None
) -> PipelineResponse:
    """
    快速训练接口
    
    Args:
        dataset: 数据集名称
        num_topics: 主题数
        vocab_size: 词表大小
        mode: embedding模式
        preset: 预设配置 (small/medium/large)
        
    Returns:
        PipelineResponse
        
    Example:
        response = quick_train("hatespeech", num_topics=20)
    """
    request = PipelineRequest(
        dataset=dataset,
        num_topics=num_topics,
        vocab_size=vocab_size,
        embedding_mode=mode,
        preset=preset
    )
    return run_pipeline(request)


def get_recommended_config(dataset: str) -> Dict[str, Any]:
    """
    获取数据集的推荐配置
    
    Args:
        dataset: 数据集名称
        
    Returns:
        推荐配置字典
    """
    if dataset in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset]
    
    # 默认配置
    return {
        "vocab_size": 5000,
        "num_topics": 20,
        "min_doc_freq": 5,
        "language": "english"
    }


# ============================================================================
# 6. CLI入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ETM Pipeline API")
    subparsers = parser.add_subparsers(dest="command")
    
    # list-options命令
    list_opts_parser = subparsers.add_parser("list-options", help="List all available options")
    
    # list-datasets命令
    list_ds_parser = subparsers.add_parser("list-datasets", help="List available datasets")
    
    # list-results命令
    list_res_parser = subparsers.add_parser("list-results", help="List training results")
    list_res_parser.add_argument("--dataset", type=str, help="Filter by dataset")
    list_res_parser.add_argument("--mode", type=str, help="Filter by mode")
    
    # train命令
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    train_parser.add_argument("--mode", type=str, default="zero_shot", help="Embedding mode")
    train_parser.add_argument("--num_topics", type=int, default=20, help="Number of topics")
    train_parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    train_parser.add_argument("--preset", type=str, help="Preset config (small/medium/large)")
    train_parser.add_argument("--dev", action="store_true", help="Dev mode")
    
    args = parser.parse_args()
    
    if args.command == "list-options":
        options = get_available_options()
        print(json.dumps(options, indent=2, default=str))
    
    elif args.command == "list-datasets":
        datasets = list_datasets()
        for ds in datasets:
            print(f"{ds.name}: {ds.size or 'N/A'} docs, labels={ds.has_labels}")
    
    elif args.command == "list-results":
        results = list_results(dataset=args.dataset, mode=args.mode)
        for r in results:
            print(f"{r.dataset}/{r.mode}: topics={r.num_topics}, vocab={r.vocab_size}")
            if r.metrics:
                print(f"  coherence={r.metrics.get('topic_coherence_avg', 'N/A'):.3f}")
    
    elif args.command == "train":
        request = PipelineRequest(
            dataset=args.dataset,
            embedding_mode=args.mode,
            num_topics=args.num_topics,
            vocab_size=args.vocab_size,
            preset=args.preset,
            dev_mode=args.dev
        )
        response = run_pipeline(request)
        if response.success:
            print(f"Success! Results saved to: {response.result_dir}")
            if response.metrics:
                print(f"Metrics: {json.dumps(response.metrics, indent=2)}")
        else:
            print(f"Failed: {response.error_message}")
    
    else:
        parser.print_help()
