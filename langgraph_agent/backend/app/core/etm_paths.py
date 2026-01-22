"""
ETM Path Management
统一管理 ETM 模块的路径配置和导入
"""

import sys
from pathlib import Path
from typing import List
from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


def setup_etm_paths() -> List[Path]:
    """
    设置 ETM 相关路径到 sys.path
    
    返回已添加的路径列表
    """
    paths_added = []
    
    # 1. ETM 目录本身（用于导入 engine_a, engine_c 等）
    etm_dir = settings.ETM_DIR
    if etm_dir.exists():
        etm_path = str(etm_dir)
        if etm_path not in sys.path:
            sys.path.insert(0, etm_path)
            paths_added.append(etm_dir)
            logger.debug(f"Added ETM directory to path: {etm_path}")
    else:
        logger.warning(f"ETM directory not found: {etm_dir}")
    
    # 2. ETM 的父目录（用于导入 ETM.preprocessing）
    etm_parent = settings.ETM_DIR.parent
    if etm_parent.exists():
        parent_path = str(etm_parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
            paths_added.append(etm_parent)
            logger.debug(f"Added ETM parent directory to path: {parent_path}")
    
    return paths_added


def check_etm_modules() -> dict:
    """
    检查 ETM 模块是否可用
    
    返回检查结果字典
    """
    results = {
        "etm_dir": str(settings.ETM_DIR),
        "etm_dir_exists": settings.ETM_DIR.exists(),
        "modules": {}
    }
    
    # 检查 engine_a 模块
    try:
        from engine_a.vocab_builder import VocabBuilder
        from engine_a.bow_generator import BOWGenerator
        results["modules"]["engine_a"] = {
            "status": "ok",
            "imports": ["VocabBuilder", "BOWGenerator"]
        }
    except ImportError as e:
        results["modules"]["engine_a"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 检查 engine_c 模块
    try:
        from engine_c.etm import ETM
        from engine_c.encoder import ETMEncoder
        from engine_c.decoder import ETMDecoder
        results["modules"]["engine_c"] = {
            "status": "ok",
            "imports": ["ETM", "ETMEncoder", "ETMDecoder"]
        }
    except ImportError as e:
        results["modules"]["engine_c"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 检查 preprocessing 模块
    try:
        # 需要从父目录导入
        from ETM.preprocessing import EmbeddingProcessor, ProcessingConfig
        results["modules"]["preprocessing"] = {
            "status": "ok",
            "imports": ["EmbeddingProcessor", "ProcessingConfig"]
        }
    except ImportError as e:
        results["modules"]["preprocessing"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 检查 trainer 模块（可选）
    try:
        from trainer.trainer import ETMTrainer
        results["modules"]["trainer"] = {
            "status": "ok",
            "imports": ["ETMTrainer"]
        }
    except ImportError as e:
        results["modules"]["trainer"] = {
            "status": "error",
            "error": str(e),
            "note": "Optional module"
        }
    
    return results


# 在模块导入时自动设置路径
setup_etm_paths()
