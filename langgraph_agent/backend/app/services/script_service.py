"""
Script Execution Service
用于执行服务器上的 bash 脚本并追踪执行状态
"""

import asyncio
import subprocess
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
from enum import Enum
from pydantic import BaseModel

from ..core.logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)


class ScriptStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScriptInfo(BaseModel):
    """脚本信息"""
    id: str
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    category: str


class ScriptJob(BaseModel):
    """脚本执行任务"""
    job_id: str
    script_id: str
    script_name: str
    parameters: Dict[str, str]
    status: ScriptStatus
    progress: float = 0.0
    message: str = ""
    logs: List[str] = []
    exit_code: Optional[int] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


# 脚本目录
SCRIPTS_DIR = Path("/root/autodl-tmp/scripts")

# 可用脚本定义
AVAILABLE_SCRIPTS = {
    "01_data_upload_clean": ScriptInfo(
        id="01_data_upload_clean",
        name="01_data_upload_clean.sh",
        description="数据上传与清洗 - 将原始文本/CSV文件清洗处理成标准格式",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "input_path", "type": "string", "required": False, "description": "输入文件路径"},
            {"name": "language", "type": "string", "required": False, "default": "english", "description": "语言 (english/chinese)"}
        ],
        category="preprocessing"
    ),
    "02_embedding_generate": ScriptInfo(
        id="02_embedding_generate",
        name="02_embedding_generate.sh",
        description="Embedding生成 - 使用Qwen3-Embedding生成文档向量",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式 (zero_shot/supervised/unsupervised)"},
            {"name": "epochs", "type": "integer", "required": False, "default": "3", "description": "训练轮数"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "16", "description": "批次大小"}
        ],
        category="preprocessing"
    ),
    "03_bow_generate": ScriptInfo(
        id="03_bow_generate",
        name="03_bow_generate.sh",
        description="BOW生成 - 构建词袋模型矩阵和词汇表",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "vocab_size", "type": "integer", "required": False, "default": "5000", "description": "词汇表大小"},
            {"name": "min_df", "type": "integer", "required": False, "default": "5", "description": "最小文档频率"}
        ],
        category="preprocessing"
    ),
    "04_vocab_embedding": ScriptInfo(
        id="04_vocab_embedding",
        name="04_vocab_embedding.sh",
        description="词汇Embedding生成 - 为词汇表中的每个词生成embedding向量",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "64", "description": "批次大小"}
        ],
        category="preprocessing"
    ),
    "05_etm_train": ScriptInfo(
        id="05_etm_train",
        name="05_etm_train.sh",
        description="ETM主题模型训练 - 训练嵌入主题模型",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量"},
            {"name": "vocab_size", "type": "integer", "required": False, "default": "5000", "description": "词汇表大小"},
            {"name": "epochs", "type": "integer", "required": False, "default": "50", "description": "训练轮数"},
            {"name": "batch_size", "type": "integer", "required": False, "default": "64", "description": "批次大小"}
        ],
        category="training"
    ),
    "06_evaluate": ScriptInfo(
        id="06_evaluate",
        name="06_evaluate.sh",
        description="模型评估 - 计算主题一致性、多样性等指标",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式"}
        ],
        category="evaluation"
    ),
    "07_visualize": ScriptInfo(
        id="07_visualize",
        name="07_visualize.sh",
        description="可视化 - 生成主题词云、分布图等可视化结果",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式"}
        ],
        category="visualization"
    ),
    "run_full_pipeline": ScriptInfo(
        id="run_full_pipeline",
        name="run_full_pipeline.sh",
        description="完整流程 - 运行从数据清洗到可视化的完整ETM流水线",
        parameters=[
            {"name": "dataset", "type": "string", "required": True, "description": "数据集名称"},
            {"name": "mode", "type": "string", "required": False, "default": "zero_shot", "description": "模式"},
            {"name": "num_topics", "type": "integer", "required": False, "default": "20", "description": "主题数量"}
        ],
        category="pipeline"
    ),
    "run_quick_demo": ScriptInfo(
        id="run_quick_demo",
        name="run_quick_demo.sh",
        description="快速演示 - 使用默认参数快速运行演示",
        parameters=[],
        category="demo"
    )
}


class ScriptService:
    """脚本执行服务"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.scripts_dir = SCRIPTS_DIR
    
    def get_available_scripts(self) -> List[ScriptInfo]:
        """获取所有可用脚本"""
        return list(AVAILABLE_SCRIPTS.values())
    
    def get_script_info(self, script_id: str) -> Optional[ScriptInfo]:
        """获取指定脚本信息"""
        return AVAILABLE_SCRIPTS.get(script_id)
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Dict]:
        """获取所有任务"""
        return list(self.jobs.values())
    
    async def execute_script(
        self,
        script_id: str,
        parameters: Dict[str, str]
    ) -> str:
        """
        执行脚本
        返回 job_id
        """
        script_info = AVAILABLE_SCRIPTS.get(script_id)
        if not script_info:
            raise ValueError(f"Unknown script: {script_id}")
        
        # 生成任务ID
        job_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 创建任务记录
        job = {
            "job_id": job_id,
            "script_id": script_id,
            "script_name": script_info.name,
            "parameters": parameters,
            "status": ScriptStatus.PENDING.value,
            "progress": 0.0,
            "message": "任务已创建，等待执行",
            "logs": [],
            "exit_code": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None
        }
        
        self.jobs[job_id] = job
        
        # 异步执行脚本
        asyncio.create_task(self._run_script(job_id, script_info, parameters))
        
        return job_id
    
    async def _run_script(
        self,
        job_id: str,
        script_info: ScriptInfo,
        parameters: Dict[str, str]
    ):
        """实际执行脚本"""
        job = self.jobs[job_id]
        
        try:
            # 更新状态为运行中
            job["status"] = ScriptStatus.RUNNING.value
            job["message"] = f"正在执行 {script_info.name}"
            job["updated_at"] = datetime.now().isoformat()
            
            # 构建命令
            script_path = self.scripts_dir / script_info.name
            cmd_args = [str(script_path)]
            
            # 按顺序添加参数
            for param_def in script_info.parameters:
                param_name = param_def["name"]
                if param_name in parameters:
                    cmd_args.append(str(parameters[param_name]))
                elif "default" in param_def:
                    cmd_args.append(str(param_def["default"]))
            
            logger.info(f"Executing script: {' '.join(cmd_args)}")
            job["logs"].append(f"[{datetime.now().isoformat()}] 执行命令: {' '.join(cmd_args)}")
            
            # 执行脚本
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.scripts_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            # 实时读取输出
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_text = line.decode('utf-8', errors='replace').rstrip()
                job["logs"].append(f"[{datetime.now().isoformat()}] {line_text}")
                job["updated_at"] = datetime.now().isoformat()
                
                # 解析进度（如果脚本输出包含进度信息）
                if "%" in line_text:
                    try:
                        # 尝试提取百分比
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)\s*%', line_text)
                        if match:
                            job["progress"] = float(match.group(1))
                    except:
                        pass
                
                # 更新消息
                job["message"] = line_text[:200] if len(line_text) > 200 else line_text
            
            # 等待进程完成
            await process.wait()
            exit_code = process.returncode
            
            job["exit_code"] = exit_code
            job["completed_at"] = datetime.now().isoformat()
            job["updated_at"] = datetime.now().isoformat()
            
            if exit_code == 0:
                job["status"] = ScriptStatus.COMPLETED.value
                job["progress"] = 100.0
                job["message"] = f"{script_info.name} 执行完成"
                logger.info(f"Script {script_info.name} completed successfully")
            else:
                job["status"] = ScriptStatus.FAILED.value
                job["error_message"] = f"脚本退出码: {exit_code}"
                job["message"] = f"执行失败 (退出码: {exit_code})"
                logger.error(f"Script {script_info.name} failed with exit code {exit_code}")
        
        except asyncio.CancelledError:
            job["status"] = ScriptStatus.CANCELLED.value
            job["message"] = "任务已取消"
            job["updated_at"] = datetime.now().isoformat()
            logger.info(f"Script job {job_id} cancelled")
        
        except Exception as e:
            job["status"] = ScriptStatus.FAILED.value
            job["error_message"] = str(e)
            job["message"] = f"执行出错: {str(e)}"
            job["updated_at"] = datetime.now().isoformat()
            logger.error(f"Script execution error: {e}", exc_info=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job["status"] not in [ScriptStatus.PENDING.value, ScriptStatus.RUNNING.value]:
            return False
        
        job["status"] = ScriptStatus.CANCELLED.value
        job["message"] = "任务已取消"
        job["updated_at"] = datetime.now().isoformat()
        
        return True


# 全局服务实例
script_service = ScriptService()
