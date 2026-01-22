"""
Scripts API Routes
用于执行和管理服务器上的 bash 脚本
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..services.script_service import (
    script_service,
    ScriptInfo,
    ScriptJob,
    ScriptStatus,
    AVAILABLE_SCRIPTS
)
from ..core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/scripts", tags=["scripts"])


class ExecuteScriptRequest(BaseModel):
    """执行脚本请求"""
    script_id: str
    parameters: Dict[str, str] = {}


class ExecuteScriptResponse(BaseModel):
    """执行脚本响应"""
    job_id: str
    script_id: str
    script_name: str
    status: str
    message: str


class ScriptJobResponse(BaseModel):
    """脚本任务响应"""
    job_id: str
    script_id: str
    script_name: str
    parameters: Dict[str, str]
    status: str
    progress: float
    message: str
    logs: List[str]
    exit_code: Optional[int]
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    error_message: Optional[str]


@router.get("", response_model=List[ScriptInfo])
async def list_scripts():
    """获取所有可用脚本列表"""
    return script_service.get_available_scripts()


@router.get("/categories")
async def list_script_categories():
    """获取脚本分类"""
    categories = {}
    for script in AVAILABLE_SCRIPTS.values():
        if script.category not in categories:
            categories[script.category] = []
        categories[script.category].append({
            "id": script.id,
            "name": script.name,
            "description": script.description
        })
    return categories


@router.get("/{script_id}", response_model=ScriptInfo)
async def get_script(script_id: str):
    """获取指定脚本信息"""
    script_info = script_service.get_script_info(script_id)
    if not script_info:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return script_info


@router.post("/execute", response_model=ExecuteScriptResponse)
async def execute_script(request: ExecuteScriptRequest):
    """
    执行指定脚本
    
    参数说明：
    - script_id: 脚本ID (如 "05_etm_train")
    - parameters: 脚本参数字典
    
    示例：
    ```json
    {
        "script_id": "05_etm_train",
        "parameters": {
            "dataset": "hatespeech",
            "mode": "zero_shot",
            "num_topics": "20",
            "epochs": "50"
        }
    }
    ```
    """
    script_info = script_service.get_script_info(request.script_id)
    if not script_info:
        raise HTTPException(status_code=404, detail=f"Script '{request.script_id}' not found")
    
    # 验证必需参数
    for param in script_info.parameters:
        if param.get("required", False) and param["name"] not in request.parameters:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameter: {param['name']}"
            )
    
    try:
        job_id = await script_service.execute_script(
            script_id=request.script_id,
            parameters=request.parameters
        )
        
        return ExecuteScriptResponse(
            job_id=job_id,
            script_id=request.script_id,
            script_name=script_info.name,
            status="pending",
            message=f"脚本 {script_info.name} 已提交执行"
        )
    
    except Exception as e:
        logger.error(f"Failed to execute script: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {str(e)}")


@router.get("/jobs", response_model=List[ScriptJobResponse])
async def list_jobs():
    """获取所有脚本任务列表"""
    jobs = script_service.get_all_jobs()
    return [ScriptJobResponse(**job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=ScriptJobResponse)
async def get_job(job_id: str):
    """获取指定任务状态"""
    job = script_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ScriptJobResponse(**job)


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """
    获取任务日志
    
    参数：
    - tail: 返回最后N行日志，默认100
    """
    job = script_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    logs = job.get("logs", [])
    if tail > 0:
        logs = logs[-tail:]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "logs": logs,
        "total_lines": len(job.get("logs", []))
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """取消正在执行的任务"""
    success = await script_service.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (not found or already finished)"
        )
    return {"message": "Job cancelled", "job_id": job_id}


# ==========================================
# 便捷端点：直接执行特定脚本
# ==========================================

class TrainRequest(BaseModel):
    """训练请求"""
    dataset: str
    mode: str = "zero_shot"
    num_topics: int = 20
    epochs: int = 50
    batch_size: int = 64


class EmbeddingRequest(BaseModel):
    """Embedding生成请求"""
    dataset: str
    mode: str = "zero_shot"
    epochs: int = 3
    batch_size: int = 16


class EvaluateRequest(BaseModel):
    """评估请求"""
    dataset: str
    mode: str = "zero_shot"


class VisualizeRequest(BaseModel):
    """可视化请求"""
    dataset: str
    mode: str = "zero_shot"


class FullPipelineRequest(BaseModel):
    """完整流程请求"""
    dataset: str
    mode: str = "zero_shot"
    num_topics: int = 20


@router.post("/train", response_model=ExecuteScriptResponse)
async def train_etm(request: TrainRequest):
    """便捷端点：执行ETM训练"""
    return await execute_script(ExecuteScriptRequest(
        script_id="05_etm_train",
        parameters={
            "dataset": request.dataset,
            "mode": request.mode,
            "num_topics": str(request.num_topics),
            "epochs": str(request.epochs),
            "batch_size": str(request.batch_size)
        }
    ))


@router.post("/embedding", response_model=ExecuteScriptResponse)
async def generate_embedding(request: EmbeddingRequest):
    """便捷端点：生成Embedding"""
    return await execute_script(ExecuteScriptRequest(
        script_id="02_embedding_generate",
        parameters={
            "dataset": request.dataset,
            "mode": request.mode,
            "epochs": str(request.epochs),
            "batch_size": str(request.batch_size)
        }
    ))


@router.post("/evaluate", response_model=ExecuteScriptResponse)
async def evaluate_model(request: EvaluateRequest):
    """便捷端点：评估模型"""
    return await execute_script(ExecuteScriptRequest(
        script_id="06_evaluate",
        parameters={
            "dataset": request.dataset,
            "mode": request.mode
        }
    ))


@router.post("/visualize", response_model=ExecuteScriptResponse)
async def visualize_results(request: VisualizeRequest):
    """便捷端点：生成可视化"""
    return await execute_script(ExecuteScriptRequest(
        script_id="07_visualize",
        parameters={
            "dataset": request.dataset,
            "mode": request.mode
        }
    ))


@router.post("/pipeline", response_model=ExecuteScriptResponse)
async def run_full_pipeline(request: FullPipelineRequest):
    """便捷端点：运行完整流程"""
    return await execute_script(ExecuteScriptRequest(
        script_id="run_full_pipeline",
        parameters={
            "dataset": request.dataset,
            "mode": request.mode,
            "num_topics": str(request.num_topics)
        }
    ))
