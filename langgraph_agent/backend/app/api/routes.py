"""
API Routes
REST endpoints for the THETA Agent System
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from ..schemas.agent import TaskRequest, TaskResponse, TaskStatus
from ..schemas.data import DatasetInfo, ResultInfo, VisualizationInfo, ProjectInfo, MetricsResponse
from ..agents.etm_agent import etm_agent
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", tags=["health"])
async def root():
    """Health check endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health", tags=["health"])
async def health_check():
    """Detailed health check"""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_id": settings.GPU_ID,
        "etm_dir_exists": settings.ETM_DIR.exists(),
        "data_dir_exists": settings.DATA_DIR.exists(),
        "result_dir_exists": settings.RESULT_DIR.exists()
    }


@router.get("/project", response_model=ProjectInfo, tags=["project"])
async def get_project_info():
    """Get project overview information"""
    import torch
    
    datasets = settings.get_available_datasets()
    results = settings.get_available_results()
    active_tasks = len([t for t in etm_agent.get_all_tasks().values() 
                       if t.get("status") == "running"])
    
    recent_results = []
    for r in results[:5]:
        recent_results.append(ResultInfo(
            dataset=r["dataset"],
            mode=r["mode"],
            timestamp="",
            path=r["path"]
        ))
    
    return ProjectInfo(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        datasets_count=len(datasets),
        results_count=len(results),
        active_tasks=active_tasks,
        gpu_available=torch.cuda.is_available(),
        gpu_id=settings.GPU_ID,
        recent_results=recent_results
    )


@router.get("/datasets", response_model=List[DatasetInfo], tags=["data"])
async def list_datasets():
    """List all available datasets"""
    datasets = []
    
    if not settings.DATA_DIR.exists():
        return datasets
    
    for dataset_dir in settings.DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            continue
        
        info = DatasetInfo(
            name=dataset_dir.name,
            path=str(dataset_dir),
            has_labels=False
        )
        
        for csv_file in csv_files:
            if "text_only" in csv_file.name or "cleaned" in csv_file.name:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, nrows=5)
                    info.size = len(pd.read_csv(csv_file))
                    info.columns = df.columns.tolist()
                    info.has_labels = any(col in df.columns for col in ['label', 'Label', 'labels'])
                except Exception:
                    pass
                break
        
        datasets.append(info)
    
    return datasets


@router.get("/datasets/{dataset_name}", response_model=DatasetInfo, tags=["data"])
async def get_dataset_info(dataset_name: str):
    """Get detailed information about a specific dataset"""
    dataset_dir = settings.DATA_DIR / dataset_name
    
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No CSV files found in dataset '{dataset_name}'")
    
    import pandas as pd
    
    csv_file = csv_files[0]
    for f in csv_files:
        if "text_only" in f.name or "cleaned" in f.name:
            csv_file = f
            break
    
    df = pd.read_csv(csv_file)
    
    return DatasetInfo(
        name=dataset_name,
        path=str(dataset_dir),
        size=len(df),
        columns=df.columns.tolist(),
        has_labels=any(col in df.columns for col in ['label', 'Label', 'labels']),
        language="english"
    )


@router.get("/results", response_model=List[ResultInfo], tags=["results"])
async def list_results():
    """List all training results"""
    results = []
    
    for result in settings.get_available_results():
        result_path = Path(result["path"])
        model_dir = result_path / "model"
        eval_dir = result_path / "evaluation"
        viz_dir = result_path / "visualization"
        
        info = ResultInfo(
            dataset=result["dataset"],
            mode=result["mode"],
            timestamp="",
            path=result["path"],
            has_model=model_dir.exists() and any(model_dir.glob("*.pt")),
            has_theta=(model_dir / "theta_*.npy").parent.exists() if model_dir.exists() else False,
            has_beta=(model_dir / "beta_*.npy").parent.exists() if model_dir.exists() else False,
            has_topic_words=any(model_dir.glob("topic_words_*.json")) if model_dir.exists() else False,
            has_visualizations=viz_dir.exists() and any(viz_dir.glob("*.png"))
        )
        
        if model_dir.exists():
            theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
            if theta_files:
                info.timestamp = theta_files[0].stem.replace("theta_", "")
        
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


@router.get("/results/{dataset}/{mode}", response_model=ResultInfo, tags=["results"])
async def get_result_info(dataset: str, mode: str):
    """Get detailed information about a specific result"""
    result_path = settings.get_result_path(dataset, mode)
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result not found for {dataset}/{mode}")
    
    model_dir = result_path / "model"
    eval_dir = result_path / "evaluation"
    viz_dir = result_path / "visualization"
    
    info = ResultInfo(
        dataset=dataset,
        mode=mode,
        timestamp="",
        path=str(result_path),
        has_model=any(model_dir.glob("*.pt")) if model_dir.exists() else False,
        has_theta=any(model_dir.glob("theta_*.npy")) if model_dir.exists() else False,
        has_beta=any(model_dir.glob("beta_*.npy")) if model_dir.exists() else False,
        has_topic_words=any(model_dir.glob("topic_words_*.json")) if model_dir.exists() else False,
        has_visualizations=any(viz_dir.glob("*.png")) if viz_dir.exists() else False
    )
    
    if model_dir.exists():
        theta_files = sorted(model_dir.glob("theta_*.npy"), reverse=True)
        if theta_files:
            info.timestamp = theta_files[0].stem.replace("theta_", "")
            
            history_file = model_dir / f"training_history_{info.timestamp}.json"
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    info.epochs_trained = history.get("epochs_trained")
    
    if eval_dir.exists():
        metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
        if metrics_files:
            with open(metrics_files[0]) as f:
                info.metrics = json.load(f)
    
    return info


@router.get("/results/{dataset}/{mode}/metrics", response_model=MetricsResponse, tags=["results"])
async def get_result_metrics(dataset: str, mode: str):
    """Get detailed metrics for a result"""
    result_path = settings.get_result_path(dataset, mode)
    eval_dir = result_path / "evaluation"
    
    if not eval_dir.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
    if not metrics_files:
        raise HTTPException(status_code=404, detail="No metrics files found")
    
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    
    timestamp = metrics_files[0].stem.replace("metrics_", "")
    
    return MetricsResponse(
        dataset=dataset,
        mode=mode,
        timestamp=timestamp,
        topic_coherence_avg=metrics.get("topic_coherence_avg"),
        topic_coherence_per_topic=metrics.get("topic_coherence_per_topic"),
        topic_diversity_td=metrics.get("topic_diversity_td"),
        topic_diversity_irbo=metrics.get("topic_diversity_irbo"),
        additional=metrics
    )


@router.get("/results/{dataset}/{mode}/topic-words", tags=["results"])
async def get_topic_words(dataset: str, mode: str, top_k: int = Query(default=10, ge=1, le=50)):
    """Get topic words for a result"""
    result_path = settings.get_result_path(dataset, mode)
    model_dir = result_path / "model"
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model results not found")
    
    topic_files = sorted(model_dir.glob("topic_words_*.json"), reverse=True)
    if not topic_files:
        raise HTTPException(status_code=404, detail="Topic words not found")
    
    with open(topic_files[0]) as f:
        topic_words = json.load(f)
    
    if top_k < 20:
        topic_words = {k: v[:top_k] for k, v in topic_words.items()}
    
    return topic_words


@router.get("/results/{dataset}/{mode}/visualizations", response_model=List[VisualizationInfo], tags=["results"])
async def list_visualizations(dataset: str, mode: str):
    """List all visualizations for a result"""
    result_path = settings.get_result_path(dataset, mode)
    viz_dir = result_path / "visualization"
    
    if not viz_dir.exists():
        return []
    
    visualizations = []
    for viz_file in viz_dir.iterdir():
        if viz_file.is_file():
            file_type = "image" if viz_file.suffix in [".png", ".jpg", ".jpeg"] else \
                       "html" if viz_file.suffix == ".html" else "other"
            
            visualizations.append(VisualizationInfo(
                name=viz_file.name,
                path=str(viz_file),
                type=file_type,
                size=viz_file.stat().st_size
            ))
    
    return visualizations


@router.get("/results/{dataset}/{mode}/visualizations/{filename}", tags=["results"])
async def get_visualization(dataset: str, mode: str, filename: str):
    """Get a specific visualization file"""
    result_path = settings.get_result_path(dataset, mode)
    viz_path = result_path / "visualization" / filename
    
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(viz_path)


@router.post("/tasks", response_model=TaskResponse, tags=["tasks"])
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and start a new ETM pipeline task"""
    dataset_dir = settings.DATA_DIR / request.dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")
    
    embeddings_dir = settings.get_result_path(request.dataset, request.mode) / "embeddings"
    emb_file = embeddings_dir / f"{request.dataset}_{request.mode}_embeddings.npy"
    if not emb_file.exists():
        raise HTTPException(
            status_code=400, 
            detail=f"Embeddings not found for {request.dataset}/{request.mode}. Please generate embeddings first."
        )
    
    from ..schemas.agent import AgentState
    initial_state = {
        "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "dataset": request.dataset,
        "mode": request.mode,
        "status": "pending",
        "current_step": "preprocess",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    async def run_task():
        await etm_agent.run_pipeline(request)
    
    background_tasks.add_task(run_task)
    
    return TaskResponse(
        task_id=initial_state["task_id"],
        status=TaskStatus.PENDING,
        current_step="preprocess",
        progress=0,
        created_at=initial_state["created_at"],
        updated_at=initial_state["updated_at"]
    )


@router.get("/tasks", response_model=List[TaskResponse], tags=["tasks"])
async def list_tasks():
    """List all tasks"""
    tasks = []
    for task_id, state in etm_agent.get_all_tasks().items():
        tasks.append(TaskResponse(
            task_id=task_id,
            status=TaskStatus(state.get("status", "pending")),
            current_step=state.get("current_step"),
            progress=_calculate_progress(state),
            metrics=state.get("metrics"),
            created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(state.get("updated_at", datetime.now().isoformat())),
            completed_at=datetime.fromisoformat(state["completed_at"]) if state.get("completed_at") else None,
            error_message=state.get("error_message")
        ))
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str):
    """Get task status"""
    state = etm_agent.get_task_status(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus(state.get("status", "pending")),
        current_step=state.get("current_step"),
        progress=_calculate_progress(state),
        metrics=state.get("metrics"),
        topic_words=state.get("topic_words"),
        visualization_paths=state.get("visualization_paths"),
        created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
        updated_at=datetime.fromisoformat(state.get("updated_at", datetime.now().isoformat())),
        completed_at=datetime.fromisoformat(state["completed_at"]) if state.get("completed_at") else None,
        error_message=state.get("error_message")
    )


@router.delete("/tasks/{task_id}", tags=["tasks"])
async def cancel_task(task_id: str):
    """Cancel a running task"""
    success = await etm_agent.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    return {"message": "Task cancelled", "task_id": task_id}


def _calculate_progress(state: dict) -> float:
    """Calculate task progress percentage"""
    steps = ["preprocess", "embedding", "training", "evaluation", "visualization"]
    completed_flags = [
        state.get("preprocess_completed", False),
        state.get("embedding_completed", False),
        state.get("training_completed", False),
        state.get("evaluation_completed", False),
        state.get("visualization_completed", False)
    ]
    
    completed = sum(completed_flags)
    
    if state.get("status") == "completed":
        return 100.0
    elif state.get("status") == "failed":
        return completed / len(steps) * 100
    else:
        return completed / len(steps) * 100
