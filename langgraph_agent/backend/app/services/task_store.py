"""
Task Store - 任务持久化存储
支持 JSON 文件持久化，确保服务器重启后任务不丢失
"""

import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class TaskStore:
    """
    任务持久化存储
    - 内存缓存 + JSON 文件持久化
    - 支持异步操作
    - 线程安全
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else settings.RESULT_DIR / "tasks.json"
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._load_tasks()
    
    def _load_tasks(self):
        """从文件加载任务"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self._tasks = json.load(f)
                logger.info(f"Loaded {len(self._tasks)} tasks from {self.storage_path}")
            else:
                self._tasks = {}
                logger.info(f"No existing tasks file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
            self._tasks = {}
    
    def _save_tasks(self):
        """保存任务到文件"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self._tasks, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")
    
    def create_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建新任务"""
        with self._lock:
            now = datetime.now().isoformat()
            task = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0,
                "current_step": None,
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
                "error_message": None,
                "logs": [],
                **task_data
            }
            self._tasks[task_id] = task
            self._save_tasks()
            logger.info(f"Created task: {task_id}")
            return task
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务"""
        with self._lock:
            return self._tasks.copy()
    
    def get_tasks_list(self, 
                       status: Optional[str] = None, 
                       dataset: Optional[str] = None,
                       limit: int = 100,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """获取任务列表，支持过滤和分页"""
        with self._lock:
            tasks = list(self._tasks.values())
            
            # 按创建时间倒序
            tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # 过滤
            if status:
                tasks = [t for t in tasks if t.get("status") == status]
            if dataset:
                tasks = [t for t in tasks if t.get("dataset") == dataset]
            
            # 分页
            return tasks[offset:offset + limit]
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新任务"""
        with self._lock:
            if task_id not in self._tasks:
                return None
            
            self._tasks[task_id].update(updates)
            self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._save_tasks()
            return self._tasks[task_id]
    
    def update_progress(self, task_id: str, progress: float, 
                        current_step: Optional[str] = None,
                        message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """更新任务进度"""
        updates = {
            "progress": progress,
            "updated_at": datetime.now().isoformat()
        }
        if current_step:
            updates["current_step"] = current_step
        if message:
            updates["message"] = message
        return self.update_task(task_id, updates)
    
    def add_log(self, task_id: str, step: str, status: str, message: str) -> None:
        """添加任务日志"""
        with self._lock:
            if task_id in self._tasks:
                log_entry = {
                    "step": step,
                    "status": status,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }
                if "logs" not in self._tasks[task_id]:
                    self._tasks[task_id]["logs"] = []
                self._tasks[task_id]["logs"].append(log_entry)
                self._tasks[task_id]["updated_at"] = datetime.now().isoformat()
                self._save_tasks()
    
    def set_running(self, task_id: str) -> Optional[Dict[str, Any]]:
        """设置任务为运行中"""
        return self.update_task(task_id, {"status": "running"})
    
    def set_completed(self, task_id: str, results: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """设置任务为已完成"""
        updates = {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat()
        }
        if results:
            updates.update(results)
        return self.update_task(task_id, updates)
    
    def set_failed(self, task_id: str, error_message: str) -> Optional[Dict[str, Any]]:
        """设置任务为失败"""
        return self.update_task(task_id, {
            "status": "failed",
            "error_message": error_message
        })
    
    def set_cancelled(self, task_id: str) -> Optional[Dict[str, Any]]:
        """设置任务为已取消"""
        return self.update_task(task_id, {"status": "cancelled"})
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save_tasks()
                logger.info(f"Deleted task: {task_id}")
                return True
            return False
    
    def cleanup_old_tasks(self, days: int = 30) -> int:
        """清理指定天数前的已完成/失败任务"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._lock:
            tasks_to_delete = []
            for task_id, task in self._tasks.items():
                if task.get("status") in ["completed", "failed", "cancelled"]:
                    completed_at = task.get("completed_at") or task.get("updated_at")
                    if completed_at and completed_at < cutoff:
                        tasks_to_delete.append(task_id)
            
            for task_id in tasks_to_delete:
                del self._tasks[task_id]
            
            if tasks_to_delete:
                self._save_tasks()
                logger.info(f"Cleaned up {len(tasks_to_delete)} old tasks")
            
            return len(tasks_to_delete)
    
    def get_recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的任务列表"""
        return self.get_tasks_list(limit=limit, offset=0)
    
    def get_stats(self) -> Dict[str, int]:
        """获取任务统计"""
        with self._lock:
            stats = {
                "total": len(self._tasks),
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0
            }
            for task in self._tasks.values():
                status = task.get("status", "pending")
                if status in stats:
                    stats[status] += 1
            return stats


# 全局单例
task_store = TaskStore()
