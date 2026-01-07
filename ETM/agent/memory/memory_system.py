"""
记忆系统 (Memory System)

管理对话历史、主题演化和知识缓存。
该模块负责存储和检索Agent的记忆。
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))


class MemorySystem:
    """
    记忆系统，管理对话历史、主题演化和知识缓存。
    
    功能：
    1. 短期记忆：存储最近的对话历史
    2. 长期记忆：存储重要信息和知识
    3. 主题记忆：跟踪主题演化
    """
    
    def __init__(
        self,
        max_history_length: int = 10,
        max_topic_history_length: int = 5,
        dev_mode: bool = False
    ):
        """
        初始化记忆系统。
        
        Args:
            max_history_length: 最大对话历史长度
            max_topic_history_length: 最大主题历史长度
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.max_history_length = max_history_length
        self.max_topic_history_length = max_topic_history_length
        self.dev_mode = dev_mode
        
        # 会话记忆
        self.sessions = defaultdict(lambda: {
            "conversation_history": [],
            "topic_history": [],
            "tool_results": [],
            "plan_history": [],
            "knowledge_cache": {},
            "metadata": {}
        })
        
        if self.dev_mode:
            print(f"[MemorySystem] Initialized successfully")
    
    def add(
        self,
        session_id: str,
        user_input: str,
        response: str,
        topic_dist: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加对话记录到记忆。
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            response: 系统响应
            topic_dist: 主题分布
            metadata: 元数据
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 添加对话记录
        timestamp = time.time()
        
        conversation_entry = {
            "timestamp": timestamp,
            "user_input": user_input,
            "response": response,
            "metadata": metadata or {}
        }
        
        session["conversation_history"].append(conversation_entry)
        
        # 限制对话历史长度
        if len(session["conversation_history"]) > self.max_history_length:
            session["conversation_history"] = session["conversation_history"][-self.max_history_length:]
        
        # 添加主题记录（如果提供）
        if topic_dist is not None:
            topic_entry = {
                "timestamp": timestamp,
                "topic_dist": topic_dist.tolist(),
                "metadata": metadata or {}
            }
            
            session["topic_history"].append(topic_entry)
            
            # 限制主题历史长度
            if len(session["topic_history"]) > self.max_topic_history_length:
                session["topic_history"] = session["topic_history"][-self.max_topic_history_length:]
            
            # 更新最后的主题分布
            session["metadata"]["last_topic_dist"] = topic_dist.tolist()
    
    def add_tool_results(
        self,
        session_id: str,
        tool_results: List[Dict[str, Any]]
    ) -> None:
        """
        添加工具调用结果到记忆。
        
        Args:
            session_id: 会话ID
            tool_results: 工具调用结果
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 添加工具结果
        timestamp = time.time()
        
        for result in tool_results:
            tool_entry = {
                "timestamp": timestamp,
                "name": result.get("name", "unknown_tool"),
                "result": result.get("result"),
                "metadata": result.get("metadata", {})
            }
            
            session["tool_results"].append(tool_entry)
    
    def add_plan_execution(
        self,
        session_id: str,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        final_response: Dict[str, Any]
    ) -> None:
        """
        添加计划执行记录到记忆。
        
        Args:
            session_id: 会话ID
            plan: 计划
            results: 执行结果
            final_response: 最终响应
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 添加计划执行记录
        timestamp = time.time()
        
        plan_entry = {
            "timestamp": timestamp,
            "plan": plan,
            "results": results,
            "final_response": final_response
        }
        
        session["plan_history"].append(plan_entry)
    
    def add_knowledge(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """
        添加知识到缓存。
        
        Args:
            session_id: 会话ID
            key: 知识键
            value: 知识值
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 添加知识
        session["knowledge_cache"][key] = value
    
    def get_context(
        self,
        session_id: str,
        include_tool_results: bool = True,
        include_plan_history: bool = False
    ) -> Dict[str, Any]:
        """
        获取会话上下文。
        
        Args:
            session_id: 会话ID
            include_tool_results: 是否包含工具调用结果
            include_plan_history: 是否包含计划历史
            
        Returns:
            会话上下文
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 构建上下文
        context = {
            "conversation_history": session["conversation_history"],
            "topic_history": session["topic_history"],
            "knowledge_cache": session["knowledge_cache"],
            "metadata": session["metadata"]
        }
        
        # 添加工具调用结果（如果需要）
        if include_tool_results:
            context["tool_results"] = session["tool_results"]
        
        # 添加计划历史（如果需要）
        if include_plan_history:
            context["plan_history"] = session["plan_history"]
        
        return context
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史。
        
        Args:
            session_id: 会话ID
            limit: 返回的记录数量
            
        Returns:
            对话历史
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 获取对话历史
        history = session["conversation_history"]
        
        # 限制记录数量
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_topic_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取主题历史。
        
        Args:
            session_id: 会话ID
            limit: 返回的记录数量
            
        Returns:
            主题历史
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 获取主题历史
        history = session["topic_history"]
        
        # 限制记录数量
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_last_topic_dist(
        self,
        session_id: str
    ) -> Optional[np.ndarray]:
        """
        获取最后的主题分布。
        
        Args:
            session_id: 会话ID
            
        Returns:
            主题分布
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 获取最后的主题分布
        last_topic_dist = session["metadata"].get("last_topic_dist")
        
        if last_topic_dist is not None:
            return np.array(last_topic_dist)
        
        return None
    
    def get_knowledge(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        获取知识缓存。
        
        Args:
            session_id: 会话ID
            key: 知识键
            default: 默认值
            
        Returns:
            知识值
        """
        # 获取会话
        session = self.sessions[session_id]
        
        # 获取知识
        return session["knowledge_cache"].get(key, default)
    
    def clear_session(
        self,
        session_id: str
    ) -> None:
        """
        清除会话。
        
        Args:
            session_id: 会话ID
        """
        # 清除会话
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def save(
        self,
        path: str,
        session_id: Optional[str] = None
    ) -> None:
        """
        保存记忆到文件。
        
        Args:
            path: 保存路径
            session_id: 会话ID（如果为None，保存所有会话）
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备数据
        if session_id is not None:
            # 保存单个会话
            if session_id in self.sessions:
                data = {session_id: self.sessions[session_id]}
            else:
                data = {}
        else:
            # 保存所有会话
            data = dict(self.sessions)
        
        # 保存到文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(
        cls,
        path: str,
        dev_mode: bool = False
    ) -> 'MemorySystem':
        """
        从文件加载记忆。
        
        Args:
            path: 加载路径
            dev_mode: 是否开启开发模式
            
        Returns:
            记忆系统实例
        """
        # 创建实例
        instance = cls(dev_mode=dev_mode)
        
        # 加载数据
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 设置会话
        for session_id, session_data in data.items():
            instance.sessions[session_id] = session_data
        
        return instance


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试记忆系统")
    parser.add_argument("--dev_mode", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    # 初始化记忆系统
    memory = MemorySystem(dev_mode=args.dev_mode)
    
    # 测试添加对话记录
    session_id = "test_session"
    
    # 模拟主题分布
    topic_dist = np.array([0.1, 0.2, 0.5, 0.1, 0.1])
    
    # 添加对话记录
    memory.add(
        session_id=session_id,
        user_input="你好，我想了解气候变化。",
        response="气候变化是指地球气候系统的长期变化，包括温度、降水和风型等。",
        topic_dist=topic_dist,
        metadata={"dominant_topics": [(2, 0.5)]}
    )
    
    # 添加知识
    memory.add_knowledge(
        session_id=session_id,
        key="user_interests",
        value=["气候变化", "可再生能源"]
    )
    
    # 获取上下文
    context = memory.get_context(session_id)
    
    print("Context:")
    print(f"  Conversation history: {len(context['conversation_history'])} entries")
    print(f"  Topic history: {len(context['topic_history'])} entries")
    print(f"  Knowledge cache: {context['knowledge_cache']}")
    
    # 获取最后的主题分布
    last_topic_dist = memory.get_last_topic_dist(session_id)
    print(f"Last topic distribution: {last_topic_dist}")
    
    # 保存记忆
    memory.save("test_memory.json", session_id)
    print("Memory saved to test_memory.json")
