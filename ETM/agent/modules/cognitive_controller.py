"""
认知控制模块 (Cognitive Controller Module)

整合主题信息与大语言模型，实现意图理解、上下文管理、推理规划和工具选择。
该模块是Agent的决策中心，负责协调各个组件的工作。
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))

# 导入其他模块
from agent.modules.topic_aware import TopicAwareModule
from agent.memory.memory_system import MemorySystem
from agent.utils.tool_registry import ToolRegistry
from agent.utils.llm_client import LLMClient


class CognitiveController:
    """
    认知控制模块，整合主题信息与大语言模型。
    
    功能：
    1. 意图理解：结合主题信息理解用户意图
    2. 上下文管理：维护对话状态和主题演化
    3. 推理规划：基于主题相关性进行推理和规划
    4. 工具选择：根据主题和意图选择合适的工具
    """
    
    def __init__(
        self,
        topic_module: TopicAwareModule,
        memory_system: MemorySystem,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        dev_mode: bool = False
    ):
        """
        初始化认知控制模块。
        
        Args:
            topic_module: 主题感知模块
            memory_system: 记忆系统
            llm_client: 大语言模型客户端
            tool_registry: 工具注册表（可选）
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.topic_module = topic_module
        self.memory = memory_system
        self.llm = llm_client
        self.tools = tool_registry or ToolRegistry()
        self.dev_mode = dev_mode
        
        if self.dev_mode:
            print(f"[CognitiveController] Initialized successfully")
    
    def process_input(
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
        # 获取会话上下文
        context = self.memory.get_context(session_id)
        
        # 获取主题分布
        topic_dist = self.topic_module.get_topic_distribution(user_input)
        dominant_topics = self.topic_module.get_dominant_topics(topic_dist)
        
        # 检测主题变化
        topic_shifted = False
        if context.get("last_topic_dist") is not None:
            topic_shifted = self.topic_module.detect_topic_shift(
                context["last_topic_dist"],
                topic_dist
            )
        
        # 构建增强提示
        if use_topic_enhancement:
            topic_info = self._get_topic_context(dominant_topics)
            enhanced_prompt = self._build_prompt(user_input, topic_info, context, topic_shifted)
        else:
            enhanced_prompt = user_input
        
        # 选择相关工具
        available_tools = None
        if use_tools:
            available_tools = self._select_relevant_tools(dominant_topics)
        
        # LLM推理
        response = self.llm.generate(
            prompt=enhanced_prompt,
            context=context,
            tools=available_tools
        )
        
        # 处理工具调用
        if use_tools and "tool_calls" in response:
            response = self._handle_tool_calls(response, session_id)
        
        # 更新记忆
        self.memory.add(
            session_id=session_id,
            user_input=user_input,
            response=response["content"],
            topic_dist=topic_dist,
            metadata={
                "dominant_topics": dominant_topics,
                "topic_shifted": topic_shifted
            }
        )
        
        # 返回结果
        result = {
            "content": response["content"],
            "topic_dist": topic_dist.tolist(),
            "dominant_topics": dominant_topics,
            "topic_shifted": topic_shifted
        }
        
        if "thinking" in response:
            result["thinking"] = response["thinking"]
        
        return result
    
    def _get_topic_context(
        self,
        dominant_topics: List[Tuple[int, float]]
    ) -> Dict[str, Any]:
        """
        获取主题上下文信息。
        
        Args:
            dominant_topics: 主导主题列表 [(topic_idx, weight), ...]
            
        Returns:
            主题上下文信息
        """
        topic_context = {
            "topics": []
        }
        
        for topic_idx, weight in dominant_topics:
            topic_words = self.topic_module.get_topic_words(topic_idx)
            
            topic_context["topics"].append({
                "id": topic_idx,
                "weight": float(weight),
                "keywords": [word for word, _ in topic_words],
                "keyword_weights": [float(w) for _, w in topic_words]
            })
        
        return topic_context
    
    def _build_prompt(
        self,
        user_input: str,
        topic_info: Dict[str, Any],
        context: Dict[str, Any],
        topic_shifted: bool
    ) -> str:
        """
        构建增强提示。
        
        Args:
            user_input: 用户输入
            topic_info: 主题信息
            context: 对话上下文
            topic_shifted: 主题是否发生变化
            
        Returns:
            增强后的提示
        """
        # 基本提示
        prompt = user_input
        
        # 添加主题信息
        if topic_info["topics"]:
            prompt += "\n\n相关主题上下文:"
            
            for topic in topic_info["topics"]:
                keywords_str = ", ".join(topic["keywords"][:5])
                prompt += f"\n- 主题 {topic['id']} (权重: {topic['weight']:.2f}): {keywords_str}"
        
        # 如果主题发生变化，添加提示
        if topic_shifted and context.get("conversation_history"):
            prompt += "\n\n注意：对话主题似乎已经发生变化。"
        
        return prompt
    
    def _select_relevant_tools(
        self,
        dominant_topics: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """
        根据主题选择相关工具。
        
        Args:
            dominant_topics: 主导主题列表
            
        Returns:
            工具列表
        """
        # 如果没有工具，返回空列表
        if not self.tools or not self.tools.get_all_tools():
            return []
        
        # 获取所有工具
        all_tools = self.tools.get_all_tools()
        
        # 如果没有主题，返回所有工具
        if not dominant_topics:
            return all_tools
        
        # 根据主题选择工具
        # 这里可以实现更复杂的工具选择逻辑
        # 目前简单返回所有工具
        return all_tools
    
    def _handle_tool_calls(
        self,
        response: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        处理工具调用。
        
        Args:
            response: LLM响应
            session_id: 会话ID
            
        Returns:
            处理后的响应
        """
        if not response.get("tool_calls"):
            return response
        
        tool_results = []
        
        for tool_call in response["tool_calls"]:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            # 调用工具
            result = self.tools.call_tool(tool_name, tool_args)
            
            tool_results.append({
                "name": tool_name,
                "result": result
            })
        
        # 将工具结果添加到上下文
        self.memory.add_tool_results(session_id, tool_results)
        
        # 使用工具结果生成最终响应
        final_response = self.llm.generate_with_tool_results(
            original_response=response,
            tool_results=tool_results,
            context=self.memory.get_context(session_id)
        )
        
        return final_response
    
    def plan(
        self,
        user_input: str,
        topic_dist: np.ndarray,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据用户输入和主题分布生成计划。
        
        Args:
            user_input: 用户输入
            topic_dist: 主题分布
            context: 对话上下文
            
        Returns:
            计划
        """
        # 获取主导主题
        dominant_topics = self.topic_module.get_dominant_topics(topic_dist)
        
        # 构建规划提示
        planning_prompt = self._build_planning_prompt(user_input, dominant_topics, context)
        
        # 生成计划
        plan = self.llm.generate_plan(planning_prompt, context)
        
        return plan
    
    def _build_planning_prompt(
        self,
        user_input: str,
        dominant_topics: List[Tuple[int, float]],
        context: Dict[str, Any]
    ) -> str:
        """
        构建规划提示。
        
        Args:
            user_input: 用户输入
            dominant_topics: 主导主题
            context: 对话上下文
            
        Returns:
            规划提示
        """
        prompt = f"请为以下用户请求制定一个详细的响应计划:\n\n{user_input}\n\n"
        
        # 添加主题信息
        if dominant_topics:
            prompt += "相关主题:\n"
            
            for topic_idx, weight in dominant_topics:
                topic_words = self.topic_module.get_topic_words(topic_idx)
                words_str = ", ".join([word for word, _ in topic_words[:5]])
                prompt += f"- 主题 {topic_idx} (权重: {weight:.2f}): {words_str}\n"
        
        # 添加上下文信息
        if context.get("conversation_history"):
            prompt += "\n对话历史摘要:\n"
            # 这里可以添加对话历史的摘要
        
        prompt += "\n请提供以下内容:\n"
        prompt += "1. 用户意图分析\n"
        prompt += "2. 需要检索的信息\n"
        prompt += "3. 响应步骤\n"
        prompt += "4. 需要使用的工具\n"
        
        return prompt
    
    def execute_plan(
        self,
        plan: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        执行计划。
        
        Args:
            plan: 计划
            session_id: 会话ID
            
        Returns:
            执行结果
        """
        # 获取上下文
        context = self.memory.get_context(session_id)
        
        # 执行计划中的步骤
        results = []
        
        for step in plan.get("steps", []):
            step_type = step.get("type")
            step_params = step.get("params", {})
            
            if step_type == "tool_call":
                # 调用工具
                tool_name = step_params.get("name")
                tool_args = step_params.get("arguments", {})
                
                result = self.tools.call_tool(tool_name, tool_args)
                results.append({"type": "tool_result", "result": result})
                
            elif step_type == "llm_call":
                # 调用LLM
                prompt = step_params.get("prompt")
                
                response = self.llm.generate(prompt, context)
                results.append({"type": "llm_result", "result": response})
        
        # 生成最终响应
        final_response = self.llm.generate_from_plan_results(
            plan=plan,
            results=results,
            context=context
        )
        
        # 更新记忆
        self.memory.add_plan_execution(
            session_id=session_id,
            plan=plan,
            results=results,
            final_response=final_response
        )
        
        return final_response


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试认知控制模块")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM模型路径")
    parser.add_argument("--vocab", type=str, required=True, help="词汇表路径")
    parser.add_argument("--input", type=str, required=True, help="测试输入")
    parser.add_argument("--dev_mode", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    # 初始化组件
    from agent.modules.topic_aware import TopicAwareModule
    from agent.memory.memory_system import MemorySystem
    from agent.utils.llm_client import LLMClient
    
    topic_module = TopicAwareModule(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        dev_mode=args.dev_mode
    )
    
    memory_system = MemorySystem()
    llm_client = LLMClient()
    
    # 初始化认知控制模块
    controller = CognitiveController(
        topic_module=topic_module,
        memory_system=memory_system,
        llm_client=llm_client,
        dev_mode=args.dev_mode
    )
    
    # 处理输入
    session_id = "test_session"
    result = controller.process_input(args.input, session_id)
    
    print(f"Response: {result['content']}")
    print(f"Dominant topics: {result['dominant_topics']}")
    print(f"Topic shifted: {result['topic_shifted']}")
