"""
大语言模型客户端 (LLM Client)

提供与大语言模型交互的接口，支持不同的模型和API。
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional, Union


class LLMClient:
    """
    大语言模型客户端，提供与大语言模型交互的接口。
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        dev_mode: bool = False
    ):
        """
        初始化大语言模型客户端。
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            api_base: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            dev_mode: 是否开启开发模式（打印调试信息）
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dev_mode = dev_mode
        
        # 检查API密钥
        if not self.api_key and not self.dev_mode:
            print("[WARNING] No API key provided. Using mock responses.")
        
        if self.dev_mode:
            print(f"[LLMClient] Initialized with model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成文本响应。
        
        Args:
            prompt: 提示文本
            context: 上下文信息
            tools: 可用工具列表
            
        Returns:
            包含响应内容的字典
        """
        if not self.api_key:
            return self._mock_response(prompt, context, tools)
        
        try:
            # 构建消息
            messages = self._build_messages(prompt, context)
            
            # 构建请求
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # 添加工具（如果提供）
            if tools:
                request_data["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                    for tool in tools
                ]
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # 解析响应
            if response.status_code == 200:
                response_data = response.json()
                
                # 提取内容
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                
                # 提取工具调用
                tool_calls = message.get("tool_calls", [])
                
                result = {
                    "content": content
                }
                
                if tool_calls:
                    result["tool_calls"] = [
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"])
                        }
                        for tool_call in tool_calls
                    ]
                
                return result
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "content": f"抱歉，我遇到了一个问题：{error_message}",
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "抱歉，我遇到了一个问题，无法生成响应。",
                "error": error_message
            }
    
    def generate_with_tool_results(
        self,
        original_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        使用工具结果生成最终响应。
        
        Args:
            original_response: 原始响应
            tool_results: 工具调用结果
            context: 上下文信息
            
        Returns:
            最终响应
        """
        if not self.api_key:
            return self._mock_tool_response(original_response, tool_results)
        
        try:
            # 构建消息
            messages = self._build_messages("", context)
            
            # 添加原始响应
            messages.append({
                "role": "assistant",
                "content": original_response.get("content", ""),
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"])
                        }
                    }
                    for i, tool_call in enumerate(original_response.get("tool_calls", []))
                ]
            })
            
            # 添加工具结果
            for i, result in enumerate(tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": result["name"],
                    "content": json.dumps(result["result"])
                })
            
            # 构建请求
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # 解析响应
            if response.status_code == 200:
                response_data = response.json()
                
                # 提取内容
                message = response_data["choices"][0]["message"]
                content = message.get("content", "")
                
                return {
                    "content": content
                }
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "content": f"抱歉，我遇到了一个问题：{error_message}",
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "抱歉，我遇到了一个问题，无法生成响应。",
                "error": error_message
            }
    
    def generate_plan(
        self,
        planning_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成计划。
        
        Args:
            planning_prompt: 规划提示
            context: 上下文信息
            
        Returns:
            计划
        """
        if not self.api_key:
            return self._mock_plan(planning_prompt)
        
        try:
            # 构建消息
            messages = self._build_messages(planning_prompt, context)
            
            # 添加系统提示
            messages.insert(0, {
                "role": "system",
                "content": "你是一个专业的规划助手，请根据用户的请求制定详细的响应计划。"
            })
            
            # 构建请求
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.5,  # 降低温度，使规划更确定
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"}  # 请求JSON格式的响应
            }
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            api_url = f"{self.api_base}/v1/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=request_data
            )
            
            # 解析响应
            if response.status_code == 200:
                response_data = response.json()
                
                # 提取内容
                content = response_data["choices"][0]["message"]["content"]
                
                # 解析JSON
                try:
                    plan = json.loads(content)
                    return plan
                except json.JSONDecodeError:
                    if self.dev_mode:
                        print(f"[LLMClient] Failed to parse JSON: {content}")
                    
                    return {
                        "steps": [],
                        "error": "Failed to parse JSON response"
                    }
            else:
                error_message = f"API request failed with status code {response.status_code}: {response.text}"
                if self.dev_mode:
                    print(f"[LLMClient] {error_message}")
                
                return {
                    "steps": [],
                    "error": error_message
                }
        
        except Exception as e:
            error_message = f"Error generating plan: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "steps": [],
                "error": error_message
            }
    
    def generate_from_plan_results(
        self,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        根据计划执行结果生成最终响应。
        
        Args:
            plan: 计划
            results: 执行结果
            context: 上下文信息
            
        Returns:
            最终响应
        """
        if not self.api_key:
            return self._mock_plan_response(plan, results)
        
        try:
            # 构建提示
            prompt = "请根据以下计划和执行结果生成最终响应：\n\n"
            
            # 添加计划
            prompt += "计划：\n"
            prompt += json.dumps(plan, ensure_ascii=False, indent=2)
            prompt += "\n\n"
            
            # 添加执行结果
            prompt += "执行结果：\n"
            prompt += json.dumps(results, ensure_ascii=False, indent=2)
            prompt += "\n\n"
            
            # 添加指令
            prompt += "请生成一个综合以上信息的完整响应。"
            
            # 生成响应
            return self.generate(prompt, context)
        
        except Exception as e:
            error_message = f"Error generating response from plan results: {str(e)}"
            if self.dev_mode:
                print(f"[LLMClient] {error_message}")
            
            return {
                "content": "抱歉，我遇到了一个问题，无法生成响应。",
                "error": error_message
            }
    
    def _build_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        构建消息列表。
        
        Args:
            prompt: 提示文本
            context: 上下文信息
            
        Returns:
            消息列表
        """
        messages = []
        
        # 添加系统消息
        messages.append({
            "role": "system",
            "content": "你是一个基于主题模型的智能助手，能够理解用户的主题兴趣，并提供相关的回答。"
        })
        
        # 添加对话历史
        if context and "conversation_history" in context:
            for entry in context["conversation_history"]:
                messages.append({
                    "role": "user",
                    "content": entry["user_input"]
                })
                
                messages.append({
                    "role": "assistant",
                    "content": entry["response"]
                })
        
        # 添加当前提示
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _mock_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成模拟响应（用于开发模式）。
        
        Args:
            prompt: 提示文本
            context: 上下文信息
            tools: 可用工具列表
            
        Returns:
            模拟响应
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock response for prompt: {prompt[:50]}...")
        
        # 简单的模拟响应
        response = {
            "content": f"这是对'{prompt[:20]}...'的模拟响应。在实际应用中，这里会返回大语言模型的真实响应。"
        }
        
        # 如果提供了工具，随机选择一个工具调用
        if tools and "搜索" in prompt.lower():
            response["tool_calls"] = [
                {
                    "name": "search",
                    "arguments": {
                        "query": prompt[:30]
                    }
                }
            ]
        
        return response
    
    def _mock_tool_response(
        self,
        original_response: Dict[str, Any],
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成模拟工具响应（用于开发模式）。
        
        Args:
            original_response: 原始响应
            tool_results: 工具调用结果
            
        Returns:
            模拟响应
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock tool response")
        
        # 简单的模拟响应
        tool_names = [result["name"] for result in tool_results]
        
        return {
            "content": f"根据工具{', '.join(tool_names)}的调用结果，我可以提供以下信息：\n\n"
                      f"这是一个模拟响应。在实际应用中，这里会返回大语言模型基于工具结果的真实响应。"
        }
    
    def _mock_plan(
        self,
        planning_prompt: str
    ) -> Dict[str, Any]:
        """
        生成模拟计划（用于开发模式）。
        
        Args:
            planning_prompt: 规划提示
            
        Returns:
            模拟计划
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock plan for prompt: {planning_prompt[:50]}...")
        
        # 简单的模拟计划
        return {
            "user_intent": "用户想要了解某个主题",
            "steps": [
                {
                    "type": "tool_call",
                    "params": {
                        "name": "search",
                        "arguments": {
                            "query": "模拟搜索查询"
                        }
                    }
                },
                {
                    "type": "llm_call",
                    "params": {
                        "prompt": "根据搜索结果生成响应"
                    }
                }
            ]
        }
    
    def _mock_plan_response(
        self,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成模拟计划响应（用于开发模式）。
        
        Args:
            plan: 计划
            results: 执行结果
            
        Returns:
            模拟响应
        """
        if self.dev_mode:
            print(f"[LLMClient] Using mock plan response")
        
        # 简单的模拟响应
        return {
            "content": "这是基于执行计划的模拟响应。在实际应用中，这里会返回大语言模型基于计划执行结果的真实响应。"
        }
