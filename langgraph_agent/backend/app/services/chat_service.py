"""
Chat Service
Handles conversational interactions and intent parsing using Qwen API
"""

import re
import json
from typing import Optional, Dict, Any, Tuple, List
from ..schemas.agent import ChatRequest, ChatResponse, TaskRequest
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Try to import dashscope
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning("dashscope not installed. Chat service will use rule-based fallback.")


class ChatService:
    """
    Service for handling chat interactions using Qwen API
    Parses user intent and generates appropriate responses/actions
    """
    
    INTENT_PATTERNS = {
        "train": [
            r"train.*(?:on|with|using)?\s*(\w+)",
            r"run.*pipeline.*(\w+)",
            r"start.*training.*(\w+)",
            r"analyze.*(\w+)",
        ],
        "status": [
            r"status.*task",
            r"how.*going",
            r"progress",
            r"what.*running",
        ],
        "results": [
            r"show.*results?",
            r"get.*results?",
            r"view.*results?",
            r"results?\s+(?:for|of)?\s*(\w+)",
        ],
        "topics": [
            r"show.*topics?",
            r"what.*topics?",
            r"topic.*words?",
        ],
        "datasets": [
            r"list.*datasets?",
            r"available.*datasets?",
            r"what.*datasets?",
        ],
        "help": [
            r"help",
            r"what.*can.*do",
            r"how.*use",
        ],
    }
    
    def __init__(self):
        self.conversation_context: Dict[str, Any] = {}
        self.use_qwen_api = DASHSCOPE_AVAILABLE and settings.QWEN_API_KEY is not None
        
        if self.use_qwen_api:
            dashscope.api_key = settings.QWEN_API_KEY
            logger.info(f"ChatService initialized with Qwen API (model: {settings.QWEN_MODEL})")
        else:
            logger.info("ChatService initialized with rule-based fallback")
    
    def parse_intent(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse user message to determine intent and extract parameters"""
        message_lower = message.lower().strip()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    params = {}
                    if match.groups():
                        params["dataset"] = match.group(1)
                    return intent, params
        
        return "unknown", {}
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message and generate response"""
        # Store context for reference
        if request.context:
            self.conversation_context["ui_context"] = request.context
        
        # Use Qwen API if available, otherwise fall back to rule-based
        if self.use_qwen_api:
            return self._process_with_qwen(request)
        else:
            return self._process_with_rules(request)
    
    def _process_with_qwen(self, request: ChatRequest) -> ChatResponse:
        """Process message using Qwen API"""
        try:
            # Get system context with UI context
            system_prompt = self._build_system_prompt(request.context)
            
            # Get conversation history from request or context
            conversation_history = request.context.get("conversation_history") if request.context else None
            
            # Get conversation history (if any)
            messages = self._build_messages(request.message, system_prompt, conversation_history)
            
            # Call Qwen API
            response = Generation.call(
                model=settings.QWEN_MODEL,
                messages=messages,
                result_format='message',
                temperature=0.7,
                max_tokens=2000,
            )
            
            if response.status_code == 200:
                ai_message = response.output.choices[0].message.content
                
                # Try to extract action and data from AI response
                parsed = self._parse_ai_response(ai_message, request.message)
                
                return ChatResponse(
                    message=ai_message,
                    action=parsed.get("action"),
                    data=parsed.get("data")
                )
            else:
                logger.error(f"Qwen API error: {response.status_code}, {response.message}")
                # Fall back to rule-based processing
                return self._process_with_rules(request)
                
        except Exception as e:
            logger.error(f"Error calling Qwen API: {e}", exc_info=True)
            # Fall back to rule-based processing
            return self._process_with_rules(request)
    
    def _process_with_rules(self, request: ChatRequest) -> ChatResponse:
        """Process message using rule-based intent parsing (fallback)"""
        intent, params = self.parse_intent(request.message)
        
        if intent == "train":
            return self._handle_train_intent(params, request.context)
        elif intent == "status":
            return self._handle_status_intent()
        elif intent == "results":
            return self._handle_results_intent(params)
        elif intent == "topics":
            return self._handle_topics_intent(params)
        elif intent == "datasets":
            return self._handle_datasets_intent()
        elif intent == "help":
            return self._handle_help_intent()
        else:
            return self._handle_unknown_intent(request.message)
    
    def _build_system_prompt(self, ui_context: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt with context information"""
        datasets = settings.get_available_datasets()
        results = settings.get_available_results()
        
        # Build UI context section
        ui_context_text = ""
        if ui_context:
            view_name = ui_context.get("current_view_name", "未知")
            current_view = ui_context.get("current_view", "")
            app_state = ui_context.get("app_state", "")
            datasets_count = ui_context.get("datasets_count", 0)
            user_datasets = ui_context.get("datasets", [])
            processing_jobs = ui_context.get("processing_jobs_count", 0)
            selected_dataset = ui_context.get("selected_dataset")
            
            ui_context_text = f"""
## 当前用户界面状态

- **当前页面**: {view_name} ({current_view})
- **应用状态**: {app_state}
- **用户数据集数量**: {datasets_count}
"""
            if user_datasets:
                datasets_info = ", ".join([f"{d.get('name', '未命名')}({d.get('fileCount', 0)}个文件)" for d in user_datasets[:5]])
                ui_context_text += f"- **用户数据集**: {datasets_info}\n"
            
            if processing_jobs > 0:
                ui_context_text += f"- **正在处理的任务**: {processing_jobs} 个\n"
            
            if selected_dataset:
                ui_context_text += f"- **当前选中数据集**: {selected_dataset}\n"
            
            # Add view-specific guidance
            view_guidance = {
                "data": "用户正在数据管理页面，可以帮助他们上传数据、创建数据集、查看现有数据。",
                "processing": "用户正在数据清洗页面，可以帮助他们清洗文本数据、去除噪音。",
                "embedding": "用户正在向量化页面，可以帮助他们生成词袋矩阵和词嵌入。",
                "training": "用户正在模型训练页面，可以帮助他们配置和启动训练任务，查看训练进度。",
                "results": "用户正在查看分析结果，可以帮助他们理解训练指标、主题词等。",
                "visualizations": "用户正在查看可视化图表，可以帮助他们解读图表含义。",
            }
            if current_view in view_guidance:
                ui_context_text += f"\n**页面指导**: {view_guidance[current_view]}\n"
        
        context_info = f"""
你是 THETA ETM 主题模型训练系统的 AI 助手。你可以**感知用户当前的界面状态**，并基于此提供精准的帮助和指导。

{ui_context_text}

## 系统功能

1. **数据管理**: 上传和管理数据集
   - 服务器可用数据集: {', '.join(datasets) if datasets else '无'}

2. **数据清洗**: 清洗和预处理文本数据

3. **向量化**: 生成词袋矩阵(BOW)和词嵌入(Embedding)

4. **训练模型**: 使用 ETM 主题模型进行训练
   - 支持的模式: zero_shot, supervised, unsupervised
   - 默认参数: 20个主题, 5000词汇量

5. **查看结果**: 查看训练结果和评估指标
   - 现有结果: {len(results)} 个

## 回复指南

- 根据用户当前所在页面提供**上下文相关**的帮助
- 如果用户问的问题与当前页面无关，先引导他们到正确的页面
- 使用友好、专业的中文回答
- 当需要执行操作时，返回 JSON 格式: {{"action": "动作名", "data": {{...}}}}

## 可用操作

- `start_task`: 开始训练任务，data: {{"dataset": "名称", "mode": "模式", "num_topics": 数量}}
- `show_datasets`: 切换到数据管理页面
- `show_results`: 切换到结果页面
"""
        return context_info.strip()
    
    def _build_messages(self, user_message: str, system_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Build message list for Qwen API"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            # Only keep last 10 messages to avoid token limit
            history = conversation_history[-10:]
            messages.extend(history)
        elif "messages" in self.conversation_context:
            # Fallback to stored context
            history = self.conversation_context["messages"][-10:]
            messages.extend(history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _parse_ai_response(self, ai_message: str, user_message: str) -> Dict[str, Any]:
        """Parse AI response to extract action and data"""
        result = {"action": None, "data": None}
        
        # Try to extract JSON from AI response
        try:
            # Look for JSON in code blocks or plain text
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', ai_message, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                result["action"] = parsed.get("action")
                result["data"] = parsed.get("data")
        except Exception:
            pass
        
        # Fall back to intent parsing if no JSON found
        if not result["action"]:
            intent, params = self.parse_intent(user_message)
            if intent == "train":
                dataset = params.get("dataset")
                if dataset:
                    result["action"] = "start_task"
                    result["data"] = {
                        "dataset": dataset,
                        "mode": "zero_shot",
                        "num_topics": 20
                    }
        
        return result
    
    def _handle_train_intent(self, params: Dict, context: Optional[Dict]) -> ChatResponse:
        """Handle training request"""
        dataset = params.get("dataset")
        
        if not dataset:
            available = settings.get_available_datasets()
            if available:
                return ChatResponse(
                    message=f"Which dataset would you like to train on? Available datasets: {', '.join(available)}",
                    action="request_dataset"
                )
            else:
                return ChatResponse(
                    message="No datasets found. Please upload a dataset first.",
                    action=None
                )
        
        dataset_dir = settings.DATA_DIR / dataset
        if not dataset_dir.exists():
            available = settings.get_available_datasets()
            return ChatResponse(
                message=f"Dataset '{dataset}' not found. Available datasets: {', '.join(available)}",
                action=None
            )
        
        return ChatResponse(
            message=f"Starting ETM training on dataset '{dataset}' with default settings (20 topics, zero_shot mode). You can monitor the progress in real-time.",
            action="start_task",
            data={
                "dataset": dataset,
                "mode": "zero_shot",
                "num_topics": 20
            }
        )
    
    def _handle_status_intent(self) -> ChatResponse:
        """Handle status query"""
        from ..agents.etm_agent import etm_agent
        
        tasks = etm_agent.get_all_tasks()
        running_tasks = [t for t in tasks.values() if t.get("status") == "running"]
        
        if not running_tasks:
            return ChatResponse(
                message="No tasks are currently running.",
                action="show_tasks"
            )
        
        status_lines = []
        for task in running_tasks:
            status_lines.append(
                f"- Task {task['task_id']}: {task.get('current_step', 'unknown')} "
                f"({task.get('dataset', 'unknown')}/{task.get('mode', 'unknown')})"
            )
        
        return ChatResponse(
            message=f"Currently running tasks:\n" + "\n".join(status_lines),
            action="show_tasks",
            data={"tasks": running_tasks}
        )
    
    def _handle_results_intent(self, params: Dict) -> ChatResponse:
        """Handle results query"""
        dataset = params.get("dataset")
        results = settings.get_available_results()
        
        if dataset:
            results = [r for r in results if r["dataset"] == dataset]
        
        if not results:
            return ChatResponse(
                message="No results found." + (f" for dataset '{dataset}'" if dataset else ""),
                action=None
            )
        
        result_lines = []
        for r in results[:5]:
            result_lines.append(f"- {r['dataset']}/{r['mode']}")
        
        return ChatResponse(
            message=f"Available results:\n" + "\n".join(result_lines),
            action="show_results",
            data={"results": results}
        )
    
    def _handle_topics_intent(self, params: Dict) -> ChatResponse:
        """Handle topic words query"""
        dataset = params.get("dataset")
        
        if not dataset:
            results = settings.get_available_results()
            if results:
                latest = results[0]
                dataset = latest["dataset"]
                mode = latest["mode"]
            else:
                return ChatResponse(
                    message="No results available. Please train a model first.",
                    action=None
                )
        else:
            mode = "zero_shot"
        
        return ChatResponse(
            message=f"Showing topic words for {dataset}/{mode}",
            action="show_topics",
            data={"dataset": dataset, "mode": mode}
        )
    
    def _handle_datasets_intent(self) -> ChatResponse:
        """Handle datasets listing"""
        datasets = settings.get_available_datasets()
        
        if not datasets:
            return ChatResponse(
                message="No datasets found in the data directory.",
                action=None
            )
        
        return ChatResponse(
            message=f"Available datasets: {', '.join(datasets)}",
            action="show_datasets",
            data={"datasets": datasets}
        )
    
    def _handle_help_intent(self) -> ChatResponse:
        """Handle help request"""
        help_text = """
I can help you with the following:

**Training**
- "Train on socialTwitter" - Start training on a dataset
- "Run pipeline on hatespeech with 30 topics" - Custom training

**Status**
- "What's the status?" - Check running tasks
- "Show progress" - View current task progress

**Results**
- "Show results" - List all training results
- "Show topics for socialTwitter" - View topic words

**Data**
- "List datasets" - Show available datasets

Just type your request and I'll help you!
        """
        return ChatResponse(
            message=help_text.strip(),
            action=None
        )
    
    def _handle_unknown_intent(self, message: str) -> ChatResponse:
        """Handle unknown intent"""
        return ChatResponse(
            message=f"I'm not sure what you mean by '{message}'. Type 'help' to see what I can do.",
            action=None
        )
    
    def get_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent suggestions based on context"""
        suggestions = []
        
        # Get context information
        current_view = context.get("current_view", "")
        datasets_count = context.get("datasets_count", 0)
        processing_jobs = context.get("processing_jobs_count", 0)
        results_count = context.get("results_count", 0)
        selected_dataset = context.get("selected_dataset")
        
        # View-specific suggestions
        if current_view == "data":
            if datasets_count == 0:
                suggestions.append({
                    "text": "上传数据集",
                    "action": "upload_dataset",
                    "description": "开始使用 THETA，先上传您的第一个数据集"
                })
            elif not selected_dataset:
                suggestions.append({
                    "text": "选择一个数据集",
                    "action": "select_dataset",
                    "description": "选择一个数据集开始处理流程"
                })
            else:
                suggestions.append({
                    "text": "清洗数据",
                    "action": "navigate",
                    "data": {"view": "processing"},
                    "description": "继续到数据清洗步骤"
                })
        
        elif current_view == "processing":
            if processing_jobs == 0:
                suggestions.append({
                    "text": "开始数据清洗",
                    "action": "start_cleaning",
                    "description": "清洗您的文本数据，去除噪音和无关内容"
                })
            else:
                suggestions.append({
                    "text": "查看向量化",
                    "action": "navigate",
                    "data": {"view": "embedding"},
                    "description": "数据清洗完成后，进行向量化处理"
                })
        
        elif current_view == "embedding":
            datasets = settings.get_available_datasets()
            if datasets:
                suggestions.append({
                    "text": "开始训练模型",
                    "action": "navigate",
                    "data": {"view": "training"},
                    "description": "向量化完成后，开始训练主题模型"
                })
        
        elif current_view == "training":
            if results_count == 0:
                suggestions.append({
                    "text": "创建训练任务",
                    "action": "start_training",
                    "description": "配置并启动您的第一个训练任务"
                })
            else:
                suggestions.append({
                    "text": "查看训练结果",
                    "action": "navigate",
                    "data": {"view": "results"},
                    "description": "查看已完成的训练结果和分析"
                })
        
        elif current_view == "results":
            if results_count > 0:
                suggestions.append({
                    "text": "查看可视化",
                    "action": "navigate",
                    "data": {"view": "visualizations"},
                    "description": "查看训练结果的可视化图表"
                })
        
        # General suggestions based on state
        if datasets_count > 0 and results_count == 0:
            suggestions.append({
                "text": "开始您的第一个训练",
                "action": "start_training",
                "description": "您有数据集，可以开始训练主题模型了"
            })
        
        # If no view-specific suggestions, provide general help
        if not suggestions:
            suggestions.append({
                "text": "查看帮助",
                "action": "help",
                "description": "了解如何使用 THETA 系统"
            })
        
        return suggestions[:5]  # Limit to 5 suggestions


chat_service = ChatService()
