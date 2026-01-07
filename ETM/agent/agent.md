# ETM Agent 框架

基于嵌入主题模型(ETM)的智能Agent框架，将主题建模能力与大语言模型结合，实现主题感知的智能交互。

## 架构

```
┌─────────────────────────────────────────────────────┐
│                  用户交互层                         │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                  认知控制层                         │
│  ┌───────────────┐   ┌───────────────┐   ┌────────┐ │
│  │  意图理解模块 │←→│  规划推理模块  │←→│ 记忆模块│ │
│  └───────┬───────┘   └───────┬───────┘   └────────┘ │
└──────────┼─────────────────┬─┼─────────────────────┘
           ↓                 │ ↓
┌──────────┼─────────────────┼─┼─────────────────────┐
│          │   知识表示层    │ │                     │
│  ┌───────┼───────┐ ┌───────┼─┼─────┐               │
│  │ 主题空间表示 │ │ 语义向量表示 │               │
│  │  (ETM theta) │ │ (Qwen向量)   │               │
│  └───────┬───────┘ └───────┬─────┘               │
└──────────┼─────────────────┼─────────────────────┘
           ↓                 ↓             
┌──────────┼─────────────────┼─────────────────────┐
│          │     模型层      │                     │
│  ┌───────┼───────┐ ┌───────┼─────┐ ┌───────────┐ │
│  │   ETM模型    │ │  Qwen模型   │ │ 工具API  │ │
│  └───────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────┘
```

## 核心组件

### 1. 主题感知模块 (Topic-Aware Module)
- 将文本映射到主题空间
- 提供主题识别、追踪和扩展功能
- 支持主题关键词提取和相似度计算

### 2. 认知控制模块 (Cognitive Controller)
- 整合主题信息与大语言模型
- 实现意图理解、上下文管理和推理规划
- 支持工具调用和响应生成

### 3. 知识表示模块 (Knowledge Module)
- 存储和检索文档
- 支持基于主题和语义的混合查询
- 使用FAISS进行高效向量检索

### 4. 记忆系统 (Memory System)
- 管理对话历史和主题演化
- 提供短期和长期记忆存储
- 支持知识缓存和工具调用结果存储

### 5. API接口 (API Interface)
- 提供RESTful API接口
- 支持聊天、文档添加、知识查询等功能
- 可通过HTTP请求与Agent交互

## 快速开始

### 初始化Agent

```python
from agent.core.topic_aware_agent import TopicAwareAgent
from agent.utils.config import AgentConfig

# 创建配置
config = AgentConfig(
    etm_model_path="/path/to/etm_model.pt",
    vocab_path="/path/to/vocab.json",
    embedding_model_path="/root/autodl-tmp/qwen3_embedding_0.6B"
)

# 初始化Agent
agent = TopicAwareAgent(config)
```

### 处理用户输入

```python
# 处理用户输入
result = agent.process(
    user_input="你好，我想了解气候变化",
    session_id="user_123"
)

# 获取响应
response = result["content"]
dominant_topics = result["dominant_topics"]
```

### 启动API服务

```bash
cd /root/autodl-tmp/ETM
export ETM_MODEL_PATH="/path/to/etm_model.pt"
export VOCAB_PATH="/path/to/vocab.json"
python -m agent.api.app
```

## 文件结构

```
/root/autodl-tmp/ETM/agent/
├── core/
│   └── topic_aware_agent.py     # 主题感知Agent主类
├── modules/
│   ├── topic_aware.py           # 主题感知模块
│   ├── cognitive_controller.py  # 认知控制模块
│   └── knowledge_module.py      # 知识表示模块
├── memory/
│   └── memory_system.py         # 记忆系统
├── utils/
│   ├── config.py                # 配置类
│   ├── tool_registry.py         # 工具注册表
│   └── llm_client.py            # 大语言模型客户端
└── api/
    └── app.py                   # API接口
```

## 特点

- **主题感知**：利用ETM的主题建模能力，理解用户意图和兴趣
- **语义增强**：结合Qwen嵌入模型，提供精准的语义理解
- **模块化设计**：各组件独立，易于扩展和定制
- **工具集成**：支持工具调用，增强Agent能力
- **记忆管理**：跟踪对话历史和主题演化
- **API接口**：提供标准化的交互方式

## 依赖

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- FAISS (可选，用于向量检索)
- Qwen嵌入模型
- ETM模型
