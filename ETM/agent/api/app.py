"""
API接口 (API Interface)

提供RESTful API接口，允许通过HTTP请求与Agent交互。
"""

import os
import sys
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parents[2]))

# 导入Agent组件
from agent.core.topic_aware_agent import TopicAwareAgent
from agent.utils.config import AgentConfig


# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_tools: bool = True
    use_topic_enhancement: bool = True


class ChatResponse(BaseModel):
    session_id: str
    message: str
    dominant_topics: List[List[float]]
    topic_shifted: bool


class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    document_id: int
    topic_dist: List[float]


class TopicRequest(BaseModel):
    topic_id: int
    top_k: int = 10


class TopicWordsResponse(BaseModel):
    topic_id: int
    words: List[Dict[str, Any]]


# 创建FastAPI应用
app = FastAPI(
    title="ETM Agent API",
    description="主题感知Agent API接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局Agent实例
agent = None


# 依赖项：获取Agent实例
def get_agent():
    global agent
    if agent is None:
        # 从环境变量加载配置
        config = AgentConfig.from_env()
        
        # 如果环境变量中没有设置，使用默认值
        if not config.etm_model_path:
            config.etm_model_path = os.environ.get(
                "ETM_MODEL_PATH",
                "/root/autodl-tmp/ETM/outputs/models/etm_model.pt"
            )
        
        if not config.vocab_path:
            config.vocab_path = os.environ.get(
                "VOCAB_PATH",
                "/root/autodl-tmp/ETM/outputs/engine_a/vocab.json"
            )
        
        # 初始化Agent
        agent = TopicAwareAgent(config, dev_mode=True)
    
    return agent


# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


# 聊天端点
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent=Depends(get_agent)):
    try:
        # 生成会话ID（如果未提供）
        session_id = request.session_id or str(uuid.uuid4())
        
        # 处理用户输入
        result = agent.process(
            user_input=request.message,
            session_id=session_id,
            use_tools=request.use_tools,
            use_topic_enhancement=request.use_topic_enhancement
        )
        
        # 构建响应
        response = {
            "session_id": session_id,
            "message": result["content"],
            "dominant_topics": [[topic_id, float(weight)] for topic_id, weight in result["dominant_topics"]],
            "topic_shifted": result["topic_shifted"]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 添加文档端点
@app.post("/documents", response_model=DocumentResponse)
def add_document(request: DocumentRequest, agent=Depends(get_agent)):
    try:
        # 创建文档
        document = {
            "text": request.text,
            "metadata": request.metadata or {}
        }
        
        # 获取文本的主题分布
        topic_dist = agent.get_topic_distribution(request.text)
        
        # 添加到知识库
        document_id = agent.add_document(document, topic_dist=topic_dist)
        
        # 构建响应
        response = {
            "document_id": document_id,
            "topic_dist": topic_dist.tolist()
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 查询知识库端点
@app.post("/query")
def query_knowledge(query: str = Body(..., embed=True), top_k: int = 5, agent=Depends(get_agent)):
    try:
        # 查询知识库
        results = agent.query_knowledge(query, top_k=top_k)
        
        # 构建响应
        response = [
            {
                "document_id": doc_id,
                "score": float(score),
                "content": document
            }
            for doc_id, score, document in results
        ]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取主题词端点
@app.get("/topics/{topic_id}/words", response_model=TopicWordsResponse)
def get_topic_words(topic_id: int, top_k: int = 10, agent=Depends(get_agent)):
    try:
        # 获取主题词
        topic_words = agent.get_topic_words(topic_id, top_k=top_k)
        
        # 构建响应
        response = {
            "topic_id": topic_id,
            "words": [
                {
                    "word": word,
                    "weight": float(weight)
                }
                for word, weight in topic_words
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取文本主题分布端点
@app.post("/analyze")
def analyze_text(text: str = Body(..., embed=True), agent=Depends(get_agent)):
    try:
        # 获取主题分布
        topic_dist = agent.get_topic_distribution(text)
        
        # 获取主导主题
        dominant_topics = agent.get_dominant_topics(topic_dist)
        
        # 构建响应
        response = {
            "topic_dist": topic_dist.tolist(),
            "dominant_topics": [
                {
                    "topic_id": topic_id,
                    "weight": float(weight),
                    "words": [
                        {
                            "word": word,
                            "weight": float(w)
                        }
                        for word, w in agent.get_topic_words(topic_id, top_k=5)
                    ]
                }
                for topic_id, weight in dominant_topics
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 主函数
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取端口
    port = int(os.environ.get("PORT", 8000))
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=port)
