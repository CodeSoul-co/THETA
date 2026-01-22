# ETM 后端集成指南

> 本文档说明如何将 ETM 模块正确接入 THETA 后端服务

## 📋 目录结构

```
THETA/
├── ETM/                          # ETM 核心模块
│   ├── engine_a/                 # BOW 和词汇表生成
│   │   ├── vocab_builder.py
│   │   └── bow_generator.py
│   ├── engine_c/                 # ETM 模型（编码器、解码器）
│   │   ├── etm.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── vocab_embedder.py
│   ├── preprocessing/            # 预处理模块（向量化）
│   │   └── embedding_processor.py
│   ├── trainer/                  # 训练器
│   │   └── trainer.py
│   ├── evaluation/               # 评估指标
│   │   └── metrics.py
│   └── visualization/            # 可视化
│       └── topic_visualizer.py
│
└── langgraph_agent/backend/      # 后端服务
    └── app/
        ├── main.py               # 入口：添加 ETM 路径到 sys.path
        ├── agents/
        │   ├── etm_agent.py      # LangGraph 代理
        │   └── nodes.py          # 节点实现（调用 ETM 模块）
        └── api/
            └── routes.py         # API 路由（预处理端点）
```

## 🔧 关键集成点

### 1. 路径配置 (`app/core/config.py`)

后端通过 `Settings.ETM_DIR` 自动检测 ETM 目录：

```python
@property
def ETM_DIR(self) -> Path:
    return self.BASE_DIR / "ETM"
```

**检查项**：
- ✅ 确保 `ETM_DIR` 指向正确的 ETM 目录
- ✅ 在服务器上：`/root/autodl-tmp/ETM`
- ✅ 在本地：`{项目根目录}/ETM`

### 2. 路径注入 (`app/main.py`)

在应用启动时，将 ETM 目录添加到 Python 路径：

```python
from .core.config import settings
sys.path.insert(0, str(settings.ETM_DIR))
```

**检查项**：
- ✅ 确保在导入任何 ETM 模块之前执行 `sys.path.insert`
- ✅ 路径必须指向 ETM 目录本身（不是父目录）

### 3. 节点中的导入 (`app/agents/nodes.py`)

在节点中导入 ETM 模块时，使用兼容性导入：

```python
# 动态添加 ETM 路径
from ..core.config import settings
ETM_PATH = settings.ETM_DIR
sys.path.insert(0, str(ETM_PATH))

# 兼容性导入
try:
    from engine_a.vocab_builder import VocabBuilder
    from engine_a.bow_generator import BOWGenerator
except ImportError:
    # 备用导入路径
    pass

try:
    from engine_c.etm import ETM
except ImportError:
    # 备用导入路径
    pass
```

**检查项**：
- ✅ 确保导入路径正确（`engine_a`, `engine_c` 等）
- ✅ 处理导入失败的情况

### 4. 预处理 API (`app/api/routes.py`)

预处理端点使用 `EmbeddingProcessor`：

```python
# 注意：这里需要插入 ETM 的父目录
sys.path.insert(0, str(settings.ETM_DIR.parent))
from ETM.preprocessing import EmbeddingProcessor, ProcessingConfig
```

**检查项**：
- ✅ 注意：预处理模块需要从 `ETM.preprocessing` 导入（不是直接 `preprocessing`）
- ✅ 确保 `ETM_DIR.parent` 指向包含 `ETM` 目录的父目录

## 🐛 常见问题排查

### 问题 1: `ModuleNotFoundError: No module named 'engine_a'`

**原因**：ETM 路径未正确添加到 `sys.path`

**解决方案**：
1. 检查 `settings.ETM_DIR` 是否正确
2. 确保在导入前执行 `sys.path.insert(0, str(settings.ETM_DIR))`
3. 验证 ETM 目录结构：
   ```bash
   ls /root/autodl-tmp/ETM/engine_a/
   # 应该看到 vocab_builder.py 和 bow_generator.py
   ```

### 问题 2: `ModuleNotFoundError: No module named 'ETM.preprocessing'`

**原因**：预处理模块导入路径错误

**解决方案**：
```python
# 错误：sys.path.insert(0, str(settings.ETM_DIR))
# 正确：需要插入父目录
sys.path.insert(0, str(settings.ETM_DIR.parent))
from ETM.preprocessing import EmbeddingProcessor
```

### 问题 3: `ImportError: cannot import name 'PipelineConfig'`

**原因**：`config.py` 不在 ETM 目录中，或路径配置错误

**解决方案**：
1. 检查 ETM 目录是否有 `config.py`
2. 如果没有，检查 `nodes.py` 中的导入逻辑：
   ```python
   from config import PipelineConfig  # 需要 ETM/config.py
   ```

### 问题 4: 向量化时找不到文本列

**已修复**：后端现在会自动检测文本列（包括 `'Consumer complaint narrative'`）

**验证**：
- ✅ 后端已更新自动检测逻辑
- ✅ 前端不再硬编码 `text_column: 'text'`

## ✅ 集成检查清单

### 后端配置

- [ ] **路径配置**
  - [ ] `ETM_DIR` 指向正确的 ETM 目录
  - [ ] `DATA_DIR` 存在且可写
  - [ ] `RESULT_DIR` 存在且可写
  - [ ] `QWEN_MODEL_PATH` 指向正确的模型目录

- [ ] **Python 路径**
  - [ ] `app/main.py` 中添加了 `sys.path.insert(0, str(settings.ETM_DIR))`
  - [ ] `app/agents/nodes.py` 中添加了路径注入
  - [ ] `app/api/routes.py` 中预处理端点使用正确的路径

- [ ] **模块导入**
  - [ ] `engine_a` 模块可以导入
  - [ ] `engine_c` 模块可以导入
  - [ ] `preprocessing` 模块可以导入
  - [ ] `trainer` 模块可以导入（如果需要）

### 功能验证

- [ ] **预处理（向量化）**
  - [ ] 可以创建预处理任务
  - [ ] 可以自动检测文本列
  - [ ] BOW 矩阵生成成功
  - [ ] 词嵌入生成成功

- [ ] **训练**
  - [ ] ETM 模型可以初始化
  - [ ] 训练循环可以运行
  - [ ] 模型参数可以保存

- [ ] **评估**
  - [ ] 评估指标可以计算
  - [ ] 结果可以返回给前端

## 🔍 调试命令

### 检查 ETM 目录结构

```bash
# 在服务器上
cd /root/autodl-tmp
ls -la ETM/
ls -la ETM/engine_a/
ls -la ETM/engine_c/
ls -la ETM/preprocessing/
```

### 检查 Python 路径

```python
# 在 Python 中测试
import sys
sys.path.insert(0, '/root/autodl-tmp/ETM')
from engine_a.vocab_builder import VocabBuilder
print("✅ engine_a 导入成功")

from engine_c.etm import ETM
print("✅ engine_c 导入成功")

sys.path.insert(0, '/root/autodl-tmp')
from ETM.preprocessing import EmbeddingProcessor
print("✅ preprocessing 导入成功")
```

### 检查后端日志

```bash
# 查看后端启动日志
tail -f /root/autodl-tmp/langgraph_agent/backend/server.log

# 或查看 uvicorn 输出
# 应该看到：
# ETM Dir: /root/autodl-tmp/ETM
# Data Dir: /root/autodl-tmp/data
# Result Dir: /root/autodl-tmp/result
```

## 📝 下一步优化建议

### 1. 统一导入方式

目前有多个地方添加路径，建议统一：

```python
# 在 app/core/etm_imports.py 中统一管理
import sys
from pathlib import Path
from .config import settings

def setup_etm_paths():
    """统一设置 ETM 相关路径"""
    etm_dir = settings.ETM_DIR
    sys.path.insert(0, str(etm_dir))
    sys.path.insert(0, str(etm_dir.parent))  # 用于 ETM.preprocessing
```

### 2. 改进错误处理

在导入失败时提供更清晰的错误信息：

```python
try:
    from engine_a.vocab_builder import VocabBuilder
except ImportError as e:
    logger.error(f"Failed to import VocabBuilder: {e}")
    logger.error(f"ETM_DIR: {settings.ETM_DIR}")
    logger.error(f"sys.path: {sys.path[:5]}")
    raise
```

### 3. 添加健康检查端点

```python
@router.get("/api/etm/health")
async def check_etm_modules():
    """检查 ETM 模块是否可用"""
    checks = {}
    try:
        from engine_a.vocab_builder import VocabBuilder
        checks["engine_a"] = "ok"
    except ImportError as e:
        checks["engine_a"] = f"error: {e}"
    
    # ... 检查其他模块
    
    return checks
```

## 🔗 相关文档

- [后端结构文档](langgraph_agent/backend/BACKEND_STRUCTURE.md)
- [ETM Agent README](ETM/agent/README.md)
- [DataClean README](ETM/dataclean/README.md)
