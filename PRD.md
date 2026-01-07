# 社会学文本主题建模平台 - 产品需求文档 (PRD)

## 1. 项目概述

### 1.1 项目名称
**Social Topic Modeling Platform (STMP)** - 基于LLM增强的社会学文本主题建模与分析平台

### 1.2 项目背景
本项目旨在构建一个端到端的社会学文本分析平台，利用大语言模型(LLM)的语义理解能力增强传统主题模型(Topic Model)的效果。核心创新在于使用 Qwen3-Embedding 模型替代传统的 TF-IDF 向量化方法，并结合 ETM (Embedded Topic Model) 实现高质量的主题发现。

### 1.3 核心价值
- **语义增强**: 利用 LLM 的预训练知识捕捉深层语义，而非仅依赖词频统计
- **领域适应**: 通过 LoRA 微调使模型适应特定社会学领域（如心理健康、仇恨言论、政治辩论等）
- **可解释性**: ETM 输出的主题具有清晰的语义解释，便于社会学研究者理解
- **多语言支持**: 支持英语、德语等多语言文本分析

---

## 2. 数据资产

### 2.1 数据集概览

| 数据集 | 规模 | 语言 | 标签 | 领域 |
|--------|------|------|------|------|
| **germanCoal** | 9,136条 | 德语 | 无 | 德国议会煤炭政策辩论 |
| **FCPB** | 208,955条 | 英语 | 无 | 金融消费者投诉 |
| **hatespeech** | 436,725条 | 英语 | 二分类 | 仇恨言论检测 |
| **mental_health** | 1,023,524条 | 英语 | 28分类 | Reddit心理健康帖子 |
| **socialTwitter** | 39,659条 | 多语言 | 二分类 | Twitter账户分类 |

**总计**: ~170万条文本，覆盖政治、金融、社交媒体、心理健康等多个社会学研究领域。

### 2.2 数据分类

**有标签数据集** (3个，用于有监督学习):
- `hatespeech`: 仇恨言论 vs 正常言论
- `mental_health`: 28个心理健康相关子版块
- `socialTwitter`: 正常账户 vs 垃圾账户

**无标签数据集** (2个，用于无监督学习):
- `germanCoal`: 德国议会演讲
- `FCPB`: 金融投诉叙述

---

## 3. 系统架构

### 3.1 双层模型体系

```
┌─────────────────────────────────────────────────────────────────┐
│                    Social Topic Modeling Platform                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  第一层: 语义嵌入层 (Semantic Embedding Layer)            │   │
│  │  ├─ 模型: Qwen3-Embedding-0.6B                           │   │
│  │  ├─ 方法: Zero-shot / LoRA微调 (有监督/无监督)            │   │
│  │  └─ 输出: 文档嵌入矩阵 X ∈ R^(N×1024)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  第二层: 主题生成层 (Topic Generation Layer)              │   │
│  │  ├─ 模型: ETM (Embedded Topic Model, VAE架构)            │   │
│  │  ├─ 输入: 文档嵌入 + BOW矩阵 + 词向量矩阵(ρ)             │   │
│  │  └─ 输出: θ(主题分布) + β(词分布) + α(主题向量)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  下游任务层 (Downstream Tasks)                            │   │
│  │  ├─ 主题可视化 (词云、主题分布图)                         │   │
│  │  ├─ 文档聚类与检索                                        │   │
│  │  └─ 分类任务验证 (F1-Score)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块划分

| 模块 | 目录 | 功能 |
|------|------|------|
| **Engine A** | `/ETM/engine_a/` | 词表构建、BOW矩阵生成 |
| **Engine B** | `/embedding/` | Qwen嵌入训练与推理 |
| **Engine C** | `/ETM/engine_c/` | ETM模型训练与推理 |
| **Data** | `/data/` | 原始数据存储 |
| **Result** | `/result/` | 结果输出（版本化，不覆盖） |

---

## 4. 功能需求

### 4.1 Embedding训练模块 (Engine B)

#### 4.1.1 Zero-shot嵌入
- **输入**: 原始文本列表
- **处理**: 直接使用预训练Qwen模型编码
- **输出**: 文档嵌入矩阵 `(N, 1024)`
- **用途**: 性能基线、快速原型

#### 4.1.2 有监督训练 (Supervised)
- **适用**: 有标签数据集 (hatespeech, mental_health, socialTwitter)
- **方法**: LoRA微调 + 对比学习损失
- **输出**: 领域适应的LoRA适配器 + 增强嵌入

#### 4.1.3 无监督训练 (Unsupervised)
- **适用**: 无标签数据集 (germanCoal, FCPB)
- **方法**: SimCSE自监督对比学习
- **输出**: 领域适应的LoRA适配器 + 增强嵌入

#### 4.1.4 联合训练 (Joint Training)
- **方法**: 所有数据集联合训练，共享嵌入空间
- **输出**: 统一的嵌入模型

### 4.2 主题建模模块 (Engine C - ETM)

#### 4.2.1 核心算法
ETM基于VAE架构，核心公式：
```
P(w|z) ∝ exp(ρᵀ αz)
```
- `ρ ∈ R^(V×1024)`: 词向量矩阵（由Qwen生成）
- `αz ∈ R^1024`: 主题向量（模型学习）

#### 4.2.2 训练流程
1. **Encoder**: 文档向量 → MLP → (μ, log σ²)
2. **Reparameterization**: z = μ + σ·ε, θ = Softmax(z)
3. **Decoder**: β = Softmax(α·ρᵀ), x̂ = θ·β
4. **Loss**: 重构误差(NLL) + KL散度

#### 4.2.3 输出矩阵
| 矩阵 | 维度 | 含义 |
|------|------|------|
| **θ (Theta)** | N × K | 文档-主题分布 |
| **β (Beta)** | K × V | 主题-词分布 |
| **α (Alpha)** | K × 1024 | 主题嵌入向量 |

### 4.3 数据预处理模块 (Engine A)

#### 4.3.1 词表构建
- 分词、去停用词
- 词频过滤 (min_df, max_df)
- 词表大小限制 (默认10000)

#### 4.3.2 BOW矩阵生成
- 输出: 稀疏矩阵 `(N, V)`
- 作为ETM Decoder的重构目标

### 4.4 可视化与分析模块

#### 4.4.1 主题可视化
- 主题词云
- 主题分布热力图
- 主题相关性矩阵

#### 4.4.2 文档分析
- 文档主题分布
- 相似文档检索
- 主题时序演变（如适用）

---

## 5. 技术流程

### 5.1 端到端流水线

```
原始文本 → [Engine A] → BOW矩阵 + 词表
    ↓
原始文本 → [Engine B] → 文档嵌入矩阵
    ↓
词表 → [Engine B] → 词向量矩阵(ρ)
    ↓
[文档嵌入 + BOW + ρ] → [Engine C/ETM] → [θ, β, α]
    ↓
[θ, β, α] → [可视化/下游任务]
```

### 5.2 训练范式对比

| 范式 | 数据要求 | 方法 | 预期效果 |
|------|----------|------|----------|
| Zero-shot | 无 | 直接推理 | 基线 |
| Supervised | 标签 | LoRA + 对比学习 | 最优 |
| Unsupervised | 无 | SimCSE | 优于Zero-shot |
| Joint | 混合 | 联合训练 | 跨领域泛化 |

---

## 6. 非功能需求

### 6.1 性能要求
- **硬件**: RTX 4090 (24GB显存)
- **Qwen微调**: 1-3小时/数据集
- **Embedding推理**: 10-20分钟/数据集
- **ETM训练**: 5-10分钟/数据集

### 6.2 可扩展性
- 模块化设计，各引擎独立运行
- 中间矩阵完全暴露接口
- 支持新数据集快速接入

### 6.3 可复现性
- 固定随机种子
- 支持断点续训
- 结果版本化存储（不覆盖）

### 6.4 代码规范
- 英文注释
- 通过 `main.py` 控制所有超参数
- `--dev` 参数用于调试模式
- 训练时输出 epoch、loss、时间等信息

---

## 7. 前后端需求（待开发）

### 7.1 后端需求

#### 7.1.1 API设计
| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/datasets` | GET | 获取数据集列表 |
| `/api/embedding/train` | POST | 启动Embedding训练 |
| `/api/embedding/status` | GET | 查询训练状态 |
| `/api/etm/train` | POST | 启动ETM训练 |
| `/api/etm/topics` | GET | 获取主题列表 |
| `/api/documents/search` | POST | 文档检索 |
| `/api/visualization/wordcloud` | GET | 获取词云数据 |

#### 7.1.2 技术栈建议
- **框架**: FastAPI / Flask
- **任务队列**: Celery + Redis (长时间训练任务)
- **数据库**: PostgreSQL (元数据) + Redis (缓存)
- **文件存储**: 本地文件系统 / MinIO

### 7.2 前端需求

#### 7.2.1 页面设计
1. **仪表盘**: 数据集概览、训练状态监控
2. **数据管理**: 数据集上传、预览、统计
3. **训练配置**: 参数设置、模式选择、启动训练
4. **结果展示**: 主题词云、分布图、交互式探索
5. **文档分析**: 单文档主题分析、相似文档检索

#### 7.2.2 技术栈建议
- **框架**: React + TypeScript
- **UI组件**: Ant Design / shadcn/ui
- **可视化**: ECharts / D3.js
- **状态管理**: Zustand / Redux

#### 7.2.3 核心交互
- 实时训练进度展示
- 主题词云交互（点击查看详情）
- 文档-主题关系可视化
- 参数调节与对比实验

---

## 8. 评估指标

### 8.1 主题质量
- **Topic Coherence (NPMI)**: 主题内词语共现度
- **Topic Diversity (TD)**: 主题间差异度

### 8.2 模型收敛
- **Reconstruction Loss**: VAE重构误差
- **KL Divergence**: 潜在空间正则化

### 8.3 下游任务（有标签数据）
- **F1-Score**: 分类任务验证
- **Accuracy**: 整体准确率

---

## 9. 项目里程碑

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | 数据清洗与预处理 | ✅ 已完成 |
| **Phase 2** | Embedding训练框架 | ✅ 已完成 |
| **Phase 3** | ETM模型实现 | ✅ 已完成 |
| **Phase 4** | 可视化与评估 | 🔄 进行中 |
| **Phase 5** | 后端API开发 | ⏳ 待开发 |
| **Phase 6** | 前端界面开发 | ⏳ 待开发 |
| **Phase 7** | 系统集成与测试 | ⏳ 待开发 |

---

## 10. 附录

### 10.1 目录结构
```
/root/autodl-tmp/
├── data/                          # 原始数据
│   ├── germanCoal/
│   ├── FCPB/
│   ├── hatespeech/
│   ├── mental_health/
│   └── socialTwitter/
├── embedding/                     # Engine B: Embedding训练
│   ├── main.py                    # 主入口
│   ├── data_loader.py
│   ├── embedder.py
│   ├── trainer.py
│   ├── outputs/                   # 嵌入输出
│   └── checkpoints/               # 模型检查点
├── ETM/                           # Engine A + C: 主题建模
│   ├── engine_a/                  # 词表与BOW
│   ├── engine_c/                  # ETM模型
│   ├── run_etm_simple.py          # ETM主入口
│   └── outputs/                   # ETM输出
├── result/                        # 版本化结果存储
└── qwen3_embedding_0.6B/          # 预训练模型
```

### 10.2 关键命令

```bash
# 激活环境
source activate jiqun

# Zero-shot嵌入
python embedding/main.py --mode zero_shot --dataset all

# 有监督训练
python embedding/main.py --mode supervised --dataset hatespeech --epochs 3

# 无监督训练
python embedding/main.py --mode unsupervised --dataset germanCoal --epochs 3

# ETM训练
python ETM/run_etm_simple.py --dataset germanCoal --embedding_mode zero_shot --num_topics 50
```

### 10.3 核心输出文件

| 文件 | 维度 | 说明 |
|------|------|------|
| `*_embeddings.npy` | N × 1024 | 文档嵌入矩阵 |
| `*_vocab_embeddings.npy` | V × 1024 | 词向量矩阵(ρ) |
| `theta_*.npy` | N × K | 文档-主题分布 |
| `beta_*.npy` | K × V | 主题-词分布 |
| `topic_embeddings_*.npy` | K × 1024 | 主题向量(α) |
| `topic_words_*.json` | - | 主题关键词列表 |

---

*文档版本: v1.0*  
*生成日期: 2026-01-04*
