# THETA 脚本架构分析报告

## 1. 入口映射表 (Shell → Python Mapping)

### 1.1 主要脚本映射

| Shell 脚本 | 核心功能 | 直接调用的 Python | YAML 参数 | .env 路径参数 |
|------------|----------|-------------------|-----------|---------------|
| `setup.sh` | 环境安装 | `pip install` | - | - |
| `clean_data.sh` | 数据清洗 | `dataclean.main` | - | `DATA_DIR` |
| `prepare_data.sh` | 数据预处理 | `prepare_data.py`, `models_config/model_config.py` | `vocab_size`, `batch_size` | `DATA_DIR`, `RESULT_DIR`, `QWEN_MODEL_*` |
| `generate_embeddings.sh` | 嵌入生成 | `main.py` (embedding) | `epochs`, `learning_rate`, `batch_size` | `EMBEDDING_MODELS_DIR`, `QWEN_MODEL_*` |
| `train_theta.sh` | THETA 训练 | `run_pipeline.py` | `num_topics`, `epochs`, `hidden_dim`, `learning_rate`, `kl_*` | `RESULT_DIR`, `DATA_DIR` |
| `train_baseline.sh` | Baseline 训练 | `run_pipeline.py`, `baseline_data.py` | `num_topics`, `epochs`, `batch_size`, `hidden_dim`, `learning_rate` | `DATA_DIR`, `RESULT_DIR`, `SBERT_MODEL_PATH` |
| `evaluate.sh` | 模型评估 | `run_pipeline.py` (--skip-train) | `num_topics`, `vocab_size` | `RESULT_DIR` |
| `visualize.sh` | 可视化生成 | `visualization.run_visualization` | `language`, `dpi` | `RESULT_DIR` |
| `compare_models.sh` | 模型对比 | 内嵌 Python 脚本 | `num_topics` | `RESULT_DIR` |
| `quick_start_chinese.sh` | 中文一键流程 | `dataclean.main`, `prepare_data.py`, `run_pipeline.py`, `visualization.run_visualization` | 硬编码默认值 | `DATA_DIR`, `RESULT_DIR` |
| `quick_start_english.sh` | 英文一键流程 | 同上 | 硬编码默认值 | `DATA_DIR`, `RESULT_DIR` |
| `download_from_hf.sh` | HF 下载 | 内嵌 Python 脚本 | - | `HF_CACHE_DIR` |

### 1.2 辅助脚本

| Shell 脚本 | 功能 | 说明 |
|------------|------|------|
| `env_setup.sh` | 环境变量加载 | 被所有脚本 source，从 `.env` 加载路径 |
| `_parse_args.sh` | 参数解析辅助 | 通用参数解析逻辑 |
| `dlc_entrypoint_sim.sh` | DLC 模拟入口 | 阿里云 DLC 环境模拟 |
| `train_multi_gpu.sh` | 多 GPU 训练 | 分布式训练支持 |

---

## 2. 数据流与产物路径 (Data & Artifact Flow)

### 2.1 quick_start_chinese.sh 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        quick_start_chinese.sh                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 1] 输入检测                                                        │
│   输入: $DATA_DIR/$DATASET/ (docx/pdf/txt 或 CSV)                        │
│   检查: 最少 5 个文档                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 1.5] 文档转换 (如需要)                                             │
│   调用: python -m dataclean.main convert                                 │
│   输出: $DATA_DIR/$DATASET/${DATASET}_cleaned.csv                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 2] 数据预处理                                                      │
│   调用: python prepare_data.py --model theta --mode zero_shot            │
│   输出:                                                                  │
│     - $RESULT_DIR/0.6B/$DATASET/data/exp_*/bow/bow_matrix.npy           │
│     - $RESULT_DIR/0.6B/$DATASET/data/exp_*/bow/vocab.json               │
│     - $RESULT_DIR/0.6B/$DATASET/data/exp_*/embeddings/embeddings.npy    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 3] 模型训练                                                        │
│   调用: python run_pipeline.py --models theta --language zh              │
│   输出:                                                                  │
│     - $RESULT_DIR/0.6B/$DATASET/models/exp_*/model/theta.npy            │
│     - $RESULT_DIR/0.6B/$DATASET/models/exp_*/model/beta.npy             │
│     - $RESULT_DIR/0.6B/$DATASET/models/exp_*/topic_words/*.json         │
│     - $RESULT_DIR/0.6B/$DATASET/models/exp_*/evaluation/metrics_*.json  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 4] 可视化生成                                                      │
│   调用: python -m visualization.run_visualization --language zh          │
│   输出:                                                                  │
│     - $RESULT_DIR/0.6B/$DATASET/zero_shot/visualization/viz_*/          │
│       ├── global/                                                        │
│       │   ├── 主题网络图.png                                             │
│       │   ├── 文档聚类图.png                                             │
│       │   └── ...                                                        │
│       └── topics/                                                        │
│           ├── topic_01/                                                  │
│           └── ...                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 train_baseline.sh 数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         train_baseline.sh                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 0] 配置加载                                                        │
│   优先级: CLI args > config/default.yaml > .env                          │
│   调用: read_yaml() 函数读取 YAML 配置                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 1] 数据预处理 (如需要)                                             │
│   检查: $WORKSPACE_DIR/bow_matrix.npy 是否存在                           │
│   如不存在:                                                              │
│     1. 检测原始文档 → 调用 doc_converter.py 转换                          │
│     2. 调用 baseline_data.prepare_baseline_data()                        │
│        - 生成 BOW 矩阵                                                   │
│        - 如模型需要 SBERT (ctm/bertopic)，生成 SBERT 嵌入                 │
│   输出: $RESULT_DIR/baseline/$DATASET/data/$DATASET/                     │
│     ├── bow_matrix.npy                                                   │
│     ├── vocab.json                                                       │
│     ├── sbert_embeddings.npy (CTM/BERTopic)                              │
│     └── config.json                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ [Step 2] 模型训练                                                        │
│   调用: python run_pipeline.py --models $MODELS --workspace_dir $WS      │
│   输出: $RESULT_DIR/$DATASET/default_user/$MODEL/exp_*/                  │
│     ├── $MODEL/theta_k20.npy, beta_k20.npy                               │
│     ├── metrics_k20.json                                                 │
│     └── en/ 或 zh/                                                       │
│         └── global/                                                      │
│             ├── topic_network.png / 主题网络图.png                        │
│             └── ...                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 环境变量读取方式

**输入端 (.env 读取)**:
```bash
# env_setup.sh 中的读取逻辑
if [ -f "$PROJECT_ROOT/.env" ]; then
    while IFS='=' read -r key value; do
        # 只在环境变量未设置时才从 .env 读取
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$PROJECT_ROOT/.env"
fi

# 关键路径变量
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/result}"
export SBERT_MODEL_PATH="${SBERT_MODEL_PATH:-$MODELS_DIR/model/baselines/sbert/...}"
```

**中间态 (数据传递)**:
- **通过文件系统**: BOW 矩阵、嵌入向量、词表等以 `.npy`/`.json` 格式存储在 workspace 目录
- **通过 CLI 参数**: `--workspace_dir` 参数传递 workspace 路径
- **不通过内存**: 各步骤独立运行，不共享内存状态

**输出端**:
```
evaluation/
├── metrics_k20.json          # 评估指标 JSON
└── metrics_20260328_*.json   # 带时间戳的指标

visualization/
├── viz_20260328_*/
│   ├── global/               # 全局图表
│   │   ├── 主题网络图.png    # zh 语言
│   │   └── topic_network.png # en 语言
│   └── topics/               # 每主题图表
│       └── topic_01/
```

---

## 3. 孤儿脚本审计 (Orphan Script Audit)

### 3.1 被 Shell 脚本直接调用的 Python 文件

| Python 文件 | 调用方 |
|-------------|--------|
| `src/models/main.py` | `generate_embeddings.sh` |
| `src/models/prepare_data.py` | `prepare_data.sh`, `quick_start_*.sh` |
| `src/models/run_pipeline.py` | `train_theta.sh`, `train_baseline.sh`, `quick_start_*.sh` |
| `src/models/experiment_manager.py` | `visualize.sh` |
| `src/models/dataclean/main.py` | `clean_data.sh`, `quick_start_*.sh` |
| `src/models/visualization/run_visualization.py` | `visualize.sh`, `quick_start_*.sh` |
| `src/models/models_config/model_config.py` | `prepare_data.sh` |
| `src/models/model/baseline_data.py` | `train_baseline.sh` (内嵌调用) |

### 3.2 被主程序 import 的模块

通过 `run_pipeline.py` 和 `main.py` 间接导入的模块：
- `config.py`, `config_loader.py`
- `model/trainer.py`, `model/baseline_trainer.py`
- `model/theta/*.py` (encoder, decoder, etm)
- `model/baseline/*.py` (lda, ctm, etm, prodlda, etc.)
- `evaluation/*.py` (metrics, unified_evaluator)
- `visualization/*.py`
- `bow/*.py`
- `data/*.py`

### 3.3 孤儿脚本列表

| Python 文件 | 状态 | 理由 |
|-------------|------|------|
| `src/embedding/balanced_sampler.py` | **[Legacy]** | 仅被 `src/embedding/trainer.py` 导入，但 embedding 模块已被 THETA 主流程取代 |
| `src/embedding/registry.py` | **[Legacy]** | 嵌入模型注册表，当前未使用 |
| `src/models/data_pipeline/async_trainer.py` | **[Legacy]** | 异步训练器，当前流程不使用 |
| `src/models/data_pipeline/column_mapper.py` | **[Redundant]** | 列映射工具，功能已被 `baseline_data.py` 覆盖 |
| `src/models/data_pipeline/csv_scanner.py` | **[Redundant]** | CSV 扫描器，功能已被 `dataclean` 模块覆盖 |
| `src/models/data_pipeline/matrix_pipeline.py` | **[Legacy]** | 矩阵流水线，当前未使用 |
| `src/models/data_pipeline/pipeline_api.py` | **[Legacy]** | API 接口，当前未使用 |
| `src/models/preprocessing/embedding_processor.py` | **[Legacy]** | 嵌入处理器，功能已整合到 `prepare_data.py` |
| `src/models/utils/migrate_result_structure.py` | **[Utility]** | 一次性迁移脚本，保留用于历史数据迁移 |
| `src/models/evaluation/advanced_metrics.py` | **[Redundant]** | 高级指标，功能已整合到 `unified_evaluator.py` |

---

## 4. 专家级 CLI 调用规范

### 4.1 直接调用 Python 覆盖 YAML 参数

```bash
# 基本用法 - 使用 YAML 默认值
python src/models/run_pipeline.py --dataset my_data --models lda

# 覆盖 num_topics (YAML 默认 20)
python src/models/run_pipeline.py --dataset my_data --models lda --num_topics 50

# 覆盖多个参数
python src/models/run_pipeline.py --dataset my_data --models ctm \
    --num_topics 30 \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --hidden_dim 512

# THETA 模型完整参数覆盖
python src/models/run_pipeline.py --dataset my_data --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 25 \
    --epochs 150 \
    --batch_size 64 \
    --hidden_dim 512 \
    --learning_rate 0.002 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 50 \
    --patience 15 \
    --language zh

# 跳过训练，仅评估和可视化
python src/models/run_pipeline.py --dataset my_data --models lda --skip-train

# 指定 workspace 目录
python src/models/run_pipeline.py --dataset my_data --models ctm \
    --workspace_dir /path/to/workspace
```

### 4.2 参数优先级逻辑

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        参数查找优先级                                    │
│                                                                         │
│   1. CLI 参数 (最高优先级)                                               │
│      └── argparse 解析的命令行参数                                       │
│                                                                         │
│   2. YAML 配置 (config/default.yaml)                                    │
│      └── ConfigLoader.get_model_defaults(model_name)                    │
│          └── 先查模型特定配置，再查 global 配置                          │
│                                                                         │
│   3. 代码硬编码默认值 (最低优先级)                                        │
│      └── argparse default= 或函数参数默认值                              │
└─────────────────────────────────────────────────────────────────────────┘
```

**ConfigLoader 实现逻辑** (`src/models/config_loader.py`):

```python
# 读取 YAML 配置
def get_model_defaults(model_name: str) -> dict:
    """获取模型默认参数，优先级: model_config > global_config"""
    model_config = yaml_data.get(model_name, {})
    global_config = yaml_data.get('global', {})
    
    # 合并配置，模型特定配置覆盖全局配置
    merged = {**global_config, **model_config}
    return merged

# 获取语言设置
def get_language(cli_lang: str = None) -> str:
    """获取语言，优先级: CLI > YAML > 默认值 'zh'"""
    if cli_lang:
        return cli_lang
    return YAMLConfig.get('visualization', {}).get('language', 'zh')
```

### 4.3 main.py 直接调用示例

```bash
# 仅生成嵌入 (不训练主题模型)
python src/models/main.py --mode zero_shot --dataset my_data --model_size 0.6B

# 使用 LoRA 微调
python src/models/main.py --mode unsupervised --dataset my_data --epochs 10 --use_lora

# 有监督微调
python src/models/main.py --mode supervised --dataset my_data --label_column category
```

---

## 5. SBERT 路径特例检查

### 5.1 当前 SBERT 加载逻辑

**位置**: `src/models/model/baseline_trainer.py` → `_generate_sbert_embeddings()`

```python
def _generate_sbert_embeddings(self):
    """
    Generate SBERT embeddings automatically when missing.
    Uses SBERT_MODEL_PATH from .env or falls back to default model.
    """
    # 1. 首先从环境变量读取
    sbert_model_path = os.environ.get('SBERT_MODEL_PATH')
    
    # 2. 如果环境变量未设置或路径不存在，尝试默认位置
    if not sbert_model_path or not os.path.exists(sbert_model_path):
        project_root = Path(__file__).parent.parent.parent.parent
        default_paths = [
            project_root / 'models' / 'sbert' / 'sentence-transformers' / 'all-MiniLM-L6-v2',
            project_root / 'embedding_models' / 'sbert' / 'all-MiniLM-L6-v2',
        ]
        for p in default_paths:
            if p.exists():
                sbert_model_path = str(p)
                break
        else:
            # 3. 最后回退到从 HuggingFace 下载
            sbert_model_path = 'all-MiniLM-L6-v2'
    
    model = SentenceTransformer(sbert_model_path)
    # ... 生成嵌入
```

### 5.2 CTM 模型 SBERT 使用确认

**结论**: ✅ **CTM 已完全使用 .env 中的 SBERT_MODEL_PATH**

1. **旧逻辑已移除**: CTM 不再从预计算的 `sbert_embeddings.npy` 文件读取
2. **自动生成**: 如果 workspace 中没有 SBERT 嵌入，`train_ctm()` 会自动调用 `_generate_sbert_embeddings()`
3. **路径优先级**: `SBERT_MODEL_PATH` (env) > 默认路径 > HuggingFace 下载

**调用链**:
```
train_baseline.sh
  └── run_pipeline.py
      └── BaselineTrainer.train_ctm()
          └── if self.sbert_embeddings is None:
              └── self._generate_sbert_embeddings()
                  └── os.environ.get('SBERT_MODEL_PATH')
```

### 5.3 .env 配置示例

```bash
# .env 文件
SBERT_MODEL_PATH=./models/sbert/sentence-transformers/all-MiniLM-L6-v2
```

---

## 附录: 目录结构参考

```
theta/
├── scripts/                    # Shell 脚本入口
│   ├── env_setup.sh           # 环境变量加载 (被所有脚本 source)
│   ├── clean_data.sh          # 数据清洗
│   ├── prepare_data.sh        # 数据预处理
│   ├── train_theta.sh         # THETA 训练
│   ├── train_baseline.sh      # Baseline 训练
│   ├── visualize.sh           # 可视化
│   ├── evaluate.sh            # 评估
│   └── quick_start_*.sh       # 一键流程
│
├── config/
│   └── default.yaml           # 默认参数配置
│
├── src/
│   ├── models/
│   │   ├── main.py            # 嵌入生成入口
│   │   ├── prepare_data.py    # 数据预处理入口
│   │   ├── run_pipeline.py    # 统一训练入口
│   │   ├── config.py          # 配置定义
│   │   ├── config_loader.py   # 三层配置加载器
│   │   ├── model/             # 模型实现
│   │   │   ├── baseline_trainer.py
│   │   │   ├── baseline_data.py
│   │   │   ├── theta/         # THETA 模型
│   │   │   └── baseline/      # Baseline 模型
│   │   ├── evaluation/        # 评估模块
│   │   ├── visualization/     # 可视化模块
│   │   └── dataclean/         # 数据清洗模块
│   │
│   └── embedding/             # 嵌入模块 [Legacy]
│
├── data/                      # 数据目录
│   └── {dataset}/
│       └── {dataset}_cleaned.csv
│
└── result/                    # 输出目录
    ├── 0.6B/{dataset}/        # THETA 结果
    │   ├── data/exp_*/        # 预处理数据
    │   ├── models/exp_*/      # 模型输出
    │   └── zero_shot/visualization/  # 可视化
    │
    └── baseline/{dataset}/    # Baseline 结果
        └── data/{dataset}/    # 预处理数据
```
