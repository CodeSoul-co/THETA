#!/bin/bash
# ============================================================================
# 完整Pipeline一键运行脚本
# ============================================================================
# 功能: 按顺序执行所有步骤（Embedding生成 -> BOW生成 -> ETM训练 -> 评估 -> 可视化）
# 用法: ./run_full_pipeline.sh <dataset_name> <mode> [num_topics] [epochs]
# 示例: ./run_full_pipeline.sh hatespeech supervised 20 50
# 
# 支持的数据集: socialTwitter, hatespeech, mental_health, germanCoal, FCPB
# 支持的模式: zero_shot, supervised, unsupervised
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"
NUM_TOPICS="${3:-20}"
EPOCHS="${4:-50}"
BATCH_SIZE="${5:-64}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETM_DIR="${PROJECT_ROOT}/ETM"
EMBEDDING_DIR="${PROJECT_ROOT}/embedding"

# GPU设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=============================================="
echo "ETM主题模型完整Pipeline"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "主题数: ${NUM_TOPICS}"
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "开始时间: $(date)"
echo "=============================================="
echo ""

# ============================================
# 步骤1: Embedding生成
# ============================================
echo "[1/5] Embedding生成..."
cd "${EMBEDDING_DIR}"

case "${MODE}" in
    "zero_shot")
        python main.py \
            --mode zero_shot \
            --dataset "${DATASET_NAME}" \
            --batch_size 16
        ;;
    "supervised")
        python main.py \
            --mode supervised \
            --dataset "${DATASET_NAME}" \
            --epochs 3 \
            --batch_size 16 \
            --use_lora
        ;;
    "unsupervised")
        python main.py \
            --mode unsupervised \
            --dataset "${DATASET_NAME}" \
            --epochs 3 \
            --batch_size 16 \
            --use_lora
        ;;
esac

echo "[1/5] Embedding生成完成!"
echo ""

# ============================================
# 步骤2: BOW矩阵生成
# ============================================
echo "[2/5] BOW矩阵生成..."
cd "${ETM_DIR}"

python -c "
import sys
sys.path.insert(0, '.')
from config import PipelineConfig, DATASET_CONFIGS
from bow.bow_generator import BOWGenerator
import pandas as pd
from pathlib import Path
import json
import scipy.sparse as sp

config = PipelineConfig()
config.data.dataset = '${DATASET_NAME}'
config.bow.vocab_size = 5000
config.bow.min_doc_freq = 5
config.bow.max_doc_freq = 0.95

if '${DATASET_NAME}' in DATASET_CONFIGS:
    ds_config = DATASET_CONFIGS['${DATASET_NAME}']
    config.bow.language = ds_config.get('language', 'english')

data_dir = Path('${PROJECT_ROOT}/data/${DATASET_NAME}')
csv_files = list(data_dir.glob('*cleaned*.csv')) + list(data_dir.glob('*text_only*.csv'))
if not csv_files:
    csv_files = list(data_dir.glob('*.csv'))

df = pd.read_csv(csv_files[0])
text_col = next((c for c in df.columns if c.lower() in ['text', 'content', 'document']), df.columns[0])
texts = df[text_col].dropna().tolist()
print(f'Loaded {len(texts)} documents')

generator = BOWGenerator(config)
bow_matrix, vocab = generator.fit_transform(texts)
print(f'BOW matrix shape: {bow_matrix.shape}')

output_dir = Path('${PROJECT_ROOT}/result/${DATASET_NAME}/bow')
output_dir.mkdir(parents=True, exist_ok=True)
sp.save_npz(output_dir / 'bow_matrix.npz', bow_matrix)
with open(output_dir / 'vocab.json', 'w') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f'Saved to {output_dir}')
"

echo "[2/5] BOW生成完成!"
echo ""

# ============================================
# 步骤3: 词汇Embedding生成
# ============================================
echo "[3/5] 词汇Embedding生成..."
cd "${EMBEDDING_DIR}"

python main.py \
    --mode generate_vocab_embeddings \
    --dataset "${DATASET_NAME}" \
    --vocab_file "${PROJECT_ROOT}/result/${DATASET_NAME}/bow/vocab.json" \
    --batch_size 64

echo "[3/5] 词汇Embedding生成完成!"
echo ""

# ============================================
# 步骤4: ETM模型训练
# ============================================
echo "[4/5] ETM模型训练..."
cd "${ETM_DIR}"

python main.py train \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --num_topics "${NUM_TOPICS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}"

echo "[4/5] ETM训练完成!"
echo ""

# ============================================
# 步骤5: 评估和可视化
# ============================================
echo "[5/5] 评估和可视化..."

# 查找最新的timestamp
TIMESTAMP=$(ls -t "${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/model/theta_"*.npy 2>/dev/null | head -1 | sed 's/.*theta_\([0-9_]*\)\.npy/\1/')

if [ -z "${TIMESTAMP}" ]; then
    echo "错误: 找不到训练好的模型"
    exit 1
fi

echo "使用时间戳: ${TIMESTAMP}"

# 评估
python main.py evaluate \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --timestamp "${TIMESTAMP}"

# 可视化
python main.py visualize \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --timestamp "${TIMESTAMP}"

echo "[5/5] 评估和可视化完成!"
echo ""

echo "=============================================="
echo "Pipeline完成!"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "结果目录结构:"
echo "  ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/"
echo "    ├── model/          # 模型文件"
echo "    ├── evaluation/     # 评估结果"
echo "    └── visualization/  # 可视化图表"
echo "=============================================="
