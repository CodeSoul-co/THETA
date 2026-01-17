#!/bin/bash
# ============================================================================
# 步骤6: 模型评估
# ============================================================================
# 功能: 评估训练好的主题模型
# 评估指标:
#   - Topic Coherence (主题一致性)
#   - Topic Diversity (主题多样性)
#   - Perplexity (困惑度)
# 输入: 训练好的模型和矩阵
# 输出: 评估报告 (result/{dataset}/{mode}/evaluation/)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤6: 模型评估"
echo "=============================================="
echo "数据集: ${DATASET_NAME}"
echo "模式: ${MODE}"
echo "=============================================="

cd "${ETM_DIR}"

# 查找最新的timestamp
TIMESTAMP=$(ls -t "${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/model/theta_"*.npy 2>/dev/null | head -1 | sed 's/.*theta_\([0-9_]*\)\.npy/\1/')

if [ -z "${TIMESTAMP}" ]; then
    echo "错误: 找不到训练好的模型，请先运行训练"
    exit 1
fi

echo "使用时间戳: ${TIMESTAMP}"

# 运行评估
python main.py evaluate \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --timestamp "${TIMESTAMP}"

echo ""
echo "=============================================="
echo "评估完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/evaluation/"
echo "=============================================="
echo ""
echo "下一步: 运行 07_visualize.sh ${DATASET_NAME} ${MODE}"
