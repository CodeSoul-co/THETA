#!/bin/bash
# ============================================================================
# 步骤7: 可视化
# ============================================================================
# 功能: 生成主题模型可视化图表
# 输出:
#   - 主题词云 (topic_wordclouds.png)
#   - 主题相似度热力图 (topic_similarity.png)
#   - 文档-主题分布 (doc_topic_dist.png)
#   - 主题演化图 (如果有时间信息)
# ============================================================================

set -e

# 参数
DATASET_NAME="${1:-hatespeech}"
MODE="${2:-zero_shot}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

echo "=============================================="
echo "步骤7: 可视化"
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

# 运行可视化
python main.py visualize \
    --dataset "${DATASET_NAME}" \
    --mode "${MODE}" \
    --timestamp "${TIMESTAMP}"

echo ""
echo "=============================================="
echo "可视化完成!"
echo "输出目录: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/visualization/"
echo "=============================================="
echo ""
echo "全部流程完成! 查看结果: ${PROJECT_ROOT}/result/${DATASET_NAME}/${MODE}/"
