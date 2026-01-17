#!/bin/bash
# ============================================================================
# 批量运行所有数据集的Pipeline
# ============================================================================
# 功能: 对所有数据集运行完整的ETM训练流程
# 用法: ./run_all_datasets.sh [mode] [num_topics] [epochs]
# 示例: ./run_all_datasets.sh zero_shot 20 50
# ============================================================================

set -e

# 参数
MODE="${1:-zero_shot}"
NUM_TOPICS="${2:-20}"
EPOCHS="${3:-50}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 数据集列表
DATASETS=("socialTwitter" "hatespeech" "mental_health" "germanCoal" "FCPB")

echo "=============================================="
echo "批量运行ETM Pipeline"
echo "=============================================="
echo "模式: ${MODE}"
echo "主题数: ${NUM_TOPICS}"
echo "训练轮数: ${EPOCHS}"
echo "数据集: ${DATASETS[*]}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "开始时间: $(date)"
echo "=============================================="
echo ""

# 记录成功和失败的数据集
SUCCESS_DATASETS=()
FAILED_DATASETS=()

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################"
    echo "# 处理数据集: ${DATASET}"
    echo "########################################"
    echo ""
    
    if bash "${SCRIPT_DIR}/run_full_pipeline.sh" "${DATASET}" "${MODE}" "${NUM_TOPICS}" "${EPOCHS}"; then
        SUCCESS_DATASETS+=("${DATASET}")
        echo "[OK] ${DATASET} 完成"
    else
        FAILED_DATASETS+=("${DATASET}")
        echo "[ERROR] ${DATASET} 失败"
    fi
done

echo ""
echo "=============================================="
echo "批量处理完成!"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "成功: ${SUCCESS_DATASETS[*]:-无}"
echo "失败: ${FAILED_DATASETS[*]:-无}"
echo "=============================================="
