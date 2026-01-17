#!/bin/bash
# ============================================================================
# 批量生成所有数据集的可视化
# ============================================================================
# 功能: 对所有已训练的模型生成可视化
# 用法: ./run_visualize_all.sh [mode]
# 示例: ./run_visualize_all.sh zero_shot
# ============================================================================

# 参数
MODE="${1:-zero_shot}"

# 项目路径
PROJECT_ROOT="/root/autodl-tmp"
ETM_DIR="${PROJECT_ROOT}/ETM"

# GPU设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 数据集列表
DATASETS=("socialTwitter" "hatespeech" "mental_health" "germanCoal" "FCPB")

echo "=============================================="
echo "批量生成可视化"
echo "=============================================="
echo "模式: ${MODE}"
echo "数据集: ${DATASETS[*]}"
echo "开始时间: $(date)"
echo "=============================================="
echo ""

cd "${ETM_DIR}"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "处理: ${DATASET} / ${MODE}"
    echo "----------------------------------------"
    
    # 查找最新的timestamp
    TIMESTAMP=$(ls -t "${PROJECT_ROOT}/result/${DATASET}/${MODE}/model/theta_"*.npy 2>/dev/null | head -1 | sed 's/.*theta_\([0-9_]*\)\.npy/\1/')
    
    if [ -z "${TIMESTAMP}" ]; then
        echo "[SKIP] ${DATASET}/${MODE} - 未找到模型"
        continue
    fi
    
    echo "时间戳: ${TIMESTAMP}"
    
    # 运行可视化
    if python main.py visualize \
        --dataset "${DATASET}" \
        --mode "${MODE}" \
        --timestamp "${TIMESTAMP}"; then
        echo "[OK] ${DATASET}/${MODE} 可视化完成"
    else
        echo "[ERROR] ${DATASET}/${MODE} 可视化失败"
    fi
done

echo ""
echo "=============================================="
echo "批量可视化完成!"
echo "结束时间: $(date)"
echo "=============================================="
