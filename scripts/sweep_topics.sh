#!/bin/bash
# =============================================================================
# THETA Topic Number Sweep
# =============================================================================
# Run THETA experiments across multiple topic counts automatically.
# Default sweep: 8, 10, 12, 14, 16 (configurable via --topics)
#
# Usage:
#   bash scripts/sweep_topics.sh --dataset EUAIACT [options]
#
# Examples:
#   bash scripts/sweep_topics.sh --dataset EUAIACT
#   bash scripts/sweep_topics.sh --dataset EUAIACT --topics "8 10 12 14 16 20"
#   bash scripts/sweep_topics.sh --dataset EUAIACT --language zh --epochs 150
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

# =============================================================================
# Default Parameters
# =============================================================================
DATASET=""
TOPIC_LIST="8 10 12 14 16"
MODEL_SIZE="0.6B"
MODE="zero_shot"
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
KL_START=0.0
KL_END=1.0
KL_WARMUP=50
PATIENCE=10
GPU=0
LANGUAGE="zh"
SKIP_VIZ=false
EXTRA_ARGS=""

# =============================================================================
# Argument Parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)       DATASET="$2";       shift 2 ;;
        --topics)        TOPIC_LIST="$2";    shift 2 ;;
        --model_size)    MODEL_SIZE="$2";    shift 2 ;;
        --mode)          MODE="$2";          shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --hidden_dim)    HIDDEN_DIM="$2";    shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --kl_start)      KL_START="$2";      shift 2 ;;
        --kl_end)        KL_END="$2";        shift 2 ;;
        --kl_warmup)     KL_WARMUP="$2";     shift 2 ;;
        --patience)      PATIENCE="$2";      shift 2 ;;
        --gpu)           GPU="$2";           shift 2 ;;
        --language)      LANGUAGE="$2";      shift 2 ;;
        --skip-viz)      SKIP_VIZ=true;      shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "Required:"
            echo "  --dataset       Dataset name"
            echo ""
            echo "Sweep Options:"
            echo "  --topics        Space-separated topic counts (default: \"8 10 12 14 16\")"
            echo ""
            echo "Training Options (same as train_theta.sh):"
            echo "  --model_size    0.6B | 4B | 8B (default: 0.6B)"
            echo "  --mode          zero_shot | supervised | unsupervised (default: zero_shot)"
            echo "  --epochs        Training epochs (default: 100)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo "  --patience      Early stopping patience (default: 10)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      zh | en (default: zh)"
            echo "  --skip-viz      Skip visualization"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset EUAIACT"
            echo "  $0 --dataset EUAIACT --topics \"8 10 12 14 16 20\""
            echo "  $0 --dataset EUAIACT --language zh --epochs 150 --skip-viz"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "[ERROR] --dataset is required"
    exit 1
fi

# Convert topic list string to array
read -ra TOPICS <<< "$TOPIC_LIST"
TOTAL=${#TOPICS[@]}

# =============================================================================
# Step 1: Prepare data once (embeddings + BOW are shared across all K)
# =============================================================================
echo "=========================================="
echo "  THETA Topic Sweep"
echo "=========================================="
echo "  Dataset  : $DATASET"
echo "  Topics   : ${TOPICS[*]}"
echo "  Model    : Qwen $MODEL_SIZE ($MODE)"
echo "  Epochs   : $EPOCHS  |  LR: $LEARNING_RATE"
echo "  Language : $LANGUAGE"
echo "=========================================="
echo ""

THETA_BASE="$RESULT_DIR/$DATASET/$MODEL_SIZE/theta"

check_data_ready() {
    local latest_exp
    latest_exp=$(ls -dt "$THETA_BASE"/exp_* 2>/dev/null | head -1)
    if [ -n "$latest_exp" ] && \
       [ -f "$latest_exp/data/embeddings/embeddings.npy" ] && \
       [ -f "$latest_exp/data/bow/bow_matrix.npy" ]; then
        echo "$latest_exp"
    fi
}

EXISTING_EXP=$(check_data_ready || true)

if [ -n "$EXISTING_EXP" ]; then
    DATA_EXP=$(basename "$EXISTING_EXP")
    echo "[Data] Reusing existing data: $DATA_EXP"
else
    echo "[Data] No preprocessed data found, running prepare_data.py..."
    cd "$ETM_DIR"
    python prepare_data.py \
        --dataset "$DATASET" \
        --model theta \
        --model_size "$MODEL_SIZE" \
        --mode "$MODE" \
        --vocab_size 5000 \
        --batch_size 32 \
        --max_length 512 \
        --gpu "$GPU"

    DATA_EXP=$(basename "$(check_data_ready)")
    if [ -z "$DATA_EXP" ]; then
        echo "[ERROR] Data preparation failed"
        exit 1
    fi
    echo "[Data] Prepared: $DATA_EXP"
fi

echo ""

# =============================================================================
# Step 2: Train one experiment per topic count
# =============================================================================
SKIP_VIZ_FLAG=""
[ "$SKIP_VIZ" = true ] && SKIP_VIZ_FLAG="--skip-viz"

SUCCESS_LIST=()
FAIL_LIST=()
IDX=0

for K in "${TOPICS[@]}"; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "  [$IDX/$TOTAL] K = $K topics"
    echo "=========================================="

    bash "$SCRIPT_DIR/train_theta.sh" \
        --dataset "$DATASET" \
        --model_size "$MODEL_SIZE" \
        --mode "$MODE" \
        --num_topics "$K" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --learning_rate "$LEARNING_RATE" \
        --kl_start "$KL_START" \
        --kl_end "$KL_END" \
        --kl_warmup "$KL_WARMUP" \
        --patience "$PATIENCE" \
        --gpu "$GPU" \
        --language "$LANGUAGE" \
        --data_exp "$DATA_EXP" \
        --exp_name "sweep_k${K}" \
        $SKIP_VIZ_FLAG \
        $EXTRA_ARGS \
    && SUCCESS_LIST+=("K=$K") \
    || FAIL_LIST+=("K=$K")

    echo ""
done

# =============================================================================
# Step 3: Summary
# =============================================================================
echo "=========================================="
echo "  Sweep Complete"
echo "=========================================="
echo "  Succeeded (${#SUCCESS_LIST[@]}/$TOTAL): ${SUCCESS_LIST[*]}"
[ ${#FAIL_LIST[@]} -gt 0 ] && echo "  Failed    (${#FAIL_LIST[@]}/$TOTAL): ${FAIL_LIST[*]}"
echo ""
echo "  Results: $RESULT_DIR/$DATASET/$MODEL_SIZE/theta/"
echo ""

# Print metrics comparison if metrics files exist
echo "--- Metrics Summary ---"
for K in "${TOPICS[@]}"; do
    EXP_DIR=$(ls -dt "$THETA_BASE"/exp_*sweep_k${K}* 2>/dev/null | head -1)
    if [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/metrics.json" ]; then
        NPMI=$(python3 -c "
import json
m = json.load(open('$EXP_DIR/metrics.json'))
npmi = m.get('npmi', m.get('NPMI', 'N/A'))
cv   = m.get('cv',   m.get('C_V',  m.get('coherence_cv', 'N/A')))
td   = m.get('td',   m.get('TD',   m.get('topic_diversity', 'N/A')))
print(f'NPMI={npmi:.4f}  C_V={cv:.4f}  TD={td:.4f}' if all(isinstance(x, float) for x in [npmi,cv,td]) else 'parsing error')
" 2>/dev/null || echo "N/A")
        echo "  K=$K : $NPMI"
    fi
done
echo "=========================================="
