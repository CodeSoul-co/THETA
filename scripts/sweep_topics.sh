#!/bin/bash
# =============================================================================
# THETA Topic Number Sweep
# =============================================================================
# Run THETA experiments across multiple datasets and topic counts.
# Default sweep: topics 8,10,12,14,16 on a single dataset.
# Use --datasets to run across multiple datasets in sequence.
#
# Usage:
#   bash scripts/sweep_topics.sh --dataset EUAIACT [options]
#   bash scripts/sweep_topics.sh --datasets "DS1 DS2 DS3" --topics "8 10 12" [options]
#
# To list available datasets (data/ subfolders with a _cleaned.csv):
#   bash scripts/sweep_topics.sh --list-datasets
#
# Examples:
#   bash scripts/sweep_topics.sh --dataset EUAIACT_p1_pre20240731
#   bash scripts/sweep_topics.sh --datasets "EUAIACT_p1_pre20240731 EUAIACT_p2_20240801_20250201" --topics "8 10 12 14 16"
#   bash scripts/sweep_topics.sh --datasets "EUAIACT_p1_pre20240731 EUAIACT_p2_20240801_20250201 EUAIACT_p3_20250202_20250801 EUAIACT_p4_post20250802" --topics "8 10 12 14 16"
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

# =============================================================================
# Default Parameters
# =============================================================================
DATASET=""
DATASET_LIST=""
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
LIST_DATASETS=false
EXTRA_ARGS=""

# =============================================================================
# Argument Parsing
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)       DATASET="$2";       shift 2 ;;
        --datasets)      DATASET_LIST="$2";  shift 2 ;;
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
        --list-datasets) LIST_DATASETS=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo "       $0 --datasets \"<name1> <name2> ...\" [options]"
            echo ""
            echo "Dataset Selection (choose one):"
            echo "  --dataset       Single dataset name"
            echo "  --datasets      Space-separated list of dataset names"
            echo "  --list-datasets List available datasets in data/ and exit"
            echo ""
            echo "Sweep Options:"
            echo "  --topics        Space-separated topic counts (default: \"8 10 12 14 16\")"
            echo ""
            echo "Training Options:"
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
            echo "  # Single dataset, default topics"
            echo "  $0 --dataset EUAIACT_p1_pre20240731"
            echo ""
            echo "  # All 4 phases, custom topics"
            echo "  $0 --datasets \"EUAIACT_p1_pre20240731 EUAIACT_p2_20240801_20250201 EUAIACT_p3_20250202_20250801 EUAIACT_p4_post20250802\" --topics \"8 10 12 14 16\""
            echo ""
            echo "  # List available datasets"
            echo "  $0 --list-datasets"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift ;;
    esac
done

# =============================================================================
# --list-datasets: show available data subfolders
# =============================================================================
if [ "$LIST_DATASETS" = true ]; then
    echo "Available datasets in $DATA_DIR:"
    for d in "$DATA_DIR"/*/; do
        name=$(basename "$d")
        [ "$name" = "example" ] && continue
        csv=$(ls "$d"*_cleaned.csv 2>/dev/null | head -1)
        if [ -n "$csv" ]; then
            rows=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$csv')))" 2>/dev/null || echo "?")
            echo "  $name  ($rows rows)"
        fi
    done
    exit 0
fi

# Build dataset array from --dataset or --datasets
if [ -n "$DATASET_LIST" ]; then
    read -ra DATASETS <<< "$DATASET_LIST"
elif [ -n "$DATASET" ]; then
    DATASETS=("$DATASET")
else
    echo "[ERROR] --dataset or --datasets is required (use --list-datasets to browse)"
    exit 1
fi

# Convert topic list string to array
read -ra TOPICS <<< "$TOPIC_LIST"
TOTAL_DS=${#DATASETS[@]}
TOTAL_K=${#TOPICS[@]}
TOTAL=$((TOTAL_DS * TOTAL_K))

# =============================================================================
# Banner
# =============================================================================
echo "=========================================="
echo "  THETA Topic × Dataset Sweep"
echo "=========================================="
echo "  Datasets  : ${DATASETS[*]}"
echo "  Topics    : ${TOPICS[*]}"
echo "  Model     : Qwen $MODEL_SIZE ($MODE)"
echo "  Epochs    : $EPOCHS  |  LR: $LEARNING_RATE"
echo "  Language  : $LANGUAGE"
echo "  Total runs: $TOTAL ($TOTAL_DS datasets × $TOTAL_K topics)"
echo "=========================================="
echo ""

SKIP_VIZ_FLAG=""
[ "$SKIP_VIZ" = true ] && SKIP_VIZ_FLAG="--skip-viz"

SUCCESS_LIST=()
FAIL_LIST=()
RUN_IDX=0

# =============================================================================
# Main loop: for each dataset, prepare data once, then sweep topics
# =============================================================================
for DS in "${DATASETS[@]}"; do

    echo ""
    echo "##################################################"
    echo "  Dataset: $DS"
    echo "##################################################"

    # Verify dataset CSV exists
    DS_CSV=$(ls "$DATA_DIR/$DS/"*_cleaned.csv 2>/dev/null | head -1)
    if [ -z "$DS_CSV" ]; then
        echo "[WARN] No *_cleaned.csv found in $DATA_DIR/$DS/ — skipping dataset"
        for K in "${TOPICS[@]}"; do
            FAIL_LIST+=("$DS/K=$K")
            RUN_IDX=$((RUN_IDX + 1))
        done
        continue
    fi

    # ---- Data preparation (once per dataset) --------------------------------
    THETA_BASE="$RESULT_DIR/$DS/$MODEL_SIZE/theta"

    _check_data_ready() {
        local latest
        latest=$(ls -dt "$THETA_BASE"/exp_* 2>/dev/null | head -1)
        if [ -n "$latest" ] && \
           [ -f "$latest/data/embeddings/embeddings.npy" ] && \
           [ -f "$latest/data/bow/bow_matrix.npy" ]; then
            echo "$latest"
        fi
    }

    EXISTING=$(_check_data_ready || true)
    if [ -n "$EXISTING" ]; then
        DATA_EXP=$(basename "$EXISTING")
        echo "[Data] Reusing: $DATA_EXP"
    else
        echo "[Data] Running prepare_data.py for $DS ..."
        cd "$ETM_DIR"
        python prepare_data.py \
            --dataset "$DS" \
            --model theta \
            --model_size "$MODEL_SIZE" \
            --mode "$MODE" \
            --vocab_size 5000 \
            --batch_size 32 \
            --max_length 512 \
            --gpu "$GPU"
        cd "$PROJECT_ROOT"

        DATA_EXP=$(basename "$(_check_data_ready)")
        if [ -z "$DATA_EXP" ]; then
            echo "[ERROR] Data preparation failed for $DS — skipping"
            for K in "${TOPICS[@]}"; do
                FAIL_LIST+=("$DS/K=$K")
                RUN_IDX=$((RUN_IDX + 1))
            done
            continue
        fi
        echo "[Data] Prepared: $DATA_EXP"
    fi
    echo ""

    # ---- Topic sweep for this dataset --------------------------------------
    for K in "${TOPICS[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        echo "------------------------------------------"
        echo "  [$RUN_IDX/$TOTAL] $DS  K=$K"
        echo "------------------------------------------"

        bash "$SCRIPT_DIR/train_theta.sh" \
            --dataset "$DS" \
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
        && SUCCESS_LIST+=("$DS/K=$K") \
        || FAIL_LIST+=("$DS/K=$K")

        echo ""
    done

done

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "  Sweep Complete  ($RUN_IDX/$TOTAL runs)"
echo "=========================================="
echo "  Succeeded (${#SUCCESS_LIST[@]}): ${SUCCESS_LIST[*]}"
[ ${#FAIL_LIST[@]} -gt 0 ] && echo "  Failed    (${#FAIL_LIST[@]}): ${FAIL_LIST[*]}"
echo ""

# Metrics table
echo "--- Metrics Summary ---"
printf "  %-42s  %s\n" "Dataset / K" "NPMI    C_V     TD"
for DS in "${DATASETS[@]}"; do
    THETA_BASE="$RESULT_DIR/$DS/$MODEL_SIZE/theta"
    for K in "${TOPICS[@]}"; do
        EXP_DIR=$(ls -dt "$THETA_BASE"/exp_*sweep_k${K}* 2>/dev/null | head -1)
        if [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/metrics.json" ]; then
            METRICS=$(python3 -c "
import json
m = json.load(open('$EXP_DIR/metrics.json'))
npmi = m.get('npmi', m.get('NPMI', None))
cv   = m.get('cv',   m.get('C_V',  m.get('coherence_cv', None)))
td   = m.get('td',   m.get('TD',   m.get('topic_diversity', None)))
if all(isinstance(x, float) for x in [npmi,cv,td]):
    print(f'{npmi:.4f}  {cv:.4f}  {td:.4f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
            printf "  %-42s  %s\n" "$DS / K=$K" "$METRICS"
        fi
    done
done
echo "=========================================="
