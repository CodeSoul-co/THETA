#!/bin/bash
# =============================================================================
# Topic Model Sweep (THETA + Baselines)
# =============================================================================
# Run topic model experiments across multiple datasets and topic counts.
# Supports THETA (default) and baseline models (lda, hdp, etm, ctm, etc.)
#
# Usage:
#   bash scripts/sweep_topics.sh --dataset DS [options]
#   bash scripts/sweep_topics.sh --datasets "DS1 DS2" --topics "8 10 12" [options]
#   bash scripts/sweep_topics.sh --model lda --datasets "DS1 DS2" --topics "8 10"
#
# Model options:
#   theta (default)  — THETA main model (Qwen embedding + ETM)
#   lda              — LDA (sklearn, CPU only)
#   hdp              — HDP (gensim, CPU only)
#   etm              — ETM neural baseline (GPU)
#   ctm              — CTM neural baseline (GPU)
#   btm              — BTM (short-text, CPU)
#   prodlda          — ProdLDA neural baseline (GPU)
#   bertopic         — BERTopic (GPU for embeddings)
#
# To list available datasets:
#   bash scripts/sweep_topics.sh --list-datasets
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
MODEL="theta"          # theta | lda | hdp | etm | ctm | btm | prodlda | bertopic
MODEL_SIZE="0.6B"      # only used for theta
MODE="zero_shot"       # zero_shot | unsupervised | supervised (theta only)
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
        --model)         MODEL="$2";         shift 2 ;;
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
            echo "  --model         Model to train (default: theta)"
            echo "                  theta | lda | hdp | etm | ctm | btm | prodlda | bertopic"
            echo "  --topics        Space-separated topic counts (default: \"8 10 12 14 16\")"
            echo ""
            echo "THETA Options:"
            echo "  --model_size    0.6B | 4B | 8B (default: 0.6B)"
            echo "  --mode          zero_shot | unsupervised | supervised (default: zero_shot)"
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
            echo "  # THETA sweep (default)"
            echo "  $0 --dataset EUAIACT_p1_pre20240731"
            echo ""
            echo "  # LDA baseline sweep across all 4 phases"
            echo "  $0 --model lda --datasets \"EUAIACT_p1_pre20240731 EUAIACT_p2_20240801_20250201 EUAIACT_p3_20250202_20250801 EUAIACT_p4_post20250802\" --topics \"8 10 12 14 16\""
            echo ""
            echo "  # Multiple baseline models"
            echo "  $0 --model \"lda etm ctm\" --dataset EUAIACT_p1_pre20240731 --topics \"10 12\""
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

# Convert topic list and model list to arrays
read -ra TOPICS <<< "$TOPIC_LIST"
read -ra MODELS <<< "$MODEL"
TOTAL_DS=${#DATASETS[@]}
TOTAL_K=${#TOPICS[@]}
TOTAL_M=${#MODELS[@]}
TOTAL=$((TOTAL_DS * TOTAL_M * TOTAL_K))

# Classify models
THETA_MODELS=("theta")
BASELINE_MODELS=("lda" "hdp" "etm" "ctm" "btm" "prodlda" "bertopic" "nvdm" "gsm" "dtm" "stm")

is_theta_model() {
    local m="$1"
    for t in "${THETA_MODELS[@]}"; do [ "$t" = "$m" ] && return 0; done
    return 1
}

# =============================================================================
# Banner
# =============================================================================
echo "=========================================="
echo "  Topic Model Sweep"
echo "=========================================="
echo "  Datasets  : ${DATASETS[*]}"
echo "  Models    : ${MODELS[*]}"
echo "  Topics    : ${TOPICS[*]}"
if is_theta_model "${MODELS[0]}"; then
    echo "  Qwen      : $MODEL_SIZE ($MODE)"
fi
echo "  Language  : $LANGUAGE"
echo "  Total runs: $TOTAL ($TOTAL_DS DS × $TOTAL_M models × $TOTAL_K topics)"
echo "=========================================="
echo ""

SKIP_VIZ_FLAG=""
[ "$SKIP_VIZ" = true ] && SKIP_VIZ_FLAG="--skip-viz"

SUCCESS_LIST=()
FAIL_LIST=()
RUN_IDX=0

# =============================================================================
# Main loop: dataset → model → topic
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
        for MDL in "${MODELS[@]}"; do
            for K in "${TOPICS[@]}"; do
                FAIL_LIST+=("$DS/$MDL/K=$K")
                RUN_IDX=$((RUN_IDX + 1))
            done
        done
        continue
    fi

    for MDL in "${MODELS[@]}"; do

        echo ""
        echo "  ---- Model: $MDL ----"

        if is_theta_model "$MDL"; then
            # ----------------------------------------------------------------
            # THETA: prepare data once (embeddings + BOW), then sweep K
            # ----------------------------------------------------------------
            THETA_BASE="$RESULT_DIR/$DS/$MODEL_SIZE/theta"

            _check_data_ready() {
                for exp in $(ls -dt "$THETA_BASE"/exp_* 2>/dev/null); do
                    [ -f "$exp/data/embeddings/embeddings.npy" ] || continue
                    [ -f "$exp/data/bow/bow_matrix.npy" ]        || continue
                    # Verify the exp was prepared with the same mode
                    local cfg="$exp/config.json"
                    if [ -f "$cfg" ]; then
                        local saved_mode
                        saved_mode=$(python3 -c "
import json, sys
try:
    m = json.load(open('$cfg'))
    print(m.get('embedding', {}).get('mode', '') or m.get('mode', ''))
except: print('')
" 2>/dev/null)
                        [ "$saved_mode" = "$MODE" ] || continue
                    fi
                    echo "$exp"
                    return
                done
            }

            EXISTING=$(_check_data_ready || true)
            if [ -n "$EXISTING" ]; then
                DATA_EXP=$(basename "$EXISTING")
                echo "  [Data] Reusing: $DATA_EXP"
            else
                echo "  [Data] Running prepare_data.py for $DS ..."
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
                    echo "  [ERROR] Data prep failed for $DS/$MDL — skipping"
                    for K in "${TOPICS[@]}"; do
                        FAIL_LIST+=("$DS/$MDL/K=$K")
                        RUN_IDX=$((RUN_IDX + 1))
                    done
                    continue
                fi
                echo "  [Data] Prepared: $DATA_EXP"
            fi

            for K in "${TOPICS[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))
                echo "  ------------------------------------------"
                echo "  [$RUN_IDX/$TOTAL] $DS  $MDL  K=$K"
                echo "  ------------------------------------------"

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
                && SUCCESS_LIST+=("$DS/$MDL/K=$K") \
                || FAIL_LIST+=("$DS/$MDL/K=$K")
                echo ""
            done

        else
            # ----------------------------------------------------------------
            # Baseline: call train_baseline.sh, one K at a time
            # ----------------------------------------------------------------
            for K in "${TOPICS[@]}"; do
                RUN_IDX=$((RUN_IDX + 1))
                echo "  ------------------------------------------"
                echo "  [$RUN_IDX/$TOTAL] $DS  $MDL  K=$K"
                echo "  ------------------------------------------"

                bash "$SCRIPT_DIR/train_baseline.sh" "$MDL" \
                    --dataset "$DS" \
                    --num_topics "$K" \
                    --epochs "$EPOCHS" \
                    --gpu "$GPU" \
                    --language "$LANGUAGE" \
                    $SKIP_VIZ_FLAG \
                    $EXTRA_ARGS \
                && SUCCESS_LIST+=("$DS/$MDL/K=$K") \
                || FAIL_LIST+=("$DS/$MDL/K=$K")
                echo ""
            done
        fi

    done  # models

done  # datasets

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
printf "  %-50s  %s\n" "Dataset / Model / K" "NPMI    C_V     TD"
for DS in "${DATASETS[@]}"; do
    for MDL in "${MODELS[@]}"; do
        if is_theta_model "$MDL"; then
            BASE_DIR="$RESULT_DIR/$DS/$MODEL_SIZE/theta"
            EXP_PATTERN="exp_*sweep_k"
        else
            BASE_DIR="$RESULT_DIR/$DS/baseline/$MDL"
            EXP_PATTERN="exp_*k"
        fi
        for K in "${TOPICS[@]}"; do
            EXP_DIR=$(ls -dt "$BASE_DIR"/${EXP_PATTERN}${K}* 2>/dev/null | head -1)
            # also check metrics.json directly in BASE_DIR for baseline
            METRICS_FILE=""
            [ -n "$EXP_DIR" ] && [ -f "$EXP_DIR/metrics.json" ] && METRICS_FILE="$EXP_DIR/metrics.json"
            [ -z "$METRICS_FILE" ] && [ -f "$BASE_DIR/metrics.json" ] && METRICS_FILE="$BASE_DIR/metrics.json"
            if [ -n "$METRICS_FILE" ]; then
                METRICS=$(python3 -c "
import json
m = json.load(open('$METRICS_FILE'))
npmi = m.get('npmi', m.get('NPMI', None))
cv   = m.get('cv',   m.get('C_V',  m.get('coherence_cv', None)))
td   = m.get('td',   m.get('TD',   m.get('topic_diversity', None)))
if all(isinstance(x, float) for x in [npmi,cv,td]):
    print(f'{npmi:.4f}  {cv:.4f}  {td:.4f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
                printf "  %-50s  %s\n" "$DS / $MDL / K=$K" "$METRICS"
            fi
        done
    done
done
echo "=========================================="
