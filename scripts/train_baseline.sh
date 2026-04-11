#!/bin/bash
# =============================================================================
# Baseline Model Training Script
# =============================================================================
# Train baseline topic models for comparison with THETA
#
# Configuration Priority (highest to lowest):
#   1. CLI arguments (--num_topics 50)
#   2. YAML config (config/default.yaml)
#   3. .env paths
#
# Supported Models:
#   Traditional: lda, hdp, stm, btm
#   Neural: etm, ctm, dtm, nvdm, gsm, prodlda, bertopic
#
# Usage:
#   ./train_baseline.sh <model> [options]           # Single model
#   ./train_baseline.sh --models <list> [options]   # Multiple models
#
# Examples:
#   ./train_baseline.sh lda --dataset mydata
#   ./train_baseline.sh --models lda,hdp --dataset mydata --num_topics 30
#   ./train_baseline.sh etm --dataset mydata --epochs 50
# =============================================================================

set -e

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

# YAML config path
YAML_CONFIG="$PROJECT_ROOT/config/default.yaml"

# =============================================================================
# Helper: Read YAML config value
# =============================================================================
read_yaml() {
    local model="$1"
    local key="$2"
    local default="$3"
    
    if [ -f "$YAML_CONFIG" ]; then
        # Try model-specific value first, then global
        local value=$(python3 -c "
import yaml
with open('$YAML_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
model_config = config.get('$model', {})
global_config = config.get('global', {})
value = model_config.get('$key', global_config.get('$key'))
print(value if value is not None else '')
" 2>/dev/null)
        if [ -n "$value" ] && [ "$value" != "None" ]; then
            echo "$value"
        else
            echo "$default"
        fi
    else
        echo "$default"
    fi
}

# =============================================================================
# Default values (will be overridden by YAML, then CLI)
# =============================================================================
DATASET=""
MODELS=""
GPU=0
SKIP_TRAIN=false
SKIP_VIZ=false
DATA_EXP=""
EXP_NAME=""
WORKSPACE_DIR=""

# These will be loaded from YAML per-model
NUM_TOPICS=""
VOCAB_SIZE=""
EPOCHS=""
BATCH_SIZE=""
HIDDEN_DIM=""
LEARNING_RATE=""
DROPOUT=""
LANGUAGE=""
MAX_ITER=""
MAX_TOPICS=""
N_ITER=""
ALPHA=""
BETA=""
INFERENCE_TYPE=""

# Pass-through arguments
PASS_THROUGH_ARGS=""

# =============================================================================
# Parse command line arguments
# =============================================================================
# Check if first argument is a model name (not starting with --)
if [[ $# -gt 0 && ! "$1" == --* ]]; then
    MODELS="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-viz) SKIP_VIZ=true; shift ;;
        --with-viz) SKIP_VIZ=false; shift ;;
        --max_iter) MAX_ITER="$2"; shift 2 ;;
        --max_topics) MAX_TOPICS="$2"; shift 2 ;;
        --n_iter) N_ITER="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --inference_type) INFERENCE_TYPE="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --data_exp) DATA_EXP="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --workspace_dir) WORKSPACE_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <model> [options]  OR  $0 --models <list> [options]"
            echo ""
            echo "Configuration Priority: CLI args > config/default.yaml > defaults"
            echo ""
            echo "Supported Models:"
            echo "  Traditional: lda, hdp, stm, btm"
            echo "  Neural:      etm, ctm, dtm, nvdm, gsm, prodlda, bertopic"
            echo ""
            echo "Required:"
            echo "  --dataset       Dataset name"
            echo ""
            echo "Training Options (defaults from config/default.yaml):"
            echo "  --num_topics    Number of topics"
            echo "  --vocab_size    Vocabulary size"
            echo "  --epochs        Training epochs (neural models)"
            echo "  --batch_size    Batch size"
            echo "  --hidden_dim    Hidden dimension"
            echo "  --learning_rate Learning rate"
            echo "  --dropout       Dropout rate"
            echo ""
            echo "Other Options:"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      Visualization language: chinese, english, zh, en (default: zh)"
            echo "  --skip-train    Skip training"
            echo "  --skip-viz      Skip visualization"
            echo "  --workspace_dir Workspace directory path"
            echo ""
            echo "Examples:"
            echo "  # Train single model with YAML defaults"
            echo "  $0 lda --dataset mydata"
            echo ""
            echo "  # Train with custom parameters (override YAML)"
            echo "  $0 lda --dataset mydata --num_topics 30 --epochs 50"
            echo ""
            echo "  # Train multiple models"
            echo "  $0 --models lda,hdp,etm --dataset mydata"
            exit 0
            ;;
        *)
            # Collect unknown arguments for pass-through
            if [[ "$1" == --* ]]; then
                if [[ $# -gt 1 && ! "$2" == --* ]]; then
                    PASS_THROUGH_ARGS="$PASS_THROUGH_ARGS $1 $2"
                    shift 2
                else
                    PASS_THROUGH_ARGS="$PASS_THROUGH_ARGS $1"
                    shift
                fi
            else
                PASS_THROUGH_ARGS="$PASS_THROUGH_ARGS $1"
                shift
            fi
            ;;
    esac
done

# =============================================================================
# Validate required parameters
# =============================================================================
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

if [ -z "$MODELS" ]; then
    echo "Error: Model name or --models is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# =============================================================================
# Load defaults from YAML for the first model
# =============================================================================
FIRST_MODEL=$(echo "$MODELS" | cut -d',' -f1)

echo "=========================================="
echo "Loading configuration for: $FIRST_MODEL"
echo "=========================================="

# Load from YAML if not set via CLI
[ -z "$NUM_TOPICS" ] && NUM_TOPICS=$(read_yaml "$FIRST_MODEL" "num_topics" "20")
[ -z "$VOCAB_SIZE" ] && VOCAB_SIZE=$(read_yaml "$FIRST_MODEL" "vocab_size" "5000")
[ -z "$EPOCHS" ] && EPOCHS=$(read_yaml "$FIRST_MODEL" "epochs" "100")
[ -z "$BATCH_SIZE" ] && BATCH_SIZE=$(read_yaml "$FIRST_MODEL" "batch_size" "64")
[ -z "$HIDDEN_DIM" ] && HIDDEN_DIM=$(read_yaml "$FIRST_MODEL" "hidden_dim" "256")
[ -z "$LEARNING_RATE" ] && LEARNING_RATE=$(read_yaml "$FIRST_MODEL" "learning_rate" "0.002")
[ -z "$DROPOUT" ] && DROPOUT=$(read_yaml "$FIRST_MODEL" "dropout" "0.2")
[ -z "$LANGUAGE" ] && LANGUAGE=$(read_yaml "visualization" "language" "zh")

# Model-specific defaults
[ -z "$MAX_ITER" ] && MAX_ITER=$(read_yaml "$FIRST_MODEL" "max_iter" "100")
[ -z "$MAX_TOPICS" ] && MAX_TOPICS=$(read_yaml "$FIRST_MODEL" "max_topics" "150")
[ -z "$N_ITER" ] && N_ITER=$(read_yaml "$FIRST_MODEL" "n_iter" "100")
[ -z "$ALPHA" ] && ALPHA=$(read_yaml "$FIRST_MODEL" "alpha" "1.0")
[ -z "$BETA" ] && BETA=$(read_yaml "$FIRST_MODEL" "beta" "0.01")
[ -z "$INFERENCE_TYPE" ] && INFERENCE_TYPE=$(read_yaml "$FIRST_MODEL" "inference_type" "zeroshot")

echo "  num_topics:    $NUM_TOPICS"
echo "  epochs:        $EPOCHS"
echo "  batch_size:    $BATCH_SIZE"
echo "  hidden_dim:    $HIDDEN_DIM"
echo "  learning_rate: $LEARNING_RATE"
echo "  language:      $LANGUAGE"
echo ""

# =============================================================================
# Step 1: Check if data needs preprocessing
# =============================================================================
DATA_CSV="$DATA_DIR/$DATASET/${DATASET}_cleaned.csv"
RAW_DATA_DIR="$DATA_DIR/$DATASET"

# Check if workspace exists with BOW matrix
if [ -z "$WORKSPACE_DIR" ]; then
    # Try to find existing workspace
    WORKSPACE_DIR="$RESULT_DIR/baseline/$DATASET/data"
    if [ -d "$WORKSPACE_DIR" ]; then
        LATEST_EXP=$(ls -dt "$WORKSPACE_DIR"/exp_* 2>/dev/null | head -1)
        if [ -n "$LATEST_EXP" ] && [ -f "$LATEST_EXP/bow_matrix.npy" ]; then
            WORKSPACE_DIR="$LATEST_EXP"
            echo "[INFO] Found existing workspace: $WORKSPACE_DIR"
        else
            WORKSPACE_DIR=""
        fi
    else
        WORKSPACE_DIR=""
    fi
fi

# If no workspace, need to preprocess data
if [ -z "$WORKSPACE_DIR" ] || [ ! -f "$WORKSPACE_DIR/bow_matrix.npy" ]; then
    echo "=========================================="
    echo "[Step 1] Data Preprocessing"
    echo "=========================================="
    
    # Check if cleaned CSV exists
    if [ ! -f "$DATA_CSV" ]; then
        echo "[INFO] No cleaned CSV found, checking for raw data..."
        
        # Check for docx/pdf/txt files
        RAW_FILES=$(find "$RAW_DATA_DIR" -type f \( -name "*.docx" -o -name "*.pdf" -o -name "*.txt" \) 2>/dev/null | head -1)
        if [ -n "$RAW_FILES" ]; then
            echo "[INFO] Found raw documents, converting to CSV..."
            python "$SCRIPT_DIR/../src/data/doc_converter.py" \
                --input_dir "$RAW_DATA_DIR" \
                --output "$DATA_CSV" \
                --language chinese
        else
            # Check for existing CSV
            EXISTING_CSV=$(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" 2>/dev/null | head -1)
            if [ -n "$EXISTING_CSV" ]; then
                DATA_CSV="$EXISTING_CSV"
                echo "[INFO] Using existing CSV: $DATA_CSV"
            else
                echo "[ERROR] No data found in $RAW_DATA_DIR"
                echo "Please provide: docx/pdf/txt files OR a CSV file with 'text' column"
                exit 1
            fi
        fi
    fi
    
    # Prepare data (generate BOW and SBERT embeddings for CTM)
    echo "[INFO] Preparing baseline data (BOW + SBERT embeddings)..."
    
    # Check if CTM is in the model list - need SBERT
    NEED_SBERT="False"
    if echo "$MODELS" | grep -qE "ctm|bertopic"; then
        NEED_SBERT="True"
    fi
    
    # Resolve to absolute paths before cd (DATA_DIR/RESULT_DIR may be relative)
    ABS_DATA_DIR="$(cd "$PROJECT_ROOT" && realpath "$DATA_DIR")"
    ABS_RESULT_DIR="$(cd "$PROJECT_ROOT" && realpath "$RESULT_DIR")"

    cd "$ETM_DIR"
    python -c "
import sys
sys.path.insert(0, '.')
from model.baseline_data import prepare_baseline_data

result = prepare_baseline_data(
    dataset='$DATASET',
    vocab_size=$VOCAB_SIZE,
    data_dir='$ABS_DATA_DIR',
    save_dir='$ABS_RESULT_DIR/baseline/$DATASET/data',
    generate_sbert=$NEED_SBERT
)
print(f'Prepared {len(result[\"texts\"])} documents')
print(f'BOW shape: {result[\"bow_matrix\"].shape}')
if result.get('sbert_embeddings') is not None:
    print(f'SBERT embeddings: {result[\"sbert_embeddings\"].shape}')
"
    
    # Find the newly created workspace
    # Try exp_* pattern first, then dataset name pattern
    WORKSPACE_DIR=$(ls -dt "$ABS_RESULT_DIR/baseline/$DATASET/data"/exp_* 2>/dev/null | head -1)
    if [ -z "$WORKSPACE_DIR" ] || [ ! -f "$WORKSPACE_DIR/bow_matrix.npy" ]; then
        # Try dataset name as subdirectory
        if [ -f "$ABS_RESULT_DIR/baseline/$DATASET/data/$DATASET/bow_matrix.npy" ]; then
            WORKSPACE_DIR="$ABS_RESULT_DIR/baseline/$DATASET/data/$DATASET"
        fi
    fi
    
    if [ -n "$WORKSPACE_DIR" ] && [ -f "$WORKSPACE_DIR/bow_matrix.npy" ]; then
        echo "[INFO] Created workspace: $WORKSPACE_DIR"
    else
        echo "[ERROR] Failed to create workspace"
        exit 1
    fi
fi

echo ""
echo "[INFO] Using workspace: $WORKSPACE_DIR"

# =============================================================================
# Step 2: Build and execute training command
# =============================================================================
cd "$ETM_DIR"

CMD="python run_pipeline.py --dataset $DATASET --models $MODELS"
CMD="$CMD --num_topics $NUM_TOPICS --vocab_size $VOCAB_SIZE"
CMD="$CMD --epochs $EPOCHS --batch_size $BATCH_SIZE"
CMD="$CMD --hidden_dim $HIDDEN_DIM --learning_rate $LEARNING_RATE"
CMD="$CMD --dropout $DROPOUT --language $LANGUAGE --gpu $GPU"

# Add workspace_dir if provided
if [ -n "$WORKSPACE_DIR" ]; then
    CMD="$CMD --workspace_dir $WORKSPACE_DIR"
fi

# Model-specific parameters
case "$FIRST_MODEL" in
    lda|stm) CMD="$CMD --max_iter $MAX_ITER" ;;
    hdp) CMD="$CMD --max_topics $MAX_TOPICS --alpha $ALPHA" ;;
    btm) CMD="$CMD --n_iter $N_ITER --alpha $ALPHA --beta $BETA" ;;
    ctm) CMD="$CMD --inference_type $INFERENCE_TYPE" ;;
esac

# Flags
[ "$SKIP_TRAIN" = true ] && CMD="$CMD --skip-train"
[ "$SKIP_VIZ" = true ] && CMD="$CMD --skip-viz"
[ -n "$DATA_EXP" ] && CMD="$CMD --data_exp $DATA_EXP"
[ -n "$EXP_NAME" ] && CMD="$CMD --exp_name $EXP_NAME"
[ -n "$PASS_THROUGH_ARGS" ] && CMD="$CMD$PASS_THROUGH_ARGS"

echo "=========================================="
echo "Baseline Model Training"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Models:     $MODELS"
echo "Language:   $LANGUAGE"
echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: $RESULT_DIR/$DATASET/"
echo "=========================================="
