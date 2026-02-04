#!/bin/bash
# THETA Data Preparation Script
# Generate embeddings and BOW for topic modeling

set -e

# Default values
DATASET=""
MODEL="theta"
MODEL_SIZE="0.6B"
MODE="zero_shot"
VOCAB_SIZE=5000
BATCH_SIZE=32
MAX_LENGTH=512
GPU=0
LANGUAGE="english"
CLEAN=false
RAW_INPUT=""
BOW_ONLY=false
CHECK_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --clean) CLEAN=true; shift ;;
        --raw-input) RAW_INPUT="$2"; shift 2 ;;
        --bow-only) BOW_ONLY=true; shift ;;
        --check-only) CHECK_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset      Dataset name (required)"
            echo "  --model        Model type: theta, baseline, dtm (default: theta)"
            echo "  --model_size   Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode         Training mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo "  --vocab_size   Vocabulary size (default: 5000)"
            echo "  --batch_size   Batch size for embedding (default: 32)"
            echo "  --max_length   Max sequence length (default: 512)"
            echo "  --gpu          GPU device ID (default: 0)"
            echo "  --language     Data language: english, chinese (default: english)"
            echo "  --clean        Clean raw data first"
            echo "  --raw-input    Path to raw input file (use with --clean)"
            echo "  --bow-only     Only generate BOW, skip embeddings"
            echo "  --check-only   Only check if files exist"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset hatespeech --model theta --model_size 0.6B --mode zero_shot"
            echo "  $0 --dataset edu_data --model theta --mode zero_shot --language chinese"
            echo "  $0 --dataset mydata --model baseline --vocab_size 5000"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

echo "=========================================="
echo "THETA Data Preparation"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Model:      $MODEL"
echo "Model Size: $MODEL_SIZE"
echo "Mode:       $MODE"
echo "Vocab Size: $VOCAB_SIZE"
echo ""

cd /root/autodl-tmp/ETM

# Build command
CMD="python prepare_data.py --dataset $DATASET --model $MODEL"

if [ "$MODEL" = "theta" ]; then
    CMD="$CMD --model_size $MODEL_SIZE --mode $MODE"
fi

CMD="$CMD --vocab_size $VOCAB_SIZE --batch_size $BATCH_SIZE --max_length $MAX_LENGTH --gpu $GPU"

if [ "$CLEAN" = true ]; then
    CMD="$CMD --clean --raw-input $RAW_INPUT --language $LANGUAGE"
fi

if [ "$BOW_ONLY" = true ]; then
    CMD="$CMD --bow-only"
fi

if [ "$CHECK_ONLY" = true ]; then
    CMD="$CMD --check-only"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "Data preparation completed!"
