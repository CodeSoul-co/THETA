#!/bin/bash
# Baseline Model Training Script
# Train LDA, ETM, CTM, or DTM models

set -e

# Default values
DATASET=""
MODELS="lda"
NUM_TOPICS=20
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
GPU=0
LANGUAGE="en"
SKIP_TRAIN=false
SKIP_VIZ=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-viz) SKIP_VIZ=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> --models <model_list> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset       Dataset name (required)"
            echo "  --models        Model list: lda, etm, ctm, dtm (comma-separated, default: lda)"
            echo "  --num_topics    Number of topics (default: 20)"
            echo "  --epochs        Training epochs (default: 100)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "  --skip-train    Skip training, only evaluate"
            echo "  --skip-viz      Skip visualization"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset hatespeech --models lda --num_topics 20"
            echo "  $0 --dataset hatespeech --models lda,etm,ctm --num_topics 20"
            echo "  $0 --dataset edu_data --models dtm --language zh"
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
echo "Baseline Model Training"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Models:     $MODELS"
echo "Num Topics: $NUM_TOPICS"
echo "Epochs:     $EPOCHS"
echo "Language:   $LANGUAGE"
echo ""

cd /root/autodl-tmp/ETM

# Build command
CMD="python run_pipeline.py --dataset $DATASET --models $MODELS"
CMD="$CMD --num_topics $NUM_TOPICS --epochs $EPOCHS --batch_size $BATCH_SIZE"
CMD="$CMD --hidden_dim $HIDDEN_DIM --learning_rate $LEARNING_RATE"
CMD="$CMD --gpu $GPU --language $LANGUAGE"

if [ "$SKIP_TRAIN" = true ]; then
    CMD="$CMD --skip-train"
fi

if [ "$SKIP_VIZ" = true ]; then
    CMD="$CMD --skip-viz"
fi

echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: /root/autodl-tmp/result/baseline/$DATASET/"
echo "=========================================="
