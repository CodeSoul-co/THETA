#!/bin/bash
# THETA Model Training Script
# Train THETA topic model with Qwen embeddings

set -e

# Default values
DATASET=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
NUM_TOPICS=20
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=512
LEARNING_RATE=0.002
KL_START=0.0
KL_END=1.0
KL_WARMUP=50
PATIENCE=10
GPU=0
LANGUAGE="en"
SKIP_TRAIN=false
SKIP_VIZ=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --hidden_dim) HIDDEN_DIM="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --kl_start) KL_START="$2"; shift 2 ;;
        --kl_end) KL_END="$2"; shift 2 ;;
        --kl_warmup) KL_WARMUP="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-viz) SKIP_VIZ=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset       Dataset name (required)"
            echo "  --model_size    Qwen model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode          Training mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo "  --num_topics    Number of topics (default: 20)"
            echo "  --epochs        Training epochs (default: 100)"
            echo "  --batch_size    Batch size (default: 64)"
            echo "  --hidden_dim    Hidden dimension (default: 512)"
            echo "  --learning_rate Learning rate (default: 0.002)"
            echo "  --kl_start      KL annealing start (default: 0.0)"
            echo "  --kl_end        KL annealing end (default: 1.0)"
            echo "  --kl_warmup     KL warmup epochs (default: 50)"
            echo "  --patience      Early stopping patience (default: 10)"
            echo "  --gpu           GPU device ID (default: 0)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "  --skip-train    Skip training, only evaluate"
            echo "  --skip-viz      Skip visualization"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset hatespeech --model_size 0.6B --mode zero_shot --num_topics 20"
            echo "  $0 --dataset edu_data --mode zero_shot --language zh"
            echo "  $0 --dataset mental_health --model_size 4B --mode supervised"
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
echo "THETA Model Training"
echo "=========================================="
echo "Dataset:     $DATASET"
echo "Model Size:  $MODEL_SIZE"
echo "Mode:        $MODE"
echo "Num Topics:  $NUM_TOPICS"
echo "Epochs:      $EPOCHS"
echo "Language:    $LANGUAGE"
echo ""

cd /root/autodl-tmp/ETM

# Build command
CMD="python run_pipeline.py --dataset $DATASET --models theta"
CMD="$CMD --model_size $MODEL_SIZE --mode $MODE"
CMD="$CMD --num_topics $NUM_TOPICS --epochs $EPOCHS --batch_size $BATCH_SIZE"
CMD="$CMD --hidden_dim $HIDDEN_DIM --learning_rate $LEARNING_RATE"
CMD="$CMD --kl_start $KL_START --kl_end $KL_END --kl_warmup $KL_WARMUP"
CMD="$CMD --patience $PATIENCE --gpu $GPU --language $LANGUAGE"

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
echo "Results saved to: /root/autodl-tmp/result/$MODEL_SIZE/$DATASET/$MODE/"
echo "=========================================="
