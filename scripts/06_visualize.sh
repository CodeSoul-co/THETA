#!/bin/bash
# THETA Visualization Script
# Generate visualizations for trained models

set -e

# Default values
RESULT_DIR=""
DATASET=""
MODEL=""
MODEL_SIZE="0.6B"
MODE="zero_shot"
NUM_TOPICS=20
LANGUAGE="en"
DPI=300
BASELINE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --result_dir) RESULT_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num_topics) NUM_TOPICS="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --dpi) DPI="$2"; shift 2 ;;
        --baseline) BASELINE=true; shift ;;
        -h|--help)
            echo "Usage: $0 --dataset <name> [options]"
            echo ""
            echo "For THETA model:"
            echo "  $0 --dataset hatespeech --model_size 0.6B --mode zero_shot --language en"
            echo ""
            echo "For Baseline models:"
            echo "  $0 --baseline --dataset hatespeech --model lda --num_topics 20 --language en"
            echo ""
            echo "Options:"
            echo "  --dataset       Dataset name (required)"
            echo "  --baseline      Use baseline model mode"
            echo "  --model         Baseline model: lda, etm, ctm, dtm"
            echo "  --model_size    THETA model size: 0.6B, 4B, 8B (default: 0.6B)"
            echo "  --mode          THETA mode: zero_shot, supervised, unsupervised (default: zero_shot)"
            echo "  --num_topics    Number of topics (default: 20)"
            echo "  --language      Visualization language: en, zh (default: en)"
            echo "  --dpi           Image DPI (default: 300)"
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
echo "THETA Visualization"
echo "=========================================="
echo "Dataset:  $DATASET"
echo "Language: $LANGUAGE"
echo ""

cd /root/autodl-tmp/ETM

if [ "$BASELINE" = true ]; then
    if [ -z "$MODEL" ]; then
        echo "Error: --model is required for baseline mode"
        exit 1
    fi
    
    RESULT_DIR="/root/autodl-tmp/result/baseline"
    echo "Model:    $MODEL (baseline)"
    echo "Topics:   $NUM_TOPICS"
    
    CMD="python -m visualization.run_visualization --baseline"
    CMD="$CMD --result_dir $RESULT_DIR --dataset $DATASET --model $MODEL"
    CMD="$CMD --num_topics $NUM_TOPICS --language $LANGUAGE --dpi $DPI"
else
    RESULT_DIR="/root/autodl-tmp/result/$MODEL_SIZE"
    echo "Model:    THETA $MODEL_SIZE"
    echo "Mode:     $MODE"
    
    CMD="python -m visualization.run_visualization"
    CMD="$CMD --result_dir $RESULT_DIR --dataset $DATASET"
    CMD="$CMD --mode $MODE --model_size $MODEL_SIZE --language $LANGUAGE --dpi $DPI"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Visualization completed!"
echo "=========================================="
