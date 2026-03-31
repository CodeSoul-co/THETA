#!/bin/bash
# =============================================================================
# THETA Quick Start - One-Click Pipeline
# =============================================================================
# Complete workflow: Raw Data → Clean → Embed → Train → Evaluate → Visualize
#
# Supports two input modes:
#   1. Directory with docx/pdf/txt files → Auto-convert to CSV
#   2. Pre-cleaned CSV file → Direct processing
#
# Usage:
#   ./quick_start.sh <dataset_name> [--language <en|zh>]
#
# Examples:
#   ./quick_start.sh my_policy_docs --language zh
#   ./quick_start.sh my_research_docs --language en
#   ./quick_start.sh my_dataset                        # defaults to English
# =============================================================================

set -e

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

# Default values
DATASET=""
LANGUAGE="cn"
MIN_DOCUMENT_COUNT=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --language|-l)
            LANGUAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <dataset_name> [--language <en|zh>]"
            echo ""
            echo "Arguments:"
            echo "  dataset_name    Name of the dataset directory in data/"
            echo ""
            echo "Options:"
            echo "  --language, -l  Language for processing: en (English) or zh (Chinese)"
            echo "                  Default: en"
            echo ""
            echo "Examples:"
            echo "  $0 my_policy_docs --language zh"
            echo "  $0 my_research_docs --language en"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$DATASET" ]; then
                DATASET="$1"
            fi
            shift
            ;;
    esac
done

# Validate dataset name
if [ -z "$DATASET" ]; then
    DATASET="my_dataset"
fi

# Map language code to full name for dataclean
if [ "$LANGUAGE" = "zh" ] || [ "$LANGUAGE" = "chinese" ] || [ "$LANGUAGE" = "cn" ]; then
    LANGUAGE_FULL="chinese"
    LANGUAGE_DISPLAY="Chinese"
    LANGUAGE="zh"
else
    LANGUAGE_FULL="english"
    LANGUAGE_DISPLAY="English"
    LANGUAGE="en"
fi

echo "=========================================="
echo "THETA Quick Start - $LANGUAGE_DISPLAY Dataset"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Language: $LANGUAGE_DISPLAY"
echo ""

# Step 1: Create dataset directory and check input
echo "[1/5] Checking input data..."

# Clean DATA_DIR from .env (remove trailing slashes, whitespace, and \r)
# Keep the value from .env, just clean format issues
DATA_DIR=$(echo "$DATA_DIR" | sed 's:/*$::' | tr -d '\r\n')

# Convert relative path to absolute path for find command compatibility
# This still uses the path from .env, just makes it absolute
if [[ "$DATA_DIR" == ./* ]]; then
    DATA_DIR="$PROJECT_ROOT/${DATA_DIR#./}"
elif [[ "$DATA_DIR" != /* ]]; then
    # If it's a relative path without ./, also convert to absolute
    DATA_DIR="$PROJECT_ROOT/$DATA_DIR"
fi

# Full path to dataset
DATASET_PATH="$DATA_DIR/$DATASET"
echo "Dataset path: $DATASET_PATH"

# Check if dataset directory exists
if [ ! -d "$DATASET_PATH" ]; then
    echo ""
    echo "============================================================"
    echo "[ERROR] Dataset directory not found: $DATASET_PATH"
    echo "============================================================"
    echo "Please create the directory and place your data files there."
    echo "============================================================"
    exit 1
fi

mkdir -p "$DATASET_PATH"

# Check for existing CSV file
CSV_FILE=$(ls "$DATASET_PATH"/*_cleaned.csv 2>/dev/null | head -1)
if [ -z "$CSV_FILE" ]; then
    CSV_FILE=$(ls "$DATASET_PATH"/*.csv 2>/dev/null | head -1)
fi

# If no CSV, check for raw documents and convert
if [ -z "$CSV_FILE" ]; then
    # Count raw documents recursively (docx, pdf, txt, doc, rtf, md)
    # Exclude temporary files starting with ~$
    DOC_COUNT=$(find "$DATASET_PATH" -type f \( -name "*.docx" -o -name "*.pdf" -o -name "*.txt" -o -name "*.doc" -o -name "*.rtf" -o -name "*.md" \) ! -name "~\$*" 2>/dev/null | wc -l)
    
    if [ "$DOC_COUNT" -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "[ERROR] No data found in $DATASET_PATH"
        echo "============================================================"
        echo "Please place one of the following:"
        echo "  - Raw documents: .docx, .pdf, .txt files (supports subdirectories)"
        echo "  - Pre-cleaned CSV: ${DATASET}_cleaned.csv"
        echo "============================================================"
        exit 1
    fi
    
    # Hard admission check for raw documents
    if [ "$DOC_COUNT" -lt "$MIN_DOCUMENT_COUNT" ]; then
        echo ""
        echo "============================================================"
        echo "[CRITICAL ERROR] Insufficient data sources"
        echo "============================================================"
        echo "Current files: $DOC_COUNT"
        echo "Minimum required: $MIN_DOCUMENT_COUNT"
        echo ""
        echo "THETA requires at least 5 independent documents for"
        echo "statistical significance and research diversity."
        echo "============================================================"
        exit 1
    fi
    
    echo "Found $DOC_COUNT raw documents (recursive search), converting to CSV..."
    echo ""
    
    # Step 1.5: Convert raw documents to CSV using dataclean
    echo "[1.5/5] Converting documents to CSV..."
    CSV_FILE="$DATASET_PATH/${DATASET}_cleaned.csv"
    
    # Use PYTHONPATH to avoid cd path issues
    PYTHONPATH="$ETM_DIR:$PYTHONPATH" python -m dataclean.main convert "$DATASET_PATH" "$CSV_FILE" --language $LANGUAGE_FULL --recursive
    
    echo "✓ CSV generated: $CSV_FILE"
    echo ""
fi

echo "Using CSV: $CSV_FILE"

# Hard admission check for CSV rows
ROW_COUNT=$(tail -n +2 "$CSV_FILE" | wc -l)

if [ "$ROW_COUNT" -lt "$MIN_DOCUMENT_COUNT" ]; then
    echo ""
    echo "============================================================"
    echo "[CRITICAL ERROR] Insufficient data sources"
    echo "============================================================"
    echo "Current rows: $ROW_COUNT"
    echo "Minimum required: $MIN_DOCUMENT_COUNT"
    echo ""
    echo "THETA requires at least 5 independent samples for"
    echo "statistical significance and research diversity."
    echo "============================================================"
    exit 1
fi
echo "[Admission Check] Found $ROW_COUNT rows (minimum: $MIN_DOCUMENT_COUNT) ✓"

# Step 2: Prepare data (generate embeddings and BOW)
echo ""
echo "[2/5] Preparing data (generating embeddings and BOW)..."
cd "$ETM_DIR"
python prepare_data.py --dataset $DATASET --model theta --model_size 0.6B --mode zero_shot --vocab_size 5000 --batch_size 32 --max_length 512 --gpu 0

# Step 3: Train model
echo ""
echo "[3/5] Training THETA model..."
python run_pipeline.py --dataset $DATASET --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language $LANGUAGE

# Step 4: Generate visualizations
echo ""
echo "[4/5] Generating visualizations..."
python -m visualization.run_visualization --result_dir "$RESULT_DIR" --dataset $DATASET --mode zero_shot --model_size 0.6B --language $LANGUAGE --dpi 300 || {
    echo "[WARNING] Some visualizations may have failed, but core outputs are available."
}

# Display topic words summary
LATEST_MODEL_EXP=$(ls -td "$RESULT_DIR/0.6B/$DATASET/models"/exp_* 2>/dev/null | head -1)
if [ -n "$LATEST_MODEL_EXP" ] && [ -d "$LATEST_MODEL_EXP/topic_words" ]; then
    TOPIC_WORDS_FILE=$(ls "$LATEST_MODEL_EXP/topic_words"/*.txt 2>/dev/null | head -1)
    if [ -n "$TOPIC_WORDS_FILE" ]; then
        echo ""
        echo "=========================================="
        echo "Top Topics Preview:"
        echo "=========================================="
        head -30 "$TOPIC_WORDS_FILE"
    fi
fi

# Step 5: Done
echo ""
echo "[5/5] Complete!"
echo ""
echo "=========================================="
echo "THETA Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  $RESULT_DIR/$DATASET/0.6B/theta/exp_*/"
echo ""
echo "Key outputs:"
echo "  - Config:       $RESULT_DIR/$DATASET/0.6B/theta/exp_*/config.json"
echo "  - Data:         $RESULT_DIR/$DATASET/0.6B/theta/exp_*/data/"
echo "  - Model:        $RESULT_DIR/$DATASET/0.6B/theta/exp_*/theta/"
echo "  - Metrics:      $RESULT_DIR/$DATASET/0.6B/theta/exp_*/metrics.json"
echo "  - Visualization: $RESULT_DIR/$DATASET/0.6B/theta/exp_*/{lang}/"
echo ""
echo "=========================================="
