#!/bin/bash
# =============================================================================
# THETA Quick Start - Chinese Dataset (True One-Click)
# =============================================================================
# Complete workflow: Raw Data → Clean → Embed → Train → Evaluate → Visualize
#
# Supports two input modes:
#   1. Directory with docx/pdf/txt files → Auto-convert to CSV
#   2. Pre-cleaned CSV file → Direct processing
#
# Usage:
#   ./quick_start_chinese.sh <dataset_name>
#
# Example:
#   ./quick_start_chinese.sh my_policy_docs
# =============================================================================

set -e

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"

DATASET=${1:-"my_chinese_dataset"}
MIN_DOCUMENT_COUNT=5

echo "=========================================="
echo "THETA Quick Start - Chinese Dataset"
echo "=========================================="
echo "Dataset: $DATASET"
echo ""

# Step 1: Create dataset directory and check input
echo "[1/5] Checking input data..."
mkdir -p "$DATA_DIR/$DATASET"

# Check for existing CSV file
CSV_FILE=$(ls "$DATA_DIR/$DATASET"/*_cleaned.csv 2>/dev/null | head -1)
if [ -z "$CSV_FILE" ]; then
    CSV_FILE=$(ls "$DATA_DIR/$DATASET"/*.csv 2>/dev/null | head -1)
fi

# If no CSV, check for raw documents and convert
if [ -z "$CSV_FILE" ]; then
    # Count raw documents (docx, pdf, txt, doc, rtf, md)
    DOC_COUNT=$(find "$DATA_DIR/$DATASET" -type f \( -name "*.docx" -o -name "*.pdf" -o -name "*.txt" -o -name "*.doc" -o -name "*.rtf" -o -name "*.md" \) 2>/dev/null | wc -l)
    
    if [ "$DOC_COUNT" -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "[ERROR] No data found in $DATA_DIR/$DATASET/"
        echo "============================================================"
        echo "Please place one of the following:"
        echo "  - Raw documents: .docx, .pdf, .txt files"
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
    
    echo "Found $DOC_COUNT raw documents, converting to CSV..."
    echo ""
    
    # Step 1.5: Convert raw documents to CSV using dataclean
    echo "[1.5/5] Converting documents to CSV..."
    CSV_FILE="$DATA_DIR/$DATASET/${DATASET}_cleaned.csv"
    
    # Use PYTHONPATH to avoid cd path issues
    PYTHONPATH="$ETM_DIR:$PYTHONPATH" python -m dataclean.main convert "$DATA_DIR/$DATASET" "$CSV_FILE" --language chinese --recursive
    
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
python run_pipeline.py --dataset $DATASET --models theta --model_size 0.6B --mode zero_shot --num_topics 20 --epochs 100 --batch_size 64 --hidden_dim 512 --learning_rate 0.002 --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 --patience 10 --gpu 0 --language zh

# Step 4: Generate visualizations
echo ""
echo "[4/5] Generating visualizations..."
python -m visualization.run_visualization --result_dir "$RESULT_DIR" --dataset $DATASET --mode zero_shot --model_size 0.6B --language zh --dpi 300 || {
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
echo "  $RESULT_DIR/0.6B/$DATASET/"
echo ""
echo "Key outputs:"
echo "  - Model:        $RESULT_DIR/0.6B/$DATASET/models/exp_*/model/"
echo "  - Embeddings:   $RESULT_DIR/0.6B/$DATASET/data/exp_*/embeddings/"
echo "  - BOW:          $RESULT_DIR/0.6B/$DATASET/data/exp_*/bow/"
echo "  - Evaluation:   $RESULT_DIR/0.6B/$DATASET/models/exp_*/evaluation/"
echo "  - Visualization: $RESULT_DIR/0.6B/$DATASET/models/exp_*/visualization/"
echo ""
echo "=========================================="
