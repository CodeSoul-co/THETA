#!/bin/bash
# =============================================================================
# THETA Environment Setup Script
# =============================================================================
# This script is sourced by all other scripts to set up environment variables.
# It automatically detects PROJECT_ROOT from the script location and loads .env
#
# Usage (in other scripts):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/env_setup.sh"
#
# WARNING: If you modify .env variables, you MUST unset the old values first!
#   Otherwise the shell will keep using the cached values.
#   
#   Option 1: Unset all THETA-related variables before re-sourcing:
#     unset QWEN_MODEL_0_6B QWEN_MODEL_4B QWEN_MODEL_8B SBERT_MODEL_PATH
#     unset DATA_DIR WORKSPACE_DIR RESULT_DIR PROJECT_ROOT
#     source scripts/env_setup.sh
#
#   Option 2: Start a new shell session
# =============================================================================

# Detect PROJECT_ROOT from script location
# This works regardless of where the script is called from
if [ -z "$PROJECT_ROOT" ]; then
    # Get the directory where this script is located
    ENV_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # PROJECT_ROOT is the parent of scripts/
    export PROJECT_ROOT="$(cd "$ENV_SETUP_DIR/.." && pwd)"
fi

# Load .env file if it exists
# Priority: external export > .env > auto-detected defaults
# Only set variables from .env if they are NOT already set in the environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | xargs)
        # Skip if key is empty after trimming
        [[ -z "$key" ]] && continue
        # Only set if not already defined in environment
        if [ -z "${!key+x}" ]; then
            # Remove surrounding quotes from value if present
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
            export "$key=$value"
        fi
    done < "$PROJECT_ROOT/.env"
fi

# =============================================================================
# Core Directory Variables (with defaults)
# =============================================================================

# Source directory (contains models and embedding)
export SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/src}"

# Models module directory (formerly ETM)
export MODELS_DIR="${MODELS_DIR:-$SRC_DIR/models}"

# ETM_DIR alias for backward compatibility with existing scripts
export ETM_DIR="${ETM_DIR:-$MODELS_DIR}"

# Agent module directory
export AGENT_DIR="${AGENT_DIR:-$PROJECT_ROOT/agent}"

# Scripts directory
export SCRIPTS_DIR="${SCRIPTS_DIR:-$PROJECT_ROOT/scripts}"

# =============================================================================
# Data Directory Variables
# =============================================================================

# Data directory (base for all data-related files)
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

# Workspace directory (for user data, shared matrices, etc.)
# Now located under data/ for better organization
export WORKSPACE_DIR="${WORKSPACE_DIR:-$DATA_DIR/workspace}"

# Raw data directory
export RAW_DATA_DIR="${RAW_DATA_DIR:-$DATA_DIR/raw_data}"

# =============================================================================
# Output Directory Variables
# =============================================================================

# Result directory (model outputs, embeddings, BOW, etc.)
export RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/result}"

# HuggingFace cache directory
export HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"

# =============================================================================
# Model Directory Variables
# =============================================================================

# Base directory for embedding models (Qwen, SBERT, etc.)
export EMBEDDING_MODELS_DIR="${EMBEDDING_MODELS_DIR:-$PROJECT_ROOT/models}"

# Qwen model paths (based on model size)
export QWEN_MODEL_0_6B="${QWEN_MODEL_0_6B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_0.6B}"
export QWEN_MODEL_4B="${QWEN_MODEL_4B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_4B}"
export QWEN_MODEL_8B="${QWEN_MODEL_8B:-$EMBEDDING_MODELS_DIR/qwen3_embedding_8B}"

# SBERT model path
export SBERT_MODEL_PATH="${SBERT_MODEL_PATH:-$PROJECT_ROOT/models/sbert/sentence-transformers/all-MiniLM-L6-v2}"

# =============================================================================
# Helper Functions
# =============================================================================

# Get Qwen model path based on model size
get_qwen_model_path() {
    local model_size="$1"
    case "$model_size" in
        0.6B) echo "$QWEN_MODEL_0_6B" ;;
        4B)   echo "$QWEN_MODEL_4B" ;;
        8B)   echo "$QWEN_MODEL_8B" ;;
        *)    echo "$QWEN_MODEL_0_6B" ;;  # Default to 0.6B
    esac
}

# Get result directory for a specific model type
get_result_dir() {
    local model_type="$1"  # "theta" or "baseline"
    local model_size="$2"  # e.g., "0.6B" (only for theta)
    
    if [ "$model_type" = "theta" ]; then
        echo "$RESULT_DIR/${model_size:-0.6B}"
    else
        echo "$RESULT_DIR/baseline"
    fi
}

# Print environment info (for debugging)
print_env_info() {
    echo "=========================================="
    echo "THETA Environment Configuration"
    echo "=========================================="
    echo "PROJECT_ROOT:         $PROJECT_ROOT"
    echo "SRC_DIR:              $SRC_DIR"
    echo "MODELS_DIR:           $MODELS_DIR"
    echo "AGENT_DIR:            $AGENT_DIR"
    echo "DATA_DIR:             $DATA_DIR"
    echo "WORKSPACE_DIR:        $WORKSPACE_DIR"
    echo "RESULT_DIR:           $RESULT_DIR"
    echo "HF_CACHE_DIR:         $HF_CACHE_DIR"
    echo "EMBEDDING_MODELS_DIR: $EMBEDDING_MODELS_DIR"
    echo "=========================================="
}

# =============================================================================
# Model Download Functions
# =============================================================================
# Download links (from README.md):
#   - Qwen3-Embedding-0.6B: https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B
#   - all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Check if a model directory contains valid model files
_is_model_valid() {
    local model_path="$1"
    local model_type="$2"  # "sbert" or "qwen"
    
    [ ! -d "$model_path" ] && return 1
    
    if [ "$model_type" = "sbert" ]; then
        # SBERT needs config.json and model files
        [ -f "$model_path/config.json" ] && \
        { [ -f "$model_path/model.safetensors" ] || [ -f "$model_path/pytorch_model.bin" ]; }
    elif [ "$model_type" = "qwen" ]; then
        # Qwen needs config.json and model files
        [ -f "$model_path/config.json" ] && \
        { [ -f "$model_path/model.safetensors" ] || [ -f "$model_path/pytorch_model.bin" ] || \
          ls "$model_path"/model*.safetensors >/dev/null 2>&1; }
    else
        return 1
    fi
}

# Download SBERT model (all-MiniLM-L6-v2)
download_sbert_model() {
    local target_path="${1:-$SBERT_MODEL_PATH}"
    
    # Convert to absolute path if relative
    if [[ "$target_path" != /* ]]; then
        target_path="$PROJECT_ROOT/$target_path"
    fi
    
    if _is_model_valid "$target_path" "sbert"; then
        [ -z "$THETA_QUIET" ] && echo "[INFO] SBERT model already exists: $target_path"
        return 0
    fi
    
    echo "=========================================="
    echo "[INFO] SBERT model not found at: $target_path"
    echo "[INFO] Downloading all-MiniLM-L6-v2 from HuggingFace..."
    echo "=========================================="
    
    # Create directory
    mkdir -p "$target_path"
    
    # Try to download using Python (use 'python' to respect conda env)
    python -c "
import sys
import os

target_path = '$target_path'

try:
    from sentence_transformers import SentenceTransformer
    
    print('Downloading sentence-transformers/all-MiniLM-L6-v2...')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Remove .gitkeep if exists
    gitkeep = os.path.join(target_path, '.gitkeep')
    if os.path.exists(gitkeep):
        os.remove(gitkeep)
    
    model.save(target_path)
    print(f'[SUCCESS] SBERT model saved to: {target_path}')
    
    # Verify
    test_model = SentenceTransformer(target_path)
    emb = test_model.encode(['test'])
    print(f'[SUCCESS] Model verified, embedding dim: {emb.shape[1]}')
    sys.exit(0)
    
except Exception as e:
    print(f'[ERROR] Failed to download SBERT model: {e}')
    print('[INFO] Please manually download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2')
    sys.exit(1)
"
    return $?
}

# Download Qwen embedding model
download_qwen_model() {
    local model_size="${1:-0.6B}"
    local target_path
    
    case "$model_size" in
        0.6B) target_path="$QWEN_MODEL_0_6B" ;;
        4B)   target_path="$QWEN_MODEL_4B" ;;
        8B)   target_path="$QWEN_MODEL_8B" ;;
        *)    target_path="$QWEN_MODEL_0_6B"; model_size="0.6B" ;;
    esac
    
    # Convert to absolute path if relative
    if [[ "$target_path" != /* ]]; then
        target_path="$PROJECT_ROOT/$target_path"
    fi
    
    if _is_model_valid "$target_path" "qwen"; then
        [ -z "$THETA_QUIET" ] && echo "[INFO] Qwen model already exists: $target_path"
        return 0
    fi
    
    echo "=========================================="
    echo "[INFO] Qwen model not found at: $target_path"
    echo "[INFO] Downloading Qwen3-Embedding-${model_size} from ModelScope..."
    echo "=========================================="
    
    # Create directory
    mkdir -p "$target_path"
    
    # Try to download using Python (use 'python' to respect conda env)
    python -c "
import sys
import os

model_size = '$model_size'
target_path = '$target_path'

# Map model size to ModelScope model ID
model_map = {
    '0.6B': 'Qwen/Qwen3-Embedding-0.6B',
    '4B': 'Qwen/Qwen3-Embedding-4B',
    '8B': 'Qwen/Qwen3-Embedding-8B'
}

model_id = model_map.get(model_size, model_map['0.6B'])

try:
    # Try ModelScope first (better for China users)
    try:
        from modelscope import snapshot_download
        print(f'Downloading {model_id} from ModelScope...')
        snapshot_download(model_id, local_dir=target_path)
        print(f'[SUCCESS] Qwen model saved to: {target_path}')
        sys.exit(0)
    except ImportError:
        print('[INFO] ModelScope not installed, trying HuggingFace...')
    
    # Fallback to HuggingFace
    from huggingface_hub import snapshot_download
    hf_model_id = model_id  # Same ID works for HuggingFace
    print(f'Downloading {hf_model_id} from HuggingFace...')
    snapshot_download(repo_id=hf_model_id, local_dir=target_path)
    print(f'[SUCCESS] Qwen model saved to: {target_path}')
    sys.exit(0)
    
except Exception as e:
    print(f'[ERROR] Failed to download Qwen model: {e}')
    print(f'[INFO] Please manually download from: https://www.modelscope.cn/models/{model_id}')
    sys.exit(1)
"
    return $?
}

# Ensure required models are available (auto-download if missing)
# Usage: ensure_models [model_type...]
#   model_type: "sbert", "qwen", "qwen:0.6B", "qwen:4B", "qwen:8B", "all"
ensure_models() {
    local models=("$@")
    local failed=0
    
    # Default to checking based on common usage
    if [ ${#models[@]} -eq 0 ]; then
        models=("sbert")
    fi
    
    for model in "${models[@]}"; do
        case "$model" in
            sbert)
                download_sbert_model || failed=1
                ;;
            qwen|qwen:0.6B)
                download_qwen_model "0.6B" || failed=1
                ;;
            qwen:4B)
                download_qwen_model "4B" || failed=1
                ;;
            qwen:8B)
                download_qwen_model "8B" || failed=1
                ;;
            all)
                download_sbert_model || failed=1
                download_qwen_model "0.6B" || failed=1
                ;;
            *)
                echo "[WARN] Unknown model type: $model"
                ;;
        esac
    done
    
    return $failed
}

# Always print env info when sourced (can be disabled with THETA_QUIET=1)
if [ -z "$THETA_QUIET" ]; then
    print_env_info
fi

# Export helper functions
export -f get_qwen_model_path
export -f get_result_dir
export -f print_env_info
export -f _is_model_valid
export -f download_sbert_model
export -f download_qwen_model
export -f ensure_models
