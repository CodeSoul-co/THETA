#!/bin/bash
# =============================================================================
# Common Argument Parsing Helper for THETA Scripts
# =============================================================================
# This file provides a reusable pattern for argument parsing with pass-through
# support. Source this file and use the parse_args_with_passthrough function.
#
# Usage in other scripts:
#   source "$SCRIPT_DIR/_parse_args.sh"
#   
#   # Define known args as associative array
#   declare -A KNOWN_ARGS=(
#       ["--dataset"]=1      # 1 = requires value
#       ["--models"]=1
#       ["--skip-train"]=0   # 0 = flag only
#       ["-h"]=0
#       ["--help"]=0
#   )
#   
#   parse_args_with_passthrough "$@"
#   # Now PASS_THROUGH_ARGS contains unknown args
# =============================================================================

# Initialize pass-through args
PASS_THROUGH_ARGS=""

# Function to check if an argument is a known flag (no value)
is_known_flag() {
    local arg="$1"
    case "$arg" in
        -h|--help|--skip-train|--skip-eval|--skip-viz|--with-viz|--clean|--bow-only|--check-only|--force|--baseline|--no_early_stopping)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to check if an argument is a known option (requires value)
is_known_option() {
    local arg="$1"
    case "$arg" in
        --dataset|--models|--model|--model_size|--mode|--num_topics|--vocab_size|--epochs|--batch_size|--hidden_dim|--learning_rate|--gpu|--language|--data_exp|--exp_name|--max_iter|--max_topics|--n_iter|--alpha|--beta|--inference_type|--dropout|--patience|--kl_start|--kl_end|--kl_warmup|--time_column|--label_column|--raw-input|--result_dir|--dpi|--model_exp|--emb_epochs|--emb_lr|--emb_max_length|--emb_batch_size|--covariate_columns|--output_dir|--max_length)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Print pass-through args before executing Python
print_passthrough_info() {
    if [ -n "$PASS_THROUGH_ARGS" ]; then
        echo "[EXEC] Running python command with extra args:$PASS_THROUGH_ARGS"
    fi
}
