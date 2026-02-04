#!/bin/bash
# THETA Data Cleaning Script
# Clean raw data for topic modeling

set -e

# Default values
LANGUAGE="english"
INPUT_FILE=""
OUTPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT_FILE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --input <input_file> --output <output_file> [--language english|chinese]"
            echo ""
            echo "Examples:"
            echo "  $0 --input /root/autodl-tmp/data/mydata/raw.csv --output /root/autodl-tmp/data/mydata/mydata_cleaned.csv --language english"
            echo "  $0 --input /root/autodl-tmp/data/chinese_data/raw.csv --output /root/autodl-tmp/data/chinese_data/chinese_data_cleaned.csv --language chinese"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: --input and --output are required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

echo "=========================================="
echo "THETA Data Cleaning"
echo "=========================================="
echo "Input:    $INPUT_FILE"
echo "Output:   $OUTPUT_FILE"
echo "Language: $LANGUAGE"
echo ""

cd /root/autodl-tmp/ETM
python -m dataclean.main --input "$INPUT_FILE" --output "$OUTPUT_FILE" --language "$LANGUAGE"

echo ""
echo "Data cleaning completed!"
echo "Output saved to: $OUTPUT_FILE"
