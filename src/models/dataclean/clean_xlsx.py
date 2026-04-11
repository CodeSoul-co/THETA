#!/usr/bin/env python
"""
Script to convert xlsx to csv and then clean using CLI.

Usage:
    python clean_xlsx.py <input_xlsx> <output_csv> [options]

Example:
    python clean_xlsx.py data/EUAIACT_dataset.xlsx data/EUAIACT_cleaned.csv --language english
"""
import os
import sys
import argparse
import subprocess
import tempfile
import pandas as pd
from pathlib import Path


def xlsx_to_csv(xlsx_path: str, csv_path: str, text_column: str = None) -> str:
    """
    Convert xlsx file to csv format.
    
    Args:
        xlsx_path: Path to input xlsx file
        csv_path: Path to output csv file
        text_column: Optional column name to use as text column
        
    Returns:
        Path to the created csv file
    """
    print(f"[Step 1] Converting xlsx to csv: {xlsx_path}")
    
    df = pd.read_excel(xlsx_path)
    print(f"  - Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    
    # If text_column is specified, rename it to 'text' for CLI compatibility
    if text_column and text_column in df.columns:
        df = df.rename(columns={text_column: 'text'})
        print(f"  - Renamed column '{text_column}' to 'text'")
    
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  - Saved to: {csv_path}")
    
    return csv_path


def run_cli_clean(input_csv: str, output_csv: str, language: str = 'english', 
                  operations: list = None) -> str:
    """
    Run dataclean CLI to clean the csv file.
    
    Args:
        input_csv: Path to input csv file
        output_csv: Path to output cleaned csv file
        language: Language for NLP processing ('english' or 'chinese')
        operations: List of cleaning operations to apply
        
    Returns:
        Path to the cleaned csv file
    """
    print(f"\n[Step 2] Running CLI clean: {input_csv}")
    
    # Build command
    script_dir = Path(__file__).parent
    main_py = script_dir / "main.py"
    
    cmd = [
        sys.executable, str(main_py),
        "convert",
        input_csv,
        output_csv,
        "--language", language
    ]
    
    # Add operations if specified
    if operations:
        for op in operations:
            cmd.extend(["--operations", op])
    
    print(f"  - Command: {' '.join(cmd)}")
    
    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  - Error: {result.stderr}")
        raise RuntimeError(f"CLI clean failed: {result.stderr}")
    
    print(result.stdout)
    print(f"  - Cleaned file saved to: {output_csv}")
    
    return output_csv


def clean_xlsx(input_xlsx: str, output_csv: str, language: str = 'english',
               text_column: str = None, operations: list = None,
               keep_temp: bool = False) -> str:
    """
    Main function to convert xlsx to csv and clean.
    
    Args:
        input_xlsx: Path to input xlsx file
        output_csv: Path to output cleaned csv file
        language: Language for NLP processing
        text_column: Column name containing text to clean
        operations: List of cleaning operations
        keep_temp: Whether to keep temporary csv file
        
    Returns:
        Path to the cleaned csv file
    """
    # Create temp csv path
    if keep_temp:
        temp_csv = output_csv.replace('.csv', '_temp.csv')
    else:
        temp_dir = tempfile.gettempdir()
        temp_csv = os.path.join(temp_dir, 'temp_xlsx_to_csv.csv')
    
    try:
        # Step 1: Convert xlsx to csv
        xlsx_to_csv(input_xlsx, temp_csv, text_column)
        
        # Step 2: Run CLI clean
        run_cli_clean(temp_csv, output_csv, language, operations)
        
        return output_csv
        
    finally:
        # Clean up temp file
        if not keep_temp and os.path.exists(temp_csv):
            os.remove(temp_csv)
            print(f"\n[Cleanup] Removed temp file: {temp_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert xlsx to csv and clean using dataclean CLI'
    )
    parser.add_argument('input_xlsx', help='Path to input xlsx file')
    parser.add_argument('output_csv', help='Path to output cleaned csv file')
    parser.add_argument('--language', '-l', default='english',
                        choices=['english', 'chinese'],
                        help='Language for NLP processing')
    parser.add_argument('--text-column', '-t', default=None,
                        help='Column name containing text to clean')
    parser.add_argument('--operations', '-p', action='append',
                        choices=['remove_urls', 'remove_html_tags', 'remove_punctuation',
                                'remove_stopwords', 'normalize_whitespace', 'remove_numbers',
                                'remove_special_chars'],
                        help='Cleaning operations to apply (can be specified multiple times)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary csv file')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_xlsx):
        print(f"Error: Input file not found: {args.input_xlsx}")
        sys.exit(1)
    
    # Run cleaning
    try:
        result = clean_xlsx(
            args.input_xlsx,
            args.output_csv,
            args.language,
            args.text_column,
            args.operations,
            args.keep_temp
        )
        print(f"\n[Done] Cleaned file: {result}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
