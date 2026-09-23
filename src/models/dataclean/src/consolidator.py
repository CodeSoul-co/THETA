"""
Module for consolidating cleaned text data into CSV files.

Implements:
- Adaptive macro-dimensionality reduction: auto paragraph splitting for large docs
- Document metadata anchoring: preserves doc_name for traceability
"""
import os
import csv
import json
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# Adaptive Splitting Constants
# =============================================================================
# Threshold for auto-splitting large documents (in characters)
LARGE_DOC_THRESHOLD = 5000  # ~2500 Chinese characters or ~1000 English words
MIN_PARAGRAPH_LENGTH = 50   # Minimum paragraph length to keep


class DataConsolidator:
    """
    Class for consolidating cleaned text data into CSV files.
    """
    
    def __init__(self, large_doc_threshold: int = LARGE_DOC_THRESHOLD, 
                 min_paragraph_length: int = MIN_PARAGRAPH_LENGTH):
        """
        Initialize the DataConsolidator class.
        
        Args:
            large_doc_threshold: Character count threshold for auto-splitting
            min_paragraph_length: Minimum paragraph length to keep
        """
        self.large_doc_threshold = large_doc_threshold
        self.min_paragraph_length = min_paragraph_length
    
    def should_auto_split(self, text: str) -> bool:
        """
        Determine if a document should be auto-split based on length.
        
        Args:
            text: Document text content
            
        Returns:
            bool: True if document exceeds threshold and should be split
        """
        return len(text) > self.large_doc_threshold
    
    def adaptive_split(self, text: str, doc_name: str) -> list:
        """
        Adaptively split a large document into paragraphs while preserving doc_name anchor.
        
        This implements "宏观降维" - converting a single massive document into
        multiple manageable samples while maintaining traceability.
        
        Args:
            text: Document text content
            doc_name: Original document name for metadata anchoring
            
        Returns:
            list: List of dicts with 'text' and 'doc_name' keys
        """
        import re
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split by multiple patterns (Chinese and English paragraph markers)
        # 1. Double newlines
        # 2. Chinese paragraph markers (full-width spaces)
        # 3. After Chinese sentence endings followed by newline
        raw_paragraphs = re.split(
            r'\n\n+|\n(?=　　)|(?<=[。！？])\n|(?<=[.!?])\n(?=[A-Z])',
            text
        )
        
        # Clean and filter paragraphs
        paragraphs = []
        for p in raw_paragraphs:
            p = p.strip()
            if p and len(p) >= self.min_paragraph_length:
                paragraphs.append(p)
        
        # If no valid paragraphs found, try splitting by single newlines
        if not paragraphs:
            paragraphs = [
                p.strip() for p in text.split('\n') 
                if p.strip() and len(p.strip()) >= self.min_paragraph_length
            ]
        
        # If still no paragraphs, keep the original text as one chunk
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        
        # Create result with doc_name anchor
        result = []
        for i, para in enumerate(paragraphs):
            result.append({
                'text': para,
                'doc_name': doc_name,
                'paragraph_id': i,
                'total_paragraphs': len(paragraphs)
            })
        
        return result
    
    def create_oneline_csv(self, file_paths, output_path, text_extractor, cleaner=None):
        """
        Create a CSV file where each row represents one file with cleaned text.
        
        Args:
            file_paths (list): List of file paths to process
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            cleaner (callable, optional): Function to clean the extracted text
            
        Returns:
            str: Path to the created CSV file
        """
        data = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                # Extract text from the file
                text = text_extractor(file_path)
                
                # Clean text if cleaner is provided
                if cleaner and callable(cleaner):
                    text = cleaner(text)
                
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                
                # Create a row for this file
                row = {
                    'filename': metadata['filename'],
                    'text': text.replace('\n', ' ').replace('\r', ''),  # Ensure text is single line
                    'file_path': metadata['path'],
                    'file_size': metadata['size_bytes'],
                    'file_type': metadata['extension']
                }
                
                data.append(row)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save to CSV
        if data:
            # Get all unique keys to use as CSV headers
            headers = set()
            for row in data:
                headers.update(row.keys())
            headers = list(headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                writer.writerows(data)
            return output_path
        else:
            raise ValueError("No data to consolidate")
    
    def extract_metadata(self, file_path):
        """
        Extract metadata from a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Metadata dictionary
        """
        file_info = Path(file_path)
        
        metadata = {
            'filename': file_info.name,
            'extension': file_info.suffix,
            'size_bytes': file_info.stat().st_size,
            'created_time': file_info.stat().st_ctime,
            'modified_time': file_info.stat().st_mtime,
            'path': str(file_info.absolute())
        }
        
        return metadata
    
    def text_to_dict_list(self, text, metadata=None, split_paragraphs=False, min_paragraph_length=10):
        """
        Convert text to a list of dictionaries.
        
        Args:
            text (str): Text content
            metadata (dict, optional): Metadata to include
            split_paragraphs (bool): Whether to split text into paragraphs
            min_paragraph_length (int): Minimum length for a paragraph to be included
            
        Returns:
            list: List of dictionaries containing the text data
        """
        if split_paragraphs:
            import re
            # Split text into paragraphs using multiple patterns:
            # 1. Double newlines (\n\n)
            # 2. Single newlines followed by indentation
            # 3. Chinese paragraph markers (　　 - full-width spaces at start)
            
            # First normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Split by double newlines or single newlines (for Chinese docs)
            raw_paragraphs = re.split(r'\n\n+|\n(?=　　)|(?<=。)\n|(?<=！)\n|(?<=？)\n', text)
            
            # Clean and filter paragraphs
            paragraphs = []
            for p in raw_paragraphs:
                p = p.strip()
                # Skip empty or too short paragraphs
                if p and len(p) >= min_paragraph_length:
                    paragraphs.append(p)
            
            # If no valid paragraphs found, try splitting by single newlines
            if not paragraphs:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) >= min_paragraph_length]
            
            # Create a list of dictionaries for each paragraph
            data = []
            for i, paragraph in enumerate(paragraphs):
                row = {'paragraph_id': i, 'text': paragraph}
                
                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        row[f'meta_{key}'] = value
                
                data.append(row)
            
            return data
        else:
            # Create a single row dictionary
            data = {'text': text}
            
            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    data[f'meta_{key}'] = value
            
            return [data]
    
    def consolidate_files(self, file_paths, output_path, text_extractor, 
                          split_paragraphs=False, one_file_per_row=True,
                          auto_split_large=True, text_cleaner=None, clean_operations=None):
        """
        Consolidate multiple text files into a single CSV file.
        
        Implements adaptive macro-dimensionality reduction:
        - Small documents: kept as single rows
        - Large documents (>5000 chars): auto-split into paragraphs with doc_name anchor
        
        Args:
            file_paths (list): List of file paths to consolidate
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            split_paragraphs (bool): Whether to split text into paragraphs
            one_file_per_row (bool): Whether to ensure one file per row in output
            auto_split_large (bool): Auto-split large documents (adaptive mode)
            text_cleaner (callable, optional): Function to clean text AFTER splitting
            clean_operations (tuple, optional): Operations to pass to text_cleaner
            
        Returns:
            str: Path to the created CSV file
        """
        all_data = []
        split_stats = {'total': 0, 'auto_split': 0, 'paragraphs_created': 0}
        
        def apply_cleaning(text):
            """Apply text cleaning if cleaner is provided."""
            if text_cleaner:
                return text_cleaner(text, operations=clean_operations)
            return text
        
        for file_path in tqdm(file_paths, desc="Consolidating files"):
            try:
                # Extract RAW text from the file (no cleaning yet to preserve paragraph markers)
                text = text_extractor(file_path)
                
                # Extract metadata
                metadata = self.extract_metadata(file_path)
                doc_name = metadata['filename']
                split_stats['total'] += 1
                
                # === ADAPTIVE MACRO-DIMENSIONALITY REDUCTION ===
                # Auto-split large documents regardless of one_file_per_row setting
                if auto_split_large and self.should_auto_split(text):
                    # Large document detected - apply adaptive splitting BEFORE cleaning
                    split_results = self.adaptive_split(text, doc_name)
                    split_stats['auto_split'] += 1
                    split_stats['paragraphs_created'] += len(split_results)
                    
                    for item in split_results:
                        # Clean each paragraph AFTER splitting
                        cleaned_text = apply_cleaning(item['text'])
                        row_data = {
                            'text': cleaned_text,
                            'doc_name': item['doc_name'],
                            'paragraph_id': item['paragraph_id'],
                            'total_paragraphs': item['total_paragraphs'],
                            **{f'meta_{k}': v for k, v in metadata.items()}
                        }
                        all_data.append(row_data)
                elif one_file_per_row:
                    # Small document - keep as single row, apply cleaning
                    cleaned_text = apply_cleaning(text)
                    row_data = {
                        'text': cleaned_text,
                        'doc_name': doc_name,
                        **{f'meta_{k}': v for k, v in metadata.items()}
                    }
                    all_data.append(row_data)
                else:
                    # Use the original behavior with potential paragraph splitting
                    cleaned_text = apply_cleaning(text)
                    row_data = self.text_to_dict_list(cleaned_text, metadata, split_paragraphs)
                    for item in row_data:
                        item['doc_name'] = doc_name
                    all_data.extend(row_data)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Print adaptive splitting statistics
        if split_stats['auto_split'] > 0:
            print(f"\n[自适应宏观降维] {split_stats['auto_split']}/{split_stats['total']} 个大文档被自动切分")
            print(f"  → 生成 {split_stats['paragraphs_created']} 个段落样本")
        
        # Save to CSV
        if all_data:
            # Get all unique keys to use as CSV headers
            headers = set()
            for row in all_data:
                headers.update(row.keys())
            headers = list(headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                writer.writerows(all_data)
            
            return output_path
        else:
            raise ValueError("No data to consolidate")
    
    def consolidate_directory(self, input_dir, output_path, text_extractor, 
                              file_extensions=None, recursive=False, split_paragraphs=False,
                              one_file_per_row=True):
        """
        Consolidate all text files in a directory into a single CSV file.
        
        Args:
            input_dir (str): Directory containing files to consolidate
            output_path (str): Path to save the CSV file
            text_extractor (callable): Function to extract text from files
            file_extensions (list, optional): List of file extensions to include
            recursive (bool): Whether to search for files recursively
            split_paragraphs (bool): Whether to split text into paragraphs
            
        Returns:
            str: Path to the created CSV file
        """
        # Get all files in the directory
        if recursive:
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) 
                        for f in filenames]
        else:
            all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                        if os.path.isfile(os.path.join(input_dir, f))]
        
        # Filter by extension if specified
        if file_extensions:
            all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in file_extensions]
        
        return self.consolidate_files(all_files, output_path, text_extractor, split_paragraphs, one_file_per_row)
