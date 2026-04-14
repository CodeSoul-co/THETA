"""
Global Vocabulary Builder for Engine A.

Builds a unified vocabulary across all datasets to ensure:
- Topic comparability across datasets
- Consistent word-to-index mapping
- Shared semantic measurement scale
"""

import os
import json
import re
from collections import Counter
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# Import StopwordManager for language detection and stopword loading
try:
    from utils.stopword_manager import StopwordManager
    STOPWORD_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from ..utils.stopword_manager import StopwordManager
        STOPWORD_MANAGER_AVAILABLE = True
    except ImportError:
        STOPWORD_MANAGER_AVAILABLE = False


@dataclass
class VocabConfig:
    """Configuration for vocabulary building"""
    min_df: int = 10                   # Minimum document frequency
    max_df_ratio: float = 0.5          # Maximum document frequency ratio
    max_vocab_size: int = 50000        # Maximum vocabulary size
    min_word_length: int = 3           # Minimum word length (for English)
    min_chinese_length: int = 1        # Minimum word length (for Chinese)
    max_word_length: int = 50          # Maximum word length
    lowercase: bool = True             # Convert to lowercase
    remove_numbers: bool = True        # Remove pure numbers
    remove_stopwords: bool = True      # Remove stopwords
    language: str = "multi"            # Language: 'en', 'de', 'zh', 'multi'
    # Language-specific vocab limits to prevent imbalance
    en_vocab_limit: int = 10000        # Max English words
    de_vocab_limit: int = 3000         # Max German words
    zh_vocab_limit: int = 10000        # Max Chinese words
    balanced_sampling: bool = True     # Balance vocab across languages


@dataclass
class VocabStats:
    """Statistics about the built vocabulary"""
    total_documents: int
    total_tokens: int
    unique_tokens_before_filter: int
    vocab_size: int
    datasets_included: List[str]
    config: Dict


class VocabBuilder:
    """
    Builds a global vocabulary from multiple datasets.
    
    The vocabulary serves as a unified "language measurement scale" 
    for all downstream ETM models.
    """
    
    # Stopwords are now loaded dynamically via StopwordManager
    # These class-level sets are kept as fallback only
    _FALLBACK_STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    
    def __init__(self, config: Optional[VocabConfig] = None, dev_mode: bool = False):
        """
        Initialize vocabulary builder.
        
        Args:
            config: Vocabulary configuration
            dev_mode: Print debug information
        """
        self.config = config or VocabConfig()
        self.dev_mode = dev_mode
        
        # Initialize stopword manager (will be populated when documents are added)
        self.stopword_manager = None
        self.stopwords = set()
        self._detected_language = None
        
        # Vocabulary data
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.doc_counts: Counter = Counter()  # Document frequency
        self.total_docs: int = 0
        self.datasets_included: List[str] = []
        
        if self.dev_mode:
            print(f"[DEV] VocabBuilder initialized with config: {asdict(self.config)}")
    
    def _is_chinese(self, char: str) -> bool:
        """Check if a character is Chinese."""
        return '\u4e00' <= char <= '\u9fff'
    
    def _has_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any(self._is_chinese(c) for c in text)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words. Uses StopwordManager for language-aware tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.config.lowercase:
            text = text.lower()
        
        tokens = []
        
        # Use StopwordManager's tokenization if available and language detected
        if self.stopword_manager is not None and self._detected_language is not None:
            tokens = self.stopword_manager.tokenize(text)
        else:
            # Fallback tokenization
            # Check if text contains Chinese
            if self._has_chinese(text):
                if JIEBA_AVAILABLE:
                    chinese_tokens = list(jieba.cut(text))
                    tokens.extend(chinese_tokens)
                else:
                    for char in text:
                        if self._is_chinese(char):
                            tokens.append(char)
            
            # Also extract English/German words
            english_tokens = re.findall(r'\b[a-zA-Z0-9äöüßÄÖÜ]+\b', text)
            tokens.extend(english_tokens)
        
        # Filter tokens
        filtered = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Check if token is Chinese
            is_chinese_token = self._has_chinese(token)
            
            # Length filter (different for Chinese and English)
            if is_chinese_token:
                if len(token) < self.config.min_chinese_length:
                    continue
            else:
                if len(token) < self.config.min_word_length:
                    continue
            
            if len(token) > self.config.max_word_length:
                continue
            
            # Number filter
            if self.config.remove_numbers and token.isdigit():
                continue
            
            # Stopword filter
            if self.config.remove_stopwords and token.lower() in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def add_documents(
        self,
        texts: List[str],
        dataset_name: str,
        show_progress: bool = True
    ) -> None:
        """
        Add documents from a dataset to build vocabulary.
        
        Args:
            texts: List of text documents
            dataset_name: Name of the dataset
            show_progress: Show progress bar
        """
        if self.dev_mode:
            print(f"[DEV] Adding {len(texts)} documents from {dataset_name}")
        
        # Auto-detect language and load stopwords on first call
        if self.stopword_manager is None and STOPWORD_MANAGER_AVAILABLE:
            self.stopword_manager = StopwordManager()
            # Detect language from first batch of documents
            self._detected_language = self.stopword_manager.detect_language_from_documents(texts)
            self.stopwords = self.stopword_manager.load_stopwords(self._detected_language)
            print(f"[VocabBuilder] Auto-detected language: {self._detected_language}, loaded {len(self.stopwords)} stopwords")
        elif self.stopword_manager is None:
            # Fallback if StopwordManager not available
            self.stopwords = self._FALLBACK_STOPWORDS.copy()
            print(f"[VocabBuilder] Using fallback stopwords ({len(self.stopwords)} words)")
        
        self.datasets_included.append(dataset_name)
        
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc=f"Tokenizing {dataset_name}")
        
        for text in iterator:
            tokens = self._tokenize(text)
            
            # Update word counts
            self.word_counts.update(tokens)
            
            # Update document frequency (unique words per doc)
            unique_tokens = set(tokens)
            self.doc_counts.update(unique_tokens)
            
            self.total_docs += 1
        
        if self.dev_mode:
            print(f"[DEV] Total documents: {self.total_docs}")
            print(f"[DEV] Unique tokens: {len(self.word_counts)}")
    
    def build_vocab(self) -> VocabStats:
        """
        Build the final vocabulary after adding all documents.
        
        Returns:
            VocabStats with vocabulary statistics
        """
        if self.dev_mode:
            print(f"[DEV] Building vocabulary from {len(self.word_counts)} unique tokens")
        
        unique_before = len(self.word_counts)
        
        # Filter by document frequency
        min_df = self.config.min_df
        max_df = int(self.total_docs * self.config.max_df_ratio)
        
        filtered_words = []
        for word, count in self.word_counts.items():
            doc_freq = self.doc_counts[word]
            if doc_freq >= min_df and doc_freq <= max_df:
                filtered_words.append((word, count))
        
        if self.dev_mode:
            print(f"[DEV] After DF filtering: {len(filtered_words)} words")
            print(f"[DEV] min_df={min_df}, max_df={max_df}")
        
        # Sort by frequency and take top N
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        if len(filtered_words) > self.config.max_vocab_size:
            filtered_words = filtered_words[:self.config.max_vocab_size]
        
        # Build word2idx and idx2word
        self.word2idx = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Calculate total tokens
        total_tokens = sum(self.word_counts.values())
        
        stats = VocabStats(
            total_documents=self.total_docs,
            total_tokens=total_tokens,
            unique_tokens_before_filter=unique_before,
            vocab_size=len(self.word2idx),
            datasets_included=self.datasets_included.copy(),
            config=asdict(self.config)
        )
        
        print(f"Vocabulary built: {stats.vocab_size} words from {stats.total_documents} documents")
        
        return stats
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.word2idx)
    
    def get_word2idx(self) -> Dict[str, int]:
        """Return word to index mapping"""
        return self.word2idx.copy()
    
    def get_idx2word(self) -> Dict[int, str]:
        """Return index to word mapping"""
        return self.idx2word.copy()
    
    def get_vocab_list(self) -> List[str]:
        """Return vocabulary as ordered list"""
        return [self.idx2word[i] for i in range(len(self.idx2word))]
    
    def save(self, output_dir: str, prefix: str = "global") -> Dict[str, str]:
        """
        Save vocabulary to files.
        
        Args:
            output_dir: Output directory
            prefix: File name prefix
            
        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save word2idx as JSON
        vocab_path = os.path.join(output_dir, f"{prefix}_vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        
        # Save vocabulary list
        vocab_list_path = os.path.join(output_dir, f"{prefix}_vocab_list.json")
        with open(vocab_list_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab_list(), f, ensure_ascii=False)
        
        # Save statistics
        stats = VocabStats(
            total_documents=self.total_docs,
            total_tokens=sum(self.word_counts.values()),
            unique_tokens_before_filter=len(self.word_counts),
            vocab_size=len(self.word2idx),
            datasets_included=self.datasets_included,
            config=asdict(self.config)
        )
        stats_path = os.path.join(output_dir, f"{prefix}_vocab_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2)
        
        paths = {
            "vocab": vocab_path,
            "vocab_list": vocab_list_path,
            "stats": stats_path
        }
        
        if self.dev_mode:
            print(f"[DEV] Saved vocabulary files:")
            for key, path in paths.items():
                print(f"[DEV]   {key}: {path}")
        
        return paths
    
    @classmethod
    def load(cls, vocab_path: str, dev_mode: bool = False) -> 'VocabBuilder':
        """
        Load vocabulary from file.
        
        Args:
            vocab_path: Path to vocab JSON file
            dev_mode: Print debug information
            
        Returns:
            VocabBuilder instance with loaded vocabulary
        """
        builder = cls(dev_mode=dev_mode)
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            builder.word2idx = json.load(f)
        
        builder.idx2word = {int(idx): word for word, idx in builder.word2idx.items()}
        
        if dev_mode:
            print(f"[DEV] Loaded vocabulary with {len(builder.word2idx)} words")
        
        return builder
