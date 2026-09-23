"""
Module for cleaning text data using NLP techniques.

NOTE: Stopwords are now managed by StopwordManager with automatic language detection.
"""
import re
import string
import os
import sys
from pathlib import Path

# Import StopwordManager
try:
    # Try relative import first
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.stopword_manager import StopwordManager
    STOPWORD_MANAGER_AVAILABLE = True
except ImportError:
    STOPWORD_MANAGER_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class TextCleaner:
    """
    Class for cleaning text data using various NLP techniques.
    Uses StopwordManager for automatic language detection and stopword loading.
    """
    
    def __init__(self, language=None):
        """
        Initialize the TextCleaner class.
        
        Args:
            language (str, optional): Language hint for NLP processing.
                                      If None, language will be auto-detected.
        """
        self.language = language
        self._detected_language = None
        self.stopword_manager = None
        self.stop_words = set()
        
        # Initialize StopwordManager if available
        if STOPWORD_MANAGER_AVAILABLE:
            self.stopword_manager = StopwordManager()
        
        # If language is explicitly specified, load stopwords immediately
        if language and self.stopword_manager:
            lang_map = {'english': 'en', 'chinese': 'zh', 'german': 'de', 
                       'spanish': 'es', 'french': 'fr', 'multi': 'en'}
            mapped_lang = lang_map.get(language, language)
            self._detected_language = mapped_lang
            self.stop_words = self.stopword_manager.load_stopwords(mapped_lang)
        elif language == 'chinese' and JIEBA_AVAILABLE:
            jieba.initialize()
    
    def detect_language_from_batch(self, texts):
        """
        从批量文本中检测语言（支持多语言混合数据集）
        
        Args:
            texts: 文本列表
        """
        if self.stopword_manager and not self._detected_language:
            self._detected_language = self.stopword_manager.detect_language_from_documents(texts)
            self.stop_words = self.stopword_manager.load_stopwords()
            print(f"[TextCleaner] Primary language: {self._detected_language}")
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_emojis(self, text):
        """
        Remove all emojis from text using the emoji library.
        
        This is more reliable than regex as it uses the official Unicode emoji list.
        """
        try:
            import emoji
            return emoji.replace_emoji(text, replace='')
        except ImportError:
            # Fallback to regex if emoji library not installed
            emoji_pattern = re.compile(
                '['
                '\U0001F600-\U0001F64F'  # emoticons
                '\U0001F300-\U0001F5FF'  # symbols & pictographs
                '\U0001F680-\U0001F6FF'  # transport & map
                '\U0001F1E0-\U0001F1FF'  # flags
                '\U0001F900-\U0001F9FF'  # supplemental symbols
                '\U0001FA00-\U0001FAFF'  # extended symbols
                '\U00002600-\U000026FF'  # misc symbols
                '\U00002700-\U000027BF'  # dingbats
                ']+', flags=re.UNICODE
            )
            return emoji_pattern.sub(r'', text)
    
    def _auto_detect_and_load_stopwords(self, text):
        """Auto-detect language and load stopwords if not already done."""
        if self.stopword_manager and not self.stop_words:
            self._detected_language = self.stopword_manager.detect_language(text)
            self.stop_words = self.stopword_manager.load_stopwords(self._detected_language)
            print(f"[TextCleaner] Auto-detected language: {self._detected_language}")
    
    def tokenize_text(self, text):
        """Tokenize text based on detected or specified language."""
        # Auto-detect language if not specified
        if not self._detected_language and self.stopword_manager:
            self._auto_detect_and_load_stopwords(text)
        
        # Use StopwordManager's tokenization if available
        if self.stopword_manager and self._detected_language:
            return self.stopword_manager.tokenize(text)
        
        # Fallback tokenization
        if self._detected_language == 'zh' or self.language == 'chinese':
            if JIEBA_AVAILABLE:
                return list(jieba.cut(text))
        
        # Simple word tokenization for other languages
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        # Auto-detect language if not done
        if not self.stop_words and self.stopword_manager:
            self._auto_detect_and_load_stopwords(text)
        
        words = self.tokenize_text(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """Apply stemming to text."""
        # Simple stemming implementation (just returns the text as is)
        # For a real implementation, you would use a stemming algorithm
        return text
    
    def lemmatize_text(self, text):
        """Apply lemmatization to text."""
        # Simple lemmatization implementation (just returns the text as is)
        # For a real implementation, you would use a lemmatization algorithm
        return text
    
    def normalize_whitespace(self, text):
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_numbers(self, text):
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def remove_special_chars(self, text):
        """Remove special characters from text."""
        return re.sub(r'[^\w\s]', '', text)
    
    def named_entity_recognition(self, text):
        """
        Perform Named Entity Recognition on text.
        
        Returns:
            dict: Dictionary with text and entities
        """
        # Simple implementation that just returns the text without entities
        # For a real implementation, you would use an NER model
        return {
            'text': text,
            'entities': []
        }
    
    def clean_text(self, text, operations=None):
        """
        Clean text using specified operations.
        
        Args:
            text (str): Text to clean
            operations (list, optional): List of cleaning operations to apply.
                                         If None, applies all basic operations.
                                         
        Returns:
            str: Cleaned text
        """
        if operations is None:
            operations = [
                'remove_urls',
                'remove_html_tags',
                'remove_emojis',
                'remove_punctuation',
                'remove_stopwords',
                'normalize_whitespace'
            ]
        
        for operation in operations:
            if hasattr(self, operation) and callable(getattr(self, operation)):
                text = getattr(self, operation)(text)
        
        return text
    
    def create_chinese_stopwords_file(self, file_path=None):
        """
        DEPRECATED: Stopwords are now managed by StopwordManager.
        Use resources/stopwords/zh.txt instead.
        
        This method is kept for backward compatibility but does nothing.
        """
        print("[WARNING] create_chinese_stopwords_file is deprecated. "
              "Stopwords are now managed by StopwordManager in resources/stopwords/")
        return None
