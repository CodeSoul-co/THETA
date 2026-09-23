"""
Stopword Manager - 语种检测与停用词管理

功能：
1. 自动检测文本语种
2. 根据语种加载对应停用词文件
3. 强制合并 common.txt 通用停用词
4. 提供分词策略（中文用 jieba，拉丁语系用正则）
"""

import os
import re
import unicodedata
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Set, List, Optional, Tuple


_custom_stopwords = ContextVar('theta_custom_stopwords', default=None)


def normalize_word(word: str) -> str:
    return unicodedata.normalize('NFKC', word).strip().casefold()


def configure_stopwords(text: Optional[str] = None):
    """Job-local override; None uses bundled lists, custom text replaces them."""
    words = None if text is None else frozenset(normalize_word(line) for line in text.splitlines()
                                               if line.strip() and not line.lstrip().startswith('#'))
    return _custom_stopwords.set(words)


class StopwordManager:
    """停用词管理器 - 自动检测语种并加载对应停用词"""
    
    LANG_MAP = {
        'zh-cn': 'zh',
        'zh-tw': 'zh',
        'zh': 'zh',
        'en': 'en',
        'de': 'de',
        'fr': 'fr',
        'es': 'es',
        'it': 'it',
        'pt': 'pt',
        'ru': 'ru',
        'ja': 'ja',
        'ko': 'ko',
    }
    
    CJK_LANGS = {'zh', 'ja', 'ko'}

    @staticmethod
    def _script_language(text: str) -> Optional[str]:
        """Prefer Unicode script evidence over statistical guesses for CJK text.

        Cleaned Chinese corpora are often already segmented with spaces. Generic
        language detectors can then misclassify them as Korean even though they
        contain no Hangul. Script ranges are deterministic for this case.
        """
        han = len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]', text))
        kana = len(re.findall(r'[\u3040-\u30ff]', text))
        hangul = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))
        if kana >= 2:
            return 'ja'
        if hangul >= 2:
            return 'ko'
        if han >= 4:
            return 'zh'
        return None
    
    def __init__(self, stopwords_dir: Optional[str] = None):
        """
        初始化停用词管理器
        
        Args:
            stopwords_dir: 停用词目录路径，默认为 resources/stopwords/
        """
        if stopwords_dir is None:
            current_dir = Path(__file__).parent
            stopwords_dir = current_dir.parent / 'resources' / 'stopwords'
        
        self.stopwords_dir = Path(stopwords_dir)
        self._stopwords: Set[str] = set()
        self._detected_lang: Optional[str] = None
        self._lang_code: Optional[str] = None
        self._lang_distribution = {}
        self._loaded_languages = set()
        self._custom = _custom_stopwords.get()

    @classmethod
    @lru_cache(maxsize=512)
    def _languages_in_text(cls, text: str) -> frozenset[str]:
        """Keep each script present in a mixed document, not only its majority."""
        languages = set()
        if re.search(r'[\u3040-\u30ff]', text): languages.add('ja')
        elif re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text): languages.add('zh')
        if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text): languages.add('ko')
        if re.search(r'[\u0400-\u052f]', text): languages.add('ru')
        # Detect the Latin portion separately: Chinese text must not hide English.
        segments = re.findall(r'[a-zA-ZÀ-ɏ]+(?:[\s\u0027’-]+[a-zA-ZÀ-ɏ]+)*', text)
        latin = ' '.join(segments)
        if latin.strip():
            try:
                from langdetect import detect, DetectorFactory
                DetectorFactory.seed = 0
                # Sentence/script boundaries also capture code-switching between
                # two Latin-script languages within the same document.
                samples = {latin, *(part for part in segments if len(part.split()) >= 3)}
                for sample in sorted(samples):
                    code = detect(sample)
                    languages.add(cls.LANG_MAP.get(code, code))
            except Exception:
                languages.add('en')
        return frozenset(languages or {'en'})
        
    def detect_language(self, text: str, sample_size: int = 1000) -> str:
        """
        检测文本语种
        
        Args:
            text: 输入文本
            sample_size: 采样字符数（默认 1000）
            
        Returns:
            语种代码 (zh, en, de, fr, etc.)
        """
        sample = text[:sample_size]
        languages = self._languages_in_text(sample)
        self._lang_distribution = {lang: 1 / len(languages) for lang in languages}
        self._detected_lang = self._script_language(sample) or sorted(languages)[0]
        self._lang_code = self._detected_lang
        return self._detected_lang
    
    def detect_language_from_documents(self, documents: List[str], sample_size: int = 100) -> str:
        """
        从多个文档中检测语种（支持多语言混合数据集）
        
        采用均匀采样策略：从数据集的前、中、后段各采样，统计语言分布，
        返回占比最高的语言。如果是多语言混合，加载所有检测到的语言的停用词。
        
        Args:
            documents: 文档列表
            sample_size: 采样文档数（默认100，均匀分布在整个数据集）
            
        Returns:
            主要语种代码
        """
        from collections import Counter
        
        n_docs = len(documents)
        if n_docs == 0:
            return self.detect_language('')
        
        actual_sample_size = min(sample_size, n_docs)
        
        if n_docs <= actual_sample_size:
            sample_indices = list(range(n_docs))
        else:
            step = n_docs // actual_sample_size
            sample_indices = list(range(0, n_docs, step))[:actual_sample_size]
        
        lang_counts = Counter()
        for idx in sample_indices:
            try:
                text = str(documents[idx])[:500]
                if text.strip():
                    for lang in sorted(self._languages_in_text(text)):
                        lang_counts[lang] += 1
            except:
                continue
        
        if not lang_counts:
            return self.detect_language('')
        
        total = sum(lang_counts.values())
        self._lang_distribution = {lang: count/total for lang, count in lang_counts.most_common()}
        
        print(f"[StopwordManager] Language distribution (sampled {len(sample_indices)} docs):")
        for lang, count in lang_counts.most_common(5):
            print(f"  {lang}: {count} ({count/total*100:.1f}%)")
        
        primary_lang = lang_counts.most_common(1)[0][0]
        self._detected_lang = primary_lang
        self._lang_code = primary_lang
        
        if lang_counts.most_common(1)[0][1] / total < 0.8:
            self._is_multilingual = True
            print(f"[StopwordManager] Detected multilingual dataset, will load multiple stopword lists")
        else:
            self._is_multilingual = False
        
        return primary_lang
    
    def load_stopwords(self, lang: Optional[str] = None) -> Set[str]:
        """
        加载停用词
        
        对于多语言混合数据集，加载所有检测到的主要语言的停用词。
        
        Args:
            lang: 语种代码，如果为 None 则使用检测到的语种
            
        Returns:
            停用词集合
        """
        if lang is None:
            lang = self._detected_lang or 'en'
        
        self._stopwords.clear()
        self._loaded_languages.clear()
        if self._custom is not None:
            self._stopwords.update(self._custom)
            return self._stopwords
        
        langs_to_load = [lang]
        
        if self._lang_distribution:
            langs_to_load = [
                l for l, ratio in self._lang_distribution.items() 
                if (self.stopwords_dir / f'{l}.txt').exists()
            ]
            if not langs_to_load:
                langs_to_load = [lang]
        
        for l in langs_to_load:
            lang_file = self.stopwords_dir / f'{l}.txt'
            if lang_file.exists():
                before_count = len(self._stopwords)
                self._load_file(lang_file)
                self._loaded_languages.add(l)
                added = len(self._stopwords) - before_count
                print(f"[StopwordManager] Loaded {added} stopwords from {l}.txt")
            else:
                print(f"[StopwordManager] Warning: {l}.txt not found")
        
        common_file = self.stopwords_dir / 'common.txt'
        if common_file.exists():
            before_count = len(self._stopwords)
            self._load_file(common_file)
            added = len(self._stopwords) - before_count
            print(f"[StopwordManager] Merged {added} stopwords from common.txt")
        
        print(f"[StopwordManager] Total stopwords: {len(self._stopwords)}")
        
        return self._stopwords
    
    def _load_file(self, filepath: Path) -> None:
        """从文件加载停用词"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        self._stopwords.add(normalize_word(word))
        except Exception as e:
            print(f"[StopwordManager] Error loading {filepath}: {e}")
    
    def get_stopwords(self) -> Set[str]:
        """获取已加载的停用词集合"""
        return self._stopwords
    
    def is_stopword(self, word: str) -> bool:
        """判断是否为停用词"""
        return normalize_word(word) in self._stopwords
    
    def filter_stopwords(self, words: List[str]) -> List[str]:
        """过滤停用词"""
        return [w for w in words if not self.is_stopword(w)]
    
    def is_cjk_language(self) -> bool:
        """判断是否为中日韩语种（需要特殊分词）"""
        return self._detected_lang in self.CJK_LANGS
    
    def tokenize(self, text: str) -> List[str]:
        """
        根据语种进行分词
        
        - 中文：使用 jieba
        - 日文：使用 jieba（可处理部分日文）或简单分割
        - 韩文：按空格分割
        - 拉丁语系：使用正则分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        text = unicodedata.normalize('NFKC', text)
        tokens = []
        # Split by script, preserving mixed Latin/Cyrillic/Arabic/Hangul tokens.
        # No branch may throw away the other scripts in the same row.
        for segment in re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+|[\uac00-\ud7af\u1100-\u11ff]+|[^\W\d_\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u1100-\u11ff]+', text):
            if re.search(r'[\u3040-\u30ff]', segment):
                lang, words = 'ja', self._tokenize_japanese(segment)
            elif re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', segment):
                lang, words = 'zh', self._tokenize_chinese(segment)
            elif re.search(r'[\uac00-\ud7af\u1100-\u11ff]', segment):
                lang, words = 'ko', self._tokenize_korean(segment)
            elif re.search(r'[\u0400-\u052f]', segment):
                lang, words = 'ru', [segment]
            else:
                lang, words = None, [segment]
            # Even a rare script outside the sampled rows gets its bundled list.
            if lang and lang not in self._loaded_languages and self._custom is None:
                file = self.stopwords_dir / f'{lang}.txt'
                if file.exists(): self._load_file(file)
                self._loaded_languages.add(lang)
            tokens.extend(normalize_word(word) for word in words if word.strip())
        return tokens
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词 - 使用 jieba"""
        try:
            import jieba
            words = jieba.lcut(text)
            return [w.strip() for w in words if w.strip()]
        except ImportError:
            print("[StopwordManager] jieba not installed, falling back to regex tokenization")
            raise RuntimeError('中文分词需要 jieba，请安装计算环境依赖')
    
    def _tokenize_japanese(self, text: str) -> List[str]:
        """日文分词"""
        try:
            import jieba
            words = jieba.lcut(text)
            return [w.strip() for w in words if w.strip()]
        except ImportError:
            return re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', text)
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """韩文分词 - 按空格分割"""
        words = re.findall(r'[\uAC00-\uD7AF]+', text)
        return [w for w in words if len(w) > 1]
    
    def _tokenize_latin(self, text: str) -> List[str]:
        """拉丁语系分词 - 使用正则"""
        return re.findall(r'[^\W\d_]+', text.casefold())
    
    def process_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        完整的文本处理流程：分词 + 可选停用词过滤
        
        Args:
            text: 输入文本
            remove_stopwords: 是否过滤停用词
            
        Returns:
            处理后的词列表
        """
        words = self.tokenize(text)
        if remove_stopwords:
            words = self.filter_stopwords(words)
        return words
    
    def auto_process(self, text: str, remove_stopwords: bool = True) -> Tuple[str, List[str]]:
        """
        自动检测语种并处理文本
        
        Args:
            text: 输入文本
            remove_stopwords: 是否过滤停用词
            
        Returns:
            (检测到的语种, 处理后的词列表)
        """
        lang = self.detect_language(text)
        self.load_stopwords(lang)
        words = self.process_text(text, remove_stopwords)
        return lang, words
    
    @property
    def detected_language(self) -> Optional[str]:
        """获取检测到的语种"""
        return self._detected_lang
    
    @property
    def original_lang_code(self) -> Optional[str]:
        """获取原始的 langdetect 返回代码"""
        return self._lang_code
    
    def get_language_name(self) -> str:
        """获取语种的可读名称"""
        names = {
            'zh': 'Chinese',
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
        }
        return names.get(self._detected_lang, 'Unknown')


_default_manager: Optional[StopwordManager] = None


def get_stopword_manager() -> StopwordManager:
    """获取全局停用词管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = StopwordManager()
    return _default_manager


def detect_and_load(text: str) -> Tuple[str, Set[str]]:
    """
    便捷函数：检测语种并加载停用词
    
    Args:
        text: 输入文本
        
    Returns:
        (语种代码, 停用词集合)
    """
    manager = get_stopword_manager()
    lang = manager.detect_language(text)
    stopwords = manager.load_stopwords(lang)
    return lang, stopwords
