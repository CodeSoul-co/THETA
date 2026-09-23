import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/models'))
from utils.stopword_manager import StopwordManager, configure_stopwords
from bow.vocab_builder import VocabBuilder, VocabConfig
from workers.api_overrides import install
from workers.model_contract import validate_parameters


class MultilingualStopwordTests(unittest.TestCase):
    def setUp(self):
        configure_stopwords(None)

    def tearDown(self):
        configure_stopwords(None)

    def test_same_row_mixed_scripts_keep_all_languages(self):
        manager = StopwordManager()
        manager.detect_language_from_documents(['the research and analysis 数据分析 和 中文研究 русский анализ 한국어 데이터'] * 3)
        manager.load_stopwords()
        words = manager.process_text('the pineapple and 数据分析 русский анализ 한국어 데이터')
        self.assertIn('pineapple', words)
        self.assertIn('数据分析', words)
        self.assertIn('анализ', words)
        self.assertIn('한국어', words)
        self.assertNotIn('the', words)
        self.assertNotIn('and', words)
        self.assertNotIn('和', manager.process_text('和 数据分析'))

    def test_minority_language_is_not_discarded_by_ten_percent_threshold(self):
        manager = StopwordManager()
        manager.detect_language_from_documents(['The research project examines software and technology for scientists.'] * 99 + ['这是一个数据分析研究的项目'])
        manager.load_stopwords()
        self.assertIn('zh', manager._loaded_languages)
        self.assertIn('en', manager._loaded_languages)
        self.assertTrue(manager.is_stopword('the'))
        self.assertTrue(manager.is_stopword('的'))

    def test_unicode_and_adjacent_scripts_do_not_drop_text(self):
        words = StopwordManager().tokenize('research数据分析русский 한국어 café Ελληνικά العربية')
        for word in ['research', '数据分析', 'русский', '한국어', 'café', 'ελληνικά', 'العربية']:
            self.assertIn(word, words)

    def test_two_latin_languages_in_one_document_load_both_lists(self):
        manager = StopwordManager()
        manager.detect_language_from_documents(['The software project provides useful tools for scientists. Die wissenschaftliche Forschung untersucht verschiedene technische Entwicklungen und neue Methoden.'])
        manager.load_stopwords()
        self.assertTrue({'en', 'de'}.issubset(manager._loaded_languages))
        self.assertTrue(manager.is_stopword('the'))
        self.assertTrue(manager.is_stopword('und'))

    def test_custom_replaces_default_and_is_used_by_actual_bow(self):
        install({'modelId': 'lda', 'params': {'text.stopwords': 'research\n数据分析\n# comment\nＣＵＳＴＯＭ'}}, Path('prepare_data.py'), ROOT)
        builder = VocabBuilder(VocabConfig(min_df=1, max_df_ratio=1, min_word_length=1))
        builder.add_documents(['the research 数据分析 custom science', 'the research 数据分析 custom market'], 'mixed', show_progress=False)
        builder.build_vocab()
        vocab = builder.get_vocab_list()
        self.assertIn('the', vocab)  # custom replaces rather than merges defaults
        for word in ['research', '数据分析', 'custom']:
            self.assertNotIn(word, vocab)
        configure_stopwords(None)
        manager = StopwordManager()
        self.assertIn('the', manager.load_stopwords('en'))

    def test_cjk_detection_does_not_depend_on_langdetect_availability(self):
        with patch.dict(sys.modules, {'langdetect': None}):
            manager = StopwordManager()
            self.assertEqual(manager.detect_language('中文文本数据分析'), 'zh')
            self.assertIn('数据分析', manager.tokenize('中文文本数据分析'))

    def test_custom_policy_validated_for_each_model(self):
        for model in ['theta', 'lda', 'btm', 'hdp', 'stm', 'dtm', 'etm', 'ctm', 'nvdm', 'gsm', 'prodlda', 'bertopic']:
            validate_parameters(ROOT, {'modelId': model, 'params': {'text.stopwords': '分析\nresearch'}, 'timeoutSeconds': 60})
        for value in ['', ['analysis'], 'a' * 101, 'bad\x00word', '词\n' * 20001]:
            with self.assertRaises(ValueError):
                validate_parameters(ROOT, {'modelId': 'lda', 'params': {'text.stopwords': value}, 'timeoutSeconds': 60})
