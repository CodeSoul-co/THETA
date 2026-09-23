"""BOW-only models still need their selected document metadata."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_data


class BowMetadataTests(unittest.TestCase):
    def test_bow_only_preserves_stm_covariates_without_embedding_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.csv'
            source.write_text('text,cov_0\n主题分析,公众号甲\n多语言数据,公众号乙\n', encoding='utf-8')
            output = root / 'workspace'
            args = SimpleNamespace(dataset='fixture', user_id='test', output_dir=str(output),
                force=True, vocab_size=2, batch_size=2, max_length=256, skip_sbert=True,
                bow_only=True, with_time=False, time_column='year', covariate_columns=['cov_0'], label_col=None)
            with patch.object(prepare_data, 'find_data_file', return_value=source), \
                 patch.object(prepare_data, 'generate_bow', return_value=(np.eye(2), ['主题', '数据'])), \
                 patch.object(prepare_data, 'generate_sbert_embeddings') as sbert, \
                 patch.object(prepare_data, 'generate_word2vec_embeddings') as word2vec:
                self.assertTrue(prepare_data.prepare_baseline_data(args))
            self.assertEqual(np.load(output / 'covariates.npy').shape, (2, 1))
            self.assertEqual(json.loads((output / 'covariate_names.json').read_text()), ['cov_0'])
            sbert.assert_not_called()
            word2vec.assert_not_called()
