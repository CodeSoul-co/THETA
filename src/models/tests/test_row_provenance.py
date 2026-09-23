"""Synthetic preprocessing lineage checks; no labels or model calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_provenance import record_cleaning, export_lineage, verify_lineage, file_hash


class RowProvenanceTests(unittest.TestCase):
    def test_real_csv_cleaning_and_bow_preparation_export_lineage_without_labels(self):
        import prepare_data
        import pandas as pd
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'raw.csv';workspace=root/'workspace'
            raw=['The red widget failed!']*6+['Blue devices connected.']*6
            pd.DataFrame({'text':raw,'channel':['x']*12}).to_csv(source,index=False)
            args=SimpleNamespace(dataset='fixture',with_time=False,time_column='year',covariate_columns=None,
                label_col='label',output_dir=str(workspace),force=True,user_id='fixture',vocab_size=10,
                batch_size=2,max_length=64,skip_sbert=True,bow_only=True)
            with patch.object(prepare_data,'DATA_DIR',root):
                prepare_data.run_dataclean(str(source),'fixture')
                self.assertTrue(prepare_data.prepare_baseline_data(args))
            texts=json.loads((workspace/'texts.json').read_text())
            record=json.loads((workspace/'row_provenance.json').read_text())
            indices=np.load(workspace/'source_rows.npy',allow_pickle=False)
            self.assertTrue(verify_lineage(record,file_hash(source),raw,texts,indices))
            self.assertEqual(len(texts),12)
            self.assertTrue((workspace/'bow_matrix.npz').exists() or (workspace/'bow_matrix.npy').exists())

    def test_cleaned_reordered_rows_remain_bound_to_original_text_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, cleaned = root/'input.csv', root/'clean.csv'
            source.write_text('text\nThe Red Widget!\n\nBlue widgets.\n')
            cleaned.write_text('cleaned_content\nred widget\n\nblue widget\n')
            raw, texts = ['The Red Widget!', '', 'Blue widgets.'], ['red widget', '', 'blue widget']
            record_cleaning(source, cleaned, raw, texts)
            export_lineage(cleaned, root, [texts[2], texts[0]], [2, 0])
            record = json.loads((root/'row_provenance.json').read_text())
            self.assertTrue(verify_lineage(record, file_hash(source), raw, [texts[2],texts[0]], [2,0]))
            self.assertFalse(verify_lineage(record, file_hash(source), raw, [texts[0],texts[2]], [2,0]))
            self.assertFalse(verify_lineage(record, file_hash(source), raw, [texts[2],texts[0]], [0,2]))
            self.assertFalse(verify_lineage(record, 'different source', raw, [texts[2],texts[0]], [2,0]))
            self.assertFalse(verify_lineage(record, file_hash(source), ['changed','',raw[2]], [texts[2],texts[0]], [2,0]))

    def test_stale_cleaned_file_or_texts_fail_and_absence_never_reuses_old_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'input.csv'; cleaned=root/'clean.csv'
            source.write_text('text\nOriginal\n');cleaned.write_text('cleaned_content\noriginal\n')
            record_cleaning(source, cleaned, ['Original'], ['original'])
            with self.assertRaisesRegex(ValueError, 'texts'):
                export_lineage(cleaned, root, ['different'], [0])
            with self.assertRaisesRegex(ValueError, 'unique'):
                export_lineage(cleaned, root, ['original','original'], [0,0])
            export_lineage(cleaned, root, ['original'], [0])
            cleaned.write_text('cleaned_content\nchanged\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                export_lineage(cleaned, root, ['changed'], [0])
            Path(str(cleaned)+'.lineage.json').unlink()
            export_lineage(cleaned, root, ['changed'], [0])
            self.assertFalse((root/'row_provenance.json').exists())

    def test_empty_text_and_non_ascii_are_exactly_preserved_in_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input.csv';cleaned=root/'clean.csv'
            source.write_text('text\n设备连接失败\n');cleaned.write_text('cleaned_content\n设备 连接\n')
            record_cleaning(source, cleaned, ['设备连接失败',''], ['设备 连接',''])
            export_lineage(cleaned,root,['设备 连接',''],[0,1])
            self.assertTrue(verify_lineage(json.loads((root/'row_provenance.json').read_text()),file_hash(source),['设备连接失败',''],['设备 连接',''],[0,1]))


if __name__ == '__main__':
    unittest.main()
