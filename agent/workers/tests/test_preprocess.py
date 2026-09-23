import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from workers.dataset.preprocess import preprocess, transform
from workers.capabilities import dataset_import


class PreprocessTests(unittest.TestCase):
    def test_real_derived_csv_preserves_source_and_saves_recipe(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            file = home / '混合数据.csv'
            original = '正文,时间,来源\n  中文 English  ,2026-09-21,A\n ,2026-09-20,B\n第二篇 research,2026-09-19,A\n'
            file.write_text(original, encoding='utf-8')
            source = dataset_import(dict(filePath=str(file), uploadDir=str(home / 'uploads')))
            result = preprocess(dict(dataset=source, home=root, uploadDir=str(home / 'uploads'),
                purpose='去掉空正文并规范日期，保留原始语言', textColumn='正文', timeColumn='时间',
                code='df["正文"] = df["正文"].fillna("").astype("str").str.strip()\ndf = df[df["正文"].str.len() > 0]\ndf["时间"] = pd.to_datetime(df["时间"], errors="coerce").dt.strftime("%Y-%m-%d")'))
            self.assertEqual(file.read_text(), original)
            self.assertEqual(result['summary']['inputRows'], 3)
            self.assertEqual(result['summary']['outputRows'], 2)
            self.assertEqual(result['profile']['rowCount'], 2)
            self.assertNotEqual(result['dataset']['datasetRef'], source['datasetRef'])
            self.assertEqual(len(result['artifacts']), 3)
            self.assertTrue(all(Path(item['path']).is_file() for item in result['artifacts']))

    def test_code_cannot_execute_host_operations_or_callbacks(self):
        frame = pd.DataFrame({'正文': ['a', 'b']})
        for code in ['import os', 'df = __import__("os").system("id")', 'df = df.to_pickle("/tmp/escape")',
                     'df = pd.read_csv("/etc/passwd")', 'df = df.apply(lambda x: x)',
                     'df = df.__class__', 'df = df.query("@secret")', 'df = df.eval("1+1")',
                     'df = df.copy(**{})', 'while True: pass']:
            with self.subTest(code=code), self.assertRaises(ValueError):
                transform(frame, code)

    def test_cleaning_filtering_metadata_and_concatenation(self):
        frame = pd.DataFrame({'title': ['A', 'A', 'B'], 'body': [' Hi ', ' Hi ', ' 世界 '], 'score': ['1', '1', '2']})
        result = transform(frame, 'df = df.drop_duplicates(subset=["title", "body"])\ndf["正文"] = df["title"] + " " + df["body"].str.strip()\ndf["分数"] = pd.to_numeric(df["score"], errors="coerce")\ndf = df[df["分数"] >= 2]')
        self.assertEqual(result['正文'].tolist(), ['B 世界'])
        self.assertEqual(len(frame), 3)

    def test_empty_result_or_invalid_time_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / 'data.csv'
            file.write_text('text,time\nhello,not-a-date\n', encoding='utf-8')
            source = dataset_import(dict(filePath=str(file), uploadDir=str(Path(root) / 'uploads')))
            base = dict(dataset=source, home=root, uploadDir=str(Path(root) / 'uploads'), purpose='测试', textColumn='text')
            with self.assertRaisesRegex(ValueError, '时间列'):
                preprocess({**base, 'timeColumn': 'time', 'code': 'df = df.copy()'})
            with self.assertRaisesRegex(ValueError, '为空'):
                preprocess({**base, 'code': 'df = df[df["text"] == "missing"]'})
