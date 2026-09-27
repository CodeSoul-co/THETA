import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workers.analysis_report import evidence, export_pdf, markdown_html, numeric_description
from workers.capabilities import dataset_import, dataset_profile
from workers.local_compute import database


class AnalysisReportTests(unittest.TestCase):
    def test_independent_sources_have_separate_verified_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = {}
            for role, numbers in [('train', [1, 2, 3]), ('validation', [10, 20]), ('test', [100, 200])]:
                file = root / f'{role}.csv'
                file.write_text('text,score\n' + '\n'.join(f'{role} research sample {i},{i}' for i in numbers), encoding='utf-8')
                descriptor = dataset_import({'filePath': str(file), 'uploadDir': str(root/'uploads')})
                sources[role] = {'dataset': descriptor, 'textColumn': 'text'}
            output = root / 'result'; output.mkdir()
            (output/'split_results.json').write_text(json.dumps({'groups': {'all': {'count': 7}, 'train': {'count': 3}, 'validation': {'count': 2}, 'test': {'count': 2}}, 'testUsesAllData': False}))
            request = {'dataset': sources['train']['dataset'], 'plan': {'modelId': 'lda', 'textColumn': 'text', 'dataSplit': {'enabled': True, 'mode': 'upload', 'sources': sources}}}
            job = {'id': 'job', 'resultDir': str(output), 'resultHash': 'verified'}
            with database(folder) as db:
                # The report must use the persisted request and validation boundary.
                db.execute('INSERT INTO jobs(id,state,request,value,updated,cancel) VALUES (?,?,?,?,?,?)', ('job', 'completed', json.dumps(request), json.dumps(job), 0, 0))
            with patch('workers.local_compute.results', return_value={'jobId': 'job'}) as read:
                result = evidence({'home': folder, 'jobId': 'job'})
                read.assert_called_once_with({'home': folder, 'jobId': 'job', 'view': 'summary'})
            self.assertEqual([result['profiles'][r]['numericDescription'][0]['mean'] for r in sources], [2, 15, 150])
            self.assertIn('验证来源', result['description'])
            self.assertIn('| 全量 | 7 |', result['description'])
            self.assertNotIn('不是独立留出测试', result['description'])
            Path(sources['test']['dataset']['managedPath']).write_text('changed')
            with patch('workers.local_compute.results', return_value={}), self.assertRaisesRegex(ValueError, '版本已变化'):
                evidence({'home': folder, 'jobId': 'job'})

    def test_pdf_has_searchable_chinese_tables_multiple_pages_and_no_active_html(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prose = '研究问题与数据分析。主题结果围绕真实统计展开，并解释研究设计与证据的关系。' * 12
            markdown = '# 中文学术报告\n\n|字段|均值|\n|---|---|\n|统计结果|12.5|\n\n' + '\n\n'.join(prose for _ in range(24)) + '\n\n## 结论分析\n\n实际结论在此。'
            (root/'analysis.md').write_text(markdown, encoding='utf-8')
            exported = export_pdf({'directory': folder})
            self.assertGreater(exported['pages'], 1)
            with pymupdf.open(root/'analysis.pdf') as pdf:
                text = ''.join(page.get_text() for page in pdf)
                self.assertIn('中文学术报告', text)
                self.assertIn('12.5', text)
                self.assertIn('实际结论在此', text)
                self.assertIn(f'THETA | {len(pdf)} / {len(pdf)}', text)
            self.assertNotIn('<img', markdown_html('<img src="https://example.invalid/a">'))
            self.assertFalse((root/'analysis.partial.pdf').exists())
