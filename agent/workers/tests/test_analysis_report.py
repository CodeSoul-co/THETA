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
                read.assert_called_once_with({'home': folder, 'jobId': 'job', 'view': 'analysis'})
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

    def test_verified_split_exhibits_embed_in_pdf_and_portable_bundle(self):
        import pymupdf
        import zipfile
        from workers.report_exhibits import collect_exhibits
        from workers.results_reader import tree_hash
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'result'; directory = Path(folder) / 'export'
            for group, weight in [('all', '0.9'), ('train', '0.8'), ('validation', '0.3'), ('test', '0.1')]:
                target = root / 'zh' / ('global' if group == 'all' else f'splits/{group}/global'); target.mkdir(parents=True)
                (target / 'topic_proportions.csv').write_text(f'topic_id,mean_weight\n1,{weight}\n2,0.2\n', encoding='utf-8')
                pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 900, 600), False)
                pix.clear_with(190); pix.save(target / '主题占比分布.png')
            catalog = collect_exhibits({'_resultHash': tree_hash(root)}, root, directory)
            self.assertEqual(len(catalog), 8)
            self.assertNotIn('topic_id', catalog[0]['statistics'])
            self.assertAlmostEqual(catalog[0]['statistics']['mean_weight']['mean'], 0.55)
            self.assertEqual([item['rows'][0]['mean_weight'] for item in catalog if item['kind'] == 'table'], ['0.9', '0.8', '0.3', '0.1'])
            self.assertEqual([item['scope'] for item in catalog if item['kind'] == 'figure'], ['全量', '训练集', '验证集', '测试集'])
            markdown = '# 完整分析报告\n\n' + '\n\n'.join(item['block'] for item in catalog) + '\n\n## 结论\n\n全量与测试集主题权重存在差异。'
            (directory / 'analysis.md').write_text(markdown, encoding='utf-8')
            export_pdf({'directory': str(directory)})
            with pymupdf.open(directory / 'analysis.pdf') as pdf:
                self.assertEqual(sum(len(page.get_images()) for page in pdf), 4)
                self.assertIn('测试集', ''.join(page.get_text() for page in pdf))
            with zipfile.ZipFile(directory / 'analysis.zip') as bundle:
                self.assertIn('analysis.md', bundle.namelist())
                self.assertEqual(len(bundle.namelist()), 9)
                self.assertEqual(bundle.read('assets/table-1.csv'), (root / 'zh/global/topic_proportions.csv').read_bytes())
            rendered = markdown_html('![远程](https://example.invalid/secret)\n\n![本地](../../secret.png)\n\n- 列表\n\n---\n\n**加粗**')
            self.assertNotIn('<img', rendered); self.assertIn('<ul>', rendered); self.assertIn('<hr', rendered); self.assertIn('<strong>', rendered)
            with self.assertRaisesRegex(ValueError, '变化'):
                collect_exhibits({'_resultHash': 'modified'}, root, directory)

    def test_native_manifest_sources_are_verified_and_preferred_over_training_plots(self):
        from workers.report_exhibits import collect_exhibits
        from workers.results_reader import file_hash
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); native = root / 'report/native/en/global'; native.mkdir(parents=True)
            table = native / 'topic_table.csv'; table.write_text('topic,word\n1,science\n', encoding='utf-8')
            manifest = root / 'report/manifest.json'
            manifest.write_text(json.dumps({'files': [{'path': str(table), 'sha256': file_hash(table)}]}))
            summary = {'reportPath': str(root / 'report/index.html')}
            catalog = collect_exhibits(summary, root / 'unused-training', root / 'export')
            self.assertEqual(catalog[0]['rows'][0]['word'], 'science')
            self.assertEqual(catalog[0]['statistics'], {})
            table.write_text('topic,word\n1,modified\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '变化'):
                collect_exhibits(summary, root / 'unused-training', root / 'export')
