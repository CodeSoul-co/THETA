import csv
import json
from pathlib import Path
import tempfile
import sqlite3
from contextlib import closing
import time
import unittest
import sys
from unittest.mock import patch

from workers.capabilities import dataset_import, dataset_profile, validate_plan, model_inspect
from workers.local_compute import normalize_dataset, database, submit, results
from workers.results_reader import tree_hash
from workers.execution_policy import execution_policy


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.csv'
        self.source.write_text('content,year,group\nTransport policy text,2020,A\nHealth policy text,2021,B\n', encoding='utf-8')
        self.dataset = dataset_import({'filePath': str(self.source), 'uploadDir': str(self.root / 'uploads')})
        self.plan = {'modelId': 'lda', 'textColumn': 'content', 'params': {'num_topics': 2}, 'rationale': 'baseline', 'timeoutSeconds': 60}

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_completed_bertopic_reports_export_warning_without_rewriting_job(self):
        from workers.local_compute import database, status
        job_id = 'job-' + 'd' * 64
        root = self.root / 'compute' / job_id / 'jobs/task-1-attempt-1/result'
        root.mkdir(parents=True)
        (root.parent / 'worker.log').write_text("KeyError: ''\n")
        value = {'id': job_id, 'status': 'completed', 'phase': 'completed', 'percent': 100,
                 'resultDir': str(root), 'plan': {'modelId': 'bertopic'}}
        with database(str(self.root)) as db:
            db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)',
                       (job_id, '{}', 'completed', json.dumps(value), 0))
        observed = status({'home': str(self.root), 'jobId': job_id})
        self.assertIn('导出不完整', observed['resultWarning'])
        self.assertIn('KeyError', observed['diagnostics']['message'])
        with database(str(self.root)) as db:
            self.assertEqual(json.loads(db.execute('SELECT value FROM jobs WHERE id=?', (job_id,)).fetchone()[0]), value)

    def test_runtime_requires_weights_and_rejects_cloud_finetuning(self):
        from workers.capabilities import runtime_check
        with tempfile.TemporaryDirectory() as home:
            asset = Path(home); (asset / 'config.json').write_text('{}')
            request = {'modelId': 'ctm', 'modelAssets': {'SBERT_MODEL_PATH': home}}
            self.assertFalse(runtime_check(request)['modelAssetsReady'])
            (asset / 'model.safetensors').write_bytes(b'probe only')
            self.assertTrue(runtime_check(request)['modelAssetsReady'])
            self.assertFalse(runtime_check({'modelId': 'theta', 'embeddingProvider': 'cloud', 'mode': 'supervised'})['ready'])

    def test_profile_is_aggregate_and_data_tampering_fails(self):
        value = dataset_profile({'dataset': self.dataset})
        self.assertEqual(value['rowCount'], 2)
        self.assertNotIn('Transport policy text', json.dumps(value))
        self.assertNotIn('sampleRows', value)
        Path(self.dataset['managedPath']).write_text('changed')
        with self.assertRaisesRegex(ValueError, '数据版本'):
            dataset_profile({'dataset': self.dataset})

    def test_schema_rejects_bad_columns_parameters_and_cloud_embedding(self):
        validate_plan({'plan': self.plan, 'dataset': self.dataset})
        for changes in [{'textColumn': 'missing'}, {'params': {'num_topics': 2.5}}, {'params': {'skip_eval': 'false'}}, {'params': {'embedding_provider': 'cloud'}}, {'modelId': 'dtm'}, {'modelId': 'stm'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_plan({'plan': {**self.plan, **changes}, 'dataset': self.dataset})

    def test_parameters_match_both_actual_engine_entrypoints_before_launch(self):
        choices = model_inspect({'modelId': 'lda'})['parameterChoices']
        self.assertEqual(choices['language'], ['en', 'zh', 'chinese', 'english'])
        for language in ['en', 'zh', 'chinese', 'english']:
            validate_plan({'plan': {**self.plan, 'params': {'language': language}}, 'dataset': self.dataset})
        for params in [{'language': 'german'},
                       {'mode': 'invented'}, {'model_size': '7B'}, {'inference_type': 'wrong'}]:
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, '允许值|不支持参数'):
                validate_plan({'plan': {**self.plan, 'params': params}, 'dataset': self.dataset})

    def test_dataset_operations_return_requested_aggregate_without_values(self):
        base = {'dataset': self.dataset, 'column': 'content'}
        text = dataset_profile({**base, 'operation': 'text_profile'})
        self.assertGreater(text['detail']['medianLength'], 0)
        categories = dataset_profile({**base, 'column': 'group', 'operation': 'categorical_profile'})
        self.assertEqual(categories['detail']['categoryCount'], 2)
        self.assertTrue(categories['detail']['categoryLabelsWithheld'])
        duplicates = dataset_profile({**base, 'operation': 'duplicates'})
        self.assertEqual(duplicates['detail']['duplicateCount'], 0)
        relation = dataset_profile({'dataset': self.dataset, 'operation': 'relationships', 'columns': ['year', 'group']})
        self.assertIsNone(relation['detail']['pearsonCorrelation'])

    def test_normalization_preserves_source_and_maps_only_selected_columns(self):
        original = self.source.read_bytes()
        output = self.root / 'normalized.csv'
        normalize_dataset({'plan': {**self.plan, 'timeColumn': 'year', 'covariates': ['group']}, 'dataset': self.dataset}, output)
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['text'], 'Transport policy text')
        self.assertEqual(rows[1]['cov_0'], 'B')
        self.assertEqual(self.source.read_bytes(), original)

    def test_durable_submission_never_starts_same_job_twice(self):
        payload = {'home': str(self.root), 'jobId': 'job-' + 'a' * 64, 'runId': 'run', 'dataset': self.dataset, 'plan': self.plan}
        payload['execution'] = execution_policy(self.plan)
        approval = {'status': 'approved', 'hash': 'receipt-hash', 'action': 'compute.submit', 'target': 'local',
                    'payload': {key: value for key, value in payload.items() if key != 'home'}, 'expiresAt': time.time() * 1000 + 60_000}
        with closing(sqlite3.connect(self.root / 'research.sqlite')) as db, db:
            db.execute('CREATE TABLE records(kind TEXT, id TEXT, value TEXT)')
            db.execute('INSERT INTO records VALUES (?,?,?)', ('effect-approval', 'receipt', json.dumps(approval)))
        payload['authorization'] = {'id': 'receipt', 'hash': 'receipt-hash'}
        with patch('workers.local_compute.runtime_check', return_value={'ready': True}), patch('workers.local_compute.subprocess.Popen') as launch:
            first = submit(payload)
            self.assertEqual(submit(payload)['id'], first['id'])
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[0][0], payload['execution']['runtime']['python'])
            self.assertEqual(first['runtime'], payload['execution']['runtime'])
            with self.assertRaisesRegex(ValueError, '不同参数'):
                submit({**payload, 'plan': {**self.plan, 'params': {'num_topics': 3}}})

    def test_result_evidence_is_bound_to_job_and_detects_tampering(self):
        job_id = 'job-' + 'b' * 64
        root = self.root / 'compute' / job_id / 'result'
        root.mkdir(parents=True)
        metrics = root / 'metrics.json'
        metrics.write_text('{"coherence":0.52}')
        for index in range(12):
            (root / f'plot-{index:02d}.png').write_bytes(b'fixture inventory')
        job = {'id': job_id, 'status': 'completed', 'phase': 'completed', 'percent': 100, 'resultDir': str(root), 'resultHash': tree_hash(root)}
        with database(str(self.root)) as db:
            db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)', (job_id, '{}', 'completed', json.dumps(job), 1))
        payload = {'home': str(self.root), 'jobId': job_id, 'view': 'summary'}
        self.assertEqual(results(payload)['evidence'][0]['content']['coherence'], 0.52)
        self.assertEqual(len(results(payload)['figures']), 5)
        first_page = results({**payload, 'view': 'figures'})
        self.assertEqual(len(first_page['figures']), 10)
        self.assertEqual(first_page['nextOffset'], 10)
        last_page = results({**payload, 'view': 'figures', 'offset': 10})
        self.assertEqual(len(last_page['figures']), 2)
        self.assertIsNone(last_page['nextOffset'])
        native = results(payload)
        native['tables'] = [{'relativePath': 'global/topic_table.csv', 'rows': [{'word': 'refund'}]}]
        manifest = self.root / 'reports' / job_id / 'report-native' / 'manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'schemaVersion': 'theta.result-report.v2', 'resultHash': job['resultHash'],
                                       'files': [], 'evidence': native, 'reportPath': 'native/index.html', 'missingEvidence': []}))
        self.assertEqual(results({**payload, 'view': 'tables'})['tables'][0]['rows'][0]['word'], 'refund')
        inventory = results({**payload, 'view': 'artifacts'})
        self.assertEqual(inventory['resultDir'], str(root))
        self.assertEqual(inventory['reportPath'], 'native/index.html')
        self.assertTrue(all(Path(item['path']).is_file() for item in inventory['files']))
        metrics.write_text('{"coherence":0.99}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            results(payload)


if __name__ == '__main__':
    unittest.main()


class CloudReadinessRegression(unittest.TestCase):
    def test_cloud_theta_does_not_require_sbert_or_local_transformer_weights(self):
        from unittest.mock import patch
        from workers.capabilities import runtime_check
        with patch('workers.capabilities.importlib.util.find_spec', side_effect=lambda name: None if name in {'sentence_transformers', 'transformers'} else object()):
            result = runtime_check({'modelId': 'theta', 'embeddingProvider': 'cloud'})
            self.assertTrue(result['ready'], result)
            self.assertIsNone(result['requiredAssetVariable'])
            self.assertIn('transformers', runtime_check({'modelId': 'theta', 'embeddingProvider': 'local'})['missingDependencies'])


class DataPreparationReadinessRegression(unittest.TestCase):
    def test_runtime_check_includes_converter_dependencies(self):
        from unittest.mock import patch
        from workers.capabilities import runtime_check
        with patch('workers.capabilities.importlib.util.find_spec', side_effect=lambda name: None if name == 'docx' else object()):
            result = runtime_check({'modelId': 'lda'})
            self.assertFalse(result['ready'])
            self.assertIn('docx', result['missingDependencies'])

    def test_segmented_chinese_is_not_misclassified_as_korean(self):
        model_root = Path(__file__).resolve().parents[3] / 'src' / 'models'
        sys.path.insert(0, str(model_root))
        try:
            from utils.stopword_manager import StopwordManager
            documents = [
                '人工智能 创业 公司 智能 系统 模型 训练 数据 研究 主题 分析',
                '政策 法规 市场 企业 发展 技术 服务 平台 创新 应用 场景',
            ] * 60
            self.assertEqual(StopwordManager().detect_language_from_documents(documents), 'zh')
        finally:
            sys.path.remove(str(model_root))
