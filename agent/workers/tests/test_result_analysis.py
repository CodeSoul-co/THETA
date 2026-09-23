import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from workers.results_reader import read_result_evidence, tree_hash
from workers.result_analysis import matrix_summary


class ResultAnalysisTests(unittest.TestCase):
    def test_native_topic_and_temporal_tables_survive_large_chart_inventory(self):
        from workers.result_analysis import presentation_evidence
        from workers.results_reader import file_hash
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            # Alphabetic early artifacts used to exhaust the 8-table/60-figure caps.
            for i in range(10):
                (root / f'diagnostic_{i}.csv').write_text('value\n1\n')
            for i in range(70):
                (root / f'chart_{i}.svg').write_text('<svg/>')
            (root / 'topic_table.csv').write_text('topic,word\n1,transport\n')
            (root / 'topic_weights_by_year.csv').write_text('year,topic_1\n2023,0.6\n2024,0.4\n')
            result = presentation_evidence(root, file_hash, [])
            tables = {entry['relativePath']: entry for entry in result['tables']}
            self.assertEqual(tables['topic_table.csv']['rows'][0]['word'], 'transport')
            self.assertEqual(tables['topic_weights_by_year.csv']['rowCount'], 2)
            self.assertEqual(len(result['figures']), 70)
            self.assertEqual(result['analysisSkipped'], [])

    def test_verified_tables_chart_sources_and_distributions_are_available(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home) / 'job-fixture'; root.mkdir()
            (root / 'topic_words_k2.json').write_text(json.dumps({'0': ['配送', '物流'], '1': ['退款', '售后']}))
            (root / 'metrics.json').write_text('{"coherence":0.52}')
            (root / 'topic_table.csv').write_text('topic_id,topic_name,strength\n1,配送,0.6\n2,退款,0.4\n')
            np.save(root / 'theta_k2.npy', np.array([[0.9, 0.1], [0.3, 0.7]]))
            np.save(root / 'beta_k2.npy', np.array([[1., 0.], [0., 1.]]))
            (root / 'topic_similarity.png').write_bytes(b'fixture image inventory only')
            (root / 'unknown_plot.png').write_bytes(b'fixture image inventory only')
            payload = {'trainingRunId': 'job-fixture', 'artifacts': [{'kind': 'results', 'path': str(root), 'sha256': tree_hash(root), 'artifactId': 'result'}]}
            result = read_result_evidence(payload)
            self.assertEqual(result['tables'][0]['rows'][0]['topic_name'], '配送')
            distribution = next(item for item in result['matrices'] if item['kind'] == 'topic_distribution')
            self.assertEqual(distribution['values'], [[0.9, 0.1], [0.3, 0.7]])
            self.assertEqual(distribution['omittedRows'], 0)
            similarity = next(item for item in result['matrices'] if item['kind'] == 'topic_word_distribution')
            self.assertEqual(similarity['values'], [[1., 0.], [0., 1.]])
            self.assertEqual(result['figures'][0]['dataSource'], 'beta_k2.npy')
            self.assertIsNone(result['figures'][1]['dataSource'])
            self.assertEqual(result['figures'][0]['basis'], 'underlying_results_not_image_pixels')
            (root / 'topic_table.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'changed'):
                read_result_evidence(payload)

    def test_invalid_distributions_and_pickle_are_not_interpreted(self):
        with tempfile.TemporaryDirectory() as home:
            file = Path(home) / 'theta.npy'
            for value in [np.array([[0., 0.]]), np.array([[np.nan, 1.]]), np.array([[-1., 2.]]), np.array([{'payload': 'object'}], dtype=object)]:
                np.save(file, value)
                with self.subTest(value=str(value)), self.assertRaises(ValueError):
                    matrix_summary(file)

    def test_nvdm_latent_coordinates_are_preserved_and_not_treated_as_probabilities(self):
        with tempfile.TemporaryDirectory() as home:
            file = Path(home)/'theta_k2.npy'; original = np.array([[-1., 2.], [0., 0.]])
            np.save(file, original)
            result = matrix_summary(file, 'nvdm')
            self.assertEqual(result['kind'], 'latent_document_representation')
            self.assertEqual(result['values'], original.tolist())
            with self.assertRaises(ValueError): matrix_summary(file, 'prodlda')
            np.save(file, np.array([[np.nan, 1.]]))
            with self.assertRaises(ValueError): matrix_summary(file, 'nvdm')
