"""Adapter regression only. Product acceptance is performed through ./theta."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from workers.result_report import generate_report
from workers.results_reader import tree_hash, file_hash
from workers.capabilities import engine_root
sys.path.insert(0, str(engine_root() / 'src/models'))
from visualization import run_visualization as native


class ResultReportTests(unittest.TestCase):
    def test_cleaning_lineage_allows_verified_join_but_reordered_map_and_changed_text_do_not(self):
        from row_provenance import record_cleaning, export_lineage
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)/'job';root.mkdir();workspace=Path(home)/'workspace';workspace.mkdir()
            theta=np.array([[.8,.2],[.1,.9]])
            np.save(root/'theta_k2.npy',theta);np.save(root/'beta_k2.npy',np.eye(2))
            (workspace/'vocab.json').write_text('["refund","shipping"]')
            source=Path(home)/'source.csv';source.write_text('text,year\nRefund please!,2024\nShipping delayed.,2025\n')
            cleaned=Path(home)/'cleaned.csv';cleaned.write_text('cleaned_content\nrefund\nshipping\n')
            record_cleaning(source,cleaned,['Refund please!','Shipping delayed.'],['refund','shipping'])
            export_lineage(cleaned,workspace,['refund','shipping'],[0,1])
            (workspace/'texts.json').write_text('["refund","shipping"]');np.save(workspace/'source_rows.npy',np.array([0,1]))
            dataset={'managedPath':str(source),'sha256':file_hash(source),'datasetRef':'fixture'}
            job={'id':'job','resultHash':tree_hash(root),'preparedHash':file_hash(source),
                 'plan':{'modelId':'btm','textColumn':'text','params':{'num_topics':2}}}
            def render(**kwargs):
                output=kwargs['output_dir'];output.mkdir(parents=True)
                (output/'topic_table.csv').write_text('topic,word\n1,refund\n2,shipping\n')
            data={'theta':theta,'beta':np.eye(2),'vocab':['refund','shipping']}
            with patch.object(native,'load_baseline_data',return_value=data),patch.object(native,'run_baseline_visualization',side_effect=render):
                report=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertTrue(report['sourceData']['matrixRowsAligned'])
                self.assertEqual(report['sourceData']['rows'][0]['text'],'Refund please!')
                np.save(workspace/'source_rows.npy',np.array([1,0]))
                invalid=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertFalse(invalid['sourceData']['matrixRowsAligned'])
                np.save(workspace/'source_rows.npy',np.array([0,1]))
                (workspace/'texts.json').write_text('["refund","different"]')
                invalid=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertFalse(invalid['sourceData']['matrixRowsAligned'])
                (workspace/'texts.json').write_text('["refund","shipping"]')
                (workspace/'row_provenance.json').unlink()
                legacy=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertFalse(legacy['sourceData']['matrixRowsAligned'])

    def test_stm_reports_descriptive_groups_without_retesting_fitted_covariates(self):
        import pandas as pd
        data = {'theta': np.array([[.8, .2], [.81, .19], [.1, .9], [.11, .89]]),
                'covariates': np.array([[0], [0], [1], [1]]), 'covariate_names': ['region'],
                'covariate_value_labels': {0: 'north', 1: 'south'}, 'Gamma': np.array([[0.], [.2]])}
        with tempfile.TemporaryDirectory() as home, patch.object(native, 'save_figure'):
            native._run_stm_specific_visualizations(data, home, dpi=72, formats=('png',))
            table = pd.read_csv(Path(home) / 'global/stm_group_topic_means.csv')
            self.assertEqual(table['n_documents'].tolist(), [2, 2])
            np.testing.assert_allclose(table['T1'], [.805, .105])
            self.assertFalse((Path(home) / 'global/exploratory_anova.csv').exists())

    def test_nearly_collinear_topic_pca_exports_without_unbounded_label_canvas(self):
        from visualization.topic_visualizer import TopicVisualizer
        import matplotlib.image as mpimg
        with tempfile.TemporaryDirectory() as home:
            # Three almost identical topics leave PC2 at numerical round-off scale.
            beta = np.array([[.3, .3, .4], [.3 + 1e-7, .3 - 1e-7, .4], [.3 - 1e-7, .3 + 1e-7, .4]])
            visualizer = TopicVisualizer(output_dir=home, dpi=100, formats=('png',))
            visualizer.visualize_intertopic_distance(np.full((12, 3), 1/3), beta)
            pixels = mpimg.imread(Path(home) / 'Intertopic Distance Map.png')
            self.assertLess(max(pixels.shape[:2]), 2000)

    def test_native_runner_receives_original_matrices_and_workspace(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home) / 'job'; root.mkdir()
            workspace = Path(home) / 'workspace'; workspace.mkdir()
            (workspace / 'vocab.json').write_text('["refund", "delivery"]')
            theta = np.array([[.8, .2], [.1, .9]])
            beta = np.eye(2)
            np.save(root / 'theta_k2.npy', theta)
            np.save(root / 'beta_k2.npy', beta)
            data = dict(theta=theta, beta=beta, vocab=['refund', 'delivery'])
            source = Path(home) / 'source.csv'
            source.write_text('text,timestamp,channel\nrefund,2026-08-01,web\ndelivery,2026-08-02,phone\n')
            dataset = {'managedPath': str(source), 'sha256': file_hash(source), 'datasetRef': 'fixture'}
            job = {'id': 'job', 'resultHash': tree_hash(root), 'preparedHash': file_hash(source),
                   'plan': {'modelId': 'lda', 'textColumn': 'text', 'covariates': ['channel'], 'params': {'num_topics': 2}}}
            def render(**kwargs):
                self.assertIs(kwargs['data']['theta'], theta)
                self.assertIs(kwargs['data']['beta'], beta)
                self.assertEqual(kwargs['data']['timestamps'][0].year, 2026)
                self.assertEqual(list(kwargs['data']['dimension_values']), ['web', 'phone'])
                output = kwargs['output_dir']; output.mkdir(parents=True)
                (output / 'topic_table.csv').write_text('topic,words\n1,refund\n2,delivery\n')
                print('[Skip] No training history available')
            with patch.object(native, 'load_baseline_data', return_value=data) as loader, patch.object(native, 'run_baseline_visualization', side_effect=render) as runner:
                report = generate_report(root, job, Path(home) / 'reports', workspace=workspace, dataset=dataset, prepared=source)
                self.assertEqual(loader.call_args.kwargs['workspace_dir'], workspace)
                self.assertEqual(report['schemaVersion'], 'theta.result-report.v2')
                self.assertTrue(any('training history' in item for item in report['missingEvidence']))
                self.assertEqual(tree_hash(root), job['resultHash'])
                for item in report['files']:
                    self.assertEqual(file_hash(Path(item['path'])), item['sha256'])
                self.assertEqual(generate_report(root, job, Path(home) / 'reports', workspace=workspace, dataset=dataset, prepared=source)['reportPath'], report['reportPath'])
                self.assertEqual(runner.call_count, 1)
                # Style-only changes must invalidate the report, just like renderer changes.
                style = Path(native.__file__).with_name('publication.py').resolve()
                self.assertIn(str(style), report['inputSignatures'])
                original_hash = file_hash
                with patch('workers.result_report.file_hash', side_effect=lambda file: 'changed-style' if Path(file).resolve() == style else original_hash(file)):
                    refreshed = generate_report(root, job, Path(home) / 'reports', workspace=workspace, dataset=dataset, prepared=source)
                self.assertNotEqual(refreshed['reportPath'], report['reportPath'])
                self.assertEqual(runner.call_count, 2)
                np.save(root / 'theta_k2.npy', np.eye(2))
                with self.assertRaisesRegex(ValueError, 'changed'):
                    generate_report(root, job, Path(home) / 'reports', workspace=workspace, dataset=dataset, prepared=source)

    def test_incomplete_bertopic_delivers_paths_without_inventing_assignments_or_figures(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home) / 'job'; root.mkdir()
            np.save(root / 'theta_k2.npy', np.array([[.4, .1], [.1, .4]]))
            np.save(root / 'beta_k2.npy', np.eye(2))
            (root / 'vocab.json').write_text('["refund", "delivery"]')
            job = {'id': 'job', 'resultHash': tree_hash(root), 'plan': {'modelId': 'bertopic', 'params': {'num_topics': 2}}}
            with patch.object(native, 'run_baseline_visualization', side_effect=AssertionError('unsafe visualization')) as runner:
                report = generate_report(root, job, Path(home) / 'reports')
            runner.assert_not_called()
            self.assertEqual(report['reportStatus'], 'incomplete')
            self.assertIn('document_topics.npy', ' '.join(report['missingEvidence']))
            self.assertEqual(report['evidence']['matrices'], [])
            self.assertEqual(report['evidence']['figures'], [])
            self.assertIsNone(report['sourceData'])
            self.assertEqual(report['resultDir'], str(root.resolve()))
            self.assertTrue(Path(report['logPath']).is_file())
            self.assertIn('结果整理未完成', Path(report['reportPath']).read_text())
            self.assertTrue(any(item['name'] == 'training/theta_k2.npy' for item in report['files']))
            self.assertEqual(tree_hash(root), job['resultHash'])

    def test_nvdm_dispatch_reuses_native_charts_without_probability_visuals(self):
        import io
        from contextlib import redirect_stdout
        from visualization import visualization_generator, topic_visualizer
        data = {'theta': np.array([[-1., 2.], [0., -2.]]), 'beta': np.eye(2),
                'vocab': ['refund', 'delivery'], 'topic_words': [(0, [('refund', 1.)]), (1, [('delivery', 1.)])]}
        with tempfile.TemporaryDirectory() as home, patch.object(visualization_generator, 'VisualizationGenerator') as generator, patch.object(topic_visualizer, 'TopicVisualizer') as visualizer, patch.object(topic_visualizer, 'generate_pyldavis_visualization') as pyldavis, redirect_stdout(io.StringIO()):
            native.run_baseline_visualization(result_dir=home, dataset='fixture', model='nvdm', num_topics=2, language='en', output_dir=Path(home)/'report', data=data)
            generator.return_value.generate_all.assert_not_called()
            self.assertEqual(generator.return_value.generate_topic_word_importance.call_count, 2)
            visualizer.return_value.visualize_topic_similarity.assert_called_once()
            visualizer.return_value.visualize_intertopic_distance.assert_not_called(); pyldavis.assert_not_called()
            self.assertIn('not probability', (Path(home)/'report/README.md').read_text())

    def test_existing_csv_generator_labels_latent_means_and_keeps_probability_default(self):
        from visualization.visualization_generator import VisualizationGenerator
        import pandas as pd
        with tempfile.TemporaryDirectory() as home:
            generator = VisualizationGenerator(
                theta=np.array([[-1., 2.], [-3., 0.]]), beta=np.eye(2),
                vocab=['refund', 'delivery'],
                topic_words=[(0, [('refund', 1.)]), (1, [('delivery', 1.)])],
                output_dir=home, language='en', formats=['svg'])
            generator.generate_topic_table(strength_label='mean_latent_coordinate')
            data = pd.read_csv(generator.global_dir/'topic_table.csv')
            self.assertEqual(data['mean_latent_coordinate'].tolist(), [-2., 1.])
            self.assertNotIn('strength', data.columns)
            generator.generate_topic_table()
            self.assertIn('strength', pd.read_csv(generator.global_dir/'topic_table.csv').columns)

    def test_filtered_dtm_rows_use_original_source_indices_and_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)/'job';root.mkdir()
            workspace=Path(home)/'workspace';workspace.mkdir()
            theta=np.array([[.8,.2],[.1,.9]])
            np.save(root/'theta_k2.npy',theta);np.save(root/'beta_k2.npy',np.eye(2))
            (workspace/'vocab.json').write_text('["refund","shipping"]')
            (workspace/'texts.json').write_text('["refund","shipping"]')
            np.save(workspace/'source_rows.npy',np.array([0,2]))
            source=Path(home)/'source.csv';source.write_text('text,year\nrefund,2024\nomitted,\nshipping,2025\n')
            dataset={'managedPath':str(source),'sha256':file_hash(source),'datasetRef':'fixture'}
            job={'id':'job','resultHash':tree_hash(root),'preparedHash':file_hash(source),
                 'plan':{'modelId':'dtm','textColumn':'text','timeColumn':'year','params':{'num_topics':2}}}
            data={'theta':theta,'beta':np.eye(2),'vocab':['refund','shipping']}
            def render(**kwargs):
                output=kwargs['output_dir'];output.mkdir(parents=True)
                (output/'topic_table.csv').write_text('topic,word\n1,refund\n2,shipping\n')
            with patch.object(native,'load_baseline_data',return_value=data),patch.object(native,'run_baseline_visualization',side_effect=render):
                report=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertTrue(report['sourceData']['matrixRowsAligned'])
                self.assertEqual([r['sourceRow'] for r in report['sourceData']['rows']],[1,3])
                self.assertEqual(report['sourceData']['excludedSourceRows'],1)
                # Same row count cannot hide reordered metadata.
                np.save(workspace/'source_rows.npy',np.array([2,0]))
                changed=generate_report(root,job,Path(home)/'reports',workspace=workspace,dataset=dataset,prepared=source)
                self.assertNotEqual(changed['reportPath'],report['reportPath'])
                self.assertFalse(changed['sourceData']['matrixRowsAligned'])
                self.assertNotIn('topicWeights',changed['sourceData']['rows'][0])
