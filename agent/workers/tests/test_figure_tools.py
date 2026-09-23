"""Regression: title edits replay native plots, never infer a chart from CSV columns."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from workers.capabilities import engine_root
from workers.figure_tools import adjust
from workers.results_reader import file_hash, tree_hash

sys.path.insert(0, str(engine_root() / 'src/models'))
from visualization.topic_visualizer import TopicVisualizer
from visualization.visualization_generator import VisualizationGenerator
from visualization.publication import selected_chart
from matplotlib import pyplot as plt

NS = {'s': 'http://www.w3.org/2000/svg'}


class FigureToolsTests(unittest.TestCase):
    def test_wordcloud_page_and_network_reuse_native_code_and_exact_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            model = home / 'model'
            model.mkdir()
            rng = np.random.default_rng(42)
            theta = rng.dirichlet(np.ones(20), 50)
            beta = rng.dirichlet(np.ones(12), 20)
            vocab = [f'word{index}' for index in range(12)]
            np.save(model / 'theta_k20.npy', theta)
            np.save(model / 'beta_k20.npy', beta)
            (model / 'vocab.json').write_text(json.dumps(vocab))
            report = home / 'report'
            native = report / 'native'
            visualizer = TopicVisualizer(output_dir=str(native / 'global'), language='zh', formats=('svg',))
            words = [(i, [(vocab[j], float(row[j])) for j in np.argsort(-row)]) for i, row in enumerate(beta)]
            grids = visualizer.visualize_wordcloud_grid(words)
            generator = VisualizationGenerator(theta, beta, vocab, words, output_dir=native, language='zh', formats=('svg',))
            geometry = []
            native_save = VisualizationGenerator._save
            def capture(instance, filename, **kwargs):
                geometry.append([[(patch.get_path().vertices.tolist(), patch.get_edgecolor(), patch.get_linestyle())
                                  for patch in ax.patches] for ax in plt.gcf().axes])
                return native_save(instance, filename, **kwargs)
            with patch.object(VisualizationGenerator, '_save', capture):
                generator.generate_topic_network()
            (native / 'additional-chart-status.json').write_text(json.dumps([
                {'chart': 'wordcloud_grid', 'status': 'generated', 'files': [str(Path(f).relative_to(native)) for f in grids]}]))
            (native / 'chart-status.json').write_text(json.dumps([
                {'chart': 'topic_network', 'status': 'generated', 'files': generator.exported_files}]))
            (report / 'manifest.json').write_text(json.dumps({
                'modelId': 'prodlda', 'resultDir': str(model), 'resultHash': tree_hash(model),
                'inputSignatures': {str(model / 'vocab.json'): file_hash(model / 'vocab.json')},
                'nativeRunner': 'src/models/visualization/run_visualization.py',
                'trainingPlan': {'params': {'num_topics': 20}},
            }))
            grid = 'native/global/topic_wordcloud_grid_3.svg'
            before = file_hash(report / grid)
            # Calling a different chart is itself a failure, even if its output is discarded.
            with patch.object(VisualizationGenerator, 'generate_topic_network', side_effect=AssertionError('wrong chart')):
                result = adjust({'reportDir': str(report), 'figure': grid, 'spec': {'title': 'English grid title'}})
            original = ET.parse(report / grid)
            edited = ET.parse(Path(result['path']).with_suffix('.svg'))
            images = lambda tree: [image.get('{http://www.w3.org/1999/xlink}href') for image in tree.findall('.//s:image', NS)]
            texts = lambda tree: [text.text for text in tree.findall('.//s:text', NS)]
            self.assertEqual(len(images(edited)), 6)
            self.assertEqual(images(original), images(edited))
            self.assertEqual(texts(original)[:-1], texts(edited)[:-1])
            self.assertEqual(texts(edited)[-1], 'English grid title')
            self.assertIn('主题 13', texts(edited))
            self.assertIn('主题 18', texts(edited))
            self.assertEqual(file_hash(report / grid), before)
            self.assertEqual(len(result['paths']), 3)
            self.assertEqual(len(list(Path(result['path']).parent.glob('*.png'))), 1)
            self.assertTrue(selected_chart('anything'), 'Edit scope must not leak into subsequent renders')
            again = adjust({'reportDir': str(report), 'figure': result['path'], 'spec': {'title': 'Revised title'}})
            self.assertEqual(again['originalFigure'], grid)
            self.assertIn('Revised title', texts(ET.parse(Path(again['path']).with_suffix('.svg'))))
            network = generator.exported_files[0]
            with patch.object(TopicVisualizer, 'visualize_wordcloud_grid', side_effect=AssertionError('wrong chart')), patch.object(VisualizationGenerator, '_save', capture):
                result = adjust({'reportDir': str(report), 'figure': network, 'spec': {'title': 'Topic Correlation Network'}})
            edited = ET.parse(Path(result['path']).with_suffix('.svg'))
            self.assertEqual(len(geometry), 2)
            self.assertTrue(geometry[0][0], 'Fixture must contain network edges')
            self.assertEqual(geometry[0], geometry[1], 'Network edges, positions and colors must remain identical')
            self.assertIn('Topic Correlation Network', texts(edited))
            with self.assertRaisesRegex(ValueError, 'CSV|kind'):
                adjust({'reportDir': str(report), 'figure': grid, 'spec': {'title': 'Wrong', 'kind': 'bar'}})
            with self.assertRaisesRegex(ValueError, '不属于'):
                adjust({'reportDir': str(report), 'figure': '../model/vocab.json', 'spec': {'title': 'Wrong'}})
            np.save(model / 'theta_k20.npy', theta * 2)
            with self.assertRaisesRegex(ValueError, '已变化'):
                adjust({'reportDir': str(report), 'figure': grid, 'spec': {'title': 'Wrong'}})


if __name__ == '__main__':
    unittest.main()
