import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/models'))
from visualization.publication import setup_style, save_figure
from matplotlib import pyplot as plt


class AdaptivePublicationTests(unittest.TestCase):
    def test_stm_large_group_preview_keeps_full_downloadable_data(self):
        import pandas as pd
        from visualization.run_visualization import _run_stm_specific_visualizations
        data = {'theta': np.ones((60, 10)) / 10,
                'covariates': np.arange(60).reshape(-1, 1),
                'covariate_names': ['来源'],
                'covariate_value_labels': {i: f'来源{i}' for i in range(60)}}
        figures = []
        def capture(fig, *args, **kwargs):
            figures.append(fig)
            self.assertLessEqual(fig.get_size_inches()[1], 13)
            title = fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title()
            self.assertIn('完整数据见CSV', title)
        with tempfile.TemporaryDirectory() as folder, patch('visualization.run_visualization.save_figure', side_effect=capture):
            _run_stm_specific_visualizations(data, folder, language='zh', dpi=72, formats=('png',))
            table = pd.read_csv(Path(folder) / 'global/stm_group_topic_means.csv')
            self.assertEqual(len(table), 60)
            np.testing.assert_allclose(table['T1'], .1)
            self.assertEqual(table['n_documents'].sum(), 60)
            self.assertEqual(len(figures), 7)

    def test_real_proportions_and_projection_layouts_at_one_hundred_topics(self):
        from visualization.topic_visualizer import TopicVisualizer, draw_document_projection
        setup_style('zh')
        with tempfile.TemporaryDirectory() as folder:
            visualizer = TopicVisualizer(output_dir=folder, language='zh', dpi=72, formats=('svg',))
            with patch.object(visualizer, '_save_or_show', side_effect=lambda fig, filename: fig):
                fig = visualizer.visualize_topic_proportions(np.ones((20, 100)) / 100, filename='主题占比分布')
            fig.canvas.draw()
            labels = sorted([label.get_window_extent() for label in fig.axes[0].get_yticklabels()], key=lambda b: b.y0)
            self.assertEqual(len(labels), 100)
            self.assertTrue(all(a.y1 < b.y0 for a, b in zip(labels, labels[1:])))
            plt.close(fig)
            rng = np.random.default_rng(42)
            fig = draw_document_projection(rng.normal(size=(400, 2)), np.arange(400) % 100, method='UMAP', language='zh', total_count=400)
            fig.canvas.draw()
            legend = fig.legends[0].get_window_extent()
            for ax in fig.axes:
                self.assertLess(legend.y1, ax.get_window_extent().y0)
            self.assertIn('文档主题投影', fig._suptitle.get_text())
            plt.close(fig)

    def test_topic_rows_keep_canvas_and_nonoverlapping_tick_labels(self):
        setup_style('zh')
        with tempfile.TemporaryDirectory() as folder:
            for topics in (10, 20, 100):
                with self.subTest(topics=topics):
                    fig, ax = plt.subplots(figsize=(8, max(4, topics * .22)))
                    ax.barh(np.arange(topics), np.ones(topics))
                    ax.set_yticks(np.arange(topics), [f'主题 {i+1}' for i in range(topics)])
                    ax.set_title('主题占比')
                    original_size = fig.get_size_inches().copy()
                    save_figure(fig, Path(folder) / f'主题占比-{topics}', dpi=72, formats=('svg',))
                    np.testing.assert_allclose(fig.get_size_inches(), original_size)
                    fig.canvas.draw()
                    labels = sorted([label.get_window_extent() for label in ax.get_yticklabels()], key=lambda b: b.y0)
                    self.assertTrue(all(a.y1 < b.y0 for a, b in zip(labels, labels[1:])))
                    plt.close(fig)
