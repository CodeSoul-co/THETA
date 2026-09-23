"""Synthetic trajectories must never be delivered as research evidence."""
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout
import unittest
import tempfile
import numpy as np
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualization.visualization_generator import VisualizationGenerator
from visualization.run_visualization import load_visualization_data
from visualization.topic_visualizer import load_etm_results


class VisualizationEvidenceTests(unittest.TestCase):
    def test_unobserved_trajectories_skip_without_random_values(self):
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as home:
            generator = VisualizationGenerator(
                theta=np.array([[.8, .2], [.1, .9]]), beta=np.eye(2),
                vocab=['refund', 'delivery'],
                topic_words=[(0, [('refund', 1.)]), (1, [('delivery', 1.)])],
                output_dir=home, formats=['svg'])
            with patch('numpy.random.normal', side_effect=AssertionError('fabricated trajectory')), patch('numpy.random.uniform', side_effect=AssertionError('fabricated trajectory')), redirect_stdout(log):
                generator.generate_kl_divergence()
                generator.generate_topic_word_dist_change(0)
                generator.generate_topic_word_sense(0)
        self.assertEqual(log.getvalue().count('[SKIP]'), 3)

    def test_nonfinite_perplexity_does_not_hide_finite_loss_curves(self):
        with tempfile.TemporaryDirectory() as home:
            generator = VisualizationGenerator(
                theta=np.array([[.8, .2], [.1, .9]]), beta=np.eye(2),
                vocab=['refund', 'delivery'],
                topic_words=[(0, [('refund', 1.)]), (1, [('delivery', 1.)])],
                output_dir=home, formats=['svg'],
                training_history={
                    'train_loss': [1.0, .8], 'val_loss': [1.1, .9],
                    'perplexity': [float('inf'), float('inf')],
                },
            )
            generator.generate_training_convergence()
            names = {path.name.lower() for path in Path(home).rglob('*.svg')}
            self.assertIn('training loss.svg', names)
            self.assertNotIn('training perplexity.svg', names)

    def test_theta_mode_directory_and_fixed_artifacts_are_loadable(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            exp = root / 'sample' / '0.6B' / 'theta' / 'exp_test'
            model = exp / 'theta' / 'zero_shot'
            bow = exp / 'data' / 'bow'
            model.mkdir(parents=True)
            bow.mkdir(parents=True)
            theta = np.array([[.8, .2], [.1, .9]])
            beta = np.array([[.7, .3], [.2, .8]])
            np.save(model / 'theta.npy', theta)
            np.save(model / 'beta.npy', beta)
            np.save(model / 'topic_embeddings.npy', np.eye(2))
            (model / 'topic_words.json').write_text(
                '{"0": [["refund", 0.7]], "1": [["delivery", 0.8]]}',
                encoding='utf-8',
            )
            (model / 'vocab.json').write_text('["refund", "delivery"]', encoding='utf-8')
            (model / 'training_history.json').write_text('{"train_loss": [1.0]}', encoding='utf-8')
            loaded = load_visualization_data(
                root, 'sample', 'zero_shot', model_size='0.6B', model_exp='exp_test'
            )
            self.assertEqual(loaded['theta'].shape, (2, 2))
            self.assertEqual(loaded['vocab'], ['refund', 'delivery'])
            fixed = load_etm_results(str(model), timestamp='legacy_timestamp')
            self.assertEqual(fixed['beta'].shape, (2, 2))
