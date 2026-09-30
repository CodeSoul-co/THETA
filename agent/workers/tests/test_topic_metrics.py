"""Full-corpus evaluation must match exact counts without full D x V copies."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/models'))
from evaluation.topic_metrics import (
    compute_topic_coherence_npmi, compute_topic_coherence_cv,
    compute_topic_coherence_umass, compute_perplexity,
)


class TopicMetricsTest(unittest.TestCase):
    def test_dense_and_sparse_batches_keep_exact_document_statistics(self):
        bow = np.tile(np.array([[2, 1, 0, 0], [1, 0, 3, 0], [0, 2, 0, 1]], dtype=np.float32), (2000, 1))
        beta = np.array([[.6, .3, .08, .02], [.08, .02, .6, .3]])
        theta = np.tile([.7, .3], (len(bow), 1))
        eps = 1e-12
        expected = []
        for i, j in [(0, 1), (2, 3)]:
            pi = ((bow[:, i] > 0).sum() + eps) / len(bow)
            pj = ((bow[:, j] > 0).sum() + eps) / len(bow)
            pij = (np.logical_and(bow[:, i] > 0, bow[:, j] > 0).sum() + eps) / len(bow)
            expected.append(np.log(pij / (pi * pj)) / -np.log(pij))
        ppl = np.exp(-np.sum(bow * np.log(np.clip(theta @ beta, eps, 1))) / bow.sum())
        for matrix in [bow, sparse.csr_matrix(bow)]:
            np.testing.assert_allclose(compute_topic_coherence_npmi(beta, matrix, top_k=2)[1], expected)
            np.testing.assert_allclose(compute_topic_coherence_cv(beta, matrix, top_k=2)[1], expected)
            self.assertAlmostEqual(compute_perplexity(beta, theta, matrix), ppl, places=8)
        np.testing.assert_allclose(compute_topic_coherence_umass(beta, bow, top_k=2)[1], compute_topic_coherence_umass(beta, sparse.csr_matrix(bow), top_k=2)[1])
        self.assertEqual(len(compute_topic_coherence_cv(beta, bow, top_k=10)[1]), 2)
        with self.assertRaises(ValueError): compute_topic_coherence_npmi(beta, bow[:0])
