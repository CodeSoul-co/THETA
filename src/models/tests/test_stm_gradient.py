"""Numerical regression for the shared STM engine; no worker job or API call."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import softmax
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.baseline.stm import _document_gradient


class STMGradientTests(unittest.TestCase):
    def test_gradient_matches_finite_difference_of_document_log_posterior(self):
        beta = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        bow = np.array([14., 2., 1.])
        eta = np.array([.4, -.2])
        mean = np.array([.1, .2])
        precision = np.array([[2., .1], [.1, 1.]])
        def objective(value):
            theta = softmax(np.append(value, 0.))
            diff = value - mean
            return bow @ np.log(theta @ beta) - .5 * diff @ precision @ diff
        epsilon = 1e-6
        numerical = np.array([(objective(eta + epsilon * axis) - objective(eta - epsilon * axis)) / (2 * epsilon) for axis in np.eye(2)])
        analytic = _document_gradient(bow, beta, softmax(np.append(eta, 0.)), eta, mean, precision)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-6)

    def test_distinct_word_evidence_breaks_uniform_initialization(self):
        beta = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        gradient = _document_gradient(np.array([20., 0., 0.]), beta, np.ones(3)/3, np.zeros(2), np.zeros(2), np.eye(2))
        self.assertGreater(gradient[0], 0)
        self.assertLess(gradient[1], 0)
