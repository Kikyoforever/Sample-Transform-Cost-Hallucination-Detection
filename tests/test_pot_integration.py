"""Integration checks with REAL POT; run after installing requirements.txt."""

import importlib.util
import unittest

import numpy as np

from src.wasserstein import pairwise_wasserstein_matrix


@unittest.skipUnless(importlib.util.find_spec("ot") is not None, "POT is not installed")
class POTIntegrationTests(unittest.TestCase):
    def test_single_token_distances(self):
        d = pairwise_wasserstein_matrix([
            np.array([[0., 0.]]),
            np.array([[3., 4.]]),
        ])
        np.testing.assert_allclose(d, [[0., 5.], [5., 0.]], atol=1e-12)

    def test_different_support_sizes(self):
        # Uniform 2-to-1 transport: 0.5*(0-1)^2 + 0.5*(2-1)^2 = 1.
        d = pairwise_wasserstein_matrix([
            np.array([[0.], [2.]]),
            np.array([[1.]]),
        ])
        np.testing.assert_allclose(d, [[0., 1.], [1., 0.]], atol=1e-12)

    def test_identical_clouds(self):
        response = np.array([[0., 1.], [2., 3.]])
        d = pairwise_wasserstein_matrix([response, response.copy()])
        np.testing.assert_allclose(d, np.zeros((2, 2)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
