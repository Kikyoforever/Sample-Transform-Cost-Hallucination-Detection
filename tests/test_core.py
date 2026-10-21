"""CPU-only unit tests that do not need real LLMs or external network access."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.embeddings import apply_shared_projection, select_response_tokens
from src.metrics import avgwd, eigenwd, scores_from_embeddings
from src.wasserstein import pairwise_wasserstein_matrix


class EmbeddingsTests(unittest.TestCase):
    def test_explicit_mask_discards_prompt_eos_and_padding(self):
        hidden = np.arange(12, dtype=float).reshape(6, 2)
        selected = select_response_tokens(
            hidden,
            np.array([False, False, True, True, False, False]),
        )
        np.testing.assert_array_equal(selected, hidden[2:4])

    def test_rejects_invalid_masks(self):
        hidden = np.ones((4, 2))
        for mask in [[0, 0, 1, 1], [False, False, False, False], [True, False]]:
            with self.assertRaises(ValueError):
                select_response_tokens(hidden, np.asarray(mask))

    def test_shared_projection(self):
        embeddings = [np.array([[1., 2.]]), np.array([[3., 4.]])]
        projected = apply_shared_projection(embeddings, np.array([[2.], [1.]]))
        np.testing.assert_allclose(projected[0], [[4.]])
        np.testing.assert_allclose(projected[1], [[10.]])


class MetricsTests(unittest.TestCase):
    def test_avgwd_average_unordered_pairs(self):
        d = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]], dtype=float)
        self.assertAlmostEqual(avgwd(d), 8 / 3)

    def test_eigenwd_matches_paper_formula(self):
        d = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
        p, alpha, eps = 0.1, 1e-3, 1e-6
        b = np.median([1., 3., 2.])
        kernel = np.exp(-d ** 2 / (2 * (b ** 2 + eps))) + alpha * np.eye(3)
        sigma = np.linalg.svd(kernel, compute_uv=False)
        expected = float(np.sum(sigma ** p) ** (1 / p) / np.linalg.norm(sigma))
        self.assertAlmostEqual(eigenwd(d, alpha=alpha, p=p, eps=eps), expected)

    def test_identical_responses_have_zero_avgwd(self):
        d = np.zeros((3, 3))
        self.assertEqual(avgwd(d), 0.0)
        self.assertTrue(np.isfinite(eigenwd(d, alpha=1e-3)))

    def test_eigenwd_requires_alpha(self):
        with self.assertRaises(TypeError):
            eigenwd(np.zeros((2, 2)))

    def test_bad_distance_matrices(self):
        with self.assertRaises(ValueError):
            avgwd(np.array([[0, 1], [2, 0]], dtype=float))
        with self.assertRaises(ValueError):
            eigenwd(np.array([[0, -1], [-1, 0]], dtype=float), alpha=1e-3)
        with self.assertRaises(ValueError):
            avgwd(np.array([[0, np.nan], [np.nan, 0]]))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            eigenwd(np.zeros((2, 2)), alpha=0)
        with self.assertRaises(ValueError):
            eigenwd(np.zeros((2, 2)), alpha=1e-3, p=2)


class MockedTransportTests(unittest.TestCase):
    def test_exact_ground_cost_and_square_root_one_token(self):
        def fake_emd2(a, b, cost):
            np.testing.assert_allclose(a, [1.])
            np.testing.assert_allclose(b, [1.])
            return float(cost[0, 0])

        stub = types.SimpleNamespace(emd2=fake_emd2)
        cloud_list = [np.array([[0.]]), np.array([[2.]]), np.array([[4.]])]
        with patch.dict(sys.modules, {"ot": stub}):
            d = pairwise_wasserstein_matrix(cloud_list)
            score_a, score_e = scores_from_embeddings(cloud_list, alpha=1e-3)
        np.testing.assert_allclose(d, [[0, 2, 4], [2, 0, 2], [4, 2, 0]])
        self.assertAlmostEqual(score_a, 8 / 3)
        self.assertTrue(np.isfinite(score_e))

    def test_variable_length_weights_are_uniform(self):
        def fake_emd2(a, b, cost):
            np.testing.assert_allclose(a, [0.5, 0.5])
            np.testing.assert_allclose(b, [1.0])
            np.testing.assert_allclose(cost, [[1.], [1.]])
            return float(np.sum(a * cost[:, 0]))

        stub = types.SimpleNamespace(emd2=fake_emd2)
        with patch.dict(sys.modules, {"ot": stub}):
            d = pairwise_wasserstein_matrix([
                np.array([[0.], [2.]]),
                np.array([[1.]])
            ])
        self.assertAlmostEqual(d[0, 1], 1.0)

    def test_invalid_embedding_cloud(self):
        with self.assertRaises(ValueError):
            pairwise_wasserstein_matrix([np.ones((1, 3)), np.empty((0, 3))])


if __name__ == "__main__":
    unittest.main()
