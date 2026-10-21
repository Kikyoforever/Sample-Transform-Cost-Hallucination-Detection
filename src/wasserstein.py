"""Pairwise exact empirical 2-Wasserstein distances for sampled responses.

Each response is a uniform empirical measure over generated-token embeddings.
See Eqs. (4)–(7) and Algorithm 1 of the accompanying NeurIPS 2026 paper.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from .embeddings import validate_clouds


def pairwise_wasserstein_matrix(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Return a symmetric K-by-K matrix of W2 distances (zero diagonal).

    For each pair: construct squared-Euclidean ground costs, solve the discrete
    optimal-transport problem with POT's exact ``ot.emd2`` routine and take the
    square root of the optimal *squared* W2 cost. Weights are uniform, and
    response lengths may differ.

    This implementation intentionally does not use Sinkhorn regularization.
    """
    clouds = validate_clouds(embeddings)
    try:
        import ot  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("POT is required: install with `pip install -r requirements.txt`.") from exc

    k = len(clouds)
    distances = np.zeros((k, k), dtype=np.float64)
    for i, source in enumerate(clouds):
        a = np.full(source.shape[0], 1.0 / source.shape[0], dtype=np.float64)
        for j in range(i + 1, k):
            target = clouds[j]
            b = np.full(target.shape[0], 1.0 / target.shape[0], dtype=np.float64)
            ground_cost = cdist(source, target, metric="sqeuclidean")
            squared_w2 = float(ot.emd2(a, b, ground_cost))
            if not np.isfinite(squared_w2) or squared_w2 < -1e-9:
                raise ArithmeticError("OT solver returned an invalid squared W2 cost.")
            distance = float(np.sqrt(max(squared_w2, 0.0)))
            distances[i, j] = distances[j, i] = distance
    return distances
