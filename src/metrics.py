"""Core AvgWD and EigenWD formulas from the NeurIPS 2026 manuscript.

This is a transparent, formula-level reference, not a verified recreation of
all numerical choices in the archived experimental code. In particular, the
reference EigenWD performs the explicitly described fixed diagonal shift but
NOT an additional adaptive positive-definiteness repair.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .wasserstein import pairwise_wasserstein_matrix


def _validate_distances(matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[0] != d.shape[1]:
        raise ValueError("Distance matrix must be square with K >= 2.")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(d, d.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(d), 0, atol=1e-10, rtol=0):
        raise ValueError("Distance matrix must have a zero diagonal.")
    return d


def avgwd(distance_matrix: np.ndarray) -> float:
    """Mean W2 distance over unordered response pairs (Eq. 8)."""
    d = _validate_distances(distance_matrix)
    return float(d[np.triu_indices(d.shape[0], k=1)].mean())


def eigenwd(
    distance_matrix: np.ndarray,
    *,
    alpha: float,
    p: float = 0.1,
    eps: float = 1e-6,
) -> float:
    """Gaussian-kernel singular-value quasi-norm ratio (Eqs. 9–13).

    The kernel bandwidth is median({D_ij > 0}) over ALL positive entries,
    defaulting to 1.0 if there are no positive entries. A fixed ``alpha`` is
    added to the diagonal. In the supplied paper ``p=0.1`` and ``eps=1e-6``,
    but the numeric value of ``alpha`` is not stated: it is required explicitly.

    No adaptive PSD repair is performed. Kernel singular values (not signed
    eigenvalues) are used, so the result is defined for indefinite kernels.
    """
    d = _validate_distances(distance_matrix)
    if not np.isfinite(p) or not 0.0 < p < 2.0:
        raise ValueError("p must be finite and satisfy 0 < p < 2.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")

    positive = d[d > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-(d * d) / (2.0 * (bandwidth * bandwidth + eps)))
    kernel += alpha * np.eye(d.shape[0], dtype=np.float64)

    singular_values = np.linalg.svd(kernel, compute_uv=False)
    numerator = float(np.sum(singular_values ** p) ** (1.0 / p))
    denominator = float(np.linalg.norm(singular_values, ord=2))
    if denominator == 0.0:
        raise ArithmeticError("Unexpected zero singular-value norm.")
    return numerator / denominator


def scores_from_embeddings(
    embeddings: Sequence[np.ndarray],
    *,
    alpha: float,
    p: float = 0.1,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Compute (AvgWD, EigenWD) from prepared continuation-token embeddings."""
    d = pairwise_wasserstein_matrix(embeddings)
    return avgwd(d), eigenwd(d, alpha=alpha, p=p, eps=eps)
