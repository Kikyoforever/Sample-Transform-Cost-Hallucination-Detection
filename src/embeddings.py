"""Input validation and explicit token selection for AvgWD/EigenWD.

The extraction of LLM hidden states, prompt/response token boundaries, and
optional experimental random projections is NOT implemented here. In particular,
we do not infer valid token positions from token IDs (such as ``token_id > 2``).
The caller is responsible for producing an accurate boolean mask.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def select_response_tokens(
    full_hidden_states: np.ndarray,
    valid_response_mask: np.ndarray,
) -> np.ndarray:
    """Select generated, non-EOS, non-padding token states via an explicit mask.

    Args:
        full_hidden_states: Array of shape (sequence_length, hidden_size).
        valid_response_mask: Boolean array of shape (sequence_length,).
            True entries must correspond *only* to valid generated-response tokens.

    The caller must align the mask with the model's tokenization and hidden-state
    indices. This function cannot determine prompt or EOS positions by itself.
    """
    states = np.asarray(full_hidden_states, dtype=np.float64)
    mask = np.asarray(valid_response_mask)
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("full_hidden_states must have shape (sequence_length, d), d >= 1.")
    if not np.isfinite(states).all():
        raise ValueError("full_hidden_states contains NaN or infinity.")
    if mask.ndim != 1 or mask.shape[0] != states.shape[0] or mask.dtype != np.dtype(bool):
        raise ValueError("valid_response_mask must be a boolean vector aligned with sequence_length.")
    if not mask.any():
        raise ValueError("No valid generated tokens remain after masking.")
    return np.ascontiguousarray(states[mask])


def validate_clouds(embeddings: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Validate K>=2 nonempty, same-dimension generated-token point clouds."""
    if len(embeddings) < 2:
        raise ValueError("At least two sampled responses are required (K >= 2).")

    clouds: list[np.ndarray] = []
    embedding_dim: int | None = None
    for idx, sample in enumerate(embeddings):
        cloud = np.asarray(sample, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[0] < 1 or cloud.shape[1] < 1:
            raise ValueError(f"Response {idx} must have shape (T, d), with T, d >= 1.")
        if not np.isfinite(cloud).all():
            raise ValueError(f"Response {idx} contains NaN or infinity.")
        if embedding_dim is None:
            embedding_dim = cloud.shape[1]
        elif cloud.shape[1] != embedding_dim:
            raise ValueError("All sampled responses must use the same embedding dimension.")
        clouds.append(np.ascontiguousarray(cloud))
    return clouds


def apply_shared_projection(
    embeddings: Sequence[np.ndarray], projection_matrix: np.ndarray
) -> list[np.ndarray]:
    """Apply one user-supplied projection matrix to ALL sampled responses.

    The caller supplies a matrix of shape (input_dim, output_dim); this module
    does not guess the random distribution, scaling, or seed used in experiments.
    """
    clouds = validate_clouds(embeddings)
    matrix = np.asarray(projection_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != clouds[0].shape[1] or matrix.shape[1] < 1:
        raise ValueError("Projection matrix shape must be (input_dim, output_dim).")
    if not np.isfinite(matrix).all():
        raise ValueError("Projection matrix contains NaN or infinity.")
    return [np.ascontiguousarray(cloud @ matrix) for cloud in clouds]
