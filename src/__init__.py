"""Formula-level reference implementation of AvgWD and EigenWD."""

from .embeddings import apply_shared_projection, select_response_tokens, validate_clouds
from .wasserstein import pairwise_wasserstein_matrix
from .metrics import avgwd, eigenwd, scores_from_embeddings

__all__ = [
    "apply_shared_projection",
    "select_response_tokens",
    "validate_clouds",
    "pairwise_wasserstein_matrix",
    "avgwd",
    "eigenwd",
    "scores_from_embeddings",
]
