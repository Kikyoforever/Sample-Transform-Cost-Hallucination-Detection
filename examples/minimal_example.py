"""Minimal self-contained example using synthetic generated-token embeddings.

Run from the repository root: python -m examples.minimal_example
This demonstration does not reproduce any NeurIPS benchmark result.
"""

import numpy as np

from src import avgwd, eigenwd, pairwise_wasserstein_matrix


def main() -> None:
    # Each array represents the valid generated tokens for one sampled answer.
    # We intentionally use toy 2D vectors, not actual language-model states.
    clouds = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 0.0], [3.0, 0.0]]),
    ]
    d = pairwise_wasserstein_matrix(clouds)
    print("W2 distance matrix:\n", d)
    print(f"AvgWD: {avgwd(d):.6f}")
    # Alpha is ILLUSTRATIVE: its experimental value is not specified in the PDF.
    print(f"EigenWD (illustrative alpha=1e-3): {eigenwd(d, alpha=1e-3):.6f}")


if __name__ == "__main__":
    main()
