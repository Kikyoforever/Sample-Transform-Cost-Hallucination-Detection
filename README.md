# Sample Transform Cost-Based Training-Free Hallucination Detector for Large Language Models

**NeurIPS 2026 · Core Method Available · Full Experimental Code Coming Soon**

This repository accompanies our paper, *Sample Transform Cost-Based Training-Free
Hallucination Detector for Large Language Models* (NeurIPS 2026). We study
training-free hallucination detection using **sample-to-sample Wasserstein
transform costs** in token-level representation space.

> **Release status:** A self-contained, formula-level implementation of the
> **AvgWD** and **EigenWD** scoring functions is available below. The complete
> experimental code (including language-model generation, hidden-state
> extraction, black-box experiments, evaluation, and table reproduction) is
> **being organized and validated — coming soon**. 

## Method overview

For a prompt, consider `K` independently sampled responses. Each response is
represented by a uniform empirical distribution over its **generated-token
hidden states**, excluding the prompt, EOS, and padding tokens. Optionally,
embeddings are projected using the **same** random projection matrix.

1. Compute the pairwise **2-Wasserstein** distances between token-embedding
   distributions via exact optimal transport with squared Euclidean ground
   cost (POT `emd2`, followed by a square root).
2. Compute **AvgWD**, the average distance over all unordered response pairs.
3. Convert the distance matrix into a Gaussian similarity matrix and compute
   **EigenWD**, a spectral-complexity score based on its singular values.

Let `D` be the `K × K` Wasserstein distance matrix. The scores are:

**AvgWD** (average pairwise transform cost):

```math
\mathrm{AvgWD}(x) = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} D_{ij}.
```

**EigenWD** (spectral complexity of the Gaussian-kernel matrix):

```math
S_{ij} = \exp\left(-\frac{D_{ij}^{2}}{2(b^{2}+\varepsilon)}\right),
\qquad S \gets S + \alpha I.
```

Here, `b` is the median of the **strictly positive** entries of `D`, with
`b = 1` when no positive entry exists.

```math
\mathrm{EigenWD}(x) = \frac{\|\sigma(S)\|_{p}}{\|\sigma(S)\|_{2}},
\qquad p = 0.1.
```

In the paper, `epsilon=1e-6`, and `b=1` when there is no positive distance.
The manuscript does **not** specify a numeric `alpha`; the example below
uses `alpha=1e-3` **only as an illustrative value**. See
[implementation notes](docs/IMPLEMENTATION_NOTES.md) for the important
difference between the paper's formula and the archived experimental code's
additional kernel PSD repair.


## Repository layout

```text
.
├── src/
│   ├── embeddings.py           # Validate/select embeddings; apply supplied projection
│   ├── wasserstein.py          # Uniform empirical OT; W2 matrix
│   └── metrics.py              # AvgWD and EigenWD
├── examples/
│   └── minimal_example.py      # Synthetic example (no model download)
├── tests/
│   ├── test_core.py
│   └── test_pot_integration.py
├── docs/
│   └── IMPLEMENTATION_NOTES.md
├── requirements.txt
└── README.md
```

## Quick start

Use Python 3.10 or 3.11 and install dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.minimal_example
```

The example uses **synthetic 2-dimensional vectors** rather than real model
hidden states. No model weights, datasets, or API credentials are needed.

A minimal Python usage example:

```python
import numpy as np
from src import scores_from_embeddings

# One array per independently sampled response to the same prompt.
# Entries must include only valid generated-response token embeddings.
responses = [
    np.array([[0., 0.], [1., 0.]]),
    np.array([[0., 1.], [1., 1.]]),
    np.array([[2., 0.], [3., 0.]]),
]

# Alpha is for demonstration; it is not verified as the original experiment value.
avg_score, eigen_score = scores_from_embeddings(responses, alpha=1e-3)
print(f"AvgWD = {avg_score:.6f}, EigenWD = {eigen_score:.6f}")
```

**Input contract:** Each response is a nonempty `T_i × d` array of finite
numbers; variable response lengths are supported, but every response must
use the same embedding dimension. Prompt, EOS, and padding tokens must be
excluded. In `src/embeddings.py`, an explicit boolean mask is available for
selecting valid response-token rows; the *caller* must correctly construct it
from the actual tokenizer/model positions. The optional projection helper
accepts a user-supplied matrix so that every response uses an identical
transformation. This release does not reconstruct the exact projection seed
or matrix used in the published experiments.



