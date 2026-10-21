# Scope and correspondence to the paper

This small repository implements the **mathematical scoring stage** of the
white-box method described in Sections 3.1–3.3 and Algorithm 1 of
*Sample Transform Cost-Based Training-Free Hallucination Detector for Large
Language Models* (NeurIPS 2026).

| Stage | Paper | This repository |
| --- | --- | --- |
| K sampled responses for the same prompt | Eq. (1) | Inputs provided by caller |
| Generated-token hidden states at the middle layer | Sec. 3.1 | Not extracted here |
| Exclude prompt/EOS/padding | Sec. 3.1 | Explicit boolean-mask helper available; caller must construct mask correctly |
| Shared projection to 128 dimensions | Sec. 3.1 / Appendix A | Matrix-application helper available; experimental matrix/seed not supplied |
| Uniform empirical token distributions | Eq. (4) | Implemented |
| Exact squared-Euclidean OT and square root | Eqs. (5)–(7) | Implemented with POT's `ot.emd2` |
| AvgWD | Eq. (8) | Implemented |
| Gaussian kernel, singular values and EigenWD | Eqs. (9)–(13) | Formula-level implementation |
| Target/auxiliary-model teacher forcing | Sec. 2.4 / App. C | Not included |
| Dataset/labeling/judge/baselines/AUROC | Sec. 4 / App. C/H | Not included |

## Items still under verification

1. **Diagonal shift (`alpha`)**: the manuscript specifies `alpha > 0` but
   does not state a particular number. The value is therefore **required** in
   this reference API. `alpha=1e-3` in the demo is illustrative only.
2. **Kernel positive-definiteness repair**: the collected original research
   script applies an additional adaptive diagonal shift to the Gaussian
   kernel. This reference follows the fixed-shift formula in the paper and
   does not apply the extra repair. This can change EigenWD; original
   experimental provenance needs verification before claiming numerical
   reproduction.
3. **Projection**: a shared 128-dimensional random projection is documented
   in the paper. Its independent random seed and full matrix-generation
   protocol have not been verified from the paper, so this repository accepts
   an explicitly supplied projection matrix rather than assuming one.
4. **Token alignment**: true response-token positions are model- and
   tokenizer-dependent. This repository does not guess them from numeric
   token-ID ranges.

This repository has **no pretrained weights, benchmark data, API credentials,
labeling pipeline, or tabulated AUROC outputs**. It cannot independently
reproduce any specific AUROC value from the paper.
