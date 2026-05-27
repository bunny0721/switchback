"""Performance evaluation utilities for switchback experiments.

Provides:

* :func:`true_effect` — the per-period long-run treatment effect of a DGP,
  obtained by paired-noise sampling under sustained treatment vs.
  sustained control. The natural reference for measuring an estimator's
  bias when the design's burn-in is shorter than the DGP's carryover.
* :func:`normality_diagnostics` — moments + SEs for an MC estimate's
  distribution (informal normality check).
* :func:`qq_compare` — empirical vs. normal-fit quantiles.
"""

from switchback.evaluation.diagnostics import (
    normality_diagnostics,
    qq_compare,
)
from switchback.evaluation.truth import true_effect

__all__ = ["true_effect", "normality_diagnostics", "qq_compare"]
