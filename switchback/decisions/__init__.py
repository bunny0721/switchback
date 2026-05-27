"""Decision rules: variance estimation, confidence intervals, ship/no-ship.

Inference (variance + CI) lives here, not on the estimator: the estimator
returns only a point estimate, and inference takes (design, estimator)
together. By default, inference is *design-based* — the variance comes
from the design's randomization distribution rather than a model on the
outcomes.

The recommended front door is :func:`inference`, which dispatches to
the appropriate variance estimator based on the design family and
returns a point estimate, variance, and confidence interval in one
call:

* :class:`switchback.design.AdaptiveBlockDesign` → :func:`block_variance`
  (eq. 28 with block-0 boundary fixes, per-pair γ matrix, factor-2
  mirror iteration).
* :class:`switchback.design.BernoulliDesign` /
  :class:`switchback.design.CompleteRandomization` → :class:`HACVariance`
  (Newey-West HAC on per-window influence sequence). Rejects
  :class:`AdaptiveBlockDesign` at construction.

The underlying pieces (:func:`block_variance`, :class:`HACVariance`,
:func:`normal_ci`) are still exposed for users who want to compute the
intermediate quantities directly.
"""

from switchback.decisions.block_variance import (
    block_confidence_interval,
    block_variance,
)
from switchback.decisions.hac_variance import (
    HACVariance,
    confidence_interval,
    normal_ci,
)
from switchback.decisions.inference import InferenceResult, inference

__all__ = [
    "inference",
    "InferenceResult",
    "block_variance",
    "block_confidence_interval",
    "HACVariance",
    "confidence_interval",
    "normal_ci",
]
