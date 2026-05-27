"""Decision rules: variance estimation, confidence intervals, ship/no-ship.

The decisions module estimates the design-based variance of a point
estimate and constructs a normal-approximation confidence interval —
the statistical machinery you need to make ship/no-ship calls from a
switchback experiment.

The recommended front door is :func:`decide`, which dispatches to the
appropriate variance estimator based on the design family and returns a
point estimate, variance, and confidence interval in one call:

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
from switchback.decisions.decide import DecisionResult, decide

__all__ = [
    "decide",
    "DecisionResult",
    "block_variance",
    "block_confidence_interval",
    "HACVariance",
    "confidence_interval",
    "normal_ci",
]
