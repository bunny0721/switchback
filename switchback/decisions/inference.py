"""Full design-based inference pipeline: estimate + variance + CI.

This module exposes :func:`inference`, the one-call front door for users
who want a point estimate, a design-based variance, and a
normal-approximation confidence interval in one go.

The function **dispatches** to the appropriate variance estimator based
on the design family:

* :class:`AdaptiveBlockDesign` → :func:`block_variance` (eq. 28 with
  block-0 boundary fixes, per-pair γ matrix, factor-2 mirror).
* :class:`BernoulliDesign`, :class:`CompleteRandomization` →
  :class:`HACVariance` (Newey-West HAC on per-window influence
  sequence).

Returns an :class:`InferenceResult` dataclass with ``.estimate``,
``.variance``, ``.ci``, ``.alpha`` attributes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from switchback.design.adaptive_block import AdaptiveBlockDesign
from switchback.design.base import BaseDesign
from switchback.estimators.base import BaseEstimator
from switchback.decisions.block_variance import block_variance
from switchback.decisions.hac_variance import HACVariance, normal_ci


@dataclass
class InferenceResult:
    """Output container for :func:`inference`.

    Attributes
    ----------
    estimate : float
        Point estimate τ̂ from the estimator.
    variance : float
        Variance estimate V̂(τ̂) from the design-dispatched variance
        estimator.
    ci : Tuple[float, float]
        Normal-approximation confidence interval at the given alpha
        level: ``(τ̂ − z·√V̂, τ̂ + z·√V̂)`` with ``z = Φ^{-1}(1 − α/2)``.
    alpha : float
        Nominal significance level used to compute ``ci`` (so the
        nominal coverage is ``1 − alpha``).
    """

    estimate: float
    variance: float
    ci: Tuple[float, float]
    alpha: float


def inference(
    design: BaseDesign,
    estimator: BaseEstimator,
    assignment: np.ndarray,
    outcomes: np.ndarray,
    alpha: float = 0.05,
) -> InferenceResult:
    r"""Full design-based inference pipeline in one call.

    Computes the point estimate from the user's estimator, then the
    design-derived variance via the appropriate dispatch, then a
    normal-approximation CI at level ``alpha``.

    Variance dispatch:

    * :class:`AdaptiveBlockDesign` → :func:`block_variance` (eq. 28 with
      block-0 boundary fixes, per-pair γ⁺/γ⁻ matrix from the Class A/B/C
      structural classification, and factor-2 mirror at forward
      δ ≤ B/2).
    * :class:`BernoulliDesign`, :class:`CompleteRandomization` →
      :class:`HACVariance` (Newey-West HAC, default ``L = 1``, truncated
      kernel).

    The estimator object is **deep-copied** internally, so the user's
    estimator is not mutated.

    Parameters
    ----------
    design : BaseDesign
        The randomisation design. Determines variance dispatch.
    estimator : BaseEstimator
        Point estimator (typically :class:`IPWEstimator` or
        :class:`HajekEstimator`).
    assignment : np.ndarray
        Realised W of length T.
    outcomes : np.ndarray
        Realised Y of length T.
    alpha : float, default 0.05
        Nominal significance level (CI has nominal coverage ``1 − α``).
        Must satisfy ``0 < α < 1``.

    Returns
    -------
    InferenceResult
        Dataclass with attributes ``estimate``, ``variance``, ``ci``,
        ``alpha``.

    Examples
    --------
    AdaptiveBlockDesign workflow:

    >>> from switchback.design import AdaptiveBlockDesign
    >>> from switchback.estimators import IPWEstimator
    >>> from switchback.decisions import inference
    >>> design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    >>> W = design.sample(672)
    >>> Y = ...  # outcomes
    >>> result = inference(design, IPWEstimator(design, m=1), W, Y, alpha=0.05)
    >>> result.estimate, result.variance, result.ci  # doctest: +SKIP

    BernoulliDesign workflow (dispatches to HAC):

    >>> from switchback.design import BernoulliDesign
    >>> design = BernoulliDesign(window_length=4, seed=0)
    >>> W = design.sample(200)
    >>> result = inference(design, IPWEstimator(design, m=3), W, Y, alpha=0.05)
    """
    W = np.asarray(assignment, dtype=int)
    Y = np.asarray(outcomes, dtype=float)

    if isinstance(design, AdaptiveBlockDesign):
        # Point estimate via a private copy of the estimator
        est = copy.deepcopy(estimator)
        est.fit(W, Y)
        tau_hat = float(est.estimate_)
        # Design-derived variance via eq. 28 + boundary fixes
        v_hat = float(block_variance(design, W, Y))
    else:
        # HACVariance.fit() handles both pieces (estimate + variance) and
        # deep-copies the estimator internally
        inf = HACVariance(design, estimator).fit(W, Y)
        tau_hat = float(inf.estimate_)
        v_hat = float(inf.variance_)

    ci = normal_ci(tau_hat, v_hat, alpha)
    return InferenceResult(
        estimate=tau_hat, variance=v_hat, ci=ci, alpha=alpha
    )
