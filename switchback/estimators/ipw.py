"""Inverse-propensity-weighted (IPW) estimator for switchback experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign
from switchback.design.complete import CompleteRandomization
from switchback.estimators.base import BaseEstimator


class IPWEstimator(BaseEstimator):
    r"""Inverse-propensity-weighted estimator with burn-in length ``m``.

    .. math::

        \hat\tau_m = \frac{1}{T-m} \sum_{t=m}^{T-1}
            \left[
                \frac{Y_t \, \mathbf{1}\{W_{t-m:t} = \mathbf{1}_{m+1}\}}
                     {\Pr(W_{t-m:t} = \mathbf{1}_{m+1})}
                -
                \frac{Y_t \, \mathbf{1}\{W_{t-m:t} = \mathbf{0}_{m+1}\}}
                     {\Pr(W_{t-m:t} = \mathbf{0}_{m+1})}
            \right].

    The parameter ``m`` is the *burn-in length*: how many past periods
    must agree with ``W_t`` before the observation counts as fully
    treated (or fully controlled). ``m = 0`` checks only ``W_t`` itself;
    larger ``m`` washes out longer carryover windows but discards more
    data (the sum starts at ``t = m``) and uses smaller propensities.

    The estimator returns only a point estimate; variance and confidence
    intervals live in :mod:`switchback.decisions`.

    Note
    ----
    Under :class:`CompleteRandomization`, the population-propensity form
    above is replaced by the **self-normalised** form (divide each arm's
    sum by the realised arm size rather than by the propensity). With
    fixed-total designs this is the natural choice, and the two forms
    coincide exactly when ``l = 1, m = 0`` (since
    ``n_w / (T - m) = π_w`` deterministically). For other CR
    configurations the self-normalised form is preferable because the
    population propensities are *marginal* (not conditional on the
    realised counts), and using the realised ones removes finite-sample
    variance contributions from ``n_w`` fluctuations. The result is that
    ``IPWEstimator`` and :class:`HajekEstimator` produce identical
    estimates under :class:`CompleteRandomization`.

    The denominators come from the design via
    :meth:`BaseDesign.consecutive_prob`, so the estimator is agnostic to
    which design produced the data.

    Parameters
    ----------
    design : BaseDesign
        The design that produced the assignment path.
    m : int
        Burn-in length. Must be a non-negative integer.

    Attributes
    ----------
    estimate_ : float
        Point estimate :math:`\hat\tau_m`, available after :meth:`fit`.
    """

    def __init__(self, design: BaseDesign, m: int):
        if not isinstance(design, BaseDesign):
            raise TypeError("design must be a BaseDesign instance")
        if not isinstance(m, (int, np.integer)) or m < 0:
            raise ValueError(f"m must be a non-negative integer, got {m!r}")
        l = int(design.l)
        if int(m) > l:
            raise ValueError(
                f"m must be ≤ design.l; got m={m}, "
                f"l={l}. The burn-in cannot exceed the design's "
                f"window length — there's no structural mechanism to "
                f"guarantee m+1 consecutive same-arm periods beyond a window. "
                f"For BernoulliDesign/CompleteRandomization, m={l-1} is the "
                f"natural 'in-window' choice; for AdaptiveBlockDesign "
                f"(l=1), use m=1 (paper's primary form) or m=0."
            )
        self.design = design
        self.m = int(m)
        self.estimate_: Optional[float] = None

    def fit(self, assignment: np.ndarray, outcomes: np.ndarray) -> "IPWEstimator":
        W, Y = self._validate(assignment, outcomes)
        T, m = W.size, self.m
        if T <= m:
            raise ValueError(f"need T > m; got T={T}, m={m}")
        if isinstance(self.design, CompleteRandomization):
            self._fit_self_normalised(W, Y, T, m)
        else:
            self._fit_population_propensity(W, Y, T, m)
        return self

    # Standard Horvitz-Thompson form (population propensity from the design).
    def _fit_population_propensity(
        self, W: np.ndarray, Y: np.ndarray, T: int, m: int
    ) -> None:
        total = 0.0
        # 1-indexed paper sum t = m+1, ..., T  ->  0-indexed t = m, ..., T-1.
        for t in range(m, T):
            window = W[t - m : t + 1]
            if np.all(window == 1):
                total += Y[t] / self.design.consecutive_prob(T, t, m, value=1)
            elif np.all(window == 0):
                total -= Y[t] / self.design.consecutive_prob(T, t, m, value=0)
            # else: indicator is 0 -> term contributes nothing.
        self.estimate_ = total / (T - m)

    # Self-normalised form (Hájek) — used under CompleteRandomization.
    def _fit_self_normalised(
        self, W: np.ndarray, Y: np.ndarray, T: int, m: int
    ) -> None:
        treated_sum, n1 = 0.0, 0
        control_sum, n0 = 0.0, 0
        for t in range(m, T):
            window = W[t - m : t + 1]
            if np.all(window == 1):
                treated_sum += Y[t]
                n1 += 1
            elif np.all(window == 0):
                control_sum += Y[t]
                n0 += 1
        if n1 == 0:
            raise ValueError(
                "No fully-treated periods (all-1 windows) observed; "
                "cannot compute self-normalised IPW estimator."
            )
        if n0 == 0:
            raise ValueError(
                "No fully-controlled periods (all-0 windows) observed; "
                "cannot compute self-normalised IPW estimator."
            )
        self.estimate_ = treated_sum / n1 - control_sum / n0
