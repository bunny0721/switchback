"""Stratified Hájek estimator: per-window-count Hájek pieces, size-weighted average."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign
from switchback.design.bernoulli import BernoulliDesign
from switchback.design.complete import CompleteRandomization
from switchback.estimators.base import BaseEstimator


_BLOCK_DESIGNS = (BernoulliDesign, CompleteRandomization)


class StratifiedHajekEstimator(BaseEstimator):
    r"""Hájek estimator stratified by the window-count of the IPW window.

    For each contributing period ``t`` (``W_{t-m:t}`` all-treated or
    all-controlled), let ``B(t)`` = number of distinct windows the window
    intersects under the design's window structure. Periods with the same
    ``B`` share a propensity, so the per-stratum Hájek is the natural
    arm-mean difference. Combine across strata via weighted average by
    realised stratum size:

    .. math::

        \hat\tau = \frac{\sum_B n_B \, (\bar Y_{B,1} - \bar Y_{B,0})}
                       {\sum_B n_B},

    where ``n_B = |S_{B,1}| + |S_{B,0}|`` and ``S_{B,w} = \{t : B(t) = B,\,
    W_{t-m:t} \equiv w\}``.

    Each per-stratum Hájek is unbiased for the same lag-``m`` causal
    estimand (asymptotically, given a stationary design distribution), so
    the combined estimator is consistent. Its variance has within-stratum
    pieces plus a cross-stratum covariance — derivation deferred.

    Parameters
    ----------
    design : BernoulliDesign or CompleteRandomization
        Window-structured design exposing ``l`` and
        ``window_index(T)``.
    m : int
        Burn-in length.

    Attributes
    ----------
    estimate_ : float
        Combined point estimate after :meth:`fit`.
    estimate_per_stratum_ : dict[int, float]
        After :meth:`fit`: ``{B: τ̂_B}`` for each ``B`` that has both
        treated and controlled contributions. Useful for variance
        analysis: ``Var(τ̂_strat) = Σ w_B² Var(τ̂_B) + 2 Σ_{B<B'} w_B w_{B'} Cov``.
    n_per_stratum_ : dict[int, tuple[int, int]]
        After :meth:`fit`: ``{B: (n_treated, n_controlled)}`` for each
        observed ``B`` (handy for inspecting the stratification).
    """

    def __init__(self, design: BaseDesign, m: int):
        if not isinstance(design, _BLOCK_DESIGNS):
            raise TypeError(
                "StratifiedHajekEstimator requires a window-structured design "
                f"(BernoulliDesign or CompleteRandomization); got {type(design).__name__}"
            )
        if not isinstance(m, (int, np.integer)) or m < 0:
            raise ValueError(f"m must be a non-negative integer, got {m!r}")
        l = int(design.l)
        if int(m) > l:
            raise ValueError(
                f"m must be ≤ design.l; got m={m}, "
                f"l={l}. The burn-in cannot exceed the design's "
                f"window length. m={l-1} is the natural 'in-window' choice."
            )
        self.design = design
        self.m = int(m)
        self.estimate_: Optional[float] = None
        self.estimate_per_stratum_: Optional[dict[int, float]] = None
        self.n_per_stratum_: Optional[dict[int, tuple[int, int]]] = None

    def fit(
        self, assignment: np.ndarray, outcomes: np.ndarray
    ) -> "StratifiedHajekEstimator":
        W, Y = self._validate(assignment, outcomes)
        T, m = W.size, self.m
        if T <= m:
            raise ValueError(f"need T > m; got T={T}, m={m}")
        K = self.design.l

        # Group contributing periods by B(t) = #windows the window intersects,
        # and within each stratum split by arm.
        strata: dict[int, dict[int, list[float]]] = {}
        for t in range(m, T):
            window = W[t - m : t + 1]
            if np.all(window == 1):
                arm = 1
            elif np.all(window == 0):
                arm = 0
            else:
                continue
            B = (t // K) - ((t - m) // K) + 1
            if B not in strata:
                strata[B] = {1: [], 0: []}
            strata[B][arm].append(float(Y[t]))

        # Combined estimate: size-weighted average of per-stratum Hájeks.
        weighted_sum = 0.0
        total_weight = 0
        n_per_stratum: dict[int, tuple[int, int]] = {}
        estimate_per_stratum: dict[int, float] = {}
        for B, arms in strata.items():
            n1, n0 = len(arms[1]), len(arms[0])
            n_per_stratum[B] = (n1, n0)
            if n1 == 0 or n0 == 0:
                continue
            tau_B = float(np.mean(arms[1]) - np.mean(arms[0]))
            estimate_per_stratum[B] = tau_B
            n_B = n1 + n0
            weighted_sum += n_B * tau_B
            total_weight += n_B

        if total_weight == 0:
            raise ValueError(
                "No stratum has both treated and controlled contributions; "
                "cannot compute stratified Hájek."
            )
        self.estimate_ = float(weighted_sum / total_weight)
        self.estimate_per_stratum_ = estimate_per_stratum
        self.n_per_stratum_ = n_per_stratum
        return self
