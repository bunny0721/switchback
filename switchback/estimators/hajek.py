"""Hájek (normalized IPW) estimator for switchback experiments.

Stratified-aware: when the design has window structure and the burn-in
length ``m`` straddles a window boundary, the estimator automatically
stratifies contributing periods by the number of windows their IPW
window intersects, computes a per-stratum Hájek (arm-mean difference),
and combines via realised-size weights. Special cases reduce to a
single estimator:

* ``m = 0``: every contributing period is in stratum ``B=1`` (window
  fits in a single design window) → one Hájek estimator.
* ``0 < m < window_length``: strata ``B=1`` and ``B=2`` both contribute
  → two estimators combined.
* ``m = window_length``: every contributing period is in stratum
  ``B=2`` (window crosses one design-window boundary) → one Hájek.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign
from switchback.design.bernoulli import BernoulliDesign
from switchback.design.complete import CompleteRandomization
from switchback.estimators.base import BaseEstimator


_WINDOW_DESIGNS = (BernoulliDesign, CompleteRandomization)


class HajekEstimator(BaseEstimator):
    r"""Stratified Hájek estimator with burn-in length ``m``.

    For each contributing period ``t`` (``W_{t-m:t}`` all-treated or
    all-controlled), let ``B(t) = (t // l) - ((t - m) // l) + 1`` be the
    number of design windows the IPW window intersects (where
    ``l = window_length``). Group contributing periods by ``B(t)`` and
    compute a per-stratum Hájek

    .. math::

        \hat\tau_B = \bar Y_{B,1} - \bar Y_{B,0},
        \qquad
        \bar Y_{B,w} = \frac{1}{|S_{B,w}|} \sum_{t \in S_{B,w}} Y_t,

    where ``S_{B,w} = {t : B(t) = B, W_{t-m:t} \equiv w}``. Combine the
    per-stratum estimates via realised-size weighting:

    .. math::

        \hat\tau = \frac{\sum_B n_B \, \hat\tau_B}{\sum_B n_B},
        \qquad n_B = |S_{B,1}| + |S_{B,0}|.

    Parameters
    ----------
    design : BernoulliDesign or CompleteRandomization
        Window-structured design.
    m : int
        Burn-in length. Non-negative integer.

    Attributes
    ----------
    estimate_ : float
        Combined point estimate after :meth:`fit`.
    estimate_per_stratum_ : dict[int, float]
        ``{B: τ̂_B}`` for each ``B`` with both treated and controlled
        contributions. Handy for variance inspection: ``Var(τ̂_strat)``
        decomposes as ``Σ w_B² Var(τ̂_B) + 2 Σ_{B<B'} w_B w_{B'} Cov``.
    n_per_stratum_ : dict[int, tuple[int, int]]
        ``{B: (n_treated, n_controlled)}`` for each observed ``B``.

    Notes
    -----
    The variance and CI live in :class:`switchback.decisions.HACVariance`,
    which builds the per-window influence-function sequence for this
    estimator and applies HAC.
    """

    def __init__(self, design: BaseDesign, m: int):
        if not isinstance(design, _WINDOW_DESIGNS):
            raise TypeError(
                "HajekEstimator requires a window-structured design "
                "(BernoulliDesign or CompleteRandomization); got "
                f"{type(design).__name__}"
            )
        if not isinstance(m, (int, np.integer)) or m < 0:
            raise ValueError(f"m must be a non-negative integer, got {m!r}")
        l = int(design.window_length)
        if int(m) > l:
            raise ValueError(
                f"m must be ≤ design.window_length; got m={m}, "
                f"window_length={l}. The burn-in cannot exceed the design's "
                f"window length. m={l-1} is the natural 'in-window' choice."
            )
        self.design = design
        self.m = int(m)
        self.estimate_: Optional[float] = None
        self.estimate_per_stratum_: Optional[dict[int, float]] = None
        self.n_per_stratum_: Optional[dict[int, tuple[int, int]]] = None

    def fit(self, assignment: np.ndarray, outcomes: np.ndarray) -> "HajekEstimator":
        W, Y = self._validate(assignment, outcomes)
        T, m = W.size, self.m
        if T <= m:
            raise ValueError(f"need T > m; got T={T}, m={m}")
        K = self.design.window_length

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
                "cannot compute Hájek estimator."
            )
        self.estimate_ = float(weighted_sum / total_weight)
        self.estimate_per_stratum_ = estimate_per_stratum
        self.n_per_stratum_ = n_per_stratum
        return self
