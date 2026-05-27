"""Design-based (Neyman-type) inference for switchback experiments.

Variance via **joint HAC on a per-window influence-function sequence**.
Confidence intervals via the normal approximation.

The plug-in
-----------
Each supported estimator τ̂ is asymptotically a sample mean of a per-window
influence-function sequence ``X_b`` (one value per design window):

* **Hájek (any design)** — stratified influence
  ``X_b = w_1 · ξ_b + w_2 · η_b`` with per-arm-demeaned ``ξ_b, η_b`` and
  realised-size weights ``w_B = n_B / Σ n_B'``. The ``ξ_b`` covers
  contributing periods at within-window positions ``p ∈ [m, l)``; the
  ``η_b`` covers ``p ∈ [0, m)`` (contributing when consecutive windows
  share their coin).

* **IPW under BernoulliDesign** — per-window IPW sum (no demeaning)
  ``X_b = (1/l) · Σ_{t in window b, contributing} Y_t · D_t / π_t``.

* **IPW under CompleteRandomization** — IPW self-normalises and
  collapses to Hájek (see :class:`IPWEstimator`), so the Hájek influence
  above is used.

Then

.. math::

    \\widehat{\\mathrm{Var}}(\\hat\\tau) \\;=\\; \\frac{1}{n_w}\\Bigl[
        \\hat\\gamma_X(0) \\,+\\, 2 \\sum_{k=1}^{L} K(k, L)\\, \\hat\\gamma_X(k)
    \\Bigr],

where ``L`` is the HAC bandwidth, ``K(k, L)`` is the kernel weight
(``1`` for truncated, ``1 - k/(L+1)`` for Bartlett), and
``γ̂_X(k) = (1/n_w) Σ_b (X_b − X̄)(X_{b+k} − X̄)``.

``L = 0`` reduces to a plain (centered) Welch variance — exact for
DGPs with no carryover or with finite carryover ``q ≤ m``. ``L = 1`` is
the recommended default: it picks up the lag-1 autocovariance from
shared window coins in the multi-stratum case and the residual
``h_t``-mediated coupling in infinite-carryover DGPs. ``L ≥ 2`` rarely
helps for designs with window length ``≥ m + 1``.
"""

from __future__ import annotations

import copy
import math
from statistics import NormalDist
from typing import Optional, Tuple

import numpy as np

from switchback.design.adaptive_block import AdaptiveBlockDesign
from switchback.design.base import BaseDesign
from switchback.design.bernoulli import BernoulliDesign
from switchback.design.complete import CompleteRandomization
from switchback.estimators.base import BaseEstimator
from switchback.estimators.hajek import HajekEstimator
from switchback.estimators.ipw import IPWEstimator


_SUPPORTED_DESIGNS = (BernoulliDesign, CompleteRandomization)


# ---------------------------------------------------------------------------
# Normal-approximation CI from (estimate, variance)
# ---------------------------------------------------------------------------

def normal_ci(
    estimate: float, variance: float, alpha: float = 0.05
) -> Tuple[float, float]:
    """Normal-approximation confidence interval from a point estimate and a
    variance estimate.

    Returns ``(estimate − z·√variance, estimate + z·√variance)`` where
    ``z = Φ^{-1}(1 − α/2)`` is the two-sided normal quantile.

    Use this as the final step after computing the variance via
    :func:`block_variance` (AdaptiveBlockDesign) or :class:`HACVariance`
    (BernoulliDesign, CompleteRandomization). Same CI logic is used
    internally by :meth:`HACVariance.confidence_interval` and the
    :func:`block_confidence_interval` convenience wrapper.

    Parameters
    ----------
    estimate : float
        Point estimate (e.g., the IPW or Hájek τ̂).
    variance : float
        Variance estimate. Must be non-negative.
    alpha : float in (0, 1), default 0.05
        Nominal significance level. The interval has nominal coverage
        ``1 − α`` under the normal approximation.

    Returns
    -------
    (low, high) : Tuple[float, float]

    Raises
    ------
    ValueError
        If ``variance < 0`` or ``alpha`` not in ``(0, 1)``.
    """
    if variance < 0:
        raise ValueError(
            f"variance must be non-negative; got {variance:.3e}. "
            "(Negative V̂ can occur with the design-derived block_variance "
            "under heavy carryover and small K — increase K or use the "
            "Corollary 1 ϕ-bound if needed.)"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    half = z * math.sqrt(variance)
    return estimate - half, estimate + half


# ---------------------------------------------------------------------------
# HAC variance on a stationary sequence
# ---------------------------------------------------------------------------

def hac_variance(
    X: np.ndarray,
    lags: int,
    kernel: str = "truncated",
    bessel: bool = False,
) -> float:
    """Bartlett-kernel or truncated-kernel HAC variance of the sample mean of X.

    Returns ``(γ̂_0 + 2 Σ K(k,L) γ̂_k) / n``.

    Parameters
    ----------
    X : np.ndarray
        Influence-function sequence (one value per aggregation unit).
    lags : int
        HAC bandwidth ``L`` (``0`` reduces to plain Welch).
    kernel : {"truncated", "bartlett"}
    bessel : bool, default False
        If True, divide the sample autocovariances by ``n - 1`` instead of
        ``n``. Recommended ("cluster-robust" or "block-wise" form) when the
        aggregation unit is a small number of clusters/blocks rather than a
        long time series.
    """
    n = X.size
    if n < 2:
        return float("nan")
    dev = X - X.mean()
    denom = (n - 1) if bessel else n
    sigma = float(np.sum(dev * dev) / denom)
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(np.sum(dev[:-k] * dev[k:]) / denom)
        if kernel == "truncated":
            w = 1.0
        elif kernel == "bartlett":
            w = 1.0 - k / (lags + 1)
        else:
            raise ValueError(f"unknown kernel: {kernel!r}")
        sigma += 2.0 * w * gk
    return sigma / n


# ---------------------------------------------------------------------------
# HACVariance
# ---------------------------------------------------------------------------

class HACVariance:
    """Joint-HAC variance plug-in + normal-approximation confidence interval
    for window-structured designs (:class:`BernoulliDesign` and
    :class:`CompleteRandomization`).

    Newey-West HAC on a length-``T/window_length`` per-window influence
    sequence. Default ``L = 1``; ``kernel = "truncated"``.

    .. note::

       **AdaptiveBlockDesign is NOT supported here.** Use
       :func:`switchback.decisions.block_variance` instead — that
       function implements eq. 28 with the design-derived ``γ⁺, γ⁻``
       weights and the block-0 boundary corrections, which is the
       correct estimator under that design's seasonal-block structure.
       HAC at the block level was tested previously and found to be
       under-calibrated under heavy carryover (only ``B = 24`` clusters
       leaves the sample autocovariances too noisy).

    Parameters
    ----------
    design : BernoulliDesign or CompleteRandomization
    estimator : IPWEstimator or HajekEstimator
    L : int, optional
        HAC bandwidth (lag truncation). Defaults to ``1``.
    kernel : {"truncated", "bartlett"}, default "truncated"

    Attributes
    ----------
    estimate_ : float
        Point estimate from the user's estimator.
    variance_ : float
        Joint-HAC variance plug-in.
    """

    def __init__(
        self,
        design: BaseDesign,
        estimator: BaseEstimator,
        L: Optional[int] = None,
        kernel: str = "truncated",
    ):
        if isinstance(design, AdaptiveBlockDesign):
            raise TypeError(
                "HACVariance does not support AdaptiveBlockDesign. "
                "Use block_variance (eq. 28 with boundary fixes) for "
                "AdaptiveBlockDesign — its structural γ-weights are the "
                "correct estimator under that design's seasonal-block "
                "structure. HACVariance is for window-structured designs "
                "(BernoulliDesign, CompleteRandomization) only."
            )
        if not isinstance(design, _SUPPORTED_DESIGNS):
            raise TypeError(
                "HACVariance requires BernoulliDesign or "
                f"CompleteRandomization; got {type(design).__name__}"
            )
        if not isinstance(estimator, (IPWEstimator, HajekEstimator)):
            raise TypeError(
                "HACVariance requires IPWEstimator or "
                f"HajekEstimator; got {type(estimator).__name__}"
            )
        if L is None:
            L = 1
        if not isinstance(L, (int, np.integer)) or L < 0:
            raise ValueError(f"L must be a non-negative integer, got {L!r}")
        if kernel not in ("truncated", "bartlett"):
            raise ValueError(
                f"kernel must be 'truncated' or 'bartlett', got {kernel!r}"
            )
        self.design = design
        self.estimator = estimator
        self.L = int(L)
        self.kernel = kernel
        self.estimate_: Optional[float] = None
        self.variance_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, assignment: np.ndarray, outcomes: np.ndarray) -> "HACVariance":
        W = np.asarray(assignment, dtype=int)
        Y = np.asarray(outcomes, dtype=float)
        if W.shape != Y.shape or W.ndim != 1:
            raise ValueError("assignment and outcomes must be 1-D of same length")
        T, m = W.size, self.estimator.m
        if T <= m:
            raise ValueError(f"need T > m; got T={T}, m={m}")

        # Point estimate via the user's estimator (deep-copied so the user's
        # estimator object is not mutated).
        est_copy = copy.deepcopy(self.estimator)
        est_copy.fit(W, Y)
        self.estimate_ = float(est_copy.estimate_)

        # Build per-window influence-function sequence and HAC.
        K = self.design.window_length
        if isinstance(self.estimator, IPWEstimator) and isinstance(
            self.design, BernoulliDesign
        ):
            X = self._build_ipw_influence(W, Y, T, m, K)
        else:
            X = self._build_hajek_influence(W, Y, T, m, K)
        self.variance_ = float(hac_variance(X, self.L, self.kernel))
        return self

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        if self.estimate_ is None or self.variance_ is None:
            raise RuntimeError("call fit() before confidence_interval()")
        return normal_ci(self.estimate_, self.variance_, alpha)

    # ------------------------------------------------------------------
    # Influence-function builders
    # ------------------------------------------------------------------

    def _build_hajek_influence(
        self, W: np.ndarray, Y: np.ndarray, T: int, m: int, K: int
    ) -> np.ndarray:
        """Stratified Hájek per-window influence ζ_b = w_1 ξ_b + w_2 η_b."""
        n = T // K
        pos_B1 = list(range(m, K))     # positions where window is in single design-window
        pos_B2 = list(range(0, m))     # positions where window crosses to previous design-window

        # ----- B=1 influence: ξ_b = 2(2 C_b − 1)(U_b − μ̂_{C_b}) -----
        if pos_B1:
            U = np.zeros(n, dtype=float)
            C = np.zeros(n, dtype=int)
            valid_xi = np.zeros(n, dtype=bool)
            for b in range(n):
                if K * b + pos_B1[-1] >= T:
                    continue
                U[b] = np.mean([Y[K * b + p] for p in pos_B1])
                C[b] = int(W[K * b])
                valid_xi[b] = True
            if valid_xi.any():
                u1 = (
                    float(U[valid_xi & (C == 1)].mean())
                    if (valid_xi & (C == 1)).any()
                    else 0.0
                )
                u0 = (
                    float(U[valid_xi & (C == 0)].mean())
                    if (valid_xi & (C == 0)).any()
                    else 0.0
                )
                mu = np.where(C == 1, u1, u0)
                xi = 2.0 * (2 * C - 1) * (U - mu)
                xi[~valid_xi] = 0.0
            else:
                xi = np.zeros(n, dtype=float)
            n_B1 = len(pos_B1) * int(valid_xi.sum())
        else:
            xi = np.zeros(n, dtype=float)
            n_B1 = 0

        # ----- B=2 influence: η_b for contributing window-pairs -----
        if pos_B2:
            V = np.zeros(n, dtype=float)
            arm = np.zeros(n, dtype=int)
            n_B2 = 0
            for b in range(1, n):
                if K * b + pos_B2[-1] >= T:
                    continue
                V[b] = np.mean([Y[K * b + p] for p in pos_B2])
                cp = int(W[K * (b - 1)])
                cc = int(W[K * b])
                if cp != cc:
                    arm[b] = 0
                elif cc == 1:
                    arm[b] = 1
                    n_B2 += len(pos_B2)
                else:
                    arm[b] = -1
                    n_B2 += len(pos_B2)
            v1 = float(V[arm == 1].mean()) if (arm == 1).any() else 0.0
            v0 = float(V[arm == -1].mean()) if (arm == -1).any() else 0.0
            eta = np.zeros(n, dtype=float)
            eta[arm == 1] = 4.0 * (V[arm == 1] - v1)
            eta[arm == -1] = -4.0 * (V[arm == -1] - v0)
        else:
            eta = np.zeros(n, dtype=float)
            n_B2 = 0

        total = n_B1 + n_B2
        if total == 0:
            raise ValueError(
                "No contributing periods to build influence-function sequence."
            )
        w1 = n_B1 / total
        w2 = n_B2 / total
        return w1 * xi + w2 * eta

    def _build_ipw_influence(
        self, W: np.ndarray, Y: np.ndarray, T: int, m: int, K: int
    ) -> np.ndarray:
        """Per-window IPW sum Z_b/l (un-demeaned; carries the level term)."""
        n = T // K
        Z = np.zeros(n, dtype=float)
        for b in range(n):
            for p in range(K):
                t = K * b + p
                if t >= T or t < m:
                    continue
                window = W[t - m : t + 1]
                if np.all(window == 1):
                    pi = self.design.consecutive_prob(T, t, m, value=1)
                    Z[b] += Y[t] / pi
                elif np.all(window == 0):
                    pi = self.design.consecutive_prob(T, t, m, value=0)
                    Z[b] -= Y[t] / pi
        return Z / K

def confidence_interval(
    design: BaseDesign,
    estimator: BaseEstimator,
    assignment: np.ndarray,
    outcomes: np.ndarray,
    alpha: float = 0.05,
    L: int = 1,
    kernel: str = "truncated",
) -> Tuple[float, float]:
    """Convenience wrapper around :class:`HACVariance`."""
    return (
        HACVariance(design=design, estimator=estimator, L=L, kernel=kernel)
        .fit(assignment, outcomes)
        .confidence_interval(alpha)
    )
