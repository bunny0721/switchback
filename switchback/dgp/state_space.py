"""DGP with a hidden AR(1) state — geometrically-decaying ("infinite") carryover."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.dgp._utils import MuLike, resolve_mu_path
from switchback.dgp.base import BaseDGP


class LatentStateDGP(BaseDGP):
    r"""Outcomes driven by a hidden AR(1) state plus a contemporaneous treatment effect.

    .. math::

        Y_t &= \mu_t + h_t + \beta_0 W_t + \varepsilon^Y_t,
        \quad \varepsilon^Y_t \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma_Y^2),\\
        h_t &= \gamma\, h_{t-1} + \alpha_0\, W_t + \varepsilon^h_t,
        \quad \varepsilon^h_t \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma_h^2).

    The hidden state ``h_t`` accumulates treatment effect with geometric
    decay (rate ``γ``). For ``|γ| < 1`` the state has a stationary
    distribution, but every past ``W_s`` continues to influence ``Y_t`` via
    ``h_t`` with weight ``γ^{t-s}`` — i.e. the carryover is technically
    *infinite* (geometrically decaying), unlike :class:`CarryoverDGP`'s
    finite-order linear carryover.

    Asymptotic lag-``m`` estimand under stationary i.i.d. ``Bern(½)``
    treatment (e.g. :class:`BernoulliDesign` with default ``p``):

    .. math::

        \tau_m = \beta_0 + \frac{\alpha_0}{1 - \gamma}\,(1 - \gamma^{m+1}).

    For ``m \to \infty`` this is the long-run effect
    ``β_0 + α_0/(1-γ)``; for ``m = 0`` it is the immediate effect
    ``β_0 + α_0`` (treatment hits ``Y`` directly through ``β_0`` and
    indirectly through the new ``h``).

    Parameters
    ----------
    mu : float, length-T array, or callable ``t -> mu_t``
        Baseline. Same conventions as :class:`SimpleDGP`'s ``mu``.
    beta_0 : float
        Direct effect of ``W_t`` on ``Y_t``.
    alpha_0 : float
        Effect of ``W_t`` on the hidden state ``h_t``.
    gamma : float
        AR(1) decay coefficient for ``h``. Must satisfy ``|γ| < 1``.
    sigma_y : float
        Std deviation of the outcome noise ``ε^Y``. Must be non-negative.
    sigma_h : float
        Std deviation of the state noise ``ε^h``. Must be non-negative.
    h0 : float
        Initial hidden state ``h_0`` (before period 0). Defaults to 0.
    seed : int or None
    """

    def __init__(
        self,
        mu: MuLike = 0.0,
        beta_0: float = 1.0,
        alpha_0: float = 0.5,
        gamma: float = 0.5,
        sigma_y: float = 1.0,
        sigma_h: float = 1.0,
        h0: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        if not (-1.0 < gamma < 1.0):
            raise ValueError(f"|gamma| < 1 required for stationarity, got {gamma}")
        if sigma_y < 0:
            raise ValueError(f"sigma_y must be non-negative, got {sigma_y}")
        if sigma_h < 0:
            raise ValueError(f"sigma_h must be non-negative, got {sigma_h}")
        self.mu = mu
        self.beta_0 = float(beta_0)
        self.alpha_0 = float(alpha_0)
        self.gamma = float(gamma)
        self.sigma_y = float(sigma_y)
        self.sigma_h = float(sigma_h)
        self.h0 = float(h0)

    def generate(self, assignment: np.ndarray) -> np.ndarray:
        W = self._validate_assignment(assignment).astype(float)
        T = W.size
        mu_t = resolve_mu_path(self.mu, T)
        eps_y = self._rng.normal(0.0, self.sigma_y, size=T)
        eps_h = self._rng.normal(0.0, self.sigma_h, size=T)

        h = np.empty(T, dtype=float)
        h_prev = self.h0
        for t in range(T):
            h_prev = self.gamma * h_prev + self.alpha_0 * W[t] + eps_h[t]
            h[t] = h_prev
        return mu_t + h + self.beta_0 * W + eps_y
