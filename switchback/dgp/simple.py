"""Simplest switchback DGP: additive baseline + treatment + i.i.d. noise."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.dgp._utils import MuLike, resolve_mu_path
from switchback.dgp.base import BaseDGP


class SimpleDGP(BaseDGP):
    r"""Outcomes with a (possibly time-varying) baseline and a constant treatment effect.

    .. math::
        Y_t = \mu_t + \tau\, W_t + \varepsilon_t,
        \quad \varepsilon_t \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2),

    for ``t = 0, 1, ..., T-1``. There is no autoregressive component and no
    carryover — given ``W_t`` and ``t``, the ``Y_t`` are conditionally
    independent. The treatment effect ``tau`` is the ATE.

    Parameters
    ----------
    mu : float, array of length T, or callable ``t -> mu_t``
        Baseline. A scalar broadcasts to every period; an array must have
        length ``T``; a callable receives ``t = np.arange(T)`` and must
        return an array of shape ``(T,)``.
    tau : float
        Treatment effect.
    sigma : float
        Standard deviation of the Gaussian noise. Must be non-negative;
        ``sigma = 0`` gives a deterministic DGP.
    seed : int or None
        Seed for the internal RNG.
    """

    def __init__(
        self,
        mu: MuLike = 0.0,
        tau: float = 1.0,
        sigma: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.mu = mu
        self.tau = float(tau)
        self.sigma = float(sigma)

    def generate(self, assignment: np.ndarray) -> np.ndarray:
        W = self._validate_assignment(assignment).astype(float)
        T = W.size
        mu_t = resolve_mu_path(self.mu, T)
        eps = self._rng.normal(0.0, self.sigma, size=T)
        return mu_t + self.tau * W + eps
