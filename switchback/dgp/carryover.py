"""DGP with finite-order treatment carryover."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from switchback.dgp._utils import MuLike, resolve_mu_path
from switchback.dgp.base import BaseDGP


class CarryoverDGP(BaseDGP):
    r"""Outcomes with finite-order treatment carryover and i.i.d. noise.

    .. math::
        Y_t = \mu_t + \sum_{k=0}^{q} \beta_k\, W_{t-k} + \varepsilon_t,
        \quad \varepsilon_t \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2),

    with ``W_s = 0`` for ``s < 0``. The carryover order ``q`` is implied by
    ``len(betas) - 1``; ``betas[0]`` is the contemporaneous (lag-0) effect.
    Setting ``betas = [beta_0]`` recovers :class:`SimpleDGP` (no carryover).

    Under sustained treatment ``W_t = 1`` for all ``t \ge q``, the long-run
    mean is :math:`\mu + \sum_k \beta_k`. The lag-``m`` causal estimand
    (which the IPW estimator with burn-in ``m`` targets) is
    :math:`\sum_{k=0}^{\min(m, q)} \beta_k`; for ``m \ge q`` this is the
    full long-run effect.

    Parameters
    ----------
    betas : sequence of float
        ``[beta_0, beta_1, ..., beta_q]``. Non-empty.
    mu : float, length-T array, or callable ``t -> mu_t``
        Baseline. Same conventions as :class:`SimpleDGP`'s ``mu``.
    sigma : float
        Standard deviation of the Gaussian noise. Must be non-negative.
    seed : int or None
    """

    def __init__(
        self,
        betas: Union[Sequence[float], np.ndarray],
        mu: MuLike = 0.0,
        sigma: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        betas_arr = np.asarray(betas, dtype=float)
        if betas_arr.ndim != 1 or betas_arr.size == 0:
            raise ValueError("betas must be a non-empty 1-D sequence")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.betas = betas_arr
        self.mu = mu
        self.sigma = float(sigma)

    @property
    def q(self) -> int:
        """Carryover order = ``len(betas) - 1``."""
        return self.betas.size - 1

    def generate(self, assignment: np.ndarray) -> np.ndarray:
        W = self._validate_assignment(assignment).astype(float)
        T = W.size
        mu_t = resolve_mu_path(self.mu, T)
        # np.convolve with mode='full' gives length T + q; truncate to T.
        # convolve(W, betas)[t] = Σ_k betas[k] * W[t-k] (with W treated as
        # zero outside [0, T)), which is exactly the carryover sum.
        carry = np.convolve(W, self.betas, mode="full")[:T]
        eps = self._rng.normal(0.0, self.sigma, size=T)
        return mu_t + carry + eps
