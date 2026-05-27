"""Compute the asymptotic 'true' average treatment effect under a DGP."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.dgp.base import BaseDGP


def true_effect(dgp: BaseDGP, T: int, seed: Optional[int] = 0) -> float:
    r"""Estimate the per-period long-run treatment effect of a DGP.

    The true effect is the average per-period potential-outcome difference
    under sustained treatment vs. sustained control:

    .. math::

        \tau^{\text{true}} \;=\; \frac{1}{T} \sum_{t=1}^{T}
            \big( Y_t(W \equiv 1) - Y_t(W \equiv 0) \big).

    Implementation: generate ``Y`` under the all-1 path and again under
    the all-0 path with the **same** RNG seed, so the noise terms cancel
    in the difference and the result reflects only the deterministic
    treatment contribution. For DGPs with infinite carryover (e.g.
    :class:`LatentStateDGP`), this converges to the long-run effect
    ``β_0 + α_0/(1-γ)`` as ``T → ∞``. For finite-order carryover (e.g.
    :class:`CarryoverDGP`), it converges to ``Σ β_k``. For DGPs without
    carryover (e.g. :class:`SimpleDGP`), it equals the contemporaneous
    effect ``τ``.

    Parameters
    ----------
    dgp : BaseDGP
        The DGP to query. Must support ``reset(seed)`` (all subclasses do).
    T : int
        Horizon to average over. Larger ``T`` → tighter estimate.
    seed : int or None
        RNG seed for the two paired calls. ``None`` skips the reset, in
        which case the two calls use whatever state ``dgp`` happens to
        be in (and noise will not cancel exactly).

    Returns
    -------
    float
        The estimated true effect.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if seed is not None:
        dgp.reset(seed=seed)
    Y1 = dgp.generate(np.ones(T, dtype=int))
    if seed is not None:
        dgp.reset(seed=seed)
    Y0 = dgp.generate(np.zeros(T, dtype=int))
    return float(np.mean(Y1 - Y0))
