"""Adaptive block-randomized switchback design with seasonal balancing.

Implements the model-assisted design of Ni & Bojinov, *Enhancing Efficiency
and Robustness for Switchback Experiments: A Model-assisted Design and
Analysis* (Algorithm 1, "Adaptive Block Randomization").

The idea
--------
Periods are grouped into ``B`` **seasonal blocks** by modular position:
block ``b`` (0-indexed) contains the periods ``{b, b + B, b + 2B, …}`` —
e.g., for ``B = 24`` on hourly data, block ``b`` collects "the b-th hour
of every day". Each block holds ``K = T / B`` periods.

Unlike :class:`BernoulliDesign` / :class:`CompleteRandomization` (where a
"window" is a *consecutive* run of periods sharing one coin), here each
period gets its own draw and adjacent periods land in *different* blocks.
The design's structure is in the joint distribution across blocks, not
within consecutive windows. Hence ``l = 1`` in our framework.

The design is generated block-by-block:

1. **Block 0** is sampled by complete randomization: exactly ``K/2``
   treated and ``K/2`` controlled periods.
2. **Block ``b ≥ 1``** is sampled adaptively conditional on block ``b−1``.
   Among the ``K/2`` periods in block ``b−1`` that landed in treatment,
   we assign their successors in block ``b`` to a random permutation of
   ``K_v/2`` treatments and ``(K − K_v)/2`` controls; symmetrically for
   the ``K/2`` periods that landed in control. This guarantees
   ``P(W_{t+1} = 1 | W_t = 1) = ρ`` with ``ρ = K_v / K``.

Theorem 1 of the paper then gives the marginal joint distribution of
consecutive pairs for ``t`` in any block ``b ≥ 1``:

.. math::

    \\Pr(W_{t-1:t} = (1,1)) = \\Pr(W_{t-1:t} = (0,0)) = \\rho/2,
    \\quad
    \\Pr(W_{t-1:t} = (1,0)) = \\Pr(W_{t-1:t} = (0,1)) = (1-\\rho)/2.

For ``t mod B = 0`` (the transition from block ``B−1`` to block 0), block
0 was sampled *unconditionally*, so the pair ``(W_{t-1}, W_t)`` is
independent with marginal ``0.5`` each — i.e. ``P((1,1)) = 0.25``.

Why this design
---------------
Two practical advantages over Bernoulli / complete-random switchback:

* **Deterministic effective sample size.** Exactly ``ρT`` periods are
  "valid" (their last-2 history is constant), so power analysis is
  predictable.
* **Seasonal balance.** Each block (e.g., "7–8 pm") gets exactly ``K/2``
  treatments and ``K/2`` controls across days, dampening the inflation
  from peak periods.

The parameter ``ρ`` trades within-block sample efficiency (large ``ρ``)
against decorrelation through more frequent switches (small ``ρ``). The
paper recommends ``ρ ∈ [0.5, 1]``: ``ρ = 0.5`` matches the average
effective count of fair Bernoulli but with smaller variance through
seasonal balance; ``ρ = 1`` minimises switching (only daily
adjustments).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign


class AdaptiveBlockDesign(BaseDesign):
    r"""Adaptive block-randomized switchback design (Ni & Bojinov).

    Use with ``m = 1`` (window of two consecutive periods) on the
    estimator — this is the paper's primary estimator, motivated by the
    fact that ``P(W_{t-1:t} = (1,1))`` is fixed at ``ρ/2`` for all
    interior periods.

    Parameters
    ----------
    B : int
        Number of seasonal blocks (the paper's ``B``). Equals the
        seasonality period, e.g. ``24`` for hourly data with daily cycle.
        Must be ``≥ 1`` and divide ``T``.
    rho : float, default 0.5
        Target fraction of consecutive-pair periods that are same-arm
        ("valid"). Must be in ``[0, 1]``. The realised ``ρ`` is
        ``K_v / K`` where ``K_v`` is the nearest even integer to
        ``rho * K`` in ``[2, K]`` — read it back with
        :meth:`effective_rho`.
    seed : int or None

    Attributes
    ----------
    l : int
        Always ``1``. Each period gets its own draw; the design's
        structure lives in the joint distribution across blocks.

    Notes
    -----
    Constraints (validated at :meth:`sample` time):

    * ``T % B == 0`` — every block has the same size ``K``.
    * ``K = T // B`` must be even — block 0 needs ``K/2`` of each arm.

    See Also
    --------
    BernoulliDesign, CompleteRandomization
    """

    #: Each period gets its own assignment — adjacent periods are in
    #: different blocks, so there is no "within-window stickiness".
    l: int = 1

    def __init__(
        self,
        B: int,
        rho: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        if not isinstance(B, (int, np.integer)) or B < 1:
            raise ValueError(f"B must be a positive integer, got {B!r}")
        if not (0.0 <= float(rho) <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.B = int(B)
        self.rho = float(rho)

    # ------------------------------------------------------------------
    # Block structure
    # ------------------------------------------------------------------

    def _validate_T(self, T: int) -> int:
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if T % self.B != 0:
            raise ValueError(
                "AdaptiveBlockDesign requires T divisible by B; "
                f"got T={T}, B={self.B}"
            )
        K = T // self.B
        if K % 2 != 0:
            raise ValueError(
                "AdaptiveBlockDesign requires K = T / B even "
                f"(block 0 uses K/2 of each arm); got K={K}"
            )
        return K

    def _K_v(self, K: int) -> int:
        """Effective ``K_v``: nearest even integer to ``ρ K`` in ``[2, K]``."""
        K_v = 2 * round(self.rho * K / 2)
        return int(max(2, min(K, K_v)))

    def effective_rho(self, T: int) -> float:
        """Realised ``ρ = K_v / K`` after rounding ``K_v`` to an even integer."""
        K = self._validate_T(T)
        return self._K_v(K) / K

    def block_index(self, T: int) -> np.ndarray:
        """Map each period in ``[0, T)`` to its seasonal block index."""
        self._validate_T(T)
        return np.arange(T, dtype=int) % self.B

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def sample(self, T: int) -> np.ndarray:
        K = self._validate_T(T)
        B = self.B
        K_v = self._K_v(K)

        W = np.zeros(T, dtype=int)

        # Block 0: complete randomization (K/2 treatments, K/2 controls).
        block0_periods = np.arange(0, T, B)  # length K
        block0_arms = np.zeros(K, dtype=int)
        block0_arms[: K // 2] = 1
        self._rng.shuffle(block0_arms)
        W[block0_periods] = block0_arms

        # Blocks 1 .. B-1: adaptive conditional generation.
        for b in range(1, B):
            prev_periods = np.arange(b - 1, T, B)  # length K
            curr_periods = np.arange(b, T, B)      # length K
            prev_W = W[prev_periods]
            ones_mask = prev_W == 1
            zeros_mask = prev_W == 0

            # After a treated period: assign K_v/2 ones, (K-K_v)/2 zeros
            # so that P(W_{t+1}=1 | W_t=1) = K_v/K = ρ.
            n_ones = int(ones_mask.sum())  # = K/2 by induction
            after_ones = np.zeros(n_ones, dtype=int)
            after_ones[: K_v // 2] = 1
            self._rng.shuffle(after_ones)
            W[curr_periods[ones_mask]] = after_ones

            # Symmetrically, after a controlled period: K_v/2 zeros,
            # (K-K_v)/2 ones, giving P(W_{t+1}=0 | W_t=0) = ρ.
            n_zeros = int(zeros_mask.sum())  # = K/2 by induction
            after_zeros = np.zeros(n_zeros, dtype=int)
            after_zeros[: (K - K_v) // 2] = 1
            self._rng.shuffle(after_zeros)
            W[curr_periods[zeros_mask]] = after_zeros

        return W

    def consecutive_prob(self, T: int, t: int, p: int, value: int = 1) -> float:
        r"""Return :math:`\Pr(W_{t-p:t} = v\,\mathbf{1}_{p+1})`.

        Supports ``p = 0`` (always ``0.5``) and ``p = 1`` (the paper's
        primary case). For ``p ≥ 2`` the joint distribution does not have
        a closed form under this design — raise ``NotImplementedError``.
        """
        if value not in (0, 1):
            raise ValueError(f"value must be 0 or 1, got {value}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if p < 0:
            raise ValueError(f"p must be non-negative, got {p}")
        if t - p < 0 or t >= T:
            raise ValueError(
                f"window [t-p, t] = [{t - p}, {t}] is out of bounds for T={T}"
            )

        K = self._validate_T(T)
        B = self.B

        if p == 0:
            # Block 0 uses K/2 / K = 0.5; for b ≥ 1, marginal is
            # ρ/2 + (1-ρ)/2 = 0.5 by Theorem 1.
            return 0.5

        if p == 1:
            rho_eff = self._K_v(K) / K
            # Pair (W_{t-1}, W_t) crosses block (t-1) mod B → block t mod B.
            # If t mod B ≥ 1 the transition uses the adaptive scheme
            # (Theorem 1); if t mod B = 0 we wrap from block B−1 to block 0,
            # which was sampled unconditionally, so the pair is independent.
            if t % B == 0:
                return 0.25  # = 0.5 * 0.5
            return 0.5 * rho_eff  # = ρ/2 for value ∈ {0, 1}

        raise NotImplementedError(
            "AdaptiveBlockDesign.consecutive_prob currently supports "
            "p ∈ {0, 1} only. The design's primary estimator uses m = 1 "
            "(window of 2 consecutive periods); for p ≥ 2 the joint "
            f"distribution depends on a chain of conditionals. Got p={p}."
        )
