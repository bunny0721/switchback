"""Bernoulli switchback designs.

A Bernoulli design partitions the time horizon into equal-length windows and
draws one independent ``Bern(p)`` coin per window; every period inside a window
inherits its window's draw. ``l`` is the only knob: it controls how
frequently the coin is flipped.

The estimator's burn-in length ``m`` is *not* part of the design; the design
is defined purely by the assignment mechanism.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign


class BernoulliDesign(BaseDesign):
    r"""Window-Bernoulli switchback design.

    Partition the horizon ``[0, T)`` into windows ``[0, K), [K, 2K), ...``
    of length ``K = l`` (the final window may be shorter). Within
    each window, every period gets the same ``Bern(p)`` draw; coins for
    different windows are independent.

    Special cases:
    * ``l = 1`` recovers per-period i.i.d. Bernoulli randomization.
    * ``l = T`` is constant treatment for the whole horizon.

    Parameters
    ----------
    l : int
        Time window length: the number of consecutive periods sharing one
        coin flip. Must be ``>= 1``.
    p : float
        Treatment probability per coin. Must be in the open interval ``(0, 1)``.
        Default ``0.5`` (fair coin).
    seed : int or None

    Notes
    -----
    The design exposes ``consecutive_prob(T, t, p, value)`` returning
    :math:`\Pr(W_{t-p:t} = v\,\mathbf{1}_{p+1})`. Under independence across
    windows, this equals ``p ** B`` (or ``(1-p) ** B``) where ``B`` is the
    number of distinct windows the window ``[t-p, t]`` intersects.
    """

    def __init__(
        self,
        l: int = 1,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        if not isinstance(l, (int, np.integer)) or l < 1:
            raise ValueError(
                f"l must be a positive integer, got {l!r}"
            )
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be strictly in (0, 1), got {p}")
        self.l = int(l)
        self.p = float(p)

    # ------------------------------------------------------------------
    # Window structure
    # ------------------------------------------------------------------

    def n_windows(self, T: int) -> int:
        """Number of windows the horizon ``T`` is split into."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        K = self.l
        return (T + K - 1) // K

    def window_index(self, T: int) -> np.ndarray:
        """Map each period in ``[0, T)`` to its window index."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        return np.arange(T, dtype=int) // self.l

    def randomization_points(self, T: int) -> np.ndarray:
        """0-indexed first period of each window."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        return np.arange(0, T, self.l, dtype=int)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def sample(self, T: int) -> np.ndarray:
        coins = (self._rng.random(self.n_windows(T)) < self.p).astype(int)
        return coins[self.window_index(T)]

    def consecutive_prob(self, T: int, t: int, p: int, value: int = 1) -> float:
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
        K = self.l
        first_window = (t - p) // K
        last_window = t // K
        n_intersected = last_window - first_window + 1
        prob = self.p if value == 1 else (1.0 - self.p)
        return float(prob ** n_intersected)
