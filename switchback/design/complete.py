"""Complete-randomization switchback design."""

from __future__ import annotations

from typing import Optional

import numpy as np

from switchback.design.base import BaseDesign


class CompleteRandomization(BaseDesign):
    r"""Complete-randomization switchback (sample without replacement at the window level).

    Partition the horizon ``[0, T)`` into ``n = T / window_length`` equal-size
    windows; uniformly sample ``n / 2`` windows to be treated and the remaining
    ``n / 2`` to be control. Within a window, every period inherits the
    window's draw, so the realised assignment path always has exactly ``T/2``
    treated and ``T/2`` controlled periods.

    Compared to :class:`BernoulliDesign` (independent coin per window), this
    design **eliminates** treatment-share variability — but introduces
    negative cross-window correlation, since the total treated count is
    fixed.

    Constraints
    -----------
    * ``T`` must be divisible by ``window_length``.
    * ``n = T / window_length`` must be even.

    Parameters
    ----------
    window_length : int
        Number of consecutive periods per window. Must be ``>= 1``.
    seed : int or None

    Notes
    -----
    The window propensity is hypergeometric: for a window ``[t-p, t]`` that
    intersects ``B`` distinct windows,

    .. math::

        \Pr(W_{t-p:t} \equiv 1) = \Pr(W_{t-p:t} \equiv 0)
        = \frac{\binom{n/2}{B}}{\binom{n}{B}}
        = \prod_{k=0}^{B-1} \frac{n/2 - k}{n - k}.

    For ``B > n/2`` the probability is zero (impossible). As ``n \to \infty``
    this converges to the Bernoulli ``(1/2)^B``.
    """

    def __init__(self, window_length: int = 1, seed: Optional[int] = None):
        super().__init__(seed=seed)
        if not isinstance(window_length, (int, np.integer)) or window_length < 1:
            raise ValueError(
                f"window_length must be a positive integer, got {window_length!r}"
            )
        self.window_length = int(window_length)

    # ------------------------------------------------------------------
    # Window structure
    # ------------------------------------------------------------------

    def n_windows(self, T: int) -> int:
        """Number of windows. Validates the divisibility/parity constraints."""
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        K = self.window_length
        if T % K != 0:
            raise ValueError(
                f"CompleteRandomization requires T divisible by window_length; "
                f"got T={T}, window_length={K}"
            )
        n = T // K
        if n % 2 != 0:
            raise ValueError(
                f"CompleteRandomization requires an even number of windows; "
                f"got n = T/window_length = {n}"
            )
        return n

    def window_index(self, T: int) -> np.ndarray:
        """Map each period in ``[0, T)`` to its window index."""
        self.n_windows(T)  # validate
        return np.arange(T, dtype=int) // self.window_length

    def randomization_points(self, T: int) -> np.ndarray:
        """0-indexed first period of each window."""
        self.n_windows(T)
        return np.arange(0, T, self.window_length, dtype=int)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def sample(self, T: int) -> np.ndarray:
        n = self.n_windows(T)
        coins = np.zeros(n, dtype=int)
        treated_idx = self._rng.choice(n, size=n // 2, replace=False)
        coins[treated_idx] = 1
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
        n = self.n_windows(T)
        K = self.window_length
        first_window = (t - p) // K
        last_window = t // K
        B = last_window - first_window + 1
        if B > n // 2:
            return 0.0
        # Pr(B specific windows all assigned to one side)
        # = (n/2)! * (n - B)! / ((n/2 - B)! * n!)
        # = ∏_{k=0..B-1} (n/2 - k) / (n - k).
        prob = 1.0
        half = n // 2
        for k in range(B):
            prob *= (half - k) / (n - k)
        return float(prob)
