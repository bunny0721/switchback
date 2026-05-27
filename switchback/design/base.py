"""Abstract base class for switchback designs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDesign(ABC):
    """A design samples assignment paths and reports the probabilities the
    Horvitz-Thompson estimator needs.

    All time indices in this API are 0-indexed (the paper uses 1-indexed
    notation; conversions are documented in subclasses).
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, T: int) -> np.ndarray:
        """Return a random assignment path of length ``T`` as a 0/1 array."""
        raise NotImplementedError

    @abstractmethod
    def consecutive_prob(self, T: int, t: int, p: int, value: int = 1) -> float:
        r"""Return :math:`\Pr(W_{t-p:t} = v\,\mathbf{1}_{p+1})` for ``v = value``.

        ``t`` and ``t - p`` are 0-indexed. ``value`` must be 0 or 1.
        """
        raise NotImplementedError

    def __call__(self, T: int) -> np.ndarray:
        return self.sample(T)
