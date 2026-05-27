"""Abstract base class for data generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDGP(ABC):
    """A data generating process for a switchback experiment.

    Concrete subclasses implement :meth:`generate`, which receives an
    assignment path of length ``T`` and returns the corresponding outcome
    path of length ``T``. Implementations may be deterministic or
    stochastic; stochastic ones should respect the ``seed`` passed at
    construction so that runs are reproducible.

    The contract is intentionally minimal — anything that can answer the
    counterfactual question *"what outcomes would I observe under this
    assignment path?"* is a valid DGP, including closed-form models, full
    discrete-event simulators, or learned replay engines.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the internal RNG. If ``seed`` is None, reuse the original seed."""
        if seed is None:
            seed = self.seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, assignment: np.ndarray) -> np.ndarray:
        """Return outcomes ``Y`` of shape ``(T,)`` for ``assignment`` of shape ``(T,)``."""
        raise NotImplementedError

    def __call__(self, assignment: np.ndarray) -> np.ndarray:
        return self.generate(assignment)

    @staticmethod
    def _validate_assignment(assignment: np.ndarray) -> np.ndarray:
        """Coerce ``assignment`` to a 1-D numpy array and sanity-check it."""
        W = np.asarray(assignment)
        if W.ndim != 1:
            raise ValueError(
                f"assignment must be 1-D (shape (T,)), got shape {W.shape}"
            )
        if W.size == 0:
            raise ValueError("assignment must have at least one period")
        return W
