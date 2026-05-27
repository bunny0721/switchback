"""Abstract base class for treatment-effect estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEstimator(ABC):
    """Maps an (assignment, outcomes) pair to a treatment-effect estimate.

    Concrete estimators implement :meth:`fit`, which stores the point
    estimate in ``self.estimate_``.
    """

    estimate_: Optional[float] = None

    @abstractmethod
    def fit(self, assignment: np.ndarray, outcomes: np.ndarray) -> "BaseEstimator":
        raise NotImplementedError

    @staticmethod
    def _validate(assignment: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = np.asarray(assignment)
        Y = np.asarray(outcomes, dtype=float)
        if W.ndim != 1 or Y.ndim != 1:
            raise ValueError("assignment and outcomes must both be 1-D")
        if W.shape != Y.shape:
            raise ValueError(
                f"assignment shape {W.shape} != outcomes shape {Y.shape}"
            )
        if W.size == 0:
            raise ValueError("inputs must have at least one period")
        if not np.all((W == 0) | (W == 1)):
            raise ValueError("assignment must contain only 0/1 entries")
        return W.astype(int), Y
