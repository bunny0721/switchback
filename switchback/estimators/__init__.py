"""Estimators that map (assignment, outcomes) to a treatment-effect estimate."""

from switchback.estimators.base import BaseEstimator
from switchback.estimators.hajek import HajekEstimator
from switchback.estimators.ipw import IPWEstimator

# Backward-compat alias: `StratifiedHajekEstimator` was the old name for the
# stratified-aware Hájek; `HajekEstimator` now has that behaviour directly.
StratifiedHajekEstimator = HajekEstimator

__all__ = [
    "BaseEstimator",
    "IPWEstimator",
    "HajekEstimator",
]
