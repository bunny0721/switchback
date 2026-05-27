"""Designs that produce assignment paths for switchback experiments.

A design exposes ``sample(T) -> np.ndarray`` returning a 1-D 0/1 path of
length ``T``, and ``consecutive_prob(T, t, p, value)`` returning the
probability that ``W_{t-p:t}`` is the all-``value`` vector — the IPW /
Hájek estimators use the latter to form weights.
"""

from switchback.design.adaptive_block import AdaptiveBlockDesign
from switchback.design.base import BaseDesign
from switchback.design.bernoulli import BernoulliDesign
from switchback.design.complete import CompleteRandomization

__all__ = [
    "BaseDesign",
    "BernoulliDesign",
    "CompleteRandomization",
    "AdaptiveBlockDesign",
]
