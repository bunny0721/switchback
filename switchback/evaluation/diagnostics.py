"""Distribution diagnostics for an estimator's Monte-Carlo sample."""

from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np


def normality_diagnostics(estimates) -> dict:
    """Basic moments + standard errors for an informal normality check.

    Returns a dict with ``n``, ``mean``, ``std``, ``skewness`` (and SE),
    and ``excess_kurtosis`` (and SE). Under iid normal sampling,
    skewness and excess kurtosis are mean-zero with standard errors
    ``√(6/n)`` and ``√(24/n)`` respectively, so values of ``|stat/SE|``
    below ~2-3 are roughly consistent with normality.
    """
    arr = np.asarray(estimates, dtype=float).ravel()
    n = arr.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        raise ValueError("estimates have zero variance — distribution is degenerate")
    centered = arr - mean
    skew = float(np.mean(centered ** 3) / std ** 3)
    excess_kurt = float(np.mean(centered ** 4) / std ** 4 - 3.0)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skewness": skew,
        "skewness_se": float(np.sqrt(6.0 / n)),
        "excess_kurtosis": excess_kurt,
        "kurtosis_se": float(np.sqrt(24.0 / n)),
    }


def qq_compare(
    estimates,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 95, 99),
) -> list:
    """Compare empirical quantiles to ``N(mean, std)`` quantiles.

    Returns a list of ``(percentile, empirical_quantile, normal_quantile)``
    tuples. Tightly aligned values across the range are evidence of
    approximate normality.
    """
    arr = np.asarray(estimates, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"need at least 2 observations, got {arr.size}")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0.0:
        raise ValueError("estimates have zero variance — distribution is degenerate")
    nd = NormalDist(mu=mean, sigma=std)
    return [
        (float(p), float(np.percentile(arr, p)), float(nd.inv_cdf(p / 100.0)))
        for p in percentiles
    ]
