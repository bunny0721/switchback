"""Internal helpers for DGP modules."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np


MuLike = Union[float, Sequence[float], np.ndarray, Callable[[np.ndarray], np.ndarray]]


def resolve_mu_path(mu: MuLike, T: int) -> np.ndarray:
    """Materialise a ``mu`` specification into an array of length ``T``.

    ``mu`` may be:

    * a scalar — broadcast to every period;
    * a length-``T`` array — used as-is;
    * a callable ``t -> mu_t`` — invoked with ``np.arange(T)``, must return
      a length-``T`` array.
    """
    if callable(mu):
        mu_t = np.asarray(mu(np.arange(T)), dtype=float)
        if mu_t.shape != (T,):
            raise ValueError(
                f"mu(t) must return shape ({T},), got {mu_t.shape}"
            )
        return mu_t
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 0:
        return np.full(T, float(mu_arr))
    if mu_arr.shape == (T,):
        return mu_arr
    raise ValueError(
        f"mu must be scalar, length-T array, or callable; "
        f"got shape {mu_arr.shape} (T={T})"
    )
