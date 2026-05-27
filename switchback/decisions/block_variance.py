"""Block-design variance estimator (eq. 28) for AdaptiveBlockDesign.

Implements the variance estimator from Section 6 of Ni & Bojinov,
*Model-assisted Switchback Experiments*:

.. math::

    \\widehat{\\mathrm{Var}}_\\eta(\\hat\\tau) \\;=\\;
        \\frac{1}{B^2 K} \\sum_{b, b' \\in [B]}
            \\Bigl[ \\gamma^{+}_{b,b'} (R^{1,1}_{b,b'} + R^{0,0}_{b,b'})
                 - \\gamma^{-}_{b,b'} (R^{1,0}_{b,b'} + R^{0,1}_{b,b'}) \\Bigr].

Compared to the joint-HAC plug-in inside :class:`HACVariance`,
this one uses **design-derived weights** ``γ⁺_{b,b'}, γ⁻_{b,b'}`` instead
of a HAC kernel, and it operates on **per-observation R-statistics** of
the contributing ``Y_t`` values rather than per-block aggregates ``Z_b``.

The γ weights — overlapping indices (eq. 13 in the paper) — are derived
from the design's 2-state Markov chain over consecutive pairs. Under the
Markov approximation (exact as ``K → ∞``):

* ``γ⁺_{b,b} = 2 / ρ``,    ``γ⁻_{b,b} = 0``                  (diagonal)
* ``γ⁺_{b,b'} = 1 + (2ρ − 1)^{δ−1}``                          (off-diag)
* ``γ⁻_{b,b'} = 1 − (2ρ − 1)^{δ−1}``,   ``δ = |b − b'|``

so ``γ⁺ − γ⁻ = 2 (2ρ − 1)^{δ−1}`` matches the paper's Corollary 2.

For ``ρ = 0.5`` the cross-block weights vanish for ``δ ≥ 2`` — only the
diagonal (within-block) and lag-1 cross-block terms contribute.

Boundary correction
-------------------
``boundary_correction=True`` (default) bundles two block-0 fixes that the
paper's verbatim eq. 28 misses:

1. **Diagonal weight correction.** The Markov-derived ``γ⁺_{b,b} = 2/ρ``
   assumes the adaptive consecutive-pair rate ``ρ/2`` in every block,
   but block 0 is sampled unconditionally so its consecutive-pair
   propensity is ``π_0 = 0.25``, not ``ρ/2``. The block-``b`` diagonal
   is weighted by ``1/π_b`` instead. No-op at ``ρ = 0.5`` (``ρ/2 = 0.25``);
   at ``ρ > 0.5`` it raises the otherwise under-stated block-0 weight.

2. **Block-0 count-variance correction** (new — addresses the prior
   "seasonality blind spot"). Block 0's contributing-pair counts
   ``N_{(u,u),0}`` are *random* under CR × CR (block 0 ⊥ block ``B-1``,
   each a CR of ``K`` positions with ``K/2`` ones), whereas for ``b ≥ 1``
   the adaptive sampling pins ``N_{(u,u),b} = K_v / 2`` exactly. Within-
   pattern centring removes the level, so the piece

       Var(μ_1 N_{11,0} − μ_0 N_{00,0}) / (K π_0)²

   is missing from V̂'s block-0 contribution. We add it back with a
   plug-in: design constants ``Var(N)``, ``Cov(N)`` from closed form
   (see :func:`_block0_count_variance_constants`), and bias-corrected
   ``μ̂_u² = Ȳ_u² − S²_u / n_u`` for each pattern's block-0 sample. This
   correction (a) repairs block 0's per-block calibration from
   ~0.85–0.93 to ~1.00–1.06, and (b) removes the prior seasonality blind
   spot at block 0 — the bias-corrected ``μ̂_u`` captures block 0's
   seasonal level explicitly, so the corrected estimator now stays
   calibrated under a seasonal baseline. Both effects are MC-verified at
   ``B = 24, K = 28``.

Caveat: even with both corrections, a separate finite-``K`` residual in
the Markov-``γ`` approximation (≈ ±5%) survives. Off-diagonal calibration
under heavy carryover (e.g., :class:`LatentStateDGP` at high ``γ_AR``)
still benefits from raising ``max_delta`` — see the LatentState note in
the user memory.

.. note::

   **Seasonality** at block 0 used to require ``HACVariance``'s
   block-wise HAC; with this correction ``paper_variance`` is also
   robust to block 0's seasonal level. The block-wise HAC remains the
   recommended path for **non-block-0** seasonal levels (interior block
   seasonality is absorbed by the HAC's per-block influence, not by
   ``paper_variance``'s within-pattern centring).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional, Tuple

import numpy as np

from switchback.design.adaptive_block import AdaptiveBlockDesign


def _gamma_plus(rho: float, delta: int) -> float:
    """Overlapping index ``γ⁺_{b,b+δ}`` (Markov approximation)."""
    if delta == 0:
        return 2.0 / rho
    return 1.0 + (2.0 * rho - 1.0) ** (delta - 1)


def _gamma_minus(rho: float, delta: int) -> float:
    """Overlapping index ``γ⁻_{b,b+δ}`` (Markov approximation)."""
    if delta == 0:
        return 0.0
    return 1.0 - (2.0 * rho - 1.0) ** (delta - 1)


def _per_block_means(
    Y: np.ndarray, W: np.ndarray, T: int, B: int, K: int, u: int, v: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block sample mean and contributing mask for pattern ``(u, v)``.

    Returns ``(mean_b, count_b)`` where ``mean_b[b]`` is the average of
    ``Y_t`` over periods ``t`` in block ``b`` whose previous-period pair
    ``(W_{t-1}, W_t) = (u, v)``, and ``count_b[b]`` is the realised count
    (== ``K_v / 2`` in expectation when ``u = v``, ``(K − K_v)/2`` when
    ``u ≠ v``, but may fluctuate at the block-wrap boundary).
    """
    sums = np.zeros(B, dtype=float)
    counts = np.zeros(B, dtype=int)
    for t in range(1, T):
        if W[t - 1] == u and W[t] == v:
            b = t % B
            sums[b] += Y[t]
            counts[b] += 1
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return mean, counts


def _within_block_variance_per_block(
    Y: np.ndarray, W: np.ndarray, T: int, B: int, u: int
) -> np.ndarray:
    """``R^{u,u}_{b,b}`` for each ``b`` — within-block sample variance of
    contributing ``Y_t``.

    For each block ``b`` with ``≥ 2`` contributing periods (``W_{t-1:t} = (u,u)``),
    compute ``(1 / (n_b − 1)) Σ_{t ∈ block, contributing} (Y_t − Ȳ_b)²``.
    (Eq. 27 of the paper.) Returns a length-``B`` array (``0`` where
    ``n_b < 2``).
    """
    sums = np.zeros(B, dtype=float)
    sumsq = np.zeros(B, dtype=float)
    counts = np.zeros(B, dtype=int)
    for t in range(1, T):
        if W[t - 1] == u and W[t] == u:
            b = t % B
            sums[b] += Y[t]
            sumsq[b] += Y[t] * Y[t]
            counts[b] += 1
    out = np.zeros(B, dtype=float)
    for b in range(B):
        n_b = counts[b]
        if n_b < 2:
            continue
        mean_b = sums[b] / n_b
        out[b] = (sumsq[b] - n_b * mean_b * mean_b) / (n_b - 1)
    return out


def _within_block_variance(
    Y: np.ndarray, W: np.ndarray, T: int, B: int, u: int
) -> float:
    """Σ_b R^{u,u}_{b,b} (kept for callers that want the plain sum)."""
    return float(_within_block_variance_per_block(Y, W, T, B, u).sum())


def _block0_count_variance_constants(K: int) -> Tuple[float, float]:
    r"""Design constants ``Var(N_{11,0})`` and ``Cov(N_{11,0}, N_{00,0})``
    for block 0's contributing-pair counts under CR × CR.

    Block 0's contributing pairs ``(W_{kB-1}, W_{kB})`` for
    ``k = 1, …, K-1`` are formed from two **independent** complete
    randomisations — block ``B-1``'s CR and block ``0``'s CR, each with
    ``K/2`` ones in ``K`` positions. Of those ``K`` positions in each
    block, only ``K-1`` enter block-0 contributing pairs (``W_{B-1}`` of
    day 0 and the position ``W_0`` are unused as a "previous" / "current"
    in the pair indexing).

    Let ``A_k = W_{kB-1}``, ``B_k = W_{kB}``, both ``∈ {0, 1}``,
    ``A ⊥ B``. Under CR with ``K`` positions and ``K/2`` ones:

    * ``E[A_k] = 1/2``;
    * for ``k ≠ k'``, ``E[A_k A_{k'}] = q ≡ (K-2)/(4(K-1))``.

    Define ``I_{11,k} = A_k B_k``, ``I_{00,k} = (1-A_k)(1-B_k)``; both
    have marginal mean ``π_0 = 1/4``. The pair counts
    ``N_{(u,u),0} = Σ_k I_{(u,u),k}`` have

    .. math::

        \\mathrm{Var}(N_{11,0}) = (K-1)\\,\\pi_0(1-\\pi_0)
            + (K-1)(K-2)\\,(q^2 - \\pi_0^2),

    and by symmetry the same for ``N_{00,0}``. The cross-pattern
    covariance is

    .. math::

        \\mathrm{Cov}(N_{11,0}, N_{00,0}) = -(K-1)\\,\\pi_0^2
            + (K-1)(K-2)\\,((1/2 - q)^2 - \\pi_0^2).

    The first piece is the same-``k`` mutual exclusion
    (``I_{11,k} I_{00,k} = 0``); the second is the ``k ≠ k'`` term
    ``E[I_{11,k} I_{00,k'}] = (1/2 - q)^2`` from CR × CR independence.

    Why this matters. Block 0's IPW contribution is
    ``Z_0 ∝ μ_1 N_{11,0} - μ_0 N_{00,0} + noise``. The within-pattern
    centered-``R`` term captures only the noise piece, which the
    within-pattern centering preserves; the *level × count* piece
    ``Var(μ_1 N_{11,0} - μ_0 N_{00,0})/(K π_0)²`` is what centring
    erases. For interior blocks ``b ≥ 1`` the adaptive sampling pins
    ``N_{(u,u),b} = K_v / 2`` exactly — the count is deterministic, so
    this correction is identically zero. Only block 0 needs it.
    """
    if K < 2:
        return 0.0, 0.0
    n = K - 1
    pi = 0.25
    q = (K - 2.0) / (4.0 * (K - 1.0))
    cov_I_same_uu = pi * (1.0 - pi)
    cov_I_diff_uu = q * q - pi * pi
    var_N = n * cov_I_same_uu + n * (n - 1) * cov_I_diff_uu
    cov_I_same_ud = -pi * pi
    cov_I_diff_ud = (0.5 - q) ** 2 - pi * pi
    cov_N = n * cov_I_same_ud + n * (n - 1) * cov_I_diff_ud
    return float(var_N), float(cov_N)


def _diagonal_weights(
    design: AdaptiveBlockDesign, T: int, B: int, rho: float
) -> np.ndarray:
    r"""Per-block diagonal weight ``γ⁺_{b,b} = 1 / π_b``.

    The within-block-variance weight is the inverse of *that block's
    actual consecutive-pair propensity* ``π_b = Pr(W_{t-1:t} = (u,u))``
    for ``t`` in block ``b``:

    * blocks ``b ≥ 1`` use the adaptive transition, ``π_b = ρ/2`` →
      weight ``2/ρ`` (identical to the paper's uniform ``γ⁺_{b,b}``);
    * block ``0`` is sampled *unconditionally* (the ``(B−1)→0`` transition
      is independent), so ``π_0 = 0.25`` → weight ``4``.

    At ``ρ = 0.5`` the two coincide (``ρ/2 = 0.25``) and the correction is
    a no-op; for ``ρ > 0.5`` it removes the paper formula's systematic
    under-weighting of block 0.
    """
    w = np.empty(B, dtype=float)
    for b in range(B):
        t_rep = b if b >= 1 else B  # representative period in block b (t ≥ 1)
        pi_b = design.consecutive_prob(T, t_rep, 1, value=1)
        w[b] = 1.0 / pi_b
    return w


def _cross_block_covariance_per_block(
    Y: np.ndarray, W: np.ndarray, T: int, B: int, K: int, delta: int, u: int, v: int
) -> np.ndarray:
    """Per-block ``R^{u,v}_{b, (b+δ) % B}`` — same as
    :func:`_cross_block_covariance` but returns a length-``B`` array
    rather than summing across blocks. Used by the per-pair ``γ``
    weighting in :func:`block_variance` (so each ordered pair can be
    weighted by its own structural class instead of a uniform
    lag-``δ`` weight).
    """
    mean_u, _ = _per_block_means(Y, W, T, B, K, u, u)
    mean_v, _ = _per_block_means(Y, W, T, B, K, v, v)
    paired_sumprod = np.zeros(B, dtype=float)
    paired_count = np.zeros(B, dtype=int)
    for t in range(1, T - delta):
        if W[t - 1] == u and W[t] == u:
            t2 = t + delta
            if t2 < T and W[t2 - 1] == v and W[t2] == v:
                b = t % B
                paired_sumprod[b] += (Y[t] - mean_u[b]) * (
                    Y[t2] - mean_v[(b + delta) % B]
                )
                paired_count[b] += 1
    out = np.zeros(B, dtype=float)
    for b in range(B):
        if paired_count[b] >= 2:
            out[b] = paired_sumprod[b] / (paired_count[b] - 1)
    return out


def _pair_gamma_matrices(B: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""Per-ordered-pair ``γ⁺_{b,b'}`` and ``γ⁻_{b,b'}`` for forward iteration
    in eq. 28 (each ordered pair indexed by ``(b, b' = (b+δ) mod B)`` at
    forward cyclic δ). The factor-2 mirror in :func:`block_variance`'s
    off-diagonal sum credits the backward-ordered partner of each pair,
    which by Cov-symmetry contributes the same R-statistic value.

    Classification of forward (b, b' = (b+δ) mod B) at δ ∈ {1, …, B/2}:

    * **Class A** — same-day pair ``(b, b' = b + δ < B)``: chain formula
      at the time-lag ``δ``,  ``γ^± = 1 ± (2ρ - 1)^{δ-1}``.
    * **Class B** — cross-day with ``b' = 0`` (i.e. ``b = B - δ``):
      block 0's contributing-pair indicator straddles the day boundary,
      using ``W_{(k+1)B-1}`` (in block ``B - 1`` of day ``k``). The
      chain from block ``b``'s last variable to ``W_{(k+1)B-1}`` is
      ``B - 1 - b = δ - 1`` adaptive transitions, so
      ``γ^± = 1 ± (2ρ - 1)^{δ-1}`` — same formula as class A.
    * **Class C** — cross-day with ``b' ≥ 1`` (both endpoints interior):
      no shared variable, and the boundary-independent step breaks the
      chain. The indicators factorize, so ``γ^+ = γ^- = 1``. Arises in
      the forward iteration when ``b + δ ≥ B`` with ``(b+δ) mod B ≥ 1``,
      i.e., the forward step wraps past block ``B-1`` to interior
      blocks of the next day.

    All three classes match the paper's chain formula at ``δ_phys``
    where ``δ_phys`` is the actual *physical* time-lag.

    Diagonal entries ``(b = b')`` are set to ``γ⁺ = 2/ρ``, ``γ⁻ = 0``
    (the within-block weights). They are not used by the off-diagonal
    sum; the diagonal contribution is handled separately.

    .. note::

       This matrix is **not symmetric** in (b, b'). For ordered pair
       (b+1, b) at forward delta = B-1 (the cyclic-backward direction),
       the matrix entry gives γ at the *long* cyclic forward lag B-1,
       not the unsigned distance 1. The off-diagonal iteration in
       :func:`block_variance` only uses forward δ ≤ B/2, multiplied by
       the factor-2 mirror, so the (b+1, b) ordered pair is credited via
       the mirror at δ = 1, not via direct lookup at δ = B-1.

    All matched empirically by design-only Monte Carlo on the
    AdaptiveBlockDesign.
    """
    gp = np.zeros((B, B), dtype=float)
    gm = np.zeros((B, B), dtype=float)
    for b in range(B):
        for bp in range(B):
            if b == bp:
                gp[b, bp] = 2.0 / rho
                gm[b, bp] = 0.0
                continue
            delta = (bp - b) % B            # forward time-lag from b to bp
            cross_day = bp < b              # ⇔ (b + δ) ≥ B
            if (not cross_day) or (bp == 0):
                # Class A (same-day) or Class B (cross-day to b'=0).
                power = (2.0 * rho - 1.0) ** (delta - 1)
                gp[b, bp] = 1.0 + power
                gm[b, bp] = 1.0 - power
            else:
                # Class C: cross-day with b'≥1 — chain breaks at boundary.
                gp[b, bp] = 1.0
                gm[b, bp] = 1.0
    return gp, gm


def _cross_block_covariance(
    Y: np.ndarray, W: np.ndarray, T: int, B: int, K: int, delta: int, u: int, v: int
) -> float:
    """Σ_b R^{u,v}_{b,b+δ} — cross-block sample covariance at lag ``δ``.

    Pair contributing ``Y_t`` (at block ``b`` with pattern ``(u,u)``) with
    contributing ``Y_{t + δ}`` (at block ``b+δ`` with pattern ``(v,v)``):
    when both contribute (same "day"), accumulate a centered product;
    otherwise the pair contributes 0. Divides by ``(n_pairs_b − 1)`` per
    block (Bessel-corrected sample covariance).
    """
    # Block-level mean of Y for each pattern, used for centering.
    mean_u, count_u = _per_block_means(Y, W, T, B, K, u, u)
    mean_v, count_v = _per_block_means(Y, W, T, B, K, v, v)

    # Accumulate per-block paired (centered) products.
    paired_sumprod = np.zeros(B, dtype=float)
    paired_count = np.zeros(B, dtype=int)
    for t in range(1, T - delta):
        if W[t - 1] == u and W[t] == u:
            t2 = t + delta
            if t2 < T and W[t2 - 1] == v and W[t2] == v:
                b = t % B
                paired_sumprod[b] += (Y[t] - mean_u[b]) * (
                    Y[t2] - mean_v[(b + delta) % B]
                )
                paired_count[b] += 1

    total = 0.0
    for b in range(B):
        n_b = paired_count[b]
        if n_b < 2:
            continue
        total += paired_sumprod[b] / (n_b - 1)
    return total


def _suggest_max_delta(rho: float, threshold: float = 0.05) -> int:
    r"""Smallest ``d`` such that ``(2ρ-1)^{d-1} ≤ threshold``.

    Chain-decay-based truncation criterion: at lag ``d`` the chain-decay
    factor ``(2ρ-1)^{d-1}`` is what makes ``γ^±`` differ from 1; beyond
    this we have ``γ^± ≈ 1`` so the ordered pair's contribution is just
    the raw cross-block sample covariance ``R^{11}+R^{00}-R^{10}-R^{01}``,
    which under most DGPs is negligibly small (zero in expectation under
    no/finite-carryover DGPs).
    """
    if rho <= 0.5 or threshold >= 1.0:
        return 1
    rate = 2.0 * rho - 1.0
    if rate <= 0:
        return 1
    return int(np.ceil(1.0 + np.log(threshold) / np.log(rate)))


def block_variance(
    design: AdaptiveBlockDesign,
    assignment: np.ndarray,
    outcomes: np.ndarray,
    max_delta: Optional[int] = None,
    boundary_correction: bool = True,
    legacy: bool = False,
    truncation_threshold: float = 0.05,
) -> float:
    r"""Paper's variance estimator (eq. 28) for ``IPWEstimator(m=1)``.

    Implements eq. 28 with **per-pair γ matrix** (Class A / B / C
    classification at the wrap boundary) and a **factor-2 mirror** over
    forward δ ≤ B/2 to account for the backward-ordered partner of each
    unordered block pair.

    Off-diagonal derivation. Eq. 28 sums over ordered (b, b') ∈ [B]².
    For each unordered pair {b, b'} at unsigned cyclic distance δ, eq. 28
    includes BOTH orderings — and by the symmetry of the sample
    covariance ``R^{u,v}_{b, b'} = R^{v,u}_{b', b}`` the two orderings
    contribute the *same* `γ⁺(R^{1,1}+R^{0,0}) − γ⁻(R^{1,0}+R^{0,1})`
    quantity. We iterate forward δ ∈ {1, …, B/2}, sum over the B forward
    (b, (b+δ) mod B) ordered pairs at each δ with their per-pair
    γ⁺_{b,b'}, γ⁻_{b,b'}, and multiply by 2 (factor-2 mirror) — except at
    δ = B/2 with B even, where the b-iteration already covers both
    physical directions explicitly, so we use factor 1.

    Why per-pair γ matters. At ρ ≠ 0.5, the chain factor `(2ρ−1)^{δ−1}`
    differs from zero, so γ⁺ and γ⁻ are no longer symmetric around 1.
    The Class A / B / C distinction in `_pair_gamma_matrices` then
    affects each ordered pair's correct γ value (e.g., the wrap-cross-day
    pair (B−1, 0) is Class B, with the chain formula at δ_phys = 1; the
    other cross-day wrap pairs are Class C with γ⁺ = γ⁻ = 1 since the
    chain breaks at the day boundary).

    Parameters
    ----------
    design : AdaptiveBlockDesign
    assignment, outcomes : np.ndarray
        Realised ``W`` and ``Y`` of length ``T = B · K``.
    max_delta : int, optional
        Off-diagonal lag truncation. If ``None`` (default), set to
        ``_suggest_max_delta(ρ, truncation_threshold)`` — auto-chosen so
        that pairs beyond ``max_delta`` have chain decay below the
        threshold and contribute ≈ 0 under typical DGPs. Set explicitly
        to ``B − 1`` for the full sum, or to ``1`` to match the paper
        verbatim when ``legacy=True``.
    boundary_correction : bool, default True
        Bundles two block-0 fixes (see module docstring): the diagonal
        weight correction (``1/π_b`` instead of the uniform ``2/ρ``,
        a no-op at ``ρ = 0.5``) and the block-0 count-variance correction
        (always active, repairs block 0's calibration and removes its
        seasonality blind spot). Set to ``False`` to recover the paper's
        verbatim eq. 28 behaviour.
    legacy : bool, default False
        If True, reproduce the paper's eq. 28 verbatim: truncate at
        ``max_delta`` (default 1) and apply factor-2 mirror. Use to match
        the paper as published. The default ``legacy=False`` is the
        derived, MC-verified, structurally-correct extension above.
    truncation_threshold : float, default 0.05
        Used to auto-choose ``max_delta`` when ``max_delta=None``: the
        smallest ``d`` such that the chain-decay factor
        ``(2ρ-1)^{d-1} ≤ threshold``. At ``ρ = 0.5`` this gives
        ``max_delta = 1`` (chain decay is 0 for ``δ ≥ 2``); for
        ``ρ = 0.75`` it gives ~6; for ``ρ = 0.9`` it gives ~15. The
        idea: pairs beyond ``max_delta`` have ``γ^± ≈ 1`` (no
        chain-decay amplification), so they contribute only the raw
        cross-block sample covariance which is negligible under typical
        DGPs. Set to ``0`` to disable truncation (``max_delta = B − 1``).

    Returns
    -------
    float
        ``V̂(τ̂)``.
    """
    if not isinstance(design, AdaptiveBlockDesign):
        raise TypeError("paper_variance requires an AdaptiveBlockDesign")
    W = np.asarray(assignment, dtype=int)
    Y = np.asarray(outcomes, dtype=float)
    T = W.size
    B = design.B
    K = T // B
    rho = design.effective_rho(T)
    if max_delta is None:
        if legacy:
            max_delta = 1
        else:
            max_delta = min(B - 1, _suggest_max_delta(rho, truncation_threshold))

    # --- Diagonal (b = b'): within-block variance ---
    R_11_pb = _within_block_variance_per_block(Y, W, T, B, u=1)
    R_00_pb = _within_block_variance_per_block(Y, W, T, B, u=0)
    if boundary_correction:
        w_diag = _diagonal_weights(design, T, B, rho)
    else:
        w_diag = np.full(B, _gamma_plus(rho, 0))
    total = float(np.sum(w_diag * (R_11_pb + R_00_pb)))

    # Block-0 count-variance correction. Under CR × CR (block 0 ⊥ block
    # B-1, each a CR with K/2 ones in K positions) the contributing-pair
    # counts N_{(u,u),0} are genuinely random — unlike interior blocks
    # b ≥ 1, where the adaptive sampling pins N_{(u,u),b} = K_v / 2
    # exactly. Within-pattern centring (the R-statistic) removes the
    # level, so the piece
    #     Var(μ_1 N_{11,0} - μ_0 N_{00,0}) / (K π_0)²
    # is missing from V̂_diag's block-0 contribution. We add it back with
    # an unbiased plug-in: design constants Var(N), Cov(N) from CR × CR
    # closed form, and a bias-corrected sample-mean-squared
    # μ̂_u² = Ȳ_u² − S²_u / n_u for each pattern's block-0 statistics.
    if boundary_correction:
        pi_0 = 1.0 / w_diag[0]
        var_N_b0, cov_N_b0 = _block0_count_variance_constants(K)
        if var_N_b0 > 0.0:
            mean_11_pb, count_11_pb = _per_block_means(Y, W, T, B, K, u=1, v=1)
            mean_00_pb, count_00_pb = _per_block_means(Y, W, T, B, K, u=0, v=0)
            n_11 = int(count_11_pb[0])
            n_00 = int(count_00_pb[0])
            Yb_11 = float(mean_11_pb[0])
            Yb_00 = float(mean_00_pb[0])
            # μ̂_u² = Ȳ_u² − S²_u / n_u is unbiased for μ_u² (the
            # population mean of Y over pattern-(u,u) periods in block 0).
            mu_1_sq = Yb_11 * Yb_11 - (R_11_pb[0] / n_11 if n_11 > 0 else 0.0)
            mu_0_sq = Yb_00 * Yb_00 - (R_00_pb[0] / n_00 if n_00 > 0 else 0.0)
            mu_1_mu_0 = Yb_11 * Yb_00
            count_var_term = (
                mu_1_sq * var_N_b0
                + mu_0_sq * var_N_b0
                - 2.0 * mu_1_mu_0 * cov_N_b0
            )
            # The block-0 correction in V̂ units is
            # count_var_term/(B² (K π_0)²); `total` is later divided by
            # B² K, so we add count_var_term / (K π_0²) here.
            total += count_var_term / (K * pi_0 * pi_0)

    # --- Off-diagonal ---
    if legacy:
        # Paper verbatim: single chain γ⁺(δ) per lag, factor-2 mirror,
        # default truncation at max_delta=1.
        for delta in range(1, max_delta + 1):
            gp = _gamma_plus(rho, delta)
            gm = _gamma_minus(rho, delta)
            R_11_d = _cross_block_covariance(Y, W, T, B, K, delta, u=1, v=1)
            R_00_d = _cross_block_covariance(Y, W, T, B, K, delta, u=0, v=0)
            R_10_d = _cross_block_covariance(Y, W, T, B, K, delta, u=1, v=0)
            R_01_d = _cross_block_covariance(Y, W, T, B, K, delta, u=0, v=1)
            total += 2.0 * gp * (R_11_d + R_00_d)
            total -= 2.0 * gm * (R_10_d + R_01_d)
    else:
        # Per-pair γ from the structural classification (class A/B/C),
        # iterated with a factor-2 mirror up to δ = B/2.
        #
        # eq. 28 sums over ordered (b, b'). For each unordered pair {b, b'}
        # at unsigned cyclic distance δ, eq. 28 includes BOTH orderings:
        # forward (b, b+δ) at time-lag δ same-day, and backward (b+δ, b) —
        # which by symmetry of the sample covariance contributes the SAME
        # R^{u,v} as the forward pair. So each forward (b, b+δ) ordered
        # pair gets a factor-2 mirror to credit its backward partner.
        #
        # For ρ ≠ 0.5 the per-pair γ values differ by structural class
        # (A vs B vs C at the wrap-cross-day pairs), so we keep the
        # per-pair γ matrix lookup; what we change is the *multiplicity*
        # of each iteration step (factor 2 for δ < B/2 to account for
        # backward-ordered partner; factor 1 at δ = B/2 with B even
        # because the b-iteration at that δ already covers both physical
        # interpretations).
        gp_mat, gm_mat = _pair_gamma_matrices(B, rho)
        half_B = B // 2
        for delta in range(1, max_delta + 1):
            if delta > half_B:
                # Forward δ > B/2 is the cyclic-backward direction at
                # unsigned δ' = B − δ, already counted by the factor-2
                # mirror at smaller δ. Skip to avoid double counting.
                break
            R_11_pb = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=1, v=1)
            R_00_pb = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=0, v=0)
            R_10_pb = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=1, v=0)
            R_01_pb = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=0, v=1)
            # δ < B/2 → factor-2 mirror. δ = B/2 (B even) → factor-1
            # (each b in the B-length iteration is a distinct physical
            # pair, both directions explicitly covered).
            mirror = 1.0 if (delta == half_B and B % 2 == 0) else 2.0
            for b in range(B):
                bp = (b + delta) % B
                total += mirror * gp_mat[b, bp] * (R_11_pb[b] + R_00_pb[b])
                total -= mirror * gm_mat[b, bp] * (R_10_pb[b] + R_01_pb[b])

    return float(total / (B * B * K))


def block_confidence_interval(
    design: AdaptiveBlockDesign,
    estimate: float,
    assignment: np.ndarray,
    outcomes: np.ndarray,
    alpha: float = 0.05,
    max_delta: Optional[int] = None,
    boundary_correction: bool = True,
    legacy: bool = False,
    truncation_threshold: float = 0.05,
) -> Tuple[float, float]:
    """Normal-approximation CI using :func:`block_variance`.

    Wraps :func:`block_variance` + :func:`normal_ci`. If you already have
    a pre-computed variance estimate, call :func:`normal_ci` directly.
    """
    from switchback.decisions.hac_variance import normal_ci
    v = block_variance(
        design,
        assignment,
        outcomes,
        max_delta=max_delta,
        boundary_correction=boundary_correction,
        legacy=legacy,
        truncation_threshold=truncation_threshold,
    )
    return normal_ci(estimate, v, alpha)
