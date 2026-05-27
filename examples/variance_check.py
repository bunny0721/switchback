"""Asymptotic-variance plug-in for stratified Hájek (BernoulliDesign, b=4, m=2).

Approach — joint HAC on the per-window influence-function sequence
----------------------------------------------------------------
The combined estimator is

    τ̂_strat = w_1 · τ̂_{B=1} + w_2 · τ̂_{B=2}

with stratum-size weights w_1 = n_1 / (n_1 + n_2), w_2 = 1 − w_1. Each
piece is asymptotically a sample mean of its influence function:

    τ̂_{B=1} ≈ τ_{B=1} + (1/n) Σ_b ξ_b
    τ̂_{B=2} ≈ τ_{B=2} + (1/n) Σ_b η_b

so the combined estimator is asymptotically a sample mean of the
weighted-sum sequence

    ζ_b = w_1 ξ_b + w_2 η_b,

and we estimate Var(τ̂_strat) by a *single* HAC on {ζ_b}:

    Var(τ̂_strat) ≈ LRV(ζ) / n,
    LRV̂(ζ) = γ̂_ζ(0) + 2 Σ_{k=1..L} K(k, L) γ̂_ζ(k).

This is algebraically identical to the three-piece decomposition

    Var(τ̂_strat) = w_1² Var(τ̂_{B=1}) + w_2² Var(τ̂_{B=2})
                  + 2 w_1 w_2 Cov(τ̂_{B=1}, τ̂_{B=2}),

each piece's HAC plug-in combined with the corresponding weight squared
or cross-product. The single-HAC form just absorbs that bookkeeping into
one autocovariance, so it's cleaner to implement and slightly cheaper.

Recommended setting: **truncated kernel, L = 1** — the structural
autocovariance for ζ is concentrated at lags 0 and ±1 (within-window
period-gap-1 coupling + adjacent-window shared coin C_b), and higher lags
are γ^7-or-smaller noise under our LatentStateDGP with γ = 0.5.
"""

from __future__ import annotations

import numpy as np

from switchback.design import BernoulliDesign
from switchback.dgp import LatentStateDGP
from switchback.estimators import StratifiedHajekEstimator


# ---------------------------------------------------------------------------
# Per-window influence-function sequences
# ---------------------------------------------------------------------------

def make_xi(W: np.ndarray, Y: np.ndarray, K: int) -> np.ndarray:
    """B=1 influence: ξ_b = 2(2 C_b − 1)(U_b − Ū_arm(C_b))."""
    T = W.size
    n = T // K
    U = np.zeros(n, dtype=float)
    C = np.zeros(n, dtype=int)
    for b in range(n):
        if 4 * b + 3 >= T:
            break
        U[b] = (Y[4 * b + 2] + Y[4 * b + 3]) / 2.0
        C[b] = int(W[4 * b + 2])
    U_bar_1 = float(U[C == 1].mean()) if (C == 1).any() else 0.0
    U_bar_0 = float(U[C == 0].mean()) if (C == 0).any() else 0.0
    mu = np.where(C == 1, U_bar_1, U_bar_0)
    return 2.0 * (2 * C - 1) * (U - mu)


def make_eta(W: np.ndarray, Y: np.ndarray, K: int) -> np.ndarray:
    """B=2 influence: η_b = 4·𝟙{b ∈ M_1}·(V_b − V̄_1) − 4·𝟙{b ∈ M_0}·(V_b − V̄_0)."""
    T = W.size
    n = T // K
    V = np.zeros(n, dtype=float)
    arm = np.zeros(n, dtype=int)
    for b in range(1, n):
        if 4 * b + 1 >= T:
            break
        V[b] = (Y[4 * b] + Y[4 * b + 1]) / 2.0
        c_prev = int(W[4 * (b - 1)])
        c_curr = int(W[4 * b])
        if c_prev != c_curr:
            arm[b] = 0
        elif c_curr == 1:
            arm[b] = 1
        else:
            arm[b] = -1
    V_bar_1 = float(V[arm == 1].mean()) if (arm == 1).any() else 0.0
    V_bar_0 = float(V[arm == -1].mean()) if (arm == -1).any() else 0.0
    eta = np.zeros(n, dtype=float)
    eta[arm == 1] = 4.0 * (V[arm == 1] - V_bar_1)
    eta[arm == -1] = -4.0 * (V[arm == -1] - V_bar_0)
    return eta


# ---------------------------------------------------------------------------
# HAC variance (Bartlett or truncated kernel) on a stationary sequence
# ---------------------------------------------------------------------------

def hac_variance(X: np.ndarray, lags: int, kernel: str = "truncated") -> float:
    """LRV(X) / n — variance of the sample mean of X under stationarity."""
    n = X.size
    if n < 2:
        return float("nan")
    dev = X - X.mean()
    sigma = float(np.sum(dev * dev) / n)
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(np.sum(dev[:-k] * dev[k:]) / n)
        if kernel == "bartlett":
            w = 1.0 - k / (lags + 1)
        elif kernel == "truncated":
            w = 1.0
        else:
            raise ValueError(f"unknown kernel: {kernel}")
        sigma += 2.0 * w * gk
    return sigma / n


# ---------------------------------------------------------------------------
# Joint variance plug-in for τ̂_strat
# ---------------------------------------------------------------------------

def stratified_hajek_variance(
    W: np.ndarray,
    Y: np.ndarray,
    n_per_stratum: dict[int, tuple[int, int]],
    K: int = 4,
    lags: int = 1,
    kernel: str = "truncated",
) -> float:
    """Single-HAC plug-in for Var(τ̂_strat) under BernoulliDesign(K, ½), m=2.

    Builds the per-window weighted influence sequence
        ζ_b = w_1 ξ_b + w_2 η_b
    and applies HAC to it.

    Parameters
    ----------
    W, Y          : observed assignment path and outcomes.
    n_per_stratum : ``StratifiedHajekEstimator.n_per_stratum_``
                    (i.e. ``{1: (n_treated_B1, n_control_B1),
                              2: (n_treated_B2, n_control_B2)}``).
    K             : window length (4 for the user's setup).
    lags, kernel  : HAC bandwidth and kernel. Recommended: ``lags=1,
                    kernel="truncated"`` for this setup.
    """
    n1 = sum(n_per_stratum.get(1, (0, 0)))
    n2 = sum(n_per_stratum.get(2, (0, 0)))
    if n1 + n2 == 0:
        return float("nan")
    w1 = n1 / (n1 + n2)
    w2 = n2 / (n1 + n2)
    xi = make_xi(W, Y, K)
    eta = make_eta(W, Y, K)
    zeta = w1 * xi + w2 * eta
    return hac_variance(zeta, lags=lags, kernel=kernel)


# ---------------------------------------------------------------------------
# MC harness
# ---------------------------------------------------------------------------

def run_mc(T: int, n_reps: int, dgp_kwargs: dict, lags: int = 1,
           kernel: str = "truncated", seed: int = 0):
    window_length, m = 4, 2
    design = BernoulliDesign(window_length=window_length, p=0.5)
    dgp = LatentStateDGP(**dgp_kwargs)

    tau_strat: list[float] = []
    plug: list[float] = []
    rng = np.random.default_rng(seed)
    for s in rng.integers(0, 10**9, size=n_reps):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        try:
            est = StratifiedHajekEstimator(design=design, m=m).fit(W, Y)
        except ValueError:
            continue
        per = est.estimate_per_stratum_ or {}
        if 1 not in per or 2 not in per:
            continue
        tau_strat.append(est.estimate_)
        plug.append(
            stratified_hajek_variance(
                W, Y, est.n_per_stratum_, K=window_length, lags=lags, kernel=kernel
            )
        )
    return np.array(tau_strat), np.array(plug)


def main() -> None:
    dgp_kwargs = dict(
        mu=0.0, beta_0=1.0, alpha_0=0.5, gamma=0.5,
        sigma_y=1.0, sigma_h=1.0,
    )

    # Headline run.
    print("=== Joint-HAC variance plug-in for stratified Hájek "
          "(BernoulliDesign, b=4, m=2) ===")
    print(f"DGP : LatentStateDGP({dgp_kwargs})")
    print(f"Plug-in: hac_variance(ζ_b = w_1 ξ_b + w_2 η_b, "
          f"lags=1, kernel='truncated')\n")

    T, n_reps = 2_000, 5_000
    taus, plug = run_mc(T, n_reps, dgp_kwargs)
    var_emp = float(np.var(taus, ddof=1))
    print(f"T = {T},  n_reps = {len(taus)}\n")
    print(f"  empirical Var(τ̂_strat)        = {var_emp:.6f}")
    print(f"  joint-HAC plug-in (mean)      = {plug.mean():.6f}")
    print(f"  joint-HAC plug-in (MC SE)     = {plug.std(ddof=1)/np.sqrt(plug.size):.6f}")
    print(f"  ratio plug-in / empirical     = {plug.mean() / var_emp:.4f}")
    print()

    # T-scaling.
    print("--- Finite-T scaling (joint HAC, truncated L=1) ---")
    print(f"  {'T':>6}  {'n_windows':>8}  {'n_reps':>7}  "
          f"{'Var_emp':>10}  {'Var̂':>10}  {'ratio':>7}")
    for T_ in (1_000, 2_000, 4_000, 8_000, 20_000):
        taus_T, plug_T = run_mc(T_, n_reps=2_000, dgp_kwargs=dgp_kwargs)
        ve = float(np.var(taus_T, ddof=1))
        vh = float(plug_T.mean())
        print(f"  {T_:>6}  {T_//4:>8}  {plug_T.size:>7}  "
              f"{ve:>10.6f}  {vh:>10.6f}  {vh/ve:>7.3f}")
    print()

    # γ scan.
    print("--- γ scan at T = 2000 (joint HAC, truncated L=1) ---")
    print(f"  {'γ':>6}  {'Var_emp':>10}  {'Var̂':>10}  {'ratio':>7}")
    for g in (0.3, 0.5, 0.7, 0.9):
        kw = dict(dgp_kwargs, gamma=g)
        taus_g, plug_g = run_mc(2_000, n_reps=2_000, dgp_kwargs=kw)
        ve = float(np.var(taus_g, ddof=1))
        vh = float(plug_g.mean())
        print(f"  {g:>6.1f}  {ve:>10.6f}  {vh:>10.6f}  {vh/ve:>7.3f}")
    print()

    # Coverage of normal-approximation CI.
    print("--- 95% CI coverage at T = 2000 (joint HAC, truncated L=1) ---")
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.975)
    # The asymptotic estimand for our setup (computed earlier via long-T MC).
    truth = 1.9208
    taus2, plug2 = run_mc(2_000, n_reps=5_000, dgp_kwargs=dgp_kwargs)
    se = np.sqrt(np.maximum(plug2, 0))
    lo = taus2 - z * se
    hi = taus2 + z * se
    covered = (lo <= truth) & (truth <= hi)
    print(f"  empirical coverage = {covered.mean() * 100:.1f}%   (target 95%)")


if __name__ == "__main__":
    main()
