"""Consistency check — Stratified Hájek (and pooled Hájek) on LatentStateDGP.

Setup:
  Y_t = mu_t + h_t + beta_0 W_t + eps^Y_t
  h_t = gamma h_{t-1} + alpha_0 W_t + eps^h_t

Two regimes:

  (A) l = 1, m = 2.
      Each period is its own window, so the closed form
          tau_m = beta_0 + alpha_0/(1-gamma) * (1 - gamma^(m+1))
      is the asymptotic estimand. Pooled and stratified Hájek both have
      a single stratum (B = m+1) and are identical numerically.

  (B) l = 4, m = 2 (the user's setup).
      Periods at within-window positions (0,1,2,3) split into two
      window-window-count strata: B=1 (interior, p∈{2,3}) and B=2 (cross,
      p∈{0,1}). The conditioning W_{t-m:t}=1 also fixes the coins of
      every window the window intersects, which forces some W's *outside*
      the window when t-m-1 happens to share a coin with the window.
      The closed form above no longer applies; we compute the
      asymptotic estimand empirically via a long-T baseline run and
      check that the moderate-T MC means converge to that.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from switchback.design import BernoulliDesign
from switchback.dgp import LatentStateDGP
from switchback.estimators import HajekEstimator, StratifiedHajekEstimator


def _empirical_truth(dgp_kwargs, design_kwargs, m, T_long=200_000) -> float:
    """Long-T single-rep estimate of the asymptotic Hájek target."""
    design = BernoulliDesign(seed=0, **design_kwargs)
    dgp = LatentStateDGP(seed=1, **dgp_kwargs)
    W = design.sample(T_long)
    Y = dgp.generate(W)
    return HajekEstimator(design=design, m=m).fit(W, Y).estimate_


def run_regime(
    label: str,
    dgp_kwargs: dict,
    design_kwargs: dict,
    m: int,
    T: int,
    n_reps: int,
    seed: int = 0,
) -> None:
    truth = _empirical_truth(dgp_kwargs, design_kwargs, m)

    pooled, stratified = [], []
    stratum_counts: dict[int, list[tuple[int, int]]] = defaultdict(list)

    design = BernoulliDesign(**design_kwargs)
    dgp = LatentStateDGP(**dgp_kwargs)
    rng = np.random.default_rng(seed)
    for s in rng.integers(0, 10**9, size=n_reps):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        try:
            est_h = HajekEstimator(design=design, m=m).fit(W, Y).estimate_
            est_s = StratifiedHajekEstimator(design=design, m=m).fit(W, Y)
        except ValueError:
            continue
        pooled.append(est_h)
        stratified.append(est_s.estimate_)
        for B, counts in (est_s.n_per_stratum_ or {}).items():
            stratum_counts[B].append(counts)
    pooled = np.array(pooled)
    stratified = np.array(stratified)

    # Naive closed form (assumes l = 1; shown for reference).
    beta_0 = dgp_kwargs["beta_0"]
    alpha_0 = dgp_kwargs["alpha_0"]
    gamma = dgp_kwargs["gamma"]
    naive_formula = beta_0 + alpha_0 / (1 - gamma) * (1 - gamma ** (m + 1))

    print(f"\n=== {label} ===")
    print(f"  T = {T}, n_reps = {n_reps}, m = {m}, "
          f"l = {design_kwargs['l']}")
    print(f"  Empirical asymptotic estimand (T=200k):  {truth:+.4f}")
    print(f"  Naive formula β_0 + α_0/(1-γ)·(1-γ^(m+1)):"
          f" {naive_formula:+.4f}   "
          f"{'(matches under window=1)' if design_kwargs['l']==1 else '(differs — window spillover)'}")

    for tag, arr in [("Pooled    ", pooled), ("Stratified", stratified)]:
        bias = arr.mean() - truth
        print(f"  {tag} Hájek: mean = {arr.mean():+.4f}  "
              f"bias_vs_truth = {bias:+.5f}  "
              f"SD_emp = {arr.std(ddof=1):.4f}")

    if stratum_counts:
        print("  Avg stratum sizes (n_treated, n_control):")
        for B in sorted(stratum_counts):
            sz = np.array(stratum_counts[B])
            print(f"    B = {B}:  ({sz[:, 0].mean():6.1f}, {sz[:, 1].mean():6.1f})  "
                  f"propensity (1/2)^{B} = {0.5 ** B:.3f}")


def main() -> None:
    dgp_kwargs = dict(
        mu=0.0, beta_0=1.0, alpha_0=0.5, gamma=0.5,
        sigma_y=1.0, sigma_h=1.0,
    )

    run_regime(
        label="(A) l = 1, m = 2  — formula applies",
        dgp_kwargs=dgp_kwargs,
        design_kwargs=dict(l=1, p=0.5),
        m=2, T=2_000, n_reps=1_500,
    )
    run_regime(
        label="(B) l = 4, m = 2  — window spillover, two strata",
        dgp_kwargs=dgp_kwargs,
        design_kwargs=dict(l=4, p=0.5),
        m=2, T=2_000, n_reps=1_500,
    )


if __name__ == "__main__":
    main()
