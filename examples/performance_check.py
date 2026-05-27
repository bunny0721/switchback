"""Performance check — stratified Hájek on LatentStateDGP under
Bernoulli(window_length=4), burn-in m=2.

The DGP has *infinite* (geometrically-decaying) carryover; the estimator
burns in only m=2 periods, so it is biased for the true (long-run)
average treatment effect. This script:

  1) computes the true long-run effect via the new evaluation module
     (paired-noise mean(Y(W≡1) − Y(W≡0)) over a long horizon),
  2) runs many MC replications of the stratified Hájek estimator,
  3) locates the center of the estimator distribution and reports bias,
  4) checks whether the MC distribution is approximately normal
     (skewness, excess kurtosis with SEs; QQ comparison).

If asymptotic normality holds, the next step (separately) is variance
analysis.
"""

from __future__ import annotations

import numpy as np

from switchback.design import BernoulliDesign
from switchback.dgp import LatentStateDGP
from switchback.estimators import StratifiedHajekEstimator
from switchback.evaluation import (
    normality_diagnostics,
    qq_compare,
    true_effect,
)


def main() -> None:
    T = 2_000
    n_reps = 3_000
    window_length = 4
    m = 2

    dgp_kwargs = dict(
        mu=0.0,
        beta_0=1.0,
        alpha_0=0.5,
        gamma=0.5,
        sigma_y=1.0,
        sigma_h=1.0,
    )

    # --- True long-run effect via paired-noise simulation ---
    truth_dgp = LatentStateDGP(**dgp_kwargs)
    tau_true = true_effect(truth_dgp, T=20_000, seed=0)

    # --- Monte Carlo ---
    design = BernoulliDesign(window_length=window_length, p=0.5)
    dgp = LatentStateDGP(**dgp_kwargs)
    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=n_reps):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        try:
            est = StratifiedHajekEstimator(design=design, m=m).fit(W, Y).estimate_
        except ValueError:
            continue
        estimates.append(est)
    estimates = np.asarray(estimates)

    diag = normality_diagnostics(estimates)
    qq = qq_compare(estimates)

    # --- Report ---
    print("=== Performance check ===")
    print(
        "DGP      : LatentStateDGP("
        f"beta_0={dgp_kwargs['beta_0']}, alpha_0={dgp_kwargs['alpha_0']}, "
        f"gamma={dgp_kwargs['gamma']}, sigma_y={dgp_kwargs['sigma_y']}, "
        f"sigma_h={dgp_kwargs['sigma_h']})"
    )
    print(
        f"Design   : BernoulliDesign(window_length={window_length}, p=0.5)"
    )
    print(f"Estimator: StratifiedHajekEstimator(m={m})")
    print(f"T = {T},  n_reps = {len(estimates)} (out of {n_reps} attempted)")
    print()

    bias = diag["mean"] - tau_true
    print("--- Center / bias ---")
    print(f"  True long-run effect   = beta_0 + alpha_0/(1-gamma)        = {tau_true:+.4f}")
    print(f"  Estimator center (mean MC)                                 = {diag['mean']:+.4f}")
    print(f"  Bias                   = center - truth                    = {bias:+.4f}")
    print(f"  Empirical SD           = {diag['std']:.4f}")
    print(f"  |Bias| / SD            = {abs(bias) / diag['std']:.3f}")
    print(f"  MC SE of mean          = SD / sqrt(n_reps)                 = "
          f"{diag['std'] / np.sqrt(diag['n']):.5f}")
    print()

    print("--- Normality check ---")
    z_skew = diag["skewness"] / diag["skewness_se"]
    z_kurt = diag["excess_kurtosis"] / diag["kurtosis_se"]
    print(f"  Skewness        = {diag['skewness']:+.4f}  "
          f"(SE = {diag['skewness_se']:.4f},  z = {z_skew:+.2f})")
    print(f"  Excess kurtosis = {diag['excess_kurtosis']:+.4f}  "
          f"(SE = {diag['kurtosis_se']:.4f},  z = {z_kurt:+.2f})")
    if abs(z_skew) < 3 and abs(z_kurt) < 3:
        verdict = "APPROXIMATELY NORMAL  (|z| < 3 for both moments)"
    else:
        verdict = "DEPARTS FROM NORMAL  (|z| >= 3 for at least one moment)"
    print(f"  Verdict: {verdict}")
    print()

    print("--- QQ comparison (empirical vs N(mean, std) quantiles) ---")
    print(f"  {'%':>5}  {'empirical':>12}  {'normal-fit':>12}  {'diff':>10}")
    for p, emp, nrm in qq:
        print(f"  {p:>5.0f}%  {emp:>+12.4f}  {nrm:>+12.4f}  {emp - nrm:>+10.5f}")


if __name__ == "__main__":
    main()
