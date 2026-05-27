"""Empirical diagnostics — designs × estimators × DGPs.

Setup (all with l=1, p=0.5):
  Designs:    BernoulliDesign  vs  CompleteRandomization
  Estimators: IPW (finite-sample Neyman variance)
              Hájek (asymptotic Welch variance)

  Scenario 1 — no carryover, m = 0
      Y_t = mu_t + beta_0 W_t + eps_t.  Estimand: beta_0.

  Scenario 2 — lag-1 carryover, m = 1
      Y_t = mu_t + beta_0 W_t + beta_1 W_{t-1} + eps_t.
      Estimand (lag-1 effect): beta_0 + beta_1.

For each (design, estimator) we report bias, empirical SD/Var of τ̂, mean
V̂, V̂/Var_emp (>= 1 ⇒ conservative on average), and 95% CI coverage.
Within each rep, both designs see the **same DGP noise** (shared seed for
``dgp.reset``), so cross-design comparisons are paired.

Expected pattern:

* Across estimators (within a design): Hájek's variance is strictly
  smaller than IPW's whenever outcomes have a non-zero level — IPW carries
  a level-dependent term that Hájek does not.

* Across designs (within an estimator): CompleteRandomization fixes the
  treated count at exactly T/2, eliminating sample-size variability and
  yielding slightly smaller true variance than Bernoulli. The IPW V̂
  formula is exact under Bernoulli; under CompleteRandomization it
  remains conservative but somewhat looser (it ignores negative
  cross-window correlations introduced by the fixed-total constraint).
"""

from __future__ import annotations

import numpy as np

from switchback.decisions import HACVariance
from switchback.design import BernoulliDesign, CompleteRandomization
from switchback.dgp import BaseDGP, CarryoverDGP, SimpleDGP
from switchback.estimators import HajekEstimator, IPWEstimator


def run_scenario(
    label: str,
    dgp: BaseDGP,
    m: int,
    true_effect: float,
    T: int = 400,
    n_reps: int = 3_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    designs = [
        ("Bernoulli", lambda: BernoulliDesign(l=1, p=0.5)),
        ("Complete ", lambda: CompleteRandomization(l=1)),
    ]
    estimators = [
        ("IPW", IPWEstimator),
        ("Hájek", HajekEstimator),
    ]

    cells = {
        (d, e): {"est": [], "var": [], "cov": []}
        for d, _ in designs
        for e, _ in estimators
    }

    for s in rng.integers(0, 10**9, size=n_reps):
        for d_name, d_factory in designs:
            design = d_factory()
            design.reset(seed=int(s))
            dgp.reset(seed=int(s) + 1)        # paired noise across designs
            W = design.sample(T)
            Y = dgp.generate(W)

            for e_name, est_cls in estimators:
                try:
                    inf = HACVariance(
                        design=design, estimator=est_cls(design=design, m=m)
                    ).fit(W, Y)
                except ValueError:
                    continue                    # skip degenerate reps
                cells[(d_name, e_name)]["est"].append(inf.estimate_)
                cells[(d_name, e_name)]["var"].append(inf.variance_)
                lo, hi = inf.confidence_interval(alpha=alpha)
                cells[(d_name, e_name)]["cov"].append(lo <= true_effect <= hi)

    print(f"\n=== {label} ===")
    print(f"  T = {T},  n_reps = {n_reps},  true effect = {true_effect:+.4f}")
    print(
        f"  {'design':<10}  {'estimator':<6}  "
        f"{'bias':>9}  {'SD_emp':>9}  {'Var_emp':>9}  "
        f"{'mean V̂':>9}  {'V̂/Var_emp':>10}  {'cov 95%':>8}"
    )
    print(f"  {'-' * 86}")
    for d_name, _ in designs:
        for e_name, _ in estimators:
            r = cells[(d_name, e_name)]
            ests = np.array(r["est"])
            vars_ = np.array(r["var"])
            bias = ests.mean() - true_effect
            sd_emp = ests.std(ddof=1)
            var_emp = ests.var(ddof=1)
            ratio = vars_.mean() / var_emp
            cov = np.mean(r["cov"]) * 100.0
            print(
                f"  {d_name:<10}  {e_name:<6}  "
                f"{bias:+9.4f}  {sd_emp:9.4f}  {var_emp:9.5f}  "
                f"{vars_.mean():9.5f}  {ratio:10.3f}  {cov:7.1f}%"
            )


def main() -> None:
    T, n_reps = 400, 3_000

    run_scenario(
        label="Scenario 1 — Y_t = mu + beta_0 W_t + eps_t  (window=1, m=0)",
        dgp=SimpleDGP(mu=0.0, tau=1.0, sigma=1.0),
        m=0,
        true_effect=1.0,
        T=T,
        n_reps=n_reps,
    )

    run_scenario(
        label="Scenario 2 — Y_t = mu + beta_0 W_t + beta_1 W_{t-1} + eps_t  (window=1, m=1)",
        dgp=CarryoverDGP(betas=[1.0, 0.5], mu=0.0, sigma=1.0),
        m=1,
        true_effect=1.0 + 0.5,
        T=T,
        n_reps=n_reps,
    )


if __name__ == "__main__":
    main()
