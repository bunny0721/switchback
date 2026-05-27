"""Quickstart: run a switchback experiment end-to-end.

Pipeline:
  BernoulliDesign(window_length)              --> assignment path
  SimpleDGP(mu, tau, sigma)                  --> outcomes
  IPWEstimator(design, m)                    --> point estimate (m = burn-in length)
  HACVariance(design, estimator)    --> variance + CI

Run with:

    python examples/quickstart.py
"""

import numpy as np

from switchback.decisions import HACVariance
from switchback.design import BernoulliDesign
from switchback.dgp import SimpleDGP
from switchback.estimators import IPWEstimator


def main() -> None:
    T = 200
    tau_true = 1.0

    design = BernoulliDesign(window_length=4, seed=0)
    dgp = SimpleDGP(mu=0.0, tau=tau_true, sigma=1.0, seed=0)

    W = design.sample(T)
    Y = dgp.generate(W)

    # No carryover in SimpleDGP, so burn-in m = 0 is appropriate.
    est = IPWEstimator(design=design, m=0)
    inf = HACVariance(design=design, estimator=est).fit(W, Y)
    lo, hi = inf.confidence_interval(alpha=0.05)

    print(f"Design          : BernoulliDesign(window_length=4)")
    print(f"  -> {design.n_windows(T)} windows over T={T}")
    print(f"True tau        = {tau_true:+.3f}")
    print(f"IPW estimate    = {inf.estimate_:+.3f}")
    print(f"Design-based var= {inf.variance_:.4f}")
    print(f"95% CI          = [{lo:+.3f}, {hi:+.3f}]")
    print(f"Treated periods = {W.sum()}/{T}")


if __name__ == "__main__":
    main()
