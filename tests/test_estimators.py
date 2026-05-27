"""Tests for switchback.estimators.IPWEstimator."""

import numpy as np
import pytest

from switchback.design import BernoulliDesign
from switchback.dgp import SimpleDGP
from switchback.estimators import (
    HajekEstimator,
    IPWEstimator,
    StratifiedHajekEstimator,
)


# ---------------------------------------------------------------------------
# Hand-checked algebra
# ---------------------------------------------------------------------------

def test_per_period_bernoulli_T4_m1_hand_computed():
    """T=4, m=1, BernoulliDesign(l=1), W=(1,1,0,0), Y=(10,20,30,40).

    For each t in {1,2,3} (0-indexed):
      t=1: window [0,1]=(1,1) -> +Y[1] / (1/2)^2 = 80
      t=2: window [1,2]=(1,0) mixed -> 0
      t=3: window [2,3]=(0,0) -> -Y[3] / (1/2)^2 = -160
    sum / (T-m) = (80 - 160) / 3 = -80/3.
    """
    design = BernoulliDesign(l=1)
    est = IPWEstimator(design=design, m=1).fit(
        np.array([1, 1, 0, 0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    assert est.estimate_ == pytest.approx(-80.0 / 3.0)


def test_window_length_changes_propensities():
    """With l=2, the same W has higher propensities -> different estimate.

    T=4, m=1, l=2, W=(1,1,0,0):
      t=1: window [0,1]=(1,1), one window -> +Y[1]/(1/2) = 40
      t=2: window [1,2]=(1,0) mixed -> 0
      t=3: window [2,3]=(0,0), one window -> -Y[3]/(1/2) = -80
    sum / 3 = -40/3.
    """
    design = BernoulliDesign(l=2)
    est = IPWEstimator(design=design, m=1).fit(
        np.array([1, 1, 0, 0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    assert est.estimate_ == pytest.approx(-40.0 / 3.0)


def test_m_zero_uses_only_W_t():
    """With m=0, each period contributes Y_t / Pr(W_t=w_t) with the right sign."""
    design = BernoulliDesign(l=1, p=0.5)
    W = np.array([1, 0, 1, 0])
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    # +Y[0]/0.5 - Y[1]/0.5 + Y[2]/0.5 - Y[3]/0.5 = 2 - 4 + 6 - 8 = -4. /T = -1.
    est = IPWEstimator(design=design, m=0).fit(W, Y)
    assert est.estimate_ == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Unbiasedness via Monte Carlo
# ---------------------------------------------------------------------------

def test_unbiasedness_under_simple_dgp():
    """Average estimate over many runs should recover tau."""
    T, m = 32, 0
    tau = 1.5
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    design = BernoulliDesign(l=1)
    est = IPWEstimator(design=design, m=m)

    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        est.fit(W, Y)
        estimates.append(est.estimate_)
    assert abs(np.mean(estimates) - tau) < 0.1


def test_unbiasedness_with_window_length_gt_one():
    """IPW remains unbiased when the design has longer windows (different propensities)."""
    T, m = 24, 0
    tau = 1.0
    dgp = SimpleDGP(mu=0.5, tau=tau, sigma=1.0)
    design = BernoulliDesign(l=3)
    est = IPWEstimator(design=design, m=m)

    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        est.fit(W, Y)
        estimates.append(est.estimate_)
    assert abs(np.mean(estimates) - tau) < 0.15


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_estimator_rejects_non_binary_assignment():
    est = IPWEstimator(design=BernoulliDesign(), m=0)
    with pytest.raises(ValueError):
        est.fit(np.array([0, 1, 2, 0]), np.zeros(4))


def test_estimator_rejects_T_le_m():
    # l=5 so the m=5 constructor accepts; T=3 then triggers T > m.
    est = IPWEstimator(design=BernoulliDesign(l=5), m=5)
    with pytest.raises(ValueError):
        est.fit(np.zeros(3, dtype=int), np.zeros(3))


def test_estimator_rejects_mismatched_lengths():
    est = IPWEstimator(design=BernoulliDesign(), m=0)
    with pytest.raises(ValueError):
        est.fit(np.zeros(5, dtype=int), np.zeros(4))


def test_estimator_requires_basedesign():
    with pytest.raises(TypeError):
        IPWEstimator(design="not a design", m=0)


def test_estimator_rejects_m_exceeding_window_length():
    """The burn-in m cannot exceed the design's window length — there's no
    structural mechanism to guarantee m+1 consecutive same-arm periods
    beyond a window. Locks in the constraint m ≤ design.l."""
    # BernoulliDesign l=2: m=2 is the boundary (allowed), m=3 not.
    IPWEstimator(design=BernoulliDesign(l=2), m=2)  # boundary OK
    with pytest.raises(ValueError, match="l"):
        IPWEstimator(design=BernoulliDesign(l=2), m=3)
    # AdaptiveBlockDesign has l=1; m=1 (paper's primary) OK, m=2 not.
    from switchback.design import AdaptiveBlockDesign
    IPWEstimator(design=AdaptiveBlockDesign(B=24, rho=0.5), m=1)  # boundary OK
    with pytest.raises(ValueError, match="l"):
        IPWEstimator(design=AdaptiveBlockDesign(B=24, rho=0.5), m=2)


def test_m_must_be_non_negative_integer():
    with pytest.raises(ValueError):
        IPWEstimator(design=BernoulliDesign(), m=-1)


# ===========================================================================
# HajekEstimator
# ===========================================================================

def test_hajek_T4_m1_hand_computed():
    """T=4, m=1, W=(1,1,0,0), Y=(10,20,30,40):
        S_1 = {1} (window [0,1] all 1)  -> mean = Y[1] = 20
        S_0 = {3} (window [2,3] all 0)  -> mean = Y[3] = 40
        τ̂_H = 20 - 40 = -20
    """
    est = HajekEstimator(design=BernoulliDesign(), m=1).fit(
        np.array([1, 1, 0, 0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )
    assert est.estimate_ == pytest.approx(-20.0)


def test_hajek_is_shift_invariant():
    """Adding a constant to Y must not change τ̂_H (key property of Hájek)."""
    rng = np.random.default_rng(0)
    W = rng.integers(0, 2, size=40)
    Y = rng.normal(0.0, 1.0, size=40)

    est_a = HajekEstimator(design=BernoulliDesign(), m=0).fit(W, Y)
    est_b = HajekEstimator(design=BernoulliDesign(), m=0).fit(W, Y + 100.0)
    assert est_a.estimate_ == pytest.approx(est_b.estimate_)


def test_hajek_unbiasedness_under_simple_dgp():
    T, m = 32, 0
    tau = 1.5
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    design = BernoulliDesign(l=1)
    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        # Need at least one treated and one controlled period.
        if W.sum() == 0 or W.sum() == T:
            continue
        estimates.append(HajekEstimator(design=design, m=m).fit(W, Y).estimate_)
    assert abs(np.mean(estimates) - tau) < 0.1


def test_hajek_raises_when_either_arm_empty():
    est = HajekEstimator(design=BernoulliDesign(), m=0)
    with pytest.raises(ValueError):
        est.fit(np.zeros(5, dtype=int), np.zeros(5))  # no treated
    with pytest.raises(ValueError):
        est.fit(np.ones(5, dtype=int), np.zeros(5))   # no control


def test_hajek_rejects_T_le_m():
    # l=5 so the m=5 constructor accepts; T=3 then triggers T > m.
    est = HajekEstimator(design=BernoulliDesign(l=5), m=5)
    with pytest.raises(ValueError):
        est.fit(np.zeros(3, dtype=int), np.zeros(3))


def test_hajek_rejects_non_binary_assignment():
    est = HajekEstimator(design=BernoulliDesign(), m=0)
    with pytest.raises(ValueError):
        est.fit(np.array([0, 1, 2, 0]), np.zeros(4))


def test_hajek_m_must_be_non_negative_integer():
    with pytest.raises(ValueError):
        HajekEstimator(design=BernoulliDesign(), m=-1)


# ===========================================================================
# StratifiedHajekEstimator
# ===========================================================================

def test_stratified_hajek_block4_m2_hand_computed():
    """T=12, l=4, m=2.
    Windows (0-indexed): [0..3], [4..7], [8..11].
    Periods 0,1 dropped (m=2). Strata (B = #windows the window crosses):
        t=2,3       -> window in window 0          (B=1)
        t=4,5       -> window crosses windows 0,1  (B=2)
        t=6,7       -> window in window 1          (B=1)
        t=8,9       -> window crosses windows 1,2  (B=2)
        t=10,11     -> window in window 2          (B=1)
    Choose W to deterministically place periods in known arms:
        window 0 = 1, window 1 = 0, window 2 = 1
    -> W = [1,1,1,1, 0,0,0,0, 1,1,1,1].
    Contributors:
        B=1, arm=1: t in {2,3,10,11}  -> Y values
        B=1, arm=0: t in {6,7}        -> Y values
        B=2, arm=*: windows mix (1,0) or (0,1) — drop.
    """
    W = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    Y = np.arange(1.0, 13.0)  # 1..12
    design = BernoulliDesign(l=4)
    est = StratifiedHajekEstimator(design, m=2).fit(W, Y)

    # B=1, arm=1: indices {2,3,10,11} -> Y = [3,4,11,12], mean = 7.5
    # B=1, arm=0: indices {6,7}       -> Y = [7,8],      mean = 7.5
    # B=1 tau = 0.0; n_B=1 = 6; only stratum.
    assert est.estimate_ == pytest.approx(0.0)
    # n_per_stratum_: B=1 has (4, 2); B=2 has (0, 0) (no contributors).
    assert est.n_per_stratum_[1] == (4, 2)


def test_stratified_hajek_consistency_simple_dgp():
    """Under SimpleDGP (no carryover) the stratified Hájek is consistent."""
    T, m = 200, 1
    tau = 1.0
    design = BernoulliDesign(l=2)
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        try:
            estimates.append(
                StratifiedHajekEstimator(design, m).fit(W, Y).estimate_
            )
        except ValueError:
            continue
    assert abs(np.mean(estimates) - tau) < 0.05


def test_stratified_hajek_validation():
    with pytest.raises(TypeError):
        StratifiedHajekEstimator(design="not a design", m=0)
    with pytest.raises(ValueError):
        StratifiedHajekEstimator(design=BernoulliDesign(), m=-1)


def test_stratified_hajek_rejects_no_contribution():
    design = BernoulliDesign(l=1)
    # Single arm only -> some stratum lacks both arms.
    est = StratifiedHajekEstimator(design, m=0)
    with pytest.raises(ValueError):
        est.fit(np.zeros(5, dtype=int), np.zeros(5))


def test_stratified_hajek_T_le_m_rejected():
    # l=5 so the m=5 constructor accepts; T=3 then triggers T > m.
    est = StratifiedHajekEstimator(design=BernoulliDesign(l=5), m=5)
    with pytest.raises(ValueError):
        est.fit(np.zeros(3, dtype=int), np.zeros(3))


def test_stratified_hajek_exposes_per_stratum_estimates():
    """Same hand-checked T=12, window=4, m=2 setup as above; ensure the
    estimate_per_stratum_ attribute is populated and matches the per-stratum
    Hájek τ_B (here only B=1 has contributors, with τ_{B=1} = 0)."""
    W = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    Y = np.arange(1.0, 13.0)
    design = BernoulliDesign(l=4)
    est = StratifiedHajekEstimator(design, m=2).fit(W, Y)
    assert est.estimate_per_stratum_ is not None
    assert 1 in est.estimate_per_stratum_
    assert est.estimate_per_stratum_[1] == pytest.approx(0.0)
    # B=2 stratum has no contributors here (all 2-window windows are mixed).
    assert 2 not in est.estimate_per_stratum_
