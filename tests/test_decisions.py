"""Tests for switchback.decisions (Neyman-type closed-form decide)."""

import numpy as np
import pytest

from switchback.decisions import (
    HACVariance,
    DecisionResult,
    confidence_interval,
    block_confidence_interval,
    block_variance,
    decide,
    normal_ci,
)
from switchback.design import (
    AdaptiveBlockDesign,
    BernoulliDesign,
    CompleteRandomization,
)
from switchback.dgp import SimpleDGP
from switchback.dgp.state_space import LatentStateDGP
from switchback.estimators import HajekEstimator, IPWEstimator


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

def test_point_estimate_matches_estimator():
    """The point estimate stored on the decide object equals the
    estimator's own estimate on the original (W, Y)."""
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(50)
    Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design=design, m=0)
    inf = HACVariance(design=design, estimator=est).fit(W, Y)
    direct = IPWEstimator(design=design, m=0).fit(W, Y).estimate_
    assert inf.estimate_ == pytest.approx(direct)


def test_variance_is_nonnegative_for_positive_outcomes():
    design = BernoulliDesign(l=3, seed=0)
    W = design.sample(40)
    # SimpleDGP with positive baseline -> all Y > 0 with high probability;
    # cross terms are then non-negative.
    Y = SimpleDGP(mu=10.0, tau=1.0, sigma=0.5, seed=0).generate(W)
    est = IPWEstimator(design=design, m=0)
    inf = HACVariance(design, est).fit(W, Y)
    assert inf.variance_ is not None and inf.variance_ >= 0


def test_confidence_interval_contains_estimate_and_widens_with_smaller_alpha():
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(60)
    Y = SimpleDGP(mu=5.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    inf = HACVariance(design, IPWEstimator(design, 0)).fit(W, Y)
    lo95, hi95 = inf.confidence_interval(0.05)
    lo99, hi99 = inf.confidence_interval(0.01)
    assert lo95 <= inf.estimate_ <= hi95
    assert hi99 - lo99 > hi95 - lo95


def test_convenience_function_matches_class():
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(40)
    Y = SimpleDGP(mu=2.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design=design, m=0)
    a = confidence_interval(design, est, W, Y, alpha=0.05)
    b = HACVariance(design, est).fit(W, Y).confidence_interval(0.05)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Closed-form correctness in the simplest case
# ---------------------------------------------------------------------------

def test_variance_matches_centered_neyman_window_length_1_m0_L0():
    """For BernoulliDesign(l=1, p=0.5), IPW with m=0, joint
    HAC at ``L=0`` reduces to the centered Welch variance on the per-window
    influence sequence ``X_b = 2 (2 W_b − 1) Y_b``:

        V̂ = (1 / T^2) * Σ_b (X_b − X̄)^2.
    """
    rng = np.random.default_rng(0)
    T = 30
    Y = rng.normal(0.0, 1.0, size=T)
    W = (rng.random(T) < 0.5).astype(int)

    X = 2.0 * (2 * W - 1) * Y
    expected = float(((X - X.mean()) ** 2).sum() / T ** 2)

    design = BernoulliDesign(l=1, p=0.5)
    est = IPWEstimator(design=design, m=0)
    inf = HACVariance(design, est, L=0).fit(W, Y)
    assert inf.variance_ == pytest.approx(expected)


def test_variance_matches_window_aggregated_form_for_window_length_2_m0_L0():
    """For BernoulliDesign(l=2, p=0.5), IPW with m=0, the
    per-window IPW influence sums the two periods in window b with a
    common sign and divides by the window length:

        X_b = (1/2) * (±2) * (Y_{2b} + Y_{2b+1}) = sign_b * (Y_{2b} + Y_{2b+1}),

    where sign_b = 2 coin_b − 1. Joint HAC at ``L=0`` gives

        V̂ = (1 / n_w^2) * Σ_b (X_b − X̄)^2,    n_w = T / 2.
    """
    rng = np.random.default_rng(1)
    T = 20
    Y = rng.normal(2.0, 1.0, size=T)
    design = BernoulliDesign(l=2, p=0.5, seed=0)
    W = design.sample(T)

    pair_sums = Y.reshape(-1, 2).sum(axis=1)
    signs = 2 * W[::2] - 1  # one coin per window
    X = signs * pair_sums
    n_w = T // 2
    expected = float(((X - X.mean()) ** 2).sum() / n_w ** 2)

    est = IPWEstimator(design=design, m=0)
    inf = HACVariance(design, est, L=0).fit(W, Y)
    assert inf.variance_ == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Conservativeness on a positive-effect, positive-baseline DGP
# ---------------------------------------------------------------------------

def test_hac_calibrated_against_empirical_variance_under_no_carryover():
    """SimpleDGP has no carryover (q = 0 ≤ m), so the joint-HAC plug-in
    with ``L=0`` is asymptotically unbiased for Var(τ̂). With 2,000 MC
    runs the average HAC variance should be close to the empirical
    variance of the estimates."""
    T, m = 24, 0
    tau = 1.0
    dgp = SimpleDGP(mu=2.0, tau=tau, sigma=1.0)
    design = BernoulliDesign(l=3)
    estimates = []
    variances = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        inf = HACVariance(
            design, IPWEstimator(design, m), L=0
        ).fit(W, Y)
        estimates.append(inf.estimate_)
        variances.append(inf.variance_)
    empirical_var = float(np.var(estimates, ddof=1))
    mean_hac = float(np.mean(variances))
    # Asymptotically unbiased — allow ±20% slack for MC noise / finite-T bias.
    assert 0.8 * empirical_var <= mean_hac <= 1.2 * empirical_var


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_design_must_be_bernoulli():
    class FakeDesign:
        pass
    with pytest.raises(TypeError):
        HACVariance(
            design=FakeDesign(),
            estimator=IPWEstimator(design=BernoulliDesign(), m=0),
        )


def test_estimator_must_be_ipw_or_hajek():
    with pytest.raises(TypeError):
        HACVariance(
            design=BernoulliDesign(), estimator="not an estimator"
        )


def test_confidence_interval_requires_fit():
    inf = HACVariance(
        design=BernoulliDesign(),
        estimator=IPWEstimator(design=BernoulliDesign(), m=0),
    )
    with pytest.raises(RuntimeError):
        inf.confidence_interval()


def test_T_must_exceed_m():
    # l=5 so the m=5 IPW constructor accepts; T=3 then triggers
    # the T > m check at fit time.
    inf = HACVariance(
        design=BernoulliDesign(l=5),
        estimator=IPWEstimator(design=BernoulliDesign(l=5), m=5),
    )
    with pytest.raises(ValueError):
        inf.fit(np.zeros(3, dtype=int), np.zeros(3))


def test_assignment_and_outcomes_must_match():
    inf = HACVariance(
        design=BernoulliDesign(),
        estimator=IPWEstimator(design=BernoulliDesign(), m=0),
    )
    with pytest.raises(ValueError):
        inf.fit(np.zeros(5, dtype=int), np.zeros(4))


# ---------------------------------------------------------------------------
# normal_ci: (estimate, variance) → CI utility
# ---------------------------------------------------------------------------

def test_normal_ci_returns_z_times_sqrt_variance_half_width():
    """normal_ci(estimate, variance, alpha) returns
    (estimate − z·√variance, estimate + z·√variance) with the right z."""
    import math
    from statistics import NormalDist
    estimate, variance, alpha = 2.0, 0.04, 0.05
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    expected_half = z * math.sqrt(variance)
    lo, hi = normal_ci(estimate, variance, alpha)
    assert (hi - lo) / 2.0 == pytest.approx(expected_half)
    assert (lo + hi) / 2.0 == pytest.approx(estimate)


def test_normal_ci_zero_variance_collapses_to_point():
    """At variance = 0 the CI collapses to the point estimate."""
    lo, hi = normal_ci(estimate=3.5, variance=0.0, alpha=0.05)
    assert lo == hi == 3.5


def test_normal_ci_rejects_negative_variance():
    with pytest.raises(ValueError, match="non-negative"):
        normal_ci(estimate=1.0, variance=-0.01, alpha=0.05)


def test_normal_ci_rejects_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        normal_ci(estimate=1.0, variance=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        normal_ci(estimate=1.0, variance=1.0, alpha=1.0)


def test_normal_ci_matches_HACVariance_confidence_interval():
    """HACVariance.confidence_interval(alpha) must equal normal_ci on its
    own (estimate_, variance_). Locks in the shared CI logic."""
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(40)
    Y = SimpleDGP(mu=1.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    inf = HACVariance(design, IPWEstimator(design, m=0)).fit(W, Y)
    ci_method = inf.confidence_interval(0.05)
    ci_util = normal_ci(inf.estimate_, inf.variance_, 0.05)
    assert ci_method == pytest.approx(ci_util)


# ===========================================================================
# Hájek dispatch — asymptotic Neyman variance
# ===========================================================================

def test_hajek_decide_point_estimate_matches_estimator():
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(50)
    Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
    est = HajekEstimator(design=design, m=0)
    inf = HACVariance(design, est).fit(W, Y)
    direct = HajekEstimator(design=design, m=0).fit(W, Y).estimate_
    assert inf.estimate_ == pytest.approx(direct)


def test_hajek_variance_at_L0_matches_centered_influence_formula():
    """For BernoulliDesign(l=1), HajekEstimator(m=0), the
    per-window influence is ``ξ_b = 2 (2 W_b − 1) (Y_b − μ̂_{W_b})`` (mean
    zero by construction). Joint HAC at ``L=0`` gives

        V̂ = (1 / T^2) * Σ_b ξ_b^2
           = (4 / T^2) * (n_1 · var(Y_1; ddof=0) + n_0 · var(Y_0; ddof=0)),

    which agrees with the textbook Welch variance asymptotically and
    differs only by ddof in finite samples.
    """
    design = BernoulliDesign(l=1, seed=0)
    W = design.sample(80)
    Y = SimpleDGP(mu=2.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    inf = HACVariance(
        design, HajekEstimator(design, m=0), L=0
    ).fit(W, Y)
    Y1, Y0 = Y[W == 1], Y[W == 0]
    T = Y.size
    expected = (
        4.0 / T ** 2
    ) * (Y1.size * Y1.var(ddof=0) + Y0.size * Y0.var(ddof=0))
    assert inf.variance_ == pytest.approx(expected)


def test_hajek_variance_is_shift_invariant():
    """Adding a constant to Y leaves V̂_H unchanged (within-arm variances
    don't see the level)."""
    design = BernoulliDesign(l=1, seed=0)
    W = design.sample(80)
    Y = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    inf_a = HACVariance(design, HajekEstimator(design, m=0)).fit(W, Y)
    inf_b = HACVariance(design, HajekEstimator(design, m=0)).fit(W, Y + 50.0)
    assert inf_a.variance_ == pytest.approx(inf_b.variance_)


def test_hajek_variance_smaller_than_ipw_when_levels_are_high():
    """Hájek's asymptotic variance is smaller than IPW's finite-sample
    variance by exactly the level term that the latter carries."""
    design = BernoulliDesign(l=1, seed=0)
    W = design.sample(200)
    # Large baseline -> IPW carries a big level contribution; Hájek doesn't.
    Y = SimpleDGP(mu=10.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    v_ipw = (
        HACVariance(design, IPWEstimator(design, 0)).fit(W, Y).variance_
    )
    v_hajek = (
        HACVariance(design, HajekEstimator(design, 0)).fit(W, Y).variance_
    )
    assert v_hajek < v_ipw


def test_hajek_decide_handles_one_obs_per_arm_gracefully():
    """The joint-HAC influence is well-defined with one observation per
    arm (each ξ_b becomes 0 since Y_b equals its arm mean), so the
    variance is reported as 0 rather than raising. The point estimate is
    still meaningful."""
    design = BernoulliDesign()
    inf = HACVariance(design, HajekEstimator(design, m=0)).fit(
        np.array([1, 0]), np.array([1.0, 2.0])
    )
    assert inf.estimate_ == pytest.approx(-1.0)
    assert inf.variance_ == pytest.approx(0.0, abs=1e-12)


def test_hajek_coverage_under_simple_dgp():
    """Empirical 95% CI coverage should be near 95% when n is moderate
    (Hájek asymptotic CI is exact under no carryover and l=1)."""
    T, m = 200, 0
    tau = 1.0
    design = BernoulliDesign(l=1)
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    rng = np.random.default_rng(0)
    covered = []
    for s in rng.integers(0, 10**9, size=1_500):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        if W.sum() < 2 or (T - W.sum()) < 2:
            continue
        lo, hi = (
            HACVariance(design, HajekEstimator(design, m)).fit(W, Y)
            .confidence_interval(0.05)
        )
        covered.append(lo <= tau <= hi)
    coverage = float(np.mean(covered))
    # Should be close to 95% with some MC slack.
    assert 0.92 <= coverage <= 0.98


# ===========================================================================
# Decisions under CompleteRandomization
# ===========================================================================

def test_decide_accepts_complete_randomization():
    """HACVariance must accept CompleteRandomization for both
    estimator types."""
    design = CompleteRandomization(l=2, seed=0)
    W = design.sample(20)
    Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
    # IPW
    inf_ipw = HACVariance(design, IPWEstimator(design, 0)).fit(W, Y)
    assert inf_ipw.variance_ is not None and inf_ipw.variance_ >= 0
    # Hájek
    inf_h = HACVariance(design, HajekEstimator(design, 0)).fit(W, Y)
    assert inf_h.variance_ is not None and inf_h.variance_ >= 0


def test_ipw_equals_hajek_under_complete_randomization_pointwise():
    """Under CompleteRandomization, IPWEstimator self-normalises and
    collapses to HajekEstimator — for ANY l and ANY m, not
    only window=1 m=0. Verify across a few configurations."""
    cases = [
        dict(l=1, m=0, T=40),
        dict(l=1, m=1, T=40),
        dict(l=2, m=0, T=40),
        dict(l=2, m=1, T=40),
    ]
    for cfg in cases:
        design = CompleteRandomization(l=cfg["l"], seed=0)
        W = design.sample(cfg["T"])
        Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
        try:
            e_ipw = IPWEstimator(design, cfg["m"]).fit(W, Y).estimate_
            e_h = HajekEstimator(design, cfg["m"]).fit(W, Y).estimate_
        except ValueError:
            continue
        assert e_ipw == pytest.approx(e_h), f"failed at {cfg}"


def test_ipw_equals_hajek_under_complete_randomization_variance():
    """Under CompleteRandomization, V̂(IPW) = V̂(Hájek) (both routed through
    the Welch path)."""
    design = CompleteRandomization(l=1, seed=0)
    T = 40
    W = design.sample(T)
    Y = SimpleDGP(mu=2.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    v_ipw = (
        HACVariance(design, IPWEstimator(design, 1)).fit(W, Y).variance_
    )
    v_h = (
        HACVariance(design, HajekEstimator(design, 1)).fit(W, Y).variance_
    )
    assert v_ipw == pytest.approx(v_h)


def test_ipw_under_bernoulli_still_uses_population_propensity():
    """Self-normalisation is opt-in for CR only — under Bernoulli, IPW
    uses the population propensity (so it differs from Hájek when there's
    a non-zero level)."""
    bern = BernoulliDesign(l=1, p=0.5, seed=0)
    T = 40
    W = bern.sample(T)
    Y = SimpleDGP(mu=2.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    e_ipw = IPWEstimator(bern, 0).fit(W, Y).estimate_
    e_h = HajekEstimator(bern, 0).fit(W, Y).estimate_
    # Generally not equal (depends on whether n_1 happens to equal T/2).
    if W.sum() != T // 2:
        assert e_ipw != pytest.approx(e_h)


def test_complete_randomization_unbiasedness_of_hajek_under_simple_dgp():
    T, m = 40, 0
    tau = 1.0
    design = CompleteRandomization(l=2)
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    estimates = []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=2_000):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        estimates.append(
            HACVariance(design, HajekEstimator(design, m)).fit(W, Y).estimate_
        )
    assert abs(np.mean(estimates) - tau) < 0.1


# ===========================================================================
# AdaptiveBlockDesign + IPW(m=1): paper variance (eq. 28)
# ===========================================================================

def test_block_variance_requires_adaptive_block_design():
    bern = BernoulliDesign(l=1)
    W = np.zeros(10, dtype=int)
    Y = np.zeros(10)
    with pytest.raises(TypeError):
        block_variance(bern, W, Y)


def test_block_variance_is_zero_for_constant_outcomes_no_boundary():
    """Without the boundary correction (paper verbatim), constant Y → every
    within-block sample variance is 0 and every cross-block centered product
    is 0, so eq. 28 returns 0 exactly."""
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    T = 672
    W = design.sample(T)
    Y = np.full(T, 3.14)
    assert block_variance(
        design, W, Y, boundary_correction=False
    ) == pytest.approx(0.0, abs=1e-12)


def test_block_variance_constant_outcomes_block0_correction():
    """With the boundary correction (default), constant Y is **not** zero —
    block 0's IPW influence is ``c · (N_{11,0} − N_{00,0}) / (K π_0)``,
    which is random under CR × CR even when Y is the constant ``c``. The
    block-0 count-variance correction exposes this:

        V̂ = c² · Var(N_{11,0} − N_{00,0}) / (B² (K π_0)²)

    matching the true Var(τ̂) for constant Y.
    """
    from switchback.decisions.block_variance import (
        _block0_count_variance_constants,
    )

    B, K = 24, 28
    T = B * K
    design = AdaptiveBlockDesign(B=B, rho=0.5, seed=0)
    W = design.sample(T)
    c = 3.14
    Y = np.full(T, c)
    var_N, cov_N = _block0_count_variance_constants(K)
    pi_0 = 0.25
    var_N_diff = 2.0 * var_N - 2.0 * cov_N
    expected = (c * c) * var_N_diff / ((B * K * pi_0) ** 2)
    got = block_variance(design, W, Y)
    assert got == pytest.approx(expected, rel=1e-10)


def test_block_variance_is_nonnegative_under_simple_dgp():
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    T = 672
    W = design.sample(T)
    Y = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    assert block_variance(design, W, Y) >= 0


def test_block_confidence_interval_contains_estimate_and_widens():
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    T = 672
    W = design.sample(T)
    Y = SimpleDGP(mu=1.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    tau_hat = IPWEstimator(design, m=1).fit(W, Y).estimate_
    lo95, hi95 = block_confidence_interval(design, tau_hat, W, Y, alpha=0.05)
    lo99, hi99 = block_confidence_interval(design, tau_hat, W, Y, alpha=0.01)
    assert lo95 <= tau_hat <= hi95
    assert hi99 - lo99 > hi95 - lo95


def test_block_variance_calibration_under_simple_dgp():
    """No-carryover DGP: mean(V̂_paper) tracks Var(τ̂) across MC trials.

    The paper's estimator is exact in the asymptotic Markov approximation;
    here we only check that its average is in the same ballpark as the
    empirical variance of τ̂_IPW(m=1) under :class:`SimpleDGP`.
    """
    T, B = 672, 24
    tau = 1.0
    design = AdaptiveBlockDesign(B=B, rho=0.5)
    dgp = SimpleDGP(mu=0.0, tau=tau, sigma=1.0)
    estimates, variances = [], []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=1_500):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        estimates.append(IPWEstimator(design, m=1).fit(W, Y).estimate_)
        variances.append(block_variance(design, W, Y))
    empirical_var = float(np.var(estimates, ddof=1))
    mean_v = float(np.mean(variances))
    assert 0.7 * empirical_var <= mean_v <= 1.4 * empirical_var


# ---------------------------------------------------------------------------
# block_variance — block-0 boundary correction
# ---------------------------------------------------------------------------

def test_block_variance_boundary_weight_is_noop_at_rho_half():
    """At ρ=0.5 the block-0 propensity (0.25) equals the adaptive rate ρ/2,
    so the boundary **weight** correction (1/π_b vs 2/ρ) is a no-op. The
    block-0 **count-variance** correction still applies (it depends only on
    K, not ρ), so V̂ with the full boundary correction is shifted up by a
    fixed amount equal to that correction.
    """
    from switchback.decisions.block_variance import (
        _block0_count_variance_constants,
        _per_block_means,
        _within_block_variance_per_block,
    )

    B, K = 24, 28
    T = B * K
    design = AdaptiveBlockDesign(B=B, rho=0.5, seed=0)
    W = design.sample(T)
    Y = SimpleDGP(mu=1.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    v_corr = block_variance(design, W, Y, boundary_correction=True)
    v_legacy = block_variance(design, W, Y, boundary_correction=False)

    # Reproduce the count-variance correction analytically.
    var_N, cov_N = _block0_count_variance_constants(K)
    R_11_pb = _within_block_variance_per_block(Y, W, T, B, u=1)
    R_00_pb = _within_block_variance_per_block(Y, W, T, B, u=0)
    mean_11, count_11 = _per_block_means(Y, W, T, B, K, u=1, v=1)
    mean_00, count_00 = _per_block_means(Y, W, T, B, K, u=0, v=0)
    n_11, n_00 = int(count_11[0]), int(count_00[0])
    Yb_11, Yb_00 = float(mean_11[0]), float(mean_00[0])
    mu_1_sq = Yb_11 * Yb_11 - R_11_pb[0] / n_11
    mu_0_sq = Yb_00 * Yb_00 - R_00_pb[0] / n_00
    count_var = (
        mu_1_sq * var_N + mu_0_sq * var_N - 2.0 * Yb_11 * Yb_00 * cov_N
    )
    pi_0 = 0.25
    expected_delta = count_var / ((B * K * pi_0) ** 2)
    assert v_corr - v_legacy == pytest.approx(expected_delta, rel=1e-10)


def test_block_variance_diagonal_weights_match_inverse_propensity():
    """Block 0 weight is 1/0.25 = 4; interior blocks are 2/ρ."""
    from switchback.decisions.block_variance import _diagonal_weights

    design = AdaptiveBlockDesign(B=24, rho=0.75, seed=0)
    T = 672
    rho = design.effective_rho(T)
    w = _diagonal_weights(design, T, design.B, rho)
    assert w[0] == pytest.approx(1.0 / 0.25)              # block 0: 4.0
    assert np.allclose(w[1:], 2.0 / rho)                  # interior: 2/ρ
    assert w[0] != pytest.approx(2.0 / rho)               # genuinely differs


def test_pair_gamma_matrices_match_paper_recursion():
    """``γ⁺_{b,b'}`` and ``γ⁻_{b,b'}`` are computed as **separate** quantities
    per the paper's eq. 13 (ratios of conditional / marginal probabilities) —
    not via the shortcut ``(2ρ−1)^{δ−1}`` which is only the *half-spread*
    around 1 (``(γ⁺ − γ⁻)/2``).

    This locks in the equivalence between our closed forms

        γ⁺ = 1 + (2ρ − 1)^{δ−1},   γ⁻ = 1 − (2ρ − 1)^{δ−1}

    and the paper's forward recursion (eqs. 67–70) for the q-functions, for
    same-day pairs ``(b, b' = b + δ < B)``. If anyone ever "simplifies" the
    formula to a single (γ⁺ − γ⁻) factor, this test catches it.
    """
    from switchback.decisions.block_variance import _pair_gamma_matrices

    def paper_recursion(rho: float, max_delta: int) -> dict:
        # eq. 69 initial values at δ=1
        q11_11 = rho * rho / 2.0
        q11_00 = 0.0
        q11_01 = 0.0
        q11_10 = rho * (1.0 - rho) / 2.0
        out = {1: (4.0 * q11_11 / (rho * rho), 4.0 * q11_00 / (rho * rho))}
        for delta in range(2, max_delta + 1):
            # eq. 68 forward update
            new_11 = (q11_11 + q11_01) * rho
            new_00 = (q11_00 + q11_10) * rho
            new_01 = (q11_00 + q11_10) * (1.0 - rho)
            new_10 = (q11_11 + q11_01) * (1.0 - rho)
            q11_11, q11_00, q11_01, q11_10 = new_11, new_00, new_01, new_10
            # eq. 70
            out[delta] = (4.0 * q11_11 / (rho * rho), 4.0 * q11_00 / (rho * rho))
        return out

    for B in (6, 12, 24):
        for rho in (0.5, 0.6, 0.75, 0.9):
            gp_mat, gm_mat = _pair_gamma_matrices(B, rho)
            rec = paper_recursion(rho, B - 1)
            for b in range(B - 1):
                for delta in range(1, B - b):
                    bp = b + delta
                    gp_rec, gm_rec = rec[delta]
                    assert gp_mat[b, bp] == pytest.approx(gp_rec, rel=1e-12), (
                        f"γ⁺ mismatch at (b={b}, b'={bp}, δ={delta}, ρ={rho})"
                    )
                    assert gm_mat[b, bp] == pytest.approx(gm_rec, rel=1e-12), (
                        f"γ⁻ mismatch at (b={b}, b'={bp}, δ={delta}, ρ={rho})"
                    )


def test_cross_pattern_R_is_zero_at_delta_0_and_1_nonzero_at_delta_geq_2():
    """``R^{1,0}_{b,b'}`` and ``R^{0,1}_{b,b'}`` are estimable from data only
    when ``δ_{b,b'} ≥ 2``. At δ=0 (same t) and δ=1 (adjacent, sharing
    ``W_t``) the events ``(W_{t-1:t}=(1,1))`` and ``(W_{t'-1:t'}=(0,0))``
    are mutually exclusive: ``W_t`` can't be both 1 and 0. Under design
    randomization no realization ever supplies a contributing pair, so
    the sample statistic is **undefined as data** — :math:`R^{1,0}` has
    no observations to estimate it from.

    The implementation handles this by an *empty-sum convention*:
    ``_cross_block_covariance_per_block`` initialises the accumulator to
    zero and never adds a term, so the returned value is numerical 0.
    This is a coding convenience, not a structural claim that "R^{1,0}
    equals 0" — and it's safe because eq. 28 multiplies these
    R-statistics by ``γ⁻``, which is also exactly 0 at δ ∈ {0, 1} (eq.
    13). So the ``γ⁻·R^{1,0}`` term is ``0·(undefined) = 0`` by
    convention, with the convention being "γ⁻=0 absorbs the absence of
    the R-statistic".

    At δ ≥ 2 the two pair-indicators don't share any W, so R^{1,0} and
    R^{0,1} become well-defined sample statistics with real data behind
    them, and γ⁻ is generally non-zero. That's where the γ⁻ piece does
    real work in the variance formula.

    Locks in (a) the empty-sum convention's numerical output at δ ∈
    {0, 1} and (b) that R^{1,0}, R^{0,1} are observable and generically
    non-zero at δ ≥ 2.
    """
    from switchback.decisions.block_variance import (
        _cross_block_covariance_per_block,
        _within_block_variance_per_block,
    )

    B, K = 24, 28
    T = B * K
    design = AdaptiveBlockDesign(B=B, rho=0.75, seed=7)
    W = design.sample(T)
    Y = LatentStateDGP(
        mu=0.0, beta_0=1.0, alpha_0=0.5, gamma=0.5,
        sigma_y=1.0, sigma_h=1.0, seed=11,
    ).generate(W)

    # δ=0 (diagonal): only R^{u,u} are defined; R^{1,0}, R^{0,1} aren't
    # even computed (they'd require same-t cross-pattern conditioning,
    # which is impossible). Code path doesn't expose them, but the
    # contract is implicit in _within_block_variance_per_block taking
    # only ``u``, not (u, v).
    R_11_diag = _within_block_variance_per_block(Y, W, T, B, u=1)
    R_00_diag = _within_block_variance_per_block(Y, W, T, B, u=0)
    assert np.any(R_11_diag != 0.0)  # non-trivial Y, so R^{1,1} ≠ 0 somewhere
    assert np.any(R_00_diag != 0.0)

    # δ=1: the sample statistic R^{1,0}_{b,b+1} is **undefined as data**
    # (no realization provides a contributing pair). The implementation's
    # empty-sum convention returns numerical 0 — that's what we lock in.
    R_10_d1 = _cross_block_covariance_per_block(Y, W, T, B, K, 1, u=1, v=0)
    R_01_d1 = _cross_block_covariance_per_block(Y, W, T, B, K, 1, u=0, v=1)
    assert np.all(R_10_d1 == 0.0), (
        "R^{1,0}_{b,b+1} must return numerical 0 by empty-sum convention "
        "— δ=1 forces W_t = 1 AND W_t = 0 simultaneously, so no "
        "contributing pair exists and the statistic is undefined-as-data."
    )
    assert np.all(R_01_d1 == 0.0)

    # δ ≥ 2: R^{1,0} and R^{0,1} are observable (no shared W variable
    # between the two pair indicators) and generically non-zero.
    R_10_d2 = _cross_block_covariance_per_block(Y, W, T, B, K, 2, u=1, v=0)
    R_01_d2 = _cross_block_covariance_per_block(Y, W, T, B, K, 2, u=0, v=1)
    assert np.any(R_10_d2 != 0.0), (
        "R^{1,0}_{b,b+2} should be observable at δ=2 (no shared W between "
        "the two pair indicators) and non-zero for a generic outcome path."
    )
    assert np.any(R_01_d2 != 0.0)


def test_block_variance_applies_gamma_plus_and_minus_separately():
    """``block_variance`` must implement eq. 28 exactly, with ``γ⁺_{b,b'}``
    weighting ``(R^{1,1}+R^{0,0})`` and ``γ⁻_{b,b'}`` weighting
    ``(R^{1,0}+R^{0,1})`` as **separate** terms — not collapsed into a
    single ``(γ⁺ − γ⁻)`` factor.

    Verifies by hand-computing V̂ via the public helpers (with γ⁺, γ⁻ kept
    distinct at every step) and matching to ``block_variance`` numerically
    on a realistic ``(W, Y)`` realization at ρ=0.75 where γ⁺ ≠ γ⁻ at every
    interior δ ≥ 1.
    """
    from switchback.decisions.block_variance import (
        _within_block_variance_per_block,
        _cross_block_covariance_per_block,
        _pair_gamma_matrices,
        _diagonal_weights,
        _block0_count_variance_constants,
        _per_block_means,
    )

    B, K = 24, 28
    T = B * K
    design = AdaptiveBlockDesign(B=B, rho=0.75, seed=42)
    W = design.sample(T)
    Y = LatentStateDGP(
        mu=0.0, beta_0=1.0, alpha_0=0.5, gamma=0.5,
        sigma_y=1.0, sigma_h=1.0, seed=43,
    ).generate(W)
    rho = design.effective_rho(T)
    max_delta = B - 1  # full sum so every δ ∈ {1,…,B-1} is exercised

    # --- Manual eq. 28: γ⁺ and γ⁻ applied separately at every step ---
    R_11_diag = _within_block_variance_per_block(Y, W, T, B, u=1)
    R_00_diag = _within_block_variance_per_block(Y, W, T, B, u=0)
    w_diag = _diagonal_weights(design, T, B, rho)
    total = float(np.sum(w_diag * (R_11_diag + R_00_diag)))

    # Block-0 count-variance correction (boundary fix)
    pi_0 = 1.0 / w_diag[0]
    var_N, cov_N = _block0_count_variance_constants(K)
    mean_11, count_11 = _per_block_means(Y, W, T, B, K, u=1, v=1)
    mean_00, count_00 = _per_block_means(Y, W, T, B, K, u=0, v=0)
    n_11 = int(count_11[0]); n_00 = int(count_00[0])
    Yb_11 = float(mean_11[0]); Yb_00 = float(mean_00[0])
    mu_1_sq = Yb_11 * Yb_11 - R_11_diag[0] / n_11
    mu_0_sq = Yb_00 * Yb_00 - R_00_diag[0] / n_00
    count_var = mu_1_sq * var_N + mu_0_sq * var_N - 2.0 * Yb_11 * Yb_00 * cov_N
    total += count_var / (K * pi_0 * pi_0)

    # Off-diagonal: γ⁺ * (R¹¹+R⁰⁰) and γ⁻ * (R¹⁰+R⁰¹) as SEPARATE operations,
    # iterated forward δ up to B/2 with factor-2 mirror to account for the
    # backward-ordered partner (which by Cov-symmetry contributes the same
    # R as the forward pair).
    gp_mat, gm_mat = _pair_gamma_matrices(B, rho)
    half_B = B // 2
    distinct_pairs = 0
    for delta in range(1, max_delta + 1):
        if delta > half_B:
            break  # backward-cyclic direction; already covered by mirror
        R11 = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=1, v=1)
        R00 = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=0, v=0)
        R10 = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=1, v=0)
        R01 = _cross_block_covariance_per_block(Y, W, T, B, K, delta, u=0, v=1)
        mirror = 1.0 if (delta == half_B and B % 2 == 0) else 2.0
        for b in range(B):
            bp = (b + delta) % B
            # γ⁺ and γ⁻ used SEPARATELY at this pair
            total += mirror * gp_mat[b, bp] * (R11[b] + R00[b])
            total -= mirror * gm_mat[b, bp] * (R10[b] + R01[b])
            if not np.isclose(gp_mat[b, bp], gm_mat[b, bp]):
                distinct_pairs += 1

    # Sanity: at ρ=0.75 γ⁺ ≠ γ⁻ at many pairs, so the separateness matters.
    assert distinct_pairs > 0, (
        "Test setup is vacuous: γ⁺ = γ⁻ at every pair; pick a ρ where they differ."
    )

    manual_V = total / (B * B * K)
    got = block_variance(design, W, Y, max_delta=max_delta)
    assert got == pytest.approx(manual_V, rel=1e-12)


def test_block_variance_boundary_correction_exact_difference():
    """corrected − legacy decomposes exactly into the two pieces the
    boundary correction adds at block 0:

    (a) **diagonal re-weight**: ``(1/π₀ − 2/ρ) · (R¹¹_{0,0} + R⁰⁰_{0,0})
        / (B² K)``;
    (b) **count-variance correction**: block-0 ``Var(μ_1 N_{11,0} −
        μ_0 N_{00,0}) / ((B K π_0)²)`` using bias-corrected μ̂² and the
        closed-form CR×CR design constants.
    """
    from switchback.decisions.block_variance import (
        _block0_count_variance_constants,
        _per_block_means,
        _within_block_variance_per_block,
        _gamma_plus,
    )

    design = AdaptiveBlockDesign(B=24, rho=0.75, seed=1)
    T = 672
    W = design.sample(T)
    Y = SimpleDGP(mu=2.0, tau=1.0, sigma=1.0, seed=1).generate(W)
    B, K = design.B, T // design.B
    rho = design.effective_rho(T)

    v_corr = block_variance(design, W, Y, boundary_correction=True)
    v_legacy = block_variance(design, W, Y, boundary_correction=False)

    r11_0 = _within_block_variance_per_block(Y, W, T, B, 1)[0]
    r00_0 = _within_block_variance_per_block(Y, W, T, B, 0)[0]
    reweight_delta = (
        (1.0 / 0.25 - _gamma_plus(rho, 0)) * (r11_0 + r00_0) / (B * B * K)
    )

    var_N, cov_N = _block0_count_variance_constants(K)
    mean_11, count_11 = _per_block_means(Y, W, T, B, K, u=1, v=1)
    mean_00, count_00 = _per_block_means(Y, W, T, B, K, u=0, v=0)
    n_11, n_00 = int(count_11[0]), int(count_00[0])
    Yb_11, Yb_00 = float(mean_11[0]), float(mean_00[0])
    mu_1_sq = Yb_11 * Yb_11 - r11_0 / n_11
    mu_0_sq = Yb_00 * Yb_00 - r00_0 / n_00
    count_var = mu_1_sq * var_N + mu_0_sq * var_N - 2.0 * Yb_11 * Yb_00 * cov_N
    count_var_delta = count_var / ((B * K * 0.25) ** 2)

    assert (v_corr - v_legacy) == pytest.approx(reweight_delta + count_var_delta)


def test_block_variance_boundary_correction_constant_outcomes_block0_only():
    """For constant Y = c the centered-R within-pattern variances and the
    off-diagonal centered products are all 0, so the boundary-corrected
    estimator collapses to the block-0 count-variance term:

        V̂ = c² · Var(N_{11,0} − N_{00,0}) / (B² (K π_0)²).

    This is the true Var(τ̂) for constant Y — block 0's CR×CR count
    fluctuations are the only source of τ̂'s variance in that limit.
    """
    from switchback.decisions.block_variance import (
        _block0_count_variance_constants,
    )

    B, K = 24, 28
    T = B * K
    design = AdaptiveBlockDesign(B=B, rho=0.75, seed=0)
    W = design.sample(T)
    c = 2.71
    Y = np.full(T, c)
    var_N, cov_N = _block0_count_variance_constants(K)
    expected = (c * c) * (2.0 * var_N - 2.0 * cov_N) / ((B * K * 0.25) ** 2)
    assert block_variance(design, W, Y, boundary_correction=True) == pytest.approx(
        expected, rel=1e-10
    )


def test_block_variance_boundary_corrected_calibrates_at_rho_075():
    """At ρ=0.75 the boundary-corrected estimator is well-calibrated
    against the true Var(τ̂_IPW(m=1)) under a no-carryover DGP.

    Note: we do *not* assert the correction beats the legacy formula in
    every config — the boundary error and a separate finite-K
    Markov-γ-approximation residual are of comparable magnitude, so the
    net per-config calibration can fall either side of 1.0. We only check
    the corrected value sits in a sensible band and that the correction
    is actually active (differs from legacy at ρ≠0.5)."""
    T, B = 672, 24
    design = AdaptiveBlockDesign(B=B, rho=0.75)
    dgp = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0)
    estimates, v_corr, v_legacy = [], [], []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=1_500):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        estimates.append(IPWEstimator(design, m=1).fit(W, Y).estimate_)
        v_corr.append(block_variance(design, W, Y, boundary_correction=True))
        v_legacy.append(block_variance(design, W, Y, boundary_correction=False))
    emp = float(np.var(estimates, ddof=1))
    mc = float(np.mean(v_corr))
    ml = float(np.mean(v_legacy))
    assert 0.85 * emp <= mc <= 1.15 * emp
    assert mc != pytest.approx(ml)  # correction is active at ρ=0.75


# ===========================================================================
# AdaptiveBlockDesign: HACVariance must reject; use block_variance instead
# ===========================================================================

def test_hac_variance_rejects_adaptive_block_design():
    """HACVariance is for window-structured designs only (BernoulliDesign,
    CompleteRandomization). AdaptiveBlockDesign has its own design-derived
    variance estimator (block_variance, eq. 28 + boundary fixes) that
    uses structural γ-weights instead of an empirical-autocovariance
    kernel — at B=24 there are too few clusters for HAC to be reliable.
    """
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    est = IPWEstimator(design, m=1)
    with pytest.raises(TypeError, match="AdaptiveBlockDesign"):
        HACVariance(design, est)


def test_block_variance_calibrated_under_seasonal_baseline():
    """The adaptive block design exists for seasonal data. With a
    seasonal baseline whose global mean is 0, block 0 still sits at its
    own seasonal level μ₀, producing a leverage term that inflates
    Var(τ̂_IPW). The default block_variance (with the block-0
    count-variance correction) captures this via the bias-corrected
    sample-mean-squared and stays calibrated. Without the correction
    (paper-verbatim boundary_correction=False), it under-estimates.
    """
    T, B = 672, 24
    design = AdaptiveBlockDesign(B=B, rho=0.5)
    # period-B seasonal baseline, global mean 0, block 0 at the peak (+5)
    mu_fn = lambda t: 5.0 * np.cos(2.0 * np.pi * (t % B) / B)  # noqa: E731
    dgp = SimpleDGP(mu=mu_fn, tau=1.0, sigma=1.0)
    estimates, v_paper, v_paper_legacy = [], [], []
    rng = np.random.default_rng(0)
    for s in rng.integers(0, 10**9, size=1_500):
        design.reset(seed=int(s))
        dgp.reset(seed=int(s) + 1)
        W = design.sample(T)
        Y = dgp.generate(W)
        est = IPWEstimator(design, m=1).fit(W, Y)
        estimates.append(est.estimate_)
        v_paper.append(block_variance(design, W, Y))
        v_paper_legacy.append(
            block_variance(design, W, Y, boundary_correction=False)
        )
    emp = float(np.var(estimates, ddof=1))
    mc_paper = float(np.mean(v_paper))
    mc_paper_legacy = float(np.mean(v_paper_legacy))
    # corrected block_variance is calibrated under block-0 seasonality
    assert 0.85 * emp <= mc_paper <= 1.15 * emp
    # uncorrected (paper-verbatim) block_variance still under-estimates,
    # proving the count-variance correction is what closes the gap
    assert mc_paper_legacy < 0.95 * emp
    assert mc_paper_legacy < mc_paper


# ===========================================================================
# decide: one-call front door (estimate + variance + CI, design-dispatched)
# ===========================================================================

def test_decide_returns_DecisionResult_with_all_fields():
    """decide returns an DecisionResult dataclass with .estimate,
    .variance, .ci, .alpha populated."""
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    W = design.sample(672)
    Y = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    result = decide(design, IPWEstimator(design, m=1), W, Y, alpha=0.05)
    assert isinstance(result, DecisionResult)
    assert isinstance(result.estimate, float)
    assert isinstance(result.variance, float) and result.variance >= 0
    assert isinstance(result.ci, tuple) and len(result.ci) == 2
    lo, hi = result.ci
    assert lo <= result.estimate <= hi
    assert result.alpha == 0.05


def test_decide_dispatches_block_variance_for_adaptive_block_design():
    """For AdaptiveBlockDesign, decide must use block_variance under
    the hood (not HAC, which would reject this design at construction)."""
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    W = design.sample(672)
    Y = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design, m=1)
    # decide's variance should match block_variance exactly
    result = decide(design, est, W, Y, alpha=0.05)
    direct_v = block_variance(design, W, Y)
    assert result.variance == pytest.approx(direct_v)
    # And the point estimate matches a direct estimator.fit
    direct_est = IPWEstimator(design, m=1).fit(W, Y).estimate_
    assert result.estimate == pytest.approx(direct_est)


def test_decide_dispatches_HACVariance_for_bernoulli_design():
    """For BernoulliDesign, decide must use HACVariance under the
    hood."""
    design = BernoulliDesign(l=4, seed=0)
    W = design.sample(80)
    Y = SimpleDGP(mu=1.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design, m=3)
    result = decide(design, est, W, Y, alpha=0.05)
    direct = HACVariance(design, est).fit(W, Y)
    assert result.estimate == pytest.approx(direct.estimate_)
    assert result.variance == pytest.approx(direct.variance_)
    assert result.ci == pytest.approx(direct.confidence_interval(0.05))


def test_decide_ci_matches_normal_ci_on_estimate_and_variance():
    """decide's .ci field must be exactly normal_ci(estimate, variance,
    alpha) — locks in the consistent CI math across the package."""
    design = AdaptiveBlockDesign(B=24, rho=0.5, seed=0)
    W = design.sample(672)
    Y = SimpleDGP(mu=0.0, tau=1.0, sigma=1.0, seed=0).generate(W)
    result = decide(design, IPWEstimator(design, m=1), W, Y, alpha=0.10)
    expected_ci = normal_ci(result.estimate, result.variance, 0.10)
    assert result.ci == pytest.approx(expected_ci)


def test_decide_does_not_mutate_user_estimator():
    """The user's estimator object must be unchanged after decide is
    called — decide deep-copies it internally."""
    design = BernoulliDesign(l=2, seed=0)
    W = design.sample(40)
    Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design, m=0)
    assert est.estimate_ is None  # pre-fit state
    _ = decide(design, est, W, Y, alpha=0.05)
    assert est.estimate_ is None  # still pre-fit


def test_decide_rejects_invalid_alpha():
    design = BernoulliDesign(seed=0)
    W = design.sample(40)
    Y = SimpleDGP(tau=1.0, sigma=1.0, seed=0).generate(W)
    est = IPWEstimator(design, m=0)
    with pytest.raises(ValueError, match="alpha"):
        decide(design, est, W, Y, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        decide(design, est, W, Y, alpha=1.0)
