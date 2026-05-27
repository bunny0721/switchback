"""Tests for switchback.dgp."""

import numpy as np
import pytest

from switchback.dgp import BaseDGP, CarryoverDGP, LatentStateDGP, SimpleDGP


# ---------------------------------------------------------------------------
# Shape / contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 5, 200])
def test_returns_outcomes_of_length_T(T):
    dgp = SimpleDGP(seed=0)
    Y = dgp.generate(np.zeros(T, dtype=int))
    assert Y.shape == (T,)
    assert np.isfinite(Y).all()


def test_callable_alias_matches_generate():
    W = np.array([0, 1, 1, 0, 1])
    dgp = SimpleDGP(seed=0)
    Y_call = dgp(W)
    dgp.reset()
    Y_gen = dgp.generate(W)
    np.testing.assert_array_equal(Y_call, Y_gen)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_outcomes():
    W = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y1 = SimpleDGP(seed=42).generate(W)
    Y2 = SimpleDGP(seed=42).generate(W)
    np.testing.assert_array_equal(Y1, Y2)


def test_reset_replays_same_outcomes():
    dgp = SimpleDGP(seed=7)
    W = np.array([0, 1, 1, 0, 1])
    Y1 = dgp.generate(W)
    dgp.reset()
    Y2 = dgp.generate(W)
    np.testing.assert_array_equal(Y1, Y2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_2d_assignment_rejected():
    with pytest.raises(ValueError):
        SimpleDGP(seed=0).generate(np.zeros((4, 2)))


def test_empty_assignment_rejected():
    with pytest.raises(ValueError):
        SimpleDGP(seed=0).generate(np.array([], dtype=int))


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        SimpleDGP(sigma=-0.1)


def test_mu_array_wrong_length_rejected():
    dgp = SimpleDGP(mu=np.zeros(10), seed=0)
    with pytest.raises(ValueError):
        dgp.generate(np.zeros(5, dtype=int))


def test_mu_callable_wrong_shape_rejected():
    dgp = SimpleDGP(mu=lambda t: np.zeros(len(t) + 1), seed=0)
    with pytest.raises(ValueError):
        dgp.generate(np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# Determinism with sigma=0 — exact algebra check
# ---------------------------------------------------------------------------

def test_deterministic_when_sigma_zero():
    """With sigma=0 the DGP returns mu_t + tau*W_t exactly."""
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    W = np.array([0, 1, 0, 1, 1])
    tau = 0.7
    Y = SimpleDGP(mu=mu, tau=tau, sigma=0.0, seed=0).generate(W)
    np.testing.assert_allclose(Y, mu + tau * W)


def test_scalar_mu_broadcasts():
    W = np.array([0, 1, 0, 1])
    Y = SimpleDGP(mu=2.5, tau=1.0, sigma=0.0, seed=0).generate(W)
    np.testing.assert_allclose(Y, 2.5 + W)


def test_callable_mu_receives_time_index():
    W = np.array([1, 1, 1, 1])
    Y = SimpleDGP(mu=lambda t: t.astype(float), tau=0.0, sigma=0.0).generate(W)
    np.testing.assert_allclose(Y, np.arange(4, dtype=float))


# ---------------------------------------------------------------------------
# Statistical sanity
# ---------------------------------------------------------------------------

def test_treatment_effect_recovered_in_large_sample():
    """Difference of means under W=1 vs W=0 recovers tau."""
    T = 50_000
    tau = 1.5
    Y1 = SimpleDGP(mu=0.3, tau=tau, sigma=1.0, seed=1).generate(np.ones(T, dtype=int))
    Y0 = SimpleDGP(mu=0.3, tau=tau, sigma=1.0, seed=2).generate(np.zeros(T, dtype=int))
    assert abs((Y1.mean() - Y0.mean()) - tau) < 0.05


# ---------------------------------------------------------------------------
# Subclass discipline
# ---------------------------------------------------------------------------

def test_basedgp_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseDGP(seed=0)  # type: ignore[abstract]


# ===========================================================================
# CarryoverDGP
# ===========================================================================

def test_carryover_returns_outcomes_of_length_T():
    dgp = CarryoverDGP(betas=[1.0, 0.5], seed=0)
    Y = dgp.generate(np.array([0, 1, 1, 0, 1]))
    assert Y.shape == (5,)
    assert np.isfinite(Y).all()


def test_carryover_q_property():
    assert CarryoverDGP(betas=[1.0]).q == 0
    assert CarryoverDGP(betas=[1.0, 0.5]).q == 1
    assert CarryoverDGP(betas=[1.0, 0.5, 0.25, 0.1]).q == 3


def test_carryover_betas_must_be_nonempty():
    with pytest.raises(ValueError):
        CarryoverDGP(betas=[])


def test_carryover_betas_must_be_1d():
    with pytest.raises(ValueError):
        CarryoverDGP(betas=[[1.0, 0.5], [0.25, 0.1]])


def test_carryover_negative_sigma_rejected():
    with pytest.raises(ValueError):
        CarryoverDGP(betas=[1.0], sigma=-0.1)


def test_carryover_deterministic_when_sigma_zero_lag1():
    """sigma=0 -> Y_t = mu_t + beta_0 W_t + beta_1 W_{t-1} exactly."""
    beta_0, beta_1 = 1.0, 0.5
    W = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    mu = np.arange(W.size, dtype=float)  # time-varying baseline 0,1,2,...
    Y = CarryoverDGP(
        betas=[beta_0, beta_1], mu=mu, sigma=0.0, seed=0
    ).generate(W)
    W_lag1 = np.concatenate(([0], W[:-1]))
    np.testing.assert_allclose(Y, mu + beta_0 * W + beta_1 * W_lag1)


def test_carryover_higher_order_matches_explicit_sum():
    """Y_t = sum_{k=0..3} beta_k W_{t-k} for q=3, sigma=0."""
    betas = np.array([1.0, 0.5, 0.25, 0.125])
    W = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    Y = CarryoverDGP(betas=betas, mu=0.0, sigma=0.0, seed=0).generate(W)
    expected = np.zeros(W.size)
    for t in range(W.size):
        for k, b in enumerate(betas):
            if t - k >= 0:
                expected[t] += b * W[t - k]
    np.testing.assert_allclose(Y, expected)


def test_carryover_with_single_beta_matches_simpledgp():
    """CarryoverDGP(betas=[beta_0]) is the no-carryover model with tau=beta_0."""
    beta_0 = 0.7
    W = np.array([1, 0, 1, 1, 0, 0, 1])
    Y_carry = CarryoverDGP(
        betas=[beta_0], mu=0.5, sigma=1.0, seed=42
    ).generate(W)
    Y_simple = SimpleDGP(tau=beta_0, mu=0.5, sigma=1.0, seed=42).generate(W)
    np.testing.assert_allclose(Y_carry, Y_simple)


def test_carryover_long_run_mean_under_sustained_treatment():
    """Under W_t = 1 for all t, large-sample mean is mu + sum(betas)."""
    betas = [1.0, 0.5, 0.25]
    mu = 0.3
    T = 50_000
    Y = CarryoverDGP(
        betas=betas, mu=mu, sigma=1.0, seed=0
    ).generate(np.ones(T, dtype=int))
    # Skip first q periods (transient), then expect mean = mu + sum(betas).
    assert abs(Y[len(betas):].mean() - (mu + sum(betas))) < 0.05


def test_carryover_lag_m_estimand_matches_partial_sum():
    """The lag-m causal effect Y_t(1_{m+1}) - Y_t(0_{m+1}) under no-noise
    DGP equals sum_{k=0..min(m, q)} beta_k."""
    betas = [1.0, 0.5, 0.25]
    q = len(betas) - 1
    T = 10
    dgp = CarryoverDGP(betas=betas, mu=0.0, sigma=0.0, seed=0)
    Y_all1 = dgp.generate(np.ones(T, dtype=int))
    Y_all0 = dgp.generate(np.zeros(T, dtype=int))
    # After the transient (t >= q), the per-period effect is sum(betas).
    diff = Y_all1[q:] - Y_all0[q:]
    np.testing.assert_allclose(diff, np.full(T - q, sum(betas)))


def test_carryover_seed_reproducibility():
    W = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    Y1 = CarryoverDGP(betas=[1.0, 0.5], seed=42).generate(W)
    Y2 = CarryoverDGP(betas=[1.0, 0.5], seed=42).generate(W)
    np.testing.assert_array_equal(Y1, Y2)


def test_carryover_callable_mu():
    W = np.array([1, 1, 1, 1])
    Y = CarryoverDGP(
        betas=[1.0, 0.5], mu=lambda t: t.astype(float), sigma=0.0, seed=0
    ).generate(W)
    # mu_t = t; Y_t = t + 1 + 0.5 * W_{t-1}.
    # t=0: 0 + 1 + 0 = 1
    # t=1: 1 + 1 + 0.5 = 2.5
    # t=2: 2 + 1 + 0.5 = 3.5
    # t=3: 3 + 1 + 0.5 = 4.5
    np.testing.assert_allclose(Y, [1.0, 2.5, 3.5, 4.5])


# ===========================================================================
# LatentStateDGP
# ===========================================================================

def test_latent_state_returns_outcomes_of_length_T():
    dgp = LatentStateDGP(seed=0)
    Y = dgp.generate(np.array([0, 1, 1, 0, 1, 0]))
    assert Y.shape == (6,)
    assert np.isfinite(Y).all()


def test_latent_state_validation():
    with pytest.raises(ValueError):
        LatentStateDGP(gamma=1.0)
    with pytest.raises(ValueError):
        LatentStateDGP(gamma=-1.5)
    with pytest.raises(ValueError):
        LatentStateDGP(sigma_y=-0.1)
    with pytest.raises(ValueError):
        LatentStateDGP(sigma_h=-0.1)


def test_latent_state_reduces_to_simple_when_alpha_and_sigma_h_zero():
    """alpha_0 = 0 and sigma_h = 0 with h0 = 0 → h_t ≡ 0 → Y_t = mu_t + beta_0 W_t + eps_y_t."""
    W = np.array([1, 0, 1, 1, 0])
    mu, beta_0, sigma_y = 0.5, 0.7, 1.0
    Y_state = LatentStateDGP(
        mu=mu, beta_0=beta_0, alpha_0=0.0, gamma=0.5,
        sigma_y=sigma_y, sigma_h=0.0, h0=0.0, seed=42,
    ).generate(W)
    Y_simple = SimpleDGP(mu=mu, tau=beta_0, sigma=sigma_y, seed=42).generate(W)
    # Both DGPs draw eps_y first; LatentStateDGP also draws eps_h afterwards
    # (size T zeros). Same first T draws -> Y identical.
    np.testing.assert_allclose(Y_state, Y_simple)


def test_latent_state_long_run_mean_under_sustained_treatment():
    """Under W ≡ 1, large-T mean of Y → mu + beta_0 + alpha_0/(1-gamma)."""
    mu, beta_0, alpha_0, gamma = 0.0, 1.0, 0.5, 0.6
    T = 50_000
    Y = LatentStateDGP(
        mu=mu, beta_0=beta_0, alpha_0=alpha_0, gamma=gamma,
        sigma_y=1.0, sigma_h=1.0, seed=0,
    ).generate(np.ones(T, dtype=int))
    expected = mu + beta_0 + alpha_0 / (1 - gamma)
    # Drop the initial transient; relax tolerance for the AR random walk.
    assert abs(Y[1000:].mean() - expected) < 0.1


def test_latent_state_long_run_mean_under_sustained_control():
    """Under W ≡ 0, large-T mean of Y → mu (h drifts around 0)."""
    mu = 0.3
    T = 50_000
    Y = LatentStateDGP(
        mu=mu, beta_0=1.0, alpha_0=0.5, gamma=0.6,
        sigma_y=1.0, sigma_h=1.0, seed=0,
    ).generate(np.zeros(T, dtype=int))
    assert abs(Y[1000:].mean() - mu) < 0.1


def test_latent_state_lag_m_estimand_matches_formula_under_bernoulli():
    """Under iid Bern(0.5), the lag-m effect (mean of Y given m+1 consecutive
    treatments minus mean given m+1 consecutive controls) converges to
    beta_0 + alpha_0/(1-gamma) * (1 - gamma^(m+1))."""
    beta_0, alpha_0, gamma = 1.0, 0.5, 0.5
    m = 2
    T = 100_000
    expected = beta_0 + alpha_0 / (1 - gamma) * (1 - gamma ** (m + 1))

    rng = np.random.default_rng(0)
    W = (rng.random(T) < 0.5).astype(int)
    Y = LatentStateDGP(
        mu=0.0, beta_0=beta_0, alpha_0=alpha_0, gamma=gamma,
        sigma_y=1.0, sigma_h=1.0, seed=1,
    ).generate(W)

    # Burn in to reach stationarity.
    burn = 1000
    treated_mask = np.array(
        [np.all(W[t - m : t + 1] == 1) for t in range(burn, T)]
    )
    control_mask = np.array(
        [np.all(W[t - m : t + 1] == 0) for t in range(burn, T)]
    )
    diff = Y[burn:][treated_mask].mean() - Y[burn:][control_mask].mean()
    assert abs(diff - expected) < 0.05


def test_latent_state_seed_reproducibility():
    W = np.array([0, 1, 1, 0, 1])
    Y1 = LatentStateDGP(seed=42).generate(W)
    Y2 = LatentStateDGP(seed=42).generate(W)
    np.testing.assert_array_equal(Y1, Y2)
