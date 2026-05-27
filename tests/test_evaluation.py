"""Tests for switchback.evaluation."""

import numpy as np
import pytest

from switchback.dgp import CarryoverDGP, LatentStateDGP, SimpleDGP
from switchback.evaluation import (
    normality_diagnostics,
    qq_compare,
    true_effect,
)


# ---------------------------------------------------------------------------
# true_effect
# ---------------------------------------------------------------------------

def test_true_effect_simple_dgp():
    """For SimpleDGP, the true effect equals tau (no carryover)."""
    dgp = SimpleDGP(mu=0.5, tau=1.5, sigma=1.0)
    assert true_effect(dgp, T=10_000, seed=0) == pytest.approx(1.5, abs=0.05)


def test_true_effect_carryover_dgp_finite_order():
    """For CarryoverDGP, the long-run sustained-treatment effect is sum(betas)."""
    betas = [1.0, 0.5, 0.25]
    dgp = CarryoverDGP(betas=betas, mu=0.0, sigma=1.0)
    expected = sum(betas)
    # Ignore a small finite-T transient (early periods see fewer carryover terms).
    assert true_effect(dgp, T=20_000, seed=0) == pytest.approx(expected, abs=0.01)


def test_true_effect_latent_state_dgp_long_run():
    """For LatentStateDGP, the true (long-run) effect = beta_0 + alpha_0/(1-gamma)."""
    beta_0, alpha_0, gamma = 1.0, 0.5, 0.5
    dgp = LatentStateDGP(
        mu=0.0, beta_0=beta_0, alpha_0=alpha_0, gamma=gamma,
        sigma_y=1.0, sigma_h=1.0,
    )
    expected = beta_0 + alpha_0 / (1.0 - gamma)
    assert true_effect(dgp, T=20_000, seed=0) == pytest.approx(expected, abs=0.05)


def test_true_effect_paired_noise_kills_variance():
    """With paired seeds, two independent invocations agree exactly (deterministic)."""
    dgp = LatentStateDGP(seed=42)
    a = true_effect(dgp, T=2_000, seed=7)
    b = true_effect(dgp, T=2_000, seed=7)
    assert a == b


def test_true_effect_T_must_be_positive():
    with pytest.raises(ValueError):
        true_effect(SimpleDGP(), T=0)


# ---------------------------------------------------------------------------
# normality_diagnostics
# ---------------------------------------------------------------------------

def test_normality_diagnostics_on_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, size=5_000)
    diag = normality_diagnostics(sample)
    assert abs(diag["mean"]) < 0.05
    assert abs(diag["std"] - 1.0) < 0.05
    # Skew and excess kurtosis should be within a few SE of zero.
    assert abs(diag["skewness"]) < 3 * diag["skewness_se"]
    assert abs(diag["excess_kurtosis"]) < 3 * diag["kurtosis_se"]


def test_normality_diagnostics_detects_skew():
    """Exponential(1) has skew=2, excess kurtosis=6 — both clearly non-zero."""
    rng = np.random.default_rng(0)
    sample = rng.exponential(scale=1.0, size=5_000)
    diag = normality_diagnostics(sample)
    assert diag["skewness"] > 1.0
    assert diag["excess_kurtosis"] > 2.0


def test_normality_diagnostics_validation():
    with pytest.raises(ValueError):
        normality_diagnostics([1.0, 2.0, 3.0])  # n < 4
    with pytest.raises(ValueError):
        normality_diagnostics([1.0, 1.0, 1.0, 1.0])  # zero variance


# ---------------------------------------------------------------------------
# qq_compare
# ---------------------------------------------------------------------------

def test_qq_compare_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(2.0, 0.5, size=5_000)
    qq = qq_compare(sample, percentiles=(10, 50, 90))
    for p, emp, nrm in qq:
        assert abs(emp - nrm) < 0.05  # empirical and N-fit quantiles align


def test_qq_compare_validation():
    with pytest.raises(ValueError):
        qq_compare([1.0])  # too few obs
    with pytest.raises(ValueError):
        qq_compare([1.0, 1.0, 1.0])  # zero variance
