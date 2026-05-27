"""Tests for switchback.design (BernoulliDesign)."""

import numpy as np
import pytest

from switchback.design import BaseDesign, BernoulliDesign, CompleteRandomization


# ---------------------------------------------------------------------------
# Window structure
# ---------------------------------------------------------------------------

def test_assignment_is_constant_within_window():
    """All periods inside a window must share the same draw."""
    design = BernoulliDesign(l=3, seed=0)
    T = 10  # windows: [0,2], [3,5], [6,8], [9,9]
    for _ in range(50):
        W = design.sample(T)
        assert len(set(W[0:3])) == 1
        assert len(set(W[3:6])) == 1
        assert len(set(W[6:9])) == 1
        # Last window is a singleton, trivially constant.


def test_window_index_partitions_horizon():
    design = BernoulliDesign(l=3)
    np.testing.assert_array_equal(
        design.window_index(T=10),
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
    )


def test_window_length_one_is_per_period():
    design = BernoulliDesign(l=1, seed=0)
    samples = np.stack([design.sample(20) for _ in range(2_000)])
    # Independent fair coins -> adjacent agreement rate ~= 1/2.
    agree = (samples[:, :-1] == samples[:, 1:]).mean()
    assert abs(agree - 0.5) < 0.02
    # Marginal mean should be ~0.5.
    assert abs(samples.mean() - 0.5) < 0.01


def test_n_windows_handles_uneven_horizon():
    design = BernoulliDesign(l=4)
    assert design.n_windows(8) == 2     # exact fit
    assert design.n_windows(9) == 3     # one extra short window
    assert design.n_windows(1) == 1


def test_randomization_points_match_window_starts():
    design = BernoulliDesign(l=3)
    np.testing.assert_array_equal(design.randomization_points(T=10), [0, 3, 6, 9])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_window_length_must_be_positive_integer():
    with pytest.raises(ValueError):
        BernoulliDesign(l=0)
    with pytest.raises(ValueError):
        BernoulliDesign(l=-1)


def test_p_must_be_in_open_unit_interval():
    with pytest.raises(ValueError):
        BernoulliDesign(p=0.0)
    with pytest.raises(ValueError):
        BernoulliDesign(p=1.0)


def test_T_must_be_positive():
    design = BernoulliDesign()
    with pytest.raises(ValueError):
        design.sample(0)


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

def test_consecutive_prob_within_one_window():
    """Window inside one window: probability is just p (or 1-p)."""
    design = BernoulliDesign(l=4, p=0.3)
    # t=2, p=2 -> window [0, 2] all in window 0.
    assert design.consecutive_prob(T=8, t=2, p=2, value=1) == pytest.approx(0.3)
    assert design.consecutive_prob(T=8, t=2, p=2, value=0) == pytest.approx(0.7)


def test_consecutive_prob_across_windows_multiplies():
    """B distinct windows intersected -> p^B (or (1-p)^B)."""
    design = BernoulliDesign(l=4, p=0.3)
    # t=4, p=2 -> window [2, 4]: indices 2,3 in window 0; index 4 in window 1.
    assert design.consecutive_prob(T=8, t=4, p=2, value=1) == pytest.approx(0.3 ** 2)
    assert design.consecutive_prob(T=8, t=4, p=2, value=0) == pytest.approx(0.7 ** 2)


def test_consecutive_prob_three_windows():
    design = BernoulliDesign(l=2, p=0.5)
    # t=5, p=4 -> window [1, 5]: index 1 in window 0; 2,3 in window 1; 4,5 in window 2.
    assert design.consecutive_prob(T=8, t=5, p=4, value=1) == pytest.approx(0.5 ** 3)


def test_window_marginal_probability_via_monte_carlo():
    """Empirical Pr(W_{t-p:t}=1_{p+1}) should match consecutive_prob."""
    design = BernoulliDesign(l=3, p=0.4, seed=0)
    T, t, p = 9, 5, 3  # window [2, 5] spans windows 0 (idx 2), 1 (3,4,5).
    expected = design.consecutive_prob(T, t, p, value=1)
    n_samples = 20_000
    hits = sum(np.all(design.sample(T)[t - p : t + 1] == 1) for _ in range(n_samples))
    assert abs(hits / n_samples - expected) < 0.02


def test_consecutive_prob_window_out_of_bounds():
    design = BernoulliDesign(l=2)
    with pytest.raises(ValueError):
        design.consecutive_prob(T=4, t=1, p=2, value=1)  # t-p = -1
    with pytest.raises(ValueError):
        design.consecutive_prob(T=4, t=4, p=0, value=1)  # t = T


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    d1 = BernoulliDesign(l=3, seed=42)
    d2 = BernoulliDesign(l=3, seed=42)
    np.testing.assert_array_equal(d1.sample(12), d2.sample(12))


def test_reset_replays_same_path():
    d = BernoulliDesign(l=2, seed=7)
    a = d.sample(10)
    d.reset()
    b = d.sample(10)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Subclass discipline
# ---------------------------------------------------------------------------

def test_basedesign_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseDesign(seed=0)  # type: ignore[abstract]


# ===========================================================================
# CompleteRandomization
# ===========================================================================

def test_complete_rand_treated_count_is_exactly_half():
    """Every realised path has exactly T/2 treated periods."""
    design = CompleteRandomization(l=2, seed=0)
    T = 20
    for _ in range(50):
        W = design.sample(T)
        assert W.sum() == T // 2


def test_complete_rand_assignment_constant_within_window():
    design = CompleteRandomization(l=3, seed=0)
    T = 24  # 8 windows (even)
    for _ in range(50):
        W = design.sample(T)
        for b in range(8):
            assert len(set(W[3 * b : 3 * (b + 1)])) == 1


def test_complete_rand_window_index_partitions_horizon():
    design = CompleteRandomization(l=3)
    np.testing.assert_array_equal(
        design.window_index(T=12),
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    )


def test_complete_rand_T_must_be_divisible_by_window_length():
    design = CompleteRandomization(l=3)
    with pytest.raises(ValueError):
        design.sample(T=10)


def test_complete_rand_n_windows_must_be_even():
    """n_windows = T/l must be even (for the half/half split)."""
    design = CompleteRandomization(l=2)
    with pytest.raises(ValueError):
        design.sample(T=6)  # n=3, odd


def test_complete_rand_window_length_must_be_positive_integer():
    with pytest.raises(ValueError):
        CompleteRandomization(l=0)
    with pytest.raises(ValueError):
        CompleteRandomization(l=-1)


def test_complete_rand_seed_reproducibility():
    a = CompleteRandomization(l=2, seed=42).sample(20)
    b = CompleteRandomization(l=2, seed=42).sample(20)
    np.testing.assert_array_equal(a, b)


# ----- Propensity (hypergeometric) -----

def test_complete_rand_marginal_prob_is_half():
    """For a single-window window, Pr = (n/2)/n = 1/2."""
    design = CompleteRandomization(l=2)
    # T=20 -> n=10, t=0, p=0 covers window 0 only.
    assert design.consecutive_prob(T=20, t=0, p=0, value=1) == pytest.approx(0.5)
    assert design.consecutive_prob(T=20, t=0, p=0, value=0) == pytest.approx(0.5)


def test_complete_rand_two_window_prob_is_hypergeometric():
    """Pr(2 specific windows both treated) = (n/2)*(n/2-1) / (n*(n-1))."""
    design = CompleteRandomization(l=2)
    T = 20
    # Window [1, 2] crosses windows 0 and 1.
    # n=10, B=2 -> Pr = 5*4 / (10*9) = 20/90 = 2/9.
    assert design.consecutive_prob(T=T, t=2, p=1, value=1) == pytest.approx(2.0 / 9.0)
    assert design.consecutive_prob(T=T, t=2, p=1, value=0) == pytest.approx(2.0 / 9.0)


def test_complete_rand_three_window_prob_is_hypergeometric():
    design = CompleteRandomization(l=2)
    # T=20, n=10, B=3 spans windows 0,1,2 (window [1, 4] over t=4, p=3).
    # Pr = 5*4*3 / (10*9*8) = 60/720 = 1/12.
    assert design.consecutive_prob(T=20, t=4, p=3, value=1) == pytest.approx(1.0 / 12.0)


def test_complete_rand_prob_zero_when_more_than_half_windows():
    """If B > n/2, no allocation has all B treated."""
    design = CompleteRandomization(l=1)
    # T=4 -> n=4, half=2; B=3 in window means impossible.
    assert design.consecutive_prob(T=4, t=2, p=2, value=1) == 0.0


def test_complete_rand_marginal_via_monte_carlo():
    """Empirical Pr(W_{t-p:t} ≡ 1) matches consecutive_prob."""
    design = CompleteRandomization(l=2, seed=0)
    T, t, p = 20, 4, 3  # B = 3 distinct windows (0, 1, 2)
    expected = design.consecutive_prob(T, t, p, value=1)
    n_samples = 20_000
    hits = sum(np.all(design.sample(T)[t - p : t + 1] == 1) for _ in range(n_samples))
    assert abs(hits / n_samples - expected) < 0.02


def test_complete_rand_converges_to_bernoulli_for_large_n():
    """Pr → (1/2)^B as n → ∞."""
    K = 1
    B = 3
    # Pick a large n.
    n = 1000
    T = n * K
    design = CompleteRandomization(l=K)
    pr = design.consecutive_prob(T=T, t=B - 1, p=B - 1, value=1)
    assert abs(pr - (0.5) ** B) < 0.005
