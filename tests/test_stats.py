"""``gaffer.core.stats`` against closed-form values.

Everything here has an analytic answer, so nothing is asserted against the
implementation's own output. Where a quantity is only defined as a sum (E[X],
Var[X]) the reference is computed by brute force over a support wide enough for
the truncation error to sit well below the tolerance.
"""
from __future__ import annotations

import math

import pytest

from gaffer.core import stats


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.25, 1.0, 1.45, 2.7, 5.0])
def test_poisson_pmf_matches_the_closed_form(lam):
    for k in range(0, 12):
        expected = math.exp(-lam) * lam ** k / math.factorial(k)
        assert stats.poisson_pmf(k, lam) == pytest.approx(expected, rel=1e-12)


def test_poisson_pmf_known_values():
    # Textbook: Poisson(1) puts e^-1 on both 0 and 1.
    assert stats.poisson_pmf(0, 1.0) == pytest.approx(math.exp(-1.0), rel=1e-15)
    assert stats.poisson_pmf(1, 1.0) == pytest.approx(math.exp(-1.0), rel=1e-15)
    assert stats.poisson_pmf(2, 1.0) == pytest.approx(math.exp(-1.0) / 2.0, rel=1e-15)


def test_poisson_pmf_degenerate_and_negative_arguments():
    assert stats.poisson_pmf(0, 0.0) == 1.0
    assert stats.poisson_pmf(1, 0.0) == 0.0
    assert stats.poisson_pmf(-1, 1.5) == 0.0
    assert stats.poisson_pmf(0, -1.0) == 1.0


def test_poisson_pmf_sums_to_one():
    for lam in (0.3, 1.45, 4.0):
        assert sum(stats.poisson_pmf(k, lam) for k in range(0, 60)) == pytest.approx(1.0, abs=1e-12)


def test_poisson_mean_and_variance():
    lam = 1.45
    support = range(0, 60)
    mean = sum(k * stats.poisson_pmf(k, lam) for k in support)
    var = sum((k - mean) ** 2 * stats.poisson_pmf(k, lam) for k in support)
    assert mean == pytest.approx(lam, abs=1e-10)
    assert var == pytest.approx(lam, abs=1e-10)


def test_poisson_cdf_sf_and_at_least_are_consistent():
    lam = 2.2
    for k in range(0, 8):
        cdf = stats.poisson_cdf(k, lam)
        assert cdf == pytest.approx(sum(stats.poisson_pmf(i, lam) for i in range(0, k + 1)))
        assert stats.poisson_sf(k, lam) == pytest.approx(1.0 - cdf, abs=1e-12)
        assert stats.poisson_at_least(k + 1, lam) == pytest.approx(1.0 - cdf, abs=1e-12)
    assert stats.poisson_at_least(0, lam) == 1.0
    assert stats.poisson_at_least(-3, lam) == 1.0


def test_poisson_at_least_one_is_one_minus_p_zero():
    lam = 1.45
    assert stats.poisson_at_least(1, lam) == pytest.approx(1.0 - math.exp(-lam), abs=1e-12)


def test_p_clean_sheet_is_exp_minus_lambda():
    for lam in (0.0, 0.4, 1.45, 3.3):
        assert stats.p_clean_sheet(lam) == pytest.approx(math.exp(-lam), rel=1e-15)
    # A negative rate is clamped, not exponentiated into a probability above 1.
    assert stats.p_clean_sheet(-2.0) == 1.0


def test_poisson_vector_length_and_contents():
    vec = stats.poisson_vector(1.2, max_k=6)
    assert len(vec) == 7
    assert vec == [pytest.approx(stats.poisson_pmf(k, 1.2)) for k in range(7)]


# ---------------------------------------------------------------------------
# Negative binomial
# ---------------------------------------------------------------------------


def test_negbin_reduces_to_poisson_for_huge_dispersion():
    for k in range(0, 10):
        assert stats.negbin_pmf(k, 2.0, 1e9) == pytest.approx(stats.poisson_pmf(k, 2.0), rel=1e-9)


def test_negbin_moments_match_mean_and_mean_plus_mean_squared_over_r():
    mean, r = 2.0, 4.0
    support = range(0, 400)
    m = sum(k * stats.negbin_pmf(k, mean, r) for k in support)
    v = sum((k - m) ** 2 * stats.negbin_pmf(k, mean, r) for k in support)
    assert m == pytest.approx(mean, abs=1e-9)
    assert v == pytest.approx(mean + mean * mean / r, abs=1e-9)


def test_negbin_is_overdispersed_relative_to_poisson():
    mean, r = 8.0, 3.0
    support = range(0, 400)
    m = sum(k * stats.negbin_pmf(k, mean, r) for k in support)
    v = sum((k - m) ** 2 * stats.negbin_pmf(k, mean, r) for k in support)
    assert v > mean  # Poisson would have variance == mean


def test_negbin_pmf_sums_to_one():
    assert sum(stats.negbin_pmf(k, 7.7, 5.0) for k in range(0, 500)) == pytest.approx(1.0, abs=1e-9)


def test_negbin_pmf_at_zero_mean():
    assert stats.negbin_pmf(0, 0.0, 4.0) == 1.0
    assert stats.negbin_pmf(2, 0.0, 4.0) == 0.0


def test_negbin_at_least_is_the_tail_sum():
    mean, r = 7.77, 4.0
    for k in (1, 5, 10, 12, 20):
        tail = sum(stats.negbin_pmf(i, mean, r) for i in range(k, 600))
        assert stats.negbin_at_least(k, mean, r) == pytest.approx(tail, abs=1e-9)
    assert stats.negbin_at_least(0, mean, r) == 1.0
    assert stats.negbin_at_least(-2, mean, r) == 1.0


def test_negbin_at_least_is_monotone_decreasing_in_k():
    values = [stats.negbin_at_least(k, 7.77, 4.0) for k in range(0, 25)]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_negbin_at_least_increases_with_the_mean():
    """The DEFCON threshold model leans on this: more actions, more chance."""
    values = [stats.negbin_at_least(10, mean, 4.0) for mean in (4.0, 6.0, 8.0, 12.0)]
    assert values == sorted(values)


def test_fit_dispersion_method_of_moments():
    # variance = mean + mean^2 / r  =>  r = mean^2 / (variance - mean)
    assert stats.fit_dispersion(2.0, 3.0) == pytest.approx(4.0)
    assert stats.fit_dispersion(8.0, 24.0) == pytest.approx(4.0)
    # Underdispersed or degenerate input falls back to "effectively Poisson".
    assert stats.fit_dispersion(2.0, 2.0) >= 1e6
    assert stats.fit_dispersion(2.0, 1.0) >= 1e6
    assert stats.fit_dispersion(0.0, 5.0) >= 1e6


def test_fit_dispersion_round_trips_through_negbin_moments():
    mean, var = 7.0, 19.0
    r = stats.fit_dispersion(mean, var)
    support = range(0, 600)
    m = sum(k * stats.negbin_pmf(k, mean, r) for k in support)
    v = sum((k - m) ** 2 * stats.negbin_pmf(k, mean, r) for k in support)
    assert m == pytest.approx(mean, abs=1e-8)
    assert v == pytest.approx(var, abs=1e-6)


# ---------------------------------------------------------------------------
# Dixon-Coles
# ---------------------------------------------------------------------------


def test_dixon_coles_tau_closed_form():
    lam, mu, rho = 1.5, 1.2, -0.06
    assert stats.dixon_coles_tau(0, 0, lam, mu, rho) == pytest.approx(1.0 - lam * mu * rho)
    assert stats.dixon_coles_tau(0, 1, lam, mu, rho) == pytest.approx(1.0 + lam * rho)
    assert stats.dixon_coles_tau(1, 0, lam, mu, rho) == pytest.approx(1.0 + mu * rho)
    assert stats.dixon_coles_tau(1, 1, lam, mu, rho) == pytest.approx(1.0 - rho)
    for x, y in ((2, 0), (0, 2), (2, 2), (3, 1)):
        assert stats.dixon_coles_tau(x, y, lam, mu, rho) == 1.0


def test_dixon_coles_is_the_identity_at_rho_zero():
    for x in range(3):
        for y in range(3):
            assert stats.dixon_coles_tau(x, y, 1.4, 1.1, 0.0) == 1.0


def test_score_matrix_is_a_probability_distribution():
    m = stats.score_matrix(1.6, 1.1)
    total = sum(p for row in m for p in row)
    assert total == pytest.approx(1.0, abs=1e-12)
    assert all(p >= 0.0 for row in m for p in row)


def test_score_matrix_reduces_to_independent_poisson_at_rho_zero():
    lam_h, lam_a = 1.6, 1.1
    m = stats.score_matrix(lam_h, lam_a, rho=0.0, max_goals=14)
    for x in (0, 1, 2, 3):
        for y in (0, 1, 2, 3):
            expected = stats.poisson_pmf(x, lam_h) * stats.poisson_pmf(y, lam_a)
            assert m[x][y] == pytest.approx(expected, abs=1e-9)


def test_match_outcome_probs_sum_to_one():
    h, d, a = stats.match_outcome_probs(1.7, 1.2)
    assert h + d + a == pytest.approx(1.0, abs=1e-12)
    assert h > a  # the stronger attack is the favourite


def test_match_outcome_probs_are_symmetric_for_equal_lambdas():
    h, d, a = stats.match_outcome_probs(1.4, 1.4)
    assert h == pytest.approx(a, abs=1e-12)
    assert d > 0.2


def test_dixon_coles_lifts_the_nil_nil_probability():
    """Negative rho is there to fix Poisson's under-prediction of 0-0 and 1-1."""
    plain = stats.score_matrix(1.3, 1.1, rho=0.0)
    corrected = stats.score_matrix(1.3, 1.1, rho=-0.06)
    assert corrected[0][0] > plain[0][0]
    assert corrected[1][1] > plain[1][1]
    assert corrected[0][1] < plain[0][1]


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------


def test_remove_vig_normalises_implied_probabilities():
    probs = stats.remove_vig([2.0, 4.0, 4.0])
    assert probs == [pytest.approx(0.5), pytest.approx(0.25), pytest.approx(0.25)]
    assert sum(probs) == pytest.approx(1.0)


def test_remove_vig_strips_the_overround():
    # A 1.90 / 1.90 book is 105.3% — after de-vigging both sides are 50%.
    probs = stats.remove_vig([1.90, 1.90])
    assert probs == [pytest.approx(0.5), pytest.approx(0.5)]


def test_remove_vig_on_empty_or_invalid_input():
    assert stats.remove_vig([]) == []
    assert stats.remove_vig([0.0, 0.0]) == []


def test_lambdas_from_odds_recovers_the_probabilities_it_was_given():
    p_home, p_draw, p_away = stats.remove_vig([1.8, 3.6, 4.5])
    lam_h, lam_a = stats.lambdas_from_odds(p_home, p_draw, p_away, total_goals_line=2.7)
    assert lam_h > lam_a  # the favourite gets the bigger rate
    assert lam_h + lam_a == pytest.approx(2.7, abs=1e-9)
    got_h, _, got_a = stats.match_outcome_probs(lam_h, lam_a)
    # Supremacy is solved by bisection, so the win margin is matched exactly and
    # the individual probabilities to within the draw's share.
    assert (got_h - got_a) == pytest.approx(p_home - p_away, abs=1e-3)


def test_lambdas_from_odds_is_symmetric_for_a_coin_flip():
    lam_h, lam_a = stats.lambdas_from_odds(0.4, 0.2, 0.4, total_goals_line=2.6)
    assert lam_h == pytest.approx(lam_a, abs=1e-3)
    assert lam_h == pytest.approx(1.3, abs=1e-3)


# ---------------------------------------------------------------------------
# Shrinkage and blending
# ---------------------------------------------------------------------------


def test_shrink_is_the_weighted_average_of_observation_and_prior():
    # w = n / (n + strength). n = 8, strength = 8 -> exactly halfway.
    assert stats.shrink(0.5, 8.0, 0.1, 8.0) == pytest.approx(0.3)
    # n = 24, strength = 8 -> w = 0.75.
    assert stats.shrink(0.5, 24.0, 0.1, 8.0) == pytest.approx(0.75 * 0.5 + 0.25 * 0.1)


def test_shrink_with_no_data_returns_the_prior():
    assert stats.shrink(0.9, 0.0, 0.12, 8.0) == 0.12
    assert stats.shrink(0.9, -1.0, 0.12, 8.0) == 0.12


def test_shrink_converges_to_the_observation_with_enough_data():
    assert stats.shrink(0.5, 1e6, 0.1, 8.0) == pytest.approx(0.5, abs=1e-4)


def test_shrink_is_monotone_in_sample_size():
    values = [stats.shrink(0.5, n, 0.1, 8.0) for n in (0, 1, 4, 8, 32, 200)]
    assert values == sorted(values)


def test_beta_binomial_rate_posterior_mean():
    # prior 0.5 worth 10 pseudo-trials, then 5 successes in 10 trials.
    assert stats.beta_binomial_rate(5, 10, 0.5, 10) == pytest.approx(0.5)
    assert stats.beta_binomial_rate(10, 10, 0.5, 10) == pytest.approx(0.75)
    assert stats.beta_binomial_rate(0, 10, 0.5, 10) == pytest.approx(0.25)
    # No trials at all: the prior survives untouched.
    assert stats.beta_binomial_rate(0, 0, 0.3, 10) == pytest.approx(0.3)


def test_weighted_blend():
    assert stats.weighted_blend([1.0, 2.0], [1.0, 1.0]) == pytest.approx(1.5)
    assert stats.weighted_blend([1.0, 2.0], [3.0, 1.0]) == pytest.approx(1.25)
    # Zero and negative weights drop out; the rest are renormalised.
    assert stats.weighted_blend([1.0, 99.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert stats.weighted_blend([], []) == 0.0
    assert stats.weighted_blend([1.0], [0.0]) == 0.0


def test_weighted_blend_ignores_none_values():
    assert stats.weighted_blend([None, 4.0], [5.0, 1.0]) == pytest.approx(4.0)


def test_exponential_decay_weight_halves_every_half_life():
    assert stats.exponential_decay_weight(0.0, 180.0) == 1.0
    assert stats.exponential_decay_weight(180.0, 180.0) == pytest.approx(0.5)
    assert stats.exponential_decay_weight(360.0, 180.0) == pytest.approx(0.25)
    assert stats.exponential_decay_weight(90.0, 180.0) == pytest.approx(0.5 ** 0.5)
    # A non-positive half-life means "no decay", not a division by zero.
    assert stats.exponential_decay_weight(500.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# Top-k (bonus ranking)
# ---------------------------------------------------------------------------


def test_top_k_probabilities_are_a_distribution_over_each_rank():
    scores = {1: 30.0, 2: 26.0, 3: 24.0, 4: 20.0, 5: 12.0}
    sd = {i: 6.0 for i in scores}
    out = stats.top_k_probabilities(scores, sd, k=3, iterations=4000, seed=7)
    assert set(out) == set(scores)
    for rank in range(3):
        assert sum(out[i][rank] for i in scores) == pytest.approx(1.0, abs=1e-9)
    for probs in out.values():
        assert all(0.0 <= p <= 1.0 for p in probs)


def test_top_k_probabilities_favour_the_best_score():
    scores = {1: 40.0, 2: 20.0, 3: 18.0}
    sd = {i: 3.0 for i in scores}
    out = stats.top_k_probabilities(scores, sd, k=3, iterations=4000, seed=7)
    # Twenty BPS clear with sd 3 is a lock on the three bonus.
    assert out[1][0] > 0.99
    assert out[2][0] < 0.01 and out[3][0] < 0.01
    # Second and third are two apart, so second is favoured but not certain.
    assert out[2][1] > out[3][1] > 0.1
    assert out[1][0] > out[2][1]


def test_top_k_probabilities_split_a_dead_heat_evenly():
    scores = {1: 25.0, 2: 25.0}
    sd = {i: 4.0 for i in scores}
    out = stats.top_k_probabilities(scores, sd, k=2, iterations=8000, seed=7)
    assert out[1][0] == pytest.approx(0.5, abs=0.03)
    assert out[2][0] == pytest.approx(0.5, abs=0.03)


def test_top_k_probabilities_is_deterministic_for_a_fixed_seed():
    scores = {1: 30.0, 2: 26.0, 3: 24.0}
    sd = {i: 5.0 for i in scores}
    a = stats.top_k_probabilities(scores, sd, k=3, iterations=2000, seed=11)
    b = stats.top_k_probabilities(scores, sd, k=3, iterations=2000, seed=11)
    assert a == b


def test_top_k_probabilities_on_an_empty_field():
    assert stats.top_k_probabilities({}, {}) == {}


# ---------------------------------------------------------------------------
# Backtest metrics
# ---------------------------------------------------------------------------


def test_rmse_and_mae_known_values():
    assert stats.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert stats.mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # errors 3 and 4: rmse = sqrt((9 + 16) / 2), mae = 3.5
    assert stats.rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert stats.mae([0.0, 0.0], [3.0, 4.0]) == pytest.approx(3.5)
    # Sign does not matter.
    assert stats.rmse([5.0], [2.0]) == pytest.approx(3.0)
    assert stats.rmse([2.0], [5.0]) == pytest.approx(3.0)


def test_rmse_is_never_below_mae():
    pred = [1.0, 4.0, 2.5, 9.0]
    actual = [2.0, 2.0, 2.5, 3.0]
    assert stats.rmse(pred, actual) >= stats.mae(pred, actual)


def test_metrics_on_empty_input_are_nan():
    assert math.isnan(stats.rmse([], []))
    assert math.isnan(stats.mae([], []))
    assert math.isnan(stats.spearman([1.0], [1.0]))


def test_spearman_perfect_monotone_relationships():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Rank correlation, not linear: a monotone but curved map is still +1.
    assert stats.spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_known_value_with_one_swap():
    # rho = 1 - 6*sum(d^2)/(n(n^2-1)); one adjacent swap in n=4 gives d^2 sum 2.
    got = stats.spearman([1, 2, 3, 4], [1, 2, 4, 3])
    assert got == pytest.approx(1.0 - 6.0 * 2.0 / (4 * (16 - 1)))


def test_spearman_handles_ties_with_average_ranks():
    assert stats.spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    # A constant series has no rank variance at all: undefined, not zero.
    assert math.isnan(stats.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
