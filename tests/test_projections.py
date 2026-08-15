"""Probability and accounting invariants over a whole projection set.

These are the assertions that catch a model change quietly producing nonsense:
a p_60 above p_appear, a NaN leaking out of a division, a component that stops
being added to the total. They run over every player and every gameweek, so
they are cheap insurance against a regression anywhere in ``gaffer.model``.
"""
from __future__ import annotations

import math

import pytest

from gaffer.core import scoring
from gaffer.core.types import PlayerFixtureProjection, PlayerGWProjection, ProjectionSet


def iter_fixture_rows(projections: ProjectionSet):
    for pid, per_gw in projections.projections.items():
        for gw, gwp in per_gw.items():
            for fp in gwp.fixtures:
                yield pid, gw, gwp, fp


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_projection_set_covers_every_player_and_gameweek(projections, game_state):
    assert projections.first_gw == 1 and projections.last_gw == 6
    assert set(projections.projections) == set(game_state.players)
    for pid, per_gw in projections.projections.items():
        assert sorted(per_gw) == [1, 2, 3, 4, 5, 6], "player %d" % pid


def test_projection_set_xp_lookup_and_missing_keys(projections):
    pid = projections.player_ids()[0]
    assert projections.xp(pid, 1) == projections.projections[pid][1].xp
    assert projections.xp(-1, 1) == 0.0  # unknown player is 0, not a KeyError
    assert projections.xp(pid, 99) == 0.0
    assert projections.gws() == [1, 2, 3, 4, 5, 6]


def test_every_projected_fixture_belongs_to_the_players_own_team(projections, game_state):
    by_id = {f.id: f for f in game_state.fixtures}
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        fixture = by_id[fp.fixture_id]
        team = game_state.players[pid].team_id
        assert team in (fixture.team_h, fixture.team_a)
        assert fp.opponent_id == fixture.opponent_of(team)
        assert fp.is_home == (team == fixture.team_h)
        assert int(fixture.gw) == gw


# ---------------------------------------------------------------------------
# Probability invariants
# ---------------------------------------------------------------------------


def test_p60_never_exceeds_p_appear_and_both_are_probabilities(projections):
    checked = 0
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert 0.0 <= fp.p_60 <= fp.p_appear <= 1.0, (
            "player %d GW%d: p_60=%r p_appear=%r" % (pid, gw, fp.p_60, fp.p_appear)
        )
        checked += 1
    assert checked > 1000, "only %d fixture rows checked" % checked


def test_p_start_never_exceeds_p_appear(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert 0.0 <= fp.p_start <= fp.p_appear + 1e-12, (
            "player %d GW%d: p_start=%r p_appear=%r" % (pid, gw, fp.p_start, fp.p_appear)
        )


def test_expected_minutes_stay_inside_a_single_match(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert 0.0 <= fp.xmins <= 90.0, "player %d GW%d: xmins=%r" % (pid, gw, fp.xmins)


def test_a_player_who_cannot_appear_has_no_expected_minutes(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        if fp.p_appear == 0.0:
            assert fp.xmins == 0.0, "player %d GW%d" % (pid, gw)
            assert fp.xp_total == pytest.approx(0.0, abs=1e-9)


def test_every_other_probability_is_in_the_unit_interval(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        for name in ("p_clean_sheet", "p_defcon"):
            value = getattr(fp, name)
            assert 0.0 <= value <= 1.0, "player %d GW%d %s=%r" % (pid, gw, name, value)


def test_no_nan_anywhere_in_the_projection_set(projections):
    numeric = (
        "p_appear", "p_start", "p_60", "xmins", "lambda_goals", "lambda_assists",
        "lambda_conceded", "lambda_saves", "p_clean_sheet", "p_defcon", "exp_bps",
        "xp_total", "sd_total",
    )
    for pid, gw, gwp, fp in iter_fixture_rows(projections):
        assert not math.isnan(gwp.xp) and not math.isnan(gwp.sd)
        for name in numeric:
            value = float(getattr(fp, name))
            assert not math.isnan(value), "player %d GW%d %s is NaN" % (pid, gw, name)
            assert not math.isinf(value), "player %d GW%d %s is inf" % (pid, gw, name)
        for name, value in fp.components().items():
            assert not math.isnan(value), "player %d GW%d component %s is NaN" % (pid, gw, name)


def test_rates_and_spreads_are_non_negative(projections):
    for pid, gw, gwp, fp in iter_fixture_rows(projections):
        for name in ("lambda_goals", "lambda_assists", "lambda_conceded",
                     "lambda_saves", "exp_bps", "sd_total"):
            assert getattr(fp, name) >= 0.0, "player %d GW%d %s" % (pid, gw, name)
        assert gwp.sd >= 0.0


# ---------------------------------------------------------------------------
# Accounting: components add up
# ---------------------------------------------------------------------------


def test_components_sum_to_the_fixture_total(projections):
    worst = 0.0
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        gap = abs(sum(fp.components().values()) - fp.xp_total)
        worst = max(worst, gap)
        assert gap < 1e-9, "player %d GW%d: components off by %.3e" % (pid, gw, gap)
    assert worst < 1e-9


def test_fixture_totals_sum_to_the_gameweek_total(projections):
    for pid, per_gw in projections.projections.items():
        for gw, gwp in per_gw.items():
            total = sum(fp.xp_total for fp in gwp.fixtures)
            assert gwp.xp == pytest.approx(total, abs=1e-9), "player %d GW%d" % (pid, gw)


def test_a_blank_gameweek_is_an_explicit_zero(projections):
    for pid, per_gw in projections.projections.items():
        for gw, gwp in per_gw.items():
            if not gwp.fixtures:
                assert gwp.is_blank is True
                assert gwp.xp == 0.0
                assert gwp.sd == 0.0


def test_gameweek_standard_deviation_is_the_root_sum_of_squares(projections):
    for pid, per_gw in projections.projections.items():
        for gw, gwp in per_gw.items():
            expected = math.sqrt(sum(fp.sd_total ** 2 for fp in gwp.fixtures))
            assert gwp.sd == pytest.approx(expected, abs=1e-9), "player %d GW%d" % (pid, gw)


# ---------------------------------------------------------------------------
# Position rules show up in the components
# ---------------------------------------------------------------------------


def test_only_goalkeepers_earn_save_points(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        if game_state.players[pid].position != scoring.GKP:
            assert fp.xp_saves == 0.0, "player %d GW%d" % (pid, gw)
            assert fp.lambda_saves == 0.0


def test_only_goalkeepers_can_gain_from_the_penalty_component(projections, game_state):
    """``xp_penalty`` is saves minus misses, so an outfield taker is negative."""
    outfield_misses = 0
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        if game_state.players[pid].position != scoring.GKP:
            assert fp.xp_penalty <= 0.0, "player %d GW%d" % (pid, gw)
            outfield_misses += int(fp.xp_penalty < 0.0)
            assert fp.xp_penalty >= scoring.PENALTY_MISS, "player %d GW%d" % (pid, gw)
    assert outfield_misses > 0, "no penalty taker carries any miss risk"


def test_goalkeepers_are_never_given_defcon_points(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        if game_state.players[pid].position == scoring.GKP:
            assert fp.p_defcon == 0.0, "player %d GW%d" % (pid, gw)
            assert fp.xp_defcon == 0.0


def test_defcon_expectation_never_exceeds_the_two_point_cap(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        cap = scoring.DEFCON_POINTS[game_state.players[pid].position]
        assert 0.0 <= fp.xp_defcon <= cap + 1e-12, "player %d GW%d" % (pid, gw)


def test_forwards_never_earn_clean_sheet_points(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        if game_state.players[pid].position == scoring.FWD:
            assert fp.xp_clean_sheet == 0.0, "player %d GW%d" % (pid, gw)


def test_only_keepers_and_defenders_are_docked_for_goals_conceded(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        position = game_state.players[pid].position
        if position in (scoring.MID, scoring.FWD):
            assert fp.xp_goals_conceded == 0.0, "player %d GW%d" % (pid, gw)
        else:
            assert fp.xp_goals_conceded <= 0.0, "player %d GW%d" % (pid, gw)


def test_clean_sheet_expectation_never_exceeds_the_rule_value(projections, game_state):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        cap = scoring.CLEAN_SHEET_POINTS[game_state.players[pid].position]
        assert 0.0 <= fp.xp_clean_sheet <= cap + 1e-12, "player %d GW%d" % (pid, gw)


def test_appearance_points_never_exceed_two(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert 0.0 <= fp.xp_appearance <= scoring.LONG_PLAY + 1e-12, (
            "player %d GW%d" % (pid, gw)
        )
        assert fp.xp_appearance == pytest.approx(
            scoring.expected_appearance_points(fp.p_appear, fp.p_60), abs=1e-9
        )


def test_cards_only_ever_cost_points(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert fp.xp_cards <= 0.0, "player %d GW%d" % (pid, gw)
        assert fp.xp_cards >= scoring.RED_CARD, "player %d GW%d" % (pid, gw)


def test_attacking_components_are_non_negative(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert fp.xp_goals >= 0.0 and fp.xp_assists >= 0.0 and fp.xp_bonus >= 0.0


def test_expected_bonus_never_exceeds_three(projections):
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert fp.xp_bonus <= 3.0 + 1e-12, "player %d GW%d" % (pid, gw)


def test_no_single_fixture_projects_an_absurd_total(projections):
    """A sanity band: nobody is worth 20 points a match in expectation."""
    for pid, gw, _gwp, fp in iter_fixture_rows(projections):
        assert -2.0 <= fp.xp_total <= 20.0, "player %d GW%d: %r" % (pid, gw, fp.xp_total)


def test_the_best_projected_player_is_actually_good(projections):
    """Guards against a silent collapse to all-zeros."""
    best = max(gwp.xp for per in projections.projections.values() for gwp in per.values())
    assert best > 3.0


# ---------------------------------------------------------------------------
# Serialisation round trip
# ---------------------------------------------------------------------------


def test_projection_set_survives_a_json_round_trip(projections):
    from gaffer.model.xp import (
        projection_set_from_dict, projection_set_to_dict, projection_sets_equal,
    )

    rebuilt = projection_set_from_dict(projection_set_to_dict(projections))
    same, why = projection_sets_equal(projections, rebuilt)
    assert same, why


# ---------------------------------------------------------------------------
# The same invariants on the synthetic set, so they run without a cache
# ---------------------------------------------------------------------------


def test_synthetic_projection_set_satisfies_the_same_invariants(synthetic_projections):
    rows = 0
    for pid, gw, gwp, fp in iter_fixture_rows(synthetic_projections):
        rows += 1
        assert 0.0 <= fp.p_60 <= fp.p_appear <= 1.0
        assert 0.0 <= fp.xmins <= 90.0
        assert not math.isnan(fp.xp_total)
        assert fp.xp_total >= 0.0
    assert rows > 0
    for pid, per_gw in synthetic_projections.projections.items():
        for gw, gwp in per_gw.items():
            assert gwp.xp == pytest.approx(sum(f.xp_total for f in gwp.fixtures), abs=1e-9)


def test_components_of_a_hand_built_projection_sum_to_the_total():
    """The contract ``components()`` states, on a value nobody fitted."""
    fp = PlayerFixtureProjection(
        player_id=1, gw=1, fixture_id=1, opponent_id=2, is_home=True,
        xp_appearance=1.8, xp_goals=0.9, xp_assists=0.4, xp_clean_sheet=1.2,
        xp_goals_conceded=-0.5, xp_saves=0.0, xp_defcon=0.6, xp_bonus=0.3,
        xp_cards=-0.13, xp_penalty=0.0,
    )
    assert sum(fp.components().values()) == pytest.approx(4.57)
    assert set(fp.components()) == {
        "appearance", "goals", "assists", "clean_sheet", "goals_conceded",
        "saves", "defcon", "bonus", "cards", "penalty",
    }


def test_gw_projection_reports_its_fixture_count():
    gwp = PlayerGWProjection(player_id=1, gw=1)
    assert gwp.n_fixtures == 0 and gwp.is_blank is True
    gwp.fixtures.append(
        PlayerFixtureProjection(player_id=1, gw=1, fixture_id=1, opponent_id=2, is_home=True)
    )
    assert gwp.n_fixtures == 1 and gwp.is_blank is False
