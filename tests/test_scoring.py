"""Every 2026/27 scoring rule, asserted against literal values.

The point of writing the expected numbers out longhand rather than deriving
them from the module is that this file is the *second* opinion. If someone
"tidies" ``DEFCON_THRESHOLD_MID_FWD`` from 12 to 10 because the defender number
is 10, nothing else in the codebase notices — every consumer reads the constant.
This file notices.

Rules are as verified against the live API on 2026-08-14 (see the module
docstring of ``gaffer.core.scoring``).
"""
from __future__ import annotations

import math

import pytest

from gaffer.core import scoring
from gaffer.model import defcon as defcon_model

GKP, DEF, MID, FWD = scoring.GKP, scoring.DEF, scoring.MID, scoring.FWD


# ---------------------------------------------------------------------------
# Flat constants
# ---------------------------------------------------------------------------


def test_position_ids_are_the_api_element_types():
    assert (GKP, DEF, MID, FWD) == (1, 2, 3, 4)
    assert scoring.POS_NAME == {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@pytest.mark.parametrize(
    "position,points", [(GKP, 10), (DEF, 6), (MID, 5), (FWD, 4)]
)
def test_goal_points_by_position(position, points):
    assert scoring.GOAL_POINTS[position] == points
    assert scoring.goal_points(position) == points
    assert scoring.goal_points(position, 2) == 2 * points
    assert scoring.goal_points(position, 0) == 0


def test_assist_is_three_for_everyone():
    assert scoring.ASSIST_POINTS == 3
    assert scoring.assist_points(2) == 6


@pytest.mark.parametrize(
    "position,points", [(GKP, 4), (DEF, 4), (MID, 1), (FWD, 0)]
)
def test_clean_sheet_points_by_position(position, points):
    assert scoring.CLEAN_SHEET_POINTS[position] == points


@pytest.mark.parametrize(
    "position,per", [(GKP, -1), (DEF, -1), (MID, 0), (FWD, 0)]
)
def test_goals_conceded_points_by_position(position, per):
    assert scoring.GOALS_CONCEDED_POINTS[position] == per


def test_discipline_and_penalty_constants():
    assert scoring.PENALTY_SAVE == 5
    assert scoring.PENALTY_MISS == -2
    assert scoring.OWN_GOAL == -2
    assert scoring.YELLOW_CARD == -1
    assert scoring.RED_CARD == -3


def test_bonus_tiers_are_three_two_one():
    assert scoring.BONUS_TIERS == (3, 2, 1)
    assert scoring.expected_bonus_points(1.0, 0.0, 0.0) == 3
    assert scoring.expected_bonus_points(0.0, 1.0, 0.0) == 2
    assert scoring.expected_bonus_points(0.0, 0.0, 1.0) == 1
    assert scoring.expected_bonus_points(0.2, 0.3, 0.5) == pytest.approx(1.7)


def test_squad_and_transfer_rules():
    assert scoring.SQUAD_SIZE == 15
    assert scoring.SQUAD_PLAY == 11
    assert scoring.BUDGET_TENTHS == 1000  # 100.0m in tenths
    assert scoring.TEAM_LIMIT == 3
    assert scoring.SQUAD_SELECT == {GKP: 2, DEF: 5, MID: 5, FWD: 3}
    assert scoring.SQUAD_MIN_PLAY == {GKP: 1, DEF: 3, MID: 2, FWD: 1}
    assert scoring.SQUAD_MAX_PLAY == {GKP: 1, DEF: 5, MID: 5, FWD: 3}
    assert scoring.TRANSFER_HIT_COST == 4
    assert scoring.MAX_BANKED_FREE_TRANSFERS == 5
    assert scoring.SELL_ON_FEE == 0.5
    assert sum(scoring.SQUAD_SELECT.values()) == scoring.SQUAD_SIZE


def test_chip_windows_match_the_2026_27_rules():
    # Two of each chip; the first set expires at the GW19 deadline.
    assert sorted(scoring.CHIP_WINDOWS) == ["3xc", "bboost", "freehit", "wildcard"]
    for chip in ("wildcard", "freehit"):
        assert scoring.CHIP_WINDOWS[chip] == ((2, 19), (20, 38))
    for chip in ("bboost", "3xc"):
        assert scoring.CHIP_WINDOWS[chip] == ((1, 19), (20, 38))
    assert scoring.FIRST_HALF_LAST_GW == 19
    assert scoring.TOTAL_GWS == 38


def test_captain_multipliers():
    assert scoring.CAPTAIN_MULTIPLIER == 2
    assert scoring.TRIPLE_CAPTAIN_MULTIPLIER == 3


# ---------------------------------------------------------------------------
# Appearance points: the 60-minute boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "minutes,points",
    [(0, 0), (1, 1), (30, 1), (58, 1), (59, 1), (60, 2), (61, 2), (90, 2), (98, 2)],
)
def test_appearance_points_boundary_at_60(minutes, points):
    assert scoring.appearance_points(minutes) == points


def test_appearance_points_negative_minutes_score_nothing():
    assert scoring.appearance_points(-5) == 0


def test_expected_appearance_points():
    # Certain 60+: 2 points. Certain to play but never 60: 1 point.
    assert scoring.expected_appearance_points(1.0, 1.0) == 2
    assert scoring.expected_appearance_points(1.0, 0.0) == 1
    assert scoring.expected_appearance_points(0.0, 0.0) == 0
    # 90% chance of playing, 70% of reaching 60: 0.2 * 1 + 0.7 * 2.
    assert scoring.expected_appearance_points(0.9, 0.7) == pytest.approx(1.6)


def test_expected_appearance_points_clamps_p60_to_p_appear():
    """P(60+) can never exceed P(1+); an inconsistent pair must not pay 2.2."""
    assert scoring.expected_appearance_points(0.5, 0.9) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clean sheets: the 60-minute condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position,points", [(GKP, 4.0), (DEF, 4.0), (MID, 1.0), (FWD, 0.0)])
def test_clean_sheet_pays_full_value_when_certain(position, points):
    assert scoring.clean_sheet_points(position, 1.0, 1.0) == pytest.approx(points)


@pytest.mark.parametrize("position", [GKP, DEF, MID, FWD])
def test_clean_sheet_pays_nothing_without_60_minutes(position):
    """A shut-out is worth zero to a player who came off on the hour minus one."""
    assert scoring.clean_sheet_points(position, 1.0, 0.0) == 0.0


def test_clean_sheet_scales_with_both_probabilities():
    assert scoring.clean_sheet_points(DEF, 0.5, 0.8) == pytest.approx(4 * 0.5 * 0.8)


def test_clean_sheet_minimum_minutes_constant():
    assert scoring.CLEAN_SHEET_MIN_MINUTES == 60


def test_synthetic_season_never_awards_a_clean_sheet_under_60_minutes(synthetic_season):
    """End-to-end on the simulated league: the 60-minute gate really binds.

    The fixture builds its rows from the real scoring helpers, so this also
    proves there is at least one shut-out played by a sub — i.e. the assertion
    is not vacuously true.
    """
    short_shutouts = 0
    for row in synthetic_season["rows"]:
        if row["minutes"] > 0 and row["conceded"] == 0:
            if row["minutes"] < 60:
                short_shutouts += 1
                assert row["clean_sheet"] == 0
            else:
                assert row["clean_sheet"] == 1
    assert short_shutouts > 0, "no sub ever came on in a shut-out; test is vacuous"


# ---------------------------------------------------------------------------
# Saves: one point per three
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "saves,points",
    [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 2), (8, 2), (9, 3)],
)
def test_saves_pay_one_per_three(saves, points):
    """The 2 vs 3 boundary: two saves are worth nothing, the third pays."""
    assert scoring.SAVES_PER == 3
    assert scoring.SAVE_POINTS == 1
    assert scoring.SAVE_POINTS * (saves // scoring.SAVES_PER) == points


def test_expected_save_points_matches_the_exact_expectation():
    lam = 3.0
    exact = sum(scoring.poisson_pmf(k, lam) * (k // 3) for k in range(0, 200))
    assert scoring.expected_save_points(lam) == pytest.approx(exact, abs=1e-6)


def test_expected_save_points_is_zero_for_a_keeper_facing_nothing():
    assert scoring.expected_save_points(0.0) == 0.0


def test_expected_save_points_increases_with_shots_faced():
    values = [scoring.expected_save_points(lam) for lam in (0.5, 1.5, 3.0, 5.0)]
    assert values == sorted(values)
    # Three saves is one point, so E[points] must stay well under E[saves]/3 + 1.
    assert values[-1] < 5.0 / 3.0


# ---------------------------------------------------------------------------
# Goals conceded: minus one per two, floored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conceded,points",
    [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2), (5, -2), (6, -3), (7, -3)],
)
def test_goals_conceded_floor_for_a_defender(conceded, points):
    """The 1 vs 2 boundary: one goal costs nothing, the second costs a point."""
    assert scoring.GOALS_CONCEDED_PER == 2
    per = scoring.GOALS_CONCEDED_POINTS[DEF]
    assert per * (conceded // scoring.GOALS_CONCEDED_PER) == points


@pytest.mark.parametrize("position", [MID, FWD])
@pytest.mark.parametrize("conceded", [0, 2, 5, 9])
def test_midfielders_and_forwards_never_lose_points_for_goals_conceded(position, conceded):
    per = scoring.GOALS_CONCEDED_POINTS[position]
    assert per * (conceded // scoring.GOALS_CONCEDED_PER) == 0
    assert scoring.expected_goals_conceded_points(position, 2.5, 1.0) == 0.0


def test_expected_goals_conceded_matches_the_exact_expectation():
    lam = 1.4
    exact = -sum(scoring.poisson_pmf(k, lam) * (k // 2) for k in range(0, 200))
    got = scoring.expected_goals_conceded_points(DEF, lam, 1.0)
    assert got == pytest.approx(exact, abs=1e-6)
    assert got < 0.0


def test_expected_goals_conceded_is_identical_for_gkp_and_def():
    assert scoring.expected_goals_conceded_points(GKP, 1.7, 1.0) == pytest.approx(
        scoring.expected_goals_conceded_points(DEF, 1.7, 1.0)
    )


def test_expected_goals_conceded_scales_with_p_appear():
    full = scoring.expected_goals_conceded_points(DEF, 1.6, 1.0)
    half = scoring.expected_goals_conceded_points(DEF, 1.6, 0.5)
    assert half == pytest.approx(full * 0.5)
    assert scoring.expected_goals_conceded_points(DEF, 1.6, 0.0) == 0.0


def test_conceded_deduction_has_no_60_minute_gate():
    """A sub-60 appearance is still docked for goals conceded.

    FPL's 60-minute condition applies to the CLEAN SHEET only. A defender who
    comes off at half time while his team ships two still loses a point.
    Gating this on P(60+) forgave ~96 points a season across the 2025/26 rows.
    """
    # A defender certain to appear but unlikely to last 60 minutes still
    # carries the full deduction for the goals let in while he is on.
    on_pitch_45 = 45.0 / 90.0
    docked = scoring.expected_goals_conceded_points(DEF, 2.4 * on_pitch_45, 1.0)
    assert docked < 0.0, "a sub-60 appearance must still be docked"

    # And the deduction tracks appearance, not the 60-minute threshold: a
    # player who never gets on is the only one who escapes it.
    assert scoring.expected_goals_conceded_points(DEF, 2.4, 0.0) == 0.0

    # Midfielders and forwards are never docked, at any minutes.
    for position in (MID, FWD):
        assert scoring.expected_goals_conceded_points(position, 3.0, 1.0) == 0.0


def test_expected_goals_conceded_is_zero_against_a_shutout_certainty():
    assert scoring.expected_goals_conceded_points(DEF, 0.0, 1.0) == 0.0


def test_expected_goals_conceded_gets_worse_as_the_opponent_gets_better():
    values = [scoring.expected_goals_conceded_points(DEF, lam, 1.0) for lam in (0.5, 1.0, 2.0, 3.0)]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# DEFCON
# ---------------------------------------------------------------------------


def defcon_award(
    position: int,
    clearances: int = 0,
    blocks: int = 0,
    interceptions: int = 0,
    tackles: int = 0,
    recoveries: int = 0,
) -> int:
    """The published rule, written out from the module's own constants.

    Defenders count CBIT, midfielders and forwards CBIRT (which adds ball
    recoveries), keepers are ineligible. This mirrors the composition in
    ``gaffer.model.defcon.season_defcon_actions``, which is exercised directly
    further down.
    """
    actions = clearances + blocks + interceptions + tackles
    if position in (scoring.MID, scoring.FWD):
        actions += recoveries
    if actions >= scoring.DEFCON_THRESHOLD[position]:
        return scoring.DEFCON_POINTS[position]
    return 0


def test_defcon_thresholds_are_10_for_def_and_12_for_mid_and_fwd():
    assert scoring.DEFCON_THRESHOLD_DEF == 10
    assert scoring.DEFCON_THRESHOLD_MID_FWD == 12
    assert scoring.DEFCON_THRESHOLD[DEF] == 10
    assert scoring.DEFCON_THRESHOLD[MID] == 12
    assert scoring.DEFCON_THRESHOLD[FWD] == 12


def test_defcon_is_worth_two_points_and_is_capped_there():
    assert scoring.DEFCON_POINTS == {GKP: 0, DEF: 2, MID: 2, FWD: 2}
    assert max(scoring.DEFCON_POINTS.values()) == 2
    # Certain to hit the threshold is still only 2 points, however many actions.
    assert scoring.defcon_points(DEF, 1.0) == 2
    assert scoring.defcon_points(MID, 1.0) == 2
    assert defcon_award(DEF, clearances=40, tackles=40) == 2


@pytest.mark.parametrize("cbit,points", [(0, 0), (8, 0), (9, 0), (10, 2), (11, 2), (25, 2)])
def test_defender_defcon_boundary_at_10_cbit(cbit, points):
    """9 CBIT is nothing, 10 pays, 11 pays the same 2 — no accumulation."""
    assert defcon_award(DEF, clearances=cbit) == points


@pytest.mark.parametrize("position", [MID, FWD])
@pytest.mark.parametrize("cbirt,points", [(0, 0), (10, 0), (11, 0), (12, 2), (13, 2), (30, 2)])
def test_mid_and_fwd_defcon_boundary_at_12_cbirt(position, cbirt, points):
    assert defcon_award(position, clearances=cbirt) == points


def test_a_defender_on_10_cbit_scores_but_a_midfielder_on_10_does_not():
    """The two thresholds really are different numbers, not one constant."""
    assert defcon_award(DEF, tackles=10) == 2
    assert defcon_award(MID, tackles=10) == 0
    assert defcon_award(FWD, tackles=10) == 0


def test_recoveries_count_for_mid_and_fwd_but_not_for_def():
    # 9 CBIT + 20 recoveries. A defender counts CBIT only and stays on 9.
    assert defcon_award(DEF, tackles=9, recoveries=20) == 0
    # The same line for a midfielder is 29 CBIRT, comfortably over 12.
    assert defcon_award(MID, tackles=9, recoveries=20) == 2
    assert defcon_award(FWD, tackles=9, recoveries=20) == 2


def test_defcon_composition_is_clearances_blocks_interceptions_tackles():
    """CBIT is a sum, not a single stat: 3+3+2+2 = 10 pays for a defender."""
    assert defcon_award(DEF, clearances=3, blocks=3, interceptions=2, tackles=2) == 2
    assert defcon_award(DEF, clearances=3, blocks=3, interceptions=2, tackles=1) == 0


def test_goalkeepers_are_ineligible_for_defcon():
    assert scoring.DEFCON_POINTS[GKP] == 0
    assert scoring.DEFCON_THRESHOLD[GKP] > 1000  # sentinel: unreachable
    assert defcon_award(GKP, clearances=50, tackles=50, recoveries=50) == 0
    assert scoring.defcon_points(GKP, 1.0) == 0


def test_defcon_points_scale_linearly_with_the_probability():
    assert scoring.defcon_points(DEF, 0.0) == 0.0
    assert scoring.defcon_points(DEF, 0.25) == pytest.approx(0.5)
    assert scoring.defcon_points(MID, 0.5) == pytest.approx(1.0)


def test_model_season_defcon_actions_uses_cbit_for_def_and_cbirt_for_mid_fwd():
    """The production composition, from ``gaffer.model.defcon``."""
    row = {
        "minutes": 900.0,
        "clearances_blocks_interceptions": 60.0,
        "tackles": 20.0,
        "recoveries": 100.0,
        "defensive_contribution": 0.0,
    }
    def_actions, minutes, confirmed = defcon_model.season_defcon_actions(row, DEF)
    assert (def_actions, minutes, confirmed) == (80.0, 900.0, True)  # CBI + T
    for position in (MID, FWD):
        actions, _, confirmed = defcon_model.season_defcon_actions(row, position)
        assert (actions, confirmed) == (180.0, True)  # CBI + T + R


def test_model_season_defcon_actions_reports_an_unconfirmed_basis():
    """Only the stored total survives: the CBIT/CBIRT basis cannot be verified."""
    row = {"minutes": 900.0, "defensive_contribution": 150.0}
    actions, minutes, confirmed = defcon_model.season_defcon_actions(row, DEF)
    assert (actions, minutes, confirmed) == (150.0, 900.0, False)


def test_model_season_defcon_actions_returns_none_when_there_is_no_signal():
    """None is not zero: a pre-2024/25 season has no defensive data at all."""
    assert defcon_model.season_defcon_actions({"minutes": 3000.0}, DEF) is None
    assert defcon_model.season_defcon_actions({"minutes": 0.0, "tackles": 5.0}, DEF) is None


def test_synthetic_season_defcon_matches_the_position_thresholds(synthetic_season):
    """The simulated league agrees with the rule, position by position."""
    seen = {DEF: [0, 0], MID: [0, 0], FWD: [0, 0]}
    for row in synthetic_season["rows"]:
        position = row["player"]["position"]
        if row["minutes"] == 0:
            continue
        expected = row["cbi"] + row["tackles"]
        if position in (MID, FWD):
            expected += row["recoveries"]
        assert row["defcon"] == expected
        if position == GKP:
            continue
        hit = row["defcon"] >= scoring.DEFCON_THRESHOLD[position]
        seen[position][int(hit)] += 1
    for position, (misses, hits) in seen.items():
        assert hits > 0 and misses > 0, "position %d never straddled the threshold" % position


# ---------------------------------------------------------------------------
# Selling price
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bought,now,sell",
    [
        (50, 50, 50),  # no change
        (50, 45, 45),  # a fall is absorbed in full
        (50, 51, 50),  # 0.1m rise: you keep nothing, half of 1 tenth floors to 0
        (50, 52, 51),  # 0.2m rise: keep 0.1m
        (50, 55, 52),  # 0.5m rise: keep 0.2m (rounded down)
        (100, 111, 105),
        (145, 160, 152),
    ],
)
def test_selling_price_keeps_half_the_profit_rounded_down(bought, now, sell):
    assert scoring.selling_price(bought, now) == sell


def test_selling_price_never_exceeds_the_current_price():
    for bought in range(38, 150, 7):
        for now in range(38, 150, 5):
            assert scoring.selling_price(bought, now) <= now


# ---------------------------------------------------------------------------
# Legality helpers
# ---------------------------------------------------------------------------


LEGAL_POSITIONS = [GKP] * 2 + [DEF] * 5 + [MID] * 5 + [FWD] * 3
LEGAL_TEAMS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]


def test_squad_is_legal_accepts_a_legal_fifteen():
    assert scoring.squad_is_legal(LEGAL_POSITIONS, LEGAL_TEAMS) is True


def test_squad_is_legal_rejects_a_wrong_size():
    assert scoring.squad_is_legal(LEGAL_POSITIONS[:14], LEGAL_TEAMS[:14]) is False


def test_squad_is_legal_rejects_a_broken_quota():
    broken = [GKP] * 3 + [DEF] * 4 + [MID] * 5 + [FWD] * 3
    assert scoring.squad_is_legal(broken, LEGAL_TEAMS) is False


def test_squad_is_legal_rejects_four_from_one_club():
    teams = [1, 1, 1, 1] + LEGAL_TEAMS[4:]
    assert scoring.squad_is_legal(LEGAL_POSITIONS, teams) is False


def test_squad_is_legal_accepts_exactly_three_from_one_club():
    teams = [1, 1, 1] + LEGAL_TEAMS[3:]
    assert scoring.squad_is_legal(LEGAL_POSITIONS, teams) is True


@pytest.mark.parametrize(
    "formation",
    [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 2, 3), (5, 3, 2), (5, 4, 1)],
)
def test_lineup_is_legal_accepts_every_real_formation(formation):
    d, m, f = formation
    assert d + m + f == 10
    positions = [GKP] + [DEF] * d + [MID] * m + [FWD] * f
    assert scoring.lineup_is_legal(positions) is True


@pytest.mark.parametrize(
    "positions",
    [
        [GKP] * 2 + [DEF] * 4 + [MID] * 4 + [FWD] * 1,  # two keepers
        [GKP] + [DEF] * 2 + [MID] * 5 + [FWD] * 3,  # only 2 defenders
        [GKP] + [DEF] * 6 + [MID] * 3 + [FWD] * 1,  # six defenders
        [GKP] + [DEF] * 5 + [MID] * 1 + [FWD] * 4,  # four forwards, one midfielder
        [GKP] + [DEF] * 4 + [MID] * 4 + [FWD] * 1,  # only 10 players
    ],
)
def test_lineup_is_legal_rejects_illegal_shapes(positions):
    assert scoring.lineup_is_legal(positions) is False


# ---------------------------------------------------------------------------
# Poisson helper inside scoring
# ---------------------------------------------------------------------------


def test_scoring_poisson_pmf_matches_the_closed_form():
    lam = 1.7
    for k in range(0, 8):
        expected = math.exp(-lam) * lam ** k / math.factorial(k)
        assert scoring.poisson_pmf(k, lam) == pytest.approx(expected, rel=1e-12)


def test_scoring_poisson_pmf_at_zero_rate():
    assert scoring.poisson_pmf(0, 0.0) == 1.0
    assert scoring.poisson_pmf(3, 0.0) == 0.0


def test_scoring_poisson_pmf_sums_to_one():
    assert sum(scoring.poisson_pmf(k, 2.5) for k in range(0, 40)) == pytest.approx(1.0, abs=1e-12)
