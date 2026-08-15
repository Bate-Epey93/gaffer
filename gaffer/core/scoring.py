"""FPL 2026/27 scoring rules.

Every constant here was read from the live official API
(https://fantasy.premierleague.com/api/bootstrap-static/ -> game_config.scoring)
on 2026-08-14. Do not edit by hand; run `python -m gaffer.core.scoring --verify`
to re-check against the API.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

# --- Positions -------------------------------------------------------------
GKP, DEF, MID, FWD = 1, 2, 3, 4
POSITIONS: Tuple[int, ...] = (GKP, DEF, MID, FWD)
POS_NAME: Dict[int, str] = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}
POS_ID: Dict[str, int] = {v: k for k, v in POS_NAME.items()}

# --- Scoring constants (verified) ------------------------------------------
SHORT_PLAY = 1  # 1-59 minutes
LONG_PLAY = 2  # 60+ minutes

GOAL_POINTS: Dict[int, int] = {GKP: 10, DEF: 6, MID: 5, FWD: 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS: Dict[int, int] = {GKP: 4, DEF: 4, MID: 1, FWD: 0}
GOALS_CONCEDED_POINTS: Dict[int, int] = {GKP: -1, DEF: -1, MID: 0, FWD: 0}
GOALS_CONCEDED_PER = 2  # -1 point per 2 conceded
SAVE_POINTS = 1
SAVES_PER = 3  # 1 point per 3 saves
PENALTY_SAVE = 5
PENALTY_MISS = -2
OWN_GOAL = -2
YELLOW_CARD = -1
RED_CARD = -3
BONUS_TIERS: Tuple[int, int, int] = (3, 2, 1)

# Defensive Contribution (DEFCON), introduced 2025/26, unchanged for 2026/27.
DEFCON_POINTS: Dict[int, int] = {GKP: 0, DEF: 2, MID: 2, FWD: 2}
# Defenders count CBIT: clearances + blocks + interceptions + tackles.
DEFCON_THRESHOLD_DEF = 10
# Midfielders and forwards count CBIRT: CBIT + ball recoveries.
DEFCON_THRESHOLD_MID_FWD = 12
DEFCON_THRESHOLD: Dict[int, int] = {
    GKP: 10 ** 9,  # ineligible
    DEF: DEFCON_THRESHOLD_DEF,
    MID: DEFCON_THRESHOLD_MID_FWD,
    FWD: DEFCON_THRESHOLD_MID_FWD,
}

CLEAN_SHEET_MIN_MINUTES = 60

# --- Squad / transfer rules (verified) -------------------------------------
SQUAD_SIZE = 15
SQUAD_PLAY = 11
BUDGET_TENTHS = 1000  # £100.0m, stored in tenths of a million
TEAM_LIMIT = 3
SQUAD_SELECT: Dict[int, int] = {GKP: 2, DEF: 5, MID: 5, FWD: 3}
SQUAD_MIN_PLAY: Dict[int, int] = {GKP: 1, DEF: 3, MID: 2, FWD: 1}
SQUAD_MAX_PLAY: Dict[int, int] = {GKP: 1, DEF: 5, MID: 5, FWD: 3}
TRANSFER_HIT_COST = 4
MAX_BANKED_FREE_TRANSFERS = 5  # 1 base + max_extra_free_transfers (4)
TRANSFERS_CAP = 20
SELL_ON_FEE = 0.5  # keep 50% of profit, rounded down to 0.1m

CAPTAIN_MULTIPLIER = 2
TRIPLE_CAPTAIN_MULTIPLIER = 3

# Chip windows, from the API chips table.
CHIP_WINDOWS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "wildcard": ((2, 19), (20, 38)),
    "freehit": ((2, 19), (20, 38)),
    "bboost": ((1, 19), (20, 38)),
    "3xc": ((1, 19), (20, 38)),
}

SEASON = "2026/27"
FIRST_HALF_LAST_GW = 19
TOTAL_GWS = 38


# --- Point helpers ---------------------------------------------------------
def appearance_points(minutes: float) -> int:
    """Points purely for playing time."""
    if minutes <= 0:
        return 0
    return LONG_PLAY if minutes >= 60 else SHORT_PLAY


def expected_appearance_points(p_appear: float, p_60: float) -> float:
    """p_appear = P(minutes >= 1); p_60 = P(minutes >= 60). p_60 <= p_appear."""
    p_60 = min(p_60, p_appear)
    return SHORT_PLAY * (p_appear - p_60) + LONG_PLAY * p_60


def goal_points(position: int, goals: float = 1.0) -> float:
    return GOAL_POINTS[position] * goals


def assist_points(assists: float = 1.0) -> float:
    return ASSIST_POINTS * assists


def clean_sheet_points(position: int, p_clean_sheet: float, p_60: float) -> float:
    """Clean sheet only pays if the player is on the pitch for 60+ minutes."""
    return CLEAN_SHEET_POINTS[position] * p_clean_sheet * p_60


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def expected_goals_conceded_points(
    position: int, lam_conceded: float, p_appear: float, max_goals: int = 12
) -> float:
    """E[-1 * floor(conceded / 2)] under Poisson(lam_conceded), scaled by P(appear).

    FPL docks conceded points for goals let in while the player is on the pitch,
    with NO minutes threshold — the 60-minute condition applies to the clean
    sheet only. So this is gated on appearing at all, and the caller is expected
    to have already scaled ``lam_conceded`` by the fraction of the match the
    player is on for. Gating it on P(60+) instead silently forgives every sub-60
    appearance: measured against the real 2025/26 rows that was 96 points a
    season of deductions that never landed.
    """
    per = GOALS_CONCEDED_POINTS[position]
    if per == 0:
        return 0.0
    total = 0.0
    for k in range(0, max_goals + 1):
        total += poisson_pmf(k, lam_conceded) * (k // GOALS_CONCEDED_PER)
    return per * total * p_appear


def expected_save_points(lam_saves: float, max_saves: int = 20) -> float:
    """E[floor(saves / 3)] under Poisson(lam_saves)."""
    if lam_saves <= 0:
        return 0.0
    total = 0.0
    for k in range(0, max_saves + 1):
        total += poisson_pmf(k, lam_saves) * (k // SAVES_PER)
    return SAVE_POINTS * total


def defcon_points(position: int, p_threshold: float) -> float:
    """DEFCON is a flat 2 points once the threshold is hit; no accumulation."""
    return DEFCON_POINTS[position] * p_threshold


def expected_bonus_points(p_bonus3: float, p_bonus2: float, p_bonus1: float) -> float:
    return 3 * p_bonus3 + 2 * p_bonus2 + 1 * p_bonus1


def expected_card_points(p_yellow: float, p_red: float) -> float:
    return YELLOW_CARD * p_yellow + RED_CARD * p_red


def selling_price(purchase_price: int, current_price: int) -> int:
    """FPL sell-on rule, all prices in tenths of a million.

    You keep 50% of any rise, rounded down to the nearest 0.1m. Falls are
    absorbed in full by the manager.
    """
    if current_price <= purchase_price:
        return current_price
    profit = current_price - purchase_price
    return purchase_price + profit // 2


def squad_is_legal(positions: Iterable[int], team_ids: Iterable[int]) -> bool:
    positions = list(positions)
    team_ids = list(team_ids)
    if len(positions) != SQUAD_SIZE:
        return False
    for pos, need in SQUAD_SELECT.items():
        if positions.count(pos) != need:
            return False
    for team in set(team_ids):
        if team_ids.count(team) > TEAM_LIMIT:
            return False
    return True


def lineup_is_legal(positions: Iterable[int]) -> bool:
    positions = list(positions)
    if len(positions) != SQUAD_PLAY:
        return False
    for pos in POSITIONS:
        count = positions.count(pos)
        if count < SQUAD_MIN_PLAY[pos] or count > SQUAD_MAX_PLAY[pos]:
            return False
    return True


def _verify_against_api() -> int:
    """Re-check every constant against the live API. Returns count of mismatches."""
    import json
    import urllib.request

    req = urllib.request.Request(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        headers={"User-Agent": "gaffer/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    scoring = data["game_config"]["scoring"]
    rules = data["game_config"]["rules"]
    problems = []

    def check(label, got, want):
        if got != want:
            problems.append("%s: api=%r local=%r" % (label, got, want))

    check("short_play", scoring["short_play"], SHORT_PLAY)
    check("long_play", scoring["long_play"], LONG_PLAY)
    check("assists", scoring["assists"], ASSIST_POINTS)
    check("saves", scoring["saves"], SAVE_POINTS)
    check("penalties_saved", scoring["penalties_saved"], PENALTY_SAVE)
    check("penalties_missed", scoring["penalties_missed"], PENALTY_MISS)
    check("own_goals", scoring["own_goals"], OWN_GOAL)
    check("yellow_cards", scoring["yellow_cards"], YELLOW_CARD)
    check("red_cards", scoring["red_cards"], RED_CARD)
    for pos in POSITIONS:
        name = POS_NAME[pos]
        check("goals[%s]" % name, scoring["goals_scored"][name], GOAL_POINTS[pos])
        check("cs[%s]" % name, scoring["clean_sheets"][name], CLEAN_SHEET_POINTS[pos])
        check("gc[%s]" % name, scoring["goals_conceded"][name], GOALS_CONCEDED_POINTS[pos])
        check(
            "defcon[%s]" % name,
            scoring["defensive_contribution"][name],
            DEFCON_POINTS[pos],
        )
    check("squad_size", rules["squad_squadsize"], SQUAD_SIZE)
    check("squad_play", rules["squad_squadplay"], SQUAD_PLAY)
    check("budget", rules["squad_total_spend"], BUDGET_TENTHS)
    check("team_limit", rules["squad_team_limit"], TEAM_LIMIT)
    check("sell_on_fee", rules["transfers_sell_on_fee"], SELL_ON_FEE)
    check(
        "max_banked_ft",
        rules["max_extra_free_transfers"] + 1,
        MAX_BANKED_FREE_TRANSFERS,
    )
    for et in data["element_types"]:
        pos = et["id"]
        check("select[%s]" % POS_NAME[pos], et["squad_select"], SQUAD_SELECT[pos])
        check("min_play[%s]" % POS_NAME[pos], et["squad_min_play"], SQUAD_MIN_PLAY[pos])
        check("max_play[%s]" % POS_NAME[pos], et["squad_max_play"], SQUAD_MAX_PLAY[pos])

    if problems:
        print("MISMATCHES (%d):" % len(problems))
        for p in problems:
            print("  " + p)
    else:
        print("All %s scoring constants match the live API." % SEASON)
    return len(problems)


if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        sys.exit(1 if _verify_against_api() else 0)
    print("FPL %s scoring module. Run with --verify to check against the API." % SEASON)
