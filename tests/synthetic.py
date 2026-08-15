"""A small, internally consistent fake Premier League.

Unit tests must not touch the network, but almost every model in gaffer fits on
last season's football-data results and last season's vaastav archive. Stubbing
those models out would leave the interesting behaviour untested, so instead the
data they read is fabricated here: a full 20-team, 380-fixture season with
per-player match rows in exactly the schemas ``gaffer.data.history`` parses.

The fabrication is a miniature FPL: scorelines come from the teams' strengths,
goals and assists are handed to players in proportion to their attacking share,
and **points are then computed with the real ``gaffer.core.scoring`` helpers**.
That last part is what makes the fixture worth having — a clean sheet is worth
what the rules say it is worth, so a model fitted on this data behaves the way
it would on the real thing, and a test that asserts an invariant is asserting it
against consistent inputs rather than noise.

Everything is seeded. Two calls with the same seed produce byte-identical CSVs.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from gaffer.core import scoring
from gaffer.core.types import Fixture, Player, Team
from gaffer.data import history as hist
from gaffer.data.cache import Cache
from gaffer.data.loaders import GameState

# (football-data name, FPL long name). Both vocabularies must resolve through
# the real mapping tables or the fixture is not exercising the real code path.
CLUBS: Tuple[Tuple[str, str], ...] = (
    ("Arsenal", "Arsenal"),
    ("Aston Villa", "Aston Villa"),
    ("Bournemouth", "Bournemouth"),
    ("Brentford", "Brentford"),
    ("Brighton", "Brighton"),
    ("Burnley", "Burnley"),
    ("Chelsea", "Chelsea"),
    ("Crystal Palace", "Crystal Palace"),
    ("Everton", "Everton"),
    ("Fulham", "Fulham"),
    ("Leeds", "Leeds"),
    ("Liverpool", "Liverpool"),
    ("Man City", "Man City"),
    ("Man United", "Man Utd"),
    ("Newcastle", "Newcastle"),
    ("Nott'm Forest", "Nott'm Forest"),
    ("Sunderland", "Sunderland"),
    ("Tottenham", "Spurs"),
    ("West Ham", "West Ham"),
    ("Wolves", "Wolves"),
)

# Squad shape per club: enough at each position for a legal 15 with the
# three-per-club limit, and enough depth for rotation to be visible.
SQUAD_SHAPE: Tuple[Tuple[int, int], ...] = (
    (scoring.GKP, 2), (scoring.DEF, 6), (scoring.MID, 6), (scoring.FWD, 3),
)
PLAYERS_PER_CLUB = sum(n for _, n in SQUAD_SHAPE)

SEASON_START = datetime(2025, 8, 15, 19, 0, tzinfo=timezone.utc)
GWS = 38

# Price ladders per position, best player first, in tenths.
PRICE_LADDER: Dict[int, Tuple[int, ...]] = {
    scoring.GKP: (55, 42),
    scoring.DEF: (62, 55, 50, 45, 42, 40),
    scoring.MID: (125, 88, 72, 60, 50, 45),
    scoring.FWD: (105, 70, 50),
}

for _fd_name, _fpl_name in CLUBS:
    assert _fd_name in hist.FOOTBALL_DATA_TO_FPL_SHORT, _fd_name
    assert _fpl_name in hist.FPL_LONG_TO_SHORT, _fpl_name
    assert hist.FOOTBALL_DATA_TO_FPL_SHORT[_fd_name] == hist.FPL_LONG_TO_SHORT[_fpl_name]


def short_name(index: int) -> str:
    return hist.FOOTBALL_DATA_TO_FPL_SHORT[CLUBS[index][0]]


def team_strength(index: int) -> Tuple[float, float]:
    """(attack, defence) multipliers. Club 0 is the best, club 19 the worst."""
    attack = 1.45 - 0.05 * index
    defence = 0.65 + 0.045 * index
    return attack, defence


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def round_robin(n_teams: int = 20) -> List[List[Tuple[int, int]]]:
    """Circle-method double round robin: 2*(n-1) rounds, everyone plays once."""
    teams = list(range(n_teams))
    fixed, rotating = teams[0], teams[1:]
    first_half: List[List[Tuple[int, int]]] = []
    for r in range(n_teams - 1):
        pairs = [(fixed, rotating[0]) if r % 2 == 0 else (rotating[0], fixed)]
        for i in range(1, n_teams // 2):
            a, b = rotating[i], rotating[-i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        first_half.append(pairs)
        rotating = rotating[1:] + rotating[:1]
    second_half = [[(a, h) for h, a in rnd] for rnd in first_half]
    return first_half + second_half


def fixture_list() -> List[Dict[str, Any]]:
    """380 fixtures with ids, gameweeks and kickoff times."""
    out: List[Dict[str, Any]] = []
    fixture_id = 1
    for gw, pairs in enumerate(round_robin(len(CLUBS)), start=1):
        base = SEASON_START + timedelta(days=7 * (gw - 1))
        for slot, (home, away) in enumerate(pairs):
            out.append({
                "id": fixture_id,
                "gw": gw,
                "home": home,
                "away": away,
                "kickoff": base + timedelta(hours=slot),
            })
            fixture_id += 1
    return out


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


def player_table() -> List[Dict[str, Any]]:
    """One row per player: id, code, club, position, price, role weights."""
    rows: List[Dict[str, Any]] = []
    player_id = 1
    for club_index, (fd_name, fpl_name) in enumerate(CLUBS):
        for position, count in SQUAD_SHAPE:
            for depth in range(count):
                price = PRICE_LADDER[position][depth]
                # Nailedness falls with squad depth; the best-paid player at a
                # club is the one who always starts, which is what the minutes
                # model's price prior is supposed to learn.
                nailed = max(0.05, 0.95 - 0.16 * depth)
                attack_share = {
                    scoring.GKP: 0.0,
                    scoring.DEF: 0.35,
                    scoring.MID: 1.0,
                    scoring.FWD: 1.6,
                }[position] * (1.0 - 0.12 * depth)
                rows.append({
                    "id": player_id,
                    "code": 100000 + player_id,
                    "club": club_index,
                    "team_name": fpl_name,
                    "fd_team": fd_name,
                    "short": hist.FPL_LONG_TO_SHORT[fpl_name],
                    "position": position,
                    "depth": depth,
                    "price": price,
                    "nailed": nailed,
                    "attack_share": max(attack_share, 0.0),
                    "web_name": "%s%d" % (hist.FPL_LONG_TO_SHORT[fpl_name], player_id),
                    "first_name": "Player",
                    "second_name": "%s%d" % (hist.FPL_LONG_TO_SHORT[fpl_name], player_id),
                })
                player_id += 1
    return rows


VAASTAV_POSITION_NAME: Dict[int, str] = {
    scoring.GKP: "GK", scoring.DEF: "DEF", scoring.MID: "MID", scoring.FWD: "FWD",
}


# ---------------------------------------------------------------------------
# One simulated season
# ---------------------------------------------------------------------------


def _score(rng: random.Random, lam: float, cap: int = 6) -> int:
    """Small Poisson draw by inversion; deterministic given the rng."""
    target = rng.random()
    cumulative = 0.0
    for k in range(cap + 1):
        cumulative += scoring.poisson_pmf(k, lam)
        if target <= cumulative:
            return k
    return cap


def simulate_season(seed: int = 20250815) -> Dict[str, Any]:
    """Play the whole season and return fixtures, results and player-match rows."""
    rng = random.Random(seed)
    players = player_table()
    by_club: Dict[int, List[Dict[str, Any]]] = {}
    for row in players:
        by_club.setdefault(row["club"], []).append(row)

    fixtures = fixture_list()
    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for fixture in fixtures:
        home, away = fixture["home"], fixture["away"]
        att_h, def_h = team_strength(home)
        att_a, def_a = team_strength(away)
        lam_h = 1.45 * att_h * def_a * 1.13
        lam_a = 1.45 * att_a * def_h
        goals = {home: _score(rng, lam_h), away: _score(rng, lam_a)}
        shots = {home: goals[home] * 3 + 5, away: goals[away] * 3 + 4}
        sot = {home: goals[home] + 3, away: goals[away] + 2}

        match_rows: List[Dict[str, Any]] = []
        for side, opponent in ((home, away), (away, home)):
            squad = by_club[side]
            # Who starts: the eleven with the highest nailedness plus noise, so
            # rotation happens but the nailed players nearly always play.
            ranked = sorted(squad, key=lambda p: -(p["nailed"] + rng.random() * 0.35))
            starters = ranked[:11]
            subs = ranked[11:14]
            conceded = goals[opponent]
            scorers = [p for p in starters if p["attack_share"] > 0]
            weights = [p["attack_share"] for p in scorers]
            for player in squad:
                if player in starters:
                    minutes = 90 if rng.random() < 0.75 else rng.randint(60, 89)
                    started = 1
                elif player in subs:
                    minutes = rng.randint(1, 30)
                    started = 0
                else:
                    minutes = 0
                    started = 0
                match_rows.append({
                    "player": player, "side": side, "opponent": opponent,
                    "minutes": minutes, "starts": started, "conceded": conceded,
                    "goals": 0, "assists": 0,
                })
            # Goals and assists go to players who were on the pitch.
            on_pitch = [r for r in match_rows if r["side"] == side and r["minutes"] > 0]
            eligible = [r for r in on_pitch if r["player"]["attack_share"] > 0]
            for _ in range(goals[side]):
                if not eligible:
                    break
                scorer = rng.choices(
                    eligible, weights=[r["player"]["attack_share"] for r in eligible])[0]
                scorer["goals"] += 1
                if rng.random() < 0.7:
                    others = [r for r in eligible if r is not scorer]
                    if others:
                        rng.choice(others)["assists"] += 1
            del scorers, weights

        for record in match_rows:
            player = record["player"]
            position = player["position"]
            minutes = record["minutes"]
            side_goals = goals[record["side"]]
            conceded = record["conceded"]
            played60 = minutes >= scoring.CLEAN_SHEET_MIN_MINUTES
            clean_sheet = 1 if (conceded == 0 and played60) else 0
            saves = _score(rng, 3.0, 9) if (position == scoring.GKP and minutes > 0) else 0
            cbi = _score(rng, {scoring.GKP: 0.5, scoring.DEF: 6.0,
                               scoring.MID: 2.5, scoring.FWD: 1.0}[position], 14) \
                if minutes > 0 else 0
            tackles = _score(rng, 1.6, 8) if minutes > 0 else 0
            recoveries = _score(rng, 4.5, 14) if minutes > 0 else 0
            yellow = 1 if (minutes > 0 and rng.random() < 0.10) else 0
            defcon_actions = cbi + tackles + (
                recoveries if position in (scoring.MID, scoring.FWD) else 0)
            hit_defcon = defcon_actions >= scoring.DEFCON_THRESHOLD[position]

            points = scoring.appearance_points(minutes)
            points += scoring.GOAL_POINTS[position] * record["goals"]
            points += scoring.ASSIST_POINTS * record["assists"]
            points += scoring.CLEAN_SHEET_POINTS[position] * clean_sheet
            if scoring.GOALS_CONCEDED_POINTS[position] and minutes > 0:
                points += scoring.GOALS_CONCEDED_POINTS[position] * (
                    conceded // scoring.GOALS_CONCEDED_PER)
            points += saves // scoring.SAVES_PER
            if hit_defcon and position != scoring.GKP:
                points += scoring.DEFCON_POINTS[position]
            points += scoring.YELLOW_CARD * yellow

            bps = (int(minutes >= 60) * 6 + int(0 < minutes < 60) * 3
                   + record["goals"] * 24 + record["assists"] * 9
                   + clean_sheet * 12 + saves * 2 + (cbi + tackles) // 3
                   + recoveries // 6 - yellow * 3)
            record.update({
                "clean_sheet": clean_sheet, "saves": saves, "cbi": cbi,
                "tackles": tackles, "recoveries": recoveries, "yellow": yellow,
                "defcon": defcon_actions, "bps": max(bps, 0), "points": points,
                "side_goals": side_goals,
            })

        # Bonus: 3/2/1 on BPS across the whole match, ties broken by player id.
        ordered = sorted(match_rows, key=lambda r: (-r["bps"], r["player"]["id"]))
        for rank, record in enumerate(ordered):
            record["bonus"] = scoring.BONUS_TIERS[rank] if rank < 3 else 0
            record["points"] += record["bonus"]

        results.append({
            "fixture": fixture, "home_goals": goals[home], "away_goals": goals[away],
            "home_shots": shots[home], "away_shots": shots[away],
            "home_sot": sot[home], "away_sot": sot[away],
        })
        for record in match_rows:
            rows.append({"fixture": fixture, **record})

    return {"players": players, "fixtures": fixtures, "results": results, "rows": rows}


# ---------------------------------------------------------------------------
# Serialisation into the two archive formats
# ---------------------------------------------------------------------------


def football_data_csv(season: Dict[str, Any], seed_offset: int = 0) -> str:
    """football-data.co.uk E0 format, including the odds columns."""
    rng = random.Random(4242 + seed_offset)
    out: List[Dict[str, Any]] = []
    for result in season["results"]:
        fixture = result["fixture"]
        home, away = CLUBS[fixture["home"]][0], CLUBS[fixture["away"]][0]
        hg, ag = result["home_goals"], result["away_goals"]
        out.append({
            "Div": "E0",
            "Date": fixture["kickoff"].strftime("%d/%m/%Y"),
            "Time": fixture["kickoff"].strftime("%H:%M"),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": hg, "FTAG": ag,
            "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
            "HS": result["home_shots"], "AS": result["away_shots"],
            "HST": result["home_sot"], "AST": result["away_sot"],
            "HC": 5, "AC": 4, "HF": 10, "AF": 11, "HY": 1, "AY": 2, "HR": 0, "AR": 0,
            # Prices that roughly encode the same strengths, with noise so the
            # odds-blending path has something non-degenerate to work with.
            "AvgH": round(1.6 + 0.12 * fixture["home"] + rng.random() * 0.3, 2),
            "AvgD": round(3.4 + rng.random() * 0.4, 2),
            "AvgA": round(1.7 + 0.12 * fixture["away"] + rng.random() * 0.3, 2),
            "Avg>2.5": round(1.75 + rng.random() * 0.3, 2),
            "Avg<2.5": round(2.05 + rng.random() * 0.3, 2),
        })
    return pd.DataFrame(out).to_csv(index=False)


def merged_gw_csv(season: Dict[str, Any]) -> str:
    """vaastav ``merged_gw.csv`` format."""
    out: List[Dict[str, Any]] = []
    for row in season["rows"]:
        player = row["player"]
        fixture = row["fixture"]
        was_home = row["side"] == fixture["home"]
        out.append({
            "name": "%s %s" % (player["first_name"], player["second_name"]),
            "position": VAASTAV_POSITION_NAME[player["position"]],
            "team": player["team_name"],
            "xP": 0.0,
            "assists": row["assists"],
            "bonus": row["bonus"],
            "bps": row["bps"],
            "clean_sheets": row["clean_sheet"],
            "creativity": round(row["assists"] * 20.0, 1),
            "element": player["id"],
            "expected_assists": round(row["assists"] * 0.8 + 0.02 * row["minutes"] / 90.0, 2),
            "expected_goal_involvements": round(
                (row["goals"] + row["assists"]) * 0.8, 2),
            "expected_goals": round(row["goals"] * 0.8 + 0.03 * row["minutes"] / 90.0, 2),
            "expected_goals_conceded": round(row["conceded"] * 0.9, 2),
            "fixture": fixture["id"],
            "goals_conceded": row["conceded"] if row["minutes"] > 0 else 0,
            "goals_scored": row["goals"],
            "ict_index": 1.0,
            "influence": 10.0,
            "kickoff_time": fixture["kickoff"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "minutes": row["minutes"],
            "modified": False,
            "opponent_team": row["opponent"] + 1,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "red_cards": 0,
            "round": fixture["gw"],
            "saves": row["saves"],
            "selected": int(1000 + player["price"] * 40),
            "starts": row["starts"],
            "team_a_score": (row["conceded"] if was_home else row["side_goals"]),
            "team_h_score": (row["side_goals"] if was_home else row["conceded"]),
            "threat": 5.0,
            "total_points": row["points"],
            "transfers_balance": 0,
            "transfers_in": 0,
            "transfers_out": 0,
            "value": player["price"],
            "was_home": was_home,
            "yellow_cards": row["yellow"],
            "clearances_blocks_interceptions": row["cbi"],
            "defensive_contribution": row["defcon"],
            "recoveries": row["recoveries"],
            "tackles": row["tackles"],
            "GW": fixture["gw"],
        })
    return pd.DataFrame(out).to_csv(index=False)


def players_raw_csv(season: Dict[str, Any]) -> str:
    """vaastav ``players_raw.csv``: the end-of-season element rows."""
    totals: Dict[int, Dict[str, float]] = {}
    for row in season["rows"]:
        acc = totals.setdefault(row["player"]["id"], {
            "minutes": 0.0, "total_points": 0.0, "goals_scored": 0.0, "assists": 0.0,
            "starts": 0.0, "saves": 0.0, "bonus": 0.0, "bps": 0.0,
        })
        acc["minutes"] += row["minutes"]
        acc["total_points"] += row["points"]
        acc["goals_scored"] += row["goals"]
        acc["assists"] += row["assists"]
        acc["starts"] += row["starts"]
        acc["saves"] += row["saves"]
        acc["bonus"] += row["bonus"]
        acc["bps"] += row["bps"]
    out = []
    for player in season["players"]:
        acc = totals.get(player["id"], {})
        out.append({
            "id": player["id"], "code": player["code"],
            "first_name": player["first_name"], "second_name": player["second_name"],
            "web_name": player["web_name"], "team": player["club"] + 1,
            "element_type": player["position"], "now_cost": player["price"],
            "minutes": acc.get("minutes", 0), "total_points": acc.get("total_points", 0),
            "goals_scored": acc.get("goals_scored", 0), "assists": acc.get("assists", 0),
            "starts": acc.get("starts", 0), "saves": acc.get("saves", 0),
            "bonus": acc.get("bonus", 0), "bps": acc.get("bps", 0),
            "penalties_order": "", "selected_by_percent": 1.0, "status": "a",
        })
    return pd.DataFrame(out).to_csv(index=False)


def seed_cache(cache: Cache, season: Dict[str, Any],
               fd_seasons: Sequence[str] = ("2023-24", "2024-25", "2025-26"),
               vaastav_seasons: Sequence[str] = ("2025-26",)) -> None:
    """Write the fabricated archives into a cache so the models find them.

    ``history._cached_csv`` stores raw CSV text under a per-source key, so
    seeding the cache is exactly equivalent to having downloaded the files —
    the parsing, team-name mapping and column normalisation all still run.
    """
    for offset, season_name in enumerate(fd_seasons):
        cache.set("football_data_%s" % hist.fd_season_code(season_name),
                  football_data_csv(season, seed_offset=offset))
    merged = merged_gw_csv(season)
    raw = players_raw_csv(season)
    for season_name in vaastav_seasons:
        cache.set("vaastav_%s_merged_gw" % hist.season_dash(season_name), merged)
        cache.set("vaastav_%s_players_raw" % hist.season_dash(season_name), raw)


# ---------------------------------------------------------------------------
# A GameState for the season about to be played
# ---------------------------------------------------------------------------


def history_past_rows(season: Dict[str, Any], season_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Prior-season totals per player id, in ``history_past`` shape."""
    out: Dict[int, Dict[str, Any]] = {}
    for row in season["rows"]:
        pid = row["player"]["id"]
        record = out.setdefault(pid, {
            "season_name": hist.season_slash(season_name),
            "element_code": row["player"]["code"],
            "start_cost": row["player"]["price"], "end_cost": row["player"]["price"],
            "minutes": 0, "starts": 0, "total_points": 0, "goals_scored": 0,
            "assists": 0, "clean_sheets": 0, "goals_conceded": 0, "own_goals": 0,
            "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
            "red_cards": 0, "saves": 0, "bonus": 0, "bps": 0,
            "expected_goals": 0.0, "expected_assists": 0.0,
            "expected_goal_involvements": 0.0, "expected_goals_conceded": 0.0,
            "clearances_blocks_interceptions": 0, "recoveries": 0, "tackles": 0,
            "defensive_contribution": 0,
        })
        record["minutes"] += row["minutes"]
        record["starts"] += row["starts"]
        record["total_points"] += row["points"]
        record["goals_scored"] += row["goals"]
        record["assists"] += row["assists"]
        record["clean_sheets"] += row["clean_sheet"]
        record["goals_conceded"] += row["conceded"] if row["minutes"] > 0 else 0
        record["yellow_cards"] += row["yellow"]
        record["saves"] += row["saves"]
        record["bonus"] += row["bonus"]
        record["bps"] += row["bps"]
        record["expected_goals"] += row["goals"] * 0.8
        record["expected_assists"] += row["assists"] * 0.8
        record["expected_goal_involvements"] += (row["goals"] + row["assists"]) * 0.8
        record["expected_goals_conceded"] += row["conceded"] * 0.9
        record["clearances_blocks_interceptions"] += row["cbi"]
        record["recoveries"] += row["recoveries"]
        record["tackles"] += row["tackles"]
        record["defensive_contribution"] += row["defcon"]
    return {pid: [record] for pid, record in out.items()}


def make_game_state(
    season: Optional[Dict[str, Any]] = None,
    current_gw: int = 1,
    season_name: str = "2026-27",
    prior_season_name: str = "2025-26",
) -> GameState:
    """The pre-season state for the *next* season, built off the simulated one.

    Mirrors the real GW1 situation: no current-season history at all, prior
    seasons in ``history_past``, and bootstrap element totals that still carry
    last season's numbers.
    """
    season = season or simulate_season()
    teams = {}
    for index, (fd_name, fpl_name) in enumerate(CLUBS):
        teams[index + 1] = Team(
            id=index + 1, name=fpl_name,
            short_name=hist.FPL_LONG_TO_SHORT[fpl_name], code=1000 + index,
            strength_overall_home=3, strength_overall_away=3)

    past = history_past_rows(season, prior_season_name)
    players: Dict[int, Player] = {}
    elements: List[Dict[str, Any]] = []
    for row in season["players"]:
        prior = past.get(row["id"], [{}])[0]
        players[row["id"]] = Player(
            id=row["id"], code=row["code"], web_name=row["web_name"],
            first_name=row["first_name"], second_name=row["second_name"],
            team_id=row["club"] + 1, position=row["position"], now_cost=row["price"],
            status="a", selected_by_percent=round(row["price"] / 20.0, 1),
            minutes=int(prior.get("minutes", 0)), starts=int(prior.get("starts", 0)),
            total_points=int(prior.get("total_points", 0)),
            expected_goals=float(prior.get("expected_goals", 0.0)),
            expected_assists=float(prior.get("expected_assists", 0.0)),
            goals_scored=int(prior.get("goals_scored", 0)),
            assists=int(prior.get("assists", 0)),
            clean_sheets=int(prior.get("clean_sheets", 0)),
            goals_conceded=int(prior.get("goals_conceded", 0)),
            saves=int(prior.get("saves", 0)), bonus=int(prior.get("bonus", 0)),
            bps=int(prior.get("bps", 0)),
            defensive_contribution=int(prior.get("defensive_contribution", 0)),
            clearances_blocks_interceptions=int(
                prior.get("clearances_blocks_interceptions", 0)),
            recoveries=int(prior.get("recoveries", 0)),
            tackles=int(prior.get("tackles", 0)),
        )
        elements.append({
            "id": row["id"], "code": row["code"], "web_name": row["web_name"],
            "first_name": row["first_name"], "second_name": row["second_name"],
            "team": row["club"] + 1, "element_type": row["position"],
            "now_cost": row["price"], "status": "a", "news": "",
            "selected_by_percent": str(round(row["price"] / 20.0, 1)),
            "minutes": int(prior.get("minutes", 0)),
            "starts": int(prior.get("starts", 0)),
            "total_points": int(prior.get("total_points", 0)),
            "expected_goals": str(prior.get("expected_goals", 0.0)),
            "expected_assists": str(prior.get("expected_assists", 0.0)),
            "form": "0.0", "points_per_game": "0.0", "ep_next": "0.0",
        })

    # Next season's schedule reuses the same round robin, one year on.
    fixtures: List[Fixture] = []
    for fixture in fixture_list():
        kickoff = fixture["kickoff"] + timedelta(days=364)
        fixtures.append(Fixture(
            id=fixture["id"], code=fixture["id"], gw=fixture["gw"],
            team_h=fixture["home"] + 1, team_a=fixture["away"] + 1,
            kickoff_time=kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            team_h_difficulty=3, team_a_difficulty=3, finished=False))

    events = []
    for gw in range(1, GWS + 1):
        deadline = SEASON_START + timedelta(days=364 + 7 * (gw - 1), hours=-2)
        events.append({
            "id": gw, "name": "Gameweek %d" % gw,
            "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": gw < current_gw, "data_checked": gw < current_gw,
            "is_previous": gw == current_gw - 1, "is_current": gw == current_gw - 1,
            "is_next": gw == current_gw, "average_entry_score": 0, "chip_plays": [],
        })

    return GameState(
        teams=teams, players=players, fixtures=fixtures, events=events,
        current_gw=current_gw, finished_gws=list(range(1, current_gw)),
        bootstrap={"elements": elements, "events": events,
                   "teams": [{"id": t.id, "name": t.name, "short_name": t.short_name,
                              "code": t.code} for t in teams.values()]},
        player_history={},
        player_history_past=past,
        elements_are_prior_season=(current_gw == 1),
        season=season_name, prior_season=prior_season_name,
        promoted_team_ids=[], fetched_at="2026-08-01T00:00:00Z", data_warnings=[])


# ---------------------------------------------------------------------------
# Projection sets
# ---------------------------------------------------------------------------


def make_projection_set(
    state: GameState,
    gws: Sequence[int] = (1, 2, 3),
    seed: int = 7,
    season: str = "2026-27",
) -> Any:
    """A deterministic ProjectionSet over every player in ``state``.

    Expected points follow price with noise, so the optimizer has a real
    problem to solve (the cheapest legal squad is not the best one) but the
    answer is stable across runs.
    """
    from gaffer.core.types import PlayerFixtureProjection, PlayerGWProjection, ProjectionSet

    rng = random.Random(seed)
    by_gw: Dict[int, Dict[int, Fixture]] = {}
    for fixture in state.fixtures:
        if fixture.gw is None:
            continue
        by_gw.setdefault(int(fixture.gw), {})[fixture.team_h] = fixture
        by_gw[int(fixture.gw)][fixture.team_a] = fixture

    out = ProjectionSet(season=season, generated_at="2026-08-01T00:00:00Z",
                        first_gw=min(gws), last_gw=max(gws))
    for pid in sorted(state.players):
        player = state.players[pid]
        base = 0.35 * (player.now_cost / 10.0) + rng.uniform(-0.4, 0.4)
        per_gw: Dict[int, PlayerGWProjection] = {}
        for gw in gws:
            fixture = by_gw.get(int(gw), {}).get(player.team_id)
            gwp = PlayerGWProjection(player_id=pid, gw=int(gw))
            if fixture is not None:
                xp = max(0.0, base + rng.uniform(-0.3, 0.3))
                projection = PlayerFixtureProjection(
                    player_id=pid, gw=int(gw), fixture_id=fixture.id,
                    opponent_id=fixture.opponent_of(player.team_id),
                    is_home=fixture.is_home_for(player.team_id),
                    p_appear=0.9, p_start=0.85, p_60=0.8, xmins=75.0,
                    xp_total=xp, sd_total=1.0 + xp * 0.4)
                gwp.fixtures.append(projection)
                gwp.xp = xp
                gwp.sd = projection.sd_total
            per_gw[int(gw)] = gwp
        out.projections[pid] = per_gw
    return out
