"""The single entry point every other module uses to get at the data.

``load_game_state`` turns the raw API payloads into the shared types from
``gaffer.core.types`` and answers the two questions everything downstream
depends on:

* **Which gameweek are we planning for?** The next event whose deadline has not
  passed and which is not finished. Once a deadline passes you cannot change
  anything, so the in-progress gameweek is never the planning target.
* **Which season do the bootstrap element totals describe?** Pre-season they
  carry *last* season's numbers, not zeros (Raya: 3330 minutes, 162 points on
  2026-08-14). ``GameState.elements_are_prior_season`` makes that explicit so no
  model silently treats 162 points as season-to-date form.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from gaffer.core import scoring
from gaffer.core.config import SEASON, Config
from gaffer.core.types import Fixture, Player, Team
from gaffer.data import history as hist
from gaffer.data.cache import Cache
from gaffer.data.fpl_api import FPLClient

log = logging.getLogger(__name__)

# Element fields the API delivers as strings ("0.00", "1.2") and every model
# expects as numbers. Casting these is not optional: `float("0.00") > 0` is a
# very different test from `"0.00" > 0`.
STRING_NUMERIC_ELEMENT_FIELDS = (
    "form",
    "points_per_game",
    "selected_by_percent",
    "value_form",
    "value_season",
    "ep_next",
    "ep_this",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "price_change_percent",
)

NUMERIC_ELEMENT_FIELDS = STRING_NUMERIC_ELEMENT_FIELDS + (
    "minutes",
    "starts",
    "starts_per_90",
    "total_points",
    "event_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "clean_sheets_per_90",
    "goals_conceded",
    "goals_conceded_per_90",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "saves_per_90",
    "bonus",
    "bps",
    "defensive_contribution",
    "defensive_contribution_per_90",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",
    "now_cost",
    "cost_change_start",
    "cost_change_event",
    "transfers_in",
    "transfers_out",
    "transfers_in_event",
    "transfers_out_event",
    "dreamteam_count",
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    "chance_of_playing_this_round",
    "chance_of_playing_next_round",
    "squad_number",
)

# element-summary `history` rows, 2025/26 schema. Used to give every player a
# consistently shaped (possibly empty) DataFrame — pre-season *nobody* has
# history, and downstream code must still be able to say df["minutes"].sum().
HISTORY_COLUMNS = (
    "element",
    "fixture",
    "opponent_team",
    "total_points",
    "was_home",
    "kickoff_time",
    "team_h_score",
    "team_a_score",
    "round",
    "modified",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "defensive_contribution",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value",
    "transfers_balance",
    "selected",
    "transfers_in",
    "transfers_out",
)

HISTORY_NUMERIC_COLUMNS = tuple(
    c
    for c in HISTORY_COLUMNS
    if c not in ("was_home", "kickoff_time", "modified")
)

# `history_past` season rows: the same string-typed numerics.
HISTORY_PAST_NUMERIC_FIELDS = (
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # reject NaN


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_teams(bootstrap: Dict[str, Any]) -> Dict[int, Team]:
    teams: Dict[int, Team] = {}
    for raw in bootstrap["teams"]:
        teams[int(raw["id"])] = Team(
            id=int(raw["id"]),
            name=raw["name"],
            short_name=raw["short_name"],
            code=int(raw["code"]),
            # Pre-season every strength_* except overall is 0 and `strength` is
            # null; `or 0` keeps that from becoming a TypeError.
            strength_overall_home=int(raw.get("strength_overall_home") or 0),
            strength_overall_away=int(raw.get("strength_overall_away") or 0),
            strength_attack_home=int(raw.get("strength_attack_home") or 0),
            strength_attack_away=int(raw.get("strength_attack_away") or 0),
            strength_defence_home=int(raw.get("strength_defence_home") or 0),
            strength_defence_away=int(raw.get("strength_defence_away") or 0),
        )
    return teams


def build_players(bootstrap: Dict[str, Any]) -> Dict[int, Player]:
    players: Dict[int, Player] = {}
    for raw in bootstrap["elements"]:
        pid = int(raw["id"])
        players[pid] = Player(
            id=pid,
            code=int(raw["code"]),
            web_name=raw["web_name"],
            first_name=raw.get("first_name") or "",
            second_name=raw.get("second_name") or "",
            team_id=int(raw["team"]),
            position=int(raw["element_type"]),
            now_cost=int(raw["now_cost"]),
            status=raw.get("status") or "a",
            news=raw.get("news") or "",
            chance_of_playing_next_round=_to_int_or_none(raw.get("chance_of_playing_next_round")),
            chance_of_playing_this_round=_to_int_or_none(raw.get("chance_of_playing_this_round")),
            selected_by_percent=_to_float(raw.get("selected_by_percent")),
            penalties_order=_to_int_or_none(raw.get("penalties_order")),
            # The API calls this corners_and_indirect_freekicks_order.
            corners_order=_to_int_or_none(raw.get("corners_and_indirect_freekicks_order")),
            direct_freekicks_order=_to_int_or_none(raw.get("direct_freekicks_order")),
            minutes=int(raw.get("minutes") or 0),
            starts=int(raw.get("starts") or 0),
            total_points=int(raw.get("total_points") or 0),
            points_per_game=_to_float(raw.get("points_per_game")),
            form=_to_float(raw.get("form")),
            ep_next=_to_float(raw.get("ep_next")),
            price_change_percent=_to_float(raw.get("price_change_percent")),
            transfers_in_event=int(raw.get("transfers_in_event") or 0),
            transfers_out_event=int(raw.get("transfers_out_event") or 0),
            expected_goals=_to_float(raw.get("expected_goals")),
            expected_assists=_to_float(raw.get("expected_assists")),
            expected_goal_involvements=_to_float(raw.get("expected_goal_involvements")),
            expected_goals_conceded=_to_float(raw.get("expected_goals_conceded")),
            goals_scored=int(raw.get("goals_scored") or 0),
            assists=int(raw.get("assists") or 0),
            clean_sheets=int(raw.get("clean_sheets") or 0),
            goals_conceded=int(raw.get("goals_conceded") or 0),
            saves=int(raw.get("saves") or 0),
            bonus=int(raw.get("bonus") or 0),
            bps=int(raw.get("bps") or 0),
            yellow_cards=int(raw.get("yellow_cards") or 0),
            red_cards=int(raw.get("red_cards") or 0),
            defensive_contribution=int(raw.get("defensive_contribution") or 0),
            clearances_blocks_interceptions=int(raw.get("clearances_blocks_interceptions") or 0),
            recoveries=int(raw.get("recoveries") or 0),
            tackles=int(raw.get("tackles") or 0),
            team_join_date=raw.get("team_join_date"),
        )
    return players


def build_fixtures(raw_fixtures: Sequence[Dict[str, Any]]) -> List[Fixture]:
    out: List[Fixture] = []
    for raw in raw_fixtures:
        out.append(
            Fixture(
                id=int(raw["id"]),
                code=int(raw["code"]),
                gw=_to_int_or_none(raw.get("event")),
                team_h=int(raw["team_h"]),
                team_a=int(raw["team_a"]),
                kickoff_time=raw.get("kickoff_time"),
                team_h_difficulty=int(raw.get("team_h_difficulty") or 3),
                team_a_difficulty=int(raw.get("team_a_difficulty") or 3),
                finished=bool(raw.get("finished")),
                team_h_score=_to_int_or_none(raw.get("team_h_score")),
                team_a_score=_to_int_or_none(raw.get("team_a_score")),
            )
        )
    # Stable order: gameweek, then kickoff, then id. Unscheduled fixtures last.
    out.sort(key=lambda f: (f.gw is None, f.gw or 0, f.kickoff_time or "", f.id))
    return out


def resolve_current_gw(events: Sequence[Dict[str, Any]], now: Optional[datetime] = None) -> int:
    """The next gameweek we can still make decisions for.

    Not simply ``is_next``: mid-season the API's is_current/is_next pair moves
    at the *deadline*, but a gameweek that has kicked off is already locked, so
    the planning target is the first unfinished event whose deadline is still
    ahead of us. Falls back to the first unfinished event (deadline passed but
    matches unplayed), then to the final gameweek once the season is over.
    """
    now = now or datetime.now(timezone.utc)
    ordered = sorted(events, key=lambda e: int(e["id"]))
    for event in ordered:
        if event.get("finished"):
            continue
        deadline = _parse_iso(event.get("deadline_time"))
        if deadline is not None and deadline > now:
            return int(event["id"])
    for event in ordered:
        if not event.get("finished"):
            return int(event["id"])
    return int(ordered[-1]["id"]) if ordered else scoring.TOTAL_GWS


_EMPTY_FRAMES: Dict[tuple, pd.DataFrame] = {}


def _empty_history_frame(columns: Sequence[str]) -> pd.DataFrame:
    """Correctly shaped zero-row history frame.

    Pre-season all 587 players need one of these, and constructing a DataFrame
    from a dict of empty Series costs ~3ms each — 1.9s of the load. Build one
    template per column set and copy it.
    """
    key = tuple(columns)
    template = _EMPTY_FRAMES.get(key)
    if template is None:
        template = pd.DataFrame({c: pd.Series(dtype="float64") for c in key})
        template["gw"] = pd.Series(dtype="float64")
        _EMPTY_FRAMES[key] = template
    return template.copy()


def _history_frame(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return _empty_history_frame(columns)
    df = pd.DataFrame(list(rows))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    for col in HISTORY_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    if "round" in df.columns:
        df["gw"] = df["round"]
        df = df.sort_values(["round", "fixture"], kind="mergesort").reset_index(drop=True)
    return df


def _clean_history_past(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cast the string-typed numerics in history_past; keep every other field."""
    out = []
    for row in rows:
        clean = dict(row)
        for key in HISTORY_PAST_NUMERIC_FIELDS:
            if key in clean:
                clean[key] = _to_float(clean[key])
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    teams: Dict[int, Team]
    players: Dict[int, Player]
    fixtures: List[Fixture]
    events: List[Dict[str, Any]]
    current_gw: int
    finished_gws: List[int]
    bootstrap: Dict[str, Any]
    player_history: Dict[int, pd.DataFrame] = field(default_factory=dict)
    player_history_past: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    # True when no gameweek of the *current* season has finished, meaning every
    # elements[] total in bootstrap describes LAST season. Models must not read
    # these as season-to-date form when this is True.
    elements_are_prior_season: bool = True
    season: str = SEASON
    prior_season: str = ""
    promoted_team_ids: List[int] = field(default_factory=list)
    fetched_at: str = ""
    data_warnings: List[str] = field(default_factory=list)
    _players_df: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    # -- lookups ------------------------------------------------------------
    def team(self, team_id: int) -> Team:
        return self.teams[team_id]

    def player(self, player_id: int) -> Player:
        return self.players[player_id]

    def short_name(self, team_id: int) -> str:
        team = self.teams.get(team_id)
        return team.short_name if team else "?%d" % team_id

    def position_name(self, player_id: int) -> str:
        return scoring.POS_NAME[self.players[player_id].position]

    def event(self, gw: int) -> Optional[Dict[str, Any]]:
        for e in self.events:
            if int(e["id"]) == gw:
                return e
        return None

    def deadline(self, gw: int) -> Optional[str]:
        event = self.event(gw)
        return event.get("deadline_time") if event else None

    def gws_remaining(self) -> List[int]:
        return [int(e["id"]) for e in self.events if int(e["id"]) >= self.current_gw]

    # -- history ------------------------------------------------------------
    def history(self, player_id: int) -> pd.DataFrame:
        """Current-season per-GW rows. Empty (but correctly shaped) pre-season."""
        df = self.player_history.get(player_id)
        if df is None:
            return _history_frame([], HISTORY_COLUMNS)
        return df

    def history_past(self, player_id: int) -> List[Dict[str, Any]]:
        return self.player_history_past.get(player_id, [])

    def prior_season_row(
        self, player_id: int, season_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """The ``history_past`` entry for one season, or None if absent.

        ``season_name`` accepts "2025-26" or "2025/26" and defaults to the
        season immediately before the current one. Returns None for players with
        no top-flight history (new signings, promoted-club players) — that is a
        real signal, not an error, and the minutes/attacking models must have a
        prior for it.
        """
        rows = self.player_history_past.get(player_id) or []
        if not rows:
            return None
        target = hist.season_slash(season_name or self.prior_season or hist.prior_season(self.season))
        for row in rows:
            if row.get("season_name") == target:
                return row
        return None

    def latest_season_row(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Most recent ``history_past`` entry, whichever season that is."""
        rows = self.player_history_past.get(player_id) or []
        return rows[-1] if rows else None

    # -- dataframe view -----------------------------------------------------
    def players_df(self, refresh: bool = False) -> pd.DataFrame:
        """All bootstrap element rows with numerics properly typed.

        Keeps every raw API column name so results can be cross-checked against
        the API, and adds: ``player_id``, ``team_id``, ``pos`` (GKP/DEF/MID/FWD),
        ``price`` (in £m), ``team_short``, ``team_name``, ``promoted``.
        """
        if self._players_df is not None and not refresh:
            return self._players_df
        df = pd.DataFrame(self.bootstrap["elements"])
        for col in NUMERIC_ELEMENT_FIELDS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("penalties_order", "corners_and_indirect_freekicks_order",
                    "direct_freekicks_order", "chance_of_playing_this_round",
                    "chance_of_playing_next_round", "squad_number"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
        df["player_id"] = df["id"].astype(int)
        df["team_id"] = df["team"].astype(int)
        df["position"] = df["element_type"].astype(int)
        df["pos"] = df["position"].map(scoring.POS_NAME)
        df["price"] = df["now_cost"] / 10.0
        df["team_short"] = df["team_id"].map({t.id: t.short_name for t in self.teams.values()})
        df["team_name"] = df["team_id"].map({t.id: t.name for t in self.teams.values()})
        df["promoted"] = df["team_id"].isin(self.promoted_team_ids)
        df["corners_order"] = df.get("corners_and_indirect_freekicks_order")
        df = df.set_index("player_id", drop=False).sort_index()
        self._players_df = df
        return df

    def summary(self) -> str:
        return (
            "%s: %d teams (%d promoted), %d players, %d fixtures, "
            "current_gw=%d, finished=%d, elements_are_prior_season=%s"
            % (
                self.season,
                len(self.teams),
                len(self.promoted_team_ids),
                len(self.players),
                len(self.fixtures),
                self.current_gw,
                len(self.finished_gws),
                self.elements_are_prior_season,
            )
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_game_state(
    config: Optional[Config] = None,
    with_histories: bool = True,
    force: bool = False,
    client: Optional[FPLClient] = None,
    detect_promoted: bool = True,
    progress: bool = True,
) -> GameState:
    """Build the whole game state from the API (via the disk cache).

    Cold this issues 2 + N player requests (589 in 2026/27) and takes ~10s;
    warm it is sub-second. ``with_histories=False`` skips the per-player fetch
    entirely, for callers that only need teams/fixtures/prices.
    """
    config = config or Config.load()
    cache = Cache(default_ttl=config.cache_ttl_seconds)
    client = client or FPLClient(config, cache)
    warnings: List[str] = []

    bootstrap = client.bootstrap(force=force)
    raw_fixtures = client.fixtures(force=force)

    teams = build_teams(bootstrap)
    players = build_players(bootstrap)
    fixtures = build_fixtures(raw_fixtures)
    events = list(bootstrap["events"])

    finished_gws = sorted(int(e["id"]) for e in events if e.get("finished"))
    current_gw = resolve_current_gw(events)
    elements_are_prior_season = not finished_gws

    player_history: Dict[int, pd.DataFrame] = {}
    player_history_past: Dict[int, List[Dict[str, Any]]] = {}
    if with_histories:
        summaries = client.all_element_summaries(
            sorted(players.keys()), force=force, progress=progress
        )
        if client.failed_element_summaries:
            warnings.append(
                "element-summary failed for %d players: %s"
                % (
                    len(client.failed_element_summaries),
                    sorted(client.failed_element_summaries),
                )
            )
        seen_columns: List[str] = []
        for pid, summary in summaries.items():
            rows = summary.get("history") or []
            if rows:
                seen_columns = list(rows[0].keys())
                break
        columns = seen_columns or list(HISTORY_COLUMNS)
        for pid in sorted(players.keys()):
            summary = summaries.get(pid)
            if summary is None:
                player_history[pid] = _history_frame([], columns)
                player_history_past[pid] = []
                continue
            player_history[pid] = _history_frame(summary.get("history") or [], columns)
            player_history_past[pid] = _clean_history_past(summary.get("history_past") or [])
    else:
        warnings.append("loaded without per-player histories (with_histories=False)")

    prior = hist.prior_season(config.season)
    promoted_ids: List[int] = []
    if detect_promoted:
        try:
            prior_results = hist.load_football_data([prior], cache=cache)
            promoted_ids = hist.detect_promoted(teams, prior_results)
            for tid in promoted_ids:
                teams[tid].promoted = True
            log.info(
                "promoted teams for %s: %s",
                config.season,
                ", ".join("%s (id %d)" % (teams[t].name, t) for t in promoted_ids) or "none",
            )
            if len(promoted_ids) != 3:
                warnings.append(
                    "expected 3 promoted teams, detected %d: %s"
                    % (len(promoted_ids), [teams[t].short_name for t in promoted_ids])
                )
        except Exception as exc:  # network or mapping failure must be visible
            msg = "promoted-team detection failed (%s: %s); Team.promoted left False" % (
                type(exc).__name__,
                exc,
            )
            log.warning(msg)
            warnings.append(msg)

    state = GameState(
        teams=teams,
        players=players,
        fixtures=fixtures,
        events=events,
        current_gw=current_gw,
        finished_gws=finished_gws,
        bootstrap=bootstrap,
        player_history=player_history,
        player_history_past=player_history_past,
        elements_are_prior_season=elements_are_prior_season,
        season=config.season,
        prior_season=prior,
        promoted_team_ids=promoted_ids,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        data_warnings=warnings,
    )
    for w in warnings:
        log.warning("data warning: %s", w)
    return state


# ---------------------------------------------------------------------------
# Fixture views
# ---------------------------------------------------------------------------


def fixtures_by_gw(state: GameState) -> Dict[int, List[Fixture]]:
    """``{gw: [fixtures]}`` with an entry for every scheduled gameweek.

    Gameweeks with no fixtures at all map to an empty list rather than being
    absent, so a league-wide blank is visible instead of a KeyError.
    """
    out: Dict[int, List[Fixture]] = {int(e["id"]): [] for e in state.events}
    for fixture in state.fixtures:
        if fixture.gw is None:
            continue
        out.setdefault(int(fixture.gw), []).append(fixture)
    for gw in out:
        out[gw].sort(key=lambda f: (f.kickoff_time or "", f.id))
    return out


def unscheduled_fixtures(state: GameState) -> List[Fixture]:
    """Fixtures with no gameweek yet (postponed, awaiting rescheduling).

    These become the doubles later in the season, so chip planning has to watch
    them.
    """
    return [f for f in state.fixtures if f.gw is None]


def team_fixtures(state: GameState, team_id: int, gws: Sequence[int]) -> List[Fixture]:
    """Every fixture a team plays across ``gws``, one entry per fixture.

    A double gameweek yields two entries for that gameweek and a blank yields
    none, so ``len(result)`` is the true number of matches, never the number of
    gameweeks.
    """
    wanted = set(int(g) for g in gws)
    out = [
        f
        for f in state.fixtures
        if f.gw is not None and int(f.gw) in wanted and team_id in (f.team_h, f.team_a)
    ]
    out.sort(key=lambda f: (f.gw or 0, f.kickoff_time or "", f.id))
    return out


def team_fixture_counts(state: GameState, gws: Sequence[int]) -> Dict[int, Dict[int, int]]:
    """``{team_id: {gw: n_fixtures}}`` — the raw material for double/blank detection."""
    counts: Dict[int, Dict[int, int]] = {
        t: {int(g): 0 for g in gws} for t in state.teams
    }
    wanted = set(int(g) for g in gws)
    for f in state.fixtures:
        if f.gw is None or int(f.gw) not in wanted:
            continue
        counts[f.team_h][int(f.gw)] += 1
        counts[f.team_a][int(f.gw)] += 1
    return counts


def prior_season_row(
    state: GameState, player_id: int, season_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Module-level twin of ``GameState.prior_season_row``."""
    return state.prior_season_row(player_id, season_name)


def player_history(state: GameState, player_id: int) -> pd.DataFrame:
    return state.history(player_id)


def player_history_past(state: GameState, player_id: int) -> List[Dict[str, Any]]:
    return state.history_past(player_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()

    print("=== cold load (cache may already be warm from a prior run) ===")
    t0 = time.time()
    state = load_game_state(cfg)
    cold = time.time() - t0
    print("  %.2fs -> %s" % (cold, state.summary()))

    t0 = time.time()
    state2 = load_game_state(cfg)
    warm = time.time() - t0
    print("=== warm load ===\n  %.2fs -> %s" % (warm, state2.summary()))

    print("\n=== assertions ===")
    assert len(state.teams) == 20, len(state.teams)
    assert len(state.players) == 587, len(state.players)
    assert len(state.fixtures) == 380, len(state.fixtures)
    assert state.current_gw == 1, state.current_gw
    assert state.elements_are_prior_season is True
    print("  20 teams / 587 players / 380 fixtures / current_gw=1: OK")

    print("\n=== GW%d fixtures ===" % state.current_gw)
    by_gw = fixtures_by_gw(state)
    for f in by_gw[state.current_gw]:
        print(
            "  %-5s %-3s v %-3s  FDR %d-%d  %s"
            % (
                "GW%d" % f.gw,
                state.short_name(f.team_h),
                state.short_name(f.team_a),
                f.team_h_difficulty,
                f.team_a_difficulty,
                f.kickoff_time,
            )
        )
    counts = team_fixture_counts(state, list(range(1, 39)))
    doubles = [(t, g) for t, per in counts.items() for g, n in per.items() if n > 1]
    blanks = [(t, g) for t, per in counts.items() for g, n in per.items() if n == 0]
    print("  scheduled doubles: %d, blanks: %d, unscheduled fixtures: %d"
          % (len(doubles), len(blanks), len(unscheduled_fixtures(state))))
    arsenal = team_fixtures(state, 1, [1, 2, 3])
    print("  team_fixtures(ARS, [1,2,3]) -> %d: %s"
          % (len(arsenal),
             ", ".join("GW%d %s%s" % (f.gw, "" if f.is_home_for(1) else "@",
                                      state.short_name(f.opponent_of(1))) for f in arsenal)))

    print("\n=== promoted detection (prior season %s) ===" % state.prior_season)
    for tid in state.promoted_team_ids:
        print("  promoted: %-16s %s (id %d)" % (state.teams[tid].name, state.teams[tid].short_name, tid))
    assert sorted(state.teams[t].short_name for t in state.promoted_team_ids) == ["COV", "HUL", "IPS"], \
        [state.teams[t].short_name for t in state.promoted_team_ids]
    print("  exactly COV, HUL, IPS: OK")

    print("\n=== players_df ===")
    df = state.players_df()
    print("  shape=%s" % str(df.shape))
    print("  dtypes of string-typed API fields: %s"
          % ", ".join("%s=%s" % (c, df[c].dtype) for c in
                      ("form", "selected_by_percent", "expected_goals", "ep_next", "price_change_percent")))
    assert df["expected_goals"].dtype.kind == "f"
    assert df["selected_by_percent"].dtype.kind == "f"
    top = df.sort_values("total_points", ascending=False).head(5)
    print(top[["web_name", "pos", "team_short", "price", "minutes", "total_points",
               "expected_goals", "expected_assists", "defensive_contribution"]].to_string(index=False))

    print("\n=== prior-season rows (%s) ===" % hist.season_slash(state.prior_season))
    picks = []
    for want in ("Haaland", "Senesi", "Anderson"):
        match = [p for p in state.players.values() if p.web_name == want]
        if match:
            picks.append(match[0].id)
    for pid in (picks or sorted(state.players)[:3]):
        p = state.player(pid)
        row = state.prior_season_row(pid)
        if row is None:
            print("  %-12s %s: no prior-season row (new to the PL)" % (p.web_name, state.position_name(pid)))
            continue
        print("  %-12s %-3s %-3s  mins=%-5d starts=%-3d pts=%-4d xG=%-5.2f xA=%-5.2f "
              "xGC=%-6.2f DEFCON(raw %s)=%d"
              % (p.web_name, state.position_name(pid), state.short_name(p.team_id),
                 row["minutes"], row["starts"], row["total_points"],
                 row["expected_goals"], row["expected_assists"], row["expected_goals_conceded"],
                 "CBIT" if p.position == 2 else ("CBIRT" if p.position in (3, 4) else "n/a"),
                 row["defensive_contribution"]))

    print("\n=== per-GW history ===")
    h = state.history(1)
    print("  player 1 history rows=%d, columns=%d (empty pre-season is expected)"
          % (len(h), len(h.columns)))
    print("  history_past seasons for player 1: %s"
          % [r["season_name"] for r in state.history_past(1)])

    if state.data_warnings:
        print("\n=== data warnings ===")
        for w in state.data_warnings:
            print("  - %s" % w)
    print("\ntimings: cold=%.2fs warm=%.2fs" % (cold, warm))
