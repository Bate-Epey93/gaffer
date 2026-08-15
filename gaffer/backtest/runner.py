"""Walk a completed season gameweek by gameweek and score the model honestly.

## The cardinal rule: no lookahead

Every projection for gameweek *G* is produced from a ``GameState`` that is
reconstructed from scratch and contains **nothing dated on or after the GW *G*
deadline**. The rule is enforced in five places, and audited in a sixth:

1. **Player history.** ``merged_gw.csv`` is pre-split per player and sliced with
   ``iloc[:cut]`` where ``cut`` is the number of rows with ``round < G``. The
   slice is the only per-player history the state ever exposes, so a model that
   reads ``state.history(pid)`` physically cannot see the answer.
2. **Season-to-date totals.** Every ``Player`` field (minutes, xG, bps, form,
   points-per-game...) comes from a cumulative sum evaluated at ``G-1``, never
   from the season aggregate.
3. **Prices and clubs.** Taken from the player's row at ``G-1`` (at ``G == 1``,
   from the GW1 row, which carries the pre-season starting price by
   construction). Prices are set at the deadline, so ``G-1`` is safe and the
   target row is never touched.
4. **Results.** Fixtures for ``gw >= G`` are built with ``finished=False`` and
   ``team_h_score=None``. ``TeamRatingModel`` only ingests finished fixtures
   with scores, and it additionally gets ``as_of=deadline(G)`` so its own
   leakage guard drops anything dated later (see ``team_ratings_as_of``).
5. **Fitted parameters.** ``reports/{attacking,defending,defcon}_fit.json`` in
   this repo were fitted for 2026/27 — whose *prior season is the season under
   test*. Reading them would hand the backtest its own answers, and rewriting
   them would corrupt the live app. ``replay_environment`` redirects every one
   of those paths to a backtest-only directory, and the seasons each fit is
   allowed to touch are passed explicitly and stop before the test season.
6. **Audit.** ``leakage_audit`` re-scans the assembled state for any history row
   with ``round >= G``, any fixture at ``gw >= G`` carrying a score, and any
   ``finished_gws`` entry at or beyond ``G``. It runs on every gameweek and any
   violation aborts the run rather than quietly degrading it.

What the replay honestly cannot reproduce is listed in ``LIMITATIONS`` and
printed with the report. The two that matter: the archive has no injury news or
availability flags, and 2024/25 (the prior season for a 2025/26 backtest)
predates FPL's defensive-action stats, so DEFCON has no historical basis until
the current season accumulates one. Both handicap the model relative to
production, not the other way round.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.backtest import metrics as M
from gaffer.core import scoring
from gaffer.core.config import CACHE_DIR, REPORTS_DIR, Config, ensure_dirs
from gaffer.core.types import Fixture, Player, Team
from gaffer.data import history as hist
from gaffer.data.cache import Cache
from gaffer.data.loaders import GameState
from gaffer.model import attacking as attacking_mod
from gaffer.model import bonus as bonus_mod
from gaffer.model import defcon as defcon_mod
from gaffer.model import defending as defending_mod
from gaffer.model import minutes as minutes_mod
from gaffer.model.team_ratings import TeamRatingModel
from gaffer.model.xp import XPEngine

log = logging.getLogger(__name__)

# FPL deadlines are 90 minutes before the first kickoff of the gameweek. The
# archive does not carry the deadline, so it is derived; erring early can only
# make the replay more conservative.
DEADLINE_BEFORE_FIRST_KICKOFF = timedelta(minutes=90)

# Counting stats that accumulate over a season. Everything a Player carries as
# a season-to-date total is summed from these.
STAT_COLUMNS: Tuple[str, ...] = (
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "clearances_blocks_interceptions", "recoveries",
    "tackles", "defensive_contribution", "influence", "creativity", "threat",
)

# Columns a history frame must expose for the models; mirrors the FPL
# element-summary `history` schema, which merged_gw follows almost exactly.
HISTORY_KEEP: Tuple[str, ...] = STAT_COLUMNS + (
    "element", "fixture", "opponent_team", "was_home", "kickoff_time",
    "team_h_score", "team_a_score", "round", "value", "selected",
    "transfers_in", "transfers_out", "transfers_balance", "ict_index",
)

# FPL's own points_per_game and form windows.
FORM_GWS = 4

LIMITATIONS: Tuple[str, ...] = (
    "No availability data. The archive carries no `status`, `news` or "
    "`chance_of_playing`, so every player is presented to the minutes model as "
    "fully fit. In production that model's first and strongest input is the "
    "injury flag; here it must infer absence from past minutes alone.",
    "No penalty order. `penalties_order` is not in the archive, so the explicit "
    "penalty component of the attacking model never fires. Penalty xG is still "
    "inside each player's historical xG, but the designated-taker uplift is not.",
    "No bookmaker odds for the fixture being projected. football-data publishes "
    "a season's odds in the same file as its results, so admitting them would "
    "admit the scoreline. Lambdas are therefore pure Dixon-Coles, exactly as "
    "they will be for real future fixtures.",
    "No DEFCON history before the test season. FPL only began publishing "
    "clearances/blocks/interceptions, tackles and recoveries in 2025/26, so a "
    "2025/26 backtest starts with no prior-season defensive-action rates at "
    "all and has to learn them from the current season as it goes. Actual "
    "points *do* include DEFCON from GW1, so defenders and defensive "
    "midfielders are under-projected early in the season.",
    "Effective ownership is approximated. `selected_by_percent` is rebuilt from "
    "the archive's raw `selected` count divided by an estimate of the total "
    "manager count (sum of selections / 15).",
)


# ---------------------------------------------------------------------------
# Scoped patches: keep the replay out of the live app's fitted parameters
# ---------------------------------------------------------------------------


@contextmanager
def replay_environment(tag: str, elasticity_seasons: Sequence[str]) -> Iterator[str]:
    """Make the model modules safe to fit inside a replay, then put them back.

    Three scoped patches, each for a reason that cannot be fixed from here:

    * **Fitted-parameter caches.** Every model caches its expensive historical
      fit to ``reports/*.json`` keyed only by a version number, never by
      season. In this repo those files hold 2026/27 fits, whose source season
      is the one under test. Reading them is leakage; writing them would
      replace the live app's parameters with older ones. Both go to a
      backtest-only directory instead.
    * **Attacking elasticity seasons.** ``load_or_fit_historical`` binds its
      ``path`` as a default argument at import time, so the module attribute
      cannot be patched — the function itself is wrapped, which also pins the
      season list to seasons that end before the test season.
    * **Manager rows in the archive.** 2024/25 was the Assistant Manager chip
      season, so ``merged_gw.csv`` carries 322 rows with ``position == "AM"``.
      ``history.VAASTAV_POSITION_TO_ELEMENT_TYPE`` has no entry for those, so
      they arrive with a null ``element_type`` and crash the minutes fit. They
      are not players and are dropped here; the fix belongs in ``history.py``.
    """
    directory = os.path.join(CACHE_DIR, "backtest_fits", tag)
    os.makedirs(directory, exist_ok=True)

    saved: List[Tuple[Any, str, Any]] = []

    def patch(module: Any, name: str, value: Any) -> None:
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    patch(defcon_mod, "FIT_PATH", os.path.join(directory, "defcon_fit.json"))
    patch(defending_mod, "FIT_PATH", os.path.join(directory, "defending_fit.json"))
    patch(bonus_mod, "BPS_FIT_PATH", os.path.join(directory, "bps_fit.json"))
    patch(minutes_mod, "PRIOR_REPORT_PATH", os.path.join(directory, "minutes_price_prior.json"))

    original_attacking = attacking_mod.load_or_fit_historical
    attacking_path = os.path.join(directory, "attacking_fit.json")

    def scoped_attacking_fit(
        seasons: Optional[Sequence[str]] = None,
        cache: Optional[Cache] = None,
        force: bool = False,
        path: str = attacking_path,
    ) -> Any:
        return original_attacking(
            seasons=list(seasons or elasticity_seasons), cache=cache, force=force,
            path=attacking_path,
        )

    patch(attacking_mod, "load_or_fit_historical", scoped_attacking_fit)

    original_gws = hist.load_vaastav_gws

    def player_rows_only(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original_gws(*args, **kwargs)
        if "element_type" in frame.columns and frame["element_type"].isna().any():
            frame = frame[frame["element_type"].notna()].copy()
        return frame

    patch(hist, "load_vaastav_gws", player_rows_only)

    # DefconModel._fit_from_data guards the *load* of the prior season but not
    # the columns it then aggregates, so a season older than 2025/26 (which is
    # every season without FPL's defensive-action stats) raises KeyError
    # instead of taking the documented "recorded fallback constants" path two
    # lines above. Route it there.
    original_defcon_fit = defcon_mod.DefconModel._fit_from_data

    def guarded_defcon_fit(self: Any, state: Any, results: Any, cache: Any) -> Any:
        try:
            return original_defcon_fit(self, state, results, cache)
        except KeyError as exc:
            message = (
                "defcon fit: prior season %s carries no %s, so tilt and dispersion "
                "fall back to the recorded constants (defcon.py raises KeyError here "
                "rather than falling back)" % (hist.prior_season(state.season), exc))
            log.warning(message)
            self.warnings.append(message)
            return defcon_mod.DefconFit()

    patch(defcon_mod.DefconModel, "_fit_from_data", guarded_defcon_fit)
    try:
        yield directory
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


@contextmanager
def team_ratings_as_of(deadline: datetime) -> Iterator[None]:
    """Force ``TeamRatingModel.fit`` to apply its leakage guard at ``deadline``.

    The model already implements exactly the guard a backtest needs — it drops
    every result dated after ``as_of`` and says how many it dropped — but
    ``XPEngine.fit`` does not plumb the parameter through. Rather than
    reimplement the engine's fit order, the default is injected here. This is
    belt and braces: the state handed to the engine already carries no future
    result, and the guard proves it by dropping zero.
    """
    original = TeamRatingModel.fit

    def patched(
        self: TeamRatingModel,
        results: Optional[pd.DataFrame] = None,
        state: Optional[Any] = None,
        seasons: Optional[Sequence[str]] = None,
        as_of: Optional[datetime] = None,
        include_state_results: bool = True,
    ) -> None:
        return original(self, results=results, state=state, seasons=seasons,
                        as_of=as_of or deadline,
                        include_state_results=include_state_results)

    TeamRatingModel.fit = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        TeamRatingModel.fit = original  # type: ignore[assignment]


def _player_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop archive rows that are not players (2024/25's Assistant Managers)."""
    if "element_type" in frame.columns and frame["element_type"].isna().any():
        return frame[frame["element_type"].notna()].copy()
    return frame


def ensure_team_name_mappings(frames: Sequence[pd.DataFrame]) -> List[str]:
    """No-op unless a season uses a club long name the mapping table lacks.

    ``gaffer.data.history.FPL_LONG_TO_SHORT`` is missing 2024/25's "Ipswich"
    (it only has "Ipswich Town"), which makes ``load_vaastav_gws("2024-25")``
    raise. That is a real bug in the data layer — it silently cost the shipped
    attacking model a whole season of fit data — and belongs there, not here.
    Until it is fixed the replay adds the alias at runtime rather than dying,
    and returns a warning so the report says out loud that it happened.
    """
    aliases = {"Ipswich": "IPS"}
    added: List[str] = []
    for name, short in aliases.items():
        if name not in hist.FPL_LONG_TO_SHORT:
            hist.FPL_LONG_TO_SHORT[name] = short
            added.append(
                "gaffer.data.history.FPL_LONG_TO_SHORT is missing %r (-> %s); the "
                "backtest added it at runtime. Fix it in history.py: without it "
                "load_vaastav_gws('2024-25') raises UnmappedTeamError." % (name, short)
            )
    return added


# ---------------------------------------------------------------------------
# Season reconstruction
# ---------------------------------------------------------------------------


@dataclass
class ReplayFixture:
    """One match of the season, in both the archive's and gaffer's terms."""

    id: int
    gw: int
    team_h: int
    team_a: int
    kickoff: Optional[pd.Timestamp]
    home_goals: int
    away_goals: int


class SeasonReplay:
    """Rebuilds the ``GameState`` a manager could have had at each deadline."""

    def __init__(
        self,
        season: str,
        config: Optional[Config] = None,
        cache: Optional[Cache] = None,
        prior_depth: int = 2,
    ) -> None:
        self.season = hist.season_dash(season)
        self.prior_depth = int(prior_depth)
        self.cache = cache or Cache()
        base = config or Config.load()
        # Everything downstream resolves "the prior season" from config.season,
        # so the replay's config must claim to be in the season under test.
        self.config = copy.deepcopy(base)
        self.config.season = self.season
        self.prior_season = hist.prior_season(self.season)
        self.warnings: List[str] = []

        self.gws: pd.DataFrame = pd.DataFrame()
        self.teams: Dict[int, Team] = {}
        self.name_to_id: Dict[str, int] = {}
        self.fixtures: List[ReplayFixture] = []
        self.deadlines: Dict[int, datetime] = {}
        self.available_gws: List[int] = []
        self.element_meta: Dict[int, Dict[str, Any]] = {}
        self.history_frames: Dict[int, pd.DataFrame] = {}
        self.history_rounds: Dict[int, np.ndarray] = {}
        self.history_past: Dict[int, List[Dict[str, Any]]] = {}
        self.cumulative: Dict[int, Dict[int, Dict[str, float]]] = {}
        self.price_at: Dict[int, Dict[int, int]] = {}
        self.team_at: Dict[int, Dict[int, int]] = {}
        self.selected_at: Dict[int, Dict[int, float]] = {}
        self.rows_at_gw: Dict[int, List[int]] = {}
        self.load_seconds: float = 0.0

    # -- loading -----------------------------------------------------------

    def load(self) -> "SeasonReplay":
        t0 = time.time()
        self.warnings.extend(ensure_team_name_mappings([]))
        gws = _player_rows(hist.load_vaastav_gws(self.season, cache=self.cache))
        gws = gws[gws["GW"].notna() & gws["element"].notna()].copy()
        # The archive occasionally writes a player-match twice (2025/26 has ten
        # such rows). Left in, they double those players' points and make a
        # single gameweek look like a double.
        duplicates = int(gws.duplicated(subset=["element", "fixture"]).sum())
        if duplicates:
            gws = gws.drop_duplicates(subset=["element", "fixture"], keep="first")
            self.warnings.append(
                "%s merged_gw contained %d duplicated player-fixture row(s); dropped"
                % (self.season, duplicates))
        gws["GW"] = gws["GW"].astype(int)
        gws["element"] = gws["element"].astype(int)
        gws = gws.sort_values(["element", "GW", "fixture"], kind="mergesort")
        for col in STAT_COLUMNS + ("value", "selected", "xP"):
            if col in gws.columns:
                gws[col] = pd.to_numeric(gws[col], errors="coerce").fillna(0.0)
            else:
                gws[col] = 0.0
                self.warnings.append(
                    "merged_gw %s has no %r column; treated as zero" % (self.season, col))
        self.gws = gws
        self.available_gws = sorted(int(g) for g in gws["GW"].unique())

        self._build_teams_and_fixtures()
        self._build_elements()
        self._build_cumulative()
        self._build_history()
        self._build_history_past()
        self.load_seconds = time.time() - t0
        return self

    def _build_teams_and_fixtures(self) -> None:
        """Recover the club id table and the fixture list from the archive.

        merged_gw names a player's own club but only gives the *opponent* as an
        id, so the id of a club is read off the other side of its own matches.
        Doing it this way rather than from a hardcoded table means the mapping
        is correct for whichever season is replayed.
        """
        gws = self.gws
        home = gws[gws["was_home"] == True]  # noqa: E712
        away = gws[gws["was_home"] == False]  # noqa: E712
        home_side = home.groupby("fixture").agg(
            name=("team", "first"), opponent=("opponent_team", "first"),
            gw=("GW", "first"), kickoff=("kickoff_time", "first"),
            hg=("team_h_score", "first"), ag=("team_a_score", "first"))
        away_side = away.groupby("fixture").agg(
            name=("team", "first"), opponent=("opponent_team", "first"))
        joined = home_side.join(away_side, how="inner", lsuffix="_h", rsuffix="_a")

        name_to_id: Dict[str, int] = {}
        fixtures: List[ReplayFixture] = []
        for fixture_id, row in joined.iterrows():
            home_id = int(row["opponent_a"])   # what the away side calls its opponent
            away_id = int(row["opponent_h"])
            name_to_id.setdefault(str(row["name_h"]), home_id)
            name_to_id.setdefault(str(row["name_a"]), away_id)
            fixtures.append(ReplayFixture(
                id=int(fixture_id), gw=int(row["gw"]), team_h=home_id, team_a=away_id,
                kickoff=row["kickoff"],
                home_goals=int(row["hg"]) if row["hg"] == row["hg"] else 0,
                away_goals=int(row["ag"]) if row["ag"] == row["ag"] else 0))
        fixtures.sort(key=lambda f: (f.gw, str(f.kickoff), f.id))
        self.fixtures = fixtures
        self.name_to_id = name_to_id

        unmapped = sorted(n for n in name_to_id if n not in hist.FPL_LONG_TO_SHORT)
        if unmapped:
            raise hist.UnmappedTeamError(
                "%s: club long names with no short-name mapping: %s"
                % (self.season, ", ".join(unmapped)))
        self.teams = {
            team_id: Team(
                id=team_id, name=name, short_name=hist.FPL_LONG_TO_SHORT[name],
                code=team_id)
            for name, team_id in sorted(name_to_id.items(), key=lambda kv: kv[1])
        }

        # Deadline = 90 minutes before the gameweek's first kickoff.
        by_gw: Dict[int, List[pd.Timestamp]] = {}
        for f in fixtures:
            if f.kickoff is not None and f.kickoff == f.kickoff:
                by_gw.setdefault(f.gw, []).append(f.kickoff)
        for gw, kickoffs in by_gw.items():
            first = min(kickoffs)
            stamp = pd.Timestamp(first)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            self.deadlines[gw] = stamp.to_pydatetime() - DEADLINE_BEFORE_FIRST_KICKOFF

        self._promoted_ids: List[int] = []
        try:
            prior_results = hist.load_football_data([self.prior_season], cache=self.cache)
            seen = hist.teams_in_results(prior_results)
            self._promoted_ids = sorted(
                t.id for t in self.teams.values() if t.short_name not in seen)
            for tid in self._promoted_ids:
                self.teams[tid].promoted = True
        except Exception as exc:
            self.warnings.append(
                "promoted-team detection failed for %s (%s: %s); Team.promoted left False"
                % (self.season, type(exc).__name__, exc))

    def _build_elements(self) -> None:
        """Player identity: stable ``code``, name, position, per-gameweek club."""
        meta: Dict[int, Dict[str, Any]] = {}
        try:
            raw = hist.load_vaastav_players_raw(self.season, cache=self.cache)
            raw = raw.set_index("id")
        except Exception as exc:
            raw = pd.DataFrame()
            self.warnings.append(
                "players_raw %s unavailable (%s: %s); player codes fall back to element ids"
                % (self.season, type(exc).__name__, exc))

        gws = self.gws
        first_rows = gws.groupby("element").first()
        for element, row in first_rows.iterrows():
            element = int(element)
            name = str(row.get("name", ""))
            parts = name.split(" ")
            info: Dict[str, Any] = {
                "code": element,
                "web_name": parts[-1] if parts else name,
                "first_name": parts[0] if parts else "",
                "second_name": " ".join(parts[1:]) if len(parts) > 1 else "",
                "position": int(row.get("element_type") or 3),
            }
            if len(raw) and element in raw.index:
                praw = raw.loc[element]
                info["code"] = int(praw.get("code", element))
                info["web_name"] = str(praw.get("web_name", info["web_name"]))
                info["first_name"] = str(praw.get("first_name", info["first_name"]))
                info["second_name"] = str(praw.get("second_name", info["second_name"]))
            meta[element] = info
        self.element_meta = meta

        # Price / club / ownership as of each gameweek, forward filled: the
        # value on a player's GW row is his price for that gameweek, and the
        # replay only ever reads the row *before* the one being projected.
        pivot_index = sorted(meta.keys())
        columns = list(range(1, max(self.available_gws) + 1))

        def _pivot(column: str, how: str = "last") -> pd.DataFrame:
            frame = gws.groupby(["element", "GW"])[column].agg(how).unstack()
            frame = frame.reindex(index=pivot_index, columns=columns)
            return frame.ffill(axis=1).bfill(axis=1)

        value = _pivot("value")
        selected = _pivot("selected")
        team_ids = gws["team"].map(self.name_to_id)
        team_frame = gws.assign(_team_id=team_ids).groupby(["element", "GW"])["_team_id"].last()
        team_frame = team_frame.unstack().reindex(index=pivot_index, columns=columns)
        team_frame = team_frame.ffill(axis=1).bfill(axis=1)

        # FPL reports ownership as a percentage; the archive stores the raw
        # count, and every manager picks exactly 15, so the manager count is
        # recoverable from the column total.
        managers = {gw: max(1.0, float(selected[gw].sum()) / scoring.SQUAD_SIZE)
                    for gw in columns}

        self.price_at = {gw: {int(e): int(v) for e, v in value[gw].items() if v == v}
                         for gw in columns}
        self.team_at = {gw: {int(e): int(v) for e, v in team_frame[gw].items() if v == v}
                        for gw in columns}
        self.selected_at = {
            gw: {int(e): 100.0 * float(v) / managers[gw]
                 for e, v in selected[gw].items() if v == v}
            for gw in columns
        }
        self.rows_at_gw = {
            int(gw): sorted(int(e) for e in grp["element"].unique())
            for gw, grp in gws.groupby("GW")
        }

    def _build_cumulative(self) -> None:
        """Season-to-date totals per player at the end of every gameweek."""
        gws = self.gws
        extra = gws.assign(_matches=1.0, _appearances=(gws["minutes"] > 0).astype(float))
        cols = list(STAT_COLUMNS) + ["_matches", "_appearances"]
        per = extra.groupby(["element", "GW"])[cols].sum()
        elements = sorted(self.element_meta.keys())
        full_index = pd.MultiIndex.from_product(
            [elements, list(range(1, max(self.available_gws) + 1))], names=["element", "GW"])
        per = per.reindex(full_index, fill_value=0.0)
        cum = per.groupby(level="element").cumsum()
        self.cumulative = {
            int(gw): cum.xs(gw, level="GW").to_dict("index")
            for gw in range(1, max(self.available_gws) + 1)
        }
        self._zero_totals = {c: 0.0 for c in cols}

    def _build_history(self) -> None:
        """One pre-sorted frame per player, sliced by round at replay time."""
        keep = [c for c in HISTORY_KEEP if c in self.gws.columns]
        frame = self.gws[keep].copy()
        frame["gw"] = frame["round"]
        for element, grp in frame.groupby("element", sort=True):
            grp = grp.sort_values(["round", "fixture"], kind="mergesort").reset_index(drop=True)
            self.history_frames[int(element)] = grp
            self.history_rounds[int(element)] = grp["round"].to_numpy(dtype=float)

    def _build_history_past(self) -> None:
        """``history_past``-shaped season rows for the seasons before this one.

        Joined on the stable player ``code``, never on the element id, which is
        reassigned every summer.
        """
        code_to_seasons: Dict[int, List[Dict[str, Any]]] = {}
        for back in range(self.prior_depth, 0, -1):
            season = hist.prior_season(self.season, back=back)
            try:
                frame = _player_rows(hist.load_vaastav_gws(season, cache=self.cache))
                raw = hist.load_vaastav_players_raw(season, cache=self.cache)
            except Exception as exc:
                self.warnings.append(
                    "prior season %s unavailable (%s: %s); players lose that season "
                    "of history" % (season, type(exc).__name__, exc))
                continue
            codes = dict(zip(raw["id"].astype(int), raw["code"].astype(int)))
            cost = raw.set_index("id") if "id" in raw.columns else pd.DataFrame()
            present = [c for c in STAT_COLUMNS if c in frame.columns]
            missing = [c for c in STAT_COLUMNS if c not in frame.columns]
            if missing:
                self.warnings.append(
                    "%s has no %s column(s); those stats are absent from that season's "
                    "history_past rows rather than fabricated as zero"
                    % (season, ", ".join(missing)))
            totals = frame.groupby("element")[present].sum()
            starts_value = frame.groupby("element")["value"].first() if "value" in frame else None
            end_value = frame.groupby("element")["value"].last() if "value" in frame else None
            for element, row in totals.iterrows():
                code = codes.get(int(element))
                if code is None:
                    continue
                record: Dict[str, Any] = {"season_name": hist.season_slash(season),
                                          "element_code": code}
                for column in present:
                    record[column] = float(row[column])
                if starts_value is not None:
                    record["start_cost"] = float(starts_value.get(int(element), 0.0))
                    record["end_cost"] = float(end_value.get(int(element), 0.0))
                code_to_seasons.setdefault(code, []).append(record)

        for element, info in self.element_meta.items():
            rows = code_to_seasons.get(int(info["code"]), [])
            # history_past is chronological, oldest first, as the API returns it.
            self.history_past[element] = sorted(rows, key=lambda r: str(r["season_name"]))

    # -- per-gameweek state -------------------------------------------------

    def deadline(self, gw: int) -> datetime:
        found = self.deadlines.get(int(gw))
        if found is not None:
            return found
        # A completely blank gameweek has no kickoff to work back from; fall
        # back to a week after the previous one.
        previous = max((g for g in self.deadlines if g < gw), default=None)
        if previous is None:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        return self.deadlines[previous] + timedelta(days=7 * (gw - previous))

    def universe(self, gw: int) -> List[int]:
        """Element ids in the game at the GW ``gw`` deadline.

        The archive writes a row for every element in the game, whether or not
        he played, so "has a row this gameweek" reproduces the bootstrap's
        element list. The exception is a club with no fixture: its players get
        no row at all, so they are carried over from the previous gameweek.
        """
        present = set(self.rows_at_gw.get(int(gw), []))
        if gw > 1:
            blanking = {t for t in self.teams if not self.team_fixtures(t, gw)}
            if blanking:
                for element in self.rows_at_gw.get(int(gw) - 1, []):
                    if self.team_at.get(gw - 1, {}).get(element) in blanking:
                        present.add(element)
        return sorted(present)

    def team_fixtures(self, team_id: int, gw: int) -> List[ReplayFixture]:
        return [f for f in self.fixtures
                if f.gw == gw and team_id in (f.team_h, f.team_a)]

    def _totals(self, element: int, gw: int) -> Dict[str, float]:
        """Season-to-date totals *before* gameweek ``gw``."""
        if gw <= 1:
            return self._zero_totals
        return self.cumulative.get(gw - 1, {}).get(element, self._zero_totals)

    def _price(self, element: int, gw: int) -> int:
        table = self.price_at.get(max(1, gw - 1), {})
        value = table.get(element)
        if value is None:
            value = self.price_at.get(1, {}).get(element, 45)
        return int(value)

    def state_for_gw(self, gw: int) -> GameState:
        """The full ``GameState`` as it stood at the GW ``gw`` deadline."""
        gw = int(gw)
        elements = self.universe(gw)
        players: Dict[int, Player] = {}
        history: Dict[int, pd.DataFrame] = {}
        history_past: Dict[int, List[Dict[str, Any]]] = {}

        for element in elements:
            info = self.element_meta[element]
            totals = self._totals(element, gw)
            recent = self._totals(element, gw) if gw <= FORM_GWS + 1 else None
            window_start = max(0, gw - 1 - FORM_GWS)
            if window_start <= 0:
                form_totals = totals
            else:
                earlier = self.cumulative.get(window_start, {}).get(element, self._zero_totals)
                form_totals = {k: totals[k] - earlier.get(k, 0.0) for k in totals}
            matches = float(totals.get("_matches", 0.0))
            appearances = float(totals.get("_appearances", 0.0))
            form_matches = float(form_totals.get("_matches", 0.0))
            team_id = self.team_at.get(max(1, gw - 1), {}).get(element)
            if team_id is None:
                team_id = self.team_at.get(gw, {}).get(element)
            if team_id is None:
                continue
            position = int(info["position"])
            players[element] = Player(
                id=element,
                code=int(info["code"]),
                web_name=str(info["web_name"]),
                first_name=str(info["first_name"]),
                second_name=str(info["second_name"]),
                team_id=int(team_id),
                position=position,
                now_cost=self._price(element, gw),
                # The archive has no availability data at all; see LIMITATIONS.
                status="a",
                news="",
                chance_of_playing_next_round=None,
                chance_of_playing_this_round=None,
                selected_by_percent=float(
                    self.selected_at.get(max(1, gw - 1), {}).get(element, 0.0)),
                penalties_order=None,
                corners_order=None,
                direct_freekicks_order=None,
                minutes=int(totals.get("minutes", 0.0)),
                starts=int(totals.get("starts", 0.0)),
                total_points=int(totals.get("total_points", 0.0)),
                points_per_game=(totals.get("total_points", 0.0) / appearances
                                 if appearances > 0 else 0.0),
                form=(form_totals.get("total_points", 0.0) / form_matches
                      if form_matches > 0 else 0.0),
                ep_next=0.0,  # deliberately withheld; it is a rival predictor
                expected_goals=float(totals.get("expected_goals", 0.0)),
                expected_assists=float(totals.get("expected_assists", 0.0)),
                expected_goal_involvements=float(
                    totals.get("expected_goal_involvements", 0.0)),
                expected_goals_conceded=float(totals.get("expected_goals_conceded", 0.0)),
                goals_scored=int(totals.get("goals_scored", 0.0)),
                assists=int(totals.get("assists", 0.0)),
                clean_sheets=int(totals.get("clean_sheets", 0.0)),
                goals_conceded=int(totals.get("goals_conceded", 0.0)),
                saves=int(totals.get("saves", 0.0)),
                bonus=int(totals.get("bonus", 0.0)),
                bps=int(totals.get("bps", 0.0)),
                yellow_cards=int(totals.get("yellow_cards", 0.0)),
                red_cards=int(totals.get("red_cards", 0.0)),
                defensive_contribution=int(totals.get("defensive_contribution", 0.0)),
                clearances_blocks_interceptions=int(
                    totals.get("clearances_blocks_interceptions", 0.0)),
                recoveries=int(totals.get("recoveries", 0.0)),
                tackles=int(totals.get("tackles", 0.0)),
                team_join_date=None,
            )
            frame = self.history_frames.get(element)
            if frame is None:
                history[element] = pd.DataFrame(columns=list(HISTORY_KEEP) + ["gw"])
            else:
                cut = int(np.searchsorted(self.history_rounds[element], gw, side="left"))
                history[element] = frame.iloc[:cut]
            history_past[element] = self.history_past.get(element, [])
            del recent

        fixtures = [
            Fixture(
                id=f.id, code=f.id, gw=f.gw, team_h=f.team_h, team_a=f.team_a,
                kickoff_time=(None if f.kickoff is None or f.kickoff != f.kickoff
                              else pd.Timestamp(f.kickoff).isoformat()),
                # The archive carries no FDR; the model builds its own anyway.
                team_h_difficulty=3, team_a_difficulty=3,
                finished=f.gw < gw,
                team_h_score=f.home_goals if f.gw < gw else None,
                team_a_score=f.away_goals if f.gw < gw else None,
            )
            for f in self.fixtures
        ]

        events = []
        for event_gw in range(1, max(self.available_gws) + 1):
            deadline = self.deadline(event_gw)
            events.append({
                "id": event_gw,
                "name": "Gameweek %d" % event_gw,
                "deadline_time": deadline.isoformat().replace("+00:00", "Z"),
                "finished": event_gw < gw,
                "data_checked": event_gw < gw,
                "is_previous": event_gw == gw - 1,
                "is_current": event_gw == gw - 1,
                "is_next": event_gw == gw,
                "average_entry_score": 0,
                "chip_plays": [],
            })

        state = GameState(
            teams=dict(self.teams),
            players=players,
            fixtures=fixtures,
            events=events,
            current_gw=gw,
            finished_gws=list(range(1, gw)),
            bootstrap=self._bootstrap_for(gw, players),
            player_history=history,
            player_history_past=history_past,
            elements_are_prior_season=(gw == 1),
            season=self.season,
            prior_season=self.prior_season,
            promoted_team_ids=list(getattr(self, "_promoted_ids", [])),
            fetched_at=self.deadline(gw).isoformat(),
            data_warnings=list(self.warnings),
        )
        return state

    def _bootstrap_for(self, gw: int, players: Dict[int, Player]) -> Dict[str, Any]:
        """A bootstrap payload shaped like the API's.

        Pre-season the real bootstrap's element totals are *last* season's
        numbers, and ``AttackingModel`` falls back to them when a player has no
        ``history_past`` row. GW1 reproduces that; from GW2 the totals are
        season-to-date, exactly as the API switches over.
        """
        elements: List[Dict[str, Any]] = []
        for pid, player in players.items():
            row: Dict[str, Any] = {
                "id": pid, "code": player.code, "web_name": player.web_name,
                "first_name": player.first_name, "second_name": player.second_name,
                "team": player.team_id, "element_type": player.position,
                "now_cost": player.now_cost, "status": player.status, "news": "",
                "selected_by_percent": player.selected_by_percent,
                "minutes": player.minutes, "starts": player.starts,
                "total_points": player.total_points,
                "points_per_game": player.points_per_game, "form": player.form,
                "ep_next": 0.0, "ep_this": 0.0,
            }
            if gw == 1:
                prior = self.history_past.get(pid, [])
                latest = prior[-1] if prior else {}
                for key, value in latest.items():
                    if key in ("season_name", "element_code"):
                        continue
                    row[key] = value
            else:
                totals = self._totals(pid, gw)
                for key in STAT_COLUMNS:
                    row[key] = totals.get(key, 0.0)
            elements.append(row)
        return {
            "elements": elements,
            "teams": [{"id": t.id, "name": t.name, "short_name": t.short_name,
                       "code": t.code} for t in self.teams.values()],
            "events": [],
            "element_types": [
                {"id": pos, "singular_name_short": scoring.POS_NAME[pos],
                 "squad_select": scoring.SQUAD_SELECT[pos],
                 "squad_min_play": scoring.SQUAD_MIN_PLAY[pos],
                 "squad_max_play": scoring.SQUAD_MAX_PLAY[pos]}
                for pos in scoring.POSITIONS
            ],
        }

    # -- actuals ------------------------------------------------------------

    def actuals_for_gw(self, gw: int) -> pd.DataFrame:
        """What every player really scored, doubles summed."""
        rows = self.gws[self.gws["GW"] == int(gw)]
        if rows.empty:
            return pd.DataFrame(columns=["player_id", "gw", "actual_points", "minutes"])
        grouped = rows.groupby("element").agg(
            actual_points=("total_points", "sum"),
            minutes=("minutes", "sum"),
            starts=("starts", "sum"),
            n_fixtures=("fixture", "count"),
            ep_next=("xP", "sum"),
            bonus=("bonus", "sum"),
        ).reset_index().rename(columns={"element": "player_id"})
        grouped["gw"] = int(gw)
        return grouped

    # -- audit --------------------------------------------------------------

    def leakage_audit(self, state: GameState, gw: int) -> List[str]:
        """Re-derive, from the assembled state, that nothing from GW>=gw is in it."""
        problems: List[str] = []
        for pid, frame in state.player_history.items():
            if len(frame) == 0:
                continue
            worst = float(pd.to_numeric(frame["round"], errors="coerce").max())
            if worst >= gw:
                problems.append(
                    "player %d history contains round %.0f >= gw %d" % (pid, worst, gw))
        for fixture in state.fixtures:
            if fixture.gw is not None and fixture.gw >= gw:
                if fixture.finished or fixture.team_h_score is not None \
                        or fixture.team_a_score is not None:
                    problems.append(
                        "fixture %d in gw %s carries a result" % (fixture.id, fixture.gw))
        if state.finished_gws and max(state.finished_gws) >= gw:
            problems.append("finished_gws contains %d >= gw %d"
                            % (max(state.finished_gws), gw))
        if state.current_gw != gw:
            problems.append("current_gw is %d, expected %d" % (state.current_gw, gw))
        for player in state.players.values():
            if player.ep_next:
                problems.append("player %d carries ep_next, a rival predictor" % player.id)
                break
        return problems[:20]


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


#: Among players who actually took the field, no pre-match forecast ranks
#: realised FPL points much above this. The model manages ~0.38, and published
#: one-week-ahead models land in the same 0.3-0.4 band, because a single
#: gameweek is mostly variance. A "forecast" well clear of it knows something
#: it should not.
EP_CONTAMINATION_RHO = 0.50


def ep_contamination_check(frame: pd.DataFrame) -> Dict[str, Any]:
    """Is the archive's stored expected-points column really a forecast?

    It is supposed to be FPL's own ``ep_this``, captured before the deadline.
    The archive is written *after* each gameweek, so what it stores is whatever
    the API held at scrape time — and FPL keeps recomputing that field. This
    checks the column against the one thing a genuine forecast cannot do: rank
    the realised points of players who played.
    """
    subset = frame[(frame.get("ep_recorded", False) == True)]  # noqa: E712
    per_gw: List[Dict[str, Any]] = []
    for gw, grp in subset.groupby("gw"):
        appeared = grp[grp.get("played", 0) > 0]
        if len(appeared) < 30:
            continue
        per_gw.append({
            "gw": int(gw),
            "ep_rho_among_appeared": M.spearman(
                appeared["baseline_ep"], appeared["actual_points"]),
            "model_rho_among_appeared": M.spearman(
                appeared["xp"], appeared["actual_points"]),
            "ep_rho_vs_realised_minutes": M.spearman(grp["baseline_ep"], grp["minutes"]),
        })
    if not per_gw:
        return {"checked": False, "contaminated": False, "per_gw": []}
    worst = max(row["ep_rho_among_appeared"] for row in per_gw)
    mean_ep = float(np.mean([row["ep_rho_among_appeared"] for row in per_gw]))
    mean_model = float(np.mean([row["model_rho_among_appeared"] for row in per_gw]))
    return {
        "checked": True,
        "contaminated": bool(worst > EP_CONTAMINATION_RHO),
        "threshold": EP_CONTAMINATION_RHO,
        "worst_gw_rho_among_appeared": worst,
        "mean_rho_among_appeared": mean_ep,
        "model_mean_rho_among_appeared": mean_model,
        "per_gw": per_gw,
        "note": (
            "The archive's expected-points column ranks the realised points of "
            "players who played at rho %.2f on average (worst gameweek %.2f), "
            "against %.2f for a model that only sees pre-deadline data. No "
            "pre-match forecast can do that: a single gameweek is mostly "
            "variance. The stored value has been recomputed after the fact, so "
            "baseline (c) is not the number the API offered at the deadline and "
            "is excluded from the pass/fail verdict. It is still printed, "
            "because being beaten by a post-hoc number is worth knowing."
            % (mean_ep, worst, mean_model)),
    }


def _fixture_bonus_check(engine: XPEngine, gw: int) -> Tuple[float, int]:
    """Largest per-fixture expected bonus pot, and how many broke the limit.

    A fixture pays 3+2+1 = 6 only when the top three BPS scores are distinct;
    ties pay 7, 8 or 9, and 2025/26 averaged 6.366 per fixture. The limit here
    is ``bonus_mod.BONUS_POT_SANITY_LIMIT``, not 6 - a breach means the ranker
    is broken, which is what this is guarding, whereas a pot of 6.4 is the game.
    """
    worst = 0.0
    breaches = 0
    for (fixture_id, bundle_gw), bundle in engine._bundles.items():
        if bundle_gw != gw:
            continue
        total = sum(p.xp_bonus for p in bundle.values())
        worst = max(worst, total)
        if total > bonus_mod.BONUS_POT_SANITY_LIMIT + 1e-6:
            breaches += 1
    return worst, breaches


def backtest_season(
    season: str = "2025-26",
    gws: Optional[Sequence[int]] = None,
    config: Optional[Config] = None,
    mc_draws: int = 2000,
    replay: Optional[SeasonReplay] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Replay ``season`` gameweek by gameweek and score the model.

    Returns the full report dict, which is also what lands in
    ``reports/backtest_{season}.json``.
    """
    started = time.time()
    replay = replay or SeasonReplay(season, config=config).load()
    season = replay.season
    gws = [int(g) for g in (gws or replay.available_gws)]
    gws = [g for g in gws if g in replay.available_gws]
    if not gws:
        raise ValueError("no gameweeks to replay for %s" % season)

    # The attacking model's elasticity fit walks whole seasons; pin it to
    # seasons that finish before the one under test.
    elasticity_seasons = [hist.prior_season(season, back=b)
                          for b in range(replay.prior_depth + 1, 0, -1)]

    predictions: List[pd.DataFrame] = []
    per_gw: List[Dict[str, Any]] = []
    audit_violations: List[str] = []
    engine_warnings: Dict[str, int] = {}
    bonus_worst = 0.0
    bonus_breaches = 0
    dropped_future = 0

    with replay_environment("%s" % season, elasticity_seasons):
        for gw in gws:
            t0 = time.time()
            state = replay.state_for_gw(gw)
            problems = replay.leakage_audit(state, gw)
            if problems:
                audit_violations.extend("GW%d: %s" % (gw, p) for p in problems)
                raise RuntimeError(
                    "lookahead detected building GW%d: %s" % (gw, problems[0]))
            build_seconds = time.time() - t0

            t0 = time.time()
            with team_ratings_as_of(replay.deadline(gw)):
                engine = XPEngine(replay.config, mc_draws=mc_draws).fit(state)
            fit_seconds = time.time() - t0
            dropped_future += int(getattr(engine.team, "dropped_future_results", 0))
            for warning in engine.warnings:
                key = warning.split(":")[0]
                engine_warnings[key] = engine_warnings.get(key, 0) + 1

            t0 = time.time()
            projections = engine.project(state, [gw])
            project_seconds = time.time() - t0

            worst, breaches = _fixture_bonus_check(engine, gw)
            bonus_worst = max(bonus_worst, worst)
            bonus_breaches += breaches

            rows: List[Dict[str, Any]] = []
            for pid, per_player in projections.projections.items():
                gwp = per_player.get(gw)
                if gwp is None:
                    continue
                player = state.players[pid]
                p_60 = max((f.p_60 for f in gwp.fixtures), default=0.0)
                p_start = max((f.p_start for f in gwp.fixtures), default=0.0)
                p_appear = max((f.p_appear for f in gwp.fixtures), default=0.0)
                rows.append({
                    "player_id": pid,
                    "gw": gw,
                    "xp": gwp.xp,
                    "sd": gwp.sd,
                    "n_fixtures_pred": len(gwp.fixtures),
                    "p_60": p_60,
                    "p_start": p_start,
                    "p_appear": p_appear,
                    "xmins": sum(f.xmins for f in gwp.fixtures),
                    "position": player.position,
                    "price": player.now_cost / 10.0,
                    "team_id": player.team_id,
                    "web_name": player.web_name,
                    "xp_defcon": sum(f.xp_defcon for f in gwp.fixtures),
                    "xp_bonus": sum(f.xp_bonus for f in gwp.fixtures),
                })
            frame = pd.DataFrame(rows)

            # Baselines, all built from the same pre-deadline information.
            totals = {e: replay._totals(e, gw) for e in frame["player_id"]}
            last_gw_points = {}
            ppg_to_date = {}
            for element in frame["player_id"]:
                cumulative = totals[element]
                previous = replay.cumulative.get(gw - 2, {}).get(
                    element, replay._zero_totals) if gw >= 2 else replay._zero_totals
                last_gw_points[element] = float(
                    cumulative.get("total_points", 0.0) - previous.get("total_points", 0.0))
                matches = float(cumulative.get("_matches", 0.0))
                ppg_to_date[element] = (
                    float(cumulative.get("total_points", 0.0)) / matches if matches > 0 else 0.0)
            frame["baseline_last_gw"] = frame["player_id"].map(last_gw_points)
            # Points per game is a rate, so it has to be multiplied up for a
            # double gameweek or the baseline is handicapped for no reason.
            frame["baseline_ppg"] = (
                frame["player_id"].map(ppg_to_date) * frame["n_fixtures_pred"].clip(lower=0))
            frame["baseline_price"] = frame["price"]

            actual = replay.actuals_for_gw(gw)
            frame["_deadline"] = replay.deadline(gw).isoformat()
            merged = M.merge_pred_actual(frame, actual)
            merged = merged.rename(columns={"ep_next": "baseline_ep"})
            # The archive only captured FPL's own expected points in some
            # gameweeks; where it did not, the column is a column of zeros, and
            # scoring that as a prediction would flatter the model enormously.
            # Mark it absent rather than let a zero masquerade as a forecast.
            ep_recorded = bool((merged["baseline_ep"] > 0).any())
            merged["ep_recorded"] = ep_recorded
            if not ep_recorded:
                merged["baseline_ep"] = np.nan
            predictions.append(merged)

            summary = {
                "gw": gw,
                "players": int(len(merged)),
                "doubles": int((merged["n_fixtures"] > 1).sum()),
                "blanks": int(len(frame) - len(merged)),
                "mean_actual": float(merged["actual_points"].mean()),
                "mean_xp": float(merged["xp"].mean()),
                "rmse": M.rmse(merged["xp"], merged["actual_points"]),
                "spearman": M.spearman(merged["xp"], merged["actual_points"]),
                "top20_hit_rate": M.hit_rate_top_n(
                    merged["xp"], merged["actual_points"], M.TOP_N_DEFAULT),
                "seconds": {"build": round(build_seconds, 2),
                            "fit": round(fit_seconds, 2),
                            "project": round(project_seconds, 2)},
            }
            per_gw.append(summary)
            if progress:
                print("  GW%-3d %4d players  xP %.2f vs actual %.2f  rho %.3f  "
                      "top20 %.2f  (%.1fs)"
                      % (gw, summary["players"], summary["mean_xp"],
                         summary["mean_actual"], summary["spearman"],
                         summary["top20_hit_rate"],
                         build_seconds + fit_seconds + project_seconds))

    frame = pd.concat(predictions, ignore_index=True)
    return _assemble_report(
        replay, frame, gws, per_gw,
        {
            "audit_violations": audit_violations,
            "engine_warnings": engine_warnings,
            "bonus_worst_fixture_total": bonus_worst,
            "bonus_breaches": bonus_breaches,
            "team_ratings_future_results_dropped": dropped_future,
            "mc_draws": mc_draws,
            "elasticity_seasons": elasticity_seasons,
            "total_seconds": round(time.time() - started, 1),
        },
    )


PREDICTORS: Tuple[Tuple[str, str, bool], ...] = (
    ("model", "xp", False),
    ("baseline_last_gw", "baseline_last_gw", False),
    ("baseline_ppg", "baseline_ppg", False),
    ("baseline_ep_next", "baseline_ep", False),
    ("baseline_price", "baseline_price", True),
)

PREDICTOR_LABELS: Dict[str, str] = {
    "model": "gaffer xP",
    "baseline_last_gw": "(a) last GW points",
    "baseline_ppg": "(b) season pts/game",
    "baseline_ep_next": "(c) FPL's own ep",
    "baseline_price": "(d) price (rank only)",
}

# ---------------------------------------------------------------------------
# Which rows the verdict is allowed to rest on
# ---------------------------------------------------------------------------
#
# The full universe is every player with a fixture, and roughly three fifths of
# it is players who never came off the bench and scored exactly zero. Rank
# correlation over that set is mostly a test of "can you predict a zero", which
# the trivial baseline "what he scored last week" also answers with a zero, for
# free and perfectly. A model can therefore lose the full-universe table while
# being strictly more useful, so that table is a diagnostic here and nothing is
# concluded from it.
#
# Every decision an FPL manager makes — transfer, captain, bench order — is made
# among players he expects to play. Those are the rows the model is answerable
# for, and the verdict is judged on them.

#: Subsets the headline verdict is judged on, in the order they are reported.
DECISION_SUBSETS: Tuple[str, ...] = ("appeared", "likely_starters")

#: Every subset the report cuts, decision-relevant or not.
SUBSET_KEYS: Tuple[str, ...] = DECISION_SUBSETS + ("full_universe",)

SUBSET_LABELS: Dict[str, str] = {
    "appeared": "players who actually took the field",
    "likely_starters": "likely starters (model p_start >= 0.5)",
    "full_universe": "every player with a fixture",
}

SUBSET_NOTES: Dict[str, str] = {
    "appeared": (
        "Conditions on the outcome, so it is not a shortlist you could draw up at "
        "the deadline. What it does isolate is the points model: given that a man "
        "played, were his points ranked correctly? A loss here is a football "
        "problem, not a minutes problem."),
    "likely_starters": (
        "Chosen before kickoff, from the model's own p_start, so this is the "
        "ex-ante version of the same question and the closest thing in this report "
        "to a real transfer shortlist. It is the subset that should carry the most "
        "weight."),
    "full_universe": (
        "Diagnostic only. Most of these rows are players who never took the field "
        "and scored zero, and the trivial baselines predict zero for them too, so "
        "they collect rank correlation for free. Nothing is concluded from this "
        "table."),
}

#: Where each subset's predictor table lives in the report dict.
SUBSET_REPORT_KEYS: Dict[str, str] = {
    "appeared": "predictors_appeared_diagnostic",
    "likely_starters": "predictors_likely_starters",
    "full_universe": "predictors",
}


def _decisive_comparison(comparisons: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The one metric a verdict turns on: per-gameweek rank correlation.

    Mirrors ``metrics.verdict``, which prefers the per-gameweek mean and falls
    back to the pooled value. Ranking players within a gameweek is the question
    a manager actually asks; the pooled number also rewards knowing which
    gameweeks are high-scoring, which nobody can act on.
    """
    if not comparisons:
        return None
    return comparisons.get("spearman_per_gw_mean") or comparisons.get("spearman")


def _headline_verdict(
    judgements: Dict[str, Dict[str, Any]],
    sizes: Dict[str, Dict[str, int]],
    degenerate: Dict[str, Any],
    contamination: Dict[str, Any],
) -> Dict[str, Any]:
    """Judge the model on the decision-relevant subsets, and say so out loud.

    Every line carries the subset it came from, because the same model beats the
    baselines on one cut and loses on another, and a verdict that does not name
    its subset is not a verdict. ``beats_all`` is true only if the model beats
    every baseline on *every* decision-relevant subset — losing on one and
    winning on another is a loss.
    """
    lines: List[str] = []
    losses: List[Dict[str, Any]] = []
    basis = [k for k in DECISION_SUBSETS if judgements.get(k, {}).get("detail")]

    if not basis:
        # Only reachable if both decision subsets were too small to score, which
        # means a handful of gameweeks were replayed. Say that rather than fall
        # back to the degenerate table and pretend it settled something.
        return {
            "beats_all": False,
            "basis": [],
            "lines": ["NO VERDICT: neither decision-relevant subset (appeared, "
                      "likely starters) had enough rows to score. Nothing here is "
                      "judged on the full universe, which is degenerate."],
            "losses": [],
            "by_subset": judgements,
            "degenerate_full_universe": degenerate,
        }

    lines.append(
        "JUDGED ON: %s. Transfers and captaincy are chosen among players expected "
        "to play, so those are the rows the model is answerable for. rho/GW below is "
        "the mean within-gameweek rank correlation, which is the metric each "
        "comparison turns on."
        % " and ".join("%s (n=%d)" % (k.replace("_", " "), sizes[k]["rows"]) for k in basis))

    for key in basis:
        judgement = judgements[key]
        size = sizes[key]
        lines.append("")
        lines.append("[%s] %s — %d rows over %d gameweek(s)"
                     % (key.upper(), SUBSET_LABELS[key], size["rows"], size["gameweeks"]))
        lines.append("    " + SUBSET_NOTES[key])
        for baseline, comparisons in judgement["detail"].items():
            decisive = _decisive_comparison(comparisons)
            if decisive is None:
                continue
            better = bool(decisive["model_better"])
            if not better:
                losses.append({
                    "subset": key,
                    "baseline": baseline,
                    "label": PREDICTOR_LABELS.get(baseline, baseline),
                    "model": decisive["model"],
                    "baseline_value": decisive["baseline"],
                    "delta": decisive["delta"],
                })
            lines.append(
                "    vs %-22s %-6s rho/GW %.4f vs %.4f (%+.4f), wins %d/%d metrics"
                % (PREDICTOR_LABELS.get(baseline, baseline),
                   "BEATS" if better else "LOSES",
                   decisive["model"], decisive["baseline"], decisive["delta"],
                   comparisons.get("_won", 0), comparisons.get("_of", 0)))
        lines.append("    => on this subset the model %s every baseline"
                     % ("BEATS" if judgement["beats_all"] else "DOES NOT BEAT"))

    # FPL's own ep is scored on the full universe, over the gameweeks the archive
    # recorded it, and only counts when the contamination check clears it. That
    # analysis is untouched; this only decides whether its result is allowed to
    # move the headline.
    ep_entry = (judgements.get("full_universe") or {}).get("detail", {}).get("baseline_ep_next")
    ep_clean = bool(contamination.get("checked")) and not contamination.get("contaminated")
    ep_ok = True
    if ep_entry is not None:
        lines.append("")
        if not contamination.get("checked"):
            lines.append("[EP] (c) FPL's own ep was not scored: the archive recorded no "
                         "usable expected-points column.")
        elif not ep_clean:
            lines.append(
                "[EP] (c) FPL's own ep is EXCLUDED from the verdict: the contamination "
                "check shows the archived value was recomputed after the gameweek was "
                "played, so it is not the forecast the API offered at the deadline. Its "
                "comparison is still printed under the full-universe diagnostic.")
        else:
            decisive = _decisive_comparison(ep_entry)
            if decisive is not None:
                better = bool(decisive["model_better"])
                if not better:
                    ep_ok = False
                    losses.append({
                        "subset": "ep_gameweeks",
                        "baseline": "baseline_ep_next",
                        "label": PREDICTOR_LABELS["baseline_ep_next"],
                        "model": decisive["model"],
                        "baseline_value": decisive["baseline"],
                        "delta": decisive["delta"],
                    })
                lines.append(
                    "[EP_GAMEWEEKS] vs %s %s rho/GW %.4f vs %.4f (%+.4f), on the full "
                    "universe restricted to the gameweeks the archive recorded ep"
                    % (PREDICTOR_LABELS["baseline_ep_next"], "BEATS" if better else "LOSES",
                       decisive["model"], decisive["baseline"], decisive["delta"]))

    beats_all = bool(all(judgements[k]["beats_all"] for k in basis) and ep_ok)

    lines.append("")
    if beats_all:
        lines.append("HEADLINE: the model BEATS every baseline on every decision-relevant "
                     "subset (%s)." % ", ".join(k.replace("_", " ") for k in basis))
    else:
        lines.append("HEADLINE: the model does NOT beat every baseline where the decisions "
                     "are made.")
        for loss in losses:
            lines.append("  on %s it loses to %s on rho/GW: %.4f vs %.4f (%+.4f)."
                         % (loss["subset"].replace("_", " "), loss["label"],
                            loss["model"], loss["baseline_value"], loss["delta"]))
        won = [k for k in basis if judgements[k]["beats_all"]]
        lost = [k for k in basis if not judgements[k]["beats_all"]]
        if won and lost:
            lines.append(
                "  It does beat every baseline on %s, but that subset conditions on the "
                "outcome. Winning there and losing on %s means the points ranking is "
                "sound and the ex-ante shortlist is not; the second is what a transfer "
                "is actually picked from, so the honest reading is that the model is not "
                "yet proven."
                % (", ".join(k.replace("_", " ") for k in won),
                   ", ".join(k.replace("_", " ") for k in lost)))
        lines.append("  Do not report this model as beating the baselines.")

    if degenerate.get("rows"):
        lines.append("")
        lines.append(
            "[FULL_UNIVERSE] diagnostic, not judged: %d of %d rows (%.0f%%) are players who "
            "did not take the field, and %d of those scored exactly zero. The last-gameweek "
            "baseline also says zero for %.0f%% of them, so it ranks that pile correctly for "
            "free. That is why the full-universe table can show the model losing to a "
            "baseline nobody would use, and why nothing is concluded from it."
            % (degenerate["rows"], degenerate["total_rows"], 100.0 * degenerate["share"],
               degenerate["rows_scoring_zero"],
               100.0 * degenerate["share_last_gw_baseline_zero"]))
        for line in (judgements.get("full_universe") or {}).get("lines", []):
            lines.append("    " + line)

    return {
        "beats_all": beats_all,
        "basis": basis,
        "lines": lines,
        "losses": losses,
        "by_subset": judgements,
        "degenerate_full_universe": degenerate,
    }


def _comparable_gameweeks(
    frame: pd.DataFrame, predictors: Sequence[Tuple[str, str, bool]]
) -> List[int]:
    """Gameweeks in which every predictor produces more than one distinct value."""
    out: List[int] = []
    for gw, grp in frame.groupby("gw"):
        usable = True
        for _, column, _ in predictors:
            if column not in grp.columns:
                continue
            values = grp[column].dropna()
            if values.empty or float(values.std(ddof=0)) <= 0:
                usable = False
                break
        if usable:
            out.append(int(gw))
    return sorted(out)


def _bonus_component_report(frame: pd.DataFrame) -> Dict[str, Any]:
    """Score the bonus component on its own, against bonus actually awarded.

    Bonus is worth up to 3 points a match but only about a tenth of a projected
    total, so a change to the bonus model that is real is still swamped in the
    headline rank correlation by the other ninety percent. Scored directly
    against the ``bonus`` column it is measurable: the season total says whether
    the level is right, the per-gameweek rank correlation and top-20 hit rate
    say whether the ordering is, and the appeared-only cut removes the ~17k rows
    where predicting zero bonus is trivially correct.
    """
    if "xp_bonus" not in frame.columns or "bonus" not in frame.columns:
        return {}
    out: Dict[str, Any] = {}
    for label, sub in (
        ("all", frame),
        ("appeared", frame[frame.get("played", 0) > 0]),
        ("likely_starters", frame[frame["p_start"] >= 0.5]),
    ):
        if len(sub) < 50:
            continue
        pred = pd.to_numeric(sub["xp_bonus"], errors="coerce").fillna(0.0).values
        actual = pd.to_numeric(sub["bonus"], errors="coerce").fillna(0.0).values
        per_gw_rho: List[float] = []
        per_gw_hit: List[float] = []
        for _, grp in sub.groupby("gw"):
            if len(grp) < 2:
                continue
            p = pd.to_numeric(grp["xp_bonus"], errors="coerce").fillna(0.0)
            a = pd.to_numeric(grp["bonus"], errors="coerce").fillna(0.0)
            per_gw_rho.append(M.spearman(p, a))
            per_gw_hit.append(M.hit_rate_top_n(p, a, 20))
        good = [v for v in per_gw_rho if v == v]
        out[label] = {
            "n": int(len(sub)),
            "spearman": M.spearman(pred, actual),
            "spearman_per_gw_mean": float(np.mean(good)) if good else float("nan"),
            "spearman_per_gw": [round(v, 6) if v == v else None for v in per_gw_rho],
            "rmse": M.rmse(pred, actual),
            "mae": M.mae(pred, actual),
            "predicted_total": float(np.sum(pred)),
            "actual_total": float(np.sum(actual)),
            "top20_hit_rate": float(np.mean(per_gw_hit)) if per_gw_hit else float("nan"),
            "top20_hit_rate_per_gw": [round(v, 4) for v in per_gw_hit],
        }
    return out


def _assemble_report(
    replay: SeasonReplay,
    frame: pd.DataFrame,
    gws: Sequence[int],
    per_gw: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    predictors = [p for p in PREDICTORS if p[1] in frame.columns]
    # Baselines (a) and (b) are constant zero until some football has been
    # played, which makes their rank correlation undefined at GW1. Scoring a
    # predictor on gameweeks where a rival is undefined is not a comparison, so
    # the headline tables run over the gameweeks where every predictor has
    # something to say. GW1 still appears in the per-gameweek table.
    comparable = _comparable_gameweeks(frame, [p for p in predictors
                                               if p[0] != "baseline_ep_next"])
    skipped = sorted(set(int(g) for g in frame["gw"]) - set(comparable))
    if comparable:
        frame = frame[frame["gw"].isin(comparable)]
    # FPL's own ep is only in the archive for some gameweeks. Comparing a
    # predictor scored on 12 gameweeks against one scored on 38 is not a
    # comparison, so it comes out of the headline table and gets its own,
    # where every predictor is restricted to the same gameweeks.
    ep_frame = frame[frame.get("ep_recorded", False) == True]  # noqa: E712
    ep_gws = sorted(int(g) for g in ep_frame["gw"].unique()) if len(ep_frame) else []
    headline_predictors = [p for p in predictors if p[0] != "baseline_ep_next"]
    overall = M.full_report(frame, headline_predictors)
    ep_report = (M.full_report(ep_frame, predictors)
                 if len(ep_gws) and len(ep_gws) < len(set(frame["gw"]))
                 else (M.full_report(frame, predictors) if len(ep_gws) else {}))

    # The full universe is dominated by players nobody would consider owning;
    # the cut over likely starters is the ex-ante shortlist a transfer is made
    # from, and it is one of the two subsets the verdict is judged on.
    starters = frame[frame["p_start"] >= 0.5]
    starters_report = M.full_report(starters, headline_predictors) if len(starters) > 50 else {}

    # Conditions on the outcome, so it is not a set anyone could pick from at the
    # deadline — but restricted to players who really did play it separates "can
    # it rank points" from "can it predict who plays", and that is a decision the
    # model has to get right. Judged, with the caveat stated every time.
    appeared = frame[frame.get("played", 0) > 0]
    appeared_report = (M.full_report(appeared, headline_predictors)
                       if len(appeared) > 50 else {})

    calibration = {
        "p_60": M.calibration(frame, "p_60", "played_60"),
        "p_appear": M.calibration(frame, "p_appear", "played"),
    }
    bonus_component = _bonus_component_report(frame)
    if "started" in frame.columns:
        calibration["p_start"] = M.calibration(frame, "p_start", "started")

    # The full-universe judgement. Kept in full, but demoted to a diagnostic:
    # see DECISION_SUBSETS for why nothing is concluded from it.
    judgement = M.verdict(overall)
    contamination = ep_contamination_check(frame) if ep_gws else {
        "checked": False, "contaminated": False, "per_gw": []}
    if ep_report:
        ep_judgement = M.verdict(ep_report)
        entry = ep_judgement["detail"].get("baseline_ep_next")
        if entry is not None:
            judgement["detail"]["baseline_ep_next"] = entry
            if not contamination.get("contaminated"):
                judgement["beats_all"] = bool(
                    judgement["beats_all"] and ep_judgement["beats_all"])
            for line in ep_judgement["lines"]:
                if line.startswith("baseline_ep_next"):
                    judgement["lines"].append(
                        "%s  [%d gameweek(s) recorded%s]"
                        % (line, len(ep_gws),
                           "; EXCLUDED from the verdict, see contamination check"
                           if contamination.get("contaminated") else ""))

    # How degenerate the full universe actually is, measured rather than
    # asserted: the non-appearance rows, what they scored, and how often the
    # trivial baseline gets them right by predicting zero.
    non_appearance = frame[frame.get("played", 0) <= 0]
    degenerate = {
        "rows": int(len(non_appearance)),
        "total_rows": int(len(frame)),
        "share": float(len(non_appearance) / len(frame)) if len(frame) else float("nan"),
        "mean_actual_points": (float(non_appearance["actual_points"].mean())
                               if len(non_appearance) else float("nan")),
        # Exact equality, not "<= 0": a handful of these rows carry a negative
        # score, and rounding them in would turn 99.99% into a false 100%.
        "rows_scoring_zero": int((non_appearance["actual_points"] == 0).sum()),
        "share_last_gw_baseline_zero": (
            float((non_appearance["baseline_last_gw"] == 0).mean())
            if len(non_appearance) and "baseline_last_gw" in non_appearance.columns
            else float("nan")),
    }

    subset_frames = {"appeared": appeared, "likely_starters": starters, "full_universe": frame}
    subset_reports = {"appeared": appeared_report, "likely_starters": starters_report,
                      "full_universe": overall}
    sizes = {
        key: {"rows": int(len(sub)), "gameweeks": int(sub["gw"].nunique()),
              "players": int(sub["player_id"].nunique())}
        for key, sub in subset_frames.items()
    }
    judgements = {
        "appeared": M.verdict(appeared_report) if appeared_report else {},
        "likely_starters": M.verdict(starters_report) if starters_report else {},
        "full_universe": judgement,
    }
    headline = _headline_verdict(judgements, sizes, degenerate, contamination)
    subsets = [
        {
            "key": key,
            "label": SUBSET_LABELS[key],
            "note": SUBSET_NOTES[key],
            "predictors_key": SUBSET_REPORT_KEYS[key],
            "decision_relevant": key in DECISION_SUBSETS,
            "judged": key in headline["basis"],
            **sizes[key],
        }
        for key in SUBSET_KEYS if subset_reports[key]
    ]

    report: Dict[str, Any] = {
        "season": replay.season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gws": list(gws),
        "universe": {
            "rows": int(len(frame)),
            "players": int(frame["player_id"].nunique()),
            "gameweeks": int(frame["gw"].nunique()),
            "double_gameweek_rows": int((frame["n_fixtures"] > 1).sum()),
            "likely_starter_rows": int(len(starters)),
            "appeared_rows": int(len(appeared)),
            "non_appearance_rows": int(len(non_appearance)),
            "comparison_gameweeks": comparable,
            "gameweeks_excluded_from_comparison": skipped,
        },
        "no_lookahead": {
            "enforcement": [
                "history sliced to round < gw",
                "season-to-date totals cumulated to gw-1",
                "prices and clubs read from the gw-1 row",
                "fixtures at gw >= current carry no score and finished=False",
                "TeamRatingModel.fit given as_of=deadline(gw)",
                "fitted-parameter caches redirected away from reports/",
            ],
            "audit_violations": meta["audit_violations"],
            "team_ratings_future_results_dropped": meta[
                "team_ratings_future_results_dropped"],
            "attacking_fit_seasons": meta["elasticity_seasons"],
        },
        "predictors": overall,
        "predictors_likely_starters": starters_report,
        "predictors_ep_gameweeks": ep_report,
        "predictors_appeared_diagnostic": appeared_report,
        "ep_gameweeks": ep_gws,
        "ep_contamination": contamination,
        "calibration": calibration,
        "bonus_component": bonus_component,
        "subsets": subsets,
        # The headline verdict, judged on the decision-relevant subsets only.
        # `verdict["by_subset"]["full_universe"]` is the old full-universe
        # judgement, unchanged, for anything that still wants it.
        "verdict": headline,
        "verdict_full_universe": judgement,
        "per_gw": per_gw,
        "invariants": {
            "bonus_worst_fixture_total": meta["bonus_worst_fixture_total"],
            "bonus_pot_limit": bonus_mod.BONUS_POT_SANITY_LIMIT,
            "bonus_breaches_over_limit": meta["bonus_breaches"],
        },
        "warnings": replay.warnings,
        "engine_warning_counts": meta["engine_warnings"],
        "limitations": list(LIMITATIONS),
        "timings": {"load_seconds": round(replay.load_seconds, 2),
                    "total_seconds": meta["total_seconds"],
                    "mc_draws": meta["mc_draws"]},
    }
    return report


def report_path(season: str) -> str:
    return os.path.join(REPORTS_DIR, "backtest_%s.json" % hist.season_dash(season))


def write_report(report: Dict[str, Any], path: Optional[str] = None) -> str:
    ensure_dirs()
    path = path or report_path(report["season"])
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
# Readable summary
# ---------------------------------------------------------------------------


#: Widest line the tables produce (``metrics.comparison_table``'s header), so
#: prose wraps to the same right edge instead of a narrower one.
RENDER_WIDTH = 94


def _rule_heading(title: str) -> str:
    text = "--- %s " % title
    return text + "-" * max(3, RENDER_WIDTH - len(text))


def _wrap(text: str, indent: str = "", hang: Optional[str] = None) -> List[str]:
    """Wrap prose to ``RENDER_WIDTH``, keeping the line's own indentation.

    A line whose body contains a run of two or more spaces is column-padded —
    a table row — and is emitted verbatim however long it is, because breaking
    it would split a number across two lines and destroy the alignment that
    makes it readable.
    """
    if not text:
        return []
    own = text[: len(text) - len(text.lstrip())]
    body = text.strip()
    prefix = indent + own
    following = hang if hang is not None else prefix + "  "
    if "  " in body or len(prefix) + len(body) <= RENDER_WIDTH:
        return [prefix + body]
    return textwrap.wrap(body, width=RENDER_WIDTH, initial_indent=prefix,
                         subsequent_indent=following) or [prefix + body]


def _subsets_for_render(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The subset descriptors, newest report shape first, older ones rebuilt."""
    subsets = report.get("subsets")
    if subsets:
        return list(subsets)
    universe = report.get("universe") or {}
    rebuilt: List[Dict[str, Any]] = []
    for key in SUBSET_KEYS:
        table = report.get(SUBSET_REPORT_KEYS[key]) or {}
        if not table:
            continue
        rebuilt.append({
            "key": key, "label": SUBSET_LABELS[key], "note": SUBSET_NOTES[key],
            "predictors_key": SUBSET_REPORT_KEYS[key],
            "decision_relevant": key in DECISION_SUBSETS,
            "judged": key in DECISION_SUBSETS,
            "rows": int((table.get("model") or {}).get("n", 0)),
            "players": int(universe.get("players", 0)),
            "gameweeks": int((table.get("model") or {}).get("n_gws", 0)),
        })
    return rebuilt


def _compact_range(gws: Sequence[int]) -> str:
    """[1,2,3,7] -> '1-3, 7'."""
    values = sorted(int(g) for g in gws)
    if not values:
        return "none"
    spans: List[Tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        spans.append((start, previous))
        start = previous = value
    spans.append((start, previous))
    return ", ".join("%d" % lo if lo == hi else "%d-%d" % (lo, hi) for lo, hi in spans)


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    head = "BACKTEST %s — gameweeks %s" % (
        report["season"],
        "%d-%d" % (min(report["gws"]), max(report["gws"])) if report["gws"] else "none")
    lines.append("=" * 78)
    lines.append(head)
    lines.append("=" * 78)
    universe = report["universe"]
    lines.append(
        "%d player-gameweeks over %d gameweeks (%d players, %d double-gameweek rows, "
        "%d likely starters)"
        % (universe["rows"], universe["gameweeks"], universe["players"],
           universe["double_gameweek_rows"], universe["likely_starter_rows"]))
    if universe.get("gameweeks_excluded_from_comparison"):
        lines.append(
            "comparison tables cover gameweek(s) %s only; %s excluded because a "
            "baseline has no data there (no football played yet)"
            % (_compact_range(universe["comparison_gameweeks"]),
               _compact_range(universe["gameweeks_excluded_from_comparison"])))
    lines.append("")

    # Decision-relevant subsets first, full universe last: the order the verdict
    # reads them in, so nobody has to work out which table was judged.
    for index, subset in enumerate(_subsets_for_render(report)):
        table = report.get(subset["predictors_key"]) or {}
        if not table:
            continue
        if index:
            lines.append("")
        lines.append(_rule_heading(
            "%s: %s" % ("JUDGED" if subset.get("judged") else "DIAGNOSTIC",
                        subset["label"].upper())))
        lines.append("    %d rows, %d players, %d gameweek(s)"
                     % (subset.get("rows", 0), subset.get("players", 0),
                        subset.get("gameweeks", 0)))
        lines.extend(_wrap(subset.get("note", ""), "    "))
        lines.append(M.comparison_table(table, PREDICTOR_LABELS))
    if report.get("predictors_ep_gameweeks"):
        lines.append("")
        lines.append("--- vs FPL'S OWN ep, ON THE %d GAMEWEEK(S) THE ARCHIVE RECORDED IT %s"
                     % (len(report["ep_gameweeks"]), "-" * 12))
        lines.append("    (gameweeks %s; the archive stores 0.0 for the rest, which is "
                     "absence, not a forecast)"
                     % ", ".join(str(g) for g in report["ep_gameweeks"]))
        lines.append(M.comparison_table(report["predictors_ep_gameweeks"], PREDICTOR_LABELS))
        contamination = report.get("ep_contamination") or {}
        if contamination.get("checked"):
            lines.append("")
            lines.append("    CONTAMINATION CHECK: %s"
                         % ("FAILED — the stored value is not a pre-deadline forecast"
                            if contamination["contaminated"] else "passed"))
            lines.append("    " + contamination["note"])
            lines.append("    %-5s %14s %16s %14s" % (
                "GW", "ep rho|played", "model rho|played", "ep rho v mins"))
            for row in contamination["per_gw"]:
                lines.append("    %-5d %14.3f %16.3f %14.3f" % (
                    row["gw"], row["ep_rho_among_appeared"],
                    row["model_rho_among_appeared"], row["ep_rho_vs_realised_minutes"]))

    lines.append("")
    lines.append(_rule_heading("VERDICT"))
    verdict = report.get("verdict") or {}
    for line in verdict.get("lines", []):
        if not line:
            lines.append("")
            continue
        lines.extend(_wrap(line, "  "))

    lines.append("")
    lines.append("--- MODEL BY POSITION " + "-" * 55)
    lines.append(M.split_table(report["predictors"], "model", "by_position", PREDICTOR_LABELS))
    lines.append("")
    lines.append("--- MODEL BY PRICE TIER " + "-" * 53)
    lines.append(M.split_table(report["predictors"], "model", "by_price_tier", PREDICTOR_LABELS))

    captain = report["predictors"]["model"]["captain"]
    lines.append("")
    lines.append("--- CAPTAINCY " + "-" * 63)
    lines.append("  highest-projected player was the top scorer in %d of %d gameweeks (%.0f%%)"
                 % (round(captain["accuracy"] * captain["n_gws"]), captain["n_gws"],
                    100 * captain["accuracy"]))
    lines.append("  mean captain haul %.2f vs perfect captain %.2f; %.2f points lost per "
                 "gameweek (%.2f with the armband doubled)"
                 % (captain["mean_captain_points"], captain["mean_perfect_points"],
                    captain["mean_points_lost"], captain["mean_points_lost_doubled"]))
    for key, entry in report["predictors"].items():
        if key == "model":
            continue
        lines.append("  %-22s %.0f%% accuracy, %.2f points lost per gameweek"
                     % (PREDICTOR_LABELS.get(key, key),
                        100 * entry["captain"]["accuracy"],
                        entry["captain"]["mean_points_lost"]))

    lines.append("")
    lines.append("--- MINUTES CALIBRATION " + "-" * 53)
    lines.append(M.calibration_table(report["calibration"]["p_60"], "p60 bucket"))
    lines.append("")
    lines.append(M.calibration_table(report["calibration"]["p_appear"], "p_appear"))

    lines.append("")
    lines.append("--- PER GAMEWEEK " + "-" * 60)
    header = "%-5s %7s %7s %8s %8s %8s %7s" % (
        "GW", "n", "mean xP", "mean act", "rho", "top20", "sec")
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["per_gw"]:
        seconds = sum(row["seconds"].values())
        lines.append("%-5d %7d %7.2f %8.2f %8.3f %8.2f %7.1f" % (
            row["gw"], row["players"], row["mean_xp"], row["mean_actual"],
            row["spearman"], row["top20_hit_rate"], seconds))

    lines.append("")
    lines.append("--- NO-LOOKAHEAD ENFORCEMENT " + "-" * 48)
    for rule in report["no_lookahead"]["enforcement"]:
        lines.append("  * " + rule)
    lines.append("  audit violations: %d" % len(report["no_lookahead"]["audit_violations"]))
    lines.append("  team-rating results dated after the deadline and dropped: %d"
                 % report["no_lookahead"]["team_ratings_future_results_dropped"])
    lines.append("  attacking elasticity fitted on: %s"
                 % ", ".join(report["no_lookahead"]["attacking_fit_seasons"]))

    lines.append("")
    lines.append("--- INVARIANTS " + "-" * 62)
    lines.append("  worst per-fixture expected bonus pot: %.4f (limit %.1f), breaches: %d"
                 % (report["invariants"]["bonus_worst_fixture_total"],
                    bonus_mod.BONUS_POT_SANITY_LIMIT,
                    report["invariants"]["bonus_breaches_over_limit"]))

    lines.append("")
    lines.append("--- WHAT THE REPLAY CANNOT SEE " + "-" * 46)
    for item in report["limitations"]:
        lines.append("  * " + item)
    if report["warnings"]:
        lines.append("")
        lines.append("--- DATA WARNINGS " + "-" * 59)
        for item in report["warnings"]:
            lines.append("  ! " + item)
    if report["engine_warning_counts"]:
        lines.append("  engine warnings by source: %s"
                     % ", ".join("%s x%d" % kv
                                 for kv in sorted(report["engine_warning_counts"].items())))
    lines.append("")
    lines.append("timings: %.1fs total, %.1fs loading, %d Monte Carlo draws per fixture"
                 % (report["timings"]["total_seconds"], report["timings"]["load_seconds"],
                    report["timings"]["mc_draws"]))
    return "\n".join(lines)


def evaluate_projections(pred: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
    """Re-exported from ``metrics`` so the spec's two functions share a module."""
    return M.evaluate_projections(pred, actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gaffer backtest")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--gws", default="", help="e.g. 1-38 or 5-12; default all")
    parser.add_argument("--draws", type=int, default=2000,
                        help="Monte Carlo draws per fixture inside the xP engine")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    if args.gws:
        lo, _, hi = args.gws.partition("-")
        WANTED: Optional[List[int]] = list(range(int(lo), int(hi or lo) + 1))
    else:
        WANTED = None

    print("loading %s..." % args.season)
    REPLAY = SeasonReplay(args.season).load()
    print("  %d gameweeks, %d players, %d fixtures, %d teams (%.1fs)"
          % (len(REPLAY.available_gws), len(REPLAY.element_meta), len(REPLAY.fixtures),
             len(REPLAY.teams), REPLAY.load_seconds))
    print("replaying...")
    REPORT = backtest_season(args.season, WANTED, mc_draws=args.draws, replay=REPLAY)
    print()
    print(format_report(REPORT))
    if not args.no_write:
        PATH = write_report(REPORT)
        print("\nwrote %s (%.1f KB)" % (PATH, os.path.getsize(PATH) / 1024.0))
