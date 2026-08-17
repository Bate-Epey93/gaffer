"""The expected-points assembler.

Every other model in ``gaffer/model`` produces one slice of a player's
gameweek: how long he plays, how many goals his team scores, how likely a
clean sheet is, whether he clears the defensive-contribution threshold, how
much bonus he takes. This module owns the join.

Three things make that join non-trivial and they drive the structure below.

1. **Bonus is contested across a fixture, not within a player.** A player's
   expected bonus depends on all twenty-two players' projected BPS, so nothing
   can be finalised player-by-player. The unit of work here is therefore a
   *fixture bundle*: every candidate on both teams is projected to the point
   just before bonus, the fixture's bonus is allocated once, and only then are
   totals closed. It is also why the run is fast — the ~50 players sharing a
   fixture share one match forecast, one bonus simulation and one Monte Carlo.

2. **A gameweek is not a fixture.** A double gameweek gives a player two
   ``PlayerFixtureProjection`` rows that sum, a blank gives him none. Every
   lookup here goes through the fixture list rather than assuming one match per
   club, and ``project`` emits a zero-xp blank row instead of a missing key.

3. **The captain question is about the tail, not the mean.** Every component's
   variance is derived in closed form — Poisson for goals, assists and saves,
   Bernoulli for the clean sheet and DEFCON, the exact first two moments of the
   goals-conceded term and its exact covariance with the clean sheet, the rank
   distribution for bonus — and those terms are what the breakdown prints. They
   are not the whole story, because the components share a minutes draw and
   bonus rides on the same events, so a single measured *coupling* term closes
   the gap between the independent sum and the truth. Both that term and
   ``p_haul`` = P(points >= 10) come from a seeded Monte Carlo that reproduces
   the real dependence rather than assuming it away: the simulated BPS moves
   with the simulated goals through ``bonus.py``'s own fitted weights, so a
   two-goal draw takes three bonus about as often as it does in the game.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import scoring, stats
from gaffer.core.config import CACHE_DIR, Config
from gaffer.core.types import (
    Fixture,
    MatchForecast,
    MinutesForecast,
    Player,
    PlayerFixtureProjection,
    PlayerGWProjection,
    ProjectionSet,
)
from gaffer.model.attacking import AttackingModel
from gaffer.model.bonus import BonusModel
from gaffer.model.defcon import DefconModel
from gaffer.model.defending import DefendingModel, minutes_split
from gaffer.model.minutes import MinutesModel
from gaffer.model.team_ratings import TeamRatingModel

if TYPE_CHECKING:  # pragma: no cover - types only
    from gaffer.data.loaders import GameState

log = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

# Haul threshold used for captaincy. FPL managers call 10+ a haul and that is
# what CaptainOption.p_haul is documented as.
HAUL_POINTS = 10

# Support of the simulated points distribution. A player cannot lose more than
# a red card plus a couple of own goals plus conceded points in one match, and
# nobody has ever scored 45 in a single fixture; both ends are clipped and the
# clipping is asserted to be empty in the smoke test.
DIST_MIN, DIST_MAX = -15, 45
DIST_SIZE = DIST_MAX - DIST_MIN + 1

# Monte-Carlo draws per player-fixture. At p_haul ~ 0.15 this gives a standard
# error of 0.006, which is finer than the model error by an order of magnitude.
DEFAULT_MC_DRAWS = 4000
MC_SEED_BASE = 1_000_003

# Truncation of the goals-conceded distribution.
MAX_CONCEDED = 12

# Cards are rare enough that one season of a rotation player's appearances is a
# handful of events, so the season before last is pooled in at half weight to
# halve the sampling error. Older seasons are dropped: disciplinary records
# drift with role and referee instructions.
CARD_SEASONS_BACK = 2
CARD_OLDER_SEASON_WEIGHT = 0.5
# Pseudo-90s of the position baseline mixed into every player's card rate.
CARD_SHRINK_90S = 12.0


# ---------------------------------------------------------------------------
# The shared per-fixture context
# ---------------------------------------------------------------------------


@dataclass
class FixtureContext:
    """Everything a component model needs about one player in one fixture.

    The five model modules accept this object (``attacking``, ``defending``,
    ``defcon`` and ``bonus`` all read it under ``TYPE_CHECKING``), so the field
    names are load-bearing: ``team_lambda`` is always the goals the *player's
    own* team is expected to score and ``opponent_lambda`` always the goals it
    is expected to concede, whichever end of the fixture he is on.
    """

    fixture: Fixture
    player: Player
    is_home: bool
    team_lambda: float
    opponent_lambda: float
    forecast: MatchForecast
    minutes: MinutesForecast

    @property
    def gw(self) -> int:
        return int(self.fixture.gw) if self.fixture.gw is not None else 0

    @property
    def team_id(self) -> int:
        return int(self.player.team_id)

    @property
    def opponent_id(self) -> int:
        return int(self.fixture.opponent_of(self.player.team_id))

    @property
    def difficulty(self) -> int:
        return int(
            self.fixture.team_h_difficulty if self.is_home else self.fixture.team_a_difficulty
        )


class _BonusView(object):
    """Projection adaptor handed to ``bonus.py``.

    ``bonus.resolve_events`` documents its own contract: the ``p_clean_sheet``
    it is given must be the *team* clean-sheet probability, ungated by minutes,
    because it applies the 60-minute gate itself. ``defending.project`` (and so
    ``PlayerFixtureProjection.p_clean_sheet``) reports the gated,
    player-earns-it probability instead. Passing the projection straight
    through would gate twice and silently shave bonus off every defender, so
    the ungated number travels in this adaptor and the field on the dataclass
    keeps the meaning the dashboard needs.

    Deliberately carries no ``cbi``/``recoveries``/``tackles``: those are not
    projected here, and leaving them absent is what makes ``bonus.py`` fall
    back to its own fitted per-90 rates rather than reading a zero.
    """

    __slots__ = (
        "p_appear", "p_60", "xmins", "lambda_goals", "lambda_assists",
        "p_clean_sheet", "lambda_saves", "lambda_conceded", "lambda_pen_saves",
        "lambda_pen_miss", "p_yellow", "p_red", "p_own_goal",
    )

    def __init__(self, **kwargs: float) -> None:
        for name in self.__slots__:
            setattr(self, name, float(kwargs.get(name, 0.0)))


# ---------------------------------------------------------------------------
# Cards and own goals
# ---------------------------------------------------------------------------


@dataclass
class CardRates:
    """Per-90 disciplinary rates for one player."""

    player_id: int
    position: int
    yellow90: float = 0.0
    red90: float = 0.0
    own_goal90: float = 0.0
    n90: float = 0.0
    source: str = "baseline"

    def reason(self) -> str:
        return "%.3f yellow/90, %.4f red/90 from %.1f 90s (%s)" % (
            self.yellow90, self.red90, self.n90, self.source)


@dataclass
class CardFit:
    """Position baselines plus every player's shrunk rate."""

    baseline_yellow90: Dict[int, float] = field(default_factory=dict)
    baseline_red90: Dict[int, float] = field(default_factory=dict)
    baseline_own_goal90: Dict[int, float] = field(default_factory=dict)
    rates: Dict[int, CardRates] = field(default_factory=dict)
    seasons_used: List[str] = field(default_factory=list)
    n90_total: float = 0.0
    source: str = ""


def fit_card_rates(state: "GameState", config: Config) -> CardFit:
    """Yellow, red and own-goal rates per 90, from the game state alone.

    Nothing else in the engine models discipline, and a yellow is worth as much
    as half a clean sheet to a defender, so it cannot be dropped. Everything
    needed is already in ``history_past`` and the current-season history that
    ``load_game_state`` fetches, so this costs no extra I/O.
    """
    mc = config.model
    w_prior = float(mc.weight_prior_season)
    w_current = float(mc.weight_season_to_date + mc.weight_recent_form)
    if w_prior + w_current <= 0:
        w_prior, w_current = 0.5, 0.5

    # First pass: weighted totals per player, and the position pools.
    totals: Dict[int, Dict[str, float]] = {}
    pool: Dict[int, Dict[str, float]] = {
        pos: {"minutes": 0.0, "yellow": 0.0, "red": 0.0, "own": 0.0} for pos in scoring.POSITIONS
    }
    seasons_used: List[str] = []

    for pid, player in state.players.items():
        acc = {"minutes": 0.0, "yellow": 0.0, "red": 0.0, "own": 0.0, "raw90": 0.0}
        current = state.history(pid)
        if current is not None and len(current) > 0:
            cur_minutes = float(_col_sum(current, "minutes"))
            if cur_minutes > 0:
                acc["minutes"] += w_current * cur_minutes
                acc["yellow"] += w_current * _col_sum(current, "yellow_cards")
                acc["red"] += w_current * _col_sum(current, "red_cards")
                acc["own"] += w_current * _col_sum(current, "own_goals")
                acc["raw90"] += cur_minutes / 90.0

        rows = state.history_past(pid) or []
        for depth, row in enumerate(reversed(rows[-CARD_SEASONS_BACK:])):
            minutes = _num(row.get("minutes"))
            if minutes <= 0:
                continue
            weight = w_prior * (CARD_OLDER_SEASON_WEIGHT ** depth)
            acc["minutes"] += weight * minutes
            acc["yellow"] += weight * _num(row.get("yellow_cards"))
            acc["red"] += weight * _num(row.get("red_cards"))
            acc["own"] += weight * _num(row.get("own_goals"))
            acc["raw90"] += (minutes / 90.0) * (CARD_OLDER_SEASON_WEIGHT ** depth)
            name = row.get("season_name")
            if name and name not in seasons_used:
                seasons_used.append(str(name))

        totals[pid] = acc
        pos = player.position
        for key in ("minutes", "yellow", "red", "own"):
            pool[pos][key] += acc[key]

    fit = CardFit(seasons_used=sorted(seasons_used))
    for pos in scoring.POSITIONS:
        minutes = pool[pos]["minutes"]
        if minutes > 0:
            fit.baseline_yellow90[pos] = pool[pos]["yellow"] / minutes * 90.0
            fit.baseline_red90[pos] = pool[pos]["red"] / minutes * 90.0
            fit.baseline_own_goal90[pos] = pool[pos]["own"] / minutes * 90.0
        else:
            # Only reachable with no history at all (a brand new game state).
            fit.baseline_yellow90[pos] = float(mc.default_yellow_rate_per_90)
            fit.baseline_red90[pos] = 0.0
            fit.baseline_own_goal90[pos] = 0.0
        fit.n90_total += minutes / 90.0

    fit.source = "history_past+current (%s)" % (", ".join(fit.seasons_used) or "none")
    for pid, player in state.players.items():
        acc = totals[pid]
        pos = player.position
        n90 = acc["raw90"]
        if acc["minutes"] > 0:
            obs_y = acc["yellow"] / acc["minutes"] * 90.0
            obs_r = acc["red"] / acc["minutes"] * 90.0
            obs_o = acc["own"] / acc["minutes"] * 90.0
            source = "own record"
        else:
            obs_y = obs_r = obs_o = 0.0
            source = "no minutes on record: %s baseline" % scoring.POS_NAME[pos]
        fit.rates[pid] = CardRates(
            player_id=pid,
            position=pos,
            yellow90=stats.shrink(obs_y, n90, fit.baseline_yellow90[pos], CARD_SHRINK_90S),
            red90=stats.shrink(obs_r, n90, fit.baseline_red90[pos], CARD_SHRINK_90S),
            own_goal90=stats.shrink(obs_o, n90, fit.baseline_own_goal90[pos], CARD_SHRINK_90S),
            n90=n90,
            source=source,
        )
    return fit


def _col_sum(frame: Any, column: str) -> float:
    if frame is None or column not in getattr(frame, "columns", []):
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

# Acklam's rational approximation to the inverse normal CDF. scipy is not
# installed in this environment and numpy has no ppf; relative error is under
# 1.15e-9 across the whole support, far tighter than any Monte-Carlo noise.
_PPF_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_PPF_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_PPF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
_PPF_PLOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p < _PPF_PLOW:
        q = math.sqrt(-2 * math.log(p))
        return (((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q
                 + _PPF_C[4]) * q + _PPF_C[5]) / ((((_PPF_D[0] * q + _PPF_D[1]) * q
                                                   + _PPF_D[2]) * q + _PPF_D[3]) * q + 1)
    if p > 1 - _PPF_PLOW:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q
                  + _PPF_C[4]) * q + _PPF_C[5]) / ((((_PPF_D[0] * q + _PPF_D[1]) * q
                                                     + _PPF_D[2]) * q + _PPF_D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_PPF_A[0] * r + _PPF_A[1]) * r + _PPF_A[2]) * r + _PPF_A[3]) * r
             + _PPF_A[4]) * r + _PPF_A[5]) * q / (((((_PPF_B[0] * r + _PPF_B[1]) * r
                                                     + _PPF_B[2]) * r + _PPF_B[3]) * r
                                                   + _PPF_B[4]) * r + 1)


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    return 1.0 if x > 1.0 else float(x)


def _conceded_moments(lam: float) -> Tuple[float, float]:
    """(E[floor(K/2)], E[floor(K/2)^2]) under Poisson(lam)."""
    e1 = e2 = 0.0
    for k in range(0, MAX_CONCEDED + 1):
        p = stats.poisson_pmf(k, lam)
        f = k // scoring.GOALS_CONCEDED_PER
        e1 += p * f
        e2 += p * f * f
    return e1, e2


def _conditional_conceded_cdf(lam: float) -> np.ndarray:
    """CDF over 1..MAX_CONCEDED given at least one goal conceded.

    The team model's ``p_cs`` carries a Dixon-Coles correction that lifts P(0-0)
    above the independent-Poisson value, so the simulation takes P(0) straight
    from the forecast and draws the rest from the Poisson conditioned on being
    non-zero. That keeps both the clean-sheet rate and the conceded-goal counts
    consistent with what the analytic components used.
    """
    pmf = np.array([stats.poisson_pmf(k, lam) for k in range(1, MAX_CONCEDED + 1)])
    total = float(pmf.sum())
    if total <= 0:
        pmf = np.zeros(MAX_CONCEDED)
        pmf[0] = 1.0
        total = 1.0
    cdf = np.cumsum(pmf / total)
    cdf[-1] = 1.0
    return cdf


# ---------------------------------------------------------------------------
# Per-fixture working state
# ---------------------------------------------------------------------------


@dataclass
class _Working:
    """One player's fixture projection mid-flight, before bonus is allocated."""

    projection: PlayerFixtureProjection
    ctx: FixtureContext
    position: int
    # Rates recovered on a per-90 basis so the simulation can re-evaluate them
    # for the started / substitute branches instead of at the blended xmins.
    rate90_goals: float = 0.0
    rate90_assists: float = 0.0
    rate90_pen_attempts: float = 0.0
    rate90_saves: float = 0.0
    rate90_pen_saves: float = 0.0
    p_start: float = 0.0
    p_sub: float = 0.0
    p60_given_start: float = 0.0
    mins_start: float = 0.0
    mins_sub: float = 0.0
    mins_scale: float = 1.0
    p_cs_match: float = 0.0
    # E[minutes | reached 60] / 90, from defending.on_pitch_fraction. Clean
    # sheets and conceded goals are settled over the minutes the player was on
    # the pitch, so the simulator must scale by the same figure the analytic
    # terms use or the two stop agreeing.
    on_pitch_fraction: float = 1.0
    p_defcon_start: float = 0.0
    p_defcon_sub: float = 0.0
    p_yellow: float = 0.0
    p_red: float = 0.0
    p_own_goal: float = 0.0
    yellow90: float = 0.0
    red90: float = 0.0
    own_goal90: float = 0.0
    penalty_miss_rate: float = 0.0
    bps_mu: float = 0.0
    bps_sd: float = 1.0
    variance: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class XPEngine:
    """Fits every component model and assembles expected points."""

    def __init__(self, config: Optional[Config] = None, mc_draws: int = DEFAULT_MC_DRAWS) -> None:
        self.config = config or Config.load()
        self.mc_draws = int(mc_draws)
        self.state: Optional["GameState"] = None

        self.team = TeamRatingModel(self.config)
        self.minutes = MinutesModel(self.config)
        self.attacking = AttackingModel(self.config)
        self.defending = DefendingModel(self.config)
        self.defcon = DefconModel(self.config)
        self.bonus = BonusModel(self.config)
        self.cards = CardFit()

        self.fitted = False
        self.warnings: List[str] = []
        self.fit_seconds: Dict[str, float] = {}
        self.timings: Dict[str, float] = {}

        self._forecasts: Dict[int, MatchForecast] = {}
        self._minutes_cache: Dict[Tuple[int, int], MinutesForecast] = {}
        self._bundles: Dict[Tuple[int, int], Dict[int, PlayerFixtureProjection]] = {}
        self._detail: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._dist: Dict[Tuple[int, int], np.ndarray] = {}
        self._gw_proj: Dict[Tuple[int, int], PlayerGWProjection] = {}
        self._haul: Dict[Tuple[int, int], float] = {}
        self._fixtures_by_gw: Dict[int, Dict[int, List[Fixture]]] = {}
        self._squads: Dict[int, List[int]] = {}
        self._bonus_mismatch_logged = False

    # -- fitting ------------------------------------------------------------

    def _apply_market_odds(self, state: "GameState") -> Dict[str, Any]:
        """Feed bookmaker prices into the team ratings, if any are available.

        Deliberately swallows everything: odds are an enhancement, and a dead
        third-party API must never be able to stop a projection being produced
        ninety minutes before a deadline.
        """
        try:
            from gaffer.core.config import CACHE_DIR
            from gaffer.data import odds as odds_mod

            payload = odds_mod.load_odds(
                CACHE_DIR,
                key=getattr(self.config, "odds_api_key", None),
                min_refresh_hours=float(getattr(self.config, "odds_min_refresh_hours",
                                                odds_mod.DEFAULT_MIN_REFRESH_HOURS)),
            )
            report = odds_mod.apply_to_model(self.team, state, payload)
            if report.get("applied"):
                self.warnings.append(
                    "odds: %d fixture(s) priced from the market — %s"
                    % (report["applied"], report.get("status", "")))
            elif payload.get("events"):
                self.warnings.append("odds: fetched but matched no fixtures — %s"
                                     % (report.get("unmatched_names") or "unknown clubs"))
            if report.get("unmatched_names"):
                self.warnings.append("odds: unrecognised club name(s) %s; add them to "
                                     "gaffer.data.odds.ODDS_NAME_TO_SHORT"
                                     % ", ".join(report["unmatched_names"]))
            return report
        except Exception as exc:  # never fatal
            self.warnings.append("odds: skipped (%s)" % exc)
            return {"applied": 0, "status": "error: %s" % exc}

    def fit(self, state: "GameState", refit: bool = False) -> "XPEngine":
        """Fit every sub-model, in dependency order.

        Team ratings go first because everything downstream is expressed
        relative to a fixture's expected goals: the attacking model scales a
        player's xG90 by the team lambda, defending turns the opponent lambda
        into a clean sheet and a shots-on-target count, DEFCON reads the ratio
        of the two as field tilt, and bonus needs both to fill in whatever the
        assembler does not hand it.
        """
        self.state = state
        self.warnings = []
        self.reset_caches()

        t0 = time.time()
        self.team.fit(state=state)
        self.fit_seconds["team_ratings"] = time.time() - t0
        self.warnings.extend("team_ratings: %s" % w for w in self.team.warnings)

        # Overlay the betting market on the fitted ratings where prices exist.
        # Pre-season this is the single largest available accuracy gain: the
        # ratings are fitted on last season, three clubs have no top-flight
        # history at all, and the market has already priced the summer. Entirely
        # optional — without ODDS_API_KEY this is a no-op and the model runs on
        # its own ratings exactly as before.
        self.odds_report = self._apply_market_odds(state)

        t0 = time.time()
        self.minutes.fit(state)
        self.fit_seconds["minutes"] = time.time() - t0

        t0 = time.time()
        self.attacking.fit(state, force_refit=refit)
        # Pin the attacking model's fixture-scaling reference to the lambda the
        # team model actually produces, not the config default: an elasticity is
        # meaningless without agreement on what "average" means.
        if self.team.mean_lambda == self.team.mean_lambda:  # not NaN
            self.attacking.set_league_average_lambda(self.team.mean_lambda)
        self.fit_seconds["attacking"] = time.time() - t0
        self.warnings.extend("attacking: %s" % w for w in self.attacking.warnings)

        t0 = time.time()
        self.defending.fit(state, team_model=self.team, refit=refit)
        self.fit_seconds["defending"] = time.time() - t0
        self.warnings.extend("defending: %s" % w for w in self.defending.warnings)

        t0 = time.time()
        self.defcon.fit(state, refit=refit)
        self.fit_seconds["defcon"] = time.time() - t0
        self.warnings.extend("defcon: %s" % w for w in self.defcon.warnings)

        t0 = time.time()
        self.bonus.fit(state)
        self.fit_seconds["bonus"] = time.time() - t0
        self.warnings.extend("bonus: %s" % w for w in self.bonus.warnings)

        t0 = time.time()
        self.cards = fit_card_rates(state, self.config)
        self.fit_seconds["cards"] = time.time() - t0

        self.fitted = True
        return self

    def reset_caches(self) -> None:
        """Drop every memo. ``project`` is pure, so this only costs time."""
        self._forecasts = {}
        self._minutes_cache = {}
        self._bundles = {}
        self._detail = {}
        self._dist = {}
        self._gw_proj = {}
        self._haul = {}
        self._fixtures_by_gw = {}
        self._squads = {}
        self._bonus_mismatch_logged = False

    # -- lookups ------------------------------------------------------------

    def _require_state(self) -> "GameState":
        if self.state is None:
            raise RuntimeError("call XPEngine.fit(state) first")
        return self.state

    def forecast(self, fixture: Fixture) -> MatchForecast:
        cached = self._forecasts.get(int(fixture.id))
        if cached is None:
            cached = self.team.forecast_fixture(fixture)
            self._forecasts[int(fixture.id)] = cached
        return cached

    def minutes_forecast(self, player_id: int, gw: int) -> MinutesForecast:
        key = (int(player_id), int(gw))
        cached = self._minutes_cache.get(key)
        if cached is None:
            cached = self.minutes.forecast(int(player_id), int(gw), self._require_state())
            self._minutes_cache[key] = cached
        return cached

    def context(self, player_id: int, fixture: Fixture, gw: int) -> FixtureContext:
        """Build the shared context for one player in one fixture."""
        state = self._require_state()
        player = state.players[int(player_id)]
        if player.team_id not in (fixture.team_h, fixture.team_a):
            raise ValueError(
                "player %d plays for team %d, which is not in fixture %d (%d v %d)"
                % (player_id, player.team_id, fixture.id, fixture.team_h, fixture.team_a))
        forecast = self.forecast(fixture)
        is_home = bool(player.team_id == fixture.team_h)
        return FixtureContext(
            fixture=fixture,
            player=player,
            is_home=is_home,
            team_lambda=forecast.lambda_h if is_home else forecast.lambda_a,
            opponent_lambda=forecast.lambda_a if is_home else forecast.lambda_h,
            forecast=forecast,
            minutes=self.minutes_forecast(player_id, gw),
        )

    def fixtures_for(self, team_id: int, gw: int) -> List[Fixture]:
        """Every fixture a club plays in one gameweek: 0 (blank), 1 or 2 (double)."""
        state = self._require_state()
        index = self._fixtures_by_gw.get(int(gw))
        if index is None:
            index = {}
            for f in state.fixtures:
                if f.gw is None or int(f.gw) != int(gw):
                    continue
                index.setdefault(int(f.team_h), []).append(f)
                index.setdefault(int(f.team_a), []).append(f)
            for team in index:
                index[team].sort(key=lambda f: (f.kickoff_time or "", f.id))
            self._fixtures_by_gw[int(gw)] = index
        return list(index.get(int(team_id), []))

    def _squad(self, team_id: int) -> List[int]:
        if not self._squads:
            state = self._require_state()
            for pid, player in state.players.items():
                self._squads.setdefault(int(player.team_id), []).append(int(pid))
            for team in self._squads:
                self._squads[team].sort()
        return self._squads.get(int(team_id), [])

    # -- per-player components (everything except bonus) --------------------

    def _core(self, player_id: int, fixture: Fixture, gw: int) -> _Working:
        """Every component that does not depend on the other 21 players."""
        state = self._require_state()
        pid = int(player_id)
        player = state.players[pid]
        pos = player.position
        ctx = self.context(pid, fixture, gw)
        mins = ctx.minutes

        p_start, m_start, p_sub, m_sub = minutes_split(mins, self.config)
        p_appear = _clip01(mins.p_appear)
        p_60 = min(_clip01(mins.p_60), p_appear)
        xmins = max(0.0, float(mins.xmins))
        # P(60+ | started). Substitutes essentially never reach 60 (minutes.py
        # fits 1.4% of substitute appearances), so attributing all of p_60 to
        # the started branch is accurate to about 0.002 and keeps the simulated
        # appearance points exactly equal to the analytic ones.
        p60_given_start = _clip01(p_60 / p_start) if p_start > 1e-9 else 0.0
        on_pitch90 = xmins / 90.0

        proj = PlayerFixtureProjection(
            player_id=pid,
            gw=int(gw),
            fixture_id=int(fixture.id),
            opponent_id=ctx.opponent_id,
            is_home=ctx.is_home,
            difficulty=ctx.difficulty,
            p_appear=p_appear,
            p_start=_clip01(mins.p_start),
            p_60=p_60,
            xmins=xmins,
        )
        work = _Working(projection=proj, ctx=ctx, position=pos)
        work.p_start = p_start
        work.p_sub = p_sub
        work.p60_given_start = p60_given_start
        # ``minutes_split`` clamps minutes-per-start into [45, 90], so for a
        # near-ever-present the start/sub mixture can fall a couple of minutes
        # short of the minutes model's own xmins. Rescale the simulated bucket
        # lengths so the mixture reproduces xmins exactly; otherwise every rate
        # in the simulation runs a few percent light against its analytic twin.
        mixture90 = (p_start * m_start + p_sub * m_sub) / 90.0
        work.mins_scale = (on_pitch90 / mixture90) if mixture90 > 1e-9 else 1.0
        work.mins_start = m_start
        work.mins_sub = m_sub

        # --- appearance ---------------------------------------------------
        proj.xp_appearance = scoring.expected_appearance_points(p_appear, p_60)
        e_app = p_appear + p_60                      # 1*(p_appear-p_60) + 2*p_60
        e_app_sq = p_appear + 3.0 * p_60             # 1*(p_appear-p_60) + 4*p_60
        work.variance["appearance"] = max(e_app_sq - e_app * e_app, 0.0)

        # --- attacking ----------------------------------------------------
        att = self.attacking.project_breakdown(pid, ctx)
        goal_pts, assist_pts, pen_miss_pts = self.attacking.expected_attacking_points(pid, ctx)
        proj.lambda_goals = att.lambda_goals
        proj.lambda_assists = att.lambda_assists
        proj.xp_goals = goal_pts
        proj.xp_assists = assist_pts
        work.variance["goals"] = (float(scoring.GOAL_POINTS[pos]) ** 2) * att.lambda_goals
        work.variance["assists"] = (float(scoring.ASSIST_POINTS) ** 2) * att.lambda_assists
        if on_pitch90 > 1e-9:
            work.rate90_goals = att.lambda_goals / on_pitch90
            work.rate90_assists = att.lambda_assists / on_pitch90
            work.rate90_pen_attempts = att.penalty_attempts / on_pitch90
        work.penalty_miss_rate = 1.0 - float(self.attacking.fit_params.penalty_conversion)

        # --- defending ----------------------------------------------------
        d = self.defending.project(pid, ctx)
        proj.lambda_conceded = d["lambda_conceded"]
        proj.p_clean_sheet = d["p_clean_sheet"]        # already gated by p_60
        proj.lambda_saves = d["lambda_saves"]
        proj.xp_clean_sheet = d["xp_clean_sheet"]
        proj.xp_goals_conceded = d["xp_goals_conceded"]
        proj.xp_saves = d["xp_saves"]
        work.p_cs_match = _clip01(d["p_clean_sheet_match"])
        work.on_pitch_fraction = float(d.get("on_pitch_fraction", 1.0))
        work.variance["clean_sheet"] = d["var_clean_sheet"]
        work.variance["saves"] = d["var_saves"]
        if on_pitch90 > 1e-9:
            work.rate90_saves = d["lambda_saves"] / on_pitch90
            work.rate90_pen_saves = d["lambda_penalty_saves"] / on_pitch90

        # Goals conceded: exact first two moments of -floor(K/2) under the
        # 60-minute gate, plus the exact covariance with the clean sheet. The
        # two are strongly dependent (a clean sheet forces the conceded term to
        # zero) and dropping the covariance understates a defender's spread.
        per = float(scoring.GOALS_CONCEDED_POINTS[pos])
        if per != 0.0:
            e1, e2 = _conceded_moments(
                proj.lambda_conceded * work.on_pitch_fraction)
            mean_c = per * p_60 * e1
            var_c = (per ** 2) * (p_60 * e2) - mean_c ** 2
            work.variance["goals_conceded"] = max(var_c, 0.0)
            # E[cs * conceded] = 0: a clean sheet means floor(K/2) = 0.
            work.variance["cs_conceded_cov"] = -2.0 * proj.xp_clean_sheet * mean_c
        else:
            work.variance["goals_conceded"] = 0.0
            work.variance["cs_conceded_cov"] = 0.0

        # --- DEFCON -------------------------------------------------------
        dc = self.defcon.project_detail(pid, ctx)
        proj.p_defcon = dc["p_defcon"]
        proj.xp_defcon = dc["xp_defcon"]
        work.variance["defcon"] = dc["var_defcon"]
        mean_start = float(dc.get("defcon_mean_if_start", 0.0))
        dispersion = float(dc.get("defcon_dispersion", 1e6))
        threshold = int(dc.get("defcon_threshold", 0) or 0)
        if threshold > 0 and mean_start > 0:
            work.p_defcon_start = stats.negbin_at_least(threshold, mean_start, dispersion)
            mean_sub = mean_start * (m_sub / m_start) if m_start > 1e-9 else 0.0
            work.p_defcon_sub = stats.negbin_at_least(threshold, mean_sub, dispersion)

        # --- cards and own goals -------------------------------------------
        rates = self.cards.rates.get(pid)
        if rates is None:
            rates = CardRates(player_id=pid, position=pos,
                              yellow90=self.cards.baseline_yellow90.get(pos, 0.0),
                              red90=self.cards.baseline_red90.get(pos, 0.0),
                              own_goal90=self.cards.baseline_own_goal90.get(pos, 0.0),
                              source="unfitted fallback")
        work.yellow90, work.red90, work.own_goal90 = (
            rates.yellow90, rates.red90, rates.own_goal90)
        work.p_yellow = _clip01(rates.yellow90 * on_pitch90)
        work.p_red = _clip01(rates.red90 * on_pitch90)
        work.p_own_goal = _clip01(rates.own_goal90 * on_pitch90)
        # A second yellow is a red and ends the match; that second-order effect
        # is left out, as is the loss of minutes after a sending-off.
        proj.xp_cards = (
            scoring.expected_card_points(work.p_yellow, work.p_red)
            + scoring.OWN_GOAL * work.p_own_goal
        )
        work.variance["cards"] = (
            (scoring.YELLOW_CARD ** 2) * work.p_yellow * (1 - work.p_yellow)
            + (scoring.RED_CARD ** 2) * work.p_red * (1 - work.p_red)
            + (scoring.OWN_GOAL ** 2) * work.p_own_goal * (1 - work.p_own_goal)
        )

        # --- penalties (saved by the keeper, missed by the taker) ----------
        proj.xp_penalty = d["xp_penalty_save"] + pen_miss_pts
        lam_pen_miss = att.penalty_attempts * work.penalty_miss_rate
        work.variance["penalty"] = (
            (scoring.PENALTY_SAVE ** 2) * d["lambda_penalty_saves"]
            + (scoring.PENALTY_MISS ** 2) * lam_pen_miss
        )

        work.detail = {
            "minutes_reason": mins.reason,
            "attacking": att,
            "defending": d,
            "defcon": dc,
            "card_rates": rates,
            "lambda_penalty_saves": d["lambda_penalty_saves"],
            "lambda_penalty_miss": lam_pen_miss,
            "p_clean_sheet_match": work.p_cs_match,
            "sot_faced": d.get("sot_faced", 0.0),
            "save_share": d.get("save_share", 0.0),
        }
        return work

    def _bonus_view(self, work: _Working) -> _BonusView:
        proj = work.projection
        return _BonusView(
            p_appear=proj.p_appear,
            p_60=proj.p_60,
            xmins=proj.xmins,
            lambda_goals=proj.lambda_goals,
            lambda_assists=proj.lambda_assists,
            p_clean_sheet=work.p_cs_match,   # ungated: bonus.py applies the gate
            lambda_saves=proj.lambda_saves,
            lambda_conceded=proj.lambda_conceded,
            lambda_pen_saves=work.detail["lambda_penalty_saves"],
            lambda_pen_miss=work.detail["lambda_penalty_miss"],
            p_yellow=work.p_yellow,
            p_red=work.p_red,
            p_own_goal=work.p_own_goal,
        )

    # -- the fixture bundle -------------------------------------------------

    def _bundle(self, fixture: Fixture, gw: int) -> Dict[int, PlayerFixtureProjection]:
        """Project every player in one fixture, allocate its bonus, close totals.

        This is the unit of work. Bonus is contested across the whole match, so
        no player in this fixture can be finished until all of them have been
        projected; and because the ~50 players share a match forecast, a bonus
        simulation and a Monte Carlo, doing it fixture-at-a-time is also what
        makes the full run take seconds.
        """
        key = (int(fixture.id), int(gw))
        cached = self._bundles.get(key)
        if cached is not None:
            return cached

        squads: List[int] = []
        for team_id in (fixture.team_h, fixture.team_a):
            squads.extend(self._squad(int(team_id)))
        squads.sort()
        state = self._require_state()

        works: Dict[int, _Working] = {}
        views: Dict[int, _BonusView] = {}
        for pid in squads:
            work = self._core(pid, fixture, gw)
            works[pid] = work
            views[pid] = self._bonus_view(work)

        # --- bonus ---------------------------------------------------------
        forecast = self.forecast(fixture)
        bps, sds = self.bonus.candidates_for_fixture(state, fixture, forecast, views)
        expected_bonus = self.bonus.project_bonus(int(fixture.id), bps, sds=sds)
        ranks = self._rank_probabilities(fixture, bps, sds, expected_bonus)

        for pid, work in works.items():
            work.bps_mu = float(bps.get(pid, 0.0))
            work.bps_sd = float(sds.get(pid, self.bonus.bps_sd(pid)))
            work.projection.exp_bps = work.bps_mu
            work.projection.xp_bonus = float(expected_bonus.get(pid, 0.0))
            p1, p2, p3 = ranks.get(pid, (0.0, 0.0, 0.0))
            e_b = work.projection.xp_bonus
            work.variance["bonus"] = max(9.0 * p1 + 4.0 * p2 + 1.0 * p3 - e_b * e_b, 0.0)
            work.detail["bonus_ranks"] = (p1, p2, p3)

        # --- simulation, then totals ---------------------------------------
        self._simulate(fixture, gw, works, ranks)

        bundle = {pid: w.projection for pid, w in works.items()}
        self._bundles[key] = bundle
        for pid, w in works.items():
            self._detail[(pid, int(fixture.id))] = w.detail
        return bundle

    def _rank_probabilities(
        self,
        fixture: Fixture,
        bps: Dict[int, float],
        sds: Dict[int, float],
        expected_bonus: Dict[int, float],
    ) -> Dict[int, Tuple[float, float, float]]:
        """P(3 bonus)/P(2 bonus)/P(1 bonus) for every candidate in this fixture.

        ``bonus.project_bonus`` returns only the expectation, but the second
        moment and the simulated bonus tier both need the full tier vector.
        The bonus model's ranker is a fixture-level event simulation — one
        clean-sheet draw per team, Poisson goals, a two-stage appearance, and
        FPL's tie rules on integer BPS — and so cannot be reproduced by
        re-running a sampler on the means and standard deviations alone; the
        vector is read back from the run that produced the expectation. The
        reconstruction is still checked against that expectation on every
        fixture: a divergence would mean the two have drifted apart, and is
        logged rather than absorbed.

        These are bonus *tiers*, not BPS ranks: under a tie two players can both
        be awarded 3. ``9*p3 + 4*p2 + p1 - E[bonus]^2`` is therefore still the
        exact variance, which is what the caller uses it for.
        """
        if not bps:
            return {}
        cached = self.bonus.rank_probabilities(int(fixture.id))
        if cached is not None:
            raw = {pid: list(value) for pid, value in cached.items()}
        else:
            # The ranker walks ``scores.keys()`` in insertion order and assigns
            # each candidate a column of the draw matrix from it, so the dict
            # must be built in the same order ``BonusModel.project_bonus``
            # builds it — sorted by id — or the two runs sample different noise
            # for the same player and the reconstructed bonus drifts.
            ids = sorted(bps.keys())
            scores = {pid: float(bps[pid]) for pid in ids}
            sigma: Dict[int, float] = {}
            for pid in ids:
                value = sds.get(pid) if sds is not None else None
                if value is None:
                    value = self.bonus.bps_sd(pid)
                sigma[pid] = max(float(value), 0.5)
            raw = stats.top_k_probabilities(
                scores, sigma, k=3,
                iterations=int(self.config.model.bps_sim_iterations),
                seed=7 + int(fixture.id),
            )
        out: Dict[int, Tuple[float, float, float]] = {}
        worst = 0.0
        for pid, probs in raw.items():
            p1, p2, p3 = (probs + [0.0, 0.0, 0.0])[:3]
            out[pid] = (p1, p2, p3)
            worst = max(worst, abs(3 * p1 + 2 * p2 + p3 - expected_bonus.get(pid, 0.0)))
        if worst > 1e-6 and not self._bonus_mismatch_logged:
            self._bonus_mismatch_logged = True
            self.warnings.append(
                "bonus rank reconstruction differs from BonusModel.project_bonus by "
                "%.2e; expected bonus still comes from the bonus model" % worst)
            log.warning(self.warnings[-1])
        return out

    # -- Monte Carlo --------------------------------------------------------

    def _simulate(
        self,
        fixture: Fixture,
        gw: int,
        works: Dict[int, _Working],
        ranks: Dict[int, Tuple[float, float, float]],
    ) -> None:
        """Seeded simulation of the whole points total, per player.

        What this buys over the analytic sum is the *coupling*. Points are not a
        bag of independent components: a clean sheet forbids conceded points,
        and — the one that actually decides captaincy — a player who scores
        twice nearly always takes three bonus. That second link is reproduced
        without inventing a correlation: the simulated BPS moves with the
        simulated goals, assists, clean sheet and saves through ``bonus.py``'s
        own fitted 2026/27 coefficients, and the bonus tier is read off that BPS
        at cutoffs chosen to reproduce the ranker's marginal exactly.
        """
        n = self.mc_draws
        rng = np.random.default_rng(MC_SEED_BASE + int(fixture.id) * 7919 + int(gw))
        for pid in sorted(works):
            work = works[pid]
            self._dist[(pid, int(fixture.id))] = self._simulate_player(work, ranks, rng, n)
            self._finalise(work)

    def _simulate_player(
        self,
        work: _Working,
        ranks: Dict[int, Tuple[float, float, float]],
        rng: np.random.Generator,
        n: int,
    ) -> np.ndarray:
        proj = work.projection
        pos = work.position
        pid = proj.player_id

        # --- minutes bucket ------------------------------------------------
        u = rng.random(n)
        started = u < work.p_start
        subbed = (u >= work.p_start) & (u < work.p_start + work.p_sub)
        on = started | subbed
        frac = np.where(started, work.mins_scale * work.mins_start / 90.0,
                        np.where(subbed, work.mins_scale * work.mins_sub / 90.0, 0.0))
        long60 = started & (rng.random(n) < work.p60_given_start)
        short = on & ~long60
        appearance = np.where(long60, scoring.LONG_PLAY,
                              np.where(on, scoring.SHORT_PLAY, 0)).astype(np.float64)

        # --- attacking ------------------------------------------------------
        goals = rng.poisson(np.maximum(work.rate90_goals * frac, 0.0))
        assists = rng.poisson(np.maximum(work.rate90_assists * frac, 0.0))
        pen_miss = rng.poisson(
            np.maximum(work.rate90_pen_attempts * work.penalty_miss_rate * frac, 0.0))

        # --- goals conceded and clean sheet ---------------------------------
        # Both settle over the minutes the player was on the pitch, exactly as
        # the analytic terms in `_project_one` do; drawing them over the whole
        # match instead is what makes the simulated mean drift below the
        # analytic total for anyone who is regularly withdrawn.
        lam_c = max(proj.lambda_conceded * work.on_pitch_fraction, 1e-6)
        cs_draw = rng.random(n) < work.p_cs_match ** work.on_pitch_fraction
        cdf = _conditional_conceded_cdf(lam_c)
        conceded = 1 + np.searchsorted(cdf, rng.random(n))
        conceded = np.where(cs_draw, 0, np.minimum(conceded, MAX_CONCEDED))
        cs_hit = cs_draw & long60
        cs_pts = scoring.CLEAN_SHEET_POINTS[pos] * cs_hit
        # Conceded points land on anyone who was on the pitch; only the clean
        # sheet needs 60+. Gating this on `long60` forgave every sub-60 cameo.
        conc_pts = scoring.GOALS_CONCEDED_POINTS[pos] * (conceded // scoring.GOALS_CONCEDED_PER) * on

        # --- saves and penalty saves ----------------------------------------
        if pos == scoring.GKP:
            saves = rng.poisson(np.maximum(work.rate90_saves * frac, 0.0))
            pen_saves = rng.poisson(np.maximum(work.rate90_pen_saves * frac, 0.0))
        else:
            saves = np.zeros(n, dtype=np.int64)
            pen_saves = np.zeros(n, dtype=np.int64)
        save_pts = scoring.SAVE_POINTS * (saves // scoring.SAVES_PER)

        # --- DEFCON, cards --------------------------------------------------
        p_dc = np.where(started, work.p_defcon_start, np.where(subbed, work.p_defcon_sub, 0.0))
        defcon = rng.random(n) < p_dc
        yellow = rng.random(n) < np.minimum(work.yellow90 * frac, 1.0)
        red = rng.random(n) < np.minimum(work.red90 * frac, 1.0)
        own_goal = rng.poisson(np.maximum(work.own_goal90 * frac, 0.0))

        # --- bonus ----------------------------------------------------------
        bonus_pts = self._simulate_bonus(
            work, ranks.get(pid, (0.0, 0.0, 0.0)), rng, n,
            goals=goals, assists=assists, clean_sheet=cs_hit, saves=saves,
            long60=long60, short=short, on=on)

        total = (
            appearance
            + scoring.GOAL_POINTS[pos] * goals
            + scoring.ASSIST_POINTS * assists
            + cs_pts + conc_pts + save_pts
            + scoring.DEFCON_POINTS[pos] * defcon
            + bonus_pts
            + scoring.YELLOW_CARD * yellow
            + scoring.RED_CARD * red
            + scoring.OWN_GOAL * own_goal
            + scoring.PENALTY_SAVE * pen_saves
            + scoring.PENALTY_MISS * pen_miss
        )
        totals = np.rint(total).astype(np.int64)

        # The components are not independent and the dependence is large enough
        # to matter for a captaincy call: everything is gated by the same
        # minutes draw, a clean sheet forbids conceded points, and a player who
        # scores twice nearly always takes three bonus. The analytic terms above
        # are the marginal variances; this is the measured remainder, so the
        # printed decomposition still adds up to the sd the model reports.
        analytic = sum(work.variance.values())
        work.variance["coupling"] = float(total.var()) - analytic

        clipped_low = int(np.count_nonzero(totals < DIST_MIN))
        clipped_high = int(np.count_nonzero(totals > DIST_MAX))
        if clipped_low or clipped_high:
            log.warning("player %d fixture %d: %d/%d simulated totals outside [%d, %d]",
                        pid, proj.fixture_id, clipped_low + clipped_high, n, DIST_MIN, DIST_MAX)
        idx = np.clip(totals - DIST_MIN, 0, DIST_SIZE - 1)
        dist = np.bincount(idx, minlength=DIST_SIZE).astype(np.float64) / float(n)

        work.detail["mc_mean"] = float(np.dot(dist, np.arange(DIST_MIN, DIST_MAX + 1)))
        work.detail["mc_sd"] = float(
            math.sqrt(max(np.dot(dist, (np.arange(DIST_MIN, DIST_MAX + 1) ** 2))
                          - work.detail["mc_mean"] ** 2, 0.0)))
        work.detail["mc_clipped"] = clipped_low + clipped_high
        return dist

    def _simulate_bonus(
        self,
        work: _Working,
        rank: Tuple[float, float, float],
        rng: np.random.Generator,
        n: int,
        goals: np.ndarray,
        assists: np.ndarray,
        clean_sheet: np.ndarray,
        saves: np.ndarray,
        long60: np.ndarray,
        short: np.ndarray,
        on: np.ndarray,
    ) -> np.ndarray:
        """Bonus tier per draw, correlated with the draw's own events.

        The player's BPS is rebuilt per draw as its expectation plus the
        deviation the simulated events imply under the fitted BPS weights, plus
        residual noise sized so the total variance still matches the bonus
        model's ``sd``. The tier cutoffs sit at the quantiles of that
        distribution implied by P(1st)/P(2nd)/P(3rd), conditioned on appearing,
        so the marginal expected bonus is preserved exactly while a two-goal
        draw now nearly always takes the three.
        """
        p1, p2, p3 = rank
        out = np.zeros(n, dtype=np.float64)
        if p1 + p2 + p3 <= 0.0:
            return out

        fit = self.bonus.fits.get(work.position)
        weights = fit.coefficients_2026_27 if fit is not None else {}
        w_goal = float(weights.get("goals", 0.0))
        w_assist = float(weights.get("assists", 0.0))
        w_cs = float(weights.get("clean_sheet", 0.0))
        w_save = float(weights.get("saves", 0.0))
        w_60 = float(weights.get("mins_60", 0.0))
        w_short = float(weights.get("mins_1_59", 0.0))

        event = (
            w_goal * goals + w_assist * assists + w_cs * clean_sheet
            + w_save * saves + w_60 * long60 + w_short * short
        ).astype(np.float64)
        event -= event.mean()
        var_event = float(event.var())
        target_var = max(work.bps_sd ** 2, 1e-6)
        # Keep at least a quarter of the spread as genuine noise: BPS also
        # rewards passes, key passes, dribbles and fouls, none of which are
        # projected, so a draw with average events is still far from certain.
        resid_var = max(target_var - var_event, 0.0625 * target_var)
        z = event + rng.normal(0.0, math.sqrt(resid_var), n)
        sd = math.sqrt(max(var_event + resid_var, 1e-12))
        z /= sd

        # Cutoffs are the *empirical* quantiles of the simulated BPS among the
        # draws where the player actually appeared, not normal quantiles: the
        # event term is a discrete mixture (a clean sheet is worth 12 BPS to a
        # keeper in one lump) and is nowhere near Gaussian, so reading the
        # cutoffs off a normal would bias the marginal by a tenth of a point.
        # Taking them empirically makes E[bonus] equal 3p1+2p2+p3 by
        # construction, to within the 1/n granularity of the sample.
        on_idx = np.flatnonzero(on)
        if on_idx.size == 0:
            return out
        frac_on = on_idx.size / float(n)
        t1 = _clip01(p1 / frac_on)
        t2 = _clip01((p1 + p2) / frac_on)
        t3 = _clip01((p1 + p2 + p3) / frac_on)
        c3, c2, c1 = np.quantile(z[on_idx], [1.0 - t3, 1.0 - t2, 1.0 - t1])
        out = np.where(z >= c1, 3.0, np.where(z >= c2, 2.0, np.where(z >= c3, 1.0, 0.0)))
        return out * on

    # -- closing a projection ------------------------------------------------

    def _finalise(self, work: _Working) -> None:
        proj = work.projection
        proj.xp_total = sum(proj.components().values())
        variance = sum(work.variance.values())
        proj.sd_total = math.sqrt(max(variance, 0.0))
        work.detail["variance"] = dict(work.variance)

    # -- public projection API ----------------------------------------------

    def project_player_fixture(
        self, player_id: int, fixture: Fixture, gw: Optional[int] = None
    ) -> PlayerFixtureProjection:
        """One player, one fixture, every field filled.

        Runs (and memoises) the whole fixture bundle, because expected bonus is
        not computable from one player in isolation.
        """
        gw = int(gw if gw is not None else (fixture.gw or 0))
        bundle = self._bundle(fixture, gw)
        pid = int(player_id)
        if pid not in bundle:
            state = self._require_state()
            team = state.players[pid].team_id if pid in state.players else None
            raise ValueError(
                "player %d (team %s) does not play in fixture %d (%d v %d)"
                % (pid, team, fixture.id, fixture.team_h, fixture.team_a))
        return bundle[pid]

    def p_haul(self, player_id: int, gw: int, threshold: int = HAUL_POINTS) -> float:
        """P(gameweek points >= threshold), summed correctly over a double."""
        key = (int(player_id), int(gw))
        if threshold == HAUL_POINTS and key in self._haul:
            return self._haul[key]
        dist = self.gw_distribution(int(player_id), int(gw))
        if dist is None:
            return 0.0
        lo = max(0, int(threshold) - DIST_MIN)
        return float(dist[lo:].sum()) if lo < len(dist) else 0.0

    def haul_table(self) -> Dict[Tuple[int, int], float]:
        """``{(player_id, gw): P(points >= 10)}`` for everything projected so far.

        ``ProjectionSet`` has no field for it, so captaincy code reads it from
        here (or from the ``haul`` key of the cached projection file, which
        ``load_projections`` restores into the same map).
        """
        return dict(self._haul)

    def gw_distribution(self, player_id: int, gw: int) -> Optional[np.ndarray]:
        """Points distribution for the gameweek, over ``DIST_MIN..DIST_MAX``.

        For a double gameweek the two fixtures' distributions are convolved:
        the totals add, so the distributions convolve, and P(haul) over a double
        is emphatically not the sum of the two single-fixture hauls.
        """
        state = self._require_state()
        pid = int(player_id)
        if pid not in state.players:
            return None
        fixtures = self.fixtures_for(state.players[pid].team_id, int(gw))
        if not fixtures:
            return None
        dists = []
        for fixture in fixtures:
            self._bundle(fixture, int(gw))
            d = self._dist.get((pid, int(fixture.id)))
            if d is not None:
                dists.append(d)
        if not dists:
            return None
        out = dists[0]
        for extra in dists[1:]:
            full = np.convolve(out, extra)
            # Both operands are indexed from DIST_MIN, so the convolution runs
            # from 2*DIST_MIN; fold it back onto the common support. Mass above
            # DIST_MAX piles into the top bin, which is exactly what a
            # P(points >= threshold) query needs and all this is used for.
            folded = np.zeros(DIST_SIZE)
            for i, p in enumerate(full):
                if p == 0.0:
                    continue
                value = (i + 2 * DIST_MIN)
                folded[min(max(value - DIST_MIN, 0), DIST_SIZE - 1)] += p
            out = folded
        return out

    def project(
        self, state: Optional["GameState"] = None, gws: Optional[Sequence[int]] = None
    ) -> ProjectionSet:
        """Every player, every gameweek in ``gws``.

        Doubles sum, blanks come back as an explicit zero-xp row with
        ``is_blank`` True rather than a missing key — chip planning reads this
        directly and a KeyError there would be silent points.
        """
        if state is not None and state is not self.state:
            self.state = state
            self.reset_caches()
        state = self._require_state()
        if not self.fitted:
            raise RuntimeError("call XPEngine.fit(state) before project()")
        if gws is None:
            horizon = int(self.config.model.default_horizon)
            gws = list(range(state.current_gw, state.current_gw + horizon))
        gws = [int(g) for g in gws]
        if not gws:
            raise ValueError("project() needs at least one gameweek")

        t0 = time.time()
        # Build every bundle first: one pass over fixtures, so each match's
        # forecast, bonus allocation and simulation happen exactly once.
        for gw in gws:
            for fixture in sorted(
                (f for f in state.fixtures if f.gw is not None and int(f.gw) == gw),
                key=lambda f: int(f.id),
            ):
                self._bundle(fixture, gw)
        self.timings["bundles_seconds"] = time.time() - t0

        t0 = time.time()
        out = ProjectionSet(
            season=state.season,
            generated_at=datetime.now(timezone.utc).isoformat(),
            first_gw=min(gws),
            last_gw=max(gws),
            model_version=MODEL_VERSION,
        )
        for pid in sorted(state.players):
            team_id = state.players[pid].team_id
            per_gw: Dict[int, PlayerGWProjection] = {}
            for gw in gws:
                gwp = PlayerGWProjection(player_id=pid, gw=gw)
                variance = 0.0
                for fixture in self.fixtures_for(team_id, gw):
                    bundle = self._bundles.get((int(fixture.id), gw))
                    if bundle is None or pid not in bundle:
                        continue
                    fp = bundle[pid]
                    gwp.fixtures.append(fp)
                    gwp.xp += fp.xp_total
                    variance += fp.sd_total ** 2
                # Independent fixtures: a double's variance is the sum. The
                # opponents differ and the matches are days apart, so there is
                # no shared clean sheet or shared bonus pool to correlate.
                gwp.sd = math.sqrt(max(variance, 0.0))
                per_gw[gw] = gwp
                self._gw_proj[(pid, gw)] = gwp
                self._haul[(pid, gw)] = self.p_haul(pid, gw) if gwp.fixtures else 0.0
            out.projections[pid] = per_gw
        self.timings["assemble_seconds"] = time.time() - t0
        self.timings["total_seconds"] = (
            self.timings["bundles_seconds"] + self.timings["assemble_seconds"])
        return out

    def gw_projection(self, player_id: int, gw: int) -> PlayerGWProjection:
        """One player's gameweek total, computing it on demand if need be."""
        key = (int(player_id), int(gw))
        cached = self._gw_proj.get(key)
        if cached is not None:
            return cached
        state = self._require_state()
        pid = int(player_id)
        gwp = PlayerGWProjection(player_id=pid, gw=int(gw))
        variance = 0.0
        for fixture in self.fixtures_for(state.players[pid].team_id, int(gw)):
            fp = self.project_player_fixture(pid, fixture, int(gw))
            gwp.fixtures.append(fp)
            gwp.xp += fp.xp_total
            variance += fp.sd_total ** 2
        gwp.sd = math.sqrt(max(variance, 0.0))
        self._gw_proj[key] = gwp
        self._haul[key] = self.p_haul(pid, int(gw)) if gwp.fixtures else 0.0
        return gwp

    # -- explanation ---------------------------------------------------------

    def explain(self, player_id: int, gw: int) -> str:
        """A breakdown a human can check line by line against the components."""
        state = self._require_state()
        pid = int(player_id)
        player = state.players[pid]
        gwp = self.gw_projection(pid, gw)
        head = "%s (%s, %s, £%.1fm) — GW%d" % (
            player.web_name, scoring.POS_NAME[player.position],
            state.short_name(player.team_id), player.price, int(gw))
        lines = [head, "=" * len(head)]

        if not gwp.fixtures:
            lines.append("  BLANK GAMEWEEK: %s have no fixture in GW%d. xP 0.00."
                         % (state.short_name(player.team_id), int(gw)))
            return "\n".join(lines)
        if len(gwp.fixtures) > 1:
            lines.append("  DOUBLE GAMEWEEK: %d fixtures, points sum." % len(gwp.fixtures))

        for fp in gwp.fixtures:
            detail = self._detail.get((pid, fp.fixture_id), {})
            fixture = next((f for f in state.fixtures if int(f.id) == fp.fixture_id), None)
            opponent = state.short_name(fp.opponent_id)
            venue = "H" if fp.is_home else "A"
            forecast = self.forecast(fixture) if fixture is not None else None
            lines.append("")
            lines.append("  fixture %d: %s v %s (%s), official FDR %d"
                         % (fp.fixture_id,
                            state.short_name(player.team_id) if fp.is_home else opponent,
                            opponent if fp.is_home else state.short_name(player.team_id),
                            venue, fp.difficulty))
            if forecast is not None:
                p_win = forecast.p_home_win if fp.is_home else forecast.p_away_win
                lines.append(
                    "    match      : team lambda %.2f, opponent lambda %.2f (source %s) "
                    "| P(win) %.0f%%, P(team CS) %.0f%%"
                    % (forecast.lambda_h if fp.is_home else forecast.lambda_a,
                       forecast.lambda_a if fp.is_home else forecast.lambda_h,
                       forecast.source, 100 * p_win,
                       100 * detail.get("p_clean_sheet_match", 0.0)))
            lines.append("    minutes    : p_start %.2f p_appear %.2f p_60 %.2f xmins %.1f"
                         % (fp.p_start, fp.p_appear, fp.p_60, fp.xmins))
            lines.append("                 %s" % detail.get("minutes_reason", "n/a"))
            att = detail.get("attacking")
            if att is not None:
                lines.append("    attacking  : xG90 %.3f xA90 %.3f, fixture x%.3f/%.3f, "
                             "finishing x%.3f" % (att.xg90, att.xa90, att.fixture_scale_goals,
                                                  att.fixture_scale_assists, att.finishing))
                lines.append("                 lambda_goals %.3f (open play %.3f + pens %.3f), "
                             "lambda_assists %.3f"
                             % (att.lambda_goals, att.lambda_goals_open_play,
                                att.lambda_goals_penalty, att.lambda_assists))
            if fixture is not None:
                ctx = self.context(pid, fixture, int(gw))
                lines.append("    defending  : %s" % self.defending.explain(pid, ctx))
                if player.position != scoring.GKP:
                    lines.append("    defcon     : %s" % self.defcon.explain(pid, ctx))
            ranks = detail.get("bonus_ranks", (0.0, 0.0, 0.0))
            lines.append("    bonus      : xBPS %.1f -> P(3)=%.0f%% P(2)=%.0f%% P(1)=%.0f%%"
                         % (fp.exp_bps, 100 * ranks[0], 100 * ranks[1], 100 * ranks[2]))
            cards = detail.get("card_rates")
            if cards is not None:
                lines.append("    discipline : %s" % cards.reason())

            lines.append("    components :")
            for name, value in fp.components().items():
                lines.append("      %-14s %+7.3f" % (name, value))
            lines.append("      %-14s %7.3f   sd %.2f" % ("TOTAL", fp.xp_total, fp.sd_total))
            variance = detail.get("variance", {})
            if variance:
                parts = " ".join("%s %.2f" % (k, v) for k, v in sorted(variance.items()) if abs(v) > 5e-3)
                lines.append("      variance     : %s" % parts)
            if "mc_mean" in detail:
                lines.append("      simulation   : mean %.3f sd %.3f over %d draws "
                             "(analytic mean %.3f sd %.3f)"
                             % (detail["mc_mean"], detail["mc_sd"], self.mc_draws,
                                fp.xp_total, fp.sd_total))

        lines.append("")
        lines.append("  GAMEWEEK TOTAL: %.3f xP, sd %.2f, P(>=%d) %.1f%%"
                     % (gwp.xp, gwp.sd, HAUL_POINTS, 100 * self.p_haul(pid, int(gw))))
        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------

    def save_projections(
        self, projections: ProjectionSet, path: Optional[str] = None
    ) -> str:
        path = path or projections_cache_path(projections.first_gw, projections.last_gw)
        haul = {
            "%d|%d" % (pid, gw): value
            for (pid, gw), value in sorted(self._haul.items())
        }
        payload = projection_set_to_dict(projections)
        payload["haul"] = haul
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    def load_projections(
        self, first_gw: int, last_gw: int, path: Optional[str] = None
    ) -> Optional[ProjectionSet]:
        path = path or projections_cache_path(first_gw, last_gw)
        if not os.path.exists(path):
            return None
        with open(path, "r") as fh:
            payload = json.load(fh)
        for key, value in (payload.get("haul") or {}).items():
            pid, gw = key.split("|")
            self._haul[(int(pid), int(gw))] = float(value)
        return projection_set_from_dict(payload)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

_FIXTURE_FIELDS = tuple(f.name for f in fields(PlayerFixtureProjection))


def projections_cache_path(first_gw: int, last_gw: int) -> str:
    return os.path.join(CACHE_DIR, "projections_%d_%d.json" % (int(first_gw), int(last_gw)))


def projection_set_to_dict(projections: ProjectionSet) -> Dict[str, Any]:
    """JSON-ready dict. Nothing is rounded: the round trip must be lossless."""
    out: Dict[str, Any] = {
        "season": projections.season,
        "generated_at": projections.generated_at,
        "first_gw": projections.first_gw,
        "last_gw": projections.last_gw,
        "model_version": projections.model_version,
        "projections": {},
    }
    for pid, per_gw in projections.projections.items():
        out["projections"][str(pid)] = {
            str(gw): {
                "player_id": gwp.player_id,
                "gw": gwp.gw,
                "xp": gwp.xp,
                "sd": gwp.sd,
                "fixtures": [
                    {name: getattr(fp, name) for name in _FIXTURE_FIELDS}
                    for fp in gwp.fixtures
                ],
            }
            for gw, gwp in per_gw.items()
        }
    return out


def projection_set_from_dict(raw: Dict[str, Any]) -> ProjectionSet:
    out = ProjectionSet(
        season=raw["season"],
        generated_at=raw["generated_at"],
        first_gw=int(raw["first_gw"]),
        last_gw=int(raw["last_gw"]),
        model_version=raw.get("model_version", MODEL_VERSION),
    )
    for pid, per_gw in raw["projections"].items():
        rebuilt: Dict[int, PlayerGWProjection] = {}
        for gw, row in per_gw.items():
            gwp = PlayerGWProjection(
                player_id=int(row["player_id"]),
                gw=int(row["gw"]),
                xp=float(row["xp"]),
                sd=float(row["sd"]),
                fixtures=[PlayerFixtureProjection(**fp) for fp in row["fixtures"]],
            )
            rebuilt[int(gw)] = gwp
        out.projections[int(pid)] = rebuilt
    return out


def projection_sets_equal(a: ProjectionSet, b: ProjectionSet) -> Tuple[bool, str]:
    """Exact structural comparison, ignoring ``generated_at``."""
    if (a.season, a.first_gw, a.last_gw, a.model_version) != (
            b.season, b.first_gw, b.last_gw, b.model_version):
        return False, "header differs"
    if set(a.projections) != set(b.projections):
        return False, "player sets differ"
    for pid, per_gw in a.projections.items():
        other = b.projections[pid]
        if set(per_gw) != set(other):
            return False, "gameweeks differ for player %d" % pid
        for gw, gwp in per_gw.items():
            o = other[gw]
            if gwp.xp != o.xp or gwp.sd != o.sd or len(gwp.fixtures) != len(o.fixtures):
                return False, "player %d GW%d differs (%r vs %r)" % (pid, gw, gwp.xp, o.xp)
            for fa, fb in zip(gwp.fixtures, o.fixtures):
                for name in _FIXTURE_FIELDS:
                    if getattr(fa, name) != getattr(fb, name):
                        return False, "player %d GW%d field %s differs (%r vs %r)" % (
                            pid, gw, name, getattr(fa, name), getattr(fb, name))
    return True, "identical"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def horizon_table(
    projections: ProjectionSet,
    state: "GameState",
    limit: int = 50,
    position: Optional[int] = None,
    min_xmins: float = 0.0,
) -> str:
    """Ranked table of total xP over the projection horizon."""
    gws = projections.gws()
    rows = []
    for pid, per_gw in projections.projections.items():
        player = state.players.get(pid)
        if player is None:
            continue
        if position is not None and player.position != position:
            continue
        total = sum(per_gw[g].xp for g in gws if g in per_gw)
        xmins = 0.0
        blanks = 0
        doubles = 0
        for g in gws:
            gwp = per_gw.get(g)
            if gwp is None or gwp.is_blank:
                blanks += 1
                continue
            if gwp.n_fixtures > 1:
                doubles += 1
            xmins += sum(f.xmins for f in gwp.fixtures)
        if xmins / max(len(gws), 1) < min_xmins:
            continue
        rows.append((total, pid, player, xmins / max(len(gws), 1), blanks, doubles))
    rows.sort(key=lambda r: -r[0])

    header = ("%-4s %-18s %-5s %-4s %6s %7s %7s %7s %6s"
              % ("#", "player", "team", "pos", "price", "xP/GW", "total", "per £m", "xmins"))
    lines = [header, "-" * len(header)]
    for i, (total, pid, player, xmins, blanks, doubles) in enumerate(rows[:limit], start=1):
        lines.append(
            "%-4d %-18s %-5s %-4s %6.1f %7.2f %7.2f %7.3f %6.1f%s"
            % (i, player.web_name[:18], state.short_name(player.team_id),
               scoring.POS_NAME[player.position], player.price,
               total / max(len(gws), 1), total, total / max(player.price, 0.1), xmins,
               ("  [%dB]" % blanks if blanks else "") + ("  [%dD]" % doubles if doubles else ""))
        )
    return "\n".join(lines)


def fixture_bonus_totals(engine: XPEngine, gws: Sequence[int]) -> Dict[int, float]:
    """``{fixture_id: sum of expected bonus}`` — must never exceed 6."""
    out: Dict[int, float] = {}
    wanted = set(int(g) for g in gws)
    for (fixture_id, gw), bundle in engine._bundles.items():
        if gw not in wanted:
            continue
        out[fixture_id] = sum(p.xp_bonus for p in bundle.values())
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _synthetic_double_blank(state: "GameState", gw: int) -> Tuple[Any, int, int, int, int]:
    """Fabricate a double and a blank in ``gw`` on a copy of the game state.

    Every club has exactly one fixture per gameweek today, so the double/blank
    paths would otherwise never execute before they matter. From roughly GW14
    onward, postponed matches get rescheduled into existing gameweeks and both
    cases go live at once — a club with two fixtures whose points must sum, and
    the club whose match was moved away, which now blanks.
    """
    import copy

    clone = copy.copy(state)
    clone.fixtures = [copy.copy(f) for f in state.fixtures]
    clone._players_df = None

    in_gw = sorted((f for f in clone.fixtures if f.gw is not None and int(f.gw) == gw),
                   key=lambda f: int(f.id))
    later = sorted((f for f in clone.fixtures if f.gw is not None and int(f.gw) == gw + 1),
                   key=lambda f: int(f.id))
    if len(in_gw) < 2 or not later:
        raise RuntimeError("need fixtures in GW%d and GW%d to build the synthetic case"
                           % (gw, gw + 1))

    # Postpone one of this gameweek's matches: both its clubs now blank.
    postponed = in_gw[0]
    blank_a, blank_b = int(postponed.team_h), int(postponed.team_a)
    postponed.gw = gw + 1

    # Reschedule one of next gameweek's matches into this one, picking a pair of
    # clubs that still have their own fixture here: both now play twice.
    moved = None
    for candidate in later:
        if candidate is postponed:
            continue
        if int(candidate.team_h) in (blank_a, blank_b) or int(candidate.team_a) in (blank_a, blank_b):
            continue
        moved = candidate
        break
    if moved is None:
        raise RuntimeError("no GW%d fixture available to reschedule into GW%d" % (gw + 1, gw))
    moved.gw = gw
    return clone, int(moved.team_h), int(moved.team_a), blank_a, blank_b


if __name__ == "__main__":
    import argparse

    from gaffer.data.loaders import load_game_state

    parser = argparse.ArgumentParser(description="xP assembler smoke test")
    parser.add_argument("--gws", default="", help="e.g. 1-6; defaults to the next 6")
    parser.add_argument("--refit", action="store_true", help="force every sub-model to refit")
    parser.add_argument("--draws", type=int, default=DEFAULT_MC_DRAWS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()

    t0 = time.time()
    game = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (game.summary(), time.time() - t0))

    if args.gws:
        lo, _, hi = args.gws.partition("-")
        GWS = list(range(int(lo), int(hi or lo) + 1))
    else:
        GWS = list(range(game.current_gw, game.current_gw + cfg.model.default_horizon))

    t0 = time.time()
    eng = XPEngine(cfg, mc_draws=args.draws).fit(game, refit=args.refit)
    fit_seconds = time.time() - t0
    print("fitted all six sub-models in %.2fs: %s"
          % (fit_seconds, ", ".join("%s %.2fs" % (k, v) for k, v in eng.fit_seconds.items())))
    print("league mean lambda %.3f, home advantage %.3f, card baselines %s"
          % (eng.team.mean_lambda, eng.team.home_adv,
             " ".join("%s %.3f y/90" % (scoring.POS_NAME[p], v)
                      for p, v in sorted(eng.cards.baseline_yellow90.items()))))
    if eng.warnings:
        print("warnings (%d):" % len(eng.warnings))
        for w in eng.warnings[:12]:
            print("   %s" % w)

    t0 = time.time()
    PS = eng.project(game, GWS)
    cold_seconds = time.time() - t0
    print("\ncold projection GW%d-%d: %d players x %d gameweeks in %.2fs "
          "(bundles %.2fs, assemble %.2fs)"
          % (GWS[0], GWS[-1], len(PS.projections), len(GWS), cold_seconds,
             eng.timings["bundles_seconds"], eng.timings["assemble_seconds"]))

    t0 = time.time()
    PS_WARM = eng.project(game, GWS)
    warm_seconds = time.time() - t0
    print("warm projection (memoised): %.3fs" % warm_seconds)

    # ---------------------------------------------------------------- tables
    print("\n=== TOP 50 BY TOTAL xP, GW%d-%d ===" % (GWS[0], GWS[-1]))
    print(horizon_table(PS, game, limit=50))

    for POS in scoring.POSITIONS:
        print("\n=== TOP 15 %s ===" % scoring.POS_NAME[POS])
        print(horizon_table(PS, game, limit=15, position=POS))

    # -------------------------------------------------------------- explains
    print("\n" + "=" * 78)
    print("FULL EXPLAIN DUMPS")
    print("=" * 78)
    BEST: Dict[int, int] = {}
    for pid_, per_gw_ in PS.projections.items():
        pos_ = game.players[pid_].position
        total_ = sum(per_gw_[g].xp for g in GWS if g in per_gw_)
        if pos_ not in BEST or total_ > sum(
                PS.projections[BEST[pos_]][g].xp for g in GWS if g in PS.projections[BEST[pos_]]):
            BEST[pos_] = pid_
    for POS in (scoring.FWD, scoring.MID, scoring.DEF, scoring.GKP):
        print("\n" + eng.explain(BEST[POS], GWS[0]))

    # ----------------------------------------------------------------- gates
    print("\n" + "=" * 78)
    print("SANITY GATES")
    print("=" * 78)
    FAILURES: List[str] = []

    def gate(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            FAILURES.append(label)

    # 1. Point bands.
    NEGATIVE = []
    HIGH = []
    NAN = []
    for pid_, per_gw_ in PS.projections.items():
        for g_, gwp_ in per_gw_.items():
            if gwp_.xp != gwp_.xp or gwp_.sd != gwp_.sd:
                NAN.append((pid_, g_))
            starter = any(f.p_start >= 0.5 for f in gwp_.fixtures)
            if starter and gwp_.xp < 0:
                NEGATIVE.append((pid_, g_, gwp_.xp))
            if gwp_.xp > 15.0:
                HIGH.append((pid_, g_, gwp_.xp))
    gate("no NaN in any xp or sd", not NAN, "%d rows checked"
         % sum(len(v) for v in PS.projections.values()))
    gate("no negative GW xP for a likely starter", not NEGATIVE,
         "worst %s" % (min(NEGATIVE, key=lambda r: r[2]) if NEGATIVE else "n/a"))
    MAXROW = max(((gwp_.xp, pid_, g_) for pid_, per in PS.projections.items()
                  for g_, gwp_ in per.items()), default=(0.0, 0, 0))
    gate("no single-GW xP above 15", not HIGH,
         "max %.2f (%s GW%d)" % (MAXROW[0], game.players[MAXROW[1]].web_name, MAXROW[2]))

    # 2. Bonus conservation.
    BONUS_SUMS = fixture_bonus_totals(eng, GWS)
    WORST_BONUS = max(BONUS_SUMS.values()) if BONUS_SUMS else 0.0
    gate("per-fixture expected bonus <= 6", WORST_BONUS <= 6.0 + 1e-6,
         "%d fixtures, max %.6f, min %.6f"
         % (len(BONUS_SUMS), WORST_BONUS, min(BONUS_SUMS.values()) if BONUS_SUMS else 0.0))

    # 3. No field on PlayerFixtureProjection is dead. The dashboard shows the
    #    whole breakdown, so a component silently left at its default would be
    #    invisible in the total but wrong on the screen.
    FILLED = {n: 0 for n in _FIXTURE_FIELDS}
    ROWS = 0
    for pid_, per_gw_ in PS.projections.items():
        for f_ in per_gw_[GWS[0]].fixtures:
            ROWS += 1
            for n in _FIXTURE_FIELDS:
                if getattr(f_, n):
                    FILLED[n] += 1
    DEAD = [n for n, c in FILLED.items() if c == 0]
    gate("no field on PlayerFixtureProjection is dead", not DEAD,
         "%d fields over %d rows, rarest %s"
         % (len(_FIXTURE_FIELDS), ROWS,
            ", ".join("%s %d" % (n, c) for n, c in sorted(FILLED.items(),
                                                          key=lambda kv: kv[1])[:3])))
    #    Position-aware, because some zeros are the rules rather than a bug: a
    #    forward earns nothing for a clean sheet and a keeper is DEFCON-ineligible.
    COMMON = ("p_appear", "p_start", "p_60", "xmins", "lambda_goals", "lambda_assists",
              "lambda_conceded", "p_clean_sheet", "exp_bps", "xp_appearance", "xp_goals",
              "xp_assists", "xp_bonus", "xp_cards", "xp_total", "sd_total")
    EXPECTED = {
        scoring.GKP: COMMON + ("lambda_saves", "xp_saves", "xp_clean_sheet",
                               "xp_goals_conceded", "xp_penalty"),
        scoring.DEF: COMMON + ("xp_clean_sheet", "xp_goals_conceded", "p_defcon", "xp_defcon"),
        scoring.MID: COMMON + ("xp_clean_sheet", "p_defcon", "xp_defcon"),
        scoring.FWD: COMMON + ("p_defcon", "xp_defcon"),
    }
    UNSET = []
    for POS in scoring.POSITIONS:
        TOP = PS.projections[BEST[POS]][GWS[0]].fixtures[0]
        UNSET.extend("%s.%s" % (scoring.POS_NAME[POS], n)
                     for n in EXPECTED[POS] if getattr(TOP, n) == 0.0)
    gate("every field the rules allow is filled, per position", not UNSET,
         "unset: %s" % (UNSET or "none, across the best GKP/DEF/MID/FWD"))

    # 4. Component sum equals the total, exactly.
    WORST_SUM = 0.0
    for pid_, per_gw_ in PS.projections.items():
        for g_, gwp_ in per_gw_.items():
            for f_ in gwp_.fixtures:
                WORST_SUM = max(WORST_SUM, abs(sum(f_.components().values()) - f_.xp_total))
            WORST_SUM = max(WORST_SUM, abs(sum(f_.xp_total for f_ in gwp_.fixtures) - gwp_.xp))
    gate("components sum to xp_total and to the GW xp", WORST_SUM < 1e-9,
         "max residual %.2e" % WORST_SUM)

    # 5. The simulation reproduces the analytic expectation. This is the real
    #    check on the component maths: every lambda, every gate and every
    #    probability has to line up for 3,000 independent simulations to land
    #    on the closed-form total.
    #    The comparison is in standard errors, not raw points: with 4,000 draws
    #    a 5-point player's simulated mean carries an se of about 0.08, so a
    #    raw-points threshold would either pass everything or fail on noise.
    #    A systematic error shows up in the *average* signed gap instead.
    GAPS = []
    COUPLING = []
    for (pid_, fid_), det_ in eng._detail.items():
        proj_ = eng._bundles.get((fid_, GWS[0]), {}).get(pid_)
        if proj_ is None or "mc_mean" not in det_:
            continue
        if proj_.xp_total >= 1.0 and proj_.sd_total > 0.1:
            se_ = proj_.sd_total / math.sqrt(eng.mc_draws)
            GAPS.append(((det_["mc_mean"] - proj_.xp_total) / se_,
                         det_["mc_mean"] - proj_.xp_total, pid_))
        var_ = det_.get("variance", {})
        indep_ = sum(v for k, v in var_.items() if k != "coupling")
        if indep_ > 0.25:
            COUPLING.append(var_.get("coupling", 0.0) / indep_)
    MAXZ = max((abs(g[0]), g[2]) for g in GAPS) if GAPS else (0.0, 0)
    BIAS = sum(g[1] for g in GAPS) / max(len(GAPS), 1)
    gate("simulated mean matches the analytic total (no bias)",
         abs(BIAS) < 0.01 and MAXZ[0] < 4.5,
         "mean signed gap %+.4f pts, worst |z| %.2f (%s) over %d player-fixtures"
         % (BIAS, MAXZ[0], game.players[MAXZ[1]].web_name if GAPS else "-", len(GAPS)))
    CLIPPED = sum(d.get("mc_clipped", 0) for d in eng._detail.values())
    gate("no simulated total fell outside the distribution support", CLIPPED == 0,
         "support [%d, %d], %d draws x %d player-fixtures"
         % (DIST_MIN, DIST_MAX, eng.mc_draws, len(eng._detail)))
    POSITIVE = sum(1 for c in COUPLING if c > 0)
    gate("component coupling is positive and bounded",
         POSITIVE >= 0.95 * len(COUPLING) and max(COUPLING) < 1.5,
         "positive for %d/%d, median %.2f, max %.2f of the independent sum"
         % (POSITIVE, len(COUPLING), sorted(COUPLING)[len(COUPLING) // 2], max(COUPLING)))

    # 6. Determinism, with every memo cleared so the second run really recomputes.
    eng.reset_caches()
    t0 = time.time()
    PS2 = eng.project(game, GWS)
    RERUN = time.time() - t0
    SAME, WHY = projection_sets_equal(PS, PS2)
    gate("project() is deterministic", SAME, "%s (recomputed in %.2fs)" % (WHY, RERUN))

    # 7. Cache round trip.
    PATH = eng.save_projections(PS)
    LOADED = eng.load_projections(PS.first_gw, PS.last_gw)
    OK, WHY2 = projection_sets_equal(PS, LOADED) if LOADED else (False, "load returned None")
    gate("projection cache round trip is lossless", OK,
         "%s -> %.1f KB, %s" % (os.path.basename(PATH), os.path.getsize(PATH) / 1024.0, WHY2))

    # 8. Haul probabilities.
    TOPC = sorted(((eng.p_haul(pid_, GWS[0]), pid_) for pid_ in PS.projections), reverse=True)[:8]
    print("\n  captaincy shortlist for GW%d (P(>=10) | xP | sd):" % GWS[0])
    for p_, pid_ in TOPC:
        g_ = PS.projections[pid_][GWS[0]]
        print("    %-18s %-4s %5.1f%%  %5.2f  %4.2f"
              % (game.players[pid_].web_name, scoring.POS_NAME[game.players[pid_].position],
                 100 * p_, g_.xp, g_.sd))
    MONO = all(eng.p_haul(pid_, GWS[0]) <= 1.0 for pid_ in PS.projections)
    gate("haul probabilities are valid", MONO and TOPC[0][0] > 0.02,
         "best %.1f%% (%s)" % (100 * TOPC[0][0], game.players[TOPC[0][1]].web_name))

    # 9. Doubles and blanks, on a fabricated schedule.
    print("\n  --- synthetic double / blank gameweek ---")
    CLONE, DA, DB, BA, BB = _synthetic_double_blank(game, GWS[0])
    eng2 = XPEngine(cfg, mc_draws=1000)
    for attr in ("team", "minutes", "attacking", "defending", "defcon", "bonus"):
        setattr(eng2, attr, getattr(eng, attr))
    eng2.cards = eng.cards
    eng2.fitted = True
    eng2.state = CLONE
    PS_DB = eng2.project(CLONE, [GWS[0]])
    DOUBLE_PID = max((p for p in CLONE.players if CLONE.players[p].team_id == DA),
                     key=lambda p: PS_DB.projections[p][GWS[0]].xp)
    BLANK_PID = max((p for p in CLONE.players if CLONE.players[p].team_id == BA),
                    key=lambda p: game.players[p].now_cost)
    DGW = PS_DB.projections[DOUBLE_PID][GWS[0]]
    BGW = PS_DB.projections[BLANK_PID][GWS[0]]
    SINGLE = PS.projections[DOUBLE_PID][GWS[0]]
    print("    double: %s (%s) %d fixtures, xP %.2f = %s | single-GW baseline %.2f"
          % (CLONE.players[DOUBLE_PID].web_name, CLONE.short_name(DA), DGW.n_fixtures,
             DGW.xp, " + ".join("%.2f" % f.xp_total for f in DGW.fixtures), SINGLE.xp))
    print("    blank : %s (%s) %d fixtures, xP %.2f, is_blank=%s"
          % (CLONE.players[BLANK_PID].web_name, CLONE.short_name(BA), BGW.n_fixtures,
             BGW.xp, BGW.is_blank))
    print("    haul  : P(>=10) single %.1f%% -> double %.1f%%"
          % (100 * eng.p_haul(DOUBLE_PID, GWS[0]), 100 * eng2.p_haul(DOUBLE_PID, GWS[0])))
    gate("double gameweek yields 2 fixtures that sum", DGW.n_fixtures == 2
         and abs(DGW.xp - sum(f.xp_total for f in DGW.fixtures)) < 1e-9
         and DGW.xp > SINGLE.xp,
         "%.2f > %.2f" % (DGW.xp, SINGLE.xp))
    gate("double gameweek sd is the pooled sd, not the sum",
         abs(DGW.sd - math.sqrt(sum(f.sd_total ** 2 for f in DGW.fixtures))) < 1e-9,
         "sd %.3f vs naive sum %.3f"
         % (DGW.sd, sum(f.sd_total for f in DGW.fixtures)))
    gate("double gameweek raises P(haul)",
         eng2.p_haul(DOUBLE_PID, GWS[0]) > eng.p_haul(DOUBLE_PID, GWS[0]),
         "%.3f -> %.3f" % (eng.p_haul(DOUBLE_PID, GWS[0]), eng2.p_haul(DOUBLE_PID, GWS[0])))
    BLANK_TEAMS = [t for t in (BA, BB) if t]
    BLANK_OK = True
    for team_ in BLANK_TEAMS:
        for pid_ in (p for p in CLONE.players if CLONE.players[p].team_id == team_):
            row = PS_DB.projections[pid_][GWS[0]]
            if not row.is_blank or row.xp != 0.0 or row.sd != 0.0 or row.fixtures:
                BLANK_OK = False
    gate("blank gameweek gives xp 0, is_blank True, no KeyError", BLANK_OK,
         "%d clubs blanked, %d players"
         % (len(BLANK_TEAMS),
            sum(1 for p in CLONE.players if CLONE.players[p].team_id in BLANK_TEAMS)))
    ALL_PRESENT = all(GWS[0] in PS_DB.projections[p] for p in CLONE.players)
    gate("every player has a row in every projected gameweek", ALL_PRESENT,
         "%d players" % len(PS_DB.projections))
    DB_BONUS = fixture_bonus_totals(eng2, [GWS[0]])
    gate("bonus still conserved with a rescheduled fixture",
         max(DB_BONUS.values()) <= 6.0 + 1e-6,
         "%d fixtures, max %.6f" % (len(DB_BONUS), max(DB_BONUS.values())))
    print("\n" + eng2.explain(DOUBLE_PID, GWS[0]))
    print("\n" + eng2.explain(BLANK_PID, GWS[0]))

    # 10. Position shares, a football sanity check rather than a code one.
    print("\n  --- projected points share by position, GW%d ---" % GWS[0])
    SHARE: Dict[int, float] = {}
    for pid_, per_gw_ in PS.projections.items():
        SHARE[game.players[pid_].position] = SHARE.get(game.players[pid_].position, 0.0) + \
            per_gw_[GWS[0]].xp
    TOT = sum(SHARE.values())
    for POS in scoring.POSITIONS:
        print("    %-4s %6.1f xP  %5.1f%%" % (scoring.POS_NAME[POS], SHARE[POS],
                                              100 * SHARE[POS] / TOT))

    print("\n  runtime: fit %.2fs | cold project %.2fs | warm project %.3fs | "
          "recompute %.2fs" % (fit_seconds, cold_seconds, warm_seconds, RERUN))
    print("\n%s" % ("ALL GATES PASSED" if not FAILURES
                    else "FAILED GATES (%d): %s" % (len(FAILURES), FAILURES)))
    raise SystemExit(1 if FAILURES else 0)

