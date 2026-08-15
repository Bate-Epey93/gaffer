"""Minutes model — the multiplier on every other point component.

Nothing else in gaffer matters if this is wrong: a 6.0 xP striker who does not
start is worth 0. The model answers four questions per player per gameweek:

    p_start   P(named in the starting XI)
    p_appear  P(minutes >= 1)
    p_60      P(minutes >= 60)   -- conditioned on starting, see below
    xmins     E[minutes]

Three ideas do the work.

**1. Evidence hierarchy with beta-binomial shrinkage.** Start opportunities are
Bernoulli trials. Current-season starts (exponentially decayed towards the most
recent gameweeks) are the strongest evidence, prior-season starts are weaker,
and a fitted position/price prior carries everything else. Each source
contributes (successes, trials) and they are pooled through
``stats.beta_binomial_rate`` with ``config.model.shrinkage_90s_minutes`` as the
prior strength. The *effective* trials a prior season is worth are not its raw
38 — they are set so that the posterior reproduces the measured cross-season
regression of start rate on start rate (slope 0.654 for players who stayed at
their club, 0.242 for players who moved: a new signing's old role says very
little about their new one).

**2. Everything empirical is fitted, nothing is asserted.** The price prior, the
club-relative price rank effect, P(60+ | started), the substitute-appearance
curve, the recency half-life, the injury-recovery and return-from-injury ramps
are all estimated from last season's per-match data (vaastav ``merged_gw``) and
written to ``reports/minutes_price_prior.json``. A run with no network reloads
that file rather than inventing constants.

**3. Club-relative price beats absolute price.** How expensive a player is
*relative to his own club's players in the same position* predicts starting far
better than his raw price (5-fold CV grouped by club: log-loss 0.488 with the
rank feature versus 0.527 with price alone). It also solves the promoted-club
problem for free — Coventry's whole squad is cheap, so absolute price says
"nobody starts", while the rank says who their first-choice XI is.

``p_60`` is conditioned on starting because substitutes essentially never reach
60 minutes: of 3136 substitute appearances in 2025/26, 1.4% reached the hour.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import scoring, stats
from gaffer.core.config import REPORTS_DIR, Config, ensure_dirs
from gaffer.core.types import MinutesForecast, Player
from gaffer.data import history as hist
from gaffer.data.cache import Cache

log = logging.getLogger(__name__)

PRIOR_REPORT_PATH = os.path.join(REPORTS_DIR, "minutes_price_prior.json")

# `n` is the "not in squad" flag the API uses for unregistered players; it is
# absent from ModelConfig.status_multiplier in some configs, so it is listed
# here explicitly alongside the other hard-out codes.
HARD_OUT_STATUSES = frozenset(("i", "s", "u", "n"))
DOUBTFUL_STATUS = "d"

# Club-relative price rank is capped: past the 9th most expensive player in a
# position group the pecking order stops carrying information (every club has a
# long tail of £4.0m youth-team names).
RANK_CAP = 9

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_PCT_RE = re.compile(r"(\d{1,3})\s*%\s*chance", re.I)
_DATE_RE = re.compile(r"(?:expected back|until|return[s]?(?:\s+on)?)\s+(\d{1,2})\s*([A-Za-z]{3,9})", re.I)
_UNKNOWN_RE = re.compile(r"unknown return date", re.I)
_DEPARTED_RE = re.compile(r"\b(?:has\s+)?(?:joined|returned to|left)\b", re.I)
_LOAN_RE = re.compile(r"on loan", re.I)
_SUSPENDED_RE = re.compile(r"suspend", re.I)
_KNOCK_RE = re.compile(r"\bknock\b", re.I)
_INJURY_RE = re.compile(r"([A-Za-z][A-Za-z\- ]*?)\s+injury", re.I)


# ---------------------------------------------------------------------------
# News parsing
# ---------------------------------------------------------------------------


@dataclass
class NewsSignal:
    """What the free-text `news` field actually tells us.

    The 2026/27 API writes these to a tight template, verified against all 73
    flagged players on 2026-08-14:
        'Shin injury - 75% chance of playing'
        'Achilles injury - Unknown return date'
        'Groin injury - Expected back 22 Aug'
        'Suspended until 6 Sep'
        'Has joined Getafe permanently'
    Anything that does not match falls back to the status code alone, which is
    the safe direction: unparsed news never *raises* availability.
    """

    raw: str = ""
    chance: Optional[float] = None        # 0-1 from "75% chance of playing"
    return_date: Optional[datetime] = None
    indefinite: bool = False              # "Unknown return date"
    departed: bool = False                # transferred or loaned away
    suspended: bool = False
    knock: bool = False
    problem: str = ""                     # "Shin", "Achilles", "Suspension"

    def label(self) -> str:
        if self.departed:
            return "left the club"
        bits = []
        if self.problem:
            bits.append(self.problem.lower())
        if self.return_date is not None:
            bits.append("back %s" % self.return_date.strftime("%d %b"))
        elif self.indefinite:
            bits.append("no return date")
        return ", ".join(bits)


def _resolve_year(day: int, month: int, now: datetime) -> datetime:
    """FPL return dates carry no year. Pick the nearest sensible one.

    A date more than three months behind `now` is next year's (a February
    return read in November); everything else is this year's.
    """
    for year in (now.year, now.year + 1, now.year - 1):
        try:
            candidate = datetime(year, month, day, 12, 0, tzinfo=timezone.utc)
        except ValueError:
            continue
        delta_days = (candidate - now).days
        if -90 <= delta_days <= 300:
            return candidate
    return datetime(now.year, month, min(day, 28), 12, 0, tzinfo=timezone.utc)


def parse_news(
    news: str,
    status: str = "a",
    chance_of_playing: Optional[int] = None,
    now: Optional[datetime] = None,
) -> NewsSignal:
    """Turn the API's `news` string into structured availability signal."""
    now = now or datetime.now(timezone.utc)
    sig = NewsSignal(raw=news or "")
    text = (news or "").strip()
    if chance_of_playing is not None:
        sig.chance = max(0.0, min(1.0, chance_of_playing / 100.0))
    if not text:
        return sig

    m = _PCT_RE.search(text)
    if m:
        sig.chance = max(0.0, min(1.0, int(m.group(1)) / 100.0))

    if _UNKNOWN_RE.search(text):
        sig.indefinite = True

    m = _DATE_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(2)[:3].lower())
        if month:
            sig.return_date = _resolve_year(int(m.group(1)), month, now)

    if _SUSPENDED_RE.search(text):
        sig.suspended = True
        sig.problem = "Suspension"
    if _KNOCK_RE.search(text):
        sig.knock = True
        sig.problem = sig.problem or "Knock"
    m = _INJURY_RE.search(text)
    if m:
        sig.problem = m.group(1).strip().title()

    # 'u' plus departure phrasing is the only combination that means gone for
    # good; a bare "joined" in other contexts (a new signing) must not zero the
    # player out, so require the status too.
    if status == "u" and (_DEPARTED_RE.search(text) or _LOAN_RE.search(text)):
        sig.departed = True
    return sig


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------


@dataclass
class MinutesFit:
    """Every empirical quantity the model needs, all estimated from real data."""

    fitted_from_season: str = ""
    fitted_at: str = ""
    n_players: int = 0
    n_player_matches: int = 0
    # config.model.shrinkage_90s_minutes at fit time. Several fitted quantities
    # are expressed relative to it, so a cached fit made at a different strength
    # must not be reused.
    shrinkage_strength: float = 0.0

    # logit p_start = a + b_logprice * log(price) + b_rank * club_position_rank
    price_prior: Dict[str, Dict[str, float]] = field(default_factory=dict)
    price_prior_cv_logloss: Dict[str, float] = field(default_factory=dict)
    # Required diagnostic: raw P(start) by position and price decile.
    price_decile_table: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    rank_table: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    prior_floor: float = 0.005
    prior_cap: float = 0.92

    # P(60+ | started), pooled by position and personalised via minutes/start.
    p60_given_start_pos: Dict[str, float] = field(default_factory=dict)
    mins_per_start_pos: Dict[str, float] = field(default_factory=dict)
    # logit P(60|start) = a + b * mins_per_start
    p60_from_mins_per_start: List[float] = field(default_factory=list)
    p60_given_sub: float = 0.0
    # Substitutes actually used per club per match: the target the per-club
    # substitute mass is normalised to. `minutes_per_team_match` is the 11 x 90
    # a club distributes in one match, reported so the resulting xmins total can
    # be checked against it without being forced to match.
    subs_per_team_match: float = 0.0
    minutes_per_team_match: float = 990.0
    # Starters per club per match by position (GKP is exactly 1; the outfield
    # split moves with formation, so it is measured rather than assumed).
    starters_per_team_match: Dict[str, float] = field(default_factory=dict)

    # P(appear | did not start) and E[minutes | sub appearance], as functions of
    # the player's own start rate. Knots are (start_rate, value).
    sub_appear_curve: List[List[float]] = field(default_factory=list)
    sub_minutes_curve: List[List[float]] = field(default_factory=list)

    # Reliability of a full prior season for predicting this season's start
    # rate: the slope of next-season rate on last-season rate.
    prior_season_reliability: float = 0.0
    new_club_reliability: float = 0.0
    within_season_reliability: float = 0.0
    recency_half_life_gws: float = 6.0
    # Half-life, in current-season matches, over which last season's evidence
    # stops being worth anything. Without it a player who has started the last
    # six can still rank below a benched premium, because the prior season and
    # the price prior together out-vote what is happening now.
    prior_decay_matches: float = 8.0
    recency_fit_logloss: float = 0.0

    # Availability dynamics, all measured on regular starters (>=50% start rate).
    cohort_baseline_start_rate: float = 0.0
    # P(start h matches later | absent the last two), h = 1..6, as a fraction of
    # the cohort baseline. Drives "injured, unknown return date" over a horizon.
    absence_recovery: List[float] = field(default_factory=list)
    # P(start k matches after a >=3 match absence ends), as a fraction of the
    # baseline. Drives the ramp after a known return date.
    return_ramp: List[float] = field(default_factory=list)
    # P(start h later | started now), as a fraction of baseline. Reported for
    # calibration; not applied (see the note in _availability).
    fitness_persistence: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = _jsonable(value)
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MinutesFit":
        fit = cls()
        for key, value in raw.items():
            if hasattr(fit, key):
                setattr(fit, key, value)
        return fit


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _expit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def _interp(knots: Sequence[Sequence[float]], x: float) -> float:
    """Piecewise-linear interpolation with flat extrapolation."""
    if not knots:
        return 0.0
    pts = sorted((float(k[0]), float(k[1])) for k in knots)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def _frac_logit(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 400, ridge: float = 1e-3
) -> np.ndarray:
    """Fractional (quasi-binomial) logistic regression by Newton-Raphson.

    `y` is a rate in [0, 1] and `w` the weight behind it, which is the right
    likelihood for "this player started 24 of his 31 available matches".
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (w * (y - p)) - ridge * beta
        weights = w * p * (1.0 - p) + 1e-9
        hess = (X * weights[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-11:
            break
    return beta


def _frac_logit_offset(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray, iters: int = 400
) -> np.ndarray:
    """`_frac_logit` with a fixed linear-predictor offset (a borrowed slope)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    off = np.asarray(offset, dtype=float)
    ridge = 1e-3
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ beta + off, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (w * (y - p)) - ridge * beta
        weights = w * p * (1.0 - p) + 1e-9
        hess = (X * weights[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        # Cap the step: with a fixed offset the likelihood can be near-flat in
        # one direction and an undamped Newton step runs off to infinity.
        norm = float(np.max(np.abs(step)))
        if norm > 4.0:
            step = step * (4.0 / norm)
        beta = beta + step
        if norm < 1e-11:
            break
    return beta


def _weighted_logloss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) / np.sum(w))


def club_position_price_ranks(
    prices: Sequence[float], team_ids: Sequence[int], positions: Sequence[int]
) -> List[float]:
    """How many same-position team-mates are more expensive. Ties share a rank.

    0 means "the most expensive goalkeeper/defender/midfielder/forward at this
    club", which is the single best pre-season predictor of starting we found.
    """
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, (team, pos) in enumerate(zip(team_ids, positions)):
        groups.setdefault((int(team), int(pos)), []).append(idx)
    out = [0.0] * len(prices)
    for members in groups.values():
        values = [float(prices[i]) for i in members]
        for i in members:
            v = float(prices[i])
            above = sum(1 for u in values if u > v)
            tied = sum(1 for u in values if u == v)
            out[i] = above + (tied - 1) / 2.0
    return out


# ---------------------------------------------------------------------------
# Per-player evidence
# ---------------------------------------------------------------------------


@dataclass
class _Role:
    """A player's long-run role, before any per-gameweek availability logic."""

    player_id: int
    p_start: float = 0.0
    p_sub_given_no_start: float = 0.0
    p60_given_start: float = 0.0
    mins_per_start: float = 0.0
    mins_per_sub: float = 0.0
    evidence: str = ""
    n_trials: float = 0.0
    price_prior: float = 0.0
    rank: float = 0.0
    source: str = "prior"


class MinutesModel:
    """Fit once per data refresh, then forecast any (player, gameweek)."""

    def __init__(self, config: Optional[Config] = None, normalize_to_eleven: bool = True) -> None:
        self.config = config or Config.load()
        # Every club fields exactly 11 starters and uses about four substitutes
        # per match. That is an arithmetic fact, not a modelling assumption, and
        # enforcing it is what makes injuries redistribute minutes to the
        # players who will actually cover — without it the squad-wide minutes
        # total falls ~12% short and every downstream xP is biased low.
        self.normalize_to_eleven = normalize_to_eleven
        self.fit_params: Optional[MinutesFit] = None
        self._roles: Dict[int, _Role] = {}
        self._ranks: Dict[int, float] = {}
        self._news: Dict[int, NewsSignal] = {}
        self._gw_dates: Dict[int, datetime] = {}
        self._europe_team_ids: List[int] = []
        self._team_members: Dict[int, List[int]] = {}
        self._team_scale: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._state_current_gw: int = 1
        self._now: datetime = datetime.now(timezone.utc)

    # -- fitting ------------------------------------------------------------

    def fit(self, state: Any, force: bool = False, write_report: bool = True) -> MinutesFit:
        """Estimate every parameter, then precompute each player's role.

        The fit needs last season's per-match rows. If they cannot be fetched
        and a previous fit is on disk, that is reloaded — never silently
        replaced by made-up numbers.
        """
        self._now = datetime.now(timezone.utc)
        prior_season = getattr(state, "prior_season", "") or hist.prior_season(self.config.season)
        fit_params: Optional[MinutesFit] = None
        if not force:
            fit_params = self._load_report(prior_season)
        if fit_params is None:
            try:
                fit_params = self._fit_from_season(prior_season)
            except Exception as exc:
                cached = self._load_report(None)
                if cached is None:
                    raise RuntimeError(
                        "minutes model could not fit from %s (%s: %s) and no "
                        "previous fit exists at %s"
                        % (prior_season, type(exc).__name__, exc, PRIOR_REPORT_PATH)
                    )
                log.warning(
                    "minutes fit failed (%s: %s); reusing %s fitted at %s",
                    type(exc).__name__, exc, PRIOR_REPORT_PATH, cached.fitted_at,
                )
                fit_params = cached
            else:
                if write_report:
                    self._write_report(fit_params)
        self.fit_params = fit_params
        self._prepare_state(state)
        return fit_params

    def _load_report(self, expect_season: Optional[str]) -> Optional[MinutesFit]:
        if not os.path.exists(PRIOR_REPORT_PATH):
            return None
        try:
            with open(PRIOR_REPORT_PATH, "r") as fh:
                raw = json.load(fh)
        except (ValueError, OSError):
            return None
        fit_params = MinutesFit.from_dict(raw)
        if not fit_params.price_prior:
            return None
        if expect_season is not None:
            if hist.season_dash(fit_params.fitted_from_season or "1900-01") != hist.season_dash(expect_season):
                return None
            if abs(fit_params.shrinkage_strength - float(self.config.model.shrinkage_90s_minutes)) > 1e-9:
                log.info("cached minutes fit used shrinkage %.3f, config says %.3f; refitting",
                         fit_params.shrinkage_strength, self.config.model.shrinkage_90s_minutes)
                return None
        return fit_params

    def _write_report(self, fit_params: MinutesFit) -> None:
        ensure_dirs()
        payload = fit_params.to_dict()
        payload["_grid"] = {
            pos: {
                "%.1f" % price: [
                    round(self._prior_from_coefs(fit_params, pos, price, rank), 4)
                    for rank in range(0, 8)
                ]
                for price in (4.0, 4.5, 5.0, 5.5, 6.5, 8.0, 11.0, 14.5)
            }
            for pos in ("GKP", "DEF", "MID", "FWD")
        }
        payload["_grid_note"] = "position -> price -> [P(start) at club-position price rank 0..7]"
        with open(PRIOR_REPORT_PATH, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        log.info("wrote %s", PRIOR_REPORT_PATH)

    def _fit_from_season(self, season: str) -> MinutesFit:
        cache = Cache(default_ttl=self.config.cache_ttl_seconds)
        gws = hist.load_vaastav_gws(season, cache=cache)
        needed = ("element", "GW", "starts", "minutes", "value", "element_type", "team_short")
        missing = [c for c in needed if c not in gws.columns]
        if missing:
            raise ValueError("vaastav %s is missing columns %s" % (season, missing))

        fit_params = MinutesFit(
            fitted_from_season=hist.season_dash(season),
            fitted_at=self._now.isoformat(),
            n_players=int(gws["element"].nunique()),
            n_player_matches=int(len(gws)),
            shrinkage_strength=float(self.config.model.shrinkage_90s_minutes),
            recency_half_life_gws=6.0,
        )

        ordered = gws.sort_values(["element", "GW", "fixture"], kind="mergesort")
        per_player = ordered.groupby("element").agg(
            et=("element_type", "first"),
            team=("team_short", "first"),
            n=("starts", "size"),
            starts=("starts", "sum"),
        )
        first_row = ordered.groupby("element").first()
        per_player["price"] = first_row["value"] / 10.0
        per_player = per_player[per_player["price"] > 0]
        per_player["rate"] = per_player["starts"] / per_player["n"]
        per_player["weight"] = per_player["n"] / float(scoring.TOTAL_GWS)
        per_player["pos"] = per_player["et"].map(scoring.POS_NAME)
        per_player["rank"] = club_position_price_ranks(
            per_player["price"].tolist(),
            [hash(t) for t in per_player["team"].tolist()],
            per_player["et"].tolist(),
        )

        self._fit_price_prior(fit_params, per_player)
        self._fit_minutes_shapes(fit_params, ordered, per_player)
        self._fit_availability_dynamics(fit_params, ordered, per_player)
        prev = self._fit_cross_season(fit_params, season, per_player, cache)
        self._fit_recency(fit_params, ordered, per_player, prev)
        return fit_params

    # -- price / rank prior -------------------------------------------------

    def _fit_price_prior(self, fit_params: MinutesFit, per_player: pd.DataFrame) -> None:
        """logit P(start) = a + b_logprice*log(price) + b_rank*club_position_rank.

        Fitted per position with each player weighted by the share of the
        season they were registered for. The rank term is capped at RANK_CAP so
        a 20th-choice midfielder is not extrapolated to an absurd log-odds.
        """
        prior: Dict[str, Dict[str, float]] = {}
        cv: Dict[str, float] = {}
        deciles: Dict[str, List[Dict[str, float]]] = {}
        ranks: Dict[str, List[Dict[str, float]]] = {}

        # Pass 1: unconstrained per position.
        raw_fits: Dict[str, np.ndarray] = {}
        for et in scoring.POSITIONS:
            pos = scoring.POS_NAME[et]
            sub = per_player[per_player["et"] == et]
            if len(sub) < 12:
                continue
            raw_fits[pos] = _frac_logit(
                self._prior_design(sub["price"].values, sub["rank"].values),
                sub["rate"].values,
                sub["weight"].values,
            )

        # Within defenders, club-position rank and price are nearly the same
        # variable (the priciest defender is rank 0 by construction), so the
        # price slope is not separately identified and can come out negative —
        # which would say an expensive defender is *less* likely to start. When
        # that happens, borrow the slope from the positions where it is
        # identified and refit the intercept and rank term around it.
        positive = [b[1] for b in raw_fits.values() if b[1] > 0]
        borrowed = float(np.mean(positive)) if positive else 0.0

        for et in scoring.POSITIONS:
            pos = scoring.POS_NAME[et]
            sub = per_player[per_player["et"] == et]
            if pos not in raw_fits:
                continue
            beta = self._fit_prior_beta(sub, borrowed)
            borrowed_slope = bool(raw_fits[pos][1] < 0)
            prior[pos] = {
                "a": float(beta[0]),
                "b_logprice": float(beta[1]),
                "b_rank": float(beta[2]),
                "n_players": int(len(sub)),
                "price_slope_borrowed": bool(borrowed_slope),
            }
            cv[pos] = self._cv_logloss(sub, borrowed)
            deciles[pos] = self._decile_table(sub)
            ranks[pos] = self._rank_table(sub)

        fit_params.price_prior = prior
        fit_params.price_prior_cv_logloss = cv
        fit_params.price_decile_table = deciles
        fit_params.rank_table = ranks
        # Nothing about a price tag alone should imply a near-certain start:
        # the 95th percentile of observed season start rates is the honest cap.
        regulars = per_player[per_player["n"] >= 30]
        if len(regulars):
            fit_params.prior_cap = float(min(0.95, max(0.80, regulars["rate"].quantile(0.95))))

    @staticmethod
    def _prior_design(prices: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.ones(len(prices)),
                np.log(np.maximum(np.asarray(prices, dtype=float), 0.1)),
                np.minimum(np.asarray(ranks, dtype=float), RANK_CAP),
            ]
        )

    @staticmethod
    def _team_folds(teams: Sequence[str], k: int = 5) -> List[int]:
        uniq = sorted(set(teams))
        rng = np.random.default_rng(11)
        order = rng.permutation(len(uniq))
        assign = {uniq[int(o)]: i % k for i, o in enumerate(order)}
        return [assign[t] for t in teams]

    def _fit_prior_beta(self, sub: pd.DataFrame, borrowed: float) -> np.ndarray:
        """[a, b_logprice, b_rank], borrowing the price slope when unidentified."""
        beta = _frac_logit(
            self._prior_design(sub["price"].values, sub["rank"].values),
            sub["rate"].values,
            sub["weight"].values,
        )
        if beta[1] >= 0:
            return beta
        offset = borrowed * np.log(np.maximum(sub["price"].values, 0.1))
        X2 = np.column_stack([np.ones(len(sub)), np.minimum(sub["rank"].values, RANK_CAP)])
        b2 = _frac_logit_offset(X2, sub["rate"].values, sub["weight"].values, offset)
        return np.array([b2[0], borrowed, b2[1]])

    def _cv_logloss(self, sub: pd.DataFrame, borrowed: float) -> float:
        """Out-of-fold log-loss with clubs held out, so a club's squad-building
        habits cannot leak between train and test."""
        fold = np.array(self._team_folds(sub["team"].tolist()))
        total = 0.0
        weight = 0.0
        for f in range(5):
            train = sub[fold != f]
            test = sub[fold == f]
            if len(test) == 0 or len(train) < 10:
                continue
            beta = self._fit_prior_beta(train, borrowed)
            z = np.clip(self._prior_design(test["price"].values, test["rank"].values) @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            total += _weighted_logloss(p, test["rate"].values, test["weight"].values) * test["weight"].sum()
            weight += test["weight"].sum()
        return float(total / weight) if weight > 0 else float("nan")

    @staticmethod
    def _decile_table(sub: pd.DataFrame) -> List[Dict[str, float]]:
        """P(start) by price decile — the table the spec asks for, and a direct
        check that the parametric prior tracks the raw data."""
        try:
            bins = pd.qcut(sub["price"], 10, duplicates="drop")
        except ValueError:
            bins = pd.cut(sub["price"], 1)
        rows: List[Dict[str, float]] = []
        for i, (interval, grp) in enumerate(sub.groupby(bins, observed=True)):
            rows.append(
                {
                    "decile": i,
                    "price_lo": float(grp["price"].min()),
                    "price_hi": float(grp["price"].max()),
                    "players": int(len(grp)),
                    "opportunities": int(grp["n"].sum()),
                    "starts": int(grp["starts"].sum()),
                    "p_start": float(grp["starts"].sum() / max(grp["n"].sum(), 1)),
                }
            )
        return rows

    @staticmethod
    def _rank_table(sub: pd.DataFrame) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        bucket = np.minimum(np.floor(sub["rank"].values), RANK_CAP).astype(int)
        frame = sub.assign(_b=bucket)
        for b, grp in frame.groupby("_b"):
            rows.append(
                {
                    "rank": int(b),
                    "players": int(len(grp)),
                    "p_start": float(grp["starts"].sum() / max(grp["n"].sum(), 1)),
                }
            )
        return rows

    @staticmethod
    def _prior_from_coefs(fit_params: MinutesFit, pos: str, price: float, rank: float) -> float:
        coefs = fit_params.price_prior.get(pos)
        if not coefs:
            return 0.2
        z = (
            coefs["a"]
            + coefs["b_logprice"] * math.log(max(price, 0.1))
            + coefs["b_rank"] * min(rank, RANK_CAP)
        )
        return min(max(_expit(z), fit_params.prior_floor), fit_params.prior_cap)

    def price_prior(self, position: int, price: float, rank: float = 0.0) -> float:
        """P(start) implied by position, price and club-position price rank."""
        if self.fit_params is None:
            raise RuntimeError("call fit() before price_prior()")
        return self._prior_from_coefs(self.fit_params, scoring.POS_NAME[position], price, rank)

    # -- minutes shapes -----------------------------------------------------

    def _fit_minutes_shapes(
        self, fit_params: MinutesFit, ordered: pd.DataFrame, per_player: pd.DataFrame
    ) -> None:
        started = ordered[ordered["starts"] == 1]
        p60_pos: Dict[str, float] = {}
        mps_pos: Dict[str, float] = {}
        for et in scoring.POSITIONS:
            pos = scoring.POS_NAME[et]
            grp = started[started["element_type"] == et]
            if len(grp) < 20:
                continue
            p60_pos[pos] = float((grp["minutes"] >= 60).mean())
            mps_pos[pos] = float(grp["minutes"].mean())
        fit_params.p60_given_start_pos = p60_pos
        fit_params.mins_per_start_pos = mps_pos

        # Personalisation: a player's average minutes per start maps smoothly
        # onto his P(60+). This is what lets a striker who is always hooked on
        # 70 minutes score differently from one who plays out every game, and
        # it works from `history_past` totals where per-match rows do not exist.
        marks = pd.DataFrame(
            {
                "element": ordered["element"].values,
                "started": (ordered["starts"] == 1).astype(float).values,
                "s60": ((ordered["starts"] == 1) & (ordered["minutes"] >= 60)).astype(float).values,
                "mins_started": np.where(ordered["starts"] == 1, ordered["minutes"], 0.0),
            }
        )
        prof = marks.groupby("element").sum()
        prof = prof.rename(columns={"started": "starts"})
        prof = prof[prof["starts"] >= 5]
        if len(prof) >= 40:
            mps = (prof["mins_started"] / prof["starts"]).values
            rate = (prof["s60"] / prof["starts"]).values
            beta = _frac_logit(
                np.column_stack([np.ones(len(mps)), mps]), rate, prof["starts"].values
            )
            fit_params.p60_from_mins_per_start = [float(beta[0]), float(beta[1])]

        not_started = ordered[ordered["starts"] == 0]
        came_on = not_started[not_started["minutes"] > 0]
        fit_params.p60_given_sub = float((came_on["minutes"] >= 60).mean()) if len(came_on) else 0.0

        # Substitutes used per club per match, counted as (appearances that were
        # not starts) / (club-matches). Club-matches are counted from the starts
        # themselves: exactly 11 per club per match.
        team_matches = float(started["starts"].sum()) / float(scoring.SQUAD_PLAY)
        if team_matches > 0:
            fit_params.subs_per_team_match = float(len(came_on) / team_matches)
            fit_params.starters_per_team_match = {
                scoring.POS_NAME[et]: float(
                    (started["element_type"] == et).sum() / team_matches
                )
                for et in scoring.POSITIONS
            }
        fit_params.minutes_per_team_match = float(scoring.SQUAD_PLAY * 90)

        # Substitute appearances as a function of the player's own start rate:
        # a fringe player who never starts also rarely comes on, while a
        # rotation-level player is the first name off the bench.
        rates = per_player["rate"]
        joined = not_started.merge(
            rates.rename("srate"), left_on="element", right_index=True, how="inner"
        )
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001]
        appear_knots: List[List[float]] = []
        minute_knots: List[List[float]] = []
        buckets = pd.cut(joined["srate"], edges, include_lowest=True)
        for _, grp in joined.groupby(buckets, observed=True):
            if len(grp) < 30:
                continue
            on = grp[grp["minutes"] > 0]
            appear_knots.append([float(grp["srate"].mean()), float((grp["minutes"] > 0).mean())])
            minute_knots.append(
                [float(grp["srate"].mean()), float(on["minutes"].mean()) if len(on) else 0.0]
            )
        fit_params.sub_appear_curve = appear_knots
        fit_params.sub_minutes_curve = minute_knots

    # -- recency ------------------------------------------------------------

    def _fit_recency(
        self,
        fit_params: MinutesFit,
        ordered: pd.DataFrame,
        per_player: pd.DataFrame,
        prev: Optional[pd.DataFrame],
    ) -> None:
        """Fit the in-season half-life and the prior-season decay together.

        Each match of last season is predicted from exactly the posterior the
        model computes — decayed current-season starts, the season-before rate
        at its fitted reliability, and the price prior — and the pair
        (half_life, prior_decay) minimising log-loss wins. Fitting them jointly
        matters: they trade off directly, and separately-tuned values leave a
        player who has started the last six ranked below a benched premium.
        """
        K = max(float(self.config.model.shrinkage_90s_minutes), 0.25)
        elements = [int(e) for e in per_player.index]
        prior_p = np.array(
            [
                self._prior_from_coefs(
                    fit_params, str(row["pos"]), float(row["price"]), float(row["rank"])
                )
                for _, row in per_player.iterrows()
            ]
        )
        prev_rate = np.zeros(len(elements))
        prev_trials_full = np.zeros(len(elements))
        if prev is not None and len(prev):
            lookup = {int(r.element): (float(r.rate_prev), bool(r.moved)) for r in prev.itertuples()}
            for i, el in enumerate(elements):
                hit = lookup.get(el)
                if hit is None:
                    continue
                rel = fit_params.new_club_reliability if hit[1] else fit_params.prior_season_reliability
                rel = max(0.02, min(0.95, rel))
                prev_rate[i] = hit[0]
                prev_trials_full[i] = K * rel / (1.0 - rel)

        by_element = {int(el): grp["starts"].values.astype(float)
                      for el, grp in ordered.groupby("element")}
        lengths = [len(by_element.get(el, [])) for el in elements]
        max_len = max(lengths) if lengths else 0
        if max_len < 8:
            return
        starts = np.zeros((len(elements), max_len))
        mask = np.zeros((len(elements), max_len), dtype=bool)
        for i, el in enumerate(elements):
            arr = by_element.get(el)
            if arr is None:
                continue
            starts[i, : len(arr)] = arr
            mask[i, : len(arr)] = True

        # Predict from the second match onwards: the early-season regime, where
        # the prior season is doing most of the work, has to be part of the
        # objective or the decay is fitted only on the data that ignores it.
        warmup = 1
        have_prev = bool(prev_trials_full.any())
        decay_grid = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 16.0, 30.0, 1e6) if have_prev else (1e6,)
        steps = np.arange(max_len, dtype=float)
        best = (fit_params.recency_half_life_gws, fit_params.prior_decay_matches, float("inf"))
        for half_life in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 1e6):
            decay = 0.5 ** (1.0 / half_life)
            ssum = np.zeros((len(elements), max_len))
            wsum = np.zeros((len(elements), max_len))
            s = np.zeros(len(elements))
            w = np.zeros(len(elements))
            for t in range(max_len):
                ssum[:, t] = s
                wsum[:, t] = w
                s = s * decay + starts[:, t] * mask[:, t]
                w = w * decay + mask[:, t].astype(float)
            for prior_decay in decay_grid:
                shrinkfac = 0.5 ** (steps / prior_decay)
                pt = prev_trials_full[:, None] * shrinkfac[None, :]
                num = ssum + pt * prev_rate[:, None] + K * prior_p[:, None]
                den = wsum + pt + K
                p = np.clip(num / den, 1e-4, 1 - 1e-4)
                use = mask & (steps[None, :] >= warmup)
                if not use.any():
                    continue
                y = starts[use]
                q = p[use]
                ll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
                if ll < best[2]:
                    best = (half_life, prior_decay, ll)
        fit_params.recency_half_life_gws = float(best[0])
        if have_prev:
            fit_params.prior_decay_matches = float(best[1])
        fit_params.recency_fit_logloss = float(best[2])

        # Half-season split: how much a 19-match sample tells you about the next
        # 19. Reported so the shrinkage strength can be sanity-checked.
        gw = ordered["GW"]
        first = ordered[gw <= 19].groupby("element")["starts"].agg(["sum", "size"])
        second = ordered[gw > 19].groupby("element")["starts"].agg(["sum", "size"])
        both = first.join(second, how="inner", lsuffix="_1", rsuffix="_2")
        both = both[(both["size_1"] >= 10) & (both["size_2"] >= 10)]
        if len(both) > 30:
            r1 = (both["sum_1"] / both["size_1"]).values
            r2 = (both["sum_2"] / both["size_2"]).values
            slope = float(np.polyfit(r1, r2, 1)[0])
            fit_params.within_season_reliability = max(0.05, min(0.98, slope))

    # -- availability dynamics ---------------------------------------------

    def _fit_availability_dynamics(
        self, fit_params: MinutesFit, ordered: pd.DataFrame, per_player: pd.DataFrame
    ) -> None:
        """Measure how absence and return actually behave, on regular starters.

        Everything here is expressed relative to the cohort's own baseline start
        rate, so it can be applied as a multiplier to any player's role.
        """
        regular = per_player[(per_player["rate"] >= 0.5) & (per_player["n"] >= 12)]
        if len(regular) < 20:
            return
        ids = set(int(i) for i in regular.index)
        baseline = float(regular["starts"].sum() / regular["n"].sum())
        fit_params.cohort_baseline_start_rate = baseline

        seqs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for el, grp in ordered.groupby("element"):
            if int(el) in ids:
                seqs[int(el)] = (
                    grp["starts"].values.astype(int),
                    grp["minutes"].values.astype(int),
                )

        horizon = 6
        out_num = np.zeros(horizon + 1)
        out_den = np.zeros(horizon + 1)
        fit_num = np.zeros(horizon + 1)
        fit_den = np.zeros(horizon + 1)
        for s, m in seqs.values():
            for t in range(len(s)):
                two_out = t >= 1 and m[t] == 0 and m[t - 1] == 0
                started = s[t] == 1
                for h in range(1, horizon + 1):
                    if t + h >= len(s):
                        break
                    if two_out:
                        out_num[h] += s[t + h]
                        out_den[h] += 1
                    if started:
                        fit_num[h] += s[t + h]
                        fit_den[h] += 1
        if baseline > 0:
            fit_params.absence_recovery = [
                float(min(1.0, (out_num[h] / out_den[h]) / baseline)) if out_den[h] > 20 else 1.0
                for h in range(1, horizon + 1)
            ]
            fit_params.fitness_persistence = [
                float((fit_num[h] / fit_den[h]) / baseline) if fit_den[h] > 20 else 1.0
                for h in range(1, horizon + 1)
            ]

        # Return ramp: the first match back after a long absence is a genuine
        # coin-flip for a starter; by the second they are back in the XI.
        ramp_num = np.zeros(6)
        ramp_den = np.zeros(6)
        for s, m in seqs.values():
            t = 0
            while t < len(s):
                if m[t] == 0:
                    j = t
                    while j < len(s) and m[j] == 0:
                        j += 1
                    if (j - t) >= 3 and j < len(s):
                        for k in range(min(5, len(s) - j)):
                            ramp_num[k + 1] += s[j + k]
                            ramp_den[k + 1] += 1
                    t = j
                else:
                    t += 1
        if baseline > 0:
            fit_params.return_ramp = [
                float(min(1.0, (ramp_num[k] / ramp_den[k]) / baseline)) if ramp_den[k] > 15 else 1.0
                for k in range(1, 6)
            ]

    # -- cross-season reliability ------------------------------------------

    def _fit_cross_season(
        self, fit_params: MinutesFit, season: str, per_player: pd.DataFrame, cache: Cache
    ) -> Optional[pd.DataFrame]:
        """How much a completed season tells you about the next one.

        Two numbers, because they are wildly different: a player who stayed put
        carries most of his role forward, a player who changed club carries very
        little of it. Both size the effective number of trials the prior season
        contributes.

        The reliability is chosen to minimise the out-of-sample log-loss of the
        *actual* posterior the model computes, not the slope of a raw
        season-on-season regression. That matters: last season's start rate and
        the price prior are strongly correlated, so a marginal regression slope
        would be counted twice and every established starter would be pushed
        towards a 0.97 start probability.
        """
        fit_params.prior_season_reliability = 0.65
        fit_params.new_club_reliability = 0.24
        try:
            back = hist.prior_season(season)
            older = hist.load_vaastav_players_raw(back, cache=cache)
            current = hist.load_vaastav_players_raw(season, cache=cache)
        except Exception as exc:  # one season of data is enough to run
            log.warning("cross-season reliability falls back to defaults (%s: %s)",
                        type(exc).__name__, exc)
            return None
        if "code" not in older.columns or "code" not in current.columns:
            return None
        old = older[["code", "starts"]].copy()
        old["rate_prev"] = old["starts"] / float(scoring.TOTAL_GWS)
        cur = current[["id", "code", "team_join_date"]].copy()
        joined = per_player.join(cur.set_index("id")[["code", "team_join_date"]], how="inner")
        joined["element"] = joined.index
        merged = joined.merge(old[["code", "rate_prev"]], on="code", how="inner")
        cutoff_all = "%d-06-01" % hist.season_start_year(season)
        merged["moved"] = merged["team_join_date"].fillna("") >= cutoff_all
        every = merged[["element", "rate_prev", "moved"]].copy()
        merged = merged[merged["n"] >= 30]
        if len(merged) < 60:
            return every
        merged["prior_p"] = [
            self._prior_from_coefs(fit_params, str(row["pos"]), float(row["price"]), float(row["rank"]))
            for _, row in merged.iterrows()
        ]
        K = max(float(self.config.model.shrinkage_90s_minutes), 0.25)
        moved = merged["moved"]
        for mask, attr in ((~moved, "prior_season_reliability"), (moved, "new_club_reliability")):
            grp = merged[mask]
            if len(grp) < 25:
                continue
            best_rel, best_ll = None, float("inf")
            for rel in np.arange(0.05, 0.92, 0.025):
                trials = K * float(rel) / (1.0 - float(rel))
                preds = np.array(
                    [
                        stats.beta_binomial_rate(trials * float(rp), trials, float(pp), K)
                        for rp, pp in zip(grp["rate_prev"].values, grp["prior_p"].values)
                    ]
                )
                score = _weighted_logloss(preds, grp["rate"].values, grp["weight"].values)
                if score < best_ll:
                    best_ll, best_rel = score, float(rel)
            if best_rel is not None:
                setattr(fit_params, attr, best_rel)
        return every

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------

    def _prepare_state(self, state: Any) -> None:
        if self.fit_params is None:
            raise RuntimeError("call fit() before preparing state")
        self._state_current_gw = int(getattr(state, "current_gw", 1))
        players: Dict[int, Player] = state.players

        ids = sorted(players.keys())
        self._ranks = dict(
            zip(
                ids,
                club_position_price_ranks(
                    [players[i].now_cost / 10.0 for i in ids],
                    [players[i].team_id for i in ids],
                    [players[i].position for i in ids],
                ),
            )
        )

        self._news = {
            pid: parse_news(
                players[pid].news,
                players[pid].status,
                players[pid].chance_of_playing_next_round,
                self._now,
            )
            for pid in ids
        }

        self._gw_dates = self._build_gw_dates(state)
        self._europe_team_ids = self._detect_european_teams(state)

        prior_profiles = self._prior_season_profiles(state)
        self._roles = {
            pid: self._build_role(state, players[pid], prior_profiles.get(players[pid].code))
            for pid in ids
        }
        self._team_members = {}
        for pid in ids:
            self._team_members.setdefault(players[pid].team_id, []).append(pid)
        self._team_scale = {}

    @staticmethod
    def _build_gw_dates(state: Any) -> Dict[int, datetime]:
        """A representative UTC datetime per gameweek: the deadline, which every
        event carries, rather than a kickoff which a blank gameweek lacks."""
        out: Dict[int, datetime] = {}
        for event in getattr(state, "events", []) or []:
            raw = event.get("deadline_time")
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            out[int(event["id"])] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return out

    def _detect_european_teams(self, state: Any) -> List[int]:
        """Prior-season top six as a proxy for European commitments.

        Only consulted when `model.rotation_risk_penalty` is non-zero, so the
        default configuration never pays for this.
        """
        if float(self.config.model.rotation_risk_penalty) <= 0:
            return []
        try:
            results = hist.load_football_data(
                [getattr(state, "prior_season", None) or hist.prior_season(self.config.season)],
                cache=Cache(default_ttl=self.config.cache_ttl_seconds),
            )
        except Exception as exc:
            log.warning("no European-competition proxy available (%s)", exc)
            return []
        points: Dict[str, int] = {}
        for _, row in results.iterrows():
            h, a = row["home_short"], row["away_short"]
            if not isinstance(h, str) or not isinstance(a, str):
                continue
            hg, ag = row["home_goals"], row["away_goals"]
            if pd.isna(hg) or pd.isna(ag):
                continue
            points.setdefault(h, 0)
            points.setdefault(a, 0)
            if hg > ag:
                points[h] += 3
            elif hg < ag:
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1
        top = [t for t, _ in sorted(points.items(), key=lambda kv: -kv[1])[:6]]
        return [t.id for t in state.teams.values() if t.short_name in top]

    def _prior_season_profiles(self, state: Any) -> Dict[int, Dict[str, float]]:
        """Exact per-match prior-season profiles keyed by the stable player code.

        `history_past` gives totals only, so P(60|start) and the substitute rate
        have to be inferred from them. The vaastav rows give the real counts, and
        FPL player codes survive the summer id reshuffle, so they join cleanly.
        """
        season = getattr(state, "prior_season", "") or hist.prior_season(self.config.season)
        try:
            cache = Cache(default_ttl=self.config.cache_ttl_seconds)
            gws = hist.load_vaastav_gws(season, cache=cache)
            raw = hist.load_vaastav_players_raw(season, cache=cache)
        except Exception as exc:
            log.warning("prior-season per-match profiles unavailable (%s: %s); "
                        "falling back to history_past totals", type(exc).__name__, exc)
            return {}
        if "code" not in raw.columns:
            return {}
        code_of = dict(zip(raw["id"].astype(int), raw["code"].astype(int)))
        out: Dict[int, Dict[str, float]] = {}
        for element, grp in gws.groupby("element"):
            code = code_of.get(int(element))
            if code is None:
                continue
            mins = grp["minutes"].values.astype(float)
            st = grp["starts"].values.astype(float)
            n_start = float(st.sum())
            started = st == 1
            subs = (st == 0) & (mins > 0)
            out[int(code)] = {
                "opportunities": float(len(grp)),
                "starts": n_start,
                "starts_60": float(((mins >= 60) & started).sum()),
                "mins_started": float(mins[started].sum()),
                "non_starts": float((st == 0).sum()),
                "sub_appearances": float(subs.sum()),
                "sub_minutes": float(mins[subs].sum()),
            }
        return out

    # -- role construction --------------------------------------------------

    def _build_role(
        self, state: Any, player: Player, profile: Optional[Dict[str, float]]
    ) -> _Role:
        fit_params = self.fit_params
        assert fit_params is not None
        pos = scoring.POS_NAME[player.position]
        rank = self._ranks.get(player.id, 0.0)
        K = max(float(self.config.model.shrinkage_90s_minutes), 0.25)
        p_prior = self._prior_from_coefs(fit_params, pos, player.now_cost / 10.0, rank)

        role = _Role(player_id=player.id, price_prior=p_prior, rank=rank)

        # --- current season (recency weighted) ---
        cur_succ = cur_trials = 0.0
        cur_starts = cur_60 = cur_mins_started = 0.0
        cur_non_starts = cur_subs = cur_sub_mins = 0.0
        n_gws_seen = 0
        history = state.history(player.id)
        if history is not None and len(history) and "starts" in history.columns:
            frame = history.sort_values("round" if "round" in history.columns else "gw")
            starts = pd.to_numeric(frame["starts"], errors="coerce").fillna(0.0).values
            minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0).values
            n_gws_seen = len(starts)
            decay = 0.5 ** (1.0 / max(fit_params.recency_half_life_gws, 0.5))
            wsum = ssum = 0.0
            for value in starts:
                wsum = wsum * decay + 1.0
                ssum = ssum * decay + float(value)
            cur_succ, cur_trials = ssum, wsum
            started_mask = starts == 1
            cur_starts = float(started_mask.sum())
            cur_60 = float(((minutes >= 60) & started_mask).sum())
            cur_mins_started = float(minutes[started_mask].sum())
            cur_non_starts = float((~started_mask).sum())
            cur_subs = float(((minutes > 0) & (~started_mask)).sum())
            cur_sub_mins = float(minutes[(minutes > 0) & (~started_mask)].sum())

        # --- prior season ---
        moved = self._joined_recently(player)
        reliability = (
            fit_params.new_club_reliability if moved else fit_params.prior_season_reliability
        )
        reliability = max(0.02, min(0.95, reliability))
        # Effective trials chosen so that, with no other evidence, the posterior
        # equals `reliability * observed + (1 - reliability) * prior` — i.e. it
        # reproduces the measured season-to-season regression exactly — then
        # decayed at the fitted rate as this season's matches accumulate.
        trials_per_season = K * reliability / (1.0 - reliability)
        trials_per_season *= 0.5 ** (n_gws_seen / max(fit_params.prior_decay_matches, 0.5))

        prior_succ = prior_trials = 0.0
        prior_starts = prior_60 = prior_mins_started = 0.0
        prior_non_starts = prior_subs = prior_sub_mins = 0.0
        prior_label = ""
        prior_rate = None
        if profile and profile["opportunities"] > 0:
            share = profile["opportunities"] / float(scoring.TOTAL_GWS)
            prior_rate = profile["starts"] / profile["opportunities"]
            prior_trials = trials_per_season * min(share, 1.0)
            prior_succ = prior_trials * prior_rate
            prior_starts = profile["starts"]
            prior_60 = profile["starts_60"]
            prior_mins_started = profile["mins_started"]
            prior_non_starts = profile["non_starts"]
            prior_subs = profile["sub_appearances"]
            prior_sub_mins = profile["sub_minutes"]
            prior_label = "%d/%d starts in %s" % (
                int(profile["starts"]),
                int(profile["opportunities"]),
                hist.season_slash(getattr(state, "prior_season", "") or "1900-01"),
            )
        else:
            row = state.prior_season_row(player.id) or state.latest_season_row(player.id)
            if row and float(row.get("minutes") or 0) > 0:
                # history_past has no opportunity count, so the denominator is
                # the full season. A January arrival is understated by this and
                # the shrinkage below is what keeps that honest.
                starts_prior = float(row.get("starts") or 0)
                prior_rate = min(1.0, starts_prior / float(scoring.TOTAL_GWS))
                prior_trials = trials_per_season
                prior_succ = prior_trials * prior_rate
                prior_starts = starts_prior
                prior_mins_started = float(row.get("minutes") or 0.0)
                prior_label = "%d starts in %s" % (
                    int(starts_prior), row.get("season_name", "the prior season"),
                )

        succ = cur_succ + prior_succ
        trials = cur_trials + prior_trials
        role.p_start = float(stats.beta_binomial_rate(succ, trials, p_prior, K))
        role.n_trials = trials

        # --- P(60 | start) ---
        role.p60_given_start = self._p60_given_start(
            fit_params, pos, cur_starts, cur_60, cur_mins_started,
            prior_starts * reliability, prior_60 * reliability,
            prior_mins_started * reliability, K,
        )
        role.mins_per_start = self._mins_per_start(
            fit_params, pos, cur_starts, cur_mins_started,
            prior_starts * reliability, prior_mins_started * reliability, K,
        )

        # --- substitute behaviour ---
        curve_sub = _interp(fit_params.sub_appear_curve, role.p_start)
        sub_succ = cur_subs + prior_subs * reliability
        sub_trials = cur_non_starts + prior_non_starts * reliability
        role.p_sub_given_no_start = float(
            stats.beta_binomial_rate(sub_succ, sub_trials, curve_sub, K)
        )
        curve_mins = _interp(fit_params.sub_minutes_curve, role.p_start)
        observed_sub_mins = (cur_sub_mins + prior_sub_mins * reliability)
        observed_sub_apps = (cur_subs + prior_subs * reliability)
        role.mins_per_sub = float(
            stats.shrink(
                observed_sub_mins / observed_sub_apps if observed_sub_apps > 0 else curve_mins,
                observed_sub_apps,
                curve_mins if curve_mins > 0 else float(self.config.model.sub_minutes),
                K,
            )
        )

        # --- rotation risk (only when the user asks for it) ---
        penalty = float(self.config.model.rotation_risk_penalty)
        if penalty > 0 and player.team_id in self._europe_team_ids:
            role.p_start *= max(0.0, 1.0 - penalty)

        # --- provenance ---
        if n_gws_seen > 0:
            role.source = "current-season"
            recent = int(min(n_gws_seen, 6))
            recent_starts = int(
                pd.to_numeric(state.history(player.id)["starts"], errors="coerce")
                .fillna(0)
                .values[-recent:]
                .sum()
            )
            role.evidence = "%d of last %d starts" % (recent_starts, recent)
            if prior_label:
                role.evidence += ", %s" % prior_label
        elif prior_label:
            role.source = "prior-season"
            role.evidence = prior_label
            if moved:
                role.evidence += " (new club, discounted)"
        else:
            role.source = "price-prior"
            role.evidence = "no PL history: %s prior for £%.1fm, %s at %s" % (
                pos,
                player.now_cost / 10.0,
                _ordinal(int(rank) + 1) + " priciest " + pos,
                state.short_name(player.team_id),
            )
        return role

    def _joined_recently(self, player: Player) -> bool:
        """Did this player change club in the window before the current season?"""
        if not player.team_join_date:
            return False
        year = hist.season_start_year(self.config.season)
        return str(player.team_join_date) >= "%d-06-01" % year

    @staticmethod
    def _p60_given_start(
        fit_params: MinutesFit,
        pos: str,
        cur_starts: float,
        cur_60: float,
        cur_mins: float,
        prior_starts: float,
        prior_60: float,
        prior_mins: float,
        K: float,
    ) -> float:
        base = fit_params.p60_given_start_pos.get(pos, 0.9)
        succ = cur_60 + prior_60
        trials = cur_starts + prior_starts
        if trials <= 0:
            return base
        if succ <= 0 and (cur_mins + prior_mins) > 0 and fit_params.p60_from_mins_per_start:
            # Totals-only evidence: recover P(60|start) from minutes per start.
            a, b = fit_params.p60_from_mins_per_start
            mps = (cur_mins + prior_mins) / trials
            base = _expit(a + b * mps)
            return float(min(max(base, 0.02), 0.999))
        return float(stats.beta_binomial_rate(succ, trials, base, K))

    @staticmethod
    def _mins_per_start(
        fit_params: MinutesFit,
        pos: str,
        cur_starts: float,
        cur_mins: float,
        prior_starts: float,
        prior_mins: float,
        K: float,
    ) -> float:
        base = fit_params.mins_per_start_pos.get(pos, 82.0)
        trials = cur_starts + prior_starts
        if trials <= 0:
            return base
        observed = (cur_mins + prior_mins) / trials
        return float(min(90.0, max(20.0, stats.shrink(observed, trials, base, K))))

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _availability(
        self, player: Player, gw: int
    ) -> Tuple[float, str]:
        """P(the player is available to be picked for this gameweek), plus a note.

        Everything the status code says is respected first; the fitted recovery
        and return curves supply the *shape* over a multi-gameweek horizon,
        which is the part a single status flag cannot express.
        """
        fit_params = self.fit_params
        assert fit_params is not None
        signal = self._news.get(player.id, NewsSignal())
        status = player.status or "a"
        horizon = max(0, int(gw) - self._state_current_gw)
        gw_date = self._gw_dates.get(int(gw))

        if status == "a":
            return 1.0, ""

        if status == DOUBTFUL_STATUS:
            chance = signal.chance
            if chance is None:
                chance = float(self.config.model.status_multiplier.get("d", 0.5))
            # A doubt is about *this* round. Further out, the doubt resolves at
            # the rate injured players actually return.
            if horizon == 0:
                return max(0.0, min(1.0, chance)), "%.0f%% chance" % (chance * 100)
            recovered = self._recovery_fraction(fit_params, horizon)
            value = chance + (1.0 - chance) * recovered
            return max(0.0, min(1.0, value)), "%.0f%% chance now" % (chance * 100)

        if status not in HARD_OUT_STATUSES:
            return float(self.config.model.status_multiplier.get(status, 1.0)), ""

        if signal.departed:
            return 0.0, "left the club"

        # A stated return date is the strongest signal there is.
        if signal.return_date is not None and gw_date is not None:
            if gw_date < signal.return_date:
                return 0.0, "out until %s" % signal.return_date.strftime("%d %b")
            matches_back = self._matches_since(signal.return_date, gw)
            if signal.suspended:
                return 1.0, "suspension ends %s" % signal.return_date.strftime("%d %b")
            ramp = fit_params.return_ramp
            idx = min(max(matches_back, 1), len(ramp)) - 1 if ramp else 0
            factor = float(ramp[idx]) if ramp else 1.0
            return max(0.0, min(1.0, factor)), "back %s" % signal.return_date.strftime("%d %b")

        # Flagged out with no date. `chance_of_playing_next_round == 0` is the
        # API stating the imminent round is gone; later rounds follow the fitted
        # recovery curve for a player who has already missed two matches.
        if horizon == 0:
            return 0.0, "out"
        return self._recovery_fraction(fit_params, horizon), "out, no return date"

    @staticmethod
    def _recovery_fraction(fit_params: MinutesFit, horizon: int) -> float:
        curve = fit_params.absence_recovery
        if not curve:
            return 0.0
        idx = min(max(horizon, 1), len(curve)) - 1
        return float(max(0.0, min(1.0, curve[idx])))

    def _matches_since(self, return_date: datetime, gw: int) -> int:
        """How many gameweeks have started on or after the stated return date."""
        count = 0
        for g in sorted(self._gw_dates):
            if g > int(gw):
                break
            if self._gw_dates[g] >= return_date:
                count += 1
        return max(count, 1)

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    @staticmethod
    def _odds_scale(probs: Sequence[float], target: float) -> float:
        """Common odds multiplier making the probabilities sum to `target`.

        Rescaling odds rather than probabilities keeps everything inside [0, 1]
        and preserves the ordering, which naive proportional scaling does not.
        """
        live = [p for p in probs if p > 1e-9]
        if not live:
            return 1.0
        total = sum(live)
        if abs(total - target) < 1e-6:
            return 1.0
        odds = [min(p, 0.999) / (1.0 - min(p, 0.999)) for p in live]
        lo, hi = 1e-4, 1e4
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            s = sum(mid * o / (1.0 + mid * o) for o in odds)
            if s < target:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def _team_adjustment(self, state: Any, team_id: int, gw: int) -> Tuple[float, float, float]:
        """(GK odds multiplier, outfield odds multiplier, substitute multiplier).

        Goalkeepers are normalised separately because exactly one starts — a
        single club-wide scaling leaves the keeper mass at ~1.15 per club, which
        would hand every backup keeper phantom minutes.
        """
        key = (int(team_id), int(gw))
        cached = self._team_scale.get(key)
        if cached is not None:
            return cached
        if not self.normalize_to_eleven:
            result = (1.0, 1.0, 1.0)
            self._team_scale[key] = result
            return result
        fit_params = self.fit_params
        assert fit_params is not None
        members = self._team_members.get(int(team_id), [])
        keepers, outfield = [], []
        for pid in members:
            avail, _ = self._availability(state.players[pid], gw)
            p = _clamp01(self._roles[pid].p_start * avail)
            (keepers if state.players[pid].position == scoring.GKP else outfield).append(p)
        lam_gk = self._odds_scale(keepers, 1.0)
        lam_out = self._odds_scale(outfield, float(scoring.SQUAD_PLAY - 1))
        raw_sub = 0.0
        for pid in members:
            avail, _ = self._availability(state.players[pid], gw)
            if avail <= 0.0:
                continue  # an unavailable player cannot be one of the four subs
            lam = lam_gk if state.players[pid].position == scoring.GKP else lam_out
            p = _start_after(_clamp01(self._roles[pid].p_start * avail), lam)
            raw_sub += (1.0 - p) * self._roles[pid].p_sub_given_no_start
        target_sub = max(fit_params.subs_per_team_match, 0.0)
        sub_scale = 1.0
        if raw_sub > 1e-6 and target_sub > 0:
            sub_scale = min(max(target_sub / raw_sub, 0.2), 5.0)
        result = (lam_gk, lam_out, sub_scale)
        self._team_scale[key] = result
        return result

    def forecast(self, player_id: int, gw: int, state: Any) -> MinutesForecast:
        if self.fit_params is None:
            raise RuntimeError("call MinutesModel.fit(state) before forecast()")
        if player_id not in self._roles:
            self._prepare_state(state)
        player = state.players[player_id]
        role = self._roles[player_id]
        fit_params = self.fit_params

        avail, note = self._availability(player, gw)
        lam_gk, lam_out, sub_scale = self._team_adjustment(state, player.team_id, gw)
        lam = lam_gk if player.position == scoring.GKP else lam_out

        p_start = _start_after(_clamp01(role.p_start * avail), lam)
        p_sub = _clamp01(role.p_sub_given_no_start * sub_scale) if avail > 0 else 0.0
        p_appear = p_start + (1.0 - p_start) * p_sub
        p_60 = (
            p_start * role.p60_given_start
            + (1.0 - p_start) * p_sub * fit_params.p60_given_sub
        )
        xmins = (
            p_start * role.mins_per_start
            + (1.0 - p_start) * p_sub * role.mins_per_sub
        )

        p_start = _clamp01(p_start)
        p_appear = _clamp01(max(p_appear, p_start))
        p_60 = _clamp01(min(p_60, p_appear))
        xmins = float(min(90.0, max(0.0, xmins)))
        if p_appear <= 0.0:
            p_start = p_60 = 0.0
            xmins = 0.0

        return MinutesForecast(
            player_id=player_id,
            gw=int(gw),
            p_appear=round(p_appear, 4),
            p_start=round(p_start, 4),
            p_60=round(p_60, 4),
            xmins=round(xmins, 2),
            reason=self._reason(player, role, avail, note, p_start),
        )

    def forecast_all(
        self, state: Any, gws: Sequence[int]
    ) -> Dict[Tuple[int, int], MinutesForecast]:
        if self.fit_params is None:
            self.fit(state)
        out: Dict[Tuple[int, int], MinutesForecast] = {}
        for pid in sorted(state.players):
            for gw in gws:
                out[(pid, int(gw))] = self.forecast(pid, int(gw), state)
        return out

    # -- explanation --------------------------------------------------------

    def _reason(
        self, player: Player, role: _Role, avail: float, note: str, p_start: float
    ) -> str:
        status = player.status or "a"
        if avail <= 0.0:
            problem = self._news.get(player.id, NewsSignal()).label()
            return "unavailable: %s" % (note or problem or _STATUS_WORD.get(status, status))
        if status == DOUBTFUL_STATUS:
            return "doubtful: %s%s, %s" % (
                note or "flagged",
                (" (%s)" % self._news[player.id].problem) if self._news[player.id].problem else "",
                role.evidence,
            )
        if status in HARD_OUT_STATUSES:
            return "returning: %s, %s" % (note or "flagged", role.evidence)
        if role.source == "price-prior":
            return "%s: %s" % (_role_word(p_start), role.evidence)
        return "%s: %s, no news" % (_role_word(p_start), role.evidence)


_STATUS_WORD = {
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
    "d": "doubtful",
}


def _role_word(p_start: float) -> str:
    if p_start >= 0.80:
        return "nailed"
    if p_start >= 0.60:
        return "likely starter"
    if p_start >= 0.35:
        return "rotation"
    if p_start >= 0.12:
        return "fringe"
    return "bench"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return "%d%s" % (n, suffix)


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _start_after(p: float, lam: float) -> float:
    """Apply a common odds multiplier to a start probability."""
    if p <= 0.0:
        return 0.0
    if lam == 1.0:
        return min(p, 1.0)
    p = min(p, 0.999)
    odds = lam * p / (1.0 - p)
    return float(min(0.995, odds / (1.0 + odds)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time

    from gaffer.data.loaders import load_game_state

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()

    t0 = time.time()
    game = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (game.summary(), time.time() - t0))

    model = MinutesModel(cfg)
    t0 = time.time()
    params = model.fit(game, force=True)
    print("fitted in %.2fs from %s (%d players, %d player-matches)"
          % (time.time() - t0, params.fitted_from_season, params.n_players, params.n_player_matches))

    print("\n=== fitted price prior: logit P(start) = a + b_price*log(price) + b_rank*rank ===")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        coefs = params.price_prior.get(pos)
        if not coefs:
            continue
        print("  %s  a=%+7.3f  b_logprice=%+6.3f  b_rank=%+6.4f   (n=%d players, CV logloss %.4f)"
              % (pos, coefs["a"], coefs["b_logprice"], coefs["b_rank"],
                 coefs["n_players"], params.price_prior_cv_logloss.get(pos, float("nan"))))
        header = "        price |" + "".join("  rank%d" % r for r in range(0, 7))
        print(header)
        for price in (4.0, 4.5, 5.0, 5.5, 6.5, 8.0, 11.0, 14.5):
            cells = "".join(
                "  %5.2f" % model.price_prior(scoring.POS_ID[pos], price, r) for r in range(0, 7)
            )
            print("        %5.1f |%s" % (price, cells))

    print("\n=== raw P(start) by position and price decile (2025/26 actuals) ===")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        rows = params.price_decile_table.get(pos, [])
        print("  %s: %s" % (pos, "  ".join(
            "£%.1f-%.1f:%.3f(n=%d)" % (r["price_lo"], r["price_hi"], r["p_start"], r["players"])
            for r in rows)))

    print("\n=== other fitted parameters ===")
    print("  P(60+|started) by position : %s"
          % ", ".join("%s=%.3f" % kv for kv in sorted(params.p60_given_start_pos.items())))
    print("  E[minutes|started]         : %s"
          % ", ".join("%s=%.1f" % kv for kv in sorted(params.mins_per_start_pos.items())))
    print("  logit P(60|start) from mins/start: a=%.4f b=%.4f" % tuple(params.p60_from_mins_per_start))
    print("  P(60+|substitute appearance): %.4f" % params.p60_given_sub)
    print("  prior-season reliability   : %.3f (stayers) / %.3f (changed club)"
          % (params.prior_season_reliability, params.new_club_reliability))
    print("  within-season reliability  : %.3f" % params.within_season_reliability)
    print("  recency half-life          : %.1f gameweeks (joint fit log-loss %.4f)"
          % (params.recency_half_life_gws, params.recency_fit_logloss))
    print("  prior-season decay         : half-life %.1f current-season matches"
          % params.prior_decay_matches)
    print("  cohort baseline start rate : %.3f" % params.cohort_baseline_start_rate)
    print("  absence recovery (h=1..6)  : %s"
          % ", ".join("%.3f" % v for v in params.absence_recovery))
    print("  return ramp (k=1..5)       : %s"
          % ", ".join("%.3f" % v for v in params.return_ramp))
    print("  fitness persistence h=1..6 : %s"
          % ", ".join("%.3f" % v for v in params.fitness_persistence))
    print("  sub-appearance curve       : %s"
          % ", ".join("%.2f->%.3f" % (a, b) for a, b in params.sub_appear_curve))
    print("  sub-minutes curve          : %s"
          % ", ".join("%.2f->%.1f" % (a, b) for a, b in params.sub_minutes_curve))
    print("  prior cap / floor          : %.3f / %.3f" % (params.prior_cap, params.prior_floor))

    gw = game.current_gw
    horizon = list(range(gw, min(gw + 6, scoring.TOTAL_GWS + 1)))
    t0 = time.time()
    forecasts = model.forecast_all(game, horizon)
    print("\nforecast_all over GW%d-%d for %d players: %d rows in %.2fs"
          % (horizon[0], horizon[-1], len(game.players), len(forecasts), time.time() - t0))

    this_gw = {pid: f for (pid, g), f in forecasts.items() if g == gw}

    print("\n=== 30 highest expected minutes, GW%d ===" % gw)
    print("  %-18s %-3s %-4s %5s %7s %6s %7s  %s"
          % ("player", "pos", "team", "price", "p_start", "p_60", "xmins", "reason"))
    top = sorted(this_gw.values(), key=lambda f: -f.xmins)[:30]
    for f in top:
        p = game.player(f.player_id)
        print("  %-18s %-3s %-4s %5.1f %7.3f %6.3f %7.1f  %s"
              % (p.web_name, game.position_name(p.id), game.short_name(p.team_id), p.price,
                 f.p_start, f.p_60, f.xmins, f.reason))

    print("\n=== every player with status != 'a' (GW%d and GW%d) ===" % (gw, horizon[-1]))
    flagged = [p for p in game.players.values() if (p.status or "a") != "a"]
    flagged.sort(key=lambda p: (p.status, -p.now_cost))
    print("  %-3s %-18s %-3s %-4s %5s | %-32s | %-32s"
          % ("st", "player", "pos", "team", "price", "GW%d" % gw, "GW%d" % horizon[-1]))
    for p in flagged:
        a = this_gw[p.id]
        b = forecasts[(p.id, horizon[-1])]
        print("  %-3s %-18s %-3s %-4s %5.1f | %-32s | %-32s"
              % (p.status, p.web_name, game.position_name(p.id), game.short_name(p.team_id),
                 p.price,
                 "ps=%.2f p60=%.2f xm=%4.1f" % (a.p_start, a.p_60, a.xmins),
                 "ps=%.2f p60=%.2f xm=%4.1f" % (b.p_start, b.p_60, b.xmins)))
        print("      GW%d: %s" % (gw, a.reason))

    print("\n=== players with no prior-season data at all (price prior only) ===")
    no_history = [
        p for p in game.players.values()
        if model._roles[p.id].source == "price-prior" and (p.status or "a") == "a"
    ]
    no_history.sort(key=lambda p: -this_gw[p.id].xmins)
    print("  %d such players; top 12 by xmins:" % len(no_history))
    for p in no_history[:12]:
        f = this_gw[p.id]
        print("  %-18s %-3s %-4s %5.1f  p_start=%.3f xmins=%5.1f  %s"
              % (p.web_name, game.position_name(p.id), game.short_name(p.team_id), p.price,
                 f.p_start, f.xmins, f.reason))

    print("\n=== assertions ===")
    bad = []
    for (pid, g), f in forecasts.items():
        vals = (f.p_appear, f.p_start, f.p_60, f.xmins)
        if any(v != v for v in vals):
            bad.append(("nan", pid, g, vals))
        elif not (0.0 <= f.p_60 <= f.p_appear <= 1.0):
            bad.append(("order", pid, g, vals))
        elif not (0.0 <= f.p_start <= 1.0) or not (0.0 <= f.xmins <= 90.0):
            bad.append(("range", pid, g, vals))
        elif not f.reason:
            bad.append(("reason", pid, g, vals))
    assert not bad, bad[:5]
    print("  0 <= p_60 <= p_appear <= 1, 0 <= p_start <= 1, 0 <= xmins <= 90, no NaN, "
          "every reason populated: OK across %d forecasts" % len(forecasts))

    for pos, value in params.p60_given_start_pos.items():
        assert 0.60 <= value <= 0.999, (pos, value)
    overall = sum(params.p60_given_start_pos.values()) / len(params.p60_given_start_pos)
    assert 0.75 <= overall <= 0.95, overall
    print("  fitted P(60+|started) per position %s, mean %.3f in [0.75, 0.95]: OK"
          % ({k: round(v, 3) for k, v in sorted(params.p60_given_start_pos.items())}, overall))

    assert 0.0 <= params.p60_given_sub <= 0.10, params.p60_given_sub
    print("  fitted P(60+|substitute) = %.4f (a sub almost never reaches 60): OK"
          % params.p60_given_sub)

    hard_out = [p for p in game.players.values() if (p.status or "a") in ("i", "s", "u")]
    assert all(this_gw[p.id].p_appear == 0.0 for p in hard_out), "hard-out player with p_appear>0"
    print("  all %d i/s/u players have p_appear = 0 for GW%d: OK" % (len(hard_out), gw))

    doubtful = [p for p in game.players.values() if (p.status or "a") == "d"]
    for p in doubtful:
        chance = (p.chance_of_playing_next_round or 50) / 100.0
        avail, _ = model._availability(p, gw)
        assert abs(avail - chance) < 1e-9, (p.web_name, avail, chance)
    print("  all %d doubtful players scaled by chance_of_playing_next_round: OK" % len(doubtful))

    returners = [
        p for p in game.players.values()
        if (p.status or "a") in ("i", "s") and model._news[p.id].return_date is not None
    ]
    recovered = [p for p in returners if forecasts[(p.id, horizon[-1])].p_appear > 0]
    print("  %d flagged players carry a parsed return date; %d are available again by GW%d"
          % (len(returners), len(recovered), horizon[-1]))

    print("\n=== news parser check ===")
    for text, status, cop in (
        ("Shin injury - 75% chance of playing", "d", 75),
        ("Achilles injury - Unknown return date", "i", 0),
        ("Groin injury - Expected back 22 Aug", "i", 0),
        ("Suspended until 6 Sep", "s", 0),
        ("Has joined Getafe permanently", "u", 0),
        ("Knock - 50% chance of playing", "d", 50),
        ("", "a", None),
    ):
        sig = parse_news(text, status, cop, datetime(2026, 8, 14, tzinfo=timezone.utc))
        print("  %-42r -> chance=%s return=%s indefinite=%s departed=%s problem=%r"
              % (text, sig.chance,
                 sig.return_date.strftime("%Y-%m-%d") if sig.return_date else None,
                 sig.indefinite, sig.departed, sig.problem))

    print("\n=== aggregate sanity ===")
    xmins_all = [f.xmins for f in this_gw.values()]
    print("  GW%d xmins: mean=%.1f median=%.1f  >=60: %d players  >=80: %d players"
          % (gw, float(np.mean(xmins_all)), float(np.median(xmins_all)),
             sum(1 for v in xmins_all if v >= 60), sum(1 for v in xmins_all if v >= 80)))
    for pos in (1, 2, 3, 4):
        ids = [p.id for p in game.players.values() if p.position == pos]
        expected = 20.0 * params.starters_per_team_match.get(scoring.POS_NAME[pos], 0.0)
        print("    %s: %3d players, sum p_start = %6.1f (2025/26 actual across 20 clubs: %.1f)"
              % (scoring.POS_NAME[pos], len(ids),
                 sum(this_gw[i].p_start for i in ids), expected))

    print("\n=== structural calibration: every club fields 11 and 990 minutes ===")
    worst = []
    for team in sorted(game.teams.values(), key=lambda t: t.short_name):
        ids = [p.id for p in game.players.values() if p.team_id == team.id]
        s = sum(this_gw[i].p_start for i in ids)
        m = sum(this_gw[i].xmins for i in ids)
        a = sum(this_gw[i].p_appear for i in ids)
        worst.append((abs(s - 11.0), team.short_name, s, a, m))
    worst.sort(reverse=True)
    print("  club   sum p_start   sum p_appear   sum xmins")
    for _, short, s, a, m in sorted(worst, key=lambda r: r[1]):
        print("  %-5s     %6.2f        %6.2f       %7.1f" % (short, s, a, m))
    max_dev = worst[0][0]
    total_mins = sum(f.xmins for f in this_gw.values())
    assert max_dev < 0.05, worst[0]
    print("  worst club deviation from 11 starters: %.4f" % max_dev)
    print("  league xmins total %.0f vs 20 x 990 = %d (%.1f%%)"
          % (total_mins, 20 * 990, 100.0 * total_mins / (20 * 990)))
    print("  fitted substitutes used per club per match: %.2f" % params.subs_per_team_match)

    print("\nreport written to %s" % PRIOR_REPORT_PATH)
