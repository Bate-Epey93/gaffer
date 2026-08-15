"""Attacking returns: goals and assists.

Three ideas carry this module.

1. **A player's rate is a shrunken blend.** Per-90 xG and xA are blended across
   the prior season, the current season to date and the last N gameweeks, then
   shrunk towards a position-and-price baseline fitted from last season. A £4.5m
   promoted-club forward with no Premier League history is not a 0.00 xG90
   player, he is an average £4.5m forward until he proves otherwise.

2. **Fixture scaling is an elasticity, not a bucket.** A player's expected goals
   move with his team's expected goals, but not one-for-one — see
   ``fit_historical`` for the fitted exponents and the comment above
   ``GOAL_ELASTICITY_PRIOR`` for what they mean.

3. **Penalties are a separate, explicit term.** Most public models either ignore
   penalties or double-count them, because Opta xG already contains 0.79 per
   penalty taken. We strip the historical penalty xG out of the base rate and
   add a forward-looking penalty term for the *current* first-choice taker. A
   player newly handed penalties gains the full term; a player who has lost them
   loses it. That asymmetry is where the edge is.

Everything empirical here is fitted from data and written to
``reports/attacking_fit.json`` with its sample size, so it can be audited.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import scoring, stats
from gaffer.core.config import REPORTS_DIR, Config
from gaffer.data import history as hist
from gaffer.data.cache import Cache

if TYPE_CHECKING:  # pragma: no cover - imported for types only
    from gaffer.data.loaders import GameState
    from gaffer.model.xp import FixtureContext

log = logging.getLogger(__name__)

FIT_PATH = os.path.join(REPORTS_DIR, "attacking_fit.json")
FIT_VERSION = 4

# Opta scores every penalty kick at exactly this xG, and FPL passes that through
# in `expected_goals`. Verified against the data rather than assumed: in the
# vaastav per-gameweek archive there are 52 player-gameweeks with xG of exactly
# 0.79 against 12-26 at each neighbouring hundredth, and rows such as
# B.Fernandes 2025-26 GW6 (penalties_missed=1, xG=0.79) pin the value to a lone
# penalty. `fit_historical` re-checks the spike and warns if it moves.
PENALTY_XG = 0.79

# The same number doubles as the conversion rate, and that is not a coincidence:
# Opta calibrates penalty xG to the long-run conversion rate. Using one value for
# both keeps stripping and adding back exactly self-consistent — remove 0.79 xG
# per historical penalty, add 0.79 expected goals per projected penalty.
PENALTY_CONVERSION = PENALTY_XG

# Fallbacks used only when the historical fit cannot run (no network, no cached
# report). These are the values `fit_historical` produced on 2026-08-14 from
# 2023-24 + 2025-26 (22,882 player-matches, 760 matches with bookmaker odds);
# they are recorded here so a cold offline run degrades to fitted numbers rather
# than to guesses, and every one of them is re-derived when the fit does run.
GOAL_ELASTICITY_PRIOR = 0.81
ASSIST_ELASTICITY_PRIOR = 1.00
PENALTY_RATE_PRIOR = 0.0815  # penalties awarded per team per match
PENALTY_TEAM_ELASTICITY_PRIOR = 1.24
FINISHING_PRIOR_XG_PRIOR = 104.0

# Regularisation for the penalty-rate-vs-team-strength exponent. Penalty awards
# are a ~0.08/match event so a per-team fit is desperately thin; this prior says
# "penalties scale roughly in proportion to attacking output (1.0), and I would
# be mildly surprised by anything outside 0.5-1.5".
PENALTY_ELASTICITY_PRIOR_MEAN = 1.0
PENALTY_ELASTICITY_PRIOR_SD = 0.5

# Finishing skill is allowed to move a projection by at most this much. The
# shrinkage below almost always keeps it well inside the cap; the cap exists so
# that a 300-minute player who converted three half-chances cannot be projected
# as a 1.15x finisher on the strength of noise.
FINISHING_CAP = 0.15
# Floor on the fitted finishing prior strength, in xG units. Even if a sample
# suggests real finishing spread, never trust a player's ratio on less than this
# much accumulated xG.
FINISHING_PRIOR_FLOOR = 25.0
# Minimum non-penalty xG in *both* seasons of a pair to enter the finishing fit.
FINISHING_MIN_XG = 3.0

# Sanity rail. Nobody in Premier League history has had a single-match goal
# expectation near this; a projection above it means an upstream lambda or a
# minutes forecast has gone wrong, and we clamp loudly rather than ship it.
MAX_LAMBDA_GOALS = 1.6
MAX_LAMBDA_ASSISTS = 1.2

# Fixture scaling is only meaningful over a plausible range of team lambdas.
MIN_LAMBDA_RATIO, MAX_LAMBDA_RATIO = 0.25, 3.0

# Minimum minutes for a player-season to enter the position/price baseline fit.
BASELINE_MIN_MINUTES = 450.0
BASELINE_MIN_PER_BUCKET = 12


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------


@dataclass
class HistoricalFit:
    """Parameters fitted from completed seasons; cached to reports/."""

    version: int = FIT_VERSION
    seasons: List[str] = field(default_factory=list)
    fitted_at: str = ""
    n_player_matches: int = 0
    n_team_matches: int = 0
    league_avg_team_lambda: float = 1.45

    goal_elasticity: float = GOAL_ELASTICITY_PRIOR
    assist_elasticity: float = ASSIST_ELASTICITY_PRIOR
    goal_elasticity_xg: float = float("nan")
    goal_elasticity_goals: float = float("nan")
    assist_elasticity_xa: float = float("nan")
    assist_elasticity_assists: float = float("nan")
    goal_elasticity_se: float = float("nan")
    assist_elasticity_se: float = float("nan")
    # Elasticity of realised team goals on the odds-implied lambda. A perfectly
    # calibrated forecast gives 1.0; the fitted player elasticities are divided
    # by this so they are expressed per unit of *calibrated* expected team goals.
    team_lambda_calibration: float = 1.0

    penalty_xg: float = PENALTY_XG
    penalty_conversion: float = PENALTY_CONVERSION
    penalty_rate_per_team_match: float = PENALTY_RATE_PRIOR
    penalty_team_elasticity: float = PENALTY_TEAM_ELASTICITY_PRIOR
    penalty_team_elasticity_raw: float = float("nan")
    penalty_team_elasticity_se: float = float("nan")
    n_penalty_misses: int = 0
    n_penalty_saves: int = 0

    finishing_prior_xg: float = FINISHING_PRIOR_XG_PRIOR
    finishing_skill_sd: float = 0.0
    finishing_theta: float = 1.0
    finishing_covariance: float = float("nan")
    finishing_covariance_se: float = float("nan")
    n_finishing_pairs: int = 0
    finishing_seasons: List[str] = field(default_factory=list)

    source: str = "prior"  # "fit", "cache" or "prior"
    warnings: List[str] = field(default_factory=list)


@dataclass
class PriceBaseline:
    """Piecewise-linear position baseline in price, fitted from last season."""

    position: int
    prices: List[float] = field(default_factory=list)
    xg90: List[float] = field(default_factory=list)
    xa90: List[float] = field(default_factory=list)
    n_players: List[int] = field(default_factory=list)

    def _interp(self, price: float, ys: List[float]) -> float:
        if not self.prices:
            return 0.0
        if len(self.prices) == 1:
            return ys[0]
        return float(np.interp(price, self.prices, ys))

    def xg(self, price: float) -> float:
        return self._interp(price, self.xg90)

    def xa(self, price: float) -> float:
        return self._interp(price, self.xa90)


@dataclass
class PlayerRates:
    """A player's shrunken per-90 attacking rates and how they were built."""

    player_id: int
    position: int
    price: float
    xg90: float = 0.0  # non-penalty, post-shrinkage
    xa90: float = 0.0
    raw_xg90: float = 0.0  # pre-shrinkage blend
    raw_xa90: float = 0.0
    baseline_xg90: float = 0.0
    baseline_xa90: float = 0.0
    n90: float = 0.0  # effective 90s behind the blend
    finishing: float = 1.0
    penalties_taken_prior: float = 0.0
    is_penalty_taker: bool = False
    was_penalty_taker: bool = False
    source_weights: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AttackingProjection:
    """One player, one fixture. Every term the breakdown needs is here."""

    player_id: int
    lambda_goals: float = 0.0
    lambda_assists: float = 0.0
    lambda_goals_open_play: float = 0.0
    lambda_goals_penalty: float = 0.0
    penalty_attempts: float = 0.0  # feeds the -2 penalty-miss term in xp.py
    xg90: float = 0.0
    xa90: float = 0.0
    finishing: float = 1.0
    fixture_scale_goals: float = 1.0
    fixture_scale_assists: float = 1.0
    minutes_fraction: float = 0.0
    team_lambda: float = 0.0
    clamped: bool = False

    @property
    def expected_goal_involvement(self) -> float:
        return self.lambda_goals + self.lambda_assists

    @property
    def involvement_variance(self) -> float:
        """Var(goals + assists). Both are Poisson counts, so the sum is Poisson."""
        return self.lambda_goals + self.lambda_assists


# ---------------------------------------------------------------------------
# Fixed-effects Poisson elasticity
# ---------------------------------------------------------------------------


def _fe_poisson_elasticity(
    group: np.ndarray, y: np.ndarray, x: np.ndarray, exposure: np.ndarray
) -> Tuple[float, float]:
    """Slope b in  E[y_it] = exp(a_i) * exposure_it * exp(b * x_it).

    The player effects a_i are profiled out analytically (for a log link the
    within-player normalisation has a closed form), leaving a one-dimensional
    concave problem solved by Newton. Quasi-Poisson: y may be continuous, which
    is what lets the same routine run on xG as well as on goals.
    """
    if len(y) < 50:
        return float("nan"), float("nan")
    Y = np.bincount(group, weights=y)
    T = float((y * x).sum())

    def score(b: float) -> Tuple[float, float]:
        w = exposure * np.exp(b * x)
        S = np.maximum(np.bincount(group, weights=w), 1e-300)
        S1 = np.bincount(group, weights=w * x)
        S2 = np.bincount(group, weights=w * x * x)
        ratio = S1 / S
        grad = T - float((Y * ratio).sum())
        hess = -float((Y * (S2 / S - ratio * ratio)).sum())
        return grad, hess

    b = 0.0
    for _ in range(80):
        grad, hess = score(b)
        if abs(hess) < 1e-12:
            break
        step = grad / hess
        nb = b - step
        if not np.isfinite(nb):
            return float("nan"), float("nan")
        if abs(nb - b) < 1e-11:
            b = nb
            break
        b = nb
    _, hess = score(b)
    se = math.sqrt(1.0 / max(-hess, 1e-12))
    return float(b), float(se)


def _poisson_log_slope(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Poisson regression log(mu) = a + b*x with an intercept. Returns (b, se)."""
    if len(y) < 10 or y.sum() <= 0:
        return float("nan"), float("nan")
    a, b = math.log(max(y.mean(), 1e-6)), 0.0
    X = np.column_stack([np.ones_like(x), x])
    A = np.eye(2)
    for _ in range(60):
        mu = np.exp(a + b * x)
        z = (a + b * x) + (y - mu) / np.maximum(mu, 1e-9)
        WX = X * mu[:, None]
        A = X.T @ WX
        try:
            sol = np.linalg.solve(A, WX.T @ z)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if abs(sol[1] - b) < 1e-11 and abs(sol[0] - a) < 1e-11:
            a, b = float(sol[0]), float(sol[1])
            break
        a, b = float(sol[0]), float(sol[1])
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return b, float("nan")
    return b, float(math.sqrt(max(cov[1, 1], 0.0)))


def _total_goals_line(p_over_25: float) -> float:
    """Total-goals line T with P(Poisson(T) >= 3) == p_over_25."""
    lo, hi = 0.3, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = 1.0 - sum(stats.poisson_pmf(k, mid) for k in range(0, 3))
        if p < p_over_25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Historical fit
# ---------------------------------------------------------------------------


def _odds_panel(season: str, cache: Cache) -> Optional[pd.DataFrame]:
    """Per player-match rows tagged with the bookmakers' expected team goals.

    The fixture forecast has to be *ex ante*: regressing a player's xG on the
    realised team xG of the same match would pick up the shared match-level
    shock (the team had a good day, everybody's numbers went up) and badly
    overstate how much a fixture moves a projection. Bookmaker 1X2 and
    over/under-2.5 prices are the market's pre-match expectation, which is
    exactly the quantity `TeamRatingModel` will supply at solve time.
    """
    gws = hist.load_vaastav_gws(season, cache=cache)
    results = hist.load_football_data([season], cache=cache).copy()
    results["date_d"] = results["date"].dt.date

    home = gws[gws["was_home"] == True].groupby("fixture")["team_short"].first()  # noqa: E712
    away = gws[gws["was_home"] == False].groupby("fixture")["team_short"].first()  # noqa: E712
    date = gws.groupby("fixture")["kickoff_time"].first()
    fx = pd.DataFrame({"home_short": home, "away_short": away, "date": date}).dropna()
    fx = fx.reset_index()
    fx["date"] = pd.to_datetime(fx["date"], utc=True, errors="coerce").dt.date

    joined = fx.merge(
        results, left_on=["date", "home_short", "away_short"],
        right_on=["date_d", "home_short", "away_short"], how="left",
    )
    # Kick-off times straddle midnight UTC, so retry the unmatched rows a day
    # either side before giving up on them.
    for offset in (1, -1):
        missing = joined["odds_home"].isna()
        if not bool(missing.any()):
            break
        shifted = results.copy()
        shifted["date_d"] = shifted["date_d"] + pd.Timedelta(days=offset)
        retry = joined.loc[missing, ["date", "home_short", "away_short"]].merge(
            shifted, left_on=["date", "home_short", "away_short"],
            right_on=["date_d", "home_short", "away_short"], how="left",
        )
        for col in ("odds_home", "odds_draw", "odds_away", "odds_over25", "odds_under25",
                    "home_goals", "away_goals"):
            joined.loc[missing, col] = retry[col].to_numpy()

    lam_h: List[float] = []
    lam_a: List[float] = []
    for _, row in joined.iterrows():
        oh, od, oa = row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")
        if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
            lam_h.append(float("nan"))
            lam_a.append(float("nan"))
            continue
        ph, pdraw, pa = stats.remove_vig([float(oh), float(od), float(oa)])
        over, under = row.get("odds_over25"), row.get("odds_under25")
        if pd.notna(over) and pd.notna(under):
            p_over, _ = stats.remove_vig([float(over), float(under)])
            total = _total_goals_line(p_over)
        else:
            total = 2.75
        h, a = stats.lambdas_from_odds(ph, pdraw, pa, total_goals_line=total)
        lam_h.append(h)
        lam_a.append(a)
    joined["lam_h"] = lam_h
    joined["lam_a"] = lam_a

    panel = gws.merge(joined[["fixture", "lam_h", "lam_a"]], on="fixture", how="left")
    panel["team_lambda"] = np.where(panel["was_home"], panel["lam_h"], panel["lam_a"])
    panel["season"] = season
    return panel[panel["team_lambda"].notna()].copy()


def _season_totals(season: str, cache: Cache, penalty_rate: float) -> Optional[pd.DataFrame]:
    """Player-season non-penalty goals and xG from the end-of-season snapshot.

    ``players_raw.csv`` carries that season's ``penalties_order``, which is the
    only record anywhere of who actually had the duty, and it needs no team-name
    mapping, so it survives seasons whose ``merged_gw`` cannot be normalised.
    """
    praw = hist.load_vaastav_players_raw(season, cache=cache)
    if "expected_goals" not in praw.columns:
        return None  # FPL only published xG from 2022-23
    num = lambda col: pd.to_numeric(praw.get(col), errors="coerce")  # noqa: E731
    df = pd.DataFrame({
        "code": num("code"),
        "minutes": num("minutes").fillna(0.0),
        "xg": num("expected_goals").fillna(0.0),
        "goals": num("goals_scored").fillna(0.0),
        "missed": num("penalties_missed").fillna(0.0),
        "order": num("penalties_order"),
    }).dropna(subset=["code"])
    taken = df["missed"] + np.where(
        df["order"] == 1, penalty_rate * (df["minutes"] / 90.0) * PENALTY_CONVERSION, 0.0)
    df["xg_np"] = (df["xg"] - PENALTY_XG * taken).clip(lower=0.0)
    df["goals_np"] = (df["goals"] - (taken - df["missed"]).clip(lower=0.0)).clip(lower=0.0)
    df["season"] = season
    return df


def fit_finishing_prior(
    seasons: Sequence[str], cache: Cache, penalty_rate: float
) -> Dict[str, Any]:
    """Split observed goals/xG spread into finishing skill and sampling noise.

    Identification comes from *consecutive seasons of the same player*: the
    sampling noise in one season is independent of the next, so the covariance
    of the two finishing ratios is the skill variance with the noise removed.
    That is a far better-identified estimate than a within-season variance
    decomposition, which has to guess how much of a single season's spread was
    luck.

    The answer is that finishing skill is small and barely detectable: the
    covariance comes out about one standard error from zero, giving a prior
    strength of order 100 xG. A player needs a *career* of finishing to move his
    own projection by a few percent, which is exactly the intended behaviour —
    goals-minus-xG is overwhelmingly noise and treating it as signal is one of
    the most common ways a public model goes wrong.
    """
    frames: List[pd.DataFrame] = []
    used: List[str] = []
    for season in seasons:
        try:
            df = _season_totals(season, cache, penalty_rate)
        except Exception as exc:
            log.warning("finishing fit: skipping %s (%s)", season, exc)
            continue
        if df is not None:
            frames.append(df)
            used.append(season)
    if len(frames) < 2:
        return {"n_pairs": 0, "seasons": used}

    pairs: List[pd.DataFrame] = []
    for first, second in zip(frames, frames[1:]):
        merged = first[["code", "xg_np", "goals_np"]].merge(
            second[["code", "xg_np", "goals_np"]], on="code", suffixes=("_1", "_2"))
        pairs.append(merged)
    panel = pd.concat(pairs, ignore_index=True)
    panel = panel[(panel["xg_np_1"] >= FINISHING_MIN_XG)
                  & (panel["xg_np_2"] >= FINISHING_MIN_XG)]
    if len(panel) < 30:
        return {"n_pairs": int(len(panel)), "seasons": used}

    theta = float((panel["goals_np_1"].sum() + panel["goals_np_2"].sum())
                  / max(panel["xg_np_1"].sum() + panel["xg_np_2"].sum(), 1e-9))
    r1 = (panel["goals_np_1"] / panel["xg_np_1"]).to_numpy(float)
    r2 = (panel["goals_np_2"] / panel["xg_np_2"]).to_numpy(float)
    # Weight each pair by the harmonic mean of its two xG totals: that is
    # proportional to the precision of the product term.
    w = 2.0 / (1.0 / panel["xg_np_1"].to_numpy(float) + 1.0 / panel["xg_np_2"].to_numpy(float))
    prod = (r1 - theta) * (r2 - theta)
    cov = float((w * prod).sum() / w.sum())
    se = float(np.std(prod, ddof=1) / math.sqrt(len(prod)))
    out: Dict[str, Any] = {
        "n_pairs": int(len(panel)), "theta": theta, "cov": cov, "cov_se": se,
        "seasons": used,
    }
    if cov <= 0:
        out["prior_xg"] = 1e6
        out["skill_sd"] = 0.0
    else:
        out["prior_xg"] = max(theta / cov, FINISHING_PRIOR_FLOOR)
        out["skill_sd"] = math.sqrt(cov)
    return out


def fit_historical(
    seasons: Optional[Sequence[str]] = None,
    cache: Optional[Cache] = None,
    finishing_seasons: Optional[Sequence[str]] = None,
) -> HistoricalFit:
    """Fit the fixture elasticities, the penalty rate and finishing skill."""
    cache = cache or Cache()
    seasons = list(seasons or ("2023-24", "2024-25", "2025-26"))
    finishing_seasons = list(
        finishing_seasons or ("2022-23", "2023-24", "2024-25", "2025-26"))
    out = HistoricalFit(fitted_at=datetime.now(timezone.utc).isoformat(), source="fit")

    panels: List[pd.DataFrame] = []
    for season in seasons:
        try:
            panels.append(_odds_panel(season, cache))
            out.seasons.append(season)
        except Exception as exc:  # a missing season must be visible, not silent
            msg = "season %s excluded from the attacking fit (%s: %s)" % (
                season, type(exc).__name__, exc)
            log.warning(msg)
            out.warnings.append(msg)
    if not panels:
        out.source = "prior"
        out.warnings.append("no historical season loaded; using recorded fitted priors")
        return out

    panel = pd.concat(panels, ignore_index=True)

    # --- penalty xG constant: re-verify the spike at 0.79 -------------------
    rounded = panel["expected_goals"].round(2)
    spike = float((rounded == PENALTY_XG).sum())
    neighbours = [float((rounded == round(PENALTY_XG + d, 2)).sum())
                  for d in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03)]
    baseline = sum(neighbours) / len(neighbours) if neighbours else 0.0
    if spike < 1.5 * max(baseline, 1.0):
        out.warnings.append(
            "no clear xG spike at %.2f (n=%d vs neighbour mean %.1f); the penalty "
            "xG constant may have changed" % (PENALTY_XG, spike, baseline))

    played = panel[panel["minutes"] > 0].copy()
    league_lambda = float(played["team_lambda"].mean())
    out.league_avg_team_lambda = league_lambda
    out.n_player_matches = int(len(played))

    x = np.log((played["team_lambda"] / league_lambda).to_numpy(float))
    exposure = (played["minutes"].to_numpy(float)) / 90.0
    keys = (played["season"].astype(str) + "_" + played["element"].astype(str)).to_numpy()
    group, _uniq = pd.factorize(keys)

    def elasticity(col: str, min_total: float = 1.0) -> Tuple[float, float]:
        y = played[col].to_numpy(float)
        totals = np.bincount(group, weights=y)
        keep = totals[group] >= min_total
        if keep.sum() < 200:
            return float("nan"), float("nan")
        sub_group, _ = pd.factorize(group[keep])
        return _fe_poisson_elasticity(sub_group, y[keep], x[keep], exposure[keep])

    b_xg, se_xg = elasticity("expected_goals")
    b_g, se_g = elasticity("goals_scored")
    b_xa, se_xa = elasticity("expected_assists")
    b_a, se_a = elasticity("assists")
    out.goal_elasticity_xg, out.goal_elasticity_goals = b_xg, b_g
    out.assist_elasticity_xa, out.assist_elasticity_assists = b_xa, b_a

    # --- how well calibrated is the regressor itself? ----------------------
    team = played.groupby(["season", "fixture", "team_short"], as_index=False).agg(
        team_goals=("goals_scored", "sum"), lam=("team_lambda", "mean"))
    cal, _cal_se = _poisson_log_slope(
        team["team_goals"].to_numpy(float),
        np.log((team["lam"] / league_lambda).to_numpy(float)),
    )
    out.n_team_matches = int(len(team))
    if np.isfinite(cal) and 0.5 < cal < 1.5:
        out.team_lambda_calibration = float(cal)

    def combine(b1: float, s1: float, b2: float, s2: float,
                fallback: float) -> Tuple[float, float]:
        pairs = [(b, s) for b, s in ((b1, s1), (b2, s2)) if np.isfinite(b) and np.isfinite(s)]
        if not pairs:
            return fallback, float("nan")
        # The xG-based and outcome-based estimates measure the same elasticity
        # off the same rows, so they are correlated; a plain mean is the honest
        # combination and inverse-variance weighting would overstate precision.
        est = sum(b for b, _ in pairs) / len(pairs)
        se = max(s for _, s in pairs)
        est /= max(out.team_lambda_calibration, 1e-6)
        se /= max(out.team_lambda_calibration, 1e-6)
        return float(min(max(est, 0.4), 1.4)), float(se)

    out.goal_elasticity, out.goal_elasticity_se = combine(
        b_xg, se_xg, b_g, se_g, GOAL_ELASTICITY_PRIOR)
    out.assist_elasticity, out.assist_elasticity_se = combine(
        b_xa, se_xa, b_a, se_a, ASSIST_ELASTICITY_PRIOR)

    # --- penalties ----------------------------------------------------------
    misses = float(panel["penalties_missed"].sum())
    saves = float(panel["penalties_saved"].sum())
    n_matches = int(panel.groupby(["season", "fixture"]).ngroups)
    team_matches = 2.0 * n_matches
    out.n_penalty_misses = int(misses)
    out.n_penalty_saves = int(saves)
    # Penalties awarded are not published per match anywhere we ingest, but the
    # misses are, and a miss is a Bernoulli(1 - conversion) draw on every
    # penalty taken. Inverting that gives the volume from real counts.
    awarded = misses / max(1.0 - PENALTY_CONVERSION, 1e-6)
    if team_matches > 0 and misses > 0:
        out.penalty_rate_per_team_match = float(awarded / team_matches)

    team_season = panel.groupby(["season", "team_short"], as_index=False).agg(
        misses=("penalties_missed", "sum"), xg=("expected_goals", "sum"))
    if len(team_season) >= 20 and team_season["misses"].sum() >= 10:
        lx = np.log((team_season["xg"] / team_season["xg"].mean()).to_numpy(float))
        b_pen, se_pen = _poisson_log_slope(team_season["misses"].to_numpy(float), lx)
        out.penalty_team_elasticity_raw, out.penalty_team_elasticity_se = b_pen, se_pen
        if np.isfinite(b_pen) and np.isfinite(se_pen) and se_pen > 0:
            prior_prec = 1.0 / (PENALTY_ELASTICITY_PRIOR_SD ** 2)
            data_prec = 1.0 / (se_pen ** 2)
            out.penalty_team_elasticity = float(
                (PENALTY_ELASTICITY_PRIOR_MEAN * prior_prec + b_pen * data_prec)
                / (prior_prec + data_prec)
            )

    # --- finishing skill ----------------------------------------------------
    try:
        fin = fit_finishing_prior(
            finishing_seasons, cache, out.penalty_rate_per_team_match)
    except Exception as exc:
        fin = {"n_pairs": 0, "seasons": []}
        out.warnings.append("finishing fit failed (%s: %s)" % (type(exc).__name__, exc))
    out.n_finishing_pairs = int(fin.get("n_pairs", 0))
    out.finishing_seasons = list(fin.get("seasons", []))
    if fin.get("n_pairs", 0) >= 30:
        out.finishing_theta = float(fin.get("theta", 1.0))
        out.finishing_covariance = float(fin.get("cov", float("nan")))
        out.finishing_covariance_se = float(fin.get("cov_se", float("nan")))
        out.finishing_prior_xg = float(fin.get("prior_xg", FINISHING_PRIOR_XG_PRIOR))
        out.finishing_skill_sd = float(fin.get("skill_sd", 0.0))
        if out.finishing_prior_xg >= 1e6:
            out.warnings.append(
                "no finishing skill detectable across %d season pairs; the finishing "
                "adjustment is inert" % out.n_finishing_pairs)
    else:
        out.warnings.append(
            "finishing prior not fitted (only %d usable season pairs); using the "
            "recorded value" % out.n_finishing_pairs)
    return out


def load_or_fit_historical(
    seasons: Optional[Sequence[str]] = None,
    cache: Optional[Cache] = None,
    force: bool = False,
    path: str = FIT_PATH,
) -> HistoricalFit:
    """Read ``reports/attacking_fit.json``, refitting when it is stale or absent.

    The fit walks two full seasons of player-matches and inverts 760 sets of
    bookmaker odds; caching it keeps a projection run in the seconds the spec
    asks for instead of redoing that work on every call.
    """
    if not force and os.path.exists(path):
        try:
            with open(path, "r") as fh:
                raw = json.load(fh)
            if int(raw.get("version", 0)) == FIT_VERSION:
                known = {f for f in HistoricalFit.__dataclass_fields__}
                fit = HistoricalFit(**{k: v for k, v in raw.items() if k in known})
                fit.source = "cache"
                return fit
        except Exception as exc:
            log.warning("could not read %s (%s); refitting", path, exc)
    try:
        fit = fit_historical(seasons, cache=cache)
    except Exception as exc:
        log.warning("attacking historical fit failed (%s: %s); using recorded priors",
                    type(exc).__name__, exc)
        fit = HistoricalFit(source="prior",
                            warnings=["historical fit failed: %s" % exc])
        return fit
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(fit), fh, indent=2)
    except OSError as exc:  # not fatal, just means we refit next time
        log.warning("could not write %s: %s", path, exc)
    return fit


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class AttackingModel:
    """Per-90 attacking rates, fixture scaling, penalties and their variance."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.fit_params = HistoricalFit()
        self.baselines: Dict[int, PriceBaseline] = {}
        self.rates_by_player: Dict[int, PlayerRates] = {}
        self.league_avg_lambda: float = self.config.model.league_avg_goals_per_team
        self.finishing_prior_xg: float = FINISHING_PRIOR_XG_PRIOR
        self.finishing_skill_sd: float = 0.0
        self.finishing_theta: float = 1.0
        self.warnings: List[str] = []
        self._state: Optional["GameState"] = None
        self._prior_takers: Dict[int, int] = {}       # player code -> penalties_order
        self._prior_team_ratio: Dict[int, float] = {}  # player code -> team attack ratio
        self._clamped: List[int] = []

    # -- fitting ------------------------------------------------------------

    def fit(self, state: "GameState", force_refit: bool = False) -> None:
        self._state = state
        self.warnings = []
        cache = Cache(default_ttl=self.config.cache_ttl_seconds)

        self.fit_params = load_or_fit_historical(cache=cache, force=force_refit)
        self.warnings.extend(self.fit_params.warnings)
        if self.fit_params.source == "prior":
            self.warnings.append(
                "attacking elasticities/penalty rate come from recorded priors, "
                "not from a fit run in this session")
        # The elasticity was fitted against a league-average team lambda; keep the
        # reference consistent with whatever the team model actually produces.
        self.league_avg_lambda = self.config.model.league_avg_goals_per_team

        self.finishing_prior_xg = max(self.fit_params.finishing_prior_xg,
                                      FINISHING_PRIOR_FLOOR)
        self.finishing_skill_sd = self.fit_params.finishing_skill_sd
        self.finishing_theta = self.fit_params.finishing_theta or 1.0

        self._load_prior_penalty_takers(state, cache)
        sample = self._prior_season_sample(state)
        self._fit_baselines(sample)
        self._build_rates(state, sample)

    def set_league_average_lambda(self, value: float) -> None:
        """Let the assembler pin the fixture-scaling reference to the team model."""
        if value and value > 0:
            self.league_avg_lambda = float(value)

    # -- prior-season penalty takers ---------------------------------------

    def _load_prior_penalty_takers(self, state: "GameState", cache: Cache) -> None:
        """Who was on penalties *last* season, and how good was their team.

        The bootstrap only ever exposes the current designated taker, so without
        this the model cannot tell a player who has just been handed penalties
        from one who has had them for years — and that distinction is the whole
        point of treating penalties separately.
        """
        self._prior_takers = {}
        self._prior_team_ratio = {}
        prior = state.prior_season or hist.prior_season(state.season)
        try:
            praw = hist.load_vaastav_players_raw(prior, cache=cache)
        except Exception as exc:
            msg = ("prior-season penalty takers unavailable (%s: %s); falling back to "
                   "the current penalties_order, which will under-strip penalty xG for "
                   "players who have just lost the duty" % (type(exc).__name__, exc))
            log.warning(msg)
            self.warnings.append(msg)
            return
        order = pd.to_numeric(praw.get("penalties_order"), errors="coerce")
        codes = pd.to_numeric(praw.get("code"), errors="coerce")
        for code, o in zip(codes, order):
            if pd.notna(code) and pd.notna(o):
                self._prior_takers[int(code)] = int(o)

        try:
            gws = hist.load_vaastav_gws(prior, cache=cache)
        except Exception:
            return
        per_team = gws.groupby("team_short")["goals_scored"].sum()
        matches = gws.groupby("team_short")["GW"].nunique().replace(0, np.nan)
        gpm = (per_team / matches).dropna()
        if gpm.empty:
            return
        ratio = (gpm / gpm.mean()).to_dict()
        element_team = gws.groupby("element")["team_short"].agg(
            lambda s: s.value_counts().index[0])
        id_to_code = dict(zip(pd.to_numeric(praw["id"], errors="coerce"),
                              pd.to_numeric(praw["code"], errors="coerce")))
        for element, short in element_team.items():
            code = id_to_code.get(element)
            if code is not None and pd.notna(code) and short in ratio:
                self._prior_team_ratio[int(code)] = float(ratio[short])

    # -- sample construction ------------------------------------------------

    def _penalties_taken(
        self, minutes: float, missed: float, was_taker: bool, team_ratio: float
    ) -> float:
        """Posterior expected penalties taken given the observed misses.

        Penalties taken is Poisson; misses and conversions are independent
        thinnings of it, so observing m misses gives E[N | m] = m + conversion *
        E[N], and the observed misses add rather than replace. That keeps a
        one-off taker who missed his only penalty from being credited with five.
        """
        expected = 0.0
        if was_taker:
            rate = self.fit_params.penalty_rate_per_team_match * (
                max(team_ratio, 0.2) ** self.fit_params.penalty_team_elasticity)
            expected = rate * (max(minutes, 0.0) / 90.0)
        return max(missed, 0.0) + expected * self.fit_params.penalty_conversion

    def _prior_season_sample(self, state: "GameState") -> pd.DataFrame:
        """One row per player with a prior-season record, penalty xG removed."""
        # Pre-season the bootstrap element totals *are* last season's numbers
        # (GameState.elements_are_prior_season), which is the only source left if
        # the per-player summaries were skipped. Never read them as prior-season
        # form once a gameweek of the current season has finished.
        fallback: Dict[int, Dict[str, Any]] = {}
        if state.elements_are_prior_season:
            for raw in state.bootstrap.get("elements", []):
                fallback[int(raw["id"])] = raw

        rows: List[Dict[str, Any]] = []
        for pid, player in state.players.items():
            row = state.prior_season_row(pid)
            seasons_back = 1
            if row is None:
                row = state.latest_season_row(pid)
                if row is not None:
                    try:
                        seasons_back = max(
                            1,
                            hist.season_start_year(state.season)
                            - hist.season_start_year(str(row.get("season_name", ""))),
                        )
                    except (ValueError, TypeError):
                        seasons_back = 2
            if row is None:
                row = fallback.get(pid)
                seasons_back = 1
            if row is None:
                continue
            minutes = float(row.get("minutes") or 0.0)
            if minutes <= 0:
                continue
            code = int(player.code)
            prior_order = self._prior_takers.get(code)
            was_taker = (prior_order == 1) if prior_order is not None \
                else (player.penalties_order == 1)
            missed = float(row.get("penalties_missed") or 0.0)
            team_ratio = self._prior_team_ratio.get(code, 1.0)
            pens = self._penalties_taken(minutes, missed, was_taker, team_ratio)
            xg = float(row.get("expected_goals") or 0.0)
            goals = float(row.get("goals_scored") or 0.0)
            rows.append({
                "player_id": pid,
                "position": player.position,
                "price": player.price,
                "minutes": minutes,
                "n90": minutes / 90.0,
                "seasons_back": seasons_back,
                "xg": max(xg - self.fit_params.penalty_xg * pens, 0.0),
                "xa": float(row.get("expected_assists") or 0.0),
                "goals": max(goals - max(pens - missed, 0.0), 0.0),
                "assists": float(row.get("assists") or 0.0),
                "pens_taken": pens,
                "was_taker": bool(was_taker),
            })
        return pd.DataFrame(rows)

    # -- baselines ----------------------------------------------------------

    def _fit_baselines(self, sample: pd.DataFrame) -> None:
        """Pooled non-penalty xG90/xA90 by position and price bucket."""
        self.baselines = {}
        usable = sample
        if len(usable):
            usable = usable[(usable["minutes"] >= BASELINE_MIN_MINUTES)
                            & (usable["seasons_back"] == 1)]
        for pos in scoring.POSITIONS:
            sub = usable[usable["position"] == pos] if len(usable) else usable
            if sub is None or len(sub) < BASELINE_MIN_PER_BUCKET:
                self.baselines[pos] = PriceBaseline(position=pos)
                self.warnings.append(
                    "no price baseline fitted for %s (only %d qualifying players)"
                    % (scoring.POS_NAME[pos], 0 if sub is None else len(sub)))
                continue
            n_buckets = max(1, min(6, len(sub) // BASELINE_MIN_PER_BUCKET))
            try:
                bucket = pd.qcut(sub["price"], n_buckets, labels=False, duplicates="drop")
            except ValueError:
                bucket = pd.Series(np.zeros(len(sub), dtype=int), index=sub.index)
            grp = sub.assign(bucket=bucket).groupby("bucket")
            agg = grp.agg(price=("price", "mean"), xg=("xg", "sum"), xa=("xa", "sum"),
                          n90=("n90", "sum"), n=("player_id", "size"))
            agg = agg[agg["n90"] > 0].sort_values("price")
            self.baselines[pos] = PriceBaseline(
                position=pos,
                prices=[float(p) for p in agg["price"]],
                # Pooled (minutes-weighted) rate, which is the empirical-Bayes
                # prior mean for the bucket, not the mean of per-player ratios.
                xg90=[float(g / n) for g, n in zip(agg["xg"], agg["n90"])],
                xa90=[float(a / n) for a, n in zip(agg["xa"], agg["n90"])],
                n_players=[int(n) for n in agg["n"]],
            )

    def baseline(self, position: int, price: float) -> Tuple[float, float]:
        base = self.baselines.get(position)
        if base is None or not base.prices:
            return 0.0, 0.0
        return base.xg(price), base.xa(price)

    # -- finishing skill ----------------------------------------------------

    def _finishing(self, goals: float, xg: float) -> float:
        """Shrunken goals-per-xG multiplier, capped at +-FINISHING_CAP.

        The prior strength comes from ``fit_finishing_prior``: about 100 xG of
        accumulated evidence before a player's own conversion ratio gets half the
        weight. On a 20-xG season that is roughly a sixth of the way from 1.0 to
        the observed ratio, so a 40% overperformer is projected about 6% up. This
        is deliberately timid. Goals minus xG is dominated by sampling noise, and
        the cross-season covariance that anchors the prior sits only about one
        standard error above zero.
        """
        if xg <= 0:
            return 1.0
        raw = goals / xg
        shrunk = stats.shrink(raw, xg, self.finishing_theta, self.finishing_prior_xg)
        # Express relative to the league conversion level so a league-average
        # finisher is exactly neutral even if xG is not perfectly calibrated.
        adj = shrunk / max(self.finishing_theta, 1e-6)
        return float(min(max(adj, 1.0 - FINISHING_CAP), 1.0 + FINISHING_CAP))

    # -- per-player rates ---------------------------------------------------

    def _current_season_window(
        self, state: "GameState", player_id: int, last_n: Optional[int]
    ) -> Optional[Dict[str, float]]:
        df = state.history(player_id)
        if df is None or not len(df):
            return None
        if last_n is not None and "gw" in df.columns and df["gw"].notna().any():
            cutoff = float(df["gw"].max()) - last_n + 1
            df = df[df["gw"] >= cutoff]
        if not len(df):
            return None
        minutes = float(pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0).sum())
        if minutes <= 0:
            return None
        return {
            "minutes": minutes,
            "xg": float(pd.to_numeric(df.get("expected_goals"), errors="coerce").fillna(0).sum()),
            "xa": float(pd.to_numeric(df.get("expected_assists"), errors="coerce").fillna(0).sum()),
            "goals": float(pd.to_numeric(df.get("goals_scored"), errors="coerce").fillna(0).sum()),
            "missed": float(pd.to_numeric(df.get("penalties_missed"), errors="coerce").fillna(0).sum()),
        }

    def _build_rates(self, state: "GameState", sample: pd.DataFrame) -> None:
        model = self.config.model
        prior_rows = {int(r["player_id"]): r for _, r in sample.iterrows()} if len(sample) else {}
        self.rates_by_player = {}

        for pid, player in state.players.items():
            base_xg, base_xa = self.baseline(player.position, player.price)
            is_taker = player.penalties_order == 1
            team_ratio = self._prior_team_ratio.get(int(player.code), 1.0)

            values_xg: List[float] = []
            values_xa: List[float] = []
            weights: List[float] = []
            labels: List[str] = []
            n90_current = 0.0
            n90_prior = 0.0
            fin_goals = 0.0
            fin_xg = 0.0

            prior = prior_rows.get(pid)
            if prior is not None and float(prior["n90"]) > 0:
                n90_prior = float(prior["n90"])
                values_xg.append(float(prior["xg"]) / n90_prior)
                values_xa.append(float(prior["xa"]) / n90_prior)
                # A record two or more seasons old still beats no record at all,
                # but it should not carry a current season's weight.
                decay = 0.5 ** (int(prior["seasons_back"]) - 1)
                weights.append(model.weight_prior_season * decay)
                labels.append("prior")
                fin_goals += float(prior["goals"])
                fin_xg += float(prior["xg"])

            std = self._current_season_window(state, pid, None)
            if std is not None:
                pens = self._penalties_taken(std["minutes"], std["missed"], is_taker, team_ratio)
                n90 = std["minutes"] / 90.0
                n90_current = n90
                xg_np = max(std["xg"] - self.fit_params.penalty_xg * pens, 0.0)
                values_xg.append(xg_np / n90)
                values_xa.append(std["xa"] / n90)
                weights.append(model.weight_season_to_date)
                labels.append("season")
                fin_goals += max(std["goals"] - max(pens - std["missed"], 0.0), 0.0)
                fin_xg += xg_np

            recent = self._current_season_window(state, pid, model.recent_form_gws)
            if recent is not None:
                pens = self._penalties_taken(recent["minutes"], recent["missed"], is_taker, team_ratio)
                n90 = recent["minutes"] / 90.0
                xg_np = max(recent["xg"] - self.fit_params.penalty_xg * pens, 0.0)
                values_xg.append(xg_np / n90)
                values_xa.append(recent["xa"] / n90)
                weights.append(model.weight_recent_form)
                labels.append("recent")

            total_w = sum(weights)
            if total_w <= 0:
                raw_xg, raw_xa, n90_eff = base_xg, base_xa, 0.0
                norm: List[float] = []
            else:
                norm = [w / total_w for w in weights]
                raw_xg = stats.weighted_blend(values_xg, weights)
                raw_xa = stats.weighted_blend(values_xa, weights)
                # Recent form is a subset of season-to-date, so the two share their
                # sample: the current season contributes its 90s once, weighted by
                # the two weights combined.
                w_by_label = dict(zip(labels, norm))
                n90_eff = (w_by_label.get("prior", 0.0) * n90_prior
                           + (w_by_label.get("season", 0.0)
                              + w_by_label.get("recent", 0.0)) * n90_current)

            shrink_n = model.shrinkage_90s_attacking
            xg90 = stats.shrink(raw_xg, n90_eff, base_xg, shrink_n)
            xa90 = stats.shrink(raw_xa, n90_eff, base_xa, shrink_n)

            prior_order = self._prior_takers.get(int(player.code))
            was_taker = (prior_order == 1) if prior_order is not None else is_taker
            parts = ["%s %.0f%%" % (l, 100 * w) for l, w in zip(labels, norm)] or ["baseline only"]
            reason = "%s, %.1f x90 -> %d%% own rate; %s" % (
                ", ".join(parts), n90_eff,
                int(round(100 * (n90_eff / (n90_eff + shrink_n)) if n90_eff > 0 else 0)),
                "penalties" if is_taker else "no penalties",
            )
            if is_taker and not was_taker:
                reason += " (NEW on penalties)"
            elif was_taker and not is_taker:
                reason += " (lost penalties)"

            self.rates_by_player[pid] = PlayerRates(
                player_id=pid,
                position=player.position,
                price=player.price,
                xg90=max(xg90, 0.0),
                xa90=max(xa90, 0.0),
                raw_xg90=max(raw_xg, 0.0),
                raw_xa90=max(raw_xa, 0.0),
                baseline_xg90=base_xg,
                baseline_xa90=base_xa,
                n90=n90_eff,
                finishing=self._finishing(fin_goals, fin_xg),
                penalties_taken_prior=float(prior["pens_taken"]) if prior is not None else 0.0,
                is_penalty_taker=is_taker,
                was_penalty_taker=bool(was_taker),
                source_weights=dict(zip(labels, norm)),
                reason=reason,
            )

    # -- public API ---------------------------------------------------------

    def rates(self, player_id: int) -> Tuple[float, float]:
        """(xG90, xA90) after blending and shrinkage, penalty xG excluded.

        Penalties live in their own term in ``project`` so that gaining or losing
        the duty changes a projection; a caller who wants the all-in rate should
        use ``rates_including_penalties``.
        """
        r = self.rates_by_player.get(player_id)
        if r is None:
            return 0.0, 0.0
        return r.xg90, r.xa90

    def rates_including_penalties(self, player_id: int) -> Tuple[float, float]:
        """(xG90, xA90) with the penalty term added back at a neutral fixture."""
        r = self.rates_by_player.get(player_id)
        if r is None:
            return 0.0, 0.0
        pen = 0.0
        if r.is_penalty_taker:
            pen = (self.fit_params.penalty_rate_per_team_match
                   * self.fit_params.penalty_conversion)
        return r.xg90 * r.finishing + pen, r.xa90

    def player_rates(self, player_id: int) -> Optional[PlayerRates]:
        return self.rates_by_player.get(player_id)

    def fixture_scale(self, team_lambda: float) -> Tuple[float, float]:
        """(goal multiplier, assist multiplier) for a team expected to score `team_lambda`.

        Fitted by profile-likelihood Poisson regression with player fixed effects
        and a minutes offset, on 22.9k player-matches priced by the bookmakers
        (see ``fit_historical``). Both exponents sit below or at 1.0: a fixture
        twice as good does not double a striker's expected goals, because the
        extra chances in a rout are shared out over more players and more
        substitutes.

        The fitted numbers were xG 0.70 (se 0.10) and xA 0.94 (se 0.13) against
        the odds lambda, 0.81 / 1.00 after correcting for that lambda's own
        calibration slope. Note that this contradicts the usual claim that goals
        respond to fixture more strongly than assists do: in this data assists
        respond at least as strongly, robustly across seasons, across minutes
        filters and with penalty takers excluded. The gap is under 1.5 standard
        errors, so the honest reading is that the two responses are hard to tell
        apart — but nothing in the data supports forcing goals to be steeper, so
        the fitted values are used as they came out.
        """
        ratio = float(team_lambda) / max(self.league_avg_lambda, 1e-6)
        ratio = min(max(ratio, MIN_LAMBDA_RATIO), MAX_LAMBDA_RATIO)
        return (ratio ** self.fit_params.goal_elasticity,
                ratio ** self.fit_params.assist_elasticity)

    def penalty_rate(self, team_lambda: float) -> float:
        """Penalties awarded to a team, per match, at this fixture's expected goals."""
        ratio = float(team_lambda) / max(self.league_avg_lambda, 1e-6)
        ratio = min(max(ratio, MIN_LAMBDA_RATIO), MAX_LAMBDA_RATIO)
        return (self.fit_params.penalty_rate_per_team_match
                * ratio ** self.fit_params.penalty_team_elasticity)

    def project_breakdown(
        self, player_id: int, fixture_ctx: "FixtureContext"
    ) -> AttackingProjection:
        r = self.rates_by_player.get(player_id)
        out = AttackingProjection(player_id=player_id)
        if r is None:
            return out

        minutes = getattr(fixture_ctx, "minutes", None)
        xmins = float(getattr(minutes, "xmins", 0.0) or 0.0) if minutes is not None else 0.0
        frac = min(max(xmins, 0.0), 90.0) / 90.0
        team_lambda = float(getattr(fixture_ctx, "team_lambda", self.league_avg_lambda) or 0.0)
        scale_g, scale_a = self.fixture_scale(team_lambda)

        open_play = r.xg90 * frac * scale_g * r.finishing
        attempts = self.penalty_rate(team_lambda) * frac if r.is_penalty_taker else 0.0
        pen_goals = attempts * self.fit_params.penalty_conversion
        lam_g = open_play + pen_goals
        lam_a = r.xa90 * frac * scale_a

        clamped = False
        if lam_g > MAX_LAMBDA_GOALS or lam_a > MAX_LAMBDA_ASSISTS:
            clamped = True
            if player_id not in self._clamped:
                self._clamped.append(player_id)
                log.warning(
                    "clamping player %d: lambda_goals=%.3f lambda_assists=%.3f "
                    "(xG90=%.3f xA90=%.3f xmins=%.1f team_lambda=%.2f)",
                    player_id, lam_g, lam_a, r.xg90, r.xa90, xmins, team_lambda)
            lam_g = min(lam_g, MAX_LAMBDA_GOALS)
            lam_a = min(lam_a, MAX_LAMBDA_ASSISTS)

        out.lambda_goals = lam_g
        out.lambda_assists = lam_a
        out.lambda_goals_open_play = min(open_play, lam_g)
        out.lambda_goals_penalty = max(lam_g - min(open_play, lam_g), 0.0)
        out.penalty_attempts = attempts
        out.xg90 = r.xg90
        out.xa90 = r.xa90
        out.finishing = r.finishing
        out.fixture_scale_goals = scale_g
        out.fixture_scale_assists = scale_a
        out.minutes_fraction = frac
        out.team_lambda = team_lambda
        out.clamped = clamped
        return out

    def project(
        self, player_id: int, fixture_ctx: "FixtureContext"
    ) -> Tuple[float, float]:
        b = self.project_breakdown(player_id, fixture_ctx)
        return b.lambda_goals, b.lambda_assists

    # -- distributional helpers --------------------------------------------

    def involvement(
        self, player_id: int, fixture_ctx: "FixtureContext"
    ) -> Tuple[float, float]:
        """(E[goals + assists], Var[goals + assists]) for one fixture.

        Goals and assists are each Poisson counts, so their sum is Poisson with
        the summed rate and mean equals variance. Treating them as independent
        slightly overstates the variance (you cannot assist your own goal), which
        is the conservative direction for a captaincy call.
        """
        b = self.project_breakdown(player_id, fixture_ctx)
        return b.expected_goal_involvement, b.involvement_variance

    def points_variance(
        self, player_id: int, fixture_ctx: "FixtureContext"
    ) -> float:
        """Var of the attacking points only, for xp.py's sd_total."""
        b = self.project_breakdown(player_id, fixture_ctx)
        r = self.rates_by_player.get(player_id)
        pos = r.position if r is not None else scoring.MID
        gp = float(scoring.GOAL_POINTS[pos])
        ap = float(scoring.ASSIST_POINTS)
        return gp * gp * b.lambda_goals + ap * ap * b.lambda_assists

    def p_returns(
        self, player_id: int, fixture_ctx: "FixtureContext", k: int = 1
    ) -> float:
        """P(at least k goals+assists)."""
        mean, _ = self.involvement(player_id, fixture_ctx)
        return stats.poisson_at_least(k, mean)

    def p_haul(self, player_id: int, fixture_ctx: "FixtureContext") -> float:
        """P(2 or more attacking returns) — the captaincy question."""
        return self.p_returns(player_id, fixture_ctx, k=2)

    def expected_attacking_points(
        self, player_id: int, fixture_ctx: "FixtureContext"
    ) -> Tuple[float, float, float]:
        """(goal points, assist points, penalty-miss points) for this fixture.

        FPL points are linear in event counts, so the expectation is just
        lambda * points-per-event; the Poisson distribution is only needed for
        the variance and the haul probabilities above. The third term is the
        -2 for a missed penalty, which follows from the same attempt rate that
        produced the penalty goals.
        """
        b = self.project_breakdown(player_id, fixture_ctx)
        r = self.rates_by_player.get(player_id)
        pos = r.position if r is not None else scoring.MID
        miss_rate = 1.0 - self.fit_params.penalty_conversion
        return (
            scoring.goal_points(pos, b.lambda_goals),
            scoring.assist_points(b.lambda_assists),
            scoring.PENALTY_MISS * b.penalty_attempts * miss_rate,
        )

    # -- explanation --------------------------------------------------------

    def explain(self, player_id: int, fixture_ctx: Optional["FixtureContext"] = None) -> str:
        r = self.rates_by_player.get(player_id)
        if r is None:
            return "player %d: no attacking rates fitted" % player_id
        name = str(player_id)
        if self._state is not None and player_id in self._state.players:
            name = self._state.players[player_id].web_name
        lines = [
            "%s (%s, £%.1fm)" % (name, scoring.POS_NAME[r.position], r.price),
            "  blend: %s" % r.reason,
            "  xG90 %.3f (raw %.3f, baseline %.3f) | xA90 %.3f (raw %.3f, baseline %.3f)"
            % (r.xg90, r.raw_xg90, r.baseline_xg90, r.xa90, r.raw_xa90, r.baseline_xa90),
            "  finishing x%.3f" % r.finishing,
        ]
        if r.is_penalty_taker:
            lines.append("  first-choice penalties (est. %.1f taken last season, stripped "
                         "from the base xG)" % r.penalties_taken_prior)
        if fixture_ctx is not None:
            b = self.project_breakdown(player_id, fixture_ctx)
            lines.append(
                "  fixture: team lambda %.2f -> x%.3f goals / x%.3f assists, xmins %.0f"
                % (b.team_lambda, b.fixture_scale_goals, b.fixture_scale_assists,
                   b.minutes_fraction * 90.0))
            lines.append(
                "  lambda_goals %.3f (open play %.3f + penalties %.3f), lambda_assists %.3f, "
                "P(2+ returns) %.1f%%"
                % (b.lambda_goals, b.lambda_goals_open_play, b.lambda_goals_penalty,
                   b.lambda_assists,
                   100.0 * stats.poisson_at_least(2, b.expected_goal_involvement)))
        return "\n".join(lines)

    def summary(self) -> str:
        f = self.fit_params
        return (
            "AttackingModel[%s]: goal elasticity %.3f, assist elasticity %.3f, "
            "penalties %.4f/team-match (elasticity %.2f, conversion %.2f), "
            "finishing prior %.0f xG, %d players rated"
            % (f.source, f.goal_elasticity, f.assist_elasticity,
               f.penalty_rate_per_team_match, f.penalty_team_elasticity,
               f.penalty_conversion, self.finishing_prior_xg, len(self.rates_by_player))
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from gaffer.core.types import MinutesForecast
    from gaffer.data.loaders import fixtures_by_gw, load_game_state

    class _Ctx(object):
        """Stand-in for gaffer.model.xp.FixtureContext (SPEC 3.8).

        xp.py owns the real one; this carries exactly the attributes documented
        there so this module can be exercised before that lands.
        """

        def __init__(self, fixture, player, is_home, team_lambda, opponent_lambda, minutes):
            self.fixture = fixture
            self.player = player
            self.is_home = is_home
            self.team_lambda = team_lambda
            self.opponent_lambda = opponent_lambda
            self.forecast = None
            self.minutes = minutes

    cfg = Config.load()
    t0 = time.time()
    state = load_game_state(cfg, progress=False)
    print("state: %s (%.2fs)" % (state.summary(), time.time() - t0))

    model = AttackingModel(cfg)
    t0 = time.time()
    model.fit(state)
    print("fit: %.2fs -> %s" % (time.time() - t0, model.summary()))
    t0 = time.time()
    AttackingModel(cfg).fit(state)
    print("refit (cached historical fit): %.2fs" % (time.time() - t0))

    f = model.fit_params
    print("\n=== fitted parameters ===")
    print("  seasons fitted           : %s (%d player-matches, %d team-matches)"
          % (", ".join(f.seasons) or "none", f.n_player_matches, f.n_team_matches))
    print("  league avg team lambda   : %.3f (fit) / %.3f (used as reference)"
          % (f.league_avg_team_lambda, model.league_avg_lambda))
    print("  goal elasticity          : %.3f  (xG %.3f, goals %.3f; se %.3f)"
          % (f.goal_elasticity, f.goal_elasticity_xg, f.goal_elasticity_goals,
             f.goal_elasticity_se))
    print("  assist elasticity        : %.3f  (xA %.3f, assists %.3f; se %.3f)"
          % (f.assist_elasticity, f.assist_elasticity_xa, f.assist_elasticity_assists,
             f.assist_elasticity_se))
    print("  team lambda calibration  : %.3f" % f.team_lambda_calibration)
    print("  penalty xG / conversion  : %.2f / %.2f" % (f.penalty_xg, f.penalty_conversion))
    print("  penalties per team-match : %.4f (from %d misses over %d team-matches)"
          % (f.penalty_rate_per_team_match, f.n_penalty_misses, f.n_team_matches))
    print("  penalty team elasticity  : %.3f (raw %.3f, se %.3f, shrunk to prior %.1f+-%.1f)"
          % (f.penalty_team_elasticity, f.penalty_team_elasticity_raw,
             f.penalty_team_elasticity_se, PENALTY_ELASTICITY_PRIOR_MEAN,
             PENALTY_ELASTICITY_PRIOR_SD))
    print("  finishing prior strength : %.0f xG (skill sd %.4f, cov %+.5f +- %.5f over "
          "%d season pairs from %s)"
          % (model.finishing_prior_xg, model.finishing_skill_sd,
             f.finishing_covariance, f.finishing_covariance_se, f.n_finishing_pairs,
             ", ".join(f.finishing_seasons) or "none"))
    print("                             a 20-xG season gets %.0f%% weight on its own "
          "conversion ratio (league ratio %.3f)"
          % (100.0 * 20.0 / (20.0 + model.finishing_prior_xg), f.finishing_theta))
    for w in model.warnings:
        print("  warning: %s" % w)

    print("\n=== fitted position / price baselines (non-penalty per-90) ===")
    for pos in scoring.POSITIONS:
        b = model.baselines.get(pos)
        if b is None or not b.prices:
            print("  %s: none" % scoring.POS_NAME[pos])
            continue
        print("  %s" % scoring.POS_NAME[pos])
        for price, xg, xa, n in zip(b.prices, b.xg90, b.xa90, b.n_players):
            print("    £%5.2fm (n=%3d)  xG90 %.4f  xA90 %.4f" % (price, n, xg, xa))

    print("\n=== fixture scaling ===")
    for lam in (0.8, 1.0, 1.45, 2.0, 2.6):
        sg, sa = model.fixture_scale(lam)
        print("  team lambda %.2f -> goals x%.3f, assists x%.3f, penalties %.4f/match"
              % (lam, sg, sa, model.penalty_rate(lam)))

    # --- build GW1 contexts -------------------------------------------------
    gw = state.current_gw
    fixtures = fixtures_by_gw(state)[gw]
    print("\n=== GW%d: %d fixtures ===" % (gw, len(fixtures)))

    # Team lambdas stand in for TeamRatingModel, which another module owns.
    # Derived from the official per-fixture FDR so the ordering is real rather
    # than flat: an FDR-2 opponent is a soft fixture, FDR-5 is a hard one.
    league = cfg.model.league_avg_goals_per_team
    home_adv = cfg.model.home_advantage

    def lam_for(difficulty: int, is_home: bool) -> float:
        base = league * (1.0 + 0.16 * (3 - int(difficulty)))
        return base * (home_adv if is_home else 1.0 / home_adv)

    team_lambda: Dict[int, float] = {}
    ctx_by_player: Dict[int, Any] = {}
    for fx in fixtures:
        team_lambda[fx.team_h] = lam_for(fx.team_h_difficulty, True)
        team_lambda[fx.team_a] = lam_for(fx.team_a_difficulty, False)

    # Minutes stand-in: price and prior-season starts, in the spirit of SPEC 3.3.
    def xmins_for(pid: int) -> float:
        p = state.players[pid]
        if p.status in ("i", "s", "u"):
            return 0.0
        row = state.prior_season_row(pid)
        mins = float(row.get("minutes") or 0.0) if row else 0.0
        if mins > 0:
            base = min(mins / 38.0, 90.0)
        else:
            base = max(0.0, min(80.0, (p.price - 4.0) * 22.0))
        if p.status == "d":
            base *= (p.chance_of_playing_next_round or 50) / 100.0
        return base

    for fx in fixtures:
        for team, is_home in ((fx.team_h, True), (fx.team_a, False)):
            for pid, p in state.players.items():
                if p.team_id != team:
                    continue
                mf = MinutesForecast(player_id=pid, gw=gw, p_appear=0.0, p_start=0.0,
                                     p_60=0.0, xmins=xmins_for(pid))
                ctx_by_player[pid] = _Ctx(fx, p, is_home, team_lambda[team],
                                          team_lambda[fx.opponent_of(team)], mf)

    rows = []
    for pid, ctx in ctx_by_player.items():
        p = state.players[pid]
        b = model.project_breakdown(pid, ctx)
        r = model.player_rates(pid)
        rows.append({
            "name": p.web_name, "pos": scoring.POS_NAME[p.position],
            "team": state.short_name(p.team_id), "price": p.price,
            "xmins": ctx.minutes.xmins,
            "xG90": b.xg90, "xA90": b.xa90, "fin": b.finishing,
            "pen": "P" if r.is_penalty_taker else ("-" if not r.was_penalty_taker else "x"),
            "lam_g": b.lambda_goals, "lam_open": b.lambda_goals_open_play,
            "lam_pen": b.lambda_goals_penalty, "lam_a": b.lambda_assists,
            "xGI": b.expected_goal_involvement,
            "p_haul": model.p_haul(pid, ctx),
            "sd_pts": math.sqrt(model.points_variance(pid, ctx)),
        })
    df = pd.DataFrame(rows).sort_values("lam_g", ascending=False)

    pd.set_option("display.width", 200)
    print("\n=== top 40 by projected lambda_goals, GW%d ===" % gw)
    print("    pen: P = first-choice taker now, x = took them last season but not now, "
          "- = never")
    print(df.head(40).to_string(index=False, float_format=lambda v: "%.3f" % v))

    print("\n=== top 20 by projected goal involvement (goals + assists), GW%d ===" % gw)
    print(df.sort_values("xGI", ascending=False).head(20)[
        ["name", "pos", "team", "price", "xmins", "xG90", "xA90", "pen",
         "lam_g", "lam_a", "xGI", "p_haul", "sd_pts"]].to_string(
        index=False, float_format=lambda v: "%.3f" % v))

    print("\n=== biggest finishing adjustments ===")
    fin = df[df["fin"] != 1.0].sort_values("fin")
    print(pd.concat([fin.head(5), fin.tail(5)])[
        ["name", "pos", "team", "xG90", "fin", "lam_g"]].to_string(
        index=False, float_format=lambda v: "%.3f" % v))

    print("\n=== top 15 by projected lambda_assists, GW%d ===" % gw)
    print(df.sort_values("lam_a", ascending=False).head(15).to_string(
        index=False, float_format=lambda v: "%.3f" % v))

    print("\n=== penalty edge: first-choice takers, sorted by the penalty term ===")
    pens = df[df["pen"] == "P"].sort_values("lam_pen", ascending=False)
    print(pens[["name", "pos", "team", "price", "xG90", "lam_open", "lam_pen", "lam_g"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v))

    print("\n=== double-count check: takers vs their own stripped base rate ===")
    for pid, r in sorted(model.rates_by_player.items(),
                         key=lambda kv: -kv[1].penalties_taken_prior)[:6]:
        p = state.players[pid]
        row = state.prior_season_row(pid)
        raw90 = (float(row["expected_goals"]) / (float(row["minutes"]) / 90.0)) \
            if row and row.get("minutes") else float("nan")
        print("  %-14s %-3s raw xG90 %.3f -> non-pen %.3f (est %.1f pens stripped, "
              "prior taker=%s, now taker=%s)"
              % (p.web_name, state.short_name(p.team_id), raw90, r.raw_xg90,
                 r.penalties_taken_prior, r.was_penalty_taker, r.is_penalty_taker))

    print("\n=== explain ===")
    wanted = []
    for want in ("Haaland", "Saka"):
        match = [p.id for p in state.players.values() if p.web_name == want]
        if match:
            wanted.append(match[0])
    # Plus the best baseline-only player: no Premier League record at all, so his
    # rate is entirely the fitted position/price prior.
    rookies = [(model.project_breakdown(pid, ctx).lambda_goals, pid)
               for pid, ctx in ctx_by_player.items()
               if model.player_rates(pid).n90 == 0]
    if rookies:
        wanted.append(max(rookies)[1])
    for pid in wanted:
        if pid in ctx_by_player:
            print(model.explain(pid, ctx_by_player[pid]))

    print("\n=== assertions ===")
    assert not df[["lam_g", "lam_a", "xG90", "xA90", "xGI", "p_haul", "sd_pts"]].isna().any().any(), \
        "NaN in projections"
    assert (df["lam_g"] >= 0).all() and (df["lam_a"] >= 0).all(), "negative lambda"
    worst = df["lam_g"].max()
    assert worst <= MAX_LAMBDA_GOALS, "lambda_goals %.3f above the %.1f rail" % (worst, MAX_LAMBDA_GOALS)
    assert worst < 1.5, "top lambda_goals %.3f is implausibly high" % worst
    assert (df["p_haul"] >= 0).all() and (df["p_haul"] <= 1).all()
    for pid, r in model.rates_by_player.items():
        assert 1.0 - FINISHING_CAP - 1e-9 <= r.finishing <= 1.0 + FINISHING_CAP + 1e-9
    assert len(model.rates_by_player) == len(state.players)
    zero_mins = [pid for pid, c in ctx_by_player.items() if c.minutes.xmins == 0]
    for pid in zero_mins[:20]:
        lg, la = model.project(pid, ctx_by_player[pid])
        assert lg == 0.0 and la == 0.0, "0 xmins must project 0 returns"
    top10 = df.head(10)
    n_prem = int(((top10["price"] >= 7.0) & top10["pos"].isin(["FWD", "MID"])).sum())
    assert n_prem >= 7, "top 10 by lambda_goals should be premium attackers, got %d" % n_prem
    print("  no NaN / all lambdas >= 0 / max lambda_goals %.3f < %.1f: OK"
          % (worst, MAX_LAMBDA_GOALS))
    print("  %d of the top 10 are >= £7.0m MID/FWD: OK" % n_prem)
    print("  0-xmins players project 0: OK")
    print("  finishing multipliers all inside +-%.0f%%: OK" % (100 * FINISHING_CAP))

    # --- mid-season blending -----------------------------------------------
    # Pre-season the current-season sources are empty, so the three-way blend and
    # the recent-form window never run above. Synthesise a GW10 state and check
    # the weights and the effective sample size against hand arithmetic.
    print("\n=== mid-season blend (synthetic GW10 history) ===")
    from dataclasses import replace as _replace

    def _hist(rows: List[Dict[str, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        frame["round"] = frame["gw"]
        return frame

    steady = [{"gw": g, "minutes": 90.0, "expected_goals": 0.30, "expected_assists": 0.10,
               "goals_scored": 0.0, "penalties_missed": 0.0} for g in range(1, 10)]
    # Cold for three gameweeks, hot for the last six: recent form must dominate.
    surge = ([{"gw": g, "minutes": 90.0, "expected_goals": 0.05, "expected_assists": 0.02,
               "goals_scored": 0.0, "penalties_missed": 0.0} for g in range(1, 4)]
             + [{"gw": g, "minutes": 90.0, "expected_goals": 0.60, "expected_assists": 0.20,
                 "goals_scored": 1.0, "penalties_missed": 0.0} for g in range(4, 10)])

    steady_id = [p.id for p in state.players.values() if p.web_name == "Watkins"][0]
    surge_id = [p.id for p in state.players.values() if p.web_name == "Gyökeres"][0]
    # A player with no prior-season record at all: only the current season counts.
    rookie_id = next(pid for pid in state.players
                     if state.prior_season_row(pid) is None
                     and state.latest_season_row(pid) is None)

    histories = dict(state.player_history)
    histories[steady_id] = _hist(steady)
    histories[surge_id] = _hist(surge)
    histories[rookie_id] = _hist(steady)
    mid_state = _replace(
        state, current_gw=10, finished_gws=list(range(1, 10)),
        elements_are_prior_season=False, player_history=histories,
    )
    mid = AttackingModel(cfg)
    mid.fit(mid_state)

    mc = cfg.model
    for pid, label in ((steady_id, "steady"), (surge_id, "surge"),
                       (rookie_id, "no prior season")):
        r = mid.player_rates(pid)
        prior_row = mid_state.prior_season_row(pid)
        n90_prior = (float(prior_row["minutes"]) / 90.0) if prior_row else 0.0
        w = r.source_weights
        print("  %-16s %-14s weights=%s n90=%.1f raw_xG90=%.3f -> shrunk %.3f"
              % (state.players[pid].web_name, "(%s)" % label,
                 {k: round(v, 3) for k, v in w.items()}, r.n90, r.raw_xg90, r.xg90))
        expect_w = {}
        if n90_prior > 0:
            expect_w["prior"] = mc.weight_prior_season
        expect_w["season"] = mc.weight_season_to_date
        expect_w["recent"] = mc.weight_recent_form
        total = sum(expect_w.values())
        expect_w = {k: v / total for k, v in expect_w.items()}
        assert set(w) == set(expect_w), (w, expect_w)
        for k in w:
            assert abs(w[k] - expect_w[k]) < 1e-9, (k, w[k], expect_w[k])
        expect_n90 = expect_w.get("prior", 0.0) * n90_prior + (
            expect_w["season"] + expect_w["recent"]) * 9.0
        assert abs(r.n90 - expect_n90) < 1e-6, (r.n90, expect_n90)

    print("  weights renormalise with a source missing, recent-form window is the "
          "last %d gameweeks: OK" % mc.recent_form_gws)
    # The surge player's blend must sit above his season-to-date average, because
    # the last six gameweeks carry the largest weight.
    r_surge = mid.player_rates(surge_id)
    surge_season = float(mid_state.history(surge_id)["expected_goals"].sum()) / 9.0
    assert r_surge.raw_xg90 > surge_season, (r_surge.raw_xg90, surge_season)
    print("  surge player: season-to-date xG90 %.3f -> blended %.3f (recent form pulls "
          "it up): OK" % (surge_season, r_surge.raw_xg90))
    r_steady = mid.player_rates(steady_id)
    assert r_steady.raw_xg90 > 0 and r_steady.n90 > 0
    assert mid.rates(steady_id) == (r_steady.xg90, r_steady.xa90)

    print("\n=== league coherence ===")
    total_90s = sum(c.minutes.xmins for c in ctx_by_player.values()) / 90.0
    print("  stand-in minutes cover %.1f of the 220 player-90s GW%d actually has "
          "(%.0f%%), so the totals below are scaled down by roughly that much"
          % (total_90s, gw, 100.0 * total_90s / 220.0))
    print("  sum of projected lambda_goals over all GW%d players: %.1f "
          "(20 teams x lambda ~ %.1f expected; %.1f after the minutes shortfall)"
          % (gw, df["lam_g"].sum(), sum(team_lambda.values()),
             sum(team_lambda.values()) * total_90s / 220.0))
    print("  share of projected goals from penalties: %.1f%% (league share of goals "
          "that are penalties: %.1f%%)"
          % (100.0 * df["lam_pen"].sum() / max(df["lam_g"].sum(), 1e-9),
             100.0 * model.fit_params.penalty_rate_per_team_match
             * model.fit_params.penalty_conversion / cfg.model.league_avg_goals_per_team))
    print("\nfit written to %s" % FIT_PATH)
