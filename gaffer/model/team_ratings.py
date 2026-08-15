"""Team strength and match forecasting — the backbone of every projection.

A Dixon-Coles style bivariate Poisson: each club carries an *attack* and a
*defence* multiplier on the league baseline, plus one global home advantage,
fitted by weighted maximum likelihood with exponential time decay.

    lambda_home = league_avg * attack[home] * defence[away] * home_advantage
    lambda_away = league_avg * attack[away] * defence[home]

Two conventions that are easy to get backwards:

* ``defence`` is a multiplier on the goals a team **concedes**, so *higher is
  worse*. Liverpool's defence lands near 0.8, a promoted side's near 1.25.
* ``league_avg`` is therefore the *away* baseline, because the home advantage
  sits only on the home side of the pair (that is the formula in SPEC.md).
  The mean lambda over a balanced fixture list is ``league_avg * (1 + ha) / 2``
  and is reported separately as ``mean_lambda``.

scipy is not installed, so the fit is a hand-rolled coordinate ascent. Every
parameter block has a closed-form conditional maximiser under the weighted
Poisson likelihood, which makes the ascent monotone and lets us verify
convergence by watching the log-likelihood rather than trusting an optimiser.
The Dixon-Coles low-score correction ``rho`` does not have a closed form and is
fitted afterwards by a grid search on the correction term alone (it is
orthogonal to the marginals by construction).

Three season-start problems are handled explicitly:

1. **Promoted clubs** (COV, HUL, IPS for 2026/27) have no or stale top-flight
   history. They are shrunk hard onto ``promoted_attack_prior`` /
   ``promoted_defence_prior`` until they have real 2026/27 matches, and carry a
   near-zero ``confidence`` so downstream code can widen its uncertainty.
2. **Prior blending**: at GW1 the ratings are 100% derived from prior seasons.
   Prior-season matches are down-weighted as current-season matches accumulate,
   fully decaying over ``ratings_prior_decay_matches``.
3. **Odds**: football-data carries 1X2 and over/under 2.5 prices for every past
   match, and the market beats any home-grown rating. Odds are blended in at
   ``odds_blend_weight`` when available; future fixtures have none, so the model
   must and does stand alone. ``use_odds`` toggles the whole path for backtests.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import stats
from gaffer.core.config import Config
from gaffer.core.types import Fixture, MatchForecast, TeamStrength
from gaffer.data import history as hist

if TYPE_CHECKING:  # import only for typing: loaders must never import the model
    from gaffer.data.loaders import GameState

log = logging.getLogger(__name__)

# Lambdas outside this band mean the fit has gone wrong; SPEC requires a warning.
LAMBDA_MIN, LAMBDA_MAX = 0.2, 4.0
# Score-matrix truncation. P(4+ goals) under lambda 2.5 is already 24%; ten is
# far into the tail and keeps the normalisation error under 1e-6.
MAX_GOALS = 10
# A full season's worth of matches, used as the unit for evidence mass.
SEASON_MATCHES = 38


def _utcnow() -> datetime:
    """Today at midnight UTC.

    The decay weights are measured from this instant, so using the wall clock
    would make two fits a second apart differ in the twelfth decimal place and
    break the determinism requirement. Day granularity is plenty: a day of drift
    moves a 180-day half-life weight by 0.4%.
    """
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Fit result containers
# ---------------------------------------------------------------------------


@dataclass
class RatingFit:
    """Output of one weighted maximum-likelihood fit, keyed by FPL short name.

    ``mass`` is the sum of time-decay weights behind each club — "matches at
    today's relevance" — and is what confidence and shrinkage key off. It is
    deliberately not the raw match count: 38 matches from two seasons ago carry
    far less information about this weekend than 38 from last spring.
    """

    league_avg: float
    home_advantage: float
    attack: Dict[str, float] = field(default_factory=dict)
    defence: Dict[str, float] = field(default_factory=dict)
    mass: Dict[str, float] = field(default_factory=dict)
    matches: Dict[str, int] = field(default_factory=dict)
    rho: float = 0.0
    n_matches: int = 0
    total_weight: float = 0.0
    iterations: int = 0
    converged: bool = False
    max_delta: float = 0.0
    loglik: float = float("nan")
    loglik_increasing: bool = True

    def teams(self) -> List[str]:
        return sorted(self.attack)


# ---------------------------------------------------------------------------
# The core weighted-Poisson coordinate ascent
# ---------------------------------------------------------------------------


def _bincount(idx: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(idx, weights=weights, minlength=n)[:n]


def _fit_multiplicative(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_count: np.ndarray,
    away_count: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    ridge_matches: float = 1.0,
    max_iter: int = 400,
    tol: float = 1e-11,
) -> Dict[str, Any]:
    """Weighted MLE of (baseline, home advantage, attack[], defence[]).

    Each block below is the exact conditional maximiser of the weighted Poisson
    log-likelihood holding the others fixed, so the ascent cannot decrease the
    likelihood and converges without a step size.

    ``ridge_matches`` adds a pseudo-observation at the league average to both
    the numerator and denominator of every team update. It is worth one match
    and exists purely to keep a club that has scored zero goals in its (few,
    heavily discounted) matches from driving its attack multiplier to exactly
    zero, which would make the log-likelihood undefined.
    """
    n = int(n_teams)
    atk = np.ones(n, dtype=float)
    dfn = np.ones(n, dtype=float)
    w = weights.astype(float)
    gh = home_count.astype(float)
    ga = away_count.astype(float)

    total_w = float(w.sum())
    if total_w <= 0 or len(gh) == 0:
        return {
            "league_avg": 0.0, "home_advantage": 1.0, "attack": atk, "defence": dfn,
            "iterations": 0, "converged": False, "max_delta": 0.0,
            "loglik": float("nan"), "worst_loglik_drop": 0.0, "loglik_increasing": True,
        }

    # Sensible starting point: overall mean and the raw home/away goal ratio.
    lg = float((w * (gh + ga)).sum() / (2.0 * total_w))
    lg = max(lg, 1e-6)
    ha_num = float((w * gh).sum())
    ha_den = float((w * ga).sum())
    ha = ha_num / ha_den if ha_den > 0 else 1.0
    # Split the raw ratio so the starting baseline sits at the overall mean.
    lg = lg * 2.0 / (1.0 + ha)

    w_gh = w * gh
    w_ga = w * ga
    scored = _bincount(home_idx, w_gh, n) + _bincount(away_idx, w_ga, n)
    conceded = _bincount(home_idx, w_ga, n) + _bincount(away_idx, w_gh, n)

    def loglik(lg_: float, ha_: float, a_: np.ndarray, d_: np.ndarray) -> float:
        lam_h = lg_ * a_[home_idx] * d_[away_idx] * ha_
        lam_a = lg_ * a_[away_idx] * d_[home_idx]
        lam_h = np.maximum(lam_h, 1e-12)
        lam_a = np.maximum(lam_a, 1e-12)
        # Constant -log(y!) dropped: it does not depend on the parameters.
        return float((w * (gh * np.log(lam_h) - lam_h + ga * np.log(lam_a) - lam_a)).sum())

    prev_ll = loglik(lg, ha, atk, dfn)
    worst_drop = 0.0
    iterations = 0
    max_delta = float("inf")
    # Held fixed across sweeps so the penalised objective it implies is stable.
    ridge = ridge_matches * lg  # in goals, so it is commensurate with the counts

    for iterations in range(1, max_iter + 1):
        prev = np.concatenate([atk, dfn, [lg, ha]])

        # attack: goals scored / expected goals scored at attack == 1
        den = (
            _bincount(home_idx, w * lg * dfn[away_idx] * ha, n)
            + _bincount(away_idx, w * lg * dfn[home_idx], n)
        )
        atk = (scored + ridge) / (den + ridge)

        # defence: goals conceded / expected goals conceded at defence == 1
        den = (
            _bincount(home_idx, w * lg * atk[away_idx], n)
            + _bincount(away_idx, w * lg * atk[home_idx] * ha, n)
        )
        dfn = (conceded + ridge) / (den + ridge)

        # home advantage: total home goals / total home goals at ha == 1
        den_ha = float((w * lg * atk[home_idx] * dfn[away_idx]).sum())
        if den_ha > 0:
            ha = ha_num / den_ha

        # baseline: total goals / total goals at league_avg == 1
        den_lg = float(
            (w * (atk[home_idx] * dfn[away_idx] * ha + atk[away_idx] * dfn[home_idx])).sum()
        )
        if den_lg > 0:
            lg = float((w * (gh + ga)).sum()) / den_lg

        # Identifiability: attack and defence are only defined up to a constant
        # factor that the baseline can absorb. Pin both geometric means to 1 so
        # "1.0 == league average" is literally true.
        g_atk = math.exp(float(np.log(np.maximum(atk, 1e-12)).mean()))
        atk = atk / g_atk
        lg *= g_atk
        g_dfn = math.exp(float(np.log(np.maximum(dfn, 1e-12)).mean()))
        dfn = dfn / g_dfn
        lg *= g_dfn

        cur = np.concatenate([atk, dfn, [lg, ha]])
        max_delta = float(np.max(np.abs(np.log(np.maximum(cur, 1e-12)) - np.log(np.maximum(prev, 1e-12)))))
        ll = loglik(lg, ha, atk, dfn)
        worst_drop = max(worst_drop, prev_ll - ll)
        prev_ll = ll
        if max_delta < tol:
            break

    return {
        "league_avg": lg,
        "home_advantage": ha,
        "attack": atk,
        "defence": dfn,
        "iterations": iterations,
        "converged": bool(max_delta < 1e-8),
        "max_delta": max_delta,
        "loglik": prev_ll,
        "worst_loglik_drop": worst_drop,
        # Each block is a conditional maximiser, so the ascent is monotone up to
        # the ridge pseudo-observation, which perturbs the plain likelihood by a
        # tiny amount. Anything larger than that means the ascent is broken.
        "loglik_increasing": bool(worst_drop <= 1e-3 * max(1.0, abs(prev_ll))),
    }


def _fit_rho(
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    gh: np.ndarray,
    ga: np.ndarray,
    w: np.ndarray,
    grid: Optional[Sequence[float]] = None,
) -> float:
    """Grid-search the Dixon-Coles low-score correction.

    tau only touches the four scorelines 0-0, 0-1, 1-0, 1-1, and its product
    form leaves the Poisson marginals untouched, so it can be fitted after the
    marginals with no loss. Negative rho means 0-0 and 1-1 happen more often
    than independence implies, which is the well-documented football result.
    """
    if grid is None:
        grid = [(-0.25 + 0.005 * i) for i in range(0, 81)]  # -0.25 .. 0.15
    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    if not (m00.any() or m01.any() or m10.any() or m11.any()):
        return 0.0

    best_rho, best_ll = 0.0, -np.inf
    for rho in grid:
        t00 = 1.0 - lam_h * lam_a * rho
        t01 = 1.0 + lam_h * rho
        t10 = 1.0 + lam_a * rho
        t11 = 1.0 - rho
        if t11 <= 0:
            continue
        if (t00[m00] <= 0).any() or (t01[m01] <= 0).any() or (t10[m10] <= 0).any():
            continue
        ll = 0.0
        ll += float((w[m00] * np.log(t00[m00])).sum())
        ll += float((w[m01] * np.log(t01[m01])).sum())
        ll += float((w[m10] * np.log(t10[m10])).sum())
        ll += float((w[m11] * np.log(t11)).sum())
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)


# ---------------------------------------------------------------------------
# Odds -> lambdas
# ---------------------------------------------------------------------------


def total_goals_from_over_under(p_over_25: float) -> float:
    """Total-goals line implied by a de-vigged P(total > 2.5).

    The sum of two independent Poissons is Poisson in the total, so the split
    between the sides is irrelevant here and the solve is a scalar bisection on
    P(N >= 3) = 1 - e^-T (1 + T + T^2/2).
    """
    p = min(max(float(p_over_25), 1e-4), 1.0 - 1e-4)
    lo, hi = 0.2, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p_over = 1.0 - math.exp(-mid) * (1.0 + mid + mid * mid / 2.0)
        if p_over < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _dc_outcome_probs_vec(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: float, max_goals: int = MAX_GOALS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised twin of ``stats.match_outcome_probs`` (verified identical).

    The scalar helper is fine for a handful of fixtures but a backtest solves
    the odds inversion for hundreds of matches inside a bisection loop, which is
    ~200x too slow in pure Python.
    """
    k = np.arange(max_goals + 1)
    lgam = np.array([math.lgamma(int(i) + 1) for i in k])
    lh = np.maximum(lam_h, 1e-12)[:, None]
    la = np.maximum(lam_a, 1e-12)[:, None]
    ph = np.exp(-lh + k[None, :] * np.log(lh) - lgam[None, :])
    pa = np.exp(-la + k[None, :] * np.log(la) - lgam[None, :])

    cdf_a = np.cumsum(pa, axis=1)
    # P(home > away) = sum_x ph[x] * P(away <= x-1)
    home = (ph[:, 1:] * cdf_a[:, :-1]).sum(axis=1)
    draw = (ph * pa).sum(axis=1)
    total = ph.sum(axis=1) * pa.sum(axis=1)

    lhv = np.maximum(lam_h, 1e-12)
    lav = np.maximum(lam_a, 1e-12)
    t00 = 1.0 - lhv * lav * rho - 1.0
    t01 = 1.0 + lhv * rho - 1.0
    t10 = 1.0 + lav * rho - 1.0
    t11 = -rho
    c00 = ph[:, 0] * pa[:, 0] * t00
    c01 = ph[:, 0] * pa[:, 1] * t01
    c10 = ph[:, 1] * pa[:, 0] * t10
    c11 = ph[:, 1] * pa[:, 1] * t11

    total = total + c00 + c01 + c10 + c11
    home = home + c10
    draw = draw + c00 + c11
    total = np.maximum(total, 1e-12)
    home = home / total
    draw = draw / total
    away = np.maximum(1.0 - home - draw, 0.0)
    return home, draw, away


def lambdas_from_odds_vec(
    p_home: np.ndarray,
    p_draw: np.ndarray,
    p_away: np.ndarray,
    total_goals: np.ndarray,
    rho: float = -0.06,
    iterations: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised twin of ``stats.lambdas_from_odds`` — same bisection, same result."""
    n = len(p_home)
    lo = np.full(n, -3.0)
    hi = np.full(n, 3.0)
    target = np.asarray(p_home, dtype=float) - np.asarray(p_away, dtype=float)
    tot = np.asarray(total_goals, dtype=float)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lam_h = np.maximum(0.05, (tot + mid) / 2.0)
        lam_a = np.maximum(0.05, (tot - mid) / 2.0)
        ph, _, pa = _dc_outcome_probs_vec(lam_h, lam_a, rho)
        go_up = (ph - pa) < target
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    sup = 0.5 * (lo + hi)
    return np.maximum(0.05, (tot + sup) / 2.0), np.maximum(0.05, (tot - sup) / 2.0)


def market_lambdas(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    odds_over25: Optional[float] = None,
    odds_under25: Optional[float] = None,
    rho: float = -0.06,
    default_total: float = 2.7,
) -> Optional[Tuple[float, float]]:
    """Bookmaker-implied (lambda_home, lambda_away) for a single match."""
    out = market_lambdas_frame(
        pd.DataFrame(
            {
                "odds_home": [odds_home],
                "odds_draw": [odds_draw],
                "odds_away": [odds_away],
                "odds_over25": [odds_over25],
                "odds_under25": [odds_under25],
            }
        ),
        rho=rho,
        default_total=default_total,
    )
    if out is None or len(out[0]) == 0 or not np.isfinite(out[0][0]):
        return None
    return float(out[0][0]), float(out[1][0])


def market_lambdas_frame(
    df: pd.DataFrame, rho: float = -0.06, default_total: float = 2.7
) -> Tuple[np.ndarray, np.ndarray]:
    """(lambda_home, lambda_away) arrays for a frame of football-data odds rows.

    Rows without a usable 1X2 price come back as NaN rather than a guess.
    """
    oh = pd.to_numeric(df.get("odds_home"), errors="coerce").to_numpy(dtype=float)
    od = pd.to_numeric(df.get("odds_draw"), errors="coerce").to_numpy(dtype=float)
    oa = pd.to_numeric(df.get("odds_away"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(oh) & np.isfinite(od) & np.isfinite(oa) & (oh > 1) & (od > 1) & (oa > 1)

    lam_h = np.full(len(df), np.nan)
    lam_a = np.full(len(df), np.nan)
    if not ok.any():
        return lam_h, lam_a

    probs = np.array(
        [stats.remove_vig([h, d, a]) for h, d, a in zip(oh[ok], od[ok], oa[ok])], dtype=float
    )
    ph, _pd_, pa = probs[:, 0], probs[:, 1], probs[:, 2]

    totals = np.full(int(ok.sum()), float(default_total))
    if "odds_over25" in df.columns and "odds_under25" in df.columns:
        ov = pd.to_numeric(df["odds_over25"], errors="coerce").to_numpy(dtype=float)[ok]
        un = pd.to_numeric(df["odds_under25"], errors="coerce").to_numpy(dtype=float)[ok]
        good = np.isfinite(ov) & np.isfinite(un) & (ov > 1) & (un > 1)
        for i in np.nonzero(good)[0]:
            p_over = stats.remove_vig([ov[i], un[i]])[0]
            totals[i] = total_goals_from_over_under(p_over)

    h, a = lambdas_from_odds_vec(ph, _pd_, pa, totals, rho=rho)
    lam_h[ok] = h
    lam_a[ok] = a
    return lam_h, lam_a


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class TeamRatingModel:
    """Fit team attack/defence ratings and forecast every fixture."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.mc = self.config.model
        self.use_odds: bool = False  # backtests flip this on; no odds exist for the future
        self.state: Optional["GameState"] = None

        self.league_avg: float = float(self.mc.league_avg_goals_per_team)
        self.home_adv: float = float(self.mc.home_advantage)
        self.rho: float = float(self.mc.dixon_coles_rho)
        self.mean_lambda: float = float("nan")

        self.goal_fit: Optional[RatingFit] = None
        self.sot_fit: Optional[RatingFit] = None
        self._strength: Dict[int, TeamStrength] = {}
        self._sot: Dict[int, Dict[str, float]] = {}
        self.sot_league_avg: float = float("nan")
        self.sot_home_adv: float = 1.0

        self.persistence: Dict[str, float] = {}
        self.prior_strength_matches: float = 0.0
        # How much a promoted club's pre-relegation top-flight record counts.
        # Derived in _fit_persistence from the measured one-season retention.
        self.promoted_history_discount: float = 0.0
        self.prior_season_weight: float = 1.0
        self.current_matches_per_team: float = 0.0
        self.fit_seasons: List[str] = []
        self.dropped_future_results: int = 0
        self.as_of: Optional[datetime] = None
        self.warnings: List[str] = []
        self.diagnostics: Dict[str, Any] = {}

        self._mass_current: Dict[str, float] = {}
        self._odds_index: Dict[Tuple[int, int], List[Tuple[Optional[datetime], float, float]]] = {}
        self._odds_by_fixture: Dict[int, Tuple[float, float]] = {}
        self._forecast_cache: Dict[Tuple[int, bool], MatchForecast] = {}
        self.odds_total_calibration: float = float("nan")
        self.odds_supremacy_bias: float = float("nan")
        self.odds_calibration_n: int = 0

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        results: Optional[pd.DataFrame] = None,
        state: Optional["GameState"] = None,
        seasons: Optional[Sequence[str]] = None,
        as_of: Optional[datetime] = None,
        include_state_results: bool = True,
    ) -> None:
        """Fit ratings from historical results plus any played current-season matches.

        ``results`` is a football-data frame from ``history.load_football_data``
        (``home_short``/``away_short``/``home_goals``/``away_goals`` and,
        optionally, shots-on-target and odds columns). Passing None loads the
        three seasons preceding ``config.season``.
        """
        if state is None:
            raise ValueError("TeamRatingModel.fit needs a GameState to map clubs to FPL ids")
        self.state = state
        self.warnings = []
        self.as_of = as_of or _utcnow()

        if results is None:
            seasons = seasons or [hist.prior_season(self.config.season, k) for k in (3, 2, 1)]
            results = hist.load_football_data(list(seasons))
        played = _state_results(state)
        frames = [results]
        if include_state_results and not played.empty:
            frames.append(played)
        allres = pd.concat(frames, ignore_index=True, sort=False)
        allres = allres[allres["home_goals"].notna() & allres["away_goals"].notna()].copy()
        allres = allres[allres["home_short"].notna() & allres["away_short"].notna()]

        # Leakage guard. A backtest replays gameweek N with as_of set to its
        # deadline and hands over a results frame that runs to the end of the
        # season; without this every "forecast" would be fitted on its own
        # answer. Time-decay weights alone would not catch it — a future match
        # has a negative age, and 0.5 ** negative is a weight above 1.
        days_ago = _days_ago(allres["date"], self.as_of, clamp=False)
        future = days_ago < 0.0
        self.dropped_future_results = int(future.sum())
        if self.dropped_future_results:
            log.info(
                "ignoring %d result(s) dated after as_of=%s",
                self.dropped_future_results, self.as_of,
            )
        allres = allres[~future].copy()
        days_ago = days_ago[~future]
        if allres.empty:
            raise ValueError("no usable match results to fit on")

        is_current = (allres.get("season") == self.config.season).to_numpy()
        self.current_matches_per_team = (
            2.0 * float(is_current.sum()) / max(len(state.teams), 1)
        )
        # Prior-season decay: at GW1 the ratings are 100% prior-derived; the
        # prior's weight falls away over ratings_prior_decay_matches of the new
        # season. Combined with the time decay this is the whole of requirement 4.
        decay_matches = max(float(self.mc.ratings_prior_decay_matches), 1e-6)
        self.prior_season_weight = float(self.mc.ratings_prior_weight_gw1) * (
            decay_matches / (decay_matches + self.current_matches_per_team)
        )

        self.fit_seasons = sorted(set(str(s) for s in allres.get("season", pd.Series(dtype=str))))
        decay = np.power(0.5, days_ago / max(float(self.mc.ratings_half_life_days), 1e-6))
        weights = decay * np.where(is_current, 1.0, self.prior_season_weight)
        allres = allres.assign(_w=weights, _days_ago=days_ago, _is_current=is_current)

        shorts = sorted(set(allres["home_short"]) | set(allres["away_short"]))
        index = {s: i for i, s in enumerate(shorts)}
        hi = allres["home_short"].map(index).to_numpy(dtype=int)
        ai = allres["away_short"].map(index).to_numpy(dtype=int)
        gh = allres["home_goals"].to_numpy(dtype=float)
        ga = allres["away_goals"].to_numpy(dtype=float)
        w = allres["_w"].to_numpy(dtype=float)

        raw = _fit_multiplicative(hi, ai, gh, ga, w, len(shorts))
        mass = _bincount(hi, w, len(shorts)) + _bincount(ai, w, len(shorts))
        counts = _bincount(hi, np.ones_like(w), len(shorts)) + _bincount(ai, np.ones_like(w), len(shorts))
        w_cur = w * is_current
        mass_cur = _bincount(hi, w_cur, len(shorts)) + _bincount(ai, w_cur, len(shorts))
        self._mass_current = {s: float(mass_cur[i]) for s, i in index.items()}

        lam_h = raw["league_avg"] * raw["attack"][hi] * raw["defence"][ai] * raw["home_advantage"]
        lam_a = raw["league_avg"] * raw["attack"][ai] * raw["defence"][hi]
        rho = _fit_rho(lam_h, lam_a, gh, ga, w)

        self.goal_fit = RatingFit(
            league_avg=raw["league_avg"],
            home_advantage=raw["home_advantage"],
            attack={s: float(raw["attack"][i]) for s, i in index.items()},
            defence={s: float(raw["defence"][i]) for s, i in index.items()},
            mass={s: float(mass[i]) for s, i in index.items()},
            matches={s: int(counts[i]) for s, i in index.items()},
            rho=rho,
            n_matches=len(allres),
            total_weight=float(w.sum()),
            iterations=raw["iterations"],
            converged=raw["converged"],
            max_delta=raw["max_delta"],
            loglik=raw["loglik"],
            loglik_increasing=raw["loglik_increasing"],
        )
        if not self.goal_fit.converged:
            self.warnings.append(
                "goal-rating fit did not converge: max_delta=%.3g after %d iterations"
                % (self.goal_fit.max_delta, self.goal_fit.iterations)
            )
        if not self.goal_fit.loglik_increasing:
            self.warnings.append("goal-rating log-likelihood was not monotone increasing")

        self.rho = rho
        # Persistence is measured on the same date-filtered frame, so a backtest
        # cannot see a future season through this path either.
        self._fit_persistence(allres)
        self._fit_sot(allres, index)
        self._build_strengths(state)
        self._index_odds(allres, state)
        self._forecast_cache = {}

        self.diagnostics = {
            "n_matches": self.goal_fit.n_matches,
            "total_weight": round(self.goal_fit.total_weight, 3),
            "iterations": self.goal_fit.iterations,
            "converged": self.goal_fit.converged,
            "loglik": round(self.goal_fit.loglik, 4),
            "worst_loglik_drop": float("%.3g" % raw["worst_loglik_drop"]),
            "seasons": self.fit_seasons,
            "dropped_future_results": self.dropped_future_results,
            "prior_season_weight": round(self.prior_season_weight, 4),
            "current_matches_per_team": round(self.current_matches_per_team, 2),
            "league_avg_away_baseline": round(self.league_avg, 4),
            "home_advantage": round(self.home_adv, 4),
            "mean_lambda": round(self.mean_lambda, 4),
            "rho": round(self.rho, 4),
            "persistence": {k: round(v, 4) for k, v in self.persistence.items()},
            "prior_strength_matches": round(self.prior_strength_matches, 3),
            "promoted_history_discount": round(self.promoted_history_discount, 4),
            "sot_league_avg": round(self.sot_league_avg, 4),
            "sot_home_advantage": round(self.sot_home_adv, 4),
            "odds_rows_indexed": sum(len(v) for v in self._odds_index.values()),
            "odds_calibration_n": self.odds_calibration_n,
            "odds_total_calibration": round(self.odds_total_calibration, 4),
            "odds_supremacy_bias": round(self.odds_supremacy_bias, 4),
        }
        for msg in self.warnings:
            log.warning("team_ratings: %s", msg)

    # -- season-to-season persistence, which sets the shrinkage strength ----

    def _fit_persistence(self, results: pd.DataFrame) -> None:
        """Measure how much of a club's rating survives into the next season.

        Regressing next season's log-attack on this season's (through the
        origin, both being normalised to geometric mean 1) gives the slope that
        minimises out-of-sample error — that is exactly the optimal shrinkage
        factor, so there is no need to invent a prior strength: it falls out of
        the data.
        """
        self.persistence = {}
        self.prior_strength_matches = 0.0
        if "season" not in results.columns:
            return
        per_season: Dict[str, RatingFit] = {}
        for season, grp in results.groupby("season"):
            grp = grp[grp["home_goals"].notna() & grp["away_goals"].notna()]
            if len(grp) < 200:
                continue
            per_season[str(season)] = _fit_season(grp)
        seasons = sorted(per_season)
        pairs_a: List[Tuple[float, float]] = []
        pairs_d: List[Tuple[float, float]] = []
        for prev, nxt in zip(seasons, seasons[1:]):
            f0, f1 = per_season[prev], per_season[nxt]
            common = sorted(set(f0.attack) & set(f1.attack))
            for t in common:
                pairs_a.append((math.log(f0.attack[t]), math.log(f1.attack[t])))
                pairs_d.append((math.log(f0.defence[t]), math.log(f1.defence[t])))
        if len(pairs_a) < 8:
            self.warnings.append(
                "not enough consecutive-season overlap to fit rating persistence; "
                "ratings used unshrunk"
            )
            return

        def slope(pairs: List[Tuple[float, float]]) -> float:
            sxx = sum(x * x for x, _ in pairs)
            sxy = sum(x * y for x, y in pairs)
            return sxy / sxx if sxx > 0 else 1.0

        beta_a = slope(pairs_a)
        beta_d = slope(pairs_d)
        beta = max(0.30, min(1.0, 0.5 * (beta_a + beta_d)))
        self.persistence = {
            "attack": beta_a, "defence": beta_d, "beta": beta, "n_pairs": float(len(pairs_a))
        }
        # The regression measured a full season of evidence viewed one season
        # later; convert its shrinkage factor into a prior strength in the same
        # decayed-mass units the ratings use.
        mass_one_season_old = SEASON_MATCHES * 0.5 ** (365.0 / max(self.mc.ratings_half_life_days, 1e-6))
        self.prior_strength_matches = (
            mass_one_season_old * (1.0 - beta) / beta if beta > 0 else 0.0
        )
        # A promoted club's old top-flight form has crossed one extra season
        # boundary *and* a division. Requiring its effective retention to be
        # beta^2 rather than beta, and inverting m/(m+k), scales its evidence
        # mass by beta/(1+beta) — a shrink derived from the data rather than
        # picked. For Ipswich (2024/25 in the PL) that turns 38 matches into
        # roughly one match-equivalent of current evidence.
        self.promoted_history_discount = beta / (1.0 + beta)

    # -- shots on target -----------------------------------------------------

    def _fit_sot(self, allres: pd.DataFrame, index: Dict[str, int]) -> None:
        """Same multiplicative fit on shots on target, for the saves model."""
        sub = allres[allres["home_sot"].notna() & allres["away_sot"].notna()]
        if len(sub) < 100:
            self.sot_fit = None
            self.warnings.append("too few shots-on-target rows to fit SOT ratings")
            return
        hi = sub["home_short"].map(index).to_numpy(dtype=int)
        ai = sub["away_short"].map(index).to_numpy(dtype=int)
        sh = sub["home_sot"].to_numpy(dtype=float)
        sa = sub["away_sot"].to_numpy(dtype=float)
        w = sub["_w"].to_numpy(dtype=float)
        raw = _fit_multiplicative(hi, ai, sh, sa, w, len(index))
        mass = _bincount(hi, w, len(index)) + _bincount(ai, w, len(index))
        counts = _bincount(hi, np.ones_like(w), len(index)) + _bincount(ai, np.ones_like(w), len(index))
        self.sot_fit = RatingFit(
            league_avg=raw["league_avg"],
            home_advantage=raw["home_advantage"],
            attack={s: float(raw["attack"][i]) for s, i in index.items()},
            defence={s: float(raw["defence"][i]) for s, i in index.items()},
            mass={s: float(mass[i]) for s, i in index.items()},
            matches={s: int(counts[i]) for s, i in index.items()},
            n_matches=len(sub),
            total_weight=float(w.sum()),
            iterations=raw["iterations"],
            converged=raw["converged"],
            max_delta=raw["max_delta"],
            loglik=raw["loglik"],
            loglik_increasing=raw["loglik_increasing"],
        )
        if not raw["converged"]:
            self.warnings.append("SOT fit did not converge (max_delta=%.3g)" % raw["max_delta"])

    # -- assembling the current league's ratings -----------------------------

    def _build_strengths(self, state: "GameState") -> None:
        assert self.goal_fit is not None
        fit = self.goal_fit
        promoted = set(getattr(state, "promoted_team_ids", []) or [])
        atk_prior = float(self.mc.promoted_attack_prior)
        def_prior = float(self.mc.promoted_defence_prior)
        k_cont = float(self.prior_strength_matches)

        raw_atk: Dict[int, float] = {}
        raw_dfn: Dict[int, float] = {}
        conf: Dict[int, float] = {}
        matches: Dict[int, int] = {}

        for tid, team in state.teams.items():
            short = team.short_name
            mass = float(fit.mass.get(short, 0.0))
            n = int(fit.matches.get(short, 0))
            a_fit = fit.attack.get(short)
            d_fit = fit.defence.get(short)
            is_promoted = tid in promoted or team.promoted
            if mass <= 0 and not is_promoted:
                # A club in this season's league with no results anywhere in the
                # fit window has just come up, whatever the promoted flag says.
                # The promoted prior is a far better answer than league average.
                self.warnings.append(
                    "%s has no results in the fit window; treated as promoted" % short
                )
                is_promoted = True

            if is_promoted:
                # SPEC: shrink hard onto the promoted prior until real
                # current-season matches exist. Their only top-flight evidence is
                # stale (Ipswich 2024/25) or absent (Coventry, Hull), and a
                # silent league-average default would badly misprice GW1 — the
                # opener is literally ARS v COV.
                m_cur = float(self._mass_current.get(short, 0.0))
                m_eff = m_cur + max(mass - m_cur, 0.0) * self.promoted_history_discount
                # If the persistence fit failed there is no measured prior
                # strength; fall back to a full season, which is a hard shrink
                # rather than the accidental "trust it completely" that a zero
                # strength would produce.
                k_prom = k_cont if k_cont > 0 else float(SEASON_MATCHES)
                w = m_eff / (m_eff + k_prom)
                base_a, base_d = atk_prior, def_prior
            else:
                w = 1.0 if k_cont <= 0 else mass / (mass + k_cont)
                base_a, base_d = 1.0, 1.0

            if a_fit is None or d_fit is None or mass <= 0:
                w = 0.0
                a_fit, d_fit = base_a, base_d
            # Geometric blend: these are multipliers, so the mean that matters
            # is the one in log space.
            raw_atk[tid] = math.exp(w * math.log(a_fit) + (1 - w) * math.log(base_a))
            raw_dfn[tid] = math.exp(w * math.log(d_fit) + (1 - w) * math.log(base_d))
            conf[tid] = max(0.0, min(1.0, w))
            matches[tid] = n

        # Re-base onto the 20 clubs actually in this season's league. This is a
        # pure reparameterisation — every pairwise lambda is unchanged — but it
        # restores the "1.0 means average team in THIS league" reading after
        # three clubs were relegated and three promoted.
        g_a = math.exp(sum(math.log(v) for v in raw_atk.values()) / max(len(raw_atk), 1))
        g_d = math.exp(sum(math.log(v) for v in raw_dfn.values()) / max(len(raw_dfn), 1))
        self.league_avg = fit.league_avg * g_a * g_d
        self.home_adv = fit.home_advantage

        self._strength = {}
        for tid in state.teams:
            self._strength[tid] = TeamStrength(
                team_id=tid,
                attack=raw_atk[tid] / g_a,
                defence=raw_dfn[tid] / g_d,
                home_advantage=self.home_adv,
                n_matches=matches[tid],
                confidence=conf[tid],
            )

        self._build_sot_rates(state)
        self.mean_lambda = self._mean_lambda(state)

    def _build_sot_rates(self, state: "GameState") -> None:
        """Per-team shots-on-target attack/defence, for the goalkeeper saves model.

        Clubs with no shot history (the promoted three) are mapped from their
        goal ratings through an elasticity fitted across the clubs that have
        both, rather than dropped to league average — a side that concedes 25%
        more goals also faces more shots on target, and the keeper model needs
        that.
        """
        self._sot = {}
        if self.sot_fit is None:
            self.sot_league_avg = float("nan")
            self.sot_home_adv = 1.0
            return
        sf = self.sot_fit
        self.sot_league_avg = sf.league_avg
        self.sot_home_adv = sf.home_advantage

        xs_a, ys_a, xs_d, ys_d = [], [], [], []
        for tid, team in state.teams.items():
            short = team.short_name
            if sf.mass.get(short, 0.0) <= 1.0 or self._strength[tid].confidence <= 0.05:
                continue
            xs_a.append(math.log(self._strength[tid].attack))
            ys_a.append(math.log(sf.attack[short]))
            xs_d.append(math.log(self._strength[tid].defence))
            ys_d.append(math.log(sf.defence[short]))

        def elasticity(xs: List[float], ys: List[float]) -> float:
            sxx = sum(x * x for x in xs)
            return (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx > 1e-9 else 1.0

        e_a = elasticity(xs_a, ys_a) if len(xs_a) >= 8 else 1.0
        e_d = elasticity(xs_d, ys_d) if len(xs_d) >= 8 else 1.0
        self.persistence["sot_attack_elasticity"] = e_a
        self.persistence["sot_defence_elasticity"] = e_d

        for tid, team in state.teams.items():
            short = team.short_name
            mass = sf.mass.get(short, 0.0)
            conf = self._strength[tid].confidence
            if mass > 1.0 and short in sf.attack and conf > 0.05:
                a = sf.attack[short]
                d = sf.defence[short]
                # Shrink the club's own shot ratings by the same confidence its
                # goal ratings earned, toward the elasticity prediction.
                pred_a = math.exp(e_a * math.log(self._strength[tid].attack))
                pred_d = math.exp(e_d * math.log(self._strength[tid].defence))
                a = math.exp(conf * math.log(a) + (1 - conf) * math.log(pred_a))
                d = math.exp(conf * math.log(d) + (1 - conf) * math.log(pred_d))
            else:
                a = math.exp(e_a * math.log(self._strength[tid].attack))
                d = math.exp(e_d * math.log(self._strength[tid].defence))
            self._sot[tid] = {"attack": a, "defence": d}

        g_a = math.exp(sum(math.log(v["attack"]) for v in self._sot.values()) / max(len(self._sot), 1))
        g_d = math.exp(sum(math.log(v["defence"]) for v in self._sot.values()) / max(len(self._sot), 1))
        for tid in self._sot:
            self._sot[tid]["attack"] /= g_a
            self._sot[tid]["defence"] /= g_d
        self.sot_league_avg = sf.league_avg * g_a * g_d

    def _mean_lambda(self, state: "GameState") -> float:
        lams = []
        for f in getattr(state, "fixtures", []) or []:
            if f.gw is None:
                continue
            lh, la = self.match_lambdas(f.team_h, f.team_a)
            lams.extend([lh, la])
        return float(np.mean(lams)) if lams else float("nan")

    # -- odds ---------------------------------------------------------------

    def _index_odds(self, allres: pd.DataFrame, state: "GameState") -> None:
        """Pre-compute market lambdas for every historical match with prices.

        Keyed by (home FPL id, away FPL id) with the match date kept, so a
        backtest can ask for "the odds for this exact fixture" and a rerun of
        the same pairing in a different season does not collide.
        """
        self._odds_index = {}
        short_to_id = {t.short_name: tid for tid, t in state.teams.items()}
        sub = allres[allres["odds_home"].notna()] if "odds_home" in allres.columns else allres.iloc[0:0]
        if sub.empty:
            return
        lam_h, lam_a = market_lambdas_frame(sub, rho=self.rho)

        # Calibration check, not a correction. Inverting a Poisson through the
        # over/under-2.5 line is only exact if goals really are Poisson; if the
        # implied totals ran systematically hot or cold this ratio would show it
        # and downstream clean-sheet numbers would be biased. Measured at 0.99
        # over 1140 matches, so no correction is applied.
        finite = np.isfinite(lam_h) & np.isfinite(lam_a)
        if finite.any():
            implied = float((lam_h[finite] + lam_a[finite]).mean())
            actual = float(
                (sub["home_goals"].to_numpy(dtype=float)[finite]
                 + sub["away_goals"].to_numpy(dtype=float)[finite]).mean()
            )
            self.odds_calibration_n = int(finite.sum())
            self.odds_total_calibration = actual / implied if implied > 0 else float("nan")
            self.odds_supremacy_bias = float(
                (sub["home_goals"].to_numpy(dtype=float)[finite]
                 - sub["away_goals"].to_numpy(dtype=float)[finite]).mean()
                - (lam_h[finite] - lam_a[finite]).mean()
            )
        for (_, row), lh, la in zip(sub.iterrows(), lam_h, lam_a):
            if not (np.isfinite(lh) and np.isfinite(la)):
                continue
            h = short_to_id.get(row["home_short"])
            a = short_to_id.get(row["away_short"])
            if h is None or a is None:
                continue  # a club not in this season's Premier League
            date = row["date"] if isinstance(row["date"], (datetime, pd.Timestamp)) else None
            self._odds_index.setdefault((h, a), []).append(
                (pd.Timestamp(date).to_pydatetime() if date is not None else None, float(lh), float(la))
            )

    def set_odds(
        self,
        fixture_id: int,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        odds_over25: Optional[float] = None,
        odds_under25: Optional[float] = None,
    ) -> bool:
        """Inject live prices for a future fixture. Returns True if usable."""
        lam = market_lambdas(odds_home, odds_draw, odds_away, odds_over25, odds_under25, rho=self.rho)
        if lam is None:
            return False
        self._odds_by_fixture[int(fixture_id)] = lam
        self._forecast_cache.pop((int(fixture_id), True), None)
        return True

    def odds_lambdas(self, fixture: Fixture) -> Optional[Tuple[float, float]]:
        """Market lambdas for a fixture, or None when no prices exist for it."""
        direct = self._odds_by_fixture.get(int(fixture.id))
        if direct is not None:
            return direct
        entries = self._odds_index.get((fixture.team_h, fixture.team_a))
        if not entries:
            return None
        kick = _parse_dt(fixture.kickoff_time)
        if kick is None:
            return entries[-1][1], entries[-1][2]
        best = None
        best_gap = None
        for date, lh, la in entries:
            if date is None:
                continue
            gap = abs((_as_utc(date) - kick).total_seconds())
            if best_gap is None or gap < best_gap:
                best, best_gap = (lh, la), gap
        # A week of tolerance covers rescheduling; anything further away is a
        # different season's meeting of the same two clubs.
        if best is None or best_gap is None or best_gap > 7 * 86400:
            return None
        return best

    # -- public API ---------------------------------------------------------

    def ratings(self) -> Dict[int, TeamStrength]:
        if not self._strength:
            raise RuntimeError("TeamRatingModel.fit() must be called before ratings()")
        return dict(self._strength)

    def strength(self, team_id: int) -> TeamStrength:
        return self._strength[team_id]

    def match_lambdas(self, home_id: int, away_id: int) -> Tuple[float, float]:
        h = self._strength[home_id]
        a = self._strength[away_id]
        lam_h = self.league_avg * h.attack * a.defence * self.home_adv
        lam_a = self.league_avg * a.attack * h.defence
        return self._clamp(lam_h, home_id, away_id), self._clamp(lam_a, away_id, home_id)

    def _clamp(self, lam: float, team_id: int, opp_id: int) -> float:
        if lam < LAMBDA_MIN or lam > LAMBDA_MAX:
            log.warning(
                "lambda %.3f for team %s vs %s is outside [%.1f, %.1f]; clamped",
                lam, team_id, opp_id, LAMBDA_MIN, LAMBDA_MAX,
            )
            return min(max(lam, LAMBDA_MIN), LAMBDA_MAX)
        return lam

    def forecast_fixture(
        self, fixture: Fixture, use_odds: Optional[bool] = None
    ) -> MatchForecast:
        use = self.use_odds if use_odds is None else use_odds
        key = (int(fixture.id), bool(use))
        cached = self._forecast_cache.get(key)
        if cached is not None:
            return cached

        lam_h, lam_a = self.match_lambdas(fixture.team_h, fixture.team_a)
        source = "model"
        if use:
            market = self.odds_lambdas(fixture)
            if market is not None:
                w = float(self.mc.odds_blend_weight)
                w = min(max(w, 0.0), 1.0)
                if w > 0:
                    # Geometric blend: lambdas are multiplicative rates.
                    lam_h = math.exp((1 - w) * math.log(lam_h) + w * math.log(max(market[0], 1e-6)))
                    lam_a = math.exp((1 - w) * math.log(lam_a) + w * math.log(max(market[1], 1e-6)))
                    source = "odds" if w >= 1.0 else "blend"

        matrix = stats.score_matrix(lam_h, lam_a, self.rho, MAX_GOALS)
        p_cs_h = sum(row[0] for row in matrix)          # away side fails to score
        p_cs_a = sum(matrix[0])                          # home side fails to score
        p_h, p_d, p_a = stats.match_outcome_probs(lam_h, lam_a, self.rho, MAX_GOALS)

        forecast = MatchForecast(
            fixture_id=int(fixture.id),
            gw=int(fixture.gw) if fixture.gw is not None else 0,
            team_h=int(fixture.team_h),
            team_a=int(fixture.team_a),
            lambda_h=lam_h,
            lambda_a=lam_a,
            p_cs_h=p_cs_h,
            p_cs_a=p_cs_a,
            p_home_win=p_h,
            p_draw=p_d,
            p_away_win=p_a,
            source=source,
        )
        self._forecast_cache[key] = forecast
        return forecast

    def forecast_all(
        self, state: "GameState", gws: Sequence[int], use_odds: Optional[bool] = None
    ) -> Dict[int, MatchForecast]:
        """``{fixture_id: MatchForecast}`` for every fixture in ``gws``.

        Keyed by fixture id, not gameweek: a double gameweek has two fixtures
        for the same club and both must survive.
        """
        wanted = set(int(g) for g in gws)
        out: Dict[int, MatchForecast] = {}
        for f in state.fixtures:
            if f.gw is None or int(f.gw) not in wanted:
                continue
            out[int(f.id)] = self.forecast_fixture(f, use_odds=use_odds)
        return out

    # -- shots on target ----------------------------------------------------

    def sot_for_rate(self, team_id: int, is_home: bool) -> float:
        """Shots on target this club records per match against an average opponent."""
        rec = self._sot.get(team_id)
        if rec is None or not np.isfinite(self.sot_league_avg):
            return float("nan")
        return self.sot_league_avg * rec["attack"] * (self.sot_home_adv if is_home else 1.0)

    def sot_faced_rate(self, team_id: int, is_home: bool) -> float:
        """Shots on target this club faces per match from an average opponent.

        Home/away is from the perspective of ``team_id``: at home the opponent
        is the away side and gets no home boost, away from home it does.
        """
        rec = self._sot.get(team_id)
        if rec is None or not np.isfinite(self.sot_league_avg):
            return float("nan")
        return self.sot_league_avg * rec["defence"] * (1.0 if is_home else self.sot_home_adv)

    def expected_sot(self, home_id: int, away_id: int) -> Tuple[float, float]:
        """Opponent-adjusted (home SOT, away SOT) for one fixture."""
        h, a = self._sot.get(home_id), self._sot.get(away_id)
        if h is None or a is None or not np.isfinite(self.sot_league_avg):
            return float("nan"), float("nan")
        return (
            self.sot_league_avg * h["attack"] * a["defence"] * self.sot_home_adv,
            self.sot_league_avg * a["attack"] * h["defence"],
        )

    # -- reporting ----------------------------------------------------------

    def season_projection(self, state: Optional["GameState"] = None) -> Dict[int, Dict[str, float]]:
        """Schedule-adjusted expected goals for and against per club, whole season.

        Unlike a rating multiplier this is directly checkable against a league
        table, which makes it the fastest way to spot a broken fit.
        """
        state = state or self.state
        out = {tid: {"gf": 0.0, "ga": 0.0, "played": 0.0, "cs": 0.0} for tid in state.teams}
        for f in state.fixtures:
            if f.gw is None:
                continue
            lh, la = self.match_lambdas(f.team_h, f.team_a)
            out[f.team_h]["gf"] += lh
            out[f.team_h]["ga"] += la
            out[f.team_h]["cs"] += math.exp(-la)
            out[f.team_h]["played"] += 1
            out[f.team_a]["gf"] += la
            out[f.team_a]["ga"] += lh
            out[f.team_a]["cs"] += math.exp(-lh)
            out[f.team_a]["played"] += 1
        return out

    def ratings_table(self, state: Optional["GameState"] = None) -> str:
        state = state or self.state
        rows = sorted(self._strength.values(), key=lambda s: -s.attack)
        proj = self.season_projection(state)
        head = (
            "%-3s %-4s %-16s %7s %7s %5s %6s %6s %5s %7s %7s"
            % ("#", "TEAM", "CLUB", "ATTACK", "DEFENCE", "CONF", "xGF38", "xGA38",
               "xCS", "SOT_FOR", "SOT_AGN")
        )
        lines = [head, "-" * len(head)]
        for i, s in enumerate(rows, 1):
            team = state.teams[s.team_id]
            p = proj[s.team_id]
            sot_f = 0.5 * (self.sot_for_rate(s.team_id, True) + self.sot_for_rate(s.team_id, False))
            sot_a = 0.5 * (self.sot_faced_rate(s.team_id, True) + self.sot_faced_rate(s.team_id, False))
            lines.append(
                "%-3d %-4s %-16s %7.3f %7.3f %5.2f %6.1f %6.1f %5.1f %7.2f %7.2f"
                % (i, team.short_name, team.name[:16], s.attack, s.defence, s.confidence,
                   p["gf"], p["ga"], p["cs"], sot_f, sot_a)
            )
        return "\n".join(lines)

    def fitted_parameters(self) -> Dict[str, Any]:
        """Everything the fit produced, for reports and reproducibility."""
        out = dict(self.diagnostics)
        out["teams"] = {
            self.state.teams[tid].short_name: {
                "attack": round(s.attack, 4),
                "defence": round(s.defence, 4),
                "confidence": round(s.confidence, 3),
                "n_matches": s.n_matches,
            }
            for tid, s in sorted(self._strength.items())
        } if self.state is not None else {}
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_season(grp: pd.DataFrame) -> RatingFit:
    """Unweighted single-season fit, used only to measure rating persistence."""
    shorts = sorted(set(grp["home_short"]) | set(grp["away_short"]))
    index = {s: i for i, s in enumerate(shorts)}
    hi = grp["home_short"].map(index).to_numpy(dtype=int)
    ai = grp["away_short"].map(index).to_numpy(dtype=int)
    gh = grp["home_goals"].to_numpy(dtype=float)
    ga = grp["away_goals"].to_numpy(dtype=float)
    w = np.ones(len(grp), dtype=float)
    raw = _fit_multiplicative(hi, ai, gh, ga, w, len(shorts))
    return RatingFit(
        league_avg=raw["league_avg"],
        home_advantage=raw["home_advantage"],
        attack={s: float(raw["attack"][i]) for s, i in index.items()},
        defence={s: float(raw["defence"][i]) for s, i in index.items()},
        n_matches=len(grp),
        iterations=raw["iterations"],
        converged=raw["converged"],
    )


def _parse_dt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _days_ago(dates: pd.Series, as_of: datetime, clamp: bool = True) -> np.ndarray:
    """Age of each match in days. Undated rows are treated as the oldest seen."""
    ts = pd.to_datetime(dates, errors="coerce")
    try:
        ts = ts.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    ref = pd.Timestamp(as_of).tz_localize(None) if as_of.tzinfo else pd.Timestamp(as_of)
    days = (ref - ts).dt.total_seconds() / 86400.0
    filled = days.fillna(days.max() if days.notna().any() else 0.0).to_numpy(dtype=float)
    return np.maximum(filled, 0.0) if clamp else filled


def _state_results(state: "GameState") -> pd.DataFrame:
    """Current-season results straight from the FPL fixture list.

    football-data does not publish the new season's file until matches are
    played (and lags a day or two after that), while the FPL API carries the
    score the moment a match finishes. This is what makes in-season rating
    updates work on GW1+1 rather than a week later.
    """
    cols = ["season", "date", "home_short", "away_short", "home_goals", "away_goals",
            "home_sot", "away_sot", "odds_home", "odds_draw", "odds_away",
            "odds_over25", "odds_under25"]
    rows = []
    for f in getattr(state, "fixtures", []) or []:
        if not f.finished or f.team_h_score is None or f.team_a_score is None:
            continue
        rows.append(
            {
                "season": getattr(state, "season", ""),
                "date": _parse_dt(f.kickoff_time),
                "home_short": state.teams[f.team_h].short_name,
                "away_short": state.teams[f.team_a].short_name,
                "home_goals": float(f.team_h_score),
                "away_goals": float(f.team_a_score),
                "home_sot": np.nan,
                "away_sot": np.nan,
                "odds_home": np.nan,
                "odds_draw": np.nan,
                "odds_away": np.nan,
                "odds_over25": np.nan,
                "odds_under25": np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return out[cols]


def build_team_ratings(
    state: "GameState", config: Optional[Config] = None, **kwargs: Any
) -> TeamRatingModel:
    """Convenience: load the historical results and fit in one call."""
    model = TeamRatingModel(config or Config.load())
    model.fit(None, state, **kwargs)
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from gaffer.data.loaders import fixtures_by_gw, load_game_state

    cfg = Config.load()
    t0 = time.time()
    state = load_game_state(cfg)
    print("loaded state in %.2fs -> %s" % (time.time() - t0, state.summary()))

    t0 = time.time()
    model = TeamRatingModel(cfg)
    results = hist.load_football_data(["2324", "2425", "2526"])
    model.fit(results, state)
    print("fitted in %.2fs" % (time.time() - t0))

    print("\n=== fit diagnostics ===")
    for k, v in model.diagnostics.items():
        print("  %-28s %s" % (k, v))

    print("\n=== 2026/27 team ratings (sorted by attack) ===")
    print(model.ratings_table(state))

    ratings = model.ratings()
    lam_all: List[float] = []
    totals: List[float] = []
    by_gw = fixtures_by_gw(state)
    for f in state.fixtures:
        if f.gw is None:
            continue
        lh, la = model.match_lambdas(f.team_h, f.team_a)
        lam_all.extend([lh, la])
        totals.append(lh + la)

    print("\n=== GW%d fixtures ===" % state.current_gw)
    print("  %-13s %6s %6s %7s %7s %6s %6s %6s %6s %6s"
          % ("FIXTURE", "lam_H", "lam_A", "p_CS_H", "p_CS_A", "P(H)", "P(D)", "P(A)",
             "sot_H", "sot_A"))
    for f in by_gw[state.current_gw]:
        fc = model.forecast_fixture(f)
        sot_h, sot_a = model.expected_sot(f.team_h, f.team_a)
        print(
            "  %-13s %6.2f %6.2f %7.3f %7.3f %6.3f %6.3f %6.3f %6.2f %6.2f"
            % (
                "%s v %s" % (state.short_name(f.team_h), state.short_name(f.team_a)),
                fc.lambda_h, fc.lambda_a, fc.p_cs_h, fc.p_cs_a,
                fc.p_home_win, fc.p_draw, fc.p_away_win, sot_h, sot_a,
            )
        )
        # The Dixon-Coles correction moves the clean-sheet probability off the
        # naive exp(-lambda); downstream code should read p_cs_* from here.
        assert abs(fc.p_cs_a - math.exp(-fc.lambda_h)) < 0.05
        assert abs(fc.p_home_win + fc.p_draw + fc.p_away_win - 1.0) < 1e-9

    print("\n=== sanity gates ===")
    mean_lambda = float(np.mean(lam_all))
    mean_total = float(np.mean(totals))
    print("  mean lambda            %.3f  (gate 1.20-1.70)" % mean_lambda)
    print("  goals per match        %.3f  (gate 2.70-2.90)" % mean_total)
    print("  home advantage         %.3f  (gate 1.05-1.25)" % model.home_adv)
    print("  lambda range           %.3f .. %.3f  (gate 0.20-4.00)" % (min(lam_all), max(lam_all)))
    print("  dixon-coles rho        %.4f" % model.rho)
    print("  fit converged          %s in %d iterations (max_delta=%.2e)"
          % (model.goal_fit.converged, model.goal_fit.iterations, model.goal_fit.max_delta))
    print("  loglik monotone        %s" % model.goal_fit.loglik_increasing)

    assert 1.20 <= mean_lambda <= 1.70, mean_lambda
    assert 2.70 <= mean_total <= 2.90, mean_total
    assert 1.05 <= model.home_adv <= 1.25, model.home_adv
    assert LAMBDA_MIN <= min(lam_all) and max(lam_all) <= LAMBDA_MAX, (min(lam_all), max(lam_all))
    assert model.goal_fit.converged and model.goal_fit.loglik_increasing

    order = sorted(ratings.values(), key=lambda s: -s.attack)
    top = [state.short_name(s.team_id) for s in order[:4]]
    bottom = [state.short_name(s.team_id) for s in order[-3:]]
    promoted = sorted(state.short_name(t) for t in state.promoted_team_ids)
    print("  top attacks            %s" % ", ".join(top))
    print("  weakest attacks        %s" % ", ".join(bottom))
    assert sorted(bottom) == promoted, (bottom, promoted)
    assert {"LIV", "ARS", "MCI"} <= set(state.short_name(s.team_id) for s in order[:5]), top
    promoted_conf = [ratings[t].confidence for t in state.promoted_team_ids]
    settled_conf = [s.confidence for s in ratings.values() if s.team_id not in state.promoted_team_ids]
    print("  promoted confidence     %s" % ", ".join("%.3f" % c for c in sorted(promoted_conf)))
    print("  established confidence  %.3f .. %.3f" % (min(settled_conf), max(settled_conf)))
    assert max(promoted_conf) < 0.25, promoted_conf
    assert min(settled_conf) > 0.5, min(settled_conf)
    # Promoted priors must actually be used, not silently replaced by the mean.
    for tid in state.promoted_team_ids:
        assert ratings[tid].attack < 0.85 and ratings[tid].defence > 1.15, (
            tid, ratings[tid].attack, ratings[tid].defence)
    print("  promoted three are the weakest attacks, all low confidence: OK")

    print("\n=== shots on target ===")
    sot_for = [model.sot_for_rate(t, True) for t in state.teams]
    sot_agn = [model.sot_faced_rate(t, True) for t in state.teams]
    print("  league SOT baseline    %.2f (home advantage %.3f)" % (model.sot_league_avg, model.sot_home_adv))
    print("  home sot_for range     %.2f .. %.2f" % (min(sot_for), max(sot_for)))
    print("  home sot_faced range   %.2f .. %.2f" % (min(sot_agn), max(sot_agn)))
    assert all(2.0 <= v <= 8.0 for v in sot_for + sot_agn), (min(sot_for + sot_agn), max(sot_for + sot_agn))

    print("\n=== odds path ===")
    # The vectorised helpers must be numerically identical to the stats module's
    # reference implementations; they exist only because a backtest inverts odds
    # for hundreds of matches inside a bisection loop.
    worst_p = 0.0
    for lh, la in ((0.4, 2.9), (1.2, 1.1), (2.5, 0.6), (1.75, 1.45)):
        ref_probs = stats.match_outcome_probs(lh, la, model.rho, MAX_GOALS)
        vec = _dc_outcome_probs_vec(np.array([lh]), np.array([la]), model.rho)
        worst_p = max(worst_p, max(abs(r - float(v[0])) for r, v in zip(ref_probs, vec)))
    print("  max |vectorised - stats.match_outcome_probs|: %.2e" % worst_p)
    assert worst_p < 1e-12, worst_p

    ref = hist.load_football_data(["2526"])
    sample = ref.tail(40).reset_index(drop=True)
    lam_h, lam_a = market_lambdas_frame(sample, rho=model.rho)
    worst = 0.0
    for i in range(0, len(sample), 4):
        row = sample.iloc[i]
        p = stats.remove_vig([row["odds_home"], row["odds_draw"], row["odds_away"]])
        tot = total_goals_from_over_under(stats.remove_vig([row["odds_over25"], row["odds_under25"]])[0])
        ref_h, ref_a = stats.lambdas_from_odds(p[0], p[1], p[2], tot)
        worst = max(worst, abs(ref_h - lam_h[i]), abs(ref_a - lam_a[i]))
    print("  max |vectorised - stats.lambdas_from_odds| over 10 matches: %.2e" % worst)
    assert worst < 1e-6, worst
    print("  market lambdas: home mean %.3f, away mean %.3f, total %.3f"
          % (np.nanmean(lam_h), np.nanmean(lam_a), np.nanmean(lam_h + lam_a)))
    print("  actual goals in those 40 matches: home %.3f away %.3f"
          % (sample["home_goals"].mean(), sample["away_goals"].mean()))
    print("  odds calibration over %d priced matches: actual/implied total %.4f, "
          "supremacy bias %+.3f goals"
          % (model.odds_calibration_n, model.odds_total_calibration, model.odds_supremacy_bias))
    assert 0.92 <= model.odds_total_calibration <= 1.08, model.odds_total_calibration

    # Injecting live prices for a future fixture must move that fixture only.
    gw1 = by_gw[state.current_gw][0]
    before = model.forecast_fixture(gw1, use_odds=True)
    assert before.source == "model", before.source
    assert model.set_odds(gw1.id, 1.30, 5.50, 9.00, 1.50, 2.55)
    after = model.forecast_fixture(gw1, use_odds=True)
    print("  set_odds on fixture %d: %s lam_h %.2f -> %.2f (%s)"
          % (gw1.id, "%s v %s" % (state.short_name(gw1.team_h), state.short_name(gw1.team_a)),
             before.lambda_h, after.lambda_h, after.source))
    assert after.source == "blend" and abs(after.lambda_h - before.lambda_h) > 1e-6
    assert model.forecast_fixture(gw1, use_odds=False).lambda_h == before.lambda_h

    # Backtest-style check: refit as of the start of 2025/26, forecast that
    # season out of sample and see whether blending odds in actually helps.
    print("\n=== out-of-sample check on 2025/26 (fit knows nothing after 2025-08-14) ===")
    as_of = datetime(2025, 8, 14, tzinfo=timezone.utc)
    past = TeamRatingModel(cfg)
    # Deliberately hand it the future season too: the leakage guard must drop it.
    past.fit(hist.load_football_data(["2324", "2425", "2526"]), state,
             as_of=as_of, include_state_results=False)
    print("  leakage guard dropped %d future results; fitted on %d"
          % (past.dropped_future_results, past.goal_fit.n_matches))
    assert past.dropped_future_results == 380, past.dropped_future_results
    assert past.fit_seasons == ["2023-24", "2024-25"], past.fit_seasons

    check = hist.load_football_data(["2526"])
    short_to_id = {t.short_name: tid for tid, t in state.teams.items()}
    check = check[check["home_short"].isin(short_to_id) & check["away_short"].isin(short_to_id)]
    mk_h, mk_a = market_lambdas_frame(check, rho=past.rho)

    def poisson_ll(lam: float, k: float) -> float:
        return -lam + k * math.log(max(lam, 1e-9)) - math.lgamma(k + 1)

    rows = []
    for (_, row), mh, ma in zip(check.iterrows(), mk_h, mk_a):
        h, a = short_to_id[row["home_short"]], short_to_id[row["away_short"]]
        lh, la = past.match_lambdas(h, a)
        if np.isfinite(mh) and np.isfinite(ma):
            rows.append((lh, la, mh, ma, float(row["home_goals"]), float(row["away_goals"])))
    print("  %-6s %8s %10s   (RMSE on goals / mean Poisson log-likelihood)" % ("weight", "RMSE", "logLik"))
    best = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        se = 0.0
        ll = 0.0
        for lh, la, mh, ma, gh_, ga_ in rows:
            bh = math.exp((1 - w) * math.log(lh) + w * math.log(mh))
            ba = math.exp((1 - w) * math.log(la) + w * math.log(ma))
            se += (bh - gh_) ** 2 + (ba - ga_) ** 2
            ll += poisson_ll(bh, gh_) + poisson_ll(ba, ga_)
        rmse = math.sqrt(se / (2 * len(rows)))
        ll /= 2 * len(rows)
        flag = " <- config odds_blend_weight" if abs(w - cfg.model.odds_blend_weight) < 1e-9 else ""
        print("  %-6.2f %8.4f %10.4f%s" % (w, rmse, ll, flag))
        if best is None or ll > best[1]:
            best = (w, ll)
    print("  %d matches; best blend weight by log-likelihood: %.2f" % (len(rows), best[0]))
    # Odds must beat the standalone model, or the whole odds path is pointless.
    ll0 = sum(poisson_ll(lh, gh_) + poisson_ll(la, ga_) for lh, la, _, _, gh_, ga_ in rows)
    ll1 = sum(poisson_ll(mh, gh_) + poisson_ll(ma, ga_) for _, _, mh, ma, gh_, ga_ in rows)
    assert ll1 > ll0, (ll0, ll1)

    print("\n=== prior decay (synthetic current-season matches) ===")
    import copy as _copy

    for n_played in (0, 20, 60, 200):
        fake = _copy.copy(state)
        fake.fixtures = list(state.fixtures)
        played = 0
        new_fixtures = []
        for f in state.fixtures:
            g = _copy.copy(f)
            if played < n_played and g.gw is not None:
                g.finished = True
                g.team_h_score = 2 if g.team_h in state.promoted_team_ids else 1
                g.team_a_score = 0
                g.kickoff_time = "2026-08-22T14:00:00Z"
                played += 1
            new_fixtures.append(g)
        fake.fixtures = new_fixtures
        m2 = TeamRatingModel(cfg)
        m2.fit(results, fake, as_of=datetime(2026, 12, 1, tzinfo=timezone.utc))
        cov = [t for t in state.promoted_team_ids if state.short_name(t) == "COV"][0]
        print("  %3d current matches -> prior weight %.3f, COV attack %.3f conf %.2f"
              % (n_played, m2.prior_season_weight, m2.ratings()[cov].attack,
                 m2.ratings()[cov].confidence))

    print("\n=== forecast_all / double-gameweek safety ===")
    horizon = list(range(state.current_gw, state.current_gw + cfg.model.default_horizon))
    t0 = time.time()
    fcs = model.forecast_all(state, horizon)
    print("  %d fixtures over GW%d-%d in %.3fs; keyed by fixture id so a double "
          "gameweek keeps both" % (len(fcs), horizon[0], horizon[-1], time.time() - t0))
    assert len(fcs) == sum(len(by_gw[g]) for g in horizon)
    assert all(f.gw in horizon for f in fcs.values())

    params = model.fitted_parameters()
    json.dumps(params)  # must round-trip into reports/ and the API
    print("  fitted_parameters() is JSON-serialisable, %d keys, %d clubs"
          % (len(params), len(params["teams"])))

    if model.warnings:
        print("\n=== warnings ===")
        for wmsg in model.warnings:
            print("  - %s" % wmsg)
    print("\nALL SANITY GATES PASSED")
