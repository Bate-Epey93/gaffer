"""Bonus points: an empirically fitted BPS model plus a top-3 simulator.

Bonus is worth up to 3 points a match and public tools systematically
under-model it, usually by attaching a flat "good players get bonus" fudge to
projected points. This module does the two things that actually matter:

1.  **The BPS weights are fitted, not asserted.** ``BonusModel.fit`` regresses
    observed per-match ``bps`` on observed event counts (minutes bucket, goals,
    assists, clean sheet, saves, penalty saves, clearances+blocks+interceptions,
    recoveries, tackles, goals conceded, cards, own goals, penalty misses) from
    the vaastav ``merged_gw.csv`` archive, one sign-constrained non-negative
    least squares fit per position. Coefficients, R-squared and residual
    standard deviation land in ``reports/bps_fit.json``.

    The fit recovers known official weights closely enough to trust it: the
    penalty-save coefficient comes out at ~8.1 against an official 8, and the
    minutes buckets at ~3 / ~6 against an official 3 / 6. The goal and assist
    coefficients come out *above* the official 12/18/24 and 9 because a goal is
    correlated with unmodelled positive BPS events (shots on target, big chances
    created, successful dribbles). That is a feature here, not a bias: we want
    the total BPS impact of a projected goal, including its correlates.

2.  **Bonus is contested across the whole match.** ``project_bonus`` takes every
    candidate from *both* teams, draws each player's BPS from a normal centred
    on their projection with a fitted variance, and reads off P(1st)/P(2nd)/
    P(3rd) via :func:`gaffer.core.stats.top_k_probabilities`. Because exactly
    one player takes each rank in every simulation draw, the expected bonus over
    a full candidate list sums to exactly 3+2+1 = 6.0 by construction. That
    conservation law is the whole point: get it wrong and every projection in
    the app inflates.

WARNING - THE 2026/27 COEFFICIENTS ARE 2025/26 FITS PLUS MANUAL RULE PATCHES.
--------------------------------------------------------------------------
No 2026/27 gameweek has been played, so the regression can only see 2025/26.
Four documented rule changes are applied on top of the fit (see
``ADJUSTMENTS_2026_27`` and ``_apply_2026_27_adjustments``):

* the -1 BPS for being tackled is removed - **unmodelled**, because "times
  tackled" is not in any available dataset. It is absorbed in the residual and
  makes this model very slightly pessimistic for take-on heavy attackers.
* clearances/blocks/interceptions move from 1 BPS per 2 actions to 1 per 3, so
  the fitted CBI coefficient is scaled by 2/3.
* goalkeeper saves are restructured (any save 2 BPS, +1 inside the box, +1 from
  a big chance). The nominal per-save value moves from 2 to ``2 + p_in_box +
  p_big_chance``; the fitted coefficient is scaled by that ratio. **This is the
  single largest uncertainty in the module** - the save-type mix is unobservable
  from our sources, so the ratio is bounded on [1.0, 2.0] and the smoke test
  prints keeper bonus at the floor, the default and the ceiling. Override
  ``gk_save_p_in_box`` / ``gk_save_p_big_chance`` on the instance and call
  ``apply_rule_adjustments()`` to move within that band.

  Read ``gk_save_marginal_diagnostic`` in the fit report before trusting keeper
  bonus. It measures what a save was *actually* worth in 2025/26 and returns
  2.81 +/- 0.06 BPS, comfortably above the official table's 2. Either the table
  understated the real value or saves correlate with unmodelled keeper BPS; both
  readings mean the 1.5x uplift applied here is on the aggressive side, and the
  effect is not small - it moves the keeper share of all bonus from 6.7% to
  14.0% on identical events. This is the first number to re-check against live
  results, and the reason the sensitivity table exists.
* the penalty-save BPS drops from 8 to 7, applied as a flat -1 on the fitted
  coefficient.

Every one of these disappears the moment real 2026/27 data exists:
:func:`BonusModel.refit_from_current_season` refits the same regression against
``GameState.player_history`` and replaces the patched coefficients wholesale.
Call it every gameweek from about GW6 onward.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gaffer.core import scoring, stats
from gaffer.core.config import REPORTS_DIR, Config, ensure_dirs
from gaffer.core.types import Fixture, MinutesForecast, Player
from gaffer.data import history as hist

log = logging.getLogger(__name__)

BPS_FIT_PATH = os.path.join(REPORTS_DIR, "bps_fit.json")

# --- Regression design -----------------------------------------------------
# Features whose BPS coefficient must be >= 0, and features whose coefficient
# must be <= 0. Splitting them lets one non-negative least squares solve handle
# both: the negative block is fed in sign-flipped and negated afterwards. Plain
# unconstrained OLS on this design puts a *positive* coefficient on red cards
# for forwards (only 3 in the sample), which is nonsense we refuse to ship.
FEATURES_POSITIVE: Tuple[str, ...] = (
    "mins_1_59",
    "mins_60",
    "goals",
    "assists",
    "clean_sheet",
    "saves",
    "penalties_saved",
    "cbi",
    "recoveries",
    "tackles",
)
FEATURES_NEGATIVE: Tuple[str, ...] = (
    "goals_conceded_pairs",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
)
FEATURES: Tuple[str, ...] = FEATURES_POSITIVE + FEATURES_NEGATIVE

# Source column for each feature in a per-match frame (vaastav merged_gw and the
# FPL element-summary `history` rows share these names). Derived features are
# built in `build_design_frame`.
_SOURCE_COLUMNS: Dict[str, str] = {
    "goals": "goals_scored",
    "assists": "assists",
    "clean_sheet": "clean_sheets",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "cbi": "clearances_blocks_interceptions",
    "recoveries": "recoveries",
    "tackles": "tackles",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",
    "penalties_missed": "penalties_missed",
}

# Counting events whose per-90 rate we estimate per player and whose
# match-to-match overdispersion we fit (BPS variance depends on them).
COUNT_EVENTS: Tuple[str, ...] = ("cbi", "recoveries", "tackles", "saves")

# --- 2026/27 rule patches --------------------------------------------------
# 2025/26 official nominal weights, used only as the denominator of a ratio so
# that the empirical "correlate" component of a fitted coefficient is preserved
# through the adjustment.
CBI_ACTIONS_PER_BPS_2526 = 2.0
CBI_ACTIONS_PER_BPS_2627 = 3.0
GK_SAVE_NOMINAL_2526 = 2.0  # official pre-2026/27 BPS table: a save is 2 BPS
GK_SAVE_BASE_2627 = 2.0  # any save
GK_SAVE_BONUS_IN_BOX = 1.0  # +1 if the save is inside the box
GK_SAVE_BONUS_BIG_CHANCE = 1.0  # +1 if the save is from a big chance
# The save-type mix is not observable from the FPL API, football-data.co.uk or
# the vaastav archive - none of them carry save location or big-chance flags.
# These two are declared assumptions, deliberately round, and the *only*
# fabricated numbers in this module. The rule itself bounds the nominal per-save
# value on [2.0, 4.0]; these put it at 3.0, the midpoint of that support.
# `save_uplift_bounds` in the fit report carries the full range and the smoke
# test prints keeper bonus at both ends.
GK_SAVE_P_IN_BOX = 0.75
GK_SAVE_P_BIG_CHANCE = 0.25
PENALTY_SAVE_BPS_DELTA_2627 = -1.0  # 8 -> 7

ADJUSTMENTS_2026_27: Dict[str, str] = {
    "tackled": "-1 BPS for being tackled removed. UNMODELLED: no dataset carries "
    "'times tackled'. Absorbed in the residual; makes attackers marginally "
    "under-projected.",
    "cbi": "1 BPS per 3 clearances/blocks/interceptions (was per 2). Fitted CBI "
    "coefficient scaled by 2/3.",
    "gk_saves": "Any save 2 BPS, +1 inside the box, +1 from a big chance. Fitted "
    "save coefficient scaled by (2 + p_in_box + p_big_chance) / 2. Largest "
    "uncertainty in the model; bounded ratio [1.0, 2.0].",
    "penalty_save": "Penalty save BPS 8 -> 7. Flat -1 on the fitted coefficient.",
}

# Minutes-bucket midpoints used when a caller gives us expected minutes but no
# explicit p_60 / p_appear.
_SUB_MINUTES_DEFAULT = 20.0

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Fitted-parameter containers
# ---------------------------------------------------------------------------


@dataclass
class PositionFit:
    """One position's BPS regression."""

    position: int
    n_rows: int
    r2: float
    resid_sd: float
    resid_mean: float
    coefficients: Dict[str, float] = field(default_factory=dict)  # as fitted, 2025/26
    coefficients_2026_27: Dict[str, float] = field(default_factory=dict)
    mean_bps: float = 0.0
    sd_bps: float = 0.0

    def weight(self, feature: str) -> float:
        return self.coefficients_2026_27.get(feature, 0.0)


@dataclass
class BonusContext:
    """Minimal fixture context so this module runs before ``xp.py`` exists.

    ``xp.FixtureContext`` is accepted directly by every method here and is the
    preferred input - this exists only so ``bonus.py`` has a self-contained
    smoke test and so callers who only have a fixture and two lambdas can still
    get a projection. Attribute names match ``xp.FixtureContext`` exactly.
    """

    fixture: Fixture
    player: Optional[Player] = None
    is_home: bool = True
    team_lambda: float = 1.45
    opponent_lambda: float = 1.45
    forecast: Optional[Any] = None
    minutes: Optional[MinutesForecast] = None


@dataclass
class EventProjection:
    """Resolved per-fixture event expectations feeding the BPS sum.

    Everything here is *conditional on the player appearing*; the appearance
    probability is applied once, at the end, in ``project_bps_and_sd``.
    """

    p_60: float = 0.0  # P(60+ | appears)
    goals: float = 0.0
    assists: float = 0.0
    clean_sheet: float = 0.0
    saves: float = 0.0
    penalties_saved: float = 0.0
    cbi: float = 0.0
    recoveries: float = 0.0
    tackles: float = 0.0
    goals_conceded_pairs: float = 0.0
    yellow_cards: float = 0.0
    red_cards: float = 0.0
    own_goals: float = 0.0
    penalties_missed: float = 0.0
    p_appear: float = 0.0
    lambda_conceded: float = 0.0

    def counts(self) -> Dict[str, float]:
        out = {f: float(getattr(self, f)) for f in FEATURES if hasattr(self, f)}
        out["mins_1_59"] = max(0.0, 1.0 - self.p_60)
        out["mins_60"] = self.p_60
        return out


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def nnls(A: np.ndarray, b: np.ndarray, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Lawson-Hanson non-negative least squares. scipy is not installed here."""
    m, n = A.shape
    passive = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=float)
    w = A.T.dot(b - A.dot(x))
    s = x.copy()
    for _ in range(max_iter):
        free = ~passive
        if not free.any() or (w[free] <= tol).all():
            break
        candidates = np.where(free)[0]
        j = int(candidates[int(np.argmax(w[candidates]))])
        passive[j] = True
        for _ in range(3 * n + 10):
            s = np.zeros(n, dtype=float)
            s[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if (s[passive] > tol).all():
                break
            blocking = passive & (s <= tol)
            denom = x[blocking] - s[blocking]
            denom = np.where(np.abs(denom) < _EPS, _EPS, denom)
            alpha = float(np.min(x[blocking] / denom))
            x = x + alpha * (s - x)
            passive = passive & (x > tol)
        x = s
        w = A.T.dot(b - A.dot(x))
    return x


def _expected_conceded_pairs(lam: float, max_goals: int = 12) -> float:
    """E[floor(goals_conceded / 2)] under Poisson(lam)."""
    if lam <= 0:
        return 0.0
    return sum(stats.poisson_pmf(k, lam) * (k // 2) for k in range(0, max_goals + 1))


def _var_conceded_pairs(lam: float, max_goals: int = 12) -> float:
    if lam <= 0:
        return 0.0
    mean = _expected_conceded_pairs(lam, max_goals)
    second = sum(stats.poisson_pmf(k, lam) * (k // 2) ** 2 for k in range(0, max_goals + 1))
    return max(0.0, second - mean * mean)


def _negbin_variance(mean: float, dispersion: float) -> float:
    """Var = mean + mean^2 / dispersion; dispersion -> inf recovers Poisson."""
    if mean <= 0:
        return 0.0
    if dispersion is None or dispersion <= 0 or dispersion > 1e6:
        return mean
    return mean + (mean * mean) / dispersion


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return min(1.0, max(0.0, x))


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or math.isinf(v):
        return default
    return v


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def build_design_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived regression columns to a per-match player frame.

    Works on both the vaastav ``merged_gw.csv`` and FPL ``element-summary``
    history frames - they share column names. Rows with zero minutes are kept by
    the caller's choice; they contribute nothing to a no-intercept fit but do
    inflate R-squared, so ``fit_bps_coefficients`` drops them.
    """
    out = df.copy()
    minutes = pd.to_numeric(out.get("minutes"), errors="coerce").fillna(0.0)
    out["mins_1_59"] = ((minutes >= 1) & (minutes < 60)).astype(float)
    out["mins_60"] = (minutes >= 60).astype(float)
    for feature, source in _SOURCE_COLUMNS.items():
        if source in out.columns:
            out[feature] = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
        else:
            out[feature] = 0.0
    conceded = pd.to_numeric(out.get("goals_conceded"), errors="coerce").fillna(0.0)
    # BPS docks goals conceded in pairs, so the linear regressor is floor(gc/2).
    out["goals_conceded_pairs"] = np.floor(conceded / 2.0)
    out["bps"] = pd.to_numeric(out.get("bps"), errors="coerce").fillna(0.0)
    out["minutes"] = minutes
    return out


def fit_bps_coefficients(
    frame: pd.DataFrame, position_column: str = "element_type"
) -> Dict[int, PositionFit]:
    """One sign-constrained NNLS fit per position on played matches."""
    design = build_design_frame(frame)
    design = design[design["minutes"] > 0]
    fits: Dict[int, PositionFit] = {}
    for position in scoring.POSITIONS:
        sub = design[pd.to_numeric(design[position_column], errors="coerce") == position]
        if len(sub) < len(FEATURES) * 5:
            log.warning(
                "position %s: only %d rows, skipping fit",
                scoring.POS_NAME[position],
                len(sub),
            )
            continue
        blocks = [sub[f].astype(float).values for f in FEATURES_POSITIVE]
        blocks += [-sub[f].astype(float).values for f in FEATURES_NEGATIVE]
        A = np.column_stack(blocks)
        b = sub["bps"].astype(float).values
        raw = nnls(A, b)
        pred = A.dot(raw)
        resid = b - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((b - b.mean()) ** 2).sum())
        coefficients: Dict[str, float] = {}
        for idx, name in enumerate(FEATURES):
            sign = 1.0 if name in FEATURES_POSITIVE else -1.0
            coefficients[name] = float(sign * raw[idx])
        fits[position] = PositionFit(
            position=position,
            n_rows=int(len(sub)),
            r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            resid_sd=float(resid.std(ddof=1)) if len(resid) > 1 else 0.0,
            resid_mean=float(resid.mean()),
            coefficients=coefficients,
            coefficients_2026_27=dict(coefficients),
            mean_bps=float(b.mean()),
            sd_bps=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        )
    return fits


def _apply_2026_27_adjustments(
    fits: Dict[int, PositionFit],
    p_in_box: float = GK_SAVE_P_IN_BOX,
    p_big_chance: float = GK_SAVE_P_BIG_CHANCE,
) -> Dict[str, Any]:
    """Patch 2025/26 coefficients with the four known 2026/27 rule changes.

    ``p_in_box`` / ``p_big_chance`` are the two unobservable save-mix
    assumptions; setting both to 0 gives the rule's floor (every save worth
    exactly 2) and both to 1 its ceiling. Returns a record of exactly what was
    applied, for the fit report.
    """
    cbi_ratio = CBI_ACTIONS_PER_BPS_2526 / CBI_ACTIONS_PER_BPS_2627  # 2/3
    save_nominal = GK_SAVE_BASE_2627 + (
        GK_SAVE_BONUS_IN_BOX * p_in_box + GK_SAVE_BONUS_BIG_CHANCE * p_big_chance
    )
    save_ratio = save_nominal / GK_SAVE_NOMINAL_2526
    record: Dict[str, Any] = {
        "cbi_scale": cbi_ratio,
        "gk_save_p_in_box": p_in_box,
        "gk_save_p_big_chance": p_big_chance,
        "gk_save_nominal_2526": GK_SAVE_NOMINAL_2526,
        "gk_save_nominal_2627": save_nominal,
        "gk_save_scale": save_ratio,
        "gk_save_scale_bounds": [
            GK_SAVE_BASE_2627 / GK_SAVE_NOMINAL_2526,
            (GK_SAVE_BASE_2627 + GK_SAVE_BONUS_IN_BOX + GK_SAVE_BONUS_BIG_CHANCE)
            / GK_SAVE_NOMINAL_2526,
        ],
        "penalty_save_delta": PENALTY_SAVE_BPS_DELTA_2627,
        "tackled_removed": "unmodelled",
        "per_position": {},
    }
    for position, fit in fits.items():
        new = dict(fit.coefficients)
        new["cbi"] = new.get("cbi", 0.0) * cbi_ratio
        if new.get("saves", 0.0) > 0:
            new["saves"] = new["saves"] * save_ratio
        if new.get("penalties_saved", 0.0) > 0:
            new["penalties_saved"] = max(
                0.0, new["penalties_saved"] + PENALTY_SAVE_BPS_DELTA_2627
            )
        fit.coefficients_2026_27 = new
        record["per_position"][scoring.POS_NAME[position]] = {
            "cbi": [fit.coefficients.get("cbi", 0.0), new["cbi"]],
            "saves": [fit.coefficients.get("saves", 0.0), new["saves"]],
            "penalties_saved": [
                fit.coefficients.get("penalties_saved", 0.0),
                new["penalties_saved"],
            ],
        }
    return record


# ---------------------------------------------------------------------------
# Auxiliary empirical fits
# ---------------------------------------------------------------------------


def _pooled_dispersion(
    frame: pd.DataFrame, column: str, group_column: str = "element", min_matches: int = 8
) -> float:
    """Within-player method-of-moments negative-binomial dispersion.

    Pooling raw counts across players would conflate between-player spread
    (Kerkez clears more than Palmer) with match-to-match noise, and only the
    latter belongs in the BPS variance.
    """
    sub = frame[frame["minutes"] >= 60]
    if len(sub) == 0 or column not in sub.columns:
        return 1e6
    grouped = sub.groupby(group_column)[column]
    means = grouped.mean()
    variances = grouped.var(ddof=1)
    counts = grouped.count()
    keep = (counts >= min_matches) & means.notna() & variances.notna() & (means > 0.05)
    if not keep.any():
        return 1e6
    mu = means[keep].values
    var = variances[keep].values
    n = counts[keep].values.astype(float)
    excess = np.maximum(var - mu, 0.0)
    num = float((n * mu * mu).sum())
    den = float((n * excess).sum())
    if den <= 0:
        return 1e6
    return max(0.05, num / den)


def _opponent_attack_index(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Per-match opponent attacking strength (1.0 = league average).

    ``opponent_team`` is a season-local team id; the name behind it is recovered
    by pairing the two clubs present in each fixture.
    """
    needed = {"fixture", "team", "opponent_team", "was_home", "team_h_score", "team_a_score"}
    if not needed.issubset(set(frame.columns)):
        return None
    id_to_name: Dict[int, str] = {}
    for _, grp in frame.groupby("fixture"):
        names = list(pd.unique(grp["team"].dropna()))
        if len(names) != 2:
            continue
        for name in names:
            other = names[0] if names[1] == name else names[1]
            ids = pd.unique(grp.loc[grp["team"] == name, "opponent_team"].dropna())
            if len(ids) == 1:
                id_to_name[int(ids[0])] = other
    if not id_to_name:
        return None
    goals_for = np.where(
        frame["was_home"].astype(bool),
        pd.to_numeric(frame["team_h_score"], errors="coerce"),
        pd.to_numeric(frame["team_a_score"], errors="coerce"),
    )
    per_match = pd.DataFrame(
        {"fixture": frame["fixture"], "team": frame["team"], "goals": goals_for}
    ).groupby(["fixture", "team"])["goals"].first().reset_index()
    attack = per_match.groupby("team")["goals"].mean()
    league = float(attack.mean())
    if league <= 0:
        return None
    index = frame["opponent_team"].map(id_to_name).map(attack) / league
    return index


def _fit_opponent_tilt(frame: pd.DataFrame) -> Dict[str, float]:
    """Log-log elasticity of each defensive action count vs opponent attack.

    Fixture polarity is real and not uniform: facing a stronger attack raises
    clearances, tackles and keeper saves (you are defending more) but *lowers*
    recoveries (you have the ball less, so fewer loose balls to win back).
    Fitting one pooled exponent for all of them would cancel the two effects
    against each other. Player means are divided out first, so this measures
    within-player fixture sensitivity rather than "bad teams defend more".
    """
    out: Dict[str, float] = {"cbi": 0.0, "tackles": 0.0, "recoveries": 0.0, "saves": 0.0}
    index = _opponent_attack_index(frame)
    if index is None:
        return out
    sub = frame[(frame["minutes"] >= 60) & index.notna()].copy()
    sub["opp_attack"] = index[sub.index]
    if len(sub) < 500:
        return out
    for action in ("cbi", "tackles", "recoveries", "saves"):
        # Saves only exist for keepers; the other three are outfield-dominated.
        rows = sub[sub["element_type"] == scoring.GKP] if action == "saves" else sub
        if len(rows) < 200:
            continue
        rate = rows[action].astype(float) / rows["minutes"].astype(float) * 90.0
        player_mean = rate.groupby(rows["element"]).transform("mean")
        n = rate.groupby(rows["element"]).transform("count")
        keep = (n >= 8) & (player_mean > 0.15)
        if keep.sum() < 200:
            continue
        y = np.log(np.maximum(rate[keep].values, 0.03) / np.maximum(player_mean[keep].values, 0.03))
        x = np.log(np.maximum(rows.loc[keep, "opp_attack"].values.astype(float), 0.05))
        A = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        out[action] = float(beta[1])
    return out


def _fit_price_start_curve(
    frame: pd.DataFrame, min_rows: int = 10
) -> Dict[int, List[Tuple[float, float]]]:
    """Price (in £m) -> P(start) per position, from last season's actual data.

    Pre-season a promoted-club player or a new signing has no top-flight history
    at all, and a zero there is not "he will not play", it is "we do not know".
    Price is the market's own minutes forecast and it is strongly monotone: this
    fits the curve rather than assuming its shape. Points are decile midpoints,
    then enforced monotone by a running maximum so interpolation never dips.
    """
    curves: Dict[int, List[Tuple[float, float]]] = {}
    needed = {"element", "element_type", "starts", "value"}
    if not needed.issubset(set(frame.columns)):
        return curves
    grouped = frame.groupby(["element_type", "element"]).agg(
        starts=("starts", "sum"), rows=("starts", "size"), price=("value", "mean")
    )
    grouped = grouped[grouped["rows"] >= min_rows]
    grouped["p_start"] = grouped["starts"] / grouped["rows"]
    grouped["price"] = grouped["price"] / 10.0
    for position in scoring.POSITIONS:
        try:
            sub = grouped.xs(position, level="element_type")
        except KeyError:
            continue
        if len(sub) < 20:
            continue
        sub = sub.sort_values("price")
        n_bins = max(4, min(10, len(sub) // 12))
        edges = np.linspace(0, len(sub), n_bins + 1).astype(int)
        points: List[Tuple[float, float]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            chunk = sub.iloc[lo:hi]
            points.append((float(chunk["price"].mean()), float(chunk["p_start"].mean())))
        running = 0.0
        monotone = []
        for price, p in points:
            running = max(running, p)
            monotone.append((price, running))
        curves[position] = monotone
    return curves


def _fit_starters_per_position(frame: pd.DataFrame) -> Dict[int, float]:
    """Average number of starters a club fields at each position.

    Formations vary, so this is fitted rather than assumed to be 1-4-4-2. It is
    the capacity each club's start probabilities are allocated against in the
    fallback minutes model, which is what keeps a squad's probabilities adding
    up to eleven players instead of twenty-five.
    """
    out: Dict[int, float] = {}
    needed = {"fixture", "team_short", "element_type", "starts"}
    if not needed.issubset(set(frame.columns)):
        return out
    counts = (
        frame.groupby(["fixture", "team_short", "element_type"])["starts"].sum().reset_index()
    )
    teams = counts.groupby(["fixture", "team_short"]).size()
    if len(teams) == 0:
        return out
    for position in scoring.POSITIONS:
        sub = counts[counts["element_type"] == position]
        if len(sub) == 0:
            continue
        out[position] = float(sub["starts"].sum()) / float(len(teams))
    return out


def _fit_start_to_60_rate(frame: pd.DataFrame) -> Dict[int, float]:
    """P(minutes >= 60 | started), per position. Substitutions are not uniform:
    forwards get hooked far more often than centre-backs."""
    out: Dict[int, float] = {}
    if "starts" not in frame.columns:
        return out
    started = frame[frame["starts"] >= 1]
    for position in scoring.POSITIONS:
        sub = started[pd.to_numeric(started["element_type"], errors="coerce") == position]
        if len(sub) < 50:
            continue
        out[position] = float((sub["minutes"] >= 60).mean())
    return out


def _fit_gk_save_marginal(frame: pd.DataFrame) -> Dict[str, float]:
    """Marginal BPS per keeper save in the source season, with a standard error.

    This is a *diagnostic on the assumption that drives the biggest adjustment
    in the module*, not an input to it. ``GK_SAVE_NOMINAL_2526 = 2.0`` is taken
    from the official pre-2026/27 BPS table; if this fitted marginal comes back
    materially above 2, the official table was not the whole story and the 1.5x
    save uplift is too aggressive. Restricted to full 90-minute keeper matches
    with the clean sheet and goals-conceded terms controlled, plus an intercept,
    since only the slope is wanted here.
    """
    out = {"marginal_bps_per_save": float("nan"), "std_error": float("nan"), "n": 0}
    sub = frame[(frame["element_type"] == scoring.GKP) & (frame["minutes"] >= 90)]
    if len(sub) < 100:
        return out
    A = np.column_stack([
        np.ones(len(sub)),
        sub["saves"].astype(float).values,
        sub["goals_conceded_pairs"].astype(float).values,
        pd.to_numeric(sub["goals_conceded"], errors="coerce").fillna(0.0).values,
        sub["clean_sheet"].astype(float).values,
        sub["penalties_saved"].astype(float).values,
    ])
    b = sub["bps"].astype(float).values
    beta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A.dot(beta)
    dof = max(1, len(sub) - rank)
    sigma2 = float((resid ** 2).sum()) / dof
    try:
        cov = sigma2 * np.linalg.pinv(A.T.dot(A))
        se = float(math.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:  # pragma: no cover
        se = float("nan")
    out["marginal_bps_per_save"] = float(beta[1])
    out["std_error"] = se
    out["n"] = int(len(sub))
    return out


def _position_baselines(frame: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Minutes-weighted per-90 event rates by position, used as shrinkage priors."""
    out: Dict[int, Dict[str, float]] = {}
    played = frame[frame["minutes"] > 0]
    for position in scoring.POSITIONS:
        sub = played[pd.to_numeric(played["element_type"], errors="coerce") == position]
        minutes = float(sub["minutes"].sum())
        rates: Dict[str, float] = {}
        if minutes <= 0:
            out[position] = {e: 0.0 for e in COUNT_EVENTS}
            continue
        for event in COUNT_EVENTS + ("yellow_cards", "red_cards", "own_goals"):
            rates[event] = float(sub[event].sum()) / minutes * 90.0
        # Attacking rates too, so the standalone fallback projection has a prior
        # for players with no top-flight history at all.
        rates["goals"] = float(sub["goals"].sum()) / minutes * 90.0
        rates["assists"] = float(sub["assists"].sum()) / minutes * 90.0
        out[position] = rates
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class BonusModel:
    """Fitted BPS weights plus the per-fixture top-3 bonus simulator."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.mc = self.config.model
        self.fits: Dict[int, PositionFit] = {}
        self.dispersion: Dict[int, Dict[str, float]] = {}
        self.tilt: Dict[str, float] = {}
        self.baselines: Dict[int, Dict[str, float]] = {}
        self.price_start_curve: Dict[int, List[Tuple[float, float]]] = {}
        self.start_to_60: Dict[int, float] = {}
        self.starters_per_position: Dict[int, float] = {}
        self.gk_save_diagnostic: Dict[str, float] = {}
        # Per-instance overrides for the two unobservable save-mix assumptions.
        # Set both to 0.0 for the rule's floor (a save is worth exactly 2 BPS,
        # i.e. no uplift at all on the fitted coefficient) or both to 1.0 for
        # its ceiling, then call `apply_rule_adjustments()`.
        self.gk_save_p_in_box: float = GK_SAVE_P_IN_BOX
        self.gk_save_p_big_chance: float = GK_SAVE_P_BIG_CHANCE
        self.player_rates: Dict[int, Dict[str, float]] = {}
        self.fallback_minutes: Dict[int, Tuple[float, float, float]] = {}
        self.adjustments: Dict[str, Any] = {}
        self.source_season: str = ""
        self.fitted_from: str = ""
        self.league_lambda: float = 1.45
        self.warnings: List[str] = []
        self.state: Optional[Any] = None
        self._sd_cache: Dict[int, float] = {}
        self._bps_cache: Dict[int, float] = {}
        self._positions: Dict[int, int] = {}
        self._by_team: Dict[int, List[int]] = {}

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        state: Optional[Any] = None,
        season: Optional[str] = None,
        frame: Optional[pd.DataFrame] = None,
        write_report: bool = True,
    ) -> None:
        """Fit BPS weights from a completed season, then patch for 2026/27.

        ``frame`` overrides the data source (used by
        ``refit_from_current_season`` and by tests). ``state`` is needed for the
        per-player defensive-action rates and for position lookups; without it
        the coefficients still fit but every projection falls back to the
        position baseline.
        """
        self.state = state
        self.warnings = []
        season = season or (getattr(state, "prior_season", "") or hist.prior_season(self.config.season))
        if frame is None:
            frame = hist.load_vaastav_gws(season)
            self.fitted_from = "vaastav merged_gw %s" % hist.season_dash(season)
        else:
            self.fitted_from = "caller-supplied frame"
        self.source_season = hist.season_dash(season)

        design = build_design_frame(frame)
        played = design[design["minutes"] > 0]

        self.fits = fit_bps_coefficients(frame)
        if not self.fits:
            raise ValueError("BPS fit produced no positions; source frame has %d rows" % len(frame))
        self.apply_rule_adjustments()

        self.dispersion = {}
        for position in scoring.POSITIONS:
            sub = played[pd.to_numeric(played["element_type"], errors="coerce") == position]
            self.dispersion[position] = {
                event: _pooled_dispersion(sub, event) for event in COUNT_EVENTS
            }
        self.tilt = _fit_opponent_tilt(played)
        self.baselines = _position_baselines(played)
        self.price_start_curve = _fit_price_start_curve(design)
        self.start_to_60 = _fit_start_to_60_rate(played)
        self.starters_per_position = _fit_starters_per_position(design)
        self.gk_save_diagnostic = _fit_gk_save_marginal(played)

        if state is not None:
            self._positions = {pid: p.position for pid, p in state.players.items()}
            self._by_team = {}
            for pid, player in state.players.items():
                self._by_team.setdefault(player.team_id, []).append(pid)
            self._build_player_rates(state)
            self._build_fallback_minutes(state)
        else:
            self.warnings.append("fitted without a GameState: per-player rates unavailable")

        if write_report:
            self.write_report()
        log.info(
            "BPS fit from %s: %s",
            self.fitted_from,
            ", ".join(
                "%s R2=%.3f sd=%.2f" % (scoring.POS_NAME[p], f.r2, f.resid_sd)
                for p, f in sorted(self.fits.items())
            ),
        )

    def apply_rule_adjustments(self) -> Dict[str, Any]:
        """Re-derive the 2026/27 coefficients from the raw fit and the current
        save-mix assumptions. Cheap, idempotent, and safe to call after changing
        ``gk_save_p_in_box`` / ``gk_save_p_big_chance``."""
        self.adjustments = _apply_2026_27_adjustments(
            self.fits, self.gk_save_p_in_box, self.gk_save_p_big_chance
        )
        self._sd_cache.clear()
        self._bps_cache.clear()
        return self.adjustments

    def refit_from_current_season(
        self, state: Any, min_rows: int = 400, write_report: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Refit the BPS regression on completed 2026/27 gameweeks.

        The manual rule patches exist only because no 2026/27 match had been
        played when this module was written. Once real rows exist the regression
        sees the new weights directly, so this **replaces** the patched
        coefficients rather than adjusting them again. Returns the fit summary,
        or None (with a warning recorded) when there is not enough data yet.
        """
        rows: List[pd.DataFrame] = []
        for pid, frame in (getattr(state, "player_history", {}) or {}).items():
            if frame is None or len(frame) == 0:
                continue
            sub = frame.copy()
            sub["element"] = pid
            sub["element_type"] = state.players[pid].position if pid in state.players else pd.NA
            rows.append(sub)
        if not rows:
            msg = "refit_from_current_season: no current-season history rows yet"
            log.info(msg)
            self.warnings.append(msg)
            return None
        current = pd.concat(rows, ignore_index=True)
        played = current[pd.to_numeric(current["minutes"], errors="coerce").fillna(0) > 0]
        if len(played) < min_rows:
            msg = "refit_from_current_season: only %d played rows (need %d)" % (
                len(played),
                min_rows,
            )
            log.info(msg)
            self.warnings.append(msg)
            return None

        fits = fit_bps_coefficients(current)
        if not fits:
            self.warnings.append("refit_from_current_season: no position had enough rows")
            return None
        # Live 2026/27 rows already embody the new rules: no patching.
        for fit in fits.values():
            fit.coefficients_2026_27 = dict(fit.coefficients)
        self.fits = fits
        self.adjustments = {
            "source": "refit from live %s data; no manual rule patches applied"
            % self.config.season
        }
        design = build_design_frame(current)
        design = design[design["minutes"] > 0]
        for position in scoring.POSITIONS:
            sub = design[pd.to_numeric(design["element_type"], errors="coerce") == position]
            if len(sub) >= 50:
                self.dispersion[position] = {
                    event: _pooled_dispersion(sub, event) for event in COUNT_EVENTS
                }
        self.source_season = hist.season_dash(self.config.season)
        self.fitted_from = "live %s element-summary history (%d played rows)" % (
            self.config.season,
            len(played),
        )
        self.state = state
        self._positions = {pid: p.position for pid, p in state.players.items()}
        self._build_player_rates(state)
        self._sd_cache.clear()
        self._bps_cache.clear()
        if write_report:
            self.write_report()
        return {
            "rows": int(len(played)),
            "positions": {
                scoring.POS_NAME[p]: {"n": f.n_rows, "r2": f.r2, "resid_sd": f.resid_sd}
                for p, f in sorted(fits.items())
            },
        }

    def _build_player_rates(self, state: Any) -> None:
        """Per-90 CBI / recoveries / tackles / saves / cards for every player.

        Pools current-season history with the prior season at the configured
        weights and shrinks towards the position baseline, so a promoted-club
        debutant gets the baseline rather than a silent zero.
        """
        w_prior = float(self.mc.weight_prior_season)
        w_current = float(self.mc.weight_season_to_date + self.mc.weight_recent_form)
        total = w_prior + w_current
        if total <= 0:
            w_prior, w_current = 0.5, 0.5
        else:
            w_prior, w_current = w_prior / total, w_current / total
        strength = float(self.mc.shrinkage_90s_defcon)
        events = COUNT_EVENTS + ("yellow_cards", "red_cards", "own_goals", "goals", "assists")

        self.player_rates = {}
        for pid, player in state.players.items():
            baseline = self.baselines.get(player.position, {})
            totals: Dict[str, float] = {e: 0.0 for e in events}
            minutes = 0.0

            current = state.history(pid) if hasattr(state, "history") else None
            if current is not None and len(current) > 0:
                cur_minutes = float(pd.to_numeric(current.get("minutes"), errors="coerce").fillna(0).sum())
                if cur_minutes > 0:
                    minutes += w_current * cur_minutes
                    for event in events:
                        column = _SOURCE_COLUMNS.get(event, event)
                        if column in current.columns:
                            totals[event] += w_current * float(
                                pd.to_numeric(current[column], errors="coerce").fillna(0).sum()
                            )

            row = state.prior_season_row(pid) if hasattr(state, "prior_season_row") else None
            if row is None and hasattr(state, "latest_season_row"):
                row = state.latest_season_row(pid)
            if row:
                prior_minutes = _finite(row.get("minutes"))
                if prior_minutes > 0:
                    minutes += w_prior * prior_minutes
                    for event in events:
                        column = _SOURCE_COLUMNS.get(event, event)
                        totals[event] += w_prior * _finite(row.get(column))

            n90 = minutes / 90.0
            rates: Dict[str, float] = {"n90": n90}
            for event in events:
                observed = (totals[event] / minutes * 90.0) if minutes > 0 else 0.0
                rates[event] = stats.shrink(observed, n90, baseline.get(event, 0.0), strength)
            self.player_rates[pid] = rates

    # -- reporting ----------------------------------------------------------

    def fitted_parameters(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_season": self.config.season,
            "source_season": self.source_season,
            "fitted_from": self.fitted_from,
            "fit_method": "sign-constrained NNLS (Lawson-Hanson), no intercept, "
            "played matches only",
            "features_positive": list(FEATURES_POSITIVE),
            "features_negative": list(FEATURES_NEGATIVE),
            "positions": {
                scoring.POS_NAME[p]: {
                    "n_rows": f.n_rows,
                    "r2": f.r2,
                    "resid_sd": f.resid_sd,
                    "resid_mean": f.resid_mean,
                    "mean_bps": f.mean_bps,
                    "sd_bps": f.sd_bps,
                    "coefficients_fitted": f.coefficients,
                    "coefficients_2026_27": f.coefficients_2026_27,
                }
                for p, f in sorted(self.fits.items())
            },
            "adjustments_2026_27": self.adjustments,
            "adjustment_notes": ADJUSTMENTS_2026_27,
            "count_dispersion": {
                scoring.POS_NAME[p]: v for p, v in sorted(self.dispersion.items())
            },
            "gk_save_marginal_diagnostic": self.gk_save_diagnostic,
            "opponent_attack_tilt_exponents": self.tilt,
            "position_baselines_per_90": {
                scoring.POS_NAME[p]: v for p, v in sorted(self.baselines.items())
            },
            "fallback_minutes_model": {
                "note": "only used when no MinutesForecast is supplied; xp.py "
                "overrides all of it",
                "starters_per_position": {
                    scoring.POS_NAME[p]: v for p, v in sorted(self.starters_per_position.items())
                },
                "p_60_given_start": {
                    scoring.POS_NAME[p]: v for p, v in sorted(self.start_to_60.items())
                },
                "price_to_p_start": {
                    scoring.POS_NAME[p]: v for p, v in sorted(self.price_start_curve.items())
                },
            },
            "caveats": [
                "Coefficients are 2025/26 fits with four documented 2026/27 rule "
                "patches applied. Call BonusModel.refit_from_current_season() "
                "once 2026/27 gameweeks complete.",
                "The removal of the -1 BPS for being tackled is unmodelled: no "
                "available dataset carries times-tackled.",
                "The goalkeeper save uplift assumes p(save in box)=%.2f and "
                "p(save from big chance)=%.2f. The rule bounds the nominal "
                "per-save value on [2.0, 4.0]." % (GK_SAVE_P_IN_BOX, GK_SAVE_P_BIG_CHANCE),
                "gk_save_marginal_diagnostic measures what a save was actually "
                "worth in the source season. If it sits well above "
                "GK_SAVE_NOMINAL_2526 (%.1f), the 2026/27 save uplift of x%.2f "
                "is too aggressive and keeper bonus here is over-projected."
                % (GK_SAVE_NOMINAL_2526,
                   (GK_SAVE_BASE_2627 + GK_SAVE_P_IN_BOX + GK_SAVE_P_BIG_CHANCE)
                   / GK_SAVE_NOMINAL_2526),
                "Per-fixture expected bonus is capped at 6.0. Real FPL pays more "
                "when BPS ties (observed 2025/26 mean 6.37, max 9); the model's "
                "continuous noise makes ties measure-zero, so 6.0 is a "
                "deliberate floor-of-reality, never an over-estimate.",
            ],
            "warnings": self.warnings,
        }

    def write_report(self, path: Optional[str] = None) -> str:
        ensure_dirs()
        path = path or BPS_FIT_PATH
        with open(path, "w") as fh:
            json.dump(self.fitted_parameters(), fh, indent=2, sort_keys=False)
        return path

    def coefficients_table(self) -> str:
        lines = []
        header = "%-24s %10s %10s %10s %10s" % ("feature", "GKP", "DEF", "MID", "FWD")
        lines.append(header)
        lines.append("-" * len(header))
        for feature in FEATURES:
            cells = []
            for position in scoring.POSITIONS:
                fit = self.fits.get(position)
                cells.append("%10.3f" % fit.weight(feature) if fit else "%10s" % "-")
            lines.append("%-24s %s" % (feature, " ".join(cells)))
        lines.append("-" * len(header))
        for label, attr, fmt in (
            ("R-squared", "r2", "%10.4f"),
            ("residual sd", "resid_sd", "%10.3f"),
            ("n rows", "n_rows", "%10d"),
        ):
            cells = []
            for position in scoring.POSITIONS:
                fit = self.fits.get(position)
                cells.append((fmt % getattr(fit, attr)) if fit else "%10s" % "-")
            lines.append("%-24s %s" % (label, " ".join(cells)))
        return "\n".join(lines)

    # -- projection ---------------------------------------------------------

    def _resolve_player(self, player: Any) -> Tuple[int, int]:
        """(player_id, position) from an id, a Player, or anything with `.id`."""
        if isinstance(player, Player):
            return int(player.id), int(player.position)
        if isinstance(player, (int, np.integer)):
            pid = int(player)
            position = self._positions.get(pid)
            if position is None and self.state is not None:
                position = self.state.players[pid].position
            if position is None:
                raise KeyError("unknown player id %d and no GameState to look it up" % pid)
            return pid, int(position)
        pid = int(getattr(player, "id"))
        return pid, int(getattr(player, "position", self._positions.get(pid, scoring.MID)))

    @staticmethod
    def _ctx_value(ctx: Any, *names: str, **kwargs: Any) -> Any:
        default = kwargs.get("default")
        if ctx is None:
            return default
        for name in names:
            if isinstance(ctx, Mapping):
                if name in ctx and ctx[name] is not None:
                    return ctx[name]
            else:
                value = getattr(ctx, name, None)
                if value is not None:
                    return value
        return default

    def _minutes_from(self, ctx: Any, projections: Any, pid: int) -> Tuple[float, float, float]:
        """(p_appear, p_60_unconditional, xmins) from whatever the caller gave us."""
        source = self._ctx_value(ctx, "minutes")
        p_appear = self._ctx_value(projections, "p_appear")
        p_60 = self._ctx_value(projections, "p_60")
        xmins = self._ctx_value(projections, "xmins")
        if p_appear is None:
            p_appear = self._ctx_value(source, "p_appear")
        if p_60 is None:
            p_60 = self._ctx_value(source, "p_60")
        if xmins is None:
            xmins = self._ctx_value(source, "xmins")
        if p_appear is None and xmins is None:
            p_appear, p_60, xmins = self._fallback_minutes(pid)
        if xmins is None:
            p_a = _clip01(_finite(p_appear))
            p_6 = _clip01(_finite(p_60)) if p_60 is not None else p_a * 0.75
            xmins = p_6 * self.mc.starter_minutes + (p_a - p_6) * _SUB_MINUTES_DEFAULT
        if p_appear is None:
            p_appear = _clip01(_finite(xmins) / self.mc.starter_minutes)
        if p_60 is None:
            # Without an explicit p_60 assume the expected minutes come from a
            # start; substitutes rarely reach 60 so this is the optimistic end.
            p_60 = _clip01(_finite(xmins) / self.mc.starter_minutes)
        p_appear = _clip01(_finite(p_appear))
        p_60 = min(_clip01(_finite(p_60)), p_appear)
        return p_appear, p_60, max(0.0, _finite(xmins))

    def price_start_prior(self, position: int, price: float) -> float:
        """P(start) implied by price alone, interpolated on the fitted curve."""
        points = self.price_start_curve.get(position)
        if not points:
            return 0.5
        if price <= points[0][0]:
            return points[0][1]
        if price >= points[-1][0]:
            return points[-1][1]
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            if x0 <= price <= x1:
                if x1 - x0 < 1e-9:
                    return y1
                w = (price - x0) / (x1 - x0)
                return y0 + w * (y1 - y0)
        return points[-1][1]

    def _build_fallback_minutes(self, state: Any) -> None:
        """Squad-consistent minutes for every player, used when nobody supplies any.

        Prior-season start share shrunk towards the fitted price-to-start curve,
        hard-gated on availability status, then **normalised within each club**:
        exactly one goalkeeper and exactly ten outfielders start every match.
        Without that constraint three Arsenal keepers each carry a start
        probability and all three collect bonus, and the promoted clubs - whose
        players have no top-flight history and therefore no observed rate at all
        - either vanish from the fixture or double-count.

        This exists so bonus.py can be run and audited on its own. ``xp.py``
        always passes a real ``MinutesForecast`` and this path is then dead.
        """
        raw: Dict[int, float] = {}
        sub_rate: Dict[int, float] = {}
        for pid, player in state.players.items():
            multiplier = self.mc.status_multiplier.get(player.status, 1.0)
            if player.status == "d" and player.chance_of_playing_next_round is not None:
                multiplier = player.chance_of_playing_next_round / 100.0
            prior = self.price_start_prior(player.position, player.price)
            row = state.prior_season_row(pid) or state.latest_season_row(pid) or {}
            minutes = _finite(row.get("minutes"))
            starts = _finite(row.get("starts"))
            observed = _clip01(starts / 38.0)
            raw[pid] = multiplier * _clip01(
                stats.shrink(observed, minutes / 90.0, prior, self.mc.shrinkage_90s_minutes)
            )
            appearances = max(starts, minutes / 90.0)
            sub_rate[pid] = min(0.45, appearances / 38.0) if appearances else 0.25

        by_team: Dict[int, Dict[int, List[int]]] = {}
        for pid, player in state.players.items():
            by_team.setdefault(player.team_id, {}).setdefault(player.position, []).append(pid)

        self.fallback_minutes = {}
        for team_id, groups in by_team.items():
            for position, pids in groups.items():
                capacity = self.starters_per_position.get(position, 0.0)
                # Waterfall rather than proportional scaling: the first-choice
                # keeper takes his whole probability and the backups get the
                # remainder, instead of three keepers each ending up on a third
                # of a start. Proportional scaling destroys the price ordering
                # exactly where it is most informative.
                remaining = capacity
                for pid in sorted(pids, key=lambda p: (-raw[p], -state.players[p].now_cost, p)):
                    p_start = _clip01(min(raw[pid], max(remaining, 0.0)))
                    remaining -= p_start
                    p_appear = _clip01(
                        p_start + (1.0 - p_start) * (sub_rate[pid] if raw[pid] > 0 else 0.0)
                    )
                    p_60 = min(p_appear, p_start * self.start_to_60.get(position, 0.8))
                    xmins = (
                        p_60 * self.mc.starter_minutes
                        + (p_appear - p_60) * _SUB_MINUTES_DEFAULT
                    )
                    self.fallback_minutes[pid] = (p_appear, p_60, xmins)

    def _fallback_minutes(self, pid: int) -> Tuple[float, float, float]:
        cached = self.fallback_minutes.get(pid)
        if cached is not None:
            return cached
        if self.state is None:
            return 0.8, 0.7, 68.0
        return 0.0, 0.0, 0.0

    def _defensive_tilt(self, action: str, opponent_lambda: float) -> float:
        """Opponent-attack multiplier on a defensive-action rate."""
        exponent = self.tilt.get(action, 0.0)
        if not exponent:
            return 1.0
        ratio = max(0.2, min(3.0, opponent_lambda / max(self.league_lambda, 0.2)))
        return float(ratio ** exponent)

    def resolve_events(
        self, player: Any, fixture_ctx: Any = None, projections: Any = None
    ) -> EventProjection:
        """Turn whatever the caller supplied into a full event projection.

        Anything present in ``projections`` wins. Anything missing is filled
        from this module's own fitted per-90 rates and the fixture lambdas, so a
        partial projection (say goals and assists only) still produces a
        complete BPS number instead of silently dropping the defensive half.
        """
        pid, position = self._resolve_player(player)
        p_appear, p_60_uncond, xmins = self._minutes_from(fixture_ctx, projections, pid)
        rates = self.player_rates.get(pid, {})
        baseline = self.baselines.get(position, {})

        opponent_lambda = _finite(
            self._ctx_value(fixture_ctx, "opponent_lambda", default=self.league_lambda),
            self.league_lambda,
        )
        lam_conceded = _finite(
            self._ctx_value(projections, "lambda_conceded", default=opponent_lambda),
            opponent_lambda,
        )

        events = EventProjection(p_appear=p_appear, lambda_conceded=lam_conceded)
        # Everything below is conditional on appearing; the p_appear factor is
        # applied once at the end so the mixture variance stays exact.
        p_a = max(p_appear, _EPS)
        events.p_60 = _clip01(p_60_uncond / p_a)
        share = (xmins / 90.0) / p_a if p_a > 0 else 0.0
        share = max(0.0, min(1.3, share))

        def per90(name: str) -> float:
            return _finite(rates.get(name, baseline.get(name, 0.0)))

        def supplied(*names: str) -> Optional[float]:
            value = self._ctx_value(projections, *names)
            return None if value is None else _finite(value)

        goals = supplied("lambda_goals", "goals")
        events.goals = (goals / p_a) if goals is not None else per90("goals") * share
        assists = supplied("lambda_assists", "assists")
        events.assists = (assists / p_a) if assists is not None else per90("assists") * share

        # Contract with defending.py (SPEC 3.5): `p_clean_sheet` is the *team*
        # probability, ungated by minutes - xp.py applies the 60-minute gate at
        # points time via scoring.clean_sheet_points. BPS uses the same rule, so
        # gate it here and do not divide by p_appear: it is not a player rate.
        cs = supplied("p_clean_sheet", "clean_sheet")
        if cs is None:
            cs = stats.p_clean_sheet(lam_conceded)
        events.clean_sheet = _clip01(cs) * events.p_60

        saves = supplied("lambda_saves", "saves")
        if position == scoring.GKP:
            if saves is not None:
                events.saves = saves / p_a
            else:
                events.saves = per90("saves") * share * self._defensive_tilt(
                    "saves", opponent_lambda
                )
        else:
            events.saves = 0.0

        for action in ("cbi", "recoveries", "tackles"):
            value = supplied(
                "lambda_%s" % action,
                action,
                "clearances_blocks_interceptions" if action == "cbi" else action,
            )
            if value is not None:
                setattr(events, action, value / p_a)
            else:
                setattr(
                    events,
                    action,
                    per90(action) * share * self._defensive_tilt(action, opponent_lambda),
                )

        events.goals_conceded_pairs = _expected_conceded_pairs(lam_conceded)

        yellow = supplied("p_yellow", "yellow_cards")
        events.yellow_cards = (
            (yellow / p_a) if yellow is not None else min(1.0, per90("yellow_cards") * share)
        )
        red = supplied("p_red", "red_cards")
        events.red_cards = (red / p_a) if red is not None else min(1.0, per90("red_cards") * share)
        own = supplied("p_own_goal", "own_goals")
        events.own_goals = (own / p_a) if own is not None else min(1.0, per90("own_goals") * share)
        events.penalties_saved = _finite(supplied("lambda_pen_saves", "penalties_saved") or 0.0) / p_a
        events.penalties_missed = _finite(supplied("lambda_pen_miss", "penalties_missed") or 0.0) / p_a
        return events

    def project_bps_and_sd(
        self, player: Any, fixture_ctx: Any = None, projections: Any = None
    ) -> Tuple[float, float]:
        """(expected BPS, standard deviation of BPS) for one player-fixture.

        The variance is built from the events themselves - Poisson for goals and
        assists, Bernoulli for the clean sheet and the minutes bucket, negative
        binomial (dispersion fitted per position) for the defensive counts -
        plus the regression residual variance for everything BPS rewards that we
        cannot see (passes, key passes, big chances, dribbles, fouls). A missed
        appearance is folded in as a mixture with a point mass at zero, which is
        what lets a bench player retain a small, honest chance of stealing bonus.
        """
        pid, position = self._resolve_player(player)
        fit = self.fits.get(position)
        if fit is None:
            return 0.0, 1.0
        events = self.resolve_events(player, fixture_ctx, projections)
        counts = events.counts()
        weights = fit.coefficients_2026_27

        mu_cond = 0.0
        for feature, value in counts.items():
            mu_cond += weights.get(feature, 0.0) * value

        dispersion = self.dispersion.get(position, {})
        var_cond = fit.resid_sd ** 2
        # Bernoulli minutes bucket: 1-59 vs 60+, a single coin flip between two
        # weights, so the variance is (w60 - w159)^2 * p(1-p).
        p60 = _clip01(events.p_60)
        var_cond += (weights.get("mins_60", 0.0) - weights.get("mins_1_59", 0.0)) ** 2 * p60 * (1 - p60)
        var_cond += weights.get("goals", 0.0) ** 2 * max(0.0, events.goals)
        var_cond += weights.get("assists", 0.0) ** 2 * max(0.0, events.assists)
        p_cs = _clip01(events.clean_sheet)
        var_cond += weights.get("clean_sheet", 0.0) ** 2 * p_cs * (1 - p_cs)
        for action in COUNT_EVENTS:
            mean = max(0.0, getattr(events, action))
            var_cond += weights.get(action, 0.0) ** 2 * _negbin_variance(
                mean, dispersion.get(action, 1e6)
            )
        var_cond += weights.get("goals_conceded_pairs", 0.0) ** 2 * _var_conceded_pairs(
            events.lambda_conceded
        )
        p_y = _clip01(events.yellow_cards)
        var_cond += weights.get("yellow_cards", 0.0) ** 2 * p_y * (1 - p_y)
        p_r = _clip01(events.red_cards)
        var_cond += weights.get("red_cards", 0.0) ** 2 * p_r * (1 - p_r)

        p_a = _clip01(events.p_appear)
        mu = p_a * mu_cond
        var = p_a * var_cond + p_a * (1.0 - p_a) * mu_cond * mu_cond
        sd = math.sqrt(max(var, 1e-6))
        mu = _finite(mu)
        sd = max(_finite(sd, 1.0), 0.5)
        self._bps_cache[pid] = mu
        self._sd_cache[pid] = sd
        return mu, sd

    def project_bps(
        self, player: Any, fixture_ctx: Any = None, projections: Any = None
    ) -> float:
        """Expected BPS for one player in one fixture."""
        return self.project_bps_and_sd(player, fixture_ctx, projections)[0]

    def bps_sd(self, player_id: int) -> float:
        """Standard deviation cached by the last ``project_bps`` for this player."""
        cached = self._sd_cache.get(int(player_id))
        if cached is not None:
            return cached
        position = self._positions.get(int(player_id), scoring.MID)
        fit = self.fits.get(position)
        return fit.resid_sd if fit else 8.0

    # -- bonus --------------------------------------------------------------

    def project_bonus(
        self,
        fixture_id: int,
        candidates: Dict[int, float],
        sds: Optional[Dict[int, float]] = None,
        iterations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[int, float]:
        """``{player_id: expected bonus}`` for one fixture.

        ``candidates`` maps player id to projected BPS and **must contain every
        player who might appear, from both teams**. Bonus is contested across
        the match, not within a squad.

        Read this before using the result: the simulator awards exactly one
        1st, 2nd and 3rd place per draw, so the returned values sum to exactly
        6.0 for *any* candidate list of three or more. An incomplete list does
        not return less bonus - it **redistributes** the missing players' share
        onto the ones you did pass, which is the single easiest way to inflate
        every projection in the app. ``candidates_for_fixture`` builds the full
        list; use it unless you have a reason not to.
        """
        if not candidates:
            return {}
        if len(candidates) < 22:
            log.warning(
                "fixture %s: only %d bonus candidates - a real match has 22+ players "
                "and a short list over-allocates bonus to those present",
                fixture_id,
                len(candidates),
            )
        ids = sorted(candidates.keys())
        scores = {pid: _finite(candidates[pid]) for pid in ids}
        sigma: Dict[int, float] = {}
        for pid in ids:
            value = None
            if sds is not None:
                value = sds.get(pid)
            if value is None:
                value = self.bps_sd(pid)
            sigma[pid] = max(_finite(value, 1.0), 0.5)

        iterations = int(iterations or self.mc.bps_sim_iterations)
        # Deterministic per fixture: the same fixture always simulates the same
        # draws, so a re-run of the projection set is bit-identical.
        seed = int(seed if seed is not None else (7 + int(fixture_id)))
        ranks = stats.top_k_probabilities(scores, sigma, k=3, iterations=iterations, seed=seed)

        bonus = {
            pid: scoring.expected_bonus_points(p[0], p[1], p[2]) for pid, p in ranks.items()
        }
        total = sum(bonus.values())
        cap = float(sum(scoring.BONUS_TIERS))
        if total > cap + 1e-6:
            # Unreachable with the Monte-Carlo ranker (each draw awards exactly
            # one 1st, 2nd and 3rd) but the cap is a hard invariant of the game,
            # so enforce it rather than trust the sampler.
            factor = cap / total
            bonus = {pid: value * factor for pid, value in bonus.items()}
            log.warning("fixture %s: bonus sum %.4f rescaled to %.1f", fixture_id, total, cap)
        return bonus

    # -- convenience --------------------------------------------------------

    def candidates_for_fixture(
        self,
        state: Any,
        fixture: Fixture,
        forecast: Optional[Any] = None,
        projections: Optional[Mapping[int, Any]] = None,
        min_p_appear: float = 0.02,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Projected BPS and sd for every player in a fixture, both teams.

        ``projections`` is an optional ``{player_id: projection}`` map from
        ``xp.py``; anything missing falls back to this module's own rates.
        Returns ``(bps, sd)`` ready to hand to ``project_bonus``.
        """
        lam_h, lam_a = self._fixture_lambdas(fixture, forecast)
        bps: Dict[int, float] = {}
        sds: Dict[int, float] = {}
        for team_id, is_home in ((fixture.team_h, True), (fixture.team_a, False)):
            team_lambda = lam_h if is_home else lam_a
            opponent_lambda = lam_a if is_home else lam_h
            squad = self._by_team.get(team_id)
            if squad is None:
                squad = [p for p, pl in state.players.items() if pl.team_id == team_id]
            for pid in squad:
                player = state.players[pid]
                ctx = BonusContext(
                    fixture=fixture,
                    player=player,
                    is_home=is_home,
                    team_lambda=team_lambda,
                    opponent_lambda=opponent_lambda,
                    forecast=forecast,
                    minutes=None,
                )
                projection = projections.get(pid) if projections else None
                value, sd = self.project_bps_and_sd(player, ctx, projection)
                p_appear = self._minutes_from(ctx, projection, pid)[0]
                if p_appear < min_p_appear:
                    continue
                bps[pid] = value
                sds[pid] = sd
        return bps, sds

    def _fixture_lambdas(
        self, fixture: Fixture, forecast: Optional[Any]
    ) -> Tuple[float, float]:
        if forecast is not None:
            lam_h = _finite(getattr(forecast, "lambda_h", None), self.league_lambda)
            lam_a = _finite(getattr(forecast, "lambda_a", None), self.league_lambda)
            return lam_h, lam_a
        # No team model supplied: fall back to the official per-fixture FDR,
        # which is the only pre-season signal the API populates. 3 is neutral.
        base = self.league_lambda
        h = base * (1.0 + 0.10 * (3 - fixture.team_h_difficulty))
        a = base * (1.0 + 0.10 * (3 - fixture.team_a_difficulty))
        return max(0.2, h * float(self.mc.home_advantage)), max(0.2, a)

    def fixture_bonus(
        self,
        state: Any,
        fixture: Fixture,
        forecast: Optional[Any] = None,
        projections: Optional[Mapping[int, Any]] = None,
    ) -> Dict[int, float]:
        """End-to-end: candidates from both teams, then expected bonus."""
        bps, sds = self.candidates_for_fixture(state, fixture, forecast, projections)
        return self.project_bonus(fixture.id, bps, sds=sds)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def backtest_bonus(
    model: BonusModel, frame: pd.DataFrame, use_patched: bool = False
) -> Dict[str, Any]:
    """Replay a completed season's *observed* event counts through the ranker.

    This deliberately feeds in real event counts rather than projected ones, so
    it isolates the half of the module that is testable today: BPS weights ->
    noise -> P(1st/2nd/3rd) -> expected bonus. The event projections themselves
    are attacking.py, defending.py and minutes.py's to answer, and their errors
    would drown this signal out.

    The weights are in-sample, so treat the BPS correlation as an upper bound.
    The parts that are *not* in-sample and are the real subject here are the
    noise calibration (does residual_sd spread the bonus correctly across a
    match?) and the positional share of bonus.
    """
    design = build_design_frame(frame)
    design = design[design["minutes"] > 0].copy()
    weights_attr = "coefficients_2026_27" if use_patched else "coefficients"

    predicted = np.zeros(len(design))
    sd = np.full(len(design), 8.0)
    positions = pd.to_numeric(design["element_type"], errors="coerce").fillna(0).astype(int).values
    for position, fit in model.fits.items():
        mask = positions == position
        if not mask.any():
            continue
        weights = getattr(fit, weights_attr)
        contribution = np.zeros(int(mask.sum()))
        sub = design[mask]
        for feature in FEATURES:
            contribution += weights.get(feature, 0.0) * sub[feature].astype(float).values
        predicted[mask] = contribution
        sd[mask] = fit.resid_sd
    design["pred_bps"] = predicted
    design["pred_sd"] = sd

    rows: List[Tuple[int, float, float, int]] = []
    hits = 0
    fixtures = 0
    for fixture_id, grp in design.groupby("fixture"):
        candidates = dict(zip(grp["element"].astype(int), grp["pred_bps"].astype(float)))
        sds = dict(zip(grp["element"].astype(int), grp["pred_sd"].astype(float)))
        expected = model.project_bonus(int(fixture_id), candidates, sds=sds)
        actual = dict(zip(grp["element"].astype(int), grp["bonus"].astype(float)))
        positions_here = dict(zip(grp["element"].astype(int), grp["element_type"].astype(int)))
        for pid, value in expected.items():
            rows.append((pid, value, actual.get(pid, 0.0), positions_here[pid]))
        fixtures += 1
        top_pred = max(expected, key=lambda k: expected[k])
        if actual.get(top_pred, 0.0) >= 3.0:
            hits += 1

    pred_values = [r[1] for r in rows]
    actual_values = [r[2] for r in rows]
    by_position: Dict[str, Dict[str, float]] = {}
    for position in scoring.POSITIONS:
        sel = [r for r in rows if r[3] == position]
        if not sel:
            continue
        by_position[scoring.POS_NAME[position]] = {
            "predicted_total": float(sum(r[1] for r in sel)),
            "actual_total": float(sum(r[2] for r in sel)),
            "predicted_share": float(sum(r[1] for r in sel) / max(sum(pred_values), 1e-9)),
            "actual_share": float(sum(r[2] for r in sel) / max(sum(actual_values), 1e-9)),
        }
    bps_pred = design["pred_bps"].values
    bps_actual = design["bps"].astype(float).values
    return {
        "fixtures": fixtures,
        "player_matches": len(rows),
        "bps_rmse": stats.rmse(bps_pred, bps_actual),
        "bps_spearman": stats.spearman(bps_pred, bps_actual),
        "bonus_rmse": stats.rmse(pred_values, actual_values),
        "bonus_mae": stats.mae(pred_values, actual_values),
        "bonus_spearman": stats.spearman(pred_values, actual_values),
        "predicted_bonus_total": float(sum(pred_values)),
        "actual_bonus_total": float(sum(actual_values)),
        "top1_hit_rate": hits / max(fixtures, 1),
        "by_position": by_position,
    }


# ---------------------------------------------------------------------------
# Self-test. tests/ is owned by another module, so the invariants that matter
# live here and run from the smoke test.
# ---------------------------------------------------------------------------


def self_test(model: BonusModel, state: Any, fixtures: Sequence[Fixture],
              forecasts: Optional[Mapping[int, Any]] = None) -> List[str]:
    """Assert every invariant this module must never break."""
    checks: List[str] = []
    cap = float(sum(scoring.BONUS_TIERS))

    for fixture in fixtures:
        forecast = forecasts.get(fixture.id) if forecasts else None
        bps, sds = model.candidates_for_fixture(state, fixture, forecast)
        assert len(bps) >= 22, "fixture %d: only %d candidates" % (fixture.id, len(bps))
        for pid, value in bps.items():
            assert value == value, "NaN BPS for player %d" % pid
            assert sds[pid] > 0 and sds[pid] == sds[pid], "bad sd for player %d" % pid
        bonus = model.project_bonus(fixture.id, bps, sds=sds)
        total = sum(bonus.values())
        assert all(v == v for v in bonus.values()), "NaN bonus in fixture %d" % fixture.id
        assert all(v >= -1e-12 for v in bonus.values()), "negative bonus in fixture %d" % fixture.id
        assert total <= cap + 1e-6, "fixture %d: bonus sums to %.6f > %.1f" % (
            fixture.id,
            total,
            cap,
        )
        assert total > cap - 1e-6, (
            "fixture %d: full candidate list sums to %.6f, expected exactly %.1f"
            % (fixture.id, total, cap)
        )
    checks.append(
        "%d fixtures: >=22 candidates, no NaN, expected bonus sums to exactly 6.0" % len(fixtures)
    )

    # A short candidate list still conserves 6.0 - it redistributes rather than
    # shrinks. Pin that down so nobody later "optimises" by passing a top-N.
    fixture = fixtures[0]
    bps, sds = model.candidates_for_fixture(state, fixture, forecasts.get(fixture.id) if forecasts else None)
    full = model.project_bonus(fixture.id, bps, sds=sds)
    top = dict(sorted(bps.items(), key=lambda kv: -kv[1])[:5])
    subset = model.project_bonus(fixture.id, top, sds=sds)
    assert sum(subset.values()) <= cap + 1e-6
    best = max(top, key=lambda k: top[k])
    assert subset[best] > full[best], "a short list must over-allocate, not under-allocate"
    checks.append(
        "top-5 subset also sums to %.3f and inflates the leader %.3f -> %.3f "
        "(conservation is why the full list is mandatory)"
        % (sum(subset.values()), full[best], subset[best])
    )

    # Degenerate candidate lists must not blow up or exceed the cap.
    assert model.project_bonus(1, {}) == {}
    two = model.project_bonus(1, {101: 30.0, 102: 20.0}, sds={101: 5.0, 102: 5.0})
    assert sum(two.values()) <= 5.0 + 1e-6, sum(two.values())
    one = model.project_bonus(1, {101: 30.0}, sds={101: 5.0})
    assert abs(sum(one.values()) - 3.0) < 1e-6, sum(one.values())
    checks.append("degenerate lists: 0 -> {}, 1 -> 3.0, 2 -> <=5.0")

    # Determinism.
    a = model.project_bonus(fixture.id, bps, sds=sds)
    b = model.project_bonus(fixture.id, bps, sds=sds)
    assert a == b, "project_bonus is not deterministic"
    checks.append("project_bonus deterministic across repeated calls")

    # Monotonicity: raising one player's BPS must not lower his bonus.
    probe = dict(bps)
    best = max(probe, key=lambda k: probe[k])
    before = model.project_bonus(fixture.id, probe, sds=sds)[best]
    probe[best] += 15.0
    after = model.project_bonus(fixture.id, probe, sds=sds)[best]
    assert after >= before - 1e-9, "bonus fell when BPS rose (%.4f -> %.4f)" % (before, after)
    checks.append("monotone in BPS (%.3f -> %.3f on a +15 BPS bump)" % (before, after))

    # Sign constraints survived the fit and the patches.
    for position, fit in model.fits.items():
        for feature in FEATURES_POSITIVE:
            assert fit.coefficients_2026_27.get(feature, 0.0) >= -1e-9, (
                "%s %s went negative" % (scoring.POS_NAME[position], feature)
            )
        for feature in FEATURES_NEGATIVE:
            assert fit.coefficients_2026_27.get(feature, 0.0) <= 1e-9, (
                "%s %s went positive" % (scoring.POS_NAME[position], feature)
            )
    checks.append("all fitted coefficients respect their sign constraints")
    return checks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from gaffer.data.loaders import fixtures_by_gw, load_game_state

    cfg = Config.load()
    t0 = time.time()
    game = load_game_state(cfg)
    print("loaded state in %.2fs -> %s" % (time.time() - t0, game.summary()))

    t0 = time.time()
    bonus_model = BonusModel(cfg)
    bonus_model.fit(game)
    print("fitted BPS model in %.2fs from %s" % (time.time() - t0, bonus_model.fitted_from))

    print("\n=== fitted BPS coefficients (2026/27, after rule patches) ===")
    print(bonus_model.coefficients_table())

    print("\n=== as fitted on 2025/26 vs patched for 2026/27 ===")
    for pos in scoring.POSITIONS:
        f = bonus_model.fits.get(pos)
        if not f:
            continue
        changed = [
            (k, f.coefficients[k], f.coefficients_2026_27[k])
            for k in FEATURES
            if abs(f.coefficients[k] - f.coefficients_2026_27[k]) > 1e-9
        ]
        if changed:
            print(
                "  %-4s %s"
                % (
                    scoring.POS_NAME[pos],
                    ", ".join("%s %.3f -> %.3f" % c for c in changed),
                )
            )
    print("  official-value sanity: fitted GKP penalties_saved = %.2f (official 2025/26 value 8)"
          % bonus_model.fits[scoring.GKP].coefficients["penalties_saved"])
    print("  official-value sanity: fitted mins_1_59 / mins_60 = %s (official 3 / 6)"
          % ", ".join("%s %.2f/%.2f" % (scoring.POS_NAME[p],
                                        bonus_model.fits[p].coefficients["mins_1_59"],
                                        bonus_model.fits[p].coefficients["mins_60"])
                      for p in scoring.POSITIONS if p in bonus_model.fits))
    print("  opponent-attack tilt exponents: %s"
          % ", ".join("%s %+.3f" % (k, v) for k, v in sorted(bonus_model.tilt.items())))
    print("  count dispersion (DEF): %s"
          % ", ".join("%s %.2f" % (k, v) for k, v in sorted(bonus_model.dispersion[scoring.DEF].items())))
    print("  report written to %s" % bonus_model.write_report())

    # Team lambdas from the real ratings model.
    from gaffer.model.team_ratings import build_team_ratings

    t0 = time.time()
    ratings = build_team_ratings(game, cfg)
    gw = game.current_gw
    gw_fixtures = fixtures_by_gw(game)[gw]
    match_forecasts = ratings.forecast_all(game, [gw])
    bonus_model.league_lambda = float(
        np.mean([f.lambda_h for f in match_forecasts.values()]
                + [f.lambda_a for f in match_forecasts.values()])
    )
    print("\nteam ratings fitted in %.2fs; league mean lambda %.3f"
          % (time.time() - t0, bonus_model.league_lambda))

    print("\n=== GW%d fixture detail: %s v %s ===" % (
        gw, game.short_name(gw_fixtures[0].team_h), game.short_name(gw_fixtures[0].team_a)))
    target = gw_fixtures[0]
    fc = match_forecasts[target.id]
    print("  model lambdas: %s %.2f - %.2f %s"
          % (game.short_name(target.team_h), fc.lambda_h, fc.lambda_a, game.short_name(target.team_a)))
    cand_bps, cand_sd = bonus_model.candidates_for_fixture(game, target, fc)
    cand_bonus = bonus_model.project_bonus(target.id, cand_bps, sds=cand_sd)
    print("  %d candidates" % len(cand_bps))
    print("  %-20s %-4s %-4s %8s %7s %8s" % ("player", "team", "pos", "xBPS", "sd", "xBonus"))
    for pid in sorted(cand_bonus, key=lambda p: -cand_bonus[p]):
        p = game.player(pid)
        print("  %-20s %-4s %-4s %8.2f %7.2f %8.4f"
              % (p.web_name[:20], game.short_name(p.team_id), game.position_name(pid),
                 cand_bps[pid], cand_sd[pid], cand_bonus[pid]))
    print("  %-20s %-4s %-4s %8s %7s %8.4f  <- must be <= 6.0"
          % ("SUM", "", "", "", "", sum(cand_bonus.values())))

    print("\n=== GW%d: expected bonus, all %d fixtures ===" % (gw, len(gw_fixtures)))
    all_bonus: Dict[int, float] = {}
    all_bps: Dict[int, float] = {}
    for f in gw_fixtures:
        b, s = bonus_model.candidates_for_fixture(game, f, match_forecasts[f.id])
        got = bonus_model.project_bonus(f.id, b, sds=s)
        all_bps.update(b)
        all_bonus.update(got)
        print("  %-3s v %-3s  candidates=%3d  sum=%.6f"
              % (game.short_name(f.team_h), game.short_name(f.team_a), len(b), sum(got.values())))

    print("\n=== top 30 by expected bonus, GW%d ===" % gw)
    print("  %-3s %-20s %-4s %-4s %6s %8s %8s" % ("#", "player", "team", "pos", "price", "xBPS", "xBonus"))
    for rank, pid in enumerate(sorted(all_bonus, key=lambda p: -all_bonus[p])[:30], 1):
        p = game.player(pid)
        print("  %-3d %-20s %-4s %-4s %6.1f %8.2f %8.3f"
              % (rank, p.web_name[:20], game.short_name(p.team_id), game.position_name(pid),
                 p.price, all_bps[pid], all_bonus[pid]))

    print("\n=== goalkeeper save-uplift sensitivity (the model's biggest unknown) ===")
    diag = bonus_model.gk_save_diagnostic
    print("  DIAGNOSTIC: %s marginal BPS per save = %.2f +/- %.2f (n=%d, 90-min keeper "
          "matches, clean sheet and goals conceded controlled)"
          % (bonus_model.source_season, diag["marginal_bps_per_save"], diag["std_error"],
             diag["n"]))
    print("  The uplift below divides by an assumed 2025/26 nominal of %.1f. The diagnostic "
          "says a save was really worth ~%.1f, so if the official table already paid more "
          "than 2, this model over-projects keeper bonus by roughly x%.2f. Re-check against "
          "GW1-5 actuals before trusting keeper bonus."
          % (GK_SAVE_NOMINAL_2526, diag["marginal_bps_per_save"],
             diag["marginal_bps_per_save"] / GK_SAVE_NOMINAL_2526))
    keepers = [pid for pid in all_bonus if game.player(pid).position == scoring.GKP]
    keepers.sort(key=lambda p: -all_bonus[p])
    base_scale = bonus_model.adjustments["gk_save_scale"]
    print("  nominal per-save BPS 2026/27 = %.2f (scale %.2f on the fitted %.2f)"
          % (bonus_model.adjustments["gk_save_nominal_2627"], base_scale,
             bonus_model.fits[scoring.GKP].coefficients["saves"]))
    top_keepers = keepers[:5]
    scenarios = (("floor 2.0", 0.0, 0.0), ("default 3.0", GK_SAVE_P_IN_BOX,
                                           GK_SAVE_P_BIG_CHANCE), ("ceiling 4.0", 1.0, 1.0))
    print("  %-20s" % "keeper" + "".join("%13s" % name for name, _, _ in scenarios))
    rows: Dict[int, List[float]] = {pid: [] for pid in top_keepers}
    for _, p_box, p_big in scenarios:
        bonus_model.gk_save_p_in_box, bonus_model.gk_save_p_big_chance = p_box, p_big
        bonus_model.apply_rule_adjustments()
        scenario: Dict[int, float] = {}
        for f in gw_fixtures:
            b, s = bonus_model.candidates_for_fixture(game, f, match_forecasts[f.id])
            scenario.update(bonus_model.project_bonus(f.id, b, sds=s))
        for pid in top_keepers:
            rows[pid].append(scenario.get(pid, 0.0))
    bonus_model.gk_save_p_in_box = GK_SAVE_P_IN_BOX
    bonus_model.gk_save_p_big_chance = GK_SAVE_P_BIG_CHANCE
    bonus_model.apply_rule_adjustments()
    for pid in top_keepers:
        print("  %-20s" % game.player(pid).web_name[:20] + "".join("%13.3f" % v for v in rows[pid]))

    print("\n=== backtest: 2025/26 observed events -> ranker -> bonus ===")
    t0 = time.time()
    bt = backtest_bonus(bonus_model, hist.load_vaastav_gws(game.prior_season))
    print("  %d fixtures, %d player-matches (%.1fs)"
          % (bt["fixtures"], bt["player_matches"], time.time() - t0))
    print("  BPS   rmse=%.2f spearman=%.3f" % (bt["bps_rmse"], bt["bps_spearman"]))
    print("  bonus rmse=%.3f mae=%.3f spearman=%.3f" % (
        bt["bonus_rmse"], bt["bonus_mae"], bt["bonus_spearman"]))
    print("  predicted bonus total %.0f vs actual %.0f (%.1f%% - real FPL pays extra on "
          "BPS ties, which the model deliberately does not)"
          % (bt["predicted_bonus_total"], bt["actual_bonus_total"],
             100.0 * bt["predicted_bonus_total"] / bt["actual_bonus_total"]))
    print("  top-1 hit rate %.3f (model's top pick actually took 3 bonus)" % bt["top1_hit_rate"])
    print("  %-5s %12s %12s" % ("pos", "pred share", "actual share"))
    for pos_name, values in bt["by_position"].items():
        print("  %-5s %11.1f%% %11.1f%%"
              % (pos_name, 100 * values["predicted_share"], 100 * values["actual_share"]))
    # Same events, same fixtures, only the coefficients change: this isolates
    # what the four 2026/27 rule patches actually do to the bonus distribution.
    patched_bt = backtest_bonus(bonus_model, hist.load_vaastav_gws(game.prior_season),
                                use_patched=True)
    gw_share: Dict[int, float] = {}
    for pid, value in all_bonus.items():
        gw_share[game.player(pid).position] = gw_share.get(game.player(pid).position, 0.0) + value
    gw_total = sum(gw_share.values()) or 1.0
    print("\n  bonus share by position - what the rule change does")
    print("  %-5s %14s %14s %14s" % ("pos", "2025/26 actual", "2026/27 rules", "GW1 forecast"))
    for position in scoring.POSITIONS:
        name = scoring.POS_NAME[position]
        if name not in bt["by_position"]:
            continue
        print("  %-5s %13.1f%% %13.1f%% %13.1f%%"
              % (name,
                 100 * bt["by_position"][name]["actual_share"],
                 100 * patched_bt["by_position"][name]["predicted_share"],
                 100 * gw_share.get(position, 0.0) / gw_total))
    print("  keeper share moves %.1f%% -> %.1f%% on identical events; that is the save "
          "restructure, and it is the number to re-check first once GW1 completes"
          % (100 * bt["by_position"]["GKP"]["actual_share"],
             100 * patched_bt["by_position"]["GKP"]["predicted_share"]))

    print("\n=== invariants ===")
    for line in self_test(bonus_model, game, gw_fixtures, match_forecasts):
        print("  PASS  %s" % line)
    assert not any(v != v for v in all_bonus.values()), "NaN in GW bonus"
    assert not any(v != v for v in all_bps.values()), "NaN in GW BPS"
    print("  PASS  no NaN across %d players in GW%d" % (len(all_bonus), gw))

    print("\n=== refit_from_current_season ===")
    result = bonus_model.refit_from_current_season(game, write_report=False)
    print("  live %s: %s" % (cfg.season, result if result else
                             "no data yet (expected pre-season) - patched 2025/26 fit retained"))
    # Exercise the real code path by replaying a completed season through the
    # live-data interface: identical per-GW rows, reached via
    # GameState.player_history instead of the vaastav CSV. This is exactly what
    # happens from about GW6 2026/27 onward, and the refitted coefficients must
    # reproduce the direct fit to floating-point noise.
    replay = hist.load_vaastav_gws(game.prior_season)
    replay_state = type("ReplayState", (), {})()
    replay_state.players = {
        int(element): Player(
            id=int(element),
            code=int(element),
            web_name=str(grp["name"].iloc[0]),
            first_name="",
            second_name="",
            team_id=0,
            position=int(grp["element_type"].iloc[0]),
            now_cost=int(grp["value"].iloc[0]),
        )
        for element, grp in replay.groupby("element")
        if pd.notna(grp["element_type"].iloc[0])
    }
    replay_state.player_history = {
        int(element): grp.reset_index(drop=True)
        for element, grp in replay.groupby("element")
        if int(element) in replay_state.players
    }
    replay_state.history = lambda pid: replay_state.player_history.get(pid)
    replay_state.prior_season_row = lambda pid, season_name=None: None
    replay_state.latest_season_row = lambda pid: None
    replay_model = BonusModel(cfg)
    replay_model.fit(game, write_report=False)
    patched = dict(replay_model.fits[scoring.MID].coefficients_2026_27)
    live = replay_model.refit_from_current_season(replay_state, write_report=False)
    print("  replayed %s through the live interface -> %d rows, R2 %s"
          % (game.prior_season, live["rows"],
             ", ".join("%s %.3f" % (k, v["r2"]) for k, v in live["positions"].items())))
    direct = fit_bps_coefficients(replay)[scoring.MID].coefficients
    refit = replay_model.fits[scoring.MID].coefficients
    worst = max(abs(direct[f] - refit[f]) for f in FEATURES)
    assert live is not None and live["rows"] > 400
    assert worst < 1e-6, "refit diverged from the direct fit by %.6f" % worst
    print("  PASS  refit reproduces the direct fit (max MID coefficient delta %.2e)" % worst)
    print("  PASS  refit drops the manual patches: MID cbi %.3f (patched) -> %.3f (refitted)"
          % (patched["cbi"], replay_model.fits[scoring.MID].coefficients_2026_27["cbi"]))

    if bonus_model.warnings:
        print("\n=== warnings ===")
        for w in bonus_model.warnings:
            print("  - %s" % w)
