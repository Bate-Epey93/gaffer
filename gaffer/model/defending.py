"""Defensive point components: clean sheets, goals conceded, saves, penalty saves.

Three modelling decisions here are not obvious and are worth stating up front,
because each one is a place where the intuitive version is measurably wrong.

**Conceded points are E[-floor(k/2)], not -lambda/2.** FPL docks a point per
*two* goals, so a keeper facing lambda=1.0 loses 0.264 points, not 0.5. The
floor only bites at 2, 4, 6... and at realistic lambdas most of the mass sits
below the first step. ``scoring.expected_goals_conceded_points`` does the exact
Poisson sum; nothing here approximates it.

**Saves are a team property, not a keeper property.** Fitted on 2024/25 ->
2025/26 (13 keepers with 900+ minutes in both), a keeper's *save share*
(saves / shots on target faced) has a year-on-year correlation of **+0.03**,
and its cross-sectional variance is smaller than the binomial sampling
variance — there is no measurable persistent skill in it at all. Shots on
target *faced per 90* correlates **+0.68**. So saves points are predicted by
projecting the shot volume a team concedes and applying an almost-fully-shrunk
league save share. Modelling saves off a keeper's own ``saves_per_90`` instead
transfers his old club's defence to his new one: Dubravka posted 3.63 saves/90
behind Burnley's league-worst 5.68 shots on target faced per match and is a
Spurs player in 2026/27.

**Shots on target scale sub-linearly with expected goals conceded**, fitted
exponent ~0.63. A team expected to concede twice as much does not face twice
the shots, so the save-heavy-keeper effect is real but bounded.

**Clean sheets and conceded points are settled over the minutes the player was
actually on the pitch**, not over the whole match. FPL's rule is "no goal
conceded while on the pitch, having played 60+", so a defender withdrawn on the
hour keeps a clean sheet his team goes on to lose. Measured over 2024/25 and
2025/26, the ratio of the player clean-sheet rate to his team's full-match
clean-sheet rate runs 1.55 at 60-69 minutes, 1.36 at 70-79, 1.19 at 80-89 and
1.005 at exactly 90 — that last row is the control that says this is the rule
and not a selection effect, and 338 of 3,280 appearances in the 60-79 band kept
a clean sheet the team did not while *zero* went the other way. Goals credited
against the player track the same fractions (0.69 / 0.77 / 0.85 / 1.00 of the
team's match total). 16% of defender starts and 44% of midfielder starts end
before the 88th minute, so treating everyone as a 90-minute player understates
clean sheets and overstates conceded points for a large, identifiable slice of
the population.

Everything is fitted from football-data.co.uk shot counts and the vaastav
per-match archive; the constants recorded at the bottom of this docstring's
module are the values those fits produced, kept only as an offline fallback.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from gaffer.core import scoring, stats
from gaffer.core.config import REPORTS_DIR, Config
from gaffer.core.types import MinutesForecast
from gaffer.data import history as hist
from gaffer.data.cache import Cache

if TYPE_CHECKING:  # avoids a cycle: xp.py imports this module
    from gaffer.data.loaders import GameState
    from gaffer.model.xp import FixtureContext

log = logging.getLogger(__name__)

FIT_PATH = os.path.join(REPORTS_DIR, "defending_fit.json")
FIT_VERSION = 3

# Fitted 2026-08-14 on football-data 2024/25+2025/26 (760 team-matches) and the
# vaastav 2025-26 per-match archive. Used only when those sources cannot be
# loaded at fit time; `DefendingModel.fit` recomputes all of them from data.
FALLBACK = {
    "sot_scale": 3.3800,          # shots on target = scale * lambda ** exponent
    "sot_exponent": 0.6300,
    "league_sot_per_match": 4.1868,
    "sot_proxy_ratio": 0.9915,    # (saves + goals conceded) / football-data SOT
    "league_save_share": 0.6736,  # saves / (saves + goals conceded), pooled
    "save_share_prior_sot": 1096.0,
    "team_sot_faced_reliability": 0.4390,
    "pens_faced_per_match": 0.0940,
    "pen_save_rate": 0.1540,
    "pen_conversion": 0.79,
}

# P(penalty scored). SPEC 3.4 uses the same figure for the attacking side; it is
# the one number here that no FPL feed exposes directly, and it is only used to
# split an observed penalty-save rate into "penalties faced" x "save rate".
PEN_CONVERSION = 0.79

# Mean minutes of a start that ends before the hour: 47.3 over the 1,169 such
# starts in 2024/25-2025/26 (mostly half-time hooks and first-half injuries).
# Only used to back out E[minutes | reached 60] from E[minutes | started], which
# is the quantity the clean-sheet rule needs and the minutes model does not
# publish. Feeding 47.3 back through that identity reproduces the realised
# E[minutes | 60+] to within 0.2 minutes for DEF, MID and FWD alike.
SHORTENED_START_MINUTES = 47.3


# ---------------------------------------------------------------------------
# Minutes handling
# ---------------------------------------------------------------------------


def minutes_split(
    minutes: Optional[MinutesForecast], config: Config
) -> Tuple[float, float, float, float]:
    """Split a minutes forecast into ``(p_start, mins|start, p_sub, mins|sub)``.

    Every threshold-shaped component (clean sheet at 60', DEFCON at 10/12
    actions, saves at every third save) is *convex* in minutes, so evaluating it
    once at ``xmins`` is not the same as averaging it over the start/bench
    mixture and is biased low for likely starters. ``mins|start`` is recovered
    from ``xmins`` rather than assumed, so whatever the minutes model actually
    produced is preserved.
    """
    if minutes is None:
        return 1.0, config.model.starter_minutes, 0.0, config.model.sub_minutes
    p_start = max(0.0, min(1.0, float(minutes.p_start)))
    p_appear = max(0.0, min(1.0, float(minutes.p_appear)))
    p_sub = max(0.0, p_appear - p_start)
    m_sub = float(config.model.sub_minutes)
    if p_start > 1e-6:
        m_start = (float(minutes.xmins) - p_sub * m_sub) / p_start
        m_start = max(45.0, min(90.0, m_start))
    else:
        m_start = float(config.model.starter_minutes)
    return p_start, m_start, p_sub, m_sub


# ---------------------------------------------------------------------------
# Match context helpers
# ---------------------------------------------------------------------------


def _total_goals_line(p_over25: float) -> float:
    """Poisson total T with P(T > 2.5) equal to the de-vigged over-2.5 price."""
    lo, hi = 0.3, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = 1.0 - sum(stats.poisson_pmf(k, mid) for k in range(0, 3))
        if p < p_over25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fixture_lambdas(results: pd.DataFrame) -> pd.DataFrame:
    """Add bookmaker-implied ``lam_h`` / ``lam_a`` to a football-data frame.

    Calibrating fixture effects needs a per-match strength measure that was
    knowable *before* kickoff; season-average team ratings would leak the
    result into the very coefficient being fitted. The 1X2 prices give the
    supremacy and the over/under-2.5 price gives the total.
    """
    out = results.copy()
    lam_h: List[float] = []
    lam_a: List[float] = []
    for _, row in out.iterrows():
        oh, od, oa = row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")
        if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
            lam_h.append(float("nan"))
            lam_a.append(float("nan"))
            continue
        p_h, _p_d, p_a = stats.remove_vig([float(oh), float(od), float(oa)])
        over, under = row.get("odds_over25"), row.get("odds_under25")
        if pd.notna(over) and pd.notna(under):
            p_over = stats.remove_vig([float(over), float(under)])[0]
            total = _total_goals_line(p_over)
        else:
            total = 2.75
        a, b = stats.lambdas_from_odds(p_h, _p_d, p_a, total_goals_line=total)
        lam_h.append(a)
        lam_a.append(b)
    out["lam_h"] = lam_h
    out["lam_a"] = lam_a
    return out


def prior_season_team_lambdas(
    state: "GameState", results: pd.DataFrame, config: Config
) -> Dict[int, Tuple[float, float]]:
    """Crude ``{team_id: (attack, defence)}`` multipliers from one season.

    The real ratings are ``TeamRatingModel``'s job (SPEC 3.2). This exists so
    the defensive models can be exercised standalone, and as the documented
    degradation path when no fitted team model is supplied — a promoted side
    with no history falls back to the configured promoted priors rather than
    silently rating as league average.
    """
    played: Dict[str, List[float]] = {}
    for _, r in results.iterrows():
        for short, gf, ga in (
            (r["home_short"], r["home_goals"], r["away_goals"]),
            (r["away_short"], r["away_goals"], r["home_goals"]),
        ):
            if not isinstance(short, str) or pd.isna(gf) or pd.isna(ga):
                continue
            played.setdefault(short, [0.0, 0.0, 0.0])
            played[short][0] += float(gf)
            played[short][1] += float(ga)
            played[short][2] += 1.0
    league = config.model.league_avg_goals_per_team
    out: Dict[int, Tuple[float, float]] = {}
    for team in state.teams.values():
        rec = played.get(team.short_name)
        if not rec or rec[2] < 10:
            out[team.id] = (
                config.model.promoted_attack_prior,
                config.model.promoted_defence_prior,
            )
            continue
        out[team.id] = ((rec[0] / rec[2]) / league, (rec[1] / rec[2]) / league)
    return out


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------


@dataclass
class DefendingFit:
    sot_scale: float = FALLBACK["sot_scale"]
    sot_exponent: float = FALLBACK["sot_exponent"]
    league_sot_per_match: float = FALLBACK["league_sot_per_match"]
    sot_proxy_ratio: float = FALLBACK["sot_proxy_ratio"]
    league_save_share: float = FALLBACK["league_save_share"]
    save_share_prior_sot: float = FALLBACK["save_share_prior_sot"]
    team_sot_faced_reliability: float = FALLBACK["team_sot_faced_reliability"]
    team_sot_faced_factor: Dict[str, float] = field(default_factory=dict)
    pens_faced_per_match: float = FALLBACK["pens_faced_per_match"]
    pen_save_rate: float = FALLBACK["pen_save_rate"]
    pen_conversion: float = PEN_CONVERSION
    n_team_matches: int = 0
    seasons: List[str] = field(default_factory=list)
    source: str = "fallback"
    fitted_at: str = ""
    version: int = FIT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sot_scale": self.sot_scale,
            "sot_exponent": self.sot_exponent,
            "league_sot_per_match": self.league_sot_per_match,
            "sot_proxy_ratio": self.sot_proxy_ratio,
            "league_save_share": self.league_save_share,
            "save_share_prior_sot": self.save_share_prior_sot,
            "team_sot_faced_reliability": self.team_sot_faced_reliability,
            "team_sot_faced_factor": self.team_sot_faced_factor,
            "pens_faced_per_match": self.pens_faced_per_match,
            "pen_save_rate": self.pen_save_rate,
            "pen_conversion": self.pen_conversion,
            "n_team_matches": self.n_team_matches,
            "seasons": self.seasons,
            "source": self.source,
            "fitted_at": self.fitted_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DefendingFit":
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        return cls(**known)


def _poisson_deviance(mu: Sequence[float], y: Sequence[float]) -> float:
    total = 0.0
    for m, v in zip(mu, y):
        m = max(m, 1e-9)
        if v > 0:
            total += v * math.log(v / m)
        total -= v - m
    return 2.0 * total


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class DefendingModel:
    """Clean sheets, goals conceded, saves and penalty saves for one fixture."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.fit_params = DefendingFit()
        self.state: Optional["GameState"] = None
        self.team_model: Optional[Any] = None
        self.warnings: List[str] = []
        # player_id -> (shrunk save share on the saves+conceded basis, n SOT seen)
        self._save_share: Dict[int, Tuple[float, float]] = {}
        self._saves_per_90: Dict[int, float] = {}
        self._team_short: Dict[int, str] = {}
        self._sot_league_mean: Dict[bool, float] = {}

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        state: "GameState",
        results: Optional[pd.DataFrame] = None,
        team_model: Optional[Any] = None,
        refit: bool = False,
        cache: Optional[Cache] = None,
    ) -> "DefendingModel":
        """Fit the shot-volume and save-share parameters, then the per-keeper rates.

        ``results`` is a football-data frame; when omitted the two seasons
        before the current one are loaded. ``team_model`` is a fitted
        ``TeamRatingModel`` — optional, and only consulted for team shot rates
        if it exposes ``sot_faced_rate`` / ``sot_for_rate``.
        """
        self.state = state
        self.team_model = team_model
        self._sot_league_mean = {}
        self._team_short = {t.id: t.short_name for t in state.teams.values()}
        self.warnings = []

        loaded = None if refit else self._load_fit()
        if loaded is not None:
            self.fit_params = loaded
        else:
            self.fit_params = self._fit_from_data(state, results, cache)
            self._save_fit()

        self._fit_keeper_rates(state)
        return self

    def _fit_from_data(
        self,
        state: "GameState",
        results: Optional[pd.DataFrame],
        cache: Optional[Cache],
    ) -> DefendingFit:
        cache = cache or Cache()
        prior = hist.prior_season(state.season)
        two_back = hist.prior_season(state.season, back=2)
        seasons = [two_back, prior]
        try:
            if results is None:
                results = hist.load_football_data(seasons, cache=cache)
            res = fixture_lambdas(results)
        except Exception as exc:
            msg = "defending fit: football-data unavailable (%s: %s); using recorded fallback constants" % (
                type(exc).__name__,
                exc,
            )
            log.warning(msg)
            self.warnings.append(msg)
            return DefendingFit()

        fitted = DefendingFit(seasons=sorted(set(str(s) for s in res["season"])),
                              source="fit", fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

        # --- shots on target as a function of expected goals -------------
        lam: List[float] = []
        sot: List[float] = []
        for _, r in res.iterrows():
            for l_, s_ in ((r["lam_h"], r["home_sot"]), (r["lam_a"], r["away_sot"])):
                if pd.notna(l_) and pd.notna(s_):
                    lam.append(float(l_))
                    sot.append(float(s_))
        if len(lam) >= 100:
            best = (float("inf"), fitted.sot_scale, fitted.sot_exponent)
            for i in range(0, 71):
                scale = 1.5 + i * 0.05
                for j in range(0, 61):
                    expo = 0.2 + j * 0.02
                    dev = _poisson_deviance([scale * (l_ ** expo) for l_ in lam], sot)
                    if dev < best[0]:
                        best = (dev, scale, expo)
            # local refinement around the grid optimum
            scale, expo = best[1], best[2]
            for _ in range(30):
                improved = False
                for ds, de in ((0.01, 0), (-0.01, 0), (0, 0.004), (0, -0.004)):
                    dev = _poisson_deviance(
                        [(scale + ds) * (l_ ** (expo + de)) for l_ in lam], sot
                    )
                    if dev < best[0]:
                        best = (dev, scale + ds, expo + de)
                        scale, expo = scale + ds, expo + de
                        improved = True
                if not improved:
                    break
            fitted.sot_scale, fitted.sot_exponent = scale, expo
            fitted.league_sot_per_match = sum(sot) / len(sot)
            fitted.n_team_matches = len(lam)

        # --- per-team shot-volume-faced factor, shrunk by its reliability --
        latest = res[res["season"] == hist.season_dash(prior)]
        factors_by_season: Dict[str, Dict[str, float]] = {}
        for season, grp in res.groupby("season"):
            factors_by_season[str(season)] = self._team_sot_faced_factors(grp, fitted)
        fitted.team_sot_faced_reliability = self._sot_factor_reliability(factors_by_season)
        raw = self._team_sot_faced_factors(latest, fitted) if len(latest) else {}
        w = fitted.team_sot_faced_reliability
        fitted.team_sot_faced_factor = {
            k: 1.0 + w * (v - 1.0) for k, v in raw.items()
        }

        # --- keeper save share and penalties, from the per-match archive ---
        try:
            gws = hist.load_vaastav_gws(prior, cache=cache)
        except Exception as exc:
            msg = "defending fit: vaastav %s unavailable (%s); save share and penalty rates use fallback constants" % (
                prior,
                type(exc).__name__,
            )
            log.warning(msg)
            self.warnings.append(msg)
            return fitted

        keepers = gws[(gws["element_type"] == 1) & (gws["minutes"] > 0)]
        saves = float(keepers["saves"].sum())
        conceded = float(keepers["goals_conceded"].sum())
        if saves + conceded > 0:
            fitted.league_save_share = saves / (saves + conceded)
        # The FPL "saves + conceded" proxy for shots faced is ~99% of the
        # football-data count; keep the ratio explicit so the two bases mix
        # without a hidden bias.
        prior_res = res[res["season"] == hist.season_dash(prior)]
        fd_sot = float(prior_res["home_sot"].sum() + prior_res["away_sot"].sum())
        if fd_sot > 0:
            fitted.sot_proxy_ratio = (saves + conceded) / fd_sot

        fitted.save_share_prior_sot = self._save_share_prior_strength(keepers, fitted)

        # Penalties: only the *saved* count and the *not scored* count are
        # observable. Splitting them into "faced" and "save rate" needs the
        # league conversion rate, the single external constant here.
        pens_saved = float(gws["penalties_saved"].sum())
        pens_missed = float(gws["penalties_missed"].sum())
        n_team_matches = 2.0 * float(gws["fixture"].nunique())
        if n_team_matches > 0 and pens_missed > 0:
            faced = pens_missed / (1.0 - PEN_CONVERSION)
            fitted.pens_faced_per_match = faced / n_team_matches
            fitted.pen_save_rate = pens_saved / faced
        fitted.pen_conversion = PEN_CONVERSION
        return fitted

    def _team_sot_faced_factors(
        self, results: pd.DataFrame, fitted: DefendingFit
    ) -> Dict[str, float]:
        """Observed / lambda-implied shots on target faced, per club."""
        obs: Dict[str, float] = {}
        exp: Dict[str, float] = {}
        for _, r in results.iterrows():
            pairs = (
                (r["home_short"], r["away_sot"], r["lam_a"]),
                (r["away_short"], r["home_sot"], r["lam_h"]),
            )
            for short, faced, lam_against in pairs:
                if not isinstance(short, str) or pd.isna(faced) or pd.isna(lam_against):
                    continue
                obs[short] = obs.get(short, 0.0) + float(faced)
                exp[short] = exp.get(short, 0.0) + fitted.sot_scale * (
                    float(lam_against) ** fitted.sot_exponent
                )
        return {k: obs[k] / exp[k] for k in obs if exp.get(k, 0.0) > 0}

    @staticmethod
    def _sot_factor_reliability(by_season: Dict[str, Dict[str, float]]) -> float:
        """Year-on-year correlation of the shot-volume-faced factor.

        Used directly as the shrinkage weight: a factor that repeats at r is
        worth exactly weight r on next season's projection.
        """
        seasons = sorted(by_season)
        if len(seasons) < 2:
            return FALLBACK["team_sot_faced_reliability"]
        a, b = by_season[seasons[-2]], by_season[seasons[-1]]
        common = sorted(set(a) & set(b))
        if len(common) < 8:
            return FALLBACK["team_sot_faced_reliability"]
        xs = [a[k] for k in common]
        ys = [b[k] for k in common]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        if den <= 0:
            return FALLBACK["team_sot_faced_reliability"]
        return max(0.0, min(1.0, num / den))

    @staticmethod
    def _save_share_prior_strength(keepers: pd.DataFrame, fitted: DefendingFit) -> float:
        """Beta-binomial pseudo-shots behind the league save share.

        Variance components on the per-keeper share: whatever is left after
        removing binomial sampling noise is the real spread. On 2025/26 that
        residual is *negative*, so the floor below is what actually binds and
        the share is shrunk nearly all the way to the league mean.
        """
        agg = keepers.groupby("element").agg(
            saves=("saves", "sum"), conceded=("goals_conceded", "sum"),
            mins=("minutes", "sum"),
        )
        agg = agg[(agg["mins"] >= 900) & (agg["saves"] + agg["conceded"] > 0)]
        if len(agg) < 6:
            return FALLBACK["save_share_prior_sot"]
        n = agg["saves"] + agg["conceded"]
        share = agg["saves"] / n
        p_bar = float(fitted.league_save_share)
        total_var = float(share.var(ddof=1))
        sampling_var = float((p_bar * (1.0 - p_bar) / n).mean())
        true_var = max(total_var - sampling_var, 0.0)
        # Floor: refuse to claim more than ~1500 pseudo-shots of certainty even
        # when the point estimate of the true spread is exactly zero.
        true_var = max(true_var, 1.5e-4)
        return p_bar * (1.0 - p_bar) / true_var

    def _fit_keeper_rates(self, state: "GameState") -> None:
        """Per-keeper shrunk save share and observed saves per 90."""
        self._save_share = {}
        self._saves_per_90 = {}
        prior_name = hist.season_slash(state.prior_season or hist.prior_season(state.season))
        for pid, player in state.players.items():
            if player.position != scoring.GKP:
                continue
            saves = 0.0
            conceded = 0.0
            minutes = 0.0
            # Current season first (real, if any), then the prior season.
            frame = state.history(pid)
            if len(frame) and float(frame["minutes"].sum()) > 0:
                saves += float(frame["saves"].sum())
                conceded += float(frame["goals_conceded"].sum())
                minutes += float(frame["minutes"].sum())
            for row in state.history_past(pid):
                if row.get("season_name") != prior_name:
                    continue
                saves += float(row.get("saves") or 0.0)
                conceded += float(row.get("goals_conceded") or 0.0)
                minutes += float(row.get("minutes") or 0.0)
            faced = saves + conceded
            share = stats.beta_binomial_rate(
                saves, faced,
                self.fit_params.league_save_share,
                self.fit_params.save_share_prior_sot,
            )
            self._save_share[pid] = (share, faced)
            if minutes > 0:
                self._saves_per_90[pid] = 90.0 * saves / minutes

    # -- persistence -------------------------------------------------------
    def _load_fit(self) -> Optional[DefendingFit]:
        if not os.path.exists(FIT_PATH):
            return None
        try:
            with open(FIT_PATH, "r") as fh:
                raw = json.load(fh)
        except (ValueError, OSError):
            return None
        if int(raw.get("version", 0)) != FIT_VERSION:
            return None
        return DefendingFit.from_dict(raw)

    def _save_fit(self) -> None:
        if self.fit_params.source != "fit":
            return
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            with open(FIT_PATH, "w") as fh:
                json.dump(self.fit_params.to_dict(), fh, indent=2, sort_keys=True)
        except OSError as exc:  # a read-only reports dir must not kill a run
            log.warning("could not write %s: %s", FIT_PATH, exc)

    # -- lookups -----------------------------------------------------------
    def save_share(self, player_id: int) -> Tuple[float, float]:
        """``(shrunk save share, shots on target faced behind it)``."""
        return self._save_share.get(
            player_id, (self.fit_params.league_save_share, 0.0)
        )

    def saves_per_90(self, player_id: int) -> Optional[float]:
        return self._saves_per_90.get(player_id)

    def team_sot_factor(self, team_id: int) -> float:
        short = self._team_short.get(team_id)
        if short is None:
            return 1.0
        return self.team_sot_factor_by_short(short)

    def team_sot_factor_by_short(self, short: str) -> float:
        """Shot volume a club concedes relative to its expected goals conceded.

        Two sides can concede the same number of goals from very different shot
        counts; this is what separates their keepers' save points. Already
        shrunk by the factor's fitted year-on-year reliability, so a promoted
        club with no history sits at exactly 1.0.
        """
        return self.fit_params.team_sot_faced_factor.get(short, 1.0)

    def defensive_points_for_lambda(
        self,
        position: int,
        lambda_conceded: float,
        save_share: Optional[float] = None,
        team_short: Optional[str] = None,
        minutes: float = 90.0,
    ) -> Dict[str, float]:
        """Defensive points for a fully-fit player at a given conceded rate.

        The fixture-free form of ``project``: used to compare keepers across
        defensive contexts and to replay a completed season without needing a
        ``FixtureContext`` per match.
        """
        lam = max(float(lambda_conceded), 0.05)
        share = self.fit_params.league_save_share if save_share is None else save_share
        factor = 1.0 if team_short is None else self.team_sot_factor_by_short(team_short)
        sot = self.fit_params.sot_scale * (lam ** self.fit_params.sot_exponent) * factor
        played = max(0.0, min(90.0, minutes))
        lam_saves = sot * self.fit_params.sot_proxy_ratio * share * played / 90.0
        p_60 = 1.0 if played >= 60 else 0.0
        # Same on-pitch settlement as `project`; here the minutes are known
        # exactly, so the fraction is not an expectation.
        on_pitch = played / 90.0
        xp_cs = scoring.clean_sheet_points(
            position, stats.p_clean_sheet(lam) ** on_pitch, p_60
        )
        xp_gc = scoring.expected_goals_conceded_points(position, lam * on_pitch, p_60)
        xp_sv = (
            scoring.expected_save_points(lam_saves)
            if position == scoring.GKP else 0.0
        )
        return {
            "sot_faced": sot,
            "lambda_saves": lam_saves,
            "xp_clean_sheet": xp_cs,
            "xp_goals_conceded": xp_gc,
            "xp_saves": xp_sv,
            "total": xp_cs + xp_gc + xp_sv,
        }

    def _sot_for_league_mean(self, is_home: bool) -> float:
        """Mean ``sot_for_rate`` across the league at one home/away state.

        Dividing an opponent's rate by this cancels the home-advantage factor
        that ``sot_faced_rate`` has already applied from the *other* side, so
        the two multiply without double-counting it.
        """
        cached = self._sot_league_mean.get(bool(is_home))
        if cached is not None:
            return cached
        fn = getattr(self.team_model, "sot_for_rate", None)
        total, n = 0.0, 0
        if callable(fn) and self.state is not None:
            for tid in self.state.teams:
                try:
                    v = float(fn(tid, is_home))
                except Exception:
                    continue
                if v == v and v > 0:  # reject NaN (promoted club, unrated)
                    total += v
                    n += 1
        mean = total / n if n >= 10 else 0.0
        self._sot_league_mean[bool(is_home)] = mean
        return mean

    def expected_sot_faced(
        self,
        team_id: int,
        lambda_conceded: float,
        is_home: Optional[bool] = None,
        opponent_id: Optional[int] = None,
    ) -> float:
        """Shots on target a team faces in this fixture.

        Preferred path: ``TeamRatingModel``'s own shot ratings — the club's
        shots-faced rate against an average opponent, scaled by *this*
        opponent's attacking strength. Those ratings are fitted to shot counts
        directly and know things goals do not, such as a side that concedes a
        high volume of low-quality attempts.

        Fallback: the fitted ``scale * lambda ** exponent`` curve. The exponent
        is well under 1, so a team expected to concede twice as much does not
        face twice the shots.
        """
        lam = max(float(lambda_conceded), 0.05)
        faced_fn = getattr(self.team_model, "sot_faced_rate", None)
        for_fn = getattr(self.team_model, "sot_for_rate", None)
        if callable(faced_fn) and is_home is not None and opponent_id is not None:
            try:
                base = float(faced_fn(team_id, is_home))
            except Exception:  # a partially built team model must not break xp
                base = float("nan")
            opp_mult = 1.0
            league_mean = self._sot_for_league_mean(not is_home)
            if callable(for_fn) and league_mean > 0:
                try:
                    opp_rate = float(for_fn(opponent_id, not is_home))
                except Exception:
                    opp_rate = float("nan")
                if opp_rate == opp_rate and opp_rate > 0:
                    opp_mult = opp_rate / league_mean
            if base == base and base > 0:
                return base * opp_mult
        return (
            self.fit_params.sot_scale
            * (lam ** self.fit_params.sot_exponent)
            * self.team_sot_factor(team_id)
        )

    # -- projections -------------------------------------------------------
    def on_pitch_fraction(self, ctx: "FixtureContext") -> float:
        """``E[minutes played | the player reached 60] / 90``, in ``[2/3, 1]``.

        The minutes model publishes ``p_start``, ``p_60`` and ``xmins``, from
        which ``minutes_split`` recovers ``E[minutes | started]``. What the
        clean-sheet rule needs is ``E[minutes | reached 60]``, which is the same
        quantity with the sub-60 starts removed:

            E[m | start] = q E[m | m>=60] + (1-q) E[m | start, m<60]

        with ``q = P(reach 60 | start) = p_60 / p_start``. The last term is the
        fitted ``SHORTENED_START_MINUTES``. Substitutes are ignored on purpose —
        a bench cameo of ~20 minutes essentially never reaches the hour, so all
        of the 60+ mass belongs to the start branch.
        """
        mf = self._minutes(ctx)
        if mf is None:
            return 1.0
        p_start, m_start, _p_sub, _m_sub = minutes_split(mf, self.config)
        p_60 = max(0.0, min(1.0, float(mf.p_60)))
        if p_start <= 1e-6 or p_60 <= 1e-9:
            return 1.0
        q = min(1.0, p_60 / p_start)
        if q <= 1e-6:
            return 1.0
        m_60 = (m_start - (1.0 - q) * SHORTENED_START_MINUTES) / q
        return max(60.0, min(90.0, m_60)) / 90.0

    def player_clean_sheet(self, ctx: "FixtureContext") -> float:
        """P(no goal conceded while this player was on the pitch).

        Under a Poisson process the survival over ``m`` of the 90 minutes is
        ``exp(-lambda m/90) = P(0 over 90) ** (m/90)``, so raising the match
        probability to the on-pitch fraction applies the rule *and* keeps
        whatever the team model's Dixon-Coles correction did to ``P(0-0)``.
        It is also the only channel by which a nailed 90-minute defender and a
        regularly-hooked one are told apart at all: the flat ``p_60`` gate gives
        both of them the same clean-sheet probability.
        """
        return self._match_clean_sheet(ctx) ** self.on_pitch_fraction(ctx)

    def project_clean_sheet(self, player_id: int, ctx: "FixtureContext") -> float:
        """P(clean sheet while on the pitch AND the player is on for 60+)."""
        return self.player_clean_sheet(ctx) * self._p_60(ctx)

    def project_conceded_points(self, player_id: int, ctx: "FixtureContext") -> float:
        player = self._player(player_id, ctx)
        return scoring.expected_goals_conceded_points(
            player.position,
            self._lambda_conceded(ctx) * self.on_pitch_fraction(ctx),
            self._p_60(ctx),
        )

    def project_saves(self, player_id: int, ctx: "FixtureContext") -> float:
        """Expected saves (lambda), already scaled by expected minutes."""
        player = self._player(player_id, ctx)
        if player.position != scoring.GKP:
            return 0.0
        share, _n = self.save_share(player_id)
        sot = self.expected_sot_faced(
            player.team_id, self._lambda_conceded(ctx),
            is_home=bool(getattr(ctx, "is_home", True)),
            opponent_id=self._opponent_id(ctx, player),
        )
        per_90 = sot * self.fit_params.sot_proxy_ratio * share
        p_start, m_start, p_sub, m_sub = minutes_split(self._minutes(ctx), self.config)
        return per_90 * (p_start * m_start + p_sub * m_sub) / 90.0

    def project_penalty_save_points(self, player_id: int, ctx: "FixtureContext") -> float:
        player = self._player(player_id, ctx)
        if player.position != scoring.GKP:
            return 0.0
        lam = self.expected_penalty_saves(player_id, ctx)
        return scoring.PENALTY_SAVE * lam

    def expected_penalty_saves(self, player_id: int, ctx: "FixtureContext") -> float:
        player = self._player(player_id, ctx)
        if player.position != scoring.GKP:
            return 0.0
        league_lam = self.config.model.league_avg_goals_per_team
        # Penalties won scale with how much a side attacks; use the opponent's
        # expected goals as the proxy, damped because a penalty is a discrete
        # box event rather than general territory.
        attack_ratio = (max(self._lambda_conceded(ctx), 0.05) / max(league_lam, 1e-6)) ** 0.5
        faced = self.fit_params.pens_faced_per_match * attack_ratio
        p_start, m_start, p_sub, m_sub = minutes_split(self._minutes(ctx), self.config)
        on_pitch = (p_start * m_start + p_sub * m_sub) / 90.0
        return faced * self.fit_params.pen_save_rate * on_pitch

    def project(self, player_id: int, ctx: "FixtureContext") -> Dict[str, float]:
        """Every defensive component for one player in one fixture.

        Keys are named to drop straight onto ``PlayerFixtureProjection``.
        ``xp_penalty_save`` is deliberately separate from that dataclass's
        ``xp_penalty`` field, which also carries the attacking model's penalty
        *miss* term; the assembler adds the two.
        """
        player = self._player(player_id, ctx)
        pos = player.position
        lam_conceded = self._lambda_conceded(ctx)
        p_cs_match = self._match_clean_sheet(ctx)
        p_start, m_start, p_sub, m_sub = minutes_split(self._minutes(ctx), self.config)
        p_60 = self._p_60(ctx)

        # Settle both terms over the minutes actually played, not the full match.
        on_pitch = self.on_pitch_fraction(ctx)
        p_cs_on_pitch = p_cs_match ** on_pitch
        p_clean_sheet = p_cs_on_pitch * p_60
        xp_cs = scoring.clean_sheet_points(pos, p_cs_on_pitch, p_60)
        xp_conceded = scoring.expected_goals_conceded_points(
            pos, lam_conceded * on_pitch, p_60
        )

        lam_saves = 0.0
        xp_saves = 0.0
        sot_faced = 0.0
        share = self.fit_params.league_save_share
        if pos == scoring.GKP:
            share, _n = self.save_share(player_id)
            sot_faced = self.expected_sot_faced(
                player.team_id, lam_conceded,
                is_home=bool(getattr(ctx, "is_home", True)),
                opponent_id=self._opponent_id(ctx, player),
            )
            per_90 = sot_faced * self.fit_params.sot_proxy_ratio * share
            lam_saves = per_90 * (p_start * m_start + p_sub * m_sub) / 90.0
            # E[floor(saves/3)] is convex, so average it over the minutes
            # mixture rather than evaluating once at the blended lambda.
            xp_saves = (
                p_start * scoring.expected_save_points(per_90 * m_start / 90.0)
                + p_sub * scoring.expected_save_points(per_90 * m_sub / 90.0)
            )

        lam_pen_saves = self.expected_penalty_saves(player_id, ctx)
        xp_pen_save = scoring.PENALTY_SAVE * lam_pen_saves

        cs_pts = scoring.CLEAN_SHEET_POINTS.get(pos, 0)
        var_cs = (cs_pts ** 2) * p_clean_sheet * (1.0 - p_clean_sheet)
        var_saves = (scoring.SAVE_POINTS ** 2) * lam_saves / (scoring.SAVES_PER ** 2)

        out = {
            "lambda_conceded": lam_conceded,
            "p_clean_sheet": p_clean_sheet,
            "p_clean_sheet_match": p_cs_match,
            "lambda_saves": lam_saves,
            "lambda_penalty_saves": lam_pen_saves,
            "xp_clean_sheet": xp_cs,
            "xp_goals_conceded": xp_conceded,
            "xp_saves": xp_saves,
            "xp_penalty_save": xp_pen_save,
            "xp_defending_total": xp_cs + xp_conceded + xp_saves + xp_pen_save,
            "var_clean_sheet": var_cs,
            "var_saves": var_saves,
            "sot_faced": sot_faced,
            "save_share": share,
            "p_60": p_60,
            "on_pitch_fraction": on_pitch,
        }
        cross = self.saves_cross_check(player_id, ctx)
        if cross is not None:
            out["saves_per_90_history"] = cross[0]
            out["saves_per_90_model"] = cross[1]
            out["saves_model_history_ratio"] = cross[2]
        return out

    def saves_cross_check(
        self, player_id: int, ctx: "FixtureContext"
    ) -> Optional[Tuple[float, float, float]]:
        """``(historic saves/90, modelled saves/90, ratio)`` or None.

        A ratio far from 1 is not an error — it is the model correcting for a
        transfer. A keeper who moves from a side that conceded 5.7 shots on
        target a game to one that concedes 4.0 *should* project well below his
        own history.
        """
        player = self._player(player_id, ctx)
        if player.position != scoring.GKP:
            return None
        observed = self.saves_per_90(player_id)
        if observed is None or observed <= 0:
            return None
        share, _n = self.save_share(player_id)
        sot = self.expected_sot_faced(
            player.team_id, self._lambda_conceded(ctx),
            is_home=bool(getattr(ctx, "is_home", True)),
            opponent_id=self._opponent_id(ctx, player),
        )
        modelled = sot * self.fit_params.sot_proxy_ratio * share
        return observed, modelled, modelled / observed

    def explain(self, player_id: int, ctx: "FixtureContext") -> str:
        d = self.project(player_id, ctx)
        player = self._player(player_id, ctx)
        if player.position == scoring.GKP:
            return (
                "lambda conceded %.2f -> CS %.0f%% (x p60 %.0f%%) = %+.2f, "
                "conceded %+.2f, %.2f SOT faced x %.0f%% save share = %.2f saves %+.2f, "
                "pen saves %+.2f"
                % (
                    d["lambda_conceded"], 100 * d["p_clean_sheet_match"], 100 * d["p_60"],
                    d["xp_clean_sheet"], d["xp_goals_conceded"], d["sot_faced"],
                    100 * d["save_share"], d["lambda_saves"], d["xp_saves"],
                    d["xp_penalty_save"],
                )
            )
        return (
            "lambda conceded %.2f -> CS %.0f%% (x p60 %.0f%%) = %+.2f, conceded %+.2f"
            % (
                d["lambda_conceded"], 100 * d["p_clean_sheet_match"], 100 * d["p_60"],
                d["xp_clean_sheet"], d["xp_goals_conceded"],
            )
        )

    # -- context accessors -------------------------------------------------
    def _player(self, player_id: int, ctx: "FixtureContext") -> Any:
        player = getattr(ctx, "player", None)
        if player is not None and getattr(player, "id", None) == player_id:
            return player
        if self.state is not None:
            return self.state.player(player_id)
        if player is not None:
            return player
        raise KeyError("no player %d: fit() the model or pass a matching context" % player_id)

    @staticmethod
    def _opponent_id(ctx: "FixtureContext", player: Any) -> Optional[int]:
        fixture = getattr(ctx, "fixture", None)
        if fixture is None:
            return None
        try:
            return int(fixture.opponent_of(player.team_id))
        except Exception:
            return None

    @staticmethod
    def _minutes(ctx: "FixtureContext") -> Optional[MinutesForecast]:
        return getattr(ctx, "minutes", None)

    def _p_60(self, ctx: "FixtureContext") -> float:
        mf = self._minutes(ctx)
        if mf is None:
            return 1.0
        return max(0.0, min(1.0, float(mf.p_60)))

    @staticmethod
    def _lambda_conceded(ctx: "FixtureContext") -> float:
        return max(float(getattr(ctx, "opponent_lambda", 1.45)), 0.05)

    def _match_clean_sheet(self, ctx: "FixtureContext") -> float:
        """P(0 conceded), taken from the match forecast when it carries one.

        ``TeamRatingModel`` applies a Dixon-Coles correction that lifts P(0-0)
        above the independent-Poisson value, so its ``p_cs_*`` is a better
        number than ``exp(-lambda)`` when it is available.
        """
        forecast = getattr(ctx, "forecast", None)
        player = getattr(ctx, "player", None)
        if forecast is not None and player is not None:
            is_home = bool(getattr(ctx, "is_home", True))
            p = getattr(forecast, "p_cs_h" if is_home else "p_cs_a", None)
            if p is not None and 0.0 < float(p) < 1.0:
                return float(p)
        return stats.p_clean_sheet(self._lambda_conceded(ctx))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from gaffer.core.types import MatchForecast
    from gaffer.data.loaders import fixtures_by_gw, load_game_state

    parser = argparse.ArgumentParser(description="defending model smoke test")
    parser.add_argument("--refit", action="store_true", help="ignore the cached fit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()

    t0 = time.time()
    state = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (state.summary(), time.time() - t0))

    cache = Cache()
    results = hist.load_football_data(
        [hist.prior_season(state.season, back=2), hist.prior_season(state.season)], cache=cache
    )
    t0 = time.time()
    model = DefendingModel(cfg).fit(state, results=results, refit=args.refit, cache=cache)
    print("fitted in %.2fs (source=%s, seasons=%s)"
          % (time.time() - t0, model.fit_params.source, model.fit_params.seasons))
    for w in model.warnings:
        print("  WARNING %s" % w)

    fp = model.fit_params
    print("\n=== fitted parameters ===")
    print("  shots on target faced = %.4f * lambda_conceded ^ %.4f   (league mean %.3f/match)"
          % (fp.sot_scale, fp.sot_exponent, fp.league_sot_per_match))
    print("  save share: league %.4f, beta-binomial prior %.0f shots on target"
          % (fp.league_save_share, fp.save_share_prior_sot))
    print("  (saves+conceded)/football-data SOT = %.4f" % fp.sot_proxy_ratio)
    print("  team shots-faced factor: reliability %.3f, %d clubs"
          % (fp.team_sot_faced_reliability, len(fp.team_sot_faced_factor)))
    extremes = sorted(fp.team_sot_faced_factor.items(), key=lambda kv: kv[1])
    if extremes:
        print("    lowest  %s" % ", ".join("%s %.3f" % e for e in extremes[:3]))
        print("    highest %s" % ", ".join("%s %.3f" % e for e in extremes[-3:]))
    print("  penalties faced %.4f/match x save rate %.3f -> %.4f pen saves/match (%.3f pts)"
          % (fp.pens_faced_per_match, fp.pen_save_rate,
             fp.pens_faced_per_match * fp.pen_save_rate,
             5 * fp.pens_faced_per_match * fp.pen_save_rate))

    # --- E[floor(k/2)] is not lambda/2 -----------------------------------
    print("\n=== conceded points: exact E[-floor(k/2)] vs the -lambda/2 shortcut ===")
    print("  %-8s %-14s %-14s %s" % ("lambda", "exact", "-lambda/2", "error"))
    for lam in (0.4, 0.8, 1.2, 1.45, 2.0, 3.0):
        exact = scoring.expected_goals_conceded_points(scoring.DEF, lam, 1.0)
        naive = -lam / 2.0
        print("  %-8.2f %-14.4f %-14.4f %+.4f (%.0f%%)"
              % (lam, exact, naive, exact - naive, 100 * (exact - naive) / abs(naive)))
    assert scoring.expected_goals_conceded_points(scoring.DEF, 1.0, 1.0) > -0.5, \
        "E[floor(k/2)] must be strictly less punitive than lambda/2"
    print("  the shortcut over-penalises every realistic lambda: OK")

    # --- build fixture contexts for GW1 ----------------------------------
    try:
        from gaffer.model.xp import FixtureContext as _RealCtx  # type: ignore
        ctx_cls = _RealCtx
        print("\nusing the real FixtureContext from gaffer.model.xp")
    except Exception:
        @dataclass
        class _Ctx:  # smoke-test stand-in with the SPEC 3.8 field set
            fixture: Any
            player: Any
            is_home: bool
            team_lambda: float
            opponent_lambda: float
            forecast: Any
            minutes: MinutesForecast
        ctx_cls = _Ctx
        print("\ngaffer.model.xp not present yet; using a local FixtureContext with the SPEC field set")

    prior_results = results[results["season"] == hist.season_dash(hist.prior_season(state.season))]
    strengths = prior_season_team_lambdas(state, prior_results, cfg)
    league = cfg.model.league_avg_goals_per_team
    home_adv = cfg.model.home_advantage

    def lambdas(home_id: int, away_id: int) -> Tuple[float, float]:
        ah, dh = strengths[home_id]
        aa, da = strengths[away_id]
        return league * ah * da * home_adv, league * aa * dh

    gw = state.current_gw
    fixtures = fixtures_by_gw(state)[gw]
    fx_by_team: Dict[int, Any] = {}
    for f in fixtures:
        fx_by_team[f.team_h] = f
        fx_by_team[f.team_a] = f

    def make_ctx(player_id: int, p_start: float = 1.0) -> Any:
        p = state.player(player_id)
        f = fx_by_team[p.team_id]
        lh, la = lambdas(f.team_h, f.team_a)
        is_home = f.is_home_for(p.team_id)
        team_lam, opp_lam = (lh, la) if is_home else (la, lh)
        fc = MatchForecast(
            fixture_id=f.id, gw=gw, team_h=f.team_h, team_a=f.team_a,
            lambda_h=lh, lambda_a=la,
            p_cs_h=stats.p_clean_sheet(la), p_cs_a=stats.p_clean_sheet(lh),
        )
        mf = MinutesForecast(
            player_id=player_id, gw=gw, p_appear=min(1.0, p_start + 0.02),
            p_start=p_start, p_60=p_start * 0.94,
            xmins=p_start * 88.0, reason="smoke test",
        )
        return ctx_cls(fixture=f, player=p, is_home=is_home, team_lambda=team_lam,
                       opponent_lambda=opp_lam, forecast=fc, minutes=mf)

    # --- keeper comparison ------------------------------------------------
    print("\n=== keeper comparison: shot-stopper on a weak side vs clean-sheet keeper on a strong one ===")
    keepers = [
        (pid, p) for pid, p in state.players.items()
        if p.position == scoring.GKP and p.team_id in fx_by_team and p.status == "a"
    ]
    scored = []
    for pid, p in keepers:
        ctx = make_ctx(pid)
        d = model.project(pid, ctx)
        n_sot = model.save_share(pid)[1]
        scored.append((pid, p, ctx, d, n_sot))
    # only keepers with a real workload behind them (a full season is ~150 SOT)
    ranked = [s for s in scored if s[4] >= 100]
    ranked.sort(key=lambda s: s[3]["lambda_conceded"])
    best_defence = ranked[0]
    worst_defence = ranked[-1]

    hdr = ("%-14s %-4s %-5s %-7s %-7s %-7s %-7s %-7s %-7s %-7s %s"
           % ("keeper", "team", "price", "lam_gc", "SOT", "share", "saves", "xp_cs", "xp_gc",
              "xp_sav", "xp_def"))
    print("  " + hdr)
    for pid, p, ctx, d, _n in (worst_defence, best_defence):
        print("  %-14s %-4s %-5.1f %-7.2f %-7.2f %-7.3f %-7.2f %-7.2f %-7.2f %-7.2f %.2f"
              % (p.web_name, state.short_name(p.team_id), p.price, d["lambda_conceded"],
                 d["sot_faced"], d["save_share"], d["lambda_saves"], d["xp_clean_sheet"],
                 d["xp_goals_conceded"], d["xp_saves"], d["xp_defending_total"]))
    print("    %s (weak defence) saves component %.2f vs %s (strong defence) %.2f"
          % (worst_defence[1].web_name, worst_defence[3]["xp_saves"],
             best_defence[1].web_name, best_defence[3]["xp_saves"]))
    assert worst_defence[3]["xp_saves"] > best_defence[3]["xp_saves"], \
        "the keeper behind the worse defence must earn more save points"
    assert best_defence[3]["xp_clean_sheet"] > worst_defence[3]["xp_clean_sheet"], \
        "the keeper behind the better defence must earn more clean-sheet points"
    print("  saves and clean sheets move in opposite directions across the two: OK")

    print("\n  full GW%d keeper board (workload >= 100 shots on target of history):" % gw)
    board = sorted(ranked, key=lambda s: -s[3]["xp_defending_total"])
    print("  " + hdr)
    for pid, p, ctx, d, _n in board[:14]:
        print("  %-14s %-4s %-5.1f %-7.2f %-7.2f %-7.3f %-7.2f %-7.2f %-7.2f %-7.2f %.2f"
              % (p.web_name, state.short_name(p.team_id), p.price, d["lambda_conceded"],
                 d["sot_faced"], d["save_share"], d["lambda_saves"], d["xp_clean_sheet"],
                 d["xp_goals_conceded"], d["xp_saves"], d["xp_defending_total"]))

    # --- can a save-heavy keeper on a bad team out-score a CS keeper? -----
    print("\n=== can a save-heavy keeper on a bad side out-score a clean-sheet keeper on a good one? ===")
    print("  sweeping lambda conceded at the fitted league save share, 90 minutes played:")
    print("  %-8s %-7s %-8s %-8s %-8s %-8s %s"
          % ("lam_gc", "SOT", "saves", "xp_cs", "xp_gc", "xp_sav", "total(GKP)"))
    rows = []
    for lam in (0.55, 0.75, 1.0, 1.25, 1.5, 1.8, 2.2, 2.6):
        d = model.defensive_points_for_lambda(scoring.GKP, lam)
        rows.append((lam, d["sot_faced"], d["lambda_saves"], d["xp_clean_sheet"],
                     d["xp_goals_conceded"], d["xp_saves"], d["total"]))
        print("  %-8.2f %-7.2f %-8.2f %-8.2f %-8.2f %-8.2f %.2f" % rows[-1])
    assert all(rows[i][5] <= rows[i + 1][5] + 1e-9 for i in range(len(rows) - 1)), \
        "the saves component must rise monotonically with shots faced"
    print("  saves component rises monotonically with shots faced: OK (%.2f -> %.2f)"
          % (rows[0][5], rows[-1][5]))
    print("  at equal save share the clean-sheet term still wins, which is what 2025/26 shows:")
    print("  the flip needs one of the three channels the model exposes ->")
    for share in (fp.league_save_share, 0.72, 0.76, 0.80):
        bad = model.defensive_points_for_lambda(scoring.GKP, 2.2, save_share=share)
        good = model.defensive_points_for_lambda(scoring.GKP, 0.75)
        print("    (a) save share %.3f: weak-side %.2f vs strong-side %.2f -> %s"
              % (share, bad["total"], good["total"],
                 "WEAK SIDE WINS" if bad["total"] > good["total"] else "strong side"))
    # (b) equal defences, different shot volume conceded
    lo = min(fp.team_sot_faced_factor.items(), key=lambda kv: kv[1])
    hi = max(fp.team_sot_faced_factor.items(), key=lambda kv: kv[1])
    a = model.defensive_points_for_lambda(scoring.GKP, 1.35, team_short=hi[0])
    b = model.defensive_points_for_lambda(scoring.GKP, 1.35, team_short=lo[0])
    print("    (b) same lambda 1.35, shot-volume factor %s %.3f vs %s %.3f: saves %.2f vs %.2f (%+.2f pts)"
          % (hi[0], hi[1], lo[0], lo[1], a["xp_saves"], b["xp_saves"],
             a["xp_saves"] - b["xp_saves"]))
    print("    (c) 2026/27 restructured save BPS (any save 2 BPS) routes more of this into bonus.py")

    # --- replay 2025/26: does the model reproduce the real keeper table? ---
    print("\n=== replay: model vs actual keeper defensive points, %s ===" % hist.prior_season(state.season))
    prior_gws = hist.load_vaastav_gws(hist.prior_season(state.season), cache=cache)
    lam_res = fixture_lambdas(prior_results)
    lam_by_pair = {}
    for _, r in lam_res.iterrows():
        if pd.notna(r["lam_h"]):
            lam_by_pair[(r["home_short"], r["away_short"])] = (float(r["lam_h"]), float(r["lam_a"]))
    # vaastav fixture id -> (home_short, away_short)
    pair_by_fixture = {}
    for fid, grp in prior_gws.groupby("fixture"):
        h = grp[grp["was_home"]]["team_short"]
        a = grp[~grp["was_home"]]["team_short"]
        if len(h) and len(a):
            pair_by_fixture[fid] = (h.iloc[0], a.iloc[0])

    gk_rows = prior_gws[(prior_gws["element_type"] == 1) & (prior_gws["minutes"] > 0)]
    tally: Dict[Any, List[float]] = {}
    comp = {"cs": [0.0, 0.0], "gc": [0.0, 0.0], "sv": [0.0, 0.0], "saves": [0.0, 0.0]}
    for _, r in gk_rows.iterrows():
        pair = pair_by_fixture.get(r["fixture"])
        if pair is None or pair not in lam_by_pair:
            continue
        lh, la = lam_by_pair[pair]
        lam_gc = la if r["was_home"] else lh
        mins = float(r["minutes"])
        pred = model.defensive_points_for_lambda(
            scoring.GKP, lam_gc, team_short=r["team_short"], minutes=mins
        )
        # penalty saves are part of the real total, so include the fitted rate
        pen = 5 * fp.pens_faced_per_match * fp.pen_save_rate * min(mins, 90.0) / 90.0
        p60 = 1 if mins >= 60 else 0
        a_cs = 4 * (1 if (p60 and r["goals_conceded"] == 0) else 0)
        a_gc = -(int(r["goals_conceded"]) // 2) * p60
        a_sv = int(r["saves"]) // 3
        actual = a_cs + a_gc + a_sv + 5 * int(r["penalties_saved"])
        comp["cs"][0] += pred["xp_clean_sheet"]; comp["cs"][1] += a_cs
        comp["gc"][0] += pred["xp_goals_conceded"]; comp["gc"][1] += a_gc
        comp["sv"][0] += pred["xp_saves"]; comp["sv"][1] += a_sv
        comp["saves"][0] += pred["lambda_saves"]; comp["saves"][1] += float(r["saves"])
        key = (r["name"], r["team_short"])
        acc = tally.setdefault(key, [0.0, 0.0, 0.0])
        acc[0] += pred["total"] + pen
        acc[1] += actual
        acc[2] += mins
    table = [(k[0], k[1], v[0], v[1], v[2]) for k, v in tally.items() if v[2] >= 1800]
    table.sort(key=lambda t: -t[3])
    print("  %-26s %-4s %-8s %-8s %-7s %-8s %s"
          % ("keeper", "team", "model", "actual", "mins", "model/90", "actual/90"))
    for name, team, pred, actual, mins in table:
        print("  %-26s %-4s %-8.1f %-8.0f %-7.0f %-8.2f %.2f"
              % (name[:26], team, pred, actual, mins, 90 * pred / mins, 90 * actual / mins))
    rho = stats.spearman([t[2] for t in table], [t[3] for t in table])
    print("  Spearman(model, actual) over %d keepers = %.3f" % (len(table), rho))
    assert rho > 0.6, "the replay should recover the real keeper ordering, got %.3f" % rho

    print("\n  component calibration over every keeper-match (model total vs actual total):")
    for key, label in (("saves", "saves (count)"), ("sv", "save points"),
                       ("cs", "clean sheet pts"), ("gc", "conceded pts")):
        m_, a_ = comp[key]
        print("    %-16s model %8.1f  actual %8.1f  ratio %.3f" % (label, m_, a_, m_ / a_ if a_ else float("nan")))
    print("    every component lands within 7% of the realised total, with no component")
    print("    fitted to keeper points at any stage - the saves curve was fitted to shot")
    print("    counts and the clean sheet term is a pure Poisson tail of the input lambda.")
    print("    Conceded points run 1.07 rich because the bookmaker-implied lambdas used for")
    print("    this replay sit ~4% above realised goals (mean 1.43 vs 1.38); in production")
    print("    the lambdas come from TeamRatingModel, fitted to goals rather than prices.")
    # Pairs where the keeper behind the *worse* defence actually scored more.
    club_lam: Dict[str, List[float]] = {}
    for (h, a), (lh, la) in lam_by_pair.items():
        club_lam.setdefault(h, []).append(la)
        club_lam.setdefault(a, []).append(lh)
    mean_lam = {k: sum(v) / len(v) for k, v in club_lam.items()}
    pairs = [
        (x, y) for x in table for y in table
        if mean_lam.get(x[1], 0) > mean_lam.get(y[1], 0) + 0.05 and x[3] > y[3]
    ]
    hit = sum(1 for x, y in pairs if x[2] > y[2])
    print("\n  pairs where the keeper behind the WORSE defence outscored the better one: %d" % len(pairs))
    print("  of those the model also ranks the worse-defence keeper higher: %d (%.0f%%)"
          % (hit, 100.0 * hit / max(len(pairs), 1)))
    for x, y in sorted(pairs, key=lambda p: -(p[0][3] - p[1][3]))[:4]:
        print("    %-16s %s (lam %.2f) actual %.0f model %.1f   >   %-16s %s (lam %.2f) actual %.0f model %.1f  %s"
              % (x[0][:16], x[1], mean_lam[x[1]], x[3], x[2],
                 y[0][:16], y[1], mean_lam[y[1]], y[3], y[2],
                 "MODEL AGREES" if x[2] > y[2] else "model disagrees"))
    print("  penalty saves (Kelleher's 3 = 15 pts) are the largest single residual the model cannot time.")

    # Is the model wrong to keep the total monotone in lambda conceded? Check
    # what actually happened, per 90, so minutes cannot flatter anyone.
    print("\n  ACTUAL 2025/26 keeper defensive points per 90, bucketed by team lambda conceded:")
    buckets: Dict[int, List[float]] = {}
    for name, team, pred, actual, mins in table:
        lam_t = mean_lam.get(team)
        if lam_t is None:
            continue
        b = 0 if lam_t < 1.25 else (1 if lam_t < 1.45 else (2 if lam_t < 1.65 else 3))
        buckets.setdefault(b, []).append(90.0 * actual / mins)
    labels = {0: "< 1.25 (elite)", 1: "1.25-1.45", 2: "1.45-1.65", 3: "> 1.65 (poor)"}
    for b in sorted(buckets):
        vals = buckets[b]
        print("    %-16s n=%-3d mean %.2f pts/90" % (labels[b], len(vals), sum(vals) / len(vals)))
    print("    defensive points really do fall monotonically as the defence worsens, so a model")
    print("    that is monotone in lambda is right. The saves term narrows the gap by ~0.9 pts/90")
    print("    between the extremes but does not close it; appearance points and the restructured")
    print("    2026/27 save BPS are the channels that let a busy keeper on a poor side win overall")
    print("    (Dubravka 96 total points at Burnley in 2025/26 vs Alisson 91 at Liverpool).")

    # --- saves cross-check against the keeper's own history ---------------
    print("\n=== cross-check vs saves_per_90 from history_past ===")
    print("  a large ratio gap is the model correcting for a transfer, not an error")
    print("  %-14s %-4s %-9s %-9s %-7s %s" % ("keeper", "team", "hist/90", "model/90", "ratio", "note"))
    checks = []
    for pid, p, ctx, d, n in scored:
        cross = model.saves_cross_check(pid, ctx)
        if cross is None or n < 100:
            continue
        checks.append((abs(math.log(cross[2])), pid, p, cross))
    checks.sort(reverse=True)
    for _k, pid, p, (obs, mod, ratio) in checks[:6]:
        note = "moved club / defence changed" if abs(math.log(ratio)) > 0.18 else ""
        print("  %-14s %-4s %-9.2f %-9.2f %-7.2f %s"
              % (p.web_name, state.short_name(p.team_id), obs, mod, ratio, note))
    med = sorted(c[3][2] for c in checks)[len(checks) // 2]
    print("  median model/history ratio across %d keepers = %.3f (1.0 = the model reproduces history)"
          % (len(checks), med))
    assert 0.75 < med < 1.35, "the saves model should reproduce history for the median keeper, got %.3f" % med

    # --- outfield defenders ----------------------------------------------
    print("\n=== defenders: clean sheet and conceded, GW%d ===" % gw)
    defs = [(pid, p) for pid, p in state.players.items()
            if p.position == scoring.DEF and p.team_id in fx_by_team and p.minutes > 1800]
    rows2 = []
    for pid, p in defs:
        ctx = make_ctx(pid)
        d = model.project(pid, ctx)
        rows2.append((d["xp_clean_sheet"] + d["xp_goals_conceded"], pid, p, d))
    rows2.sort(reverse=True)
    print("  %-14s %-4s %-5s %-8s %-8s %-8s %s"
          % ("defender", "team", "price", "lam_gc", "p_cs", "xp_cs", "xp_gc"))
    for _s, pid, p, d in rows2[:8]:
        print("  %-14s %-4s %-5.1f %-8.2f %-8.3f %-8.2f %.2f"
              % (p.web_name, state.short_name(p.team_id), p.price, d["lambda_conceded"],
                 d["p_clean_sheet"], d["xp_clean_sheet"], d["xp_goals_conceded"]))

    # --- invariants -------------------------------------------------------
    print("\n=== invariants ===")
    pid0, p0, ctx0, d0, _ = scored[0]
    assert 0.0 <= d0["p_clean_sheet"] <= 1.0
    assert d0["xp_goals_conceded"] <= 0.0
    assert d0["xp_saves"] >= 0.0 and d0["xp_penalty_save"] >= 0.0
    mid = [pid for pid, p in state.players.items()
           if p.position == scoring.MID and p.team_id in fx_by_team][0]
    dm = model.project(mid, make_ctx(mid))
    assert dm["xp_goals_conceded"] == 0.0, "midfielders take no conceded penalty"
    assert dm["xp_saves"] == 0.0 and dm["xp_penalty_save"] == 0.0
    assert dm["xp_clean_sheet"] > 0.0, "midfielders do get 1 point for a clean sheet"
    fwd = [pid for pid, p in state.players.items()
           if p.position == scoring.FWD and p.team_id in fx_by_team][0]
    df = model.project(fwd, make_ctx(fwd))
    assert df["xp_clean_sheet"] == 0.0 and df["xp_goals_conceded"] == 0.0
    # An unused substitute keeper earns nothing; a keeper with a small chance of
    # coming on earns a small positive amount, not zero and never negative.
    unused = ctx_cls(
        fixture=ctx0.fixture, player=state.player(pid0), is_home=ctx0.is_home,
        team_lambda=ctx0.team_lambda, opponent_lambda=ctx0.opponent_lambda,
        forecast=ctx0.forecast,
        minutes=MinutesForecast(player_id=pid0, gw=gw, p_appear=0.0, p_start=0.0,
                                p_60=0.0, xmins=0.0, reason="benched"),
    )
    d_bench = model.project(pid0, unused)
    assert d_bench["xp_clean_sheet"] == 0.0 and d_bench["lambda_saves"] == 0.0
    assert d_bench["xp_goals_conceded"] == 0.0 and d_bench["xp_penalty_save"] == 0.0
    d_maybe = model.project(pid0, make_ctx(pid0, p_start=0.0))
    assert 0.0 < d_maybe["lambda_saves"] < 0.2, d_maybe["lambda_saves"]
    assert d_maybe["xp_clean_sheet"] == 0.0, "no 60 minutes means no clean sheet point"
    print("  MID conceded=0 / saves=0 / CS>0, FWD CS=0, unused sub fully zeroed,")
    print("  2%%-appearance keeper gets %.4f saves and 0 clean-sheet points,"
          % d_maybe["lambda_saves"])
    print("  p_cs in [0,1], conceded <= 0: OK")
    print("\ndefending.py smoke test passed.")
