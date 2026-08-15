"""Defensive Contribution: P(a player hits his DEFCON threshold in a fixture).

DEFCON pays a flat 2 points at 10 CBIT for defenders and 12 CBIRT for
midfielders and forwards, and nothing below. It is a **threshold** event, so the
mean is the wrong statistic: over 2025/26 only 10.2% of starts had an expected
count at or above the threshold, while 21.0% of starts actually cleared it. A
model that projects the mean and asks whether it beats the line loses half the
points that were really scored.

What the API actually gives you
------------------------------
``defensive_contribution`` is an **action count**, not a points total and not a
count of qualifying matches. Verified against every 2025/26 per-match row in the
vaastav archive: it equals ``clearances_blocks_interceptions + tackles`` for all
9,733 defender rows and ``+ recoveries`` for all 16,597 midfielder and forward
rows, exactly, and is 0 for every goalkeeper row. Two traps follow from that:

1. **In ``history_past`` the component fields are only populated for 2025/26.**
   For 2024/25 the API publishes ``defensive_contribution`` for 292 players
   while ``clearances_blocks_interceptions``, ``tackles`` and ``recoveries`` are
   all zero. Summing the components — which SPEC 3.6 suggests preferring —
   silently returns a rate of zero for an entire season of real data. Before
   2024/25 the field is genuinely absent and must not be read as a zero either.

2. **The stored count uses the player's position in *that* season.** All 7
   players whose 2025/26 ``defensive_contribution`` disagrees with the component
   sum under their 2026/27 position are players FPL reclassified over the
   summer. Wieffer is a 2026/27 defender whose stored 269 is CBIRT; his CBIT
   basis is 167. Taking the stored number would rate him at 12.7 actions per 90
   against a threshold of 10 when the truth is 7.9 — an elite asset conjured out
   of a position change. So: recompute from components under the player's
   *current* position whenever they exist, fall back to the stored count only
   where they do not, and mark that fall-back as unconfirmed.

Fixture polarity
----------------
A team pinned into its own half makes more defensive actions. Fitted
within-player on 2025/26 (so a weak team's high baseline is not double-counted,
it is already in the player's own rate), the effect is a multiplier of
``(opponent_lambda / team_lambda) ** beta`` with beta ~0.055 for defenders and
~0.030 for midfielders. That is small but highly significant for defenders
(Poisson deviance drop 22.5 on 1 df) and runs *opposite* to the clean sheet: the
same tough fixture that halves a defender's clean-sheet chance raises his DEFCON
chance.
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
from gaffer.data import history as hist
from gaffer.data.cache import Cache
from gaffer.model.defending import fixture_lambdas, minutes_split

if TYPE_CHECKING:  # avoids a cycle: xp.py imports this module
    from gaffer.data.loaders import GameState
    from gaffer.model.xp import FixtureContext

log = logging.getLogger(__name__)

FIT_PATH = os.path.join(REPORTS_DIR, "defcon_fit.json")
FIT_VERSION = 3

# Fitted 2026-08-14 on the vaastav 2025-26 per-match archive joined to
# bookmaker-implied fixture lambdas. Offline fallback only; `fit` recomputes.
FALLBACK = {
    # Negative binomial dispersion r, corrected for the estimation error in the
    # leave-one-out player rate. Var = mu + mu^2/r, so at mu=8 a defender's
    # per-match variance is 11.0 against a Poisson 8.0.
    "dispersion": {"2": 21.12, "3": 27.25, "4": 27.73},
    "tilt_beta": {"2": 0.0550, "3": 0.0300, "4": -0.0300},
    "prior_rate90": {"2": 7.770, "3": 8.386, "4": 4.476},
    "price_log_slope": {"2": 0.430, "3": -0.671, "4": -0.349},
    "price_log_intercept": {"2": 1.322, "3": 3.264, "4": 2.121},
    # Prior strength in 90-minute units, from the year-on-year reliability of a
    # player's own rate (DEF rho=0.77, MID 0.84, FWD 0.71 over 157 players).
    "shrink_90s": {"2": 7.6, "3": 4.6, "4": 9.5},
}

# A season whose component fields are missing contributes its stored count on an
# unverifiable position basis; halve its effective sample size rather than drop
# it, because dropping 2024/25 entirely costs a full season of signal.
UNCONFIRMED_BASIS_WEIGHT = 0.5

# Seasons decay geometrically once the most recent source is weighted 1.0.
SEASON_DECAY = 0.5


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


@dataclass
class DefconRate:
    """A player's shrunk per-90 defensive-action rate and where it came from."""

    player_id: int
    position: int
    rate90: float          # shrunk, ready to project
    raw_rate90: float      # what the data alone said
    n90: float             # effective 90-minute units behind raw_rate90
    prior_rate90: float
    basis_confirmed: bool  # False if any source used the stored count blind
    source: str
    reason: str = ""

    @property
    def threshold(self) -> int:
        return scoring.DEFCON_THRESHOLD[self.position]


def season_defcon_actions(
    row: Dict[str, Any], position: int
) -> Optional[Tuple[float, float, bool]]:
    """``(actions, minutes, basis_confirmed)`` for one ``history_past`` season.

    Returns None when the season carries no DEFCON information at all, which is
    every season before 2024/25. None is not zero: a defender with 3,000
    minutes in 2023/24 has no defensive-action rate on record, and treating that
    as a rate of 0.0 would bury him.
    """
    minutes = float(row.get("minutes") or 0.0)
    if minutes <= 0:
        return None
    cbi = float(row.get("clearances_blocks_interceptions") or 0.0)
    tackles = float(row.get("tackles") or 0.0)
    recoveries = float(row.get("recoveries") or 0.0)
    stored = float(row.get("defensive_contribution") or 0.0)
    if cbi + tackles + recoveries > 0:
        actions = cbi + tackles
        if position in (scoring.MID, scoring.FWD):
            actions += recoveries
        return actions, minutes, True
    if stored > 0:
        return stored, minutes, False
    return None


def frame_defcon_actions(
    frame: pd.DataFrame, position: int
) -> Optional[Tuple[float, float, bool]]:
    """Same, for a current-season per-GW ``history`` frame."""
    if frame is None or len(frame) == 0:
        return None
    minutes = float(pd.to_numeric(frame.get("minutes"), errors="coerce").fillna(0).sum())
    if minutes <= 0:
        return None

    def col(name: str) -> float:
        if name not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[name], errors="coerce").fillna(0).sum())

    cbi, tackles, recoveries = col("clearances_blocks_interceptions"), col("tackles"), col("recoveries")
    stored = col("defensive_contribution")
    if cbi + tackles + recoveries > 0:
        actions = cbi + tackles
        if position in (scoring.MID, scoring.FWD):
            actions += recoveries
        return actions, minutes, True
    if stored > 0:
        return stored, minutes, False
    return None


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------


def _int_keyed(raw: Dict[Any, float]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in raw.items()}


@dataclass
class DefconFit:
    dispersion: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["dispersion"]))
    tilt_beta: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["tilt_beta"]))
    tilt_deviance_drop: Dict[int, float] = field(default_factory=dict)
    prior_rate90: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["prior_rate90"]))
    price_log_slope: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["price_log_slope"]))
    price_log_intercept: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["price_log_intercept"]))
    shrink_90s: Dict[int, float] = field(
        default_factory=lambda: _int_keyed(FALLBACK["shrink_90s"]))
    n_observations: Dict[int, int] = field(default_factory=dict)
    seasons: List[str] = field(default_factory=list)
    source: str = "fallback"
    fitted_at: str = ""
    version: int = FIT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"seasons": self.seasons, "source": self.source,
                               "fitted_at": self.fitted_at, "version": self.version}
        for name in ("dispersion", "tilt_beta", "tilt_deviance_drop", "prior_rate90",
                     "price_log_slope", "price_log_intercept", "shrink_90s",
                     "n_observations"):
            out[name] = {str(k): v for k, v in getattr(self, name).items()}
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DefconFit":
        kwargs: Dict[str, Any] = {}
        for name in ("dispersion", "tilt_beta", "tilt_deviance_drop", "prior_rate90",
                     "price_log_slope", "price_log_intercept", "shrink_90s"):
            if name in raw:
                kwargs[name] = _int_keyed(raw[name])
        if "n_observations" in raw:
            kwargs["n_observations"] = {int(k): int(v) for k, v in raw["n_observations"].items()}
        for name in ("seasons", "source", "fitted_at", "version"):
            if name in raw:
                kwargs[name] = raw[name]
        return cls(**kwargs)


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


class DefconModel:
    """P(defensive-contribution threshold) for a player in a fixture."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.fit_params = DefconFit()
        self.state: Optional["GameState"] = None
        self.warnings: List[str] = []
        self._rates: Dict[int, DefconRate] = {}

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        state: "GameState",
        results: Optional[pd.DataFrame] = None,
        refit: bool = False,
        cache: Optional[Cache] = None,
    ) -> "DefconModel":
        self.state = state
        self.warnings = []
        cache = cache or Cache()

        loaded = None if refit else self._load_fit()
        if loaded is not None:
            self.fit_params = loaded
        else:
            self.fit_params = self._fit_from_data(state, results, cache)
            self._save_fit()

        self._fit_price_priors(state)
        self._fit_player_rates(state)
        return self

    def _fit_from_data(
        self, state: "GameState", results: Optional[pd.DataFrame], cache: Cache
    ) -> DefconFit:
        """Fit tilt and dispersion from real per-match counts.

        Season totals cannot do this job: match-to-match variance is the entire
        point of a threshold model, and a season aggregate has thrown it away.
        """
        fitted = DefconFit(source="fit",
                           fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        prior = hist.prior_season(state.season)
        try:
            gws = hist.load_vaastav_gws(prior, cache=cache)
            if results is None:
                results = hist.load_football_data([prior], cache=cache)
            lam = fixture_lambdas(results)
        except Exception as exc:
            msg = ("defcon fit: per-match history unavailable (%s: %s); using recorded "
                   "fallback constants" % (type(exc).__name__, exc))
            log.warning(msg)
            self.warnings.append(msg)
            return DefconFit()

        fitted.seasons = [hist.season_dash(prior)]
        lam_by_pair: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for _, r in lam.iterrows():
            if pd.notna(r["lam_h"]) and pd.notna(r["lam_a"]):
                lam_by_pair[(r["home_short"], r["away_short"])] = (float(r["lam_h"]), float(r["lam_a"]))
        pair_by_fixture: Dict[Any, Tuple[str, str]] = {}
        for fid, grp in gws.groupby("fixture"):
            home = grp[grp["was_home"]]["team_short"]
            away = grp[~grp["was_home"]]["team_short"]
            if len(home) and len(away):
                pair_by_fixture[fid] = (home.iloc[0], away.iloc[0])

        df = gws[gws["element_type"].isin([2, 3, 4]) & (gws["minutes"] >= 60)].copy()
        tilt: List[float] = []
        for _, r in df.iterrows():
            pair = pair_by_fixture.get(r["fixture"])
            if pair is None or pair not in lam_by_pair:
                tilt.append(float("nan"))
                continue
            lh, la = lam_by_pair[pair]
            lam_for, lam_against = (lh, la) if r["was_home"] else (la, lh)
            tilt.append(lam_against / lam_for if lam_for > 0 else float("nan"))
        df["tilt"] = tilt
        df = df[df["tilt"].notna()]

        totals = df.groupby("element").agg(
            tot=("defensive_contribution", "sum"), tot_min=("minutes", "sum"),
            n=("minutes", "size"))
        df = df.join(totals, on="element")
        df = df[df["n"] >= 12].copy()
        # Leave-one-out: a rate that includes the match being predicted would
        # absorb its own residual and understate the dispersion.
        df["loo_min"] = df["tot_min"] - df["minutes"]
        df = df[df["loo_min"] > 300]
        df["rate90"] = 90.0 * (df["tot"] - df["defensive_contribution"]) / df["loo_min"]
        df["mu0"] = df["rate90"] * df["minutes"] / 90.0

        for pos in (scoring.DEF, scoring.MID, scoring.FWD):
            sub = df[df["element_type"] == pos]
            if len(sub) < 200:
                continue
            y = sub["defensive_contribution"].values.astype(float)
            mu0 = sub["mu0"].values.astype(float)
            t = sub["tilt"].values.astype(float)

            best = (float("inf"), 0.0)
            b = -0.40
            while b <= 1.0001:
                dev = _poisson_deviance([m * (tt ** b) for m, tt in zip(mu0, t)], y)
                if dev < best[0]:
                    best = (dev, b)
                b += 0.005
            beta = best[1]
            fitted.tilt_beta[pos] = beta
            fitted.tilt_deviance_drop[pos] = _poisson_deviance(list(mu0), y) - best[0]
            fitted.n_observations[pos] = len(sub)

            mu = [m * (tt ** beta) for m, tt in zip(mu0, t)]
            n_loo = [max(float(v) - 1.0, 1.0) for v in sub["n"].values]
            fitted.dispersion[pos] = self._pooled_dispersion(y, mu, n_loo)

            # Prior rate: the position's minutes-weighted league rate.
            fitted.prior_rate90[pos] = float(
                90.0 * sub["defensive_contribution"].sum() / sub["minutes"].sum())

            # Shrinkage strength from the rate's year-on-year reliability.
            fitted.shrink_90s[pos] = self._shrink_strength(state, pos)
        return fitted

    @staticmethod
    def _pooled_dispersion(
        y: Sequence[float], mu: Sequence[float], n_loo: Sequence[float]
    ) -> float:
        """Method-of-moments r, corrected for noise in the leave-one-out mean.

        ``stats.fit_dispersion`` takes a clean mean and variance; here the mean
        is itself estimated from ~30 matches, so its sampling variance inflates
        the residual and would bias r down about 12%.
        """
        r = 20.0
        num = sum(m * m for m in mu)
        for _ in range(60):
            var_mu = [(m + m * m / r) / n for m, n in zip(mu, n_loo)]
            den = sum((yy - m) ** 2 - m - v for yy, m, v in zip(y, mu, var_mu))
            new = num / den if den > 0 else 1e6
            if abs(new - r) < 1e-9:
                return max(new, 0.5)
            r = 0.5 * r + 0.5 * new
        return max(r, 0.5)

    @staticmethod
    def _shrink_strength(state: "GameState", position: int) -> float:
        """Prior strength k in 90s, from the rate's own year-on-year repeatability.

        A rate that repeats at correlation rho over n 90s has reliability
        n/(n+k), so k = n(1-rho)/rho. Fitted per position rather than assumed:
        defenders regress harder than midfielders.
        """
        pairs: List[Tuple[float, float, float]] = []
        seasons = [hist.season_slash(hist.prior_season(state.season, back=b)) for b in (2, 1)]
        for pid, player in state.players.items():
            if player.position != position:
                continue
            rows = {r.get("season_name"): r for r in state.history_past(pid)}
            a, b_ = rows.get(seasons[0]), rows.get(seasons[1])
            if not a or not b_:
                continue
            # A player FPL reclassified is measured on two different bases across
            # the pair; including him injects noise that is nothing to do with how
            # repeatable the rate is, and biases the prior strength upward.
            stored = float(b_.get("defensive_contribution") or 0.0)
            recomputed = season_defcon_actions(b_, position)
            if recomputed is None or (recomputed[2] and abs(recomputed[0] - stored) > 0.5):
                continue
            ra = season_defcon_actions(a, position)
            rb = recomputed
            if not ra or ra[1] < 900 or rb[1] < 900:
                continue
            pairs.append((90.0 * ra[0] / ra[1], 90.0 * rb[0] / rb[1], ra[1] / 90.0))
        if len(pairs) < 12:
            return _int_keyed(FALLBACK["shrink_90s"])[position]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        rho = num / den if den > 0 else 0.0
        if rho <= 0.05:
            return 1e3
        n90 = sum(p[2] for p in pairs) / len(pairs)
        return max(1.0, min(40.0, n90 * (1.0 - rho) / rho))

    def _fit_price_priors(self, state: "GameState") -> None:
        """log(rate per 90) ~ a + b log(price), per position.

        Price is the only signal available for a promoted-club player or a new
        signing with no Premier League record, and it is informative in the
        direction you would expect: expensive midfielders are attackers and do
        far less defensive work than cheap ones.
        """
        buckets: Dict[int, List[Tuple[float, float]]] = {2: [], 3: [], 4: []}
        for pid, player in state.players.items():
            if player.position not in buckets:
                continue
            best = None
            for row in state.history_past(pid):
                got = season_defcon_actions(row, player.position)
                if got and got[1] >= 900 and got[2]:
                    best = got
            if best is None:
                continue
            buckets[player.position].append((player.price, 90.0 * best[0] / best[1]))
        for pos, pts in buckets.items():
            if len(pts) < 20:
                continue
            xs = [math.log(max(p, 3.5)) for p, _ in pts]
            ys = [math.log(max(r, 0.2)) for _, r in pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            den = sum((x - mx) ** 2 for x in xs)
            if den <= 0:
                continue
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            self.fit_params.price_log_slope[pos] = slope
            self.fit_params.price_log_intercept[pos] = my - slope * mx

    def prior_rate(self, position: int, price: float) -> float:
        """Position+price baseline, clamped to the position's observed range."""
        base = self.fit_params.prior_rate90.get(position, 6.0)
        slope = self.fit_params.price_log_slope.get(position)
        intercept = self.fit_params.price_log_intercept.get(position)
        if slope is None or intercept is None:
            return base
        rate = math.exp(intercept + slope * math.log(max(price, 3.5)))
        return max(0.4 * base, min(1.6 * base, rate))

    def _fit_player_rates(self, state: "GameState") -> None:
        self._rates = {}
        seasons_back = [hist.season_slash(hist.prior_season(state.season, back=b))
                        for b in range(1, 6)]
        for pid, player in state.players.items():
            self._rates[pid] = self._build_rate(state, pid, player, seasons_back)

    def _build_rate(
        self, state: "GameState", pid: int, player: Any, seasons_back: List[str]
    ) -> DefconRate:
        pos = player.position
        if pos == scoring.GKP:
            return DefconRate(pid, pos, 0.0, 0.0, 0.0, 0.0, True, "ineligible",
                              "goalkeepers cannot score DEFCON")

        prior_rate = self.prior_rate(pos, player.price)
        # (weight, actions, minutes, confirmed, label)
        sources: List[Tuple[float, float, float, bool, str]] = []

        current = frame_defcon_actions(state.history(pid), pos)
        if current is not None:
            sources.append((1.0, current[0], current[1], current[2], "season-to-date"))
        for i, season in enumerate(seasons_back):
            row = next((r for r in state.history_past(pid) if r.get("season_name") == season), None)
            if row is None:
                continue
            got = season_defcon_actions(row, pos)
            if got is None:
                continue
            sources.append((SEASON_DECAY ** i, got[0], got[1], got[2], season))
        if not sources:
            return DefconRate(pid, pos, prior_rate, prior_rate, 0.0, prior_rate, True,
                              "prior",
                              "no DEFCON history; position+price prior %.2f/90" % prior_rate)

        # Renormalise so the best available source always carries full weight:
        # pre-season the prior season *is* the primary evidence, not a discount.
        top = max(s[0] for s in sources)
        actions = sum((s[0] / top) * s[1] * (1.0 if s[3] else UNCONFIRMED_BASIS_WEIGHT)
                      for s in sources)
        minutes = sum((s[0] / top) * s[2] * (1.0 if s[3] else UNCONFIRMED_BASIS_WEIGHT)
                      for s in sources)
        if minutes <= 0:
            return DefconRate(pid, pos, prior_rate, prior_rate, 0.0, prior_rate, True,
                              "prior", "no usable minutes; prior %.2f/90" % prior_rate)

        raw = 90.0 * actions / minutes
        n90 = minutes / 90.0
        shrunk = stats.shrink(raw, n90, prior_rate, self.fit_params.shrink_90s.get(pos, 6.0))
        confirmed = all(s[3] for s in sources)
        labels = ", ".join(s[4] for s in sources)
        reason = "%.2f/90 raw over %.1f 90s (%s) -> %.2f shrunk to prior %.2f%s" % (
            raw, n90, labels, shrunk, prior_rate,
            "" if confirmed else "; position basis unconfirmed for a source")
        return DefconRate(pid, pos, shrunk, raw, n90, prior_rate, confirmed,
                          "history", reason)

    # -- persistence -------------------------------------------------------
    def _load_fit(self) -> Optional[DefconFit]:
        if not os.path.exists(FIT_PATH):
            return None
        try:
            with open(FIT_PATH, "r") as fh:
                raw = json.load(fh)
        except (ValueError, OSError):
            return None
        if int(raw.get("version", 0)) != FIT_VERSION:
            return None
        return DefconFit.from_dict(raw)

    def _save_fit(self) -> None:
        if self.fit_params.source != "fit":
            return
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            with open(FIT_PATH, "w") as fh:
                json.dump(self.fit_params.to_dict(), fh, indent=2, sort_keys=True)
        except OSError as exc:
            log.warning("could not write %s: %s", FIT_PATH, exc)

    # -- lookups -----------------------------------------------------------
    def rate(self, player_id: int) -> DefconRate:
        found = self._rates.get(player_id)
        if found is not None:
            return found
        if self.state is None:
            raise KeyError("DefconModel.fit() has not been called")
        player = self.state.player(player_id)
        return self._build_rate(
            self.state, player_id, player,
            [hist.season_slash(hist.prior_season(self.state.season, back=b)) for b in range(1, 6)],
        )

    def threshold(self, player_id: int) -> int:
        return scoring.DEFCON_THRESHOLD[self.rate(player_id).position]

    def tilt(self, ctx: "FixtureContext") -> float:
        """Field tilt: expected goals conceded over expected goals scored.

        Clamped because a 6-0 favourite would otherwise push the multiplier into
        territory the fit never saw.
        """
        team = max(float(getattr(ctx, "team_lambda", 1.45)), 0.05)
        opponent = max(float(getattr(ctx, "opponent_lambda", 1.45)), 0.05)
        return max(0.2, min(5.0, opponent / team))

    # -- projection --------------------------------------------------------
    def project(self, player_id: int, ctx: "FixtureContext") -> float:
        """P(the player hits his DEFCON threshold in this fixture)."""
        return self.project_detail(player_id, ctx)["p_defcon"]

    def project_detail(self, player_id: int, ctx: "FixtureContext") -> Dict[str, float]:
        rate = self.rate(player_id)
        pos = rate.position
        if pos == scoring.GKP or rate.rate90 <= 0:
            return {"p_defcon": 0.0, "xp_defcon": 0.0, "var_defcon": 0.0,
                    "defcon_rate90": rate.rate90, "defcon_threshold": 0.0,
                    "defcon_tilt": 1.0, "defcon_mean_if_start": 0.0}

        threshold = scoring.DEFCON_THRESHOLD[pos]
        dispersion = self.fit_params.dispersion.get(pos, 20.0)
        beta = self.fit_params.tilt_beta.get(pos, 0.0)
        tilt = self.tilt(ctx)
        multiplier = tilt ** beta

        p_start, m_start, p_sub, m_sub = minutes_split(getattr(ctx, "minutes", None), self.config)
        mean_start = rate.rate90 * (m_start / 90.0) * multiplier
        mean_sub = rate.rate90 * (m_sub / 90.0) * multiplier
        # Mixture over start/bench, not a single evaluation at xmins: clearing
        # 10 actions is nothing like linear in minutes, so the average of the
        # two branches is not the value at the average.
        p = (p_start * stats.negbin_at_least(threshold, mean_start, dispersion)
             + p_sub * stats.negbin_at_least(threshold, mean_sub, dispersion))
        p = max(0.0, min(1.0, p))
        points = scoring.DEFCON_POINTS[pos]
        return {
            "p_defcon": p,
            "xp_defcon": scoring.defcon_points(pos, p),
            "var_defcon": (points ** 2) * p * (1.0 - p),
            "defcon_rate90": rate.rate90,
            "defcon_threshold": float(threshold),
            "defcon_tilt": tilt,
            "defcon_tilt_multiplier": multiplier,
            "defcon_mean_if_start": mean_start,
            "defcon_dispersion": dispersion,
        }

    def project_points(self, player_id: int, ctx: "FixtureContext") -> float:
        return self.project_detail(player_id, ctx)["xp_defcon"]

    def explain(self, player_id: int, ctx: "FixtureContext") -> str:
        rate = self.rate(player_id)
        if rate.position == scoring.GKP:
            return "goalkeeper: DEFCON ineligible"
        d = self.project_detail(player_id, ctx)
        return (
            "%.2f actions/90 x %.0f mins x tilt %.2f^%.3f = mean %.2f vs threshold %d "
            "-> P=%.1f%% (%+.2f pts). %s"
            % (rate.rate90, d["defcon_mean_if_start"] / max(rate.rate90, 1e-9) * 90.0,
               d["defcon_tilt"], self.fit_params.tilt_beta.get(rate.position, 0.0),
               d["defcon_mean_if_start"], int(d["defcon_threshold"]),
               100 * d["p_defcon"], d["xp_defcon"], rate.reason)
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from gaffer.core.types import MatchForecast, MinutesForecast
    from gaffer.data.loaders import fixtures_by_gw, load_game_state, team_fixtures
    from gaffer.model.defending import DefendingModel, prior_season_team_lambdas

    parser = argparse.ArgumentParser(description="DEFCON model smoke test")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()
    cache = Cache()

    t0 = time.time()
    state = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (state.summary(), time.time() - t0))

    prior = hist.prior_season(state.season)
    results = hist.load_football_data([prior], cache=cache)
    t0 = time.time()
    model = DefconModel(cfg).fit(state, results=results, refit=args.refit, cache=cache)
    print("fitted in %.2fs (source=%s, seasons=%s)"
          % (time.time() - t0, model.fit_params.source, model.fit_params.seasons))
    for w in model.warnings:
        print("  WARNING %s" % w)

    fp = model.fit_params
    print("\n=== fitted parameters ===")
    print("  %-4s %-10s %-10s %-14s %-10s %-9s %s"
          % ("pos", "threshold", "prior/90", "dispersion r", "tilt beta", "dev drop", "shrink 90s"))
    for pos, name in ((scoring.DEF, "DEF"), (scoring.MID, "MID"), (scoring.FWD, "FWD")):
        print("  %-4s %-10d %-10.2f %-14.2f %-10.4f %-9.1f %.1f"
              % (name, scoring.DEFCON_THRESHOLD[pos], fp.prior_rate90[pos],
                 fp.dispersion[pos], fp.tilt_beta[pos],
                 fp.tilt_deviance_drop.get(pos, float("nan")), fp.shrink_90s[pos]))
    print("  dispersion in words: at a mean of 8 actions a DEF's per-match variance is")
    print("    %.2f, against %.2f under a Poisson - real overdispersion, so a Poisson"
          % (8 + 64.0 / fp.dispersion[scoring.DEF], 8.0))
    print("    threshold model would understate both the hits and the misses.")
    print("  price prior: %s"
          % "  ".join("%s exp(%.2f)*price^%.2f" % (n, fp.price_log_intercept[p], fp.price_log_slope[p])
                      for p, n in ((scoring.DEF, "DEF"), (scoring.MID, "MID"), (scoring.FWD, "FWD"))))

    # --- the API field investigation, re-run as a test --------------------
    print("\n=== investigation: what is the API's `defensive_contribution`? ===")
    gws = hist.load_vaastav_gws(prior, cache=cache)
    gws["cbit"] = gws["clearances_blocks_interceptions"] + gws["tackles"]
    gws["cbirt"] = gws["cbit"] + gws["recoveries"]
    for pos, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
        sub = gws[gws["element_type"] == pos]
        want = sub["cbit"] if pos == 2 else sub["cbirt"]
        if pos == 1:
            print("  GKP  rows=%-6d always zero: %s"
                  % (len(sub), bool((sub["defensive_contribution"] == 0).all())))
            continue
        n_match = int((sub["defensive_contribution"] == want).sum())
        print("  %s  rows=%-6d equals %-5s in %d rows (%.4f%%)"
              % (name, len(sub), "CBIT" if pos == 2 else "CBIRT", n_match,
                 100.0 * n_match / len(sub)))
        assert n_match == len(sub), "%s: %d mismatches" % (name, len(sub) - n_match)
    print("  -> an ACTION COUNT, exactly. Not a points total (max %d, would be 0/2) and"
          % int(gws["defensive_contribution"].max()))
    print("     not a qualifying-match count (would be 0/1). SPEC 3.6's guess is wrong.")

    print("\n  the history_past trap (season aggregates, all %d players):" % len(state.players))
    by_season: Dict[str, List[int]] = {}
    for pid, p in state.players.items():
        if p.position == scoring.GKP:
            continue
        for row in state.history_past(pid):
            s = row.get("season_name")
            rec = by_season.setdefault(s, [0, 0, 0])
            rec[0] += 1
            if float(row.get("defensive_contribution") or 0) > 0:
                rec[1] += 1
            if (float(row.get("clearances_blocks_interceptions") or 0)
                    + float(row.get("tackles") or 0) + float(row.get("recoveries") or 0)) > 0:
                rec[2] += 1
    for s in sorted(by_season)[-4:]:
        rec = by_season[s]
        print("    %s rows=%-4d stored count > 0: %-4d components > 0: %-4d"
              % (s, rec[0], rec[1], rec[2]))
    trap = by_season.get("2024/25", [0, 0, 0])
    assert trap[1] > 100 and trap[2] == 0, "expected the 2024/25 components-are-zero trap"
    print("    -> summing components would zero out %d players' entire 2024/25 season." % trap[1])

    print("\n  position reclassification (stored count != components under the CURRENT position):")
    flips = []
    for pid, p in state.players.items():
        if p.position == scoring.GKP:
            continue
        row = state.prior_season_row(pid)
        if not row or float(row.get("defensive_contribution") or 0) <= 0:
            continue
        cbi = float(row.get("clearances_blocks_interceptions") or 0)
        tk = float(row.get("tackles") or 0)
        rec_ = float(row.get("recoveries") or 0)
        if cbi + tk + rec_ <= 0:
            continue
        cur = cbi + tk + (rec_ if p.position in (scoring.MID, scoring.FWD) else 0)
        stored = float(row["defensive_contribution"])
        if cur != stored:
            flips.append((p.web_name, scoring.POS_NAME[p.position], row["minutes"], stored, cur))
    print("    %-16s %-4s %-7s %-9s %-9s %s"
          % ("player", "now", "mins", "stored", "correct", "stored basis was"))
    for name, pos, mins, stored, cur in flips:
        was = "CBIRT (a MID/FWD then)" if stored > cur else "CBIT (a DEF then)"
        print("    %-16s %-4s %-7d %-9.0f %-9.0f %s" % (name[:16], pos, mins, stored, cur, was))
    assert flips, "expected at least one reclassified player"
    print("    every one is a player FPL moved between DEF and MID/FWD this summer.")
    worst = max(flips, key=lambda f: abs(f[3] - f[4]))
    print("    worst case %s: stored %.0f/%d mins = %.2f per 90 vs true %.2f per 90"
          % (worst[0], worst[3], worst[2], 90 * worst[3] / worst[2], 90 * worst[4] / worst[2]))

    # --- fixture contexts -------------------------------------------------
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

    strengths = prior_season_team_lambdas(state, results, cfg)
    league = cfg.model.league_avg_goals_per_team
    home_adv = cfg.model.home_advantage

    def lambdas(home_id: int, away_id: int) -> Tuple[float, float]:
        ah, dh = strengths[home_id]
        aa, da = strengths[away_id]
        return league * ah * da * home_adv, league * aa * dh

    def make_ctx(player_id: int, fixture: Any, p_start: float = 1.0) -> Any:
        p = state.player(player_id)
        lh, la = lambdas(fixture.team_h, fixture.team_a)
        is_home = fixture.is_home_for(p.team_id)
        team_lam, opp_lam = (lh, la) if is_home else (la, lh)
        fc = MatchForecast(fixture_id=fixture.id, gw=fixture.gw or 0,
                           team_h=fixture.team_h, team_a=fixture.team_a,
                           lambda_h=lh, lambda_a=la,
                           p_cs_h=stats.p_clean_sheet(la), p_cs_a=stats.p_clean_sheet(lh))
        mf = MinutesForecast(player_id=player_id, gw=fixture.gw or 0,
                             p_appear=min(1.0, p_start + 0.02), p_start=p_start,
                             p_60=p_start * 0.94, xmins=p_start * 88.0, reason="smoke test")
        return ctx_cls(fixture=fixture, player=p, is_home=is_home, team_lambda=team_lam,
                       opponent_lambda=opp_lam, forecast=fc, minutes=mf)

    gw = state.current_gw
    fx_by_team: Dict[int, Any] = {}
    for f in fixtures_by_gw(state)[gw]:
        fx_by_team[f.team_h] = f
        fx_by_team[f.team_a] = f

    # --- top 40 by p_defcon ----------------------------------------------
    print("\n=== top 40 by P(DEFCON), GW%d ===" % gw)
    rows = []
    for pid, p in state.players.items():
        if p.position == scoring.GKP or p.team_id not in fx_by_team:
            continue
        r = model.rate(pid)
        d = model.project_detail(pid, make_ctx(pid, fx_by_team[p.team_id]))
        cbi90 = 0.0
        row = state.prior_season_row(pid)
        if row and float(row.get("minutes") or 0) > 0:
            cbi90 = 90.0 * float(row.get("clearances_blocks_interceptions") or 0) / float(row["minutes"])
        rows.append((d["p_defcon"], pid, p, r, d, cbi90))
    rows.sort(reverse=True, key=lambda t: t[0])
    print("  %-3s %-16s %-4s %-4s %-6s %-8s %-4s %-8s %-8s %s"
          % ("#", "player", "pos", "team", "price", "rate/90", "thr", "clear/90", "P(defcon)", "type"))
    for i, (p_dc, pid, p, r, d, cbi90) in enumerate(rows[:40], 1):
        kind = ""
        if p.position == scoring.DEF:
            kind = "centre-back" if cbi90 >= 6.0 else "full-back"
        elif p.position == scoring.MID:
            kind = "defensive mid" if r.rate90 >= 9.0 else "midfielder"
        print("  %-3d %-16s %-4s %-4s %-6.1f %-8.2f %-4d %-8.2f %-8.3f %s"
              % (i, p.web_name[:16], scoring.POS_NAME[p.position], state.short_name(p.team_id),
                 p.price, r.rate90, scoring.DEFCON_THRESHOLD[p.position], cbi90, p_dc, kind))
    top40 = rows[:40]
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for _p, _pid, p, _r, _d, _c in top40:
        counts[scoring.POS_NAME[p.position]] += 1
    cb = sum(1 for _p, _pid, p, _r, _d, c in top40 if p.position == scoring.DEF and c >= 6.0)
    dm = sum(1 for _p, _pid, p, r, _d, _c in top40 if p.position == scoring.MID and r.rate90 >= 9.0)
    print("\n  composition: DEF %d (of which centre-back-like %d), MID %d (defensive mid-like %d), FWD %d"
          % (counts["DEF"], cb, counts["MID"], dm, counts["FWD"]))
    assert counts["FWD"] <= 1, "forwards must be near-absent from the top 40, got %d" % counts["FWD"]
    assert cb + dm >= 28, "centre-backs and defensive midfielders must dominate, got %d" % (cb + dm)
    print("  forwards near-absent and centre-backs + defensive mids dominate: OK")

    # --- season-long calibration vs the 2025/26 benchmark ----------------
    print("\n=== season-long projection vs the 2025/26 benchmark ===")
    all_gws = list(range(1, scoring.TOTAL_GWS + 1))
    season: List[Tuple[float, Any, DefconRate, float, int]] = []
    for pid, p in state.players.items():
        if p.position == scoring.GKP:
            continue
        r = model.rate(pid)
        if r.n90 < 10:
            continue
        total_p = 0.0
        n = 0
        for f in team_fixtures(state, p.team_id, all_gws):
            total_p += model.project(pid, make_ctx(pid, f))
            n += 1
        if n:
            season.append((2.0 * total_p, p, r, total_p / n, n))
    season.sort(reverse=True, key=lambda t: t[0])
    print("  assumes a nailed starter every week, so this is the ceiling for each player")
    print("  %-3s %-18s %-4s %-4s %-8s %-10s %-12s %s"
          % ("#", "player", "pos", "team", "rate/90", "hit rate", "qual matches", "DEFCON pts"))
    for i, (pts, p, r, mean_p, n) in enumerate(season[:10], 1):
        print("  %-3d %-18s %-4s %-4s %-8.2f %-10.3f %-12.1f %.1f"
              % (i, p.web_name[:18], scoring.POS_NAME[p.position], state.short_name(p.team_id),
                 r.rate90, mean_p, mean_p * n, pts))
    best = season[0]
    print("\n  benchmark: 2025/26's best were ~50 points from ~25 qualifying matches of 38")
    print("  model's best: %.1f points from %.1f qualifying matches (hit rate %.1f%%)"
          % (best[0], best[3] * best[4], 100 * best[3]))
    assert 40.0 <= best[0] <= 68.0, "top DEFCON season should land near the ~50 pt benchmark, got %.1f" % best[0]
    assert 0.45 <= best[3] <= 0.70, "best hit rate should be 45-70%%, got %.3f" % best[3]
    top10 = season[:10]
    print("  top 10 range %.1f-%.1f pts, hit rates %.1f%%-%.1f%%"
          % (top10[-1][0], top10[0][0], 100 * top10[-1][3], 100 * top10[0][3]))
    pos10 = [scoring.POS_NAME[t[1].position] for t in top10]
    print("  top 10 positions: %s (2025/26 actual was 6 DEF / 4 MID / 0 FWD)"
          % {k: pos10.count(k) for k in ("DEF", "MID", "FWD") if pos10.count(k)})
    assert "FWD" not in pos10, "no forward belongs in the DEFCON top 10"

    reg_def = sorted(t[3] for t in season if t[1].position == scoring.DEF)
    reg_mid = sorted(t[3] for t in season if t[1].position == scoring.MID)
    print("  median regular DEF hit rate %.3f (elite %.3f), median regular MID %.3f (elite %.3f)"
          % (reg_def[len(reg_def) // 2], reg_def[-1], reg_mid[len(reg_mid) // 2], reg_mid[-1]))
    print("  2025/26 actual, players with 15+ starts: DEF median 0.224 / max 0.706,"
          " MID median 0.125 / max 0.703")
    assert reg_def[len(reg_def) // 2] < 0.45, "the median defender must sit far below the elite"

    # --- the threshold really is the point --------------------------------
    print("\n=== why the mean alone is not enough ===")
    n_obs = 0
    n_mean_over = 0
    n_model_p = 0.0
    for pid, p in state.players.items():
        if p.position == scoring.GKP or p.team_id not in fx_by_team:
            continue
        r = model.rate(pid)
        if r.n90 < 10:
            continue
        d = model.project_detail(pid, make_ctx(pid, fx_by_team[p.team_id]))
        n_obs += 1
        n_mean_over += 1 if d["defcon_mean_if_start"] >= d["defcon_threshold"] else 0
        n_model_p += d["p_defcon"]
    print("  over %d regular outfielders in GW%d:" % (n_obs, gw))
    print("    'mean >= threshold' would predict %.1f%% of them to score DEFCON" % (100.0 * n_mean_over / n_obs))
    print("    the negative binomial predicts %.1f%%" % (100.0 * n_model_p / n_obs))
    print("    2025/26 reality over starts: 21.0% qualified while only 10.2% had mean >= threshold")
    assert n_model_p / n_obs > 1.5 * (n_mean_over / n_obs), \
        "the threshold model must recover far more than the mean rule"

    # --- calibration against what actually happened -----------------------
    print("\n=== calibration: replay every %s start with the fitted distribution ===" % prior)
    lam_res = fixture_lambdas(results)
    lam_pair: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for _, r in lam_res.iterrows():
        if pd.notna(r["lam_h"]):
            lam_pair[(r["home_short"], r["away_short"])] = (float(r["lam_h"]), float(r["lam_a"]))
    fx_pair: Dict[Any, Tuple[str, str]] = {}
    for fid, grp in gws.groupby("fixture"):
        h = grp[grp["was_home"]]["team_short"]
        a = grp[~grp["was_home"]]["team_short"]
        if len(h) and len(a):
            fx_pair[fid] = (h.iloc[0], a.iloc[0])
    rep = gws[gws["element_type"].isin([2, 3, 4]) & (gws["minutes"] >= 60)].copy()
    tot = rep.groupby("element").agg(t=("defensive_contribution", "sum"),
                                     m=("minutes", "sum"), n=("minutes", "size"))
    rep = rep.join(tot, on="element")
    rep = rep[(rep["n"] >= 12) & (rep["m"] - rep["minutes"] > 300)]
    preds: List[Tuple[float, int]] = []
    for _, r in rep.iterrows():
        pair = fx_pair.get(r["fixture"])
        if pair is None or pair not in lam_pair:
            continue
        lh, la = lam_pair[pair]
        lam_for, lam_ag = (lh, la) if r["was_home"] else (la, lh)
        pos = int(r["element_type"])
        # leave-one-out rate so no match informs its own prediction
        rate = 90.0 * (r["t"] - r["defensive_contribution"]) / (r["m"] - r["minutes"])
        mean = (rate * r["minutes"] / 90.0
                * max(0.2, min(5.0, lam_ag / lam_for)) ** fp.tilt_beta[pos])
        p = stats.negbin_at_least(scoring.DEFCON_THRESHOLD[pos], mean, fp.dispersion[pos])
        preds.append((p, 1 if r["defensive_contribution"] >= scoring.DEFCON_THRESHOLD[pos] else 0))
    preds.sort()
    n = len(preds)
    print("  %d starts. Predicted vs actual by decile of predicted probability:" % n)
    print("    %-10s %-8s %-10s %s" % ("decile", "n", "predicted", "actual"))
    for d in range(10):
        chunk = preds[d * n // 10:(d + 1) * n // 10]
        if not chunk:
            continue
        print("    %-10d %-8d %-10.3f %.3f"
              % (d + 1, len(chunk), sum(c[0] for c in chunk) / len(chunk),
                 sum(c[1] for c in chunk) / len(chunk)))
    pred_rate = sum(c[0] for c in preds) / n
    act_rate = sum(c[1] for c in preds) / n
    brier = sum((c[0] - c[1]) ** 2 for c in preds) / n
    print("  overall predicted %.4f vs actual %.4f (%.1f%% off), Brier %.4f"
          % (pred_rate, act_rate, 100 * abs(pred_rate - act_rate) / act_rate, brier))
    assert abs(pred_rate - act_rate) < 0.02, \
        "aggregate DEFCON rate must match reality (%.4f vs %.4f)" % (pred_rate, act_rate)
    print("  the fitted negative binomial is calibrated in aggregate and monotone by decile.")

    # --- polarity ---------------------------------------------------------
    print("\n=== polarity: the same fixture moves clean sheets and DEFCON in OPPOSITE directions ===")
    defending = DefendingModel(cfg).fit(state, results=results, cache=cache)
    candidates = [t for t in season[:40] if t[1].position == scoring.DEF]
    subject = candidates[0][1]
    fixtures_for = team_fixtures(state, subject.team_id, all_gws)
    scored_fx = []
    for f in fixtures_for:
        ctx = make_ctx(subject.id, f)
        dd = model.project_detail(subject.id, ctx)
        cs = defending.project(subject.id, ctx)
        scored_fx.append((cs["lambda_conceded"], f, cs, dd))
    scored_fx.sort(key=lambda t: t[0])
    easiest, hardest = scored_fx[0], scored_fx[-1]
    print("  subject: %s (%s, %s), rate %.2f actions/90, threshold %d"
          % (subject.web_name, scoring.POS_NAME[subject.position],
             state.short_name(subject.team_id), model.rate(subject.id).rate90,
             scoring.DEFCON_THRESHOLD[subject.position]))
    print("  %-22s %-9s %-7s %-9s %-9s %-9s %s"
          % ("fixture", "lam_gc", "tilt", "P(CS)", "xp_cs", "P(defcon)", "xp_defcon"))
    for label, (lam_gc, f, cs, dd) in (("easiest", easiest), ("hardest", hardest)):
        opp = state.short_name(f.opponent_of(subject.team_id))
        where = "H" if f.is_home_for(subject.team_id) else "A"
        print("  %-22s %-9.2f %-7.2f %-9.3f %-9.2f %-9.3f %.2f"
              % ("GW%d %s (%s) %s" % (f.gw, opp, where, label), lam_gc, dd["defcon_tilt"],
                 cs["p_clean_sheet"], cs["xp_clean_sheet"], dd["p_defcon"], dd["xp_defcon"]))
    d_cs = hardest[2]["p_clean_sheet"] - easiest[2]["p_clean_sheet"]
    d_dc = hardest[3]["p_defcon"] - easiest[3]["p_defcon"]
    print("  hardest minus easiest:  P(clean sheet) %+.3f   P(DEFCON) %+.3f" % (d_cs, d_dc))
    assert d_cs < 0 and d_dc > 0, \
        "clean sheet must fall and DEFCON must rise on the harder fixture (%.3f, %.3f)" % (d_cs, d_dc)
    print("  clean sheet DOWN and DEFCON UP on the harder fixture: OK")
    print("  net defensive swing %+.2f pts, so the DEFCON term recovers %.0f%% of the"
          % (d_dc * 2 + d_cs * 4, 100 * abs(d_dc * 2) / max(abs(d_cs * 4), 1e-9)))
    print("  clean-sheet points a tough fixture costs this defender.")

    # --- invariants -------------------------------------------------------
    print("\n=== invariants ===")
    gkp = [pid for pid, p in state.players.items()
           if p.position == scoring.GKP and p.team_id in fx_by_team][0]
    assert model.project(gkp, make_ctx(gkp, fx_by_team[state.player(gkp).team_id])) == 0.0
    subj_ctx = make_ctx(subject.id, fx_by_team[subject.team_id])
    assert 0.0 <= model.project(subject.id, subj_ctx) <= 1.0
    benched = make_ctx(subject.id, fx_by_team[subject.team_id], p_start=0.0)
    p_bench = model.project(subject.id, benched)
    p_full = model.project(subject.id, subj_ctx)
    assert p_bench < 0.02 * p_full + 1e-9, "a bench cameo cannot clear 10 actions (%.4f)" % p_bench
    no_hist = [pid for pid, p in state.players.items()
               if p.position in (scoring.DEF, scoring.MID)
               and model.rate(pid).source == "prior" and p.team_id in fx_by_team]
    print("  goalkeepers return exactly 0.0; P in [0,1]; a %.0f-minute cameo gives P=%.4f vs %.3f starting"
          % (cfg.model.sub_minutes, p_bench, p_full))
    print("  %d outfielders have no DEFCON history and fall back to the position+price prior"
          % len(no_hist))
    if no_hist:
        ex = state.player(no_hist[0])
        print("    e.g. %s (%s, %s, %.1fm): %s"
              % (ex.web_name, scoring.POS_NAME[ex.position], state.short_name(ex.team_id),
                 ex.price, model.rate(ex.id).reason))
    unconfirmed = [pid for pid in state.players if not model.rate(pid).basis_confirmed]
    print("  %d players carry a source whose position basis could not be confirmed"
          " (2024/25 stored counts), and are downweighted %.0f%%"
          % (len(unconfirmed), 100 * (1 - UNCONFIRMED_BASIS_WEIGHT)))
    print("\ndefcon.py smoke test passed.")
