"""Configuration. Defaults live here; `config.json` in the project root overrides."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where the fetched FPL data, the fitted-model reports and the projection cache
# live. Normally alongside the code, but a hosted deploy needs them on a mounted
# volume instead: on Render the repo is rebuilt from git on every deploy, so
# anything written next to the code is thrown away, and a cold start would
# refetch 587 player summaries and refit from scratch. Point GAFFER_DATA_DIR at
# the disk mount and the cache survives deploys and restarts.
_ENV_DATA_DIR = (os.environ.get("GAFFER_DATA_DIR") or "").strip()
DATA_DIR = os.path.abspath(_ENV_DATA_DIR) if _ENV_DATA_DIR else os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
# Reports are model fits, so they belong with the data they were fitted from.
REPORTS_DIR = os.path.join(DATA_DIR, "reports") if _ENV_DATA_DIR else os.path.join(PROJECT_ROOT, "reports")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

FPL_API_BASE = "https://fantasy.premierleague.com/api"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
SEASON = "2026-27"


@dataclass
class ModelConfig:
    # --- Blending weights for per-90 attacking rates ---
    # Prior season, current season to date, and recent form (last N gameweeks).
    weight_prior_season: float = 0.30
    weight_season_to_date: float = 0.30
    weight_recent_form: float = 0.40
    recent_form_gws: int = 6

    # Shrinkage: how many 90s of data before a player's own rate is trusted
    # over the position/price baseline.
    shrinkage_90s_attacking: float = 8.0
    shrinkage_90s_defcon: float = 6.0
    shrinkage_90s_minutes: float = 4.0

    # --- Team ratings ---
    # Exponential time decay for match results, in days (half-life).
    ratings_half_life_days: float = 180.0
    # How much weight the previous season's final rating keeps at GW1.
    ratings_prior_weight_gw1: float = 1.0
    # Matches of the new season needed before the prior is fully replaced.
    ratings_prior_decay_matches: float = 10.0
    # Promoted teams get a prior drawn from historical promoted-team performance.
    promoted_attack_prior: float = 0.78
    promoted_defence_prior: float = 1.22
    league_avg_goals_per_team: float = 1.45
    home_advantage: float = 1.13
    dixon_coles_rho: float = -0.06

    # Blend model lambdas with bookmaker-implied lambdas when odds are available.
    odds_blend_weight: float = 0.5

    # --- Minutes ---
    # Status codes map to a hard availability multiplier.
    status_multiplier: Dict[str, float] = field(
        default_factory=lambda: {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}
    )
    # Minutes a starter is assumed to play when they start.
    starter_minutes: float = 82.0
    sub_minutes: float = 20.0
    # Rotation penalty applied to players at clubs in European competition.
    rotation_risk_penalty: float = 0.0

    # --- Bonus ---
    bonus_model: str = "bps_sim"  # "bps_sim" or "historical_rate"
    bps_sim_iterations: int = 2000

    # --- Cards ---
    default_yellow_rate_per_90: float = 0.13

    # --- Projection horizon ---
    default_horizon: int = 6


@dataclass
class OptimizerConfig:
    horizon: int = 6
    decay: float = 0.84
    solver: str = "auto"  # "auto", "highs", "cbc"
    time_limit_seconds: int = 120
    mip_gap: float = 0.001
    # Value assigned to a free transfer carried into the next gameweek.
    ft_value: float = 1.6
    # Value of £0.1m of squad value, in points, used to break ties.
    value_per_tenth: float = 0.03
    # Bench weights: how much a bench player's projected points count towards
    # the objective, by bench order (autosub insurance).
    bench_weights: List[float] = field(default_factory=lambda: [0.12, 0.08, 0.05, 0.02])
    # Minimum expected gain over the horizon before a -4 hit is recommended.
    hit_threshold: float = 4.0
    max_hits_per_gw: int = 2
    # Force-include / force-exclude player ids.
    locked_in: List[int] = field(default_factory=list)
    locked_out: List[int] = field(default_factory=list)
    # Risk appetite: 0 = pure expected points, 1 = maximum differential chasing.
    differential_weight: float = 0.0
    # How much the captaincy slot values *ceiling* on top of mean expected
    # points. The armband doubles one player, so what it is really worth is the
    # right tail, not the average -- and the objective otherwise sees a 30.1 xP
    # midfielder and a 29.5 xP striker with a far bigger spread as very nearly
    # the same thing. The captain's contribution becomes xp + w * sd.
    #
    # 0.0 reproduces the old behaviour exactly and stays the default: this is a
    # deliberate risk preference, not a free accuracy win. Raising it buys
    # variance, which is what climbing overall rank needs and what protecting a
    # rank does not. ~0.3-0.5 is a sane range; 1.0 is aggressive.
    captain_ceiling_weight: float = 0.0
    # Ignore players below this projected minutes threshold entirely.
    min_xmins_to_consider: float = 10.0


@dataclass
class Config:
    season: str = SEASON
    entry_id: Optional[int] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    cache_ttl_seconds: int = 300
    odds_api_key: Optional[str] = None
    # The Odds API free tier allows 500 requests a month and the site rebuilds
    # hourly (720). One request covers every upcoming fixture, so refetching at
    # most every 6 hours is ~120 a month — inside the free tier with room for
    # manual builds before a deadline.
    odds_min_refresh_hours: float = 6.0
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        path = path or CONFIG_PATH
        if os.path.exists(path):
            with open(path, "r") as fh:
                raw = json.load(fh)
            cfg = cls.from_dict(raw)
        env_entry = os.environ.get("FPL_ENTRY_ID")
        if env_entry:
            cfg.entry_id = int(env_entry)
        env_odds = os.environ.get("ODDS_API_KEY")
        if env_odds:
            cfg.odds_api_key = env_odds
        return cfg

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        model = ModelConfig(**{**asdict(ModelConfig()), **raw.get("model", {})})
        opt = OptimizerConfig(**{**asdict(OptimizerConfig()), **raw.get("optimizer", {})})
        top = {
            f.name: raw[f.name]
            for f in fields(cls)
            if f.name in raw and f.name not in ("model", "optimizer")
        }
        return cls(model=model, optimizer=opt, **top)

    def save(self, path: Optional[str] = None) -> None:
        path = path or CONFIG_PATH
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
