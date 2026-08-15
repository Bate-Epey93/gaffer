"""The FastAPI backend behind the dashboard.

This process exists for one hard reason: **fantasy.premierleague.com sends no
CORS headers**, so a browser page cannot read it directly. Every byte the
dashboard needs is fetched here, server side, and re-served from localhost.

Two other things shape the design.

* **A request must never fit a model.** Loading the game state, fitting the six
  sub-models and projecting 587 players over the horizon costs a few seconds;
  doing that per request would make the dashboard unusable. It happens once at
  startup into a module-level ``Snapshot``, and afterwards only when the
  snapshot ages past its TTL or ``POST /api/refresh`` says so. A stale snapshot
  is served *while* the replacement builds in the background, so no request
  waits on a refit.
* **The engine is not thread-safe.** ``XPEngine`` memoises fixture bundles as
  it goes, so any call that can extend those memos (``explain``,
  ``gw_projection``, an out-of-horizon ``project``) is taken under the
  snapshot's own lock. Reading an already-built ``ProjectionSet`` needs no lock
  at all, which is what the hot endpoints do.
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gaffer.core import scoring
from gaffer.core.config import CACHE_DIR, PROJECT_ROOT, Config
from gaffer.core.types import (
    CaptainOption,
    GWDecision,
    Plan,
    PlayerFixtureProjection,
    PlayerGWProjection,
    ProjectionSet,
    SquadPick,
    SquadState,
)
from gaffer.data.cache import Cache
from gaffer.data.fpl_api import FPLClient, FPLError, FPLNotFound
from gaffer.data.loaders import GameState, load_game_state
from gaffer.model.xp import MODEL_VERSION, XPEngine, projection_set_to_dict

log = logging.getLogger(__name__)

WEB_DIR = os.path.join(PROJECT_ROOT, "gaffer", "web")
DEFAULT_PORT = 8770

# How long a fitted snapshot is served before a background rebuild is started.
# The FPL API is Fastly-cached at max-age=300 and prices move once a day, so
# anything under five minutes only burns CPU; fifteen keeps news and price
# changes fresh without ever blocking a request.
STATE_TTL_SECONDS = int(os.environ.get("GAFFER_STATE_TTL", "900"))

# Localhost only. The dashboard is served from this same origin, so CORS is
# really just for the case where someone runs the static files from another
# local port while developing.
LOCALHOST_ORIGIN_RE = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

_COMPONENT_KEYS: Tuple[str, ...] = tuple(PlayerFixtureProjection(0, 0, 0, 0, True).components().keys())

SORT_KEYS = (
    "xp", "horizon_xp", "value", "price", "ownership", "xmins", "p_haul",
    "sd", "form", "p_defcon", "name", "team",
)


# ---------------------------------------------------------------------------
# Peer modules built by other parts of the project
# ---------------------------------------------------------------------------
# The optimizer and backtest packages are separate modules; resolving them
# lazily (and by attribute, not by file path) means this server starts and
# serves the whole data/model half of the API even while they are still being
# built, and reports exactly what is missing instead of dying at import time.

_PEERS: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    "pick_initial_squad": (("gaffer.optimize.squad",), "squad optimizer", "gaffer.optimize"),
    "pick_lineup": (("gaffer.optimize.squad",), "squad optimizer", "gaffer.optimize"),
    "squad_problems": (("gaffer.optimize.squad",), "squad optimizer", "gaffer.optimize"),
    "plan": (("gaffer.optimize.planner",), "multi-gameweek planner", "gaffer.optimize"),
    "recommend_chips": (("gaffer.optimize.chips",), "chip advisor", "gaffer.optimize"),
    "detect_double_blank_gws": (("gaffer.optimize.chips",), "chip advisor", "gaffer.optimize"),
    "captain_options": (("gaffer.optimize.strategy",), "captaincy / strategy", "gaffer.optimize"),
    "effective_ownership": (("gaffer.optimize.strategy",), "captaincy / strategy", "gaffer.optimize"),
    "rank_risk_report": (("gaffer.optimize.strategy",), "captaincy / strategy", "gaffer.optimize"),
    "backtest_season": (
        ("gaffer.backtest", "gaffer.backtest.runner", "gaffer.backtest.season"),
        "backtester", "gaffer.backtest",
    ),
    "evaluate_projections": (
        ("gaffer.backtest", "gaffer.backtest.metrics"), "backtester", "gaffer.backtest",
    ),
    # Used by the CLI, which writes the backtest report through the backtester's
    # own serialiser rather than inventing a second format.
    "write_report": (
        ("gaffer.backtest", "gaffer.backtest.runner"), "backtester", "gaffer.backtest",
    ),
}

# Only successful lookups are memoised, so a module that lands after the server
# started is picked up on the next call rather than being negative-cached.
_RESOLVED: Dict[str, Any] = {}


def _scan_package(package_name: str, attr: str) -> Tuple[Optional[Any], List[str]]:
    """Look for ``attr`` in every submodule of a package.

    The spec fixes the function names but not which file inside
    ``gaffer/optimize`` or ``gaffer/backtest`` defines them, so after the named
    candidates miss we search the package rather than 503 on a filename.
    """
    import pkgutil

    problems: List[str] = []
    try:
        package = importlib.import_module(package_name)
    except Exception as exc:
        return None, ["%s: %s: %s" % (package_name, type(exc).__name__, exc)]
    fn = getattr(package, attr, None)
    if fn is not None:
        return fn, problems
    for info in pkgutil.iter_modules(getattr(package, "__path__", [])):
        full = "%s.%s" % (package_name, info.name)
        try:
            module = importlib.import_module(full)
        except Exception as exc:
            problems.append("%s: %s: %s" % (full, type(exc).__name__, exc))
            continue
        fn = getattr(module, attr, None)
        if fn is not None:
            return fn, problems
    return None, problems


def resolve(name: str) -> Any:
    """Return a peer function, or raise a 503 naming exactly what is missing."""
    cached = _RESOLVED.get(name)
    if cached is not None:
        return cached
    candidates, label, package = _PEERS[name]
    problems: List[str] = []
    for mod_name in candidates:
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:  # not built yet, or broken on import
            problems.append("%s: %s: %s" % (mod_name, type(exc).__name__, exc))
            continue
        fn = getattr(module, name, None)
        if fn is not None:
            _RESOLVED[name] = fn
            return fn
        problems.append("%s: no attribute %s" % (mod_name, name))
    fn, more = _scan_package(package, name)
    problems.extend(more)
    if fn is not None:
        _RESOLVED[name] = fn
        return fn
    raise HTTPException(
        status_code=503,
        detail="%s is unavailable: %s() not found (%s)" % (label, name, "; ".join(problems) or "no candidates"),
    )


@contextmanager
def peer_errors(label: str) -> Any:
    """Turn an optimizer/backtester failure into 422 rather than a bare 500.

    An infeasible MILP or a rejected squad is a statement about the *request*,
    not a server fault, and the message the solver produced is the most useful
    thing we can hand back.
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        if (type(exc).__module__ or "").startswith("gaffer."):
            raise HTTPException(
                status_code=422,
                detail="%s failed: %s: %s" % (label, type(exc).__name__, exc),
            )
        raise


def peer_status() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in sorted(_PEERS):
        try:
            resolve(name)
            out[name] = "ok"
        except HTTPException as exc:
            out[name] = "missing"
            log.debug("peer %s unavailable: %s", name, exc.detail)
    return out


# ---------------------------------------------------------------------------
# Snapshot + store
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """One fully fitted view of the world. Immutable once published."""

    state: GameState
    engine: XPEngine
    projections: ProjectionSet
    gws: List[int]
    built_at: float
    built_iso: str
    seconds: Dict[str, float]
    forced: bool
    haul: Dict[Tuple[int, int], float]
    # Re-entrant: helpers that take the lock are called from endpoints that
    # already hold it (an explanation walks the same fixtures it forecasts).
    lock: Any = field(default_factory=threading.RLock)
    # Projection sets for ranges outside the default horizon, computed on demand.
    extra: Dict[Tuple[int, int], ProjectionSet] = field(default_factory=dict)
    # The recommended fifteen, memoised: it is a MILP solve (~6s) and three
    # endpoints fall back to it when no entry id is known.
    fallback: Optional[Tuple[SquadState, str]] = None
    # Chip recommendations per squad, memoised for the same reason.
    chips: Dict[Tuple[int, ...], Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.built_at)


class Store:
    """Holds the current snapshot and rebuilds it off the request path."""

    def __init__(
        self,
        config: Optional[Config] = None,
        horizon: Optional[int] = None,
        ttl: int = STATE_TTL_SECONDS,
    ) -> None:
        self.config = config or Config.load()
        self.horizon = int(horizon or self.config.model.default_horizon)
        self.ttl = int(ttl)
        self.snapshot: Optional[Snapshot] = None
        self.error: Optional[str] = None
        self.error_at: Optional[str] = None
        self.refreshing = False
        self.builds = 0
        self._build_lock = threading.Lock()
        self._flag_lock = threading.Lock()

    # -- building -----------------------------------------------------------
    def _build(self, force: bool = False) -> Snapshot:
        t0 = time.time()
        state = load_game_state(self.config, force=force, progress=False)
        t_load = time.time() - t0

        t1 = time.time()
        engine = XPEngine(self.config)
        engine.fit(state)
        t_fit = time.time() - t1

        last = min(scoring.TOTAL_GWS, state.current_gw + self.horizon - 1)
        gws = list(range(state.current_gw, last + 1)) or [state.current_gw]
        t2 = time.time()
        projections = engine.project(state, gws)
        t_project = time.time() - t2

        try:
            engine.save_projections(projections)
        except OSError as exc:  # a read-only cache dir must not break serving
            log.warning("could not cache projections: %s", exc)

        snapshot = Snapshot(
            state=state,
            engine=engine,
            projections=projections,
            gws=gws,
            built_at=time.time(),
            built_iso=datetime.now(timezone.utc).isoformat(),
            seconds={
                "load": round(t_load, 3),
                "fit": round(t_fit, 3),
                "project": round(t_project, 3),
                "total": round(time.time() - t0, 3),
            },
            forced=bool(force),
            haul=engine.haul_table(),
        )
        self.snapshot = snapshot
        self.error = None
        self.error_at = None
        self.builds += 1
        log.info(
            "snapshot built in %.2fs (load %.2f, fit %.2f, project %.2f) GW%d-%d",
            snapshot.seconds["total"], t_load, t_fit, t_project, gws[0], gws[-1],
        )
        return snapshot

    def refresh(self, force: bool = False) -> Snapshot:
        """Rebuild now, blocking. ``force`` re-fetches from the FPL API."""
        with self._build_lock:
            return self._build(force=force)

    def _refresh_background(self) -> None:
        try:
            with self._build_lock:
                self._build(force=False)
        except Exception as exc:  # keep serving the stale snapshot
            self.error = "%s: %s" % (type(exc).__name__, exc)
            self.error_at = datetime.now(timezone.utc).isoformat()
            log.warning("background refresh failed: %s", self.error)
        finally:
            with self._flag_lock:
                self.refreshing = False

    def get(self) -> Snapshot:
        """The current snapshot; builds synchronously only if there is none."""
        snapshot = self.snapshot
        if snapshot is not None:
            if snapshot.age > self.ttl:
                self._kick()
            return snapshot
        with self._build_lock:
            if self.snapshot is not None:
                return self.snapshot
            try:
                return self._build(force=False)
            except Exception as exc:
                self.error = "%s: %s" % (type(exc).__name__, exc)
                self.error_at = datetime.now(timezone.utc).isoformat()
                raise HTTPException(
                    status_code=503,
                    detail="model state is unavailable: %s. The FPL API may be "
                           "unreachable; try POST /api/refresh." % self.error,
                )

    def _kick(self) -> None:
        with self._flag_lock:
            if self.refreshing:
                return
            self.refreshing = True
        threading.Thread(target=self._refresh_background, name="gaffer-refresh", daemon=True).start()

    # -- reporting ----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        snapshot = self.snapshot
        return {
            "loaded": snapshot is not None,
            "built_at": snapshot.built_iso if snapshot else None,
            "age_seconds": round(snapshot.age, 1) if snapshot else None,
            "ttl_seconds": self.ttl,
            "stale": bool(snapshot is not None and snapshot.age > self.ttl),
            "refresh_in_progress": self.refreshing,
            "builds": self.builds,
            "build_seconds": snapshot.seconds if snapshot else None,
            "last_error": self.error,
            "last_error_at": self.error_at,
        }


STORE = Store()


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------


def cache_freshness() -> Dict[str, Any]:
    """Age of the on-disk API cache, which is the real data-freshness answer."""
    cache = Cache(default_ttl=STORE.config.cache_ttl_seconds)
    out: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    for key in ("bootstrap", "fixtures", "event_status"):
        age = cache.age_seconds(key)
        out["%s_age_seconds" % key] = round(age, 1) if age is not None else None
    # 587 element summaries: stat() them rather than parse them.
    ages: List[float] = []
    now = time.time()
    try:
        for name in os.listdir(CACHE_DIR):
            if name.startswith("element_summary_") and name.endswith(".json"):
                ages.append(now - os.path.getmtime(os.path.join(CACHE_DIR, name)))
    except OSError:
        pass
    out["element_summaries"] = {
        "count": len(ages),
        "newest_age_seconds": round(min(ages), 1) if ages else None,
        "oldest_age_seconds": round(max(ages), 1) if ages else None,
    }
    return out


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """JSON-safe: NaN and infinity are not valid JSON and break the browser."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _clean(asdict(value))
    return value


def _team_block(state: GameState, team_id: int) -> Dict[str, Any]:
    team = state.teams.get(team_id)
    return {
        "id": team_id,
        "short": team.short_name if team else "?",
        "name": team.name if team else "unknown",
    }


def fixture_projection_dict(state: GameState, fp: PlayerFixtureProjection) -> Dict[str, Any]:
    row = {name: getattr(fp, name) for name in _FIXTURE_FIELDS}
    row["opponent"] = state.short_name(fp.opponent_id)
    row["venue"] = "H" if fp.is_home else "A"
    row["components"] = fp.components()
    return _clean(row)


_FIXTURE_FIELDS: Tuple[str, ...] = tuple(
    f for f in PlayerFixtureProjection.__dataclass_fields__  # type: ignore[attr-defined]
)


def _aggregate(gwp: Optional[PlayerGWProjection]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(components, rates) for a gameweek, summing over a double's fixtures.

    Rates that are counts (the lambdas, expected BPS) sum, because a double
    gameweek really does give a player two shots at them. Rates that are
    probabilities of a per-match event are averaged: a 0.35 clean-sheet chance
    in each of two fixtures is not a 0.70 chance of anything.
    """
    components = {k: 0.0 for k in _COMPONENT_KEYS}
    rates = {
        "lambda_goals": 0.0, "lambda_assists": 0.0, "lambda_conceded": 0.0,
        "lambda_saves": 0.0, "exp_bps": 0.0, "p_clean_sheet": 0.0, "p_defcon": 0.0,
        "p_start": 0.0, "p_appear": 0.0, "p_60": 0.0, "xmins": 0.0,
    }
    if gwp is None or not gwp.fixtures:
        return components, rates
    n = float(len(gwp.fixtures))
    for fp in gwp.fixtures:
        for key, value in fp.components().items():
            components[key] += value
        for key in ("lambda_goals", "lambda_assists", "lambda_conceded",
                    "lambda_saves", "exp_bps", "xmins"):
            rates[key] += getattr(fp, key)
        for key in ("p_clean_sheet", "p_defcon", "p_start", "p_appear", "p_60"):
            rates[key] += getattr(fp, key) / n
    return components, rates


def player_row(
    snapshot: Snapshot,
    player_id: int,
    gw: int,
    horizon_gws: Sequence[int],
    with_fixtures: bool = True,
) -> Dict[str, Any]:
    """One player's payload: totals *and* the full component breakdown.

    The dashboard has to be able to audit every number in one click, so the
    breakdown travels with the total on every endpoint, never as a follow-up
    request.
    """
    state = snapshot.state
    player = state.players[player_id]
    per_gw = snapshot.projections.projections.get(player_id, {})
    gwp = per_gw.get(gw)
    components, rates = _aggregate(gwp)

    engine = snapshot.engine
    try:
        xg90, xa90 = engine.attacking.rates(player_id)
    except Exception:  # a player the attacking model never saw
        xg90, xa90 = 0.0, 0.0
    defcon_rate90 = 0.0
    if player.position != scoring.GKP:
        try:
            defcon_rate90 = engine.defcon.rate(player_id).rate90
        except Exception:
            defcon_rate90 = 0.0

    horizon_xp = sum(per_gw[g].xp for g in horizon_gws if g in per_gw)
    xp = gwp.xp if gwp else 0.0
    row: Dict[str, Any] = {
        "id": player_id,
        "name": player.web_name,
        "full_name": ("%s %s" % (player.first_name, player.second_name)).strip(),
        "team_id": player.team_id,
        "team": state.short_name(player.team_id),
        "team_name": state.teams[player.team_id].name if player.team_id in state.teams else "",
        "position": player.position,
        "pos": scoring.POS_NAME[player.position],
        "price": player.price,
        "now_cost": player.now_cost,
        "status": player.status,
        "news": player.news,
        "selected_by_percent": player.selected_by_percent,
        "form": player.form,
        "ep_next": player.ep_next,
        "penalties_order": player.penalties_order,
        "gw": gw,
        "xp": xp,
        "sd": gwp.sd if gwp else 0.0,
        "p_haul": snapshot.haul.get((player_id, gw), 0.0),
        "n_fixtures": gwp.n_fixtures if gwp else 0,
        "is_blank": bool(gwp is None or gwp.is_blank),
        "horizon_xp": horizon_xp,
        "horizon_gws": list(horizon_gws),
        "value": horizon_xp / max(player.price, 0.1),
        "gw_xp": {str(g): (per_gw[g].xp if g in per_gw else 0.0) for g in horizon_gws},
        "minutes": {
            "p_start": rates["p_start"],
            "p_appear": rates["p_appear"],
            "p_60": rates["p_60"],
            "xmins": rates["xmins"],
        },
        "rates": {
            "xg90": xg90,
            "xa90": xa90,
            "defcon_rate90": defcon_rate90,
            "lambda_goals": rates["lambda_goals"],
            "lambda_assists": rates["lambda_assists"],
            "lambda_conceded": rates["lambda_conceded"],
            "lambda_saves": rates["lambda_saves"],
            "p_clean_sheet": rates["p_clean_sheet"],
            "p_defcon": rates["p_defcon"],
            "exp_bps": rates["exp_bps"],
        },
        "components": components,
    }
    # Flat aliases. The dashboard reads these names directly off a player row
    # (see the contract block at the top of gaffer/web/app.js); duplicating ten
    # floats is cheaper than making the client walk two nested objects.
    row.update({
        "player_id": player_id,
        "web_name": player.web_name,
        "team_short": row["team"],
        "element_type": player.position,
        "chance_of_playing_next_round": player.chance_of_playing_next_round,
        "total_points": player.total_points,
        "p_start": rates["p_start"],
        "p_appear": rates["p_appear"],
        "p_60": rates["p_60"],
        "xmins": rates["xmins"],
        "p_defcon": rates["p_defcon"],
        "xg90": xg90,
        "xa90": xa90,
    })
    if with_fixtures:
        row["fixtures"] = [
            fixture_projection_dict(state, fp) for fp in (gwp.fixtures if gwp else [])
        ]
    return _clean(row)


def player_gws(snapshot: Snapshot, player_id: int, gws: Sequence[int]) -> List[Dict[str, Any]]:
    """Per-gameweek rows with the fixtures, in the shape the dashboard merges."""
    state = snapshot.state
    per_gw = snapshot.projections.projections.get(player_id, {})
    out = []
    for g in gws:
        gwp = per_gw.get(g)
        components, rates = _aggregate(gwp)
        out.append(_clean({
            "gw": g,
            "xp": gwp.xp if gwp else 0.0,
            "sd": gwp.sd if gwp else 0.0,
            "p_haul": snapshot.haul.get((player_id, g), 0.0),
            "is_blank": bool(gwp is None or gwp.is_blank),
            "n_fixtures": gwp.n_fixtures if gwp else 0,
            "components": components,
            "rates": rates,
            "fixtures": [fixture_projection_dict(state, fp) for fp in (gwp.fixtures if gwp else [])],
        }))
    return out


def fixture_row(snapshot: Snapshot, fixture: Any) -> Dict[str, Any]:
    state = snapshot.state
    with snapshot.lock:
        forecast = snapshot.engine.forecast(fixture)
    return _clean({
        "id": fixture.id,
        "code": fixture.code,
        "gw": fixture.gw,
        "kickoff_time": fixture.kickoff_time,
        "finished": fixture.finished,
        "score": (
            None if fixture.team_h_score is None
            else {"home": fixture.team_h_score, "away": fixture.team_a_score}
        ),
        "home": dict(
            _team_block(state, fixture.team_h),
            **{
                "lambda": forecast.lambda_h,
                "p_clean_sheet": forecast.p_cs_h,
                "difficulty": fixture.team_h_difficulty,
                "p_win": forecast.p_home_win,
            }
        ),
        "away": dict(
            _team_block(state, fixture.team_a),
            **{
                "lambda": forecast.lambda_a,
                "p_clean_sheet": forecast.p_cs_a,
                "difficulty": fixture.team_a_difficulty,
                "p_win": forecast.p_away_win,
            }
        ),
        "p_draw": forecast.p_draw,
        "total_goals": forecast.lambda_h + forecast.lambda_a,
        "source": forecast.source,
        # Flat MatchForecast aliases: the dashboard's fixture ticker reads these.
        "team_h": fixture.team_h,
        "team_a": fixture.team_a,
        "team_h_difficulty": fixture.team_h_difficulty,
        "team_a_difficulty": fixture.team_a_difficulty,
        "lambda_h": forecast.lambda_h,
        "lambda_a": forecast.lambda_a,
        "p_cs_h": forecast.p_cs_h,
        "p_cs_a": forecast.p_cs_a,
        "p_home_win": forecast.p_home_win,
        "p_away_win": forecast.p_away_win,
    })


def _named(state: GameState, player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if player_id is None:
        return None
    player = state.players.get(int(player_id))
    if player is None:
        return {"id": int(player_id)}
    return {
        "id": player.id,
        "name": player.web_name,
        "pos": scoring.POS_NAME[player.position],
        "team": state.short_name(player.team_id),
        "price": player.price,
    }


def decision_dict(state: GameState, decision: GWDecision) -> Dict[str, Any]:
    out = _clean(asdict(decision))
    out["transfers"] = [
        dict(t, out_player=_named(state, t["out_id"]), in_player=_named(state, t["in_id"]))
        for t in out.get("transfers", [])
    ]
    out["lineup_players"] = [_named(state, p) for p in decision.lineup]
    out["bench_players"] = [_named(state, p) for p in decision.bench]
    out["squad_players"] = [_named(state, p) for p in decision.squad]
    out["captain_player"] = _named(state, decision.captain)
    out["vice_captain_player"] = _named(state, decision.vice_captain)
    return out


def plan_dict(state: GameState, plan: Plan) -> Dict[str, Any]:
    return _clean({
        "first_gw": plan.first_gw,
        "horizon": plan.horizon,
        "objective": plan.objective,
        "decay": plan.decay,
        "solver_status": plan.solver_status,
        "solve_seconds": plan.solve_seconds,
        "total_expected_points": plan.total_expected_points,
        "decisions": [decision_dict(state, d) for d in plan.decisions],
    })


def captain_dict(state: GameState, option: CaptainOption) -> Dict[str, Any]:
    out = _clean(asdict(option))
    out["player"] = _named(state, option.player_id)
    return out


def squad_dict(state: GameState, squad: SquadState) -> Dict[str, Any]:
    out = _clean(asdict(squad))
    out["picks"] = [
        dict(p, player=_named(state, p["player_id"])) for p in out.get("picks", [])
    ]
    out["squad_value"] = squad.squad_value / 10.0
    out["bank_m"] = squad.bank / 10.0
    out["budget_m"] = squad.budget / 10.0
    return out


# ---------------------------------------------------------------------------
# Squad state from the FPL API
# ---------------------------------------------------------------------------


def load_squad_state(entry_id: int, gw: Optional[int] = None) -> Tuple[SquadState, Dict[str, Any]]:
    """The manager's real squad, from the public entry endpoints.

    The public picks endpoint carries no purchase price, so selling price is
    reported as today's price and flagged in ``meta``: a squad loaded this way
    slightly understates the bank for anyone sitting on price rises. Only the
    authenticated ``my-team`` endpoint has the true figures.
    """
    snapshot = STORE.get()
    state = snapshot.state
    gw = int(gw or state.current_gw)
    client = FPLClient(STORE.config, Cache(default_ttl=STORE.config.cache_ttl_seconds))

    try:
        entry = client.entry(entry_id)
    except FPLNotFound:
        raise HTTPException(status_code=404, detail="FPL entry %d does not exist" % entry_id)
    except FPLError as exc:
        raise HTTPException(status_code=502, detail="FPL API error for entry %d: %s" % (entry_id, exc))

    meta: Dict[str, Any] = {
        "entry_id": entry_id,
        "name": entry.get("name"),
        "player_name": ("%s %s" % (entry.get("player_first_name") or "",
                                   entry.get("player_last_name") or "")).strip(),
        "gw": gw,
        "selling_price_is_current_price": True,
        "warnings": [],
    }

    bank = 0
    free_transfers = 1
    chips_used: List[str] = []
    total_points = int(entry.get("summary_overall_points") or 0)
    overall_rank = entry.get("summary_overall_rank")
    try:
        history = client.entry_history(entry_id)
        for row in history.get("current") or []:
            if int(row["event"]) == gw - 1:
                bank = int(row.get("bank") or 0)
        chips_used = [c.get("name") for c in (history.get("chips") or []) if c.get("name")]
    except FPLError as exc:
        meta["warnings"].append("entry history unavailable (%s); bank assumed 0" % exc)

    picks_payload: Dict[str, Any] = {}
    picks: List[SquadPick] = []
    source_gw = gw
    for candidate in (gw, gw - 1):
        if candidate < 1:
            continue
        try:
            picks_payload = client.entry_picks(entry_id, candidate)
            source_gw = candidate
            break
        except FPLNotFound:
            continue
        except FPLError as exc:
            raise HTTPException(status_code=502, detail="FPL API error: %s" % exc)
    if not picks_payload:
        raise HTTPException(
            status_code=404,
            detail="no picks for entry %d at GW%d — squads are only published "
                   "after the GW1 deadline (%s)"
                   % (entry_id, gw, state.deadline(1) or "unknown"),
        )

    if source_gw != gw:
        meta["warnings"].append("no GW%d picks yet; using GW%d" % (gw, source_gw))
    meta["picks_gw"] = source_gw
    entry_history = picks_payload.get("entry_history") or {}
    if entry_history:
        bank = int(entry_history.get("bank") or bank)
        total_points = int(entry_history.get("total_points") or total_points)

    for raw in picks_payload.get("picks") or []:
        pid = int(raw["element"])
        player = state.players.get(pid)
        price = player.now_cost if player else 0
        picks.append(
            SquadPick(
                player_id=pid,
                purchase_price=price,
                selling_price=price,
                multiplier=int(raw.get("multiplier") or 0),
                position_in_squad=int(raw.get("position") or 0),
            )
        )
    chip = picks_payload.get("active_chip")
    if chip and chip not in chips_used:
        chips_used.append(chip)

    squad = SquadState(
        gw=gw,
        picks=picks,
        bank=bank,
        free_transfers=free_transfers,
        chips_used=chips_used,
        entry_id=entry_id,
        total_points=total_points,
        overall_rank=int(overall_rank) if overall_rank else None,
    )
    return squad, meta


def _fallback_squad(snapshot: Snapshot) -> Tuple[SquadState, str]:
    """A squad to reason about when no entry id was supplied.

    Falls back to the optimizer's own recommended fifteen, which is the right
    default for a pre-season user who has not made a team yet, and to an empty
    squad if the optimizer is not available.
    """
    if snapshot.fallback is not None:
        return snapshot.fallback
    state = snapshot.state
    try:
        pick = resolve("pick_initial_squad")
    except HTTPException:
        return SquadState(gw=state.current_gw), "empty"
    with snapshot.lock:
        if snapshot.fallback is not None:
            return snapshot.fallback
        with peer_errors("the squad optimizer"):
            decision = pick(snapshot.projections, state, STORE.config, snapshot.gws)
        picks = []
        for i, pid in enumerate(decision.squad, start=1):
            price = state.players[pid].now_cost if pid in state.players else 0
            picks.append(SquadPick(player_id=pid, purchase_price=price, selling_price=price,
                                   position_in_squad=i))
        squad = SquadState(gw=decision.gw, picks=picks, bank=decision.bank_after,
                           free_transfers=decision.free_transfers_after)
        snapshot.fallback = (squad, "recommended")
    return snapshot.fallback


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """Body of POST /api/optimize."""

    horizon: Optional[int] = Field(default=None, ge=1, le=38)
    chips: Optional[List[str]] = None
    locks: Optional[Dict[str, List[int]]] = None
    locked_in: Optional[List[int]] = None
    locked_out: Optional[List[int]] = None
    entry_id: Optional[int] = None
    squad: Optional[List[int]] = None
    budget: Optional[float] = None
    bank: Optional[float] = None
    free_transfers: Optional[int] = None
    decay: Optional[float] = None

    def resolved_locks(self) -> Tuple[List[int], List[int]]:
        locks = self.locks or {}
        lin = list(self.locked_in or locks.get("in") or locks.get("locked_in") or [])
        lout = list(self.locked_out or locks.get("out") or locks.get("locked_out") or [])
        return [int(p) for p in lin], [int(p) for p in lout]


class RefreshRequest(BaseModel):
    force: bool = False
    horizon: Optional[int] = Field(default=None, ge=1, le=38)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build once, off the request path. A failure here is logged and left for
    # /api/health to report rather than killing the process: the dashboard
    # should be able to say "the FPL API is down", not fail to start.
    def build() -> None:
        try:
            STORE.refresh(force=False)
        except Exception as exc:
            STORE.error = "%s: %s" % (type(exc).__name__, exc)
            STORE.error_at = datetime.now(timezone.utc).isoformat()
            log.error("startup build failed: %s", STORE.error)

    threading.Thread(target=build, name="gaffer-startup", daemon=True).start()
    yield


def create_app(config: Optional[Config] = None, horizon: Optional[int] = None) -> FastAPI:
    global STORE
    if config is not None or horizon is not None:
        STORE = Store(config=config, horizon=horizon)

    app = FastAPI(
        title="gaffer",
        description="FPL 2026/27 expected-points engine. The browser talks to "
                    "this, never to fantasy.premierleague.com (which sends no "
                    "CORS headers).",
        version=MODEL_VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCALHOST_ORIGIN_RE,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # -- errors: always JSON with a usable message, never a stack trace ------
    # Registered against Starlette's class, not FastAPI's subclass, so that a
    # 404 raised by the static-file mount comes back in the same shape as ours.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _ERROR_NAMES.get(exc.status_code, "error"),
                     "detail": exc.detail, "path": request.url.path},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request",
                     "detail": [
                         {"field": ".".join(str(p) for p in e.get("loc", [])),
                          "message": e.get("msg", "")}
                         for e in exc.errors()
                     ],
                     "path": request.url.path},
        )

    @app.exception_handler(FPLError)
    async def _fpl_error(request: Request, exc: FPLError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_error",
                     "detail": "the FPL API did not answer: %s" % exc,
                     "path": request.url.path},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error",
                     "detail": "%s: %s" % (type(exc).__name__, exc),
                     "path": request.url.path},
        )

    # -- health -------------------------------------------------------------
    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        snapshot = STORE.snapshot
        status = "ok"
        if snapshot is None:
            status = "error" if STORE.error else "loading"
        elif STORE.error or snapshot.age > STORE.ttl:
            status = "degraded"
        out: Dict[str, Any] = {
            "status": status,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "season": STORE.config.season,
            "model": {
                "fitted": bool(snapshot is not None and snapshot.engine.fitted),
                "version": MODEL_VERSION,
                "horizon": STORE.horizon,
                "fit_seconds": snapshot.engine.fit_seconds if snapshot else {},
                "project_seconds": snapshot.engine.timings if snapshot else {},
                "warnings": snapshot.engine.warnings if snapshot else [],
            },
            "snapshot": STORE.status(),
            "data": cache_freshness(),
            "modules": peer_status(),
            "web_dir": {"path": WEB_DIR,
                        "exists": os.path.isdir(WEB_DIR),
                        "files": _web_files()},
        }
        if snapshot is not None:
            state = snapshot.state
            out["counts"] = {
                "teams": len(state.teams),
                "players": len(state.players),
                "fixtures": len(state.fixtures),
                "projected_players": len(snapshot.projections.projections),
            }
            out["gameweek"] = {
                "current": state.current_gw,
                "deadline": state.deadline(state.current_gw),
                "finished": state.finished_gws,
                "elements_are_prior_season": state.elements_are_prior_season,
            }
            out["projections"] = {
                "first_gw": snapshot.projections.first_gw,
                "last_gw": snapshot.projections.last_gw,
                "generated_at": snapshot.projections.generated_at,
                "model_version": snapshot.projections.model_version,
            }
            out["data"]["data_warnings"] = state.data_warnings
        return _clean(out)

    # -- state --------------------------------------------------------------
    @app.get("/api/state")
    def get_state() -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        gw = state.current_gw
        deadline = state.deadline(gw)
        seconds_to_deadline = None
        if deadline:
            try:
                dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                seconds_to_deadline = (dt - datetime.now(timezone.utc)).total_seconds()
            except ValueError:
                seconds_to_deadline = None
        return _clean({
            "season": state.season,
            "current_gw": gw,
            "gw": gw,
            "deadline": deadline,
            "deadline_time": deadline,
            "seconds_to_deadline": seconds_to_deadline,
            "finished_gws": state.finished_gws,
            "gws_remaining": state.gws_remaining(),
            "elements_are_prior_season": state.elements_are_prior_season,
            "counts": {
                "teams": len(state.teams),
                "players": len(state.players),
                "fixtures": len(state.fixtures),
                "unscheduled_fixtures": len([f for f in state.fixtures if f.gw is None]),
            },
            # Flat aliases for the dashboard's state contract.
            "n_teams": len(state.teams),
            "n_players": len(state.players),
            "n_fixtures": len(state.fixtures),
            "first_gw": snapshot.gws[0],
            "last_gw": snapshot.gws[-1],
            "fitted": bool(snapshot.engine.fitted),
            "entry_id": STORE.config.entry_id,
            "teams": [
                {"id": t.id, "short": t.short_name, "short_name": t.short_name,
                 "name": t.name, "code": t.code, "promoted": t.promoted}
                for t in sorted(state.teams.values(), key=lambda t: t.id)
            ],
            "promoted_team_ids": state.promoted_team_ids,
            "projection_gws": snapshot.gws,
            "chip_windows": {k: [list(v[0]), list(v[1])] for k, v in scoring.CHIP_WINDOWS.items()},
            "first_half_last_gw": scoring.FIRST_HALF_LAST_GW,
            "first_half_deadline": state.deadline(scoring.FIRST_HALF_LAST_GW),
            "budget": scoring.BUDGET_TENTHS / 10.0,
            "data_warnings": state.data_warnings,
            "model_warnings": snapshot.engine.warnings,
            "generated_at": snapshot.built_iso,
        })

    # -- players ------------------------------------------------------------
    @app.get("/api/players")
    def get_players(
        gw: Optional[int] = Query(None, ge=1, le=38, description="gameweek; defaults to the current one"),
        pos: Optional[str] = Query(None, description="GKP/DEF/MID/FWD or 1-4"),
        max_cost: Optional[float] = Query(None, gt=0, description="price ceiling in £m"),
        min_cost: Optional[float] = Query(None, gt=0),
        sort: str = Query("xp", description="one of %s" % ", ".join(SORT_KEYS)),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        # No limit by default: this is a single-user local API and a dashboard
        # that silently sees 100 of 587 players is worse than a 3MB response.
        limit: Optional[int] = Query(None, ge=1, le=2000, description="default: every match"),
        offset: int = Query(0, ge=0),
        team: Optional[str] = Query(None, description="team short name or id"),
        search: Optional[str] = Query(None, description="substring of the player name"),
        min_xmins: float = Query(0.0, ge=0.0),
        horizon: Optional[int] = Query(None, ge=1, le=38),
        with_fixtures: bool = Query(True),
        with_gws: bool = Query(False, description="embed every gameweek's breakdown "
                                                  "(large: this is the projection set)"),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        gw = int(gw or state.current_gw)
        if gw not in snapshot.gws:
            raise HTTPException(
                status_code=400,
                detail="GW%d is outside the projected range %d-%d; "
                       "POST /api/refresh with a larger horizon or use /api/projections"
                       % (gw, snapshot.gws[0], snapshot.gws[-1]),
            )
        horizon_gws = snapshot.gws[: int(horizon)] if horizon else snapshot.gws

        position = _parse_position(pos)
        team_id = _parse_team(state, team)
        needle = (search or "").strip().lower()

        rows: List[Dict[str, Any]] = []
        for pid, player in state.players.items():
            if position is not None and player.position != position:
                continue
            if team_id is not None and player.team_id != team_id:
                continue
            if max_cost is not None and player.price > max_cost + 1e-9:
                continue
            if min_cost is not None and player.price < min_cost - 1e-9:
                continue
            if needle and needle not in player.web_name.lower() \
                    and needle not in ("%s %s" % (player.first_name, player.second_name)).lower():
                continue
            row = player_row(snapshot, pid, gw, horizon_gws, with_fixtures=with_fixtures)
            if row["minutes"]["xmins"] < min_xmins:
                continue
            if with_gws:
                row["gws"] = player_gws(snapshot, pid, horizon_gws)
            rows.append(row)

        rows.sort(key=_sort_key(sort), reverse=(order == "desc"))
        total = len(rows)
        page = rows[offset:] if limit is None else rows[offset: offset + limit]
        return {
            "gw": gw,
            "horizon_gws": list(horizon_gws),
            "sort": sort,
            "order": order,
            "total": total,
            "offset": offset,
            "count": len(page),
            "players": page,
        }

    @app.get("/api/player/{player_id}")
    def get_player(
        player_id: int,
        gw: Optional[int] = Query(None, ge=1, le=38),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        if player_id not in state.players:
            raise HTTPException(status_code=404, detail="no player with id %d" % player_id)
        gw = int(gw or state.current_gw)
        if gw not in snapshot.gws:
            raise HTTPException(
                status_code=400,
                detail="GW%d is outside the projected range %d-%d"
                       % (gw, snapshot.gws[0], snapshot.gws[-1]),
            )
        player = state.players[player_id]
        row = player_row(snapshot, player_id, gw, snapshot.gws)

        by_gw = player_gws(snapshot, player_id, snapshot.gws)

        with snapshot.lock:
            try:
                explanation = snapshot.engine.explain(player_id, gw)
            except Exception as exc:
                explanation = "explanation unavailable: %s: %s" % (type(exc).__name__, exc)
            minutes = snapshot.engine.minutes_forecast(player_id, gw)
            upcoming = []
            for g in snapshot.gws:
                for fixture in snapshot.engine.fixtures_for(player.team_id, g):
                    upcoming.append(fixture_row(snapshot, fixture))

        history = state.history(player_id)
        past = state.history_past(player_id)
        return {
            "player": row,
            "by_gw": by_gw,
            "gws": by_gw,  # alias: the dashboard drawer reads `gws`
            "explanation": explanation,
            "minutes_reason": minutes.reason,
            "fixtures": upcoming,
            "history_rows": int(len(history)),
            "history_past": _clean([
                {k: v for k, v in season.items()
                 if k in ("season_name", "minutes", "starts", "total_points", "goals_scored",
                          "assists", "clean_sheets", "expected_goals", "expected_assists",
                          "expected_goals_conceded", "saves", "bonus", "bps",
                          "defensive_contribution", "yellow_cards", "red_cards")}
                for season in past
            ]),
        }

    # -- fixtures -----------------------------------------------------------
    @app.get("/api/fixtures")
    def get_fixtures(
        gw: Optional[int] = Query(None, ge=1, le=38),
        first: Optional[int] = Query(None, ge=1, le=38),
        last: Optional[int] = Query(None, ge=1, le=38),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        if gw is not None:
            wanted = [int(gw)]
        elif first is not None or last is not None:
            lo = int(first or state.current_gw)
            hi = int(last or lo)
            if hi < lo:
                raise HTTPException(status_code=400, detail="last (%d) is before first (%d)" % (hi, lo))
            wanted = list(range(lo, hi + 1))
        else:
            wanted = [state.current_gw]

        out = []
        for fixture in state.fixtures:
            if fixture.gw is None or int(fixture.gw) not in wanted:
                continue
            out.append(fixture_row(snapshot, fixture))
        counts: Dict[int, int] = {}
        with snapshot.lock:
            for tid in state.teams:
                for g in wanted:
                    if len(snapshot.engine.fixtures_for(tid, g)) != 1:
                        counts[tid] = counts.get(tid, 0) + 1
        return {
            "gws": wanted,
            "count": len(out),
            "fixtures": out,
            "teams_with_double_or_blank": [
                {"team": state.short_name(t), "team_id": t} for t in sorted(counts)
            ],
        }

    # -- projections --------------------------------------------------------
    @app.get("/api/projections")
    def get_projections(
        first: Optional[int] = Query(None, ge=1, le=38),
        last: Optional[int] = Query(None, ge=1, le=38),
        player_ids: Optional[str] = Query(None, description="comma-separated ids"),
        compact: bool = Query(False, description="drop the per-fixture rows"),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        lo = int(first or snapshot.gws[0])
        hi = int(last or snapshot.gws[-1])
        if hi < lo:
            raise HTTPException(status_code=400, detail="last (%d) is before first (%d)" % (hi, lo))

        if lo >= snapshot.gws[0] and hi <= snapshot.gws[-1]:
            projections = snapshot.projections
        else:
            key = (lo, hi)
            with snapshot.lock:
                projections = snapshot.extra.get(key)
                if projections is None:
                    projections = snapshot.engine.project(snapshot.state, list(range(lo, hi + 1)))
                    snapshot.extra[key] = projections
                    snapshot.haul.update(snapshot.engine.haul_table())

        payload = projection_set_to_dict(projections)
        wanted = None
        if player_ids:
            wanted = set()
            for chunk in player_ids.split(","):
                chunk = chunk.strip()
                if chunk:
                    try:
                        wanted.add(int(chunk))
                    except ValueError:
                        raise HTTPException(status_code=400,
                                            detail="player_ids must be integers, got %r" % chunk)
        trimmed: Dict[str, Any] = {}
        for pid, per_gw in payload["projections"].items():
            if wanted is not None and int(pid) not in wanted:
                continue
            rows = {}
            for gw, row in per_gw.items():
                if not (lo <= int(gw) <= hi):
                    continue
                rows[gw] = {k: v for k, v in row.items() if not (compact and k == "fixtures")}
            trimmed[pid] = rows
        payload["projections"] = trimmed
        payload["first_gw"] = lo
        payload["last_gw"] = hi
        payload["haul"] = {
            "%d|%d" % (pid, gw): value
            for (pid, gw), value in snapshot.haul.items()
            if lo <= gw <= hi and (wanted is None or pid in wanted)
        }
        return _clean(payload)

    # -- squad --------------------------------------------------------------
    @app.get("/api/squad")
    def get_squad(
        entry_id: Optional[int] = Query(None, ge=1),
        gw: Optional[int] = Query(None, ge=1, le=38),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        # Validate before doing any work. Without this an out-of-horizon gw
        # produced a structurally valid payload in which every xP was 0.0 and
        # the lineup had been picked off those zeros — silently wrong is worse
        # than a 400, and every sibling endpoint already 400s here.
        gw = int(gw or state.current_gw)
        if gw not in snapshot.gws:
            raise HTTPException(
                status_code=400,
                detail="GW%d is outside the projected range %d-%d; "
                       "POST /api/refresh with a larger horizon"
                       % (gw, snapshot.gws[0], snapshot.gws[-1]),
            )
        entry_id = entry_id or STORE.config.entry_id
        if entry_id is None:
            squad, source = _fallback_squad(snapshot)
            meta = {"warnings": ["no entry_id given; showing the recommended squad"]}
        else:
            squad, meta = load_squad_state(int(entry_id), gw)
            source = "entry"

        out = squad_dict(state, squad)
        # `gw` is the gameweek the xP figures below are for; the squad itself
        # may have been loaded from an earlier one.
        out["squad_gw"] = out["gw"]
        out["gw"] = gw
        out["source"] = source
        out["meta"] = _clean(meta)
        ids = squad.player_ids()
        out["players"] = [
            player_row(snapshot, pid, gw, snapshot.gws, with_fixtures=False)
            for pid in ids if pid in state.players
        ]
        out["xp_gw"] = sum(snapshot.projections.xp(pid, gw) for pid in ids)
        out["xp_horizon"] = sum(
            snapshot.projections.xp(pid, g) for pid in ids for g in snapshot.gws
        )
        try:
            lineup_fn = resolve("pick_lineup")
        except HTTPException:
            out["lineup"] = None
        else:
            with peer_errors("lineup selection"):
                lineup, bench, captain, vice = lineup_fn(ids, snapshot.projections, gw, state)
            out["lineup"] = {
                "gw": gw,
                "lineup": lineup,
                "bench": bench,
                "captain": captain,
                "vice_captain": vice,
                "lineup_players": [_named(state, p) for p in lineup],
                "bench_players": [_named(state, p) for p in bench],
                "captain_player": _named(state, captain),
                "vice_captain_player": _named(state, vice),
                "xp": sum(snapshot.projections.xp(p, gw) for p in lineup)
                      + snapshot.projections.xp(captain, gw),
            }
        return _clean(out)

    # -- optimize -----------------------------------------------------------
    @app.post("/api/optimize")
    def post_optimize(body: Optional[OptimizeRequest] = Body(None)) -> Dict[str, Any]:
        body = body or OptimizeRequest()
        snapshot = STORE.get()
        state = snapshot.state
        locked_in, locked_out = body.resolved_locks()

        config = Config.load()
        config.optimizer.horizon = int(body.horizon or STORE.config.optimizer.horizon)
        config.optimizer.locked_in = locked_in
        config.optimizer.locked_out = locked_out
        if body.decay is not None:
            config.optimizer.decay = float(body.decay)

        horizon = min(int(config.optimizer.horizon), len(snapshot.gws))
        gws = snapshot.gws[:horizon]

        entry_id = body.entry_id if body.entry_id is not None else STORE.config.entry_id
        squad: Optional[SquadState] = None
        source = "none"
        if body.squad:
            picks = []
            for i, pid in enumerate(body.squad, start=1):
                if int(pid) not in state.players:
                    raise HTTPException(status_code=400, detail="no player with id %s" % pid)
                price = state.players[int(pid)].now_cost
                picks.append(SquadPick(player_id=int(pid), purchase_price=price,
                                       selling_price=price, position_in_squad=i))
            # Reject an illegal fifteen here, with the reasons, rather than
            # letting the MILP come back "Infeasible" and say nothing useful.
            budget = sum(p.selling_price for p in picks) + int(round((body.bank or 0.0) * 10))
            problems = resolve("squad_problems")(
                [p.player_id for p in picks], state, budget)
            if problems:
                raise HTTPException(
                    status_code=400,
                    detail="the squad in the request body is not legal: %s" % "; ".join(problems),
                )
            squad = SquadState(
                gw=state.current_gw, picks=picks,
                bank=int(round((body.bank or 0.0) * 10)),
                free_transfers=int(body.free_transfers or 1),
                entry_id=entry_id,
            )
            source = "body"
        elif entry_id is not None:
            try:
                squad, _meta = load_squad_state(int(entry_id))
                source = "entry"
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                squad = None
                source = "none (%s)" % exc.detail

        if squad is not None and squad.picks:
            plan_fn = resolve("plan")
            if body.free_transfers is not None:
                squad.free_transfers = int(body.free_transfers)
            if body.bank is not None:
                squad.bank = int(round(body.bank * 10))
            with peer_errors("the transfer planner"):
                result = plan_fn(state, squad, snapshot.projections, config, body.chips)
            plan_payload = plan_dict(state, result)
            # The plan is returned both nested and spread across the top level:
            # the dashboard treats any object with `decisions` as the plan.
            return _clean(dict(plan_payload, **{
                "mode": "plan",
                "squad_source": source,
                "gws": gws,
                "chips_requested": body.chips,
                "locked_in": locked_in,
                "locked_out": locked_out,
                "plan": plan_payload,
            }))

        pick = resolve("pick_initial_squad")
        with peer_errors("the squad optimizer"):
            decision = pick(snapshot.projections, state, config, gws)
        payload = decision_dict(state, decision)
        # Same single-shape rule as the plan branch: one gameweek's decision,
        # presented as a one-entry plan so the dashboard renders it identically.
        return _clean({
            "mode": "initial_squad",
            "squad_source": source,
            "first_gw": decision.gw,
            "horizon": horizon,
            "gws": gws,
            "decay": config.optimizer.decay,
            "objective": decision.expected_points,
            "solver_status": "",
            "locked_in": locked_in,
            "locked_out": locked_out,
            "decision": payload,
            "decisions": [payload],
        })

    # -- captain ------------------------------------------------------------
    @app.get("/api/captain")
    def get_captain(
        gw: Optional[int] = Query(None, ge=1, le=38),
        entry_id: Optional[int] = Query(None, ge=1),
        squad: Optional[str] = Query(None, description="comma-separated player ids"),
        limit: int = Query(15, ge=1, le=100),
    ) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        gw = int(gw or state.current_gw)
        if gw not in snapshot.gws:
            raise HTTPException(status_code=400,
                                detail="GW%d is outside the projected range %d-%d"
                                       % (gw, snapshot.gws[0], snapshot.gws[-1]))
        entry_id = entry_id if entry_id is not None else STORE.config.entry_id
        if squad:
            ids = [int(p) for p in squad.split(",") if p.strip()]
            source = "query"
        elif entry_id is not None:
            # An entry with no published picks yet — which is every entry until
            # the GW1 deadline — must not kill this endpoint. /api/chips and
            # /api/optimize already fall back to the recommended fifteen here.
            try:
                squad_state, _meta = load_squad_state(int(entry_id), gw)
                ids = squad_state.player_ids()
                source = "entry"
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                squad_state, source = _fallback_squad(snapshot)
                ids = squad_state.player_ids()
        else:
            squad_state, source = _fallback_squad(snapshot)
            ids = squad_state.player_ids()
        options_fn = resolve("captain_options")
        with peer_errors("the captaincy model"):
            options = options_fn(ids, snapshot.projections, gw, state, STORE.config)
        return _clean({
            "gw": gw,
            "squad_source": source,
            "squad": ids,
            "options": [captain_dict(state, o) for o in options[:limit]],
        })

    # -- chips --------------------------------------------------------------
    @app.get("/api/chips")
    def get_chips(entry_id: Optional[int] = Query(None, ge=1)) -> Dict[str, Any]:
        snapshot = STORE.get()
        state = snapshot.state
        entry_id = entry_id if entry_id is not None else STORE.config.entry_id
        if entry_id is not None:
            try:
                squad, _meta = load_squad_state(int(entry_id))
                source = "entry"
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                squad, source = _fallback_squad(snapshot)
        else:
            squad, source = _fallback_squad(snapshot)

        detect = resolve("detect_double_blank_gws")
        recommend = resolve("recommend_chips")
        with peer_errors("the chip advisor"):
            doubles_blanks = detect(state)
            # Free Hit and Wildcard each solve their own MILP, so this is the
            # most expensive endpoint in the API; the answer only changes when
            # the snapshot or the squad does.
            key = tuple(sorted(squad.player_ids()))
            with snapshot.lock:
                recommendations = snapshot.chips.get(key)
                if recommendations is None:
                    recommendations = recommend(state, squad, snapshot.projections, STORE.config)
                    snapshot.chips[key] = recommendations
        expiry = _chip_expiry(state, squad)
        return _clean({
            "squad_source": source,
            "current_gw": state.current_gw,
            "chips_used": squad.chips_used,
            # `expected_gain` is the dashboard's name for the same number the
            # chip advisor calls `points_gain`.
            "recommendations": [
                dict(rec, expected_gain=rec.get("points_gain")) for rec in recommendations
            ],
            "double_blank_gws": doubles_blanks,
            "doubles": {gw: info.get("doubles", []) for gw, info in doubles_blanks.items()},
            "blanks": {gw: info.get("blanks", []) for gw, info in doubles_blanks.items()},
            "first_half_deadline": expiry["deadline"],
            "expiry": expiry,
        })

    # -- refresh ------------------------------------------------------------
    @app.post("/api/refresh")
    def post_refresh(body: Optional[RefreshRequest] = Body(None)) -> Dict[str, Any]:
        body = body or RefreshRequest()
        if body.horizon:
            STORE.horizon = int(body.horizon)
        t0 = time.time()
        try:
            snapshot = STORE.refresh(force=bool(body.force))
        except FPLError as exc:
            raise HTTPException(status_code=502, detail="FPL API refresh failed: %s" % exc)
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail="refresh failed: %s: %s" % (type(exc).__name__, exc))
        return _clean({
            "refreshed": True,
            "forced": bool(body.force),
            "seconds": round(time.time() - t0, 2),
            "built_at": snapshot.built_iso,
            "gws": snapshot.gws,
            "counts": {
                "teams": len(snapshot.state.teams),
                "players": len(snapshot.state.players),
                "fixtures": len(snapshot.state.fixtures),
            },
            "build_seconds": snapshot.seconds,
            "data": cache_freshness(),
            "model_warnings": snapshot.engine.warnings,
            "data_warnings": snapshot.state.data_warnings,
        })

    # -- static dashboard ---------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> Any:
        path = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(path):
            return FileResponse(path)
        return HTMLResponse(
            "<html><head><title>gaffer</title></head><body "
            "style='font:14px ui-monospace,monospace;background:#111;color:#ddd;padding:2rem'>"
            "<h1>gaffer</h1><p>The API is up but no dashboard is installed at "
            "<code>%s</code>.</p><p>Try <a style='color:#6cf' href='/api/health'>/api/health</a> "
            "or <a style='color:#6cf' href='/api/docs'>/api/docs</a>.</p></body></html>" % WEB_DIR,
            status_code=200,
        )

    if os.path.isdir(WEB_DIR):
        # Mounted last so every /api route above wins the match.
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app


_ERROR_NAMES = {
    400: "bad_request",
    404: "not_found",
    422: "invalid_request",
    500: "internal_error",
    502: "upstream_error",
    503: "unavailable",
}


def _web_files() -> List[str]:
    try:
        return sorted(os.listdir(WEB_DIR))
    except OSError:
        return []


def _parse_position(pos: Optional[str]) -> Optional[int]:
    if pos is None or pos == "":
        return None
    text = str(pos).strip().upper()
    if text in scoring.POS_ID:
        return scoring.POS_ID[text]
    try:
        value = int(text)
    except ValueError:
        raise HTTPException(status_code=400,
                            detail="pos must be one of GKP, DEF, MID, FWD or 1-4, got %r" % pos)
    if value not in scoring.POSITIONS:
        raise HTTPException(status_code=400, detail="pos %d is not 1-4" % value)
    return value


def _parse_team(state: GameState, team: Optional[str]) -> Optional[int]:
    if team is None or team == "":
        return None
    text = str(team).strip()
    for t in state.teams.values():
        if text.upper() == t.short_name.upper() or text.lower() == t.name.lower():
            return t.id
    try:
        value = int(text)
    except ValueError:
        raise HTTPException(status_code=400, detail="unknown team %r" % team)
    if value not in state.teams:
        raise HTTPException(status_code=400, detail="no team with id %d" % value)
    return value


def _num(value: Any) -> float:
    """Sortable float. ``_clean`` turns NaN into None, which does not compare."""
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _sort_key(sort: str) -> Any:
    if sort not in SORT_KEYS:
        raise HTTPException(status_code=400,
                            detail="sort must be one of %s, got %r" % (", ".join(SORT_KEYS), sort))
    if sort == "name":
        return lambda r: r["name"].lower()
    if sort == "team":
        return lambda r: r["team"]
    if sort == "ownership":
        return lambda r: _num(r["selected_by_percent"])
    if sort == "xmins":
        return lambda r: _num(r["minutes"]["xmins"])
    if sort == "p_defcon":
        return lambda r: _num(r["rates"]["p_defcon"])
    return lambda r: _num(r.get(sort))


def _chip_expiry(state: GameState, squad: SquadState) -> Dict[str, Any]:
    """First-half chips die at the GW19 deadline; escalate the warning from GW14.

    Two of each chip are issued, one per half of the season. An unplayed
    first-half chip is simply lost on 2 January, which is the single most
    common unforced error in FPL.
    """
    used = set(squad.chips_used or [])
    current = state.current_gw
    deadline = state.deadline(scoring.FIRST_HALF_LAST_GW)
    remaining_gws = max(0, scoring.FIRST_HALF_LAST_GW - current + 1)
    first_half = [c for c in scoring.CHIP_WINDOWS if c not in used]
    if current > scoring.FIRST_HALF_LAST_GW:
        level = "expired"
    elif remaining_gws <= 2:
        level = "critical"
    elif current >= 14:
        level = "urgent"
    elif current >= 10:
        level = "watch"
    else:
        level = "none"
    warnings = []
    if first_half and level not in ("none",):
        for chip in first_half:
            warnings.append(
                "%s unused with %d gameweek%s left before the GW%d deadline (%s)"
                % (chip, remaining_gws, "" if remaining_gws == 1 else "s",
                   scoring.FIRST_HALF_LAST_GW, deadline or "unknown")
            )
    return {
        "first_half_last_gw": scoring.FIRST_HALF_LAST_GW,
        "deadline": deadline,
        "gws_remaining": remaining_gws,
        "unused_first_half_chips": first_half,
        "level": level,
        "warnings": warnings,
    }


app = create_app()


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT, log_level: str = "info") -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=log_level)


# ---------------------------------------------------------------------------
# Smoke test: start the real server and curl every endpoint.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import socket
    import sys

    import requests
    import uvicorn

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    base = "http://127.0.0.1:%d" % port

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    print("gaffer API smoke test on %s" % base)
    t0 = time.time()
    ready = False
    while time.time() - t0 < 240:
        try:
            health = requests.get(base + "/api/health", timeout=10).json()
        except Exception:
            time.sleep(0.5)
            continue
        if health.get("status") in ("ok", "degraded"):
            ready = True
            break
        if health.get("status") == "error":
            print("startup failed: %s" % health["snapshot"]["last_error"])
            break
        time.sleep(1.0)
    print("  ready in %.1fs (status=%s)\n" % (time.time() - t0, ready))

    failures = []

    def call(method: str, path: str, expect: Any = 200, body: Any = None,
             show: Any = "", want_json: bool = True) -> Any:
        expected = (expect,) if isinstance(expect, int) else tuple(expect)
        url = base + path
        t = time.time()
        try:
            if method == "GET":
                resp = requests.get(url, timeout=180)
            else:
                resp = requests.post(url, json=body or {}, timeout=600)
        except Exception as exc:
            failures.append("%s %s -> %s" % (method, path, exc))
            print("  FAIL %-46s %s" % (path, exc))
            return None
        ms = 1000 * (time.time() - t)
        if not want_json:
            ok = resp.status_code in expected
            if not ok:
                failures.append("%s %s -> HTTP %d (want %s)" % (method, path, resp.status_code, expected))
            print("  %-4s %-46s %3d %7.0fms  %s"
                  % ("ok" if ok else "FAIL", path, resp.status_code, ms,
                     "%s, %d bytes" % (resp.headers.get("content-type", "?"), len(resp.content))))
            return resp.text
        try:
            payload = resp.json()
        except ValueError:
            failures.append("%s %s -> non-JSON body" % (method, path))
            print("  FAIL %-46s non-JSON" % path)
            return None
        ok = resp.status_code in expected
        if not ok:
            failures.append("%s %s -> HTTP %d (want %s): %s"
                            % (method, path, resp.status_code, expected,
                               json.dumps(payload)[:160]))
        note = show or ""
        if callable(show):
            try:
                note = show(payload)
            except Exception as exc:
                note = "note failed: %s" % exc
        label = "ok" if ok else "FAIL"
        if ok and resp.status_code == 503:
            label = "PEER"  # the peer module is not built yet; the API is fine
        print("  %-4s %-46s %3d %7.0fms  %s" % (label, path, resp.status_code, ms, note))
        return payload

    call("GET", "/api/health", show=lambda p: "status=%s players=%s fit=%.2fs modules=%s"
         % (p["status"], p.get("counts", {}).get("players"),
            sum(p["model"]["fit_seconds"].values()),
            "/".join(sorted(set(p["modules"].values())))))
    call("GET", "/api/state", show=lambda p: "GW%d deadline %s, %d teams"
         % (p["current_gw"], p["deadline"], len(p["teams"])))
    players = call("GET", "/api/players?gw=1&limit=5",
                   show=lambda p: "%d of %d, top=%s %.2f xP"
                   % (p["count"], p["total"], p["players"][0]["name"], p["players"][0]["xp"]))
    call("GET", "/api/players?gw=1&pos=DEF&max_cost=5.0&sort=value&limit=3",
         show=lambda p: "top value DEF <=5.0: %s"
         % ", ".join("%s %.2f" % (r["name"], r["value"]) for r in p["players"]))
    call("GET", "/api/players?gw=1&sort=nonsense", expect=400,
         show=lambda p: p["detail"][:60])
    top_id = players["players"][0]["id"] if players and players.get("players") else 1
    call("GET", "/api/player/%d" % top_id,
         show=lambda p: "%s, %d gws, explanation %d chars"
         % (p["player"]["name"], len(p["by_gw"]), len(p["explanation"])))
    call("GET", "/api/player/999999", expect=404, show=lambda p: p["detail"])
    call("GET", "/api/fixtures?gw=1",
         show=lambda p: "%d fixtures, first %s %.2f-%.2f"
         % (p["count"], p["fixtures"][0]["home"]["short"],
            p["fixtures"][0]["home"]["lambda"], p["fixtures"][0]["away"]["lambda"]))
    call("GET", "/api/projections?first=1&last=2&compact=1&player_ids=%d" % top_id,
         show=lambda p: "%d players, gws %d-%d" % (len(p["projections"]), p["first_gw"], p["last_gw"]))
    call("GET", "/api/squad", expect=(200, 503),
         show=lambda p: "source=%s, %d picks" % (p["source"], len(p["picks"]))
         if "picks" in p else str(p.get("detail"))[:70])
    call("POST", "/api/optimize", body={"horizon": 3}, expect=(200, 503),
         show=lambda p: str(p.get("mode") or p.get("detail"))[:70])
    call("GET", "/api/captain?gw=1", expect=(200, 503),
         show=lambda p: "%d options" % len(p["options"]) if "options" in p
         else str(p.get("detail"))[:70])
    call("GET", "/api/chips", expect=(200, 503),
         show=lambda p: "%d recommendations, expiry=%s"
         % (len(p["recommendations"]), p["expiry"]["level"]) if "recommendations" in p
         else str(p.get("detail"))[:70])
    call("POST", "/api/refresh", body={"force": False},
         show=lambda p: "rebuilt in %ss" % p.get("seconds"))
    call("GET", "/", want_json=False)
    call("GET", "/api/openapi.json", show=lambda p: "%d paths" % len(p["paths"]))

    # GATE: every player payload carries the whole breakdown, and it adds up.
    print("\ncomponent breakdown gate")
    full = requests.get(base + "/api/players?gw=1&limit=1000", timeout=180).json()
    rows = full["players"]
    keys = set(_COMPONENT_KEYS)
    worst = 0.0
    seen = {k: 0 for k in keys}
    missing = 0
    for row in rows:
        if set(row["components"]) != keys:
            missing += 1
            continue
        total = sum(row["components"].values())
        worst = max(worst, abs(total - (row["xp"] or 0.0)))
        for k, v in row["components"].items():
            if abs(v) > 1e-12:
                seen[k] += 1
    empty = [k for k, n in seen.items() if n == 0]
    print("  %d players, %d missing a component key, max |sum(components) - xp| = %.2e"
          % (len(rows), missing, worst))
    print("  components populated somewhere: %s"
          % ", ".join("%s=%d" % (k, seen[k]) for k in sorted(seen)))
    if missing or worst > 1e-9 or empty or len(rows) != 587:
        failures.append("component gate: missing=%d worst=%.2e empty=%s rows=%d"
                        % (missing, worst, empty, len(rows)))

    print("\n%d failure(s)" % len(failures))
    for f in failures:
        print("  - %s" % f)
    server.should_exit = True
    thread.join(timeout=10)
    sys.exit(1 if failures else 0)
