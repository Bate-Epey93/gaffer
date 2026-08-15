"""Decide whether the cached FPL data is stale, and bring it forward if it is.

``python -m gaffer.cli autorefresh`` is built to be called on a timer — launchd
runs it every hour — so the *usual* answer must be "nothing to do", reached in
well under a second and without a single HTTP request. Everything expensive
lives behind that decision.

The policy has two speeds, because the value of fresh data is not constant:

* normally, refetch when the cached ``bootstrap`` is older than 24 hours;
* when the next deadline is less than 12 hours away, tighten that to 2 hours,
  because press conferences, injury news and price changes all land in the last
  day and that is exactly when the dashboard gets opened.

Staleness is measured on the ``bootstrap`` key alone. Prices, team news, player
status, chip availability and all 38 deadlines arrive in that one payload, so
its age *is* the age of everything a pre-deadline decision depends on.

A fetch is only half the job. The dashboard is instant because a fitted
projection set is already sitting in ``data/cache``; a run that refetched the
data and left the projections behind would make the next dashboard open pay for
a full fit. So after fetching, this refits the model and rewrites the projection
cache for the same gameweek range the server builds. It also handles the
converse: data still fresh but the projection file missing or older than the
bootstrap it was built from is a pure local recompute, with no network at all.

Two safety properties matter because this runs unattended:

* **single flight.** An ``flock`` on ``data/autorefresh.lock`` means a slow run
  and the next hourly tick cannot both be fetching. The loser exits 0 quietly.
* **backing off.** Failures are counted in ``data/autorefresh.json`` and the
  next attempt is pushed out 15m, 30m, 1h, 2h, 4h, then 6h. If the FPL API is
  down, gaffer stops knocking instead of hammering it every hour.

It never runs the backtest. The backtest walks a whole season and takes ~170s;
nothing on a timer or on an HTTP path may start it, and this module does not
import ``gaffer.backtest`` at all.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gaffer.core import scoring
from gaffer.core.config import CACHE_DIR, DATA_DIR, Config, ensure_dirs
from gaffer.data.cache import TTL_FOREVER, Cache
from gaffer.data.fpl_api import FPLClient, FPLError
from gaffer.data.loaders import load_game_state, resolve_current_gw
from gaffer.model.xp import XPEngine, projections_cache_path

# -- policy -----------------------------------------------------------------

#: Normal staleness limit for the cached bootstrap.
DAILY_MAX_AGE = 24 * 3600.0
#: How close a deadline has to be before the tighter limit applies.
DEADLINE_WINDOW = 12 * 3600.0
#: Staleness limit inside that window.
PRE_DEADLINE_MAX_AGE = 2 * 3600.0
#: The cache key whose age stands for "how old is the data".
FRESHNESS_KEY = "bootstrap"

#: Backoff after consecutive failures: 15m, 30m, 1h, 2h, 4h, then capped.
BACKOFF_FIRST = 15 * 60.0
BACKOFF_MAX = 6 * 3600.0

LOCK_PATH = os.path.join(DATA_DIR, "autorefresh.lock")
STATE_PATH = os.path.join(DATA_DIR, "autorefresh.json")

ACTION_FETCH = "fetch"
ACTION_RECOMPUTE = "recompute"
ACTION_SKIP = "skip"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def human_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 90:
        return "%s%.0fs" % (sign, seconds)
    if seconds < 5400:
        return "%s%.0fm" % (sign, seconds / 60.0)
    if seconds < 48 * 3600:
        return "%s%.1fh" % (sign, seconds / 3600.0)
    return "%s%.1fd" % (sign, seconds / 86400.0)


def backoff_seconds(failures: int) -> float:
    """Delay before the next attempt after ``failures`` consecutive errors."""
    if failures <= 0:
        return 0.0
    return min(BACKOFF_FIRST * (2.0 ** (failures - 1)), BACKOFF_MAX)


# ---------------------------------------------------------------------------
# Run state (last success, failure streak, backoff)
# ---------------------------------------------------------------------------


def read_state(path: str = STATE_PATH) -> Dict[str, Any]:
    """The persisted run state. A missing or corrupt file reads as empty."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(state: Dict[str, Any], path: str = STATE_PATH) -> None:
    """Persist run state atomically; a killed run must not corrupt it."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".autorefresh-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def record_success(
    now: datetime, stats: Dict[str, Any], path: str = STATE_PATH
) -> Dict[str, Any]:
    state = read_state(path)
    state.update({
        "last_attempt": now.isoformat(),
        "last_success": now.isoformat(),
        "last_action": stats.get("action"),
        "last_seconds": round(float(stats.get("seconds", 0.0)), 2),
        "consecutive_failures": 0,
        "last_error": None,
        "retry_after": None,
    })
    write_state(state, path)
    return state


def record_failure(
    now: datetime, error: str, path: str = STATE_PATH
) -> Dict[str, Any]:
    state = read_state(path)
    failures = int(state.get("consecutive_failures") or 0) + 1
    delay = backoff_seconds(failures)
    state.update({
        "last_attempt": now.isoformat(),
        "consecutive_failures": failures,
        "last_error": error[:500],
        "retry_after": (now + timedelta(seconds=delay)).isoformat(),
    })
    write_state(state, path)
    return state


# ---------------------------------------------------------------------------
# Single flight
# ---------------------------------------------------------------------------


class SingleFlight:
    """An advisory ``flock`` so two runs never fetch at the same time.

    launchd fires this hourly and a cold run can take half a minute; the point
    is that the second caller finds the lock held and leaves, rather than
    doubling the load on the FPL API. The lock is released by the kernel when
    the process dies, so a crashed run cannot wedge every later one.
    """

    def __init__(self, path: str = LOCK_PATH) -> None:
        self.path = path
        self.fd: Optional[int] = None
        self.holder: str = ""

    def acquire(self) -> bool:
        import fcntl  # POSIX only, and this whole feature is macOS/launchd

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                self.holder = os.read(fd, 64).decode("utf-8", "replace").strip()
            except OSError:
                self.holder = ""
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, ("pid %d at %s\n" % (os.getpid(), utcnow().isoformat())).encode())
        self.fd = fd
        return True

    def release(self) -> None:
        import fcntl

        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    action: str
    reason: str
    cache_age: Optional[float] = None
    max_age: float = DAILY_MAX_AGE
    next_gw: Optional[int] = None
    deadline: Optional[datetime] = None
    seconds_to_deadline: Optional[float] = None
    near_deadline: bool = False
    gws: List[int] = field(default_factory=list)
    projections_path: str = ""
    projections_fresh: bool = False
    consecutive_failures: int = 0
    retry_after: Optional[datetime] = None

    @property
    def works(self) -> bool:
        return self.action in (ACTION_FETCH, ACTION_RECOMPUTE)

    @property
    def fetches(self) -> bool:
        return self.action == ACTION_FETCH


@dataclass
class CachedData:
    """What the cached bootstrap tells us, without going near the network."""

    exists: bool = False
    fetched_at: Optional[datetime] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def age(self, now: datetime) -> Optional[float]:
        """Seconds since this data was fetched, measured against ``now``.

        Not ``Cache.age_seconds``: that measures against the wall clock, and the
        policy has to use the same clock it compares deadlines against or the
        two halves of a decision disagree.
        """
        if not self.exists or self.fetched_at is None:
            return None
        return max(0.0, (now - self.fetched_at).total_seconds())


def read_cached_bootstrap(cache: Optional[Cache] = None) -> CachedData:
    """Age and the 38 events, from one read of the cached bootstrap.

    Read at any age on purpose: a bootstrap that is *too old to use* is still
    the right place to read deadlines from, and the deadlines are what decide
    whether we are allowed to go and fetch a newer one.
    """
    cache = cache or Cache(default_ttl=TTL_FOREVER)
    path = cache.path_for(FRESHNESS_KEY)
    if not os.path.exists(path):
        return CachedData()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        # A truncated cache file is a cold start, not a crash.
        return CachedData()
    if not isinstance(payload, dict) or "data" not in payload:
        return CachedData()
    fetched = parse_iso(payload.get("fetched_at"))
    if fetched is None:
        fetched = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    data = payload.get("data")
    events = data.get("events") if isinstance(data, dict) else None
    return CachedData(exists=True, fetched_at=fetched,
                      events=list(events) if isinstance(events, list) else [])


def cached_events(cache: Optional[Cache] = None) -> List[Dict[str, Any]]:
    return read_cached_bootstrap(cache).events


def next_deadline(
    events: Sequence[Dict[str, Any]], now: Optional[datetime] = None
) -> Optional[Tuple[int, datetime]]:
    """``(gw, deadline)`` for the soonest deadline still in the future."""
    now = now or utcnow()
    best: Optional[Tuple[int, datetime]] = None
    for event in events:
        when = parse_iso(event.get("deadline_time"))
        if when is None or when <= now:
            continue
        if best is None or when < best[1]:
            best = (int(event["id"]), when)
    return best


def horizon_gws(events: Sequence[Dict[str, Any]], config: Config,
                now: Optional[datetime] = None) -> List[int]:
    """The gameweek range the server projects, derived without loading state.

    Must match ``Store._build`` exactly, or autorefresh would warm a projection
    file the server never reads.
    """
    current = resolve_current_gw(events, now) if events else 1
    horizon = int(config.model.default_horizon)
    last = min(scoring.TOTAL_GWS, current + horizon - 1)
    return list(range(current, last + 1)) or [current]


def projections_are_fresh(path: str) -> bool:
    """A projection file counts only if it is newer than its own inputs."""
    if not os.path.exists(path):
        return False
    bootstrap = os.path.join(CACHE_DIR, "bootstrap.json")
    if os.path.exists(bootstrap) and os.path.getmtime(bootstrap) > os.path.getmtime(path):
        return False
    return True


def plan(
    config: Config,
    now: Optional[datetime] = None,
    force: bool = False,
    state_path: str = STATE_PATH,
    cache: Optional[Cache] = None,
) -> Decision:
    """Work out what, if anything, needs doing. Touches the network never."""
    now = now or utcnow()
    cache = cache or Cache(default_ttl=config.cache_ttl_seconds)
    run_state = read_state(state_path)
    failures = int(run_state.get("consecutive_failures") or 0)
    retry_after = parse_iso(run_state.get("retry_after"))

    cached = read_cached_bootstrap(cache)
    events = cached.events
    upcoming = next_deadline(events, now)
    gws = horizon_gws(events, config, now)
    path = projections_cache_path(gws[0], gws[-1])
    fresh_projections = projections_are_fresh(path)
    age = cached.age(now)

    to_deadline = None
    near = False
    max_age = DAILY_MAX_AGE
    if upcoming is not None:
        to_deadline = (upcoming[1] - now).total_seconds()
        near = to_deadline <= DEADLINE_WINDOW
        if near:
            max_age = PRE_DEADLINE_MAX_AGE

    def decide(action: str, reason: str) -> Decision:
        return Decision(
            action=action, reason=reason, cache_age=age, max_age=max_age,
            next_gw=upcoming[0] if upcoming else None,
            deadline=upcoming[1] if upcoming else None,
            seconds_to_deadline=to_deadline, near_deadline=near, gws=gws,
            projections_path=path, projections_fresh=fresh_projections,
            consecutive_failures=failures, retry_after=retry_after,
        )

    if force:
        return decide(ACTION_FETCH, "--force")

    # Backing off comes before every other test: if the API just failed, the
    # fact that the data is stale is the reason we are backing off, not a
    # reason to try again.
    if retry_after is not None and retry_after > now:
        return decide(
            ACTION_SKIP,
            "backing off after %d consecutive failures; next attempt in %s"
            % (failures, human_seconds((retry_after - now).total_seconds())),
        )

    if age is None:
        return decide(ACTION_FETCH, "no cached bootstrap; this is a cold start")

    if age >= max_age:
        if near:
            return decide(
                ACTION_FETCH,
                "GW%d deadline in %s and the data is %s old (limit %s near a deadline)"
                % (upcoming[0], human_seconds(to_deadline), human_seconds(age),
                   human_seconds(max_age)),
            )
        return decide(
            ACTION_FETCH,
            "data is %s old (daily limit %s)" % (human_seconds(age), human_seconds(max_age)),
        )

    if not fresh_projections:
        return decide(
            ACTION_RECOMPUTE,
            "data is current but %s is %s" % (
                os.path.basename(path),
                "missing" if not os.path.exists(path) else "older than the data it was built from",
            ),
        )

    return decide(
        ACTION_SKIP,
        "data is %s old, limit %s%s; projections current"
        % (human_seconds(age), human_seconds(max_age),
           " (deadline in %s)" % human_seconds(to_deadline) if near else ""),
    )


# ---------------------------------------------------------------------------
# The work
# ---------------------------------------------------------------------------


class _Silent:
    enabled = False

    def say(self, message: str) -> None:
        pass

    def step(self, message: str) -> "_Silent":
        return self

    def __enter__(self) -> "_Silent":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def perform(decision: Decision, config: Config, progress: Any = None) -> Dict[str, Any]:
    """Do what ``decision`` calls for. Raises on failure; the caller backs off.

    Deliberately *not* here: anything that walks a season. The backtest takes
    about 170 seconds and belongs to an interactive run only.
    """
    progress = progress or _Silent()
    started = time.time()
    ensure_dirs()
    stats: Dict[str, Any] = {"action": decision.action, "fetched": False}

    if decision.fetches:
        cache = Cache(default_ttl=config.cache_ttl_seconds)
        client = FPLClient(config, cache)
        with progress.step("fetching bootstrap-static (prices, news, injuries, deadlines)"):
            client.bootstrap(force=True)
        with progress.step("fetching fixtures"):
            client.fixtures(force=True)
        try:
            with progress.step("fetching event status"):
                client.event_status()
        except FPLError as exc:  # not load-bearing; never sink a run for it
            progress.say("  event-status unavailable (%s); continuing" % exc)
        stats["fetched"] = True

    t0 = time.time()
    # force=False on purpose: element summaries carry their own 6-hour TTL, so
    # this refetches the 587 player pages at most four times a day instead of
    # on every run.
    state = load_game_state(config, with_histories=True, force=False,
                            progress=bool(getattr(progress, "enabled", False)))
    stats["load_seconds"] = round(time.time() - t0, 2)
    stats["players"] = len(state.players)
    stats["current_gw"] = state.current_gw

    t1 = time.time()
    engine = XPEngine(config)
    with progress.step("fitting the model"):
        engine.fit(state)
    stats["fit_seconds"] = round(time.time() - t1, 2)

    last = min(scoring.TOTAL_GWS, state.current_gw + int(config.model.default_horizon) - 1)
    gws = list(range(state.current_gw, last + 1)) or [state.current_gw]
    t2 = time.time()
    with progress.step("projecting %d players over GW%d-%d" % (len(state.players), gws[0], gws[-1])):
        projections = engine.project(state, gws)
    stats["project_seconds"] = round(time.time() - t2, 2)

    stats["projections_path"] = engine.save_projections(projections)
    stats["gws"] = gws
    stats["seconds"] = round(time.time() - started, 2)
    for warning in state.data_warnings:
        progress.say("  data warning: %s" % warning)
    return stats
