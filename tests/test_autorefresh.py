"""The unattended refresh policy.

Everything here runs against a temporary cache directory, so the decisions are
asserted against *constructed* ages and deadlines rather than whatever the real
``data/cache`` happens to look like when the suite runs. Nothing touches the
network: ``plan`` is a pure function of the clock, the cache and the run state,
which is the property that makes an hourly launchd job safe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from gaffer.core.config import PROJECT_ROOT, Config
from gaffer.data.cache import Cache
from gaffer.ops import autorefresh as auto

NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures: a cache directory we control completely
# ---------------------------------------------------------------------------


def write_bootstrap(cache_dir: str, age_seconds: float, deadlines: dict) -> None:
    """A ``bootstrap`` cache entry of a chosen age holding chosen deadlines."""
    events = [
        {"id": gw, "deadline_time": when.isoformat().replace("+00:00", "Z"),
         "finished": when < NOW}
        for gw, when in sorted(deadlines.items())
    ]
    payload = {
        "fetched_at": (NOW - timedelta(seconds=age_seconds)).isoformat(),
        "data": {"events": events, "elements": [], "teams": []},
    }
    with open(os.path.join(cache_dir, "bootstrap.json"), "w") as fh:
        json.dump(payload, fh)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """An isolated cache dir, projection path and run-state file."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir)
    monkeypatch.setattr(auto, "CACHE_DIR", cache_dir)
    monkeypatch.setattr("gaffer.model.xp.CACHE_DIR", cache_dir)
    config = Config()
    config.cache_ttl_seconds = 300

    class Env:
        dir = cache_dir
        cache = Cache(cache_dir, default_ttl=300)
        state_path = str(tmp_path / "autorefresh.json")
        cfg = config

        def plan(self, force: bool = False, now: datetime = NOW):
            return auto.plan(self.cfg, now=now, force=force,
                             state_path=self.state_path, cache=self.cache)

        def touch_projections(self, first: int = 1, last: int = 6,
                              newer_than_bootstrap: bool = True) -> str:
            from gaffer.model.xp import projections_cache_path

            path = projections_cache_path(first, last)
            with open(path, "w") as fh:
                fh.write("{}")
            boot = os.path.join(cache_dir, "bootstrap.json")
            if os.path.exists(boot):
                stamp = os.path.getmtime(boot) + (60 if newer_than_bootstrap else -60)
                os.utime(path, (stamp, stamp))
            return path

    return Env()


def far_deadlines() -> dict:
    """GW1 six days out: the normal, nothing-imminent case."""
    return {gw: NOW + timedelta(days=6 + 7 * (gw - 1)) for gw in range(1, 6)}


def imminent_deadlines(hours: float) -> dict:
    return {1: NOW + timedelta(hours=hours), 2: NOW + timedelta(hours=hours + 168)}


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_cold_start_fetches(env):
    decision = env.plan()
    assert decision.action == auto.ACTION_FETCH
    assert "cold start" in decision.reason


def test_fresh_data_and_fresh_projections_does_nothing(env):
    write_bootstrap(env.dir, age_seconds=3600, deadlines=far_deadlines())
    env.touch_projections()
    decision = env.plan()
    assert decision.action == auto.ACTION_SKIP
    assert decision.max_age == auto.DAILY_MAX_AGE
    assert not decision.works


def test_data_older_than_a_day_fetches(env):
    write_bootstrap(env.dir, age_seconds=25 * 3600, deadlines=far_deadlines())
    env.touch_projections()
    decision = env.plan()
    assert decision.action == auto.ACTION_FETCH
    assert "daily limit" in decision.reason


@pytest.mark.parametrize("hours_out,age_hours,expected", [
    # Inside the 12h window the limit tightens to 2h, so 3h-old data is stale
    # that close to a deadline and perfectly fine the day before.
    (6.0, 3.0, auto.ACTION_FETCH),
    (6.0, 1.0, auto.ACTION_SKIP),
    (30.0, 3.0, auto.ACTION_SKIP),
    (11.9, 2.1, auto.ACTION_FETCH),
    (12.1, 2.1, auto.ACTION_SKIP),
])
def test_deadline_window_tightens_the_limit(env, hours_out, age_hours, expected):
    write_bootstrap(env.dir, age_seconds=age_hours * 3600,
                    deadlines=imminent_deadlines(hours_out))
    env.touch_projections()
    decision = env.plan()
    assert decision.action == expected, decision.reason
    assert decision.near_deadline == (hours_out <= 12.0)
    assert decision.max_age == (auto.PRE_DEADLINE_MAX_AGE if hours_out <= 12.0
                                else auto.DAILY_MAX_AGE)


def test_missing_projections_recompute_without_fetching(env):
    write_bootstrap(env.dir, age_seconds=600, deadlines=far_deadlines())
    decision = env.plan()
    assert decision.action == auto.ACTION_RECOMPUTE
    assert decision.works and not decision.fetches
    assert "missing" in decision.reason


def test_projections_older_than_the_data_recompute(env):
    write_bootstrap(env.dir, age_seconds=600, deadlines=far_deadlines())
    env.touch_projections(newer_than_bootstrap=False)
    decision = env.plan()
    assert decision.action == auto.ACTION_RECOMPUTE
    assert "older than the data" in decision.reason


def test_past_deadlines_are_ignored(env):
    """A finished gameweek's deadline must not look imminent forever."""
    deadlines = {1: NOW - timedelta(hours=1), 2: NOW + timedelta(days=6)}
    write_bootstrap(env.dir, age_seconds=3 * 3600, deadlines=deadlines)
    env.touch_projections(first=2, last=7)
    decision = env.plan()
    assert decision.next_gw == 2
    assert decision.action == auto.ACTION_SKIP


# ---------------------------------------------------------------------------
# Backoff and single flight
# ---------------------------------------------------------------------------


def test_backoff_schedule_doubles_and_caps():
    assert auto.backoff_seconds(0) == 0
    assert auto.backoff_seconds(1) == 15 * 60
    assert auto.backoff_seconds(2) == 30 * 60
    assert auto.backoff_seconds(3) == 3600
    assert auto.backoff_seconds(50) == auto.BACKOFF_MAX


def test_failures_push_the_next_attempt_out(env):
    write_bootstrap(env.dir, age_seconds=48 * 3600, deadlines=far_deadlines())
    env.touch_projections()
    assert env.plan().action == auto.ACTION_FETCH

    state = auto.record_failure(NOW, "FPLError: connection refused", env.state_path)
    assert state["consecutive_failures"] == 1

    # Stale data is now a reason to wait, not a reason to hammer the API.
    decision = env.plan(now=NOW + timedelta(minutes=5))
    assert decision.action == auto.ACTION_SKIP
    assert "backing off" in decision.reason
    assert decision.consecutive_failures == 1

    # ...but only until the backoff expires.
    assert env.plan(now=NOW + timedelta(minutes=20)).action == auto.ACTION_FETCH

    # A second failure doubles it.
    auto.record_failure(NOW, "FPLError: connection refused", env.state_path)
    assert env.plan(now=NOW + timedelta(minutes=20)).action == auto.ACTION_SKIP


def test_force_overrides_freshness_and_backoff(env):
    write_bootstrap(env.dir, age_seconds=60, deadlines=far_deadlines())
    env.touch_projections()
    auto.record_failure(NOW, "boom", env.state_path)
    assert env.plan().action == auto.ACTION_SKIP
    forced = env.plan(force=True)
    assert forced.action == auto.ACTION_FETCH
    assert forced.reason == "--force"


def test_success_clears_the_failure_streak(env):
    auto.record_failure(NOW, "boom", env.state_path)
    auto.record_failure(NOW, "boom", env.state_path)
    state = auto.record_success(NOW, {"action": "fetch", "seconds": 1.0}, env.state_path)
    assert state["consecutive_failures"] == 0
    assert state["retry_after"] is None
    assert auto.read_state(env.state_path)["last_success"] == NOW.isoformat()


def test_corrupt_state_file_reads_as_empty(tmp_path):
    path = str(tmp_path / "autorefresh.json")
    with open(path, "w") as fh:
        fh.write("{not json")
    assert auto.read_state(path) == {}


def test_single_flight_excludes_a_second_holder(tmp_path):
    path = str(tmp_path / "autorefresh.lock")
    first, second = auto.SingleFlight(path), auto.SingleFlight(path)
    assert first.acquire() is True
    assert second.acquire() is False
    assert "pid %d" % os.getpid() in second.holder
    first.release()
    assert second.acquire() is True
    second.release()


# ---------------------------------------------------------------------------
# Contracts with the rest of the system
# ---------------------------------------------------------------------------


def test_horizon_matches_what_the_server_projects(env):
    """The warmed file has to be the file ``Store._build`` looks for."""
    from gaffer.core import scoring

    write_bootstrap(env.dir, age_seconds=60, deadlines=far_deadlines())
    events = auto.cached_events(env.cache)
    gws = auto.horizon_gws(events, env.cfg, NOW)
    horizon = env.cfg.model.default_horizon
    assert gws[0] == 1
    assert gws[-1] == min(scoring.TOTAL_GWS, gws[0] + horizon - 1)
    assert gws == list(range(gws[0], gws[-1] + 1))


def test_autorefresh_never_pulls_in_the_backtest():
    """~170s of season walking must not be reachable from a timed job."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import gaffer.ops.autorefresh; "
         "print([m for m in sys.modules if m.startswith('gaffer.backtest')])"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_human_seconds_reads_like_a_human():
    assert auto.human_seconds(None) == "-"
    assert auto.human_seconds(45) == "45s"
    assert auto.human_seconds(600) == "10m"
    assert auto.human_seconds(3 * 3600) == "3.0h"
    assert auto.human_seconds(3 * 86400) == "3.0d"
