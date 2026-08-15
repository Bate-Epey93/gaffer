"""The offline guarantee, and the disk cache that provides it.

If any of these fail the rest of the suite is no longer trustworthy: a test that
silently fetches live data is a test that passes on Tuesday and fails on
Wednesday for reasons that have nothing to do with the code.
"""
from __future__ import annotations

import json
import os
import time

import pytest
import requests

from gaffer.core.config import Config
from gaffer.data.cache import DEFAULT_TTL, TTL_FOREVER, Cache, safe_key
from gaffer.data.fpl_api import FPLClient
from gaffer.data.loaders import load_game_state

from tests.conftest import FrozenCache, OfflineFPLClient


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------


def test_the_offline_client_refuses_to_fetch(config):
    client = OfflineFPLClient(config, FrozenCache(default_ttl=TTL_FOREVER))
    with pytest.raises(AssertionError, match="tried to reach the network"):
        client._fetch("https://fantasy.premierleague.com/api/bootstrap-static/")


def test_a_whole_game_state_loads_with_requests_blocked(monkeypatch, config, game_state):
    """Not just "the cache was warm" — HTTP itself is amputated for this load."""

    def blocked(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("the loader issued an HTTP request")

    monkeypatch.setattr(requests.Session, "get", blocked)
    monkeypatch.setattr(requests.Session, "request", blocked)

    client = OfflineFPLClient(config, FrozenCache(default_ttl=TTL_FOREVER))
    state = load_game_state(config, client=client, progress=False)

    assert len(state.teams) == 20
    assert len(state.fixtures) == 380
    assert len(state.players) == len(game_state.players)
    assert state.data_warnings == []


def test_the_real_client_would_have_gone_online(config):
    """Sanity: the offline client is not a no-op wrapper around nothing."""
    assert OfflineFPLClient._fetch is not FPLClient._fetch


# ---------------------------------------------------------------------------
# FrozenCache
# ---------------------------------------------------------------------------


def test_frozen_cache_serves_an_entry_a_plain_cache_would_call_stale(tmp_path):
    plain = Cache(str(tmp_path), default_ttl=DEFAULT_TTL)
    plain.set("bootstrap", {"elements": [1, 2, 3]})

    # Backdate the stamp by a day: well past the 300s live TTL.
    path = plain.path_for("bootstrap")
    with open(path) as fh:
        payload = json.load(fh)
    payload["fetched_at"] = "2020-01-01T00:00:00+00:00"
    with open(path, "w") as fh:
        json.dump(payload, fh)

    assert plain.get("bootstrap") is None
    assert FrozenCache(str(tmp_path)).get("bootstrap") == {"elements": [1, 2, 3]}


def test_frozen_cache_still_returns_none_for_a_key_that_was_never_written(tmp_path):
    assert FrozenCache(str(tmp_path)).get("no_such_key") is None


# ---------------------------------------------------------------------------
# Cache mechanics
# ---------------------------------------------------------------------------


def test_cache_round_trips_json(tmp_path):
    cache = Cache(str(tmp_path))
    value = {"a": [1, 2, {"b": None}], "c": 1.5}
    cache.set("thing", value)
    assert cache.get("thing", ttl=TTL_FOREVER) == value
    assert cache.has("thing") is True


def test_cache_expires_an_entry_once_the_ttl_passes(tmp_path):
    cache = Cache(str(tmp_path), default_ttl=1)
    cache.set("thing", 42)
    assert cache.get("thing") == 42
    assert cache.is_fresh("thing") is True
    time.sleep(1.05)
    assert cache.get("thing") is None
    assert cache.is_fresh("thing") is False
    # The file is still there, and a forever read still finds it.
    assert cache.has("thing") is True
    assert cache.get("thing", ttl=TTL_FOREVER) == 42


def test_cache_keys_with_slashes_become_safe_filenames(tmp_path):
    cache = Cache(str(tmp_path))
    cache.set("element-summary/12/", {"history": []})
    name = os.path.basename(cache.path_for("element-summary/12/"))
    assert "/" not in name
    assert name.endswith(".json")
    assert cache.get("element-summary/12/", ttl=TTL_FOREVER) == {"history": []}


def test_safe_key_shortens_and_disambiguates_long_keys():
    a, b = safe_key("x" * 400), safe_key("y" + "x" * 399)
    assert len(a) <= 120 and len(b) <= 120
    assert a != b


def test_cache_ignores_a_corrupt_entry_rather_than_raising(tmp_path):
    cache = Cache(str(tmp_path))
    cache.set("thing", 1)
    with open(cache.path_for("thing"), "w") as fh:
        fh.write("{not json")
    assert cache.get("thing", ttl=TTL_FOREVER) is None


def test_cache_ignores_an_entry_with_no_data_key(tmp_path):
    cache = Cache(str(tmp_path))
    with open(cache.path_for("thing"), "w") as fh:
        json.dump({"fetched_at": "2026-01-01T00:00:00+00:00"}, fh)
    assert cache.get("thing", ttl=TTL_FOREVER) is None


def test_cache_set_leaves_no_temp_files_behind(tmp_path):
    cache = Cache(str(tmp_path))
    for i in range(5):
        cache.set("thing_%d" % i, i)
    leftovers = [n for n in os.listdir(str(tmp_path)) if n.startswith(".tmp-")]
    assert leftovers == []


def test_cache_keys_and_clear(tmp_path):
    cache = Cache(str(tmp_path))
    cache.set("live_1", 1)
    cache.set("live_2", 2)
    cache.set("bootstrap", 3)
    assert sorted(cache.keys("live_*")) == ["live_1", "live_2"]
    assert cache.clear("live_*") == 2
    assert cache.keys() == ["bootstrap"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_2026_27_setup():
    cfg = Config()
    assert cfg.season == "2026-27"
    assert cfg.optimizer.solver in ("auto", "highs", "cbc")
    assert 0.0 < cfg.optimizer.decay <= 1.0
    assert cfg.optimizer.horizon >= 1
    assert len(cfg.optimizer.bench_weights) == 4
    assert all(0.0 <= w <= 1.0 for w in cfg.optimizer.bench_weights)
    # Autosub insurance must be worth less than a starting place, and the
    # further down the bench the less it is worth.
    assert cfg.optimizer.bench_weights == sorted(cfg.optimizer.bench_weights, reverse=True)


def test_config_from_dict_overrides_only_what_it_is_given():
    cfg = Config.from_dict({"optimizer": {"horizon": 3}, "log_level": "DEBUG"})
    assert cfg.optimizer.horizon == 3
    assert cfg.log_level == "DEBUG"
    assert cfg.optimizer.decay == Config().optimizer.decay  # untouched
    assert cfg.model.league_avg_goals_per_team == Config().model.league_avg_goals_per_team
