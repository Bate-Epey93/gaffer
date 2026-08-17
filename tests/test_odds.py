"""Bookmaker odds: parsing, budget discipline, and failing softly.

The risk with this feature is not that it breaks loudly — it is that it breaks
quietly. A club name the map does not recognise, or a quota that ran out three
weeks ago, would leave the model silently back on its own fitted ratings with
nothing in the output to say so. So most of what is asserted here is about the
failure paths reporting themselves.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gaffer.data import odds as odds_mod


# --- fixtures ---------------------------------------------------------------

def event(home, away, h=2.0, d=3.4, a=3.8, over=1.9, under=1.95):
    """One event in The Odds API's shape."""
    outcomes_h2h = [
        {"name": home, "price": h},
        {"name": away, "price": a},
        {"name": "Draw", "price": d},
    ]
    markets = [{"key": "h2h", "outcomes": outcomes_h2h}]
    if over and under:
        markets.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": over, "point": 2.5},
            {"name": "Under", "price": under, "point": 2.5},
        ]})
    return {
        "home_team": home, "away_team": away,
        "commence_time": "2026-08-21T19:00:00Z",
        "bookmakers": [{"key": "bk", "title": "Book", "markets": markets}],
    }


class FakeTeam:
    def __init__(self, tid, short):
        self.id = tid
        self.short_name = short


class FakeFixture:
    def __init__(self, fid, home, away, finished=False):
        self.id = fid
        self.team_h = home
        self.team_a = away
        self.finished = finished
        self.kickoff_time = "2026-08-21T19:00:00Z"


class FakeState:
    def __init__(self):
        self.teams = {1: FakeTeam(1, "ARS"), 7: FakeTeam(7, "COV"),
                      15: FakeTeam(15, "MCI"), 3: FakeTeam(3, "BOU")}
        self.fixtures = [FakeFixture(1, 1, 7), FakeFixture(2, 15, 3)]


class FakeModel:
    def __init__(self, accept=True):
        self.calls = []
        self.use_odds = False
        self._accept = accept

    def set_odds(self, fid, h, d, a, over=None, under=None):
        self.calls.append({"fixture": fid, "home": h, "draw": d, "away": a,
                           "over25": over, "under25": under})
        return self._accept


# --- price extraction -------------------------------------------------------

def test_best_prices_reads_h2h_and_the_2_5_line():
    prices = odds_mod._best_prices(event("Arsenal", "Coventry City"))
    assert prices["home"] == 2.0 and prices["draw"] == 3.4 and prices["away"] == 3.8
    assert prices["over25"] == 1.9 and prices["under25"] == 1.95


def test_best_prices_takes_the_best_across_bookmakers():
    ev = event("Arsenal", "Coventry City", h=2.0)
    ev["bookmakers"].append({"key": "b2", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Arsenal", "price": 2.4},
        {"name": "Coventry City", "price": 3.1},
        {"name": "Draw", "price": 3.2},
    ]}]})
    prices = odds_mod._best_prices(ev)
    assert prices["home"] == 2.4, "should take the best available home price"


def test_other_total_lines_are_ignored():
    """The conversion expects the 2.5 line; 3.5 would silently mean something else."""
    ev = event("Arsenal", "Coventry City", over=None, under=None)
    ev["bookmakers"][0]["markets"].append({"key": "totals", "outcomes": [
        {"name": "Over", "price": 3.0, "point": 3.5},
        {"name": "Under", "price": 1.4, "point": 3.5},
    ]})
    prices = odds_mod._best_prices(ev)
    assert "over25" not in prices and "under25" not in prices


def test_an_event_without_a_full_1x2_is_rejected():
    ev = event("Arsenal", "Coventry City")
    ev["bookmakers"][0]["markets"][0]["outcomes"] = [{"name": "Arsenal", "price": 2.0}]
    assert odds_mod._best_prices(ev) is None


# --- applying to the model --------------------------------------------------

def test_prices_reach_the_model_and_enable_the_market_path():
    model, state = FakeModel(), FakeState()
    report = odds_mod.apply_to_model(model, state, {
        "events": [event("Arsenal", "Coventry City"),
                   event("Manchester City", "AFC Bournemouth")]})
    assert report["applied"] == 2
    assert model.use_odds is True, "the market path must be switched on once priced"
    assert {c["fixture"] for c in model.calls} == {1, 2}


def test_finished_fixtures_are_not_priced():
    model, state = FakeModel(), FakeState()
    state.fixtures[0].finished = True
    report = odds_mod.apply_to_model(model, state, {"events": [event("Arsenal", "Coventry City")]})
    assert report["applied"] == 0
    assert model.use_odds is False


def test_an_unknown_club_name_is_reported_not_guessed():
    """A silently unmatched club leaves that fixture on the fitted rating while
    its neighbours use the market — an inconsistency nobody would spot."""
    model, state = FakeModel(), FakeState()
    report = odds_mod.apply_to_model(model, state, {
        "events": [event("Arsenal FC Reserves", "Coventry City")]})
    assert "Arsenal FC Reserves" in report["unmatched_names"]
    assert report["applied"] == 0


def test_the_market_path_stays_off_when_the_model_rejects_every_price():
    model, state = FakeModel(accept=False), FakeState()
    report = odds_mod.apply_to_model(model, state, {"events": [event("Arsenal", "Coventry City")]})
    assert report["applied"] == 0
    assert model.use_odds is False


def test_no_events_is_a_clean_no_op():
    model, state = FakeModel(), FakeState()
    report = odds_mod.apply_to_model(model, state, {"events": []})
    assert report["applied"] == 0 and model.use_odds is False


def test_every_club_in_the_current_league_is_mapped():
    """The map must cover the 20 clubs actually playing, or fixtures drop out."""
    shorts = set(odds_mod.ODDS_NAME_TO_SHORT.values())
    for short in ("ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "COV", "CRY", "EVE",
                  "FUL", "HUL", "IPS", "LEE", "LIV", "MCI", "MUN", "NEW", "NFO",
                  "TOT", "SUN"):
        assert short in shorts, "no odds-API name maps to %s" % short


# --- the budget -------------------------------------------------------------

def test_a_fresh_cache_is_reused_without_calling_the_api(tmp_path, monkeypatch):
    """The free tier is 500 calls a month against 720 hourly builds, so reuse is
    not an optimisation here — it is what keeps the feature working."""
    payload = {"events": [event("Arsenal", "Coventry City")],
               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "remaining": "412"}
    (tmp_path / odds_mod.CACHE_NAME).write_text(json.dumps(payload))

    def explode(*a, **k):
        raise AssertionError("the API must not be called while the cache is fresh")

    monkeypatch.setattr(odds_mod, "_fetch", explode)
    monkeypatch.setenv("ODDS_API_KEY", "x")
    result = odds_mod.load_odds(str(tmp_path), min_refresh_hours=6.0)
    assert result["from_cache"] is True
    assert len(result["events"]) == 1


def test_a_stale_cache_triggers_a_refetch(tmp_path, monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(hours=12)
    (tmp_path / odds_mod.CACHE_NAME).write_text(json.dumps(
        {"events": [], "fetched_at": old.isoformat(timespec="seconds")}))
    called = {}

    def fake_fetch(key, region, markets, timeout):
        called["yes"] = True
        return [event("Arsenal", "Coventry City")], {"remaining": "400", "used": "100"}

    monkeypatch.setattr(odds_mod, "_fetch", fake_fetch)
    result = odds_mod.load_odds(str(tmp_path), key="k", min_refresh_hours=6.0)
    assert called.get("yes") and result["from_cache"] is False
    assert "400" in result["status"]


def test_a_failed_fetch_falls_back_to_the_stale_cache(tmp_path, monkeypatch):
    """Yesterday's market is far closer to the truth than no market at all."""
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    (tmp_path / odds_mod.CACHE_NAME).write_text(json.dumps(
        {"events": [event("Arsenal", "Coventry City")],
         "fetched_at": old.isoformat(timespec="seconds")}))

    def boom(*a, **k):
        raise odds_mod.OddsUnavailable("odds quota exhausted (429).")

    monkeypatch.setattr(odds_mod, "_fetch", boom)
    result = odds_mod.load_odds(str(tmp_path), key="k", min_refresh_hours=1.0)
    assert len(result["events"]) == 1
    assert "quota" in result["status"] and "stale cache" in result["status"]


def test_no_key_never_raises_and_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    result = odds_mod.load_odds(str(tmp_path))
    assert result["events"] == []
    assert "ODDS_API_KEY" in result["status"]


def test_a_missing_cache_directory_is_harmless(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    result = odds_mod.load_odds("/nonexistent/path/that/does/not/exist")
    assert result["events"] == []
