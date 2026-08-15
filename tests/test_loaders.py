"""Data loader invariants for the 2026/27 season.

The load runs entirely from ``data/cache`` through an ``FPLClient`` whose
``_fetch`` raises, so a passing run is also proof that the loader needs no
network once the cache is warm.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import pytest

from gaffer.core import scoring
from gaffer.data import loaders


# ---------------------------------------------------------------------------
# The league
# ---------------------------------------------------------------------------


def test_there_are_twenty_teams(game_state):
    assert len(game_state.teams) == 20
    assert sorted(game_state.teams) == list(range(1, 21))
    assert len({t.short_name for t in game_state.teams.values()}) == 20
    assert len({t.code for t in game_state.teams.values()}) == 20


def test_there_are_380_fixtures(game_state):
    assert len(game_state.fixtures) == 380
    assert len({f.id for f in game_state.fixtures}) == 380


def test_every_fixture_maps_to_two_distinct_known_teams(game_state):
    known = set(game_state.teams)
    for fixture in game_state.fixtures:
        assert fixture.team_h in known, "fixture %d home team %d" % (fixture.id, fixture.team_h)
        assert fixture.team_a in known, "fixture %d away team %d" % (fixture.id, fixture.team_a)
        assert fixture.team_h != fixture.team_a, "fixture %d is a team playing itself" % fixture.id


def test_every_team_plays_38_matches_19_home_and_19_away(game_state):
    home, away = Counter(), Counter()
    for fixture in game_state.fixtures:
        home[fixture.team_h] += 1
        away[fixture.team_a] += 1
    for team_id in game_state.teams:
        assert home[team_id] == 19, game_state.short_name(team_id)
        assert away[team_id] == 19, game_state.short_name(team_id)


def test_every_pair_of_teams_meets_exactly_once_each_way(game_state):
    pairs = Counter((f.team_h, f.team_a) for f in game_state.fixtures)
    assert set(pairs.values()) == {1}
    assert len(pairs) == 380


def test_fixtures_are_scheduled_across_38_gameweeks(game_state):
    scheduled = [f for f in game_state.fixtures if f.gw is not None]
    assert len(scheduled) == 380, "%d fixtures have no gameweek" % (380 - len(scheduled))
    assert sorted({int(f.gw) for f in scheduled}) == list(range(1, 39))


def test_fixture_helpers_agree_with_the_raw_fields(game_state):
    for fixture in game_state.fixtures[:50]:
        assert fixture.opponent_of(fixture.team_h) == fixture.team_a
        assert fixture.opponent_of(fixture.team_a) == fixture.team_h
        assert fixture.is_home_for(fixture.team_h) is True
        assert fixture.is_home_for(fixture.team_a) is False


# ---------------------------------------------------------------------------
# The players
# ---------------------------------------------------------------------------


def test_the_player_pool_is_the_2026_27_one(game_state):
    assert len(game_state.players) == 587
    positions = Counter(p.position for p in game_state.players.values())
    assert set(positions) == set(scoring.POSITIONS)
    # There must be enough of every position to build a legal 15 many times over.
    for position, need in scoring.SQUAD_SELECT.items():
        assert positions[position] > need * 10, positions


def test_every_player_belongs_to_a_known_team_and_a_real_position(game_state):
    known = set(game_state.teams)
    for pid, player in game_state.players.items():
        assert player.id == pid
        assert player.team_id in known, "player %d team %d" % (pid, player.team_id)
        assert player.position in scoring.POSITIONS
        assert player.now_cost > 0
        assert player.price == player.now_cost / 10.0


def test_every_club_can_field_a_full_squad(game_state):
    by_team = {}
    for player in game_state.players.values():
        by_team.setdefault(player.team_id, Counter())[player.position] += 1
    assert len(by_team) == 20
    for team_id, counts in by_team.items():
        for position in scoring.POSITIONS:
            assert counts[position] >= 2, (game_state.short_name(team_id), counts)


def test_prices_are_stored_in_tenths_of_a_million(game_state):
    costs = [p.now_cost for p in game_state.players.values()]
    assert all(isinstance(c, int) for c in costs)
    assert min(costs) >= 35  # nobody is cheaper than 3.5m
    assert max(costs) <= 200


# ---------------------------------------------------------------------------
# Season state
# ---------------------------------------------------------------------------


def test_the_season_has_not_started(game_state):
    assert game_state.season == "2026-27"
    assert game_state.current_gw == 1
    assert game_state.finished_gws == []
    assert game_state.elements_are_prior_season is True
    assert len(game_state.events) == 38


def test_bootstrap_element_totals_carry_last_season(game_state):
    """Pre-season the API's element rows are LAST season's totals, not zeros."""
    minutes = [p.minutes for p in game_state.players.values()]
    assert max(minutes) > 2000, "no player has prior-season minutes"
    assert game_state.elements_are_prior_season is True


def test_exactly_three_promoted_clubs_are_detected(game_state):
    shorts = sorted(game_state.teams[t].short_name for t in game_state.promoted_team_ids)
    assert shorts == ["COV", "HUL", "IPS"]
    for team_id in game_state.promoted_team_ids:
        assert game_state.teams[team_id].promoted is True


def test_the_loader_reported_no_data_warnings(game_state):
    assert game_state.data_warnings == [], game_state.data_warnings


def test_gameweek_deadlines_are_in_order_and_gw1_is_the_known_one(game_state):
    deadlines = [game_state.deadline(gw) for gw in range(1, 39)]
    assert all(d for d in deadlines)
    assert deadlines == sorted(deadlines)
    assert deadlines[0].startswith("2026-08-21T17:30")


def test_gws_remaining_covers_the_whole_season_pre_season(game_state):
    assert game_state.gws_remaining() == list(range(1, 39))


# ---------------------------------------------------------------------------
# resolve_current_gw
# ---------------------------------------------------------------------------


def _event(gw, deadline, finished=False):
    return {"id": gw, "deadline_time": deadline, "finished": finished}


def test_resolve_current_gw_picks_the_first_deadline_still_ahead():
    events = [
        _event(1, "2026-08-21T17:30:00Z", finished=True),
        _event(2, "2026-08-28T17:30:00Z", finished=False),
        _event(3, "2026-09-04T17:30:00Z", finished=False),
    ]
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    assert loaders.resolve_current_gw(events, now) == 2


def test_resolve_current_gw_skips_a_gameweek_whose_deadline_has_passed():
    """A gameweek in progress is locked; the planning target is the next one."""
    events = [
        _event(1, "2026-08-21T17:30:00Z", finished=False),
        _event(2, "2026-08-28T17:30:00Z", finished=False),
    ]
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)  # GW1 kicked off yesterday
    assert loaders.resolve_current_gw(events, now) == 2


def test_resolve_current_gw_falls_back_to_the_first_unfinished_gameweek():
    events = [
        _event(1, "2026-08-21T17:30:00Z", finished=True),
        _event(2, "2026-08-28T17:30:00Z", finished=False),
    ]
    now = datetime(2026, 8, 29, tzinfo=timezone.utc)  # every deadline gone
    assert loaders.resolve_current_gw(events, now) == 2


def test_resolve_current_gw_returns_the_last_gameweek_once_the_season_ends():
    events = [
        _event(37, "2027-05-10T17:30:00Z", finished=True),
        _event(38, "2027-05-17T17:30:00Z", finished=True),
    ]
    now = datetime(2027, 6, 1, tzinfo=timezone.utc)
    assert loaders.resolve_current_gw(events, now) == 38


def test_resolve_current_gw_ignores_event_order_in_the_payload():
    events = [
        _event(3, "2026-09-04T17:30:00Z"),
        _event(1, "2026-08-21T17:30:00Z"),
        _event(2, "2026-08-28T17:30:00Z"),
    ]
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert loaders.resolve_current_gw(events, now) == 1


# ---------------------------------------------------------------------------
# Fixture views
# ---------------------------------------------------------------------------


def test_fixtures_by_gw_has_an_entry_for_every_gameweek(game_state):
    by_gw = loaders.fixtures_by_gw(game_state)
    assert sorted(by_gw) == list(range(1, 39))
    assert sum(len(v) for v in by_gw.values()) == 380
    for gw, fixtures in by_gw.items():
        assert all(int(f.gw) == gw for f in fixtures)


def test_team_fixture_counts_cover_every_team_and_gameweek(game_state):
    counts = loaders.team_fixture_counts(game_state, list(range(1, 39)))
    assert set(counts) == set(game_state.teams)
    for team_id, per_gw in counts.items():
        assert sorted(per_gw) == list(range(1, 39))
        assert sum(per_gw.values()) == 38, game_state.short_name(team_id)


def test_the_pre_season_schedule_has_no_blanks_or_doubles(game_state):
    """Before the cups start moving fixtures every club plays exactly once a week."""
    counts = loaders.team_fixture_counts(game_state, list(range(1, 39)))
    seen = {n for per_gw in counts.values() for n in per_gw.values()}
    assert seen == {1}
    assert loaders.unscheduled_fixtures(game_state) == []


def test_team_fixtures_returns_one_entry_per_match(game_state):
    for team_id in list(game_state.teams)[:5]:
        window = loaders.team_fixtures(game_state, team_id, [1, 2, 3])
        assert len(window) == 3
        assert [int(f.gw) for f in window] == [1, 2, 3]
        assert all(team_id in (f.team_h, f.team_a) for f in window)


# ---------------------------------------------------------------------------
# players_df
# ---------------------------------------------------------------------------


def test_players_df_is_one_row_per_player_with_numeric_dtypes(game_state):
    df = game_state.players_df()
    assert len(df) == len(game_state.players)
    assert list(df.index) == sorted(game_state.players)
    for column in ("form", "selected_by_percent", "expected_goals", "ep_next",
                   "price_change_percent", "minutes", "total_points"):
        assert pd.api.types.is_numeric_dtype(df[column]), column
    assert set(df["pos"]) <= {"GKP", "DEF", "MID", "FWD"}
    assert df["price"].equals(df["now_cost"] / 10.0)
    assert df["promoted"].sum() > 0


def test_players_df_team_labels_match_the_team_table(game_state):
    df = game_state.players_df()
    assert df["team_short"].isna().sum() == 0
    for team_id, short in df[["team_id", "team_short"]].drop_duplicates().values:
        assert game_state.teams[int(team_id)].short_name == short


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_current_season_history_is_empty_but_correctly_shaped(game_state):
    """Pre-season nobody has played; ``df["minutes"].sum()`` must still work."""
    for pid in sorted(game_state.players)[:20]:
        df = game_state.history(pid)
        assert len(df) == 0
        assert "minutes" in df.columns
        assert df["minutes"].sum() == 0


def test_history_of_an_unknown_player_is_an_empty_frame(game_state):
    df = game_state.history(-1)
    assert len(df) == 0
    assert "minutes" in df.columns


def test_prior_season_rows_are_numeric_and_named_with_a_slash(game_state):
    found = 0
    for pid in sorted(game_state.players):
        row = game_state.prior_season_row(pid)
        if row is None:
            continue  # new signings and promoted-club players: a real signal
        found += 1
        assert row["season_name"] == "2025/26"
        assert isinstance(row["minutes"], (int, float))
        assert isinstance(row["expected_goals"], float)
        if found >= 50:
            break
    assert found >= 50, "only %d players had a 2025/26 row" % found


def test_promoted_club_players_have_no_prior_premier_league_season(game_state):
    """The three promoted clubs are new: this is the cold-start the model faces."""
    promoted = set(game_state.promoted_team_ids)
    assert promoted
    without = 0
    total = 0
    for pid, player in game_state.players.items():
        if player.team_id not in promoted:
            continue
        total += 1
        if game_state.prior_season_row(pid) is None:
            without += 1
    assert total > 0
    assert without / total > 0.5, "%d of %d promoted-club players had a PL season" % (
        total - without, total
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_reports_the_headline_numbers(game_state):
    summary = game_state.summary()
    assert "20 teams" in summary
    assert "587 players" in summary
    assert "380 fixtures" in summary
    assert "current_gw=1" in summary


# ---------------------------------------------------------------------------
# The same invariants on the synthetic league, without any cache
# ---------------------------------------------------------------------------


def test_synthetic_league_has_20_teams_and_380_fixtures(synthetic_state):
    assert len(synthetic_state.teams) == 20
    assert len(synthetic_state.fixtures) == 380
    known = set(synthetic_state.teams)
    for fixture in synthetic_state.fixtures:
        assert fixture.team_h in known and fixture.team_a in known
        assert fixture.team_h != fixture.team_a
    counts = Counter()
    for fixture in synthetic_state.fixtures:
        counts[fixture.team_h] += 1
        counts[fixture.team_a] += 1
    assert set(counts.values()) == {38}


def test_synthetic_league_players_map_to_known_teams(synthetic_state):
    known = set(synthetic_state.teams)
    assert len(synthetic_state.players) > 0
    for pid, player in synthetic_state.players.items():
        assert player.team_id in known
        assert player.position in scoring.POSITIONS
