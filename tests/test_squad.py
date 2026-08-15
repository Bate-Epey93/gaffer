"""Squad legality, asserted against a real MILP solve.

Legality is checked here *independently* of ``squad.squad_problems``: the counts
are recomputed straight off ``state.players`` so that a bug shared between the
solver and its own checker cannot hide. ``squad_problems`` is then tested
separately against deliberately broken squads, so it is known not to be vacuous.

Two solves run: one on the real 2026/27 game with the cached projections, and
one on the synthetic league, which needs no cache at all.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

import pytest

from gaffer.core import scoring
from gaffer.optimize import squad as sq

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Independent legality checks
# ---------------------------------------------------------------------------


def assert_squad_is_legal(squad_ids: Sequence[int], state: Any, budget: int = 1000) -> None:
    ids = list(squad_ids)
    assert len(ids) == 15, "squad has %d players" % len(ids)
    assert len(set(ids)) == 15, "squad contains a duplicate"

    positions = Counter(state.players[p].position for p in ids)
    assert positions[scoring.GKP] == 2, positions
    assert positions[scoring.DEF] == 5, positions
    assert positions[scoring.MID] == 5, positions
    assert positions[scoring.FWD] == 3, positions

    clubs = Counter(state.players[p].team_id for p in ids)
    assert max(clubs.values()) <= 3, clubs.most_common(3)

    cost = sum(state.players[p].now_cost for p in ids)
    assert cost <= budget, "squad costs %d tenths of a budget of %d" % (cost, budget)


def assert_lineup_is_legal(lineup: Sequence[int], bench: Sequence[int],
                           squad_ids: Sequence[int], state: Any) -> None:
    assert len(lineup) == 11
    assert len(bench) == 4
    assert not set(lineup) & set(bench)
    assert set(lineup) | set(bench) == set(squad_ids)

    counts = Counter(state.players[p].position for p in lineup)
    assert counts[scoring.GKP] == 1, counts
    assert 3 <= counts[scoring.DEF] <= 5, counts
    assert 2 <= counts[scoring.MID] <= 5, counts
    assert 1 <= counts[scoring.FWD] <= 3, counts
    assert counts[scoring.DEF] + counts[scoring.MID] + counts[scoring.FWD] == 10

    # The reserve keeper occupies bench slot 0 and only ever replaces the keeper.
    assert state.players[bench[0]].position == scoring.GKP
    assert sum(1 for p in bench if state.players[p].position == scoring.GKP) == 1


# ---------------------------------------------------------------------------
# Formations
# ---------------------------------------------------------------------------


def test_valid_formations_are_the_eight_legal_shapes():
    assert sorted(sq.valid_formations()) == sorted(
        [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
         (5, 2, 3), (5, 3, 2), (5, 4, 1)]
    )
    assert len(sq.FORMATIONS) == 8
    for d, m, f in sq.FORMATIONS:
        assert d + m + f == 10
        assert scoring.lineup_is_legal([scoring.GKP] + [scoring.DEF] * d
                                       + [scoring.MID] * m + [scoring.FWD] * f)


def test_formation_name():
    positions = [scoring.GKP] + [scoring.DEF] * 3 + [scoring.MID] * 4 + [scoring.FWD] * 3
    assert sq.formation_name(positions) == "3-4-3"


# ---------------------------------------------------------------------------
# The real solve
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_decision(projections, game_state, config):
    return sq.pick_initial_squad(projections, game_state, config, [1, 2, 3])


def test_real_optimised_squad_is_legal(real_decision, game_state):
    assert_squad_is_legal(real_decision.squad, game_state)


def test_real_optimised_lineup_is_a_legal_xi(real_decision, game_state):
    assert_lineup_is_legal(
        real_decision.lineup, real_decision.bench, real_decision.squad, game_state
    )


def test_real_squad_spends_the_budget_and_reports_the_bank(real_decision, game_state):
    cost = sum(game_state.players[p].now_cost for p in real_decision.squad)
    assert real_decision.bank_after == scoring.BUDGET_TENTHS - cost
    assert real_decision.bank_after >= 0
    # An optimizer that leaves more than 5.0m unspent has not tried.
    assert real_decision.bank_after <= 50


def test_real_captain_and_vice_are_distinct_starters(real_decision, projections, game_state):
    assert real_decision.captain in real_decision.lineup
    assert real_decision.vice_captain in real_decision.lineup
    assert real_decision.captain != real_decision.vice_captain
    view = sq.build_views(projections, game_state, [real_decision.gw])[real_decision.gw]
    cap_xp = view.get_xp(real_decision.captain)
    assert all(view.get_xp(p) <= cap_xp + 1e-9 for p in real_decision.lineup)


def test_real_bench_is_ordered_by_expected_points_times_appearance(
    real_decision, projections, game_state
):
    view = sq.build_views(projections, game_state, [real_decision.gw])[real_decision.gw]
    keys = [view.get_xp(p) * view.get_appear(p) for p in real_decision.bench[1:]]
    assert keys == sorted(keys, reverse=True)


def test_real_solve_is_reproducible_across_seeds(projections, game_state, config):
    """A MILP optimum is unique in value; two seeds must agree on the objective."""
    views = sq.build_views(projections, game_state, [1, 2, 3])
    a = sq.solve_squad_milp(projections, game_state, config, [1, 2, 3], views=views, seed=7)
    b = sq.solve_squad_milp(projections, game_state, config, [1, 2, 3], views=views, seed=4242)
    assert a.objective == pytest.approx(b.objective, abs=2e-3)
    assert a.squad == b.squad
    assert a.report.status == "Optimal"


def test_real_solve_respects_a_reduced_budget(projections, game_state, config):
    budget = 950
    solution = sq.solve_squad_milp(
        projections, game_state, config, [1, 2, 3], budget=budget
    )
    assert_squad_is_legal(solution.squad, game_state, budget=budget)
    assert solution.cost <= budget
    assert solution.bank == budget - solution.cost


def test_real_solve_respects_locked_in_and_locked_out(
    projections, game_state, fresh_config, real_decision
):
    pool = sq.candidate_ids(projections, game_state, fresh_config, [1, 2, 3])
    owned = set(real_decision.squad)
    # Force in a forward the optimizer did not want, so the lock has to bind.
    lock_in = max(
        (p for p in pool if game_state.players[p].position == scoring.FWD and p not in owned),
        key=lambda p: (game_state.players[p].now_cost, p),
    )
    lock_out = real_decision.squad[0]
    fresh_config.optimizer.locked_in = [lock_in]
    fresh_config.optimizer.locked_out = [lock_out]

    decision = sq.pick_initial_squad(projections, game_state, fresh_config, [1, 2, 3])
    assert lock_in in decision.squad
    assert lock_out not in decision.squad
    assert_squad_is_legal(decision.squad, game_state)


def reweighted(projections, weight) -> Any:
    """A copy of ``projections`` with each player's xP passed through ``weight``.

    Fixture rows are shared, not copied: ``build_views`` only reads their ids,
    minutes and appearance probabilities, none of which change here.
    """
    from gaffer.core.types import PlayerGWProjection, ProjectionSet

    out = ProjectionSet(
        season=projections.season, generated_at=projections.generated_at,
        first_gw=projections.first_gw, last_gw=projections.last_gw,
    )
    for pid, per_gw in projections.projections.items():
        out.projections[pid] = {
            gw: PlayerGWProjection(
                player_id=pid, gw=gw, xp=weight(pid, gwp.xp), sd=gwp.sd,
                fixtures=list(gwp.fixtures),
            )
            for gw, gwp in per_gw.items()
        }
    return out


def test_the_three_per_club_limit_actually_binds(projections, game_state, config):
    """Make one club irresistible; the solver must still stop at three of them.

    Without this the club constraint is never tested, because on real
    projections the best 15 happens to be spread across enough clubs that
    deleting the constraint changes nothing.
    """
    target = Counter(
        game_state.players[p].team_id
        for p in sq.candidate_ids(projections, game_state, config, [1, 2, 3])
    ).most_common(1)[0][0]

    boosted = reweighted(
        projections,
        lambda pid, xp: xp + (10.0 if game_state.players[pid].team_id == target else 0.0),
    )
    solution = sq.solve_squad_milp(boosted, game_state, config, [1, 2, 3])

    clubs = Counter(game_state.players[p].team_id for p in solution.squad)
    assert clubs[target] == scoring.TEAM_LIMIT, (
        "constraint did not bind: %d players from %s"
        % (clubs[target], game_state.short_name(target))
    )
    assert max(clubs.values()) <= scoring.TEAM_LIMIT
    assert_squad_is_legal(solution.squad, game_state)


def test_the_xi_keeps_three_defenders_even_when_defenders_score_nothing(
    real_decision, projections, game_state
):
    """The 3-defender floor is a rule, not a preference the data happens to share."""
    worthless = reweighted(
        projections,
        lambda pid, xp: 0.0 if game_state.players[pid].position == scoring.DEF else xp,
    )
    lineup, bench, _, _ = sq.pick_lineup(
        list(real_decision.squad), worthless, 1, game_state
    )
    counts = Counter(game_state.players[p].position for p in lineup)
    assert counts[scoring.DEF] == scoring.SQUAD_MIN_PLAY[scoring.DEF] == 3
    assert_lineup_is_legal(lineup, bench, real_decision.squad, game_state)


def test_the_xi_keeps_a_forward_even_when_forwards_score_nothing(
    real_decision, projections, game_state
):
    worthless = reweighted(
        projections,
        lambda pid, xp: 0.0 if game_state.players[pid].position == scoring.FWD else xp,
    )
    lineup, bench, _, _ = sq.pick_lineup(
        list(real_decision.squad), worthless, 1, game_state
    )
    counts = Counter(game_state.players[p].position for p in lineup)
    assert counts[scoring.FWD] == scoring.SQUAD_MIN_PLAY[scoring.FWD] == 1
    assert_lineup_is_legal(lineup, bench, real_decision.squad, game_state)


def test_the_xi_never_fields_more_than_five_defenders(
    real_decision, projections, game_state
):
    dominant = reweighted(
        projections,
        lambda pid, xp: xp + (10.0 if game_state.players[pid].position == scoring.DEF else 0.0),
    )
    lineup, bench, _, _ = sq.pick_lineup(
        list(real_decision.squad), dominant, 1, game_state
    )
    counts = Counter(game_state.players[p].position for p in lineup)
    assert counts[scoring.DEF] == scoring.SQUAD_MAX_PLAY[scoring.DEF] == 5
    assert counts[scoring.GKP] == 1, "a second keeper cannot start"
    assert_lineup_is_legal(lineup, bench, real_decision.squad, game_state)


def test_the_squad_quota_binds_when_one_position_dominates(
    projections, game_state, config
):
    """Five brilliant midfielders is the limit, however good the sixth is."""
    dominant = reweighted(
        projections,
        lambda pid, xp: xp + (10.0 if game_state.players[pid].position == scoring.MID else 0.0),
    )
    solution = sq.solve_squad_milp(dominant, game_state, config, [1, 2, 3])
    counts = Counter(game_state.players[p].position for p in solution.squad)
    assert counts[scoring.MID] == 5
    assert_squad_is_legal(solution.squad, game_state)


def test_locking_a_player_in_and_out_at_once_is_refused(
    projections, game_state, fresh_config
):
    fresh_config.optimizer.locked_in = [1]
    fresh_config.optimizer.locked_out = [1]
    with pytest.raises(sq.OptimizeError):
        sq.candidate_ids(projections, game_state, fresh_config, [1, 2, 3])


def test_an_impossible_budget_raises_rather_than_returning_an_illegal_squad(
    projections, game_state, config
):
    """No legal 15 fits in 40.0m; the solver must refuse, not shave a player."""
    with pytest.raises(sq.OptimizeError):
        sq.solve_squad_milp(projections, game_state, config, [1, 2, 3], budget=400)


def test_solve_needs_at_least_one_gameweek(projections, game_state, config):
    with pytest.raises(sq.OptimizeError):
        sq.solve_squad_milp(projections, game_state, config, [])


def test_solve_refuses_gameweeks_it_has_no_projections_for(
    projections, game_state, config
):
    with pytest.raises(sq.OptimizeError):
        sq.solve_squad_milp(projections, game_state, config, [37, 38])


# ---------------------------------------------------------------------------
# The synthetic solve: same guarantees with no cache at all
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_decision(synthetic_projections, synthetic_state, config):
    return sq.pick_initial_squad(synthetic_projections, synthetic_state, config, [1, 2, 3])


def test_synthetic_optimised_squad_is_legal(synthetic_decision, synthetic_state):
    assert_squad_is_legal(synthetic_decision.squad, synthetic_state)
    assert_lineup_is_legal(
        synthetic_decision.lineup,
        synthetic_decision.bench,
        synthetic_decision.squad,
        synthetic_state,
    )


def test_synthetic_solve_beats_a_hand_built_legal_squad(
    synthetic_decision, synthetic_projections, synthetic_state, config
):
    """The optimum must be at least as good as the cheapest legal alternative."""
    view = sq.build_views(synthetic_projections, synthetic_state, [1])[1]
    by_pos: Dict[int, List[int]] = {p: [] for p in scoring.POSITIONS}
    for pid, player in sorted(synthetic_state.players.items()):
        by_pos[player.position].append(pid)
    cheap: List[int] = []
    clubs: Counter = Counter()
    for pos, need in sorted(scoring.SQUAD_SELECT.items()):
        picked = 0
        for pid in sorted(by_pos[pos], key=lambda p: synthetic_state.players[p].now_cost):
            team = synthetic_state.players[pid].team_id
            if clubs[team] >= scoring.TEAM_LIMIT:
                continue
            cheap.append(pid)
            clubs[team] += 1
            picked += 1
            if picked == need:
                break
        assert picked == need
    assert_squad_is_legal(cheap, synthetic_state)

    best = sq.expected_gw_points(synthetic_decision.squad, synthetic_projections, 1, synthetic_state)
    worst = sq.expected_gw_points(cheap, synthetic_projections, 1, synthetic_state)
    assert best > worst


# ---------------------------------------------------------------------------
# The legality checker itself
# ---------------------------------------------------------------------------


def test_squad_problems_is_silent_on_a_legal_squad(real_decision, game_state):
    assert sq.squad_problems(real_decision.squad, game_state) == []


def test_squad_problems_catches_a_missing_player(real_decision, game_state):
    problems = sq.squad_problems(real_decision.squad[:-1], game_state)
    assert any("14 players" in p for p in problems)


def test_squad_problems_catches_a_duplicate(real_decision, game_state):
    broken = list(real_decision.squad[:-1]) + [real_decision.squad[0]]
    problems = sq.squad_problems(broken, game_state)
    assert any("duplicate" in p for p in problems)


def test_squad_problems_catches_a_broken_quota(real_decision, game_state):
    keeper = next(p for p in real_decision.squad
                  if game_state.players[p].position == scoring.GKP)
    spare_def = next(
        p for p in game_state.players
        if game_state.players[p].position == scoring.DEF and p not in set(real_decision.squad)
    )
    broken = [p for p in real_decision.squad if p != keeper] + [spare_def]
    problems = sq.squad_problems(broken, game_state)
    assert any("GKP quota" in p for p in problems)
    assert any("DEF quota" in p for p in problems)


def test_squad_problems_catches_a_fourth_player_from_one_club(game_state):
    """Four defenders from the same club, everything else legal."""
    by_team: Dict[int, Dict[int, List[int]]] = {}
    for pid, player in sorted(game_state.players.items()):
        by_team.setdefault(player.team_id, {}).setdefault(player.position, []).append(pid)
    team = next(t for t, per in by_team.items() if len(per.get(scoring.DEF, [])) >= 4)
    four = by_team[team][scoring.DEF][:4]

    squad = list(four)
    clubs = Counter({team: 4})
    need = {scoring.GKP: 2, scoring.DEF: 1, scoring.MID: 5, scoring.FWD: 3}
    for pos, count in sorted(need.items()):
        picked = 0
        for pid, player in sorted(game_state.players.items()):
            if player.position != pos or pid in set(squad):
                continue
            if clubs[player.team_id] >= scoring.TEAM_LIMIT:
                continue
            squad.append(pid)
            clubs[player.team_id] += 1
            picked += 1
            if picked == count:
                break
        assert picked == count
    assert len(squad) == 15

    problems = sq.squad_problems(squad, game_state, budget=100000)
    assert any("limit is 3" in p for p in problems)


def test_squad_problems_catches_going_over_budget(real_decision, game_state):
    problems = sq.squad_problems(real_decision.squad, game_state, budget=500)
    assert any("budget is 500" in p for p in problems)


def test_lineup_problems_catches_an_illegal_formation(real_decision, game_state):
    """Swap a starting defender for the benched keeper: two keepers, 2 DEF."""
    lineup = list(real_decision.lineup)
    bench = list(real_decision.bench)
    starter_def = next(p for p in lineup if game_state.players[p].position == scoring.DEF)
    reserve_gk = bench[0]
    lineup[lineup.index(starter_def)] = reserve_gk
    bench[0] = starter_def

    problems = sq.lineup_problems(lineup, bench, real_decision.squad, game_state)
    assert any("illegal formation" in p for p in problems)
    assert any("bench slot 0 is not the reserve keeper" in p for p in problems)


def test_lineup_problems_catches_a_player_in_both_lists(real_decision, game_state):
    lineup = list(real_decision.lineup)
    bench = list(real_decision.bench)
    bench[1] = lineup[1]
    problems = sq.lineup_problems(lineup, bench, real_decision.squad, game_state)
    assert any("both the lineup and the bench" in p for p in problems)


def test_assert_legal_raises_on_a_broken_squad(real_decision, game_state):
    with pytest.raises(sq.OptimizeError):
        sq.assert_legal(
            real_decision.squad[:-1], real_decision.lineup, real_decision.bench, game_state
        )


def test_assert_legal_passes_on_the_real_decision(real_decision, game_state):
    sq.assert_legal(
        real_decision.squad, real_decision.lineup, real_decision.bench, game_state
    )


# ---------------------------------------------------------------------------
# Lineup mechanics
# ---------------------------------------------------------------------------


def test_pick_lineup_rejects_a_squad_that_is_not_fifteen(
    projections, game_state, real_decision
):
    with pytest.raises(sq.OptimizeError):
        sq.pick_lineup(list(real_decision.squad[:14]), projections, 1, game_state)


def test_bench_boost_adds_exactly_the_bench(real_decision, projections, game_state):
    view = sq.build_views(projections, game_state, [real_decision.gw])[real_decision.gw]
    plain = sq.lineup_expected_points(
        real_decision.lineup, real_decision.bench, real_decision.captain, view
    )
    boosted = sq.lineup_expected_points(
        real_decision.lineup, real_decision.bench, real_decision.captain, view, "bboost"
    )
    assert boosted - plain == pytest.approx(
        sum(view.get_xp(p) for p in real_decision.bench), abs=1e-9
    )


def test_triple_captain_adds_exactly_one_more_captain(real_decision, projections, game_state):
    view = sq.build_views(projections, game_state, [real_decision.gw])[real_decision.gw]
    plain = sq.lineup_expected_points(
        real_decision.lineup, real_decision.bench, real_decision.captain, view
    )
    tripled = sq.lineup_expected_points(
        real_decision.lineup, real_decision.bench, real_decision.captain, view, "3xc"
    )
    assert tripled - plain == pytest.approx(view.get_xp(real_decision.captain), abs=1e-9)


def test_captain_multiplier_by_chip():
    assert sq.captain_multiplier(None) == 2
    assert sq.captain_multiplier("bboost") == 2
    assert sq.captain_multiplier("3xc") == 3


def test_chips_never_change_the_legality_of_the_eleven(
    real_decision, projections, game_state
):
    for chip in (None, "bboost", "3xc", "freehit"):
        lineup, bench, captain, vice = sq.pick_lineup(
            list(real_decision.squad), projections, 1, game_state, chip=chip
        )
        assert_lineup_is_legal(lineup, bench, real_decision.squad, game_state)
        assert captain in lineup and vice in lineup


def test_choose_captain_prefers_a_vice_in_another_match_inside_tolerance():
    kickoffs = {1: ("A",), 2: ("A",), 3: ("B",), 4: ("B",)}
    view = sq.GWView(gw=1, xp={1: 9.0, 2: 6.0, 3: 5.8, 4: 3.0}, kickoffs=kickoffs)
    captain, vice = sq.choose_captain([1, 2, 3, 4], view)
    assert captain == 1
    assert vice == 3  # 0.2 behind, but kicks off in a different match

    far = sq.GWView(gw=1, xp={1: 9.0, 2: 6.0, 3: 5.0, 4: 3.0}, kickoffs=kickoffs)
    captain, vice = sq.choose_captain([1, 2, 3, 4], far)
    assert captain == 1
    assert vice == 2  # 1.0 behind is outside the tolerance; take the points


def test_best_lineup_benches_a_player_with_no_fixture(
    real_decision, projections, game_state
):
    """A blank gameweek must drop the blanked player out of the XI."""
    from gaffer.core.types import PlayerGWProjection, ProjectionSet

    edge = ProjectionSet(
        season=projections.season, generated_at=projections.generated_at,
        first_gw=1, last_gw=1,
    )
    for pid in real_decision.squad:
        src = projections.projections[pid][1]
        edge.projections[pid] = {
            1: PlayerGWProjection(player_id=pid, gw=1, xp=src.xp, sd=src.sd,
                                  fixtures=list(src.fixtures))
        }
    view = sq.build_views(projections, game_state, [1])[1]
    bench_positions = {
        game_state.players[p].position for p in real_decision.bench if view.get_xp(p) > 0
    }
    blanked = min(
        (p for p in real_decision.lineup
         if game_state.players[p].position in bench_positions),
        key=lambda p: (view.get_xp(p), p),
    )
    edge.projections[blanked][1].fixtures = []
    edge.projections[blanked][1].xp = 0.0

    lineup, bench, _, _ = sq.pick_lineup(list(real_decision.squad), edge, 1, game_state)
    assert blanked not in lineup
    assert_lineup_is_legal(lineup, bench, real_decision.squad, game_state)


def test_build_views_sums_a_double_gameweek(real_decision, projections, game_state):
    from gaffer.core.types import PlayerGWProjection, ProjectionSet

    pid = real_decision.captain
    src = projections.projections[pid][1]
    single_view = sq.build_views(projections, game_state, [1])[1]

    doubled = ProjectionSet(season=projections.season, generated_at=projections.generated_at,
                            first_gw=1, last_gw=1)
    doubled.projections[pid] = {
        1: PlayerGWProjection(player_id=pid, gw=1, xp=src.xp * 2, sd=src.sd,
                              fixtures=list(src.fixtures) + list(src.fixtures))
    }
    view = sq.build_views(doubled, game_state, [1])[1]

    assert view.n_fixtures[pid] == 2
    assert view.xmins[pid] == pytest.approx(2 * single_view.xmins[pid])
    p1 = src.fixtures[0].p_appear
    # Two independent chances to appear, not a doubled probability.
    assert view.get_appear(pid) == pytest.approx(1.0 - (1.0 - p1) ** 2)
    assert view.get_appear(pid) <= 1.0


# ---------------------------------------------------------------------------
# Auto-substitution: FPL only makes a sub if the resulting formation is legal
# ---------------------------------------------------------------------------


def _autosub_fixture(specs):
    """(xi, bench, projections, state) for a hand-built 15.

    ``specs`` is a list of ``(player_id, position, xp, p_appear)`` with the XI
    first (ids 1-11) and the bench in autosub order (ids 12-15).
    """
    from gaffer.core.types import (
        Player, PlayerFixtureProjection, PlayerGWProjection, ProjectionSet,
    )

    class _State:
        def __init__(self, players):
            self.players = players

    players = {
        pid: Player(id=pid, code=pid, web_name="P%d" % pid, first_name="",
                    second_name="", team_id=1 + (pid % 5), position=pos, now_cost=50)
        for pid, pos, _xp, _pa in specs
    }
    projections = ProjectionSet(season="2026-27", generated_at="", first_gw=1, last_gw=1)
    for pid, _pos, xp, pa in specs:
        gwp = PlayerGWProjection(player_id=pid, gw=1)
        gwp.fixtures.append(PlayerFixtureProjection(
            player_id=pid, gw=1, fixture_id=1, opponent_id=2, is_home=True,
            p_appear=pa, p_start=pa, p_60=pa, xmins=90 * pa, xp_total=xp))
        gwp.xp = xp
        projections.projections[pid] = {1: gwp}
    xi = [s[0] for s in specs[:11]]
    bench = [s[0] for s in specs[11:]]
    return xi, bench, projections, _State(players)


def test_autosub_will_not_break_the_minimum_defender_count():
    """A 3-defender XI cannot cover a missing defender with a midfield sub.

    2-6-2 is not a legal team, so FPL skips the midfielders on the bench and
    takes the defender in slot 3 instead.
    """
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 1.0)]
        + [(2, scoring.DEF, 4.0, 0.0)]                                   # blanks
        + [(3, scoring.DEF, 4.0, 1.0), (4, scoring.DEF, 4.0, 1.0)]
        + [(i, scoring.MID, 5.0, 1.0) for i in range(5, 10)]
        + [(i, scoring.FWD, 5.0, 1.0) for i in (10, 11)]
        + [(12, scoring.GKP, 1.0, 1.0), (13, scoring.MID, 6.0, 1.0),
           (14, scoring.MID, 6.0, 1.0), (15, scoring.DEF, 1.0, 1.0)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(1.0)


def test_autosub_makes_no_substitution_when_no_formation_survives():
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 1.0)]
        + [(i, scoring.DEF, 4.0, 0.0) for i in range(2, 5)]               # all three blank
        + [(i, scoring.MID, 5.0, 1.0) for i in range(5, 10)]
        + [(i, scoring.FWD, 5.0, 1.0) for i in (10, 11)]
        + [(12, scoring.GKP, 1.0, 1.0)]
        + [(i, scoring.MID, 3.0, 1.0) for i in (13, 14, 15)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(0.0)


def test_autosub_will_not_exceed_the_maximum_defender_count():
    """5-4-1 with a missing midfielder cannot take a sixth defender."""
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 1.0)]
        + [(i, scoring.DEF, 4.0, 1.0) for i in range(2, 7)]
        + [(7, scoring.MID, 5.0, 0.0)]                                    # blanks
        + [(i, scoring.MID, 5.0, 1.0) for i in range(8, 11)]
        + [(11, scoring.FWD, 5.0, 1.0)]
        + [(12, scoring.GKP, 1.0, 1.0), (13, scoring.DEF, 9.0, 1.0),
           (14, scoring.MID, 3.0, 1.0), (15, scoring.FWD, 2.0, 1.0)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(3.0)


def test_autosub_takes_the_legal_bench_player_when_the_formation_allows_it():
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 1.0)]
        + [(i, scoring.DEF, 4.0, 1.0) for i in range(2, 6)]
        + [(6, scoring.MID, 5.0, 0.0)]                                    # blanks
        + [(i, scoring.MID, 5.0, 1.0) for i in range(7, 10)]
        + [(i, scoring.FWD, 5.0, 1.0) for i in (10, 11)]
        + [(12, scoring.GKP, 1.0, 1.0), (13, scoring.MID, 7.0, 1.0),
           (14, scoring.DEF, 2.0, 1.0), (15, scoring.FWD, 2.0, 1.0)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(7.0)


def test_the_reserve_keeper_only_ever_replaces_the_keeper():
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 0.0)]                                      # keeper blanks
        + [(i, scoring.DEF, 4.0, 1.0) for i in range(2, 6)]
        + [(i, scoring.MID, 5.0, 1.0) for i in range(6, 10)]
        + [(i, scoring.FWD, 5.0, 1.0) for i in (10, 11)]
        + [(12, scoring.GKP, 3.0, 1.0), (13, scoring.MID, 7.0, 1.0),
           (14, scoring.DEF, 2.0, 1.0), (15, scoring.FWD, 2.0, 1.0)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    # Only the reserve keeper's 3.0; no outfielder blanked, so nothing else.
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(3.0)


def test_a_bench_player_who_blanks_passes_the_slot_down():
    from gaffer.optimize.chips import _autosub_value

    specs = (
        [(1, scoring.GKP, 4.0, 1.0)]
        + [(i, scoring.DEF, 4.0, 1.0) for i in range(2, 6)]
        + [(6, scoring.MID, 5.0, 0.0)]
        + [(i, scoring.MID, 5.0, 1.0) for i in range(7, 10)]
        + [(i, scoring.FWD, 5.0, 1.0) for i in (10, 11)]
        + [(12, scoring.GKP, 1.0, 1.0), (13, scoring.MID, 0.0, 0.0),
           (14, scoring.MID, 4.0, 1.0), (15, scoring.FWD, 2.0, 1.0)]
    )
    xi, bench, projections, state = _autosub_fixture(specs)
    assert _autosub_value(xi, bench, projections, 1, state) == pytest.approx(4.0)
