"""Squad selection: the MILP that turns projections into a legal 15, and the
lineup rules that turn a 15 into an XI, a bench order, a captain and a vice.

Two things matter more than cleverness here. First, legality: a squad that
breaks a quota or the budget by one tenth is not a slightly worse answer, it is
not an answer at all, so every constraint is stated explicitly and re-checked on
the way out. Second, money is integer: `now_cost` is in tenths of a million and
stays that way from end to end. Floats would round £100.0m into £100.00000001m
and quietly make an illegal squad look feasible.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pulp

from gaffer.core import scoring
from gaffer.core.config import Config
from gaffer.core.types import GWDecision, ProjectionSet

# Bench slot 1 is the first outfield sub, slot 3 the last; the reserve keeper is
# handled separately (see `bench_weights`).
N_OUTFIELD_BENCH = 3

# A vice-captain this far behind the best alternative is still worth taking if
# he plays at a different time to the captain. Beyond it, take the points.
VICE_XP_TOLERANCE = 0.35

# Nudges the solver off exactly-tied optima in a fixed direction so two runs of
# the same problem return the same squad. Small enough to never change a real
# decision (a full squad of 15 can shift the objective by at most 1.5e-3).
TIEBREAK_EPS = 1e-4


class OptimizeError(RuntimeError):
    """Raised when the solver fails or returns something illegal."""


# ---------------------------------------------------------------------------
# Formations
# ---------------------------------------------------------------------------


def valid_formations() -> List[Tuple[int, int, int]]:
    """Every legal (DEF, MID, FWD) outfield split. One keeper is implied."""
    out: List[Tuple[int, int, int]] = []
    for d in range(scoring.SQUAD_MIN_PLAY[scoring.DEF], scoring.SQUAD_MAX_PLAY[scoring.DEF] + 1):
        for m in range(scoring.SQUAD_MIN_PLAY[scoring.MID], scoring.SQUAD_MAX_PLAY[scoring.MID] + 1):
            for f in range(
                scoring.SQUAD_MIN_PLAY[scoring.FWD], scoring.SQUAD_MAX_PLAY[scoring.FWD] + 1
            ):
                if d + m + f == scoring.SQUAD_PLAY - 1:
                    out.append((d, m, f))
    return out


FORMATIONS: List[Tuple[int, int, int]] = valid_formations()


def formation_name(lineup_positions: Sequence[int]) -> str:
    counts = [list(lineup_positions).count(p) for p in (scoring.DEF, scoring.MID, scoring.FWD)]
    return "-".join(str(c) for c in counts)


# ---------------------------------------------------------------------------
# Per-gameweek view of a ProjectionSet
# ---------------------------------------------------------------------------


@dataclass
class GWView:
    """Everything the optimizer reads about one gameweek, precomputed once.

    Pulling these out of the nested ProjectionSet ahead of time matters: the
    planner touches xp[player][gw] inside a loop that runs hundreds of thousands
    of times while building constraints.
    """

    gw: int
    xp: Dict[int, float] = field(default_factory=dict)
    sd: Dict[int, float] = field(default_factory=dict)
    # Expected points available in the right tail: P(haul) x how far a haul
    # clears the mean. This, not the standard deviation, is what the armband is
    # actually buying — sd is symmetric and counts a blank as "upside".
    tail: Dict[int, float] = field(default_factory=dict)
    p_appear: Dict[int, float] = field(default_factory=dict)
    xmins: Dict[int, float] = field(default_factory=dict)
    n_fixtures: Dict[int, int] = field(default_factory=dict)
    kickoffs: Dict[int, Tuple[str, ...]] = field(default_factory=dict)

    def get_xp(self, player_id: int) -> float:
        return self.xp.get(player_id, 0.0)

    def get_appear(self, player_id: int) -> float:
        return self.p_appear.get(player_id, 0.0)


def build_views(
    projections: ProjectionSet, state: Any, gws: Sequence[int]
) -> Dict[int, GWView]:
    kickoff_by_fixture = {f.id: (f.kickoff_time or "") for f in state.fixtures}
    views: Dict[int, GWView] = {}
    # Imported lazily: strategy pulls pick_lineup back out of this module, and a
    # module-level import in both directions is a cycle waiting to happen.
    try:
        from gaffer.optimize.strategy import haul_profiles
    except Exception:  # pragma: no cover - the ceiling term is optional
        haul_profiles = None
    for gw in gws:
        view = GWView(gw=int(gw))
        profiles = {}
        if haul_profiles is not None:
            try:
                profiles = haul_profiles(projections, int(gw))
            except Exception:
                profiles = {}
        for pid, per_gw in projections.projections.items():
            gwp = per_gw.get(int(gw))
            if gwp is None:
                continue
            view.xp[pid] = gwp.xp
            view.sd[pid] = gwp.sd
            prof = profiles.get(pid)
            view.tail[pid] = (
                prof.p_haul * max(0.0, prof.e_points_if_haul - gwp.xp) if prof else 0.0
            )
            view.n_fixtures[pid] = len(gwp.fixtures)
            view.xmins[pid] = sum(f.xmins for f in gwp.fixtures)
            # A double gameweek gives two independent chances to appear.
            miss = 1.0
            for f in gwp.fixtures:
                miss *= 1.0 - f.p_appear
            view.p_appear[pid] = 1.0 - miss
            view.kickoffs[pid] = tuple(
                sorted(kickoff_by_fixture.get(f.fixture_id, "") for f in gwp.fixtures)
            )
        views[int(gw)] = view
    return views


def horizon_xp(views: Dict[int, GWView], player_id: int, gws: Sequence[int]) -> float:
    return sum(views[g].get_xp(player_id) for g in gws if g in views)


def mean_xmins(views: Dict[int, GWView], player_id: int, gws: Sequence[int]) -> float:
    """Average expected minutes across the gameweeks the player actually has a
    fixture in. Averaging over blanks too would filter out a fine player who
    happens to have one blank in the window."""
    vals = [
        views[g].xmins.get(player_id, 0.0)
        for g in gws
        if g in views and views[g].n_fixtures.get(player_id, 0) > 0
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def bench_weights(config: Config) -> Tuple[List[float], float]:
    """``(outfield bench slot weights 1..3, reserve keeper weight)``.

    `config.optimizer.bench_weights` is read as *ordered by autosub likelihood*,
    not by the order FPL displays the bench in. The reserve keeper only ever
    scores when the first-choice keeper records zero minutes, which is far rarer
    than any outfield autosub, so he takes the smallest of the four weights and
    the three outfield slots take the rest in order.
    """
    raw = [float(w) for w in (config.optimizer.bench_weights or [])]
    while len(raw) < 4:
        raw.append(0.0)
    return raw[:3], raw[3]


def captain_multiplier(chip: Optional[str]) -> int:
    return scoring.TRIPLE_CAPTAIN_MULTIPLIER if chip == "3xc" else scoring.CAPTAIN_MULTIPLIER


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class SolveReport:
    solver: str = ""
    status: str = ""
    seconds: float = 0.0
    objective: float = 0.0
    n_variables: int = 0
    n_constraints: int = 0

    def __str__(self) -> str:
        return "%s %s in %.2fs, objective %.4f (%d vars, %d rows)" % (
            self.solver,
            self.status,
            self.seconds,
            self.objective,
            self.n_variables,
            self.n_constraints,
        )


def build_solver(
    config: Config,
    solver_name: Optional[str] = None,
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    seed: Optional[int] = None,
    msg: bool = False,
) -> Tuple[Any, str]:
    """HiGHS if it is installed, CBC otherwise. Never silently picks neither."""
    wanted = (solver_name or config.optimizer.solver or "auto").lower()
    order = {
        "auto": ["highs", "cbc"],
        "highs": ["highs"],
        "cbc": ["cbc"],
        "pulp_cbc_cmd": ["cbc"],
    }.get(wanted)
    if order is None:
        raise OptimizeError("unknown solver %r (want auto, highs or cbc)" % wanted)

    limit = int(time_limit if time_limit is not None else config.optimizer.time_limit_seconds)
    gap = float(mip_gap if mip_gap is not None else config.optimizer.mip_gap)
    tried: List[str] = []
    for name in order:
        try:
            if name == "highs":
                kwargs: Dict[str, Any] = {}
                if seed is not None:
                    kwargs["random_seed"] = int(seed)
                solver = pulp.HiGHS(msg=msg, timeLimit=limit, gapRel=gap, **kwargs)
                label = "HiGHS"
            else:
                options = []
                if seed is not None:
                    options += ["randomCbcSeed", str(int(seed))]
                solver = pulp.PULP_CBC_CMD(
                    msg=msg, timeLimit=limit, gapRel=gap, options=options, threads=1
                )
                label = "CBC"
            if solver.available():
                return solver, label
            tried.append("%s (not available)" % name)
        except Exception as exc:  # pragma: no cover - depends on local install
            tried.append("%s (%s)" % (name, exc))
    raise OptimizeError("no MILP solver available; tried %s" % ", ".join(tried))


def solve_or_raise(problem: pulp.LpProblem, solver: Any, label: str) -> SolveReport:
    """Solve and refuse anything that is not a proven optimum.

    A time-limited or infeasible MILP silently handing back its incumbent is the
    single easiest way to ship an illegal team, so the status is checked here and
    nowhere else has to remember to.
    """
    t0 = time.time()
    problem.solve(solver)
    seconds = time.time() - t0
    status = pulp.LpStatus[problem.status]
    report = SolveReport(
        solver=label,
        status=status,
        seconds=seconds,
        objective=float(pulp.value(problem.objective) or 0.0),
        n_variables=len(problem.variables()),
        n_constraints=len(problem.constraints),
    )
    if status != "Optimal":
        raise OptimizeError(
            "%s returned status %r for %s after %.1fs — refusing to use a "
            "non-optimal squad" % (label, status, problem.name, seconds)
        )
    return report


def _binary_value(var: Any) -> int:
    value = var.value()
    if value is None:
        return 0
    return 1 if value > 0.5 else 0


# ---------------------------------------------------------------------------
# Lineup selection
# ---------------------------------------------------------------------------


def best_lineup(
    squad_ids: Sequence[int],
    view: GWView,
    state: Any,
) -> Tuple[List[int], List[int]]:
    """Highest-xP legal XI from a 15, by exhaustive formation search.

    Within a formation the best XI is simply the top-k by xP in each position, so
    enumerating the eight legal formations is exact and costs microseconds — no
    need for a solver here.
    """
    by_pos: Dict[int, List[int]] = {p: [] for p in scoring.POSITIONS}
    for pid in squad_ids:
        by_pos[state.players[pid].position].append(pid)
    for pos in scoring.POSITIONS:
        by_pos[pos].sort(
            key=lambda pid: (-view.get_xp(pid), -view.get_appear(pid), pid)
        )
    if len(by_pos[scoring.GKP]) < 1:
        raise OptimizeError("squad has no goalkeeper")

    keeper = by_pos[scoring.GKP][0]
    best_total = None
    best_pick: List[int] = []
    for d, m, f in FORMATIONS:
        if (
            len(by_pos[scoring.DEF]) < d
            or len(by_pos[scoring.MID]) < m
            or len(by_pos[scoring.FWD]) < f
        ):
            continue
        pick = (
            by_pos[scoring.DEF][:d] + by_pos[scoring.MID][:m] + by_pos[scoring.FWD][:f]
        )
        total = sum(view.get_xp(p) for p in pick)
        # Ties broken on the formation order in FORMATIONS, which is fixed, so
        # the same inputs always yield the same shape.
        if best_total is None or total > best_total + 1e-12:
            best_total = total
            best_pick = pick
    if best_total is None:
        raise OptimizeError(
            "no legal formation available from squad of %d" % len(squad_ids)
        )

    lineup = [keeper] + best_pick
    bench = [pid for pid in squad_ids if pid not in set(lineup)]
    return lineup, bench


def order_bench(bench_ids: Sequence[int], view: GWView, state: Any) -> List[int]:
    """FPL bench order: the reserve keeper occupies his own slot and is never
    ranked against outfielders (he can only ever replace the other keeper).
    Outfielders are ranked by xP weighted by the chance they actually appear —
    an autosub is worthless if the player on the bench did not play either."""
    keepers = [p for p in bench_ids if state.players[p].position == scoring.GKP]
    outfield = [p for p in bench_ids if state.players[p].position != scoring.GKP]
    outfield.sort(
        key=lambda pid: (
            -(view.get_xp(pid) * view.get_appear(pid)),
            -view.get_xp(pid),
            pid,
        )
    )
    keepers.sort(key=lambda pid: (-view.get_xp(pid), pid))
    return keepers + outfield


def choose_captain(
    lineup: Sequence[int], view: GWView
) -> Tuple[int, int]:
    """(captain, vice) — captain is the top xP starter.

    The vice only ever scores when the captain records zero minutes, so the
    second-best starter is the default. Where two candidates are within
    `VICE_XP_TOLERANCE` of each other, prefer one kicking off at a different time
    to the captain: a vice in the captain's own match shares his postponement
    risk and tells you nothing you did not already know.
    """
    ranked = sorted(lineup, key=lambda pid: (-view.get_xp(pid), pid))
    captain = ranked[0]
    rest = ranked[1:]
    if not rest:
        return captain, captain
    best = rest[0]
    cap_kicks = set(view.kickoffs.get(captain, ()))
    top_xp = view.get_xp(best)
    for pid in rest:
        if view.get_xp(pid) < top_xp - VICE_XP_TOLERANCE:
            break
        kicks = set(view.kickoffs.get(pid, ()))
        if kicks and cap_kicks and kicks.isdisjoint(cap_kicks):
            return captain, pid
    return captain, best


def pick_lineup(
    squad_ids: List[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    chip: Optional[str] = None,
    view: Optional[GWView] = None,
) -> Tuple[List[int], List[int], int, int]:
    """``(lineup, bench_in_order, captain_id, vice_id)`` for one gameweek.

    Bench boost does not change the XI — FPL still requires a legal eleven, the
    bench simply also scores — so the same selection runs for every chip and only
    the scoring in `lineup_expected_points` differs.
    """
    if len(squad_ids) != scoring.SQUAD_SIZE:
        raise OptimizeError(
            "pick_lineup needs %d players, got %d" % (scoring.SQUAD_SIZE, len(squad_ids))
        )
    view = view if view is not None else build_views(projections, state, [gw])[int(gw)]
    lineup, bench = best_lineup(squad_ids, view, state)
    bench = order_bench(bench, view, state)
    captain, vice = choose_captain(lineup, view)
    return lineup, bench, captain, vice


def lineup_expected_points(
    lineup: Sequence[int],
    bench: Sequence[int],
    captain: int,
    view: GWView,
    chip: Optional[str] = None,
) -> float:
    """Gross expected points for a gameweek, before any transfer hits."""
    total = sum(view.get_xp(p) for p in lineup)
    total += view.get_xp(captain) * (captain_multiplier(chip) - 1)
    if chip == "bboost":
        total += sum(view.get_xp(p) for p in bench)
    return total


def expected_gw_points(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    chip: Optional[str] = None,
    view: Optional[GWView] = None,
) -> float:
    view = view if view is not None else build_views(projections, state, [gw])[int(gw)]
    lineup, bench, captain, _ = pick_lineup(
        list(squad_ids), projections, gw, state, chip=chip, view=view
    )
    return lineup_expected_points(lineup, bench, captain, view, chip)


# ---------------------------------------------------------------------------
# Legality checks
# ---------------------------------------------------------------------------


def squad_problems(
    squad_ids: Sequence[int], state: Any, budget: int = scoring.BUDGET_TENTHS,
    prices: Optional[Dict[int, int]] = None,
) -> List[str]:
    """Everything wrong with a 15, as plain sentences. Empty list means legal."""
    problems: List[str] = []
    ids = list(squad_ids)
    if len(ids) != scoring.SQUAD_SIZE:
        problems.append("squad has %d players, needs %d" % (len(ids), scoring.SQUAD_SIZE))
    if len(set(ids)) != len(ids):
        problems.append("squad contains a duplicate player")
    positions = [state.players[p].position for p in ids]
    for pos, need in sorted(scoring.SQUAD_SELECT.items()):
        got = positions.count(pos)
        if got != need:
            problems.append(
                "%s quota is %d, squad has %d" % (scoring.POS_NAME[pos], need, got)
            )
    teams: Dict[int, int] = {}
    for p in ids:
        teams[state.players[p].team_id] = teams.get(state.players[p].team_id, 0) + 1
    for team_id, count in sorted(teams.items()):
        if count > scoring.TEAM_LIMIT:
            problems.append(
                "%d players from %s, limit is %d"
                % (count, state.short_name(team_id), scoring.TEAM_LIMIT)
            )
    prices = prices or {}
    cost = sum(int(prices.get(p, state.players[p].now_cost)) for p in ids)
    if cost > budget:
        problems.append("squad costs %d tenths, budget is %d" % (cost, budget))
    return problems


def lineup_problems(
    lineup: Sequence[int], bench: Sequence[int], squad_ids: Sequence[int], state: Any
) -> List[str]:
    problems: List[str] = []
    if len(lineup) != scoring.SQUAD_PLAY:
        problems.append("lineup has %d players, needs %d" % (len(lineup), scoring.SQUAD_PLAY))
    if len(bench) != scoring.SQUAD_SIZE - scoring.SQUAD_PLAY:
        problems.append("bench has %d players, needs 4" % len(bench))
    if set(lineup) & set(bench):
        problems.append("a player is in both the lineup and the bench")
    if set(list(lineup) + list(bench)) != set(squad_ids):
        problems.append("lineup plus bench is not the squad")
    positions = [state.players[p].position for p in lineup]
    if not scoring.lineup_is_legal(positions):
        problems.append(
            "illegal formation %s (%d GKP)"
            % (formation_name(positions), positions.count(scoring.GKP))
        )
    if bench and state.players[bench[0]].position != scoring.GKP:
        problems.append("bench slot 0 is not the reserve keeper")
    return problems


def assert_legal(
    squad_ids: Sequence[int],
    lineup: Sequence[int],
    bench: Sequence[int],
    state: Any,
    budget: int = scoring.BUDGET_TENTHS,
    prices: Optional[Dict[int, int]] = None,
) -> None:
    problems = squad_problems(squad_ids, state, budget, prices) + lineup_problems(
        lineup, bench, squad_ids, state
    )
    if problems:
        raise OptimizeError("illegal squad: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------


def candidate_ids(
    projections: ProjectionSet,
    state: Any,
    config: Config,
    gws: Sequence[int],
    views: Optional[Dict[int, GWView]] = None,
    extra: Iterable[int] = (),
) -> List[int]:
    """Players worth putting in front of the solver.

    Anyone below `min_xmins_to_consider` is dropped: an injured or third-choice
    player is not a cheap enabler, he is a zero. Locked-in players and anyone the
    caller already owns survive the filter regardless — you cannot optimise
    around a player by pretending he does not exist.
    """
    views = views if views is not None else build_views(projections, state, gws)
    forced = set(int(p) for p in config.optimizer.locked_in) | set(int(p) for p in extra)
    blocked = set(int(p) for p in config.optimizer.locked_out)
    threshold = float(config.optimizer.min_xmins_to_consider)
    out: List[int] = []
    for pid in sorted(projections.projections.keys()):
        if pid not in state.players:
            continue
        if pid in blocked:
            continue
        if pid in forced or mean_xmins(views, pid, gws) >= threshold:
            out.append(pid)
    for pid in sorted(forced):
        if pid not in state.players:
            raise OptimizeError("locked-in player %d is not in the game state" % pid)
        if pid in blocked:
            raise OptimizeError("player %d is both locked in and locked out" % pid)
    missing = [p for p in forced if p not in set(out)]
    if missing:
        raise OptimizeError(
            "locked-in players have no projection: %s" % sorted(missing)
        )
    return out


# ---------------------------------------------------------------------------
# The squad MILP
# ---------------------------------------------------------------------------


@dataclass
class SquadSolution:
    squad: List[int] = field(default_factory=list)
    lineup_by_gw: Dict[int, List[int]] = field(default_factory=dict)
    captain_by_gw: Dict[int, int] = field(default_factory=dict)
    cost: int = 0
    bank: int = 0
    objective: float = 0.0
    horizon_points: float = 0.0
    gws: List[int] = field(default_factory=list)
    report: SolveReport = field(default_factory=SolveReport)


def solve_squad_milp(
    projections: ProjectionSet,
    state: Any,
    config: Config,
    gws: Sequence[int],
    budget: int = scoring.BUDGET_TENTHS,
    locked_in: Optional[Sequence[int]] = None,
    locked_out: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[int]] = None,
    views: Optional[Dict[int, GWView]] = None,
    prices: Optional[Dict[int, int]] = None,
    solver_name: Optional[str] = None,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    bench_mode: str = "ordered",
    msg: bool = False,
) -> SquadSolution:
    """Pick the 15 that maximises decayed expected points over `gws`.

    Variables
      x[p]     player p is in the squad (binary)
      y[p, w]  player p starts in gameweek w (binary, y <= x)
      c[p, w]  player p is captain in gameweek w (continuous in [0, 1], c <= y)
      z[p,w,k] benched player p sits in outfield bench slot k (continuous)

    `c` and `z` are deliberately not declared integer. With x and y fixed, both
    sub-problems are assignment problems on a totally unimodular matrix, so the
    LP optimum is already integral at every vertex; declaring them binary would
    only give the brancher thousands of extra variables to chew on.
    """
    gws = [int(g) for g in gws]
    if not gws:
        raise OptimizeError("solve_squad_milp needs at least one gameweek")
    views = views if views is not None else build_views(projections, state, gws)
    missing = [g for g in gws if g not in views]
    if missing:
        raise OptimizeError("projections do not cover gameweeks %s" % missing)

    locked_in = [int(p) for p in (locked_in if locked_in is not None else config.optimizer.locked_in)]
    locked_out = [
        int(p) for p in (locked_out if locked_out is not None else config.optimizer.locked_out)
    ]
    if candidates is None:
        candidates = candidate_ids(projections, state, config, gws, views=views)
    pool = sorted(set(int(p) for p in candidates) | set(locked_in))
    pool = [p for p in pool if p not in set(locked_out)]
    if not pool:
        raise OptimizeError("no candidate players survived filtering")

    price = {p: int((prices or {}).get(p, state.players[p].now_cost)) for p in pool}
    position = {p: state.players[p].position for p in pool}
    team_of = {p: state.players[p].team_id for p in pool}
    for pos, need in scoring.SQUAD_SELECT.items():
        have = sum(1 for p in pool if position[p] == pos)
        if have < need:
            raise OptimizeError(
                "only %d %s candidates, need %d — relax min_xmins_to_consider"
                % (have, scoring.POS_NAME[pos], need)
            )

    decay = float(config.optimizer.decay)
    ceiling_weight = max(0.0, float(getattr(config.optimizer, "captain_ceiling_weight", 0.0)))
    out_weights, gk_weight = bench_weights(config)
    if bench_mode not in ("ordered", "flat"):
        raise OptimizeError("bench_mode must be 'ordered' or 'flat'")
    flat_weight = sum(out_weights) / max(len(out_weights), 1)

    prob = pulp.LpProblem("gaffer_squad", pulp.LpMaximize)
    x = {p: pulp.LpVariable("x_%d" % p, cat=pulp.LpBinary) for p in pool}
    y = {
        (p, w): pulp.LpVariable("y_%d_%d" % (p, w), cat=pulp.LpBinary)
        for p in pool
        for w in gws
    }
    c = {
        (p, w): pulp.LpVariable("c_%d_%d" % (p, w), lowBound=0, upBound=1)
        for p in pool
        for w in gws
    }
    outfield = [p for p in pool if position[p] != scoring.GKP]
    z: Dict[Tuple[int, int, int], Any] = {}
    if bench_mode == "ordered":
        z = {
            (p, w, k): pulp.LpVariable("z_%d_%d_%d" % (p, w, k), lowBound=0, upBound=1)
            for p in outfield
            for w in gws
            for k in range(N_OUTFIELD_BENCH)
        }

    # --- objective ---------------------------------------------------------
    terms = []
    for i, w in enumerate(gws):
        d = decay ** i
        view = views[w]
        for p in pool:
            xp = view.get_xp(p)
            if xp:
                terms.append(d * xp * y[(p, w)])
                # The armband doubles one player, so its value is the right
                # tail rather than the average. With ceiling_weight at 0 this is
                # exactly the old mean-only term; above 0 a high-variance
                # premium is preferred to a marginally higher-mean safe pick,
                # which is the trade a manager chasing overall rank wants and
                # the one a manager protecting a rank does not.
                cap = xp
                if ceiling_weight:
                    cap += ceiling_weight * max(0.0, float(view.tail.get(p, 0.0)))
                terms.append(d * cap * c[(p, w)])
        for p in pool:
            if position[p] == scoring.GKP and gk_weight:
                xp = view.get_xp(p)
                if xp:
                    terms.append(d * gk_weight * xp * (x[p] - y[(p, w)]))
        if bench_mode == "ordered":
            for p in outfield:
                xp = view.get_xp(p)
                if not xp:
                    continue
                for k in range(N_OUTFIELD_BENCH):
                    if out_weights[k]:
                        terms.append(d * out_weights[k] * xp * z[(p, w, k)])
        elif flat_weight:
            for p in outfield:
                xp = view.get_xp(p)
                if xp:
                    terms.append(d * flat_weight * xp * (x[p] - y[(p, w)]))
    # Deterministic tie-break: among equal-value squads take the cheaper one.
    terms.append(-TIEBREAK_EPS * pulp.lpSum(price[p] * x[p] for p in pool) / 1000.0)
    prob += pulp.lpSum(terms)

    # --- squad constraints -------------------------------------------------
    prob += pulp.lpSum(x[p] for p in pool) == scoring.SQUAD_SIZE, "squad_size"
    for pos, need in sorted(scoring.SQUAD_SELECT.items()):
        prob += (
            pulp.lpSum(x[p] for p in pool if position[p] == pos) == need,
            "quota_%s" % scoring.POS_NAME[pos],
        )
    prob += (
        pulp.lpSum(price[p] * x[p] for p in pool) <= int(budget),
        "budget",
    )
    for team_id in sorted(set(team_of.values())):
        members = [p for p in pool if team_of[p] == team_id]
        if len(members) > scoring.TEAM_LIMIT:
            prob += (
                pulp.lpSum(x[p] for p in members) <= scoring.TEAM_LIMIT,
                "club_%d" % team_id,
            )
    for p in locked_in:
        prob += x[p] == 1, "lock_in_%d" % p

    # --- lineup constraints ------------------------------------------------
    for w in gws:
        prob += (
            pulp.lpSum(y[(p, w)] for p in pool) == scoring.SQUAD_PLAY,
            "xi_size_%d" % w,
        )
        for pos in scoring.POSITIONS:
            members = [p for p in pool if position[p] == pos]
            prob += (
                pulp.lpSum(y[(p, w)] for p in members) >= scoring.SQUAD_MIN_PLAY[pos],
                "xi_min_%s_%d" % (scoring.POS_NAME[pos], w),
            )
            prob += (
                pulp.lpSum(y[(p, w)] for p in members) <= scoring.SQUAD_MAX_PLAY[pos],
                "xi_max_%s_%d" % (scoring.POS_NAME[pos], w),
            )
        for p in pool:
            prob += y[(p, w)] <= x[p], "start_in_squad_%d_%d" % (p, w)
            prob += c[(p, w)] <= y[(p, w)], "captain_starts_%d_%d" % (p, w)
        prob += pulp.lpSum(c[(p, w)] for p in pool) == 1, "one_captain_%d" % w
        if bench_mode == "ordered":
            for k in range(N_OUTFIELD_BENCH):
                prob += (
                    pulp.lpSum(z[(p, w, k)] for p in outfield) == 1,
                    "bench_slot_%d_%d" % (k, w),
                )
            for p in outfield:
                prob += (
                    pulp.lpSum(z[(p, w, k)] for k in range(N_OUTFIELD_BENCH))
                    == x[p] - y[(p, w)],
                    "bench_member_%d_%d" % (p, w),
                )

    solver, label = build_solver(
        config, solver_name=solver_name, time_limit=time_limit, seed=seed, msg=msg
    )
    report = solve_or_raise(prob, solver, label)

    squad = sorted(p for p in pool if _binary_value(x[p]))
    lineup_by_gw: Dict[int, List[int]] = {}
    captain_by_gw: Dict[int, int] = {}
    for w in gws:
        lineup_by_gw[w] = sorted(p for p in pool if _binary_value(y[(p, w)]))
        picks = [(c[(p, w)].value() or 0.0, p) for p in pool]
        captain_by_gw[w] = max(picks)[1] if picks else 0
    cost = sum(price[p] for p in squad)

    problems = squad_problems(squad, state, int(budget), price)
    if problems:
        raise OptimizeError(
            "solver returned an illegal squad (%s): %s" % (label, "; ".join(problems))
        )

    horizon = 0.0
    for i, w in enumerate(gws):
        view = views[w]
        lineup, bench, captain, _ = pick_lineup(
            squad, projections, w, state, view=view
        )
        horizon += (decay ** i) * lineup_expected_points(lineup, bench, captain, view)

    return SquadSolution(
        squad=squad,
        lineup_by_gw=lineup_by_gw,
        captain_by_gw=captain_by_gw,
        cost=cost,
        bank=int(budget) - cost,
        objective=report.objective,
        horizon_points=horizon,
        gws=list(gws),
        report=report,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pick_initial_squad(
    projections: ProjectionSet,
    state: Any,
    config: Config,
    gws: Optional[List[int]] = None,
    budget: int = scoring.BUDGET_TENTHS,
    chip: Optional[str] = None,
    solver_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> GWDecision:
    """The GW1 (or wildcard) squad, chosen over a whole horizon rather than one
    gameweek: a squad built for a single fixture round is a squad you have to
    immediately transfer out of."""
    if gws is None:
        first = projections.first_gw
        last = min(projections.last_gw, first + int(config.optimizer.horizon) - 1)
        gws = list(range(first, last + 1))
    gws = [int(g) for g in gws]
    views = build_views(projections, state, gws)
    solution = solve_squad_milp(
        projections,
        state,
        config,
        gws,
        budget=budget,
        views=views,
        solver_name=solver_name,
        seed=seed,
    )
    gw = gws[0]
    view = views[gw]
    lineup, bench, captain, vice = pick_lineup(
        solution.squad, projections, gw, state, chip=chip, view=view
    )
    assert_legal(solution.squad, lineup, bench, state, budget)
    points = lineup_expected_points(lineup, bench, captain, view, chip)

    notes = [
        "Initial squad chosen over GW%d-%d (decay %.2f), %s."
        % (gws[0], gws[-1], config.optimizer.decay, solution.report),
        "Spend %.1fm of %.1fm, %.1fm in the bank."
        % (solution.cost / 10.0, budget / 10.0, solution.bank / 10.0),
        "Horizon expected points %.1f over %d gameweeks (starting XI plus captain, "
        "before any transfers)." % (solution.horizon_points, len(gws)),
        "GW%d: %s, captain %s (%.2f xP), vice %s (%.2f xP)."
        % (
            gw,
            formation_name([state.players[p].position for p in lineup]),
            state.players[captain].web_name,
            view.get_xp(captain),
            state.players[vice].web_name,
            view.get_xp(vice),
        ),
    ]
    if config.optimizer.locked_in:
        notes.append(
            "Forced in by config: %s"
            % ", ".join(
                state.players[p].web_name
                for p in config.optimizer.locked_in
                if p in state.players
            )
        )
    if config.optimizer.locked_out:
        notes.append(
            "Excluded by config: %s"
            % ", ".join(
                state.players[p].web_name
                for p in config.optimizer.locked_out
                if p in state.players
            )
        )

    return GWDecision(
        gw=gw,
        transfers=[],
        hits=0,
        chip=chip,
        lineup=lineup,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        squad=list(solution.squad),
        bank_after=solution.bank,
        free_transfers_after=1,
        expected_points=points,
        expected_points_net=points,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def squad_table(
    decision: GWDecision,
    projections: ProjectionSet,
    state: Any,
    gws: Sequence[int],
    views: Optional[Dict[int, GWView]] = None,
) -> str:
    views = views if views is not None else build_views(projections, state, gws)
    role: Dict[int, str] = {}
    for pid in decision.lineup:
        role[pid] = "XI"
    for i, pid in enumerate(decision.bench):
        role[pid] = "B%d" % i
    if decision.captain is not None:
        role[decision.captain] = "XI (C)"
    if decision.vice_captain is not None:
        role[decision.vice_captain] = "XI (V)"

    header = "%-4s %-18s %-5s %6s %8s %8s  %s" % (
        "pos",
        "player",
        "team",
        "price",
        "GW%d xP" % gws[0],
        "%dGW xP" % len(gws),
        "role",
    )
    lines = [header, "-" * len(header)]
    ordered = decision.lineup + decision.bench
    ordered.sort(
        key=lambda pid: (
            0 if pid in set(decision.lineup) else 1,
            state.players[pid].position,
            -horizon_xp(views, pid, gws),
        )
    )
    for pid in ordered:
        player = state.players[pid]
        lines.append(
            "%-4s %-18s %-5s %6.1f %8.2f %8.2f  %s"
            % (
                scoring.POS_NAME[player.position],
                player.web_name[:18],
                state.short_name(player.team_id),
                player.price,
                views[gws[0]].get_xp(pid),
                horizon_xp(views, pid, gws),
                role.get(pid, ""),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _main() -> int:
    from gaffer.core.config import Config as _Config
    from gaffer.data.loaders import load_game_state
    from gaffer.model.xp import XPEngine

    parser = argparse.ArgumentParser(description="gaffer squad optimizer smoke test")
    parser.add_argument("--gws", default="", help="e.g. 1-6")
    parser.add_argument("--budget", type=int, default=scoring.BUDGET_TENTHS)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    cfg = _Config.load()
    t0 = time.time()
    game = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (game.summary(), time.time() - t0))

    if args.gws:
        lo, _, hi = args.gws.partition("-")
        GWS = list(range(int(lo), int(hi or lo) + 1))
    else:
        GWS = list(range(game.current_gw, game.current_gw + cfg.optimizer.horizon))

    engine = XPEngine(cfg)
    PS = engine.load_projections(GWS[0], GWS[-1])
    if PS is None or args.refit:
        print("projecting GW%d-%d from scratch..." % (GWS[0], GWS[-1]))
        engine.fit(game)
        PS = engine.project(game, GWS)
        engine.save_projections(PS)
    print("projections: %d players, GW%d-%d" % (len(PS.projections), PS.first_gw, PS.last_gw))

    VIEWS = build_views(PS, game, GWS)
    POOL = candidate_ids(PS, game, cfg, GWS, views=VIEWS)
    print(
        "candidate pool %d of %d players (min_xmins %.0f)"
        % (len(POOL), len(PS.projections), cfg.optimizer.min_xmins_to_consider)
    )

    print("\n" + "=" * 78)
    print("INITIAL SQUAD, GW%d-%d HORIZON" % (GWS[0], GWS[-1]))
    print("=" * 78)
    DEC = pick_initial_squad(PS, game, cfg, GWS, budget=args.budget)
    print(squad_table(DEC, PS, game, GWS, views=VIEWS))
    for note in DEC.notes:
        print("  - %s" % note)

    COST = sum(game.players[p].now_cost for p in DEC.squad)
    print(
        "\nsquad cost %.1fm, bank %.1fm, GW%d expected points %.2f"
        % (COST / 10.0, DEC.bank_after / 10.0, DEC.gw, DEC.expected_points)
    )
    print(
        "starting XI (%s): %s"
        % (
            formation_name([game.players[p].position for p in DEC.lineup]),
            ", ".join(game.players[p].web_name for p in DEC.lineup),
        )
    )
    print("bench in order: %s" % ", ".join(game.players[p].web_name for p in DEC.bench))
    print(
        "captain %s, vice %s"
        % (game.players[DEC.captain].web_name, game.players[DEC.vice_captain].web_name)
    )

    # ------------------------------------------------------------------ gates
    print("\n" + "=" * 78)
    print("SANITY GATES")
    print("=" * 78)
    FAILURES: List[str] = []

    def gate(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            FAILURES.append(label)

    POSITIONS = [game.players[p].position for p in DEC.squad]
    CLUBS: Dict[int, int] = {}
    for p in DEC.squad:
        CLUBS[game.players[p].team_id] = CLUBS.get(game.players[p].team_id, 0) + 1
    gate("squad has exactly 15 players", len(DEC.squad) == 15, "%d" % len(DEC.squad))
    gate("no duplicate players", len(set(DEC.squad)) == 15)
    gate(
        "position quotas exact",
        all(POSITIONS.count(k) == v for k, v in scoring.SQUAD_SELECT.items()),
        " ".join(
            "%s %d" % (scoring.POS_NAME[k], POSITIONS.count(k))
            for k in scoring.POSITIONS
        ),
    )
    gate(
        "no more than 3 per club",
        max(CLUBS.values()) <= 3,
        "worst %s %d"
        % (
            game.short_name(max(CLUBS, key=lambda t: CLUBS[t])),
            max(CLUBS.values()),
        ),
    )
    gate("cost within budget", COST <= args.budget, "%d <= %d tenths" % (COST, args.budget))
    gate("bank is consistent", DEC.bank_after == args.budget - COST, "%d tenths" % DEC.bank_after)
    gate(
        "lineup is legal",
        not lineup_problems(DEC.lineup, DEC.bench, DEC.squad, game),
        "%s, bench slot 0 %s"
        % (
            formation_name([game.players[p].position for p in DEC.lineup]),
            game.position_name(DEC.bench[0]),
        ),
    )
    gate(
        "captain and vice are in the XI",
        DEC.captain in DEC.lineup and DEC.vice_captain in DEC.lineup,
    )
    gate("captain is not the vice", DEC.captain != DEC.vice_captain)
    CAP_XP = VIEWS[GWS[0]].get_xp(DEC.captain)
    gate(
        "captain is the top-xP starter",
        all(VIEWS[GWS[0]].get_xp(p) <= CAP_XP + 1e-9 for p in DEC.lineup),
        "%.2f xP" % CAP_XP,
    )
    # On a real GW1 the second-best starter is already in another match, so the
    # kickoff preference never has to bind. Exercise it directly.
    KICKS = {1: ("A",), 2: ("A",), 3: ("B",), 4: ("B",)}
    _, NEAR = choose_captain([1, 2, 3, 4], GWView(gw=0, xp={1: 9.0, 2: 6.0, 3: 5.8, 4: 3.0}, kickoffs=KICKS))
    _, FAR = choose_captain([1, 2, 3, 4], GWView(gw=0, xp={1: 9.0, 2: 6.0, 3: 5.0, 4: 3.0}, kickoffs=KICKS))
    gate(
        "vice takes a different kickoff inside tolerance",
        NEAR == 3 and FAR == 2,
        "0.2 xP behind -> #%d (other match), 1.0 behind -> #%d (captain's match)" % (NEAR, FAR),
    )

    BENCH_OUT = DEC.bench[1:]
    BENCH_KEYS = [
        VIEWS[GWS[0]].get_xp(p) * VIEWS[GWS[0]].get_appear(p) for p in BENCH_OUT
    ]
    gate(
        "outfield bench is ordered by xp * p_appear",
        all(BENCH_KEYS[i] >= BENCH_KEYS[i + 1] - 1e-9 for i in range(len(BENCH_KEYS) - 1)),
        " > ".join("%.2f" % k for k in BENCH_KEYS),
    )

    # Same optimum from a different solver and a different seed. MILP optima are
    # unique in value even when the argmax is not, so the objectives must match.
    ALT: Dict[str, Any] = {}
    for LABEL, KWARGS in (
        ("HiGHS seed 7", {"solver_name": "highs", "seed": 7}),
        ("HiGHS seed 4242", {"solver_name": "highs", "seed": 4242}),
        ("CBC", {"solver_name": "cbc"}),
    ):
        try:
            SOL = solve_squad_milp(PS, game, cfg, GWS, budget=args.budget, views=VIEWS, **KWARGS)
            ALT[LABEL] = SOL
            print(
                "  %-16s objective %.6f, horizon %.3f, cost %d, %s"
                % (LABEL, SOL.objective, SOL.horizon_points, SOL.cost, SOL.report)
            )
        except OptimizeError as exc:
            print("  %-16s FAILED: %s" % (LABEL, exc))
    BASE = ALT.get("HiGHS seed 7")
    gate(
        "all solvers and seeds agree on the objective",
        bool(ALT)
        and all(abs(s.objective - BASE.objective) < 2e-3 for s in ALT.values()),
        "spread %.2e over %d runs"
        % (
            max(abs(s.objective - BASE.objective) for s in ALT.values()) if ALT else 0.0,
            len(ALT),
        ),
    )
    gate(
        "all solvers and seeds return the same 15",
        all(s.squad == BASE.squad for s in ALT.values()),
        "%d distinct squads" % len(set(tuple(s.squad) for s in ALT.values())),
    )

    # Blanks and doubles do not exist in the schedule yet — every club has exactly
    # one fixture in every gameweek — so the code paths that handle them would go
    # untested until the day they suddenly matter. Fabricate both.
    from gaffer.core.types import PlayerGWProjection as _PGP

    EDGE = ProjectionSet(
        season=PS.season, generated_at=PS.generated_at, first_gw=GWS[0], last_gw=GWS[0]
    )
    for PID in DEC.squad:
        SRC = PS.projections[PID][GWS[0]]
        EDGE.projections[PID] = {
            GWS[0]: _PGP(
                player_id=PID, gw=GWS[0], xp=SRC.xp, sd=SRC.sd, fixtures=list(SRC.fixtures)
            )
        }
    BENCH_POS = set(
        game.players[p].position for p in DEC.bench if VIEWS[GWS[0]].get_xp(p) > 0
    )
    BLANK = min(
        (p for p in DEC.lineup if game.players[p].position in BENCH_POS),
        key=lambda p: (VIEWS[GWS[0]].get_xp(p), p),
    )
    EDGE.projections[BLANK][GWS[0]].fixtures = []
    EDGE.projections[BLANK][GWS[0]].xp = 0.0
    DOUBLE = DEC.captain
    SRC = EDGE.projections[DOUBLE][GWS[0]]
    SRC.fixtures = list(SRC.fixtures) + list(SRC.fixtures)
    SRC.xp = SRC.xp * 2

    EDGE_VIEW = build_views(EDGE, game, [GWS[0]])[GWS[0]]
    E_LINEUP, E_BENCH, E_CAP, _ = pick_lineup(DEC.squad, EDGE, GWS[0], game, view=EDGE_VIEW)
    P1 = PS.projections[DOUBLE][GWS[0]].fixtures[0].p_appear
    gate(
        "blank gameweek benches the player with no fixture",
        BLANK not in E_LINEUP,
        "%s (0 fixtures) dropped from the XI" % game.players[BLANK].web_name,
    )
    gate(
        "double gameweek sums both fixtures",
        EDGE_VIEW.n_fixtures[DOUBLE] == 2
        and abs(EDGE_VIEW.xmins[DOUBLE] - 2 * VIEWS[GWS[0]].xmins[DOUBLE]) < 1e-9
        and abs(EDGE_VIEW.get_appear(DOUBLE) - (1 - (1 - P1) ** 2)) < 1e-9,
        "%s: %.0f mins, p_appear %.4f -> %.4f"
        % (
            game.players[DOUBLE].web_name,
            EDGE_VIEW.xmins[DOUBLE],
            P1,
            EDGE_VIEW.get_appear(DOUBLE),
        ),
    )
    gate(
        "XI stays legal across a blank and a double",
        not lineup_problems(E_LINEUP, E_BENCH, DEC.squad, game) and E_CAP == DOUBLE,
        "%s, captain %s"
        % (
            formation_name([game.players[p].position for p in E_LINEUP]),
            game.players[E_CAP].web_name,
        ),
    )

    # Locks must survive the filter and the solve. Pick a forward the optimizer
    # did NOT want, so the lock actually has to bind.
    LOCK_IN = max(
        (
            p
            for p in POOL
            if game.players[p].position == scoring.FWD and p not in set(DEC.squad)
        ),
        key=lambda p: (game.players[p].now_cost, p),
    )
    LOCK_CFG = _Config.load()
    LOCK_CFG.optimizer.locked_in = [LOCK_IN]
    LOCK_CFG.optimizer.locked_out = [DEC.squad[0]]
    LOCK_DEC = pick_initial_squad(PS, game, LOCK_CFG, GWS, budget=args.budget)
    gate(
        "locked_in / locked_out respected",
        LOCK_IN in LOCK_DEC.squad and DEC.squad[0] not in LOCK_DEC.squad,
        "in %s, out %s"
        % (game.players[LOCK_IN].web_name, game.players[DEC.squad[0]].web_name),
    )

    # Chips only change the scoring, never the legality of the eleven.
    for CHIP in ("bboost", "3xc"):
        L, B, C, V = pick_lineup(DEC.squad, PS, GWS[0], game, chip=CHIP)
        PTS = lineup_expected_points(L, B, C, VIEWS[GWS[0]], CHIP)
        gate(
            "chip %-6s keeps a legal XI" % CHIP,
            not lineup_problems(L, B, DEC.squad, game),
            "%.2f xP vs %.2f plain" % (PTS, DEC.expected_points),
        )
    BB = lineup_expected_points(
        DEC.lineup, DEC.bench, DEC.captain, VIEWS[GWS[0]], "bboost"
    )
    TC = lineup_expected_points(DEC.lineup, DEC.bench, DEC.captain, VIEWS[GWS[0]], "3xc")
    gate(
        "bboost adds exactly the bench",
        abs(BB - DEC.expected_points - sum(VIEWS[GWS[0]].get_xp(p) for p in DEC.bench)) < 1e-9,
        "+%.2f" % (BB - DEC.expected_points),
    )
    gate(
        "3xc adds exactly one more captain",
        abs(TC - DEC.expected_points - VIEWS[GWS[0]].get_xp(DEC.captain)) < 1e-9,
        "+%.2f" % (TC - DEC.expected_points),
    )

    print("\n%d gate(s) failed%s" % (len(FAILURES), (": " + ", ".join(FAILURES)) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(_main())
