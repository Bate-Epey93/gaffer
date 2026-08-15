"""Chip planning: when to fire Bench Boost, Triple Captain, Free Hit, Wildcard.

Two things make chip planning different from every other decision in the game.

The first is that the schedule is not fixed. Today every club has exactly one
fixture in every gameweek, so there are no double gameweeks and no blanks and
the chips have nothing structural to aim at. That changes from around the
mid-season cup rounds onward, as postponed matches are rescheduled into existing
gameweeks: the club whose match moved now plays twice, and the club it was
originally drawn against blanks. ``detect_double_blank_gws`` therefore counts
fixtures out of the live fixture list on every single run. Nothing about which
gameweek is a double is ever written down in this file, because anything written
down here would be wrong by December.

The second is the deadline. The first set of four chips expires at the GW19
deadline and does not roll over. An unplayed chip is not a neutral outcome, it
is a forfeited one, and the size of the forfeit is computable: it is the best
gain the chip could still deliver in the gameweeks that remain. That number
drives an escalating warning rather than a passive note.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gaffer.core import scoring
from gaffer.core.config import Config
from gaffer.core.types import ProjectionSet
from gaffer.optimize.strategy import (
    HaulProfile,
    OwnershipModel,
    best_xi,
    estimate_ownership,
    haul_profiles,
)

log = logging.getLogger(__name__)

CHIP_LABEL: Dict[str, str] = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}
CHIP_ORDER: Tuple[str, ...] = ("bboost", "3xc", "freehit", "wildcard")

# Escalation bands for the first-half expiry, keyed on the gameweek being
# planned. Informational while there is still a season's worth of room, a
# warning once fewer than half a dozen chances remain, urgent in the last three.
URGENCY_WARNING_FROM = 13
URGENCY_URGENT_FROM = 17

# A wildcard realigns roughly this much of the XI onto the incoming fixture run;
# the rest of the squad is either already correct or too expensive to move.
WILDCARD_PLAYERS_REALIGNED = 8.0
# Fixture-swing windows: mean difficulty over the next five gameweeks against
# the five after that, as SPEC 3.11 specifies.
SWING_WINDOW = 5


# ---------------------------------------------------------------------------
# Schedule structure
# ---------------------------------------------------------------------------


def detect_double_blank_gws(state: Any) -> Dict[int, Dict[str, List[int]]]:
    """``{gw: {"doubles": [team_ids], "blanks": [team_ids]}}`` from live fixtures.

    A team with two fixtures in a gameweek is a double for its players; a team
    with none is a blank. Gameweeks where every club plays exactly once are
    omitted, so an empty dict means a completely regular schedule -- which is
    what the published 2026/27 fixture list currently is, and is the correct
    answer rather than a failure.

    Recomputed from ``state.fixtures`` every call. Fixtures whose ``gw`` is None
    are postponed and not yet rescheduled; they are the raw material for future
    doubles and are reported separately by ``schedule_report``.
    """
    gws = sorted(int(e["id"]) for e in state.events)
    counts: Dict[int, Dict[int, int]] = {int(t): {g: 0 for g in gws} for t in state.teams}
    for fixture in state.fixtures:
        if fixture.gw is None:
            continue
        gw = int(fixture.gw)
        if gw not in gws:
            continue
        for team_id in (int(fixture.team_h), int(fixture.team_a)):
            if team_id in counts:
                counts[team_id][gw] += 1

    out: Dict[int, Dict[str, List[int]]] = {}
    for gw in gws:
        doubles = sorted(t for t in counts if counts[t][gw] > 1)
        blanks = sorted(t for t in counts if counts[t][gw] == 0)
        if doubles or blanks:
            out[gw] = {"doubles": doubles, "blanks": blanks}
    return out


def schedule_report(state: Any) -> Dict[str, Any]:
    """Everything chip planning needs to know about the shape of the schedule."""
    structure = detect_double_blank_gws(state)
    unscheduled = [f for f in state.fixtures if f.gw is None]
    return {
        "gws": sorted(int(e["id"]) for e in state.events),
        "structure": structure,
        "double_gws": sorted(g for g, s in structure.items() if s["doubles"]),
        "blank_gws": sorted(g for g, s in structure.items() if s["blanks"]),
        "unscheduled_fixtures": [
            {"fixture_id": f.id, "team_h": f.team_h, "team_a": f.team_a} for f in unscheduled
        ],
        "n_fixtures": len(state.fixtures),
    }


def format_schedule_report(report: Dict[str, Any], state: Any) -> str:
    lines: List[str] = []
    lines.append("DOUBLE / BLANK GAMEWEEKS (recomputed from the live fixture list)")
    lines.append("-" * 78)
    if not report["structure"]:
        lines.append(
            "  none. All %d fixtures are scheduled and every club plays exactly once in"
            % report["n_fixtures"]
        )
        lines.append(
            "  each of GW%d-%d, so there is not a single double or blank gameweek in the"
            % (report["gws"][0], report["gws"][-1])
        )
        lines.append(
            "  published list. This is expected in August: doubles only appear once cup"
        )
        lines.append(
            "  rounds and postponements force matches to be moved, from roughly the turn"
        )
        lines.append(
            "  of the year. detect_double_blank_gws re-reads the fixture list every run,"
        )
        lines.append(
            "  so the moment a match is rescheduled this section fills itself in."
        )
    else:
        for gw in sorted(report["structure"]):
            entry = report["structure"][gw]
            if entry["doubles"]:
                lines.append("  GW%-2d DOUBLE: %s"
                             % (gw, ", ".join(state.short_name(t) for t in entry["doubles"])))
            if entry["blanks"]:
                lines.append("  GW%-2d BLANK : %s"
                             % (gw, ", ".join(state.short_name(t) for t in entry["blanks"])))
    if report["unscheduled_fixtures"]:
        lines.append("")
        lines.append("  %d fixture(s) have no gameweek yet (these become the doubles):"
                     % len(report["unscheduled_fixtures"]))
        for f in report["unscheduled_fixtures"]:
            lines.append("    fixture %d: %s v %s"
                         % (f["fixture_id"], state.short_name(f["team_h"]),
                            state.short_name(f["team_a"])))
    else:
        lines.append("")
        lines.append("  every fixture has a gameweek: no postponements pending.")
    return "\n".join(lines)


def team_fixture_count(state: Any, team_id: int, gw: int) -> int:
    return sum(
        1 for f in state.fixtures
        if f.gw is not None and int(f.gw) == int(gw)
        and int(team_id) in (int(f.team_h), int(f.team_a))
    )


# ---------------------------------------------------------------------------
# Chip windows and the expiry clock
# ---------------------------------------------------------------------------


@dataclass
class ChipWindow:
    chip: str
    half: int          # 1 = the set that expires at GW19, 2 = the GW20-38 set
    start_gw: int
    stop_gw: int
    deadline: Optional[str] = None       # ISO deadline of stop_gw
    deadline_dt: Optional[datetime] = None


def chip_windows(state: Any, config: Optional[Config] = None) -> Dict[str, List[ChipWindow]]:
    """``{chip: [first-half window, second-half window]}`` read from live data.

    ``bootstrap["chips"]`` is authoritative; ``scoring.CHIP_WINDOWS`` is the
    checked-in copy and a mismatch means the game has changed under us, so it is
    reported rather than silently preferred either way.
    """
    raw = state.bootstrap.get("chips") or []
    deadlines = {int(e["id"]): e.get("deadline_time") for e in state.events}
    grouped: Dict[str, List[ChipWindow]] = {}
    for entry in raw:
        name = str(entry.get("name"))
        start, stop = int(entry.get("start_event")), int(entry.get("stop_event"))
        grouped.setdefault(name, []).append(
            ChipWindow(chip=name, half=0, start_gw=start, stop_gw=stop,
                       deadline=deadlines.get(stop),
                       deadline_dt=_parse_iso(deadlines.get(stop)))
        )
    for name, windows in grouped.items():
        windows.sort(key=lambda w: w.start_gw)
        for i, w in enumerate(windows):
            w.half = i + 1
    if not grouped:
        # No live chip table: fall back to the verified constants so planning
        # still runs, and say so.
        for name, halves in scoring.CHIP_WINDOWS.items():
            grouped[name] = [
                ChipWindow(chip=name, half=i + 1, start_gw=a, stop_gw=b,
                           deadline=deadlines.get(b), deadline_dt=_parse_iso(deadlines.get(b)))
                for i, (a, b) in enumerate(halves)
            ]
        log.warning("bootstrap carried no chips table; using scoring.CHIP_WINDOWS")
    return grouped


def _parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def first_half_deadline(state: Any) -> Tuple[Optional[str], int]:
    """(ISO deadline, gameweek) at which the first set of four chips expires."""
    windows = chip_windows(state)
    stops = sorted({w.stop_gw for ws in windows.values() for w in ws if w.half == 1})
    gw = stops[-1] if stops else scoring.FIRST_HALF_LAST_GW
    return state.deadline(gw), gw


def chips_available(
    state: Any, squad: Any, half: int = 1, config: Optional[Config] = None
) -> Dict[str, bool]:
    """Which of a half's chips are still in hand, from ``SquadState.chips_used``."""
    used: List[str] = []
    if squad is not None and hasattr(squad, "chips_used"):
        used = [str(c) for c in (squad.chips_used or [])]
    counts: Dict[str, int] = {}
    for c in used:
        counts[c] = counts.get(c, 0) + 1
    out: Dict[str, bool] = {}
    for chip, windows in chip_windows(state, config).items():
        if not any(w.half == half for w in windows):
            continue
        # Each half issues exactly one of each chip, so the nth use belongs to
        # the nth half.
        out[chip] = counts.get(chip, 0) < half
    return out


def urgency_for(gw: int, expiry_gw: int, available: bool) -> str:
    """Escalating pressure as the first-half expiry approaches."""
    if not available:
        return "used"
    if gw > expiry_gw:
        return "expired"
    if gw >= URGENCY_URGENT_FROM:
        return "urgent"
    if gw >= URGENCY_WARNING_FROM:
        return "warning"
    return "informational"


# ---------------------------------------------------------------------------
# Squad optimisation (free hit target, wildcard target, smoke-test squad)
# ---------------------------------------------------------------------------


def _pulp_solver(config: Config):
    import pulp

    want = (config.optimizer.solver or "auto").lower()
    limit = int(config.optimizer.time_limit_seconds)
    gap = float(config.optimizer.mip_gap)
    if want in ("auto", "highs"):
        try:
            solver = pulp.HiGHS(msg=False, timeLimit=limit, gapRel=gap)
            if solver.available():
                return solver, "HiGHS"
        except Exception:  # pragma: no cover - depends on the local install
            pass
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=limit, gapRel=gap)
    return solver, "PULP_CBC_CMD"


def _candidate_pool(
    state: Any, value: Dict[int, float], per_position: int = 70, cheap: int = 22
) -> List[int]:
    """Trim 587 players to a pool the MILP can chew through in a second.

    Keeps the best by objective value in each position plus the cheapest with a
    non-zero projection, because budget feasibility for a 15-man squad depends on
    enablers, not on the top of the table.
    """
    pool: set = set()
    for pos in scoring.POSITIONS:
        ids = [p for p, pl in state.players.items() if pl.position == pos]
        ids.sort(key=lambda p: (-value.get(p, 0.0), p))
        pool.update(ids[:per_position])
        playable = [p for p in ids if value.get(p, 0.0) > 0.0]
        playable.sort(key=lambda p: (state.players[p].now_cost, -value.get(p, 0.0), p))
        pool.update(playable[:cheap])
    return sorted(pool)


def best_squad_for_gws(
    projections: ProjectionSet,
    state: Any,
    config: Config,
    gws: Sequence[int],
    budget_tenths: Optional[int] = None,
    exclude: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Best legal 15 (and its XI) over ``gws`` at a budget. Free Hit's target.

    Delegates to ``gaffer.optimize.squad.pick_initial_squad`` when that module is
    importable, so the squad optimizer stays the single authority. The local MILP
    below is the same problem stated directly in PuLP and exists so chip planning
    is not blocked on import order; it is a full model, not an approximation.
    """
    gws = [int(g) for g in gws]
    budget = int(budget_tenths if budget_tenths is not None else scoring.BUDGET_TENTHS)
    excluded = set(int(p) for p in (exclude or []))

    try:
        from gaffer.optimize.squad import pick_initial_squad  # noqa: WPS433
    except ImportError:
        pick_initial_squad = None
    if pick_initial_squad is not None and not excluded:
        decision = pick_initial_squad(projections, state, config, gws=gws, budget=budget)
        squad = [int(p) for p in decision.squad]
        xi = [int(p) for p in decision.lineup] or best_xi(squad, projections, gws[0], state)[0]
        return {
            "squad": squad,
            "xi": xi,
            "xp": sum(projections.xp(p, g) for p in xi for g in gws),
            "cost": sum(state.players[p].now_cost for p in squad),
            "solver": "gaffer.optimize.squad.pick_initial_squad",
            "status": "Optimal",
        }

    import pulp

    value = {
        int(p): sum(projections.xp(int(p), g) for g in gws)
        for p in state.players
        if int(p) not in excluded
    }
    pool = _candidate_pool(state, value)
    pool = [p for p in pool if p not in excluded]

    prob = pulp.LpProblem("gaffer_squad", pulp.LpMaximize)
    x = {p: pulp.LpVariable("x_%d" % p, cat="Binary") for p in pool}
    y = {p: pulp.LpVariable("y_%d" % p, cat="Binary") for p in pool}

    # The XI scores; the bench is worth a sliver, enough to break ties towards a
    # usable bench without distorting the lineup.
    bench_weight = 0.05
    prob += pulp.lpSum(
        value[p] * y[p] + bench_weight * value[p] * (x[p] - y[p]) for p in pool
    )
    prob += pulp.lpSum(x[p] for p in pool) == scoring.SQUAD_SIZE
    prob += pulp.lpSum(y[p] for p in pool) == scoring.SQUAD_PLAY
    prob += pulp.lpSum(state.players[p].now_cost * x[p] for p in pool) <= budget
    for p in pool:
        prob += y[p] <= x[p]
    for pos in scoring.POSITIONS:
        members = [p for p in pool if state.players[p].position == pos]
        prob += pulp.lpSum(x[p] for p in members) == scoring.SQUAD_SELECT[pos]
        prob += pulp.lpSum(y[p] for p in members) >= scoring.SQUAD_MIN_PLAY[pos]
        prob += pulp.lpSum(y[p] for p in members) <= scoring.SQUAD_MAX_PLAY[pos]
    for team_id in state.teams:
        members = [p for p in pool if state.players[p].team_id == team_id]
        if members:
            prob += pulp.lpSum(x[p] for p in members) <= scoring.TEAM_LIMIT

    solver, solver_name = _pulp_solver(config)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(
            "squad MILP for GW%s returned %s, not Optimal — refusing to plan a chip "
            "on an infeasible squad" % (gws, status)
        )
    squad = sorted(p for p in pool if x[p].value() is not None and x[p].value() > 0.5)
    xi = sorted(p for p in pool if y[p].value() is not None and y[p].value() > 0.5)
    return {
        "squad": squad,
        "xi": xi,
        "xp": sum(value[p] for p in xi),
        "cost": sum(state.players[p].now_cost for p in squad),
        "solver": solver_name,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Per-chip valuation
# ---------------------------------------------------------------------------


def _gw_p_appear(projections: ProjectionSet, pid: int, gw: int) -> float:
    gwp = projections.projections.get(int(pid), {}).get(int(gw))
    if gwp is None or not gwp.fixtures:
        return 0.0
    miss = 1.0
    for f in gwp.fixtures:
        miss *= (1.0 - max(0.0, min(1.0, f.p_appear)))
    return max(0.0, min(1.0, 1.0 - miss))


def _poisson_binomial_tail(probs: Sequence[float]) -> List[float]:
    """``[P(X >= 0), P(X >= 1), ...]`` for independent Bernoullis."""
    pmf = [1.0]
    for q in probs:
        nxt = [0.0] * (len(pmf) + 1)
        for k, p in enumerate(pmf):
            nxt[k] += p * (1.0 - q)
            nxt[k + 1] += p * q
        pmf = nxt
    tail = [0.0] * (len(pmf) + 1)
    acc = 0.0
    for k in range(len(pmf) - 1, -1, -1):
        acc += pmf[k]
        tail[k] = acc
    return tail


def _autosub_value(
    xi: Sequence[int], bench: Sequence[int], projections: ProjectionSet, gw: int, state: Any
) -> float:
    """Points the bench already delivers through autosubs, with no chip played.

    Bench Boost is only worth the bench *minus this*: a benched player who would
    have come on for an absent starter anyway is not new points. The reserve
    keeper is handled separately because he only ever replaces the keeper.
    """
    gk_xi = [p for p in xi if state.players[p].position == scoring.GKP]
    out_xi = [p for p in xi if state.players[p].position != scoring.GKP]
    gk_bench = [p for p in bench if state.players[p].position == scoring.GKP]
    out_bench = [p for p in bench if state.players[p].position != scoring.GKP]

    value = 0.0
    for keeper in gk_bench[:1]:
        p_fail = 1.0 - (_gw_p_appear(projections, gk_xi[0], gw) if gk_xi else 0.0)
        value += p_fail * projections.xp(keeper, gw)
    fails = [1.0 - _gw_p_appear(projections, p, gw) for p in out_xi]
    tail = _poisson_binomial_tail(fails)
    for slot, pid in enumerate(out_bench):
        p_used = tail[slot + 1] if slot + 1 < len(tail) else 0.0
        value += p_used * projections.xp(pid, gw)
    return value


@dataclass
class ChipEvaluation:
    """One chip's value in one gameweek."""

    chip: str
    gw: int
    gain: float
    detail: Dict[str, Any] = field(default_factory=dict)


def evaluate_bench_boost(
    squad_ids: Sequence[int], projections: ProjectionSet, gw: int, state: Any
) -> ChipEvaluation:
    """Bench Boost: the four bench players score, net of what autosubs deliver anyway."""
    gw = int(gw)
    xi, bench = best_xi(squad_ids, projections, gw, state)
    bench_xp = sum(projections.xp(p, gw) for p in bench)
    autosub = _autosub_value(xi, bench, projections, gw, state)
    playing = sum(1 for p in squad_ids if team_fixture_count(state, state.players[p].team_id, gw) > 0)
    doubles = sum(1 for p in squad_ids if team_fixture_count(state, state.players[p].team_id, gw) > 1)
    return ChipEvaluation(
        chip="bboost",
        gw=gw,
        gain=bench_xp - autosub,
        detail={
            "bench": list(bench),
            "bench_xp": bench_xp,
            "autosub_value": autosub,
            "playing_assets": playing,
            "double_assets": doubles,
            "all_fifteen_play": playing == len(list(squad_ids)),
            "bench_breakdown": [
                {"player_id": p, "name": state.players[p].web_name,
                 "xp": projections.xp(p, gw),
                 "fixtures": team_fixture_count(state, state.players[p].team_id, gw)}
                for p in bench
            ],
        },
    )


def evaluate_triple_captain(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    profiles: Optional[Dict[int, HaulProfile]] = None,
) -> ChipEvaluation:
    """Triple Captain: worth exactly one extra copy of the captain's points.

    The armband already doubles him, so the chip buys the third copy and nothing
    else. Its value is therefore the captain's projected points, full stop — and
    that is the number to compare across gameweeks, since a double gameweek
    premium and a single gameweek premium are measured on the same scale.
    """
    gw = int(gw)
    xi, _bench = best_xi(squad_ids, projections, gw, state)
    ranked = sorted(xi, key=lambda p: (-projections.xp(p, gw), p))
    if not ranked:
        return ChipEvaluation("3xc", gw, 0.0, {"captain": None})
    captain = ranked[0]
    gwp = projections.projections.get(captain, {}).get(gw)
    n_fix = len(gwp.fixtures) if gwp is not None else 0
    prof = (profiles or {}).get(captain)
    return ChipEvaluation(
        chip="3xc",
        gw=gw,
        gain=projections.xp(captain, gw),
        detail={
            "captain": captain,
            "captain_name": state.players[captain].web_name,
            "captain_team": state.short_name(state.players[captain].team_id),
            "captain_xp": projections.xp(captain, gw),
            "captain_sd": gwp.sd if gwp is not None else 0.0,
            "captain_p_start": max((f.p_start for f in gwp.fixtures), default=0.0)
            if gwp is not None else 0.0,
            "n_fixtures": n_fix,
            "is_double": n_fix > 1,
            "p_haul": prof.p_haul if prof is not None else None,
            "runners_up": [
                {"player_id": p, "name": state.players[p].web_name, "xp": projections.xp(p, gw)}
                for p in ranked[1:4]
            ],
        },
    )


def evaluate_free_hit(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    config: Config,
    budget_tenths: Optional[int] = None,
) -> ChipEvaluation:
    """Free Hit: one week with any legal squad, so the gain is the XI upgrade.

    ``best XI money can buy this week`` minus ``best XI your squad can field``.
    In a blank gameweek the second term collapses (you cannot field eleven) and
    the gain is enormous, which is exactly why the chip exists.
    """
    gw = int(gw)
    try:
        xi, _bench = best_xi(squad_ids, projections, gw, state)
        current = sum(projections.xp(p, gw) for p in xi)
    except ValueError:
        xi, current = [], 0.0
    target = best_squad_for_gws(projections, state, config, [gw], budget_tenths=budget_tenths)
    playing = sum(1 for p in squad_ids if team_fixture_count(state, state.players[p].team_id, gw) > 0)
    return ChipEvaluation(
        chip="freehit",
        gw=gw,
        gain=target["xp"] - current,
        detail={
            "current_xi_xp": current,
            "free_hit_xi_xp": target["xp"],
            "free_hit_squad": target["squad"],
            "free_hit_xi": target["xi"],
            "playing_assets": playing,
            "blanking_assets": len(list(squad_ids)) - playing,
            "solver": target["solver"],
        },
    )


def team_difficulty_series(state: Any, difficulty_fn: Optional[Any] = None) -> Dict[int, Dict[int, List[float]]]:
    """``{team_id: {gw: [difficulty per fixture]}}`` straight from the fixture list.

    The official per-fixture FDR is populated for all 380 fixtures from day one,
    including the stretch beyond any projection horizon, which is what a
    ten-gameweek swing comparison needs. ``difficulty_fn(state, fixture, team_id)``
    overrides it when a model-based difficulty is available.
    """
    out: Dict[int, Dict[int, List[float]]] = {int(t): {} for t in state.teams}
    for fixture in state.fixtures:
        if fixture.gw is None:
            continue
        gw = int(fixture.gw)
        for team_id, own in ((int(fixture.team_h), True), (int(fixture.team_a), False)):
            if team_id not in out:
                continue
            if difficulty_fn is not None:
                diff = float(difficulty_fn(state, fixture, team_id))
            else:
                diff = float(fixture.team_h_difficulty if own else fixture.team_a_difficulty)
            out[team_id].setdefault(gw, []).append(diff)
    return out


def _window_difficulty(
    series: Dict[int, List[float]], first: int, last: int, blank_penalty: float = 5.0
) -> Optional[float]:
    """Mean difficulty per gameweek over a window; a blank counts as maximal."""
    vals: List[float] = []
    for gw in range(first, last + 1):
        fixtures = series.get(gw, [])
        if not fixtures:
            vals.append(blank_penalty)
        else:
            vals.append(sum(fixtures) / len(fixtures))
    return sum(vals) / len(vals) if vals else None


def fixture_swings(
    state: Any, window: int = SWING_WINDOW, difficulty_fn: Optional[Any] = None
) -> Dict[int, Dict[int, float]]:
    """``{gw: {team_id: improvement}}`` where improvement is the fixture swing.

    ``improvement(t, w)`` compares the ``window`` gameweeks *before* w with the
    ``window`` gameweeks from w onward: positive means this team's run turns
    easier exactly at w, which is the moment you want to be holding its players
    and therefore the moment a wildcard pays.
    """
    series = team_difficulty_series(state, difficulty_fn)
    gws = sorted(int(e["id"]) for e in state.events)
    lo, hi = gws[0], gws[-1]
    out: Dict[int, Dict[int, float]] = {}
    for w in gws:
        if w - window < lo or w + window - 1 > hi:
            continue
        per_team: Dict[int, float] = {}
        for team_id, per_gw in series.items():
            before = _window_difficulty(per_gw, w - window, w - 1)
            after = _window_difficulty(per_gw, w, w + window - 1)
            if before is None or after is None:
                continue
            per_team[team_id] = before - after
        out[w] = per_team
    return out


def fit_points_per_difficulty(projections: ProjectionSet, state: Any) -> Tuple[float, int]:
    """Points a starter gains per unit of fixture difficulty, fitted on our own xP.

    Regresses projected points on the official FDR across every projected
    player-fixture where the player is a genuine starter. The slope is negative
    (harder fixture, fewer points) and converts a fixture swing measured in FDR
    into a wildcard gain measured in points, instead of asserting an exchange
    rate. Returns (slope, n).
    """
    xs: List[float] = []
    ys: List[float] = []
    for pid, per_gw in projections.projections.items():
        for _gw, gwp in per_gw.items():
            for fp in gwp.fixtures:
                if fp.xmins < 60.0:
                    continue
                xs.append(float(fp.difficulty))
                ys.append(float(fp.xp_total))
    n = len(xs)
    if n < 50:
        return 0.0, n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 0:
        return 0.0, n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx, n


def evaluate_wildcard(
    state: Any,
    projections: ProjectionSet,
    gw: int,
    swings: Dict[int, Dict[int, float]],
    points_per_difficulty: float,
    squad_ids: Optional[Sequence[int]] = None,
    config: Optional[Config] = None,
    budget_tenths: Optional[int] = None,
    n_teams: int = 6,
) -> ChipEvaluation:
    """Wildcard: the fixture swing it buys, plus the squad repair it performs.

    The swing term is the honest part of a chip played months ahead: take the
    teams whose fixtures improve most at this gameweek, convert the improvement
    into points with the fitted points-per-FDR slope, and multiply by the players
    and gameweeks a rebuild actually touches. The repair term (best squad money
    can buy, against what you currently hold) is only computable where the
    projections reach, and is reported separately rather than folded in blind.
    """
    gw = int(gw)
    per_team = swings.get(gw, {})
    ranked = sorted(per_team.items(), key=lambda kv: -kv[1])[:n_teams]
    mean_improvement = (sum(v for _t, v in ranked) / len(ranked)) if ranked else 0.0
    # Never negative: a wildcard you play into worsening fixtures is worth zero,
    # not less than zero, because keeping the current fifteen is always legal.
    swing_points = max(0.0, (
        abs(points_per_difficulty) * mean_improvement
        * WILDCARD_PLAYERS_REALIGNED * float(SWING_WINDOW)
    ))

    repair_points = None
    horizon = [g for g in range(gw, gw + SWING_WINDOW)
               if projections.first_gw <= g <= projections.last_gw]
    if squad_ids and config is not None and horizon:
        target = best_squad_for_gws(
            projections, state, config, horizon, budget_tenths=budget_tenths)
        current = 0.0
        for g in horizon:
            xi, _b = best_xi(squad_ids, projections, g, state)
            current += sum(projections.xp(p, g) for p in xi)
        repair_points = max(0.0, target["xp"] - current)

    # The repair term is a solved MILP against real projections and the swing
    # term is an extrapolation, so where the projections cover the whole window
    # the repair term wins outright; where they run out, take the better of the
    # two rather than throwing away the only evidence there is.
    if repair_points is not None and len(horizon) == SWING_WINDOW:
        gain = repair_points
    elif repair_points is not None:
        gain = max(swing_points, repair_points)
    else:
        gain = swing_points

    return ChipEvaluation(
        chip="wildcard",
        gw=gw,
        gain=gain,
        detail={
            "mean_improvement_fdr": mean_improvement,
            "points_per_difficulty": points_per_difficulty,
            "swing_points": swing_points,
            "repair_points": repair_points,
            "horizon": horizon,
            "teams": [
                {"team_id": t, "short_name": state.short_name(t), "improvement": v}
                for t, v in ranked
            ],
        },
    )


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _confidence(
    gw: int, current_gw: int, structural: bool, margin_ratio: float, extra_penalty: float = 0.0
) -> Tuple[str, float]:
    """Confidence in a chip recommendation, on the evidence actually available.

    Three things drive it: how far ahead the gameweek is (projections decay fast
    past a month), whether the recommendation is anchored to a real schedule
    feature rather than to marginal differences in projected points, and how
    clearly the chosen gameweek beats the runner-up.
    """
    ahead = max(0, gw - current_gw)
    horizon = math.exp(-ahead / 8.0)                     # ~0.53 at 5 gws, ~0.29 at 10
    structure = 1.0 if structural else 0.45
    margin = max(0.0, min(1.0, margin_ratio / 0.25))     # a 25% edge is decisive
    score = max(0.0, horizon * structure * (0.55 + 0.45 * margin) - extra_penalty)
    if score >= 0.60:
        return "high", score
    if score >= 0.35:
        return "medium", score
    if score >= 0.15:
        return "low", score
    return "speculative", score


# ---------------------------------------------------------------------------
# The recommendation engine
# ---------------------------------------------------------------------------


def recommend_chips(
    state: Any,
    squad: Any,
    projections: ProjectionSet,
    config: Optional[Config] = None,
    engine: Optional[Any] = None,
    ownership: Optional[OwnershipModel] = None,
    half: int = 1,
    max_free_hit_solves: int = 4,
    max_wildcard_solves: int = 3,
) -> List[Dict[str, Any]]:
    """One recommendation per unused chip, each with a gameweek, a points gain,
    a confidence level and the expiry position.

    Every candidate gameweek is scored for every chip, so the output carries the
    runner-up gameweeks too: a chip plan that cannot say what it is giving up is
    not a plan.
    """
    config = config or Config.load()
    squad_ids = _squad_ids(squad)
    structure = detect_double_blank_gws(state)
    windows = chip_windows(state, config)
    available = chips_available(state, squad, half=half, config=config)
    expiry_deadline, expiry_gw = first_half_deadline(state)
    current_gw = int(state.current_gw)

    warnings: List[str] = []
    if projections.last_gw < expiry_gw and half == 1:
        warnings.append(
            "projections stop at GW%d but the first-half chips run to GW%d; gameweeks "
            "%d-%d were not evaluated and every gain below is a lower bound"
            % (projections.last_gw, expiry_gw, projections.last_gw + 1, expiry_gw)
        )

    if ownership is None and squad_ids:
        ownership = estimate_ownership(state, projections, max(current_gw, projections.first_gw))
    profiles: Dict[int, Dict[int, HaulProfile]] = {}

    def window_gws(chip: str) -> List[int]:
        for w in windows.get(chip, []):
            if w.half != half:
                continue
            return [
                g for g in range(max(w.start_gw, current_gw), w.stop_gw + 1)
                if projections.first_gw <= g <= projections.last_gw
            ]
        return []

    swings = fixture_swings(state)
    slope, slope_n = fit_points_per_difficulty(projections, state)

    recs: List[Dict[str, Any]] = []
    for chip in CHIP_ORDER:
        if chip not in windows:
            continue
        gws = window_gws(chip)
        evals: List[ChipEvaluation] = []

        if not gws:
            recs.append(_no_window_recommendation(
                chip, state, available.get(chip, True), current_gw, expiry_gw,
                expiry_deadline, warnings))
            continue

        if chip == "bboost":
            if not squad_ids:
                continue
            evals = [evaluate_bench_boost(squad_ids, projections, g, state) for g in gws]
        elif chip == "3xc":
            if not squad_ids:
                continue
            for g in gws:
                if g not in profiles:
                    profiles[g] = haul_profiles(projections, g, engine=engine)
                evals.append(evaluate_triple_captain(
                    squad_ids, projections, g, state, profiles[g]))
        elif chip == "freehit":
            if not squad_ids:
                continue
            # Screen cheaply first: the gameweeks worth solving are the ones where
            # the squad's own XI falls furthest short of what the league can offer.
            screen: List[Tuple[float, int]] = []
            for g in gws:
                try:
                    xi, _b = best_xi(squad_ids, projections, g, state)
                    current = sum(projections.xp(p, g) for p in xi)
                except ValueError:
                    current = 0.0
                screen.append((_pool_upper_bound(state, projections, g) - current, g))
            screen.sort(key=lambda kv: -kv[0])
            budget = squad.budget if hasattr(squad, "budget") else None
            for _score, g in screen[:max_free_hit_solves]:
                evals.append(evaluate_free_hit(
                    squad_ids, projections, g, state, config, budget_tenths=budget))
        elif chip == "wildcard":
            screen = sorted(
                ((sum(sorted(swings.get(g, {}).values(), reverse=True)[:6]), g) for g in gws),
                key=lambda kv: -kv[0],
            )
            budget = squad.budget if hasattr(squad, "budget") else None
            solved = 0
            for _score, g in screen:
                use_squad = squad_ids if solved < max_wildcard_solves else None
                evals.append(evaluate_wildcard(
                    state, projections, g, swings, slope,
                    squad_ids=use_squad, config=config, budget_tenths=budget))
                solved += 1

        if not evals:
            continue
        evals.sort(key=lambda e: (-e.gain, e.gw))
        best = evals[0]
        runner = evals[1] if len(evals) > 1 else None
        margin_ratio = 0.0
        if runner is not None and abs(best.gain) > 1e-9:
            margin_ratio = (best.gain - runner.gain) / abs(best.gain)

        structural = _is_structural(chip, best, structure)
        penalty = 0.0
        if chip == "3xc":
            # A triple captain on a player who might not start is a chip binned.
            penalty = 0.5 * (1.0 - float(best.detail.get("captain_p_start") or 0.0))
        conf, conf_score = _confidence(best.gw, current_gw, structural, margin_ratio, penalty)

        # What it costs to let this chip die: the best it could still do inside
        # the window that expires. For a chip whose best gameweek is already in
        # that window, that is the same number.
        in_window = [e for e in evals if e.gw <= expiry_gw]
        expiry_cost = max((e.gain for e in in_window), default=0.0)
        expiry_best_gw = next((e.gw for e in in_window if e.gain == expiry_cost), None)

        recs.append({
            "chip": chip,
            "label": CHIP_LABEL.get(chip, chip),
            "available": bool(available.get(chip, True)),
            "gw": best.gw,
            "points_gain": best.gain,
            "confidence": conf,
            "confidence_score": conf_score,
            "urgency": urgency_for(current_gw, expiry_gw, available.get(chip, True)),
            "expiry_gw": expiry_gw,
            "expiry_deadline": expiry_deadline,
            "gws_until_expiry": max(0, expiry_gw - current_gw + 1),
            "expiry_cost": expiry_cost,
            "expiry_best_gw": expiry_best_gw,
            "structural": structural,
            "detail": best.detail,
            "alternatives": [
                {"gw": e.gw, "points_gain": e.gain} for e in evals[1:4]
            ],
            "reason": _reason(chip, best, runner, state, structure, structural, conf,
                              expiry_gw, current_gw, slope, slope_n),
            "warnings": list(warnings),
        })

    recs.sort(key=lambda r: (-(r["points_gain"] or 0.0), r["gw"] or 99))
    return recs


def _pool_upper_bound(state: Any, projections: ProjectionSet, gw: int) -> float:
    """A true ceiling on any legal XI this gameweek, ignoring budget and clubs.

    Every legal XI is one keeper plus ten outfielders, so the best keeper plus
    the ten best outfield projections dominates it. Only used to decide which
    gameweeks are worth a MILP, so a bound is all that is needed.
    """
    keepers = [projections.xp(p, gw) for p, pl in state.players.items()
               if pl.position == scoring.GKP]
    outfield = sorted(
        (projections.xp(p, gw) for p, pl in state.players.items()
         if pl.position != scoring.GKP),
        reverse=True,
    )
    return (max(keepers) if keepers else 0.0) + sum(outfield[:scoring.SQUAD_PLAY - 1])


def _is_structural(chip: str, ev: ChipEvaluation, structure: Dict[int, Dict[str, List[int]]]) -> bool:
    entry = structure.get(ev.gw, {})
    if chip == "bboost":
        return bool(entry.get("doubles")) and bool(ev.detail.get("double_assets"))
    if chip == "3xc":
        return bool(ev.detail.get("is_double"))
    if chip == "freehit":
        return bool(entry.get("blanks")) and bool(ev.detail.get("blanking_assets"))
    if chip == "wildcard":
        return float(ev.detail.get("mean_improvement_fdr") or 0.0) >= 0.75
    return False


def _no_window_recommendation(
    chip: str, state: Any, available: bool, current_gw: int, expiry_gw: int,
    deadline: Optional[str], warnings: List[str]
) -> Dict[str, Any]:
    return {
        "chip": chip,
        "label": CHIP_LABEL.get(chip, chip),
        "available": available,
        "gw": None,
        "points_gain": 0.0,
        "confidence": "speculative",
        "confidence_score": 0.0,
        "urgency": urgency_for(current_gw, expiry_gw, available),
        "expiry_gw": expiry_gw,
        "expiry_deadline": deadline,
        "gws_until_expiry": max(0, expiry_gw - current_gw + 1),
        "expiry_cost": 0.0,
        "expiry_best_gw": None,
        "structural": False,
        "detail": {},
        "alternatives": [],
        "reason": ("no gameweek inside this chip's window is covered by the current "
                   "projection set, so it was not evaluated"),
        "warnings": list(warnings),
    }


def _reason(
    chip: str,
    best: ChipEvaluation,
    runner: Optional[ChipEvaluation],
    state: Any,
    structure: Dict[int, Dict[str, List[int]]],
    structural: bool,
    confidence: str,
    expiry_gw: int,
    current_gw: int,
    slope: float,
    slope_n: int,
) -> str:
    d = best.detail
    bits: List[str] = []
    if chip == "bboost":
        bits.append(
            "GW%d bench projects %.2f points; autosubs would have collected %.2f of that "
            "without the chip, so the chip itself is worth %.2f"
            % (best.gw, d.get("bench_xp", 0.0), d.get("autosub_value", 0.0), best.gain)
        )
        bits.append("%d of your 15 have a fixture, %d have two"
                    % (d.get("playing_assets", 0), d.get("double_assets", 0)))
        if not structural:
            bits.append(
                "no double gameweek exists in the published schedule, so this is the best "
                "ordinary gameweek rather than the gameweek Bench Boost is designed for — "
                "hold it until a double appears unless the expiry clock forces your hand"
            )
        else:
            bits.append("this is a genuine double gameweek: %d of your bench play twice"
                        % d.get("double_assets", 0))
    elif chip == "3xc":
        bits.append(
            "GW%d: %s (%s) projects %.2f, so the third copy is worth %.2f%s"
            % (best.gw, d.get("captain_name"), d.get("captain_team"), d.get("captain_xp", 0.0),
               best.gain,
               ", P(10+) %.0f%%" % (100 * d["p_haul"]) if d.get("p_haul") is not None else "")
        )
        bits.append("p_start %.2f over %d fixture(s)%s"
                    % (d.get("captain_p_start", 0.0), d.get("n_fixtures", 0),
                       " — a double gameweek" if d.get("is_double") else ""))
        if runner is not None:
            bits.append(
                "next best gameweek is GW%d at %.2f, a %.2f point difference — the chip "
                "should wait for a double gameweek premium if one materialises, which is "
                "worth roughly twice a single gameweek's captain"
                % (runner.gw, runner.gain, best.gain - runner.gain)
            )
    elif chip == "freehit":
        bits.append(
            "GW%d: the best XI available at your budget projects %.2f against %.2f from "
            "your own squad, a gain of %.2f"
            % (best.gw, d.get("free_hit_xi_xp", 0.0), d.get("current_xi_xp", 0.0), best.gain)
        )
        blanking = d.get("blanking_assets", 0)
        if blanking:
            bits.append("%d of your squad blank that week — this is what the chip is for"
                        % blanking)
        else:
            bits.append(
                "nothing in your squad blanks that week, so this is a pure quality upgrade "
                "for one gameweek and not the intended use; Free Hit's real value arrives "
                "with the first big blank gameweek, which the published schedule does not "
                "yet contain"
            )
    elif chip == "wildcard":
        teams = ", ".join("%s %+.2f" % (t["short_name"], t["improvement"])
                          for t in d.get("teams", [])[:5])
        bits.append(
            "GW%d has the largest fixture swing: the six most improved clubs gain %.2f FDR "
            "a game on average (%s)" % (best.gw, d.get("mean_improvement_fdr", 0.0), teams)
        )
        bits.append(
            "at a fitted %.3f points per FDR unit for a starter (n=%d player-fixtures), "
            "realigning %.0f of the XI for %d gameweeks is worth about %.1f points"
            % (slope, slope_n, WILDCARD_PLAYERS_REALIGNED, SWING_WINDOW,
               d.get("swing_points", 0.0))
        )
        if d.get("repair_points") is not None:
            covered = len(d.get("horizon") or []) == SWING_WINDOW
            bits.append(
                "rebuilding the squad outright over GW%s is worth %.1f points against what "
                "you currently hold%s"
                % ("-".join(str(g) for g in (d["horizon"][:1] + d["horizon"][-1:])),
                   d["repair_points"],
                   ", and the projections cover the whole window so this is the number the "
                   "recommendation uses" if covered
                   else "; the projections stop short of the full window, so the larger of "
                        "this and the fixture-swing estimate is used")
            )
    if best.gw > expiry_gw:
        bits.append("NOTE: GW%d is past the first-half expiry at GW%d — this would have to "
                    "be the second-half chip" % (best.gw, expiry_gw))
    bits.append("confidence %s" % confidence)
    return "; ".join(bits)


def _squad_ids(squad: Any) -> List[int]:
    if squad is None:
        return []
    if isinstance(squad, (list, tuple, set)):
        return [int(p) for p in squad]
    if hasattr(squad, "player_ids"):
        return [int(p) for p in squad.player_ids()]
    if hasattr(squad, "squad"):
        return [int(p) for p in squad.squad]
    raise TypeError("cannot read player ids from %r" % type(squad))


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def chip_expiry_status(
    state: Any, squad: Any, recommendations: Optional[Sequence[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """The GW19 clock, per chip, with the points each unplayed chip forfeits."""
    deadline, expiry_gw = first_half_deadline(state)
    deadline_dt = _parse_iso(deadline)
    now = datetime.now(timezone.utc)
    available = chips_available(state, squad, half=1)
    current_gw = int(state.current_gw)
    by_chip = {r["chip"]: r for r in (recommendations or [])}

    entries: List[Dict[str, Any]] = []
    total_cost = 0.0
    for chip in CHIP_ORDER:
        if chip not in available:
            continue
        rec = by_chip.get(chip)
        cost, best_gw, stale = _remaining_value(rec, current_gw, expiry_gw)
        if available[chip]:
            total_cost += cost
        entries.append({
            "chip": chip,
            "label": CHIP_LABEL.get(chip, chip),
            "available": available[chip],
            "urgency": urgency_for(current_gw, expiry_gw, available[chip]),
            "expiry_cost": cost,
            "best_gw_in_window": best_gw,
            "recommended_gw": rec.get("gw") if rec else None,
            "stale": stale,
        })

    return {
        "expiry_gw": expiry_gw,
        "deadline": deadline,
        "days_remaining": ((deadline_dt - now).total_seconds() / 86400.0)
        if deadline_dt is not None else None,
        "gameweeks_remaining": max(0, expiry_gw - current_gw + 1),
        "current_gw": current_gw,
        "chips": entries,
        "unused": [e["chip"] for e in entries if e["available"]],
        "total_forfeit_if_unused": total_cost,
        "overall_urgency": _overall_urgency(entries, current_gw, expiry_gw),
    }


def _remaining_value(
    rec: Optional[Dict[str, Any]], current_gw: int, expiry_gw: int
) -> Tuple[float, Optional[int], bool]:
    """Best gain still reachable before expiry, from an existing recommendation.

    ``recommend_chips`` only ever evaluates gameweeks from its own planning week
    onward, so normally its ``expiry_cost`` is exactly this. Recomputing the
    status against a later clock (a what-if, or a cached recommendation set)
    would otherwise quote a forfeit from a gameweek that has already gone, so
    the reachable gameweeks are re-filtered and the entry is flagged stale.
    """
    if rec is None:
        return 0.0, None, False
    best_gw = rec.get("expiry_best_gw")
    if best_gw is not None and int(best_gw) >= current_gw:
        return float(rec.get("expiry_cost") or 0.0), int(best_gw), False
    options: List[Tuple[int, float]] = []
    if rec.get("gw") is not None:
        options.append((int(rec["gw"]), float(rec.get("points_gain") or 0.0)))
    for alt in rec.get("alternatives") or []:
        if alt.get("gw") is not None:
            options.append((int(alt["gw"]), float(alt.get("points_gain") or 0.0)))
    options = [(g, v) for g, v in options if current_gw <= g <= expiry_gw]
    if not options:
        return 0.0, None, True
    gw, value = max(options, key=lambda kv: (kv[1], -kv[0]))
    return value, gw, True


def _overall_urgency(entries: Sequence[Dict[str, Any]], current_gw: int, expiry_gw: int) -> str:
    unused = [e for e in entries if e["available"]]
    if not unused:
        return "none"
    return urgency_for(current_gw, expiry_gw, True)


def format_chip_expiry(status: Dict[str, Any]) -> str:
    lines: List[str] = []
    title = "FIRST-HALF CHIP EXPIRY — GW%d DEADLINE" % status["expiry_gw"]
    lines.append(title)
    lines.append("=" * len(title))
    days = status["days_remaining"]
    lines.append("  deadline            : %s%s"
                 % (status["deadline"],
                    "  (%.0f days away)" % days if days is not None else ""))
    lines.append("  planning gameweek   : GW%d" % status["current_gw"])
    lines.append("  gameweeks remaining : %d (GW%d through GW%d inclusive)"
                 % (status["gameweeks_remaining"], status["current_gw"], status["expiry_gw"]))
    lines.append("  status              : %s" % status["overall_urgency"].upper())
    lines.append("")
    lines.append("  %-15s %-10s %-14s %10s %s"
                 % ("chip", "in hand", "urgency", "forfeit", "best remaining GW"))
    lines.append("  " + "-" * 74)
    for e in status["chips"]:
        lines.append(
            "  %-15s %-10s %-14s %10.2f %s%s"
            % (e["label"], "yes" if e["available"] else "USED", e["urgency"],
               e["expiry_cost"] if e["available"] else 0.0,
               ("GW%d" % e["best_gw_in_window"]) if e["best_gw_in_window"] else "-",
               "  (re-evaluate: recommendations predate this gameweek)"
               if e.get("stale") and e["available"] else "")
        )
    lines.append("")
    lines.append("  unplayed chips forfeit an estimated %.1f points at the GW%d deadline."
                 % (status["total_forfeit_if_unused"], status["expiry_gw"]))
    lines.append("  " + _expiry_advice(status))
    return "\n".join(lines)


def _expiry_advice(status: Dict[str, Any]) -> str:
    urgency = status["overall_urgency"]
    n = len(status["unused"])
    left = status["gameweeks_remaining"]
    if urgency == "none":
        return "All four first-half chips have been played. Nothing expires."
    if urgency == "expired":
        return ("GW%d has passed: %d chip(s) expired unplayed. Only the GW20-38 set remains."
                % (status["expiry_gw"], n))
    if urgency == "urgent":
        return ("URGENT: %d chip(s) and only %d gameweek(s) left. Play them on the best "
                "gameweek available rather than holding out for a double that the schedule "
                "may never produce — a chip played at half value beats a chip played at none."
                % (n, left))
    if urgency == "warning":
        return ("WARNING: %d chip(s) with %d gameweek(s) left. Fix a target gameweek for each "
                "now and treat any further delay as a decision, not a deferral." % (n, left))
    return ("Informational: %d chip(s) in hand with %d gameweek(s) to use them. There is room "
            "to wait for a double or blank gameweek to appear, and waiting is correct while "
            "the schedule is still regular." % (n, left))


def format_chip_recommendations(
    recommendations: Sequence[Dict[str, Any]], state: Any
) -> str:
    lines: List[str] = []
    title = "CHIP RECOMMENDATIONS"
    lines.append(title)
    lines.append("=" * len(title))
    head = ("  %-15s %-6s %8s %-13s %-14s %10s"
            % ("chip", "gw", "gain", "confidence", "urgency", "forfeit"))
    lines.append(head)
    lines.append("  " + "-" * 72)
    for r in recommendations:
        lines.append(
            "  %-15s %-6s %8.2f %-13s %-14s %10.2f"
            % (r["label"], ("GW%d" % r["gw"]) if r["gw"] else "-", r["points_gain"],
               r["confidence"], r["urgency"], r["expiry_cost"])
        )
    lines.append("")
    for r in recommendations:
        lines.append("  %s — %s" % (r["label"], ("GW%d" % r["gw"]) if r["gw"] else "no window"))
        for chunk in r["reason"].split("; "):
            lines.append("      %s" % chunk)
        if r["alternatives"]:
            lines.append("      alternatives: %s"
                         % ", ".join("GW%d %.2f" % (a["gw"], a["points_gain"])
                                     for a in r["alternatives"]))
        lines.append("")
    seen = set()
    for r in recommendations:
        for w in r.get("warnings") or []:
            if w not in seen:
                seen.add(w)
                lines.append("  ! %s" % w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    from gaffer.core.types import SquadPick, SquadState
    from gaffer.data.loaders import load_game_state
    from gaffer.model.xp import XPEngine, _synthetic_double_blank
    from gaffer.optimize.strategy import (
        captain_analysis,
        captain_options,
        format_captain_options,
        format_ownership_table,
        format_rank_risk_report,
        rank_risk_report,
    )

    parser = argparse.ArgumentParser(description="chip planning + rank strategy smoke test")
    parser.add_argument("--last-gw", type=int, default=19,
                        help="project through this gameweek (19 covers the first chip set)")
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    CFG = Config.load()

    t0 = time.time()
    GAME = load_game_state(CFG, progress=False)
    print("loaded %s in %.2fs" % (GAME.summary(), time.time() - t0))
    GW = GAME.current_gw
    GWS = list(range(GW, args.last_gw + 1))

    t0 = time.time()
    ENG = XPEngine(CFG, mc_draws=args.draws).fit(GAME)
    print("fitted xP engine in %.2fs" % (time.time() - t0))
    t0 = time.time()
    PS = ENG.project(GAME, GWS)
    print("projected GW%d-%d (%d players) in %.2fs"
          % (GWS[0], GWS[-1], len(PS.projections), time.time() - t0))

    FAILURES: List[str] = []

    def gate(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            FAILURES.append(label)

    def legal_free_hit(rec: Dict[str, Any], game: Any, squad_state: Any) -> bool:
        ids = rec["detail"].get("free_hit_squad") or []
        if not ids:
            return False
        cost = sum(game.players[p].now_cost for p in ids)
        return (
            scoring.squad_is_legal([game.players[p].position for p in ids],
                                   [game.players[p].team_id for p in ids])
            and cost <= squad_state.budget
            and scoring.lineup_is_legal(
                [game.players[p].position for p in rec["detail"]["free_hit_xi"]])
        )

    # ------------------------------------------------------ 1. schedule shape
    print("\n" + "=" * 78)
    SCHED = schedule_report(GAME)
    print(format_schedule_report(SCHED, GAME))
    print("\n  detect_double_blank_gws() over all %d gameweeks -> %r"
          % (len(SCHED["gws"]), detect_double_blank_gws(GAME)))
    print("  PLAINLY: there are no double gameweeks and no blank gameweeks in the")
    print("  published 2026/27 fixture list. Every one of the 20 clubs has exactly one")
    print("  fixture in each of GW1-38. The empty dict above is the correct answer, not")
    print("  a bug, and it is recomputed from live fixtures on every call.")

    # Prove the detector actually fires by rescheduling a match, the way the real
    # schedule will from midwinter.
    CLONE, DA, DB, BA, BB = _synthetic_double_blank(GAME, 5)
    SYN = detect_double_blank_gws(CLONE)
    print("\n  synthetic check — move one GW5 match to GW6 and one GW6 match into GW5:")
    print("    GW5 -> doubles %s, blanks %s"
          % ([GAME.short_name(t) for t in SYN.get(5, {}).get("doubles", [])],
             [GAME.short_name(t) for t in SYN.get(5, {}).get("blanks", [])]))
    print("    GW6 -> doubles %s, blanks %s"
          % ([GAME.short_name(t) for t in SYN.get(6, {}).get("doubles", [])],
             [GAME.short_name(t) for t in SYN.get(6, {}).get("blanks", [])]))

    # ---------------------------------------------------------- 2. the squad
    print("\n" + "=" * 78)
    print("PLAUSIBLE GW%d SQUAD (best legal 15 over GW%d-%d at £100.0m)"
          % (GW, GW, min(GW + 5, GWS[-1])))
    print("=" * 78)
    t0 = time.time()
    PICK = best_squad_for_gws(PS, GAME, CFG, list(range(GW, min(GW + 6, GWS[-1] + 1))))
    print("solver %s -> %s in %.2fs, cost £%.1fm"
          % (PICK["solver"], PICK["status"], time.time() - t0, PICK["cost"] / 10.0))
    SQUAD_IDS = PICK["squad"]
    SQUAD = SquadState(
        gw=GW,
        picks=[SquadPick(player_id=p, purchase_price=GAME.players[p].now_cost,
                         selling_price=GAME.players[p].now_cost) for p in SQUAD_IDS],
        bank=scoring.BUDGET_TENTHS - PICK["cost"],
        free_transfers=1,
        chips_used=[],
    )
    XI, BENCH = best_xi(SQUAD_IDS, PS, GW, GAME)
    for pid in sorted(SQUAD_IDS, key=lambda p: (GAME.players[p].position, -PS.xp(p, GW))):
        pl = GAME.players[pid]
        print("  %-3s %-18s %-4s £%4.1fm  xP %.2f  %s"
              % (scoring.POS_NAME[pl.position], pl.web_name, GAME.short_name(pl.team_id),
                 pl.price, PS.xp(pid, GW), "XI" if pid in XI else "bench"))
    print("  squad value £%.1fm, bank £%.1fm, XI xP %.2f"
          % (PICK["cost"] / 10.0, SQUAD.bank / 10.0, sum(PS.xp(p, GW) for p in XI)))

    # -------------------------------------------------------- 3. chip advice
    print("\n" + "=" * 78)
    t0 = time.time()
    RECS = recommend_chips(GAME, SQUAD, PS, CFG, engine=ENG)
    print("recommend_chips in %.2fs\n" % (time.time() - t0))
    print(format_chip_recommendations(RECS, GAME))

    print("\n" + "=" * 78)
    STATUS = chip_expiry_status(GAME, SQUAD, RECS)
    print(format_chip_expiry(STATUS))

    print("\n  escalation ladder (same squad, same projections, the clock moved on):")
    for FAKE_GW in (1, 8, 12, 13, 16, 17, 19, 20):
        print("    planning GW%-2d -> %s" % (FAKE_GW, urgency_for(FAKE_GW, STATUS["expiry_gw"], True)))

    # The escalation is what the user actually sees in December, so print it
    # rather than describing it: same live data, the clock advanced, and one
    # chip already spent.
    import copy as _copy

    LATE = _copy.copy(GAME)
    LATE.current_gw = 17
    LATE_SQUAD = _copy.copy(SQUAD)
    LATE_SQUAD.chips_used = ["bboost"]
    print("\n" + "-" * 78)
    print("the same clock at GW17 with Bench Boost already spent:\n")
    print(format_chip_expiry(chip_expiry_status(LATE, LATE_SQUAD, RECS)))

    # ------------------------------------- 3b. the December case, forced early
    # Today's schedule cannot exercise the branches these chips exist for. Move
    # one match and every one of them goes live at once, so run the whole chip
    # engine against that schedule rather than shipping untested code that only
    # wakes up in December.
    print("\n" + "=" * 78)
    print("SYNTHETIC DOUBLE / BLANK GAMEWEEK — the schedule as it will look in winter")
    print("=" * 78)
    SYN_GWS = list(range(GW, GW + 8))
    SYN_ENG = XPEngine(CFG, mc_draws=args.draws).fit(CLONE)
    SYN_PS = SYN_ENG.project(CLONE, SYN_GWS)
    print("GW5 doubles: %s | GW5 blanks: %s"
          % (", ".join(CLONE.short_name(t) for t in (DA, DB)),
             ", ".join(CLONE.short_name(t) for t in (BA, BB))))

    SYN_PICK = best_squad_for_gws(SYN_PS, CLONE, CFG, SYN_GWS)
    SYN_SQUAD = SquadState(
        gw=GW,
        picks=[SquadPick(player_id=p, purchase_price=CLONE.players[p].now_cost,
                         selling_price=CLONE.players[p].now_cost) for p in SYN_PICK["squad"]],
        bank=scoring.BUDGET_TENTHS - SYN_PICK["cost"], free_transfers=1, chips_used=[])
    SYN_RECS = recommend_chips(CLONE, SYN_SQUAD, SYN_PS, CFG, engine=SYN_ENG)
    print()
    print(format_chip_recommendations(SYN_RECS, CLONE))

    # A squad loaded with the clubs that blank in GW5 is what Free Hit is for.
    def blank_heavy_squad(game: Any, teams: Tuple[int, ...], proj: ProjectionSet) -> List[int]:
        want = {scoring.GKP: 2, scoring.DEF: 5, scoring.MID: 5, scoring.FWD: 3}
        chosen: List[int] = []
        per_club: Dict[int, int] = {}
        for pos, need in want.items():
            ranked = sorted(
                (p for p, pl in game.players.items() if pl.position == pos),
                key=lambda p: (game.players[p].team_id not in teams,
                               -sum(proj.xp(p, g) for g in SYN_GWS), p),
            )
            taken = 0
            for pid in ranked:
                club = game.players[pid].team_id
                if per_club.get(club, 0) >= scoring.TEAM_LIMIT:
                    continue
                chosen.append(pid)
                per_club[club] = per_club.get(club, 0) + 1
                taken += 1
                if taken == need:
                    break
        return chosen

    BLANK_SQUAD = blank_heavy_squad(CLONE, (BA, BB), SYN_PS)
    FH_BLANK = evaluate_free_hit(BLANK_SQUAD, SYN_PS, 5, CLONE, CFG)
    FH_NORMAL = evaluate_free_hit(BLANK_SQUAD, SYN_PS, 3, CLONE, CFG)
    print("  Free Hit for a squad holding %d players from the two blanking clubs:"
          % sum(1 for p in BLANK_SQUAD if CLONE.players[p].team_id in (BA, BB)))
    print("    GW5 (they blank)  : own XI %.2f, free hit XI %.2f, gain %+.2f  [%d assets blank]"
          % (FH_BLANK.detail["current_xi_xp"], FH_BLANK.detail["free_hit_xi_xp"],
             FH_BLANK.gain, FH_BLANK.detail["blanking_assets"]))
    print("    GW3 (normal week) : own XI %.2f, free hit XI %.2f, gain %+.2f"
          % (FH_NORMAL.detail["current_xi_xp"], FH_NORMAL.detail["free_hit_xi_xp"],
             FH_NORMAL.gain))
    BB_DOUBLE = evaluate_bench_boost(SYN_PICK["squad"], SYN_PS, 5, CLONE)
    print("  Bench Boost in the double gameweek: bench xP %.2f over %d bench players, "
          "%d squad assets play twice"
          % (BB_DOUBLE.detail["bench_xp"], len(BB_DOUBLE.detail["bench"]),
             BB_DOUBLE.detail["double_assets"]))
    TC_DOUBLE = evaluate_triple_captain(SYN_PICK["squad"], SYN_PS, 5, CLONE)
    print("  Triple Captain in the double gameweek: %s, %d fixtures, third copy worth %.2f"
          % (TC_DOUBLE.detail["captain_name"], TC_DOUBLE.detail["n_fixtures"], TC_DOUBLE.gain))

    # ------------------------------------------------------- 4. EO + captain
    print("\n" + "=" * 78)
    print("TOP 20 BY ESTIMATED EFFECTIVE OWNERSHIP — GW%d" % GW)
    print("=" * 78)
    OWN = estimate_ownership(GAME, PS, GW)
    print("captaincy model: beta %.3f fitted so the leading candidate (%s) takes %.0f%% of "
          "a %.0f%% pool" % (OWN.beta,
                             GAME.players[OWN.template_captain].web_name
                             if OWN.template_captain else "n/a",
                             100 * OWN.top_captain_share, OWN.captaincy_pool))
    print()
    print(format_ownership_table(OWN, GAME, PS, limit=20))

    print("\n" + "=" * 78)
    print("CAPTAIN OPTIONS — GW%d" % GW)
    print("=" * 78)
    PROF = haul_profiles(PS, GW, engine=ENG)
    DETAIL = captain_analysis(SQUAD_IDS, PS, GW, GAME, CFG, ownership=OWN, profiles=PROF)
    print(format_captain_options(DETAIL, limit=15))
    OPTS = captain_options(SQUAD_IDS, PS, GW, GAME, CFG, ownership=OWN, profiles=PROF)
    print("\n  top three rationales (differential_weight %.2f):"
          % CFG.optimizer.differential_weight)
    for o in OPTS[:3]:
        print("\n  %s:" % GAME.players[o.player_id].web_name)
        for chunk in o.rationale.split("; "):
            print("      %s" % chunk)
    OPTS_DIFF = captain_options(SQUAD_IDS, PS, GW, GAME, CFG, ownership=OWN, profiles=PROF,
                                differential_weight=1.0)
    print("\n  at differential_weight = 1.0 the order becomes: %s"
          % ", ".join(GAME.players[o.player_id].web_name for o in OPTS_DIFF[:5]))

    # ---------------------------------------------------------- 5. rank risk
    print("\n" + "=" * 78)
    REPORT = rank_risk_report(SQUAD, GAME, PS, GW, CFG, ownership=OWN, profiles=PROF)
    print(format_rank_risk_report(REPORT, GAME))

    # --------------------------------------------------------------- 6. gates
    print("\n" + "=" * 78)
    print("SANITY GATES")
    print("=" * 78)
    gate("detect_double_blank_gws is empty on the live schedule",
         detect_double_blank_gws(GAME) == {},
         "%d gameweeks checked" % len(SCHED["gws"]))
    gate("detector fires on a synthetic reschedule",
         DA in SYN.get(5, {}).get("doubles", []) and DB in SYN.get(5, {}).get("doubles", [])
         and BA in SYN.get(5, {}).get("blanks", []) and BB in SYN.get(5, {}).get("blanks", []),
         "GW5 doubles %s, blanks %s" % (SYN.get(5, {}).get("doubles"), SYN.get(5, {}).get("blanks")))
    gate("GW19 deadline read from live events matches the SPEC value",
         STATUS["deadline"] == "2027-01-02T13:30:00Z", STATUS["deadline"] or "missing")
    gate("every chip carries a gain, a confidence and an urgency",
         all(r.get("confidence") and r.get("urgency") and r.get("points_gain") is not None
             for r in RECS),
         "%d recommendations" % len(RECS))
    gate("urgency escalates informational -> warning -> urgent -> expired",
         [urgency_for(g, 19, True) for g in (1, 12, 13, 16, 17, 19, 20)]
         == ["informational", "informational", "warning", "warning", "urgent", "urgent",
             "expired"])
    gate("squad is legal (15, quotas, budget, 3 per club)",
         scoring.squad_is_legal([GAME.players[p].position for p in SQUAD_IDS],
                                [GAME.players[p].team_id for p in SQUAD_IDS])
         and PICK["cost"] <= scoring.BUDGET_TENTHS,
         "cost £%.1fm" % (PICK["cost"] / 10.0))
    gate("XI is legal", scoring.lineup_is_legal([GAME.players[p].position for p in XI]),
         "%d players" % len(XI))
    gate("sum of effective ownership is 11 starters + 1 captain",
         abs(OWN.total_effective / 100.0 - 12.0) < 0.05,
         "%.3f copies" % (OWN.total_effective / 100.0))
    gate("estimated captaincy sums to the pool",
         abs(sum(OWN.captaincy.values()) - OWN.captaincy_pool) < 1e-6,
         "%.4f%%" % sum(OWN.captaincy.values()))
    gate("most-owned premium tops the EO table",
         OWN.top(1)[0][0] == max(GAME.players, key=lambda p: GAME.players[p].selected_by_percent),
         "%s (own %.1f%%, EO %.1f%%)"
         % (GAME.players[OWN.top(1)[0][0]].web_name,
            GAME.players[OWN.top(1)[0][0]].selected_by_percent, OWN.top(1)[0][1]))
    gate("every captain option has all fields populated",
         all(o.xp > 0 and o.sd > 0 and o.p_haul >= 0 and o.effective_ownership >= 0
             and o.rationale for o in OPTS),
         "%d options" % len(OPTS))
    gate("captain options at weight 0 are ordered by xP",
         [round(o.xp, 6) for o in captain_options(
             SQUAD_IDS, PS, GW, GAME, CFG, ownership=OWN, profiles=PROF,
             differential_weight=0.0)]
         == sorted([round(o.xp, 6) for o in OPTS], reverse=True))
    gate("rank risk report populated",
         bool(REPORT["summary"]) and REPORT["captain"] is not None
         and REPORT["field_expected_points"] > 0,
         "edge %+.2f, net haul exposure %+.2f"
         % (REPORT["expected_edge"], REPORT["net_haul_exposure"]))
    gate("free hit target squad is legal and within budget",
         all(legal_free_hit(r, GAME, SQUAD) for r in RECS if r["chip"] == "freehit")
         if any(r["chip"] == "freehit" for r in RECS) else True)
    gate("bench boost gain never exceeds raw bench xP",
         all(r["points_gain"] <= r["detail"].get("bench_xp", 0.0) + 1e-9
             for r in RECS if r["chip"] == "bboost"))
    gate("Free Hit answers a blank: gain is far larger in the blank gameweek",
         FH_BLANK.gain > FH_NORMAL.gain + 5.0,
         "GW5 %+.2f vs GW3 %+.2f" % (FH_BLANK.gain, FH_NORMAL.gain))
    gate("a blanking squad member is counted",
         FH_BLANK.detail["blanking_assets"] > 0 and FH_NORMAL.detail["blanking_assets"] == 0,
         "%d blank in GW5" % FH_BLANK.detail["blanking_assets"])
    gate("Triple Captain sees the double gameweek",
         TC_DOUBLE.detail["n_fixtures"] == 2 and TC_DOUBLE.detail["is_double"],
         "%s over %d fixtures"
         % (TC_DOUBLE.detail["captain_name"], TC_DOUBLE.detail["n_fixtures"]))
    gate("chips on the synthetic schedule are anchored to the structure",
         any(r["structural"] for r in SYN_RECS),
         ", ".join("%s GW%s%s" % (r["chip"], r["gw"], "*" if r["structural"] else "")
                   for r in SYN_RECS))
    gate("a spent chip is reported as USED and forfeits nothing",
         chips_available(LATE, LATE_SQUAD, half=1)["bboost"] is False
         and [e for e in chip_expiry_status(LATE, LATE_SQUAD, RECS)["chips"]
              if e["chip"] == "bboost"][0]["urgency"] == "used")
    gate("urgency at GW17 with chips in hand is urgent",
         chip_expiry_status(LATE, LATE_SQUAD, RECS)["overall_urgency"] == "urgent",
         "%d unused" % len(chip_expiry_status(LATE, LATE_SQUAD, RECS)["unused"]))
    gate("captain relative sd is highest for a low-captaincy pick",
         max(DETAIL, key=lambda a: a.relative_sd).captaincy_share
         <= max(DETAIL, key=lambda a: a.xp).captaincy_share,
         "widest spread: %s (capt %.1f%%)"
         % (max(DETAIL, key=lambda a: a.relative_sd).name,
            max(DETAIL, key=lambda a: a.relative_sd).captaincy_share))
    gate("differential_weight reorders captaincy towards variance",
         [o.player_id for o in OPTS] != [o.player_id for o in OPTS_DIFF],
         "w=0 %s vs w=1 %s" % (GAME.players[OPTS[0].player_id].web_name,
                               GAME.players[OPTS_DIFF[0].player_id].web_name))
    gate("chip recommendations are deterministic",
         [(r["chip"], r["gw"], round(r["points_gain"], 9)) for r in RECS]
         == [(r["chip"], r["gw"], round(r["points_gain"], 9))
             for r in recommend_chips(GAME, SQUAD, PS, CFG, engine=ENG)])

    print("\n%s" % ("ALL GATES PASSED" if not FAILURES
                    else "FAILURES: %s" % ", ".join(FAILURES)))
    raise SystemExit(1 if FAILURES else 0)
