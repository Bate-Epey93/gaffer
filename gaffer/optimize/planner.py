"""Multi-gameweek transfer planner.

The squad optimizer answers "what is the best 15 right now". This answers the
harder question: given the 15 you already own, what should you do over the next
few gameweeks, and — much more often — what should you not do.

The strategy findings that shape the model, and where each one lives:

* A free transfer is worth roughly two points. The free transfers still banked
  at the end of the horizon are credited at `optimizer.ft_value`, and because
  `ft_end = ft_start + weeks - used - wasted`, that single term charges exactly
  `ft_value` for every transfer spent out of a scarce stock and charges nothing
  for one spent in a week when the stock was going to overflow the cap of five
  anyway. A flat per-transfer charge on top of it would tax the same transfer
  twice over and would also tax the ones that were genuinely free.
* A hit has to clear `optimizer.hit_threshold` points **over the horizon**. The
  objective charges the threshold, not the -4, so the hurdle is configurable;
  the reported `expected_points_net` always charges the real -4.
* The last two world champions took near-zero hits. After the MILP solves, the
  first gameweek's move — the only one you actually commit to — is re-solved
  against a forced no-transfer alternative. If the gross gain does not clear the
  value of the free transfer it is spending, the transfer is banked and the note
  says so in as many words.

Money is integer tenths of a million from end to end, including the sell-on
rule: you get your purchase price back plus half of any rise, rounded down.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pulp

from gaffer.core import scoring
from gaffer.core.config import Config
from gaffer.core.types import GWDecision, Plan, ProjectionSet, SquadPick, SquadState, Transfer
from gaffer.optimize.squad import (
    GWView,
    N_OUTFIELD_BENCH,
    OptimizeError,
    bench_weights,
    build_solver,
    build_views,
    captain_multiplier,
    formation_name,
    horizon_xp,
    lineup_expected_points,
    lineup_problems,
    mean_xmins,
    pick_lineup,
    solve_or_raise,
    squad_problems,
)

CHIPS = ("wildcard", "freehit", "bboost", "3xc")
# Chips that suspend the normal transfer rules for the gameweek they are played.
UNLIMITED_TRANSFER_CHIPS = ("wildcard", "freehit")

# Keeps the free-transfer ledger tight when nothing else in the objective cares
# about it (e.g. ft_value set to zero). Six gameweeks at five transfers moves the
# objective by at most 3e-3, far below any real decision margin.
FT_LEDGER_EPS = 1e-4

# Transfers that gain nothing still satisfy the constraints, so break the tie
# towards standing still.
CHURN_EPS = 1e-3


# ---------------------------------------------------------------------------
# Squad state helpers
# ---------------------------------------------------------------------------


def sell_values(squad: SquadState, state: Any) -> Dict[int, int]:
    """What each owned player is worth if sold now, in tenths.

    Prefers the selling price the API reports, but recomputes it from the
    purchase price when it is missing or plainly wrong — a squad state carrying a
    zero selling price would silently hand the planner a free £X.Xm.
    """
    out: Dict[int, int] = {}
    for pick in squad.picks:
        now = int(state.players[pick.player_id].now_cost)
        derived = scoring.selling_price(int(pick.purchase_price), now)
        stored = int(pick.selling_price or 0)
        out[pick.player_id] = stored if stored > 0 else derived
    return out


def synthetic_squad_state(
    squad_ids: Sequence[int],
    state: Any,
    gw: int,
    bank: int = 0,
    free_transfers: int = 1,
    purchase_prices: Optional[Dict[int, int]] = None,
) -> SquadState:
    """A SquadState for a set of ids, with the sell-on rule applied properly."""
    picks: List[SquadPick] = []
    for i, pid in enumerate(squad_ids):
        now = int(state.players[pid].now_cost)
        bought = int((purchase_prices or {}).get(pid, now))
        picks.append(
            SquadPick(
                player_id=int(pid),
                purchase_price=bought,
                selling_price=scoring.selling_price(bought, now),
                multiplier=1 if i < scoring.SQUAD_PLAY else 0,
                position_in_squad=i + 1,
            )
        )
    return SquadState(
        gw=int(gw), picks=picks, bank=int(bank), free_transfers=int(free_transfers)
    )


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------


def transfer_candidates(
    projections: ProjectionSet,
    state: Any,
    config: Config,
    gws: Sequence[int],
    squad: SquadState,
    views: Optional[Dict[int, GWView]] = None,
    per_position: Optional[Dict[int, int]] = None,
    value_per_position: int = 15,
) -> List[int]:
    """Who the planner is allowed to buy.

    The full 485-player pool makes the transfer MILP an order of magnitude
    slower for no gain — nobody outside the top of each position by projected
    points or by points per million is a plausible transfer target. The squad you
    already own is always in the pool whatever it projects, otherwise the model
    could not choose to keep a player it would not buy today.
    """
    views = views if views is not None else build_views(projections, state, gws)
    per_position = per_position or {scoring.GKP: 12, scoring.DEF: 45, scoring.MID: 55, scoring.FWD: 30}
    blocked = set(int(p) for p in config.optimizer.locked_out)
    threshold = float(config.optimizer.min_xmins_to_consider)

    ranked: Dict[int, List[Tuple[float, float, int]]] = {p: [] for p in scoring.POSITIONS}
    for pid in sorted(projections.projections.keys()):
        if pid not in state.players or pid in blocked:
            continue
        if mean_xmins(views, pid, gws) < threshold:
            continue
        total = horizon_xp(views, pid, gws)
        price = max(state.players[pid].now_cost, 1)
        ranked[state.players[pid].position].append((total, total / price, pid))

    pool = set(int(p) for p in squad.player_ids())
    pool |= set(int(p) for p in config.optimizer.locked_in)
    for pos, rows in ranked.items():
        by_points = sorted(rows, key=lambda r: (-r[0], r[2]))[: per_position.get(pos, 20)]
        by_value = sorted(rows, key=lambda r: (-r[1], r[2]))[:value_per_position]
        pool |= set(r[2] for r in by_points) | set(r[2] for r in by_value)
    pool -= blocked
    missing = [p for p in pool if p not in state.players]
    if missing:
        raise OptimizeError("candidate pool contains unknown players: %s" % sorted(missing))
    return sorted(pool)


# ---------------------------------------------------------------------------
# Fixture ticker, for the explanations
# ---------------------------------------------------------------------------


def fixture_ticker(state: Any, team_id: int, gws: Sequence[int], limit: int = 5) -> str:
    """``COV(H) NEW(A) ...`` for a team over a set of gameweeks. A blank shows as
    a dash and a double gameweek shows both opponents."""
    by_gw: Dict[int, List[str]] = {int(g): [] for g in gws}
    for fixture in state.fixtures:
        if fixture.gw is None or int(fixture.gw) not in by_gw:
            continue
        if fixture.team_h == team_id:
            by_gw[int(fixture.gw)].append("%s(H)" % state.short_name(fixture.team_a))
        elif fixture.team_a == team_id:
            by_gw[int(fixture.gw)].append("%s(A)" % state.short_name(fixture.team_h))
    parts = []
    for g in list(gws)[:limit]:
        got = by_gw.get(int(g)) or ["-"]
        parts.append("+".join(got))
    return " ".join(parts)


def evaluate_transfer(
    out_id: int,
    in_id: int,
    projections: ProjectionSet,
    state: Any,
    gws: Sequence[int],
    config: Config,
    views: Optional[Dict[int, GWView]] = None,
    decayed: bool = False,
) -> float:
    """Projected points gained by swapping `out_id` for `in_id` over `gws`.

    Squad-level, not lineup-level: it ignores whether either player would have
    started, so it is a headline number for the notes, not the thing the MILP
    optimises.
    """
    views = views if views is not None else build_views(projections, state, gws)
    decay = float(config.optimizer.decay)
    total = 0.0
    for i, g in enumerate(gws):
        view = views.get(int(g))
        if view is None:
            continue
        delta = view.get_xp(in_id) - view.get_xp(out_id)
        total += (decay ** i) * delta if decayed else delta
    return total


def should_take_hit(gain: float, config: Config) -> bool:
    """A -4 is only worth it if the gain clears the threshold over the whole
    horizon. One good gameweek is not enough."""
    return gain >= float(config.optimizer.hit_threshold)


# ---------------------------------------------------------------------------
# Chip plan
# ---------------------------------------------------------------------------


def normalise_chip_plan(
    chip_plan: Optional[Dict[int, str]],
    gws: Sequence[int],
    chips_available: Optional[Sequence[str]] = None,
) -> Dict[int, str]:
    if not chip_plan:
        return {}
    out: Dict[int, str] = {}
    seen: Dict[str, int] = {}
    for gw, chip in chip_plan.items():
        gw = int(gw)
        chip = str(chip).lower()
        if chip not in CHIPS:
            raise OptimizeError("unknown chip %r (want one of %s)" % (chip, ", ".join(CHIPS)))
        if gw not in [int(g) for g in gws]:
            raise OptimizeError("chip %s is planned for GW%d, outside the horizon" % (chip, gw))
        if gw in out:
            raise OptimizeError("two chips planned for GW%d" % gw)
        if chip in seen:
            raise OptimizeError(
                "%s planned twice, GW%d and GW%d" % (chip, seen[chip], gw)
            )
        windows = scoring.CHIP_WINDOWS.get(chip, ())
        if windows and not any(lo <= gw <= hi for lo, hi in windows):
            raise OptimizeError("%s cannot be played in GW%d" % (chip, gw))
        if chips_available is not None and chip not in [str(c).lower() for c in chips_available]:
            raise OptimizeError("%s is not in chips_available" % chip)
        seen[chip] = gw
        out[gw] = chip
    return out


# ---------------------------------------------------------------------------
# The transfer MILP
# ---------------------------------------------------------------------------


@dataclass
class _Solved:
    """Raw MILP output, before it is turned into GWDecisions."""

    squad_by_gw: Dict[int, List[int]] = field(default_factory=dict)
    persistent_by_gw: Dict[int, List[int]] = field(default_factory=dict)
    ins_by_gw: Dict[int, List[int]] = field(default_factory=dict)
    outs_by_gw: Dict[int, List[int]] = field(default_factory=dict)
    bank_by_gw: Dict[int, int] = field(default_factory=dict)
    ft_before: Dict[int, int] = field(default_factory=dict)
    ft_after: Dict[int, int] = field(default_factory=dict)
    hits_by_gw: Dict[int, int] = field(default_factory=dict)
    used_by_gw: Dict[int, int] = field(default_factory=dict)
    objective: float = 0.0
    points: float = 0.0  # decayed expected points only, no charges
    status: str = ""
    seconds: float = 0.0
    solver: str = ""


def _add_squad_shape(
    prob: pulp.LpProblem,
    membership: Dict[int, Any],
    pool: Sequence[int],
    position: Dict[int, int],
    team_of: Dict[int, int],
    tag: str,
) -> None:
    prob += pulp.lpSum(membership[p] for p in pool) == scoring.SQUAD_SIZE, "size_%s" % tag
    for pos, need in sorted(scoring.SQUAD_SELECT.items()):
        prob += (
            pulp.lpSum(membership[p] for p in pool if position[p] == pos) == need,
            "quota_%s_%s" % (scoring.POS_NAME[pos], tag),
        )
    for team_id in sorted(set(team_of.values())):
        members = [p for p in pool if team_of[p] == team_id]
        if len(members) > scoring.TEAM_LIMIT:
            prob += (
                pulp.lpSum(membership[p] for p in members) <= scoring.TEAM_LIMIT,
                "club_%d_%s" % (team_id, tag),
            )


def _solve_plan(
    state: Any,
    squad: SquadState,
    projections: ProjectionSet,
    config: Config,
    gws: Sequence[int],
    views: Dict[int, GWView],
    pool: Sequence[int],
    chip_by_gw: Dict[int, str],
    forbid_transfers_in: Sequence[int] = (),
    solver_name: Optional[str] = None,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
    msg: bool = False,
) -> _Solved:
    gws = [int(g) for g in gws]
    pool = sorted(int(p) for p in pool)
    owned = [int(p) for p in squad.player_ids()]
    for pid in owned:
        if pid not in pool:
            raise OptimizeError("owned player %d is missing from the candidate pool" % pid)

    position = {p: state.players[p].position for p in pool}
    team_of = {p: state.players[p].team_id for p in pool}
    buy_cost = {p: int(state.players[p].now_cost) for p in pool}
    sell_of = sell_values(squad, state)
    sell_value = {p: int(sell_of.get(p, buy_cost[p])) for p in pool}

    opt = config.optimizer
    decay = float(opt.decay)
    ft_value = float(opt.ft_value)
    hit_hurdle = float(opt.hit_threshold)
    value_per_tenth = float(opt.value_per_tenth)
    out_weights, gk_weight = bench_weights(config)
    forbid = set(int(g) for g in forbid_transfers_in)
    locked_in = [int(p) for p in opt.locked_in]
    # A locked-out player you already own has to be sold, not merely left out of
    # the XI, so he is banned from the squad rather than filtered from the pool
    # (filtering him would have made owning him unrepresentable instead).
    locked_out = [int(p) for p in opt.locked_out if int(p) in pool]

    prob = pulp.LpProblem("gaffer_plan", pulp.LpMaximize)

    # Persistent squad after gameweek w's transfers. On a free-hit week this is
    # frozen and a separate `fh` squad is fielded, which is what makes the squad
    # revert the following gameweek.
    x = {
        (p, w): pulp.LpVariable("x_%d_%d" % (p, w), cat=pulp.LpBinary)
        for p in pool
        for w in gws
    }
    tin = {
        (p, w): pulp.LpVariable("in_%d_%d" % (p, w), cat=pulp.LpBinary)
        for p in pool
        for w in gws
    }
    tout = {
        (p, w): pulp.LpVariable("out_%d_%d" % (p, w), cat=pulp.LpBinary)
        for p in pool
        for w in gws
    }
    y = {
        (p, w): pulp.LpVariable("y_%d_%d" % (p, w), cat=pulp.LpBinary)
        for p in pool
        for w in gws
    }
    # Captain and bench-slot assignments are totally unimodular once x and y are
    # integral, so the LP already returns integral values for them; declaring
    # them binary would only give the brancher thousands more variables.
    cap = {
        (p, w): pulp.LpVariable("c_%d_%d" % (p, w), lowBound=0, upBound=1)
        for p in pool
        for w in gws
    }
    outfield = [p for p in pool if position[p] != scoring.GKP]
    z = {
        (p, w, k): pulp.LpVariable("z_%d_%d_%d" % (p, w, k), lowBound=0, upBound=1)
        for p in outfield
        for w in gws
        for k in range(N_OUTFIELD_BENCH)
    }
    bank = {
        w: pulp.LpVariable("bank_%d" % w, lowBound=0) for w in gws
    }
    ft = {
        w: pulp.LpVariable("ft_%d" % w, lowBound=0, upBound=scoring.MAX_BANKED_FREE_TRANSFERS,
                           cat=pulp.LpInteger)
        for w in gws
    }
    ft_end = pulp.LpVariable(
        "ft_end", lowBound=0, upBound=scoring.MAX_BANKED_FREE_TRANSFERS, cat=pulp.LpInteger
    )
    used = {
        w: pulp.LpVariable("used_%d" % w, lowBound=0, upBound=scoring.MAX_BANKED_FREE_TRANSFERS,
                           cat=pulp.LpInteger)
        for w in gws
    }
    hits = {
        w: pulp.LpVariable("hits_%d" % w, lowBound=0, upBound=int(opt.max_hits_per_gw),
                           cat=pulp.LpInteger)
        for w in gws
    }

    # Free-hit squads live in their own variables for the one gameweek they exist.
    fh: Dict[Tuple[int, int], Any] = {}
    keep: Dict[Tuple[int, int], Any] = {}
    for w, chip in chip_by_gw.items():
        if chip != "freehit":
            continue
        for p in pool:
            fh[(p, w)] = pulp.LpVariable("fh_%d_%d" % (p, w), cat=pulp.LpBinary)
            if buy_cost[p] > sell_value[p]:
                # Keeping a player you already own on a free hit costs his selling
                # price, not his (higher) current price.
                keep[(p, w)] = pulp.LpVariable("keep_%d_%d" % (p, w), lowBound=0, upBound=1)

    # --- squad-state transition -------------------------------------------
    prev_const = {p: (1 if p in set(owned) else 0) for p in pool}
    for i, w in enumerate(gws):
        chip = chip_by_gw.get(w)
        frozen = chip == "freehit" or w in forbid
        for p in pool:
            previous = prev_const[p] if i == 0 else x[(p, gws[i - 1])]
            prob += (
                x[(p, w)] == previous + tin[(p, w)] - tout[(p, w)],
                "flow_%d_%d" % (p, w),
            )
            prob += tin[(p, w)] + tout[(p, w)] <= 1, "no_churn_%d_%d" % (p, w)
            if frozen:
                prob += tin[(p, w)] == 0, "freeze_in_%d_%d" % (p, w)
                prob += tout[(p, w)] == 0, "freeze_out_%d_%d" % (p, w)
        _add_squad_shape(prob, {p: x[(p, w)] for p in pool}, pool, position, team_of, "x%d" % w)
        for p in locked_in:
            prob += x[(p, w)] == 1, "lock_in_%d_%d" % (p, w)
        for p in locked_out:
            # On a frozen week the squad cannot change, so banning him outright
            # would be infeasible; he simply cannot be fielded.
            if not frozen:
                prob += x[(p, w)] == 0, "lock_out_%d_%d" % (p, w)
            prob += y[(p, w)] == 0, "lock_out_xi_%d_%d" % (p, w)
            if (p, w) in fh:
                prob += fh[(p, w)] == 0, "lock_out_fh_%d_%d" % (p, w)

        n_transfers = pulp.lpSum(tin[(p, w)] for p in pool)
        prev_bank = squad.bank if i == 0 else bank[gws[i - 1]]
        if chip == "freehit":
            # No money changes hands: the squad and the bank both revert.
            prob += bank[w] == prev_bank, "bank_%d" % w
        else:
            prob += (
                bank[w]
                == prev_bank
                + pulp.lpSum(sell_value[p] * tout[(p, w)] for p in pool)
                - pulp.lpSum(buy_cost[p] * tin[(p, w)] for p in pool),
                "bank_%d" % w,
            )

        # --- free transfers and hits --------------------------------------
        if i == 0:
            prob += (
                ft[w] == max(0, min(int(squad.free_transfers), scoring.MAX_BANKED_FREE_TRANSFERS)),
                "ft_start",
            )
        if chip in UNLIMITED_TRANSFER_CHIPS:
            prob += used[w] == 0, "no_ft_used_%d" % w
            prob += hits[w] == 0, "no_hits_%d" % w
            prob += n_transfers <= scoring.SQUAD_SIZE, "transfer_cap_%d" % w
        else:
            prob += used[w] <= ft[w], "used_le_ft_%d" % w
            prob += used[w] <= n_transfers, "used_le_n_%d" % w
            prob += n_transfers <= used[w] + hits[w], "cover_transfers_%d" % w
            prob += n_transfers <= scoring.TRANSFERS_CAP, "transfer_cap_%d" % w
        nxt = ft_end if i == len(gws) - 1 else ft[gws[i + 1]]
        prob += nxt <= ft[w] - used[w] + 1, "ft_roll_%d" % w
        prob += nxt <= scoring.MAX_BANKED_FREE_TRANSFERS, "ft_cap_%d" % w

        # --- what is actually fielded this gameweek -----------------------
        if chip == "freehit":
            fielded = {p: fh[(p, w)] for p in pool}
            _add_squad_shape(prob, fielded, pool, position, team_of, "fh%d" % w)
            prev_x = {p: (prev_const[p] if i == 0 else x[(p, gws[i - 1])]) for p in pool}
            discount = []
            for p in pool:
                if (p, w) in keep:
                    prob += keep[(p, w)] <= fh[(p, w)], "keep_in_fh_%d_%d" % (p, w)
                    prob += keep[(p, w)] <= prev_x[p], "keep_owned_%d_%d" % (p, w)
                    discount.append((buy_cost[p] - sell_value[p]) * keep[(p, w)])
            prob += (
                pulp.lpSum(buy_cost[p] * fh[(p, w)] for p in pool) - pulp.lpSum(discount)
                <= prev_bank + pulp.lpSum(sell_value[p] * prev_x[p] for p in pool),
                "fh_budget_%d" % w,
            )
        else:
            fielded = {p: x[(p, w)] for p in pool}

        for p in pool:
            prob += y[(p, w)] <= fielded[p], "start_in_squad_%d_%d" % (p, w)
            prob += cap[(p, w)] <= y[(p, w)], "captain_starts_%d_%d" % (p, w)
        prob += pulp.lpSum(y[(p, w)] for p in pool) == scoring.SQUAD_PLAY, "xi_size_%d" % w
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
        prob += pulp.lpSum(cap[(p, w)] for p in pool) == 1, "one_captain_%d" % w
        for k in range(N_OUTFIELD_BENCH):
            prob += (
                pulp.lpSum(z[(p, w, k)] for p in outfield) == 1,
                "bench_slot_%d_%d" % (k, w),
            )
        for p in outfield:
            prob += (
                pulp.lpSum(z[(p, w, k)] for k in range(N_OUTFIELD_BENCH))
                == fielded[p] - y[(p, w)],
                "bench_member_%d_%d" % (p, w),
            )

    # --- objective ---------------------------------------------------------
    points_terms = []
    charge_terms = []
    for i, w in enumerate(gws):
        d = decay ** i
        chip = chip_by_gw.get(w)
        mult = captain_multiplier(chip) - 1
        view = views[w]
        for p in pool:
            xp = view.get_xp(p)
            if not xp:
                continue
            points_terms.append(d * xp * y[(p, w)])
            points_terms.append(d * mult * xp * cap[(p, w)])
        if chip == "bboost":
            # Every bench point counts, so the autosub discount goes away.
            for p in pool:
                xp = view.get_xp(p)
                if not xp:
                    continue
                member = fh[(p, w)] if (p, w) in fh else x[(p, w)]
                points_terms.append(d * xp * (member - y[(p, w)]))
        else:
            for p in pool:
                xp = view.get_xp(p)
                if not xp:
                    continue
                if position[p] == scoring.GKP and gk_weight:
                    member = fh[(p, w)] if (p, w) in fh else x[(p, w)]
                    points_terms.append(d * gk_weight * xp * (member - y[(p, w)]))
            for p in outfield:
                xp = view.get_xp(p)
                if not xp:
                    continue
                for k in range(N_OUTFIELD_BENCH):
                    if out_weights[k]:
                        points_terms.append(d * out_weights[k] * xp * z[(p, w, k)])
        charge_terms.append(-d * hit_hurdle * hits[w])
        charge_terms.append(-CHURN_EPS * pulp.lpSum(tin[(p, w)] for p in pool))
        charge_terms.append(FT_LEDGER_EPS * ft[w])

    # Terminal value. Over the horizon the ledger reads
    #     ft_end = ft_start + len(gws) - total_used - total_wasted
    # so crediting the surviving stock at ft_value charges exactly ft_value for
    # every free transfer spent, and charges nothing for one spent in a week when
    # the stock was going to overflow the cap of 5 anyway. That is the whole
    # opportunity cost of a transfer, correctly priced; an extra flat per-transfer
    # charge on top would tax the same transfer twice and would also tax the ones
    # that were free.
    terminal = decay ** len(gws)
    last = gws[-1]
    squad_value_end = pulp.lpSum(sell_value[p] * x[(p, last)] for p in pool)
    charge_terms.append(terminal * ft_value * ft_end)
    charge_terms.append(terminal * value_per_tenth * (bank[last] + squad_value_end))

    points_expr = pulp.lpSum(points_terms)
    prob += points_expr + pulp.lpSum(charge_terms)

    solver, label = build_solver(
        config, solver_name=solver_name, time_limit=time_limit, seed=seed, msg=msg
    )
    report = solve_or_raise(prob, solver, label)

    out = _Solved(
        objective=report.objective,
        points=float(pulp.value(points_expr) or 0.0),
        status=report.status,
        seconds=report.seconds,
        solver=label,
    )
    for i, w in enumerate(gws):
        chip = chip_by_gw.get(w)
        persistent = sorted(p for p in pool if (x[(p, w)].value() or 0) > 0.5)
        if chip == "freehit":
            fielded_ids = sorted(p for p in pool if (fh[(p, w)].value() or 0) > 0.5)
        else:
            fielded_ids = persistent
        out.persistent_by_gw[w] = persistent
        out.squad_by_gw[w] = fielded_ids
        out.ins_by_gw[w] = sorted(p for p in pool if (tin[(p, w)].value() or 0) > 0.5)
        out.outs_by_gw[w] = sorted(p for p in pool if (tout[(p, w)].value() or 0) > 0.5)
        out.bank_by_gw[w] = int(round(bank[w].value() or 0.0))
        out.ft_before[w] = int(round(ft[w].value() or 0.0))
        out.ft_after[w] = int(
            round((ft_end if i == len(gws) - 1 else ft[gws[i + 1]]).value() or 0.0)
        )
        out.hits_by_gw[w] = int(round(hits[w].value() or 0.0))
        out.used_by_gw[w] = int(round(used[w].value() or 0.0))
    return out


# ---------------------------------------------------------------------------
# Turning a solved MILP into a readable plan
# ---------------------------------------------------------------------------


def _pair_transfers(
    ins: Sequence[int],
    outs: Sequence[int],
    state: Any,
    views: Dict[int, GWView],
    gws: Sequence[int],
) -> List[Tuple[int, int]]:
    """Match each player out to a player in.

    The MILP only produces two sets. Position quotas hold every gameweek, so the
    multiset of positions coming in equals the multiset going out and the pairing
    is well defined within a position; ranking both sides by projected points
    makes the pair that gets written into the notes the sensible one.
    """
    pairs: List[Tuple[int, int]] = []
    for pos in scoring.POSITIONS:
        outs_pos = sorted(
            (p for p in outs if state.players[p].position == pos),
            key=lambda p: (-horizon_xp(views, p, gws), p),
        )
        ins_pos = sorted(
            (p for p in ins if state.players[p].position == pos),
            key=lambda p: (-horizon_xp(views, p, gws), p),
        )
        if len(outs_pos) != len(ins_pos):
            raise OptimizeError(
                "transfer positions do not balance for %s: %d out, %d in"
                % (scoring.POS_NAME[pos], len(outs_pos), len(ins_pos))
            )
        pairs.extend(zip(outs_pos, ins_pos))
    return pairs


def _move_note(
    out_id: int,
    in_id: int,
    state: Any,
    views: Dict[int, GWView],
    gws: Sequence[int],
    remaining: Sequence[int],
) -> str:
    out_p = state.players[out_id]
    in_p = state.players[in_id]
    gain = sum(views[g].get_xp(in_id) - views[g].get_xp(out_id) for g in remaining)
    out_mins = mean_xmins(views, out_id, remaining)
    in_mins = mean_xmins(views, in_id, remaining)
    return (
        "OUT %s (%s %s, %.1fm, %.0f mins/GW) -> IN %s (%s %s, %.1fm, %.0f mins/GW): "
        "%+.2f xP over GW%d-%d. Fixtures out %s | in %s."
        % (
            out_p.web_name,
            state.short_name(out_p.team_id),
            scoring.POS_NAME[out_p.position],
            out_p.price,
            out_mins,
            in_p.web_name,
            state.short_name(in_p.team_id),
            scoring.POS_NAME[in_p.position],
            in_p.price,
            in_mins,
            gain,
            remaining[0],
            remaining[-1],
            fixture_ticker(state, out_p.team_id, remaining),
            fixture_ticker(state, in_p.team_id, remaining),
        )
    )


def _build_decisions(
    solved: _Solved,
    state: Any,
    squad: SquadState,
    projections: ProjectionSet,
    config: Config,
    gws: Sequence[int],
    views: Dict[int, GWView],
    chip_by_gw: Dict[int, str],
) -> Tuple[List[GWDecision], float]:
    decisions: List[GWDecision] = []
    decay = float(config.optimizer.decay)
    decayed_points = 0.0
    held = sorted(int(p) for p in squad.player_ids())
    for i, w in enumerate(gws):
        chip = chip_by_gw.get(w)
        fielded = solved.squad_by_gw[w]
        remaining = [g for g in gws if g >= w]
        problems = squad_problems(fielded, state, budget=10 ** 9)
        if problems:
            raise OptimizeError("GW%d squad is illegal: %s" % (w, "; ".join(problems)))

        # The free-hit revert is the classic place for this model to be quietly
        # wrong, so the invariant is checked here rather than only in a test: a
        # free hit must leave the owned squad byte-for-byte unchanged, and every
        # other gameweek must move it by exactly the transfers made.
        persistent = solved.persistent_by_gw[w]
        if chip == "freehit":
            if persistent != held:
                raise OptimizeError(
                    "free hit in GW%d changed the owned squad by %d players — the "
                    "revert is broken"
                    % (w, len(set(persistent) ^ set(held)))
                )
        else:
            expected = sorted(
                (set(held) - set(solved.outs_by_gw[w])) | set(solved.ins_by_gw[w])
            )
            if persistent != expected:
                raise OptimizeError(
                    "GW%d squad does not match its own transfers (%d players adrift)"
                    % (w, len(set(persistent) ^ set(expected)))
                )
            held = persistent

        lineup, bench, captain, vice = pick_lineup(
            list(fielded), projections, w, state, chip=chip, view=views[w]
        )
        gross = lineup_expected_points(lineup, bench, captain, views[w], chip)
        decayed_points += (decay ** i) * gross
        hits = solved.hits_by_gw[w]
        pairs = _pair_transfers(
            solved.ins_by_gw[w], solved.outs_by_gw[w], state, views, remaining
        )
        transfers = [
            Transfer(
                gw=w,
                out_id=o,
                in_id=n,
                out_price=int(state.players[o].now_cost),
                in_price=int(state.players[n].now_cost),
            )
            for o, n in pairs
        ]

        gain = sum(
            sum(views[g].get_xp(n) - views[g].get_xp(o) for g in remaining)
            for o, n in pairs
        )
        notes: List[str] = []
        if chip:
            notes.append(_chip_note(chip, w, state, fielded, lineup, bench, captain, views[w]))
        if chip == "freehit":
            notes.append(
                "  %d of the free-hit 15 are borrowed for the week; GW%d starts again "
                "from the %d you own." % (len(set(fielded) - set(held)), w + 1, len(held))
            )
        if transfers:
            n = len(transfers)
            notes.append(
                "GW%d: %d transfer%s, %d free transfer%s used, %d hit%s (%d points)."
                % (
                    w, n, "" if n == 1 else "s",
                    solved.used_by_gw[w], "" if solved.used_by_gw[w] == 1 else "s",
                    hits, "" if hits == 1 else "s", -scoring.TRANSFER_HIT_COST * hits,
                )
            )
            for out_id, in_id in pairs:
                notes.append("  " + _move_note(out_id, in_id, state, views, gws, remaining))
        elif not chip:
            notes.append(
                "GW%d: no transfer. %d free transfer%s banked for GW%d."
                % (
                    w, solved.ft_after[w], "" if solved.ft_after[w] == 1 else "s",
                    w + 1,
                )
            )
        if hits:
            notes.append(
                "  Hit justified: %+.2f xP over GW%d-%d against a %.1f point hurdle "
                "(%d x -4 = %d actual points)."
                % (
                    gain, remaining[0], remaining[-1], config.optimizer.hit_threshold,
                    hits, -scoring.TRANSFER_HIT_COST * hits,
                )
            )
        elif transfers and i > 0 and gain < float(config.optimizer.ft_value):
            # Only the first gameweek is actually committed to, so a thin gain
            # further out is not wrong, it is just not yet worth acting on.
            notes.append(
                "  Provisional: gains only %.2f xP over GW%d-%d, under the %.2f a free "
                "transfer is worth. Re-plan at the GW%d deadline before making it."
                % (gain, remaining[0], remaining[-1], config.optimizer.ft_value, w)
            )
        notes.append(
            "  XI %s, captain %s (%.2f xP), vice %s (%.2f xP). Bank %.1fm, "
            "%d free transfer%s for GW%d."
            % (
                formation_name([state.players[p].position for p in lineup]),
                state.players[captain].web_name,
                views[w].get_xp(captain),
                state.players[vice].web_name,
                views[w].get_xp(vice),
                solved.bank_by_gw[w] / 10.0,
                solved.ft_after[w],
                "" if solved.ft_after[w] == 1 else "s",
                w + 1,
            )
        )

        decisions.append(
            GWDecision(
                gw=w,
                transfers=transfers,
                hits=hits,
                chip=chip,
                lineup=lineup,
                bench=bench,
                captain=captain,
                vice_captain=vice,
                squad=list(fielded),
                bank_after=solved.bank_by_gw[w],
                free_transfers_after=solved.ft_after[w],
                expected_points=gross,
                expected_points_net=gross - scoring.TRANSFER_HIT_COST * hits,
                notes=notes,
            )
        )
    return decisions, decayed_points


def _chip_note(
    chip: str,
    gw: int,
    state: Any,
    squad: Sequence[int],
    lineup: Sequence[int],
    bench: Sequence[int],
    captain: int,
    view: GWView,
) -> str:
    if chip == "freehit":
        return (
            "GW%d FREE HIT: unlimited free transfers for this gameweek only, then the "
            "squad reverts to exactly what it was before. Nothing bought here is kept."
            % gw
        )
    if chip == "wildcard":
        return (
            "GW%d WILDCARD: unlimited transfers, no hits, and the new squad is "
            "permanent." % gw
        )
    if chip == "bboost":
        return (
            "GW%d BENCH BOOST: all 15 score, bench adds %.2f xP."
            % (gw, sum(view.get_xp(p) for p in bench))
        )
    if chip == "3xc":
        return (
            "GW%d TRIPLE CAPTAIN on %s: %.2f xP instead of %.2f."
            % (gw, state.players[captain].web_name, 3 * view.get_xp(captain), 2 * view.get_xp(captain))
        )
    return "GW%d chip %s." % (gw, chip)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan(
    state: Any,
    squad: SquadState,
    projections: ProjectionSet,
    config: Config,
    chips_available: Optional[List[str]] = None,
    horizon: Optional[int] = None,
    chip_plan: Optional[Dict[int, str]] = None,
    candidates: Optional[Sequence[int]] = None,
    patience_check: bool = True,
    solver_name: Optional[str] = None,
    seed: Optional[int] = None,
    time_limit: Optional[int] = None,
) -> Plan:
    """The transfer plan for the next `horizon` gameweeks.

    `chip_plan` places a chip in a specific gameweek and the MILP then models it
    properly. `chips_available` restricts which chips `chip_plan` may use, and
    when no chip is placed it is only used to add a suggestion to the notes —
    picking the gameweek for a chip is `optimize/chips.py`'s job, not this one's.
    """
    first = int(squad.gw)
    horizon = int(horizon if horizon is not None else config.optimizer.horizon)
    last = min(projections.last_gw, first + horizon - 1)
    if last < first:
        raise OptimizeError(
            "projections end at GW%d but the squad state is for GW%d"
            % (projections.last_gw, first)
        )
    gws = list(range(first, last + 1))
    if len(squad.picks) != scoring.SQUAD_SIZE:
        raise OptimizeError(
            "squad state has %d picks, needs %d" % (len(squad.picks), scoring.SQUAD_SIZE)
        )

    views = build_views(projections, state, gws)
    chip_by_gw = normalise_chip_plan(chip_plan, gws, chips_available)
    if candidates is None:
        candidates = transfer_candidates(projections, state, config, gws, squad, views=views)
    pool = sorted(set(int(p) for p in candidates) | set(int(p) for p in squad.player_ids()))

    t0 = time.time()
    solved = _solve_plan(
        state, squad, projections, config, gws, views, pool, chip_by_gw,
        solver_name=solver_name, seed=seed, time_limit=time_limit,
    )
    patience_note: Optional[str] = None

    # Only the first gameweek's move is actually committed to, so that is the one
    # worth testing against doing nothing. A transfer that gains less than the
    # free transfer it spends is churn, and gets banked instead.
    if (
        patience_check
        and solved.ins_by_gw[gws[0]]
        and chip_by_gw.get(gws[0]) is None
        and solved.hits_by_gw[gws[0]] == 0
    ):
        alt = _solve_plan(
            state, squad, projections, config, gws, views, pool, chip_by_gw,
            forbid_transfers_in=[gws[0]],
            solver_name=solver_name, seed=seed, time_limit=time_limit,
        )
        gross_gain = solved.points - alt.points
        if gross_gain < float(config.optimizer.ft_value):
            patience_note = (
                "Banked the GW%d transfer: the best move gains only %.2f xP across "
                "GW%d-%d, under the %.2f points a free transfer is worth. Rolling it "
                "leaves %d free transfers for GW%d and keeps the options open."
                % (
                    gws[0], gross_gain, gws[0], gws[-1], config.optimizer.ft_value,
                    alt.ft_after[gws[0]], gws[0] + 1,
                )
            )
            solved = alt
        else:
            patience_note = (
                "GW%d transfer clears the patience test: %+.2f xP across GW%d-%d "
                "against the %.2f points a free transfer is worth."
                % (gws[0], gross_gain, gws[0], gws[-1], config.optimizer.ft_value)
            )
    elif patience_check and not solved.ins_by_gw[gws[0]] and chip_by_gw.get(gws[0]) is None:
        patience_note = (
            "No GW%d transfer is worth the free transfer it would spend; it rolls to "
            "GW%d (%d banked)." % (gws[0], gws[0] + 1, solved.ft_after[gws[0]])
        )

    decisions, decayed_points = _build_decisions(
        solved, state, squad, projections, config, gws, views, chip_by_gw
    )
    if patience_note:
        decisions[0].notes.insert(0, patience_note)
    for suggestion in _chip_suggestions(chips_available, chip_by_gw, decisions, views, gws, state):
        decisions[0].notes.append(suggestion)

    return Plan(
        first_gw=first,
        horizon=len(gws),
        decisions=decisions,
        objective=solved.objective,
        decay=float(config.optimizer.decay),
        solver_status="%s %s" % (solved.solver, solved.status),
        solve_seconds=time.time() - t0,
    )


def _chip_suggestions(
    chips_available: Optional[Sequence[str]],
    chip_by_gw: Dict[int, str],
    decisions: Sequence[GWDecision],
    views: Dict[int, GWView],
    gws: Sequence[int],
    state: Any,
) -> List[str]:
    """Cheap post-hoc read on the two chips that do not change the squad.

    Wildcard and free hit change what you own and so have to be placed by the
    caller before the MILP runs; bench boost and triple captain only change how
    the gameweek scores, so their best week can be read straight off the plan.
    """
    if not chips_available:
        return []
    have = set(str(c).lower() for c in chips_available) - set(chip_by_gw.values())
    out: List[str] = []
    if "bboost" in have:
        best = max(
            decisions, key=lambda d: sum(views[d.gw].get_xp(p) for p in d.bench)
        )
        out.append(
            "Bench boost note: the best bench in this plan is GW%d at %.2f xP. "
            "Worth holding for a double gameweek unless that is unusually high."
            % (best.gw, sum(views[best.gw].get_xp(p) for p in best.bench))
        )
    if "3xc" in have:
        best = max(decisions, key=lambda d: views[d.gw].get_xp(d.captain))
        out.append(
            "Triple captain note: the best captain in this plan is %s in GW%d "
            "(%.2f xP, so the chip is worth %.2f)."
            % (
                state.players[best.captain].web_name,
                best.gw,
                views[best.gw].get_xp(best.captain),
                views[best.gw].get_xp(best.captain),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plan_table(result: Plan, state: Any, views: Dict[int, GWView]) -> str:
    header = "%-5s %-7s %-5s %5s %6s %6s %7s  %s" % (
        "gw", "chip", "moves", "hits", "xP", "net xP", "bank", "FT after"
    )
    lines = [header, "-" * (len(header) + 10)]
    for d in result.decisions:
        lines.append(
            "%-5d %-7s %-5d %5d %6.2f %6.2f %6.1fm %7d"
            % (
                d.gw,
                d.chip or "-",
                len(d.transfers),
                d.hits,
                d.expected_points,
                d.expected_points_net,
                d.bank_after / 10.0,
                d.free_transfers_after,
            )
        )
    lines.append(
        "%-5s %-7s %-5d %5d %6.2f %6.2f"
        % (
            "total",
            "",
            sum(len(d.transfers) for d in result.decisions),
            sum(d.hits for d in result.decisions),
            sum(d.expected_points for d in result.decisions),
            result.total_expected_points,
        )
    )
    return "\n".join(lines)


def plan_notes(result: Plan) -> str:
    lines: List[str] = []
    for d in result.decisions:
        lines.append("GW%d" % d.gw)
        for note in d.notes:
            lines.append("   " + note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _weak_squad(PS, game, cfg, GWS, drop_top: int = 22):
    """A legal but deliberately mediocre starting squad, so the planner has
    something real to fix. Built by banning the best players outright rather
    than by hand-picking, which keeps it legal by construction."""
    from gaffer.optimize.squad import candidate_ids, solve_squad_milp

    views = build_views(PS, game, GWS)
    pool = candidate_ids(PS, game, cfg, GWS, views=views)
    ranked = sorted(pool, key=lambda p: (-horizon_xp(views, p, GWS), p))
    banned = set(ranked[:drop_top])
    solution = solve_squad_milp(
        PS, game, cfg, GWS, views=views,
        candidates=[p for p in pool if p not in banned],
        budget=scoring.BUDGET_TENTHS - 25,  # leave 2.5m in the bank to plan with
    )
    return solution.squad, scoring.BUDGET_TENTHS - solution.cost


def _main() -> int:
    from gaffer.core.config import Config as _Config
    from gaffer.data.loaders import load_game_state
    from gaffer.model.xp import XPEngine

    parser = argparse.ArgumentParser(description="gaffer transfer planner smoke test")
    parser.add_argument("--gws", default="", help="e.g. 1-6")
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
        engine.fit(game)
        PS = engine.project(game, GWS)
        engine.save_projections(PS)
    VIEWS = build_views(PS, game, GWS)
    print("projections: %d players, GW%d-%d" % (len(PS.projections), PS.first_gw, PS.last_gw))

    # ------------------------------------------------------- synthetic squad
    IDS, BANK = _weak_squad(PS, game, cfg, GWS)
    # Half the squad was bought 0.3m cheaper than today's price, so the sell-on
    # rule actually bites: you get the rise back only in whole 0.1m halves.
    BOUGHT = {
        pid: game.players[pid].now_cost - (3 if i % 2 == 0 else 0)
        for i, pid in enumerate(IDS)
    }
    SQ = synthetic_squad_state(IDS, game, GWS[0], bank=BANK, free_transfers=1,
                               purchase_prices=BOUGHT)
    print(
        "\nsynthetic squad for GW%d: value %.1fm, bank %.1fm, %d FT, %d xP over GW%d-%d"
        % (
            SQ.gw, SQ.squad_value / 10.0, SQ.bank / 10.0, SQ.free_transfers,
            sum(horizon_xp(VIEWS, p, GWS) for p in IDS), GWS[0], GWS[-1],
        )
    )
    SELL = sell_values(SQ, game)
    RISERS = [p for p in IDS if SELL[p] != game.players[p].now_cost]
    print(
        "  sell-on rule bites on %d players, e.g. %s bought %.1fm now %.1fm sells %.1fm"
        % (
            len(RISERS),
            game.players[RISERS[0]].web_name,
            BOUGHT[RISERS[0]] / 10.0,
            game.players[RISERS[0]].now_cost / 10.0,
            SELL[RISERS[0]] / 10.0,
        )
        if RISERS
        else "  no price rises in this synthetic squad"
    )

    POOL = transfer_candidates(PS, game, cfg, GWS, SQ, views=VIEWS)
    print("candidate pool %d players" % len(POOL))

    print("\n" + "=" * 78)
    print("PLAN, GW%d-%d" % (GWS[0], GWS[-1]))
    print("=" * 78)
    P = plan(game, SQ, PS, cfg, candidates=POOL)
    print(plan_table(P, game, VIEWS))
    print("\n" + plan_notes(P))
    print(
        "\nobjective %.3f, %s, %.2fs total"
        % (P.objective, P.solver_status, P.solve_seconds)
    )

    # -------------------------------------------------------------- gates
    print("\n" + "=" * 78)
    print("SANITY GATES")
    print("=" * 78)
    FAILURES: List[str] = []

    def gate(label: str, ok: bool, detail: str = "") -> None:
        print("  [%s] %-48s %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            FAILURES.append(label)

    FTS = [d.free_transfers_after for d in P.decisions]
    gate(
        "free transfers stay within 0..5",
        all(0 <= f <= scoring.MAX_BANKED_FREE_TRANSFERS for f in FTS),
        "after each GW: %s" % FTS,
    )
    BANKS = [d.bank_after for d in P.decisions]
    gate("bank never goes negative", all(b >= 0 for b in BANKS),
         "tenths: %s" % BANKS)
    gate(
        "every gameweek squad is legal",
        all(not squad_problems(d.squad, game, budget=10 ** 9) for d in P.decisions),
        "%d gameweeks" % len(P.decisions),
    )
    gate(
        "every lineup is legal",
        all(not lineup_problems(d.lineup, d.bench, d.squad, game) for d in P.decisions),
        ", ".join(formation_name([game.players[p].position for p in d.lineup]) for d in P.decisions),
    )
    gate(
        "net points = gross minus 4 per hit",
        all(
            abs(d.expected_points_net - (d.expected_points - 4 * d.hits)) < 1e-9
            for d in P.decisions
        ),
        "%d hits total" % sum(d.hits for d in P.decisions),
    )
    # Rebuild the money ledger by hand from the transfers and compare.
    LEDGER_OK = True
    CASH = SQ.bank
    HELD = set(IDS)
    for d in P.decisions:
        if d.chip == "freehit":
            if d.bank_after != CASH:
                LEDGER_OK = False
            continue
        for t in d.transfers:
            CASH += SELL.get(t.out_id, game.players[t.out_id].now_cost)
            CASH -= game.players[t.in_id].now_cost
            HELD.discard(t.out_id)
            HELD.add(t.in_id)
            SELL[t.in_id] = game.players[t.in_id].now_cost
        if d.bank_after != CASH:
            LEDGER_OK = False
    gate("bank ledger reproduces the MILP exactly", LEDGER_OK, "final %.1fm" % (CASH / 10.0))
    # And the free-transfer ledger.
    FT_OK = True
    HAVE = SQ.free_transfers
    for d in P.decisions:
        n = len(d.transfers)
        if d.chip in UNLIMITED_TRANSFER_CHIPS:
            expect_hits, used = 0, 0
        else:
            used = min(n, HAVE)
            expect_hits = n - used
        if d.hits != expect_hits:
            FT_OK = False
        HAVE = min(scoring.MAX_BANKED_FREE_TRANSFERS, HAVE - used + 1)
        if d.free_transfers_after != HAVE:
            FT_OK = False
    gate("free-transfer ledger reproduces the MILP exactly", FT_OK, "ends on %d" % HAVE)

    # ------------------------------------------------- hit threshold behaviour
    print("\n--- forced-hit scenario ---")
    HIT_CFG = _Config.load()
    HIT_CFG.optimizer.hit_threshold = 4.0
    P_HIT = plan(game, SQ, PS, HIT_CFG, candidates=POOL, patience_check=False)
    N_HITS = sum(d.hits for d in P_HIT.decisions)
    GAINS = []
    for d in P_HIT.decisions:
        if not d.hits:
            continue
        REM = [g for g in GWS if g >= d.gw]
        GAINS.append(
            sum(
                sum(VIEWS[g].get_xp(t.in_id) - VIEWS[g].get_xp(t.out_id) for g in REM)
                for t in d.transfers
            )
        )
    print(
        "hit_threshold 4.0 -> %d hits, per-gameweek horizon gains %s"
        % (N_HITS, ["%.2f" % g for g in GAINS])
    )
    NO_HIT_CFG = _Config.load()
    NO_HIT_CFG.optimizer.hit_threshold = 100.0
    P_NOHIT = plan(game, SQ, PS, NO_HIT_CFG, candidates=POOL, patience_check=False)
    print(
        "hit_threshold 100.0 -> %d hits, %d transfers"
        % (
            sum(d.hits for d in P_NOHIT.decisions),
            sum(len(d.transfers) for d in P_NOHIT.decisions),
        )
    )
    gate(
        "a hit is only taken when it clears the threshold",
        all(g >= HIT_CFG.optimizer.hit_threshold for g in GAINS),
        "min gain %.2f vs hurdle %.1f" % (min(GAINS) if GAINS else float("inf"), 4.0),
    )
    gate(
        "an unreachable threshold stops every hit",
        sum(d.hits for d in P_NOHIT.decisions) == 0,
        "%d hits" % sum(d.hits for d in P_NOHIT.decisions),
    )
    gate(
        "raising the hurdle never increases churn",
        sum(len(d.transfers) for d in P_NOHIT.decisions)
        <= sum(len(d.transfers) for d in P_HIT.decisions),
        "%d vs %d transfers"
        % (
            sum(len(d.transfers) for d in P_NOHIT.decisions),
            sum(len(d.transfers) for d in P_HIT.decisions),
        ),
    )

    # ------------------------------------------------------------- free hit
    print("\n--- free hit in GW%d ---" % GWS[1])
    P_FH = plan(
        game, SQ, PS, cfg, chips_available=["freehit"],
        chip_plan={GWS[1]: "freehit"}, candidates=POOL, patience_check=False,
    )
    print(plan_table(P_FH, game, VIEWS))
    BEFORE = set(P_FH.decisions[0].squad)
    DURING = set(P_FH.decisions[1].squad)
    # What the manager owns going into GW3, i.e. GW3's squad with GW3's own
    # transfers undone. That is what has to match the pre-free-hit squad.
    AFTER = set(P_FH.decisions[2].squad)
    AFTER = (AFTER - set(t.in_id for t in P_FH.decisions[2].transfers)) | set(
        t.out_id for t in P_FH.decisions[2].transfers
    )
    print(
        "  GW%d squad %d players, GW%d free-hit squad shares %d of them, GW%d squad "
        "before its own transfers shares %d"
        % (GWS[0], len(BEFORE), GWS[1], len(BEFORE & DURING), GWS[2], len(BEFORE & AFTER))
    )
    gate(
        "free hit reverts the squad exactly",
        AFTER == BEFORE,
        "%d changed" % len(BEFORE ^ AFTER),
    )
    gate(
        "free hit costs no transfers and no hits",
        P_FH.decisions[1].hits == 0 and not P_FH.decisions[1].transfers,
        "FT after %d" % P_FH.decisions[1].free_transfers_after,
    )
    gate(
        "free hit leaves the bank untouched",
        P_FH.decisions[1].bank_after == P_FH.decisions[0].bank_after,
        "%.1fm" % (P_FH.decisions[1].bank_after / 10.0),
    )
    gate(
        "free hit actually changes the fielded squad",
        DURING != BEFORE,
        "%d players swapped in for the week" % len(DURING - BEFORE),
    )
    FH_COST = sum(
        (game.players[p].now_cost if p not in BEFORE else sell_values(SQ, game).get(p, game.players[p].now_cost))
        for p in DURING
    )
    FH_BUDGET = P_FH.decisions[0].bank_after + sum(
        sell_values(SQ, game).get(p, game.players[p].now_cost) for p in BEFORE
    )
    gate(
        "free-hit squad is inside the free-hit budget",
        FH_COST <= FH_BUDGET,
        "%.1fm of %.1fm" % (FH_COST / 10.0, FH_BUDGET / 10.0),
    )

    # -------------------------------------------------------------- wildcard
    # The wildcard window opens at GW2 (scoring.CHIP_WINDOWS), so GW1 is not a
    # legal place to play it and normalise_chip_plan says so.
    print("\n--- wildcard in GW%d ---" % GWS[1])
    try:
        plan(game, SQ, PS, cfg, chip_plan={GWS[0]: "wildcard"}, candidates=POOL)
        WINDOW_OK = False
    except OptimizeError:
        WINDOW_OK = True
    gate("wildcard is refused outside its window", WINDOW_OK, "GW%d rejected" % GWS[0])

    P_WC = plan(
        game, SQ, PS, cfg, chips_available=["wildcard"],
        chip_plan={GWS[1]: "wildcard"}, candidates=POOL, patience_check=False,
    )
    WC = P_WC.decisions[1]
    WC_MOVES = len(WC.transfers)
    print(plan_table(P_WC, game, VIEWS))
    gate(
        "wildcard makes unlimited free transfers",
        WC.hits == 0 and WC_MOVES > 2,
        "%d moves, 0 hits, FT after %d" % (WC_MOVES, WC.free_transfers_after),
    )
    gate(
        "wildcard keeps its free transfers",
        WC.free_transfers_after
        == min(scoring.MAX_BANKED_FREE_TRANSFERS, P_WC.decisions[0].free_transfers_after + 1),
        "%d -> %d" % (P_WC.decisions[0].free_transfers_after, WC.free_transfers_after),
    )
    KEPT = set(WC.squad) - set(t.in_id for t in P_WC.decisions[2].transfers)
    gate(
        "wildcard squad is permanent",
        KEPT <= set(P_WC.decisions[2].squad),
        "GW%d keeps %d of the wildcard 15"
        % (GWS[2], len(set(WC.squad) & set(P_WC.decisions[2].squad))),
    )
    gate(
        "wildcard beats the plain plan over the horizon",
        P_WC.total_expected_points > P.total_expected_points,
        "%.1f vs %.1f" % (P_WC.total_expected_points, P.total_expected_points),
    )

    # --------------------------------------------------------- bboost / 3xc
    P_BB = plan(game, SQ, PS, cfg, chips_available=["bboost"],
                chip_plan={GWS[0]: "bboost"}, candidates=POOL, patience_check=False)
    P_TC = plan(game, SQ, PS, cfg, chips_available=["3xc"],
                chip_plan={GWS[0]: "3xc"}, candidates=POOL, patience_check=False)
    gate(
        "bench boost raises gameweek points",
        P_BB.decisions[0].expected_points > P.decisions[0].expected_points,
        "%.2f vs %.2f" % (P_BB.decisions[0].expected_points, P.decisions[0].expected_points),
    )
    gate(
        "triple captain raises gameweek points",
        P_TC.decisions[0].expected_points > P.decisions[0].expected_points,
        "%.2f vs %.2f" % (P_TC.decisions[0].expected_points, P.decisions[0].expected_points),
    )

    # ------------------------------------------------------------- patience
    print("\n--- patience test on an already-optimal squad ---")
    from gaffer.optimize.squad import pick_initial_squad

    GOOD = pick_initial_squad(PS, game, cfg, GWS)
    GOOD_SQ = synthetic_squad_state(GOOD.squad, game, GWS[0], bank=GOOD.bank_after, free_transfers=1)
    GOOD_POOL = transfer_candidates(PS, game, cfg, GWS, GOOD_SQ, views=VIEWS)
    P_GOOD = plan(game, GOOD_SQ, PS, cfg, candidates=GOOD_POOL)
    print(plan_table(P_GOOD, game, VIEWS))
    for note in P_GOOD.decisions[0].notes:
        print("   " + note)
    gate(
        "an optimal squad is left alone in GW%d" % GWS[0],
        not P_GOOD.decisions[0].transfers,
        "%d transfers, %d banked FT"
        % (len(P_GOOD.decisions[0].transfers), P_GOOD.decisions[0].free_transfers_after),
    )
    gate(
        "the plan explains why it stood still",
        any("bank" in n.lower() or "roll" in n.lower() for n in P_GOOD.decisions[0].notes),
    )
    gate(
        "total churn over the horizon stays low on a good squad",
        sum(len(d.transfers) for d in P_GOOD.decisions) <= len(GWS),
        "%d transfers over %d gameweeks"
        % (sum(len(d.transfers) for d in P_GOOD.decisions), len(GWS)),
    )

    # -------------------------------------------------------------- locks
    # Locking out a player you already own has to force a sale, not just a
    # benching, and locking one in has to survive every gameweek.
    LOCK_CFG = _Config.load()
    LOCK_OUT = max(IDS, key=lambda p: (horizon_xp(VIEWS, p, GWS), p))
    LOCK_IN = max(
        (p for p in POOL if p not in set(IDS) and game.players[p].position == scoring.DEF),
        key=lambda p: (horizon_xp(VIEWS, p, GWS), p),
    )
    LOCK_CFG.optimizer.locked_out = [LOCK_OUT]
    LOCK_CFG.optimizer.locked_in = [LOCK_IN]
    P_LOCK = plan(game, SQ, PS, LOCK_CFG, candidates=POOL, patience_check=False)
    gate(
        "locked_out owned player is sold and never fielded",
        all(LOCK_OUT not in d.squad and LOCK_OUT not in d.lineup for d in P_LOCK.decisions),
        "%s gone from GW%d" % (game.players[LOCK_OUT].web_name, GWS[0]),
    )
    gate(
        "locked_in player is held every gameweek",
        all(LOCK_IN in d.squad for d in P_LOCK.decisions),
        "%s held for %d gameweeks" % (game.players[LOCK_IN].web_name, len(P_LOCK.decisions)),
    )

    # ------------------------------------------------- the exported helpers
    FIRST = next(d for d in P.decisions if d.transfers)
    MOVE = FIRST.transfers[0]
    REM = [g for g in GWS if g >= FIRST.gw]
    RAW = evaluate_transfer(MOVE.out_id, MOVE.in_id, PS, game, REM, cfg, views=VIEWS)
    DEC_GAIN = evaluate_transfer(
        MOVE.out_id, MOVE.in_id, PS, game, REM, cfg, views=VIEWS, decayed=True
    )
    HAND = sum(VIEWS[g].get_xp(MOVE.in_id) - VIEWS[g].get_xp(MOVE.out_id) for g in REM)
    gate(
        "evaluate_transfer matches a hand-summed gain",
        abs(RAW - HAND) < 1e-9 and DEC_GAIN < RAW,
        "%.3f raw, %.3f decayed over GW%d-%d" % (RAW, DEC_GAIN, REM[0], REM[-1]),
    )
    gate(
        "should_take_hit brackets the threshold exactly",
        should_take_hit(cfg.optimizer.hit_threshold, cfg)
        and not should_take_hit(cfg.optimizer.hit_threshold - 1e-9, cfg),
        "hurdle %.1f" % cfg.optimizer.hit_threshold,
    )

    # --------------------------------------------------------- determinism
    P2 = plan(game, SQ, PS, cfg, candidates=POOL)
    gate(
        "two identical runs give the same plan",
        [sorted(d.squad) for d in P2.decisions] == [sorted(d.squad) for d in P.decisions]
        and abs(P2.objective - P.objective) < 1e-6,
        "objective %.6f vs %.6f" % (P2.objective, P.objective),
    )
    P3 = plan(game, SQ, PS, cfg, candidates=POOL, solver_name="cbc")
    gate(
        "CBC and HiGHS agree on the objective",
        abs(P3.objective - P.objective) < 5e-3,
        "%.6f vs %.6f" % (P3.objective, P.objective),
    )

    print(
        "\n%d gate(s) failed%s"
        % (len(FAILURES), (": " + ", ".join(FAILURES)) if FAILURES else "")
    )
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(_main())
