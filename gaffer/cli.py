"""Command line front end.

``python -m gaffer.cli <command>``. Every command prints an aligned ASCII
table, not JSON: this is the interface a human uses to decide a transfer, and a
wall of braces is not a decision aid.

Two habits run through all of it. Long operations narrate themselves on stderr
— a cold run fetches 587 player summaries and the user has to see it moving —
while the tables themselves go to stdout, so ``gaffer project | less`` stays
clean. And anything that costs seconds to compute (the projection set) is
reused from ``data/cache`` when it is newer than the data it was built from,
and silently recomputed when it is not.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gaffer.core import scoring
from gaffer.core.config import CACHE_DIR, PROJECT_ROOT, Config, ensure_dirs
from gaffer.core.types import (
    CaptainOption,
    GWDecision,
    Plan,
    ProjectionSet,
    SquadPick,
    SquadState,
)
from gaffer.data.cache import Cache
from gaffer.data.fpl_api import FPLClient, FPLError, FPLNotFound
from gaffer.data.loaders import GameState, load_game_state
from gaffer.model import bonus as bonus_model
from gaffer.model.xp import XPEngine, projections_cache_path

DEFAULT_PORT = 8770
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_UNAVAILABLE = 3


class PeerUnavailable(RuntimeError):
    """A sibling module (optimizer, backtester) is not built yet."""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _visible(text: str) -> int:
    return len(text)


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    align: Optional[str] = None,
    empty: str = "(nothing to show)",
) -> str:
    """A plain ASCII box table. ``align`` is one of 'l'/'r'/'c' per column."""
    cols = len(headers)
    body = [["" if c is None else str(c) for c in row] for row in rows]
    for row in body:
        while len(row) < cols:
            row.append("")
    if not body:
        widths = [_visible(str(h)) for h in headers]
        # The "nothing here" line spans every column, so widen the last one
        # until the box is big enough to hold it rather than overflowing it.
        inner = sum(widths) + 3 * (cols - 1)
        if _visible(empty) > inner:
            widths[-1] += _visible(empty) - inner
            inner = _visible(empty)
        return "\n".join([
            _rule(widths),
            _row(headers, widths, "l" * cols),
            _rule(widths),
            "| " + empty.ljust(inner) + " |",
            _rule(widths),
        ])
    widths = []
    for i in range(cols):
        widths.append(max(_visible(str(headers[i])), max(_visible(r[i]) for r in body)))
    align = (align or "l" * cols).ljust(cols, "l")
    out = [_rule(widths), _row(headers, widths, "l" * cols), _rule(widths)]
    for row in body:
        out.append(_row(row, widths, align))
    out.append(_rule(widths))
    return "\n".join(out)


def _rule(widths: Sequence[int]) -> str:
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def _row(cells: Sequence[Any], widths: Sequence[int], align: str) -> str:
    parts = []
    for i, cell in enumerate(cells):
        text = str(cell)
        width = widths[i]
        how = align[i] if i < len(align) else "l"
        if how == "r":
            parts.append(text.rjust(width))
        elif how == "c":
            parts.append(text.center(width))
        else:
            parts.append(text.ljust(width))
    return "| " + " | ".join(parts) + " |"


def heading(text: str) -> str:
    return "\n%s\n%s" % (text, "=" * len(text))


def f2(value: Any, spec: str = "%.2f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return spec % number


def pct(value: Any, spec: str = "%.0f%%") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return spec % (100.0 * number)


def trunc(text: Any, width: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


class Progress:
    """Narrates long work on stderr so stdout stays a clean table."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.t0 = time.time()

    def say(self, message: str) -> None:
        if not self.enabled:
            return
        sys.stderr.write("[%6.1fs] %s\n" % (time.time() - self.t0, message))
        sys.stderr.flush()

    def step(self, message: str) -> "_Step":
        return _Step(self, message)


class _Step:
    def __init__(self, progress: Progress, message: str) -> None:
        self.progress = progress
        self.message = message
        self.started = 0.0

    def __enter__(self) -> "_Step":
        self.started = time.time()
        self.progress.say("%s ..." % self.message)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        took = time.time() - self.started
        if exc is None:
            self.progress.say("%s: done in %.2fs" % (self.message, took))
        else:
            self.progress.say("%s: FAILED after %.2fs" % (self.message, took))
        return False


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


@dataclass
class Context:
    config: Config
    progress: Progress
    state: Optional[GameState] = None
    engine: Optional[XPEngine] = None
    projections: Optional[ProjectionSet] = None
    gws: List[int] = field(default_factory=list)
    projections_source: str = ""

    def load(self, force: bool = False, with_histories: bool = True) -> GameState:
        if self.state is not None and not force:
            return self.state
        with self.progress.step("loading game state (587 player summaries when cold)"):
            self.state = load_game_state(
                self.config, with_histories=with_histories, force=force,
                progress=self.progress.enabled,
            )
        self.progress.say("  %s" % self.state.summary())
        for warning in self.state.data_warnings:
            self.progress.say("  data warning: %s" % warning)
        return self.state

    def fit(self) -> XPEngine:
        if self.engine is not None and self.engine.fitted:
            return self.engine
        state = self.load()
        engine = XPEngine(self.config)
        with self.progress.step("fitting the model (team, minutes, attacking, defending, defcon, bonus)"):
            engine.fit(state)
        self.progress.say(
            "  fit: %s"
            % ", ".join("%s %.2fs" % (k, v) for k, v in engine.fit_seconds.items())
        )
        for warning in engine.warnings:
            self.progress.say("  model warning: %s" % warning)
        self.engine = engine
        return engine

    def project(self, gws: Sequence[int], allow_cache: bool = True, save: bool = True) -> ProjectionSet:
        """Projections for ``gws``, from the cache when it is still valid.

        "Valid" means the cached file was written *after* the bootstrap it was
        built from: prices, news and deadlines all live in bootstrap, so a
        projection older than the data behind it is simply wrong.
        """
        gws = [int(g) for g in gws]
        self.gws = gws
        path = projections_cache_path(gws[0], gws[-1])
        if allow_cache and _cache_is_current(path):
            engine = self.engine or XPEngine(self.config)
            loaded = engine.load_projections(gws[0], gws[-1])
            if loaded is not None:
                self.engine = engine
                self.projections = loaded
                self.projections_source = "cache (%s, %s)" % (
                    os.path.basename(path), _age_text(os.path.getmtime(path)))
                self.progress.say("projections: reusing %s" % self.projections_source)
                self.load()
                return loaded
        engine = self.fit()
        with self.progress.step("projecting %d players over GW%d-%d"
                                % (len(self.load().players), gws[0], gws[-1])):
            self.projections = engine.project(self.state, gws)
        self.projections_source = "computed"
        if save:
            try:
                written = engine.save_projections(self.projections)
                self.progress.say("  cached to %s" % written)
            except OSError as exc:
                self.progress.say("  could not cache projections: %s" % exc)
        return self.projections


def _cache_is_current(path: str) -> bool:
    if not os.path.exists(path):
        return False
    bootstrap = os.path.join(CACHE_DIR, "bootstrap.json")
    if os.path.exists(bootstrap) and os.path.getmtime(bootstrap) > os.path.getmtime(path):
        return False
    return True


def _age_text(mtime: float) -> str:
    age = max(0.0, time.time() - mtime)
    if age < 90:
        return "%.0fs old" % age
    if age < 5400:
        return "%.0fm old" % (age / 60.0)
    return "%.1fh old" % (age / 3600.0)


def parse_gws(text: Optional[str], state: Optional[GameState] = None,
              horizon: Optional[int] = None, config: Optional[Config] = None) -> List[int]:
    """"1-6", "1,3,5", "4" or None (meaning current .. current+horizon-1)."""
    if not text:
        first = state.current_gw if state is not None else 1
        span = int(horizon or (config.model.default_horizon if config else 6))
        last = min(scoring.TOTAL_GWS, first + span - 1)
        return list(range(first, last + 1))
    out: List[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            lo_text, hi_text = chunk.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise SystemExit("bad --gws %r: %d-%d runs backwards" % (text, lo, hi))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    out = sorted(set(out))
    bad = [g for g in out if g < 1 or g > scoring.TOTAL_GWS]
    if bad:
        raise SystemExit("bad --gws %r: %s outside 1-%d" % (text, bad, scoring.TOTAL_GWS))
    return out


def parse_ids(text: Optional[str], state: GameState) -> List[int]:
    """Player ids or names, comma separated. Names must match exactly one player."""
    if not text:
        return []
    out: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isdigit():
            pid = int(chunk)
            if pid not in state.players:
                raise SystemExit("no player with id %d" % pid)
            out.append(pid)
            continue
        matches = [p.id for p in state.players.values()
                   if p.web_name.lower() == chunk.lower()]
        if not matches:
            matches = [p.id for p in state.players.values()
                       if chunk.lower() in p.web_name.lower()]
        if not matches:
            raise SystemExit("no player matching %r" % chunk)
        if len(matches) > 1:
            names = ", ".join("%s (%d)" % (state.players[m].web_name, m) for m in matches[:8])
            raise SystemExit("%r matches %d players: %s" % (chunk, len(matches), names))
        out.append(matches[0])
    return out


def resolve_peer(name: str) -> Any:
    """Look up a function owned by another module, or explain its absence."""
    from fastapi import HTTPException  # local: only peer-using commands pay for it

    from gaffer.api.server import resolve

    try:
        return resolve(name)
    except HTTPException as exc:
        raise PeerUnavailable(str(exc.detail))


def resolve_squad(ctx: Context, entry_id: Optional[int], squad_text: Optional[str],
                  bank: Optional[float], free_transfers: Optional[int]) -> Tuple[SquadState, str]:
    """The squad to reason about: explicit ids, a real FPL entry, or the model's own."""
    state = ctx.load()
    if squad_text:
        ids = parse_ids(squad_text, state)
        picks = [
            SquadPick(player_id=pid, purchase_price=state.players[pid].now_cost,
                      selling_price=state.players[pid].now_cost, position_in_squad=i)
            for i, pid in enumerate(ids, start=1)
        ]
        return SquadState(gw=state.current_gw, picks=picks,
                          bank=int(round((bank or 0.0) * 10)),
                          free_transfers=int(free_transfers or 1)), "--squad"
    entry_id = entry_id if entry_id is not None else ctx.config.entry_id
    if entry_id is not None:
        from gaffer.api.server import STORE, load_squad_state
        STORE.config = ctx.config
        with ctx.progress.step("loading FPL entry %d" % entry_id):
            try:
                squad, meta = load_squad_state(int(entry_id))
            except Exception as exc:
                detail = getattr(exc, "detail", exc)
                raise SystemExit("could not load entry %d: %s" % (entry_id, detail))
        for warning in meta.get("warnings", []):
            ctx.progress.say("  %s" % warning)
        if bank is not None:
            squad.bank = int(round(bank * 10))
        if free_transfers is not None:
            squad.free_transfers = int(free_transfers)
        return squad, "entry %d" % entry_id

    pick_initial_squad = resolve_peer("pick_initial_squad")
    with ctx.progress.step("no entry id given; solving for the recommended squad"):
        decision = pick_initial_squad(ctx.projections, state, ctx.config, ctx.gws)
    picks = [
        SquadPick(player_id=pid, purchase_price=state.players[pid].now_cost,
                  selling_price=state.players[pid].now_cost, position_in_squad=i)
        for i, pid in enumerate(decision.squad, start=1)
    ]
    return SquadState(gw=decision.gw, picks=picks, bank=decision.bank_after,
                      free_transfers=int(free_transfers or decision.free_transfers_after)), \
        "recommended squad"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_refresh(args: argparse.Namespace, ctx: Context) -> int:
    ensure_dirs()
    cache = Cache(default_ttl=ctx.config.cache_ttl_seconds)
    client = FPLClient(ctx.config, cache)

    with ctx.progress.step("fetching bootstrap-static (players, teams, events, prices)"):
        client.bootstrap(force=True)
    with ctx.progress.step("fetching fixtures"):
        client.fixtures(force=True)
    if args.hard:
        ctx.progress.say("--hard: re-fetching all element summaries, this takes a minute")
    state = ctx.load(force=args.hard)

    print(heading("gaffer refresh — %s" % ctx.config.season))
    print(render_table(
        ["item", "value"],
        [
            ["season", state.season],
            ["teams", len(state.teams)],
            ["players", len(state.players)],
            ["fixtures", len(state.fixtures)],
            ["unscheduled fixtures", len([f for f in state.fixtures if f.gw is None])],
            ["current gameweek", "GW%d" % state.current_gw],
            ["deadline", state.deadline(state.current_gw) or "-"],
            ["finished gameweeks", len(state.finished_gws)],
            ["bootstrap totals are", "last season" if state.elements_are_prior_season
                                     else "this season to date"],
            ["promoted", ", ".join(state.short_name(t) for t in state.promoted_team_ids) or "-"],
        ],
        align="lr",
    ))

    ages = []
    for key in ("bootstrap", "fixtures"):
        age = cache.age_seconds(key)
        ages.append([key, "-" if age is None else "%.0fs" % age])
    summary_ages = []
    for name in os.listdir(CACHE_DIR):
        if name.startswith("element_summary_") and name.endswith(".json"):
            summary_ages.append(time.time() - os.path.getmtime(os.path.join(CACHE_DIR, name)))
    ages.append(["element summaries", "%d files, newest %s" % (
        len(summary_ages),
        "-" if not summary_ages else _age_text(time.time() - min(summary_ages)))])
    print("\n" + render_table(["cache key", "age"], ages, align="lr"))

    flagged = [
        p for p in state.players.values()
        if p.status != "a" and p.now_cost >= 55
    ]
    flagged.sort(key=lambda p: -p.now_cost)
    if flagged:
        print("\n" + render_table(
            ["player", "team", "pos", "price", "status", "news"],
            [[p.web_name, state.short_name(p.team_id), scoring.POS_NAME[p.position],
              f2(p.price, "%.1f"), p.status, trunc(p.news, 58)] for p in flagged[:15]],
            align="llrrll",
        ))
        print("(%d flagged players at £5.5m or more; showing %d)"
              % (len(flagged), min(15, len(flagged))))
    if state.data_warnings:
        print("\ndata warnings:")
        for warning in state.data_warnings:
            print("  - %s" % warning)
    print("\nNext: gaffer project --gws %d-%d"
          % (state.current_gw, min(scoring.TOTAL_GWS, state.current_gw + 5)))
    return EXIT_OK


def cmd_project(args: argparse.Namespace, ctx: Context) -> int:
    state = ctx.load()
    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    projections = ctx.project(gws, allow_cache=not args.fresh, save=not args.no_save)

    position = None
    if args.pos:
        key = args.pos.strip().upper()
        if key not in scoring.POS_ID:
            raise SystemExit("--pos must be GKP, DEF, MID or FWD")
        position = scoring.POS_ID[key]

    rows: List[List[Any]] = []
    ranked: List[Tuple[float, int]] = []
    for pid, per_gw in projections.projections.items():
        player = state.players.get(pid)
        if player is None:
            continue
        if position is not None and player.position != position:
            continue
        total = sum(per_gw[g].xp for g in gws if g in per_gw)
        xmins = sum(sum(f.xmins for f in per_gw[g].fixtures) for g in gws if g in per_gw)
        if xmins / len(gws) < args.min_xmins:
            continue
        if args.max_cost is not None and player.price > args.max_cost + 1e-9:
            continue
        ranked.append((total, pid))
    ranked.sort(key=lambda r: (-r[0], r[1]))

    for rank, (total, pid) in enumerate(ranked[: args.limit], start=1):
        player = state.players[pid]
        per_gw = projections.projections[pid]
        cells: List[Any] = [
            rank, trunc(player.web_name, 16), state.short_name(player.team_id),
            scoring.POS_NAME[player.position], f2(player.price, "%.1f"),
        ]
        for g in gws:
            gwp = per_gw.get(g)
            if gwp is None or gwp.is_blank:
                cells.append("-")
            elif gwp.n_fixtures > 1:
                cells.append("%s*" % f2(gwp.xp, "%.1f"))
            else:
                cells.append(f2(gwp.xp, "%.1f"))
        xmins = sum(sum(f.xmins for f in per_gw[g].fixtures) for g in gws if g in per_gw) / len(gws)
        cells.extend([f2(total), f2(total / max(player.price, 0.1), "%.2f"), f2(xmins, "%.0f")])
        rows.append(cells)

    headers = ["#", "player", "team", "pos", "£m"] + ["GW%d" % g for g in gws] + \
              ["total", "per £m", "xmins"]
    align = "llllr" + "r" * len(gws) + "rrr"
    print(heading("projected points GW%d-%d  (%s)" % (gws[0], gws[-1], ctx.projections_source)))
    print(render_table(headers, rows, align=align))
    print("* double gameweek, - blank. %d players ranked, showing %d."
          % (len(ranked), len(rows)))

    if args.explain:
        engine = ctx.fit()
        for pid in parse_ids(args.explain, state):
            print(heading("breakdown"))
            print(engine.explain(pid, gws[0]))
    return EXIT_OK


def cmd_squad(args: argparse.Namespace, ctx: Context) -> int:
    state = ctx.load()
    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    projections = ctx.project(gws, allow_cache=not args.fresh)

    config = ctx.config
    config.optimizer.locked_in = parse_ids(args.lock_in, state)
    config.optimizer.locked_out = parse_ids(args.lock_out, state)
    if args.decay is not None:
        config.optimizer.decay = float(args.decay)
    if getattr(args, "captain_ceiling", None) is not None:
        config.optimizer.captain_ceiling_weight = float(args.captain_ceiling)

    pick_initial_squad = resolve_peer("pick_initial_squad")
    budget = int(round(float(args.budget) * 10))
    with ctx.progress.step("solving the squad MILP (15 players, £%.1fm, GW%d-%d)"
                           % (args.budget, gws[0], gws[-1])):
        decision = pick_initial_squad(projections, state, config, gws, budget=budget,
                                      chip=args.chip)

    _print_decision(state, projections, decision, gws, title="recommended squad")
    return EXIT_OK


def _print_decision(state: GameState, projections: ProjectionSet, decision: GWDecision,
                    gws: Sequence[int], title: str) -> None:
    lineup = set(decision.lineup)
    role: Dict[int, str] = {pid: "XI" for pid in decision.lineup}
    for i, pid in enumerate(decision.bench):
        role[pid] = "bench %d" % i
    if decision.captain is not None:
        role[decision.captain] = "XI (C)"
    if decision.vice_captain is not None:
        role[decision.vice_captain] = "XI (V)"

    def rows_for(ids: Sequence[int]) -> List[List[Any]]:
        out = []
        for pid in ids:
            player = state.players[pid]
            total = sum(projections.xp(pid, g) for g in gws)
            out.append([
                scoring.POS_NAME[player.position], trunc(player.web_name, 16),
                state.short_name(player.team_id), f2(player.price, "%.1f"),
                f2(projections.xp(pid, decision.gw)), f2(total),
                f2(player.selected_by_percent, "%.1f"), role.get(pid, ""),
            ])
        return out

    ordered = sorted(decision.lineup,
                     key=lambda pid: (state.players[pid].position,
                                      -sum(projections.xp(pid, g) for g in gws)))
    headers = ["pos", "player", "team", "£m", "GW%d" % decision.gw,
               "GW%d-%d" % (gws[0], gws[-1]), "own%", "role"]
    align = "lllrrrrl"
    print(heading("%s — GW%d" % (title, decision.gw)))
    print(render_table(headers, rows_for(ordered), align=align))
    print("\nbench (in order):")
    print(render_table(headers, rows_for(decision.bench), align=align))

    cost = sum(state.players[pid].now_cost for pid in decision.squad)
    counts: Dict[str, int] = {}
    for pid in decision.lineup:
        counts[scoring.POS_NAME[state.players[pid].position]] = \
            counts.get(scoring.POS_NAME[state.players[pid].position], 0) + 1
    formation = "%d-%d-%d" % (counts.get("DEF", 0), counts.get("MID", 0), counts.get("FWD", 0))
    horizon_total = sum(projections.xp(pid, g) for pid in decision.squad for g in gws)
    print("\n" + render_table(
        ["item", "value"],
        [
            ["formation", formation],
            ["captain", _label(state, decision.captain)],
            ["vice captain", _label(state, decision.vice_captain)],
            ["squad cost", "£%.1fm" % (cost / 10.0)],
            ["in the bank", "£%.1fm" % (decision.bank_after / 10.0)],
            ["GW%d expected points" % decision.gw, f2(decision.expected_points)],
            ["GW%d-%d squad xP" % (gws[0], gws[-1]), f2(horizon_total)],
            ["chip", decision.chip or "-"],
            ["hits", decision.hits],
        ],
        align="lr",
    ))
    if decision.notes:
        print("\nnotes:")
        for note in decision.notes:
            print("  - %s" % note)


def _label(state: GameState, player_id: Optional[int]) -> str:
    if player_id is None:
        return "-"
    player = state.players.get(int(player_id))
    if player is None:
        return str(player_id)
    return "%s (%s, £%.1fm)" % (player.web_name, state.short_name(player.team_id), player.price)


def _short_label(state: GameState, player_id: Optional[int]) -> str:
    if player_id is None:
        return "-"
    player = state.players.get(int(player_id))
    if player is None:
        return str(player_id)
    return "%s (%s)" % (player.web_name, state.short_name(player.team_id))


def cmd_plan(args: argparse.Namespace, ctx: Context) -> int:
    state = ctx.load()
    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    projections = ctx.project(gws, allow_cache=not args.fresh)
    config = ctx.config
    config.optimizer.horizon = len(gws)
    config.optimizer.locked_in = parse_ids(args.lock_in, state)
    config.optimizer.locked_out = parse_ids(args.lock_out, state)
    if args.decay is not None:
        config.optimizer.decay = float(args.decay)

    squad, source = resolve_squad(ctx, args.entry_id, args.squad, args.bank, args.free_transfers)
    plan_fn = resolve_peer("plan")
    chips = [c.strip() for c in (args.chips or "").split(",") if c.strip()] or None
    with ctx.progress.step("planning GW%d-%d from %s" % (gws[0], gws[-1], source)):
        plan: Plan = plan_fn(state, squad, projections, config, chips)

    print(heading("plan GW%d-%d — squad from %s" % (gws[0], gws[-1], source)))
    rows = []
    for decision in plan.decisions:
        moves = []
        for transfer in decision.transfers:
            moves.append("%s -> %s"
                         % (state.players[transfer.out_id].web_name
                            if transfer.out_id in state.players else transfer.out_id,
                            state.players[transfer.in_id].web_name
                            if transfer.in_id in state.players else transfer.in_id))
        rows.append([
            "GW%d" % decision.gw,
            trunc("; ".join(moves) or "roll", 42),
            decision.hits or "",
            decision.chip or "",
            trunc(_short_label(state, decision.captain), 22),
            f2(decision.expected_points),
            f2(decision.expected_points_net),
            "%.1f" % (decision.bank_after / 10.0),
            decision.free_transfers_after,
        ])
    print(render_table(
        ["gw", "transfers", "hits", "chip", "captain", "xP", "net xP", "bank", "FT"],
        rows, align="llrrlrrrr",
    ))
    print("\n" + render_table(
        ["item", "value"],
        [
            ["solver status", plan.solver_status or "-"],
            ["solve seconds", f2(plan.solve_seconds)],
            ["objective", f2(plan.objective)],
            ["decay", f2(plan.decay)],
            ["total net expected points", f2(plan.total_expected_points)],
            ["hits taken", sum(d.hits for d in plan.decisions)],
        ],
        align="lr",
    ))
    for decision in plan.decisions:
        if decision.notes:
            print("\nGW%d:" % decision.gw)
            for note in decision.notes:
                print("  - %s" % note)
    return EXIT_OK


def cmd_captain(args: argparse.Namespace, ctx: Context) -> int:
    state = ctx.load()
    gw = int(args.gw or state.current_gw)
    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    if gw not in gws:
        gws = sorted(set(gws) | {gw})
        gws = list(range(min(gws), max(gws) + 1))
    projections = ctx.project(gws, allow_cache=not args.fresh)
    squad, source = resolve_squad(ctx, args.entry_id, args.squad, None, None)
    captain_options = resolve_peer("captain_options")
    options: List[CaptainOption] = captain_options(
        squad.player_ids(), projections, gw, state, ctx.config)

    print(heading("captain options GW%d — from %s" % (gw, source)))
    rows = []
    shown = options[: args.limit]
    for rank, option in enumerate(shown, start=1):
        player = state.players.get(option.player_id)
        rows.append([
            rank,
            trunc(player.web_name if player else option.player_id, 16),
            state.short_name(player.team_id) if player else "-",
            scoring.POS_NAME[player.position] if player else "-",
            f2(player.price, "%.1f") if player else "-",
            f2(option.xp), f2(option.sd), pct(option.p_haul),
            # effective_ownership is already a percentage: ownership% + captaincy%,
            # and can exceed 100 for a template captain.
            f2(option.effective_ownership, "%.0f%%"),
            f2(option.ev_vs_field, "%+.2f"),
        ])
    print(render_table(
        ["#", "player", "team", "pos", "£m", "xP", "sd", "P(10+)", "EO", "ev vs field"],
        rows, align="lllrrrrrrr",
    ))
    print("\nEO is ownership% plus captaincy%; over 100% means a haul you do not "
          "own costs you rank.\n")
    for rank, option in enumerate(shown[: args.explain], start=1):
        player = state.players.get(option.player_id)
        print("%d. %s — %s" % (rank, player.web_name if player else option.player_id,
                               option.rationale or "(no rationale)"))
    return EXIT_OK


def cmd_chips(args: argparse.Namespace, ctx: Context) -> int:
    state = ctx.load()
    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    projections = ctx.project(gws, allow_cache=not args.fresh)
    squad, source = resolve_squad(ctx, args.entry_id, args.squad, None, None)

    detect = resolve_peer("detect_double_blank_gws")
    recommend = resolve_peer("recommend_chips")
    with ctx.progress.step("scanning fixtures for doubles and blanks"):
        double_blank = detect(state)
    with ctx.progress.step("evaluating chips"):
        recommendations = recommend(state, squad, projections, ctx.config)

    print(heading("chips — squad from %s" % source))
    rows = []
    seen_warnings: List[str] = []
    for rec in recommendations:
        alternatives = ", ".join(
            "GW%s %+.1f" % (alt.get("gw"), alt.get("points_gain") or 0.0)
            for alt in (rec.get("alternatives") or [])[:2]
        )
        rows.append([
            str(rec.get("label") or rec.get("chip") or "?"),
            "GW%s" % rec.get("gw") if rec.get("gw") else "-",
            f2(rec.get("points_gain")),
            str(rec.get("confidence") or ""),
            str(rec.get("urgency") or ""),
            "" if rec.get("available", True) else "used",
            alternatives or "-",
        ])
        for warning in rec.get("warnings") or []:
            if warning not in seen_warnings:
                seen_warnings.append(warning)
    print(render_table(
        ["chip", "gw", "gain", "confidence", "urgency", "state", "next best"],
        rows, align="llrlllr", empty="no chip is worth playing yet"))
    for rec in recommendations:
        if rec.get("reason"):
            print("\n%s (GW%s, %+.1f pts): %s"
                  % (rec.get("label") or rec.get("chip"), rec.get("gw"),
                     rec.get("points_gain") or 0.0, rec["reason"]))
    for warning in seen_warnings:
        print("\n  ! %s" % warning)

    interesting = []
    for gw in sorted(double_blank):
        info = double_blank[gw] or {}
        doubles = info.get("doubles") or []
        blanks = info.get("blanks") or []
        if not doubles and not blanks:
            continue
        interesting.append([
            "GW%d" % gw,
            trunc(", ".join(state.short_name(t) for t in doubles) or "-", 44),
            trunc(", ".join(state.short_name(t) for t in blanks) or "-", 44),
        ])
    print("\n" + render_table(["gw", "doubles", "blanks"], interesting, align="lll",
                              empty="every club plays exactly once in every gameweek"))

    from gaffer.api.server import _chip_expiry
    expiry = _chip_expiry(state, squad)
    print("\n" + render_table(
        ["item", "value"],
        [
            ["first-half chips expire", "GW%d deadline (%s)"
             % (expiry["first_half_last_gw"], expiry["deadline"] or "unknown")],
            ["gameweeks remaining", expiry["gws_remaining"]],
            ["unused first-half chips", ", ".join(expiry["unused_first_half_chips"]) or "none"],
            ["expiry escalation", expiry["level"]],
        ],
        align="lr",
    ))
    for warning in expiry["warnings"]:
        print("  ! %s" % warning)
    return EXIT_OK


def cmd_backtest(args: argparse.Namespace, ctx: Context) -> int:
    backtest_season = resolve_peer("backtest_season")
    gws = parse_gws(args.gws) if args.gws else None
    span = "GW%d-%d" % (gws[0], gws[-1]) if gws else "the whole season"
    ctx.progress.say("backtesting %s over %s — the season is replayed one gameweek "
                     "at a time, refitting the model from pre-deadline data each "
                     "time, so this takes minutes" % (args.season, span))
    with ctx.progress.step("backtest %s" % args.season):
        report = backtest_season(args.season, gws, ctx.config)

    print(heading("backtest %s — %s" % (report.get("season", args.season), span)))
    universe = report.get("universe") or {}
    print(render_table(
        ["item", "value"],
        [[k.replace("_", " "), v] for k, v in universe.items()]
        + [["gameweeks replayed", len(report.get("gws") or [])],
           ["seconds", (report.get("timings") or {}).get("total_seconds", "-")]],
        align="lr",
    ))

    # One table per subset, decision-relevant ones first. A user who only sees
    # the full universe sees the table that is dominated by players who never
    # play, which is the one table nothing should be concluded from.
    subsets = report.get("subsets") or [
        {"key": "full_universe", "label": "every player with a fixture",
         "predictors_key": "predictors", "judged": False, "note": ""},
    ]
    for subset in subsets:
        predictors = report.get(subset.get("predictors_key")) or {}
        if not predictors:
            continue
        rows = []
        for key, entry in predictors.items():
            captain = entry.get("captain") or {}
            rows.append([
                key, f2(entry.get("rmse"), "%.3f"), f2(entry.get("mae"), "%.3f"),
                f2(entry.get("spearman"), "%.4f"),
                f2(entry.get("spearman_per_gw_mean"), "%.4f"),
                f2(entry.get("top20_hit_rate"), "%.3f"),
                # `accuracy` (captain was THE single top scorer of ~840) is a
                # lottery and reads as 0.000 for every predictor. The top-5/10
                # rates are the part a captain pick can actually control.
                f2(captain.get("top5_rate"), "%.3f"),
                f2(captain.get("top10_rate"), "%.3f"),
                f2(captain.get("mean_points_lost"), "%.2f"),
            ])
        print("\n%s — %s (%s rows, %s gameweeks)"
              % ("JUDGED" if subset.get("judged") else "DIAGNOSTIC",
                 subset.get("label", subset.get("key", "")),
                 subset.get("rows", "?"), subset.get("gameweeks", "?")))
        for line in textwrap.wrap(subset.get("note", ""), width=96):
            print("  " + line)
        print(render_table(
            ["predictor", "RMSE", "MAE", "rho", "rho/GW", "top20", "capt top5", "capt top10", "capt lost"],
            rows, align="lrrrrrrrr"))

    predictors = report.get("predictors") or {}

    model = predictors.get("model") or {}
    by_position = model.get("by_position") or {}
    if by_position:
        rows = []
        for label, metrics in by_position.items():
            rows.append([label, metrics.get("n"), f2(metrics.get("rmse"), "%.3f"),
                         f2(metrics.get("mae"), "%.3f"),
                         f2(metrics.get("spearman"), "%.4f"),
                         f2(metrics.get("mean_pred")), f2(metrics.get("mean_actual"))])
        print("\n" + render_table(
            ["position", "n", "RMSE", "MAE", "rho", "mean xP", "mean actual"],
            rows, align="lrrrrrr"))

    judgement = report.get("verdict") or {}
    if judgement:
        basis = ", ".join(k.replace("_", " ") for k in judgement.get("basis", []))
        print("\nverdict (judged on %s): %s"
              % (basis or "no decision-relevant subset",
                 "the model beats every baseline where the decisions are made"
                 if judgement.get("beats_all")
                 else "the model does NOT beat every baseline where the decisions are made"))
        for line in judgement.get("lines", []):
            if not line:
                print()
                continue
            own = line[: len(line) - len(line.lstrip())]
            body = line.strip()
            # A column-padded line is a table row; wrapping it would split a
            # number across two lines. Print those as they are.
            if "  " in body:
                print("  " + own + body)
                continue
            for index, wrapped in enumerate(textwrap.wrap(body, width=96) or [""]):
                print("  %s%s%s" % (own, "" if index == 0 else "  ", wrapped))
    invariants = report.get("invariants") or {}
    if invariants:
        print("\ninvariants: %s" % ", ".join("%s %s" % (k, v) for k, v in invariants.items()))
    for warning in (report.get("warnings") or [])[:5]:
        print("  ! %s" % warning)
    if not args.no_save:
        try:
            write_report = resolve_peer("write_report")
        except PeerUnavailable as exc:
            print("\ncould not write the report file: %s" % exc)
        else:
            print("\nfull report: %s" % write_report(report))
    return EXIT_OK


def cmd_autorefresh(args: argparse.Namespace, ctx: Context) -> int:
    """The unattended path: decide whether to refresh, then do only that.

    Called hourly by launchd, so the common case has to be a cheap no. All the
    policy lives in gaffer.ops.autorefresh; this is the reporting shell around
    it. It never runs the backtest.
    """
    from gaffer.ops import autorefresh as auto

    config = ctx.config
    if getattr(args, "horizon", None):
        config.model.default_horizon = int(args.horizon)

    now = auto.utcnow()
    decision = auto.plan(config, now=now, force=args.force)

    def report(outcome: str, extra: Optional[List[List[Any]]] = None) -> None:
        rows: List[List[Any]] = [
            ["decision", decision.action],
            ["why", decision.reason],
            ["data age", auto.human_seconds(decision.cache_age)],
            ["staleness limit", auto.human_seconds(decision.max_age)],
            ["next deadline", "GW%d in %s" % (decision.next_gw,
                                              auto.human_seconds(decision.seconds_to_deadline))
                              if decision.next_gw else "-"],
            ["deadline window", "yes (%s limit)" % auto.human_seconds(auto.PRE_DEADLINE_MAX_AGE)
                                if decision.near_deadline else "no"],
            ["horizon", "GW%d-%d" % (decision.gws[0], decision.gws[-1])],
            ["projections", "%s (%s)" % (os.path.basename(decision.projections_path),
                                         "current" if decision.projections_fresh else "stale")],
        ]
        if decision.consecutive_failures:
            rows.append(["failure streak", "%d, retry after %s" % (
                decision.consecutive_failures,
                decision.retry_after.isoformat() if decision.retry_after else "-")])
        rows.extend(extra or [])
        rows.append(["outcome", outcome])
        print(heading("gaffer autorefresh — %s" % now.isoformat(timespec="seconds")))
        print(render_table(["item", "value"], rows, align="ll"))

    lock = auto.SingleFlight()
    if not lock.acquire():
        # Not an error: launchd fired while a slow run was still going.
        report("skipped — another autorefresh holds %s (%s)"
               % (os.path.basename(auto.LOCK_PATH), lock.holder or "unknown pid"))
        return EXIT_OK
    try:
        if args.dry_run:
            report("dry run — would %s" % ("do nothing" if not decision.works
                                           else decision.action))
            return EXIT_OK
        if not decision.works:
            report("nothing to do")
            return EXIT_OK

        ctx.progress.say("autorefresh: %s (%s)" % (decision.action, decision.reason))
        try:
            stats = auto.perform(decision, config, ctx.progress)
        except Exception as exc:  # noqa: BLE001 - any failure must back off, not crash
            state = auto.record_failure(now, "%s: %s" % (type(exc).__name__, exc))
            delay = auto.backoff_seconds(int(state["consecutive_failures"]))
            report("FAILED: %s: %s" % (type(exc).__name__, exc),
                   [["backing off", "%s (failure %d), next attempt %s"
                     % (auto.human_seconds(delay), state["consecutive_failures"],
                        state["retry_after"])]])
            return EXIT_FAIL
        auto.record_success(now, stats)
        report("done in %.1fs" % stats["seconds"], [
            ["fetched from FPL", "yes" if stats["fetched"] else "no (local recompute)"],
            ["load / fit / project", "%.1fs / %.1fs / %.1fs" % (
                stats["load_seconds"], stats["fit_seconds"], stats["project_seconds"])],
            ["players projected", stats["players"]],
            ["written", stats["projections_path"]],
        ])
        return EXIT_OK
    finally:
        lock.release()


def cmd_serve(args: argparse.Namespace, ctx: Context) -> int:
    import uvicorn

    from gaffer.api import server as server_module

    if args.horizon:
        server_module.STORE.horizon = int(args.horizon)
    server_module.STORE.config = ctx.config
    url = "http://%s:%d" % ("localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host,
                            args.port)
    print(heading("gaffer server"))
    print(render_table(
        ["what", "where"],
        [
            ["dashboard", url],
            ["health", url + "/api/health"],
            ["API docs", url + "/api/docs"],
            ["static files", os.path.join(PROJECT_ROOT, "gaffer", "web")],
            ["horizon", "%d gameweeks" % server_module.STORE.horizon],
            ["snapshot TTL", "%ds" % server_module.STORE.ttl],
        ],
        align="ll",
    ))
    from gaffer.api.auth import configured_password, startup_warning

    warning = startup_warning(args.host)
    if warning:
        print("\n  !! %s" % warning)
    elif configured_password():
        print("\n  password protection is ON (GAFFER_PASSWORD is set).")

    print("\nThe browser must never call fantasy.premierleague.com directly — it "
          "sends no CORS headers.\nEverything is proxied and cached here. Ctrl-C to stop.\n")
    sys.stdout.flush()
    uvicorn.run(server_module.app, host=args.host, port=args.port, log_level=args.log_level)
    return EXIT_OK


def cmd_h2h(args: argparse.Namespace, ctx: Context) -> int:
    """Score a head-to-head tie on win probability, not expected points."""
    from gaffer.optimize import h2h as H

    state = ctx.load()
    gw = int(args.gw) if args.gw else state.current_gw
    engine = ctx.fit()
    ctx.project([gw], allow_cache=not args.fresh)

    my_id = args.entry or ctx.config.entry_id
    if not my_id:
        print("no entry id: pass --entry or set FPL_ENTRY_ID")
        return EXIT_FAIL

    client = FPLClient(ctx.config, Cache(default_ttl=ctx.config.cache_ttl_seconds))
    stale_note: List[str] = []
    names = {p.id: p.web_name for p in state.players.values()}

    def load(entry_id: int, label: str):
        """Their XI for `gw`, falling back to the last one FPL published.

        This fallback is what makes the command usable at all. Squads go public
        only at the deadline, which is the moment the decision is already made —
        so before a deadline the only estimate of an opponent's XI is the last
        one they showed, and using it beats refusing to answer.
        """
        payload = None
        used_gw = gw
        for candidate in (gw, gw - 1, gw - 2):
            if candidate < 1:
                continue
            try:
                payload = client.entry_picks(int(entry_id), candidate)
                used_gw = candidate
                break
            except FPLNotFound:
                continue
            except FPLError as exc:
                print("%s (%s): FPL API error: %s" % (label, entry_id, exc))
                return None
        if payload is None:
            print("%s (%s): no published picks at GW%d or earlier."
                  % (label, entry_id, gw))
            return None
        if used_gw != gw:
            stale_note.append("%s: GW%d picks are not public yet, so this uses "
                              "their GW%d squad as the estimate."
                              % (label, gw, used_gw))
        lineup, captain = H.lineup_from_picks(payload)
        if len(lineup) < 11:
            print("%s (%s): only %d starters found" % (label, entry_id, len(lineup)))
            return None
        return lineup, captain

    mine = load(my_id, "your team")
    theirs = load(args.opponent, "opponent")
    if not mine or not theirs:
        return EXIT_FAIL

    my_dist = H.squad_distribution(engine, gw, mine[0], captain=mine[1])
    their_dist = H.squad_distribution(engine, gw, theirs[0], captain=theirs[1])
    my_stats = H.summarise(*my_dist)
    their_stats = H.summarise(*their_dist)
    result = H.win_probability(my_dist, their_dist)

    print(heading("head to head — GW%d" % gw))
    for note in stale_note:
        print("  ! %s" % note)
    print(render_table(
        ["side", "mean", "sd", "10th", "median", "90th", "captain"],
        [["you (%s)" % my_id, f2(my_stats["mean"], "%.1f"), f2(my_stats["sd"], "%.1f"),
          my_stats["p10"], my_stats["median"], my_stats["p90"],
          names.get(mine[1], "-")],
         ["them (%s)" % args.opponent, f2(their_stats["mean"], "%.1f"),
          f2(their_stats["sd"], "%.1f"), their_stats["p10"], their_stats["median"],
          their_stats["p90"], names.get(theirs[1], "-")]],
        align="lrrrrrl"))

    print("\n" + render_table(
        ["outcome", "probability"],
        [["win", pct(result["win"])], ["draw", pct(result["draw"])],
         ["loss", pct(result["loss"])]],
        align="lr"))

    print("\n" + textwrap.fill(H.advice(my_stats, their_stats, result), 78))

    options = H.captain_options(engine, gw, mine[0], their_dist, names)
    if options:
        print(heading("armband, ranked by win probability"))
        rows = []
        for i, o in enumerate(options[:8], 1):
            rows.append([i, trunc(o["name"], 16), f2(o["mean"], "%.1f"),
                         f2(o["sd"], "%.1f"), o["p90"], pct(o["win"]),
                         pct(o["score"])])
        print(render_table(
            ["#", "captain", "mean", "sd", "90th", "P(win)", "win+½draw"],
            rows, align="rlrrrrr"))
        best, current = options[0], None
        for o in options:
            if o["player_id"] == mine[1]:
                current = o
        if current and best["player_id"] != current["player_id"]:
            print("\n%s over %s is worth %+.1f points of win probability this week, "
                  "even though it costs %+.2f expected points."
                  % (best["name"], current["name"],
                     100 * (best["score"] - current["score"]),
                     best["mean"] - current["mean"]))
    return EXIT_OK


def cmd_export(args: argparse.Namespace, ctx: Context) -> int:
    """Freeze the dashboard to disk for a static host."""
    from gaffer.ops.export import ExportError, export_site, format_summary

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    print(heading("gaffer export"))
    sys.stdout.flush()
    try:
        with ctx.progress.step("exporting the dashboard to %s" % out_dir):
            summary = export_site(
                out_dir,
                config=ctx.config,
                horizon=int(args.horizon),
                include_players=not args.no_players,
                progress=ctx.progress.say,
            )
    except ExportError as exc:
        print("\nexport failed: %s" % exc)
        return EXIT_FAIL

    print("\n" + format_summary(summary))
    print(render_table(
        ["item", "value"],
        [
            ["output", summary["out_dir"]],
            ["gameweeks", "GW%d-%d" % (summary["gws"][0], summary["gws"][-1])],
            ["files", summary["files"]],
            ["JSON size", "%.1f MB" % (summary["bytes"] / (1024.0 * 1024.0))],
            ["player breakdowns", summary["players"]],
            ["generated at", summary["generated_at"]],
            ["took", "%.1fs" % summary["seconds"]],
        ],
        align="lr",
    ))
    print("\nServe it locally to check it:  python -m http.server -d %s 8000" % summary["out_dir"])
    return EXIT_OK


def cmd_verify(args: argparse.Namespace, ctx: Context) -> int:
    checks: List[Tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        checks.append((label, bool(ok), detail))

    print(heading("1. scoring constants against the live API"))
    sys.stdout.flush()
    proc = subprocess.run(
        [sys.executable, "-m", "gaffer.core.scoring", "--verify"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    output = (proc.stdout + proc.stderr).strip()
    print(output or "(no output)")
    sys.stdout.flush()
    check("scoring constants match the API", proc.returncode == 0,
          "exit %d" % proc.returncode)

    state = ctx.load()
    check("20 teams", len(state.teams) == 20, "%d" % len(state.teams))
    check("players loaded", len(state.players) > 400, "%d" % len(state.players))
    check("380 fixtures", len(state.fixtures) == 380, "%d" % len(state.fixtures))
    check("current gameweek in range", 1 <= state.current_gw <= scoring.TOTAL_GWS,
          "GW%d" % state.current_gw)
    check("current gameweek has a deadline", bool(state.deadline(state.current_gw)),
          state.deadline(state.current_gw) or "missing")
    check("38 events", len(state.events) == scoring.TOTAL_GWS, "%d" % len(state.events))

    fixture_ids = [f.id for f in state.fixtures]
    check("fixture ids unique", len(set(fixture_ids)) == len(fixture_ids),
          "%d unique of %d" % (len(set(fixture_ids)), len(fixture_ids)))
    unknown_team = [f.id for f in state.fixtures
                    if f.team_h not in state.teams or f.team_a not in state.teams]
    check("every fixture maps to known teams", not unknown_team, str(unknown_team[:5]))
    per_team: Dict[int, int] = {t: 0 for t in state.teams}
    for fixture in state.fixtures:
        per_team[fixture.team_h] += 1
        per_team[fixture.team_a] += 1
    odd = {state.short_name(t): n for t, n in per_team.items() if n != 38}
    check("every club has 38 fixtures", not odd, str(odd))
    priced = [p for p in state.players.values() if p.now_cost < 35 or p.now_cost > 200]
    check("prices inside £3.5m-£20.0m", not priced,
          ", ".join("%s %.1f" % (p.web_name, p.price) for p in priced[:5]))

    gws = parse_gws(args.gws, state, args.horizon, ctx.config)
    projections = ctx.project(gws, allow_cache=not args.fresh)
    check("projection set covers the horizon",
          projections.first_gw == gws[0] and projections.last_gw == gws[-1],
          "GW%d-%d" % (projections.first_gw, projections.last_gw))
    check("every player projected",
          len(projections.projections) == len(state.players),
          "%d of %d" % (len(projections.projections), len(state.players)))

    nan_fields: List[str] = []
    bad_sum = 0.0
    negative = []
    rows = 0
    bonus_by_fixture: Dict[int, float] = {}
    for pid, per_gw in projections.projections.items():
        for gw, gwp in per_gw.items():
            rows += 1
            if not math.isfinite(gwp.xp) or not math.isfinite(gwp.sd):
                nan_fields.append("player %d GW%d total" % (pid, gw))
            component_total = 0.0
            for fp in gwp.fixtures:
                for name, value in vars(fp).items():
                    if isinstance(value, float) and not math.isfinite(value):
                        nan_fields.append("player %d GW%d %s" % (pid, gw, name))
                component_total += sum(fp.components().values())
                bonus_by_fixture[fp.fixture_id] = \
                    bonus_by_fixture.get(fp.fixture_id, 0.0) + fp.xp_bonus
            bad_sum = max(bad_sum, abs(component_total - gwp.xp))
            if gwp.xp < -1e-9:
                negative.append((pid, gw, gwp.xp))
    check("no NaN or infinity in projections", not nan_fields,
          "%d bad fields, e.g. %s" % (len(nan_fields), nan_fields[:3]))
    check("components sum to the gameweek total", bad_sum < 1e-9, "max residual %.2e" % bad_sum)
    check("no negative gameweek totals", not negative, str(negative[:3]))
    # A fixture pays 3+2+1 = 6 only when the top three BPS scores are distinct.
    # Ties pay 7, 8 or 9 and 2025/26 averaged 6.366 per fixture, so the check is
    # against the sanity limit rather than 6: a pot of 6.4 is the game working,
    # a pot above the limit is the bonus ranker broken.
    limit = bonus_model.BONUS_POT_SANITY_LIMIT
    over = {fid: total for fid, total in bonus_by_fixture.items() if total > limit + 1e-6}
    check("expected bonus per fixture <= %.0f" % limit, not over,
          "%d fixtures over, e.g. %s" % (len(over), list(over.items())[:3]))
    check("projection rows present", rows == len(state.players) * len(gws),
          "%d rows for %d players x %d gws" % (rows, len(state.players), len(gws)))

    print(heading("2. data and projection integrity"))
    print(render_table(
        ["check", "result", "detail"],
        [[label, "PASS" if ok else "FAIL", trunc(detail, 60)] for label, ok, detail in checks],
        align="lll",
    ))
    failed = [c for c in checks if not c[1]]
    print("\n%d checks, %d passed, %d failed" % (len(checks), len(checks) - len(failed), len(failed)))

    # Odds are optional, so their absence is not a failed check — but whether
    # the market is in play changes how much to trust a pre-season forecast, and
    # that should never be something you have to guess at.
    report = getattr(ctx.engine, "odds_report", None) if ctx.engine else None
    if report is not None:
        from gaffer.data import odds as odds_mod
        age = odds_mod.cache_age_hours(CACHE_DIR)
        print(heading("3. bookmaker odds (optional)"))
        print(render_table(
            ["item", "value"],
            [
                ["fixtures priced", report.get("applied", 0)],
                ["status", trunc(report.get("status", "—"), 70)],
                ["cache age", "%.1fh" % age if age is not None else "no cache"],
                ["unrecognised clubs", ", ".join(report.get("unmatched_names") or []) or "none"],
            ],
            align="ll",
        ))
        if not report.get("applied"):
            print("  The model is running on its own fitted ratings. Set ODDS_API_KEY "
                  "to overlay the betting market.")
    return EXIT_OK if not failed else EXIT_FAIL


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaffer",
        description="FPL 2026/27 expected-points engine. Projects every player, "
                    "picks the squad, plans the transfers and explains itself.",
        epilog="Typical first run:  python -m gaffer.cli refresh  &&  "
               "python -m gaffer.cli project --gws 1-6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="path to config.json (default: the project root)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress progress output on stderr")
    # The same two flags on every subcommand, so `gaffer serve --quiet` works as
    # well as `gaffer --quiet serve`. SUPPRESS keeps the subparser copy from
    # overwriting a value the main parser already took.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--quiet", "-q", action="store_true", default=argparse.SUPPRESS,
                        help="suppress progress output on stderr")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    def add(name: str, help_text: str, description: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, help=help_text, description=description,
                                     parents=[common])

    def horizon_args(sub: argparse.ArgumentParser, with_gws: bool = True) -> None:
        if with_gws:
            sub.add_argument("--gws", help="gameweeks: '1-6', '1,3,5' or '4'. "
                                           "Default: the next --horizon gameweeks.")
        sub.add_argument("--horizon", type=int,
                         help="number of gameweeks to look ahead (default %d)"
                              % Config().model.default_horizon)
        sub.add_argument("--fresh", action="store_true",
                         help="recompute projections instead of reusing data/cache")

    p = add("refresh", "re-fetch FPL data into data/cache",
            "Force a fresh bootstrap and fixture list from the FPL API. "
            "Player histories are re-fetched only with --hard.")
    p.add_argument("--hard", action="store_true",
                   help="also re-fetch all 587 element summaries (takes about a minute)")
    p.set_defaults(func=cmd_refresh)

    p = add("project", "project expected points for every player",
            "Fit the model and rank every player by projected points over a "
            "gameweek range. Doubles are marked *, blanks -.")
    horizon_args(p)
    p.add_argument("--limit", type=int, default=30, help="rows to print (default 30)")
    p.add_argument("--pos", help="restrict to GKP, DEF, MID or FWD")
    p.add_argument("--max-cost", type=float, help="price ceiling in £m")
    p.add_argument("--min-xmins", type=float, default=0.0,
                   help="drop players below this average projected minutes")
    p.add_argument("--explain", help="print the full breakdown for these players "
                                     "(ids or names, comma separated)")
    p.add_argument("--no-save", action="store_true", help="do not write the projection cache")
    p.set_defaults(func=cmd_project)

    p = add("squad", "pick the best legal 15 for a budget",
            "Solve for the highest expected-points squad over the horizon, subject "
            "to budget, position quotas and the 3-per-club limit.")
    horizon_args(p)
    p.add_argument("--budget", type=float, default=scoring.BUDGET_TENTHS / 10.0,
                   help="budget in £m (default %.1f)" % (scoring.BUDGET_TENTHS / 10.0))
    p.add_argument("--lock-in", help="players that must be in the squad (ids or names)")
    p.add_argument("--lock-out", help="players that must not be (ids or names)")
    p.add_argument("--captain-ceiling", type=float, metavar="W",
                   help="value the armband's ceiling as well as its mean: the captain "
                        "slot scores xp + W*sd. 0 (default) is pure expected points; "
                        "0.3-0.5 buys variance, which is what climbing rank needs")
    p.add_argument("--chip", choices=sorted(scoring.CHIP_WINDOWS),
                   help="assume this chip is active in the first gameweek")
    p.add_argument("--decay", type=float, help="per-gameweek weight decay (default %.2f)"
                                               % Config().optimizer.decay)
    p.set_defaults(func=cmd_squad)

    p = add("plan", "plan transfers over the horizon",
            "Multi-gameweek transfer plan for an existing squad: who to sell, when "
            "to roll, when a -4 is worth taking.")
    horizon_args(p)
    p.add_argument("--entry-id", type=int, help="your FPL entry (team) id")
    p.add_argument("--squad", help="15 player ids or names, comma separated")
    p.add_argument("--bank", type=float, help="money in the bank, £m")
    p.add_argument("--free-transfers", type=int, help="free transfers available now")
    p.add_argument("--chips", help="chips the planner may use, comma separated")
    p.add_argument("--lock-in", help="players that must stay")
    p.add_argument("--lock-out", help="players that must go / never come in")
    p.add_argument("--decay", type=float, help="per-gameweek weight decay")
    p.set_defaults(func=cmd_plan)

    p = add("captain", "rank captain options for a gameweek",
            "Expected points, standard deviation, haul probability and effective "
            "ownership for every captain candidate.")
    horizon_args(p)
    p.add_argument("--gw", type=int, help="gameweek (default: the current one)")
    p.add_argument("--entry-id", type=int, help="your FPL entry (team) id")
    p.add_argument("--squad", help="15 player ids or names, comma separated")
    p.add_argument("--limit", type=int, default=10, help="rows to print (default 10)")
    p.add_argument("--explain", type=int, default=3, metavar="N",
                   help="print the full rationale for the top N options (default 3)")
    p.set_defaults(func=cmd_captain)

    p = add("chips", "chip recommendations and the GW19 expiry countdown",
            "Where the doubles and blanks are, which chip to play when, and how "
            "long the first set has left.")
    horizon_args(p)
    p.add_argument("--entry-id", type=int, help="your FPL entry (team) id")
    p.add_argument("--squad", help="15 player ids or names, comma separated")
    p.set_defaults(func=cmd_chips)

    p = add("backtest", "score the model against a finished season",
            "Walk a season gameweek by gameweek using only pre-deadline data and "
            "compare projections with what actually happened.")
    p.add_argument("--season", default="2025-26", help="season to backtest (default 2025-26)")
    p.add_argument("--gws", help="gameweeks to score, e.g. '1-38' (default: the whole season)")
    p.add_argument("--no-save", action="store_true",
                   help="do not write reports/backtest_{season}.json")
    p.set_defaults(func=cmd_backtest)

    p = add("autorefresh", "refresh the data only if it has gone stale",
            "The unattended refresh. Decides whether the cached data needs "
            "refetching (older than 24h, or older than 2h when the next deadline "
            "is inside 12h), and if it does, refetches it, refits the model and "
            "rewrites the projection cache so the dashboard opens instantly. "
            "Cheap and idempotent when the answer is no, single-flighted through "
            "a lock file, and it backs off instead of retrying a failing API. "
            "Safe to call every hour from launchd. Never runs the backtest.")
    p.add_argument("--force", action="store_true",
                   help="refresh regardless of age, backoff or freshness")
    p.add_argument("--dry-run", action="store_true",
                   help="print the decision and exit without fetching anything")
    p.add_argument("--horizon", type=int,
                   help="gameweeks to project (default %d, matching the server)"
                        % Config().model.default_horizon)
    p.set_defaults(func=cmd_autorefresh)

    p = add("serve", "run the API and dashboard",
            "Start the FastAPI backend on port %d and serve gaffer/web/ at /."
            % DEFAULT_PORT)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="port (default %d)" % DEFAULT_PORT)
    p.add_argument("--horizon", type=int, help="gameweeks to project at startup")
    p.add_argument("--log-level", default="info",
                   choices=["critical", "error", "warning", "info", "debug"])
    p.set_defaults(func=cmd_serve)

    p = add("h2h", "win the head-to-head, not the week",
            "Score a head-to-head tie on P(beating your opponent) rather than on "
            "expected points. Those differ: as an underdog you want variance, as a "
            "favourite you want none, and an expected-points optimiser plays both "
            "the same way.")
    p.add_argument("--opponent", type=int, required=True, help="their FPL entry id")
    p.add_argument("--entry", type=int, help="your entry id (default: FPL_ENTRY_ID)")
    p.add_argument("--gw", type=int, help="gameweek (default: current)")
    p.add_argument("--fresh", action="store_true", help="recompute projections")
    p.set_defaults(func=cmd_h2h)

    p = add("export", "freeze the dashboard into static files",
            "Compute everything once and write the whole dashboard to a directory "
            "of flat JSON plus the web assets, ready to publish to any static host. "
            "The result opens instantly and costs nothing to serve; what it cannot "
            "do is re-optimise against constraints you change on the phone, because "
            "that needs a solver.")
    p.add_argument("--out", default="site", help="output directory (default: site)")
    p.add_argument("--horizon", type=int, default=6,
                   help="gameweeks to project (default 6)")
    p.add_argument("--no-players", action="store_true",
                   help="skip the 587 per-player breakdown files")
    p.set_defaults(func=cmd_export)

    p = add("verify", "check the scoring constants and the data",
            "Re-check every scoring constant against the live API, then audit the "
            "loaded data and the projections for holes.")
    horizon_args(p)
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    config = Config.load(args.config)
    ctx = Context(config=config, progress=Progress(enabled=not args.quiet))
    try:
        return int(args.func(args, ctx))
    except PeerUnavailable as exc:
        sys.stderr.write("\n%s is not available yet:\n  %s\n" % (args.command, exc))
        return EXIT_UNAVAILABLE
    except FPLNotFound as exc:
        sys.stderr.write("\nthe FPL API returned 404: %s\n" % exc)
        return EXIT_FAIL
    except FPLError as exc:
        sys.stderr.write("\nthe FPL API is not answering: %s\n" % exc)
        return EXIT_FAIL
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
