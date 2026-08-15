"""Rank strategy: effective ownership, captaincy, differentials, rank risk.

Maximising expected points is the correct objective for reaching the top 10k.
It is not the objective that produces rank 1. Rank is a *relative* quantity:
what moves you up is not the points you score, it is the points you score that
the field does not. Two managers who both own the same 150%-effective-ownership
striker and both watch him score a hat-trick end the gameweek in exactly the
same relative position, having gained nothing on each other and lost ground to
nobody. The same hat-trick, from a 4%-owned striker, moves one of them past
millions of teams.

Everything in this module is built on one identity. Write ``m_you(p)`` for the
number of copies of player ``p`` your team scores (0 benched or not owned, 1
starting, 2 captain, 3 triple captain) and ``m_field(p) = EO(p)/100`` for the
number of copies the average team scores. Then

    your score - field score = sum over p of (m_you(p) - m_field(p)) * points(p)

Every number this module reports is a term of that sum. It follows immediately
that:

* Expected *relative* points are maximised by maximising expected points --
  ownership does not enter the mean at all. Anyone who tells you a differential
  has higher EV is confusing the mean with the tail.
* What ownership changes is the *shape* of your rank distribution. Rank 1
  requires a large positive deviation from the field, and only players where
  ``m_you - m_field`` is large and positive can produce one.
* A player at EO above 100% is a player you lose ground on by *not* owning:
  his ``m_you - m_field`` is ``-EO/100``, a guaranteed negative term every time
  he returns. At EO above 200% even captaining him leaves you behind the field.

The captaincy half of effective ownership is not published by the FPL API, so it
is estimated here. The estimate is a conditional logit over projected points,
weighted by ownership, calibrated so that the single most-captained player takes
a stated share of all captaincies. That share is the assumption, it is a
parameter, and it is printed with every report rather than buried.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gaffer.core import scoring
from gaffer.core.config import Config
from gaffer.core.types import CaptainOption, ProjectionSet
from gaffer.model.xp import DIST_MIN, norm_ppf

log = logging.getLogger(__name__)

# A "haul" is the 10-point gameweek: the return that actually moves rank. Kept
# in step with xp.HAUL_POINTS but re-declared so callers can override it.
HAUL_THRESHOLD = 10

# --- Assumptions behind the captaincy estimate ------------------------------
# The share of all captaincies taken by the single most-captained player. In a
# normal gameweek the consensus captain takes somewhere between 45% and 70% of
# active captaincies; 55% is the middle of that and is the one free parameter of
# the model. Override it per gameweek when you have real captaincy data (the
# various public "top 10k" scrapes publish it) rather than adjusting anything
# else in here.
DEFAULT_TOP_CAPTAIN_SHARE = 0.55
# Every entry captains exactly one player, so captaincy percentages sum to 100.
DEFAULT_CAPTAINCY_POOL = 100.0
# Below this starting-ownership a player attracts no measurable captaincy.
DEFAULT_CAPTAIN_OWNERSHIP_FLOOR = 3.0

# Squad shape, used by the start-share model. 2 keepers of whom 1 starts, and 13
# outfielders of whom 10 start.
_N_GKP = scoring.SQUAD_SELECT[scoring.GKP]
_N_OUTFIELD = scoring.SQUAD_SIZE - _N_GKP
_N_OUTFIELD_START = scoring.SQUAD_PLAY - 1
_N_OUTFIELD_BENCH = _N_OUTFIELD - _N_OUTFIELD_START

# Reporting thresholds for rank_risk_report.
TEMPLATE_EO = 40.0       # "highly owned": the field is heavily exposed to him
DIFFERENTIAL_EO = 15.0   # "a differential": the field is barely exposed


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _squad_ids(squad: Any) -> List[int]:
    """Accept a SquadState, a GWDecision-like object, or a bare list of ids."""
    if squad is None:
        return []
    if isinstance(squad, (list, tuple, set)):
        return [int(p) for p in squad]
    if hasattr(squad, "player_ids"):
        return [int(p) for p in squad.player_ids()]
    if hasattr(squad, "squad"):
        return [int(p) for p in squad.squad]
    raise TypeError("cannot read player ids from %r" % type(squad))


def _binom_at_most(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    p = _clip01(p)
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, min(k, n) + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p)
        )
    return _clip01(total)


def _weighted_cdf(values: Dict[int, float], weights: Dict[int, float]) -> Dict[int, float]:
    """Weighted P(X < x) + half the tie mass, evaluated at each key's own value.

    The half-tie term is the standard continuity correction: two players on
    identical projected points are equally likely to be the one benched.
    """
    ids = sorted(values, key=lambda i: (values[i], i))
    total = sum(max(weights.get(i, 0.0), 0.0) for i in ids)
    out: Dict[int, float] = {}
    if total <= 0:
        return {i: 0.5 for i in ids}
    below = 0.0
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and values[ids[j + 1]] == values[ids[i]]:
            j += 1
        tie = sum(max(weights.get(ids[m], 0.0), 0.0) for m in range(i, j + 1))
        for m in range(i, j + 1):
            out[ids[m]] = (below + 0.5 * tie) / total
        below += tie
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# Haul profiles: P(>=10) and E[points | >=10]
# ---------------------------------------------------------------------------


@dataclass
class HaulProfile:
    """What happens when a player has the kind of gameweek that moves rank."""

    player_id: int
    gw: int
    xp: float
    sd: float
    p_haul: float
    e_points_if_haul: float
    source: str = "normal-tail"


def haul_profiles(
    projections: ProjectionSet,
    gw: int,
    engine: Optional[Any] = None,
    haul: Optional[Dict[Tuple[int, int], float]] = None,
    threshold: int = HAUL_THRESHOLD,
) -> Dict[int, HaulProfile]:
    """``{player_id: HaulProfile}`` for one gameweek.

    With an ``XPEngine`` the numbers are read straight off the simulated points
    distribution. Without one (the API and dashboard read cached projections,
    which carry xp, sd and the haul probability but not the full distribution)
    the conditional mean comes from a normal tail *pinned to the true exceedance
    probability*: solve ``z`` from the known P(haul) rather than from the normal
    itself, then take ``mu + sd * phi(z) / P(haul)``. That keeps the one number
    we know exactly, exact.
    """
    gw = int(gw)
    out: Dict[int, HaulProfile] = {}
    for pid, per_gw in projections.projections.items():
        gwp = per_gw.get(gw)
        if gwp is None:
            continue
        xp, sd = float(gwp.xp), float(gwp.sd)
        dist = None
        if engine is not None:
            dist = engine.gw_distribution(int(pid), gw)
        if dist is not None:
            lo = max(0, int(threshold) - DIST_MIN)
            if lo < len(dist):
                p = float(dist[lo:].sum())
                mass = 0.0
                for idx in range(lo, len(dist)):
                    mass += (idx + DIST_MIN) * float(dist[idx])
                e_haul = (mass / p) if p > 1e-12 else float(threshold)
            else:
                p, e_haul = 0.0, float(threshold)
            out[int(pid)] = HaulProfile(int(pid), gw, xp, sd, _clip01(p), e_haul, "distribution")
            continue

        p = None
        if haul is not None:
            p = haul.get((int(pid), gw))
        if p is None:
            p = 0.0 if sd <= 1e-9 else _clip01(1.0 - _normal_cdf((threshold - xp) / sd))
        p = _clip01(float(p))
        if p <= 1e-9 or sd <= 1e-9:
            e_haul = float(threshold)
            p = max(p, 0.0)
        else:
            z = norm_ppf(1.0 - p)
            e_haul = max(float(threshold), xp + sd * _norm_pdf(z) / p)
        out[int(pid)] = HaulProfile(int(pid), gw, xp, sd, p, e_haul, "normal-tail")
    return out


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Effective ownership
# ---------------------------------------------------------------------------


@dataclass
class OwnershipModel:
    """Estimated field exposure for one gameweek, in percentage points."""

    gw: int
    ownership: Dict[int, float] = field(default_factory=dict)          # squad ownership, from the API
    start_share: Dict[int, float] = field(default_factory=dict)        # P(an owner starts him)
    starting_ownership: Dict[int, float] = field(default_factory=dict)  # ownership * start_share
    captaincy: Dict[int, float] = field(default_factory=dict)          # % of entries captaining him
    effective: Dict[int, float] = field(default_factory=dict)          # EO = starting + captaincy
    beta: float = 0.0
    captaincy_pool: float = DEFAULT_CAPTAINCY_POOL
    top_captain_share: float = DEFAULT_TOP_CAPTAIN_SHARE
    template_captain: Optional[int] = None
    total_ownership: float = 0.0
    total_effective: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def eo(self, player_id: int) -> float:
        return self.effective.get(int(player_id), 0.0)

    def field_multiplier(self, player_id: int) -> float:
        """Copies of this player the average team scores. EO expressed in copies."""
        return self.eo(player_id) / 100.0

    def top(self, n: int = 20) -> List[Tuple[int, float]]:
        return sorted(self.effective.items(), key=lambda kv: -kv[1])[:n]


def _fit_start_shares(
    xp: Dict[int, float], ownership: Dict[int, float], positions: Dict[int, int]
) -> Dict[int, float]:
    """P(a manager who owns this player starts him this gameweek).

    The API's ``selected_by_percent`` counts squad ownership, bench included, but
    only the starting XI scores. A £4.0m fifth defender owned by 30% of the field
    is not 30% of the field's *exposure*; he is closer to 4%.

    Model: a manager's 15 are drawn from the ownership-weighted pool of players
    (the standard mean-field approximation), and he benches the worst 4 of them
    by projected points. So an outfielder is benched exactly when at most 2 of
    his owner's other 12 outfielders project below him, and the backup keeper is
    the one of the two with the lower projection. Both are then closed forms in
    the ownership-weighted CDF of projected points. There is nothing to tune: the
    construction puts 4 of every 15 on the bench by symmetry.
    """
    gkp = {p: v for p, v in xp.items() if positions.get(p) == scoring.GKP}
    out_ = {p: v for p, v in xp.items() if positions.get(p) not in (None, scoring.GKP)}
    shares: Dict[int, float] = {}

    cdf_g = _weighted_cdf(gkp, ownership)
    for pid, f in cdf_g.items():
        # Of two keepers, the higher projection starts.
        shares[pid] = _clip01(f)

    cdf_o = _weighted_cdf(out_, ownership)
    for pid, f in cdf_o.items():
        # Benched iff at most (bench slots - 1) of the other 12 project lower.
        p_bench = _binom_at_most(_N_OUTFIELD_BENCH - 1, _N_OUTFIELD - 1, f)
        shares[pid] = _clip01(1.0 - p_bench)
    return shares


def _calibrate_start_shares(
    shares: Dict[int, float], ownership: Dict[int, float], target: float
) -> Tuple[Dict[int, float], float]:
    """Scale the odds of starting so the ownership-weighted start rate hits target.

    The mean-field draw is only approximate (real squads respect position quotas
    and the budget), so the implied rate lands near 11/15 but not on it. A single
    multiplicative shift in odds space fixes the total without reordering anyone.
    """
    total_own = sum(max(ownership.get(p, 0.0), 0.0) for p in shares)
    if total_own <= 0:
        return dict(shares), 1.0

    def rate(k: float) -> float:
        acc = 0.0
        for pid, s in shares.items():
            w = max(ownership.get(pid, 0.0), 0.0)
            if w <= 0:
                continue
            s = _clip01(s)
            adj = (k * s) / (k * s + (1.0 - s)) if 0.0 < s < 1.0 else s
            acc += w * adj
        return acc / total_own

    lo, hi = 1e-4, 1e4
    if rate(lo) > target:
        return {p: _clip01(s) for p, s in shares.items()}, lo
    if rate(hi) < target:
        return {p: _clip01(s) for p, s in shares.items()}, hi
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if rate(mid) < target:
            lo = mid
        else:
            hi = mid
    k = math.sqrt(lo * hi)
    out: Dict[int, float] = {}
    for pid, s in shares.items():
        s = _clip01(s)
        out[pid] = s if s <= 0.0 or s >= 1.0 else (k * s) / (k * s + (1.0 - s))
    return out, k


def _fit_captaincy(
    xp: Dict[int, float],
    starting_ownership: Dict[int, float],
    pool: float,
    top_share: float,
    own_floor: float,
) -> Tuple[Dict[int, float], float, Optional[int]]:
    """Distribute the captaincy pool over the candidates.

    Conditional logit: you can only captain a player you own, so the weight is
    ``ownership * exp(beta * xp)``. ``beta`` controls how sharply captaincy
    concentrates on the highest projected scorer and is not guessed -- it is
    solved so that the leading candidate takes ``top_share`` of all captaincies,
    which is a statement about the world that can be checked against published
    captaincy data.
    """
    cands = {
        p: v for p, v in xp.items()
        if starting_ownership.get(p, 0.0) >= own_floor and v > 0.0
    }
    if not cands:
        return {p: 0.0 for p in xp}, 0.0, None
    template = max(sorted(cands), key=lambda p: cands[p])

    def shares(beta: float) -> Dict[int, float]:
        top = max(cands.values())
        weights = {
            p: starting_ownership.get(p, 0.0) * math.exp(beta * (cands[p] - top))
            for p in cands
        }
        total = sum(weights.values())
        if total <= 0:
            return {p: 0.0 for p in cands}
        return {p: pool * w / total for p, w in weights.items()}

    # Share of the leading candidate is strictly increasing in beta (it has the
    # largest exponent in the set), so bisection is exact.
    target = pool * top_share
    lo, hi = 0.0, 40.0
    if shares(lo).get(template, 0.0) >= target:
        beta = 0.0
    elif shares(hi).get(template, 0.0) <= target:
        beta = hi
    else:
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if shares(mid).get(template, 0.0) < target:
                lo = mid
            else:
                hi = mid
        beta = 0.5 * (lo + hi)

    out = {p: 0.0 for p in xp}
    out.update(shares(beta))
    return out, beta, template


def estimate_ownership(
    state: Any,
    projections: ProjectionSet,
    gw: int,
    captaincy_pool: float = DEFAULT_CAPTAINCY_POOL,
    top_captain_share: float = DEFAULT_TOP_CAPTAIN_SHARE,
    captain_ownership_floor: float = DEFAULT_CAPTAIN_OWNERSHIP_FLOOR,
    captaincy_overrides: Optional[Dict[int, float]] = None,
    ownership_overrides: Optional[Dict[int, float]] = None,
) -> OwnershipModel:
    """Full effective-ownership model for one gameweek.

    ``captaincy_overrides`` takes real captaincy percentages where you have them
    (from a top-10k scrape, say); the remaining pool is redistributed over
    everyone else, so a partial override is fine.
    """
    gw = int(gw)
    xp: Dict[int, float] = {}
    ownership: Dict[int, float] = {}
    positions: Dict[int, int] = {}
    for pid, player in state.players.items():
        pid = int(pid)
        gwp = projections.projections.get(pid, {}).get(gw)
        xp[pid] = float(gwp.xp) if gwp is not None else 0.0
        own = float(player.selected_by_percent)
        if ownership_overrides and pid in ownership_overrides:
            own = float(ownership_overrides[pid])
        ownership[pid] = own
        positions[pid] = int(player.position)

    warnings: List[str] = []
    total_own = sum(ownership.values())
    # Every entry picks 15, so squad ownership must sum to ~1500 points of
    # percentage. A big miss means the ownership field is not what we think.
    if not (1200.0 <= total_own <= 1800.0):
        warnings.append(
            "sum of selected_by_percent is %.0f, expected ~%d (15 picks per entry); "
            "effective ownership levels will be scaled wrong"
            % (total_own, 100 * scoring.SQUAD_SIZE)
        )

    shares = _fit_start_shares(xp, ownership, positions)
    target_rate = float(scoring.SQUAD_PLAY) / float(scoring.SQUAD_SIZE)
    shares, _k = _calibrate_start_shares(shares, ownership, target_rate)
    starting = {p: ownership[p] * shares.get(p, 0.0) for p in ownership}

    captaincy, beta, template = _fit_captaincy(
        xp, starting, captaincy_pool, top_captain_share, captain_ownership_floor
    )
    if captaincy_overrides:
        fixed = {int(p): max(0.0, float(v)) for p, v in captaincy_overrides.items()}
        used = sum(fixed.values())
        left = max(0.0, captaincy_pool - used)
        rest = {p: v for p, v in captaincy.items() if p not in fixed}
        rest_total = sum(rest.values())
        captaincy = dict(fixed)
        for p, v in rest.items():
            captaincy[p] = (left * v / rest_total) if rest_total > 0 else 0.0
        for p in xp:
            captaincy.setdefault(p, 0.0)
        if used > captaincy_pool:
            warnings.append(
                "captaincy overrides sum to %.1f%%, above the %.1f%% pool; "
                "the remaining players were given zero" % (used, captaincy_pool)
            )
        template = max(sorted(captaincy), key=lambda p: captaincy[p]) if captaincy else None

    effective = {p: starting.get(p, 0.0) + captaincy.get(p, 0.0) for p in ownership}

    assumptions = [
        "squad ownership is the API's selected_by_percent, taken as given",
        "start share: a manager benches the 4 lowest-projected of his 15, with the "
        "squad drawn ownership-weighted; closed form, no free parameter, then "
        "shifted in odds space so the weighted start rate is exactly 11/15",
        "captaincy: conditional logit, weight = starting ownership * exp(%.2f * xP), "
        "calibrated so the leading candidate takes %.0f%% of a %.0f%% pool"
        % (beta, 100.0 * top_captain_share, captaincy_pool),
        "captaincy below %.1f%% starting ownership is treated as zero" % captain_ownership_floor,
        "chip usage (triple captain, bench boost) is not modelled, so EO is a "
        "slight under-estimate in chip-heavy gameweeks",
    ]
    model = OwnershipModel(
        gw=gw,
        ownership=ownership,
        start_share=shares,
        starting_ownership=starting,
        captaincy=captaincy,
        effective=effective,
        beta=beta,
        captaincy_pool=captaincy_pool,
        top_captain_share=top_captain_share,
        template_captain=template,
        total_ownership=total_own,
        total_effective=sum(effective.values()),
        assumptions=assumptions,
        warnings=warnings,
    )
    for w in warnings:
        log.warning("ownership model GW%d: %s", gw, w)
    return model


def effective_ownership(
    state: Any, projections: ProjectionSet, gw: int, **kwargs: Any
) -> Dict[int, float]:
    """``{player_id: EO%}`` where EO = starting ownership% + captaincy%.

    See ``estimate_ownership`` for the model and its assumptions; this is the
    SPEC-shaped wrapper that hands back just the numbers.
    """
    return estimate_ownership(state, projections, gw, **kwargs).effective


def field_expected_score(
    ownership: OwnershipModel, projections: ProjectionSet, gw: int
) -> float:
    """Expected gameweek score of the average team: sum of EO/100 * xP."""
    total = 0.0
    for pid, eo in ownership.effective.items():
        gwp = projections.projections.get(pid, {}).get(int(gw))
        if gwp is not None:
            total += (eo / 100.0) * gwp.xp
    return total


def field_expected_captain_xp(
    ownership: OwnershipModel, projections: ProjectionSet, gw: int
) -> float:
    """Expected points the field gets from its captain armband alone."""
    total = 0.0
    weight = 0.0
    for pid, share in ownership.captaincy.items():
        if share <= 0:
            continue
        gwp = projections.projections.get(pid, {}).get(int(gw))
        if gwp is None:
            continue
        total += share * gwp.xp
        weight += share
    return total / weight if weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Lineup helper
# ---------------------------------------------------------------------------


def _gw_p_appear(projections: ProjectionSet, pid: int, gw: int) -> float:
    gwp = projections.projections.get(int(pid), {}).get(int(gw))
    if gwp is None or not gwp.fixtures:
        return 0.0
    miss = 1.0
    for f in gwp.fixtures:
        miss *= (1.0 - _clip01(f.p_appear))
    return _clip01(1.0 - miss)


def best_xi(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    chip: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """``(starting XI, bench in autosub order)`` for one gameweek.

    Delegates to ``gaffer.optimize.squad.pick_lineup`` when that module is
    importable so there is one authority on lineup legality. The local path is
    an exact enumeration of the eleven legal formations, not an approximation:
    with the shape fixed, the best XI is the top-N by projected points in each
    position, so eleven small sorts cover the whole space.
    """
    ids = [int(p) for p in squad_ids]
    try:
        from gaffer.optimize.squad import pick_lineup  # noqa: WPS433 (parallel module)
    except ImportError:
        pick_lineup = None
    if pick_lineup is not None:
        lineup, bench, _c, _v = pick_lineup(ids, projections, int(gw), state, chip=chip)
        return [int(p) for p in lineup], [int(p) for p in bench]

    by_pos: Dict[int, List[int]] = {p: [] for p in scoring.POSITIONS}
    for pid in ids:
        by_pos[state.players[pid].position].append(pid)
    xp = {pid: projections.xp(pid, int(gw)) for pid in ids}
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: (-xp[p], p))

    # Bench Boost does not change the selection: FPL still requires a legal XI
    # and a bench, the bench simply also scores. Only the valuation changes, and
    # that lives in chips.evaluate_bench_boost.
    best: Optional[Tuple[float, List[int]]] = None
    for n_def in range(scoring.SQUAD_MIN_PLAY[scoring.DEF], scoring.SQUAD_MAX_PLAY[scoring.DEF] + 1):
        for n_mid in range(scoring.SQUAD_MIN_PLAY[scoring.MID], scoring.SQUAD_MAX_PLAY[scoring.MID] + 1):
            n_fwd = scoring.SQUAD_PLAY - 1 - n_def - n_mid
            if not (scoring.SQUAD_MIN_PLAY[scoring.FWD] <= n_fwd <= scoring.SQUAD_MAX_PLAY[scoring.FWD]):
                continue
            need = {scoring.GKP: 1, scoring.DEF: n_def, scoring.MID: n_mid, scoring.FWD: n_fwd}
            if any(len(by_pos[p]) < need[p] for p in need):
                continue
            xi: List[int] = []
            for pos in scoring.POSITIONS:
                xi.extend(by_pos[pos][: need[pos]])
            total = sum(xp[p] for p in xi)
            if best is None or total > best[0]:
                best = (total, xi)
    if best is None:
        raise ValueError("squad of %d cannot field a legal XI" % len(ids))

    xi = best[1]
    on = set(xi)
    bench = [p for p in ids if p not in on]
    # The reserve keeper only ever replaces the keeper, so he holds bench slot 0
    # regardless of projection; the outfielders behind him rank by the points
    # they would actually deliver if called on.
    gks = [p for p in bench if state.players[p].position == scoring.GKP]
    others = [p for p in bench if state.players[p].position != scoring.GKP]
    others.sort(key=lambda p: (-(xp[p] * _gw_p_appear(projections, p, gw)), p))
    return xi, gks + others


# ---------------------------------------------------------------------------
# Captaincy
# ---------------------------------------------------------------------------


@dataclass
class CaptainAnalysis:
    """Everything behind one ``CaptainOption``, for the dashboard and the API."""

    player_id: int
    name: str
    team: str
    position: str
    xp: float
    sd: float
    p_haul: float
    e_points_if_haul: float
    effective_ownership: float
    captaincy_share: float
    starting_ownership: float
    # Expected points from the armband, minus what the field's armband returns.
    ev_vs_field: float
    # Copies of this player you hold as captain, net of the field's EO. Negative
    # means his haul helps the field more than it helps you.
    net_copies: float
    # Rank-relative points if he hauls: net_copies * E[points | haul].
    swing_if_hauls: float
    # Expected rank-relative points across every haul event in the gameweek,
    # under this captaincy choice.
    differential_ev: float
    # Standard deviation of (your captain's return - the field's captain return).
    # This is where effective ownership actually bites on a captaincy decision.
    relative_sd: float
    # relative_sd against the highest-xP option: the extra swing you buy.
    upside_edge: float
    # The blend the configured differential_weight actually recommends.
    rank_ev: float
    is_template_captain: bool
    rationale: str


def _field_captain_candidates(
    ownership: OwnershipModel, projections: ProjectionSet, gw: int, limit: int = 15
) -> List[int]:
    ranked = sorted(ownership.captaincy.items(), key=lambda kv: -kv[1])
    return [int(p) for p, share in ranked[:limit] if share > 0.0]


def captain_analysis(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    config: Optional[Config] = None,
    ownership: Optional[OwnershipModel] = None,
    engine: Optional[Any] = None,
    haul: Optional[Dict[Tuple[int, int], float]] = None,
    profiles: Optional[Dict[int, HaulProfile]] = None,
    differential_weight: Optional[float] = None,
    threshold: int = HAUL_THRESHOLD,
    triple_captain: bool = False,
) -> List[CaptainAnalysis]:
    """Rank every squad member as a captain, on points and on rank."""
    config = config or Config.load()
    gw = int(gw)
    ids = [int(p) for p in squad_ids]
    w = config.optimizer.differential_weight if differential_weight is None else float(differential_weight)
    w = _clip01(w)
    mult = scoring.TRIPLE_CAPTAIN_MULTIPLIER if triple_captain else scoring.CAPTAIN_MULTIPLIER

    if ownership is None:
        ownership = estimate_ownership(state, projections, gw)
    if profiles is None:
        profiles = haul_profiles(projections, gw, engine=engine, haul=haul, threshold=threshold)

    xi, _bench = best_xi(ids, projections, gw, state)
    starters = set(xi)
    field_capt = field_expected_captain_xp(ownership, projections, gw)
    candidates = set(_field_captain_candidates(ownership, projections, gw))
    template = ownership.template_captain

    def profile(pid: int) -> HaulProfile:
        return profiles.get(pid, HaulProfile(pid, gw, 0.0, 0.0, 0.0, float(threshold)))

    def multiplier(pid: int, captain: int) -> float:
        if pid == captain:
            return float(mult)
        return 1.0 if pid in starters else 0.0

    def variance(pid: int) -> float:
        gwp = projections.projections.get(pid, {}).get(gw)
        return (float(gwp.sd) ** 2) if gwp is not None else 0.0

    # The field's armband is a mixture: fraction c_j of teams captain player j.
    # Your relative captain return is R_i = mult * P_i - sum_j c_j * P_j, so the
    # candidate's own share appears as (mult - c_i) and everyone else's as c_j.
    shares = {p: ownership.captaincy.get(p, 0.0) / 100.0 for p in candidates}
    field_var = sum((c ** 2) * variance(p) for p, c in shares.items())

    out: List[CaptainAnalysis] = []
    for pid in ids:
        gwp = projections.projections.get(pid, {}).get(gw)
        xp = float(gwp.xp) if gwp is not None else 0.0
        sd = float(gwp.sd) if gwp is not None else 0.0
        prof = profile(pid)
        eo = ownership.eo(pid)
        net_copies = float(mult) - eo / 100.0
        swing = net_copies * prof.e_points_if_haul

        # Expected rank-relative points restricted to haul events. Rank is moved
        # by hauls, not by the difference between a 3 and a 4, so this is the
        # part of the relative-score identity that actually matters, evaluated
        # over every player the field is meaningfully exposed to.
        diff_ev = 0.0
        for other in candidates | {pid}:
            po = profile(other)
            diff_ev += po.p_haul * (multiplier(other, pid) - ownership.eo(other) / 100.0) \
                * po.e_points_if_haul

        # Relative variance. Captaining the template captain correlates your
        # gameweek with the field's and damps the swing; captaining someone the
        # field has not got leaves both his variance and the template's working
        # against each other, which is the only way a captaincy pick moves you a
        # long way in either direction.
        c_i = shares.get(pid, 0.0)
        rel_var = field_var - (c_i ** 2) * variance(pid) + ((float(mult) - c_i) ** 2) * variance(pid)
        relative_sd = math.sqrt(max(rel_var, 0.0))

        ev_vs_field = xp - field_capt
        out.append(
            CaptainAnalysis(
                player_id=pid,
                name=state.players[pid].web_name,
                team=state.short_name(state.players[pid].team_id),
                position=scoring.POS_NAME[state.players[pid].position],
                xp=xp,
                sd=sd,
                p_haul=prof.p_haul,
                e_points_if_haul=prof.e_points_if_haul,
                effective_ownership=eo,
                captaincy_share=ownership.captaincy.get(pid, 0.0),
                starting_ownership=ownership.starting_ownership.get(pid, 0.0),
                ev_vs_field=ev_vs_field,
                net_copies=net_copies,
                swing_if_hauls=swing,
                differential_ev=diff_ev,
                relative_sd=relative_sd,
                upside_edge=0.0,
                rank_ev=0.0,
                is_template_captain=(pid == template),
                rationale="",
            )
        )

    out.sort(key=lambda a: (-a.xp, a.player_id))
    best_points = out[0] if out else None
    # Measure the extra swing against the option you would take on points alone,
    # so differential_weight trades in comparable units: one point of expected
    # points against one point of extra relative standard deviation.
    reference_sd = best_points.relative_sd if best_points is not None else 0.0
    for a in out:
        a.upside_edge = a.relative_sd - reference_sd
        a.rank_ev = a.ev_vs_field + w * a.upside_edge
    for a in out:
        a.rationale = _captain_rationale(a, best_points, field_capt, w, threshold, mult)
    return out


def _captain_rationale(
    a: CaptainAnalysis,
    best: Optional[CaptainAnalysis],
    field_capt: float,
    w: float,
    threshold: int,
    mult: float,
) -> str:
    bits: List[str] = []
    bits.append(
        "%.2f xP (sd %.2f), P(>=%d) %.0f%%; the field's armband returns %.2f, so the "
        "armband alone is worth %+.2f against the average team"
        % (a.xp, a.sd, threshold, 100 * a.p_haul, field_capt, a.ev_vs_field)
    )
    if a.effective_ownership >= 100.0 * mult:
        bits.append(
            "EO %.0f%% is at or above the %.0f copies you would hold as captain: he is "
            "priced into the field at your own exposure, so his haul gains you nothing and "
            "his blank costs you nothing -- a pure hedge, not an attack"
            % (a.effective_ownership, mult)
        )
    elif a.effective_ownership >= 100.0:
        bits.append(
            "EO %.0f%% means the field already scores %.2f copies of him: captaining him "
            "nets only %+.2f copies (%+.1f rank-relative points if he hauls), while NOT "
            "owning him would cost %.1f -- he is close to compulsory, and captaining him "
            "is defence"
            % (a.effective_ownership, a.effective_ownership / 100.0, a.net_copies,
               a.swing_if_hauls, a.effective_ownership / 100.0 * a.e_points_if_haul)
        )
    elif a.effective_ownership >= 50.0:
        bits.append(
            "EO %.0f%% is template territory rather than a differential: you net %+.2f "
            "copies, %+.1f rank-relative points on a haul, and roughly half the field "
            "moves with you" % (a.effective_ownership, a.net_copies, a.swing_if_hauls)
        )
    else:
        bits.append(
            "EO %.0f%% is a genuine differential: captaining him nets %+.2f copies against "
            "the field, %+.1f rank-relative points if he hauls"
            % (a.effective_ownership, a.net_copies, a.swing_if_hauls)
        )
    bits.append(
        "captaincy share %.1f%% of the field, so your week is %s the template's: relative "
        "sd %.2f (the spread of your armband against the field's)"
        % (a.captaincy_share,
           "tightly coupled to" if a.captaincy_share >= 25.0 else "largely decoupled from",
           a.relative_sd)
    )
    if best is not None and a.player_id != best.player_id:
        gap = best.xp - a.xp
        bits.append(
            "against the top-xP option (%s) this costs %.2f expected points and buys "
            "%+.2f of extra relative standard deviation -- worth taking only if you need "
            "the variance to climb, never if you are protecting a rank"
            % (best.name, gap, a.upside_edge)
        )
    bits.append(
        "note the honest caveat: among players you already own, effective ownership does "
        "not change the EXPECTED value of a captaincy pick at all (the field's armband is "
        "the same whoever you choose) -- it changes how correlated your gameweek is with "
        "the field's, which is what actually decides whether a good week is a green arrow"
    )
    bits.append(
        "differential_weight %.2f -> rank_ev %+.2f = expected edge %+.2f plus %.2f x "
        "upside edge %+.2f" % (w, a.rank_ev, a.ev_vs_field, w, a.upside_edge)
    )
    return "; ".join(bits)


def captain_options(
    squad_ids: Sequence[int],
    projections: ProjectionSet,
    gw: int,
    state: Any,
    config: Optional[Config] = None,
    **kwargs: Any
) -> List[CaptainOption]:
    """SPEC-shaped captain ranking, ordered by the configured objective.

    ``CaptainOption.ev_vs_field`` is the armband's expected points minus what the
    field's armband returns -- the honest "am I ahead of the average team this
    week" number, and the one to lead with.

    The ordering respects ``config.optimizer.differential_weight``. At 0 it is a
    pure expected-points ranking. Above 0 it trades expected points for *relative
    standard deviation*: ``rank_ev = ev_vs_field + weight * upside_edge``, where
    upside_edge is how much wider the spread of (your armband minus the field's
    armband) becomes compared with the top-xP pick. One point of expected points
    against one point of extra swing, which is a trade a manager can actually
    reason about.

    The subtlety worth stating plainly, because most rank advice gets it wrong:
    among players you already own, effective ownership does not change the
    expected value of a captaincy choice. The field's armband return is a
    constant, so swapping your captain from A to B moves your relative score by
    exactly ``xP(A) - xP(B)`` whatever their ownership. What EO changes is
    correlation. Captain the 55%-captained template and your gameweek rises and
    falls with the field's; captain someone the field has not got and both his
    variance and the template's are free to move against each other. That is why
    a differential captain is a rank *risk* instrument and not a free lunch, and
    it is why ``differential_weight`` moves variance rather than mean.
    """
    detail = captain_analysis(squad_ids, projections, gw, state, config, **kwargs)
    detail.sort(key=lambda a: (-a.rank_ev, -a.xp, a.player_id))
    return [
        CaptainOption(
            player_id=a.player_id,
            xp=a.xp,
            sd=a.sd,
            p_haul=a.p_haul,
            effective_ownership=a.effective_ownership,
            ev_vs_field=a.ev_vs_field,
            rationale=a.rationale,
        )
        for a in detail
    ]


def format_captain_options(detail: Sequence[CaptainAnalysis], limit: int = 15) -> str:
    header = ("%-3s %-16s %-4s %-4s %6s %5s %6s %7s %7s %7s %8s %6s %7s %8s"
              % ("#", "player", "team", "pos", "xP", "sd", "P(10+)", "EO%", "capt%",
                 "netCopy", "swingH", "relSD", "upside", "rank_ev"))
    lines = [header, "-" * len(header)]
    for i, a in enumerate(detail[:limit], start=1):
        lines.append(
            "%-3d %-16s %-4s %-4s %6.2f %5.2f %5.0f%% %6.1f%% %6.1f%% %+7.2f %+8.1f "
            "%6.2f %+7.2f %+8.2f%s"
            % (i, a.name[:16], a.team, a.position, a.xp, a.sd, 100 * a.p_haul,
               a.effective_ownership, a.captaincy_share, a.net_copies,
               a.swing_if_hauls, a.relative_sd, a.upside_edge, a.rank_ev,
               "  <- template" if a.is_template_captain else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Differentials
# ---------------------------------------------------------------------------


@dataclass
class DifferentialScore:
    player_id: int
    name: str
    team: str
    position: str
    price: float
    xp: float
    effective_ownership: float
    p_haul: float
    e_points_if_haul: float
    # Rank-relative points you gain over the field when he hauls, above what his
    # projection already promised. Zero for a fully-owned player, negative for a
    # player at EO above 100% (his haul lifts the field more than it lifts you).
    tail_edge: float
    score: float
    weight: float
    rationale: str


def differential_table(
    state: Any,
    projections: ProjectionSet,
    gw: int,
    config: Optional[Config] = None,
    ownership: Optional[OwnershipModel] = None,
    engine: Optional[Any] = None,
    haul: Optional[Dict[Tuple[int, int], float]] = None,
    profiles: Optional[Dict[int, HaulProfile]] = None,
    differential_weight: Optional[float] = None,
    player_ids: Optional[Iterable[int]] = None,
    threshold: int = HAUL_THRESHOLD,
) -> List[DifferentialScore]:
    config = config or Config.load()
    gw = int(gw)
    w = config.optimizer.differential_weight if differential_weight is None else float(differential_weight)
    w = _clip01(w)
    if ownership is None:
        ownership = estimate_ownership(state, projections, gw)
    if profiles is None:
        profiles = haul_profiles(projections, gw, engine=engine, haul=haul, threshold=threshold)
    ids = [int(p) for p in (player_ids if player_ids is not None else state.players.keys())]

    rows: List[DifferentialScore] = []
    for pid in ids:
        player = state.players.get(pid)
        if player is None:
            continue
        gwp = projections.projections.get(pid, {}).get(gw)
        xp = float(gwp.xp) if gwp is not None else 0.0
        prof = profiles.get(pid, HaulProfile(pid, gw, xp, 0.0, 0.0, float(threshold)))
        eo = ownership.eo(pid)
        # The rank-relevant part of a haul: the points the field does NOT also
        # bank. (1 - EO/100) is your net exposure holding one copy; subtracting
        # the projection leaves only the surprise, since the projection is
        # already priced into everyone's expectations.
        tail = prof.p_haul * (1.0 - eo / 100.0) * max(0.0, prof.e_points_if_haul - xp)
        score = (1.0 - w) * xp + w * tail
        rows.append(
            DifferentialScore(
                player_id=pid,
                name=player.web_name,
                team=state.short_name(player.team_id),
                position=scoring.POS_NAME[player.position],
                price=player.price,
                xp=xp,
                effective_ownership=eo,
                p_haul=prof.p_haul,
                e_points_if_haul=prof.e_points_if_haul,
                tail_edge=tail,
                score=score,
                weight=w,
                rationale=_differential_rationale(player.web_name, xp, eo, prof, tail, w, threshold),
            )
        )
    rows.sort(key=lambda r: (-r.score, r.player_id))
    return rows


def _differential_rationale(
    name: str, xp: float, eo: float, prof: HaulProfile, tail: float, w: float, threshold: int
) -> str:
    if w <= 0.0:
        mode = ("differential_weight 0: this is a pure expected-points ranking, ownership "
                "is ignored entirely")
    elif w >= 1.0:
        mode = ("differential_weight 1: this is a pure rank-chasing ranking -- expected "
                "points are ignored and only the points the field does not also bank count, "
                "so template premiums fall to the bottom and can score negative")
    else:
        mode = ("differential_weight %.2f: %.0f%% expected points, %.0f%% rank-relative "
                "upside" % (w, 100 * (1 - w), 100 * w))
    if eo >= 100.0:
        own = ("EO %.0f%% -- above 100%%, so owning one copy still leaves you %+.2f copies "
               "behind the field and his haul costs you ground; you own him to avoid "
               "falling, not to climb" % (eo, 1.0 - eo / 100.0))
    elif eo >= 50.0:
        own = ("EO %.0f%% is template territory: %+.2f copies clear of the field, so he "
               "protects a rank more than he wins one" % (eo, 1.0 - eo / 100.0))
    else:
        own = "EO %.0f%% leaves you %+.2f copies clear of the field" % (eo, 1.0 - eo / 100.0)
    return ("%.2f xP, P(>=%d) %.0f%% at %.1f points when it lands; %s. Tail edge %+.2f. %s"
            % (xp, threshold, 100 * prof.p_haul, prof.e_points_if_haul, own, tail, mode))


def differential_score(
    player_id: int,
    state: Any,
    projections: ProjectionSet,
    gw: int,
    config: Optional[Config] = None,
    **kwargs: Any
) -> float:
    """One player's rank-chasing score for a gameweek.

    ``config.optimizer.differential_weight`` is the mode switch:

    * **0.0 (default)** -- the score is exactly projected points. Ownership does
      not enter, which is correct if the target is a good rank rather than the
      top of the table, because ownership genuinely does not change expected
      points.
    * **between 0 and 1** -- a linear blend.
    * pass ``ownership=`` and ``profiles=`` when scoring more than one player:
      each bare call refits the whole ownership model over all 587 players.
    * **1.0** -- the score is the tail edge alone: ``P(haul) * (1 - EO/100) *
      (E[points | haul] - xP)``, the expected points from a haul that the field
      does *not* also collect. Template premiums score near zero and players
      above 100% EO score negative. Chasing rank 1 this is the right number;
      chasing a safe green arrow it is a good way to fall over.
    """
    rows = differential_table(
        state, projections, gw, config=config, player_ids=[int(player_id)], **kwargs
    )
    return rows[0].score if rows else 0.0


def format_differential_table(rows: Sequence[DifferentialScore], limit: int = 20) -> str:
    header = ("%-3s %-16s %-4s %-4s %6s %6s %7s %6s %7s %8s %7s"
              % ("#", "player", "team", "pos", "price", "xP", "EO%", "P(10+)",
                 "E[haul]", "tailEdge", "score"))
    lines = [header, "-" * len(header)]
    for i, r in enumerate(rows[:limit], start=1):
        lines.append(
            "%-3d %-16s %-4s %-4s %6.1f %6.2f %6.1f%% %5.0f%% %7.1f %+8.2f %7.2f"
            % (i, r.name[:16], r.team, r.position, r.price, r.xp, r.effective_ownership,
               100 * r.p_haul, r.e_points_if_haul, r.tail_edge, r.score)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rank risk
# ---------------------------------------------------------------------------


def rank_risk_report(
    squad: Any,
    state: Any,
    projections: ProjectionSet,
    gw: int,
    config: Optional[Config] = None,
    ownership: Optional[OwnershipModel] = None,
    engine: Optional[Any] = None,
    haul: Optional[Dict[Tuple[int, int], float]] = None,
    profiles: Optional[Dict[int, HaulProfile]] = None,
    captain_id: Optional[int] = None,
    template_eo: float = TEMPLATE_EO,
    differential_eo: float = DIFFERENTIAL_EO,
    threshold: int = HAUL_THRESHOLD,
    limit: int = 12,
) -> Dict[str, Any]:
    """Where this squad stands to gain and lose rank in one gameweek.

    Two lists and one number:

    * **exposure gaps** -- highly-owned players you do not own. Every one is a
      standing negative term in your relative score: when he returns, the field
      banks ``EO/100`` copies and you bank none.
    * **differentials held** -- players you own that the field mostly does not.
      These are the only positions that can produce a large positive deviation,
      which is the only thing that produces rank 1.
    * **net haul exposure** -- the sum of both sides, in rank-relative points.
      Positive means the gameweek's big swings are, on balance, working for you.
    """
    config = config or Config.load()
    gw = int(gw)
    ids = _squad_ids(squad)
    if ownership is None:
        ownership = estimate_ownership(state, projections, gw)
    if profiles is None:
        profiles = haul_profiles(projections, gw, engine=engine, haul=haul, threshold=threshold)

    xi, bench = best_xi(ids, projections, gw, state) if ids else ([], [])
    if captain_id is None and xi:
        detail = captain_analysis(
            ids, projections, gw, state, config, ownership=ownership, profiles=profiles,
            threshold=threshold,
        )
        detail.sort(key=lambda a: (-a.rank_ev, -a.xp, a.player_id))
        captain_id = detail[0].player_id if detail else None
    owned = set(ids)
    starters = set(xi)

    def multiplier(pid: int) -> float:
        if pid == captain_id:
            return float(scoring.CAPTAIN_MULTIPLIER)
        return 1.0 if pid in starters else 0.0

    def prof(pid: int) -> HaulProfile:
        return profiles.get(pid, HaulProfile(pid, gw, 0.0, 0.0, 0.0, float(threshold)))

    gaps: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []
    net_haul = 0.0
    field_points = 0.0
    my_points = 0.0

    for pid, eo in ownership.effective.items():
        player = state.players.get(pid)
        if player is None:
            continue
        gwp = projections.projections.get(pid, {}).get(gw)
        xp = float(gwp.xp) if gwp is not None else 0.0
        p = prof(pid)
        m = multiplier(pid) if pid in owned else 0.0
        net = m - eo / 100.0
        field_points += (eo / 100.0) * xp
        my_points += m * xp
        net_haul += p.p_haul * net * p.e_points_if_haul

        if pid not in owned and eo >= template_eo and p.p_haul > 0.0:
            gaps.append({
                "player_id": pid,
                "name": player.web_name,
                "team": state.short_name(player.team_id),
                "position": scoring.POS_NAME[player.position],
                "price": player.price,
                "xp": xp,
                "effective_ownership": eo,
                "captaincy_share": ownership.captaincy.get(pid, 0.0),
                "p_haul": p.p_haul,
                "e_points_if_haul": p.e_points_if_haul,
                # What his haul costs you, right now, in points against the field.
                "damage_if_hauls": -(eo / 100.0) * p.e_points_if_haul,
                "expected_damage": -p.p_haul * (eo / 100.0) * p.e_points_if_haul,
            })
        if pid in owned and eo <= differential_eo and m > 0.0:
            held.append({
                "player_id": pid,
                "name": player.web_name,
                "team": state.short_name(player.team_id),
                "position": scoring.POS_NAME[player.position],
                "price": player.price,
                "xp": xp,
                "effective_ownership": eo,
                "multiplier": m,
                "p_haul": p.p_haul,
                "e_points_if_haul": p.e_points_if_haul,
                "upside_if_hits": net * p.e_points_if_haul,
                "expected_upside": p.p_haul * net * p.e_points_if_haul,
            })

    gaps.sort(key=lambda r: r["expected_damage"])
    held.sort(key=lambda r: -r["expected_upside"])

    # Template coverage: of the field's exposure to its most-owned players, how
    # much of it do you match?
    top_eo = sorted(ownership.effective.items(), key=lambda kv: -kv[1])[:20]
    covered = sum(eo for pid, eo in top_eo if pid in owned)
    total_top = sum(eo for _pid, eo in top_eo)

    worst_case = sum(r["damage_if_hauls"] for r in gaps[:3])
    best_case = sum(r["upside_if_hits"] for r in held[:3])

    report: Dict[str, Any] = {
        "gw": gw,
        "squad_size": len(ids),
        "captain_id": captain_id,
        "captain": state.players[captain_id].web_name if captain_id in state.players else None,
        "xi": xi,
        "bench": bench,
        "my_expected_points": my_points,
        "field_expected_points": field_points,
        "expected_edge": my_points - field_points,
        "net_haul_exposure": net_haul,
        "template_coverage": (covered / total_top * 100.0) if total_top > 0 else 0.0,
        "exposure_gaps": gaps[:limit],
        "differentials_held": held[:limit],
        "worst_case_top3_gaps": worst_case,
        "best_case_top3_differentials": best_case,
        "ownership_assumptions": list(ownership.assumptions),
        "ownership_warnings": list(ownership.warnings),
        "beta": ownership.beta,
        "template_captain": ownership.template_captain,
        "template_captain_name": (
            state.players[ownership.template_captain].web_name
            if ownership.template_captain in state.players else None
        ),
        "thresholds": {
            "template_eo": template_eo,
            "differential_eo": differential_eo,
            "haul": threshold,
        },
    }
    report["summary"] = _rank_risk_summary(report)
    return report


def _rank_risk_summary(r: Dict[str, Any]) -> str:
    bits = []
    edge = r["expected_edge"]
    bits.append(
        "GW%d: %.1f expected points against a field average of %.1f, an edge of %+.1f"
        % (r["gw"], r["my_expected_points"], r["field_expected_points"], edge)
    )
    net = r["net_haul_exposure"]
    if net >= 0.5:
        bits.append(
            "net haul exposure %+.2f: the gameweek's big returns break your way on balance"
            % net
        )
    elif net <= -0.5:
        bits.append(
            "net haul exposure %+.2f: you are behind the field on the events that move "
            "rank, and a green arrow this week needs the template to misfire" % net
        )
    else:
        bits.append(
            "net haul exposure %+.2f: you are level with the field on the big swings, "
            "which is a fine place to protect a rank and a poor place to chase one" % net
        )
    bits.append("template coverage %.0f%% of the field's top-20 exposure" % r["template_coverage"])
    if r["exposure_gaps"]:
        g = r["exposure_gaps"][0]
        bits.append(
            "biggest single hole: %s (EO %.0f%%), %.1f points against you if he hauls"
            % (g["name"], g["effective_ownership"], -g["damage_if_hauls"])
        )
    if r["differentials_held"]:
        d = r["differentials_held"][0]
        bits.append(
            "best differential held: %s (EO %.0f%%), %+.1f points for you if he hits"
            % (d["name"], d["effective_ownership"], d["upside_if_hits"])
        )
    return ". ".join(bits) + "."


def format_rank_risk_report(report: Dict[str, Any], state: Any = None) -> str:
    gw = report["gw"]
    lines: List[str] = []
    title = "RANK RISK REPORT — GW%d" % gw
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("  captain            : %s" % (report.get("captain") or "n/a"))
    lines.append("  template captain   : %s (the armband the field is on)"
                 % (report.get("template_captain_name") or "n/a"))
    lines.append("  your expected pts  : %6.1f" % report["my_expected_points"])
    lines.append("  field average      : %6.1f" % report["field_expected_points"])
    lines.append("  expected edge      : %+6.1f  (positive = you out-score the average team)"
                 % report["expected_edge"])
    lines.append("  net haul exposure  : %+6.2f  (rank-relative points across every 10+ event)"
                 % report["net_haul_exposure"])
    lines.append("  template coverage  : %5.0f%%  (share of the field's top-20 exposure you hold)"
                 % report["template_coverage"])
    lines.append("")

    lines.append("  WHAT HURTS — highly-owned players you do NOT own")
    lines.append("  " + "-" * 88)
    head = ("  %-16s %-4s %-4s %6s %6s %7s %6s %9s %9s"
            % ("player", "team", "pos", "price", "xP", "EO%", "P(10+)", "if hauls", "expected"))
    lines.append(head)
    if not report["exposure_gaps"]:
        lines.append("  (none: you own every player the field is meaningfully exposed to)")
    for g in report["exposure_gaps"]:
        lines.append(
            "  %-16s %-4s %-4s %6.1f %6.2f %6.1f%% %5.0f%% %+9.1f %+9.2f"
            % (g["name"][:16], g["team"], g["position"], g["price"], g["xp"],
               g["effective_ownership"], 100 * g["p_haul"], g["damage_if_hauls"],
               g["expected_damage"])
        )
    lines.append("  worst case if the top three all haul: %+.1f points against the field"
                 % report["worst_case_top3_gaps"])
    lines.append("")

    lines.append("  WHAT WINS — differentials you hold")
    lines.append("  " + "-" * 88)
    head = ("  %-16s %-4s %-4s %6s %6s %7s %6s %9s %9s"
            % ("player", "team", "pos", "price", "xP", "EO%", "P(10+)", "if hits", "expected"))
    lines.append(head)
    if not report["differentials_held"]:
        lines.append("  (none: this is a template squad — it cannot lose much rank and it "
                     "cannot win any)")
    for d in report["differentials_held"]:
        lines.append(
            "  %-16s %-4s %-4s %6.1f %6.2f %6.1f%% %5.0f%% %+9.1f %+9.2f"
            % (d["name"][:16], d["team"], d["position"], d["price"], d["xp"],
               d["effective_ownership"], 100 * d["p_haul"], d["upside_if_hits"],
               d["expected_upside"])
        )
    lines.append("  best case if the top three all hit: %+.1f points on the field"
                 % report["best_case_top3_differentials"])
    lines.append("")
    lines.append("  VERDICT")
    lines.append("  " + "-" * 88)
    for chunk in report["summary"].split(". "):
        chunk = chunk.strip()
        if chunk:
            lines.append("  - %s" % (chunk if chunk.endswith(".") else chunk + "."))
    lines.append("")
    lines.append("  ownership model assumptions")
    for a in report["ownership_assumptions"]:
        lines.append("    * %s" % a)
    for w in report["ownership_warnings"]:
        lines.append("    ! %s" % w)
    return "\n".join(lines)


def format_ownership_table(
    model: OwnershipModel, state: Any, projections: ProjectionSet, limit: int = 20
) -> str:
    gw = model.gw
    header = ("%-3s %-18s %-4s %-4s %6s %6s %7s %7s %7s %7s"
              % ("#", "player", "team", "pos", "price", "xP", "own%", "start%", "capt%", "EO%"))
    lines = [header, "-" * len(header)]
    for i, (pid, eo) in enumerate(model.top(limit), start=1):
        player = state.players[pid]
        gwp = projections.projections.get(pid, {}).get(gw)
        lines.append(
            "%-3d %-18s %-4s %-4s %6.1f %6.2f %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
            % (i, player.web_name[:18], state.short_name(player.team_id),
               scoring.POS_NAME[player.position], player.price,
               gwp.xp if gwp is not None else 0.0,
               model.ownership.get(pid, 0.0), 100 * model.start_share.get(pid, 0.0),
               model.captaincy.get(pid, 0.0), eo)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    from gaffer.data.loaders import load_game_state
    from gaffer.model.xp import XPEngine

    parser = argparse.ArgumentParser(description="rank strategy smoke test")
    parser.add_argument("--gw", type=int, default=0, help="gameweek (default: current)")
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()

    t0 = time.time()
    game = load_game_state(cfg, progress=False)
    print("loaded %s in %.2fs" % (game.summary(), time.time() - t0))
    GW = args.gw or game.current_gw

    eng = XPEngine(cfg, mc_draws=args.draws).fit(game)
    t0 = time.time()
    PS = eng.project(game, list(range(GW, GW + 6)))
    print("projected GW%d-%d in %.2fs" % (GW, GW + 5, time.time() - t0))

    print("\n" + "=" * 78)
    print("EFFECTIVE OWNERSHIP — GW%d" % GW)
    print("=" * 78)
    OWN = estimate_ownership(game, PS, GW)
    print("beta %.3f, pool %.0f%%, top share %.0f%%, template captain %s"
          % (OWN.beta, OWN.captaincy_pool, 100 * OWN.top_captain_share,
             game.players[OWN.template_captain].web_name if OWN.template_captain else "n/a"))
    print("sum ownership %.0f%% (expect ~1500), sum EO %.0f%% (expect ~1200 = 11 starters + captain)"
          % (OWN.total_ownership, OWN.total_effective))
    print()
    print(format_ownership_table(OWN, game, PS, limit=20))

    PROF = haul_profiles(PS, GW, engine=eng)

    print("\n" + "=" * 78)
    print("DIFFERENTIALS — pure points (w=0) vs pure rank chasing (w=1)")
    print("=" * 78)
    for W in (0.0, 1.0):
        rows = differential_table(game, PS, GW, cfg, ownership=OWN, profiles=PROF,
                                  differential_weight=W)
        print("\ndifferential_weight = %.1f" % W)
        print(format_differential_table(rows, limit=12))

    print("\n" + "=" * 78)
    print("SANITY")
    print("=" * 78)
    print("  sum EO / 100 = %.2f copies (11 starters + 1 captain = 12.00)"
          % (OWN.total_effective / 100.0))
    print("  field expected GW score = %.1f" % field_expected_score(OWN, PS, GW))
    print("  field expected captain return = %.2f" % field_expected_captain_xp(OWN, PS, GW))
