"""Head-to-head: play to win the week, not to maximise the week.

Classic rank rewards total points, so maximising expected points is correct
there. H2H rewards beating one opponent, and a 90-point win and a 60-point win
are both worth exactly one win. The objective is therefore P(my score > theirs),
which is emphatically not the same thing as E[my score]:

  - Behind on projection, variance is your friend. A safe 48 loses to their safe
    55 almost every time; you need the right tail, so you WANT the lumpier squad.
  - Ahead on projection, variance is your enemy. Cut it and bank the win.

An expected-points optimiser is blind to this. It plays every week identically
whether you are a heavy favourite or a heavy underdog, which is a real edge
given away — the fix is to score decisions against the opponent's distribution
rather than against a scalar.

The distributions are exact rather than sampled. Each player's per-gameweek
points distribution is already simulated by the engine, and points are additive
and (treated as) independent across players, so the squad distribution is their
convolution. That gives P(win) to the decimal without a Monte Carlo, and makes
captain comparisons stable rather than jittering by a few tenths per run.

The independence assumption is the honest weak point: two Arsenal players share
a clean sheet, so a real squad's variance is slightly wider than this computes.
It biases P(win) toward 50% a little, and it biases it the same way for both
sides, so the ranking of captain choices survives it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gaffer.model.xp import DIST_MIN


class H2HError(RuntimeError):
    pass


def player_distribution(engine: Any, player_id: int, gw: int) -> Optional[np.ndarray]:
    dist = engine.gw_distribution(int(player_id), int(gw))
    if dist is None:
        return None
    arr = np.asarray(dist, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else None


def squad_distribution(
    engine: Any, gw: int, lineup: Sequence[int], captain: Optional[int] = None,
    triple: bool = False,
) -> Tuple[np.ndarray, int]:
    """(probabilities, offset) for a starting XI. Index i is `offset + i` points.

    The captain is convolved a second time (a third for Triple Captain), which
    is exactly what doubling his score means for the distribution — not scaling
    the axis, which would be wrong for anything but the mean.
    """
    total = np.array([1.0])
    offset = 0
    copies = 3 if triple else 2
    for pid in lineup:
        dist = player_distribution(engine, pid, gw)
        if dist is None:
            continue
        reps = copies if (captain is not None and int(pid) == int(captain)) else 1
        for _ in range(reps):
            total = np.convolve(total, dist)
            offset += DIST_MIN
    if total.size <= 1:
        raise H2HError("no player distributions available for GW%d" % gw)
    return total, offset


def summarise(probs: np.ndarray, offset: int) -> Dict[str, float]:
    points = np.arange(probs.size) + offset
    mean = float((points * probs).sum())
    sd = float(np.sqrt(((points - mean) ** 2 * probs).sum()))
    cdf = np.cumsum(probs)

    def percentile(q: float) -> int:
        return int(points[min(int(np.searchsorted(cdf, q)), points.size - 1)])

    return {
        "mean": mean, "sd": sd,
        "p10": percentile(0.10), "median": percentile(0.50), "p90": percentile(0.90),
    }


def win_probability(
    mine: Tuple[np.ndarray, int], theirs: Tuple[np.ndarray, int]
) -> Dict[str, float]:
    """P(win), P(draw), P(loss) for two independent score distributions."""
    my_probs, my_offset = mine
    their_probs, their_offset = theirs
    their_cdf = np.cumsum(their_probs)

    win = draw = 0.0
    for i, p in enumerate(my_probs):
        if p <= 0:
            continue
        score = i + my_offset
        # index of the opponent scoring strictly less than `score`
        below = score - their_offset - 1
        if below >= their_probs.size - 1:
            p_below = 1.0
        elif below < 0:
            p_below = 0.0
        else:
            p_below = float(their_cdf[below])
        at = score - their_offset
        p_at = float(their_probs[at]) if 0 <= at < their_probs.size else 0.0
        win += p * p_below
        draw += p * p_at
    win = float(min(max(win, 0.0), 1.0))
    draw = float(min(max(draw, 0.0), 1.0))
    return {"win": win, "draw": draw, "loss": float(max(0.0, 1.0 - win - draw))}


def captain_options(
    engine: Any, gw: int, lineup: Sequence[int], opponent: Tuple[np.ndarray, int],
    names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Every armband in the XI, ranked by P(win) rather than by expected points.

    These orders differ, and that difference is the whole point of the module:
    as an underdog the highest-xP captain is often not the one that most often
    wins the week, because winning needs the tail rather than the average.
    """
    names = names or {}
    out: List[Dict[str, Any]] = []
    for pid in lineup:
        try:
            mine = squad_distribution(engine, gw, lineup, captain=pid)
        except H2HError:
            continue
        stats = summarise(*mine)
        result = win_probability(mine, opponent)
        out.append({
            "player_id": int(pid),
            "name": names.get(int(pid), str(pid)),
            "mean": stats["mean"], "sd": stats["sd"], "p90": stats["p90"],
            "win": result["win"], "draw": result["draw"], "loss": result["loss"],
            # A draw is half a win in most H2H scoring, so rank on that.
            "score": result["win"] + 0.5 * result["draw"],
        })
    out.sort(key=lambda row: -row["score"])
    return out


def lineup_from_picks(picks_payload: Dict[str, Any]) -> Tuple[List[int], Optional[int]]:
    """(starting XI, captain) from an FPL picks payload.

    multiplier 0 is a bench player, 2 the captain, 3 a Triple Captain.
    """
    lineup: List[int] = []
    captain: Optional[int] = None
    for pick in picks_payload.get("picks") or []:
        pid = pick.get("element") or pick.get("player_id") or pick.get("id")
        if pid is None:
            continue
        multiplier = int(pick.get("multiplier") or 0)
        if multiplier > 0:
            lineup.append(int(pid))
        if multiplier >= 2:
            captain = int(pid)
    return lineup, captain


def advice(mine_stats: Dict[str, float], theirs_stats: Dict[str, float],
           result: Dict[str, float]) -> str:
    """Which way variance cuts this week, in one sentence."""
    gap = mine_stats["mean"] - theirs_stats["mean"]
    win = result["win"] + 0.5 * result["draw"]
    if gap < -2.0:
        return ("You are the underdog by %.1f points on projection, so variance is "
                "your friend: prefer the higher-ceiling captain and the differential, "
                "because a safe week loses this tie %.0f%% of the time."
                % (-gap, 100 * (1 - win)))
    if gap > 2.0:
        return ("You are %.1f points ahead on projection, so variance is your enemy: "
                "take the safe captain and the template, and bank the %.0f%%."
                % (gap, 100 * win))
    return ("Within %.1f points of each other, so this is close to a coin flip at "
            "%.0f%%. The armband decides it — take the highest win probability below, "
            "not the highest expected score." % (abs(gap), 100 * win))
