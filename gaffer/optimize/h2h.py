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


def find_opponent(matches: Dict[str, Any], entry_id: int, gw: int) -> Optional[Dict[str, Any]]:
    """Who you face in `gw`, from a league's fixture list.

    An H2H league pairs you with "AVERAGE" when the entrant count is odd; that
    is a real opponent scoring the league average, but it has no entry id and
    therefore no squad to model, so it is reported and skipped rather than
    silently dropped.
    """
    for match in matches.get("results") or []:
        if int(match.get("event") or 0) != int(gw):
            continue
        for me, them in ((1, 2), (2, 1)):
            if match.get("entry_%d_entry" % me) == int(entry_id):
                return {
                    "entry": match.get("entry_%d_entry" % them),
                    "name": match.get("entry_%d_name" % them),
                    "player": match.get("entry_%d_player_name" % them),
                    "is_average": match.get("entry_%d_entry" % them) is None,
                    "is_knockout": bool(match.get("is_knockout")),
                }
    return None


def build_report(client: Any, engine: Any, state: Any, entry_id: int,
                 gw: int, names: Dict[int, str]) -> Dict[str, Any]:
    """Every H2H tie this gameweek, scored on win probability.

    Runs in CI rather than the browser for the same reason the squad import
    does: reading another manager's picks is a cross-origin request the FPL API
    refuses, and CI is not a browser.
    """
    from datetime import datetime, timezone

    def picks_for(target: int) -> Optional[Tuple[List[int], Optional[int], int]]:
        for candidate in (gw, gw - 1, gw - 2):
            if candidate < 1:
                continue
            try:
                payload = client.entry_picks(int(target), candidate)
            except Exception:
                continue
            lineup, captain = lineup_from_picks(payload)
            if len(lineup) >= 11:
                return lineup, captain, candidate
        return None

    report: Dict[str, Any] = {
        "gw": int(gw), "entry_id": int(entry_id),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leagues": [], "unavailable": None,
    }

    mine = picks_for(entry_id)
    if not mine:
        report["unavailable"] = ("your GW%d squad is not published yet — FPL "
                                 "releases picks only after the deadline" % gw)
        return report
    my_lineup, my_captain, my_gw = mine
    my_dist = squad_distribution(engine, gw, my_lineup, captain=my_captain)
    my_stats = summarise(*my_dist)
    my_stats["captain"] = names.get(my_captain, "-")
    my_stats["picks_gw"] = my_gw
    report["mine"] = my_stats

    try:
        entry = client.entry(int(entry_id))
        leagues = (entry.get("leagues") or {}).get("h2h") or []
    except Exception as exc:
        report["unavailable"] = "could not read your leagues: %s" % exc
        return report

    for league in leagues:
        lid = league.get("id")
        row: Dict[str, Any] = {"league_id": lid, "name": league.get("name")}
        try:
            matches = client.h2h_matches(int(lid), int(entry_id))
        except Exception as exc:
            row["error"] = str(exc)
            report["leagues"].append(row)
            continue

        opponent = find_opponent(matches, entry_id, gw)
        if not opponent:
            row["error"] = "no GW%d fixture in this league" % gw
            report["leagues"].append(row)
            continue
        row["opponent"] = opponent
        if opponent.get("is_average") or not opponent.get("entry"):
            row["error"] = ("this week you play the league AVERAGE, which has no "
                            "squad to model")
            report["leagues"].append(row)
            continue

        theirs = picks_for(int(opponent["entry"]))
        if not theirs:
            row["error"] = "their squad is not published yet"
            report["leagues"].append(row)
            continue
        their_lineup, their_captain, their_gw = theirs
        their_dist = squad_distribution(engine, gw, their_lineup, captain=their_captain)
        their_stats = summarise(*their_dist)
        their_stats["captain"] = names.get(their_captain, "-")
        their_stats["picks_gw"] = their_gw

        result = win_probability(my_dist, their_dist)
        row.update({
            "theirs": their_stats,
            "win": result["win"], "draw": result["draw"], "loss": result["loss"],
            "advice": advice(my_stats, their_stats, result),
            "captains": captain_options(engine, gw, my_lineup, their_dist, names)[:8],
            "stale": (my_gw != gw or their_gw != gw),
        })
        report["leagues"].append(row)
    return report
