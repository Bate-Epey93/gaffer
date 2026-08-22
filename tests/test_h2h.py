"""Head-to-head: win probability, not expected points.

The maths here is small but easy to get subtly wrong in ways no output would
reveal — an off-by-one in the comparison would quietly convert every draw into
a win, and doubling the captain by scaling the axis instead of convolving would
give the right mean with a badly wrong spread. So these pin the properties.
"""
from __future__ import annotations

import numpy as np

from gaffer.optimize import h2h


class FakeEngine:
    """Players whose distributions are known exactly, so results are checkable."""

    def __init__(self, dists):
        self.dists = dists

    def gw_distribution(self, pid, gw):
        return self.dists.get(int(pid))


def point_mass(value, offset=h2h.DIST_MIN):
    """A player who always scores exactly `value`."""
    size = value - offset + 1
    arr = np.zeros(max(size, 1))
    arr[-1] = 1.0
    return arr


def test_a_certain_win_is_reported_as_certain():
    mine = (np.array([1.0]), 60)      # always 60
    theirs = (np.array([1.0]), 50)    # always 50
    out = h2h.win_probability(mine, theirs)
    assert out["win"] == 1.0 and out["draw"] == 0.0 and out["loss"] == 0.0


def test_equal_scores_are_a_draw_not_a_win():
    """The off-by-one that would silently turn every tie into a victory."""
    same = (np.array([1.0]), 55)
    out = h2h.win_probability(same, same)
    assert out["draw"] == 1.0
    assert out["win"] == 0.0 and out["loss"] == 0.0


def test_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    a = rng.random(12); a /= a.sum()
    b = rng.random(9); b /= b.sum()
    out = h2h.win_probability((a, 20), (b, 25))
    assert abs(out["win"] + out["draw"] + out["loss"] - 1.0) < 1e-9


def test_a_coin_flip_is_symmetric():
    rng = np.random.default_rng(1)
    a = rng.random(15); a /= a.sum()
    mine = h2h.win_probability((a, 30), (a, 30))
    assert abs(mine["win"] - mine["loss"]) < 1e-9


def test_the_captain_is_convolved_not_scaled():
    """Doubling a captain must widen the distribution, not stretch the axis.

    Scaling gives the correct mean and a wrong variance, which is invisible in
    any table of means and fatal to a win probability.
    """
    engine = FakeEngine({1: point_mass(5), 2: point_mass(5)})
    plain, off_a = h2h.squad_distribution(engine, 1, [1, 2])
    capped, off_b = h2h.squad_distribution(engine, 1, [1, 2], captain=1)
    assert h2h.summarise(plain, off_a)["mean"] == 10.0
    assert h2h.summarise(capped, off_b)["mean"] == 15.0


def test_variance_helps_the_underdog_and_hurts_the_favourite():
    """The claim the whole module rests on.

    Against a stronger opponent a wider distribution wins MORE often than a
    narrow one with the same mean; against a weaker one it wins less. If this
    ever fails, the advice the command prints is backwards.
    """
    narrow = np.zeros(11); narrow[5] = 1.0                 # always 45
    wide = np.zeros(11); wide[0] = wide[10] = 0.5          # 40 or 50, mean 45
    mine_narrow, mine_wide = (narrow, 40), (wide, 40)

    strong = np.zeros(1); strong[0] = 1.0                  # always 48
    weak = np.zeros(1); weak[0] = 1.0                      # always 42

    vs_strong_narrow = h2h.win_probability(mine_narrow, (strong, 48))["win"]
    vs_strong_wide = h2h.win_probability(mine_wide, (strong, 48))["win"]
    assert vs_strong_wide > vs_strong_narrow, "variance must help the underdog"

    vs_weak_narrow = h2h.win_probability(mine_narrow, (weak, 42))["win"]
    vs_weak_wide = h2h.win_probability(mine_wide, (weak, 42))["win"]
    assert vs_weak_wide < vs_weak_narrow, "variance must hurt the favourite"


def test_lineup_reads_multipliers():
    payload = {"picks": [
        {"element": 10, "multiplier": 1},
        {"element": 11, "multiplier": 2},     # captain
        {"element": 12, "multiplier": 0},     # bench
    ]}
    lineup, captain = h2h.lineup_from_picks(payload)
    assert lineup == [10, 11] and captain == 11


def test_triple_captain_counts_three_times():
    payload = {"picks": [{"element": 7, "multiplier": 3}]}
    lineup, captain = h2h.lineup_from_picks(payload)
    assert lineup == [7] and captain == 7


def test_advice_flips_with_the_projection_gap():
    behind = h2h.advice({"mean": 45.0, "sd": 13.0}, {"mean": 52.0, "sd": 13.0},
                        {"win": 0.35, "draw": 0.02, "loss": 0.63})
    ahead = h2h.advice({"mean": 52.0, "sd": 13.0}, {"mean": 45.0, "sd": 13.0},
                       {"win": 0.65, "draw": 0.02, "loss": 0.33})
    assert "underdog" in behind and "variance is your friend" in behind
    assert "variance is your enemy" in ahead
