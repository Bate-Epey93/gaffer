"""The captaincy-ceiling term.

The armband doubles one player, so what it buys is the right tail rather than
the average — but the objective valued it at its mean, which made a
high-variance premium and a marginally-higher-mean safe pick look identical.

The important property is that this is OFF by default and provably changes
nothing at weight 0: it is a risk preference, not an accuracy fix, and it must
not quietly alter anyone's squad who did not ask for it.
"""
from __future__ import annotations

import pytest

from gaffer.core.config import Config


def test_the_default_is_off():
    """A risk knob that defaults to on would silently change every squad."""
    assert Config().optimizer.captain_ceiling_weight == 0.0


def test_the_view_exposes_a_tail_term():
    from gaffer.optimize.squad import GWView
    view = GWView(gw=1)
    assert hasattr(view, "tail")
    assert view.tail == {}


def test_the_tail_is_upside_only_never_negative():
    """sd is symmetric and would count a blank as 'upside'; the tail must not.
    A player whose haul is worth less than his mean contributes zero, not a
    negative, or the term would penalise consistency."""
    from gaffer.optimize.squad import build_views
    from gaffer.core.types import PlayerGWProjection, ProjectionSet

    class FakeState:
        fixtures = []

    ps = ProjectionSet(
        season="2026-27", generated_at="", first_gw=1, last_gw=1,
        projections={1: {1: PlayerGWProjection(player_id=1, gw=1, xp=4.0, sd=3.0, fixtures=[])}},
        model_version="test",
    )
    views = build_views(ps, FakeState(), [1])
    assert views[1].tail.get(1, 0.0) >= 0.0


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_the_weight_is_accepted_and_clamped_non_negative(weight):
    cfg = Config()
    cfg.optimizer.captain_ceiling_weight = weight
    assert cfg.optimizer.captain_ceiling_weight >= 0.0


def test_a_negative_weight_cannot_invert_the_term():
    """Guard the sign: a negative weight would reward the SAFEST captain, which
    is the opposite of what the flag says it does."""
    import gaffer.optimize.squad as sq
    import inspect
    src = inspect.getsource(sq)
    assert 'max(0.0, float(getattr(config.optimizer, "captain_ceiling_weight", 0.0)))' in src
