"""Shared fixtures. Nothing in this suite is allowed to touch the network.

Two independent worlds are available to a test:

* the **real** 2026/27 game, rebuilt from the warm ``data/cache`` through the
  production loader, with an ``FPLClient`` whose ``_fetch`` raises. If a cache
  entry is missing the load fails loudly rather than silently going online, and
  the affected tests are skipped with a message saying which key was missing.
* the **synthetic** league from ``tests/synthetic.py``: 20 clubs, 380 fixtures,
  a whole simulated prior season scored with the real ``gaffer.core.scoring``
  helpers. It needs no cache at all, so the invariants that matter still get
  asserted on a machine that has never run ``refresh``.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import pytest

from gaffer.core.config import CACHE_DIR, Config
from gaffer.data.cache import TTL_FOREVER, Cache
from gaffer.data.fpl_api import FPLClient


class FrozenCache(Cache):
    """A cache that never expires an entry.

    ``bootstrap`` has a 5-minute TTL and ``element-summary`` a 6-hour one, so a
    warm cache goes stale between runs and the loader would quietly reach for
    the network. Pinning every read to ``TTL_FOREVER`` makes the suite depend on
    the *contents* of the cache and not on when it was written.
    """

    def get(self, key: str, ttl: Optional[int] = None) -> Any:
        return super().get(key, ttl=TTL_FOREVER)


class OfflineFPLClient(FPLClient):
    """An FPL client with the network amputated."""

    def _fetch(self, url: str) -> Any:  # pragma: no cover - only runs on failure
        raise AssertionError(
            "the test suite tried to reach the network: GET %s. Every fixture "
            "must be served from data/cache; run `python -m gaffer.cli refresh` "
            "if the cache is cold." % url
        )


def _cache_is_warm() -> bool:
    return all(
        os.path.exists(os.path.join(CACHE_DIR, name))
        for name in ("bootstrap.json", "fixtures.json")
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> Config:
    cfg = Config.load()
    cfg.cache_ttl_seconds = TTL_FOREVER
    return cfg


@pytest.fixture()
def fresh_config() -> Config:
    """A throwaway Config for tests that mutate ``optimizer.locked_in`` etc."""
    cfg = Config.load()
    cfg.cache_ttl_seconds = TTL_FOREVER
    return cfg


# ---------------------------------------------------------------------------
# The real 2026/27 game, offline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def game_state(config: Config):
    """``GameState`` for 2026/27 built from the warm cache, offline."""
    if not _cache_is_warm():
        pytest.skip("data/cache has no bootstrap.json/fixtures.json")
    from gaffer.data.loaders import load_game_state

    client = OfflineFPLClient(config, FrozenCache(default_ttl=TTL_FOREVER))
    try:
        return load_game_state(config, client=client, progress=False)
    except AssertionError as exc:  # OfflineFPLClient refused to go online
        pytest.skip("warm cache incomplete: %s" % exc)


@pytest.fixture(scope="session")
def projections(config: Config):
    """The cached GW1-6 ``ProjectionSet`` written by ``gaffer.cli project``."""
    from gaffer.model.xp import XPEngine, projections_cache_path

    path = projections_cache_path(1, 6)
    if not os.path.exists(path):
        pytest.skip("no cached projection set at %s" % path)
    ps = XPEngine(config).load_projections(1, 6)
    if ps is None:  # pragma: no cover - only if the file vanishes mid-run
        pytest.skip("could not load %s" % path)
    return ps


# ---------------------------------------------------------------------------
# The synthetic league (no cache, no network)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_season():
    from tests import synthetic

    return synthetic.simulate_season()


@pytest.fixture(scope="session")
def synthetic_state(synthetic_season):
    from tests import synthetic

    return synthetic.make_game_state(synthetic_season, current_gw=1)


@pytest.fixture(scope="session")
def synthetic_projections(synthetic_state):
    from tests import synthetic

    return synthetic.make_projection_set(synthetic_state, gws=(1, 2, 3))
