"""The static export.

A broken export is a nasty failure: the site still loads, still looks right, and
quietly serves nothing or the wrong thing. So these tests assert the shape of
the bundle rather than that the command exited zero — the manifest exists, the
routes the dashboard asks for are all present as files, and the per-player
documents carry the per-gameweek overrides the front end reconstructs from.

The export runs the real model, so this is slow; it is one export shared by
every test in the module.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("httpx", reason="the export drives the app through TestClient")

from gaffer.ops.export import export_site  # noqa: E402

HORIZON = 2  # keep the fixture cheap; the shape does not depend on the length


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    summary = export_site(str(out), horizon=HORIZON, include_players=True)
    return {"dir": str(out), "summary": summary}


def read(site, *parts):
    with open(os.path.join(site["dir"], *parts), encoding="utf-8") as handle:
        return json.load(handle)


# --- the bundle exists in the shape the dashboard expects ------------------

def test_manifest_marks_the_build_as_static(site):
    """The front end switches modes on this flag; without it the site tries to
    call an API that is not there."""
    manifest = read(site, "data", "manifest.json")
    assert manifest["static"] is True
    assert manifest["generated_at"]
    assert manifest["gw"] >= 1
    assert len(manifest["gws"]) == HORIZON


def test_every_route_the_dashboard_calls_is_on_disk(site):
    """Each of these maps to a request() path in app.js. A missing file here is
    a view that renders an error on the phone."""
    manifest = read(site, "data", "manifest.json")
    required = ["state.json", "fixtures.json", "players.json",
                "projections.json", "chips.json", "optimize.json"]
    required += ["captain-gw%d.json" % g for g in manifest["gws"]]
    for name in required:
        path = os.path.join(site["dir"], "data", name)
        assert os.path.exists(path), "missing %s" % name
        assert os.path.getsize(path) > 0, "empty %s" % name


def test_the_web_assets_are_copied(site):
    for name in ("index.html", "app.js", "views.js", "ui.js", "styles.css",
                 "sw.js", "manifest.webmanifest"):
        assert os.path.exists(os.path.join(site["dir"], name)), name
    assert os.path.isdir(os.path.join(site["dir"], "icons"))
    # Pages runs Jekyll over the tree unless this file is present.
    assert os.path.exists(os.path.join(site["dir"], ".nojekyll"))


# --- the payloads are real -------------------------------------------------

def test_players_payload_holds_the_whole_league(site):
    players = read(site, "data", "players.json")["players"]
    assert len(players) > 500
    assert all("id" in p and "xp" in p for p in players[:20])


def test_the_frozen_squad_is_a_legal_fifteen(site):
    """If the export captured a broken solve, the phone shows a broken squad."""
    squad = read(site, "data", "optimize.json")
    picks = squad.get("squad") or squad.get("picks") or []
    if not picks and squad.get("decisions"):
        picks = squad["decisions"][0].get("squad") or []
    assert len(picks) == 15, "expected 15 players, got %d" % len(picks)


def test_captain_files_are_per_gameweek(site):
    """Each gameweek gets its own file; serving GW1's answer for GW4 would be
    wrong in a way nobody would notice."""
    manifest = read(site, "data", "manifest.json")
    seen = []
    for gw in manifest["gws"]:
        payload = read(site, "data", "captain-gw%d.json" % gw)
        assert payload.get("gw") == gw
        options = payload.get("options") or []
        assert options, "no captain options for GW%d" % gw
        seen.append(json.dumps(options[:3], sort_keys=True))
    if len(seen) > 1:
        assert len(set(seen)) > 1, "every gameweek returned identical captains"


# --- the per-player documents ---------------------------------------------

def test_player_documents_carry_per_gameweek_overrides(site):
    """`player` and `explanation` vary by gameweek; the rest does not. The front
    end rebuilds the live shape from these, so they must all be present."""
    manifest = read(site, "data", "manifest.json")
    player_dir = os.path.join(site["dir"], "data", "player")
    ids = [f[:-5] for f in os.listdir(player_dir) if f.endswith(".json")]
    assert len(ids) > 400

    doc = read(site, "data", "player", "%s.json" % ids[0])
    assert "by_gw" in doc and "fixtures" in doc
    overrides = doc["by_gw_overrides"]
    for gw in manifest["gws"]:
        assert str(gw) in overrides, "no override for GW%d" % gw
        assert "player" in overrides[str(gw)]


def test_optimize_request_is_recorded_for_the_adapter(site):
    """The front end compares the requested solve against this before serving
    the frozen one, so it must be present and match what the dashboard sends."""
    manifest = read(site, "data", "manifest.json")
    assert manifest["optimize_request"]["horizon"] == HORIZON


def test_summary_reports_what_it_wrote(site):
    summary = site["summary"]
    assert summary["players"] > 400
    assert summary["bytes"] > 100_000
    assert summary["seconds"] > 0


# --- installable from a subdirectory ---------------------------------------
# GitHub Pages serves a project site from /<repo>/, not the domain root. Every
# absolute path in the PWA shell is a 404 there, and the failure is silent:
# the page still loads, but the app is not installable and has no offline mode.

def test_manifest_paths_are_relative_not_root_absolute(site):
    """start_url "/" installed from Pages launches the wrong site entirely."""
    with open(os.path.join(site["dir"], "manifest.webmanifest"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["start_url"] == "./"
    assert manifest["scope"] == "./"
    assert manifest["id"] == "./"
    for icon in manifest["icons"]:
        assert not icon["src"].startswith("/"), icon["src"]


def test_service_worker_placeholders_are_substituted(site):
    """An unsubstituted BUILD pins the cache name, so one deploy's JavaScript
    is served cache-first forever."""
    with open(os.path.join(site["dir"], "sw.js"), encoding="utf-8") as fh:
        source = fh.read()
    assert '"__GAFFER_BUILD__"' not in source
    assert "const BUILD = " in source


def test_service_worker_derives_its_own_base(site):
    """The worker must not assume it lives at the origin root."""
    with open(os.path.join(site["dir"], "sw.js"), encoding="utf-8") as fh:
        source = fh.read()
    assert "self.location.pathname.replace" in source
    # The precache list has to be built from that base, not from "/". Match
    # actual array entries, not prose: the comments above SHELL quote the old
    # root-absolute paths precisely to explain why they were wrong.
    entries = [line.strip() for line in source.splitlines()
               if line.strip().startswith('"/') and line.strip().endswith('",')]
    assert not entries, "root-absolute precache entries: %s" % entries
    assert 'BASE + "styles.css"' in source


def test_app_registers_the_worker_relatively(site):
    with open(os.path.join(site["dir"], "app.js"), encoding="utf-8") as fh:
        source = fh.read()
    assert "register('sw.js'" in source
    assert "register('/sw.js'" not in source
