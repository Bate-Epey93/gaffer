"""Freeze the whole dashboard into a directory of static files.

The point is a phone that opens instantly. A hosted server on a free tier sleeps
when idle, and waking gaffer is not a web framework booting — it is refetching
587 player summaries and refitting six models, which is thirty to ninety seconds
of spinner at the exact moment you are checking a deadline. Nothing about the
weekly workflow actually needs a live server: the projections only change when
the data does, so they can be computed on a schedule and served as flat JSON
from a CDN, which costs nothing and answers in milliseconds.

The one design decision worth stating: every payload here is captured by calling
the real FastAPI app in-process through Starlette's TestClient, not by
re-serialising the model a second time. A parallel serialiser would drift from
the live one the first time either changed, and the dashboard would then need to
handle two subtly different shapes. Driving the actual routes means the static
files are the same bytes the live API would have returned, so the front end can
treat both modes identically.

What is lost, and it is worth being honest about it: re-optimising against
constraints you change on the phone (a different budget, a locked player, a
longer horizon) needs a solver, and a solver needs a server. The frozen site
ships the default optimisation only. Everything else — projections, the squad,
captaincy, chips, per-player breakdowns — is fully present.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from gaffer.core.config import PROJECT_ROOT, Config

WEB_DIR = os.path.join(PROJECT_ROOT, "gaffer", "web")

# Files copied verbatim from gaffer/web into the site root.
ASSET_FILES = (
    "index.html",
    "app.js",
    "ui.js",
    "views.js",
    "styles.css",
    "sw.js",
    "manifest.webmanifest",
)
ASSET_DIRS = ("icons",)


class ExportError(RuntimeError):
    pass


def _write_json(path: str, payload: Any) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # separators drop the whitespace: this is machine-read only, and the saving
    # runs to megabytes across 587 player files.
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return len(body.encode("utf-8"))


def _shell_build_hash(out_dir: str) -> str:
    """A short digest of the shell, used to name the service worker's cache.

    The live server stamps every shell URL with its mtime, so a cached hit is
    provably the right bytes. The exported site has no such stamping, which
    makes the shell cache-first *forever* unless the cache name itself changes
    when the code does. Hashing the shell gives exactly that.
    """
    digest = hashlib.sha256()
    for name in sorted(ASSET_FILES):
        path = os.path.join(WEB_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()[:12]


def _rewrite_manifest(path: str) -> None:
    """Make the web app manifest work from a subdirectory.

    GitHub Pages serves a project site from /<repo>/, but the checked-in
    manifest declares start_url, scope and id as "/". Installed from Pages that
    launches the *domain root*, which is a different site entirely — the icon
    lands on the home screen and opens the wrong page. Relative values resolve
    against the manifest's own URL, so they are correct at any depth.
    """
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["id"] = "./"
    manifest["start_url"] = "./"
    manifest["scope"] = "./"
    for icon in manifest.get("icons", []):
        icon["src"] = icon["src"].lstrip("/")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _copy_assets(out_dir: str) -> List[str]:
    copied = []
    build = _shell_build_hash(out_dir)
    for name in ASSET_FILES:
        src = os.path.join(WEB_DIR, name)
        if not os.path.exists(src):
            raise ExportError("missing web asset: %s" % src)
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        if name == "sw.js":
            # The server substitutes these at serve time; nothing does it for a
            # static build, so an unsubstituted BUILD would pin every deploy to
            # one cache name and serve the first build's JavaScript forever.
            with open(dst, encoding="utf-8") as handle:
                source = handle.read()
            source = source.replace('"__GAFFER_BUILD__"', json.dumps(build))
            with open(dst, "w", encoding="utf-8") as handle:
                handle.write(source)
        elif name == "manifest.webmanifest":
            _rewrite_manifest(dst)
        copied.append(name)
    for name in ASSET_DIRS:
        src = os.path.join(WEB_DIR, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(out_dir, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(name + "/")
    return copied


def export_site(
    out_dir: str,
    config: Optional[Config] = None,
    horizon: int = 6,
    include_players: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Render every dashboard route to disk under ``out_dir``.

    Returns a summary dict: what was written, how big it is, how long it took.
    """
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover - depends on httpx being present
        raise ExportError(
            "the export needs httpx installed (pip install -r requirements.txt): %s" % exc
        )

    from gaffer.api.server import create_app

    say = progress or (lambda _m: None)
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "data")

    app = create_app(config=config, horizon=horizon)
    # Loopback peer plus the shared secret when one is configured: the export
    # runs inside CI where GAFFER_PASSWORD may well be set, and the auth gate
    # quite correctly does not exempt loopback once a password exists.
    headers = {}
    password = (os.environ.get("GAFFER_PASSWORD") or "").strip()
    if password:
        headers["X-Gaffer-Key"] = password
    client = TestClient(app, client=("127.0.0.1", 5000), headers=headers)

    written: Dict[str, int] = {}

    def capture(route: str, filename: str) -> Any:
        response = client.get(route)
        if response.status_code != 200:
            raise ExportError("%s returned %d: %s"
                              % (route, response.status_code, response.text[:200]))
        payload = response.json()
        written[filename] = _write_json(os.path.join(data_dir, filename), payload)
        return payload

    say("fitting the model and capturing /state (this is the slow one)")
    state = capture("/api/state", "state.json")

    gw = int(state.get("gw") or state.get("current_gw") or 1)
    gws = [g for g in range(gw, gw + horizon) if g <= 38]

    say("fixtures")
    capture("/api/fixtures", "fixtures.json")

    say("projections for GW%d-%d" % (gws[0], gws[-1]))
    capture("/api/players?gw=%d&horizon=%d" % (gw, horizon), "players.json")
    capture("/api/projections?first=%d&last=%d" % (gws[0], gws[-1]), "projections.json")

    say("captaincy, %d gameweek(s)" % len(gws))
    for g in gws:
        capture("/api/captain?gw=%d" % g, "captain-gw%d.json" % g)

    say("chips (solves its own MILPs, ~20s)")
    capture("/api/chips", "chips.json")

    say("the recommended squad")
    # Freeze exactly the request the dashboard makes, so the static adapter can
    # recognise it and serve this file. Anything else is a different question
    # and honestly needs the solver.
    optimize_request: Dict[str, Any] = {"horizon": horizon}
    entry_id = getattr(config or Config(), "entry_id", None)
    if entry_id:
        optimize_request["entry_id"] = int(entry_id)
    response = client.post("/api/optimize", json=optimize_request)
    if response.status_code != 200:
        raise ExportError("/api/optimize returned %d: %s"
                          % (response.status_code, response.text[:200]))
    written["optimize.json"] = _write_json(
        os.path.join(data_dir, "optimize.json"), response.json())

    player_count = 0
    if include_players:
        # Only `player` and `explanation` vary by gameweek; `by_gw`, `fixtures`
        # and the history blocks are identical across the horizon. So one file
        # per player carries the shared part once and the two varying fields
        # keyed by gameweek, instead of one file per player per gameweek.
        ids = [int(p["id"]) for p in (state.get("players") or [])] or None
        if ids is None:
            listing = client.get("/api/players?gw=%d&horizon=%d" % (gw, horizon)).json()
            ids = [int(p["id"]) for p in listing.get("players", [])]
        say("per-player breakdowns (%d players)" % len(ids))
        for index, pid in enumerate(ids):
            base = client.get("/api/player/%d?gw=%d" % (pid, gw))
            if base.status_code != 200:
                continue
            doc = base.json()
            overrides: Dict[str, Any] = {}
            for g in gws:
                if g == gw:
                    overrides[str(g)] = {"player": doc.get("player"),
                                         "explanation": doc.get("explanation")}
                    continue
                extra = client.get("/api/player/%d?gw=%d" % (pid, g))
                if extra.status_code != 200:
                    continue
                body = extra.json()
                overrides[str(g)] = {"player": body.get("player"),
                                     "explanation": body.get("explanation")}
            doc["by_gw_overrides"] = overrides
            written["player/%d.json" % pid] = _write_json(
                os.path.join(data_dir, "player", "%d.json" % pid), doc)
            player_count += 1
            if progress and index and index % 100 == 0:
                say("  %d/%d players" % (index, len(ids)))

    say("copying the dashboard assets")
    assets = _copy_assets(out_dir)

    # The dashboard probes for this file at boot: present means "you are looking
    # at a frozen snapshot, do not try to POST anything".
    manifest = {
        "static": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": (config or Config()).season,
        "gw": gw,
        "gws": gws,
        "horizon": horizon,
        "deadline": state.get("deadline") or state.get("next_deadline"),
        # The exact POST body optimize.json answers; the front end compares
        # against it rather than guessing which solves are covered.
        "optimize_request": optimize_request,
        "players_exported": player_count,
        "bytes": sum(written.values()),
        "routes": sorted(written.keys()),
    }
    _write_json(os.path.join(data_dir, "manifest.json"), manifest)

    # Pages serves any path starting with an underscore oddly and runs Jekyll
    # over the tree unless told not to; .nojekyll turns that off.
    with open(os.path.join(out_dir, ".nojekyll"), "w") as handle:
        handle.write("")

    summary = {
        "out_dir": os.path.abspath(out_dir),
        "files": len(written) + len(assets) + 2,
        "bytes": sum(written.values()),
        "players": player_count,
        "gw": gw,
        "gws": gws,
        "seconds": round(time.time() - started, 1),
        "generated_at": manifest["generated_at"],
    }
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    return (
        "exported GW%d-%d to %s\n"
        "  %d files, %.1f MB of JSON, %d player breakdowns, %.1fs"
        % (summary["gws"][0], summary["gws"][-1], summary["out_dir"],
           summary["files"], summary["bytes"] / (1024.0 * 1024.0),
           summary["players"], summary["seconds"])
    )
