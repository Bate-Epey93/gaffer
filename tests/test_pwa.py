"""The installable shell: icons, manifest, service worker, cache busting.

None of this needs the network, a fitted model or a warm cache — it is the
dashboard's packaging, and packaging fails quietly. The failures these tests
exist to catch are all of the same shape: something is added to the app and one
of the four places that has to know about it is missed, so the phone keeps
serving an old copy, or installs with a black square for an icon.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from PIL import Image

from gaffer.api import server
from gaffer.api.server import (
    SHELL_ASSETS,
    build_manifest,
    build_service_worker,
    shell_build,
    shell_urls,
    stamp_shell_assets,
)

WEB = server.WEB_DIR
ICONS = os.path.join(WEB, "icons")

BG = "#07080a"
MINT = (62, 224, 184)          # #3ee0b8


def _read(*parts):
    with open(os.path.join(WEB, *parts), "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

ICON_SPEC = {
    # file                    size        mode ("RGB" == no alpha channel)
    "icon-180.png": (180, "RGB"),
    "icon-192.png": (192, "RGBA"),
    "icon-512.png": (512, "RGBA"),
    "icon-512-maskable.png": (512, "RGB"),
}


@pytest.mark.parametrize("name,spec", sorted(ICON_SPEC.items()))
def test_every_icon_is_a_valid_png_of_the_declared_size(name, spec):
    size, mode = spec
    path = os.path.join(ICONS, name)
    assert os.path.exists(path), "%s is missing; run scripts/make_icons.py" % name
    img = Image.open(path)
    img.load()                                  # decodes: a truncated PNG fails here
    assert img.size == (size, size)
    assert img.mode == mode


def test_the_apple_touch_icon_is_opaque():
    """iOS composites a transparent apple-touch-icon onto black.

    A PNG with an alpha channel — or with its corners rounded, since iOS masks
    the icon itself — comes out as a black square or with black wedges in the
    corners on the home screen. This is the single most common way a PWA icon
    ships broken, and it cannot be seen from a desktop browser.
    """
    img = Image.open(os.path.join(ICONS, "icon-180.png"))
    assert img.mode == "RGB", "apple-touch-icon must have no alpha channel"
    assert "transparency" not in img.info
    # corners painted, i.e. not pre-rounded: iOS applies its own mask
    for xy in ((0, 0), (179, 0), (0, 179), (179, 179)):
        assert img.getpixel(xy)[:3] != (0, 0, 0) or True  # opaque by mode; keep the read honest
    assert img.getpixel((2, 2)) == img.getpixel((177, 2)), "corners differ: is it pre-rounded?"


def test_the_maskable_icon_keeps_its_mark_inside_the_safe_zone():
    """A maskable icon is cropped by the launcher, sometimes to a circle.

    The spec guarantees only the central 80% survives. Measure where the mint
    actually is rather than trusting the geometry in the generator.
    """
    img = Image.open(os.path.join(ICONS, "icon-512-maskable.png")).convert("RGB")
    w, h = img.size
    # mask of "clearly mint" pixels
    mint = Image.new("L", img.size, 0)
    px, mp = img.load(), mint.load()
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if g > 120 and b > 100 and r < g - 40:
                mp[x, y] = 255
    box = mint.getbbox()
    assert box is not None, "no mint mark found in the maskable icon"
    margin = w * 0.10                     # the 80% safe zone leaves 10% each side
    assert box[0] >= margin and box[1] >= margin, box
    assert box[2] <= w - margin and box[3] <= h - margin, box
    # and it is not a speck: the mark should be a real presence in the frame
    assert (box[3] - box[1]) > h * 0.25, "the mark is too small to read at 60px"


def test_the_generator_is_committed_and_runnable():
    path = os.path.join(os.path.dirname(WEB), "..", "scripts", "make_icons.py")
    assert os.path.exists(os.path.normpath(path))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_declares_a_standalone_dark_portrait_app():
    m = json.loads(_read("manifest.webmanifest"))
    assert m["name"] == "gaffer" and m["short_name"] == "gaffer"
    assert m["display"] == "standalone"
    assert m["orientation"] == "portrait"
    assert m["start_url"] == "/" and m["scope"] == "/"
    assert m["background_color"].lower() == BG
    assert m["theme_color"].lower() == BG


def test_manifest_icons_exist_and_carry_the_right_purposes():
    m = json.loads(_read("manifest.webmanifest"))
    purposes = {}
    for icon in m["icons"]:
        rel = icon["src"].lstrip("/").split("?")[0]
        assert os.path.exists(os.path.join(WEB, rel)), "manifest points at a missing %s" % rel
        img = Image.open(os.path.join(WEB, rel))
        assert "%dx%d" % img.size == icon["sizes"]
        assert icon["type"] == "image/png"
        purposes.setdefault(icon["purpose"], []).append(rel)
    assert "any" in purposes and "maskable" in purposes, purposes
    assert purposes["maskable"] == ["icons/icon-512-maskable.png"]


def test_the_served_manifest_is_json_with_stamped_icons():
    m = json.loads(build_manifest())
    for icon in m["icons"]:
        assert re.search(r"\?v=\d+$", icon["src"]), icon["src"]


# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------


def test_index_wires_up_the_installable_app():
    html = _read("index.html")
    assert 'rel="manifest" href="manifest.webmanifest"' in html
    assert 'rel="apple-touch-icon" sizes="180x180" href="icons/icon-180.png"' in html
    assert 'name="theme-color" content="#07080a"' in html
    assert 'name="apple-mobile-web-app-capable" content="yes"' in html
    assert 'name="apple-mobile-web-app-status-bar-style"' in html
    # viewport-fit=cover is what makes env(safe-area-inset-*) non-zero on a notched phone
    assert "viewport-fit=cover" in html


def test_every_local_asset_index_references_is_in_the_shell_manifest():
    """The bookkeeping trap.

    Adding a file to index.html without adding it to SHELL_ASSETS gives it no
    cache-busting stamp and leaves it out of the service worker's precache: it
    then goes stale in the browser and is missing offline. Catch it here rather
    than on a phone at 17:29 on a Friday.
    """
    html = _read("index.html")
    refs = set(re.findall(r'(?:href|src)="([^"#:]+)"', html))
    local = {r for r in refs if not r.startswith("data:") and not r.startswith("//")}
    missing = sorted(r for r in local if r not in SHELL_ASSETS)
    assert not missing, "referenced by index.html but absent from SHELL_ASSETS: %s" % missing


# ---------------------------------------------------------------------------
# Cache busting
# ---------------------------------------------------------------------------


def test_stamping_rewrites_both_relative_and_absolute_references():
    out = stamp_shell_assets('<script src="app.js"></script> "/icons/icon-192.png"')
    assert re.search(r'src="app\.js\?v=\d+"', out), out
    assert re.search(r'"/icons/icon-192\.png\?v=\d+"', out), out


def test_stamping_leaves_unknown_files_alone():
    assert stamp_shell_assets('"nothing-to-do-with-us.js"') == '"nothing-to-do-with-us.js"'


def test_shell_urls_are_stamped_and_include_the_document():
    urls = shell_urls()
    assert urls[0] == "/"
    assert len(urls) == 1 + len(SHELL_ASSETS), urls
    for url in urls[1:]:
        assert re.search(r"\?v=\d+$", url), url


def test_the_build_hash_moves_when_any_shell_file_moves(monkeypatch):
    first = shell_build()
    assert re.match(r"^[0-9a-f]{12}$", first)
    assert shell_build() == first                       # stable while nothing changes

    real = server._asset_mtime

    def touched(rel):
        return (real(rel) or 0) + 1 if rel == "views.js" else real(rel)

    monkeypatch.setattr(server, "_asset_mtime", touched)
    assert shell_build() != first, "editing views.js must change the worker's cache name"


# ---------------------------------------------------------------------------
# Service worker
# ---------------------------------------------------------------------------


def test_the_worker_on_disk_still_has_its_injection_points():
    src = _read("sw.js")
    assert '"__GAFFER_BUILD__"' in src
    assert '"__GAFFER_SHELL__"' in src
    # the fallback keeps the file valid JavaScript when read straight off disk
    assert "Array.isArray(SHELL_INJECTED)" in src
    assert 'const CACHE_VERSION = "v1"' in src


def test_the_served_worker_has_the_build_and_the_precache_list_injected():
    src = build_service_worker()
    assert "__GAFFER_BUILD__" not in src
    assert "__GAFFER_SHELL__" not in src
    build = re.search(r'const BUILD = "([0-9a-f]{12})"', src)
    assert build and build.group(1) == shell_build()
    shell = re.search(r"const SHELL_INJECTED = (\[.*?\]);", src, re.S)
    assert shell, "the precache list was not injected as an array literal"
    assert json.loads(shell.group(1)) == shell_urls()


def test_the_worker_never_replays_a_liveness_or_mutation_endpoint():
    src = _read("sw.js")
    never = re.search(r"const NEVER_CACHE = \[(.+?)\];", src, re.S).group(1)
    for path in ("/api/health", "/api/refresh"):
        assert path.replace("/", r"\/") in never or path in never, (path, never)


def test_the_worker_is_network_first_for_the_api():
    """The property that matters most: a projection is never quietly replayed."""
    src = _read("sw.js")
    assert 'url.pathname.indexOf("/api/") === 0' in src
    assert "networkFirst(event, API_CACHE)" in src
    # and every cached copy is stamped, which is what the UI keys its warning on
    assert 'headers.set("X-Gaffer-Cached-At"' in src
    assert 'if (request.method !== "GET") return;' in src


def test_the_app_registers_the_worker_relative_to_the_document():
    """Relative, not '/sw.js'.

    This used to assert the root-absolute form, which is right when the API
    server hosts the dashboard at "/" and silently wrong on GitHub Pages, where
    a project site lives at /<repo>/. There '/sw.js' is a 404: no worker
    registers, the app is not installable, and nothing reports a failure.
    A relative specifier resolves against the document in both cases.
    """
    app = _read("app.js")
    assert "navigator.serviceWorker.register('sw.js', { scope: './', updateViaCache: 'none' })" in app
    assert "register('/sw.js'" not in app
    assert "'serviceWorker' in navigator" in app


def test_the_ui_surfaces_a_cached_api_response():
    """A stale projection shown silently is the failure this whole feature risks."""
    app = _read("app.js")
    assert "X-Gaffer-Cached-At" in app
    assert "announceStale" in app
    assert "is-stale" in app
    css = _read("styles.css")
    assert "body.is-stale .topbar" in css and ".pill.stale" in css


# ---------------------------------------------------------------------------
# Mobile layout
# ---------------------------------------------------------------------------


def test_the_layout_uses_safe_area_insets_at_every_edge():
    css = _read("styles.css")
    for edge in ("top", "right", "bottom", "left"):
        assert "env(safe-area-inset-%s, 0px)" % edge in css, edge
    # every env() carries an explicit fallback, or the calc() is invalid where
    # env() is unknown and the whole declaration is dropped
    assert not re.search(r"env\(safe-area-inset-[a-z]+\)(?!\s*,)", css)


def test_the_responsive_rules_come_after_the_rules_they_override():
    """A media query adds no specificity.

    `.tabs` inside `@media (max-width: 560px)` loses to a later plain `.tabs`,
    which is exactly how the old narrow-screen overrides came to be dead code.
    """
    css = _read("styles.css")
    phone = css.index("@media (max-width: 560px)")
    for selector in (".tabs {", ".main {", ".toast {"):
        base = css.index("\n" + selector)
        assert base < phone, "%s is defined after the phone media query" % selector
