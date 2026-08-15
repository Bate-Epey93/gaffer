#!/usr/bin/env python3
"""Render the gaffer app-icon set into ``gaffer/web/icons``.

Run it from anywhere:

    /Users/hardpro/My\\ Builds/gaffer/.venv/bin/python scripts/make_icons.py

The icons are committed, so this script exists to make them reproducible and
reviewable rather than to run at request time. Nothing in the server calls it.

Design
------
A dark field (#07080a, lifted to a barely-there vertical gradient so the icon
does not read as a hole punched in the home screen), a soft mint glow behind
the mark, and a heavy lowercase ``g`` in mint (#3ee0b8). The mark is pasted
through its own **ink bounding box**, not through the font's metrics box, so
every font in the fallback chain lands optically centred and at exactly the
requested height. Everything is drawn at 4x and reduced with LANCZOS, which is
what keeps the curves clean at 60px.

Four files, and the differences between them matter:

``icon-180.png``           apple-touch-icon. **Opaque, square, no rounded
                           corners** — iOS applies its own superellipse mask,
                           and an icon with transparency is composited onto
                           black, so a rounded PNG gets dark wedges in the
                           corners and a transparent one goes black entirely.
``icon-192.png``           manifest ``purpose="any"``. Rounded, transparent
``icon-512.png``           outside the corner radius, which is what a browser
                           expects to draw as-is.
``icon-512-maskable.png``  manifest ``purpose="maskable"``. Full-bleed and
                           opaque, with the mark kept inside the 80% safe zone
                           so an aggressive launcher mask (circle, squircle,
                           teardrop) cannot clip it.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- constants --

BG_TOP = (13, 16, 21)      # #0d1015 — the dashboard topbar gradient, top
BG_BOTTOM = (6, 7, 10)     # #06070a — a hair under --bg (#07080a), bottom
MINT = (62, 224, 184)      # #3ee0b8 — the dashboard accent

SS = 4                     # supersampling factor
CORNER_RATIO = 0.2237      # Apple's icon corner radius as a fraction of the side

# Heaviest first. A double-storey ``g`` in a black weight is the mark; each
# candidate is *rendered and measured* before it is trusted (see _load_font).
FONT_CANDIDATES: Tuple[Tuple[str, int, Optional[str]], ...] = (
    ("/System/Library/Fonts/Supplemental/Arial Black.ttf", 0, None),
    ("/System/Library/Fonts/SFNS.ttf", 0, "Heavy"),
    ("/System/Library/Fonts/SFNS.ttf", 0, "Bold"),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, None),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, None),
    ("/System/Library/Fonts/Helvetica.ttc", 1, None),
    ("/Library/Fonts/Arial Black.ttf", 0, None),
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "gaffer", "web", "icons")


# -------------------------------------------------------------------- font --

def _load_font(px: int) -> Tuple[Optional[ImageFont.FreeTypeFont], str]:
    """First candidate that both loads *and* renders a plausible ``g``.

    A font file existing is not evidence that it renders: .ttc collections hand
    back a different face than the name suggests, variable fonts ignore a
    weight this build of FreeType cannot select, and a missing glyph draws as
    an empty box or as nothing at all. So each candidate draws the glyph and
    the result is measured; anything that inks less than 2% of its own box, or
    comes out implausibly narrow or wide, is rejected and we move on.
    """
    for path, index, variation in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            font = ImageFont.truetype(path, px, index=index)
        except (OSError, ValueError):
            continue
        label = "%s%s" % (os.path.basename(path), " [%s]" % variation if variation else "")
        if variation:
            try:
                font.set_variation_by_name(variation)
            except (OSError, ValueError, AttributeError):
                continue  # this build cannot select the weight; do not settle for regular
        probe = Image.new("L", (px * 2, px * 2), 0)
        ImageDraw.Draw(probe).text((px // 2, px // 2), "g", font=font, fill=255)
        box = probe.getbbox()
        if box is None:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        if w < px * 0.20 or h < px * 0.30 or w > px * 1.6 or h > px * 1.8:
            continue
        ink = sum(probe.crop(box).point(lambda v: 1 if v > 128 else 0).getdata())
        if ink < 0.02 * w * h:
            continue
        return font, label
    return None, "drawn geometry"


# -------------------------------------------------------------------- mark --

def _glyph_mask(height: int) -> Tuple[Image.Image, str]:
    """An 8-bit mask of the mark whose ink is exactly ``height`` tall."""
    font, label = _load_font(int(height * 1.6))
    if font is not None:
        px = int(height * 1.6)
        canvas = Image.new("L", (px * 3, px * 3), 0)
        ImageDraw.Draw(canvas).text((px, px // 2), "g", font=font, fill=255)
        box = canvas.getbbox()
        assert box is not None  # _load_font already proved the glyph inks
        mask = canvas.crop(box)
        scale = height / float(mask.height)
        return mask.resize((max(1, int(round(mask.width * scale))), height),
                           Image.LANCZOS), label
    return _drawn_g(height), label


def _drawn_g(height: int) -> Image.Image:
    """Fallback mark: a single-storey ``g`` built from arcs and bars.

    Used only if every font in the chain fails to render. It is deliberately
    geometric rather than an attempt to imitate a typeface — a clean shape that
    reads as a ``g`` beats a mangled glyph.
    """
    h = height
    w = int(h * 0.74)
    stroke = int(h * 0.155)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    bowl = int(h * 0.60)                       # x-height bowl
    d.ellipse([stroke // 2, stroke // 2, bowl - stroke // 2, bowl - stroke // 2],
              outline=255, width=stroke)
    stem_x = bowl - stroke
    d.rectangle([stem_x, stroke // 2, stem_x + stroke, h - int(h * 0.22)], fill=255)
    d.arc([stem_x - int(h * 0.42), h - int(h * 0.44), stem_x + stroke, h - stroke // 2],
          start=0, end=150, fill=255, width=stroke)
    return img


# -------------------------------------------------------------------- field --

def _gradient(size: int) -> Image.Image:
    grad = Image.new("RGB", (1, size))
    px = grad.load()
    for y in range(size):
        t = y / float(max(1, size - 1))
        px[0, y] = tuple(int(round(a + (b - a) * t)) for a, b in zip(BG_TOP, BG_BOTTOM))
    return grad.resize((size, size), Image.BILINEAR)


def render(size: int, rounded: bool, opaque: bool, mark_ratio: float,
           rim: bool) -> Tuple[Image.Image, str]:
    big = size * SS
    field = _gradient(big).convert("RGBA")

    if rounded:
        shape = Image.new("L", (big, big), 0)
        ImageDraw.Draw(shape).rounded_rectangle(
            [0, 0, big - 1, big - 1], radius=int(big * CORNER_RATIO), fill=255)
        field.putalpha(shape)

    # a soft mint bloom behind the mark: keeps a near-black icon from vanishing
    # into a dark wallpaper without adding anything that has to stay legible
    glow = Image.new("L", (big, big), 0)
    r = int(big * mark_ratio * 0.80)
    ImageDraw.Draw(glow).ellipse(
        [big // 2 - r, big // 2 - r, big // 2 + r, big // 2 + r], fill=52)
    glow = glow.filter(ImageFilter.GaussianBlur(big * 0.11))
    field.alpha_composite(Image.merge("RGBA", (
        Image.new("L", (big, big), MINT[0]),
        Image.new("L", (big, big), MINT[1]),
        Image.new("L", (big, big), MINT[2]),
        glow)))

    mask, font_label = _glyph_mask(int(big * mark_ratio))
    mark = Image.new("RGBA", mask.size, MINT + (0,))
    mark.putalpha(mask)
    field.alpha_composite(mark, ((big - mask.width) // 2, (big - mask.height) // 2))

    if rim:
        inset = int(big * 0.028)
        ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(ring).rounded_rectangle(
            [inset, inset, big - 1 - inset, big - 1 - inset],
            radius=int((big - 2 * inset) * CORNER_RATIO),
            outline=MINT + (46,), width=max(1, int(big * 0.007)))
        field.alpha_composite(ring)

    out = field.resize((size, size), Image.LANCZOS)
    if opaque:
        flat = Image.new("RGB", (size, size), BG_BOTTOM)
        flat.paste(out, (0, 0), out)
        out = flat
    return out, font_label


# --------------------------------------------------------------------- main --

def main(argv: List[str]) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = (
        # name,                  size, rounded, opaque, mark, rim
        ("icon-180.png",          180, False,   True,   0.58, False),
        ("icon-192.png",          192, True,    False,  0.58, True),
        ("icon-512.png",          512, True,    False,  0.58, True),
        ("icon-512-maskable.png", 512, False,   True,   0.42, False),
    )
    label = ""
    print("%-24s %6s %6s %-6s %9s" % ("file", "size", "mode", "alpha", "bytes"))
    for name, size, rounded, opaque, mark, rim in jobs:
        img, label = render(size, rounded=rounded, opaque=opaque,
                            mark_ratio=mark, rim=rim)
        path = os.path.join(OUT_DIR, name)
        img.save(path, "PNG", optimize=True)
        print("%-24s %6s %6s %-6s %9d"
              % (name, "%dx%d" % img.size, img.mode,
                 "yes" if img.mode == "RGBA" else "no", os.path.getsize(path)))
    print("\nmark rendered from: %s" % label)
    print("written to: %s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
