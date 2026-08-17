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

BG_TOP = (55, 0, 60)       # #37003C — FPL's own purple, top of the field
BG_BOTTOM = (18, 0, 26)    # #12001a — the dashboard background, bottom
MINT = (0, 255, 135)       # #00FF87 — FPL green, the single accent
BALL_WHITE = (255, 255, 255)
BALL_INK = (43, 0, 48)     # the purple field, reused as the ball's panels

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


def _pentagon(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
              rotation: float, fill: Tuple[int, int, int, int]) -> None:
    import math
    pts = []
    for i in range(5):
        a = rotation + i * (2 * math.pi / 5)
        pts.append((cx + r * math.sin(a), cy - r * math.cos(a)))
    draw.polygon(pts, fill=fill)


def _ball(d: int) -> Image.Image:
    """A football, drawn to survive being 16px wide.

    Detail is the enemy here: a faithful 32-panel ball turns to grey mush at
    favicon size. So it is one central pentagon plus five at the seams, which is
    the least a shape can carry and still read unmistakably as a football. The
    panels are the icon's own purple rather than black, so the ball belongs to
    the mark instead of sitting on top of it.
    """
    import math
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.ellipse([0, 0, d - 1, d - 1], fill=BALL_WHITE + (255,))

    cx = cy = (d - 1) / 2.0
    R = d / 2.0
    panels = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panels)
    _pentagon(pd, cx, cy, R * 0.30, 0.0, BALL_INK + (255,))
    for i in range(5):
        a = i * (2 * math.pi / 5) + math.pi / 5
        px_ = cx + R * 0.74 * math.sin(a)
        py_ = cy - R * 0.74 * math.cos(a)
        _pentagon(pd, px_, py_, R * 0.26, a + math.pi, BALL_INK + (255,))

    # Clip the panels to the sphere: the outer five overhang by design, so that
    # they read as panels running off the edge rather than as floating dots.
    clip = Image.new("L", (d, d), 0)
    ImageDraw.Draw(clip).ellipse([0, 0, d - 1, d - 1], fill=255)
    panels.putalpha(Image.composite(panels.getchannel("A"), Image.new("L", (d, d), 0), clip))
    img.alpha_composite(panels)

    # A touch of shading so it reads as a sphere, not a sticker.
    shade = Image.new("L", (d, d), 0)
    ImageDraw.Draw(shade).ellipse([int(d * 0.10), int(d * 0.10), d - 1, d - 1], fill=70)
    shade = shade.filter(ImageFilter.GaussianBlur(d * 0.10))
    shade = Image.composite(shade, Image.new("L", (d, d), 0), clip)
    img.alpha_composite(Image.merge("RGBA", (
        Image.new("L", (d, d), BALL_INK[0]),
        Image.new("L", (d, d), BALL_INK[1]),
        Image.new("L", (d, d), BALL_INK[2]),
        shade)))
    return img


def _motion(canvas: Image.Image, bx: float, by: float, d: float) -> None:
    """Speed lines trailing the ball, tapering as they recede.

    Three strokes, not more: the streaks have to say "moving" at a glance and
    then get out of the way of the mark. They shorten and fade with distance
    from the ball, which is what reads as motion rather than as stripes, and
    they stop short of the ball so it stays a clean circle.
    """
    dr = ImageDraw.Draw(canvas)
    for i, (dy, length, alpha, width) in enumerate((
        (-0.24, 0.95, 225, 0.16),
        (0.04, 1.40, 180, 0.14),
        (0.32, 0.80, 130, 0.12),
    )):
        y = by + d * dy
        x1 = bx - d * 0.62
        x0 = x1 - d * length
        w = max(1, int(d * width))
        dr.rounded_rectangle([x0, y - w / 2.0, x1, y + w / 2.0],
                             radius=w / 2.0, fill=MINT + (alpha,))


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
    r = int(big * mark_ratio * 0.62)
    ImageDraw.Draw(glow).ellipse(
        [big // 2 - r, big // 2 - r, big // 2 + r, big // 2 + r], fill=30)
    glow = glow.filter(ImageFilter.GaussianBlur(big * 0.13))
    field.alpha_composite(Image.merge("RGBA", (
        Image.new("L", (big, big), MINT[0]),
        Image.new("L", (big, big), MINT[1]),
        Image.new("L", (big, big), MINT[2]),
        glow)))

    # The mark is a lockup: the ``g`` with a football struck away from it, the
    # streaks trailing back toward the letter. Both are composed on one layer
    # and centred together, so the pair is optically centred rather than the
    # letter being centred and the ball hanging off the edge.
    gh = int(big * mark_ratio * 0.86)
    mask, font_label = _glyph_mask(gh)
    ball_d = int(gh * 0.42)

    # The ball flies in a clear band ABOVE the letter, not across it. The first
    # attempt overlapped the two and ran the speed lines straight through the
    # bowl, which read as a strikethrough rather than as movement — the streaks
    # have to travel through empty space to say "motion".
    band = int(ball_d * 1.05)
    lock_w = mask.width + int(ball_d * 0.55)
    lock_h = mask.height + band
    lock = Image.new("RGBA", (lock_w, lock_h), (0, 0, 0, 0))

    letter = Image.new("RGBA", mask.size, MINT + (0,))
    letter.putalpha(mask)
    lock.alpha_composite(letter, (0, lock_h - mask.height))

    # Struck away to the upper right, streaks trailing back over the letter's
    # shoulder through the empty band.
    bx = lock_w - ball_d
    by = 0
    _motion(lock, bx + ball_d / 2.0, by + ball_d / 2.0, float(ball_d))
    lock.alpha_composite(_ball(ball_d), (int(bx), int(by)))

    field.alpha_composite(lock, ((big - lock_w) // 2, (big - lock_h) // 2))

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
