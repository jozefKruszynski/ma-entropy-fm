"""
Covers for the albums and the tracks, in the look of the covers of Music Assistant.

Music Assistant has AI generated covers for playlists (the Playlist Metadata provider): a rich
gradient of two or three colours chosen for the mood, a soft bloom and fine grain, a big white
line-art illustration cropped off the top and right edges, and the title in the calm lower-left
corner. This module makes the covers of Entropy FM the same way.

- The cover that is always there is drawn locally as a 1024 pixel JPEG in that look (players need
  a raster image; Music Assistant passes an SVG through as it is): the
  gradient follows the style and the key and mode of the day, the glyph is that of the style (a
  sequencer for Berlin School, a horizon for Ambient, a record for Lo-fi, a drum for Ritual, torn slices of a waveform for Cinematic Glitch)
  and is shaped by the events of the day, the dots-to-wave of the icon is a small
  detail, and the title is set lower left by the overlay of the covers. It is the same every time for the same area, day, style.
- When the user turns it on, an AI image engine of Music Assistant makes the cover instead, from
  the prompt of the upstream presets, with the title drawn over it afterwards. The only things in
  that prompt are the title (the area, which is the user's own label for their own cover) and a
  `{context}` of mood words from a fixed list: no entity id, name, person, time or count.
  A cover is made once and kept, again only when its title, style, prompt or engine changes, the
  old one stays until the new one is ready, there is a limit an hour, and a failure waits an hour,
  then six, then a day, then until something changes. The cover that is drawn is shown meanwhile.

The module imports nothing from Music Assistant (see `maart` for the upstream presets).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .timeline import VoiceKind, stable_hash

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from PIL import Image

    from .timeline import Series
    from .titles import DayFacts

SIZE = 1024
BINS = 96
MAX_DOTS = 160
COVER_OFF = "off"
COVER_PROCEDURAL = "procedural"
COVER_AI = "ai"
DEFAULT_AI_PER_HOUR = 30
AI_TIMEOUT_S = 300.0
# a failed cover is tried again after these delays, and after the last one only when it changes
FAILURE_BACKOFF_S = (3600.0, 6 * 3600.0, 24 * 3600.0)
MAX_STORE_BYTES = 120_000_000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

# the colours of the styles: the dark end of the gradient, the middle, the bright end and an accent
PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "ambient": ((14, 40, 66), (46, 120, 140), (130, 210, 205), (255, 214, 140)),
    "berlin_school": ((34, 10, 52), (170, 52, 100), (255, 150, 84), (255, 214, 120)),
    "lofi": ((46, 26, 24), (140, 84, 62), (244, 176, 104), (214, 124, 150)),
    "ritual": ((26, 14, 22), (126, 40, 48), (226, 112, 66), (255, 206, 138)),
    "glitch": ((10, 12, 24), (52, 38, 112), (214, 62, 96), (120, 230, 255)),
}
# the words of the context: all fixed, none comes from the data
STYLE_WORDS: dict[str, str] = {
    "ambient": "soft drifting light and slow breathing shapes",
    "berlin_school": "glowing sequences of light over a quiet machine landscape at sunset",
    "lofi": "a dusty warm evening, worn textures and tape grain",
    "ritual": "embers and slow ripples around a great drum in the dark, a quiet ceremony",
    "glitch": "a dark film title sequence, a waveform cut into slices and torn out of place",
}
ROOT_COLOURS = (
    "red",
    "red-orange",
    "orange",
    "amber",
    "yellow",
    "green",
    "teal",
    "cyan",
    "blue",
    "indigo",
    "violet",
    "magenta",
)
MODE_MOODS: dict[str, str] = {
    "dorian": "wistful and open",
    "aeolian": "dark and tender",
    "mixolydian": "bright and relaxed",
    "ionian": "clear and warm",
    "lydian": "airy and floating",
    "phrygian": "tense and shadowed",
    "pentatonic": "simple and calm",
}
MOTIF_WORDS: dict[VoiceKind, str] = {
    VoiceKind.DISHWASHER: "interlocking arpeggios like rippling water",
    VoiceKind.HOB: "small points of warm flame",
    VoiceKind.PHASE: "gentle shifts from one scene to the next",
    VoiceKind.MEDIA: "soft waves of sound",
    VoiceKind.COVER: "slow light through blinds",
    VoiceKind.DRONE: "a low steady hum",
    VoiceKind.CHORD: "a warm kitchen glow",
    VoiceKind.PLUCK: "single bright notes",
    VoiceKind.TEMPERATURE: "a faint haze of warmth",
    VoiceKind.HUMIDITY: "a veil of mist",
}


@dataclass(frozen=True)
class DayProfile:
    """What the day looked like, with nothing in it that names anyone or anything."""

    # activity in each quarter of an hour of the day, 0 to 1
    bins: np.ndarray = field(default_factory=lambda: np.zeros(BINS))
    # (fraction of the day, strength, kind index) of the events, strongest first
    events: tuple[tuple[float, float, int], ...] = ()
    kinds: frozenset[VoiceKind] = frozenset()
    intensity: float = 0.5

    def parts(self, count: int) -> list[float]:
        """Return the activity of the day in `count` equal parts, each 0 to 1."""
        cut = np.array_split(self.bins, count)
        values = [float(chunk.sum()) for chunk in cut]
        peak = max(values) or 1.0
        return [value / peak for value in values]


def profile_of(series: Sequence[Series]) -> DayProfile:
    """
    Describe a day from its series: the activity round the clock, its events and what kinds played.

    :param series: Every series of the area on the day.
    """
    bins = np.zeros(BINS)
    found: list[tuple[float, float, int]] = []
    kinds: set[VoiceKind] = set()
    order = list(VoiceKind)
    for item in series:
        if item.kind == VoiceKind.SUN:
            continue
        moved = np.abs(np.diff(item.norm))
        points = np.flatnonzero(moved > 0.1) + 1
        if len(points):
            kinds.add(item.kind)
        for index in points:
            fraction = float(np.clip(item.times[index] / 86400.0, 0.0, 0.9999))
            bins[int(fraction * BINS)] += float(moved[index - 1])
            found.append((fraction, float(moved[index - 1]), order.index(item.kind)))
    peak = float(bins.max())
    bins = bins / peak if peak > 0 else bins
    found.sort(key=lambda event: (-event[1], event[0], event[2]))
    kept = tuple(sorted(found[:MAX_DOTS]))
    intensity = float(np.clip(np.sum(bins) / (BINS * 0.25), 0.0, 1.0))
    return DayProfile(bins, kept, frozenset(kinds), intensity)


# -- the drawn cover -----------------------------------------------------------------------------


def _hex(colour: np.ndarray) -> str:
    """Return a colour as #rrggbb."""
    r, g, b = (int(np.clip(round(float(c)), 0, 255)) for c in colour)
    return f"#{r:02x}{g:02x}{b:02x}"


def mood_colours(style_key: str, root: int, mode: str, seed: str) -> list[str]:
    """
    Return the three colours of the gradient of a style on a day, dark to bright, and the accent.

    The key of the day shifts the colours a little, the mode makes them lighter or darker, and the
    seed shifts the middle of the gradient so that days of one style differ but stay a family.

    :param style_key: The style.
    :param root: The key of the day, 0 to 11.
    :param mode: The mode of the day.
    :param seed: Text that is the same for the same area, day and style.
    """
    dark, middle, bright, accent = (
        np.array(c, dtype=np.float64) for c in PALETTES.get(style_key, PALETTES["ambient"])
    )
    shift = 0.4 * (root % 12) / 11.0
    mix = np.array([[1 - shift, shift, 0.0], [0.0, 1 - shift, shift], [shift, 0.0, 1 - shift]])
    light = {"aeolian": 0.88, "phrygian": 0.84, "dorian": 0.96, "mixolydian": 1.08, "lydian": 1.14}
    lift = light.get(mode, 1.0)
    rng = np.random.default_rng(stable_hash("mood", seed) % 2**32)
    wander = 0.88 + 0.24 * float(rng.random())
    colours = [mix @ c * lift for c in (dark, middle * wander, bright, accent)]
    return [_hex(np.clip(c, 0, 255)) for c in colours]


# A glyph is a list of drawing operations in its own units, centred on (0, 0):
#   ("line", [(x, y), ...], width, colour, alpha)   a stroked line with round ends and joints
#   ("fill", [(x, y), ...], colour)                 a filled polygon
#   ("ring", x, y, radius, width, colour, alpha)    a stroked circle
#   ("disc", x, y, radius, colour)                  a filled circle
#   ("dash", x, y, radius, width, colour)           a dashed circle
#   ("box", x, y, size, width, colour)              a rounded square, stroked (width) or filled (0)
# The colour is "w" (white) or "a" (the accent of the style).
type Op = tuple[Any, ...]


def _arc(  # noqa: PLR0913, PLR0917
    cx: float, cy: float, radius: float, start: float, end: float, steps: int = 48
) -> list[tuple[float, float]]:
    """Return the points of an arc from `start` to `end` degrees (0 is to the right, y points down)."""
    angles = np.radians(np.linspace(start, end, steps))
    return [(cx + radius * float(np.cos(a)), cy + radius * float(np.sin(a))) for a in angles]


def _glyph_sequencer(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return stepped sequencer bars: as high as what happened in each stretch of the day."""
    steps = 9 + int(rng.integers(0, 5))
    levels = profile.parts(steps)
    gap = 720 // steps
    ops: list[Op] = []
    top = int(np.argmax(levels))
    for index, level in enumerate(levels):
        height = 90 + 330 * (0.25 + 0.75 * level)
        x = -330 + index * gap
        colour = "a" if index == top else "w"
        ops.append(
            ("line", [(x, 330), (x, 330 - height)], 30 if index % 4 == 0 else 20, colour, 1.0)
        )
        ops.append(
            ("line", [(x - 24, 330 - height - 36), (x + 24, 330 - height - 36)], 8, "w", 1.0)
        )
    ops.append(("line", [(-370, 372), (420, 372)], 10, "w", 1.0))
    return ops


def _glyph_horizon(profile: DayProfile, rng: np.random.Generator) -> list[Op]:  # noqa: ARG001
    """Return a horizon arc: a sun at the hour of the busiest time, and still water."""
    peak = int(np.argmax(profile.bins)) / BINS
    ops: list[Op] = [("line", [(-420, 80), (520, 80)], 22, "w", 1.0)]
    ops.append(("line", _arc(0, 80, 360, 180, 360), 18, "w", 1.0))
    sun_x = -250 + 500 * peak
    sun_y = 80 - (max(0.0, 1 - (sun_x / 360) ** 2) ** 0.5) * 360
    ops.append(("disc", sun_x, sun_y, 40, "a"))
    for index, width in enumerate((420, 300, 190)):
        y = 230 + index * 70
        ops.append(("line", [(-width / 2, y), (width / 2, y)], 16 - 3 * index, "w", 1.0))
    return ops


def _glyph_record(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a record with grooves, a label and a tonearm."""
    grooves = 3 + int(rng.integers(0, 4))
    ops: list[Op] = [("ring", 0, 0, 400, 26, "w", 1.0)]
    ops.extend(("ring", 0, 0, float(r), 7, "w", 0.8) for r in np.linspace(330, 165, grooves))
    ops.append(("disc", 0, 0, 92, "a"))
    ops.append(("disc", 0, 0, 14, "w"))
    tilt = 8 + 14 * profile.intensity
    ops.append(("line", [(470, -320), (360, -tilt * 4), (250, 20 + tilt * 6)], 20, "w", 1.0))
    ops.append(("ring", 470, -320, 30, 14, "w", 1.0))
    return ops


def _glyph_drum(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a taiko seen from the front with a swirl of three commas on the skin, and ripples that grow with the day's intensity."""
    lugs = 8 + 2 * int(rng.integers(0, 3))
    ops: list[Op] = [("ring", 0, 0, 340, 26, "w", 1.0), ("ring", 0, 0, 280, 8, "w", 0.8)]
    for index in range(lugs):
        angle = 2.0 * np.pi * index / lugs
        ops.append(("disc", 310.0 * float(np.cos(angle)), 310.0 * float(np.sin(angle)), 11, "w"))
    turn = float(rng.uniform(0.0, 120.0))
    for index in range(3):
        start = turn + 120.0 * index
        cx, cy = 100.0 * float(np.cos(np.radians(start))), 100.0 * float(np.sin(np.radians(start)))
        ops.append(
            (
                "line",
                _arc(cx, cy, 62, start + 150, start + 400),
                26,
                "a" if index == 0 else "w",
                1.0,
            )
        )
    ops.extend(
        ("ring", 0, 0, 400 + 46 * index, 5, "w", 0.7 - 0.2 * index)
        for index in range(1 + round(2.0 * profile.intensity))
    )
    return ops


def _glyph_slices(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a waveform of bars as high as the day's activity, with a few slices torn sideways and a cut gap."""
    count = 17 + 2 * int(rng.integers(0, 3))
    levels = profile.parts(count)
    top = int(np.argmax(levels))
    torn = {int(i) for i in rng.choice(count, size=3, replace=False)}
    gap = int(rng.integers(3, count - 3))
    ops: list[Op] = []
    for index, level in enumerate(levels):
        if index == gap:
            continue
        x = -380.0 + index * (760.0 / (count - 1))
        half = 40.0 + 300.0 * (0.15 + 0.85 * level)
        shift = 0.0
        if index in torn:
            shift = float(rng.choice((-1.0, 1.0))) * (60.0 + 90.0 * float(rng.random()))
            ops.append(("line", [(x - 16, -shift), (x + 16, -shift)], 6, "w", 0.8))
        colour = "a" if index == top else "w"
        ops.append(("line", [(x, shift - half), (x, shift + half)], 18, colour, 1.0))
    ops.append(("line", [(-420, 0), (-398, 0)], 8, "w", 0.8))
    ops.append(("line", [(398, 0), (440, 0)], 8, "w", 0.8))
    return ops


GLYPHS = {
    "berlin_school": _glyph_sequencer,
    "ambient": _glyph_horizon,
    "lofi": _glyph_record,
    "ritual": _glyph_drum,
    "glitch": _glyph_slices,
}
# the hours that tell a night day from an evening day from a daytime one, for the cover of an album
NIGHT_FROM_HOUR = 22
NIGHT_UNTIL_HOUR = 6
EVENING_FROM_HOUR = 17


def album_style(facts: DayFacts | None, offered: Sequence[str]) -> str:
    """
    Return the style whose glyph is the cover of the album of a day, by the character of the day.

    A busy day is glitch, a quiet one ambient. Between the two the busiest hour decides: at night
    ritual, in the evening lofi, in the daytime berlin_school. A style that is not offered is
    replaced by the first offered one, which is also the answer for a day without facts.

    :param facts: What the day was like, None when it is not known.
    :param offered: The keys of the offered styles, in their order; there is at least one.
    """
    from .titles import BUSY_ABOVE, QUIET_BELOW  # noqa: PLC0415

    if facts is None or not facts.has_data:
        return offered[0]
    if facts.intensity >= BUSY_ABOVE:
        wanted = "glitch"
    elif facts.intensity < QUIET_BELOW:
        wanted = "ambient"
    elif facts.busiest_hour >= NIGHT_FROM_HOUR or facts.busiest_hour < NIGHT_UNTIL_HOUR:
        wanted = "ritual"
    elif facts.busiest_hour >= EVENING_FROM_HOUR:
        wanted = "lofi"
    else:
        wanted = "berlin_school"
    return wanted if wanted in offered else offered[0]


# how far each glyph reaches below its centre, in its own units
REACH = {
    "berlin_school": 380,
    "ambient": 330,
    "lofi": 420,
    "ritual": 450,
    "glitch": 420,
}
SUPERSAMPLE = 2


def _stops(colours: list[str], t: np.ndarray) -> np.ndarray:
    """Return the colour of a gradient of three stops at positions t between 0 and 1."""
    rgb = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in colours], dtype=np.float32)
    channels = [np.interp(t, [0.0, 0.55, 1.0], rgb[:3, k]) for k in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def _background(
    colours: list[str],
    angle: float,
    bloom: tuple[float, float],
    size: int,
    width: int | None = None,
) -> np.ndarray:
    """Return the gradient, the bloom and the darkening of the lower-left, as float RGB (width x size, square by default)."""
    x, y = np.meshgrid(
        (np.arange(width or size, dtype=np.float32) + 0.5) / (width or size),
        (np.arange(size, dtype=np.float32) + 0.5) / size,
    )
    theta = np.radians(angle)
    t = np.clip(0.5 + (x - 0.5) * np.cos(theta) * 0.9 + (y - 0.5) * np.sin(theta) * 0.9, 0.0, 1.0)
    canvas = _stops(colours, t)
    glow = np.clip(1.0 - np.hypot(x - bloom[0], y - bloom[1]) / 0.6, 0.0, 1.0) ** 2 * 0.34
    canvas += glow[..., None] * (255.0 - canvas)
    shade = np.clip((x * -1 + y * 1 + 0.2) / 1.2, 0.0, 1.0) ** 1.4 * 0.55
    return canvas * (1.0 - shade[..., None])


def _draw_ops(  # noqa: C901, PLR0913, PLR0917
    ops: list[Op],
    place: Callable[[float, float], tuple[float, float]],
    scale: float,
    turn: float,
    size: int,
    width: int | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Draw glyph operations into two masks (white and accent) at the size (height) of the layer; the width is the height unless given."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    masks = {
        "w": Image.new("L", (width or size, size), 0),
        "a": Image.new("L", (width or size, size), 0),
    }
    draws = {key: ImageDraw.Draw(image) for key, image in masks.items()}
    k = size / SIZE
    spin = np.radians(turn)

    def map_point(x: float, y: float) -> tuple[float, float]:
        px, py = x * np.cos(spin) - y * np.sin(spin), x * np.sin(spin) + y * np.cos(spin)
        gx, gy = place(px * scale, py * scale)
        return gx * k, gy * k

    def stroke(
        draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: float, alpha: float
    ) -> None:
        w = max(1, round(width * scale * k))
        fill = round(255 * alpha)
        draw.line(points, fill=fill, width=w, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fill)

    for op in ops:
        kind = op[0]
        if kind == "line":
            _, pts, width, colour, alpha = op
            stroke(draws[colour], [map_point(x, y) for x, y in pts], width, alpha)
        elif kind == "fill":
            _, pts, colour = op
            draws[colour].polygon([map_point(x, y) for x, y in pts], fill=255)
        elif kind in {"ring", "disc"}:
            if kind == "ring":
                _, cx, cy, radius, width, colour, alpha = op
                stroke(
                    draws[colour],
                    [map_point(*pt) for pt in _arc(cx, cy, radius, 0, 360, 96)],
                    width,
                    alpha,
                )
            else:
                _, cx, cy, radius, colour = op
                x, y = map_point(cx, cy)
                r = radius * scale * k
                draws[colour].ellipse((x - r, y - r, x + r, y + r), fill=255)
        elif kind == "dash":
            _, cx, cy, radius, width, colour = op
            for start in range(0, 360, 20):
                stroke(
                    draws[colour],
                    [map_point(*pt) for pt in _arc(cx, cy, radius, start, start + 11, 8)],
                    width,
                    1.0,
                )
        elif kind == "box":
            _, cx, cy, side, width, colour = op
            half, corner = side / 2, side * 0.22
            pts = (
                _arc(cx + half - corner, cy + half - corner, corner, 0, 90, 8)
                + _arc(cx - half + corner, cy + half - corner, corner, 90, 180, 8)
                + _arc(cx - half + corner, cy - half + corner, corner, 180, 270, 8)
                + _arc(cx + half - corner, cy - half + corner, corner, 270, 360, 8)
            )
            if width:
                stroke(draws[colour], [map_point(x, y) for x, y in [*pts, pts[0]]], width, 1.0)
            else:
                draws[colour].polygon([map_point(x, y) for x, y in pts], fill=255)
    return masks["w"], masks["a"]


def _wave_ops() -> tuple[list[Op], list[Op]]:
    """Return the dots-to-wave of the icon: five dots that turn into one wave, in sheet units."""
    x, y = 64, 84
    dots: list[Op] = [("disc", x + index * 34, y, 6, "w") for index in range(5)]
    wave: list[Op] = [
        (
            "line",
            [
                (x + 190 + step * 10, y + 14 * float(np.sin(step * 0.62)) * min(1.0, step / 5))
                for step in range(25)
            ],
            5,
            "w",
            0.85,
        )
    ]
    return dots, wave


def procedural_cover(  # noqa: PLR0913, PLR0917
    profile: DayProfile, style_key: str, root: int, mode: str, seed: str, title: str
) -> bytes:
    """
    Draw the cover of a day as a 1024 pixel JPEG in the look of the covers of Music Assistant.

    The gradient follows the style and the key and mode of the day; the glyph is the one of the
    style, shaped by the events of the day, large and cropped off the top-right; the dots-to-wave of
    the icon is a small detail; the title is set lower left by the overlay of the covers.

    :param profile: The day, see `profile_of`.
    :param style_key: The style, which picks the palette and the glyph.
    :param root: The key of the day, 0 to 11.
    :param mode: The mode of the day.
    :param seed: Text that gives the same cover for the same area, day and style.
    :param title: The title, such as "Kitchen, Tue 17 Nov".
    """
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    from .maart import draw_title_on  # noqa: PLC0415

    rng = np.random.default_rng(stable_hash("glyph", seed) % 2**32)
    colours = mood_colours(style_key, root, mode, seed)
    angle = float(rng.random()) * 80
    gx = 560 + float(rng.integers(0, 200))
    scale = 0.95 + 0.4 * float(rng.random())
    gy = min(250 + float(rng.integers(0, 220)), 700 - REACH.get(style_key, 400) * scale)
    turn = float(rng.random()) * 28 - 14
    bloom = (0.55 + 0.3 * float(rng.random()), 0.12 + 0.28 * float(rng.random()))
    ops = GLYPHS.get(style_key, _glyph_horizon)(profile, rng)
    canvas = _background(colours[:3], angle, bloom, SIZE)
    layer = SIZE * SUPERSAMPLE
    white, accent = _draw_ops(ops, lambda x, y: (gx + x, gy + y), scale, turn, layer)
    dots, wave = _wave_ops()
    small_white, _ = _draw_ops([*dots, *wave], lambda x, y: (x, y), 1.0, 0.0, layer)
    accent_rgb = np.array([int(colours[3][i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    box = Image.Resampling.BOX
    mask_w = (
        np.maximum(
            np.asarray(white.resize((SIZE, SIZE), box), dtype=np.float32),
            np.asarray(small_white.resize((SIZE, SIZE), box), dtype=np.float32),
        )[..., None]
        / 255.0
    )
    mask_a = np.asarray(accent.resize((SIZE, SIZE), box), dtype=np.float32)[..., None] / 255.0
    canvas = canvas * (1 - mask_w) + 255.0 * mask_w
    canvas = canvas * (1 - mask_a) + accent_rgb * mask_a
    # fine film grain, the same for the same seed
    canvas += rng.normal(0.0, 2.2, (SIZE, SIZE, 1)).astype(np.float32)
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    image = draw_title_on(image, title)
    out = BytesIO()
    image.save(out, "JPEG", optimize=True, quality=90)
    return out.getvalue()


# -- what the image engine is told ---------------------------------------------------------------


def describe_day(
    style_key: str, root: int, mode: str, intensity: float, kinds: Sequence[VoiceKind] = ()
) -> str:
    """
    Return the {context} of the prompt for a day, as sentences that each end with a space.

    It is put together from lists of words that are in this module. Nothing about the house gets
    in: not an id, a name, an area, a person, a time or a count. Only the style, the colour and the
    mood of the key, how lively the day was in three steps, a line about the light of a day and a few
    abstract motifs for the kinds of things that went on.

    :param style_key: The style.
    :param root: The key of the day, 0 to 11.
    :param mode: The mode of the day.
    :param intensity: How lively the day was, 0 to 1.
    :param kinds: The kinds of voice that played, which pick the motifs.
    """
    colour = ROOT_COLOURS[root % 12]
    mood = MODE_MOODS.get(mode, "calm")
    pace = (
        "quiet and still"
        if intensity < 0.34
        else "gently lively"
        if intensity < 0.67
        else "busy and bright"
    )
    wanted = set(kinds)
    motifs = [MOTIF_WORDS[kind] for kind in MOTIF_WORDS if kind in wanted][:3]
    text = (
        f"The cover sets the mood of a day of home life as generative music: "
        f"{STYLE_WORDS.get(style_key, STYLE_WORDS['ambient'])}. "
        f"Dominant colour {colour}, mood {mood}, the day feels {pace}. "
        "Light that moves from a bright morning through a warm evening to a soft night. "
    )
    if motifs:
        text += f"Abstract motifs: {', '.join(motifs)}. "
    return text


# -- keeping, limiting and making covers -----------------------------------------------------------


class RateLimiter:
    """At most `limit` events in any hour, counted by the clock that is passed in."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.monotonic) -> None:
        """
        Set the limit.

        :param limit: The most events an hour; 0 lets none through.
        :param clock: Returns seconds, for tests.
        """
        self.limit = limit
        self._clock = clock
        self._times: deque[float] = deque()

    def reset(self) -> None:
        """Forget the events of the last hour."""
        self._times.clear()

    def allow(self) -> bool:
        """Return True and count an event when one more is within the limit."""
        now = self._clock()
        while self._times and now - self._times[0] >= 3600.0:
            self._times.popleft()
        if len(self._times) >= self.limit:
            return False
        self._times.append(now)
        return True


def is_image(data: bytes) -> bool:
    """Return True for bytes that begin like a PNG, a JPEG or an SVG."""
    return len(data) > 100 and (
        data.startswith((PNG_MAGIC, JPEG_MAGIC))
        or data.lstrip()[:5].lower() in {b"<svg ", b"<?xml"}
    )


class CoverStore:
    """Generated covers on disk: those of pinned days stay, the others are cached with an LRU."""

    def __init__(
        self, permanent: str | Path, cache: str | Path, limit: int = MAX_STORE_BYTES
    ) -> None:
        """
        Use two folders.

        :param permanent: Where covers of pinned days live, in the storage of the server.
        :param cache: Where the others live, in the cache of the server.
        :param limit: The most bytes the cache folder may hold.
        """
        self.permanent = Path(permanent)
        self.cache = Path(cache)
        self.limit = limit

    @staticmethod
    def name(key: str) -> str:
        """Return the file name of a cover key."""
        return hashlib.sha256(key.encode()).hexdigest()[:32] + ".img"

    def get(self, key: str) -> bytes | None:
        """Return a stored cover, or None; a file that is not an image is thrown away."""
        for folder in (self.permanent, self.cache):
            path = folder / self.name(key)
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if not is_image(data):
                path.unlink(missing_ok=True)
                continue
            if folder is self.cache:
                path.touch()
            return data
        return None

    def put(self, key: str, data: bytes, *, pinned: bool) -> None:
        """
        Keep a cover; write it to a temporary file and rename it, so a crash leaves nothing broken.

        :param key: The cover key.
        :param data: The image.
        :param pinned: True for a day that is in the library: it stays for good.
        """
        folder = self.permanent if pinned else self.cache
        write_atomic(folder / self.name(key), data)
        if not pinned:
            self._evict()

    def clear(self) -> int:
        """
        Remove every cover in both folders, and the slots of the last good covers; nothing else.

        :returns: How many covers were removed.
        """
        removed = 0
        for folder in (self.permanent, self.cache):
            for path in [*folder.glob("*.img"), *folder.glob("*.img.part")]:
                path.unlink(missing_ok=True)
                removed += 1
        (self.cache / "slots.json").unlink(missing_ok=True)
        return removed

    def _evict(self) -> None:
        """Remove the least recently used covers of the cache until it fits."""
        files = sorted(
            (p for p in self.cache.glob("*.img") if p.is_file()), key=lambda p: p.stat().st_mtime
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.limit:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)


def write_atomic(path: Path, data: bytes) -> None:
    """
    Write a file by writing a temporary one and renaming it.

    :param path: The file.
    :param data: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(path)


class CoverMaker:
    """
    Decides what a cover is: nothing, drawn, or made by an engine in the background.

    A cover made by an engine is asked for the first time somebody wants it, and the drawn cover
    (or the engine's previous one, when only the title, style, prompt or engine changed) is shown
    until it is ready.
    """

    def __init__(
        self,
        store: CoverStore,
        limiter: RateLimiter,
        clock: Callable[[], float] = time.time,
        timeout: float = AI_TIMEOUT_S,
    ) -> None:
        """
        Set up the maker.

        :param store: Where AI covers are kept.
        :param limiter: How many AI covers an hour.
        :param clock: Returns seconds since the epoch, for tests.
        :param timeout: The longest an engine may take.
        """
        self.store = store
        self.limiter = limiter
        self._clock = clock
        self.timeout = timeout
        # key of a cover to (number of failures, when it may be tried again; inf for never)
        self._failures: dict[str, tuple[int, float]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._slots_path = store.cache / "slots.json"
        self._slots: dict[str, str] = self._load_slots()
        self.generated = 0

    def _load_slots(self) -> dict[str, str]:
        """Read which key was the last good cover of each slot, so that it stays across restarts."""
        try:
            data = json.loads(self._slots_path.read_text())
        except OSError, ValueError:
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _save_slots(self) -> None:
        with contextlib.suppress(OSError):
            write_atomic(self._slots_path, json.dumps(self._slots, sort_keys=True).encode())

    def backing_off(self, key: str) -> bool:
        """Return whether a cover failed recently, or too often to be tried again for now."""
        failure = self._failures.get(key)
        return failure is not None and self._clock() < failure[1]

    async def wait(self) -> None:
        """Wait for the covers that are being made (for tests and for a clean unload)."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def stop(self) -> None:
        """Cancel what is being made."""
        for task in self._tasks.values():
            task.cancel()

    async def clear(self) -> int:
        """
        Throw away every cover made so far, and what was remembered about them.

        Covers that are being made are cancelled, the failures and the slots of the last good covers
        are forgotten and the AI limiter starts again, so that the new covers are not held back by
        the old ones.

        :returns: How many covers were removed.
        """
        tasks = list(self._tasks.values())
        self.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._failures.clear()
        self._slots.clear()
        self.limiter.reset()
        return await asyncio.to_thread(self.store.clear)

    async def make(  # noqa: PLR0913
        self,
        mode: str,
        slot: str,
        key: str,
        draw: Callable[[], bytes],
        generate: Callable[[], Awaitable[bytes]] | None,
        *,
        pinned: bool = False,
    ) -> bytes | None:
        """
        Return the cover to show for a slot now, or None when covers are off.

        :param mode: `off`, `procedural` or `ai`.
        :param slot: One id for an area, a day and a style, whatever the settings.
        :param key: What shapes the cover (title, style, prompt, engine and the slot): a new key
            means a new cover, and the one of the old key is shown until it is ready.
        :param draw: Draws the procedural cover.
        :param generate: Makes the whole AI cover (asks the engine, draws the title); None when
            there is no engine.
        :param pinned: Whether the day is in the library.
        """
        if mode == COVER_OFF:
            return None
        if mode != COVER_AI or generate is None:
            return await self._drawn(slot, draw)
        kept = self.store.get(key)
        if kept is not None:
            return kept
        if key not in self._tasks and not self.backing_off(key) and self.limiter.allow():
            self._tasks[key] = asyncio.create_task(self._run(slot, key, generate, pinned=pinned))
        previous = self._slots.get(slot)
        old = self.store.get(previous) if previous and previous != key else None
        return old if old is not None else await self._drawn(slot, draw)

    async def _drawn(self, slot: str, draw: Callable[[], bytes]) -> bytes:
        """Return the drawn cover of a slot: from the cache when it was made before, else drawn."""
        key = f"drawn|{slot}"
        kept = self.store.get(key)
        if kept is not None:
            return kept
        image = await asyncio.to_thread(draw)
        if is_image(image):
            await asyncio.to_thread(self.store.put, key, image, pinned=False)
        return image

    async def _run(
        self, slot: str, key: str, generate: Callable[[], Awaitable[bytes]], *, pinned: bool
    ) -> None:
        """Make one cover, keep it, and remember the failure when it does not work."""
        try:
            image = await asyncio.wait_for(generate(), self.timeout)
            if not is_image(image):
                raise ValueError("the engine did not return an image")  # noqa: TRY301
            await asyncio.to_thread(self.store.put, key, image, pinned=pinned)
            self._slots[slot] = key
            self._save_slots()
            self._failures.pop(key, None)
            self.generated += 1
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - whatever the engine does, the drawn cover stays
            count = self._failures.get(key, (0, 0.0))[0] + 1
            wait = FAILURE_BACKOFF_S[count - 1] if count <= len(FAILURE_BACKOFF_S) else float("inf")
            self._failures[key] = (count, self._clock() + wait)
        finally:
            self._tasks.pop(key, None)
