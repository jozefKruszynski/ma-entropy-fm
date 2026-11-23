"""
Covers for the albums and the tracks, in the look of the covers of Music Assistant.

Music Assistant has AI generated covers for playlists (the Playlist Metadata provider): a rich
gradient of two or three colours chosen for the mood, a soft bloom and fine grain, a big white
line-art illustration cropped off the top and right edges, and the title in the calm lower-left
corner. This module draws the covers of Entropy FM in the same look, locally, as a 1024 pixel JPEG
(players need a raster image; Music Assistant passes an SVG through as it is): the gradient follows
the style and the key and mode of the day, the glyph is that of the style (a sequencer for Berlin
School, a horizon for Ambient, a record for Lo-fi, a drum for Ritual, torn slices of a waveform for
Cinematic Glitch) and is shaped by the events of the day, the dots-to-wave of the icon is a small
detail, and the title is set lower left. It is the same every time for the same area, day, style.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import IMAGE_SEED_MODULUS
from .timeline import DAY_SECONDS, VoiceKind, stable_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PIL import Image, ImageDraw, ImageFont

    from .timeline import Series
    from .titles import DayFacts

SIZE = 1024
BINS = 96
MAX_DOTS = 160
# a series moves the profile where its normalised value changes by more than this between two points
MOVE_THRESHOLD = 0.1
# the last fraction of the day that still falls in the last bin
LAST_FRACTION = 0.9999
# the share of the bins that is full on a day of full intensity
FULL_INTENSITY_SHARE = 0.25

# the colours of the styles: the dark end of the gradient, the middle, the bright end and an accent
PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "ambient": ((14, 40, 66), (46, 120, 140), (130, 210, 205), (255, 214, 140)),
    "berlin_school": ((34, 10, 52), (170, 52, 100), (255, 150, 84), (255, 214, 120)),
    "lofi": ((46, 26, 24), (140, 84, 62), (244, 176, 104), (214, 124, 150)),
    "ritual": ((26, 14, 22), (126, 40, 48), (226, 112, 66), (255, 206, 138)),
    "glitch": ((10, 12, 24), (52, 38, 112), (214, 62, 96), (120, 230, 255)),
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
        points = np.flatnonzero(moved > MOVE_THRESHOLD) + 1
        if len(points):
            kinds.add(item.kind)
        for index in points:
            fraction = float(np.clip(item.times[index] / DAY_SECONDS, 0.0, LAST_FRACTION))
            bins[int(fraction * BINS)] += float(moved[index - 1])
            found.append((fraction, float(moved[index - 1]), order.index(item.kind)))
    peak = float(bins.max())
    bins = bins / peak if peak > 0 else bins
    found.sort(key=lambda event: (-event[1], event[0], event[2]))
    kept = tuple(sorted(found[:MAX_DOTS]))
    intensity = float(np.clip(np.sum(bins) / (BINS * FULL_INTENSITY_SHARE), 0.0, 1.0))
    return DayProfile(bins, kept, frozenset(kinds), intensity)


# -- the drawn cover -----------------------------------------------------------------------------


# the key of the day moves the colours by up to this share, over the 12 keys (0 to 11)
KEY_SHIFT = 0.4
KEYS = 12
LAST_KEY = 11.0
# the colours of a mode are lighter (above 1) or darker (below 1); a mode that is not listed is 1
MODE_LIFT = {"aeolian": 0.88, "phrygian": 0.84, "dorian": 0.96, "mixolydian": 1.08, "lydian": 1.14}
DEFAULT_LIFT = 1.0
# the middle of the gradient is scaled by this much plus up to this much more, by the seed
WANDER_BASE = 0.88
WANDER_RANGE = 0.24
BYTE_MAX = 255


def _hex(colour: np.ndarray) -> str:
    """Return a colour as #rrggbb."""
    r, g, b = (int(np.clip(round(float(c)), 0, BYTE_MAX)) for c in colour)
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
    shift = KEY_SHIFT * (root % KEYS) / LAST_KEY
    mix = np.array([[1 - shift, shift, 0.0], [0.0, 1 - shift, shift], [shift, 0.0, 1 - shift]])
    lift = MODE_LIFT.get(mode, DEFAULT_LIFT)
    rng = np.random.default_rng(stable_hash("mood", seed) % IMAGE_SEED_MODULUS)
    wander = WANDER_BASE + WANDER_RANGE * float(rng.random())
    colours = [mix @ c * lift for c in (dark, middle * wander, bright, accent)]
    return [_hex(np.clip(c, 0, BYTE_MAX)) for c in colours]


# A glyph is a list of drawing operations in its own units, centred on (0, 0):
#   ("line", [(x, y), ...], width, colour, alpha)   a stroked line with round ends and joints
#   ("fill", [(x, y), ...], colour)                 a filled polygon
#   ("ring", x, y, radius, width, colour, alpha)    a stroked circle
#   ("disc", x, y, radius, colour)                  a filled circle
#   ("dash", x, y, radius, width, colour)           a dashed circle
#   ("box", x, y, size, width, colour)              a rounded square, stroked (width) or filled (0)
# The colour is "w" (white) or "a" (the accent of the style).
type Op = tuple[Any, ...]


# the kinds of drawing operation
OP_LINE = "line"
OP_FILL = "fill"
OP_RING = "ring"
OP_DISC = "disc"
OP_DASH = "dash"
OP_BOX = "box"
# the two colours of a glyph: white, and the accent of the style
WHITE = "w"
ACCENT = "a"
OPAQUE = 1.0
FULL_TURN = 360
QUARTER_TURN = 90
ARC_STEPS = 48

# the sequencer of Berlin School: bars on a rail, the busiest one in the accent
SEQ_MIN_STEPS = 9
SEQ_EXTRA_STEPS = 5
SEQ_SPAN = 720
SEQ_LEFT = -330
SEQ_BASE = 330
SEQ_MIN_HEIGHT = 90
SEQ_HEIGHT_RANGE = 330
SEQ_FLOOR = 0.25
SEQ_LEVEL_SHARE = 0.75
SEQ_ACCENT_EVERY = 4
SEQ_WIDE_WIDTH = 30
SEQ_WIDTH = 20
SEQ_CAP_HALF = 24
SEQ_CAP_RISE = 36
SEQ_CAP_WIDTH = 8
SEQ_RAIL = ((-370, 372), (420, 372))
SEQ_RAIL_WIDTH = 10

# the horizon of Ambient: a sun on an arc, and three strokes of still water
HORIZON_LINE = ((-420, 80), (520, 80))
HORIZON_WIDTH = 22
HORIZON_Y = 80
HORIZON_RADIUS = 360
HORIZON_ARC = (180, 360)
HORIZON_ARC_WIDTH = 18
SUN_LEFT = -250
SUN_TRAVEL = 500
SUN_RADIUS = 40
WATER_WIDTHS = (420, 300, 190)
WATER_TOP = 230
WATER_PITCH = 70
WATER_WIDTH = 16
WATER_TAPER = 3

# the record of Lo-fi: grooves, a label and a tonearm that lifts with the intensity of the day
RECORD_MIN_GROOVES = 3
RECORD_EXTRA_GROOVES = 4
RECORD_RADIUS = 400
RECORD_RIM = 26
GROOVE_OUTER = 330
GROOVE_INNER = 165
GROOVE_WIDTH = 7
GROOVE_ALPHA = 0.8
RECORD_LABEL_RADIUS = 92
RECORD_SPINDLE_RADIUS = 14
TONEARM_TILT = 8
TONEARM_TILT_RANGE = 14
TONEARM_PIVOT = (470, -320)
TONEARM_ELBOW_X = 360
TONEARM_ELBOW_FACTOR = 4
TONEARM_HEAD_X = 250
TONEARM_HEAD_Y = 20
TONEARM_HEAD_FACTOR = 6
TONEARM_WIDTH = 20
PIVOT_RADIUS = 30
PIVOT_WIDTH = 14

# the taiko of Ritual: a rim with lugs, a swirl of commas on the skin, ripples that grow with the day
DRUM_MIN_LUGS = 8
DRUM_LUG_STEP = 2
DRUM_LUG_CHOICES = 3
DRUM_RIM_RADIUS = 340
DRUM_RIM_WIDTH = 26
DRUM_SKIN_RADIUS = 280
DRUM_SKIN_WIDTH = 8
DRUM_SKIN_ALPHA = 0.8
DRUM_LUG_RING = 310.0
DRUM_LUG_RADIUS = 11
COMMAS = 3
COMMA_SPACING = 120.0
COMMA_CENTRE = 100.0
COMMA_RADIUS = 62
COMMA_FROM = 150
COMMA_TO = 400
COMMA_WIDTH = 26
RIPPLE_RADIUS = 400
RIPPLE_PITCH = 46
RIPPLE_WIDTH = 5
RIPPLE_ALPHA = 0.7
RIPPLE_FADE = 0.2
RIPPLE_MIN = 1
RIPPLE_RANGE = 2.0

# the slices of Cinematic Glitch: a waveform of bars with a few torn sideways and one cut out
SLICES_MIN = 17
SLICES_STEP = 2
SLICES_CHOICES = 3
SLICES_LEFT = -380.0
SLICES_SPAN = 760.0
SLICE_MIN_HALF = 40.0
SLICE_HALF_RANGE = 300.0
SLICE_FLOOR = 0.15
SLICE_LEVEL_SHARE = 0.85
SLICE_WIDTH = 18
TORN_COUNT = 3
CUT_MARGIN = 3
TEAR_MIN = 60.0
TEAR_RANGE = 90.0
TEAR_HALF = 16
TEAR_WIDTH = 6
TEAR_ALPHA = 0.8
SLICES_TICKS = (((-420, 0), (-398, 0)), ((398, 0), (440, 0)))
SLICES_TICK_WIDTH = 8
SLICES_TICK_ALPHA = 0.8


def _arc(
    cx: float, cy: float, radius: float, start: float, end: float, steps: int = ARC_STEPS
) -> list[tuple[float, float]]:
    """Return the points of an arc from `start` to `end` degrees (0 is to the right, y points down)."""
    angles = np.radians(np.linspace(start, end, steps))
    return [(cx + radius * float(np.cos(a)), cy + radius * float(np.sin(a))) for a in angles]


def _glyph_sequencer(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return stepped sequencer bars: as high as what happened in each stretch of the day."""
    steps = SEQ_MIN_STEPS + int(rng.integers(0, SEQ_EXTRA_STEPS))
    levels = profile.parts(steps)
    gap = SEQ_SPAN // steps
    ops: list[Op] = []
    top = int(np.argmax(levels))
    for index, level in enumerate(levels):
        height = SEQ_MIN_HEIGHT + SEQ_HEIGHT_RANGE * (SEQ_FLOOR + SEQ_LEVEL_SHARE * level)
        x = SEQ_LEFT + index * gap
        colour = ACCENT if index == top else WHITE
        width = SEQ_WIDE_WIDTH if index % SEQ_ACCENT_EVERY == 0 else SEQ_WIDTH
        ops.append((OP_LINE, [(x, SEQ_BASE), (x, SEQ_BASE - height)], width, colour, OPAQUE))
        cap = SEQ_BASE - height - SEQ_CAP_RISE
        ops.append(
            (
                OP_LINE,
                [(x - SEQ_CAP_HALF, cap), (x + SEQ_CAP_HALF, cap)],
                SEQ_CAP_WIDTH,
                WHITE,
                OPAQUE,
            )
        )
    ops.append((OP_LINE, list(SEQ_RAIL), SEQ_RAIL_WIDTH, WHITE, OPAQUE))
    return ops


def _glyph_horizon(profile: DayProfile, rng: np.random.Generator) -> list[Op]:  # noqa: ARG001
    """Return a horizon arc: a sun at the hour of the busiest time, and still water."""
    peak = int(np.argmax(profile.bins)) / BINS
    ops: list[Op] = [(OP_LINE, list(HORIZON_LINE), HORIZON_WIDTH, WHITE, OPAQUE)]
    ops.append(
        (
            OP_LINE,
            _arc(0, HORIZON_Y, HORIZON_RADIUS, *HORIZON_ARC),
            HORIZON_ARC_WIDTH,
            WHITE,
            OPAQUE,
        )
    )
    sun_x = SUN_LEFT + SUN_TRAVEL * peak
    sun_y = HORIZON_Y - (max(0.0, 1 - (sun_x / HORIZON_RADIUS) ** 2) ** 0.5) * HORIZON_RADIUS
    ops.append((OP_DISC, sun_x, sun_y, SUN_RADIUS, ACCENT))
    for index, width in enumerate(WATER_WIDTHS):
        y = WATER_TOP + index * WATER_PITCH
        ops.append(
            (
                OP_LINE,
                [(-width / 2, y), (width / 2, y)],
                WATER_WIDTH - WATER_TAPER * index,
                WHITE,
                OPAQUE,
            )
        )
    return ops


def _glyph_record(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a record with grooves, a label and a tonearm."""
    grooves = RECORD_MIN_GROOVES + int(rng.integers(0, RECORD_EXTRA_GROOVES))
    ops: list[Op] = [(OP_RING, 0, 0, RECORD_RADIUS, RECORD_RIM, WHITE, OPAQUE)]
    ops.extend(
        (OP_RING, 0, 0, float(r), GROOVE_WIDTH, WHITE, GROOVE_ALPHA)
        for r in np.linspace(GROOVE_OUTER, GROOVE_INNER, grooves)
    )
    ops.append((OP_DISC, 0, 0, RECORD_LABEL_RADIUS, ACCENT))
    ops.append((OP_DISC, 0, 0, RECORD_SPINDLE_RADIUS, WHITE))
    tilt = TONEARM_TILT + TONEARM_TILT_RANGE * profile.intensity
    arm = [
        TONEARM_PIVOT,
        (TONEARM_ELBOW_X, -tilt * TONEARM_ELBOW_FACTOR),
        (TONEARM_HEAD_X, TONEARM_HEAD_Y + tilt * TONEARM_HEAD_FACTOR),
    ]
    ops.append((OP_LINE, arm, TONEARM_WIDTH, WHITE, OPAQUE))
    ops.append((OP_RING, *TONEARM_PIVOT, PIVOT_RADIUS, PIVOT_WIDTH, WHITE, OPAQUE))
    return ops


def _glyph_drum(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a taiko seen from the front with a swirl of three commas on the skin, and ripples that grow with the day's intensity."""
    lugs = DRUM_MIN_LUGS + DRUM_LUG_STEP * int(rng.integers(0, DRUM_LUG_CHOICES))
    ops: list[Op] = [
        (OP_RING, 0, 0, DRUM_RIM_RADIUS, DRUM_RIM_WIDTH, WHITE, OPAQUE),
        (OP_RING, 0, 0, DRUM_SKIN_RADIUS, DRUM_SKIN_WIDTH, WHITE, DRUM_SKIN_ALPHA),
    ]
    for index in range(lugs):
        angle = 2.0 * np.pi * index / lugs
        ops.append(
            (
                OP_DISC,
                DRUM_LUG_RING * float(np.cos(angle)),
                DRUM_LUG_RING * float(np.sin(angle)),
                DRUM_LUG_RADIUS,
                WHITE,
            )
        )
    turn = float(rng.uniform(0.0, COMMA_SPACING))
    for index in range(COMMAS):
        start = turn + COMMA_SPACING * index
        cx = COMMA_CENTRE * float(np.cos(np.radians(start)))
        cy = COMMA_CENTRE * float(np.sin(np.radians(start)))
        ops.append(
            (
                OP_LINE,
                _arc(cx, cy, COMMA_RADIUS, start + COMMA_FROM, start + COMMA_TO),
                COMMA_WIDTH,
                ACCENT if index == 0 else WHITE,
                OPAQUE,
            )
        )
    ops.extend(
        (
            OP_RING,
            0,
            0,
            RIPPLE_RADIUS + RIPPLE_PITCH * index,
            RIPPLE_WIDTH,
            WHITE,
            RIPPLE_ALPHA - RIPPLE_FADE * index,
        )
        for index in range(RIPPLE_MIN + round(RIPPLE_RANGE * profile.intensity))
    )
    return ops


def _glyph_slices(profile: DayProfile, rng: np.random.Generator) -> list[Op]:
    """Return a waveform of bars as high as the day's activity, with a few slices torn sideways and a cut gap."""
    count = SLICES_MIN + SLICES_STEP * int(rng.integers(0, SLICES_CHOICES))
    levels = profile.parts(count)
    top = int(np.argmax(levels))
    torn = {int(i) for i in rng.choice(count, size=TORN_COUNT, replace=False)}
    gap = int(rng.integers(CUT_MARGIN, count - CUT_MARGIN))
    ops: list[Op] = []
    for index, level in enumerate(levels):
        if index == gap:
            continue
        x = SLICES_LEFT + index * (SLICES_SPAN / (count - 1))
        half = SLICE_MIN_HALF + SLICE_HALF_RANGE * (SLICE_FLOOR + SLICE_LEVEL_SHARE * level)
        shift = 0.0
        if index in torn:
            shift = float(rng.choice((-1.0, 1.0))) * (TEAR_MIN + TEAR_RANGE * float(rng.random()))
            ops.append(
                (
                    OP_LINE,
                    [(x - TEAR_HALF, -shift), (x + TEAR_HALF, -shift)],
                    TEAR_WIDTH,
                    WHITE,
                    TEAR_ALPHA,
                )
            )
        colour = ACCENT if index == top else WHITE
        ops.append((OP_LINE, [(x, shift - half), (x, shift + half)], SLICE_WIDTH, colour, OPAQUE))
    ops.extend(
        (OP_LINE, list(tick), SLICES_TICK_WIDTH, WHITE, SLICES_TICK_ALPHA) for tick in SLICES_TICKS
    )
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
# a style that has no listed reach, and a glyph that is not listed, reach and look like the horizon
DEFAULT_REACH = 400
SUPERSAMPLE = 2
# where the red, green and blue pairs of a #rrggbb colour start, and the base of its numbers
HEX_FIELDS = (1, 3, 5)
HEX_BASE = 16
# the gradient has three stops, the middle one at this position between 0 and 1
GRADIENT_STOPS = 3
STOP_POSITIONS = [0.0, 0.55, 1.0]
CHANNELS = 3
# the gradient runs along an angle, and is this wide of the cover
GRADIENT_CENTRE = 0.5
GRADIENT_SPREAD = 0.9
# the bloom: its radius in cover widths, and how much of the way to white it goes at its centre
BLOOM_RADIUS = 0.6
BLOOM_STRENGTH = 0.34
# the darkening of the lower left: an offset and a span of the diagonal, a curve and how dark it gets
SHADE_DIAGONAL_OFFSET = 0.2
SHADE_DIAGONAL_SPAN = 1.2
SHADE_CURVE = 1.4
SHADE_DARKNESS = 0.55
# the cell of the corner of a rounded square, as a share of its side, and the steps of its arcs
BOX_CORNER = 0.22
BOX_ARC_STEPS = 8
# a ring is drawn as a polygon of this many steps; a dashed circle has a dash every pitch, this long
RING_STEPS = 96
DASH_PITCH = 20
DASH_LENGTH = 11
DASH_STEPS = 8
# a stroke is at least this wide in pixels
MIN_STROKE = 1
# the dots-to-wave of the icon, in sheet units: where it starts, the dots, and the wave they turn into
WAVE_ORIGIN = (64, 84)
WAVE_DOTS = 5
WAVE_DOT_PITCH = 34
WAVE_DOT_RADIUS = 6
WAVE_START = 190
WAVE_STEPS = 25
WAVE_STEP = 10
WAVE_AMPLITUDE = 14
WAVE_FREQUENCY = 0.62
WAVE_RAMP = 5
WAVE_WIDTH = 5
WAVE_ALPHA = 0.85
# where the glyph is put on the cover, in cover units, with a random part on top of each
GLYPH_ANGLE_RANGE = 80
GLYPH_X = 560
GLYPH_X_RANGE = 200
GLYPH_Y = 250
GLYPH_Y_RANGE = 220
GLYPH_BOTTOM = 700
GLYPH_SCALE = 0.95
GLYPH_SCALE_RANGE = 0.4
GLYPH_TURN_RANGE = 28
GLYPH_TURN_HALF = 14
# the bloom sits at a random place around this one, in cover widths
BLOOM_X = 0.55
BLOOM_X_RANGE = 0.3
BLOOM_Y = 0.12
BLOOM_Y_RANGE = 0.28
# the standard deviation of the film grain
GRAIN = 2.2
JPEG_QUALITY = 90
GLYPH_COLOURS = 3
ACCENT_COLOUR = 3


def _stops(colours: list[str], t: np.ndarray) -> np.ndarray:
    """Return the colour of a gradient of three stops at positions t between 0 and 1."""
    rgb = np.array(
        [[int(c[i : i + 2], HEX_BASE) for i in HEX_FIELDS] for c in colours], dtype=np.float32
    )
    channels = [np.interp(t, STOP_POSITIONS, rgb[:GRADIENT_STOPS, k]) for k in range(CHANNELS)]
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
    t = np.clip(
        GRADIENT_CENTRE
        + (x - GRADIENT_CENTRE) * np.cos(theta) * GRADIENT_SPREAD
        + (y - GRADIENT_CENTRE) * np.sin(theta) * GRADIENT_SPREAD,
        0.0,
        1.0,
    )
    canvas = _stops(colours, t)
    glow = (
        np.clip(1.0 - np.hypot(x - bloom[0], y - bloom[1]) / BLOOM_RADIUS, 0.0, 1.0) ** 2
        * BLOOM_STRENGTH
    )
    canvas += glow[..., None] * (BYTE_MAX - canvas)
    shade = (
        np.clip((x * -1 + y * 1 + SHADE_DIAGONAL_OFFSET) / SHADE_DIAGONAL_SPAN, 0.0, 1.0)
        ** SHADE_CURVE
        * SHADE_DARKNESS
    )
    return canvas * (1.0 - shade[..., None])


def _draw_ops(
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
        WHITE: Image.new("L", (width or size, size), 0),
        ACCENT: Image.new("L", (width or size, size), 0),
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
        w = max(MIN_STROKE, round(width * scale * k))
        fill = round(BYTE_MAX * alpha)
        draw.line(points, fill=fill, width=w, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fill)

    for op in ops:
        kind = op[0]
        if kind == OP_LINE:
            _, pts, width, colour, alpha = op
            stroke(draws[colour], [map_point(x, y) for x, y in pts], width, alpha)
        elif kind == OP_FILL:
            _, pts, colour = op
            draws[colour].polygon([map_point(x, y) for x, y in pts], fill=BYTE_MAX)
        elif kind in {OP_RING, OP_DISC}:
            if kind == OP_RING:
                _, cx, cy, radius, width, colour, alpha = op
                stroke(
                    draws[colour],
                    [map_point(*pt) for pt in _arc(cx, cy, radius, 0, FULL_TURN, RING_STEPS)],
                    width,
                    alpha,
                )
            else:
                _, cx, cy, radius, colour = op
                x, y = map_point(cx, cy)
                r = radius * scale * k
                draws[colour].ellipse((x - r, y - r, x + r, y + r), fill=BYTE_MAX)
        elif kind == OP_DASH:
            _, cx, cy, radius, width, colour = op
            for start in range(0, FULL_TURN, DASH_PITCH):
                stroke(
                    draws[colour],
                    [
                        map_point(*pt)
                        for pt in _arc(cx, cy, radius, start, start + DASH_LENGTH, DASH_STEPS)
                    ],
                    width,
                    OPAQUE,
                )
        elif kind == OP_BOX:
            _, cx, cy, side, width, colour = op
            half, corner = side / 2, side * BOX_CORNER
            q = QUARTER_TURN
            pts = (
                _arc(cx + half - corner, cy + half - corner, corner, 0, q, BOX_ARC_STEPS)
                + _arc(cx - half + corner, cy + half - corner, corner, q, 2 * q, BOX_ARC_STEPS)
                + _arc(cx - half + corner, cy - half + corner, corner, 2 * q, 3 * q, BOX_ARC_STEPS)
                + _arc(cx + half - corner, cy - half + corner, corner, 3 * q, 4 * q, BOX_ARC_STEPS)
            )
            if width:
                stroke(draws[colour], [map_point(x, y) for x, y in [*pts, pts[0]]], width, OPAQUE)
            else:
                draws[colour].polygon([map_point(x, y) for x, y in pts], fill=BYTE_MAX)
    return masks["w"], masks["a"]


def _wave_ops() -> tuple[list[Op], list[Op]]:
    """Return the dots-to-wave of the icon: five dots that turn into one wave, in sheet units."""
    x, y = WAVE_ORIGIN
    dots: list[Op] = [
        (OP_DISC, x + index * WAVE_DOT_PITCH, y, WAVE_DOT_RADIUS, WHITE)
        for index in range(WAVE_DOTS)
    ]
    wave: list[Op] = [
        (
            OP_LINE,
            [
                (
                    x + WAVE_START + step * WAVE_STEP,
                    y
                    + WAVE_AMPLITUDE
                    * float(np.sin(step * WAVE_FREQUENCY))
                    * min(1.0, step / WAVE_RAMP),
                )
                for step in range(WAVE_STEPS)
            ],
            WAVE_WIDTH,
            WHITE,
            WAVE_ALPHA,
        )
    ]
    return dots, wave


def procedural_cover(
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

    rng = np.random.default_rng(stable_hash("glyph", seed) % IMAGE_SEED_MODULUS)
    colours = mood_colours(style_key, root, mode, seed)
    angle = float(rng.random()) * GLYPH_ANGLE_RANGE
    gx = GLYPH_X + float(rng.integers(0, GLYPH_X_RANGE))
    scale = GLYPH_SCALE + GLYPH_SCALE_RANGE * float(rng.random())
    gy = min(
        GLYPH_Y + float(rng.integers(0, GLYPH_Y_RANGE)),
        GLYPH_BOTTOM - REACH.get(style_key, DEFAULT_REACH) * scale,
    )
    turn = float(rng.random()) * GLYPH_TURN_RANGE - GLYPH_TURN_HALF
    bloom = (
        BLOOM_X + BLOOM_X_RANGE * float(rng.random()),
        BLOOM_Y + BLOOM_Y_RANGE * float(rng.random()),
    )
    ops = GLYPHS.get(style_key, _glyph_horizon)(profile, rng)
    canvas = _background(colours[:GLYPH_COLOURS], angle, bloom, SIZE)
    layer = SIZE * SUPERSAMPLE
    white, accent = _draw_ops(ops, lambda x, y: (gx + x, gy + y), scale, turn, layer)
    dots, wave = _wave_ops()
    small_white, _ = _draw_ops([*dots, *wave], lambda x, y: (x, y), 1.0, 0.0, layer)
    accent_rgb = np.array(
        [int(colours[ACCENT_COLOUR][i : i + 2], HEX_BASE) for i in HEX_FIELDS], dtype=np.float32
    )
    box = Image.Resampling.BOX
    mask_w = (
        np.maximum(
            np.asarray(white.resize((SIZE, SIZE), box), dtype=np.float32),
            np.asarray(small_white.resize((SIZE, SIZE), box), dtype=np.float32),
        )[..., None]
        / BYTE_MAX
    )
    mask_a = np.asarray(accent.resize((SIZE, SIZE), box), dtype=np.float32)[..., None] / BYTE_MAX
    canvas = canvas * (1 - mask_w) + BYTE_MAX * mask_w
    canvas = canvas * (1 - mask_a) + accent_rgb * mask_a
    # fine film grain, the same for the same seed
    canvas += rng.normal(0.0, GRAIN, (SIZE, SIZE, 1)).astype(np.float32)
    image = Image.fromarray(np.clip(canvas, 0, BYTE_MAX).astype(np.uint8), "RGB")
    image = draw_title_on(image, title)
    out = BytesIO()
    image.save(out, "JPEG", optimize=True, quality=JPEG_QUALITY)
    return out.getvalue()


# -- the title ---------------------------------------------------------------------------------------

# the title is drawn as the covers of Music Assistant do (Apache-2.0, the Music Assistant authors)
TITLE_MARGIN = 0.07
TITLE_FONT_SIZE = 0.065
TITLE_LINE_HEIGHT = 1.08
TITLE_SHRINK = 0.9
MAX_TITLE_LINES = 3
MIN_TITLE_SCALE = 0.4
SHADE_ALPHA = 170
SHADE_FALLOFF = 1.5
SHADOW_OFFSET = 3
SHADOW_COLOUR = (0, 0, 0, 120)
BLACK = (0, 0, 0)
TRANSPARENT = (0, 0, 0, 0)
# the shade covers this part of the height, from the bottom
SHADE_HEIGHT_DIVISOR = 2
TITLE_COLOUR = (255, 255, 255, 255)
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Windows/Fonts/arialbd.ttf",
)


def draw_title_on(canvas: Image.Image, title: str) -> Image.Image:
    """
    Draw the title in the lower-left corner of a cover, over a shade that fades out upwards.

    :param canvas: A square RGB image.
    :param title: The title.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    size = canvas.width
    margin = int(size * TITLE_MARGIN)
    font, font_size, lines = _fit_title(title, int(size * TITLE_FONT_SIZE), size - 2 * margin)
    line_height = int(font_size * TITLE_LINE_HEIGHT)
    top = size - margin - line_height * len(lines)
    shade = Image.new("RGBA", canvas.size, TRANSPARENT)
    draw_shade = ImageDraw.Draw(shade)
    height = size // SHADE_HEIGHT_DIVISOR
    for offset in range(height):
        alpha = int(SHADE_ALPHA * (1 - offset / height) ** SHADE_FALLOFF)
        draw_shade.line([(0, size - 1 - offset), (size, size - 1 - offset)], fill=(*BLACK, alpha))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shade)
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        y = top + index * line_height
        draw.text((margin + SHADOW_OFFSET, y + SHADOW_OFFSET), line, fill=SHADOW_COLOUR, font=font)
        draw.text((margin, y), line, fill=TITLE_COLOUR, font=font)
    return canvas.convert("RGB")


def _fit_title(
    title: str, font_size: int, max_width: int
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, list[str]]:
    """
    Return the font, its size and the lines that fit the whole title within the width.

    The font shrinks until the title takes at most a few lines; nothing of it is ever dropped.

    :param title: The title to fit.
    :param font_size: The size to start from.
    :param max_width: The width each line has to fit in, in pixels.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    min_size = max(1, int(font_size * MIN_TITLE_SCALE))
    while True:
        font = _font(font_size)
        lines = _wrap_title(title, font, max_width, measure)
        if len(lines) <= MAX_TITLE_LINES or font_size <= min_size:
            return font, font_size, lines
        font_size = max(min_size, int(font_size * TITLE_SHRINK))


def _wrap_title(
    title: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    measure: ImageDraw.ImageDraw,
) -> list[str]:
    """Return the title wrapped into lines that fit the width, breaking only too long words."""
    lines: list[str] = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}" if current else word
        if measure.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        # a word wider than the line on its own is split over as many lines as it needs
        current = ""
        for char in word:
            if current and measure.textlength(current + char, font=font) > max_width:
                lines.append(current)
                current = ""
            current += char
    if current:
        lines.append(current)
    return lines or [title]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a bold TrueType font at the given size, falling back to Pillow's built-in."""
    from PIL import ImageFont  # noqa: PLC0415

    for path in FONT_PATHS:
        if Path(path).exists():
            with contextlib.suppress(OSError):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)
