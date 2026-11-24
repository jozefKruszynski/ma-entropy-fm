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
detail that gauges the energy of the day, a thin strip along the foot shows the day round the clock,
and the title is set lower left. The energy of the day (the one the music was made with) also moves the
colours and every glyph, so that a cover tells at a glance what kind of mix it is. It is the same every
time for the same area, day, style and energy.

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
# the energy of a day of no more and no less energy than usual
MIDDLE = 0.5

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


@dataclass(frozen=True)
class CoverMood:
    """How busy the day was against the room's own days, as the music was made with it."""

    # 0 to 1, the energy of the character of the day
    energy: float = MIDDLE


def profile_of(series: Sequence[Series], length: float = DAY_SECONDS) -> DayProfile:
    """
    Describe a day from its series: the activity round the clock, its events and what kinds played.

    :param series: Every series of the area on the day.
    :param length: The length of the day in seconds, which is not 24 hours on a change of daylight saving time.
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
            fraction = float(np.clip(item.times[index] / length, 0.0, LAST_FRACTION))
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
# the energies at which the looks of a quiet, a medium and a busy day are pinned (beyond them a look holds)
ENERGY_QUIET = 0.1
ENERGY_BUSY = 0.9
# energy lights the bars: at 0.5 all are lit at full brightness; below it the quiet bars (those under a
# threshold that rises up to this much as the energy falls) are a ghost, and every lit bar is dimmer
SEQ_UNLIT_RANGE = 1.2
SEQ_UNLIT_ALPHA = 0.3
SEQ_DIM_FLOOR = 0.55
SEQ_DIM_SLOPE = 0.9

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
WATER_FIRST_WIDTH = 420
WATER_NARROWING = 115
WATER_MIN_WIDTH = 90
WATER_TOP = 230
# the lines of water share the stretch that three of them had; the more lines, the closer they sit
WATER_SPAN = 140
WATER_LINES_AT_REST = 3
WATER_PITCH = 70
WATER_WIDTH = 16
WATER_TAPER = 3
WATER_MIN_WIDTH_STROKE = 8
# the water has this many lines at no energy, and this many more at full energy (three at half)
WATER_MIN_LINES = 1
WATER_EXTRA_LINES = 4
# a halo of rings round the sun above half energy: this many at full energy, a step out each
HALO_RINGS = 3
HALO_PITCH = 26
HALO_WIDTH = 5
HALO_ALPHA = 0.7
HALO_FADE = 0.25

# the record of Lo-fi: grooves, a label and a tonearm that lifts with the intensity of the day
RECORD_MIN_GROOVES = 2
RECORD_EXTRA_GROOVES = 2
# the energy adds grooves, so that a busy record is dense and a quiet one has a few
RECORD_ENERGY_GROOVES = 8
RECORD_RADIUS = 400
RECORD_RIM = 26
GROOVE_OUTER = 330
GROOVE_INNER = 165
GROOVE_WIDTH = 7
GROOVE_ALPHA = 0.8
RECORD_LABEL_RADIUS = 92
# the label is small on a quiet day and wide on a busy one, at the energies of quiet, medium and busy
RECORD_LABEL_RADII = (62.0, 92.0, 122.0)
RECORD_SPINDLE_RADIUS = 14
TONEARM_TILT = 8
TONEARM_TILT_RANGE = 14
# where the head of the tonearm is (x, y) for a quiet day (lifted off the record, out at the edge), a medium
# one and a busy one (deep in the grooves)
TONEARM_HEADS_X = (445.0, 290.0, 215.0)
TONEARM_HEADS_Y = (-170.0, 20.0, 70.0)
# blur arcs round the rim of a spinning record: up to this many, from this energy, this long and far out
SPIN_ARCS = 7
SPIN_FROM = 0.55
SPIN_ARC_LENGTH = 38
SPIN_ARC_RADIUS = 440
SPIN_ARC_PITCH = 24
SPIN_ARC_WIDTH = 9
SPIN_ARC_ALPHA = 0.65
SPIN_RINGS = 3
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
RIPPLE_MIN = 0
RIPPLE_RANGE = 3.0
# the rim has this many lugs more or fewer than the seed says: fewer at rest, more when busy
DRUM_LUG_SWING = 4

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
SLICE_WIDTH = 14
CUT_MARGIN = 3
TEAR_MIN = 60.0
TEAR_RANGE = 90.0
TEAR_HALF = 16
TEAR_WIDTH = 6
TEAR_ALPHA = 0.8
# energy cuts more slices (this many more at full energy, fewer at none), tears more of them, and further
SLICES_SWING = 9
TORN_MIN = 0
TORN_RANGE = 6
TEAR_AT_REST = 0.3
TEAR_STRENGTH = 1.3
SLICES_TICKS = (((-420, 0), (-398, 0)), ((398, 0), (440, 0)))
SLICES_TICK_WIDTH = 8
SLICES_TICK_ALPHA = 0.8


def _by_energy(energy: float, looks: tuple[float, float, float]) -> float:
    """
    Return a value that follows the energy of the day through three pinned looks.

    :param energy: The energy of the day, 0 to 1.
    :param looks: The value of a quiet day (energy 0.1), of a medium one (0.5) and of a busy one (0.9).
    """
    return float(np.interp(energy, (ENERGY_QUIET, MIDDLE, ENERGY_BUSY), looks))


def _arc(
    cx: float, cy: float, radius: float, start: float, end: float, steps: int = ARC_STEPS
) -> list[tuple[float, float]]:
    """Return the points of an arc from `start` to `end` degrees (0 is to the right, y points down)."""
    angles = np.radians(np.linspace(start, end, steps))
    return [(cx + radius * float(np.cos(a)), cy + radius * float(np.sin(a))) for a in angles]


def _glyph_sequencer(profile: DayProfile, rng: np.random.Generator, energy: float) -> list[Op]:
    """Return stepped sequencer bars: as high as what happened in each stretch of the day, and more of them lit and brighter on a busy day."""
    steps = SEQ_MIN_STEPS + int(rng.integers(0, SEQ_EXTRA_STEPS))
    levels = profile.parts(steps)
    gap = SEQ_SPAN // steps
    ops: list[Op] = []
    top = int(np.argmax(levels))
    unlit_below = max(0.0, MIDDLE - energy) * SEQ_UNLIT_RANGE
    brightness = min(OPAQUE, SEQ_DIM_FLOOR + SEQ_DIM_SLOPE * energy)
    for index, level in enumerate(levels):
        height = SEQ_MIN_HEIGHT + SEQ_HEIGHT_RANGE * (SEQ_FLOOR + SEQ_LEVEL_SHARE * level)
        x = SEQ_LEFT + index * gap
        colour = ACCENT if index == top else WHITE
        width = SEQ_WIDE_WIDTH if index % SEQ_ACCENT_EVERY == 0 else SEQ_WIDTH
        alpha = SEQ_UNLIT_ALPHA if level < unlit_below and index != top else brightness
        ops.append((OP_LINE, [(x, SEQ_BASE), (x, SEQ_BASE - height)], width, colour, alpha))
        cap = SEQ_BASE - height - SEQ_CAP_RISE
        ops.append(
            (
                OP_LINE,
                [(x - SEQ_CAP_HALF, cap), (x + SEQ_CAP_HALF, cap)],
                SEQ_CAP_WIDTH,
                WHITE,
                alpha,
            )
        )
    ops.append((OP_LINE, list(SEQ_RAIL), SEQ_RAIL_WIDTH, WHITE, OPAQUE))
    return ops


def _glyph_horizon(profile: DayProfile, rng: np.random.Generator, energy: float) -> list[Op]:  # noqa: ARG001
    """Return a horizon arc: a sun at the hour of the busiest time, and water that has more lines and a halo round the sun when the day was busy."""
    peak = _busiest(profile) / BINS
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
    ops.extend(
        (
            OP_RING,
            sun_x,
            sun_y,
            SUN_RADIUS + HALO_PITCH * (ring + 1),
            HALO_WIDTH,
            WHITE,
            HALO_ALPHA - HALO_FADE * ring,
        )
        for ring in range(round(HALO_RINGS * max(0.0, 2 * energy - 1)))
    )
    lines = WATER_MIN_LINES + round(WATER_EXTRA_LINES * energy)
    pitch = WATER_SPAN / max(1, lines - 1) if lines > WATER_LINES_AT_REST else WATER_PITCH
    for index in range(lines):
        width = max(WATER_MIN_WIDTH, WATER_FIRST_WIDTH - WATER_NARROWING * index)
        y = WATER_TOP + index * pitch
        ops.append(
            (
                OP_LINE,
                [(-width / 2, y), (width / 2, y)],
                max(WATER_MIN_WIDTH_STROKE, WATER_WIDTH - WATER_TAPER * index),
                WHITE,
                OPAQUE,
            )
        )
    return ops


def _glyph_record(profile: DayProfile, rng: np.random.Generator, energy: float) -> list[Op]:  # noqa: ARG001
    """Return a record that is sparse and lifted off by the tonearm on a quiet day, and dense, with a wide label, blur arcs of its spin and the tonearm deep in the grooves on a busy one."""
    grooves = (
        RECORD_MIN_GROOVES
        + int(rng.integers(0, RECORD_EXTRA_GROOVES))
        + round(RECORD_ENERGY_GROOVES * energy)
    )
    ops: list[Op] = [(OP_RING, 0, 0, RECORD_RADIUS, RECORD_RIM, WHITE, OPAQUE)]
    ops.extend(
        (OP_RING, 0, 0, float(r), GROOVE_WIDTH, WHITE, GROOVE_ALPHA)
        for r in np.linspace(GROOVE_OUTER, GROOVE_INNER, grooves)
    )
    label = _by_energy(energy, RECORD_LABEL_RADII)
    ops.append((OP_DISC, 0, 0, label, ACCENT))
    ops.append((OP_DISC, 0, 0, RECORD_SPINDLE_RADIUS, WHITE))
    arcs = round(SPIN_ARCS * max(0.0, (energy - SPIN_FROM) / (1 - SPIN_FROM)))
    for index in range(arcs):
        start = index * FULL_TURN / arcs + float(rng.uniform(0.0, FULL_TURN / arcs / 2))
        radius = SPIN_ARC_RADIUS + SPIN_ARC_PITCH * (index % SPIN_RINGS)
        ops.append(
            (
                OP_LINE,
                _arc(0, 0, radius, start, start + SPIN_ARC_LENGTH),
                SPIN_ARC_WIDTH,
                WHITE,
                SPIN_ARC_ALPHA,
            )
        )
    tilt = TONEARM_TILT + TONEARM_TILT_RANGE * energy
    head = (_by_energy(energy, TONEARM_HEADS_X), _by_energy(energy, TONEARM_HEADS_Y))
    arm = [TONEARM_PIVOT, (TONEARM_ELBOW_X, -tilt * TONEARM_ELBOW_FACTOR), head]
    ops.append((OP_LINE, arm, TONEARM_WIDTH, WHITE, OPAQUE))
    ops.append((OP_RING, *TONEARM_PIVOT, PIVOT_RADIUS, PIVOT_WIDTH, WHITE, OPAQUE))
    return ops


def _glyph_drum(profile: DayProfile, rng: np.random.Generator, energy: float) -> list[Op]:  # noqa: ARG001
    """Return a taiko seen from the front with a swirl of three commas on the skin, and ripples that grow with the energy of the day."""
    lugs = DRUM_MIN_LUGS + DRUM_LUG_STEP * int(rng.integers(0, DRUM_LUG_CHOICES))
    lugs += round(DRUM_LUG_SWING * (2 * energy - 1))
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
        for index in range(RIPPLE_MIN + round(RIPPLE_RANGE * energy))
    )
    return ops


def _glyph_slices(profile: DayProfile, rng: np.random.Generator, energy: float) -> list[Op]:
    """Return a waveform of bars as high as the day's activity, with a few slices torn sideways and a cut gap; a busy day has more slices and more torn."""
    count = SLICES_MIN + SLICES_STEP * int(rng.integers(0, SLICES_CHOICES))
    count += round(SLICES_SWING * (2 * energy - 1))
    levels = profile.parts(count)
    top = int(np.argmax(levels))
    torn = {
        int(i) for i in rng.choice(count, size=TORN_MIN + round(TORN_RANGE * energy), replace=False)
    }
    tear = TEAR_AT_REST + TEAR_STRENGTH * energy
    gap = int(rng.integers(CUT_MARGIN, count - CUT_MARGIN))
    ops: list[Op] = []
    for index, level in enumerate(levels):
        if index == gap:
            continue
        x = SLICES_LEFT + index * (SLICES_SPAN / (count - 1))
        half = SLICE_MIN_HALF + SLICE_HALF_RANGE * (SLICE_FLOOR + SLICE_LEVEL_SHARE * level)
        shift = 0.0
        if index in torn:
            shift = (
                float(rng.choice((-1.0, 1.0)))
                * tear
                * (TEAR_MIN + TEAR_RANGE * float(rng.random()))
            )
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
# by energy (quiet, medium, busy): the bloom is small and faint on a quiet day, wide and strong on a busy one
BLOOM_RADII = (0.42, BLOOM_RADIUS, 1.0)
BLOOM_STRENGTHS = (0.1, BLOOM_STRENGTH, 0.62)
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
WAVE_DOT_RADIUS = 8
WAVE_START = 190
WAVE_STEPS = 25
WAVE_STEP = 10
# the wave is the gauge of the energy of the day: a flat slow line at 0, a big tight wave at 1
WAVE_AMPLITUDE_MIN = 1.0
WAVE_AMPLITUDE_RANGE = 40.0
WAVE_FREQUENCY_MIN = 0.18
WAVE_FREQUENCY_RANGE = 0.95
WAVE_RAMP = 5
WAVE_WIDTH = 8
WAVE_ALPHA = 0.95
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
    bloom_strength: float = BLOOM_STRENGTH,
    bloom_radius: float = BLOOM_RADIUS,
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
        np.clip(1.0 - np.hypot(x - bloom[0], y - bloom[1]) / bloom_radius, 0.0, 1.0) ** 2
        * bloom_strength
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
    weight: float = 1.0,
    fade: float = OPAQUE,
) -> tuple[Image.Image, Image.Image]:
    """
    Draw glyph operations into two masks (white and accent) at the size (height) of the layer.

    :param ops: The operations of the glyph.
    :param place: Where a point of the glyph (already scaled) lands on the layer, in cover units.
    :param scale: The size of the glyph.
    :param turn: The turn of the glyph in degrees.
    :param size: The height of the layer in pixels.
    :param width: The width of the layer in pixels, the height when not given.
    :param weight: What the width of every stroke is multiplied by.
    :param fade: What the opacity of everything is multiplied by.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    masks = {
        WHITE: Image.new("L", (width or size, size), 0),
        ACCENT: Image.new("L", (width or size, size), 0),
    }
    draws = {key: ImageDraw.Draw(image) for key, image in masks.items()}
    k = size / SIZE
    spin = np.radians(turn)
    solid = round(BYTE_MAX * fade)

    def map_point(x: float, y: float) -> tuple[float, float]:
        px, py = x * np.cos(spin) - y * np.sin(spin), x * np.sin(spin) + y * np.cos(spin)
        gx, gy = place(px * scale, py * scale)
        return gx * k, gy * k

    def stroke(
        draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: float, alpha: float
    ) -> None:
        w = max(MIN_STROKE, round(width * scale * k * weight))
        fill = round(BYTE_MAX * alpha * fade)
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
            draws[colour].polygon([map_point(x, y) for x, y in pts], fill=solid)
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
                draws[colour].ellipse((x - r, y - r, x + r, y + r), fill=solid)
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
                draws[colour].polygon([map_point(x, y) for x, y in pts], fill=solid)
    return masks["w"], masks["a"]


def wave_shape(energy: float) -> tuple[float, float]:
    """
    Return the amplitude (sheet units) and the frequency (radians a step) of the wave gauge.

    :param energy: The energy of the day, 0 to 1.
    """
    return (
        WAVE_AMPLITUDE_MIN + WAVE_AMPLITUDE_RANGE * energy,
        WAVE_FREQUENCY_MIN + WAVE_FREQUENCY_RANGE * energy,
    )


def _wave_ops(energy: float = MIDDLE) -> tuple[list[Op], list[Op]]:
    """
    Return the dots-to-wave of the icon: five dots that turn into one wave, in sheet units.

    The wave is the gauge of the energy of the day: the more energy, the taller and the tighter it is.

    :param energy: The energy of the day, 0 to 1.
    """
    x, y = WAVE_ORIGIN
    amplitude, frequency = wave_shape(energy)
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
                    y + amplitude * float(np.sin(step * frequency)) * min(1.0, step / WAVE_RAMP),
                )
                for step in range(WAVE_STEPS)
            ],
            WAVE_WIDTH,
            WHITE,
            WAVE_ALPHA,
        )
    ]
    return dots, wave


# the strip of the day along the foot, in the margin under the title: a bar for every quarter of an hour
STRIP_LEFT = 72
STRIP_RIGHT = 952
STRIP_BASE = 1012
STRIP_MIN_HEIGHT = 6
STRIP_HEIGHT = 58
# the heights follow the activity to this power, so that a small bump still shows beside the peak
STRIP_CURVE = 0.6
STRIP_WIDTH = 9
STRIP_ALPHA = 0.8
STRIP_PEAK_ALPHA = 1.0
# the colours follow the energy: the saturation and the brightness are multiplied by these factors on a quiet,
# a medium and a busy day (pinned at ENERGY_QUIET, MIDDLE and ENERGY_BUSY), and the bloom is stronger or weaker
TINT_SATURATIONS = (0.2, 1.0, 1.45)
TINT_VALUES = (0.6, 1.0, 1.15)
# cool when quiet, hot when busy: red goes up and blue down by up to this much of themselves
TINT_WARMTH = 0.1
TINT_WARMTH_RGB = (1.0, 0.0, -1.0)
# the glyph by energy: its size, the width of its strokes and its opacity
GLYPH_SIZES = (0.7, 1.0, 1.2)
GLYPH_WEIGHTS = (0.55, 1.0, 1.4)
GLYPH_FADES = (0.75, OPAQUE, OPAQUE)
# a quiet glyph is drawn this far left of where a full size one would be, in cover units
QUIET_PULL = 140
# the glyph never reaches higher than this above the cover (its top is cropped off at 0 and a bit more)
GLYPH_TOP_LIMIT = -20
# above this energy the glyph leaves ghost copies (up to GHOSTS of them), like a double exposure
GHOST_FROM = 0.6
GHOSTS = 3
GHOST_ALPHAS = (0.3, 0.2, 0.12)
GHOST_STEP = 34.0
GHOST_TURN = 3.0
# the grain, by energy, and above the energy of GHOST_FROM a coarse noise texture and faint scanlines
GRAINS = (1.2, GRAIN, 4.6)
TEXTURE_AMPLITUDE = 4.0
TEXTURE_CELLS = 96
SCANLINE_PITCH = 4
SCANLINE_DARKNESS = 0.16
LUMA = (0.299, 0.587, 0.114)


def _strip_ops(profile: DayProfile) -> list[Op]:
    """Return the strip of the day: a thin bar for each quarter of an hour, with the busiest in the accent."""
    busiest = _busiest(profile)
    pitch = (STRIP_RIGHT - STRIP_LEFT) / (BINS - 1)
    ops: list[Op] = []
    for index, value in enumerate(profile.bins):
        x = STRIP_LEFT + index * pitch
        height = STRIP_MIN_HEIGHT + STRIP_HEIGHT * float(np.clip(value, 0.0, 1.0)) ** STRIP_CURVE
        is_peak = index == busiest and profile.bins[index] > 0
        ops.append(
            (
                OP_LINE,
                [(x, STRIP_BASE), (x, STRIP_BASE - height)],
                STRIP_WIDTH,
                ACCENT if is_peak else WHITE,
                STRIP_PEAK_ALPHA if is_peak else STRIP_ALPHA,
            )
        )
    return ops


def _tint(colour: str, energy: float) -> str:
    """
    Return a colour of the palette: vivid, bright and hot on a busy day, grey, dark and cool on a quiet one.

    :param colour: A #rrggbb colour.
    :param energy: The energy of the day, 0 to 1; a medium day gets the colour as it is.
    """
    swing = float(np.clip((energy - MIDDLE) / (ENERGY_BUSY - MIDDLE), -1.0, 1.0))
    rgb = np.array([int(colour[i : i + 2], HEX_BASE) for i in HEX_FIELDS], dtype=np.float64)
    grey = float(np.dot(rgb, LUMA))
    rgb = (grey + (rgb - grey) * _by_energy(energy, TINT_SATURATIONS)) * _by_energy(
        energy, TINT_VALUES
    )
    rgb *= 1 + TINT_WARMTH * swing * np.array(TINT_WARMTH_RGB)
    return _hex(np.clip(rgb, 0, BYTE_MAX))


def _ghosts(
    glyph: Image.Image, centre: tuple[float, float], energy: float, seed: str
) -> np.ndarray:
    """
    Return the ghost copies of the glyph as one 0 to 1 mask: offset, slightly turned and fading, like a double exposure.

    :param glyph: The mask of the glyph at cover size.
    :param centre: Where the glyph is centred on the cover.
    :param energy: The energy of the day; below GHOST_FROM there are none.
    :param seed: Text that gives the same ghosts for the same cover.
    """
    from PIL import Image  # noqa: PLC0415

    count = 0
    if energy > GHOST_FROM:
        count = min(GHOSTS, 1 + int(GHOSTS * _busyness(energy)))
    mask = np.zeros((glyph.height, glyph.width), dtype=np.float32)
    rng = np.random.default_rng(stable_hash("ghost", seed) % IMAGE_SEED_MODULUS)
    heading = float(rng.uniform(0.0, 2 * np.pi))
    sign = float(rng.choice((-1.0, 1.0)))
    for index in range(count):
        step = GHOST_STEP * (index + 1)
        moved = glyph.rotate(
            sign * GHOST_TURN * (index + 1),
            resample=Image.Resampling.BILINEAR,
            center=centre,
            translate=(round(step * float(np.cos(heading))), round(step * float(np.sin(heading)))),
        )
        mask = np.maximum(
            mask, np.asarray(moved, dtype=np.float32) / BYTE_MAX * GHOST_ALPHAS[index]
        )
    return mask


def _busyness(energy: float) -> float:
    """Return how far the energy of a day is above GHOST_FROM, 0 at it or below and 1 at full energy."""
    return max(0.0, (energy - GHOST_FROM) / (1 - GHOST_FROM))


def _busiest(profile: DayProfile) -> int:
    """Return the bin of the busiest quarter of an hour of the day, which the sun and the strip both mark."""
    return int(np.argmax(profile.bins))


def procedural_cover(
    profile: DayProfile,
    style_key: str,
    root: int,
    mode: str,
    seed: str,
    title: str,
    mood: CoverMood | None = None,
) -> bytes:
    """
    Draw the cover of a day as a 1024 pixel JPEG in the look of the covers of Music Assistant.

    The gradient follows the style and the key and mode of the day; the glyph is the one of the
    style, shaped by the events of the day, large and cropped off the top-right; the dots-to-wave of
    the icon is a small detail that gauges the energy; a thin strip along the foot shows the day
    round the clock; the title is set lower left by the overlay of the covers.

    The energy of the day moves the saturation, the brightness and the bloom of the colours, the
    amplitude and tightness of the wave and the glyph of every style. It is the energy of the mood,
    or the share of the day that was active when there is no mood.

    :param profile: The day, see `profile_of`.
    :param style_key: The style, which picks the palette and the glyph.
    :param root: The key of the day, 0 to 11.
    :param mode: The mode of the day.
    :param seed: Text that gives the same cover for the same area, day and style.
    :param title: The title, such as "Kitchen, Tue 17 Nov".
    :param mood: How busy the day was against the room's own days, as the music was made with it.
    """
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image, ImageChops  # noqa: PLC0415

    rng = np.random.default_rng(stable_hash("glyph", seed) % IMAGE_SEED_MODULUS)
    energy = float(np.clip(profile.intensity if mood is None else mood.energy, 0.0, 1.0))
    colours = [_tint(c, energy) for c in mood_colours(style_key, root, mode, seed)]
    angle = float(rng.random()) * GLYPH_ANGLE_RANGE
    size = _by_energy(energy, GLYPH_SIZES)
    gx = (
        GLYPH_X
        + float(rng.integers(0, GLYPH_X_RANGE))
        - QUIET_PULL * max(0.0, 1 - size) / (1 - GLYPH_SIZES[0])
    )
    reach = REACH.get(style_key, DEFAULT_REACH)
    # a big glyph is cropped harder, but not so far that its foot would come down on the title
    scale = min(
        (GLYPH_SCALE + GLYPH_SCALE_RANGE * float(rng.random())) * size,
        (GLYPH_BOTTOM - GLYPH_TOP_LIMIT) / reach,
    )
    gy = min(GLYPH_Y + float(rng.integers(0, GLYPH_Y_RANGE)), GLYPH_BOTTOM - reach * scale)
    turn = float(rng.random()) * GLYPH_TURN_RANGE - GLYPH_TURN_HALF
    bloom = (
        BLOOM_X + BLOOM_X_RANGE * float(rng.random()),
        BLOOM_Y + BLOOM_Y_RANGE * float(rng.random()),
    )
    ops = GLYPHS.get(style_key, _glyph_horizon)(profile, rng, energy)
    canvas = _background(
        colours[:GLYPH_COLOURS],
        angle,
        bloom,
        SIZE,
        bloom_strength=_by_energy(energy, BLOOM_STRENGTHS),
        bloom_radius=_by_energy(energy, BLOOM_RADII),
    )
    layer = SIZE * SUPERSAMPLE
    white, accent = _draw_ops(
        ops,
        lambda x, y: (gx + x, gy + y),
        scale,
        turn,
        layer,
        weight=_by_energy(energy, GLYPH_WEIGHTS),
        fade=_by_energy(energy, GLYPH_FADES),
    )
    dots, wave = _wave_ops(energy)
    strip = _strip_ops(profile)
    small_white, small_accent = _draw_ops(
        [*dots, *wave, *strip], lambda x, y: (x, y), 1.0, 0.0, layer
    )
    accent_rgb = np.array(
        [int(colours[ACCENT_COLOUR][i : i + 2], HEX_BASE) for i in HEX_FIELDS], dtype=np.float32
    )
    box = Image.Resampling.BOX
    glyph_w = white.resize((SIZE, SIZE), box)
    glyph_a = accent.resize((SIZE, SIZE), box)
    ghosts = _ghosts(ImageChops.lighter(glyph_w, glyph_a), (gx, gy), energy, seed)[..., None]
    canvas = canvas * (1 - ghosts) + BYTE_MAX * ghosts
    mask_w = (
        np.maximum(
            np.asarray(glyph_w, dtype=np.float32),
            np.asarray(small_white.resize((SIZE, SIZE), box), dtype=np.float32),
        )[..., None]
        / BYTE_MAX
    )
    mask_a = (
        np.maximum(
            np.asarray(glyph_a, dtype=np.float32),
            np.asarray(small_accent.resize((SIZE, SIZE), box), dtype=np.float32),
        )[..., None]
        / BYTE_MAX
    )
    canvas = canvas * (1 - mask_w) + BYTE_MAX * mask_w
    canvas = canvas * (1 - mask_a) + accent_rgb * mask_a
    # fine film grain, the same for the same seed
    canvas += rng.normal(0.0, _by_energy(energy, GRAINS), (SIZE, SIZE, 1)).astype(np.float32)
    busy = _busyness(energy)
    if busy:
        texture = rng.normal(0.0, TEXTURE_AMPLITUDE * busy, (TEXTURE_CELLS, TEXTURE_CELLS))
        canvas += np.asarray(
            Image.fromarray(texture.astype(np.float32), "F").resize(
                (SIZE, SIZE), Image.Resampling.BICUBIC
            )
        )[..., None]
        canvas[::SCANLINE_PITCH] *= 1 - SCANLINE_DARKNESS * busy
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
