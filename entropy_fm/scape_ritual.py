"""
Ritual: one plucked line, a slit drum and a woodblock that answer it off the beat, over a deep choir drone, stretched string washes, a slow chant leader and taiko.

The generative system is a slow ceremony in a Japanese pentatonic (hirajoshi, in or kumoi, on the root of the day's key).
One short call-and-response cell, 6 to 7.2 seconds long on a pulse of eight, is plucked on a strumstick, a nylon
string or a felt piano (the day deals which, and there is never a second plucked voice), round and low-passed;
about one repeat in three moves a single note by one step of the scale and never further from where the cell began,
so it stays a chant and never becomes noodling. The cell is the only thing that is plucked: two instruments of
wood answer it. A slit drum (the CC0 log drum of the VCSL, its two tongues read a few semitones up or down onto
the scale) plays a slow, sparse pattern of nine, eleven or thirteen pulses, always on an off-beat of the pluck's
pulse, so that it rests where the pluck plays, answers in the gaps and, being of another length, phases slowly
against the cell without ever sounding on the same instant (a hit within 120 ms of a pluck is dropped). A woodblock adds a very few
accents on off-beats, lower in the mix, in a pattern of fifteen to nineteen pulses. A few times in a piece, at a
calm moment or after an anomaly or a shower, a deep wooden shishi-odoshi clack falls: a soft tick of the
woodblock and, a moment later, a hollow knock of the slit drum, lowered by about six semitones. Below them a
drone of a cello section and a pad, and the pad an octave up (with a hint of
vibrato that begins after a second or two and is different for every voice) moves root, fifth, root, fourth over
minutes, its low-pass opening and closing like breath over a dozen seconds and its level swelling over eleven.
Stretched washes of string sections and piano (`stretch`) overlap under everything. Now and then a chant leader, a
cello section playing two or three long notes of the scale with rests, answers the end of the pluck cell. In the
busy part of the day a soft heartbeat drum (a low tom, no click) keeps a slow pulse and a few big taiko strikes
fall on that pulse, with a short accelerating roll into a strike at an unusual moment. There are no bells, no
chimes, no gongs and no noise: the only things that are not tones are the skins of the drums and the wood.

The day is spread over the piece as in the other styles. The night is drone and washes alone; the pluck comes in
as the day wakes, the slit drum a little later and the woodblock last, and they leave in the same order as the
evening settles; the lead and the drums are for the awake and busy stretches.

How the house steers it: the energy of the day leans the mode of the piece, still (long cells, as below), walking
(cells of 5 seconds) or driving (cells of 4.2 seconds, a pulse of 114 a minute, a matsuri-style taiko ostinato on the
pulse with the odaiko on the phrase heads, the wood on the off-beats, a leader that answers the drums and a drone that
moves more); the tie band is wide, so similar days rotate between modes, and the pulse never changes in a piece. The
rest of the character of the day picks the identity. A quiet day gets the darker `in` tuning, long
cells, a sparse pluck, the slit drum's core only, the slowest chant leader, a settled drone path that holds long and
no heartbeat or taiko; a busy day gets the brighter `kumoi`, short cells, more pluck, the full patterns of slit drum
and woodblock, an attentive leader, a faster heartbeat, more taiko and a drone that moves more. The warmth of the day
chooses the pluck voice and the washes, its shape the drone path. The root is the root of the day's key. The form of
the day (`Plan.sections`) brings the wood and the drums in and out on phrase boundaries (a cell is a bar), the room's
motif replaces the cell in the first and the last peak or plateau section, the later one developed, and the strongest anomaly brings the wood and the
drums in at the next boundary. Within a day, activity (with the daylight and the presence in the room) brings the
pluck and the wood in and out and sets their loudness, how many of the hits of the slit drum and the woodblock play
(a pattern has a core that always plays and extra hits that the activity calls in), the heartbeat and how often a
taiko is struck; the brightness opens the filters of the pluck and the drone; the power of the appliances thickens
the drone and hardens the strikes; humidity makes the room bigger; the lights and the evening make the chant leader
more likely; an anomaly is a roll into one big strike, a swell, a dip of the filters, a move of the drone to a fourth
or a fifth and, in its quiet after, a clack (so is a shower). The house never writes a note: the notes come from the
cell, the patterns, the path of the drone, the pulse and the phrases of the leader.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING

import numpy as np

from . import taiko
from .gen import dips, swells, weighted
from .harmony import MOTIF_CHOICE, develop_motif, motif_seed, motif_sites
from .score import CR, Layer, Note, Score
from .synth import smoothstep

if TYPE_CHECKING:
    from .plan import Plan

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
HIRAJOSHI = (0, 2, 3, 7, 8)
IN_SEN = (0, 1, 5, 7, 8)
KUMOI = (0, 2, 3, 7, 9)
# the tunings: a name, the five notes above the root (the root is the root of the day's key) and how bright it sounds, 0 to 1
TUNINGS: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("in", IN_SEN, 0.1),
    ("hirajoshi", HIRAJOSHI, 0.45),
    ("kumoi", KUMOI, 0.85),
)
# the length in seconds of the pluck cell (eight pulses of a slot each), and what the day calls it
LENGTHS: tuple[tuple[str, float], ...] = (
    ("6.0 seconds", 6.0),
    ("6.4 seconds", 6.4),
    ("6.8 seconds", 6.8),
    ("7.2 seconds", 7.2),
    ("5.0 seconds", 5.0),
    ("4.2 seconds", 4.2),
)
# the mode of a piece: how it moves, and the energy of day it suits. A wide tolerance lets similar days rotate between modes
MODES: tuple[tuple[str, float], ...] = (("still", 0.0), ("walking", 0.5), ("driving", 1.0))
MODE_TOLERANCE = 0.5
STILL, WALKING, DRIVING = (name for name, _ in MODES)
# the cells each mode plucks (the pulse never changes in a piece: 4.2 seconds is a pulse of 114 a minute) and how much
# busier than the day a mode plays its pluck, its wood and its drone
MODE_LENGTHS = {STILL: LENGTHS[:4], WALKING: LENGTHS[4:5], DRIVING: LENGTHS[5:]}
MODE_LIFT = {STILL: 0.0, WALKING: 0.15, DRIVING: 0.3}
# which voice plucks the line: the only plucked voice of the day, and how loud its notes are
PLUCKS: tuple[tuple[str, str, float], ...] = (
    ("strumstick", "strum_pluck", 0.62),
    ("nylon", "nylon_ks", 0.62),
    ("felt piano", "felt_piano", 0.56),
)
# the cells: (a slot on the grid of eight of the loop, a step of the scale from the root; 0 is the root at about G3).
# Three notes, a rest, and the answer: every cell calls and responds.
Cell = tuple[tuple[int, int], ...]
CELLS: tuple[tuple[str, Cell], ...] = (
    ("call and fall", ((0, 3), (1, 5), (2, 7), (4, 6), (5, 5), (6, 3))),
    ("upper turn", ((0, 5), (1, 7), (2, 5), (4, 3), (5, 5), (6, 3))),
    ("rising answer", ((0, 3), (2, 5), (3, 6), (5, 5), (6, 3), (7, 2))),
    ("descent", ((0, 7), (1, 6), (2, 5), (4, 3), (5, 5))),
    ("rocking", ((0, 5), (1, 6), (2, 5), (4, 4), (5, 3), (6, 4))),
    ("leap", ((0, 6), (1, 5), (3, 7), (4, 6), (6, 5))),
    ("sigh", ((0, 5), (2, 6), (3, 5), (5, 3), (6, 4))),
    ("bright call", ((0, 7), (1, 5), (2, 6), (4, 5), (5, 4), (6, 3))),
)
# a hit of the wood: (a pulse of the pluck's grid in the pattern, where in the pulse it falls (0.5 is the off-beat, 0.75
# the late one), which of the tones of the scale in the register, how much activity is needed to call it in (0 is the core))
Hit = tuple[int, float, int, float]
# the patterns of the slit drum: the length of the loop in pulses (never a multiple or a half of the pluck's eight) and the hits
SLIT_PATTERNS: tuple[tuple[str, int, tuple[Hit, ...]], ...] = (
    (
        "log walk",
        11,
        (
            (1, 0.5, 0, 0.0),
            (4, 0.5, 3, 0.0),
            (6, 0.75, 1, 0.55),
            (8, 0.5, 5, 0.0),
            (10, 0.5, 2, 0.8),
        ),
    ),
    ("limp", 9, ((0, 0.5, 2, 0.0), (3, 0.75, 0, 0.0), (5, 0.5, 3, 0.6), (7, 0.5, 5, 0.0))),
    (
        "long answer",
        13,
        (
            (2, 0.5, 0, 0.0),
            (5, 0.5, 5, 0.0),
            (7, 0.75, 3, 0.5),
            (9, 0.5, 1, 0.0),
            (12, 0.5, 4, 0.8),
        ),
    ),
    (
        "stumble",
        11,
        (
            (0, 0.75, 3, 0.0),
            (3, 0.5, 0, 0.0),
            (5, 0.5, 1, 0.5),
            (8, 0.75, 5, 0.0),
            (9, 0.5, 2, 0.8),
        ),
    ),
)
# the patterns of the woodblock: a few accents far apart, in loops of another length again
BLOCK_PATTERNS: tuple[tuple[str, int, tuple[Hit, ...]], ...] = (
    ("sparse taps", 17, ((3, 0.5, 0, 0.0), (10, 0.5, 2, 0.3), (14, 0.75, 1, 0.6))),
    ("far apart", 19, ((6, 0.5, 1, 0.0), (13, 0.5, 0, 0.35), (17, 0.5, 2, 0.65))),
    ("one and a late one", 15, ((4, 0.5, 2, 0.0), (11, 0.75, 0, 0.45))),
)
# no wood sounds within this many seconds of a plucked note (or of another wooden hit, twice as far)
MIN_GAP = 0.12
# the shishi-odoshi: the tick of the woodblock, then the knock of the slit drum, this many seconds later
CLACK_DELAY = 0.14
# the path of the drone: the interval above the root of each stretch (seven is the fifth, five the fourth), repeated
DRONE_PATHS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("settled", (0, 7, 0, 5)),
    ("rising", (0, 5, 7, 0)),
    ("rocking", (0, 7, 5, 7)),
    ("deep", (0, 0, 7, 5)),
)
# the washes: an instrument and the step (in semitones) above the drone root one octave up; the last one pulls
WASH_RECIPES: tuple[tuple[str, int], ...] = (
    ("wash:cello", 0),
    ("wash:cello", 7),
    ("wash:felt", -4),
    ("wash:violin", 12),
    ("wash:cello", 6),
)
WASH_SETS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("low cellos", (0, 1, 4, 2)),
    ("violin veil", (0, 3, 1, 2)),
    ("pulled", (0, 4, 3, 1)),
)
# how often the chant leader answers: the range of seconds between two phrases
LEAD_PRESENCE: tuple[tuple[str, tuple[float, float]], ...] = (
    ("rare", (95.0, 150.0)),
    ("sparse", (55.0, 90.0)),
    ("answering", (32.0, 55.0)),
    ("attentive", (22.0, 38.0)),
)
HEARTBEATS: tuple[tuple[str, int], ...] = (("single beat", 0), ("lub-dub", 1))
# what each member of a family suits. Energy: 0 a quiet day to 1 a busy one; warmth: -1 cold to 1 warm
LENGTH_ENERGY = {
    "6.0 seconds": 1.0,
    "6.4 seconds": 0.67,
    "6.8 seconds": 0.33,
    "7.2 seconds": 0.0,
    "5.0 seconds": 0.5,
    "4.2 seconds": 1.0,
}
PLUCK_WARMTH = {"strumstick": -0.3, "nylon": 0.3, "felt piano": 0.8}
CELL_ENERGY = {
    "descent": 0.05,
    "sigh": 0.2,
    "rocking": 0.4,
    "call and fall": 0.5,
    "rising answer": 0.6,
    "upper turn": 0.7,
    "leap": 0.8,
    "bright call": 0.95,
}
SLIT_ENERGY = {"long answer": 0.1, "limp": 0.4, "log walk": 0.65, "stumble": 0.9}
BLOCK_ENERGY = {"one and a late one": 0.0, "far apart": 0.5, "sparse taps": 1.0}
WASH_WARMTH = {"low cellos": -0.5, "pulled": 0.0, "violin veil": 0.7}
LEAD_ENERGY = {"rare": 0.0, "sparse": 0.33, "answering": 0.66, "attentive": 1.0}
HEART_ENERGY = {"single beat": 0.25, "lub-dub": 0.75}
# the drone paths: the energy each suits and the shapes of day it follows (a morning settles, an evening rises)
PATH_ENERGY = {"deep": 0.05, "settled": 0.3, "rocking": 0.6, "rising": 0.9}
PATH_SHAPES = {
    "deep": ("flat",),
    "settled": ("morning", "all_day"),
    "rocking": ("peak",),
    "rising": ("evening",),
}
SHAPE_BONUS = 0.15
# the brightness the day asks of a tuning: warmth and energy raise it, a late night lowers it
TUNING_WARMTH = 0.35
TUNING_ENERGY = 0.3
TUNING_LATE = 0.25
LATE_DARKEN = 0.5
# the form: a cell is a bar; the section strength each layer needs to play, and what the strongest anomaly lifts the next one to
SLIT_FROM = 0.3
BLOCK_FROM = 0.45
DRUMS_FROM = 0.6
LEAD_FROM = 0.3
EVENT_FORM = 0.7
# the ostinato of the driving mode, a bar of eight pulses on the grid of the pluck, in the vocabulary of matsuri
# drumming: the don of the odaiko on the even pulses (the head of the bar the strongest), the ji of the shime on the
# odd ones, the doko of a quick pickup and a rim click: (pulse, where in it, drum, weight, activity needed)
OSTINATO: tuple[tuple[int, float, str, float, float], ...] = (
    (0, 0.0, "odaiko", 0.7, 0.0),
    (2, 0.0, "odaiko", 0.52, 0.0),
    (4, 0.0, "odaiko", 0.62, 0.0),
    (6, 0.0, "odaiko", 0.52, 0.0),
    (1, 0.0, "shime", 0.4, 0.0),
    (3, 0.0, "shime", 0.4, 0.0),
    (5, 0.0, "shime", 0.4, 0.0),
    (7, 0.0, "shime", 0.4, 0.0),
    (3, 0.25, "shime", 0.3, 0.4),
    (7, 0.25, "shime", 0.3, 0.4),
    (2, 0.25, "taiko_rim", 0.5, 0.6),
    (6, 0.25, "taiko_rim", 0.5, 0.6),
)
# a strong don falls on the head of every fourth bar, and the ostinato needs a section and a room this strong
PHRASE_BARS = 4
HEAD_LIFT = 0.22
DRIVE_FROM = 0.7
OSTINATO_GATE = 0.3
# the roll into a peak or plateau section lasts this many pulses; the leader of a driving piece waits this share of the gap
ROLL_PULSES = 4
LEAD_DRIVE_SCALE = 0.7
# the energy below which a day has no heartbeat or taiko of its own
DRUMS_ENERGY = 0.3
# how much of the pluck plays on the quietest and the busiest day, how often a whole loop of the wood rests
PLUCK_DENSITY = (0.3, 1.0)
REST_SHARE = (0.5, 0.1)
# a pattern's extra hits follow the activity of the day against its own peak; the energy lifts or lowers what they need
ACTIVITY_FLOOR = 0.15
EXTRA_REACH = 0.4
EXTRA_NEED = 0.6
# how much more often a loop rests in a calm moment than in a middling one, and how much less in the busiest
REST_ACTIVITY = 0.8
# how far a busy moment lifts the section of the form towards the strength a layer needs to play
FORM_LIFT = 0.35
MIN_PEAK_ACTIVITY = 0.2
# the heartbeat in beats per minute, a taiko strike's chance per beat, the drone's stretch in seconds (low, high): quiet then busy
BEATS_PER_MINUTE = (40.0, 58.0)
TAIKO_CHANCE = (0.03, 0.2)
DRONE_SPAN_QUIET = (85.0, 140.0)
DRONE_SPAN_BUSY = (45.0, 80.0)
# the room's motif on the pluck: never below this note (C4)
# the last stretch of a piece is drone and washes: no cell ends within this many seconds of the end
PLUCK_TAIL = 21.0
# the chant leader sings within this range of notes
LEAD_RANGE = (53, 78)
MOTIF_FLOOR = 60
MOTIF_ROLES = ("peak", "plateau")
MOTIF_ACCENT = 1.1
# the pluck must be this awake for a cell to be given to the motif
MOTIF_AWAKE = 0.35
MOTIF_OCTAVE_STEPS = 5
# a cell is this many slots; the motif is at most as long
CELL_SLOTS = 8.0
WASH_SECONDS = 24.0
DRONE_STEP = 18.0
DRONE_OVERLAP = 4.5


def _closeness(tag: float, wanted: float, span: float = 1.0) -> float:
    """Return 1 when a tag is what the day wants and 0 when it is a whole span away."""
    return 1.0 - abs(tag - wanted) / span


def _lfo(count: int, period: float, phase: float) -> np.ndarray:
    """Return a slow 0 to 1 swing with `period` seconds, a value every tenth of a second."""
    t = (np.arange(count) + 0.5) / CR
    return np.asarray(0.5 - 0.5 * np.cos(2.0 * np.pi * (t / period + phase)))


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Ritual soundscape.

    :param plan: The curves and the deals of the day.
    """
    quality = plan.quality
    length = plan.length
    rng = plan.rng("ritual")
    light = quality.max_layers <= 4

    character = plan.character
    energy = character.energy
    warmth = float(np.clip(character.warmth - LATE_DARKEN * character.late_night, -1.0, 1.0))
    brightness = float(
        np.clip(
            0.5
            + TUNING_WARMTH * character.warmth
            + TUNING_ENERGY * (energy - 0.5)
            - TUNING_LATE * character.late_night,
            0.0,
            1.0,
        )
    )
    cell_wanted = 0.7 * energy + 0.3 * (0.5 + 0.5 * warmth)
    tuning_name, steps, _ = plan.pick(
        "tuning", TUNINGS, [_closeness(bright, brightness) for _, _, bright in TUNINGS]
    )
    root_pc = plan.scale.root
    mode, _ = plan.pick(
        "ritual mode", MODES, [_closeness(wanted, energy) for _, wanted in MODES], MODE_TOLERANCE
    )
    driving = mode == DRIVING
    drive_energy = float(min(energy + MODE_LIFT[mode], 1.0))
    lengths = MODE_LENGTHS[mode]
    _, pluck_len = plan.pick(
        "pluck cell length", lengths, [_closeness(LENGTH_ENERGY[n], energy) for n, _ in lengths]
    )
    pluck_name, pluck_voice, pluck_loud = plan.pick(
        "pluck voice", PLUCKS, [_closeness(PLUCK_WARMTH[n], warmth, 2.0) for n, _, _ in PLUCKS]
    )
    cell_name, cell = plan.pick(
        "pluck cell", CELLS, [_closeness(CELL_ENERGY[n], cell_wanted) for n, _ in CELLS]
    )
    slit_name, slit_pulses, slit_hits = plan.pick(
        "slit drum pattern",
        SLIT_PATTERNS,
        [_closeness(SLIT_ENERGY[n], energy) for n, _, _ in SLIT_PATTERNS],
    )
    block_name, block_pulses, block_hits = plan.pick(
        "woodblock pattern",
        BLOCK_PATTERNS,
        [_closeness(BLOCK_ENERGY[n], energy) for n, _, _ in BLOCK_PATTERNS],
    )
    path_name, path = plan.pick(
        "drone path",
        DRONE_PATHS,
        [
            _closeness(PATH_ENERGY[n], energy) + SHAPE_BONUS * (character.shape in PATH_SHAPES[n])
            for n, _ in DRONE_PATHS
        ],
    )
    wash_name, wash_order = plan.pick(
        "wash set", WASH_SETS, [_closeness(WASH_WARMTH[n], warmth, 2.0) for n, _ in WASH_SETS]
    )
    lead_name, lead_gap = plan.pick(
        "chant leader",
        LEAD_PRESENCE,
        [_closeness(LEAD_ENERGY[n], energy) for n, _ in LEAD_PRESENCE],
    )
    heart_name, heart_style = plan.pick(
        "heartbeat", HEARTBEATS, [_closeness(HEART_ENERGY[n], energy) for n, _ in HEARTBEATS]
    )
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["root"] = str(root_pc)

    def ladder(index: int, base: int) -> float:
        """Return the pitch of a step of the scale: five notes an octave, `base` is the root."""
        octave, degree = divmod(index, 5)
        return float(base + 12 * octave + steps[degree])

    pluck_base = 55 + (root_pc - 55) % 12
    lead_base = 53 + (root_pc - 53) % 12
    lead_steps = [i for i in range(-2, 7) if LEAD_RANGE[0] <= ladder(i, lead_base) <= LEAD_RANGE[1]]
    drone_root = 38 + (root_pc - 38) % 12

    # -- the curves of the day
    count = plan.count
    activity = plan.activity
    # how awake the room is: the daylight, lifted by activity; a little of the evening stays while the house is lively
    awake = np.clip(
        plan.daylight * (0.4 + 0.6 * activity) + 0.3 * plan.evening * activity, 0.0, 1.0
    )
    busy = plan.gate(awake, 0.34, 0.6)
    # the plucks and the chant are the base of the day: a room that is calm is awake enough for the pluck by day
    # and in the evening, and the house moves how loud it is and how often it is moved
    awake_base = np.clip(
        plan.daylight * (0.62 + 0.38 * activity) + 0.6 * plan.evening * (0.6 + 0.4 * activity),
        0.0,
        1.0,
    )
    relative_activity = np.clip(activity / max(float(activity.max()), MIN_PEAK_ACTIVITY), 0.0, 1.0)
    swell = swells(plan)
    breath = _lfo(count, float(rng.uniform(11.5, 13.5)), float(rng.uniform()))
    surge = 0.55 + 0.45 * _lfo(count, float(rng.uniform(10.0, 12.0)), float(rng.uniform()))

    notes: list[Note] = []

    # -- the form: a cell is a bar, so the sections change every 8 cells; the strongest anomaly lifts the next section
    sections = plan.sections(pluck_len)
    starts = [start for start, _, _, _ in sections]
    strength = [intensity for _, _, _, intensity in sections]
    if plan.events:
        strongest, _ = max(plan.events, key=lambda event: event[1])
        after = bisect_right(starts, strongest)
        if after < len(sections):
            strength[after] = max(strength[after], EVENT_FORM)
    lifted = {i for i, (_, _, _, base) in enumerate(sections) if strength[i] > base}
    drums_on = energy >= DRUMS_ENERGY

    def form_at(second: float) -> float:
        return strength[max(bisect_right(starts, second) - 1, 0)]

    def drums_at(second: float) -> bool:
        index = max(bisect_right(starts, second) - 1, 0)
        return (drums_on or index in lifted) and strength[index] >= DRUMS_FROM

    # -- the drone: a cello section and a Juno pad with a vibrato of their own, a quiet pad an octave up, and a round sub; the root moves along the path, and at an anomaly it goes to the fifth or the fourth
    segments: list[tuple[float, float, int]] = []
    clock, place = 0.0, 0
    while clock < length:
        span = float(
            rng.uniform(
                DRONE_SPAN_QUIET[0] + (DRONE_SPAN_BUSY[0] - DRONE_SPAN_QUIET[0]) * drive_energy,
                DRONE_SPAN_QUIET[1] + (DRONE_SPAN_BUSY[1] - DRONE_SPAN_QUIET[1]) * drive_energy,
            )
        )
        end = min(clock + span, float(length))
        if length - end < 35.0:
            end = float(length)
        segments.append((clock, end, path[place % len(path)] if clock > 0.0 else 0))
        clock, place = end, place + 1
    for second, _ in plan.events:
        for k, (start, end, _interval) in enumerate(segments):
            if start <= second < end and end - second > 12.0:
                far = int(weighted(rng, (7, 5), (0.6, 0.4)))
                segments[k] = (start, second, segments[k][2])
                segments.insert(k + 1, (second, min(second + 45.0, end), far))
                if second + 45.0 < end:
                    segments.insert(k + 2, (second + 45.0, end, segments[k][2]))
                break
    index = 0
    for start, end, interval in segments:
        second = start
        while second < end - 3.0:
            dur = float(min(DRONE_STEP + DRONE_OVERLAP, end + DRONE_OVERLAP - second))
            pitch = float(drone_root + interval)
            vel = 0.5 + 0.12 * plan.at(plan.power, second)
            notes.append(Note(second, dur, pitch, vel, "tape_strings", "drone", index % 2, 6.0))
            notes.append(
                Note(second, dur, pitch, vel - 0.05, "juno_pad", "drone", (index + 1) % 2, 5.0)
            )
            if not light:
                notes.append(Note(second, dur, pitch + 12.0, 0.3, "juno_pad", "drone", index % 2))
            sub = pitch - 12.0 if pitch - 12.0 >= 26.0 else pitch
            notes.append(Note(second, dur, sub, 0.6, "sub_round", "sub", 1, 0.0))
            second += DRONE_STEP
            index += 1

    # -- the washes: stretched sections that overlap and replace each other, drawn from a few recipes so that each is made once
    if not light:
        recipes = [WASH_RECIPES[i] for i in wash_order]
        second, turn = 0.0, int(rng.integers(0, len(recipes)))
        while second < length:
            voice, step = recipes[turn % len(recipes)]
            pitch = float(drone_root + 12 + step)
            notes.append(Note(second, WASH_SECONDS, pitch, 0.6, voice, "wash", 0, 0.0))
            second += float(rng.uniform(15.0, 24.0))
            turn += 1

    # -- the pluck: one cell that comes in as the day wakes and leaves as the evening settles; the only plucked voice.
    # On a quiet day it plays sparsely; in the first and the last peak or plateau section the room's motif replaces one cell
    slot = pluck_len / CELL_SLOTS
    pluck_rng = plan.rng("loop", "pluck")
    base_index = [step for _, step in cell]
    current = list(base_index)
    origin = 0.0
    occupied = {where for where, _ in cell}
    pluck_onsets: list[float] = []
    last_cells: list[
        tuple[float, float, float]
    ] = []  # (end of the response, the pitch it ended on, length of the loop)
    thin = PLUCK_DENSITY[0] + (PLUCK_DENSITY[1] - PLUCK_DENSITY[0]) * drive_energy
    cells = int((length - PLUCK_TAIL) / pluck_len)

    def pluck_presence(second: float) -> float:
        value = float(np.clip((plan.at(awake_base, second) - 0.1) / 0.26, 0.0, 1.0))
        return float(smoothstep(np.array(value)))

    # the motif lives on the ladder of the tuning: a degree of the day's scale lands on the nearest step of the pentatonic
    def motif_on_ladder(number: int) -> tuple[tuple[tuple[int, float], ...], list[int], int]:
        """Return the motif of a statement, its steps on the ladder and the pitch it starts from."""
        # a statement is at most one cell long
        motif = develop_motif(plan.motif(), number, motif_seed(plan), max_beats=CELL_SLOTS)
        steps = [round(degree * MOTIF_OCTAVE_STEPS / plan.scale.size) for degree, _ in motif]
        base = pluck_base
        while min(ladder(step, base) for step in steps) < MOTIF_FLOOR:
            base += 12
        return motif, steps, base

    awake_firsts: list[int] = []
    motif_sections = [i for i, (_, _, role, _) in enumerate(sections) if role in MOTIF_ROLES]
    for i in motif_sections or range(len(sections)):
        first = round(sections[i][0] / pluck_len)
        last = min(round(sections[i][1] / pluck_len), cells)
        awake_cells = [k for k in range(first, last) if pluck_presence(k * pluck_len) > MOTIF_AWAKE]
        if awake_cells:
            awake_firsts.append(awake_cells[0])
        if awake_firsts and not motif_sections:
            break
    motif_cells = {k: motif_on_ladder(number) for number, k in enumerate(motif_sites(awake_firsts))}
    motif_played = 0
    for k in range(cells):
        second = origin + k * pluck_len
        if second + pluck_len > length - PLUCK_TAIL:
            break
        presence = pluck_presence(second)
        if k in motif_cells:
            motif_played += 1
            motif, motif_steps, motif_base = motif_cells[k]
            clock = 0.0
            pitch = float(motif_base)
            for (_, beats), step in zip(motif, motif_steps, strict=True):
                when = second + clock * slot
                pitch = ladder(step, motif_base)
                vel = pluck_loud * MOTIF_ACCENT * (0.78 + 0.3 * max(presence, MOTIF_AWAKE))
                notes.append(
                    Note(
                        when,
                        max(0.7, beats * slot),
                        pitch,
                        float(np.clip(vel, 0.1, 1.0)),
                        pluck_voice,
                        "plucks",
                        patch,
                    )
                )
                pluck_onsets.append(when)
                clock += beats
            last_cells.append((second + clock * slot, pitch, pluck_len))
            continue
        if presence > 0.04 and pluck_rng.random() < presence**0.5 * thin * (
            0.5 + 0.5 * form_at(second)
        ):
            if pluck_rng.random() < 0.33:
                pick = int(pluck_rng.integers(0, len(current)))
                current[pick] = int(
                    np.clip(
                        current[pick] + int(pluck_rng.choice((-1, 1))),
                        base_index[pick] - 1,
                        base_index[pick] + 1,
                    )
                )
            pitch = pluck_base
            for number, ((where, _), step) in enumerate(zip(cell, current, strict=True)):
                when = second + where * slot + float(pluck_rng.uniform(-0.008, 0.008))
                accent = 1.12 if number == 0 else 1.0
                vel = (
                    pluck_loud
                    * accent
                    * (0.78 + 0.3 * presence)
                    * (0.92 + 0.12 * plan.at(plan.bright, when))
                    * float(pluck_rng.uniform(0.92, 1.06))
                )
                pitch = ladder(step, pluck_base)
                notes.append(
                    Note(
                        when,
                        0.7,
                        pitch,
                        float(np.clip(vel, 0.1, 1.0)),
                        pluck_voice,
                        "plucks",
                        patch,
                    )
                )
                pluck_onsets.append(when)
            last_cells.append((second + (cell[-1][0] + 1.0) * slot, pitch, pluck_len))
    pluck_onsets.sort()

    # -- the wood: everything of the wood is placed on the pulse of the pluck's grid, on its off-beats, and checked against
    # the notes that are plucked, so that nothing of the wood ever sounds with a pluck
    def tones(low: int, high: int) -> list[float]:
        """Return the notes of the day's scale from `low` to `high`, lowest first."""
        return [float(m) for m in range(low, high + 1) if (m - root_pc) % 12 in steps]

    slit_tones = tones(42, 56)
    block_tones = tones(83, 90)
    knock_tone = tones(39, 50)[0]
    tick_tone = block_tones[0]
    wood_hits: list[float] = []  # every sound of the wood, for the gaps between them
    slit_count = block_count = 0

    def free(when: float, gap: float = MIN_GAP, wood_gap: float = 0.0) -> bool:
        """Return True when `when` is clear of the plucked notes by `gap` and of the other wooden hits by `wood_gap`."""
        for sorted_times, apart in ((pluck_onsets, gap), (sorted(wood_hits), wood_gap)):
            if not sorted_times or apart <= 0.0:
                continue
            k = bisect_left(sorted_times, when)
            near = sorted_times[max(k - 1, 0) : k + 1]
            if any(abs(when - other) < apart for other in near):
                return False
        return True

    # the shishi-odoshi: a soft tick of the woodblock, then a hollow knock of the slit drum, lowered; at the quiet after an
    # anomaly or a shower and at a few calm moments, a few times in a piece and never close to each other
    clack_rng = plan.rng("clack")
    wanted_clacks = int(np.clip(length / 100.0, 1, 5))
    clack_times: list[float] = []
    candidates = [
        (second + 2.8, "event")
        for second, _ in plan.events
        if plan.at(plan.night, second) < 0.6 and plan.at(awake_base, second) > 0.15
    ]
    wander = float(clack_rng.uniform(25.0, 60.0))
    while wander < length - 12.0:
        if (
            plan.at(awake_base, wander) > 0.3
            and plan.at(activity, wander) < 0.55
            and clack_rng.random() < 0.7
        ):
            candidates.append((wander, "calm"))
        wander += float(clack_rng.uniform(35.0, 70.0))
    for at, _kind in sorted(candidates, key=lambda c: (c[1] != "event", c[0])):
        if len(clack_times) >= wanted_clacks or at > length - 8.0:
            continue
        if any(abs(at - other) < 30.0 for other in clack_times):
            continue
        for push in (0.0, 0.3, -0.3, 0.6):
            when = at + push
            if free(when - 0.2, 0.3) and free(when, 0.3) and free(when + CLACK_DELAY + 0.2, 0.3):
                clack_times.append(when)
                break
    if not light:
        for when in clack_times:
            notes.append(
                Note(
                    when, 0.12, tick_tone, float(clack_rng.uniform(0.3, 0.4)), "woodblock", "clack"
                )
            )
            notes.append(
                Note(
                    when + CLACK_DELAY,
                    0.14,
                    knock_tone,
                    float(clack_rng.uniform(0.74, 0.86)),
                    "slit_drum",
                    "clack",
                )
            )
            wood_hits.extend((when, when + CLACK_DELAY))

    def lay_pattern(  # noqa: PLR0913, PLR0917
        hits: tuple[Hit, ...],
        pulses: int,
        tone_list: list[float],
        voice: str,
        layer: str,
        gate: np.ndarray,
        loud: float,
        pattern_rng: np.random.Generator,
        clear_of_wood: float,
        rest_share: float,
        from_strength: float,
    ) -> int:
        """Place a pattern on the pulses of the pluck, loop after loop; return how many hits were played."""
        played = 0
        loop = 0
        while origin + loop * pulses * slot < length - 2.0:
            # now and then a whole loop is left out, so that the pattern breathes
            calm = 1.0 - 2.0 * plan.at(relative_activity, origin + loop * pulses * slot)
            resting = pattern_rng.random() < rest_share * (1.0 + REST_ACTIVITY * calm)
            for pulse, frac, tone, need in hits:
                if resting:
                    continue
                when = origin + (loop * pulses + pulse + frac) * slot
                if when >= length - 2.0:
                    continue
                here = plan.at(gate, when)
                drive = ACTIVITY_FLOOR + (1.0 - ACTIVITY_FLOOR) * plan.at(relative_activity, when)
                roll = pattern_rng.random()
                if (
                    form_at(when) + FORM_LIFT * plan.at(relative_activity, when) < from_strength
                    or here < 0.05
                    or roll > here**0.5
                    or need * EXTRA_NEED > drive + EXTRA_REACH * (drive_energy - 0.5)
                    or not free(when, MIN_GAP, clear_of_wood)
                ):
                    continue
                # a hit that falls where the pluck rests is the answer: a little firmer than one that fills a gap
                answer = 1.1 if pulse % 8 not in occupied else 0.95
                vel = loud * answer * (0.8 + 0.25 * drive) * float(pattern_rng.uniform(0.92, 1.06))
                notes.append(
                    Note(
                        when,
                        0.5,
                        tone_list[tone % len(tone_list)],
                        float(np.clip(vel, 0.1, 1.0)),
                        voice,
                        layer,
                    )
                )
                wood_hits.append(when)
                played += 1
            loop += 1
        return played

    slit_gate = plan.gate(awake_base, 0.2, 0.5)
    slit_count = lay_pattern(
        slit_hits,
        slit_pulses,
        slit_tones,
        "slit_drum",
        "slit",
        slit_gate,
        0.62,
        plan.rng("slit"),
        0.45,
        REST_SHARE[0] + (REST_SHARE[1] - REST_SHARE[0]) * drive_energy,
        SLIT_FROM,
    )
    if not light:
        block_count = lay_pattern(
            block_hits,
            block_pulses,
            block_tones,
            "woodblock",
            "block",
            plan.gate(awake_base, 0.4, 0.7),
            0.5,
            plan.rng("block"),
            0.3,
            REST_SHARE[0] + (REST_SHARE[1] - REST_SHARE[0]) * drive_energy,
            BLOCK_FROM,
        )

    # -- the chant leader: a cello section answers the end of a pluck cell with two or three long notes and rests
    lead_notes = 0
    if not light and last_cells:
        lead_rng = plan.rng("lead")
        last_phrase = -1000.0
        previous = 1
        lead_gap = (
            (lead_gap[0] * LEAD_DRIVE_SCALE, lead_gap[1] * LEAD_DRIVE_SCALE)
            if driving
            else lead_gap
        )
        for end, pitch, _ in sorted(last_cells):
            if end - last_phrase < lead_gap[0] or end > length - 14.0 or form_at(end) < LEAD_FROM:
                continue
            here = plan.at(awake_base, end)
            wanted = 0.45 + 0.55 * min(
                1.0, here + 0.5 * plan.at(plan.lights, end) + 0.5 * plan.at(plan.evening, end)
            )
            wait = lead_gap[0] + (lead_gap[1] - lead_gap[0]) * float(lead_rng.random())
            if here < 0.25 or end - last_phrase < wait or lead_rng.random() > wanted:
                continue
            # the pitch of the answer: the step of the lead's scale nearest to where the pluck ended, then a stepwise walk home
            reach = [
                (abs(ladder(i, lead_base) - (pitch - 12.0 * round((pitch - lead_base) / 12.0))), i)
                for i in lead_steps
            ]
            current_step = min(reach)[1] if lead_rng.random() < 0.7 else previous
            count_in_phrase = int(weighted(lead_rng, (2, 3), (0.45, 0.55)))
            clock = end + 0.5 + float(lead_rng.uniform(0.0, 0.9))
            if driving:
                clock = (end // pluck_len + 1.0) * pluck_len + 2.0 * slot
            for number in range(count_in_phrase):
                final = number == count_in_phrase - 1
                if final:
                    current_step = int(weighted(lead_rng, (0, 3, 5), (0.5, 0.25, 0.25)))
                elif number > 0:
                    current_step = int(
                        np.clip(
                            current_step + int(lead_rng.choice((-2, -1, -1, 1))),
                            lead_steps[0],
                            lead_steps[-1],
                        )
                    )
                dur = float(lead_rng.uniform(5.0, 8.0) if final else lead_rng.uniform(3.2, 5.0))
                vel = float(lead_rng.uniform(0.46, 0.58)) * (
                    0.85 + 0.15 * plan.at(plan.bright, clock)
                )
                notes.append(
                    Note(
                        clock,
                        dur,
                        ladder(current_step, lead_base),
                        vel,
                        "chant_cello",
                        "lead",
                        patch,
                        6.0,
                    )
                )
                lead_notes += 1
                previous = current_step
                clock += dur + float(lead_rng.uniform(0.7, 1.9))
            last_phrase = end

    # -- the pulse and the taiko, in the busy stretches only
    bpm = BEATS_PER_MINUTE[0] + (BEATS_PER_MINUTE[1] - BEATS_PER_MINUTE[0]) * energy
    if driving:
        bpm = 60.0 / slot
    beat = 60.0 / bpm
    taiko_hits: list[tuple[float, str, float]] = []

    def drum_gate(second: float) -> float:
        """Return how busy the room is where the form and the energy of the day let the drums play, else 0."""
        return plan.at(busy, second) if drums_at(second) else 0.0

    big: list[float] = []
    quiet = 0.0
    drum_rng = plan.rng("taiko")
    strikes = 0
    rolls = 0
    grid = 0.0 if driving else float(rng.uniform(0.0, beat))

    def on_grid(second: float) -> float:
        return grid + round((second - grid) / beat) * beat

    # an anomaly that is not in the dead of the night is a roll into one big strike
    event_starts: list[float] = []
    for second, _ in plan.events:
        if plan.at(plan.night, second) < 0.6:
            end = on_grid(second)
            span = float(drum_rng.uniform(3.0, 4.2))
            taiko_hits.extend(
                (hit.time, "odaiko", hit.velocity)
                for hit in taiko.roll(end - span, end, "odaiko", 0.18, 0.92)
            )
            event_starts.append(end - span)
            big.append(end)
            rolls += 1
    # a roll into the head of every peak or plateau section
    if driving:
        for start, _, role, _ in sections[1:]:
            if role in MOTIF_ROLES and drum_gate(start) > OSTINATO_GATE:
                span = ROLL_PULSES * slot
                taiko_hits.extend(
                    (hit.time, "odaiko", hit.velocity)
                    for hit in taiko.roll(start - span, start, "odaiko", 0.18, 0.9)
                )
                event_starts.append(start - span)
                big.append(start)
                rolls += 1
    # chance strikes on the beat, rarer when the house is calm, never close to another one
    others: list[float] = [*big]
    beat_time = grid
    while beat_time < length - 4.0:
        gate = drum_gate(beat_time)
        near = any(abs(beat_time - other) < 5.0 for other in others)
        inside_roll = any(
            start - 0.5 <= beat_time <= end + beat
            for start, end in zip(event_starts, big, strict=True)
        )
        if gate > 0.55 and not near and not inside_roll and not driving:
            chance = (
                TAIKO_CHANCE[0]
                + (TAIKO_CHANCE[1] - TAIKO_CHANCE[0])
                * energy
                * (0.5 + 0.5 * plan.at(activity, beat_time))
            ) * gate
            if drum_rng.random() < chance:
                vel = (
                    0.52
                    + 0.26 * plan.at(activity, beat_time)
                    + 0.14 * plan.at(plan.power, beat_time)
                )
                taiko_hits.append((beat_time, "odaiko", float(vel)))
                others.append(beat_time)
                strikes += 1
                extra = drum_rng.random()
                if extra < 0.18:
                    taiko_hits.append((beat_time + beat * 1.5, "taiko_rim", 0.5))
                elif extra < 0.32:
                    for k, v in enumerate((0.45, 0.35, 0.55)):
                        taiko_hits.append((beat_time + beat * (1.0 + 0.5 * k), "shime", v))
        beat_time += beat
    # a rare unprompted roll at the height of a busy stretch, not more often than every minute and a half
    roll_clock = 0.0
    while roll_clock < length - 12.0:
        roll_clock += float(drum_rng.uniform(30.0, 60.0))
        if (
            roll_clock < length - 12.0
            and drum_gate(roll_clock) > 0.7
            and drum_rng.random() < 0.55
            and not any(abs(roll_clock - other) < 20.0 for other in others)
        ):
            end = on_grid(roll_clock)
            taiko_hits.extend(
                (hit.time, "odaiko", hit.velocity)
                for hit in taiko.roll(end - 3.4, end, "odaiko", 0.18, 0.85)
            )
            event_starts.append(end - 3.4)
            big.append(end)
            others.append(end)
            rolls += 1
            roll_clock += 90.0
    # the ostinato, on the pulse of the pluck: it builds with the section, calls in more hits in the busy part and stops in
    # a release or an outro
    ostinato_hits = 0
    ostinato_rng = plan.rng("ostinato")
    bar = 0
    while driving and bar * pluck_len < length - 4.0:
        for pulse, frac, drum, weight, need in OSTINATO:
            when = (bar * CELL_SLOTS + pulse + frac) * slot
            form = form_at(when)
            drive = ACTIVITY_FLOOR + (1.0 - ACTIVITY_FLOOR) * plan.at(relative_activity, when)
            if (
                when >= length - 4.0
                or form < DRIVE_FROM
                or drum_gate(when) <= OSTINATO_GATE
                or need * EXTRA_NEED > drive + EXTRA_REACH * (drive_energy - 0.5)
                or any(
                    start - 0.5 <= when <= end + beat
                    for start, end in zip(event_starts, big, strict=True)
                )
            ):
                continue
            head = HEAD_LIFT if pulse == 0 and bar % PHRASE_BARS == 0 else 0.0
            vel = (weight + head) * (0.7 + 0.3 * form) * float(ostinato_rng.uniform(0.94, 1.04))
            taiko_hits.append((when, drum, vel))
            ostinato_hits += 1
        bar += 1
    # one last soft strike as the busy stretch ends
    awake_beats = [t for t in np.arange(grid, length - 4.0, beat) if drum_gate(float(t)) > 0.55]
    if taiko_hits and awake_beats:
        closing = float(awake_beats[-1]) + 4.0 * beat
        if closing < length - 6.0 and not any(abs(closing - other) < 4.0 for other in others):
            taiko_hits.append((closing, "odaiko", 0.5))
            others.append(closing)
            quiet = closing
    for when, hit, vel in taiko_hits:
        notes.append(
            Note(
                when,
                0.2,
                36.0,
                float(np.clip(vel, 0.1, 1.0)),
                f"drum:{hit}",
                "taiko",
                int(drum_rng.integers(0, 3)),
            )
        )
    # the heartbeat fills the beats between the big strikes
    heart_rng = plan.rng("heart")
    beat_time = grid
    beats = 0
    while beat_time < length - 3.0:
        gate = drum_gate(beat_time)
        taken = any(
            abs(beat_time - hit_time) < 0.9 * beat
            for hit_time, kind, _ in taiko_hits
            if kind == "odaiko"
        )
        rolling = any(
            start - 0.2 <= beat_time <= end + 0.6 * beat
            for start, end in zip(event_starts, big, strict=True)
        )
        if gate > 0.3 and not taken and not rolling and not driving:
            vel = (0.22 + 0.12 * gate + 0.04 * plan.at(plan.power, beat_time)) * float(
                heart_rng.uniform(0.92, 1.06)
            )
            variant = int(heart_rng.integers(0, 3))
            notes.append(Note(beat_time, 0.2, 36.0, vel, "drum:heart", "heart", variant))
            if heart_style == 1:
                notes.append(
                    Note(
                        beat_time + 0.27 * beat,
                        0.2,
                        36.0,
                        vel * 0.6,
                        "drum:heart",
                        "heart",
                        (variant + 1) % 3,
                    )
                )
            beats += 1
        beat_time += beat

    notes = [n for n in notes if n.t < length]

    # -- the layers
    bright = plan.bright
    low_cut, high_cut = 520.0, 2100.0
    drone_cut = low_cut * (high_cut / low_cut) ** breath * (0.85 + 0.3 * bright) * dips(plan, 0.85)
    pluck_cut = 2500.0 * (0.75 + 0.45 * bright) * dips(plan, 0.6)
    block_cut = 3300.0 * (0.8 + 0.4 * bright) * dips(plan, 0.6)
    drone_presence = np.clip(surge * (0.9 + 0.12 * plan.power) * (1.0 + 0.25 * swell), 0.0, 1.25)
    wash_presence = np.clip(
        (0.45 + 0.45 * plan.night + 0.2 * swell) * (1.0 - 0.25 * busy), 0.0, 1.0
    )
    layers = {
        "drone": Layer(
            "drone", 1.0, drone_presence, drone_cut, 0.0, reverb=0.4, title="choir drone, breathing"
        ),
        "sub": Layer("sub", 0.9, drone_presence, None, 0.0, reverb=0.1, title="round sub"),
        "wash": Layer("wash", 1.0, wash_presence, None, 0.0, reverb=0.8, title="stretched washes"),
        "plucks": Layer("plucks", 1.0, None, pluck_cut, 0.0, reverb=0.62, title="plucked line"),
        "slit": Layer("slit", 1.0, None, None, 0.0, reverb=0.5, title="slit drum"),
        "block": Layer("block", 1.0, None, block_cut, 0.0, reverb=0.5, title="woodblock"),
        "clack": Layer("clack", 1.0, None, None, 0.0, reverb=0.65, title="shishi-odoshi"),
        "lead": Layer("lead", 1.0, None, None, 0.0, reverb=0.6, title="chant leader"),
        "taiko": Layer("taiko", 1.0, None, None, 0.0, reverb=0.5, title="taiko"),
        "heart": Layer("heart", 1.0, None, None, 0.0, reverb=0.25, title="heartbeat"),
    }
    space = np.clip(0.5 + 0.25 * plan.humid + 0.2 * swell + 0.1 * plan.night, 0.0, 0.95)
    score = Score(
        length,
        notes,
        layers,
        [],
        space,
        events=list(plan.events),
        compress_db=-15.0,
        level_db=-18.8,
    )
    plan.choices["patch"] = str(patch)
    plan.choices[MOTIF_CHOICE] = str(motif_played)
    score.choices = dict(plan.choices)
    used = score.voices_used()
    score.notes_about = [
        (
            f"{tuning_name} on {PITCH_NAMES[root_pc]}, {mode} ({60.0 / slot:.0f} pulses a minute); one plucked line on {pluck_name} (cell '{cell_name}' of {pluck_len:.1f} seconds, about one repeat in three moving one note by a step), "
            f"the only plucked voice. Slit drum '{slit_name}' ({slit_pulses} pulses, {slit_count} hits) and woodblock '{block_name}' "
            f"({block_pulses} pulses, {block_count} hits) on the off-beats of the pluck's pulse, never within {int(MIN_GAP * 1000)} ms of a plucked note; "
            f"{len(clack_times)} shishi-odoshi clacks."
        ),
        (
            f"Drone on a cello section and a Juno pad (and a quiet one an octave up) with a vibrato of 4 to 6 cents that begins after a second or two (rate and phase of each note its own), path '{path_name}' "
            f"({len(segments)} stretches), washes '{wash_name}' ({used.get('wash:cello', 0) + used.get('wash:violin', 0) + used.get('wash:felt', 0)} placed, {WASH_SECONDS:.0f} seconds each)."
        ),
        (
            f"Chant leader '{lead_name}' ({lead_notes} long notes of a cello section in the scale, with rests); heartbeat '{heart_name}' at {bpm:.0f} per minute "
            f"({beats} beats in the busy stretches), {strikes} taiko strikes on the pulse, {ostinato_hits} hits of the ostinato and {rolls} accelerating rolls."
        ),
    ]
    if quiet:
        score.notes_about.append(
            f"A last soft strike at {int(quiet) // 60}:{int(quiet) % 60:02d} as the busy stretch ends."
        )
    return score
