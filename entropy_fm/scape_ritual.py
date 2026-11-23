"""
Ritual: one plucked line, a slit drum and a woodblock that answer it off the beat, over a deep choir drone, stretched string washes, a slow chant leader and taiko.

The generative system is a slow ceremony in a Japanese pentatonic (hirajoshi, in or kumoi, on the root of the day's key).
One short call-and-response cell, 6 to 7.2 seconds long on a pulse of eight, is plucked on a strumstick or a felt
piano (the room's own, the same every day, and there is never a second plucked voice), round and low-passed;
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

The day is spread over the piece as in the other styles. Who leads is the room's too (`lead`): the pluck states the motif in the cells of a phrase and plays every cell, the chant
leader answers every half phrase with three or four notes, or the slit drum plays its whole pattern on the pitches of
the motif, in the daytime build, peak and plateau, the others stepping back; a door or motion hands the next phrase to the
pluck, an appliance or the hob to the slit drum, a light, a cover or the media to the chant leader. The night is drone and washes alone; the pluck comes in
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
chooses the washes, its shape the drone path; the voices of the pluck, the leader and the drone pad are the room's own
and stay the same every day (`signature`). The root is the root of the day's key. The form of
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import lead, taiko
from .gen import dips, swells, weighted
from .harmony import (
    MOTIF_CHOICE,
    SIGNED_TAG_SPAN,
    closeness,
    develop_motif,
    motif_seed,
    motif_sites,
)
from .plan import LIGHT_MAX_LAYERS
from .score import (
    CURVE_RATE,
    LAYER_DRONE,
    LAYER_LEAD,
    LAYER_PLUCKS,
    LAYER_WASH,
    VOICE_TAPE_STRINGS,
    Layer,
    Note,
    Score,
)
from .signature import SigVoice, patch_of, signature, trim_layer, trim_velocity
from .synth import SEMITONES_PER_OCTAVE, smoothstep
from .timeline import SECONDS_PER_MINUTE

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
# how loud the notes of the plucked line are
PLUCK_LOUD = 0.62
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
LAYER_SUB = "sub"
LAYER_SLIT = "slit"
LAYER_BLOCK = "block"
LAYER_CLACK = "clack"
LAYER_TAIKO = "taiko"
LAYER_HEART = "heart"
VOICE_SUB_ROUND = "sub_round"
VOICE_WOODBLOCK = "woodblock"
VOICE_SLIT_DRUM = "slit_drum"
DRUM_HEART = "drum:heart"
HIT_ODAIKO = "odaiko"
HIT_SHIME = "shime"
HIT_RIM = "taiko_rim"
CLACK_EVENT = "event"
CLACK_CALM = "calm"
# the ostinato of the driving mode, a bar of eight pulses on the grid of the pluck, in the vocabulary of matsuri
# drumming: the don of the odaiko on the even pulses (the head of the bar the strongest), the ji of the shime on the
# odd ones, the doko of a quick pickup and a rim click: (pulse, where in it, drum, weight, activity needed)
OSTINATO: tuple[tuple[int, float, str, float, float], ...] = (
    (0, 0.0, HIT_ODAIKO, 0.7, 0.0),
    (2, 0.0, HIT_ODAIKO, 0.52, 0.0),
    (4, 0.0, HIT_ODAIKO, 0.62, 0.0),
    (6, 0.0, HIT_ODAIKO, 0.52, 0.0),
    (1, 0.0, HIT_SHIME, 0.4, 0.0),
    (3, 0.0, HIT_SHIME, 0.4, 0.0),
    (5, 0.0, HIT_SHIME, 0.4, 0.0),
    (7, 0.0, HIT_SHIME, 0.4, 0.0),
    (3, 0.25, HIT_SHIME, 0.3, 0.4),
    (7, 0.25, HIT_SHIME, 0.3, 0.4),
    (2, 0.25, HIT_RIM, 0.5, 0.6),
    (6, 0.25, HIT_RIM, 0.5, 0.6),
)
# a strong don falls on the head of every fourth bar, and the ostinato needs a section and a room this strong
PHRASE_BARS = 4
HEAD_LIFT = 0.22
DRIVE_FROM = 0.7
OSTINATO_GATE = 0.3
# the roll into a peak or plateau section lasts this many pulses; the leader of a driving piece waits this share of the gap
ROLL_PULSES = 4
# the strength of the first strike of a roll
ROLL_START_LEVEL = 0.18
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
# the last stretch of a piece is drone and washes: no cell ends within this many seconds of the end
PLUCK_TAIL = 21.0
# the chant leader sings within this range of notes
LEAD_RANGE = (53, 78)
# the room's motif on the pluck: never below this note (C4)
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
# the bus gains that make the plucked line the lead: the drone sits under it and the two woods answer a few dB below it
DRONE_GAIN = 0.24
SUB_GAIN = 0.22
PLUCK_GAIN = 1.5
WASH_GAIN = 0.7
SLIT_GAIN = 2.3
BLOCK_GAIN = 19.0
TAIKO_GAIN = 1.2
DRONE_OVERLAP = 4.5
# the three layers that can lead: the room picks its own, and a happening of the day may hand a phrase to another
LEADS = (LAYER_PLUCKS, LAYER_SLIT, LAYER_LEAD)
LEAD_BY_TRAIT = {
    lead.PERCUSSIVE: LAYER_PLUCKS,
    lead.RHYTHMIC: LAYER_SLIT,
    lead.SUSTAINED: LAYER_LEAD,
}
# the plucks that lead state the motif, developed, in these cells of a phrase and play every cell; the chant leader
# that leads answers at least every this share of a phrase, with this many notes, and the slit drum that leads plays its
# whole pattern on the pitches of the motif
LEAD_MOTIF_CELLS = (0, 4)
# what each layer is trimmed by while it leads, so that every lead sits a little over the others and no more
LEAD_TRIM_DB = {LAYER_PLUCKS: 0.0, LAYER_SLIT: 4.9, LAYER_LEAD: -1.0}
# the cell length that the day wants: the energy and the warmth, by these shares
CELL_ENERGY_SHARE = 0.7
CELL_WARMTH_SHARE = 0.3
# the lowest note of the pluck, the leader and the drone: the root is put in the octave above
PLUCK_LOWEST = 55
LEAD_LOWEST = 53
LEAD_STEPS_FROM = -2
LEAD_STEPS_TO = 7
DRONE_LOWEST = 38
AWAKE_DAYLIGHT_FLOOR = 0.4
AWAKE_DAYLIGHT_ACTIVITY = 0.6
AWAKE_EVENING = 0.3
BUSY_GATE_FROM = 0.34
BUSY_GATE_TO = 0.6
BASE_DAYLIGHT_FLOOR = 0.62
BASE_DAYLIGHT_ACTIVITY = 0.38
BASE_EVENING_SHARE = 0.6
BASE_EVENING_FLOOR = 0.6
BASE_EVENING_ACTIVITY = 0.4
BREATH_SECONDS_MIN = 11.5
BREATH_SECONDS_MAX = 13.5
SURGE_FLOOR = 0.55
SURGE_SHARE = 0.45
SURGE_SECONDS_MIN = 10.0
SURGE_SECONDS_MAX = 12.0
DRONE_MIN_TAIL = 35.0
FAR_MIN_SPAN = 12.0
FAR_SECONDS = 45.0
DRONE_VELOCITY = 0.5
DRONE_VELOCITY_POWER = 0.12
DRONE_STRINGS_VIBRATO = 6.0
DRONE_VOICE_SOFTER = 0.05
DRONE_VOICE_VIBRATO = 5.0
DRONE_OCTAVE_VELOCITY = 0.3
SUB_LOWEST = 26.0
SUB_VELOCITY = 0.6
WASH_VELOCITY = 0.6
WASH_GAP_MIN = 15.0
WASH_GAP_MAX = 24.0
PLUCK_PRESENCE_FLOOR = 0.78
PLUCK_PRESENCE_SHARE = 0.3
MOTIF_MIN_HOLD = 0.7
VELOCITY_MIN = 0.1
MIN_CELL_PRESENCE = 0.04
CELL_SWAP_CHANCE = 0.33
PLUCK_TIMING_JITTER = 0.008
FIRST_NOTE_ACCENT = 1.12
PLUCK_BRIGHT_FLOOR = 0.92
PLUCK_BRIGHT_SHARE = 0.12
HUMANISE_LOW = 0.92
HUMANISE_HIGH = 1.06
PLUCK_HOLD = 0.7
SLIT_TONES_FROM = 42
SLIT_TONES_TO = 56
BLOCK_TONES_FROM = 83
BLOCK_TONES_TO = 90
KNOCK_TONES_FROM = 39
KNOCK_TONES_TO = 50
CLACK_TICK_HOLD = 0.12
CLACK_TICK_VELOCITY_MIN = 0.3
CLACK_TICK_VELOCITY_MAX = 0.4
CLACK_KNOCK_HOLD = 0.14
CLACK_KNOCK_VELOCITY_MIN = 0.74
CLACK_KNOCK_VELOCITY_MAX = 0.86
SLIT_GATE_FROM = 0.2
SLIT_GATE_TO = 0.5
SLIT_LOUD = 0.45
SLIT_CLEAR = 0.45
BLOCK_GATE_FROM = 0.4
BLOCK_GATE_TO = 0.7
BLOCK_LOUD = 0.4
BLOCK_CLEAR = 0.3
MIN_HIT_PRESENCE = 0.05
ANSWER_BOOST = 1.1
ANSWER_CYCLE = 8
ANSWER_PLAIN = 0.95
HIT_VELOCITY_FLOOR = 0.8
HIT_VELOCITY_DRIVE = 0.25
NO_PHRASE_YET = -1000.0
CHANT_END_MARGIN = 14.0
CHANT_WANT_FLOOR = 0.45
CHANT_WANT_SHARE = 0.55
CHANT_MIN_HERE = 0.25
CHANT_STAY_CHANCE = 0.7
CHANT_START_GAP = 0.5
CHANT_GAP_JITTER = 0.9
CHANT_FINAL_HOLD_MIN = 5.0
CHANT_FINAL_HOLD_MAX = 8.0
CHANT_HOLD_MIN = 3.2
CHANT_HOLD_MAX = 5.0
CHANT_VELOCITY_MIN = 0.46
CHANT_VELOCITY_MAX = 0.58
CHANT_BRIGHT_FLOOR = 0.85
CHANT_BRIGHT_SHARE = 0.15
CHANT_VIBRATO = 6.0
CHANT_REST_MIN = 0.7
CHANT_REST_MAX = 1.9
ROLL_MAX_NIGHT = 0.6
ROLL_SPAN_MIN = 3.0
ROLL_SPAN_MAX = 4.2
EVENT_ROLL_STRENGTH = 0.92
SECTION_ROLL_STRENGTH = 0.9
NEAR_SECONDS = 5.0
BEAT_GATE = 0.55
TAIKO_VELOCITY = 0.52
TAIKO_VELOCITY_ACTIVITY = 0.26
TAIKO_VELOCITY_POWER = 0.14
EXTRA_RIM_CHANCE = 0.18
RIM_DELAY_BEATS = 1.5
RIM_VELOCITY = 0.5
EXTRA_ROLL_CHANCE = 0.32
ROLL_END_MARGIN = 12.0
ROLL_GAP_MIN = 30.0
ROLL_GAP_MAX = 60.0
ROLL_MIN_GATE = 0.7
ROLL_CHANCE = 0.55
ROLL_APART = 20.0
ROLL_LEAD_SECONDS = 3.4
ROLL_HEIGHT_STRENGTH = 0.85
ROLL_RESET = 90.0
OSTINATO_FLOOR = 0.7
OSTINATO_FORM = 0.3
OSTINATO_HUMANISE_LOW = 0.94
OSTINATO_HUMANISE_HIGH = 1.04
BUSY_END_MARGIN = 4.0
# the seconds before the end of the piece, or of a stretch of the drone, from which nothing new starts
PATTERN_END_MARGIN = 2.0
DRONE_SEGMENT_END_MARGIN = 3.0
TAIKO_END_MARGIN = 4.0
OSTINATO_END_MARGIN = 4.0
HEART_END_MARGIN = 3.0
# the patches of a taiko hit and of a heartbeat
TAIKO_PATCHES = 3
HEART_PATCHES = 3
CLOSE_END_MARGIN = 6.0
CLOSE_APART = 4.0
DRUM_HOLD = 0.2
DRUM_PITCH = 36.0
HEART_CLEAR_BEATS = 0.9
HEART_ROLL_BEFORE = 0.2
HEART_ROLL_AFTER_BEATS = 0.6
HEART_GATE = 0.3
HEART_VELOCITY = 0.22
HEART_VELOCITY_GATE = 0.12
HEART_VELOCITY_POWER = 0.04
HEART_ECHO_DELAY = 0.27
HEART_ECHO_VELOCITY = 0.6
DRONE_CUT_LOW = 520.0
DRONE_CUT_HIGH = 2100.0
DRONE_CUT_BRIGHT_FLOOR = 0.85
DRONE_CUT_BRIGHT_SHARE = 0.3
DRONE_DIP_DEPTH = 0.85
PLUCK_CUT_BASE = 2500.0
PLUCK_CUT_BRIGHT_FLOOR = 0.75
PLUCK_CUT_BRIGHT_SHARE = 0.45
WOOD_DIP_DEPTH = 0.6
BLOCK_CUT_BASE = 3300.0
BLOCK_CUT_BRIGHT_FLOOR = 0.8
BLOCK_CUT_BRIGHT_SHARE = 0.4
DRONE_PRESENCE_FLOOR = 0.9
DRONE_PRESENCE_POWER = 0.12
DRONE_PRESENCE_SWELL = 0.25
DRONE_PRESENCE_MAX = 1.25
WASH_PRESENCE_FLOOR = 0.45
WASH_PRESENCE_NIGHT = 0.45
WASH_PRESENCE_SWELL = 0.2
WASH_BUSY_DUCK = 0.25
DRONE_REVERB = 0.4
SUB_REVERB = 0.1
WASH_REVERB = 0.8
PLUCK_REVERB = 0.62
WOOD_REVERB = 0.5
CLACK_REVERB = 0.65
LEAD_REVERB = 0.6
TAIKO_REVERB = 0.5
HEART_REVERB = 0.25
SPACE_BASE = 0.5
SPACE_HUMID = 0.25
SPACE_SWELL = 0.2
SPACE_NIGHT = 0.1
SPACE_MAX = 0.95
COMPRESS_DB = -17.0
LEVEL_DB = -16.6
PENTATONIC_STEPS = 5
# the leader moves by one of these steps, down more often than up
CHANT_STEP_MOVES = (-2, -1, -1, 1)
MILLISECONDS = 1000
# a far centre of the drone is one of these steps, by these weights; the leader sings these notes to a phrase, and steps by these
FAR_STEPS = (7, 5)
FAR_WEIGHTS = (0.6, 0.4)
CHANT_SHORT_NOTES = (2, 3)
CHANT_NOTES_WEIGHTS = (0.45, 0.55)
NEXT_CHANT_STEPS = (0, 3, 5)
NEXT_CHANT_WEIGHTS = (0.5, 0.25, 0.25)
# the velocities of the soft strikes that roll after a taiko hit
EXTRA_ROLL_VELOCITIES = (0.45, 0.35, 0.55)
# a clack is pushed by these seconds until the wood is clear around it
CLACK_PUSHES = (0.0, 0.3, -0.3, 0.6)
PLUCK_AWAKE_FROM = 0.1
PLUCK_AWAKE_SPAN = 0.26
EVENT_CLACK_DELAY = 2.8
CLACK_MAX_NIGHT = 0.6
CLACK_MIN_AWAKE = 0.15
WANDER_MIN = 25.0
WANDER_MAX = 60.0
WANDER_END_MARGIN = 12.0
WANDER_MIN_AWAKE = 0.3
WANDER_MAX_ACTIVITY = 0.55
WANDER_CHANCE = 0.7
WANDER_GAP_MIN = 35.0
WANDER_GAP_MAX = 70.0
CLACK_PIECE_SECONDS = 100.0
CLACK_MAX = 5
CLACK_END_MARGIN = 8.0
CLACK_APART = 30.0
CLACK_MARGIN = 0.2
CLACK_FREE_SPAN = 0.3
# the wood comes in this many seconds after the first pluck, and the woodblock this long after the slit drum
WOOD_AFTER_SECONDS = 2.0
LEAD_CHANT_GAP = 0.5
LEAD_CHANT_NOTES = (3, 4)


def compose(plan: Plan) -> Score:
    """
    Make the score of a Ritual soundscape.

    :param plan: The curves and the deals of the day.
    """
    piece = _Piece(plan)
    _pick_tuning_and_mode(piece)
    _pick_voices_and_patterns(piece)
    _set_up_pitches(piece)
    _draw_curves(piece)
    _lay_out_form(piece)
    _plan_drone(piece)
    _play_drone(piece)
    _play_washes(piece)
    _plan_pluck(piece)
    _play_pluck(piece)
    _set_up_wood(piece)
    _place_clacks(piece)
    _play_wood_patterns(piece)
    _play_chant_leader(piece)
    _set_up_drums(piece)
    _roll_into_events(piece)
    _roll_into_sections(piece)
    _strike_on_beats(piece)
    _roll_at_height(piece)
    _play_ostinato(piece)
    _close_busy_stretch(piece)
    _write_taiko(piece)
    _play_heartbeat(piece)
    lead.back_off_followers(piece, frozenset({LAYER_SLIT, LAYER_LEAD}), "lead back")
    _add_layers(piece)
    return _assemble(piece)


def _lfo(count: int, period: float, phase: float) -> np.ndarray:
    """Return a slow 0 to 1 swing with `period` seconds, a value every tenth of a second."""
    t = (np.arange(count) + 0.5) / CURVE_RATE
    return np.asarray(0.5 - 0.5 * np.cos(2.0 * np.pi * (t / period + phase)))


@dataclass
class _Piece:
    """The working state of one Ritual piece, read and appended to by the sections of `compose`."""

    plan: Plan
    length: int = field(init=False)
    rng: np.random.Generator = field(init=False)
    light: bool = field(init=False)
    energy: float = field(init=False)
    root_pc: int = field(init=False)
    warmth: float = field(init=False)
    tuning_name: str = field(init=False)
    steps: tuple[int, ...] = field(init=False)
    mode: str = field(init=False)
    driving: bool = field(init=False)
    drive_energy: float = field(init=False)
    pluck_len: float = field(init=False)
    pluck_sig: SigVoice = field(init=False)
    lead_sig: SigVoice = field(init=False)
    drone_sig: SigVoice = field(init=False)
    cell_name: str = field(init=False)
    cell: Cell = field(init=False)
    slit_name: str = field(init=False)
    slit_pulses: int = field(init=False)
    slit_hits: tuple[Hit, ...] = field(init=False)
    block_name: str = field(init=False)
    block_pulses: int = field(init=False)
    block_hits: tuple[Hit, ...] = field(init=False)
    path_name: str = field(init=False)
    path: tuple[int, ...] = field(init=False)
    wash_name: str = field(init=False)
    wash_order: tuple[int, ...] = field(init=False)
    lead_name: str = field(init=False)
    lead_gap: tuple[float, float] = field(init=False)
    heart_name: str = field(init=False)
    heart_style: int = field(init=False)
    patch: int = field(init=False)
    pluck_base: int = field(init=False)
    lead_base: int = field(init=False)
    lead_steps: list[int] = field(init=False)
    drone_root: int = field(init=False)
    activity: np.ndarray = field(init=False)
    busy: np.ndarray = field(init=False)
    awake_base: np.ndarray = field(init=False)
    relative_activity: np.ndarray = field(init=False)
    swell: np.ndarray = field(init=False)
    breath: np.ndarray = field(init=False)
    surge: np.ndarray = field(init=False)
    notes: list[Note] = field(default_factory=list)
    sections: list[tuple[float, float, str, float]] = field(init=False)
    starts: list[float] = field(init=False)
    strength: list[float] = field(init=False)
    lead_grid: float = field(init=False)
    default_lead: str = field(init=False)
    leads: lead.Leads = field(init=False)
    lifted: set[int] = field(init=False)
    drums_on: bool = field(init=False)
    segments: list[tuple[float, float, int]] = field(init=False)
    slot: float = field(init=False)
    pluck_rng: np.random.Generator = field(init=False)
    base_index: list[int] = field(init=False)
    current: list[int] = field(init=False)
    occupied: set[int] = field(init=False)
    pluck_onsets: list[float] = field(default_factory=list)
    # (end of the response, the pitch it ended on, length of the loop)
    last_cells: list[tuple[float, float, float]] = field(default_factory=list)
    thin: float = field(init=False)
    cells: int = field(init=False)
    motif_cells: dict[int, _Statement] = field(init=False)
    lead_cells: dict[int, _Statement] = field(init=False)
    motif_played: int = 0
    slit_tones: list[float] = field(init=False)
    block_tones: list[float] = field(init=False)
    knock_tone: float = field(init=False)
    tick_tone: float = field(init=False)
    # every sound of the wood, for the gaps between them
    wood_hits: list[float] = field(default_factory=list)
    # how many notes of the motif the slit drum has played in a phrase it leads
    contour: dict[int, int] = field(default_factory=dict)
    slit_count: int = 0
    block_count: int = 0
    clack_times: list[float] = field(default_factory=list)
    lead_notes: int = 0
    bpm: float = field(init=False)
    beat: float = field(init=False)
    grid: float = field(init=False)
    drum_rng: np.random.Generator = field(init=False)
    taiko_hits: list[tuple[float, str, float]] = field(default_factory=list)
    big: list[float] = field(default_factory=list)
    event_starts: list[float] = field(default_factory=list)
    others: list[float] = field(default_factory=list)
    quiet: float = 0.0
    strikes: int = 0
    rolls: int = 0
    ostinato_hits: int = 0
    beats: int = 0
    layers: dict[str, Layer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Take what the plan already says; the first draw of the piece comes after it."""
        self.length = self.plan.length
        self.rng = self.plan.rng("ritual")
        self.light = self.plan.quality.max_layers <= LIGHT_MAX_LAYERS
        self.energy = self.plan.character.energy
        self.root_pc = self.plan.scale.root


class _Statement(NamedTuple):
    """The motif of one statement: its notes, their steps on the ladder and the pitch it starts from."""

    motif: tuple[tuple[int, float], ...]
    steps: list[int]
    base: int


class _WoodPart(NamedTuple):
    """One pattern of the wood and how it is played."""

    hits: tuple[Hit, ...]
    pulses: int
    tone_list: list[float]
    voice: str
    layer: str
    gate: np.ndarray
    loud: float
    rng: np.random.Generator
    clear_of_wood: float
    from_strength: float
    not_before: float


def _pick_tuning_and_mode(piece: _Piece) -> None:
    """Pick the tuning, the mode and the length of the pluck cell that the day suits."""
    plan = piece.plan
    character = plan.character
    energy = piece.energy
    piece.warmth = float(np.clip(character.warmth - LATE_DARKEN * character.late_night, -1.0, 1.0))
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
    piece.tuning_name, piece.steps, _ = plan.pick(
        "tuning", TUNINGS, [closeness(bright, brightness) for _, _, bright in TUNINGS]
    )
    piece.mode, _ = plan.pick(
        "ritual mode", MODES, [closeness(wanted, energy) for _, wanted in MODES], MODE_TOLERANCE
    )
    piece.driving = piece.mode == DRIVING
    piece.drive_energy = float(min(energy + MODE_LIFT[piece.mode], 1.0))
    lengths = MODE_LENGTHS[piece.mode]
    _, piece.pluck_len = plan.pick(
        "pluck cell length", lengths, [closeness(LENGTH_ENERGY[n], energy) for n, _ in lengths]
    )


def _pick_voices_and_patterns(piece: _Piece) -> None:
    """Pick the voices of the room and the cell, the patterns, the drone path, the washes, the leader and the heartbeat."""
    plan = piece.plan
    energy = piece.energy
    warmth = piece.warmth
    piece.pluck_sig = signature(plan, LAYER_PLUCKS, "pluck voice")
    piece.lead_sig = signature(plan, LAYER_LEAD, "leader voice")
    piece.drone_sig = signature(plan, LAYER_DRONE, "drone voice")
    cell_wanted = CELL_ENERGY_SHARE * energy + CELL_WARMTH_SHARE * (0.5 + 0.5 * warmth)
    piece.cell_name, piece.cell = plan.pick(
        "pluck cell", CELLS, [closeness(CELL_ENERGY[n], cell_wanted) for n, _ in CELLS]
    )
    piece.slit_name, piece.slit_pulses, piece.slit_hits = plan.pick(
        "slit drum pattern",
        SLIT_PATTERNS,
        [closeness(SLIT_ENERGY[n], energy) for n, _, _ in SLIT_PATTERNS],
    )
    piece.block_name, piece.block_pulses, piece.block_hits = plan.pick(
        "woodblock pattern",
        BLOCK_PATTERNS,
        [closeness(BLOCK_ENERGY[n], energy) for n, _, _ in BLOCK_PATTERNS],
    )
    piece.path_name, piece.path = plan.pick(
        "drone path",
        DRONE_PATHS,
        [
            closeness(PATH_ENERGY[n], energy)
            + SHAPE_BONUS * (plan.character.shape in PATH_SHAPES[n])
            for n, _ in DRONE_PATHS
        ],
    )
    piece.wash_name, piece.wash_order = plan.pick(
        "wash set",
        WASH_SETS,
        [closeness(WASH_WARMTH[n], warmth, SIGNED_TAG_SPAN) for n, _ in WASH_SETS],
    )
    piece.lead_name, piece.lead_gap = plan.pick(
        "chant leader",
        LEAD_PRESENCE,
        [closeness(LEAD_ENERGY[n], energy) for n, _ in LEAD_PRESENCE],
    )
    piece.heart_name, piece.heart_style = plan.pick(
        "heartbeat", HEARTBEATS, [closeness(HEART_ENERGY[n], energy) for n, _ in HEARTBEATS]
    )
    piece.patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["root"] = str(piece.root_pc)


def _set_up_pitches(piece: _Piece) -> None:
    """Set the lowest note of the pluck, the leader and the drone, and the steps the leader sings."""
    root_pc = piece.root_pc
    piece.pluck_base = PLUCK_LOWEST + (root_pc - PLUCK_LOWEST) % SEMITONES_PER_OCTAVE
    piece.lead_base = LEAD_LOWEST + (root_pc - LEAD_LOWEST) % SEMITONES_PER_OCTAVE
    piece.lead_steps = [
        i
        for i in range(LEAD_STEPS_FROM, LEAD_STEPS_TO)
        if LEAD_RANGE[0] <= _ladder(piece, i, piece.lead_base) <= LEAD_RANGE[1]
    ]
    piece.drone_root = DRONE_LOWEST + (root_pc - DRONE_LOWEST) % SEMITONES_PER_OCTAVE


def _draw_curves(piece: _Piece) -> None:
    """Draw the curves of the day: how awake and busy the room is, and the slow swings of the drone."""
    plan = piece.plan
    rng = piece.rng
    count = plan.count
    activity = plan.activity
    piece.activity = activity
    # how awake the room is: the daylight, lifted by activity; a little of the evening stays while the house is lively
    awake = np.clip(
        plan.daylight * (AWAKE_DAYLIGHT_FLOOR + AWAKE_DAYLIGHT_ACTIVITY * activity)
        + AWAKE_EVENING * plan.evening * activity,
        0.0,
        1.0,
    )
    piece.busy = plan.gate(awake, BUSY_GATE_FROM, BUSY_GATE_TO)
    # the plucks and the chant are the base of the day: a room that is calm is awake enough for the pluck by day
    # and in the evening, and the house moves how loud it is and how often it is moved
    piece.awake_base = np.clip(
        plan.daylight * (BASE_DAYLIGHT_FLOOR + BASE_DAYLIGHT_ACTIVITY * activity)
        + BASE_EVENING_SHARE
        * plan.evening
        * (BASE_EVENING_FLOOR + BASE_EVENING_ACTIVITY * activity),
        0.0,
        1.0,
    )
    piece.relative_activity = np.clip(
        activity / max(float(activity.max()), MIN_PEAK_ACTIVITY), 0.0, 1.0
    )
    piece.swell = swells(plan)
    piece.breath = _lfo(
        count, float(rng.uniform(BREATH_SECONDS_MIN, BREATH_SECONDS_MAX)), float(rng.uniform())
    )
    piece.surge = SURGE_FLOOR + SURGE_SHARE * _lfo(
        count, float(rng.uniform(SURGE_SECONDS_MIN, SURGE_SECONDS_MAX)), float(rng.uniform())
    )


def _lay_out_form(piece: _Piece) -> None:
    """Lay out the form: a cell is a bar, so the sections change every 8 cells; the strongest anomaly lifts the next section."""
    plan = piece.plan
    piece.sections = plan.sections(piece.pluck_len)
    sections = piece.sections
    piece.starts = [start for start, _, _, _ in sections]
    piece.strength = [intensity for _, _, _, intensity in sections]
    if plan.events:
        strongest, _ = max(plan.events, key=lambda event: event[1])
        after = bisect_right(piece.starts, strongest)
        if after < len(sections):
            piece.strength[after] = max(piece.strength[after], EVENT_FORM)
    piece.lead_grid = plan.form_grid(piece.pluck_len)
    piece.default_lead = lead.default_lead(plan, LEADS)
    piece.leads = lead.phrase_leads(
        plan,
        piece.lead_grid,
        piece.default_lead,
        lead.daytime_phrases(plan, sections, piece.lead_grid),
        LEAD_BY_TRAIT,
    )
    piece.lifted = {i for i, (_, _, _, base) in enumerate(sections) if piece.strength[i] > base}
    piece.drums_on = piece.energy >= DRUMS_ENERGY


def _plan_drone(piece: _Piece) -> None:
    """Split the piece into stretches of the drone along the path; at an anomaly the root goes to the fifth or the fourth."""
    plan = piece.plan
    rng = piece.rng
    length = piece.length
    path = piece.path
    segments: list[tuple[float, float, int]] = []
    clock, place = 0.0, 0
    while clock < length:
        span = float(
            rng.uniform(
                DRONE_SPAN_QUIET[0]
                + (DRONE_SPAN_BUSY[0] - DRONE_SPAN_QUIET[0]) * piece.drive_energy,
                DRONE_SPAN_QUIET[1]
                + (DRONE_SPAN_BUSY[1] - DRONE_SPAN_QUIET[1]) * piece.drive_energy,
            )
        )
        end = min(clock + span, float(length))
        if length - end < DRONE_MIN_TAIL:
            end = float(length)
        segments.append((clock, end, path[place % len(path)] if clock > 0.0 else 0))
        clock, place = end, place + 1
    for second, _ in plan.events:
        for k, (start, end, _interval) in enumerate(segments):
            if start <= second < end and end - second > FAR_MIN_SPAN:
                far = int(weighted(rng, FAR_STEPS, FAR_WEIGHTS))
                segments[k] = (start, second, segments[k][2])
                segments.insert(k + 1, (second, min(second + FAR_SECONDS, end), far))
                if second + FAR_SECONDS < end:
                    segments.insert(k + 2, (second + FAR_SECONDS, end, segments[k][2]))
                break
    piece.segments = segments


def _play_drone(piece: _Piece) -> None:
    """Play the drone: a cello section and a Juno pad with a vibrato of their own, a quiet pad an octave up, and a round sub."""
    plan = piece.plan
    drone_sig = piece.drone_sig
    notes = piece.notes
    index = 0
    for start, end, interval in piece.segments:
        second = start
        while second < end - DRONE_SEGMENT_END_MARGIN:
            dur = float(min(DRONE_STEP + DRONE_OVERLAP, end + DRONE_OVERLAP - second))
            pitch = float(piece.drone_root + interval)
            vel = DRONE_VELOCITY + DRONE_VELOCITY_POWER * plan.at(plan.power, second)
            notes.append(
                Note(
                    second,
                    dur,
                    pitch,
                    vel,
                    VOICE_TAPE_STRINGS,
                    LAYER_DRONE,
                    index % 2,
                    DRONE_STRINGS_VIBRATO,
                )
            )
            notes.append(
                Note(
                    second,
                    dur,
                    pitch,
                    trim_velocity(drone_sig, vel - DRONE_VOICE_SOFTER),
                    drone_sig.voice,
                    LAYER_DRONE,
                    patch_of(drone_sig, (index + 1) % 2),
                    DRONE_VOICE_VIBRATO,
                )
            )
            if not piece.light:
                notes.append(
                    Note(
                        second,
                        dur,
                        pitch + SEMITONES_PER_OCTAVE,
                        trim_velocity(drone_sig, DRONE_OCTAVE_VELOCITY),
                        drone_sig.voice,
                        LAYER_DRONE,
                        patch_of(drone_sig, index % 2),
                    )
                )
            sub = (
                pitch - SEMITONES_PER_OCTAVE
                if pitch - SEMITONES_PER_OCTAVE >= SUB_LOWEST
                else pitch
            )
            notes.append(Note(second, dur, sub, SUB_VELOCITY, VOICE_SUB_ROUND, LAYER_SUB, 1, 0.0))
            second += DRONE_STEP
            index += 1


def _play_washes(piece: _Piece) -> None:
    """Play the washes: stretched sections that overlap and replace each other, drawn from a few recipes so that each is made once."""
    if piece.light:
        return
    rng = piece.rng
    recipes = [WASH_RECIPES[i] for i in piece.wash_order]
    second, turn = 0.0, int(rng.integers(0, len(recipes)))
    while second < piece.length:
        voice, step = recipes[turn % len(recipes)]
        pitch = float(piece.drone_root + SEMITONES_PER_OCTAVE + step)
        piece.notes.append(
            Note(second, WASH_SECONDS, pitch, WASH_VELOCITY, voice, LAYER_WASH, 0, 0.0)
        )
        second += float(rng.uniform(WASH_GAP_MIN, WASH_GAP_MAX))
        turn += 1


def _plan_pluck(piece: _Piece) -> None:
    """Set up the pluck: one cell that, on a quiet day, plays sparsely, and the cells of the room's motif that replace it."""
    plan = piece.plan
    piece.slot = piece.pluck_len / CELL_SLOTS
    piece.pluck_rng = plan.rng("loop", "pluck")
    piece.base_index = [step for _, step in piece.cell]
    piece.current = list(piece.base_index)
    piece.occupied = {where for where, _ in piece.cell}
    piece.thin = PLUCK_DENSITY[0] + (PLUCK_DENSITY[1] - PLUCK_DENSITY[0]) * piece.drive_energy
    piece.cells = int((piece.length - PLUCK_TAIL) / piece.pluck_len)
    piece.motif_cells = {
        k: _motif_on_ladder(piece, number)
        for number, k in enumerate(motif_sites(_awake_firsts(piece)))
    }
    # the motif of the room also opens the phrases that the plucks lead, and answers in the middle of them
    piece.lead_cells = {
        round(number * piece.lead_grid / piece.pluck_len) + cell: _motif_on_ladder(
            piece, len(MOTIF_ROLES) + len(LEAD_MOTIF_CELLS) * number + place
        )
        for number, who in enumerate(piece.leads.phrases)
        if who == LAYER_PLUCKS
        for place, cell in enumerate(LEAD_MOTIF_CELLS)
    }


def _play_pluck(piece: _Piece) -> None:
    """Play the pluck: the cell comes in as the day wakes and leaves as the evening settles; the only plucked voice."""
    pluck_len = piece.pluck_len
    for k in range(piece.cells):
        second = k * pluck_len
        if second + pluck_len > piece.length - PLUCK_TAIL:
            break
        presence = _pluck_presence(piece, second)
        if k in piece.motif_cells or k in piece.lead_cells:
            piece.motif_played += k in piece.motif_cells
            _play_motif_cell(
                piece, second, presence, piece.motif_cells.get(k) or piece.lead_cells[k]
            )
            continue
        _play_cell(piece, second, presence)
    piece.pluck_onsets.sort()


def _play_motif_cell(piece: _Piece, second: float, presence: float, statement: _Statement) -> None:
    """Play the motif of the room in place of the cell."""
    motif, motif_steps, motif_base = statement
    slot = piece.slot
    clock = 0.0
    pitch = float(motif_base)
    for (_, beats), step in zip(motif, motif_steps, strict=True):
        when = second + clock * slot
        pitch = _ladder(piece, step, motif_base)
        vel = (
            PLUCK_LOUD
            * MOTIF_ACCENT
            * (PLUCK_PRESENCE_FLOOR + PLUCK_PRESENCE_SHARE * max(presence, MOTIF_AWAKE))
        )
        piece.notes.append(
            Note(
                when,
                max(MOTIF_MIN_HOLD, beats * slot),
                pitch,
                float(np.clip(vel, VELOCITY_MIN, 1.0)),
                piece.pluck_sig.voice,
                LAYER_PLUCKS,
                patch_of(piece.pluck_sig, piece.patch),
            )
        )
        piece.pluck_onsets.append(when)
        clock += beats
    piece.last_cells.append((second + clock * slot, pitch, piece.pluck_len))


def _play_cell(piece: _Piece, second: float, presence: float) -> None:
    """Play the cell when the room is awake enough, or when the plucks lead."""
    who = piece.leads.at(second)
    backing = lead.LEAD_BACK_KEEP if who not in {None, LAYER_PLUCKS} else 1.0
    if presence > MIN_CELL_PRESENCE and (
        who == LAYER_PLUCKS
        or piece.pluck_rng.random()
        < presence**0.5 * piece.thin * (0.5 + 0.5 * _form_at(piece, second)) * backing
    ):
        _state_cell(piece, second, presence)


def _state_cell(piece: _Piece, second: float, presence: float) -> None:
    """Play the cell once; about one repeat in three moves a note by a step, never further than one from where it began."""
    plan = piece.plan
    pluck_rng = piece.pluck_rng
    current = piece.current
    base_index = piece.base_index
    slot = piece.slot
    if pluck_rng.random() < CELL_SWAP_CHANCE:
        pick = int(pluck_rng.integers(0, len(current)))
        current[pick] = int(
            np.clip(
                current[pick] + int(pluck_rng.choice((-1, 1))),
                base_index[pick] - 1,
                base_index[pick] + 1,
            )
        )
    pitch = float(piece.pluck_base)
    for number, ((where, _), step) in enumerate(zip(piece.cell, current, strict=True)):
        when = (
            second
            + where * slot
            + float(pluck_rng.uniform(-PLUCK_TIMING_JITTER, PLUCK_TIMING_JITTER))
        )
        accent = FIRST_NOTE_ACCENT if number == 0 else 1.0
        vel = (
            PLUCK_LOUD
            * accent
            * (PLUCK_PRESENCE_FLOOR + PLUCK_PRESENCE_SHARE * presence)
            * (PLUCK_BRIGHT_FLOOR + PLUCK_BRIGHT_SHARE * plan.at(plan.bright, when))
            * float(pluck_rng.uniform(HUMANISE_LOW, HUMANISE_HIGH))
        )
        pitch = _ladder(piece, step, piece.pluck_base)
        piece.notes.append(
            Note(
                when,
                PLUCK_HOLD,
                pitch,
                float(np.clip(vel, VELOCITY_MIN, 1.0)),
                piece.pluck_sig.voice,
                LAYER_PLUCKS,
                patch_of(piece.pluck_sig, piece.patch),
            )
        )
        piece.pluck_onsets.append(when)
    piece.last_cells.append((second + (piece.cell[-1][0] + 1.0) * slot, pitch, piece.pluck_len))


def _set_up_wood(piece: _Piece) -> None:
    """Set the tones of the wood. Everything of the wood is placed on the pulse of the pluck's grid, on its off-beats, and checked against the notes that are plucked, so that nothing of the wood ever sounds with a pluck."""
    piece.slit_tones = _tones(piece, SLIT_TONES_FROM, SLIT_TONES_TO)
    piece.block_tones = _tones(piece, BLOCK_TONES_FROM, BLOCK_TONES_TO)
    piece.knock_tone = _tones(piece, KNOCK_TONES_FROM, KNOCK_TONES_TO)[0]
    piece.tick_tone = piece.block_tones[0]


def _place_clacks(piece: _Piece) -> None:
    """Place the shishi-odoshi: a soft tick of the woodblock, then a hollow knock of the slit drum, lowered; at the quiet after an anomaly or a shower and at a few calm moments, a few times in a piece and never close to each other."""
    clack_rng = piece.plan.rng("clack")
    _choose_clacks(piece, _clack_candidates(piece, clack_rng))
    for calm in _calmest_moments(piece):
        if piece.clack_times:
            break
        _choose_clacks(piece, [calm])
    if piece.light:
        return
    for when in piece.clack_times:
        piece.notes.append(
            Note(
                when,
                CLACK_TICK_HOLD,
                piece.tick_tone,
                float(clack_rng.uniform(CLACK_TICK_VELOCITY_MIN, CLACK_TICK_VELOCITY_MAX)),
                VOICE_WOODBLOCK,
                LAYER_CLACK,
            )
        )
        piece.notes.append(
            Note(
                when + CLACK_DELAY,
                CLACK_KNOCK_HOLD,
                piece.knock_tone,
                float(clack_rng.uniform(CLACK_KNOCK_VELOCITY_MIN, CLACK_KNOCK_VELOCITY_MAX)),
                VOICE_SLIT_DRUM,
                LAYER_CLACK,
            )
        )
        piece.wood_hits.extend((when, when + CLACK_DELAY))


def _play_wood_patterns(piece: _Piece) -> None:
    """Play the slit drum, and the woodblock when the quality allows it, on the pulses of the pluck."""
    plan = piece.plan
    slit_gate = plan.gate(piece.awake_base, SLIT_GATE_FROM, SLIT_GATE_TO)
    piece.slit_count = _lay_pattern(
        piece,
        _WoodPart(
            piece.slit_hits,
            piece.slit_pulses,
            piece.slit_tones,
            VOICE_SLIT_DRUM,
            LAYER_SLIT,
            slit_gate,
            SLIT_LOUD,
            plan.rng("slit"),
            SLIT_CLEAR,
            SLIT_FROM,
            min(piece.pluck_onsets, default=0.0) + WOOD_AFTER_SECONDS,
        ),
    )
    first_slit = min((n.t for n in piece.notes if n.layer == LAYER_SLIT), default=0.0)
    if not piece.light:
        piece.block_count = _lay_pattern(
            piece,
            _WoodPart(
                piece.block_hits,
                piece.block_pulses,
                piece.block_tones,
                VOICE_WOODBLOCK,
                LAYER_BLOCK,
                plan.gate(piece.awake_base, BLOCK_GATE_FROM, BLOCK_GATE_TO),
                BLOCK_LOUD,
                plan.rng("block"),
                BLOCK_CLEAR,
                BLOCK_FROM,
                first_slit + WOOD_AFTER_SECONDS,
            ),
        )


def _lay_pattern(piece: _Piece, part: _WoodPart) -> int:
    """Place a pattern on the pulses of the pluck, loop after loop; return how many hits were played."""
    plan = piece.plan
    slot = piece.slot
    rest_share = REST_SHARE[0] + (REST_SHARE[1] - REST_SHARE[0]) * piece.drive_energy
    played = 0
    loop = 0
    while loop * part.pulses * slot < piece.length - PATTERN_END_MARGIN:
        # now and then a whole loop is left out, so that the pattern breathes
        calm = 1.0 - 2.0 * plan.at(piece.relative_activity, loop * part.pulses * slot)
        resting = part.rng.random() < rest_share * (1.0 + REST_ACTIVITY * calm)
        for hit in part.hits:
            when = (loop * part.pulses + hit[0] + hit[1]) * slot
            played += _lay_hit(piece, part, when, hit, resting=resting)
        loop += 1
    return played


def _lay_hit(piece: _Piece, part: _WoodPart, when: float, hit: Hit, *, resting: bool) -> bool:
    """Place one hit of a pattern unless it is left out; return whether it was played."""
    plan = piece.plan
    pulse, _, tone, need = hit
    # the slit drum that leads plays every hit of its pattern, on the pitches of the motif
    melodic = part.layer == LAYER_SLIT and piece.leads.at(when) == LAYER_SLIT
    if (resting and not melodic) or when >= piece.length - PATTERN_END_MARGIN:
        return False
    here = plan.at(part.gate, when)
    drive = _drive_at(piece, when)
    roll = part.rng.random()
    if (
        here < MIN_HIT_PRESENCE
        or when < part.not_before
        or not _free(piece, when, MIN_GAP, part.clear_of_wood)
    ):
        return False
    if not melodic and (
        _form_at(piece, when) + FORM_LIFT * plan.at(piece.relative_activity, when)
        < part.from_strength
        or roll > here**0.5
        or need * EXTRA_NEED > drive + EXTRA_REACH * (piece.drive_energy - 0.5)
    ):
        return False
    pitch_index = _melodic_index(piece, when) if melodic else tone
    # a hit that falls where the pluck rests is the answer: a little firmer than one that fills a gap
    answer = ANSWER_BOOST if pulse % ANSWER_CYCLE not in piece.occupied else ANSWER_PLAIN
    vel = (
        part.loud
        * answer
        * (HIT_VELOCITY_FLOOR + HIT_VELOCITY_DRIVE * drive)
        * float(part.rng.uniform(HUMANISE_LOW, HUMANISE_HIGH))
    )
    piece.notes.append(
        Note(
            when,
            0.5,
            part.tone_list[pitch_index % len(part.tone_list)],
            float(np.clip(vel, VELOCITY_MIN, 1.0)),
            part.voice,
            part.layer,
        )
    )
    piece.wood_hits.append(when)
    return True


def _play_chant_leader(piece: _Piece) -> None:
    """Play the chant leader: a cello section answers the end of a pluck cell with two or three long notes and rests."""
    if piece.light or not piece.last_cells:
        return
    lead_rng = piece.plan.rng("lead")
    last_phrase = NO_PHRASE_YET
    phrase_end = 0.0
    previous = 1
    if piece.driving:
        piece.lead_gap = (
            piece.lead_gap[0] * LEAD_DRIVE_SCALE,
            piece.lead_gap[1] * LEAD_DRIVE_SCALE,
        )
    for end, pitch, _ in sorted(piece.last_cells):
        leading = piece.leads.at(end) == LAYER_LEAD
        if _chant_waits(piece, lead_rng, end, (last_phrase, phrase_end), leading=leading):
            continue
        phrase_end, previous = _sing_phrase(piece, lead_rng, end, pitch, previous, leading=leading)
        last_phrase = end


def _chant_waits(
    piece: _Piece,
    lead_rng: np.random.Generator,
    end: float,
    last: tuple[float, float],
    *,
    leading: bool,
) -> bool:
    """Return True when the leader keeps quiet after the cell that ends at `end`; `last` is where the last phrase began and ended."""
    plan = piece.plan
    last_phrase, phrase_end = last
    lead_gap = piece.lead_gap
    gap_least = LEAD_CHANT_GAP * piece.lead_grid if leading else lead_gap[0]
    if (
        end - last_phrase < gap_least
        or end < phrase_end
        or end > piece.length - CHANT_END_MARGIN
        or _form_at(piece, end) < LEAD_FROM
    ):
        return True
    here = plan.at(piece.awake_base, end)
    wanted = CHANT_WANT_FLOOR + CHANT_WANT_SHARE * min(
        1.0, here + 0.5 * plan.at(plan.lights, end) + 0.5 * plan.at(plan.evening, end)
    )
    wait = lead_gap[0] + (lead_gap[1] - lead_gap[0]) * float(lead_rng.random())
    return here < CHANT_MIN_HERE or (
        not leading and (end - last_phrase < wait or lead_rng.random() > wanted)
    )


def _sing_phrase(
    piece: _Piece,
    lead_rng: np.random.Generator,
    end: float,
    pitch: float,
    previous: int,
    *,
    leading: bool,
) -> tuple[float, int]:
    """Sing one phrase; return where it ended and the step it ended on."""
    plan = piece.plan
    lead_sig = piece.lead_sig
    lead_base = piece.lead_base
    # the pitch of the answer: the step of the lead's scale nearest to where the pluck ended, then a stepwise walk home
    reach = [
        (
            abs(
                _ladder(piece, i, lead_base)
                - (pitch - SEMITONES_PER_OCTAVE * round((pitch - lead_base) / SEMITONES_PER_OCTAVE))
            ),
            i,
        )
        for i in piece.lead_steps
    ]
    current_step = min(reach)[1] if lead_rng.random() < CHANT_STAY_CHANCE else previous
    count_in_phrase = int(
        weighted(lead_rng, LEAD_CHANT_NOTES if leading else CHANT_SHORT_NOTES, CHANT_NOTES_WEIGHTS)
    )
    clock = end + CHANT_START_GAP + float(lead_rng.uniform(0.0, CHANT_GAP_JITTER))
    if piece.driving:
        clock = (end // piece.pluck_len + 1.0) * piece.pluck_len + 2.0 * piece.slot
    for number in range(count_in_phrase):
        final = number == count_in_phrase - 1
        if final or number > 0:
            current_step = _next_chant_step(piece, lead_rng, current_step, final=final)
        dur = float(
            lead_rng.uniform(CHANT_FINAL_HOLD_MIN, CHANT_FINAL_HOLD_MAX)
            if final
            else lead_rng.uniform(CHANT_HOLD_MIN, CHANT_HOLD_MAX)
        )
        vel = float(lead_rng.uniform(CHANT_VELOCITY_MIN, CHANT_VELOCITY_MAX)) * (
            CHANT_BRIGHT_FLOOR + CHANT_BRIGHT_SHARE * plan.at(plan.bright, clock)
        )
        piece.notes.append(
            Note(
                clock,
                dur,
                _ladder(piece, current_step, lead_base),
                vel,
                lead_sig.voice,
                LAYER_LEAD,
                patch_of(lead_sig, piece.patch),
                CHANT_VIBRATO,
            )
        )
        piece.lead_notes += 1
        previous = current_step
        clock += dur + float(lead_rng.uniform(CHANT_REST_MIN, CHANT_REST_MAX))
    return clock, previous


def _set_up_drums(piece: _Piece) -> None:
    """Set the pulse of the heartbeat and the taiko, which play in the busy stretches only."""
    piece.bpm = BEATS_PER_MINUTE[0] + (BEATS_PER_MINUTE[1] - BEATS_PER_MINUTE[0]) * piece.energy
    if piece.driving:
        piece.bpm = SECONDS_PER_MINUTE / piece.slot
    piece.beat = SECONDS_PER_MINUTE / piece.bpm
    piece.drum_rng = piece.plan.rng("taiko")
    piece.grid = 0.0 if piece.driving else float(piece.rng.uniform(0.0, piece.beat))


def _roll_into_events(piece: _Piece) -> None:
    """Make an anomaly that is not in the dead of the night a roll into one big strike."""
    plan = piece.plan
    for second, _ in plan.events:
        if plan.at(plan.night, second) < ROLL_MAX_NIGHT:
            end = _on_grid(piece, second)
            span = float(piece.drum_rng.uniform(ROLL_SPAN_MIN, ROLL_SPAN_MAX))
            _roll_into_strike(piece, end - span, end, EVENT_ROLL_STRENGTH)


def _roll_into_sections(piece: _Piece) -> None:
    """Roll into the head of every peak or plateau section."""
    if not piece.driving:
        return
    for start, _, role, _ in piece.sections[1:]:
        if role in MOTIF_ROLES and _drum_gate(piece, start) > OSTINATO_GATE:
            _roll_into_strike(piece, start - ROLL_PULSES * piece.slot, start, SECTION_ROLL_STRENGTH)


def _strike_on_beats(piece: _Piece) -> None:
    """Strike on the beat by chance, rarer when the house is calm and never close to another strike."""
    piece.others = [*piece.big]
    beat = piece.beat
    beat_time = piece.grid
    while beat_time < piece.length - TAIKO_END_MARGIN:
        gate = _drum_gate(piece, beat_time)
        near = any(abs(beat_time - other) < NEAR_SECONDS for other in piece.others)
        inside_roll = _in_roll(piece, beat_time, 0.5, beat)
        if gate > BEAT_GATE and not near and not inside_roll and not piece.driving:
            _strike_beat(piece, beat_time, gate)
        beat_time += beat


def _strike_beat(piece: _Piece, beat_time: float, gate: float) -> None:
    """Strike the taiko on this beat by chance, now and then with a rim click or a short run of the shime."""
    plan = piece.plan
    drum_rng = piece.drum_rng
    beat = piece.beat
    chance = (
        TAIKO_CHANCE[0]
        + (TAIKO_CHANCE[1] - TAIKO_CHANCE[0])
        * piece.energy
        * (0.5 + 0.5 * plan.at(piece.activity, beat_time))
    ) * gate
    if drum_rng.random() < chance:
        vel = (
            TAIKO_VELOCITY
            + TAIKO_VELOCITY_ACTIVITY * plan.at(piece.activity, beat_time)
            + TAIKO_VELOCITY_POWER * plan.at(plan.power, beat_time)
        )
        piece.taiko_hits.append((beat_time, HIT_ODAIKO, float(vel)))
        piece.others.append(beat_time)
        piece.strikes += 1
        extra = drum_rng.random()
        if extra < EXTRA_RIM_CHANCE:
            piece.taiko_hits.append((beat_time + beat * RIM_DELAY_BEATS, HIT_RIM, RIM_VELOCITY))
        elif extra < EXTRA_ROLL_CHANCE:
            for k, v in enumerate(EXTRA_ROLL_VELOCITIES):
                piece.taiko_hits.append((beat_time + beat * (1.0 + 0.5 * k), HIT_SHIME, v))


def _roll_at_height(piece: _Piece) -> None:
    """Roll unprompted at the height of a busy stretch, not more often than every minute and a half."""
    drum_rng = piece.drum_rng
    length = piece.length
    roll_clock = 0.0
    while roll_clock < length - ROLL_END_MARGIN:
        roll_clock += float(drum_rng.uniform(ROLL_GAP_MIN, ROLL_GAP_MAX))
        if (
            roll_clock < length - ROLL_END_MARGIN
            and _drum_gate(piece, roll_clock) > ROLL_MIN_GATE
            and drum_rng.random() < ROLL_CHANCE
            and not any(abs(roll_clock - other) < ROLL_APART for other in piece.others)
        ):
            end = _on_grid(piece, roll_clock)
            _roll_into_strike(piece, end - ROLL_LEAD_SECONDS, end, ROLL_HEIGHT_STRENGTH)
            piece.others.append(end)
            roll_clock += ROLL_RESET


def _play_ostinato(piece: _Piece) -> None:
    """Play the ostinato, on the pulse of the pluck: it builds with the section, calls in more hits in the busy part and stops in a release or an outro."""
    plan = piece.plan
    slot = piece.slot
    ostinato_rng = plan.rng("ostinato")
    bar = 0
    while piece.driving and bar * piece.pluck_len < piece.length - OSTINATO_END_MARGIN:
        for pulse, frac, drum, weight, need in OSTINATO:
            when = (bar * CELL_SLOTS + pulse + frac) * slot
            form = _form_at(piece, when)
            if _ostinato_rests(piece, when, form, need):
                continue
            head = HEAD_LIFT if pulse == 0 and bar % PHRASE_BARS == 0 else 0.0
            vel = (
                (weight + head)
                * (OSTINATO_FLOOR + OSTINATO_FORM * form)
                * float(ostinato_rng.uniform(OSTINATO_HUMANISE_LOW, OSTINATO_HUMANISE_HIGH))
            )
            piece.taiko_hits.append((when, drum, vel))
            piece.ostinato_hits += 1
        bar += 1


def _close_busy_stretch(piece: _Piece) -> None:
    """Strike one last soft strike as the busy stretch ends."""
    beat = piece.beat
    length = piece.length
    awake_beats = [
        t
        for t in np.arange(piece.grid, length - BUSY_END_MARGIN, beat)
        if _drum_gate(piece, float(t)) > BEAT_GATE
    ]
    if piece.taiko_hits and awake_beats:
        closing = float(awake_beats[-1]) + 4.0 * beat
        if closing < length - CLOSE_END_MARGIN and not any(
            abs(closing - other) < CLOSE_APART for other in piece.others
        ):
            piece.taiko_hits.append((closing, HIT_ODAIKO, 0.5))
            piece.others.append(closing)
            piece.quiet = closing


def _write_taiko(piece: _Piece) -> None:
    """Turn the hits of the taiko into notes."""
    for when, hit, vel in piece.taiko_hits:
        piece.notes.append(
            Note(
                when,
                DRUM_HOLD,
                DRUM_PITCH,
                float(np.clip(vel, VELOCITY_MIN, 1.0)),
                f"drum:{hit}",
                LAYER_TAIKO,
                int(piece.drum_rng.integers(0, TAIKO_PATCHES)),
            )
        )


def _play_heartbeat(piece: _Piece) -> None:
    """Fill the beats between the big strikes with the heartbeat."""
    heart_rng = piece.plan.rng("heart")
    beat = piece.beat
    beat_time = piece.grid
    while beat_time < piece.length - HEART_END_MARGIN:
        gate = _drum_gate(piece, beat_time)
        taken = any(
            abs(beat_time - hit_time) < HEART_CLEAR_BEATS * beat
            for hit_time, kind, _ in piece.taiko_hits
            if kind == HIT_ODAIKO
        )
        rolling = _in_roll(piece, beat_time, HEART_ROLL_BEFORE, HEART_ROLL_AFTER_BEATS * beat)
        if gate > HEART_GATE and not taken and not rolling and not piece.driving:
            _beat_heart(piece, heart_rng, beat_time, gate)
            piece.beats += 1
        beat_time += beat


def _beat_heart(
    piece: _Piece, heart_rng: np.random.Generator, beat_time: float, gate: float
) -> None:
    """Play one beat of the heart, a lub-dub when the day suits it."""
    vel = (
        HEART_VELOCITY
        + HEART_VELOCITY_GATE * gate
        + HEART_VELOCITY_POWER * piece.plan.at(piece.plan.power, beat_time)
    ) * float(heart_rng.uniform(HUMANISE_LOW, HUMANISE_HIGH))
    variant = int(heart_rng.integers(0, HEART_PATCHES))
    piece.notes.append(
        Note(beat_time, DRUM_HOLD, DRUM_PITCH, vel, DRUM_HEART, LAYER_HEART, variant)
    )
    if piece.heart_style == 1:
        piece.notes.append(
            Note(
                beat_time + HEART_ECHO_DELAY * piece.beat,
                DRUM_HOLD,
                DRUM_PITCH,
                vel * HEART_ECHO_VELOCITY,
                DRUM_HEART,
                LAYER_HEART,
                (variant + 1) % HEART_PATCHES,
            )
        )


def _add_layers(piece: _Piece) -> None:
    """Make the layers with their filters, levels and room, and tune them to who leads."""
    plan = piece.plan
    bright = plan.bright
    swell = piece.swell
    low_cut, high_cut = DRONE_CUT_LOW, DRONE_CUT_HIGH
    drone_cut = (
        low_cut
        * (high_cut / low_cut) ** piece.breath
        * (DRONE_CUT_BRIGHT_FLOOR + DRONE_CUT_BRIGHT_SHARE * bright)
        * dips(plan, DRONE_DIP_DEPTH)
    )
    pluck_cut = (
        PLUCK_CUT_BASE
        * (PLUCK_CUT_BRIGHT_FLOOR + PLUCK_CUT_BRIGHT_SHARE * bright)
        * dips(plan, WOOD_DIP_DEPTH)
    )
    block_cut = (
        BLOCK_CUT_BASE
        * (BLOCK_CUT_BRIGHT_FLOOR + BLOCK_CUT_BRIGHT_SHARE * bright)
        * dips(plan, WOOD_DIP_DEPTH)
    )
    drone_presence = np.clip(
        piece.surge
        * (DRONE_PRESENCE_FLOOR + DRONE_PRESENCE_POWER * plan.power)
        * (1.0 + DRONE_PRESENCE_SWELL * swell),
        0.0,
        DRONE_PRESENCE_MAX,
    )
    wash_presence = np.clip(
        (WASH_PRESENCE_FLOOR + WASH_PRESENCE_NIGHT * plan.night + WASH_PRESENCE_SWELL * swell)
        * (1.0 - WASH_BUSY_DUCK * piece.busy),
        0.0,
        1.0,
    )
    piece.layers = {
        LAYER_DRONE: Layer(
            LAYER_DRONE,
            DRONE_GAIN,
            drone_presence,
            drone_cut,
            0.0,
            reverb=DRONE_REVERB,
            title="choir drone, breathing",
        ),
        LAYER_SUB: Layer(
            LAYER_SUB, SUB_GAIN, drone_presence, None, 0.0, reverb=SUB_REVERB, title="round sub"
        ),
        LAYER_WASH: Layer(
            "wash",
            WASH_GAIN,
            wash_presence,
            None,
            0.0,
            reverb=WASH_REVERB,
            title="stretched washes",
        ),
        LAYER_PLUCKS: Layer(
            "plucks", PLUCK_GAIN, None, pluck_cut, 0.0, reverb=PLUCK_REVERB, title="plucked line"
        ),
        LAYER_SLIT: Layer(
            LAYER_SLIT, SLIT_GAIN, None, None, 0.0, reverb=WOOD_REVERB, title="slit drum"
        ),
        LAYER_BLOCK: Layer(
            LAYER_BLOCK, BLOCK_GAIN, None, block_cut, 0.0, reverb=WOOD_REVERB, title="woodblock"
        ),
        LAYER_CLACK: Layer(
            LAYER_CLACK, 1.0, None, None, 0.0, reverb=CLACK_REVERB, title="shishi-odoshi"
        ),
        LAYER_LEAD: Layer(
            LAYER_LEAD, 1.0, None, None, 0.0, reverb=LEAD_REVERB, title="chant leader"
        ),
        LAYER_TAIKO: Layer(
            LAYER_TAIKO, TAIKO_GAIN, None, None, 0.0, reverb=TAIKO_REVERB, title="taiko"
        ),
        LAYER_HEART: Layer(
            LAYER_HEART, TAIKO_GAIN, None, None, 0.0, reverb=HEART_REVERB, title="heartbeat"
        ),
    }
    lead.apply_curves(
        piece.layers, lead.lead_curves(plan, piece.leads, LEADS), lead.LEAD_STEP_DB, LEAD_TRIM_DB
    )
    trim_layer(piece.layers[LAYER_PLUCKS], piece.pluck_sig)
    trim_layer(piece.layers[LAYER_LEAD], piece.lead_sig)


def _assemble(piece: _Piece) -> Score:
    """Put the notes and the layers together as the score, with the notes about how it was made."""
    plan = piece.plan
    space = np.clip(
        SPACE_BASE
        + SPACE_HUMID * plan.humid
        + SPACE_SWELL * piece.swell
        + SPACE_NIGHT * plan.night,
        0.0,
        SPACE_MAX,
    )
    score = Score(
        piece.length,
        piece.notes,
        piece.layers,
        [],
        space,
        events=list(plan.events),
        compress_db=COMPRESS_DB,
        level_db=LEVEL_DB,
    )
    plan.choices["patch"] = str(piece.patch)
    plan.choices[MOTIF_CHOICE] = str(piece.motif_played)
    plan.choices[lead.LEAD_HANDOVERS_CHOICE] = str(len(piece.leads.handed))
    score.leads = piece.leads
    score.choices = dict(plan.choices)
    score.notes_about = _describe(piece, score.voices_used())
    return score


def _describe(piece: _Piece, used: dict[str, int]) -> list[str]:
    """Return the notes about how the piece was made; `used` counts the notes of each voice."""
    drone_sig = piece.drone_sig
    lead_sig = piece.lead_sig
    quiet = piece.quiet
    notes_about = [
        (
            f"{piece.tuning_name} on {PITCH_NAMES[piece.root_pc]}, {piece.mode} ({SECONDS_PER_MINUTE / piece.slot:.0f} pulses a minute); one plucked line on {piece.pluck_sig.voice} (cell '{piece.cell_name}' of {piece.pluck_len:.1f} seconds, about one repeat in three moving one note by a step), "
            f"the only plucked voice. Slit drum '{piece.slit_name}' ({piece.slit_pulses} pulses, {piece.slit_count} hits) and woodblock '{piece.block_name}' "
            f"({piece.block_pulses} pulses, {piece.block_count} hits) on the off-beats of the pluck's pulse, never within {int(MIN_GAP * MILLISECONDS)} ms of a plucked note; "
            f"{len(piece.clack_times)} shishi-odoshi clacks."
        ),
        (
            f"Drone on a cello section and {drone_sig.label} (and a quiet one an octave up) with a vibrato of 4 to 6 cents that begins after a second or two (rate and phase of each note its own), path '{piece.path_name}' "
            f"({len(piece.segments)} stretches), washes '{piece.wash_name}' ({used.get('wash:cello', 0) + used.get('wash:violin', 0) + used.get('wash:felt', 0)} placed, {WASH_SECONDS:.0f} seconds each)."
        ),
        (
            f"Chant leader '{piece.lead_name}' ({piece.lead_notes} long notes of {lead_sig.label} in the scale, with rests); heartbeat '{piece.heart_name}' at {piece.bpm:.0f} per minute "
            f"({piece.beats} beats in the busy stretches), {piece.strikes} taiko strikes on the pulse, {piece.ostinato_hits} hits of the ostinato and {piece.rolls} accelerating rolls."
        ),
        lead.summary(piece.default_lead, piece.leads),
    ]
    if quiet:
        notes_about.append(
            f"A last soft strike at {int(quiet) // SECONDS_PER_MINUTE}:{int(quiet) % SECONDS_PER_MINUTE:02d} as the busy stretch ends."
        )
    return notes_about


def _ladder(piece: _Piece, index: int, base: int) -> float:
    """Return the pitch of a step of the scale: five notes an octave, `base` is the root."""
    octave, degree = divmod(index, PENTATONIC_STEPS)
    return float(base + SEMITONES_PER_OCTAVE * octave + piece.steps[degree])


def _tones(piece: _Piece, low: int, high: int) -> list[float]:
    """Return the notes of the day's scale from `low` to `high`, lowest first."""
    return [
        float(m)
        for m in range(low, high + 1)
        if (m - piece.root_pc) % SEMITONES_PER_OCTAVE in piece.steps
    ]


def _form_at(piece: _Piece, second: float) -> float:
    """Return the strength of the section of the form at a second."""
    return piece.strength[max(bisect_right(piece.starts, second) - 1, 0)]


def _drums_at(piece: _Piece, second: float) -> bool:
    """Return True where the form and the energy of the day let the drums play."""
    index = max(bisect_right(piece.starts, second) - 1, 0)
    return (piece.drums_on or index in piece.lifted) and piece.strength[index] >= DRUMS_FROM


def _drum_gate(piece: _Piece, second: float) -> float:
    """Return how busy the room is where the form and the energy of the day let the drums play, else 0."""
    return piece.plan.at(piece.busy, second) if _drums_at(piece, second) else 0.0


def _on_grid(piece: _Piece, second: float) -> float:
    """Return the beat nearest to a second."""
    return piece.grid + round((second - piece.grid) / piece.beat) * piece.beat


def _drive_at(piece: _Piece, second: float) -> float:
    """Return how much of its extra hits a pattern plays at a second, from the activity of the room."""
    return ACTIVITY_FLOOR + (1.0 - ACTIVITY_FLOOR) * piece.plan.at(piece.relative_activity, second)


def _in_roll(piece: _Piece, second: float, lead_in: float, tail: float) -> bool:
    """Return True when a second falls in a roll, from `lead_in` seconds before it starts to `tail` seconds after the strike."""
    return any(
        start - lead_in <= second <= end + tail
        for start, end in zip(piece.event_starts, piece.big, strict=True)
    )


def _roll_into_strike(piece: _Piece, start: float, end: float, final: float) -> None:
    """Add a roll that starts at `start` and arrives at a big strike at `end`."""
    piece.taiko_hits.extend(
        (hit.time, HIT_ODAIKO, hit.velocity)
        for hit in taiko.roll(start, end, HIT_ODAIKO, ROLL_START_LEVEL, final)
    )
    piece.event_starts.append(start)
    piece.big.append(end)
    piece.rolls += 1


def _ostinato_rests(piece: _Piece, when: float, form: float, need: float) -> bool:
    """Return True where the ostinato leaves a hit out: at the end, in a weak section or room, or inside a roll."""
    return (
        when >= piece.length - OSTINATO_END_MARGIN
        or form < DRIVE_FROM
        or _drum_gate(piece, when) <= OSTINATO_GATE
        or need * EXTRA_NEED > _drive_at(piece, when) + EXTRA_REACH * (piece.drive_energy - 0.5)
        or _in_roll(piece, when, 0.5, piece.beat)
    )


def _free(piece: _Piece, when: float, gap: float = MIN_GAP, wood_gap: float = 0.0) -> bool:
    """Return True when `when` is clear of the plucked notes by `gap` and of the other wooden hits by `wood_gap`."""
    for sorted_times, apart in ((piece.pluck_onsets, gap), (sorted(piece.wood_hits), wood_gap)):
        if not sorted_times or apart <= 0.0:
            continue
        k = bisect_left(sorted_times, when)
        near = sorted_times[max(k - 1, 0) : k + 1]
        if any(abs(when - other) < apart for other in near):
            return False
    return True


def _pluck_presence(piece: _Piece, second: float) -> float:
    """Return how awake the room is for the pluck at a second, 0 to 1."""
    value = float(
        np.clip(
            (piece.plan.at(piece.awake_base, second) - PLUCK_AWAKE_FROM) / PLUCK_AWAKE_SPAN,
            0.0,
            1.0,
        )
    )
    return float(smoothstep(np.array(value)))


def _motif_on_ladder(piece: _Piece, number: int) -> _Statement:
    """Return the motif of a statement, on the ladder of the tuning: a degree of the day's scale lands on the nearest step of the pentatonic."""
    plan = piece.plan
    # a statement is at most one cell long
    motif = develop_motif(plan.motif(), number, motif_seed(plan), max_beats=CELL_SLOTS)
    motif_steps = [round(degree * MOTIF_OCTAVE_STEPS / plan.scale.size) for degree, _ in motif]
    base = piece.pluck_base
    while min(_ladder(piece, step, base) for step in motif_steps) < MOTIF_FLOOR:
        base += SEMITONES_PER_OCTAVE
    return _Statement(motif, motif_steps, base)


def _awake_firsts(piece: _Piece) -> list[int]:
    """Return the first awake cell of the sections that the motif of the room may open."""
    sections = piece.sections
    pluck_len = piece.pluck_len
    awake_firsts: list[int] = []
    motif_sections = [i for i, (_, _, role, _) in enumerate(sections) if role in MOTIF_ROLES]
    for i in motif_sections or range(len(sections)):
        first = round(sections[i][0] / pluck_len)
        last = min(round(sections[i][1] / pluck_len), piece.cells)
        awake_cells = [
            k for k in range(first, last) if _pluck_presence(piece, k * pluck_len) > MOTIF_AWAKE
        ]
        if awake_cells:
            awake_firsts.append(awake_cells[0])
        if awake_firsts and not motif_sections:
            break
    return awake_firsts


def _melodic_index(piece: _Piece, when: float) -> int:
    """Return the step of the motif that the slit drum plays next in the phrase it leads."""
    plan = piece.plan
    number = int(when / piece.lead_grid)
    motif = develop_motif(plan.motif(), number, motif_seed(plan))
    piece.contour[number] = piece.contour.get(number, -1) + 1
    degree = motif[piece.contour[number] % len(motif)][0]
    return round(degree * MOTIF_OCTAVE_STEPS / plan.scale.size)


def _next_chant_step(
    piece: _Piece, lead_rng: np.random.Generator, current_step: int, *, final: bool
) -> int:
    """Return the next step of the phrase of the leader: the last note comes home, the others walk."""
    if final:
        return int(weighted(lead_rng, NEXT_CHANT_STEPS, NEXT_CHANT_WEIGHTS))
    return int(
        np.clip(
            current_step + int(lead_rng.choice(CHANT_STEP_MOVES)),
            piece.lead_steps[0],
            piece.lead_steps[-1],
        )
    )


def _clack_candidates(piece: _Piece, clack_rng: np.random.Generator) -> list[tuple[float, str]]:
    """Return the moments where a clack may fall: the quiet after an anomaly and a few calm moments."""
    plan = piece.plan
    candidates = [
        (second + EVENT_CLACK_DELAY, CLACK_EVENT)
        for second, _ in plan.events
        if plan.at(plan.night, second) < CLACK_MAX_NIGHT
        and plan.at(piece.awake_base, second) > CLACK_MIN_AWAKE
    ]
    wander = float(clack_rng.uniform(WANDER_MIN, WANDER_MAX))
    while wander < piece.length - WANDER_END_MARGIN:
        if (
            plan.at(piece.awake_base, wander) > WANDER_MIN_AWAKE
            and plan.at(piece.activity, wander) < WANDER_MAX_ACTIVITY
            and clack_rng.random() < WANDER_CHANCE
        ):
            candidates.append((wander, CLACK_CALM))
        wander += float(clack_rng.uniform(WANDER_GAP_MIN, WANDER_GAP_MAX))
    return candidates


def _calmest_moments(piece: _Piece) -> list[tuple[float, str]]:
    """Return the calmest awake moments of the piece, calmest first, as clack candidates for a piece that has none."""
    plan = piece.plan
    seconds = [
        float(s)
        for s in np.arange(WANDER_MIN, piece.length - WANDER_END_MARGIN)
        if plan.at(piece.awake_base, float(s)) > WANDER_MIN_AWAKE
    ]
    seconds.sort(key=lambda s: plan.at(piece.activity, s))
    return [(s, CLACK_CALM) for s in seconds]


def _choose_clacks(piece: _Piece, candidates: list[tuple[float, str]]) -> None:
    """Choose the clacks from the candidates: the anomalies first, a few in a piece, clear of the pluck and of each other."""
    length = piece.length
    wanted_clacks = int(np.clip(length / CLACK_PIECE_SECONDS, 1, CLACK_MAX))
    clack_times = piece.clack_times
    for at, _kind in sorted(candidates, key=lambda c: (c[1] != CLACK_EVENT, c[0])):
        if len(clack_times) >= wanted_clacks or at > length - CLACK_END_MARGIN:
            continue
        if any(abs(at - other) < CLACK_APART for other in clack_times):
            continue
        for push in CLACK_PUSHES:
            when = at + push
            if (
                _free(piece, when - CLACK_MARGIN, CLACK_FREE_SPAN)
                and _free(piece, when, CLACK_FREE_SPAN)
                and _free(piece, when + CLACK_DELAY + CLACK_MARGIN, CLACK_FREE_SPAN)
            ):
                clack_times.append(when)
                break
