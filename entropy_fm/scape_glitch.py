"""
Cinematic Glitch: swells, brass-like stabs, a growling bass and hard half-time drums, glitched by what happens in the house.

The generative system is a film score cut up. A slow chord loop in a dark mode (aeolian, dorian or phrygian, in the key
of the day, with chords that stay inside the mode) moves every two bars, four when the day is quiet or the house sleeps,
over a tempo of 54 to 72 beats per minute whose drums count twice that. The chords are saw swells that open as they rise,
a cello and violin section under them, stretched washes of strings, and brass-like stabs on the changes. The bass is a
growl that wobbles through vowels in time with the tempo; its character (drive, fold, FM, how fast it wobbles) is one
of six rows that the heat of the house picks, from a held note under the swells to the full drop. The drums are a
composed kit (a punchy kick, a layered snare with a dark plate, hats, rim, ticks, ghost snares, fills) with five
levels, from a kick and a snare to the full groove, and a drum bus that is glued and saturated;
the bass, pads, strings and stabs duck from every kick.

The edits are what the house plays: a stutter, a tape stop, a hard brake of 60 to 150 ms (a fast fall to nothing, a cut,
a gap of a sixteenth, a snap back on the grid, sometimes twice in a row), a grain or spectral freeze, a crush of one
hit, a gate chop, a beat repeat of the kit, a beat or half a beat played backwards, a repeat of a slice an octave up, an
octave down or slowed (a jump, never a glide), a bit drop (a few kilohertz and three to five bits, with no filter, so it
grates) and a roll whose repeats get shorter and louder (1/8, 1/16, 1/32). Where the kit plays, most bars get one, on the
grid: about one in every one to two bars on a busy day and one in every two to four on a quiet one, the activity of the
house and the energy of the day moving it between them; the ramps (a roll, a reversal, a brake) lead into the bar line, and
the same kind rarely plays twice in a row. Most cut the whole mix; about a fifth are on an instrument of their own, the
strings, the growl or the stabs (stuttered, reversed, gated, repeated or dropped), apart from those of the drums, and no two
edits overlap on a bus. The hottest peaks or plateaus of a piece (one or two by the energy of the day, at most 16 bars
each, and never where the house is calm) go all the way: an edit in nearly every bar, almost all of the whole mix, with gates that cut to silence. A door, a switch, an appliance that starts or stops, a light, someone moving brings its bar an edit that it favours
(a door a reversal or a brake, an appliance a bit drop, a light a pitch repeat, motion a roll or a gate), and the older
rule stays: one of the classic kinds at most every couple of bars, on the grid. Which are favoured is dealt by the day. The
big moments are drop sequences: a reversed swell, a snare roll that stutters, a reversed snare, a tape stop on the whole mix,
a silence, then the drop (stab, sub boom, taiko, the full growl and the big pattern). They come at the peaks of the day and
at an anomaly; an anomaly in the night is a reversed swell into a distant boom, nothing else, and the night has no drums to
cut.

How the house steers it: the energy of the day leans the mode of the piece (quiet days slow, busy days fast, but similar days
rotate between them): heavy at 54 to 60 beats per minute (sparse, huge hits, a held sub growl, long freezes and tape stops,
reversed swells), half-time at 64 to 68 and driving at 70 to 72. The mode picks the tempo inside its band and the family of
drums (half-time, trap, broken, ghosted break, cinematic low hits, industrial, one-drop dub or IDM stutter, each with three or
four variants that change with the role of the section, and a fill into every new section). The energy also picks how long a
chord lasts, the favoured edits, the bass phrases, and scales the heat of the house so that a quiet day strips the kit back (a kick and
a snare, a held growl, one or two drops) and a busy day brings the full kit, the whole growl and up to six drops; the
warmth picks the character of the growl and the washes. The form of the day (intro, build, peak, release, plateau, outro) is
on phrases of eight bars: the intro and the outro are swells and washes alone, the bass and the kit enter and leave on the
phrase lines, and every peak begins with a drop that the build before it leads into. The strongest event of the day gets its
drop on the next phrase line, and the motif of the room is played by the strings after the drop of the first and the last peak (or at
the start of a plateau), the later one developed.

The day: the night is swells, washes and distant booms alone; as the house wakes the bass enters, then sparse drums, which grow
with the activity and the power of the appliances to the full pattern in the busy stretches; towards the evening the drums
thin out and leave, and the swells come back. The house never writes a note: the notes come from the chord loop, the bass phrases, the
patterns and the edits, and the day deals which of each.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from . import glitch, taiko
from .character import chord_bars, tempo_target
from .gen import weighted
from .glitch_kits import DRUM_ENERGY, FAMILIES, KINDS, STEPS, Pattern
from .harmony import MOTIF_CHOICE, develop_motif, motif_seed, motif_sites
from .plan import (
    BUILD,
    CLIMAX,
    FORM_BARS,
    INTRO,
    MIN_FORM_CELLS,
    OUTRO,
    PLATEAU,
    RELEASE,
    SHORT_FORM_BARS,
    TEMPO_TOLERANCE,
)
from .score import CR, Edit, Glitch, Layer, Note, Score
from .styles import GLITCH
from .synth import smoothstep

if TYPE_CHECKING:
    from collections.abc import Callable

    from .plan import Plan


class Mode(NamedTuple):
    """How a piece moves: its band of tempo and how sparse, how heavy and how stretched it is."""

    name: str
    tempo: tuple[int, int]
    # the energy of the day that it suits, and the energy of the edits that it favours
    centre: float
    edit_energy: float
    # the highest level of the kit (1 to 5) outside a drop, and the highest row of the growl outside a drop
    level_cap: int
    growl_cap: int
    # how much longer the freezes and tape stops run, and the share of freezes that are grains
    stretch: float
    grain_share: float
    # reversed swells into every section after the first
    swells: bool
    # the lowest level of the kit that sounds through the day (0 leaves the drums to the heat), and whether the phrases
    # are marked by a long freeze, a tape stop or a brake even when the house is quiet
    drum_floor: int = 0
    phrase_edits: bool = False


MODES = (
    Mode(
        "heavy", (54, 60), 0.15, 0.2, 3, 1, 2.0, 0.2, swells=True, drum_floor=2, phrase_edits=True
    ),
    Mode("half-time", (64, 68), 0.5, 0.5, 4, 4, 1.0, 0.5, swells=False),
    Mode("driving", (70, 72), 0.85, 0.9, 5, 5, 1.0, 0.5, swells=False),
)
# quiet days lean slow and busy days fast, but days this close to the best fit rotate between the modes
MODE_TOLERANCE = 0.4
# the families of a mode are picked on the energy of the day, within this of the best fit
FAMILY_TOLERANCE = 0.3
# the kit gets this many half bars to a group of two bars, and the last of them can be a fill
SLOTS = 4
# the variant of each role of the form (a build, a peak and a plateau do not share a bar)
ROLE_VARIANT = {INTRO: 0, BUILD: 0, CLIMAX: 1, RELEASE: 3, PLATEAU: 2, OUTRO: 0}
# the bars after which the variant of a pattern turns over
VARIANT_BARS = 4
# the softest kick that each role of the form plays: a build and a release keep the downbeat, a peak every kick
ROLE_KICK_FLOOR = {INTRO: 0.9, BUILD: 0.9, CLIMAX: 0.0, RELEASE: 0.9, PLATEAU: 0.7, OUTRO: 0.9}
# a phrase mark of a heavy day: the bar of each phrase that it falls on, the day (0 to 1) it needs, the bars it keeps clear of a drop
PHRASE_MARK_BARS = (2, 5)
PHRASE_MARK_DAY = 0.5
PHRASE_MARK_CLEAR = 3
PHRASE_MARK_KINDS = ("tape", "spectral", "grain", "brake")
# the share of the tape stops that are long (a wind down of a beat and a half), times the stretch of the mode
TAPE_LONG_SHARE = 0.5


class _Shape(NamedTuple):
    """The parameters of a groove edit, and the steps of a bar that it may start on."""

    params: dict[str, Any]
    starts: tuple[int, ...]


class Loop(NamedTuple):
    """Four chords (semitones above the root, 0 minor or 1 major) and how much their roots move (0 a vamp, 1 a chase)."""

    name: str
    chords: tuple[tuple[int, int], ...]
    motion: float


# the loops of chords of each mode, whose triads stay inside it (the fifth of the phrygian is diminished, so it has none)
LOOPS: dict[str, tuple[Loop, ...]] = {
    "aeolian": (
        Loop("i bIII iv v", ((0, 0), (3, 1), (5, 0), (7, 0)), 0.3),
        Loop("i iv bVI bVII", ((0, 0), (5, 0), (8, 1), (10, 1)), 0.5),
        Loop("i bVI iv v", ((0, 0), (8, 1), (5, 0), (7, 0)), 0.65),
        Loop("i bVII bVI v", ((0, 0), (10, 1), (8, 1), (7, 0)), 0.85),
    ),
    "dorian": (
        Loop("i bIII IV bVII", ((0, 0), (3, 1), (5, 1), (10, 1)), 0.3),
        Loop("i IV bIII bVII", ((0, 0), (5, 1), (3, 1), (10, 1)), 0.5),
        Loop("i bVII IV v", ((0, 0), (10, 1), (5, 1), (7, 0)), 0.65),
        Loop("i ii bIII bVII", ((0, 0), (2, 0), (3, 1), (10, 1)), 0.85),
    ),
    "phrygian": (
        Loop("i bvii bVI iv", ((0, 0), (10, 0), (8, 1), (5, 0)), 0.3),
        Loop("i bII bVI iv", ((0, 0), (1, 1), (8, 1), (5, 0)), 0.55),
        Loop("i bII bIII bvii", ((0, 0), (1, 1), (3, 1), (10, 0)), 0.85),
    ),
}
# the drop: a major chord a semitone above the root on phrygian days, a tone below it on the others
DROP_CHORDS = {"phrygian": (1, 1), "aeolian": (10, 1), "dorian": (10, 1)}
# the bass phrases of two bars (eight beats): (beat, length in beats, semitones above the chord root)
Phrase = tuple[tuple[float, float, int], ...]
HOLD_A: Phrase = ((0.0, 7.4, 0),)
SYNC_A: Phrase = (
    (0, 2.25, 0), (2.5, 0.5, 0), (3.0, 0.75, 0), (4, 1.25, 0), (5.5, 0.5, 0), (6, 1.0, 0), (7, 0.5, 12),
)  # fmt: skip
BUSY_A: Phrase = (
    (0, 0.75, 0), (0.75, 0.25, 0), (1, 0.5, 0), (1.5, 0.5, 12), (2, 0.75, 0), (3, 0.25, 0), (3.25, 0.25, 0),
    (3.5, 0.5, 7), (4, 1.5, 0), (5.5, 0.5, 0), (6, 0.5, 0), (6.5, 0.25, 0), (6.75, 0.25, 0), (7, 0.5, 10),
)  # fmt: skip
HOLD_B: Phrase = ((0.0, 3.4, 0), (4.0, 3.4, 0))
SYNC_B: Phrase = (
    (0, 1.75, 0), (2, 0.25, 0), (2.5, 0.5, 0), (3.5, 0.5, 0), (4, 2.0, 0), (6, 0.75, 0), (7, 0.5, 12),
)  # fmt: skip
BUSY_B: Phrase = (
    (0, 0.5, 0), (0.5, 0.5, 0), (1, 0.5, 12), (1.5, 0.5, 0), (2, 1.0, 0), (3, 0.5, 0), (3.5, 0.5, 7),
    (4, 0.5, 0), (4.5, 0.5, 0), (5, 0.5, 12), (5.5, 0.5, 0), (6, 0.75, 0), (6.75, 0.25, 0), (7, 0.5, 10), (7.5, 0.5, 0),
)  # fmt: skip
HOLD_C: Phrase = ((0.0, 5.4, 0), (6.0, 1.9, 7))
SYNC_C: Phrase = (
    (0, 0.75, 0), (1, 0.5, 0), (1.5, 1.0, 0), (3, 0.75, 0), (4, 0.75, 0), (5, 0.5, 0), (5.5, 1.0, 0), (7, 0.5, 12),
)  # fmt: skip
BUSY_C: Phrase = (
    (0, 0.25, 0), (0.25, 0.25, 0), (0.5, 0.5, 0), (1, 0.5, 12), (1.5, 0.5, 0), (2, 0.75, 0), (3, 0.25, 0), (3.25, 0.25, 0),
    (3.5, 0.5, 10), (4, 0.5, 0), (4.5, 0.25, 0), (4.75, 0.25, 0), (5, 0.5, 12), (5.5, 0.5, 0), (6, 1.0, 0), (7, 0.5, 7), (7.5, 0.5, 0),
)  # fmt: skip
BASS_SETS: tuple[tuple[str, tuple[Phrase, Phrase, Phrase]], ...] = (
    ("held, syncopated, busy", (HOLD_A, SYNC_A, BUSY_A)),
    ("two held, driving, eighths", (HOLD_B, SYNC_B, BUSY_B)),
    ("held and a fifth, broken, rolling", (HOLD_C, SYNC_C, BUSY_C)),
)
# the character of the growl: which vowels the wobble sweeps
GROWLS: tuple[tuple[str, int], ...] = (
    ("U O A E", 0),
    ("wide, bright formants", 1),
    ("dark, narrow formants", 2),
)
# the least of the heat that the base of the day (the bass, the first drums) is made from, on the quietest and the busiest day
BODY_FLOOR = (0.1, 0.4)
# the heat of the house is scaled by this, on the quietest and the busiest day: a quiet day strips the kit back
HEAT_GAIN = (0.55, 1.6)
# how much of the heat each role of the form lets through: the intro and the outro are swells and washes alone
ROLE_HEAT = {INTRO: 0.0, BUILD: 0.75, CLIMAX: 1.0, RELEASE: 0.7, PLATEAU: 0.9, OUTRO: 0.0}
# the most drops of a piece, on the quietest and the busiest day (a third as many in the cheapest tier or a short piece)
MOST_DROPS = (1, 6)
# the share of the events of the house that bring an edit: the least, and what the energy of the day and the heat add
EDIT_CHANCE = 0.35
EDIT_ENERGY = 0.35
EDIT_HEAT = 0.3
# the groove: in a bar where the kit plays, the chance of an edit is the least, plus what the energy of the day and the
# heat of the bar add; a bar with an event of the house is this much more likely, and a bar is tried this many times
GROOVE_CHANCE = 0.4
GROOVE_ENERGY = 0.12
GROOVE_HEAT = 0.1
GROOVE_HOUSE_BOOST = 1.6
GROOVE_TRIES = 3
# the bars before a drop and after it, and the bar before a new section, that the groove leaves alone
GROOVE_BEFORE_DROP = 1
GROOVE_AFTER_DROP = 3
# where a groove edit lands: the share of the instruments (strings, bass, stabs), of the kit and of the whole mix
GROOVE_BUSES: tuple[tuple[str, float], ...] = (("whole", 0.65), ("kit", 0.12), ("instrument", 0.23))
# the hottest peaks or plateaus of a piece go all the way: an edit in nearly every bar, almost all of the mix, gates
# that cut to silence; one such section on a calm day, up to two on a busy one, each at most this many bars, and only
# where the house is at least this hot, so a quiet stretch never goes extreme
EXTREME_CHANCE = 0.95
EXTREME_BUSES: tuple[tuple[str, float], ...] = (("whole", 0.9), ("instrument", 0.1))
EXTREME_GATE_FLOOR = 0.0
EXTREME_SECTIONS = (1, 2)
EXTREME_MAX_BARS = 16
EXTREME_MIN_HEAT = 0.5
INSTRUMENT_BUSES: tuple[tuple[str, float], ...] = (("bass", 0.4), ("str", 0.35), ("stab", 0.25))
KIT_BUSES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("kick", "snare", "hats"), 0.4),
    (("snare",), 0.25),
    (("kick",), 0.15),
    (("hats",), 0.2),
)
# the weight of each kind of groove edit, before the favour of the day and the event of the house
GROOVE_KINDS = {
    "stutter": 2.0,
    "roll": 1.5,
    "reverse": 2.0,
    "pitchrep": 2.0,
    "bitdrop": 1.5,
    "gate": 1.5,
    "brake": 0.8,
}
# the kinds that lead into the next bar (a roll, a reversal, a brake) are more likely on the last bar of two or four
GROOVE_RAMPS = {"roll", "reverse", "brake"}
GROOVE_RAMP_BOOST = 2.0
# a kind that sounded in one of the last few edits is this much less likely
GROOVE_RECENT = 4
GROOVE_REPEAT_PENALTY = 0.4
# what each kind of event of the house brings to its bar; the kinds that it does not name are this likely
GROOVE_HOUSE: dict[str, dict[str, float]] = {
    "door": {"reverse": 3.0, "brake": 2.0, "roll": 1.0},
    "motion": {"stutter": 2.0, "roll": 2.0, "gate": 2.0, "pitchrep": 1.0},
    "light": {"pitchrep": 3.0, "gate": 2.0, "reverse": 1.0},
    "appliance": {"bitdrop": 3.0, "stutter": 1.0, "brake": 1.0},
}
GROOVE_HOUSE_OTHER = {"stutter": 2.0, "gate": 1.0, "pitchrep": 1.0}
GROOVE_HOUSE_UNNAMED = 0.3
# the steps (an eighth of a beat) of a bar
BAR_STEPS = 32
# the gates of the groove: eight sixteenths, two beats
GROOVE_GATES = (
    (1, 0, 1, 1, 0, 1, 0, 1),
    (1, 1, 0, 1, 0, 0, 1, 1),
    (1, 0, 0, 1, 1, 0, 1, 0),
)
# the read speeds of a pitch repeat: an octave up, an octave down, and slowed
PITCH_RATES = (2.0, 0.5, 2.0 / 3.0)
# a roll of a beat and a half (the rest are a beat), a reversal of a half beat or a beat (in steps), and how a pitch repeat
# is made: its share of sixteenth slices (the rest are eighths of two), and how much each repeat falls
ROLL_LONG_SHARE = 0.5
REVERSE_STEPS = (8, 16)
PITCH_SIXTEENTH_SHARE = 0.5
PITCH_FADE = 0.12
STUTTER_FADE = 0.07
GATE_FLOOR = 0.05
BRAKE_STOP = (0.07, 0.14)
# the sample rate and the depth of a bit drop: the held samples (44.1 kHz over this), and the bits
BIT_DROP_FACTORS = (12, 22)
BIT_DROP_BITS = (3, 5)
# the drop chord comes this many bars after the start of a peak; the motif of the room follows it
# the cheapest tier plays the accented hats alone (every other step)
LIGHT_HAT_FLOOR = 0.35
# how long a swell rings on past its chord: the cheapest tier lets it go sooner
SWELL_TAIL = 4.5
LIGHT_TAIL = 3.0
TOM_PITCH = 36.0
MOTIF_AFTER_DROP = 2
MOTIF_VELOCITY = 0.75
# the strings of a chord: the lower voices are the root and the fifth, the upper ones are led from chord to chord
STRINGS_LOW = (0, 7)
STRINGS_UP = 3
# the highest the upper strings go above the pad root (the root of the next octave but one is the limit)
STRINGS_TOP = 26
MOTIF_LOWEST = 60
# what the house heat has to be for each part to enter (the day moves them by a few hundredths)
THRESHOLDS = {"bass": 0.26, "kick": 0.40, "hat": 0.50, "ghost": 0.58, "full": 0.70, "big": 0.84}
# what the families suit: the energy of the day (0 to 1) for the patterns, the brightness (-1 to 1) of the day for the
# character of the growl and the washes (the average of its warmth and its energy)
BASS_ENERGY = {
    "held, syncopated, busy": 0.25,
    "two held, driving, eighths": 0.55,
    "held and a fifth, broken, rolling": 0.85,
}
FAVOUR_ENERGY = {"freezes": 0.1, "stretches": 0.25, "brakes": 0.4, "gates": 0.55, "stutters": 0.9}
GROWL_BRIGHTNESS = {"dark, narrow formants": -0.8, "U O A E": 0.0, "wide, bright formants": 0.8}
WASH_BRIGHTNESS = {"cellos": -0.6, "cellos and violins": 0.1, "cellos and felt piano": 0.7}
# the favoured edits of the day: the weight of each family of edit
FAVOURS: tuple[tuple[str, dict[str, float]], ...] = (
    ("stutters", {"stutter": 3.0, "brake": 1.0, "freeze": 1.0, "gate": 1.0, "tape": 0.5}),
    ("brakes", {"stutter": 1.0, "brake": 3.0, "freeze": 1.0, "gate": 1.0, "tape": 0.7}),
    ("freezes", {"stutter": 0.5, "brake": 1.5, "freeze": 3.0, "gate": 0.7, "tape": 1.5}),
    ("stretches", {"stutter": 0.5, "brake": 2.0, "freeze": 2.0, "gate": 0.7, "tape": 3.0}),
    ("gates", {"stutter": 1.0, "brake": 1.0, "freeze": 1.0, "gate": 3.0, "tape": 0.5}),
)
# the stretched washes: which strings sound under the swells
WASH_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cellos", ("wash:cello",)),
    ("cellos and violins", ("wash:cello", "wash:violin")),
    ("cellos and felt piano", ("wash:cello", "wash:felt")),
)
# what each kind of event of the house may bring, and how much it likes it
EVENT_EDITS: dict[str, dict[str, float]] = {
    "door": {"brake": 3.0, "tape": 1.0, "stutter": 2.0},
    "motion": {"stutter": 3.0, "gate": 2.0, "freeze": 2.0, "brake": 1.0},
    "light": {"freeze": 3.0, "gate": 2.0, "stutter": 1.0},
    "appliance": {"crush": 3.0, "brake": 2.0, "stutter": 1.0},
}
DEFAULT_EDITS = {"stutter": 2.0, "gate": 1.0, "freeze": 1.0}
# the most that one kind plays in a row, and the edits that give way to keep it so
MOST_IN_A_ROW = 2
GIVES_WAY = {"event", "house", "groove"}
# the kinds of edit that each choice of the events of the house makes
CHOICE_KINDS = {
    "brake": {"brake"},
    "tape": {"tape"},
    "stutter": {"stutter", "roll"},
    "gate": {"gate"},
    "freeze": {"grain", "spectral"},
    "crush": {"crush"},
}
GATE_PATTERNS = (
    (1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1),
    (1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1),
)
WHOLE = ("pad", "str", "wash", "stab", "bass", "kick", "snare", "hats")
SOFT = ("pad", "str", "wash", "stab", "bass")
KIT = ("kick", "snare", "hats")
# the ducking of the layers from the kick: (depth, time constant)
DUCK = {
    "bass": (0.7, 0.14),
    "pad": (0.4, 0.22),
    "str": (0.35, 0.22),
    "wash": (0.35, 0.22),
    "stab": (0.3, 0.12),
}
# the gain of the drum buses: the kick sits within a few dB of the strings, the hats about ten under the snare
KICK_GAIN, SNARE_GAIN, HATS_GAIN = 6.0, 1.8, 1.8
# the constant gain of the master, in dB: the arrangement is the dynamics of the style
MASTER_GAIN_DB = -4.2
GROWL_STATE_AT = (0.26, 0.34, 0.44, 0.56, 0.70, 0.84)


@dataclass
class _Chord:
    """A chord of two or four bars."""

    bar: int
    bars: int
    root: int
    pad: int
    quality: int
    drop: bool = False


def _bass_root(pitch_class: int) -> int:
    """Return the lowest pitch from E1 up with a pitch class (the growl sits between 41 and 78 Hz)."""
    return 28 + (pitch_class - 28) % 12


class _Book:
    """The edits of the piece, and the stretches that they hold, so that none overlaps another on a bus."""

    def __init__(self) -> None:
        self.edits: list[Edit] = []
        self.held: dict[str, list[tuple[float, float]]] = {}

    def free(self, layers: tuple[str, ...], a: float, b: float) -> bool:
        """Return True when none of the layers is held between a and b."""
        return not any(
            a < end and b > start for key in layers for start, end in self.held.get(key, [])
        )

    def add(self, edit: Edit) -> bool:
        """Add an edit when its layers are free; return whether it was added."""
        a, b = edit.t - edit.back, edit.t + edit.dur
        if not edit.layers or not self.free(edit.layers, edit.t - 0.001, b):
            return False
        for key in edit.layers:
            self.held.setdefault(key, []).append((a + edit.back - 0.001, b))
        self.edits.append(edit)
        return True


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Cinematic Glitch soundscape.

    :param plan: The curves and the deals of the day.
    """
    quality = plan.quality
    length = plan.length
    light = quality.max_layers <= 4
    rng = plan.rng("glitch")
    drum_rng = plan.rng("glitch drums")
    edit_rng = plan.rng("glitch edits")

    character = plan.character
    energy = character.energy
    brightness = 0.5 * character.warmth + 0.5 * (2.0 * energy - 1.0)
    tonic = plan.scale.root
    mode_name = plan.scale.mode if plan.scale.mode in LOOPS else "aeolian"
    phrygian = mode_name == "phrygian"
    mode = plan.pick(
        "mode",
        MODES,
        [1.0 - abs(entry.centre - energy) for entry in MODES],
        tolerance=MODE_TOLERANCE,
    )
    low_bpm, high_bpm = mode.tempo
    target = tempo_target(GLITCH, character)
    tempos = tuple(float(bpm) for bpm in range(low_bpm, high_bpm + 1))
    bpm = plan.pick(
        "tempo",
        tempos,
        [1.0 - abs((t - low_bpm) / (high_bpm - low_bpm) - target) for t in tempos],
        tolerance=TEMPO_TOLERANCE,
    )
    loop_name, loop = plan.pick(
        "chord loop",
        LOOPS[mode_name],
        [1.0 - abs(entry.motion - energy) for entry in LOOPS[mode_name]],
    )[:2]
    suited = tuple(f for f in FAMILIES if mode.name in f.modes)
    family = plan.pick(
        "drum patterns",
        suited,
        [1.0 - abs(DRUM_ENERGY[f.name] - energy) for f in suited],
        tolerance=FAMILY_TOLERANCE,
    )
    drums_name, pats = family.name, family.patterns
    bass_name, (hold, sync, busy) = plan.pick(
        "bass phrases",
        BASS_SETS,
        [1.0 - abs(BASS_ENERGY[name] - energy) for name, _ in BASS_SETS],
    )
    growl_name, vowels = plan.pick(
        "growl character",
        GROWLS,
        [1.0 - abs(GROWL_BRIGHTNESS[name] - brightness) / 2.0 for name, _ in GROWLS],
    )
    favour_name, favour = plan.pick(
        "favoured edits",
        FAVOURS,
        [1.0 - abs(FAVOUR_ENERGY[name] - mode.edit_energy) for name, _ in FAVOURS],
    )
    wash_name, wash_voices = plan.pick(
        "washes",
        WASH_SETS,
        [1.0 - abs(WASH_BRIGHTNESS[name] - brightness) / 2.0 for name, _ in WASH_SETS],
    )
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)

    beat = 60.0 / bpm
    bar = 4.0 * beat
    phrase_bars = FORM_BARS if length / (FORM_BARS * bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
    span_options = tuple(
        sorted({min(max(o, 2), 4) for o in chord_bars(character) if o <= phrase_bars})
    )
    third = 1.0 / 3.0
    within = float(np.clip(energy / third - min(int(energy / third), 2), 0.0, 1.0))
    base_span = plan.pick(
        "chord bars",
        span_options,
        [
            1.0 - abs(index / max(len(span_options) - 1, 1) - (1.0 - within))
            for index in range(len(span_options))
        ],
    )
    step = beat / 8.0
    u8, u16, u32 = beat / 2.0, beat / 4.0, beat / 8.0
    bars = int((length - 5.0) / bar)

    def snap(second: float) -> float:
        return round(second / step) * step

    # -- the curves: the day (when the house is awake), the heat (how lively it is then), the parts and their thresholds
    count = plan.count
    hour = plan.hour
    day = np.asarray(
        smoothstep((hour - 5.5) / 3.0)
        * (1.0 - 0.6 * smoothstep((hour - 18.0) / 3.5))
        * (1.0 - smoothstep((hour - 22.5) / 1.5))
    )
    lively = np.clip(plan.activity + 0.5 * plan.power, 0.0, 1.0)
    # the form: what each role lets through of the heat, from one phrase line to the next
    sections = plan.sections(bar)
    form = np.zeros(count)
    for first, last, role, _ in sections:
        form[int(first * CR) : int(last * CR) + 1] = ROLE_HEAT[role]
    gain = HEAT_GAIN[0] + (HEAT_GAIN[1] - HEAT_GAIN[0]) * energy
    heat = np.clip(lively * day * 1.1 * gain * form, 0.0, 1.0)
    # the body of the day: the bass and the first drums are the base by day, and the house adds the rest; the drops,
    # the stabs and the edits follow the heat itself, which is how lively it is
    floor = BODY_FLOOR[0] + (BODY_FLOOR[1] - BODY_FLOOR[0]) * energy
    body = np.clip(plan.lift(lively, floor) * day * 1.1 * gain * form, 0.0, 1.0)
    limits = {k: v + float(rng.uniform(-0.04, 0.04)) for k, v in THRESHOLDS.items()}
    first_drum = plan.piece_second_of_hour(6.5)
    last_drum = plan.piece_second_of_hour(21.5)

    def heat_in(a: float, b: float, curve: np.ndarray = heat) -> float:
        i, j = int(a * CR), max(int(b * CR), int(a * CR) + 1)
        return float(curve[min(i, count - 1) : min(j, count)].mean())

    def drums_ok(second: float) -> bool:
        return first_drum <= second < last_drum and plan.at(day, second) > 0.5

    notes: list[Note] = []
    spans: dict[str, list[tuple[float, float]]] = {}

    def put(  # noqa: PLR0913, PLR0917
        t: float, dur: float, midi: float, vel: float, voice: str, layer: str, patch_: int = 0
    ) -> None:
        if t < length - 1.0:
            notes.append(Note(t, dur, midi, vel, voice, layer, patch_))
            spans.setdefault(layer, []).append((t, t + min(dur, 3.0)))

    def alive(layers: tuple[str, ...], a: float, b: float) -> tuple[str, ...]:
        return tuple(key for key in layers if any(s < b and e > a for s, e in spans.get(key, [])))

    # -- the drops: where the big moments are (an anomaly, and the peaks of the busy stretches), on a bar of even number
    per_bar = [heat_in(b * bar, (b + 1) * bar) for b in range(bars + 1)]
    smooth_bars = np.convolve(np.pad(per_bar, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    drops: dict[int, str] = {}
    strongest = max(plan.events, key=lambda e: e[1])[0] if plan.events else -1.0
    for second, _strength in plan.events:
        d = max(2 * round(second / bar / 2.0), 2)
        if second == strongest:
            d = max(phrase_bars * math.ceil(second / (phrase_bars * bar)), phrase_bars)
        if d + 4 < bars:
            here = float(day[min(int(d * bar * CR), count - 1)])
            drops[d] = (
                "far"
                if here < 0.2
                else ("drums" if drums_ok(d * bar) and per_bar[d] > limits["kick"] else "hit")
            )
    for first, _, role, _ in sections:
        d = round(first / bar)
        if role == CLIMAX and d >= 4 and d + 4 < bars and d not in drops:
            drops[d] = "drums" if drums_ok(d * bar) and per_bar[d] > limits["kick"] else "hit"
    cap = MOST_DROPS[0] + round((MOST_DROPS[1] - MOST_DROPS[0]) * energy)
    ranked = sorted(range(4, max(bars - 6, 5)), key=lambda b: -smooth_bars[b])
    for b in ranked:
        d = 2 * (b // 2)
        if (
            smooth_bars[b] >= 0.60
            and drums_ok(d * bar)
            and all(abs(d - other) >= 20 for other in drops)
            and d + 4 < bars
            and len(drops) < (cap // 2 if light or plan.condensed else cap)
        ):
            drops[d] = "drums"

    # -- the chords: two bars each, four while the house sleeps, a drop where one is
    chords: list[_Chord] = []
    place, b = 0, 0
    drop_chord = DROP_CHORDS[mode_name]
    while b < bars:
        if b in drops:
            offset, qual = drop_chord
            span, is_drop = 2, True
        else:
            offset, qual = loop[place % len(loop)]
            place += 1
            calm = b % 4 == 0 and (base_span == 4 or heat_in(b * bar, (b + 4) * bar) < 0.2)
            span, is_drop = (4 if calm else 2), False
            for d in drops:
                if b < d < b + span:
                    span = d - b
        root = _bass_root(tonic + offset)
        chords.append(_Chord(b, span, root, root + 12 if root <= 36 else root, qual, is_drop))
        b += span

    def chord_at(second: float) -> _Chord:
        return next((c for c in reversed(chords) if c.bar * bar <= second), chords[0])

    # -- swells, strings and stabs
    stab_count = 0
    strings_led: list[float] = []
    for chord in chords:
        t = chord.bar * bar
        h = heat_in(t, t + chord.bars * bar)
        calm = chord.bars == 4 or h < 0.12
        kind = "swell_big" if chord.drop else ("swell_slow" if calm else "swell")
        lead = 0.6 if chord.drop else 1.5
        total = round((chord.bars * bar + (LIGHT_TAIL if light else SWELL_TAIL)) * 2.0) / 2.0
        put(
            max(t - lead, 0.0),
            total,
            chord.pad,
            0.9 if calm else 1.0,
            f"glitch:{kind}",
            "pad",
            chord.quality,
        )
        strings_led = _lead_strings(chord, strings_led, 1 if light else STRINGS_UP)
        svel = 0.8 if chord.drop else float(np.clip(0.4 + 0.3 * h, 0.4, 0.7))
        for note in [chord.pad + o for o in STRINGS_LOW] + strings_led:
            put(
                max(t - 0.2, 0.0),
                chord.bars * bar + 0.8,
                note,
                svel,
                "tape_strings",
                "str",
            )
        if chord.drop:
            continue
        if h >= 0.3 and drum_rng.random() < 0.25 + 0.7 * h and t > 8.0:
            vel = float(np.clip(0.55 + 0.35 * h, 0.5, 0.9))
            dur = float(drum_rng.choice((0.9, 1.0, 1.1, 1.3)))
            voice = "glitch:stab_long" if h > 0.75 else "glitch:stab"
            put(t, dur + (0.6 if h > 0.75 else 0.0), chord.pad, vel, voice, "stab", chord.quality)
            stab_count += 1

    # -- the motif of the room: the strings play it after the drop of the first and the last peak or plateau
    motif_candidates = {
        round(first / bar) + (MOTIF_AFTER_DROP if role == CLIMAX else 0)
        for first, _, role, _ in sections
        if role in {CLIMAX, PLATEAU}
    }
    motif_bars = sorted(m for m in motif_candidates if 0 < m < bars - 2) or [bars // 2]
    plan.choices[MOTIF_CHOICE] = str(len(motif_sites(motif_bars)))
    for number, bar_at in enumerate(motif_sites(motif_bars)):
        clock = bar_at * bar
        for midi, beats in _motif_notes(
            plan, develop_motif(plan.motif(), number, motif_seed(plan))
        ):
            put(clock, beats * beat, float(midi), MOTIF_VELOCITY, "tape_strings", "str")
            clock += beats * beat

    # -- the washes: stretched strings under everything, softer when the day is loud
    if not light:
        second = 0.0
        turn = int(rng.integers(0, 4))
        while second < length - 10.0:
            voice = wash_voices[turn % len(wash_voices)]
            # the washes sit on the root and the fifth of the day: two pitches per voice, made once each
            pitch = (
                _bass_root(tonic) + (0, 7)[(turn // 2) % 2] + (36 if voice == "wash:violin" else 24)
            )
            put(second, 24.0, float(pitch), 0.25 if voice == "wash:violin" else 0.6, voice, "wash")
            second += float(rng.uniform(15.0, 24.0))
            turn += 1

    # -- the bass: a phrase for every two bars, its row of the growl and its pattern from the heat
    bass_notes = 0
    state_by_bar: dict[int, int] = {}
    previous_state = -1
    for chord in chords:
        for k in range(0, chord.bars, 2):
            b0 = chord.bar + k
            t0 = b0 * bar
            h = heat_in(t0, t0 + 2 * bar, body)
            loud_drop = chord.drop and drops.get(chord.bar) != "far"
            if not (h >= limits["bass"] and plan.at(day, t0) > 0.35) and not loud_drop:
                previous_state = -1
                continue
            state = max(sum(1 for edge in GROWL_STATE_AT if h >= edge) - 1, 0)
            if previous_state >= 0:
                state = min(state, previous_state + 1)
            state = min(state, mode.growl_cap)
            if loud_drop:
                state = 5
            if light:
                state = min(state, 3)
            previous_state = state
            state_by_bar[b0] = state
            if state <= 1:
                phrase = hold
            elif state == 2:
                phrase = sync
            elif state == 3:
                phrase = sync if (b0 // 2) % 3 else busy
            else:
                phrase = busy if (b0 // 2) % 3 else sync
            if light and phrase is busy:
                phrase = sync
            for beat_at, beats, interval in phrase:
                semi = 1 if phrase is busy and phrygian and interval == 7 else interval
                phase8 = 2 * (round((beat_at * GROWL_RATE[state] % 1.0) * 4.0) % 4)
                put(
                    t0 + beat_at * beat,
                    beats * beat,
                    float(chord.root + semi),
                    1.0,
                    "glitch:growl",
                    "bass",
                    state * 8 + phase8,
                )
                bass_notes += 1

    # -- the drums: a pattern for every half bar, its level from the heat, growing by a step every two bars
    kicks: list[tuple[float, float]] = []
    kit_hits: dict[str, list[float]] = {"kick": [], "snare": []}
    levels: dict[int, int] = {}
    last_level = 0
    drop_bars = {d for d, kind in drops.items() if kind == "drums"}
    section_starts = sorted({round(first / bar) for first, *_ in sections if first > 0})
    role_of_bar = {
        b: role
        for first, last, role, _ in sections
        for b in range(round(first / bar), round(last / bar) + 1)
    }
    for b0 in range(0, bars, 2):
        t0 = b0 * bar
        h = heat_in(t0, t0 + 2 * bar, body)
        raw = sum(1 for key in ("kick", "hat", "ghost", "full", "big") if h >= limits[key])
        if not drums_ok(t0) or not drums_ok(t0 + 2 * bar - 0.1):
            raw = 0
        elif plan.at(form, t0) > 0.0 and state_by_bar and b0 > min(state_by_bar):
            raw = max(raw, mode.drum_floor)
        level = max(min(raw, last_level + 1), last_level - 2) if raw else 0
        level = min(level, mode.level_cap)
        if b0 in drop_bars:
            level = 5
        elif b0 - 2 in drop_bars and level < 3:
            level = max(level, 3)
        if light:
            level = min(level, 3)
        levels[b0] = last_level = level
    drum_events = 0
    fills = 0
    for b0, level in levels.items():
        if level == 0:
            continue
        kind = KINDS[level - 1]
        variants = pats[kind]
        role = role_of_bar.get(b0, PLATEAU)
        into = b0 + 2 in section_starts and b0 + 2 not in drop_bars
        floor = 0.0 if b0 in drop_bars else ROLE_KICK_FLOOR[role]
        span = variants[0]["span"]
        slot = 0
        while slot < SLOTS:
            t0 = b0 * bar + slot * 2 * beat
            last = slot + span >= SLOTS
            pattern = variants[(ROLE_VARIANT[role] + b0 // VARIANT_BARS + last) % len(variants)]
            window = (SLOTS - 1 - slot) * STEPS if into and last else None
            drum_events += _place_half_bar(
                put, kicks, kit_hits, pattern, t0, step, light, window, floor
            )
            slot += span
        if into:
            fill = family.fills[section_starts.index(b0 + 2) % len(family.fills)]
            drum_events += _place_half_bar(
                put, kicks, kit_hits, fill, b0 * bar + 3 * 2 * beat, step, light
            )
            fills += 1

    # -- the book of edits
    book = _Book()
    stats = {
        "brake": 0,
        "double": 0,
        "tape": 0,
        "stutter": 0,
        "freeze": 0,
        "gate": 0,
        "crush": 0,
        "reverse": 0,
        "pitchrep": 0,
        "bitdrop": 0,
        "roll": 0,
        "drop": 0,
    }

    def edit(
        t: float, kind: str, layers: tuple[str, ...], params: dict[str, Any], why: str
    ) -> bool:
        layers = alive(layers, t, t + glitch.edit_extent(kind, params))
        if not layers or t < 1.0 or t > length - 6.0:
            return False
        return book.add(
            Edit(
                round(t, 4),
                kind,
                layers,
                glitch.edit_extent(kind, params),
                glitch.edit_history(kind, params),
                params,
                why,
            )
        )

    def stutter_edit(
        t: float, layers: tuple[str, ...], slices: list[float], gains: list[float], why: str
    ) -> bool:
        return edit(t, "stutter", layers, {"slices": tuple(slices), "gains": tuple(gains)}, why)

    def roll_edit(
        t: float, layers: tuple[str, ...], slices: list[float], gains: list[float], why: str
    ) -> bool:
        return edit(t, "roll", layers, {"slices": tuple(slices), "gains": tuple(gains)}, why)

    def brake_edit(t: float, layers: tuple[str, ...], stop: float, span: float, why: str) -> bool:
        return edit(t, "brake", layers, {"stop": round(stop, 3), "span": span}, why)

    def tape_edit(  # noqa: PLR0913, PLR0917
        t: float, layers: tuple[str, ...], stop: float, span: float, wind: float, why: str
    ) -> bool:
        return edit(t, "tape", layers, {"stop": stop, "span": span, "wind": wind}, why)

    def freeze_edit(t: float, seed: int, why: str) -> bool:
        if light or edit_rng.random() < mode.grain_share:
            return edit(
                t,
                "grain",
                ("stab", "pad", "str"),
                {"length": beat * 1.5 * mode.stretch, "grain": 0.11, "seed": seed, "jitter": 0.02},
                why,
            )
        return edit(
            t,
            "spectral",
            ("pad", "str", "wash"),
            {"length": beat * 2.0 * mode.stretch, "back": 0.9, "seed": seed},
            why,
        )

    # the structure of the phrases: beat repeats, kick retriggers, snare rolls, a reversed snare into the accent
    phrase_edits = 0
    if mode.swells and not light:
        for start in section_starts:
            if start not in drops and start < bars:
                entering = chord_at(start * bar)
                put(
                    start * bar - bar,
                    bar,
                    float(entering.pad),
                    0.7,
                    "glitch:revstab",
                    "fx",
                    entering.quality,
                )
    for p in range(1, bars // 8 + 1):
        b_end = 8 * p - 1
        if b_end >= bars:
            break
        level = levels.get(2 * (b_end // 2), 0)
        if (
            level >= 3
            and b_end + 1 not in section_starts
            and drum_rng.random() < 0.55 + 0.4 * float(per_bar[b_end] - 0.5)
        ):
            t = b_end * bar + 3 * beat
            if roll_edit(t, KIT, [u16] * 2 + [u32] * 4, [1, 1, 0.9, 0.9, 0.8, 0.8], "phrase"):
                phrase_edits += 1
                stats["roll"] += 1
        b_mid = 8 * (p - 1) + int(drum_rng.choice((3, 5)))
        level = levels.get(2 * (b_mid // 2), 0)
        if level >= 2 and drum_rng.random() < 0.4:
            hits = [
                t
                for t in kit_hits["kick"]
                if b_mid * bar + beat < t < (b_mid + 1) * bar and (t - b_mid * bar) / step > 6
            ]
            if hits:
                t = hits[-1]
                if stutter_edit(t, ("kick",), [u32] * 4, [1, 0.9, 0.8, 0.7], "phrase"):
                    kicks.extend((t + i * u32, 0.6) for i in range(1, 4))
                    phrase_edits += 1
                    stats["stutter"] += 1
        b_roll = 8 * (p - 1) + int(drum_rng.choice((2, 6)))
        level = levels.get(2 * (b_roll // 2), 0)
        if level >= 3 and drum_rng.random() < 0.35:
            t = b_roll * bar + 2 * beat
            if any(abs(s - t) < 0.01 for s in kit_hits["snare"]) and roll_edit(
                t, ("snare",), [u8, u8, u16, u16], [0.8, 0.9, 1.0, 1.15], "phrase"
            ):
                phrase_edits += 1
                stats["roll"] += 1
        b_mark = 8 * (p - 1) + int(drum_rng.choice(PHRASE_MARK_BARS))
        if (
            mode.phrase_edits
            and b_mark not in section_starts
            and plan.at(day, b_mark * bar) > PHRASE_MARK_DAY
            and all(abs(b_mark - d) > PHRASE_MARK_CLEAR for d in drops)
        ):
            t = b_mark * bar
            buses = WHOLE if levels.get(2 * (b_mark // 2), 0) > 0 else SOFT
            mark = str(drum_rng.choice(PHRASE_MARK_KINDS))
            if mark == "tape":
                done = tape_edit(t - 1.5 * beat, buses, 0.45, 1.5 * beat, 0.18, "phrase")
            elif mark == "brake":
                done = brake_edit(t - 2 * step, buses, 0.1, 2 * step, "phrase")
            else:
                done = freeze_edit(t, int(t), "phrase")
            stats["freeze" if mark in {"spectral", "grain"} else mark] += int(done)
            phrase_edits += int(done)
        b_start = 8 * p
        if (
            b_start < bars
            and level >= 3
            and drum_rng.random() < 0.5
            and levels.get(b_start, 0) >= 3
        ):
            t = b_start * bar
            put(t - 1.0, 1.0, 0.0, 0.8, "glitch:revsnare", "fx")
            put(t, 0.5, 0.0, 1.0, "glitch:kick+", "kick")
            put(t, 0.5, 0.0, 1.0, "glitch:snare", "snare")
            kicks.append((t, 1.0))
            phrase_edits += 1

    # -- the drops: a reversed swell, a snare roll, a reversed snare, a tape stop on the mix, a silence, the drop
    drop_count = 0
    for d, kind in sorted(drops.items()):
        t_d = d * bar
        found = next((c for c in chords if c.bar == d), None)
        if found is None:
            continue
        chord = found
        drop_count += 1
        stats["drop"] += 1
        h = per_bar[d]
        if kind == "far":
            put(t_d - bar, bar, float(chord.pad), 0.5, "glitch:revstab", "fx", chord.quality)
            put(t_d, 4.0, float(chord.root), 0.55, "glitch:boomfar", "fx")
            continue
        put(t_d - bar, bar, float(chord.pad), 0.9, "glitch:revstab", "fx", chord.quality)
        put(t_d, 2.6, chord.pad, 1.0, "glitch:stab_big", "stab", chord.quality)
        put(t_d, 3.4, float(chord.root), 0.95, "glitch:boom", "fx")
        if kind == "hit":
            tape_edit(t_d - 0.5 * beat, SOFT, 0.38, 0.5 * beat, 0.0, "drop")
            continue
        # a snare roll one bar ahead: one hit repeated, with a crescendo; the reversed snare ends on the drop
        t_roll = t_d - bar + 2 * beat
        put(t_roll, 0.4, 0.0, 0.7, "glitch:snare", "snare")
        roll_edit(t_roll, ("snare",), [u8, u8, u16, u16], [0.7, 0.85, 1.0, 1.2], "drop")
        put(t_d - 1.1, 1.1, 0.0, 0.9, "glitch:revsnare", "fx")
        tape_edit(t_d - 0.5 * beat, WHOLE, 0.38, 0.5 * beat, 0.0, "drop")
        for event in taiko.roll(t_d - bar, t_d, "odaiko", 0.2, 0.95, 0.6, 0.11):
            put(event.time, 0.2, 36.0, event.velocity, "drum:odaiko", "fx")
        for when, vel in ((2 * beat, 0.95), (bar, 0.85), (bar + 2 * beat, 0.9), (2 * bar, 0.8)):
            put(t_d + when, 0.2, 36.0, vel, "drum:odaiko", "fx")
        # the closing accent and a beat repeat on the last beat of the big pattern
        stutter_edit(
            t_d + 2 * bar - beat, KIT, [u32] * 4 + [u16, u16], [1, 1, 0.9, 0.9, 0.8, 0.8], "drop"
        )
        put(t_d + 2 * bar, 0.5, 0.0, 1.0, "glitch:kick+", "kick")
        put(t_d + 2 * bar, 0.5, 0.0, 1.0, "glitch:snare", "snare")
        kicks.append((t_d + 2 * bar, 1.0))
        del h

    # -- the night: distant booms, every half minute or so, never on the grid of anything
    far_booms = 0
    clock = float(rng.uniform(14.0, 30.0))
    while clock < length - 12.0:
        if plan.at(plan.night, clock) > 0.7 and plan.at(day, clock) < 0.15:
            where = chord_at(clock)
            put(
                snap(clock),
                4.0,
                float(where.root),
                float(rng.uniform(0.3, 0.5)),
                "glitch:boomfar",
                "fx",
            )
            far_booms += 1
            clock += float(rng.uniform(26.0, 60.0))
        else:
            clock += 8.0

    # -- the house: every switch, door and appliance may bring an edit, on the grid, a few bars apart
    event_edits = 0
    last_end = -1e9
    triggers = list(plan.triggers)
    big_times = [d * bar for d in drops]
    for second, _strength, what in triggers:
        if second < 6.0 or second > length - 8.0:
            continue
        if any(t - bar <= second <= t + 3 * bar for t in big_times):
            continue
        h = float(plan.at(heat, second))
        kit_on = levels.get(2 * (int(second / bar) // 2), 0) > 0
        asleep = plan.at(day, second) < 0.2
        gap = bar * (2.5 if kit_on else (4.0 if h > 0.12 else 7.0))
        if (
            second - last_end < gap
            or edit_rng.random() > EDIT_CHANCE + EDIT_ENERGY * energy + EDIT_HEAT * h
        ):
            continue
        options = dict(EVENT_EDITS.get(what, DEFAULT_EDITS))
        weights = {k: w * favour.get(_family(k), 1.0) for k, w in options.items()}
        if kit_on:
            weights["stutter"] = weights.get("stutter", 0.0) * 1.0
        else:
            weights.pop("crush", None)
        if asleep:
            weights.pop("brake", None)
            weights.pop("tape", None)
        if light:
            pass
        if not weights:
            continue
        next_to = _kinds_next_to(book, second)
        fresh = {k: w for k, w in weights.items() if not CHOICE_KINDS[k] & next_to}
        choice = weighted(edit_rng, list(fresh or weights), list((fresh or weights).values()))
        t_beat = math.ceil(second / beat) * beat
        t_bar = math.ceil((second + 0.4) / bar) * bar
        done = False
        if choice == "brake":
            buses = WHOLE if kit_on else SOFT
            stop = float(edit_rng.uniform(0.07, 0.14))
            if edit_rng.random() < 0.35:
                first = brake_edit(t_bar - 6 * step, buses, stop, 2 * step, "event")
                done = (
                    brake_edit(
                        t_bar - 2 * step,
                        buses,
                        float(edit_rng.uniform(0.07, 0.14)),
                        2 * step,
                        "event",
                    )
                    or first
                )
                stats["double"] += int(done)
            else:
                done = brake_edit(t_bar - 2 * step, buses, stop, 2 * step, "event")
            stats["brake"] += int(done)
            end = t_bar
        elif choice == "tape":
            buses = WHOLE if kit_on else SOFT
            if edit_rng.random() < TAPE_LONG_SHARE * mode.stretch:
                done = tape_edit(t_bar - 1.5 * beat, buses, 0.45, 1.5 * beat, 0.18, "event")
            else:
                done = tape_edit(t_bar - 0.5 * beat, buses, 0.38, 0.5 * beat, 0.0, "event")
            stats["tape"] += int(done)
            end = t_bar
        elif choice == "stutter":
            if kit_on and edit_rng.random() < 0.4:
                done = roll_edit(
                    t_beat, KIT, [u16] * 2 + [u32] * 4, [1, 1, 0.9, 0.9, 0.8, 0.8], "event"
                )
                stats["roll"] += int(done)
            else:
                pick = int(edit_rng.integers(0, 3))
                slices, gains = (
                    ([u8] * 2 + [u16] * 2 + [u32] * 4, [1, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5]),
                    ([u16] * 4 + [u32] * 6, [1, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]),
                    ([u16] * 2 + [u32] * 4, [1, 0.9, 0.9, 0.8, 0.7, 0.6]),
                )[pick]
                done = stutter_edit(t_beat, SOFT, slices, gains, "event")
                stats["stutter"] += int(done)
            end = t_beat + 2 * beat
        elif choice == "gate":
            gate_pattern = GATE_PATTERNS[int(edit_rng.integers(0, len(GATE_PATTERNS)))]
            done = edit(
                t_beat,
                "gate",
                ("pad", "str", "wash", "bass"),
                {"step": u16, "pattern": gate_pattern, "floor": 0.05},
                "event",
            )
            stats["gate"] += int(done)
            end = t_beat + len(gate_pattern) * u16
        elif choice == "freeze":
            done = freeze_edit(t_beat, int(second), "event")
            stats["freeze"] += int(done)
            end = t_beat + 2.5 * beat * mode.stretch
        else:  # crush: one snare hit
            hits = [t for t in kit_hits["snare"] if second <= t < second + bar]
            if hits:
                done = edit(
                    hits[0], "crush", ("snare",), {"dur": 0.2, "factor": 4, "bits": 8}, "event"
                )
            stats["crush"] += int(done)
            end = second + bar
        if done:
            event_edits += 1
            last_end = end
            if asleep and what == "door" and edit_rng.random() < 0.6:
                where = chord_at(second)
                put(snap(second) + beat, 4.0, float(where.root), 0.35, "glitch:boomfar", "fx")
                far_booms += 1

    # -- the groove: where the kit plays, most bars may get an edit of their own, on the grid; the activity of the house
    # moves the chance, and an event of the house in a bar brings it and picks the kind
    groove_edits = 0
    groove_rng = plan.rng("glitch groove")
    house_bars: dict[int, str] = {}
    extreme_bars = _extreme_bars(sections, per_bar, bar, energy)
    for second, _strength, what in triggers:
        house_bars.setdefault(int(second / bar), what)
    for b in range(2, bars - 1):
        if (
            levels.get(2 * (b // 2), 0) == 0
            or b + 1 in section_starts
            or any(d - GROOVE_BEFORE_DROP <= b < d + GROOVE_AFTER_DROP for d in drops)
        ):
            continue
        extreme = b in extreme_bars
        chance = (
            EXTREME_CHANCE
            if extreme
            else GROOVE_CHANCE + GROOVE_ENERGY * energy + GROOVE_HEAT * per_bar[b]
        )
        bus_weights = EXTREME_BUSES if extreme else GROOVE_BUSES
        visit = house_bars.get(b)
        if visit is not None:
            chance = min(chance * GROOVE_HOUSE_BOOST, 1.0)
        if groove_rng.random() > chance:
            continue
        ordered = sorted(book.edits, key=lambda e: e.t)
        middle = b * bar + bar / 2.0
        before = [e.kind for e in ordered if e.t <= middle]
        next_to = _kinds_next_to(book, middle)
        for _ in range(GROOVE_TRIES):
            category = weighted(
                groove_rng, [c for c, _ in bus_weights], [w for _, w in bus_weights]
            )
            weights = {}
            for kind, base in GROOVE_KINDS.items():
                if (kind == "brake" and category != "whole") or kind in next_to:
                    continue
                weight = base * favour.get(_family(kind), 1.0)
                if visit is not None:
                    house = GROOVE_HOUSE.get(visit, GROOVE_HOUSE_OTHER)
                    weight *= house.get(kind, GROOVE_HOUSE_UNNAMED)
                if kind in GROOVE_RAMPS and b % 4 == 3:
                    weight *= GROOVE_RAMP_BOOST
                weights[kind] = weight * GROOVE_REPEAT_PENALTY ** before[-GROOVE_RECENT:].count(
                    kind
                )
            if not weights:
                break
            kind = weighted(groove_rng, list(weights), list(weights.values()))
            params, starts = GROOVE_SHAPES[kind](groove_rng, beat)
            if extreme and kind == "gate":
                params = {**params, "floor": EXTREME_GATE_FLOOR}
            t = b * bar + int(groove_rng.choice(starts)) * step
            reach = glitch.edit_extent(kind, params)
            if category == "instrument":
                open_buses = [(k, w) for k, w in INSTRUMENT_BUSES if alive((k,), t, t + reach)]
                if not open_buses:
                    continue
                cut_buses: tuple[str, ...] = (
                    weighted(groove_rng, [k for k, _ in open_buses], [w for _, w in open_buses]),
                )
            elif category == "kit":
                cut_buses = weighted(
                    groove_rng, [k for k, _ in KIT_BUSES], [w for _, w in KIT_BUSES]
                )
            else:
                cut_buses = ("pad", "str", "wash", "bass") if kind == "gate" else WHOLE
            if edit(t, kind, cut_buses, params, "groove" if visit is None else "house"):
                stats[kind] += 1
                groove_edits += 1
                break

    # a kind that would play a third time in a row gives way; the edits of the structure stay
    kept: list[Edit] = []
    for cut in sorted(book.edits, key=lambda e: e.t):
        if all(e.kind == cut.kind for e in kept[-MOST_IN_A_ROW:]) and len(kept) >= MOST_IN_A_ROW:
            if cut.why in GIVES_WAY:
                continue
            if kept[-1].why in GIVES_WAY:
                kept.pop()
        kept.append(cut)
    book.edits = kept

    # -- the layers
    count_curve = np.full(count, 3700.0)
    str_cut = count_curve * _dips(plan)
    layers = {
        "pad": Layer("pad", 1.0, None, None, 0.0, reverb=0.45, title="swells"),
        "str": Layer("str", 0.6, None, str_cut, 0.0, reverb=0.4, title="cello and violin sections"),
        "wash": Layer("wash", 0.55, None, str_cut, 0.0, reverb=0.4, title="stretched washes"),
        "stab": Layer("stab", 1.0, None, None, 0.0, reverb=0.45, title="brass-like stabs"),
        "bass": Layer("bass", 1.0, None, None, 0.0, reverb=0.0, title="growl bass"),
        "kick": Layer("kick", KICK_GAIN, None, None, 0.0, reverb=0.0, title="kick"),
        "snare": Layer("snare", SNARE_GAIN, None, None, 0.0, reverb=0.18, title="snare"),
        "hats": Layer("hats", HATS_GAIN, None, None, 0.0, reverb=0.0, title="hats, ticks and rim"),
        "fx": Layer(
            "fx", 1.0, None, None, 0.0, reverb=0.3, title="booms, reversed swells and taiko"
        ),
    }
    space = np.clip((0.80 + 0.10 * plan.night) / 0.9, 0.0, 1.0)
    if light:
        space = space * 0.55
    gain_db = np.full(length * CR + 1, MASTER_GAIN_DB)
    kicks.sort()
    setup = Glitch(
        bpm,
        sorted(book.edits, key=lambda e: e.t),
        kicks,
        dict(DUCK),
        ("kick", "snare", "hats"),
        vowels,
    )
    score = Score(
        length,
        sorted([n for n in notes if n.t < length], key=lambda n: (n.t, n.layer, n.voice, n.midi)),
        layers,
        [],
        space,
        tempo=bpm,
        events=list(plan.events),
        level_db=-20.0,
        gain_db=gain_db,
        glitch=setup,
    )
    score.choices = dict(plan.choices)
    kinds = ", ".join(f"{v} {k}" for k, v in stats.items() if v)
    score.notes_about = [
        (
            f"{mode.name} mode, {bpm:.0f} beats per minute, the kit counting {2 * bpm:.0f}; root {tonic_name_of(tonic)}, {mode_name}; "
            f"chords '{loop_name}' every two bars (four when the day is quiet or the house sleeps); the drop chord is a {'semitone above' if phrygian else 'tone below'} the root."
        ),
        (
            f"Growl '{growl_name}' ({bass_notes} notes, bass phrases '{bass_name}'), drum patterns '{drums_name}' ({drum_events} hits, {fills} fills), "
            f"{stab_count} stabs, washes '{wash_name}'; the heat of the house picks the row of the growl, the pattern of the kit and how often it is glitched."
        ),
        (
            f"Edits (favoured: {favour_name}): {kinds or 'none'}; {event_edits} from the events of the house, {groove_edits} from the groove of the bars, {phrase_edits} from the structure; "
            f"{drop_count} drops (the big moments), {far_booms} distant booms in the night."
        ),
    ]
    return score


# the wobbles per beat of each row of the growl: the phase of a note is where the wobble stands when it starts
GROWL_RATE = tuple(row[4] for row in glitch.GROWL_STATES)


def tonic_name_of(pitch_class: int) -> str:
    """Return the name of a pitch class."""
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[pitch_class % 12]


def _roll_shape(rng: np.random.Generator, beat: float) -> _Shape:
    u8, u16, u32 = beat / 2.0, beat / 4.0, beat / 8.0
    if rng.random() < ROLL_LONG_SHARE:
        return _Shape(
            {"slices": (u8, u16, u16, *[u32] * 4), "gains": (0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)},
            (BAR_STEPS - 12,),
        )
    return _Shape(
        {"slices": (u16, u16, *[u32] * 4), "gains": (0.75, 0.85, 0.9, 1.0, 1.1, 1.2)},
        (BAR_STEPS - 8,),
    )


def _reverse_shape(rng: np.random.Generator, beat: float) -> _Shape:
    steps = int(rng.choice(REVERSE_STEPS))
    last = BAR_STEPS - steps
    return _Shape({"length": steps * beat / 8.0}, (last, last, last - 8, last - 16))


def _pitch_shape(rng: np.random.Generator, beat: float) -> _Shape:
    rate = float(PITCH_RATES[int(rng.integers(0, len(PITCH_RATES)))])
    if rng.random() < PITCH_SIXTEENTH_SHARE:
        slice_s, repeats = beat / 4.0, int(rng.integers(3, 5))
    else:
        slice_s, repeats = beat / 2.0, 2
    gains = tuple(1.0 - PITCH_FADE * i for i in range(repeats))
    params = {"slice": slice_s, "repeats": repeats, "rate": rate, "gains": gains}
    return _Shape(params, (24, 24, 16, 8))


def _bitdrop_shape(rng: np.random.Generator, beat: float) -> _Shape:
    params = {
        "dur": beat / 2.0 if rng.random() < 0.5 else beat,
        "factor": int(rng.integers(BIT_DROP_FACTORS[0], BIT_DROP_FACTORS[1] + 1)),
        "bits": int(rng.integers(BIT_DROP_BITS[0], BIT_DROP_BITS[1] + 1)),
    }
    return _Shape(params, (0, 16, 24, 8))


def _gate_shape(rng: np.random.Generator, beat: float) -> _Shape:
    pattern = GROOVE_GATES[int(rng.integers(0, len(GROOVE_GATES)))]
    return _Shape({"step": beat / 2.0, "pattern": pattern, "floor": GATE_FLOOR}, (16, 8))


def _brake_shape(rng: np.random.Generator, beat: float) -> _Shape:
    stop = round(float(rng.uniform(*BRAKE_STOP)), 3)
    return _Shape({"stop": stop, "span": beat / 4.0}, (BAR_STEPS - 2,))


def _stutter_shape(rng: np.random.Generator, beat: float) -> _Shape:
    if rng.random() < 0.5:
        return _Shape({"slices": (beat / 2.0,) * 4, "gains": (1.0, 0.9, 0.8, 0.7)}, (16, 8))
    gains = tuple(1.0 - STUTTER_FADE * i for i in range(8))
    return _Shape({"slices": (beat / 4.0,) * 8, "gains": gains}, (16, 8))


# the parameters of each kind of groove edit, and the steps of a bar (an eighth of a beat) that it may start on; the ramps
# (a roll, a reversal, a brake) end on the bar line, the rest start on a beat or a half beat
GROOVE_SHAPES: dict[str, Callable[[np.random.Generator, float], _Shape]] = {
    "stutter": _stutter_shape,
    "roll": _roll_shape,
    "reverse": _reverse_shape,
    "pitchrep": _pitch_shape,
    "bitdrop": _bitdrop_shape,
    "gate": _gate_shape,
    "brake": _brake_shape,
}


def _extreme_bars(
    sections: list[tuple[float, float, str, float]], per_bar: list[float], bar: float, energy: float
) -> set[int]:
    """
    Return the bars where the groove edits go all the way: the start of the hottest peaks or plateaus that are hot enough.

    :param sections: The sections of the form, as seconds, role and level.
    :param per_bar: The heat of the house in each bar.
    :param bar: The length of a bar in seconds.
    :param energy: The energy of the day, which allows a second extreme section.
    """
    candidates = []
    for first, last, role, _ in sections:
        a, b = round(first / bar), round(last / bar)
        heat = float(np.mean(per_bar[a:b])) if b > a else 0.0
        if role in {CLIMAX, PLATEAU} and heat >= EXTREME_MIN_HEAT:
            candidates.append((heat, a, min(b, a + EXTREME_MAX_BARS)))
    low, high = EXTREME_SECTIONS
    chosen = sorted(candidates, reverse=True)[: low + round((high - low) * energy)]
    return {x for _, a, b in chosen for x in range(a, b)}


def _kinds_next_to(book: _Book, t: float) -> set[str]:
    """Return the kinds of the edits just before and just after a second."""
    before = max((e for e in book.edits if e.t <= t), key=lambda e: e.t, default=None)
    after = min((e for e in book.edits if e.t > t), key=lambda e: e.t, default=None)
    return {e.kind for e in (before, after) if e is not None}


def _family(kind: str) -> str:
    """Return the family of a kind of edit that the day may favour."""
    return {
        "crush": "stutter",
        "roll": "stutter",
        "bitdrop": "stutter",
        "pitchrep": "freeze",
        "reverse": "tape",
    }.get(kind, kind)


def _dips(plan: Plan) -> np.ndarray:
    """Return the dip of the cutoff at the anomalies (a filter that closes and opens, never a pitch)."""
    from .gen import dips  # noqa: PLC0415

    return dips(plan, 0.6, 0.9, 4.0)


def _place_half_bar(  # noqa: C901, PLR0913, PLR0917
    put: Callable[..., None],
    kicks: list[tuple[float, float]],
    kit_hits: dict[str, list[float]],
    pattern: Pattern,
    t0: float,
    step: float,
    light: bool,  # noqa: FBT001
    until: int | None = None,
    kick_floor: float = 0.0,
) -> int:
    """
    Place a pattern of the kit from second t0; return how many hits.

    :param until: Place only the steps before this one (the rest of the pattern gives way to a fill).
    :param kick_floor: Leave out the kicks softer than this.
    """
    count = 0
    place = put
    limit = until if until is not None else len(pattern["hats"])

    def at(s: int) -> float:
        return t0 + s * step

    for s, v in pattern["kick"]:
        if s < limit and v >= kick_floor:
            place(at(s), 0.5, 0.0, v, "glitch:kick", "kick")
            kicks.append((at(s), min(1.0, 0.5 + 0.5 * v)))
            kit_hits["kick"].append(at(s))
            count += 1
    for s, v in pattern["snare"]:
        if s < limit:
            place(at(s), 0.5, 0.0, v, "glitch:snare", "snare")
            kit_hits["snare"].append(at(s))
            count += 1
    for s, v in enumerate(pattern["hats"]):
        if v >= (LIGHT_HAT_FLOOR if light else 0.0) and v and s < limit:
            place(at(s), 0.2, 0.0, v, "glitch:hat", "hats")
            count += 1
    if light:
        return count
    lanes = (
        ("ghost", "glitch:ghost", "snare", 0.3, 0.0),
        ("open", "glitch:open", "hats", 0.5, 0.0),
        ("ticks", "glitch:tick", "hats", 0.2, 0.0),
        ("rim", "glitch:rim", "hats", 0.2, 0.0),
        ("toms", "drum:odaiko", "fx", 0.2, TOM_PITCH),
    )
    for name, voice, layer, dur, midi in lanes:
        for s, v in pattern[name]:
            if s < limit:
                place(at(s), dur, midi, v, voice, layer)
                count += 1
    return count


def _motif_notes(plan: Plan, motif: tuple[tuple[int, float], ...]) -> list[tuple[int, float]]:
    """Return a motif as (MIDI note, beats) in the key and mode of the day, from C4 up."""
    octave = 4
    while min(plan.scale.midi(degree, octave) for degree, _ in motif) < MOTIF_LOWEST:
        octave += 1
    return [(plan.scale.midi(degree, octave), beats) for degree, beats in motif]


def _lead_strings(chord: _Chord, previous: list[float], count: int) -> list[float]:
    """
    Return the upper voices of the strings of a chord, placed to move the least from the voices before.

    :param chord: The chord.
    :param previous: The upper voices of the chord before (empty for the first).
    :param count: How many upper voices: the third, then the fifth and the root.
    """
    third = 3 if chord.quality == 0 else 4
    tones = (third, 7, 0)[:count]
    first = [float(chord.pad + o) for o in (third + 12, 19, 24)[:count]]
    if not previous:
        return first
    choices = [
        [
            chord.pad + tone + 12 * k
            for k in range(3)
            if STRINGS_LOW[-1] < tone + 12 * k <= STRINGS_TOP + 5
        ]
        for tone in tones
    ]
    best = min(
        itertools.product(*choices),
        key=lambda voices: (
            sum(abs(a - b) for a, b in zip(sorted(voices), sorted(previous), strict=False)),
            sum(voices),
        ),
    )
    return [float(v) for v in sorted(best)]
