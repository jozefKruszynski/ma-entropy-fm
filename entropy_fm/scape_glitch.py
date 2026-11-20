"""
Cinematic Glitch: swells, brass-like stabs, a growling bass and hard half-time drums, glitched by what happens in the house.

The generative system is a film score cut up. A slow chord loop in a dark minor (aeolian or phrygian, on a root the
day deals) moves every two bars, four when the house sleeps, over a tempo of about 70 beats per minute whose drums
count 140: the kick on the first beat, a big snare on the third. The chords are saw swells that open as they rise,
a cello and violin section under them, stretched washes of strings, and brass-like stabs on the changes. The bass is a
growl that wobbles through vowels in time with the tempo; its character (drive, fold, FM, how fast it wobbles) is one
of six rows that the heat of the house picks, from a held note under the swells to the full drop. The drums are a
composed kit (a punchy kick, a layered snare with a dark plate, hats, rim, ticks, ghost snares, fills) with five
patterns, from a kick and a snare to the full groove, two variants each, and a drum bus that is glued and saturated;
the bass, pads, strings and stabs duck from every kick.

The edits are what the house plays: a stutter, a tape stop, a hard brake of 60 to 150 ms (a fast fall to nothing, a cut,
a gap of a sixteenth, a snap back on the grid, sometimes twice in a row), a grain or spectral freeze, a crush of one
hit, a gate chop, a beat repeat of the kit. A door, a switch, an appliance that starts or stops, a light, someone moving
brings one of them, on the grid and at most one in a couple of bars; which are favoured is dealt by the day. The big moments
are drop sequences: a reversed swell, a snare roll that stutters, a reversed snare, a tape stop on the whole mix, a silence,
then the drop (stab, sub boom, taiko, the full growl and the big pattern). They come at the peaks of the day and at an anomaly;
an anomaly in the night is a reversed swell into a distant boom, nothing else.

The day: the night is swells, washes and distant booms alone; as the house wakes the bass enters, then sparse drums, which grow
with the activity and the power of the appliances to the full pattern in the busy stretches; towards the evening the drums
thin out and leave, and the swells come back. The house never writes a note: the notes come from the chord loop, the bass phrases, the
patterns and the edits, and the day deals which of each.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import glitch, taiko
from .gen import weighted
from .score import CR, Edit, Glitch, Layer, Note, Score
from .synth import smoothstep

if TYPE_CHECKING:
    from collections.abc import Callable

    from .plan import Plan

# the tempo of the bar of four beats; the drums count twice that (kick on one, snare on three of every two beats)
TEMPOS = (68.0, 70.0, 72.0)
# the root of the day: the name and the pitch class
TONICS: tuple[tuple[str, int], ...] = (("F", 5), ("E", 4), ("G", 7), ("F#", 6), ("A", 9))
MODES: tuple[tuple[str, bool], ...] = (("aeolian", False), ("phrygian", True))
# the loops of chords: (semitones above the root, quality) for four chords of two bars; the drop chord of the aeolian days
LOOPS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("i bVI iv v", ((0, 0), (8, 1), (5, 0), (7, 0))),
    ("i bVII bVI v", ((0, 0), (10, 1), (8, 1), (7, 0))),
    ("i iv bVI bVII", ((0, 0), (5, 0), (8, 1), (10, 1))),
    ("i bIII iv v", ((0, 0), (3, 1), (5, 0), (7, 0))),
)
# the drop: a major chord a semitone above the root on phrygian days, a tone below it on aeolian ones
DROP_CHORD = {True: (1, 1), False: (10, 1)}
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
# the least of the heat that the base of the day (the bass, the first drums) is made from
BODY_FLOOR = 0.3
# what the house heat has to be for each part to enter (the day moves them by a few hundredths)
THRESHOLDS = {"bass": 0.26, "kick": 0.40, "hat": 0.50, "ghost": 0.58, "full": 0.70, "big": 0.84}
# the patterns of one half bar of the drums: sixteen steps of the 140 grid (an eighth of a beat of the tempo)
Pattern = dict[str, Any]
H8 = (0.5, 0, 0.3, 0, 0.5, 0, 0.3, 0, 0.5, 0, 0.3, 0, 0.5, 0, 0.3, 0)
H16 = (0.55, 0.22, 0.4, 0.22) * 3 + (0.55, 0.22, 0.4, 0.3)
H16B = (0.6, 0.25, 0.45, 0.25) * 3 + (0.6, 0.3, 0.5, 0.35)
H16C = (0.4, 0.2, 0.55, 0.22) * 3 + (0.4, 0.22, 0.55, 0.3)


def _p(  # noqa: PLR0913, PLR0917
    kick: list[tuple[int, float]],
    snare: list[tuple[int, float]],
    ghost: list[tuple[int, float]] | None = None,
    hats: tuple[float, ...] | None = None,
    rim: list[tuple[int, float]] | None = None,
    ticks: list[tuple[int, float]] | None = None,
    open_: list[tuple[int, float]] | None = None,
) -> Pattern:
    return {
        "kick": kick,
        "snare": snare,
        "ghost": ghost or [],
        "hats": hats or (),
        "rim": rim or [],
        "ticks": ticks or [],
        "open": open_ or [],
    }


KINDS = ("sparse", "light", "mid", "full", "big")
# fmt: off
DRUM_SETS: tuple[tuple[str, dict[str, list[Pattern]]], ...] = (
    ("straight", {
        "sparse": [_p([(0, 1.0)], [(8, 0.95)]), _p([(0, 1.0)], [(8, 0.95)], [(15, 0.18)])],
        "light": [_p([(0, 1.0), (10, 0.8)], [(8, 1.0)], None, H8),
                  _p([(0, 1.0), (10, 0.75), (13, 0.5)], [(8, 1.0)], [(15, 0.2)], H8)],
        "mid": [_p([(0, 1.0), (7, 0.7), (10, 0.85)], [(8, 1.0)], [(3, 0.22), (11, 0.2), (15, 0.25)], H16, [(5, 0.5)], [(2, 0.4), (6, 0.35), (14, 0.4)]),
                _p([(0, 1.0), (10, 0.85), (14, 0.6)], [(8, 1.0)], [(3, 0.22), (13, 0.2)], H16, [(11, 0.5)], [(2, 0.4), (6, 0.35), (12, 0.4)], [(6, 0.4)])],
        "full": [_p([(0, 1.0), (3, 0.6), (7, 0.8), (10, 0.95)], [(8, 1.0)], [(5, 0.22), (11, 0.22), (13, 0.2), (15, 0.28)], H16B, [(1, 0.4), (11, 0.55)], [(2, 0.45), (6, 0.4), (14, 0.45)], [(6, 0.45), (14, 0.5)]),
                 _p([(0, 1.0), (6, 0.7), (10, 0.95), (12, 0.7), (14, 0.8)], [(8, 1.0)], [(3, 0.22), (11, 0.2), (15, 0.3)], H16B, [(5, 0.5)], [(2, 0.45), (10, 0.4), (13, 0.4)], [(6, 0.45), (14, 0.5)])],
        "big": [_p([(0, 1.0), (3, 0.7), (7, 0.9), (10, 1.0), (14, 0.9)], [(8, 1.0)], [(5, 0.25), (11, 0.25), (13, 0.22), (15, 0.3)], H16B, [(1, 0.45), (11, 0.6)], [(2, 0.5), (6, 0.45), (10, 0.45)], [(6, 0.5), (14, 0.55)]),
                _p([(0, 1.0), (6, 0.8), (10, 1.0), (12, 0.8), (14, 0.9), (15, 0.6)], [(8, 1.0)], [(3, 0.25), (11, 0.25), (13, 0.2)], H16B, [(5, 0.55)], [(2, 0.5), (12, 0.45)], [(6, 0.5), (14, 0.55)])],
    }),
    ("off the beat", {
        "sparse": [_p([(0, 1.0)], [(8, 0.95)], [(14, 0.16)]), _p([(0, 1.0), (11, 0.5)], [(8, 0.95)])],
        "light": [_p([(0, 1.0), (11, 0.8)], [(8, 1.0)], None, H8),
                  _p([(0, 1.0), (6, 0.6), (11, 0.75)], [(8, 1.0)], [(15, 0.2)], H8)],
        "mid": [_p([(0, 1.0), (6, 0.7), (11, 0.85)], [(8, 1.0)], [(2, 0.2), (10, 0.2), (14, 0.25)], H16C, [(4, 0.5)], [(2, 0.4), (7, 0.35), (13, 0.4)]),
                _p([(0, 1.0), (3, 0.6), (11, 0.85)], [(8, 1.0)], [(5, 0.22), (13, 0.2)], H16C, [(12, 0.5)], [(1, 0.4), (6, 0.35), (14, 0.4)], [(7, 0.4)])],
        "full": [_p([(0, 1.0), (3, 0.6), (6, 0.8), (11, 0.95)], [(8, 1.0)], [(4, 0.22), (10, 0.22), (13, 0.2), (15, 0.28)], H16B, [(2, 0.4), (12, 0.55)], [(2, 0.45), (7, 0.4), (14, 0.45)], [(7, 0.45), (15, 0.5)]),
                 _p([(0, 1.0), (7, 0.7), (11, 0.95), (13, 0.7), (15, 0.7)], [(8, 1.0)], [(3, 0.22), (10, 0.2), (14, 0.3)], H16B, [(5, 0.5)], [(2, 0.45), (9, 0.4), (12, 0.4)], [(7, 0.45), (15, 0.5)])],
        "big": [_p([(0, 1.0), (3, 0.7), (6, 0.9), (11, 1.0), (13, 0.8), (15, 0.7)], [(8, 1.0)], [(4, 0.25), (10, 0.25), (14, 0.22)], H16B, [(1, 0.45), (12, 0.6)], [(2, 0.5), (7, 0.45), (13, 0.45)], [(7, 0.5), (15, 0.55)]),
                _p([(0, 1.0), (5, 0.7), (7, 0.8), (11, 1.0), (14, 0.9)], [(8, 1.0)], [(3, 0.25), (10, 0.25), (13, 0.2), (15, 0.3)], H16B, [(2, 0.55)], [(2, 0.5), (12, 0.45)], [(7, 0.5), (15, 0.55)])],
    }),
    ("stuttering", {
        "sparse": [_p([(0, 1.0)], [(8, 0.95)]), _p([(0, 1.0), (14, 0.5)], [(8, 0.95)], [(5, 0.16)])],
        "light": [_p([(0, 1.0), (10, 0.8), (14, 0.5)], [(8, 1.0)], None, H8),
                  _p([(0, 1.0), (7, 0.7)], [(8, 1.0)], [(13, 0.2)], H8)],
        "mid": [_p([(0, 1.0), (3, 0.6), (10, 0.85), (14, 0.7)], [(8, 1.0)], [(5, 0.22), (11, 0.2), (15, 0.25)], H16, [(7, 0.5)], [(2, 0.4), (6, 0.35), (12, 0.4)]),
                _p([(0, 1.0), (2, 0.5), (10, 0.85)], [(8, 1.0)], [(4, 0.22), (13, 0.2)], H16, [(11, 0.5)], [(3, 0.4), (6, 0.35), (14, 0.4)], [(6, 0.4)])],
        "full": [_p([(0, 1.0), (2, 0.55), (7, 0.8), (10, 0.95), (13, 0.6)], [(8, 1.0)], [(5, 0.22), (11, 0.22), (15, 0.28)], H16B, [(1, 0.4), (11, 0.55)], [(3, 0.45), (6, 0.4), (14, 0.45)], [(6, 0.45), (14, 0.5)]),
                 _p([(0, 1.0), (6, 0.7), (7, 0.6), (10, 0.95), (14, 0.8)], [(8, 1.0)], [(3, 0.22), (11, 0.2), (13, 0.25), (15, 0.3)], H16B, [(5, 0.5)], [(2, 0.45), (10, 0.4), (13, 0.4)], [(6, 0.45), (14, 0.5)])],
        "big": [_p([(0, 1.0), (2, 0.6), (3, 0.7), (7, 0.9), (10, 1.0), (14, 0.9)], [(8, 1.0)], [(5, 0.25), (11, 0.25), (13, 0.22), (15, 0.3)], H16B, [(1, 0.45), (11, 0.6)], [(2, 0.5), (6, 0.45), (10, 0.45)], [(6, 0.5), (14, 0.55)]),
                _p([(0, 1.0), (6, 0.8), (7, 0.6), (10, 1.0), (12, 0.8), (14, 0.9), (15, 0.6)], [(8, 1.0)], [(3, 0.25), (11, 0.25), (13, 0.2)], H16B, [(5, 0.55)], [(2, 0.5), (12, 0.45)], [(6, 0.5), (14, 0.55)])],
    }),
)
# fmt: on
# the favoured edits of the day: the weight of each family of edit
FAVOURS: tuple[tuple[str, dict[str, float]], ...] = (
    ("stutters", {"stutter": 3.0, "brake": 1.0, "freeze": 1.0, "gate": 1.0, "tape": 0.5}),
    ("brakes", {"stutter": 1.0, "brake": 3.0, "freeze": 1.0, "gate": 1.0, "tape": 0.7}),
    ("freezes", {"stutter": 1.0, "brake": 1.0, "freeze": 3.0, "gate": 1.0, "tape": 0.5}),
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

    bpm = plan.deal("tempo", TEMPOS)
    _, tonic = plan.deal("tonic", TONICS)
    mode_name, phrygian = plan.deal("mode", MODES)
    loop_name, loop = plan.deal("chord loop", LOOPS)
    drums_name, pats = plan.deal("drum patterns", DRUM_SETS)
    bass_name, (hold, sync, busy) = plan.deal("bass phrases", BASS_SETS)
    growl_name, vowels = plan.deal("growl character", GROWLS)
    favour_name, favour = plan.deal("favoured edits", FAVOURS)
    wash_name, wash_voices = plan.deal("washes", WASH_SETS)
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)

    beat = 60.0 / bpm
    bar = 4.0 * beat
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
    heat = np.clip(lively * day * 1.1, 0.0, 1.0)
    # the body of the day: the bass and the first drums are the base by day, and the house adds the rest; the drops,
    # the stabs and the edits follow the heat itself, which is how lively it is
    body = np.clip(plan.lift(lively, BODY_FLOOR) * day * 1.1, 0.0, 1.0)
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
    for second, _strength in plan.events:
        d = max(2 * round(second / bar / 2.0), 2)
        if d + 4 < bars:
            here = float(day[min(int(d * bar * CR), count - 1)])
            drops[d] = (
                "far"
                if here < 0.2
                else ("drums" if drums_ok(d * bar) and per_bar[d] > limits["kick"] else "hit")
            )
    ranked = sorted(range(4, max(bars - 6, 5)), key=lambda b: -smooth_bars[b])
    for b in ranked:
        d = 2 * (b // 2)
        if (
            smooth_bars[b] >= 0.60
            and drums_ok(d * bar)
            and all(abs(d - other) >= 20 for other in drops)
            and d + 4 < bars
            and len(drops) < (3 if light or plan.condensed else 6)
        ):
            drops[d] = "drums"

    # -- the chords: two bars each, four while the house sleeps, a drop where one is
    chords: list[_Chord] = []
    place, b = 0, 0
    drop_chord = DROP_CHORD[phrygian]
    while b < bars:
        if b in drops:
            offset, qual = drop_chord
            span, is_drop = 2, True
        else:
            offset, qual = loop[place % len(loop)]
            place += 1
            calm = b % 4 == 0 and heat_in(b * bar, (b + 4) * bar) < 0.2
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
    for chord in chords:
        t = chord.bar * bar
        h = heat_in(t, t + chord.bars * bar)
        calm = chord.bars == 4 or h < 0.12
        kind = "swell_big" if chord.drop else ("swell_slow" if calm else "swell")
        lead = 0.6 if chord.drop else 1.5
        total = round((chord.bars * bar + 4.5) * 2.0) / 2.0
        put(
            max(t - lead, 0.0),
            total,
            chord.pad,
            0.9 if calm and h < 0.05 else 1.0,
            f"glitch:{kind}",
            "pad",
            chord.quality,
        )
        string_notes = (
            (0, 7, 15 if chord.quality == 0 else 16)
            if light
            else (0, 7, 15 if chord.quality == 0 else 16, 19, 24)
        )
        svel = 0.8 if chord.drop else float(np.clip(0.4 + 0.3 * h, 0.4, 0.7))
        for o in string_notes:
            put(
                max(t - 0.2, 0.0),
                chord.bars * bar + 0.8,
                chord.pad + o,
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
    variant = int(drum_rng.integers(0, 2))
    for b0 in range(0, bars, 2):
        t0 = b0 * bar
        h = heat_in(t0, t0 + 2 * bar, body)
        raw = sum(1 for key in ("kick", "hat", "ghost", "full", "big") if h >= limits[key])
        if not drums_ok(t0) or not drums_ok(t0 + 2 * bar - 0.1):
            raw = 0
        level = max(min(raw, last_level + 1), last_level - 2) if raw else 0
        if b0 in drop_bars:
            level = 5
        elif b0 - 2 in drop_bars and level < 3:
            level = max(level, 3)
        if light:
            level = min(level, 3)
        levels[b0] = last_level = level
    drum_events = 0
    for b0, level in levels.items():
        if level == 0:
            continue
        kind = KINDS[level - 1]
        for half in range(4):
            t0 = b0 * bar + half * 2 * beat
            pattern = pats[kind][(half + variant + (b0 // 2)) % 2]
            drum_events += _place_half_bar(put, kicks, kit_hits, pattern, t0, step, light)

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
        "repeat": 0,
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

    def brake_edit(t: float, layers: tuple[str, ...], stop: float, span: float, why: str) -> bool:
        return edit(t, "brake", layers, {"stop": round(stop, 3), "span": span}, why)

    def tape_edit(  # noqa: PLR0913, PLR0917
        t: float, layers: tuple[str, ...], stop: float, span: float, wind: float, why: str
    ) -> bool:
        return edit(t, "tape", layers, {"stop": stop, "span": span, "wind": wind}, why)

    # the structure of the phrases: beat repeats, kick retriggers, snare rolls, a reversed snare into the accent
    phrase_edits = 0
    for p in range(1, bars // 8 + 1):
        b_end = 8 * p - 1
        if b_end >= bars:
            break
        level = levels.get(2 * (b_end // 2), 0)
        if level >= 3 and drum_rng.random() < 0.55 + 0.4 * float(per_bar[b_end] - 0.5):
            t = b_end * bar + 3 * beat
            if stutter_edit(t, KIT, [u16] * 2 + [u32] * 4, [1, 1, 0.9, 0.9, 0.8, 0.8], "phrase"):
                phrase_edits += 1
                stats["repeat"] += 1
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
            if any(abs(s - t) < 0.01 for s in kit_hits["snare"]) and stutter_edit(
                t, ("snare",), [u8, u8, u16, u16], [0.8, 0.9, 1.0, 1.15], "phrase"
            ):
                phrase_edits += 1
                stats["stutter"] += 1
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
        stutter_edit(t_roll, ("snare",), [u8, u8, u16, u16], [0.7, 0.85, 1.0, 1.2], "drop")
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
        if second - last_end < gap or edit_rng.random() > 0.55 + 0.3 * h:
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
        choice = weighted(edit_rng, list(weights), list(weights.values()))
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
            if edit_rng.random() < 0.5:
                done = tape_edit(t_bar - 1.5 * beat, buses, 0.45, 1.5 * beat, 0.18, "event")
            else:
                done = tape_edit(t_bar - 0.5 * beat, buses, 0.38, 0.5 * beat, 0.0, "event")
            stats["tape"] += int(done)
            end = t_bar
        elif choice == "stutter":
            if kit_on and edit_rng.random() < 0.4:
                done = stutter_edit(
                    t_beat, KIT, [u16] * 2 + [u32] * 4, [1, 1, 0.9, 0.9, 0.8, 0.8], "event"
                )
                stats["repeat"] += int(done)
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
            if light or edit_rng.random() < 0.5:
                done = edit(
                    t_beat,
                    "grain",
                    ("stab", "pad", "str"),
                    {"length": beat * 1.5, "grain": 0.11, "seed": int(second), "jitter": 0.02},
                    "event",
                )
            else:
                done = edit(
                    t_beat,
                    "spectral",
                    ("pad", "str", "wash"),
                    {"length": beat * 2.0, "back": 0.9, "seed": int(second)},
                    "event",
                )
            stats["freeze"] += int(done)
            end = t_beat + 2.5 * beat
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

    # -- the layers
    count_curve = np.full(count, 3700.0)
    str_cut = count_curve * _dips(plan)
    layers = {
        "pad": Layer("pad", 1.0, None, None, 0.0, reverb=0.45, title="swells"),
        "str": Layer("str", 0.6, None, str_cut, 0.0, reverb=0.4, title="cello and violin sections"),
        "wash": Layer("wash", 0.55, None, str_cut, 0.0, reverb=0.4, title="stretched washes"),
        "stab": Layer("stab", 1.0, None, None, 0.0, reverb=0.45, title="brass-like stabs"),
        "bass": Layer("bass", 1.0, None, None, 0.0, reverb=0.0, title="growl bass"),
        "kick": Layer("kick", 1.0, None, None, 0.0, reverb=0.0, title="kick"),
        "snare": Layer("snare", 1.0, None, None, 0.0, reverb=0.18, title="snare"),
        "hats": Layer("hats", 1.0, None, None, 0.0, reverb=0.0, title="hats, ticks and rim"),
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
            f"{bpm:.0f} beats per minute, the kit counting {2 * bpm:.0f} (kick on one, snare on three); root {tonic_name_of(tonic)}, {mode_name}; "
            f"chords '{loop_name}' every two bars, four when the house sleeps; the drop chord is a {'semitone above' if phrygian else 'tone below'} the root."
        ),
        (
            f"Growl '{growl_name}' ({bass_notes} notes, bass phrases '{bass_name}'), drum patterns '{drums_name}' ({drum_events} hits), "
            f"{stab_count} stabs, washes '{wash_name}'; the heat of the house picks the row of the growl, the pattern of the kit and how often it is glitched."
        ),
        (
            f"Edits (favoured: {favour_name}): {kinds or 'none'}; {event_edits} from the events of the house, {phrase_edits} from the structure; "
            f"{drop_count} drops (the big moments), {far_booms} distant booms in the night."
        ),
    ]
    return score


# the wobbles per beat of each row of the growl: the phase of a note is where the wobble stands when it starts
GROWL_RATE = tuple(row[4] for row in glitch.GROWL_STATES)


def tonic_name_of(pitch_class: int) -> str:
    """Return the name of a pitch class."""
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[pitch_class % 12]


def _family(kind: str) -> str:
    """Return the family of a kind of edit that the day may favour."""
    return {"crush": "stutter", "tape": "tape"}.get(kind, kind)


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
) -> int:
    """Place one half bar of the kit (sixteen steps) from second t0; return how many hits."""
    count = 0
    place = put

    def at(s: int) -> float:
        return t0 + s * step

    for s, v in pattern["kick"]:
        place(at(s), 0.5, 0.0, v, "glitch:kick", "kick")
        kicks.append((at(s), min(1.0, 0.5 + 0.5 * v)))
        kit_hits["kick"].append(at(s))
        count += 1
    for s, v in pattern["snare"]:
        place(at(s), 0.5, 0.0, v, "glitch:snare", "snare")
        kit_hits["snare"].append(at(s))
        count += 1
    for s, v in enumerate(pattern["hats"]):
        if v:
            place(at(s), 0.2, 0.0, v, "glitch:hat", "hats")
            count += 1
    if light:
        return count
    for s, v in pattern["ghost"]:
        place(at(s), 0.3, 0.0, v, "glitch:ghost", "snare")
        count += 1
    for s, v in pattern["open"]:
        place(at(s), 0.5, 0.0, v, "glitch:open", "hats")
        count += 1
    for s, v in pattern["ticks"]:
        place(at(s), 0.2, 0.0, v, "glitch:tick", "hats")
        count += 1
    for s, v in pattern["rim"]:
        place(at(s), 0.2, 0.0, v, "glitch:rim", "hats")
        count += 1
    return count
