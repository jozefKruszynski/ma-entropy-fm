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
warmth picks the character of the growl and the washes; the string voice and the bass voice (the growl, a reese or an FM
growl) are the room's own and stay the same every day (`signature`). The form of the day (intro, build, peak, release, plateau, outro) is
on phrases of eight bars: the intro and the outro are swells and washes alone, the bass and the kit enter and leave on the
phrase lines, and every peak begins with a drop that the build before it leads into. The strongest event of the day gets its
drop on the next phrase line, and the motif of the room is played by the strings after the drop of the first and the last peak (or at
the start of a plateau), the later one developed.

Who leads is the room's too (`lead`): a cinematic trailer piano or the strings play the motif in the daytime build, peak and
plateau; neither the growl nor the swells can carry a melody, and a stab on a single-note line stops sounding like a stab, so
there are two. The piano plays it slowly as intervals (a felt piano with open fifths and sixths, a grand piano in octaves or a
grand piano with open fifths and sixths, one for each room), in a long dark hall of its own (`render.hall_board`), ducks from
the kick like the strings and is in the edits that touch the strings or the stabs. A door, motion, an appliance or the hob
hands the next phrase to the piano, a light, a cover or the media to the strings; the lines go straight into the notes, so
that a handover never changes the edits.

The day: the night is swells, washes and distant booms alone; as the house wakes the bass enters, then sparse drums, which grow
with the activity and the power of the appliances to the full pattern in the busy stretches; towards the evening the drums
thin out and leave, and the swells come back. The house never writes a note: the notes come from the chord loop, the bass phrases, the
patterns and the edits, and the day deals which of each.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from . import glitch, lead, taiko
from .character import chord_bars, tempo_target
from .gen import dips, weighted
from .glitch_kits import DRUM_ENERGY, FAMILIES, KINDS, STEPS, Family, Pattern
from .harmony import (
    MOTIF_CHOICE,
    PIANO_FIFTH_STEPS,
    PIANO_HALL_SEND,
    PIANO_SIXTH_STEPS,
    SIGNED_TAG_SPAN,
    bars_fit,
    closeness,
    develop_motif,
    motif_notes,
    motif_seed,
    motif_sites,
)
from .plan import (
    BUILD,
    CLIMAX,
    FORM_BARS,
    INTRO,
    LIGHT_MAX_LAYERS,
    MIN_FORM_CELLS,
    OUTRO,
    PICK_TOLERANCE,
    PLATEAU,
    RELEASE,
    SHORT_FORM_BARS,
    TEMPO_TOLERANCE,
    TRIGGER_APPLIANCE,
    TRIGGER_DOOR,
    TRIGGER_LIGHT,
    TRIGGER_MOTION,
)
from .score import (
    CURVE_RATE,
    LAYER_BASS,
    LAYER_HATS,
    LAYER_KICK,
    LAYER_PIANO,
    LAYER_SNARE,
    LAYER_STR,
    LAYER_WASH,
    Edit,
    Glitch,
    Layer,
    Note,
    Score,
)
from .signature import SigVoice, gain_of_db, patch_of, signature, trim_layer
from .styles import GLITCH
from .synth import FIFTH_SEMITONES, SEMITONES_PER_OCTAVE, smoothstep
from .timeline import SECONDS_PER_MINUTE, stable_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from .plan import Plan


LAYER_FX = "fx"
LAYER_PAD = "pad"
LAYER_STAB = "stab"
EDIT_STUTTER = "stutter"
EDIT_ROLL = "roll"
EDIT_REVERSE = "reverse"
EDIT_PITCHREP = "pitchrep"
EDIT_BITDROP = "bitdrop"
EDIT_GATE = "gate"
EDIT_BRAKE = "brake"
EDIT_TAPE = "tape"
EDIT_FREEZE = "freeze"
EDIT_SPECTRAL = "spectral"
EDIT_CRUSH = "crush"
EDIT_GRAIN = "grain"
# the lengths of a note as parts of a beat: an eighth, a sixteenth and a thirty-second
EIGHTH = 2.0
SIXTEENTH = 4.0
THIRTY_SECOND = 8.0
# a long stab lasts one of these seconds, a wash goes up a fifth every other turn
STAB_LENGTHS = (0.9, 1.0, 1.1, 1.3)
WASH_FIFTHS = (0, FIFTH_SEMITONES)
# the gains of the stutters and rolls that a phrase and a drop make
KICK_STUTTER_GAINS = [1, 0.9, 0.8, 0.7]
PHRASE_ROLL_GAINS = [0.8, 0.9, 1.0, 1.15]
DROP_ROLL_GAINS = [0.7, 0.85, 1.0, 1.2]
# the taiko hits after a drop as (beats, bars, velocity) from the drop
DROP_TAIKO_HITS = ((2, 0, 0.95), (0, 1, 0.85), (2, 1, 0.9), (0, 2, 0.8))
# the strings play the fifth, then the fifth over an octave and two octaves over the root
FIFTH_ABOVE_OCTAVE = FIFTH_SEMITONES + SEMITONES_PER_OCTAVE
TWO_OCTAVES = 2 * SEMITONES_PER_OCTAVE


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
PHRASE_MARK_KINDS = (EDIT_TAPE, EDIT_SPECTRAL, EDIT_GRAIN, EDIT_BRAKE)
# the share of the tape stops that are long (a wind down of a beat and a half), times the stretch of the mode
TAPE_LONG_SHARE = 0.5
# the bars of the phrases of the structure edits, and the bar of a phrase that a kick retrigger or a snare roll falls on
PHRASE_EDIT_BARS = 8
# the kit level from which a phrase rolls the kit and the snare and plays an accent, and the one from which it retriggers a kick
PHRASE_EDIT_MIN_LEVEL = 3
KICK_STUTTER_MIN_LEVEL = 2
# the chance of the reversed snare into the accent of a phrase
ACCENT_CHANCE = 0.5
PHRASE_STUTTER_BARS = (3, 5)
PHRASE_ROLL_BARS = (2, 6)
# the tape stops: a long one winds down over a beat and a half and winds back up, a short one drops over half a beat
TAPE_LONG_BEATS, TAPE_LONG_STOP, TAPE_LONG_WIND = 1.5, 0.45, 0.18
TAPE_SHORT_BEATS, TAPE_SHORT_STOP = 0.5, 0.38
# the gains of a roll of the kit and of the beat repeat that closes a drop
ROLL_GAINS = (1, 1, 0.9, 0.9, 0.8, 0.8)
# why an edit is placed: by an event of the house, by the groove, by a phrase or by a drop
WHY_EVENT = "event"
WHY_HOUSE = "house"
WHY_GROOVE = "groove"
WHY_PHRASE = "phrase"
WHY_DROP = "drop"
# what the notes about the piece count besides the kinds of edit
STAT_DROP = "drop"
STAT_DOUBLE = "double"
# the voices of the kit and of the drops
VOICE_KICK = "glitch:kick"
VOICE_KICK_ACCENT = "glitch:kick+"
VOICE_SNARE = "glitch:snare"
VOICE_REVSTAB = "glitch:revstab"
VOICE_REVSNARE = "glitch:revsnare"
VOICE_BOOMFAR = "glitch:boomfar"
# the seconds that a hit of the kick or the snare rings, and that a boom of the far drop and of the night rings
KIT_HIT_HOLD = 0.5
FAR_BOOM_HOLD = 4.0
# the velocity of the reversed swell into a far drop
FAR_REVSTAB_VELOCITY = 0.5
# the kinds of edit that the notes about the piece count, in the order that they are told
STAT_KINDS = (
    EDIT_BRAKE,
    STAT_DOUBLE,
    EDIT_TAPE,
    EDIT_STUTTER,
    EDIT_FREEZE,
    EDIT_GATE,
    EDIT_CRUSH,
    EDIT_REVERSE,
    EDIT_PITCHREP,
    EDIT_BITDROP,
    EDIT_ROLL,
    STAT_DROP,
)


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
# the chance that a section plays a neighbouring loop of its mode, when the section before it played the loop of the day
LOOP_NEIGHBOUR_CHANCE = 0.4
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
# the most drops of a piece, on the quietest and the busiest day, before it is divided by DROPS_CAP_DIVISOR
MOST_DROPS = (1, 6)
# the heat adds drum drops only while the piece has fewer drops than the most drops divided by this
DROPS_CAP_DIVISOR = 2
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
INSTRUMENT_BUSES: tuple[tuple[str, float], ...] = (
    (LAYER_BASS, 0.4),
    (LAYER_STR, 0.35),
    (LAYER_STAB, 0.25),
)
KIT_BUSES: tuple[tuple[tuple[str, ...], float], ...] = (
    ((LAYER_KICK, LAYER_SNARE, LAYER_HATS), 0.4),
    ((LAYER_SNARE,), 0.25),
    ((LAYER_KICK,), 0.15),
    ((LAYER_HATS,), 0.2),
)
# the weight of each kind of groove edit, before the favour of the day and the event of the house
GROOVE_KINDS = {
    EDIT_STUTTER: 2.0,
    EDIT_ROLL: 1.5,
    EDIT_REVERSE: 2.0,
    EDIT_PITCHREP: 2.0,
    EDIT_BITDROP: 1.5,
    EDIT_GATE: 1.5,
    EDIT_BRAKE: 0.8,
}
# the kinds that lead into the next bar (a roll, a reversal, a brake) are more likely on the last bar of two or four
GROOVE_RAMPS = {EDIT_ROLL, EDIT_REVERSE, EDIT_BRAKE}
GROOVE_RAMP_BOOST = 2.0
# a kind that sounded in one of the last few edits is this much less likely
GROOVE_RECENT = 4
GROOVE_REPEAT_PENALTY = 0.4
# what each kind of event of the house brings to its bar; the kinds that it does not name are this likely
GROOVE_HOUSE: dict[str, dict[str, float]] = {
    TRIGGER_DOOR: {EDIT_REVERSE: 3.0, EDIT_BRAKE: 2.0, EDIT_ROLL: 1.0},
    TRIGGER_MOTION: {EDIT_STUTTER: 2.0, EDIT_ROLL: 2.0, EDIT_GATE: 2.0, EDIT_PITCHREP: 1.0},
    TRIGGER_LIGHT: {EDIT_PITCHREP: 3.0, EDIT_GATE: 2.0, EDIT_REVERSE: 1.0},
    TRIGGER_APPLIANCE: {EDIT_BITDROP: 3.0, EDIT_STUTTER: 1.0, EDIT_BRAKE: 1.0},
}
GROOVE_HOUSE_OTHER = {EDIT_STUTTER: 2.0, EDIT_GATE: 1.0, EDIT_PITCHREP: 1.0}
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
# a house stutter plays this many eighths, sixteenths and thirty-seconds of a beat, at these gains
HOUSE_STUTTERS = (
    (2, 2, 4, (1, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5)),
    (0, 4, 6, (1, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45)),
    (0, 2, 4, (1, 0.9, 0.9, 0.8, 0.7, 0.6)),
)
# the gains of the slices of a groove roll, and how many steps before the bar line it starts
ROLL_LONG_GAINS = (0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)
ROLL_SHORT_GAINS = (0.75, 0.85, 0.9, 1.0, 1.1, 1.2)
ROLL_LONG_BACK = 12
ROLL_SHORT_BACK = 8
# a reversal starts this many steps before the bar line less than its length, the other edits on these steps of a bar
REVERSE_BACKS = (0, 0, 8, 16)
PITCH_STARTS = (24, 24, 16, 8)
BITDROP_STARTS = (0, 16, 24, 8)
GATE_STARTS = (16, 8)
STUTTER_STARTS = (16, 8)
PITCH_REPEATS = (3, 5)
STUTTER_SHORT_SHARE = 0.5
STUTTER_SHORT_GAINS = (1.0, 0.9, 0.8, 0.7)
STUTTER_LONG_COUNT = 8
# the semitones of the third of a minor and of a major chord
MINOR_THIRD = 3
MAJOR_THIRD = 4
# the drop chord comes this many bars after the start of a peak; the motif of the room follows it
# the cheapest tier plays the accented hats alone (every other step)
LIGHT_HAT_FLOOR = 0.35
# how long a swell rings on past its chord: the cheapest tier lets it go sooner
SWELL_TAIL = 4.5
LIGHT_TAIL = 3.0
TOM_PITCH = 36.0
MOTIF_AFTER_DROP = 2
MOTIF_VELOCITY = 0.75
# a lead phrase of the strings has two statements of the motif, developed, this many bars in
LEAD_STATEMENTS = (1, 5)
LEAD_STATEMENT_BEATS = 16.0
LEAD_STRINGS_VELOCITY = 0.6
LEAD_STRINGS_TAIL = 1.05
# what each layer is trimmed by while it leads, so that every lead sits a little over the others and no more
LEAD_TRIM_DB = {LAYER_PIANO: 0.0, LAYER_STR: 0.0}
# the trailer piano plays the motif slowly, as intervals, in statements that start a bar into a lead phrase and have
# a breath between them; a note rings a little over twice its length, and the first of a statement has a soft low
# note a further octave down
PIANO_FIRST_BAR = 1
PIANO_STATEMENTS = 4
PIANO_VELOCITY = 0.7
PIANO_DEPTH_SHARE = 0.4
PIANO_STRETCH = 2.0
PIANO_RING = 2.0
PIANO_BREATH_BEATS = 2.0
PIANO_LOWEST, PIANO_HIGHEST = 60, 72
PIANO_CUTOFF_HZ = 4500.0
PIANO_ROOM_SEND = 0.4
# the whole piano layer, dry and both sends together, is this far down so that a piano-led stretch is not much louder than the rest of the piece
PIANO_LEVEL_DB = -6.0
PIANO_GAIN = gain_of_db(PIANO_LEVEL_DB)
# what each variant is trimmed by to sit 2 dB over the strings at a velocity of 0.7 (the felt piano is the quietest)
FELT_OPEN_TRIM_DB, GRAND_OCTAVES_TRIM_DB, GRAND_OPEN_TRIM_DB = 7.8, 3.8, 1.5
# the kit level and growl row of a drop, and the highest of either in the cheap tiers
DROP_LEVEL = 5
LIGHT_LEVEL_CAP = 3
# the lowest kit level two bars after a drop
AFTER_DROP_LEVEL = 3
# the cutoff dip at an anomaly: how deep, how long it holds, and how long it takes to open again
DIP_DEPTH = 0.6
DIP_HOLD_SECONDS = 0.9
DIP_RETURN_SECONDS = 4.0
BEAT_COUNT = 4.0
PIANO_CHOICE = "piano lead"
# an edit holds a bus from this many seconds before its start
EDIT_SLACK = 0.001
# the strings of a chord: the lower voices are the root and the fifth, the upper ones are led from chord to chord
STRINGS_LOW = (0, 7)
STRINGS_UP = 3
# the highest the upper strings go above the pad root (the root of the next octave but one is the limit)
STRINGS_TOP = 26
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
    (
        "stutters",
        {EDIT_STUTTER: 3.0, EDIT_BRAKE: 1.0, EDIT_FREEZE: 1.0, EDIT_GATE: 1.0, EDIT_TAPE: 0.5},
    ),
    (
        "brakes",
        {EDIT_STUTTER: 1.0, EDIT_BRAKE: 3.0, EDIT_FREEZE: 1.0, EDIT_GATE: 1.0, EDIT_TAPE: 0.7},
    ),
    (
        "freezes",
        {EDIT_STUTTER: 0.5, EDIT_BRAKE: 1.5, EDIT_FREEZE: 3.0, EDIT_GATE: 0.7, EDIT_TAPE: 1.5},
    ),
    (
        "stretches",
        {EDIT_STUTTER: 0.5, EDIT_BRAKE: 2.0, EDIT_FREEZE: 2.0, EDIT_GATE: 0.7, EDIT_TAPE: 3.0},
    ),
    (
        "gates",
        {EDIT_STUTTER: 1.0, EDIT_BRAKE: 1.0, EDIT_FREEZE: 1.0, EDIT_GATE: 3.0, EDIT_TAPE: 0.5},
    ),
)
# the stretched washes: which strings sound under the swells
WASH_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cellos", ("wash:cello",)),
    ("cellos and violins", ("wash:cello", "wash:violin")),
    ("cellos and felt piano", ("wash:cello", "wash:felt")),
)
# what each kind of event of the house may bring, and how much it likes it
EVENT_EDITS: dict[str, dict[str, float]] = {
    TRIGGER_DOOR: {EDIT_BRAKE: 3.0, EDIT_TAPE: 1.0, EDIT_STUTTER: 2.0},
    TRIGGER_MOTION: {EDIT_STUTTER: 3.0, EDIT_GATE: 2.0, EDIT_FREEZE: 2.0, EDIT_BRAKE: 1.0},
    TRIGGER_LIGHT: {EDIT_FREEZE: 3.0, EDIT_GATE: 2.0, EDIT_STUTTER: 1.0},
    TRIGGER_APPLIANCE: {EDIT_CRUSH: 3.0, EDIT_BRAKE: 2.0, EDIT_STUTTER: 1.0},
}
DEFAULT_EDITS = {EDIT_STUTTER: 2.0, EDIT_GATE: 1.0, EDIT_FREEZE: 1.0}
# the most that one kind plays in a row, and the edits that give way to keep it so
MOST_IN_A_ROW = 2
GIVES_WAY = {WHY_EVENT, WHY_HOUSE, WHY_GROOVE}
# the kinds of edit that each choice of the events of the house makes
CHOICE_KINDS = {
    EDIT_BRAKE: {EDIT_BRAKE},
    EDIT_TAPE: {EDIT_TAPE},
    EDIT_STUTTER: {EDIT_STUTTER, EDIT_ROLL},
    EDIT_GATE: {EDIT_GATE},
    EDIT_FREEZE: {EDIT_GRAIN, EDIT_SPECTRAL},
    EDIT_CRUSH: {EDIT_CRUSH},
}
# the layers that can lead: the room picks its own, and a happening of the day may hand a phrase to another; the
# piano is percussive and rhythmic at once, and neither the growl nor a pad can carry a melody here, so there are two
LEADS = (LAYER_PIANO, LAYER_STR)
LEAD_BY_TRAIT = {
    lead.PERCUSSIVE: LAYER_PIANO,
    lead.RHYTHMIC: LAYER_PIANO,
    lead.SUSTAINED: LAYER_STR,
}
# the bass root lies from this note up
BASS_LOWEST = 28
# an edit starts this far before its first second, so that nothing else is on its buses
EDIT_CLEARANCE = 0.001
END_SECONDS = 5.0
DAY_RISE_HOUR = 5.5
DAY_RISE_HOURS = 3.0
DAY_FALL_SHARE = 0.6
DAY_FALL_HOUR = 18.0
DAY_FALL_HOURS = 3.5
NIGHT_FALL_HOUR = 22.5
NIGHT_FALL_HOURS = 1.5
HEAT_BOOST = 1.1
THRESHOLD_JITTER = 0.04
FIRST_DRUM_HOUR = 6.5
LAST_DRUM_HOUR = 21.5
FAR_DROP_BELOW = 0.2
DROP_FIRST_BAR = 4
DROP_END_BARS = 6
DROP_MIN_END = 5
DROP_HEAT = 0.60
DROP_APART_BARS = 20
CHORD_SPAN_HEAT = 0.2
PAD_LIFT_BELOW = 36
CALM_HEAT = 0.12
SWELL_AHEAD_DROP = 0.6
SWELL_AHEAD = 1.5
CALM_VELOCITY = 0.9
STAB_DROP_VELOCITY = 0.8
STAB_VELOCITY_FLOOR = 0.4
STAB_VELOCITY_HEAT = 0.3
STAB_VELOCITY_MIN = 0.4
STAB_VELOCITY_MAX = 0.7
STAB_EARLY = 0.2
STAB_TAIL = 0.8
STAB_MIN_HEAT = 0.3
STAB_CHANCE = 0.25
STAB_CHANCE_HEAT = 0.7
STAB_AFTER = 8.0
LONG_STAB_VELOCITY = 0.55
LONG_STAB_HEAT = 0.35
LONG_STAB_MAX = 0.9
LONG_STAB_HEAT_FROM = 0.75
LONG_STAB_EXTRA = 0.6
WASH_END_MARGIN = 10.0
WASH_VIOLIN_BASE = 36
WASH_CELLO_BASE = 24
WASH_HOLD = 24.0
WASH_VIOLIN_VELOCITY = 0.25
WASH_CELLO_VELOCITY = 0.6
WASH_GAP_MIN = 15.0
WASH_GAP_MAX = 24.0
BASS_AWAKE_DAY = 0.35
BASS_PATCH_PHASES = 8
LEVEL_CLEAR = 0.1
SECTION_REVSTAB_VELOCITY = 0.7
PHRASE_ROLL_CHANCE = 0.55
PHRASE_ROLL_SHARE = 0.4
KICK_STUTTER_CHANCE = 0.4
KICK_STUTTER_AFTER_STEPS = 6
KICK_STUTTER_STRENGTH = 0.6
SNARE_ROLL_CHANCE = 0.35
SNARE_HIT_EPSILON = 0.01
PHRASE_BRAKE_STOP = 0.1
REVSNARE_VELOCITY = 0.8
BOOMFAR_VELOCITY = 0.55
REVSTAB_VELOCITY = 0.9
STAB_BIG_HOLD = 2.6
BOOM_HOLD = 3.4
BOOM_VELOCITY = 0.95
DROP_SNARE_HOLD = 0.4
DROP_SNARE_VELOCITY = 0.7
REVSNARE_LEAD = 1.1
REVSNARE_DROP_VELOCITY = 0.9
DROP_ROLL_FROM = 0.2
DROP_ROLL_TO = 0.95
DROP_ROLL_FIRST_GAP = 0.6
DROP_ROLL_LAST_GAP = 0.11
DRUM_HOLD = 0.2
DRUM_PITCH = 36.0
NIGHT_BOOM_FIRST_MIN = 14.0
NIGHT_BOOM_FIRST_MAX = 30.0
NIGHT_BOOM_END_MARGIN = 12.0
NIGHT_BOOM_NIGHT = 0.7
NIGHT_BOOM_DAY = 0.15
NIGHT_BOOM_VELOCITY_MIN = 0.3
NIGHT_BOOM_VELOCITY_MAX = 0.5
NIGHT_BOOM_GAP_MIN = 26.0
NIGHT_BOOM_GAP_MAX = 60.0
NIGHT_BOOM_RETRY = 8.0
NO_EDIT_YET = -1e9
HOUSE_EDIT_FROM = 6.0
HOUSE_EDIT_END_MARGIN = 8.0
ASLEEP_DAY = 0.2
KIT_GAP_BARS = 2.5
DAY_GAP_BARS = 4.0
NIGHT_GAP_BARS = 7.0
ASLEEP_DOOR_CHANCE = 0.6
HOUSE_EDIT_STRENGTH = 0.35
EDIT_SNAP = 0.4
HOUSE_BRAKE_CHANCE = 0.35
BRAKE_LEAD_STEPS = 6
HOUSE_STUTTER_CHANCE = 0.4
FREEZE_BEATS = 2.5
CRUSH_DUR = 0.2
CRUSH_FACTOR = 4
CRUSH_BITS = 8
STR_CUT_BASE = 3700.0
PAD_REVERB = 0.45
STR_GAIN = 0.6
STR_REVERB = 0.4
WASH_GAIN = 0.55
WASH_REVERB = 0.4
STAB_REVERB = 0.45
SNARE_REVERB = 0.18
FX_REVERB = 0.3
SPACE_BASE = 0.80
SPACE_NIGHT = 0.10
SPACE_DIVISOR = 0.9
SPACE_DROP = 0.55
LEVEL_DB = -20.0
EDIT_START = 1.0
EDIT_END_MARGIN = 6.0
FREEZE_LENGTH_BEATS = 1.5
FREEZE_GRAIN = 0.11
FREEZE_JITTER = 0.02
REVERSE_LENGTH_BEATS = 2.0
REVERSE_BACK = 0.9
HAT_HOLD = 0.2
STRINGS_HEADROOM = 5
PIANO_SCALE_OCTAVES_TO = 9
GATE_PATTERNS = (
    (1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1),
    (1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1),
)
WHOLE = (
    LAYER_PAD,
    LAYER_STR,
    LAYER_WASH,
    LAYER_STAB,
    LAYER_BASS,
    LAYER_KICK,
    LAYER_SNARE,
    LAYER_HATS,
)
SOFT = (LAYER_PAD, LAYER_STR, LAYER_WASH, LAYER_STAB, LAYER_BASS)
KIT = (LAYER_KICK, LAYER_SNARE, LAYER_HATS)
# the ducking of the layers from the kick: (depth, time constant)
DUCK = {
    LAYER_BASS: (0.7, 0.14),
    LAYER_PAD: (0.4, 0.22),
    LAYER_STR: (0.35, 0.22),
    LAYER_WASH: (0.35, 0.22),
    LAYER_STAB: (0.3, 0.12),
    LAYER_PIANO: (0.35, 0.22),
}
# the gain of the drum buses: the kick sits within a few dB of the strings, the hats about ten under the snare
KICK_GAIN, SNARE_GAIN, HATS_GAIN = 6.0, 1.8, 1.8
# the constant gain of the master, in dB: the arrangement is the dynamics of the style
MASTER_GAIN_DB = -4.2
GROWL_STATE_AT = (0.26, 0.34, 0.44, 0.56, 0.70, 0.84)


def compose(plan: Plan) -> Score:
    """
    Make the score of a Cinematic Glitch soundscape.

    :param plan: The curves and the deals of the day.
    """
    piece = _Piece(plan)
    _pick_character(piece)
    _pick_voices(piece)
    _set_up_time(piece)
    _draw_curves(piece)
    _find_drops(piece)
    _lay_chords(piece)
    _play_swells_and_stabs(piece)
    _choose_motif_sites(piece)
    _play_lead_lines(piece)
    _play_motif(piece)
    _play_washes(piece)
    _play_bass(piece)
    _set_up_drum_levels(piece)
    _play_drums(piece)
    _play_section_swells(piece)
    _play_phrase_edits(piece)
    _play_drop_sequences(piece)
    _play_night_booms(piece)
    _edit_for_house(piece)
    _edit_grooves(piece)
    _thin_repeats(piece)
    _add_layers(piece)
    return _assemble(piece)


# the wobbles per beat of each row of the growl: the phase of a note is where the wobble stands when it starts
GROWL_RATE = tuple(row[4] for row in glitch.GROWL_STATES)


def tonic_name_of(pitch_class: int) -> str:
    """Return the name of a pitch class."""
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[
        pitch_class % SEMITONES_PER_OCTAVE
    ]


class _Shape(NamedTuple):
    """The parameters of a groove edit, and the steps of a bar that it may start on."""

    params: dict[str, Any]
    starts: tuple[int, ...]


class _PianoVariant(NamedTuple):
    """A way to play the trailer piano: its voice and trim, and whether it has a fifth or sixth under the melody."""

    sig: SigVoice
    open_intervals: bool
    label: str


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
    return BASS_LOWEST + (pitch_class - BASS_LOWEST) % SEMITONES_PER_OCTAVE


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
        if not edit.layers or not self.free(edit.layers, edit.t - EDIT_CLEARANCE, b):
            return False
        for key in edit.layers:
            self.held.setdefault(key, []).append((a + edit.back - EDIT_CLEARANCE, b))
        self.edits.append(edit)
        return True


@dataclass
class _Piece:
    """The working state of one Glitch piece, read and appended to by the sections of `compose`."""

    plan: Plan
    length: int = field(init=False)
    light: bool = field(init=False)
    rng: np.random.Generator = field(init=False)
    drum_rng: np.random.Generator = field(init=False)
    edit_rng: np.random.Generator = field(init=False)
    groove_rng: np.random.Generator = field(init=False)
    energy: float = field(init=False)
    brightness: float = field(init=False)
    tonic: int = field(init=False)
    mode_name: str = field(init=False)
    phrygian: bool = field(init=False)
    mode: Mode = field(init=False)
    bpm: float = field(init=False)
    loop_name: str = field(init=False)
    loop: tuple[tuple[int, int], ...] = field(init=False)
    family: Family = field(init=False)
    bass_name: str = field(init=False)
    hold: Phrase = field(init=False)
    sync: Phrase = field(init=False)
    busy: Phrase = field(init=False)
    growl_name: str = field(init=False)
    vowels: int = field(init=False)
    favour_name: str = field(init=False)
    favour: dict[str, float] = field(init=False)
    wash_name: str = field(init=False)
    wash_voices: tuple[str, ...] = field(init=False)
    str_sig: SigVoice = field(init=False)
    bass_sig: SigVoice = field(init=False)
    beat: float = field(init=False)
    bar: float = field(init=False)
    phrase_bars: int = field(init=False)
    base_span: int = field(init=False)
    step: float = field(init=False)
    u8: float = field(init=False)
    u16: float = field(init=False)
    u32: float = field(init=False)
    bars: int = field(init=False)
    count: int = field(init=False)
    day: np.ndarray = field(init=False)
    heat: np.ndarray = field(init=False)
    body: np.ndarray = field(init=False)
    form: np.ndarray = field(init=False)
    limits: dict[str, float] = field(init=False)
    first_drum: float = field(init=False)
    last_drum: float = field(init=False)
    sections: list[tuple[float, float, str, float]] = field(init=False)
    lead_grid: float = field(init=False)
    default_lead: str = field(init=False)
    leads: lead.Leads = field(init=False)
    notes: list[Note] = field(default_factory=list)
    spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    per_bar: list[float] = field(init=False)
    drops: dict[int, str] = field(default_factory=dict)
    chords: list[_Chord] = field(default_factory=list)
    stab_count: int = 0
    motif_bars: list[int] = field(init=False)
    stated: set[int] = field(init=False)
    piano_variant: _PianoVariant = field(init=False)
    bass_notes: int = 0
    state_by_bar: dict[int, int] = field(default_factory=dict)
    kicks: list[tuple[float, float]] = field(default_factory=list)
    kit_hits: dict[str, list[float]] = field(
        default_factory=lambda: {LAYER_KICK: [], LAYER_SNARE: []}
    )
    levels: dict[int, int] = field(default_factory=dict)
    drop_bars: set[int] = field(init=False)
    section_starts: list[int] = field(init=False)
    role_of_bar: dict[int, str] = field(init=False)
    drum_events: int = 0
    fills: int = 0
    book: _Book = field(default_factory=_Book)
    stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STAT_KINDS, 0))
    phrase_edits: int = 0
    drop_count: int = 0
    far_booms: int = 0
    event_edits: int = 0
    groove_edits: int = 0
    layers: dict[str, Layer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Take what the plan already says; the first draw of the piece comes after it."""
        plan = self.plan
        self.length = plan.length
        self.light = plan.quality.max_layers <= LIGHT_MAX_LAYERS
        self.rng = plan.rng("glitch")
        self.drum_rng = plan.rng("glitch drums")
        self.edit_rng = plan.rng("glitch edits")
        self.energy = plan.character.energy
        self.brightness = 0.5 * plan.character.warmth + 0.5 * (2.0 * self.energy - 1.0)
        self.tonic = plan.scale.root


class _Placed(NamedTuple):
    """An edit of the house that was tried: whether it was placed and the second where its bar ends."""

    done: bool
    end: float


class _GrooveBar(NamedTuple):
    """What a groove edit of one bar looks at: the house, the kinds around it and the bar's own odds."""

    number: int
    extreme: bool
    visit: str | None
    before: list[str]
    next_to: set[str]


def _pick_character(piece: _Piece) -> None:
    """Pick the mode, the tempo, the chord loop and the family of drums from the energy of the day."""
    plan = piece.plan
    energy = piece.energy
    piece.mode_name = plan.scale.mode if plan.scale.mode in LOOPS else "aeolian"
    piece.phrygian = piece.mode_name == "phrygian"
    mode = plan.pick(
        "mode",
        MODES,
        [closeness(entry.centre, energy) for entry in MODES],
        tolerance=MODE_TOLERANCE,
    )
    piece.mode = mode
    low_bpm, high_bpm = mode.tempo
    target = tempo_target(GLITCH, plan.character)
    tempos = tuple(float(bpm) for bpm in range(low_bpm, high_bpm + 1))
    piece.bpm = plan.pick(
        "tempo",
        tempos,
        [closeness((t - low_bpm) / (high_bpm - low_bpm), target) for t in tempos],
        tolerance=TEMPO_TOLERANCE,
    )
    piece.loop_name, piece.loop = plan.pick(
        "chord loop",
        LOOPS[piece.mode_name],
        [closeness(entry.motion, energy) for entry in LOOPS[piece.mode_name]],
    )[:2]
    suited = tuple(f for f in FAMILIES if mode.name in f.modes)
    piece.family = plan.pick(
        "drum patterns",
        suited,
        [closeness(DRUM_ENERGY[f.name], energy) for f in suited],
        tolerance=FAMILY_TOLERANCE,
    )


def _pick_voices(piece: _Piece) -> None:
    """Pick the bass phrases, the growl, the favoured edits, the washes and the two voices of the room."""
    plan = piece.plan
    energy = piece.energy
    piece.bass_name, (piece.hold, piece.sync, piece.busy) = plan.pick(
        "bass phrases",
        BASS_SETS,
        [closeness(BASS_ENERGY[name], energy) for name, _ in BASS_SETS],
    )
    piece.growl_name, piece.vowels = plan.pick(
        "growl character",
        GROWLS,
        [
            closeness(GROWL_BRIGHTNESS[name], piece.brightness, SIGNED_TAG_SPAN)
            for name, _ in GROWLS
        ],
    )
    piece.favour_name, piece.favour = plan.pick(
        "favoured edits",
        FAVOURS,
        [closeness(FAVOUR_ENERGY[name], piece.mode.edit_energy) for name, _ in FAVOURS],
    )
    piece.wash_name, piece.wash_voices = plan.pick(
        "washes",
        WASH_SETS,
        [
            closeness(WASH_BRIGHTNESS[name], piece.brightness, SIGNED_TAG_SPAN)
            for name, _ in WASH_SETS
        ],
    )
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)
    piece.str_sig = signature(plan, LAYER_STR, "string voice")
    piece.bass_sig = signature(plan, LAYER_BASS, "bass voice")


def _set_up_time(piece: _Piece) -> None:
    """Fix the beat, the bar, the length of a phrase and of a chord, and the grid of the edits."""
    plan = piece.plan
    energy = piece.energy
    piece.beat = beat = SECONDS_PER_MINUTE / piece.bpm
    piece.bar = bar = 4.0 * beat
    piece.phrase_bars = (
        FORM_BARS if piece.length / (FORM_BARS * bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
    )
    span_options = tuple(
        sorted({min(max(o, 2), 4) for o in chord_bars(plan.character) if o <= piece.phrase_bars})
    )
    piece.base_span = plan.pick("chord bars", span_options, bars_fit(len(span_options), energy))
    piece.step = beat / THIRTY_SECOND
    piece.u8, piece.u16, piece.u32 = beat / EIGHTH, beat / SIXTEENTH, beat / THIRTY_SECOND
    piece.bars = int((piece.length - END_SECONDS) / bar)


def _draw_curves(piece: _Piece) -> None:
    """Draw the curves: the day (when the house is awake), the heat (how lively it is then), the parts and their thresholds."""
    plan = piece.plan
    bar = piece.bar
    energy = piece.energy
    piece.count = plan.count
    hour = plan.hour
    piece.day = np.asarray(
        smoothstep((hour - DAY_RISE_HOUR) / DAY_RISE_HOURS)
        * (1.0 - DAY_FALL_SHARE * smoothstep((hour - DAY_FALL_HOUR) / DAY_FALL_HOURS))
        * (1.0 - smoothstep((hour - NIGHT_FALL_HOUR) / NIGHT_FALL_HOURS))
    )
    lively = np.clip(plan.activity + 0.5 * plan.power, 0.0, 1.0)
    # the form: what each role lets through of the heat, from one phrase line to the next
    piece.sections = plan.sections(bar)
    piece.lead_grid = plan.form_grid(bar)
    piece.default_lead = lead.default_lead(plan, LEADS)
    piece.leads = lead.phrase_leads(
        plan,
        piece.lead_grid,
        piece.default_lead,
        lead.daytime_phrases(plan, piece.sections, piece.lead_grid),
        LEAD_BY_TRAIT,
    )
    piece.form = np.zeros(piece.count)
    for first, last, role, _ in piece.sections:
        piece.form[int(first * CURVE_RATE) : int(last * CURVE_RATE) + 1] = ROLE_HEAT[role]
    gain = HEAT_GAIN[0] + (HEAT_GAIN[1] - HEAT_GAIN[0]) * energy
    piece.heat = np.clip(lively * piece.day * HEAT_BOOST * gain * piece.form, 0.0, 1.0)
    # the body of the day: the bass and the first drums are the base by day, and the house adds the rest; the drops,
    # the stabs and the edits follow the heat itself, which is how lively it is
    floor = BODY_FLOOR[0] + (BODY_FLOOR[1] - BODY_FLOOR[0]) * energy
    piece.body = np.clip(
        plan.lift(lively, floor) * piece.day * HEAT_BOOST * gain * piece.form, 0.0, 1.0
    )
    piece.limits = {
        k: v + float(piece.rng.uniform(-THRESHOLD_JITTER, THRESHOLD_JITTER))
        for k, v in THRESHOLDS.items()
    }
    piece.first_drum = plan.piece_second_of_hour(FIRST_DRUM_HOUR)
    piece.last_drum = plan.piece_second_of_hour(LAST_DRUM_HOUR)


def _find_drops(piece: _Piece) -> None:
    """Find where the big moments are (an anomaly, and the peaks of the busy stretches), on a bar of even number."""
    bar = piece.bar
    piece.per_bar = [_heat_in(piece, b * bar, (b + 1) * bar) for b in range(piece.bars + 1)]
    _drops_at_events(piece)
    _drops_at_peaks(piece)
    _drops_at_heat(piece)


def _drops_at_events(piece: _Piece) -> None:
    """Put a drop at every anomaly, the strongest one on the next phrase line."""
    plan = piece.plan
    bar = piece.bar
    phrase_bars = piece.phrase_bars
    strongest = max(plan.events, key=lambda e: e[1])[0] if plan.events else -1.0
    for second, _strength in plan.events:
        d = max(2 * round(second / bar / 2.0), 2)
        if second == strongest:
            d = max(phrase_bars * math.ceil(second / (phrase_bars * bar)), phrase_bars)
        if d + 4 < piece.bars:
            here = float(piece.day[min(int(d * bar * CURVE_RATE), piece.count - 1)])
            piece.drops[d] = "far" if here < FAR_DROP_BELOW else _drums_or_hit(piece, d)


def _drops_at_peaks(piece: _Piece) -> None:
    """Put a drop at the start of every peak that has none."""
    for first, _, role, _ in piece.sections:
        d = round(first / piece.bar)
        if role == CLIMAX and d >= 4 and d + 4 < piece.bars and d not in piece.drops:
            piece.drops[d] = _drums_or_hit(piece, d)


def _drops_at_heat(piece: _Piece) -> None:
    """Put drums drops at the hottest bars that are far enough from the others, as many as the day allows."""
    bars = piece.bars
    drops = piece.drops
    smooth_bars = np.convolve(np.pad(piece.per_bar, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    cap = MOST_DROPS[0] + round((MOST_DROPS[1] - MOST_DROPS[0]) * piece.energy)
    ranked = sorted(
        range(DROP_FIRST_BAR, max(bars - DROP_END_BARS, DROP_MIN_END)),
        key=lambda b: -smooth_bars[b],
    )
    for b in ranked:
        d = 2 * (b // 2)
        if (
            smooth_bars[b] >= DROP_HEAT
            and _drums_ok(piece, d * piece.bar)
            and all(abs(d - other) >= DROP_APART_BARS for other in drops)
            and d + 4 < bars
            and len(drops) < cap // DROPS_CAP_DIVISOR
        ):
            drops[d] = "drums"


def _lay_chords(piece: _Piece) -> None:
    """Lay the chords: two bars each, four while the house sleeps, a drop where one is."""
    bar = piece.bar
    place, b = 0, 0
    drop_chord = DROP_CHORDS[piece.mode_name]
    starts = [first for first, _, _, _ in piece.sections]
    loops = _section_loops(piece)
    loop = piece.loop
    while b < piece.bars:
        section_loop = loops[max(bisect.bisect_right(starts, b * bar) - 1, 0)]
        if section_loop != loop:
            loop, place = section_loop, 0
        if b in piece.drops:
            offset, qual = drop_chord
            span, is_drop = 2, True
        else:
            offset, qual = loop[place % len(loop)]
            place += 1
            calm = b % 4 == 0 and (
                piece.base_span == 4 or _heat_in(piece, b * bar, (b + 4) * bar) < CHORD_SPAN_HEAT
            )
            span, is_drop = (4 if calm else 2), False
            for d in piece.drops:
                if b < d < b + span:
                    span = d - b
        root = _bass_root(piece.tonic + offset)
        piece.chords.append(
            _Chord(
                b,
                span,
                root,
                root + SEMITONES_PER_OCTAVE if root <= PAD_LIFT_BELOW else root,
                qual,
                is_drop,
            )
        )
        b += span


def _section_loops(piece: _Piece) -> list[tuple[tuple[int, int], ...]]:
    """Return the chord loop of each section: the one of the day, or now and then a close neighbour of its mode."""
    family = LOOPS[piece.mode_name]
    fits = [closeness(entry.motion, piece.energy) for entry in family]
    home = next(i for i, entry in enumerate(family) if entry.chords == piece.loop)
    others = sorted((i for i in range(len(family)) if i != home), key=lambda i: (-fits[i], i))
    near = [i for i in others if fits[i] >= max(fits) - PICK_TOLERANCE] or others[:1]
    rng = piece.plan.rng("chord loops")
    loops = [piece.loop]
    for _ in piece.sections[1:]:
        if loops[-1] == piece.loop and rng.random() < LOOP_NEIGHBOUR_CHANCE:
            loops.append(family[near[int(rng.integers(0, len(near)))]].chords)
        else:
            loops.append(piece.loop)
    return loops


def _play_swells_and_stabs(piece: _Piece) -> None:
    """Play the swells, the strings and the stabs, one set for each chord."""
    bar = piece.bar
    light = piece.light
    strings_led: list[float] = []
    for chord in piece.chords:
        t = chord.bar * bar
        h = _heat_in(piece, t, t + chord.bars * bar)
        calm = chord.bars == 4 or h < CALM_HEAT
        kind = "swell_big" if chord.drop else ("swell_slow" if calm else "swell")
        ahead = SWELL_AHEAD_DROP if chord.drop else SWELL_AHEAD
        total = round((chord.bars * bar + (LIGHT_TAIL if light else SWELL_TAIL)) * 2.0) / 2.0
        _put(
            piece,
            max(t - ahead, 0.0),
            total,
            chord.pad,
            CALM_VELOCITY if calm else 1.0,
            f"glitch:{kind}",
            LAYER_PAD,
            chord.quality,
        )
        strings_led = _lead_strings(chord, strings_led, 1 if light else STRINGS_UP)
        svel = (
            STAB_DROP_VELOCITY
            if chord.drop
            else float(
                np.clip(
                    STAB_VELOCITY_FLOOR + STAB_VELOCITY_HEAT * h,
                    STAB_VELOCITY_MIN,
                    STAB_VELOCITY_MAX,
                )
            )
        )
        for note in [chord.pad + o for o in STRINGS_LOW] + strings_led:
            _put(
                piece,
                max(t - STAB_EARLY, 0.0),
                chord.bars * bar + STAB_TAIL,
                note,
                svel,
                piece.str_sig.voice,
                LAYER_STR,
                patch_of(piece.str_sig, 0),
            )
        if chord.drop:
            continue
        if (
            h >= STAB_MIN_HEAT
            and piece.drum_rng.random() < STAB_CHANCE + STAB_CHANCE_HEAT * h
            and t > STAB_AFTER
        ):
            vel = float(np.clip(LONG_STAB_VELOCITY + LONG_STAB_HEAT * h, 0.5, LONG_STAB_MAX))
            dur = float(piece.drum_rng.choice(STAB_LENGTHS))
            voice = "glitch:stab_long" if h > LONG_STAB_HEAT_FROM else "glitch:stab"
            _put(
                piece,
                t,
                dur + (LONG_STAB_EXTRA if h > LONG_STAB_HEAT_FROM else 0.0),
                chord.pad,
                vel,
                voice,
                LAYER_STAB,
                chord.quality,
            )
            piece.stab_count += 1


def _choose_motif_sites(piece: _Piece) -> None:
    """Choose the bars of the motif of the room: after the drop of the first and the last peak or plateau."""
    bar = piece.bar
    plan = piece.plan
    candidates = {
        round(first / bar) + (MOTIF_AFTER_DROP if role == CLIMAX else 0)
        for first, _, role, _ in piece.sections
        if role in {CLIMAX, PLATEAU}
    }
    piece.motif_bars = sorted(m for m in candidates if 0 < m < piece.bars - 2) or [piece.bars // 2]
    plan.choices[MOTIF_CHOICE] = str(len(motif_sites(piece.motif_bars)))
    piece.stated = {int(bar_at * bar / piece.lead_grid) for bar_at in motif_sites(piece.motif_bars)}
    piece.piano_variant = _piano_variant(plan)


def _play_lead_lines(piece: _Piece) -> None:
    """Play the phrases that the piano or the strings lead."""
    # the lead's lines go straight into the notes, so that the edits (which look at what plays) are the same for any lead
    for number, who in enumerate(piece.leads.phrases):
        if who == LAYER_PIANO:
            piece.notes.extend(
                _piano_phrase(piece.plan, piece.piano_variant, piece.chords, number, piece.beat)
            )
        elif who == LAYER_STR and number not in piece.stated:
            _play_string_statements(piece, number)


def _play_string_statements(piece: _Piece, number: int) -> None:
    """Play the statements of the motif that the strings lead in a phrase."""
    plan = piece.plan
    beat = piece.beat
    sig = piece.str_sig
    for k, bars_in in enumerate(LEAD_STATEMENTS):
        motif = develop_motif(
            plan.motif(),
            len(LEAD_STATEMENTS) * number + k,
            motif_seed(plan),
            LEAD_STATEMENT_BEATS,
        )
        clock = number * piece.lead_grid + bars_in * piece.bar
        for midi, beats in motif_notes(plan, motif):
            piece.notes.append(
                Note(
                    clock,
                    beats * beat * LEAD_STRINGS_TAIL,
                    float(midi),
                    LEAD_STRINGS_VELOCITY,
                    sig.voice,
                    LAYER_STR,
                    patch_of(sig, 0),
                )
            )
            clock += beats * beat


def _play_motif(piece: _Piece) -> None:
    """Play the motif of the room on the strings at its bars, the later one developed."""
    plan = piece.plan
    beat = piece.beat
    for number, bar_at in enumerate(motif_sites(piece.motif_bars)):
        clock = bar_at * piece.bar
        for midi, beats in motif_notes(plan, develop_motif(plan.motif(), number, motif_seed(plan))):
            _put(
                piece,
                clock,
                beats * beat,
                float(midi),
                MOTIF_VELOCITY,
                piece.str_sig.voice,
                LAYER_STR,
                patch_of(piece.str_sig, 0),
            )
            clock += beats * beat


def _play_washes(piece: _Piece) -> None:
    """Play the washes: stretched strings under everything, softer when the day is loud."""
    if piece.light:
        return
    rng = piece.rng
    second = 0.0
    turn = int(rng.integers(0, 4))
    while second < piece.length - WASH_END_MARGIN:
        voice = piece.wash_voices[turn % len(piece.wash_voices)]
        # the washes sit on the root and the fifth of the day: two pitches per voice, made once each
        pitch = (
            _bass_root(piece.tonic)
            + WASH_FIFTHS[(turn // 2) % 2]
            + (WASH_VIOLIN_BASE if voice == "wash:violin" else WASH_CELLO_BASE)
        )
        _put(
            piece,
            second,
            WASH_HOLD,
            float(pitch),
            WASH_VIOLIN_VELOCITY if voice == "wash:violin" else WASH_CELLO_VELOCITY,
            voice,
            LAYER_WASH,
        )
        second += float(rng.uniform(WASH_GAP_MIN, WASH_GAP_MAX))
        turn += 1


def _play_bass(piece: _Piece) -> None:
    """Play the bass: a phrase for every two bars, its row of the growl and its pattern from the heat."""
    bar = piece.bar
    previous_state = -1
    for chord in piece.chords:
        for k in range(0, chord.bars, 2):
            b0 = chord.bar + k
            t0 = b0 * bar
            h = _heat_in(piece, t0, t0 + 2 * bar, piece.body)
            loud_drop = chord.drop and piece.drops.get(chord.bar) != "far"
            awake = h >= piece.limits["bass"] and piece.plan.at(piece.day, t0) > BASS_AWAKE_DAY
            if not awake and not loud_drop:
                previous_state = -1
                continue
            state = _growl_state(piece, h, previous_state, loud_drop=loud_drop)
            previous_state = state
            piece.state_by_bar[b0] = state
            _play_bass_phrase(piece, chord, b0, state)


def _growl_state(piece: _Piece, h: float, previous_state: int, *, loud_drop: bool) -> int:
    """Return the row of the growl that the heat picks, held to a step up from the last one."""
    state = max(sum(1 for edge in GROWL_STATE_AT if h >= edge) - 1, 0)
    if previous_state >= 0:
        state = min(state, previous_state + 1)
    state = min(state, piece.mode.growl_cap)
    if loud_drop:
        state = DROP_LEVEL
    if piece.light:
        state = min(state, LIGHT_LEVEL_CAP)
    return state


def _play_bass_phrase(piece: _Piece, chord: _Chord, b0: int, state: int) -> None:
    """Play the bass phrase of two bars from bar b0, in a row of the growl."""
    beat = piece.beat
    if state <= 1:
        phrase = piece.hold
    elif state == 2:
        phrase = piece.sync
    elif state == 3:
        phrase = piece.sync if (b0 // 2) % 3 else piece.busy
    else:
        phrase = piece.busy if (b0 // 2) % 3 else piece.sync
    if piece.light and phrase is piece.busy:
        phrase = piece.sync
    for beat_at, beats, interval in phrase:
        semi = (
            1
            if phrase is piece.busy and piece.phrygian and interval == FIFTH_SEMITONES
            else interval
        )
        phase8 = 2 * (round((beat_at * GROWL_RATE[state] % 1.0) * 4.0) % 4)
        _put(
            piece,
            b0 * piece.bar + beat_at * beat,
            beats * beat,
            float(chord.root + semi),
            1.0,
            piece.bass_sig.voice,
            LAYER_BASS,
            patch_of(piece.bass_sig, state * BASS_PATCH_PHASES + phase8),
        )
        piece.bass_notes += 1


def _set_up_drum_levels(piece: _Piece) -> None:
    """Set the level of the kit for every two bars from the heat, growing by a step every two bars."""
    bar = piece.bar
    sections = piece.sections
    piece.drop_bars = {d for d, kind in piece.drops.items() if kind == "drums"}
    piece.section_starts = sorted({round(first / bar) for first, *_ in sections if first > 0})
    piece.role_of_bar = {
        b: role
        for first, last, role, _ in sections
        for b in range(round(first / bar), round(last / bar) + 1)
    }
    last_level = 0
    for b0 in range(0, piece.bars, 2):
        level = _level_at(piece, b0, last_level)
        piece.levels[b0] = last_level = level


def _level_at(piece: _Piece, b0: int, last_level: int) -> int:
    """Return the level of the kit (0 to 5) for the two bars from bar b0."""
    bar = piece.bar
    mode = piece.mode
    t0 = b0 * bar
    h = _heat_in(piece, t0, t0 + 2 * bar, piece.body)
    raw = sum(1 for key in ("kick", "hat", "ghost", "full", "big") if h >= piece.limits[key])
    if not _drums_ok(piece, t0) or not _drums_ok(piece, t0 + 2 * bar - LEVEL_CLEAR):
        raw = 0
    elif (
        piece.plan.at(piece.form, t0) > 0.0 and piece.state_by_bar and b0 > min(piece.state_by_bar)
    ):
        raw = max(raw, mode.drum_floor)
    level = max(min(raw, last_level + 1), last_level - 2) if raw else 0
    level = min(level, mode.level_cap)
    if b0 in piece.drop_bars:
        level = DROP_LEVEL
    elif b0 - 2 in piece.drop_bars and level < AFTER_DROP_LEVEL:
        level = max(level, AFTER_DROP_LEVEL)
    if piece.light:
        level = min(level, LIGHT_LEVEL_CAP)
    return level


def _kit_level(piece: _Piece, bar: int) -> int:
    """Return the level of the kit in a bar, which is set for every two bars; 0 when it does not play."""
    return piece.levels.get(2 * (bar // 2), 0)


def _play_drums(piece: _Piece) -> None:
    """Play a pattern of the kit for every half bar, and a fill into every new section."""
    bar = piece.bar
    beat = piece.beat
    family = piece.family
    for b0, level in piece.levels.items():
        if level == 0:
            continue
        variants = family.patterns[KINDS[level - 1]]
        role = piece.role_of_bar.get(b0, PLATEAU)
        into = b0 + 2 in piece.section_starts and b0 + 2 not in piece.drop_bars
        floor = 0.0 if b0 in piece.drop_bars else ROLE_KICK_FLOOR[role]
        span = variants[0]["span"]
        slot = 0
        while slot < SLOTS:
            t0 = b0 * bar + slot * 2 * beat
            last = slot + span >= SLOTS
            pattern = variants[(ROLE_VARIANT[role] + b0 // VARIANT_BARS + last) % len(variants)]
            window = (SLOTS - 1 - slot) * STEPS if into and last else None
            piece.drum_events += _place_half_bar(piece, pattern, t0, window, floor)
            slot += span
        if into:
            fill = family.fills[piece.section_starts.index(b0 + 2) % len(family.fills)]
            piece.drum_events += _place_half_bar(piece, fill, b0 * bar + 3 * 2 * beat)
            piece.fills += 1


def _play_section_swells(piece: _Piece) -> None:
    """Play a reversed swell into every section after the first, in a mode that has them."""
    if not piece.mode.swells or piece.light:
        return
    bar = piece.bar
    for start in piece.section_starts:
        if start not in piece.drops and start < piece.bars:
            entering = _chord_at(piece.chords, start * bar, bar)
            _put(
                piece,
                start * bar - bar,
                bar,
                float(entering.pad),
                SECTION_REVSTAB_VELOCITY,
                VOICE_REVSTAB,
                LAYER_FX,
                entering.quality,
            )


def _play_phrase_edits(piece: _Piece) -> None:
    """Play the structure of the phrases: beat repeats, kick retriggers, snare rolls, a reversed snare into the accent."""
    for p in range(1, piece.bars // PHRASE_EDIT_BARS + 1):
        if PHRASE_EDIT_BARS * p - 1 >= piece.bars:
            break
        _phrase_roll(piece, p)
        _phrase_kick_stutter(piece, p)
        level = _phrase_snare_roll(piece, p)
        _phrase_mark(piece, p)
        _phrase_accent(piece, p, level)


def _phrase_roll(piece: _Piece, p: int) -> None:
    """Roll the kit into the end of phrase p."""
    b_end = PHRASE_EDIT_BARS * p - 1
    level = _kit_level(piece, b_end)
    if (
        level >= PHRASE_EDIT_MIN_LEVEL
        and b_end + 1 not in piece.section_starts
        and piece.drum_rng.random()
        < PHRASE_ROLL_CHANCE + PHRASE_ROLL_SHARE * float(piece.per_bar[b_end] - 0.5)
    ):
        t = b_end * piece.bar + 3 * piece.beat
        u16, u32 = piece.u16, piece.u32
        if _roll_edit(piece, t, KIT, [u16] * 2 + [u32] * 4, list(ROLL_GAINS), WHY_PHRASE):
            piece.phrase_edits += 1
            piece.stats[EDIT_ROLL] += 1


def _phrase_kick_stutter(piece: _Piece, p: int) -> None:
    """Retrigger the last kick of a bar in the middle of phrase p."""
    bar = piece.bar
    b_mid = PHRASE_EDIT_BARS * (p - 1) + int(piece.drum_rng.choice(PHRASE_STUTTER_BARS))
    level = _kit_level(piece, b_mid)
    if level >= KICK_STUTTER_MIN_LEVEL and piece.drum_rng.random() < KICK_STUTTER_CHANCE:
        hits = [
            t
            for t in piece.kit_hits[LAYER_KICK]
            if b_mid * bar + piece.beat < t < (b_mid + 1) * bar
            and (t - b_mid * bar) / piece.step > KICK_STUTTER_AFTER_STEPS
        ]
        if hits:
            t = hits[-1]
            u32 = piece.u32
            if _stutter_edit(piece, t, (LAYER_KICK,), [u32] * 4, KICK_STUTTER_GAINS, WHY_PHRASE):
                piece.kicks.extend((t + i * u32, KICK_STUTTER_STRENGTH) for i in range(1, 4))
                piece.phrase_edits += 1
                piece.stats[EDIT_STUTTER] += 1


def _phrase_snare_roll(piece: _Piece, p: int) -> int:
    """Roll the snare in phrase p; return the level of the kit there."""
    b_roll = PHRASE_EDIT_BARS * (p - 1) + int(piece.drum_rng.choice(PHRASE_ROLL_BARS))
    level = _kit_level(piece, b_roll)
    if level >= PHRASE_EDIT_MIN_LEVEL and piece.drum_rng.random() < SNARE_ROLL_CHANCE:
        t = b_roll * piece.bar + 2 * piece.beat
        u8, u16 = piece.u8, piece.u16
        if any(abs(s - t) < SNARE_HIT_EPSILON for s in piece.kit_hits[LAYER_SNARE]) and _roll_edit(
            piece, t, (LAYER_SNARE,), [u8, u8, u16, u16], PHRASE_ROLL_GAINS, WHY_PHRASE
        ):
            piece.phrase_edits += 1
            piece.stats[EDIT_ROLL] += 1
    return level


def _phrase_mark(piece: _Piece, p: int) -> None:
    """Mark phrase p of a heavy day with a long freeze, a tape stop or a brake."""
    bar = piece.bar
    b_mark = PHRASE_EDIT_BARS * (p - 1) + int(piece.drum_rng.choice(PHRASE_MARK_BARS))
    if (
        piece.mode.phrase_edits
        and b_mark not in piece.section_starts
        and piece.plan.at(piece.day, b_mark * bar) > PHRASE_MARK_DAY
        and all(abs(b_mark - d) > PHRASE_MARK_CLEAR for d in piece.drops)
    ):
        t = b_mark * bar
        buses = WHOLE if _kit_level(piece, b_mark) > 0 else SOFT
        mark = str(piece.drum_rng.choice(PHRASE_MARK_KINDS))
        if mark == EDIT_TAPE:
            done = _tape_long(piece, t, buses, WHY_PHRASE)
        elif mark == EDIT_BRAKE:
            done = _brake_edit(
                piece, t - 2 * piece.step, buses, PHRASE_BRAKE_STOP, 2 * piece.step, WHY_PHRASE
            )
        else:
            done = _freeze_edit(piece, t, int(t), WHY_PHRASE)
        piece.stats[EDIT_FREEZE if mark in {EDIT_SPECTRAL, EDIT_GRAIN} else mark] += int(done)
        piece.phrase_edits += int(done)


def _phrase_accent(piece: _Piece, p: int, level: int) -> None:
    """Play a reversed snare into the accent on the first bar after phrase p, when the kit plays there."""
    b_start = PHRASE_EDIT_BARS * p
    if (
        b_start < piece.bars
        and level >= PHRASE_EDIT_MIN_LEVEL
        and piece.drum_rng.random() < ACCENT_CHANCE
        and _kit_level(piece, b_start) >= PHRASE_EDIT_MIN_LEVEL
    ):
        t = b_start * piece.bar
        _put(piece, t - 1.0, 1.0, 0.0, REVSNARE_VELOCITY, VOICE_REVSNARE, LAYER_FX)
        _put(piece, t, KIT_HIT_HOLD, 0.0, 1.0, VOICE_KICK_ACCENT, LAYER_KICK)
        _put(piece, t, KIT_HIT_HOLD, 0.0, 1.0, VOICE_SNARE, LAYER_SNARE)
        piece.kicks.append((t, 1.0))
        piece.phrase_edits += 1


def _play_drop_sequences(piece: _Piece) -> None:
    """Play the drops: a reversed swell, a snare roll, a reversed snare, a tape stop on the mix, a silence, the drop."""
    for d, kind in sorted(piece.drops.items()):
        chord = next((c for c in piece.chords if c.bar == d), None)
        if chord is None:
            continue
        piece.drop_count += 1
        piece.stats[STAT_DROP] += 1
        if kind == "far":
            _play_far_drop(piece, d, chord)
        else:
            _play_loud_drop(piece, d, chord, kind)


def _play_far_drop(piece: _Piece, d: int, chord: _Chord) -> None:
    """Play the drop of the night: a reversed swell into a distant boom."""
    t_d = d * piece.bar
    _put(
        piece,
        t_d - piece.bar,
        piece.bar,
        float(chord.pad),
        FAR_REVSTAB_VELOCITY,
        VOICE_REVSTAB,
        LAYER_FX,
        chord.quality,
    )
    _put(piece, t_d, FAR_BOOM_HOLD, float(chord.root), BOOMFAR_VELOCITY, VOICE_BOOMFAR, LAYER_FX)


def _play_loud_drop(piece: _Piece, d: int, chord: _Chord, kind: str) -> None:
    """Play the stab and the boom of a drop, and the sequence into it when the drums play."""
    bar = piece.bar
    beat = piece.beat
    t_d = d * bar
    _put(
        piece,
        t_d - bar,
        bar,
        float(chord.pad),
        REVSTAB_VELOCITY,
        VOICE_REVSTAB,
        LAYER_FX,
        chord.quality,
    )
    _put(piece, t_d, STAB_BIG_HOLD, chord.pad, 1.0, "glitch:stab_big", LAYER_STAB, chord.quality)
    _put(piece, t_d, BOOM_HOLD, float(chord.root), BOOM_VELOCITY, "glitch:boom", LAYER_FX)
    if kind == "hit":
        _tape_short(piece, t_d, SOFT, WHY_DROP)
        return
    # a snare roll one bar ahead: one hit repeated, with a crescendo; the reversed snare ends on the drop
    t_roll = t_d - bar + 2 * beat
    u8, u16, u32 = piece.u8, piece.u16, piece.u32
    _put(piece, t_roll, DROP_SNARE_HOLD, 0.0, DROP_SNARE_VELOCITY, VOICE_SNARE, LAYER_SNARE)
    _roll_edit(piece, t_roll, (LAYER_SNARE,), [u8, u8, u16, u16], DROP_ROLL_GAINS, WHY_DROP)
    _put(
        piece,
        t_d - REVSNARE_LEAD,
        REVSNARE_LEAD,
        0.0,
        REVSNARE_DROP_VELOCITY,
        VOICE_REVSNARE,
        LAYER_FX,
    )
    _tape_short(piece, t_d, WHOLE, WHY_DROP)
    for event in taiko.roll(
        t_d - bar,
        t_d,
        "odaiko",
        DROP_ROLL_FROM,
        DROP_ROLL_TO,
        DROP_ROLL_FIRST_GAP,
        DROP_ROLL_LAST_GAP,
    ):
        _put(piece, event.time, DRUM_HOLD, DRUM_PITCH, event.velocity, "drum:odaiko", LAYER_FX)
    for beats, bars, vel in DROP_TAIKO_HITS:
        when = beats * beat + bars * bar
        _put(piece, t_d + when, DRUM_HOLD, DRUM_PITCH, vel, "drum:odaiko", LAYER_FX)
    # the closing accent and a beat repeat on the last beat of the big pattern
    _stutter_edit(
        piece, t_d + 2 * bar - beat, KIT, [u32] * 4 + [u16, u16], list(ROLL_GAINS), WHY_DROP
    )
    _put(piece, t_d + 2 * bar, KIT_HIT_HOLD, 0.0, 1.0, VOICE_KICK_ACCENT, LAYER_KICK)
    _put(piece, t_d + 2 * bar, KIT_HIT_HOLD, 0.0, 1.0, VOICE_SNARE, LAYER_SNARE)
    piece.kicks.append((t_d + 2 * bar, 1.0))


def _play_night_booms(piece: _Piece) -> None:
    """Play the distant booms of the night, every half minute or so, never on the grid of anything."""
    plan = piece.plan
    rng = piece.rng
    clock = float(rng.uniform(NIGHT_BOOM_FIRST_MIN, NIGHT_BOOM_FIRST_MAX))
    while clock < piece.length - NIGHT_BOOM_END_MARGIN:
        if (
            plan.at(plan.night, clock) > NIGHT_BOOM_NIGHT
            and plan.at(piece.day, clock) < NIGHT_BOOM_DAY
        ):
            where = _chord_at(piece.chords, clock, piece.bar)
            _put(
                piece,
                _snap(piece, clock),
                FAR_BOOM_HOLD,
                float(where.root),
                float(rng.uniform(NIGHT_BOOM_VELOCITY_MIN, NIGHT_BOOM_VELOCITY_MAX)),
                VOICE_BOOMFAR,
                LAYER_FX,
            )
            piece.far_booms += 1
            clock += float(rng.uniform(NIGHT_BOOM_GAP_MIN, NIGHT_BOOM_GAP_MAX))
        else:
            clock += NIGHT_BOOM_RETRY


def _edit_for_house(piece: _Piece) -> None:
    """Bring an edit for the switches, doors and appliances of the house, on the grid, a few bars apart."""
    plan = piece.plan
    bar = piece.bar
    length = piece.length
    last_end = NO_EDIT_YET
    big_times = [d * bar for d in piece.drops]
    for second, _strength, what in plan.triggers:
        if second < HOUSE_EDIT_FROM or second > length - HOUSE_EDIT_END_MARGIN:
            continue
        if any(t - bar <= second <= t + 3 * bar for t in big_times):
            continue
        h = float(plan.at(piece.heat, second))
        kit_on = _kit_level(piece, int(second / bar)) > 0
        asleep = plan.at(piece.day, second) < ASLEEP_DAY
        gap = bar * (
            KIT_GAP_BARS if kit_on else (DAY_GAP_BARS if h > CALM_HEAT else NIGHT_GAP_BARS)
        )
        if (
            second - last_end < gap
            or piece.edit_rng.random() > EDIT_CHANCE + EDIT_ENERGY * piece.energy + EDIT_HEAT * h
        ):
            continue
        choice = _house_choice(piece, second, what, kit_on=kit_on, asleep=asleep)
        if choice is None:
            continue
        placed = _house_edit(piece, choice, second, kit_on=kit_on)
        if placed.done:
            piece.event_edits += 1
            last_end = placed.end
            if asleep and what == TRIGGER_DOOR and piece.edit_rng.random() < ASLEEP_DOOR_CHANCE:
                where = _chord_at(piece.chords, second, bar)
                _put(
                    piece,
                    _snap(piece, second) + piece.beat,
                    FAR_BOOM_HOLD,
                    float(where.root),
                    HOUSE_EDIT_STRENGTH,
                    VOICE_BOOMFAR,
                    LAYER_FX,
                )
                piece.far_booms += 1


def _house_choice(
    piece: _Piece, second: float, what: str, *, kit_on: bool, asleep: bool
) -> str | None:
    """Return the kind of edit that an event of the house brings, or None when none is left to bring."""
    options = dict(EVENT_EDITS.get(what, DEFAULT_EDITS))
    weights = {k: w * piece.favour.get(_family(k), 1.0) for k, w in options.items()}
    if not kit_on:
        weights.pop(EDIT_CRUSH, None)
    if asleep:
        weights.pop(EDIT_BRAKE, None)
        weights.pop(EDIT_TAPE, None)
    if not weights:
        return None
    next_to = _kinds_next_to(piece.book, second)
    fresh = {k: w for k, w in weights.items() if not CHOICE_KINDS[k] & next_to}
    return weighted(piece.edit_rng, list(fresh or weights), list((fresh or weights).values()))


def _house_edit(piece: _Piece, choice: str, second: float, *, kit_on: bool) -> _Placed:
    """Place the edit of a kind that an event of the house brought, on the grid after the event."""
    t_beat = math.ceil(second / piece.beat) * piece.beat
    t_bar = math.ceil((second + EDIT_SNAP) / piece.bar) * piece.bar
    if choice == EDIT_BRAKE:
        return _house_brake(piece, t_bar, kit_on=kit_on)
    if choice == EDIT_TAPE:
        return _house_tape(piece, t_bar, kit_on=kit_on)
    if choice == EDIT_STUTTER:
        return _house_stutter(piece, t_beat, kit_on=kit_on)
    if choice == EDIT_GATE:
        return _house_gate(piece, t_beat)
    if choice == EDIT_FREEZE:
        return _house_freeze(piece, t_beat, second)
    return _house_crush(piece, second)


def _house_brake(piece: _Piece, t_bar: float, *, kit_on: bool) -> _Placed:
    """Brake into the bar line, sometimes twice."""
    step = piece.step
    edit_rng = piece.edit_rng
    buses = WHOLE if kit_on else SOFT
    stop = float(edit_rng.uniform(*BRAKE_STOP))
    if edit_rng.random() < HOUSE_BRAKE_CHANCE:
        first = _brake_edit(
            piece, t_bar - BRAKE_LEAD_STEPS * step, buses, stop, 2 * step, WHY_EVENT
        )
        done = (
            _brake_edit(
                piece,
                t_bar - 2 * step,
                buses,
                float(edit_rng.uniform(*BRAKE_STOP)),
                2 * step,
                WHY_EVENT,
            )
            or first
        )
        piece.stats[STAT_DOUBLE] += int(done)
    else:
        done = _brake_edit(piece, t_bar - 2 * step, buses, stop, 2 * step, WHY_EVENT)
    piece.stats[EDIT_BRAKE] += int(done)
    return _Placed(done, t_bar)


def _house_tape(piece: _Piece, t_bar: float, *, kit_on: bool) -> _Placed:
    """Stop the tape into the bar line, long or short."""
    buses = WHOLE if kit_on else SOFT
    if piece.edit_rng.random() < TAPE_LONG_SHARE * piece.mode.stretch:
        done = _tape_long(piece, t_bar, buses, WHY_EVENT)
    else:
        done = _tape_short(piece, t_bar, buses, WHY_EVENT)
    piece.stats[EDIT_TAPE] += int(done)
    return _Placed(done, t_bar)


def _house_stutter(piece: _Piece, t_beat: float, *, kit_on: bool) -> _Placed:
    """Roll the kit or stutter the soft buses on the beat."""
    u8, u16, u32 = piece.u8, piece.u16, piece.u32
    if kit_on and piece.edit_rng.random() < HOUSE_STUTTER_CHANCE:
        done = _roll_edit(piece, t_beat, KIT, [u16] * 2 + [u32] * 4, list(ROLL_GAINS), WHY_EVENT)
        piece.stats[EDIT_ROLL] += int(done)
    else:
        pick = int(piece.edit_rng.integers(0, len(HOUSE_STUTTERS)))
        eighths, sixteenths, thirty_seconds, house_gains = HOUSE_STUTTERS[pick]
        slices = [u8] * eighths + [u16] * sixteenths + [u32] * thirty_seconds
        gains = list(house_gains)
        done = _stutter_edit(piece, t_beat, SOFT, slices, gains, WHY_EVENT)
        piece.stats[EDIT_STUTTER] += int(done)
    return _Placed(done, t_beat + 2 * piece.beat)


def _house_gate(piece: _Piece, t_beat: float) -> _Placed:
    """Gate the pads, strings, washes and bass on the beat."""
    gate_pattern = GATE_PATTERNS[int(piece.edit_rng.integers(0, len(GATE_PATTERNS)))]
    done = _edit(
        piece,
        t_beat,
        EDIT_GATE,
        (LAYER_PAD, LAYER_STR, LAYER_WASH, LAYER_BASS),
        {"step": piece.u16, "pattern": gate_pattern, "floor": GATE_FLOOR},
        WHY_EVENT,
    )
    piece.stats[EDIT_GATE] += int(done)
    return _Placed(done, t_beat + len(gate_pattern) * piece.u16)


def _house_freeze(piece: _Piece, t_beat: float, second: float) -> _Placed:
    """Freeze the soft buses on the beat."""
    done = _freeze_edit(piece, t_beat, int(second), WHY_EVENT)
    piece.stats[EDIT_FREEZE] += int(done)
    return _Placed(done, t_beat + FREEZE_BEATS * piece.beat * piece.mode.stretch)


def _house_crush(piece: _Piece, second: float) -> _Placed:
    """Crush the first snare hit within a bar."""
    done = False
    hits = [t for t in piece.kit_hits[LAYER_SNARE] if second <= t < second + piece.bar]
    if hits:
        done = _edit(
            piece,
            hits[0],
            EDIT_CRUSH,
            (LAYER_SNARE,),
            {"dur": CRUSH_DUR, "factor": CRUSH_FACTOR, "bits": CRUSH_BITS},
            WHY_EVENT,
        )
    piece.stats[EDIT_CRUSH] += int(done)
    return _Placed(done, second + piece.bar)


def _edit_grooves(piece: _Piece) -> None:
    """
    Give most of the bars where the kit plays an edit of their own, on the grid.

    The activity of the house moves the chance, and an event of the house in a bar brings it and picks the kind.
    """
    plan = piece.plan
    bar = piece.bar
    piece.groove_rng = plan.rng("glitch groove")
    house_bars: dict[int, str] = {}
    extreme_bars = _extreme_bars(piece.sections, piece.per_bar, bar, piece.energy)
    for second, _strength, what in plan.triggers:
        house_bars.setdefault(int(second / bar), what)
    for b in range(2, piece.bars - 1):
        if (
            _kit_level(piece, b) == 0
            or b + 1 in piece.section_starts
            or any(d - GROOVE_BEFORE_DROP <= b < d + GROOVE_AFTER_DROP for d in piece.drops)
        ):
            continue
        extreme = b in extreme_bars
        chance = (
            EXTREME_CHANCE
            if extreme
            else GROOVE_CHANCE + GROOVE_ENERGY * piece.energy + GROOVE_HEAT * piece.per_bar[b]
        )
        visit = house_bars.get(b)
        if visit is not None:
            chance = min(chance * GROOVE_HOUSE_BOOST, 1.0)
        if piece.groove_rng.random() > chance:
            continue
        ordered = sorted(piece.book.edits, key=lambda e: e.t)
        middle = b * bar + bar / 2.0
        groove = _GrooveBar(
            b,
            extreme,
            visit,
            [e.kind for e in ordered if e.t <= middle],
            _kinds_next_to(piece.book, middle),
        )
        _groove_edit(piece, groove)


def _groove_edit(piece: _Piece, groove: _GrooveBar) -> None:
    """Try a few times to place one edit in a bar of the groove."""
    rng = piece.groove_rng
    bus_weights = EXTREME_BUSES if groove.extreme else GROOVE_BUSES
    for _ in range(GROOVE_TRIES):
        category = weighted(rng, [c for c, _ in bus_weights], [w for _, w in bus_weights])
        weights = _groove_weights(piece, groove, category)
        if not weights:
            break
        kind = weighted(rng, list(weights), list(weights.values()))
        params, starts = GROOVE_SHAPES[kind](rng, piece.beat)
        if groove.extreme and kind == EDIT_GATE:
            params = {**params, "floor": EXTREME_GATE_FLOOR}
        t = groove.number * piece.bar + int(rng.choice(starts)) * piece.step
        cut_buses = _groove_buses(piece, category, kind, t, glitch.edit_extent(kind, params))
        if cut_buses is None:
            continue
        if _edit(
            piece, t, kind, cut_buses, params, WHY_GROOVE if groove.visit is None else WHY_HOUSE
        ):
            piece.stats[kind] += 1
            piece.groove_edits += 1
            break


def _groove_weights(piece: _Piece, groove: _GrooveBar, category: str) -> dict[str, float]:
    """Return how much each kind of edit is wanted in a bar of the groove."""
    weights: dict[str, float] = {}
    for kind, base in GROOVE_KINDS.items():
        if (kind == EDIT_BRAKE and category != "whole") or kind in groove.next_to:
            continue
        weight = base * piece.favour.get(_family(kind), 1.0)
        if groove.visit is not None:
            house = GROOVE_HOUSE.get(groove.visit, GROOVE_HOUSE_OTHER)
            weight *= house.get(kind, GROOVE_HOUSE_UNNAMED)
        if kind in GROOVE_RAMPS and groove.number % 4 == 3:
            weight *= GROOVE_RAMP_BOOST
        weights[kind] = weight * GROOVE_REPEAT_PENALTY ** groove.before[-GROOVE_RECENT:].count(kind)
    return weights


def _groove_buses(
    piece: _Piece, category: str, kind: str, t: float, reach: float
) -> tuple[str, ...] | None:
    """Return the buses that a groove edit cuts, or None when an instrument of its own is silent."""
    rng = piece.groove_rng
    if category == "instrument":
        open_buses = [(k, w) for k, w in INSTRUMENT_BUSES if _alive(piece, (k,), t, t + reach)]
        if not open_buses:
            return None
        return (weighted(rng, [k for k, _ in open_buses], [w for _, w in open_buses]),)
    if category == "kit":
        return weighted(rng, [k for k, _ in KIT_BUSES], [w for _, w in KIT_BUSES])
    return (LAYER_PAD, LAYER_STR, LAYER_WASH, LAYER_BASS) if kind == EDIT_GATE else WHOLE


def _thin_repeats(piece: _Piece) -> None:
    """Drop the edits that would play a kind a third time in a row; the edits of the structure stay."""
    kept: list[Edit] = []
    for cut in sorted(piece.book.edits, key=lambda e: e.t):
        if all(e.kind == cut.kind for e in kept[-MOST_IN_A_ROW:]) and len(kept) >= MOST_IN_A_ROW:
            if cut.why in GIVES_WAY:
                continue
            if kept[-1].why in GIVES_WAY:
                kept.pop()
        kept.append(cut)
    piece.book.edits = kept


def _add_layers(piece: _Piece) -> None:
    """Make the layers, trim them to their voices and put the curves of the lead on them."""
    plan = piece.plan
    count_curve = np.full(piece.count, STR_CUT_BASE)
    str_cut = count_curve * _dips(plan)
    layers = {
        LAYER_PAD: Layer(LAYER_PAD, 1.0, None, None, 0.0, reverb=PAD_REVERB, title="swells"),
        LAYER_STR: Layer(
            LAYER_STR,
            STR_GAIN,
            None,
            str_cut,
            0.0,
            reverb=STR_REVERB,
            title="cello and violin sections",
        ),
        LAYER_WASH: Layer(
            LAYER_WASH, WASH_GAIN, None, str_cut, 0.0, reverb=WASH_REVERB, title="stretched washes"
        ),
        LAYER_STAB: Layer(
            LAYER_STAB, 1.0, None, None, 0.0, reverb=STAB_REVERB, title="brass-like stabs"
        ),
        LAYER_PIANO: Layer(
            LAYER_PIANO,
            PIANO_GAIN,
            None,
            np.full(piece.count, PIANO_CUTOFF_HZ) * _dips(plan),
            0.0,
            reverb=PIANO_ROOM_SEND,
            hall=PIANO_HALL_SEND,
            title="trailer piano",
        ),
        LAYER_BASS: Layer(LAYER_BASS, 1.0, None, None, 0.0, reverb=0.0, title="growl bass"),
        LAYER_KICK: Layer(LAYER_KICK, KICK_GAIN, None, None, 0.0, reverb=0.0, title="kick"),
        LAYER_SNARE: Layer(
            LAYER_SNARE, SNARE_GAIN, None, None, 0.0, reverb=SNARE_REVERB, title=LAYER_SNARE
        ),
        LAYER_HATS: Layer(
            LAYER_HATS, HATS_GAIN, None, None, 0.0, reverb=0.0, title="hats, ticks and rim"
        ),
        LAYER_FX: Layer(
            LAYER_FX,
            1.0,
            None,
            None,
            0.0,
            reverb=FX_REVERB,
            title="booms, reversed swells and taiko",
        ),
    }
    trim_layer(layers[LAYER_STR], piece.str_sig)
    trim_layer(layers[LAYER_BASS], piece.bass_sig)
    trim_layer(layers[LAYER_PIANO], piece.piano_variant.sig)
    lead.apply_curves(
        layers, lead.lead_curves(plan, piece.leads, LEADS), lead.LEAD_STEP_DB, LEAD_TRIM_DB
    )
    piece.layers = layers


def _assemble(piece: _Piece) -> Score:
    """Put the notes, the layers and the edits together as the score, with the notes about how it was made."""
    plan = piece.plan
    length = piece.length
    space = np.clip((SPACE_BASE + SPACE_NIGHT * plan.night) / SPACE_DIVISOR, 0.0, 1.0)
    if piece.light:
        space = space * SPACE_DROP
    gain_db = np.full(length * CURVE_RATE + 1, MASTER_GAIN_DB)
    piece.kicks.sort()
    setup = Glitch(
        piece.bpm,
        _with_piano(sorted(piece.book.edits, key=lambda e: e.t)),
        piece.kicks,
        dict(DUCK),
        (LAYER_KICK, LAYER_SNARE, LAYER_HATS),
        piece.vowels,
    )
    score = Score(
        length,
        sorted(
            [n for n in piece.notes if n.t < length], key=lambda n: (n.t, n.layer, n.voice, n.midi)
        ),
        piece.layers,
        [],
        space,
        tempo=piece.bpm,
        events=list(plan.events),
        level_db=LEVEL_DB,
        gain_db=gain_db,
        glitch=setup,
    )
    plan.choices[lead.LEAD_HANDOVERS_CHOICE] = str(len(piece.leads.handed))
    score.leads = piece.leads
    score.choices = dict(plan.choices)
    score.notes_about = _describe(piece)
    return score


def _describe(piece: _Piece) -> list[str]:
    """Return the notes about how the piece was made."""
    mode = piece.mode
    bpm = piece.bpm
    kinds = ", ".join(f"{v} {k}" for k, v in piece.stats.items() if v)
    return [
        (
            f"{mode.name} mode, {bpm:.0f} beats per minute, the kit counting {2 * bpm:.0f}; root {tonic_name_of(piece.tonic)}, {piece.mode_name}; "
            f"chords '{piece.loop_name}' every two bars (four when the day is quiet or the house sleeps); the drop chord is a {'semitone above' if piece.phrygian else 'tone below'} the root."
        ),
        (
            f"Growl '{piece.growl_name}' ({piece.bass_notes} notes, bass phrases '{piece.bass_name}'), drum patterns '{piece.family.name}' ({piece.drum_events} hits, {piece.fills} fills), "
            f"{piece.stab_count} stabs, washes '{piece.wash_name}'; the heat of the house picks the row of the growl, the pattern of the kit and how often it is glitched."
        ),
        (
            f"Edits (favoured: {piece.favour_name}): {kinds or 'none'}; {piece.event_edits} from the events of the house, {piece.groove_edits} from the groove of the bars, {piece.phrase_edits} from the structure; "
            f"{piece.drop_count} drops (the big moments), {piece.far_booms} distant booms in the night."
        ),
        lead.summary(piece.default_lead, piece.leads),
    ]


def _heat_in(piece: _Piece, a: float, b: float, curve: np.ndarray | None = None) -> float:
    """Return the mean of a curve (the heat by default) between two seconds."""
    if curve is None:
        curve = piece.heat
    i, j = int(a * CURVE_RATE), max(int(b * CURVE_RATE), int(a * CURVE_RATE) + 1)
    return float(curve[min(i, piece.count - 1) : min(j, piece.count)].mean())


def _drums_ok(piece: _Piece, second: float) -> bool:
    """Return True when the kit may play at a second: in the hours of the drums, with the house awake."""
    return piece.first_drum <= second < piece.last_drum and piece.plan.at(piece.day, second) > 0.5


def _drums_or_hit(piece: _Piece, d: int) -> str:
    """Return the kind of a drop on bar d: drums when the kit plays and the bar is hot enough, else a hit."""
    hot = _drums_ok(piece, d * piece.bar) and piece.per_bar[d] > piece.limits["kick"]
    return "drums" if hot else "hit"


def _snap(piece: _Piece, second: float) -> float:
    """Return a second moved to the grid of the edits."""
    return round(second / piece.step) * piece.step


def _put(
    piece: _Piece,
    t: float,
    dur: float,
    midi: float,
    vel: float,
    voice: str,
    layer: str,
    patch: int = 0,
) -> None:
    """Add a note, unless it starts in the last second, and remember the stretch that it sounds."""
    if t < piece.length - 1.0:
        piece.notes.append(Note(t, dur, midi, vel, voice, layer, patch))
        piece.spans.setdefault(layer, []).append((t, t + min(dur, 3.0)))


def _alive(piece: _Piece, layers: tuple[str, ...], a: float, b: float) -> tuple[str, ...]:
    """Return the layers that sound somewhere between two seconds."""
    return tuple(key for key in layers if any(s < b and e > a for s, e in piece.spans.get(key, [])))


def _edit(
    piece: _Piece,
    t: float,
    kind: str,
    layers: tuple[str, ...],
    params: dict[str, Any],
    why: str,
) -> bool:
    """Add an edit on the layers that sound under it; return whether it was added."""
    layers = _alive(piece, layers, t, t + glitch.edit_extent(kind, params))
    if not layers or t < EDIT_START or t > piece.length - EDIT_END_MARGIN:
        return False
    return piece.book.add(
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


def _stutter_edit(
    piece: _Piece,
    t: float,
    layers: tuple[str, ...],
    slices: list[float],
    gains: list[float],
    why: str,
) -> bool:
    return _edit(
        piece, t, EDIT_STUTTER, layers, {"slices": tuple(slices), "gains": tuple(gains)}, why
    )


def _roll_edit(
    piece: _Piece,
    t: float,
    layers: tuple[str, ...],
    slices: list[float],
    gains: list[float],
    why: str,
) -> bool:
    return _edit(piece, t, EDIT_ROLL, layers, {"slices": tuple(slices), "gains": tuple(gains)}, why)


def _brake_edit(
    piece: _Piece, t: float, layers: tuple[str, ...], stop: float, span: float, why: str
) -> bool:
    return _edit(piece, t, EDIT_BRAKE, layers, {"stop": round(stop, 3), "span": span}, why)


def _tape_long(piece: _Piece, t: float, layers: tuple[str, ...], why: str) -> bool:
    """Add a tape stop that winds down over a beat and a half, ending at a second."""
    span = TAPE_LONG_BEATS * piece.beat
    params = {"stop": TAPE_LONG_STOP, "span": span, "wind": TAPE_LONG_WIND}
    return _edit(piece, t - span, EDIT_TAPE, layers, params, why)


def _tape_short(piece: _Piece, t: float, layers: tuple[str, ...], why: str) -> bool:
    """Add a tape stop that winds down over half a beat, ending at a second."""
    span = TAPE_SHORT_BEATS * piece.beat
    params = {"stop": TAPE_SHORT_STOP, "span": span, "wind": 0.0}
    return _edit(piece, t - span, EDIT_TAPE, layers, params, why)


def _freeze_edit(piece: _Piece, t: float, seed: int, why: str) -> bool:
    """Add a grain or a spectral freeze, by the share of grains of the mode."""
    mode = piece.mode
    beat = piece.beat
    if piece.light or piece.edit_rng.random() < mode.grain_share:
        return _edit(
            piece,
            t,
            EDIT_GRAIN,
            (LAYER_STAB, LAYER_PAD, LAYER_STR),
            {
                "length": beat * FREEZE_LENGTH_BEATS * mode.stretch,
                "grain": FREEZE_GRAIN,
                "seed": seed,
                "jitter": FREEZE_JITTER,
            },
            why,
        )
    return _edit(
        piece,
        t,
        EDIT_SPECTRAL,
        (LAYER_PAD, LAYER_STR, LAYER_WASH),
        {"length": beat * REVERSE_LENGTH_BEATS * mode.stretch, "back": REVERSE_BACK, "seed": seed},
        why,
    )


def _roll_shape(rng: np.random.Generator, beat: float) -> _Shape:
    u8, u16, u32 = beat / EIGHTH, beat / SIXTEENTH, beat / THIRTY_SECOND
    if rng.random() < ROLL_LONG_SHARE:
        return _Shape(
            {"slices": (u8, u16, u16, *[u32] * 4), "gains": ROLL_LONG_GAINS},
            (BAR_STEPS - ROLL_LONG_BACK,),
        )
    return _Shape(
        {"slices": (u16, u16, *[u32] * 4), "gains": ROLL_SHORT_GAINS},
        (BAR_STEPS - ROLL_SHORT_BACK,),
    )


def _reverse_shape(rng: np.random.Generator, beat: float) -> _Shape:
    steps = int(rng.choice(REVERSE_STEPS))
    last = BAR_STEPS - steps
    return _Shape(
        {"length": steps * beat / THIRTY_SECOND}, tuple(last - back for back in REVERSE_BACKS)
    )


def _pitch_shape(rng: np.random.Generator, beat: float) -> _Shape:
    rate = float(PITCH_RATES[int(rng.integers(0, len(PITCH_RATES)))])
    if rng.random() < PITCH_SIXTEENTH_SHARE:
        slice_s, repeats = beat / SIXTEENTH, int(rng.integers(*PITCH_REPEATS))
    else:
        slice_s, repeats = beat / EIGHTH, 2
    gains = tuple(1.0 - PITCH_FADE * i for i in range(repeats))
    params = {"slice": slice_s, "repeats": repeats, "rate": rate, "gains": gains}
    return _Shape(params, PITCH_STARTS)


def _bitdrop_shape(rng: np.random.Generator, beat: float) -> _Shape:
    params = {
        "dur": beat / EIGHTH if rng.random() < 0.5 else beat,
        "factor": int(rng.integers(BIT_DROP_FACTORS[0], BIT_DROP_FACTORS[1] + 1)),
        "bits": int(rng.integers(BIT_DROP_BITS[0], BIT_DROP_BITS[1] + 1)),
    }
    return _Shape(params, BITDROP_STARTS)


def _gate_shape(rng: np.random.Generator, beat: float) -> _Shape:
    pattern = GROOVE_GATES[int(rng.integers(0, len(GROOVE_GATES)))]
    return _Shape({"step": beat / EIGHTH, "pattern": pattern, "floor": GATE_FLOOR}, GATE_STARTS)


def _brake_shape(rng: np.random.Generator, beat: float) -> _Shape:
    stop = round(float(rng.uniform(*BRAKE_STOP)), 3)
    return _Shape({"stop": stop, "span": beat / SIXTEENTH}, (BAR_STEPS - 2,))


def _stutter_shape(rng: np.random.Generator, beat: float) -> _Shape:
    if rng.random() < STUTTER_SHORT_SHARE:
        return _Shape(
            {"slices": (beat / EIGHTH,) * 4, "gains": STUTTER_SHORT_GAINS}, STUTTER_STARTS
        )
    gains = tuple(1.0 - STUTTER_FADE * i for i in range(STUTTER_LONG_COUNT))
    return _Shape(
        {"slices": (beat / SIXTEENTH,) * STUTTER_LONG_COUNT, "gains": gains}, STUTTER_STARTS
    )


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
        EDIT_CRUSH: EDIT_STUTTER,
        EDIT_ROLL: EDIT_STUTTER,
        EDIT_BITDROP: EDIT_STUTTER,
        EDIT_PITCHREP: EDIT_FREEZE,
        EDIT_REVERSE: EDIT_TAPE,
    }.get(kind, kind)


def _dips(plan: Plan) -> np.ndarray:
    """Return the dip of the cutoff at the anomalies (a filter that closes and opens, never a pitch)."""
    return dips(plan, DIP_DEPTH, DIP_HOLD_SECONDS, DIP_RETURN_SECONDS)


def _place_half_bar(
    piece: _Piece,
    pattern: Pattern,
    t0: float,
    until: int | None = None,
    kick_floor: float = 0.0,
) -> int:
    """
    Place a pattern of the kit from second t0; return how many hits.

    :param piece: The piece being made.
    :param pattern: The pattern: the steps of each layer of the kit and their velocities.
    :param t0: The second of the first step.
    :param until: Place only the steps before this one (the rest of the pattern gives way to a fill).
    :param kick_floor: Leave out the kicks softer than this.
    """
    count = 0
    limit = until if until is not None else len(pattern[LAYER_HATS])

    def at(s: int) -> float:
        return t0 + s * piece.step

    for s, v in pattern[LAYER_KICK]:
        if s < limit and v >= kick_floor:
            _put(piece, at(s), KIT_HIT_HOLD, 0.0, v, VOICE_KICK, LAYER_KICK)
            piece.kicks.append((at(s), min(1.0, 0.5 + 0.5 * v)))
            piece.kit_hits[LAYER_KICK].append(at(s))
            count += 1
    for s, v in pattern[LAYER_SNARE]:
        if s < limit:
            _put(piece, at(s), KIT_HIT_HOLD, 0.0, v, VOICE_SNARE, LAYER_SNARE)
            piece.kit_hits[LAYER_SNARE].append(at(s))
            count += 1
    for s, v in enumerate(pattern[LAYER_HATS]):
        if v >= (LIGHT_HAT_FLOOR if piece.light else 0.0) and v and s < limit:
            _put(piece, at(s), HAT_HOLD, 0.0, v, "glitch:hat", LAYER_HATS)
            count += 1
    if piece.light:
        return count
    for name, voice, layer, dur, midi in KIT_LANES:
        for s, v in pattern[name]:
            if s < limit:
                _put(piece, at(s), dur, midi, v, voice, layer)
                count += 1
    return count


def _lead_strings(chord: _Chord, previous: list[float], count: int) -> list[float]:
    """
    Return the upper voices of the strings of a chord, placed to move the least from the voices before.

    :param chord: The chord.
    :param previous: The upper voices of the chord before (empty for the first).
    :param count: How many upper voices: the third, then the fifth and the root.
    """
    third = MINOR_THIRD if chord.quality == 0 else MAJOR_THIRD
    tones = (third, FIFTH_SEMITONES, 0)[:count]
    first = [
        float(chord.pad + o)
        for o in (third + SEMITONES_PER_OCTAVE, FIFTH_ABOVE_OCTAVE, TWO_OCTAVES)[:count]
    ]
    if not previous:
        return first
    choices = [
        [
            chord.pad + tone + SEMITONES_PER_OCTAVE * k
            for k in range(3)
            if STRINGS_LOW[-1] < tone + SEMITONES_PER_OCTAVE * k <= STRINGS_TOP + STRINGS_HEADROOM
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


def _piano_variant(plan: Plan) -> _PianoVariant:
    """Return the way the room plays its piano lead, the same on every day, and note it in the choices."""
    variant = PIANO_VARIANTS[
        stable_hash("piano lead", plan.area_key, plan.style) % len(PIANO_VARIANTS)
    ]
    plan.choices[PIANO_CHOICE] = variant.label
    return variant


def _piano_phrase(
    plan: Plan, variant: _PianoVariant, chords: list[_Chord], number: int, beat: float
) -> list[Note]:
    """
    Return the notes of the trailer piano in a phrase: statements of the motif as intervals that fit in it.

    :param plan: The plan of the piece.
    :param variant: How the room plays the piano.
    :param chords: The chords of the piece, for the lower notes of the open intervals.
    :param number: The number of the phrase.
    :param beat: The length of a beat in seconds.
    """
    bar = BEAT_COUNT * beat
    phrase = plan.form_grid(bar)
    scale = plan.scale
    scale_notes = sorted(
        scale.midi(d, o) for o in range(1, PIANO_SCALE_OCTAVES_TO) for d in range(scale.size)
    )
    out: list[Note] = []
    clock = number * phrase + PIANO_FIRST_BAR * bar
    # a statement takes about this many times its beats, so a short phrase gets short variations
    longest = (phrase / beat - PIANO_FIRST_BAR * BEAT_COUNT) / (PIANO_STRETCH + 1.0)
    for k in range(PIANO_STATEMENTS):
        motif = develop_motif(
            plan.motif(), PIANO_STATEMENTS * number + k, motif_seed(plan), longest
        )
        line = motif_notes(plan, motif)
        last_onset = clock + sum(_piano_stride(beats, beat) for _, beats in line[:-1])
        if last_onset >= (number + 1) * phrase:
            if out:
                break
            # a phrase the piano leads is never silent: a first statement too long for it is cut short
            line = _fitting(line, clock, (number + 1) * phrase, beat)
        for index, (midi, beats) in enumerate(line):
            upper = _fold(midi, PIANO_LOWEST, PIANO_HIGHEST)
            lower = (
                _open_below(upper, scale_notes, _chord_at(chords, clock, bar))
                if variant.open_intervals
                else upper - SEMITONES_PER_OCTAVE
            )
            ring = beats * beat * PIANO_STRETCH + PIANO_RING
            sig = variant.sig
            out += [
                Note(
                    clock,
                    ring,
                    float(note),
                    PIANO_VELOCITY,
                    sig.voice,
                    LAYER_PIANO,
                    patch_of(sig, 0),
                )
                for note in (upper, lower)
            ]
            if index == 0:
                out.append(
                    Note(
                        clock,
                        ring + PIANO_RING,
                        float(lower - SEMITONES_PER_OCTAVE),
                        PIANO_VELOCITY * PIANO_DEPTH_SHARE,
                        sig.voice,
                        LAYER_PIANO,
                        patch_of(sig, 0),
                    )
                )
            clock += _piano_stride(beats, beat)
        clock += PIANO_BREATH_BEATS * beat
    return out


def _fitting(
    line: list[tuple[int, float]], start: float, end: float, beat: float
) -> list[tuple[int, float]]:
    """Return the notes of a statement of the piano that start before the end of its phrase."""
    kept = []
    clock = start
    for midi, beats in line:
        if clock >= end:
            break
        kept.append((midi, beats))
        clock += _piano_stride(beats, beat)
    return kept


def _piano_stride(beats: float, beat: float) -> float:
    """Return how far the piano moves on after a note of so many beats: the note stretched, and a beat to breathe."""
    return beats * beat * PIANO_STRETCH + beat


def _fold(midi: int, lowest: int, highest: int) -> int:
    """Return the note moved by octaves into a range."""
    while midi < lowest:
        midi += SEMITONES_PER_OCTAVE
    while midi > highest:
        midi -= SEMITONES_PER_OCTAVE
    return midi


def _chord_at(chords: list[_Chord], second: float, bar: float) -> _Chord:
    return next((c for c in reversed(chords) if c.bar * bar <= second), chords[0])


def _open_below(upper: int, scale_notes: list[int], chord: _Chord) -> int:
    """Return the note a fifth or a sixth under a melody note: the one that is a tone of the chord, else the nearer."""
    tones = {
        (chord.root + step) % SEMITONES_PER_OCTAVE
        for step in (0, MINOR_THIRD if chord.quality == 0 else MAJOR_THIRD, FIFTH_SEMITONES)
    }
    near = min(range(len(scale_notes)), key=lambda i: abs(scale_notes[i] - upper))
    fifth, sixth = scale_notes[near - PIANO_FIFTH_STEPS], scale_notes[near - PIANO_SIXTH_STEPS]
    if fifth % SEMITONES_PER_OCTAVE in tones:
        return fifth
    if sixth % SEMITONES_PER_OCTAVE in tones:
        return sixth

    def distance(note: int) -> int:
        return min(
            min((note - tone) % SEMITONES_PER_OCTAVE, (tone - note) % SEMITONES_PER_OCTAVE)
            for tone in tones
        )

    return fifth if distance(fifth) <= distance(sixth) else sixth


def _with_piano(edits: list[Edit]) -> list[Edit]:
    """Return the edits with the piano on every one that touches the strings or the stabs, where the piano is free."""
    free_from = 0.0
    out = []
    for cut in edits:
        with_piano = {LAYER_STR, LAYER_STAB} & set(cut.layers) and cut.t - EDIT_SLACK >= free_from
        out.append(replace(cut, layers=(*cut.layers, LAYER_PIANO)) if with_piano else cut)
        if with_piano:
            free_from = cut.t + cut.dur
    return out


# the lanes of the kit beyond the three drums: (pattern, voice, layer, hold in seconds, pitch)
KIT_LANES = (
    ("ghost", "glitch:ghost", LAYER_SNARE, 0.3, 0.0),
    ("open", "glitch:open", LAYER_HATS, 0.5, 0.0),
    ("ticks", "glitch:tick", LAYER_HATS, 0.2, 0.0),
    ("rim", "glitch:rim", LAYER_HATS, 0.2, 0.0),
    ("toms", "drum:odaiko", LAYER_FX, 0.2, TOM_PITCH),
)
PIANO_VARIANTS = (
    _PianoVariant(
        SigVoice("felt_piano", None, FELT_OPEN_TRIM_DB),
        open_intervals=True,
        label="felt piano, open fifths and sixths",
    ),
    _PianoVariant(
        SigVoice("grand_piano", None, GRAND_OCTAVES_TRIM_DB),
        open_intervals=False,
        label="grand piano, octaves",
    ),
    _PianoVariant(
        SigVoice("grand_piano", None, GRAND_OPEN_TRIM_DB),
        open_intervals=True,
        label="grand piano, open fifths and sixths",
    ),
)


# the parameters of each kind of groove edit, and the steps of a bar (an eighth of a beat) that it may start on; the ramps
# (a roll, a reversal, a brake) end on the bar line, the rest start on a beat or a half beat
GROOVE_SHAPES: dict[str, Callable[[np.random.Generator, float], _Shape]] = {
    EDIT_STUTTER: _stutter_shape,
    EDIT_ROLL: _roll_shape,
    EDIT_REVERSE: _reverse_shape,
    EDIT_PITCHREP: _pitch_shape,
    EDIT_BITDROP: _bitdrop_shape,
    EDIT_GATE: _gate_shape,
    EDIT_BRAKE: _brake_shape,
}
