"""
Lo-fi: a dusty swung boom-bap loop, felt piano or organ on jazzy progressions, a round bass, vinyl and tape.

The generative system is a band that plays a composed loop and never plays it the same way twice. The day picks
a progression of four chords with sevenths, ninths and elevenths that agrees with the mode of the day (ii V I and
backdoor moves on the major, lydian and mixolydian days, i IV bVII and i VI iv V on the dorian and aeolian ones), a
comping rhythm, a bass pattern and a drum pattern; every chord is voiced a little differently each time and led from
the one before (common tones held, the other voices by step), and the band varies the loop by rule: a chord is
sometimes swapped for a related one, the comping rhythm drifts to a neighbour, ghost notes and fills come and go,
and a chord chop or a plucked line is played now and then. At night the drums rest and the loop becomes an ambient
piece of keys and vinyl.

How the house steers it: the energy of the day picks the tempo inside 70 to 88, how long a chord lasts (four to eight
bars on a quiet day, one or two on a busy one), the swing (a quiet day lies back, a busy one plays straighter), the
comping, the bass pattern and the drum pattern (a soft half time kit when quiet, a rolling syncopated kit when busy);
the warmth picks the rounder or the glassier strings. The keys, the bass and the plucked line are the room's own
and stay the same every day (`signature`). The shape of the day is the form: intro,
build, peak, release and outro enter and leave on phrases of eight bars, a build ends in a fill, and the motif of the
room is played on the plucked line in the first and the last peak or plateau. The strongest event of the day turns the next phrase into
a breakdown (the kit and the bass leave and come back with a fill) or brings the strings in. Who leads is the room's too: the keys, the plucked line or the strings (picked by
the room, the same every day). The plucked line leads with phrases of the motif and its developments that answer the
chords, the strings with a slow legato top line over a softer bed, and the others step back; a door or motion hands the
next phrase of a daytime build, peak or plateau to the pluck, an appliance or the hob to the keys, a light, a cover
or the media to the strings, at most one phrase in three, and the lead then returns to the
room's own. On top of that the
activity sets the density of the drums and the comping, the power of appliances the weight of the bass, the
temperature and the sun the tape low-pass, humidity the room, the lights and the evening the extensions of the
chords, a running dishwasher brings in the drum loop, and an anomaly closes the tape low-pass and opens it again (a
filter dip, never a pitch glide).

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import itertools
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from . import lead
from .character import chord_bars, tempo_target
from .gen import block_walk, dips, swells, swung
from .harmony import (
    MOTIF_CHOICE,
    SIGNED_TAG_SPAN,
    STEPS_PER_BAR,
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
    PLATEAU,
    RELEASE,
    RUN_DISH,
    SHORT_FORM_BARS,
    TEMPO_TOLERANCE,
)
from .score import (
    LAYER_BASS,
    LAYER_KEYS,
    LAYER_PLUCK,
    LAYER_STRINGS,
    Layer,
    Note,
    Score,
    Texture,
)
from .signature import SigVoice, patch_of, signature, trim_layer
from .styles import LOFI
from .synth import SEMITONES_PER_OCTAVE
from .timeline import SECONDS_PER_MINUTE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .plan import Plan

# the sound of a chord above its root, in semitones, as the keys play it
QUALITIES: dict[str, tuple[int, ...]] = {
    "m7": (3, 7, 10),
    "m9": (3, 7, 10, 14),
    "m11": (3, 10, 14, 17),
    "maj9": (4, 7, 11, 14),
    "dom9": (4, 10, 14),
    "dom13": (4, 10, 14, 21),
}
# the extended version of a quality, which the lights and the evening bring in
EXTENDED = {
    "m7": "m9",
    "m9": "m11",
    "maj9": "maj9",
    "dom9": "dom13",
    "m11": "m11",
    "dom13": "dom13",
}
Chord = tuple[int, str]
DOMINANT = 7


class Progression(NamedTuple):
    """Four chords, the modes whose notes they stay in, and how much they move (0 a vamp, 1 a chain of cadences)."""

    name: str
    chords: tuple[Chord, ...]
    modes: frozenset[str]
    motion: float


MAJOR = frozenset({"major"})
MIXOLYDIAN = frozenset({"mixolydian"})
LYDIAN = frozenset({"lydian"})
DORIAN = frozenset({"dorian"})
AEOLIAN = frozenset({"aeolian"})
# fmt: off
PROGRESSIONS: tuple[Progression, ...] = (
    Progression("ii V I vi", ((2, "m9"), (7, "dom9"), (0, "maj9"), (9, "m9")), MAJOR, 0.9),
    Progression("I vi ii V", ((0, "maj9"), (9, "m9"), (2, "m9"), (7, "dom9")), MAJOR, 0.8),
    Progression("backdoor iv bVII I vi", ((5, "m9"), (10, "dom9"), (0, "maj9"), (9, "m9")), MAJOR, 0.7),
    Progression("IV vi ii I", ((5, "maj9"), (9, "m9"), (2, "m9"), (0, "maj9")), MAJOR, 0.55),
    Progression("vi IV I V", ((9, "m9"), (5, "maj9"), (0, "maj9"), (7, "dom13")), MAJOR, 0.4),
    Progression("ii bVII I IV", ((2, "m9"), (10, "maj9"), (0, "dom9"), (5, "maj9")), MIXOLYDIAN, 0.8),
    Progression("I v IV bVII", ((0, "dom9"), (7, "m9"), (5, "maj9"), (10, "maj9")), MIXOLYDIAN, 0.5),
    Progression("I bVII IV I", ((0, "dom9"), (10, "maj9"), (5, "maj9"), (0, "dom13")), MIXOLYDIAN, 0.25),
    Progression("vi II V I", ((9, "m9"), (2, "dom9"), (7, "maj9"), (0, "maj9")), LYDIAN, 0.8),
    Progression("I V II vi", ((0, "maj9"), (7, "maj9"), (2, "dom9"), (9, "m9")), LYDIAN, 0.55),
    Progression("I II iii vi", ((0, "maj9"), (2, "dom9"), (4, "m9"), (9, "m9")), LYDIAN, 0.3),
    Progression("i IV v bVII", ((0, "m9"), (5, "dom9"), (7, "m9"), (10, "maj9")), DORIAN, 0.7),
    Progression("i bIII IV bVII", ((0, "m11"), (3, "maj9"), (5, "dom9"), (10, "maj9")), DORIAN, 0.5),
    Progression("i IV bVII i", ((0, "m9"), (5, "dom9"), (10, "maj9"), (0, "m11")), DORIAN, 0.25),
    Progression("i VI iv V", ((0, "m9"), (8, "maj9"), (5, "m11"), (7, "dom13")), AEOLIAN, 0.8),
    Progression("i VII VI V", ((0, "m11"), (10, "dom9"), (8, "maj9"), (7, "dom13")), AEOLIAN, 0.6),
    Progression("i iv VII III", ((0, "m9"), (5, "m11"), (10, "dom9"), (3, "maj9")), AEOLIAN, 0.35),
)
# fmt: on
# a progression of the mode of the day counts this much, how well its motion fits the energy the rest
MODE_WEIGHT = 0.7
# substitutes for a chord: (semitones from its root, quality); a dominant is swapped only when it is the V
SUBSTITUTES = {
    "dom9": (6, "dom9"),
    "dom13": (6, "dom9"),
    "maj9": (9, "m9"),
    "m9": (3, "maj9"),
    "m7": (3, "maj9"),
    "m11": (3, "maj9"),
}


class Pattern(NamedTuple):
    """A rhythm of the band and the energy (0 to 1) of the day that it suits."""

    name: str
    steps: tuple[Any, ...]
    energy: float


# comping: (start step of 16, length in steps, relative velocity)
COMPS: tuple[Pattern, ...] = (
    Pattern("held", ((0, 12, 1.0),), 0.0),
    Pattern("late", ((3, 5, 0.95), (10, 5, 0.8)), 0.25),
    Pattern("push", ((0, 6, 1.0), (6, 3, 0.7), (10, 6, 0.85)), 0.45),
    Pattern("two and four", ((2, 4, 0.9), (8, 4, 1.0), (12, 3, 0.7)), 0.6),
    Pattern("syncopated", ((0, 3, 1.0), (3, 3, 0.75), (8, 6, 0.9)), 0.8),
    Pattern("stabs", ((0, 2, 1.0), (6, 2, 0.7), (8, 2, 0.9), (14, 2, 0.6)), 0.95),
)
# bass: (start step, length in steps, 'r' root, 'f' fifth, 'o' octave, 't' third or 's' seventh of the chord)
BASSES: tuple[Pattern, ...] = (
    Pattern("whole note", ((0, 14, "r"),), 0.0),
    Pattern("long and short", ((0, 4, "r"), (8, 2, "r"), (11, 2, "f"), (14, 2, "r")), 0.15),
    Pattern("root and third", ((0, 6, "r"), (8, 4, "t"), (14, 2, "f")), 0.3),
    Pattern("walking root", ((0, 6, "r"), (10, 2, "f"), (14, 2, "r")), 0.4),
    Pattern("seventh fall", ((0, 4, "r"), (6, 2, "s"), (8, 4, "f"), (14, 2, "t")), 0.5),
    Pattern("round", ((0, 3, "r"), (3, 1, "r"), (6, 2, "f"), (10, 2, "r"), (13, 2, "o")), 0.6),
    Pattern("arpeggio", ((0, 3, "r"), (4, 2, "t"), (8, 2, "f"), (11, 2, "s"), (14, 2, "o")), 0.7),
    Pattern("step", ((0, 2, "r"), (4, 2, "r"), (8, 2, "f"), (10, 2, "r"), (14, 1, "o")), 0.85),
    Pattern(
        "syncopated",
        ((0, 2, "r"), (3, 1, "r"), (6, 2, "t"), (10, 1, "f"), (11, 2, "r"), (14, 2, "o")),
        1.0,
    ),
)
BASS_TONES = {"r": 0, "f": 7, "o": 12}
BASS_THIRD, BASS_SEVENTH = "t", "s"
# the minor and the major seventh above the root; every quality holds one, and its first interval is the third
SEVENTHS = (10, 11)
# drums on a 16-step bar: kick steps, snare steps, ghost steps (a soft snare or a rim)
DRUMS: tuple[Pattern, ...] = (
    Pattern("half time", ((0, 10), (8,), (6, 15)), 0.0),
    Pattern("lazy", ((0, 3, 10), (4, 12), (9, 14)), 0.2),
    Pattern("classic", ((0, 7, 10), (4, 12), (15,)), 0.45),
    Pattern("late kick", ((0, 5, 8, 11), (4, 12), ()), 0.55),
    Pattern("shuffle", ((0, 8, 11), (4, 12), (14,)), 0.7),
    Pattern("boom bap", ((0, 7, 10), (4, 12), (6, 14)), 0.78),
    Pattern("dusty", ((0, 3, 10, 13), (4, 12), (7, 15)), 0.85),
    Pattern("rolling", ((0, 6, 10, 13), (4, 12), (9,)), 0.9),
    Pattern("head nod", ((0, 2, 7, 10), (4, 12), (9, 14)), 1.0),
)


class Voice(NamedTuple):
    """A voice and how warm it sounds, -1 (glassy, dark) to 1 (round, warm)."""

    key: str
    warmth: float


STRING_VOICES = (Voice("juno_pad", 0.5), Voice("string_machine", -0.4))
TEMPOS = tuple(float(bpm) for bpm in range(int(LOFI.tempo[0]), int(LOFI.tempo[1]) + 1, 2))
# the swing of a sixteenth (how late the odd ones are): a quiet day lies back, a busy one plays straighter
SWING_QUIET = 0.30
SWING_BUSY = 0.10
SWING_DETAIL = 0.02
# the keys play their chords between these notes
KEYS_LOW = 53
KEYS_HIGH = 76
KEYS_CENTRE = 62
# the widest a voicing is: an octave and a fourth
KEYS_SPAN = 17
# the bass root lies from E1 up, and no note of the bass reaches C3 (MIDI 48)
BASS_LOWEST = 28
BASS_CEILING = 48
MOTIF_VELOCITY = 0.6
# the least strength of the drums by day for the quietest and the busiest day, and the same in the cheapest tier
DRUMS_FLOOR = (0.2, 0.85)
LIGHT_DRUMS_FLOOR = (0.15, 0.55)
STRINGS_FLOOR = 0.35
# the kit comes in over this stretch of energy, and a quiet day outside the night keeps this much of it
KIT_FROM = 0.1
KIT_FULL = 0.4
KIT_SOFT = 0.6
# the chord chops are a short accent, so their layer plays louder than the other layers to be heard
CHOP_GAIN = 1.2
# the keys sit a few dB behind the drums and the bass, softened by a little more room
KEYS_GAIN = 0.44
KEYS_REVERB = 0.45
# the plucked line leads from just under the drums
PLUCK_GAIN = 0.67
# how much of the kit, the comping and the bass each role of the form plays
KIT_BY_ROLE = {INTRO: 0.0, BUILD: 0.6, CLIMAX: 1.0, RELEASE: 0.5, PLATEAU: 0.85, OUTRO: 0.0}
COMP_BY_ROLE = {INTRO: 0.4, BUILD: 0.7, CLIMAX: 1.0, RELEASE: 0.7, PLATEAU: 0.85, OUTRO: 0.4}
SILENT_BASS = frozenset({INTRO, OUTRO})
# the share of the comping that is kept, from the quietest to the fullest moment
COMP_KEEP = (0.25, 1.0)
COMP_ENERGY_SHARE = 0.55
# the hits of the fill that leads out of a build or a breakdown: (step of 16, velocity)
FILL = ((10, 0.4), (12, 0.5), (13, 0.6), (14, 0.75), (15, 0.9))
KIT_LEAST = 0.12
KIT_PLAYING = 0.15
HAT_EIGHTH_LEAST = 0.45
HAT_SIXTEENTH_FROM = 0.55
HAT_SIXTEENTH_MOST = 0.8
# the three layers that can lead; the room picks its own, and a happening of the day may hand a phrase to another
LEADS = (LAYER_KEYS, LAYER_PLUCK, LAYER_STRINGS)
# what each layer is trimmed by while it leads: the pluck and the strings sit a little over the others, the keys, the
# front instrument, further
LEAD_TRIM_DB = {LAYER_KEYS: -4.0, LAYER_PLUCK: -2.0, LAYER_STRINGS: 2.0}
LEAD_BY_TRAIT = {
    lead.PERCUSSIVE: LAYER_PLUCK,
    lead.RHYTHMIC: LAYER_KEYS,
    lead.SUSTAINED: LAYER_STRINGS,
}
# the comping keeps this share of its hits and the chops this share of their chance when another layer leads
KEYS_BACK_KEEP = 0.7
CHOP_BACK_SHARE = 0.4
# the plucked line: a statement of the motif developed, at most this many beats, in the first and the middle bar of a
# phrase; the line is moved by up to this many scale steps so that it opens on a tone of the chord
PLUCK_LINE_BEATS = 8.0
PLUCK_LINE_VELOCITY = 0.46
PLUCK_NOTE_SHARE = 0.95
CHORD_SHIFTS = (0, -1, 1, -2, 2, -3, 3)
# the strings line: one note of up to this many bars at a time from this note up an octave, a leap of at most
# this many semitones between chord tones, a bar of rest at the end of a phrase, a slow vibrato
STRINGS_LINE_LOW = 64
STRINGS_LINE_VELOCITY = 0.5
STRINGS_LINE_BARS = 2
STRINGS_LINE_LEAP = 4
STRINGS_LINE_REST_BARS = 1
STRINGS_NOTE_SHARE = 0.97
STRINGS_VIBRATO_CENTS = 10.0
# under a strings line the held triad is this much softer, and the layer is at least this present
STRINGS_BED_SHARE = 0.7
STRINGS_LEAD_PRESENCE = 0.5
BEATS_PER_BAR = 4
# the patches of the keys and of the drum kit that a day picks from, and of the other voices
FULL_PATCH_COUNT = 3
SHORT_PATCH_COUNT = 2
LAYER_DRUMS = "drums"
LAYER_CHOP = "chop"
TEXTURE_VINYL = "vinyl"
DOMINANT_PREFIX = "dom"
DRUM_SNARE = "drum:snare_lofi"
DRUM_KICK = "drum:kick_lofi"
DRUM_RIM = "drum:rim"
DRUM_HAT = "drum:hat_closed"
# a chord chop starts this many steps after the beat, a plucked flourish steps through one of these shapes in semitones
CHOP_STEPS = (6, 10, 14)
FLOURISH_MINOR = (0, 2, 3, 7, 10)
FLOURISH_MAJOR = (0, 2, 4, 7, 9)
# the strings hold the root, the fifth and the octave
STRINGS_BED_TONES = (0, 7, SEMITONES_PER_OCTAVE)
# the activity that brings the kit in over this stretch
KIT_GATE_FROM = 0.22
KIT_GATE_TO = 0.5
# the night takes this share of the kit away
NIGHT_KIT_DUCK = 0.85
# the chance per bar that a chord is swapped, and what the activity adds
SWAP_BASE = 0.12
SWAP_ACTIVITY = 0.3
COMP_DRIFT_CHANCE = 0.25
# the chance per loop that the bass or the drums leave their pattern for a neighbour of it, or come back to it
RHYTHM_DRIFT_CHANCE = 0.2
MOTIF_NOTE_SHARE = 0.95
# the lights or the evening this strong bring in the extended chords
EXTENDED_LIGHTS = 0.5
EXTENDED_EVENING = 0.55
# a comping hit this long is never left out by the thinning
MAX_THINNED_STEPS = 6
COMP_TIMING_JITTER = 0.012
COMP_HOLD_OVERLAP = 1.1
CHORD_HOLD_SHARE = 0.95
CHORD_REVERSE_CHANCE = 0.3
COMP_VELOCITY = 0.5
COMP_VELOCITY_REL = 0.12
COMP_BRIGHT_FLOOR = 0.85
COMP_BRIGHT_SHARE = 0.15
COMP_VELOCITY_JITTER = 0.05
COMP_VELOCITY_MIN = 0.2
COMP_VELOCITY_MAX = 0.95
# the notes of a chord are struck between these seconds apart
STRUM_MIN = 0.008
STRUM_MAX = 0.028
BASS_REST_CHANCE = 0.08
BASS_TIMING_JITTER = 0.006
BASS_HOLD_SHARE = 0.9
BASS_MIN_HOLD = 0.12
BASS_VELOCITY = 0.62
BASS_VELOCITY_POWER = 0.15
BASS_VELOCITY_JITTER = 0.04
BASS_VELOCITY_MIN = 0.3
BASS_VELOCITY_MAX = 0.95
FILL_DELAY = 0.012
DRUM_HOLD = 0.3
FILL_VELOCITY_FLOOR = 0.7
FILL_VELOCITY_SHARE = 0.3
DRUM_TIMING_JITTER = 0.006
KICK_VELOCITY = 0.55
KICK_VELOCITY_LEVEL = 0.25
KICK_KEEP = 0.9
SNARE_TIMING_JITTER = 0.01
SNARE_DELAY = 0.012
SNARE_VELOCITY = 0.55
SNARE_VELOCITY_LEVEL = 0.3
SNARE_VELOCITY_JITTER = 0.04
GHOST_HOLD = 0.2
GHOST_VELOCITY = 0.25
GHOST_VELOCITY_RANDOM = 0.1
GHOST_CHANCE = 0.15
GHOST_CHANCE_ENERGY = 0.7
HAT_ACCENT_EIGHTH = 0.75
HAT_ACCENT_SIXTEENTH = 0.45
HAT_HOLD = 0.1
HAT_VELOCITY = 0.35
HAT_VELOCITY_LEVEL = 0.25
HAT_VELOCITY_JITTER = 0.03
CHOP_BASE = 0.02
CHOP_ENERGY = 0.14
CHOP_MIN_DRUMS = 0.3
CHOP_STAGGER = 0.006
CHOP_HOLD = 0.12
FLOURISH_CYCLE = 12
FLOURISH_BAR = 6
FLOURISH_MIN_ACTIVITY = 0.12
FLOURISH_CHANCE = 0.15
FLOURISH_CHANCE_ENERGY = 0.55
FLOURISH_NOTES_MIN = 3
FLOURISH_NOTES_MAX = 6
STRINGS_MIN_WARMTH = 0.3
STRINGS_BED_STAGGER = 0.1
STRINGS_BED_VELOCITY = 0.3
STRINGS_BED_VELOCITY_WARM = 0.2
STRINGS_BED_INTENSITY_FLOOR = 0.6
STRINGS_BED_INTENSITY_SHARE = 0.4
TONE_WALK_SECONDS = 60.0
TONE_WALK_RANGE = 0.1
TONE_FLOOR = 0.35
TONE_BRIGHT = 0.5
KEYS_CUTOFF_BASE = 1500.0
KEYS_CUTOFF_RANGE = 3.4
KEYS_DIP_DEPTH = 0.88
# seconds that a dip of the cutoff stays closed, and takes to open again
KEYS_DIP_HOLD = 1.0
KEYS_DIP_BACK = 4.0
DRUMS_CUTOFF_BASE = 3500.0
DRUMS_CUTOFF_RANGE = 3.0
DRUMS_DIP_DEPTH = 0.7
DRUMS_DIP_HOLD = 1.0
DRUMS_DIP_BACK = 3.0
SWEEP_RISE = 3.0
SWEEP_FALL = 6.0
KEYS_RESONANCE = 0.12
BASS_PRESENCE_FLOOR = 0.7
BASS_PRESENCE_POWER = 0.3
BASS_PRESENCE_ACTIVITY = 0.4
DRUMS_GAIN = 0.85
DRUMS_RESONANCE = 0.1
DRUMS_REVERB = 0.1
CHOP_CUTOFF = 0.6
CHOP_RESONANCE = 0.2
CHOP_ECHO = 0.2
CHOP_REVERB = 0.35
PLUCK_REVERB = 0.4
PLUCK_ECHO = 0.15
STRINGS_GAIN = 0.7
STRINGS_GATE_FROM = 0.15
STRINGS_CUTOFF = 0.8
STRINGS_RESONANCE = 0.1
STRINGS_REVERB = 0.6
SPACE_BASE = 0.18
SPACE_HUMID = 0.25
SPACE_SWEEP = 0.2
SPACE_MAX = 0.8
VINYL_GAIN = 0.003
ECHO_BEATS = 0.75
ECHO_FEEDBACK = 0.38
COMPRESS_DB = -17.0
WOBBLE = 0.9
LEVEL_DB = -10.2


def compose(plan: Plan) -> Score:
    """
    Make the score of a Lo-fi soundscape.

    :param plan: The curves and the deals of the day.
    """
    piece = _Piece(plan)
    _pick_tempo(piece)
    _pick_loop(piece)
    _pick_voices_and_swing(piece)
    _set_up_kit(piece)
    _lay_out_form(piece)
    _plan_fills_and_motif(piece)
    _plan_leads(piece)
    _play_bars(piece)
    _add_layers(piece)
    _tune_layers(piece)
    return _assemble(piece)


def _warm_fit(voices: Sequence[Voice], warmth: float) -> list[float]:
    """Return how well each voice suits the warmth of the day."""
    return [closeness(voice.warmth, warmth, SIGNED_TAG_SPAN) for voice in voices]


def _voicing(
    scale_root: int, chord: Chord, previous: Sequence[int] | None, *, extended: bool
) -> list[int]:
    """Return the notes of a chord above its root (the root is the bass's), led from the previous voicing."""
    offset, quality = chord
    name = EXTENDED[quality] if extended else quality
    tones = [
        (scale_root + offset + interval) % SEMITONES_PER_OCTAVE for interval in QUALITIES[name]
    ]
    places = [
        [n for n in range(KEYS_LOW, KEYS_HIGH + 1) if n % SEMITONES_PER_OCTAVE == tone]
        for tone in tones
    ]
    best: list[int] = []
    best_cost = float("inf")
    for combo in itertools.product(*places):
        notes = sorted(combo)
        span = notes[-1] - notes[0]
        if span > KEYS_SPAN:
            continue
        if previous:
            moved = sum(abs(a - b) for a, b in zip(sorted(previous), notes, strict=False))
            cost = float(moved + 2 * abs(len(previous) - len(notes)))
        else:
            cost = abs(sum(notes) / len(notes) - KEYS_CENTRE) * len(notes)
        cost += span / KEYS_SPAN
        if cost < best_cost:
            best, best_cost = notes, cost
    return best


@dataclass
class _Piece:
    """The working state of one Lo-fi piece, read and appended to by the sections of `compose`."""

    plan: Plan
    length: int = field(init=False)
    rng: np.random.Generator = field(init=False)
    energy: float = field(init=False)
    root: int = field(init=False)
    light: bool = field(init=False)
    bpm: float = field(init=False)
    beat: float = field(init=False)
    step: float = field(init=False)
    bar: float = field(init=False)
    phrase_bars: int = field(init=False)
    phrase: float = field(init=False)
    prog: Progression = field(init=False)
    comp_pattern: Pattern = field(init=False)
    bass_pattern: Pattern = field(init=False)
    drum_pattern: Pattern = field(init=False)
    keys_sig: SigVoice = field(init=False)
    bass_sig: SigVoice = field(init=False)
    pluck_sig: SigVoice = field(init=False)
    keys_voice: str = field(init=False)
    bass_voice: str = field(init=False)
    pluck_voice: str = field(init=False)
    strings_voice: str = field(init=False)
    chord_span: int = field(init=False)
    patch: int = field(init=False)
    keys_patch: int = field(init=False)
    bass_patch: int = field(init=False)
    pluck_patch: int = field(init=False)
    swing: float = field(init=False)
    kit_scale: float = field(init=False)
    drums_on: np.ndarray = field(init=False)
    eighths: float = field(init=False)
    sixteenths: float = field(init=False)
    sections: list[tuple[float, float, str, float]] = field(init=False)
    starts: list[float] = field(init=False)
    breakdown: tuple[float, float] = field(init=False)
    strings_from: float = field(init=False)
    fill_bars: set[int] = field(init=False)
    motif_bars: dict[int, list[tuple[int, float]]] = field(init=False)
    default_lead: str = field(init=False)
    leads: lead.Leads = field(init=False)
    line: _Line = field(init=False)
    current: list[Chord] = field(init=False)
    comp_now: Pattern = field(init=False)
    bass_now: Pattern = field(init=False)
    drum_now: Pattern = field(init=False)
    notes: list[Note] = field(default_factory=list)
    variations: dict[str, int] = field(
        default_factory=lambda: {
            "swap": 0,
            "comp": 0,
            "chop": 0,
            "pluck": 0,
            "motif": 0,
            "bass": 0,
            "drums": 0,
        }
    )
    chord_index: int = 0
    previous: list[int] | None = None
    keys_cut: np.ndarray = field(init=False)
    sweep: np.ndarray = field(init=False)
    layers: dict[str, Layer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Take what the plan already says; the first draw of the piece comes after it."""
        self.length = self.plan.length
        self.rng = self.plan.rng("lofi")
        self.energy = self.plan.character.energy
        self.root = self.plan.scale.root
        self.light = self.plan.quality.max_layers <= LIGHT_MAX_LAYERS


class _Moment(NamedTuple):
    """One chord of the loop as it is played: where it starts, the form around it and how it is voiced."""

    bar_index: int
    second: float
    role: str
    intensity: float
    in_break: bool
    chord: Chord
    voicing: list[int]
    here_drums: float
    lead_now: str | None


def _pick_tempo(piece: _Piece) -> None:
    """Pick the tempo, which sets the grid, and the length of a phrase."""
    plan = piece.plan
    low, high = LOFI.tempo
    target = tempo_target(LOFI, plan.character)
    piece.bpm = plan.pick(
        "tempo",
        TEMPOS,
        [closeness((t - low) / (high - low), target) for t in TEMPOS],
        tolerance=TEMPO_TOLERANCE,
    )
    piece.beat = SECONDS_PER_MINUTE / piece.bpm
    piece.step = piece.beat / 4.0
    piece.bar = piece.beat * 4.0
    piece.phrase_bars = (
        FORM_BARS if piece.length / (FORM_BARS * piece.bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
    )
    piece.phrase = piece.phrase_bars * piece.bar


def _pick_loop(piece: _Piece) -> None:
    """Pick the progression and the comping, bass and drum patterns of the day."""
    plan = piece.plan
    energy = piece.energy
    piece.prog = plan.pick(
        "progression",
        PROGRESSIONS,
        [
            MODE_WEIGHT * float(plan.scale.mode in p.modes)
            + (1.0 - MODE_WEIGHT) * closeness(p.motion, energy)
            for p in PROGRESSIONS
        ],
    )
    piece.comp_pattern = plan.pick("comping", COMPS, [closeness(c.energy, energy) for c in COMPS])
    piece.bass_pattern = plan.pick(
        "bass pattern", BASSES, [closeness(b.energy, energy) for b in BASSES]
    )
    piece.drum_pattern = plan.pick(
        "drum pattern", DRUMS, [closeness(d.energy, energy) for d in DRUMS]
    )
    piece.current = list(piece.prog.chords)
    piece.comp_now = piece.comp_pattern
    piece.bass_now = piece.bass_pattern
    piece.drum_now = piece.drum_pattern


def _pick_voices_and_swing(piece: _Piece) -> None:
    """Pick the voices, the length of a chord, the patch and the swing."""
    plan = piece.plan
    piece.keys_sig = signature(plan, LAYER_KEYS, "keys")
    piece.bass_sig = signature(plan, LAYER_BASS, "bass voice")
    piece.pluck_sig = signature(plan, LAYER_PLUCK, "pluck")
    piece.keys_voice = piece.keys_sig.voice
    piece.bass_voice = piece.bass_sig.voice
    piece.pluck_voice = piece.pluck_sig.voice
    piece.strings_voice = plan.pick(
        "strings", STRING_VOICES, _warm_fit(STRING_VOICES, plan.character.warmth)
    ).key
    options = tuple(o for o in chord_bars(plan.character) if o <= piece.phrase_bars) or (
        piece.phrase_bars,
    )
    piece.chord_span = plan.pick("chord bars", options, bars_fit(len(options), piece.energy))
    piece.patch = int(plan.rng("patch").integers(0, FULL_PATCH_COUNT))
    plan.choices["patch"] = str(piece.patch)
    piece.keys_patch = patch_of(piece.keys_sig, piece.patch)
    piece.bass_patch = patch_of(piece.bass_sig, piece.patch % SHORT_PATCH_COUNT)
    piece.pluck_patch = patch_of(piece.pluck_sig, piece.patch % SHORT_PATCH_COUNT)
    piece.swing = (
        SWING_QUIET
        + (SWING_BUSY - SWING_QUIET) * piece.energy
        + SWING_DETAIL * float(plan.rng("swing").uniform(-1.0, 1.0))
    )


def _set_up_kit(piece: _Piece) -> None:
    """
    Set how much of the kit plays and how busy its hats are.

    The kit is part of the base of the day: the house moves how hard it plays, the energy of the day how much of it
    plays, and the night lets it rest.
    """
    plan = piece.plan
    energy = piece.energy
    floor_quiet, floor_busy = LIGHT_DRUMS_FLOOR if piece.light else DRUMS_FLOOR
    kit_scale = float(np.clip((energy - KIT_FROM) / (KIT_FULL - KIT_FROM), 0.0, 1.0))
    kit_scale = kit_scale * kit_scale * (3.0 - 2.0 * kit_scale)
    piece.kit_scale = KIT_SOFT + (1.0 - KIT_SOFT) * kit_scale
    piece.drums_on = (
        plan.lift(
            np.maximum(plan.gate(plan.activity, KIT_GATE_FROM, KIT_GATE_TO), plan.runs[RUN_DISH]),
            floor_quiet + (floor_busy - floor_quiet) * energy,
        )
        * piece.kit_scale
        * (1.0 - NIGHT_KIT_DUCK * plan.night)
    )
    piece.eighths = HAT_EIGHTH_LEAST + (1.0 - HAT_EIGHTH_LEAST) * energy
    piece.sixteenths = (
        HAT_SIXTEENTH_MOST * max(energy - HAT_SIXTEENTH_FROM, 0.0) / (1.0 - HAT_SIXTEENTH_FROM)
    )


def _lay_out_form(piece: _Piece) -> None:
    """Lay out the form on phrases, and find the strongest event that turns the next phrase."""
    plan = piece.plan
    piece.sections = plan.sections(piece.bar)
    piece.starts = [s[0] for s in piece.sections]
    piece.breakdown = (0.0, 0.0)
    piece.strings_from = float(piece.length)
    if plan.events:
        when, _ = max(plan.events, key=lambda e: e[1])
        edge = (int(when / piece.phrase) + 1) * piece.phrase
        if edge + 2 * piece.phrase < piece.length:
            if _role_at(piece, edge)[0] in {CLIMAX, PLATEAU}:
                piece.breakdown = (edge, edge + piece.phrase)
            else:
                piece.strings_from = edge


def _plan_fills_and_motif(piece: _Piece) -> None:
    """Find the bars that end in a fill and the bars where the motif of the room is stated."""
    plan = piece.plan
    sections, bar = piece.sections, piece.bar
    piece.fill_bars = {
        round(end / bar) - 1 for _, end, role, _ in sections if role == BUILD and end < piece.length
    }
    if piece.breakdown[1]:
        piece.fill_bars.add(round(piece.breakdown[1] / bar) - 1)
    sites = [i for i, (_, _, role, _) in enumerate(sections) if role in {CLIMAX, PLATEAU}]
    piece.motif_bars = {
        round(sections[i][0] / bar): motif_notes(
            plan, develop_motif(plan.motif(), number, motif_seed(plan))
        )
        for number, i in enumerate(motif_sites(sites or [len(sections) - 1]))
    }


def _plan_leads(piece: _Piece) -> None:
    """Hand the phrases to the layers that lead and set up the melodic lines."""
    plan = piece.plan
    piece.default_lead = lead.default_lead(plan, LEADS)
    allowed = [
        ok and not piece.breakdown[0] <= index * piece.phrase < piece.breakdown[1]
        for index, ok in enumerate(lead.daytime_phrases(plan, piece.sections, piece.phrase))
    ]
    piece.leads = lead.phrase_leads(plan, piece.phrase, piece.default_lead, allowed, LEAD_BY_TRAIT)
    piece.line = _Line(
        plan,
        piece.bar,
        piece.phrase_bars,
        piece.pluck_voice,
        piece.pluck_patch,
        piece.strings_voice,
        piece.patch % SHORT_PATCH_COUNT,
    )


def _role_at(piece: _Piece, second: float) -> tuple[str, float]:
    """Return the role of the form at a second and how intense it is."""
    found = piece.sections[max(bisect_right(piece.starts, second) - 1, 0)]
    return found[2], found[3]


def _in_breakdown(piece: _Piece, second: float) -> bool:
    """Return whether the second lies in the breakdown."""
    return piece.breakdown[0] <= second < piece.breakdown[1]


def _play_bars(piece: _Piece) -> None:
    """Play the loop bar by bar: the variations, the motif and, where a chord starts, everything on it."""
    for bar_index in range(int(piece.length / piece.bar)):
        second = bar_index * piece.bar
        if bar_index % (4 * piece.chord_span) == 0 and bar_index > 0:
            _vary_loop(piece, second)
        _state_motif(piece, bar_index, second)
        if bar_index % piece.chord_span == 0:
            _play_chord(piece, bar_index, second)


def _vary_loop(piece: _Piece, second: float) -> None:
    """Vary the loop by rule: a chord may be swapped for a related one, the comping, the bass and the drums may drift."""
    rng = piece.rng
    current, chords = piece.current, piece.prog.chords
    if rng.random() < SWAP_BASE + SWAP_ACTIVITY * piece.plan.at(piece.plan.activity, second):
        k = int(rng.integers(0, len(current)))
        off, qual = current[k]
        sub = SUBSTITUTES.get(qual)
        if qual.startswith(DOMINANT_PREFIX) and off != DOMINANT:
            sub = None
        if sub is not None and current[k] == chords[k]:
            current[k] = (off + sub[0], sub[1])
            piece.variations["swap"] += 1
        else:
            current[k] = chords[k]
    if rng.random() < COMP_DRIFT_CHANCE:
        at_index = COMPS.index(piece.comp_now)
        neighbours = [i for i in (at_index - 1, at_index + 1) if 0 <= i < len(COMPS)]
        piece.comp_now = COMPS[neighbours[int(rng.integers(0, len(neighbours)))]]
        piece.variations["comp"] += 1
    if rng.random() < RHYTHM_DRIFT_CHANCE:
        piece.bass_now = _drifted(rng, BASSES, piece.bass_now, piece.bass_pattern)
        piece.variations["bass"] += 1
    if rng.random() < RHYTHM_DRIFT_CHANCE:
        piece.drum_now = _drifted(rng, DRUMS, piece.drum_now, piece.drum_pattern)
        piece.variations["drums"] += 1


def _drifted(
    rng: np.random.Generator, family: tuple[Pattern, ...], now: Pattern, home: Pattern
) -> Pattern:
    """Return the neighbour of the pattern in the family when it is the pattern of the day, otherwise that pattern."""
    if now != home:
        return home
    at_index = family.index(now)
    neighbours = [i for i in (at_index - 1, at_index + 1) if 0 <= i < len(family)]
    return family[neighbours[int(rng.integers(0, len(neighbours)))]]


def _state_motif(piece: _Piece, bar_index: int, second: float) -> None:
    """State the motif of the room on the plucked line where the form asks for it."""
    if bar_index not in piece.motif_bars or _in_breakdown(piece, second):
        return
    piece.variations["motif"] += 1
    clock = second
    for midi, beats in piece.motif_bars[bar_index]:
        piece.notes.append(
            Note(
                clock,
                beats * piece.beat * MOTIF_NOTE_SHARE,
                float(midi),
                MOTIF_VELOCITY,
                piece.pluck_voice,
                LAYER_PLUCK,
                piece.pluck_patch,
            )
        )
        clock += beats * piece.beat


def _play_chord(piece: _Piece, bar_index: int, second: float) -> None:
    """Play everything that starts with a chord: the comping, the lead, the bass, the drums and the extras."""
    plan = piece.plan
    role, intensity = _role_at(piece, second)
    in_break = _in_breakdown(piece, second)
    chord = piece.current[piece.chord_index % len(piece.current)]
    piece.chord_index += 1
    here_drums = plan.at(piece.drums_on, second) * (0.0 if in_break else KIT_BY_ROLE[role])
    extended = (
        plan.at(plan.lights, second) > EXTENDED_LIGHTS
        or plan.at(plan.evening, second) > EXTENDED_EVENING
    )
    voicing = _voicing(piece.root, chord, piece.previous, extended=extended)
    piece.previous = voicing
    moment = _Moment(
        bar_index,
        second,
        role,
        intensity,
        in_break,
        chord,
        voicing,
        here_drums,
        piece.leads.at(second),
    )
    _play_comping(piece, moment)
    _play_chord_line(piece, moment)
    _play_rhythm_section(piece, moment)
    _play_chop(piece, moment)
    _play_plucked_flourish(piece, moment)
    _play_strings_bed(piece, moment)


def _play_comping(piece: _Piece, moment: _Moment) -> None:
    """Play the keys on the chord; with the drums at rest it holds the chord, a quiet piece."""
    plan, rng = piece.plan, piece.rng
    second = moment.second
    playing = moment.here_drums > KIT_PLAYING
    pattern = piece.comp_now.steps if playing else COMPS[0].steps
    density = float(
        np.clip(
            COMP_ENERGY_SHARE * piece.energy
            + (1.0 - COMP_ENERGY_SHARE) * plan.at(plan.activity, second),
            0.0,
            1.0,
        )
        * COMP_BY_ROLE[moment.role]
        * (0.5 + 0.5 * moment.intensity)
    )
    keep = COMP_KEEP[0] + (COMP_KEEP[1] - COMP_KEEP[0]) * density
    if moment.lead_now not in {None, LAYER_KEYS}:
        keep *= KEYS_BACK_KEEP
    rounds = 1 if piece.light or not playing else piece.chord_span
    for b in range(rounds):
        for start, long, rel in pattern:
            if rng.random() > keep and long < MAX_THINNED_STEPS:
                continue
            at = (
                second
                + b * piece.bar
                + swung(start, piece.step, piece.swing)
                + float(rng.normal(0.0, COMP_TIMING_JITTER))
            )
            hold = (
                min(long * piece.step * COMP_HOLD_OVERLAP, piece.bar - start * piece.step)
                if playing
                else piece.chord_span * piece.bar * CHORD_HOLD_SHARE
            )
            order = list(moment.voicing)
            if rng.random() < CHORD_REVERSE_CHANCE:
                order.reverse()
            for k, midi in enumerate(order):
                vel = float(
                    np.clip(
                        (COMP_VELOCITY + COMP_VELOCITY_REL * rel)
                        * (COMP_BRIGHT_FLOOR + COMP_BRIGHT_SHARE * plan.at(plan.bright, second))
                        + rng.normal(0.0, COMP_VELOCITY_JITTER),
                        COMP_VELOCITY_MIN,
                        COMP_VELOCITY_MAX,
                    )
                )
                piece.notes.append(
                    Note(
                        at + k * float(rng.uniform(STRUM_MIN, STRUM_MAX)),
                        hold,
                        float(midi),
                        vel,
                        piece.keys_voice,
                        LAYER_KEYS,
                        piece.keys_patch,
                    )
                )


def _play_chord_line(piece: _Piece, moment: _Moment) -> None:
    """Play the melodic line of the lead on the chord when the plucked line or the strings lead."""
    chord_tones = {n % SEMITONES_PER_OCTAVE for n in moment.voicing} | {
        (piece.root + moment.chord[0]) % SEMITONES_PER_OCTAVE
    }
    if moment.lead_now == LAYER_PLUCK:
        piece.notes.extend(
            piece.line.pluck(
                moment.bar_index,
                piece.chord_span,
                moment.second,
                chord_tones,
                set(piece.motif_bars),
            )
        )
    elif moment.lead_now == LAYER_STRINGS:
        piece.notes.extend(
            piece.line.strings(moment.bar_index, piece.chord_span, moment.second, chord_tones)
        )


def _play_rhythm_section(piece: _Piece, moment: _Moment) -> None:
    """Play the bass and the drums bar by bar over the length of the chord."""
    root_midi = BASS_LOWEST + (piece.root + moment.chord[0] - BASS_LOWEST) % SEMITONES_PER_OCTAVE
    for b in range(piece.chord_span):
        base = moment.second + b * piece.bar
        role_b, intensity_b = _role_at(piece, base)
        silent = moment.in_break or role_b in SILENT_BASS
        if not silent:
            _play_bass(piece, moment, base, root_midi)
        _play_drum_bar(piece, moment.bar_index + b, base, role_b, intensity_b)


def _play_bass(piece: _Piece, moment: _Moment, base: float, root_midi: int) -> None:
    """Play the bass of one bar: the pattern on the root of the chord."""
    plan, rng = piece.plan, piece.rng
    for start, long, tone in piece.bass_now.steps:
        offset = _bass_offset(tone, moment.chord[1])
        if root_midi + offset >= BASS_CEILING:
            offset = 0
        if rng.random() < BASS_REST_CHANCE:
            continue
        piece.notes.append(
            Note(
                base
                + swung(start, piece.step, piece.swing)
                + float(rng.normal(0.0, BASS_TIMING_JITTER)),
                max(long * piece.step * BASS_HOLD_SHARE, BASS_MIN_HOLD),
                float(root_midi + offset),
                float(
                    np.clip(
                        BASS_VELOCITY
                        + BASS_VELOCITY_POWER * plan.at(plan.power, moment.second)
                        + rng.normal(0, BASS_VELOCITY_JITTER),
                        BASS_VELOCITY_MIN,
                        BASS_VELOCITY_MAX,
                    )
                ),
                piece.bass_voice,
                LAYER_BASS,
                piece.bass_patch,
            )
        )


def _play_drum_bar(
    piece: _Piece, this_bar: int, base: float, role_b: str, intensity_b: float
) -> None:
    """Play the drums of one bar: the fill that leads out of a build or a breakdown, then the kit."""
    plan = piece.plan
    kit_role = 0.0 if _in_breakdown(piece, base) else KIT_BY_ROLE[role_b]
    level = plan.at(piece.drums_on, base) * kit_role
    if this_bar in piece.fill_bars:
        ahead = KIT_BY_ROLE[_role_at(piece, base + piece.bar)[0]]
        if plan.at(piece.drums_on, base) * ahead >= KIT_LEAST:
            piece.notes.extend(
                Note(
                    base + fill_step * piece.step + FILL_DELAY,
                    DRUM_HOLD,
                    0.0,
                    fill_vel * (FILL_VELOCITY_FLOOR + FILL_VELOCITY_SHARE * intensity_b),
                    DRUM_SNARE,
                    LAYER_DRUMS,
                    piece.patch % FULL_PATCH_COUNT,
                )
                for fill_step, fill_vel in FILL
            )
    if level < KIT_LEAST:
        return
    kicks, snares, ghosts = piece.drum_now.steps
    _play_kicks(piece, kicks, base, level)
    _play_snares(piece, snares, base, level)
    _play_ghosts(piece, ghosts, base)
    _play_hats(piece, base, level)


def _play_kicks(piece: _Piece, kicks: tuple[int, ...], base: float, level: float) -> None:
    """Play the kicks of one bar; all but the first may drop out."""
    rng = piece.rng
    piece.notes.extend(
        Note(
            base + swung(k, piece.step, piece.swing) + float(rng.normal(0.0, DRUM_TIMING_JITTER)),
            DRUM_HOLD,
            0.0,
            KICK_VELOCITY + KICK_VELOCITY_LEVEL * level,
            DRUM_KICK,
            LAYER_DRUMS,
            piece.patch % FULL_PATCH_COUNT,
        )
        for k in kicks
        if k == 0 or rng.random() < KICK_KEEP
    )


def _play_snares(piece: _Piece, snares: tuple[int, ...], base: float, level: float) -> None:
    """Play the snares of one bar."""
    rng = piece.rng
    piece.notes.extend(
        Note(
            base + k * piece.step + float(rng.normal(0.0, SNARE_TIMING_JITTER)) + SNARE_DELAY,
            DRUM_HOLD,
            0.0,
            SNARE_VELOCITY
            + SNARE_VELOCITY_LEVEL * level
            + float(rng.normal(0, SNARE_VELOCITY_JITTER)),
            DRUM_SNARE,
            LAYER_DRUMS,
            piece.patch % FULL_PATCH_COUNT,
        )
        for k in snares
    )


def _play_ghosts(piece: _Piece, ghosts: tuple[int, ...], base: float) -> None:
    """Play the ghost notes of one bar, a soft snare or a rim."""
    plan, rng = piece.plan, piece.rng
    piece.notes.extend(
        Note(
            base + swung(k, piece.step, piece.swing),
            GHOST_HOLD,
            0.0,
            GHOST_VELOCITY + GHOST_VELOCITY_RANDOM * float(rng.random()),
            DRUM_RIM,
            LAYER_DRUMS,
            piece.patch % SHORT_PATCH_COUNT,
        )
        for k in ghosts
        if rng.random()
        < GHOST_CHANCE
        + GHOST_CHANCE_ENERGY * piece.energy * (0.5 + 0.5 * plan.at(plan.activity, base))
    )


def _play_hats(piece: _Piece, base: float, level: float) -> None:
    """Play the hats of one bar: the quarters always, the eighths and the sixteenths by chance."""
    rng = piece.rng
    for k in range(STEPS_PER_BAR):
        if k % 4 != 0:
            if k % 2 == 0 and rng.random() > piece.eighths:
                continue
            if k % 2 == 1 and (piece.light or level < 0.5 or rng.random() > piece.sixteenths):
                continue
        elif piece.kit_scale < 1.0 and rng.random() > piece.kit_scale:
            continue
        accent = 1.0 if k % 4 == 0 else HAT_ACCENT_EIGHTH if k % 2 == 0 else HAT_ACCENT_SIXTEENTH
        piece.notes.append(
            Note(
                base
                + swung(k, piece.step, piece.swing)
                + float(rng.normal(0.0, DRUM_TIMING_JITTER)),
                HAT_HOLD,
                0.0,
                accent * (HAT_VELOCITY + HAT_VELOCITY_LEVEL * level)
                + float(rng.normal(0, HAT_VELOCITY_JITTER)),
                DRUM_HAT,
                LAYER_DRUMS,
                piece.patch % FULL_PATCH_COUNT,
            )
        )


def _play_chop(piece: _Piece, moment: _Moment) -> None:
    """Play a chord chop now and then, as ear candy."""
    plan, rng = piece.plan, piece.rng
    second = moment.second
    chop_chance = CHOP_BASE + CHOP_ENERGY * piece.energy * plan.at(plan.activity, second)
    if moment.lead_now not in {None, LAYER_KEYS}:
        chop_chance *= CHOP_BACK_SHARE
    if moment.here_drums > CHOP_MIN_DRUMS and rng.random() < chop_chance:
        piece.variations["chop"] += 1
        at = second + float(rng.choice(CHOP_STEPS)) * piece.step
        for k, midi in enumerate(moment.voicing):
            piece.notes.append(
                Note(
                    at + k * CHOP_STAGGER,
                    CHOP_HOLD,
                    float(midi + SEMITONES_PER_OCTAVE),
                    0.5,
                    piece.keys_voice,
                    LAYER_CHOP,
                    piece.keys_patch,
                )
            )


def _play_plucked_flourish(piece: _Piece, moment: _Moment) -> None:
    """Play, once in a while, a plucked line on top of the chord."""
    plan, rng = piece.plan, piece.rng
    second = moment.second
    if not (
        moment.bar_index % FLOURISH_CYCLE == FLOURISH_BAR
        and moment.lead_now != LAYER_PLUCK
        and moment.bar_index not in piece.motif_bars
        and plan.at(plan.activity, second) > FLOURISH_MIN_ACTIVITY
        and rng.random() < FLOURISH_CHANCE + FLOURISH_CHANCE_ENERGY * piece.energy
    ):
        return
    piece.variations["pluck"] += 1
    top = moment.voicing[0] + SEMITONES_PER_OCTAVE
    shape = FLOURISH_MINOR if moment.chord[1].startswith("m") else FLOURISH_MAJOR
    piece.notes.extend(
        Note(
            second + k * piece.beat * 0.5,
            0.5,
            float(top + shape[int(rng.integers(0, len(shape)))]),
            0.5,
            piece.pluck_voice,
            LAYER_PLUCK,
            piece.pluck_patch,
        )
        for k in range(int(rng.integers(FLOURISH_NOTES_MIN, FLOURISH_NOTES_MAX)))
    )


def _play_strings_bed(piece: _Piece, moment: _Moment) -> None:
    """
    Play the strings: a soft triad on the chord (a tonic pedal where the chord changes every bar).

    The strings swell in when the lights are on in the evening, and come in from the phrase after the strongest event.
    """
    plan = piece.plan
    second = moment.second
    chord_span = piece.chord_span
    warm = max(plan.at(plan.lights, second) * plan.at(plan.evening, second), STRINGS_FLOOR)
    hold_bars = chord_span if chord_span >= 2 else FORM_BARS
    enters = moment.role != INTRO or second >= piece.strings_from
    if (
        moment.bar_index % hold_bars == 0
        and warm > STRINGS_MIN_WARMTH
        and enters
        and not moment.in_break
    ):
        held = (
            moment.voicing[:3]
            if chord_span >= 2
            else [
                KEYS_LOW + (piece.root - KEYS_LOW) % SEMITONES_PER_OCTAVE + o
                for o in STRINGS_BED_TONES
            ]
        )
        for k, midi in enumerate(held):
            piece.notes.append(
                Note(
                    second + STRINGS_BED_STAGGER * k,
                    piece.bar * hold_bars,
                    float(midi),
                    (STRINGS_BED_VELOCITY + STRINGS_BED_VELOCITY_WARM * warm)
                    * (STRINGS_BED_INTENSITY_FLOOR + STRINGS_BED_INTENSITY_SHARE * moment.intensity)
                    * (STRINGS_BED_SHARE if moment.lead_now == LAYER_STRINGS else 1.0),
                    piece.strings_voice,
                    LAYER_STRINGS,
                    piece.patch % SHORT_PATCH_COUNT,
                )
            )


def _add_layers(piece: _Piece) -> None:
    """Drop the notes outside the piece and set up the layers, with the tape low-pass that the day moves."""
    plan = piece.plan
    power, activity = plan.power, plan.activity
    piece.notes = [n for n in piece.notes if 0.0 <= n.t < piece.length]
    walk = block_walk(piece.rng, piece.length, TONE_WALK_SECONDS, TONE_WALK_RANGE)
    tone = np.clip(TONE_FLOOR + TONE_BRIGHT * plan.bright + walk, 0.0, 1.0)
    piece.keys_cut = (
        KEYS_CUTOFF_BASE
        * KEYS_CUTOFF_RANGE**tone
        * dips(plan, KEYS_DIP_DEPTH, KEYS_DIP_HOLD, KEYS_DIP_BACK)
    )
    drums_cut = (
        DRUMS_CUTOFF_BASE
        * DRUMS_CUTOFF_RANGE**tone
        * dips(plan, DRUMS_DIP_DEPTH, DRUMS_DIP_HOLD, DRUMS_DIP_BACK)
    )
    piece.sweep = swells(plan, SWEEP_RISE, SWEEP_FALL)
    keys_cut = piece.keys_cut
    piece.layers = {
        LAYER_KEYS: Layer(
            LAYER_KEYS,
            KEYS_GAIN,
            None,
            keys_cut,
            KEYS_RESONANCE,
            reverb=KEYS_REVERB,
            title="keys",
        ),
        LAYER_BASS: Layer(
            LAYER_BASS,
            1.0,
            BASS_PRESENCE_FLOOR
            + BASS_PRESENCE_POWER * np.clip(power + activity * BASS_PRESENCE_ACTIVITY, 0, 1),
            None,
            0.0,
            reverb=0.0,
            highpass=0.0,
            title="bass",
        ),
        LAYER_DRUMS: Layer(
            LAYER_DRUMS,
            DRUMS_GAIN,
            None,
            drums_cut,
            DRUMS_RESONANCE,
            reverb=DRUMS_REVERB,
            title="drums",
        ),
        LAYER_CHOP: Layer(
            LAYER_CHOP,
            CHOP_GAIN,
            None,
            keys_cut * CHOP_CUTOFF,
            CHOP_RESONANCE,
            reverb=CHOP_REVERB,
            echo=CHOP_ECHO,
            title="chord chops",
        ),
        LAYER_PLUCK: Layer(
            LAYER_PLUCK,
            PLUCK_GAIN,
            None,
            None,
            0.0,
            reverb=PLUCK_REVERB,
            echo=PLUCK_ECHO,
            title="plucked line",
        ),
        LAYER_STRINGS: Layer(
            LAYER_STRINGS,
            STRINGS_GAIN,
            plan.lift(plan.gate(plan.lights * plan.evening, STRINGS_GATE_FROM, 0.5), STRINGS_FLOOR),
            keys_cut * STRINGS_CUTOFF,
            STRINGS_RESONANCE,
            reverb=STRINGS_REVERB,
            title="strings",
        ),
    }


def _tune_layers(piece: _Piece) -> None:
    """Follow the lead on the layers, and trim the layers of the room's own voices."""
    layers = piece.layers
    curves = lead.lead_curves(piece.plan, piece.leads, LEADS)
    strings_presence = layers[LAYER_STRINGS].presence
    if strings_presence is not None:
        layers[LAYER_STRINGS].presence = np.maximum(
            strings_presence, STRINGS_LEAD_PRESENCE * np.clip(curves[LAYER_STRINGS], 0.0, 1.0)
        )
    lead.apply_curves(layers, curves, lead.LEAD_STEP_DB, LEAD_TRIM_DB)
    for key, sig in (
        (LAYER_KEYS, piece.keys_sig),
        (LAYER_CHOP, piece.keys_sig),
        (LAYER_BASS, piece.bass_sig),
        (LAYER_PLUCK, piece.pluck_sig),
    ):
        trim_layer(layers[key], sig)


def _assemble(piece: _Piece) -> Score:
    """Put the notes and the layers together as the score, with the notes about how it was made."""
    plan = piece.plan
    space = np.clip(
        SPACE_BASE + SPACE_HUMID * plan.humid + SPACE_SWEEP * piece.sweep, 0.0, SPACE_MAX
    )
    score = Score(
        piece.length,
        piece.notes,
        piece.layers,
        [Texture(TEXTURE_VINYL, TEXTURE_VINYL, VINYL_GAIN, None)],
        space,
        echo_seconds=piece.beat * ECHO_BEATS,
        echo_feedback=ECHO_FEEDBACK,
        tempo=piece.bpm,
        events=list(plan.events),
        compress_db=COMPRESS_DB,
        wobble=WOBBLE,
        level_db=LEVEL_DB,
    )
    variations = piece.variations
    plan.choices[MOTIF_CHOICE] = str(variations["motif"])
    plan.choices[lead.LEAD_HANDOVERS_CHOICE] = str(len(piece.leads.handed))
    score.leads = piece.leads
    score.choices = dict(plan.choices)
    form = ", ".join(f"{role} {round((b - a) / piece.bar)}" for a, b, role, _ in piece.sections)
    broke = f", a breakdown at {piece.breakdown[0]:.0f} s" if piece.breakdown[1] else ""
    chord_span = piece.chord_span
    score.notes_about = [
        (
            f"{piece.bpm:.0f} BPM, swing {piece.swing:.2f}, progression {piece.prog.name} ({chord_span} bar{'s' if chord_span > 1 else ''} a chord) on {piece.keys_voice}; "
            f"comping '{piece.comp_pattern.name}', bass '{piece.bass_pattern.name}' on {piece.bass_voice}, drums '{piece.drum_pattern.name}'."
        ),
        f"Form in bars: {form}; the motif of the room {variations['motif']} times{broke}.",
        lead.summary(piece.default_lead, piece.leads),
        f"Variations by rule: {variations['swap']} chord swaps, {variations['comp']} comping changes, {variations['bass']} bass and {variations['drums']} drum changes, {variations['chop']} chops, {variations['pluck']} plucked lines.",
    ]
    return score


@dataclass
class _Line:
    """The melodic lines of the lead: the plucked one from the motif of the room, the strings one slow and legato."""

    plan: Plan
    bar: float
    phrase_bars: int
    pluck_voice: str
    pluck_patch: int
    strings_voice: str
    strings_patch: int
    previous: int | None = None
    cursor: int = 0

    def pluck(
        self,
        bar_index: int,
        chord_span: int,
        second: float,
        chord_tones: set[int],
        taken: set[int],
    ) -> list[Note]:
        """
        Return the plucked line of a chord: the motif, developed, in the first and the middle bar of a phrase.

        :param bar_index: The first bar of the chord.
        :param chord_span: How many bars the chord lasts.
        :param second: The start of the chord.
        :param chord_tones: The pitch classes of the chord.
        :param taken: The bars where the motif is stated already.
        """
        places = (0, self.phrase_bars // 2)
        beat = self.bar / BEATS_PER_BAR
        found: list[Note] = []
        for b in range(chord_span):
            number = bar_index + b
            within = number % self.phrase_bars
            if within not in places or number in taken:
                continue
            statement = (number // self.phrase_bars) * len(places) + places.index(within)
            motif = develop_motif(
                self.plan.motif(), statement, motif_seed(self.plan), PLUCK_LINE_BEATS
            )
            clock = second + b * self.bar
            for midi, beats in motif_notes(self.plan, _opening_on(self.plan, motif, chord_tones)):
                found.append(
                    Note(
                        clock,
                        beats * beat * PLUCK_NOTE_SHARE,
                        float(midi),
                        PLUCK_LINE_VELOCITY,
                        self.pluck_voice,
                        LAYER_PLUCK,
                        self.pluck_patch,
                    )
                )
                clock += beats * beat
        return found

    def strings(
        self, bar_index: int, chord_span: int, second: float, chord_tones: set[int]
    ) -> list[Note]:
        """
        Return the strings line of a chord: long single notes, a chord tone as the chord comes, then steps.

        :param bar_index: The first bar of the chord.
        :param chord_span: How many bars the chord lasts.
        :param second: The start of the chord.
        :param chord_tones: The pitch classes of the chord.
        """
        degrees = [degree for degree, _ in self.plan.motif()]
        found: list[Note] = []
        for b in range(0, chord_span, min(chord_span, STRINGS_LINE_BARS)):
            sounding = (
                self.phrase_bars - STRINGS_LINE_REST_BARS - (bar_index + b) % self.phrase_bars
            )
            if sounding <= 0:
                break
            wanted = (
                STRINGS_LINE_LOW
                + (self.plan.scale.midi(degrees[self.cursor % len(degrees)], 4) - STRINGS_LINE_LOW)
                % SEMITONES_PER_OCTAVE
            )
            self.cursor += 1
            self.previous = self._next(wanted, chord_tones, strong=b == 0)
            found.append(
                Note(
                    second + b * self.bar,
                    min(sounding, STRINGS_LINE_BARS, chord_span - b)
                    * self.bar
                    * STRINGS_NOTE_SHARE,
                    float(self.previous),
                    STRINGS_LINE_VELOCITY,
                    self.strings_voice,
                    LAYER_STRINGS,
                    self.strings_patch,
                    STRINGS_VIBRATO_CENTS,
                )
            )
        return found

    def _next(self, wanted: int, chord_tones: set[int], *, strong: bool) -> int:
        """Return the next note of the strings line: a chord tone on a strong bar, else a scale step to the wanted one."""
        previous = self.previous
        if strong or previous is None:
            tones = [
                n
                for n in range(STRINGS_LINE_LOW, STRINGS_LINE_LOW + SEMITONES_PER_OCTAVE + 1)
                if n % SEMITONES_PER_OCTAVE in chord_tones
            ]
            nearest = min(tones, key=lambda n: abs(n - wanted))
            if previous is None or abs(nearest - previous) <= STRINGS_LINE_LEAP:
                return nearest
            return min(tones, key=lambda n: (abs(n - previous), abs(n - wanted)))
        scale_tones = {
            (self.plan.scale.root + interval) % SEMITONES_PER_OCTAVE
            for interval in self.plan.scale.intervals
        }
        direction = -1 if wanted < previous else 1
        for _ in range(2):
            moved = previous + direction
            while moved % SEMITONES_PER_OCTAVE not in scale_tones:
                moved += direction
            if STRINGS_LINE_LOW <= moved <= STRINGS_LINE_LOW + SEMITONES_PER_OCTAVE:
                return moved
            direction = -direction
        return previous


def _opening_on(
    plan: Plan, motif: tuple[tuple[int, float], ...], chord_tones: set[int]
) -> tuple[tuple[int, float], ...]:
    """Return a motif moved by the fewest scale steps so that its first note is a tone of the chord."""
    first = motif[0][0]
    for shift in CHORD_SHIFTS:
        if plan.scale.midi(first + shift, 4) % SEMITONES_PER_OCTAVE in chord_tones:
            return tuple((degree + shift, beats) for degree, beats in motif)
    return motif


def _bass_offset(tone: str, quality: str) -> int:
    """Return how many semitones a tone of a bass pattern lies above the root of a chord of a quality."""
    if tone == BASS_THIRD:
        return QUALITIES[quality][0]
    if tone == BASS_SEVENTH:
        return next(i for i in QUALITIES[quality] if i in SEVENTHS)
    return BASS_TONES[tone]
