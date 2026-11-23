"""
Berlin School: hypnotic sequences of different lengths that interlock and mutate by rule, over strings and an organ.

The generative system is a step sequencer with rules. Three sequencer lines run on the same 16th-note grid but
have different lengths (16, 12 and 8 or 7 steps), so they phase against each other and their combination only
repeats after a long time. The cells are composed (the vocabulary of Phaedra and Rubycon: octave jumps, a held
root, arpeggios); the rules change them over minutes: every four bars each line may drop a step, add one from the
chord, move one by a degree or an octave, rotate, or be pulled back towards its first form, and the whole
sequence is transposed when the chord root moves (a progression of the day, over a tonic pedal in the organ). Strings
and choir hold the chord, led from chord to chord, and a slow lead plays composed phrases of long notes into a tape
echo, and states the motif of the room in the first and the last peak or plateau. Who leads is the room's too (`lead`): the
first or the second sequencer line keeps the rhythm of its cell and takes the pitches of the motif, or the slow lead plays
a composed phrase in every phrase, in the daytime build, peak and plateau, the others stepping back and sparser; a door or
motion hands the next phrase to the first line, an appliance or the hob to the second, a light, a cover or the media to
the slow lead. There is no drum kit on a quiet day; on a busy one
in the morning and the day (never at night, gone before the evening) a soft kick, warm toms and a light shaker follow
the grid of the sequencer, far below it.

How the house steers it: the character of the day picks the identity: the tempo inside 102 to 116, the cells, the
voices of the second line and the strings (warm days the rounder ones; the first sequence and the lead are the room's own
and stay the same every day), the progression (fourth and fifth moves in the mode of the day, still on a quiet
day, restless on a busy one), the bars a chord lasts (8 on a quiet day, 2 on a busy one), the kit and the growling bass
(busy days only). Its energy also sets how fast the lines mutate and how many of them play, and its shape sets the
form: the lines, the choir and the drums join on 8-bar lines by the intensity of the section. The curves of the moment
work on top: activity brings the lines in, the power of appliances opens the filter, sets the accents and brings the
growl; the temperature and the sun move the cutoff; humidity sets the room and the echo; the lights and the evening
bring in the choir; a running dishwasher fades in the second line and a washer the third; an anomaly is a dip of the
filter with an echo throw and a swell of the strings, and the strongest one moves the next phrase to the relative key.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import lead
from .character import OCTAVE_DEGREES, tempo_target
from .gen import block_walk, dips, swells, weighted
from .harmony import (
    BARS_PER_PHRASE,
    GROWLS,
    LEAD_OCTAVE,
    MOTIF_CHOICE,
    NO_GROWL,
    STEPS_PER_BAR,
    Progression,
    VoiceTag,
    closeness,
    develop_motif,
    harmonic_rhythm,
    held_runs,
    lead_octave,
    motif_seed,
    motif_sites,
    pick_voice,
    progressions_for,
    section_curve,
    voice_lead,
)
from .plan import CLIMAX, LIGHT_MAX_LAYERS, PLATEAU, RUN_DISH, RUN_WASH, TEMPO_TOLERANCE
from .score import (
    CURVE_RATE,
    LAYER_GROWL,
    LAYER_LEAD,
    LAYER_SEQ_A,
    LAYER_STRINGS,
    VOICE_TAPE_CHOIR,
    Layer,
    Note,
    Score,
)
from .signature import SigVoice, patch_of, signature, trim_layer
from .styles import BERLIN_SCHOOL
from .timeline import SECONDS_PER_MINUTE

if TYPE_CHECKING:
    from .plan import Plan
    from .quality import Quality
    from .synth import Scale

Cell = tuple[int | None, ...]
# lines of the sequencer: degrees of the scale from the chord root (7 is the octave), None is a rest
CELLS_A: tuple[tuple[str, Cell], ...] = (
    ("octave jumps", (0, 0, 7, 0, 4, 0, 7, 0, 0, 0, 7, 0, 6, 0, 4, 0)),
    ("held root", (0, None, 0, 7, 0, None, 4, 0, 0, None, 0, 7, 6, None, 4, 3)),
    ("arpeggio", (0, 4, 7, 4, 0, 4, 7, 4, 0, 3, 7, 3, 0, 3, 6, 3)),
    ("pedal and fifth", (0, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 4, 0, 3, 0)),
    ("sparse pulse", (0, None, 7, None, 4, None, 7, None, 0, 0, 7, None, 6, None, 4, None)),
    ("climb", (0, 2, 4, 7, 0, 2, 4, 7, 0, 3, 5, 7, 0, 3, 5, 8)),
)
CELLS_B: tuple[tuple[str, Cell], ...] = (
    ("rising third", (4, None, 7, None, 9, None, 7, None, 4, None, 2, None)),
    ("answer", (7, None, None, 4, None, 7, None, 9, None, None, 7, None)),
    ("triad", (0, None, 4, None, 7, None, 4, None, 0, None, 6, None)),
    ("stutter", (4, 4, None, 7, None, 4, 9, None, 7, None, None, 4)),
)
CELLS_C: tuple[tuple[str, Cell], ...] = (
    ("eight", (0, None, None, 4, None, None, 7, None)),
    ("seven", (7, None, 4, None, None, 7, None)),
    ("nine", (0, None, 0, None, 4, None, None, None, 7)),
    ("five", (4, None, 7, None, 9)),
)
# what each cell suits, as the energy of the day (0 quiet, 1 busy): the fewer notes and the more of one pedal tone, the quieter
CELL_ENERGY = {
    "octave jumps": 0.55,
    "held root": 0.45,
    "arpeggio": 0.8,
    "pedal and fifth": 0.15,
    "sparse pulse": 0.05,
    "climb": 0.95,
    "rising third": 0.3,
    "answer": 0.15,
    "triad": 0.55,
    "stutter": 0.9,
    "eight": 0.2,
    "seven": 0.45,
    "nine": 0.7,
    "five": 0.9,
}
# chord roots of the day, as scale degrees, one chord after another; every move is by a fourth or a fifth (3 or 4
# scale degrees), the last chord moves to the first the same way, and the organ holds the tonic as a pedal under them.
# The third item is how much the harmony moves (0 still, 1 restless), which is what the energy of the day asks for
PROGRESSIONS: tuple[Progression, ...] = (
    ("i iv", (0, 3), 0.1),
    ("i v", (0, 4), 0.15),
    ("i iv i v", (0, 3, 0, 4), 0.4),
    ("i v i iv", (0, 4, 0, 3), 0.45),
    ("i iv VII iv", (0, 3, 6, 3), 0.6),
    ("i v ii v", (0, 4, 1, 4), 0.75),
    ("i iv VII III VII iv", (0, 3, 6, 2, 6, 3), 0.9),
)
# the relative key of a mode, as the move in scale degrees that makes the same notes sound from a new tonic
RELATIVE_SHIFT = {"aeolian": 2, "dorian": -1, "phrygian": -2, "mixolydian": 3}
# composed lead phrases: (degree from the chord root, length in beats)
LEAD_PHRASES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("rising", ((4, 8), (5, 4), (7, 12))),
    ("falling", ((7, 8), (6, 4), (4, 12))),
    ("held", ((2, 16), (4, 8))),
    ("turn", ((4, 6), (2, 2), (3, 8), (0, 12))),
    ("sigh", ((7, 12), (4, 4), (5, 12))),
    ("open", ((0, 16), (4, 16))),
)
LAYER_SEQ_B = "seqB"
LAYER_SEQ_C = "seqC"
LAYER_CHOIR = "choir"
LAYER_BED = "bed"
LAYER_PULSE = "pulse"
LAYER_TOMS = "toms"
LAYER_SHAKER = "shaker"
LINE_KEYS = (LAYER_SEQ_A, LAYER_SEQ_B, LAYER_SEQ_C)
VOICE_DRAWBAR_ORGAN = "drawbar_organ"
VOICE_MOOG_SEQ = "moog_seq"
DRUM_KICK_SOFT = "drum:kick_soft"
DRUM_TOM_SOFT = "drum:tom_soft"
DRUM_SHAKER = "drum:shaker"
SECOND_VOICES: dict[str, VoiceTag] = {
    "juno_pluck": (0.35, 0.6),
    VOICE_MOOG_SEQ: (0.6, 0.1),
    "blip_synth": (0.85, -0.5),
}
# the three layers that can lead: the room picks its own, and a happening of the day may hand a phrase to another
LEADS = (LAYER_SEQ_A, LAYER_SEQ_B, LAYER_LEAD)
LEAD_BY_TRAIT = {
    lead.PERCUSSIVE: LAYER_SEQ_A,
    lead.RHYTHMIC: LAYER_SEQ_B,
    lead.SUSTAINED: LAYER_LEAD,
}
# what each layer is trimmed by while it leads, so that every lead sits a little over the others and no more
LEAD_TRIM_DB = {LAYER_SEQ_A: -6.1, LAYER_SEQ_B: -3.5, LAYER_LEAD: -0.4}
# the weights of the rules of the sequencer: a drop needs a dense cell, an add one with rests
DROP_WEIGHT = 0.28
DROP_MIN_DENSITY = 0.4
DROP_SPARSE_WEIGHT = 0.02
ADD_WEIGHT = 0.28
ADD_MAX_DENSITY = 0.85
MOVE_WEIGHT = 0.25
ROTATE_WEIGHT = 0.07
REVERT_WEIGHT = 0.2
ADD_OCTAVE_CHANCE = 0.3
STEPS_PER_BEAT = 4
# the sequencer changes a cell every this many bars, by one of these rules
MUTATE_BARS = 4
RULE_DROP = "drop"
RULE_ADD = "add"
RULE_MOVE = "move"
RULE_ROTATE = "rotate"
RULE_REVERT = "revert"
# a moved note goes this many scale degrees, by these weights
MOVE_STEPS = (1, -1, OCTAVE_DEGREES, -OCTAVE_DEGREES)
MOVE_WEIGHTS = (0.35, 0.35, 0.15, 0.15)
# the pitch (0 lowest) of the toms of a fill, from its first step
TOM_FILL_PITCHES = (2, 1, 1, 0)
# the anomaly swells the strings on these degrees of the chord
SWELL_CHORD = (0, 2, 4)
# the anomaly swell is in this octave, its top tone an octave above
ANOMALY_OCTAVE = 3
# the length in seconds of a cutoff sweep of each sequencer line
SWEEP_PERIODS = (70.0, 110.0, 150.0)
# tiers with this many layers or more play the full set
RICH_MIN_LAYERS = 6
# the second line follows the activity between these, the third one the same on a higher range
DISH_GATE_FROM = 0.3
DISH_GATE_TO = 0.55
WASH_GATE_FROM = 0.55
WASH_GATE_TO = 0.8
LINE_NIGHT_FLOOR = 0.4
LINE_NIGHT_SHARE = 0.6
LINE_NIGHT_DUCK = 0.4
MUTATE_POWER_SHARE = 0.3
MIN_LINE_PRESENCE = 0.06
SEQ_RESONANCE = 0.4
SEQ_REVERB = 0.25
SEQ_ECHO = 0.3
SEQ_A_ECHO = 0.16
SEQ_ACCENT_VELOCITY = 0.62
SEQ_ACCENT_POWER = 0.28
SEQ_VELOCITY_FLOOR = 0.6
SEQ_VELOCITY_SHARE = 0.4
SEQ_VELOCITY_JITTER = 0.03
SEQ_VELOCITY_MIN = 0.2
SEQ_HOLD_SHARE = 0.82
STRINGS_STAGGER = 0.12
CHOIR_STAGGER = 0.2
STRINGS_VELOCITY = 0.42
CHOIR_VELOCITY = 0.3
HOLD_SWELL_VELOCITY = 0.1
ORGAN_OCTAVE = 2
ORGAN_FIFTH_DEGREE = 4
ORGAN_FIFTH_DELAY = 0.3
ORGAN_FIFTH_VELOCITY = 0.36
MOTIF_NOTE_SHARE = 0.95
PHRASE_NOTE_SHARE = 0.95
PHRASE_VELOCITY_HERE = 0.12
LEAD_START_MIN = 12.0
LEAD_START_MAX = 30.0
LEAD_END_MARGIN = 20.0
LEAD_HERE_FLOOR = 0.3
LEAD_HERE_ACTIVITY = 0.7
LEAD_NIGHT_DUCK = 0.6
LEAD_MIN_HERE = 0.22
LEAD_CHANCE_FLOOR = 0.4
LEAD_CHANCE_HERE = 0.6
LEAD_HIGH_CHANCE = 0.3
LEAD_GAP_JITTER_LOW = 0.8
LEAD_GAP_JITTER_HIGH = 1.2
LEAD_GAP_QUIET = 1.2
NIGHT_GATE_FROM = 0.1
NIGHT_GATE_TO = 0.5
EVENING_GATE_FROM = 0.15
EVENING_GATE_TO = 0.5
PULSE_GATE_FROM = 0.5
PULSE_GATE_TO = 0.75
PULSE_POWER_SHARE = 0.8
POWER_GATE_FROM = 0.5
POWER_GATE_TO = 0.8
TOMS_GATE_FROM = 0.62
TOMS_GATE_TO = 0.85
SHAKER_GATE_FROM = 0.72
SHAKER_GATE_TO = 0.9
PULSE_MIN_LEVEL = 0.1
PULSE_BEAT_LEVEL = 0.45
PULSE_HOLD = 0.3
PULSE_VELOCITY = 0.28
PULSE_VELOCITY_LEVEL = 0.2
TOM_MIN_LEVEL = 0.1
TOM_CHANCE = 0.45
TOM_VELOCITY = 0.36
TOM_VELOCITY_LEVEL = 0.2
TOM_VELOCITY_JITTER = 0.03
TOM_HOLD = 0.3
TOM_FILL_CYCLE = 8
TOM_FILL_BAR = 7
TOM_FILL_FROM_STEP = 12
TOM_FILL_MIN_LEVEL = 0.35
TOM_FILL_CHANCE = 0.8
TOM_FILL_VELOCITY = 0.4
TOM_FILL_VELOCITY_LEVEL = 0.2
SHAKER_MIN_LEVEL = 0.1
SHAKER_EIGHTH_CHANCE = 0.9
SHAKER_ACCENT = 0.4
SHAKER_VELOCITY = 0.26
SHAKER_VELOCITY_FLOOR = 0.6
SHAKER_VELOCITY_LEVEL = 0.4
SHAKER_HOLD = 0.1
SHAKER_SIXTEENTH_MIN_LEVEL = 0.6
SHAKER_SIXTEENTH_CHANCE = 0.35
SHAKER_GHOST_VELOCITY = 0.2
GROWL_POWER_FROM = 0.45
GROWL_POWER_TO = 0.8
GROWL_NIGHT_FROM = 0.25
GROWL_NIGHT_TO = 0.5
GROWL_NIGHT_SHARE = 0.7
GROWL_MIN_GATE = 0.2
GROWL_GAP = 0.1
GROWL_VELOCITY = 0.6
SWEEP_RISE = 4.0
SWEEP_FALL = 9.0
ANOMALY_STAGGER = 0.3
ANOMALY_HOLD = 14.0
ANOMALY_VELOCITY = 0.5
ANOMALY_VELOCITY_STRENGTH = 0.3
CUT_WALK_SECONDS = 40.0
CUT_WALK_RANGE = 0.12
SWEEP_PERIOD_JITTER = 0.2
CUT_FLOOR = 0.1
CUT_BRIGHT = 0.25
CUT_POWER = 0.25
CUT_TRIANGLE = 0.3
SEQ_CUTOFF_BASE = 380.0
SEQ_CUTOFF_RANGE = 16.0
SEQ_DIP_DEPTH = 0.8
DIP_HOLD = 0.9
DIP_BACK = 4.0
BED_GAIN = 0.9
BED_GATE_FROM = 0.1
BED_GATE_TO = 0.4
BED_CUTOFF_BASE = 380.0
BED_CUTOFF_RANGE = 12.0
BED_CUTOFF_FLOOR = 0.3
BED_CUTOFF_BRIGHT = 0.7
BED_DIP_DEPTH = 0.6
BED_RESONANCE = 0.1
BED_REVERB = 0.7
CHOIR_GAIN = 0.8
CHOIR_GATE_FROM = 0.35
CHOIR_GATE_TO = 0.7
CHOIR_REVERB = 0.8
STRINGS_GAIN = 0.8
STRINGS_PRESENCE_FLOOR = 0.45
STRINGS_PRESENCE_POWER = 0.4
STRINGS_REVERB = 0.3
LEAD_GAIN = 0.9
LEAD_REVERB = 0.5
LEAD_ECHO = 0.55
PULSE_REVERB = 0.1
TOMS_GAIN = 0.7
TOMS_CUTOFF = 1800.0
TOMS_REVERB = 0.25
SHAKER_CUTOFF = 6500.0
SHAKER_REVERB = 0.2
GROWL_GAIN = 0.7
GROWL_REVERB = 0.15
SPACE_BASE = 0.3
SPACE_HUMID = 0.35
SPACE_SWEEP = 0.2
SPACE_MAX = 0.95
SHIMMER = 0.1
LEVEL_DB = -19.6
# a sequence that leads plays every step of its cell, at least this present, and its pitches follow the motif of the
# room (developed, at most this long)
LEAD_LINE_PRESENCE = 0.8
LEAD_LINE_BEATS = 16.0
# the lead line is Sequence A: it sits above the others; a blip as the second line sits behind it, and darker
SEQ_A_GAIN = 1.4
BLIP_VOICE = "blip_synth"
BLIP_GAIN = 0.28
BLIP_CUTOFF_SCALE = 0.45
# the soft kit sits a few dB under the sequence, so that its pulse is felt where it plays
PULSE_GAIN = 1.25
SHAKER_GAIN = 1.05
STRINGS: dict[str, VoiceTag] = {"tape_strings": (0.3, 0.5), "string_machine": (0.7, -0.3)}
# warm electronic toms on the 16th grid: (step of the bar, which tom: 0 is the lowest of three), and the energy each suits
TomFigure = tuple[str, float, tuple[tuple[int, int], ...]]
TOM_FIGURES: tuple[TomFigure, ...] = (
    ("sparse", 0.3, ((10, 1), (14, 0))),
    ("walking", 0.6, ((6, 2), (10, 1), (14, 0))),
    ("busy", 0.9, ((3, 2), (7, 1), (11, 0), (14, 1))),
)
# the kit of the later Berlin School, as (name, energy it suits, toms, shaker)
Kit = tuple[str, float, bool, bool]
KITS: tuple[Kit, ...] = (
    ("no kit", 0.1, False, False),
    ("kick", 0.4, False, False),
    ("kick and toms", 0.65, True, False),
    ("kick, toms and shaker", 0.9, True, True),
)
TEMPO_LOW, TEMPO_HIGH = BERLIN_SCHOOL.tempo
TEMPO_STEP = 2.0
TEMPOS = tuple(
    float(TEMPO_LOW + TEMPO_STEP * index)
    for index in range(int((TEMPO_HIGH - TEMPO_LOW) / TEMPO_STEP) + 1)
)
# the strings and the choir: voices as the degree of the chord each plays, and the octave each starts in over the lowest
# (an open voicing), inside a window of two octaves
STRINGS_OFFSETS = (0, 2, 4, 6)
STRINGS_LIFTS = (0, 0, 7, 7)
STRINGS_LOW = 0
CHOIR_OFFSETS = (0, 4, 0)
CHOIR_LIFTS = (0, 0, 7)
CHOIR_LOW = 7
VOICING_OCTAVE = 3
HOLD_TAIL = 3.0
# the growl holds the tonic in this octave, which keeps it under C3
GROWL_OCTAVE = 2
MOTIF_VELOCITY = 0.62
LEAD_MIN_INTENSITY = 0.3
# bars between two lead phrases on a quiet and on a busy day
LEAD_GAP_BARS = (12.0, 3.0)
# the drums come in at an activity that the energy of the day moves by up to this much
DRUM_THRESHOLD_SHIFT = 0.3
# the drums come in and the layers join by the intensity of the section of the form
PULSE_MIN_INTENSITY = 0.45
TOMS_MIN_INTENSITY = 0.6
SHAKER_MIN_INTENSITY = 0.75
GROWL_MIN_INTENSITY = 0.55
CHOIR_MIN_INTENSITY = 0.3
SECOND_LINE_MIN_INTENSITY = 0.45
THIRD_LINE_MIN_INTENSITY = 0.7
# the share of the strings, the organ and the first line that stays under the intensity of the form
FORM_FLOOR = 0.55
# how far the second and third lines sound at their least, on a quiet and on a busy day
SECOND_LINE_FLOOR = (0.0, 0.9)
THIRD_LINE_FLOOR = (0.0, 0.5)
# the chord the sequencer draws from when a rule adds a step, as degrees from the root
SEQUENCER_CHORD_TONES = (0, 2, 4, 6)
# the chance in a step of four bars that a line mutates, on a quiet and on a busy day
MUTATE_RATE = (0.04, 0.9)


# the least presence of the choir, the base of the piece
CHOIR_FLOOR = 0.5
# below this strength of its gate a line plays on every second step only
THIN_BELOW = 0.85


def compose(plan: Plan) -> Score:
    """
    Make the score of a Berlin School soundscape.

    :param plan: The curves and the deals of the day.
    """
    piece = _Piece(plan)
    _pick_tempo_and_cells(piece)
    _pick_voices_and_kit(piece)
    _lay_out_form(piece)
    _plan_progression(piece)
    _play_sequencer(piece)
    _hold_chords(piece)
    _play_organ_bed(piece)
    _state_motif(piece)
    _play_slow_lead(piece)
    _play_lead_phrases(piece)
    _play_drums(piece)
    _play_growl(piece)
    _play_anomaly_swells(piece)
    lead.back_off_followers(piece, frozenset(LEADS), "lead")
    _sweep_cutoffs(piece)
    _add_support_layers(piece)
    _add_drum_layers(piece)
    return _assemble(piece)


def _mutate(
    rng: np.random.Generator,
    cell: list[int | None],
    base: Cell,
    rate: float,
    chord: tuple[int, ...],
) -> str:
    """Apply one rule of the sequencer to a cell, with the chance `rate`; return what it did ('' for nothing)."""
    if rng.random() > rate:
        return ""
    steps = len(cell)
    notes = [i for i, v in enumerate(cell) if v is not None]
    rests = [i for i, v in enumerate(cell) if v is None]
    density = len(notes) / steps
    rule = weighted(
        rng,
        (RULE_DROP, RULE_ADD, RULE_MOVE, RULE_ROTATE, RULE_REVERT),
        (
            DROP_WEIGHT if density > DROP_MIN_DENSITY else DROP_SPARSE_WEIGHT,
            ADD_WEIGHT if density < ADD_MAX_DENSITY and rests else 0.0,
            MOVE_WEIGHT,
            ROTATE_WEIGHT,
            REVERT_WEIGHT,
        ),
    )
    if rule == RULE_DROP and len(notes) > 2:
        cell[int(rng.choice([i for i in notes if i != 0] or notes))] = None
    elif rule == RULE_ADD and rests:
        cell[int(rng.choice(rests))] = int(rng.choice(chord)) + OCTAVE_DEGREES * int(
            rng.random() < ADD_OCTAVE_CHANCE
        )
    elif rule == RULE_MOVE and notes:
        index = int(rng.choice(notes))
        value = cell[index]
        if value is not None:
            cell[index] = value + int(weighted(rng, MOVE_STEPS, MOVE_WEIGHTS))
    elif rule == RULE_ROTATE:
        cell.insert(0, cell.pop())
    else:
        diff = [i for i in range(steps) if cell[i] != base[i]]
        if not diff:
            return ""
        i = int(rng.choice(diff))
        cell[i] = base[i]
        rule = RULE_REVERT
    return rule


@dataclass
class _Piece:
    """The working state of one Berlin School piece, read and appended to by the sections of `compose`."""

    plan: Plan
    scale: Scale = field(init=False)
    quality: Quality = field(init=False)
    length: int = field(init=False)
    energy: float = field(init=False)
    rng: np.random.Generator = field(init=False)
    bpm: float = field(init=False)
    step: float = field(init=False)
    bar: float = field(init=False)
    phrase: float = field(init=False)
    beat: float = field(init=False)
    name_a: str = field(init=False)
    cell_a: Cell = field(init=False)
    name_b: str = field(init=False)
    cell_b: Cell = field(init=False)
    name_c: str = field(init=False)
    cell_c: Cell = field(init=False)
    seq_sig: SigVoice = field(init=False)
    lead_sig: SigVoice = field(init=False)
    voice_a: str = field(init=False)
    voice_b: str = field(init=False)
    lead_voice: str = field(init=False)
    strings: str = field(init=False)
    growl: str = field(init=False)
    chord_len_bars: int = field(init=False)
    prog_name: str = field(init=False)
    progression: tuple[int, ...] = field(init=False)
    kit_name: str = field(init=False)
    toms_on: bool = field(init=False)
    shaker_on: bool = field(init=False)
    tom_name: str = field(init=False)
    tom_figure: tuple[tuple[int, int], ...] = field(init=False)
    patch: int = field(init=False)
    seq_patch: int = field(init=False)
    lead_patch: int = field(init=False)
    lines: int = field(init=False)
    light_tier: bool = field(init=False)
    spans: list[tuple[float, float, str, float]] = field(init=False)
    intensity: np.ndarray = field(init=False)
    lead_grid: float = field(init=False)
    default_lead: str = field(init=False)
    leads: lead.Leads = field(init=False)
    curves: dict[str, np.ndarray] = field(init=False)
    changes: list[tuple[float, int]] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    layers: dict[str, Layer] = field(default_factory=dict)
    mutations: dict[str, int] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    statements: list[float] = field(default_factory=list)
    taken: list[tuple[float, float]] = field(default_factory=list)
    phrases_played: list[str] = field(default_factory=list)
    sweep: np.ndarray = field(init=False)
    growl_gate: np.ndarray = field(init=False)
    pulse_gate: np.ndarray = field(init=False)
    tom_gate: np.ndarray = field(init=False)
    shaker_gate: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Take what the plan already says; the first draw of the piece comes after it."""
        self.scale = self.plan.scale
        self.quality = self.plan.quality
        self.length = self.plan.length
        self.energy = self.plan.character.energy
        self.rng = self.plan.rng("berlin")


def _pick_tempo_and_cells(piece: _Piece) -> None:
    """Pick the tempo, which sets the grid, and the cells of the three sequencer lines."""
    plan = piece.plan
    energy = piece.energy
    piece.bpm = plan.pick(
        "tempo",
        TEMPOS,
        [
            1.0
            - abs(
                (t - TEMPO_LOW) / (TEMPO_HIGH - TEMPO_LOW)
                - tempo_target(BERLIN_SCHOOL, plan.character)
            )
            for t in TEMPOS
        ],
        tolerance=TEMPO_TOLERANCE,
    )
    piece.step = SECONDS_PER_MINUTE / piece.bpm / STEPS_PER_BEAT
    piece.bar = piece.step * STEPS_PER_BAR
    piece.phrase = BARS_PER_PHRASE * piece.bar
    piece.beat = SECONDS_PER_MINUTE / piece.bpm
    piece.name_a, piece.cell_a = plan.pick(
        "sequence A", CELLS_A, [closeness(CELL_ENERGY[n], energy) for n, _ in CELLS_A]
    )
    piece.name_b, piece.cell_b = plan.pick(
        "sequence B", CELLS_B, [closeness(CELL_ENERGY[n], energy) for n, _ in CELLS_B]
    )
    piece.name_c, piece.cell_c = plan.pick(
        "sequence C", CELLS_C, [closeness(CELL_ENERGY[n], energy) for n, _ in CELLS_C]
    )


def _pick_voices_and_kit(piece: _Piece) -> None:
    """Pick the voices, the progression, the kit and the patch of the day."""
    plan = piece.plan
    energy = piece.energy
    piece.seq_sig = signature(plan, LAYER_SEQ_A, "sequence voice")
    piece.lead_sig = signature(plan, LAYER_LEAD, "lead voice")
    piece.voice_a, piece.lead_voice = piece.seq_sig.voice, piece.lead_sig.voice
    piece.voice_b = pick_voice(plan, "second voice", SECOND_VOICES)
    piece.strings = pick_voice(plan, "strings", STRINGS)
    piece.growl = pick_voice(plan, "growl bass", GROWLS)
    piece.chord_len_bars = harmonic_rhythm(plan)
    candidates = progressions_for(piece.scale.mode, PROGRESSIONS)
    piece.prog_name, piece.progression, _ = plan.pick(
        "progression",
        candidates,
        [closeness(drive, energy) for _, _, drive in candidates],
    )
    piece.kit_name, _, piece.toms_on, piece.shaker_on = plan.pick(
        "kit", KITS, [closeness(tag, energy) for _, tag, _, _ in KITS]
    )
    piece.tom_name, _, piece.tom_figure = plan.pick(
        "tom figure", TOM_FIGURES, [closeness(tag, energy) for _, tag, _ in TOM_FIGURES]
    )
    piece.patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(piece.patch)
    piece.seq_patch = patch_of(piece.seq_sig, piece.patch)
    piece.lead_patch = patch_of(piece.lead_sig, piece.patch)


def _lay_out_form(piece: _Piece) -> None:
    """Set the number of lines, the form of the piece and who leads each phrase."""
    plan = piece.plan
    piece.lines = 3 if piece.quality.max_layers >= RICH_MIN_LAYERS else 2
    piece.light_tier = piece.quality.max_layers <= LIGHT_MAX_LAYERS
    piece.spans = plan.sections(piece.bar)
    piece.intensity = section_curve(plan, piece.spans, lambda _role, level: level, piece.bar)

    piece.lead_grid = plan.form_grid(piece.bar)
    piece.default_lead = lead.default_lead(plan, LEADS)
    piece.leads = lead.phrase_leads(
        plan,
        piece.lead_grid,
        piece.default_lead,
        lead.daytime_phrases(plan, piece.spans, piece.lead_grid),
        LEAD_BY_TRAIT,
    )
    piece.curves = lead.lead_curves(plan, piece.leads, LEADS)


def _role_gate(piece: _Piece, minimum: float) -> np.ndarray:
    """Return the curve that is 1 where the intensity of the form reaches `minimum`."""
    return section_curve(
        piece.plan, piece.spans, lambda _role, level: float(level >= minimum), piece.bar
    )


def _plan_progression(piece: _Piece) -> None:
    """Set a chord root every few bars; the strongest anomaly moves the next phrase to the relative key."""
    plan = piece.plan
    chord_len = piece.chord_len_bars * piece.bar
    relative_phrase = -1
    if plan.events:
        strongest = max(plan.events, key=lambda event: event[1])[0]
        relative_phrase = math.ceil(strongest / piece.phrase)
    shift = RELATIVE_SHIFT.get(piece.scale.mode, 0)
    index = 0
    while index * chord_len < piece.length:
        in_relative = index * piece.chord_len_bars // BARS_PER_PHRASE == relative_phrase
        root = piece.progression[index % len(piece.progression)] + (shift if in_relative else 0)
        piece.changes.append((index * chord_len, root))
        index += 1
    piece.starts = [start for start, _ in piece.changes]


def _root_at(piece: _Piece, second: float) -> int:
    """Return the chord root that sounds at a second."""
    return piece.changes[max(bisect.bisect_right(piece.starts, second) - 1, 0)][1]


def _play_sequencer(piece: _Piece) -> None:
    """Play the sequencer lines, each on its own cell and with its own mutations."""
    lines_spec = [
        (LAYER_SEQ_A, piece.cell_a, piece.voice_a, 3),
        (LAYER_SEQ_B, piece.cell_b, piece.voice_b, 4),
        (LAYER_SEQ_C, piece.cell_c, VOICE_MOOG_SEQ, 4),
    ][: piece.lines]
    for line_key, base, voice, octave in lines_spec:
        _play_line(piece, line_key, base, voice, octave)


def _line_presence(piece: _Piece, line_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return how present a line is over the piece, and the same without the pull of the form."""
    plan = piece.plan
    energy = piece.energy
    presence = np.ones(plan.count) if line_key == LAYER_SEQ_A else None
    if line_key == LAYER_SEQ_B:
        second_gate = np.maximum(_role_gate(piece, SECOND_LINE_MIN_INTENSITY), plan.runs[RUN_DISH])
        # the second line is part of the base on a busy day, and comes with the busy stretches on a quiet one; the
        # cheapest tier has no floor under it, since the line is the dearest thing that tier plays
        floor = (
            0.0
            if piece.light_tier
            else SECOND_LINE_FLOOR[0] + (SECOND_LINE_FLOOR[1] - SECOND_LINE_FLOOR[0]) * energy
        )
        presence = second_gate * plan.lift(
            np.maximum(plan.gate(plan.activity, DISH_GATE_FROM, DISH_GATE_TO), plan.runs[RUN_DISH]),
            floor,
        )
    if line_key == LAYER_SEQ_C:
        third_gate = np.maximum(_role_gate(piece, THIRD_LINE_MIN_INTENSITY), plan.runs[RUN_WASH])
        floor = THIRD_LINE_FLOOR[0] + (THIRD_LINE_FLOOR[1] - THIRD_LINE_FLOOR[0]) * energy
        presence = third_gate * plan.lift(
            np.maximum(plan.gate(plan.activity, WASH_GATE_FROM, WASH_GATE_TO), plan.runs[RUN_WASH]),
            floor,
        )
    assert presence is not None
    # a line that is only at its floor plays on every second step: the base is a lighter line, not a missing one
    calm_line = presence if line_key != LAYER_SEQ_A else np.ones(plan.count)
    presence = (
        presence * (LINE_NIGHT_FLOOR + LINE_NIGHT_SHARE * (1.0 - plan.night))
        if line_key != LAYER_SEQ_A
        else (0.5 + 0.5 * (1.0 - LINE_NIGHT_DUCK * plan.night))
        * (FORM_FLOOR + (1.0 - FORM_FLOOR) * piece.intensity)
    )
    if line_key in piece.curves:
        raised = LEAD_LINE_PRESENCE * np.clip(piece.curves[line_key], 0.0, 1.0)
        presence, calm_line = np.maximum(presence, raised), np.maximum(calm_line, raised)
    return presence, calm_line


def _mutate_rate(piece: _Piece, second: float) -> float:
    """Return the chance that a line mutates in the step of four bars at a second."""
    plan = piece.plan
    local = min(
        plan.at(plan.activity, second) + MUTATE_POWER_SHARE * plan.at(plan.power, second), 1.0
    )
    return MUTATE_RATE[0] + (MUTATE_RATE[1] - MUTATE_RATE[0]) * (0.5 * piece.energy + 0.5 * local)


def _play_line(piece: _Piece, line_key: str, base: Cell, voice: str, octave: int) -> None:
    """Play one line of the sequencer over the whole piece and add its layer."""
    plan = piece.plan
    cell: list[int | None] = list(base)
    presence, calm_line = _line_presence(piece, line_key)
    total_steps = int(piece.length / piece.step)
    done = 0
    for global_step in range(total_steps):
        second = global_step * piece.step
        if global_step % (STEPS_PER_BAR * MUTATE_BARS) == 0 and global_step > 0:
            rate = _mutate_rate(piece, second)
            if _mutate(piece.rng, cell, base, rate, SEQUENCER_CHORD_TONES):
                done += 1
        position = global_step % len(cell)
        value = cell[position]
        if value is None:
            continue
        here = plan.at(presence, second)
        if here < MIN_LINE_PRESENCE:
            continue
        if (
            line_key != LAYER_SEQ_A
            and global_step % 2 == 1
            and plan.at(calm_line, second) < THIN_BELOW
        ):
            continue
        piece.notes.append(
            _sequence_note(piece, line_key, voice, octave, second, position, value, here)
        )
    piece.mutations[line_key] = done
    behind_blip = line_key == LAYER_SEQ_B and voice == BLIP_VOICE
    piece.layers[line_key] = Layer(
        line_key,
        SEQ_A_GAIN if line_key == LAYER_SEQ_A else BLIP_GAIN if behind_blip else 1.0,
        presence,
        None,
        SEQ_RESONANCE,
        reverb=SEQ_REVERB,
        echo=SEQ_ECHO if line_key != LAYER_SEQ_A else SEQ_A_ECHO,
        title=f"sequence {line_key[-1]}",
    )


def _sequence_note(
    piece: _Piece,
    line_key: str,
    voice: str,
    octave: int,
    second: float,
    position: int,
    value: int,
    here: float,
) -> Note:
    """Return the note of a line at a step, accented on the beat of its cell."""
    plan = piece.plan
    accent = position % 4 == 0
    vel = (
        SEQ_ACCENT_VELOCITY + SEQ_ACCENT_POWER * plan.at(plan.power, second) if accent else 0.5
    ) * (SEQ_VELOCITY_FLOOR + SEQ_VELOCITY_SHARE * here)
    vel = float(
        np.clip(vel + float(piece.rng.normal(0.0, SEQ_VELOCITY_JITTER)), SEQ_VELOCITY_MIN, 1.0)
    )
    degree = value + _root_at(piece, second)
    note_octave = octave
    if piece.leads.at(second) == line_key:
        # the line that leads keeps the rhythm of its cell and takes the pitches of the motif
        number = int(second / piece.lead_grid)
        motif = develop_motif(plan.motif(), number, motif_seed(plan), LEAD_LINE_BEATS)
        piece.counts[number] = piece.counts.get(number, -1) + 1
        degree = motif[piece.counts[number] % len(motif)][0] + _root_at(piece, second)
        note_octave = lead_octave(piece.scale, motif)
    return Note(
        second,
        piece.step * SEQ_HOLD_SHARE,
        float(piece.scale.midi(degree, note_octave)),
        vel,
        voice,
        line_key,
        piece.seq_patch if line_key == LAYER_SEQ_A else piece.patch,
    )


def _hold_chords(piece: _Piece) -> None:
    """Strings and choir hold the chord and are led from chord to chord: a shared tone stays, the others move by step."""
    plan = piece.plan
    roots = [root for _, root in piece.changes]
    ends = [*piece.starts[1:], float(piece.length)]
    holds = [
        (
            LAYER_STRINGS,
            piece.strings,
            STRINGS_OFFSETS,
            STRINGS_LIFTS,
            STRINGS_LOW,
            STRINGS_STAGGER,
            True,
        ),
        (
            LAYER_CHOIR,
            VOICE_TAPE_CHOIR,
            CHOIR_OFFSETS,
            CHOIR_LIFTS,
            CHOIR_LOW,
            CHOIR_STAGGER,
            piece.quality.max_layers > LIGHT_MAX_LAYERS,
        ),
    ]
    for layer_key, voice, offsets, lifts, low, stagger, on in holds:
        if not on:
            continue
        voicing = voice_lead(roots, offsets, lifts, low)
        for k, start, end, degree in held_runs(
            voicing,
            piece.starts,
            ends,
            lambda at: round(at / piece.bar) % BARS_PER_PHRASE == 0,
        ):
            swell = (
                plan.at(plan.activity, start)
                if layer_key == LAYER_STRINGS
                else max(plan.at(plan.lights, start), plan.at(plan.evening, start))
            )
            piece.notes.append(
                Note(
                    start + stagger * k,
                    end - start + HOLD_TAIL,
                    float(piece.scale.midi(degree, VOICING_OCTAVE)),
                    (STRINGS_VELOCITY if layer_key == LAYER_STRINGS else CHOIR_VELOCITY)
                    + HOLD_SWELL_VELOCITY * swell,
                    voice,
                    layer_key,
                    piece.patch,
                )
            )


def _play_organ_bed(piece: _Piece) -> None:
    """Play the organ bed: the tonic and the fifth as a pedal under the moving chords, long."""
    second = 0.0
    while second < piece.length:
        piece.notes.append(
            Note(
                second,
                piece.phrase + 2.0,
                float(piece.scale.midi(0, ORGAN_OCTAVE)),
                0.5,
                VOICE_DRAWBAR_ORGAN,
                LAYER_BED,
                piece.patch,
            )
        )
        piece.notes.append(
            Note(
                second + ORGAN_FIFTH_DELAY,
                piece.phrase + 2.0,
                float(piece.scale.midi(ORGAN_FIFTH_DEGREE, ORGAN_OCTAVE)),
                ORGAN_FIFTH_VELOCITY,
                VOICE_DRAWBAR_ORGAN,
                LAYER_BED,
                piece.patch,
            )
        )
        second += piece.phrase


def _state_motif(piece: _Piece) -> None:
    """
    Play the room's motif in the lead voice, in the first and the last peak or plateau of the form.

    The later statement is developed; the slow phrases leave room for it.
    """
    plan = piece.plan
    spans = piece.spans
    sites = [i for i, (_, _, role, _) in enumerate(spans) if role in {CLIMAX, PLATEAU}]
    sites = motif_sites(sites or [max(range(len(spans)), key=lambda i: spans[i][3])])
    for number, i in enumerate(sites):
        start, end, _, _ = spans[i]
        motif = develop_motif(plan.motif(), number, motif_seed(plan))
        motif_octave = lead_octave(piece.scale, motif)
        motif_seconds = sum(beats for _, beats in motif) * piece.beat
        at = min(start + piece.bar, max(end - motif_seconds - piece.bar, 0.0))
        piece.statements.append(at)
        piece.taken.append((at - piece.bar, at + motif_seconds + piece.bar))
        clock = at
        for degree, beats in motif:
            span = beats * piece.beat
            piece.notes.append(
                Note(
                    clock,
                    span * MOTIF_NOTE_SHARE,
                    float(piece.scale.midi(degree, motif_octave)),
                    MOTIF_VELOCITY,
                    piece.lead_voice,
                    LAYER_LEAD,
                    piece.lead_patch,
                )
            )
            clock += span

    plan.choices[MOTIF_CHOICE] = str(len(piece.statements))


def _lay_phrase(
    piece: _Piece, clock: float, phrase_notes: tuple[tuple[int, int], ...], octave: int, here: float
) -> float:
    """Put a composed phrase of the slow lead from `clock` on; return where it ends."""
    for degree, beats in phrase_notes:
        span = beats * piece.beat
        piece.notes.append(
            Note(
                clock,
                span * PHRASE_NOTE_SHARE,
                float(piece.scale.midi(degree + _root_at(piece, clock), octave)),
                0.5 + PHRASE_VELOCITY_HERE * here,
                piece.lead_voice,
                LAYER_LEAD,
                piece.lead_patch,
            )
        )
        clock += span
    return clock


def _play_slow_lead(piece: _Piece) -> None:
    """Play the slow lead: composed phrases of long notes, on the bar, further apart the quieter the day."""
    plan = piece.plan
    rng = piece.rng
    bar = piece.bar
    lead_gate = _role_gate(piece, LEAD_MIN_INTENSITY)
    gap_bars = LEAD_GAP_BARS[0] + (LEAD_GAP_BARS[1] - LEAD_GAP_BARS[0]) * piece.energy
    second = round(float(rng.uniform(LEAD_START_MIN, LEAD_START_MAX)) / bar) * bar
    while second < piece.length - LEAD_END_MARGIN:
        here = (LEAD_HERE_FLOOR + LEAD_HERE_ACTIVITY * plan.at(plan.activity, second)) * (
            1.0 - LEAD_NIGHT_DUCK * plan.at(plan.night, second)
        )
        if (
            plan.at(lead_gate, second) > 0.5
            and here > LEAD_MIN_HERE
            and rng.random() < LEAD_CHANCE_FLOOR + LEAD_CHANCE_HERE * here
        ):
            name, lead_phrase = weighted(rng, LEAD_PHRASES, (1.0,) * len(LEAD_PHRASES))
            span_total = sum(beats for _, beats in lead_phrase) * piece.beat
            clash = [
                end for begin, end in piece.taken if begin < second + span_total and second < end
            ]
            if clash:
                second = max(clash)
                continue
            piece.phrases_played.append(name)
            clock = _lay_phrase(
                piece, second, lead_phrase, LEAD_OCTAVE + int(rng.random() < LEAD_HIGH_CHANCE), here
            )
            second = (
                round(
                    (
                        clock
                        + gap_bars
                        * bar
                        * float(rng.uniform(LEAD_GAP_JITTER_LOW, LEAD_GAP_JITTER_HIGH))
                        * (LEAD_GAP_QUIET - here)
                    )
                    / bar
                )
                * bar
            )
        else:
            second += bar * 2


def _play_lead_phrases(piece: _Piece) -> None:
    """When the slow lead leads, every phrase it has gets a composed phrase of its own."""
    lead_rng = piece.plan.rng("lead phrases")
    for number, who in enumerate(piece.leads.phrases):
        begin = number * piece.lead_grid
        if who == LAYER_LEAD and not any(
            n.layer == LAYER_LEAD and begin <= n.t < begin + piece.lead_grid for n in piece.notes
        ):
            name, lead_phrase = weighted(lead_rng, LEAD_PHRASES, (1.0,) * len(LEAD_PHRASES))
            piece.phrases_played.append(name)
            _lay_phrase(piece, begin, lead_phrase, LEAD_OCTAVE, 1.0)


def _play_drums(piece: _Piece) -> None:
    """
    Play the drums of the later Berlin School (Moondawn, Force Majeure): sparse and under the sequencer.

    They are locked to its 16th grid. The kit is the day's: none on a quiet one, a kick, then toms and a shaker as the
    day gets busier. They play in the busy stretches of the morning and the day only, never in the night, and are gone
    before the evening.
    """
    drum_rng = piece.plan.rng("drums")
    _set_drum_gates(piece)
    _play_pulse(piece)
    if piece.quality.max_layers >= RICH_MIN_LAYERS:
        _play_kit_grid(piece, drum_rng)


def _set_drum_gates(piece: _Piece) -> None:
    """Set the curves on which the kick, the toms and the shaker come in."""
    plan = piece.plan
    activity = plan.activity
    window = (1.0 - plan.gate(plan.night, NIGHT_GATE_FROM, NIGHT_GATE_TO)) * (
        1.0 - plan.gate(plan.evening, EVENING_GATE_FROM, EVENING_GATE_TO)
    )
    # a busy day opens the drums at a lower activity, a quiet one only at a higher
    lower = DRUM_THRESHOLD_SHIFT * (0.5 - piece.energy)
    has_kick = piece.kit_name != KITS[0][0]
    pulse_gate = np.maximum(
        plan.gate(activity, PULSE_GATE_FROM + lower, PULSE_GATE_TO + lower),
        PULSE_POWER_SHARE * plan.gate(plan.power, POWER_GATE_FROM, POWER_GATE_TO),
    )
    pulse_gate = pulse_gate * window * _role_gate(piece, PULSE_MIN_INTENSITY)
    piece.pulse_gate = (
        pulse_gate if has_kick and piece.quality.max_layers > LIGHT_MAX_LAYERS else pulse_gate * 0.0
    )
    piece.tom_gate = (
        plan.gate(activity, TOMS_GATE_FROM + lower, TOMS_GATE_TO + lower)
        * window
        * _role_gate(piece, TOMS_MIN_INTENSITY)
        * float(piece.toms_on)
    )
    piece.shaker_gate = (
        plan.gate(activity, SHAKER_GATE_FROM + lower, SHAKER_GATE_TO + lower)
        * window
        * _role_gate(piece, SHAKER_MIN_INTENSITY)
        * float(piece.shaker_on)
    )
    if piece.quality.max_layers < RICH_MIN_LAYERS:
        piece.tom_gate, piece.shaker_gate = piece.tom_gate * 0.0, piece.shaker_gate * 0.0


def _play_pulse(piece: _Piece) -> None:
    """Play the kick as a steady quarter-note pulse."""
    plan = piece.plan
    second = 0.0
    beat_index = 0
    while second < piece.length:
        level = plan.at(piece.pulse_gate, second)
        # a steady quarter-note pulse; while the house is only starting to get busy, every other beat
        if level > PULSE_MIN_LEVEL and (level > PULSE_BEAT_LEVEL or beat_index % 2 == 0):
            piece.notes.append(
                Note(
                    second,
                    PULSE_HOLD,
                    0.0,
                    PULSE_VELOCITY + PULSE_VELOCITY_LEVEL * level,
                    DRUM_KICK_SOFT,
                    "pulse",
                    piece.patch,
                )
            )
        second += piece.beat
        beat_index += 1


def _play_kit_grid(piece: _Piece, drum_rng: np.random.Generator) -> None:
    """Play the toms and the shaker on the 16th grid of every bar."""
    total_bars = int(piece.length / piece.bar)
    for bar_index in range(total_bars):
        for step_index in range(STEPS_PER_BAR):
            at = bar_index * piece.bar + step_index * piece.step
            _play_toms(piece, drum_rng, at, bar_index, step_index)
            _play_shaker(piece, drum_rng, at, step_index)


def _play_toms(
    piece: _Piece, drum_rng: np.random.Generator, at: float, bar_index: int, step_index: int
) -> None:
    """Play the toms of a step: the figure of the day, and a fill in the last bar of every eight."""
    level = piece.plan.at(piece.tom_gate, at)
    if level > TOM_MIN_LEVEL and drum_rng.random() < TOM_CHANCE + 0.5 * level:
        for hit_step, pitch in piece.tom_figure:
            if hit_step == step_index:
                vel = (
                    TOM_VELOCITY
                    + TOM_VELOCITY_LEVEL * level
                    + float(drum_rng.normal(0.0, TOM_VELOCITY_JITTER))
                )
                piece.notes.append(Note(at, TOM_HOLD, 0.0, vel, DRUM_TOM_SOFT, "toms", pitch))
    if (
        bar_index % TOM_FILL_CYCLE == TOM_FILL_BAR
        and step_index >= TOM_FILL_FROM_STEP
        and level > TOM_FILL_MIN_LEVEL
        and drum_rng.random() < TOM_FILL_CHANCE
    ):
        vel = (
            TOM_FILL_VELOCITY
            + TOM_FILL_VELOCITY_LEVEL * level
            + float(drum_rng.normal(0.0, TOM_VELOCITY_JITTER))
        )
        piece.notes.append(
            Note(
                at,
                TOM_HOLD,
                0.0,
                vel,
                DRUM_TOM_SOFT,
                "toms",
                TOM_FILL_PITCHES[step_index - TOM_FILL_FROM_STEP],
            )
        )


def _play_shaker(piece: _Piece, drum_rng: np.random.Generator, at: float, step_index: int) -> None:
    """Play the shaker of a step: the even steps, and the odd ones when the house is busy."""
    level = piece.plan.at(piece.shaker_gate, at)
    if level > SHAKER_MIN_LEVEL:
        if step_index % 2 == 0 and drum_rng.random() < SHAKER_EIGHTH_CHANCE:
            vel = (SHAKER_ACCENT if step_index % 4 == 2 else SHAKER_VELOCITY) * (
                SHAKER_VELOCITY_FLOOR + SHAKER_VELOCITY_LEVEL * level
            )
            piece.notes.append(Note(at, SHAKER_HOLD, 0.0, vel, DRUM_SHAKER, "shaker", 1))
        elif (
            step_index % 2 == 1
            and level > SHAKER_SIXTEENTH_MIN_LEVEL
            and drum_rng.random() < SHAKER_SIXTEENTH_CHANCE
        ):
            piece.notes.append(
                Note(at, SHAKER_HOLD, 0.0, SHAKER_GHOST_VELOCITY, DRUM_SHAKER, "shaker", 1)
            )


def _play_growl(piece: _Piece) -> None:
    """Play a growl, on the busy days: when the load is heavy, or the night is intense; held for two bars on the tonic."""
    plan = piece.plan
    growl_gate = np.maximum(
        plan.gate(plan.power, GROWL_POWER_FROM, GROWL_POWER_TO),
        plan.gate(plan.activity * plan.night, GROWL_NIGHT_FROM, GROWL_NIGHT_TO) * GROWL_NIGHT_SHARE,
    )
    piece.growl_gate = (
        growl_gate * _role_gate(piece, GROWL_MIN_INTENSITY) * float(piece.growl != NO_GROWL)
    )
    second = 0.0
    while second < piece.length:
        if plan.at(piece.growl_gate, second) > GROWL_MIN_GATE:
            piece.notes.append(
                Note(
                    second,
                    piece.bar * 2 - GROWL_GAP,
                    float(piece.scale.midi(0, GROWL_OCTAVE)),
                    GROWL_VELOCITY,
                    piece.growl,
                    LAYER_GROWL,
                    piece.patch,
                )
            )
        second += piece.bar * 2


def _play_anomaly_swells(piece: _Piece) -> None:
    """Play the anomalies: a swell of the strings on the chord of the moment."""
    plan = piece.plan
    piece.sweep = swells(plan, SWEEP_RISE, SWEEP_FALL)
    for second, strength in plan.events:
        for k, offset_ in enumerate(SWELL_CHORD):
            piece.notes.append(
                Note(
                    second + ANOMALY_STAGGER * k,
                    ANOMALY_HOLD,
                    float(
                        piece.scale.midi(
                            _root_at(piece, second + 2.0) + offset_, ANOMALY_OCTAVE + k // 2
                        )
                    ),
                    ANOMALY_VELOCITY + ANOMALY_VELOCITY_STRENGTH * strength,
                    VOICE_TAPE_CHOIR,
                    LAYER_STRINGS,
                    piece.patch,
                )
            )


def _sweep_cutoffs(piece: _Piece) -> None:
    """Give each sequencer line a cutoff sweep of its own, moved by the house."""
    plan = piece.plan
    rng = piece.rng
    cut_walk = block_walk(rng, piece.length, CUT_WALK_SECONDS, CUT_WALK_RANGE)
    for i, key in enumerate(LINE_KEYS[: piece.lines]):
        period = SWEEP_PERIODS[i] * (1.0 + SWEEP_PERIOD_JITTER * float(rng.random()))
        phase = float(rng.uniform(0, 1))
        seconds = (np.arange(plan.count) + 0.5) / CURVE_RATE
        triangle = 0.5 + 0.5 * np.sin(2 * np.pi * (seconds / period + phase))
        x = np.clip(
            CUT_FLOOR
            + CUT_BRIGHT * plan.bright
            + CUT_POWER * plan.power
            + CUT_TRIANGLE * triangle
            + cut_walk,
            0.0,
            1.0,
        )
        scale_ = BLIP_CUTOFF_SCALE if key == LAYER_SEQ_B and piece.voice_b == BLIP_VOICE else 1.0
        piece.layers[key].cutoff = (
            scale_
            * SEQ_CUTOFF_BASE
            * SEQ_CUTOFF_RANGE**x
            * dips(plan, SEQ_DIP_DEPTH, DIP_HOLD, DIP_BACK)
        )


def _add_support_layers(piece: _Piece) -> None:
    """Add the layers of the strings, the choir, the organ bed and the slow lead."""
    plan = piece.plan
    layers = piece.layers
    form = FORM_FLOOR + (1.0 - FORM_FLOOR) * piece.intensity
    layers[LAYER_STRINGS] = Layer(
        LAYER_STRINGS,
        BED_GAIN,
        (0.5 + 0.5 * np.maximum(plan.gate(plan.activity, BED_GATE_FROM, BED_GATE_TO), piece.sweep))
        * form,
        BED_CUTOFF_BASE
        * BED_CUTOFF_RANGE ** np.clip(BED_CUTOFF_FLOOR + BED_CUTOFF_BRIGHT * plan.bright, 0, 1)
        * dips(plan, BED_DIP_DEPTH, DIP_HOLD, DIP_BACK),
        BED_RESONANCE,
        reverb=BED_REVERB,
        title="strings",
    )
    layers[LAYER_CHOIR] = Layer(
        LAYER_CHOIR,
        CHOIR_GAIN,
        plan.lift(
            plan.gate(np.maximum(plan.lights, plan.evening), CHOIR_GATE_FROM, CHOIR_GATE_TO),
            CHOIR_FLOOR,
        )
        * _role_gate(piece, CHOIR_MIN_INTENSITY),
        None,
        0.0,
        reverb=CHOIR_REVERB,
        title="choir",
    )
    layers[LAYER_BED] = Layer(
        LAYER_BED,
        STRINGS_GAIN,
        (
            STRINGS_PRESENCE_FLOOR
            + STRINGS_PRESENCE_POWER * np.clip(plan.power + plan.activity * 0.5, 0.0, 1.0)
        )
        * form,
        None,
        0.0,
        reverb=STRINGS_REVERB,
        title="organ bed",
    )
    layers[LAYER_LEAD] = Layer(
        LAYER_LEAD, LEAD_GAIN, None, None, 0.0, reverb=LEAD_REVERB, echo=LEAD_ECHO, title="lead"
    )
    trim_layer(layers[LAYER_SEQ_A], piece.seq_sig)
    trim_layer(layers[LAYER_LEAD], piece.lead_sig)


def _add_drum_layers(piece: _Piece) -> None:
    """Add the layers of the kit and the growl, and set what the layers that lead are trimmed by."""
    plan = piece.plan
    layers = piece.layers
    layers[LAYER_PULSE] = Layer(
        LAYER_PULSE, PULSE_GAIN, piece.pulse_gate, None, 0.0, reverb=PULSE_REVERB, title="pulse"
    )
    if piece.quality.max_layers >= RICH_MIN_LAYERS:
        # the toms and the shaker are rolled off, so that they are warm and have no click
        layers[LAYER_TOMS] = Layer(
            LAYER_TOMS,
            TOMS_GAIN,
            piece.tom_gate,
            TOMS_CUTOFF * dips(plan, SEQ_DIP_DEPTH, DIP_HOLD, DIP_BACK),
            0.0,
            reverb=TOMS_REVERB,
            title="toms",
        )
        layers[LAYER_SHAKER] = Layer(
            LAYER_SHAKER,
            SHAKER_GAIN,
            piece.shaker_gate,
            SHAKER_CUTOFF * dips(plan, SEQ_DIP_DEPTH, DIP_HOLD, DIP_BACK),
            0.0,
            reverb=SHAKER_REVERB,
            title="shaker",
        )
    layers[LAYER_GROWL] = Layer(
        LAYER_GROWL,
        GROWL_GAIN,
        piece.growl_gate,
        None,
        0.0,
        reverb=GROWL_REVERB,
        title="growl bass",
    )
    lead.apply_curves(layers, piece.curves, lead.LEAD_STEP_DB, LEAD_TRIM_DB)


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
        [],
        space,
        echo_seconds=piece.step * 3.0,
        echo_feedback=0.5,
        tempo=piece.bpm,
        events=list(plan.events),
        shimmer=SHIMMER,
        level_db=LEVEL_DB,
    )
    plan.choices[lead.LEAD_HANDOVERS_CHOICE] = str(len(piece.leads.handed))
    score.leads = piece.leads
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{piece.bpm:.0f} BPM in {piece.scale.mode}; sequence A '{piece.name_a}' (16 steps, {piece.voice_a}), B '{piece.name_b}' (12 steps, {piece.voice_b}), C '{piece.name_c}' ({len(piece.cell_c)} steps); "
            f"progression {piece.prog_name}, a chord every {piece.chord_len_bars} bars over a tonic pedal; kit '{piece.kit_name}', toms '{piece.tom_name}'; mutations applied: {sum(piece.mutations.values())}."
        ),
        lead.summary(piece.default_lead, piece.leads),
        f"Lead {piece.lead_voice} played the motif {len(piece.statements)} times and {len(piece.phrases_played)} composed phrases; strings {piece.strings}; growl {piece.growl}.",
    ]
    return score
