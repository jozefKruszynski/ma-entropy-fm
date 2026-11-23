"""
Ambient: tape loops of unrelated lengths that phase against each other, a sparse felt piano, slow drones.

The generative system is the one of Brian Eno's "Music for Airports": every loop plays one long note and
repeats after a time of its own, and the lengths share no common measure, so the loops never line up the same
way twice. A loop sometimes records a new note from the pitch set of the day, and the set itself moves slowly
between harmonic centres (a Markov chain), so the harmony changes pass by pass and no change is ever sudden.
The piano is a Poisson process that draws from the same set. There is no beat.

How the house steers it: the character of the day picks the identity. A quiet day gets long loops, two of them,
few notes, a rare piano, a close harmonic plateau that holds for minutes and open fifths; a busy day gets short
loops, five of them, more notes and piano, wider centres that move every minute and stacked thirds. A warm day
takes the rounder loop voices (Juno, strings), a cold or late one the darker ones; the piano and the drone are the room's own
and stay the same every day (`signature`). The form of
the day (`Plan.sections`) lets loops and strings enter and leave on phrase boundaries, and the strongest anomaly
brings the full set of loops in at the next boundary. The room's motif is played on the piano in the first and the last peak or
plateau section, the later one developed, in open fifths and sixths as in Cinematic Glitch, and the piano rings in
the long hall of that style. Who leads is the room's too (`lead`): the piano states the motif, the loop pads
play it slowly or the strings walk the chord of the centre, in the build, peak and plateau phrases of the day, the others
stepping back and sparser, by only 1.5 dB here so that it stays calm; a door, motion, an appliance or the hob hands the next
phrase to the piano, a light, a cover or the media to the pads. Within a day, activity brings loops in and out, the power of the appliances sets the drone, the
temperature and the sun open the pads' filter, humidity makes the room bigger, the lights and the evening swell
the strings, and an anomaly is a swell and a new harmonic centre.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import lead
from .gen import block_walk, dips, markov, swells, weighted
from .harmony import (
    LEAD_OCTAVE,
    MOTIF_CHOICE,
    PIANO_FIFTH_STEPS,
    PIANO_HALL_SEND,
    PIANO_SIXTH_STEPS,
    SIGNED_TAG_SPAN,
    closeness,
    develop_motif,
    motif_seed,
    motif_sites,
    voice_lead,
)
from .plan import CLIMAX, PLATEAU
from .score import (
    LAYER_DRONE,
    LAYER_GROWL,
    LAYER_PIANO,
    LAYER_STRINGS,
    VOICE_TAPE_CHOIR,
    VOICE_TAPE_STRINGS,
    Layer,
    Note,
    Score,
    Texture,
)
from .signature import SigVoice, gain_of_db, patch_of, signature, trim_layer, trim_velocity
from .synth import FIFTH_SEMITONES, SEMITONES_PER_OCTAVE

if TYPE_CHECKING:
    from .plan import Plan
    from .quality import Quality
    from .synth import Scale

# pitch sets as scale degrees with weights (0 is the root of the day); each is a sound of its own
SETS: tuple[tuple[str, tuple[tuple[int, float], ...]], ...] = (
    ("open fifths", ((0, 3.0), (4, 3.0), (1, 2.0), (5, 2.0))),
    ("stacked thirds", ((0, 2.0), (2, 2.0), (4, 2.0), (6, 2.0), (1, 1.5))),
    ("suspended", ((0, 3.0), (4, 2.0), (1, 2.0), (3, 2.0))),
    ("added sixth", ((0, 2.0), (2, 1.5), (4, 2.0), (5, 2.0))),
    ("Budd", ((0, 2.0), (1, 2.0), (2, 1.0), (4, 2.0), (6, 1.0))),
    ("airports", ((0, 2.0), (1, 1.0), (3, 1.5), (4, 2.0), (5, 2.0))),
)
# harmonic centres: a degree and where it may go next
CENTRES: tuple[tuple[str, dict[int, tuple[tuple[int, float], ...]]], ...] = (
    (
        "wide",
        {
            0: ((3, 0.35), (5, 0.3), (4, 0.2), (2, 0.15)),
            3: ((0, 0.5), (5, 0.2), (4, 0.3)),
            5: ((3, 0.4), (0, 0.3), (4, 0.3)),
            4: ((0, 0.5), (5, 0.3), (3, 0.2)),
            2: ((0, 0.5), (3, 0.3), (5, 0.2)),
        },
    ),
    ("close", {0: ((4, 0.5), (5, 0.5)), 4: ((0, 0.6), (5, 0.4)), 5: ((0, 0.5), (4, 0.5))}),
    ("plagal", {0: ((3, 0.6), (5, 0.4)), 3: ((0, 0.7), (5, 0.3)), 5: ((3, 0.5), (0, 0.5))}),
    (
        "Budd",
        {
            0: ((5, 0.5), (3, 0.3), (2, 0.2)),
            5: ((3, 0.5), (0, 0.5)),
            3: ((0, 0.6), (2, 0.4)),
            2: ((0, 0.6), (5, 0.4)),
        },
    ),
)
# loop lengths in seconds that no two share a simple ratio: they phase and almost never line up
LENGTHS: tuple[tuple[float, ...], ...] = (
    (17.2, 19.2, 27.6, 30.8, 33.3, 35.6, 39.7),
    (18.1, 21.6, 26.0, 31.2, 33.6, 37.3, 40.2),
    (17.3, 24.9, 27.8, 29.9, 32.1, 35.8, 38.5),
    (29.2, 35.0, 42.4, 47.6, 51.1, 54.8, 61.3),
    (11.1, 13.1, 20.8, 22.8, 24.7, 26.8, 29.7),
)
# how busy a day each set of lengths suits (0 quiet, 1 busy): the quiet days get the long loops
LENGTH_ENERGY = (0.5, 0.3, 0.7, 0.0, 1.0)
# how much colour each pitch set has, 0 bare to 1 full of thirds: the busy and warm days take the fuller ones
SET_COLOUR = {
    "open fifths": 0.05,
    "suspended": 0.25,
    "airports": 0.4,
    "Budd": 0.55,
    "added sixth": 0.75,
    "stacked thirds": 0.95,
}
# how far each table of centres travels, 0 a plateau on two neighbours to 1 wide moves
CENTRE_TRAVEL = {"close": 0.0, "plagal": 0.3, "Budd": 0.6, "wide": 1.0}
VOICE_BOWED_PAD = "bowed_pad"
VOICE_GLASS_PAD = "glass_pad"
VOICE_JUNO_PAD = "juno_pad"
VOICE_STRING_MACHINE = "string_machine"
VOICE_REESE = "reese"
LAYER_PADS = "pads"
TEXTURE_ROOM = "room"
# how round (1) or glassy and dark (-1) each voice is
LOOP_WARMTH = {
    VOICE_TAPE_STRINGS: 0.3,
    VOICE_TAPE_CHOIR: 0.1,
    VOICE_BOWED_PAD: -0.2,
    VOICE_GLASS_PAD: -0.9,
    VOICE_JUNO_PAD: 0.8,
    VOICE_STRING_MACHINE: 0.6,
}
# a late night darkens the voices by this much; the colour of a pitch set follows warmth, energy and the late night
LATE_DARKEN = 0.5
COLOUR_WARMTH = 0.3
COLOUR_ENERGY = 0.4
COLOUR_LATE = 0.2
LOOP_VOICES = (
    VOICE_TAPE_STRINGS,
    VOICE_TAPE_CHOIR,
    VOICE_BOWED_PAD,
    VOICE_GLASS_PAD,
    VOICE_JUNO_PAD,
    VOICE_STRING_MACHINE,
)
# the least presence of each loop: the first three are the base of the piece
STRINGS_FLOOR = 0.4
LOOP_FLOORS = (0.6, 0.5, 0.42, 0.0, 0.0)
# the register of each loop: octaves of the root (4 holds middle C)
OCTAVES = (3, 4, 3, 4, 5, 3, 4)
# the form: a notional bar, so that sections change every 8 bars (40 seconds); a loop enters when the section is at least this strong
SECTION_BAR = 5.0
LOOP_GATES = (0.0, 0.1, 0.25, 0.4, 0.55)
STRINGS_GATE = 0.2
LOOP_ENTRY_STAGGER = 1.5
# the strongest anomaly lifts the next section to this strength: the full set of loops enters
EVENT_FORM = 0.7
# loops: two on the quietest day, five on the busiest; the share of the passes that sound
MIN_LOOPS = 2
MAX_LOOPS = 5
DENSITY_QUIET = 0.5
# how long a harmonic centre holds (seconds, low and high) on the quietest and the busiest day
DWELL_QUIET = (150.0, 240.0)
DWELL_BUSY = (50.0, 90.0)
# the piano: a Poisson rate (per second) that the day sets, and what the activity adds
PIANO_BASE = (0.02, 0.08)
PIANO_ACTIVITY = (0.04, 0.14)
PIANO_GAP = 2.2
# the centre that an anomaly moves to is one of these degrees, by these weights
SHIFT_CENTRES = (3, 4, 5, 2)
SHIFT_WEIGHTS = (0.3, 0.3, 0.25, 0.15)
# the octave of a felt piano note, and the interval of its second note, by their weights
PIANO_OCTAVES = (3, 4, 5)
PIANO_OCTAVE_WEIGHTS = (0.25, 0.5, 0.25)
DYAD_INTERVALS = (2, 4, 3)
DYAD_WEIGHTS = (0.4, 0.4, 0.2)
# the drone plays on the centre two octaves under middle C and its fifth, the swell stacks these degrees
DRONE_OCTAVE = 2
DRONE_FIFTH = 4
STRINGS_TRIAD = (0, 2, 4)
# the strings triad is voice led from one centre to the next: the top tone starts an octave above, and the voices keep
# inside two octaves from this degree of this octave
STRINGS_LIFTS = (0, 0, 7)
STRINGS_LOW = 0
STRINGS_OCTAVE = 3
SWELL_OFFSETS = (0, 2, 4, 6)
# the swell chord is in this octave, its fourth tone an octave above
SWELL_OCTAVE = 3
# the strings come every so many seconds (low, high) on the quietest and the busiest day
STRINGS_GAP_QUIET = (40.0, 64.0)
STRINGS_GAP_BUSY = (22.0, 34.0)
# a growl is for the busiest days, when the power is high
GROWL_ENERGY = 0.8
# the room's motif on the piano: seconds per beat on the quietest and the busiest day, never below this note (C4)
MOTIF_BEAT = (2.0, 1.1)
MOTIF_FLOOR = 60
MOTIF_OCTAVE = 4
MOTIF_TAIL = 2.5
MOTIF_VELOCITY = 0.4
MOTIF_ENTRY = 1.0
MOTIF_ROLES = (CLIMAX, PLATEAU)
# a statement of the motif is played as open intervals, as in Cinematic Glitch, and the first of a statement has a soft
# low note a further octave down
MOTIF_DEPTH_SHARE = 0.4
# a lead is gentler here than in the styles with a beat: the leading layer comes up and the others step back by this many dB
LEAD_STEP_DB = 1.5
# what each layer is trimmed by while it leads, so that every lead sits a little over the others and no more
LEAD_TRIM_DB = {LAYER_PIANO: -4.2, LAYER_PADS: -8.9, LAYER_STRINGS: -5.5}
# the choir sits this much under the other loop voices of the pads
CHOIR_TRIM_DB = -6.0
CHOIR = SigVoice(VOICE_TAPE_CHOIR, None, CHOIR_TRIM_DB)
# the strings that lead are at least this present
LEAD_STRINGS_PRESENCE = 0.8
# the pads state the motif stretched over this share of a phrase; the strings walk the chord of the centre, a note of
# LEAD_STRINGS_HOLD seconds every LEAD_STRINGS_STEP, up and down
LEAD_PADS_SPAN = 0.75
LEAD_PADS_VELOCITY = 0.5
LEAD_STRINGS_WALK = (0, 2, 4, 6, 4, 2)
LEAD_STRINGS_STEP = 7.0
LEAD_STRINGS_HOLD = 12.0
LEAD_STRINGS_VELOCITY = 0.45
LEAD_VIBRATO_CENTS = 8.0
# the three layers that can lead: the room picks its own, and a happening of the day may hand a phrase to another;
# Ambient has no rhythmic layer, so the piano stands in for it too, and the strings only lead by the room's choice
LEADS = (LAYER_PIANO, LAYER_PADS, LAYER_STRINGS)
LEAD_BY_TRAIT = {
    lead.PERCUSSIVE: LAYER_PIANO,
    lead.RHYTHMIC: LAYER_PIANO,
    lead.SUSTAINED: LAYER_PADS,
}
# a lit house shortens the dwell of a centre by up to this share
LIT_DWELL_SHORTENING = 0.3
# before this hour of the clock the second loop is the bowed pad
MORNING_CALM_HOUR = 10.0
# in an evening this lit every third loop plays the voice of the strings
EVENING_STRINGS_LEVEL = 0.6
# the activity over the threshold of a loop that brings it in fully
LOOP_FADE_IN = 0.3
# the share of its presence that a loop loses at night, the first one never
LOOP_NIGHT_DUCK = 0.45
# the thresholds of activity of the loops are spread between these
LOOP_THRESHOLD_LOW = -0.4
LOOP_THRESHOLD_HIGH = 0.2
# a loop starts within this share of its length
LOOP_START_SHARE = 0.6
# the chance that a loop records a new note, and what the lights add
RERECORD_BASE = 0.18
RERECORD_LIGHTS = 0.25
# the chance that a pass sounds: this much, and the rest by the presence to the power
PASS_FLOOR = 0.15
PASS_SHARE = 0.85
PASS_EXPONENT = 0.6
MIN_LOOP_PRESENCE = 0.03
# a loop note lasts this share of the loop at least and at most, never over the longest
LOOP_NOTE_SHARE_LOW = 0.55
LOOP_NOTE_SHARE_HIGH = 0.8
LOOP_NOTE_MAX = 22.0
LOOP_VELOCITY = 0.32
LOOP_VELOCITY_PRESENCE = 0.24
LOOP_VELOCITY_BRIGHT = 0.08
# the first piano note comes between these seconds
PIANO_START_MIN = 4.0
PIANO_START_MAX = 12.0
PIANO_NIGHT_DUCK = 0.55
# the rate of the piano follows the form: this much on its weakest section, and the rest
PIANO_FORM_FLOOR = 0.4
PIANO_FORM_SHARE = 0.6
MIN_PIANO_RATE = 0.01
PIANO_VELOCITY = 0.24
PIANO_VELOCITY_RANDOM = 0.2
PIANO_VELOCITY_BRIGHT = 0.08
PIANO_HOLD_MIN = 3.0
PIANO_HOLD_MAX = 5.5
# a piano note is sometimes followed by a second one
PIANO_DYAD_CHANCE = 0.2
PIANO_DYAD_DELAY = 0.32
PIANO_DYAD_VELOCITY = 0.8
DRONE_HOLD = 26.0
DRONE_VELOCITY = 0.55
# a new drone note every this many seconds, and its fifth on every other one
DRONE_STEP = 22.0
DRONE_FIFTH_HOLD = 24.0
DRONE_FIFTH_VELOCITY = 0.4
STRINGS_START = 8.0
STRINGS_NIGHT_DUCK = 0.6
# the strings need this much warmth to play
STRINGS_MIN_WARMTH = 0.2
STRINGS_STAGGER = 0.15
STRINGS_HOLD = 22.0
STRINGS_VELOCITY = 0.3
STRINGS_VELOCITY_WARM = 0.2
GROWL_POWER_FROM = 0.6
GROWL_POWER_TO = 0.9
GROWL_MIN_PRESENCE = 0.3
GROWL_HOLD = 14.0
GROWL_STEP = 16.0
SWELL_STAGGER = 0.4
SWELL_HOLD = 18.0
SWELL_VELOCITY = 0.4
SWELL_VELOCITY_STRENGTH = 0.3
# the cutoff of the pads in Hz, and the factor that the brightness opens it by, to the power
PADS_CUTOFF_BASE = 480.0
PADS_CUTOFF_RANGE = 12.0
CUTOFF_FLOOR = 0.15
CUTOFF_BRIGHT = 0.85
CUTOFF_SWELL = 0.25
CUTOFF_CEILING = 1.2
CUTOFF_DIP = 0.7
CUTOFF_WALK_SECONDS = 50.0
CUTOFF_WALK_HZ = 120.0
DRONE_PRESENCE_FLOOR = 0.4
DRONE_PRESENCE_POWER = 0.6
DRONE_PRESENCE_ACTIVITY = 0.25
PADS_RESONANCE = 0.12
PADS_REVERB = 0.85
# the piano is sent to the long hall of Cinematic Glitch (PIANO_HALL_SEND), and the layer comes down by this much to make room for it
PIANO_HALL_LEVEL_DB = -3.0
PIANO_DRY_GAIN = 1.2
PIANO_GAIN = PIANO_DRY_GAIN * gain_of_db(PIANO_HALL_LEVEL_DB)
PIANO_REVERB = 0.7
PIANO_ECHO = 0.12
DRONE_GAIN = 0.9
DRONE_REVERB = 0.35
STRINGS_GAIN = 0.8
STRINGS_GATE_FROM = 0.15
STRINGS_REVERB = 0.8
GROWL_REVERB = 0.2
SPACE_BASE = 0.3
SPACE_HUMID = 0.3
SPACE_SWELL = 0.25
SPACE_MAX = 0.95
ROOM_TEXTURE_GAIN = 0.004
ECHO_SECONDS = 0.62
ECHO_FEEDBACK = 0.35
SHIMMER = 0.22
COMPRESS_DB = -24.0
LEVEL_DB = -7.0


def compose(plan: Plan) -> Score:
    """
    Make the score of an ambient soundscape.

    :param plan: The curves and the deals of the day.
    """
    piece = _Piece(plan)
    _pick_pitch_material(piece)
    _pick_voices_and_patch(piece)
    _count_loops(piece)
    _lay_out_form(piece)
    _walk_centres(piece)
    _play_loops(piece)
    _state_room_motif(piece)
    _play_lead_phrases(piece)
    _play_felt_piano(piece)
    _play_drones(piece)
    _play_strings(piece)
    _play_growl(piece)
    _play_anomaly_swells(piece)
    lead.back_off_followers(piece, frozenset(LEADS), "lead")
    _add_layers(piece)
    return _assemble(piece)


@dataclass
class _Piece:
    """The working state of one ambient piece, read and appended to by the sections of `compose`."""

    plan: Plan
    scale: Scale = field(init=False)
    quality: Quality = field(init=False)
    length: int = field(init=False)
    energy: float = field(init=False)
    rng: np.random.Generator = field(init=False)
    activity: np.ndarray = field(init=False)
    night: np.ndarray = field(init=False)
    warmth: float = field(init=False)
    set_name: str = field(init=False)
    pitch_set: tuple[tuple[int, float], ...] = field(init=False)
    degrees: list[int] = field(init=False)
    weights: list[float] = field(init=False)
    centre_name: str = field(init=False)
    centre_table: dict[int, tuple[tuple[int, float], ...]] = field(init=False)
    lengths: tuple[float, ...] = field(init=False)
    voice_a: str = field(init=False)
    voice_b: str = field(init=False)
    piano_sig: SigVoice = field(init=False)
    drone_sig: SigVoice = field(init=False)
    piano: str = field(init=False)
    drone_voice: str = field(init=False)
    patch: int = field(init=False)
    piano_patch: int = field(init=False)
    drone_patch: int = field(init=False)
    loops: int = field(init=False)
    density: float = field(init=False)
    sections: list[tuple[float, float, str, float]] = field(init=False)
    starts: list[float] = field(init=False)
    form: list[float] = field(init=False)
    lead_grid: float = field(init=False)
    default_lead: str = field(init=False)
    leads: lead.Leads = field(init=False)
    centres: list[tuple[float, int]] = field(init=False)
    order: np.ndarray = field(init=False)
    beat_seconds: float = field(init=False)
    targets: list[int] = field(init=False)
    growl_presence: np.ndarray = field(init=False)
    shape_kind: np.ndarray = field(init=False)
    layers: dict[str, Layer] = field(default_factory=dict)
    notes: list[Note] = field(default_factory=list)
    statements: list[float] = field(default_factory=list)
    statement_spans: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Take what the plan already says; the first draw of the piece comes after it."""
        self.scale = self.plan.scale
        self.quality = self.plan.quality
        self.length = self.plan.length
        self.energy = self.plan.character.energy
        self.rng = self.plan.rng("ambient")
        self.activity = self.plan.activity
        self.night = self.plan.night


def _pick_pitch_material(piece: _Piece) -> None:
    """Pick the pitch set, the harmonic centres, the loop lengths and the two loop voices."""
    plan = piece.plan
    character = plan.character
    energy = piece.energy
    piece.warmth = float(np.clip(character.warmth - LATE_DARKEN * character.late_night, -1.0, 1.0))
    colour = float(
        np.clip(
            0.5
            + COLOUR_WARMTH * character.warmth
            + COLOUR_ENERGY * (energy - 0.5)
            - COLOUR_LATE * character.late_night,
            0.0,
            1.0,
        )
    )
    sets = list(SETS)
    piece.set_name, piece.pitch_set = plan.pick(
        "pitch set", sets, [closeness(SET_COLOUR[name], colour) for name, _ in sets]
    )
    piece.centre_name, piece.centre_table = plan.pick(
        "harmonic centres",
        CENTRES,
        [closeness(CENTRE_TRAVEL[name], energy) for name, _ in CENTRES],
    )
    piece.lengths = plan.pick(
        "loop lengths", LENGTHS, [closeness(tag, energy) for tag in LENGTH_ENERGY]
    )
    piece.voice_a = plan.pick(
        "loop voice A",
        LOOP_VOICES,
        [closeness(LOOP_WARMTH[v], piece.warmth, SIGNED_TAG_SPAN) for v in LOOP_VOICES],
    )
    others = [v for v in LOOP_VOICES if v != piece.voice_a]
    piece.voice_b = plan.pick(
        "loop voice B",
        others,
        [closeness(LOOP_WARMTH[v], piece.warmth, SIGNED_TAG_SPAN) for v in others],
    )
    piece.degrees = [d for d, _ in piece.pitch_set]
    piece.weights = [w for _, w in piece.pitch_set]


def _pick_voices_and_patch(piece: _Piece) -> None:
    """Pick the voices of the piano and the drone, and the patch bank of the piece."""
    plan = piece.plan
    piece.piano_sig = signature(plan, LAYER_PIANO, "piano")
    piece.drone_sig = signature(plan, LAYER_DRONE, "drone voice")
    piece.piano, piece.drone_voice = piece.piano_sig.voice, piece.drone_sig.voice
    piece.patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(piece.patch)
    piece.piano_patch = patch_of(piece.piano_sig, piece.patch)
    piece.drone_patch = patch_of(piece.drone_sig, piece.patch)


def _count_loops(piece: _Piece) -> None:
    """Set how many loops play and how many of their passes sound."""
    # two loops are the floor of a quiet day and five the full set of a busy one (fewer in the cheapest tier)
    piece.loops = int(
        np.clip(
            MIN_LOOPS + round((MAX_LOOPS - MIN_LOOPS) * piece.energy),
            MIN_LOOPS,
            max(piece.quality.max_layers - 2, MIN_LOOPS),
        )
    )
    piece.density = DENSITY_QUIET + (1.0 - DENSITY_QUIET) * piece.energy


def _lay_out_form(piece: _Piece) -> None:
    """Lay out the strength of each section, and who leads in each phrase."""
    plan = piece.plan
    # the strongest anomaly lifts the next section, so the full set enters
    piece.sections = plan.sections(SECTION_BAR)
    piece.starts = [start for start, _, _, _ in piece.sections]
    piece.form = [intensity for _, _, _, intensity in piece.sections]
    if plan.events:
        at, _ = max(plan.events, key=lambda event: event[1])
        after = bisect_right(piece.starts, at)
        if after < len(piece.sections):
            piece.form[after] = max(piece.form[after], EVENT_FORM)

    piece.lead_grid = plan.form_grid(SECTION_BAR)
    piece.default_lead = lead.default_lead(plan, LEADS)
    piece.leads = lead.phrase_leads(
        plan,
        piece.lead_grid,
        piece.default_lead,
        lead.daytime_phrases(plan, piece.sections, piece.lead_grid),
        LEAD_BY_TRAIT,
    )


def _section_of(piece: _Piece, second: float) -> int:
    """Return the index of the section of the form that a second lies in."""
    return max(bisect_right(piece.starts, second) - 1, 0)


def _entry_after(piece: _Piece, second: float, gate: float) -> float | None:
    """Return the start of the next section strong enough for a layer with this gate, or None."""
    for index in range(_section_of(piece, second) + 1, len(piece.sections)):
        if piece.form[index] >= gate:
            return piece.starts[index]
    return None


def _walk_centres(piece: _Piece) -> None:
    """Walk the harmonic centres: a Markov chain that dwells for minutes on a quiet day and moves at an anomaly."""
    plan = piece.plan
    rng = piece.rng
    energy = piece.energy
    piece.centres = [(0.0, 0)]
    dwell_low = DWELL_QUIET[0] + (DWELL_BUSY[0] - DWELL_QUIET[0]) * energy
    dwell_high = DWELL_QUIET[1] + (DWELL_BUSY[1] - DWELL_QUIET[1]) * energy
    clock = float(rng.uniform(dwell_low, dwell_high))
    state = 0
    shifts = {round(jump) for jump, _ in plan.events}
    while clock < piece.length:
        state = markov(rng, state, dict(piece.centre_table.items()))
        piece.centres.append((clock, state))
        lit = plan.at(plan.lights, clock)
        clock += float(rng.uniform(dwell_low, dwell_high)) * (1.0 - LIT_DWELL_SHORTENING * lit)
    for jump in sorted(shifts):
        far = weighted(rng, SHIFT_CENTRES, SHIFT_WEIGHTS)
        piece.centres.append((float(jump), far))
    piece.centres.sort()


def _centre_at(piece: _Piece, second: float) -> int:
    """Return the harmonic centre in force at a second."""
    value = 0
    for start, degree in piece.centres:
        if start <= second:
            value = degree
    return value


def _pick_degree(piece: _Piece, second: float, previous: int | None) -> int:
    """Return a degree of the pitch set on the centre at a second, staying near the previous one more often than not."""
    w = list(piece.weights)
    if previous is not None:
        # a loop that re-records stays near its old pitch more often than not
        w = [
            x * (2.0 if abs(d - previous) <= 1 else 1.0)
            for x, d in zip(w, piece.degrees, strict=True)
        ]
    return weighted(piece.rng, piece.degrees, w) + _centre_at(piece, second)


def _chord_strings_voice(piece: _Piece) -> str:
    """Return the voice of the triads and the leading strings."""
    return VOICE_STRING_MACHINE if piece.patch == 0 else VOICE_JUNO_PAD


def _loop_voice(piece: _Piece, index: int, second: float) -> str:
    """Return the voice that loop `index` plays at a second."""
    plan = piece.plan
    morning_calm = plan.at(plan.hour, second) < MORNING_CALM_HOUR
    if plan.at(plan.evening, second) > EVENING_STRINGS_LEVEL and index % 3 == 2:
        return _chord_strings_voice(piece)
    if morning_calm and index == 1:
        return VOICE_BOWED_PAD
    return piece.voice_a if index % 2 == 0 else piece.voice_b


def _loop_presence(piece: _Piece, index: int, second: float, threshold: float) -> float:
    """Return how present loop `index` is at a second."""
    plan = piece.plan
    presence = float(
        np.clip((plan.at(piece.activity, second) - threshold) / LOOP_FADE_IN, 0.0, 1.0)
    )
    # the first three loops are the base and never leave; the activity brings in the rest
    return max(presence, LOOP_FLOORS[min(index, len(LOOP_FLOORS) - 1)]) * (
        1.0 - LOOP_NIGHT_DUCK * plan.at(piece.night, second) * (index > 0)
    )


def _play_loops(piece: _Piece) -> None:
    """Play the tape loops: each repeats one long note after a time of its own."""
    plan = piece.plan
    rng = piece.rng
    piece.order = rng.permutation(len(piece.lengths))[: piece.loops]
    thresholds = np.linspace(LOOP_THRESHOLD_LOW, LOOP_THRESHOLD_HIGH, piece.loops)
    for index, which in enumerate(piece.order):
        loop_length = float(piece.lengths[int(which)])
        octave = OCTAVES[index % len(OCTAVES)]
        second = float(rng.uniform(0.0, loop_length * LOOP_START_SHARE))
        degree: int | None = None
        while second < piece.length:
            gate = LOOP_GATES[min(index, len(LOOP_GATES) - 1)]
            if piece.form[_section_of(piece, second)] < gate:
                # the loop waits for the next phrase boundary at which the form is strong enough
                entry = _entry_after(piece, second, gate)
                if entry is None:
                    break
                second = entry + LOOP_ENTRY_STAGGER * index
                continue
            presence = _loop_presence(piece, index, second, thresholds[index])
            if degree is None or rng.random() < RERECORD_BASE + RERECORD_LIGHTS * plan.at(
                plan.lights, second
            ):
                degree = _pick_degree(piece, second, degree)
            if (
                rng.random() < piece.density * (PASS_FLOOR + PASS_SHARE * presence**PASS_EXPONENT)
                and presence > MIN_LOOP_PRESENCE
            ):
                use = _loop_voice(piece, index, second)
                dur = min(
                    loop_length * float(rng.uniform(LOOP_NOTE_SHARE_LOW, LOOP_NOTE_SHARE_HIGH)),
                    LOOP_NOTE_MAX,
                )
                midi = piece.scale.midi(degree, octave)
                vel = (
                    LOOP_VELOCITY
                    + LOOP_VELOCITY_PRESENCE * presence
                    + LOOP_VELOCITY_BRIGHT * plan.at(plan.bright, second)
                )
                piece.notes.append(
                    Note(
                        second,
                        dur,
                        float(midi),
                        _pad_velocity(use, vel),
                        use,
                        LAYER_PADS,
                        piece.patch,
                    )
                )
            second += loop_length


def _state_motif(piece: _Piece, begin: float, motif: tuple[tuple[int, float], ...]) -> float:
    """Play a statement of the motif on the piano from `begin`, as open intervals; return how long it lasts."""
    shift = 0
    while min(piece.scale.midi(degree, MOTIF_OCTAVE) for degree, _ in motif) + shift < MOTIF_FLOOR:
        shift += SEMITONES_PER_OCTAVE
    clock = begin
    for index, (degree, beats) in enumerate(motif):
        upper = piece.scale.midi(degree, MOTIF_OCTAVE) + shift
        lower = _fifth_or_sixth_below(piece, degree) + shift
        hold = beats * piece.beat_seconds + MOTIF_TAIL
        pitches = [(upper, MOTIF_VELOCITY), (lower, MOTIF_VELOCITY)]
        if index == 0:
            pitches.append((lower - SEMITONES_PER_OCTAVE, MOTIF_VELOCITY * MOTIF_DEPTH_SHARE))
        piece.notes.extend(
            Note(clock, hold, float(pitch), velocity, piece.piano, LAYER_PIANO, piece.piano_patch)
            for pitch, velocity in pitches
        )
        clock += beats * piece.beat_seconds
    return clock - begin


def _state_room_motif(piece: _Piece) -> None:
    """State the motif of the room on the piano, in the first and the last peak or plateau section, the later one developed."""
    plan = piece.plan
    energy = piece.energy
    piece.beat_seconds = MOTIF_BEAT[0] + (MOTIF_BEAT[1] - MOTIF_BEAT[0]) * energy
    targets = [i for i, (_, _, role, _) in enumerate(piece.sections) if role in MOTIF_ROLES]
    piece.targets = motif_sites(targets or [len(piece.sections) // 2])
    for number, i in enumerate(piece.targets):
        motif = develop_motif(plan.motif(), number, motif_seed(plan))
        motif_span = sum(beats for _, beats in motif) * piece.beat_seconds
        begin = piece.sections[i][0] + MOTIF_ENTRY
        if begin + motif_span + MOTIF_TAIL >= piece.length:
            if number:
                continue
            begin = max(piece.length - motif_span - MOTIF_TAIL - MOTIF_ENTRY, 0.0)
        piece.statements.append(begin)
        piece.statement_spans.append(_state_motif(piece, begin, motif))

    plan.choices[MOTIF_CHOICE] = str(len(piece.statements))


def _play_lead_phrases(piece: _Piece) -> None:
    """
    Play the phrases that the room hands to a leading layer.

    The piano states the motif, developed, the pads play it slowly and the strings walk the chord of the centre; the
    piano's phrases are kept clear of the felt piano.
    """
    plan = piece.plan
    for number, who in enumerate(piece.leads.phrases):
        begin = number * piece.lead_grid
        if who == LAYER_PIANO:
            motif = develop_motif(
                plan.motif(),
                len(piece.targets) + number,
                motif_seed(plan),
                (piece.lead_grid - MOTIF_TAIL - 2.0 * MOTIF_ENTRY) / piece.beat_seconds,
            )
            piece.statements.append(begin + MOTIF_ENTRY)
            piece.statement_spans.append(_state_motif(piece, begin + MOTIF_ENTRY, motif))
        elif who == LAYER_PADS:
            _lead_pads(piece, number, begin)
        elif who == LAYER_STRINGS:
            _lead_strings(piece, begin)


def _lead_pads(piece: _Piece, number: int, begin: float) -> None:
    """Play the motif slowly on the pads, stretched over a share of the phrase."""
    plan = piece.plan
    motif = develop_motif(plan.motif(), len(piece.targets) + number, motif_seed(plan))
    stretch = LEAD_PADS_SPAN * piece.lead_grid / sum(beats for _, beats in motif)
    clock = begin
    for degree, beats in motif:
        pitch = piece.scale.midi(degree + _centre_at(piece, begin), LEAD_OCTAVE)
        piece.notes.append(
            Note(
                clock,
                beats * stretch + MOTIF_TAIL,
                float(pitch),
                _pad_velocity(piece.voice_a, LEAD_PADS_VELOCITY),
                piece.voice_a,
                LAYER_PADS,
                piece.patch,
                LEAD_VIBRATO_CENTS,
            )
        )
        clock += beats * stretch


def _lead_strings(piece: _Piece, begin: float) -> None:
    """Walk the chord of the centre on the strings, up and down, inside one phrase."""
    strings = _chord_strings_voice(piece)
    for k, offset in enumerate(LEAD_STRINGS_WALK):
        at = begin + k * LEAD_STRINGS_STEP
        if at + LEAD_STRINGS_HOLD > begin + piece.lead_grid:
            break
        pitch = piece.scale.midi(_centre_at(piece, at) + offset, LEAD_OCTAVE)
        piece.notes.append(
            Note(
                at,
                LEAD_STRINGS_HOLD,
                float(pitch),
                LEAD_STRINGS_VELOCITY,
                strings,
                LAYER_STRINGS,
                piece.patch,
                LEAD_VIBRATO_CENTS,
            )
        )


def _in_statement(piece: _Piece, second: float) -> bool:
    """Return whether a second falls in or just around a statement of the motif."""
    return any(
        begin - PIANO_GAP <= second <= begin + span + MOTIF_TAIL
        for begin, span in zip(piece.statements, piece.statement_spans, strict=True)
    )


def _play_felt_piano(piece: _Piece) -> None:
    """Play the felt piano: a Poisson process drawing from the set of the day."""
    plan = piece.plan
    rng = piece.rng
    energy = piece.energy
    scale = piece.scale
    second = float(rng.uniform(PIANO_START_MIN, PIANO_START_MAX))
    while second < piece.length:
        night_here = plan.at(piece.night, second)
        rate = (
            PIANO_BASE[0]
            + (PIANO_BASE[1] - PIANO_BASE[0]) * energy
            + (PIANO_ACTIVITY[0] + (PIANO_ACTIVITY[1] - PIANO_ACTIVITY[0]) * energy)
            * plan.at(piece.activity, second)
        ) * (1.0 - PIANO_NIGHT_DUCK * night_here)
        rate *= PIANO_FORM_FLOOR + PIANO_FORM_SHARE * piece.form[_section_of(piece, second)]
        second += float(rng.exponential(1.0 / max(rate, MIN_PIANO_RATE))) + PIANO_GAP
        if second >= piece.length:
            break
        if _in_statement(piece, second):
            continue
        use = piece.piano
        degree_ = weighted(rng, piece.degrees, piece.weights) + _centre_at(piece, second)
        octave = weighted(rng, PIANO_OCTAVES, PIANO_OCTAVE_WEIGHTS)
        vel = (
            PIANO_VELOCITY
            + PIANO_VELOCITY_RANDOM * float(rng.random())
            + PIANO_VELOCITY_BRIGHT * plan.at(plan.bright, second)
        )
        dur = float(rng.uniform(PIANO_HOLD_MIN, PIANO_HOLD_MAX))
        piece.notes.append(
            Note(
                second,
                dur,
                float(scale.midi(degree_, octave)),
                vel,
                use,
                LAYER_PIANO,
                piece.piano_patch,
            )
        )
        if rng.random() < PIANO_DYAD_CHANCE:
            other = degree_ + int(weighted(rng, DYAD_INTERVALS, DYAD_WEIGHTS))
            piece.notes.append(
                Note(
                    second + PIANO_DYAD_DELAY,
                    dur,
                    float(scale.midi(other, octave)),
                    vel * PIANO_DYAD_VELOCITY,
                    use,
                    LAYER_PIANO,
                    piece.piano_patch,
                )
            )


def _play_drones(piece: _Piece) -> None:
    """Play the drones: a low note on the centre, overlapping, and its fifth."""
    scale = piece.scale
    second = 0.0
    while second < piece.length:
        degree_ = _centre_at(piece, second)
        piece.notes.append(
            Note(
                second,
                DRONE_HOLD,
                float(scale.midi(degree_, DRONE_OCTAVE)),
                DRONE_VELOCITY,
                piece.drone_voice,
                LAYER_DRONE,
                piece.drone_patch,
            )
        )
        if int(second / DRONE_STEP) % 2 == 1:
            piece.notes.append(
                Note(
                    second + 1.0,
                    DRONE_FIFTH_HOLD,
                    float(scale.midi(degree_ + DRONE_FIFTH, DRONE_OCTAVE)),
                    DRONE_FIFTH_VELOCITY,
                    piece.drone_voice,
                    LAYER_DRONE,
                    piece.drone_patch,
                )
            )
        second += DRONE_STEP


def _play_strings(piece: _Piece) -> None:
    """Play the strings: a voice led triad of a string machine or a Juno pad all day, swelling in when the lights are on in the evening."""
    plan = piece.plan
    rng = piece.rng
    energy = piece.energy
    second = STRINGS_START
    triads: list[tuple[float, int, float]] = []
    while second < piece.length:
        # a quiet triad of the string machine plays all day; the lights and the evening make it warmer and fuller
        warm = max(
            plan.at(plan.lights, second) * plan.at(plan.evening, second),
            STRINGS_FLOOR * (1.0 - STRINGS_NIGHT_DUCK * plan.at(piece.night, second)),
        )
        gap_low = STRINGS_GAP_QUIET[0] + (STRINGS_GAP_BUSY[0] - STRINGS_GAP_QUIET[0]) * energy
        gap_high = STRINGS_GAP_QUIET[1] + (STRINGS_GAP_BUSY[1] - STRINGS_GAP_QUIET[1]) * energy
        second += float(rng.uniform(gap_low, gap_high))
        if piece.form[_section_of(piece, second)] < STRINGS_GATE:
            continue
        if warm > STRINGS_MIN_WARMTH and rng.random() < 0.5 + 0.5 * warm:
            triads.append((second, _centre_at(piece, second), warm))
    if not triads:
        return
    strings = _chord_strings_voice(piece)
    voicing = voice_lead([root for _, root, _ in triads], STRINGS_TRIAD, STRINGS_LIFTS, STRINGS_LOW)
    for (second, _, warm), degrees in zip(triads, voicing, strict=True):
        for k, degree in enumerate(degrees):
            piece.notes.append(
                Note(
                    second + STRINGS_STAGGER * k,
                    STRINGS_HOLD,
                    float(piece.scale.midi(degree, STRINGS_OCTAVE)),
                    STRINGS_VELOCITY + STRINGS_VELOCITY_WARM * warm,
                    strings,
                    LAYER_STRINGS,
                    piece.patch,
                )
            )


def _play_growl(piece: _Piece) -> None:
    """Play a growl under a heavy load, rarely in ambient: on the busiest days only, and only when the power is high."""
    plan = piece.plan
    growl_day = piece.energy >= GROWL_ENERGY
    piece.growl_presence = plan.gate(plan.power, GROWL_POWER_FROM, GROWL_POWER_TO) * (
        1.0 if growl_day else 0.0
    )
    if growl_day and piece.growl_presence.max() > GROWL_MIN_PRESENCE:
        second = 0.0
        while second < piece.length:
            if plan.at(piece.growl_presence, second) > GROWL_MIN_PRESENCE:
                piece.notes.append(
                    Note(
                        second,
                        GROWL_HOLD,
                        float(piece.scale.midi(_centre_at(piece, second), 1)),
                        0.5,
                        VOICE_REESE,
                        LAYER_GROWL,
                        1,
                    )
                )
            second += GROWL_STEP


def _play_anomaly_swells(piece: _Piece) -> None:
    """Swell the glass pads on the new centre at each anomaly; the dip of the brightness follows in the layers."""
    plan = piece.plan
    piece.shape_kind = swells(plan)
    for second, strength in plan.events:
        root = _centre_at(piece, second + 1.0)
        for k, offset in enumerate(SWELL_OFFSETS):
            piece.notes.append(
                Note(
                    second + SWELL_STAGGER * k,
                    SWELL_HOLD,
                    float(piece.scale.midi(root + offset, SWELL_OCTAVE + k // 3)),
                    SWELL_VELOCITY + SWELL_VELOCITY_STRENGTH * strength,
                    VOICE_GLASS_PAD,
                    LAYER_PADS,
                    piece.patch,
                )
            )


def _add_layers(piece: _Piece) -> None:
    """Build the layers, with the curves of the leads applied and the voices trimmed."""
    plan = piece.plan
    cutoff = (
        PADS_CUTOFF_BASE
        * PADS_CUTOFF_RANGE
        ** np.clip(
            CUTOFF_FLOOR + CUTOFF_BRIGHT * plan.bright + CUTOFF_SWELL * piece.shape_kind,
            0.0,
            CUTOFF_CEILING,
        )
        * dips(plan, CUTOFF_DIP)
    )
    cutoff = cutoff + block_walk(piece.rng, piece.length, CUTOFF_WALK_SECONDS, CUTOFF_WALK_HZ)
    drone_presence = np.clip(
        DRONE_PRESENCE_FLOOR
        + DRONE_PRESENCE_POWER * plan.power
        + DRONE_PRESENCE_ACTIVITY * piece.activity,
        0.0,
        1.0,
    )
    layers = {
        LAYER_PADS: Layer(
            LAYER_PADS,
            1.0,
            None,
            cutoff,
            PADS_RESONANCE,
            reverb=PADS_REVERB,
            title="tape-loop pads",
        ),
        LAYER_PIANO: Layer(
            LAYER_PIANO,
            PIANO_GAIN,
            None,
            None,
            0.0,
            reverb=PIANO_REVERB,
            echo=PIANO_ECHO,
            hall=PIANO_HALL_SEND,
            title="felt piano",
        ),
        LAYER_DRONE: Layer(
            LAYER_DRONE, DRONE_GAIN, drone_presence, None, 0.0, reverb=DRONE_REVERB, title="drone"
        ),
        LAYER_STRINGS: Layer(
            LAYER_STRINGS,
            STRINGS_GAIN,
            plan.lift(plan.gate(plan.lights * plan.evening, STRINGS_GATE_FROM, 0.5), STRINGS_FLOOR),
            None,
            0.0,
            reverb=STRINGS_REVERB,
            title="strings",
        ),
        LAYER_GROWL: Layer(
            LAYER_GROWL, 0.5, piece.growl_presence, None, 0.0, reverb=GROWL_REVERB, title="growl"
        ),
    }
    curves = lead.lead_curves(plan, piece.leads, LEADS)
    strings_presence = layers[LAYER_STRINGS].presence
    if strings_presence is not None:
        layers[LAYER_STRINGS].presence = np.maximum(
            strings_presence, LEAD_STRINGS_PRESENCE * np.clip(curves[LAYER_STRINGS], 0.0, 1.0)
        )
    lead.apply_curves(layers, curves, LEAD_STEP_DB, LEAD_TRIM_DB)
    trim_layer(layers[LAYER_PIANO], piece.piano_sig)
    trim_layer(layers[LAYER_DRONE], piece.drone_sig)
    piece.layers = layers


def _assemble(piece: _Piece) -> Score:
    """Put the notes and the layers together as the score, with the notes about how it was made."""
    plan = piece.plan
    energy = piece.energy
    space = np.clip(
        SPACE_BASE + SPACE_HUMID * plan.humid + SPACE_SWELL * piece.shape_kind, 0.0, SPACE_MAX
    )
    score = Score(
        piece.length,
        piece.notes,
        piece.layers,
        [
            Texture(TEXTURE_ROOM, TEXTURE_ROOM, ROOM_TEXTURE_GAIN),
        ],
        space,
        echo_seconds=ECHO_SECONDS,
        echo_feedback=ECHO_FEEDBACK,
        events=list(plan.events),
        shimmer=SHIMMER,
        compress_db=COMPRESS_DB,
        level_db=LEVEL_DB,
    )
    plan.choices[lead.LEAD_HANDOVERS_CHOICE] = str(len(piece.leads.handed))
    score.leads = piece.leads
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{piece.loops} tape loops of {', '.join(f'{piece.lengths[int(i)]:.1f}' for i in piece.order)} seconds (no two share a measure), on {piece.voice_a} and {piece.voice_b}, "
            f"pitch set '{piece.set_name}', harmonic centres '{piece.centre_name}' ({sum(1 for _ in piece.centres)} moves)."
        ),
        f"Piano '{piece.piano}' as a Poisson process (about {PIANO_BASE[0] + (PIANO_BASE[1] - PIANO_BASE[0]) * energy + (PIANO_ACTIVITY[0] + (PIANO_ACTIVITY[1] - PIANO_ACTIVITY[0]) * energy) * float(np.mean(piece.activity)):.2f} a second at the mean activity), drone on {piece.drone_voice}.",
        lead.summary(piece.default_lead, piece.leads),
    ]
    return score


def _pad_velocity(voice: str, velocity: float) -> float:
    """Return the velocity of a pad note, the choir trimmed under the other voices."""
    return trim_velocity(CHOIR, velocity) if voice == VOICE_TAPE_CHOIR else float(velocity)


def _fifth_or_sixth_below(piece: _Piece, degree: int) -> int:
    """Return the note a fifth under a degree of the motif, or a sixth where the fifth of the scale is diminished."""
    upper = piece.scale.midi(degree, MOTIF_OCTAVE)
    fifth = piece.scale.midi(degree - PIANO_FIFTH_STEPS, MOTIF_OCTAVE)
    if upper - fifth == FIFTH_SEMITONES:
        return fifth
    return piece.scale.midi(degree - PIANO_SIXTH_STEPS, MOTIF_OCTAVE)
