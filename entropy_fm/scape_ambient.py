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
takes the rounder voices (felt piano, Juno, strings), a cold or late one the glassier and darker ones. The form of
the day (`Plan.sections`) lets loops and strings enter and leave on phrase boundaries, and the strongest anomaly
brings the full set of loops in at the next boundary. The room's motif is played on the piano in the first and the last peak or
plateau section, the later one developed. Within a day, activity brings loops in and out, the power of the appliances sets the drone, the
temperature and the sun open the pads' filter, humidity makes the room bigger, the lights and the evening swell
the strings, and an anomaly is a swell and a new harmonic centre.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING

import numpy as np

from .gen import block_walk, dips, markov, swells, weighted
from .harmony import MOTIF_CHOICE, develop_motif, motif_seed, motif_sites
from .score import Layer, Note, Score, Texture

if TYPE_CHECKING:
    from .plan import Plan

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
# how round (1) or glassy and dark (-1) each voice is
LOOP_WARMTH = {
    "tape_strings": 0.3,
    "tape_choir": 0.1,
    "bowed_pad": -0.2,
    "glass_pad": -0.9,
    "juno_pad": 0.8,
    "string_machine": 0.6,
}
PIANO_WARMTH = {"felt_piano": 0.8, "grand_piano": -0.3}
DRONE_WARMTH = {"sub_round": 0.6, "drawbar_organ": 0.2, "bowed_pad": -0.4}
# a late night darkens the voices by this much; the colour of a pitch set follows warmth, energy and the late night
LATE_DARKEN = 0.5
COLOUR_WARMTH = 0.3
COLOUR_ENERGY = 0.4
COLOUR_LATE = 0.2
LOOP_VOICES = ("tape_strings", "tape_choir", "bowed_pad", "glass_pad", "juno_pad", "string_machine")
PIANOS = ("felt_piano", "grand_piano")
DRONES = ("sub_round", "bowed_pad", "drawbar_organ")
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
MOTIF_ROLES = ("peak", "plateau")


def _closeness(tag: float, wanted: float, span: float = 1.0) -> float:
    """Return 1 when a tag is what the day wants and 0 when it is a whole span away."""
    return 1.0 - abs(tag - wanted) / span


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of an ambient soundscape.

    :param plan: The curves and the deals of the day.
    """
    scale = plan.scale
    quality = plan.quality
    length = plan.length
    rng = plan.rng("ambient")
    character = plan.character
    energy = character.energy
    warmth = float(np.clip(character.warmth - LATE_DARKEN * character.late_night, -1.0, 1.0))
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
    set_name, pitch_set = plan.pick(
        "pitch set", sets, [_closeness(SET_COLOUR[name], colour) for name, _ in sets]
    )
    centre_name, centre_table = plan.pick(
        "harmonic centres",
        CENTRES,
        [_closeness(CENTRE_TRAVEL[name], energy) for name, _ in CENTRES],
    )
    lengths = plan.pick("loop lengths", LENGTHS, [_closeness(tag, energy) for tag in LENGTH_ENERGY])
    voice_a = plan.pick(
        "loop voice A",
        LOOP_VOICES,
        [_closeness(LOOP_WARMTH[v], warmth, 2.0) for v in LOOP_VOICES],
    )
    others = [v for v in LOOP_VOICES if v != voice_a]
    voice_b = plan.pick(
        "loop voice B", others, [_closeness(LOOP_WARMTH[v], warmth, 2.0) for v in others]
    )
    piano = plan.pick("piano", PIANOS, [_closeness(PIANO_WARMTH[v], warmth, 2.0) for v in PIANOS])
    drone_voice = plan.pick(
        "drone voice", DRONES, [_closeness(DRONE_WARMTH[v], warmth, 2.0) for v in DRONES]
    )
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)

    # two loops are the floor of a quiet day and five the full set of a busy one (fewer in the cheapest tier)
    loops = int(
        np.clip(
            MIN_LOOPS + round((MAX_LOOPS - MIN_LOOPS) * energy),
            MIN_LOOPS,
            max(quality.max_layers - 2, MIN_LOOPS),
        )
    )
    density = DENSITY_QUIET + (1.0 - DENSITY_QUIET) * energy
    activity = plan.activity
    night = plan.night

    # -- the form: the strength of each section; the strongest anomaly lifts the next one, so the full set enters
    sections = plan.sections(SECTION_BAR)
    starts = [start for start, _, _, _ in sections]
    form = [intensity for _, _, _, intensity in sections]
    if plan.events:
        at, _ = max(plan.events, key=lambda event: event[1])
        after = bisect_right(starts, at)
        if after < len(sections):
            form[after] = max(form[after], EVENT_FORM)

    def section_of(second: float) -> int:
        return max(bisect_right(starts, second) - 1, 0)

    def entry_after(second: float, gate: float) -> float | None:
        """Return the start of the next section strong enough for a layer with this gate, or None."""
        for index in range(section_of(second) + 1, len(sections)):
            if form[index] >= gate:
                return starts[index]
        return None

    # -- the harmonic centres: a Markov chain that dwells for minutes on a quiet day and moves at an anomaly
    centres: list[tuple[float, int]] = [(0.0, 0)]
    dwell_low = DWELL_QUIET[0] + (DWELL_BUSY[0] - DWELL_QUIET[0]) * energy
    dwell_high = DWELL_QUIET[1] + (DWELL_BUSY[1] - DWELL_QUIET[1]) * energy
    clock = float(rng.uniform(dwell_low, dwell_high))
    state = 0
    shifts = {round(jump) for jump, _ in plan.events}
    while clock < length:
        state = markov(rng, state, dict(centre_table.items()))
        centres.append((clock, state))
        lit = plan.at(plan.lights, clock)
        clock += float(rng.uniform(dwell_low, dwell_high)) * (1.0 - 0.3 * lit)
    for jump in sorted(shifts):
        far = weighted(rng, (3, 4, 5, 2), (0.3, 0.3, 0.25, 0.15))
        centres.append((float(jump), far))
    centres.sort()

    def centre_at(second: float) -> int:
        value = 0
        for start, degree in centres:
            if start <= second:
                value = degree
        return value

    degrees = [d for d, _ in pitch_set]
    weights = [w for _, w in pitch_set]

    def pick_degree(second: float, previous: int | None) -> int:
        w = list(weights)
        if previous is not None:
            # a loop that re-records stays near its old pitch more often than not
            w = [
                x * (2.0 if abs(d - previous) <= 1 else 1.0)
                for x, d in zip(w, degrees, strict=True)
            ]
        return weighted(rng, degrees, w) + centre_at(second)

    notes: list[Note] = []
    # -- the tape loops
    order = rng.permutation(len(lengths))[:loops]
    thresholds = np.linspace(-0.4, 0.2, loops)
    for index, which in enumerate(order):
        loop_length = float(lengths[int(which)])
        voice = voice_a if index % 2 == 0 else voice_b
        octave = OCTAVES[index % len(OCTAVES)]
        second = float(rng.uniform(0.0, loop_length * 0.6))
        degree: int | None = None
        while second < length:
            gate = LOOP_GATES[min(index, len(LOOP_GATES) - 1)]
            if form[section_of(second)] < gate:
                # the loop waits for the next phrase boundary at which the form is strong enough
                entry = entry_after(second, gate)
                if entry is None:
                    break
                second = entry + LOOP_ENTRY_STAGGER * index
                continue
            presence = float(
                np.clip((plan.at(activity, second) - thresholds[index]) / 0.3, 0.0, 1.0)
            )
            # the first three loops are the base and never leave; the activity brings in the rest
            presence = max(presence, LOOP_FLOORS[min(index, len(LOOP_FLOORS) - 1)]) * (
                1.0 - 0.45 * plan.at(night, second) * (index > 0)
            )
            if degree is None or rng.random() < 0.18 + 0.25 * plan.at(plan.lights, second):
                degree = pick_degree(second, degree)
            if rng.random() < density * (0.15 + 0.85 * presence**0.6) and presence > 0.03:
                morning_calm = plan.at(plan.hour, second) < 10.0
                use = voice
                if plan.at(plan.evening, second) > 0.6 and index % 3 == 2:
                    use = "string_machine" if patch == 0 else "juno_pad"
                elif morning_calm and index == 1:
                    use = "bowed_pad"
                dur = min(loop_length * float(rng.uniform(0.55, 0.8)), 22.0)
                midi = scale.midi(degree, octave)
                vel = 0.32 + 0.24 * presence + 0.08 * plan.at(plan.bright, second)
                notes.append(Note(second, dur, float(midi), float(vel), use, "pads", patch))
            second += loop_length

    # -- the room's motif: the piano states it as scale degrees of the day's key in the first and the last peak or
    # plateau section, the later one developed
    beat_seconds = MOTIF_BEAT[0] + (MOTIF_BEAT[1] - MOTIF_BEAT[0]) * energy
    targets = [i for i, (_, _, role, _) in enumerate(sections) if role in MOTIF_ROLES]
    targets = motif_sites(targets or [len(sections) // 2])
    statements: list[float] = []
    statement_spans: list[float] = []
    for number, i in enumerate(targets):
        motif = develop_motif(plan.motif(), number, motif_seed(plan))
        motif_pitches = [scale.midi(degree, MOTIF_OCTAVE) for degree, _ in motif]
        while min(motif_pitches) < MOTIF_FLOOR:
            motif_pitches = [pitch + 12 for pitch in motif_pitches]
        motif_span = sum(beats for _, beats in motif) * beat_seconds
        begin = sections[i][0] + MOTIF_ENTRY
        if begin + motif_span + MOTIF_TAIL >= length:
            if number:
                continue
            begin = max(length - motif_span - MOTIF_TAIL - MOTIF_ENTRY, 0.0)
        statements.append(begin)
        statement_spans.append(motif_span)
        clock = begin
        for pitch, (_, beats) in zip(motif_pitches, motif, strict=True):
            hold = beats * beat_seconds
            notes.append(
                Note(clock, hold + MOTIF_TAIL, float(pitch), MOTIF_VELOCITY, piano, "piano", patch)
            )
            clock += hold

    plan.choices[MOTIF_CHOICE] = str(len(statements))

    # -- the felt piano: a Poisson process drawing from the set of the day
    second = float(rng.uniform(4.0, 12.0))
    while second < length:
        night_here = plan.at(night, second)
        rate = (
            PIANO_BASE[0]
            + (PIANO_BASE[1] - PIANO_BASE[0]) * energy
            + (PIANO_ACTIVITY[0] + (PIANO_ACTIVITY[1] - PIANO_ACTIVITY[0]) * energy)
            * plan.at(activity, second)
        ) * (1.0 - 0.55 * night_here)
        rate *= 0.4 + 0.6 * form[section_of(second)]
        second += float(rng.exponential(1.0 / max(rate, 0.01))) + PIANO_GAP
        if second >= length:
            break
        if any(
            begin - PIANO_GAP <= second <= begin + span + MOTIF_TAIL
            for begin, span in zip(statements, statement_spans, strict=True)
        ):
            continue
        hour = plan.at(plan.hour, second)
        use = piano
        if hour < 11.0:
            use = "felt_piano"
        elif plan.at(plan.evening, second) > 0.5 and rng.random() < 0.35:
            use = "tx_ep"
        degree_ = weighted(rng, degrees, weights) + centre_at(second)
        octave = weighted(rng, (3, 4, 5), (0.25, 0.5, 0.25))
        vel = 0.24 + 0.2 * float(rng.random()) + 0.08 * plan.at(plan.bright, second)
        dur = float(rng.uniform(3.0, 5.5))
        notes.append(
            Note(second, dur, float(scale.midi(degree_, octave)), vel, use, "piano", patch)
        )
        if rng.random() < 0.2:
            other = degree_ + int(weighted(rng, (2, 4, 3), (0.4, 0.4, 0.2)))
            notes.append(
                Note(
                    second + 0.32,
                    dur,
                    float(scale.midi(other, octave)),
                    vel * 0.8,
                    use,
                    "piano",
                    patch,
                )
            )

    # -- the drones: a low note on the centre, overlapping, and its fifth
    second = 0.0
    while second < length:
        degree_ = centre_at(second)
        notes.append(
            Note(second, 26.0, float(scale.midi(degree_, 2)), 0.55, drone_voice, "drone", patch)
        )
        if int(second / 22.0) % 2 == 1:
            notes.append(
                Note(
                    second + 1.0,
                    24.0,
                    float(scale.midi(degree_ + 4, 2)),
                    0.4,
                    drone_voice,
                    "drone",
                    patch,
                )
            )
        second += 22.0
    drone_presence = np.clip(0.4 + 0.6 * plan.power + 0.25 * activity, 0.0, 1.0)

    # -- the strings: a triad of a string machine or a Juno pad all day, swelling in when the lights are on in the evening
    second = 8.0
    while second < length:
        # a quiet triad of the string machine plays all day; the lights and the evening make it warmer and fuller
        warm = max(
            plan.at(plan.lights, second) * plan.at(plan.evening, second),
            STRINGS_FLOOR * (1.0 - 0.6 * plan.at(night, second)),
        )
        gap_low = STRINGS_GAP_QUIET[0] + (STRINGS_GAP_BUSY[0] - STRINGS_GAP_QUIET[0]) * energy
        gap_high = STRINGS_GAP_QUIET[1] + (STRINGS_GAP_BUSY[1] - STRINGS_GAP_QUIET[1]) * energy
        second += float(rng.uniform(gap_low, gap_high))
        if form[section_of(second)] < STRINGS_GATE:
            continue
        if warm > 0.2 and rng.random() < 0.5 + 0.5 * warm:
            root = centre_at(second)
            strings = "string_machine" if patch == 0 else "juno_pad"
            for k, offset in enumerate((0, 2, 4)):
                notes.append(
                    Note(
                        second + 0.15 * k,
                        22.0,
                        float(scale.midi(root + offset, 3 + (k == 2))),
                        0.3 + 0.2 * warm,
                        strings,
                        "strings",
                        patch,
                    )
                )

    # -- a growl under a heavy load, rarely in ambient: on the busiest days only, and only when the power is high
    growl_day = energy >= GROWL_ENERGY
    growl_presence = plan.gate(plan.power, 0.6, 0.9) * (1.0 if growl_day else 0.0)
    if growl_day and growl_presence.max() > 0.3:
        second = 0.0
        while second < length:
            if plan.at(growl_presence, second) > 0.3:
                notes.append(
                    Note(
                        second,
                        14.0,
                        float(scale.midi(centre_at(second), 1)),
                        0.5,
                        "reese",
                        "growl",
                        1,
                    )
                )
            second += 16.0

    # -- anomalies: a swell of glass pads on the new centre, and a dip of the brightness
    shape_kind = swells(plan)
    for second, strength in plan.events:
        root = centre_at(second + 1.0)
        for k, offset in enumerate((0, 2, 4, 6)):
            notes.append(
                Note(
                    second + 0.4 * k,
                    18.0,
                    float(scale.midi(root + offset, 3 + k // 3)),
                    0.4 + 0.3 * strength,
                    "glass_pad",
                    "pads",
                    patch,
                )
            )

    notes = [n for n in notes if n.t < length]
    # -- the layers
    cutoff = (
        480.0
        * 12.0 ** np.clip(0.15 + 0.85 * plan.bright + 0.25 * shape_kind, 0.0, 1.2)
        * dips(plan, 0.7)
    )
    cutoff = cutoff + block_walk(rng, length, 50.0, 120.0)
    layers = {
        "pads": Layer("pads", 1.0, None, cutoff, 0.12, reverb=0.85, title="tape-loop pads"),
        "piano": Layer("piano", 1.2, None, None, 0.0, reverb=0.7, echo=0.12, title="felt piano"),
        "drone": Layer("drone", 0.9, drone_presence, None, 0.0, reverb=0.35, title="drone"),
        "strings": Layer(
            "strings",
            0.8,
            plan.lift(plan.gate(plan.lights * plan.evening, 0.15, 0.5), STRINGS_FLOOR),
            None,
            0.0,
            reverb=0.8,
            title="strings",
        ),
        "growl": Layer("growl", 0.5, growl_presence, None, 0.0, reverb=0.2, title="growl"),
    }
    space = np.clip(0.3 + 0.3 * plan.humid + 0.25 * shape_kind, 0.0, 0.95)
    score = Score(
        length,
        notes,
        layers,
        [
            Texture("room", "room", 0.004),
        ],
        space,
        echo_seconds=0.62,
        echo_feedback=0.35,
        events=list(plan.events),
        shimmer=0.22,
        compress_db=-24.0,
        level_db=-7.0,
    )
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{loops} tape loops of {', '.join(f'{lengths[int(i)]:.1f}' for i in order)} seconds (no two share a measure), on {voice_a} and {voice_b}, "
            f"pitch set '{set_name}', harmonic centres '{centre_name}' ({sum(1 for _ in centres)} moves)."
        ),
        f"Piano '{piano}' as a Poisson process (about {PIANO_BASE[0] + (PIANO_BASE[1] - PIANO_BASE[0]) * energy + (PIANO_ACTIVITY[0] + (PIANO_ACTIVITY[1] - PIANO_ACTIVITY[0]) * energy) * float(np.mean(activity)):.2f} a second at the mean activity), drone on {drone_voice}.",
    ]
    return score
