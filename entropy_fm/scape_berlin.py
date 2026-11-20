"""
Berlin School: hypnotic sequences of different lengths that interlock and mutate by rule, over strings and an organ.

The generative system is a step sequencer with rules. Three sequencer lines run on the same 16th-note grid but
have different lengths (16, 12 and 8 or 7 steps), so they phase against each other and their combination only
repeats after a long time. The cells are composed (the vocabulary of Phaedra and Rubycon: octave jumps, a held
root, arpeggios); the rules change them over minutes: every four bars each line may drop a step, add one from the
chord, move one by a degree or an octave, rotate, or be pulled back towards its first form, and the whole
sequence is transposed when the chord root moves (a progression of the day, over a tonic pedal in the organ). Strings
and choir hold the chord, led from chord to chord, and a slow lead plays composed phrases of long notes into a tape
echo, and states the motif of the room in the first and the last peak or plateau. There is no drum kit on a quiet day; on a busy one
in the morning and the day (never at night, gone before the evening) a soft kick, warm toms and a light shaker follow
the grid of the sequencer, far below it.

How the house steers it: the character of the day picks the identity: the tempo inside 102 to 116, the cells, the
voices (warm days the rounder ones), the progression (fourth and fifth moves in the mode of the day, still on a quiet
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
from typing import TYPE_CHECKING

import numpy as np

from .character import tempo_target
from .gen import block_walk, dips, swells, weighted
from .harmony import (
    BARS_PER_PHRASE,
    GROWLS,
    LEAD_OCTAVE,
    MOTIF_CHOICE,
    NO_GROWL,
    Progression,
    VoiceTag,
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
from .plan import CLIMAX, PLATEAU, TEMPO_TOLERANCE
from .score import CR, Layer, Note, Score
from .styles import BERLIN_SCHOOL

if TYPE_CHECKING:
    from .plan import Plan

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
SEQ_VOICES: dict[str, VoiceTag] = {"moog_seq": (0.7, 0.2), "moog_bass": (0.3, -0.3)}
SECOND_VOICES: dict[str, VoiceTag] = {
    "juno_pluck": (0.35, 0.6),
    "moog_seq": (0.6, 0.1),
    "blip_synth": (0.85, -0.5),
}
LEADS: dict[str, VoiceTag] = {"analog_lead": (0.65, 0.4), "flute_lead": (0.3, -0.4)}
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
# the chance in a step of four bars that a line mutates, on a quiet and on a busy day
MUTATE_RATE = (0.04, 0.9)


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
        ("drop", "add", "move", "rotate", "revert"),
        (
            0.28 if density > 0.4 else 0.02,
            0.28 if density < 0.85 and rests else 0.0,
            0.25,
            0.07,
            0.2,
        ),
    )
    if rule == "drop" and len(notes) > 2:
        cell[int(rng.choice([i for i in notes if i != 0] or notes))] = None
    elif rule == "add" and rests:
        cell[int(rng.choice(rests))] = int(rng.choice(chord)) + 7 * int(rng.random() < 0.3)
    elif rule == "move" and notes:
        index = int(rng.choice(notes))
        value = cell[index]
        if value is not None:
            cell[index] = value + int(weighted(rng, (1, -1, 7, -7), (0.35, 0.35, 0.15, 0.15)))
    elif rule == "rotate":
        cell.insert(0, cell.pop())
    else:
        diff = [i for i in range(steps) if cell[i] != base[i]]
        if not diff:
            return ""
        i = int(rng.choice(diff))
        cell[i] = base[i]
        rule = "revert"
    return rule


# the least presence of the choir, the base of the piece
CHOIR_FLOOR = 0.5
# below this strength of its gate a line plays on every second step only
THIN_BELOW = 0.85


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Berlin School soundscape.

    :param plan: The curves and the deals of the day.
    """
    scale = plan.scale
    quality = plan.quality
    length = plan.length
    character = plan.character
    energy = character.energy
    rng = plan.rng("berlin")
    bpm = plan.pick(
        "tempo",
        TEMPOS,
        [
            1.0
            - abs(
                (t - TEMPO_LOW) / (TEMPO_HIGH - TEMPO_LOW) - tempo_target(BERLIN_SCHOOL, character)
            )
            for t in TEMPOS
        ],
        tolerance=TEMPO_TOLERANCE,
    )
    step = 60.0 / bpm / 4.0
    bar = step * 16.0
    phrase = BARS_PER_PHRASE * bar
    beat = 60.0 / bpm
    name_a, cell_a = plan.pick(
        "sequence A", CELLS_A, [1.0 - abs(CELL_ENERGY[n] - energy) for n, _ in CELLS_A]
    )
    name_b, cell_b = plan.pick(
        "sequence B", CELLS_B, [1.0 - abs(CELL_ENERGY[n] - energy) for n, _ in CELLS_B]
    )
    name_c, cell_c = plan.pick(
        "sequence C", CELLS_C, [1.0 - abs(CELL_ENERGY[n] - energy) for n, _ in CELLS_C]
    )
    voice_a = pick_voice(plan, "sequence voice", SEQ_VOICES)
    voice_b = pick_voice(plan, "second voice", SECOND_VOICES)
    lead_voice = pick_voice(plan, "lead voice", LEADS)
    strings = pick_voice(plan, "strings", STRINGS)
    growl = pick_voice(plan, "growl bass", GROWLS)
    chord_len_bars = harmonic_rhythm(plan)
    candidates = progressions_for(scale.mode, PROGRESSIONS)
    prog_name, progression, _ = plan.pick(
        "progression",
        candidates,
        [1.0 - abs(drive - energy) for _, _, drive in candidates],
    )
    kit_name, _, toms_on, shaker_on = plan.pick(
        "kit", KITS, [1.0 - abs(tag - energy) for _, tag, _, _ in KITS]
    )
    tom_name, _, tom_figure = plan.pick(
        "tom figure", TOM_FIGURES, [1.0 - abs(tag - energy) for _, tag, _ in TOM_FIGURES]
    )
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)
    activity, power = plan.activity, plan.power
    lines = 3 if quality.max_layers >= 6 else 2
    light_tier = quality.max_layers <= 4
    spans = plan.sections(bar)
    intensity = section_curve(plan, spans, lambda _role, level: level, bar)

    def role_gate(minimum: float) -> np.ndarray:
        return section_curve(plan, spans, lambda _role, level: float(level >= minimum), bar)

    # -- the progression: a chord root every few bars; the strongest anomaly moves the next phrase to the relative key
    chord_len = chord_len_bars * bar
    relative_phrase = -1
    if plan.events:
        strongest = max(plan.events, key=lambda event: event[1])[0]
        relative_phrase = math.ceil(strongest / phrase)
    shift = RELATIVE_SHIFT.get(scale.mode, 0)
    changes: list[tuple[float, int]] = []
    index = 0
    while index * chord_len < length:
        in_relative = index * chord_len_bars // BARS_PER_PHRASE == relative_phrase
        root = progression[index % len(progression)] + (shift if in_relative else 0)
        changes.append((index * chord_len, root))
        index += 1
    starts = [start for start, _ in changes]

    def root_at(second: float) -> int:
        return changes[max(bisect.bisect_right(starts, second) - 1, 0)][1]

    chord = (0, 2, 4, 6)
    notes: list[Note] = []
    layers: dict[str, Layer] = {}
    mutations: dict[str, int] = {}
    lines_spec = [
        ("seqA", cell_a, voice_a, 3),
        ("seqB", cell_b, voice_b, 4),
        ("seqC", cell_c, "moog_seq", 4),
    ][:lines]
    second_gate = np.maximum(role_gate(SECOND_LINE_MIN_INTENSITY), plan.runs["dish"])
    third_gate = np.maximum(role_gate(THIRD_LINE_MIN_INTENSITY), plan.runs["wash"])
    for line_key, base, voice, octave in lines_spec:
        cell: list[int | None] = list(base)
        presence = np.ones(plan.count) if line_key == "seqA" else None
        if line_key == "seqB":
            # the second line is part of the base on a busy day, and comes with the busy stretches on a quiet one; the
            # cheapest tier has no floor under it, since the line is the dearest thing that tier plays
            floor = (
                0.0
                if light_tier
                else SECOND_LINE_FLOOR[0] + (SECOND_LINE_FLOOR[1] - SECOND_LINE_FLOOR[0]) * energy
            )
            presence = second_gate * plan.lift(
                np.maximum(plan.gate(plan.activity, 0.3, 0.55), plan.runs["dish"]), floor
            )
        if line_key == "seqC":
            floor = THIRD_LINE_FLOOR[0] + (THIRD_LINE_FLOOR[1] - THIRD_LINE_FLOOR[0]) * energy
            presence = third_gate * plan.lift(
                np.maximum(plan.gate(plan.activity, 0.55, 0.8), plan.runs["wash"]), floor
            )
        assert presence is not None
        # a line that is only at its floor plays on every second step: the base is a lighter line, not a missing one
        calm_line = presence if line_key != "seqA" else np.ones(plan.count)
        presence = (
            presence * (0.4 + 0.6 * (1.0 - plan.night))
            if line_key != "seqA"
            else (0.5 + 0.5 * (1.0 - 0.4 * plan.night))
            * (FORM_FLOOR + (1.0 - FORM_FLOOR) * intensity)
        )
        total_steps = int(length / step)
        done = 0
        for global_step in range(total_steps):
            second = global_step * step
            if global_step % (16 * 4) == 0 and global_step > 0:
                local = min(plan.at(activity, second) + 0.3 * plan.at(power, second), 1.0)
                rate = MUTATE_RATE[0] + (MUTATE_RATE[1] - MUTATE_RATE[0]) * (
                    0.5 * energy + 0.5 * local
                )
                if _mutate(rng, cell, base, rate, chord):
                    done += 1
            value = cell[global_step % len(cell)]
            if value is None:
                continue
            here = plan.at(presence, second)
            if here < 0.06:
                continue
            if (
                line_key != "seqA"
                and global_step % 2 == 1
                and plan.at(calm_line, second) < THIN_BELOW
            ):
                continue
            accent = (global_step % len(cell)) % 4 == 0
            vel = (0.62 + 0.28 * plan.at(power, second) if accent else 0.5) * (0.6 + 0.4 * here)
            vel = float(np.clip(vel + float(rng.normal(0.0, 0.03)), 0.2, 1.0))
            degree = value + root_at(second)
            notes.append(
                Note(
                    second,
                    step * 0.82,
                    float(scale.midi(degree, octave)),
                    vel,
                    voice,
                    line_key,
                    patch,
                )
            )
        mutations[line_key] = done
        layers[line_key] = Layer(
            line_key,
            1.0,
            presence,
            None,
            0.4,
            reverb=0.25,
            echo=0.3 if line_key != "seqA" else 0.16,
            title=f"sequence {line_key[-1]}",
        )

    # -- strings and choir hold the chord and are led from chord to chord: a shared tone stays, the others move by step
    roots = [root for _, root in changes]
    ends = [*starts[1:], float(length)]
    choir_gate = role_gate(CHOIR_MIN_INTENSITY)
    holds = [
        ("strings", strings, STRINGS_OFFSETS, STRINGS_LIFTS, STRINGS_LOW, 0.12, True),
        ("choir", "tape_choir", CHOIR_OFFSETS, CHOIR_LIFTS, CHOIR_LOW, 0.2, quality.max_layers > 4),
    ]
    for layer_key, voice, offsets, lifts, low, stagger, on in holds:
        if not on:
            continue
        voicing = voice_lead(roots, offsets, lifts, low)
        for k, start, end, degree in held_runs(
            voicing, starts, ends, lambda at: round(at / bar) % BARS_PER_PHRASE == 0
        ):
            swell = (
                plan.at(activity, start)
                if layer_key == "strings"
                else max(plan.at(plan.lights, start), plan.at(plan.evening, start))
            )
            notes.append(
                Note(
                    start + stagger * k,
                    end - start + HOLD_TAIL,
                    float(scale.midi(degree, VOICING_OCTAVE)),
                    (0.42 if layer_key == "strings" else 0.3) + 0.1 * swell,
                    voice,
                    layer_key,
                    patch,
                )
            )
    # -- the organ bed: the tonic and the fifth as a pedal under the moving chords, long
    second = 0.0
    while second < length:
        notes.append(
            Note(second, phrase + 2.0, float(scale.midi(0, 2)), 0.5, "drawbar_organ", "bed", patch)
        )
        notes.append(
            Note(
                second + 0.3,
                phrase + 2.0,
                float(scale.midi(4, 2)),
                0.36,
                "drawbar_organ",
                "bed",
                patch,
            )
        )
        second += phrase

    # -- the room's motif in the lead voice, in the first and the last peak or plateau of the form, the later one
    # developed; the slow phrases leave room for it
    sites = [i for i, (_, _, role, _) in enumerate(spans) if role in {CLIMAX, PLATEAU}]
    sites = motif_sites(sites or [max(range(len(spans)), key=lambda i: spans[i][3])])
    statements: list[float] = []
    taken: list[tuple[float, float]] = []
    for number, i in enumerate(sites):
        start, end, _, _ = spans[i]
        motif = develop_motif(plan.motif(), number, motif_seed(plan))
        motif_octave = lead_octave(scale, motif)
        motif_seconds = sum(beats for _, beats in motif) * beat
        at = min(start + bar, max(end - motif_seconds - bar, 0.0))
        statements.append(at)
        taken.append((at - bar, at + motif_seconds + bar))
        clock = at
        for degree, beats in motif:
            span = beats * beat
            notes.append(
                Note(
                    clock,
                    span * 0.95,
                    float(scale.midi(degree, motif_octave)),
                    MOTIF_VELOCITY,
                    lead_voice,
                    "lead",
                    patch,
                )
            )
            clock += span

    plan.choices[MOTIF_CHOICE] = str(len(statements))

    # -- the slow lead: composed phrases of long notes, on the bar, further apart the quieter the day
    lead_gate = role_gate(LEAD_MIN_INTENSITY)
    gap_bars = LEAD_GAP_BARS[0] + (LEAD_GAP_BARS[1] - LEAD_GAP_BARS[0]) * energy
    second = round(float(rng.uniform(12.0, 30.0)) / bar) * bar
    phrases_played: list[str] = []
    while second < length - 20.0:
        here = (0.3 + 0.7 * plan.at(activity, second)) * (1.0 - 0.6 * plan.at(plan.night, second))
        if plan.at(lead_gate, second) > 0.5 and here > 0.22 and rng.random() < 0.4 + 0.6 * here:
            name, lead_phrase = weighted(rng, LEAD_PHRASES, (1.0,) * len(LEAD_PHRASES))
            span_total = sum(beats for _, beats in lead_phrase) * beat
            clash = [end for begin, end in taken if begin < second + span_total and second < end]
            if clash:
                second = max(clash)
                continue
            phrases_played.append(name)
            clock = second
            octave = LEAD_OCTAVE + int(rng.random() < 0.3)
            for degree, beats in lead_phrase:
                span = beats * beat
                notes.append(
                    Note(
                        clock,
                        span * 0.95,
                        float(scale.midi(degree + root_at(clock), octave)),
                        0.5 + 0.12 * here,
                        lead_voice,
                        "lead",
                        patch,
                    )
                )
                clock += span
            second = (
                round((clock + gap_bars * bar * float(rng.uniform(0.8, 1.2)) * (1.2 - here)) / bar)
                * bar
            )
        else:
            second += bar * 2

    # -- the drums of the later Berlin School (Moondawn, Force Majeure): sparse and under the sequencer, locked to its
    # 16th grid. The kit is the day's: none on a quiet one, a kick, then toms and a shaker as the day gets busier. They
    # play in the busy stretches of the morning and the day only, never in the night, and are gone before the evening.
    drum_rng = plan.rng("drums")
    window = (1.0 - plan.gate(plan.night, 0.1, 0.5)) * (1.0 - plan.gate(plan.evening, 0.15, 0.5))
    # a busy day opens the drums at a lower activity, a quiet one only at a higher
    lower = DRUM_THRESHOLD_SHIFT * (0.5 - energy)
    has_kick = kit_name != KITS[0][0]
    pulse_gate = np.maximum(
        plan.gate(activity, 0.5 + lower, 0.75 + lower), 0.8 * plan.gate(power, 0.5, 0.8)
    )
    pulse_gate = pulse_gate * window * role_gate(PULSE_MIN_INTENSITY)
    pulse_gate = pulse_gate if has_kick and quality.max_layers > 4 else pulse_gate * 0.0
    tom_gate = (
        plan.gate(activity, 0.62 + lower, 0.85 + lower)
        * window
        * role_gate(TOMS_MIN_INTENSITY)
        * float(toms_on)
    )
    shaker_gate = (
        plan.gate(activity, 0.72 + lower, 0.9 + lower)
        * window
        * role_gate(SHAKER_MIN_INTENSITY)
        * float(shaker_on)
    )
    if quality.max_layers < 6:
        tom_gate, shaker_gate = tom_gate * 0.0, shaker_gate * 0.0
    second = 0.0
    beat_index = 0
    while second < length:
        level = plan.at(pulse_gate, second)
        # a steady quarter-note pulse; while the house is only starting to get busy, every other beat
        if level > 0.1 and (level > 0.45 or beat_index % 2 == 0):
            notes.append(
                Note(second, 0.3, 0.0, 0.28 + 0.2 * level, "drum:kick_soft", "pulse", patch)
            )
        second += beat
        beat_index += 1
    if quality.max_layers >= 6:
        total_bars = int(length / bar)
        for bar_index in range(total_bars):
            for step_index in range(16):
                at = bar_index * bar + step_index * step
                level = plan.at(tom_gate, at)
                if level > 0.1 and drum_rng.random() < 0.45 + 0.5 * level:
                    for hit_step, pitch in tom_figure:
                        if hit_step == step_index:
                            vel = 0.36 + 0.2 * level + float(drum_rng.normal(0.0, 0.03))
                            notes.append(Note(at, 0.3, 0.0, vel, "drum:tom_soft", "toms", pitch))
                if (
                    bar_index % 8 == 7
                    and step_index >= 12
                    and level > 0.35
                    and drum_rng.random() < 0.8
                ):
                    vel = 0.4 + 0.2 * level + float(drum_rng.normal(0.0, 0.03))
                    notes.append(
                        Note(
                            at,
                            0.3,
                            0.0,
                            vel,
                            "drum:tom_soft",
                            "toms",
                            (2, 1, 1, 0)[step_index - 12],
                        )
                    )
                level = plan.at(shaker_gate, at)
                if level > 0.1:
                    if step_index % 2 == 0 and drum_rng.random() < 0.9:
                        vel = (0.4 if step_index % 4 == 2 else 0.26) * (0.6 + 0.4 * level)
                        notes.append(Note(at, 0.1, 0.0, vel, "drum:shaker", "shaker", 1))
                    elif step_index % 2 == 1 and level > 0.6 and drum_rng.random() < 0.35:
                        notes.append(Note(at, 0.1, 0.0, 0.2, "drum:shaker", "shaker", 1))

    # -- a growl, on the busy days: when the load is heavy, or the night is intense; held for two bars on the tonic
    growl_gate = np.maximum(
        plan.gate(power, 0.45, 0.8), plan.gate(activity * plan.night, 0.25, 0.5) * 0.7
    )
    growl_gate = growl_gate * role_gate(GROWL_MIN_INTENSITY) * float(growl != NO_GROWL)
    second = 0.0
    while second < length:
        if plan.at(growl_gate, second) > 0.2:
            notes.append(
                Note(
                    second,
                    bar * 2 - 0.1,
                    float(scale.midi(0, GROWL_OCTAVE)),
                    0.6,
                    growl,
                    "growl",
                    patch,
                )
            )
        second += bar * 2

    # -- anomalies: a swell of the strings on the chord of the moment
    sweep = swells(plan, 4.0, 9.0)
    for second, strength in plan.events:
        for k, offset_ in enumerate((0, 2, 4)):
            notes.append(
                Note(
                    second + 0.3 * k,
                    14.0,
                    float(scale.midi(root_at(second + 2.0) + offset_, 3 + k // 2)),
                    0.5 + 0.3 * strength,
                    "tape_choir",
                    "strings",
                    patch,
                )
            )

    notes = [n for n in notes if n.t < length]
    # -- layers: cutoff sweeps of their own, moved by the house
    cut_walk = block_walk(rng, length, 40.0, 0.12)
    for i, key in enumerate(("seqA", "seqB", "seqC")[:lines]):
        period = (70.0, 110.0, 150.0)[i] * (1.0 + 0.2 * float(rng.random()))
        phase = float(rng.uniform(0, 1))
        seconds = (np.arange(plan.count) + 0.5) / CR
        triangle = 0.5 + 0.5 * np.sin(2 * np.pi * (seconds / period + phase))
        x = np.clip(0.1 + 0.25 * plan.bright + 0.25 * power + 0.3 * triangle + cut_walk, 0.0, 1.0)
        layers[key].cutoff = 380.0 * 16.0**x * dips(plan, 0.8, 0.9, 4.0)
    form = FORM_FLOOR + (1.0 - FORM_FLOOR) * intensity
    layers["strings"] = Layer(
        "strings",
        0.9,
        (0.5 + 0.5 * np.maximum(plan.gate(activity, 0.1, 0.4), sweep)) * form,
        380.0 * 12.0 ** np.clip(0.3 + 0.7 * plan.bright, 0, 1) * dips(plan, 0.6, 0.9, 4.0),
        0.1,
        reverb=0.7,
        title="strings",
    )
    layers["choir"] = Layer(
        "choir",
        0.8,
        plan.lift(plan.gate(np.maximum(plan.lights, plan.evening), 0.35, 0.7), CHOIR_FLOOR)
        * choir_gate,
        None,
        0.0,
        reverb=0.8,
        title="choir",
    )
    layers["bed"] = Layer(
        "bed",
        0.8,
        (0.45 + 0.4 * np.clip(power + activity * 0.5, 0.0, 1.0)) * form,
        None,
        0.0,
        reverb=0.3,
        title="organ bed",
    )
    layers["lead"] = Layer("lead", 0.9, None, None, 0.0, reverb=0.5, echo=0.55, title="lead")
    layers["pulse"] = Layer("pulse", 0.7, pulse_gate, None, 0.0, reverb=0.1, title="pulse")
    if quality.max_layers >= 6:
        # the toms and the shaker are rolled off, so that they are warm and have no click
        layers["toms"] = Layer(
            "toms",
            0.7,
            tom_gate,
            1800.0 * dips(plan, 0.8, 0.9, 4.0),
            0.0,
            reverb=0.25,
            title="toms",
        )
        layers["shaker"] = Layer(
            "shaker",
            0.6,
            shaker_gate,
            6500.0 * dips(plan, 0.8, 0.9, 4.0),
            0.0,
            reverb=0.2,
            title="shaker",
        )
    layers["growl"] = Layer("growl", 0.7, growl_gate, None, 0.0, reverb=0.15, title="growl bass")
    space = np.clip(0.3 + 0.35 * plan.humid + 0.2 * sweep, 0.0, 0.95)
    score = Score(
        length,
        notes,
        layers,
        [],
        space,
        echo_seconds=step * 3.0,
        echo_feedback=0.5,
        tempo=bpm,
        events=list(plan.events),
        shimmer=0.1,
        level_db=-19.6,
    )
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{bpm:.0f} BPM in {scale.mode}; sequence A '{name_a}' (16 steps, {voice_a}), B '{name_b}' (12 steps, {voice_b}), C '{name_c}' ({len(cell_c)} steps); "
            f"progression {prog_name}, a chord every {chord_len_bars} bars over a tonic pedal; kit '{kit_name}', toms '{tom_name}'; mutations applied: {sum(mutations.values())}."
        ),
        f"Lead {lead_voice} played the motif {len(statements)} times and {len(phrases_played)} composed phrases; strings {strings}; growl {growl}.",
    ]
    return score
