"""
Berlin School: hypnotic sequences of different lengths that interlock and mutate by rule, over strings and an organ.

The generative system is a step sequencer with rules. Three sequencer lines run on the same 16th-note grid but
have different lengths (16, 12 and 8 or 7 steps), so they phase against each other and their combination only
repeats after a long time. The cells are composed (the vocabulary of Phaedra and Rubycon: octave jumps, a held
root, arpeggios); the rules change them over minutes: every four bars each line may drop a step, add one from the
chord, move one by a degree or an octave, rotate, or be pulled back towards its first form, and the whole
sequence is transposed when the chord root moves (a progression of the day, every eight bars). Strings and
choir hold the chord, a drawbar organ is the bed, and a slow lead plays composed phrases of long notes into a
tape echo. There is no drum kit; when the house is busy in the morning and the day (never at night, gone before the
evening) a soft kick, warm toms and a light shaker follow the grid of the sequencer, far below it.

How the house steers it: activity brings the lines in and sets how fast they mutate, the power of appliances
opens the filter and sets the accents, brings the pulse and, when high, the growling bass; the temperature and
the sun move the cutoff; humidity sets the room and the echo; the lights and the evening bring in the
choir and the string swell; a running dishwasher fades in the second line and a washer the third; an anomaly is a
dip of the filter with an echo throw, a swell of the strings and a step of the progression.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .gen import block_walk, dips, swells, weighted
from .score import CR, Layer, Note, Score

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
# chord roots of the day, as scale degrees, one for every eight bars
PROGRESSIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("i VII VI VII", (0, 6, 5, 6)),
    ("i VI VII i", (0, 5, 6, 0)),
    ("i iv v iv", (0, 3, 4, 3)),
    ("i III VII iv", (0, 2, 6, 3)),
    ("i VII iv VI", (0, 6, 3, 5)),
    ("i v VI iv", (0, 4, 5, 3)),
)
# composed lead phrases: (degree from the chord root, length in beats)
LEAD_PHRASES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("rising", ((4, 8), (5, 4), (7, 12))),
    ("falling", ((7, 8), (6, 4), (4, 12))),
    ("held", ((2, 16), (4, 8))),
    ("turn", ((4, 6), (2, 2), (3, 8), (0, 12))),
    ("sigh", ((7, 12), (4, 4), (5, 12))),
    ("open", ((0, 16), (4, 16))),
)
SEQ_VOICES = ("moog_seq", "moog_bass")
SECOND_VOICES = ("juno_pluck", "blip_synth", "moog_seq")
LEADS = ("analog_lead", "flute_lead")
STRINGS = ("tape_strings", "string_machine")
GROWLS = ("reese", "fm_growl")
# warm electronic toms on the 16th grid: (step of the bar, which tom: 0 is the lowest of three)
TOM_FIGURES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((10, 1), (14, 0)),
    ((6, 2), (10, 1), (14, 0)),
    ((3, 2), (7, 1), (11, 0), (14, 1)),
)
TEMPOS = (104.0, 106.0, 108.0, 110.0, 112.0, 114.0)


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


# the least presence of the second and the third sequencer line, and of the choir: the base of the piece (the third
# line, the dearest voice, comes with the busy stretches of the day)
LINE_FLOORS = (0.7, 0.0)
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
    rng = plan.rng("berlin")
    bpm = plan.deal("tempo", TEMPOS)
    step = 60.0 / bpm / 4.0
    bar = step * 16.0
    name_a, cell_a = plan.deal("sequence A", CELLS_A)
    name_b, cell_b = plan.deal("sequence B", CELLS_B)
    name_c, cell_c = plan.deal("sequence C", CELLS_C)
    prog_name, progression = plan.deal("progression", PROGRESSIONS)
    voice_a = plan.deal("sequence voice", SEQ_VOICES)
    voice_b = plan.deal("second voice", SECOND_VOICES)
    lead_voice = plan.deal("lead voice", LEADS)
    strings = plan.deal("strings", STRINGS)
    growl = plan.deal("growl bass", GROWLS)
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)
    activity, power = plan.activity, plan.power
    lines = 3 if quality.max_layers >= 6 else 2
    light_tier = quality.max_layers <= 4

    # -- the progression: a chord root every eight bars, a step on at an anomaly
    changes = []
    clock, index = 0.0, 0
    jumps = sorted(second for second, _ in plan.events)
    offset = 0
    while clock < length:
        changes.append((clock, progression[index % len(progression)] + offset))
        for jump in jumps:
            if clock <= jump < clock + bar * 8:
                offset += 2
        index += 1
        clock += bar * 8
    changes.append((float(length) + bar * 8, 0))

    def root_at(second: float) -> int:
        value = changes[0][1]
        for start, degree in changes:
            if start <= second:
                value = degree
        return value

    chord = (0, 2, 4, 6)

    notes: list[Note] = []
    layers: dict[str, Layer] = {}
    mutations: dict[str, int] = {}
    lines_spec = [
        ("seqA", cell_a, voice_a, 3, 0.0),
        ("seqB", cell_b, voice_b, 4, 0.12),
        ("seqC", cell_c, "moog_seq", 4, 0.5),
    ][:lines]
    for line_key, base, voice, octave, _thr in lines_spec:
        cell: list[int | None] = list(base)
        presence = np.ones(plan.count) if line_key == "seqA" else None
        if line_key == "seqB":
            # the second line is part of the base: the day moves how far it sounds, never whether; the third comes with the busy day.
            # The cheapest tier has no floor under it: the line is the dearest thing that tier plays, so it comes with the day
            presence = plan.lift(
                np.maximum(
                    plan.gate(plan.activity, 0.3, 0.55),
                    plan.runs["dish"],
                ),
                0.0 if light_tier else LINE_FLOORS[0],
            )
        if line_key == "seqC":
            presence = plan.lift(
                np.maximum(plan.gate(plan.activity, 0.55, 0.8), plan.runs["wash"]), LINE_FLOORS[1]
            )
        assert presence is not None
        # a line that is only at its floor plays on every second step: the base is a lighter line, not a missing one
        calm_line = presence if line_key != "seqA" else np.ones(plan.count)
        presence = (
            presence * (0.4 + 0.6 * (1.0 - plan.night))
            if line_key != "seqA"
            else 0.5 + 0.5 * (1.0 - 0.4 * plan.night)
        )
        total_steps = int(length / step)
        done = 0
        for global_step in range(total_steps):
            second = global_step * step
            if global_step % (16 * 4) == 0 and global_step > 0:
                rate = 0.12 + 0.45 * plan.at(activity, second) + 0.15 * plan.at(power, second)
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

    # -- strings and choir hold the chord for eight bars; the choir joins when the lights are on
    second = 0.0
    while second < length:
        root = root_at(second)
        for k, offset_ in enumerate((0, 2, 4, 6)):
            notes.append(
                Note(
                    second + 0.12 * k,
                    bar * 8 + 3.0,
                    float(scale.midi(root + offset_, 3 + (k >= 2))),
                    0.42 + 0.1 * plan.at(activity, second),
                    strings,
                    "strings",
                    patch,
                )
            )
        if quality.max_layers > 4:
            for k, offset_ in enumerate((0, 4, 7)):
                notes.append(
                    Note(
                        second + 0.2 * k,
                        bar * 8 + 3.0,
                        float(scale.midi(root + offset_, 4)),
                        0.3
                        + 0.1 * max(plan.at(plan.lights, second), plan.at(plan.evening, second)),
                        "tape_choir",
                        "choir",
                        patch,
                    )
                )
        second += bar * 8
    # -- the organ bed: the root and the fifth, long
    second = 0.0
    while second < length:
        root = root_at(second)
        notes.append(
            Note(
                second,
                bar * 8 + 2.0,
                float(scale.midi(root, 2)),
                0.5,
                "drawbar_organ",
                "bed",
                patch,
            )
        )
        notes.append(
            Note(
                second + 0.3,
                bar * 8 + 2.0,
                float(scale.midi(root + 4, 2)),
                0.36,
                "drawbar_organ",
                "bed",
                patch,
            )
        )
        second += bar * 8

    # -- the slow lead: composed phrases of long notes, one every 25 to 50 seconds when the house is up
    second = float(rng.uniform(12.0, 30.0))
    phrases_played: list[str] = []
    while second < length - 20.0:
        here = (0.3 + 0.7 * plan.at(activity, second)) * (1.0 - 0.6 * plan.at(plan.night, second))
        if here > 0.22 and rng.random() < 0.4 + 0.6 * here:
            name, phrase = weighted(rng, LEAD_PHRASES, (1.0,) * len(LEAD_PHRASES))
            phrases_played.append(name)
            clock = second
            octave = 4 + int(rng.random() < 0.3)
            for degree, beats in phrase:
                span = beats * 60.0 / bpm
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
            second = clock + float(rng.uniform(14.0, 30.0)) * (1.2 - here)
        else:
            second += 14.0

    # -- the drums of the later Berlin School (Moondawn, Force Majeure): sparse and under the sequencer, locked to its
    # 16th grid. They come in the busy stretches of the morning and the day only, never in the night, and are gone
    # before the evening. A soft kick comes first, then warm toms, then a light shaker as the house gets busier.
    drum_rng = plan.rng("drums")
    # a quiet day may get no drums at all; the toms and the shaker are chosen by the day as well
    day_mood = 0.0 if drum_rng.random() < 0.12 else 0.75 + 0.25 * float(drum_rng.random())
    toms_on = float(drum_rng.random() < 0.8)
    shaker_on = float(drum_rng.random() < 0.7)
    tom_figure = int(drum_rng.integers(0, len(TOM_FIGURES)))
    window = (1.0 - plan.gate(plan.night, 0.1, 0.5)) * (1.0 - plan.gate(plan.evening, 0.15, 0.5))
    window = window * day_mood
    pulse_gate = np.maximum(plan.gate(activity, 0.5, 0.75), 0.8 * plan.gate(power, 0.5, 0.8))
    pulse_gate = pulse_gate * window if quality.max_layers > 4 else pulse_gate * 0.0
    tom_gate = plan.gate(activity, 0.62, 0.85) * window * toms_on
    shaker_gate = plan.gate(activity, 0.72, 0.9) * window * shaker_on
    if quality.max_layers < 6:
        tom_gate, shaker_gate = tom_gate * 0.0, shaker_gate * 0.0
    beat = 60.0 / bpm
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
                if level > 0.1 and drum_rng.random() < 0.3 + 0.55 * level:
                    for hit_step, pitch in TOM_FIGURES[tom_figure]:
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

    # -- a growl: when the load is heavy, or the night is intense; held for a bar on the chord root
    growl_gate = np.maximum(
        plan.gate(power, 0.45, 0.8), plan.gate(activity * plan.night, 0.25, 0.5) * 0.7
    )
    growl_gate = growl_gate * (1.0 if plan.rng("growl").random() < 0.8 else 0.0)
    second = 0.0
    while second < length:
        if plan.at(growl_gate, second) > 0.2:
            notes.append(
                Note(
                    second,
                    bar * 2 - 0.1,
                    float(scale.midi(root_at(second), 1 + int(patch))),
                    0.6,
                    growl,
                    "growl",
                    patch,
                )
            )
        second += bar * 2

    # -- anomalies: a swell of the strings on a new step of the progression
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
    layers["strings"] = Layer(
        "strings",
        0.9,
        0.5 + 0.5 * np.maximum(plan.gate(activity, 0.1, 0.4), sweep),
        380.0 * 12.0 ** np.clip(0.3 + 0.7 * plan.bright, 0, 1) * dips(plan, 0.6, 0.9, 4.0),
        0.1,
        reverb=0.7,
        title="strings",
    )
    layers["choir"] = Layer(
        "choir",
        0.8,
        plan.lift(plan.gate(np.maximum(plan.lights, plan.evening), 0.35, 0.7), CHOIR_FLOOR),
        None,
        0.0,
        reverb=0.8,
        title="choir",
    )
    layers["bed"] = Layer(
        "bed",
        0.8,
        0.45 + 0.4 * np.clip(power + activity * 0.5, 0.0, 1.0),
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
            f"{bpm:.0f} BPM; sequence A '{name_a}' (16 steps, {voice_a}), B '{name_b}' (12 steps, {voice_b}), C '{name_c}' ({len(cell_c)} steps); "
            f"progression {prog_name}; mutations applied: {sum(mutations.values())}."
        ),
        f"Lead {lead_voice} played {len(phrases_played)} composed phrases; strings {strings}; growl {growl}.",
    ]
    return score
