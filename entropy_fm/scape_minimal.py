"""
Minimal: a hypnotic 4/4 with a soft deep kick, shuffled percussion, a short figure that evolves, a rolling bass.

The generative system has three parts. The percussion is made of Euclidean rhythms with a probability on every
step and a shuffle, so that it is always the same groove and never the same bar. The figure is a composed cell of
a few notes that mutates by rule (a step moves a degree, an accent moves, a note drops out or comes back) while
its filter, its echo and its level walk slowly at random. And the arrangement follows the form of the day: every
8 bars the elements that play are the first of a fixed order (kick, figure, bass, hats, clicks, percussion, shaker,
pad, acid line), as many as the intensity of the section asks for, so the piece is built by addition and subtraction.

How the house steers it: the character of the day picks the identity: the tempo inside 118 to 124, the figure,
the bass and the percussion pattern (sparse on a quiet day, syncopated and full on a busy one), the voices (warm days
the rounder ones), the growl bass (a busy day only) and the chord rhythm and progression of the pad (fourth and fifth
moves, slow on a quiet day). Its energy sets the number of elements from three to nine and how fast the figure
mutates; its shape sets the form, and the strongest anomaly is a breakdown for a phrase. The room's motif is stated by
the blip in the first and the last peak or plateau, with the figure stepping aside. The curves of the moment work on top: the power
of appliances brings in the growl, the temperature and the sun open the filters, humidity sets the room and the echo,
the lights and the evening bring in a quiet pad, a running dishwasher holds the clicks and the percussion in, a
washer the acid line, and an anomaly is a dip of the filters with an echo throw.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .character import tempo_target
from .gen import block_walk, dips, euclid, swells, swung, weighted
from .harmony import (
    BARS_PER_PHRASE,
    GROWLS,
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
from .plan import BUILD, CLIMAX, PLATEAU, RELEASE, TEMPO_TOLERANCE
from .score import CR, Layer, Note, Score
from .styles import MINIMAL

if TYPE_CHECKING:
    from .plan import Plan

Cell = tuple[int | None, ...]
FIGURES: tuple[tuple[str, Cell], ...] = (
    (
        "off-beat three",
        (0, None, None, 0, None, 3, None, None, 0, None, None, 5, None, None, 3, None),
    ),
    (
        "rising seven",
        (0, None, 3, None, None, 0, None, 5, None, None, 0, None, 7, None, None, None),
    ),
    (
        "double and step",
        (0, 0, None, None, 3, None, 0, None, None, 5, None, None, 3, None, None, None),
    ),
    (
        "sparse stair",
        (0, None, None, None, 0, None, None, 3, None, None, 5, None, None, None, 3, None),
    ),
    ("answer", (5, None, 0, None, None, 3, None, None, 0, None, 0, None, None, 7, None, None)),
    ("tight", (0, None, 0, None, None, None, 3, None, None, 0, None, None, 5, None, None, 3)),
)
# the energy of the day (0 quiet, 1 busy) that each figure, bass pattern, percussion and growl pattern suits
FIGURE_ENERGY = {
    "sparse stair": 0.1,
    "off-beat three": 0.3,
    "answer": 0.5,
    "rising seven": 0.6,
    "double and step": 0.75,
    "tight": 0.9,
}
# rolling bass: semitones from the root, None for a rest, one per step
BASS_PATTERNS: tuple[tuple[str, tuple[int | None, ...]], ...] = (
    ("steady roll", (0, None, 0, 0, None, 0, None, 0, 0, None, 0, 0, None, 0, 12, None)),
    ("off-beat", (None, 0, None, 0, None, 0, 0, None, None, 0, None, 0, None, 0, None, 10)),
    ("octave pulse", (0, None, 12, None, 0, None, 12, None, 0, None, 12, None, 0, None, 10, None)),
    ("dotted", (0, None, None, 0, None, None, 0, None, 0, None, None, 0, None, 7, None, None)),
)
BASS_ENERGY = {"dotted": 0.15, "octave pulse": 0.4, "steady roll": 0.65, "off-beat": 0.85}
# the percussion lines: (hits, steps, rotation) of the Euclidean rhythm of each
PERCUSSION: tuple[
    tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...
] = (
    ("classic", (5, 16, 2), (3, 8, 1), (7, 16, 0)),
    ("sparse", (3, 16, 4), (2, 8, 3), (5, 16, 1)),
    ("busy", (7, 16, 1), (5, 8, 0), (9, 16, 2)),
    ("syncopated", (5, 16, 3), (3, 8, 2), (7, 16, 5)),
)
PERCUSSION_ENERGY = {"sparse": 0.1, "classic": 0.45, "syncopated": 0.7, "busy": 0.9}
GROWL_PATTERNS = (
    ("sparse", 0.3, ((2, 4), (8, 4), (14, 2))),
    ("held", 0.55, ((0, 5), (6, 2), (10, 4))),
    ("four on the floor", 0.85, ((0, 3), (4, 3), (8, 3), (12, 3))),
)
# what each voice suits as (energy 0 to 1, warmth -1 to 1)
FIGURE_VOICES: dict[str, VoiceTag] = {"blip_synth": (0.5, -0.3), "chord_stab": (0.7, 0.3)}
BASS_VOICES: dict[str, VoiceTag] = {"sub_round": (0.3, 0.5), "moog_bass": (0.6, -0.1)}
PADS: dict[str, VoiceTag] = {
    "string_machine": (0.5, 0.4),
    "juno_pad": (0.35, 0.6),
    "glass_pad": (0.4, -0.6),
}
KICKS: dict[str, VoiceTag] = {"kick_soft": (0.3, 0.4), "kick_deep": (0.7, -0.2)}
# chord roots of the pad as scale degrees, moves by a fourth or a fifth only, and how much the harmony moves
PAD_PROGRESSIONS: tuple[Progression, ...] = (
    ("i", (0,), 0.0),
    ("i iv", (0, 3), 0.3),
    ("i v", (0, 4), 0.4),
    ("i iv i v", (0, 3, 0, 4), 0.65),
    ("i iv VII iv", (0, 3, 6, 3), 0.9),
)
TEMPO_LOW, TEMPO_HIGH = MINIMAL.tempo
TEMPO_STEP = 1.0
TEMPOS = tuple(
    float(TEMPO_LOW + TEMPO_STEP * index)
    for index in range(int((TEMPO_HIGH - TEMPO_LOW) / TEMPO_STEP) + 1)
)
# the shuffle of the sixteenth notes, laid back on a quiet day and straighter on a busy one, and a little of its own
SWING = (0.18, 0.10)
SWING_DETAIL = 0.02
ELEMENTS = ("kick", "hat", "click", "perc", "shaker", "figure", "bass", "acid", "pad")
# the order in which elements join as the day gets busier, and leave as it quietens
ORDER = ("kick", "figure", "bass", "hat", "click", "perc", "shaker", "pad", "acid")
# the number of elements on a day at its quietest and its busiest; the intensity of the section sets it on a curve
ELEMENT_COUNT = (3, 9)
ELEMENT_CURVE = 1.5
# how far the activity of the moment moves the intensity of a phrase, and the night takes away
ACTIVITY_SWING = 0.3
NIGHT_SHARE = 0.3
# in a peak or plateau the fourth of every four bars of a phrase adds an element on the third and drops one on the last
MICRO_CYCLE = (0, 0, 1, -1)
BREAKDOWN_KEEPS = frozenset({"bass", "figure", "pad"})
# the figure mutates with this chance every four bars, on a quiet and on a busy day
MUTATE_RATE = (0.05, 0.85)
# the growl joins by the intensity of the section
GROWL_MIN_INTENSITY = 0.55
# the room's motif is stated four bars into the first and the last peak or plateau; the blip plays it
MOTIF_OFFSET_BARS = 4
MOTIF_VELOCITY = 0.7
MOTIF_VOICE = "blip_synth"
# the pad voices, in octave 3 and one octave above for the top one
PAD_OFFSETS = (0, 2, 4)
PAD_LIFTS = (0, 0, 7)
VOICING_OCTAVE = 3
PAD_TAIL = 2.0


def _intensity_at(spans: list[tuple[float, float, str, float]], second: float) -> tuple[str, float]:
    """Return the role of the form at a second and its intensity; a build and a release move from one section to the next."""
    for index, (start, end, role, level) in enumerate(spans):
        if start <= second < end:
            if role not in {BUILD, RELEASE}:
                return role, level
            before = spans[index - 1][3] if index else level
            after = spans[index + 1][3] if index + 1 < len(spans) else level
            progress = (second - start) / (end - start)
            return role, before + (after - before) * progress
    return spans[-1][2], spans[-1][3]


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Minimal soundscape.

    :param plan: The curves and the deals of the day.
    """
    scale = plan.scale
    length = plan.length
    character = plan.character
    energy = character.energy
    rng = plan.rng("minimal")
    bpm = plan.pick(
        "tempo",
        TEMPOS,
        [
            1.0 - abs((t - TEMPO_LOW) / (TEMPO_HIGH - TEMPO_LOW) - tempo_target(MINIMAL, character))
            for t in TEMPOS
        ],
        tolerance=TEMPO_TOLERANCE,
    )
    beat = 60.0 / bpm
    step = beat / 4.0
    bar = beat * 4.0
    phrase = BARS_PER_PHRASE * bar
    swing = (
        SWING[0] + (SWING[1] - SWING[0]) * energy + SWING_DETAIL * float(plan.rng("swing").random())
    )
    fig_name, figure = plan.pick(
        "figure", FIGURES, [1.0 - abs(FIGURE_ENERGY[n] - energy) for n, _ in FIGURES]
    )
    bass_name, bass_pattern = plan.pick(
        "bass pattern",
        BASS_PATTERNS,
        [1.0 - abs(BASS_ENERGY[n] - energy) for n, _ in BASS_PATTERNS],
    )
    perc_name, hat_rhythm, click_rhythm, shaker_rhythm = plan.pick(
        "percussion",
        PERCUSSION,
        [1.0 - abs(PERCUSSION_ENERGY[row[0]] - energy) for row in PERCUSSION],
    )
    figure_voice = pick_voice(plan, "figure voice", FIGURE_VOICES)
    bass_voice = pick_voice(plan, "bass voice", BASS_VOICES)
    growl_voice = pick_voice(plan, "growl bass", GROWLS)
    pad_voice = pick_voice(plan, "pad", PADS)
    kick = pick_voice(plan, "kick", KICKS)
    growl_name, _, growl_pattern = plan.pick(
        "growl pattern", GROWL_PATTERNS, [1.0 - abs(tag - energy) for _, tag, _ in GROWL_PATTERNS]
    )
    pad_bars = harmonic_rhythm(plan)
    pad_options = progressions_for(scale.mode, PAD_PROGRESSIONS)
    pad_name, pad_roots, _ = plan.pick(
        "pad progression", pad_options, [1.0 - abs(drive - energy) for _, _, drive in pad_options]
    )
    patch = int(plan.rng("patch").integers(0, 3))
    plan.choices["patch"] = str(patch)
    activity, power = plan.activity, plan.power
    bars = int(length / bar)
    spans = plan.sections(bar)

    # -- the arrangement: every 8 bars the first elements of the order play, as many as the form and the day ask for;
    # the strongest anomaly takes the kit out for one phrase (a breakdown) and the music builds back
    breakdown = -1
    if plan.events:
        breakdown = math.ceil(max(plan.events, key=lambda event: event[1])[0] / phrase)
    on = dict.fromkeys(ELEMENTS, False)
    timeline: dict[str, list[tuple[int, int]]] = {name: [] for name in ELEMENTS}
    started = dict.fromkeys(ELEMENTS, -1)
    moves: list[str] = []
    phrases = math.ceil(bars / BARS_PER_PHRASE)
    for index in range(phrases + 1):
        boundary = index * BARS_PER_PHRASE
        second = min(index * phrase + phrase / 2.0, length - 1.0)
        role, level = _intensity_at(spans, second)
        level = float(
            np.clip(
                (level + ACTIVITY_SWING * (plan.at(activity, second) - 0.5))
                * (1.0 - NIGHT_SHARE * plan.at(plan.night, second)),
                0.0,
                1.0,
            )
        )
        count = ELEMENT_COUNT[0] + round(
            (ELEMENT_COUNT[1] - ELEMENT_COUNT[0]) * level**ELEMENT_CURVE
        )
        if role in {CLIMAX, PLATEAU}:
            count += MICRO_CYCLE[index % len(MICRO_CYCLE)]
        count = int(np.clip(count, ELEMENT_COUNT[0], ELEMENT_COUNT[1]))
        playing = set(ORDER[:count]) if index < phrases else set()
        if index == breakdown:
            playing &= BREAKDOWN_KEEPS
            moves.append(f"breakdown at bar {boundary}")
        for name in ELEMENTS:
            if name in playing and not on[name]:
                on[name] = True
                started[name] = boundary
                moves.append(f"{name} in at bar {boundary}")
            elif name not in playing and on[name]:
                on[name] = False
                timeline[name].append((started[name], boundary))
                moves.append(f"{name} out at bar {boundary}")

    def presence_of(name: str) -> np.ndarray:
        curve = np.zeros(plan.count)
        for first, last in timeline[name]:
            a, b = int(first * bar * CR), int(last * bar * CR)
            ramp = np.clip((np.arange(plan.count) - a) / (bar * CR), 0.0, 1.0) * np.clip(
                (b - np.arange(plan.count)) / (bar * CR), 0.0, 1.0
            )
            curve = np.maximum(curve, ramp)
        return curve

    presence = {name: presence_of(name) for name in ELEMENTS}
    for name in ELEMENTS:
        if name in {"click", "perc"}:
            presence[name] = np.maximum(presence[name], plan.runs["dish"])
        if name == "acid":
            presence[name] = np.maximum(presence[name], plan.runs["wash"])
    notes: list[Note] = []
    hats = euclid(*hat_rhythm)
    clicks = euclid(*click_rhythm)
    shakers = euclid(*shaker_rhythm)
    cell: list[int | None] = list(figure)
    mutated = 0
    root = 0
    sub_root = 30 + ((scale.root) % 12)
    if sub_root < 33:
        sub_root += 12
    growl_gate = np.maximum(
        plan.gate(power, 0.4, 0.75), plan.gate(activity * plan.night, 0.3, 0.55) * 0.8
    )
    growl_on = growl_voice != NO_GROWL
    growl_gate = growl_gate * section_curve(
        plan, spans, lambda _role, level: float(level >= GROWL_MIN_INTENSITY), bar
    )
    # the room's motif: stated in the blip in the first and the last peak or plateau, the later one developed, with
    # the figure stepping aside for it
    sites = [i for i, (_, _, role, _) in enumerate(spans) if role in {CLIMAX, PLATEAU}]
    sites = motif_sites(sites or [max(range(len(spans)), key=lambda i: spans[i][3])])
    statements: list[float] = []
    statement_seconds: list[float] = []
    for number, i in enumerate(sites):
        start, end, _, _ = spans[i]
        motif = develop_motif(plan.motif(), number, motif_seed(plan))
        motif_octave = lead_octave(scale, motif)
        motif_seconds = sum(beats for _, beats in motif) * beat
        at = min(start + MOTIF_OFFSET_BARS * bar, max(end - motif_seconds, 0.0))
        statements.append(at)
        statement_seconds.append(motif_seconds)
        clock = at
        for degree, beats in motif:
            notes.append(
                Note(
                    clock,
                    beats * beat * 0.95,
                    float(scale.midi(degree, motif_octave)),
                    MOTIF_VELOCITY,
                    MOTIF_VOICE,
                    "figure",
                    0,
                )
            )
            clock += beats * beat

    plan.choices[MOTIF_CHOICE] = str(len(statements))

    def in_statement(second: float) -> bool:
        return any(
            at <= second < at + seconds
            for at, seconds in zip(statements, statement_seconds, strict=True)
        )

    acid_pattern = [0, None, 0, 12, None, 0, 10, None, 0, None, 12, 0, None, 7, None, 0]

    for bar_index in range(bars):
        second = bar_index * bar
        if bar_index % 4 == 0 and bar_index > 0:
            # the figure mutates by rule, faster when the house is lively
            rate = MUTATE_RATE[0] + (MUTATE_RATE[1] - MUTATE_RATE[0]) * (
                0.5 * energy + 0.5 * plan.at(activity, second)
            )
            if rng.random() < rate:
                notes_at = [i for i, v in enumerate(cell) if v is not None]
                rests = [i for i, v in enumerate(cell) if v is None]
                rule = weighted(
                    rng,
                    ("move", "drop", "add", "shift"),
                    (
                        0.4,
                        0.2 if len(notes_at) > 3 else 0.0,
                        0.2 if len(notes_at) < 8 else 0.0,
                        0.2,
                    ),
                )
                if rule == "move":
                    i = int(rng.choice(notes_at))
                    cell[i] = (cell[i] or 0) + int(weighted(rng, (1, -1, 2), (0.4, 0.4, 0.2)))
                elif rule == "drop":
                    cell[int(rng.choice([i for i in notes_at if i != 0] or notes_at))] = None
                elif rule == "add" and rests:
                    cell[int(rng.choice(rests))] = int(rng.choice((0, 3, 5)))
                else:
                    i = int(rng.choice(notes_at))
                    j = (i + int(rng.choice((-1, 1)))) % len(cell)
                    cell[i], cell[j] = cell[j], cell[i]
                mutated += 1
            if rng.random() < 0.2:
                # pulled back towards the first form now and then, so that it does not wander off
                diff = [i for i in range(len(cell)) if cell[i] != figure[i]]
                if diff:
                    i = int(rng.choice(diff))
                    cell[i] = figure[i]
        for k in range(16):
            at = second + swung(k, step, swing)
            beat_position = k % 4 == 0
            if beat_position and plan.at(presence["kick"], at) > 0.05:
                notes.append(
                    Note(
                        at,
                        0.3,
                        0.0,
                        0.7 * (0.6 + 0.4 * plan.at(presence["kick"], at)),
                        f"drum:{kick}",
                        "kick",
                        patch,
                    )
                )
            if hats[k % len(hats)] and plan.at(presence["hat"], at) > 0.05 and rng.random() < 0.9:
                open_hat = k % 8 == 6
                notes.append(
                    Note(
                        at,
                        0.1,
                        0.0,
                        (0.55 if k % 4 == 2 else 0.35) + float(rng.normal(0, 0.04)),
                        "drum:hat_open" if open_hat else "drum:hat_closed",
                        "hat",
                        patch % 3,
                    )
                )
            if (
                clicks[k % len(clicks)]
                and plan.at(presence["click"], at) > 0.05
                and rng.random() < 0.88
            ):
                notes.append(
                    Note(
                        at,
                        0.1,
                        0.0,
                        0.45 + 0.25 * float(rng.random()),
                        "drum:click",
                        "click",
                        patch % 3 + 3 * int(rng.integers(0, 2)),
                    )
                )
            if (
                shakers[k % len(shakers)]
                and plan.at(presence["shaker"], at) > 0.05
                and rng.random() < 0.8
            ):
                notes.append(
                    Note(at, 0.1, 0.0, 0.3 + 0.2 * float(rng.random()), "drum:shaker", "shaker", 0)
                )
            if k in (4, 12) and plan.at(presence["perc"], at) > 0.05 and rng.random() < 0.75:
                notes.append(
                    Note(
                        at + 0.004,
                        0.2,
                        0.0,
                        0.4 + 0.15 * float(rng.random()),
                        "drum:rim",
                        "perc",
                        patch % 2,
                    )
                )
            value = cell[k]
            if (
                value is not None
                and plan.at(presence["figure"], at) > 0.05
                and not in_statement(at)
            ):
                evening = plan.at(plan.evening, at) > 0.5
                voice = (
                    "chord_stab"
                    if (figure_voice == "chord_stab" or (evening and patch == 2))
                    else "blip_synth"
                )
                accent = k % 4 == 0 or (k == 3 and bar_index % 2 == 0)
                vel = (0.7 if accent else 0.5) * (0.65 + 0.35 * plan.at(plan.bright, at)) + float(
                    rng.normal(0, 0.03)
                )
                midi = float(scale.midi(value, 4 if voice == "blip_synth" else 3))
                if voice == "chord_stab":
                    for extra in (0, 7):
                        notes.append(  # noqa: PERF401
                            Note(
                                at,
                                step * 1.6,
                                midi + extra,
                                vel * (1.0 if extra == 0 else 0.7),
                                voice,
                                "figure",
                                patch % 2,
                            )
                        )
                else:
                    notes.append(Note(at, step * 0.9, midi, vel, voice, "figure", patch % 2))
            semis = bass_pattern[k]
            if semis is not None and plan.at(presence["bass"], at) > 0.05:
                notes.append(
                    Note(
                        at,
                        step * 0.9,
                        float(sub_root + semis),
                        0.7,
                        bass_voice,
                        "bass",
                        1 if bass_voice == "sub_round" else patch % 2,
                    )
                )
            acid = acid_pattern[k]
            if acid is not None and plan.at(presence["acid"], at) > 0.05:
                notes.append(
                    Note(
                        at,
                        step * 0.8,
                        float(sub_root + 12 + acid),
                        0.5 + 0.4 * float(k % 4 == 0),
                        "acid_bass",
                        "acid",
                        patch % 2,
                    )
                )
        # the growl: held notes on a pattern of the day, when the load is heavy or the night is intense
        if growl_on and plan.at(growl_gate, second) > 0.2:
            for start, long in growl_pattern:
                notes.append(
                    Note(
                        second + start * step,
                        long * step * 0.95,
                        float(sub_root + root),
                        0.75,
                        growl_voice,
                        "growl",
                        patch % 2,
                    )
                )

    # -- a pad in the evening, quiet: the chords of its progression, led so that the tones they share stay
    chord_len = pad_bars * bar
    chords = math.ceil(length / chord_len)
    pad_starts = [index * chord_len for index in range(chords)]
    pad_ends = [min(start + chord_len, float(length)) for start in pad_starts]
    pad_played = [
        plan.at(plan.lights * plan.evening, start) > 0.2 and plan.at(presence["pad"], start) > 0.05
        for start in pad_starts
    ]
    if any(pad_played):
        voicing = voice_lead(
            [pad_roots[index % len(pad_roots)] for index in range(chords)],
            PAD_OFFSETS,
            PAD_LIFTS,
            0,
        )
        for k, start, end, degree in held_runs(
            voicing,
            pad_starts,
            pad_ends,
            lambda at: round(at / bar) % BARS_PER_PHRASE == 0,
            pad_played,
        ):
            notes.append(
                Note(
                    start + 0.1 * k,
                    end - start + PAD_TAIL,
                    float(scale.midi(degree, VOICING_OCTAVE)),
                    0.38,
                    pad_voice,
                    "pad",
                    patch % 2,
                )
            )

    for second, strength in plan.events:
        # an anomaly: a throw of the figure into the echo (the dip is in the curves)
        notes.append(
            Note(second, step * 3, float(scale.midi(0, 4)), 0.8, "blip_synth", "figure", 0)
        )
        del strength
    notes = [n for n in notes if 0.0 <= n.t < length]
    # -- layers
    cut_walk = block_walk(rng, length, 45.0, 0.18)
    seconds = (np.arange(plan.count) + 0.5) / CR
    echo_walk = np.clip(0.55 + block_walk(rng, length, 30.0, 0.3), 0.1, 1.0)
    figure_cut = (
        450.0
        * 14.0
        ** np.clip(
            0.15 + 0.35 * plan.bright + 0.3 * power + cut_walk + 0.25 * np.sin(seconds / 33.0),
            0.0,
            1.0,
        )
        * dips(plan, 0.8, 0.9, 3.0)
    )
    layers = {
        "kick": Layer("kick", 1.0, presence["kick"], None, 0.0, reverb=0.04, title="kick"),
        "hat": Layer("hat", 0.8, presence["hat"], None, 0.0, reverb=0.12, title="hats"),
        "click": Layer(
            "click", 0.9, presence["click"], None, 0.0, reverb=0.2, echo=0.25, title="clicks"
        ),
        "shaker": Layer("shaker", 0.7, presence["shaker"], None, 0.0, reverb=0.1, title="shaker"),
        "perc": Layer(
            "perc", 0.8, presence["perc"], None, 0.0, reverb=0.2, echo=0.2, title="percussion"
        ),
        "figure": Layer(
            "figure",
            0.9,
            presence["figure"],
            figure_cut,
            0.3,
            reverb=0.2,
            echo=0.55,
            echo_curve=echo_walk,
            title="figure",
        ),
        "bass": Layer("bass", 1.0, presence["bass"], None, 0.0, reverb=0.0, title="rolling bass"),
        "growl": Layer(
            "growl",
            0.8,
            growl_gate if growl_on else np.zeros(plan.count),
            None,
            0.0,
            reverb=0.05,
            title="growling bass",
        ),
        "acid": Layer(
            "acid",
            0.7,
            presence["acid"],
            600.0 * 6.0 ** np.clip(0.3 + 0.5 * power + cut_walk, 0, 1) * dips(plan, 0.8, 0.9, 3.0),
            0.2,
            reverb=0.1,
            echo=0.15,
            title="acid line",
        ),
        "pad": Layer(
            "pad",
            0.7,
            presence["pad"] * plan.gate(plan.lights * plan.evening, 0.1, 0.4),
            None,
            0.0,
            reverb=0.7,
            title="pad",
        ),
    }
    sweep = swells(plan, 3.0, 6.0)
    space = np.clip(0.2 + 0.25 * plan.humid + 0.15 * sweep, 0.0, 0.7)
    score = Score(
        length,
        notes,
        layers,
        [],
        space,
        echo_seconds=beat * 0.75,
        echo_feedback=0.52,
        tempo=bpm,
        events=list(plan.events),
        compress_db=-15.0,
        shimmer=0.08,
        level_db=-13.5,
    )
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{bpm:.0f} BPM in {scale.mode}, shuffle {swing:.2f}; figure '{fig_name}' on {figure_voice} (mutated {mutated} times), bass '{bass_name}' on {bass_voice}, "
            f"percussion '{perc_name}', kick {kick}, growl {growl_voice} '{growl_name}', pad {pad_voice} '{pad_name}' every {pad_bars} bars."
        ),
        f"The room's motif was stated {len(statements)} times.",
        "Arrangement moves: " + ", ".join(moves[:10]) + (" ..." if len(moves) > 10 else ""),
    ]
    return score
