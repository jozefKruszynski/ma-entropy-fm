"""
Minimal: a hypnotic 4/4 with a soft deep kick, shuffled percussion, a short figure that evolves, a rolling bass.

The generative system has three parts. The percussion is made of Euclidean rhythms with a probability on every
step and a shuffle, so that it is always the same groove and never the same bar. The figure is a composed cell of
a few notes that mutates by rule (a step moves a degree, an accent moves, a note drops out or comes back) while
its filter, its echo and its level walk slowly at random. And the arrangement is a state machine: every 8 or 16
bars one element comes in or goes out (kick, hats, clicks, percussion, shaker, figure, bass, acid line, pad),
towards a number of elements that the activity of the house sets, so the piece is built by subtraction and addition.

How the house steers it: activity sets how many elements play and how fast the figure mutates, the power of
appliances brings in the growling bass (more often than in any other style) and the acid line, the
temperature and the sun open the filters, humidity sets the room and the echo, the lights and the evening bring in
a quiet pad, a running dishwasher holds the clicks and the percussion in, a washer the acid line, and an
anomaly is a dip of the filters with an echo throw.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .gen import block_walk, dips, euclid, swells, swung, weighted
from .score import CR, Layer, Note, Score

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
# rolling bass: semitones from the root, None for a rest, one per step
BASS_PATTERNS: tuple[tuple[str, tuple[int | None, ...]], ...] = (
    ("steady roll", (0, None, 0, 0, None, 0, None, 0, 0, None, 0, 0, None, 0, 12, None)),
    ("off-beat", (None, 0, None, 0, None, 0, 0, None, None, 0, None, 0, None, 0, None, 10)),
    ("octave pulse", (0, None, 12, None, 0, None, 12, None, 0, None, 12, None, 0, None, 10, None)),
    ("dotted", (0, None, None, 0, None, None, 0, None, 0, None, None, 0, None, 7, None, None)),
)
# the percussion lines: (hits, steps, rotation) of the Euclidean rhythm of each
PERCUSSION: tuple[
    tuple[str, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], ...
] = (
    ("classic", (5, 16, 2), (3, 8, 1), (7, 16, 0)),
    ("sparse", (3, 16, 4), (2, 8, 3), (5, 16, 1)),
    ("busy", (7, 16, 1), (5, 8, 0), (9, 16, 2)),
    ("syncopated", (5, 16, 3), (3, 8, 2), (7, 16, 5)),
)
GROWL_PATTERNS = (
    ((0, 5), (6, 2), (10, 4)),
    ((0, 3), (4, 3), (8, 3), (12, 3)),
    ((2, 4), (8, 4), (14, 2)),
)
FIGURE_VOICES = ("blip_synth", "chord_stab")
BASS_VOICES = ("sub_round", "moog_bass")
GROWLS = ("reese", "fm_growl")
PADS = ("string_machine", "juno_pad", "glass_pad")
KICKS = ("kick_soft", "kick_deep")
TEMPOS = (118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0)
ELEMENTS = ("kick", "hat", "click", "perc", "shaker", "figure", "bass", "acid", "pad")
# how readily each element comes in, and goes out
ENTER = {
    "kick": 3.0,
    "hat": 2.0,
    "click": 1.5,
    "perc": 1.5,
    "shaker": 1.0,
    "figure": 2.5,
    "bass": 2.0,
    "acid": 0.6,
    "pad": 0.7,
}
LEAVE = {
    "kick": 0.15,
    "hat": 1.0,
    "click": 1.2,
    "perc": 1.2,
    "shaker": 1.4,
    "figure": 0.6,
    "bass": 0.5,
    "acid": 1.5,
    "pad": 1.2,
}


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Minimal soundscape.

    :param plan: The curves and the deals of the day.
    """
    scale = plan.scale
    length = plan.length
    rng = plan.rng("minimal")
    bpm = plan.deal("tempo", TEMPOS)
    beat = 60.0 / bpm
    step = beat / 4.0
    bar = beat * 4.0
    swing = 0.1 + 0.08 * float(plan.rng("swing").random())
    fig_name, figure = plan.deal("figure", FIGURES)
    bass_name, bass_pattern = plan.deal("bass pattern", BASS_PATTERNS)
    perc_name, hat_rhythm, click_rhythm, shaker_rhythm = plan.deal("percussion", PERCUSSION)
    figure_voice = plan.deal("figure voice", FIGURE_VOICES)
    bass_voice = plan.deal("bass voice", BASS_VOICES)
    growl_voice = plan.deal("growl bass", GROWLS)
    pad_voice = plan.deal("pad", PADS)
    kick = plan.deal("kick", KICKS)
    growl_pattern = plan.deal("growl pattern", GROWL_PATTERNS)
    patch = int(plan.rng("patch").integers(0, 3))
    plan.choices["patch"] = str(patch)
    activity, power = plan.activity, plan.power
    bars = int(length / bar)

    # -- the arrangement: a state machine that adds or removes an element every 8 or 16 bars
    def forced(name: str, second: float) -> bool:
        return (name in {"click", "perc"} and plan.at(plan.runs["dish"], second) > 0.4) or (
            name == "acid" and plan.at(plan.runs["wash"], second) > 0.4
        )

    on = {name: name in {"kick", "figure"} for name in ELEMENTS}
    timeline: dict[str, list[tuple[int, int]]] = {name: [] for name in ELEMENTS}
    started = {name: (0 if on[name] else -1) for name in ELEMENTS}
    boundary = 0
    moves: list[str] = []
    while boundary < bars:
        second = boundary * bar
        target = 2.0 + 5.0 * plan.at(activity, second) * (1.0 - 0.3 * plan.at(plan.night, second))
        count = sum(on.values())
        if count < round(target) or (count == round(target) and rng.random() < 0.3):
            off = [n for n in ELEMENTS if not on[n]]
            if off:
                pick = weighted(rng, off, [ENTER[n] for n in off])
                on[pick] = True
                started[pick] = boundary
                moves.append(f"{pick} in at bar {boundary}")
        else:
            live = [n for n in ELEMENTS if on[n] and not forced(n, second)]
            if live:
                pick = weighted(rng, live, [LEAVE[n] for n in live])
                on[pick] = False
                timeline[pick].append((started[pick], boundary))
                moves.append(f"{pick} out at bar {boundary}")
        for name in ELEMENTS:
            if forced(name, second) and not on[name]:
                on[name] = True
                started[name] = boundary
        boundary += int(rng.choice([8, 16]))
    for name in ELEMENTS:
        if on[name]:
            timeline[name].append((started[name], bars + 1))

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
    growl_on = plan.rng("growl").random() < 0.85
    acid_pattern = [0, None, 0, 12, None, 0, 10, None, 0, None, 12, 0, None, 7, None, 0]

    for bar_index in range(bars):
        second = bar_index * bar
        if bar_index % 4 == 0 and bar_index > 0:
            # the figure mutates by rule, faster when the house is lively
            rate = 0.15 + 0.5 * plan.at(activity, second)
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
            if value is not None and plan.at(presence["figure"], at) > 0.05:
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
        # a pad in the evening, quiet, one chord every 8 bars
        if (
            bar_index % 8 == 0
            and plan.at(plan.lights * plan.evening, second) > 0.2
            and plan.at(presence["pad"], second) > 0.05
        ):
            for k, degree in enumerate((0, 2, 4)):
                notes.append(
                    Note(
                        second + 0.1 * k,
                        bar * 8,
                        float(scale.midi(degree + (3 if bar_index % 16 == 8 else 0), 3 + (k == 2))),
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
            f"{bpm:.0f} BPM, shuffle {swing:.2f}; figure '{fig_name}' on {figure_voice} (mutated {mutated} times), bass '{bass_name}' on {bass_voice}, "
            f"percussion '{perc_name}', kick {kick}, growl {growl_voice}, pad {pad_voice}."
        ),
        "Arrangement moves: " + ", ".join(moves[:10]) + (" ..." if len(moves) > 10 else ""),
    ]
    return score
