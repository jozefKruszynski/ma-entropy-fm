"""
Ambient: tape loops of unrelated lengths that phase against each other, a sparse felt piano, slow drones.

The generative system is the one of Brian Eno's "Music for Airports": every loop plays one long note and
repeats after a time of its own, and the lengths share no common measure, so the loops never line up the same
way twice. A loop sometimes records a new note from the pitch set of the day, and the set itself moves slowly
between harmonic centres (a Markov chain), so the harmony changes pass by pass and no change is ever sudden.
The piano is a Poisson process that draws from the same set. There is no beat.

How the house steers it: activity brings loops in and out (more of them, and more often re-recorded), the
power of the appliances sets the drone, the temperature and the sun open the pads' filter, humidity makes the
room bigger, the lights and the evening swell the strings, and an anomaly is a swell and a new harmonic centre.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .gen import block_walk, dips, markov, swells, weighted
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
)
LOOP_VOICES = ("tape_strings", "tape_choir", "bowed_pad", "glass_pad", "juno_pad", "string_machine")
PIANOS = ("felt_piano", "grand_piano")
DRONES = ("sub_round", "bowed_pad", "drawbar_organ")
# the register of each loop: octaves of the root (4 holds middle C)
OCTAVES = (3, 4, 3, 4, 5, 3, 4)


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of an ambient soundscape.

    :param plan: The curves and the deals of the day.
    """
    scale = plan.scale
    quality = plan.quality
    length = plan.length
    rng = plan.rng("ambient")
    set_name, pitch_set = plan.deal("pitch set", SETS)
    centre_name, centre_table = plan.deal("harmonic centres", CENTRES)
    lengths = plan.deal("loop lengths", LENGTHS)
    voice_a = plan.deal("loop voice A", LOOP_VOICES)
    others = [v for v in LOOP_VOICES if v != voice_a]
    voice_b = plan.deal("loop voice B", others)
    piano = plan.deal("piano", PIANOS)
    drone_voice = plan.deal("drone voice", DRONES)
    patch = int(plan.rng("patch").integers(0, 2))
    plan.choices["patch"] = str(patch)

    busy = plan.analysis.busyness
    loops = int(np.clip(2 + round(3.0 * busy), 2, max(quality.max_layers - 2, 2)))
    activity = plan.activity
    night = plan.night

    # -- the harmonic centres: a Markov chain that waits 80 to 140 seconds, and moves at an anomaly
    centres: list[tuple[float, int]] = [(0.0, 0)]
    clock = float(rng.uniform(60.0, 110.0))
    state = 0
    shifts = {round(jump) for jump, _ in plan.events}
    while clock < length:
        state = markov(rng, state, dict(centre_table.items()))
        centres.append((clock, state))
        lit = plan.at(plan.lights, clock)
        clock += float(rng.uniform(80.0, 140.0)) * (1.0 - 0.3 * lit)
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
    thresholds = np.linspace(-0.3, 0.45, loops)
    for index, which in enumerate(order):
        loop_length = float(lengths[int(which)])
        voice = voice_a if index % 2 == 0 else voice_b
        octave = OCTAVES[index % len(OCTAVES)]
        second = float(rng.uniform(0.0, loop_length * 0.6))
        degree: int | None = None
        while second < length:
            presence = float(
                np.clip((plan.at(activity, second) - thresholds[index]) / 0.3, 0.0, 1.0)
            )
            presence = max(presence, 0.45 if index == 0 else 0.0) * (
                1.0 - 0.35 * plan.at(night, second) * (index > 0)
            )
            if degree is None or rng.random() < 0.18 + 0.25 * plan.at(plan.lights, second):
                degree = pick_degree(second, degree)
            if rng.random() < 0.15 + 0.85 * presence**0.6 and presence > 0.03:
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

    # -- the felt piano: a Poisson process drawing from the set of the day
    second = float(rng.uniform(4.0, 12.0))
    while second < length:
        night_here = plan.at(night, second)
        rate = (0.03 + 0.1 * plan.at(activity, second)) * (1.0 - 0.7 * night_here)
        second += float(rng.exponential(1.0 / max(rate, 0.01))) + 2.2
        if second >= length:
            break
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

    # -- the evening strings: a triad of a string machine or a Juno pad, swelling in when the lights are on
    second = 8.0
    while second < length:
        warm = plan.at(plan.lights, second) * plan.at(plan.evening, second)
        second += float(rng.uniform(26.0, 40.0))
        if warm > 0.3 and rng.random() < 0.5 + 0.5 * warm:
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

    # -- a growl under a heavy load, rarely in ambient: one day in four, and only when the power is high
    growl_day = plan.rng("growl").random() < 0.25
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
            plan.gate(plan.lights * plan.evening, 0.15, 0.5),
            None,
            0.0,
            reverb=0.8,
            title="evening strings",
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
        f"Piano '{piano}' as a Poisson process (about {0.03 + 0.1 * float(np.mean(activity)):.2f} a second at the mean activity), drone on {drone_voice}.",
    ]
    return score
