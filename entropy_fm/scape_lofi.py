"""
Lo-fi: a dusty swung boom-bap loop, electric piano on jazzy progressions, a round bass, vinyl and tape.

The generative system is a band that plays a composed loop and never plays it the same way twice. The day deals
a progression of four chords with sevenths, ninths and elevenths, a comping rhythm, a bass pattern and a drum pattern;
every chord is voiced a little differently each time (which notes, which order, how late), and the band varies
the loop by rule: a chord is sometimes swapped for a related one (a tritone substitution, a relative), the comping
rhythm drifts to a neighbour, ghost notes and fills come and go with probabilities that follow the house, and a
chord chop or a plucked line is played now and then. At night the drums rest and the loop becomes an ambient
piece of electric piano and vinyl.

How the house steers it: activity sets the density of the drums and the comping and how often the loop is varied,
the power of appliances sets the weight of the bass, the temperature and the sun set the tape low-pass, humidity the
room, the lights and the evening the extensions of the chords and a swell of strings, a running dishwasher brings in
the drum loop, and an anomaly closes the tape low-pass and opens it again (a filter dip, never a pitch glide).

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .gen import block_walk, dips, swells, swung, weighted
from .score import Layer, Note, Score, Texture

if TYPE_CHECKING:
    from .plan import Plan

# the sound of a chord above its root, in semitones, as an electric piano plays it
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
PROGRESSIONS: tuple[tuple[str, tuple[Chord, ...]], ...] = (
    ("ii V I vi", ((2, "m9"), (7, "dom9"), (0, "maj9"), (9, "m9"))),
    ("i iv VII III", ((0, "m9"), (5, "m11"), (10, "dom9"), (3, "maj9"))),
    ("I IV iii vi", ((0, "maj9"), (5, "maj9"), (4, "m9"), (9, "m9"))),
    ("i VI ii V", ((0, "m9"), (8, "maj9"), (2, "m11"), (7, "dom13"))),
    ("IV iii ii I", ((5, "maj9"), (4, "m7"), (2, "m9"), (0, "maj9"))),
    ("i VII VI V", ((0, "m11"), (10, "dom9"), (8, "maj9"), (7, "dom13"))),
    ("I vi ii V", ((0, "maj9"), (9, "m9"), (2, "m9"), (7, "dom9"))),
    ("vi IV I V", ((9, "m9"), (5, "maj9"), (0, "maj9"), (7, "dom13"))),
)
# substitutes for a chord: (semitones from its root, quality)
SUBSTITUTES = {
    "dom9": (6, "dom9"),
    "dom13": (6, "dom9"),
    "maj9": (9, "m9"),
    "m9": (3, "maj9"),
    "m7": (3, "maj9"),
    "m11": (3, "maj9"),
}
# comping: (start step of 16, length in steps, relative velocity)
Comp = tuple[tuple[int, int, float], ...]
COMPS: tuple[tuple[str, Comp], ...] = (
    ("held", ((0, 12, 1.0),)),
    ("push", ((0, 6, 1.0), (6, 3, 0.7), (10, 6, 0.85))),
    ("two and four", ((2, 4, 0.9), (8, 4, 1.0), (12, 3, 0.7))),
    ("stabs", ((0, 2, 1.0), (6, 2, 0.7), (8, 2, 0.9), (14, 2, 0.6))),
    ("syncopated", ((0, 3, 1.0), (3, 3, 0.75), (8, 6, 0.9))),
    ("late", ((3, 5, 0.95), (10, 5, 0.8))),
)
# bass: (start step, length in steps, 'r' root, 'f' fifth, 'o' octave)
Bass = tuple[tuple[int, int, str], ...]
BASSES: tuple[tuple[str, Bass], ...] = (
    ("walking root", ((0, 6, "r"), (10, 2, "f"), (14, 2, "r"))),
    ("round", ((0, 3, "r"), (3, 1, "r"), (6, 2, "f"), (10, 2, "r"), (13, 2, "o"))),
    ("long and short", ((0, 4, "r"), (8, 2, "r"), (11, 2, "f"), (14, 2, "r"))),
    ("step", ((0, 2, "r"), (4, 2, "r"), (8, 2, "f"), (10, 2, "r"), (14, 1, "o"))),
)
# drums on a 16-step bar: kick steps, snare steps, ghost steps (a soft snare or a rim), hat accents
Drum = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
DRUMS: tuple[tuple[str, Drum], ...] = (
    ("classic", ((0, 7, 10), (4, 12), (15,))),
    ("lazy", ((0, 3, 10), (4, 12), (9, 14))),
    ("late kick", ((0, 5, 8, 11), (4, 12), ())),
    ("rolling", ((0, 6, 10, 13), (4, 12), (9,))),
    ("shuffle", ((0, 8, 11), (4, 12), (14,))),
    ("half time", ((0, 10), (8,), (6, 15))),
)
KEYS = ("tx_ep", "fm_ep")
BASS_VOICES = ("sub_round", "moog_bass")
PLUCKS = ("strum_pluck", "nylon_ks")
TEMPOS = (72.0, 75.0, 78.0, 80.0, 82.0, 85.0, 88.0)


def _voicing(scale_root: int, chord: Chord, extended: bool) -> list[int]:  # noqa: FBT001
    """Return the notes of a chord above its root, closed, centred on the middle of the electric piano."""
    offset, quality = chord
    name = EXTENDED[quality] if extended else quality
    root = scale_root + offset
    base = 48 + (root % 12)
    if base + 3 < 53:
        base += 12
    return [base + i for i in QUALITIES[name]]
    # drop the highest tone an octave for a wider voicing now and then; the caller decides by chance


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Lo-fi soundscape.

    :param plan: The curves and the deals of the day.
    """
    quality = plan.quality
    length = plan.length
    rng = plan.rng("lofi")
    bpm = plan.deal("tempo", TEMPOS)
    beat = 60.0 / bpm
    step = beat / 4.0
    bar = beat * 4.0
    prog_name, progression = plan.deal("progression", PROGRESSIONS)
    comp_name, comp = plan.deal("comping", COMPS)
    bass_name, bass_pattern = plan.deal("bass pattern", BASSES)
    drum_name, drum = plan.deal("drum pattern", DRUMS)
    keys_voice = plan.deal("keys", KEYS)
    bass_voice = plan.deal("bass voice", BASS_VOICES)
    pluck_voice = plan.deal("pluck", PLUCKS)
    patch = int(plan.rng("patch").integers(0, 3))
    plan.choices["patch"] = str(patch)
    swing = 0.18 + 0.1 * float(plan.rng("swing").random())
    root = plan.scale.root
    activity, power = plan.activity, plan.power
    chord_bars = 1 if plan.rng("chord bars").random() < 0.55 else 2

    drums_on = np.maximum(plan.gate(activity, 0.22, 0.5), plan.runs["dish"]) * (
        1.0 - 0.85 * plan.night
    )
    notes: list[Note] = []
    variations = {"swap": 0, "comp": 0, "chop": 0, "pluck": 0}
    current = list(progression)
    comp_now = comp
    bars = int(length / bar)
    chord_index = 0
    for bar_index in range(bars):
        second = bar_index * bar
        if bar_index % (4 * chord_bars) == 0 and bar_index > 0:
            # the loop is varied by rule: a chord may be swapped for a related one, the comping may drift
            lit = plan.at(plan.lights, second)
            if rng.random() < 0.12 + 0.3 * plan.at(activity, second):
                k = int(rng.integers(0, len(current)))
                off, qual = current[k]
                sub = SUBSTITUTES.get(qual)
                if sub is not None and current[k] == progression[k]:
                    current[k] = (off + sub[0], sub[1])
                    variations["swap"] += 1
                else:
                    current[k] = progression[k]
            if rng.random() < 0.25:
                comp_now = weighted(rng, [c for _, c in COMPS], (1.0,) * len(COMPS))
                variations["comp"] += 1
            del lit
        if bar_index % chord_bars != 0:
            continue
        chord = current[chord_index % len(current)]
        chord_index += 1
        here_drums = plan.at(drums_on, second)
        extended = plan.at(plan.lights, second) > 0.5 or plan.at(plan.evening, second) > 0.55
        voicing = _voicing(root, chord, extended)
        # with the drums at rest the electric piano holds the chord, a quiet piece
        pattern = comp_now if here_drums > 0.15 else COMPS[0][1]
        span = chord_bars * bar
        for start, long, rel in pattern:
            if rng.random() > 0.55 + 0.45 * (0.4 + plan.at(activity, second)) and long < 6:
                continue
            at = second + swung(start, step, swing) + float(rng.normal(0.0, 0.012))
            hold = min(long * step * 1.1, span - start * step) if here_drums > 0.15 else span * 0.95
            order = list(voicing)
            if rng.random() < 0.3:
                order.reverse()
            for k, midi in enumerate(order):
                vel = float(
                    np.clip(
                        (0.5 + 0.12 * rel) * (0.85 + 0.15 * plan.at(plan.bright, second))
                        + rng.normal(0.0, 0.05),
                        0.2,
                        0.95,
                    )
                )
                notes.append(
                    Note(
                        at + k * float(rng.uniform(0.008, 0.028)),
                        hold,
                        float(midi),
                        vel,
                        keys_voice,
                        "keys",
                        patch,
                    )
                )
        # the bass follows the root of the chord
        root_midi = 28 + ((root + chord[0]) % 12)
        if root_midi < 31:
            root_midi += 12
        for b in range(chord_bars):
            for start, long, tone in bass_pattern:
                offset = {"r": 0, "f": 7, "o": 12}[tone]
                at = second + b * bar + swung(start, step, swing)
                if rng.random() < 0.08:
                    continue
                notes.append(
                    Note(
                        at + float(rng.normal(0.0, 0.006)),
                        max(long * step * 0.9, 0.12),
                        float(root_midi + offset),
                        float(
                            np.clip(
                                0.62 + 0.15 * plan.at(power, second) + rng.normal(0, 0.04),
                                0.3,
                                0.95,
                            )
                        ),
                        bass_voice,
                        "bass",
                        1 if bass_voice == "sub_round" else patch % 2,
                    )
                )
        # drums
        for b in range(chord_bars):
            base = second + b * bar
            level = plan.at(drums_on, base)
            if level < 0.12:
                continue
            kicks, snares, ghosts = drum
            for k in kicks:
                if k == 0 or rng.random() < 0.9:
                    notes.append(  # noqa: PERF401
                        Note(
                            base + swung(k, step, swing) + float(rng.normal(0.0, 0.006)),
                            0.3,
                            0.0,
                            0.55 + 0.25 * level,
                            "drum:kick_lofi",
                            "drums",
                            patch % 3,
                        )
                    )
            for k in snares:
                notes.append(  # noqa: PERF401
                    Note(
                        base + k * step + float(rng.normal(0.0, 0.01)) + 0.012,
                        0.3,
                        0.0,
                        0.55 + 0.3 * level + float(rng.normal(0, 0.04)),
                        "drum:snare_lofi",
                        "drums",
                        patch % 3,
                    )
                )
            for k in ghosts:
                if rng.random() < 0.3 + 0.5 * plan.at(activity, base):
                    notes.append(  # noqa: PERF401
                        Note(
                            base + swung(k, step, swing),
                            0.2,
                            0.0,
                            0.25 + 0.1 * float(rng.random()),
                            "drum:rim",
                            "drums",
                            patch % 2,
                        )
                    )
            skip = 0.3 * (1.0 - plan.at(activity, base))
            for k in range(16):
                if k % 2 == 1 and rng.random() < 0.5 + skip * 0.5 and k % 4 != 3:
                    continue
                if k % 2 == 1 and k % 4 == 1 and level < 0.5:
                    continue
                accent = 1.0 if k % 4 == 0 else 0.75 if k % 2 == 0 else 0.45
                hit_at = swung(k, step, swing)
                notes.append(
                    Note(
                        base + hit_at + float(rng.normal(0.0, 0.006)),
                        0.1,
                        0.0,
                        accent * (0.35 + 0.25 * level) + float(rng.normal(0, 0.03)),
                        "drum:hat_closed",
                        "drums",
                        patch % 3,
                    )
                )
        # ear candy: a chord chop now and then, and once in a while a plucked line
        if here_drums > 0.3 and rng.random() < 0.05 + 0.1 * plan.at(activity, second):
            variations["chop"] += 1
            at = second + float(rng.choice([6, 10, 14])) * step
            for k, midi in enumerate(voicing):
                notes.append(
                    Note(at + k * 0.006, 0.12, float(midi + 12), 0.5, keys_voice, "chop", patch)
                )
        if bar_index % 12 == 6 and plan.at(activity, second) > 0.3 and rng.random() < 0.7:
            variations["pluck"] += 1
            scale_tones = (
                [voicing[0] + 12 + o for o in (0, 2, 3, 7, 10)]
                if chord[1].startswith("m")
                else [voicing[0] + 12 + o for o in (0, 2, 4, 7, 9)]
            )
            at = second
            for k in range(int(rng.integers(3, 6))):
                notes.append(  # noqa: PERF401
                    Note(
                        at + k * beat * 0.5,
                        0.5,
                        float(scale_tones[int(rng.integers(0, len(scale_tones)))]),
                        0.5,
                        pluck_voice,
                        "pluck",
                        patch % 2,
                    )
                )
        # the evening strings: a soft triad swelling in when the lights are on
        warm = plan.at(plan.lights, second) * plan.at(plan.evening, second)
        if bar_index % 8 == 0 and warm > 0.3:
            strings = "juno_pad" if patch % 2 else "string_machine"
            for k, midi in enumerate(voicing[:3]):
                notes.append(
                    Note(
                        second + 0.1 * k,
                        bar * 8,
                        float(midi),
                        0.3 + 0.2 * warm,
                        strings,
                        "strings",
                        patch % 2,
                    )
                )

    notes = [n for n in notes if 0.0 <= n.t < length]
    # -- layers
    walk = block_walk(rng, length, 60.0, 0.1)
    tone = np.clip(0.35 + 0.5 * plan.bright + walk, 0.0, 1.0)
    keys_cut = 1500.0 * 3.4**tone * dips(plan, 0.88, 1.0, 4.0)
    drums_cut = 3500.0 * 3.0**tone * dips(plan, 0.7, 1.0, 3.0)
    sweep = swells(plan, 3.0, 6.0)
    layers = {
        "keys": Layer("keys", 1.0, None, keys_cut, 0.12, reverb=0.3, title="electric piano"),
        "bass": Layer(
            "bass",
            1.0,
            0.7 + 0.3 * np.clip(power + activity * 0.4, 0, 1),
            None,
            0.0,
            reverb=0.0,
            highpass=0.0,
            title="bass",
        ),
        "drums": Layer("drums", 0.85, None, drums_cut, 0.1, reverb=0.1, title="drums"),
        "chop": Layer(
            "chop", 0.8, None, keys_cut * 0.6, 0.2, reverb=0.35, echo=0.2, title="chord chops"
        ),
        "pluck": Layer("pluck", 0.8, None, None, 0.0, reverb=0.4, echo=0.15, title="plucked line"),
        "strings": Layer(
            "strings",
            0.7,
            plan.gate(plan.lights * plan.evening, 0.15, 0.5),
            keys_cut * 0.8,
            0.1,
            reverb=0.6,
            title="evening strings",
        ),
    }
    space = np.clip(0.18 + 0.25 * plan.humid + 0.2 * sweep, 0.0, 0.8)
    score = Score(
        length,
        notes,
        layers,
        [Texture("vinyl", "vinyl", 0.003, None)],
        space,
        echo_seconds=beat * 0.75,
        echo_feedback=0.38,
        tempo=bpm,
        events=list(plan.events),
        compress_db=-17.0,
        wobble=0.9,
        level_db=-10.2,
    )
    score.choices = dict(plan.choices)
    score.notes_about = [
        (
            f"{bpm:.0f} BPM, swing {swing:.2f}, progression {prog_name} ({chord_bars} bar{'s' if chord_bars > 1 else ''} a chord) on {keys_voice}; "
            f"comping '{comp_name}', bass '{bass_name}' on {bass_voice}, drums '{drum_name}'."
        ),
        f"Variations by rule: {variations['swap']} chord swaps, {variations['comp']} comping changes, {variations['chop']} chops, {variations['pluck']} plucked lines.",
    ]
    del quality
    return score
