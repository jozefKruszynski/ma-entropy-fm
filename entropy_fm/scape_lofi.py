"""
Lo-fi: a dusty swung boom-bap loop, electric piano on jazzy progressions, a round bass, vinyl and tape.

The generative system is a band that plays a composed loop and never plays it the same way twice. The day picks
a progression of four chords with sevenths, ninths and elevenths that agrees with the mode of the day (ii V I and
backdoor moves on the major, lydian and mixolydian days, i IV bVII and i VI iv V on the dorian and aeolian ones), a
comping rhythm, a bass pattern and a drum pattern; every chord is voiced a little differently each time and led from
the one before (common tones held, the other voices by step), and the band varies the loop by rule: a chord is
sometimes swapped for a related one, the comping rhythm drifts to a neighbour, ghost notes and fills come and go,
and a chord chop or a plucked line is played now and then. At night the drums rest and the loop becomes an ambient
piece of electric piano and vinyl.

How the house steers it: the energy of the day picks the tempo inside 70 to 88, how long a chord lasts (four to eight
bars on a quiet day, one or two on a busy one), the swing (a quiet day lies back, a busy one plays straighter), the
comping, the bass pattern and the drum pattern (a soft half time kit when quiet, a rolling syncopated kit when busy);
the warmth picks the rounder or the glassier keys, bass, pluck and strings. The shape of the day is the form: intro,
build, peak, release and outro enter and leave on phrases of eight bars, a build ends in a fill, and the motif of the
room is played on the plucked line in the first and the last peak or plateau. The strongest event of the day turns the next phrase into
a breakdown (the kit and the bass leave and come back with a fill) or brings the strings in. On top of that the
activity sets the density of the drums and the comping, the power of appliances the weight of the bass, the
temperature and the sun the tape low-pass, humidity the room, the lights and the evening the extensions of the
chords, a running dishwasher brings in the drum loop, and an anomaly closes the tape low-pass and opens it again (a
filter dip, never a pitch glide).

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import itertools
from bisect import bisect_right
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .character import chord_bars, tempo_target
from .gen import block_walk, dips, swells, swung
from .harmony import MOTIF_CHOICE, develop_motif, motif_seed, motif_sites
from .plan import (
    BUILD,
    CLIMAX,
    FORM_BARS,
    INTRO,
    MIN_FORM_CELLS,
    OUTRO,
    PLATEAU,
    RELEASE,
    SHORT_FORM_BARS,
    TEMPO_TOLERANCE,
)
from .score import Layer, Note, Score, Texture
from .styles import LOFI

if TYPE_CHECKING:
    from collections.abc import Sequence

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
# bass: (start step, length in steps, 'r' root, 'f' fifth, 'o' octave)
BASSES: tuple[Pattern, ...] = (
    Pattern("long and short", ((0, 4, "r"), (8, 2, "r"), (11, 2, "f"), (14, 2, "r")), 0.15),
    Pattern("walking root", ((0, 6, "r"), (10, 2, "f"), (14, 2, "r")), 0.4),
    Pattern("round", ((0, 3, "r"), (3, 1, "r"), (6, 2, "f"), (10, 2, "r"), (13, 2, "o")), 0.6),
    Pattern("step", ((0, 2, "r"), (4, 2, "r"), (8, 2, "f"), (10, 2, "r"), (14, 1, "o")), 0.85),
)
BASS_TONES = {"r": 0, "f": 7, "o": 12}
# drums on a 16-step bar: kick steps, snare steps, ghost steps (a soft snare or a rim)
DRUMS: tuple[Pattern, ...] = (
    Pattern("half time", ((0, 10), (8,), (6, 15)), 0.0),
    Pattern("lazy", ((0, 3, 10), (4, 12), (9, 14)), 0.2),
    Pattern("classic", ((0, 7, 10), (4, 12), (15,)), 0.45),
    Pattern("late kick", ((0, 5, 8, 11), (4, 12), ()), 0.55),
    Pattern("shuffle", ((0, 8, 11), (4, 12), (14,)), 0.7),
    Pattern("rolling", ((0, 6, 10, 13), (4, 12), (9,)), 0.9),
)


class Voice(NamedTuple):
    """A voice and how warm it sounds, -1 (glassy, dark) to 1 (round, warm)."""

    key: str
    warmth: float


KEYS = (Voice("tx_ep", 0.5), Voice("fm_ep", -0.5))
BASS_VOICES = (Voice("sub_round", 0.7), Voice("moog_bass", -0.3))
PLUCKS = (Voice("strum_pluck", 0.4), Voice("nylon_ks", -0.2))
STRING_VOICES = (Voice("juno_pad", 0.5), Voice("string_machine", -0.4))
TEMPOS = tuple(float(bpm) for bpm in range(int(LOFI.tempo[0]), int(LOFI.tempo[1]) + 1, 2))
# the swing of a sixteenth (how late the odd ones are): a quiet day lies back, a busy one plays straighter
SWING_QUIET = 0.30
SWING_BUSY = 0.10
SWING_DETAIL = 0.02
# the electric piano plays its chords between these notes
KEYS_LOW = 53
KEYS_HIGH = 76
KEYS_CENTRE = 62
# the widest a voicing is: an octave and a fourth
KEYS_SPAN = 17
# the bass root lies from E1 up, and no note of the bass reaches C3 (MIDI 48)
BASS_LOWEST = 28
BASS_CEILING = 48
# the plucked line states the motif of the room from C4 up
MOTIF_LOWEST = 60
MOTIF_VELOCITY = 0.6
# the least strength of the drums by day for the quietest and the busiest day, and the same in the cheapest tier
DRUMS_FLOOR = (0.2, 0.85)
LIGHT_DRUMS_FLOOR = (0.15, 0.55)
LIGHT_LAYERS = 4
STRINGS_FLOOR = 0.35
# the kit comes in over this stretch of energy, and a quiet day outside the night keeps this much of it
KIT_FROM = 0.1
KIT_FULL = 0.4
KIT_SOFT = 0.6
# the chord chops are a short accent, so their layer plays louder than the other layers to be heard
CHOP_GAIN = 1.2
# the electric piano sits a few dB behind the drums and the bass, softened by a little more room
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
THIRD = 1.0 / 3.0


def _fit_to(value: float, target: float) -> float:
    """Return how close a tag is to what the day asks, 1 when equal."""
    return 1.0 - abs(value - target)


def _warm_fit(voices: Sequence[Voice], warmth: float) -> list[float]:
    """Return how well each voice suits the warmth of the day."""
    return [1.0 - abs(voice.warmth - warmth) / 2.0 for voice in voices]


def _bars_fit(options: Sequence[int], energy: float) -> list[float]:
    """Return how well each count of bars suits: the busier within its third of energy, the shorter."""
    within = float(np.clip(energy / THIRD - min(int(energy / THIRD), 2), 0.0, 1.0))
    last = max(len(options) - 1, 1)
    return [_fit_to(index / last, 1.0 - within) for index in range(len(options))]


def _voicing(
    scale_root: int, chord: Chord, previous: Sequence[int] | None, *, extended: bool
) -> list[int]:
    """Return the notes of a chord above its root (the root is the bass's), led from the previous voicing."""
    offset, quality = chord
    name = EXTENDED[quality] if extended else quality
    tones = [(scale_root + offset + interval) % 12 for interval in QUALITIES[name]]
    places = [[n for n in range(KEYS_LOW, KEYS_HIGH + 1) if n % 12 == tone] for tone in tones]
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


def _motif_notes(plan: Plan, motif: tuple[tuple[int, float], ...]) -> list[tuple[int, float]]:
    """Return a motif as (MIDI note, beats) in the key and mode of the day, from C4 up."""
    octave = 4
    while min(plan.scale.midi(degree, octave) for degree, _ in motif) < MOTIF_LOWEST:
        octave += 1
    return [(plan.scale.midi(degree, octave), beats) for degree, beats in motif]


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Lo-fi soundscape.

    :param plan: The curves and the deals of the day.
    """
    length = plan.length
    rng = plan.rng("lofi")
    character = plan.character
    energy = character.energy
    warmth = character.warmth
    root = plan.scale.root
    activity, power = plan.activity, plan.power
    light = plan.quality.max_layers <= LIGHT_LAYERS
    low, high = LOFI.tempo
    target = tempo_target(LOFI, character)
    bpm = plan.pick(
        "tempo",
        TEMPOS,
        [_fit_to((t - low) / (high - low), target) for t in TEMPOS],
        tolerance=TEMPO_TOLERANCE,
    )
    beat = 60.0 / bpm
    step = beat / 4.0
    bar = beat * 4.0
    phrase_bars = FORM_BARS if length / (FORM_BARS * bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
    phrase = phrase_bars * bar
    prog = plan.pick(
        "progression",
        PROGRESSIONS,
        [
            MODE_WEIGHT * float(plan.scale.mode in p.modes)
            + (1.0 - MODE_WEIGHT) * _fit_to(p.motion, energy)
            for p in PROGRESSIONS
        ],
    )
    comp_pattern = plan.pick("comping", COMPS, [_fit_to(c.energy, energy) for c in COMPS])
    bass_pattern = plan.pick("bass pattern", BASSES, [_fit_to(b.energy, energy) for b in BASSES])
    drum_pattern = plan.pick("drum pattern", DRUMS, [_fit_to(d.energy, energy) for d in DRUMS])
    keys_voice = plan.pick("keys", KEYS, _warm_fit(KEYS, warmth)).key
    bass_voice = plan.pick("bass voice", BASS_VOICES, _warm_fit(BASS_VOICES, warmth)).key
    pluck_voice = plan.pick("pluck", PLUCKS, _warm_fit(PLUCKS, warmth)).key
    strings_voice = plan.pick("strings", STRING_VOICES, _warm_fit(STRING_VOICES, warmth)).key
    options = tuple(o for o in chord_bars(character) if o <= phrase_bars) or (phrase_bars,)
    chord_span = plan.pick("chord bars", options, _bars_fit(options, energy))
    patch = int(plan.rng("patch").integers(0, 3))
    plan.choices["patch"] = str(patch)
    swing = (
        SWING_QUIET
        + (SWING_BUSY - SWING_QUIET) * energy
        + SWING_DETAIL * float(plan.rng("swing").uniform(-1.0, 1.0))
    )

    # the kit is part of the base of the day: the house moves how hard it plays, the energy of the day how much of it
    # plays, and the night lets it rest
    floor_quiet, floor_busy = LIGHT_DRUMS_FLOOR if light else DRUMS_FLOOR
    kit_scale = float(np.clip((energy - KIT_FROM) / (KIT_FULL - KIT_FROM), 0.0, 1.0))
    kit_scale = kit_scale * kit_scale * (3.0 - 2.0 * kit_scale)
    kit_scale = KIT_SOFT + (1.0 - KIT_SOFT) * kit_scale
    drums_on = (
        plan.lift(
            np.maximum(plan.gate(activity, 0.22, 0.5), plan.runs["dish"]),
            floor_quiet + (floor_busy - floor_quiet) * energy,
        )
        * kit_scale
        * (1.0 - 0.85 * plan.night)
    )

    # the form on phrases, and the strongest event that turns the next phrase
    sections = plan.sections(bar)
    starts = [s[0] for s in sections]

    def role_at(second: float) -> tuple[str, float]:
        found = sections[max(bisect_right(starts, second) - 1, 0)]
        return found[2], found[3]

    breakdown = (0.0, 0.0)
    strings_from = float(length)
    if plan.events:
        when, _ = max(plan.events, key=lambda e: e[1])
        edge = (int(when / phrase) + 1) * phrase
        if edge + 2 * phrase < length:
            if role_at(edge)[0] in {CLIMAX, PLATEAU}:
                breakdown = (edge, edge + phrase)
            else:
                strings_from = edge
    fill_bars = {
        round(end / bar) - 1 for _, end, role, _ in sections if role == BUILD and end < length
    }
    if breakdown[1]:
        fill_bars.add(round(breakdown[1] / bar) - 1)
    sites = [i for i, (_, _, role, _) in enumerate(sections) if role in {CLIMAX, PLATEAU}]
    motif_bars = {
        round(sections[i][0] / bar): _motif_notes(
            plan, develop_motif(plan.motif(), number, motif_seed(plan))
        )
        for number, i in enumerate(motif_sites(sites or [len(sections) - 1]))
    }

    notes: list[Note] = []
    variations = {"swap": 0, "comp": 0, "chop": 0, "pluck": 0, "motif": 0}
    current = list(prog.chords)
    comp_now = comp_pattern
    bars = int(length / bar)
    chord_index = 0
    previous: list[int] | None = None
    kicks, snares, ghosts = drum_pattern.steps
    eighths = HAT_EIGHTH_LEAST + (1.0 - HAT_EIGHTH_LEAST) * energy
    sixteenths = (
        HAT_SIXTEENTH_MOST * max(energy - HAT_SIXTEENTH_FROM, 0.0) / (1.0 - HAT_SIXTEENTH_FROM)
    )
    for bar_index in range(bars):
        second = bar_index * bar
        if bar_index % (4 * chord_span) == 0 and bar_index > 0:
            # the loop is varied by rule: a chord may be swapped for a related one, the comping may drift
            if rng.random() < 0.12 + 0.3 * plan.at(activity, second):
                k = int(rng.integers(0, len(current)))
                off, qual = current[k]
                sub = SUBSTITUTES.get(qual)
                if qual.startswith("dom") and off != DOMINANT:
                    sub = None
                if sub is not None and current[k] == prog.chords[k]:
                    current[k] = (off + sub[0], sub[1])
                    variations["swap"] += 1
                else:
                    current[k] = prog.chords[k]
            if rng.random() < 0.25:
                at_index = COMPS.index(comp_now)
                neighbours = [i for i in (at_index - 1, at_index + 1) if 0 <= i < len(COMPS)]
                comp_now = COMPS[neighbours[int(rng.integers(0, len(neighbours)))]]
                variations["comp"] += 1
        if bar_index in motif_bars and not breakdown[0] <= second < breakdown[1]:
            variations["motif"] += 1
            clock = second
            for midi, beats in motif_bars[bar_index]:
                notes.append(
                    Note(
                        clock,
                        beats * beat * 0.95,
                        float(midi),
                        MOTIF_VELOCITY,
                        pluck_voice,
                        "pluck",
                        patch % 2,
                    )
                )
                clock += beats * beat
        if bar_index % chord_span != 0:
            continue
        role, intensity = role_at(second)
        in_break = breakdown[0] <= second < breakdown[1]
        chord = current[chord_index % len(current)]
        chord_index += 1
        here_drums = plan.at(drums_on, second) * (0.0 if in_break else KIT_BY_ROLE[role])
        extended = plan.at(plan.lights, second) > 0.5 or plan.at(plan.evening, second) > 0.55
        voicing = _voicing(root, chord, previous, extended=extended)
        previous = voicing
        # with the drums at rest the electric piano holds the chord, a quiet piece
        playing = here_drums > KIT_PLAYING
        pattern = comp_now.steps if playing else COMPS[0].steps
        density = float(
            np.clip(
                COMP_ENERGY_SHARE * energy + (1.0 - COMP_ENERGY_SHARE) * plan.at(activity, second),
                0.0,
                1.0,
            )
            * COMP_BY_ROLE[role]
            * (0.5 + 0.5 * intensity)
        )
        keep = COMP_KEEP[0] + (COMP_KEEP[1] - COMP_KEEP[0]) * density
        rounds = 1 if light or not playing else chord_span
        for b in range(rounds):
            for start, long, rel in pattern:
                if rng.random() > keep and long < 6:
                    continue
                at = second + b * bar + swung(start, step, swing) + float(rng.normal(0.0, 0.012))
                hold = (
                    min(long * step * 1.1, bar - start * step)
                    if playing
                    else chord_span * bar * 0.95
                )
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
        root_midi = BASS_LOWEST + (root + chord[0] - BASS_LOWEST) % 12
        for b in range(chord_span):
            base = second + b * bar
            role_b, intensity_b = role_at(base)
            silent = in_break or role_b in SILENT_BASS
            for start, long, tone in () if silent else bass_pattern.steps:
                offset = BASS_TONES[tone]
                if root_midi + offset >= BASS_CEILING:
                    offset = 0
                if rng.random() < 0.08:
                    continue
                notes.append(
                    Note(
                        base + swung(start, step, swing) + float(rng.normal(0.0, 0.006)),
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
            this_bar = bar_index + b
            kit_role = 0.0 if breakdown[0] <= base < breakdown[1] else KIT_BY_ROLE[role_b]
            level = plan.at(drums_on, base) * kit_role
            if this_bar in fill_bars:
                ahead = KIT_BY_ROLE[role_at(base + bar)[0]]
                if plan.at(drums_on, base) * ahead >= KIT_LEAST:
                    for fill_step, fill_vel in FILL:
                        notes.append(
                            Note(
                                base + fill_step * step + 0.012,
                                0.3,
                                0.0,
                                fill_vel * (0.7 + 0.3 * intensity_b),
                                "drum:snare_lofi",
                                "drums",
                                patch % 3,
                            )
                        )
            if level < KIT_LEAST:
                continue
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
                if rng.random() < 0.15 + 0.7 * energy * (0.5 + 0.5 * plan.at(activity, base)):
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
            for k in range(16):
                if k % 4 != 0:
                    if k % 2 == 0 and rng.random() > eighths:
                        continue
                    if k % 2 == 1 and (light or level < 0.5 or rng.random() > sixteenths):
                        continue
                elif kit_scale < 1.0 and rng.random() > kit_scale:
                    continue
                accent = 1.0 if k % 4 == 0 else 0.75 if k % 2 == 0 else 0.45
                notes.append(
                    Note(
                        base + swung(k, step, swing) + float(rng.normal(0.0, 0.006)),
                        0.1,
                        0.0,
                        accent * (0.35 + 0.25 * level) + float(rng.normal(0, 0.03)),
                        "drum:hat_closed",
                        "drums",
                        patch % 3,
                    )
                )
        # ear candy: a chord chop now and then, and once in a while a plucked line
        if here_drums > 0.3 and rng.random() < 0.02 + 0.14 * energy * plan.at(activity, second):
            variations["chop"] += 1
            at = second + float(rng.choice([6, 10, 14])) * step
            for k, midi in enumerate(voicing):
                notes.append(
                    Note(at + k * 0.006, 0.12, float(midi + 12), 0.5, keys_voice, "chop", patch)
                )
        if (
            bar_index % 12 == 6
            and bar_index not in motif_bars
            and plan.at(activity, second) > 0.12
            and rng.random() < 0.15 + 0.55 * energy
        ):
            variations["pluck"] += 1
            lead = voicing[0] + 12
            shape = (0, 2, 3, 7, 10) if chord[1].startswith("m") else (0, 2, 4, 7, 9)
            for k in range(int(rng.integers(3, 6))):
                notes.append(  # noqa: PERF401
                    Note(
                        second + k * beat * 0.5,
                        0.5,
                        float(lead + shape[int(rng.integers(0, len(shape)))]),
                        0.5,
                        pluck_voice,
                        "pluck",
                        patch % 2,
                    )
                )
        # the strings: a soft triad on the chord (a tonic pedal where the chord changes every bar), swelling in
        # when the lights are on in the evening, and in from the phrase after the strongest event
        warm = max(plan.at(plan.lights, second) * plan.at(plan.evening, second), STRINGS_FLOOR)
        hold_bars = chord_span if chord_span >= 2 else FORM_BARS
        enters = role != INTRO or second >= strings_from
        if bar_index % hold_bars == 0 and warm > 0.3 and enters and not in_break:
            held = (
                voicing[:3]
                if chord_span >= 2
                else [KEYS_LOW + (root - KEYS_LOW) % 12 + o for o in (0, 7, 12)]
            )
            for k, midi in enumerate(held):
                notes.append(
                    Note(
                        second + 0.1 * k,
                        bar * hold_bars,
                        float(midi),
                        (0.3 + 0.2 * warm) * (0.6 + 0.4 * intensity),
                        strings_voice,
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
        "keys": Layer(
            "keys", KEYS_GAIN, None, keys_cut, 0.12, reverb=KEYS_REVERB, title="electric piano"
        ),
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
            "chop", CHOP_GAIN, None, keys_cut * 0.6, 0.2, reverb=0.35, echo=0.2, title="chord chops"
        ),
        "pluck": Layer(
            "pluck", PLUCK_GAIN, None, None, 0.0, reverb=0.4, echo=0.15, title="plucked line"
        ),
        "strings": Layer(
            "strings",
            0.7,
            plan.lift(plan.gate(plan.lights * plan.evening, 0.15, 0.5), STRINGS_FLOOR),
            keys_cut * 0.8,
            0.1,
            reverb=0.6,
            title="strings",
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
    plan.choices[MOTIF_CHOICE] = str(variations["motif"])
    score.choices = dict(plan.choices)
    form = ", ".join(f"{role} {round((b - a) / bar)}" for a, b, role, _ in sections)
    broke = f", a breakdown at {breakdown[0]:.0f} s" if breakdown[1] else ""
    score.notes_about = [
        (
            f"{bpm:.0f} BPM, swing {swing:.2f}, progression {prog.name} ({chord_span} bar{'s' if chord_span > 1 else ''} a chord) on {keys_voice}; "
            f"comping '{comp_pattern.name}', bass '{bass_pattern.name}' on {bass_voice}, drums '{drum_pattern.name}'."
        ),
        f"Form in bars: {form}; the motif of the room {variations['motif']} times{broke}.",
        f"Variations by rule: {variations['swap']} chord swaps, {variations['comp']} comping changes, {variations['chop']} chops, {variations['pluck']} plucked lines.",
    ]
    return score
