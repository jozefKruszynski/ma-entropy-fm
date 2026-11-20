"""
The harmony the styles with chords share: voice leading, held tones, the voices a day picks, the length of a chord.

A voice is tagged with the energy and the warmth it suits, and the character of the day picks the voice that fits it
best; the progressions are chord roots as scale degrees, from which the ones that rest on a diminished chord in the
mode of the day are left out; `voice_lead` keeps the voices of the strings and the pads moving as little as possible
and `held_runs` turns the result into long notes. Berlin School uses it with its own vocabulary.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from .character import chord_bars
from .score import CR
from .synth import SCALES
from .timeline import stable_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from .character import DayCharacter
    from .plan import Plan
    from .synth import Scale

    Motif = tuple[tuple[int, float], ...]

# chord roots of a progression as scale degrees, and how much the harmony moves (0 still, 1 restless)
Progression = tuple[str, tuple[int, ...], float]
# what each voice suits as (energy 0 to 1, warmth -1 to 1): warm days get the rounder voices, cold or late days the
# glassier and darker ones, busy days the sharper and heavier ones
VoiceTag = tuple[float, float]
# the growling bass is a busy-day voice; "none" is the day that has no growl at all
NO_GROWL = "none"
GROWLS: dict[str, VoiceTag] = {NO_GROWL: (0.25, 0.0), "reese": (0.8, -0.3), "fm_growl": (0.95, 0.3)}
VOICE_ENERGY_WEIGHT = 0.7
VOICE_WARMTH_WEIGHT = 0.3
WARMTH_SPAN = 2.0
LATE_NIGHT_COOLING = 0.3
# the chord on a degree of a mode is stacked from thirds of the mode; a diminished one is no chord to rest on for bars
DIMINISHED_FIFTH = 6
TRIAD_THIRDS = (2, 4)
# a chord changes no faster than every two bars: the lines are cells of one bar and the sequence needs time to be heard
MIN_CHORD_BARS = 2
BARS_PER_PHRASE = 8
WINDOW_DEGREES = 14
OCTAVE_DEGREES = 7
# the lead: the motif is played in this octave, or the next when it would fall under middle C
LEAD_OCTAVE = 4
LEAD_FLOOR_MIDI = 60
# a piece states the motif of the room at most this many times: the first peak or plateau and the last
MOTIF_STATEMENTS = 2
# the choice that the log and the description say how many statements the piece has
MOTIF_CHOICE = "motif statements"
# a later statement is the motif developed: moved to another chord tone, in longer or shorter notes, with its tail
# turned over, or followed by itself a step up
MOTIF_TRANSPOSE_STEPS = (2, 4)
MOTIF_STRETCH = 2.0
MIN_MOTIF_NOTE_BEATS = 0.5
MAX_MOTIF_BEATS = 16.0
# the tail of the motif is turned over from this note on, so the first two intervals keep their direction
MOTIF_INVERT_FROM = 3
MOTIF_SEQUENCE_STEP = 1


def voice_fit(tag: VoiceTag, character: DayCharacter) -> float:
    """Return how well a voice suits the day: 1 when its energy and warmth are the day's."""
    warmth = character.warmth - (LATE_NIGHT_COOLING if character.late_night else 0.0)
    return (
        1.0
        - VOICE_ENERGY_WEIGHT * abs(tag[0] - character.energy)
        - VOICE_WARMTH_WEIGHT * abs(tag[1] - warmth) / WARMTH_SPAN
    )


def pick_voice(plan: Plan, component: str, family: dict[str, VoiceTag]) -> str:
    """Pick the voice of a family that suits the character of the day."""
    keys = tuple(family)
    return plan.pick(component, keys, [voice_fit(family[key], plan.character) for key in keys])


def triad_is_diminished(mode: str, degree: int) -> bool:
    """Return True when the chord on a degree of a mode has a diminished fifth."""
    steps = SCALES[mode]
    size = len(steps)

    def semitones(offset: int) -> int:
        shift, index = divmod(degree + offset, size)
        return 12 * shift + steps[index] - steps[degree % size]

    return semitones(TRIAD_THIRDS[1]) % 12 == DIMINISHED_FIFTH


def progressions_for(mode: str, family: tuple[Progression, ...]) -> list[Progression]:
    """
    Return the progressions of a family that never rest on a diminished chord in a mode.

    :param mode: The mode of the day.
    :param family: The progressions to choose from.
    """
    return [item for item in family if not any(triad_is_diminished(mode, root) for root in item[1])]


def harmonic_rhythm(plan: Plan) -> int:
    """Pick how many bars a chord lasts: long on a quiet day, short on a busy one, never under two bars."""
    options = [bars for bars in chord_bars(plan.character) if bars >= MIN_CHORD_BARS] or [
        MIN_CHORD_BARS
    ]
    span = math.log2(BARS_PER_PHRASE / MIN_CHORD_BARS)
    return plan.pick(
        "harmonic rhythm",
        options,
        [
            1.0 - abs(math.log2(bars / MIN_CHORD_BARS) / span - (1.0 - plan.character.energy))
            for bars in options
        ],
    )


def section_curve(
    plan: Plan,
    spans: list[tuple[float, float, str, float]],
    value_of: Callable[[str, float], float],
    ramp: float,
) -> np.ndarray:
    """Return a curve that holds a value for each section of the form and changes over `ramp` seconds from its start."""
    curve = np.zeros(plan.count)
    for start, end, role, intensity in spans:
        curve[int(start * CR) : int(end * CR)] = value_of(role, intensity)
    width = max(int(ramp * CR), 1)
    padded = np.concatenate((np.full(width - 1, curve[0]), curve))
    return np.asarray(np.convolve(padded, np.ones(width) / width, mode="valid"))


def voice_lead(
    roots: list[int], offsets: tuple[int, ...], lifts: tuple[int, ...], low: int
) -> list[list[int]]:
    """
    Return the scale degree of every voice for each chord, so that the voices move as little as possible.

    A tone that two chords share stays where it is, the others move by the nearest step; the voices stay within two
    octaves from `low`, and no two of them play the same degree.

    :param roots: The root of each chord, as a scale degree.
    :param offsets: The degree of the chord (0 is the root, 2 the third) that each voice plays.
    :param lifts: The degrees above the lowest position that each voice starts at in the first chord.
    :param low: The lowest degree of the window.
    """
    voices = len(offsets)
    first = [
        low + lift + (roots[0] + offset - low) % OCTAVE_DEGREES
        for offset, lift in zip(offsets, lifts, strict=True)
    ]
    found = [first]
    for root in roots[1:]:
        classes = [(root + offset) % OCTAVE_DEGREES for offset in offsets]
        previous = found[-1]
        best: tuple[int, list[int]] | None = None
        for order in itertools.permutations(range(voices)):
            options = [
                [
                    degree
                    for degree in range(low, low + WINDOW_DEGREES)
                    if degree % OCTAVE_DEGREES == classes[order[voice]]
                ]
                for voice in range(voices)
            ]
            for choice in itertools.product(*options):
                if len(set(choice)) < voices:
                    continue
                cost = sum((a - b) ** 2 for a, b in zip(choice, previous, strict=True))
                if best is None or cost < best[0]:
                    best = (cost, list(choice))
        found.append(best[1] if best else previous)
    return found


def held_runs(
    voicing: list[list[int]],
    starts: list[float],
    ends: list[float],
    restart_at: Callable[[float], bool],
    active: list[bool] | None = None,
) -> list[tuple[int, float, float, int]]:
    """
    Return the notes of a voiced chord progression as (voice, start, end, degree), a tone that stays held through.

    :param voicing: The degree of every voice in each chord (`voice_lead`).
    :param starts: Where each chord starts, in seconds.
    :param ends: Where each chord ends.
    :param restart_at: Says for a start whether a held tone is struck again there, such as at a phrase.
    :param active: Whether the chord is played at all; a rest ends every tone. Default is all of them.
    """
    runs: list[tuple[int, float, float, int]] = []
    for voice in range(len(voicing[0])):
        run: tuple[int, float, float, int] | None = None
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if active is not None and not active[index]:
                if run:
                    runs.append(run)
                run = None
                continue
            degree = voicing[index][voice]
            if run and run[3] == degree and not restart_at(start):
                run = (voice, run[1], end, degree)
                continue
            if run:
                runs.append(run)
            run = (voice, start, end, degree)
        if run:
            runs.append(run)
    return runs


def lead_octave(scale: Scale, motif: tuple[tuple[int, float], ...]) -> int:
    """Return the octave that keeps the whole motif at or above middle C."""
    lowest = min(degree for degree, _ in motif)
    return LEAD_OCTAVE if scale.midi(lowest, LEAD_OCTAVE) >= LEAD_FLOOR_MIDI else LEAD_OCTAVE + 1


def motif_seed(plan: Plan) -> int:
    """Return a number that is the same for this room on this day and differs from day to day."""
    return stable_hash("motif", plan.style, plan.area_key, str(plan.day), plan.seed) % 2**63


def motif_sites(sites: list[int]) -> list[int]:
    """
    Return the places that state the motif: the first one and the last when it is another, at most `MOTIF_STATEMENTS`.

    :param sites: The peak and plateau sections, or any places that could carry the motif, in order.
    """
    if len(sites) <= 1:
        return list(sites)
    return [sites[0], sites[-1]]


def develop_motif(
    motif: Motif, statement: int, seed: int, max_beats: float = MAX_MOTIF_BEATS
) -> Motif:
    """
    Return the motif for a statement: the first as it is, each later one a variation that opens like it.

    :param motif: The motif of the room as (scale degree, beats).
    :param statement: Which statement of the piece, 0 for the first.
    :param seed: Picks the variation; the same seed gives the same one (`motif_seed`).
    :param max_beats: The longest a statement may be, in beats; a longer variation becomes a shorter one.
    """
    if statement <= 0:
        return motif
    rng = np.random.default_rng(stable_hash("develop motif", seed, statement) % 2**63)
    kinds = ["transpose", "stretch", "shrink", "sequence"]
    if len(motif) > MOTIF_INVERT_FROM:
        kinds.append("invert")
    kind = kinds[int(rng.integers(0, len(kinds)))]
    total = sum(beats for _, beats in motif)
    if kind == "transpose":
        shift = MOTIF_TRANSPOSE_STEPS[int(rng.integers(0, len(MOTIF_TRANSPOSE_STEPS)))]
        return tuple((degree + shift, beats) for degree, beats in motif)
    if kind == "invert":
        pivot = motif[MOTIF_INVERT_FROM - 1][0]
        return tuple(
            (2 * pivot - degree if index >= MOTIF_INVERT_FROM else degree, beats)
            for index, (degree, beats) in enumerate(motif)
        )
    if kind == "sequence" and total * 2 <= max_beats:
        return motif + tuple((degree + MOTIF_SEQUENCE_STEP, beats) for degree, beats in motif)
    if kind == "stretch" and total * MOTIF_STRETCH <= max_beats:
        return tuple((degree, beats * MOTIF_STRETCH) for degree, beats in motif)
    return tuple(
        (degree, max(beats / MOTIF_STRETCH, MIN_MOTIF_NOTE_BEATS)) for degree, beats in motif
    )
