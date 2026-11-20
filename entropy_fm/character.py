"""
The character of a day: a few numbers that say what kind of day it was, so that the data picks the identity of a piece.

`measure_character` reads the series of a room (or the house) on a day and gives a `DayCharacter`: how busy it was
against the room's own days, where in the day the activity sat, how warm the day was, how many stretches of activity
there were and how regular, what happened out of the ordinary, and whether it was a day off. The helpers turn it into
musical choices (the key and mode, the position in the tempo range, the harmonic rhythm) and give every room a short
motif of its own.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from .day_analysis import STEP
from .energy import activity
from .palette import (
    HOME_MODE_WEIGHT,
    MODE_WARMTH,
    DayTraits,
    circle_position,
    day_warmth,
    pick_mode,
    season_warmth,
)
from .synth import Scale
from .timeline import BINARY_KINDS, VoiceKind, stable_hash

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from .day_analysis import DayAnalysis
    from .styles import Style
    from .timeline import Series

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
MORNING = "morning"
EVENING = "evening"
ALL_DAY = "all_day"
PEAK = "peak"
FLAT = "flat"
HOURS_PER_DAY = 24.0
SECONDS_PER_HOUR = 3600.0
# the share of the day at which a character falls into the low or the high third
LOW_BELOW = 1.0 / 3.0
HIGH_FROM = 2.0 / 3.0
# the raw activity (0 to 1, as `energy.activity` reads it) that is none, and that is as busy as a room gets: a day
# against its own previous days spreads over this range, so that real days use the whole scale
ENERGY_FLOOR = 0.03
ENERGY_CEILING = 0.60
# a day that is not measured against its own days has its own, wider scale: the series are scaled by the day itself
ENERGY_FLOOR_ABSOLUTE = 0.05
ENERGY_CEILING_ABSOLUTE = 0.70
# the mean and the busiest two hours of the day both count, the mean more: one loud evening is not a busy day
ENERGY_MEAN_WEIGHT = 0.7
ENERGY_TOP_WEIGHT = 0.3
ENERGY_TOP_STEPS = 8
# with this many previous days of the room the energy is mostly where the day sits between their quietest and busiest
MIN_WEEK_DAYS = 3
ENERGY_WEEK_WEIGHT = 0.7
ENERGY_ABSOLUTE_WEIGHT = 0.3
# a week whose days differ by less than this in raw energy has no quiet or busy day: each sits in the middle
MIN_WEEK_SPREAD = 0.02
# the shape of the day
MORNING_START = 5.0
MORNING_END = 12.0
EVENING_START = 17.0
WAKING_START = 7.0
WAKING_END = 22.0
SKEW_SHARE = 0.55
PEAK_WINDOW_HOURS = 3.0
PEAK_WINDOW_SHARE = 0.4
# a curve whose range is under this is flat; one that is above this level over most of the waking day is all day
FLAT_RANGE = 0.08
ALL_DAY_LEVEL = 0.5
ALL_DAY_SHARE = 0.6
PEAK_SMOOTH_STEPS = 3
# stretches of activity: above this share of the day's top (and at least this level), at least half an hour long,
# gaps of an hour or less joined
STRETCH_SHARE = 0.35
STRETCH_FLOOR = 0.05
MIN_STRETCH_STEPS = 2
JOIN_GAP_STEPS = 4
NO_REGULARITY = 0.5
# what counts for the events of a day
SHOWER_WEIGHT = 0.5
ANOMALY_WEIGHT = 0.5
APPLIANCE_WEIGHT = 0.2
COOKING_WEIGHT = 0.15
LATE_NIGHT_WEIGHT = 0.3
APPLIANCE_MOMENTS = frozenset({"appliance", "cooking", "phase"})
# real activity after this hour makes it a late night
LATE_NIGHT_START = 23.5
LATE_NIGHT_LEVEL = 0.5
WORKDAY_PREFIX = "binary_sensor.workday"
NOON = 12.0
# how bright the mode of a day is aimed to be: the weights of what pulls it
BRIGHT_WARMTH = 0.7
BRIGHT_ENERGY = 0.6
BRIGHT_DAY_OFF = 0.1
DARK_LATE_NIGHT = 0.4
# the day moves the key a fifth towards the cold (flat) or the warm (sharp) side beyond this warmth
FIFTH_WARMTH = 0.3
# the tempo: the share of its range that energy and the shape of the day set
TEMPO_ENERGY_WEIGHT = 0.8
TEMPO_SHAPE_BIAS = {ALL_DAY: 0.1, PEAK: 0.05, MORNING: 0.0, EVENING: 0.0, FLAT: -0.1}
TEMPO_DAY_OFF = -0.08
TEMPO_BASE = 0.1
# bars a chord lasts, by how busy the day was
CHORD_BARS = {LOW: (4, 8), MEDIUM: (2, 4), HIGH: (1, 2)}
WARM_LABEL = 0.15


@dataclass(frozen=True)
class DayCharacter:
    """What a day was like, in the few numbers that choose the identity of a piece."""

    # 0 to 1: how busy the day was against the room's own days
    energy: float
    # low, medium or high: the thirds of energy
    level: str
    # morning, evening, all_day, peak or flat
    shape: str
    # 0 to 1: where in the day (share of 24 hours) the main peak is
    peak: float
    # -1 to 1: temperature, sun and weather against the season
    warmth: float
    # the number of distinct active stretches
    stretches: int
    # 0 to 1: how evenly spaced and how alike those stretches are
    regularity: float
    # 0 to 1: showers, anomalies, appliance runs and a late night combined
    events: float
    # real activity after 23:30
    late_night: bool
    # a weekend, or a workday sensor that says off
    day_off: bool

    def label(self) -> str:
        """Return one short line for the log, such as "high, evening, warm 0.4, 3 stretches, events 0.4, workday"."""
        if abs(self.warmth) < WARM_LABEL:
            weather = f"mild {self.warmth:+.1f}"
        else:
            weather = f"{'warm' if self.warmth > 0 else 'cold'} {abs(self.warmth):.1f}"
        parts = [
            self.level,
            self.shape.replace("_", " "),
            weather,
            f"{self.stretches} stretches",
            f"events {self.events:.1f}",
            "day off" if self.day_off else "workday",
        ]
        if self.late_night:
            parts.append("late night")
        return ", ".join(parts)


NEUTRAL = DayCharacter(
    energy=0.5,
    level=MEDIUM,
    shape=FLAT,
    peak=0.5,
    warmth=0.0,
    stretches=0,
    regularity=NO_REGULARITY,
    events=0.0,
    late_night=False,
    day_off=False,
)


def level_of(energy: float) -> str:
    """
    Return the third of the scale that an energy falls in.

    :param energy: 0 to 1.
    """
    if energy < LOW_BELOW:
        return LOW
    return HIGH if energy >= HIGH_FROM else MEDIUM


def _without_workday(series: Sequence[Series]) -> list[Series]:
    """Return the series that are the house itself: a workday sensor says what kind of day it is, not what happened."""
    return [item for item in series if not item.entity_id.startswith(WORKDAY_PREFIX)]


def _workday_from(series: Sequence[Series]) -> bool | None:
    """Return what a workday sensor in the series says about the day, or None when there is none."""
    for item in series:
        if item.entity_id.startswith(WORKDAY_PREFIX):
            return bool(item.at(np.array([NOON * SECONDS_PER_HOUR]))[0] > 0.5)
    return None


def _lively(series: Sequence[Series], analysis: DayAnalysis) -> np.ndarray:
    """Return how lively the room was in each quarter of an hour, from the series; the curve of the analysis if empty."""
    steps = max(int(min(analysis.covered, 86400.0) / STEP), 1)
    if not series:
        return np.asarray(analysis.curve, dtype=np.float64)
    return np.asarray(activity(series, (np.arange(steps) + 0.5) * STEP), dtype=np.float64)


def _raw_of(lively: np.ndarray) -> float:
    """Return the blend of the mean and the busiest two hours of the activity of a day."""
    busiest = np.sort(lively)[-ENERGY_TOP_STEPS:]
    return ENERGY_MEAN_WEIGHT * float(lively.mean()) + ENERGY_TOP_WEIGHT * float(busiest.mean())


def raw_energy(series: Sequence[Series], covered: float) -> float:
    """
    Return how busy a day was on the scale of the activity, before it is compared with any other day.

    :param series: The series of the room on the day, rescaled as the provider does.
    :param covered: How many seconds of the day have happened.
    """
    steps = max(int(min(covered, 86400.0) / STEP), 1)
    lively = np.asarray(activity(_without_workday(series), (np.arange(steps) + 0.5) * STEP))
    return _raw_of(lively)


def _place_among(raw: float, week: Sequence[float]) -> float:
    """Return where a day sits between the quietest (0) and the busiest (1) of it and the days before it."""
    low, high = min(raw, *week), max(raw, *week)
    if high - low < MIN_WEEK_SPREAD:
        return 0.5
    return (raw - low) / (high - low)


def _energy_of(lively: np.ndarray, week: Sequence[float], *, measured: bool) -> float:
    """Return 0 to 1 how busy the day was: against the days before it when there are enough, else on its own."""
    raw = _raw_of(lively)
    floor, ceiling = (
        (ENERGY_FLOOR, ENERGY_CEILING)
        if measured
        else (ENERGY_FLOOR_ABSOLUTE, ENERGY_CEILING_ABSOLUTE)
    )
    absolute = float(np.clip((raw - floor) / (ceiling - floor), 0.0, 1.0))
    if len(week) < MIN_WEEK_DAYS:
        return absolute
    return ENERGY_WEEK_WEIGHT * _place_among(raw, week) + ENERGY_ABSOLUTE_WEIGHT * absolute


def _shape_of(lively: np.ndarray) -> str:
    """Name the shape of the day from where the activity mass sits."""
    hours = (np.arange(len(lively)) + 0.5) * STEP / SECONDS_PER_HOUR
    total = float(lively.sum())
    if total <= 0.0 or float(np.ptp(lively)) < FLAT_RANGE:
        return FLAT
    waking = (hours >= WAKING_START) & (hours < WAKING_END)
    if (
        waking.any()
        and float(np.mean(lively[waking] >= ALL_DAY_LEVEL * lively.max())) >= ALL_DAY_SHARE
    ):
        return ALL_DAY
    if float(lively[(hours >= MORNING_START) & (hours < MORNING_END)].sum()) / total >= SKEW_SHARE:
        return MORNING
    if float(lively[hours >= EVENING_START].sum()) / total >= SKEW_SHARE:
        return EVENING
    width = max(int(PEAK_WINDOW_HOURS * SECONDS_PER_HOUR / STEP), 1)
    window = np.convolve(lively, np.ones(width), mode="valid")
    return PEAK if float(window.max()) / total >= PEAK_WINDOW_SHARE else ALL_DAY


def _peak_of(lively: np.ndarray) -> float:
    """Return where in the day (share of 24 hours) the main peak of the activity is."""
    smooth = np.convolve(lively, np.ones(PEAK_SMOOTH_STEPS) / PEAK_SMOOTH_STEPS, mode="same")
    return float(
        np.clip((int(np.argmax(smooth)) + 0.5) * STEP / SECONDS_PER_HOUR / HOURS_PER_DAY, 0.0, 1.0)
    )


def _stretches_of(lively: np.ndarray) -> list[tuple[int, int]]:
    """Return the stretches of the day with activity as (first step, last step), the near ones joined."""
    top = float(lively.max()) if len(lively) else 0.0
    active = lively >= max(STRETCH_FLOOR, STRETCH_SHARE * top)
    found: list[list[int]] = []
    for step, on in enumerate(active):
        if not on:
            continue
        if found and step - found[-1][1] <= JOIN_GAP_STEPS:
            found[-1][1] = step
        else:
            found.append([step, step])
    return [(a, b) for a, b in found if b - a + 1 >= MIN_STRETCH_STEPS]


def _variation(values: np.ndarray) -> float:
    """Return the spread of some values as a share of their mean."""
    return float(np.std(values)) / max(float(np.mean(values)), 1e-9)


def _regularity_of(stretches: Sequence[tuple[int, int]]) -> float:
    """Return 0 to 1 how evenly spaced and how alike the stretches are; undecided for fewer than three."""
    if len(stretches) < 3:
        return NO_REGULARITY
    starts = np.array([a for a, _ in stretches], dtype=np.float64)
    lengths = np.array([b - a + 1 for a, b in stretches], dtype=np.float64)
    gaps = np.diff(starts)
    spread = _variation(gaps) + _variation(lengths)
    return float(1.0 / (1.0 + spread))


def _late_night(series: Sequence[Series], covered: float) -> bool:
    """Return True when something was switched on or an appliance started after the late night hour."""
    start = LATE_NIGHT_START * SECONDS_PER_HOUR
    if covered < start:
        return False
    for item in series:
        if item.is_baseline_load:
            continue
        if item.kind in BINARY_KINDS and len(item.onsets(start, covered)):
            return True
        if item.kind in {VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER} and len(
            item.crossings(start, covered, LATE_NIGHT_LEVEL, rising=True)
        ):
            return True
    return False


def _events_of(analysis: DayAnalysis, series: Sequence[Series], *, late: bool) -> float:
    """Return 0 to 1 how much out of the ordinary happened: showers, anomalies, appliance runs, a late night."""
    steady = {item.entity_id for item in series if item.is_baseline_load}
    score = LATE_NIGHT_WEIGHT if late else 0.0
    for moment in analysis.moments:
        if moment.source in steady:
            continue
        if moment.kind == "shower":
            score += SHOWER_WEIGHT * moment.strength
        elif moment.kind == "anomaly":
            score += ANOMALY_WEIGHT * moment.strength
        elif moment.kind in APPLIANCE_MOMENTS:
            score += COOKING_WEIGHT if moment.kind == "cooking" else APPLIANCE_WEIGHT
    return float(1.0 - math.exp(-score))


def measure_character(  # noqa: PLR0913
    analysis: DayAnalysis,
    traits: DayTraits | None,
    day: date | None,
    series: Sequence[Series],
    *,
    workday: bool | None = None,
    week: Sequence[float] = (),
) -> DayCharacter:
    """
    Measure the character of a day of a room or of the house.

    :param analysis: The analysis of the day; its moments are the events and it says whether the series are measured
        against the room's own days.
    :param traits: What the day was like (temperature against the usual, weather), when known.
    :param day: The day; without it the season is unknown and the weekday counts as a workday.
    :param series: The series of the room on the day, rescaled as the provider does; a workday sensor among them says
        whether it is a day off.
    :param workday: Whether the day is a workday, when the caller knows; it wins over the series and the weekday.
    :param week: The `raw_energy` of the room on its previous days (up to seven), on the same scale as the series;
        with at least `MIN_WEEK_DAYS` of them the energy is mostly where the day sits between their quietest and busiest.
    """
    house = _without_workday(series)
    lively = _lively(house, analysis)
    stretches = _stretches_of(lively)
    late = _late_night(house, analysis.covered)
    events = _events_of(analysis, house, late=late)
    energy = _energy_of(lively, week, measured=analysis.measured)
    if workday is None:
        workday = _workday_from(series)
    day_off = (day is not None and day.weekday() >= 5) if workday is None else not workday
    # warmth is the day against its season: the season itself already moves the key through the weekly walk
    season = season_warmth(day) if day is not None else 0.0
    return DayCharacter(
        energy=energy,
        level=level_of(energy),
        shape=_shape_of(lively),
        peak=_peak_of(lively),
        warmth=day_warmth(traits or DayTraits(), season, season_weight=0.0),
        stretches=len(stretches),
        regularity=_regularity_of(stretches),
        events=events,
        late_night=late,
        day_off=day_off,
    )


def scale_of_day(style: Style, home: Scale, day: date, character: DayCharacter) -> Scale:
    """
    Return the key and mode of a day: the weekly walk on the circle of fifths, a fifth either way for the warmth.

    The key walks a fifth a week (`palette.circle_position`); a cold day goes a fifth to the flat side and a warm one
    a fifth to the sharp side, so that days next to each other stay within two fifths. The mode is the one of the
    style that is nearest in brightness to what the day asks: bright for a warm, busy day, dark for a cold, quiet or
    late one. The mode the user chose keeps a small say.

    :param style: The style, which limits the modes.
    :param home: The key and mode the user chose.
    :param day: The day.
    :param character: The character of the day.
    """
    step = 0
    if character.warmth <= -FIFTH_WARMTH:
        step = -1
    elif character.warmth >= FIFTH_WARMTH:
        step = 1
    walk = circle_position(day) - circle_position(day - timedelta(days=1))
    # on the day the walk moves, the day step may not go the same way again: a fifth of each makes two
    if (walk > 0 and step > 0) or (walk < 0 and step < 0):
        step = 0
    root = (home.root + 7 * (circle_position(day) + step)) % 12
    brightness = (
        BRIGHT_WARMTH * character.warmth
        + BRIGHT_ENERGY * (2.0 * character.energy - 1.0)
        + (BRIGHT_DAY_OFF if character.day_off else 0.0)
        - (DARK_LATE_NIGHT if character.late_night else 0.0)
    )
    target = float(
        np.clip(HOME_MODE_WEIGHT * MODE_WARMTH.get(home.mode, -0.5) + brightness, -1.0, 1.0)
    )
    return Scale(root, pick_mode(style.modes, target))


def tempo_target(style: Style, character: DayCharacter) -> float:
    """
    Return where in the tempo range of the style a day sits, 0 (slowest) to 1 (fastest).

    :param style: The style; the range itself is `style.tempo`, which the caller maps this onto.
    :param character: The character of the day.
    """
    del style
    target = (
        TEMPO_BASE
        + TEMPO_ENERGY_WEIGHT * character.energy
        + TEMPO_SHAPE_BIAS.get(character.shape, 0.0)
        + (TEMPO_DAY_OFF if character.day_off else 0.0)
    )
    return float(np.clip(target, 0.0, 1.0))


def chord_bars(character: DayCharacter) -> tuple[int, ...]:
    """
    Return how many bars a chord may last on a day: long on a quiet day, short on a busy one.

    :param character: The character of the day.
    """
    return CHORD_BARS[character.level]


# the motif of a room: this many notes, of these lengths in beats, in a total of 4 or 8 beats
MOTIF_NOTES = (3, 4, 5)
MOTIF_BEATS = (4, 8)
NOTE_BEATS = (0.5, 1.0, 1.5, 2.0, 3.0)
MOTIF_ENDS = (0, 2, 4)
# degrees of a seven note scale: an octave is seven, a fifth is four
OCTAVE_DEGREES = 7
LARGEST_LEAP = 4
LEAP_SIZES = (2, 3, 4)
LEAP_CHANCE = 0.7
DOWN_CHANCE = 0.5
LOWEST_DEGREE = -2
HIGHEST_DEGREE = 9
MOTIF_ATTEMPTS = 40
FALLBACK_MOTIF = ((0, 1.0), (2, 1.0), (4, 2.0))


def _note_lengths(rng: np.random.Generator, count: int, beats: int) -> tuple[float, ...]:
    """Return the lengths of `count` notes that make `beats`, the last one not short, none over half the motif."""
    fits = [
        lengths
        for lengths in itertools.product(NOTE_BEATS, repeat=count)
        if sum(lengths) == beats and lengths[-1] >= 1.0 and max(lengths) <= beats / 2
    ]
    return fits[int(rng.integers(0, len(fits)))]


def _degrees(rng: np.random.Generator, count: int) -> tuple[int, ...] | None:
    """Return the degrees of a motif that ends on the root, the third or the fifth and mostly moves by step, or None."""
    end = MOTIF_ENDS[int(rng.integers(0, len(MOTIF_ENDS)))]
    leap_at = int(rng.integers(0, count - 1)) if rng.random() < LEAP_CHANCE else -1
    path = [end]
    for move in range(count - 2, -1, -1):
        size = LEAP_SIZES[int(rng.integers(0, len(LEAP_SIZES)))] if move == leap_at else 1
        path.insert(0, path[0] + (1 if rng.random() < DOWN_CHANCE else -1) * size)
    inside = all(LOWEST_DEGREE <= degree <= HIGHEST_DEGREE for degree in path)
    if not inside or max(path) - min(path) > OCTAVE_DEGREES or len(set(path)) < 2:
        return None
    return tuple(path)


@functools.cache
def room_motif(area_key: str) -> tuple[tuple[int, float], ...]:
    """
    Return the motif of a room: 3 to 5 notes as (scale degree, beats), the same every time for the same room.

    It stays within an octave, moves mostly by step with at most one leap of a fifth or less, ends on the root, the
    third or the fifth of the scale, and is 4 or 8 beats long. The caller moves it into the key and mode of the day.

    :param area_key: The room or the house, such as the id of an area.
    """
    rng = np.random.default_rng(stable_hash("room_motif", area_key) % 2**63)
    count = MOTIF_NOTES[int(rng.integers(0, len(MOTIF_NOTES)))]
    beats = MOTIF_BEATS[int(rng.integers(0, len(MOTIF_BEATS)))]
    for _ in range(MOTIF_ATTEMPTS):
        degrees = _degrees(rng, count)
        if degrees is not None:
            return tuple(zip(degrees, _note_lengths(rng, count, beats), strict=True))
    return FALLBACK_MOTIF
