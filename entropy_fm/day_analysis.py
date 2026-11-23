"""
Read a day as a story: how lively it was hour by hour, where it changes, and what stood out.

A day of history is a lot of entities and a lot of nothing. This module boils it down to a curve of
activity (one value for every quarter of an hour), the five to seven stretches of the day that the
curve falls into, the salient moments (an appliance that ran, a hob that cooked, something unusual,
the busiest time) and a few numbers for the character of the day. `plan` turns that into the curves of a soundscape.

The module is pure and deterministic, and imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from .energy import activity
from .references import showers
from .timeline import DAY_SECONDS, POWER_KINDS, SECONDS_PER_HOUR, VoiceKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .timeline import Series

# the curve has a value for every quarter of an hour
STEP = 900.0
# a stretch of the day is at least this long (in steps): an hour and a half
MIN_SECTION_STEPS = 6
MIN_SECTIONS = 5
MAX_SECTIONS = 7
# kinds that do not make anything happen: they only colour the day
QUIET_KINDS = frozenset(
    {VoiceKind.SUN, VoiceKind.TEMPERATURE, VoiceKind.HUMIDITY, VoiceKind.PAD_PARTIAL}
)
# kinds that are read as a level, not counted as things that happen: the CO2 of a room changes all the time
LEVEL_KINDS = QUIET_KINDS | {VoiceKind.AIR}
# when the series are measured against their own days, a quiet day stays quiet: the curve is stretched to at least this
# range (the range of 1 stretches every day to the full scale, which is what a day without a reference gets)
REFERENCE_SPAN = 0.4
# the names of the stretches of a day
LABEL_NIGHT = "night"
LABEL_WAKING = "waking"
LABEL_DAYTIME = "daytime"
LABEL_COOKING = "cooking"
LABEL_EVENING = "evening"
LABEL_WINDDOWN = "winddown"
# the kinds of the moments that stood out
MOMENT_APPLIANCE = "appliance"
MOMENT_PHASE = "phase"
MOMENT_COOKING = "cooking"
MOMENT_ANOMALY = "anomaly"
MOMENT_SHOWER = "shower"
MOMENT_PEAK = "peak"
# the hours of the day (on the clock) where the stretches change: night is before the first and from the last
NIGHT_END_HOUR = 5.5
NIGHT_START_HOUR = 23.0
WAKING_END_HOUR = 10.0
DAYTIME_END_HOUR = 17.0
EVENING_END_HOUR = 21.5
# the busiest stretch is cooking when it is between these hours
COOKING_START_HOUR = 11.0
# a step of the normalised value that is more than this counts as something that happened
MOVE_THRESHOLD = 0.05
# the curve is the energy macro and the density of events, in these shares, and the density is smoothed over this many steps
BASE_SHARE = 0.65
EVENTS_SHARE = 0.35
DENSITY_SMOOTH_STEPS = 3
ENERGY_SMOOTH_STEPS = 2.0
MIN_CURVE_RANGE = 1e-9
FLAT_CURVE_LEVEL = 0.5
# a day that changes more than these spreads of its curve has one section more, and another
EVENTFUL_SPREAD = 0.22
VERY_EVENTFUL_SPREAD = 0.3
MIN_GAIN_STEP = 1e-12
# a rising crossing of these levels is an appliance run, and a hob that is switched on
APPLIANCE_LEVEL = 0.2
HOB_LEVEL = 0.05
PHASE_STRENGTH = 0.5
# the evening (warmth) and the day (brightness), in hours of the clock
WARMTH_FROM_HOUR = 17.0
WARMTH_UNTIL_HOUR = 23.5
BRIGHTNESS_FROM_HOUR = 8.0
BRIGHTNESS_UNTIL_HOUR = 17.0
# busyness: the mean of the curve scaled, each anomaly adds this much, and it is halved with the share of the events of a day that is this many
BUSYNESS_CURVE_SCALE = 1.4
BUSYNESS_ANOMALY = 0.15
BUSYNESS_EVENTS_FULL = 400.0


@dataclass(frozen=True)
class DaySection:
    """A stretch of the day that is of one kind, in seconds of the day."""

    start: float
    end: float
    # night, waking, daytime, cooking, evening or winddown
    label: str
    # the mean activity in it, 0 to 1
    level: float

    @property
    def length(self) -> float:
        """Return the length in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class Moment:
    """Something that stood out: when (seconds of the day), what kind, how much."""

    time: float
    kind: str
    strength: float
    # an entity id for something unusual; never used in a prompt
    source: str = ""


@dataclass(frozen=True)
class DayAnalysis:
    """What a day was like."""

    curve: np.ndarray
    covered: float
    sections: tuple[DaySection, ...]
    moments: tuple[Moment, ...] = field(default_factory=tuple)
    # how much happened, how much of it was in the evening (warm), how much by day (bright); 0 to 1
    busyness: float = 0.5
    warmth: float = 0.5
    brightness: float = 0.5
    # the series were measured against their own previous days, so the curve says how lively the day was
    measured: bool = False

    def moments_of(self, kind: str) -> list[Moment]:
        """Return the moments of one kind, in the order of the day."""
        return [moment for moment in self.moments if moment.kind == kind]


def activity_curve(series: Sequence[Series], covered: float = DAY_SECONDS) -> np.ndarray:
    """
    Return how lively the house was in each quarter of an hour, 0 to 1.

    It is what the energy macro says (appliances working, lights on, things happening) and the
    density of events, so that a day of many small events counts as busy even without a big load.
    Series that are measured against their own days (`references.rescale`) keep a quiet day quiet:
    the curve is stretched to at least `REFERENCE_SPAN` of the scale. Others are stretched to all of it.

    :param series: Every voice of the day.
    :param covered: How many seconds of the day have happened (less than a day for today).
    """
    count = max(int(min(covered, DAY_SECONDS) / STEP), 1)
    times = (np.arange(count) + 0.5) * STEP
    base = activity(series, times, smooth=STEP * ENERGY_SMOOTH_STEPS)
    density = np.zeros(count)
    for item in series:
        if item.kind in LEVEL_KINDS:
            continue
        moved = np.flatnonzero(np.abs(np.diff(item.norm)) > MOVE_THRESHOLD) + 1
        if len(moved):
            hits = np.clip((item.times[moved] // STEP).astype(int), 0, count - 1)
            np.add.at(density, hits, 1.0)
    smoothed = np.convolve(
        density, np.ones(DENSITY_SMOOTH_STEPS) / float(DENSITY_SMOOTH_STEPS), mode="same"
    )
    peak = float(smoothed.max())
    events = smoothed / peak if peak > 0 else smoothed
    curve = BASE_SHARE * base + EVENTS_SHARE * events
    low, high = float(curve.min()), float(curve.max())
    if high - low < MIN_CURVE_RANGE:
        return np.full(count, FLAT_CURVE_LEVEL)
    if any(item.ref is not None for item in series):
        return np.asarray(np.clip((curve - low) / max(high - low, REFERENCE_SPAN), 0.0, 1.0))
    return np.asarray((curve - low) / (high - low), dtype=np.float64)


def change_points(curve: np.ndarray, sections: int) -> list[int]:
    """
    Cut a curve into `sections` stretches where it changes, with a binary segmentation.

    The cut that lowers the squared deviation from the mean of the stretches the most is made first,
    and so on, never leaving a stretch shorter than `MIN_SECTION_STEPS`. The result is the list of
    first steps of the stretches after the first.

    :param curve: The activity, one value per step.
    :param sections: How many stretches to cut it into.
    """
    cuts: list[int] = []
    bounds = [0, len(curve)]
    while len(bounds) - 1 < sections:
        best: tuple[float, int] | None = None
        for left, right in pairwise(bounds):
            if right - left < 2 * MIN_SECTION_STEPS:
                continue
            whole = float(((curve[left:right] - curve[left:right].mean()) ** 2).sum())
            for cut in range(left + MIN_SECTION_STEPS, right - MIN_SECTION_STEPS + 1):
                parts = sum(
                    float(((curve[a:b] - curve[a:b].mean()) ** 2).sum())
                    for a, b in ((left, cut), (cut, right))
                )
                gain = whole - parts
                if best is None or gain > best[0] + MIN_GAIN_STEP:
                    best = (gain, cut)
        if best is None:
            break
        bounds = sorted([*bounds, best[1]])
        cuts.append(best[1])
    return sorted(cuts)


def sections_of(curve: np.ndarray, covered: float) -> tuple[DaySection, ...]:
    """
    Return the five to seven stretches of the day that the curve falls into.

    The number follows how much the day changes: a flat day has five, an eventful one up to seven.

    :param curve: The activity curve.
    :param covered: How many seconds of the day have happened.
    """
    steps = len(curve)
    wanted = MIN_SECTIONS
    spread = float(np.std(curve))
    if spread > EVENTFUL_SPREAD:
        wanted += 1
    if spread > VERY_EVENTFUL_SPREAD:
        wanted += 1
    wanted = int(
        np.clip(wanted, MIN_SECTIONS, min(MAX_SECTIONS, max(steps // MIN_SECTION_STEPS, 1)))
    )
    cuts = change_points(curve, wanted)
    edges = [0, *cuts, steps]
    levels = [float(curve[a:b].mean()) for a, b in pairwise(edges)]
    peak = int(np.argmax(levels))
    end_of_day = min(covered, DAY_SECONDS)
    found = []
    for index, (a, b) in enumerate(pairwise(edges)):
        start = a * STEP
        end = end_of_day if b == steps else b * STEP
        found.append(
            DaySection(start, end, _label(start, end, is_peak=index == peak), levels[index])
        )
    return tuple(found)


def analyze_day(
    series: Sequence[Series],
    covered: float = DAY_SECONDS,
    anomalies: Sequence[tuple[float, float, str]] = (),
) -> DayAnalysis:
    """
    Analyse a day: its curve, its stretches, its salient moments and its character.

    :param series: Every voice of the day, with times in seconds since midnight.
    :param covered: How many seconds of the day have happened.
    :param anomalies: Unusual events as (second of the day, score, entity id), from `surprise`.
    """
    curve = activity_curve(series, covered)
    sections = sections_of(curve, covered)
    hours = (np.arange(len(curve)) + 0.5) * STEP / SECONDS_PER_HOUR
    total = float(curve.sum()) or 1.0
    warmth = float(curve[(hours >= WARMTH_FROM_HOUR) & (hours < WARMTH_UNTIL_HOUR)].sum() / total)
    brightness = float(
        curve[(hours >= BRIGHTNESS_FROM_HOUR) & (hours < BRIGHTNESS_UNTIL_HOUR)].sum() / total
    )
    busyness = float(
        np.clip(curve.mean() * BUSYNESS_CURVE_SCALE + BUSYNESS_ANOMALY * len(anomalies), 0.0, 1.0)
    )
    # a flat curve is normalised to 0.5, so how much really happened is told by the events
    events = sum(
        int((np.abs(np.diff(s.norm)) > MOVE_THRESHOLD).sum())
        for s in series
        if s.kind not in LEVEL_KINDS
    )
    busyness = float(
        np.clip(0.5 * busyness + 0.5 * min(events / BUSYNESS_EVENTS_FULL, 1.0), 0.0, 1.0)
    )
    return DayAnalysis(
        curve,
        min(covered, DAY_SECONDS),
        sections,
        _moments(series, curve, anomalies, covered),
        busyness,
        warmth,
        brightness,
        any(item.ref is not None for item in series),
    )


def _label(start: float, end: float, *, is_peak: bool) -> str:
    """Name a stretch of the day by when it is and how busy it is."""
    middle = ((start + end) / 2.0 % DAY_SECONDS) / SECONDS_PER_HOUR
    if middle < NIGHT_END_HOUR or middle >= NIGHT_START_HOUR:
        return LABEL_NIGHT
    if is_peak and COOKING_START_HOUR <= middle <= EVENING_END_HOUR:
        return LABEL_COOKING
    if middle < WAKING_END_HOUR:
        return LABEL_WAKING
    if middle < DAYTIME_END_HOUR:
        return LABEL_DAYTIME
    if middle < EVENING_END_HOUR:
        return LABEL_EVENING
    return LABEL_WINDDOWN


def _moments(
    series: Sequence[Series],
    curve: np.ndarray,
    anomalies: Sequence[tuple[float, float, str]],
    covered: float = DAY_SECONDS,
) -> tuple[Moment, ...]:
    """Collect the salient moments: appliance runs, hob cooking, anomalies, showers and the busiest time."""
    found: list[Moment] = []
    for item in series:
        if item.kind in POWER_KINDS and item.kind != VoiceKind.DRONE:
            found.extend(
                Moment(float(t), MOMENT_APPLIANCE, float(item.norm.max()), item.entity_id)
                for t in item.crossings(0.0, DAY_SECONDS, APPLIANCE_LEVEL, rising=True)
            )
        elif item.kind == VoiceKind.PHASE:
            found.extend(
                Moment(float(t), MOMENT_PHASE, PHASE_STRENGTH, item.entity_id)
                for t in item.changes(0.0, DAY_SECONDS)
            )
        elif item.kind == VoiceKind.HOB:
            found.extend(
                Moment(float(t), MOMENT_COOKING, float(item.norm.max()), item.entity_id)
                for t in item.crossings(0.0, DAY_SECONDS, HOB_LEVEL, rising=True)
            )
    found.extend(
        Moment(float(t), MOMENT_ANOMALY, float(score), entity) for t, score, entity in anomalies
    )
    for item in series:
        found.extend(
            Moment(float(t), MOMENT_SHOWER, float(strength), item.entity_id)
            for t, strength in showers(item, covered)
        )
    found.append(Moment((int(np.argmax(curve)) + 0.5) * STEP, MOMENT_PEAK, float(curve.max())))
    return tuple(sorted(found, key=lambda m: (m.time, m.kind, m.source)))
