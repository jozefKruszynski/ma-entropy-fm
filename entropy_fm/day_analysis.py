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
from .timeline import VoiceKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .timeline import Series

DAY_SECONDS = 86400.0
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
POWER_KINDS = frozenset({VoiceKind.DISHWASHER, VoiceKind.DRUM, VoiceKind.DRONE})


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

    def moments_of(self, kind: str) -> list[Moment]:
        """Return the moments of one kind, in the order of the day."""
        return [moment for moment in self.moments if moment.kind == kind]


def activity_curve(series: Sequence[Series], covered: float = DAY_SECONDS) -> np.ndarray:
    """
    Return how lively the house was in each quarter of an hour, 0 to 1.

    It is what the energy macro says (appliances working, lights on, things happening) and the
    density of events, so that a day of many small events counts as busy even without a big load.

    :param series: Every voice of the day.
    :param covered: How many seconds of the day have happened (less than a day for today).
    """
    count = max(int(min(covered, DAY_SECONDS) / STEP), 1)
    times = (np.arange(count) + 0.5) * STEP
    base = activity(series, times, smooth=STEP * 2.0)
    density = np.zeros(count)
    for item in series:
        if item.kind in QUIET_KINDS:
            continue
        moved = np.flatnonzero(np.abs(np.diff(item.norm)) > 0.05) + 1
        if len(moved):
            hits = np.clip((item.times[moved] // STEP).astype(int), 0, count - 1)
            np.add.at(density, hits, 1.0)
    smoothed = np.convolve(density, np.ones(3) / 3.0, mode="same")
    peak = float(smoothed.max())
    events = smoothed / peak if peak > 0 else smoothed
    curve = 0.65 * base + 0.35 * events
    low, high = float(curve.min()), float(curve.max())
    if high - low < 1e-9:
        return np.full(count, 0.5)
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
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, cut)
        if best is None:
            break
        bounds = sorted([*bounds, best[1]])
        cuts.append(best[1])
    return sorted(cuts)


def _label(start: float, end: float, *, is_peak: bool) -> str:
    """Name a stretch of the day by when it is and how busy it is."""
    middle = ((start + end) / 2.0 % DAY_SECONDS) / 3600.0
    if middle < 5.5 or middle >= 23.0:
        return "night"
    if is_peak and 11.0 <= middle <= 21.5:
        return "cooking"
    if middle < 10.0:
        return "waking"
    if middle < 17.0:
        return "daytime"
    if middle < 21.5:
        return "evening"
    return "winddown"


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
    if spread > 0.22:
        wanted += 1
    if spread > 0.3:
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


def _moments(
    series: Sequence[Series], curve: np.ndarray, anomalies: Sequence[tuple[float, float, str]]
) -> tuple[Moment, ...]:
    """Collect the salient moments: appliance runs, hob cooking, anomalies and the busiest time."""
    found: list[Moment] = []
    for item in series:
        if item.kind in POWER_KINDS and item.kind != VoiceKind.DRONE:
            found.extend(
                Moment(float(t), "appliance", float(item.norm.max()), item.entity_id)
                for t in item.crossings(0.0, DAY_SECONDS, 0.2, rising=True)
            )
        elif item.kind == VoiceKind.PHASE:
            found.extend(
                Moment(float(t), "phase", 0.5, item.entity_id)
                for t in item.changes(0.0, DAY_SECONDS)
            )
        elif item.kind == VoiceKind.HOB:
            found.extend(
                Moment(float(t), "cooking", float(item.norm.max()), item.entity_id)
                for t in item.crossings(0.0, DAY_SECONDS, 0.05, rising=True)
            )
    found.extend(
        Moment(float(t), "anomaly", float(score), entity) for t, score, entity in anomalies
    )
    found.append(Moment((int(np.argmax(curve)) + 0.5) * STEP, "peak", float(curve.max())))
    return tuple(sorted(found, key=lambda m: (m.time, m.kind, m.source)))


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
    hours = (np.arange(len(curve)) + 0.5) * STEP / 3600.0
    total = float(curve.sum()) or 1.0
    warmth = float(curve[(hours >= 17.0) & (hours < 23.5)].sum() / total)
    brightness = float(curve[(hours >= 8.0) & (hours < 17.0)].sum() / total)
    busyness = float(np.clip(curve.mean() * 1.4 + 0.15 * len(anomalies), 0.0, 1.0))
    # a flat curve is normalised to 0.5, so how much really happened is told by the events
    events = sum(
        int((np.abs(np.diff(s.norm)) > 0.05).sum()) for s in series if s.kind not in QUIET_KINDS
    )
    busyness = float(np.clip(0.5 * busyness + 0.5 * min(events / 400.0, 1.0), 0.0, 1.0))
    return DayAnalysis(
        curve,
        min(covered, DAY_SECONDS),
        sections,
        _moments(series, curve, anomalies),
        busyness,
        warmth,
        brightness,
    )
