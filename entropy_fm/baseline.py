"""
A rolling baseline per entity: what the house usually does, built from the days before.

For every entity the baseline keeps the hours of the day in which its episodes usually start, how
many days it is active at all, how many episodes there are in a day, the typical time between
episodes, and for climate sensors the usual level. Surprise (see `surprise`) compares a day with it:
what always happens fades into the texture, what does not is featured.

An episode is one stretch in which something is going on: a door open, a light on, motion, an
appliance working. The module is pure: it works on `Series` and imports nothing from Music
Assistant, so the baseline can be unit tested and cached as plain JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .timeline import BINARY_KINDS, MIN_POWER_PEAK, VoiceKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .timeline import Series

SCHEMA = 1
DAY_SECONDS = 86400.0
# a baseline of fewer days than this is trusted only in part
MIN_EVIDENCE_DAYS = 4
POWER_KINDS = frozenset({VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER})
# a power sensor has episodes only when it has a clear on and off level
MIN_POWER_SPAN = 3.0


def episodes(series: Series, end: float = DAY_SECONDS) -> list[tuple[float, float]]:
    """
    Return the stretches of a day in which something is going on, as seconds from midnight.

    :param series: One entity on one day.
    :param end: Where the day ends, for a stretch that is still going.
    """
    if series.kind in BINARY_KINDS:
        active = series.norm > 0.5
    elif series.kind in POWER_KINDS:
        values = series.values
        low, high = float(np.percentile(values, 10)), float(np.percentile(values, 98))
        if high < MIN_POWER_PEAK or high < MIN_POWER_SPAN * max(low, 1e-9):
            return []
        active = values > low + 0.25 * (high - low)
    else:
        return []
    found: list[tuple[float, float]] = []
    start: float | None = None
    for time, on in zip(series.times, active, strict=True):
        if on and start is None:
            start = float(time)
        elif not on and start is not None:
            found.append((start, float(time)))
            start = None
    if start is not None:
        found.append((start, max(end, start)))
    return found


@dataclass(frozen=True)
class EntityBaseline:
    """What an entity usually does."""

    entity_id: str
    # how many days it is built from
    days: int
    # per hour of the day: the share of days with an episode that starts in it
    hours: tuple[float, ...]
    # the share of days with any episode
    active_days: float
    # episodes per day
    rate: float
    # the typical time between two episodes, in seconds, or 0
    cycle_s: float
    # for climate sensors: the mean of the day means, and how much they vary
    mean: float
    spread: float

    @property
    def evidence(self) -> float:
        """Return how far to trust the baseline: 1 with enough days, less with fewer."""
        return min(1.0, self.days / MIN_EVIDENCE_DAYS)

    def familiarity(self, hour: float) -> float:
        """
        Return how usual it is for an episode to start around an hour, from 0 to 1.

        :param hour: The hour of the day, 0 to 24; a neighbouring hour counts for half.
        """
        values = np.asarray(self.hours)
        smooth = np.minimum(1.0, values + 0.5 * (np.roll(values, 1) + np.roll(values, -1)))
        position = (hour % 24.0) - 0.5
        first = int(np.floor(position))
        fraction = position - first
        return float(smooth[first % 24] * (1.0 - fraction) + smooth[(first + 1) % 24] * fraction)

    def to_json(self) -> dict[str, Any]:
        """Return the baseline as plain data for the cache."""
        return {
            "id": self.entity_id,
            "days": self.days,
            "hours": [round(value, 4) for value in self.hours],
            "active": round(self.active_days, 4),
            "rate": round(self.rate, 3),
            "cycle": round(self.cycle_s, 1),
            "mean": round(self.mean, 4),
            "spread": round(self.spread, 4),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EntityBaseline:
        """
        Read a baseline from what `to_json` made.

        :param data: The stored data.
        """
        return cls(
            str(data["id"]),
            int(data["days"]),
            tuple(float(value) for value in data["hours"]),
            float(data["active"]),
            float(data["rate"]),
            float(data["cycle"]),
            float(data["mean"]),
            float(data["spread"]),
        )


def has_data(day: Sequence[Series]) -> bool:
    """
    Return True when a day has any history at all, which tells a quiet day from a purged one.

    :param day: The series of the entities of one area on one day.
    """
    return any(len(series.times) > 1 for series in day)


def build_baseline(entity_id: str, days: Sequence[Series]) -> EntityBaseline:
    """
    Build the baseline of an entity from its series on the days before.

    :param entity_id: The entity.
    :param days: The entity on each of those days.
    """
    count = len(days)
    hours = np.zeros(24)
    active = 0
    total = 0
    gaps: list[float] = []
    means: list[float] = []
    for series in days:
        found = episodes(series)
        active += bool(found)
        total += len(found)
        for hour in {int(start // 3600) % 24 for start, _ in found}:
            hours[hour] += 1.0
        gaps.extend(np.diff([start for start, _ in found]).tolist())
        means.append(float(np.mean(series.values)))
    scale = max(count, 1)
    return EntityBaseline(
        entity_id,
        count,
        tuple(float(value) for value in hours / scale),
        active / scale,
        total / scale,
        float(np.median(gaps)) if gaps else 0.0,
        float(np.mean(means)) if means else 0.0,
        float(np.std(means)) if means else 0.0,
    )


def build_baselines(days: Sequence[Sequence[Series]]) -> dict[str, EntityBaseline]:
    """
    Build the baseline of every entity from the series of the days before.

    :param days: For each day, the series of the entities of one area; days without data are skipped.
    """
    filled = [day for day in days if has_data(day)]
    by_entity: dict[str, list[Series]] = {}
    for day in filled:
        for series in day:
            by_entity.setdefault(series.entity_id, []).append(series)
    return {entity_id: build_baseline(entity_id, series) for entity_id, series in by_entity.items()}
