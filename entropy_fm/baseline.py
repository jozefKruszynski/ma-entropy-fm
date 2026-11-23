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
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .timeline import (
    BINARY_KINDS,
    DAY_SECONDS,
    HOURS_PER_DAY,
    MIN_POWER_PEAK,
    POWER_KINDS,
    SECONDS_PER_HOUR,
    Reference,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .timeline import Series

# a baseline of fewer days than this is trusted only in part
MIN_EVIDENCE_DAYS = 4
# the percentiles of the values that make the scale of an entity (see `timeline.Reference`)
REFERENCE_LOW, REFERENCE_MID, REFERENCE_HIGH = 5.0, 50.0, 95.0
# a power sensor has episodes only when it has a clear on and off level
MIN_POWER_SPAN = 3.0
# a binary entity is on above this, and a value that moves less than this is the same
BINARY_ON_LEVEL = 0.5
# a value that moves less than this is constant, and a span under it is none
SPAN_EPSILON = 1e-9
# a power sensor is on above its low level by this share of the way to its high one (the 10th and the 98th percentile)
POWER_LOW_PERCENTILE = 10
POWER_HIGH_PERCENTILE = 98
POWER_ON_SHARE = 0.25
# the highest daily value is the 98th percentile
PEAK_PERCENTILE = 98.0
PERCENT = 100.0
# the hours of the neighbours of an hour count for this much of it
NEIGHBOUR_SHARE = 0.5
# the decimals that the stored values keep
SHARE_DECIMALS = 4
RATE_DECIMALS = 3
CYCLE_DECIMALS = 1


def episodes(series: Series, end: float = DAY_SECONDS) -> list[tuple[float, float]]:
    """
    Return the stretches of a day in which something is going on, as seconds from midnight.

    :param series: One entity on one day.
    :param end: Where the day ends, for a stretch that is still going.
    """
    if series.kind in BINARY_KINDS:
        active = series.norm > BINARY_ON_LEVEL
    elif series.kind in POWER_KINDS:
        values = series.values
        low = float(np.percentile(values, POWER_LOW_PERCENTILE))
        high = float(np.percentile(values, POWER_HIGH_PERCENTILE))
        if high < MIN_POWER_PEAK or high < MIN_POWER_SPAN * max(low, SPAN_EPSILON):
            return []
        active = values > low + POWER_ON_SHARE * (high - low)
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
    # the scale of the values over those days: the 5th, 50th and 95th percentile, the highest daily 98th
    # percentile, the hours on and the times on in a day (binary entities), and whether nothing ever changed
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    peak: float = 0.0
    on_hours: float = 0.0
    on_count: float = 0.0
    flat: bool = False

    @property
    def reference(self) -> Reference:
        """Return the scale of the entity, to measure a new day against."""
        return Reference(
            self.days,
            self.low,
            self.mid,
            self.high,
            self.peak,
            self.on_hours,
            self.on_count,
            self.flat,
        )

    @property
    def evidence(self) -> float:
        """Return how far to trust the baseline: 1 with enough days, less with fewer."""
        return min(1.0, self.days / MIN_EVIDENCE_DAYS)

    def familiarity(self, hour: float) -> float:
        """
        Return how usual it is for an episode to start around an hour, from 0 to 1.

        :param hour: The hour of the day, 0 to 24; a neighbouring hour counts for half.
        """
        smooth = self._smooth_hours
        position = (hour % HOURS_PER_DAY) - 0.5
        first = int(np.floor(position))
        fraction = position - first
        return float(
            smooth[first % HOURS_PER_DAY] * (1.0 - fraction)
            + smooth[(first + 1) % HOURS_PER_DAY] * fraction
        )

    def to_json(self) -> dict[str, Any]:
        """Return the baseline as plain data for the cache."""
        return {
            "id": self.entity_id,
            "days": self.days,
            "hours": [round(value, SHARE_DECIMALS) for value in self.hours],
            "active": round(self.active_days, SHARE_DECIMALS),
            "rate": round(self.rate, RATE_DECIMALS),
            "cycle": round(self.cycle_s, CYCLE_DECIMALS),
            "mean": round(self.mean, SHARE_DECIMALS),
            "spread": round(self.spread, SHARE_DECIMALS),
            "low": round(self.low, SHARE_DECIMALS),
            "mid": round(self.mid, SHARE_DECIMALS),
            "high": round(self.high, SHARE_DECIMALS),
            "peak": round(self.peak, SHARE_DECIMALS),
            "on_hours": round(self.on_hours, RATE_DECIMALS),
            "on_count": round(self.on_count, RATE_DECIMALS),
            "flat": self.flat,
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
            float(data.get("low", 0.0)),
            float(data.get("mid", 0.0)),
            float(data.get("high", 0.0)),
            float(data.get("peak", 0.0)),
            float(data.get("on_hours", 0.0)),
            float(data.get("on_count", 0.0)),
            bool(data.get("flat", False)),
        )

    @cached_property
    def _smooth_hours(self) -> np.ndarray:
        values = np.asarray(self.hours)
        return np.asarray(
            np.minimum(1.0, values + NEIGHBOUR_SHARE * (np.roll(values, 1) + np.roll(values, -1)))
        )


def weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, percentiles: Sequence[float]
) -> list[float]:
    """
    Return percentiles of values that each count for as long as they lasted.

    :param values: The values.
    :param weights: How long each lasted, in seconds.
    :param percentiles: The percentiles to return, 0 to 100.
    """
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return [0.0 for _ in percentiles]
    order = np.argsort(values, kind="stable")
    ordered, share = values[order], np.cumsum(weights[order])
    share = (share - 0.5 * weights[order]) / share[-1]
    return [float(np.interp(p / PERCENT, share, ordered)) for p in percentiles]


def day_weights(series: Series, end: float = DAY_SECONDS) -> np.ndarray:
    """Return how long (seconds) each value of a series lasted, to the end of the day."""
    return np.maximum(np.diff(np.append(series.times, max(end, float(series.times[-1])))), 0.0)


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
    hours = np.zeros(HOURS_PER_DAY)
    active = 0
    total = 0
    gaps: list[float] = []
    means: list[float] = []
    pooled: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    peaks: list[float] = []
    on_seconds = 0.0
    on_periods = 0
    changed = False
    first_value: float | None = None
    for series in days:
        if float(np.ptp(series.values)) > SPAN_EPSILON or (
            first_value is not None and abs(float(series.values[0]) - first_value) > SPAN_EPSILON
        ):
            changed = True
        first_value = float(series.values[0])
        span = day_weights(series)
        pooled.append(series.values)
        weights.append(span)
        peaks.append(weighted_percentiles(series.values, span, (PEAK_PERCENTILE,))[0])
        if series.kind in BINARY_KINDS:
            on_seconds += float(span[series.values > BINARY_ON_LEVEL].sum())
        found = episodes(series)
        on_periods += len(found) if series.kind in BINARY_KINDS else 0
        active += bool(found)
        total += len(found)
        for hour in {int(start // SECONDS_PER_HOUR) % HOURS_PER_DAY for start, _ in found}:
            hours[hour] += 1.0
        gaps.extend(np.diff([start for start, _ in found]).tolist())
        means.append(float(np.mean(series.values)))
    scale = max(count, 1)
    values = np.concatenate(pooled) if pooled else np.zeros(0)
    low, mid, high = weighted_percentiles(
        values,
        np.concatenate(weights) if weights else np.zeros(0),
        (REFERENCE_LOW, REFERENCE_MID, REFERENCE_HIGH),
    )
    return EntityBaseline(
        entity_id,
        count,
        tuple(float(value) for value in hours / scale),
        active / scale,
        total / scale,
        float(np.median(gaps)) if gaps else 0.0,
        float(np.mean(means)) if means else 0.0,
        float(np.std(means)) if means else 0.0,
        low,
        mid,
        high,
        max(peaks, default=0.0),
        on_seconds / SECONDS_PER_HOUR / scale,
        on_periods / scale,
        not changed,
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
