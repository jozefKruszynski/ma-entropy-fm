"""
Measure every entity against its own previous days, and leave out what says nothing.

A CO2 monitor in a bedroom reads 400 to 3700 ppm, one in an office 400 to 1200: both are as lively on their busiest
day. So the values of a numeric sensor are scaled with the percentiles of the days before (`baseline`), not with the
extremes of one day, which makes a quiet day quiet and keeps a sensor-rich house from sitting at its maximum. A sensor
that is dead (it said nothing today, or nothing has changed for days) is dropped. A spike of humidity far above the
usual level of a room (a shower) is found here too.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .baseline import day_weights, weighted_percentiles
from .timeline import (
    BINARY_KINDS,
    DAY_SECONDS,
    MIN_POWER_PEAK,
    MIN_TEMPERATURE_SPAN,
    POWER_KINDS,
    SECONDS_PER_HOUR,
    VoiceKind,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .baseline import EntityBaseline
    from .timeline import Reference, Series

# the days a reference needs before it replaces the percentiles of the day itself
MIN_REFERENCE_DAYS = 3
# kinds whose values are a level that says something when it is constant: a sensor that never moves is dead
NUMERIC_KINDS = frozenset(
    {
        VoiceKind.DRONE,
        VoiceKind.DRUM,
        VoiceKind.DISHWASHER,
        VoiceKind.TEMPERATURE,
        VoiceKind.HUMIDITY,
        VoiceKind.PAD_PARTIAL,
        VoiceKind.AIR,
    }
)
# a reading that moves less than this is constant (a stuck sensor reads the same to the last digit)
FLAT_EPSILON = 1e-6
# the narrowest scale of a CO2 monitor: a room that always reads 400 to 450 ppm has nothing to say
MIN_AIR_SPAN = 250.0
# a spike of humidity is a shower when it is this far (percent points) over the usual level, and at least this
# many times the usual spread, of the room
SHOWER_RISE = 12.0
SHOWER_SPREAD = 2.5
# a shower closer than this (seconds) to the one before it is the same one
SHOWER_MERGE = 1800.0
SHOWER_MAX = 3
# humidity that stays up for longer than this is the weather (or a laundry day), not a shower
SHOWER_LONGEST = 3 * SECONDS_PER_HOUR
# a series needs this many readings to hold a shower
MIN_SHOWER_READINGS = 4
# the percentiles of the day that give the usual level and spread of a room without a reference
SHOWER_PERCENTILES = (10.0, 50.0, 90.0)
# the strength of a shower is this at the limit, grows by one for each of this many points above it, within these bounds
SHOWER_STRENGTH_BASE = 0.4
SHOWER_STRENGTH_POINTS = 25.0
SHOWER_STRENGTH_MIN = 0.3
SHOWER_STRENGTH_MAX = 1.0


def is_dead(item: Series, ref: Reference | None) -> bool:
    """
    Return True for a sensor that says nothing: a numeric one that reads the same all day and did not move before.

    A binary entity (a light, a door) is dead only when the days before are known and nothing changed in any of
    them or today: a door that stays shut on a quiet day is not dead, a lamp that is always on is.

    :param item: The series of the day.
    :param ref: What it read on the days before, when known.
    """
    if item.kind in BINARY_KINDS:
        return ref is not None and ref.flat and float(np.ptp(item.values)) < FLAT_EPSILON
    if item.kind not in NUMERIC_KINDS:
        return False
    constant = float(np.ptp(item.values)) < FLAT_EPSILON
    if not constant:
        return False
    # a sensor that was quiet today but moved on the days before (a TV that was off) is only quiet
    return ref is None or ref.flat or (ref.high - ref.low) < FLAT_EPSILON


def rescale(
    series: Sequence[Series], baselines: Mapping[str, EntityBaseline] | None = None
) -> tuple[list[Series], list[str]]:
    """
    Measure each series against its own previous days and drop the dead ones.

    Without a baseline (or with fewer than `MIN_REFERENCE_DAYS` days) a series keeps the scale of its day. When every
    series would be dropped none is.

    :param series: The series of the area or the house on the day.
    :param baselines: What each entity usually does, by entity id.
    :return: The series to play, and the ids of the ones that were dropped.
    """
    found = baselines or {}
    kept: list[Series] = []
    dropped: list[str] = []
    for item in series:
        ref = _reference(found.get(item.entity_id))
        if is_dead(item, ref):
            dropped.append(item.entity_id)
            continue
        if ref is None:
            kept.append(item)
            continue
        norm = _normalised(item, ref)
        kept.append(replace(item, ref=ref) if norm is None else replace(item, norm=norm, ref=ref))
    if not kept:
        return list(series), []
    return kept, dropped


def showers(item: Series, covered: float = DAY_SECONDS) -> list[tuple[float, float]]:
    """
    Return the humidity spikes of a series as (second of the day, strength 0 to 1): a shower, a boiling pot.

    A spike is a stretch over the usual level of the room by `SHOWER_RISE` points and `SHOWER_SPREAD` times its usual
    spread; the level is the reference of the days before, or the median of the day itself without one. The moment is
    where the humidity climbs the most, and its strength grows with how far above the level it went.

    :param item: A humidity series (other kinds give nothing).
    :param covered: How many seconds of the day have happened.
    """
    if item.kind != VoiceKind.HUMIDITY or len(item.times) < MIN_SHOWER_READINGS:
        return []
    if item.ref is not None:
        mid, spread = item.ref.mid, item.ref.high - item.ref.low
    else:
        weights = day_weights(item, covered)
        low, mid, high = weighted_percentiles(item.values, weights, SHOWER_PERCENTILES)
        spread = high - low
    limit = mid + max(SHOWER_RISE, SHOWER_SPREAD * spread)
    above = item.values > limit
    found: list[tuple[float, float]] = []
    index = 0
    count = len(item.values)
    while index < count:
        if not above[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and above[end + 1]:
            end += 1
        until = float(item.times[end + 1]) if end + 1 < count else covered
        if until - float(item.times[index]) > SHOWER_LONGEST:
            index = end + 1
            continue
        peak = float(item.values[index : end + 1].max())
        # where it climbs: the start of the stretch
        start = float(item.times[index])
        strength = float(
            np.clip(
                SHOWER_STRENGTH_BASE + (peak - limit) / SHOWER_STRENGTH_POINTS,
                SHOWER_STRENGTH_MIN,
                SHOWER_STRENGTH_MAX,
            )
        )
        if start < covered:
            if found and start - found[-1][0] < SHOWER_MERGE:
                found[-1] = (found[-1][0], max(found[-1][1], strength))
            else:
                found.append((start, strength))
        index = end + 1
    return sorted(found, key=lambda moment: -moment[1])[:SHOWER_MAX]


def _reference(baseline: EntityBaseline | None) -> Reference | None:
    """Return the reference of a series when its baseline has enough days."""
    if baseline is None or baseline.days < MIN_REFERENCE_DAYS:
        return None
    return baseline.reference


def _normalised(item: Series, ref: Reference) -> np.ndarray | None:
    """Return the values of a series scaled with its reference, or None to keep the scale of the day."""
    values = item.values
    if item.kind == VoiceKind.AIR:
        span = max(ref.high - ref.low, MIN_AIR_SPAN)
        return np.clip((values - ref.low) / span, 0.0, 1.0)
    if item.kind in POWER_KINDS:
        if ref.peak < MIN_POWER_PEAK:
            return None
        return np.clip(values / ref.peak, 0.0, 1.0)
    if item.kind == VoiceKind.TEMPERATURE:
        span = max(ref.high - ref.low, MIN_TEMPERATURE_SPAN)
        return np.clip((values - ref.mid) / span + 0.5, 0.0, 1.0)
    return None
