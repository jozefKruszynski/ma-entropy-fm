"""
How lively the house is: one value between 0 and 1, from what is happening in it.

Appliances working, people moving, lights on and the air of a room filling with CO2 make it rise; it is the base of
the activity curve of a day (`day_analysis`) and of the traits of a day (`palette`). Motion and occupancy sensors are
the primary sign that someone is there; a house that has none is read from the CO2 of its rooms (the level is how
many people, a rise is someone coming in), from plugs and meters that draw power, and from how long and how often
the lights are on. Each term is only there when the house has its sensors, and the weights are those of the terms that
are, so the sources are blended per room by what the room has.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .synth import smoothstep
from .timeline import VoiceKind, as_array

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from .timeline import Series

# the half hour of house time over which activity is averaged
SMOOTH_SECONDS = 1800.0
POWER_KINDS = frozenset({VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER})
EVENT_KINDS = frozenset({VoiceKind.PLUCK, VoiceKind.BLIP, VoiceKind.MEDIA})
# a fridge that runs a third of the time is not activity: only what rises above it counts
POWER_FLOOR = 0.25
# a rise of CO2 of this share of the usual range in twenty minutes is someone entering, in full
AIR_RISE = 0.25
RISE_SECONDS = 1200.0
# how much of the CO2 of the night counts: someone asleep is there, and still
SLEEP_QUIET = 0.55
# humidity over its usual level by this many points is the weather; the next points are someone being there
STEAM_RISE = 6.0
STEAM_SPAN = 20.0
# lights that switch on this many times an hour are as lively as lights can be
SWITCHES_PER_HOUR = 2.5


def occupancy(
    air: Sequence[Series], house: npt.ArrayLike, smooth: float = SMOOTH_SECONDS
) -> npt.NDArray[np.float64]:
    """
    Return how many people are about, from the CO2 of the rooms, 0 to 1.

    The level (against what the room usually reads) says how many are there; a rise within twenty minutes says that
    someone has come in and counts a bit more than the level.

    :param air: The CO2 series, already measured against the days before.
    :param house: Moments in seconds of house time.
    :param smooth: The width in seconds over which the level is averaged.
    """
    times = as_array(house)
    level = np.mean([item.at(times, smooth) for item in air], axis=0)
    rise = np.max(
        [
            np.clip(
                (item.at(times, RISE_SECONDS / 2) - item.at(times - RISE_SECONDS, RISE_SECONDS / 2))
                / AIR_RISE,
                0.0,
                1.0,
            )
            for item in air
        ],
        axis=0,
    )
    clock = (times / 3600.0) % 24.0
    asleep = np.where(
        clock >= 12.0,
        smoothstep((clock - 22.5) / 1.0),
        1.0 - smoothstep((clock - 6.0) / 1.5),
    )
    # a room that fills with CO2 in the night is a person asleep: present, and quiet
    return as_array(np.clip(0.8 * level + 0.5 * rise, 0.0, 1.0) * (1.0 - SLEEP_QUIET * asleep))


def steam_level(
    item: Series, house: npt.ArrayLike, smooth: float = SMOOTH_SECONDS
) -> npt.NDArray[np.float64]:
    """
    Return how far the humidity of a room is over its usual level, 0 to 1: someone has had a shower, or is cooking.

    The usual level is the median of the days before (of the day itself without them); a rise of up to
    `STEAM_RISE` points is the weather and counts for nothing.

    :param item: A humidity series.
    :param house: Moments in seconds of house time.
    :param smooth: The width in seconds over which it is averaged.
    """
    times = as_array(house)
    usual = item.ref.mid if item.ref is not None else float(np.median(item.values))
    level = item.at(times, smooth) * 100.0
    return as_array(np.clip((level - usual - STEAM_RISE) / STEAM_SPAN, 0.0, 1.0))


def light_level(
    item: Series, house: npt.ArrayLike, smooth: float = SMOOTH_SECONDS
) -> npt.NDArray[np.float64]:
    """
    Return how much a light says about the house, 0 to 1: how long it is on, how often it is switched.

    A lamp that stays on says less than one that is switched about, and a day with more light than usual (the hours
    of the days before are the measure) says more than one with less.

    :param item: A light or a switch.
    :param house: Moments in seconds of house time.
    :param smooth: The width in seconds over which it is averaged.
    """
    times = as_array(house)
    on = item.at(times, smooth)
    starts = item.onsets(0.0, float("inf"))
    if len(starts):
        low = np.searchsorted(starts, times - smooth / 2.0)
        high = np.searchsorted(starts, times + smooth / 2.0)
        switches = np.clip((high - low) / (smooth / 3600.0) / SWITCHES_PER_HOUR, 0.0, 1.0)
    else:
        switches = np.zeros(len(times))
    level = 0.65 * on + 0.35 * switches
    if item.ref is not None and item.ref.on_hours > 0.05:
        today = float(
            np.sum(np.diff(np.append(item.times, max(86400.0, item.times[-1])))[item.norm > 0.5])
        )
        level = level * float(np.clip(today / 3600.0 / item.ref.on_hours, 0.6, 1.4))
    return as_array(np.clip(level, 0.0, 1.0))


def activity(
    series: Sequence[Series], house: npt.ArrayLike, smooth: float = SMOOTH_SECONDS
) -> npt.NDArray[np.float64]:
    """
    Return how lively the house is at some moments, from 0 to 1.

    :param series: The voices of the track.
    :param house: Moments in seconds of house time.
    :param smooth: The width in seconds of house time over which activity is averaged.
    """
    times = as_array(house)
    terms: list[tuple[float, np.ndarray]] = []
    power = [item for item in series if item.kind in POWER_KINDS and not item.is_baseline_load]
    if power:
        loudest = np.max([item.at(times, smooth) for item in power], axis=0)
        terms.append((0.4, np.clip((loudest - POWER_FLOOR) / 0.6, 0.0, 1.0)))
    events = [item for item in series if item.kind in EVENT_KINDS]
    if events:
        share = np.mean([item.at(times, smooth) for item in events], axis=0)
        terms.append((0.4, np.clip(share * 3.0, 0.0, 1.0)))
    # an appliance that runs, a hob that is on: the more of them, the livelier
    working = [item for item in series if item.kind in {VoiceKind.PHASE, VoiceKind.HOB}]
    if working:
        busy = np.max([item.at(times, smooth) for item in working], axis=0)
        terms.append((0.3, np.clip(busy * 3.0, 0.0, 1.0)))
    air = [item for item in series if item.kind == VoiceKind.AIR]
    if air:
        terms.append((0.25 if events else 0.5, occupancy(air, times, smooth)))
    steam = [item for item in series if item.kind == VoiceKind.HUMIDITY and len(item.times) > 3]
    if steam:
        terms.append((0.3, np.max([steam_level(item, times, smooth) for item in steam], axis=0)))
    lights = [item for item in series if item.kind == VoiceKind.CHORD]
    if lights:
        terms.append((0.2, np.mean([light_level(item, times, smooth) for item in lights], axis=0)))
    if not terms:
        return np.full(len(times), 0.5)
    total = sum(weight for weight, _ in terms)
    return as_array(sum(weight * value for weight, value in terms) / total)
