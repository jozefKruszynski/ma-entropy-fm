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
from .timeline import (
    DAY_SECONDS,
    HOURS_PER_DAY,
    ON_LEVEL,
    PERCENT,
    POWER_KINDS,
    SECONDS_PER_HOUR,
    VoiceKind,
    as_array,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from .timeline import Series

# the half hour of house time over which activity is averaged
SMOOTH_SECONDS = 1800.0
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
# the CO2 of a room says how many are there, and a rise counts a bit less: the blend of the two
OCCUPANCY_LEVEL_WEIGHT = 0.8
OCCUPANCY_RISE_WEIGHT = 0.5
# the night, by the hour of the clock: asleep from this hour over a short ramp, and awake from this hour over a longer one
NOON_HOUR = 12.0
ASLEEP_HOUR = 22.5
ASLEEP_RAMP_HOURS = 1.0
AWAKE_HOUR = 6.0
AWAKE_RAMP_HOURS = 1.5
# lights that switch on this many times an hour are as lively as lights can be
SWITCHES_PER_HOUR = 2.5
# a light is how long it is on and how often it is switched, in these shares
LIGHT_ON_WEIGHT = 0.65
LIGHT_SWITCH_WEIGHT = 0.35
# a light is read against its usual hours when it was on for more than this many of them, and a day moves it this far
MIN_USUAL_ON_HOURS = 0.05
LIGHT_DAY_FACTOR_MIN = 0.6
LIGHT_DAY_FACTOR_MAX = 1.4
# the terms of the activity and their weights; the more of the house a room has sensors for, the more terms count
POWER_RANGE = 0.6
POWER_WEIGHT = 0.4
EVENT_WEIGHT = 0.4
EVENT_GAIN = 3.0
WORKING_WEIGHT = 0.3
WORKING_GAIN = 3.0
# the CO2 counts less when the room has motion sensors, which are the primary sign
AIR_WEIGHT = 0.5
AIR_WEIGHT_WITH_EVENTS = 0.25
STEAM_WEIGHT = 0.3
LIGHT_WEIGHT = 0.2
# a humidity series needs more points than this to say anything
MIN_STEAM_POINTS = 3
# a house with no sensors that say anything is as lively as this
IDLE_ACTIVITY = 0.5


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
    clock = (times / SECONDS_PER_HOUR) % HOURS_PER_DAY
    asleep = np.where(
        clock >= NOON_HOUR,
        smoothstep((clock - ASLEEP_HOUR) / ASLEEP_RAMP_HOURS),
        1.0 - smoothstep((clock - AWAKE_HOUR) / AWAKE_RAMP_HOURS),
    )
    # a room that fills with CO2 in the night is a person asleep: present, and quiet
    return as_array(
        np.clip(OCCUPANCY_LEVEL_WEIGHT * level + OCCUPANCY_RISE_WEIGHT * rise, 0.0, 1.0)
        * (1.0 - SLEEP_QUIET * asleep)
    )


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
    level = item.at(times, smooth) * PERCENT
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
        switches = np.clip((high - low) / (smooth / SECONDS_PER_HOUR) / SWITCHES_PER_HOUR, 0.0, 1.0)
    else:
        switches = np.zeros(len(times))
    level = LIGHT_ON_WEIGHT * on + LIGHT_SWITCH_WEIGHT * switches
    if item.ref is not None and item.ref.on_hours > MIN_USUAL_ON_HOURS:
        today = float(
            np.sum(
                np.diff(np.append(item.times, max(DAY_SECONDS, item.times[-1])))[
                    item.norm > ON_LEVEL
                ]
            )
        )
        level = level * float(
            np.clip(
                today / SECONDS_PER_HOUR / item.ref.on_hours,
                LIGHT_DAY_FACTOR_MIN,
                LIGHT_DAY_FACTOR_MAX,
            )
        )
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
        terms.append((POWER_WEIGHT, np.clip((loudest - POWER_FLOOR) / POWER_RANGE, 0.0, 1.0)))
    events = [item for item in series if item.kind in EVENT_KINDS]
    if events:
        share = np.mean([item.at(times, smooth) for item in events], axis=0)
        terms.append((EVENT_WEIGHT, np.clip(share * EVENT_GAIN, 0.0, 1.0)))
    # an appliance that runs, a hob that is on: the more of them, the livelier
    working = [item for item in series if item.kind in {VoiceKind.PHASE, VoiceKind.HOB}]
    if working:
        busy = np.max([item.at(times, smooth) for item in working], axis=0)
        terms.append((WORKING_WEIGHT, np.clip(busy * WORKING_GAIN, 0.0, 1.0)))
    air = [item for item in series if item.kind == VoiceKind.AIR]
    if air:
        terms.append(
            (AIR_WEIGHT_WITH_EVENTS if events else AIR_WEIGHT, occupancy(air, times, smooth))
        )
    steam = [
        item
        for item in series
        if item.kind == VoiceKind.HUMIDITY and len(item.times) > MIN_STEAM_POINTS
    ]
    if steam:
        terms.append(
            (STEAM_WEIGHT, np.max([steam_level(item, times, smooth) for item in steam], axis=0))
        )
    lights = [item for item in series if item.kind == VoiceKind.CHORD]
    if lights:
        terms.append(
            (LIGHT_WEIGHT, np.mean([light_level(item, times, smooth) for item in lights], axis=0))
        )
    if not terms:
        return np.full(len(times), IDLE_ACTIVITY)
    total = sum(weight for weight, _ in terms)
    return as_array(sum(weight * value for weight, value in terms) / total)
