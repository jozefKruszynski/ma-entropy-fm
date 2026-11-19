"""
How lively the house is: one value between 0 and 1, from what is happening in it.

Appliances working, people moving and lights on make it rise; it is the base of the activity curve of a
day (`day_analysis`) and of the traits of a day (`palette`).

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    power = [item for item in series if item.kind in POWER_KINDS]
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
    lights = [item for item in series if item.kind == VoiceKind.CHORD]
    if lights:
        terms.append((0.2, np.mean([item.at(times, smooth) for item in lights], axis=0)))
    if not terms:
        return np.full(len(times), 0.5)
    total = sum(weight for weight, _ in terms)
    return as_array(sum(weight * value for weight, value in terms) / total)
