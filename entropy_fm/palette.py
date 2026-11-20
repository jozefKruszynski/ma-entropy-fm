"""
The day picks its own palette, and the palette drifts slowly with the weeks and the seasons.

Deterministic settings come from the character of the day:

| Day trait | Musical parameter |
|---|---|
| weekday or weekend | tempo and busyness |
| mean temperature (against the usual) or the weather | key and mode: cold days dark, warm bright |
| total energy use | arrangement density |
| hours someone was home | how much the melody moves |

On top of that the key walks the circle of fifths from week to week and the season brightens or
darkens the sound, so two Tuesdays a month apart sound related but not identical. Hours at home
come only from aggregated motion and occupancy sensors; persons and device trackers stay out.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .baseline import DAY_SECONDS, episodes
from .energy import activity
from .synth import Scale
from .timeline import PRESENCE_DOMAINS, VoiceKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    from .baseline import EntityBaseline
    from .styles import Style
    from .timeline import Series

# how bright a mode is: the dark ones for cold days, the bright ones for warm days
MODE_WARMTH = {
    "phrygian": -1.0,
    "aeolian": -0.9,
    "minor_pentatonic": -0.6,
    "dorian": -0.5,
    "mixolydian": 0.3,
    "major_pentatonic": 0.5,
    "major": 0.6,
    "lydian": 1.0,
}
# the weather states of Home Assistant, from cold and grey to bright and warm
WEATHER_WARMTH = {
    "clear-night": 0.2,
    "cloudy": -0.3,
    "exceptional": -0.4,
    "fog": -0.5,
    "hail": -0.7,
    "lightning": -0.3,
    "lightning-rainy": -0.5,
    "partlycloudy": 0.3,
    "pouring": -0.7,
    "rainy": -0.6,
    "snowy": -0.8,
    "snowy-rainy": -0.8,
    "sunny": 1.0,
    "windy": 0.0,
    "windy-variant": 0.0,
}
# the key moves this far from home on the circle of fifths, and back, over twelve weeks
FIFTHS_REACH = 3
WALK_WEEKS = 2 * 2 * FIFTHS_REACH
# how much the room of the day weighs in the mode, against the mode the user chose
HOME_MODE_WEIGHT = 0.25
# how much the season counts in the warmth of a day, against its temperature (1) and its weather (0.8)
SEASON_WEIGHT = 0.5
WEEKEND_TEMPO = -0.15
WEEKEND_BUSY = 0.15
SEASON_TEMPO = 0.05
SEASON_FILTER = 0.12
HOME_HOURS_FOR_FULL_MOVEMENT = 14.0
MOTION_CLASSES = frozenset({"motion", "occupancy", "moving"})


@dataclass(frozen=True)
class DayTraits:
    """What a day was like, as far as it is known; anything unknown is None."""

    # the day mean of the temperature in standard deviations from the usual
    temp_z: float | None = None
    # a state of a Home Assistant weather entity, such as "rainy"
    weather: str | None = None
    # the mean energy of the house on the day, 0 to 1
    energy: float | None = None
    # the number of hours in which there was motion
    home_hours: float | None = None


@dataclass(frozen=True)
class DayPalette:
    """The settings the day derives."""

    root: int
    mode: str
    # added to the position in the tempo range of the style
    tempo_shift: float
    # 0 to 1: how full the arrangement is
    density: float
    # 0 to 1: how much the melodies move
    melodic: float
    # -1 (midwinter) to 1 (midsummer)
    season: float
    # the position of the key on the circle of fifths, relative to home
    fifths: int
    weekend: bool

    @property
    def density_scale(self) -> float:
        """Return what the density of the style is multiplied by."""
        return 0.6 + 0.8 * self.density

    @property
    def filter_scale(self) -> float:
        """Return what the cutoff of the filters is multiplied by: brighter in summer."""
        return 1.0 + SEASON_FILTER * self.season


def week_index(day: date) -> int:
    """
    Return the number of the week, counted in ISO weeks from a fixed start.

    :param day: The day.
    """
    return (day.toordinal() - 1) // 7


def circle_position(day: date) -> int:
    """
    Return how many fifths from home the key of a week is: it walks out and back, a step a week.

    Over twelve weeks the key goes three fifths flat of home, passes home, goes three fifths sharp
    and returns.

    :param day: The day.
    """
    phase = week_index(day) % WALK_WEEKS
    triangle = phase if phase <= WALK_WEEKS // 2 else WALK_WEEKS - phase
    return triangle - FIFTHS_REACH


def season_warmth(day: date) -> float:
    """
    Return -1 in midwinter and 1 in midsummer, on the northern hemisphere.

    :param day: The day.
    """
    return -math.cos(2.0 * math.pi * (day.timetuple().tm_yday + 10) / 365.0)


def day_warmth(traits: DayTraits, season: float, season_weight: float = SEASON_WEIGHT) -> float:
    """
    Return how warm and bright the day is, -1 to 1, from what is known of it.

    :param traits: What the day was like.
    :param season: The warmth of the season.
    :param season_weight: How much the season counts against what the day measured; 0 measures the day against
        its season only, and a day with nothing measured is then mild.
    """
    parts: list[tuple[float, float]] = [(season_weight, season)] if season_weight > 0 else []
    if traits.temp_z is not None:
        parts.append((1.0, float(np.clip(traits.temp_z / 2.0, -1.0, 1.0))))
    if traits.weather is not None and traits.weather in WEATHER_WARMTH:
        parts.append((0.8, WEATHER_WARMTH[traits.weather]))
    if not parts:
        return 0.0
    return float(sum(w * v for w, v in parts) / sum(w for w, _ in parts))


def pick_mode(modes: Sequence[str], target: float) -> str:
    """
    Return the mode of a style that is closest in brightness to a target.

    :param modes: The modes the style allows.
    :param target: A brightness, -1 to 1.
    """
    return min(modes, key=lambda mode: (abs(MODE_WARMTH.get(mode, 0.0) - target), mode))


def derive_palette(style: Style, day: date, home: Scale, traits: DayTraits) -> DayPalette:
    """
    Derive the palette of a day.

    :param style: The style of the track, which limits the modes.
    :param day: The day.
    :param home: The key and mode the user chose, which the day moves away from.
    :param traits: What the day was like.
    """
    season = season_warmth(day)
    fifths = circle_position(day)
    root = (home.root + 7 * fifths) % 12
    warmth = day_warmth(traits, season)
    anchor = MODE_WARMTH.get(home.mode, -0.5)
    target = float(np.clip(HOME_MODE_WEIGHT * anchor + warmth, -1.0, 1.0))
    mode = pick_mode(style.modes, target)
    weekend = day.weekday() >= 5
    energy = 0.5 if traits.energy is None else float(np.clip(traits.energy, 0.0, 1.0))
    busy = energy + (WEEKEND_BUSY if weekend else 0.0)
    tempo = (WEEKEND_TEMPO if weekend else 0.0) + SEASON_TEMPO * season
    home_hours = 8.0 if traits.home_hours is None else traits.home_hours
    return DayPalette(
        root=root,
        mode=mode,
        tempo_shift=tempo,
        density=float(np.clip(busy, 0.0, 1.0)),
        melodic=float(np.clip(home_hours / HOME_HOURS_FOR_FULL_MOVEMENT, 0.0, 1.0)),
        season=season,
        fifths=fifths,
        weekend=weekend,
    )


# a room whose CO2 is this far up its usual range has someone in it
AIR_HOME_LEVEL = 0.3


def measure_traits(
    series: Sequence[Series],
    baselines: Mapping[str, EntityBaseline],
    covered: float = DAY_SECONDS,
    weather: str | None = None,
) -> DayTraits:
    """
    Measure what a day was like from its series.

    :param series: Every voice of the area or the house on the day.
    :param baselines: What each entity usually does.
    :param covered: How many seconds of the day have happened.
    :param weather: The state of a weather entity, when there is one.
    """
    z_values = []
    for item in series:
        baseline = baselines.get(item.entity_id)
        if item.kind.value == "temperature" and baseline is not None and baseline.days >= 3:
            spread = max(baseline.spread, 0.6)
            z_values.append((float(np.mean(item.values)) - baseline.mean) / spread)
    grid = np.arange(0.0, max(covered, 900.0), 900.0) + 450.0
    energy = float(np.mean(activity(series, grid))) if series else None
    hours: set[int] = set()
    for item in series:
        domain = item.entity_id.partition(".")[0]
        if domain in PRESENCE_DOMAINS:
            continue
        if item.is_motion and item.device_class in MOTION_CLASSES | {""}:
            for start, end in episodes(item, covered):
                hours.update(range(int(start // 3600), int(min(end, covered) // 3600) + 1))
    has_motion = any(
        item.is_motion and item.entity_id.partition(".")[0] not in PRESENCE_DOMAINS
        for item in series
    )
    home_hours = float(len(hours)) if has_motion else None
    air = [item for item in series if item.kind == VoiceKind.AIR]
    if home_hours is None and air:
        # without motion sensors the CO2 of the rooms says when someone was about: the hours it is up
        steps = (np.arange(int(min(covered, DAY_SECONDS) // 3600)) + 0.5) * 3600.0
        level = np.mean([item.at(steps, 1800.0) for item in air], axis=0)
        home_hours = float(np.count_nonzero(level > AIR_HOME_LEVEL))
    return DayTraits(
        temp_z=float(np.mean(z_values)) if z_values else None,
        weather=weather,
        energy=energy,
        home_hours=home_hours,
    )
