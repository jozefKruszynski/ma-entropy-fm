"""Tests of the day palette and the slow drift."""

from datetime import date, timedelta

import numpy as np

from entropy_fm.baseline import build_baseline
from entropy_fm.palette import (
    DayTraits,
    circle_position,
    derive_palette,
    measure_traits,
    pick_mode,
    season_warmth,
    week_index,
)
from entropy_fm.styles import AMBIENT, STYLES
from entropy_fm.synth import Scale
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series

from .test_baseline import HOUR, door

HOME = Scale(0, "dorian")
TUESDAY = date(2026, 3, 3)
SATURDAY = date(2026, 3, 7)


def test_weekend_is_slower_and_busier_than_a_weekday() -> None:
    """Same house, same weather: Saturday has another tempo and more going on."""
    traits = DayTraits(energy=0.4)
    week = derive_palette(AMBIENT, TUESDAY, HOME, traits)
    weekend = derive_palette(AMBIENT, SATURDAY, HOME, traits)
    assert not week.weekend
    assert weekend.weekend
    assert weekend.tempo_shift < week.tempo_shift
    assert weekend.density > week.density


def test_a_warm_day_gets_a_brighter_mode_than_a_cold_one() -> None:
    """The mode follows the temperature in standard deviations from the usual."""
    style = STYLES["berlin_school"]
    brightness = {"aeolian": -1, "dorian": 0, "mixolydian": 1, "lydian": 2, "major": 3}
    cold = derive_palette(style, TUESDAY, HOME, DayTraits(temp_z=-3.0, weather="snowy"))
    hot = derive_palette(style, TUESDAY, HOME, DayTraits(temp_z=3.0, weather="sunny"))
    assert cold.mode in style.modes
    assert hot.mode in style.modes
    assert brightness.get(hot.mode, 0) >= brightness.get(cold.mode, 0)
    assert hot.mode != cold.mode


def test_the_mode_is_always_one_the_style_allows() -> None:
    """Whatever the day, a style never plays a mode it does not have."""
    for style in STYLES.values():
        for z in (-3.0, 0.0, 3.0):
            palette = derive_palette(style, TUESDAY, HOME, DayTraits(temp_z=z))
            assert palette.mode in style.modes
    assert pick_mode(["dorian", "major"], 2.0) in {"dorian", "major"}


def test_the_key_walks_the_circle_of_fifths_one_step_a_week() -> None:
    """Within a week the key is fixed; the next week it moves a fifth; it comes back in 12 weeks."""
    monday = date(2026, 3, 2)
    week = [circle_position(monday + timedelta(days=n)) for n in range(7)]
    assert len(set(week)) == 1
    steps = [
        circle_position(monday + timedelta(weeks=n + 1))
        - circle_position(monday + timedelta(weeks=n))
        for n in range(12)
    ]
    assert all(abs(step) == 1 for step in steps)
    assert circle_position(monday + timedelta(weeks=12)) == circle_position(monday)
    positions = {circle_position(monday + timedelta(weeks=n)) for n in range(12)}
    assert positions == set(range(-3, 4))
    assert week_index(monday + timedelta(weeks=1)) == week_index(monday) + 1


def test_the_root_follows_the_walk_from_the_home_key() -> None:
    """The root is the home key moved by seven semitones per step, wrapped to the octave."""
    for offset in range(0, 70, 7):
        day = TUESDAY + timedelta(days=offset)
        palette = derive_palette(AMBIENT, day, Scale(2, "dorian"), DayTraits())
        assert palette.fifths == circle_position(day)
        assert palette.root == (2 + 7 * palette.fifths) % 12


def test_the_season_shifts_the_palette() -> None:
    """Midsummer is warm and bright, midwinter is the opposite."""
    summer, winter = date(2026, 7, 20), date(2026, 1, 20)
    assert season_warmth(summer) > 0.8
    assert season_warmth(winter) < -0.8
    bright = derive_palette(AMBIENT, summer, HOME, DayTraits())
    dark = derive_palette(AMBIENT, winter, HOME, DayTraits())
    assert bright.filter_scale > dark.filter_scale
    assert bright.tempo_shift > dark.tempo_shift


def test_density_follows_the_energy_of_the_house() -> None:
    """A busy day is denser than a quiet one, in the scale that goes to the arrangement."""
    quiet = derive_palette(AMBIENT, TUESDAY, HOME, DayTraits(energy=0.05))
    busy = derive_palette(AMBIENT, TUESDAY, HOME, DayTraits(energy=0.9))
    assert busy.density_scale > quiet.density_scale
    assert 0.6 <= quiet.density_scale <= busy.density_scale <= 1.4


def test_melodic_movement_follows_the_hours_someone_was_home() -> None:
    """Few hours of motion give still melodies, many hours give moving ones."""
    away = derive_palette(AMBIENT, TUESDAY, HOME, DayTraits(home_hours=2.0))
    home = derive_palette(AMBIENT, TUESDAY, HOME, DayTraits(home_hours=16.0))
    assert home.melodic > away.melodic
    assert home.melodic == 1.0


def _motion(entity_id: str, device_class: str, hours: list[int]) -> Series:
    times: list[float] = []
    values: list[float] = []
    for hour in hours:
        times += [hour * HOUR, hour * HOUR + 1800.0]
        values += [1.0, 0.0]
    info = EntityInfo(entity_id, entity_id, "a", VoiceKind.PLUCK, device_class=device_class)
    return build_series(info, times, values)


def test_hours_home_come_from_motion_only() -> None:
    """The hours counted are those with motion; a door does not say anyone is home."""
    motion = _motion("binary_sensor.m", "motion", [7, 8, 18])
    traits = measure_traits([motion], {})
    assert traits.home_hours == 3.0
    only_door = measure_traits([door("binary_sensor.d", [7 * HOUR, 9 * HOUR])], {})
    assert only_door.home_hours is None


def test_a_presence_entity_does_not_count_as_hours_home() -> None:
    """A person or tracker is not motion, even when it is passed along."""
    person = build_series(
        EntityInfo("person.jozef", "jozef", "a", VoiceKind.PLUCK),
        [0.0, 8 * HOUR],
        [1.0, 0.0],
    )
    assert measure_traits([person], {}).home_hours is None


def test_the_temperature_z_score_uses_the_baseline() -> None:
    """A day 6 degrees above the usual is several standard deviations."""
    info = EntityInfo("sensor.t", "t", "a", VoiceKind.TEMPERATURE)
    cool = [build_series(info, [0.0, 43200.0], [20.0, 21.0 + n % 2]) for n in range(7)]
    baseline = build_baseline("sensor.t", cool)
    hot = build_series(info, [0.0, 43200.0], [26.0, 28.0])
    traits = measure_traits([hot], {"sensor.t": baseline})
    assert traits.temp_z is not None
    assert traits.temp_z > 3.0
    assert measure_traits([hot], {}).temp_z is None


def test_the_palette_is_deterministic() -> None:
    """The same inputs, the same day."""
    traits = DayTraits(temp_z=1.0, energy=0.5, home_hours=9.0)
    assert derive_palette(AMBIENT, TUESDAY, HOME, traits) == derive_palette(
        AMBIENT, TUESDAY, HOME, traits
    )
    assert np.isfinite(season_warmth(TUESDAY))
