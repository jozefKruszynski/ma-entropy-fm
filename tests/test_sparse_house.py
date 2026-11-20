"""Tests of a house without motion sensors: CO2 as presence, showers, dead sensors, per-room scales, a full base."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from entropy_fm.baseline import EntityBaseline, build_baselines
from entropy_fm.day_analysis import analyze_day
from entropy_fm.energy import activity, occupancy
from entropy_fm.plan import MAX_EVENTS, MAX_SHOWERS, build_plan
from entropy_fm.quality import STANDARD
from entropy_fm.references import MIN_REFERENCE_DAYS, is_dead, rescale, showers
from entropy_fm.roles import classify
from entropy_fm.styles import STYLES
from entropy_fm.synth import Scale
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series

from .scape_tools import layers_playing, the_demo, the_sparse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from demo_render import Demo

HOUR = 3600.0
DAY = 86400.0


def _series(  # noqa: PLR0913, PLR0917
    entity_id: str,
    kind: VoiceKind,
    times: list[float],
    values: list[float],
    device_class: str = "",
    why: str = "",
) -> Series:
    return build_series(
        EntityInfo(entity_id, entity_id, "room", kind, device_class=device_class, why=why),
        times,
        values,
    )


def _co2(peak_hour: float, peak: float = 1500.0, base: float = 420.0) -> Series:
    """Return a CO2 monitor that rises to a peak at an hour of the day and falls back."""
    hours = np.arange(0.0, 24.0, 0.25)
    values = base + (peak - base) * np.exp(-(((hours - peak_hour) / 1.5) ** 2))
    return _series("sensor.co2", VoiceKind.AIR, (hours * HOUR).tolist(), values.tolist())


def _shower(hour: float, peak: float, *, other: float = 40.0) -> Series:
    """Return a bathroom humidity sensor with a shower."""
    times = np.arange(0.0, DAY, 120.0)
    since = times - hour * HOUR
    bump = np.where(since > 0, (peak - other) * np.clip(since / 420.0, 0, 1), 0.0)
    bump = bump * np.exp(-np.clip(since - 420.0, 0, None) / 1500.0)
    return _series(
        "sensor.bath_humidity", VoiceKind.HUMIDITY, times.tolist(), (other + bump).tolist()
    )


def _door(hours: list[float]) -> Series:
    times: list[float] = []
    values: list[float] = []
    for hour in hours:
        times += [hour * HOUR, hour * HOUR + 60.0]
        values += [1.0, 0.0]
    return _series("binary_sensor.motion", VoiceKind.PLUCK, times, values, "motion")


# ---- roles


def test_co2_monitors_are_air_by_device_class_and_by_name() -> None:
    """A carbon dioxide sensor is the presence of a room, however it is named; the user can label it too."""
    by_class = classify(
        "sensor.living_room_air_quality_monitor_carbon_dioxide", "x", "carbon_dioxide", "ppm", "700"
    )
    by_name = classify("sensor.bedroom_co2", "x", None, "ppm", "700")
    assert by_class is not None
    assert by_name is not None
    assert by_class.kind == by_name.kind == VoiceKind.AIR
    assert by_class.description == "role: co2 (device class)"
    voc = classify("sensor.voc", "x", "volatile_organic_compounds", "ppb", "90")
    assert voc is not None
    assert voc.kind == VoiceKind.PAD_PARTIAL


def test_the_role_of_an_entity_is_kept_on_its_series() -> None:
    """The role in the description of an entity is read back, so that a fridge can be told from a TV."""
    fridge = _series(
        "sensor.f", VoiceKind.DRONE, [0.0, 10.0], [5.0, 60.0], why="role: fridge (name)"
    )
    tv = _series(
        "sensor.tv", VoiceKind.DRONE, [0.0, 10.0], [5.0, 60.0], why="role: drone (device class)"
    )
    assert fridge.is_baseline_load
    assert not tv.is_baseline_load


# ---- showers


def test_a_humidity_spike_well_over_the_usual_is_a_shower() -> None:
    """A rise of 30 points is found once, at the start of the climb; a flat day and a damp day are not showers."""
    found = showers(_shower(7.0, 78.0))
    assert len(found) == 1
    assert found[0][0] == pytest.approx(7 * HOUR, abs=300.0)
    assert 0.3 <= found[0][1] <= 1.0
    assert showers(_shower(7.0, 40.0)) == []
    damp = _series("sensor.h", VoiceKind.HUMIDITY, [0.0, 40000.0], [45.0, 55.0])
    assert showers(damp) == []


def test_a_shower_is_measured_against_the_usual_level_of_the_room() -> None:
    """Against the days before, a spike is a shower and a day that is damp all day (the weather) is not."""
    days = [[_shower(7.0, 70.0, other=44.0)] for _ in range(5)]
    baseline = build_baselines(days)["sensor.bath_humidity"]
    found = rescale([_shower(7.0, 72.0, other=44.0)], {"sensor.bath_humidity": baseline})[0][0]
    assert found.ref is not None
    assert len(showers(found)) == 1
    damp = _series(
        "sensor.bath_humidity", VoiceKind.HUMIDITY, [0.0, 600.0, 90000.0], [60.0, 75.0, 75.0]
    )
    measured = rescale([damp], {"sensor.bath_humidity": baseline})[0][0]
    assert showers(measured) == []


def test_showers_become_texture_moments_at_most_a_couple_a_piece() -> None:
    """Three showers in one day give two moments, and with the anomalies never more than four events."""
    times = np.arange(0.0, DAY, 120.0)
    values = np.full(len(times), 40.0)
    for hour in (6.5, 12.5, 19.0):
        since = times - hour * HOUR
        values = values + np.where(
            (since > 0) & (since < 1800), 38.0 * np.clip(since / 420.0, 0, 1), 0.0
        )
    series = [_series("sensor.h", VoiceKind.HUMIDITY, times.tolist(), values.tolist())]
    analysis = analyze_day(
        series, DAY, [(3 * HOUR, 0.9, "a"), (9 * HOUR, 0.8, "b"), (22 * HOUR, 0.7, "c")]
    )
    assert len(analysis.moments_of("shower")) == 3
    plan = build_plan(series, analysis, "ambient", "x", Scale(2, "dorian"), STANDARD)
    assert 1 <= len(plan.events) <= MAX_EVENTS
    quiet = analyze_day(series, DAY)
    plan = build_plan(series, quiet, "ambient", "x", Scale(2, "dorian"), STANDARD)
    assert 1 <= len(plan.events) <= MAX_SHOWERS


# ---- dead sensors


def test_dead_and_constant_sensors_are_left_out() -> None:
    """A stuck temperature, a sensor that said nothing and a lamp that never changed play nothing."""
    stuck = _series("sensor.t", VoiceKind.TEMPERATURE, [0.0], [19.5])
    silent = build_series(EntityInfo("sensor.plant", "p", "r", VoiceKind.PAD_PARTIAL), [], [])
    lamp = _series("light.always", VoiceKind.CHORD, [0.0], [1.0])
    alive = _co2(19.0)
    baselines = build_baselines([[stuck, silent, lamp, alive]] * 5)
    kept, dropped = rescale([stuck, silent, lamp, alive], baselines)
    assert [item.entity_id for item in kept] == ["sensor.co2"]
    assert set(dropped) == {"sensor.t", "sensor.plant", "light.always"}


def test_a_quiet_day_of_a_sensor_that_moves_is_not_dead() -> None:
    """A TV that stayed off today but was on the days before, and a door that stayed shut, still count."""
    tv_on = _series("sensor.tv", VoiceKind.DRONE, [0.0, 70000.0, 80000.0], [0.4, 150.0, 0.4])
    tv_off = _series("sensor.tv", VoiceKind.DRONE, [0.0], [0.4])
    door = _door([8.0])
    baselines = build_baselines([[tv_on, door]] * 5)
    assert not is_dead(tv_off, baselines["sensor.tv"].reference)
    assert not is_dead(_door([]), baselines["binary_sensor.motion"].reference)
    # without any days before, a constant number is dead and a door is not
    assert is_dead(tv_off, None)
    assert not is_dead(_door([]), None)


def test_a_day_of_all_dead_sensors_keeps_them() -> None:
    """Nothing is dropped when nothing would be left."""
    stuck = _series("sensor.t", VoiceKind.TEMPERATURE, [0.0], [19.5])
    kept, dropped = rescale([stuck], {})
    assert kept == [stuck]
    assert dropped == []


def test_a_baseline_remembers_the_scale_and_old_data_still_loads() -> None:
    """The scale of an entity survives the cache; a baseline without it (an old cache) reads as no scale."""
    days = [[_co2(19.0, 1500.0 + 100.0 * k)] for k in range(5)]
    baseline = build_baselines(days)["sensor.co2"]
    again = EntityBaseline.from_json(baseline.to_json())
    assert again.high == pytest.approx(baseline.high, abs=1e-3)
    assert 380.0 < again.low < 450.0
    assert again.high > again.mid >= again.low
    assert not again.flat
    legacy = {
        k: v
        for k, v in baseline.to_json().items()
        if k not in {"low", "mid", "high", "peak", "flat"}
    }
    assert EntityBaseline.from_json(legacy).high == 0.0


# ---- presence: motion first, the new sources alongside


def test_co2_is_the_presence_of_a_house_without_motion() -> None:
    """The level of the CO2 and a rise in it say someone is there; without any other sensor it moves activity."""
    series = [_co2(19.0)]
    baselines = build_baselines([series] * 5)
    measured, _ = rescale(series, baselines)
    grid = np.array([4.0, 11.0, 19.0]) * HOUR
    level = activity(measured, grid)
    assert level[2] > level[1] + 0.3
    assert level[2] > level[0] + 0.3
    entering = occupancy(measured, np.array([17.5, 19.0, 23.0]) * HOUR)
    assert entering[0] > 0.3  # still climbing: someone has just come in


def test_a_sleeper_is_present_and_quiet() -> None:
    """CO2 that peaks at six in the morning is somebody asleep: it counts for less than the same peak in the evening."""
    asleep = occupancy(
        rescale([_co2(5.0)], build_baselines([[_co2(5.0)]] * 5))[0], np.array([5.0 * HOUR])
    )
    awake = occupancy(
        rescale([_co2(19.0)], build_baselines([[_co2(19.0)]] * 5))[0], np.array([19.0 * HOUR])
    )
    assert 0.1 < asleep[0] < 0.6 * awake[0]


def test_motion_stays_the_primary_signal_and_the_new_sources_add() -> None:
    """With motion and CO2 both, each shows in the activity; motion alone is what it always was; none is pinned."""
    motion = _door([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    air = _co2(19.0)
    baselines = build_baselines([[motion, air]] * 5)
    both, _ = rescale([motion, air], baselines)
    only_motion = [motion]
    grid = np.array([4.0, 10.4, 19.0]) * HOUR
    blended = activity(both, grid)
    plain = activity(only_motion, grid)
    assert blended[1] > blended[0] + 0.2  # the motion is heard
    assert blended[2] > blended[0] + 0.2  # and so is the room filling with CO2
    assert plain[1] > plain[0] + 0.2
    assert plain[2] == pytest.approx(plain[0], abs=0.1)  # without CO2 the evening says nothing
    assert np.all((blended >= 0.0) & (blended <= 1.0))
    # the CO2 term is the smaller of the two when motion is there
    air_only = activity(rescale([air], baselines)[0], grid)
    assert blended[1] > air_only[1] + 0.1


def test_plugs_and_lights_count_and_a_fridge_does_not() -> None:
    """A TV in the evening and lights that are switched about are activity; a compressor cycling all day is not."""
    tv = _series(
        "sensor.tv",
        VoiceKind.DRONE,
        [0.0, 19 * HOUR, 22 * HOUR],
        [0.4, 150.0, 0.4],
        why="role: drone (device class)",
    )
    fridge_times = np.arange(0.0, DAY, 2400.0)
    fridge = _series(
        "sensor.fridge",
        VoiceKind.DRONE,
        fridge_times.tolist(),
        [60.0 if k % 2 == 0 else 1.0 for k in range(len(fridge_times))],
        why="role: fridge (name)",
    )
    grid = np.array([4.0, 20.5]) * HOUR
    assert activity([fridge], grid)[0] == pytest.approx(
        0.5, abs=1e-9
    )  # nothing is left: the empty answer
    with_tv = activity([tv, fridge], grid)
    assert with_tv[1] > with_tv[0] + 0.25
    flickered = _series(
        "light.k",
        VoiceKind.CHORD,
        [0.0, 7 * HOUR, 7.2 * HOUR, 12 * HOUR, 12.3 * HOUR],
        [0.0, 1.0, 0.0, 1.0, 0.0],
    )
    lamp = _series("light.l", VoiceKind.CHORD, [0.0, 7 * HOUR], [0.0, 1.0])
    near = np.array([7.1, 12.1, 3.0]) * HOUR
    switched = activity([flickered], near)
    assert switched[0] > switched[2]
    assert switched[1] > switched[2]
    assert activity([lamp], np.array([20.0 * HOUR]))[0] <= 0.66


# ---- the scale of each room


def _rooms(demo_house: Demo) -> Iterator[tuple[str, list[Any], dict[str, Any]]]:
    index = 13
    for area in demo_house.days[index]:
        if area == "sun":
            continue
        series = demo_house.days[index][area] + demo_house.days[index]["sun"]
        baselines = demo_house.baselines(index, (area,))
        yield area, series, baselines


@pytest.mark.parametrize("house", ["motion", "sparse"])
def test_every_room_is_measured_against_itself_and_stays_in_range(house: str) -> None:
    """After the rescale every curve is between 0 and 1, and a room is neither pinned at the top nor flat."""
    demo_house = the_demo() if house == "motion" else the_sparse()
    seen = 0
    for area, series, baselines in _rooms(demo_house):
        measured, _ = rescale(series, baselines)
        for item in measured:
            assert np.all((item.norm >= 0.0) & (item.norm <= 1.0)), item.entity_id
        analysis = analyze_day(measured, DAY)
        assert analysis.measured
        plan = build_plan(measured, analysis, "ambient", area, Scale(2, "dorian"), STANDARD)
        assert float(plan.activity.min()) >= 0.0
        assert float(plan.activity.max()) <= 1.0
        assert 0.12 < float(plan.activity.mean()) < 0.8, area
        assert float(plan.activity.std()) > 0.05, area
        seen += 1
    assert seen >= 2


def test_a_quiet_room_has_highs_and_lows_and_a_busy_one_is_not_pinned() -> None:
    """Two CO2 monitors with a very different range both use their whole range, and a calm day is calmer."""

    def day(peak: float) -> Series:
        return _co2(19.0, peak, 410.0)

    for peak in (700.0, 3200.0):
        days = [[day(peak)] for _ in range(5)]
        baselines = build_baselines(days)
        measured = rescale(days[0], baselines)[0][0]
        assert measured.norm.max() > 0.85
        assert measured.norm.min() < 0.1
    baselines = build_baselines([[day(3200.0)] for _ in range(5)])
    calm = rescale([day(1200.0)], baselines)[0][0]
    assert (
        0.15 < calm.norm.max() < 0.5
    )  # a day of 1200 in a room that reaches 3200 is a calm day there
    assert MIN_REFERENCE_DAYS == 3


def test_a_calm_day_is_calmer_than_a_busy_one_in_the_same_room() -> None:
    """Measured against the days before, the same room has a quieter activity on a quiet day."""
    baselines = build_baselines([[_co2(19.0, 2400.0)] for _ in range(5)])
    busy = analyze_day(rescale([_co2(19.0, 2400.0)], baselines)[0], DAY)
    calm = analyze_day(rescale([_co2(19.0, 900.0)], baselines)[0], DAY)
    assert busy.curve.max() > 0.9
    assert calm.curve.max() < 0.6


# ---- a full base, whatever the day


def _calm_day() -> list[Series]:
    """Return a day of a room that has nothing happening: a fridge, a stuck thermometer and a lamp for an hour."""
    fridge = _series(
        "sensor.fridge",
        VoiceKind.DRONE,
        [0.0, 600.0, 1200.0, 5000.0],
        [60.0, 1.0, 60.0, 1.0],
        why="role: fridge (name)",
    )
    lamp = _series("light.l", VoiceKind.CHORD, [0.0, 19 * HOUR, 20 * HOUR], [0.0, 1.0, 0.0])
    sun = build_series(
        EntityInfo("sensor.sun_solar_elevation", "sun", None, VoiceKind.SUN),
        np.arange(0.0, DAY, 300.0).tolist(),
        (-14 + 36 * np.cos(2 * np.pi * (np.arange(0.0, DAY, 300.0) / HOUR - 11.75) / 24)).tolist(),
    )
    return [fridge, lamp, sun]


# the fewest layers that play in nine of ten minutes of a calm day: the base of each style
BASE_LAYERS = {"ambient": 3, "berlin_school": 5, "lofi": 3, "minimal": 4, "ritual": 3, "glitch": 3}


@pytest.mark.parametrize("style", list(STYLES))
def test_a_calm_day_still_plays_the_whole_base_of_the_style(style: str) -> None:
    """On a day with nothing in it every style keeps its base layers playing, minute after minute."""
    from datetime import date  # noqa: PLC0415

    from entropy_fm.scape import analysis_of, scape_score  # noqa: PLC0415

    series = _calm_day()
    analysis = analysis_of(series)
    score, _ = scape_score(
        series,
        analysis,
        style,
        f"calm|{style}",
        Scale(2, "dorian"),
        day=date(2026, 11, 17),
        area_key="calm",
    )
    counts = sorted(layers_playing(score))
    assert counts[int(len(counts) * 0.1)] >= BASE_LAYERS[style], (style, counts)


def test_the_rooms_of_the_sparse_house_differ() -> None:
    """The rooms of the sparse house have a plan of their own: their activity does not follow one curve."""
    demo_house = the_sparse()
    curves = {}
    for room in ("living", "bedroom", "office"):
        _, plan = demo_house.score(13, (room,), "ambient")
        activity_curve = np.asarray(plan.activity)  # type: ignore[attr-defined]
        curves[room] = np.interp(
            np.linspace(0, 1, 1000), np.linspace(0, 1, len(activity_curve)), activity_curve
        )
    assert float(np.abs(curves["living"] - curves["bedroom"]).mean()) > 0.1
    assert float(np.abs(curves["office"] - curves["bedroom"]).mean()) > 0.1
