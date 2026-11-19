"""Tests of the pure timeline code: classification, privacy and the control signals."""

import numpy as np
import pytest

from entropy_fm.timeline import (
    EntityInfo,
    Series,
    VoiceKind,
    build_series,
    is_excluded,
    normalise,
    parse_history,
    raw_to_json,
    select_voices,
    series_from_json,
    stable_hash,
    state_to_value,
    thin_numeric,
)

DAY_START = 1_700_000_000.0


def _info(entity_id: str, kind: VoiceKind, *, cumulative: bool = False) -> EntityInfo:
    return EntityInfo(entity_id, entity_id, "kitchen", kind, cumulative)


def test_stable_hash_does_not_depend_on_the_process() -> None:
    """The hash that seeds the music is a fixed number, unlike hash() of a string."""
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("b", "a")


@pytest.mark.parametrize(
    "entity_id",
    [
        "person.someone",
        "device_tracker.phone",
        "lock.front_door",
        "alarm_control_panel.home",
        "camera.garden",
        "sensor.wifi_password",
        "switch.door_code_enabled",
        "SENSOR.Wifi_PASSWORD",
    ],
)
def test_private_entities_are_excluded_by_default(entity_id: str) -> None:
    """Presence, locks, alarms, cameras and secrets never play."""
    assert is_excluded(entity_id)


def test_presence_toggle_only_lifts_presence() -> None:
    """Including presence lets persons through but not locks or secrets."""
    assert not is_excluded("person.someone", include_presence=True)
    assert not is_excluded("device_tracker.phone", include_presence=True)
    assert is_excluded("lock.front_door", include_presence=True)
    assert is_excluded("sensor.wifi_password", include_presence=True)


def test_extra_exclusion_globs() -> None:
    """The patterns of the user are matched on the entity id, ignoring case."""
    assert is_excluded("sensor.bedroom_temp", ["sensor.bedroom_*"])
    assert is_excluded("light.Hall_1", [" light.hall_* "])
    assert not is_excluded("light.kitchen", ["sensor.bedroom_*", ""])


def test_state_to_value() -> None:
    """States become numbers; unknown ones say nothing."""
    assert state_to_value("on") == 1.0
    assert state_to_value("off") == 0.0
    assert state_to_value("21.5") == 21.5
    assert state_to_value("unavailable") is None
    assert state_to_value("unknown") is None
    assert state_to_value("nan") is None


def test_parse_history_reads_minimal_responses() -> None:
    """Entries of the minimal format become seconds since the start of the day."""
    raw = {
        "light.a": [
            {"s": "off", "lu": DAY_START - 100},
            {"s": "on", "lu": DAY_START + 3600},
            {"s": "unavailable", "lu": DAY_START + 3700},
            {"s": "off", "lu": DAY_START + 7200},
        ],
        "sensor.t": [{"s": "20.5", "lu": DAY_START + 10}, {"s": "21", "lc": DAY_START + 99999}],
    }
    parsed = parse_history(raw, DAY_START, 86400)
    assert parsed["light.a"] == ([0.0, 3600.0, 7200.0], [0.0, 1.0, 0.0])
    assert parsed["sensor.t"] == ([10.0, 86400.0], [20.5, 21.0])


def test_binary_series_starts_off_when_the_first_change_is_late() -> None:
    """A light that is first seen turning on was off before."""
    series = build_series(_info("light.a", VoiceKind.CHORD), [3600.0, 7200.0], [1.0, 0.0])
    assert series.times[0] == 0.0
    assert series.norm.tolist() == [0.0, 1.0, 0.0]


def test_numeric_series_holds_its_first_value_backwards() -> None:
    """A temperature first seen at noon was already that at midnight."""
    series = build_series(_info("sensor.t", VoiceKind.TEMPERATURE), [43200.0], [21.0])
    assert series.times.tolist() == [0.0, 43200.0]
    assert series.values.tolist() == [21.0, 21.0]


def test_empty_history_is_a_silent_series() -> None:
    """An entity that never changed still makes a valid series."""
    series = build_series(_info("binary_sensor.m", VoiceKind.PLUCK), [], [])
    assert series.norm.tolist() == [0.0]
    assert series.onsets(0, 86400).size == 0


def test_energy_meter_becomes_a_rate() -> None:
    """A meter that counts up plays as how fast it counts; a reset is not a negative rate."""
    info = _info("sensor.e", VoiceKind.DRONE, cumulative=True)
    series = build_series(info, [0.0, 3600.0, 7200.0, 10800.0], [10.0, 11.0, 13.0, 0.5])
    assert series.values[0] == pytest.approx(1.0 / 3600)
    assert series.values[2] == pytest.approx(2.0 / 3600)
    assert series.values[-1] == 0.0


def test_power_is_normalised_against_its_peak() -> None:
    """The highest regular reading is full scale, nothing goes below zero."""
    norm = normalise(VoiceKind.DRONE, np.array([0.0, 40.0, 80.0, 80.0, -5.0] * 20))
    assert norm.max() == pytest.approx(1.0)
    assert norm.min() == 0.0
    assert norm[1] == pytest.approx(0.5)


def test_standby_power_is_silent() -> None:
    """A sensor that never leaves standby is normalised to zero, not blown up to full scale."""
    assert normalise(VoiceKind.DRONE, np.array([0.2, 0.3, 0.25])).max() == 0.0


def test_small_temperature_changes_stay_small() -> None:
    """A degree of drift moves the signal by a third, not across the whole range."""
    norm = normalise(VoiceKind.TEMPERATURE, np.array([20.0, 20.5, 21.0]))
    assert norm.max() - norm.min() == pytest.approx(1.0 / 3.0)
    wide = normalise(VoiceKind.TEMPERATURE, np.array([19.0, 23.0]))
    assert wide.tolist() == [0.0, 1.0]


def test_humidity_is_a_fraction() -> None:
    """Percent relative humidity maps to 0..1."""
    assert normalise(VoiceKind.HUMIDITY, np.array([0.0, 45.0, 100.0])).tolist() == [0.0, 0.45, 1.0]


def test_smoothing_is_an_exact_moving_average() -> None:
    """A step smoothed over 50 seconds is a ramp, whatever the sample grid."""
    series = build_series(_info("light.a", VoiceKind.CHORD), [100.0], [1.0])
    at = series.at(np.array([99.0, 100.0, 125.0, 150.0, 400.0]), smooth=50.0)
    assert at == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert series.at(np.array([99.0, 100.0]), smooth=0.0).tolist() == [0.0, 1.0]


def test_onsets_are_the_off_to_on_changes() -> None:
    """Only a rising edge counts, and only inside the asked stretch."""
    series = build_series(
        _info("binary_sensor.m", VoiceKind.PLUCK),
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
    )
    assert series.onsets(0, 100).tolist() == [10.0, 30.0, 50.0]
    assert series.onsets(15, 50).tolist() == [30.0]


def test_thinning_keeps_big_changes_and_the_ends() -> None:
    """A jittering sensor is thinned, a jump is not."""
    times = np.arange(0.0, 100.0, 1.0)
    values = np.where(times < 50, 20.0, 25.0) + 0.001 * (times % 2)
    kept_t, kept_v = thin_numeric(times, values)
    assert len(kept_t) < 20
    assert kept_t[0] == 0.0
    assert kept_t[-1] == 99.0
    assert 50.0 in kept_t
    assert kept_v.max() > 24.9


def test_series_survive_the_cache_round_trip() -> None:
    """What goes into the cache comes back as the same series."""
    info = _info("sensor.p", VoiceKind.DRONE)
    times = [0.0, 100.0, 200.0, 300.0]
    values = [0.0, 80.0, 0.0, 85.0]
    original = build_series(info, times, values)
    stored = raw_to_json(info, times, values)
    assert isinstance(stored["t"], list)
    again = series_from_json(info, stored)
    assert again.times.tolist() == original.times.tolist()
    assert again.norm.tolist() == original.norm.tolist()


def test_select_voices_keeps_the_most_active() -> None:
    """A big house is cut down to the busiest voices of each kind."""
    busy = build_series(
        _info("binary_sensor.busy", VoiceKind.PLUCK), [10.0, 20.0, 30.0, 40.0], [1.0, 0.0, 1.0, 0.0]
    )
    quiet = build_series(_info("binary_sensor.quiet", VoiceKind.PLUCK), [10.0], [1.0])
    chosen = select_voices([quiet, busy], {VoiceKind.PLUCK: 1})
    assert [item.entity_id for item in chosen] == ["binary_sensor.busy"]


def test_series_type() -> None:
    """build_series gives a Series."""
    assert isinstance(build_series(_info("light.a", VoiceKind.CHORD), [], []), Series)
