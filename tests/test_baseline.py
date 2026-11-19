"""Tests of the rolling baseline: what an entity usually does."""

import numpy as np

from entropy_fm.baseline import (
    EntityBaseline,
    build_baseline,
    build_baselines,
    episodes,
    has_data,
)
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series

HOUR = 3600.0


def door(entity_id: str, opens: list[float]) -> Series:
    """Return a door that is opened for a minute at each of the given seconds."""
    times: list[float] = []
    values: list[float] = []
    for start in opens:
        times += [start, start + 60.0]
        values += [1.0, 0.0]
    info = EntityInfo(entity_id, entity_id, "hall", VoiceKind.PLUCK, device_class="door")
    return build_series(info, times, values)


def washer(entity_id: str, starts: list[float]) -> Series:
    """Return a power sensor that draws 2 kW for an hour at each start."""
    times: list[float] = [0.0]
    values: list[float] = [2.0]
    for start in starts:
        times += [start, start + HOUR]
        values += [2000.0, 2.0]
    return build_series(EntityInfo(entity_id, entity_id, "k", VoiceKind.DRUM), times, values)


def test_episodes_of_a_binary_sensor_and_a_power_sensor() -> None:
    """Each time it is on is an episode; so is each run of a machine, but not a flat line."""
    assert episodes(door("a", [3 * HOUR, 8 * HOUR])) == [
        (3 * HOUR, 3 * HOUR + 60.0),
        (8 * HOUR, 8 * HOUR + 60.0),
    ]
    found = episodes(washer("w", [10 * HOUR]))
    assert len(found) == 1
    assert found[0][0] == 10 * HOUR
    flat = build_series(EntityInfo("f", "f", "k", VoiceKind.DRUM), [0.0, 5000.0], [90.0, 91.0])
    assert episodes(flat) == []


def test_a_baseline_knows_the_usual_hours_and_trusts_itself_with_enough_days() -> None:
    """A door opened at 08:00 every day is familiar at 08:00 and strange at 03:00."""
    days = [door("d", [8 * HOUR + 120 * n]) for n in range(7)]
    baseline = build_baseline("d", days)
    assert baseline.days == 7
    assert baseline.active_days == 1.0
    assert baseline.rate == 1.0
    assert baseline.evidence == 1.0
    assert baseline.familiarity(8.5) > 0.9
    assert baseline.familiarity(3.5) == 0.0
    assert baseline.familiarity(7.0) > 0.0
    # with fewer days it trusts itself less
    assert build_baseline("d", days[:2]).evidence == 0.5


def test_the_hour_wraps_around_midnight() -> None:
    """Midnight is next to 23:00."""
    days = [door("d", [23 * HOUR + 1800.0])] * 5
    baseline = build_baseline("d", days)
    assert baseline.familiarity(23.9) > 0.0
    assert baseline.familiarity(0.1) > 0.0
    assert baseline.familiarity(12.0) == 0.0


def test_a_baseline_survives_the_cache() -> None:
    """It is stored as plain data and read back the same."""
    baseline = build_baseline("d", [door("d", [8 * HOUR]), door("d", [9 * HOUR])])
    data = baseline.to_json()
    assert isinstance(data["hours"], list)
    again = EntityBaseline.from_json(data)
    assert again.entity_id == "d"
    assert again.days == 2
    assert np.allclose(again.hours, baseline.hours, atol=1e-4)
    assert again.rate == baseline.rate


def test_baselines_skip_days_without_data() -> None:
    """A day the history has nothing for does not count as a quiet day."""
    empty = build_series(EntityInfo("d", "d", "hall", VoiceKind.PLUCK), [], [])
    assert not has_data([empty])
    built = build_baselines([[empty], [door("d", [HOUR])], [door("d", [2 * HOUR])]])
    assert built["d"].days == 2
    assert build_baselines([[empty]]) == {}
