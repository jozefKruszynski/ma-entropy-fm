"""Tests of the surprise: routine fades, the unusual comes forward."""

from datetime import date
from itertools import pairwise

import numpy as np
import pytest

from entropy_fm.baseline import build_baseline
from entropy_fm.day_analysis import analyze_day
from entropy_fm.palette import DayTraits, derive_palette
from entropy_fm.scape import scape_score
from entropy_fm.styles import AMBIENT
from entropy_fm.surprise import (
    BOOST,
    FEATURE_THRESHOLD,
    ROUTINE_FADE,
    Surprise,
    climate_anomaly,
    find_anomalies,
)
from entropy_fm.synth import Scale
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series

from .test_baseline import HOUR, door, washer


def _usual_door() -> list[Series]:
    return [door("binary_sensor.door", [8 * HOUR]) for _ in range(7)]


def test_a_door_at_the_usual_hour_is_routine_and_at_three_in_the_night_is_featured() -> None:
    """The same door: at 08:00 it fades into the texture, at 03:00 it is the event of the day."""
    baseline = build_baseline("binary_sensor.door", _usual_door())
    routine = door("binary_sensor.door", [8 * HOUR])
    night = door("binary_sensor.door", [3 * HOUR])
    assert find_anomalies([routine], {baseline.entity_id: baseline}) == []
    found = find_anomalies([night], {baseline.entity_id: baseline})
    assert len(found) == 1
    assert found[0].start == 3 * HOUR
    assert found[0].score >= FEATURE_THRESHOLD

    quiet = Surprise([routine], {baseline.entity_id: baseline})
    loud = Surprise([night], {baseline.entity_id: baseline})
    assert float(quiet.gain("binary_sensor.door", 8.2 * HOUR)) == pytest.approx(
        1.0 - ROUTINE_FADE, abs=0.12
    )
    assert float(loud.gain("binary_sensor.door", 3 * HOUR)) > 1.0 + 0.8 * BOOST
    # away from both, an entity that has a baseline is at full level
    assert float(quiet.gain("binary_sensor.door", 15 * HOUR)) == 1.0


def test_a_new_entity_without_history_has_no_surprise() -> None:
    """Nothing is known about it: it plays as it is."""
    surprise = Surprise([door("a", [3 * HOUR])], {})
    assert surprise.anomalies == []
    assert np.array_equal(surprise.gain("a", np.array([0.0, 5000.0])), np.ones(2))
    assert surprise.event_gain("a", 3 * HOUR) == 1.0


def test_one_day_of_history_is_not_enough_to_call_anything_unusual() -> None:
    """With one day the baseline has little evidence, so nothing reaches the threshold."""
    baseline = build_baseline("binary_sensor.door", [door("binary_sensor.door", [8 * HOUR])])
    assert (
        find_anomalies([door("binary_sensor.door", [3 * HOUR])], {baseline.entity_id: baseline})
        == []
    )


def test_an_unusual_washing_machine_day_is_featured() -> None:
    """A machine that never ran in the evening runs at 22:00."""
    usual = [washer("sensor.washer", [9 * HOUR]) for _ in range(6)]
    baseline = build_baseline("sensor.washer", usual)
    today = washer("sensor.washer", [22 * HOUR])
    found = find_anomalies([today], {"sensor.washer": baseline})
    assert [a.start for a in found] == [22 * HOUR]
    assert found[0].end > found[0].start


def test_a_heatwave_is_a_climate_anomaly() -> None:
    """A day mean far above the usual is featured; a normal day is not."""
    info = EntityInfo("sensor.t", "t", "a", VoiceKind.TEMPERATURE)
    cool = [build_series(info, [0.0, 43200.0], [20.0 + n % 2, 21.0]) for n in range(7)]
    baseline = build_baseline("sensor.t", cool)
    hot = build_series(info, [0.0, 43200.0], [30.0, 36.0])
    normal = build_series(info, [0.0, 43200.0], [20.0, 21.5])
    assert climate_anomaly(normal, baseline) is None
    anomaly = climate_anomaly(hot, baseline)
    assert anomaly is not None
    assert anomaly.kind == "climate"
    assert anomaly.score > 0.5
    assert 0.0 <= anomaly.start < 86400.0


def test_at_most_a_few_are_featured_and_not_on_top_of_each_other() -> None:
    """Featured moments are spaced out and limited, strongest first."""
    baseline = build_baseline("binary_sensor.door", _usual_door())
    crowded = door("binary_sensor.door", [3 * HOUR + 60 * n for n in range(5)])
    assert len(find_anomalies([crowded], {baseline.entity_id: baseline})) == 1
    spread = door("binary_sensor.door", [(1 + 2 * n) * HOUR for n in range(8)])
    found = find_anomalies([spread], {baseline.entity_id: baseline})
    assert len(found) <= 4
    assert all(b.start - a.start >= 1800.0 for a, b in pairwise(found))


def test_featured_moments_are_placed_in_audio_time() -> None:
    """The anomaly of 03:00 is heard at the right second of a compressed track."""
    baseline = build_baseline("binary_sensor.door", _usual_door())
    surprise = Surprise([door("binary_sensor.door", [3 * HOUR])], {baseline.entity_id: baseline})
    inside = surprise.featured(origin=2 * HOUR, compression=60.0, duration_s=120)
    assert len(inside) == 1
    assert inside[0][0] == pytest.approx(60.0)
    assert inside[0][2] == "binary_sensor.door"
    assert surprise.featured(origin=4 * HOUR, compression=60.0, duration_s=120) == []


def test_surprise_is_deterministic() -> None:
    """The same history gives the same anomalies and the same curves."""
    baseline = build_baseline("binary_sensor.door", _usual_door())
    first = Surprise([door("binary_sensor.door", [3 * HOUR])], {baseline.entity_id: baseline})
    second = Surprise([door("binary_sensor.door", [3 * HOUR])], {baseline.entity_id: baseline})
    assert first.anomalies == second.anomalies
    times = np.linspace(0, 86000, 50)
    assert np.array_equal(
        first.gain("binary_sensor.door", times), second.gain("binary_sensor.door", times)
    )


def _score(**kwargs: object) -> object:
    voices = [
        door("binary_sensor.door", [3 * HOUR]),
        washer("sensor.washer", [3 * HOUR + 600]),
    ]
    analysis = analyze_day(voices)
    return scape_score(voices, analysis, "lofi", "s", Scale(2, "dorian"), **kwargs)[0]  # type: ignore[arg-type]


def test_a_soundscape_with_surprise_and_a_palette_is_deterministic_and_differs() -> None:
    """The day character is deterministic, and it changes the score that is made."""
    door_base = build_baseline("binary_sensor.door", _usual_door())
    washer_base = build_baseline("sensor.washer", [washer("sensor.washer", [9 * HOUR])] * 6)
    voices = [door("binary_sensor.door", [3 * HOUR]), washer("sensor.washer", [3 * HOUR + 600])]
    surprise = Surprise(voices, {"binary_sensor.door": door_base, "sensor.washer": washer_base})
    palette = derive_palette(AMBIENT, date(2026, 3, 7), Scale(2, "dorian"), DayTraits(energy=0.7))
    anomalies = [(a.start, a.score, a.entity_id) for a in surprise.anomalies]
    analysis = analyze_day(voices, 86400.0, anomalies)
    plain = scape_score(voices, analyze_day(voices), "lofi", "s", Scale(2, "dorian"))[0]
    first = scape_score(voices, analysis, "lofi", "s", Scale(2, "dorian"), palette=palette)[0]
    again = scape_score(voices, analysis, "lofi", "s", Scale(2, "dorian"), palette=palette)[0]
    assert first.notes == again.notes
    assert first.events
    assert not plain.events
    assert [(n.t, n.midi) for n in first.notes] != [(n.t, n.midi) for n in plain.notes]
