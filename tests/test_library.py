"""Tests of the mapping from Home Assistant onto albums and tracks."""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from entropy_fm.library import (
    album_id,
    album_name,
    build_catalog,
    candidate_ids,
    covered_seconds,
    day_bounds,
    parse_album_id,
    parse_track_id,
    playable,
    recent_days,
    track_id,
)
from entropy_fm.styles import STYLES
from entropy_fm.timeline import VoiceKind

VIENNA = ZoneInfo("Europe/Vienna")
AREAS = {"kitchen": "Kitchen", "living_room": "Living Room", "empty": "Attic"}
ENTITY_AREAS = {
    "sensor.fridge_power": "kitchen",
    "binary_sensor.kitchen_motion": "kitchen",
    "light.kitchen_ceiling": "kitchen",
    "sensor.living_temp": "living_room",
    "person.jozef": "living_room",
    "lock.front_door": "living_room",
    "sensor.router_password": "living_room",
    "sensor.no_area": None,
    "button.restart": "kitchen",
}


def _state(entity_id: str, state: str, **attributes: str) -> dict[str, object]:
    return {"entity_id": entity_id, "state": state, "attributes": attributes}


STATES = [
    _state("sensor.fridge_power", "80", device_class="power", unit_of_measurement="W"),
    _state("binary_sensor.kitchen_motion", "off", device_class="motion", friendly_name="Motion"),
    _state("light.kitchen_ceiling", "on", friendly_name="Ceiling"),
    _state("sensor.living_temp", "21", device_class="temperature", unit_of_measurement="°C"),
    _state("person.jozef", "home"),
    _state("lock.front_door", "locked"),
    _state("sensor.router_password", "123"),
    _state("sensor.no_area", "5"),
    _state("sun.sun", "above_horizon"),
]


def test_candidates_need_an_area_and_a_voice_domain() -> None:
    """Only entities of a house area, in a domain that can play and not private, are fetched."""
    assert candidate_ids(ENTITY_AREAS) == [
        "binary_sensor.kitchen_motion",
        "light.kitchen_ceiling",
        "sensor.fridge_power",
        "sensor.living_temp",
    ]
    with_presence = candidate_ids(ENTITY_AREAS, include_presence=True)
    assert "person.jozef" in with_presence
    assert "lock.front_door" not in with_presence
    assert "sensor.living_temp" not in candidate_ids(ENTITY_AREAS, ["sensor.living_*"])


def test_catalog_groups_entities_by_area_and_drops_the_private_ones() -> None:
    """Areas become artists with their voices; empty areas and private entities are gone."""
    catalog = build_catalog(AREAS, ENTITY_AREAS, STATES)
    assert list(catalog.areas) == ["kitchen", "living_room"]
    kitchen = catalog.entities["kitchen"]
    assert [info.name for info in kitchen] == ["Ceiling", "Motion", "sensor.fridge_power"]
    assert {info.kind for info in kitchen} == {VoiceKind.CHORD, VoiceKind.PLUCK, VoiceKind.DRONE}
    living = [info.entity_id for info in catalog.entities["living_room"]]
    assert living == ["sensor.living_temp"]
    assert catalog.sun is not None
    assert catalog.sun.entity_id == "sun.sun"


def test_catalog_includes_presence_only_on_request() -> None:
    """Persons appear only when the user included them."""
    catalog = build_catalog(AREAS, ENTITY_AREAS, STATES, include_presence=True)
    assert "person.jozef" in [info.entity_id for info in catalog.entities["living_room"]]


def test_the_elevation_sensor_wins_over_the_sun_entity() -> None:
    """When there is a numeric elevation, it is the one that plays."""
    states = [*STATES, _state("sensor.sun_solar_elevation", "12.5", unit_of_measurement="°")]
    catalog = build_catalog(AREAS, ENTITY_AREAS, states)
    assert catalog.sun is not None
    assert catalog.sun.entity_id == "sensor.sun_solar_elevation"


def test_ids_round_trip() -> None:
    """Album and track ids carry the area, the day and the target."""
    day = date(2026, 11, 17)
    assert album_id("kitchen", day) == "kitchen|2026-11-17"
    assert parse_album_id("kitchen|2026-11-17") == ("kitchen", day)
    item = track_id("kitchen", day, "sensor.fridge_power")
    assert item == "kitchen|2026-11-17|sensor.fridge_power"
    ref = parse_track_id(item)
    assert (ref.area_id, ref.day, ref.target) == ("kitchen", day, "sensor.fridge_power")
    assert track_id("kitchen", day) == "kitchen|2026-11-17|_scape"


def test_track_ids_carry_the_style() -> None:
    """The style is the fourth part of a track id; an id without one is Ambient, as it was."""
    day = date(2026, 11, 17)
    item = track_id("kitchen", day, "sensor.fridge_power", "berlin_school")
    assert item == "kitchen|2026-11-17|sensor.fridge_power|berlin_school"
    assert parse_track_id(item).style == "berlin_school"
    assert parse_track_id(item).target == "sensor.fridge_power"
    assert parse_track_id("kitchen|2026-11-17|_scape").style == "ambient"
    # every style gives an id of its own for the same day
    assert len({track_id("kitchen", day, style=key) for key in STYLES}) == len(STYLES)


@pytest.mark.parametrize("bad", ["", "kitchen", "kitchen|", "kitchen|nonsense", "|2026-11-17"])
def test_bad_ids_are_refused(bad: str) -> None:
    """A malformed album id raises ValueError, which the provider turns into not found."""
    with pytest.raises(ValueError, match=r".*"):
        parse_album_id(bad)


def test_track_ids_need_three_parts() -> None:
    """A track id without a target is not a track id."""
    with pytest.raises(ValueError, match=r".*"):
        parse_track_id("kitchen|2026-11-17")


def test_album_name() -> None:
    """Names read like "Kitchen, Tue 17 Nov"."""
    assert album_name("Kitchen", date(2026, 11, 17)) == "Kitchen, Tue 17 Nov"
    assert album_name("Whole House", date(2026, 1, 1)) == "Whole House, Thu 1 Jan"


def test_recent_days_start_with_today() -> None:
    """The library goes back from today, newest first."""
    assert recent_days(date(2026, 3, 2), 3) == [
        date(2026, 3, 2),
        date(2026, 3, 1),
        date(2026, 2, 28),
    ]


def test_a_day_with_a_clock_change_is_not_24_hours() -> None:
    """The last Sunday of March has 23 hours in Vienna, of October 25."""
    spring = day_bounds(date(2026, 3, 29), VIENNA)
    autumn = day_bounds(date(2026, 10, 25), VIENNA)
    assert spring[1].timestamp() - spring[0].timestamp() == 23 * 3600
    assert autumn[1].timestamp() - autumn[0].timestamp() == 25 * 3600


def test_the_covered_seconds_of_a_day_stop_at_now_for_today() -> None:
    """A past day is covered whole; today only up to what has happened."""
    now = datetime(2026, 11, 17, 12, 0, tzinfo=VIENNA)
    assert covered_seconds(date(2026, 11, 16), VIENNA, now) == 24 * 3600
    assert covered_seconds(date(2026, 11, 17), VIENNA, now.astimezone(UTC)) == 12 * 3600
    assert covered_seconds(date(2026, 11, 18), VIENNA, now) == 0


def test_an_album_of_a_day_that_has_barely_begun_is_not_offered() -> None:
    """A soundscape needs a few hours of the day to have happened."""
    early = datetime(2026, 11, 17, 0, 5, tzinfo=VIENNA)
    assert not playable(date(2026, 11, 17), VIENNA, early)
    assert playable(date(2026, 11, 16), VIENNA, early)
    assert not playable(date(2026, 11, 17), VIENNA, datetime(2026, 11, 17, 1, 0, tzinfo=VIENNA))
    assert playable(date(2026, 11, 17), VIENNA, datetime(2026, 11, 17, 3, 30, tzinfo=VIENNA))
