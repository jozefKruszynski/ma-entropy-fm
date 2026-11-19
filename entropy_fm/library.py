"""
Map Home Assistant onto a music library: areas are artists, days of an area are albums.

This module is pure: it works on plain dictionaries as Home Assistant returns them and on the
types of `timeline`, and imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from .const import (
    MIN_SCAPE_SECONDS_OF_DAY,
    SCAPE_ID,
)
from .day_analysis import QUIET_KINDS
from .roles import EntityMeta, classify
from .styles import DEFAULT_STYLE
from .timeline import SUN_ELEVATION_ENTITY, SUN_ENTITY, EntityInfo, VoiceKind, is_excluded

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import tzinfo

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# the domains whose entities can become a voice
VOICE_DOMAINS = frozenset(
    {
        "sensor",
        "binary_sensor",
        "light",
        "switch",
        "input_boolean",
        "fan",
        "media_player",
        "cover",
        "select",
        "input_select",
        "person",
        "device_tracker",
    }
)
POWER_KINDS = frozenset({VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER})
SUN_ENTITIES = (SUN_ELEVATION_ENTITY, SUN_ENTITY)


@dataclass(frozen=True)
class Catalog:
    """The areas of the house and the entities that can play in each."""

    # area id to name
    areas: dict[str, str] = field(default_factory=dict)
    # area id to its entities, only areas with at least one
    entities: dict[str, list[EntityInfo]] = field(default_factory=dict)
    # the entity that tells where the sun is, when the house has one
    sun: EntityInfo | None = None
    # entity id to when its state last changed (epoch seconds), for those whose state said so
    changed: dict[str, float] = field(default_factory=dict)
    # area id to the icon of the area in Home Assistant, for the areas that have one
    icons: dict[str, str] = field(default_factory=dict)

    def all_infos(self) -> list[EntityInfo]:
        """Return the entities of every area."""
        return [info for area in self.entities.values() for info in area]


@dataclass(frozen=True)
class TrackRef:
    """What a track id points at."""

    area_id: str
    day: date
    # always SCAPE_ID: a track is the soundscape of a room (or the house) on a day, in a style
    target: str
    # the key of the style the track is played in
    style: str = DEFAULT_STYLE


def resolve_area(
    entity_id: str,
    entity_area: Mapping[str, str | None],
    entity_device_area: Mapping[str, str | None],
) -> str | None:
    """
    Return the area of an entity: its own, else the one of its device.

    :param entity_id: The entity id.
    :param entity_area: The area set on each entity.
    :param entity_device_area: The area of the device of each entity.
    """
    return entity_area.get(entity_id) or entity_device_area.get(entity_id)


def candidate_ids(
    entity_areas: Mapping[str, str | None],
    exclude_globs: Iterable[str] = (),
    *,
    include_presence: bool = False,
) -> list[str]:
    """
    Return the entities that have an area, can make a voice and are not excluded.

    This decides which states are fetched, so that a big house is not dumped in full.

    :param entity_areas: Entity id to its resolved area id.
    :param exclude_globs: The user's patterns of entities to leave out.
    :param include_presence: Whether persons and device trackers may play.
    """
    globs = list(exclude_globs)
    return sorted(
        entity_id
        for entity_id, area_id in entity_areas.items()
        if area_id
        and entity_id.split(".", 1)[0] in VOICE_DOMAINS
        and not is_excluded(entity_id, globs, include_presence=include_presence)
    )


def build_catalog(  # noqa: PLR0913
    areas: Mapping[str, str],
    entity_areas: Mapping[str, str | None],
    states: Iterable[Mapping[str, Any]],
    exclude_globs: Iterable[str] = (),
    *,
    include_presence: bool = False,
    meta: Mapping[str, EntityMeta] | None = None,
    overrides: Mapping[str, str] | None = None,
    icons: Mapping[str, str] | None = None,
) -> Catalog:
    """
    Build the catalog from the registries and the current states, without any history.

    :param areas: Area id to area name.
    :param entity_areas: Entity id to its resolved area id.
    :param states: The current states of the candidate entities, and of the sun.
    :param exclude_globs: The user's patterns of entities to leave out.
    :param include_presence: Whether persons and device trackers may play.
    :param meta: What the registries say about each entity: platform, device, labels.
    :param overrides: Entity id to a role the user chose in the settings.
    :param icons: Area id to the icon set on the area in Home Assistant.
    """
    globs = list(exclude_globs)
    per_area: dict[str, list[EntityInfo]] = {}
    changed: dict[str, float] = {}
    sun: EntityInfo | None = None
    for state in states:
        entity_id = str(state.get("entity_id", ""))
        attributes = state.get("attributes") or {}
        name = str(attributes.get("friendly_name") or entity_id)
        if entity_id in SUN_ENTITIES:
            # the elevation sensor, when there is one, tells more than above or below
            if sun is None or entity_id == SUN_ELEVATION_ENTITY:
                sun = EntityInfo(entity_id, name, None, VoiceKind.SUN)
            continue
        area_id = entity_areas.get(entity_id)
        if not area_id or area_id not in areas:
            continue
        if is_excluded(entity_id, globs, include_presence=include_presence):
            continue
        found = classify(
            entity_id,
            name,
            attributes.get("device_class"),
            attributes.get("unit_of_measurement"),
            str(state.get("state")),
            meta=(meta or {}).get(entity_id),
            override=(overrides or {}).get(entity_id.lower()),
            include_presence=include_presence,
        )
        if found is None:
            continue
        info = EntityInfo(
            entity_id,
            name,
            area_id,
            found.kind,
            found.cumulative,
            found.description,
            str(attributes.get("device_class") or "").lower(),
            (meta or {}).get(entity_id, EntityMeta()).device_id,
        )
        per_area.setdefault(area_id, []).append(info)
        if (stamp := _epoch(state.get("last_changed"))) is not None:
            changed[entity_id] = stamp
    for infos in per_area.values():
        # an energy meter of a device that has a power sensor says the same thing, slower
        powered = {
            info.device_id
            for info in infos
            if info.device_id and not info.cumulative and info.kind in POWER_KINDS
        }
        infos[:] = [info for info in infos if not (info.cumulative and info.device_id in powered)]
    entities = {
        area_id: sorted(infos, key=lambda info: (info.name.lower(), info.entity_id))
        for area_id, infos in sorted(per_area.items(), key=lambda item: areas[item[0]].lower())
    }
    return Catalog(
        {area_id: areas[area_id] for area_id in entities},
        entities,
        sun,
        changed,
        {area_id: icon for area_id, icon in (icons or {}).items() if area_id in entities},
    )


def _epoch(value: object) -> float | None:
    """Return a time that Home Assistant gave, as text or a datetime, in epoch seconds; None when it is not one."""
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def quiet_areas(catalog: Catalog, since: datetime) -> set[str]:
    """
    Return the areas in which nothing happened since a time: no entity that makes a note or an event changed.

    Only what can make something happen counts: the temperature, the humidity, the sun and the like colour the
    day but never play, so an area that has nothing else is quiet. An entity whose state has not changed since
    `since` did not change in the window either. When Home Assistant did not say when an entity last changed, the
    area is not called quiet: it is offered unless we know better.

    :param catalog: The areas and their entities, with when each state last changed.
    :param since: The start of the window.
    """
    start = since.timestamp()
    quiet: set[str] = set()
    for area_id, infos in catalog.entities.items():
        moving = (info for info in infos if info.kind not in QUIET_KINDS)
        if not any(catalog.changed.get(info.entity_id, start) >= start for info in moving):
            quiet.add(area_id)
    return quiet


def album_name(area_name: str, day: date) -> str:
    """
    Return the name of an album, such as "Kitchen, Tue 17 Nov".

    :param area_name: The name of the area.
    :param day: The day.
    """
    return f"{area_name}, {WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def album_id(area_id: str, day: date) -> str:
    """
    Return the item id of an album.

    :param area_id: The area id, or the id of the whole house.
    :param day: The day.
    """
    return f"{area_id}|{day.isoformat()}"


def track_id(area_id: str, day: date, target: str = SCAPE_ID, style: str | None = None) -> str:
    """
    Return the item id of a track; it holds the style, so that each style is its own version.

    :param area_id: The area id, or the id of the whole house.
    :param day: The day.
    :param target: The entity id of a voice, or the id of the ensemble.
    :param style: The key of the style; without one the id is the old one, which plays Ambient.
    """
    item = f"{area_id}|{day.isoformat()}|{target}"
    return f"{item}|{style}" if style else item


def parse_album_id(item_id: str) -> tuple[str, date]:
    """
    Split an album id into area id and day.

    :param item_id: An id made by `album_id`.
    :raises ValueError: When it is not one.
    """
    area_id, _, day = item_id.partition("|")
    if not area_id or not day:
        raise ValueError(item_id)
    return area_id, date.fromisoformat(day)


def parse_track_id(item_id: str) -> TrackRef:
    """
    Split a track id into its parts; a missing style is the default one.

    :param item_id: An id made by `track_id`.
    :raises ValueError: When it is not one.
    """
    area_id, _, rest = item_id.partition("|")
    day, _, rest = rest.partition("|")
    target, _, style = rest.partition("|")
    if not area_id or not day or not target:
        raise ValueError(item_id)
    return TrackRef(area_id, date.fromisoformat(day), target, style or DEFAULT_STYLE)


def recent_days(today: date, count: int) -> list[date]:
    """
    Return the last days, newest first, starting with today.

    :param today: Today in the time zone of the house.
    :param count: How many days.
    """
    return [today - timedelta(days=offset) for offset in range(count)]


def day_bounds(day: date, tz: tzinfo) -> tuple[datetime, datetime]:
    """
    Return the local start and end of a day; a day with a clock change is 23 or 25 hours.

    :param day: The day.
    :param tz: The time zone of the house.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    following = day + timedelta(days=1)
    return start, datetime(following.year, following.month, following.day, tzinfo=tz)


def covered_seconds(day: date, tz: tzinfo, now: datetime) -> float:
    """
    Return how many seconds of the day have happened: the whole day for a past one.

    :param day: The day.
    :param tz: The time zone of the house.
    :param now: The current time.
    """
    start, end = day_bounds(day, tz)
    now = now.astimezone(tz)
    if now >= end:
        return end.timestamp() - start.timestamp()
    return max(0.0, now.timestamp() - start.timestamp())


def track_seconds(day: date, tz: tzinfo, now: datetime, compression: float) -> int:
    """
    Return the length of a track in seconds of audio: only up to now for today.

    :param day: The day.
    :param tz: The time zone of the house.
    :param now: The current time.
    :param compression: Seconds of house time per second of audio.
    """
    return int(covered_seconds(day, tz, now) / compression)


def playable(day: date, tz: tzinfo, now: datetime, compression: float) -> bool:
    """
    Return True when the day has enough in it to make an album.

    :param day: The day.
    :param tz: The time zone of the house.
    :param now: The current time.
    :param compression: Seconds of house time per second of audio.
    """
    del compression  # a soundscape needs a few hours of the day, whatever the compression
    return covered_seconds(day, tz, now) >= MIN_SCAPE_SECONDS_OF_DAY
