"""Tests of the settings that pick the rooms and the piece length, through the provider and a fake house."""

import asyncio
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError

from entropy_fm import EntropyFMProvider
from entropy_fm.const import (
    CONF_HIDE_QUIET,
    CONF_HOUSE_SELECTED,
    CONF_KEEP_ROOMS,
    CONF_LENGTH,
    CONF_PRERENDER,
    CONF_ROOMS,
    CONF_STYLES,
    WHOLE_HOUSE_ID,
)
from entropy_fm.library import Catalog, TrackRef, build_catalog, quiet_areas
from entropy_fm.plan import condensed_seconds
from entropy_fm.timeline import EntityInfo, VoiceKind
from music_assistant.mass import MusicAssistant
from tests import fake_hass as fake_module
from tests.fake_hass import RegistryEntry, _state
from tests.test_library_provider import _quiet
from tests.test_prerender import _prepared, _yesterday
from tests.test_provider import _collect, _setup, _stream

ALL = ["Whole House", "Kitchen", "Living Room"]


def _ago(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


@pytest.fixture
def hallway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a Hallway whose only motion sensor has not changed in a month, and a Cellar with a recent door."""
    areas: dict[str, dict[str, Any]] = {
        "hallway": {"area_id": "hallway", "name": "Hallway", "aliases": [], "picture": None},
        "cellar": {"area_id": "cellar", "name": "Cellar", "aliases": [], "picture": None},
    }
    entities = {
        "binary_sensor.hallway_motion": RegistryEntry("zha", None, "hallway"),
        "sensor.hallway_temperature": RegistryEntry("zha", None, "hallway"),
        "binary_sensor.cellar_door": RegistryEntry("zha", None, "cellar"),
    }
    states = [
        {
            **_state("binary_sensor.hallway_motion", "off", device_class="motion"),
            "last_changed": _ago(30),
        },
        {
            # the temperature moves all the time, but it never plays a note
            **_state(
                "sensor.hallway_temperature",
                "19",
                device_class="temperature",
                unit_of_measurement="°C",
            ),
            "last_changed": _ago(0.01),
        },
        {
            **_state("binary_sensor.cellar_door", "off", device_class="door"),
            "last_changed": _ago(3),
        },
    ]
    for key, value in areas.items():
        monkeypatch.setitem(fake_module.AREAS, key, value)
    for key, entry in entities.items():
        monkeypatch.setitem(fake_module.ENTITIES, key, entry)
    monkeypatch.setattr(fake_module, "STATES", [*fake_module.STATES, *states])


async def _names(provider: EntropyFMProvider) -> list[str]:
    return [artist.name for artist in await _collect(provider.get_library_artists())]


# ---------------------------------------------------------------- quiet rooms, without a server


def _info(entity_id: str, kind: VoiceKind) -> EntityInfo:
    return EntityInfo(entity_id, entity_id, "a", kind)


def test_a_room_is_quiet_when_nothing_that_plays_changed_in_the_window() -> None:
    """Static sensors, and sensors that only colour the day, make a quiet room; one change makes it live."""
    since = datetime(2026, 11, 10, tzinfo=UTC)
    old = since.timestamp() - 86400.0
    new = since.timestamp() + 86400.0
    entities = {
        "still": [_info("binary_sensor.a", VoiceKind.PLUCK)],
        "temperature_only": [
            _info("sensor.t", VoiceKind.TEMPERATURE),
            _info("sensor.h", VoiceKind.HUMIDITY),
        ],
        "moved": [_info("binary_sensor.b", VoiceKind.PLUCK), _info("light.c", VoiceKind.CHORD)],
        "unknown": [_info("binary_sensor.d", VoiceKind.PLUCK)],
    }
    changed = {
        "binary_sensor.a": old,
        "sensor.t": new,
        "sensor.h": new,
        "binary_sensor.b": old,
        "light.c": new,
    }
    catalog = Catalog({k: k for k in entities}, entities, None, changed)
    # nothing is said about when "unknown" last changed: it is offered, as we know no better
    assert quiet_areas(catalog, since) == {"still", "temperature_only"}


def test_the_catalog_reads_when_a_state_last_changed() -> None:
    """The catalog keeps the time of the last change of each entity that plays, whatever shape Home Assistant gave it."""
    states = [
        {
            "entity_id": "binary_sensor.k_door",
            "state": "off",
            "attributes": {"device_class": "door"},
            "last_changed": "2026-11-18T10:00:00+00:00",
        },
        {
            "entity_id": "binary_sensor.k_motion",
            "state": "off",
            "attributes": {"device_class": "motion"},
            "last_changed": "garbage",
        },
    ]
    catalog = build_catalog(
        {"k": "Kitchen"},
        {"binary_sensor.k_door": "k", "binary_sensor.k_motion": "k"},
        states,
    )
    assert catalog.changed == {
        "binary_sensor.k_door": datetime(2026, 11, 18, 10, tzinfo=UTC).timestamp()
    }


# ---------------------------------------------------------------- the settings


async def test_the_new_settings_and_their_defaults(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Rooms (empty is all), the whole house on the ticked rooms, hide quiet rooms and the piece length."""
    provider, _ = await _setup(mass, fake_hass)
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    rooms = entries[CONF_ROOMS]
    assert rooms.multi_value
    assert rooms.default_value == []
    assert [option.value for option in rooms.options] == ["kitchen", "living_room"]
    assert [option.value for option in entries[CONF_KEEP_ROOMS].options] == [
        "kitchen",
        "living_room",
    ]
    assert entries[CONF_HOUSE_SELECTED].default_value is True
    assert entries[CONF_HIDE_QUIET].default_value is True
    length = entries[CONF_LENGTH]
    assert length.default_value == "full"
    assert [option.value for option in length.options] == ["full", "condensed"]


async def test_no_room_ticked_is_every_room_and_a_ticked_room_is_alone(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty selection offers every room; ticking Kitchen takes Living Room out of browse, search and the library."""
    provider, _ = await _setup(mass, fake_hass)
    assert await _names(provider) == ALL
    root = await provider.browse("entropy_fm://")
    assert [item.name for item in root] == ALL

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    assert await _names(provider) == ["Whole House", "Kitchen"]
    assert [item.name for item in await provider.browse("entropy_fm://")] == [
        "Whole House",
        "Kitchen",
    ]
    with pytest.raises(KeyError):
        await provider.browse("entropy_fm://area/living_room")
    assert await provider.get_artist_albums("living_room") == []
    assert await provider.get_artist_albums("kitchen")

    found = await provider.search(
        "Living", [MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK], limit=50
    )
    assert not found.artists
    assert not found.albums
    assert not found.tracks
    found = await provider.search("Kitchen", [MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK])
    assert found.artists
    assert found.albums
    assert found.tracks

    albums = await _collect(provider.get_library_albums())
    assert {album.item_id.split("|")[0] for album in albums} == {"_house", "kitchen"}
    tracks = await _collect(provider.get_library_tracks())
    assert {track.item_id.split("|")[0] for track in tracks} == {"_house", "kitchen"}

    # a room is still found by its id, so that a queue that holds it keeps playing
    assert await provider.get_artist("living_room")


async def test_a_ticked_room_that_does_not_exist_leaves_nothing(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When none of the ticked rooms is in the house, no room and no whole house is offered."""
    provider, _ = await _setup(mass, fake_hass)
    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["attic"]})
    assert await _names(provider) == []


async def test_the_whole_house_is_made_of_the_ticked_rooms_unless_that_is_off(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The house mix reads the ticked rooms only; with the setting off it reads every room."""
    provider, _ = await _setup(mass, fake_hass)
    catalog = await provider._catalog_all()  # noqa: SLF001
    assert provider._house_rooms(catalog) == ["kitchen", "living_room"]  # noqa: SLF001
    parsed = TrackRef(WHOLE_HOUSE_ID, await _yesterday(provider), "_scape", "ambient")
    every = await provider._gather(parsed)  # noqa: SLF001
    assert {s.entity_id for s in every.series} >= {
        "binary_sensor.kitchen_door",
        "light.living_room_ceiling",
    }

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    assert provider._house_rooms(catalog) == ["kitchen"]  # noqa: SLF001
    only = await provider._gather(parsed)  # noqa: SLF001
    ids = {s.entity_id for s in only.series}
    assert "binary_sensor.kitchen_door" in ids
    assert not ids & {"light.living_room_ceiling", "binary_sensor.living_room_window"}

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"], CONF_HOUSE_SELECTED: False})
    assert provider._house_rooms(catalog) == ["kitchen", "living_room"]  # noqa: SLF001
    again = await provider._gather(parsed)  # noqa: SLF001
    assert {s.entity_id for s in again.series} == {s.entity_id for s in every.series}


@pytest.mark.usefixtures("hallway")
async def test_a_quiet_room_is_hidden_unless_the_setting_is_off(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A room whose motion sensor has not moved all window, and whose temperature never plays, is not offered."""
    provider, _ = await _setup(mass, fake_hass)
    assert await _names(provider) == ["Whole House", "Cellar", "Kitchen", "Living Room"]
    # nothing was read from the history to tell
    assert fake_hass.hass.history_calls == []
    # a room that is merely quiet on a day still offers that day: the cellar has every day of the window
    cellar = await provider.get_artist_albums("cellar")
    kitchen = await provider.get_artist_albums("kitchen")
    assert len(cellar) == len(kitchen) > 5

    _quiet(provider, monkeypatch, **{CONF_HIDE_QUIET: False})
    assert await _names(provider) == ["Whole House", "Cellar", "Hallway", "Kitchen", "Living Room"]

    # a smaller window makes the cellar quiet as well: its door last changed three days ago
    _quiet(provider, monkeypatch, **{CONF_HIDE_QUIET: True, "days": 2})
    assert "Cellar" not in await _names(provider)


async def test_a_deselected_room_leaves_the_library_but_keeps_its_pins_and_its_cached_audio(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned days of a room that is no longer ticked are hidden, not deleted, and come back when it is ticked again."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await provider.get_artist_albums("living_room")
    old = albums[-1]
    assert await provider.library_add(old)
    area, day = old.item_id.split("|")
    assert provider.pins.has(area, date.fromisoformat(day))
    provider.audio_cache.root.mkdir(parents=True, exist_ok=True)
    cached = provider.audio_cache.root / ("b" * 40 + ".opus")
    cached.write_bytes(b"x" * 1000)

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    listed = {a.item_id for a in await _collect(provider.get_library_albums())}
    assert old.item_id not in listed
    assert not any(item.startswith("living_room|") for item in listed)
    assert provider.pins.has(area, date.fromisoformat(day))
    assert old.item_id in provider.membership.albums
    assert cached.exists()

    _quiet(provider, monkeypatch, **{CONF_ROOMS: []})
    assert old.item_id in {a.item_id for a in await _collect(provider.get_library_albums())}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_the_pre_render_leaves_out_rooms_that_are_not_offered(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With all rooms, the whole house and the ticked rooms are rendered, and the others are not."""
    provider, played = await _prepared(
        mass,
        fake_hass,
        monkeypatch,
        **{CONF_PRERENDER: "all", CONF_STYLES: ["ambient"], CONF_ROOMS: ["kitchen"]},
    )
    assert await provider._prerender_yesterday() == 2  # noqa: SLF001
    assert {ref.area_id for ref in played} == {"_house", "kitchen"}


# ---------------------------------------------------------------- the length


async def test_the_audio_cache_key_holds_the_length_and_the_rooms_of_the_house(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full and condensed never share a file, and neither do two house mixes made of other rooms."""
    provider, _ = await _setup(mass, fake_hass)
    day = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001
    today = (await provider._now()).date()  # noqa: SLF001
    kitchen = TrackRef("kitchen", day, "_scape", "lofi")
    house = TrackRef(WHOLE_HOUSE_ID, day, "_scape", "lofi")
    full = provider._audio_key(kitchen, today)  # noqa: SLF001
    full_house = provider._audio_key(house, today, await provider._catalog_all())  # noqa: SLF001
    assert full is not None
    assert full_house is not None

    _quiet(provider, monkeypatch, **{CONF_LENGTH: "condensed"})
    short = provider._audio_key(kitchen, today)  # noqa: SLF001
    short_house = provider._audio_key(house, today, await provider._catalog_all())  # noqa: SLF001
    assert short not in {None, full}
    assert short_house not in {None, full_house}

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    # a room does not change with the ticks; the whole house does
    assert provider._audio_key(kitchen, today) == short  # noqa: SLF001
    assert provider._audio_key(house, today, await provider._catalog_all()) != short_house  # noqa: SLF001
    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"], CONF_LENGTH: "full"})
    assert provider._audio_key(house, today, await provider._catalog_all()) != full_house  # noqa: SLF001


async def test_a_condensed_track_is_listed_with_the_length_it_plays(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow styles show 7 minutes and fast ones 5, the stream says the same, and the audio is made."""
    provider, _ = await _setup(mass, fake_hass)
    real = provider.get_config_value
    monkeypatch.setattr(
        provider,
        "get_config_value",
        lambda key, default=None: {"render_quality": "light", CONF_LENGTH: "condensed"}.get(
            key, real(key, default)
        ),
    )
    past = (await provider.get_artist_albums("kitchen"))[-1]
    tracks = await provider.get_album_tracks(past.item_id)
    listed = {t.item_id.rsplit("|", 1)[1]: t.duration for t in tracks}
    assert listed == {
        "ambient": 420,
        "berlin_school": 420,
        "lofi": 300,
        "minimal": 300,
        "ritual": 420,
        "glitch": 300,
    }
    for style in ("lofi", "ambient"):
        item_id = f"{past.item_id}|_scape|{style}"
        details = await provider.get_stream_details(item_id, MediaType.TRACK)
        assert details.duration == condensed_seconds(style)
        assert (await provider.get_track(item_id)).duration == details.duration
        assert len(await _stream(provider, item_id, seek=100, chunks=2)) == 2 * 44100 * 4


async def test_a_condensed_soundscape_of_today_still_needs_three_hours_of_the_day(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing and the stream refuse a day with less than three hours, whatever the length."""
    from entropy_fm.const import MIN_SCAPE_SECONDS_OF_DAY  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    _quiet(provider, monkeypatch, **{CONF_LENGTH: "condensed"})
    day = date(2026, 11, 18)
    two_hours = int(2 * 3600 / provider._compression)  # noqa: SLF001
    four_hours = int(4 * 3600 / provider._compression)  # noqa: SLF001
    assert not provider._day_tracks("kitchen", "Kitchen", day, two_hours, None)  # noqa: SLF001
    tracks = provider._day_tracks("kitchen", "Kitchen", day, four_hours, None)  # noqa: SLF001
    assert len(tracks) == 6
    assert two_hours * provider._compression < MIN_SCAPE_SECONDS_OF_DAY  # noqa: SLF001

    async def early() -> datetime:
        tz = await provider._timezone()  # noqa: SLF001
        return datetime.now(tz).replace(hour=2, minute=0)

    monkeypatch.setattr(provider, "_now", early)
    with pytest.raises(MediaNotFoundError):
        await provider._scape_of(TrackRef("kitchen", (await early()).date(), "_scape", "lofi"))  # noqa: SLF001


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_the_pre_render_renders_the_length_that_is_set(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Condensed pre-renders into its own file; the full file of the same track is not there."""
    provider, played = await _prepared(
        mass,
        fake_hass,
        monkeypatch,
        **{
            CONF_PRERENDER: "pinned",
            CONF_KEEP_ROOMS: ["kitchen"],
            CONF_STYLES: ["ambient"],
            CONF_LENGTH: "condensed",
        },
    )
    day = await _yesterday(provider)
    today = (await provider._now()).date()  # noqa: SLF001
    assert await provider._prerender_yesterday() == 1  # noqa: SLF001
    ref = TrackRef("kitchen", day, "_scape", "ambient")
    assert played == [ref]
    short = provider._audio_key(ref, today)  # noqa: SLF001
    assert short is not None
    assert provider.audio_cache.path(short).is_file()
    _quiet(provider, monkeypatch, **{CONF_LENGTH: "full"})
    full = provider._audio_key(ref, today)  # noqa: SLF001
    assert full is not None
    assert full != short
    assert not provider.audio_cache.path(full).exists()


# ---------------------------------------------------------------- the library follows the rooms


async def _sync(mass: MusicAssistant, provider: EntropyFMProvider) -> None:
    """Run the library sync of the provider to the end."""
    tasks = await mass.music.start_sync(providers=[provider.instance_id])
    assert tasks
    for _ in range(600):
        await asyncio.sleep(0.05)
        if not mass.music.active_sync_tasks:
            return
    pytest.fail("the library sync did not end")


async def _library_artists(mass: MusicAssistant, provider: EntropyFMProvider) -> set[str]:
    """Return the artists in the library views of the server, which only list what is in the library."""
    items = await mass.music.artists.library_items(provider=provider.instance_id)
    return {artist.name for artist in items}


async def _library_albums(mass: MusicAssistant, provider: EntropyFMProvider) -> set[str]:
    items = await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
    return {album.artists[0].name for album in items}


async def test_a_change_of_the_rooms_asks_for_one_library_sync_and_the_views_follow(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unticking a room starts a sync of this source; the room leaves the album view, and returns when ticked."""
    provider, _ = await _setup(mass, fake_hass)
    asked: list[list[str]] = []
    real_start = mass.music.start_sync

    async def start_sync(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs["providers"])
        return await real_start(*args, **kwargs)

    monkeypatch.setattr(mass.music, "start_sync", start_sync)
    # the first time there is nothing known, so it syncs; the same rooms again do not
    assert await provider.sync_if_rooms_changed()
    assert not await provider.sync_if_rooms_changed()
    assert asked == [[provider.instance_id]]
    await _sync(mass, provider)
    asked.clear()
    assert await _library_albums(mass, provider) == {"Whole House", "Kitchen", "Living Room"}

    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    assert await provider.sync_if_rooms_changed()
    assert not await provider.sync_if_rooms_changed()
    assert len(asked) == 1
    await _sync(mass, provider)
    assert await _library_albums(mass, provider) == {"Whole House", "Kitchen"}
    assert await _library_artists(mass, provider) == {"Whole House", "Kitchen"}

    _quiet(provider, monkeypatch, **{CONF_ROOMS: []})
    assert await provider.sync_if_rooms_changed()
    await _sync(mass, provider)
    assert await _library_albums(mass, provider) == {"Whole House", "Kitchen", "Living Room"}
    assert await _library_artists(mass, provider) == {"Whole House", "Kitchen", "Living Room"}


async def test_no_sync_is_asked_for_when_home_assistant_cannot_tell_the_rooms(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed or empty catalog never starts a sync, and the known rooms are not overwritten."""
    from music_assistant_models.errors import ProviderUnavailableError  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    asked: list[object] = []

    async def start_sync(*_args: Any, **kwargs: Any) -> list[Any]:
        asked.append(kwargs)
        return []

    monkeypatch.setattr(mass.music, "start_sync", start_sync)

    async def failing() -> Catalog:
        raise ProviderUnavailableError("down")

    monkeypatch.setattr(provider, "_get_catalog", failing)
    assert not await provider.sync_if_rooms_changed()

    async def empty() -> Catalog:
        return Catalog()

    monkeypatch.setattr(provider, "_get_catalog", empty)
    assert not await provider.sync_if_rooms_changed()
    assert asked == []


async def test_a_reload_asks_for_the_sync_in_the_background(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load starts a task that waits, then asks; unloading cancels it."""
    monkeypatch.setattr("entropy_fm.LIBRARY_SYNC_DELAY", 0.0)
    provider, _ = await _setup(mass, fake_hass)
    task = provider._sync_task  # noqa: SLF001
    assert task is not None
    await asyncio.wait({task}, timeout=30)
    assert task.done()
    assert (Path(provider.storage_root) / "offered_rooms.json").exists()


async def test_the_keep_list_only_offers_the_rooms_that_are_shown(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no room ticked to show, every room can be kept; with some ticked, only those."""
    provider, _ = await _setup(mass, fake_hass)

    async def keep_options() -> list[str]:
        entries = {e.key: e for e in await provider.get_config_entries()}
        assert list(entries).index(CONF_ROOMS) < list(entries).index(CONF_KEEP_ROOMS)
        return [str(option.value) for option in entries[CONF_KEEP_ROOMS].options]

    assert sorted(await keep_options()) == ["kitchen", "living_room"]
    _quiet(provider, monkeypatch, **{CONF_ROOMS: ["kitchen"]})
    assert await keep_options() == ["kitchen"]


async def test_a_change_of_the_piece_length_asks_for_a_library_sync_and_every_listing_agrees(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Condensed asks for one sync like a change of rooms; get_track, album and library tracks list its lengths."""
    provider, _ = await _setup(mass, fake_hass)
    asked: list[list[str]] = []

    async def start_sync(*_args: Any, **kwargs: Any) -> list[Any]:
        asked.append(kwargs["providers"])
        return []

    monkeypatch.setattr(mass.music, "start_sync", start_sync)
    assert await provider.sync_if_rooms_changed()
    assert not await provider.sync_if_rooms_changed()
    full = [t.duration async for t in provider.get_library_tracks()]
    assert set(full) == {900}

    _quiet(provider, monkeypatch, **{CONF_LENGTH: "condensed"})
    assert await provider.sync_if_rooms_changed()
    assert not await provider.sync_if_rooms_changed()
    assert len(asked) == 2

    albums = [
        a for a in await provider.get_artist_albums("kitchen") if a.item_id.startswith("kitchen|")
    ]
    listed = await provider.get_album_tracks(albums[2].item_id)
    for track in listed:
        style = track.item_id.rsplit("|", 1)[1]
        assert track.duration == condensed_seconds(style)
        assert (await provider.get_track(track.item_id)).duration == track.duration
    library = [t async for t in provider.get_library_tracks()]
    assert library
    assert {t.duration for t in library} == {condensed_seconds("ambient")}
    assert {t.duration for t in library} != set(full)

    # back to full: asked again
    _quiet(provider, monkeypatch, **{CONF_LENGTH: "full"})
    assert await provider.sync_if_rooms_changed()
    assert len(asked) == 3
    assert {t.duration async for t in provider.get_library_tracks()} == {900}


async def test_a_sync_marker_of_an_earlier_version_asks_for_one_more_sync(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker used to be the list of rooms; a list is not the current marker, so one sync follows."""
    import json  # noqa: PLC0415

    from entropy_fm.const import OFFERED_FILE  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    asked: list[object] = []

    async def start_sync(*_args: Any, **kwargs: Any) -> list[Any]:
        asked.append(kwargs)
        return []

    monkeypatch.setattr(mass.music, "start_sync", start_sync)
    Path(provider.storage_root).mkdir(parents=True, exist_ok=True)
    (Path(provider.storage_root) / OFFERED_FILE).write_text(
        json.dumps(["_house", "kitchen", "living_room"])
    )
    assert await provider.sync_if_rooms_changed()
    assert not await provider.sync_if_rooms_changed()
    assert len(asked) == 1
