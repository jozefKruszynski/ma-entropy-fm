"""Tests of the library, the pins and the audio cache through the provider and a fake house."""

import asyncio
import itertools
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.media_items import Album, Artist, Track

from entropy_fm import EntropyFMProvider
from entropy_fm.const import CONF_QUALITY, STORAGE_FOLDER
from entropy_fm.pins import load_membership
from music_assistant.mass import MusicAssistant
from tests.test_provider import (  # noqa: F401
    _all_albums,
    _collect,
    _setup,
    _stream,
)


def _quiet(provider: EntropyFMProvider, monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    """Change settings of the provider for a test."""
    real = provider.get_config_value
    monkeypatch.setattr(
        provider, "get_config_value", lambda key, default=None: values.get(key, real(key, default))
    )


async def _album(provider: EntropyFMProvider, area: str, index: int) -> Album:
    """Return the album of an area, `index` days from the newest that is a whole day."""
    albums = [a for a in await provider.get_artist_albums(area) if a.item_id.startswith(f"{area}|")]
    return albums[index]


async def test_the_features_of_a_streaming_provider_with_a_library_to_edit(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The provider says it can edit artists, albums and tracks of the library."""
    from music_assistant_models.enums import ProviderFeature  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    for feature in (
        ProviderFeature.LIBRARY_ARTISTS_EDIT,
        ProviderFeature.LIBRARY_ALBUMS_EDIT,
        ProviderFeature.LIBRARY_TRACKS_EDIT,
        ProviderFeature.ARTIST_ALBUMS,
        ProviderFeature.BROWSE,
        ProviderFeature.SEARCH,
    ):
        assert feature in provider.supported_features
    assert provider.is_streaming_provider


def _clock(provider: EntropyFMProvider, monkeypatch: pytest.MonkeyPatch, shift_days: int) -> None:
    """Move the clock of the provider some days on, as midnights go by."""
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    real = provider._days_and_catalog  # noqa: SLF001

    async def shifted() -> tuple:
        days, catalog, tz, now = await real()
        later = now + timedelta(days=shift_days)
        return days, catalog, tz, later

    async def now() -> datetime:
        return datetime.now(UTC).astimezone(await provider._timezone()) + timedelta(days=shift_days)  # noqa: SLF001

    monkeypatch.setattr(provider, "_days_and_catalog", shifted)
    monkeypatch.setattr(provider, "_now", now)


async def test_the_library_always_has_every_room_and_a_window_of_days(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """With no action: the rooms and the whole house are artists, each with the last 7 days."""
    provider, _ = await _setup(mass, fake_hass)
    artists = await _collect(provider.get_library_artists())
    assert [a.name for a in artists] == ["Whole House", "Kitchen", "Living Room"]
    albums = await _collect(provider.get_library_albums())
    per_area: dict[str, list[str]] = {}
    for album in albums:
        area, day = album.item_id.split("|")[:2]
        per_area.setdefault(area, []).append(day)
    assert set(per_area) == {"_house", "kitchen", "living_room"}
    for days in per_area.values():
        assert 5 <= len(days) <= 7
        assert days == sorted(days, reverse=True)
    # one track per album in the library: the default style; the album holds all the styles
    tracks = await _collect(provider.get_library_tracks())
    assert len(tracks) == len(albums)
    assert {t.item_id.rsplit("|", 1)[1] for t in tracks} == {"ambient"}
    assert len(await provider.get_album_tracks(albums[0].item_id)) == 6


async def test_the_window_rolls_over_at_midnight(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A day on, the newest day is added and the oldest one leaves; the setting is clamped."""
    provider, _ = await _setup(mass, fake_hass)
    before = [
        a.item_id
        for a in await _collect(provider.get_library_albums())
        if a.item_id.startswith("kitchen|")
    ]
    _clock(provider, monkeypatch, 1)
    after = [
        a.item_id
        for a in await _collect(provider.get_library_albums())
        if a.item_id.startswith("kitchen|")
    ]
    assert after[0] > before[0]
    assert before[-1] not in after
    assert set(before[:-1]) <= set(after)
    # the setting cannot go past the days that Home Assistant keeps
    _quiet(provider, monkeypatch, days=3, library_days=9)
    assert provider._library_days == 3  # noqa: SLF001
    _quiet(provider, monkeypatch, days=10, library_days=2)
    assert provider._library_days == 2  # noqa: SLF001


async def test_adding_an_album_pins_the_day_and_it_stays_after_the_window_moves_on(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin lives in the storage path of the server; a pinned day outlives the window."""
    provider, _ = await _setup(mass, fake_hass)
    _quiet(provider, monkeypatch, library_days=2)
    album = await _album(provider, "kitchen", 5)
    day = album.item_id.split("|")[1]
    window = [a.item_id for a in await _collect(provider.get_library_albums())]
    assert album.item_id not in window
    assert await provider.library_add(album)
    pin_path = provider.pins.path("kitchen", date.fromisoformat(day))
    assert pin_path.is_file()
    assert Path(mass.storage_path) / STORAGE_FOLDER in pin_path.parents
    assert Path(__file__).parents[1] / "entropy_fm" not in pin_path.parents
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    _clock(provider, monkeypatch, 3)
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    # outside the window, removing it makes it drop out
    assert await provider.library_remove(album.item_id, MediaType.ALBUM)
    assert not pin_path.exists()
    assert album.item_id not in [a.item_id for a in await _collect(provider.get_library_albums())]


async def test_removing_an_album_inside_the_window_sticks_until_it_is_added_again(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A removed album is unpinned and left out of the listings, also inside the window; adding it clears that."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 1)
    day = date.fromisoformat(album.item_id.split("|")[1])
    default_track = album.item_id + "|_scape|ambient"
    assert await provider.library_add(album)
    assert provider.pins.has("kitchen", day)
    assert await provider.library_remove(album.item_id, MediaType.ALBUM)
    assert not provider.pins.has("kitchen", day)
    assert album.item_id not in [a.item_id for a in await _collect(provider.get_library_albums())]
    # the track of the removed day goes with it, the other days are untouched
    tracks = [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert default_track not in tracks
    assert len(tracks) == len(await _collect(provider.get_library_albums()))
    # it is still there to browse and to add again
    assert album.item_id in [a.item_id for a in await provider.get_artist_albums("kitchen")]
    assert await provider.library_add(album)
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    assert default_track in [t.item_id for t in await _collect(provider.get_library_tracks())]


async def test_removing_a_track_inside_the_window_sticks_until_it_is_added_again(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The default track of a day in the window stays out after a remove, and the removal is stored."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 2)
    item_id = album.item_id + "|_scape|ambient"
    before = [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert item_id in before
    assert await provider.library_remove(item_id, MediaType.TRACK)
    after = [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert item_id not in after
    assert set(before) - set(after) == {item_id}
    # the album stays; only the track was removed; and it survives a restart
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    stored = load_membership(provider.storage_root)
    assert stored.removed == {item_id: album.item_id.split("|")[1]}
    # a track of another style is another track
    other = album.item_id + "|_scape|minimal"
    assert await provider.library_add(await provider.get_track(other))
    assert other in [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert await provider.library_add(await provider.get_track(item_id))
    assert item_id in [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert not provider.membership.removed


async def test_the_removed_items_of_days_nobody_lists_are_forgotten(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list of removed items only holds days that the browse window still has."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 1)
    item_id = album.item_id + "|_scape|ambient"
    await provider.library_remove(item_id, MediaType.TRACK)
    await provider.library_remove(album.item_id, MediaType.ALBUM)
    assert len(provider.membership.removed) == 2
    _clock(provider, monkeypatch, 4)
    await provider._forget_old_removals()  # noqa: SLF001
    assert len(provider.membership.removed) == 2
    _clock(provider, monkeypatch, 30)
    await provider._forget_old_removals()  # noqa: SLF001
    assert provider.membership.removed == {}
    assert load_membership(provider.storage_root).removed == {}


async def test_a_removal_sticks_through_the_library_sync_of_the_server(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Remove a track and an album from the library of the server: the next sync does not bring them back."""
    from tests.test_rooms import _sync  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    await _sync(mass, provider)
    tracks = await mass.music.tracks.library_items(provider=provider.instance_id, limit=5000)
    albums = await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
    track = next(
        t
        for t in tracks
        if (t.item_id and "kitchen" in t.name.lower()) or t.artists[0].name == "Kitchen"
    )
    album = next(a for a in albums if a.artists[0].name == "Living Room")
    track_prov = next(
        m.item_id for m in track.provider_mappings if m.provider_instance == provider.instance_id
    )
    album_prov = next(
        m.item_id for m in album.provider_mappings if m.provider_instance == provider.instance_id
    )
    # what the server does when the user removes them: it tells the provider, and drops its own record
    assert await provider.library_remove(track_prov, MediaType.TRACK)
    assert await provider.library_remove(album_prov, MediaType.ALBUM)
    await mass.music.tracks.remove_item_from_library(track.item_id)
    await mass.music.albums.remove_item_from_library(album.item_id)
    await _sync(mass, provider)
    names = {
        m.item_id
        for t in await mass.music.tracks.library_items(provider=provider.instance_id, limit=5000)
        for m in t.provider_mappings
    }
    assert track_prov not in names
    assert album_prov + "|_scape|ambient" not in names
    ids = {
        m.item_id
        for a in await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
        for m in a.provider_mappings
    }
    assert album_prov not in ids
    # adding the album again brings it back with the next sync
    await provider.library_add(await provider.get_album(album_prov))
    await _sync(mass, provider)
    ids = {
        m.item_id
        for a in await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
        for m in a.provider_mappings
    }
    assert album_prov in ids


async def test_a_room_cannot_be_removed_and_the_log_says_how_to_hide_it(
    mass: MusicAssistant, fake_hass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Removing an artist stays unsupported; the log points at *Rooms to show*."""
    provider, _ = await _setup(mass, fake_hass)
    with caplog.at_level("INFO"):
        assert not await provider.library_remove("kitchen", MediaType.ARTIST)
    assert "Rooms to show" in caplog.text
    assert provider.membership.removed == {}
    assert not await provider.library_remove("nonsense", MediaType.TRACK)
    assert provider.membership.removed == {}


async def test_adding_a_track_pins_its_day_and_it_stays_while_an_album_needs_it(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A day that an album and a track both need is deleted only when both are gone."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 2)
    tracks = await provider.get_album_tracks(album.item_id)
    track = next(t for t in tracks if t.item_id.endswith("|minimal"))
    assert await provider.library_add(track)
    assert await provider.library_add(album)
    assert ("kitchen", date.fromisoformat(album.item_id.split("|")[1])) in provider.pins.all()
    library = await _collect(provider.get_library_tracks())
    assert track.item_id in [t.item_id for t in library]
    await provider.library_remove(album.item_id, MediaType.ALBUM)
    assert provider.pins.all()
    await provider.library_remove(track.item_id, MediaType.TRACK)
    assert provider.pins.all() == []


async def test_a_room_is_always_in_the_library_and_cannot_be_added_or_removed(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Adding a room does nothing, and removing it says no; neither pins anything."""
    provider, _ = await _setup(mass, fake_hass)
    artist = await provider.get_artist("kitchen")
    assert await provider.library_add(artist)
    assert provider.pins.all() == []
    assert not await provider.library_remove("kitchen", MediaType.ARTIST)
    assert "Kitchen" in [a.name for a in await _collect(provider.get_library_artists())]


async def test_keep_every_day_of_these_rooms_is_a_setting(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rooms in the setting have every day pinned; taking one out lets its pins go."""
    provider, _ = await _setup(mass, fake_hass)
    options = [
        o.value
        for entry in await provider._room_entries()  # noqa: SLF001
        if entry.key == "keep_rooms"
        for o in entry.options or []
    ]
    assert sorted(options) == ["kitchen", "living_room"]
    browsable = await provider.get_artist_albums("kitchen")
    _quiet(provider, monkeypatch, keep_rooms=["kitchen"])
    await provider._pin_followed()  # noqa: SLF001
    pinned = {day for area, day in provider.pins.all() if area == "kitchen"}
    # every day that is over; today is pinned when it is
    today = str((await provider._now()).date())  # noqa: SLF001
    assert len(pinned) == len([a for a in browsable if a.item_id.split("|")[1] < today])
    assert len(pinned) >= len(browsable) - 2
    assert not any(area == "living_room" for area, _ in provider.pins.all())
    # a day that went missing is pinned again by the daily check
    oldest = min(pinned)
    provider.pins.delete("kitchen", oldest)
    await provider._pin_followed()  # noqa: SLF001
    assert provider.pins.has("kitchen", oldest)
    _quiet(provider, monkeypatch, keep_rooms=[])
    await provider._pin_followed()  # noqa: SLF001
    assert provider.pins.all() == []


async def test_rooms_that_artists_followed_move_into_the_setting(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier version kept the days of followed artists; they become the room setting."""
    provider, _ = await _setup(mass, fake_hass)
    provider.membership.followed["kitchen"] = "2026-11-01"
    saved: list[dict] = []

    async def save(domain: str, values: dict, instance_id: str | None = None) -> None:  # noqa: ARG001
        saved.append(values)

    monkeypatch.setattr(mass.config, "save_provider_config", save)
    assert provider._keep_rooms == {"kitchen"}  # noqa: SLF001
    await provider._migrate_followed()  # noqa: SLF001
    assert saved == [{"keep_rooms": ["kitchen"]}]
    assert provider.membership.followed == {}


async def test_the_library_sync_of_the_server_keeps_pinned_days(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a streaming provider an album that leaves the listing is not deleted; a pinned one stays."""
    provider, _ = await _setup(mass, fake_hass)
    pinned = await _album(provider, "kitchen", 1)
    plain = await _album(provider, "kitchen", 0)
    await provider.library_add(pinned)
    await provider.sync_library(MediaType.ALBUM)
    db_ids = {
        album.name: album
        for album in await mass.music.albums.library_items(provider=provider.instance_id)
    }
    assert pinned.name in db_ids
    assert plain.name in db_ids
    # three days on: both are out of the window, only the pinned one is still listed
    _quiet(provider, monkeypatch, library_days=1)
    _clock(provider, monkeypatch, 3)
    listed = {a.item_id for a in await _collect(provider.get_library_albums())}
    assert pinned.item_id in listed
    assert plain.item_id not in listed
    await provider.sync_library(MediaType.ALBUM)
    # the album that is no longer listed is not deleted from the database; it is no longer "in library"
    kept = await mass.music.albums.library_items(provider=provider.instance_id)
    names = {album.name for album in kept}
    assert pinned.name in names
    assert plain.name not in names
    every = await mass.music.albums.get_library_item_by_prov_id(plain.item_id, provider.instance_id)
    assert every is not None
    assert not any(
        m.in_library for m in every.provider_mappings if m.provider_instance == provider.instance_id
    )


async def test_a_pinned_day_plays_after_home_assistant_has_forgotten_it(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the recorder purges the day, and the days browsed shrink, the pin still plays."""
    provider, fake = await _setup(mass, fake_hass)
    # a fixed tier: the benchmark of Auto may finish between the two renders and change it
    _quiet(provider, monkeypatch, day_character=False, **{CONF_QUALITY: "standard"})
    # the lossy audio cache is off, so that the two renders can be compared sample for sample
    provider.audio_cache.limit = 0
    album = await _album(provider, "kitchen", 7)
    item_id = f"{album.item_id}|_scape|berlin_school"
    await provider.library_add(album)
    before = await _stream(provider, item_id, seek=300, chunks=2)
    # the recorder purges everything, the server forgets what it cached, and only 2 days are kept
    fake.hass.purged = True
    await mass.cache.clear()
    _quiet(provider, monkeypatch, day_character=False, days=2)
    browsable = {a.item_id for a in await provider.get_artist_albums("kitchen")}
    assert album.item_id in browsable
    assert len(browsable) <= 3
    after = await _stream(provider, item_id, seek=300, chunks=2)
    assert after == before
    assert any(after)
    # the track and the album can be looked up too
    assert (await provider.get_album(album.item_id)).name == album.name
    assert (await provider.get_track(item_id)).album is not None
    # without the pin the day would be gone
    await provider.library_remove(album.item_id, MediaType.ALBUM)
    with pytest.raises(Exception, match="not found"):
        await provider.get_album(album.item_id)


async def test_the_engine_version_is_in_the_pin_and_a_pin_renders_with_the_current_engine(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The pin keeps the history and the engine version of the day it was made; no audio."""
    from entropy_fm.const import ENGINE_VERSION  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 4)
    await provider.library_add(album)
    pin = provider.pins.read(
        "kitchen", __import__("datetime").date.fromisoformat(album.item_id.split("|")[1])
    )
    assert pin is not None
    assert pin.engine == ENGINE_VERSION
    assert pin.history
    assert not any(
        path.suffix in {".opus", ".mp3", ".wav"} for path in provider.pins.root.rglob("*")
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_a_past_day_is_cached_after_it_played_to_the_end_and_streams_from_the_file(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A miss renders live and records; the next stream is the file; today is never cached."""
    from music_assistant_models.enums import StreamType  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 5)
    item_id = f"{album.item_id}|_scape|minimal"
    details = await provider.get_stream_details(item_id, MediaType.TRACK)
    assert details.stream_type == StreamType.CUSTOM
    assert isinstance(details.data, str)
    # play the whole track, cut to six seconds so that the test is quick
    module = sys.modules[type(provider).__module__]
    real_render = module.render_pcm

    def short(*args: Any, **kwargs: Any) -> Any:
        inner = real_render(*args, **kwargs)
        try:
            yield from itertools.islice(inner, 6)
        finally:
            inner.close()

    monkeypatch.setattr(module, "render_pcm", short)
    seen = 0
    async for _chunk in provider.get_audio_stream(details, 0):
        seen += 1
    assert seen == 6
    monkeypatch.setattr(module, "render_pcm", real_render)
    again = await provider.get_stream_details(item_id, MediaType.TRACK)
    assert again.stream_type == StreamType.LOCAL_FILE
    assert isinstance(again.path, str)
    assert Path(again.path).is_file()
    assert Path(again.path).is_relative_to(Path(mass.cache_path) / STORAGE_FOLDER)
    assert again.can_seek
    # a stream that is stopped halfway records nothing
    other = f"{album.item_id}|_scape|lofi"
    details = await provider.get_stream_details(other, MediaType.TRACK)
    audio = provider.get_audio_stream(details, 0)
    await anext(audio)
    await audio.aclose()
    await asyncio.sleep(0.2)
    assert (
        await provider.get_stream_details(other, MediaType.TRACK)
    ).stream_type == StreamType.CUSTOM
    assert not list((Path(mass.cache_path) / STORAGE_FOLDER).glob("*.part"))
    # a seek never records, and the cache switched off never serves
    _quiet(provider, monkeypatch, audio_cache_mb=0)
    provider.audio_cache.limit = 0
    assert (
        await provider.get_stream_details(item_id, MediaType.TRACK)
    ).stream_type == StreamType.CUSTOM


async def test_today_is_never_cached(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """The key of a track of today is None: it is not recorded and not served."""
    from datetime import datetime  # noqa: PLC0415

    from entropy_fm.library import parse_track_id  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    now = await provider._now()  # noqa: SLF001
    if now.hour < 1:
        pytest.skip("too early in the day")
    today = parse_track_id(f"kitchen|{now.date().isoformat()}|_scape|ambient")
    assert provider._audio_key(today, now.date()) is None  # noqa: SLF001
    yesterday = parse_track_id(
        f"kitchen|{(now.date().replace(day=now.day)).isoformat()}|_scape|ambient"
    )
    del yesterday, datetime


async def test_the_settings_say_the_cache_and_the_tier(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Quality defaults to Auto, which is Standard until the benchmark has run; the cache is 500 MB."""
    provider, _ = await _setup(mass, fake_hass)
    assert provider.audio_cache.limit == 500 * 1_000_000
    assert provider.quality.key in {"light", "standard", "rich"}
    assert Track is not None
    assert Artist is not None


async def test_voice_tracks_of_an_earlier_version_are_dropped_and_ensemble_pins_stay(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A track of one entity leaves the library with its pins; an ensemble track keeps its own."""
    provider, _ = await _setup(mass, fake_hass)
    keep = await _album(provider, "kitchen", 3)
    gone = await _album(provider, "kitchen", 6)
    old_id = f"{keep.item_id}|_song|ambient"
    keep_id = f"{keep.item_id}|_scape|ambient"
    stale_id = f"{gone.item_id}|sensor.fridge_power|ambient"
    # an old library holds a timeline: it is the soundscape of its day now, with the pins of its day
    provider.membership.tracks.add(old_id)
    await provider._remember(  # noqa: SLF001
        old_id, [("kitchen", date.fromisoformat(keep.item_id.split("|")[1]))]
    )
    await provider._migrate_ensemble_tracks()  # noqa: SLF001
    assert old_id not in provider.membership.tracks
    # an old library: the voice track is in the membership, with the pins it needed
    provider.membership.tracks.add(stale_id)
    await provider._remember(  # noqa: SLF001
        stale_id, [("kitchen", date.fromisoformat(gone.item_id.split("|")[1]))]
    )
    assert provider.pins.has("kitchen", date.fromisoformat(gone.item_id.split("|")[1]))
    await provider._drop_voice_tracks()  # noqa: SLF001
    assert stale_id not in provider.membership.tracks
    assert not provider.pins.has("kitchen", date.fromisoformat(gone.item_id.split("|")[1]))
    assert keep_id in provider.membership.tracks
    assert provider.pins.has("kitchen", date.fromisoformat(keep.item_id.split("|")[1]))
    listed = [t.item_id for t in await _collect(provider.get_library_tracks())]
    assert keep_id in listed
    assert not any("sensor.fridge_power" in item for item in listed)
    voice = await provider.get_track(keep_id)
    assert not await provider.library_add(
        voice.__class__.from_dict({**voice.to_dict(), "item_id": stale_id})
    )
