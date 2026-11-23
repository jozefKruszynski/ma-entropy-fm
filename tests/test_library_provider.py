"""Tests of the library, the pins and the audio cache through the provider and a fake house."""

import asyncio
import itertools
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album, Artist, Track

from entropy_fm import EntropyFMProvider
from entropy_fm.constants import CONF_QUALITY, STORAGE_FOLDER
from entropy_fm.library import Catalog, parse_track_id
from entropy_fm.pins import load_membership
from entropy_fm.styles import STYLES
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


async def _heart(
    provider: EntropyFMProvider, item: Album | Artist | Track | str, kind: Any = None
) -> None:
    """Heart an item, as Music Assistant tells the provider when the user taps the heart."""
    if isinstance(item, str):
        await provider.set_favorite(item, kind, favorite=True)
    else:
        await provider.set_favorite(item.item_id, item.media_type, favorite=True)


async def _unheart(
    provider: EntropyFMProvider, item_id: str, kind: Any, *, value: bool | None = None
) -> None:
    """Take the heart off an item (None clears the state, False is a dislike)."""
    await provider.set_favorite(item_id, kind, favorite=value)


async def test_the_features_hearts_but_no_library_edit(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The provider has hearts for rooms, albums and tracks; the library itself is not edited."""
    from music_assistant_models.enums import ProviderFeature  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    for feature in (
        ProviderFeature.FAVORITE_ARTISTS_EDIT,
        ProviderFeature.FAVORITE_ALBUMS_EDIT,
        ProviderFeature.FAVORITE_TRACKS_EDIT,
        ProviderFeature.LIBRARY_ARTISTS,
        ProviderFeature.LIBRARY_ALBUMS,
        ProviderFeature.LIBRARY_TRACKS,
        ProviderFeature.ARTIST_ALBUMS,
        ProviderFeature.BROWSE,
        ProviderFeature.SEARCH,
    ):
        assert feature in provider.supported_features
    for feature in (
        ProviderFeature.LIBRARY_ARTISTS_EDIT,
        ProviderFeature.LIBRARY_ALBUMS_EDIT,
        ProviderFeature.LIBRARY_TRACKS_EDIT,
    ):
        assert feature not in provider.supported_features
    assert not provider.is_streaming_provider


def _clock(provider: EntropyFMProvider, monkeypatch: pytest.MonkeyPatch, shift_days: int) -> None:
    """Move the clock of the provider some days on, as midnights go by."""
    from datetime import UTC, datetime, tzinfo  # noqa: PLC0415

    real = provider._days_and_catalog  # noqa: SLF001

    async def shifted() -> tuple[list[date], Catalog, tzinfo, datetime]:
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
    """With no action: the rooms and the whole house are artists, each with the days of the window."""
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
        assert 5 <= len(days) <= 10
        assert days == sorted(days, reverse=True)
    # every track of every album in the library: all five styles
    tracks = await _collect(provider.get_library_tracks())
    assert len(tracks) == len(STYLES) * len(albums)
    assert len({t.item_id for t in tracks}) == len(tracks)
    assert {t.item_id.rsplit("|", 1)[1] for t in tracks} == {
        "ambient",
        "berlin_school",
        "lofi",
        "ritual",
        "glitch",
    }
    for album in albums:
        own = {t.item_id for t in tracks if t.item_id.startswith(album.item_id + "|")}
        assert own == {t.item_id for t in await provider.get_album_tracks(album.item_id)}
    # nothing is a favourite until the user hearts it; never False, which would be a dislike
    assert all(item.favorite is None for item in [*artists, *albums, *tracks])


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
    _quiet(provider, monkeypatch, days=99)
    assert provider._days == 10  # noqa: SLF001
    _quiet(provider, monkeypatch, days=2)
    assert provider._days == 2  # noqa: SLF001


async def test_hearting_an_album_pins_the_day_and_it_stays_after_the_window_moves_on(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin lives in the storage path of the server; a hearted day outlives the window and shows its heart."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 5)
    _quiet(provider, monkeypatch, days=2)
    day = album.item_id.split("|")[1]
    window = [a.item_id for a in await _collect(provider.get_library_albums())]
    assert album.item_id not in window
    await _heart(provider, album)
    pin_path = provider.pins.path("kitchen", date.fromisoformat(day))
    assert pin_path.is_file()
    assert Path(mass.storage_path) / STORAGE_FOLDER in pin_path.parents
    assert Path(__file__).parents[1] / "entropy_fm" not in pin_path.parents
    listed = await _collect(provider.get_library_albums())
    assert [a.favorite for a in listed if a.item_id == album.item_id] == [True]
    assert all(a.favorite is None for a in listed if a.item_id != album.item_id)
    # all five tracks of the kept day are in the library, too
    own = [
        t.item_id
        for t in await _collect(provider.get_library_tracks())
        if t.item_id.startswith(album.item_id)
    ]
    assert len(own) == len(STYLES)
    _clock(provider, monkeypatch, 3)
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    # it survives a restart
    assert album.item_id in load_membership(provider.storage_root).albums
    # outside the window, un-hearting releases the pin and the day ages out
    await _unheart(provider, album.item_id, MediaType.ALBUM)
    assert not pin_path.exists()
    assert album.item_id not in [a.item_id for a in await _collect(provider.get_library_albums())]
    assert album.item_id not in load_membership(provider.storage_root).albums


async def test_a_day_inside_the_window_stays_listed_when_it_is_unhearted(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Nothing is removed from the overviews by a heart: un-hearting only releases the pin."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 1)
    day = date.fromisoformat(album.item_id.split("|")[1])
    track_id = album.item_id + "|_scape|ambient"
    await _heart(provider, album)
    assert provider.pins.has("kitchen", day)
    await _unheart(provider, album.item_id, MediaType.ALBUM)
    assert not provider.pins.has("kitchen", day)
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    assert track_id in [t.item_id for t in await _collect(provider.get_library_tracks())]
    # a dislike releases too, a heart on something that is not ours is refused, and on another type does nothing
    await _heart(provider, album)
    await _unheart(provider, album.item_id, MediaType.ALBUM, value=False)
    assert not provider.pins.has("kitchen", day)
    for kind in (MediaType.TRACK, MediaType.ALBUM):
        with pytest.raises(MediaNotFoundError):
            await _heart(provider, "nonsense", kind)
    await _heart(provider, "kitchen|2026-01-01", MediaType.PLAYLIST)
    assert provider.pins.all() == []
    assert not provider.membership.tracks
    assert not provider.membership.albums


async def test_hearting_a_track_pins_its_day_and_reports_the_heart(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heart on one style keeps the day; only that track shows the heart; un-hearting lets the day go."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 4)
    day = date.fromisoformat(album.item_id.split("|")[1])
    item_id = album.item_id + "|_scape|lofi"
    track = await provider.get_track(item_id)
    _quiet(provider, monkeypatch, days=2)
    assert album.item_id not in [a.item_id for a in await _collect(provider.get_library_albums())]
    await _heart(provider, track)
    assert provider.pins.has("kitchen", day)
    assert album.item_id in [a.item_id for a in await _collect(provider.get_library_albums())]
    tracks = {t.item_id: t for t in await _collect(provider.get_library_tracks())}
    assert len([i for i in tracks if i.startswith(album.item_id)]) == len(STYLES)
    assert [i for i, t in tracks.items() if t.favorite] == [item_id]
    assert (await provider.get_track(item_id)).favorite is True
    await _unheart(provider, item_id, MediaType.TRACK)
    assert not provider.pins.has("kitchen", day)
    assert album.item_id not in [a.item_id for a in await _collect(provider.get_library_albums())]
    assert all(t.favorite is None for t in await _collect(provider.get_library_tracks()))
    # the day is outside the window and no longer pinned, so the track is gone, heart and all
    with pytest.raises(MediaNotFoundError):
        await provider.get_track(item_id)


async def test_a_heart_on_the_whole_house_keeps_the_day_of_every_room(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The day of the whole house is made of its rooms; un-hearting lets all of them go."""
    provider, _ = await _setup(mass, fake_hass)
    album = (await provider.get_artist_albums("_house"))[3]
    day = date.fromisoformat(album.item_id.split("|")[1])
    await _heart(provider, album)
    assert {area for area, d in provider.pins.all() if d == day} == {"kitchen", "living_room"}
    await _unheart(provider, album.item_id, MediaType.ALBUM)
    assert provider.pins.all() == []


async def test_a_heart_on_a_room_keeps_every_day_of_it(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Hearting a room pins all its days that are over; un-hearting releases them, unless a heart or the setting needs them."""
    provider, _ = await _setup(mass, fake_hass)
    browsable = await provider.get_artist_albums("kitchen")
    today = str((await provider._now()).date())  # noqa: SLF001
    over = {a.item_id.split("|")[1] for a in browsable if a.item_id.split("|")[1] < today}
    await _heart(provider, await provider.get_artist("kitchen"))
    assert (await provider.get_artist("kitchen")).favorite is True
    assert (await provider.get_artist("living_room")).favorite is None
    artists = {a.item_id: a.favorite for a in await _collect(provider.get_library_artists())}
    assert artists["kitchen"] is True
    assert artists["living_room"] is None
    pinned = {day.isoformat() for area, day in provider.pins.all() if area == "kitchen"}
    assert pinned == over
    assert not any(area == "living_room" for area, _ in provider.pins.all())
    assert "kitchen" in load_membership(provider.storage_root).followed
    # a heart on an album of that room needs its day too, so that day stays when the room is let go
    album = next(a for a in browsable if a.item_id.split("|")[1] in over)
    await _heart(provider, album)
    await _unheart(provider, "kitchen", MediaType.ARTIST)
    assert provider.membership.followed == {}
    assert [(area, day.isoformat()) for area, day in provider.pins.all()] == [
        ("kitchen", album.item_id.split("|")[1])
    ]
    assert (await provider.get_artist("kitchen")).favorite is None
    # a room that is not ours cannot be hearted
    with pytest.raises(MediaNotFoundError):
        await _heart(provider, "attic", MediaType.ARTIST)
    assert "attic" not in provider.membership.followed


async def test_the_library_sync_of_the_server_lists_every_track_and_records_the_hearts(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """After a sync the library has all five tracks per album, and the heart is on the item that the sync reads."""
    from tests.test_rooms import _sync  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 1)
    await _heart(provider, album)
    await _sync(mass, provider)
    albums = await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
    tracks = await mass.music.tracks.library_items(provider=provider.instance_id, limit=5000)
    assert len(tracks) == len(STYLES) * len(albums)
    # what the server is given as the heart of the provider
    listed = {a.item_id: a.favorite async for a in provider.get_library_albums()}
    assert listed[album.item_id] is True
    assert [i for i, favorite in listed.items() if favorite] == [album.item_id]


async def test_adding_a_track_pins_its_day_and_it_stays_while_an_album_needs_it(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A day that an album and a track both need is deleted only when both are gone."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 2)
    tracks = await provider.get_album_tracks(album.item_id)
    track = next(t for t in tracks if t.item_id.endswith("|lofi"))
    await _heart(provider, track)
    await _heart(provider, album)
    assert ("kitchen", date.fromisoformat(album.item_id.split("|")[1])) in provider.pins.all()
    library = await _collect(provider.get_library_tracks())
    assert track.item_id in [t.item_id for t in library]
    await _unheart(provider, album.item_id, MediaType.ALBUM)
    assert provider.pins.all()
    await _unheart(provider, track.item_id, MediaType.TRACK)
    assert provider.pins.all() == []


async def test_a_room_is_always_in_the_library_whatever_the_heart_says(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Un-hearting a room leaves it in the library; only hearting it pins days."""
    provider, _ = await _setup(mass, fake_hass)
    await _unheart(provider, "kitchen", MediaType.ARTIST)
    assert provider.pins.all() == []
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
    assert sorted(options, key=str) == ["kitchen", "living_room"]
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


async def test_the_library_sync_of_the_server_keeps_pinned_days(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An album that leaves the listing is removed from the database; a pinned one stays."""
    provider, _ = await _setup(mass, fake_hass)
    pinned = await _album(provider, "kitchen", 1)
    plain = await _album(provider, "kitchen", 0)
    await _heart(provider, pinned)
    await provider.sync_library(MediaType.ALBUM)
    db_ids = {
        album.name: album
        for album in await mass.music.albums.library_items(provider=provider.instance_id)
    }
    assert pinned.name in db_ids
    assert plain.name in db_ids
    # three days on: both are out of the window, only the pinned one is still listed
    _quiet(provider, monkeypatch, days=1)
    _clock(provider, monkeypatch, 3)
    listed = {a.item_id for a in await _collect(provider.get_library_albums())}
    assert pinned.item_id in listed
    assert plain.item_id not in listed
    await provider.sync_library(MediaType.ALBUM)
    kept = await mass.music.albums.library_items(provider=provider.instance_id)
    names = {album.name for album in kept}
    assert pinned.name in names
    assert plain.name not in names
    gone = await mass.music.albums.get_library_item_by_prov_id(plain.item_id, provider.instance_id)
    assert gone is None


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
    await _heart(provider, album)
    before = await _stream(provider, item_id, seek=200, chunks=2)
    # the recorder purges everything, the server forgets what it cached, and only 2 days are kept
    fake.hass.purged = True
    await mass.cache.clear()
    _quiet(provider, monkeypatch, day_character=False, days=2)
    browsable = {a.item_id for a in await provider.get_artist_albums("kitchen")}
    assert album.item_id in browsable
    assert len(browsable) <= 3
    after = await _stream(provider, item_id, seek=200, chunks=2)
    assert after == before
    assert any(after)
    # the track and the album can be looked up too
    assert (await provider.get_album(album.item_id)).name == album.name
    assert (await provider.get_track(item_id)).album is not None
    # without the pin the day would be gone
    await _unheart(provider, album.item_id, MediaType.ALBUM)
    with pytest.raises(Exception, match="not found"):
        await provider.get_album(album.item_id)


async def test_the_engine_version_is_in_the_pin_and_a_pin_renders_with_the_current_engine(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The pin keeps the history and the engine version of the day it was made; no audio."""
    from entropy_fm.constants import ENGINE_VERSION  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 4)
    await _heart(provider, album)
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
    item_id = f"{album.item_id}|_scape|ambient"
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


# ---------------------------------------------------------------- the ids


async def test_a_track_id_holds_the_style_and_nothing_else() -> None:
    """The id of a track is the day, the target and the style; nothing after the style is an id."""
    from entropy_fm.library import track_id  # noqa: PLC0415

    day = date(2026, 11, 10)
    assert track_id("kitchen", day, "_scape", "lofi") == "kitchen|2026-11-10|_scape|lofi"
    assert parse_track_id("kitchen|2026-11-10|_scape|lofi").style == "lofi"
    # the id before there were styles still is the default style
    assert parse_track_id("kitchen|2026-11-10|_scape").style == "ambient"
    # the marks that earlier versions put after the style are not ids any more
    for old in ("soundscape", "condensed", "more"):
        with pytest.raises(ValueError):  # noqa: PT011
            parse_track_id(f"kitchen|2026-11-10|_scape|lofi|{old}")


async def test_after_a_sync_the_library_has_the_condensed_tracks(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The server stores the listed tracks, each with the condensed length of its style."""
    from entropy_fm.plan import condensed_seconds  # noqa: PLC0415
    from tests.test_rooms import _sync  # noqa: PLC0415

    provider, _ = await _setup(mass, fake_hass)
    await _sync(mass, provider)
    stored = await mass.music.tracks.library_items(provider=provider.instance_id, limit=5000)
    assert stored
    assert all(t.duration == condensed_seconds(_style_of(t)) for t in stored)


def _style_of(track: Track) -> str:
    """Return the style that the id of a stored track names."""
    return parse_track_id(track.provider_mappings.copy().pop().item_id).style


async def test_listings_say_the_length_the_stream_says_for_past_days(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The library, the album, the track and the stream agree for finished days, in every style."""
    provider, _ = await _setup(mass, fake_hass)
    today = (await provider._now()).date()  # noqa: SLF001
    library = [t for t in await _collect(provider.get_library_tracks()) if _day(t) < today]
    assert library
    for track in library[:: max(len(library) // 12, 1)]:
        details = await provider.get_stream_details(track.item_id, MediaType.TRACK)
        assert track.duration == details.duration
        assert (await provider.get_track(track.item_id)).duration == details.duration
    album = library[0].album
    assert album is not None
    listed = {t.item_id: t.duration for t in await provider.get_album_tracks(album.item_id)}
    assert all(listed[t.item_id] == t.duration for t in library if t.album == album)


def _day(track: Track) -> date:
    """Return the day of a track."""
    return date.fromisoformat(track.item_id.split("|")[1])


async def test_a_heart_on_something_that_is_not_there_is_refused_before_anything_is_written(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A malformed id, an unknown room and a room name that climbs out of the folder are not found, and no pin is made."""
    provider, _ = await _setup(mass, fake_hass)
    day = "2026-11-10"
    for kind, item_id in (
        (MediaType.ALBUM, "nonsense"),
        (MediaType.TRACK, "nonsense"),
        (MediaType.ALBUM, f"..|{day}"),
        (MediaType.ALBUM, f"attic|{day}"),
        (MediaType.TRACK, f"..|{day}|_scape|lofi"),
        (MediaType.TRACK, f"attic|{day}|_scape|lofi"),
        (MediaType.TRACK, f"kitchen|{day}|_scape|vaporwave"),
        (MediaType.TRACK, f"kitchen|{day}|sensor.fridge_power|lofi"),
    ):
        with pytest.raises(MediaNotFoundError):
            await provider.set_favorite(item_id, kind, favorite=True)
    with pytest.raises(MediaNotFoundError):
        await provider.set_favorite("nonsense", MediaType.ALBUM, favorite=None)
    assert not provider.pins.all()
    assert not provider.membership.albums
    assert not provider.membership.tracks
