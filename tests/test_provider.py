"""Tests of the provider, loaded by a real server that has a fake Home Assistant plugin."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from music_assistant_models.enums import ConfigEntryType, MediaType, ProviderTier
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album

from entropy_fm import EntropyFMProvider
from entropy_fm.const import CONF_LENGTH, MIN_SCAPE_SECONDS_OF_DAY
from entropy_fm.plan import FULL
from entropy_fm.styles import STYLES
from music_assistant.mass import MusicAssistant
from music_assistant.models.music_provider import MusicProvider
from tests import DOMAIN

CHUNK_BYTES = 44100 * 2 * 2
PLUGIN_WAIT_SECONDS = 2
VIENNA = ZoneInfo("Europe/Vienna")


async def _setup(
    mass: MusicAssistant, fake: MagicMock, length: str | None = None
) -> tuple[EntropyFMProvider, MagicMock]:
    """Add the provider, which finds the fake plugin; `length` sets the piece length, else it is the default."""
    step = await mass.config.setup_provider(DOMAIN)
    assert step.result is not None
    provider = mass.get_provider(step.result["instance_id"])
    assert isinstance(provider, MusicProvider)
    if length is not None:
        real = provider.get_config_value
        provider.get_config_value = lambda key, default=None: (  # type: ignore[assignment]
            length if key == CONF_LENGTH else real(key, default)
        )
    # the server loads the provider as its own module: it is the class of this repository by name
    return cast("EntropyFMProvider", provider), fake


def _target(item_id: str) -> str:
    """Return the target part of a track id: the soundscape."""
    return item_id.split("|")[2]


async def _collect[T](generator: AsyncIterator[T]) -> list[T]:
    return [item async for item in generator]


async def _all_albums(provider: MusicProvider) -> list[Album]:
    """Return the albums that can be browsed: every day of the whole house and of each area."""
    found: list[Album] = []
    for area_id in ("_house", "kitchen", "living_room"):
        found.extend(await provider.get_artist_albums(area_id))
    return found


async def test_provider_loads(mass: MusicAssistant) -> None:
    """The server loads the provider from this repository as a community provider."""
    assert mass.get_provider_manifest(DOMAIN).tier == ProviderTier.COMMUNITY
    manifest = mass.get_provider_manifest(DOMAIN)
    assert manifest.requirements == ["numpy==2.3.5", "pedalboard==0.9.26"]


async def test_setup_without_the_home_assistant_plugin_fails_clearly(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the plugin and without a URL the provider says what is missing."""
    real_sleep = asyncio.sleep

    async def quick_sleep(delay: float, *args: object) -> None:
        # the provider waits two seconds between looks for the plugin; do not wait in a test
        await real_sleep(0 if delay == PLUGIN_WAIT_SECONDS else delay, *args)

    monkeypatch.setattr(asyncio, "sleep", quick_sleep)
    with pytest.raises(Exception, match="Home Assistant plugin"):
        await mass.config.setup_provider(DOMAIN)


async def test_settings_have_the_expected_entries(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The settings page offers the documented options with their defaults."""
    provider, _ = await _setup(mass, fake_hass)
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert set(entries) == {
        "use_ha_plugin",
        "url",
        "token",
        "verify_ssl",
        "days",
        "key",
        "scale",
        "include_presence",
        "role_overrides",
        "exclude",
        "styles",
        "default_style",
        "energy_offset",
        "day_character",
        "render_quality",
        "audio_cache_mb",
        "prerender",
        "covers",
        "keep_rooms",
        "library_days",
        "rooms",
        "house_selected_rooms",
        "hide_quiet_rooms",
        "piece_length",
    }
    assert entries["key"].default_value == "D"
    assert entries["scale"].default_value == "dorian"
    assert entries["include_presence"].default_value is False
    assert entries["styles"].multi_value
    assert entries["styles"].default_value == list(STYLES)
    assert [option.value for option in entries["styles"].options] == list(STYLES)
    assert entries["default_style"].default_value == "ambient"
    assert entries["energy_offset"].default_value == 0
    assert entries["energy_offset"].range == (-50, 50)
    assert entries["token"].type == ConfigEntryType.SECURE_STRING
    assert entries["days"].range == (1, 10)
    assert entries["days"].default_value == 10
    assert entries["render_quality"].default_value == "auto"
    assert [option.value for option in entries["render_quality"].options] == [
        "auto",
        "light",
        "standard",
        "rich",
    ]
    assert entries["audio_cache_mb"].default_value == 500
    assert entries["prerender"].default_value == "pinned"
    assert [option.value for option in entries["prerender"].options] == ["off", "pinned", "all"]


async def test_browsing_shows_every_area_and_day_and_the_library_has_the_rooms(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Areas are artists, a whole house too, each has an album per day; the library has the rooms."""
    provider, fake = await _setup(mass, fake_hass)
    assert not provider.is_streaming_provider
    # the library is the rooms and a window of days from the start, whatever was added
    artists = await _collect(provider.get_library_artists())
    assert [a.name for a in artists] == ["Whole House", "Kitchen", "Living Room"]
    library_albums = await _collect(provider.get_library_albums())
    assert library_albums
    assert len(await _collect(provider.get_library_tracks())) == len(STYLES) * len(library_albums)
    # the library reads the days of its window for the titles of their tracks
    listed = len(fake.hass.history_calls)

    folders = await provider.browse(f"{provider.instance_id}://")
    assert [folder.name for folder in folders] == ["Whole House", "Kitchen", "Living Room"]
    albums = await provider.browse(f"{provider.instance_id}://area/kitchen")
    kitchen = [album for album in albums if isinstance(album, Album)]
    # ten days back, minus today when it has hardly begun
    assert 9 <= len(kitchen) <= 10
    assert kitchen[0].name.startswith("Kitchen, ")
    assert kitchen[0].item_id.startswith("kitchen|")
    assert all(mapping.is_unique for album in kitchen for mapping in album.provider_mappings)
    assert len(await _all_albums(provider)) == len(kitchen) * 3
    # browsing reads registries and states, never history
    assert len(fake.hass.history_calls) == listed
    with pytest.raises(KeyError):
        await provider.browse(f"{provider.instance_id}://nonsense")


async def test_the_library_is_rooms_days_and_ensembles_and_private_entities_stay_out(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Artists are rooms, albums their days, tracks the soundscape in each style; entities only shape."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await _all_albums(provider)
    names = {a.artists[0].name for a in albums}
    assert names == {"Kitchen", "Living Room", "Whole House"}
    today = datetime.now(UTC).astimezone(VIENNA).date().isoformat()
    for album in albums:
        if album.item_id.split("|")[1] >= today:
            continue
        tracks = await provider.get_album_tracks(album.item_id)
        # the soundscape of the day in each style
        assert [_target(t.item_id) for t in tracks] == ["_scape"] * len(STYLES)
        assert [t.item_id.rsplit("|", 1)[1] for t in tracks] == list(STYLES)
        assert [t.track_number for t in tracks] == list(range(1, len(STYLES) + 1))
        assert all(t.name for t in tracks)
    diagnostics = await provider.get_diagnostics()
    assert diagnostics is not None
    voices = diagnostics["voices"]
    assert isinstance(voices, dict)
    for hidden in (
        "person.jozef",
        "device_tracker.phone",
        "lock.front_door",
        "camera.garden",
        "sensor.router_password",
        "sensor.cellar_power",
        "sun.sun",
    ):
        assert hidden not in voices
    # the entities shape the sound: the fridge through its device, the lamp through its own area
    assert {"sensor.fridge_power", "light.lamp"} <= set(voices)


async def test_roles_are_explained_in_the_diagnostics(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A German name, a label and a plain meter each get a voice and say why."""
    provider, _ = await _setup(mass, fake_hass)
    diagnostics = await provider.get_diagnostics()
    assert diagnostics is not None
    voices = diagnostics["voices"]
    assert isinstance(voices, dict)
    assert voices["sensor.geschirrspueler_leistung"] == "dishwasher, role: dishwasher (name)"
    assert voices["sensor.kitchen_machine_power"] == "dishwasher, role: dishwasher (label)"
    assert "role: washer (name)" in str(voices["sensor.washing_machine_power"])
    assert "role: fridge (name)" in str(voices["sensor.fridge_power"])


async def test_whole_house_album_is_the_house_in_every_style(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The whole house album has the soundscape of the house, once in each style, each with its own title."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await _all_albums(provider)
    house = next(a for a in albums if a.item_id.startswith("_house|"))
    tracks = await provider.get_album_tracks(house.item_id)
    assert len({track.name for track in tracks}) == len(STYLES)
    assert {track.artists[0].name for track in tracks} == {"Whole House"}


async def test_a_full_track_is_listed_with_the_length_it_plays(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A past day is listed with its real length (10 to 22 minutes), the one the stream tells."""
    provider, _ = await _setup(mass, fake_hass, FULL)
    albums = await _all_albums(provider)
    past = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    tracks = await provider.get_album_tracks(past.item_id)
    assert len(tracks) == len(STYLES)
    for track in tracks:
        details = await provider.get_stream_details(track.item_id, MediaType.TRACK)
        assert 600 <= track.duration <= 1320
        assert track.duration == details.duration


async def test_get_track_and_unknown_items(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """Known ids resolve; unknown ones are not found."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await _all_albums(provider)
    album = next(a for a in albums if a.item_id.startswith("kitchen|"))
    track = await provider.get_track(f"{album.item_id}|_scape")
    assert track.name
    assert track.version == STYLES["ambient"].version
    # the song and the timeline of earlier versions are not tracks any more
    for old in ("_song", "_ensemble"):
        with pytest.raises(MediaNotFoundError):
            await provider.get_track(f"{album.item_id}|{old}")
    assert (await provider.get_artist("kitchen")).name == "Kitchen"
    assert (await provider.get_album(album.item_id)).name.startswith("Kitchen, ")
    for bad in (
        "nonsense",
        "kitchen|2020-01-01|_scape",
        f"{album.item_id}|person.jozef",
        f"{album.item_id}|sensor.fridge_power",
        f"{album.item_id}|@living_room",
    ):
        with pytest.raises(MediaNotFoundError):
            await provider.get_track(bad)
    with pytest.raises(MediaNotFoundError):
        await provider.get_album("garage|2026-01-01")
    with pytest.raises(MediaNotFoundError):
        await provider.get_artist("garage")


async def test_search(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """Search finds artists, albums and tracks by name."""
    provider, _ = await _setup(mass, fake_hass)
    result = await provider.search(
        "kitchen", [MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK], 50
    )
    assert [artist.name for artist in result.artists] == ["Kitchen"]
    assert result.albums
    assert all("kitchen" in album.name.lower() for album in result.albums)
    assert result.tracks
    assert result.tracks[0].version
    # entities are not tracks, so a search for one finds nothing
    assert (await provider.search("counter lamp", [MediaType.TRACK])).tracks == []


async def _stream(provider: MusicProvider, item_id: str, seek: int = 0, chunks: int = 3) -> bytes:
    details = await provider.get_stream_details(item_id, MediaType.TRACK)
    audio = provider.get_audio_stream(details, seek)
    data = b""
    try:
        for _ in range(chunks):
            data += await asyncio.wait_for(anext(audio), 60)
    finally:
        await audio.aclose()
    return data


async def test_stream_is_deterministic_seekable_pcm(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A track streams one second per chunk of 16 bit stereo, and the same seek is the same audio."""
    provider, fake = await _setup(mass, fake_hass, FULL)
    # a fixed tier: the benchmark of Auto may finish between two renders and change the tier
    real = provider.get_config_value
    monkeypatch.setattr(
        provider,
        "get_config_value",
        lambda key, default=None: "standard" if key == "quality" else real(key, default),
    )
    albums = await _all_albums(provider)
    past = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    item_id = f"{past.item_id}|_scape"

    details = await provider.get_stream_details(item_id, MediaType.TRACK)
    assert details.audio_format.sample_rate == 44100
    assert details.audio_format.bit_depth == 16
    assert details.audio_format.channels == 2
    assert details.allow_seek
    assert 600 <= (details.duration or 0) <= 1320

    first = await _stream(provider, item_id, seek=400)
    calls = len(fake.hass.history_calls)
    second = await _stream(provider, item_id, seek=400)
    assert len(first) == 3 * CHUNK_BYTES
    # the same audio, to within a rounding step: other tests of the same process leave state (the
    # caches of notes and kits, the graph of the engine) that may move a sample by one step of
    # 16 bits; the strict comparison is in test_a_style_is_deterministic_and_seekable
    a = np.frombuffer(first, dtype="<i2").astype(np.int32)
    b = np.frombuffer(second, dtype="<i2").astype(np.int32)
    assert np.abs(a - b).max() <= 2
    assert (a != b).mean() < 0.01
    assert first != await _stream(provider, item_id, seek=401)
    assert any(first)
    # the day, its sun and the days before it for the baseline were read once; then the cache
    assert len(fake.hass.history_calls) == calls
    call = next(c for c in fake.hass.history_calls if c["entity_ids"] != ["sun.sun"])
    assert "sensor.fridge_power" in call["entity_ids"]
    assert "person.jozef" not in call["entity_ids"]
    start = datetime.fromisoformat(call["start_time"]).astimezone(VIENNA)
    assert (start.hour, start.minute) == (0, 0)


async def test_every_kind_of_track_streams(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """The soundscape of a room and of the whole house play."""
    provider, _ = await _setup(mass, fake_hass, FULL)
    albums = await _all_albums(provider)
    kitchen = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    house = [a for a in albums if a.item_id.startswith("_house|")][-1]
    for item_id in (f"{kitchen.item_id}|_scape", f"{house.item_id}|_scape"):
        data = await _stream(provider, item_id, seek=500, chunks=2)
        assert len(data) == 2 * CHUNK_BYTES, item_id


async def test_soundscapes_come_in_every_style_as_versions(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Each of the five styles is a version of the soundscape of the same day."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await _all_albums(provider)
    album = next(a for a in albums if a.item_id.startswith("kitchen|"))
    tracks = await provider.get_album_tracks(album.item_id)
    assert len(tracks) == len(STYLES) == 5
    assert list(STYLES) == ["ambient", "berlin_school", "lofi", "ritual", "glitch"]
    assert [track.version for track in tracks] == [style.version for style in STYLES.values()]
    assert [track.item_id.rsplit("|", 1)[1] for track in tracks] == list(STYLES)
    # a title of its own for each style
    assert len({track.name for track in tracks}) == len(STYLES)
    assert {_target(track.item_id) for track in tracks} == {"_scape"}
    other = await provider.get_track(f"{album.item_id}|_scape|lofi")
    assert other.version == STYLES["lofi"].version


async def test_one_style_offered_shows_no_version(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a single style there is nothing to tell apart; the default style is always offered."""
    provider, _ = await _setup(mass, fake_hass)
    real = provider.get_config_value
    values = {"styles": ["lofi"], "default_style": "ambient"}
    monkeypatch.setattr(
        provider, "get_config_value", lambda key, default=None: values.get(key, real(key, default))
    )
    albums = await _all_albums(provider)
    album = next(a for a in albums if a.item_id.startswith("kitchen|"))
    tracks = await provider.get_album_tracks(album.item_id)
    scapes = [track for track in tracks if _target(track.item_id) == "_scape"]
    assert [track.item_id.rsplit("|", 1)[1] for track in scapes] == ["ambient", "lofi"]
    values["styles"] = ["ambient"]
    tracks = await provider.get_album_tracks(album.item_id)
    assert {track.version for track in tracks} == {""}


async def test_the_style_in_the_id_chooses_the_sound(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The same day sounds different per style; an id without a style is Ambient, as before."""
    provider, _ = await _setup(mass, fake_hass, FULL)
    albums = await _all_albums(provider)
    past = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    base = f"{past.item_id}|_scape"
    legacy = await _stream(provider, base, seek=300, chunks=2)
    ambient = await _stream(provider, f"{base}|ambient", seek=300, chunks=2)
    other = await _stream(provider, f"{base}|lofi", seek=300, chunks=2)
    again = await _stream(provider, f"{base}|lofi", seek=300, chunks=2)
    assert legacy == ambient
    assert other == again
    assert other != ambient
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details(f"{base}|vaporwave", MediaType.TRACK)


async def test_every_style_streams(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """Each style plays the soundscape: whole seconds of PCM."""
    provider, _ = await _setup(mass, fake_hass, FULL)
    albums = await _all_albums(provider)
    past = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    for style in STYLES:
        data = await _stream(provider, f"{past.item_id}|_scape|{style}", seek=420, chunks=2)
        assert len(data) == 2 * CHUNK_BYTES, style


async def test_a_soundscape_is_as_long_as_its_day_and_streams_the_same_each_time(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The length of a soundscape comes from its day (10 to 22 minutes); its audio repeats."""
    provider, _ = await _setup(mass, fake_hass, FULL)
    real = provider.get_config_value
    monkeypatch.setattr(
        provider,
        "get_config_value",
        lambda key, default=None: "light" if key == "render_quality" else real(key, default),
    )
    albums = await _all_albums(provider)
    past = [a for a in albums if a.item_id.startswith("kitchen|")][-1]
    for style in ("lofi", "ambient"):
        item_id = f"{past.item_id}|_scape|{style}"
        details = await provider.get_stream_details(item_id, MediaType.TRACK)
        assert 600 <= (details.duration or 0) <= 1320
        first = await _stream(provider, item_id, seek=200, chunks=2)
        second = await _stream(provider, item_id, seek=200, chunks=2)
        assert len(first) == 2 * CHUNK_BYTES
        a = np.frombuffer(first, dtype="<i2").astype(np.int32)
        b = np.frombuffer(second, dtype="<i2").astype(np.int32)
        assert np.abs(a - b).max() <= 2, (style, (a != b).mean())
        assert any(first)
    details = await provider.get_stream_details(f"{past.item_id}|_scape|lofi", MediaType.TRACK)
    assert (await provider.get_track(f"{past.item_id}|_scape|lofi")).duration == details.duration
    # a soundscape of today needs three hours of the day; before that it is not found
    assert MIN_SCAPE_SECONDS_OF_DAY == 3 * 3600
