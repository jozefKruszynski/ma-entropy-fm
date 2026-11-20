"""Tests of the images of the artists: the rooms, drawn in the look of the covers."""

from datetime import timedelta
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
from music_assistant_models.enums import ImageType, MediaType
from PIL import Image

from entropy_fm.const import CONF_COVERS, WHOLE_HOUSE_ID
from entropy_fm.covers import COVER_OFF
from entropy_fm.roomart import (
    HOURS,
    WIDE_HEIGHT,
    WIDE_WIDTH,
    RoomProfile,
    _figure,
    artist_fanart,
    artist_thumb,
    room_colours,
    seeded_profile,
    week_profile,
)
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series
from music_assistant.mass import MusicAssistant
from tests.test_library_provider import _quiet
from tests.test_provider import _collect, _setup

if TYPE_CHECKING:
    import pytest
    from music_assistant_models.media_items import Artist


def _picture(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def _day(
    hours: list[float], kind: VoiceKind = VoiceKind.PLUCK, entity: str = "binary_sensor.m"
) -> list[Series]:
    """Return one binary entity that is on for ten minutes at each of the hours."""
    times, values = [0.0], [0.0]
    for hour in hours:
        times += [hour * 3600.0, hour * 3600.0 + 600.0]
        values += [1.0, 0.0]
    return [build_series(EntityInfo(entity, entity, "x", kind), times, values)]


def _profile(hours: list[float]) -> RoomProfile:
    profile = week_profile([_day(hours)] * 3)
    assert profile is not None
    return profile


def test_a_week_of_activity_becomes_a_clock_of_presence_light_and_life() -> None:
    """The busy hours are the high ones; lights and appliances land in their own bands."""
    lights = _day([19, 20, 21], VoiceKind.CHORD, "light.l")
    media = _day([20], VoiceKind.MEDIA, "media_player.tv")
    motion = _day([7, 8, 18])
    profile = week_profile([[*motion, *lights, *media]] * 4)
    assert profile is not None
    assert profile.days == 4
    assert len(profile.presence) == len(profile.light) == len(profile.life) == HOURS
    assert int(np.argmax(profile.presence)) in {7, 8, 18}
    assert profile.presence[3] == 0
    assert int(np.argmax(profile.light)) == 20
    assert profile.light[7] == 0
    assert 20 in np.flatnonzero(profile.life)
    assert profile.presence.max() == 1.0


def test_a_week_without_history_makes_no_profile() -> None:
    """Nothing kept, or only things that do not play a part, gives None and the figure from the id."""
    assert week_profile([]) is None
    assert week_profile([[], []]) is None
    quiet = _day([], VoiceKind.PLUCK)
    assert week_profile([quiet, quiet]) is None


def test_the_figure_without_data_is_seeded_by_the_id_and_deterministic() -> None:
    """The same id always gives the same figure, another id another one; the profile has all three bands."""
    first = seeded_profile("hall")
    again = seeded_profile("hall")
    assert first.digest() == again.digest()
    assert first.digest() != seeded_profile("garage").digest()
    assert first.days == 0
    assert first.presence.max() > 0
    assert artist_thumb("hall", "Hall", first) == artist_thumb("hall", "Hall", again)
    assert artist_fanart("hall", first) == artist_fanart("hall", again)


def test_the_images_are_deterministic_and_follow_the_week_and_the_palette_the_id() -> None:
    """Same data, same bytes; rooms with other weeks look different; the palette is the id's alone."""
    morning, evening = _profile([6, 7, 8]), _profile([19, 20, 21, 22])
    first = artist_thumb("kitchen", "Kitchen", morning)
    assert first == artist_thumb("kitchen", "Kitchen", _profile([6, 7, 8]))
    assert artist_thumb("kitchen", "Kitchen", evening) != first
    assert artist_thumb("kitchen", "Kitchen", morning, title=False) != first
    assert artist_fanart("kitchen", morning) == artist_fanart("kitchen", morning)
    assert artist_fanart("kitchen", evening) != artist_fanart("kitchen", morning)
    assert room_colours("kitchen") == room_colours("kitchen")
    assert room_colours("kitchen") != room_colours("garage")
    assert morning.digest() != evening.digest()


def test_the_figure_is_in_white_on_the_gradient_and_the_whole_house_is_richer() -> None:
    """Both sizes are JPEG, the line art shows, and the house has more drawn than a room of the same week."""
    profile = _profile([6, 7, 8, 18, 19])
    thumb = _picture(artist_thumb("kitchen", "Kitchen", profile))
    wide = _picture(artist_fanart("kitchen", profile))
    assert (thumb.format, thumb.size) == ("JPEG", (1024, 1024))
    assert (wide.format, wide.size) == ("JPEG", (WIDE_WIDTH, WIDE_HEIGHT))
    assert np.asarray(thumb.convert("L")).max() > 240
    assert np.asarray(wide.convert("L")).max() > 240
    assert len(_figure(profile, rich=True)) > len(_figure(profile, rich=False))
    assert artist_thumb(WHOLE_HOUSE_ID, "Whole House", profile) != artist_thumb(
        "room", "x", profile
    )
    assert room_colours(WHOLE_HOUSE_ID) == room_colours(WHOLE_HOUSE_ID)


def test_no_glyph_tables_are_left() -> None:
    """The images are abstract: no kinds of rooms, no icons."""
    import entropy_fm.roomart as module  # noqa: PLC0415

    assert not hasattr(module, "room_kind")
    assert not hasattr(module, "GLYPHS")


def _images(artist: Artist) -> dict[ImageType, str]:
    assert artist.metadata.images
    return {image.type: image.path for image in artist.metadata.images}


async def test_artists_carry_a_thumb_and_a_backdrop_that_the_provider_serves(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Library, get_artist, search and the artist of an album all have both images; they resolve to JPEGs."""
    provider, _ = await _setup(mass, fake_hass)
    library = await _collect(provider.get_library_artists())
    assert [a.name for a in library] == ["Whole House", "Kitchen", "Living Room"]
    found: list[Any] = [*library, await provider.get_artist("kitchen")]
    search = await provider.search("kitchen", [MediaType.ARTIST])
    found.extend(search.artists)
    albums = await provider.get_artist_albums("kitchen")
    found.extend(
        artist for album in albums for artist in album.artists if hasattr(album, "artists")
    )
    assert found
    for artist in found:
        images = _images(artist)
        assert set(images) == {ImageType.THUMB, ImageType.FANART}
        assert all(provider.accepts_stored_image_path(path) for path in images.values())
    kitchen = _images(await provider.get_artist("kitchen"))
    thumb = await provider.resolve_image(kitchen[ImageType.THUMB])
    fanart = await provider.resolve_image(kitchen[ImageType.FANART])
    assert isinstance(thumb, bytes)
    assert isinstance(fanart, bytes)
    assert thumb.startswith(b"\xff\xd8\xff")
    assert _picture(thumb).size == (1024, 1024)
    assert _picture(fanart).size == (WIDE_WIDTH, WIDE_HEIGHT)
    # drawn once: the second time comes from the cache folder
    assert await provider.resolve_image(kitchen[ImageType.THUMB]) == thumb
    assert any(provider.covers.store.cache.glob("*.img"))
    house = _images(await provider.get_artist(WHOLE_HOUSE_ID))
    assert isinstance(await provider.resolve_image(house[ImageType.THUMB]), bytes)


async def test_a_room_is_drawn_from_its_kept_days_and_otherwise_from_its_id(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Home Assistant is not asked for the image: kept days make the profile, else the seeded figure."""
    provider, _ = await _setup(mass, fake_hass)
    catalog = await provider._catalog_all()  # noqa: SLF001
    assert (await provider._room_profile("kitchen", catalog)).days == 0  # noqa: SLF001
    kitchen = _images(await provider.get_artist("kitchen"))[ImageType.THUMB]
    seeded = await provider.resolve_image(kitchen)
    assert seeded == artist_thumb("kitchen", "Kitchen", seeded_profile("kitchen"))
    tz, now = await provider._timezone(), await provider._now()  # noqa: SLF001
    yesterday = now.astimezone(tz).date() - timedelta(days=1)
    await provider.history.day_series("kitchen", catalog.entities["kitchen"], yesterday, tz, now)
    profile = await provider._room_profile("kitchen", catalog)  # noqa: SLF001
    assert profile.days == 1
    again = await provider.resolve_image(kitchen)
    assert isinstance(again, bytes)


async def test_unknown_artist_images_are_refused(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A room that does not exist, a wrong kind and a missing generation are not served."""
    provider, _ = await _setup(mass, fake_hass)
    gen = f"g{provider._cover_generation}"  # noqa: SLF001
    assert await provider.resolve_image(f"artist|nowhere|thumb|{gen}") is None
    assert await provider.resolve_image(f"artist|kitchen|logo|{gen}") is None
    assert await provider.resolve_image("artist|kitchen|thumb") is None
    assert await provider.resolve_image("artist|kitchen|thumb|gx") is None
    assert await provider.resolve_image(f"artist|kitchen|thumb|{gen}") is not None


async def test_with_covers_off_there_are_no_artist_images_either(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is attached and nothing is served."""
    provider, _ = await _setup(mass, fake_hass)
    before = _images(await provider.get_artist("kitchen"))
    _quiet(provider, monkeypatch, **{CONF_COVERS: COVER_OFF})
    artists: list[Any] = await _collect(provider.get_library_artists())
    artists.append(await provider.get_artist("kitchen"))
    assert all(not artist.metadata.images for artist in artists)
    for path in before.values():
        assert await provider.resolve_image(path) is None
