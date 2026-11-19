"""Tests of the images of the artists: the rooms, drawn in the look of the covers."""

from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from music_assistant_models.enums import ImageType, MediaType
from PIL import Image

from entropy_fm.const import CONF_COVERS, WHOLE_HOUSE_ID
from entropy_fm.covers import COVER_OFF
from entropy_fm.roomart import (
    GENERIC,
    GLYPHS,
    HOUSE,
    ICON_TO_KIND,
    WIDE_HEIGHT,
    WIDE_WIDTH,
    artist_fanart,
    artist_thumb,
    room_colours,
    room_kind,
)
from music_assistant.mass import MusicAssistant
from tests import fake_hass as fake_module
from tests.test_library_provider import _quiet
from tests.test_provider import _collect, _setup

if TYPE_CHECKING:
    from music_assistant_models.media_items import Artist


def _picture(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


@pytest.mark.parametrize(
    ("icon", "kind"),
    [
        ("mdi:sofa", "living"),
        ("mdi:fridge", "kitchen"),
        ("mdi:bed", "bedroom"),
        ("mdi:shower", "bathroom"),
        ("mdi:desk", "office"),
        ("mdi:garage", "garage"),
        ("mdi:flower", "garden"),
        ("mdi:washing-machine", "laundry"),
        ("phu:sofa", "living"),
    ],
)
def test_the_icon_of_the_area_tells_the_kind_of_room_before_its_name(icon: str, kind: str) -> None:
    """An icon wins over a name that says something else."""
    assert room_kind("Something else", "x", icon) == kind
    assert room_kind("Garage", "garage", icon) == kind


def test_an_icon_that_is_not_known_falls_back_to_the_name() -> None:
    """An unknown icon does not hide the name."""
    assert room_kind("Kitchen", "kitchen", "mdi:rocket") == "kitchen"
    assert room_kind("Mystery", "mystery", "mdi:rocket") == GENERIC


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Kitchen", "kitchen"),
        ("Keuken", "kitchen"),
        ("Küche", "kitchen"),
        ("Living Room", "living"),
        ("Woonkamer", "living"),
        ("Wohnzimmer", "living"),
        ("Bedroom", "bedroom"),
        ("Slaapkamer", "bedroom"),
        ("Schlafzimmer", "bedroom"),
        ("Bathroom", "bathroom"),
        ("Badkamer", "bathroom"),
        ("Badezimmer", "bathroom"),
        ("Office", "office"),
        ("Kantoor", "office"),
        ("Büro", "office"),
        ("Garage", "garage"),
        ("Garden", "garden"),
        ("Tuin", "garden"),
        ("Garten", "garden"),
        ("Hallway", "hall"),
        ("Gang", "hall"),
        ("Flur", "hall"),
        ("Dining Room", "dining"),
        ("Eetkamer", "dining"),
        ("Esszimmer", "dining"),
        ("Laundry", "laundry"),
        ("Wasruimte", "laundry"),
        ("Waschküche", "laundry"),
        ("Bijkeuken", "laundry"),
        ("Kids Room", "kids"),
        ("Kinderkamer", "kids"),
        ("Kinderzimmer", "kids"),
        ("Guest Room", "guest"),
        ("Gastenkamer", "guest"),
        ("Gästezimmer", "guest"),
        ("Attic", "attic"),
        ("Zolder", "attic"),
        ("Dachboden", "attic"),
        ("Basement", "basement"),
        ("Kelder", "basement"),
        ("Keller", "basement"),
        ("Balcony", "balcony"),
        ("Balkon", "balcony"),
        ("Terrace", "terrace"),
        ("Terras", "terrace"),
        ("Terrasse", "terrace"),
        ("Stairs", "stairs"),
        ("Treppe", "stairs"),
        ("Toilet", "toilet"),
        ("WC", "toilet"),
        # who it is for does not hide what it is
        ("Guest Bedroom", "bedroom"),
        ("Kids Bathroom", "bathroom"),
        ("Gäste WC", "toilet"),
    ],
)
def test_the_kind_of_room_is_told_by_words_in_english_dutch_and_german(
    name: str, kind: str
) -> None:
    """Names are matched without accents and case, as words or the start of words."""
    assert room_kind(name, "") == kind


def test_the_id_is_read_when_the_name_says_nothing_and_unknown_rooms_are_generic() -> None:
    """The area id is the second chance; words do not match in the middle of a short word."""
    assert room_kind("Room 2", "living_room_2") == "living"
    assert room_kind("Hobby", "hobby") == GENERIC
    assert room_kind("Abadi", "abadi") == GENERIC
    assert room_kind("Wcheck", "wcheck") == GENERIC


def test_the_whole_house_is_a_house_whatever_it_is_called() -> None:
    """The house has its own glyph and amber palette."""
    assert room_kind("Whole House", WHOLE_HOUSE_ID) == HOUSE
    assert room_kind("Kitchen", WHOLE_HOUSE_ID, "mdi:fridge") == HOUSE
    assert room_colours(WHOLE_HOUSE_ID) == room_colours(WHOLE_HOUSE_ID)


def test_every_kind_of_room_has_a_glyph_and_every_icon_a_kind_that_has_one() -> None:
    """Nothing in the tables points at a glyph that does not exist."""
    assert set(ICON_TO_KIND.values()) <= set(GLYPHS)
    assert {kind for kind in GLYPHS if kind != GENERIC} <= {*ICON_TO_KIND.values()}


def test_the_images_of_a_room_are_deterministic_and_the_palette_follows_the_id() -> None:
    """The same room is drawn the same every time; another id has other colours; the kind changes the glyph."""
    first = artist_thumb("kitchen", "Kitchen", "kitchen")
    assert first == artist_thumb("kitchen", "Kitchen", "kitchen")
    assert room_colours("kitchen") == room_colours("kitchen")
    assert room_colours("kitchen") != room_colours("garage")
    assert artist_thumb("kitchen", "Kitchen", "garage") != first
    assert artist_thumb("kitchen", "Kitchen", "kitchen", title=False) != first
    assert artist_fanart("kitchen", "kitchen") == artist_fanart("kitchen", "kitchen")
    assert artist_fanart("garage", "kitchen") != artist_fanart("kitchen", "kitchen")


@pytest.mark.parametrize("kind", sorted(GLYPHS))
def test_every_glyph_is_drawn_as_a_square_jpeg_and_a_wide_one(kind: str) -> None:
    """The thumb is 1024 square, the backdrop is landscape, both JPEG, and the glyph shows."""
    thumb = _picture(artist_thumb(f"room_{kind}", kind.title(), kind))
    assert thumb.format == "JPEG"
    assert thumb.size == (1024, 1024)
    wide = _picture(artist_fanart(f"room_{kind}", kind))
    assert wide.format == "JPEG"
    assert wide.size == (WIDE_WIDTH, WIDE_HEIGHT)
    # white line art on a gradient: there are bright pixels
    assert np.asarray(thumb.convert("L")).max() > 240
    assert np.asarray(wide.convert("L")).max() > 240


@pytest.fixture
def icons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the living room an icon of a flower in the area registry."""
    monkeypatch.setitem(
        fake_module.AREAS,
        "living_room",
        {**fake_module.AREAS["living_room"], "icon": "mdi:flower"},
    )


def _images(artist: Artist) -> dict[ImageType, str]:
    assert artist.metadata.images
    return {image.type: image.path for image in artist.metadata.images}


@pytest.mark.usefixtures("icons")
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


@pytest.mark.usefixtures("icons")
async def test_the_icon_of_the_area_is_read_from_the_registry_and_picks_the_glyph(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A living room with the icon of a flower is drawn as a garden, a kitchen without icon as a kitchen."""
    provider, _ = await _setup(mass, fake_hass)
    assert provider.hass is not None
    assert await provider.hass.area_icons() == {"living_room": "mdi:flower"}
    room = _images(await provider.get_artist("living_room"))[ImageType.THUMB]
    assert await provider.resolve_image(room) == artist_thumb(
        "living_room", "Living Room", "garden"
    )
    kitchen = _images(await provider.get_artist("kitchen"))[ImageType.THUMB]
    assert await provider.resolve_image(kitchen) == artist_thumb("kitchen", "Kitchen", "kitchen")


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
