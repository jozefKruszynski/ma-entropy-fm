"""Tests of the covers: drawn in the look of Music Assistant."""

import time
from datetime import date, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from hass_client.exceptions import BaseHassClientError
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import Album
from PIL import Image

from entropy_fm.constants import COVER_GENERATION, COVER_LIVE_MARK, WHOLE_HOUSE_ID
from entropy_fm.covers import (
    PALETTES,
    DayProfile,
    album_style,
    draw_title_on,
    procedural_cover,
    profile_of,
)
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series
from entropy_fm.titles import DayFacts
from music_assistant.mass import MusicAssistant
from tests.test_provider import _setup


def _day() -> list[Series]:
    """Return a day full of personal names, ids and numbers."""
    infos = [
        EntityInfo("sensor.annas_bedroom_power", "Anna's Heater", "marek_office", VoiceKind.DRONE),
        EntityInfo("sensor.siemens_dishwasher_operation_state", "Karl", "kitchen", VoiceKind.PHASE),
        EntityInfo("sensor.bilbo_oven_state", "Bilbo", "kitchen", VoiceKind.PHASE),
        EntityInfo("light.sarah_lamp", "Sarah's lamp in Wels", "vienna_flat", VoiceKind.CHORD),
        EntityInfo("media_player.jozef_sonos", "Jozef's Sonos", "kitchen", VoiceKind.MEDIA),
    ]
    times = [0.0, 20000.0, 40000.0, 61234.0, 70000.0]
    return [
        build_series(info, times, [0.0, 1.0 + index % 3, 0.0, 2.0, 0.0])
        for index, info in enumerate(infos)
    ]


def _cover(style: str = "berlin_school", seed: str = "kitchen|2026-11-17|berlin_school") -> bytes:
    return procedural_cover(profile_of(_day()), style, 2, "dorian", seed, "Kitchen, Tue 17 Nov")


def _picture(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


@pytest.mark.parametrize("style", list(PALETTES))
def test_a_drawn_cover_is_a_square_jpeg_in_the_look_of_music_assistant(style: str) -> None:
    """A 1024 pixel JPEG: a gradient, white glyph strokes cropped off the top-right, a title."""
    image = _picture(_cover(style, f"kitchen|2026-11-17|{style}"))
    assert image.format == "JPEG"
    assert image.size == (1024, 1024)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    # white line art in the upper and right part, and a calm darkened lower-left corner for the title
    upper_right = pixels[:600, 450:]
    lower_left = pixels[800:, :300]
    assert (upper_right.min(axis=2) > 235).mean() > 0.01
    assert lower_left.mean() < upper_right.mean()
    # a gradient of at least two colours: the corners differ a lot
    assert (
        abs(pixels[40, 40] - pixels[980, 980]).max() > 20
        or abs(pixels[40, 980] - pixels[980, 40]).max() > 20
    )
    # the title is white text in the lower-left corner
    assert (lower_left.min(axis=2) > 235).mean() > 0.003


def test_a_drawn_cover_is_the_same_every_time_and_days_and_styles_differ() -> None:
    """The same area, day and style draw the same bytes; another day or style draws another."""
    first = _cover()
    assert first == _cover()
    other_day = _cover(seed="kitchen|2026-11-18|berlin_school")
    other_style = _cover("ritual", "kitchen|2026-11-17|ritual")
    assert other_day != first
    assert other_style != first
    quiet = procedural_cover(DayProfile(), "ambient", 5, "aeolian", "x", "Kitchen")
    assert quiet != first
    base = np.asarray(_picture(_cover("lofi", "k|2026-11-10|lofi")).convert("L"), dtype=np.float32)
    nxt = np.asarray(_picture(_cover("lofi", "k|2026-11-11|lofi")).convert("L"), dtype=np.float32)
    assert np.abs(base - nxt).mean() > 3.0


def test_a_cover_is_cheap_and_small() -> None:
    """A 1024 pixel cover takes well under a second and a couple of hundred kilobytes."""
    _cover()
    started = time.process_time()
    data = _cover("ritual", "x|y|ritual")
    assert time.process_time() - started < 2.0
    assert len(data) < 250_000


def test_a_long_title_is_wrapped_and_set_lower_left() -> None:
    """The title is white on a shade in the lower-left corner, and a very long one still fits the cover."""
    plain = Image.new("RGB", (1024, 1024), (40, 60, 90))
    for title in ("Kitchen, Tue 17 Nov", "Whole House, " + "Dishwasher at Dusk " * 6):
        drawn = draw_title_on(plain, title)
        assert drawn.size == (1024, 1024)
        pixels = np.asarray(drawn, dtype=np.float32)
        assert (pixels[700:, :512].min(axis=2) > 235).mean() > 0.003
        assert np.array_equal(pixels[:300], np.asarray(plain, dtype=np.float32)[:300])


async def test_the_house_cover_shares_the_history_of_its_rooms(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A cover reads its day once, like a track; the house reads nothing of its own."""
    provider, _ = await _setup(mass, fake_hass)
    calls = fake_hass.hass.history_calls
    today = (await provider._now()).date()  # noqa: SLF001
    past = today - timedelta(days=2)

    def path_of(area_id: str, day: date) -> str:
        return f"cover|{area_id}|{day}|ambient|g{COVER_GENERATION}"

    for area_id in provider._house_rooms(await provider._catalog_all()):  # noqa: SLF001
        assert isinstance(await provider.resolve_image(path_of(area_id, past)), bytes)
    asked = len(calls)
    assert isinstance(await provider.resolve_image(path_of(WHOLE_HOUSE_ID, past)), bytes)
    assert len(calls) == asked
    assert isinstance(await provider.resolve_image(path_of("kitchen", today)), bytes)


async def test_a_cover_that_cannot_be_read_is_an_error_and_never_an_empty_cover(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server caches what it is given, so a day that Home Assistant did not answer for is not drawn empty."""
    provider, _ = await _setup(mass, fake_hass)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001

    async def broken(*_a: Any, **_k: Any) -> Any:
        raise BaseHassClientError("down")

    monkeypatch.setattr(provider, "_series_of", broken)
    with pytest.raises(ProviderUnavailableError):
        await provider.resolve_image(f"cover|kitchen|{past}|ambient|g{COVER_GENERATION}")


async def test_albums_and_tracks_carry_a_drawn_cover_that_the_provider_serves(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The default is a drawn JPEG: every album has a thumb that resolves to the same bytes."""
    provider, _ = await _setup(mass, fake_hass)
    albums = await provider.get_artist_albums("kitchen")
    album = next(a for a in albums if isinstance(a, Album) and a.item_id.startswith("kitchen|"))
    images = album.metadata.images
    assert images
    image = next(iter(images))
    assert not image.remotely_accessible
    cover = await provider.resolve_image(image.path)
    assert isinstance(cover, bytes)
    assert _picture(cover).size == (1024, 1024)
    assert await provider.resolve_image(image.path) == cover
    assert provider.accepts_stored_image_path(image.path)
    assert await provider.resolve_image("cover|kitchen|not-a-date|ambient") is None
    assert await provider.resolve_image("somewhere/else.png") is None
    tracks = await provider.get_album_tracks(album.item_id)
    assert all(track.metadata.images for track in tracks)


async def test_a_cover_path_of_a_room_or_day_that_is_not_offered_draws_nothing(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A crafted path must not make the provider read the history of any room or day."""
    provider, _ = await _setup(mass, fake_hass)
    calls = fake_hass.hass.history_calls
    asked = len(calls)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001
    assert await provider.resolve_image(f"cover|nowhere|{past}|ambient|g{COVER_GENERATION}") is None
    ancient = f"cover|kitchen|2001-01-01|ambient|g{COVER_GENERATION}"
    assert await provider.resolve_image(ancient) is None
    assert len(calls) == asked


async def test_a_day_that_is_not_over_has_a_cover_path_of_its_own(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Today's cover path is marked, so the finished cover gets a new path; both forms are served."""
    provider, _ = await _setup(mass, fake_hass)
    today = (await provider._now()).date()  # noqa: SLF001

    def path_of(day: date) -> str:
        images = provider._images("kitchen", day, "ambient")  # noqa: SLF001
        assert images
        return next(iter(images)).path

    live, past = path_of(today), path_of(today - timedelta(days=1))
    assert f"|{COVER_LIVE_MARK}|" in live
    assert f"|{COVER_LIVE_MARK}|" not in past
    assert isinstance(await provider.resolve_image(live), bytes)
    assert isinstance(await provider.resolve_image(past), bytes)
    assert await provider.resolve_image(live.replace(COVER_LIVE_MARK, "later")) is None


# ---------------------------------------------------------------- redraw covers


async def test_there_is_no_cover_setting_and_only_current_paths_are_served(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """There is no image engine setting, and paths without the generation or the figure are not served."""
    provider, _ = await _setup(mass, fake_hass)
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert not {
        "covers",
        "cover_engine",
        "cover_style",
        "cover_prompt",
        "cover_title",
        "redraw_covers",
    } & set(entries)
    day = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001
    assert await provider.resolve_image(f"cover|kitchen|{day}|ambient|g0")
    assert await provider.resolve_image(f"cover|kitchen|{day}|ambient") is None
    assert await provider.resolve_image("artist|kitchen|thumb|g0") is None


STYLES = list(PALETTES)


@pytest.mark.parametrize(
    ("facts", "style"),
    [
        (DayFacts(0.1, 14, "door", 3), "ambient"),
        (DayFacts(0.9, 14, "door", 30), "glitch"),
        (DayFacts(0.5, 23, "door", 10), "ritual"),
        (DayFacts(0.5, 3, "door", 10), "ritual"),
        (DayFacts(0.5, 19, "tv", 10), "lofi"),
        (DayFacts(0.5, 11, "door", 10), "berlin_school"),
    ],
)
def test_the_cover_of_an_album_follows_the_character_of_the_day(
    facts: DayFacts, style: str
) -> None:
    """Quiet, busy, night, evening and daytime days each get the glyph of another style."""
    assert album_style(facts, STYLES) == style


def test_the_cover_of_an_album_is_always_an_offered_style() -> None:
    """A style that is not offered is replaced by the first offered one, as is a day without facts."""
    assert album_style(DayFacts(0.9, 14, "door", 30), ["lofi", "ambient"]) == "lofi"
    assert album_style(DayFacts(0.1, 14, "door", 3), ["lofi", "ambient"]) == "ambient"
    assert album_style(None, ["lofi", "ambient"]) == "lofi"
    assert album_style(DayFacts(), ["ritual"]) == "ritual"
