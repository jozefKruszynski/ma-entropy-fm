"""Tests of the covers: drawn in the look of Music Assistant."""

import time
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from hass_client.exceptions import BaseHassClientError
from music_assistant_models.errors import MediaNotFoundError, ProviderUnavailableError
from music_assistant_models.media_items import Album
from PIL import Image

from entropy_fm import constants as consts
from entropy_fm.character import NEUTRAL
from entropy_fm.constants import (
    COVER_CHARACTER_OFF,
    COVER_CHARACTER_ON,
    COVER_GENERATION,
    COVER_LIVE_MARK,
    FACTS_TODAY_EXPIRATION,
    PAST_DAY_EXPIRATION,
    SCAPE_ID,
    WHOLE_HOUSE_ID,
)
from entropy_fm.covers import (
    ACCENT,
    BINS,
    OP_DISC,
    PALETTES,
    STRIP_LEFT,
    STRIP_RIGHT,
    SUN_LEFT,
    SUN_TRAVEL,
    CoverMood,
    DayProfile,
    _glyph_horizon,
    _strip_ops,
    _wave_ops,
    album_style,
    draw_title_on,
    procedural_cover,
    profile_of,
    wave_shape,
)
from entropy_fm.library import TrackRef, day_bounds
from entropy_fm.palette import DayTraits
from entropy_fm.timeline import DAY_SECONDS, EntityInfo, Series, VoiceKind, build_series
from entropy_fm.titles import DayFacts
from music_assistant.mass import MusicAssistant
from tests.test_provider import _setup

MID = CoverMood(0.5)
AMSTERDAM = ZoneInfo("Europe/Amsterdam")
# the days of Europe/Amsterdam that have 25 and 23 hours
LONG_DAY = date(2026, 10, 25)
SHORT_DAY = date(2026, 3, 29)


def _path(area_id: str, day: date | str, generation: int = COVER_GENERATION) -> str:
    """Return the cover path of a finished day with the day character on, as the albums name it."""
    return (
        f"cover|{area_id}|{day}|ambient|{COVER_CHARACTER_ON}|e{consts.ENGINE_VERSION}|g{generation}"
    )


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


def _cover(
    style: str = "berlin_school",
    seed: str = "kitchen|2026-11-17|berlin_school",
    mood: CoverMood | None = MID,
) -> bytes:
    return procedural_cover(
        profile_of(_day()), style, 2, "dorian", seed, "Kitchen, Tue 17 Nov", mood
    )


def _grey(data: bytes) -> np.ndarray:
    return np.asarray(_picture(data).convert("L"), dtype=np.float32)


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


def test_the_wave_gauge_is_a_flat_line_at_rest_and_a_big_tight_wave_when_busy() -> None:
    """The dots-to-wave grows monotonically with energy; flat at 0, many times taller and tighter at 1."""
    amplitudes = [wave_shape(e)[0] for e in (0.0, 0.5, 1.0)]
    frequencies = [wave_shape(e)[1] for e in (0.0, 0.5, 1.0)]
    assert amplitudes == sorted(amplitudes)
    assert frequencies == sorted(frequencies)
    assert amplitudes[0] < 2
    assert amplitudes[2] > 20 * amplitudes[0]
    assert frequencies[2] > 4 * frequencies[0]
    # the drawn wave: its points swing further in y, and cross the middle more often, when the energy is high
    swings = []
    for energy in (0.0, 1.0):
        _, wave = _wave_ops(energy)
        ys = np.array([y for _, y in wave[0][1]])
        crossings = int((np.diff(np.sign(ys - ys.mean())) != 0).sum())
        swings.append((float(ys.max() - ys.min()), crossings))
    assert swings[1][0] > 20 * swings[0][0]
    assert swings[1][1] > swings[0][1]


def _colour(data: bytes) -> tuple[float, float]:
    """Return the mean chroma and the mean brightness of the cover, leaving out the white of the glyph."""
    rgb = np.asarray(_picture(data), dtype=np.float32)
    ground = rgb.mean(axis=-1) < 200
    return (
        float((rgb.max(axis=-1) - rgb.min(axis=-1))[ground].mean()),
        float(rgb.max(axis=-1)[ground].mean()),
    )


@pytest.mark.parametrize("style", list(PALETTES))
def test_a_busy_cover_and_a_quiet_one_differ_clearly_and_each_is_the_same_every_time(
    style: str,
) -> None:
    """Busy is much more saturated and brighter, inked more and grainier than quiet; same energy, same bytes."""
    seed = f"kitchen|2026-11-17|{style}"
    quiet = _cover(style, seed, CoverMood(0.05))
    busy = _cover(style, seed, CoverMood(0.95))
    assert quiet == _cover(style, seed, CoverMood(0.05))
    assert busy == _cover(style, seed, CoverMood(0.95))
    assert np.abs(_grey(quiet) - _grey(busy)).mean() > 15.0
    (quiet_chroma, quiet_bright), (busy_chroma, busy_bright) = _colour(quiet), _colour(busy)
    assert busy_chroma > 2 * quiet_chroma
    assert busy_bright > 1.3 * quiet_bright
    # the glyph is white: a quiet one covers clearly less of the cover than a busy one
    top = slice(0, 880)
    assert (_grey(busy)[top] > 235).mean() > 2 * (_grey(quiet)[top] > 235).mean()


def test_the_energy_grows_steadily_from_quiet_to_busy() -> None:
    """Colour, brightness and the area of the glyph rise with every step of energy."""
    seed = "kitchen|2026-11-17|lofi"
    covers = [_cover("lofi", seed, CoverMood(e)) for e in (0.1, 0.5, 0.95)]
    chroma = [_colour(c)[0] for c in covers]
    brightness = [_colour(c)[1] for c in covers]
    assert chroma == sorted(chroma)
    assert brightness == sorted(brightness)
    ink = [float((_grey(c)[:880] > 235).mean()) for c in covers]
    assert ink == sorted(ink)


def test_a_busy_day_leaves_ghost_copies_of_the_glyph_and_a_quiet_one_none() -> None:
    """Above 0.6 energy the glyph is echoed; at 0.6 and below it is not."""
    ghosts = import_module("entropy_fm.covers")._ghosts  # noqa: SLF001
    glyph = Image.new("L", (256, 256), 0)
    glyph.paste(255, (100, 100, 140, 140))
    assert ghosts(glyph, (120, 120), 0.3, "s").max() == 0
    assert ghosts(glyph, (120, 120), 0.6, "s").max() == 0
    busy = ghosts(glyph, (120, 120), 0.95, "s")
    assert busy.max() > 0.1
    assert busy.max() < 0.5
    assert np.array_equal(busy, ghosts(glyph, (120, 120), 0.95, "s"))


def test_a_cover_without_a_mood_goes_by_the_activity_of_the_day() -> None:
    """With no character the cover takes the share of the day that was active, and still draws."""
    profile = profile_of(_day())
    plain = procedural_cover(profile, "lofi", 2, "dorian", "k|d|lofi", "Kitchen")
    assert _picture(plain).size == (1024, 1024)
    same = procedural_cover(
        profile, "lofi", 2, "dorian", "k|d|lofi", "Kitchen", CoverMood(profile.intensity)
    )
    assert plain == same


def test_the_strip_shows_the_day_round_the_clock_above_the_foot_of_the_cover() -> None:
    """A morning day has ink in the strip on the left and a evening day on the right; the title stays clear."""
    bins = np.zeros(96)
    bins[28:36] = 1.0
    morning = procedural_cover(
        DayProfile(bins), "ambient", 2, "dorian", "s", "Kitchen", CoverMood(0.5)
    )
    evening = procedural_cover(
        DayProfile(bins[::-1].copy()), "ambient", 2, "dorian", "s", "Kitchen", CoverMood(0.5)
    )
    strip = slice(966, 1002)

    def ink(data: bytes) -> tuple[float, float]:
        picture = _grey(data)
        base = _grey(
            procedural_cover(DayProfile(), "ambient", 2, "dorian", "s", "Kitchen", CoverMood(0.5))
        )
        diff = np.clip(picture - base, 0, None)[strip]
        return float(diff[:, :512].sum()), float(diff[:, 512:].sum())

    left, right = ink(morning)
    assert left > 5 * right
    left, right = ink(evening)
    assert right > 5 * left


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
        return _path(area_id, day)

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
        await provider.resolve_image(_path("kitchen", past))


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
    assert await provider.resolve_image(_path("nowhere", past)) is None
    assert await provider.resolve_image(_path("kitchen", "2001-01-01")) is None
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
    assert await provider.resolve_image(_path("kitchen", day, 0))
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


async def test_the_mood_of_a_cover_is_the_character_of_the_track_and_is_kept(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """A cover reads the character and traits of its day once, and is drawn from them."""
    provider, _ = await _setup(mass, fake_hass)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001
    first = await provider._cover_mood("kitchen", past, "ambient")  # noqa: SLF001
    assert first is not None
    character = first[1]
    assert 0.0 <= character.energy <= 1.0
    calls = len(fake_hass.hass.history_calls)
    assert await provider._cover_mood("kitchen", past, "ambient") == first  # noqa: SLF001
    assert len(fake_hass.hass.history_calls) == calls
    cover = await provider._cover("kitchen", past, "ambient", character=True)  # noqa: SLF001
    assert isinstance(cover, bytes)


def _sun_and_accent(profile: DayProfile) -> tuple[int, int | None]:
    """Return the bin of the sun of the horizon and the bin of the strip that is drawn in the accent."""
    rng = np.random.default_rng(0)
    sun = next(op for op in _glyph_horizon(profile, rng, 0.5) if op[0] == OP_DISC)
    sun_bin = round((sun[1] - SUN_LEFT) / SUN_TRAVEL * BINS)
    pitch = (STRIP_RIGHT - STRIP_LEFT) / (BINS - 1)
    accents = [
        round((op[1][0][0] - STRIP_LEFT) / pitch) for op in _strip_ops(profile) if op[3] == ACCENT
    ]
    assert len(accents) <= 1
    return sun_bin, accents[0] if accents else None


def test_the_sun_and_the_strip_accent_sit_on_the_busiest_quarter_of_an_hour() -> None:
    """Both mark the busiest bin, and a day with any activity has its accent; a still one has none."""
    bins = np.zeros(BINS)
    bins[[10, 40, 70]] = (0.3, 1.0, 0.6)
    assert _sun_and_accent(DayProfile(bins)) == (40, 40)
    faint = np.zeros(BINS)
    faint[5] = 0.01
    assert _sun_and_accent(DayProfile(faint))[1] == 5
    assert _sun_and_accent(DayProfile())[1] is None


@pytest.mark.parametrize("day", [LONG_DAY, SHORT_DAY])
def test_a_day_of_a_daylight_saving_change_fills_the_strip_round_its_own_clock(day: date) -> None:
    """Evenly spread events of a 25 or 23 hour day fill every bin alike: no spike at the end, no empty tail."""
    start, end = day_bounds(day, AMSTERDAM)
    length = end.timestamp() - start.timestamp()
    assert length != DAY_SECONDS
    info = EntityInfo("light.lamp", "Lamp", "kitchen", VoiceKind.CHORD)
    steps = BINS * 4
    times = [(index + 0.5) * length / steps for index in range(steps)]
    profile = profile_of(
        [build_series(info, times, [float(index % 2) for index in range(steps)])], length
    )
    assert profile.bins.min() > 0.5
    assert profile.bins[-1] <= profile.bins[: BINS - 1].max()


async def test_the_cover_has_the_key_and_mode_of_the_track_of_its_day(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same area, day and style give the cover and the track the same palette."""
    provider, _ = await _setup(mass, fake_hass)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001
    drawn: list[Any] = []
    real = _module_of(provider).derive_palette

    def spy(*args: Any) -> Any:
        drawn.append(real(*args))
        return drawn[-1]

    monkeypatch.setattr(_module_of(provider), "derive_palette", spy)
    await provider._cover("kitchen", past, "ambient", character=True)  # noqa: SLF001
    scape = await provider._scape_of(TrackRef("kitchen", past, SCAPE_ID, "ambient"))  # noqa: SLF001
    assert scape.palette is not None
    assert (drawn[0].root, drawn[0].mode) == (scape.palette.root, scape.palette.mode)


async def test_a_cover_without_the_day_character_goes_by_the_activity_and_reads_nothing(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the setting off the cover is drawn from the share of the day that was active, with no reading."""
    provider, _ = await _setup(mass, fake_hass)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001

    async def unread(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("read")

    monkeypatch.setattr(provider, "_cover_mood", unread)
    cover = await provider._cover("kitchen", past, "ambient", character=False)  # noqa: SLF001
    assert _picture(cover).size == (1024, 1024)


async def test_a_finished_day_that_cannot_be_read_fails_and_a_running_one_falls_back(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished day keeps its cover forever, so a failed reading must not leave a fallback under its path."""
    provider, _ = await _setup(mass, fake_hass)
    now = await provider._now()  # noqa: SLF001

    async def broken(*_args: Any, **_kwargs: Any) -> None:
        raise BaseHassClientError("away")

    async def missing(*_args: Any, **_kwargs: Any) -> None:
        raise MediaNotFoundError("gone")

    monkeypatch.setattr(provider, "_gather", broken)
    with pytest.raises(BaseHassClientError):
        await provider._cover_mood("kitchen", now.date() - timedelta(days=3), "ambient")  # noqa: SLF001
    monkeypatch.setattr(provider, "_gather", missing)
    with pytest.raises(MediaNotFoundError):
        await provider._cover_mood("kitchen", now.date() - timedelta(days=4), "ambient")  # noqa: SLF001
    monkeypatch.setattr(provider, "_gather", broken)
    monkeypatch.setattr(provider, "_day_is_running", lambda _day: True)
    assert await provider._cover_mood("kitchen", now.date() - timedelta(days=3), "ambient") is None  # noqa: SLF001


async def test_the_cover_path_carries_the_day_character_and_the_engine(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toggling the setting or a new engine gives a new path, so the server fetches the cover anew."""
    provider, _ = await _setup(mass, fake_hass)
    past = (await provider._now()).date() - timedelta(days=2)  # noqa: SLF001

    def path_of() -> str:
        return next(iter(provider._images("kitchen", past, "ambient"))).path  # noqa: SLF001

    on = path_of()
    assert f"|{COVER_CHARACTER_ON}|" in on
    real = provider.get_config_value
    monkeypatch.setattr(
        provider,
        "get_config_value",
        lambda key, default=None: False if key == consts.CONF_DAY_CHARACTER else real(key, default),
    )
    off = path_of()
    assert off == on.replace(f"|{COVER_CHARACTER_ON}|", f"|{COVER_CHARACTER_OFF}|")
    monkeypatch.setattr(_module_of(provider), "ENGINE_VERSION", consts.ENGINE_VERSION + 1)
    assert path_of() not in {on, off}
    assert isinstance(await provider.resolve_image(off), bytes)


@pytest.mark.parametrize("day", [LONG_DAY, SHORT_DAY])
async def test_a_finished_day_of_a_daylight_saving_change_is_kept_as_finished(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch, day: date
) -> None:
    """A finished 25 or 23 hour day is read once for good, not every hour like today."""
    provider, _ = await _setup(mass, fake_hass)
    provider._tz = AMSTERDAM  # noqa: SLF001
    later = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=3)
    monkeypatch.setattr(provider, "_now", _returning(later.astimezone(AMSTERDAM)))
    monkeypatch.setattr(_module_of(provider), "utc", lambda: later)
    monkeypatch.setattr(provider, "_gather", _returning(None))
    monkeypatch.setattr(provider, "_measure_day", lambda *_a: (DayTraits(), None, NEUTRAL))
    keys: list[tuple[Any, ...]] = []
    expirations: list[int] = []
    real_key = provider._reading_key  # noqa: SLF001

    def key_spy(*args: Any) -> str:
        keys.append(args)
        return real_key(*args)

    async def set_spy(_key: str, _value: Any, *, expiration: int, **_kwargs: Any) -> None:
        expirations.append(expiration)

    monkeypatch.setattr(provider, "_reading_key", key_spy)
    monkeypatch.setattr(provider.mass.cache, "set", set_spy)
    assert await provider._cover_mood("kitchen", day, "ambient") is not None  # noqa: SLF001
    assert keys[0][3] == ""
    assert expirations == [PAST_DAY_EXPIRATION]
    assert FACTS_TODAY_EXPIRATION not in expirations


def _returning(value: Any) -> Any:
    """Return an async function that returns a value, whatever it is called with."""

    async def give(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return give


def _module_of(provider: Any) -> Any:
    """Return the module that the loaded provider runs from, which is the one to patch."""
    return import_module(type(provider).__module__)
