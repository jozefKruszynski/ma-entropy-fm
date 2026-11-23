"""Tests of the titles and the descriptions of tracks and albums, and of what the listings do with them."""

import itertools
import re
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import Album, Track

from entropy_fm.styles import STYLES
from entropy_fm.timeline import Series, VoiceKind
from entropy_fm.titles import (
    BANKS,
    FORBIDDEN_TITLES,
    MAX_TITLE,
    DayFacts,
    album_review,
    day_line,
    facts_of,
    make_title,
    track_description,
)
from music_assistant.mass import MusicAssistant
from tests.test_library_provider import _album
from tests.test_provider import _setup

HOUR = 3600.0
DAY = date(2026, 11, 17)
BUSY = DayFacts(0.8, 15, "dishwasher", 3, "door", 14 * 60 + 12, "warm", 1)
STYLE_KEYS = list(STYLES)


def _series(
    entity_id: str,
    name: str,
    kind: VoiceKind,
    points: list[tuple[float, float]],
    device_class: str = "",
    start: float = 0.0,
) -> Series:
    """Return a series that steps through (hour, value) points."""
    times = np.array([0.0] + [hour * HOUR for hour, _ in points])
    values = np.array([start] + [value for _, value in points])
    peak = float(values.max()) or 1.0
    return Series(entity_id, name, kind, times, values, values / peak, device_class=device_class)


def _house() -> list[Series]:
    """Return a day with a dishwasher at 20:30, a door at 03:12 and names that must never surface."""
    return [
        _series(
            "sensor.jozef_dishwasher_power",
            "Jozef Dishwasher",
            VoiceKind.DISHWASHER,
            [(20.5, 2000), (21.0, 0), (22.0, 1800), (22.5, 0)],
        ),
        _series(
            "binary_sensor.sarahs_back_door",
            "Sarah's door",
            VoiceKind.PLUCK,
            [(3.2, 1), (3.21, 0), (9.0, 1), (9.01, 0)],
            device_class="door",
        ),
        _series(
            "sensor.kitchen_temperature",
            "Temperature",
            VoiceKind.TEMPERATURE,
            [(6, 25.0), (12, 26.0)],
            start=25.0,
        ),
    ]


def test_the_same_inputs_give_the_same_title() -> None:
    """A title is deterministic per room, day and style, in either language."""
    for lang, style in itertools.product(BANKS, STYLE_KEYS):
        first = make_title(BUSY, style, "kitchen|2026-11-17", "Kitchen", lang)
        assert first == make_title(BUSY, style, "kitchen|2026-11-17", "Kitchen", lang)
    titles = {
        make_title(BUSY, "ambient", f"kitchen|2026-11-{d:02d}", "Kitchen") for d in range(1, 29)
    }
    assert len(titles) > 3


def test_each_style_has_a_title_of_its_own() -> None:
    """The six styles of one room and day are told apart by their titles, in English and in Dutch."""
    for lang in BANKS:
        titles = [
            make_title(BUSY, style, "kitchen|2026-11-17", "Kitchen", lang) for style in STYLE_KEYS
        ]
        assert len(set(titles)) == len(STYLE_KEYS), titles
    assert make_title(BUSY, "glitch", "x", "Kitchen") != make_title(
        BUSY, "glitch", "x", "Kitchen", "nl"
    )


def test_a_day_without_data_gets_a_plain_title_of_the_style() -> None:
    """Without facts, or with a day that has nothing in it, the title is generic and never fails."""
    for lang, style in itertools.product(BANKS, STYLE_KEYS):
        plain = BANKS[lang].styles[style].plain
        assert make_title(None, style, "kitchen|2026-11-17", "Kitchen", lang) in plain
        assert make_title(DayFacts(), style, "kitchen|2026-11-17", "Kitchen", lang) in plain
    # an unknown style or language still gives a title
    assert make_title(BUSY, "polka", "x", "Kitchen", "xx")
    assert make_title(None, "polka", "x")


def test_titles_are_short_and_say_nothing_private() -> None:
    """Whatever the facts, a title is at most 40 characters, has no id, no real song title and no name."""
    kinds = [
        "",
        "dishwasher",
        "laundry",
        "hob",
        "door",
        "window",
        "motion",
        "light",
        "media",
        "blinds",
    ]
    seen: set[str] = set()
    for lang, style, kind, hour, intensity, climate in itertools.product(
        BANKS, STYLE_KEYS, kinds, (0, 6, 12, 15, 18, 21), (0.1, 0.5, 0.9), ("", "warm", "cool")
    ):
        facts = DayFacts(intensity, hour, kind, 7, kind, hour * 60 + 5, climate, hour % 7)
        for room in ("Kitchen", "The Very Long Name of a Living Room Upstairs", "House"):
            for seed in ("a|2026-11-17", "b|2026-11-18"):
                title = make_title(facts, style, seed, room, lang)
                seen.add(title)
                assert 0 < len(title) <= MAX_TITLE, title
                assert title.lower() not in FORBIDDEN_TITLES, title
                assert not re.search(r"[._|]|sensor|jozef|sarah", title, re.IGNORECASE), title
    assert len(seen) > 200


def test_every_word_bank_title_avoids_the_titles_of_the_inspiring_music() -> None:
    """The plain titles and the single words of the banks are not titles of the records that inspired the styles."""
    for words in BANKS.values():
        for style in words.styles.values():
            for title in style.plain:
                assert title.lower() not in FORBIDDEN_TITLES
            for word in (*style.adjectives, *style.nouns):
                assert word.lower() not in FORBIDDEN_TITLES


def test_the_facts_of_a_day_hold_kinds_and_times_and_never_names() -> None:
    """The dishwasher and the door at night are what the day was about; no entity id or name is kept."""
    facts = facts_of(_house(), DAY)
    assert facts.has_data
    assert facts.thing == "dishwasher"
    assert facts.event == "door"
    assert facts.event_minute == 3 * 60 + 12
    assert facts.climate == "warm"
    assert facts.weekday == DAY.weekday() == 1
    assert not facts.partial
    assert "jozef" not in repr(facts).lower()
    assert "sarah" not in repr(facts).lower()
    assert facts.key == facts_of(_house(), DAY).key
    for lang, style in itertools.product(BANKS, STYLE_KEYS):
        title = make_title(facts, style, "kitchen|2026-11-17", "Kitchen", lang)
        assert not re.search(r"jozef|sarah|sensor|binary", title, re.IGNORECASE), title


def test_a_day_with_nothing_but_quiet_voices_has_no_facts() -> None:
    """Only a temperature (or nothing) gives a day without data, which gets a plain title."""
    temperature = _series("sensor.t", "T", VoiceKind.TEMPERATURE, [(6, 21.0), (12, 22.0)])
    assert not facts_of([temperature], DAY).has_data
    assert not facts_of([], DAY).has_data
    assert facts_of([temperature], DAY, covered=4 * HOUR).partial


def test_every_style_has_a_description_that_tells_its_history() -> None:
    """A description names the inspiration, how the house drives the style and the day, in both languages."""
    needles = {
        "ambient": "Eno",
        "berlin_school": "Schulze",
        "lofi": "vinyl",
        "ritual": "Geinoh",
        "glitch": "KOAN",
    }
    assert set(needles) == set(STYLES)
    seen: set[str] = set()
    for lang in BANKS:
        for style in STYLE_KEYS:
            text = track_description(style, BUSY, lang)
            seen.add(text)
            assert 3 <= len(re.findall(r"[.!?](?:\s|$)", text)) <= 6, text
            assert text.endswith(day_line(BUSY, lang))
            assert BANKS[lang].styles[style].history in text
            assert "—" not in text
    assert len(seen) == 2 * len(STYLE_KEYS)
    for style, needle in needles.items():
        assert needle in track_description(style, BUSY)
        assert needle in track_description(style, BUSY, "nl")
    assert "Moondawn" in track_description("berlin_school", None)
    assert "tape loops" in track_description("ambient", None)


def test_the_line_about_the_day() -> None:
    """The line says how the day was, when it was busiest and what stood out, and copes with thin days."""
    assert day_line(BUSY) == "A busy afternoon; a door at 14:12."
    assert day_line(BUSY, "nl") == "Een drukke namiddag; een deur om 14:12."
    weekend = DayFacts(0.1, 9, "hob", 3, "hob", 7 * 60 + 40, "", 6, partial=True)
    assert day_line(weekend) == "So far, a quiet weekend morning; the hob at 07:40."
    assert "not much" in day_line(None)
    assert "weinig" in day_line(DayFacts(), "nl")


def test_the_review_of_an_album_names_the_room_the_date_and_the_kind_of_day() -> None:
    """The review is short, says the room and the day, and the kind of day when it is known."""
    review = album_review("Kitchen", DAY, BUSY)
    assert review.startswith("Kitchen on Tuesday 2026-11-17")
    assert "busy day" in review
    assert len(review) < 200
    assert "busy" not in album_review("Kitchen", DAY, None)
    assert "Keuken op dinsdag 2026-11-17" in album_review("Keuken", DAY, BUSY, "nl")


async def test_the_tracks_of_an_album_have_titles_descriptions_and_the_style_as_version(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """Each style of a day is a track with its own title; the style is the version, shown in brackets."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 3)
    tracks = await provider.get_album_tracks(album.item_id)
    assert [t.version for t in tracks] == [s.version for s in STYLES.values()]
    assert [t.version for t in tracks][-1] == "Cinematic Glitch"
    names = [t.name for t in tracks]
    assert len(set(names)) == len(STYLES), names
    assert not any("Soundscape" in name or "mix" in name for name in names)
    day = date.fromisoformat(album.item_id.split("|")[1])
    facts = await provider._facts("kitchen", day)  # noqa: SLF001
    for track, style in zip(tracks, STYLE_KEYS, strict=True):
        assert track.metadata.description == track_description(style, facts)
    album_of = (await provider.get_track(tracks[0].item_id)).album
    review = album_of.metadata.description if isinstance(album_of, Album) else None
    assert review is not None
    assert review.startswith("Kitchen on ")
    # the same on every route
    assert [t.name for t in await provider.get_album_tracks(album.item_id)] == names
    assert (await provider.get_track(tracks[2].item_id)).name == names[2]
    assert (await provider.get_album(album.item_id)).metadata.description == review
    # the house has the same, with a word for itself
    house = await _album(provider, "_house", 3)
    assert len({t.name for t in await provider.get_album_tracks(house.item_id)}) == len(STYLES)


async def test_the_titles_follow_the_day_and_the_listings_agree(
    mass: MusicAssistant, fake_hass: MagicMock
) -> None:
    """The library lists the same title that the album and the track have; another room has other titles."""
    provider, _ = await _setup(mass, fake_hass)
    cold = [t async for t in provider.get_library_tracks()]
    assert cold
    # the library reads nothing: until a day has been read (by opening it) its tracks have the plain titles
    for track in cold[:6]:
        assert track.album
        await provider.get_album_tracks(track.album.item_id)
    library = [t async for t in provider.get_library_tracks()]
    for track in library[:6]:
        again = await provider.get_track(track.item_id)
        assert again.name == track.name
        assert again.metadata.description == track.metadata.description
    kitchen = [t for t in library if t.item_id.startswith("kitchen|")]
    living = [t for t in library if t.item_id.startswith("living_room|")]
    assert len({t.name for t in kitchen}) > 1
    assert {t.name for t in kitchen} != {t.name for t in living}


async def test_a_title_is_kept_for_the_hour(mass: MusicAssistant, fake_hass: MagicMock) -> None:
    """Reading a day twice in an hour is one reading, so that the title of today does not flicker."""
    provider, fake = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 3)
    day = date.fromisoformat(album.item_id.split("|")[1])
    first = await provider._facts("kitchen", day)  # noqa: SLF001
    calls = len(fake.hass.history_calls)
    assert await provider._facts("kitchen", day) == first  # noqa: SLF001
    assert len(fake.hass.history_calls) == calls
    assert await provider._facts("kitchen", day, cached_only=True) == first  # noqa: SLF001
    # the reading lives in the cache of the server, not in the provider
    await mass.cache.clear()
    assert first is not None
    assert await provider._facts("kitchen", day, cached_only=True) is None  # noqa: SLF001


async def test_without_home_assistant_the_listing_still_works_with_plain_titles(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A history that cannot be read gives plain titles; the listing never fails."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 3)

    async def down(*_args: object, **_kwargs: object) -> list[Series]:
        raise ProviderUnavailableError("Home Assistant is away")

    monkeypatch.setattr(provider, "_series_of", down)
    # search and browsing only use what is kept: plain titles, and they never fail
    found = await provider.search("Kitchen", [MediaType.TRACK], limit=500)
    mine = [t for t in found.tracks if t.item_id.startswith(album.item_id)]
    assert len(mine) == len(STYLES)
    for track, style in zip(mine, STYLE_KEYS, strict=True):
        assert isinstance(track, Track)
        assert track.name in BANKS["en"].styles[style].plain
        assert (track.metadata.description or "").endswith(day_line(None))
    assert [a.name for a in await provider.get_artist_albums("kitchen")]


async def test_the_library_listing_fails_instead_of_listing_plain_titles(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the server stores for good is never made from a history that could not be read."""
    provider, _ = await _setup(mass, fake_hass)

    async def down(*_args: object, **_kwargs: object) -> None:
        raise ProviderUnavailableError("Home Assistant is away")

    with monkeypatch.context() as patched:
        patched.setattr(provider.history, "prefetch", down)
        with pytest.raises(ProviderUnavailableError):
            [t async for t in provider.get_library_tracks()]
        with pytest.raises(ProviderUnavailableError):
            [a async for a in provider.get_library_albums()]

    async def series_down(*_args: object, **_kwargs: object) -> list[Series]:
        raise ProviderUnavailableError("Home Assistant is away")

    monkeypatch.setattr(provider, "_series_of", series_down)
    await mass.cache.clear()
    with pytest.raises(ProviderUnavailableError):
        [t async for t in provider.get_library_tracks()]


async def test_titles_and_descriptions_are_dutch_when_the_server_is(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The language of the server picks the word banks; an unknown language gives English."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 3)
    english = await provider.get_album_tracks(album.item_id)
    monkeypatch.setattr(type(provider), "_language", property(lambda _self: "nl"))
    dutch = await provider.get_album_tracks(album.item_id)
    assert [t.name for t in dutch] != [t.name for t in english]
    text = dutch[0].metadata.description or ""
    assert "Brian Eno" in text
    assert "tapelussen" in text
    dutch_album = (await provider.get_track(dutch[0].item_id)).album
    review = dutch_album.metadata.description if isinstance(dutch_album, Album) else None
    assert review is not None
    assert " op " in review
    monkeypatch.setattr(type(provider), "_language", property(lambda _self: "fr"))
    assert [t.name for t in await provider.get_album_tracks(album.item_id)] == [
        t.name for t in english
    ]


async def test_what_the_server_would_store_is_never_made_from_an_unreadable_day(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The album, its tracks and a single track fail when the day cannot be read; browsing falls back."""
    provider, _ = await _setup(mass, fake_hass)
    album = await _album(provider, "kitchen", 3)
    await mass.cache.clear()

    async def down(*_args: object, **_kwargs: object) -> list[Series]:
        raise ProviderUnavailableError("Home Assistant is away")

    monkeypatch.setattr(provider, "_series_of", down)
    with pytest.raises(ProviderUnavailableError):
        await provider.get_album(album.item_id)
    with pytest.raises(ProviderUnavailableError):
        await provider.get_album_tracks(album.item_id)
    with pytest.raises(ProviderUnavailableError):
        await provider.get_track(f"{album.item_id}|_scape|lofi")
    assert await provider.get_artist_albums("kitchen")
