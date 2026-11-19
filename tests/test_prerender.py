"""Tests of the pre-render of the last day: which tracks, when, and that it stays out of the way."""

import asyncio
import shutil
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from entropy_fm import EntropyFMProvider, prerender
from entropy_fm.const import CONF_AUDIO_CACHE_MB, CONF_KEEP_ROOMS, CONF_PRERENDER, CONF_STYLES
from entropy_fm.library import TrackRef, album_id, track_id
from music_assistant.mass import MusicAssistant
from tests.test_library_provider import _quiet
from tests.test_provider import _setup

DAY = date(2026, 11, 18)
ROOMS = ["_house", "kitchen", "living_room"]
STYLES = ["ambient", "lofi", "glitch"]


def _choose(mode: str, **kwargs: Any) -> list[tuple[str, str]]:
    found = prerender.choose_tracks(
        mode,
        DAY,
        rooms=kwargs.pop("rooms", ROOMS),
        pinned_rooms=kwargs.pop("pinned_rooms", []),
        offered=kwargs.pop("offered", STYLES),
        default_style=kwargs.pop("default_style", "ambient"),
        **kwargs,
    )
    assert all(ref.day == DAY and ref.target == "_scape" for ref in found)
    return [(ref.area_id, ref.style) for ref in found]


def test_nothing_is_chosen_when_it_is_off_or_nothing_is_pinned() -> None:
    """Off renders nothing, and so does pinned rooms when no room is pinned."""
    assert _choose("off", pinned_rooms=["kitchen"]) == []
    assert _choose("pinned") == []
    assert _choose("nonsense", pinned_rooms=["kitchen"]) == []


def test_pinned_rooms_get_every_offered_style_with_the_default_one_first() -> None:
    """A kept room is rendered in the default style first and then in each style that is offered."""
    assert _choose("pinned", pinned_rooms=["kitchen"]) == [
        ("kitchen", "ambient"),
        ("kitchen", "lofi"),
        ("kitchen", "glitch"),
    ]
    # a room that is not there any more is not asked for
    assert _choose("pinned", pinned_rooms=["attic"]) == []


def test_all_rooms_get_the_default_style_and_pinned_rooms_all_of_them() -> None:
    """With all rooms, the whole house and every room come in the default style before any other style."""
    chosen = _choose("all", pinned_rooms=["kitchen"], offered=["lofi", "ambient"])
    assert chosen[:3] == [("_house", "ambient"), ("kitchen", "ambient"), ("living_room", "ambient")]
    assert chosen[3:] == [("kitchen", "lofi")]


def test_the_default_style_is_rendered_even_when_it_is_not_in_the_offered_list() -> None:
    """The library lists the default style whatever is offered, so it is rendered."""
    assert _choose(
        "pinned", pinned_rooms=["kitchen"], offered=["lofi"], default_style="ritual"
    ) == [
        ("kitchen", "ritual"),
        ("kitchen", "lofi"),
    ]


def test_what_was_added_to_the_library_for_that_day_is_pinned_too() -> None:
    """An album added for the day gets every style, a track only its own, and other days are ignored."""
    chosen = _choose(
        "pinned",
        added_albums=[album_id("living_room", DAY), album_id("kitchen", DAY - timedelta(days=1))],
        added_tracks=[
            track_id("kitchen", DAY, style="glitch"),
            track_id("kitchen", DAY - timedelta(days=2), style="lofi"),
            "garbage",
        ],
    )
    assert chosen == [
        ("living_room", "ambient"),
        ("kitchen", "glitch"),
        ("living_room", "lofi"),
        ("living_room", "glitch"),
    ]


def test_the_run_is_shortly_after_the_next_midnight_of_the_house() -> None:
    """The wait is to the next local midnight plus the delay, also on a night of a clock change."""
    vienna = ZoneInfo("Europe/Vienna")
    now = datetime(2026, 11, 18, 23, 0, tzinfo=vienna)
    assert prerender.seconds_until_run(now, 600.0) == 3600.0 + 600.0
    just_after = datetime(2026, 11, 19, 0, 0, 5, tzinfo=vienna)
    assert prerender.seconds_until_run(just_after, 0.0) == pytest.approx(86395.0)
    # the night of 25 October 2026 has 25 hours: from 01:00 the first time there are 24 hours to midnight
    before = datetime(2026, 10, 24, 23, 0, tzinfo=vienna)
    assert prerender.seconds_until_run(before, 0.0) == 3600.0
    assert prerender.seconds_until_run(datetime(2026, 10, 25, 1, 0, tzinfo=vienna), 0.0) == 86400.0


def test_the_cache_keeps_a_fifth_free_for_what_is_played() -> None:
    """A track fits while the cache would stay under 80 percent of its size."""
    limit = 500_000_000
    seconds = 1000.0
    assert prerender.fits(0, limit, seconds)
    assert prerender.fits(380_000_000, limit, seconds)
    assert not prerender.fits(390_000_000, limit, seconds)
    assert not prerender.fits(0, 0, seconds)


# ---------------------------------------------------------------- the provider


def _fake_pcm(played: list[TrackRef], chunks: int = 3) -> Callable[..., Iterator[bytes]]:
    """Return a stand-in for `render_pcm` that notes which track it was asked for and plays a few seconds."""

    def render(_score: object, _quality: object, seed: str, _seek: int = 0) -> Iterator[bytes]:
        area, day, style = seed.split("|")
        played.append(TrackRef(area, date.fromisoformat(day), "_scape", style))
        silence = bytes(44100 * 4)
        for _ in range(chunks):
            yield silence

    return render


async def _prepared(
    mass: MusicAssistant,
    fake_hass: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    **values: object,
) -> tuple[EntropyFMProvider, list[TrackRef]]:
    """Set up the provider with a fake renderer, and the settings of a test."""
    provider, _ = await _setup(mass, fake_hass)
    played: list[TrackRef] = []
    module = sys.modules[type(provider).__module__]
    monkeypatch.setattr(module, "render_pcm", _fake_pcm(played))

    async def no_score(_ref: TrackRef) -> None:
        return None

    monkeypatch.setattr(provider, "_build_score", no_score)
    monkeypatch.setattr(prerender, "YIELD_SECONDS", 0.0)
    monkeypatch.setattr(prerender, "POLL_SECONDS", 0.02)
    settings: dict[str, object] = {"day_character": False, **values}
    _quiet(provider, monkeypatch, **settings)
    provider.audio_cache.limit = int(settings.get(CONF_AUDIO_CACHE_MB, 500)) * 1_000_000  # type: ignore[call-overload]
    return provider, played


async def _yesterday(provider: EntropyFMProvider) -> date:
    return (await provider._now()).date() - timedelta(days=1)  # noqa: SLF001


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_the_last_day_of_pinned_rooms_is_rendered_into_the_cache_once(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every offered style of a kept room is rendered; a second run skips what the cache has."""
    provider, played = await _prepared(
        mass,
        fake_hass,
        monkeypatch,
        **{
            CONF_PRERENDER: "pinned",
            CONF_KEEP_ROOMS: ["kitchen"],
            CONF_STYLES: ["ambient", "lofi"],
        },
    )
    day = await _yesterday(provider)
    assert await provider._prerender_yesterday() == 2  # noqa: SLF001
    assert played == [TrackRef("kitchen", day, "_scape", s) for s in ("ambient", "lofi")]
    today = (await provider._now()).date()  # noqa: SLF001
    for ref in played:
        key = provider._audio_key(ref, today)  # noqa: SLF001
        assert key is not None
        assert provider.audio_cache.path(key).is_file()
    # nothing is left half written, and a second night renders nothing that is cached
    assert not list(provider.audio_cache.root.glob("*.part"))
    assert await provider._prerender_yesterday() == 0  # noqa: SLF001
    assert len(played) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_all_rooms_renders_the_default_style_of_every_room_and_skips_the_cached(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole house and each room come in the default style; a track that is cached already is skipped."""
    provider, played = await _prepared(
        mass, fake_hass, monkeypatch, **{CONF_PRERENDER: "all", CONF_STYLES: ["ambient", "lofi"]}
    )
    day = await _yesterday(provider)
    today = (await provider._now()).date()  # noqa: SLF001
    rooms = set(provider._artist_names(await provider._catalog_all()))  # noqa: SLF001
    cached = TrackRef("kitchen", day, "_scape", "ambient")
    key = provider._audio_key(cached, today)  # noqa: SLF001
    assert key is not None
    recorder = provider.audio_cache.recorder(key)
    assert recorder is not None
    await recorder.feed(bytes(44100 * 4 * 2))
    assert await recorder.commit()
    assert await provider._prerender_yesterday() == len(rooms) - 1  # noqa: SLF001
    assert {ref.area_id for ref in played} == rooms - {"kitchen"}
    assert {ref.style for ref in played} == {"ambient"}


async def test_nothing_is_rendered_when_it_is_off_or_the_cache_is_off(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off does nothing, and so does a cache of 0 megabytes."""
    provider, played = await _prepared(
        mass, fake_hass, monkeypatch, **{CONF_PRERENDER: "off", CONF_KEEP_ROOMS: ["kitchen"]}
    )
    assert await provider._prerender_yesterday() == 0  # noqa: SLF001
    _quiet(provider, monkeypatch, **{CONF_PRERENDER: "all", CONF_KEEP_ROOMS: ["kitchen"]})
    provider.audio_cache.limit = 0
    assert not provider.audio_cache.enabled
    assert await provider._prerender_yesterday() == 0  # noqa: SLF001
    assert played == []


async def test_it_stops_when_the_cache_has_no_room_instead_of_making_room(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache that is too small for a track is left alone: nothing is rendered and nothing is evicted."""
    provider, played = await _prepared(
        mass, fake_hass, monkeypatch, **{CONF_PRERENDER: "all", CONF_AUDIO_CACHE_MB: 1}
    )
    provider.audio_cache.root.mkdir(parents=True, exist_ok=True)
    old = provider.audio_cache.root / ("a" * 40 + ".opus")
    old.write_bytes(b"x" * 1000)
    assert await provider._prerender_yesterday() == 0  # noqa: SLF001
    assert played == []
    assert old.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_it_waits_while_a_track_is_playing_and_goes_on_afterwards(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a live stream the pre-render renders nothing; when the stream ends it carries on."""
    provider, played = await _prepared(
        mass, fake_hass, monkeypatch, **{CONF_PRERENDER: "pinned", CONF_KEEP_ROOMS: ["kitchen"]}
    )
    provider._streams = 1  # noqa: SLF001
    task = asyncio.create_task(provider._prerender_yesterday())  # noqa: SLF001
    await asyncio.sleep(0.5)
    assert not task.done()
    assert played == []
    provider._streams = 0  # noqa: SLF001
    assert await asyncio.wait_for(task, 60) > 0
    assert played


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_it_pauses_in_the_middle_of_a_track_and_a_live_stream_counts(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream that starts while a track is being rendered holds the next second back; ending counts down."""
    provider, played = await _prepared(
        mass,
        fake_hass,
        monkeypatch,
        **{CONF_PRERENDER: "pinned", CONF_KEEP_ROOMS: ["kitchen"], CONF_STYLES: ["ambient"]},
    )
    chunks_seen: list[int] = []
    module = sys.modules[type(provider).__module__]

    def slow(_score: object, _quality: object, _seed: str, _seek: int = 0) -> Iterator[bytes]:
        for index in range(6):
            chunks_seen.append(index)
            if index == 1:
                provider._streams += 1  # noqa: SLF001
            yield bytes(44100 * 4)

    monkeypatch.setattr(module, "render_pcm", slow)
    task = asyncio.create_task(provider._prerender_yesterday())  # noqa: SLF001
    await asyncio.sleep(0.6)
    held = len(chunks_seen)
    assert held == 2
    await asyncio.sleep(0.3)
    assert len(chunks_seen) == held
    provider._streams = 0  # noqa: SLF001
    await asyncio.wait_for(task, 60)
    assert len(chunks_seen) >= 6
    assert played == []


async def test_a_live_stream_is_counted_while_it_renders(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count of live streams is up while the generator runs and back to zero when it is closed."""
    provider, _ = await _prepared(mass, fake_hass, monkeypatch)
    from music_assistant_models.enums import MediaType  # noqa: PLC0415

    from tests.test_library_provider import _album  # noqa: PLC0415

    album = await _album(provider, "kitchen", 5)
    details = await provider.get_stream_details(f"{album.item_id}|_scape|ambient", MediaType.TRACK)
    assert provider._streams == 0  # noqa: SLF001
    audio = provider.get_audio_stream(details, 0)
    await anext(audio)
    assert provider._streams == 1  # noqa: SLF001
    await audio.aclose()
    assert provider._streams == 0  # noqa: SLF001


def test_the_module_keeps_a_sensible_pace() -> None:
    """The pre-render never rests longer than it works, and gives up on a stream that never ends."""
    assert 0.0 < prerender.YIELD_SECONDS < 1.0
    assert prerender.GIVE_UP_SECONDS >= 600.0
    assert prerender.STARTUP_DELAY_SECONDS > 60.0
    assert datetime.now(UTC).tzinfo is not None
