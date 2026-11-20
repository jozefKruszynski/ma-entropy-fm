"""Tests of the covers: drawn in the look of Music Assistant, or made by an image engine."""

import asyncio
import importlib
import json
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import date
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from music_assistant_models.config_entries import ConfigActionResult, ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import ActionUnavailable
from music_assistant_models.media_items import Album
from PIL import Image

from entropy_fm import maart
from entropy_fm.const import CONF_COVER_LIMIT, CONF_COVER_STYLE, CONF_COVERS, CONF_REDRAW_COVERS
from entropy_fm.covers import (
    COVER_AI,
    COVER_OFF,
    COVER_PROCEDURAL,
    DEFAULT_AI_PER_HOUR,
    FAILURE_BACKOFF_S,
    PALETTES,
    CoverMaker,
    CoverStore,
    DayProfile,
    RateLimiter,
    describe_day,
    procedural_cover,
    profile_of,
)
from entropy_fm.maart import CoverLook, build_prompt
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series
from music_assistant.mass import MusicAssistant
from tests.test_library_provider import _quiet
from tests.test_provider import _setup

if TYPE_CHECKING:
    from entropy_fm import EntropyFMProvider

PERSONAL = ("Anna", "Marek", "Jozef", "Karl", "Bilbo", "Sarah", "Wels", "Vienna", "Siemens")
LOOK = CoverLook(style="music_assistant", custom_prompt="", title_overlay=True)


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
    other_style = _cover("minimal", "kitchen|2026-11-17|minimal")
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
    data = _cover("minimal", "x|y|minimal")
    assert time.process_time() - started < 2.0
    assert len(data) < 250_000


def test_the_context_holds_only_mood_words_and_nothing_of_the_house() -> None:
    """No id, name, area, person, time or count gets into the {context}, whatever the day holds."""
    series = _day()
    profile = profile_of(series)
    for style in (*PALETTES, "unknown"):
        context = describe_day(style, 7, "dorian", profile.intensity, sorted(profile.kinds))
        lowered = context.lower()
        for item in series:
            for token in (item.entity_id, item.name, *item.entity_id.replace(".", "_").split("_")):
                if len(token) > 3 and token.lower() not in {
                    "light",
                    "sensor",
                    "state",
                    "power",
                    "media",
                    "player",
                }:
                    assert token.lower() not in lowered, (token, context)
        for name in (*PERSONAL, "kitchen", "living"):
            assert name.lower() not in lowered.replace("kitchen glow", ""), (name, context)
        assert not any(ch.isdigit() for ch in context), context
        assert context.endswith(" ")


def test_the_prompt_is_built_like_the_playlist_covers_of_music_assistant() -> None:
    """The base prompt, the preset and the no-text rule of a title that is overlaid afterwards."""
    context = describe_day("minimal", 2, "dorian", 0.5, [VoiceKind.DISHWASHER])
    prompt = build_prompt("Kitchen", LOOK, context)
    assert 'called "Kitchen"' in prompt
    assert context in prompt
    assert "Music Assistant playlist cover" in prompt
    assert "lower-left area" in prompt
    assert "No text" in prompt
    assert "rippling water" in prompt
    drawn = build_prompt(
        "Kitchen", CoverLook(style="bold", custom_prompt="", title_overlay=False), context
    )
    assert 'the playlist title "Kitchen"' in drawn
    for value, _ in maart.style_options():
        assert build_prompt(
            "Kitchen", CoverLook(style=value, custom_prompt="", title_overlay=True), context
        )
    assert next(v for v, _ in maart.style_options()) == "music_assistant"


def test_a_custom_prompt_has_the_placeholders_of_music_assistant() -> None:
    """{name}, {title}, {context} and {concept} are filled; anything else stays as typed."""
    look = CoverLook(
        style="custom",
        custom_prompt="A {name} / {title}: {context}| {concept} {x}",
        title_overlay=False,
    )
    prompt = build_prompt("Kitchen", look, "MOOD ", "A CONCEPT")
    assert prompt == "A Kitchen / Kitchen: MOOD | A CONCEPT {x}"
    assert build_prompt(
        "Kitchen", CoverLook(style="custom", custom_prompt="", title_overlay=False), "M "
    )


def test_the_copy_of_the_upstream_presets_is_used_when_the_server_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no upstream module the vendored presets work; with one they are what is used."""
    monkeypatch.setattr(maart, "_UPSTREAM", {})
    real = importlib.import_module

    def refuse(name: str, package: str | None = None) -> ModuleType:
        if name.startswith(maart.UPSTREAM_MODULE):
            raise ImportError(name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", refuse)
    assert maart.upstream() is None
    assert "Music Assistant playlist cover" in build_prompt("K", LOOK, "C ")
    # a server that has the module: its presets and its text win
    monkeypatch.setattr(maart, "_UPSTREAM", {})
    fake = SimpleNamespace(
        Style=maart.Style,
        STYLE_PRESETS={
            s: maart.StylePreset(
                art="UPSTREAM ART", typography="t", title_area="x", layout=maart.TitleLayout.CENTER
            )
            for s in maart.Style
        },
        BASE_PROMPT="Cover {name}. {context}{concept} {style}",
        NO_TEXT_RULES="Nothing in the {area}.",
        TITLE_RULES="T",
        FULL_BLEED="F",
        DEFAULT_CONCEPT="DC",
        DEFAULT_CUSTOM_PROMPT="D",
        CUSTOM_LAYOUT=maart.TitleLayout.TOP_LEFT_LARGE,
    )

    def provide(name: str, package: str | None = None) -> ModuleType | SimpleNamespace:
        if name == f"{maart.UPSTREAM_MODULE}.ai_constants":
            return fake
        if name.startswith(maart.UPSTREAM_MODULE):
            raise ImportError(name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", provide)
    assert build_prompt("K", LOOK, "C ") == "Cover K. C DC UPSTREAM ART Nothing in the x. F"


def test_the_title_is_drawn_over_an_engine_image_as_a_square_jpeg() -> None:
    """The overlay (upstream's or the copy) returns a 1024 pixel JPEG that differs from its input."""
    source = BytesIO()
    Image.new("RGB", (300, 200), (20, 90, 160)).save(source, "PNG")
    out = maart.render_with_title(source.getvalue(), LOOK, "Kitchen, Tue 17 Nov")
    image = Image.open(BytesIO(out))
    assert image.size == (1024, 1024)
    assert image.format == "JPEG"
    plain = maart.render_with_title(
        source.getvalue(),
        CoverLook(style="music_assistant", custom_prompt="", title_overlay=False),
        "K",
    )
    assert plain != out


def test_a_rate_limiter_lets_n_through_an_hour_and_the_default_is_30() -> None:
    """The limit counts an hour of the clock that is passed in."""
    assert DEFAULT_AI_PER_HOUR == 30
    now = [0.0]
    limiter = RateLimiter(2, lambda: now[0])
    assert limiter.allow()
    assert limiter.allow()
    assert not limiter.allow()
    now[0] = 3601.0
    assert limiter.allow()
    assert not RateLimiter(0).allow()


PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 200
DRAWN = b"\xff\xd8\xff" + b"d" * 200


def _maker(tmp_path: Path, limit: int = 30, clock: Callable[[], float] | None = None) -> CoverMaker:
    store = CoverStore(tmp_path / "keep", tmp_path / "cache")
    return CoverMaker(store, RateLimiter(limit), clock or (lambda: 1_000_000.0), timeout=2.0)


async def _make(  # noqa: PLR0913
    maker: CoverMaker,
    key: str,
    generate: Callable[[], Awaitable[bytes]] | None,
    mode: str = COVER_AI,
    *,
    slot: str = "s",
    pinned: bool = False,
) -> bytes | None:
    return await maker.make(mode, slot, key, lambda: DRAWN, generate, pinned=pinned)


async def test_an_ai_cover_is_made_in_the_background_once_and_kept(tmp_path: Path) -> None:
    """The drawn cover is shown first; the made one is served from the store after that."""
    calls: list[int] = []

    async def generate() -> bytes:
        calls.append(1)
        return PNG

    maker = _maker(tmp_path)
    assert await _make(maker, "k1", generate) == DRAWN
    await maker.wait()
    assert await _make(maker, "k1", generate) == PNG
    assert await _make(maker, "k1", generate) == PNG
    assert len(calls) == 1
    await _make(maker, "k2", generate, slot="t", pinned=True)
    await maker.wait()
    assert len(list((tmp_path / "keep").glob("*.img"))) == 1
    # the drawn covers that were shown meanwhile are cached too
    assert len(list((tmp_path / "cache").glob("*.img"))) >= 1


async def test_the_old_cover_stays_until_the_new_one_is_ready_and_after_a_restart(
    tmp_path: Path,
) -> None:
    """A changed key (title, style, prompt, engine) is made anew; the previous cover shows till then."""
    gate = asyncio.Event()

    async def first() -> bytes:
        return PNG

    async def second() -> bytes:
        await gate.wait()
        return PNG + b"2"

    maker = _maker(tmp_path)
    await _make(maker, "old", first)
    await maker.wait()
    assert await _make(maker, "new", second) == PNG
    assert await _make(maker, "new", second) == PNG
    gate.set()
    await maker.wait()
    assert await _make(maker, "new", second) == PNG + b"2"
    # a new maker, as after a restart, still knows what was last shown for the slot
    again = _maker(tmp_path)
    assert await _make(again, "newer", first) == PNG + b"2"


async def test_failures_back_off_for_an_hour_six_hours_a_day_and_then_until_something_changes(
    tmp_path: Path,
) -> None:
    """A failed cover is retried after 1 h, 6 h and 24 h; after that only a new key tries again."""
    now = [1_000_000.0]
    attempts: list[int] = []

    async def boom() -> bytes:
        attempts.append(1)
        raise RuntimeError("down")

    maker = _maker(tmp_path, clock=lambda: now[0])
    assert FAILURE_BACKOFF_S == (3600.0, 21600.0, 86400.0)
    for wait in (*FAILURE_BACKOFF_S, 10 * 86400.0):
        assert await _make(maker, "k", boom) == DRAWN
        await maker.wait()
        before = len(attempts)
        assert await _make(maker, "k", boom) == DRAWN
        await maker.wait()
        assert len(attempts) == before
        now[0] += wait + 1
    # the first try and its three retries (after 1 h, 6 h, 24 h) are all there is: then it waits
    # for a change of key, however long it is left
    assert len(attempts) == 4
    assert await _make(maker, "k", boom) == DRAWN
    await maker.wait()
    assert len(attempts) == 4
    assert await _make(maker, "changed", boom) == DRAWN
    await maker.wait()
    assert len(attempts) == 5


async def test_a_slow_engine_an_answer_that_is_not_an_image_and_the_rate_limit_fall_back(
    tmp_path: Path,
) -> None:
    """Whatever goes wrong the drawn cover is shown; over the limit nothing is asked."""

    async def rubbish() -> bytes:
        return b"not an image"

    async def slow() -> bytes:
        await asyncio.sleep(10)
        return PNG

    maker = _maker(tmp_path)
    for index, generate in enumerate((rubbish, slow)):
        assert await _make(maker, f"k{index}", generate) == DRAWN
        await maker.wait()
        assert await _make(maker, f"k{index}", generate) == DRAWN
    calls: list[int] = []

    async def counting() -> bytes:
        calls.append(1)
        return PNG

    limited = _maker(tmp_path / "other", limit=1)
    await _make(limited, "x1", counting)
    await _make(limited, "x2", counting)
    await limited.wait()
    assert len(calls) == 1
    assert await _make(limited, "x3", counting, COVER_OFF) is None
    assert await _make(limited, "x4", counting, COVER_PROCEDURAL) == DRAWN
    assert await _make(limited, "x5", None) == DRAWN


def _provider_module(provider: object, name: str) -> ModuleType:
    """Return a module of the provider as the server loaded it."""
    return importlib.import_module(f"{type(provider).__module__}.{name}")


def _ai_covers_on(provider: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the hidden AI covers on, as the constant in const.py does when core Music Assistant allows it."""
    monkeypatch.setattr(sys.modules[type(provider).__module__], "AI_COVERS_ENABLED", True)


async def test_ai_covers_and_redraw_are_hidden_and_a_stored_ai_reads_as_drawn(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only off and drawn are offered, the AI settings and the redraw button are gone, "ai" falls back."""
    provider, _ = await _setup(mass, fake_hass)
    engine = MagicMock()
    engine.uid = "p--e"
    api = _provider_module(provider, "imageapi")

    async def one(_: MusicAssistant) -> list[MagicMock]:
        return [engine]

    monkeypatch.setattr(api, "engines", one)
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert [o.value for o in entries[CONF_COVERS].options or []] == [COVER_OFF, COVER_PROCEDURAL]
    assert entries[CONF_COVERS].default_value == COVER_PROCEDURAL
    for key in ("cover_engine", "cover_style", "cover_prompt", "cover_title", CONF_COVER_LIMIT):
        assert key not in entries
    assert CONF_REDRAW_COVERS not in entries
    with pytest.raises(ActionUnavailable):
        await provider.handle_config_action(CONF_REDRAW_COVERS)
    _quiet(provider, monkeypatch, **{CONF_COVERS: COVER_AI})
    assert provider._cover_mode() == COVER_PROCEDURAL  # noqa: SLF001
    albums = await provider.get_artist_albums("kitchen")
    album = next(a for a in albums if a.item_id.startswith("kitchen|"))
    assert album.metadata.images
    assert await provider.resolve_image(next(iter(album.metadata.images)).path) is not None


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


async def test_covers_can_be_off(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With covers off the items have no image and nothing is served."""
    provider, _ = await _setup(mass, fake_hass)
    _quiet(provider, monkeypatch, **{CONF_COVERS: COVER_OFF})
    albums = await provider.get_artist_albums("kitchen")
    album = next(a for a in albums if a.item_id.startswith("kitchen|"))
    assert not album.metadata.images
    assert await provider.resolve_image(f"cover|kitchen|{date(2026, 11, 17)}|ambient") is None


async def test_the_ai_options_are_hidden_without_an_engine_and_the_api_may_be_missing(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No image engines (or no API at all) means no AI option; the provider still loads."""
    provider, _ = await _setup(mass, fake_hass)
    _ai_covers_on(provider, monkeypatch)
    api = _provider_module(provider, "imageapi")

    def keys(entries: Iterable[ConfigEntry]) -> dict[str, ConfigEntry]:
        return {entry.key: entry for entry in entries}

    entries = keys(await provider._cover_entries())  # noqa: SLF001
    assert COVER_AI not in [option.value for option in entries[CONF_COVERS].options or []]
    assert CONF_COVER_LIMIT not in entries
    assert CONF_COVER_STYLE not in entries
    monkeypatch.setattr(api, "_module", lambda: None)
    assert await api.engines(mass) == []
    assert await api.resolve(mass, "x") is None
    with pytest.raises(RuntimeError):
        await api.generate(mass, "x", "p")
    monkeypatch.undo()
    _ai_covers_on(provider, monkeypatch)
    engine = MagicMock()
    engine.uid = "p--e"
    engine.name = "E"
    engine.provider.name = "P"

    async def one(_: MusicAssistant) -> list[MagicMock]:
        return [engine]

    monkeypatch.setattr(api, "engines", one)
    entries = keys(await provider._cover_entries())  # noqa: SLF001
    assert COVER_AI in [option.value for option in entries[CONF_COVERS].options or []]
    assert entries[CONF_COVER_LIMIT].default_value == 30
    styles = [option.value for option in entries[CONF_COVER_STYLE].options or []]
    assert styles[0] == "music_assistant"
    assert {"neon", "bold", "soft_gradient", "risograph", "gouache", "custom"} <= set(styles)


async def test_the_engine_gets_the_area_as_title_and_a_context_of_mood_words(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt has the area as the title only; no entity, name or number of the fake house."""
    provider, _ = await _setup(mass, fake_hass)
    _ai_covers_on(provider, monkeypatch)
    api = _provider_module(provider, "imageapi")
    prompts: list[str] = []
    fail = [False]
    source = BytesIO()
    Image.new("RGB", (64, 64), (10, 80, 140)).save(source, "PNG")

    async def generate(_mass: MusicAssistant, _setting: str | None, prompt: str) -> bytes:
        prompts.append(prompt)
        if fail[0]:
            raise RuntimeError("the engine is down")
        return source.getvalue()

    engine = MagicMock()
    engine.uid = "p--e"

    async def resolve(_mass: MusicAssistant, _setting: str | None) -> object:
        return engine

    monkeypatch.setattr(api, "generate", generate)
    monkeypatch.setattr(api, "resolve", resolve)
    _quiet(provider, monkeypatch, **{CONF_COVERS: COVER_AI})
    albums = [
        a for a in await provider.get_artist_albums("kitchen") if a.item_id.startswith("kitchen|")
    ]
    paths = [next(iter(a.metadata.images or [])).path for a in albums[:2]]
    first = await provider.resolve_image(paths[0])
    assert isinstance(first, bytes)
    assert _picture(first).size == (1024, 1024)
    await provider.covers.wait()
    made = await provider.resolve_image(paths[0])
    assert isinstance(made, bytes)
    assert made.startswith(b"\xff\xd8")
    assert made != first
    assert len(prompts) == 1
    prompt = prompts[0]
    assert 'called "Kitchen"' in prompt
    body = prompt.split("Style:")[0]
    context = body.split('called "Kitchen".')[1]
    forbidden = (
        "kitchen",
        "living",
        "fridge",
        "lamp",
        "washing",
        "geschirr",
        "jozef",
        "motion",
        "door",
    )
    assert not any(word in context.lower().replace("kitchen glow", "") for word in forbidden), (
        context
    )
    assert not any(ch.isdigit() for ch in context)
    assert not any(day in prompt for day in ("Nov", "2026"))
    # the same day again comes from the store; a failing engine leaves the drawn cover
    assert await provider.resolve_image(paths[0]) == made
    assert len(prompts) == 1
    fail[0] = True
    shown = await provider.resolve_image(paths[1])
    assert isinstance(shown, bytes)
    await provider.covers.wait()
    assert _picture(shown).size == (1024, 1024)
    assert await provider.resolve_image(paths[1]) == shown


# ---------------------------------------------------------------- redraw covers


async def _album_image_paths(mass: MusicAssistant, provider: EntropyFMProvider) -> set[str]:
    """Return the cover paths that the library views of the server hold for the albums of this source."""
    items = await mass.music.albums.library_items(provider=provider.instance_id, limit=5000)
    return {image.path for album in items for image in album.metadata.images or []}


async def test_redraw_covers_clears_the_covers_counts_a_redraw_and_asks_for_a_sync(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The button throws away only the covers, numbers the paths anew, keeps the old ones readable and syncs."""
    provider, _ = await _setup(mass, fake_hass)
    _ai_covers_on(provider, monkeypatch)
    entry = next(e for e in await provider.get_config_entries() if e.key == CONF_REDRAW_COVERS)
    assert entry.type == ConfigEntryType.ACTION
    assert entry.action == CONF_REDRAW_COVERS

    albums = await provider.get_artist_albums("kitchen")
    album = next(a for a in albums if isinstance(a, Album) and a.item_id.startswith("kitchen|"))
    assert album.metadata.images
    old = next(iter(album.metadata.images)).path
    assert old.endswith("|g0")
    assert await provider.resolve_image(old)
    store = provider.covers.store
    assert list(store.cache.glob("*.img"))
    other = Path(provider.storage_root) / "other.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("{}")
    pinned_audio = Path(provider.storage_root) / "pins" / "keep.json"
    pinned_audio.parent.mkdir(parents=True, exist_ok=True)
    pinned_audio.write_text("{}")
    store.put("pinned-cover", b"\x89PNG\r\n\x1a\n" + b"\0" * 200, pinned=True)
    provider.covers.limiter.allow()
    provider.covers._failures["x"] = (1, 1e12)  # noqa: SLF001

    asked: list[list[str]] = []
    real_start = mass.music.start_sync

    async def start_sync(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs["providers"])
        return await real_start(*args, **kwargs)

    monkeypatch.setattr(mass.music, "start_sync", start_sync)
    result = await mass.config.invoke_provider_config_action(
        provider.instance_id, CONF_REDRAW_COVERS
    )
    assert isinstance(result, ConfigActionResult)
    assert result.translation_key == "redraw_covers.result"
    assert asked == [[provider.instance_id]]

    # the covers are gone, everything else stays
    assert not list(store.cache.glob("*.img"))
    assert not list(store.permanent.glob("*.img"))
    assert not (store.cache / "slots.json").exists()
    assert not provider.covers._failures  # noqa: SLF001
    assert not provider.covers.limiter._times  # noqa: SLF001
    assert other.exists()
    assert pinned_audio.exists()
    await _paths_are_new_and_the_sync_follows(mass, provider, old)


async def _paths_are_new_and_the_sync_follows(
    mass: MusicAssistant, provider: EntropyFMProvider, old: str
) -> None:
    """Check the paths after a redraw: new ones are made, old ones read, and the sync stores them."""
    # the generation is kept in our own storage and is part of the paths
    assert provider._cover_generation == 1  # noqa: SLF001
    assert json.loads((Path(provider.storage_root) / "cover_generation.json").read_text()) == 1
    albums = await provider.get_artist_albums("kitchen")
    album = next(a for a in albums if isinstance(a, Album) and a.item_id.startswith("kitchen|"))
    assert album.metadata.images
    new = next(iter(album.metadata.images)).path
    assert new.endswith("|g1")
    assert new != old
    assert new.removesuffix("|g1") == old.removesuffix("|g0")

    # old and old-old forms still resolve, a bad generation does not
    assert await provider.resolve_image(new)
    assert await provider.resolve_image(old)
    assert await provider.resolve_image(old.removesuffix("|g0"))
    assert await provider.resolve_image(f"{old.removesuffix('|g0')}|gx") is None
    assert await provider.resolve_image(f"{old}|extra") is None
    assert provider.accepts_stored_image_path(old)

    # the sync gives the stored albums the new paths
    for _ in range(600):
        await asyncio.sleep(0.05)
        if not mass.music.active_sync_tasks:
            break
    paths = await _album_image_paths(mass, provider)
    assert paths
    assert all(path.endswith("|g1") for path in paths), paths


async def test_an_unknown_action_is_refused_and_the_generation_survives_a_reload(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the redraw button is handled, and the number is read back from the storage."""
    provider, _ = await _setup(mass, fake_hass)
    _ai_covers_on(provider, monkeypatch)
    with pytest.raises(ActionUnavailable):
        await provider.handle_config_action("nothing")
    path = Path(provider.storage_root) / "cover_generation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("7")
    assert provider._read_cover_generation() == 7  # noqa: SLF001
    path.write_text("junk")
    assert provider._read_cover_generation() == 0  # noqa: SLF001
