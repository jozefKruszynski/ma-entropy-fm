"""
Entropy FM for Music Assistant.

A music source that turns the history of your Home Assistant into generative soundscapes.
Every area of the house is an artist, every day of an area an album, and the track of an album
is one long, slowly evolving piece in each of six styles. The house steers the texture, the
density, the brightness, the space and the movement of the piece; it never writes the notes. The
audio is synthesised while it plays, from samples that ship with the provider; no audio is
stored, and nothing leaves the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import re
import threading
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hass_client import HomeAssistantClient
from hass_client.exceptions import AuthenticationFailed, BaseHassClientError
from hass_client.utils import get_websocket_url
from music_assistant_models.config_entries import (
    ConfigActionResult,
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    AlbumType,
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    ProviderUnavailableError,
    SetupFailedError,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemMetadata,
    MediaItemType,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.models.music_provider import MusicProvider

from . import imageapi, prerender
from .audiocache import AudioCache, cache_key
from .autotune import benchmark, heaviest_style
from .baseline import EntityBaseline
from .const import (
    ARTIST_PATH_PREFIX,
    CONF_AUDIO_CACHE_MB,
    CONF_COVER_ENGINE,
    CONF_COVER_LIMIT,
    CONF_COVER_PROMPT,
    CONF_COVER_STYLE,
    CONF_COVER_TITLE,
    CONF_COVERS,
    CONF_DAY_CHARACTER,
    CONF_DAYS,
    CONF_DEFAULT_STYLE,
    CONF_ENERGY_OFFSET,
    CONF_EXCLUDE,
    CONF_HIDE_QUIET,
    CONF_HOUSE_SELECTED,
    CONF_INCLUDE_PRESENCE,
    CONF_KEEP_ROOMS,
    CONF_KEY,
    CONF_LENGTH,
    CONF_LIBRARY_DAYS,
    CONF_PRERENDER,
    CONF_QUALITY,
    CONF_REDRAW_COVERS,
    CONF_ROLES,
    CONF_ROOMS,
    CONF_SCALE,
    CONF_STYLES,
    CONF_TOKEN,
    CONF_URL,
    CONF_USE_PLUGIN,
    CONF_VERIFY_SSL,
    COVER_GENERATION_FILE,
    COVER_PATH_PREFIX,
    DEFAULT_AUDIO_CACHE_MB,
    DEFAULT_DAYS,
    DEFAULT_KEY,
    DEFAULT_LIBRARY_DAYS,
    DEFAULT_SCALE,
    ENGINE_VERSION,
    HASS_DOMAIN,
    LEGACY_TARGETS,
    LIBRARY_SYNC_DELAY,
    MAX_DAYS,
    MAX_ENERGY_OFFSET,
    MIN_SCAPE_SECONDS_OF_DAY,
    OFFERED_FILE,
    PLAYABLE_COMPRESSION,
    SCAPE_ID,
    STORAGE_FOLDER,
    WHOLE_HOUSE_ID,
    WHOLE_HOUSE_NAME,
)
from .covers import (
    COVER_AI,
    COVER_OFF,
    COVER_PROCEDURAL,
    DEFAULT_AI_PER_HOUR,
    CoverMaker,
    CoverStore,
    RateLimiter,
    describe_day,
    procedural_cover,
    profile_of,
)
from .day_analysis import DayAnalysis
from .hass_access import HassAccess
from .history import HistoryStore
from .library import (
    Catalog,
    TrackRef,
    album_id,
    album_name,
    build_catalog,
    covered_seconds,
    day_bounds,
    parse_album_id,
    parse_track_id,
    playable,
    quiet_areas,
    recent_days,
    track_id,
    track_seconds,
)
from .maart import CoverLook, build_prompt, render_with_title, style_options
from .palette import DayPalette, DayTraits, derive_palette, measure_traits
from .pins import (
    Pin,
    PinStore,
    entity_from_dict,
    entity_to_dict,
    load_membership,
    save_membership,
    write_atomic,
)
from .plan import CONDENSED, DEFAULT_LENGTH, FULL, LENGTHS, nominal_seconds
from .quality import (
    AUTO,
    DEFAULT_QUALITY,
    ORDER,
    QUALITIES,
    STANDARD,
    Quality,
    choose_tier,
    get_quality,
)
from .references import rescale
from .render import render_pcm
from .roles import parse_overrides
from .roomart import artist_fanart, artist_thumb, room_kind
from .scape import analysis_of, duration_of, scape_score
from .score import Score
from .styles import DEFAULT_STYLE, STYLES, Style, get_style
from .surprise import Surprise
from .synth import NOTE_NAMES, SAMPLE_RATE, SCALES, Scale
from .timeline import EntityInfo, Series
from .titles import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    DayFacts,
    album_review,
    facts_of,
    make_title,
    track_description,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.helpers.json import SerializableType
    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
    ProviderFeature.LIBRARY_ARTISTS,
    ProviderFeature.LIBRARY_ALBUMS,
    ProviderFeature.LIBRARY_TRACKS,
    # the library is the shown rooms and their days, nothing to edit; a heart is what keeps a day
    ProviderFeature.FAVORITE_ARTISTS_EDIT,
    ProviderFeature.FAVORITE_ALBUMS_EDIT,
    ProviderFeature.FAVORITE_TRACKS_EDIT,
    ProviderFeature.ARTIST_ALBUMS,
}
ARTIST_KEY = "@artist|"
# the target of a track: the soundscape of a day
TRACK_TARGETS = frozenset({SCAPE_ID})
# how many soundscapes are kept between a request for their length and the stream
MAX_KEPT_SCAPES = 16
# how long the benchmark of the Auto quality stays valid, in seconds
AUTO_EXPIRATION = 30 * 86400
CACHE_CATEGORY_AUTO = 6
# how many readings of a day (for its titles) are kept
MAX_KEPT_FACTS = 256
# how often the days of the followed areas are pinned, in seconds
MAINTENANCE_SECONDS = 6 * 3600
# how long a read of the registries is reused, so that a library sync is one read
CATALOG_TTL = 300
# the core plugin may still be loading when this provider starts
PLUGIN_WAIT_ATTEMPTS = 6
PLUGIN_WAIT_SECONDS = 2
SCALE_TITLES = {
    "dorian": "Dorian",
    "aeolian": "Natural minor (Aeolian)",
    "mixolydian": "Mixolydian",
    "lydian": "Lydian",
    "major": "Major (Ionian)",
    "minor_pentatonic": "Minor pentatonic",
    "major_pentatonic": "Major pentatonic",
}


@dataclass
class _Gathered:
    """What a track of a day is made of."""

    series: list[Series]
    whole_day: list[Series]
    baselines: dict[str, EntityBaseline]
    covered: float


@dataclass
class _Scape:
    """The soundscape of a day in a style: what it was made from, how the day was read and how long it is."""

    gathered: _Gathered
    analysis: DayAnalysis
    palette: DayPalette | None
    seed: str
    duration: int


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the provider for one settings entry."""
    return EntropyFMProvider(mass, manifest, config, SUPPORTED_FEATURES)


def _lower_priority() -> None:
    """Make the thread that does the pre-render the politest one of the process (where the system allows it)."""
    with contextlib.suppress(AttributeError, OSError):
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)


class EntropyFMProvider(MusicProvider):
    """Plays the history of Home Assistant as generative soundscapes."""

    hass: HassAccess
    history: HistoryStore
    _own_client: HomeAssistantClient | None = None
    _listen_task: asyncio.Task[None] | None = None
    _catalog: Catalog | None = None
    _catalog_time: float = 0.0
    _catalog_lock: asyncio.Lock
    _tz: tzinfo | None = None
    _auto_quality: Quality = STANDARD
    _maintenance_task: asyncio.Task[None] | None = None
    _benchmark_task: asyncio.Task[None] | None = None
    _prerender_task: asyncio.Task[None] | None = None
    _sync_task: asyncio.Task[None] | None = None
    _cover_generation: int = 0
    _prerender_pool: ThreadPoolExecutor | None = None
    # how many live streams are being rendered now, which the pre-render waits for
    _streams: int = 0

    @property
    def is_streaming_provider(self) -> bool:
        """
        Return True: what the source offers is larger than the library.

        Browsing and search show every day that Home Assistant still has; the library only holds
        what the user added, and those days are pinned.
        """
        return True

    async def _room_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the settings that pick the rooms: which are offered, what the house is made of, which are kept."""
        try:
            names = self._artist_names(await self._catalog_all())
        except Exception:  # noqa: BLE001 - without Home Assistant there is nothing to pick yet
            names = {}
        options = [
            ConfigValueOption(area_id, title)
            for area_id, title in names.items()
            if area_id != WHOLE_HOUSE_ID
        ]
        # the rooms of "keep every day" are the ones that are shown, when some are picked
        selected = self._selected_rooms
        keep_options = [option for option in options if not selected or option.value in selected]
        return (
            # an empty list is every room, including the ones that Home Assistant gets later
            ConfigEntry(
                key=CONF_ROOMS,
                type=ConfigEntryType.STRING,
                multi_value=True,
                default_value=[],
                required=False,
                options=options,
            ),
            ConfigEntry(
                key=CONF_HOUSE_SELECTED,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
            ),
            ConfigEntry(
                key=CONF_HIDE_QUIET,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
            ),
            ConfigEntry(
                key=CONF_KEEP_ROOMS,
                type=ConfigEntryType.STRING,
                multi_value=True,
                default_value=[],
                required=False,
                options=keep_options,
            ),
        )

    async def _cover_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the settings of the covers.

        The AI option is only offered when the server has an image engine, and its settings (which
        engine, how many covers an hour) only appear when it is chosen.
        """
        has_engine = bool(await imageapi.engines(self.mass))
        options = [
            ConfigValueOption(COVER_OFF, "Off"),
            ConfigValueOption(COVER_PROCEDURAL, "Drawn from the day"),
        ]
        if has_engine:
            options.append(ConfigValueOption(COVER_AI, "AI image engine"))
        entries: list[ConfigEntry] = [
            ConfigEntry(
                key=CONF_COVERS,
                type=ConfigEntryType.STRING,
                default_value=COVER_PROCEDURAL,
                required=False,
                options=options,
            )
        ]
        if has_engine:
            entries.extend(
                await imageapi.picker_entries(self.mass, CONF_COVER_ENGINE, CONF_COVERS, COVER_AI)
            )
            entries.extend(
                [
                    ConfigEntry(
                        key=CONF_COVER_STYLE,
                        type=ConfigEntryType.STRING,
                        default_value="music_assistant",
                        options=[ConfigValueOption(v, t) for v, t in style_options()],
                        depends_on=CONF_COVERS,
                        depends_on_value=COVER_AI,
                        required=False,
                    ),
                    ConfigEntry(
                        key=CONF_COVER_PROMPT,
                        type=ConfigEntryType.STRING,
                        default_value="",
                        depends_on=CONF_COVER_STYLE,
                        depends_on_value="custom",
                        required=False,
                    ),
                    ConfigEntry(
                        key=CONF_COVER_TITLE,
                        type=ConfigEntryType.BOOLEAN,
                        default_value=True,
                        depends_on=CONF_COVERS,
                        depends_on_value=COVER_AI,
                        required=False,
                    ),
                    ConfigEntry(
                        key=CONF_COVER_LIMIT,
                        type=ConfigEntryType.INTEGER,
                        default_value=DEFAULT_AI_PER_HOUR,
                        range=(1, 120),
                        depends_on=CONF_COVERS,
                        depends_on_value=COVER_AI,
                        required=False,
                    ),
                ]
            )
        entries.append(
            ConfigEntry(
                key=CONF_REDRAW_COVERS,
                type=ConfigEntryType.ACTION,
                action=CONF_REDRAW_COVERS,
                required=False,
            )
        )
        return tuple(entries)

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the settings of the provider."""
        return (
            ConfigEntry(
                key=CONF_USE_PLUGIN,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_URL,
                type=ConfigEntryType.STRING,
                default_value="",
                required=False,
                depends_on=CONF_USE_PLUGIN,
                depends_on_value=False,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_TOKEN,
                type=ConfigEntryType.SECURE_STRING,
                default_value="",
                required=False,
                depends_on=CONF_USE_PLUGIN,
                depends_on_value=False,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_VERIFY_SSL,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
                depends_on=CONF_USE_PLUGIN,
                depends_on_value=False,
                advanced=True,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_DAYS,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_DAYS,
                range=(1, MAX_DAYS),
                required=False,
            ),
            ConfigEntry(
                key=CONF_LIBRARY_DAYS,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_LIBRARY_DAYS,
                range=(1, MAX_DAYS),
                required=False,
            ),
            *await self._room_entries(),
            ConfigEntry(
                key=CONF_QUALITY,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_QUALITY,
                required=False,
                options=[ConfigValueOption(AUTO, "Auto")]
                + [ConfigValueOption(tier.key, tier.title) for tier in QUALITIES.values()],
            ),
            ConfigEntry(
                key=CONF_AUDIO_CACHE_MB,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_AUDIO_CACHE_MB,
                range=(0, 20000),
                required=False,
            ),
            ConfigEntry(
                key=CONF_PRERENDER,
                type=ConfigEntryType.STRING,
                default_value=prerender.DEFAULT_PRERENDER,
                required=False,
                options=[
                    ConfigValueOption(prerender.OFF, "Off"),
                    ConfigValueOption(prerender.PINNED, "Pinned rooms"),
                    ConfigValueOption(prerender.ALL, "All rooms"),
                ],
            ),
            *await self._cover_entries(),
            ConfigEntry(
                key=CONF_LENGTH,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_LENGTH,
                required=False,
                options=[
                    ConfigValueOption(FULL, "Full (10 to 22 minutes)"),
                    ConfigValueOption(CONDENSED, "Condensed (5 to 7 minutes)"),
                ],
            ),
            ConfigEntry(
                key=CONF_KEY,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_KEY,
                required=False,
                options=[ConfigValueOption(name, name) for name in NOTE_NAMES],
            ),
            ConfigEntry(
                key=CONF_SCALE,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_SCALE,
                required=False,
                options=[ConfigValueOption(name, SCALE_TITLES[name]) for name in SCALES],
            ),
            ConfigEntry(
                key=CONF_STYLES,
                type=ConfigEntryType.STRING,
                default_value=list(STYLES),
                required=False,
                multi_value=True,
                options=[ConfigValueOption(key, style.title) for key, style in STYLES.items()],
            ),
            ConfigEntry(
                key=CONF_DEFAULT_STYLE,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_STYLE,
                required=False,
                options=[ConfigValueOption(key, style.title) for key, style in STYLES.items()],
            ),
            ConfigEntry(
                key=CONF_ENERGY_OFFSET,
                type=ConfigEntryType.INTEGER,
                default_value=0,
                range=(-MAX_ENERGY_OFFSET, MAX_ENERGY_OFFSET),
                required=False,
            ),
            ConfigEntry(
                key=CONF_DAY_CHARACTER,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
            ),
            ConfigEntry(
                key=CONF_INCLUDE_PRESENCE,
                type=ConfigEntryType.BOOLEAN,
                default_value=False,
                required=False,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_ROLES,
                type=ConfigEntryType.STRING,
                default_value="",
                required=False,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_EXCLUDE,
                type=ConfigEntryType.STRING,
                default_value="",
                required=False,
                advanced=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Connect to Home Assistant, through the core plugin or with the URL and token."""
        self._catalog_lock = asyncio.Lock()
        if self.get_config_value(CONF_USE_PLUGIN, default=True):
            await self._wait_for_plugin()
            self.hass = HassAccess(plugin_getter=lambda: self.mass.get_provider(HASS_DOMAIN))
        else:
            self._own_client = await self._connect_own_client()
            self.hass = HassAccess(client=self._own_client)
        self.storage_root = str(Path(self.mass.storage_path) / STORAGE_FOLDER)
        self.pins = PinStore(self.storage_root)
        self.membership = load_membership(self.storage_root)
        self._cover_generation = self._read_cover_generation()
        self._scapes: OrderedDict[tuple[str, date, str, int, str, tuple[str, ...]], _Scape] = (
            OrderedDict()
        )
        # what a day was like, for its titles: kept for the hour (a past day for good)
        self._facts_kept: OrderedDict[tuple[Any, ...], DayFacts] = OrderedDict()
        self.history = HistoryStore(
            self.hass, self.mass.cache, self.instance_id, self.logger, self._pinned_scope
        )
        self.audio_cache = AudioCache(
            str(Path(self.mass.cache_path) / STORAGE_FOLDER),
            int(self.get_config_value(CONF_AUDIO_CACHE_MB, DEFAULT_AUDIO_CACHE_MB) or 0),
        )
        self.covers = CoverMaker(
            CoverStore(
                Path(self.storage_root) / "covers",
                Path(self.mass.cache_path) / STORAGE_FOLDER / "covers",
            ),
            RateLimiter(DEFAULT_AI_PER_HOUR),
        )
        await self._drop_voice_tracks()
        await self._migrate_ensemble_tracks()
        await self._migrate_library_model()
        self._pin_lock = asyncio.Lock()
        self._benchmark_task = self.mass.create_task(self._pick_quality())
        self._maintenance_task = self.mass.create_task(self._maintain())
        self._prerender_task = self.mass.create_task(self._prerender_loop())
        self._sync_task = self.mass.create_task(self._sync_when_rooms_change())

    def _read_cover_generation(self) -> int:
        """Return the number of the last redraw of the covers, from our own storage; 0 when there was none."""
        try:
            value = json.loads((Path(self.storage_root) / COVER_GENERATION_FILE).read_text())
        except OSError, ValueError:
            return 0
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

    async def handle_config_action(
        self, action: str
    ) -> tuple[ConfigEntry, ...] | ConfigActionResult | None:
        """
        Run the button *Redraw covers*.

        It throws away the covers that Entropy FM made, counts the redraws so that the cover
        paths (and with them the image cache of the server) are new, and asks the server to sync
        this source so that the stored albums and tracks take the new paths. New covers are
        made as albums are shown.

        :param action: The action id of the pressed button.
        """
        if action != CONF_REDRAW_COVERS:
            return await super().handle_config_action(action)
        await self.covers.clear()
        self._cover_generation += 1
        await asyncio.to_thread(
            write_atomic,
            Path(self.storage_root) / COVER_GENERATION_FILE,
            json.dumps(self._cover_generation).encode(),
        )
        await self.mass.music.start_sync(providers=[self.instance_id])
        return ConfigActionResult(
            message="The covers are thrown away; new ones are made as albums are shown.",
            translation_key=f"{CONF_REDRAW_COVERS}.result",
        )

    async def get_diagnostics(self) -> dict[str, SerializableType] | None:
        """Return why each entity has its voice, so that a surprising sound can be explained."""
        try:
            catalog = await self._get_catalog()
        except Exception as err:  # noqa: BLE001 - diagnostics must not fail
            return {"error": str(err)}
        return {
            "connection": "own" if self._own_client is not None else "plugin",
            "areas": len(catalog.areas),
            "voices": {
                info.entity_id: f"{info.kind.value}, {info.why}"
                for infos in catalog.entities.values()
                for info in infos
            },
        }

    async def unload(self, is_removed: bool = False) -> None:  # noqa: FBT001, FBT002
        """Close the connection of our own, if there is one."""
        tasks = (
            self._maintenance_task,
            self._benchmark_task,
            self._prerender_task,
            self._sync_task,
        )
        for task in tasks:
            if task is not None:
                task.cancel()
        self._maintenance_task = self._benchmark_task = self._prerender_task = None
        self._sync_task = None
        if self._prerender_pool is not None:
            self._prerender_pool.shutdown(wait=False, cancel_futures=True)
            self._prerender_pool = None
        self.covers.stop()
        if self._listen_task is not None:
            self._listen_task.cancel()
            self._listen_task = None
        if self._own_client is not None:
            await self._own_client.disconnect()
            self._own_client = None

    async def _wait_for_plugin(self) -> None:
        """Wait a little for the core Home Assistant plugin, and fail clearly without it."""
        for attempt in range(PLUGIN_WAIT_ATTEMPTS):
            plugin = self.mass.get_provider(HASS_DOMAIN)
            if plugin is not None and getattr(plugin, "hass", None) is not None:
                return
            if attempt < PLUGIN_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(PLUGIN_WAIT_SECONDS)
        msg = (
            "Entropy FM needs the Home Assistant plugin of Music Assistant: add and connect "
            "it under Settings > Providers, or turn off 'Use the Home Assistant plugin' and "
            "enter a URL and a long-lived access token"
        )
        raise SetupFailedError(msg)

    async def _connect_own_client(self) -> HomeAssistantClient:
        """Connect with the URL and the token from the settings."""
        url = str(self.get_config_value(CONF_URL, "") or "").strip()
        token = str(self.get_config_value(CONF_TOKEN, "") or "").strip()
        if not url or not token:
            msg = "Enter the URL and a long-lived access token of Home Assistant"
            raise SetupFailedError(msg)
        verify = bool(self.get_config_value(CONF_VERIFY_SSL, default=True))
        session = self.mass.http_session if verify else self.mass.http_session_no_ssl
        client = HomeAssistantClient(get_websocket_url(url), token, session)
        try:
            await client.connect()
        except AuthenticationFailed as err:
            raise LoginFailed("Home Assistant did not accept the access token") from err
        except BaseHassClientError as err:
            raise SetupFailedError(str(err) or err.__class__.__name__) from err
        self._listen_task = self.mass.create_task(self._listen(client))
        return client

    async def _listen(self, client: HomeAssistantClient) -> None:
        """Process the messages of our own connection until it ends."""
        try:
            await client.start_listening()
        except BaseHassClientError as err:
            self.logger.warning("Connection to Home Assistant lost: %s", err)

    # -- settings ----------------------------------------------------------------------------

    @property
    def _library_days(self) -> int:
        """Return how many days, counted back from today, of every room are in the library."""
        value = int(self.get_config_value(CONF_LIBRARY_DAYS, DEFAULT_LIBRARY_DAYS) or 1)
        return max(1, min(value, self._days))

    @property
    def _keep_rooms(self) -> set[str]:
        """Return the rooms whose every day is kept: the setting, and what artists asked before."""
        chosen: object = self.get_config_value(CONF_KEEP_ROOMS, [])
        listed = {str(item) for item in chosen} if isinstance(chosen, list | tuple) else set()
        return listed | set(self.membership.followed)

    @property
    def _selected_rooms(self) -> set[str]:
        """Return the rooms that are ticked in *Rooms*; empty means every room, those added later too."""
        chosen: object = self.get_config_value(CONF_ROOMS, [])
        return {str(item) for item in chosen} if isinstance(chosen, list | tuple) else set()

    @property
    def _length_mode(self) -> str:
        """Return the length of the pieces: full (10 to 22 minutes) or condensed (5 to 7)."""
        mode = str(self.get_config_value(CONF_LENGTH, DEFAULT_LENGTH) or DEFAULT_LENGTH)
        return mode if mode in LENGTHS else DEFAULT_LENGTH

    @property
    def _language(self) -> str:
        """Return the language of titles and descriptions: the one of the server when it is offered, else English."""
        try:
            language = str(self.mass.metadata.preferred_language)
        except Exception:  # noqa: BLE001
            return DEFAULT_LANGUAGE
        return language if language in LANGUAGES else DEFAULT_LANGUAGE

    def _is_selected(self, area_id: str) -> bool:
        """Return whether a room is ticked in *Rooms* (every room is, when none is)."""
        selected = self._selected_rooms
        return not selected or area_id in selected

    def _house_rooms(self, catalog: Catalog) -> list[str]:
        """Return the rooms that the whole house is made of: the ticked ones, or every room when that is off."""
        if not self.get_config_value(CONF_HOUSE_SELECTED, default=True):
            return list(catalog.areas)
        return [area_id for area_id in catalog.areas if self._is_selected(area_id)]

    async def _sync_when_rooms_change(self) -> None:
        """Wait for the server to register the provider, then sync if the offered rooms changed; a reload cancels it."""
        await asyncio.sleep(LIBRARY_SYNC_DELAY)
        try:
            await self.sync_if_rooms_changed()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.warning(
                "Could not ask for a library sync after the rooms changed", exc_info=True
            )

    async def sync_if_rooms_changed(self) -> bool:
        """
        Ask the server to sync this source when the rooms that are offered differ from last time.

        The server only notices a change of the offered rooms at its next sync, which also takes
        the items of a room that left out of the library views. A change of *Piece length* asks the
        same way, so that the items are listed again with their new length. It is a plain request
        to the music controller for this instance; the library itself is never written. The rooms
        and the length of the last request are kept in our own storage. Nothing happens when
        Home Assistant cannot tell the rooms, or when no room is offered.

        Returns whether a sync was asked for.
        """
        try:
            catalog = await self._get_catalog()
        except ProviderUnavailableError:
            self.logger.debug("Home Assistant is not available, no library sync is asked for")
            return False
        offered = await self._offered(await self._catalog_all())
        if not catalog.areas or not offered:
            return False
        wanted = {"rooms": sorted(offered), "length": self._length_mode}
        path = Path(self.storage_root) / OFFERED_FILE
        try:
            known = json.loads(await asyncio.to_thread(path.read_text))
        except OSError, ValueError:
            known = None
        if known == wanted:
            return False
        what = (
            "length of the pieces"
            if isinstance(known, dict) and known.get("rooms") == wanted["rooms"]
            else "rooms that are offered"
        )
        self.logger.info("The %s changed, syncing the library", what)
        await self.mass.music.start_sync(providers=[self.instance_id])
        await asyncio.to_thread(write_atomic, path, json.dumps(wanted).encode())
        return True

    async def _offered(self, catalog: Catalog) -> dict[str, str]:
        """
        Return the artists that are offered: the whole house, then the rooms that are ticked and not quiet.

        A room that is not offered is gone from browsing, search, the library and the pre-render; what was
        pinned or cached for it stays where it is.

        :param catalog: The areas and their entities.
        """
        names = self._artist_names(catalog)
        rooms = {
            area_id: name
            for area_id, name in names.items()
            if area_id != WHOLE_HOUSE_ID and self._is_selected(area_id)
        }
        if rooms and self.get_config_value(CONF_HIDE_QUIET, default=True):
            tz, now = await self._timezone(), await self._now()
            first = recent_days(now.date(), self._days)[-1]
            quiet = quiet_areas(catalog, day_bounds(first, tz)[0])
            rooms = {area_id: name for area_id, name in rooms.items() if area_id not in quiet}
        return {WHOLE_HOUSE_ID: WHOLE_HOUSE_NAME, **rooms} if rooms else {}

    @property
    def _days(self) -> int:
        """Return how many days back the library goes."""
        return max(1, min(MAX_DAYS, int(self.get_config_value(CONF_DAYS, DEFAULT_DAYS) or 1)))

    @property
    def _compression(self) -> int:
        """Return the seconds of the day per second of the shortest track, which decides when a day is playable."""
        return PLAYABLE_COMPRESSION

    @property
    def quality(self) -> Quality:
        """Return the render tier: the one the user chose, or the one that Auto picked."""
        chosen = str(self.get_config_value(CONF_QUALITY, DEFAULT_QUALITY) or DEFAULT_QUALITY)
        return self._auto_quality if chosen == AUTO else get_quality(chosen)

    async def _pick_quality(self) -> None:
        """Run the benchmark of the Auto setting once per machine and engine version, and keep it."""
        if str(self.get_config_value(CONF_QUALITY, DEFAULT_QUALITY)) != AUTO:
            return
        style = heaviest_style(offered.key for offered in self._offered_styles)
        key = f"auto|{ENGINE_VERSION}|{style}"
        costs = await self.mass.cache.get(
            key, provider=self.instance_id, category=CACHE_CATEGORY_AUTO
        )
        if not costs:
            costs = await asyncio.to_thread(benchmark, ORDER, style)
            await self.mass.cache.set(
                key,
                costs,
                expiration=AUTO_EXPIRATION,
                provider=self.instance_id,
                category=CACHE_CATEGORY_AUTO,
            )
        self._auto_quality = choose_tier(costs)
        self.logger.info(
            "Render quality: %s (ms of CPU per second of audio, in %s: %s)",
            self._auto_quality.title,
            style,
            {name: round(value, 1) for name, value in costs.items()},
        )

    @property
    def _scale(self) -> Scale:
        """Return the key and mode the music is in."""
        key = str(self.get_config_value(CONF_KEY, DEFAULT_KEY))
        mode = str(self.get_config_value(CONF_SCALE, DEFAULT_SCALE))
        root = NOTE_NAMES.index(key) if key in NOTE_NAMES else NOTE_NAMES.index(DEFAULT_KEY)
        return Scale(root, mode if mode in SCALES else DEFAULT_SCALE)

    @property
    def _default_style(self) -> Style:
        """Return the style of tracks that have no version of their own."""
        return get_style(str(self.get_config_value(CONF_DEFAULT_STYLE, DEFAULT_STYLE)))

    @property
    def _offered_styles(self) -> list[Style]:
        """Return the styles the library offers, the default one always among them."""
        chosen = self.get_config_value(CONF_STYLES, list(STYLES))
        keys = [str(key) for key in chosen] if isinstance(chosen, list | tuple) else [str(chosen)]
        styles = [STYLES[key] for key in STYLES if key in keys]
        default = self._default_style
        if default not in styles:
            styles.insert(0, default)
        return styles

    @property
    def _energy_offset(self) -> float:
        """Return how far the user moves the energy of the house, from -0.5 to 0.5."""
        value = int(self.get_config_value(CONF_ENERGY_OFFSET, 0) or 0)
        return max(-MAX_ENERGY_OFFSET, min(MAX_ENERGY_OFFSET, value)) / 100.0

    @property
    def _day_character(self) -> bool:
        """Return whether a day picks its own palette and the unusual comes forward."""
        return bool(self.get_config_value(CONF_DAY_CHARACTER, default=True))

    @property
    def _exclude_globs(self) -> list[str]:
        """Return the patterns of entities the user keeps out."""
        return [
            part.strip() for part in str(self.get_config_value(CONF_EXCLUDE, "") or "").split(",")
        ]

    @property
    def _include_presence(self) -> bool:
        """Return whether persons and device trackers play."""
        return bool(self.get_config_value(CONF_INCLUDE_PRESENCE, default=False))

    # -- the catalog -------------------------------------------------------------------------

    async def _timezone(self) -> tzinfo:
        """Return the time zone of the house."""
        if self._tz is None:
            self._tz = await self.hass.time_zone()
        return self._tz

    async def _now(self) -> datetime:
        """Return the current time in the time zone of the house."""
        return datetime.now(UTC).astimezone(await self._timezone())

    async def _get_catalog(self) -> Catalog:
        """Return the areas and entities, read from the registries and the states only."""
        loop_time = asyncio.get_running_loop().time()
        async with self._catalog_lock:
            if self._catalog is not None and loop_time - self._catalog_time < CATALOG_TTL:
                return self._catalog
            try:
                snapshot = await self.hass.candidate_states(
                    self._exclude_globs, include_presence=self._include_presence
                )
            except BaseHassClientError as err:
                raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err
            self._catalog = build_catalog(
                snapshot.areas,
                snapshot.entity_areas,
                snapshot.states,
                self._exclude_globs,
                include_presence=self._include_presence,
                meta=snapshot.meta,
                overrides=parse_overrides(str(self.get_config_value(CONF_ROLES, "") or "")),
                icons=snapshot.icons,
            )
            self._catalog_time = loop_time
            return self._catalog

    async def _days_and_catalog(self) -> tuple[list[date], Catalog, tzinfo, datetime]:
        """Return the days that have an album, the catalog, the time zone and the time."""
        catalog = await self._get_catalog()
        tz = await self._timezone()
        now = datetime.now(UTC).astimezone(tz)
        days = [
            day
            for day in recent_days(now.date(), self._days)
            if playable(day, tz, now, self._compression)
        ]
        return days, catalog, tz, now

    # -- media items -------------------------------------------------------------------------

    def _mapping(self, item_id: str) -> ProviderMapping:
        """Return where an item lives on this source."""
        return ProviderMapping(
            item_id=item_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            # what this house did is its own: another instance of the provider has other days
            is_unique=True,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE, sample_rate=SAMPLE_RATE, bit_depth=16
            ),
        )

    def _artist(self, area_id: str, name: str) -> Artist:
        """Return the artist of an area."""
        return Artist(
            item_id=area_id,
            provider=self.instance_id,
            name=name,
            provider_mappings={self._mapping(area_id)},
            metadata=MediaItemMetadata(images=self._artist_images(area_id)),
            # True for a heart; never False, which Music Assistant reads as a dislike
            favorite=True if area_id in self.membership.followed else None,
        )

    def _artist_images(self, area_id: str) -> UniqueList[MediaItemImage] | None:
        """Return the drawn thumb and the wide backdrop of a room, or None when covers are off."""
        if self._cover_mode() == COVER_OFF:
            return None
        return UniqueList(
            [
                MediaItemImage(
                    type=image_type,
                    path=f"{ARTIST_PATH_PREFIX}|{area_id}|{kind}|g{self._cover_generation}",
                    provider=self.instance_id,
                    remotely_accessible=False,
                )
                for image_type, kind in ((ImageType.THUMB, "thumb"), (ImageType.FANART, "fanart"))
            ]
        )

    def _images(
        self, area_id: str, day: date, style_key: str | None
    ) -> UniqueList[MediaItemImage] | None:
        """Return the cover of an area, a day and a style, or None when covers are off."""
        if self._cover_mode() == COVER_OFF:
            return None
        path = (
            f"{COVER_PATH_PREFIX}|{area_id}|{day.isoformat()}|{style_key or ''}"
            f"|g{self._cover_generation}"
        )
        return UniqueList(
            [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=path,
                    provider=self.instance_id,
                    remotely_accessible=False,
                )
            ]
        )

    def _cover_mode(self) -> str:
        """Return what the covers are: off, procedural or AI."""
        mode = str(self.get_config_value(CONF_COVERS, COVER_PROCEDURAL) or COVER_PROCEDURAL)
        return mode if mode in {COVER_OFF, COVER_PROCEDURAL, COVER_AI} else COVER_PROCEDURAL

    def accepts_stored_image_path(self, path: str) -> bool:
        """Accept the covers of this provider when they are read back from stored data."""
        return path.startswith((f"{COVER_PATH_PREFIX}|", f"{ARTIST_PATH_PREFIX}|"))

    async def resolve_image(self, path: str) -> str | bytes | None:
        """
        Return the cover of an area, a day and a style, as PNG or JPEG bytes.

        :param path: `cover|<area id>|<day>|<style>|g<number of the redraw>`, as the albums and
            tracks name it; the form without the last part, which stored data may still hold, is
            accepted too.
        """
        parts = path.split("|")
        if parts[0] == ARTIST_PATH_PREFIX:
            return await self._artist_image(parts)
        if len(parts) not in {4, 5} or parts[0] != COVER_PATH_PREFIX:
            return None
        if len(parts) == 5 and not re.fullmatch(r"g\d+", parts[4]):
            return None
        area_id, day_text, style_key = parts[1:4]
        try:
            day = date.fromisoformat(day_text)
        except ValueError:
            return None
        return await self._cover(area_id, day, style_key or DEFAULT_STYLE)

    async def _artist_image(self, parts: list[str]) -> bytes | None:
        """
        Return the image of a room: drawn once, kept in the cache folder of the covers.

        :param parts: `artist|<area id>|thumb or fanart|g<number of the redraw>`, split at the bars.
        """
        if (
            len(parts) != 4
            or parts[2] not in {"thumb", "fanart"}
            or not re.fullmatch(r"g\d+", parts[3])
            or self._cover_mode() == COVER_OFF
        ):
            return None
        area_id, kind = parts[1], parts[2]
        try:
            catalog = await self._catalog_all()
        except Exception:
            self.logger.debug("No rooms for the image of %s", area_id, exc_info=True)
            catalog = Catalog()
        if area_id == WHOLE_HOUSE_ID:
            name = WHOLE_HOUSE_NAME
        elif area_id in catalog.areas:
            name = catalog.areas[area_id]
        else:
            return None
        room = room_kind(name, area_id, catalog.icons.get(area_id))
        # a new name or a new kind of room is a new image; the old file ages out of the cache
        slot = f"artist|{area_id}|{name}|{room}|{kind}"
        if kind == "fanart":
            return await self.covers.make(
                COVER_PROCEDURAL, slot, slot, lambda: artist_fanart(area_id, room), None
            )
        return await self.covers.make(
            COVER_PROCEDURAL, slot, slot, lambda: artist_thumb(area_id, name, room), None
        )

    async def _cover(self, area_id: str, day: date, style_key: str) -> bytes | None:
        """
        Return the cover of a day: drawn from its events, or made by the image engine.

        An engine cover is made in the background; until it is ready the drawn one is shown.
        """
        mode = self._cover_mode()
        if mode == COVER_OFF:
            return None
        style = get_style(style_key)
        area_name = WHOLE_HOUSE_NAME if area_id == WHOLE_HOUSE_ID else area_id
        try:
            catalog = await self._catalog_all()
            area_name = catalog.areas.get(area_id, area_name)
            infos = self._infos_for(area_id, catalog, day)
            if infos is None and area_id == WHOLE_HOUSE_ID:
                infos = catalog.all_infos()
            series = (
                await self._series_of(
                    area_id, infos, day, await self._timezone(), await self._now()
                )
                if infos
                else []
            )
        except Exception:
            self.logger.debug("No history for the cover of %s on %s", area_id, day, exc_info=True)
            series = []
        profile = profile_of(series)
        palette = derive_palette(style, day, self._scale, DayTraits(energy=profile.intensity))
        slot = f"{area_id}|{day.isoformat()}|{style.key}"
        title = album_name(area_name, day)
        look = CoverLook(
            style=str(
                self.get_config_value(CONF_COVER_STYLE, "music_assistant") or "music_assistant"
            ),
            custom_prompt=str(self.get_config_value(CONF_COVER_PROMPT, "") or ""),
            title_overlay=bool(self.get_config_value(CONF_COVER_TITLE, True)),  # noqa: FBT003,
        )
        setting = str(self.get_config_value(CONF_COVER_ENGINE, "global") or "global")
        self.covers.limiter.limit = int(
            self.get_config_value(CONF_COVER_LIMIT, DEFAULT_AI_PER_HOUR) or DEFAULT_AI_PER_HOUR
        )
        engine = await imageapi.resolve(self.mass, setting) if mode == COVER_AI else None
        key = json.dumps(
            [
                slot,
                title,
                look.style,
                look.custom_prompt,
                look.title_overlay,
                getattr(engine, "uid", ""),
            ]
        )

        async def generate() -> bytes:
            # the area is the title of the cover; the context is mood words and nothing else
            prompt = build_prompt(
                area_name,
                look,
                describe_day(
                    style.key, palette.root, palette.mode, profile.intensity, sorted(profile.kinds)
                ),
            )
            image = await imageapi.generate(self.mass, setting, prompt)
            return await asyncio.to_thread(render_with_title, image, look, title)

        return await self.covers.make(
            mode,
            slot,
            key,
            lambda: procedural_cover(profile, style.key, palette.root, palette.mode, slot, title),
            generate if engine is not None else None,
            pinned=self.pins.has(area_id, day),
        )

    def _album(
        self, area_id: str, area_name: str, day: date, facts: DayFacts | None = None
    ) -> Album:
        """
        Return the album of one day of an area.

        :param area_id: The room, or the whole house.
        :param area_name: The name of the room.
        :param day: The day.
        :param facts: What the day was like; the review says less without it.
        """
        item_id = album_id(area_id, day)
        return Album(
            item_id=item_id,
            provider=self.instance_id,
            name=album_name(area_name, day),
            artists=UniqueList([self._artist(area_id, area_name)]),
            year=day.year,
            album_type=AlbumType.ALBUM,
            metadata=MediaItemMetadata(
                genres={"Generative"},
                description=album_review(area_name, day, facts, self._language),
                images=self._images(area_id, day, self._offered_styles[0].key),
            ),
            provider_mappings={self._mapping(item_id)},
            favorite=True if item_id in self.membership.albums else None,
        )

    def _track(  # noqa: PLR0913, PLR0917
        self,
        ref: TrackRef,
        area_name: str,
        name: str,
        number: int,
        duration: int,
        description: str,
        version: str | None = None,
        facts: DayFacts | None = None,
    ) -> Track:
        """Return one track of an album, in the style of the reference."""
        item_id = track_id(ref.area_id, ref.day, ref.target, ref.style)
        style = get_style(ref.style)
        # with a single style offered there is nothing to tell apart, so no version is shown
        if version is None:
            version = style.version if len(self._offered_styles) > 1 else ""
        return Track(
            item_id=item_id,
            provider=self.instance_id,
            name=name,
            version=version,
            duration=duration,
            artists=UniqueList([self._artist(ref.area_id, area_name)]),
            album=self._album(ref.area_id, area_name, ref.day, facts),
            disc_number=1,
            track_number=number,
            metadata=MediaItemMetadata(
                genres=set(style.genres),
                description=description,
                images=self._images(ref.area_id, ref.day, ref.style or style.key),
            ),
            provider_mappings={self._mapping(item_id)},
            favorite=True if item_id in self.membership.tracks else None,
        )

    def _day_tracks(
        self,
        area_id: str,
        name: str,
        day: date,
        duration: int,
        facts: DayFacts | None,
    ) -> list[Track]:
        """
        Return the tracks of an album: the soundscape of the day in each style that is offered.

        Every track has a title of its own, made from the facts of the day and its style, and the
        description of its style.

        :param area_id: The room, or the whole house.
        :param name: The name of the room.
        :param day: The day.
        :param duration: The seconds of the day that have happened, in tracks of the shortest compression.
        :param facts: What the day was like, None when it could not be read.
        """
        lang = self._language
        room = name if area_id != WHOLE_HOUSE_ID else ("Huis" if lang == "nl" else "House")
        seed = f"{area_id}|{day.isoformat()}"
        # a soundscape needs a few hours of the day; the day of today has none of it in the morning
        scapes = duration * self._compression >= MIN_SCAPE_SECONDS_OF_DAY
        return [
            self._track(
                TrackRef(area_id, day, SCAPE_ID, style.key),
                name,
                make_title(facts, style.key, seed, room, lang),
                number,
                nominal_seconds(style.key, self._length_mode),
                track_description(style.key, facts, lang),
                facts=facts,
            )
            for number, style in enumerate(self._offered_styles, start=1)
            if scapes
        ]

    def _tracks_of(
        self,
        catalog: Catalog,
        area_id: str,
        day: date,
        duration: int,
        facts: DayFacts | None = None,
    ) -> list[Track]:
        """
        Return the tracks of any album: its soundscape in each style.

        :param catalog: The registries.
        :param area_id: The room, or the whole house.
        :param day: The day.
        :param duration: The seconds of the day that have happened at the shortest compression.
        :param facts: What the day was like, None when it could not be read.
        """
        if area_id == WHOLE_HOUSE_ID:
            return self._day_tracks(area_id, WHOLE_HOUSE_NAME, day, duration, facts)
        return self._day_tracks(area_id, catalog.areas[area_id], day, duration, facts)

    async def _facts(
        self, area_id: str, day: date, *, cached_only: bool = False
    ) -> DayFacts | None:
        """
        Return what a day of a room (or of the house) was like, for its titles; None when it cannot be read.

        A reading is kept for the hour, so that the title of today stays the same within an hour; a past day
        does not change. Home Assistant being away, or a day without history, never fails a listing.

        :param area_id: The room, or the whole house.
        :param day: The day.
        :param cached_only: Do not read the history of the day, only use a reading that is kept.
        """
        try:
            tz, now = await self._timezone(), await self._now()
            covered = covered_seconds(day, tz, now)
            catalog = await self._catalog_all()
            rooms = tuple(self._house_rooms(catalog)) if area_id == WHOLE_HOUSE_ID else (area_id,)
            key = (area_id, day, now.hour if covered < 86400 else -1, rooms)
            found = self._facts_kept.get(key)
            if found is not None:
                self._facts_kept.move_to_end(key)
                return found
            if cached_only or covered < MIN_SCAPE_SECONDS_OF_DAY:
                return None
            series: list[Series] = []
            for room in rooms:
                infos = self._infos_for(room, catalog, day)
                if infos:
                    series.extend(await self._series_of(room, infos, day, tz, now))
            found = await asyncio.to_thread(facts_of, series, day, covered)
        except Exception:
            self.logger.debug("Could not read %s on %s for its titles", area_id, day, exc_info=True)
            return None
        self._facts_kept[key] = found
        while len(self._facts_kept) > MAX_KEPT_FACTS:
            self._facts_kept.popitem(last=False)
        return found

    @staticmethod
    def _artist_names(catalog: Catalog) -> dict[str, str]:
        """Return the artists: the whole house first, then each area."""
        if not catalog.areas:
            return {}
        return {WHOLE_HOUSE_ID: WHOLE_HOUSE_NAME, **catalog.areas}

    # -- what can be browsed and what is in the library -------------------------------------

    def _pinned_scope(self, scope: str, day: date) -> dict[str, Any] | None:
        """Return the history that was pinned for an area (or the sun) on a day, if there is one."""
        if scope == "_sun":
            for area_id in self.membership.snapshot:
                pin = self.pins.read(area_id, day)
                if pin is not None and pin.complete and pin.sun is not None:
                    return dict(pin.sun["data"])
            return None
        pin = self.pins.read(scope, day)
        return dict(pin.history) if pin is not None and pin.complete else None

    async def _catalog_all(self) -> Catalog:
        """Return the catalog and, for areas that Home Assistant no longer has, what was pinned."""
        try:
            catalog = await self._get_catalog()
        except ProviderUnavailableError:
            if not self.membership.snapshot:
                raise
            catalog = Catalog()
        missing = {
            area_id: snap
            for area_id, snap in self.membership.snapshot.items()
            if area_id not in catalog.areas and area_id != WHOLE_HOUSE_ID
        }
        if not missing:
            return catalog
        areas = {**catalog.areas, **{area_id: snap["name"] for area_id, snap in missing.items()}}
        entities = {
            **catalog.entities,
            **{
                area_id: [entity_from_dict(item) for item in snap["entities"]]
                for area_id, snap in missing.items()
            },
        }
        return Catalog(areas, entities, catalog.sun, catalog.changed, catalog.icons)

    async def _pinned_house_days(self) -> set[date]:
        """Return the days that every room of the whole house has pinned."""
        rooms = set(self._house_rooms(await self._catalog_all()))
        pinned = [set(self.pins.days(area)) for area in self.membership.snapshot if area in rooms]
        return set.intersection(*pinned) if pinned else set()

    async def _browse_days(self, area_id: str) -> list[date]:
        """
        Return every day that an area can be played on, newest first.

        That is every day that Home Assistant still has (up to the setting) and every pinned day,
        however old.
        """
        _, _, tz, now = await self._days_and_catalog()
        found = {
            day
            for day in recent_days(now.date(), self._days)
            if playable(day, tz, now, self._compression)
        }
        if area_id == WHOLE_HOUSE_ID:
            found.update(await self._pinned_house_days())
        else:
            found.update(self.pins.days(area_id))
        return sorted(found, reverse=True)

    # -- the library -------------------------------------------------------------------------

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Browse the house: an artist folder per area, and in it an album per day that can be played.

        :param path: The path to browse, such as `entropy_fm://area/kitchen`.
        """
        parts = [part for part in path.split("://", 1)[-1].split("/") if part]
        catalog = await self._catalog_all()
        names = await self._offered(catalog)
        if not parts:
            return [
                BrowseFolder(
                    item_id=area_id,
                    provider=self.instance_id,
                    name=name,
                    path=f"{self.instance_id}://area/{area_id}",
                )
                for area_id, name in names.items()
            ]
        if parts[0] == "area" and len(parts) == 2 and parts[1] in names:
            return await self.get_artist_albums(parts[1])
        msg = "Invalid subpath"
        raise KeyError(msg)

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Return the items whose name contains the search query, on every day that can be played.

        :param search_query: The text to search for.
        :param media_types: The media types to include.
        :param limit: The maximum number of items per media type.
        """
        query = search_query.lower()
        result = SearchResults()
        catalog = await self._catalog_all()
        tz, now = await self._timezone(), await self._now()
        names = await self._offered(catalog)
        if MediaType.ARTIST in media_types:
            result.artists = [
                self._artist(area_id, name)
                for area_id, name in names.items()
                if query in name.lower()
            ][:limit]
        if MediaType.ALBUM in media_types:
            albums = [
                self._album(area_id, name, day, await self._facts(area_id, day, cached_only=True))
                for area_id, name in names.items()
                for day in await self._browse_days(area_id)
            ]
            result.albums = [album for album in albums if query in album.name.lower()][:limit]
        if MediaType.TRACK in media_types:
            found: list[Track] = []
            for area_id in names:
                for day in await self._browse_days(area_id):
                    duration = track_seconds(day, tz, now, self._compression)
                    facts = await self._facts(area_id, day, cached_only=True)
                    found.extend(
                        track
                        for track in self._tracks_of(catalog, area_id, day, duration, facts)
                        if query
                        in f"{track.name} {track.album.name if track.album else ''} {track.version}".lower()
                    )
                    if len(found) >= limit:
                        break
            result.tracks = found[:limit]
        return result

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Return every room, and the whole house: the artists of the library are always there."""
        for area_id, name in (await self._offered(await self._catalog_all())).items():
            yield self._artist(area_id, name)

    async def _library_days_of(self, area_id: str) -> list[date]:
        """
        Return the days of a room that are in the library, newest first.

        That is the last days of the setting, counted back from today (a window that rolls over at
        midnight), and every pinned day, which stays when it leaves the window or when Home
        Assistant has purged it.
        """
        _, _, tz, now = await self._days_and_catalog()
        found = {
            day
            for day in recent_days(now.date(), self._library_days)
            if playable(day, tz, now, self._compression)
        }
        if area_id == WHOLE_HOUSE_ID:
            found.update(await self._pinned_house_days())
        else:
            found.update(self.pins.days(area_id))
        return sorted(found, reverse=True)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Return the days of every room in the window, and every day that a heart or a kept room keeps."""
        for area_id, name in (await self._offered(await self._catalog_all())).items():
            for day in await self._library_days_of(area_id):
                yield self._album(area_id, name, day, await self._facts(area_id, day))

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Return every track of every album of the library: the soundscape of the day in each style that is offered."""
        catalog = await self._catalog_all()
        tz, now = await self._timezone(), await self._now()
        for area_id in await self._offered(catalog):
            for day in await self._library_days_of(area_id):
                duration = track_seconds(day, tz, now, self._compression)
                facts = await self._facts(area_id, day)
                for track in self._tracks_of(catalog, area_id, day, duration, facts):
                    yield track

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Return the artist with the given id."""
        names = self._artist_names(await self._catalog_all())
        if prov_artist_id not in names:
            raise MediaNotFoundError(f"Artist {prov_artist_id} not found")
        return self._artist(prov_artist_id, names[prov_artist_id])

    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Return the album of every day of an area that can be played; none for a room that is not offered."""
        catalog = await self._catalog_all()
        names = self._artist_names(catalog)
        if prov_artist_id not in names:
            raise MediaNotFoundError(f"Artist {prov_artist_id} not found")
        if prov_artist_id not in await self._offered(catalog):
            return []
        return [
            self._album(
                prov_artist_id,
                names[prov_artist_id],
                day,
                await self._facts(prov_artist_id, day, cached_only=True),
            )
            for day in await self._browse_days(prov_artist_id)
        ]

    async def get_album(self, prov_album_id: str) -> Album:
        """Return the album with the given id."""
        area_id, day, names = await self._check_album(prov_album_id)
        return self._album(area_id, names[area_id], day, await self._facts(area_id, day))

    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Return the tracks of the album with the given id."""
        area_id, day, _ = await self._check_album(prov_album_id)
        catalog = await self._catalog_all()
        duration = track_seconds(day, await self._timezone(), await self._now(), self._compression)
        return self._tracks_of(catalog, area_id, day, duration, await self._facts(area_id, day))

    async def get_track(self, prov_track_id: str) -> Track:
        """Return the track with the given id."""
        ref = await self._check_track(prov_track_id)
        catalog = await self._catalog_all()
        duration = track_seconds(
            ref.day, await self._timezone(), await self._now(), self._compression
        )
        wanted = track_id(ref.area_id, ref.day, ref.target, ref.style)
        facts = await self._facts(ref.area_id, ref.day)
        for track in self._tracks_of(catalog, ref.area_id, ref.day, duration, facts):
            if track.item_id == wanted:
                return track
        raise MediaNotFoundError(f"Track {prov_track_id} not found")

    async def _check_album(self, item_id: str) -> tuple[str, date, dict[str, str]]:
        """Return the parts of an album id after checking that the album exists."""
        try:
            area_id, day = parse_album_id(item_id)
        except ValueError as err:
            raise MediaNotFoundError(f"Album {item_id} not found") from err
        names = self._artist_names(await self._catalog_all())
        if area_id not in names or day not in await self._browse_days(area_id):
            raise MediaNotFoundError(f"Album {item_id} not found")
        return area_id, day, names

    async def _check_track(self, item_id: str) -> TrackRef:
        """Return the parts of a track id after checking that its album exists."""
        try:
            ref = parse_track_id(item_id)
        except ValueError as err:
            raise MediaNotFoundError(f"Track {item_id} not found") from err
        # a track is the soundscape of a room in a style; the voice tracks of earlier versions are gone
        if ref.style not in STYLES or ref.target not in TRACK_TARGETS:
            raise MediaNotFoundError(f"Track {item_id} not found")
        await self._check_album(album_id(ref.area_id, ref.day))
        return ref

    # -- pins: adding to the library keeps the days ------------------------------------------

    def _areas_of(self, area_id: str, catalog: Catalog) -> list[str]:
        """Return the areas whose days a thing of the library is made of."""
        return [area_id] if area_id != WHOLE_HOUSE_ID else self._house_rooms(catalog)

    async def _pin_day(self, area_id: str, day: date) -> bool:
        """
        Pin one area on one day: its event timeline is written to the storage of the server.

        A day that is not over yet is pinned as it is and pinned again when it is.

        :param area_id: The area.
        :param day: The day.
        :return: Whether the day is pinned.
        """
        async with self._pin_lock:
            existing = self.pins.read(area_id, day)
            if existing is not None and existing.complete:
                return True
            catalog = await self._catalog_all()
            infos = catalog.entities.get(area_id)
            tz, now = await self._timezone(), await self._now()
            if infos is None:
                return existing is not None
            complete = now.astimezone(tz).date() > day
            try:
                data = await self.history.day_data(area_id, infos, day, tz, now)
                sun = None
                if catalog.sun is not None:
                    sun = {
                        "entity": entity_to_dict(catalog.sun),
                        "data": await self.history.day_data("_sun", [catalog.sun], day, tz, now),
                    }
            except BaseHassClientError as err:
                self.logger.warning("Could not pin %s on %s: %s", area_id, day, err)
                return existing is not None
            if not any(item.get("t") for item in data.values()):
                self.logger.debug("Nothing to pin for %s on %s", area_id, day)
                return existing is not None
            names = self._artist_names(catalog)
            await asyncio.to_thread(
                self.pins.write,
                Pin(
                    area_id,
                    names.get(area_id, area_id),
                    day,
                    [entity_to_dict(info) for info in infos],
                    data,
                    sun,
                    complete=complete,
                ),
            )
            self.membership.snapshot[area_id] = {
                "name": names.get(area_id, area_id),
                "entities": [entity_to_dict(info) for info in infos],
            }
            return True

    async def _remember(self, key: str, pairs: list[tuple[str, date]]) -> None:
        """Pin the days of a thing and write down that it needs them."""
        pinned = [(area, day) for area, day in pairs if await self._pin_day(area, day)]
        self.membership.add(key, pinned)
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)

    async def set_favorite(
        self,
        prov_item_id: str,
        media_type: MediaType,
        favorite: bool | None,  # noqa: FBT001
    ) -> None:
        """
        Keep or release what the user hearted.

        A heart on an album or a track pins its day (for the whole house, the day of every room), so that it
        stays in the library however old it gets and after Home Assistant has purged it. A heart on a room keeps
        every day of that room. Anything else than a like (a dislike, or the state cleared) releases what the
        item kept; a pin is deleted when nothing else needs it, and the day then ages out of the library as
        every day does.

        :param prov_item_id: The id of the item on this provider.
        :param media_type: Its type.
        :param favorite: True for a like, False for a dislike, None when the state is cleared.
        """
        catalog = await self._catalog_all()
        keep = bool(favorite)
        try:
            if media_type == MediaType.ARTIST:
                await self._favorite_room(prov_item_id, catalog, keep=keep)
                return
            if media_type == MediaType.ALBUM:
                area_id, day = parse_album_id(prov_item_id)
                kept = self.membership.albums
            elif media_type == MediaType.TRACK:
                ref = parse_track_id(prov_item_id)
                if ref.target not in TRACK_TARGETS:
                    return
                area_id, day = ref.area_id, ref.day
                kept = self.membership.tracks
            else:
                return
        except ValueError:
            return
        if keep:
            kept.add(prov_item_id)
            await self._remember(
                prov_item_id, [(area, day) for area in self._areas_of(area_id, catalog)]
            )
            return
        kept.discard(prov_item_id)
        await self._release(prov_item_id)

    async def _favorite_room(self, area_id: str, catalog: Catalog, *, keep: bool) -> None:
        """Keep every day of a room that was hearted, or let go of it."""
        if keep:
            if area_id not in self._artist_names(catalog):
                return
            self.membership.followed[area_id] = (await self._now()).date().isoformat()
            await asyncio.to_thread(save_membership, self.storage_root, self.membership)
            await self._pin_followed()
            return
        self.membership.followed.pop(area_id, None)
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)
        await self._release_unkept_rooms()

    async def _release(self, key: str) -> None:
        """Forget what an item needs and delete the pins that nothing else needs."""
        freed = self.membership.remove(key)
        for area_id, pinned_day in freed:
            await asyncio.to_thread(self.pins.delete, area_id, date.fromisoformat(pinned_day))
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)

    async def _release_unkept_rooms(self) -> None:
        """Let go of the pins of rooms that are no longer in *Keep every day of these rooms*."""
        keep = self._keep_rooms
        freed: set[tuple[str, str]] = set()
        for key in [k for k in self.membership.needs if k.startswith(ARTIST_KEY)]:
            if key.removeprefix(ARTIST_KEY) not in keep:
                freed |= self.membership.remove(key)
        if not freed:
            return
        for area_id, day in freed:
            await asyncio.to_thread(self.pins.delete, area_id, date.fromisoformat(day))
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)

    async def _migrate_library_model(self) -> None:
        """
        Move the data of the earlier library model into the model of hearts.

        What was added to the library is what is hearted now: the albums, the tracks, and the rooms that were
        followed keep their pins and their days, and Music Assistant is told of the hearts at the next sync.
        The list of removed items is dropped (every day of the window is in the library now); it is gone from
        the file at the first save.
        """
        if not self.membership.legacy_removed:
            return
        self.membership.legacy_removed = False
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)
        self.logger.info("The library now follows hearts: the list of removed items is dropped")

    async def _migrate_ensemble_tracks(self) -> None:
        """
        Move the tracks that were added as songs or timelines to the soundscape of the same day and style.

        The soundscape is the only track now; the pins are of the day, so they stay as they are.
        """
        moved = False
        for item_id in sorted(self.membership.tracks):
            try:
                ref = parse_track_id(item_id)
            except ValueError:
                continue
            if ref.target not in LEGACY_TARGETS:
                continue
            new = track_id(ref.area_id, ref.day, SCAPE_ID, ref.style)
            self.membership.tracks.discard(item_id)
            self.membership.tracks.add(new)
            if item_id in self.membership.needs:
                self.membership.needs[new] = self.membership.needs.pop(item_id)
            moved = True
        if moved:
            await asyncio.to_thread(save_membership, self.storage_root, self.membership)

    async def _drop_voice_tracks(self) -> None:
        """
        Forget the tracks of single voices that an earlier version put in the library.

        A library holds rooms, their days and the ensemble of each day. A track of one entity or of
        one area inside the whole house is dropped with the pins that only it needed; the pins of
        the days that an album, an artist or an ensemble track needs stay.
        """
        stale = [item_id for item_id in self.membership.tracks if self._is_voice_track(item_id)]
        if not stale:
            return
        freed: set[tuple[str, str]] = set()
        for item_id in stale:
            self.membership.tracks.discard(item_id)
            freed |= self.membership.remove(item_id)
        for area_id, day in freed:
            await asyncio.to_thread(self.pins.delete, area_id, date.fromisoformat(day))
        await asyncio.to_thread(save_membership, self.storage_root, self.membership)
        self.logger.info("Dropped %d tracks of single voices from the library", len(stale))

    @staticmethod
    def _is_voice_track(item_id: str) -> bool:
        """Return whether a track id points at a single voice and not at an ensemble."""
        try:
            return parse_track_id(item_id).target not in TRACK_TARGETS
        except ValueError:
            return True

    async def _maintain(self) -> None:
        """Pin the days of the followed areas that are over, and the days that were pinned early."""
        await asyncio.sleep(30)
        while True:
            try:
                await self._pin_followed()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - keep the loop alive
                self.logger.warning("Pinning the days of the followed areas failed: %s", err)
            await asyncio.sleep(MAINTENANCE_SECONDS)

    async def _pin_followed(self) -> None:
        """Pin every day that a followed area has and that is not pinned yet or was pinned early."""
        catalog = await self._catalog_all()
        await self._release_unkept_rooms()
        keep = self._keep_rooms
        if not keep:
            return
        today = (await self._now()).date()
        for area_id in sorted(keep & set(await self._offered(catalog))):
            areas = self._areas_of(area_id, catalog)
            for day in await self._browse_days(area_id):
                pairs = [(area, day) for area in areas]
                known = {
                    tuple(pair) for pair in self.membership.needs.get(f"{ARTIST_KEY}{area_id}", [])
                }
                stale = any(
                    (pin := self.pins.read(area, day)) is not None
                    and not pin.complete
                    and day < today
                    for area in areas
                )
                if day >= today and not stale:
                    continue
                if stale:
                    for area in areas:
                        self.pins.delete(area, day)
                missing = any(not self.pins.has(area, day) for area in areas)
                if (
                    missing
                    or stale
                    or any((area, day.isoformat()) not in known for area, day in pairs)
                ):
                    await self._remember(f"{ARTIST_KEY}{area_id}", pairs)

    # -- streaming ---------------------------------------------------------------------------

    def _audio_key(self, ref: TrackRef, today: date, catalog: Catalog | None = None) -> str | None:
        """
        Return the key of the cached audio of a track, or None for today and when it is off.

        It holds everything that changes the sound, the length of the piece (full or condensed) and, for the
        whole house, the rooms it is made of.

        :param ref: The track.
        :param today: Today in the time zone of the house.
        :param catalog: The registries, which tell the rooms of the whole house.
        """
        if not self.audio_cache.enabled or ref.day >= today:
            return None
        rooms = (
            ",".join(sorted(self._house_rooms(catalog or self._catalog or Catalog())))
            if ref.area_id == WHOLE_HOUSE_ID
            else ""
        )
        return cache_key(
            ENGINE_VERSION,
            self.quality.key,
            track_id(ref.area_id, ref.day, ref.target, ref.style),
            self._compression,
            self._length_mode,
            rooms,
            self._scale,
            self._energy_offset,
            self._day_character,
            self._include_presence,
            self.get_config_value(CONF_ROLES, ""),
            self.get_config_value(CONF_EXCLUDE, ""),
        )

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return how the server gets the audio of a track: from the cache, or rendered live."""
        ref = await self._check_track(item_id)
        # a soundscape is as long as its day was busy: 10 to 22 minutes, known only from the day
        duration = (await self._scape_of(ref)).duration
        if duration < 1:
            raise MediaNotFoundError(f"Track {item_id} has no audio yet")
        key = self._audio_key(ref, (await self._now()).date(), await self._catalog_all())
        if key is not None and (cached := await asyncio.to_thread(self.audio_cache.get, key)):
            return StreamDetails(
                provider=self.instance_id,
                item_id=item_id,
                audio_format=AudioFormat(
                    content_type=ContentType.OGG, sample_rate=SAMPLE_RATE, channels=2
                ),
                media_type=MediaType.TRACK,
                stream_type=StreamType.LOCAL_FILE,
                path=str(cached),
                duration=duration,
                allow_seek=True,
                can_seek=True,
            )
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=SAMPLE_RATE,
                bit_depth=16,
                channels=2,
            ),
            media_type=MediaType.TRACK,
            stream_type=StreamType.CUSTOM,
            duration=duration,
            allow_seek=True,
            can_seek=True,
            data=key,
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """
        Yield the audio of a track as 16-bit stereo PCM, one second per chunk.

        A track of a past day that plays from the start to the end is also written to the audio
        cache in the background; a track that is stopped or fails leaves nothing behind.

        :param streamdetails: What get_stream_details returned for the track.
        :param seek_position: The second to start at.
        """
        ref = parse_track_id(streamdetails.item_id)
        int(streamdetails.duration or 0)
        # the pre-render waits for every stream that is being rendered
        self._streams += 1
        try:
            try:
                score = await self._build_score(ref)
            except BaseHassClientError as err:
                raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err
            key = streamdetails.data if isinstance(streamdetails.data, str) else None
            recorder = self.audio_cache.recorder(key) if key and seek_position == 0 else None
            chunks = render_pcm(score, self.quality, self._render_seed(ref), seek_position)
            finished = False
            try:
                while chunk := await asyncio.to_thread(next, chunks, b""):
                    if recorder is not None:
                        await recorder.feed(chunk)
                    yield chunk
                finished = True
            finally:
                chunks.close()
                if recorder is not None:
                    if finished:
                        await recorder.commit()
                    else:
                        await recorder.abort()
        finally:
            self._streams -= 1

    @staticmethod
    def _render_seed(ref: TrackRef) -> str:
        """Return the seed of the textures of a track, which is the same for a live stream and a pre-render."""
        return f"{ref.area_id}|{ref.day.isoformat()}|{ref.style}"

    # -- the pre-render ----------------------------------------------------------------------

    @property
    def _prerender_mode(self) -> str:
        """Return which past days are rendered ahead: off, pinned or all."""
        mode = str(self.get_config_value(CONF_PRERENDER, prerender.DEFAULT_PRERENDER))
        return mode if mode in prerender.MODES else prerender.DEFAULT_PRERENDER

    async def _prerender_loop(self) -> None:
        """Pre-render once after the start, and then shortly after every midnight of the house."""
        await asyncio.sleep(prerender.STARTUP_DELAY_SECONDS)
        if self._benchmark_task is not None:
            await asyncio.wait({self._benchmark_task})
        while True:
            try:
                await self._prerender_yesterday()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - keep the loop alive
                self.logger.warning("Pre-rendering the last day failed: %s", err)
            wait = prerender.seconds_until_run(
                await self._now(),
                prerender.MIDNIGHT_DELAY_SECONDS
                + random.uniform(0.0, prerender.MIDNIGHT_JITTER_SECONDS),  # noqa: S311
            )
            await asyncio.sleep(wait)

    async def _prerender_yesterday(self) -> int:
        """
        Render the tracks of yesterday that the audio cache does not have, one at a time.

        :return: How many were rendered.
        """
        mode = self._prerender_mode
        if mode == prerender.OFF or not self.audio_cache.enabled:
            return 0
        catalog = await self._catalog_all()
        day = (await self._now()).date() - timedelta(days=1)
        added = self.membership
        refs = prerender.choose_tracks(
            mode,
            day,
            rooms=await self._offered(catalog),
            pinned_rooms=self._keep_rooms,
            offered=[style.key for style in self._offered_styles],
            default_style=self._default_style.key,
            added_albums=added.albums,
            added_tracks=added.tracks,
        )
        done = 0
        for ref in refs:
            outcome = await self._prerender_one(ref)
            if outcome is None:
                break
            done += outcome
        if refs:
            self.logger.info("Pre-rendered %d of %d tracks of %s", done, len(refs), day)
        return done

    async def _prerender_wait(self) -> bool:
        """Wait while a live stream is rendering; return False when it went on too long, so that the track is dropped."""
        waited = 0.0
        while self._streams > 0:
            if waited >= prerender.GIVE_UP_SECONDS:
                return False
            await asyncio.sleep(prerender.POLL_SECONDS)
            waited += prerender.POLL_SECONDS
        return True

    async def _prerender_room_for(self, ref: TrackRef) -> tuple[str, float] | None:
        """Return the key and the length of a track that is worth rendering: not cached, not today, still readable."""
        key = self._audio_key(ref, (await self._now()).date(), await self._catalog_all())
        if key is None or self.audio_cache.path(key).exists():
            return None
        try:
            await self._check_track(track_id(ref.area_id, ref.day, ref.target, ref.style))
            return key, (await self._scape_of(ref)).duration
        except MediaNotFoundError, BaseHassClientError, ProviderUnavailableError:
            return None

    async def _prerender_one(self, ref: TrackRef) -> int | None:
        """
        Render one track into the audio cache.

        :param ref: The track.
        :return: 1 when it was rendered, 0 when it was skipped, None when the pre-render should stop.
        """
        wanted = await self._prerender_room_for(ref)
        if wanted is None:
            return 0
        key, duration = wanted
        size = await asyncio.to_thread(self.audio_cache.size)
        if not prerender.fits(size, self.audio_cache.limit, duration):
            self.logger.debug("Pre-render stops: the audio cache has no room for more")
            return None
        recorder = self.audio_cache.recorder(key) if await self._prerender_wait() else None
        if recorder is None:
            return 0
        if self._prerender_pool is None:
            self._prerender_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="entropy-prerender", initializer=_lower_priority
            )
        loop = asyncio.get_running_loop()
        chunks = None
        finished = False
        try:
            score = await self._build_score(ref)
            chunks = render_pcm(score, self.quality, self._render_seed(ref), 0)
            while chunk := await loop.run_in_executor(self._prerender_pool, next, chunks, b""):
                await recorder.feed(chunk)
                await asyncio.sleep(prerender.YIELD_SECONDS)
                if not await self._prerender_wait():
                    return None
            finished = True
        finally:
            if chunks is not None:
                chunks.close()
            if finished:
                await recorder.commit()
            else:
                await recorder.abort()
        self.logger.debug("Pre-rendered %s", track_id(ref.area_id, ref.day, ref.target, ref.style))
        return 1

    async def _series_of(
        self, scope: str, infos: list[EntityInfo], day: date, tz: tzinfo, now: datetime
    ) -> list[Series]:
        """Return the history of a group of entities on a day."""
        return await self.history.day_series(scope, infos, day, tz, now)

    async def _gather(self, ref: TrackRef) -> _Gathered:
        """Fetch what a track of a day is made of: the series, the whole day and the baselines."""
        catalog = await self._catalog_all()
        tz = await self._timezone()
        now = await self._now()
        series: list[Series] = []
        # everything the areas of the track did on the day, for the character of the day
        whole_day: list[Series] = []
        baselines: dict[str, EntityBaseline] = {}
        area_ids = self._house_rooms(catalog) if ref.area_id == WHOLE_HOUSE_ID else [ref.area_id]
        for area_id in area_ids:
            infos = self._infos_for(area_id, catalog, ref.day)
            if infos is None:
                raise MediaNotFoundError(f"Area {area_id} not found")
            # the whole area is read at once so that its tracks share one cached request
            fetched = await self._series_of(area_id, infos, ref.day, tz, now)
            wanted = {info.entity_id for info in infos}
            series.extend(item for item in fetched if item.entity_id in wanted)
            whole_day.extend(fetched)
            if self._day_character:
                baselines.update(await self._baselines(area_id, infos, ref, tz, now))
        if catalog.sun is not None:
            series.extend(await self._series_of("_sun", [catalog.sun], ref.day, tz, now))
        # every sensor is measured against its own previous days, and the dead ones play nothing
        series, dropped = rescale(series, baselines)
        whole_day, _ = rescale(whole_day, baselines)
        if dropped:
            self.logger.debug("Left out %d sensors that say nothing: %s", len(dropped), dropped)
        return _Gathered(series, whole_day, baselines, covered_seconds(ref.day, tz, now))

    async def _scape_of(self, ref: TrackRef) -> _Scape:
        """
        Return the soundscape of a day in a style: how the day was read and the length of its audio.

        A past day is kept, so that the length of a soundscape and its audio are made from one reading; today
        is read again every ten minutes of the day, and a soundscape of today needs three hours of it.
        """
        covered = covered_seconds(ref.day, await self._timezone(), await self._now())
        if covered < MIN_SCAPE_SECONDS_OF_DAY:
            raise MediaNotFoundError("There is not enough of this day yet for a soundscape")
        rooms = (
            tuple(self._house_rooms(await self._catalog_all()))
            if ref.area_id == WHOLE_HOUSE_ID
            else ()
        )
        mode = self._length_mode
        key = (
            ref.area_id,
            ref.day,
            ref.style,
            int(covered // 600) if covered < 86400 else -1,
            mode,
            rooms,
        )
        found = self._scapes.get(key)
        if found is not None:
            self._scapes.move_to_end(key)
            return found
        gathered = await self._gather(ref)
        style = get_style(ref.style)
        seed = f"{ref.area_id}|{ref.day.isoformat()}|{ref.target}|{ref.style}"
        palette = None
        anomalies: list[tuple[float, float, str]] = []
        if self._day_character:
            traits = measure_traits(gathered.whole_day, gathered.baselines, gathered.covered)
            palette = derive_palette(style, ref.day, self._scale, traits)
            featured = Surprise(gathered.series, gathered.baselines).featured(
                0.0, 1.0, min(gathered.covered, 86400.0)
            )
            anomalies = [(t, score, entity) for t, score, entity in featured if score >= 0.5]

        def make() -> _Scape:
            analysis = analysis_of(gathered.series, gathered.covered, anomalies)
            return _Scape(gathered, analysis, palette, seed, duration_of(analysis, ref.style, mode))

        found = await asyncio.to_thread(make)
        self._scapes[key] = found
        while len(self._scapes) > MAX_KEPT_SCAPES:
            self._scapes.popitem(last=False)
        return found

    async def _build_score(self, ref: TrackRef) -> Score:
        """Fetch what a track needs and let the generative system of its style write the score, in a thread."""
        found = await self._scape_of(ref)
        score, _ = await asyncio.to_thread(
            scape_score,
            found.gathered.series,
            found.analysis,
            ref.style,
            found.seed,
            self._scale,
            day=ref.day,
            palette=found.palette,
            quality=self.quality,
            energy_offset=self._energy_offset,
            area_key=ref.area_id,
            length_mode=self._length_mode,
        )
        return score

    def _infos_for(self, area_id: str, catalog: Catalog, day: date) -> list[EntityInfo] | None:
        """Return the entities of an area on a day: those of its pin when it has one, else today's."""
        pin = self.pins.read(area_id, day)
        if pin is not None and pin.complete:
            return [entity_from_dict(item) for item in pin.entities]
        return catalog.entities.get(area_id)

    async def _baselines(
        self, area_id: str, infos: list[EntityInfo], ref: TrackRef, tz: tzinfo, now: datetime
    ) -> dict[str, EntityBaseline]:
        """Return what the entities of an area usually do; without a baseline there is no surprise."""
        try:
            return await self.history.baselines(area_id, infos, ref.day, tz, now)
        except BaseHassClientError as err:
            self.logger.debug("No baseline for %s: %s", area_id, err)
            return {}
