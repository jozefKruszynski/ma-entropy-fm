"""The Entropy FM provider: Home Assistant history served as generative soundscapes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import threading
from collections import OrderedDict
from collections.abc import AsyncGenerator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hass_client import HomeAssistantClient
from hass_client.exceptions import AuthenticationFailed, BaseHassClientError
from hass_client.utils import get_websocket_url
from music_assistant_models.config_entries import (
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
    MusicAssistantError,
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

from music_assistant.helpers.datetime import utc
from music_assistant.models.music_provider import MusicProvider

from . import prerender
from .audiocache import AudioCache, cache_key
from .autotune import benchmark, heaviest_style
from .baseline import EntityBaseline
from .character import DayCharacter, measure_character
from .constants import (
    ANOMALY_MIN_SCORE,
    ARTIST_AREA_INDEX,
    ARTIST_FIGURE_INDEX,
    ARTIST_KEY,
    ARTIST_KIND_INDEX,
    ARTIST_PATH_PARTS,
    ARTIST_PATH_PREFIX,
    ARTIST_PROFILE_DAYS,
    AUDIO_BIT_DEPTH,
    AUDIO_CHANNELS,
    AUTO_EXPIRATION,
    BROWSE_AREA_FOLDER,
    CACHE_CATEGORY_AUTO,
    CACHE_CATEGORY_FACTS,
    CATALOG_TTL,
    CATEGORY_ADVANCED,
    CHECKSUM_LENGTH,
    CONF_AUDIO_CACHE_MB,
    CONF_DAY_CHARACTER,
    CONF_DAYS,
    CONF_EXCLUDE,
    CONF_INCLUDE_PRESENCE,
    CONF_KEEP_ROOMS,
    CONF_QUALITY,
    CONF_ROLES,
    CONF_ROOMS,
    CONF_STYLES,
    CONF_TOKEN,
    CONF_URL,
    CONF_USE_PLUGIN,
    CONF_VERIFY_SSL,
    COVER_AREA_INDEX,
    COVER_DAY_INDEX,
    COVER_GENERATION,
    COVER_LIVE_INDEX,
    COVER_LIVE_MARK,
    COVER_PATH_PARTS,
    COVER_PATH_PREFIX,
    COVER_STYLE_INDEX,
    DEFAULT_AUDIO_CACHE_MB,
    DEFAULT_DAYS,
    ENGINE_VERSION,
    ERROR_CONNECTION_MISSING,
    ERROR_LOGIN_REFUSED,
    ERROR_PLUGIN_MISSING,
    FACTS_SCHEMA,
    FACTS_TODAY_EXPIRATION,
    GENERATION_PATTERN,
    HASS_DOMAIN,
    HOME_KEY,
    HOME_MODE,
    HOUSE_ROOM_NAMES,
    MAINTENANCE_SECONDS,
    MAINTENANCE_START_SECONDS,
    MAX_AUDIO_CACHE_MB,
    MAX_CONCURRENT_STREAMS,
    MAX_COVER_DRAWS,
    MAX_DAYS,
    MAX_KEPT_SCAPES,
    MIN_SCAPE_SECONDS_OF_DAY,
    OFFERED_FILE,
    PAST_DAY_EXPIRATION,
    PLUGIN_WAIT_SECONDS,
    RECONNECT_FACTOR,
    RECONNECT_FIRST_SECONDS,
    RECONNECT_MAX_SECONDS,
    SCAPE_ID,
    SECONDS_BUCKET,
    STORAGE_FOLDER,
    SUN_SCOPE,
    TIME_ZONE_RETRY_SECONDS,
    TRACK_TARGETS,
    WHOLE_HOUSE_ID,
    WHOLE_HOUSE_NAME,
)
from .covers import (
    album_style,
    procedural_cover,
    profile_of,
)
from .day_analysis import DayAnalysis
from .hass_access import HASS_ERRORS, HassAccess
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
)
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
from .plan import PieceOptions, condensed_seconds
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
from .roomart import (
    SHAPES,
    RoomProfile,
    artist_fanart,
    artist_thumb,
    figure_shape,
    seeded_profile,
    week_profile,
)
from .scape import analysis_of, scape_score
from .score import Score
from .styles import DEFAULT_STYLE, STYLES, Style, get_style
from .surprise import Surprise
from .synth import NOTE_NAMES, SAMPLE_RATE, Scale
from .timeline import DAY_SECONDS, EntityInfo, Series
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
    from music_assistant.helpers.json import SerializableType
    from music_assistant.providers.hass import HomeAssistantProvider

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
# what a pass of a background loop fails with, which the loop logs and survives
BACKGROUND_ERRORS = (MusicAssistantError, BaseHassClientError, OSError, ValueError)


@dataclass
class _Gathered:
    """What a track of a day is made of."""

    series: list[Series]
    whole_day: list[Series]
    baselines: dict[str, EntityBaseline]
    covered: float
    # how busy the earlier days of the same rooms were, as `character.raw_energy` reads them
    week: list[float] = field(default_factory=list)


@dataclass
class _Scape:
    """The soundscape of a day in a style: what it was made from, how the day was read and how long it is."""

    gathered: _Gathered
    analysis: DayAnalysis
    palette: DayPalette | None
    seed: str
    duration: int
    # None when the day-character setting is off: the user's key stays and nothing is picked by the day
    character: DayCharacter | None = None


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
    _tz_retry_at = 0.0
    _auto_quality: Quality = STANDARD
    _benchmark_task: asyncio.Task[None] | None = None
    _loaded: bool = False
    _prerender_pool: ThreadPoolExecutor | None = None
    # how many live streams are being rendered now, which the pre-render waits for; the event is set while none is
    _streams: int = 0
    _no_streams: asyncio.Event

    @property
    def is_streaming_provider(self) -> bool:
        """
        Return False: the soundscapes are made from this house, not fetched from a service.

        A library sync then removes what is no longer listed, such as a day that left the window
        unpinned.
        """
        return False

    @property
    def max_concurrent_streams(self) -> int:
        """Return how many soundscapes may play at once; each is rendered on the CPU of the server."""
        return MAX_CONCURRENT_STREAMS

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
            *await self._room_entries(),
            ConfigEntry(
                key=CONF_QUALITY,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.STRING,
                default_value=DEFAULT_QUALITY,
                required=False,
                options=[ConfigValueOption(AUTO), *(ConfigValueOption(key) for key in QUALITIES)],
            ),
            ConfigEntry(
                key=CONF_AUDIO_CACHE_MB,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.INTEGER,
                default_value=DEFAULT_AUDIO_CACHE_MB,
                range=(0, MAX_AUDIO_CACHE_MB),
                required=False,
            ),
            ConfigEntry(
                key=CONF_STYLES,
                type=ConfigEntryType.STRING,
                default_value=list(STYLES),
                required=False,
                multi_value=True,
                options=[ConfigValueOption(key) for key in STYLES],
            ),
            ConfigEntry(
                key=CONF_DAY_CHARACTER,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.BOOLEAN,
                default_value=True,
                required=False,
            ),
            ConfigEntry(
                key=CONF_INCLUDE_PRESENCE,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.BOOLEAN,
                default_value=False,
                required=False,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_ROLES,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.STRING,
                default_value="",
                required=False,
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_EXCLUDE,
                category=CATEGORY_ADVANCED,
                type=ConfigEntryType.STRING,
                default_value="",
                required=False,
                advanced=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Connect to Home Assistant, through the core plugin or with the URL and token."""
        self._catalog_lock = asyncio.Lock()
        self._no_streams = asyncio.Event()
        self._no_streams.set()
        if self.get_config_value(CONF_USE_PLUGIN, default=True):
            await self._wait_for_plugin()
            self.hass = HassAccess(plugin_getter=self._hass_plugin)
        else:
            self._own_client = await self._connect_own_client()
            self.hass = HassAccess(client=self._own_client)
            self._listen_task = self.mass.create_task(self._listen(self._own_client))
        self.storage_root = str(Path(self.mass.storage_path) / STORAGE_FOLDER)
        self.pins = PinStore(self.storage_root)
        self.membership = await asyncio.to_thread(load_membership, self.storage_root)
        self._loaded = True
        # numpy analysis objects that cannot be serialised, so they are not in the cache of the server
        self._scapes: OrderedDict[tuple[str, date, str, int, tuple[str, ...]], _Scape] = (
            OrderedDict()
        )
        self.history = HistoryStore(
            self.hass, self.mass.cache, self.instance_id, self.logger, self._pinned_scope
        )
        # megabytes of rendered audio, kept as files: the cache of the server only holds JSON
        self.audio_cache = AudioCache(
            str(Path(self.mass.cache_path) / STORAGE_FOLDER),
            int(self.get_config_value(CONF_AUDIO_CACHE_MB, DEFAULT_AUDIO_CACHE_MB) or 0),
        )
        self._pin_lock = asyncio.Lock()
        self._cover_draws = asyncio.Semaphore(MAX_COVER_DRAWS)
        self._benchmark_task = self.mass.create_task(self._pick_quality())
        self.mass.call_later(MAINTENANCE_START_SECONDS, self._maintain, task_id=self._maintain_id)
        self.mass.call_later(
            prerender.STARTUP_DELAY_SECONDS, self._prerender, task_id=self._prerender_id
        )

    async def loaded_in_mass(self) -> None:
        """Ask for a library sync in the background when the rooms that are offered differ from last time."""
        await super().loaded_in_mass()
        self.mass.create_task(self._ask_for_sync(), task_id=self._sync_id)

    async def get_diagnostics(self) -> dict[str, SerializableType] | None:
        """Return why each entity has its voice, so that a surprising sound can be explained."""
        try:
            catalog = await self._get_catalog()
        except ProviderUnavailableError as err:
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
        self.mass.cancel_task(self._sync_id)
        for task_id in (self._maintain_id, self._prerender_id):
            self.mass.cancel_timer(task_id)
            self.mass.cancel_task(task_id)
        if self._benchmark_task is not None:
            self._benchmark_task.cancel()
            self._benchmark_task = None
        if self._prerender_pool is not None:
            self._prerender_pool.shutdown(wait=False, cancel_futures=True)
            self._prerender_pool = None
        if self._listen_task is not None:
            self._listen_task.cancel()
            self._listen_task = None
        if self._own_client is not None:
            await self._own_client.disconnect()
            self._own_client = None

    async def sync_if_rooms_changed(self) -> bool:
        """
        Ask the server to sync this source when the rooms that are offered differ from last time.

        The server only notices a change of the offered rooms at its next sync, which also takes
        the items of a room that left out of the library views. It is a plain request to the music
        controller for this instance; the library itself is never written. The rooms of the last
        request are kept in our own storage. Nothing happens when Home Assistant cannot tell the
        rooms, or when no room is offered.

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
        wanted = sorted(offered)
        path = Path(self.storage_root) / OFFERED_FILE
        try:
            known = json.loads(await asyncio.to_thread(path.read_text))
        except OSError, ValueError:
            known = None
        if known == wanted:
            return False
        self.logger.info("The rooms that are offered changed, syncing the library")
        await self.mass.music.start_sync(providers=[self.instance_id])
        await asyncio.to_thread(write_atomic, path, json.dumps(wanted).encode())
        return True

    @property
    def quality(self) -> Quality:
        """Return the render tier: the one the user chose, or the one that Auto picked."""
        chosen = str(self.get_config_value(CONF_QUALITY, DEFAULT_QUALITY) or DEFAULT_QUALITY)
        return self._auto_quality if chosen == AUTO else get_quality(chosen)

    def accepts_stored_image_path(self, path: str) -> bool:
        """Accept the covers of this provider when they are read back from stored data."""
        return path.startswith((f"{COVER_PATH_PREFIX}|", f"{ARTIST_PATH_PREFIX}|"))

    async def resolve_image(self, path: str) -> str | bytes | None:
        """
        Return the cover of an area, a day and a style, as PNG or JPEG bytes.

        :param path: `cover|<area id>|<day>|<style>|g<cover generation>`, as the albums and
            tracks name it, with `live` before the last part while the day is going on.
        """
        try:
            return await self._image_of(path.split("|"))
        except BaseHassClientError as err:
            raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err

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
        if parts[0] == BROWSE_AREA_FOLDER and len(parts) == 2 and parts[1] in names:
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
        kept = (
            await self._prefetch()
            if {MediaType.ALBUM, MediaType.TRACK} & set(media_types)
            else None
        )
        if MediaType.ARTIST in media_types:
            result.artists = [
                self._artist(area_id, name)
                for area_id, name in names.items()
                if query in name.lower()
            ][:limit]
        if MediaType.ALBUM in media_types:
            albums = [
                self._album(
                    area_id, name, day, await self._facts(area_id, day, cached_only=True, kept=kept)
                )
                for area_id, name in names.items()
                for day in await self._browse_days(area_id)
            ]
            result.albums = [album for album in albums if query in album.name.lower()][:limit]
        if MediaType.TRACK in media_types:
            found: list[Track] = []
            for area_id in names:
                for day in await self._browse_days(area_id):
                    enough = playable(day, tz, now)
                    facts = await self._facts(area_id, day, cached_only=True, kept=kept)
                    found.extend(
                        track
                        for track in self._tracks_of(catalog, area_id, day, enough, facts)
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

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """
        Return the days of every room in the window, and every day that a heart or a kept room keeps.

        :raises ProviderUnavailableError: When Home Assistant does not answer, so that no album is listed
            with a plain review that the server would then keep.
        """
        catalog = await self._catalog_all()
        names = await self._offered(catalog)
        days = {area_id: await self._browse_days(area_id) for area_id in names}
        kept = await self._prefetch()
        await self._prime_history(days, catalog, kept)
        for area_id, name in names.items():
            for day in days[area_id]:
                facts = await self._facts(area_id, day, kept=kept, strict=True)
                yield self._album(area_id, name, day, facts)

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """
        Return every track of every album of the library: the soundscape of the day in each style that is offered.

        :raises ProviderUnavailableError: When Home Assistant does not answer, so that no track is listed
            with a plain title that the server would then keep.
        """
        catalog = await self._catalog_all()
        tz, now = await self._timezone(), await self._now()
        names = await self._offered(catalog)
        days = {area_id: await self._browse_days(area_id) for area_id in names}
        kept = await self._prefetch()
        await self._prime_history(days, catalog, kept)
        for area_id in names:
            for day in days[area_id]:
                enough = playable(day, tz, now)
                facts = await self._facts(area_id, day, kept=kept, strict=True)
                for track in self._tracks_of(catalog, area_id, day, enough, facts):
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
        kept = await self._prefetch()
        return [
            self._album(
                prov_artist_id,
                names[prov_artist_id],
                day,
                await self._facts(prov_artist_id, day, cached_only=True, kept=kept),
            )
            for day in await self._browse_days(prov_artist_id)
        ]

    async def get_album(self, prov_album_id: str) -> Album:
        """Return the album with the given id."""
        area_id, day, names = await self._check_album(prov_album_id)
        facts = await self._facts(area_id, day, strict=True)
        return self._album(area_id, names[area_id], day, facts, full=True)

    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Return the tracks of the album with the given id."""
        area_id, day, _ = await self._check_album(prov_album_id)
        catalog = await self._catalog_all()
        enough = playable(day, await self._timezone(), await self._now())
        facts = await self._facts(area_id, day, strict=True)
        return self._tracks_of(catalog, area_id, day, enough, facts)

    async def get_track(self, prov_track_id: str) -> Track:
        """Return the track with the given id."""
        ref = await self._check_track(prov_track_id)
        catalog = await self._catalog_all()
        enough = playable(ref.day, await self._timezone(), await self._now())
        facts = await self._facts(ref.area_id, ref.day, strict=True)
        for track in self._tracks_of(catalog, ref.area_id, ref.day, enough, facts, full=True):
            if track.item_id == ref.item_id:
                return track
        raise MediaNotFoundError(f"Track {prov_track_id} not found")

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
        if media_type == MediaType.ARTIST:
            await self._favorite_room(prov_item_id, catalog, keep=keep)
            return
        try:
            if media_type == MediaType.ALBUM:
                area_id, day = parse_album_id(prov_item_id)
                kept = self.membership.albums
            elif media_type == MediaType.TRACK:
                ref = self._parse_track(prov_item_id) if keep else parse_track_id(prov_item_id)
                area_id, day = ref.area_id, ref.day
                kept = self.membership.tracks
            else:
                return
        except ValueError as err:
            raise MediaNotFoundError(f"Item {prov_item_id} not found") from err
        if not keep:
            kept.discard(prov_item_id)
            await self._release(prov_item_id)
            return
        # the area comes from the client: only a room of the house gets a pin
        if area_id not in self._artist_names(catalog):
            raise MediaNotFoundError(f"Item {prov_item_id} not found")
        kept.add(prov_item_id)
        await self._remember(
            prov_item_id, [(area, day) for area in self._areas_of(area_id, catalog)]
        )

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return how the server gets the audio of a track: from the cache, or rendered live."""
        ref = await self._check_track(item_id)
        try:
            duration = (await self._scape_of(ref)).duration
        except BaseHassClientError as err:
            raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err
        if duration < 1:
            raise MediaNotFoundError(f"Track {item_id} has no audio yet")
        key = self._audio_key(ref, (await self._now()).date(), await self._catalog_all())
        if key is not None and (cached := await asyncio.to_thread(self.audio_cache.get, key)):
            return StreamDetails(
                provider=self.instance_id,
                item_id=item_id,
                audio_format=AudioFormat(
                    content_type=ContentType.OGG, sample_rate=SAMPLE_RATE, channels=AUDIO_CHANNELS
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
                bit_depth=AUDIO_BIT_DEPTH,
                channels=AUDIO_CHANNELS,
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
        # the pre-render waits for every stream that is being rendered
        self._stream_started()
        try:
            try:
                score = await self._build_score(ref)
            except BaseHassClientError as err:
                raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err
            key = streamdetails.data if isinstance(streamdetails.data, str) else None
            recorder = await self.audio_cache.recorder(key) if key and seek_position == 0 else None
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
            self._stream_ended()

    async def _image_of(self, parts: list[str]) -> bytes | None:
        """
        Return the image of a split cover or artist path, None for a path that is not served.

        :param parts: The path, split at the bars.
        """
        if parts[0] == ARTIST_PATH_PREFIX:
            return await self._artist_image(parts)
        if len(parts) > COVER_LIVE_INDEX and parts[COVER_LIVE_INDEX] == COVER_LIVE_MARK:
            del parts[COVER_LIVE_INDEX]
        if len(parts) != COVER_PATH_PARTS or parts[0] != COVER_PATH_PREFIX:
            return None
        if not re.fullmatch(GENERATION_PATTERN, parts[-1]):
            return None
        area_id, day_text, style_key = (
            parts[COVER_AREA_INDEX],
            parts[COVER_DAY_INDEX],
            parts[COVER_STYLE_INDEX],
        )
        try:
            day = date.fromisoformat(day_text)
        except ValueError:
            return None
        try:
            await self._check_album(album_id(area_id, day))
            return await self._cover(area_id, day, style_key or self._default_style.key)
        except MediaNotFoundError:
            return None

    async def _room_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the settings that pick the rooms: which are offered, what the house is made of, which are kept."""
        names: dict[str, str] = {}
        # the settings are also asked for before the provider has connected, when there is nothing to pick yet
        if self._loaded:
            with contextlib.suppress(ProviderUnavailableError):
                names = self._artist_names(await self._catalog_all())
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
                key=CONF_KEEP_ROOMS,
                type=ConfigEntryType.STRING,
                multi_value=True,
                default_value=[],
                required=False,
                options=keep_options,
            ),
        )

    async def _wait_for_plugin(self) -> None:
        """Wait a little for the core Home Assistant plugin, and fail clearly without it."""
        if self._hass_plugin() is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self.mass.get_provider_ready_event(HASS_DOMAIN).wait(), PLUGIN_WAIT_SECONDS
                )
        if self._hass_plugin() is not None:
            return
        msg = (
            "Entropy FM needs the Home Assistant plugin of Music Assistant: add and connect "
            "it under Settings > Providers, or turn off 'Use the Home Assistant plugin' and "
            "enter a URL and a long-lived access token"
        )
        raise SetupFailedError(
            msg, translation_key=ERROR_PLUGIN_MISSING, translation_owner=self.translation_owner
        )

    async def _connect_own_client(self) -> HomeAssistantClient:
        """Connect with the URL and the token from the settings."""
        url = str(self.get_config_value(CONF_URL, "") or "").strip()
        token = str(self.get_config_value(CONF_TOKEN, "") or "").strip()
        if not url or not token:
            msg = "Enter the URL and a long-lived access token of Home Assistant"
            raise SetupFailedError(
                msg,
                translation_key=ERROR_CONNECTION_MISSING,
                translation_owner=self.translation_owner,
            )
        verify = bool(self.get_config_value(CONF_VERIFY_SSL, default=True))
        session = self.mass.http_session if verify else self.mass.http_session_no_ssl
        client = HomeAssistantClient(get_websocket_url(url), token, session)
        try:
            await client.connect()
        except AuthenticationFailed as err:
            raise LoginFailed(
                "Home Assistant did not accept the access token",
                translation_key=ERROR_LOGIN_REFUSED,
                translation_owner=self.translation_owner,
            ) from err
        except BaseHassClientError as err:
            raise SetupFailedError(str(err) or err.__class__.__name__) from err
        return client

    async def _listen(self, client: HomeAssistantClient) -> None:
        """Process the messages of our own connection; when it ends, connect again until the provider unloads."""
        while True:
            try:
                await client.start_listening()
                reason = "the connection was closed"
            except BaseHassClientError as err:
                reason = str(err) or err.__class__.__name__
            self.logger.warning("Connection to Home Assistant lost: %s", reason)
            with contextlib.suppress(*HASS_ERRORS, OSError):
                await client.disconnect()
            client = await self._reconnect()
            self._own_client = client
            self.hass.set_client(client)
            self.logger.info("Connected to Home Assistant again")

    async def _reconnect(self) -> HomeAssistantClient:
        """Connect again, waiting longer after each failure, for as long as it takes."""
        delay = RECONNECT_FIRST_SECONDS
        while True:
            await asyncio.sleep(delay)
            try:
                return await self._connect_own_client()
            except (SetupFailedError, LoginFailed, OSError) as err:
                self.logger.debug("Home Assistant is still away: %s", err)
            delay = min(delay * RECONNECT_FACTOR, RECONNECT_MAX_SECONDS)

    @property
    def _maintain_id(self) -> str:
        """Return the id of the timer and the task of the maintenance."""
        return f"{self.instance_id}_maintain"

    @property
    def _sync_id(self) -> str:
        """Return the id of the task that asks for the library sync."""
        return f"{self.instance_id}_sync"

    async def _ask_for_sync(self) -> None:
        """Ask for a library sync when the offered rooms changed; a failure is logged and not raised."""
        try:
            await self.sync_if_rooms_changed()
        except MusicAssistantError, OSError:
            self.logger.warning(
                "Could not ask for a library sync after the rooms changed", exc_info=True
            )

    @property
    def _prerender_id(self) -> str:
        """Return the id of the timer and the task of the pre-render."""
        return f"{self.instance_id}_prerender"

    def _hass_plugin(self) -> HomeAssistantProvider | None:
        """Return the core Home Assistant plugin when it is loaded; it can reload, so it is asked every time."""
        return cast("HomeAssistantProvider | None", self.mass.get_provider(HASS_DOMAIN))

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
    def _language(self) -> str:
        """Return the language of titles and descriptions: the one of the server when it is offered, else English."""
        try:
            language = str(self.mass.metadata.preferred_language)
        except AttributeError, KeyError:
            return DEFAULT_LANGUAGE
        return language if language in LANGUAGES else DEFAULT_LANGUAGE

    def _is_selected(self, area_id: str) -> bool:
        """Return whether a room is ticked in *Rooms* (every room is, when none is)."""
        selected = self._selected_rooms
        return not selected or area_id in selected

    def _house_rooms(self, catalog: Catalog) -> list[str]:
        """Return the rooms that the whole house is made of: the ticked ones, or every room when none is ticked."""
        return [area_id for area_id in catalog.areas if self._is_selected(area_id)]

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
        if rooms:
            tz, now = await self._timezone(), await self._now()
            first = recent_days(now.date(), self._days)[-1]
            quiet = quiet_areas(catalog, day_bounds(first, tz)[0])
            rooms = {area_id: name for area_id, name in rooms.items() if area_id not in quiet}
        return {WHOLE_HOUSE_ID: WHOLE_HOUSE_NAME, **rooms} if rooms else {}

    @property
    def _days(self) -> int:
        """Return how many days back the library goes."""
        return max(1, min(MAX_DAYS, int(self.get_config_value(CONF_DAYS, DEFAULT_DAYS) or 1)))

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
        """Return the key and mode that the days move away from."""
        return Scale(NOTE_NAMES.index(HOME_KEY), HOME_MODE)

    @property
    def _offered_styles(self) -> list[Style]:
        """Return the styles the library offers, never none."""
        chosen = self.get_config_value(CONF_STYLES, list(STYLES))
        keys = [str(key) for key in chosen] if isinstance(chosen, list | tuple) else [str(chosen)]
        return [STYLES[key] for key in STYLES if key in keys] or [get_style(DEFAULT_STYLE)]

    @property
    def _default_style(self) -> Style:
        """Return the first style that is offered, which the pre-render and the length of a day go by."""
        return self._offered_styles[0]

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

    async def _timezone(self) -> tzinfo:
        """Return the time zone of the house; the one of this server for a while when Home Assistant gives none."""
        loop_time = asyncio.get_running_loop().time()
        if self._tz is None and loop_time >= self._tz_retry_at:
            self._tz = await self.hass.time_zone()
            if self._tz is None:
                self._tz_retry_at = loop_time + TIME_ZONE_RETRY_SECONDS
        return self._tz or utc().astimezone().tzinfo or UTC

    async def _now(self) -> datetime:
        """Return the current time in the time zone of the house."""
        return utc().astimezone(await self._timezone())

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
            )
            self._catalog_time = loop_time
            return self._catalog

    async def _days_and_catalog(self) -> tuple[list[date], Catalog, tzinfo, datetime]:
        """Return the days that have an album, the catalog, the time zone and the time."""
        catalog = await self._get_catalog()
        tz = await self._timezone()
        now = utc().astimezone(tz)
        days = [day for day in recent_days(now.date(), self._days) if playable(day, tz, now)]
        return days, catalog, tz, now

    def _mapping(self, item_id: str) -> ProviderMapping:
        """Return where an item lives on this source."""
        return ProviderMapping(
            item_id=item_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            # what this house did is its own: another instance of the provider has other days
            is_unique=True,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=SAMPLE_RATE,
                bit_depth=AUDIO_BIT_DEPTH,
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

    def _credited_artist(self, area_id: str, name: str, *, full: bool) -> Artist | ItemMapping:
        """Return the artist as credited on an item: a mapping in a listing, the full artist for a single item."""
        artist = self._artist(area_id, name)
        return artist if full else ItemMapping.from_item(artist)

    def _nested_album(
        self, area_id: str, name: str, day: date, facts: DayFacts | None, *, full: bool
    ) -> Album | ItemMapping:
        """Return the album of a track: a mapping in a listing, the full album for a single item."""
        album = self._album(area_id, name, day, facts, full=full)
        return album if full else ItemMapping.from_item(album)

    def _artist_images(self, area_id: str) -> UniqueList[MediaItemImage]:
        """Return the drawn thumb and the wide backdrop of a room."""
        return UniqueList(
            [
                MediaItemImage(
                    type=image_type,
                    path=(
                        f"{ARTIST_PATH_PREFIX}|{area_id}|{kind}|{figure_shape(area_id)}"
                        f"|g{COVER_GENERATION}"
                    ),
                    provider=self.instance_id,
                    remotely_accessible=False,
                )
                for image_type, kind in (
                    (ImageType.THUMB, ImageType.THUMB.value),
                    (ImageType.FANART, ImageType.FANART.value),
                )
            ]
        )

    def _images(self, area_id: str, day: date, style_key: str | None) -> UniqueList[MediaItemImage]:
        """Return the cover of an area, a day and a style."""
        # a day that is still going on has its own path, so the finished cover is fetched anew once it is over
        live = f"|{COVER_LIVE_MARK}" if self._day_is_running(day) else ""
        path = (
            f"{COVER_PATH_PREFIX}|{area_id}|{day.isoformat()}|{style_key or ''}"
            f"{live}|g{COVER_GENERATION}"
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

    def _day_is_running(self, day: date) -> bool:
        """Return whether a day is not over yet in the house, or on this server while the house time zone is unknown."""
        return utc().astimezone(self._tz).date() <= day

    async def _artist_image(self, parts: list[str]) -> bytes | None:
        """
        Return the image of a room, drawn from the week that is kept.

        :param parts: `artist|<area id>|thumb or fanart|<figure>|g<cover generation>`, split at the bars.
        """
        if (
            len(parts) != ARTIST_PATH_PARTS
            or parts[ARTIST_KIND_INDEX] not in {ImageType.THUMB.value, ImageType.FANART.value}
            or parts[ARTIST_FIGURE_INDEX] not in SHAPES
            or not re.fullmatch(GENERATION_PATTERN, parts[-1])
        ):
            return None
        area_id, kind = parts[ARTIST_AREA_INDEX], parts[ARTIST_KIND_INDEX]
        catalog = await self._catalog_all()
        if area_id == WHOLE_HOUSE_ID:
            name = WHOLE_HOUSE_NAME
        elif area_id in catalog.areas:
            name = catalog.areas[area_id]
        else:
            return None
        profile = await self._room_profile(area_id, catalog)
        async with self._cover_draws:
            if kind == ImageType.FANART.value:
                return await asyncio.to_thread(artist_fanart, area_id, profile)
            return await asyncio.to_thread(artist_thumb, area_id, name, profile)

    async def _room_profile(self, area_id: str, catalog: Catalog) -> RoomProfile:
        """
        Return the typical day of a room (or of the house) from the history of the last days that is kept.

        Home Assistant is never asked: days that are not in the cache or pinned are left out, and a room
        with none gets a figure made from its id. A read that fails raises, so that no fallback is kept.
        """
        tz, now = await self._timezone(), await self._now()
        today = now.astimezone(tz).date()
        rooms = [area_id] if area_id != WHOLE_HOUSE_ID else list(self._house_rooms(catalog))
        days: list[list[Series]] = []
        for offset in range(1, ARTIST_PROFILE_DAYS + 1):
            day = today - timedelta(days=offset)
            found: list[Series] = []
            for room in rooms:
                infos = await self._infos_for(room, catalog, day)
                series = await self.history.cached_series(room, infos, day) if infos else None
                found.extend(series or [])
            days.append(found)
        profile = await asyncio.to_thread(week_profile, days)
        return profile or seeded_profile(area_id)

    async def _cover(self, area_id: str, day: date, style_key: str) -> bytes:
        """Return the cover of a day, drawn from its events."""
        style = get_style(style_key)
        area_name = WHOLE_HOUSE_NAME if area_id == WHOLE_HOUSE_ID else area_id
        catalog = await self._catalog_all()
        area_name = catalog.areas.get(area_id, area_name)
        profile = profile_of(await self._cover_series(area_id, catalog, day))
        palette = derive_palette(style, day, self._scale, DayTraits(energy=profile.intensity))
        slot = f"{area_id}|{day.isoformat()}|{style.key}"
        title = album_name(area_name, day)
        async with self._cover_draws:
            return await asyncio.to_thread(
                procedural_cover, profile, style.key, palette.root, palette.mode, slot, title
            )

    async def _cover_series(self, area_id: str, catalog: Catalog, day: date) -> list[Series]:
        """
        Return the series of a day for its cover, read once and then kept like those of a track.

        :param area_id: The room, or the whole house, which is its rooms together.
        :param catalog: The rooms and their entities.
        :param day: The day.
        """
        rooms = self._house_rooms(catalog) if area_id == WHOLE_HOUSE_ID else [area_id]
        tz, now = await self._timezone(), await self._now()
        found: list[Series] = []
        for room in rooms:
            infos = await self._infos_for(room, catalog, day)
            if infos:
                found.extend(await self._series_of(room, infos, day, tz, now))
        return found

    def _album(
        self,
        area_id: str,
        area_name: str,
        day: date,
        facts: DayFacts | None = None,
        *,
        full: bool = False,
    ) -> Album:
        """
        Return the album of one day of an area.

        :param area_id: The room, or the whole house.
        :param area_name: The name of the room.
        :param day: The day.
        :param facts: What the day was like; the review says less without it.
        :param full: Give the artist as a full object (a single item), not as a mapping (a listing).
        """
        item_id = album_id(area_id, day)
        return Album(
            item_id=item_id,
            provider=self.instance_id,
            name=album_name(area_name, day),
            artists=UniqueList([self._credited_artist(area_id, area_name, full=full)]),
            year=day.year,
            album_type=AlbumType.ALBUM,
            metadata=MediaItemMetadata(
                genres={"Generative"},
                description=album_review(area_name, day, facts, self._language),
                images=self._images(
                    area_id, day, album_style(facts, [style.key for style in self._offered_styles])
                ),
            ),
            provider_mappings={self._mapping(item_id)},
            favorite=True if item_id in self.membership.albums else None,
        )

    def _track(
        self,
        ref: TrackRef,
        area_name: str,
        name: str,
        number: int,
        duration: int,
        description: str,
        version: str | None = None,
        facts: DayFacts | None = None,
        *,
        full: bool = False,
    ) -> Track:
        """Return one track of an album, in the style of the reference; its album and artist are mappings unless full."""
        item_id = ref.item_id
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
            artists=UniqueList([self._credited_artist(ref.area_id, area_name, full=full)]),
            album=self._nested_album(ref.area_id, area_name, ref.day, facts, full=full),
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
        enough_of_day: bool,  # noqa: FBT001
        facts: DayFacts | None,
        *,
        full: bool = False,
    ) -> list[Track]:
        """
        Return the tracks of an album: the soundscape of the day in each style that is offered.

        Every track has a title of its own, made from the facts of the day and its style, and the
        description of its style.

        :param area_id: The room, or the whole house.
        :param name: The name of the room.
        :param day: The day.
        :param enough_of_day: Whether enough of the day has happened for a soundscape, see `library.playable`.
        :param facts: What the day was like, None when it could not be read.
        :param full: Give the album and artist as full objects (a single item), not as mappings (a listing).
        """
        lang = self._language
        room = (
            name
            if area_id != WHOLE_HOUSE_ID
            else HOUSE_ROOM_NAMES.get(lang, HOUSE_ROOM_NAMES[DEFAULT_LANGUAGE])
        )
        seed = f"{area_id}|{day.isoformat()}"
        return [
            self._track(
                TrackRef(area_id, day, SCAPE_ID, style.key),
                name,
                make_title(facts, style.key, seed, room, lang),
                number,
                condensed_seconds(style.key),
                track_description(style.key, facts, lang),
                facts=facts,
                full=full,
            )
            for number, style in enumerate(self._offered_styles, start=1)
            if enough_of_day
        ]

    def _tracks_of(
        self,
        catalog: Catalog,
        area_id: str,
        day: date,
        enough_of_day: bool,  # noqa: FBT001
        facts: DayFacts | None = None,
        *,
        full: bool = False,
    ) -> list[Track]:
        """
        Return the tracks of any album: its soundscape in each style.

        :param catalog: The registries.
        :param area_id: The room, or the whole house.
        :param day: The day.
        :param enough_of_day: Whether enough of the day has happened for a soundscape, see `library.playable`.
        :param facts: What the day was like, None when it could not be read.
        :param full: Give the album and artist as full objects (a single item), not as mappings (a listing).
        """
        name = WHOLE_HOUSE_NAME if area_id == WHOLE_HOUSE_ID else catalog.areas[area_id]
        return self._day_tracks(area_id, name, day, enough_of_day, facts, full=full)

    async def _facts(
        self,
        area_id: str,
        day: date,
        *,
        cached_only: bool = False,
        kept: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> DayFacts | None:
        """
        Return what a day of a room (or of the house) was like, for its titles; None when it cannot be read.

        A reading is kept for the hour, so that the title of today stays the same within an hour; a past day
        does not change. Home Assistant being away, or a day without history, never fails a listing.

        :param area_id: The room, or the whole house.
        :param day: The day.
        :param cached_only: Do not read the history of the day, only use a reading that is kept.
        :param kept: The readings of a whole listing, fetched at once; None to ask the cache for this one.
        :param strict: Raise instead of returning None when the day cannot be read, for what the server stores.
        :raises ProviderUnavailableError: When strict and Home Assistant does not answer.
        """
        try:
            tz, now = await self._timezone(), await self._now()
            covered = covered_seconds(day, tz, now)
            catalog = await self._catalog_all()
            key = self._facts_key(area_id, day, now, covered, catalog)
            seen = (
                kept.get(key)
                if kept is not None
                else await self.mass.cache.get(
                    key, provider=self.instance_id, category=CACHE_CATEGORY_FACTS
                )
            )
            if seen is not None:
                return DayFacts(**seen)
            if cached_only or covered < MIN_SCAPE_SECONDS_OF_DAY:
                return None
            series: list[Series] = []
            for room in self._facts_rooms(area_id, catalog):
                infos = await self._infos_for(room, catalog, day)
                if infos:
                    series.extend(await self._series_of(room, infos, day, tz, now))
            found = await asyncio.to_thread(facts_of, series, day, covered)
        except HASS_ERRORS as err:
            if strict:
                raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err
            self.logger.debug("Could not read %s on %s for its titles", area_id, day, exc_info=True)
            return None
        await self.mass.cache.set(
            key,
            asdict(found),
            expiration=FACTS_TODAY_EXPIRATION if covered < DAY_SECONDS else PAST_DAY_EXPIRATION,
            provider=self.instance_id,
            category=CACHE_CATEGORY_FACTS,
        )
        return found

    async def _prefetch(self) -> dict[str, Any]:
        """Return every reading of a day that is kept, in one read of the cache, for a listing."""
        return await self.mass.cache.get_all(
            provider=self.instance_id, category=CACHE_CATEGORY_FACTS
        )

    async def _prime_history(
        self, days: Mapping[str, Sequence[date]], catalog: Catalog, kept: dict[str, Any]
    ) -> None:
        """
        Read the history of the rooms and days that a listing has no kept reading for, once per day.

        Without it each room and day is its own request to Home Assistant.

        :param days: The days of each room, or of the whole house.
        :param catalog: The registries.
        :param kept: The kept readings of the listing.
        :raises ProviderUnavailableError: When Home Assistant does not answer.
        """
        tz, now = await self._timezone(), await self._now()
        rooms_by_day: dict[date, set[str]] = {}
        for area_id, area_days in days.items():
            for day in area_days:
                covered = covered_seconds(day, tz, now)
                if (
                    covered >= MIN_SCAPE_SECONDS_OF_DAY
                    and self._facts_key(area_id, day, now, covered, catalog) not in kept
                ):
                    rooms_by_day.setdefault(day, set()).update(self._facts_rooms(area_id, catalog))
        try:
            for day, rooms in sorted(rooms_by_day.items(), reverse=True):
                groups = {
                    room: infos
                    for room in sorted(rooms)
                    if (infos := await self._infos_for(room, catalog, day))
                }
                await self.history.prefetch(groups, day, tz, now)
        except HASS_ERRORS as err:
            raise ProviderUnavailableError(f"Home Assistant did not answer: {err}") from err

    def _facts_rooms(self, area_id: str, catalog: Catalog) -> tuple[str, ...]:
        """Return the rooms whose history a reading of an area is made from: the room, or every room of the house."""
        return tuple(self._house_rooms(catalog)) if area_id == WHOLE_HOUSE_ID else (area_id,)

    def _facts_key(
        self, area_id: str, day: date, now: datetime, covered: float, catalog: Catalog
    ) -> str:
        """Return the cache key of the facts of a day; today's are told apart by the hour."""
        return self._reading_key(
            (FACTS_SCHEMA, *(field.name for field in fields(DayFacts))),
            area_id,
            day,
            str(now.hour) if covered < DAY_SECONDS else "",
            self._facts_rooms(area_id, catalog),
        )

    def _reading_key(
        self,
        made_with: tuple[object, ...],
        area_id: str,
        day: date,
        bucket: str,
        rooms: tuple[str, ...],
    ) -> str:
        """
        Return the cache key of a reading of a day.

        The key holds what the reading was made with (the engine, what picks the entities, and the
        schema in `made_with`), so that a reading made with other settings is never found, and the
        bucket of the day, which tells today's readings apart.
        """
        parts = (
            *made_with,
            ENGINE_VERSION,
            self._include_presence,
            self.get_config_value(CONF_ROLES, ""),
            self.get_config_value(CONF_EXCLUDE, ""),
        )
        made = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:CHECKSUM_LENGTH]
        return f"{area_id}|{day.isoformat()}|{bucket}|{','.join(rooms)}|{made}"

    @staticmethod
    def _artist_names(catalog: Catalog) -> dict[str, str]:
        """Return the artists: the whole house first, then each area."""
        if not catalog.areas:
            return {}
        return {WHOLE_HOUSE_ID: WHOLE_HOUSE_NAME, **catalog.areas}

    async def _pinned_scope(self, scope: str, day: date) -> dict[str, Any] | None:
        """Return the history that was pinned for an area (or the sun) on a day, if there is one."""
        if scope == SUN_SCOPE:
            for area_id in self.membership.snapshot:
                pin = await asyncio.to_thread(self.pins.read, area_id, day)
                if pin is not None and pin.complete and pin.sun is not None:
                    return dict(pin.sun["data"])
            return None
        pin = await asyncio.to_thread(self.pins.read, scope, day)
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
        return Catalog(areas, entities, catalog.sun, catalog.changed)

    async def _pinned_house_days(self) -> set[date]:
        """Return the days that every room of the whole house has pinned."""
        rooms = set(self._house_rooms(await self._catalog_all()))
        pinned = [
            set(await asyncio.to_thread(self.pins.days, area))
            for area in self.membership.snapshot
            if area in rooms
        ]
        return set.intersection(*pinned) if pinned else set()

    async def _browse_days(self, area_id: str) -> list[date]:
        """
        Return every day that an area can be played on, newest first.

        That is every day that Home Assistant still has (up to the setting) and every pinned day,
        however old.
        """
        _, _, tz, now = await self._days_and_catalog()
        found = {day for day in recent_days(now.date(), self._days) if playable(day, tz, now)}
        if area_id == WHOLE_HOUSE_ID:
            found.update(await self._pinned_house_days())
        else:
            found.update(await asyncio.to_thread(self.pins.days, area_id))
        return sorted(found, reverse=True)

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

    @staticmethod
    def _parse_track(item_id: str) -> TrackRef:
        """Return the parts of a track id; it must be a soundscape in a style that exists."""
        try:
            ref = parse_track_id(item_id)
        except ValueError as err:
            raise MediaNotFoundError(f"Track {item_id} not found") from err
        # a track is the soundscape of a room in a style; the voice tracks of earlier versions are gone
        if ref.style not in STYLES or ref.target not in TRACK_TARGETS:
            raise MediaNotFoundError(f"Track {item_id} not found")
        return ref

    async def _check_track(self, item_id: str) -> TrackRef:
        """Return the parts of a track id after checking that its album exists."""
        ref = self._parse_track(item_id)
        await self._check_album(album_id(ref.area_id, ref.day))
        return ref

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
            existing = await asyncio.to_thread(self.pins.read, area_id, day)
            if existing is not None and existing.complete:
                return True
            # a file that cannot be read (a newer format) is the only copy of the day: never write over it
            if existing is None and await asyncio.to_thread(self.pins.has, area_id, day):
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
                        "data": await self.history.day_data(SUN_SCOPE, [catalog.sun], day, tz, now),
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

    async def _favorite_room(self, area_id: str, catalog: Catalog, *, keep: bool) -> None:
        """Keep every day of a room that was hearted, or let go of it."""
        if keep:
            if area_id not in self._artist_names(catalog):
                raise MediaNotFoundError(f"Artist {area_id} not found")
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

    async def _maintain(self) -> None:
        """Pin the days of the followed areas that are over, and the days that were pinned early; then again later."""
        try:
            await self._pin_followed()
        except BACKGROUND_ERRORS as err:
            self.logger.warning("Pinning the days of the followed areas failed: %s", err)
        finally:
            # an error that is not expected costs this run, not every later one
            if not self.unloading:
                self.mass.call_later(MAINTENANCE_SECONDS, self._maintain, task_id=self._maintain_id)

    async def _any_pin_missing(self, areas: list[str], day: date) -> bool:
        """
        Return whether any of the areas has no pin on a day.

        :param areas: The areas.
        :param day: The day.
        """
        for area in areas:
            if not await asyncio.to_thread(self.pins.has, area, day):
                return True
        return False

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
                stale = False
                for area in areas:
                    pin = await asyncio.to_thread(self.pins.read, area, day)
                    if pin is not None and not pin.complete and day < today:
                        stale = True
                        break
                if day >= today and not stale:
                    continue
                if stale:
                    for area in areas:
                        await asyncio.to_thread(self.pins.delete, area, day)
                missing = await self._any_pin_missing(areas, day)
                if (
                    missing
                    or stale
                    or any((area, day.isoformat()) not in known for area, day in pairs)
                ):
                    await self._remember(f"{ARTIST_KEY}{area_id}", pairs)

    def _audio_key(self, ref: TrackRef, today: date, catalog: Catalog | None = None) -> str | None:
        """
        Return the key of the cached audio of a track, or None for today and when it is off.

        It holds everything that changes the sound and, for the whole house, the rooms it is made of.

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
            ref.item_id,
            rooms,
            self._scale,
            self._day_character,
            self._include_presence,
            self.get_config_value(CONF_ROLES, ""),
            self.get_config_value(CONF_EXCLUDE, ""),
        )

    @staticmethod
    def _render_seed(ref: TrackRef) -> str:
        """Return the seed of the textures of a track, which is the same for a live stream and a pre-render."""
        return f"{ref.area_id}|{ref.day.isoformat()}|{ref.style}"

    async def _prerender(self) -> None:
        """Pre-render the last day once the benchmark of Auto is done; then again shortly after the next midnight."""
        if self._benchmark_task is not None:
            await asyncio.wait({self._benchmark_task})
        wait = prerender.seconds_until_run(
            await self._now(),
            prerender.MIDNIGHT_DELAY_SECONDS
            + random.uniform(0.0, prerender.MIDNIGHT_JITTER_SECONDS),  # noqa: S311
        )
        try:
            await self._prerender_yesterday()
        except BACKGROUND_ERRORS as err:
            self.logger.warning("Pre-rendering the last day failed: %s", err)
        finally:
            # an error that is not expected costs this run, not every later one
            if not self.unloading:
                self.mass.call_later(wait, self._prerender, task_id=self._prerender_id)

    async def _prerender_yesterday(self) -> int:
        """
        Render the tracks of yesterday that the audio cache does not have, one at a time.

        :return: How many were rendered.
        """
        if not self.audio_cache.enabled:
            return 0
        catalog = await self._catalog_all()
        day = (await self._now()).date() - timedelta(days=1)
        added = self.membership
        refs = prerender.choose_tracks(
            day,
            rooms=await self._offered(catalog),
            pinned_rooms=self._keep_rooms,
            offered=[style.key for style in self._offered_styles],
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
        try:
            await asyncio.wait_for(self._no_streams.wait(), prerender.GIVE_UP_SECONDS)
        except TimeoutError:
            return False
        return True

    def _stream_started(self) -> None:
        """Count a live stream, which holds the pre-render back."""
        self._streams += 1
        self._no_streams.clear()

    def _stream_ended(self) -> None:
        """Count a live stream out, and let the pre-render go on when it was the last."""
        self._streams -= 1
        if self._streams == 0:
            self._no_streams.set()

    async def _prerender_room_for(self, ref: TrackRef) -> tuple[str, float] | None:
        """Return the key and the length of a track that is worth rendering: not cached, not today, still readable."""
        key = self._audio_key(ref, (await self._now()).date(), await self._catalog_all())
        if key is None or await asyncio.to_thread(self.audio_cache.path(key).exists):
            return None
        try:
            await self._check_track(ref.item_id)
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
        recorder = await self.audio_cache.recorder(key) if await self._prerender_wait() else None
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
                if recorder.failed:
                    return None
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
        self.logger.debug("Pre-rendered %s", ref.item_id)
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
        groups: list[tuple[str, list[EntityInfo]]] = []
        area_ids = self._house_rooms(catalog) if ref.area_id == WHOLE_HOUSE_ID else [ref.area_id]
        for area_id in area_ids:
            infos = await self._infos_for(area_id, catalog, ref.day)
            if infos is None:
                raise MediaNotFoundError(f"Area {area_id} not found")
            # the whole area is read at once so that its tracks share one cached request
            fetched = await self._series_of(area_id, infos, ref.day, tz, now)
            wanted = {info.entity_id for info in infos}
            series.extend(item for item in fetched if item.entity_id in wanted)
            whole_day.extend(fetched)
            if self._day_character:
                baselines.update(await self._baselines(area_id, infos, ref, tz, now))
                groups.append((area_id, infos))
        if catalog.sun is not None:
            series.extend(await self._series_of(SUN_SCOPE, [catalog.sun], ref.day, tz, now))
        # every sensor is measured against its own previous days, and the dead ones play nothing
        series, dropped = await asyncio.to_thread(rescale, series, baselines)
        whole_day, _ = await asyncio.to_thread(rescale, whole_day, baselines)
        if dropped:
            self.logger.debug("Left out %d sensors that say nothing: %s", len(dropped), dropped)
        week = await self._week(groups, baselines, ref, tz, now) if self._day_character else []
        return _Gathered(series, whole_day, baselines, covered_seconds(ref.day, tz, now), week)

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
        key = (
            ref.area_id,
            ref.day,
            ref.style,
            int(covered // SECONDS_BUCKET) if covered < DAY_SECONDS else -1,
            rooms,
        )
        found = self._scapes.get(key)
        if found is not None:
            self._scapes.move_to_end(key)
            return found
        gathered = await self._gather(ref)
        style = get_style(ref.style)
        seed = f"{ref.area_id}|{ref.day.isoformat()}|{ref.target}|{ref.style}"

        def make() -> _Scape:
            palette = None
            traits: DayTraits | None = None
            anomalies: list[tuple[float, float, str]] = []
            if self._day_character:
                traits = measure_traits(gathered.whole_day, gathered.baselines, gathered.covered)
                palette = derive_palette(style, ref.day, self._scale, traits)
                featured = Surprise(gathered.series, gathered.baselines).featured(
                    0.0, 1.0, min(gathered.covered, DAY_SECONDS)
                )
                anomalies = [
                    (t, score, entity)
                    for t, score, entity in featured
                    if score >= ANOMALY_MIN_SCORE
                ]
            analysis = analysis_of(gathered.series, gathered.covered, anomalies)
            character = None
            if traits is not None:
                character = measure_character(
                    analysis, traits, ref.day, gathered.series, week=gathered.week
                )
                self.logger.debug("Character of %s: %s", seed, character.label())
            return _Scape(
                gathered,
                analysis,
                palette,
                seed,
                condensed_seconds(ref.style),
                character,
            )

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
            quality=self.quality,
            options=PieceOptions(
                day=ref.day,
                palette=found.palette,
                area_key=ref.area_id,
                character=found.character,
            ),
        )
        return score

    async def _infos_for(
        self, area_id: str, catalog: Catalog, day: date
    ) -> list[EntityInfo] | None:
        """Return the entities of an area on a day: those of its pin when it has one, else today's."""
        pin = await asyncio.to_thread(self.pins.read, area_id, day)
        if pin is not None and pin.complete:
            return [entity_from_dict(item) for item in pin.entities]
        return catalog.entities.get(area_id)

    async def _week(
        self,
        groups: list[tuple[str, list[EntityInfo]]],
        baselines: dict[str, EntityBaseline],
        ref: TrackRef,
        tz: tzinfo,
        now: datetime,
    ) -> list[float]:
        """Return how busy the earlier days of some rooms were; without them the day is measured on its own."""
        return await self.history.week_energies(groups, ref.day, tz, now, baselines)

    async def _baselines(
        self, area_id: str, infos: list[EntityInfo], ref: TrackRef, tz: tzinfo, now: datetime
    ) -> dict[str, EntityBaseline]:
        """Return what the entities of an area usually do; without a baseline there is no surprise."""
        return await self.history.baselines(area_id, infos, ref.day, tz, now)


def _lower_priority() -> None:
    """Make the thread that does the pre-render the politest one of the process (where the system allows it)."""
    with contextlib.suppress(AttributeError, OSError):
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), prerender.NICE_LEVEL)
