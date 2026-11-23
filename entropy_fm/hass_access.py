"""
One view of Home Assistant for the two ways Entropy FM can reach it.

Either through the core Home Assistant plugin of Music Assistant, whose connection it shares, or
through a connection of its own made with a URL and a long-lived token.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hass_client.exceptions import BaseHassClientError
from music_assistant_models.errors import ProviderUnavailableError

from .constants import HISTORY_CHUNK
from .library import SUN_ENTITIES, candidate_ids
from .roles import EntityMeta
from .timeline import parse_history

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from hass_client import HomeAssistantClient

    from music_assistant.providers.hass import HomeAssistantProvider

LOGGER = logging.getLogger(__name__)
# what a read of Home Assistant fails with when it is away or does not answer
HASS_ERRORS = (BaseHassClientError, ProviderUnavailableError)
# what the answer for the time zone fails with: an error, no `time_zone` in it, or a zone that is not known
TIME_ZONE_ERRORS = (*HASS_ERRORS, KeyError, ValueError, ZoneInfoNotFoundError)
# what the registries fail with when they do not have the shape that is read
REGISTRY_ERRORS = (*HASS_ERRORS, KeyError)
# how many entities are asked of the registry of the plugin in one call
REGISTRY_BATCH = 200


class Snapshot(NamedTuple):
    """What one read of Home Assistant gives for building the library."""

    areas: dict[str, str]
    entity_areas: dict[str, str | None]
    states: list[dict[str, Any]]
    meta: dict[str, EntityMeta]


class HassAccess:
    """Reads the registries, the states and the history of Home Assistant."""

    def __init__(
        self,
        client: HomeAssistantClient | None = None,
        plugin_getter: Callable[[], HomeAssistantProvider | None] | None = None,
    ) -> None:
        """
        Wrap a connection.

        :param client: A connection of our own; used when there is no plugin.
        :param plugin_getter: Returns the core Home Assistant provider, whose connection and
            cached registries are used. It is asked every time, because the plugin can reload.
        """
        self._client = client
        self._plugin_getter = plugin_getter
        # one history request at a time: the server of the user is polite company
        self._lock = asyncio.Lock()

    def set_client(self, client: HomeAssistantClient) -> None:
        """
        Use a new connection of our own, after the old one was lost.

        :param client: The connection that replaces it.
        """
        self._client = client

    @property
    def plugin(self) -> HomeAssistantProvider | None:
        """Return the core plugin when this access goes through it."""
        if self._plugin_getter is None:
            return None
        plugin = self._plugin_getter()
        if plugin is None:
            msg = "The Home Assistant plugin of Music Assistant is not available"
            raise ProviderUnavailableError(msg)
        return plugin

    @property
    def client(self) -> HomeAssistantClient:
        """Return the connection to Home Assistant."""
        plugin = self.plugin
        client = plugin.hass if plugin is not None else self._client
        if client is None:
            msg = "There is no connection to Home Assistant"
            raise ProviderUnavailableError(msg)
        return client

    async def areas(self) -> dict[str, str]:
        """Return the area registry as area id to name."""
        if self.plugin is not None:
            registry = await self.plugin.get_area_registry()
            return {area_id: str(area["name"]) for area_id, area in registry.items()}
        return {
            str(area["area_id"]): str(area["name"])
            for area in await self.client.get_area_registry()
        }

    async def entity_areas(self) -> dict[str, str | None]:
        """Return each entity id with its area: its own, else the one of its device."""
        result: dict[str, str | None] = {}
        if self.plugin is not None:
            entities = await self.plugin.get_entity_registry()
            devices = await self.plugin.get_device_registry()
            for entity_id, entry in entities.items():
                device = devices.get(entry.device_id) if entry.device_id else None
                result[entity_id] = entry.area_id or (device or {}).get("area_id")
            return result
        devices_by_id = {device["id"]: device for device in await self.client.get_device_registry()}
        for entry in await self.client.get_entity_registry():
            if entry.get("disabled_by"):
                continue
            device = devices_by_id.get(entry.get("device_id") or "")
            result[entry["entity_id"]] = entry.get("area_id") or (device or {}).get("area_id")
        return result

    async def states(self, entity_ids: Sequence[str]) -> list[dict[str, Any]]:
        """
        Return the current states of the given entities and of the sun.

        :param entity_ids: The entities to read; entities without a state are absent.
        """
        wanted = {*entity_ids, *SUN_ENTITIES}
        if self.plugin is not None:
            return [
                dict(state) for state in await self.plugin.get_states(entity_ids=sorted(wanted))
            ]
        return [
            dict(state) for state in await self.client.get_states() if state["entity_id"] in wanted
        ]

    async def entity_meta(self, entity_ids: Sequence[str]) -> dict[str, EntityMeta]:
        """
        Return the platform, translation key, device model and labels of entities.

        This is best effort: when Home Assistant does not answer, the entities simply have no
        hints and are classified by their domain and device class.

        :param entity_ids: The entities to look up.
        """
        try:
            return await self._entity_meta(entity_ids)
        except REGISTRY_ERRORS:
            LOGGER.warning(
                "Could not read the hints of the entities from Home Assistant", exc_info=True
            )
            return {}

    async def candidate_states(
        self, exclude_globs: Iterable[str], *, include_presence: bool
    ) -> Snapshot:
        """
        Return the areas, the area of each entity, the states and the hints of those that may play.

        :param exclude_globs: The user's patterns of entities to leave out.
        :param include_presence: Whether persons and device trackers may play.
        """
        areas = await self.areas()
        entity_areas = await self.entity_areas()
        ids = candidate_ids(
            {entity_id: area for entity_id, area in entity_areas.items() if area in areas},
            exclude_globs,
            include_presence=include_presence,
        )
        return Snapshot(
            areas,
            entity_areas,
            await self.states(ids),
            await self.entity_meta(ids),
        )

    async def time_zone(self) -> tzinfo | None:
        """Return the time zone of the house, or None when it cannot be told."""
        try:
            config = cast("dict[str, Any]", await self.client.get_config())
            return ZoneInfo(str(config["time_zone"]))
        except TIME_ZONE_ERRORS:
            LOGGER.warning(
                "Could not read the time zone of Home Assistant, the one of this server is used",
                exc_info=True,
            )
            return None

    async def history(
        self, entity_ids: Sequence[str], start: datetime, end: datetime, day_start_ts: float
    ) -> dict[str, tuple[list[float], list[float]]]:
        """
        Return the history of entities as seconds since the start of the day and values.

        :param entity_ids: The entities to read.
        :param start: Start of the period.
        :param end: End of the period.
        :param day_start_ts: Epoch seconds of the start of the day, the zero of the times.
        """
        result: dict[str, tuple[list[float], list[float]]] = {}
        length = (end - start).total_seconds()
        for first in range(0, len(entity_ids), HISTORY_CHUNK):
            chunk = list(entity_ids[first : first + HISTORY_CHUNK])
            async with self._lock:
                raw = await self.client.send_command(
                    "history/history_during_period",
                    start_time=start.astimezone(UTC).isoformat(),
                    end_time=end.astimezone(UTC).isoformat(),
                    entity_ids=chunk,
                    minimal_response=True,
                    no_attributes=True,
                    significant_changes_only=False,
                )
            result.update(parse_history(cast("dict[str, Any]", raw), day_start_ts, length))
        return result

    async def _entity_meta(self, entity_ids: Sequence[str]) -> dict[str, EntityMeta]:
        """Read the full registry entries, the devices and the labels."""
        client = self.client
        plugin = self.plugin
        wanted = set(entity_ids)
        if plugin is not None:
            entries: dict[str, Any] = {}
            for first in range(0, len(entity_ids), REGISTRY_BATCH):
                entries.update(
                    await plugin.get_entity_registry_entries(
                        entity_ids[first : first + REGISTRY_BATCH]
                    )
                )
            devices = {str(d["id"]): d for d in (await plugin.get_device_registry()).values()}
        else:
            entries = {
                str(e["entity_id"]): e
                for e in await client.get_entity_registry()
                if e["entity_id"] in wanted
            }
            devices = {str(d["id"]): d for d in await client.get_device_registry()}
        label_names = {
            str(label["label_id"]): str(label["name"])
            for label in cast(
                "list[dict[str, Any]]", await client.send_command("config/label_registry/list")
            )
        }
        result: dict[str, EntityMeta] = {}
        for entity_id, entry in entries.items():
            device = devices.get(str(entry.get("device_id") or ""), {})
            label_ids = [*(entry.get("labels") or []), *(device.get("labels") or [])]
            labels = {
                text.lower()
                for label_id in label_ids
                for text in (str(label_id), label_names.get(str(label_id), ""))
                if text
            }
            result[entity_id] = EntityMeta(
                platform=str(entry.get("platform") or ""),
                translation_key=str(entry.get("translation_key") or ""),
                manufacturer=str(device.get("manufacturer") or ""),
                model=str(device.get("model") or ""),
                labels=frozenset(labels),
                entity_category=str(entry.get("entity_category") or ""),
                device_id=str(entry.get("device_id") or ""),
            )
        return result
