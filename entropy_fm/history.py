"""
Fetch the history of a day from Home Assistant, once, and keep it in the cache of the server.

The library never calls this. Only playing a track does, and then once per area and day: a past
day stays in the cache for a week, today for five minutes.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

from .baseline import DAY_SECONDS, EntityBaseline, build_baselines, has_data
from .character import raw_energy
from .const import (
    BASELINE_DAYS,
    BASELINE_EXPIRATION,
    BASELINE_SCHEMA,
    CACHE_CATEGORY_BASELINE,
    CACHE_CATEGORY_HISTORY,
    CACHE_SCHEMA,
    PAST_DAY_EXPIRATION,
    TODAY_EXPIRATION,
    WEEK_ENERGY_SCHEMA,
)
from .library import day_bounds
from .references import rescale
from .timeline import Series, build_series, raw_to_json, series_from_json

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import date, datetime, tzinfo
    from logging import Logger

    from .hass_access import HassAccess
    from .timeline import EntityInfo


class Cache(Protocol):
    """The part of the cache controller of the server that is used."""

    async def get(
        self,
        key: str,
        provider: str = ...,
        category: int = ...,
        checksum: str | int | None = ...,
        default: Any = ...,
    ) -> Any:
        """Return the cached data or the default."""

    async def set(  # noqa: PLR0913, PLR0917
        self,
        key: str,
        data: Any,
        expiration: int = ...,
        provider: str = ...,
        category: int = ...,
        checksum: str | None = ...,
    ) -> None:
        """Store data."""


def series_from_data(infos: Sequence[EntityInfo], data: dict[str, Any]) -> list[Series]:
    """
    Make the series of some entities from raw history.

    :param infos: The entities.
    :param data: Entity id to the raw history, as `HistoryStore.day_data` returns.
    """
    return [
        series_from_json(info, data[info.entity_id])
        if info.entity_id in data
        else build_series(info, [], [])
        for info in infos
    ]


class HistoryStore:
    """Gives the series of the entities of an area on a day, from the cache when it can."""

    def __init__(
        self,
        hass: HassAccess,
        cache: Cache,
        provider_id: str,
        logger: Logger,
        pin_source: Callable[[str, date], dict[str, Any] | None] | None = None,
    ) -> None:
        """
        Prepare the store.

        :param hass: Access to Home Assistant.
        :param cache: The cache of the server.
        :param provider_id: The instance id of the provider, which namespaces the cache.
        :param logger: Where to log.
        :param pin_source: Gives the history that was pinned for a group and a day, or None; a
            pinned day is read from there before Home Assistant is asked.
        """
        self.hass = hass
        self.cache = cache
        self.provider_id = provider_id
        self.logger = logger
        self.pin_source = pin_source
        self._locks: dict[str, asyncio.Lock] = {}

    async def day_data(
        self,
        scope: str,
        infos: Sequence[EntityInfo],
        day: date,
        tz: tzinfo,
        now: datetime,
    ) -> dict[str, Any]:
        """
        Return the raw history of some entities for one day, as the cache stores it.

        :param scope: Names the group in the cache, such as the area id, or "_sun".
        :param infos: The entities of the group.
        :param day: The day.
        :param tz: The time zone of the house.
        :param now: The current time; today is only read up to it.
        """
        if not infos:
            return {}
        start, end = day_bounds(day, tz)
        is_today = now.astimezone(tz) < end
        key, checksum = self._day_key(scope, infos, day)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            data = await self.cache.get(
                key, provider=self.provider_id, category=CACHE_CATEGORY_HISTORY, checksum=checksum
            )
            if data is None and self.pin_source is not None:
                data = self.pin_source(scope, day)
            if data is None:
                data = await self._fetch(infos, tz, now, start, end)
                await self.cache.set(
                    key,
                    data,
                    expiration=TODAY_EXPIRATION if is_today else PAST_DAY_EXPIRATION,
                    provider=self.provider_id,
                    category=CACHE_CATEGORY_HISTORY,
                    checksum=checksum,
                )
        return dict(data)

    def _day_key(self, scope: str, infos: Sequence[EntityInfo], day: date) -> tuple[str, str]:
        """Return the key and the checksum under which the history of a group and a day is cached."""
        key = f"{scope}|{day.isoformat()}"
        checksum = hashlib.sha256(
            f"{CACHE_SCHEMA}|{','.join(sorted(info.entity_id for info in infos))}".encode()
        ).hexdigest()[:16]
        return key, checksum

    async def cached_series(
        self, scope: str, infos: Sequence[EntityInfo], day: date
    ) -> list[Series] | None:
        """
        Return the series of a group on a day only when they are already kept: never asks Home Assistant.

        :param scope: Names the group in the cache, such as the area id.
        :param infos: The entities of the group.
        :param day: The day.
        :returns: The series, or None when the day is neither in the cache nor pinned.
        """
        if not infos:
            return None
        key, checksum = self._day_key(scope, infos, day)
        data = await self.cache.get(
            key, provider=self.provider_id, category=CACHE_CATEGORY_HISTORY, checksum=checksum
        )
        if data is None and self.pin_source is not None:
            data = self.pin_source(scope, day)
        return None if data is None else series_from_data(infos, dict(data))

    async def day_series(
        self,
        scope: str,
        infos: Sequence[EntityInfo],
        day: date,
        tz: tzinfo,
        now: datetime,
    ) -> list[Series]:
        """
        Return the series of some entities for one day.

        :param scope: Names the group in the cache, such as the area id, or "_sun".
        :param infos: The entities of the group.
        :param day: The day.
        :param tz: The time zone of the house.
        :param now: The current time; today is only read up to it.
        """
        data = await self.day_data(scope, infos, day, tz, now)
        return series_from_data(infos, data)

    async def baselines(  # noqa: PLR0913, PLR0917
        self,
        scope: str,
        infos: Sequence[EntityInfo],
        day: date,
        tz: tzinfo,
        now: datetime,
        days: int = BASELINE_DAYS,
    ) -> dict[str, EntityBaseline]:
        """
        Return what the entities of a group usually do, from the days before a day.

        The baseline is built from the cached history of those days (a missing day is read once)
        and cached itself, because the days before a day do not change.

        :param scope: Names the group in the cache, such as the area id.
        :param infos: The entities of the group.
        :param day: The day the baseline is for; only days before it are used.
        :param tz: The time zone of the house.
        :param now: The current time.
        :param days: How many days back the baseline reaches.
        """
        if not infos:
            return {}
        key = f"baseline|{scope}|{day.isoformat()}"
        checksum = hashlib.sha256(
            f"{BASELINE_SCHEMA}|{days}|{','.join(sorted(info.entity_id for info in infos))}".encode()
        ).hexdigest()[:16]
        cached = await self.cache.get(
            key, provider=self.provider_id, category=CACHE_CATEGORY_BASELINE, checksum=checksum
        )
        if cached is not None:
            return {item["id"]: EntityBaseline.from_json(item) for item in cached}
        before = [
            await self.day_series(scope, infos, day - timedelta(days=offset), tz, now)
            for offset in range(1, days + 1)
        ]
        built = build_baselines(before)
        await self.cache.set(
            key,
            [item.to_json() for item in built.values()],
            expiration=BASELINE_EXPIRATION,
            provider=self.provider_id,
            category=CACHE_CATEGORY_BASELINE,
            checksum=checksum,
        )
        return built

    async def week_energies(  # noqa: PLR0913, PLR0917
        self,
        groups: Sequence[tuple[str, Sequence[EntityInfo]]],
        day: date,
        tz: tzinfo,
        now: datetime,
        baselines: Mapping[str, EntityBaseline],
        days: int = BASELINE_DAYS,
    ) -> list[float]:
        """
        Return how busy the days before a day were, on the scale the day itself is measured on.

        Each earlier day is read from the cached history, rescaled with the baselines of the day and measured with
        `character.raw_energy`; a day without any history is left out. The result is cached, because the days
        before a day do not change.

        :param groups: The rooms as (scope, entities); a house is all its rooms, measured together.
        :param day: The day the energies are for; only days before it are used.
        :param tz: The time zone of the house.
        :param now: The current time.
        :param baselines: What the entities usually do on the day, by entity id.
        :param days: How many days back to look.
        """
        infos = [info for _, group in groups for info in group]
        if not infos:
            return []
        key = f"week|{','.join(scope for scope, _ in groups)}|{day.isoformat()}"
        checksum = hashlib.sha256(
            f"{WEEK_ENERGY_SCHEMA}|{days}|{','.join(sorted(info.entity_id for info in infos))}".encode()
        ).hexdigest()[:16]
        cached = await self.cache.get(
            key, provider=self.provider_id, category=CACHE_CATEGORY_BASELINE, checksum=checksum
        )
        if cached is not None:
            return [float(item) for item in cached]
        before: list[list[Series]] = []
        for offset in range(1, days + 1):
            earlier = day - timedelta(days=offset)
            found: list[Series] = []
            for scope, group in groups:
                found.extend(await self.day_series(scope, group, earlier, tz, now))
            if has_data(found):
                before.append(found)

        def measure() -> list[float]:
            return [raw_energy(rescale(found, baselines)[0], DAY_SECONDS) for found in before]

        energies = await asyncio.to_thread(measure)
        await self.cache.set(
            key,
            energies,
            expiration=BASELINE_EXPIRATION,
            provider=self.provider_id,
            category=CACHE_CATEGORY_BASELINE,
            checksum=checksum,
        )
        return energies

    async def _fetch(
        self,
        infos: Sequence[EntityInfo],
        tz: tzinfo,
        now: datetime,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Ask Home Assistant for the history and return what the cache stores."""
        until = min(end, now.astimezone(tz))
        self.logger.debug("Reading history of %d entities from %s", len(infos), start.date())
        history = await self.hass.history(
            [info.entity_id for info in infos], start, until, start.timestamp()
        )
        return {
            info.entity_id: raw_to_json(info, *history.get(info.entity_id, ([], [])))
            for info in infos
        }
