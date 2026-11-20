"""Tests of the history store: how busy the days before a day were, read once and kept."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from entropy_fm.history import HistoryStore
from entropy_fm.timeline import EntityInfo, VoiceKind

VIENNA = ZoneInfo("Europe/Vienna")
HOUR = 3600.0
TODAY = date(2026, 11, 17)
NOW = datetime(2026, 11, 18, 6, 0, tzinfo=UTC)
LAMP = EntityInfo("light.lamp", "Lamp", "room", VoiceKind.CHORD)
# hours the lamp was on, by days before today; the third day before has no history at all
HOURS_ON = {1: 10.0, 2: 3.0, 4: 6.0}


class FakeCache:
    """A cache that keeps everything in a dict."""

    def __init__(self) -> None:
        """Start empty."""
        self.data: dict[tuple[str, int], Any] = {}

    async def get(self, key: str, provider: str = "", category: int = 0, **_: Any) -> Any:
        """Return what was stored under a key, or None."""
        return self.data.get((key, category))

    async def set(
        self, key: str, data: Any, provider: str = "", category: int = 0, **_: Any
    ) -> None:
        """Store data."""
        self.data[key, category] = data


class FakeHass:
    """Home Assistant, whose lamp was on for `HOURS_ON` hours from 18:00 a few days ago."""

    def __init__(self) -> None:
        """Count the requests."""
        self.calls = 0

    async def history(
        self, entity_ids: list[str], start: datetime, end: datetime, day_start_ts: float
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Return the lamp of the day that starts at `start`."""
        del end, day_start_ts
        self.calls += 1
        ago = (TODAY - start.astimezone(VIENNA).date()).days
        hours = HOURS_ON.get(ago)
        if hours is None:
            return {}
        return {entity_ids[0]: ([0.0, 6 * HOUR, (6 + hours) * HOUR], [0.0, 1.0, 0.0])}


async def test_the_energies_of_the_days_before_are_read_once_and_kept() -> None:
    """A day without history is left out, a busier day reads higher, and the second call asks nobody."""
    hass = FakeHass()
    store = HistoryStore(hass, FakeCache(), "entropy_fm", logging.getLogger(__name__))  # type: ignore[arg-type]
    groups = [("room", [LAMP])]

    week = await store.week_energies(groups, TODAY, VIENNA, NOW, {})
    assert len(week) == len(HOURS_ON)
    assert week[0] > week[2] > week[1]
    calls = hass.calls

    assert await store.week_energies(groups, TODAY, VIENNA, NOW, {}) == week
    assert hass.calls == calls
    assert await store.week_energies([], TODAY, VIENNA, NOW, {}) == []


async def test_the_earlier_days_of_a_house_are_measured_together() -> None:
    """The rooms of a house count as one: the same days, each room read through its own scope."""
    hass = FakeHass()
    store = HistoryStore(hass, FakeCache(), "entropy_fm", logging.getLogger(__name__))  # type: ignore[arg-type]
    other = EntityInfo("light.other", "Other", "hall", VoiceKind.CHORD)
    week = await store.week_energies([("room", [LAMP]), ("hall", [other])], TODAY, VIENNA, NOW, {})
    assert len(week) == len(HOURS_ON)
