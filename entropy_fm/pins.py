"""
Pins: the days of the library that Entropy FM keeps for good, after Home Assistant has forgotten.

The recorder of Home Assistant keeps ten days by default. What the user adds to the library of
Music Assistant (an album, a track, or an area as an artist) is pinned: the event timeline of the
area on that day, compressed, is written once into the storage folder of the server (never into the
folder of the provider). A pinned day plays from its pin, rendered by the engine of the moment, so a
better engine makes old days sound better; the version of the engine that was current when the day
was pinned is in the pin. No audio is stored here, only the history that the music is made from.

`Membership` says what the user added and which pins every item needs, so that removing an item
deletes the pins that nothing else needs. Everything is written to a temporary file and moved into
place, so a crash cannot leave half a file.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .const import ENGINE_VERSION
from .timeline import EntityInfo, VoiceKind

PIN_FORMAT = 1
INDEX_FILE = "library.json"
PIN_FOLDER = "pins"
SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_name(text: str) -> str:
    """
    Return text that is safe as a file name.

    :param text: Any text, such as an area id.
    """
    return SAFE.sub("_", text) or "_"


def write_atomic(path: Path, data: bytes) -> None:
    """
    Write a file so that it is either whole or not there: temporary file, then rename.

    :param path: The file.
    :param data: Its content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass
class Pin:
    """One area on one day: the entities that played and what each did, as the cache stores it."""

    area_id: str
    area_name: str
    day: date
    # the EntityInfo of each entity, as plain dictionaries
    entities: list[dict[str, Any]]
    # entity id to {"t": [...], "v": [...]}
    history: dict[str, Any]
    # the entity that tells where the sun is and its history, when the house has one
    sun: dict[str, Any] | None = None
    engine: int = ENGINE_VERSION
    # False for a day that was still going on when it was pinned: it is read again later
    complete: bool = True
    format: int = PIN_FORMAT

    def to_json(self) -> dict[str, Any]:
        """Return the pin as plain data."""
        return {
            "format": self.format,
            "engine": self.engine,
            "area_id": self.area_id,
            "area_name": self.area_name,
            "day": self.day.isoformat(),
            "entities": self.entities,
            "history": self.history,
            "sun": self.sun,
            "complete": self.complete,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Pin:
        """
        Read a pin from plain data.

        :param data: What `to_json` made.
        :raises ValueError: When it is not a pin of this format.
        """
        if data.get("format") != PIN_FORMAT:
            msg = "unknown pin format"
            raise ValueError(msg)
        return cls(
            area_id=str(data["area_id"]),
            area_name=str(data["area_name"]),
            day=date.fromisoformat(data["day"]),
            entities=list(data["entities"]),
            history=dict(data["history"]),
            sun=data.get("sun"),
            engine=int(data.get("engine", 0)),
            complete=bool(data.get("complete", True)),
        )


class PinStore:
    """The pins on disk."""

    def __init__(self, root: str | Path) -> None:
        """
        Use a folder; it is made when the first pin is written.

        :param root: The folder for this provider in the storage path of the server.
        """
        self.root = Path(root)

    def path(self, area_id: str, day: date) -> Path:
        """
        Return where a pin lives.

        :param area_id: The area.
        :param day: The day.
        """
        return self.root / PIN_FOLDER / safe_name(area_id) / f"{day.isoformat()}.json.gz"

    def write(self, pin: Pin) -> None:
        """
        Write a pin.

        :param pin: What to keep.
        """
        raw = json.dumps(pin.to_json(), separators=(",", ":")).encode()
        write_atomic(self.path(pin.area_id, pin.day), gzip.compress(raw, 6, mtime=0))

    def read(self, area_id: str, day: date) -> Pin | None:
        """
        Read a pin; a file that cannot be read is deleted and counts as missing.

        :param area_id: The area.
        :param day: The day.
        """
        path = self.path(area_id, day)
        if not path.is_file():
            return None
        try:
            return Pin.from_json(json.loads(gzip.decompress(path.read_bytes())))
        except OSError, ValueError, KeyError, EOFError, TypeError:
            path.unlink(missing_ok=True)
            return None

    def has(self, area_id: str, day: date) -> bool:
        """
        Return whether a pin exists.

        :param area_id: The area.
        :param day: The day.
        """
        return self.path(area_id, day).is_file()

    def delete(self, area_id: str, day: date) -> None:
        """
        Delete a pin, and the folder of its area when that is empty.

        :param area_id: The area.
        :param day: The day.
        """
        path = self.path(area_id, day)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()

    def days(self, area_id: str) -> list[date]:
        """
        Return the pinned days of an area, newest first.

        :param area_id: The area.
        """
        folder = self.root / PIN_FOLDER / safe_name(area_id)
        found: list[date] = []
        if folder.is_dir():
            for path in folder.glob("*.json.gz"):
                try:
                    found.append(date.fromisoformat(path.name.removesuffix(".json.gz")))
                except ValueError:
                    continue
        return sorted(found, reverse=True)

    def all(self) -> list[tuple[str, date]]:
        """Return every pin as the id of its area and its day, read from the pins themselves."""
        found: list[tuple[str, date]] = []
        base = self.root / PIN_FOLDER
        if not base.is_dir():
            return found
        for folder in sorted(base.iterdir()):
            for day in self.days(folder.name):
                pin = self.read(folder.name, day)
                if pin is not None:
                    found.append((pin.area_id, day))
        return found


@dataclass
class Membership:
    """What the user added to the library, and the pins each of it needs."""

    # area id to the day it was added as an artist
    followed: dict[str, str] = field(default_factory=dict)
    albums: set[str] = field(default_factory=set)
    tracks: set[str] = field(default_factory=set)
    # an added item to the pins it needs: lists of [area id, day]
    needs: dict[str, list[list[str]]] = field(default_factory=dict)
    # what the library knows of an area that Home Assistant may no longer have: name and entities
    snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)
    # albums and tracks that the user removed from the library, to the ISO day they belong to: the library
    # window lists them again at every sync unless they are left out
    removed: dict[str, str] = field(default_factory=dict)

    def forget_removed_before(self, oldest: date) -> None:
        """
        Drop the removed items of days before a day: they are not listed any more, so nothing has to be left out.

        :param oldest: The first day that is still listed.
        """
        for item_id, day in list(self.removed.items()):
            try:
                old = date.fromisoformat(day) < oldest
            except ValueError:
                old = True
            if old:
                del self.removed[item_id]

    def needed(self) -> set[tuple[str, str]]:
        """Return every pin that something in the library needs, as area id and ISO day."""
        return {(area, day) for pairs in self.needs.values() for area, day in pairs}

    def add(self, key: str, pairs: list[tuple[str, date]]) -> None:
        """
        Say that an item needs some pins.

        :param key: The item: its id, or `@artist|<area id>` for an area that is followed.
        :param pairs: The area and the day of each pin.
        """
        known = {tuple(pair) for pair in self.needs.get(key, [])}
        known.update((area, day.isoformat()) for area, day in pairs)
        self.needs[key] = [list(pair) for pair in sorted(known)]

    def remove(self, key: str) -> set[tuple[str, str]]:
        """
        Forget an item and return the pins that nothing needs any longer.

        :param key: The item.
        """
        before = self.needed()
        self.needs.pop(key, None)
        return before - self.needed()

    def to_json(self) -> dict[str, Any]:
        """Return the membership as plain data."""
        return {
            "followed": self.followed,
            "albums": sorted(self.albums),
            "tracks": sorted(self.tracks),
            "needs": self.needs,
            "snapshot": self.snapshot,
            "removed": self.removed,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Membership:
        """
        Read the membership from plain data.

        :param data: What `to_json` made.
        """
        return cls(
            followed=dict(data.get("followed", {})),
            albums=set(data.get("albums", [])),
            tracks=set(data.get("tracks", [])),
            needs={
                key: [list(pair) for pair in pairs] for key, pairs in data.get("needs", {}).items()
            },
            snapshot=dict(data.get("snapshot", {})),
            removed={str(key): str(day) for key, day in dict(data.get("removed", {})).items()},
        )


def load_membership(root: str | Path) -> Membership:
    """
    Read the membership of the library; a file that is missing or broken gives an empty one.

    :param root: The folder of the provider in the storage path.
    """
    path = Path(root) / INDEX_FILE
    try:
        return Membership.from_json(json.loads(path.read_text()))
    except OSError, ValueError, TypeError, AttributeError:
        return Membership()


def save_membership(root: str | Path, membership: Membership) -> None:
    """
    Write the membership of the library.

    :param root: The folder of the provider in the storage path.
    :param membership: What to write.
    """
    write_atomic(Path(root) / INDEX_FILE, json.dumps(membership.to_json(), indent=1).encode())


def entity_to_dict(info: EntityInfo) -> dict[str, Any]:
    """
    Return an entity as plain data for a pin.

    :param info: The entity.
    """
    return {
        "entity_id": info.entity_id,
        "name": info.name,
        "area_id": info.area_id,
        "kind": info.kind.value,
        "cumulative": info.cumulative,
        "why": info.why,
        "device_class": info.device_class,
        "device_id": info.device_id,
    }


def entity_from_dict(data: dict[str, Any]) -> EntityInfo:
    """
    Read an entity from a pin.

    :param data: What `entity_to_dict` made.
    """
    return EntityInfo(
        str(data["entity_id"]),
        str(data["name"]),
        data.get("area_id"),
        VoiceKind(data["kind"]),
        bool(data.get("cumulative", False)),
        str(data.get("why", "")),
        str(data.get("device_class", "")),
        str(data.get("device_id", "")),
    )
