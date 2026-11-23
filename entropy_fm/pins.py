"""
Pins: the days of the library that Entropy FM keeps for good, after Home Assistant has forgotten.

The recorder of Home Assistant keeps ten days by default. What the user hearts in
Music Assistant (an album, a track, or a room as an artist) is pinned: the event timeline of the
area on that day, compressed, is written once into the storage folder of the server (never into the
folder of the provider). A pinned day plays from its pin, rendered by the engine of the moment, so a
better engine makes old days sound better; the version of the engine that was current when the day
was pinned is in the pin. No audio is stored here, only the history that the music is made from.

`Membership` says what the user hearted and which pins every item needs, so that un-hearting an item
deletes the pins that nothing else needs. Everything is written to a temporary file and moved into
place, so a crash cannot leave half a file.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import os
import re
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .constants import ENGINE_VERSION
from .timeline import EntityInfo, VoiceKind

LOGGER = logging.getLogger(__name__)
PIN_FORMAT = 1
INDEX_FILE = "library.json"
PIN_FOLDER = "pins"
SAFE = re.compile(r"[^A-Za-z0-9_.-]")
TEMP_SUFFIX = ".part"
BROKEN_SUFFIX = ".bak"


def safe_name(text: str) -> str:
    """
    Return text that is safe as a file name: no separators, and never a name of only dots.

    :param text: Any text, such as an area id.
    """
    name = SAFE.sub("_", text)
    return name if name.strip(".") else "_"


def write_atomic(path: Path, data: bytes) -> None:
    """
    Write a file so that it is either whole or not there: unique temporary file, then rename.

    Safe for threads and processes writing the same file at once: the last rename wins.

    :param path: The file.
    :param data: Its content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=TEMP_SUFFIX, dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
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
        Read a pin; a corrupt file is deleted and counts as missing, one of an unknown format or that cannot be read stays.

        :param area_id: The area.
        :param day: The day.
        """
        path = self.path(area_id, day)
        if not path.is_file():
            return None
        try:
            return Pin.from_json(json.loads(gzip.decompress(path.read_bytes())))
        except (gzip.BadGzipFile, zlib.error, EOFError, json.JSONDecodeError) as err:
            LOGGER.warning("Deleted the corrupt pin of %s on %s: %s", area_id, day, err)
            path.unlink(missing_ok=True)
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
            LOGGER.debug("Left the unreadable pin of %s on %s alone: %s", area_id, day, err)
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
    """What the user hearted (rooms, albums, tracks), and the pins each of it needs."""

    # area id to the day it was hearted as an artist: every day of the room is kept
    followed: dict[str, str] = field(default_factory=dict)
    albums: set[str] = field(default_factory=set)
    tracks: set[str] = field(default_factory=set)
    # an added item to the pins it needs: lists of [area id, day]
    needs: dict[str, list[list[str]]] = field(default_factory=dict)
    # what the library knows of an area that Home Assistant may no longer have: name and entities
    snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        )


def load_membership(root: str | Path) -> Membership:
    """
    Read the membership of the library; a missing file gives an empty one, a broken one is moved aside first.

    :param root: The folder of the provider in the storage path.
    """
    path = Path(root) / INDEX_FILE
    try:
        text = path.read_text()
    except OSError:
        return Membership()
    try:
        return Membership.from_json(json.loads(text))
    except (ValueError, TypeError, AttributeError) as err:
        broken = path.with_name(path.name + BROKEN_SUFFIX)
        LOGGER.warning("The library file could not be read and was moved to %s: %s", broken, err)
        with contextlib.suppress(OSError):
            path.replace(broken)
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
