"""Tests of the pins, the library membership and the audio cache, on temporary folders."""

import asyncio
import gzip
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.audiocache import AudioCache, cache_key
from entropy_fm.constants import ENGINE_VERSION
from entropy_fm.pins import (
    BROKEN_SUFFIX,
    INDEX_FILE,
    Membership,
    Pin,
    PinStore,
    entity_from_dict,
    entity_to_dict,
    load_membership,
    safe_name,
    save_membership,
    write_atomic,
)
from entropy_fm.timeline import EntityInfo, VoiceKind

DAY = date(2026, 11, 10)


def _pin(area: str = "kitchen", day: date = DAY, *, complete: bool = True) -> Pin:
    info = EntityInfo("sensor.fridge_power", "Fridge", area, VoiceKind.DRONE, why="role: fridge")
    return Pin(
        area,
        "Kitchen",
        day,
        [entity_to_dict(info)],
        {"sensor.fridge_power": {"t": [0.0, 60.0], "v": [1.0, 80.0]}},
        {"entity": {}, "data": {"sun.sun": {"t": [0.0], "v": [1.0]}}},
        complete=complete,
    )


def test_a_name_of_only_dots_is_never_a_folder_name(tmp_path: Path) -> None:
    """Separators and dots cannot climb out of the pin folder."""
    assert [safe_name(name) for name in ("..", ".", "...", "", "a/../b")] == [
        "_",
        "_",
        "_",
        "_",
        "a_.._b",
    ]
    store = PinStore(tmp_path / "root")
    assert store.path("..", date(2026, 11, 10)).parent == tmp_path / "root" / "pins" / "_"


def test_a_pin_is_written_compressed_read_back_and_deleted(tmp_path: Path) -> None:
    """A pin round-trips, lives under the folder it was given, and is deleted with its folder."""
    store = PinStore(tmp_path / "entropy_fm")
    assert store.read("kitchen", DAY) is None
    store.write(_pin())
    path = store.path("kitchen", DAY)
    assert path.is_file()
    assert path.is_relative_to(tmp_path / "entropy_fm")
    assert json.loads(gzip.decompress(path.read_bytes()))["engine"] == ENGINE_VERSION
    pin = store.read("kitchen", DAY)
    assert pin is not None
    assert pin.history["sensor.fridge_power"]["v"] == [1.0, 80.0]
    assert pin.engine == ENGINE_VERSION
    assert entity_from_dict(pin.entities[0]).kind == VoiceKind.DRONE
    assert store.days("kitchen") == [DAY]
    assert store.all() == [("kitchen", DAY)]
    store.delete("kitchen", DAY)
    assert not store.has("kitchen", DAY)
    assert not path.parent.exists()
    # the pin is small: a day of a handful of entities is a few hundred bytes
    store.write(_pin())
    assert path.stat().st_size < 2000


def test_a_damaged_pin_counts_as_missing_and_is_removed(tmp_path: Path) -> None:
    """A truncated or foreign file never raises: it is deleted and the day is just not pinned."""
    store = PinStore(tmp_path)
    store.write(_pin())
    path = store.path("kitchen", DAY)
    path.write_bytes(path.read_bytes()[:20])
    assert store.read("kitchen", DAY) is None
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(b'{"format": 99}'))
    assert store.read("kitchen", DAY) is None


def test_a_pin_of_an_unknown_format_stays_on_disk(tmp_path: Path) -> None:
    """A newer or older engine may own the file: it may be the only copy of a purged day."""
    store = PinStore(tmp_path)
    path = store.path("kitchen", DAY)
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b'{"format": 99}'))
    assert store.read("kitchen", DAY) is None
    assert path.exists()
    path.write_bytes(gzip.compress(b'{"format": 1}'))
    assert store.read("kitchen", DAY) is None
    assert path.exists()


def test_a_corrupt_gzip_pin_is_deleted(tmp_path: Path) -> None:
    """A file that is gzip but not JSON is damaged, not foreign: it goes."""
    store = PinStore(tmp_path)
    path = store.path("kitchen", DAY)
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"not json at all"))
    assert store.read("kitchen", DAY) is None
    assert not path.exists()


def test_a_broken_library_file_is_moved_aside(tmp_path: Path) -> None:
    """The membership is the only record of what was hearted: a broken file is kept as a backup."""
    (tmp_path / INDEX_FILE).write_text("{not json")
    assert load_membership(tmp_path).followed == {}
    assert not (tmp_path / INDEX_FILE).exists()
    assert (tmp_path / (INDEX_FILE + BROKEN_SUFFIX)).read_text() == "{not json"
    assert load_membership(tmp_path).followed == {}


def test_a_pin_that_cannot_be_read_stays_and_a_foreign_file_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read error is not proof that the pin is bad: it may be the only copy of a purged day."""
    store = PinStore(tmp_path)
    store.write(_pin())
    path = store.path("kitchen", DAY)

    def denied(_self: Path) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", denied)
    assert store.read("kitchen", DAY) is None
    assert path.exists()
    monkeypatch.undo()
    path.write_bytes(b"not gzip at all")
    assert store.read("kitchen", DAY) is None
    assert not path.exists()


def test_writers_of_one_file_at_once_do_not_share_a_temporary_file(tmp_path: Path) -> None:
    """Threads writing the same file each get their own temporary file and none fails."""
    target = tmp_path / "sub" / "file.bin"
    writers = 16
    with ThreadPoolExecutor(writers) as pool:
        list(pool.map(lambda n: write_atomic(target, bytes([n]) * 1000), range(writers)))
    assert len(set(target.read_bytes())) == 1
    assert [p.name for p in target.parent.iterdir()] == ["file.bin"]


def test_an_atomic_write_leaves_nothing_half_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the write fails the old file stays and no temporary file is left."""
    target = tmp_path / "file.bin"
    write_atomic(target, b"old")

    def broken(*_args: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["file.bin"]


def test_membership_knows_which_pins_each_item_needs(tmp_path: Path) -> None:
    """Removing an item frees only the pins that nothing else needs; the file round-trips."""
    other = date(2026, 11, 11)
    member = Membership()
    member.albums.add("kitchen|2026-11-10")
    member.add("kitchen|2026-11-10", [("kitchen", DAY)])
    member.tracks.add("kitchen|2026-11-10|_scape|berlin_school")
    member.add("kitchen|2026-11-10|_scape|berlin_school", [("kitchen", DAY)])
    member.followed["kitchen"] = "2026-11-01"
    member.add("@artist|kitchen", [("kitchen", DAY), ("kitchen", other)])
    member.snapshot["kitchen"] = {"name": "Kitchen", "entities": []}
    assert member.needed() == {("kitchen", DAY.isoformat()), ("kitchen", other.isoformat())}
    assert member.remove("kitchen|2026-11-10") == set()
    assert member.remove("kitchen|2026-11-10|_scape|berlin_school") == set()
    assert member.remove("@artist|kitchen") == {
        ("kitchen", DAY.isoformat()),
        ("kitchen", other.isoformat()),
    }
    save_membership(tmp_path, member)
    again = load_membership(tmp_path)
    assert again.albums == member.albums
    assert again.snapshot == member.snapshot
    (tmp_path / "library.json").write_text("not json")
    assert load_membership(tmp_path) == Membership()


def _wav_like(seconds: float) -> bytes:
    """Return silence as 16 bit stereo PCM."""
    return np.zeros(int(seconds * 44100) * 2, dtype="<i2").tobytes()


def test_the_cache_key_changes_with_anything_that_changes_the_sound() -> None:
    """A different engine, tier or setting is another entry."""
    first = cache_key(ENGINE_VERSION, "standard", "kitchen|2026-11-10|_scape", 60)
    assert first == cache_key(ENGINE_VERSION, "standard", "kitchen|2026-11-10|_scape", 60)
    assert first != cache_key(ENGINE_VERSION + 1, "standard", "kitchen|2026-11-10|_scape", 60)
    assert first != cache_key(ENGINE_VERSION, "rich", "kitchen|2026-11-10|_scape", 60)
    assert first != cache_key(ENGINE_VERSION, "standard", "kitchen|2026-11-10|_scape", 120)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_a_track_is_recorded_hit_missed_and_corrupt_entries_are_dropped(
    tmp_path: Path,
) -> None:
    """A finished recording is an Opus file that the cache hands out; damage means a miss."""
    cache = AudioCache(tmp_path / "entropy_fm", 50)
    key = cache_key("a")
    assert cache.get(key) is None
    assert cache.misses == 1
    recorder = await cache.recorder(key)
    assert recorder is not None
    for _ in range(3):
        await recorder.feed(_wav_like(1.0))
    assert not cache.path(key).exists()
    assert await recorder.commit()
    path = cache.get(key)
    assert path is not None
    assert path.read_bytes()[:4] == b"OggS"
    assert cache.hits == 1
    assert await cache.recorder(key) is None
    # a half written file under the real name is a miss and is deleted
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    assert cache.get(key) is None
    assert not path.exists()
    path.write_bytes(b"x" * 5000)
    assert cache.get(key) is None
    assert not path.exists()


async def test_a_track_is_recorded_by_one_recorder_at_a_time_each_with_its_own_file(
    tmp_path: Path,
) -> None:
    """A second recorder of a key that is being recorded is refused until the first one is done."""
    cache = AudioCache(tmp_path / "entropy_fm", 50)
    key = cache_key("b")
    first = await cache.recorder(key)
    assert first is not None
    assert await cache.recorder(key) is None
    other = await cache.recorder(cache_key("c"))
    assert other is not None
    assert other.temp != first.temp
    assert first.temp.suffix == ".part"
    await first.abort()
    again = await cache.recorder(key)
    assert again is not None
    assert again.temp != first.temp


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_an_aborted_recording_leaves_no_entry_and_no_temporary_file(tmp_path: Path) -> None:
    """A track that is stopped halfway never becomes an entry."""
    cache = AudioCache(tmp_path / "cache", 50)
    recorder = await cache.recorder("stopped")
    assert recorder is not None
    await recorder.feed(_wav_like(1.0))
    await recorder.abort()
    assert list((tmp_path / "cache").iterdir()) == []
    assert cache.get("stopped") is None
    # nothing was fed: a commit makes no entry either
    empty = await cache.recorder("empty")
    assert empty is not None
    assert not await empty.commit()
    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_the_least_recently_used_entries_go_first(tmp_path: Path) -> None:
    """When the folder is over the limit the oldest used files are deleted, the newest kept."""
    cache = AudioCache(tmp_path / "cache", 1)
    keys = []
    for index in range(4):
        key = cache_key("entry", index)
        recorder = await cache.recorder(key)
        assert recorder is not None
        await recorder.feed(
            np.random.default_rng(index).integers(-3000, 3000, 44100 * 8, dtype="<i2").tobytes()
        )
        assert await recorder.commit()
        keys.append(key)
        os.utime(cache.path(key), (1000 + index, 1000 + index))
    size = cache.path(keys[0]).stat().st_size
    assert cache.size() >= size
    cache.limit = int(size * 2.5)
    # using the oldest entry makes it the newest
    os.utime(cache.path(keys[0]))
    assert cache.evict() >= 1
    assert cache.get(keys[0]) is not None
    assert cache.get(keys[3]) is not None
    assert cache.get(keys[1]) is None
    assert cache.size() <= cache.limit


async def test_a_cache_of_zero_is_off(tmp_path: Path) -> None:
    """With a limit of 0 nothing is recorded and nothing is found."""
    cache = AudioCache(tmp_path / "cache", 0)
    assert not cache.enabled
    assert await cache.recorder("x") is None
    assert cache.get("x") is None
    assert asyncio.get_event_loop_policy() is not None
