"""Tests of the pins, the library membership and the audio cache, on temporary folders."""

import asyncio
import gzip
import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.audiocache import AudioCache, cache_key
from entropy_fm.const import ENGINE_VERSION
from entropy_fm.pins import (
    Membership,
    Pin,
    PinStore,
    entity_from_dict,
    entity_to_dict,
    load_membership,
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
    recorder = cache.recorder(key)
    assert recorder is not None
    for _ in range(3):
        await recorder.feed(_wav_like(1.0))
    assert not cache.path(key).exists()
    assert await recorder.commit()
    path = cache.get(key)
    assert path is not None
    assert path.read_bytes()[:4] == b"OggS"
    assert cache.hits == 1
    assert cache.recorder(key) is None
    # a half written file under the real name is a miss and is deleted
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    assert cache.get(key) is None
    assert not path.exists()
    path.write_bytes(b"x" * 5000)
    assert cache.get(key) is None
    assert not path.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_an_aborted_recording_leaves_no_entry_and_no_temporary_file(tmp_path: Path) -> None:
    """A track that is stopped halfway never becomes an entry."""
    cache = AudioCache(tmp_path / "cache", 50)
    recorder = cache.recorder("stopped")
    assert recorder is not None
    await recorder.feed(_wav_like(1.0))
    await recorder.abort()
    assert list((tmp_path / "cache").iterdir()) == []
    assert cache.get("stopped") is None
    # nothing was fed: a commit makes no entry either
    empty = cache.recorder("empty")
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
        recorder = cache.recorder(key)
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


def test_a_cache_of_zero_is_off(tmp_path: Path) -> None:
    """With a limit of 0 nothing is recorded and nothing is found."""
    cache = AudioCache(tmp_path / "cache", 0)
    assert not cache.enabled
    assert cache.recorder("x") is None
    assert cache.get("x") is None
    assert asyncio.get_event_loop_policy() is not None
