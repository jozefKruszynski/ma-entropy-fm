"""
The audio cache of Entropy FM: rendered tracks of past days, kept as Opus files with LRU eviction.

A day that is over never changes, so a track that was rendered once can be kept and played from
the file, which can also be seeked and costs next to no CPU. The cache lives in the cache folder of
the server. An entry is written while the track plays (see `Recorder`): the audio goes to ffmpeg,
which encodes it into a temporary file, and only a track that played to the very end is moved to its
name, so a stopped or failed render never leaves an entry. Today is never cached. An entry that
looks damaged is deleted and counts as missing. When the folder is bigger than the limit the files
that were used least recently are deleted first.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import time
from pathlib import Path

BITRATE = "96k"
SUFFIX = ".opus"
PART = ".part"
MIN_BYTES = 200
# a file that the process of an earlier run left behind is stale after this long
STALE_PART_SECONDS = 3600.0


def cache_key(*parts: object) -> str:
    """
    Return the name of an entry: a hash of everything that changes what the track sounds like.

    :param parts: The values; their text is joined.
    """
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:40]


class AudioCache:
    """A folder of Opus files with a size limit."""

    def __init__(self, root: str | Path, limit_mb: int) -> None:
        """
        Use a folder; with a limit of 0 the cache is off.

        :param root: The folder, in the cache path of the server.
        :param limit_mb: The most megabytes it may hold, 0 for no cache.
        """
        self.root = Path(root)
        self.limit = int(limit_mb) * 1_000_000
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """Return whether the cache is on."""
        return self.limit > 0

    def path(self, key: str) -> Path:
        """
        Return where the entry of a key lives.

        :param key: A key made by `cache_key`.
        """
        return self.root / f"{key}{SUFFIX}"

    def get(self, key: str) -> Path | None:
        """
        Return the file of an entry, or None when it is missing or damaged.

        A hit counts as a use: the entry is the last to be evicted.

        :param key: A key made by `cache_key`.
        """
        if not self.enabled:
            return None
        path = self.path(key)
        if not path.is_file():
            self.misses += 1
            return None
        if not self._sound(path):
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        with contextlib.suppress(OSError):
            os.utime(path)
        self.hits += 1
        return path

    @staticmethod
    def _sound(path: Path) -> bool:
        """Check that a file is a whole Ogg file: it starts and ends with an Ogg page."""
        try:
            size = path.stat().st_size
            if size < MIN_BYTES:
                return False
            with path.open("rb") as handle:
                if handle.read(4) != b"OggS":
                    return False
                handle.seek(max(size - 65536, 0))
                tail = handle.read()
        except OSError:
            return False
        # the last page of a stream has the end-of-stream flag, bit 2 of the byte after the version
        last = tail.rfind(b"OggS")
        return last >= 0 and len(tail) > last + 5 and bool(tail[last + 5] & 0x04)

    def size(self) -> int:
        """Return the bytes that the entries use."""
        if not self.root.is_dir():
            return 0
        return sum(path.stat().st_size for path in self.root.glob(f"*{SUFFIX}") if path.is_file())

    def evict(self) -> int:
        """Delete the least recently used entries until the cache fits; return how many."""
        if not self.root.is_dir():
            return 0
        now = time.time()
        for part in self.root.glob(f"*{PART}"):
            with contextlib.suppress(OSError):
                if now - part.stat().st_mtime > STALE_PART_SECONDS:
                    part.unlink()
        entries = []
        for path in self.root.glob(f"*{SUFFIX}"):
            with contextlib.suppress(OSError):
                stat = path.stat()
                entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in entries)
        removed = 0
        for _, size, path in sorted(entries):
            if total <= self.limit:
                break
            with contextlib.suppress(OSError):
                path.unlink()
                total -= size
                removed += 1
        return removed

    def clear(self) -> None:
        """Delete everything."""
        if self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)

    def recorder(self, key: str) -> Recorder | None:
        """
        Start recording a track into the cache; None when the cache is off or has it already.

        :param key: A key made by `cache_key`.
        """
        if not self.enabled or self.path(key).exists():
            return None
        return Recorder(self, key)


class Recorder:
    """Encodes PCM that is fed to it into a temporary file, and commits it at the end."""

    def __init__(self, cache: AudioCache, key: str) -> None:
        """
        Prepare the recording; nothing starts before the first chunk.

        :param cache: The cache.
        :param key: The key of the entry.
        """
        self.cache = cache
        self.key = key
        self.temp = cache.root / f"{key}.{os.getpid()}{PART}"
        self._process: asyncio.subprocess.Process | None = None
        self.failed = False

    async def feed(self, chunk: bytes) -> None:
        """
        Give the encoder a chunk of 16 bit stereo PCM at 44.1 kHz.

        :param chunk: The bytes.
        """
        if self.failed:
            return
        try:
            if self._process is None:
                await self._start()
            process = self._process
            if process is None or process.stdin is None:
                return
            process.stdin.write(chunk)
            await process.stdin.drain()
        except OSError, RuntimeError, ConnectionError:
            self.failed = True
            await self.abort()

    async def _start(self) -> None:
        """Start ffmpeg."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.failed = True
            return
        self.cache.root.mkdir(parents=True, exist_ok=True)
        self._process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            BITRATE,
            "-f",
            "ogg",
            str(self.temp),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def commit(self) -> bool:
        """Finish the encoding and move the file into place; return whether the entry exists."""
        process = self._process
        if self.failed or process is None or process.stdin is None:
            await self.abort()
            return False
        try:
            process.stdin.close()
            code = await process.wait()
        except OSError, RuntimeError:
            await self.abort()
            return False
        final = self.cache.path(self.key)
        if code == 0 and self.cache._sound(self.temp):  # noqa: SLF001
            self.temp.replace(final)
            await asyncio.to_thread(self.cache.evict)
            return True
        await self.abort()
        return False

    async def abort(self) -> None:
        """Stop and delete the temporary file; the entry is never made."""
        process, self._process = self._process, None
        if process is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        self.temp.unlink(missing_ok=True)
