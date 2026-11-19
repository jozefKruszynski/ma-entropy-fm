"""
A small sampler in numpy.

Multisamples with key and velocity zones, pitch shifting by resampling, an ADSR with a release at note-off,
sustain loops, a voice limit and choke groups, and a load-on-demand cache.

Instruments are folders of OGG files under `samples/` with an `instrument.json` (see
`scripts/fetch_samples.py`). Nothing here is random: the same triggers give the same samples, to the
last bit. The work is cheap, one gather and one multiply per sample, and a rendered note is cached, so
a repeated drum hit costs a lookup.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .synth import SAMPLE_RATE, pan_gains

if TYPE_CHECKING:
    from collections.abc import Sequence

SAMPLES_DIR = Path(__file__).parent / "samples"
# the longest a voice can ring, in seconds, which bounds the search for the voices of a block
MAX_VOICE_SECONDS = 12.0
CUT_FADE_SECONDS = 0.012
# the shift of a sample by resampling is kept within this many semitones of its root
MAX_SHIFT = 6.0


@dataclass(frozen=True)
class Zone:
    """A sample and the keys and velocities (1 to 127) that it plays for."""

    file: str
    root: float
    lo: int = 0
    hi: int = 127
    vlo: int = 1
    vhi: int = 127
    gain: float = 1.0
    name: str = ""
    # a sustain loop, in samples at 44.1 kHz (0 means the sample does not loop)
    loop_start: int = 0
    loop_end: int = 0

    def covers(self, note: int, velocity: int) -> bool:
        """Return True when the zone plays for a key and a velocity."""
        return self.lo <= note <= self.hi and self.vlo <= velocity <= self.vhi


class SampleCache:
    """Decoded samples, loaded when first needed, the least recently used dropped over a size."""

    def __init__(self, limit_mb: float = 32.0) -> None:
        """
        Make an empty cache.

        :param limit_mb: The most it keeps, in megabytes.
        """
        self.limit = int(limit_mb * 1e6)
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._bytes = 0
        self.loads = 0

    def get(self, path: Path) -> np.ndarray:
        """
        Return the samples of a file as mono float32 at 44.1 kHz.

        :param path: An OGG (or any file that pedalboard reads).
        """
        item = self._items.get(path)
        if item is not None:
            self._items.move_to_end(path)
            return item
        from pedalboard.io import AudioFile  # noqa: PLC0415

        with AudioFile(str(path)) as handle:
            data = handle.read(handle.frames)
            rate = handle.samplerate
        wave = np.ascontiguousarray(data.mean(axis=0), dtype=np.float32)
        if int(rate) != SAMPLE_RATE:
            wave = _resample(wave, SAMPLE_RATE / float(rate))
        self.loads += 1
        self._items[path] = wave
        self._bytes += wave.nbytes
        while self._bytes > self.limit and len(self._items) > 1:
            _, dropped = self._items.popitem(last=False)
            self._bytes -= dropped.nbytes
        return wave

    @property
    def size(self) -> int:
        """Return how many samples are held."""
        return len(self._items)


def _resample(wave: np.ndarray, ratio: float) -> np.ndarray:
    """Return a wave at `ratio` times the rate, by a cubic read."""
    count = int(len(wave) * ratio)
    return _read(wave, np.arange(count) / ratio)


def _read(wave: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Read a wave at fractional positions with a cubic (Catmull-Rom) interpolation."""
    padded = np.concatenate((np.zeros(1, np.float32), wave, np.zeros(3, np.float32)))
    base = np.floor(position).astype(np.int64)
    frac = (position - base).astype(np.float32)
    base = np.clip(base, 0, len(wave))
    y0, y1, y2, y3 = (padded[base + k] for k in range(4))
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    return np.asarray(((a * frac + b) * frac + c) * frac + y1, dtype=np.float32)


_CACHE = SampleCache()


def velocity_gain(velocity: float) -> float:
    """Return the loudness of a velocity of 0 to 1: a gentle curve, so that soft hits stay audible."""
    return float(np.clip(velocity, 0.0, 1.0) ** 0.8)


@dataclass
class Instrument:
    """A multisample: zones, an envelope and the shape of its voices."""

    name: str
    folder: Path
    zones: tuple[Zone, ...]
    attack: float = 0.0
    decay: float = 0.0
    sustain: float = 1.0
    release: float = 0.2
    voices: int = 16
    cache: SampleCache = field(default_factory=lambda: _CACHE, repr=False)

    @classmethod
    def load(cls, folder: Path, cache: SampleCache | None = None) -> Instrument:
        """
        Read an instrument from its folder.

        :param folder: A folder with an `instrument.json` and the OGG files that it names.
        :param cache: Where the decoded samples are kept; shared by default.
        """
        data = json.loads((folder / "instrument.json").read_text())
        zones: list[Zone] = [Zone(**item) for item in data.get("zones", [])]
        for group, shots in data.get("shots", {}).items():
            zones.extend(Zone(shot["file"], 60, name=shot["name"]) for shot in shots)
            del group
        return cls(
            data.get("name", folder.name),
            folder,
            tuple(zones),
            data.get("attack", 0.0),
            data.get("decay", 0.0),
            data.get("sustain", 1.0),
            data.get("release", 0.2),
            data.get("voices", 16),
            cache or _CACHE,
        )

    @property
    def groups(self) -> dict[str, list[str]]:
        """Return the names of the one-shots of an instrument by group (kick, hat and so on)."""
        data = json.loads((self.folder / "instrument.json").read_text())
        return {
            group: [shot["name"] for shot in shots]
            for group, shots in data.get("shots", {}).items()
        }

    def fold(self, note: float, group: str = "") -> float:
        """Return a note moved by octaves into the range of the instrument (a sample is never shifted too far)."""
        roots = [zone.root for zone in self.zones if not group or zone.name == group] or [
            zone.root for zone in self.zones
        ]
        low, high = min(roots) - MAX_SHIFT, max(roots) + MAX_SHIFT
        while note > high:
            note -= 12.0
        while note < low:
            note += 12.0
        return note

    def zone_for(self, note: float, velocity: float, group: str = "") -> Zone:
        """
        Return the zone that plays a key at a velocity: the nearest key if none covers it.

        :param note: The MIDI note.
        :param velocity: 0 to 1.
        :param group: Only the zones of this name (the violins of the strings, say).
        """
        key = round(note)
        vel = int(np.clip(round(velocity * 127.0), 1, 127))
        pool = [zone for zone in self.zones if not group or zone.name == group] or list(self.zones)
        found = [zone for zone in pool if zone.covers(key, vel)]
        if found:
            return min(found, key=lambda zone: abs(zone.root - note))
        layered = [zone for zone in pool if zone.vlo <= vel <= zone.vhi] or pool
        return min(layered, key=lambda zone: abs(zone.root - note))

    def named(self, name: str) -> Zone:
        """
        Return the one-shot called `name`.

        :param name: The name from `instrument.json`.
        """
        for zone in self.zones:
            if zone.name == name:
                return zone
        msg = f"{self.name} has no sample {name}"
        raise KeyError(msg)

    def data(self, zone: Zone) -> np.ndarray:
        """Return the decoded samples of a zone."""
        return self.cache.get(self.folder / zone.file)

    def render(  # noqa: PLR0913, PLR0917
        self,
        note: float = 60.0,
        velocity: float = 0.8,
        duration: float | None = None,
        name: str = "",
        semitones: float = 0.0,
        group: str = "",
        attack: float | None = None,
    ) -> np.ndarray:
        """
        Return one note as mono float32.

        The sample is read at the ratio that moves its root to the note (or by `semitones`, for a
        one-shot), a sustain loop is read round and round while the key is held, the envelope is
        applied, and the note ends `release` seconds after `duration`.

        :param note: The MIDI note, ignored for a named one-shot.
        :param velocity: 0 to 1.
        :param duration: How long the key is held in seconds, or None to play the whole sample.
        :param name: A one-shot by name.
        :param semitones: A shift of a one-shot, in semitones.
        :param group: Only the zones of this name are used.
        :param attack: The attack in seconds, instead of the one of the instrument.
        """
        if not name:
            note = self.fold(note, group)
        zone = self.named(name) if name else self.zone_for(note, velocity, group)
        source = self.data(zone)
        shift = semitones if name else float(np.clip(note - zone.root, -MAX_SHIFT, MAX_SHIFT))
        ratio = 2.0 ** (shift / 12.0)
        looped = zone.loop_end > zone.loop_start > 0 and duration is not None
        length = int((len(source) - 1) / ratio)
        if duration is not None:
            wanted = int((duration + self.release * 2.5) * SAMPLE_RATE)
            length = wanted if looped else min(length, wanted)
        length = max(length, 1)
        if looped:
            position = np.arange(length) * ratio
            span = zone.loop_end - zone.loop_start
            position = np.where(
                position >= zone.loop_end,
                zone.loop_start + (position - zone.loop_start) % span,
                position,
            )
            wave = _read(source, position)
        else:
            wave = source if abs(ratio - 1.0) < 1e-9 else _read(source, np.arange(length) * ratio)
        wave = np.array(wave[:length], dtype=np.float32)
        wave *= np.float32(zone.gain * velocity_gain(velocity) * self._layer_trim(zone, velocity))
        return self._envelope(wave, duration, self.attack if attack is None else attack)

    @staticmethod
    def _layer_trim(zone: Zone, velocity: float) -> float:
        """Return a gain that lets the velocity move the level inside a layer, not only between layers."""
        if zone.vhi - zone.vlo >= 126:
            return 1.0
        vel = np.clip(velocity * 127.0, zone.vlo, zone.vhi)
        position = (vel - zone.vlo) / max(zone.vhi - zone.vlo, 1)
        return float(0.55 + 0.45 * position)

    def _envelope(self, wave: np.ndarray, duration: float | None, attack: float) -> np.ndarray:
        """Apply the attack, the decay to the sustain, and the release after the key goes up."""
        count = len(wave)
        env = np.ones(count, dtype=np.float32)
        if attack > 0.0:
            n = min(int(attack * SAMPLE_RATE), count)
            env[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        else:
            n = min(32, count)
            env[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if self.decay > 0.0 and self.sustain < 1.0:
            t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
            env = env * (self.sustain + (1.0 - self.sustain) * np.exp(-t / self.decay)).astype(
                np.float32
            )
        if duration is not None:
            off = int(duration * SAMPLE_RATE)
            if off < count:
                tail = np.arange(count - off, dtype=np.float32) / SAMPLE_RATE
                env[off:] *= np.exp(-tail / max(self.release / 3.0, 1e-3)).astype(np.float32)
        tail_n = min(int(0.01 * SAMPLE_RATE), count)
        env[-tail_n:] *= np.linspace(1.0, 0.0, tail_n, dtype=np.float32)
        return wave * env


@dataclass(frozen=True)
class Trigger:
    """A note or a hit to play: when, what, how hard, how long, and in which choke group."""

    time: float
    instrument: str
    velocity: float
    note: float = 60.0
    name: str = ""
    duration: float | None = None
    semitones: float = 0.0
    group: str = ""
    attack: float | None = None
    # a hit cuts the voices of the same group that are still ringing (an open hat under a closed one)
    choke: str = ""
    pan: float = 0.0
    gain: float = 1.0
    # which bus of the mix it goes to
    bus: str = ""


@dataclass
class Voice:
    """A trigger that was admitted: where it starts in samples and where it was cut, if it was."""

    trigger: Trigger
    start: int
    cut: int | None = None


class Sampler:
    """Plays triggers from a set of instruments, with a voice limit and choke groups."""

    def __init__(self, instruments: dict[str, Instrument], limit: int = 32) -> None:
        """
        Make a sampler.

        :param instruments: The instruments by name.
        :param limit: The most voices that ring at once; the oldest one is cut when it is passed.
        """
        self.instruments = instruments
        self.limit = limit
        self._waves: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._wave_bytes = 0
        self.wave_limit = 24_000_000

    @classmethod
    def from_folder(
        cls, names: Sequence[str], root: Path | None = None, limit: int = 32
    ) -> Sampler:
        """
        Make a sampler of the instruments in folders of `samples/`.

        :param names: The folder names.
        :param root: The folder of the folders, `entropy_fm/samples` by default.
        :param limit: The voice limit.
        """
        base = root or SAMPLES_DIR
        return cls({name: Instrument.load(base / name) for name in names}, limit)

    def schedule(self, triggers: Sequence[Trigger]) -> list[Voice]:
        """
        Admit the triggers in time order and cut voices: choke groups first, then the voice limit.

        The result is sorted by start and does not depend on the order of the input.

        :param triggers: Everything that is to play.
        """
        ordered = sorted(triggers, key=lambda t: (t.time, t.instrument, t.name, t.note))
        voices: list[Voice] = []
        live: list[Voice] = []
        for trigger in ordered:
            start = round(trigger.time * SAMPLE_RATE)
            live = [voice for voice in live if self._end(voice) > start]
            if trigger.choke:
                for voice in live:
                    if voice.trigger.choke == trigger.choke and voice.cut is None:
                        voice.cut = start
            same = [v for v in live if v.cut is None and v.trigger.instrument == trigger.instrument]
            limit = self._limit_of(trigger)
            while len(same) >= limit:
                oldest = min(same, key=lambda v: v.start)
                oldest.cut = start
                same.remove(oldest)
            voice = Voice(trigger, start)
            voices.append(voice)
            live.append(voice)
        return voices

    def _limit_of(self, trigger: Trigger) -> int:
        """Return the voice limit that applies: the instrument's own, or the sampler's."""
        own = self.instruments[trigger.instrument].voices
        return min(own, self.limit)

    def _end(self, voice: Voice) -> int:
        """Return the sample at which a voice has ended, counting its release."""
        trigger = voice.trigger
        instrument = self.instruments[trigger.instrument]
        if voice.cut is not None:
            return voice.cut + int(CUT_FADE_SECONDS * SAMPLE_RATE)
        held = trigger.duration if trigger.duration is not None else MAX_VOICE_SECONDS
        return voice.start + int(
            (min(held + instrument.release * 4.0, MAX_VOICE_SECONDS)) * SAMPLE_RATE
        )

    def wave(self, voice: Voice) -> np.ndarray:
        """
        Return the mono wave of a voice, with its cut applied, cached.

        :param voice: From `schedule`.
        """
        trigger = voice.trigger
        length_cut = None if voice.cut is None else voice.cut - voice.start
        key = (
            trigger.instrument,
            trigger.name,
            round(trigger.note, 2),
            round(trigger.semitones, 2),
            round(trigger.velocity * 127.0),
            None if trigger.duration is None else round(trigger.duration, 2),
            length_cut,
            trigger.group,
            trigger.attack,
        )
        found = self._waves.get(key)
        if found is not None:
            self._waves.move_to_end(key)
            return found
        wave = self.instruments[trigger.instrument].render(
            trigger.note,
            trigger.velocity,
            trigger.duration,
            trigger.name,
            trigger.semitones,
            trigger.group,
            trigger.attack,
        )
        if length_cut is not None:
            fade = int(CUT_FADE_SECONDS * SAMPLE_RATE)
            wave = wave[: length_cut + fade].copy()
            if len(wave) > length_cut:
                wave[length_cut:] *= np.linspace(1.0, 0.0, len(wave) - length_cut, dtype=np.float32)
        wave = wave * np.float32(trigger.gain)
        self._waves[key] = wave
        self._wave_bytes += wave.nbytes
        while self._wave_bytes > self.wave_limit and len(self._waves) > 1:
            _, dropped = self._waves.popitem(last=False)
            self._wave_bytes -= dropped.nbytes
        return wave

    def mix(
        self, voices: Sequence[Voice], first: int, count: int, bus: str | None = None
    ) -> np.ndarray:
        """
        Mix the voices that sound between two samples into a stereo block.

        :param voices: From `schedule`, sorted by start.
        :param first: The first sample of the block.
        :param count: The number of samples.
        :param bus: When given, only the voices of that bus.
        """
        out = np.zeros((count, 2), dtype=np.float32)
        starts = [voice.start for voice in voices]
        low = bisect_left(starts, first - int(MAX_VOICE_SECONDS * SAMPLE_RATE))
        high = bisect_right(starts, first + count)
        for voice in voices[low:high]:
            if bus is not None and voice.trigger.bus != bus:
                continue
            wave = self.wave(voice)
            begin = max(first, voice.start)
            end = min(first + count, voice.start + len(wave))
            if end <= begin:
                continue
            piece = wave[begin - voice.start : end - voice.start]
            left, right = pan_gains(voice.trigger.pan)
            out[begin - first : end - first, 0] += piece * np.float32(left)
            out[begin - first : end - first, 1] += piece * np.float32(right)
        return out
