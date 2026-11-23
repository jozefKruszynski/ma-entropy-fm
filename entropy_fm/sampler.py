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
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synth import SAMPLE_RATE, SEMITONES_PER_OCTAVE

SAMPLES_DIR = Path(__file__).parent / "samples"
# the shift of a sample by resampling is kept within this many semitones of its root
MAX_SHIFT = 6.0
INSTRUMENT_FILE = "instrument.json"
KEY_SHOTS = "shots"
BYTES_PER_MB = 1e6
DEFAULT_CACHE_MB = 32.0
# the root of a one-shot, which is not pitched
ONE_SHOT_ROOT = 60
DEFAULT_VOICES = 16
DEFAULT_RELEASE = 0.2
# the loudness of a velocity follows this power
VELOCITY_CURVE = 0.8
MIDI_VELOCITY_MAX = 127
# a layer that spans nearly every velocity is not trimmed within itself
FULL_LAYER_SPAN = 126
# inside a layer the level rises from this share to full over the velocities of the layer
LAYER_TRIM_FLOOR = 0.55
LAYER_TRIM_RANGE = 0.45
UNITY_RATIO_TOLERANCE = 1e-9
# a note is made this many releases longer than the key is held
RELEASE_TAILS = 2.5
# a release dies away with a time constant of this share of it, at the least this many seconds
SAMPLE_RELEASE_DIVISOR = 3.0
MIN_RELEASE_TAU = 1e-3
# the click-free rise of a note that has no attack, in samples, and the fall at its end, in seconds
CLICK_FADE_SAMPLES = 32
END_FADE_SECONDS = 0.01


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

    def __init__(self, limit_mb: float = DEFAULT_CACHE_MB) -> None:
        """
        Make an empty cache.

        :param limit_mb: The most it keeps, in megabytes.
        """
        self.limit = int(limit_mb * BYTES_PER_MB)
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._bytes = 0
        self.loads = 0
        # live streams and the pre-render render in different threads
        self._lock = threading.Lock()

    def get(self, path: Path) -> np.ndarray:
        """
        Return the samples of a file as mono float32 at 44.1 kHz.

        :param path: An OGG (or any file that pedalboard reads).
        """
        with self._lock:
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
        with self._lock:
            # another thread may have decoded the same file meanwhile: keep its entry
            item = self._items.get(path)
            if item is not None:
                self._items.move_to_end(path)
                return item
            self.loads += 1
            self._items[path] = wave
            self._bytes += wave.nbytes
            while self._bytes > self.limit and len(self._items) > 1:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= dropped.nbytes
        return wave


_CACHE = SampleCache()


def velocity_gain(velocity: float) -> float:
    """Return the loudness of a velocity of 0 to 1: a gentle curve, so that soft hits stay audible."""
    return float(np.clip(velocity, 0.0, 1.0) ** VELOCITY_CURVE)


@dataclass
class Instrument:
    """A multisample: zones, an envelope and the shape of its voices."""

    name: str
    folder: Path
    zones: tuple[Zone, ...]
    attack: float = 0.0
    decay: float = 0.0
    sustain: float = 1.0
    release: float = DEFAULT_RELEASE
    voices: int = DEFAULT_VOICES
    cache: SampleCache = field(default_factory=lambda: _CACHE, repr=False)

    @classmethod
    def load(cls, folder: Path, cache: SampleCache | None = None) -> Instrument:
        """
        Read an instrument from its folder.

        :param folder: A folder with an `instrument.json` and the OGG files that it names.
        :param cache: Where the decoded samples are kept; shared by default.
        """
        data = json.loads((folder / INSTRUMENT_FILE).read_text())
        zones: list[Zone] = [Zone(**item) for item in data.get("zones", [])]
        for shots in data.get(KEY_SHOTS, {}).values():
            zones.extend(Zone(shot["file"], ONE_SHOT_ROOT, name=shot["name"]) for shot in shots)
        return cls(
            data.get("name", folder.name),
            folder,
            tuple(zones),
            data.get("attack", 0.0),
            data.get("decay", 0.0),
            data.get("sustain", 1.0),
            data.get("release", DEFAULT_RELEASE),
            data.get("voices", DEFAULT_VOICES),
            cache or _CACHE,
        )

    def fold(self, note: float, group: str = "") -> float:
        """Return a note moved by octaves into the range of the instrument (a sample is never shifted too far)."""
        roots = [zone.root for zone in self.zones if not group or zone.name == group] or [
            zone.root for zone in self.zones
        ]
        low, high = min(roots) - MAX_SHIFT, max(roots) + MAX_SHIFT
        while note > high:
            note -= SEMITONES_PER_OCTAVE
        while note < low:
            note += SEMITONES_PER_OCTAVE
        return note

    def zone_for(self, note: float, velocity: float, group: str = "") -> Zone:
        """
        Return the zone that plays a key at a velocity: the nearest key if none covers it.

        :param note: The MIDI note.
        :param velocity: 0 to 1.
        :param group: Only the zones of this name (the violins of the strings, say).
        """
        key = round(note)
        vel = int(np.clip(round(velocity * MIDI_VELOCITY_MAX), 1, MIDI_VELOCITY_MAX))
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

    def render(
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
        ratio = 2.0 ** (shift / SEMITONES_PER_OCTAVE)
        looped = zone.loop_end > zone.loop_start > 0 and duration is not None
        length = int((len(source) - 1) / ratio)
        if duration is not None:
            wanted = int((duration + self.release * RELEASE_TAILS) * SAMPLE_RATE)
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
            wave = read_cubic(source, position)
        else:
            wave = (
                source
                if abs(ratio - 1.0) < UNITY_RATIO_TOLERANCE
                else read_cubic(source, np.arange(length) * ratio)
            )
        wave = np.array(wave[:length], dtype=np.float32)
        wave *= np.float32(zone.gain * velocity_gain(velocity) * self._layer_trim(zone, velocity))
        return self._envelope(wave, duration, self.attack if attack is None else attack)

    @staticmethod
    def _layer_trim(zone: Zone, velocity: float) -> float:
        """Return a gain that lets the velocity move the level inside a layer, not only between layers."""
        if zone.vhi - zone.vlo >= FULL_LAYER_SPAN:
            return 1.0
        vel = np.clip(velocity * MIDI_VELOCITY_MAX, zone.vlo, zone.vhi)
        position = (vel - zone.vlo) / max(zone.vhi - zone.vlo, 1)
        return float(LAYER_TRIM_FLOOR + LAYER_TRIM_RANGE * position)

    def _envelope(self, wave: np.ndarray, duration: float | None, attack: float) -> np.ndarray:
        """Apply the attack, the decay to the sustain, and the release after the key goes up."""
        count = len(wave)
        env = np.ones(count, dtype=np.float32)
        if attack > 0.0:
            n = min(int(attack * SAMPLE_RATE), count)
            env[:n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        else:
            n = min(CLICK_FADE_SAMPLES, count)
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
                env[off:] *= np.exp(
                    -tail / max(self.release / SAMPLE_RELEASE_DIVISOR, MIN_RELEASE_TAU)
                ).astype(np.float32)
        tail_n = min(int(END_FADE_SECONDS * SAMPLE_RATE), count)
        env[-tail_n:] *= np.linspace(1.0, 0.0, tail_n, dtype=np.float32)
        return wave * env


# the Catmull-Rom spline through four neighbours y0..y3 is a t^3 + b t^2 + c t + y1, where
# a = A_Y0 y0 + A_Y1 y1 - A_Y2 y2 + A_Y3 y3, b = y0 - B_Y1 y1 + B_Y2 y2 - B_Y3 y3 and c = C_Y0 y0 + C_Y2 y2
CATMULL_A_Y0 = -0.5
CATMULL_A_Y1 = 1.5
CATMULL_A_Y2 = 1.5
CATMULL_A_Y3 = 0.5
CATMULL_B_Y1 = 2.5
CATMULL_B_Y2 = 2.0
CATMULL_B_Y3 = 0.5
CATMULL_C_Y0 = -0.5
CATMULL_C_Y2 = 0.5
# the read takes one zero before the wave and three after it, for the neighbours of the first and last samples
CUBIC_PAD_BEFORE = 1
CUBIC_PAD_AFTER = 3
CUBIC_TAPS = 4


def read_cubic(wave: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    Read a wave at fractional positions with a cubic (Catmull-Rom) interpolation.

    :param wave: Mono samples.
    :param position: Where to read, in samples.
    """
    padded = np.concatenate(
        (np.zeros(CUBIC_PAD_BEFORE, np.float32), wave, np.zeros(CUBIC_PAD_AFTER, np.float32))
    )
    base = np.floor(position).astype(np.int64)
    frac = (position - base).astype(np.float32)
    base = np.clip(base, 0, len(wave))
    y0, y1, y2, y3 = (padded[base + k] for k in range(CUBIC_TAPS))
    a = CATMULL_A_Y0 * y0 + CATMULL_A_Y1 * y1 - CATMULL_A_Y2 * y2 + CATMULL_A_Y3 * y3
    b = y0 - CATMULL_B_Y1 * y1 + CATMULL_B_Y2 * y2 - CATMULL_B_Y3 * y3
    c = CATMULL_C_Y0 * y0 + CATMULL_C_Y2 * y2
    return np.asarray(((a * frac + b) * frac + c) * frac + y1, dtype=np.float32)


def _resample(wave: np.ndarray, ratio: float) -> np.ndarray:
    """Return a wave at `ratio` times the rate, by a cubic read."""
    count = int(len(wave) * ratio)
    return read_cubic(wave, np.arange(count) / ratio)
