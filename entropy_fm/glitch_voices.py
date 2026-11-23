"""
The notes of Cinematic Glitch that are not in the voice library: swells, stabs, booms, the growl bass, the kit.

A note of the score named `glitch:<what>` is made here, once, and kept in the cache of notes (`voices.NOTES`) like the
notes of the library, so that a chord that comes round again costs nothing. What a note carries:

| voice | midi | patch | vel |
|---|---|---|---|
| `glitch:swell`, `swell_slow`, `swell_big` | the root of the chord's pad | 0 minor, 1 major | the level against the design |
| `glitch:stab`, `stab_big`, `stab_long`, `stab_short` | the root of the chord's pad | 0 minor, 1 major | how hard |
| `glitch:revstab` | the root of the chord's pad | 0 minor, 1 major | the level against the design |
| `glitch:boom`, `boomfar` | the pitch | 0 | level and strength |
| `glitch:revsnare` | 0 | 0 | how hard the snare was |
| `glitch:growl` | the pitch | the row of `GROWL_STATES` times 8, plus the phase of the wobble in eighths of a cycle | the level |
| `glitch:kick` and the other hits of `glitch_drums` | 0 | 0 | how hard |

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from . import glitch, glitch_drums
from .dsp import SR
from .glitch_drums import VELOCITY_STEP
from .timeline import SECONDS_PER_MINUTE
from .voices import ENERGY_FLOOR, KEY_DECIMALS, MIN_VELOCITY, NOTES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .quality import Quality
    from .score import Glitch, Note

PREFIX = "glitch:"
# how a chord is voiced from the root of its pad, by quality (m minor, M major): the pad, the strings and the stab
PAD_VOICING = {0: (0, 7, 12, 15, 19), 1: (0, 7, 12, 16, 19)}
STAB_VOICING = {0: (0, 7, 15, 19), 1: (0, 7, 16, 19)}
# the swells: attack in seconds, the opening of the filter in Hz, the level of the pad (RMS)
SWELLS = {
    "swell": (3.5, (420.0, 3300.0), 0.05),
    "swell_slow": (5.5, (420.0, 3300.0), 0.035),
    "swell_big": (0.7, (420.0, 4800.0), 0.10),
}
# the stabs: the peak of the filter in Hz, the release in seconds, the level (RMS)
STABS = {
    "stab": (2300.0, 0.6, 0.16),
    "stab_big": (2800.0, 1.2, 0.27),
    "stab_long": (2700.0, 1.0, 0.16),
    "stab_short": (2300.0, 0.25, 0.16),
}
# the notes that take long to make and come round again, which the cache keeps in a pool of their own
BOOM_SECONDS = 3.4
BOOM = "boom"
BOOM_FAR = "boomfar"
GROWL = "growl"
REV_STAB = "revstab"
REV_SNARE = "revsnare"
DEAR = frozenset({*STABS, GROWL, REV_STAB, REV_SNARE, BOOM, BOOM_FAR})
BOOM_FAR_EXTRA_SECONDS = 0.6
# a note may sound this long after its end
REACH_PAD_SECONDS = 1.0
# a tier with this many layers or fewer makes the cheaper sounds
LIGHT_MAX_LAYERS = 4
# the swell is stacked of this many saws (3 when light), of the notes of its chord (the first 3 when light), this many cents apart
SWELL_SAWS = 5
LIGHT_SWELL_SAWS = 3
LIGHT_SWELL_NOTES = 3
SWELL_DETUNE_CENTS = 13.0
SWELL_SEED_FACTOR = 7
# the pad of a swell dies away over this many seconds, which its loudness is not measured over
SWELL_RELEASE_SECONDS = 3.0
# the stab is measured over this stretch of its start, in seconds, at the most
STAB_MEASURE_START = 0.12
STAB_MEASURE_END = 0.9
# the fade at the end of a sound that is turned round, in seconds
REVERSE_FADE_SECONDS = 0.004
# the reversed stab: the length, the velocity and the filter peak of its stab, its release and its gain, the hall that it goes through
REVSTAB_STAB_SECONDS = 0.7
REVSTAB_STAB_VELOCITY = 0.9
REVSTAB_PEAK_HZ = 2400.0
REVSTAB_RELEASE_SECONDS = 0.4
REVSTAB_STAB_GAIN = 0.2
REVSTAB_HALL = (4.5, 3.0)
REVSTAB_GAIN = 0.9
REVSTAB_POWER = 1.8
# the reversed snare: the gain of the snare, its hall, the gain and the rise of what is turned round
REVSNARE_SNARE_GAIN = 0.8
REVSNARE_HALL = (3.0, 1.5)
REVSNARE_GAIN = 0.9
REVSNARE_POWER = 2.0
MAX_VELOCITY = 1.5
DEFAULT_TEMPO = 70.0
# a growl note carries its state and the phase of its wobble in eighths of a cycle
GROWL_PHASES = 8
# the bytes that the stacks of swells may fill
STACKS_LIMIT_BYTES = 48_000_000


def is_glitch(voice: str) -> bool:
    """Return True for a voice that is made here."""
    return voice.startswith(PREFIX)


def prepare(notes: Iterable[Note], tier: Quality) -> None:
    """
    Tell the cache of stacks which swells a piece is going to make, so that each stack is made once, for the longest.

    :param notes: The notes of the score.
    :param tier: The quality tier.
    """
    light = tier.max_layers <= LIGHT_MAX_LAYERS
    for note in notes:
        name = note.voice.removeprefix(PREFIX)
        if note.voice.startswith(PREFIX) and name in SWELLS:
            STACKS.expect(
                _stack_key(name, note.midi, note.patch, light), round(note.dur, KEY_DECIMALS)
            )


def render(
    note: Note, tier: Quality, setup: Glitch | None, cache_mb: float | None = None
) -> np.ndarray:
    """
    Return the audio of a note of the style, stereo float32, from the cache when it was made.

    :param note: A note whose voice starts with `glitch:`.
    :param tier: The quality tier; the cheapest makes cheaper sounds.
    :param setup: The tempo and the vowels of the day, which the growl needs.
    :param cache_mb: The size of the cache that may be used (the tier's by default).
    """
    name = note.voice.removeprefix(PREFIX)
    limit = float(tier.cache_mb if cache_mb is None else cache_mb)
    light = tier.max_layers <= LIGHT_MAX_LAYERS
    if name in glitch_drums.HIT_NAMES:
        return glitch_drums.hit(name, note.vel, limit)
    vel = (
        round(float(np.clip(note.vel, MIN_VELOCITY, MAX_VELOCITY)) / VELOCITY_STEP) * VELOCITY_STEP
    )
    dur = round(note.dur, KEY_DECIMALS)
    tempo = setup.tempo if setup else DEFAULT_TEMPO
    vowels = setup.vowels if setup else 0
    key = (
        PREFIX,
        name,
        round(note.midi, KEY_DECIMALS),
        note.patch,
        dur,
        round(vel, KEY_DECIMALS),
        light,
        tempo,
        vowels,
    )
    found = NOTES.get(key)
    if found is not None:
        return found if found.dtype == np.float32 else found.astype(np.float32)
    if name in SWELLS:
        wave = _swell(name, note.midi, note.patch, dur, vel, light)
        stored = wave.astype(np.float16)
        NOTES.put(key, stored, limit, dear=True)
        # the first time is what every later time is, bit for bit
        return stored.astype(np.float32)
    if name in STABS:
        wave = _stab(name, note.midi, note.patch, dur, vel, light)
    elif name == BOOM:
        wave = glitch.sub_boom(note.midi, BOOM_SECONDS, vel)
    elif name == BOOM_FAR:
        wave = glitch.sub_boom(note.midi, BOOM_SECONDS + BOOM_FAR_EXTRA_SECONDS, vel, drum=False)
    elif name == REV_STAB:
        wave = _revstab(note.midi, note.patch, dur, vel)
    elif name == REV_SNARE:
        wave = _revsnare(dur, vel, limit)
    elif name == GROWL:
        state, phase = divmod(note.patch, GROWL_PHASES)
        wave = glitch.growl_note(
            note.midi, dur, state, phase / GROWL_PHASES, SECONDS_PER_MINUTE / tempo, vowels, light
        ) * np.float32(vel)
    else:
        message = f"no such voice: {note.voice}"
        raise KeyError(message)
    stored = wave.astype(np.float16)
    NOTES.put(key, stored, limit, dear=name in DEAR)
    # the first time is what every later time is, bit for bit
    return stored.astype(np.float32)


def reach(note: Note) -> float:
    """Return how many seconds after its start a note of the style may still sound."""
    name = note.voice.removeprefix(PREFIX)
    if name in {BOOM, BOOM_FAR}:
        return BOOM_SECONDS + REACH_PAD_SECONDS
    return note.dur + REACH_PAD_SECONDS


def _rms(x: np.ndarray, lo: float, hi: float) -> float:
    seg = x[int(lo * SR) : int(hi * SR)]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + ENERGY_FLOOR


class _Stacks:
    """The saw stacks of swells, a few of them kept: swells of one chord share a stack whatever their level and length."""

    LIMIT_BYTES = STACKS_LIMIT_BYTES

    def __init__(self) -> None:
        """Make an empty set."""
        self._items: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._longest: dict[tuple[object, ...], float] = {}
        self._lock = threading.Lock()

    def expect(self, key: tuple[object, ...], dur: float) -> None:
        """Say that a swell of this stack is going to be made, `dur` seconds long (a stack is made for the longest)."""
        with self._lock:
            self._longest[key] = max(self._longest.get(key, 0.0), dur)

    def get(self, key: tuple[object, ...], n: int, make: Callable[[int], np.ndarray]) -> np.ndarray:
        """Return a stack of at least `n` samples, from the set when it is there and long enough."""
        with self._lock:
            found = self._items.get(key)
            if found is not None and len(found) >= n:
                self._items.move_to_end(key)
                return found
            longest = int(self._longest.get(key, 0.0) * SR)
        stack = make(max(n, longest))
        with self._lock:
            self._items[key] = stack
            self._items.move_to_end(key)
            while (
                sum(v.nbytes for v in self._items.values()) > self.LIMIT_BYTES
                and len(self._items) > 1
            ):
                self._items.popitem(last=False)
        return stack

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._items.clear()
            self._longest.clear()


STACKS = _Stacks()


def _stack_key(kind: str, midi: float, quality: int, light: bool) -> tuple[object, ...]:  # noqa: FBT001
    attack, opening, _ = SWELLS[kind]
    return (attack, opening, round(midi, KEY_DECIMALS), quality, light)


def _swell(kind: str, midi: float, quality: int, dur: float, vel: float, light: bool) -> np.ndarray:  # noqa: FBT001
    attack, opening, level = SWELLS[kind]
    offsets = PAD_VOICING[quality]
    notes = tuple(midi + o for o in (offsets[:LIGHT_SWELL_NOTES] if light else offsets))
    voices = LIGHT_SWELL_SAWS if light else SWELL_SAWS
    seed = int(midi * SWELL_SEED_FACTOR) + quality
    stack = STACKS.get(
        _stack_key(kind, midi, quality, light),
        int(dur * SR),
        lambda n: glitch.swell_stack(notes, n, attack, opening, voices, SWELL_DETUNE_CENTS, seed),
    )
    wave = glitch.swell_pad(
        list(notes),
        dur,
        attack,
        SWELL_RELEASE_SECONDS,
        opening,
        voices=voices,
        seed=seed,
        stack=stack,
    )
    wave *= np.float32(level * vel / _rms(wave, attack, dur - SWELL_RELEASE_SECONDS))
    return wave


def _stab(kind: str, midi: float, quality: int, dur: float, vel: float, light: bool) -> np.ndarray:  # noqa: FBT001
    peak, release, level = STABS[kind]
    notes = [midi + o for o in STAB_VOICING[quality]]
    wave = glitch.brass_stab(notes, dur, min(vel, 1.0), peak, release, light=light)
    return wave * np.float32(
        level * vel / _rms(wave, STAB_MEASURE_START, min(dur, STAB_MEASURE_END))
    )


def _reverse(src: np.ndarray, gain: float, power: float) -> np.ndarray:
    """Return a sound turned round, rising as it goes, with a tiny fade at the end."""
    r = src[::-1] * np.float32(gain)
    n = len(r)
    env = (np.arange(n) / n) ** power
    f = int(REVERSE_FADE_SECONDS * SR)
    env[-f:] *= glitch.smooth_ramp(f)[::-1]
    return (r * env[:, None]).astype(np.float32)


def _revstab(midi: float, quality: int, dur: float, vel: float) -> np.ndarray:
    notes = [midi + o for o in STAB_VOICING[quality]]
    stab = glitch.brass_stab(
        notes, REVSTAB_STAB_SECONDS, REVSTAB_STAB_VELOCITY, REVSTAB_PEAK_HZ, REVSTAB_RELEASE_SECONDS
    ) * np.float32(REVSTAB_STAB_GAIN)
    src = glitch.hall(stab, *REVSTAB_HALL)[: int(dur * SR)]
    return _reverse(src, REVSTAB_GAIN * vel, REVSTAB_POWER)


def _revsnare(dur: float, vel: float, cache_mb: float) -> np.ndarray:
    snare = glitch_drums.hit(glitch_drums.SNARE, vel, cache_mb) * np.float32(REVSNARE_SNARE_GAIN)
    src = glitch.hall(snare, *REVSNARE_HALL)[: int(dur * SR)]
    return _reverse(src, REVSNARE_GAIN, REVSNARE_POWER)
