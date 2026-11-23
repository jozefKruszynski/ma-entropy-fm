"""
The voice library of Entropy FM: every instrument that the soundscapes draw from, made in numpy.

A voice turns one note (a pitch, a length and a velocity) into stereo audio, with its release.
Each voice has a character of its own, its own gain and its own brightness (no blanket filter and no
per-note normalising): synthesised basses (a warm round sub, a Moog-style ladder bass, a growling
Reese, an FM growl, an acid bass), strings and pads (a string machine, a Juno-style pad, a supersaw,
tape strings, a tape choir, a glassy wavetable pad, a bowed pad, a drawbar organ), leads and keys (an
analog lead, a flute lead, an FM electric piano, three sampled pianos, a Juno pluck, Karplus-Strong
nylon, a sampled strumstick). The sampled ones read the CC0 samples in `samples/`.

Everything is band-limited (see `dsp`), a vibrato never exceeds 15 cents, and nothing glides: a pitch
is constant for the length of a note. Notes are made from absolute time and cached, so the same
note is the same samples every time.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import LowpassFilter

from . import dsp
from .dsp import (
    CENTS_PER_OCTAVE,
    LADDER_HP24,
    SAW,
    SQUARE,
    SR,
    TRIANGLE,
    ensemble,
    envelope,
    harmonic_table,
    ladder,
    midi_hz,
    osc,
    sine,
)
from .quality import STANDARD, Quality
from .sampler import BYTES_PER_MB, SAMPLES_DIR, Instrument, read_cubic
from .timeline import SEED_MODULUS, stable_hash

if TYPE_CHECKING:
    from collections.abc import Callable

# the loudness that every voice is brought to by its own constant gain: RMS of a reference note
REFERENCE_RMS = 0.09
# the longest note that is made in one piece, in seconds
MAX_NOTE_SECONDS = 26.0
MAX_VIBRATO_CENTS = 15.0
# the velocity of the reference note, on the grid of velocities that notes are made at (eighths)
REF_VELOCITY = 0.75
# a note is quieter as the velocity falls, by this power of the ratio to the reference velocity
VELOCITY_POWER = 0.7
FAMILY_BASS = "bass"
FAMILY_PAD = "pad"
FAMILY_LEAD = "lead"
FAMILY_KEY = "key"
FAMILY_PLUCK = "pluck"
# the quietest velocity that a note is made at
MIN_VELOCITY = 0.05
# the notes are cached at these decimals of pitch, length and velocity, and seeded by them
KEY_DECIMALS = 2
VIBRATO_CENTS_DECIMALS = 1
# a voice is made of at least this many samples
MIN_NOTE_SAMPLES = 64
SEED_TAG_VIBRATO = "vibrato"
CACHE_TAG_CORE = "core"
# the cache key of a tier that has the full unison, and the unison below which a tier keeps its own key
FULL_TIER_KEY = "full"
FULL_UNISON = 1
# a tier with at least this unison makes the rich version of a voice, one below it the cheap one
RICH_UNISON = 0.5
# the two channels of a stereo signal
CHANNELS = (0, 1)
# the pitch that a table is chosen for is this much above the note, so that a vibrato never outgrows it
TABLE_HEADROOM = 1.01
# the velocities that notes are made at: this many steps from 0 to 1
VELOCITY_STEPS = 8
# the length of the note that the gain of a voice is measured on, in seconds
REFERENCE_NOTE_SECONDS = 2.5
# the lengths up to which a note is rounded to a step (seconds), and the step of the longest notes
QUANTISE_STEPS = ((0.6, 0.02), (1.5, 0.1), (3.0, 0.25), (8.0, 0.5))
QUANTISE_LONGEST_STEP = 1.0
# the rate (Hz), the onset (s) and the wait (s) of a vibrato are drawn from these ranges
VIBRATO_RATE_RANGE = (4.5, 5.5)
VIBRATO_ONSET_RANGE = (1.0, 2.0)
VIBRATO_WAIT_RANGE = (0.2, 0.5)
POOL_SHORT = "short"
POOL_LONG = "long"
POOL_DEAR = "dear"
# the search for the loop point is this far each way from its target, in seconds
CORE_ALIGN_SEARCH = 0.35
# a floor under an energy or an RMS that is divided by, far under anything that is heard
ENERGY_FLOOR = 1e-12
# the delays of the cheapest tier's widening, in samples, and the share of the delayed copy
WIDE_DELAY_LEFT = 331
WIDE_DELAY_RIGHT = 577
WIDE_DIRECT_GAIN = 0.85
WIDE_DELAYED_GAIN = 0.3
PULSE_GAIN = 0.8
# the tape's delay in samples, and the swing of its wow and flutter, in samples; rates in Hz
TAPE_DELAY = 60.0
TAPE_WOW_DEPTH = 38.0
TAPE_WOW_RATE = 0.63
TAPE_FLUTTER_DEPTH = 1.4
TAPE_FLUTTER_RATE = 6.1
TAPE_PAIR_DELAY = 80.0
TAPE_PAIR_WOW_DEPTH = 36.0
TAPE_PAIR_WOW_RATE = 0.55
TAPE_PAIR_FLUTTER_DEPTH = 1.2
TAPE_PAIR_FLUTTER_RATE = 6.3
SAMPLES_STRINGS = "strings"
GROUP_CELLOS = "cellos"
GROUP_VIOLINS = "violins"
# strings below this note come from the cellos, the rest from the violins
CELLO_SPLIT_MIDI = 53
TAPE_STRINGS_LOWPASS_HZ = 5200.0
# a sampled voice is panned by its pitch: away from this note, a span of semitones to the full pan, which stops at this far
SAMPLED_PAN_CENTRE_MIDI = 60.0
MAX_SAMPLED_PAN = 0.4
SAMPLED_PAN_SPAN = 60.0
# notes held longer than this are made from a core that is looped: a pad does not change in its sustain
TILE_FROM = 8.0
CORE_HOLD = 4.8
CORE_LOOP_START = 2.4
CORE_CROSS = 0.5
TILED_FAMILIES = frozenset({FAMILY_PAD, FAMILY_BASS})
_SAMPLERS: dict[str, Instrument] = {}
_GAINS: dict[tuple[str, str], float] = {}
_JUMPS: dict[tuple[object, ...], int] = {}
_SAMPLERS_LOCK = threading.Lock()
_GAINS_LOCK = threading.Lock()
_JUMPS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Ctx:
    """What a voice is asked to play."""

    midi: float
    freq: float
    dur: float
    vel: float
    tier: Quality
    rng: np.random.Generator
    patch: int
    n: int


@dataclass(frozen=True)
class Voice:
    """A voice of the library."""

    key: str
    title: str
    # bass, pad, lead, key or pluck
    family: str
    # the pitch the loudness is measured at, and how long the reference note is held
    ref_midi: float
    release: float
    make: Callable[[Ctx], np.ndarray]
    # the brightness of the voice in the mix, a hint for the styles: 0 dark to 1 bright
    level: float = 1.0
    patches: int = 2
    # the pitch that the partials are multiples of, as a share of the note (a sub oscillator makes it 0.5)
    fundamental: float = 1.0
    # whether the voice is made of samples, whose velocity layers and curve are those of the sampler
    sampled: bool = False
    # how many levels of velocity are told apart: a voice that is made note by note through a filter keeps few
    velocity_steps: int = VELOCITY_STEPS


def sampler(name: str) -> Instrument:
    """Return a sampled instrument of `samples/`, loaded once."""
    with _SAMPLERS_LOCK:
        if name not in _SAMPLERS:
            _SAMPLERS[name] = Instrument.load(SAMPLES_DIR / name)
        return _SAMPLERS[name]


class NoteCache:
    """
    Notes that were made, kept up to a size, the least recently used dropped first.

    Long notes have their own pool, and so do the dear ones (swells and growls that take a tenth of a second or more
    to make and come round again): a dear note is never pushed out by cheap long notes, and the pool has a share of
    its own on top of the limit.
    """

    LONG_SAMPLES = 2 * SR
    DEAR_SHARE = 0.85
    LONG_SHARE = 0.6
    SHORT_SHARE = 0.4

    def __init__(self) -> None:
        """Make an empty cache."""
        self._short: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._long: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._dear: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._bytes = dict.fromkeys((POOL_SHORT, POOL_LONG, POOL_DEAR), 0)
        self._lock = threading.Lock()

    def get(self, key: tuple[object, ...]) -> np.ndarray | None:
        """Return a note and mark it as used."""
        with self._lock:
            for pool in (self._short, self._long, self._dear):
                item = pool.get(key)
                if item is not None:
                    pool.move_to_end(key)
                    return item
        return None

    def put(
        self, key: tuple[object, ...], wave: np.ndarray, limit_mb: float, *, dear: bool = False
    ) -> None:
        """
        Keep a note; each pool may use its share of the limit, and drops its oldest notes over it.

        :param key: The key of the note.
        :param wave: The audio.
        :param limit_mb: The size of the cache that may be used.
        :param dear: The note is costly to make and comes round again: keep it in the pool of its own.
        """
        name, pool = self._pool(wave, dear=dear)
        share = {
            POOL_LONG: self.LONG_SHARE,
            POOL_SHORT: self.SHORT_SHARE,
            POOL_DEAR: self.DEAR_SHARE,
        }[name]
        with self._lock:
            if key in pool:
                self._bytes[name] -= pool[key].nbytes
            pool[key] = wave
            self._bytes[name] += wave.nbytes
            while self._bytes[name] > limit_mb * share * BYTES_PER_MB and len(pool) > 1:
                _, old = pool.popitem(last=False)
                self._bytes[name] -= old.nbytes

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._short.clear()
            self._long.clear()
            self._dear.clear()
            self._bytes = dict.fromkeys((POOL_SHORT, POOL_LONG, POOL_DEAR), 0)

    @property
    def size(self) -> int:
        """Return how many notes are held."""
        with self._lock:
            return len(self._short) + len(self._long) + len(self._dear)

    def _pool(
        self, wave: np.ndarray, *, dear: bool
    ) -> tuple[str, OrderedDict[tuple[object, ...], np.ndarray]]:
        if dear:
            return POOL_DEAR, self._dear
        if len(wave) > self.LONG_SAMPLES:
            return POOL_LONG, self._long
        return POOL_SHORT, self._short


def quantised(duration: float) -> float:
    """Round a length to a step that grows with the length, so that more notes can be shared."""
    step = next((s for limit, s in QUANTISE_STEPS if duration < limit), QUANTISE_LONGEST_STEP)
    return min(max(round(duration / step) * step, step), MAX_NOTE_SECONDS)


def gain_of(key: str, tier: Quality) -> float:
    """Return the constant gain of a voice, from the loudness of a reference note (RMS 0.09)."""
    cache_key = (key, tier.key)
    gain = _GAINS.get(cache_key)
    if gain is None:
        voice = VOICES[key]
        ref = _make(key, voice.ref_midi, REFERENCE_NOTE_SECONDS, REF_VELOCITY, tier, 0)
        held = ref[: int(REFERENCE_NOTE_SECONDS * SR)]
        gain = REFERENCE_RMS / max(dsp.rms(held), dsp.MIN_LEVEL)
        with _GAINS_LOCK:
            gain = _GAINS.setdefault(cache_key, gain)
    return gain


def render_note(
    key: str,
    midi: float,
    dur: float,
    vel: float = 0.7,
    tier: Quality = STANDARD,
    patch: int = 0,
    vibrato: float = 0.0,
) -> np.ndarray:
    """
    Return one note of a voice as stereo float32 with its release, from the cache when it was made.

    :param key: A key of `VOICES`.
    :param midi: The pitch.
    :param dur: How long the key is held, in seconds (rounded for the cache).
    :param vel: 0 to 1.
    :param tier: The quality tier; it lowers the unison of the expensive voices.
    :param patch: A variant of the voice (0 or 1), chosen by the day.
    :param vibrato: The depth in cents (a peak value, at most `MAX_VIBRATO_CENTS`) of a vibrato that begins after the
        first moments of the note and fades in; its rate (4.5 to 5.5 Hz) and phase are the note's own. 0 for none.
    """
    dur = quantised(dur)
    steps = VOICES[key].velocity_steps
    vel = max(round(float(np.clip(vel, MIN_VELOCITY, 1.0)) * steps) / steps, 1.0 / steps / 2.0)
    if vibrato <= 0.0:
        return _plain_note(key, midi, dur, vel, tier, patch)
    cents = round(min(vibrato, MAX_VIBRATO_CENTS), VIBRATO_CENTS_DECIMALS)
    cache_key = (key, round(midi, KEY_DECIMALS), dur, vel, _tier_key(tier), patch, cents)
    found = NOTES.get(cache_key)
    if found is not None:
        return found
    # every note has a rate, a phase and a delay of its own, so that voices sung together do not wobble together
    rng = np.random.default_rng(
        stable_hash(SEED_TAG_VIBRATO, key, round(midi, KEY_DECIMALS), dur, vel, patch)
        % SEED_MODULUS
    )
    wave = dsp.pitch_vibrato(
        _plain_note(key, midi, dur, vel, tier, patch),
        cents,
        float(rng.uniform(*VIBRATO_RATE_RANGE)),
        float(rng.uniform(*VIBRATO_ONSET_RANGE)),
        float(rng.uniform()),
        float(rng.uniform(*VIBRATO_WAIT_RANGE)),
    )
    if dur < TILE_FROM or VOICES[key].family not in TILED_FAMILIES:
        NOTES.put(cache_key, wave, tier.cache_mb)
    return wave


def _tier_key(tier: Quality) -> str:
    return tier.key if tier.unison < FULL_UNISON else FULL_TIER_KEY


def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / SR


def _mono(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    return dsp.to_stereo(x, pan)


# ---------------------------------------------------------------- basses


# a second harmonic fades in under the sine: its gain, its phase in cycles and its decay time
SUB_ROUND_OCTAVE_GAIN = 0.32
SUB_ROUND_OCTAVE_PHASE = 0.25
SUB_ROUND_OCTAVE_DECAY_SECONDS = 0.35
# the second patch is a triangle with a fainter octave
SUB_ROUND_TRIANGLE_GAIN = 0.9
SUB_ROUND_TRIANGLE_OCTAVE_GAIN = 0.2
SUB_ROUND_TRIANGLE_OCTAVE_DECAY_SECONDS = 0.2
SUB_ROUND_DRIVE = 1.3
SUB_ROUND_ENVELOPE_ATTACK = 0.012
SUB_ROUND_ENVELOPE_DECAY = 0.5
SUB_ROUND_ENVELOPE_SUSTAIN = 0.85
SUB_ROUND_ENVELOPE_RELEASE = 0.14


def _sub_round(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    body = sine(c.freq, c.n) + SUB_ROUND_OCTAVE_GAIN * sine(
        2 * c.freq, c.n, SUB_ROUND_OCTAVE_PHASE
    ) * np.exp(-t / SUB_ROUND_OCTAVE_DECAY_SECONDS)
    if c.patch == 1:
        body = osc(
            TRIANGLE, c.freq, c.n
        ) * SUB_ROUND_TRIANGLE_GAIN + SUB_ROUND_TRIANGLE_OCTAVE_GAIN * sine(
            2 * c.freq, c.n
        ) * np.exp(-t / SUB_ROUND_TRIANGLE_OCTAVE_DECAY_SECONDS)
    body = np.tanh(SUB_ROUND_DRIVE * body) / SUB_ROUND_DRIVE
    return _mono(
        body
        * envelope(
            c.n,
            SUB_ROUND_ENVELOPE_ATTACK,
            SUB_ROUND_ENVELOPE_DECAY,
            SUB_ROUND_ENVELOPE_SUSTAIN,
            c.dur,
            SUB_ROUND_ENVELOPE_RELEASE,
        )
    )


MOOG_BASS_SAW_GAIN = 0.55
MOOG_BASS_SQUARE_GAIN = 0.4
MOOG_BASS_SQUARE_DETUNE = 1.0023
MOOG_BASS_SQUARE_PHASE = 0.3
MOOG_BASS_SUB_GAIN = 0.5
# the cutoff peaks at this, plus this per unit of velocity, plus this on the second patch (Hz)
MOOG_BASS_PEAK_HZ = 700.0
MOOG_BASS_PEAK_PER_VELOCITY_HZ = 1500.0
MOOG_BASS_PEAK_PATCH_HZ = 400.0
# the cutoff rests at the higher of this and a multiple of the pitch, and its peak is at least a multiple of that
MOOG_BASS_MIN_CUTOFF_HZ = 150.0
MOOG_BASS_CUTOFF_PITCH_RATIO = 2.4
MOOG_BASS_PEAK_MIN_RATIO = 3.0
MOOG_BASS_SWEEP_SECONDS = 0.20
MOOG_BASS_LADDER_RESONANCE = 0.42
MOOG_BASS_LADDER_DRIVE = 1.5
MOOG_BASS_CHUNK = 64
MOOG_BASS_ENVELOPE_ATTACK = 0.004
MOOG_BASS_ENVELOPE_DECAY = 0.4
MOOG_BASS_ENVELOPE_SUSTAIN = 0.65
MOOG_BASS_ENVELOPE_RELEASE = 0.12


def _moog_bass(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    wave = (
        MOOG_BASS_SAW_GAIN * osc(SAW, f, c.n)
        + MOOG_BASS_SQUARE_GAIN
        * osc(SQUARE, f * MOOG_BASS_SQUARE_DETUNE, c.n, MOOG_BASS_SQUARE_PHASE)
        + MOOG_BASS_SUB_GAIN * sine(f, c.n)
    )
    top = (
        MOOG_BASS_PEAK_HZ
        + MOOG_BASS_PEAK_PER_VELOCITY_HZ * c.vel
        + (MOOG_BASS_PEAK_PATCH_HZ if c.patch else 0.0)
    )
    low = max(MOOG_BASS_MIN_CUTOFF_HZ, MOOG_BASS_CUTOFF_PITCH_RATIO * f)
    cutoff = low + (max(top, low * MOOG_BASS_PEAK_MIN_RATIO) - low) * np.exp(
        -t / MOOG_BASS_SWEEP_SECONDS
    )
    wave = ladder(
        wave, cutoff, MOOG_BASS_LADDER_RESONANCE, MOOG_BASS_LADDER_DRIVE, chunk=MOOG_BASS_CHUNK
    )
    return _mono(
        wave
        * envelope(
            c.n,
            MOOG_BASS_ENVELOPE_ATTACK,
            MOOG_BASS_ENVELOPE_DECAY,
            MOOG_BASS_ENVELOPE_SUSTAIN,
            c.dur,
            MOOG_BASS_ENVELOPE_RELEASE,
        )
    )


REESE_CENTS_FULL = (-10.0, 0.0, 12.0)
REESE_CENTS_LIGHT = (-9.0, 10.0)
# the second patch narrows the detune to this share
REESE_PATCH_DETUNE_SHARE = 0.55
REESE_PARTIAL_LIMIT_HZ = 4500.0
REESE_DRIVE_FLOOR = 1.8
REESE_DRIVE_PER_VELOCITY = 1.4
# the rate of the filter's LFO in Hz: this much plus up to this much more, drawn per note
REESE_LFO_HZ = 0.10
REESE_LFO_SPREAD_HZ = 0.22
REESE_CUTOFF_HZ = 170.0
REESE_CUTOFF_SWING_HZ = 760.0
REESE_VELOCITY_FLOOR = 0.65
REESE_VELOCITY_RANGE = 0.7
REESE_LADDER_RESONANCE = 0.52
REESE_SUB_GAIN = 0.7
REESE_SAW_GAIN = 0.85
REESE_ENVELOPE_ATTACK = 0.02
REESE_ENVELOPE_DECAY = 0.4
REESE_ENVELOPE_SUSTAIN = 0.9
REESE_ENVELOPE_RELEASE = 0.25


def _reese(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    cents: tuple[float, ...] = (
        REESE_CENTS_FULL if c.tier.unison >= RICH_UNISON else REESE_CENTS_LIGHT
    )
    if c.patch == 1:
        cents = tuple(c * REESE_PATCH_DETUNE_SHARE for c in cents)
    saws = sum(
        osc(
            SAW,
            f * 2.0 ** (cd / CENTS_PER_OCTAVE),
            c.n,
            float(c.rng.uniform()),
            limit=REESE_PARTIAL_LIMIT_HZ,
        )
        for cd in cents
    ) / math.sqrt(len(cents))
    # a little waveshaping BEFORE the filter, on band-limited saws, so that the harmonics it makes stay harmonic
    drive = REESE_DRIVE_FLOOR + REESE_DRIVE_PER_VELOCITY * c.vel
    shaped = np.tanh(drive * saws) / math.tanh(drive)
    lfo = REESE_LFO_HZ + REESE_LFO_SPREAD_HZ * float(c.rng.uniform())
    cutoff = (
        REESE_CUTOFF_HZ
        + REESE_CUTOFF_SWING_HZ * (0.5 + 0.5 * np.sin(2 * np.pi * (lfo * t + c.rng.uniform())))
    ) * (REESE_VELOCITY_FLOOR + REESE_VELOCITY_RANGE * c.vel)
    filt = ladder(shaped, cutoff, REESE_LADDER_RESONANCE, 1.0, chunk=c.tier.control_chunk // 2)
    sub = sine(f, c.n) * REESE_SUB_GAIN
    wave = (REESE_SAW_GAIN * filt + sub) * envelope(
        c.n,
        REESE_ENVELOPE_ATTACK,
        REESE_ENVELOPE_DECAY,
        REESE_ENVELOPE_SUSTAIN,
        c.dur,
        REESE_ENVELOPE_RELEASE,
    )
    return _mono(wave)


FM_GROWL_LFO_HZ = 0.18
FM_GROWL_LFO_SPREAD_HZ = 0.3
# the index of the modulation: its floor and its swing with the LFO, then the share of both that velocity keeps
FM_GROWL_INDEX_FLOOR = 0.7
FM_GROWL_INDEX_SWING = 3.6
FM_GROWL_VELOCITY_FLOOR = 0.6
FM_GROWL_VELOCITY_RANGE = 0.6
FM_GROWL_PAIR_GAIN = 0.8
FM_GROWL_SECOND_PAIR_GAIN = 0.35
FM_GROWL_SECOND_PAIR_INDEX_SHARE = 0.6
FM_GROWL_SECOND_PAIR_RATIO = 3
FM_GROWL_LOWPASS_HZ = 2300.0
FM_GROWL_LADDER_RESONANCE = 0.2
FM_GROWL_WAVE_GAIN = 0.8
FM_GROWL_SUB_GAIN = 0.65
FM_GROWL_ENVELOPE_ATTACK = 0.012
FM_GROWL_ENVELOPE_DECAY = 0.4
FM_GROWL_ENVELOPE_SUSTAIN = 0.9
FM_GROWL_ENVELOPE_RELEASE = 0.2


def _fm_growl(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    ratio = 2.0 if c.patch == 0 else 1.0
    lfo = FM_GROWL_LFO_HZ + FM_GROWL_LFO_SPREAD_HZ * float(c.rng.uniform())
    index = (
        FM_GROWL_INDEX_FLOOR
        + FM_GROWL_INDEX_SWING * (0.5 + 0.5 * np.sin(2 * np.pi * (lfo * t + c.rng.uniform())))
    ) * (FM_GROWL_VELOCITY_FLOOR + FM_GROWL_VELOCITY_RANGE * c.vel)
    phase = 2 * np.pi * f * t
    wave = np.sin(phase + index * np.sin(ratio * phase))
    if c.tier.rich_bass:
        wave = FM_GROWL_PAIR_GAIN * wave + FM_GROWL_SECOND_PAIR_GAIN * np.sin(
            2 * phase
            + FM_GROWL_SECOND_PAIR_INDEX_SHARE * index * np.sin(FM_GROWL_SECOND_PAIR_RATIO * phase)
        )
    wave = ladder(wave.astype(np.float32), FM_GROWL_LOWPASS_HZ, FM_GROWL_LADDER_RESONANCE, 1.0)
    out = (FM_GROWL_WAVE_GAIN * wave + FM_GROWL_SUB_GAIN * sine(f, c.n)) * envelope(
        c.n,
        FM_GROWL_ENVELOPE_ATTACK,
        FM_GROWL_ENVELOPE_DECAY,
        FM_GROWL_ENVELOPE_SUSTAIN,
        c.dur,
        FM_GROWL_ENVELOPE_RELEASE,
    )
    return _mono(out)


ACID_BASS_PEAK_HZ = 1200.0
ACID_BASS_PEAK_PER_VELOCITY_HZ = 4800.0
ACID_BASS_MIN_CUTOFF_HZ = 260.0
ACID_BASS_CUTOFF_PITCH_RATIO = 3.0
ACID_BASS_PEAK_MIN_RATIO = 3.0
ACID_BASS_SWEEP_SECONDS = 0.11
ACID_BASS_LADDER_RESONANCE = 0.84
ACID_BASS_LADDER_DRIVE = 2.4
ACID_BASS_CHUNK = 32
ACID_BASS_OUTPUT_DRIVE = 1.8
ACID_BASS_ENVELOPE_ATTACK = 0.003
ACID_BASS_ENVELOPE_DECAY = 0.2
ACID_BASS_ENVELOPE_SUSTAIN = 0.55
ACID_BASS_ENVELOPE_RELEASE = 0.06


def _acid_bass(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    wave = osc(SAW if c.patch == 0 else SQUARE, c.freq, c.n)
    top = ACID_BASS_PEAK_HZ + ACID_BASS_PEAK_PER_VELOCITY_HZ * c.vel
    low = max(ACID_BASS_MIN_CUTOFF_HZ, ACID_BASS_CUTOFF_PITCH_RATIO * c.freq)
    cutoff = low + (max(top, low * ACID_BASS_PEAK_MIN_RATIO) - low) * np.exp(
        -t / ACID_BASS_SWEEP_SECONDS
    )
    wave = ladder(
        wave, cutoff, ACID_BASS_LADDER_RESONANCE, ACID_BASS_LADDER_DRIVE, chunk=ACID_BASS_CHUNK
    )
    wave = np.tanh(ACID_BASS_OUTPUT_DRIVE * wave) / ACID_BASS_OUTPUT_DRIVE
    return _mono(
        wave
        * envelope(
            c.n,
            ACID_BASS_ENVELOPE_ATTACK,
            ACID_BASS_ENVELOPE_DECAY,
            ACID_BASS_ENVELOPE_SUSTAIN,
            c.dur,
            ACID_BASS_ENVELOPE_RELEASE,
        )
    )


# ---------------------------------------------------------------- strings and pads


# the two saws are detuned up and down by these ratios; the one below starts at a phase of its own (cycles)
STRING_MACHINE_DETUNE_UP = 1.0021
STRING_MACHINE_PARTIAL_LIMIT_HZ = 7000.0
STRING_MACHINE_DETUNE_DOWN = 0.9979
STRING_MACHINE_DOWN_PHASE = 0.4
STRING_MACHINE_OCTAVE_GAIN = 0.55
STRING_MACHINE_OCTAVE_PHASE = 0.2
STRING_MACHINE_ENVELOPE_ATTACK = 0.7
STRING_MACHINE_ENVELOPE_RELEASE = 1.5
STRING_MACHINE_WIDE_RATES = (0.55, 0.7, 5.6)
STRING_MACHINE_WIDE_DEPTH_MS = 2.0
STRING_MACHINE_WIDE_TAPS = 3
STRING_MACHINE_WIDE_MIX = 0.5
STRING_MACHINE_SAW_GAIN = 0.5


def _string_machine(c: Ctx) -> np.ndarray:
    f = c.freq
    base = osc(SAW, f * STRING_MACHINE_DETUNE_UP, c.n, limit=STRING_MACHINE_PARTIAL_LIMIT_HZ) + osc(
        SAW,
        f * STRING_MACHINE_DETUNE_DOWN,
        c.n,
        STRING_MACHINE_DOWN_PHASE,
        limit=STRING_MACHINE_PARTIAL_LIMIT_HZ,
    )
    top = STRING_MACHINE_OCTAVE_GAIN * osc(
        SAW, 2 * f, c.n, STRING_MACHINE_OCTAVE_PHASE, limit=STRING_MACHINE_PARTIAL_LIMIT_HZ
    )
    wave = (STRING_MACHINE_SAW_GAIN * base + top) * envelope(
        c.n, STRING_MACHINE_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, STRING_MACHINE_ENVELOPE_RELEASE
    )
    return _wide(
        wave,
        c,
        STRING_MACHINE_WIDE_RATES,
        STRING_MACHINE_WIDE_DEPTH_MS,
        STRING_MACHINE_WIDE_TAPS,
        STRING_MACHINE_WIDE_MIX,
    )


def _wide(
    wave: np.ndarray, c: Ctx, rates: tuple[float, ...], depth_ms: float, taps: int, mix: float
) -> np.ndarray:
    """Make a mono voice stereo: an ensemble where the tier has chorus, two short fixed delays on the cheapest."""
    if c.tier.chorus:
        return ensemble(wave, c.rng, rates, depth_ms, taps, mix)
    left = wave * WIDE_DIRECT_GAIN + np.roll(wave, WIDE_DELAY_LEFT) * WIDE_DELAYED_GAIN
    right = wave * WIDE_DIRECT_GAIN + np.roll(wave, WIDE_DELAY_RIGHT) * WIDE_DELAYED_GAIN
    return np.stack([left, right], axis=1).astype(np.float32)


def _pulse(f: float, width: np.ndarray, n: int, phase: float = 0.0) -> np.ndarray:
    """A band-limited pulse of a changing width: the difference of two saws."""  # noqa: D401
    tab = dsp.table(SAW, dsp.level_for(f))
    first = dsp.read_table(tab, phase + f * np.arange(n) / SR)
    second = dsp.read_table(tab, phase + f * np.arange(n) / SR + width)
    return (first - second) * PULSE_GAIN


JUNO_PAD_WIDTH = 0.5
JUNO_PAD_WIDTH_SWING = 0.16
JUNO_PAD_WIDTH_LFO_HZ = 0.33
JUNO_PAD_PARTIAL_LIMIT_HZ = 9000.0
JUNO_PAD_SUB_GAIN = 0.35
JUNO_PAD_OSC_GAIN = 0.5
JUNO_PAD_ENVELOPE_ATTACK = 0.32
JUNO_PAD_ENVELOPE_RELEASE = 1.3
JUNO_PAD_WIDE_RATES = (0.5, 0.83)
JUNO_PAD_WIDE_DEPTH_MS = 1.8
JUNO_PAD_WIDE_TAPS = 2
JUNO_PAD_WIDE_MIX = 0.5


def _juno_pad(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    width = JUNO_PAD_WIDTH + JUNO_PAD_WIDTH_SWING * np.sin(
        2 * np.pi * (JUNO_PAD_WIDTH_LFO_HZ * t + c.rng.uniform())
    )
    wave = (
        JUNO_PAD_OSC_GAIN * osc(SAW, f, c.n, limit=JUNO_PAD_PARTIAL_LIMIT_HZ)
        + JUNO_PAD_OSC_GAIN * _pulse(f, width, c.n)
        + JUNO_PAD_SUB_GAIN * sine(f / 2, c.n)
    )
    wave = wave * envelope(
        c.n, JUNO_PAD_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, JUNO_PAD_ENVELOPE_RELEASE
    )
    return _wide(
        wave, c, JUNO_PAD_WIDE_RATES, JUNO_PAD_WIDE_DEPTH_MS, JUNO_PAD_WIDE_TAPS, JUNO_PAD_WIDE_MIX
    )


# the minimum and the number of saws of the supersaw (at the full unison), the detune spread and what the second patch adds
SUPERSAW_MIN_SAWS = 3
SUPERSAW_FULL_SAWS = 7
SUPERSAW_SPREAD_CENTS = 18.0
SUPERSAW_PATCH_SPREAD_CENTS = 8.0
SUPERSAW_PARTIAL_LIMIT_HZ = 10000.0
SUPERSAW_PAN_WIDTH = 0.85
SUPERSAW_ENVELOPE_ATTACK = 0.14
SUPERSAW_HIGHPASS_HZ = 130.0
SUPERSAW_LADDER_RESONANCE = 0.1


def _supersaw(c: Ctx) -> np.ndarray:
    count = max(SUPERSAW_MIN_SAWS, round(SUPERSAW_FULL_SAWS * c.tier.unison))
    spread = SUPERSAW_SPREAD_CENTS + SUPERSAW_PATCH_SPREAD_CENTS * (c.patch % 2)
    offsets = np.linspace(-1.0, 1.0, count) * spread
    left = np.zeros(c.n, dtype=np.float32)
    right = np.zeros(c.n, dtype=np.float32)
    for index, cents in enumerate(offsets):
        wave = osc(
            SAW,
            c.freq * 2.0 ** (cents / CENTS_PER_OCTAVE),
            c.n,
            float(c.rng.uniform()),
            limit=SUPERSAW_PARTIAL_LIMIT_HZ,
        )
        pan = (index / max(count - 1, 1)) * 2.0 - 1.0
        stereo = dsp.to_stereo(wave, pan * SUPERSAW_PAN_WIDTH)
        left += stereo[:, 0]
        right += stereo[:, 1]
    scale = 1.0 / math.sqrt(count)
    env = envelope(c.n, SUPERSAW_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, 1.0)
    out = np.stack([left, right], axis=1) * scale * env[:, None]
    return np.stack(
        [
            ladder(out[:, k], SUPERSAW_HIGHPASS_HZ, SUPERSAW_LADDER_RESONANCE, mode=LADDER_HP24)
            for k in CHANNELS
        ],
        axis=1,
    )


def _tape(wave: np.ndarray, c: Ctx, depth: float = 1.0) -> np.ndarray:
    """Tape wow and flutter: a drift of a few cents, which is a wobble and never a glide."""
    if not c.tier.wobble or depth <= 0.0:
        return wave
    t = _t(len(wave))
    wow = TAPE_WOW_DEPTH * np.sin(2 * np.pi * (TAPE_WOW_RATE * t + c.rng.uniform()))
    flutter = TAPE_FLUTTER_DEPTH * np.sin(2 * np.pi * (TAPE_FLUTTER_RATE * t + c.rng.uniform()))
    return dsp.delay_read(wave, TAPE_DELAY + depth * (wow + flutter))


def _tape_pair(wave: np.ndarray, c: Ctx, depth: float) -> np.ndarray:
    """Two copies on the tape with their own wow, left and right: a width that is a shimmer, not a comb."""
    if not c.tier.wobble:
        return dsp.to_stereo(wave)
    t = _t(len(wave))
    sides = []
    for _side in range(2):
        wow = TAPE_PAIR_WOW_DEPTH * np.sin(
            2 * np.pi * (TAPE_PAIR_WOW_RATE * t + c.rng.uniform())
        ) + TAPE_PAIR_FLUTTER_DEPTH * np.sin(
            2 * np.pi * (TAPE_PAIR_FLUTTER_RATE * t + c.rng.uniform())
        )
        sides.append(dsp.delay_read(wave, TAPE_PAIR_DELAY + depth * wow))
    return np.stack(sides, axis=1)


TAPE_STRINGS_ATTACK = 0.55
TAPE_STRINGS_PATCH_ATTACK = 0.35


def _tape_strings(c: Ctx) -> np.ndarray:
    group = GROUP_CELLOS if c.midi < CELLO_SPLIT_MIDI else GROUP_VIOLINS
    wave = sampler(SAMPLES_STRINGS).render(
        c.midi,
        c.vel,
        c.dur,
        group=group,
        attack=TAPE_STRINGS_ATTACK + TAPE_STRINGS_PATCH_ATTACK * (c.patch % 2),
    )
    wave = _tape(wave, c, 1.0)
    # tape lets little above 5 kHz through: the voice's own colour, not a blanket filter
    wave = np.asarray(
        LowpassFilter(TAPE_STRINGS_LOWPASS_HZ).process(wave[None, :], SR)[0], dtype=np.float32
    )
    return _tape_pair(wave, c, 1.0)


CHANT_CELLO_ATTACK = 0.42
CHANT_CELLO_PATCH_ATTACK = 0.2
CHANT_CELLO_LOWPASS_HZ = 3000.0
CHANT_CELLO_PAN = 0.08
CHANT_CELLO_PATCH_PAN = -0.08


def _chant_cello(c: Ctx) -> np.ndarray:
    """A cello section for a slow line: the warm low sections only, a patient attack, a dark top, nearly centred."""  # noqa: D401
    wave = sampler(SAMPLES_STRINGS).render(
        c.midi,
        c.vel,
        c.dur,
        group=GROUP_CELLOS,
        attack=CHANT_CELLO_ATTACK + CHANT_CELLO_PATCH_ATTACK * (c.patch % 2),
    )
    # a steep low-pass: the section is warm, and its bow noise above 4 kHz is not wanted under a slow line
    wave = ladder(wave, CHANT_CELLO_LOWPASS_HZ, 0.0, 1.0)
    return dsp.to_stereo(wave, CHANT_CELLO_PAN if c.patch == 0 else CHANT_CELLO_PATCH_PAN)


_VOWELS = {
    0: ((800.0, 1.0, 90.0), (1150.0, 0.55, 100.0), (2900.0, 0.22, 140.0), (3900.0, 0.09, 160.0)),
    1: ((500.0, 1.0, 80.0), (800.0, 0.5, 90.0), (2830.0, 0.12, 130.0)),
}


TAPE_CHOIR_PARTIAL_LIMIT_HZ = 7000.0
# the exponent of the bell of each formant and the floor of partial amplitudes under them
TAPE_CHOIR_FORMANT_EXPONENT = -0.5
TAPE_CHOIR_FLOOR_AMPLITUDE = 0.04
TAPE_CHOIR_AMPLITUDE_DECIMALS = 3
TAPE_CHOIR_VOICES = 3
TAPE_CHOIR_VOICES_LIGHT = 2
TAPE_CHOIR_CENTS = 7.0
# the choir fades in out of the drone, and its bright formants are rolled off above this, to the fourth power
TAPE_CHOIR_ENVELOPE_ATTACK = 3.0
TAPE_CHOIR_DARK_HZ = 2200.0
TAPE_CHOIR_DARK_POWER = 4
TAPE_CHOIR_ENVELOPE_RELEASE = 1.6
TAPE_CHOIR_TAPE_DEPTH = 0.8


def _tape_choir(c: Ctx) -> np.ndarray:
    formants = _VOWELS[c.patch % 2]
    f = c.freq
    kmax = int(TAPE_CHOIR_PARTIAL_LIMIT_HZ / f)
    k = np.arange(1, kmax + 1) * f
    amps = np.zeros(kmax)
    for centre, gain, width in formants:
        amps += gain * np.exp(TAPE_CHOIR_FORMANT_EXPONENT * ((k - centre) / width) ** 2)
    amps += TAPE_CHOIR_FLOOR_AMPLITUDE / np.arange(1, kmax + 1)
    amps /= 1.0 + (k / TAPE_CHOIR_DARK_HZ) ** TAPE_CHOIR_DARK_POWER
    tab = harmonic_table(
        tuple(float(a) for a in np.round(amps / amps.max(), TAPE_CHOIR_AMPLITUDE_DECIMALS))
    )
    voices = TAPE_CHOIR_VOICES if c.tier.unison >= RICH_UNISON else TAPE_CHOIR_VOICES_LIGHT
    wave = np.zeros(c.n, dtype=np.float32)
    for index in range(voices):
        cents = (index - (voices - 1) / 2.0) * TAPE_CHOIR_CENTS
        detuned = f * 2.0 ** (cents / CENTS_PER_OCTAVE)
        wave += dsp.wavetable_osc(tab, detuned, c.n, float(c.rng.uniform()))
    wave = (
        wave
        / math.sqrt(voices)
        * envelope(c.n, TAPE_CHOIR_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, TAPE_CHOIR_ENVELOPE_RELEASE)
    )
    return _tape_pair(wave, c, TAPE_CHOIR_TAPE_DEPTH)


GLASS_PAD_PARTIAL_LIMIT_HZ = 12000.0
# the first table: odd partials fall off as this power, the even ones are this much of 1/k
GLASS_PAD_ODD_ROLLOFF = 0.9
GLASS_PAD_EVEN_AMPLITUDE = 0.12
# the second table: a power falloff with a bell at a partial (centre, what the second patch moves it by, width) and a floor
GLASS_PAD_BELL_ROLLOFF = 0.55
GLASS_PAD_BELL_EXPONENT = -0.5
GLASS_PAD_BELL_CENTRE = 6
GLASS_PAD_BELL_PATCH_SHIFT = 3
GLASS_PAD_BELL_WIDTH = 2.2
GLASS_PAD_FLOOR_AMPLITUDE = 0.05
GLASS_PAD_AMPLITUDE_DECIMALS = 4
GLASS_PAD_MORPH_LFO_HZ = 0.06
# the two sides of the pad: the channel and the detune in cents
GLASS_PAD_SIDES = ((0, -5.0), (1, 5.0))
GLASS_PAD_ENVELOPE_ATTACK = 1.1
GLASS_PAD_ENVELOPE_RELEASE = 2.0
GLASS_PAD_GAIN = 0.8


def _glass_pad(c: Ctx) -> np.ndarray:
    f = c.freq
    kmax = int(GLASS_PAD_PARTIAL_LIMIT_HZ / f)
    k = np.arange(1, kmax + 1)
    a = np.where(k % 2 == 1, 1.0 / k**GLASS_PAD_ODD_ROLLOFF, GLASS_PAD_EVEN_AMPLITUDE / k)
    b = (1.0 / k**GLASS_PAD_BELL_ROLLOFF) * np.exp(
        GLASS_PAD_BELL_EXPONENT
        * (
            (k - (GLASS_PAD_BELL_CENTRE + GLASS_PAD_BELL_PATCH_SHIFT * (c.patch % 2)))
            / GLASS_PAD_BELL_WIDTH
        )
        ** 2
    ) + GLASS_PAD_FLOOR_AMPLITUDE / k
    tab_a = harmonic_table(tuple(float(x) for x in np.round(a, GLASS_PAD_AMPLITUDE_DECIMALS)))
    tab_b = harmonic_table(
        tuple(float(x) for x in np.round(b / b.max(), GLASS_PAD_AMPLITUDE_DECIMALS))
    )
    t = _t(c.n)
    morph = 0.5 + 0.5 * np.sin(2 * np.pi * (GLASS_PAD_MORPH_LFO_HZ * t + c.rng.uniform()))
    left = np.zeros(c.n, dtype=np.float32)
    right = np.zeros(c.n, dtype=np.float32)
    for side, cents in GLASS_PAD_SIDES:
        ph = float(c.rng.uniform())
        mix = (1 - morph) * dsp.wavetable_osc(
            tab_a, f * 2.0 ** (cents / CENTS_PER_OCTAVE), c.n, ph
        ) + morph * dsp.wavetable_osc(tab_b, f * 2.0 ** (cents / CENTS_PER_OCTAVE), c.n, ph)
        (left if side == 0 else right)[:] = mix
    env = envelope(c.n, GLASS_PAD_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, GLASS_PAD_ENVELOPE_RELEASE)
    return np.stack([left, right], axis=1) * env[:, None] * GLASS_PAD_GAIN


BOWED_PAD_PARTIAL_LIMIT_HZ = 9000.0
BOWED_PAD_ROLLOFF = 1.15
BOWED_PAD_AMPLITUDE_DECIMALS = 4
BOWED_PAD_DETUNE_CENTS = (-3.0, 3.0)
BOWED_PAD_VIBRATO_RATE_HZ = 4.8
BOWED_PAD_VIBRATO_RATE_SPREAD_HZ = 0.3
BOWED_PAD_VIBRATO_CENTS = 6.0
BOWED_PAD_VIBRATO_WAIT_SECONDS = 0.8
BOWED_PAD_BAND_NOISE_LOW_HZ = 1500.0
BOWED_PAD_BAND_NOISE_HIGH_HZ = 4200.0
BOWED_PAD_BOW_LFO_HZ = 0.13
BOWED_PAD_BOW_GAIN = 0.012
BOWED_PAD_WAVE_GAIN = 0.6
BOWED_PAD_ENVELOPE_ATTACK = 0.9
BOWED_PAD_ENVELOPE_RELEASE = 1.5


def _bowed_pad(c: Ctx) -> np.ndarray:
    f = c.freq
    kmax = int(BOWED_PAD_PARTIAL_LIMIT_HZ / f)
    k = np.arange(1, kmax + 1)
    tab = harmonic_table(
        tuple(float(x) for x in np.round(1.0 / k**BOWED_PAD_ROLLOFF, BOWED_PAD_AMPLITUDE_DECIMALS))
    )
    wave = np.zeros(c.n, dtype=np.float32)
    for cents in BOWED_PAD_DETUNE_CENTS:
        phase = dsp.vibrato_phase(
            f * 2.0 ** (cents / CENTS_PER_OCTAVE),
            c.n,
            BOWED_PAD_VIBRATO_RATE_HZ + BOWED_PAD_VIBRATO_RATE_SPREAD_HZ * float(c.rng.uniform()),
            BOWED_PAD_VIBRATO_CENTS,
            BOWED_PAD_VIBRATO_WAIT_SECONDS,
            float(c.rng.uniform()),
        )
        wave += dsp.read_table(tab, phase + float(c.rng.uniform()))
    t = _t(c.n)
    bow = (
        dsp.band_noise(c.rng, c.n, BOWED_PAD_BAND_NOISE_LOW_HZ, BOWED_PAD_BAND_NOISE_HIGH_HZ)
        * (0.5 + 0.5 * np.sin(2 * np.pi * BOWED_PAD_BOW_LFO_HZ * t))
        * BOWED_PAD_BOW_GAIN
    )
    wave = (wave * BOWED_PAD_WAVE_GAIN + bow) * envelope(
        c.n, BOWED_PAD_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, BOWED_PAD_ENVELOPE_RELEASE
    )
    return dsp.to_stereo(wave, 0.0)


# the drawbars (partial: level) of each patch
DRAWBAR_ORGAN_BARS_PATCH_0 = {1: 1.0, 2: 0.75, 3: 0.4, 4: 0.34, 6: 0.14, 8: 0.1}
DRAWBAR_ORGAN_BARS_PATCH_1 = {1: 1.0, 2: 0.2, 3: 0.55, 4: 0.1, 5: 0.12}
DRAWBAR_ORGAN_PARTIAL_LIMIT_HZ = 10000.0
DRAWBAR_ORGAN_TREMOLO_DEPTH = 0.04
DRAWBAR_ORGAN_TREMOLO_HZ = 0.35
DRAWBAR_ORGAN_ENVELOPE_ATTACK = 0.5
DRAWBAR_ORGAN_ENVELOPE_RELEASE = 1.2
DRAWBAR_ORGAN_WIDE_TAPS = 2
DRAWBAR_ORGAN_WIDE_RATES = (0.4, 0.6)
DRAWBAR_ORGAN_WIDE_DEPTH_MS = 1.4
DRAWBAR_ORGAN_WIDE_MIX = 0.35


def _drawbar_organ(c: Ctx) -> np.ndarray:
    f = c.freq
    bars = DRAWBAR_ORGAN_BARS_PATCH_0 if c.patch == 0 else DRAWBAR_ORGAN_BARS_PATCH_1
    kmax = int(DRAWBAR_ORGAN_PARTIAL_LIMIT_HZ / f)
    amps = [bars.get(k, 0.0) for k in range(1, kmax + 1)]
    tab = harmonic_table(tuple(amps))
    t = _t(c.n)
    trem = 1.0 + DRAWBAR_ORGAN_TREMOLO_DEPTH * np.sin(2 * np.pi * DRAWBAR_ORGAN_TREMOLO_HZ * t)
    wave = (
        dsp.wavetable_osc(tab, f, c.n)
        * trem
        * envelope(
            c.n, DRAWBAR_ORGAN_ENVELOPE_ATTACK, 1.0, 1.0, c.dur, DRAWBAR_ORGAN_ENVELOPE_RELEASE
        )
    )
    return _wide(
        wave,
        c,
        DRAWBAR_ORGAN_WIDE_RATES,
        DRAWBAR_ORGAN_WIDE_DEPTH_MS,
        DRAWBAR_ORGAN_WIDE_TAPS,
        DRAWBAR_ORGAN_WIDE_MIX,
    )


# ---------------------------------------------------------------- leads, keys and plucks


ANALOG_LEAD_VIBRATO_RATE_HZ = 5.2
ANALOG_LEAD_VIBRATO_CENTS = 12.0
ANALOG_LEAD_VIBRATO_WAIT_SECONDS = 0.45
ANALOG_LEAD_PARTIAL_LIMIT_HZ = 10000.0
ANALOG_LEAD_SAW_GAIN = 0.6
ANALOG_LEAD_SECOND_GAIN = 0.45
ANALOG_LEAD_SECOND_DETUNE = 1.0017
ANALOG_LEAD_SECOND_PHASE = 0.2
# the cutoff starts at this and opens by this much (Hz) with this time constant, by a share of the opening that velocity keeps
ANALOG_LEAD_CUTOFF_HZ = 900.0
ANALOG_LEAD_OPEN_HZ = 2300.0
ANALOG_LEAD_OPEN_SECONDS = 0.7
ANALOG_LEAD_VELOCITY_FLOOR = 0.6
ANALOG_LEAD_VELOCITY_RANGE = 0.6
ANALOG_LEAD_LADDER_RESONANCE = 0.22
ANALOG_LEAD_CHUNK = 256
ANALOG_LEAD_ENVELOPE_ATTACK = 0.04
ANALOG_LEAD_ENVELOPE_DECAY = 0.5
ANALOG_LEAD_ENVELOPE_SUSTAIN = 0.85
ANALOG_LEAD_ENVELOPE_RELEASE = 0.7
ANALOG_LEAD_PAN = 0.1


def _analog_lead(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    phase = dsp.vibrato_phase(
        c.freq,
        c.n,
        ANALOG_LEAD_VIBRATO_RATE_HZ,
        ANALOG_LEAD_VIBRATO_CENTS,
        ANALOG_LEAD_VIBRATO_WAIT_SECONDS,
        float(c.rng.uniform()),
    )
    second = SQUARE if c.patch == 0 else TRIANGLE
    lvl = dsp.level_for(c.freq * TABLE_HEADROOM, ANALOG_LEAD_PARTIAL_LIMIT_HZ)
    wave = ANALOG_LEAD_SAW_GAIN * dsp.read_table(
        dsp.table(SAW, lvl), phase
    ) + ANALOG_LEAD_SECOND_GAIN * dsp.read_table(
        dsp.table(second, lvl),
        phase * ANALOG_LEAD_SECOND_DETUNE + ANALOG_LEAD_SECOND_PHASE,
    )
    cutoff = ANALOG_LEAD_CUTOFF_HZ + ANALOG_LEAD_OPEN_HZ * (
        1.0 - np.exp(-t / ANALOG_LEAD_OPEN_SECONDS)
    ) * (ANALOG_LEAD_VELOCITY_FLOOR + ANALOG_LEAD_VELOCITY_RANGE * c.vel)
    wave = ladder(wave, cutoff, ANALOG_LEAD_LADDER_RESONANCE, 1.0, chunk=ANALOG_LEAD_CHUNK)
    return _mono(
        wave
        * envelope(
            c.n,
            ANALOG_LEAD_ENVELOPE_ATTACK,
            ANALOG_LEAD_ENVELOPE_DECAY,
            ANALOG_LEAD_ENVELOPE_SUSTAIN,
            c.dur,
            ANALOG_LEAD_ENVELOPE_RELEASE,
        ),
        ANALOG_LEAD_PAN,
    )


FLUTE_LEAD_VIBRATO_RATE_HZ = 4.9
FLUTE_LEAD_VIBRATO_CENTS = 10.0
FLUTE_LEAD_VIBRATO_WAIT_SECONDS = 0.5
FLUTE_LEAD_TRIANGLE_GAIN = 0.8
FLUTE_LEAD_TRIANGLE_LIMIT_HZ = 6000.0
FLUTE_LEAD_SQUARE_GAIN = 0.22
FLUTE_LEAD_SQUARE_LIMIT_HZ = 3000.0
FLUTE_LEAD_BAND_NOISE_LOW_HZ = 1800.0
FLUTE_LEAD_BAND_NOISE_HIGH_HZ = 5000.0
# the breath: a puff of this level at the start that falls with this time constant to a floor, then its gain
FLUTE_LEAD_PUFF_LEVEL = 0.35
FLUTE_LEAD_PUFF_SECONDS = 0.12
FLUTE_LEAD_BREATH_FLOOR = 0.05
FLUTE_LEAD_BREATH_GAIN = 0.04
FLUTE_LEAD_ENVELOPE_ATTACK = 0.09
FLUTE_LEAD_ENVELOPE_DECAY = 0.6
FLUTE_LEAD_ENVELOPE_SUSTAIN = 0.9
FLUTE_LEAD_ENVELOPE_RELEASE = 0.35
FLUTE_LEAD_PAN = -0.1


def _flute_lead(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    phase = dsp.vibrato_phase(
        c.freq,
        c.n,
        FLUTE_LEAD_VIBRATO_RATE_HZ,
        FLUTE_LEAD_VIBRATO_CENTS,
        FLUTE_LEAD_VIBRATO_WAIT_SECONDS,
        float(c.rng.uniform()),
    )
    wave = FLUTE_LEAD_TRIANGLE_GAIN * dsp.read_table(
        dsp.table(
            TRIANGLE,
            dsp.level_for(c.freq * TABLE_HEADROOM, FLUTE_LEAD_TRIANGLE_LIMIT_HZ),
        ),
        phase,
    ) + FLUTE_LEAD_SQUARE_GAIN * dsp.read_table(
        dsp.table(SQUARE, dsp.level_for(c.freq * TABLE_HEADROOM, FLUTE_LEAD_SQUARE_LIMIT_HZ)),
        phase,
    )
    breath = (
        dsp.band_noise(c.rng, c.n, FLUTE_LEAD_BAND_NOISE_LOW_HZ, FLUTE_LEAD_BAND_NOISE_HIGH_HZ)
        * (FLUTE_LEAD_PUFF_LEVEL * np.exp(-t / FLUTE_LEAD_PUFF_SECONDS) + FLUTE_LEAD_BREATH_FLOOR)
        * FLUTE_LEAD_BREATH_GAIN
    )
    return _mono(
        (wave + breath)
        * envelope(
            c.n,
            FLUTE_LEAD_ENVELOPE_ATTACK,
            FLUTE_LEAD_ENVELOPE_DECAY,
            FLUTE_LEAD_ENVELOPE_SUSTAIN,
            c.dur,
            FLUTE_LEAD_ENVELOPE_RELEASE,
        ),
        FLUTE_LEAD_PAN,
    )


FM_EP_INDEX_FLOOR = 1.4
FM_EP_INDEX_PER_VELOCITY = 1.8
FM_EP_INDEX_DECAY_SECONDS = 0.9
FM_EP_INDEX_DECAY_PATCH_SECONDS = 0.5
FM_EP_TINE_GAIN = 0.45
FM_EP_TINE_DECAY_SECONDS = 0.045
FM_EP_TINE_INDEX = 0.9
FM_EP_TINE_RATIO = 14.0
# the ring of a note falls with a time constant of this many seconds, and this many more for a low note
FM_EP_RING_SECONDS = 1.4
FM_EP_RING_LOW_EXTRA_SECONDS = 1.2
FM_EP_RING_SHORTEST_HZ = 600.0
FM_EP_TINE_MIX = 0.6
FM_EP_ENVELOPE_ATTACK = 0.004
FM_EP_ENVELOPE_DECAY = 3.0
FM_EP_ENVELOPE_HOLD = 0.4
FM_EP_ENVELOPE_RELEASE = 0.5
FM_EP_PAN = 0.05


def _fm_ep(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    carrier = 2 * np.pi * f * t
    index = (FM_EP_INDEX_FLOOR + FM_EP_INDEX_PER_VELOCITY * c.vel) * np.exp(
        -t / (FM_EP_INDEX_DECAY_SECONDS + FM_EP_INDEX_DECAY_PATCH_SECONDS * (c.patch % 2))
    )
    body = np.sin(carrier + index * np.sin(carrier))
    tine = (
        FM_EP_TINE_GAIN
        * c.vel
        * np.exp(-t / FM_EP_TINE_DECAY_SECONDS)
        * np.sin(carrier + FM_EP_TINE_INDEX * np.sin(FM_EP_TINE_RATIO * carrier))
    )
    amp = np.exp(
        -t
        / (
            FM_EP_RING_SECONDS
            + FM_EP_RING_LOW_EXTRA_SECONDS * (1.0 - min(f / FM_EP_RING_SHORTEST_HZ, 1.0))
        )
    )
    wave = (body + tine * FM_EP_TINE_MIX) * amp
    wave = wave * envelope(
        c.n,
        FM_EP_ENVELOPE_ATTACK,
        FM_EP_ENVELOPE_DECAY,
        1.0,
        c.dur + FM_EP_ENVELOPE_HOLD,
        FM_EP_ENVELOPE_RELEASE,
    )
    return dsp.to_stereo(wave.astype(np.float32), FM_EP_PAN)


def _sampled(name: str, c: Ctx, attack: float | None = None) -> np.ndarray:
    wave = sampler(name).render(c.midi, c.vel, c.dur, attack=attack)
    pan = float(
        np.clip(
            (c.midi - SAMPLED_PAN_CENTRE_MIDI) / SAMPLED_PAN_SPAN,
            -MAX_SAMPLED_PAN,
            MAX_SAMPLED_PAN,
        )
    )
    return dsp.to_stereo(wave, pan)


def _felt_piano(c: Ctx) -> np.ndarray:
    return _sampled("felt_piano", c)


def _grand_piano(c: Ctx) -> np.ndarray:
    return _sampled("grand_piano", c)


def _tx_ep(c: Ctx) -> np.ndarray:
    return _sampled("ep", c)


def _strum_pluck(c: Ctx) -> np.ndarray:
    return _sampled("pluck", c)


def _struck(name: str, c: Ctx) -> np.ndarray:
    """Return a struck wooden sample, closed with silence to the length of the note (a hit that rings shorter than it is held)."""
    wave = _sampled(name, c)
    if len(wave) < c.n:
        wave = np.concatenate((wave, np.zeros((c.n - len(wave), 2), dtype=wave.dtype)))
    return wave


def _slit_drum(c: Ctx) -> np.ndarray:
    return _struck("slit_drum", c)


def _woodblock(c: Ctx) -> np.ndarray:
    return _struck("woodblock", c)


JUNO_PLUCK_PULSE_WIDTH = 0.42
JUNO_PLUCK_SAW_GAIN = 0.55
JUNO_PLUCK_PULSE_GAIN = 0.5
JUNO_PLUCK_PARTIAL_LIMIT_HZ = 9000.0
JUNO_PLUCK_PEAK_HZ = 500.0
JUNO_PLUCK_PEAK_PER_VELOCITY_HZ = 3400.0
JUNO_PLUCK_BASE_CUTOFF_HZ = 380.0
JUNO_PLUCK_SWEEP_SECONDS = 0.22
JUNO_PLUCK_LADDER_RESONANCE = 0.28
JUNO_PLUCK_CHUNK = 96
JUNO_PLUCK_ENVELOPE_ATTACK = 0.004
JUNO_PLUCK_ENVELOPE_DECAY = 0.45
JUNO_PLUCK_ENVELOPE_RELEASE = 0.35
JUNO_PLUCK_WIDE_RATE_HZ = 0.7
JUNO_PLUCK_WIDE_DEPTH_MS = 1.6
JUNO_PLUCK_WIDE_TAPS = 2
JUNO_PLUCK_WIDE_MIX = 0.5


def _juno_pluck(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    width = np.full(c.n, JUNO_PLUCK_PULSE_WIDTH)
    wave = JUNO_PLUCK_SAW_GAIN * osc(
        SAW, f, c.n, limit=JUNO_PLUCK_PARTIAL_LIMIT_HZ
    ) + JUNO_PLUCK_PULSE_GAIN * _pulse(f, width, c.n)
    top = JUNO_PLUCK_PEAK_HZ + JUNO_PLUCK_PEAK_PER_VELOCITY_HZ * c.vel
    cutoff = JUNO_PLUCK_BASE_CUTOFF_HZ + (top - JUNO_PLUCK_BASE_CUTOFF_HZ) * np.exp(
        -t / JUNO_PLUCK_SWEEP_SECONDS
    )
    wave = ladder(wave, cutoff, JUNO_PLUCK_LADDER_RESONANCE, 1.0, chunk=JUNO_PLUCK_CHUNK)
    wave = wave * envelope(
        c.n,
        JUNO_PLUCK_ENVELOPE_ATTACK,
        JUNO_PLUCK_ENVELOPE_DECAY,
        0.0,
        c.dur,
        JUNO_PLUCK_ENVELOPE_RELEASE,
    )
    return _wide(
        wave,
        c,
        (JUNO_PLUCK_WIDE_RATE_HZ,),
        JUNO_PLUCK_WIDE_DEPTH_MS,
        JUNO_PLUCK_WIDE_TAPS,
        JUNO_PLUCK_WIDE_MIX,
    )


MOOG_SEQ_SAW_GAIN = 0.6
MOOG_SEQ_SQUARE_GAIN = 0.5
MOOG_SEQ_SAW_LIMIT_HZ = 9000.0
MOOG_SEQ_SQUARE_DETUNE = 1.0031
MOOG_SEQ_SQUARE_PHASE = 0.17
MOOG_SEQ_SQUARE_LIMIT_HZ = 7000.0
MOOG_SEQ_MIN_CUTOFF_HZ = 420.0
MOOG_SEQ_CUTOFF_PITCH_RATIO = 2.6
# the peak of the cutoff above its rest: this many Hz times (a floor plus a share that rises with velocity to a power), more on the second patch
MOOG_SEQ_PEAK_HZ = 4200.0
MOOG_SEQ_PEAK_VELOCITY_FLOOR = 0.25
MOOG_SEQ_PEAK_VELOCITY_RANGE = 0.9
MOOG_SEQ_PEAK_VELOCITY_CURVE = 1.6
MOOG_SEQ_PEAK_PATCH_GAIN = 1.25
MOOG_SEQ_SWEEP_SECONDS = 0.10
MOOG_SEQ_SWEEP_PER_VELOCITY_SECONDS = 0.04
MOOG_SEQ_RESONANCE_FLOOR = 0.46
MOOG_SEQ_RESONANCE_PER_VELOCITY = 0.1
MOOG_SEQ_LADDER_DRIVE = 1.6
MOOG_SEQ_CHUNK_LIGHT = 96
MOOG_SEQ_CHUNK_FULL = 48
MOOG_SEQ_ENVELOPE_ATTACK = 0.003
MOOG_SEQ_ENVELOPE_DECAY = 0.5
MOOG_SEQ_ENVELOPE_SUSTAIN = 0.85
MOOG_SEQ_ENVELOPE_RELEASE = 0.07
MOOG_SEQ_PAN_PATCH_0 = -0.05
MOOG_SEQ_PAN_PATCH_1 = 0.1


def _moog_seq(c: Ctx) -> np.ndarray:
    """A sequencer note: band-limited saw and square through the ladder, the cutoff opened by the velocity (the accent)."""  # noqa: D401
    t = _t(c.n)
    f = c.freq
    wave = MOOG_SEQ_SAW_GAIN * osc(
        SAW, f, c.n, limit=MOOG_SEQ_SAW_LIMIT_HZ
    ) + MOOG_SEQ_SQUARE_GAIN * osc(
        SQUARE,
        f * MOOG_SEQ_SQUARE_DETUNE,
        c.n,
        MOOG_SEQ_SQUARE_PHASE,
        limit=MOOG_SEQ_SQUARE_LIMIT_HZ,
    )
    low = max(MOOG_SEQ_MIN_CUTOFF_HZ, MOOG_SEQ_CUTOFF_PITCH_RATIO * f)
    top = low + MOOG_SEQ_PEAK_HZ * (
        MOOG_SEQ_PEAK_VELOCITY_FLOOR
        + MOOG_SEQ_PEAK_VELOCITY_RANGE * c.vel**MOOG_SEQ_PEAK_VELOCITY_CURVE
    ) * (MOOG_SEQ_PEAK_PATCH_GAIN if c.patch else 1.0)
    cutoff = low + (top - low) * np.exp(
        -t / (MOOG_SEQ_SWEEP_SECONDS + MOOG_SEQ_SWEEP_PER_VELOCITY_SECONDS * c.vel)
    )
    wave = ladder(
        wave,
        cutoff,
        MOOG_SEQ_RESONANCE_FLOOR + MOOG_SEQ_RESONANCE_PER_VELOCITY * c.vel,
        MOOG_SEQ_LADDER_DRIVE,
        chunk=MOOG_SEQ_CHUNK_LIGHT if c.tier.unison < RICH_UNISON else MOOG_SEQ_CHUNK_FULL,
    )
    gate = envelope(
        c.n,
        MOOG_SEQ_ENVELOPE_ATTACK,
        MOOG_SEQ_ENVELOPE_DECAY,
        MOOG_SEQ_ENVELOPE_SUSTAIN,
        c.dur,
        MOOG_SEQ_ENVELOPE_RELEASE,
    )
    return _mono(wave * gate, MOOG_SEQ_PAN_PATCH_0 if c.patch == 0 else MOOG_SEQ_PAN_PATCH_1)


BLIP_SYNTH_SQUARE_LIMIT_HZ = 7000.0
BLIP_SYNTH_SQUARE_GAIN = 0.8
BLIP_SYNTH_TRIANGLE_GAIN = 0.3
BLIP_SYNTH_TRIANGLE_RATIO = 2.0
BLIP_SYNTH_CUTOFF_PITCH_RATIO = 2.0
BLIP_SYNTH_RESONANCE = 0.5
BLIP_SYNTH_PEAK_HZ = 500.0
BLIP_SYNTH_PEAK_PER_VELOCITY_HZ = 3200.0
BLIP_SYNTH_MIN_CUTOFF_HZ = 280.0
BLIP_SYNTH_PEAK_MIN_RATIO = 3.0
BLIP_SYNTH_SWEEP_BASE_HZ = 280.0
BLIP_SYNTH_SWEEP_SECONDS = 0.05
BLIP_SYNTH_LADDER_DRIVE = 1.3
BLIP_SYNTH_CHUNK_LIGHT = 96
BLIP_SYNTH_CHUNK_FULL = 48
BLIP_SYNTH_ENVELOPE_ATTACK = 0.002
BLIP_SYNTH_ENVELOPE_DECAY = 0.07
BLIP_SYNTH_PATCH_DECAY = 0.05
BLIP_SYNTH_ENVELOPE_RELEASE = 0.05
BLIP_SYNTH_PAN = 0.15


def _blip_synth(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    wave = osc(
        SQUARE, c.freq, c.n, limit=BLIP_SYNTH_SQUARE_LIMIT_HZ
    ) * BLIP_SYNTH_SQUARE_GAIN + BLIP_SYNTH_TRIANGLE_GAIN * osc(
        TRIANGLE, c.freq * BLIP_SYNTH_TRIANGLE_RATIO, c.n
    )
    top = BLIP_SYNTH_PEAK_HZ + BLIP_SYNTH_PEAK_PER_VELOCITY_HZ * c.vel
    cutoff = max(BLIP_SYNTH_MIN_CUTOFF_HZ, BLIP_SYNTH_CUTOFF_PITCH_RATIO * c.freq) + (
        max(top, BLIP_SYNTH_PEAK_MIN_RATIO * c.freq) - BLIP_SYNTH_SWEEP_BASE_HZ
    ) * np.exp(-t / BLIP_SYNTH_SWEEP_SECONDS)
    wave = ladder(
        wave,
        cutoff,
        BLIP_SYNTH_RESONANCE,
        BLIP_SYNTH_LADDER_DRIVE,
        chunk=BLIP_SYNTH_CHUNK_LIGHT if c.tier.unison < RICH_UNISON else BLIP_SYNTH_CHUNK_FULL,
    )
    return _mono(
        wave
        * envelope(
            c.n,
            BLIP_SYNTH_ENVELOPE_ATTACK,
            BLIP_SYNTH_ENVELOPE_DECAY + BLIP_SYNTH_PATCH_DECAY * (c.patch % 2),
            0.0,
            c.dur,
            BLIP_SYNTH_ENVELOPE_RELEASE,
        ),
        BLIP_SYNTH_PAN,
    )


CHORD_STAB_SAW_GAIN = 0.6
CHORD_STAB_DETUNE_UP = 1.004
CHORD_STAB_PARTIAL_LIMIT_HZ = 8000.0
CHORD_STAB_DETUNE_DOWN = 0.996
CHORD_STAB_DOWN_PHASE = 0.31
CHORD_STAB_PEAK_HZ = 900.0
CHORD_STAB_PEAK_PER_VELOCITY_HZ = 2100.0
CHORD_STAB_MIN_CUTOFF_HZ = 260.0
CHORD_STAB_CUTOFF_PITCH_RATIO = 1.8
CHORD_STAB_SWEEP_BASE_HZ = 260.0
CHORD_STAB_SWEEP_SECONDS = 0.18
CHORD_STAB_LADDER_RESONANCE = 0.32
CHORD_STAB_CHUNK = 64
CHORD_STAB_ENVELOPE_ATTACK = 0.004
CHORD_STAB_ENVELOPE_DECAY = 0.3
CHORD_STAB_PATCH_DECAY = 0.1
CHORD_STAB_ENVELOPE_RELEASE = 0.25
CHORD_STAB_PAN = -0.15


def _chord_stab(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    wave = CHORD_STAB_SAW_GAIN * osc(
        SAW, f * CHORD_STAB_DETUNE_UP, c.n, limit=CHORD_STAB_PARTIAL_LIMIT_HZ
    ) + CHORD_STAB_SAW_GAIN * osc(
        SAW,
        f * CHORD_STAB_DETUNE_DOWN,
        c.n,
        CHORD_STAB_DOWN_PHASE,
        limit=CHORD_STAB_PARTIAL_LIMIT_HZ,
    )
    top = CHORD_STAB_PEAK_HZ + CHORD_STAB_PEAK_PER_VELOCITY_HZ * c.vel
    cutoff = max(CHORD_STAB_MIN_CUTOFF_HZ, CHORD_STAB_CUTOFF_PITCH_RATIO * f) + (
        top - CHORD_STAB_SWEEP_BASE_HZ
    ) * np.exp(-t / CHORD_STAB_SWEEP_SECONDS)
    wave = ladder(wave, cutoff, CHORD_STAB_LADDER_RESONANCE, 1.0, chunk=CHORD_STAB_CHUNK)
    wave = wave * envelope(
        c.n,
        CHORD_STAB_ENVELOPE_ATTACK,
        CHORD_STAB_ENVELOPE_DECAY + CHORD_STAB_PATCH_DECAY * (c.patch % 2),
        0.0,
        c.dur,
        CHORD_STAB_ENVELOPE_RELEASE,
    )
    return dsp.to_stereo(wave, CHORD_STAB_PAN)


NYLON_KS_MIN_PERIOD_SAMPLES = 8
# the averaging filter delays the loop by half a sample
NYLON_KS_LOOP_DELAY_SAMPLES = 0.5
NYLON_KS_SEED_PAD_SAMPLES = 8
NYLON_KS_SEED_SKIP = 4
NYLON_KS_SEED_TAPS = 5
NYLON_KS_LEVEL_FLOOR = 0.6
NYLON_KS_LEVEL_PER_VELOCITY = 0.4
NYLON_KS_LOOP_DECAY = 0.9985
NYLON_KS_LOOP_DECAY_PER_PITCH = 0.0008
NYLON_KS_DECAY_PITCH_HZ = 880.0
NYLON_KS_DECAY_PITCH_MAX = 1.5
NYLON_KS_ENVELOPE_ATTACK = 0.002
NYLON_KS_ENVELOPE_DECAY = 2.0
NYLON_KS_ENVELOPE_HOLD = 1.5
NYLON_KS_ENVELOPE_RELEASE = 0.4
NYLON_KS_PAN = 0.1


def _nylon_ks(c: Ctx) -> np.ndarray:
    period = SR / c.freq
    size = max(round(period), NYLON_KS_MIN_PERIOD_SAMPLES)
    actual = SR / (size + NYLON_KS_LOOP_DELAY_SAMPLES)
    length = int(c.n * actual / c.freq) + 2 * size
    seed = np.convolve(
        c.rng.uniform(-1, 1, size + NYLON_KS_SEED_PAD_SAMPLES),
        np.ones(NYLON_KS_SEED_TAPS) / NYLON_KS_SEED_TAPS,
        mode="same",
    )[NYLON_KS_SEED_SKIP : size + NYLON_KS_SEED_SKIP]
    out = np.zeros(length + size + 1, dtype=np.float64)
    out[1 : size + 1] = seed * (NYLON_KS_LEVEL_FLOOR + NYLON_KS_LEVEL_PER_VELOCITY * c.vel)
    decay = NYLON_KS_LOOP_DECAY - NYLON_KS_LOOP_DECAY_PER_PITCH * min(
        c.freq / NYLON_KS_DECAY_PITCH_HZ, NYLON_KS_DECAY_PITCH_MAX
    )
    for start in range(size + 1, length + size + 1, size):
        end = min(start + size, len(out))
        out[start:end] = (
            decay * 0.5 * (out[start - size : end - size] + out[start - size - 1 : end - size - 1])
        )
    raw = out[1 : length + 1].astype(np.float32)
    # the loop is a little flat for the averaging filter: read it a touch faster so that it is in tune
    pos = np.arange(c.n) * (actual / c.freq)
    wave = read_cubic(raw, pos)
    return dsp.to_stereo(
        (
            wave
            * envelope(
                c.n,
                NYLON_KS_ENVELOPE_ATTACK,
                NYLON_KS_ENVELOPE_DECAY,
                1.0,
                c.dur + NYLON_KS_ENVELOPE_HOLD,
                NYLON_KS_ENVELOPE_RELEASE,
            )
        ).astype(np.float32),
        NYLON_KS_PAN,
    )


VOICES: dict[str, Voice] = {
    v.key: v
    for v in (
        Voice("sub_round", "Warm round sub", FAMILY_BASS, 36, 0.2, _sub_round, patches=2),
        Voice("moog_bass", "Moog-style ladder bass", FAMILY_BASS, 41, 0.2, _moog_bass),
        Voice("reese", "Growling Reese bass", FAMILY_BASS, 36, 0.3, _reese),
        Voice("fm_growl", "FM growl bass", FAMILY_BASS, 38, 0.3, _fm_growl),
        Voice("acid_bass", "Acid 303-style bass", FAMILY_BASS, 40, 0.1, _acid_bass),
        Voice("string_machine", "String machine", FAMILY_PAD, 57, 1.6, _string_machine),
        Voice("juno_pad", "Juno-style pad", FAMILY_PAD, 57, 1.4, _juno_pad, fundamental=0.5),
        Voice("supersaw", "Supersaw ensemble", FAMILY_PAD, 57, 1.1, _supersaw),
        Voice("tape_strings", "Tape strings", FAMILY_PAD, 55, 2.0, _tape_strings, sampled=True),
        Voice(
            "chant_cello", "Chant cello section", FAMILY_LEAD, 60, 1.6, _chant_cello, sampled=True
        ),
        Voice("tape_choir", "Tape choir", FAMILY_PAD, 57, 1.8, _tape_choir),
        Voice("glass_pad", "Glassy wavetable pad", FAMILY_PAD, 60, 2.2, _glass_pad),
        Voice("bowed_pad", "Bowed pad", FAMILY_PAD, 55, 1.7, _bowed_pad),
        Voice("drawbar_organ", "Drawbar organ", FAMILY_PAD, 48, 1.3, _drawbar_organ),
        Voice("analog_lead", "Singing analog lead", FAMILY_LEAD, 64, 0.8, _analog_lead),
        Voice("flute_lead", "Flute-like lead", FAMILY_LEAD, 72, 0.5, _flute_lead),
        Voice("fm_ep", "FM electric piano", FAMILY_KEY, 60, 1.2, _fm_ep),
        Voice("felt_piano", "Felt piano", FAMILY_KEY, 60, 0.6, _felt_piano, sampled=True),
        Voice("grand_piano", "Soft grand piano", FAMILY_KEY, 60, 0.6, _grand_piano, sampled=True),
        Voice("tx_ep", "Sampled FM electric piano", FAMILY_KEY, 60, 0.6, _tx_ep, sampled=True),
        Voice(
            "moog_seq",
            "Moog-style sequencer voice",
            FAMILY_PLUCK,
            55,
            0.15,
            _moog_seq,
            velocity_steps=4,
        ),
        Voice("blip_synth", "Blip synth", FAMILY_PLUCK, 67, 0.2, _blip_synth, velocity_steps=4),
        Voice("chord_stab", "Chord stab", FAMILY_PLUCK, 57, 0.5, _chord_stab),
        Voice("juno_pluck", "Juno-style pluck", FAMILY_PLUCK, 64, 0.4, _juno_pluck),
        Voice("nylon_ks", "Nylon pluck (Karplus-Strong)", FAMILY_PLUCK, 60, 0.6, _nylon_ks),
        Voice("strum_pluck", "Strumstick pluck", FAMILY_PLUCK, 55, 0.4, _strum_pluck, sampled=True),
        Voice(
            "slit_drum",
            "Slit drum (wooden log drum)",
            FAMILY_PLUCK,
            48,
            0.25,
            _slit_drum,
            sampled=True,
        ),
        Voice("woodblock", "Woodblock", FAMILY_PLUCK, 84, 0.15, _woodblock, sampled=True),
    )
}


NOTES = NoteCache()


def _make(key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int) -> np.ndarray:
    voice = VOICES[key]
    n = max(int((dur + voice.release) * SR), MIN_NOTE_SAMPLES)
    rng = np.random.default_rng(
        stable_hash(
            key,
            round(midi, KEY_DECIMALS),
            round(dur, KEY_DECIMALS),
            round(vel, KEY_DECIMALS),
            patch,
        )
        % SEED_MODULUS
    )
    ctx = Ctx(midi, midi_hz(midi), dur, vel, tier, rng, patch, n)
    wave = np.asarray(voice.make(ctx), dtype=np.float32)
    if wave.ndim == 1:
        wave = dsp.to_stereo(wave)
    if not voice.sampled:
        # a softer key is a quieter note as well as a darker one
        wave = wave * np.float32((max(vel, MIN_VELOCITY) / REF_VELOCITY) ** VELOCITY_POWER)
    return wave


def _core(key: str, midi: float, tier: Quality, patch: int) -> np.ndarray:
    """Return the core of a voice at a pitch: a note held for CORE_HOLD seconds, with its release."""
    cache_key = (CACHE_TAG_CORE, key, round(midi, KEY_DECIMALS), _tier_key(tier), patch)
    found = NOTES.get(cache_key)
    if found is None:
        found = (
            _make(key, midi, CORE_HOLD, REF_VELOCITY, tier, patch) * np.float32(gain_of(key, tier))
        ).astype(np.float16)
        NOTES.put(cache_key, found, tier.cache_mb)
    return found.astype(np.float32)


def _align(core: np.ndarray, end: int, target: int, cross: int, search: int) -> int:
    """
    Return the point near `target` that the end of a piece can jump to without a seam.

    The `cross` samples before `end` are compared, by normalised correlation, with the `cross` samples before
    each candidate point within `search` of the target; a tone that is lined up in phase crossfades without a dip.
    """
    mono = core.mean(axis=1)
    low = max(target - search, cross)
    high = min(target + search, len(mono))
    ref = mono[end - cross : end]
    region = mono[low - cross : high]
    size = 1 << int(np.ceil(np.log2(len(region) + cross)))
    corr = np.fft.irfft(np.fft.rfft(region, size) * np.conj(np.fft.rfft(ref, size)), size)[
        : high - low
    ]
    energy = np.concatenate(([0.0], np.cumsum(region.astype(np.float64) ** 2)))
    local = energy[cross : cross + (high - low)] - energy[: high - low]
    score = corr / np.sqrt(
        np.maximum(local, ENERGY_FLOOR)
        * max(float(np.sum(ref.astype(np.float64) ** 2)), ENERGY_FLOOR)
    )
    return int(low + int(np.argmax(score)))


def _tiled(key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int) -> np.ndarray:
    """
    Make a long note from the looped sustain of a core: the attack, the loop as often as needed, the release.

    Every join fades the end of what is there into what comes before the point that it jumps to, which is how a
    sampler loops, and the point is chosen where the tone is in phase, so that the join has neither a seam nor a dip.
    """
    core = _core(key, midi, tier, patch)
    hold, cross = int(CORE_HOLD * SR), int(CORE_CROSS * SR)
    jump_key = (key, round(midi, KEY_DECIMALS), _tier_key(tier), patch)
    start = _JUMPS.get(jump_key)
    if start is None:
        start = _align(core, hold, int(CORE_LOOP_START * SR), cross, int(CORE_ALIGN_SEARCH * SR))
        with _JUMPS_LOCK:
            start = _JUMPS.setdefault(jump_key, start)
    want = int(dur * SR)
    fade = np.linspace(0.0, np.pi / 2.0, cross, dtype=np.float32)
    fade_out, fade_in = np.cos(fade)[:, None], np.sin(fade)[:, None]
    pieces = [core[:hold].copy()]
    pos, at = hold, hold
    while pos < want:
        prev = pieces[-1]
        prev[-cross:] = prev[-cross:] * fade_out + core[start - cross : start] * fade_in
        count = max(min(hold - start, want - pos), cross)
        pieces.append(core[start : start + count].copy())
        pos += count
        at = start + count
    tail_from = hold
    if at != hold:
        prev = pieces[-1]
        prev[-cross:] = prev[-cross:] * fade_out + core[tail_from - cross : tail_from] * fade_in
    out = np.concatenate([*pieces, core[tail_from:]])
    return out * np.float32((vel / REF_VELOCITY) ** VELOCITY_POWER)


def _plain_note(
    key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int
) -> np.ndarray:
    """Return a note without vibrato; `dur` and `vel` are already on the grid of the cache."""
    cache_key = (key, round(midi, KEY_DECIMALS), dur, vel, _tier_key(tier), patch)
    found = NOTES.get(cache_key)
    if found is not None:
        return found
    if dur >= TILE_FROM and VOICES[key].family in TILED_FAMILIES:
        # assembled from a cached core in a few milliseconds: not worth keeping, a long note is big
        return _tiled(key, midi, dur, vel, tier, patch)
    wave = _make(key, midi, dur, vel, tier, patch) * np.float32(gain_of(key, tier))
    NOTES.put(cache_key, wave, tier.cache_mb)
    return wave
