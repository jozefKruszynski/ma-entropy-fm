"""
The signal processing of the soundscapes: band-limited oscillators, envelopes, a ladder filter and tape.

Band-limited oscillators, envelopes, a resonant ladder filter with a moving cutoff, a modulated delay for
chorus and tape, and noise: numpy and pedalboard.

Every oscillator reads a wavetable that has only the partials that fit under the limit of its octave,
so nothing aliases. Notes are made in one piece from absolute positions in time, which is what lets a
soundscape start at any second: nothing here keeps state between notes.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import LadderFilter

from .synth import A4_HZ, A4_MIDI, SAMPLE_RATE, SEMITONES_PER_OCTAVE

if TYPE_CHECKING:
    import numpy.typing as npt

SR = SAMPLE_RATE
TABLE_SIZE = 4096
# the highest partial that a table holds, in Hz
PARTIAL_LIMIT = 19000.0
# the number of band-limited tables of a shape, two to an octave
TABLE_LEVELS = 22
LOWEST = 20.0
SAW = "saw"
SQUARE = "square"
TRIANGLE = "triangle"
SINE = "sine"
KINDS = (SAW, SQUARE, TRIANGLE, SINE)
LADDER_LP24 = "lp24"
LADDER_LP12 = "lp12"
LADDER_BP24 = "bp24"
LADDER_HP24 = "hp24"
CENTS_PER_OCTAVE = 1200.0
# the divisor that turns an amplitude ratio's decibels into the exponent of 10
AMPLITUDE_DB_DIVISOR = 20.0
# what is added to a measured peak or RMS that is divided by, or taken as the least of a gain, so that silence is not
# a division by zero
SILENCE_FLOOR = 1e-9
# the lowest peak or RMS that is worth bringing to a level, and the lowest gain that is taken a logarithm of
MIN_LEVEL = 1e-6
# the tables of the shapes are made two to an octave
LEVELS_PER_OCTAVE = 2.0
# the lowest limit that a shape is dulled to, in Hz
MIN_PARTIAL_LIMIT = 100.0
SHAPE_TABLES_CACHED = 64
HARMONIC_TABLES_CACHED = 256
# the shortest time that a rise, a decay or a fade is worked out over, in seconds
MIN_ENVELOPE_SECONDS = 1e-3
MIN_FADE_SECONDS = 0.05
# the delay line of a vibrato never reads closer than this to the present, in samples
VIBRATO_BASE_DELAY_SAMPLES = 2.0
# the release is the time that the level takes to fall to the rest, a share of it is the time constant
ENVELOPE_RELEASE_DIVISOR = 4.0
# the last stretch of every envelope is faded to nothing, in seconds
ENVELOPE_END_FADE_SECONDS = 0.02
CUTOFF_MIN_HZ = 20.0
CUTOFF_MAX_HZ = 18000.0
LADDER_CHUNK = 128
STREAM_LADDER_CHUNK = 256
STREAM_LADDER_START_CUTOFF_HZ = 2000.0
MS_TO_SECONDS = 1e-3
# the grains of the shimmer, in seconds
SHIMMER_GRAIN_SECONDS = 0.06
ENSEMBLE_BASE_DELAY_SECONDS = 0.006
# a tap whose LFO is faster than this swings only a small share of the depth
ENSEMBLE_FAST_RATE_HZ = 2.0
ENSEMBLE_FAST_DEPTH_SHARE = 0.06
ENSEMBLE_TAP_DELAY_SPREAD = 0.4
ENSEMBLE_WIDTH = 0.8
# a band of noise is cut from a loop of this many seconds
NOISE_LOOP_SECONDS = 6
NOISE_SEED_LOW_FACTOR = 7
# the band edge of the noise is smoothed by a Hann window of a 400th of the loop, 5 samples at the least
NOISE_EDGE_DIVISOR = 400
NOISE_EDGE_MIN_SAMPLES = 5
_LOOPS_LOCK = threading.Lock()
_LOOPS: dict[tuple[float, float], np.ndarray] = {}
_LADDER_MODES = {
    LADDER_LP24: LadderFilter.Mode.LPF24,
    LADDER_LP12: LadderFilter.Mode.LPF12,
    LADDER_BP24: LadderFilter.Mode.BPF24,
    LADDER_HP24: LadderFilter.Mode.HPF24,
}


def midi_hz(note: float) -> float:
    """
    Return the frequency of a MIDI note.

    :param note: The note number, 69 is A at 440 Hz.
    """
    return float(A4_HZ * 2.0 ** ((note - A4_MIDI) / SEMITONES_PER_OCTAVE))


@lru_cache(maxsize=SHAPE_TABLES_CACHED)
def table(kind: str, level: int) -> np.ndarray:
    """
    Return one cycle of a shape that is band-limited for the octave `level` (two levels an octave).

    :param kind: saw, square, triangle or sine.
    :param level: 0 is for the notes up to about 28 Hz; the partials stop under 19 kHz of the top note.
    """
    top = LOWEST * 2.0 ** ((level + 1) / LEVELS_PER_OCTAVE)
    count = max(int(PARTIAL_LIMIT / top), 1)
    _k, amp = _harmonics(kind, count)
    spectrum = np.zeros(TABLE_SIZE // 2 + 1, dtype=np.complex128)
    count = min(count, TABLE_SIZE // 2 - 1)
    spectrum[1 : count + 1] = -1j * amp[:count] * TABLE_SIZE / 2.0
    wave = np.fft.irfft(spectrum, TABLE_SIZE)
    peak = float(np.abs(wave).max()) or 1.0
    wave = wave / peak
    return np.append(wave, wave[0]).astype(np.float32)


@lru_cache(maxsize=HARMONIC_TABLES_CACHED)
def harmonic_table(amps: tuple[float, ...], phases: tuple[float, ...] = ()) -> np.ndarray:
    """
    Return one cycle made of the given partial amplitudes (partial 1 first), normalised to a peak of 1.

    :param amps: The amplitude of each partial; the table has exactly these and no others.
    :param phases: Optional phase of each partial in radians.
    """
    count = min(len(amps), TABLE_SIZE // 2 - 1)
    spectrum = np.zeros(TABLE_SIZE // 2 + 1, dtype=np.complex128)
    ph = np.array(phases[:count] if phases else np.zeros(count))
    if len(ph) < count:
        ph = np.pad(ph, (0, count - len(ph)))
    spectrum[1 : count + 1] = -1j * np.array(amps[:count]) * np.exp(1j * ph) * TABLE_SIZE / 2.0
    wave = np.fft.irfft(spectrum, TABLE_SIZE)
    wave = wave / (float(np.abs(wave).max()) or 1.0)
    return np.append(wave, wave[0]).astype(np.float32)


def level_for(freq: float, limit: float = PARTIAL_LIMIT) -> int:
    """
    Return the table level for the highest frequency that will be played, with a limit on the partials.

    :param freq: The highest pitch in Hz.
    :param limit: The highest partial that may sound, in Hz; lowers the brightness of the shape.
    """
    base = max(math.ceil(LEVELS_PER_OCTAVE * math.log2(max(freq, LOWEST) / LOWEST)) - 1, 0)
    if limit < PARTIAL_LIMIT:
        # a duller shape: use the table of a higher octave, which has fewer partials
        base += max(
            round(LEVELS_PER_OCTAVE * math.log2(PARTIAL_LIMIT / max(limit, MIN_PARTIAL_LIMIT))), 0
        )
    return min(base, TABLE_LEVELS - 1)


def read_table(tab: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Read a table at phases in cycles (any real numbers) with a linear interpolation."""
    # x - floor(x) is the same number as x % 1.0 for every float (it is exact), and several times quicker
    pos = (phase - np.floor(phase)) * TABLE_SIZE
    i0 = pos.astype(np.int64)
    frac = (pos - i0).astype(np.float32)
    return tab[i0] * (1.0 - frac) + tab[i0 + 1] * frac


def osc(
    kind: str,
    freq: float | npt.NDArray[np.float64],
    n: int,
    phase0: float = 0.0,
    limit: float = PARTIAL_LIMIT,
) -> np.ndarray:
    """
    Return a band-limited oscillator.

    :param kind: saw, square, triangle or sine.
    :param freq: The pitch in Hz, one value or one per sample (a vibrato).
    :param n: The number of samples.
    :param phase0: The starting phase in cycles.
    :param limit: The highest partial in Hz (a duller sound when lower).
    """
    if kind == SINE:
        return sine(freq, n, phase0)
    if np.isscalar(freq):
        phase = phase0 + float(freq) * np.arange(n, dtype=np.float64) / SR  # type: ignore[arg-type]
        top = float(freq)  # type: ignore[arg-type]
    else:
        arr = np.asarray(freq, dtype=np.float64)
        phase = phase0 + np.cumsum(arr) / SR
        top = float(arr.max())
    return read_table(table(kind, level_for(top, limit)), phase)


def sine(freq: float | npt.NDArray[np.float64], n: int, phase0: float = 0.0) -> np.ndarray:
    """Return a sine, with a constant pitch or one per sample."""
    if np.isscalar(freq):
        phase = phase0 + float(freq) * np.arange(n, dtype=np.float64) / SR  # type: ignore[arg-type]
    else:
        phase = phase0 + np.cumsum(np.asarray(freq, dtype=np.float64)) / SR
    return np.sin(2.0 * np.pi * phase).astype(np.float32)


def wavetable_osc(
    tab: np.ndarray, freq: float | npt.NDArray[np.float64], n: int, phase0: float = 0.0
) -> np.ndarray:
    """Play a table made by `harmonic_table` (its partials were chosen to fit, so it is not mipmapped)."""
    if np.isscalar(freq):
        phase = phase0 + float(freq) * np.arange(n, dtype=np.float64) / SR  # type: ignore[arg-type]
    else:
        phase = phase0 + np.cumsum(np.asarray(freq, dtype=np.float64)) / SR
    return read_table(tab, phase)


def vibrato(n: int, rate: float, cents: float, delay: float, phase: float = 0.0) -> np.ndarray:
    """
    Return a frequency multiplier for a vibrato of at most `cents`, that fades in after `delay`.

    :param n: Samples.
    :param rate: Hz.
    :param cents: The depth, a peak value (15 or less keeps it a vibrato and not a glide).
    :param delay: Seconds before it starts; it fades in over the same time.
    :param phase: Starting phase in cycles.
    """
    t = np.arange(n) / SR
    depth = (
        np.clip((t - delay) / max(delay, MIN_FADE_SECONDS), 0.0, 1.0) if delay > 0 else np.ones(n)
    )
    return np.power(
        2.0, depth * cents / CENTS_PER_OCTAVE * np.sin(2.0 * np.pi * (rate * t + phase))
    ).astype(np.float64)


def vibrato_phase(
    freq: float, n: int, rate: float, cents: float, delay: float, phase: float = 0.0
) -> np.ndarray:
    """
    Return the phase in cycles of a tone of `freq` Hz with a vibrato of at most `cents`, fading in after `delay`.

    The integral of the vibrato is worked out in closed form (the depth moves slowly), so it costs a few
    operations a sample and is exact to a hundredth of a cent. A vibrato of 15 cents or less is not a glide.

    :param freq: The pitch in Hz.
    :param n: Samples.
    :param rate: The vibrato rate in Hz.
    :param cents: The depth, a peak value.
    :param delay: Seconds before it starts; it fades in over the same time.
    :param phase: Starting phase of the vibrato in cycles.
    """
    t = np.arange(n, dtype=np.float64) / SR
    depth = np.clip((t - delay) / max(delay, MIN_FADE_SECONDS), 0.0, 1.0) if delay > 0 else 1.0
    swing = cents / CENTS_PER_OCTAVE * math.log(2.0) * freq / (2.0 * math.pi * rate)
    return freq * t - swing * depth * np.cos(2.0 * np.pi * (rate * t + phase))


def envelope(
    n: int, attack: float, decay: float, sustain: float, hold: float, release: float
) -> np.ndarray:
    """
    Return an ADSR over `n` samples: the key is held for `hold` seconds, then `release` seconds die away.

    :param n: Total samples, including the release.
    :param attack: Seconds of the rise (a smooth curve).
    :param decay: Seconds in which the level falls to the sustain.
    :param sustain: The held level, 0 to 1.
    :param hold: Seconds that the key is down.
    :param release: Seconds of the fall after the key goes up.
    """
    t = np.arange(n, dtype=np.float64) / SR
    rise = np.clip(t / max(attack, MIN_ENVELOPE_SECONDS), 0.0, 1.0)
    rise = rise * rise * (3.0 - 2.0 * rise)
    fall = sustain + (1.0 - sustain) * np.exp(
        -np.maximum(t - attack, 0.0) / max(decay, MIN_ENVELOPE_SECONDS)
    )
    env = rise * fall
    after = np.maximum(t - hold, 0.0)
    held = env[min(int(hold * SR), n - 1)] if n else 0.0
    env = np.where(
        t > hold,
        held * np.exp(-after / max(release / ENVELOPE_RELEASE_DIVISOR, MIN_ENVELOPE_SECONDS)),
        env,
    )
    ending = min(int(ENVELOPE_END_FADE_SECONDS * SR), n)
    if ending:
        env[-ending:] *= np.linspace(1.0, 0.0, ending)
    return env.astype(np.float32)


def ladder(
    x: np.ndarray,
    cutoff: float | np.ndarray,
    resonance: float = 0.3,
    drive: float = 1.0,
    chunk: int = LADDER_CHUNK,
    mode: str = LADDER_LP24,
) -> np.ndarray:
    """
    Filter a mono signal with a resonant Moog-style ladder whose cutoff may move.

    :param x: Mono samples.
    :param cutoff: Hz, one value or one per sample (it is read every `chunk` samples).
    :param resonance: 0 to 1.
    :param drive: Input drive of the ladder, 1 and up.
    :param chunk: Samples between two cutoff updates.
    :param mode: lp24, lp12, bp24 or hp24.
    """
    plug = LadderFilter(
        mode=_LADDER_MODES[mode], cutoff_hz=float(np.mean(cutoff)), resonance=resonance, drive=drive
    )
    sig = np.ascontiguousarray(x, dtype=np.float32)[None, :]
    if np.isscalar(cutoff):
        return np.asarray(plug.process(sig, SR, reset=True)[0], dtype=np.float32)
    arr = np.asarray(cutoff, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float32)
    starts = range(0, len(x), chunk)
    # one clip for all the chunks: the loop below calls a native filter, and anything else in it is overhead
    cuts = np.clip(
        arr[np.minimum(np.arange(0, len(x), chunk) + chunk // 2, len(arr) - 1)],
        CUTOFF_MIN_HZ,
        CUTOFF_MAX_HZ,
    )
    for start, cut in zip(starts, cuts.tolist(), strict=True):
        plug.cutoff_hz = cut
        out[start : start + chunk] = plug.process(sig[:, start : start + chunk], SR, reset=False)[0]
    return out


class StreamLadder:
    """A ladder filter that keeps its state between blocks, for the buses of the mix."""

    def __init__(self, resonance: float = 0.2, drive: float = 1.0, mode: str = LADDER_LP24) -> None:
        """
        Make the filter.

        :param resonance: 0 to 1.
        :param drive: 1 and up.
        :param mode: lp24, lp12, bp24 or hp24.
        """
        self.plug = LadderFilter(
            mode=_LADDER_MODES[mode],
            cutoff_hz=STREAM_LADDER_START_CUTOFF_HZ,
            resonance=resonance,
            drive=drive,
        )

    def process(
        self, block: np.ndarray, cutoff: np.ndarray, chunk: int = STREAM_LADDER_CHUNK
    ) -> np.ndarray:
        """
        Filter a stereo block (n, 2) with a cutoff for each chunk of samples.

        :param block: Float samples, shape (n, 2).
        :param cutoff: Hz, one per chunk: its length is the number of chunks in the block.
        :param chunk: Samples between cutoff updates.
        """
        sig = np.ascontiguousarray(block.T, dtype=np.float32)
        out = np.empty_like(sig)
        for index, start in enumerate(range(0, sig.shape[1], chunk)):
            self.plug.cutoff_hz = float(
                min(max(cutoff[min(index, len(cutoff) - 1)], CUTOFF_MIN_HZ), CUTOFF_MAX_HZ)
            )
            out[:, start : start + chunk] = self.plug.process(
                sig[:, start : start + chunk], SR, reset=False
            )
        return np.ascontiguousarray(out.T)


def delay_read(x: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Read `x` at positions `n - delay[n]` (samples) with a linear interpolation; before the start is silence."""
    n = len(x)
    pos = np.arange(n, dtype=np.float32)
    pos -= delay.astype(np.float32)
    behind = pos < 0.0
    np.maximum(pos, 0.0, out=pos)
    i0 = pos.astype(np.int32)
    frac = pos - i0
    src = x.astype(np.float32, copy=False)
    low = src[i0]
    out = low + (src[np.minimum(i0 + 1, n - 1)] - low) * frac
    out[behind] = 0.0
    return out


def pitch_vibrato(
    wave: np.ndarray,
    cents: float,
    rate: float,
    onset: float = 1.5,
    phase: float = 0.0,
    wait: float = 0.3,
) -> np.ndarray:
    """
    Return a finished note with a vibrato of `cents` (a peak value) at `rate` Hz, which fades in after `wait` seconds.

    The pitch is moved by a delay line that swings by a fraction of a millisecond, so it works on any sound, a
    sample or a pad, and costs one interpolated read a sample. The depth rises smoothly over `onset` seconds, so
    a note begins steady and starts to sing only once it is held. A vibrato of a few cents is no glide.

    :param wave: Samples, mono (n,) or stereo (n, 2).
    :param cents: The depth, a peak value.
    :param rate: The rate in Hz.
    :param onset: Seconds that the depth takes to reach its full value, after `wait`.
    :param phase: Starting phase of the swing in cycles.
    :param wait: Seconds of a steady tone before the vibrato begins.
    """
    n = len(wave)
    t = np.arange(n, dtype=np.float32) / SR
    ramp = np.clip((t - wait) / max(onset, MIN_FADE_SECONDS), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    # a delay of A sin(w t) samples bends the pitch by A w / SR of itself
    swing = (2.0 ** (cents / CENTS_PER_OCTAVE) - 1.0) * SR / (2.0 * math.pi * rate)
    delay = (
        swing
        + VIBRATO_BASE_DELAY_SAMPLES
        + swing * ramp * np.sin(np.float32(2.0 * np.pi) * (rate * t + phase))
    )
    # the positions are float64: a float32 position is only good to a thirtieth of a sample late in a long note,
    # which is a jitter that can be heard under a pure tone
    pos = np.arange(n, dtype=np.float64) - delay
    first = pos.astype(np.int32)
    frac = (pos - first).astype(np.float32)
    np.minimum(first, n - 2, out=first)
    src = wave.astype(np.float32, copy=False)
    # the read positions are the same for every channel, so they are worked out once
    if src.ndim == 1:
        low = src[first]
        return np.asarray(low + (src[first + 1] - low) * frac, dtype=np.float32)
    out = np.empty_like(src)
    for k in range(src.shape[1]):
        column = np.ascontiguousarray(src[:, k])
        low = column[first]
        out[:, k] = low + (column[first + 1] - low) * frac
    return out


def ensemble(
    mono: np.ndarray,
    rng: np.random.Generator,
    rates: tuple[float, ...] = (0.55, 0.7, 5.5),
    depth_ms: float = 2.2,
    taps: int = 3,
    mix: float = 0.7,
) -> np.ndarray:
    """
    Make a stereo ensemble from a mono signal: several delay lines that LFOs move, like a string machine.

    :param mono: The dry signal.
    :param rng: Seeds the LFO phases.
    :param rates: The LFO rates in Hz, slow ones first.
    :param depth_ms: The swing of the delay in milliseconds.
    :param taps: How many delay lines.
    :param mix: The share of the delayed signal.
    """
    n = len(mono)
    t = np.arange(n) / SR
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    base = ENSEMBLE_BASE_DELAY_SECONDS * SR
    for tap in range(taps):
        rate = rates[tap % len(rates)]
        phase = rng.uniform(0, 1)
        mod = np.sin(2 * np.pi * (rate * t + phase))
        # the fast LFO is small, the slow ones carry the swing; the tap is pitch modulation of a few cents
        size = (
            depth_ms
            * MS_TO_SECONDS
            * SR
            * (ENSEMBLE_FAST_DEPTH_SHARE if rate > ENSEMBLE_FAST_RATE_HZ else 1.0)
        )
        delayed = delay_read(mono, base * (1 + ENSEMBLE_TAP_DELAY_SPREAD * tap) + size * mod)
        side = -1.0 + 2.0 * (tap % 2) if taps > 1 else 0.0
        left += delayed * (0.5 - 0.5 * side * ENSEMBLE_WIDTH)
        right += delayed * (0.5 + 0.5 * side * ENSEMBLE_WIDTH)
    left /= taps / 2.0
    right /= taps / 2.0
    dry = mono * (1.0 - mix)
    return np.stack([dry + left * mix, dry + right * mix], axis=1).astype(np.float32)


def to_stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Return a mono signal as (n, 2) with an equal-power pan, -1 left to 1 right."""
    angle = (pan + 1.0) * math.pi / 4.0
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=1).astype(np.float32)


def noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return white noise with a standard deviation of 1."""
    return rng.standard_normal(n).astype(np.float32)


def band_noise(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """
    Return noise limited to a band, of unit standard deviation, cut from a cached loop at a random place.

    A loop of 6 seconds is made once for each band by the spectrum; a longer stretch wraps round it.

    :param rng: Picks the place in the loop.
    :param n: The number of samples.
    :param low: The lower edge of the band in Hz.
    :param high: The upper edge of the band in Hz.
    """
    key = (float(low), float(high))
    loop = _LOOPS.get(key)
    if loop is None:
        with _LOOPS_LOCK:
            loop = _LOOPS.get(key)
            if loop is None:
                loop = _band_noise(
                    np.random.default_rng(int(low * NOISE_SEED_LOW_FACTOR + high)),
                    NOISE_LOOP_SECONDS * SR,
                    low,
                    high,
                )
                loop.flags.writeable = False
                _LOOPS[key] = loop
    start = int(rng.integers(0, len(loop)))
    if n <= len(loop) - start:
        return loop[start : start + n]
    return np.take(loop, np.arange(start, start + n), mode="wrap")


def rms(x: np.ndarray) -> float:
    """Return the root mean square."""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def db(x: float) -> float:
    """Return a gain as decibels."""
    return AMPLITUDE_DB_DIVISOR * math.log10(max(x, SILENCE_FLOOR))


class Shimmer:
    """
    A granular octave up: overlapping grains of 60 ms read at twice the speed, as the shimmer of a reverb.

    It keeps the end of the last block so that blocks of one second join without a seam. Hann windows at half
    a grain overlap to a constant level, so a steady tone comes out steady and an octave higher.
    """

    GRAIN = round(SHIMMER_GRAIN_SECONDS * SR)

    def __init__(self) -> None:
        """Make an empty shimmer."""
        self._tail = np.zeros((2 * self.GRAIN, 2), dtype=np.float32)
        self._window = np.hanning(self.GRAIN + 1)[:-1].astype(np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Return a block (n, 2), n a whole number of grains, shifted up an octave and delayed by one grain.

        :param block: Stereo samples; their length is cut to a multiple of half a grain.
        """
        grain = self.GRAIN
        hop = grain // 2
        count = len(block) // hop
        buffer = np.concatenate([self._tail, block])
        out = np.zeros((count * hop + hop, 2), dtype=np.float32)
        base = 2 * grain - grain
        for k in range(count):
            start = base + k * hop - grain // 2
            source = buffer[start : start + 2 * grain : 2]
            out[k * hop : k * hop + grain] += source * self._window[:, None]
        self._tail = buffer[-2 * grain :].copy()
        return out[: len(block)]


def _harmonics(kind: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return partial numbers and amplitudes of a shape, with the Lanczos factor that tames the ripple."""
    k = np.arange(1, count + 1, dtype=np.float64)
    if kind == SAW:
        amp = 1.0 / k
    elif kind == SQUARE:
        amp = np.where(k % 2 == 1, 1.0 / k, 0.0)
    elif kind == TRIANGLE:
        amp = np.where(k % 2 == 1, 1.0 / k**2 * np.where(((k - 1) // 2) % 2 == 0, 1.0, -1.0), 0.0)
    else:
        amp = np.where(k == 1, 1.0, 0.0)
    sigma = np.sinc(k / (count + 1.0)) ** 0.5
    return k, amp * sigma


def _band_noise(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """Make noise limited to a band by the spectrum (no filter state)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.where((freqs >= low) & (freqs <= high), 1.0, 0.0)
    edge = np.hanning(max(n // NOISE_EDGE_DIVISOR, NOISE_EDGE_MIN_SAMPLES))
    shape = np.convolve(shape, edge / edge.sum(), mode="same")
    out = np.fft.irfft(spec * shape, n).astype(np.float32)
    return out / (float(out.std()) or 1.0)
