"""
Stretched textures: a held note of a string section or a piano smeared into a slow wash, the way Paul Nasca's Paulstretch does.

Every window of the source keeps the magnitude spectrum of its stretch of the sample and gets new random phases,
and the windows overlap-add into a smooth, endless wash with no grain, no pitch change and no click. Nothing is
added to the source: the high end of the wash is the high end of the sample, rolled off steeply above 2.8 kHz in
the spectrum itself, so there is no noise floor and no hiss. Left and right are stretched with different random
phases, which is all of the width.

A wash costs about twelve milliseconds of CPU per second of its own length, which is a lot for a layer, so
a wash is made once and kept in the cache of notes (`voices.NOTES`): the soundscape draws every wash of a day from a
few recipes (an instrument and a pitch) and plays them again and again.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import numpy as np

from .dsp import SR
from .timeline import stable_hash
from .voices import MAX_NOTE_SECONDS, NOTES, REF_VELOCITY, REFERENCE_RMS, VELOCITY_POWER, sampler

# the voice names of the washes in a score, with the instrument folder and the group of its samples
WASHES: dict[str, tuple[str, str]] = {
    "wash:cello": ("strings", "cellos"),
    "wash:violin": ("strings", "violins"),
    "wash:felt": ("felt_piano", ""),
}
WINDOW = 1 << 16
ROLL_OFF_HZ = 2800.0
# the stretch never goes below this: a wash that is hardly stretched would show its source
MIN_FACTOR = 1.5
ATTACK_SKIP = 0.6


def is_wash(voice: str) -> bool:
    """Return True when a voice of a score is a stretched texture."""
    return voice in WASHES


def paulstretch(x: np.ndarray, factor: float, seed: int = 0, window: int = WINDOW) -> np.ndarray:
    """
    Return a mono signal stretched `factor` times, by overlap-add of Hann windows whose phases are random.

    :param x: The source, at least a window long.
    :param factor: How many times longer the result is, 1 or more.
    :param seed: Seeds the phases.
    :param window: The window in samples (a power of two).
    """
    rng = np.random.default_rng(seed)
    hop_out = window // 4
    hop_in = hop_out / factor
    win = np.hanning(window).astype(np.float32)
    count = max(int((len(x) - window) / hop_in), 1)
    out = np.zeros(int(count * hop_out) + window, np.float32)
    freqs = np.fft.rfftfreq(window, 1.0 / SR)
    shape = (1.0 / (1.0 + (freqs / ROLL_OFF_HZ) ** 8)).astype(np.float32)
    # the magnitude of the lowest bins is never negative, so what is left of them would add up to a constant offset
    shape[freqs < 20.0] = 0.0
    padded = np.concatenate([x, np.zeros(window, np.float32)]).astype(np.float32)
    phases = rng.random((count, len(freqs)), dtype=np.float32)
    for i in range(count):
        a = int(i * hop_in)
        spec = np.abs(np.fft.rfft(padded[a : a + window] * win)) * shape
        spec = spec * np.exp(2j * np.pi * phases[i]).astype(np.complex64)
        out[i * hop_out : i * hop_out + window] += (
            np.fft.irfft(spec, window).astype(np.float32) * win
        )
    return out / np.float32(1.5)


def _fades(n: int, fade_in: float, fade_out: float) -> np.ndarray:
    env = np.ones(n, np.float32)
    a, b = int(fade_in * SR), int(fade_out * SR)
    if a:
        env[:a] = np.sin(np.linspace(0.0, np.pi / 2.0, a)) ** 2
    if b:
        env[-b:] *= np.cos(np.linspace(0.0, np.pi / 2.0, b)) ** 2
    return env


def _make(voice: str, midi: float, dur: float) -> np.ndarray:
    name, group = WASHES[voice]
    n = int(dur * SR)
    source = sampler(name).render(midi, 0.6, None, group=group)[int(ATTACK_SKIP * SR) :]
    # the stretch that makes the wash as long as asked, from as much of the sample as is needed
    keep = min(len(source), int(WINDOW + max(n - WINDOW, 0) / MIN_FACTOR))
    source = source[: max(keep, WINDOW + 1)]
    factor = max((n - WINDOW) / max(len(source) - WINDOW, 1), 1.0)
    seed = stable_hash("wash", voice, round(midi, 2)) % (2**31)
    sides = [paulstretch(source, factor, seed + side) for side in range(2)]
    wave = np.zeros((n, 2), np.float32)
    for side, part in enumerate(sides):
        kept = part[:n]
        wave[: len(kept), side] = kept
    wave *= _fades(n, min(6.0, dur / 3.0), min(7.0, dur / 3.0))[:, None]
    body = wave[n // 4 : 3 * n // 4]
    return wave * np.float32(REFERENCE_RMS / max(float(np.sqrt(np.mean(body**2))), 1e-6))


def render_wash(
    voice: str, midi: float, dur: float, vel: float = 0.6, cache_mb: float = 128.0
) -> np.ndarray:
    """
    Return a stretched texture as stereo float32, as long as `dur`, with long fades in and out; cached.

    :param voice: A key of `WASHES`.
    :param midi: The pitch of the source note (the sample is moved to it by at most six semitones).
    :param dur: The length in seconds, at most `MAX_NOTE_SECONDS`.
    :param vel: 0 to 1, the loudness as for a voice.
    :param cache_mb: The size of the cache of notes that may be used.
    """
    dur = float(round(min(max(dur, 8.0), MAX_NOTE_SECONDS)))
    vel = round(float(np.clip(vel, 0.1, 1.0)) * 8.0) / 8.0
    key = (voice, round(midi, 2), dur, vel)
    found = NOTES.get(key)
    if found is not None:
        return found
    wave = _make(voice, midi, dur) * np.float32((vel / REF_VELOCITY) ** VELOCITY_POWER)
    NOTES.put(key, wave, cache_mb)
    return wave
