"""
Hard half-time drums for Cinematic Glitch: a layered kick, a big snare, tight hats and ticks.

Nothing here is a noise bed. The kick is a clean sine body with a short pitch drop and a tight pitched transient
(two damped sines, no noise), saturated at four times the rate. The snare is a synthetic body (two sines near 200
and 330 Hz) with a short, band-limited noise burst and the bundled snare sample under it, saturated, then gated by
its own envelope. Hats and shaker ticks are short noise bursts, band-limited above and below, each over in
a few tens of milliseconds. Hits are made once and cached with the notes of the voice library, with the tone of
their bus (a low-pass at 6.5, 9 and 12.5 kHz for the kick, the snares and the hats) already in them.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import Convolution, Pedalboard  # type: ignore[attr-defined]

from . import dsp
from .drums import render_hit
from .dsp import SR, ladder
from .glitch import oversampled
from .voices import NOTES

if TYPE_CHECKING:
    from collections.abc import Callable

# the hits of the style: the key of a note of the score is `glitch:<name>`
HIT_NAMES = ("kick", "kick+", "snare", "ghost", "hat", "open", "tick", "rim")
# the velocities that hits are made at: a step of a twentieth
VELOCITY_STEP = 0.05


def _sat(x: np.ndarray, drive: float) -> np.ndarray:
    """Soft clip at four times the rate: the harmonics that it makes above the band are removed, not folded."""
    return oversampled(lambda y: np.tanh(drive * y) / math.tanh(drive), x, 4)


def _finish(x: np.ndarray, peak: float, fade: float = 0.02, tone: float = 0.0) -> np.ndarray:
    """Fade the end, bring the peak to `peak`, and pass the tone of the bus (a low-pass at `tone` Hz) if there is one."""
    x = x.astype(np.float32)
    f = int(fade * SR)
    x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    x *= np.float32(peak / max(float(np.abs(x).max()), 1e-6))
    if tone > 0.0:
        x = ladder(x, tone, 0.0, 1.0)
    return dsp.to_stereo(x)


def _kick(vel: float, kind: str) -> np.ndarray:
    """Make a kick: a sine body with a pitch drop of a few tens of ms, a pitched transient, saturated."""
    tight = kind == "tight"
    n = int((0.34 if tight else 0.62) * SR)
    t = np.arange(n) / SR
    end, start, tau = (56.0, 215.0, 0.017) if tight else (49.0, 185.0, 0.026)
    freq = end + (start - end) * np.exp(-t / tau)
    body = np.sin(2.0 * np.pi * np.cumsum(freq) / SR)
    rise = 1.0 - np.exp(-t / 0.0012)
    amp = rise * np.exp(-t / (0.085 if tight else 0.16))
    tail = 0.0 if tight else 0.38 * np.sin(2.0 * np.pi * end * t) * np.exp(-t / 0.30) * rise
    snap = (
        np.sin(2.0 * np.pi * 1850.0 * t) * np.exp(-t / 0.0012)
        + 0.8 * np.sin(2.0 * np.pi * 720.0 * t) * np.exp(-t / 0.004)
    ) * (1.0 - np.exp(-t / 0.0002))
    x = body * amp + tail + (0.16 + 0.12 * vel) * snap
    x = _sat(x * (0.7 + 0.5 * vel), 1.5 + 1.2 * vel)
    return _finish(x, 0.95 * (0.55 + 0.45 * vel), tone=6500.0)


def _snare(vel: float, kind: str) -> np.ndarray:
    """Make a snare: body sines, a short band-limited noise burst, the bundled sample, saturated and gated."""
    ghost = kind == "ghost"
    n = int((0.16 if ghost else 0.42) * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7 + int(vel * 10))
    drop = 1.0 + 0.22 * np.exp(-t / 0.014)
    body = 0.9 * np.sin(2.0 * np.pi * np.cumsum(200.0 * drop) / SR) + 0.45 * np.sin(
        2.0 * np.pi * np.cumsum(332.0 * (1.0 + 0.12 * np.exp(-t / 0.012))) / SR
    )
    body *= (1.0 - np.exp(-t / 0.0008)) * np.exp(-t / (0.035 if ghost else 0.09))
    burst = rng.standard_normal(n).astype(np.float32)
    burst = ladder(ladder(burst, 1500.0, 0.0, 1.0, mode="hp24"), 5200.0, 0.0, 1.0)
    burst = burst * ((1.0 - np.exp(-t / 0.0005)) * np.exp(-t / (0.02 if ghost else 0.05))).astype(
        np.float32
    )
    x = body + (0.75 if ghost else 0.95) * burst * 2.2
    if not ghost:
        shot = render_hit("snare_lofi", 0.9).mean(axis=1)[:n]
        shot = ladder(np.pad(shot, (0, n - len(shot))).astype(np.float32), 6500.0, 0.0, 1.0)
        x = x + 0.4 * shot / max(float(np.abs(shot).max()), 1e-6)
    gate = np.where(
        t < (0.12 if ghost else 0.3), 1.0, np.exp(-(t - (0.12 if ghost else 0.3)) / 0.02)
    )
    x = _sat(x * gate * (0.6 + 0.6 * vel), 1.8)
    return _finish(x, 0.9 * (0.45 + 0.55 * vel), tone=9000.0)


HAT_SPECS = {
    "closed": (0.0004, 0.011, 6200.0, 11500.0, 0.09),
    "open": (0.0006, 0.055, 6000.0, 11000.0, 0.45),
    "shaker": (0.006, 0.03, 3000.0, 7500.0, 0.2),
}


def _hat(vel: float, kind: str) -> np.ndarray:
    """Make a hat or a tick: closed, open, or shaker (a softer, lower, slower burst)."""
    attack, decay, low, high, length = HAT_SPECS[kind]
    n = int(length * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(31 + int(vel * 10) + len(kind))
    x = rng.standard_normal(n).astype(np.float32)
    x = ladder(ladder(x, low, 0.0, 1.0, mode="hp24"), high, 0.0, 1.0)
    x = x * ((1.0 - np.exp(-t / attack)) * np.exp(-t / decay)).astype(np.float32)
    return _finish(x, 0.5 * (0.3 + 0.7 * vel), 0.01, tone=12500.0)


def _rim(vel: float) -> np.ndarray:
    """Make the bundled cross-stick, scaled, with the tone of the hats."""
    wave = render_hit("rim", vel) * np.float32(0.8)
    return np.stack([ladder(wave[:, k], 12500.0, 0.0, 1.0) for k in range(2)], axis=1)


# the gain with which each hit goes into its bus: the open hats and ticks sit under the closed ones, the rim over them
_MAKERS: dict[str, tuple[Callable[[float], np.ndarray], float]] = {
    "kick": (lambda v: _kick(v, "main" if v >= 0.85 else "tight"), 1.0),
    "kick+": (lambda v: _kick(v, "main"), 1.1),
    "snare": (lambda v: _snare(v, "big"), 1.0),
    "ghost": (lambda v: _snare(v, "ghost"), 0.55),
    "hat": (lambda v: _hat(v, "closed"), 1.0),
    "open": (lambda v: _hat(v, "open"), 0.8),
    "tick": (lambda v: _hat(v, "shaker"), 0.9),
    "rim": (_rim, 1.3),
}


def hit(name: str, vel: float, cache_mb: float = 128.0) -> np.ndarray:
    """
    Return one hit of the kit as stereo float32, from the cache when it was made.

    :param name: One of `HIT_NAMES`.
    :param vel: 0 to 1; hits are made on a grid of a twentieth.
    :param cache_mb: The size of the cache of notes that may be used.
    """
    vel = round(float(np.clip(vel, 0.05, 1.0)) / VELOCITY_STEP) * VELOCITY_STEP
    key = ("glitch-hit", name, round(vel, 2))
    found = NOTES.get(key)
    if found is not None:
        return found
    make, gain = _MAKERS[name]
    wave = make(vel) * np.float32(gain)
    NOTES.put(key, wave, cache_mb)
    return wave


def plate_ir(seconds: float = 0.3) -> np.ndarray:
    """Return a short, dark, gated plate-like room: a decaying burst of band-limited noise that is cut off at the end."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n), np.float32)
    for side in range(2):
        rng = np.random.default_rng(100 + side)
        x = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.07).astype(np.float32)
        x = ladder(ladder(x, 5200.0, 0.0, 1.0), 220.0, 0.0, 1.0, mode="hp24")
        x[-int(0.04 * SR) :] *= np.linspace(1.0, 0.0, int(0.04 * SR), dtype=np.float32)
        out[side] = x / math.sqrt(float(np.sum(x.astype(np.float64) ** 2)))
    return out


def plate_board() -> Pedalboard:
    """Return the short plate as a board whose wet signal alone comes out."""
    return Pedalboard([Convolution(plate_ir(), mix=1.0, sample_rate=SR)])


def duck_gain(  # noqa: PLR0913, PLR0917
    kicks: list[tuple[float, float]],
    start: float,
    n: int,
    depth: float,
    tau: float,
    ramp: float = 0.002,
) -> np.ndarray:
    """
    Return the gain of `n` samples that begin at second `start`, which dips at every (second, strength) of `kicks`.

    The gain recovers with time constant `tau`. The dip begins `ramp` seconds before the hit with a smooth fall, so it
    never clicks.

    :param kicks: The kicks of the piece as (second, strength), sorted.
    :param start: The second of the first sample.
    :param n: The number of samples.
    :param depth: The share of the level that the strongest kick takes away.
    :param tau: The time constant of the recovery in seconds.
    :param ramp: The length of the fall in seconds.
    """
    gain = np.ones(n, np.float32)
    r = int(ramp * SR)
    reach = tau * 6.0
    first = start - reach - ramp
    for t, strength in kicks:
        if t < first:
            continue
        if t - ramp > start + n / SR:
            break
        shape = _duck_shape(tau, ramp)
        a = round((t - start) * SR) - r
        lo, hi = max(a, 0), min(a + len(shape), n)
        if hi <= lo:
            continue
        dip = 1.0 - depth * strength * shape[lo - a : hi - a]
        gain[lo:hi] = np.minimum(gain[lo:hi], dip)
    return gain


@lru_cache(maxsize=16)
def _duck_shape(tau: float, ramp: float) -> np.ndarray:
    """Return the fall and the recovery of one dip, as float32: a smooth fall over `ramp`, then an exponential."""
    r = int(ramp * SR)
    k = np.arange(-r, int(tau * 6.0 * SR))
    env = np.exp(-np.maximum(k, 0) / (tau * SR))
    env[:r] = np.sin(np.linspace(0.0, np.pi / 2.0, r, endpoint=False)) ** 2
    shape: np.ndarray = env.astype(np.float32)
    shape.flags.writeable = False
    return shape
