"""
Hard half-time drums for Cinematic Glitch: a layered kick, a big snare, tight hats and ticks.

Nothing here is a noise bed. The kick is a clean sine body with a short pitch drop and a tight pitched transient
(two damped sines, no noise), saturated at a multiple of the rate. The snare is a synthetic body (two sines near 200
and 330 Hz) with a short, band-limited noise burst and the bundled snare sample under it, saturated, then gated by
its own envelope. Hats and shaker ticks are short noise bursts, band-limited above and below, each over in
a few tens of milliseconds. Hits are made once and cached with the notes of the voice library, with the tone of
their bus (a low-pass at `KICK_TONE_HZ`, `SNARE_TONE_HZ` and `HAT_TONE_HZ` for the kick, the snares and the hats) already in them.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import Convolution, Pedalboard  # type: ignore[attr-defined]

from . import dsp
from .drums import RIM as LOFI_RIM
from .drums import SNARE_LOFI, render_hit
from .dsp import LADDER_HP24, MIN_LEVEL, SR, ladder
from .glitch import OVERSAMPLE_FULL, oversampled
from .voices import KEY_DECIMALS, MIN_VELOCITY, NOTES

if TYPE_CHECKING:
    from collections.abc import Callable

# the hits of the style: the key of a note of the score is `glitch:<name>`
KICK = "kick"
KICK_PLUS = "kick+"
SNARE = "snare"
GHOST = "ghost"
HAT = "hat"
OPEN = "open"
TICK = "tick"
RIM = "rim"
HIT_NAMES = (KICK, KICK_PLUS, SNARE, GHOST, HAT, OPEN, TICK, RIM)
# the kinds of kick, snare and hat that the hits are made of
KICK_MAIN = "main"
KICK_TIGHT = "tight"
SNARE_BIG = "big"
SNARE_GHOST = "ghost"
HAT_CLOSED = "closed"
HAT_OPEN = "open"
HAT_SHAKER = "shaker"
# a kick at this velocity or more is the main kick, a softer one the tight kick
MAIN_KICK_VELOCITY = 0.85
# the velocities that hits are made at: a step of a twentieth
VELOCITY_STEP = 0.05
CACHE_TAG_HIT = "glitch-hit"
DEFAULT_CACHE_MB = 128.0
# the tail of a dip lasts this many time constants
DUCK_TAUS = 6.0
DUCK_SHAPES_CACHED = 16


def hit(name: str, vel: float, cache_mb: float = DEFAULT_CACHE_MB) -> np.ndarray:
    """
    Return one hit of the kit as stereo float32, from the cache when it was made.

    :param name: One of `HIT_NAMES`.
    :param vel: 0 to 1; hits are made on a grid of a twentieth.
    :param cache_mb: The size of the cache of notes that may be used.
    """
    vel = round(float(np.clip(vel, MIN_VELOCITY, 1.0)) / VELOCITY_STEP) * VELOCITY_STEP
    key = (CACHE_TAG_HIT, name, round(vel, KEY_DECIMALS))
    found = NOTES.get(key)
    if found is not None:
        return found
    make, gain = _MAKERS[name]
    wave = make(vel) * np.float32(gain)
    NOTES.put(key, wave, cache_mb)
    return wave


# the seed of the room, the time constant of its decay, the cutoffs of its two filters and the length of the cut at the end
PLATE_SEED = 100
PLATE_DECAY_SECONDS = 0.07
PLATE_LOWPASS_HZ = 5200.0
PLATE_HIGHPASS_HZ = 220.0
PLATE_GATE_SECONDS = 0.04


def plate_ir(seconds: float = 0.3) -> np.ndarray:
    """Return a short, dark, gated plate-like room: a decaying burst of band-limited noise that is cut off at the end."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n), np.float32)
    for side in range(2):
        rng = np.random.default_rng(PLATE_SEED + side)
        x = rng.standard_normal(n).astype(np.float32) * np.exp(-t / PLATE_DECAY_SECONDS).astype(
            np.float32
        )
        x = ladder(
            ladder(x, PLATE_LOWPASS_HZ, 0.0, 1.0),
            PLATE_HIGHPASS_HZ,
            0.0,
            1.0,
            mode=LADDER_HP24,
        )
        x[-int(PLATE_GATE_SECONDS * SR) :] *= np.linspace(
            1.0, 0.0, int(PLATE_GATE_SECONDS * SR), dtype=np.float32
        )
        out[side] = x / math.sqrt(float(np.sum(x.astype(np.float64) ** 2)))
    return out


def plate_board() -> Pedalboard:
    """Return the short plate as a board whose wet signal alone comes out."""
    return Pedalboard([Convolution(plate_ir(), mix=1.0, sample_rate=SR)])


def duck_gain(
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
    reach = tau * DUCK_TAUS
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


def _sat(x: np.ndarray, drive: float) -> np.ndarray:
    """Soft clip at a multiple of the rate (`OVERSAMPLE_FULL`): the harmonics that it makes above the band are removed, not folded."""
    return oversampled(lambda y: np.tanh(drive * y) / math.tanh(drive), x, OVERSAMPLE_FULL)


def _finish(x: np.ndarray, peak: float, fade: float = 0.02, tone: float = 0.0) -> np.ndarray:
    """Fade the end, bring the peak to `peak`, and pass the tone of the bus (a low-pass at `tone` Hz) if there is one."""
    x = x.astype(np.float32)
    f = int(fade * SR)
    x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    x *= np.float32(peak / max(float(np.abs(x).max()), MIN_LEVEL))
    if tone > 0.0:
        x = ladder(x, tone, 0.0, 1.0)
    return dsp.to_stereo(x)


# the length of a kick, in seconds: the tight one and the main one
KICK_TIGHT_SECONDS = 0.34
KICK_MAIN_SECONDS = 0.62
# the pitch drop of a kick: the Hz that it ends on, the Hz that it starts from, and the time constant in seconds
KICK_TIGHT_DROP = (56.0, 215.0, 0.017)
KICK_MAIN_DROP = (49.0, 185.0, 0.026)
KICK_BODY_ATTACK_SECONDS = 0.0012
KICK_TIGHT_DECAY_SECONDS = 0.085
KICK_MAIN_DECAY_SECONDS = 0.16
KICK_TAIL_GAIN = 0.38
KICK_TAIL_DECAY_SECONDS = 0.30
# the two damped sines of the transient: their pitch in Hz, the gain of the low one and the decay of each in seconds
KICK_SNAP_HIGH_HZ = 1850.0
KICK_SNAP_HIGH_DECAY_SECONDS = 0.0012
KICK_SNAP_LOW_GAIN = 0.8
KICK_SNAP_LOW_HZ = 720.0
KICK_SNAP_LOW_DECAY_SECONDS = 0.004
KICK_SNAP_ATTACK_SECONDS = 0.0002
# the level of the transient in the kick at velocity 0, and what each unit of velocity adds
KICK_SNAP_LEVEL_FLOOR = 0.16
KICK_SNAP_LEVEL_PER_VELOCITY = 0.12
# the gain into the saturator and its drive, at velocity 0 and per unit of velocity
KICK_INPUT_FLOOR = 0.7
KICK_INPUT_PER_VELOCITY = 0.5
KICK_DRIVE_FLOOR = 1.5
KICK_DRIVE_PER_VELOCITY = 1.2
# the peak of the kick: its top, the share of it at velocity 0 and the share that velocity adds
KICK_PEAK = 0.95
KICK_PEAK_VELOCITY_FLOOR = 0.55
KICK_PEAK_VELOCITY_RANGE = 0.45
KICK_TONE_HZ = 6500.0


def _kick(vel: float, kind: str) -> np.ndarray:
    """Make a kick: a sine body with a pitch drop of a few tens of ms, a pitched transient, saturated."""
    tight = kind == KICK_TIGHT
    n = int((KICK_TIGHT_SECONDS if tight else KICK_MAIN_SECONDS) * SR)
    t = np.arange(n) / SR
    end, start, tau = KICK_TIGHT_DROP if tight else KICK_MAIN_DROP
    freq = end + (start - end) * np.exp(-t / tau)
    body = np.sin(2.0 * np.pi * np.cumsum(freq) / SR)
    rise = 1.0 - np.exp(-t / KICK_BODY_ATTACK_SECONDS)
    amp = rise * np.exp(-t / (KICK_TIGHT_DECAY_SECONDS if tight else KICK_MAIN_DECAY_SECONDS))
    tail = (
        0.0
        if tight
        else KICK_TAIL_GAIN
        * np.sin(2.0 * np.pi * end * t)
        * np.exp(-t / KICK_TAIL_DECAY_SECONDS)
        * rise
    )
    snap = (
        np.sin(2.0 * np.pi * KICK_SNAP_HIGH_HZ * t) * np.exp(-t / KICK_SNAP_HIGH_DECAY_SECONDS)
        + KICK_SNAP_LOW_GAIN
        * np.sin(2.0 * np.pi * KICK_SNAP_LOW_HZ * t)
        * np.exp(-t / KICK_SNAP_LOW_DECAY_SECONDS)
    ) * (1.0 - np.exp(-t / KICK_SNAP_ATTACK_SECONDS))
    x = body * amp + tail + (KICK_SNAP_LEVEL_FLOOR + KICK_SNAP_LEVEL_PER_VELOCITY * vel) * snap
    x = _sat(
        x * (KICK_INPUT_FLOOR + KICK_INPUT_PER_VELOCITY * vel),
        KICK_DRIVE_FLOOR + KICK_DRIVE_PER_VELOCITY * vel,
    )
    return _finish(
        x,
        KICK_PEAK * (KICK_PEAK_VELOCITY_FLOOR + KICK_PEAK_VELOCITY_RANGE * vel),
        tone=KICK_TONE_HZ,
    )


# the length of a snare, in seconds: the ghost note and the big one
SNARE_GHOST_SECONDS = 0.16
SNARE_BIG_SECONDS = 0.42
# the seed of the noise of a snare, and of a hat, to which the velocity times this is added
SNARE_SEED = 7
SEED_VELOCITY_SCALE = 10
# the two sines of the body: the gain and the Hz of each, and how far above its pitch each starts (a share)
# and how long it takes to settle
SNARE_LOW_DROP_SHARE = 0.22
SNARE_LOW_DROP_SECONDS = 0.014
SNARE_LOW_GAIN = 0.9
SNARE_LOW_HZ = 200.0
SNARE_HIGH_GAIN = 0.45
SNARE_HIGH_HZ = 332.0
SNARE_HIGH_DROP_SHARE = 0.12
SNARE_HIGH_DROP_SECONDS = 0.012
SNARE_BODY_ATTACK_SECONDS = 0.0008
SNARE_GHOST_BODY_DECAY_SECONDS = 0.035
SNARE_BIG_BODY_DECAY_SECONDS = 0.09
# the band of the noise burst, in Hz
SNARE_BURST_HIGHPASS_HZ = 1500.0
SNARE_BURST_LOWPASS_HZ = 5200.0
SNARE_BURST_ATTACK_SECONDS = 0.0005
SNARE_GHOST_BURST_DECAY_SECONDS = 0.02
SNARE_BIG_BURST_DECAY_SECONDS = 0.05
# the gain of the burst of the ghost note and of the big one, and the gain of the burst against the body
SNARE_GHOST_BURST_GAIN = 0.75
SNARE_BIG_BURST_GAIN = 0.95
SNARE_BURST_GAIN = 2.2
# the bundled snare sample under the big snare: its velocity, its low-pass in Hz and its gain
SNARE_SAMPLE_VELOCITY = 0.9
SNARE_SAMPLE_LOWPASS_HZ = 6500.0
SNARE_SAMPLE_GAIN = 0.4
# the gate lets the snare through until this second, then closes with the time constant below
SNARE_GHOST_GATE_SECONDS = 0.12
SNARE_BIG_GATE_SECONDS = 0.3
SNARE_GATE_CLOSE_SECONDS = 0.02
# the gain into the saturator at velocity 0 and per unit of velocity, and its drive
SNARE_INPUT_FLOOR = 0.6
SNARE_INPUT_PER_VELOCITY = 0.6
SNARE_DRIVE = 1.8
SNARE_PEAK = 0.9
SNARE_PEAK_VELOCITY_FLOOR = 0.45
SNARE_PEAK_VELOCITY_RANGE = 0.55
SNARE_TONE_HZ = 9000.0


def _snare(vel: float, kind: str) -> np.ndarray:
    """Make a snare: body sines, a short band-limited noise burst, the bundled sample, saturated and gated."""
    ghost = kind == SNARE_GHOST
    n = int((SNARE_GHOST_SECONDS if ghost else SNARE_BIG_SECONDS) * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(SNARE_SEED + int(vel * SEED_VELOCITY_SCALE))
    drop = 1.0 + SNARE_LOW_DROP_SHARE * np.exp(-t / SNARE_LOW_DROP_SECONDS)
    body = SNARE_LOW_GAIN * np.sin(
        2.0 * np.pi * np.cumsum(SNARE_LOW_HZ * drop) / SR
    ) + SNARE_HIGH_GAIN * np.sin(
        2.0
        * np.pi
        * np.cumsum(
            SNARE_HIGH_HZ * (1.0 + SNARE_HIGH_DROP_SHARE * np.exp(-t / SNARE_HIGH_DROP_SECONDS))
        )
        / SR
    )
    body *= (1.0 - np.exp(-t / SNARE_BODY_ATTACK_SECONDS)) * np.exp(
        -t / (SNARE_GHOST_BODY_DECAY_SECONDS if ghost else SNARE_BIG_BODY_DECAY_SECONDS)
    )
    burst = rng.standard_normal(n).astype(np.float32)
    burst = ladder(
        ladder(burst, SNARE_BURST_HIGHPASS_HZ, 0.0, 1.0, mode=LADDER_HP24),
        SNARE_BURST_LOWPASS_HZ,
        0.0,
        1.0,
    )
    burst = burst * (
        (1.0 - np.exp(-t / SNARE_BURST_ATTACK_SECONDS))
        * np.exp(-t / (SNARE_GHOST_BURST_DECAY_SECONDS if ghost else SNARE_BIG_BURST_DECAY_SECONDS))
    ).astype(np.float32)
    x = (
        body
        + (SNARE_GHOST_BURST_GAIN if ghost else SNARE_BIG_BURST_GAIN) * burst * SNARE_BURST_GAIN
    )
    if not ghost:
        shot = render_hit(SNARE_LOFI, SNARE_SAMPLE_VELOCITY).mean(axis=1)[:n]
        shot = ladder(
            np.pad(shot, (0, n - len(shot))).astype(np.float32), SNARE_SAMPLE_LOWPASS_HZ, 0.0, 1.0
        )
        x = x + SNARE_SAMPLE_GAIN * shot / max(float(np.abs(shot).max()), MIN_LEVEL)
    gate = np.where(
        t < (SNARE_GHOST_GATE_SECONDS if ghost else SNARE_BIG_GATE_SECONDS),
        1.0,
        np.exp(
            -(t - (SNARE_GHOST_GATE_SECONDS if ghost else SNARE_BIG_GATE_SECONDS))
            / SNARE_GATE_CLOSE_SECONDS
        ),
    )
    x = _sat(x * gate * (SNARE_INPUT_FLOOR + SNARE_INPUT_PER_VELOCITY * vel), SNARE_DRIVE)
    return _finish(
        x,
        SNARE_PEAK * (SNARE_PEAK_VELOCITY_FLOOR + SNARE_PEAK_VELOCITY_RANGE * vel),
        tone=SNARE_TONE_HZ,
    )


# (attack and decay in seconds, high-pass and low-pass in Hz, length in seconds)
HAT_SPECS = {
    HAT_CLOSED: (0.0004, 0.011, 6200.0, 11500.0, 0.09),
    HAT_OPEN: (0.0006, 0.055, 6000.0, 11000.0, 0.45),
    HAT_SHAKER: (0.006, 0.03, 3000.0, 7500.0, 0.2),
}


HAT_SEED = 31
# the peak of a hat: the scale, the share at velocity 0 and the share that velocity adds
HAT_PEAK = 0.5
HAT_PEAK_VELOCITY_FLOOR = 0.3
HAT_PEAK_VELOCITY_RANGE = 0.7
HAT_FADE_SECONDS = 0.01
HAT_TONE_HZ = 12500.0


def _hat(vel: float, kind: str) -> np.ndarray:
    """Make a hat or a tick: closed, open, or shaker (a softer, lower, slower burst)."""
    attack, decay, low, high, length = HAT_SPECS[kind]
    n = int(length * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(HAT_SEED + int(vel * SEED_VELOCITY_SCALE) + len(kind))
    x = rng.standard_normal(n).astype(np.float32)
    x = ladder(ladder(x, low, 0.0, 1.0, mode=LADDER_HP24), high, 0.0, 1.0)
    x = x * ((1.0 - np.exp(-t / attack)) * np.exp(-t / decay)).astype(np.float32)
    return _finish(
        x,
        HAT_PEAK * (HAT_PEAK_VELOCITY_FLOOR + HAT_PEAK_VELOCITY_RANGE * vel),
        HAT_FADE_SECONDS,
        tone=HAT_TONE_HZ,
    )


RIM_GAIN = 0.8


def _rim(vel: float) -> np.ndarray:
    """Make the bundled cross-stick, scaled, with the tone of the hats."""
    wave = render_hit(LOFI_RIM, vel) * np.float32(RIM_GAIN)
    return np.stack([ladder(wave[:, k], HAT_TONE_HZ, 0.0, 1.0) for k in range(2)], axis=1)


# the gain with which each hit goes into its bus: the open hats and ticks sit under the closed ones, the rim over them
_MAKERS: dict[str, tuple[Callable[[float], np.ndarray], float]] = {
    KICK: (lambda v: _kick(v, KICK_MAIN if v >= MAIN_KICK_VELOCITY else KICK_TIGHT), 1.0),
    KICK_PLUS: (lambda v: _kick(v, KICK_MAIN), 1.1),
    SNARE: (lambda v: _snare(v, SNARE_BIG), 1.0),
    GHOST: (lambda v: _snare(v, SNARE_GHOST), 0.55),
    HAT: (lambda v: _hat(v, HAT_CLOSED), 1.0),
    OPEN: (lambda v: _hat(v, HAT_OPEN), 0.8),
    TICK: (lambda v: _hat(v, HAT_SHAKER), 0.9),
    RIM: (_rim, 1.3),
}


@lru_cache(maxsize=DUCK_SHAPES_CACHED)
def _duck_shape(tau: float, ramp: float) -> np.ndarray:
    """Return the fall and the recovery of one dip, as float32: a smooth fall over `ramp`, then an exponential."""
    r = int(ramp * SR)
    k = np.arange(-r, int(tau * DUCK_TAUS * SR))
    env = np.exp(-np.maximum(k, 0) / (tau * SR))
    env[:r] = np.sin(np.linspace(0.0, np.pi / 2.0, r, endpoint=False)) ** 2
    shape: np.ndarray = env.astype(np.float32)
    shape.flags.writeable = False
    return shape
