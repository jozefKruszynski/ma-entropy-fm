"""
Synthesised taiko: a deep odaiko, a smaller shime-daiko, a quiet wooden rim click and a soft heartbeat drum, no samples.

A strike is a handful of membrane modes (the ratios of an ideal circular membrane, 1, 1.59, 2.14, 2.30, 2.65)
that share one pitch drop: the skin is stretched by the stroke, so the pitch starts up to half as high again
and settles within a few tens of milliseconds. Under it sits a short body thump, a sine that sweeps down, and
a soft skin attack made of two quickly damped low sines rather than noise, so there is no hiss anywhere. The
phase of the drop is computed in closed form, so the strike is a few vectorised float32 sines.

Rolls accelerate: the gaps between strikes shrink and the strikes grow louder into a final strike.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .synth import SAMPLE_RATE, TWO_PI

RING_DB = 55.0
# a decay is timed by the seconds that it takes to fall 60 dB, and the decay exponent is that fall as a natural log
T60_DB = 60.0
T60_DECAY_EXPONENT = 6.9078
MEMBRANE = np.array([1.0, 1.59, 2.14, 2.30, 2.65, 2.92])
KIND_RIM = "rim"
# the rim click: its length, its partials (Hz, amplitude, decay), the range of its detune, the rise and the gain
RIM_SECONDS = 0.12
RIM_PARTIALS = ((830, 1.0, 0.012), (1330, 0.6, 0.009), (2150, 0.3, 0.006))
RIM_DETUNE_RANGE = (0.95, 1.05)
RIM_ATTACK_TAU = 0.0004
RIM_GAIN = 0.25
# a drum is tuned within this range of its pitch, from one strike to the next
PITCH_SPREAD_RANGE = (0.97, 1.03)
# the ring is this much longer for a hard stroke: from this at velocity 0, and this more at velocity 1
RING_VELOCITY_FLOOR = 0.8
RING_VELOCITY_RANGE = 0.3
# the drop of the pitch and the level of a mode, from this at velocity 0 and this much more at velocity 1
DROP_VELOCITY_FLOOR = 0.5
DROP_VELOCITY_RANGE = 0.5
MODE_VELOCITY_FLOOR = 0.6
MODE_VELOCITY_RANGE = 0.4
# each mode rings shorter than the one below it by this much more of the fundamental's time
MODE_DAMPING_STEP = 0.9
# the phases of the modes are drawn from this many radians (a little short of a whole turn)
MODE_PHASE_RADIANS = 6.28
DEFAULT_RAMP_SECONDS = 0.001
STRIKE_FADE_SECONDS = 0.04
STRIKE_GAIN = 0.4
# the body thump: its length, level, decay, the pitch share of its sweep, and the depth and time of the sweep
THUMP_SECONDS = 0.5
THUMP_GAIN = 0.8
THUMP_TAU = 0.11
THUMP_PITCH = 0.75
THUMP_SWEEP_DEPTH = 0.6
THUMP_SWEEP_TAU = 0.05
# the weight of the stroke: a damped low sine, its length, level, decay and pitch
WEIGHT_SECONDS = 0.04
WEIGHT_GAIN = 0.55
WEIGHT_TAU = 0.011
WEIGHT_HZ = 88.0
# the soft skin attack: its length, the decay, and its two damped sines (a multiple of the pitch, a level)
KNOCK_SECONDS = 0.06
KNOCK_TAU = 0.012
KNOCK_PARTIALS = ((3.1, 0.25), (5.3, 0.12))
# a roll: each gap is this many times the one after it (the gaps shrink), and the strikes grow by this power
ROLL_GAP_SHRINK = 0.82
ROLL_VELOCITY_CURVE = 1.5
# the seed of the strike of an event of a render: this many to a render, plus the index of the event
RENDER_SEED_STRIDE = 1000

# per kind: fundamental Hz, pitch drop (fraction above the final pitch), drop time constant, mode amplitudes, fundamental t60
KINDS: dict[str, dict[str, Any]] = {
    "odaiko": {
        "f0": 54.0,
        "drop": 0.45,
        "tau": 0.035,
        "amps": (1.0, 0.45, 0.3, 0.22, 0.12, 0.08),
        "t60": 2.2,
        "thump": 1.0,
    },
    "shime": {
        "f0": 190.0,
        "drop": 0.12,
        "tau": 0.02,
        "amps": (1.0, 0.6, 0.4, 0.3, 0.2, 0.12),
        "t60": 0.45,
        "thump": 0.0,
    },
    # a low tom played with a soft mallet: a slow attack, almost no skin knock, a short warm decay, so it never clicks
    "heart": {
        "f0": 62.0,
        "drop": 0.3,
        "tau": 0.045,
        "amps": (1.0, 0.28, 0.12, 0.06, 0.03, 0.02),
        "t60": 0.8,
        "thump": 0.7,
        "knock": 0.15,
        "ramp": 0.012,
        "weight": 0.35,
    },
}


def strike(
    kind: str = "odaiko", velocity: float = 0.8, seed: int = 0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Return one strike as mono float32."""
    rng = np.random.default_rng(seed)
    if kind == KIND_RIM:
        n = int(RIM_SECONDS * sr)
        t = np.arange(n, dtype=np.float32) / sr
        out = np.zeros(n, np.float32)
        base = rng.uniform(*RIM_DETUNE_RANGE)
        for fr, a, d in RIM_PARTIALS:
            out += (a * np.exp(-t / d) * np.sin(TWO_PI * fr * base * t)).astype(np.float32)
        out *= (1.0 - np.exp(-t / RIM_ATTACK_TAU)).astype(np.float32)
        return out * np.float32(RIM_GAIN * velocity)
    spec = KINDS[kind]
    f0 = spec["f0"] * rng.uniform(*PITCH_SPREAD_RANGE)
    t60 = spec["t60"] * (RING_VELOCITY_FLOOR + RING_VELOCITY_RANGE * velocity)
    n = int(t60 * RING_DB / T60_DB * sr)
    t = np.arange(n, dtype=np.float64) / sr
    drop = spec["drop"] * (DROP_VELOCITY_FLOOR + DROP_VELOCITY_RANGE * velocity)
    tau = spec["tau"]
    base = (t + drop * tau * (1.0 - np.exp(-t / tau))) * TWO_PI * f0  # radians of the fundamental
    out = np.zeros(n, np.float32)
    for i, (ratio, a) in enumerate(zip(MEMBRANE, spec["amps"], strict=True)):
        mode_t60 = t60 / (1.0 + MODE_DAMPING_STEP * i)
        end = min(n, int(mode_t60 * RING_DB / T60_DB * sr))
        # higher modes lose their drop faster, which keeps the stroke from sounding like a laser
        ph = (base[:end] * ratio).astype(np.float32) + np.float32(rng.random() * MODE_PHASE_RADIANS)
        env = np.exp(t[:end].astype(np.float32) * np.float32(-T60_DECAY_EXPONENT / mode_t60))
        out[:end] += (
            np.float32(a * (MODE_VELOCITY_FLOOR + MODE_VELOCITY_RANGE * velocity))
            * env
            * np.sin(ph)
        )
    _body(spec, velocity, f0, t.astype(np.float32), out)
    ramp = spec.get("ramp", DEFAULT_RAMP_SECONDS)
    out *= (1.0 - np.exp(-t.astype(np.float32) / ramp)).astype(np.float32)
    fade = min(n, int(STRIKE_FADE_SECONDS * sr))
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out * np.float32(STRIKE_GAIN)


@dataclass(frozen=True)
class TaikoEvent:
    """One strike: when, which drum, how hard, where."""

    time: float
    kind: str = "odaiko"
    velocity: float = 0.8
    pan: float = 0.0


def roll(
    start: float,
    end: float,
    kind: str = "odaiko",
    v0: float = 0.25,
    v1: float = 0.9,
    first_gap: float = 0.55,
    last_gap: float = 0.1,
) -> list[TaikoEvent]:
    """
    Return a slow accelerating roll from `start` that arrives with a final strike at `end`.

    The gaps shrink geometrically from `first_gap` to `last_gap` and the strikes grow from v0 to v1.

    :param start: When the roll begins, in seconds.
    :param end: When the final strike falls, in seconds.
    :param kind: The drum, a key of `KINDS`.
    :param v0: The velocity of the first strike.
    :param v1: The velocity of the final strike.
    :param first_gap: The widest gap, in seconds.
    :param last_gap: The gap before the final strike, in seconds.
    """
    times = [end]
    gap = last_gap
    ratio = 1.0 / ROLL_GAP_SHRINK
    while times[-1] - gap > start:
        times.append(times[-1] - gap)
        gap = min(gap * ratio, first_gap)
    times.reverse()
    return [
        TaikoEvent(
            t, kind, v0 + (v1 - v0) * (i / max(len(times) - 1, 1)) ** ROLL_VELOCITY_CURVE, 0.0
        )
        for i, t in enumerate(times)
    ]


def render(
    events: list[TaikoEvent], duration: float, seed: int = 0, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Return the events as stereo float32 of shape (n, 2), `duration` seconds long plus the ringing tail."""
    notes = []
    end = int(duration * sr)
    for i, ev in enumerate(events):
        x = strike(ev.kind, ev.velocity, seed * RENDER_SEED_STRIDE + i, sr)
        start = int(ev.time * sr)
        end = max(end, start + len(x))
        notes.append((start, x, ev.pan))
    mix = np.zeros((end, 2), np.float32)
    for start, x, pan in notes:
        ang = (np.clip(pan, -1.0, 1.0) + 1.0) * math.pi / 4.0
        mix[start : start + len(x), 0] += x * math.cos(ang)
        mix[start : start + len(x), 1] += x * math.sin(ang)
    return mix


def _body(
    kind: dict[str, Any], velocity: float, f0: float, t32: np.ndarray, out: np.ndarray
) -> None:
    """Add the thump and the soft skin attack."""
    if kind["thump"] > 0.0:
        m = min(len(t32), int(THUMP_SECONDS * SAMPLE_RATE))
        t = t32[:m]
        out[:m] += (
            THUMP_GAIN
            * kind["thump"]
            * velocity
            * np.exp(-t / THUMP_TAU)
            * np.sin(
                TWO_PI
                * (
                    THUMP_PITCH
                    * f0
                    * (t + THUMP_SWEEP_DEPTH * THUMP_SWEEP_TAU * (1 - np.exp(-t / THUMP_SWEEP_TAU)))
                )
            )
        ).astype(np.float32)
    if kind["thump"] > 0.0:
        # weight in the first 20 ms: a damped low sine that lands with the stroke, still noise-free
        m = min(len(t32), int(WEIGHT_SECONDS * SAMPLE_RATE))
        t = t32[:m]
        out[:m] += (
            kind.get("weight", 1.0)
            * WEIGHT_GAIN
            * velocity
            * np.exp(-t / WEIGHT_TAU)
            * np.sin(TWO_PI * WEIGHT_HZ * t)
        ).astype(np.float32)
    # soft attack: two damped low sines, a muffled thud
    m = min(len(t32), int(KNOCK_SECONDS * SAMPLE_RATE))
    t = t32[:m]
    for ratio, a in KNOCK_PARTIALS:
        fr = ratio * f0
        out[:m] += (
            kind.get("knock", 1.0)
            * a
            * velocity**2
            * np.exp(-t / KNOCK_TAU)
            * np.sin(TWO_PI * fr * t)
        ).astype(np.float32)
