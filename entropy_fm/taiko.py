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

SR = 44100
RING_DB = 55.0
TWO_PI = 2.0 * math.pi
MEMBRANE = np.array([1.0, 1.59, 2.14, 2.30, 2.65, 2.92])

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


def _body(
    kind: dict[str, Any], velocity: float, f0: float, t32: np.ndarray, out: np.ndarray
) -> None:
    """Add the thump and the soft skin attack."""
    if kind["thump"] > 0.0:
        m = min(len(t32), int(0.5 * SR))
        t = t32[:m]
        out[:m] += (
            0.8
            * kind["thump"]
            * velocity
            * np.exp(-t / 0.11)
            * np.sin(TWO_PI * (0.75 * f0 * (t + 0.6 * 0.05 * (1 - np.exp(-t / 0.05)))))
        ).astype(np.float32)
    if kind["thump"] > 0.0:
        # weight in the first 20 ms: a damped low sine that lands with the stroke, still noise-free
        m = min(len(t32), int(0.04 * SR))
        t = t32[:m]
        out[:m] += (
            kind.get("weight", 1.0)
            * 0.55
            * velocity
            * np.exp(-t / 0.011)
            * np.sin(TWO_PI * 88.0 * t)
        ).astype(np.float32)
    # soft attack: two damped low sines, a muffled thud
    m = min(len(t32), int(0.06 * SR))
    t = t32[:m]
    for fr, a in ((3.1 * f0, 0.25), (5.3 * f0, 0.12)):
        out[:m] += (
            kind.get("knock", 1.0) * a * velocity**2 * np.exp(-t / 0.012) * np.sin(TWO_PI * fr * t)
        ).astype(np.float32)


def strike(kind: str = "odaiko", velocity: float = 0.8, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Return one strike as mono float32."""
    rng = np.random.default_rng(seed)
    if kind == "rim":
        n = int(0.12 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        out = np.zeros(n, np.float32)
        base = rng.uniform(0.95, 1.05)
        for fr, a, d in ((830, 1.0, 0.012), (1330, 0.6, 0.009), (2150, 0.3, 0.006)):
            out += (a * np.exp(-t / d) * np.sin(TWO_PI * fr * base * t)).astype(np.float32)
        out *= (1.0 - np.exp(-t / 0.0004)).astype(np.float32)
        return out * np.float32(0.25 * velocity)
    spec = KINDS[kind]
    f0 = spec["f0"] * rng.uniform(0.97, 1.03)
    t60 = spec["t60"] * (0.8 + 0.3 * velocity)
    n = int(t60 * RING_DB / 60.0 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    drop = spec["drop"] * (0.5 + 0.5 * velocity)
    tau = spec["tau"]
    base = (t + drop * tau * (1.0 - np.exp(-t / tau))) * TWO_PI * f0  # radians of the fundamental
    out = np.zeros(n, np.float32)
    for i, (ratio, a) in enumerate(zip(MEMBRANE, spec["amps"], strict=True)):
        mode_t60 = t60 / (1.0 + 0.9 * i)
        end = min(n, int(mode_t60 * RING_DB / 60.0 * sr))
        # higher modes lose their drop faster, which keeps the stroke from sounding like a laser
        ph = (base[:end] * ratio).astype(np.float32) + np.float32(rng.random() * 6.28)
        env = np.exp(t[:end].astype(np.float32) * np.float32(-6.9078 / mode_t60))
        out[:end] += np.float32(a * (0.6 + 0.4 * velocity)) * env * np.sin(ph)
    _body(spec, velocity, f0, t.astype(np.float32), out)
    out *= (1.0 - np.exp(-t.astype(np.float32) / spec.get("ramp", 0.001))).astype(np.float32)
    fade = min(n, int(0.04 * sr))
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out * np.float32(0.4)


@dataclass(frozen=True)
class TaikoEvent:
    """One strike: when, which drum, how hard, where."""

    time: float
    kind: str = "odaiko"
    velocity: float = 0.8
    pan: float = 0.0


def roll(  # noqa: PLR0913, PLR0917
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
    """
    times = [end]
    gap = last_gap
    ratio = 1.0 / 0.82
    while times[-1] - gap > start:
        times.append(times[-1] - gap)
        gap = min(gap * ratio, first_gap)
    times.reverse()
    return [
        TaikoEvent(t, kind, v0 + (v1 - v0) * (i / max(len(times) - 1, 1)) ** 1.5, 0.0)
        for i, t in enumerate(times)
    ]


def render(events: list[TaikoEvent], duration: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Return the events as stereo float32 of shape (n, 2), `duration` seconds long plus the ringing tail."""
    notes = []
    end = int(duration * sr)
    for i, ev in enumerate(events):
        x = strike(ev.kind, ev.velocity, seed * 1000 + i, sr)
        start = int(ev.time * sr)
        end = max(end, start + len(x))
        notes.append((start, x, ev.pan))
    mix = np.zeros((end, 2), np.float32)
    for start, x, pan in notes:
        ang = (np.clip(pan, -1.0, 1.0) + 1.0) * math.pi / 4.0
        mix[start : start + len(x), 0] += x * math.cos(ang)
        mix[start : start + len(x), 1] += x * math.sin(ang)
    return mix
