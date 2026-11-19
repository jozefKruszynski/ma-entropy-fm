"""
The percussion of the soundscapes: a soft kick, a dusty snare, hats, a shaker, clicks and a rim, and the taiko of Ritual.

Kicks, hats and shakers are made in numpy, so they cost nothing to ship; the snare, the cross stick and
the clicks are the CC0 one-shots of the VCSL (see `samples/LICENSES.md`), through the sampler. The
kick's body falls from 130 Hz to its pitch in a few tens of milliseconds: that is the thump of the
drum, a fall below 150 Hz, and nothing sustained glides. Hits are cached like the notes of `voices`.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import numpy as np

from . import dsp, taiko
from .dsp import SR
from .timeline import stable_hash
from .voices import NOTES, REFERENCE_RMS, sampler

HITS = (
    "kick_soft",
    "kick_deep",
    "kick_lofi",
    "snare_lofi",
    "rim",
    "hat_closed",
    "hat_open",
    "shaker",
    "click",
    "tom_soft",
    "odaiko",
    "shime",
    "taiko_rim",
    "heart",
)
# a hit is this loud when its velocity is 1: the RMS over its own length, relative to a voice's 0.09
LEVELS = {
    "kick_soft": 1.0,
    "kick_deep": 1.1,
    "kick_lofi": 1.0,
    "snare_lofi": 0.8,
    "rim": 0.5,
    "hat_closed": 0.22,
    "hat_open": 0.2,
    "shaker": 0.2,
    "click": 0.6,
    "tom_soft": 0.8,
    "odaiko": 1.0,
    "shime": 0.8,
    "taiko_rim": 0.5,
    "heart": 0.9,
}


def _kick(  # noqa: PLR0913, PLR0917
    rng: np.random.Generator, vel: float, pitch: float, start: float, tau: float, decay: float
) -> np.ndarray:
    n = int(decay * 5.0 * SR)
    t = np.arange(n) / SR
    freq = pitch + (start - pitch) * np.exp(-t / tau)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / decay)
    click = rng.standard_normal(n) * np.exp(-t / 0.0018) * 0.06 * vel
    out = (np.tanh(1.5 * body * (0.6 + 0.6 * vel)) / 1.5 + click).astype(np.float32)
    out[: int(0.0015 * SR)] *= np.linspace(0.0, 1.0, int(0.0015 * SR))
    return out


def _hat(rng: np.random.Generator, vel: float, decay: float, low: float) -> np.ndarray:
    n = int(decay * 6.0 * SR)
    t = np.arange(n) / SR
    wave = dsp.band_noise(rng, n, low, 16000.0) * np.exp(-t / decay)
    wave[: int(0.0008 * SR)] *= np.linspace(0.0, 1.0, int(0.0008 * SR))
    return (wave * (0.5 + 0.5 * vel)).astype(np.float32)


def _make(name: str, vel: float, patch: int) -> np.ndarray:  # noqa: C901
    rng = np.random.default_rng(stable_hash(name, round(vel, 2), patch) % (2**63))
    if name == "kick_soft":
        out = _kick(rng, vel, 50.0 - 3.0 * (patch % 2), 125.0, 0.022, 0.20 + 0.03 * (patch % 2))
    elif name == "kick_deep":
        out = _kick(rng, vel, 44.0 + 2.0 * (patch % 2), 110.0, 0.03, 0.34)
    elif name == "kick_lofi":
        shot = sampler("drums").render(
            0.0,
            vel,
            None,
            name=("bassdrum_hit_mp1", "bassdrum_hit_pp1", "bassdrum_hit_mf1")[patch % 3],
        )
        body = _kick(rng, vel, 52.0, 90.0, 0.03, 0.16)
        out = np.zeros(max(len(shot), len(body)), dtype=np.float32)
        out[: len(shot)] += shot * 0.9
        out[: len(body)] += body * 0.35
    elif name == "snare_lofi":
        shot = sampler("drums").render(
            0.0,
            vel,
            None,
            name=(
                "Snare3M_HitSN_v3_rr1_Mid",
                "Snare3M_HitSN_v2_rr1_Mid",
                "Snare3M_HitSN_v4_rr1_Mid",
            )[patch % 3],
        )
        out = np.tanh(1.6 * shot) / 1.6
    elif name == "rim":
        out = sampler("drums").render(
            0.0,
            vel,
            None,
            name=("Snare3M_Xstick_v2_rr1_Mid", "Snare3M_stick_v3_rr1_Mid")[patch % 2],
        )
    elif name == "hat_closed":
        out = _hat(rng, vel, 0.012 + 0.004 * (patch % 3), 6500.0)
    elif name == "hat_open":
        out = _hat(rng, vel, 0.07, 6000.0)
    elif name == "shaker":
        n = int(0.16 * SR)
        t = np.arange(n) / SR
        # patch 1 is the soft shaker of the Berlin School: a lower, narrower band and a gentler attack
        soft = patch == 1
        env = np.minimum(t / (0.016 if soft else 0.012), 1.0) * np.exp(
            -t / (0.04 if soft else 0.03)
        )
        band = (2800.0, 6000.0) if soft else (4500.0, 9500.0)
        out = (dsp.band_noise(rng, n, *band) * env * (0.5 + 0.5 * vel)).astype(np.float32)
    elif name == "click":
        names = ("Claves1_Hit_v1_rr1_Mid", "Claves1_Hit_v2_rr1_Mid", "Claves2_Hit_v1_rr1_Mid")
        out = sampler("drums").render(
            0.0, vel, None, name=names[patch % 3], semitones=float((patch // 3) * 2 - 2)
        )
    elif name in {"odaiko", "shime", "heart"}:
        out = taiko.strike(name, vel, patch)
    elif name == "taiko_rim":
        out = taiko.strike("rim", vel, patch)
    else:
        out = _kick(rng, vel, 96.0 + 12.0 * (patch % 3), 140.0, 0.03, 0.22)
    return np.asarray(out, dtype=np.float32)


_GAINS: dict[str, float] = {}


def render_hit(name: str, vel: float = 0.8, patch: int = 0) -> np.ndarray:
    """
    Return one hit as a stereo float array, from the cache when it was made.

    :param name: One of `HITS`.
    :param vel: 0 to 1.
    :param patch: The variant of the day (the kick's pitch, which snare, which click).
    """
    vel = round(float(np.clip(vel, 0.05, 1.0)) * 8.0) / 8.0
    key = ("hit", name, vel, patch)
    found = NOTES.get(key)
    if found is not None:
        return found
    if name not in _GAINS:
        ref = _make(name, 1.0, 0)
        _GAINS[name] = (
            REFERENCE_RMS
            * LEVELS[name]
            / max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), 1e-6)
            * 1.0
        )
        # a short hit has a low RMS over its own length; the gain is limited so that its peak stays sane
        _GAINS[name] = min(_GAINS[name], 0.95 / max(float(np.abs(ref).max()), 1e-6))
    wave = _make(name, vel, patch) * np.float32(_GAINS[name] * (0.3 + 0.7 * vel))
    stereo = dsp.to_stereo(wave)
    NOTES.put(key, stereo, 48.0)
    return stereo
