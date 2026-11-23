"""
The percussion of the soundscapes: a soft kick, a dusty snare, hats, a shaker, clicks and a rim, and the taiko of Ritual.

Kicks, hats and shakers are made in numpy, so they cost nothing to ship; the snare, the cross stick and
the clicks are the CC0 one-shots of the VCSL (see `samples/LICENSES.md`), through the sampler. The
kick's body falls from 130 Hz to its pitch in a few tens of milliseconds: that is the thump of the
drum, a fall below 150 Hz, and nothing sustained glides. Hits are cached like the notes of `voices`.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import threading

import numpy as np

from . import dsp, taiko
from .dsp import MIN_LEVEL, SR
from .timeline import SEED_MODULUS, stable_hash
from .voices import KEY_DECIMALS, MIN_VELOCITY, NOTES, REFERENCE_RMS, VELOCITY_STEPS, sampler

# the voice of a note that is a hit of this module: `drum:<hit>`
DRUM_PREFIX = "drum:"
KICK_SOFT = "kick_soft"
KICK_DEEP = "kick_deep"
KICK_LOFI = "kick_lofi"
SNARE_LOFI = "snare_lofi"
RIM = "rim"
HAT_CLOSED = "hat_closed"
HAT_OPEN = "hat_open"
SHAKER = "shaker"
CLICK = "click"
TOM_SOFT = "tom_soft"
ODAIKO = "odaiko"
SHIME = "shime"
TAIKO_RIM = "taiko_rim"
HEART = "heart"
HITS = (
    KICK_SOFT,
    KICK_DEEP,
    KICK_LOFI,
    SNARE_LOFI,
    RIM,
    HAT_CLOSED,
    HAT_OPEN,
    SHAKER,
    CLICK,
    TOM_SOFT,
    ODAIKO,
    SHIME,
    TAIKO_RIM,
    HEART,
)
# a hit is this loud when its velocity is 1: the RMS over its own length, relative to a voice's 0.09
HIT_LEVELS = {
    KICK_SOFT: 1.0,
    KICK_DEEP: 1.1,
    KICK_LOFI: 1.0,
    SNARE_LOFI: 0.8,
    RIM: 0.5,
    HAT_CLOSED: 0.22,
    HAT_OPEN: 0.2,
    SHAKER: 0.2,
    CLICK: 0.6,
    TOM_SOFT: 0.8,
    ODAIKO: 1.0,
    SHIME: 0.8,
    TAIKO_RIM: 0.5,
    HEART: 0.9,
}
SAMPLES_DRUMS = "drums"
# the one-shots of the VCSL that the variants of the day choose from
KICK_LOFI_SHOTS = ("bassdrum_hit_mp1", "bassdrum_hit_pp1", "bassdrum_hit_mf1")
SNARE_LOFI_SHOTS = (
    "Snare3M_HitSN_v3_rr1_Mid",
    "Snare3M_HitSN_v2_rr1_Mid",
    "Snare3M_HitSN_v4_rr1_Mid",
)
RIM_SHOTS = ("Snare3M_Xstick_v2_rr1_Mid", "Snare3M_stick_v3_rr1_Mid")
CLICK_SHOTS = ("Claves1_Hit_v1_rr1_Mid", "Claves1_Hit_v2_rr1_Mid", "Claves2_Hit_v1_rr1_Mid")
# the hits are kept in this many megabytes of the cache of notes
HIT_CACHE_MB = 48.0
CACHE_TAG_HIT = "hit"
# the peak that a hit is not louder than, whatever its RMS asks for
PEAK_LIMIT = 0.95
# the gain of a hit is this much at no velocity and rises by this much to full velocity
GAIN_VELOCITY_FLOOR = 0.3
GAIN_VELOCITY_RANGE = 0.7
# the kick ends after this many decays; the click that opens it
KICK_DECAYS = 5.0
KICK_CLICK_TAU = 0.0018
KICK_CLICK_GAIN = 0.06
KICK_DRIVE = 1.5
KICK_ATTACK_SECONDS = 0.0015
# the hat ends after this many decays, is noise up to this frequency and rises over this time
HAT_DECAYS = 6.0
HAT_HIGH_HZ = 16000.0
HAT_ATTACK_SECONDS = 0.0008
_GAINS_LOCK = threading.Lock()
_GAINS: dict[str, float] = {}


def render_hit(name: str, vel: float = 0.8, patch: int = 0) -> np.ndarray:
    """
    Return one hit as a stereo float array, from the cache when it was made.

    :param name: One of `HITS`.
    :param vel: 0 to 1.
    :param patch: The variant of the day (the kick's pitch, which snare, which click).
    """
    vel = round(float(np.clip(vel, MIN_VELOCITY, 1.0)) * VELOCITY_STEPS) / VELOCITY_STEPS
    key = (CACHE_TAG_HIT, name, vel, patch)
    found = NOTES.get(key)
    if found is not None:
        return found
    hit_gain = _GAINS.get(name)
    if hit_gain is None:
        ref = _make(name, 1.0, 0)
        gain = (
            REFERENCE_RMS
            * HIT_LEVELS[name]
            / max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), MIN_LEVEL)
        )
        # a short hit has a low RMS over its own length; the gain is limited so that its peak stays sane
        with _GAINS_LOCK:
            hit_gain = _GAINS.setdefault(
                name, min(gain, PEAK_LIMIT / max(float(np.abs(ref).max()), MIN_LEVEL))
            )
    wave = _make(name, vel, patch) * np.float32(
        hit_gain * (GAIN_VELOCITY_FLOOR + GAIN_VELOCITY_RANGE * vel)
    )
    stereo = dsp.to_stereo(wave)
    NOTES.put(key, stereo, HIT_CACHE_MB)
    return stereo


# the level that goes into the soft clip of a kick at velocity 0, and what velocity adds
KICK_INPUT_FLOOR = 0.6
KICK_INPUT_PER_VELOCITY = 0.6


def _kick(
    rng: np.random.Generator, vel: float, pitch: float, start: float, tau: float, decay: float
) -> np.ndarray:
    n = int(decay * KICK_DECAYS * SR)
    t = np.arange(n) / SR
    freq = pitch + (start - pitch) * np.exp(-t / tau)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / decay)
    click = rng.standard_normal(n) * np.exp(-t / KICK_CLICK_TAU) * KICK_CLICK_GAIN * vel
    out = (
        np.tanh(KICK_DRIVE * body * (KICK_INPUT_FLOOR + KICK_INPUT_PER_VELOCITY * vel)) / KICK_DRIVE
        + click
    ).astype(np.float32)
    attack = int(KICK_ATTACK_SECONDS * SR)
    out[:attack] *= np.linspace(0.0, 1.0, attack)
    return out


def _hat(rng: np.random.Generator, vel: float, decay: float, low: float) -> np.ndarray:
    n = int(decay * HAT_DECAYS * SR)
    t = np.arange(n) / SR
    wave = dsp.band_noise(rng, n, low, HAT_HIGH_HZ) * np.exp(-t / decay)
    attack = int(HAT_ATTACK_SECONDS * SR)
    wave[:attack] *= np.linspace(0.0, 1.0, attack)
    return (wave * (NOISE_HIT_VELOCITY_FLOOR + NOISE_HIT_VELOCITY_RANGE * vel)).astype(np.float32)


# a kick has this many variants, one of which the day chooses
KICK_VARIANTS = 2
# the soft kick: its pitch, how much a variant lowers it, where the pitch starts (Hz), the time constant of its drop,
# its decay and what a variant adds to it (seconds)
SOFT_KICK_PITCH_HZ = 50.0
SOFT_KICK_PITCH_STEP_HZ = 3.0
SOFT_KICK_START_HZ = 125.0
SOFT_KICK_DROP_SECONDS = 0.022
SOFT_KICK_DECAY_SECONDS = 0.20
SOFT_KICK_DECAY_STEP_SECONDS = 0.03
# the deep kick, with the same parts (a variant raises its pitch)
DEEP_KICK_PITCH_HZ = 44.0
DEEP_KICK_PITCH_STEP_HZ = 2.0
DEEP_KICK_START_HZ = 110.0
DEEP_KICK_DROP_SECONDS = 0.03
DEEP_KICK_DECAY_SECONDS = 0.34
# the lo-fi kick: the body that goes under the sample
LOFI_KICK_PITCH_HZ = 52.0
LOFI_KICK_START_HZ = 90.0
LOFI_KICK_DROP_SECONDS = 0.03
LOFI_KICK_DECAY_SECONDS = 0.16
LOFI_KICK_SAMPLE_GAIN = 0.9
LOFI_KICK_BODY_GAIN = 0.35
# the drive of the soft clip of the lo-fi snare
SNARE_DRIVE = 1.6
# the closed hat: its decay, what each variant adds, and the low edge of its band in Hz
HAT_CLOSED_DECAY_SECONDS = 0.012
HAT_CLOSED_DECAY_STEP_SECONDS = 0.004
HAT_CLOSED_VARIANTS = 3
HAT_CLOSED_LOW_HZ = 6500.0
HAT_OPEN_DECAY_SECONDS = 0.07
HAT_OPEN_LOW_HZ = 6000.0
# the level of a hat and of a shaker at velocity 0, and what velocity adds
NOISE_HIT_VELOCITY_FLOOR = 0.5
NOISE_HIT_VELOCITY_RANGE = 0.5
# the shaker: its length, the attack and the decay of the soft one and of the other, and their bands in Hz
SHAKER_SECONDS = 0.16
SHAKER_SOFT_ATTACK_SECONDS = 0.016
SHAKER_ATTACK_SECONDS = 0.012
SHAKER_SOFT_DECAY_SECONDS = 0.04
SHAKER_DECAY_SECONDS = 0.03
SHAKER_SOFT_BAND_HZ = (2800.0, 6000.0)
SHAKER_BAND_HZ = (4500.0, 9500.0)
# the click is moved by this many semitones for every set of its shots, from this one up
CLICK_STEP_SEMITONES = 2
CLICK_BASE_SEMITONES = -2
# the tom, with the same parts (a variant raises its pitch)
TOM_PITCH_HZ = 96.0
TOM_PITCH_STEP_HZ = 12.0
TOM_VARIANTS = 3
TOM_START_HZ = 140.0
TOM_DROP_SECONDS = 0.03
TOM_DECAY_SECONDS = 0.22


def _make(name: str, vel: float, patch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_hash(name, round(vel, KEY_DECIMALS), patch) % SEED_MODULUS)
    if name == KICK_SOFT:
        out = _kick(
            rng,
            vel,
            SOFT_KICK_PITCH_HZ - SOFT_KICK_PITCH_STEP_HZ * (patch % KICK_VARIANTS),
            SOFT_KICK_START_HZ,
            SOFT_KICK_DROP_SECONDS,
            SOFT_KICK_DECAY_SECONDS + SOFT_KICK_DECAY_STEP_SECONDS * (patch % KICK_VARIANTS),
        )
    elif name == KICK_DEEP:
        out = _kick(
            rng,
            vel,
            DEEP_KICK_PITCH_HZ + DEEP_KICK_PITCH_STEP_HZ * (patch % KICK_VARIANTS),
            DEEP_KICK_START_HZ,
            DEEP_KICK_DROP_SECONDS,
            DEEP_KICK_DECAY_SECONDS,
        )
    elif name == KICK_LOFI:
        shot = sampler(SAMPLES_DRUMS).render(
            0.0, vel, None, name=KICK_LOFI_SHOTS[patch % len(KICK_LOFI_SHOTS)]
        )
        body = _kick(
            rng,
            vel,
            LOFI_KICK_PITCH_HZ,
            LOFI_KICK_START_HZ,
            LOFI_KICK_DROP_SECONDS,
            LOFI_KICK_DECAY_SECONDS,
        )
        out = np.zeros(max(len(shot), len(body)), dtype=np.float32)
        out[: len(shot)] += shot * LOFI_KICK_SAMPLE_GAIN
        out[: len(body)] += body * LOFI_KICK_BODY_GAIN
    elif name == SNARE_LOFI:
        shot = sampler(SAMPLES_DRUMS).render(
            0.0, vel, None, name=SNARE_LOFI_SHOTS[patch % len(SNARE_LOFI_SHOTS)]
        )
        out = np.tanh(SNARE_DRIVE * shot) / SNARE_DRIVE
    elif name == RIM:
        out = sampler(SAMPLES_DRUMS).render(0.0, vel, None, name=RIM_SHOTS[patch % len(RIM_SHOTS)])
    elif name == HAT_CLOSED:
        out = _hat(
            rng,
            vel,
            HAT_CLOSED_DECAY_SECONDS
            + HAT_CLOSED_DECAY_STEP_SECONDS * (patch % HAT_CLOSED_VARIANTS),
            HAT_CLOSED_LOW_HZ,
        )
    elif name == HAT_OPEN:
        out = _hat(rng, vel, HAT_OPEN_DECAY_SECONDS, HAT_OPEN_LOW_HZ)
    elif name == SHAKER:
        n = int(SHAKER_SECONDS * SR)
        t = np.arange(n) / SR
        # patch 1 is the soft shaker of the Berlin School: a lower, narrower band and a gentler attack
        soft = patch == 1
        env = np.minimum(
            t / (SHAKER_SOFT_ATTACK_SECONDS if soft else SHAKER_ATTACK_SECONDS), 1.0
        ) * np.exp(-t / (SHAKER_SOFT_DECAY_SECONDS if soft else SHAKER_DECAY_SECONDS))
        band = SHAKER_SOFT_BAND_HZ if soft else SHAKER_BAND_HZ
        out = (
            dsp.band_noise(rng, n, *band)
            * env
            * (NOISE_HIT_VELOCITY_FLOOR + NOISE_HIT_VELOCITY_RANGE * vel)
        ).astype(np.float32)
    elif name == CLICK:
        out = sampler(SAMPLES_DRUMS).render(
            0.0,
            vel,
            None,
            name=CLICK_SHOTS[patch % len(CLICK_SHOTS)],
            semitones=float(
                (patch // len(CLICK_SHOTS)) * CLICK_STEP_SEMITONES + CLICK_BASE_SEMITONES
            ),
        )
    elif name in {ODAIKO, SHIME, HEART}:
        out = taiko.strike(name, vel, patch)
    elif name == TAIKO_RIM:
        out = taiko.strike(RIM, vel, patch)
    else:
        out = _kick(
            rng,
            vel,
            TOM_PITCH_HZ + TOM_PITCH_STEP_HZ * (patch % TOM_VARIANTS),
            TOM_START_HZ,
            TOM_DROP_SECONDS,
            TOM_DECAY_SECONDS,
        )
    return np.asarray(out, dtype=np.float32)
