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
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import LowpassFilter

from . import dsp
from .dsp import SR, ensemble, envelope, harmonic_table, ladder, midi_hz, osc, sine
from .quality import STANDARD, Quality
from .sampler import SAMPLES_DIR, Instrument, _read
from .timeline import stable_hash

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
    velocity_steps: int = 8


def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / SR


def _mono(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    return dsp.to_stereo(x, pan)


# ---------------------------------------------------------------- basses


def _sub_round(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    body = sine(c.freq, c.n) + 0.32 * sine(2 * c.freq, c.n, 0.25) * np.exp(-t / 0.35)
    if c.patch == 1:
        body = osc("triangle", c.freq, c.n) * 0.9 + 0.2 * sine(2 * c.freq, c.n) * np.exp(-t / 0.2)
    body = np.tanh(1.3 * body) / 1.3
    return _mono(body * envelope(c.n, 0.012, 0.5, 0.85, c.dur, 0.14))


def _moog_bass(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    wave = (
        0.55 * osc("saw", f, c.n) + 0.4 * osc("square", f * 1.0023, c.n, 0.3) + 0.5 * sine(f, c.n)
    )
    top = 700.0 + 1500.0 * c.vel + (400.0 if c.patch else 0.0)
    low = max(150.0, 2.4 * f)
    cutoff = low + (max(top, low * 3.0) - low) * np.exp(-t / 0.20)
    wave = ladder(wave, cutoff, 0.42, 1.5, chunk=64)
    return _mono(wave * envelope(c.n, 0.004, 0.4, 0.65, c.dur, 0.12))


def _reese(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    cents = [-10.0, 0.0, 12.0] if c.tier.unison >= 0.5 else [-9.0, 10.0]
    if c.patch == 1:
        cents = [c * 0.55 for c in cents]
    saws = sum(
        osc("saw", f * 2.0 ** (cd / 1200.0), c.n, float(c.rng.uniform()), limit=4500.0)
        for cd in cents
    ) / math.sqrt(len(cents))
    # a little waveshaping BEFORE the filter, on band-limited saws, so that the harmonics it makes stay harmonic
    drive = 1.8 + 1.4 * c.vel
    shaped = np.tanh(drive * saws) / math.tanh(drive)
    lfo = 0.10 + 0.22 * float(c.rng.uniform())
    cutoff = (170.0 + 760.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (lfo * t + c.rng.uniform())))) * (
        0.65 + 0.7 * c.vel
    )
    filt = ladder(shaped, cutoff, 0.52, 1.0, chunk=c.tier.control_chunk // 2)
    sub = sine(f, c.n) * 0.7
    wave = (0.85 * filt + sub) * envelope(c.n, 0.02, 0.4, 0.9, c.dur, 0.25)
    return _mono(wave)


def _fm_growl(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    ratio = 2.0 if c.patch == 0 else 1.0
    lfo = 0.18 + 0.3 * float(c.rng.uniform())
    index = (0.7 + 3.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (lfo * t + c.rng.uniform())))) * (
        0.6 + 0.6 * c.vel
    )
    phase = 2 * np.pi * f * t
    wave = np.sin(phase + index * np.sin(ratio * phase))
    if c.tier.rich_bass:
        wave = 0.8 * wave + 0.35 * np.sin(2 * phase + 0.6 * index * np.sin(3 * phase))
    wave = ladder(wave.astype(np.float32), 2300.0, 0.2, 1.0)
    out = (0.8 * wave + 0.65 * sine(f, c.n)) * envelope(c.n, 0.012, 0.4, 0.9, c.dur, 0.2)
    return _mono(out)


def _acid_bass(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    wave = osc("saw" if c.patch == 0 else "square", c.freq, c.n)
    top = 1200.0 + 4800.0 * c.vel
    low = max(260.0, 3.0 * c.freq)
    cutoff = low + (max(top, low * 3.0) - low) * np.exp(-t / 0.11)
    wave = ladder(wave, cutoff, 0.84, 2.4, chunk=32)
    wave = np.tanh(1.8 * wave) / 1.8
    return _mono(wave * envelope(c.n, 0.003, 0.2, 0.55, c.dur, 0.06))


# ---------------------------------------------------------------- strings and pads


def _string_machine(c: Ctx) -> np.ndarray:
    f = c.freq
    base = osc("saw", f * 1.0021, c.n, limit=7000.0) + osc(
        "saw", f * 0.9979, c.n, 0.4, limit=7000.0
    )
    top = 0.55 * osc("saw", 2 * f, c.n, 0.2, limit=7000.0)
    wave = (0.5 * base + top) * envelope(c.n, 0.7, 1.0, 1.0, c.dur, 1.5)
    return _wide(wave, c, (0.55, 0.7, 5.6), 2.0, 3, 0.5)


def _wide(  # noqa: PLR0913, PLR0917
    wave: np.ndarray, c: Ctx, rates: tuple[float, ...], depth_ms: float, taps: int, mix: float
) -> np.ndarray:
    """Make a mono voice stereo: an ensemble where the tier has chorus, two short fixed delays on the cheapest."""
    if c.tier.chorus:
        return ensemble(wave, c.rng, rates, depth_ms, taps, mix)
    left = wave * 0.85 + np.roll(wave, 331) * 0.3
    right = wave * 0.85 + np.roll(wave, 577) * 0.3
    return np.stack([left, right], axis=1).astype(np.float32)


def _pulse(f: float, width: np.ndarray, n: int, phase: float = 0.0) -> np.ndarray:
    """A band-limited pulse of a changing width: the difference of two saws."""  # noqa: D401
    tab = dsp.table("saw", dsp.level_for(f))
    first = dsp.read_table(tab, phase + f * np.arange(n) / SR)
    second = dsp.read_table(tab, phase + f * np.arange(n) / SR + width)
    return (first - second) * 0.8


def _juno_pad(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    width = 0.5 + 0.16 * np.sin(2 * np.pi * (0.33 * t + c.rng.uniform()))
    wave = (
        0.5 * osc("saw", f, c.n, limit=9000.0)
        + 0.5 * _pulse(f, width, c.n)
        + 0.35 * sine(f / 2, c.n)
    )
    wave = wave * envelope(c.n, 0.32, 1.0, 1.0, c.dur, 1.3)
    return _wide(wave, c, (0.5, 0.83), 1.8, 2, 0.5)


def _supersaw(c: Ctx) -> np.ndarray:
    count = max(3, round(7 * c.tier.unison))
    spread = 18.0 + 8.0 * (c.patch % 2)
    offsets = np.linspace(-1.0, 1.0, count) * spread
    left = np.zeros(c.n, dtype=np.float32)
    right = np.zeros(c.n, dtype=np.float32)
    for index, cents in enumerate(offsets):
        wave = osc(
            "saw", c.freq * 2.0 ** (cents / 1200.0), c.n, float(c.rng.uniform()), limit=10000.0
        )
        pan = (index / max(count - 1, 1)) * 2.0 - 1.0
        stereo = dsp.to_stereo(wave, pan * 0.85)
        left += stereo[:, 0]
        right += stereo[:, 1]
    scale = 1.0 / math.sqrt(count)
    env = envelope(c.n, 0.14, 1.0, 1.0, c.dur, 1.0)
    out = np.stack([left, right], axis=1) * scale * env[:, None]
    return np.stack([ladder(out[:, k], 130.0, 0.1, mode="hp24") for k in (0, 1)], axis=1)


def _tape(wave: np.ndarray, c: Ctx, depth: float = 1.0) -> np.ndarray:
    """Tape wow and flutter: a drift of a few cents, which is a wobble and never a glide."""
    if not c.tier.wobble or depth <= 0.0:
        return wave
    t = _t(len(wave))
    wow = 38.0 * np.sin(2 * np.pi * (0.63 * t + c.rng.uniform()))
    flutter = 1.4 * np.sin(2 * np.pi * (6.1 * t + c.rng.uniform()))
    return dsp.delay_read(wave, 60.0 + depth * (wow + flutter))


def _tape_pair(wave: np.ndarray, c: Ctx, depth: float) -> np.ndarray:
    """Two copies on the tape with their own wow, left and right: a width that is a shimmer, not a comb."""
    if not c.tier.wobble:
        return dsp.to_stereo(wave)
    t = _t(len(wave))
    sides = []
    for _side in range(2):
        wow = 36.0 * np.sin(2 * np.pi * (0.55 * t + c.rng.uniform())) + 1.2 * np.sin(
            2 * np.pi * (6.3 * t + c.rng.uniform())
        )
        sides.append(dsp.delay_read(wave, 80.0 + depth * wow))
    return np.stack(sides, axis=1)


_SAMPLERS: dict[str, Instrument] = {}


def sampler(name: str) -> Instrument:
    """Return a sampled instrument of `samples/`, loaded once."""
    if name not in _SAMPLERS:
        _SAMPLERS[name] = Instrument.load(SAMPLES_DIR / name)
    return _SAMPLERS[name]


def _tape_strings(c: Ctx) -> np.ndarray:
    group = "cellos" if c.midi < 53 else "violins"
    wave = sampler("strings").render(
        c.midi, c.vel, c.dur, group=group, attack=0.55 + 0.35 * (c.patch % 2)
    )
    wave = _tape(wave, c, 1.0)
    # tape lets little above 5 kHz through: the voice's own colour, not a blanket filter
    wave = np.asarray(LowpassFilter(5200.0).process(wave[None, :], SR)[0], dtype=np.float32)
    return _tape_pair(wave, c, 1.0)


def _chant_cello(c: Ctx) -> np.ndarray:
    """A cello section for a slow line: the warm low sections only, a patient attack, a dark top, nearly centred."""  # noqa: D401
    wave = sampler("strings").render(
        c.midi, c.vel, c.dur, group="cellos", attack=0.42 + 0.2 * (c.patch % 2)
    )
    # a steep low-pass: the section is warm, and its bow noise above 4 kHz is not wanted under a slow line
    wave = ladder(wave, 3000.0, 0.0, 1.0)
    return dsp.to_stereo(wave, 0.08 if c.patch == 0 else -0.08)


_VOWELS = {
    0: ((800.0, 1.0, 90.0), (1150.0, 0.55, 100.0), (2900.0, 0.22, 140.0), (3900.0, 0.09, 160.0)),
    1: ((500.0, 1.0, 80.0), (800.0, 0.5, 90.0), (2830.0, 0.12, 130.0)),
}


def _tape_choir(c: Ctx) -> np.ndarray:
    formants = _VOWELS[c.patch % 2]
    f = c.freq
    kmax = int(7000.0 / f)
    k = np.arange(1, kmax + 1) * f
    amps = np.zeros(kmax)
    for centre, gain, width in formants:
        amps += gain * np.exp(-0.5 * ((k - centre) / width) ** 2)
    amps += 0.04 / np.arange(1, kmax + 1)
    tab = harmonic_table(tuple(float(a) for a in np.round(amps / amps.max(), 3)))
    voices = 3 if c.tier.unison >= 0.5 else 2
    wave = np.zeros(c.n, dtype=np.float32)
    for index in range(voices):
        cents = (index - (voices - 1) / 2.0) * 7.0
        phase = dsp.vibrato_phase(
            f * 2.0 ** (cents / 1200.0), c.n, 5.0 + 0.4 * index, 6.0, 0.7, float(c.rng.uniform())
        )
        wave += dsp.read_table(tab, phase + float(c.rng.uniform()))
    wave = wave / math.sqrt(voices) * envelope(c.n, 0.9, 1.0, 1.0, c.dur, 1.6)
    return _tape_pair(wave, c, 0.8)


def _glass_pad(c: Ctx) -> np.ndarray:
    f = c.freq
    kmax = int(12000.0 / f)
    k = np.arange(1, kmax + 1)
    a = np.where(k % 2 == 1, 1.0 / k**0.9, 0.12 / k)
    b = (1.0 / k**0.55) * np.exp(-0.5 * ((k - (6 + 3 * (c.patch % 2))) / 2.2) ** 2) + 0.05 / k
    tab_a = harmonic_table(tuple(float(x) for x in np.round(a, 4)))
    tab_b = harmonic_table(tuple(float(x) for x in np.round(b / b.max(), 4)))
    t = _t(c.n)
    morph = 0.5 + 0.5 * np.sin(2 * np.pi * (0.06 * t + c.rng.uniform()))
    left = np.zeros(c.n, dtype=np.float32)
    right = np.zeros(c.n, dtype=np.float32)
    for side, cents in ((0, -5.0), (1, 5.0)):
        ph = float(c.rng.uniform())
        mix = (1 - morph) * dsp.wavetable_osc(
            tab_a, f * 2.0 ** (cents / 1200.0), c.n, ph
        ) + morph * dsp.wavetable_osc(tab_b, f * 2.0 ** (cents / 1200.0), c.n, ph)
        (left if side == 0 else right)[:] = mix
    env = envelope(c.n, 1.1, 1.0, 1.0, c.dur, 2.0)
    return np.stack([left, right], axis=1) * env[:, None] * 0.8


def _bowed_pad(c: Ctx) -> np.ndarray:
    f = c.freq
    kmax = int(9000.0 / f)
    k = np.arange(1, kmax + 1)
    tab = harmonic_table(tuple(float(x) for x in np.round(1.0 / k**1.15, 4)))
    wave = np.zeros(c.n, dtype=np.float32)
    for cents in (-3.0, 3.0):
        phase = dsp.vibrato_phase(
            f * 2.0 ** (cents / 1200.0),
            c.n,
            4.8 + 0.3 * float(c.rng.uniform()),
            6.0,
            0.8,
            float(c.rng.uniform()),
        )
        wave += dsp.read_table(tab, phase + float(c.rng.uniform()))
    t = _t(c.n)
    bow = (
        dsp.band_noise(c.rng, c.n, 1500.0, 4200.0)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * t))
        * 0.012
    )
    wave = (wave * 0.6 + bow) * envelope(c.n, 0.9, 1.0, 1.0, c.dur, 1.5)
    return dsp.to_stereo(wave, 0.0) * 1.0 + 0.0 * dsp.to_stereo(bow, 0.5)


def _drawbar_organ(c: Ctx) -> np.ndarray:
    f = c.freq
    bars = (
        {1: 1.0, 2: 0.75, 3: 0.4, 4: 0.34, 6: 0.14, 8: 0.1}
        if c.patch == 0
        else {1: 1.0, 2: 0.2, 3: 0.55, 4: 0.1, 5: 0.12}
    )
    kmax = int(10000.0 / f)
    amps = [bars.get(k, 0.0) for k in range(1, kmax + 1)]
    tab = harmonic_table(tuple(amps))
    t = _t(c.n)
    trem = 1.0 + 0.04 * np.sin(2 * np.pi * 0.35 * t)
    wave = dsp.wavetable_osc(tab, f, c.n) * trem * envelope(c.n, 0.5, 1.0, 1.0, c.dur, 1.2)
    return _wide(wave, c, (0.4, 0.6), 1.4, 2, 0.35)


# ---------------------------------------------------------------- leads, keys and plucks


def _analog_lead(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    phase = dsp.vibrato_phase(c.freq, c.n, 5.2, 12.0, 0.45, float(c.rng.uniform()))
    second = "square" if c.patch == 0 else "triangle"
    lvl = dsp.level_for(c.freq * 1.01, 10000.0)
    wave = 0.6 * dsp.read_table(dsp.table("saw", lvl), phase) + 0.45 * dsp.read_table(
        dsp.table(second, lvl), phase * 1.0017 + 0.2
    )
    cutoff = 900.0 + 2300.0 * (1.0 - np.exp(-t / 0.7)) * (0.6 + 0.6 * c.vel)
    wave = ladder(wave, cutoff, 0.22, 1.0, chunk=256)
    return _mono(wave * envelope(c.n, 0.04, 0.5, 0.85, c.dur, 0.7), 0.1)


def _flute_lead(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    phase = dsp.vibrato_phase(c.freq, c.n, 4.9, 10.0, 0.5, float(c.rng.uniform()))
    wave = 0.8 * dsp.read_table(
        dsp.table("triangle", dsp.level_for(c.freq * 1.01, 6000.0)), phase
    ) + 0.22 * dsp.read_table(dsp.table("square", dsp.level_for(c.freq * 1.01, 3000.0)), phase)
    breath = dsp.band_noise(c.rng, c.n, 1800.0, 5000.0) * (0.35 * np.exp(-t / 0.12) + 0.05) * 0.04
    return _mono((wave + breath) * envelope(c.n, 0.09, 0.6, 0.9, c.dur, 0.35), -0.1)


def _fm_ep(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    carrier = 2 * np.pi * f * t
    index = (1.4 + 1.8 * c.vel) * np.exp(-t / (0.9 + 0.5 * (c.patch % 2)))
    body = np.sin(carrier + index * np.sin(carrier))
    tine = 0.45 * c.vel * np.exp(-t / 0.045) * np.sin(carrier + 0.9 * np.sin(14.0 * carrier))
    amp = np.exp(-t / (1.4 + 1.2 * (1.0 - min(f / 600.0, 1.0))))
    wave = (body + tine * 0.6) * amp
    wave = wave * envelope(c.n, 0.004, 3.0, 1.0, c.dur + 0.4, 0.5)
    return dsp.to_stereo(wave.astype(np.float32), 0.05)


def _sampled(name: str, c: Ctx, attack: float | None = None) -> np.ndarray:
    wave = sampler(name).render(c.midi, c.vel, c.dur, attack=attack)
    pan = float(np.clip((c.midi - 60.0) / 60.0, -0.4, 0.4))
    return dsp.to_stereo(wave, pan)


def _felt_piano(c: Ctx) -> np.ndarray:
    return _sampled("felt_piano", c)


def _grand_piano(c: Ctx) -> np.ndarray:
    return _sampled("grand_piano", c)


def _tx_ep(c: Ctx) -> np.ndarray:
    return _sampled("ep", c)


def _strum_pluck(c: Ctx) -> np.ndarray:
    return _sampled("pluck", c)


def _juno_pluck(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    width = np.full(c.n, 0.42)
    wave = 0.55 * osc("saw", f, c.n, limit=9000.0) + 0.5 * _pulse(f, width, c.n)
    top = 500.0 + 3400.0 * c.vel
    cutoff = 380.0 + (top - 380.0) * np.exp(-t / 0.22)
    wave = ladder(wave, cutoff, 0.28, 1.0, chunk=96)
    wave = wave * envelope(c.n, 0.004, 0.45, 0.0, c.dur, 0.35)
    return _wide(wave, c, (0.7,), 1.6, 2, 0.5)


def _moog_seq(c: Ctx) -> np.ndarray:
    """A sequencer note: band-limited saw and square through the ladder, the cutoff opened by the velocity (the accent)."""  # noqa: D401
    t = _t(c.n)
    f = c.freq
    wave = 0.6 * osc("saw", f, c.n, limit=9000.0) + 0.5 * osc(
        "square", f * 1.0031, c.n, 0.17, limit=7000.0
    )
    low = max(420.0, 2.6 * f)
    top = low + 4200.0 * (0.25 + 0.9 * c.vel**1.6) * (1.25 if c.patch else 1.0)
    cutoff = low + (top - low) * np.exp(-t / (0.10 + 0.04 * c.vel))
    wave = ladder(wave, cutoff, 0.46 + 0.1 * c.vel, 1.6, chunk=96 if c.tier.unison < 0.5 else 48)
    gate = envelope(c.n, 0.003, 0.5, 0.85, c.dur, 0.07)
    return _mono(wave * gate, -0.05 if c.patch == 0 else 0.1)


def _blip_synth(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    wave = osc("square", c.freq, c.n, limit=7000.0) * 0.8 + 0.3 * osc("triangle", c.freq * 2.0, c.n)
    top = 500.0 + 3200.0 * c.vel
    cutoff = max(280.0, 2.0 * c.freq) + (max(top, 3.0 * c.freq) - 280.0) * np.exp(-t / 0.05)
    wave = ladder(wave, cutoff, 0.5, 1.3, chunk=96 if c.tier.unison < 0.5 else 48)
    return _mono(wave * envelope(c.n, 0.002, 0.07 + 0.05 * (c.patch % 2), 0.0, c.dur, 0.05), 0.15)


def _chord_stab(c: Ctx) -> np.ndarray:
    t = _t(c.n)
    f = c.freq
    wave = 0.6 * osc("saw", f * 1.004, c.n, limit=8000.0) + 0.6 * osc(
        "saw", f * 0.996, c.n, 0.31, limit=8000.0
    )
    top = 900.0 + 2100.0 * c.vel
    cutoff = max(260.0, 1.8 * f) + (top - 260.0) * np.exp(-t / 0.18)
    wave = ladder(wave, cutoff, 0.32, 1.0, chunk=64)
    wave = wave * envelope(c.n, 0.004, 0.3 + 0.1 * (c.patch % 2), 0.0, c.dur, 0.25)
    return dsp.to_stereo(wave, -0.15)


def _nylon_ks(c: Ctx) -> np.ndarray:
    period = SR / c.freq
    size = max(round(period), 8)
    actual = SR / (size + 0.5)
    length = int(c.n * actual / c.freq) + 2 * size
    seed = np.convolve(c.rng.uniform(-1, 1, size + 8), np.ones(5) / 5.0, mode="same")[4 : size + 4]
    out = np.zeros(length + size + 1, dtype=np.float64)
    out[1 : size + 1] = seed * (0.6 + 0.4 * c.vel)
    decay = 0.9985 - 0.0008 * min(c.freq / 880.0, 1.5)
    for start in range(size + 1, length + size + 1, size):
        end = min(start + size, len(out))
        out[start:end] = (
            decay * 0.5 * (out[start - size : end - size] + out[start - size - 1 : end - size - 1])
        )
    raw = out[1 : length + 1].astype(np.float32)
    # the loop is a little flat for the averaging filter: read it a touch faster so that it is in tune
    pos = np.arange(c.n) * (actual / c.freq)
    wave = _read(raw, pos)
    return dsp.to_stereo(
        (wave * envelope(c.n, 0.002, 2.0, 1.0, c.dur + 1.5, 0.4)).astype(np.float32), 0.1
    )


VOICES: dict[str, Voice] = {
    v.key: v
    for v in (
        Voice("sub_round", "Warm round sub", "bass", 36, 0.2, _sub_round, patches=2),
        Voice("moog_bass", "Moog-style ladder bass", "bass", 41, 0.2, _moog_bass),
        Voice("reese", "Growling Reese bass", "bass", 36, 0.3, _reese),
        Voice("fm_growl", "FM growl bass", "bass", 38, 0.3, _fm_growl),
        Voice("acid_bass", "Acid 303-style bass", "bass", 40, 0.1, _acid_bass),
        Voice("string_machine", "String machine", "pad", 57, 1.6, _string_machine),
        Voice("juno_pad", "Juno-style pad", "pad", 57, 1.4, _juno_pad, fundamental=0.5),
        Voice("supersaw", "Supersaw ensemble", "pad", 57, 1.1, _supersaw),
        Voice("tape_strings", "Tape strings", "pad", 55, 2.0, _tape_strings, sampled=True),
        Voice("chant_cello", "Chant cello section", "lead", 60, 1.6, _chant_cello, sampled=True),
        Voice("tape_choir", "Tape choir", "pad", 57, 1.8, _tape_choir),
        Voice("glass_pad", "Glassy wavetable pad", "pad", 60, 2.2, _glass_pad),
        Voice("bowed_pad", "Bowed pad", "pad", 55, 1.7, _bowed_pad),
        Voice("drawbar_organ", "Drawbar organ", "pad", 48, 1.3, _drawbar_organ),
        Voice("analog_lead", "Singing analog lead", "lead", 64, 0.8, _analog_lead),
        Voice("flute_lead", "Flute-like lead", "lead", 72, 0.5, _flute_lead),
        Voice("fm_ep", "FM electric piano", "key", 60, 1.2, _fm_ep),
        Voice("felt_piano", "Felt piano", "key", 60, 0.6, _felt_piano, sampled=True),
        Voice("grand_piano", "Soft grand piano", "key", 60, 0.6, _grand_piano, sampled=True),
        Voice("tx_ep", "Sampled FM electric piano", "key", 60, 0.6, _tx_ep, sampled=True),
        Voice(
            "moog_seq", "Moog-style sequencer voice", "pluck", 55, 0.15, _moog_seq, velocity_steps=4
        ),
        Voice("blip_synth", "Blip synth", "pluck", 67, 0.2, _blip_synth, velocity_steps=4),
        Voice("chord_stab", "Chord stab", "pluck", 57, 0.5, _chord_stab),
        Voice("juno_pluck", "Juno-style pluck", "pluck", 64, 0.4, _juno_pluck),
        Voice("nylon_ks", "Nylon pluck (Karplus-Strong)", "pluck", 60, 0.6, _nylon_ks),
        Voice("strum_pluck", "Strumstick pluck", "pluck", 55, 0.4, _strum_pluck, sampled=True),
    )
}

_GAINS: dict[tuple[str, str], float] = {}


class NoteCache:
    """
    Notes that were made, kept up to a size, the least recently used dropped first.

    Long notes have their own pool, and so do the dear ones (swells and growls that take a tenth of a second or more
    to make and come round again): a dear note is never pushed out by cheap long notes, and the pool has a share of
    its own on top of the limit.
    """

    LONG_SAMPLES = 2 * SR
    DEAR_SHARE = 0.85

    def __init__(self) -> None:
        """Make an empty cache."""
        self._short: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._long: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._dear: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._bytes = {"short": 0, "long": 0, "dear": 0}

    def _pool(
        self, wave: np.ndarray, *, dear: bool
    ) -> tuple[str, OrderedDict[tuple[object, ...], np.ndarray]]:
        if dear:
            return "dear", self._dear
        return ("long", self._long) if len(wave) > self.LONG_SAMPLES else ("short", self._short)

    def get(self, key: tuple[object, ...]) -> np.ndarray | None:
        """Return a note and mark it as used."""
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
        if key in pool:
            self._bytes[name] -= pool[key].nbytes
        pool[key] = wave
        self._bytes[name] += wave.nbytes
        share = {"long": 0.6, "short": 0.4, "dear": self.DEAR_SHARE}[name]
        while self._bytes[name] > limit_mb * share * 1e6 and len(pool) > 1:
            _, old = pool.popitem(last=False)
            self._bytes[name] -= old.nbytes

    def clear(self) -> None:
        """Forget everything."""
        self._short.clear()
        self._long.clear()
        self._dear.clear()
        self._bytes = {"short": 0, "long": 0, "dear": 0}

    @property
    def size(self) -> int:
        """Return how many notes are held."""
        return len(self._short) + len(self._long) + len(self._dear)


NOTES = NoteCache()


def quantised(duration: float) -> float:
    """Round a length to a step that grows with the length, so that more notes can be shared."""
    if duration < 0.6:
        step = 0.02
    elif duration < 3.0:
        step = 0.1 if duration < 1.5 else 0.25
    else:
        step = 0.5 if duration < 8.0 else 1.0
    return min(max(round(duration / step) * step, step), MAX_NOTE_SECONDS)


def _make(key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int) -> np.ndarray:  # noqa: PLR0913, PLR0917
    voice = VOICES[key]
    n = max(int((dur + voice.release) * SR), 64)
    rng = np.random.default_rng(
        stable_hash(key, round(midi, 2), round(dur, 2), round(vel, 2), patch) % (2**63)
    )
    ctx = Ctx(midi, midi_hz(midi), dur, vel, tier, rng, patch, n)
    wave = np.asarray(voice.make(ctx), dtype=np.float32)
    if wave.ndim == 1:
        wave = dsp.to_stereo(wave)
    if not voice.sampled:
        # a softer key is a quieter note as well as a darker one
        wave = wave * np.float32((max(vel, 0.05) / REF_VELOCITY) ** VELOCITY_POWER)
    return wave


def gain_of(key: str, tier: Quality) -> float:
    """Return the constant gain of a voice, from the loudness of a reference note (RMS 0.09)."""
    cache_key = (key, tier.key)
    if cache_key not in _GAINS:
        voice = VOICES[key]
        ref = _make(key, voice.ref_midi, 2.5, REF_VELOCITY, tier, 0)
        held = ref[: int(2.5 * SR)]
        _GAINS[cache_key] = REFERENCE_RMS / max(dsp.rms(held), 1e-6)
    return _GAINS[cache_key]


# notes held longer than this are made from a core that is looped: a pad does not change in its sustain
TILE_FROM = 8.0
CORE_HOLD = 4.8
CORE_LOOP_START = 2.4
CORE_CROSS = 0.5
TILED_FAMILIES = frozenset({"pad", "bass"})


def _core(key: str, midi: float, tier: Quality, patch: int) -> np.ndarray:
    """Return the core of a voice at a pitch: a note held for CORE_HOLD seconds, with its release."""
    cache_key = ("core", key, round(midi, 2), tier.key if tier.unison < 1 else "full", patch)
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
        np.maximum(local, 1e-12) * max(float(np.sum(ref.astype(np.float64) ** 2)), 1e-12)
    )
    return int(low + int(np.argmax(score)))


_JUMPS: dict[tuple[object, ...], int] = {}


def _tiled(key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int) -> np.ndarray:  # noqa: PLR0913, PLR0917
    """
    Make a long note from the looped sustain of a core: the attack, the loop as often as needed, the release.

    Every join fades the end of what is there into what comes before the point that it jumps to, which is how a
    sampler loops, and the point is chosen where the tone is in phase, so that the join has neither a seam nor a dip.
    """
    core = _core(key, midi, tier, patch)
    hold, cross = int(CORE_HOLD * SR), int(CORE_CROSS * SR)
    jump_key = (key, round(midi, 2), tier.key if tier.unison < 1 else "full", patch)
    if jump_key not in _JUMPS:
        _JUMPS[jump_key] = _align(core, hold, int(CORE_LOOP_START * SR), cross, int(0.35 * SR))
    start = _JUMPS[jump_key]
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


def _plain_note(  # noqa: PLR0913, PLR0917
    key: str, midi: float, dur: float, vel: float, tier: Quality, patch: int
) -> np.ndarray:
    """Return a note without vibrato; `dur` and `vel` are already on the grid of the cache."""
    cache_key = (key, round(midi, 2), dur, vel, tier.key if tier.unison < 1 else "full", patch)
    found = NOTES.get(cache_key)
    if found is not None:
        return found
    if dur >= TILE_FROM and VOICES[key].family in TILED_FAMILIES:
        # assembled from a cached core in a few milliseconds: not worth keeping, a long note is big
        return _tiled(key, midi, dur, vel, tier, patch)
    wave = _make(key, midi, dur, vel, tier, patch) * np.float32(gain_of(key, tier))
    NOTES.put(cache_key, wave, tier.cache_mb)
    return wave


def render_note(  # noqa: PLR0913, PLR0917
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
    vel = max(round(float(np.clip(vel, 0.05, 1.0)) * steps) / steps, 1.0 / steps / 2.0)
    if vibrato <= 0.0:
        return _plain_note(key, midi, dur, vel, tier, patch)
    cents = round(min(vibrato, MAX_VIBRATO_CENTS), 1)
    cache_key = (
        key,
        round(midi, 2),
        dur,
        vel,
        tier.key if tier.unison < 1 else "full",
        patch,
        cents,
    )
    found = NOTES.get(cache_key)
    if found is not None:
        return found
    # every note has a rate, a phase and a delay of its own, so that voices sung together do not wobble together
    rng = np.random.default_rng(
        stable_hash("vibrato", key, round(midi, 2), dur, vel, patch) % (2**63)
    )
    wave = dsp.pitch_vibrato(
        _plain_note(key, midi, dur, vel, tier, patch),
        cents,
        float(rng.uniform(4.5, 5.5)),
        float(rng.uniform(1.0, 2.0)),
        float(rng.uniform()),
        float(rng.uniform(0.2, 0.5)),
    )
    if dur < TILE_FROM or VOICES[key].family not in TILED_FAMILIES:
        NOTES.put(cache_key, wave, tier.cache_mb)
    return wave
