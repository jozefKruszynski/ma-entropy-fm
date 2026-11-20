"""
The sounds and the edits of Cinematic Glitch: swells, brass-like stabs, a growling bass, and splices that glitch them.

Nothing here is random noise. The sounds are the engine's own tools (band-limited wavetable saws, the ladder
filter, the cello samples, the stretched washes, the taiko), the distortion is made at four times the rate and
filtered back, and every edit is a splice with a short raised-cosine fade at both ends, so none of them clicks:

* stutter: the first slice of a buffer repeated 1/8, 1/16, 1/32 long, repeats overlapped by the fade;
* tape stop: the buffer is read at a falling speed until it stands, an exception to the rule that nothing glides;
* brake: the same, but short and hard (60 to 150 ms): a fast exponential fall to zero, a cut, a gap, and a snap back
  on the grid;
* grain freeze and spectral freeze: a moment held, as Hann grains or as a stretched spectrum;
* reverse swell: a sound with its tail, turned round, that ends on a hit (a note of the score);
* crush: a short sample-rate and bit-depth reduction of one hit, pre-filtered and filtered again after;
* gate chop: a rhythmic gate whose edges are smoothed;
* reverse: a beat or half a beat of a bus played backwards, so that its hit lands on the next grid line;
* pitch repeat: a slice repeated at a fixed speed (an octave up, an octave down, or slowed), a jump and never a glide;
* bit drop: a sample rate of a few kilohertz and three to five bits, with no filter before it, so it aliases and grates;
* roll: repeats of a slice that get shorter (1/8, 1/16, 1/32) and louder.

The new kinds are slices, reversals and index tricks on the buffer (no FFT), so that they can be dealt often.

The sounds are made as notes (a swell, a stab, one bass note) so that they are cached like the voices of the
library; the edits are made on the buses of the render (`glitch_render`).

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from pedalboard import Convolution, Pedalboard  # type: ignore[attr-defined]

from . import dsp
from .dsp import SR, ladder, midi_hz, osc
from .render import impulse
from .stretch import paulstretch

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------- small tools


def smooth_ramp(n: int) -> np.ndarray:
    """Return a raised-cosine rise from 0 to 1 over n samples."""
    return (np.sin(np.linspace(0.0, np.pi / 2.0, max(n, 1), dtype=np.float32)) ** 2).astype(
        np.float32
    )


def _fast_length(n: int) -> int:
    """Return the smallest even number from n up whose only prime factors are 2, 3 and 5 (the lengths that an FFT likes)."""
    n += n % 2
    while True:
        rest = n
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return n
        n += 2


def oversampled(  # noqa: PLR0913, PLR0917
    fn: Callable[..., np.ndarray],
    x: np.ndarray,
    k: int = 4,
    aux: tuple[np.ndarray | float | np.floating[Any], ...] = (),
    block: int = 1 << 15,
    pad: int = 2048,
) -> np.ndarray:
    """
    Apply a memoryless function at k times the rate and return the result at the normal rate.

    The signal is raised by zero-padding its spectrum, shaped, then cut back to the audio band with a taper above
    17 kHz, so the new harmonics that fall above the band are removed and cannot fold back. Blocks overlap by
    `pad` samples, which are thrown away, so that the edge of a block leaves no trace. A block is lengthened to a
    length that an FFT is quick at (with more of the signal around it, never less), and the transforms are done in
    single precision.

    :param fn: Called with the raised block and the raised blocks of `aux` (control curves).
    :param x: Mono samples.
    :param k: The factor.
    :param aux: Arrays as long as x that fn needs (a drive curve), or numbers that are the same all along.
    :param block: The samples made in one go.
    :param pad: The samples of context on each side of a block.
    """
    n = len(x)
    tail = pad + block + 4096
    xp = np.concatenate([np.zeros(pad), x.astype(np.float32), np.zeros(tail)]).astype(np.float32)
    auxp: list[np.ndarray | float | np.floating[Any]] = [
        np.concatenate([np.full(pad, a[0]), a, np.full(tail, a[-1])])
        if isinstance(a, np.ndarray)
        else a
        for a in aux
    ]
    out = np.empty(n, np.float32)
    for a in range(0, n, block):
        b = min(a + block, n)
        m = _fast_length((b - a) + 2 * pad)
        seg = xp[a : a + m]
        spec = np.fft.rfft(seg)
        up = np.zeros(m * k // 2 + 1, np.complex64)
        up[: len(spec)] = spec
        y = np.fft.irfft(up, m * k) * np.float32(k)
        extras = [np.repeat(ap[a : a + m], k) if isinstance(ap, np.ndarray) else ap for ap in auxp]
        y = fn(y, *extras)
        sp = np.fft.rfft(y)[: m // 2 + 1]
        f = np.arange(m // 2 + 1) / (m / 2.0)
        taper = np.clip((1.0 - f) / 0.2, 0.0, 1.0)
        taper = 0.5 - 0.5 * np.cos(np.pi * taper)
        z = np.fft.irfft(sp * taper, m) / k
        out[a:b] = z[pad : pad + (b - a)]
    return out


def hall(
    x: np.ndarray, seconds: float = 4.5, tail: float = 4.0, hp: float = 170.0, lp: float = 6200.0
) -> np.ndarray:
    """Return the wet signal only of the engine's convolution room (stereo in, stereo out, with a tail)."""
    pad = int(tail * SR)
    src = np.pad(x, ((0, pad), (0, 0))).astype(np.float32)
    board = Pedalboard([Convolution(impulse(seconds, 11), mix=1.0, sample_rate=SR)])
    wet = board(np.ascontiguousarray(src.T), SR).T
    wet = np.stack([ladder(wet[:, k], hp, 0.0, 1.0, mode="hp24") for k in range(2)], axis=1)
    wet = np.stack([ladder(wet[:, k], lp, 0.0, 1.0) for k in range(2)], axis=1)
    return wet.astype(np.float32)


# ---------------------------------------------------------------- cinematic sounds


def swell_stack(  # noqa: PLR0913, PLR0917
    midis: tuple[float, ...],
    n: int,
    attack: float,
    open_hz: tuple[float, float],
    voices: int,
    spread: float,
    seed: int,
) -> np.ndarray:
    """
    Return the stacked, detuned band-limited saws of a swell through its opening filter, before the envelope.

    Everything in it is causal and starts from the same state, so the first `n` samples of a longer stack are the
    stack of `n` samples, bit for bit: swells of one chord that differ in length, or in level, share one stack.
    """
    rng = np.random.default_rng(seed)
    left = np.zeros(n, np.float32)
    right = np.zeros(n, np.float32)
    cents = np.linspace(-spread, spread, voices)
    for m in midis:
        for index, cd in enumerate(cents):
            wave = osc(
                "saw", midi_hz(m) * 2.0 ** (cd / 1200.0), n, float(rng.uniform()), limit=8000.0
            )
            pan = (
                (index / max(voices - 1, 1) * 2.0 - 1.0)
                * 0.8
                * (1.0 if rng.uniform() > 0.5 else -1.0)
            )
            ang = (pan + 1.0) * math.pi / 4.0
            left += wave * math.cos(ang)
            right += wave * math.sin(ang)
    scale = 1.0 / math.sqrt(len(midis) * voices)
    t = np.arange(n) / SR
    shape = np.clip(t / attack, 0.0, 1.0) ** 1.4
    cutoff = open_hz[0] + (open_hz[1] - open_hz[0]) * shape
    out = []
    for side in (left, right):
        filt = ladder(side * scale, cutoff, 0.16, 1.0, chunk=512)
        out.append(ladder(filt, 70.0, 0.0, 1.0, mode="hp24"))
    return np.stack(out, axis=1).astype(np.float32)


def swell_pad(  # noqa: PLR0913, PLR0917
    midis: list[float],
    dur: float,
    attack: float = 3.5,
    release: float = 3.0,
    open_hz: tuple[float, float] = (420.0, 3300.0),
    voices: int = 5,
    spread: float = 13.0,
    seed: int = 0,
    stack: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return a slow swell: stacked, detuned band-limited saws per note, spread over the stereo field.

    The ladder filter opens with the swell, so the pad is dark where it starts and bright at its crest.

    :param midis: The notes.
    :param dur: The length in seconds.
    :param attack: The rise in seconds.
    :param release: The fall in seconds.
    :param open_hz: The cutoff at the start and at the crest.
    :param voices: The detuned saws per note.
    :param spread: The detune in cents, either way.
    :param seed: Seeds the phases and the pans.
    :param stack: The `swell_stack` of these notes, at least `dur` long, when it was made before.
    """
    n = int(dur * SR)
    if stack is None:
        stack = swell_stack(tuple(midis), n, attack, open_hz, voices, spread, seed)
    rise = smooth_ramp(int(attack * SR))
    env = np.ones(n, np.float32)
    env[: len(rise)] = rise
    fall = smooth_ramp(int(release * SR))[::-1]
    env[n - len(fall) :] *= fall
    return (stack[:n] * env[:, None]).astype(np.float32)


def _saw_stack(
    freq: float, n: int, cents: tuple[float, ...], rng: np.random.Generator, limit: float
) -> np.ndarray:
    return np.asarray(
        sum(
            osc("saw", freq * 2.0 ** (cd / 1200.0), n, float(rng.uniform()), limit=limit)
            for cd in cents
        )
        / math.sqrt(len(cents))
    )


def brass_stab(  # noqa: PLR0913, PLR0917
    midis: list[float],
    dur: float = 1.2,
    vel: float = 0.9,
    peak_hz: float = 2300.0,
    release: float = 0.6,
    seed: int = 0,
    light: bool = False,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """
    Return a low brass-like stab: saw stacks, shaped before the filter, a fast filter envelope, a slight vibrato.

    The filter blooms in about 70 ms and then closes slowly, which is the blat of a brass chord that is held a moment.
    """
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    sig = np.zeros(n, np.float32)
    cents = (-9.0, 8.0) if light else (-11.0, -4.0, 5.0, 12.0)
    for index, m in enumerate(midis):
        sig += _saw_stack(midi_hz(m), n, cents, rng, 7000.0) * (1.0 if index else 0.8)
        if index < 2:
            sig += 0.5 * osc("saw", midi_hz(m) / 2.0, n, float(rng.uniform()), limit=3000.0)
    sig /= math.sqrt(len(midis))
    drive = 1.4 + 1.4 * vel
    sig = oversampled(lambda y: np.tanh(drive * y) / math.tanh(drive), sig, 2 if light else 4)
    cutoff = 380.0 + (peak_hz * (0.55 + 0.45 * vel) - 380.0) * (1.0 - np.exp(-t / 0.055)) * np.exp(
        -t / 0.95
    )
    sig = ladder(sig, cutoff, 0.2, 1.1, chunk=128 if light else 64)
    env = dsp.envelope(n, 0.045, 0.5, 0.62, dur, release)
    sig = sig * env
    sides = [dsp.pitch_vibrato(sig, 7.0, 5.1, 0.6, ph, 0.28) for ph in (0.0, 0.31)]
    wave = np.stack(sides, axis=1)
    return (wave * (0.7 + 0.3 * vel)).astype(np.float32)


def sub_boom(
    midi: float = 29.0,
    dur: float = 3.4,
    vel: float = 0.9,
    drum: bool = True,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """Return a sub boom: a sine that drops a little at the start, a long decay, and an odaiko strike on top."""
    from .taiko import strike  # noqa: PLC0415

    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi_hz(midi)
    phase = 2.0 * np.pi * (f * t + 0.8 * f * 0.06 * (1.0 - np.exp(-t / 0.06)))
    body = np.sin(phase) * np.exp(-t / 0.95) * (1.0 - np.exp(-t / 0.006))
    body = np.tanh(1.5 * body) / 1.5
    out = body.astype(np.float32) * vel
    if drum:
        hit = strike("odaiko", 0.85)
        out[: len(hit)] += hit[:n] * 0.9 * vel
    return dsp.to_stereo(out)


# ---------------------------------------------------------------- the growl bass

# the two formants of the vowels that the wobble sweeps through (U, O, A, E), and two other sets for the day to choose
VOWEL_SETS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((300.0, 870.0), (450.0, 800.0), (800.0, 1150.0), (400.0, 2000.0)),
    ((350.0, 600.0), (550.0, 1000.0), (700.0, 1220.0), (300.0, 2300.0)),
    ((270.0, 780.0), (400.0, 700.0), (650.0, 1080.0), (500.0, 1700.0)),
)
# (drive, fold, FM index, low-pass cutoff in Hz, wobbles per beat, depth of the wobble, level): the growl from a held
# note under the swells (0) to the full drop (5)
GROWL_STATES: tuple[tuple[float, float, float, float, float, float, float], ...] = (
    (1.2, 0.05, 0.5, 300.0, 0.5, 0.20, 0.35),
    (1.5, 0.15, 1.2, 700.0, 0.5, 0.40, 0.55),
    (1.9, 0.30, 1.8, 700.0, 1.0, 0.60, 0.70),
    (2.6, 0.50, 3.0, 1500.0, 2.0, 0.80, 0.80),
    (3.4, 0.80, 4.2, 2200.0, 2.0, 0.90, 0.90),
    (4.0, 1.00, 5.0, 2800.0, 3.0, 1.00, 1.00),
)


def _shape(y: np.ndarray, drive: np.ndarray | float, fold: np.ndarray | float) -> np.ndarray:
    """Soft clip plus a sine fold; drive and fold are control curves or numbers."""
    clipped = np.tanh(drive * y)
    folded = np.sin(1.7 * drive * y * (0.6 + fold))
    return np.asarray((0.65 * clipped + 0.35 * folded * fold) / (0.7 + 0.3 * fold))


def growl_note(  # noqa: PLR0913, PLR0917
    midi: float,
    dur: float,
    state: int,
    phase0: float,
    beat: float,
    vowels: int = 0,
    light: bool = False,  # noqa: FBT001, FBT002
    target: float = 0.15,
) -> np.ndarray:
    """
    Return one note of the growl bass as stereo audio.

    Two detuned band-limited saws and an FM pair are gated, shaped (soft clip and sine fold at four times the rate),
    and then pass a low-pass and two moving band-passes that sweep through vowels, a different phase for each side.
    A clean sine at the pitch sits under it, which is never distorted. The wobble is locked to the tempo: its phase
    at the start of the note is `phase0` cycles, so a note that starts on the grid wobbles in step with the others.

    On the light tier it is cheaper: shaped at twice the rate, one formant and one side, filters that move less often.

    :param midi: The pitch, constant for the whole note.
    :param dur: How long the note is held.
    :param state: The row of `GROWL_STATES`.
    :param phase0: The phase of the wobble at the start, in cycles.
    :param beat: The length of a beat in seconds, for the wobble.
    :param vowels: The set of `VOWEL_SETS`.
    :param light: Make the cheap version.
    :param target: The RMS of the note while it sounds, before the level of its state.
    """
    drive_v, fold_v, index_v, cut_v, wob_v, depth_v, level_v = GROWL_STATES[state]
    total = dur + 0.2
    n = int(total * SR)
    osf = 2 if light else 4
    freq = np.full(n, midi_hz(midi))
    gate = np.zeros(n)
    gate[: int(dur * SR)] = 1.0
    kern = np.hanning(int(0.014 * SR))
    gate = np.convolve(gate, kern / kern.sum(), mode="same")
    cycles = np.cumsum(freq) / SR
    saw1 = osc("saw", freq * 2.0 ** (-9.0 / 1200.0), n, 0.0, limit=4500.0)
    saw2 = osc("saw", freq * 2.0 ** (10.0 / 1200.0), n, 0.37, limit=4500.0)
    tau = 2.0 * np.pi
    fm = np.sin(tau * cycles + index_v * np.sin(tau * 2.0 * cycles))
    if not light:
        fm = fm + 0.4 * np.sin(tau * 2.0 * cycles + 0.6 * index_v * np.sin(tau * 3.0 * cycles))
    src = (0.5 * (saw1 + saw2) + 0.7 * fm).astype(np.float32)
    src = ladder(src, 60.0, 0.0, 1.0, mode="hp24") * gate.astype(np.float32)
    shaped = oversampled(_shape, src, osf, aux=(np.float32(drive_v), np.float32(fold_v)))
    wob_phase = phase0 + wob_v * np.arange(n) / (beat * SR)
    formants = VOWEL_SETS[vowels % len(VOWEL_SETS)]
    positions = np.linspace(0.0, len(formants) - 1.0, 64)
    f1_tab = np.interp(positions, np.arange(len(formants)), [v[0] for v in formants])
    f2_tab = np.interp(positions, np.arange(len(formants)), [v[1] for v in formants])
    chunk = 512 if light else 256
    sides = []
    for offset in (0.0,) if light else (0.0, 0.23):
        morph = np.clip(0.5 + 0.5 * depth_v * np.sin(tau * (wob_phase + offset)), 0.0, 1.0)
        f1 = np.interp(morph * 63.0, np.arange(64), f1_tab)
        f2 = np.interp(morph * 63.0, np.arange(64), f2_tab)
        low = ladder(shaped, cut_v * (0.8 + 0.4 * morph), 0.34, 1.0, chunk=chunk)
        b1 = ladder(shaped, f1, 0.62, 1.0, chunk=chunk, mode="bp24")
        if light:
            sides.append(0.5 * low + 1.3 * b1)
        else:
            b2 = ladder(shaped, f2, 0.58, 1.0, chunk=chunk, mode="bp24")
            sides.append(0.5 * low + 1.1 * b1 + 0.8 * b2)
    # the light version has one side, which is made once and played on both
    channels = sides[:1] if light else sides
    out = np.stack(
        [
            ladder(oversampled(lambda y: np.tanh(1.25 * y) / 1.25, side, osf), 6500.0, 0.0, 1.0)
            for side in channels
        ],
        axis=1,
    )
    if light:
        out = np.repeat(out, 2, axis=1)
    sub = np.sin(tau * cycles) + 0.18 * np.sin(tau * 2.0 * cycles + 0.5)
    sub = np.tanh(1.2 * sub) / 1.2 * gate
    mix = out * gate[:, None] + 0.48 * sub[:, None]
    active = np.abs(mix).max(axis=1) > 1e-3
    body = (
        float(np.sqrt(np.mean(mix[active].astype(np.float64) ** 2))) + 1e-9 if active.any() else 1.0
    )
    return (mix * (target * level_v / body)).astype(np.float32)


# ---------------------------------------------------------------- the glitch edits


def _ramp(n: int, fade_in: int, fade_out: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    a = min(fade_in, n // 2)
    b = min(fade_out, n // 2)
    if a:
        w[:a] = smooth_ramp(a)
    if b:
        w[n - b :] *= smooth_ramp(b)[::-1]
    return w


def splice(
    buf: np.ndarray, start: int, piece: np.ndarray, fade_in: float = 0.003, fade_out: float = 0.003
) -> None:
    """Replace a stretch of a stereo buffer with a piece, crossfading with the original at both ends."""
    s = max(start, 0)
    e = min(start + len(piece), len(buf))
    if e <= s:
        return
    piece = piece[s - start : e - start]
    w = _ramp(e - s, int(fade_in * SR), int(fade_out * SR))[:, None]
    buf[s:e] = buf[s:e] * (1.0 - w) + piece * w


def _repeats(cap: np.ndarray, lens: list[int], gains: list[float] | None, f: int) -> np.ndarray:
    """Return the repeats of the start of `cap`, one per length, overlapped by `f` samples with complementary ramps."""
    total = sum(lens)
    piece = np.zeros((total + f, 2), np.float32)
    pos = 0
    for i, length in enumerate(lens):
        last = i == len(lens) - 1
        seg = cap[: length if last else length + f]
        w = _ramp(len(seg), 0 if i == 0 else f, f)[:, None]
        g = gains[i] if gains else 1.0
        piece[pos : pos + len(seg)] += seg * w * g
        pos += length
    return piece[:total]


def stutter(
    buf: np.ndarray,
    t0: float,
    slices: list[float],
    gains: list[float] | None = None,
    fade_s: float = 0.0025,
) -> None:
    """
    Repeat the start of a buffer from `t0`: one repeat for every length in `slices` (seconds).

    The repeats overlap by the fade, with complementary ramps, so the level never dips at a join.
    """
    s = int(t0 * SR)
    f = int(fade_s * SR)
    lens = [int(x * SR) for x in slices]
    cap = buf[s : s + max(lens) + f].copy()
    splice(buf, s, _repeats(cap, lens, gains, f), fade_s, fade_s)


def reverse_chunk(buf: np.ndarray, t0: float, length_s: float, fade_s: float = 0.003) -> None:
    """Play `length_s` of a buffer from `t0` backwards, crossfading with the original at both ends."""
    s = int(t0 * SR)
    seg = buf[s : s + int(length_s * SR)]
    splice(buf, s, seg[::-1].copy(), fade_s, fade_s)


def pitch_repeat(  # noqa: PLR0913, PLR0917
    buf: np.ndarray,
    t0: float,
    slice_s: float,
    repeats: int,
    rate: float,
    gains: list[float] | None = None,
    fade_s: float = 0.0025,
) -> None:
    """
    Repeat a slice of a buffer from `t0` at a fixed read speed: 2 is an octave up, 0.5 an octave down.

    A jump to another pitch that holds, never a glide. The slice is read by linear interpolation (after a two-tap
    average when it is read faster than it was made, so that the highs do not fold much), which costs a few
    array operations and no transform.

    :param buf: The stereo buffer, edited in place.
    :param t0: Where the first repeat starts, in seconds of the buffer.
    :param slice_s: The length of each repeat, in seconds of the output.
    :param repeats: How many repeats.
    :param rate: The read speed of the slice.
    :param gains: The gain of each repeat.
    :param fade_s: The overlap of the repeats.
    """
    s = int(t0 * SR)
    n = int(slice_s * SR)
    f = int(fade_s * SR)
    need = int((n + f) * rate) + 2
    src = buf[s : s + need]
    if len(src) < need:
        src = np.pad(src, ((0, need - len(src)), (0, 0)))
    if rate > 1.0:
        src = (src[:-1] + src[1:]) * np.float32(0.5)
        src = np.concatenate([src, src[-1:]])
    pos = np.arange(n + f) * rate
    j0 = pos.astype(np.int64)
    fr = (pos - j0)[:, None].astype(np.float32)
    seg = src[j0] * (1.0 - fr) + src[j0 + 1] * fr
    splice(buf, s, _repeats(seg, [n] * repeats, gains, f), fade_s, fade_s)


def bit_drop(buf: np.ndarray, t0: float, dur: float, factor: int = 16, bits: int = 4) -> None:
    """
    Drop the sample rate to a few kilohertz and the depth to a few bits, with no filter before or after.

    Everything above the new Nyquist folds back and the steps of the rounding are heard as grit: much harsher than
    `crush`. The level of the stretch is kept.

    :param buf: The stereo buffer, edited in place.
    :param t0: Where it starts, in seconds of the buffer.
    :param dur: How long it lasts.
    :param factor: The samples that each held value lasts for (44.1 kHz over this is the new rate).
    :param bits: The depth in bits.
    """
    s = int(t0 * SR)
    seg = buf[s : s + int(dur * SR)]
    if not len(seg):
        return
    held = np.repeat(seg[::factor], factor, axis=0)[: len(seg)]
    peak = float(np.abs(seg).max()) + 1e-9
    levels = np.float32(2 ** (bits - 1))
    out = np.round(held / peak * levels) / levels * peak
    ref = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + 1e-9
    got = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-9
    splice(
        buf,
        s,
        (out * np.float32(min(ref / got, BIT_DROP_MAX_GAIN))).astype(np.float32),
        0.002,
        0.004,
    )


def tape_stop(  # noqa: PLR0913, PLR0917
    buf: np.ndarray,
    t0: float,
    stop_s: float = 0.45,
    resume_t: float | None = None,
    wind_s: float = 0.18,
    curve: float = 1.7,
) -> None:
    """
    Slow the material down to a stop from `t0` over `stop_s`; it is silent until `resume_t`, where it winds up again.

    A deliberate exception to the rule that nothing glides: it lasts a fraction of a second. The read speed
    falls as (1 - u) to the power `curve`; the level follows the speed, so it dies with the motor. The wind-up
    reads the material so that it arrives in step with the original at `resume_t`.
    """
    s = int(t0 * SR)
    m = int(stop_s * SR)
    s1 = int(resume_t * SR) if resume_t is not None else s + m + int(0.02 * SR)
    orig = buf[s : s1 + 8].copy()
    u = np.arange(m) / m
    rate = (1.0 - u) ** curve
    rel = np.concatenate(([0.0], np.cumsum(rate)[:-1]))
    j0 = rel.astype(np.int64)
    fr = (rel - j0)[:, None].astype(np.float32)
    out = orig[j0] * (1.0 - fr) + orig[j0 + 1] * fr
    out *= (rate**0.4)[:, None].astype(np.float32)
    tail = int(0.012 * SR)
    out[-tail:] *= smooth_ramp(tail)[::-1][:, None]
    piece = np.zeros((s1 - s, 2), np.float32)
    piece[:m] = out
    r = int(wind_s * SR)
    if resume_t is not None and wind_s > 0.0 and (s1 - s - r) > m:
        rate_r = (np.arange(r) / r) ** 1.6
        pos = (s1 - s) - float(np.sum(rate_r)) + np.concatenate(([0.0], np.cumsum(rate_r)[:-1]))
        k0 = pos.astype(np.int64)
        fk = (pos - k0)[:, None].astype(np.float32)
        wind = orig[k0] * (1.0 - fk) + orig[k0 + 1] * fk
        wind *= (rate_r**0.4)[:, None].astype(np.float32)
        piece[s1 - s - r :] = wind
    splice(buf, s, piece, 0.003, 0.004)


def brake_stop(
    buf: np.ndarray,
    t0: float,
    stop_s: float = 0.1,
    resume_t: float | None = None,
    fall: float = 5.0,
) -> None:
    """
    Stop the material like a brake on a record: short, hard, and back on the grid.

    From `t0` the material is read at a speed that falls exponentially (`fall` e-folds over `stop_s`, 60 to 150 ms),
    so the pitch plunges to nothing; the level follows the speed to the power 0.35; then a cut of about a millisecond,
    silence until `resume_t`, and the original material snaps back with a short fade. A glide, like the tape stop,
    and for the same reason allowed: it is a tenth of a second long.

    :param buf: The stereo buffer, edited in place.
    :param t0: Where the brake begins, in seconds of the buffer.
    :param stop_s: How long the fall is.
    :param resume_t: Where the material returns, in seconds of the buffer.
    :param fall: The number of e-folds of the speed.
    """
    s = int(t0 * SR)
    m = int(stop_s * SR)
    s1 = int(resume_t * SR) if resume_t is not None else s + m + int(0.02 * SR)
    orig = buf[s : s1 + 8].copy()
    u = np.arange(m) / m
    rate = np.exp(-fall * u)
    rel = np.concatenate(([0.0], np.cumsum(rate)[:-1]))
    j0 = rel.astype(np.int64)
    fr = (rel - j0)[:, None].astype(np.float32)
    out = orig[j0] * (1.0 - fr) + orig[j0 + 1] * fr
    out *= (rate**0.35)[:, None].astype(np.float32)
    tail = int(0.0015 * SR)
    out[-tail:] *= smooth_ramp(tail)[::-1][:, None]
    piece = np.zeros((s1 - s, 2), np.float32)
    piece[:m] = out
    splice(buf, s, piece, 0.0015, 0.0015)


def grain_freeze(  # noqa: PLR0913, PLR0917
    buf: np.ndarray,
    t0: float,
    length_s: float,
    grain_s: float = 0.12,
    jitter_s: float = 0.02,
    seed: int = 0,
) -> None:
    """Hold the moment before `t0` for `length_s` as overlapped Hann grains read from slightly different points."""
    s = int(t0 * SR)
    n = int(length_s * SR)
    g = int(grain_s * SR)
    g += g % 2
    h = g // 2
    win = np.sin(np.linspace(0.0, np.pi, g, endpoint=False, dtype=np.float32)) ** 2
    rng = np.random.default_rng(seed)
    out = np.zeros((n + g, 2), np.float32)
    for index, k in enumerate(range(0, n, h)):
        p = max(s - g - int(rng.uniform(0.0, jitter_s) * SR), 0)
        w = win.copy()
        if index == 0:
            w[:h] = 1.0
        out[k : k + g] += buf[p : p + g] * w[:, None]
    splice(buf, s, out[:n], 0.004, 0.05)


def spectral_freeze(  # noqa: PLR0913, PLR0917
    buf: np.ndarray,
    t0: float,
    length_s: float,
    back_s: float = 0.9,
    window: int = 1 << 14,
    seed: int = 0,
) -> None:
    """Hold the moment before `t0` as a smooth drone: its spectrum with new phases (the engine's Paulstretch)."""
    s = int(t0 * SR)
    n = int(length_s * SR)
    cap = buf[max(s - int(back_s * SR), 0) : s]
    factor = max(n / max(len(cap) - window, 1), 1.0)
    sides = []
    for ch in range(2):
        st = paulstretch(np.ascontiguousarray(cap[:, ch]), factor, seed + ch, window)[:n]
        sides.append(np.pad(st, (0, n - len(st))))
    out = np.stack(sides, axis=1)
    ref = float(np.sqrt(np.mean(cap.astype(np.float64) ** 2))) + 1e-9
    got = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) + 1e-9
    splice(buf, s, out * np.float32(ref / got), 0.08, 0.15)


def crush(buf: np.ndarray, t0: float, dur: float, factor: int = 4, bits: int = 9) -> None:
    """
    Reduce the sample rate and the bit depth of one short hit.

    The hit is low-passed below the new Nyquist first, held (zero order) and rounded, and low-passed again at
    5.5 kHz after, so that what is left is a gritty, dark texture and not a fizz.
    """
    s = int(t0 * SR)
    n = int(dur * SR)
    seg = buf[s : s + n].copy()
    out = []
    levels = float(2 ** (bits - 1))
    for ch in range(2):
        x = ladder(np.ascontiguousarray(seg[:, ch]), 0.42 * SR / factor, 0.0, 1.0, mode="lp24")
        held = np.repeat(x[::factor], factor)[: len(x)]
        held = np.round(held * levels) / levels
        out.append(ladder(held.astype(np.float32), 5500.0, 0.0, 1.0))
    splice(buf, s, np.stack(out, axis=1), 0.003, 0.006)


def gate_chop(  # noqa: PLR0913, PLR0917
    buf: np.ndarray,
    t0: float,
    step_s: float,
    pattern: list[int],
    floor: float = 0.04,
    smooth_s: float = 0.004,
) -> None:
    """Chop a stretch of a buffer with a gate, one step of the pattern for every `step_s`; the edges are smoothed."""
    step = int(step_s * SR)
    env = np.repeat(np.array([1.0 if p else floor for p in pattern]), step)
    k = max(int(smooth_s * SR), 3)
    kern = np.hanning(k)
    kern /= kern.sum()
    padded = np.concatenate([np.ones(k), env, np.ones(k)])
    sm = np.convolve(padded, kern, mode="same")[k : k + len(env)]
    s = int(t0 * SR)
    e = min(s + len(sm), len(buf))
    buf[s:e] *= sm[: e - s, None].astype(np.float32)


# the most that a bit drop may lift a stretch to hold its level
BIT_DROP_MAX_GAIN = 1.5
EDIT_KINDS = (
    "stutter",
    "tape",
    "brake",
    "grain",
    "spectral",
    "crush",
    "gate",
    "reverse",
    "pitchrep",
    "bitdrop",
    "roll",
)


def edit_extent(kind: str, params: dict[str, Any]) -> float:
    """Return how many seconds from its start an edit changes."""
    extents = {
        "stutter": lambda p: sum(p["slices"]) + 0.01,
        "roll": lambda p: sum(p["slices"]) + 0.01,
        "reverse": lambda p: p["length"] + 0.01,
        "pitchrep": lambda p: p["slice"] * p["repeats"] + 0.01,
        "bitdrop": lambda p: p["dur"] + 0.01,
        "tape": lambda p: p["span"],
        "brake": lambda p: p["span"],
        "grain": lambda p: p["length"] + 0.06,
        "spectral": lambda p: p["length"] + 0.16,
        "crush": lambda p: p["dur"] + 0.01,
        "gate": lambda p: p["step"] * len(p["pattern"]) + 0.02,
    }
    return float(extents[kind](params))


def edit_history(kind: str, params: dict[str, Any]) -> float:
    """Return how many seconds before its start an edit reads the material."""
    if kind == "spectral":
        return float(params["back"]) + 0.05
    if kind == "grain":
        return float(params["grain"]) + float(params.get("jitter", 0.02)) + 0.05
    return 0.02


def _apply_slice_edit(buf: np.ndarray, t0: float, kind: str, params: dict[str, Any]) -> None:
    """Make a reversal, a pitch repeat or a bit drop."""
    if kind == "reverse":
        reverse_chunk(buf, t0, float(params["length"]))
    elif kind == "pitchrep":
        pitch_repeat(
            buf,
            t0,
            float(params["slice"]),
            int(params["repeats"]),
            float(params["rate"]),
            list(params["gains"]),
        )
    else:
        bit_drop(buf, t0, float(params["dur"]), int(params["factor"]), int(params["bits"]))


def apply_edit(buf: np.ndarray, t0: float, kind: str, params: dict[str, Any]) -> None:
    """
    Make one edit on a stereo buffer, in place.

    :param buf: The material.
    :param t0: The second of the buffer where the edit starts.
    :param kind: One of `EDIT_KINDS`.
    :param params: What the kind needs (see the composer).
    """
    if kind in {"reverse", "pitchrep", "bitdrop"}:
        _apply_slice_edit(buf, t0, kind, params)
    elif kind in {"stutter", "roll"}:
        stutter(buf, t0, list(params["slices"]), list(params["gains"]))
    elif kind == "tape":
        tape_stop(buf, t0, float(params["stop"]), t0 + float(params["span"]), float(params["wind"]))
    elif kind == "brake":
        brake_stop(buf, t0, float(params["stop"]), t0 + float(params["span"]))
    elif kind == "grain":
        grain_freeze(
            buf, t0, float(params["length"]), float(params["grain"]), seed=int(params["seed"])
        )
    elif kind == "spectral":
        spectral_freeze(
            buf, t0, float(params["length"]), float(params["back"]), seed=int(params["seed"])
        )
    elif kind == "crush":
        crush(buf, t0, float(params["dur"]), int(params["factor"]), int(params["bits"]))
    elif kind == "gate":
        gate_chop(buf, t0, float(params["step"]), list(params["pattern"]), float(params["floor"]))
