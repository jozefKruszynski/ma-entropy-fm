"""
The render loop of Entropy FM: a score in, audio out, one second at a time.

Notes of the score are mixed into the buses of their layers from the cache of made notes (`voices`, `drums`),
at their absolute positions, so any second can be rendered without the ones before it. The buses go through
their filters (cutoffs that the house moves), into a shared reverb and echo, and the textures are added. The
master is a slow leveler that holds the loudness of the style, and a look-ahead peak limiter with a ceiling of
-2.4 dBFS that should barely work. A seek renders a few seconds ahead of the start to warm up filters and reverb.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import (  # type: ignore[attr-defined]
    Compressor,
    Convolution,
    Delay,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    Reverb,
)

from . import dsp
from .drums import render_hit
from .dsp import SR, Shimmer, StreamLadder
from .limiter import Limiter
from .score import CR, Layer, Note, Score, Texture
from .stretch import is_wash, render_wash
from .synth import smoothstep
from .timeline import stable_hash
from .voices import VOICES, render_note

if TYPE_CHECKING:
    from collections.abc import Generator

    from .quality import Quality

WARMUP_SECONDS = 8
FADE_IN_SECONDS = 1.5
FADE_OUT_SECONDS = 2.5
# no note rings longer than this after its start, in seconds
REACH_SECONDS = 36.0
CEILING_DB = -2.4
# the leveler follows a rise of the level at once and a fall slowly; it takes gain away faster than it gives it
LEVEL_DOWN_DB = 1.0
LEVEL_UP_DB = 0.4
LEVEL_RISE_RATE = 0.6
LEVEL_FALL_RATE = 0.08
LEVEL_MIN = 10 ** (-2.0 / 20.0)
LEVEL_MAX = 10 ** (2.0 / 20.0)
# how loud the reverb return is against the dry mix at a space of 1
REVERB_WET = 0.9
ECHO_WET = 0.6


@lru_cache(maxsize=8)
def impulse(seconds: float, seed: int) -> np.ndarray:
    """Make a stereo room: noise that decays, darker as it goes, a little different left and right."""
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros((2, n), dtype=np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    for side in range(2):
        spec = np.fft.rfft(rng.standard_normal(n))
        # high frequencies die faster: the decay time falls with frequency
        t_half = seconds * (0.25 + 0.75 / (1.0 + (freqs / 3500.0) ** 1.4))
        t = np.arange(n) / SR
        decay = np.exp(-6.9 * t / seconds)
        wave = np.fft.irfft(spec * (1.0 / (1.0 + (freqs / 9000.0) ** 2)), n)
        mixed = wave * decay
        # the darkening: blend a low-passed copy in as time goes on
        low = np.fft.irfft(np.fft.rfft(wave) / (1.0 + (freqs / 1400.0) ** 2), n) * decay
        share = np.clip(t / (seconds * 0.5), 0.0, 1.0)
        out[side] = (mixed * (1 - share) + low * share).astype(np.float32)
        del t_half
    fade = int(0.03 * SR)
    out[:, :fade] *= np.linspace(0.0, 1.0, fade)
    out[:, int(0.0035 * SR) :] *= 1.0
    out /= np.sqrt(np.sum(out.astype(np.float64) ** 2, axis=1))[:, None].astype(np.float32)
    return out


def _reverb_board(tier: Quality) -> Pedalboard:
    if tier.reverb == "freeverb":
        return Pedalboard(
            [
                HighpassFilter(180.0),
                Reverb(room_size=0.82, damping=0.55, wet_level=1.0, dry_level=0.0, width=1.0),
            ]
        )
    seconds = 2.6 if tier.reverb == "room" else 5.0
    return Pedalboard(
        [HighpassFilter(180.0), Convolution(impulse(seconds, 11), mix=1.0, sample_rate=SR)]
    )


class Renderer:
    """Turns a score into audio, block by block."""

    # the blocks that the output lags behind the block that was rendered (0 here; see `glitch_render`)
    latency = 0

    def __init__(
        self, score: Score, tier: Quality, seed: str = "", solo: frozenset[str] | None = None
    ) -> None:
        """
        Prepare the buses and the effects.

        :param score: What to play.
        :param tier: The quality tier.
        :param seed: Seeds the textures.
        :param solo: When given, only these layers play (and no textures): to hear a part on its own.
        """
        self.solo = solo
        self.score = score
        self.tier = tier
        self.seed = stable_hash(seed, "textures")
        self.notes = sorted(score.notes, key=lambda n: (n.t, n.layer, n.voice, n.midi))
        self.starts = [n.t for n in self.notes]
        self.ladders: dict[str, StreamLadder] = {}
        for key, layer in score.layers.items():
            if layer.cutoff is not None:
                self.ladders[key] = StreamLadder(
                    layer.resonance, 1.0, "lp24" if tier.chorus else "lp12"
                )
        self.highpass = {
            key: StreamLadder(0.0, 1.0, "hp24")
            for key, layer in score.layers.items()
            if layer.highpass > 0.0
        }
        self.reverb = _reverb_board(tier)
        self.echo = Pedalboard(
            [
                Delay(delay_seconds=score.echo_seconds, feedback=score.echo_feedback, mix=1.0),
                LowpassFilter(3600.0),
            ]
        )
        self.has_echo = any(layer.echo for layer in score.layers.values())
        # the shimmer of the richest tier: the reverb send an octave up, fed into the hall
        self.shimmer = Shimmer() if tier.granular and score.shimmer > 0.0 else None
        # a gentle compressor for the styles with drums: it takes the edge off the peaks, so that the limiter has little to do
        self.compressor = (
            Pedalboard(
                [
                    Compressor(
                        threshold_db=score.compress_db, ratio=3.0, attack_ms=4.0, release_ms=110.0
                    )
                ]
            )
            if score.compress_db is not None
            else None
        )
        self.limiter = Limiter(CEILING_DB, SR, stride=tier.limiter_stride)
        self._ms: float | None = None
        self._first = True
        self._gain = 1.0
        self._dc = np.zeros(2)
        self._noise: dict[str, np.ndarray] = {}
        self._wobble_tail = np.zeros((4096, 2), dtype=np.float32)
        # the waves of the notes that are sounding, held so that the cache cannot make a long note again every second
        self._active: dict[int, tuple[float, np.ndarray]] = {}
        self._finished: set[int] = set()

    # -- notes -------------------------------------------------------------------------------------

    def _wave(self, note: Note) -> np.ndarray:
        if note.voice.startswith("drum:"):
            return render_hit(note.voice[5:], note.vel, note.patch)
        if is_wash(note.voice):
            return render_wash(note.voice, note.midi, note.dur, note.vel, self.tier.cache_mb)
        return render_note(
            note.voice, note.midi, note.dur, note.vel, self.tier, note.patch, note.vib
        )

    def _reach(self, note: Note) -> float:
        if note.voice.startswith("drum:"):
            return 2.5
        if is_wash(note.voice):
            return note.dur + 1.0
        return min(note.dur + VOICES[note.voice].release * 4.0 + 1.0, REACH_SECONDS + 8.0)

    def _bus(self, block: int) -> dict[str, np.ndarray]:
        start, end = float(block), float(block + 1)
        buses: dict[str, np.ndarray] = {}
        low = bisect_left(self.starts, start - REACH_SECONDS - 8.0)
        high = bisect_right(self.starts, end)
        for note in self.notes[low:high]:
            if note.t + self._reach(note) < start or id(note) in self._finished:
                continue
            if self.solo is not None and note.layer not in self.solo:
                continue
            held = self._active.get(id(note))
            if held is None:
                wave = self._wave(note)
                self._active[id(note)] = (note.t + len(wave) / SR, wave)
            else:
                wave = held[1]
            first = round(note.t * SR) - block * SR
            a, b = max(first, 0), min(first + len(wave), SR)
            if b <= a:
                continue
            bus = buses.get(note.layer)
            if bus is None:
                bus = buses[note.layer] = np.zeros((SR, 2), dtype=np.float32)
            bus[a:b] += wave[a - first : b - first]
        for key in [k for k, (until, _) in self._active.items() if until < start]:
            del self._active[key]
            self._finished.add(key)
        return buses

    # -- curves ------------------------------------------------------------------------------------

    @staticmethod
    def _ramp(curve: np.ndarray | None, block: int, scale: float = 1.0) -> np.ndarray:
        """Return a curve over a block as a straight ramp between its values at the two edges (it moves slowly)."""
        if curve is None:
            return np.full(SR, scale, dtype=np.float32)
        top = len(curve) - 1.0
        first = float(np.interp(min(max(block * CR, 0.0), top), np.arange(len(curve)), curve))
        last = float(np.interp(min(max((block + 1) * CR, 0.0), top), np.arange(len(curve)), curve))
        return np.linspace(first * scale, last * scale, SR, dtype=np.float32)

    @staticmethod
    def _chunks(curve: np.ndarray, block: int, chunk: int) -> np.ndarray:
        """Return a curve at the middle of each chunk of a block."""
        centres = (np.arange(0, SR, chunk) + chunk / 2.0) / SR
        return np.interp((block + centres) * CR, np.arange(len(curve)), curve)

    @staticmethod
    def _edges(curve: np.ndarray | None, block: int) -> tuple[float, float]:
        """Return the value of a curve at the start and the end of a block."""
        if curve is None:
            return 1.0, 1.0
        top = len(curve) - 1.0
        first = float(np.interp(min(max(block * CR, 0.0), top), np.arange(len(curve)), curve))
        last = float(np.interp(min(max((block + 1) * CR, 0.0), top), np.arange(len(curve)), curve))
        return first, last

    # -- textures ----------------------------------------------------------------------------------

    def _loop(self, key: str, kind: str) -> np.ndarray:
        if key not in self._noise:
            rng = np.random.default_rng((self.seed + stable_hash(key)) % (2**63))
            n = 14 * SR
            if kind == "air":
                low, high = 500.0, 6500.0
            else:
                low, high = 40.0, 260.0
            stereo = np.stack(
                [dsp.band_noise(rng, n, low, high), dsp.band_noise(rng, n, low, high)], axis=1
            )
            cross = 2 * SR
            fade = np.linspace(0.0, np.pi / 2, cross)
            body = stereo[: n - cross].copy()
            body[:cross] = (
                body[:cross] * np.sin(fade)[:, None] + stereo[n - cross :] * np.cos(fade)[:, None]
            )
            self._noise[key] = body
        return self._noise[key]

    def _texture(self, texture: Texture, block: int) -> np.ndarray:
        if texture.kind in {"air", "room"}:
            loop = self._loop(texture.key, texture.kind)
            start = (block * SR) % len(loop)
            piece = loop[start : start + SR]
            if len(piece) < SR:
                piece = np.concatenate([piece, loop[: SR - len(piece)]])
            edges = self._edges(texture.level, block)
            wander = 0.75 + 0.25 * np.sin(2 * np.pi * block * 0.07 + (self.seed % 7))
            ramp = np.linspace(
                edges[0] * texture.gain, edges[1] * texture.gain, SR, dtype=np.float32
            )
            ramp = ramp * np.float32(wander)
            return piece * ramp[:, None]
        fade = self._ramp(texture.level, block)
        rng = np.random.default_rng((self.seed + block * 7919 + stable_hash(texture.key)) % (2**63))
        if texture.kind == "hiss":
            wave = dsp.noise(rng, SR) * texture.gain
            return np.stack([wave, np.roll(wave, 7)], axis=1) * fade[:, None]
        # vinyl: crackle, a few pops a second of random size, and a very quiet rumble of dust
        out = np.zeros((SR, 2), dtype=np.float32)
        count = int(rng.poisson(9.0))
        for _ in range(count):
            at = int(rng.integers(0, SR - 600))
            size = int(rng.integers(8, 90))
            amp = (
                float(rng.exponential(1.0)) ** 1.5
                * texture.gain
                * (1.0 if rng.random() < 0.97 else 3.0)
            )
            tick = rng.standard_normal(size) * np.exp(-np.arange(size) / max(size / 4.0, 1.0)) * amp
            side = rng.uniform(-0.6, 0.6)
            out[at : at + size, 0] += tick * (1 - side) * 0.5
            out[at : at + size, 1] += tick * (1 + side) * 0.5
        return out * fade[:, None]

    # -- one block ---------------------------------------------------------------------------------

    def step(self, head: int) -> np.ndarray | None:
        """Take the next block that the loop renders and return the block that is ready, which lags by `latency`."""
        return self.render_block(head)

    def render_block(self, block: int) -> np.ndarray:  # noqa: C901
        """Render the block at `block` seconds as float32 stereo, before the master."""
        score = self.score
        buses = self._bus(block)
        dry = np.zeros((SR, 2), dtype=np.float32)
        send_reverb = np.zeros((SR, 2), dtype=np.float32)
        send_echo = np.zeros((SR, 2), dtype=np.float32)
        for key, layer in score.layers.items():
            bus = buses.get(key)
            if bus is None:
                continue
            bus *= self._ramp(layer.presence, block, layer.gain)[:, None]
            if layer.cutoff is not None:
                bus = self.ladders[key].process(
                    bus,
                    self._chunks(layer.cutoff, block, self.tier.control_chunk),
                    self.tier.control_chunk,
                )
            if key in self.highpass:
                bus = self.highpass[key].process(bus, np.array([layer.highpass]), SR)
            dry += bus
            if layer.reverb:
                send_reverb += bus * np.float32(layer.reverb)
            if layer.echo:
                send_echo += bus * self._ramp(layer.echo_curve, block, layer.echo)[:, None]
        for texture in () if self.solo is not None else score.textures:
            tex = self._texture(texture, block)
            dry += tex
            send_reverb += tex * np.float32(0.3)
        if self.shimmer is not None:
            send_reverb += (self.shimmer.process(send_reverb) * np.float32(score.shimmer)).astype(
                np.float32
            )
        wet = self.reverb(np.ascontiguousarray(send_reverb.T), SR, reset=False).T
        space = self._ramp(score.space, block, REVERB_WET)
        out = dry + wet.astype(np.float32) * space[:, None]
        if self.has_echo:
            echo = self.echo(np.ascontiguousarray(send_echo.T), SR, reset=False).T
            out = out + echo.astype(np.float32) * np.float32(ECHO_WET)
        if score.wobble > 0.0 and self.tier.wobble:
            out = self._wobble(out, block)
        return np.asarray(out, dtype=np.float32)

    def _wobble(self, mix: np.ndarray, block: int) -> np.ndarray:
        """Tape wow and flutter over the whole mix: a few cents of drift, never a glide."""
        depth = self.score.wobble
        both = np.concatenate([self._wobble_tail, mix])
        base = len(self._wobble_tail)
        t = block + np.arange(SR) / SR
        delay = 40.0 + depth * (
            30.0 * np.sin(2 * np.pi * 0.47 * t)
            + 8.0 * np.sin(2 * np.pi * 0.113 * t + 1.0)
            + 0.9 * np.sin(2 * np.pi * 5.3 * t)
        )
        self._wobble_tail = both[-4096:].copy()
        out = np.empty_like(mix)
        pos = base + np.arange(SR) - delay
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0).astype(np.float32)[:, None]
        out[:] = both[i0] * (1.0 - frac) + both[i0 + 1] * frac
        return out

    # -- master ------------------------------------------------------------------------------------

    def level(self, out: np.ndarray, block: int) -> np.ndarray:
        """
        Hold the loudness of the mix at the level of the style.

        The score carries a gain that was estimated from its notes (`levels.feed_forward`), the same for every seek; it is
        applied first. What is left is a slow trim of at most 3 dB towards the target, from the mean square that
        was measured after it, which takes gain away faster than it gives it, so that it neither pumps nor lets a loud
        moment through.

        :param out: The block, before the master.
        :param block: Its index.
        """
        if self.score.gain_db is not None:
            edges = self._ramp(self.score.gain_db, block)
            out = out * (10.0 ** (edges / 20.0))[:, None].astype(np.float32)
        target = 10 ** (self.score.level_db / 10.0)
        ms = float(np.einsum("ij,ij->", out, out)) / out.size
        if self._ms is None:
            self._ms = ms
        else:
            rate = LEVEL_RISE_RATE if ms > self._ms else LEVEL_FALL_RATE
            self._ms += (ms - self._ms) * rate
        wanted = min(max(float(np.sqrt(target / max(self._ms, 1e-10))), LEVEL_MIN), LEVEL_MAX)
        if self._first:
            self._gain = 1.0
            self._first = False
        gain = min(
            max(wanted, self._gain / 10 ** (LEVEL_DOWN_DB / 20.0)),
            self._gain * 10 ** (LEVEL_UP_DB / 20.0),
        )
        ramp = np.linspace(self._gain, gain, len(out), dtype=np.float32)
        self._gain = gain
        return out * ramp[:, None]

    def finish(self, mix: np.ndarray, block: int, start_s: int) -> np.ndarray:
        """Apply the leveler, the limiter and the fades to a rendered block."""
        self._dc = 0.9 * self._dc + 0.1 * mix.mean(axis=0)
        out = self.level(mix - self._dc.astype(np.float32), block)
        if self.compressor is not None:
            out = self.compressor(np.ascontiguousarray(out.T), SR, reset=False).T.astype(np.float32)
        out = self.limiter.process(out)
        if 0 <= block - start_s < FADE_IN_SECONDS + 1:
            time = block + np.arange(len(out)) / SR
            out = out * smoothstep((time - start_s) / FADE_IN_SECONDS).astype(np.float32)[:, None]
        if self.score.length - block <= FADE_OUT_SECONDS + 1:
            time = block + np.arange(len(out)) / SR
            out = (
                out
                * smoothstep((self.score.length - time) / FADE_OUT_SECONDS).astype(np.float32)[
                    :, None
                ]
            )
        return np.asarray(out, dtype=np.float32)


def make_renderer(
    score: Score, tier: Quality, seed: str = "", solo: frozenset[str] | None = None
) -> Renderer:
    """Return the renderer that a score needs: Cinematic Glitch splices its buses, every other style plays them as they are."""
    if score.glitch is not None:
        from .glitch_render import GlitchRenderer  # noqa: PLC0415

        return GlitchRenderer(score, tier, seed, solo)
    return Renderer(score, tier, seed, solo)


def render_blocks(  # noqa: PLR0913, PLR0917
    score: Score,
    tier: Quality,
    seed: str = "",
    start_s: int = 0,
    stats: dict[str, float] | None = None,
    end_s: int | None = None,
    solo: frozenset[str] | None = None,
) -> Generator[np.ndarray]:
    """
    Yield the audio of a score as float blocks of one second, shape (44100, 2).

    :param score: What to play.
    :param tier: The quality tier.
    :param seed: Seeds the textures.
    :param start_s: The second to start at; a few seconds before it are rendered and dropped.
    :param stats: When given, filled with what the limiter did.
    :param end_s: Stop before this second (the whole piece by default).
    :param solo: Only these layers play.
    """
    renderer = make_renderer(score, tier, seed, solo)
    stop = score.length if end_s is None else min(end_s, score.length)
    lag = renderer.latency
    for head in range(max(0, start_s - WARMUP_SECONDS), stop + lag):
        mix = renderer.step(head)
        block = head - lag
        if mix is None:
            continue
        done = renderer.finish(mix, block, start_s)
        if block >= start_s:
            yield done
    if stats is not None:
        limiter = renderer.limiter
        stats["mean_reduction_db"] = limiter.mean_reduction_db
        stats["max_reduction_db"] = limiter.reduction_max
        stats["clip_share"] = limiter.clip_share


_DITHER = np.random.default_rng(16).random((SR * 2 + 4096, 2))


def to_pcm16(block: np.ndarray, block_index: int = 0) -> bytes:
    """
    Convert a float block to 16 bit little endian stereo PCM with a whisper of dither.

    :param block: Float samples between -1 and 1, shape (n, 2).
    :param block_index: Moves the dither, so that it is the same bytes every time.
    """
    offset = (block_index * 997) % 4096
    noise = _DITHER[offset : offset + len(block)] - _DITHER[offset + 1 : offset + 1 + len(block)]
    scaled = np.clip(np.round((block + noise / 32768.0) * 32767.0), -32768, 32767)
    return scaled.astype("<i2").tobytes()


def render_pcm(score: Score, tier: Quality, seed: str = "", start_s: int = 0) -> Generator[bytes]:
    """
    Yield the audio of a score as 16 bit stereo PCM, one second per chunk.

    :param score: What to play.
    :param tier: The quality tier.
    :param seed: Seeds the textures.
    :param start_s: The second to start at.
    """
    for offset, block in enumerate(render_blocks(score, tier, seed, start_s)):
        yield to_pcm16(block, start_s + offset)


def render_all(  # noqa: PLR0913, PLR0917
    score: Score,
    tier: Quality,
    seed: str = "",
    stats: dict[str, float] | None = None,
    start_s: int = 0,
    end_s: int | None = None,
    solo: frozenset[str] | None = None,
) -> np.ndarray:
    """Render a whole score (or the seconds from `start_s` to `end_s`) into one array, shape (samples, 2)."""
    blocks = list(render_blocks(score, tier, seed, start_s, stats, end_s, solo))
    return np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.float32)


__all__ = [
    "Layer",
    "Renderer",
    "make_renderer",
    "render_all",
    "render_blocks",
    "render_pcm",
    "to_pcm16",
]
