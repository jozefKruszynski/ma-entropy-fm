"""
The render loop of Entropy FM: a score in, audio out, one second at a time.

Notes of the score are mixed into the buses of their layers from the cache of made notes (`voices`, `drums`),
at their absolute positions, so any second can be rendered without the ones before it. The buses go through
their filters (cutoffs that the house moves), into a shared reverb and echo (and a layer that asks for it into a long
hall of its own), and the textures are added. The master is a slow leveler that holds the loudness of the style, and a
look-ahead peak limiter with a ceiling of -2.4 dBFS that should barely work. A seek renders a few seconds ahead of the
start to warm up filters and reverb.

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
from .drums import DRUM_PREFIX, render_hit
from .dsp import (
    AMPLITUDE_DB_DIVISOR,
    LADDER_HP24,
    LADDER_LP12,
    LADDER_LP24,
    SR,
    Shimmer,
    StreamLadder,
)
from .levels import POWER_DB_FACTOR
from .limiter import Limiter
from .quality import REVERB_FREEVERB, REVERB_ROOM
from .score import CURVE_RATE, Layer, Note, Score, Texture
from .stretch import is_wash, render_wash
from .synth import smoothstep
from .timeline import SEED_MODULUS, stable_hash
from .voices import VOICES, render_note

if TYPE_CHECKING:
    from collections.abc import Generator

    from .quality import Quality

WARMUP_SECONDS = 8
MASTER_FADE_IN_SECONDS = 1.5
MASTER_FADE_OUT_SECONDS = 2.5
# the longest that a note rings after its start, in seconds, before the margin below is added to it
REACH_SECONDS = 36.0
CEILING_DB = -2.4
# the leveler follows a rise of the level at once and a fall slowly; it takes gain away faster than it gives it
LEVEL_DOWN_DB = 1.0
LEVEL_UP_DB = 0.4
LEVEL_RISE_RATE = 0.6
LEVEL_FALL_RATE = 0.08
# the trim of the leveler stays within this many dB of no trim
LEVEL_RANGE_DB = 2.0
LEVEL_MIN = 10 ** (-LEVEL_RANGE_DB / AMPLITUDE_DB_DIVISOR)
LEVEL_MAX = 10 ** (LEVEL_RANGE_DB / AMPLITUDE_DB_DIVISOR)
# the mean square of the level of a score is floored to this
MIN_MEAN_SQUARE = 1e-10
# how loud the reverb return is against the dry mix at a space of 1
REVERB_WET = 0.9
ECHO_WET = 0.6
# the long hall that a layer can ask for (`Layer.hall`): a dark cinematic hall, with a gap before it starts, which
# like the room keeps the low end out of the tail and is a little darker at the top
LAYER_HALL_SECONDS = 7.0
HALL_PREDELAY_SECONDS = 0.05
HALL_SEED = 17
HALL_HIGHPASS_HZ = 170.0
LAYER_HALL_LOWPASS_HZ = 4800.0
# on a tier without the hall the send of a layer goes to the shared reverb, this many times as large (the Freeverb of
# the cheapest tier is some thirty times hotter than a convolution)
HALL_FALLBACK_SEND = 0.06
# the seconds that the tail of the hall is kept beyond its own length
HALL_IDLE_MARGIN_SECONDS = 2
# the seconds after the last send until the tail of the hall is gone and it is not run
HALL_IDLE_SECONDS = int(LAYER_HALL_SECONDS + HALL_PREDELAY_SECONDS) + HALL_IDLE_MARGIN_SECONDS
# the cached room impulses
IMPULSE_CACHE_SIZE = 8
# the decay of an impulse is 60 dB (a natural log of 1000) down by its end, and it is darker as it goes
IMPULSE_DECAY_EXPONENT = 6.9
IMPULSE_LOWPASS_HZ = 9000.0
IMPULSE_DARKEN_HZ = 1400.0
# the darkening is complete this share of the way through the impulse
IMPULSE_DARKEN_SPAN = 0.5
IMPULSE_FADE_IN_SECONDS = 0.03
# the shared reverb: a high-pass under it, the settings of the Freeverb, and the length of the room of the convolutions
REVERB_HIGHPASS_HZ = 180.0
FREEVERB_ROOM_SIZE = 0.82
FREEVERB_DAMPING = 0.55
ROOM_REVERB_SECONDS = 2.6
RICH_REVERB_SECONDS = 5.0
REVERB_SEED = 11
ECHO_LOWPASS_HZ = 3600.0
# the compressor of the styles with drums
COMPRESSOR_RATIO = 3.0
COMPRESSOR_ATTACK_MS = 4.0
COMPRESSOR_RELEASE_MS = 110.0
SEED_TAG_TEXTURES = "textures"
# the share of a texture that goes to the shared reverb
TEXTURE_REVERB_SEND = 0.3
# the wobble keeps the end of a block, in samples, to read from the next
WOBBLE_TAIL_SAMPLES = 4096
# the estimate of the DC of the mix keeps this share of itself, and takes this share of the mean of each block
DC_KEEP = 0.9
DC_TAKE = 0.1
# how long after its start a hit rings at the most, in seconds, and how much longer than its length and release a note does
DRUM_REACH_SECONDS = 2.5
WASH_REACH_PAD_SECONDS = 1.0
RELEASE_REACH_FACTOR = 4.0
REACH_PAD_SECONDS = 1.0
# a note rings at most this long more than `REACH_SECONDS`, and the notes that are looked at start up to
# `REACH_SECONDS` plus this before a block
REACH_MARGIN_SECONDS = 8.0
TEXTURE_ROOM = "room"
# the room rumble: the length of its loop and of the crossfade of its ends, its band, and the slow drift of its level
ROOM_LOOP_SECONDS = 14
ROOM_LOOP_CROSS_SECONDS = 2
ROOM_BAND_HZ = (40.0, 260.0)
ROOM_WANDER_BASE = 0.75
ROOM_WANDER_DEPTH = 0.25
ROOM_WANDER_RATE = 0.07
ROOM_WANDER_SEED_MODULUS = 7
# the vinyl: the seed of a block, the pops a second, where they may fall (away from the end), their sizes in samples,
# the share of the loud ones and their gain, the decay of a pop in sizes and how far to the side they sit
TEXTURE_BLOCK_SEED_PRIME = 7919
VINYL_POPS_PER_SECOND = 9.0
VINYL_POP_END_MARGIN = 600
VINYL_POP_SIZES = (8, 90)
VINYL_LOUD_SHARE = 0.97
VINYL_LOUD_GAIN = 3.0
VINYL_POP_CURVE = 1.5
VINYL_POP_DECAY_SIZES = 4.0
VINYL_POP_PAN = 0.6
# the tape wobble of the mix: a base delay and three movements (wow, a slow drift, flutter), each a swing in samples at a rate in Hz
WOBBLE_BASE_DELAY = 40.0
WOBBLE_WOW_DEPTH = 30.0
WOBBLE_WOW_RATE = 0.47
WOBBLE_DRIFT_DEPTH = 8.0
WOBBLE_DRIFT_RATE = 0.113
WOBBLE_DRIFT_PHASE = 1.0
WOBBLE_FLUTTER_DEPTH = 0.9
WOBBLE_FLUTTER_RATE = 5.3
# the dither: a seed, and the period of its offsets in samples
DITHER_SEED = 16
DITHER_PERIOD = 4096
# the table of dither is two seconds long and a period more, so that an offset never runs off its end
DITHER_TABLE_SAMPLES = SR * 2 + DITHER_PERIOD
DITHER_OFFSET_PRIME = 997
PCM_FULL_SCALE = 32768
PCM_MAX = 32767
PCM_MIN = -32768


@lru_cache(maxsize=IMPULSE_CACHE_SIZE)
def impulse(seconds: float, seed: int) -> np.ndarray:
    """Make a stereo room: noise that decays, darker as it goes, a little different left and right."""
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros((2, n), dtype=np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    for side in range(2):
        spec = np.fft.rfft(rng.standard_normal(n))
        t = np.arange(n) / SR
        decay = np.exp(-IMPULSE_DECAY_EXPONENT * t / seconds)
        wave = np.fft.irfft(spec * (1.0 / (1.0 + (freqs / IMPULSE_LOWPASS_HZ) ** 2)), n)
        mixed = wave * decay
        # the darkening: blend a low-passed copy in as time goes on
        low = np.fft.irfft(np.fft.rfft(wave) / (1.0 + (freqs / IMPULSE_DARKEN_HZ) ** 2), n) * decay
        share = np.clip(t / (seconds * IMPULSE_DARKEN_SPAN), 0.0, 1.0)
        out[side] = (mixed * (1 - share) + low * share).astype(np.float32)
    fade = int(IMPULSE_FADE_IN_SECONDS * SR)
    out[:, :fade] *= np.linspace(0.0, 1.0, fade)
    out /= np.sqrt(np.sum(out.astype(np.float64) ** 2, axis=1))[:, None].astype(np.float32)
    return out


@lru_cache(maxsize=1)
def hall_impulse() -> np.ndarray:
    """Make the stereo hall: a long dark room (`impulse`) that starts after a pre-delay."""
    gap = np.zeros((2, int(HALL_PREDELAY_SECONDS * SR)), dtype=np.float32)
    return np.concatenate([gap, impulse(LAYER_HALL_SECONDS, HALL_SEED)], axis=1)


def hall_board() -> Pedalboard:
    """Return a new long hall; a board keeps the tail of what it was fed, so every renderer needs its own."""
    return Pedalboard(
        [
            HighpassFilter(HALL_HIGHPASS_HZ),
            Convolution(hall_impulse(), mix=1.0, sample_rate=SR),
            LowpassFilter(LAYER_HALL_LOWPASS_HZ),
        ]
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
        self.seed = stable_hash(seed, SEED_TAG_TEXTURES)
        self.notes = sorted(score.notes, key=lambda n: (n.t, n.layer, n.voice, n.midi))
        self.starts = [n.t for n in self.notes]
        self.ladders: dict[str, StreamLadder] = {}
        for key, layer in score.layers.items():
            if layer.cutoff is not None:
                self.ladders[key] = StreamLadder(
                    layer.resonance, 1.0, LADDER_LP24 if tier.chorus else LADDER_LP12
                )
        self.highpass = {
            key: StreamLadder(0.0, 1.0, LADDER_HP24)
            for key, layer in score.layers.items()
            if layer.highpass > 0.0
        }
        self.reverb = _reverb_board(tier)
        # the hall is made only where a layer that plays asks for it, and not at all on a tier that cannot afford it
        used = {note.layer for note in score.notes}
        self.hall = (
            hall_board()
            if tier.layer_hall
            and any(layer.hall > 0.0 and key in used for key, layer in score.layers.items())
            else None
        )
        self._hall_idle = HALL_IDLE_SECONDS
        self.echo = Pedalboard(
            [
                Delay(delay_seconds=score.echo_seconds, feedback=score.echo_feedback, mix=1.0),
                LowpassFilter(ECHO_LOWPASS_HZ),
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
                        threshold_db=score.compress_db,
                        ratio=COMPRESSOR_RATIO,
                        attack_ms=COMPRESSOR_ATTACK_MS,
                        release_ms=COMPRESSOR_RELEASE_MS,
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
        self._wobble_tail = np.zeros((WOBBLE_TAIL_SAMPLES, 2), dtype=np.float32)
        # the waves of the notes that are sounding, held so that the cache cannot make a long note again every second
        self._active: dict[int, tuple[float, np.ndarray]] = {}
        self._finished: set[int] = set()

    # -- one block ---------------------------------------------------------------------------------

    def step(self, head: int) -> np.ndarray | None:
        """Take the next block that the loop renders and return the block that is ready, which lags by `latency`."""
        return self.render_block(head)

    def render_block(self, block: int) -> np.ndarray:
        """Render the block at `block` seconds as float32 stereo, before the master."""
        score = self.score
        buses = self._bus(block)
        dry = np.zeros((SR, 2), dtype=np.float32)
        send_reverb = np.zeros((SR, 2), dtype=np.float32)
        send_hall = np.zeros((SR, 2), dtype=np.float32)
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
            self._send(layer, bus, send_reverb, send_hall)
            if layer.echo:
                send_echo += bus * self._ramp(layer.echo_curve, block, layer.echo)[:, None]
        for texture in () if self.solo is not None else score.textures:
            tex = self._texture(texture, block)
            dry += tex
            send_reverb += tex * np.float32(TEXTURE_REVERB_SEND)
        if self.shimmer is not None:
            send_reverb += (self.shimmer.process(send_reverb) * np.float32(score.shimmer)).astype(
                np.float32
            )
        wet = self.reverb(np.ascontiguousarray(send_reverb.T), SR, reset=False).T
        space = self._ramp(score.space, block, REVERB_WET)
        out = self._hall_return(dry + wet.astype(np.float32) * space[:, None], send_hall, space)
        if self.has_echo:
            echo = self.echo(np.ascontiguousarray(send_echo.T), SR, reset=False).T
            out = out + echo.astype(np.float32) * np.float32(ECHO_WET)
        if score.wobble > 0.0 and self.tier.wobble:
            out = self._wobble(out, block)
        return np.asarray(out, dtype=np.float32)

    # -- master ------------------------------------------------------------------------------------

    def level(self, out: np.ndarray, block: int) -> np.ndarray:
        """
        Hold the loudness of the mix at the level of the style.

        The score carries a gain that was estimated from its notes (`levels.feed_forward`), the same for every seek; it is
        applied first. What is left is a slow trim of at most 2 dB towards the target, from the mean square that
        was measured after it, which takes gain away faster than it gives it, so that it neither pumps nor lets a loud
        moment through.

        :param out: The block, before the master.
        :param block: Its index.
        """
        if self.score.gain_db is not None:
            edges = self._ramp(self.score.gain_db, block)
            out = out * (10.0 ** (edges / AMPLITUDE_DB_DIVISOR))[:, None].astype(np.float32)
        target = 10 ** (self.score.level_db / POWER_DB_FACTOR)
        ms = float(np.einsum("ij,ij->", out, out)) / out.size
        if self._ms is None:
            self._ms = ms
        else:
            rate = LEVEL_RISE_RATE if ms > self._ms else LEVEL_FALL_RATE
            self._ms += (ms - self._ms) * rate
        wanted = min(
            max(float(np.sqrt(target / max(self._ms, MIN_MEAN_SQUARE))), LEVEL_MIN), LEVEL_MAX
        )
        if self._first:
            self._gain = 1.0
            self._first = False
        gain = min(
            max(wanted, self._gain / 10 ** (LEVEL_DOWN_DB / AMPLITUDE_DB_DIVISOR)),
            self._gain * 10 ** (LEVEL_UP_DB / AMPLITUDE_DB_DIVISOR),
        )
        ramp = np.linspace(self._gain, gain, len(out), dtype=np.float32)
        self._gain = gain
        return out * ramp[:, None]

    def finish(self, mix: np.ndarray, block: int, start_s: int) -> np.ndarray:
        """Apply the leveler, the limiter and the fades to a rendered block."""
        self._dc = DC_KEEP * self._dc + DC_TAKE * mix.mean(axis=0)
        out = self.level(mix - self._dc.astype(np.float32), block)
        if self.compressor is not None:
            out = self.compressor(np.ascontiguousarray(out.T), SR, reset=False).T.astype(np.float32)
        out = self.limiter.process(out)
        if 0 <= block - start_s < MASTER_FADE_IN_SECONDS + 1:
            time = block + np.arange(len(out)) / SR
            out = (
                out
                * smoothstep((time - start_s) / MASTER_FADE_IN_SECONDS).astype(np.float32)[:, None]
            )
        if self.score.length - block <= MASTER_FADE_OUT_SECONDS + 1:
            time = block + np.arange(len(out)) / SR
            out = (
                out
                * smoothstep((self.score.length - time) / MASTER_FADE_OUT_SECONDS).astype(
                    np.float32
                )[:, None]
            )
        return np.asarray(out, dtype=np.float32)

    def _wave(self, note: Note) -> np.ndarray:
        if note.voice.startswith(DRUM_PREFIX):
            return render_hit(note.voice[len(DRUM_PREFIX) :], note.vel, note.patch)
        if is_wash(note.voice):
            return render_wash(note.voice, note.midi, note.dur, note.vel, self.tier.cache_mb)
        return render_note(
            note.voice, note.midi, note.dur, note.vel, self.tier, note.patch, note.vib
        )

    def _reach(self, note: Note) -> float:
        if note.voice.startswith(DRUM_PREFIX):
            return DRUM_REACH_SECONDS
        if is_wash(note.voice):
            return note.dur + WASH_REACH_PAD_SECONDS
        return min(
            note.dur + VOICES[note.voice].release * RELEASE_REACH_FACTOR + REACH_PAD_SECONDS,
            REACH_SECONDS + REACH_MARGIN_SECONDS,
        )

    def _bus(self, block: int) -> dict[str, np.ndarray]:
        start, end = float(block), float(block + 1)
        buses: dict[str, np.ndarray] = {}
        low = bisect_left(self.starts, start - REACH_SECONDS - REACH_MARGIN_SECONDS)
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

    def _send(self, layer: Layer, bus: np.ndarray, room: np.ndarray, hall: np.ndarray) -> None:
        """Add a bus to the sends of the shared reverb and of the hall, as the layer and the tier ask."""
        amount = layer.reverb
        if layer.hall > 0.0:
            if self.tier.layer_hall:
                hall += bus * np.float32(layer.hall)
            else:
                amount += layer.hall * HALL_FALLBACK_SEND
        if amount:
            room += bus * np.float32(amount)

    def _hall_return(self, out: np.ndarray, hall: np.ndarray, space: np.ndarray) -> np.ndarray:
        """Return the mix with the hall added to it, as wet as the shared reverb is."""
        if self.hall is None:
            return out
        self._hall_idle = 0 if hall.any() else self._hall_idle + 1
        if self._hall_idle >= HALL_IDLE_SECONDS:
            return out
        wet = self.hall(np.ascontiguousarray(hall.T), SR, reset=False).T
        return out + wet.astype(np.float32) * space[:, None]

    @staticmethod
    def _ramp(curve: np.ndarray | None, block: int, scale: float = 1.0) -> np.ndarray:
        """Return a curve over a block as a straight ramp between its values at the two edges (it moves slowly)."""
        if curve is None:
            return np.full(SR, scale, dtype=np.float32)
        first, last = Renderer._edges(curve, block)
        return np.linspace(first * scale, last * scale, SR, dtype=np.float32)

    @staticmethod
    def _chunks(curve: np.ndarray, block: int, chunk: int) -> np.ndarray:
        """Return a curve at the middle of each chunk of a block."""
        centres = (np.arange(0, SR, chunk) + chunk / 2.0) / SR
        return np.interp((block + centres) * CURVE_RATE, np.arange(len(curve)), curve)

    @staticmethod
    def _edges(curve: np.ndarray | None, block: int) -> tuple[float, float]:
        """Return the value of a curve at the start and the end of a block."""
        if curve is None:
            return 1.0, 1.0
        top = len(curve) - 1.0
        first = float(
            np.interp(min(max(block * CURVE_RATE, 0.0), top), np.arange(len(curve)), curve)
        )
        last = float(
            np.interp(min(max((block + 1) * CURVE_RATE, 0.0), top), np.arange(len(curve)), curve)
        )
        return first, last

    def _loop(self, key: str) -> np.ndarray:
        if key not in self._noise:
            rng = np.random.default_rng((self.seed + stable_hash(key)) % SEED_MODULUS)
            n = ROOM_LOOP_SECONDS * SR
            # the room is a low rumble, nothing above 260 Hz
            low, high = ROOM_BAND_HZ
            stereo = np.stack(
                [dsp.band_noise(rng, n, low, high), dsp.band_noise(rng, n, low, high)], axis=1
            )
            cross = ROOM_LOOP_CROSS_SECONDS * SR
            fade = np.linspace(0.0, np.pi / 2, cross)
            body = stereo[: n - cross].copy()
            body[:cross] = (
                body[:cross] * np.sin(fade)[:, None] + stereo[n - cross :] * np.cos(fade)[:, None]
            )
            self._noise[key] = body
        return self._noise[key]

    def _texture(self, texture: Texture, block: int) -> np.ndarray:
        if texture.kind == TEXTURE_ROOM:
            loop = self._loop(texture.key)
            start = (block * SR) % len(loop)
            piece = loop[start : start + SR]
            if len(piece) < SR:
                piece = np.concatenate([piece, loop[: SR - len(piece)]])
            edges = self._edges(texture.level, block)
            wander = ROOM_WANDER_BASE + ROOM_WANDER_DEPTH * np.sin(
                2 * np.pi * block * ROOM_WANDER_RATE + (self.seed % ROOM_WANDER_SEED_MODULUS)
            )
            ramp = np.linspace(
                edges[0] * texture.gain, edges[1] * texture.gain, SR, dtype=np.float32
            )
            ramp = ramp * np.float32(wander)
            return piece * ramp[:, None]
        fade = self._ramp(texture.level, block)
        rng = np.random.default_rng(
            (self.seed + block * TEXTURE_BLOCK_SEED_PRIME + stable_hash(texture.key)) % SEED_MODULUS
        )
        # vinyl: crackle, a few pops a second of random size, and a very quiet rumble of dust
        out = np.zeros((SR, 2), dtype=np.float32)
        count = int(rng.poisson(VINYL_POPS_PER_SECOND))
        for _ in range(count):
            at = int(rng.integers(0, SR - VINYL_POP_END_MARGIN))
            size = int(rng.integers(*VINYL_POP_SIZES))
            amp = (
                float(rng.exponential(1.0)) ** VINYL_POP_CURVE
                * texture.gain
                * (1.0 if rng.random() < VINYL_LOUD_SHARE else VINYL_LOUD_GAIN)
            )
            tick = (
                rng.standard_normal(size)
                * np.exp(-np.arange(size) / max(size / VINYL_POP_DECAY_SIZES, 1.0))
                * amp
            )
            side = rng.uniform(-VINYL_POP_PAN, VINYL_POP_PAN)
            out[at : at + size, 0] += tick * (1 - side) * 0.5
            out[at : at + size, 1] += tick * (1 + side) * 0.5
        return out * fade[:, None]

    def _wobble(self, mix: np.ndarray, block: int) -> np.ndarray:
        """Tape wow and flutter over the whole mix: a few cents of drift, never a glide."""
        depth = self.score.wobble
        both = np.concatenate([self._wobble_tail, mix])
        base = len(self._wobble_tail)
        t = block + np.arange(SR) / SR
        delay = WOBBLE_BASE_DELAY + depth * (
            WOBBLE_WOW_DEPTH * np.sin(2 * np.pi * WOBBLE_WOW_RATE * t)
            + WOBBLE_DRIFT_DEPTH * np.sin(2 * np.pi * WOBBLE_DRIFT_RATE * t + WOBBLE_DRIFT_PHASE)
            + WOBBLE_FLUTTER_DEPTH * np.sin(2 * np.pi * WOBBLE_FLUTTER_RATE * t)
        )
        self._wobble_tail = both[-WOBBLE_TAIL_SAMPLES:].copy()
        out = np.empty_like(mix)
        pos = base + np.arange(SR) - delay
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0).astype(np.float32)[:, None]
        out[:] = both[i0] * (1.0 - frac) + both[i0 + 1] * frac
        return out


def make_renderer(
    score: Score, tier: Quality, seed: str = "", solo: frozenset[str] | None = None
) -> Renderer:
    """Return the renderer that a score needs: Cinematic Glitch splices its buses, every other style plays them as they are."""
    if score.glitch is not None:
        from .glitch_render import GlitchRenderer  # noqa: PLC0415

        return GlitchRenderer(score, tier, seed, solo)
    return Renderer(score, tier, seed, solo)


def render_blocks(
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


def to_pcm16(block: np.ndarray, block_index: int = 0) -> bytes:
    """
    Convert a float block to 16 bit little endian stereo PCM with a whisper of dither.

    :param block: Float samples between -1 and 1, shape (n, 2).
    :param block_index: Moves the dither, so that it is the same bytes every time.
    """
    offset = (block_index * DITHER_OFFSET_PRIME) % DITHER_PERIOD
    noise = _DITHER[offset : offset + len(block)] - _DITHER[offset + 1 : offset + 1 + len(block)]
    scaled = np.clip(np.round((block + noise / PCM_FULL_SCALE) * PCM_MAX), PCM_MIN, PCM_MAX)
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


def render_all(
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


def _reverb_board(tier: Quality) -> Pedalboard:
    if tier.reverb == REVERB_FREEVERB:
        return Pedalboard(
            [
                HighpassFilter(REVERB_HIGHPASS_HZ),
                Reverb(
                    room_size=FREEVERB_ROOM_SIZE,
                    damping=FREEVERB_DAMPING,
                    wet_level=1.0,
                    dry_level=0.0,
                    width=1.0,
                ),
            ]
        )
    seconds = ROOM_REVERB_SECONDS if tier.reverb == REVERB_ROOM else RICH_REVERB_SECONDS
    return Pedalboard(
        [
            HighpassFilter(REVERB_HIGHPASS_HZ),
            Convolution(impulse(seconds, REVERB_SEED), mix=1.0, sample_rate=SR),
        ]
    )


_DITHER = np.random.default_rng(DITHER_SEED).random((DITHER_TABLE_SAMPLES, 2))


__all__ = [
    "Layer",
    "Renderer",
    "make_renderer",
    "render_all",
    "render_blocks",
    "render_pcm",
    "to_pcm16",
]
