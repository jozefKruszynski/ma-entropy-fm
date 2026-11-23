"""
The render of Cinematic Glitch: the buses of the style, spliced, ducked from the kick, with a drum bus of its own.

The other styles play their buses as they are. This style edits them: a stutter, a tape stop, a freeze is a splice of
what a bus played a moment before and a moment after, so it needs the audio of a few seconds ahead. The renderer
therefore runs three blocks behind the loop of `render.render_blocks`: it makes the dry buses of a block (notes,
presence, filters), keeps them for a few seconds, and the output of three blocks earlier is made from them. An edit
reads its stretch of a bus once, when the first block that it touches is made, and keeps the result until it is over.
Edits never overlap on a bus (the composer sees to it), so each is a function of the unedited audio and nothing else,
and a seek into the piece sounds like the continuous render.

From the edited buses of a block:

* the bass, the pads, the strings, the washes and the stabs are ducked from the kicks (the kicks are in the score);
* the kick, the snare and the hats make the drum bus: the snare gets a parallel compressor and a short dark plate, the
  sum goes through a glue compressor (-16 dB, 3:1, 10/140 ms) and a little tanh, which at this level is as clean as the
  oversampled one that made the hits (-73 dB apart) and, unlike it, has no seam from block to block;
* the layers go to a hall of 4.5 seconds through the sends of the layers, and a layer that asks for it (the trailer piano)
  also to a longer, darker one of its own (`render.hall_board`).

The loudness is held by the gain of the score alone, so that the night is as quiet as its arrangement and the drops
as loud: the leveler of the other styles would lift the one and press the other.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING

import numpy as np
from pedalboard import (  # type: ignore[attr-defined]
    Compressor,
    Convolution,
    LadderFilter,
    Pedalboard,
)

from . import glitch, glitch_drums, glitch_voices
from .dsp import AMPLITUDE_DB_DIVISOR, LADDER_HP24, SR, StreamLadder
from .glitch import GLITCH_HALL_LOWPASS_HZ, GLITCH_HALL_SECONDS
from .glitch_voices import LIGHT_MAX_LAYERS
from .render import HALL_HIGHPASS_HZ, REVERB_SEED, REVERB_WET, Renderer, impulse
from .score import Note

if TYPE_CHECKING:
    from .quality import Quality
    from .score import Score

# the blocks between the dry buses and the output: the longest edit (2.6 s) plus the block that it starts in
LATENCY = 3
# the buses are kept this many blocks behind the output, for the edits that read what came before
KEEP = 4
PARALLEL_AMOUNT = 0.6
PLATE_AMOUNT = 0.35
HAT_GAIN = 0.8
MASTER_HP_HZ = 26.0
# the parallel compressor of the snare: threshold in dB, ratio, attack and release in ms
PARALLEL_THRESHOLD_DB = -28.0
PARALLEL_RATIO = 10.0
PARALLEL_ATTACK_MS = 0.6
PARALLEL_RELEASE_MS = 80.0
# the glue compressor of the drum bus
GLUE_THRESHOLD_DB = -16.0
GLUE_RATIO = 3.0
GLUE_ATTACK_MS = 10.0
GLUE_RELEASE_MS = 140.0
GLUE_DRIVE = 1.2
# the drum bus is silent after it has been quiet for more than this many blocks, and starts out as long quiet as this
DRUM_QUIET_BLOCKS = 1
DRUM_QUIET_START = 99
# an edit is read from this far before its start and after its end (seconds), and the piece that is kept begins and ends this far out
EDIT_READ_LEAD = 0.05
EDIT_READ_TAIL = 0.1
EDIT_PIECE_LEAD = 0.03
EDIT_PIECE_TAIL = 0.05
EDIT_REACH_PAD = 0.2
# a duck is felt a little over the time constants of its recovery, as `glitch_drums` has them
DUCK_REACH_PAD = 0.01


class GlitchRenderer(Renderer):
    """Renders a score that carries a `Glitch`: the buses are spliced before they are mixed."""

    latency = LATENCY

    def __init__(
        self, score: Score, tier: Quality, seed: str = "", solo: frozenset[str] | None = None
    ) -> None:
        """
        Prepare the buses, the drum bus and the hall.

        :param score: What to play; `score.glitch` must be set.
        :param tier: The quality tier.
        :param seed: Seeds the textures (there are none).
        :param solo: When given, only these layers play.
        """
        super().__init__(score, tier, seed, solo)
        assert score.glitch is not None
        self.setup = score.glitch
        glitch_voices.prepare(score.notes, tier)
        self.light = tier.max_layers <= LIGHT_MAX_LAYERS
        self.edits = sorted(self.setup.edits, key=lambda e: e.t)
        self._edit_starts = [e.t for e in self.edits]
        self._edit_reach = max((e.dur for e in self.edits), default=0.0) + EDIT_REACH_PAD
        self._kick_times = [t for t, _ in self.setup.kicks]
        self._raw: dict[str, dict[int, np.ndarray]] = {key: {} for key in score.layers}
        self._results: dict[tuple[int, str], tuple[int, np.ndarray] | None] = {}
        self._origin: int | None = None
        if not self.light:
            self.reverb = Pedalboard(
                [
                    Convolution(impulse(GLITCH_HALL_SECONDS, REVERB_SEED), mix=1.0, sample_rate=SR),
                    LadderFilter(mode=LadderFilter.Mode.HPF24, cutoff_hz=HALL_HIGHPASS_HZ),
                    LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=GLITCH_HALL_LOWPASS_HZ),
                ]
            )
            self._parallel = Pedalboard(
                [
                    Compressor(
                        threshold_db=PARALLEL_THRESHOLD_DB,
                        ratio=PARALLEL_RATIO,
                        attack_ms=PARALLEL_ATTACK_MS,
                        release_ms=PARALLEL_RELEASE_MS,
                    )
                ]
            )
            self._plate = glitch_drums.plate_board()
        self._glue = Pedalboard(
            [
                Compressor(
                    threshold_db=GLUE_THRESHOLD_DB,
                    ratio=GLUE_RATIO,
                    attack_ms=GLUE_ATTACK_MS,
                    release_ms=GLUE_RELEASE_MS,
                )
            ]
        )
        self._drum_quiet = DRUM_QUIET_START
        self._hp = StreamLadder(0.0, 1.0, LADDER_HP24)

    # -- the loop ----------------------------------------------------------------------------------

    def step(self, head: int) -> np.ndarray | None:
        """
        Make the dry buses of a block and return the mix of the block that is `LATENCY` behind it.

        :param head: The block whose notes are played into the buses now.
        """
        if self._origin is None:
            self._origin = head
        buses = self._bus(head)
        for key, layer in self.score.layers.items():
            bus = buses.get(key)
            if bus is None:
                continue
            bus *= self._ramp(layer.presence, head, layer.gain)[:, None]
            if layer.cutoff is not None:
                bus = self.ladders[key].process(
                    bus,
                    self._chunks(layer.cutoff, head, self.tier.control_chunk),
                    self.tier.control_chunk,
                )
            if key in self.highpass:
                bus = self.highpass[key].process(bus, np.array([layer.highpass]), SR)
            self._raw[key][head] = bus
        block = head - LATENCY
        if block < self._origin:
            return None
        mix = self._mix(block)
        for blocks in self._raw.values():
            for old in [b for b in blocks if b < block - KEEP]:
                del blocks[old]
        for index in [k for k in self._results if self._over(k[0], block)]:
            del self._results[index]
        return mix

    # -- edits -------------------------------------------------------------------------------------

    # -- the mix -----------------------------------------------------------------------------------

    def level(self, out: np.ndarray, block: int) -> np.ndarray:
        """Apply the gain of the score and nothing else: the arrangement is the dynamics of this style."""
        if self.score.gain_db is not None:
            edges = self._ramp(self.score.gain_db, block)
            out = out * (10.0 ** (edges / AMPLITUDE_DB_DIVISOR))[:, None].astype(np.float32)
        return out

    def _wave(self, note: Note) -> np.ndarray:
        if glitch_voices.is_glitch(note.voice):
            return glitch_voices.render(note, self.tier, self.setup)
        return super()._wave(note)

    def _reach(self, note: Note) -> float:
        if glitch_voices.is_glitch(note.voice):
            return glitch_voices.reach(note)
        return super()._reach(note)

    def _over(self, index: int, block: int) -> bool:
        edit = self.edits[index]
        return edit.t + edit.dur + EDIT_READ_TAIL < block

    def _result(self, index: int, key: str) -> tuple[int, np.ndarray] | None:
        """Return the audio of a bus over an edit, as (first sample, samples), made once from the unedited bus."""
        found = (index, key)
        if found in self._results:
            return self._results[found]
        edit = self.edits[index]
        first = math.floor(edit.t - edit.back - EDIT_READ_LEAD)
        last = math.floor(edit.t + edit.dur + EDIT_READ_TAIL)
        raws = self._raw[key]
        if not any(b in raws for b in range(first, last + 1)):
            self._results[found] = None
            return None
        buf = np.zeros(((last - first + 1) * SR, 2), np.float32)
        for b in range(first, last + 1):
            raw = raws.get(b)
            if raw is not None:
                buf[(b - first) * SR : (b - first + 1) * SR] = raw
        glitch.apply_edit(buf, edit.t - first, edit.kind, edit.params)
        a = round((edit.t - EDIT_PIECE_LEAD) * SR)
        b = round((edit.t + edit.dur + EDIT_PIECE_TAIL) * SR)
        piece = buf[a - first * SR : b - first * SR].copy()
        self._results[found] = (a, piece)
        return self._results[found]

    def _edited(self, key: str, block: int) -> np.ndarray | None:
        """Return a bus over a block with the edits that touch it, or None when it is silent."""
        base = self._raw[key].get(block)
        lo = bisect_left(self._edit_starts, block - self._edit_reach)
        hi = bisect_right(self._edit_starts, block + 1.0)
        out: np.ndarray | None = base
        for index in range(lo, hi):
            edit = self.edits[index]
            if key not in edit.layers or (self.solo is not None and key not in self.solo):
                continue
            if edit.t + edit.dur + EDIT_PIECE_TAIL < block or edit.t - EDIT_PIECE_LEAD > block + 1:
                continue
            result = self._result(index, key)
            if result is None:
                continue
            start, piece = result
            a, b = max(start, block * SR), min(start + len(piece), (block + 1) * SR)
            if b <= a:
                continue
            if out is base:
                out = np.zeros((SR, 2), np.float32) if base is None else base.copy()
            assert out is not None
            out[a - block * SR : b - block * SR] = piece[a - start : b - start]
        return out

    def _ducked(self, key: str, bus: np.ndarray, block: int) -> np.ndarray:
        depth, tau = self.setup.duck[key]
        reach = tau * glitch_drums.DUCK_TAUS + DUCK_REACH_PAD
        lo = bisect_left(self._kick_times, block - reach)
        hi = bisect_right(self._kick_times, block + 1.0)
        if hi <= lo:
            return bus
        gain = glitch_drums.duck_gain(self.setup.kicks[lo:hi], float(block), SR, depth, tau)
        return np.asarray(bus * gain[:, None], dtype=np.float32)

    def _drum_bus(self, block: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the drum bus of a block and the snare alone (for its send), as (bus, snare)."""
        kick_key, snare_key, hat_key = self.setup.drum_layers
        kick = self._edited(kick_key, block)
        snare = self._edited(snare_key, block)
        hats = self._edited(hat_key, block)
        if kick is None and snare is None and hats is None:
            self._drum_quiet += 1
            if self._drum_quiet > DRUM_QUIET_BLOCKS:
                return None, None
        else:
            self._drum_quiet = 0
        zero = np.zeros((SR, 2), np.float32)
        kick = zero if kick is None else kick
        snare = zero if snare is None else snare
        hats = zero if hats is None else hats
        if self.light:
            raw = kick + snare + hats * np.float32(HAT_GAIN)
        else:
            wet = self._parallel(np.ascontiguousarray(snare.T), SR, reset=False).T
            room = self._plate(np.ascontiguousarray(snare.T), SR, reset=False).T
            raw = (
                kick
                + snare
                + np.float32(PARALLEL_AMOUNT) * wet
                + hats * np.float32(HAT_GAIN)
                + np.float32(PLATE_AMOUNT) * room
            )
        glued = self._glue(np.ascontiguousarray(raw.T.astype(np.float32)), SR, reset=False).T
        bus = (np.tanh(GLUE_DRIVE * glued) / GLUE_DRIVE).astype(np.float32)
        return bus, snare

    def _mix(self, block: int) -> np.ndarray:
        """Return the mix of a block before the master: ducked buses, the drum bus, the hall."""
        score = self.score
        drum_keys = set(self.setup.drum_layers)
        dry = np.zeros((SR, 2), dtype=np.float32)
        send = np.zeros((SR, 2), dtype=np.float32)
        send_hall = np.zeros((SR, 2), dtype=np.float32)
        for key, layer in score.layers.items():
            if key in drum_keys:
                continue
            bus = self._edited(key, block)
            if bus is None:
                continue
            if key in self.setup.duck:
                bus = self._ducked(key, bus, block)
            dry += bus
            self._send(layer, bus, send, send_hall)
        drums, snare = self._drum_bus(block)
        if drums is not None:
            dry += drums
        if snare is not None:
            send += snare * np.float32(score.layers[self.setup.drum_layers[1]].reverb)
        wet = self.reverb(np.ascontiguousarray(send.T), SR, reset=False).T
        space = self._ramp(score.space, block, REVERB_WET)
        out = self._hall_return(dry + wet.astype(np.float32) * space[:, None], send_hall, space)
        out = self._hp.process(out, np.array([MASTER_HP_HZ]), SR)
        return np.asarray(out, dtype=np.float32)
