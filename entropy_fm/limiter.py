"""
A streaming peak limiter with look-ahead, for the master of the mix.

The output is delayed by the look-ahead (5 ms). The gain ramps down linearly over the look-ahead
into every peak that would pass the ceiling, so that the limiter does not distort the waveform,
and comes back up with a smooth release. Its work is done at a twentieth of the sample rate,
which is plenty for a gain that moves in milliseconds, and it keeps what it needs between blocks,
so the blocks of one second join without a seam. A last hard clip to the ceiling catches whatever is
left over (inter-sample overshoot of the interpolated gain); it should almost never act, and the
limiter counts when it does.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import numpy as np

from .dsp import AMPLITUDE_DB_DIVISOR, MIN_LEVEL, SILENCE_FLOOR
from .synth import SAMPLE_RATE

LOOKAHEAD_S = 0.005
RELEASE_S = 0.15
# the gain is worked out every this many samples
STRIDE = 20
# a gain above this is no reduction at all
UNITY_GAIN = 0.9999
# a sample is clipped only when it is this much over the ceiling
CLIP_TOLERANCE = 1.001


class Limiter:
    """A look-ahead peak limiter that works on blocks of stereo float samples."""

    def __init__(
        self,
        ceiling_db: float,
        sample_rate: int = SAMPLE_RATE,
        lookahead: float = LOOKAHEAD_S,
        release: float = RELEASE_S,
        stride: int = STRIDE,
    ) -> None:
        """
        Set the ceiling and the time constants.

        :param ceiling_db: The highest level of a sample, in dBFS.
        :param sample_rate: The sample rate.
        :param lookahead: The look-ahead, and the delay of the output, in seconds.
        :param release: The time in which the gain recovers about two thirds of the way, seconds.
        :param stride: The gain is worked out every this many samples; it must divide the block.
        """
        self.ceiling = float(10 ** (ceiling_db / AMPLITUDE_DB_DIVISOR))
        self.stride = stride
        self.delay = max(stride, round(lookahead * sample_rate / stride) * stride)
        self.steps = self.delay // stride
        self.recover = float(1.0 - np.exp(-stride / (release * sample_rate)))
        self._audio = np.zeros((self.delay, 2), dtype=np.float32)
        self._need = np.ones(self.steps)
        self._gain = 1.0
        # for the report: samples seen, samples the last clip touched, the gain reduction in dB
        self.samples = 0
        self.clipped = 0
        self.reduction_sum = 0.0
        self.reduction_max = 0.0
        self.blocks = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Limit a block and return the same number of samples, delayed by the look-ahead.

        :param block: Stereo samples, shape (n, 2) with n a multiple of `STRIDE`.
        """
        count = len(block)
        full = np.concatenate([self._audio, block.astype(np.float32)])
        peak_here = float(np.abs(block).max())
        if self._gain >= UNITY_GAIN and peak_here < self.ceiling and not (self._need < 1.0).any():
            # nothing reaches the ceiling and nothing is held down: the block is only delayed
            self._audio = full[count:]
            self._need = np.ones(self.steps)
            self.samples += block.size
            self.blocks += 1
            return np.asarray(full[:count], dtype=np.float32)
        peak = np.abs(block).max(axis=1)
        stride = self.stride
        groups = peak[: count // stride * stride].reshape(-1, stride).max(axis=1)
        need = np.concatenate(
            [self._need, np.minimum(1.0, self.ceiling / np.maximum(groups, SILENCE_FLOOR))]
        )
        # allowed[d] = min over the look-ahead of the ramp into each peak that is ahead of d
        steps = self.steps
        outputs = count // stride
        allowed = np.ones(outputs)
        for offset in range(steps + 1):
            ramp = 1.0 - (1.0 - need[offset : offset + outputs]) * (1.0 - offset / (steps + 1))
            np.minimum(allowed, ramp, out=allowed)
        gain = np.empty(outputs)
        current = self._gain
        for index in range(outputs):
            limit = allowed[index]
            current = limit if limit < current else current + (1.0 - current) * self.recover
            gain[index] = min(current, limit)
        self._gain = float(current)
        points = np.arange(outputs) * stride
        # between the points the gain is linear, and never above the one that its own group needs
        smooth = np.interp(np.arange(count), points, gain)
        per_sample = np.minimum(smooth, np.repeat(gain, stride)).astype(np.float32)
        out = full[:count] * per_sample[:, None]
        self._audio = full[count:]
        self._need = need[outputs:]
        hot = np.abs(out) > self.ceiling * CLIP_TOLERANCE
        self.clipped += int(hot.sum())
        if hot.any():
            out = np.where(hot, np.sign(out) * self.ceiling, out).astype(np.float32)
        self.samples += out.size
        reduction = -AMPLITUDE_DB_DIVISOR * np.log10(np.maximum(gain, MIN_LEVEL))
        self.reduction_sum += float(reduction.mean())
        self.reduction_max = max(self.reduction_max, float(reduction.max()))
        self.blocks += 1
        return np.asarray(out, dtype=np.float32)

    @property
    def mean_reduction_db(self) -> float:
        """Return the mean gain reduction in dB over what has been processed."""
        return self.reduction_sum / self.blocks if self.blocks else 0.0

    @property
    def clip_share(self) -> float:
        """Return the share of samples that the last clip had to touch."""
        return self.clipped / self.samples if self.samples else 0.0
