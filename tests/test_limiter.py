"""Tests of the look-ahead peak limiter."""

import numpy as np

from entropy_fm.limiter import Limiter

SAMPLE_RATE = 44100
CEILING_DB = -2.4
HOT_AMPLITUDE = 3.0
BLOCKS = 4
FREQUENCY_HZ = 220.0


def test_the_limiter_keeps_every_sample_under_the_ceiling() -> None:
    """A signal three times over full scale comes out under the ceiling, block after block."""
    limiter = Limiter(CEILING_DB, SAMPLE_RATE)
    ceiling = 10 ** (CEILING_DB / 20.0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    wave = (HOT_AMPLITUDE * np.sin(2 * np.pi * FREQUENCY_HZ * t)).astype(np.float32)
    block = np.stack([wave, wave], axis=1)
    for _ in range(BLOCKS):
        out = limiter.process(block)
        assert np.abs(out).max() <= ceiling * (1.0 + 1e-6)
    assert limiter.mean_reduction_db > 0.0
