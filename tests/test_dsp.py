"""Tests of the signal processing helpers."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from entropy_fm import dsp

BAND_LOW_HZ = 913.0
BAND_HIGH_HZ = 2711.0
SAMPLES = 4096
THREADS = 8
SEED = 5
A4_MIDI = 69
A4_HZ = 440.0


def test_band_noise_is_the_same_for_threads_that_ask_at_once() -> None:
    """Threads that ask for a new band together get what a single caller gets afterwards."""

    def draw(_index: int) -> np.ndarray:
        return dsp.band_noise(np.random.default_rng(SEED), SAMPLES, BAND_LOW_HZ, BAND_HIGH_HZ)

    with ThreadPoolExecutor(THREADS) as pool:
        waves = list(pool.map(draw, range(THREADS)))
    alone = draw(0)
    assert all(np.array_equal(wave, alone) for wave in waves)


def test_the_oscillator_kinds_are_named() -> None:
    """The four shapes have names, and a table exists for each."""
    assert dsp.KINDS == (dsp.SAW, dsp.SQUARE, dsp.TRIANGLE, dsp.SINE)
    assert all(len(dsp.table(kind, 0)) == dsp.TABLE_SIZE + 1 for kind in dsp.KINDS)


def test_the_tuning_note_is_440_hz() -> None:
    """MIDI note 69 is A at 440 Hz."""
    assert dsp.midi_hz(A4_MIDI) == pytest.approx(A4_HZ)
