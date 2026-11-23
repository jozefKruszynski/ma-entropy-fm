"""Tests of the sampler: zones, velocity layers, pitch shifting, envelopes, loops, one-shots, the cache."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.drums import SNARE_LOFI_SHOTS
from entropy_fm.sampler import MAX_SHIFT, SAMPLES_DIR, Instrument, SampleCache
from entropy_fm.synth import SAMPLE_RATE

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import fetch_samples


def _pitch(wave: np.ndarray) -> float:
    """Return the MIDI pitch of a note by autocorrelation (the one the sample builder uses), after its attack."""
    measured = fetch_samples.measure_pitch(wave[int(0.1 * SAMPLE_RATE) :].astype(np.float64))
    assert measured is not None
    return measured


def test_a_key_picks_the_nearest_zone_and_a_velocity_picks_the_layer() -> None:
    """The felt piano has a soft and a medium layer, a note every seven semitones."""
    piano = Instrument.load(SAMPLES_DIR / "felt_piano")
    soft = piano.zone_for(60.0, 0.2)
    loud = piano.zone_for(60.0, 0.95)
    assert soft.file != loud.file
    assert soft.vhi < loud.vlo or soft.vlo < loud.vlo
    assert abs(soft.root - 60.0) < 4.0
    assert piano.zone_for(61.0, 0.5).root == piano.zone_for(61.0, 0.5).root


def test_a_note_is_shifted_by_resampling_to_the_pitch_asked_for() -> None:
    """A sample is read faster or slower so that the note lands on the key, within a few cents."""
    piano = Instrument.load(SAMPLES_DIR / "grand_piano")
    for note in (52.0, 58.0, 63.0, 71.0):
        wave = piano.render(note, 0.7, 2.0)
        assert abs(_pitch(wave) - note) < 0.25, note


def test_a_note_far_from_every_sample_is_folded_by_octaves_not_stretched() -> None:
    """The pluck has samples from D3 to A5: a note two octaves lower plays an octave multiple, not a slowed sample."""
    pluck = Instrument.load(SAMPLES_DIR / "pluck")
    wave = pluck.render(36.0, 0.7, 1.0)
    assert (_pitch(wave) - 36.0) % 12.0 < 0.3 or 12.0 - (_pitch(wave) - 36.0) % 12.0 < 0.3
    assert MAX_SHIFT <= 6.0


def test_the_envelope_has_an_attack_and_a_release_after_the_key_goes_up() -> None:
    """The strings rise slowly, hold, and die away after the held length."""
    strings = Instrument.load(SAMPLES_DIR / "strings")
    wave = strings.render(57.0, 0.7, 3.0, group="violins")
    assert len(wave) > 3.0 * SAMPLE_RATE
    early = np.abs(wave[: int(0.1 * SAMPLE_RATE)]).max()
    middle = np.abs(wave[int(2.0 * SAMPLE_RATE) : int(2.5 * SAMPLE_RATE)]).max()
    late = np.abs(wave[-int(0.2 * SAMPLE_RATE) :]).max()
    assert early < 0.5 * middle
    assert late < 0.05 * middle


def test_a_sustain_loop_holds_a_note_longer_than_the_sample() -> None:
    """A note of 14 seconds from a sample of under 10 seconds goes on without a gap."""
    strings = Instrument.load(SAMPLES_DIR / "strings")
    wave = strings.render(60.0, 0.7, 14.0, group="violins")
    assert len(wave) > 14 * SAMPLE_RATE
    level = np.sqrt(np.convolve(wave**2, np.ones(4410) / 4410, mode="same"))
    held = level[int(3 * SAMPLE_RATE) : int(13 * SAMPLE_RATE)]
    assert held.min() > 0.2 * held.max()


def test_a_named_one_shot_plays_its_whole_sample() -> None:
    """The drum one-shots are found by name."""
    drums = Instrument.load(SAMPLES_DIR / "drums")
    shot = drums.render(0.0, 0.8, None, name=SNARE_LOFI_SHOTS[0])
    assert 0.1 < len(shot) / SAMPLE_RATE < 1.0
    with pytest.raises(KeyError):
        drums.named("nothing")


def test_threads_loading_the_same_samples_never_miscount_the_cache() -> None:
    """The bytes the cache counts are the bytes it holds, however many threads decode one file."""
    files = sorted((SAMPLES_DIR / "strings").glob("*.ogg"))[:3]
    cache = SampleCache(limit_mb=1000)
    jobs = 4
    with ThreadPoolExecutor(jobs * len(files)) as pool:
        waves = list(pool.map(cache.get, files * jobs))
    assert len(cache._items) == len(files)  # noqa: SLF001
    assert cache._bytes == sum(wave.nbytes for wave in {id(w): w for w in waves}.values())  # noqa: SLF001
    assert all(cache.get(path) is waves[files.index(path)] for path in files)
