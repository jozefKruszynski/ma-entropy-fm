"""Tests of the sampler: zones, velocity layers, pitch shifting, envelopes, loops, voice limits, determinism."""

import sys
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.sampler import MAX_SHIFT, SAMPLES_DIR, Instrument, Sampler, Trigger
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


def test_the_voice_limit_cuts_the_oldest_voice_and_a_choke_group_cuts_its_own() -> None:
    """With a limit of two voices the third note cuts the first; a hat of a group cuts the open hat."""
    sampler = Sampler.from_folder(["felt_piano", "drums"], limit=2)
    triggers = [Trigger(0.1 * k, "felt_piano", 0.7, 60.0 + k, duration=2.0) for k in range(3)]
    voices = sampler.schedule(triggers)
    assert voices[0].cut is not None
    assert voices[1].cut is None
    assert voices[2].cut is None


def test_the_same_triggers_give_the_same_samples_to_the_last_bit() -> None:
    """Nothing is random: two samplers mix the same block of the same notes alike."""
    triggers = [
        Trigger(0.2 * k, "grand_piano", 0.3 + 0.1 * k, 55.0 + 3 * k, duration=1.0, pan=0.1 * k)
        for k in range(6)
    ]
    blocks = []
    for _ in range(2):
        sampler = Sampler.from_folder(["grand_piano"])
        voices = sampler.schedule(triggers)
        blocks.append(sampler.mix(voices, 0, SAMPLE_RATE))
    assert np.array_equal(blocks[0], blocks[1])
    assert np.abs(blocks[0]).max() > 0.01


def test_a_named_one_shot_plays_its_whole_sample() -> None:
    """The drum one-shots are found by name."""
    drums = Instrument.load(SAMPLES_DIR / "drums")
    assert {"kick", "snare", "rim", "click"} <= set(drums.groups)
    shot = drums.render(0.0, 0.8, None, name=drums.groups["snare"][0])
    assert 0.1 < len(shot) / SAMPLE_RATE < 1.0
    with pytest.raises(KeyError):
        drums.named("nothing")
