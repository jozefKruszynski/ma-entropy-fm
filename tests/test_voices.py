"""Tests of the voice library: distinct fingerprints, no aliasing, vibrato and glide limits, constant loudness, determinism."""

import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.quality import LIGHT, RICH, STANDARD
from entropy_fm.voices import MAX_NOTE_SECONDS, VOICES, gain_of, quantised, render_note

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import voice_report

from .pitch_tools import find_glides

RATE = 44100
# the voices that are made of samples: real pianos and strings are not harmonic to the last cent
SAMPLED = {
    "felt_piano",
    "grand_piano",
    "tx_ep",
    "strum_pluck",
    "tape_strings",
    "chant_cello",
    "slit_drum",
    "woodblock",
}


def _print(key: str, midi: float | None = None) -> tuple[voice_report.Fingerprint, float]:
    voice = VOICES[key]
    pitch = voice.ref_midi if midi is None else midi
    f0 = 440.0 * 2.0 ** ((pitch - 69.0) / 12.0) * voice.fundamental
    return voice_report.fingerprint(render_note(key, pitch, 2.6, 0.7, STANDARD), f0), f0


def _vector(fp: voice_report.Fingerprint) -> np.ndarray:
    """The numbers that tell a voice from another: where its energy is, how fast it starts, its harmonics, its movement."""  # noqa: D401
    return np.array(
        [
            np.log2(fp.centroid) / 2.0,
            np.log10(max(fp.attack_ms, 2.0)),
            *fp.profile,
            np.sqrt(fp.modulation),
        ]
    )


def test_the_library_has_the_voices_the_styles_draw_from() -> None:
    """Basses, strings and pads, leads and keys, plucks: the families that were asked for."""
    families = {voice.family for voice in VOICES.values()}
    assert families == {"bass", "pad", "lead", "key", "pluck"}
    for key in (
        "sub_round",
        "moog_bass",
        "reese",
        "fm_growl",
        "acid_bass",
        "string_machine",
        "juno_pad",
        "supersaw",
        "tape_strings",
        "tape_choir",
        "glass_pad",
        "bowed_pad",
        "analog_lead",
        "flute_lead",
        "fm_ep",
        "felt_piano",
        "tx_ep",
        "juno_pluck",
        "nylon_ks",
    ):
        assert key in VOICES
    assert len(VOICES) >= 22


def test_every_voice_has_a_fingerprint_of_its_own() -> None:
    """No two voices have the same centroid, attack, harmonic profile and modulation."""
    vectors = {key: _vector(_print(key)[0]) for key in VOICES}
    for first, second in itertools.combinations(vectors, 2):
        distance = float(np.linalg.norm(vectors[first] - vectors[second]))
        assert distance > 0.12, (first, second, distance)


def test_the_voices_are_not_aliased_and_the_growl_makes_harmonics_only() -> None:
    """Energy outside the harmonics stays under 2 percent for the synthesised voices (windows widened for ensembles)."""
    for key in VOICES:
        if key in SAMPLED:
            continue
        for shift in (0.0, 24.0):
            fp, _ = _print(key, VOICES[key].ref_midi + shift)
            assert fp.inharmonic_wide < 0.02, (key, shift, fp.inharmonic_wide)
        if VOICES[key].family in {"lead", "key"} or key in {
            "sub_round",
            "moog_bass",
            "fm_growl",
            "reese",
        }:
            fp, _ = _print(key)
            assert fp.inharmonic < 0.02, (key, fp.inharmonic)


def test_a_voice_is_as_loud_as_every_other_at_its_reference_note() -> None:
    """Each voice has a constant gain of its own: the RMS of the reference note is the same for all."""
    levels = []
    for key, voice in VOICES.items():
        wave = render_note(key, voice.ref_midi, 2.5, 0.7, STANDARD)
        levels.append(float(np.sqrt(np.mean(wave[: int(2.5 * RATE)] ** 2))))
    assert max(levels) / min(levels) < 1.3


def test_no_note_is_normalised_to_a_peak() -> None:
    """A quiet note and a hard one of a voice keep their ratio: the level moves with the velocity."""
    soft = render_note("analog_lead", 64.0, 1.0, 0.3, STANDARD)
    hard = render_note("analog_lead", 64.0, 1.0, 0.9, STANDARD)
    assert np.abs(soft).max() < 0.8 * np.abs(hard).max()


def test_a_vibrato_is_at_most_fifteen_cents() -> None:
    """The singing lead and the flute move by a vibrato and never by more than 15 cents; the pads by less."""
    for key in ("analog_lead", "flute_lead"):
        wave = render_note(key, 69.0, 4.0, 0.7, STANDARD).mean(axis=1)
        frame, hop = 8192, 2048
        window = np.hanning(frame)
        freqs = np.fft.rfftfreq(frame, 1.0 / RATE)
        found = []
        for start in range(int(1.5 * RATE), int(3.6 * RATE), hop):
            spectrum = np.abs(np.fft.rfft(wave[start : start + frame] * window))
            band = (freqs > 380) & (freqs < 500)
            peak = int(np.argmax(spectrum * band))
            shift = (
                0.5
                * (np.log(spectrum[peak - 1]) - np.log(spectrum[peak + 1]))
                / (
                    np.log(spectrum[peak - 1])
                    - 2 * np.log(spectrum[peak])
                    + np.log(spectrum[peak + 1])
                )
            )
            found.append(1200.0 * np.log2((peak + shift) * RATE / frame / 440.0))
        # a vibrato of at most 15 cents swings 30 cents peak to peak round the pitch, and the note is in tune
        assert max(found) - min(found) <= 32.0, (key, found)
        assert abs(float(np.mean(found))) < 8.0


@pytest.mark.parametrize(
    "key",
    ["analog_lead", "flute_lead", "string_machine", "fm_growl", "reese", "tape_choir", "nylon_ks"],
)
def test_a_note_does_not_glide(key: str) -> None:
    """The glide tracker finds nothing above 150 Hz in a short phrase of a voice."""
    notes = [
        render_note(key, pitch, 1.4, 0.7, STANDARD).mean(axis=1)
        for pitch in (57.0, 60.0, 64.0, 62.0, 57.0)
    ]
    phrase = np.concatenate(notes)
    assert find_glides(phrase, 3.0, 1.5, -1) == []
    assert find_glides(phrase, 3.0, 1.5, 1) == []


def test_a_note_is_the_same_every_time_and_a_cheaper_tier_has_fewer_saws() -> None:
    """Notes are deterministic; the supersaw has seven saws in Rich and three in Light, which is why it costs less."""
    first = render_note("supersaw", 57.0, 2.0, 0.7, RICH)
    second = render_note("supersaw", 57.0, 2.0, 0.7, RICH)
    assert np.array_equal(first, second)
    light = render_note("supersaw", 57.0, 2.0, 0.7, LIGHT)
    assert not np.array_equal(first[: len(light)], light[: len(first)])
    assert LIGHT.unison < RICH.unison


def test_a_long_note_is_made_from_a_loop_that_has_no_seam() -> None:
    """A note of 20 seconds has no click and no hole at the joins of its sustain."""
    for key in ("string_machine", "tape_choir", "reese", "drawbar_organ"):
        wave = render_note(key, 57.0, 20.0, 0.7, STANDARD).mean(axis=1)
        assert len(wave) > 20 * RATE
        steps = np.abs(np.diff(wave))
        assert steps.max() < 6.0 * np.abs(wave).max() * 0.05 + 0.05, key
        level = np.sqrt(np.convolve(wave**2, np.ones(2205) / 2205, mode="same"))[
            5 * RATE : 18 * RATE
        ]
        assert level.min() > 0.1 * level.max(), key


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0.01, 0.02),
        (0.59, 0.58),
        (1.0, 1.0),
        (2.9, 3.0),
        (5.0, 5.0),
        (9.6, 10.0),
        (99.0, MAX_NOTE_SECONDS),
    ],
)
def test_quantised_steps_grow_with_the_length(length: float, expected: float) -> None:
    """A note is rounded to a coarser step the longer it is, and never past the longest note."""
    assert quantised(length) == pytest.approx(expected)


def test_gain_is_the_same_for_threads_that_ask_at_once() -> None:
    """The gain of a voice does not depend on how many threads work it out together."""
    with ThreadPoolExecutor(4) as pool:
        gains = list(pool.map(lambda _i: gain_of("flute_lead", STANDARD), range(4)))
    assert len(set(gains)) == 1
