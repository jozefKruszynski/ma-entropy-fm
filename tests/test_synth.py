"""Tests of the scales, pitches and panning."""

import pytest

from entropy_fm.synth import Scale, midi_to_hz, pan_gains, smoothstep


def test_scale_quantises_to_the_mode() -> None:
    """Every degree of D dorian is a note of D dorian, and degrees wrap into octaves."""
    scale = Scale(2, "dorian")
    notes = [scale.midi(degree, 4) for degree in range(8)]
    assert notes == [62, 64, 65, 67, 69, 71, 72, 74]
    assert scale.midi(-1, 4) == 60
    assert midi_to_hz(69) == pytest.approx(440.0)


def test_the_gamelan_tunings_are_gone() -> None:
    """The four styles are equal tempered: no slendro and no pelog."""
    from entropy_fm.synth import SCALES  # noqa: PLC0415

    assert "slendro" not in SCALES
    assert "pelog" not in SCALES


def test_pan_is_equal_power_and_smoothstep_is_a_ramp() -> None:
    """A pan keeps the power, and the smooth ramp goes from 0 to 1."""
    for pan in (-1.0, -0.3, 0.0, 0.8, 1.0):
        left, right = pan_gains(pan)
        assert left**2 + right**2 == pytest.approx(1.0)
    assert float(smoothstep(0.0)) == 0.0
    assert float(smoothstep(1.0)) == 1.0
    assert 0.0 < float(smoothstep(0.5)) < 1.0
