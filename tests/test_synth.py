"""Tests of the scales and the smooth ramp."""

import numpy as np

from entropy_fm.synth import Scale, smoothstep


def test_scale_quantises_to_the_mode() -> None:
    """Every degree of D dorian is a note of D dorian, and degrees wrap into octaves."""
    scale = Scale(2, "dorian")
    notes = [scale.midi(degree, 4) for degree in range(8)]
    assert notes == [62, 64, 65, 67, 69, 71, 72, 74]
    assert scale.midi(-1, 4) == 60


def test_the_gamelan_tunings_are_gone() -> None:
    """The four styles are equal tempered: no slendro and no pelog."""
    from entropy_fm.synth import SCALES  # noqa: PLC0415

    assert "slendro" not in SCALES
    assert "pelog" not in SCALES


def test_smoothstep_is_a_ramp() -> None:
    """The smooth ramp goes from 0 to 1."""
    assert float(smoothstep(np.array(0.0))) == 0.0
    assert float(smoothstep(np.array(1.0))) == 1.0
    assert 0.0 < float(smoothstep(np.array(0.5))) < 1.0
