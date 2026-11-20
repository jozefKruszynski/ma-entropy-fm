"""Tests of the master: the limiter, the leveler and the loudness targets of the styles."""

import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from entropy_fm.limiter import Limiter

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _blocks(signal: np.ndarray, limiter: Limiter) -> np.ndarray:
    return np.concatenate(
        [limiter.process(signal[i : i + 44100]) for i in range(0, len(signal), 44100)]
    )


def test_the_limiter_never_passes_its_ceiling_and_leaves_quiet_audio_alone() -> None:
    """Loud peaks are held to the ceiling without the clip acting; quiet audio is only delayed."""
    rng = np.random.default_rng(3)
    loud = np.clip(rng.normal(0, 0.4, (44100 * 5, 2)), -2.0, 2.0).astype(np.float32)
    limiter = Limiter(-1.6)
    out = _blocks(loud, limiter)
    assert np.abs(out).max() <= 10 ** (-1.6 / 20) * 1.0011
    assert limiter.clip_share < 0.0001
    quiet = (
        (0.1 * np.sin(np.arange(44100 * 2) / 20.0))[:, None].repeat(2, axis=1).astype(np.float32)
    )
    limiter = Limiter(-1.6)
    out = _blocks(quiet, limiter)
    delay = limiter.delay
    assert np.allclose(out[delay:], quiet[:-delay], atol=1e-6)
    assert limiter.reduction_max < 0.01


def test_the_limiter_joins_blocks_without_a_seam() -> None:
    """One long block and five short ones give the same samples: the state crosses the blocks."""
    rng = np.random.default_rng(4)
    signal = np.clip(rng.normal(0, 0.5, (44100 * 4, 2)), -1.9, 1.9).astype(np.float32)
    whole = Limiter(-1.6).process(signal)
    split = _blocks(signal, Limiter(-1.6))
    assert np.allclose(whole, split, atol=1e-5)


def _loudness_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    import loudness_report  # noqa: PLC0415

    return loudness_report


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize(
    "style", ["ambient", "berlin_school", "lofi", "minimal", "ritual", "glitch"]
)
def test_every_style_lands_in_its_loudness_band_with_the_limiter_hardly_working(style: str) -> None:
    """A stretch of the busy part of the piece: integrated loudness in the band, true peak under -1 dBTP, limiter under 2 dB."""
    from .scape_tools import demo, window  # noqa: PLC0415

    report = _loudness_module()
    score, plan = demo(style)
    stats: dict[str, float] = {}
    numbers = report.ebur128(window(score, plan, 60, stats=stats))
    low, high = report.TARGETS[style]
    assert low - 1.5 <= numbers["integrated"] <= high + 1.5, numbers
    assert numbers["true_peak"] <= -1.0, numbers
    assert stats["max_reduction_db"] <= 2.0, stats
    assert stats["mean_reduction_db"] < 0.5, stats
    # a soundscape is held flat: the loudest three seconds are not far above the whole
    assert numbers["short_max"] <= numbers["integrated"] + 3.0, numbers
