"""Tests that in Berlin School the driving sequence A leads and a blip as sequence B sits behind it."""

from __future__ import annotations

import numpy as np

from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all
from entropy_fm.score import Score

from .scape_tools import demo

# a day of the demo house whose second line is the blip synth, with both rooms playing
BLIP_DAY = 7
BLIP_AREAS = ("kitchen", "living")
SEQ_A_LEAD_DB = 3.0
WINDOW_SECONDS = 40
SILENCE_DB = 1e-9


def _solo_db(score: Score, key: str, start: int) -> float:
    """Return the level in dB of one layer alone, in the window that starts at `start`."""
    audio = render_all(
        score,
        STANDARD,
        "test",
        None,
        start_s=start,
        end_s=start + WINDOW_SECONDS,
        solo=frozenset({key}),
    )
    return float(20 * np.log10(np.sqrt(np.mean(audio**2)) + SILENCE_DB))


def test_sequence_a_leads_a_blip_second_line() -> None:
    """The driving line is louder than the blip line by a clear margin in the busiest window."""
    from soundscapes import pick_windows  # noqa: PLC0415

    score, plan = demo("berlin_school", BLIP_DAY, BLIP_AREAS)
    assert "blip_synth" in {n.voice for n in score.notes if n.layer == "seqB"}
    start = pick_windows(plan, WINDOW_SECONDS)[1][1]
    assert _solo_db(score, "seqA", start) >= _solo_db(score, "seqB", start) + SEQ_A_LEAD_DB
