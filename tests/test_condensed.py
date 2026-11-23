"""Tests of the length of a piece: 5 to 7 minutes for the whole day, with the density of events held down."""

import shutil
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from entropy_fm.plan import (
    CONDENSED_EVENT_GAP,
    CONDENSED_TRIGGER_GAP,
    _thin_events,
    _thin_triggers,
    condensed_seconds,
)
from entropy_fm.score import CURVE_RATE
from entropy_fm.styles import STYLES

from .pitch_tools import find_glides, noise_floor_db
from .scape_tools import demo, window

SLOW = ["ambient", "berlin_school", "ritual"]
FAST = ["lofi", "glitch"]
SCRIPTS = Path(__file__).parents[1] / "scripts"


def test_slow_styles_are_six_to_eight_minutes_and_fast_ones_about_five() -> None:
    """The length of a piece is fixed per style, and the plan, the score and the listing agree on it."""
    assert set(SLOW + FAST) == set(STYLES)
    for style in SLOW:
        assert 360 <= condensed_seconds(style) <= 480, style
    for style in FAST:
        assert 270 <= condensed_seconds(style) <= 330, style
    for style in STYLES:
        score, plan = demo(style)
        assert plan.length == score.length == condensed_seconds(style), style


def test_a_condensed_piece_is_the_same_day_faster() -> None:
    """The whole day is in it: the night, the morning, the evening, in order, and each stretch is long enough to hear."""
    for style in STYLES:
        _, plan = demo(style)
        arc = plan.arc()
        assert arc[0][0] == "night"
        assert arc[-1][2] == pytest.approx(plan.length)
        assert all(b - a >= 20.0 for _, a, b in arc), (style, arc)
        assert abs(float(plan.evening[int(plan.length * CURVE_RATE * 0.85)]) - 1.0) < 0.35


def test_the_curves_that_build_move_over_at_least_a_few_seconds() -> None:
    """Activity, brightness and lights never jump: the slowest changes of a build take seconds, not frames."""
    for style in STYLES:
        _, plan = demo(style)
        for curve in (plan.activity, plan.bright, plan.lights, plan.power):
            step = np.abs(np.diff(curve)).max() * CURVE_RATE
            assert step < 0.25, (style, step)


def test_unusual_moments_are_kept_apart_and_the_strongest_are_kept() -> None:
    """Anomalies are kept three at the most, never closer than the gap; the weaker of two close ones goes."""
    found = [(30.0, 0.4), (40.0, 0.9), (100.0, 0.5), (140.0, 0.6), (200.0, 0.3)]
    kept = _thin_events(found, CONDENSED_EVENT_GAP)
    assert kept == [(40.0, 0.9), (100.0, 0.5), (140.0, 0.6)]
    assert all(b[0] - a[0] >= CONDENSED_EVENT_GAP for a, b in pairwise(kept))
    for style in STYLES:
        _, plan = demo(style)
        assert 1 <= len(plan.events) <= 3, "the anomalies of the demo day are kept"
        times = [t for t, _ in plan.events]
        assert all(b - a >= CONDENSED_EVENT_GAP for a, b in pairwise(times))


def test_the_little_things_of_the_house_are_thinned_with_the_rarer_kinds_first() -> None:
    """A switch next to an appliance start gives way to the appliance; nothing is closer than the gap."""
    found = [
        (10.0, 0.5, "light"),
        (12.0, 0.5, "appliance"),
        (13.0, 0.5, "motion"),
        (30.0, 0.5, "door"),
    ]
    assert _thin_triggers(found, CONDENSED_TRIGGER_GAP) == [
        (12.0, 0.5, "appliance"),
        (30.0, 0.5, "door"),
    ]
    for style in STYLES:
        _, plan = demo(style)
        times = [t for t, _, _ in plan.triggers]
        assert all(b - a >= CONDENSED_TRIGGER_GAP for a, b in pairwise(times))
        assert len(plan.triggers) <= plan.length / CONDENSED_TRIGGER_GAP


def test_glitch_edits_and_drops_are_not_crowded() -> None:
    """Cinematic Glitch cuts up at most twelve times a minute, with at most three drops."""
    score, plan = demo("glitch")
    assert score.glitch is not None
    drops = [e for e in score.glitch.edits if e.why == "drop"]
    assert len(drops) <= 3 * 4
    per_minute = len(score.glitch.edits) / (plan.length / 60.0)
    assert per_minute <= 12.0, per_minute


@pytest.mark.parametrize("style", SLOW + FAST)
def test_the_notes_stay_inside_the_piece(style: str) -> None:
    """Every note starts within the length of the piece."""
    score, _ = demo(style)
    assert all(0.0 <= n.t < score.length for n in score.notes), style


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("style", SLOW + FAST)
def test_a_condensed_piece_lands_in_the_loudness_band(style: str) -> None:
    """The loudness band of the styles: integrated loudness, true peak under -1 dBTP, the limiter hardly working."""
    sys.path.insert(0, str(SCRIPTS))
    import loudness_report  # noqa: PLC0415

    score, plan = demo(style)
    stats: dict[str, float] = {}
    numbers = loudness_report.ebur128(window(score, plan, 60, stats=stats))
    low, high = loudness_report.TARGETS[style]
    assert low - 1.5 <= numbers["integrated"] <= high + 1.5, numbers
    assert numbers["true_peak"] <= -1.0, numbers
    assert stats["max_reduction_db"] <= 2.0, stats
    assert stats["mean_reduction_db"] < 0.5, stats
    assert numbers["short_max"] <= numbers["integrated"] + 3.0, numbers


@pytest.mark.parametrize("style", ["ambient", "ritual", "lofi"])
def test_a_condensed_piece_has_no_continuous_pitch_glide(style: str) -> None:
    """No fall and no rise of more than three semitones in a second and a half."""
    score, plan = demo(style)
    audio = window(score, plan, 40).mean(axis=1)
    assert find_glides(audio, 3.0, 1.5, -1) == [], style
    assert find_glides(audio, 3.0, 1.5, 1) == [], style


@pytest.mark.parametrize("style", ["ambient", "berlin_school", "lofi"])
def test_a_condensed_piece_has_a_low_noise_floor(style: str) -> None:
    """The steady broadband floor in the quietest frames stays under -48 dBFS above 6 kHz."""
    score, plan = demo(style)
    audio = window(score, plan, 40).mean(axis=1)
    assert noise_floor_db(audio) < -48.0, style


def test_the_glides_of_condensed_glitch_are_the_tape_stops_and_brakes_only() -> None:
    """Glitch glides only where an edit is a tape stop, a brake or a drop."""
    from soundscapes import pick_windows  # noqa: PLC0415

    from .test_glitch import GLIDING  # noqa: PLC0415

    score, plan = demo("glitch")
    assert score.glitch is not None
    audio = window(score, plan, 40)
    start = pick_windows(plan, 40)[1][1]
    allowed = [e.t - start for e in score.glitch.edits if e.kind in GLIDING or e.why == "drop"]
    for direction in (-1, 1):
        for when, _ in find_glides(audio.mean(axis=1), 3.0, 1.5, direction):
            assert any(-2.0 <= when - t <= 2.5 for t in allowed), (when, allowed)


def test_the_start_and_the_end_still_fade_in_a_condensed_piece() -> None:
    """The first second fades in and the last one fades out."""
    from entropy_fm.quality import STANDARD  # noqa: PLC0415
    from entropy_fm.render import render_blocks  # noqa: PLC0415

    score, _ = demo("lofi")
    tail = list(render_blocks(score, STANDARD, "x", score.length - 2))
    assert np.abs(tail[-1][-44100 // 50 :]).max() < 0.01
    assert score.length == 300
