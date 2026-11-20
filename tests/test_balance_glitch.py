"""Tests of the balance of Cinematic Glitch: half-time drums that move with the form, and a heavy mode that is never bare."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import pytest

from entropy_fm import scape_glitch
from entropy_fm.character import ALL_DAY, HIGH, LOW, NEUTRAL, scale_of_day
from entropy_fm.plan import BUILD, CLIMAX, PLATEAU, RELEASE, Plan
from entropy_fm.score import Score
from entropy_fm.styles import GLITCH
from entropy_fm.synth import Scale

from .scape_tools import demo

HOME = Scale(2, "dorian")
HEAVY, HALF_TIME, DRIVING = scape_glitch.MODES
DAYS = range(5)
# the kit counts twice the tempo; half-time puts about one kick on every two beats of that count or fewer
KIT_BEATS_PER_BEAT = 2
MOST_KICKS_PER_KIT_BEAT = 0.7
# sections of a piece must differ by this share in kicks per minute
SECTION_SPREAD = 1.3
MIN_SECTION_SECONDS = 30.0
# the daytime stretches of a heavy piece: the share of minutes with drums, and the edits per minute
HEAVY_DRUM_SHARE = 0.4
HEAVY_EDITS_PER_MINUTE = 0.9
SECONDS = 60.0
DAY_ARCS = ("morning", "daytime")


def _made(index: int, energy: float, mode: scape_glitch.Mode) -> tuple[Score, Plan]:
    """Return the score and the plan of the nth day of the demo room, on a character of this energy, in one mode."""
    _, base = demo("glitch")
    assert base.day is not None
    level = LOW if energy < 0.34 else (HIGH if energy > 0.67 else NEUTRAL.level)
    character = replace(NEUTRAL, energy=energy, level=level, shape=ALL_DAY, stretches=3)
    day = base.day + timedelta(days=index)
    scale = scale_of_day(GLITCH, HOME, day, character)
    plan = replace(base, character=character, scale=scale, day=day, choices={}, notes=[])
    with patch.object(scape_glitch, "MODES", (mode,)):
        return scape_glitch.compose(plan), plan


def _kicks(score: Score, a: float = 0.0, b: float | None = None) -> list[float]:
    end = score.length if b is None else b
    return [n.t for n in score.notes if n.voice == "glitch:kick" and a <= n.t < end]


def _in_arcs(plan: Plan, names: tuple[str, ...]) -> tuple[float, float]:
    spans = [(a, b) for name, a, b in plan.arc() if name in names]
    return min(a for a, _ in spans), max(b for _, b in spans)


@pytest.mark.parametrize("mode", [HALF_TIME, DRIVING], ids=lambda m: m.name)
def test_half_time_and_driving_play_well_under_a_kick_a_beat_of_the_kit(
    mode: scape_glitch.Mode,
) -> None:
    """On average the kick lands on fewer than seven in ten beats of the count of the kit, never four on the floor."""
    for index in DAYS:
        for energy in (0.5, 0.95):
            score, _ = _made(index, energy, mode)
            kicks = _kicks(score)
            assert kicks
            minutes = {int(t // SECONDS) for t in kicks}
            kit_beats = len(minutes) * SECONDS / 60.0 * score.tempo * KIT_BEATS_PER_BEAT
            assert len(kicks) / kit_beats < MOST_KICKS_PER_KIT_BEAT


@pytest.mark.parametrize("mode", [HALF_TIME, DRIVING], ids=lambda m: m.name)
def test_the_kick_moves_with_the_form(mode: scape_glitch.Mode) -> None:
    """The sections of a busy piece differ in kicks per minute by a good margin, they are not one flat groove."""
    score, plan = _made(1, 0.95, mode)
    bar = 4.0 * 60.0 / score.tempo
    rates = [
        len(_kicks(score, a, b)) / (b - a) * SECONDS
        for a, b, role, _ in plan.sections(bar)
        if role in {BUILD, CLIMAX, RELEASE, PLATEAU} and b - a >= MIN_SECTION_SECONDS
    ]
    played = [rate for rate in rates if rate > 0.0]
    assert len(played) >= 2
    assert max(played) >= SECTION_SPREAD * min(played)


def test_heavy_mode_has_drums_and_edits_through_the_day() -> None:
    """Even on a quiet day, heavy plays sparse drums through most of the day and edits about once a minute."""
    for index in DAYS:
        score, plan = _made(index, 0.1, HEAVY)
        assert score.glitch is not None
        a, b = _in_arcs(plan, DAY_ARCS)
        minutes = int((b - a) // SECONDS)
        with_drums = {
            int((n.t - a) // SECONDS)
            for n in score.notes
            if n.layer in {"kick", "snare"} and a <= n.t < b
        }
        assert len(with_drums) / minutes >= HEAVY_DRUM_SHARE
        edits = [e for e in score.glitch.edits if a <= e.t < b]
        assert len(edits) / (b - a) * SECONDS >= HEAVY_EDITS_PER_MINUTE
