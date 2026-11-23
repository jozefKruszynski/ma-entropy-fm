"""Tests of the balance of Lo-fi: a quiet day outside the night keeps a soft kit, and the plucked line is heard."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np

from entropy_fm import scape_lofi
from entropy_fm.character import ALL_DAY, FLAT, HIGH, LOW, NEUTRAL, scale_of_day
from entropy_fm.plan import Plan
from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all
from entropy_fm.score import Score
from entropy_fm.styles import STYLES
from entropy_fm.synth import Scale

from .scape_tools import demo

QUIET = replace(NEUTRAL, energy=0.06, level=LOW, shape=FLAT, warmth=-0.8, stretches=1)
BUSY = replace(NEUTRAL, energy=0.95, level=HIGH, shape=ALL_DAY, warmth=0.8, stretches=4)
HOME = Scale(2, "dorian")
# a note of the kit counts as outside the night where the night curve is under this
DAYTIME = 0.3
# the pluck is heard when it is no more than this many dB under the keys while it plays
MOST_UNDER_KEYS_DB = 12.0
WINDOW_SECONDS = 10
LEAD_SECONDS = 2
SILENCE_FLOOR = 1e-9


@cache
def _made(*, quiet: bool) -> tuple[Score, Plan]:
    """Return the score and the plan of one day of the demo house, made on a clearly quiet or busy character."""
    _, base = demo("lofi")
    assert base.day is not None
    character = QUIET if quiet else BUSY
    scale = scale_of_day(STYLES["lofi"], HOME, base.day, character)
    plan = replace(base, character=character, scale=scale, choices={}, notes=[])
    return scape_lofi.compose(plan), plan


def _db(samples: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(samples**2)) + SILENCE_FLOOR))


def test_a_quiet_day_keeps_a_soft_kit_outside_the_night() -> None:
    """The drums of a quiet day play by day, sparser and softer than on a busy one, and rest at night."""
    quiet, plan = _made(quiet=True)
    busy, _ = _made(quiet=False)
    drums = [n for n in quiet.notes if n.layer == "drums"]
    by_day = [n for n in drums if plan.at(plan.night, n.t) < DAYTIME]
    assert by_day
    assert {"drum:kick_lofi", "drum:hat_closed"} & {n.voice for n in by_day}
    assert len(drums) < len([n for n in busy.notes if n.layer == "drums"])
    deep_night = [n for n in drums if plan.at(plan.night, n.t) > 1.0 - DAYTIME]
    assert len(deep_night) < len(by_day)


def test_the_plucked_line_is_heard_behind_the_keys() -> None:
    """Where the plucked line plays, it is no more than 12 dB under the keys."""
    score, _ = _made(quiet=False)
    plucks = sorted(n.t for n in score.notes if n.layer == "pluck")
    assert plucks
    start = int(max(plucks[0] - LEAD_SECONDS, 0))
    end = start + WINDOW_SECONDS
    pluck = _db(
        render_all(score, STANDARD, "a", None, start_s=start, end_s=end, solo=frozenset({"pluck"}))
    )
    keys = _db(
        render_all(score, STANDARD, "a", None, start_s=start, end_s=end, solo=frozenset({"keys"}))
    )
    assert keys - pluck <= MOST_UNDER_KEYS_DB
    assert pluck < keys
