"""Tests that the motif of a room is developed between statements and stated at most twice in a piece."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from itertools import pairwise

import pytest

from entropy_fm.character import (
    ALL_DAY,
    FLAT,
    HIGH,
    LOW,
    NEUTRAL,
    DayCharacter,
    room_motif,
    scale_of_day,
)
from entropy_fm.harmony import (
    LEAD_FLOOR_MIDI,
    MIN_MOTIF_NOTE_BEATS,
    MOTIF_CHOICE,
    MOTIF_STATEMENTS,
    develop_motif,
    lead_octave,
    motif_seed,
    motif_sites,
)
from entropy_fm.scape import COMPOSERS
from entropy_fm.styles import STYLES
from entropy_fm.synth import Scale

from .scape_tools import demo

QUIET = replace(NEUTRAL, energy=0.08, level=LOW, shape=FLAT, warmth=-0.8, stretches=1)
BUSY = replace(NEUTRAL, energy=0.95, level=HIGH, shape=ALL_DAY, warmth=0.8, stretches=4)
HOME = Scale(2, "dorian")
ROOMS = ("kitchen", "bedroom", "office", "bathroom", "living_room", "_house")
SEEDS = range(40)
STYLE_KEYS = ("ambient", "berlin_school", "lofi", "ritual", "glitch")
FIRST_DAY = date(2026, 11, 16)


def _contour(motif: tuple[tuple[int, float], ...]) -> list[int]:
    """Return the direction of each interval of a motif: 1 up, -1 down, 0 level."""
    return [(b > a) - (b < a) for (a, _), (b, _) in pairwise(motif)]


def test_the_first_statement_is_the_motif_itself() -> None:
    """Statement 0 is returned unchanged for any seed."""
    for room in ROOMS:
        motif = room_motif(room)
        assert all(develop_motif(motif, 0, seed) == motif for seed in SEEDS)


def test_later_statements_differ_but_open_like_the_motif() -> None:
    """Every variation differs from the motif and keeps the direction of its first two intervals."""
    for room in ROOMS:
        motif = room_motif(room)
        for seed in SEEDS:
            developed = develop_motif(motif, 1, seed)
            assert developed != motif, (room, seed)
            assert _contour(developed)[:2] == _contour(motif)[:2], (room, seed)
            assert all(beats >= MIN_MOTIF_NOTE_BEATS for _, beats in developed)


def test_development_is_deterministic_and_varies_with_the_seed() -> None:
    """The same seed gives the same variation; across seeds a room hears several different ones."""
    motif = room_motif("kitchen")
    assert develop_motif(motif, 1, 7) == develop_motif(motif, 1, 7)
    assert len({develop_motif(motif, 1, seed) for seed in SEEDS}) >= 3


def test_a_variation_can_be_kept_short() -> None:
    """With a cap on the length no variation is longer than it."""
    cap = 8.0
    for room in ROOMS:
        for seed in SEEDS:
            developed = develop_motif(room_motif(room), 1, seed, max_beats=cap)
            assert sum(beats for _, beats in developed) <= max(
                cap, sum(beats for _, beats in room_motif(room))
            )


def test_the_variation_stays_above_the_lead_floor_when_the_octave_is_chosen_for_it() -> None:
    """The octave picked for a developed motif keeps all of its notes at or above middle C."""
    scale = scale_of_day(STYLES["berlin_school"], HOME, FIRST_DAY, BUSY)
    for room in ROOMS:
        for seed in SEEDS:
            developed = develop_motif(room_motif(room), 1, seed)
            octave = lead_octave(scale, developed)
            assert min(scale.midi(degree, octave) for degree, _ in developed) >= LEAD_FLOOR_MIDI


def test_the_sites_are_the_first_and_the_last() -> None:
    """At most two places carry the motif: the first one and the last when it is another."""
    assert motif_sites([]) == []
    assert motif_sites([4]) == [4]
    assert motif_sites([2, 5]) == [2, 5]
    assert motif_sites([1, 3, 6, 9]) == [1, 9]
    assert len(motif_sites(list(range(20)))) == MOTIF_STATEMENTS


@pytest.mark.parametrize("style", STYLE_KEYS)
@pytest.mark.parametrize("character", [QUIET, BUSY], ids=["quiet", "busy"])
def test_each_style_states_the_motif_at_least_once_and_at_most_twice(
    style: str, character: DayCharacter
) -> None:
    """The motif of the room is stated once or twice in a piece, not in every peak or plateau."""
    _, base = demo(style)
    assert base.day is not None
    scale = scale_of_day(STYLES[style], HOME, base.day, character)
    plan = replace(base, character=character, scale=scale, choices={}, notes=[])
    score = COMPOSERS[style](plan)
    assert 1 <= int(score.choices[MOTIF_CHOICE]) <= MOTIF_STATEMENTS, style


def test_the_variation_changes_from_day_to_day() -> None:
    """The seed of a room differs from day to day, so the variation of the second statement does too."""
    _, base = demo("berlin_school")
    seeds = {motif_seed(replace(base, day=FIRST_DAY + timedelta(days=n))) for n in range(10)}
    assert len(seeds) == 10
    variations = {develop_motif(room_motif("kitchen"), 1, seed) for seed in seeds}
    assert len(variations) >= 3
