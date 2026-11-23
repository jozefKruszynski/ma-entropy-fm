"""Tests of the trailer piano of Cinematic Glitch: its lead candidates, its variants, its levels and the stabs it leaves alone."""

from __future__ import annotations

import numpy as np
import pytest

from entropy_fm import lead, scape_glitch
from entropy_fm.harmony import PIANO_HALL_SEND
from entropy_fm.score import Score
from entropy_fm.synth import SEMITONES_PER_OCTAVE

from .lead_tools import (
    BY_TRAIT,
    GLITCH_PIANO_WINDOW_DB,
    compose_with,
    lead_over_others,
    made,
    room_without_lead,
)

STYLE = "glitch"
PIANO = "piano"
DAYS = (7, 9, 11)
# rooms of the demo house and a few more, to find rooms of every variant
ROOMS = (
    "kitchen",
    "bedroom",
    "office",
    "bathroom",
    "living_room",
    "_house",
    "hall",
    "garage",
    "attic",
    "cellar",
    "study",
    "porch",
    "nursery",
    "pantry",
)
PIANO_LEVEL_DAYS = (7, 9)
LEVELS_PER_VARIANT = 10
STAB_LAYER = "stab"


def _stabs(score: Score) -> list[tuple[float, float, float, float, str, int]]:
    return [
        (n.t, n.dur, n.midi, n.vel, n.voice, n.patch) for n in score.notes if n.layer == STAB_LAYER
    ]


def test_the_lead_candidates_are_the_piano_and_the_strings() -> None:
    """The piano takes the percussive and the rhythmic phrases, the strings the sustained ones."""
    assert set(scape_glitch.LEADS) == {PIANO, "str"}
    assert BY_TRAIT[STYLE] == {
        lead.PERCUSSIVE: PIANO,
        lead.RHYTHMIC: PIANO,
        lead.SUSTAINED: "str",
    }


@pytest.mark.parametrize("candidates", [(PIANO,), ("str",)])
def test_the_stabs_play_no_lead_line(
    monkeypatch: pytest.MonkeyPatch, candidates: tuple[str, ...]
) -> None:
    """The stabs are the same whoever leads: no line of the motif, no backing off."""
    _, plan = made(STYLE, "kitchen")
    plain = compose_with(STYLE, plan, [])
    monkeypatch.setattr(scape_glitch, "LEADS", candidates)
    led = compose_with(STYLE, plan, [])
    assert {phrase for phrase in led.leads.phrases if phrase} == set(candidates)  # type: ignore[union-attr]
    assert _stabs(led) == _stabs(plain)


def test_a_room_keeps_its_piano_variant_and_the_rooms_differ() -> None:
    """The variant of a room is the same on every day, and the rooms do not all play the same one."""
    seen = set()
    for room in ROOMS[:8]:
        variants = {made(STYLE, room, day)[1].choices[scape_glitch.PIANO_CHOICE] for day in DAYS}
        assert len(variants) == 1
        seen |= variants
    assert len(seen) > 1
    assert seen <= {variant.label for variant in scape_glitch.PIANO_VARIANTS}


def test_the_piano_plays_intervals_under_the_melody() -> None:
    """Every piano note comes with a lower one: an octave or a fifth or sixth under, and a soft depth note starts a statement."""
    for room in ROOMS[:4]:
        score, plan = made(STYLE, room)
        notes = [n for n in score.notes if n.layer == PIANO]
        assert notes
        by_start: dict[float, list[float]] = {}
        for note in notes:
            by_start.setdefault(note.t, []).append(note.midi)
        octaves = plan.choices[scape_glitch.PIANO_CHOICE].endswith("octaves")
        for pitches in by_start.values():
            top = max(pitches)
            assert len(pitches) in {2, 3}
            gap = top - sorted(pitches)[-2]
            assert gap == SEMITONES_PER_OCTAVE if octaves else 5 <= gap <= 9
        soft = [n for n in notes if n.vel < scape_glitch.PIANO_VELOCITY]
        assert soft
        assert all(
            n.vel == pytest.approx(scape_glitch.PIANO_VELOCITY * scape_glitch.PIANO_DEPTH_SHARE)
            for n in soft
        )


def test_the_piano_is_in_the_edits_that_touch_the_strings_or_the_stabs() -> None:
    """A splice of the strings or the stabs takes the piano with it, and none overlaps another on the piano."""
    score, _ = made(STYLE, "bedroom")
    assert score.glitch is not None
    edits = score.glitch.edits
    with_piano = [e for e in edits if PIANO in e.layers]
    assert with_piano
    assert all({"str", STAB_LAYER} & set(e.layers) for e in with_piano)
    ends = [e.t + e.dur for e in with_piano]
    assert all(e.t >= end for e, end in zip(with_piano[1:], ends, strict=False))
    assert score.glitch.duck[PIANO] == scape_glitch.DUCK["str"]


def test_the_piano_has_its_own_hall_and_a_little_of_the_room() -> None:
    """The piano layer asks for the long hall with a strong send, and for a small send to the shared reverb."""
    score, _ = made(STYLE, "bedroom")
    layer = score.layers[PIANO]
    assert layer.hall == PIANO_HALL_SEND > 1.0
    assert 0.0 < layer.reverb < layer.hall
    assert all(other.hall == 0.0 for key, other in score.layers.items() if key != PIANO)


def test_the_piano_sits_a_few_decibels_over_the_others_in_every_variant() -> None:
    """In the phrases it leads, each variant sits 0 to 6 dB over the loudest other melodic layer at the median, measured with its hall."""
    found: dict[str, list[float]] = {variant.label: [] for variant in scape_glitch.PIANO_VARIANTS}
    for room in ROOMS:
        for day in PIANO_LEVEL_DAYS:
            score, plan = made(STYLE, room, day)
            label = plan.choices[scape_glitch.PIANO_CHOICE]
            if len(found[label]) >= LEVELS_PER_VARIANT or PIANO not in score.leads.phrases:  # type: ignore[union-attr]
                continue
            measured = lead_over_others(STYLE, score, PIANO)
            if measured is not None:
                found[label].append(measured[0])
    for label, levels in found.items():
        assert levels, label
        low, high = GLITCH_PIANO_WINDOW_DB
        assert low <= float(np.median(levels)) <= high, (label, levels)


def test_a_handover_gives_the_piano_notes_in_a_room_that_does_not_lead_with_it() -> None:
    """A door hands the next phrase to the piano, which then plays."""
    _, plan = made(STYLE, room_without_lead(STYLE, PIANO))
    base = compose_with(STYLE, plan, [])
    assert base.leads is not None
    spans = [i for i, who in enumerate(base.leads.phrases) if who is not None]
    for index in spans[1:]:
        trigger = (index * base.leads.phrase - 1.0, 0.5, "door")
        led = compose_with(STYLE, plan, [trigger])
        if led.choices[lead.LEAD_HANDOVERS_CHOICE] == "1" and any(
            n.layer == PIANO for n in led.notes
        ):
            return
    raise AssertionError("no phrase was handed to the piano")
