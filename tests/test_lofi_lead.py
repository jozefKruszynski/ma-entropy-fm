"""Tests of who leads in Lo-fi: the room's own lead, the day handing a phrase to another layer, and the lines."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pytest

from entropy_fm import lead, scape_lofi
from entropy_fm.day_analysis import analyze_day
from entropy_fm.plan import FORM_BARS, PieceOptions, Plan
from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all
from entropy_fm.scape import scape_score
from entropy_fm.score import LAYER_KEYS, Score
from entropy_fm.synth import Scale

from .scape_tools import the_demo

ROOMS = ("kitchen", "bedroom", "office", "bathroom", "living_room", "_house", "hall", "garage")
HOME = Scale(2, "dorian")
DAY = 7
# one room of each default lead on the demo days
ROOM_OF = {"keys": "living_room", "pluck": "bedroom", "strings": "kitchen"}
# a happening that hands the next phrase to a layer
KIND_OF = {"keys": "appliance", "pluck": "door", "strings": "light"}
TRIGGER_STRENGTH = 0.5
SECONDS_BEFORE = 1.0
# most of the phrases of the lead sections have the line of the lead
MOST_PHRASES = 0.75
OVERLAP = 1e-6
SILENCE_FLOOR = 1e-9
DB_PER_BEL = 20.0


@lru_cache(maxsize=64)
def _made(room: str, index: int = DAY) -> tuple[Score, Plan]:
    """Return the score and the plan of a day of the demo house in Lo-fi, for a room key."""
    house = the_demo()
    series = house.days[index]["kitchen"] + house.days[index]["sun"]
    return scape_score(
        series,
        analyze_day(series),
        "lofi",
        f"{room}|{house.day_of(index).isoformat()}|_scape|lofi",
        HOME,
        quality=STANDARD,
        options=PieceOptions(day=house.day_of(index), area_key=room),
    )


def _compose(plan: Plan, triggers: list[tuple[float, float, str]]) -> Score:
    """Return the score of a plan with other happenings in the house."""
    return scape_lofi.compose(replace(plan, triggers=triggers, choices={}, notes=[]))


def _lead_phrases(score: Score, plan: Plan) -> list[tuple[float, float]]:
    """Return the phrases that can have a lead: build, peak or plateau, by day."""
    bar = 4 * 60.0 / score.tempo
    phrase = FORM_BARS * bar
    found = []
    for start, end, role, _ in plan.sections(bar):
        first = int(np.ceil(start / phrase))
        for index in range(first, int(end / phrase)):
            at = index * phrase
            if role in lead.LEAD_ROLES and plan.at(plan.night, at) < lead.LEAD_DAYTIME:
                found.append((at, at + phrase))
    return found


def _count(score: Score, layer: str, span: tuple[float, float], *, line: bool = False) -> int:
    return len(
        [
            n
            for n in score.notes
            if n.layer == layer and span[0] <= n.t < span[1] and (n.vib > 0 or not line)
        ]
    )


def _handed(layer: str) -> tuple[Score, Score, tuple[float, float]]:
    """Return the score of the day without happenings, with one that hands a phrase to the layer, and that phrase."""
    room = next(r for r in ROOMS if _made(r)[1].choices["lead"] != layer)
    base_score, plan = _made(room)
    base = _compose(plan, [])
    for span in _lead_phrases(base_score, plan)[1:]:
        trigger = (span[0] - SECONDS_BEFORE, TRIGGER_STRENGTH, KIND_OF[layer])
        led = _compose(plan, [trigger])
        if led.choices[lead.LEAD_HANDOVERS_CHOICE] == "1":
            return base, led, span
    raise AssertionError("no phrase could be handed over")


def test_rooms_have_different_default_leads() -> None:
    """Over a handful of rooms the default lead is not always the same, and every room has one."""
    leads = {_made(room)[1].choices[lead.LEAD_CHOICE] for room in ROOMS}
    assert leads <= set(scape_lofi.LEADS)
    assert len(leads) > 1


def test_a_room_keeps_its_default_lead_on_different_days() -> None:
    """The lead of a room does not change from day to day."""
    for room in ROOMS[:3]:
        leads = {_made(room, index)[1].choices[lead.LEAD_CHOICE] for index in (7, 9, 11, 13)}
        assert len(leads) == 1


@pytest.mark.parametrize("layer", ["pluck", "strings", "keys"])
def test_a_happening_hands_a_phrase_to_its_layer(layer: str) -> None:
    """A door, a light or an appliance in the daytime hands the next phrase to the pluck, the strings or the keys."""
    base, led, span = _handed(layer)
    if layer == "keys":
        assert _count(led, "keys", span) > _count(base, "keys", span)
    elif layer == "pluck":
        assert _count(led, "pluck", span) > _count(base, "pluck", span)
    else:
        assert _count(led, "strings", span, line=True) > _count(base, "strings", span, line=True)


def test_handovers_are_spaced() -> None:
    """At most one phrase in `HANDOVER_GAP` is handed over."""
    score, plan = _made("kitchen")
    spans = _lead_phrases(score, plan)
    triggers = [(span[0] - SECONDS_BEFORE, TRIGGER_STRENGTH, "door") for span in spans]
    handed = int(_compose(plan, triggers).choices[lead.LEAD_HANDOVERS_CHOICE])
    assert 0 < handed <= len(spans) / lead.HANDOVER_GAP + 1


def test_a_plucked_line_runs_through_the_phrases_when_the_pluck_leads() -> None:
    """The pluck plays in most daytime phrases of the build, peak and plateau when it is the lead."""
    score, plan = _made(ROOM_OF["pluck"])
    led = _compose(plan, [])
    spans = _lead_phrases(score, plan)
    assert len(spans) > 2
    playing = [span for span in spans if _count(led, "pluck", span)]
    assert len(playing) >= MOST_PHRASES * len(spans)
    assert sum(_count(led, "pluck", span) for span in spans) > sum(
        _count(_compose(_made(ROOM_OF["keys"])[1], []), "pluck", span) for span in spans
    )


def test_the_strings_line_is_one_voice_over_the_bed() -> None:
    """When the strings lead, the top line never has two notes sounding at once, and the bed stays."""
    score, plan = _made(ROOM_OF["strings"])
    led = _compose(plan, [])
    line = sorted((n for n in led.notes if n.layer == "strings" and n.vib > 0), key=lambda n: n.t)
    bed = [n for n in led.notes if n.layer == "strings" and n.vib == 0]
    assert line
    assert bed
    assert all(b.t - a.t - a.dur >= -OVERLAP for a, b in pairwise(line))
    spans = _lead_phrases(score, plan)
    playing = [span for span in spans if _count(led, "strings", span, line=True)]
    assert len(playing) >= MOST_PHRASES * len(spans)


@pytest.mark.parametrize("layer", ["pluck", "strings", "keys"])
def test_the_lead_is_louder_in_its_phrase_than_when_it_does_not_lead(layer: str) -> None:
    """The same notes of a layer are louder in a phrase it leads than in one where another layer does."""
    base, led, span = _handed(layer)
    dry = replace(led, compress_db=None, gain_db=None)
    back = replace(dry, layers={**dry.layers, layer: base.layers[layer]})
    start, end = int(span[0]), int(span[1])
    solo = frozenset({layer})

    def level(score: Score) -> float:
        audio = render_all(score, STANDARD, "a", None, start_s=start, end_s=end, solo=solo)
        return float(DB_PER_BEL * np.log10(np.sqrt(np.mean(audio**2)) + SILENCE_FLOOR))

    # the keys are trimmed down while they lead, so they only have to come up
    least = 0.0 if layer == LAYER_KEYS else lead.LEAD_STEP_DB
    assert level(dry) - level(back) > least
