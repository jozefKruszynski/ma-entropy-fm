"""Tests of who leads in every style: the room's own lead, the happenings handing a phrase over, the night, the levels."""

from __future__ import annotations

import pytest

from entropy_fm import lead, scape_ambient

from .lead_tools import (
    BY_TRAIT,
    GLITCH_PIANO_WINDOW_DB,
    LEADS,
    LEVEL_WINDOW_DB,
    ROOMS,
    compose_with,
    count,
    lead_level,
    lead_phrases,
    made,
    room_without_lead,
)

STYLES = tuple(LEADS)
GLITCH_PIANO = ("glitch", "piano")
DAYS = (7, 9, 11, 13)
# a happening of the house for each kind of sound
KIND_OF = {lead.PERCUSSIVE: "door", lead.RHYTHMIC: "appliance", lead.SUSTAINED: "light"}
TRIGGER_STRENGTH = 0.5
SECONDS_BEFORE = 1.0


@pytest.mark.parametrize("style", STYLES)
def test_rooms_have_different_default_leads(style: str) -> None:
    """Over a handful of rooms the default lead is not always the same, and every room records one."""
    leads = set()
    for room in ROOMS:
        score, plan = made(style, room)
        leads.add(plan.choices[lead.LEAD_CHOICE])
        assert lead.LEAD_HANDOVERS_CHOICE in score.choices
        assert score.choices[lead.LEAD_CHOICE] == plan.choices[lead.LEAD_CHOICE]
    assert leads <= set(LEADS[style])
    assert len(leads) > 1


@pytest.mark.parametrize("style", STYLES)
def test_a_room_keeps_its_default_lead_on_different_days(style: str) -> None:
    """The lead of a room does not change from day to day."""
    for room in ROOMS[:3]:
        assert len({made(style, room, day)[1].choices[lead.LEAD_CHOICE] for day in DAYS}) == 1


@pytest.mark.parametrize("trait", [lead.PERCUSSIVE, lead.RHYTHMIC, lead.SUSTAINED])
@pytest.mark.parametrize("style", STYLES)
def test_a_happening_hands_a_phrase_to_the_layer_of_its_kind(style: str, trait: str) -> None:
    """A door, an appliance or a light in the daytime hands the next phrase to the layer that suits it."""
    layer = BY_TRAIT[style][trait]
    room = room_without_lead(style, layer)
    score, plan = made(style, room)
    base = compose_with(style, plan, [])
    for span in lead_phrases(style, score, plan)[1:]:
        trigger = (span[0] - SECONDS_BEFORE, TRIGGER_STRENGTH, KIND_OF[trait])
        led = compose_with(style, plan, [trigger])
        if led.choices[lead.LEAD_HANDOVERS_CHOICE] == "1" and count(led, layer, span) > count(
            base, layer, span
        ):
            return
    raise AssertionError(f"no phrase of {style} was handed to the {layer}")


@pytest.mark.parametrize("style", STYLES)
def test_handovers_are_spaced(style: str) -> None:
    """At most one phrase in `HANDOVER_GAP` is handed over."""
    score, plan = made(style, ROOMS[0])
    spans = lead_phrases(style, score, plan)
    triggers = [(span[0] - SECONDS_BEFORE, TRIGGER_STRENGTH, "light") for span in spans]
    handed = int(compose_with(style, plan, triggers).choices[lead.LEAD_HANDOVERS_CHOICE])
    assert handed <= len(spans) / lead.HANDOVER_GAP + 1


@pytest.mark.parametrize("style", STYLES)
def test_nobody_leads_at_night_and_nothing_is_handed_over_then(style: str) -> None:
    """A night phrase has no lead, and a happening in the night hands nothing over."""
    score, plan = made(style, ROOMS[0])
    assert score.leads is not None
    day = {start for start, _ in lead_phrases(style, score, plan)}
    night = [
        index
        for index in range(len(score.leads.phrases))
        if plan.at(plan.night, index * score.leads.phrase) >= lead.LEAD_DAYTIME
    ]
    assert night
    for index in night:
        assert score.leads.phrases[index] is None
        assert index * score.leads.phrase not in day
    triggers = [
        ((index - 1) * score.leads.phrase, TRIGGER_STRENGTH, kind)
        for index in night
        for kind in KIND_OF.values()
    ]
    led = compose_with(style, plan, triggers)
    assert led.leads is not None
    assert all(led.leads.phrases[index] is None for index in night)


@pytest.mark.parametrize(
    ("style", "layer"), [(style, layer) for style in STYLES for layer in LEADS[style]]
)
def test_the_lead_sits_at_most_a_few_decibels_over_the_others(style: str, layer: str) -> None:
    """In the phrases it leads, a layer is inside its level window over the loudest other melodic layer, hall included."""
    found = lead_level(style, layer)
    low, high = GLITCH_PIANO_WINDOW_DB if (style, layer) == GLITCH_PIANO else LEVEL_WINDOW_DB
    if found is None:
        pytest.skip(f"{style} {layer} leads in none of the sampled rooms")
    assert low <= found <= high, found


def test_the_ambient_lead_is_a_smaller_step_than_the_others() -> None:
    """Ambient stays calm: its lead comes up less than the lead of the styles with a beat."""
    assert scape_ambient.LEAD_STEP_DB < lead.LEAD_STEP_DB


def test_a_handover_in_glitch_does_not_change_the_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The splices of Cinematic Glitch are the same with a lead handed over and without."""
    score, plan = made("glitch", room_without_lead("glitch", "str"))
    spans = lead_phrases("glitch", score, plan)
    triggers = [(span[0] - SECONDS_BEFORE, TRIGGER_STRENGTH, "light") for span in spans[1:4]]
    handed = compose_with("glitch", plan, triggers)
    assert int(handed.choices[lead.LEAD_HANDOVERS_CHOICE]) > 0
    monkeypatch.setattr(lead, "HANDOVER_TRAIT", {})
    plain = compose_with("glitch", plan, triggers)
    assert plain.choices[lead.LEAD_HANDOVERS_CHOICE] == "0"
    assert handed.glitch is not None
    assert plain.glitch is not None
    assert handed.glitch.edits == plain.glitch.edits
