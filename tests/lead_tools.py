"""Helpers of the tests of who leads: a day of the demo house in any style, and the level of a lead in its phrases."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from entropy_fm import lead, scape_ambient, scape_berlin, scape_glitch, scape_lofi, scape_ritual
from entropy_fm.day_analysis import analyze_day
from entropy_fm.plan import PieceOptions
from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all
from entropy_fm.scape import COMPOSERS, scape_score
from entropy_fm.synth import Scale

from .scape_tools import the_demo

if TYPE_CHECKING:
    from entropy_fm.plan import Plan
    from entropy_fm.score import Score

ROOMS = ("kitchen", "bedroom", "office", "bathroom", "living_room", "_house", "hall", "garage")
HOME = Scale(2, "dorian")
DAY = 7
DB_PER_BEL = 20.0
SILENCE_FLOOR = 1e-9
BAR_BEATS = 4
SECONDS_PER_MINUTE = 60.0
LEADS = {
    "ambient": scape_ambient.LEADS,
    "berlin_school": scape_berlin.LEADS,
    "lofi": scape_lofi.LEADS,
    "ritual": scape_ritual.LEADS,
    "glitch": scape_glitch.LEADS,
}
BY_TRAIT = {
    "ambient": scape_ambient.LEAD_BY_TRAIT,
    "berlin_school": scape_berlin.LEAD_BY_TRAIT,
    "lofi": scape_lofi.LEAD_BY_TRAIT,
    "ritual": scape_ritual.LEAD_BY_TRAIT,
    "glitch": scape_glitch.LEAD_BY_TRAIT,
}
# the melodic and harmonic layers a lead is measured against; the drums, the bass and the drones are not among them
MELODIC = {
    "ambient": ("piano", "pads", "strings"),
    "berlin_school": ("seqA", "seqB", "seqC", "lead", "strings", "choir"),
    "lofi": ("keys", "chop", "pluck", "strings"),
    "ritual": ("plucks", "slit", "lead"),
    "glitch": ("str", "stab", "piano"),
}
# the lead sits this far above the loudest other melodic or harmonic layer, in dB
LEVEL_WINDOW_DB = (0.0, 4.0)
# the Glitch piano is set 6 dB down on purpose and sits this far over the strings and stabs on the median of a short piece
GLITCH_PIANO_WINDOW_DB = (0.0, 6.0)
# a short piece has few led phrases and every room and day lands somewhere else, so a lead is judged by the median of these
LEVEL_DAYS = (7, 9, 11, 13)
LEVEL_SAMPLES = 12
# a layer that starts fewer notes than this in the phrase, or sounds this many dB under the lead, is not there to compare with
LEAST_NOTES = 3
ABSENT_DB = 15.0


@lru_cache(maxsize=128)
def made(style: str, room: str, index: int = DAY) -> tuple[Score, Plan]:
    """Return the score and the plan of a day of the demo house in a style, for a room key."""
    house = the_demo()
    series = house.days[index]["kitchen"] + house.days[index]["sun"]
    return scape_score(
        series,
        analyze_day(series),
        style,
        f"{room}|{house.day_of(index).isoformat()}|_scape|{style}",
        HOME,
        quality=STANDARD,
        options=PieceOptions(day=house.day_of(index), area_key=room),
    )


def compose_with(style: str, plan: Plan, triggers: list[tuple[float, float, str]]) -> Score:
    """Return the score of a plan with other happenings in the house."""
    return COMPOSERS[style](replace(plan, triggers=triggers, choices={}, notes=[]))


def room_without_lead(style: str, layer: str) -> str:
    """Return a room of the demo house whose own lead in the style is not the layer."""
    return next(room for room in ROOMS if made(style, room)[1].choices[lead.LEAD_CHOICE] != layer)


def bar_of(style: str, score: Score, plan: Plan) -> float:
    """Return the length of a bar of a piece in seconds, the unit its form is cut in."""
    if style == "ambient":
        return scape_ambient.SECTION_BAR
    if style == "ritual":
        return float(plan.choices["pluck cell length"].split()[0])
    tempo = score.glitch.tempo if score.glitch is not None else score.tempo
    return BAR_BEATS * SECONDS_PER_MINUTE / tempo


def lead_phrases(style: str, score: Score, plan: Plan) -> list[tuple[float, float]]:
    """Return the phrases of the form that can have a lead: build, peak or plateau, by day."""
    bar = bar_of(style, score, plan)
    phrase = plan.form_grid(bar)
    allowed = lead.daytime_phrases(plan, plan.sections(bar), phrase)
    return [(i * phrase, (i + 1) * phrase) for i, ok in enumerate(allowed) if ok]


def count(score: Score, layer: str, span: tuple[float, float]) -> int:
    """Return how many notes a layer starts in a span."""
    return sum(1 for n in score.notes if n.layer == layer and span[0] <= n.t < span[1])


def _solo_db(score: Score, layer: str, start: int, end: int) -> float:
    audio = render_all(
        score, STANDARD, "a", None, start_s=start, end_s=end, solo=frozenset({layer})
    )
    return float(DB_PER_BEL * np.log10(np.sqrt(np.mean(audio**2)) + SILENCE_FLOOR))


def lead_over_others(style: str, score: Score, leader: str) -> tuple[float, str] | None:
    """
    Return how many dB the leading layer sits over the loudest other melodic layer in a phrase it leads, and that layer.

    The score is measured without the compressor and the leveler on a copy, in the middle one of the phrases the layer
    leads; a layer with a long hall is judged with it. The master gain of Cinematic Glitch stays, which keeps the
    limiter off the peaks of a loud piano. None means that no other layer plays enough to compare with.
    """
    assert score.leads is not None
    phrases = [i for i, who in enumerate(score.leads.phrases) if who == leader]
    middle = phrases[len(phrases) // 2]
    start, end = int(middle * score.leads.phrase), int((middle + 1) * score.leads.phrase)
    bare = replace(
        score, compress_db=None, gain_db=score.gain_db if score.glitch is not None else None
    )
    mine = _solo_db(bare, leader, start, end)
    others = {
        layer: _solo_db(bare, layer, start, end)
        for layer in MELODIC[style]
        if layer != leader
        and layer in bare.layers
        and count(bare, layer, (start, end)) >= LEAST_NOTES
    }
    others = {layer: db for layer, db in others.items() if db > mine - ABSENT_DB}
    if not others:
        return None
    loudest = max(others, key=others.__getitem__)
    return mine - others[loudest], loudest


def lead_level(style: str, layer: str) -> float | None:
    """Return the median over rooms and days of how far a layer sits over the others while it leads, None if never."""
    levels: list[float] = []
    for room in ROOMS:
        for day in LEVEL_DAYS:
            score, plan = made(style, room, day)
            if plan.choices[lead.LEAD_CHOICE] != layer or len(levels) >= LEVEL_SAMPLES:
                continue
            assert score.leads is not None
            measured = (
                lead_over_others(style, score, layer) if layer in score.leads.phrases else None
            )
            if measured is not None:
                levels.append(measured[0])
    return float(np.median(levels)) if levels else None
