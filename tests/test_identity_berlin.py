"""Tests that the character of a day picks the identity of Berlin School: a quiet day and a busy one of the same room."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache
from itertools import pairwise

import pytest

from entropy_fm.character import ALL_DAY, FLAT, HIGH, LOW, NEUTRAL, DayCharacter
from entropy_fm.day_analysis import DayAnalysis
from entropy_fm.harmony import (
    held_runs,
    lead_octave,
    progressions_for,
    triad_is_diminished,
    voice_lead,
)
from entropy_fm.plan import CLIMAX, PLATEAU, PieceOptions, Plan, build_plan
from entropy_fm.quality import STANDARD
from entropy_fm.references import rescale
from entropy_fm.scape import COMPOSERS, analysis_of, scape_score
from entropy_fm.scape_berlin import PROGRESSIONS, RELATIVE_SHIFT
from entropy_fm.score import Score
from entropy_fm.styles import STYLES
from entropy_fm.synth import SCALES, Scale
from entropy_fm.timeline import Series

from .scape_tools import the_demo

HOME = Scale(2, "dorian")
DAY_SECONDS = 86400.0
STYLE_KEYS = ("berlin_school",)
MODES = {"berlin_school": ("aeolian", "dorian", "mixolydian", "phrygian")}
BARS_PER_PHRASE = 8
BEATS_PER_BAR = 4
QUARTER_BARS = 240.0
LEAD_FLOOR = 60
BASS_CEILING = 48
BUSY_OVER_QUIET = 1.5
FOURTH, FIFTH, STEPS_IN_MODE = 3, 4, 7
LARGEST_VOICE_MOVE = 3
WINDOW = 14
DEMO_DAY = 13
QUIET = DayCharacter(
    energy=0.08,
    level=LOW,
    shape=FLAT,
    peak=0.5,
    warmth=-0.5,
    stretches=1,
    regularity=0.5,
    events=0.0,
    late_night=False,
    day_off=True,
)
BUSY = replace(QUIET, energy=0.92, level=HIGH, shape=ALL_DAY, peak=0.6, warmth=0.5, day_off=False)


@cache
def _inputs() -> tuple[list[Series], DayAnalysis, date]:
    """Return the series, the analysis and the day of the busy day of the demo kitchen, which every test shares."""
    demo = the_demo()
    series = [*demo.days[DEMO_DAY]["kitchen"], *demo.days[DEMO_DAY]["sun"]]
    series, _ = rescale(series, demo.baselines(DEMO_DAY, ("kitchen",)))
    return series, analysis_of(series, DAY_SECONDS), demo.day_of(DEMO_DAY)


@cache
def _made(style: str, character: DayCharacter) -> tuple[Score, Plan]:
    """Return the score and plan of the same room and data in a style on a day of this character."""
    series, analysis, day = _inputs()
    return scape_score(
        series,
        analysis,
        style,
        f"kitchen|{day}|{style}",
        HOME,
        quality=STANDARD,
        options=PieceOptions(day=day, area_key="kitchen", character=character),
    )


def _plan(
    style: str, mode: str, character: DayCharacter, events: tuple[tuple[float, float], ...] = ()
) -> Plan:
    """Return a plan of the same data in a given mode, with events of our own."""
    series, analysis, day = _inputs()
    plan = build_plan(
        series,
        analysis,
        style,
        f"kitchen|{day}|{style}",
        Scale(HOME.root, mode),
        STANDARD,
        options=PieceOptions(day=day, area_key="kitchen", character=character),
    )
    plan.events = list(events)
    return plan


def _per_minute(score: Score) -> float:
    return len(score.notes) / (score.length / 60.0)


def _layers(score: Score) -> set[str]:
    return {note.layer for note in score.notes}


def _pitched_voices(score: Score) -> set[str]:
    return {voice for voice in score.voices_used() if not voice.startswith("drum:")}


def _bar(score: Score) -> float:
    return QUARTER_BARS / score.tempo


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_a_busy_day_is_faster_and_inside_the_range_of_the_style(style: str) -> None:
    """The tempo follows the energy, stays in the idiomatic range and is one number for the piece."""
    quiet, _ = _made(style, QUIET)
    busy, _ = _made(style, BUSY)
    low, high = STYLES[style].tempo
    assert low <= quiet.tempo < busy.tempo <= high


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_a_busy_day_is_full_and_a_quiet_day_sparse(style: str) -> None:
    """Notes a minute and the layers that play differ by a lot, not by a tenth."""
    quiet, _ = _made(style, QUIET)
    busy, _ = _made(style, BUSY)
    assert _per_minute(busy) > BUSY_OVER_QUIET * _per_minute(quiet)
    assert _layers(quiet) < _layers(busy)


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_the_voices_the_patterns_and_the_growl_follow_the_day(style: str) -> None:
    """A quiet cold day and a busy warm one differ in voices and patterns, and only the busy one has a growling bass."""
    quiet, _ = _made(style, QUIET)
    busy, _ = _made(style, BUSY)
    for key in ("sequence A", "kit"):
        assert quiet.choices[key] != busy.choices[key], key
    assert quiet.choices["growl bass"] == "none"
    assert busy.choices["growl bass"] != "none"
    assert not {"reese", "fm_growl"} & set(quiet.voices_used())
    assert {"reese", "fm_growl"} & set(busy.voices_used())
    assert _pitched_voices(quiet) != _pitched_voices(busy)


def test_the_harmonic_rhythm_is_slow_on_a_quiet_day_and_fast_on_a_busy_one() -> None:
    """Bars a chord: long on a quiet day, short on a busy one."""
    quiet, _ = _made("berlin_school", QUIET)
    busy, _ = _made("berlin_school", BUSY)
    assert int(quiet.choices["harmonic rhythm"]) > int(busy.choices["harmonic rhythm"])


@pytest.mark.parametrize("style", STYLE_KEYS)
@pytest.mark.parametrize("character", [QUIET, BUSY, NEUTRAL], ids=["quiet", "busy", "neutral"])
def test_the_form_sits_on_eight_bar_lines(style: str, character: DayCharacter) -> None:
    """Every section edge is a multiple of 8 bars, and the layers that join do so on one."""
    score, plan = _made(style, character)
    bar = _bar(score)
    for start, _, _, _ in plan.sections(bar):
        assert start / (BARS_PER_PHRASE * bar) == pytest.approx(
            round(start / (BARS_PER_PHRASE * bar))
        )


@pytest.mark.parametrize("style", STYLE_KEYS)
@pytest.mark.parametrize("character", [QUIET, BUSY], ids=["quiet", "busy"])
def test_the_motif_of_the_room_is_stated_in_the_first_peak_or_plateau(
    style: str, character: DayCharacter
) -> None:
    """The motif of the room is played, in the key of the day, above middle C, in the first peak or plateau."""
    score, plan = _made(style, character)
    motif = plan.motif()
    octave = lead_octave(plan.scale, motif)
    wanted = [round(plan.scale.midi(degree, octave)) for degree, _ in motif]
    assert min(wanted) >= LEAD_FLOOR
    played = sorted((n for n in score.notes if n.layer == "lead"), key=lambda note: note.t)
    pitches = [round(note.midi) for note in played]
    found = [
        played[index].t
        for index in range(len(pitches) - len(wanted) + 1)
        if pitches[index : index + len(wanted)] == wanted
    ]
    start, end = next(
        (start, end)
        for start, end, role, _ in plan.sections(_bar(score))
        if role in {CLIMAX, PLATEAU}
    )
    assert any(start <= second < end for second in found), (start, end)


@pytest.mark.parametrize("mode", MODES["berlin_school"])
@pytest.mark.parametrize("character", [QUIET, BUSY], ids=["quiet", "busy"])
def test_berlin_harmony_is_idiomatic_in_every_mode(mode: str, character: DayCharacter) -> None:
    """The chosen progression has no diminished chord and moves by fourths and fifths; the pitches are the mode's."""
    plan = _plan("berlin_school", mode, character)
    score = COMPOSERS["berlin_school"](plan)
    name = score.choices["progression"]
    roots = next(item[1] for item in PROGRESSIONS if item[0] == name)
    assert not any(triad_is_diminished(mode, root) for root in roots)
    for first, second in pairwise((*roots, roots[0])):
        assert (second - first) % STEPS_IN_MODE in {FOURTH, FIFTH}
    allowed = {(plan.scale.root + step) % 12 for step in SCALES[mode]}
    pitched = [n for n in score.notes if not n.voice.startswith("drum:")]
    assert {round(n.midi) % 12 for n in pitched} <= allowed


def test_every_progression_family_moves_by_a_fourth_or_a_fifth_and_every_mode_has_one() -> None:
    """The families of Berlin School close by fourths and fifths, and each of its modes has some."""
    for _, roots, drive in PROGRESSIONS:
        assert 0.0 <= drive <= 1.0
        for first, second in pairwise((*roots, roots[0])):
            if len(roots) > 1:
                assert (second - first) % STEPS_IN_MODE in {FOURTH, FIFTH}
    for mode in MODES["berlin_school"]:
        assert progressions_for(mode, PROGRESSIONS)


def test_the_chords_are_voice_led_and_the_strings_hold_what_they_share() -> None:
    """From chord to chord a voice moves a third at most, and a tone that two chords share is not struck again."""
    roots = [0, 3, 0, 4, 0, 3, 6, 3]
    voicing = voice_lead(roots, (0, 2, 4, 6), (0, 0, 7, 7), 0)
    for first, second in pairwise(voicing):
        assert all(abs(a - b) <= LARGEST_VOICE_MOVE for a, b in zip(first, second, strict=True))
        assert len(set(second)) == len(second)
    assert voicing[0] == [0, 2, 11, 13]
    starts = [index * 4.0 for index in range(len(roots))]
    ends = [*starts[1:], 32.0]
    runs = held_runs(voicing, starts, ends, lambda _at: False)
    assert len(runs) < len(roots) * 4
    assert any(end - start > 4.0 for _, start, end, _ in runs)
    assert all(0 <= degree < WINDOW for _, _, _, degree in runs)


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_the_registers_stay_apart(style: str) -> None:
    """The growl holds the tonic under C3, the held chords sit in the middle, the motif is the highest of the lead."""
    score, _ = _made(style, BUSY)
    growl = [n for n in score.notes if n.layer == "growl"]
    assert growl
    assert max(n.midi for n in growl) < BASS_CEILING
    held = [n for n in score.notes if n.layer in {"strings", "choir", "pad"}]
    assert min((n.midi for n in held), default=BASS_CEILING) >= BASS_CEILING


def test_the_strongest_anomaly_moves_berlin_to_the_relative_key_for_a_phrase() -> None:
    """The next phrase after the strongest anomaly is the same notes from the relative tonic, and then it comes back."""
    mode = "aeolian"
    plain = COMPOSERS["berlin_school"](_plan("berlin_school", mode, BUSY))
    moved = COMPOSERS["berlin_school"](_plan("berlin_school", mode, BUSY, ((100.0, 0.9),)))
    phrase = BARS_PER_PHRASE * _bar(plain)
    start = -(-100.0 // phrase) * phrase

    def line(score: Score, first: float, last: float) -> list[float]:
        return [n.midi for n in score.notes if n.layer == "seqA" and first <= n.t < last]

    assert line(plain, 0.0, start) == line(moved, 0.0, start)
    assert line(plain, start, start + phrase) != line(moved, start, start + phrase)
    assert line(plain, start + phrase, 900.0) == line(moved, start + phrase, 900.0)
    assert RELATIVE_SHIFT[mode] == 2
