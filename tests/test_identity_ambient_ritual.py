"""Tests that the character of a day picks the identity of Ambient and Ritual: a quiet day and a busy one of the same room."""

from __future__ import annotations

import ast
import itertools
from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from entropy_fm import scape_ambient, scape_ritual
from entropy_fm.character import ALL_DAY, FLAT, HIGH, LOW, DayCharacter
from entropy_fm.plan import Plan
from entropy_fm.quality import STANDARD
from entropy_fm.references import rescale
from entropy_fm.scape import analysis_of, scape_score
from entropy_fm.score import Score
from entropy_fm.synth import Scale

from .scape_tools import the_demo

HOME = Scale(2, "dorian")
SEPARATE = 0.5
DAY_SECONDS = 86400.0
FULL_PIECE_BARS = 8
ALIGNED_BARS = 4
LONG_PIECE = 600
HEART_TICKS = 4
WEEK = 7
DRIVING_PULSE = (108.0, 122.0)
QUIET = DayCharacter(
    energy=0.05,
    level=LOW,
    shape=FLAT,
    peak=0.5,
    warmth=-0.6,
    stretches=1,
    regularity=0.5,
    events=0.0,
    late_night=False,
    day_off=True,
)
BUSY = DayCharacter(
    energy=0.95,
    level=HIGH,
    shape=ALL_DAY,
    peak=0.6,
    warmth=0.7,
    stretches=5,
    regularity=0.6,
    events=0.4,
    late_night=False,
    day_off=False,
)


@cache
def _made(
    style: str, character: DayCharacter, room: str = "kitchen", day: int = -1
) -> tuple[Score, Plan]:
    """Return the score and plan of the demo room on its anomaly day (or on another day) with the given character."""
    demo = the_demo()
    index = len(demo.days) - 1 if day < 0 else day
    series = [*demo.days[index][room], *demo.days[index]["sun"]]
    series, _ = rescale(series, demo.baselines(index, (room,)))
    return scape_score(
        series,
        analysis_of(series, DAY_SECONDS),
        style,
        f"{room}|{index}|_scape|{style}",
        HOME,
        day=demo.day_of(index),
        quality=STANDARD,
        area_key=room,
        character=character,
    )


def _per_minute(score: Score, layers: set[str]) -> float:
    return 60.0 * sum(n.layer in layers for n in score.notes) / score.length


def _mean_length(score: Score) -> float:
    return float(np.mean(ast.literal_eval(score.choices["loop lengths"])))


def _contains(sequence: list[float], wanted: list[float]) -> bool:
    return any(
        sequence[i : i + len(wanted)] == wanted for i in range(len(sequence) - len(wanted) + 1)
    )


def _on_phrase_lines(plan: Plan, bar: float) -> None:
    sections = plan.sections(bar)
    grid = FULL_PIECE_BARS if plan.length / (FULL_PIECE_BARS * bar) >= 10 else ALIGNED_BARS
    for start, _, _, _ in sections[1:]:
        assert start / (grid * bar) == pytest.approx(round(start / (grid * bar)))


def test_ambient_loops_are_long_and_few_on_a_quiet_day_and_short_and_many_on_a_busy_one() -> None:
    """The set of lengths, the number of loops, the piano and the notes per minute follow the energy."""
    quiet, quiet_plan = _made("ambient", QUIET)
    busy, busy_plan = _made("ambient", BUSY)
    assert _mean_length(quiet) > 1.5 * _mean_length(busy)
    assert _per_minute(busy, {"pads"}) > 2.0 * _per_minute(quiet, {"pads"})
    assert _per_minute(busy, {"piano"}) > 1.5 * _per_minute(quiet, {"piano"})
    assert _per_minute(busy, {"pads", "piano", "strings"}) > 2.0 * _per_minute(
        quiet, {"pads", "piano", "strings"}
    )
    assert int(quiet.notes_about[0].split()[0]) == scape_ambient.MIN_LOOPS
    assert int(busy.notes_about[0].split()[0]) >= scape_ambient.MAX_LOOPS - 1
    assert quiet_plan.scale != busy_plan.scale


def test_ambient_voices_and_harmony_follow_the_warmth_and_the_energy() -> None:
    """A warm busy day takes the rounder voices, the fuller pitch set and the wider centres; a cold quiet one the others."""
    quiet, _ = _made("ambient", QUIET)
    busy, _ = _made("ambient", BUSY)
    warmth = scape_ambient.LOOP_WARMTH
    assert warmth[busy.choices["loop voice A"]] > warmth[quiet.choices["loop voice A"]]
    assert (
        scape_ambient.PIANO_WARMTH[busy.choices["piano"]]
        > (scape_ambient.PIANO_WARMTH[quiet.choices["piano"]])
    )
    assert (
        scape_ambient.SET_COLOUR[busy.choices["pitch set"]]
        > (scape_ambient.SET_COLOUR[quiet.choices["pitch set"]])
    )
    travel = scape_ambient.CENTRE_TRAVEL
    assert travel[busy.choices["harmonic centres"]] > travel[quiet.choices["harmonic centres"]]
    assert not [n for n in quiet.notes if n.layer == "growl"]


def test_ambient_states_the_motif_of_the_room_on_the_piano_in_the_key_of_the_day() -> None:
    """The motif is on the piano as scale degrees of the day's scale, above C4, in the first peak or plateau section."""
    for character in (QUIET, BUSY):
        score, plan = _made("ambient", character)
        piano = [n.midi for n in sorted(score.notes, key=lambda n: n.t) if n.layer == "piano"]
        expected = [float(plan.scale.midi(d, 4)) for d, _ in plan.motif()]
        while min(expected) < scape_ambient.MOTIF_FLOOR:
            expected = [m + 12 for m in expected]
        assert _contains(piano, expected), character.label()
        assert min(expected) >= 60.0


def test_ambient_sections_sit_on_phrase_lines_and_a_loop_enters_on_one() -> None:
    """The sections change on multiples of 8 bars, and the loops that wait for a section start at its edge."""
    for character in (QUIET, BUSY):
        _, plan = _made("ambient", character)
        _on_phrase_lines(plan, scape_ambient.SECTION_BAR)
    score, plan = _made("ambient", BUSY)
    starts = [s for s, _, _, _ in plan.sections(scape_ambient.SECTION_BAR)]
    firsts = sorted(n.t for n in score.notes if n.layer == "pads")
    assert firsts
    pads_after_edges = [t for t in firsts if any(0.0 <= t - s <= 20.0 for s in starts[1:])]
    assert pads_after_edges


def test_ritual_picks_the_tuning_pattern_and_pulse_from_the_day() -> None:
    """A quiet cold day is darker, slower, sparser and without a heartbeat; a warm busy one is brighter, quicker and drums."""
    quiet, quiet_plan = _made("ritual", QUIET)
    busy, busy_plan = _made("ritual", BUSY)
    brightness = {name: value for name, _, value in scape_ritual.TUNINGS}
    assert brightness[busy.choices["tuning"]] > brightness[quiet.choices["tuning"]]
    seconds = dict(scape_ritual.LENGTHS)
    assert seconds[quiet.choices["pluck cell length"]] > seconds[busy.choices["pluck cell length"]]
    energy = scape_ritual.SLIT_ENERGY
    assert energy[busy.choices["slit drum pattern"]] > energy[quiet.choices["slit drum pattern"]]
    warmth = scape_ritual.PLUCK_WARMTH
    assert warmth[busy.choices["pluck voice"]] > warmth[quiet.choices["pluck voice"]]
    lead = scape_ritual.LEAD_ENERGY
    assert lead[busy.choices["chant leader"]] > lead[quiet.choices["chant leader"]]
    assert _per_minute(busy, {"plucks", "slit", "block"}) > 1.5 * _per_minute(
        quiet, {"plucks", "slit", "block"}
    )
    assert not [n for n in quiet.notes if n.layer in {"heart", "taiko"}]
    assert len([n for n in busy.notes if n.layer in {"heart", "taiko"}]) > HEART_TICKS
    assert busy_plan.scale != quiet_plan.scale


def test_ritual_stays_pentatonic_on_the_root_of_the_day_and_the_pulse_stays_in_range() -> None:
    """Every pitched note is in the day's tuning on the root of plan.scale; the heartbeat is between 40 and 58 a minute."""
    for character in (QUIET, BUSY):
        score, plan = _made("ritual", character)
        _, steps, _ = next(t for t in scape_ritual.TUNINGS if t[0] == score.choices["tuning"])
        root = plan.scale.root
        assert int(score.choices["root"]) == root
        allowed = {(root + step) % 12 for step in steps}
        assert allowed
        pitched = [n for n in score.notes if n.layer in {"plucks", "slit", "block", "lead"}]
        assert all(round(n.midi) % 12 in allowed for n in pitched)
        drone = [n for n in score.notes if n.layer in {"drone", "sub"}]
        assert {(round(n.midi) - root) % 12 for n in drone} <= {0, 5, 7}
        assert 4.0 <= dict(scape_ritual.LENGTHS)[score.choices["pluck cell length"]] <= 7.2
    busy = _made("ritual", BUSY)[0]
    heart = sorted(n.t for n in busy.notes if n.layer == "heart")
    if busy.choices["ritual mode"] == scape_ritual.DRIVING:
        assert not heart
        pulse = 60.0 / (dict(scape_ritual.LENGTHS)[busy.choices["pluck cell length"]] / 8.0)
        assert DRIVING_PULSE[0] <= pulse <= DRIVING_PULSE[1]
        return
    beats = [t for before, t in itertools.pairwise([-10.0, *heart]) if t - before > 0.6]
    pulse = 60.0 / float(np.median(np.diff(beats)))
    low, high = scape_ritual.BEATS_PER_MINUTE
    assert low - 1.0 <= pulse <= high + 1.0


def test_ritual_states_the_motif_on_the_pluck_and_the_sections_sit_on_phrase_lines() -> None:
    """The motif is plucked in the first peak or plateau section, above C4, on the ladder of the tuning."""
    for character in (QUIET, BUSY):
        score, plan = _made("ritual", character)
        _, steps, _ = next(t for t in scape_ritual.TUNINGS if t[0] == score.choices["tuning"])
        base = 55 + (plan.scale.root - 55) % 12
        ladder = [round(d * 5 / plan.scale.size) for d, _ in plan.motif()]

        def pitch(step: int, root: int, steps: tuple[int, ...] = steps) -> float:
            octave, degree = divmod(step, 5)
            return float(root + 12 * octave + steps[degree])

        while min(pitch(s, base) for s in ladder) < scape_ritual.MOTIF_FLOOR:
            base += 12
        expected = [pitch(s, base) for s in ladder]
        plucked = [n.midi for n in sorted(score.notes, key=lambda n: n.t) if n.layer == "plucks"]
        assert _contains(plucked, expected), character.label()
        cell = dict(scape_ritual.LENGTHS)[score.choices["pluck cell length"]]
        _on_phrase_lines(plan, cell)


def test_ritual_drums_and_wood_enter_with_the_form() -> None:
    """On the busy day nothing of the drums plays in a section that is too weak for them."""
    score, plan = _made("ritual", BUSY)
    cell = dict(scape_ritual.LENGTHS)[score.choices["pluck cell length"]]
    sections = plan.sections(cell)
    for start, end, _, intensity in sections:
        inside = [n for n in score.notes if start <= n.t < end and n.layer in {"heart", "taiko"}]
        if intensity < scape_ritual.DRUMS_FROM:
            assert not [n for n in inside if n.layer == "heart"], (start, intensity)


def test_ritual_modes_lean_on_the_energy_but_similar_days_rotate() -> None:
    """A quiet week never drives and a busy one never stands still; across the week every mode comes up."""
    modes = {}
    for energy in (0.05, 0.5, 0.95):
        character = replace(QUIET, energy=energy)
        modes[energy] = {
            _made("ritual", character, day=day)[0].choices["ritual mode"] for day in range(WEEK)
        }
    assert scape_ritual.DRIVING not in modes[0.05]
    assert scape_ritual.STILL not in modes[0.95]
    assert modes[0.05] | modes[0.5] | modes[0.95] == {name for name, _ in scape_ritual.MODES}
    assert len(modes[0.5]) >= 2
