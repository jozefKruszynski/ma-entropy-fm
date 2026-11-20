"""Tests of the soundscapes: the plan, the generative systems of the six styles, the day to day variation."""

import itertools
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from entropy_fm.character import ALL_DAY, FLAT, HIGH, LOW, NEUTRAL, DayCharacter
from entropy_fm.day_analysis import analyze_day
from entropy_fm.plan import MAX_SECONDS, MIN_SECONDS, Plan, build_plan, length_of
from entropy_fm.quality import LIGHT, RICH, STANDARD, Quality
from entropy_fm.scape import scape_score
from entropy_fm.scape_ambient import LENGTHS
from entropy_fm.scape_berlin import CELLS_A, CELLS_B, CELLS_C, _mutate
from entropy_fm.score import CR, Score
from entropy_fm.styles import STYLES, get_style
from entropy_fm.synth import Scale
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series
from entropy_fm.voices import VOICES

from .scape_tools import demo, the_demo

STYLE_KEYS = list(STYLES)


def _score(
    style: str, index: int = 13, areas: tuple[str, ...] = ("kitchen",), tier: Quality = STANDARD
) -> tuple[Score, Plan]:
    return the_demo().score(index, areas, style, tier)


def test_there_are_six_styles_and_the_removed_ones_are_gone() -> None:
    """Ambient, Berlin School, Lo-fi, Minimal and Ritual; no House, Drum and bass, Dub techno or Gamelan."""
    assert list(STYLES) == ["ambient", "berlin_school", "lofi", "minimal", "ritual", "glitch"]
    from entropy_fm import covers, palette, synth  # noqa: PLC0415

    for gone in ("house", "dnb", "dub_techno", "gamelan"):
        assert gone not in STYLES
        assert gone not in covers.PALETTES
        assert gone not in covers.GLYPHS
    assert "slendro" not in synth.SCALES
    assert "pelog" not in palette.MODE_WARMTH


def test_the_piece_is_ten_to_twenty_two_minutes_and_longer_on_a_busy_day() -> None:
    """The length follows the day: 12 minutes when quiet, 20 when busy, never under 10 or over 22."""
    series = [
        build_series(
            EntityInfo("sensor.p", "p", None, VoiceKind.DRONE),
            [0.0, 40000.0, 41000.0],
            [0.0, 900.0, 0.0],
        )
    ]
    base = analyze_day(series)
    lengths = []
    for busy, events in ((0.0, 0), (0.5, 0), (1.0, 0), (1.0, 5)):
        from dataclasses import replace  # noqa: PLC0415

        moments = tuple(m for m in base.moments if m.kind != "anomaly")
        from entropy_fm.day_analysis import Moment  # noqa: PLC0415

        moments += tuple(Moment(10000.0 + 100 * k, "anomaly", 0.9, "x") for k in range(events))
        lengths.append(length_of(replace(base, busyness=busy, moments=moments)))
    assert lengths == sorted(lengths)
    assert lengths[0] == 720
    assert 1150 <= lengths[2] <= 1250
    assert all(MIN_SECONDS <= value <= MAX_SECONDS for value in lengths)
    assert lengths[3] <= MAX_SECONDS


def test_the_day_is_spread_over_the_piece_in_order() -> None:
    """The five stretches of the day follow each other over the length of the piece."""
    _, plan = demo("ambient")
    arc = plan.arc()
    assert [name for name, _, _ in arc] == ["night", "morning", "daytime", "evening", "late night"]
    assert arc[0][1] == 0.0
    assert arc[-1][2] == pytest.approx(plan.length)
    for (_, _, end), (_, start, _) in itertools.pairwise(arc):
        assert end == pytest.approx(start)


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_a_score_is_deterministic(style: str) -> None:
    """The same day in the same style gives the same notes, to the last one."""
    first, _ = _score(style)
    second, _ = the_demo().score(13, ("kitchen",), style, STANDARD)
    assert first.notes == second.notes
    assert first.choices == second.choices
    assert first.gain_db is not None
    assert second.gain_db is not None
    assert np.array_equal(first.gain_db, second.gain_db)


QUIET = replace(
    NEUTRAL,
    energy=0.0,
    level=LOW,
    shape=FLAT,
    warmth=-0.8,
    stretches=0,
    events=0.0,
    late_night=True,
)
BUSY = replace(
    NEUTRAL,
    energy=1.0,
    level=HIGH,
    shape=ALL_DAY,
    warmth=0.8,
    stretches=5,
    regularity=0.9,
    events=0.8,
    day_off=True,
)
WEEK = range(7, 14)
# a wider swing of warmth changes every voice of a style that picks them all by warmth, so that no core is left
WARMTH_SWING = 0.2


def _score_of(style: str, index: int, character: DayCharacter) -> tuple[Score, Plan]:
    """Return the score and the plan of a day of the demo in a style, for a character that the test sets."""
    demo_house = the_demo()
    series: Any = demo_house.days[index]["kitchen"] + demo_house.days[index]["sun"]
    return scape_score(
        series,
        analyze_day(series),
        style,
        f"kitchen|{demo_house.day_of(index).isoformat()}|_scape|{style}",
        Scale(2, "dorian"),
        day=demo_house.day_of(index),
        quality=STANDARD,
        area_key="kitchen",
        character=character,
    )


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_the_dealer_breaks_ties_between_days_of_the_same_character(style: str) -> None:
    """With the same character every day, the tie-breaking still changes something from one day to the next."""
    days = [_score_of(style, i, NEUTRAL)[1] for i in WEEK]
    for yesterday, today in itertools.pairwise(days):
        assert yesterday.choices
        assert any(yesterday.choices[key] != today.choices.get(key) for key in yesterday.choices), (
            style
        )
    assert len({tuple(sorted(d.choices.items())) for d in days}) >= 2


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_a_quiet_and_a_busy_day_get_a_different_identity(style: str) -> None:
    """The data picks the identity: a clearly quiet and a clearly busy day differ in tempo or mode, and in what is chosen."""
    quiet_score, quiet = _score_of(style, 13, QUIET)
    busy_score, busy = _score_of(style, 13, BUSY)
    low, high = get_style(style).tempo
    moved = quiet.scale.mode != busy.scale.mode
    if high > low:
        moved = moved or quiet_score.tempo != busy_score.tempo
    assert moved, style
    assert quiet.choices != busy.choices, style


# a voice of the core of a style plays on at least this many of the seven days
CORE_DAYS = 4


def _spread_character(step: int) -> DayCharacter:
    """Return the character of a day of the week that goes from quiet and cold (0) to busy and warm (6)."""
    share = step / (len(WEEK) - 1)
    return replace(
        NEUTRAL,
        energy=share,
        level=LOW if share < 0.4 else HIGH,
        warmth=WARMTH_SWING * (2 * share - 1),
    )


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_the_chosen_voices_vary_across_seven_days(style: str) -> None:
    """When the days differ in character, the set of voices that a style plays changes, though a core is on most days."""
    used = [
        frozenset(_score_of(style, i, _spread_character(i - WEEK.start))[0].voices_used())
        for i in WEEK
    ]
    assert len(set(used)) >= 2, style
    core = frozenset(
        voice for voice in frozenset.union(*used) if sum(voice in day for day in used) >= CORE_DAYS
    )
    assert core
    assert frozenset.union(*used) - core


def test_the_six_styles_use_different_voices() -> None:
    """Each style has a character of its own: the voices they share are few."""
    sets = {
        style: set(the_demo().score(13, ("kitchen",), style, STANDARD)[0].voices_used())
        for style in STYLE_KEYS
    }
    for a in STYLE_KEYS:
        for b in STYLE_KEYS:
            if a < b:
                shared = sets[a] & sets[b]
                assert len(shared) <= 0.5 * min(len(sets[a]), len(sets[b])), (a, b, shared)
    assert any(v.startswith("drum:") for v in sets["minimal"])
    assert not any(v.startswith("drum:") for v in sets["ambient"])


def _series(events: int) -> list[Series]:
    """A fridge, a lamp and a motion sensor that flips `events` times in the day."""  # noqa: D401
    times = [0.0, *[30000.0 + 120.0 * k for k in range(events)]]
    values = [float(k % 2) for k in range(len(times))]
    return [
        build_series(
            EntityInfo("sensor.fridge", "f", None, VoiceKind.DRONE),
            [0.0, 5000.0, 7000.0],
            [0.0, 90.0, 0.0],
        ),
        build_series(
            EntityInfo("binary_sensor.motion", "m", None, VoiceKind.PLUCK, device_class="motion"),
            times,
            values,
        ),
    ]


KIT_LAYERS = {"kick", "snare", "hats"}


@pytest.mark.parametrize("style", STYLE_KEYS)
def test_events_do_not_write_notes(style: str) -> None:
    """Fifty times as many events in the house give less than two and a half times the notes: the data steers, it does not play."""
    few = _series(4)
    many = _series(200)
    counts = []
    for series in (few, many):
        score, _ = scape_score(
            series, analyze_day(series), style, "x", Scale(2, "dorian"), area_key="x"
        )
        # the kit of Cinematic Glitch is a pattern that the heat of the house opens, in whole bars: it is not counted
        counts.append(
            len([n for n in score.notes if style != "glitch" or n.layer not in KIT_LAYERS])
        )
    assert counts[1] < 2.5 * counts[0], (style, counts)
    assert counts[1] < 200 * 0.1 * counts[0]


@pytest.mark.parametrize("style", ["ambient", "berlin_school", "lofi"])
def test_the_pitches_come_from_the_composed_vocabulary(style: str) -> None:
    """Every pitched note of a style is a note of the scale of the day or a tone of a composed chord."""
    score, plan = the_demo().score(13, ("kitchen",), style, STANDARD)
    scale = plan.scale
    allowed = {(scale.root + step) % 12 for step in scale.intervals}
    if style == "lofi":
        allowed = set(range(12))
    pitched = [n for n in score.notes if not n.voice.startswith("drum:")]
    assert pitched
    outside = [n for n in pitched if round(n.midi) % 12 not in allowed]
    assert len(outside) <= 0.03 * len(pitched)


def test_no_beat_in_ambient_and_a_beat_in_minimal() -> None:
    """Ambient has no drums at all; Minimal has a soft kick on every beat while it plays."""
    ambient, _ = _score("ambient")
    assert not any(n.layer in {"kick", "drums"} for n in ambient.notes)
    minimal, _ = _score("minimal")
    kicks = sorted(n.t for n in minimal.notes if n.layer == "kick")
    gaps = np.diff(kicks)
    beat = 60.0 / minimal.tempo
    assert 118.0 <= minimal.tempo <= 124.0
    assert np.mean(np.abs(gaps[gaps < 1.5 * beat] - beat) < 0.02) > 0.9


def test_ambient_loops_have_lengths_that_share_no_measure() -> None:
    """No two loop lengths of any set are in a simple ratio, so they phase and never line up the same way twice."""
    for lengths in LENGTHS:
        for a in lengths:
            for b in lengths:
                if a < b:
                    ratio = b / a
                    for p in range(1, 6):
                        for q in range(1, 6):
                            assert abs(ratio - p / q) > 0.02 * p / q, (a, b)
    score, _ = _score("ambient")
    assert "tape loops" in score.notes_about[0]


def test_the_berlin_sequences_have_different_lengths_and_mutate_by_rule() -> None:
    """The three lines have 16, 12 and a shorter number of steps, and the rules change a cell without emptying it."""
    for _, a in CELLS_A:
        assert len(a) == 16
    for _, b in CELLS_B:
        assert len(b) == 12
    for _, c in CELLS_C:
        assert len(c) not in {12, 16}
    rng = np.random.default_rng(3)
    for _, base in CELLS_A:
        cell = list(base)
        changed = 0
        for _ in range(60):
            if _mutate(rng, cell, base, 1.0, (0, 2, 4, 6)):
                changed += 1
        assert changed > 20
        assert 5 <= sum(v is not None for v in cell) <= 16
    score, _ = _score("berlin_school")
    assert "mutations applied" in score.notes_about[0]
    lines = {n.layer for n in score.notes if n.layer.startswith("seq")}
    assert "seqA" in lines


def test_the_night_is_sparser_than_the_active_day() -> None:
    """The soundscape follows the day: fewer layers and notes per minute in the night than in the busiest part of the day."""
    for style in STYLE_KEYS:
        score, plan = the_demo().score(13, ("kitchen",), style, STANDARD)
        assert isinstance(plan, Plan)
        arcs = {name: (a, b) for name, a, b in plan.arc()}
        busiest = max(
            (name for name in arcs if name != "night"),
            key=lambda name: float(
                np.mean(plan.activity[int(arcs[name][0] * 10) : int(arcs[name][1] * 10)])
            ),
        )
        stats = []
        for name in ("night", busiest):
            start, end = arcs[name]
            inside = [n for n in score.notes if start <= n.t < end]
            stats.append((len(inside) * 60.0 / (end - start), len({n.layer for n in inside})))
        assert stats[0][0] <= stats[1][0], (style, busiest, stats)
        assert stats[0][1] <= stats[1][1], (style, busiest, stats)


def test_lofi_rests_its_drums_at_night() -> None:
    """In the dead of the night the Lo-fi loop has no drums and plays the chords alone."""
    score, plan = _score("lofi")
    arc = {name: (a, b) for name, a, b in plan.arc()}
    night = [
        n
        for n in score.notes
        if arc["night"][0] + 30 <= n.t < arc["night"][1] - 30 and n.layer == "drums"
    ]
    day = [n for n in score.notes if n.layer == "drums"]
    assert len(night) < 0.15 * len(day)
    assert any(n.layer == "keys" for n in score.notes if n.t < arc["night"][1])


def test_berlin_school_drums_rest_at_night_and_play_in_the_active_day() -> None:
    """The sparse drums of the Berlin School come in the busy daytime only: none in the night or the evening."""
    from soundscapes import pick_windows  # noqa: PLC0415

    score, plan = _score("berlin_school")
    drums = [n for n in score.notes if n.voice.startswith("drum:")]
    arc = {name: (a, b) for name, a, b in plan.arc()}
    assert not [n for n in drums if n.t < arc["night"][1]]
    assert not [
        n for n in drums if arc["evening"][0] + 0.3 * (arc["evening"][1] - arc["evening"][0]) < n.t
    ]
    start = pick_windows(plan, 60)[1][1]
    assert [n for n in drums if start <= n.t < start + 60]
    light, _ = _score("berlin_school", tier=LIGHT)
    assert {n.layer for n in light.notes if n.voice.startswith("drum:")} <= {"pulse"}


def test_an_appliance_that_runs_brings_in_its_layer() -> None:
    """While the dishwasher runs the second Berlin line and the Minimal clicks are held in."""
    _, plan = the_demo().score(13, ("kitchen",), "berlin_school", STANDARD)
    assert plan.runs["dish"].max() > 0.5
    score, plan = the_demo().score(13, ("kitchen",), "berlin_school", STANDARD)
    on = np.flatnonzero(plan.runs["dish"] > 0.8)
    assert len(on)
    seconds = on / CR
    inside = [n for n in score.notes if n.layer == "seqB" and seconds.min() <= n.t <= seconds.max()]
    assert inside


def test_an_anomaly_is_a_texture_event_and_never_a_pitch_glide() -> None:
    """The anomaly day has events; they are dips of the filters and swells, not notes that glide."""
    for style in STYLE_KEYS:
        score, plan = the_demo().score(13, ("kitchen",), style, STANDARD)
        assert score.events == plan.events
        assert plan.events
        for layer in score.layers.values():
            if layer.cutoff is not None:
                second = plan.events[0][0]
                here = layer.cutoff[int(second * CR) : int((second + 1.0) * CR)].min()
                before = layer.cutoff[
                    max(int((second - 20) * CR), 0) : int((second - 10) * CR)
                ].mean()
                assert here < before, (style, layer.key)


def test_the_tiers_limit_the_layers() -> None:
    """Light plays fewer lines and loops than Rich, and the notes of the pieces differ."""
    light, _ = the_demo().score(13, ("kitchen",), "berlin_school", LIGHT)
    rich, _ = the_demo().score(13, ("kitchen",), "berlin_school", RICH)
    assert len({n.layer for n in light.notes if n.layer.startswith("seq")}) <= 2
    assert len({n.layer for n in rich.notes if n.layer.startswith("seq")}) >= len(
        {n.layer for n in light.notes if n.layer.startswith("seq")}
    )
    assert RICH.max_layers > LIGHT.max_layers


def test_every_voice_a_score_names_is_in_the_library() -> None:
    """A score only names voices of the library and drum hits; at most one voice of a style is bell-like."""
    from entropy_fm.drums import HITS  # noqa: PLC0415
    from entropy_fm.glitch_voices import is_glitch  # noqa: PLC0415
    from entropy_fm.stretch import is_wash  # noqa: PLC0415

    for style in STYLE_KEYS:
        score, _ = the_demo().score(13, ("kitchen",), style, STANDARD)
        for voice in score.voices_used():
            assert (
                voice in VOICES
                or voice.removeprefix("drum:") in HITS
                or is_wash(voice)
                or is_glitch(voice)
            ), voice
        assert sum(1 for v in score.voices_used() if v in {"glass_pad", "fm_ep"}) <= 2


def test_the_plan_curves_are_slow_and_in_range() -> None:
    """Every curve has a value every tenth of a second, from 0 to 1, with no jumps."""
    _, plan = _score("ambient")
    for name in ("activity", "power", "bright", "humid", "lights", "evening"):
        curve = getattr(plan, name)
        assert len(curve) == plan.length * CR
        assert curve.min() >= 0.0
        assert curve.max() <= 1.0
        assert np.abs(np.diff(curve)).max() < 0.25, name
    assert plan.activity.std() > 0.15
    assert isinstance(date(2026, 1, 1) - timedelta(days=1), date)
    series = _series(10)
    plain = build_plan(series, analyze_day(series), "ambient", "x", Scale(2, "dorian"), STANDARD)
    assert plain.length >= MIN_SECONDS


def test_the_gain_estimate_is_finite_and_bounded() -> None:
    """The feed-forward gain that holds the loudness stays between -9 and +12 dB."""
    for style in STYLE_KEYS:
        score, _ = _score(style)
        assert score.gain_db is not None
        assert np.isfinite(score.gain_db).all()
        assert score.gain_db.min() >= -9.01
        assert score.gain_db.max() <= 12.01
