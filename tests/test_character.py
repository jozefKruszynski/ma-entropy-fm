"""Tests of the character of a day: what it measures, how it picks the key, the form and the motif of a piece."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from entropy_fm.baseline import build_baselines
from entropy_fm.character import (
    ALL_DAY,
    EVENING,
    FLAT,
    HIGH,
    LARGEST_LEAP,
    LOW,
    MEDIUM,
    MORNING,
    NEUTRAL,
    OCTAVE_DEGREES,
    PEAK,
    DayCharacter,
    chord_bars,
    level_of,
    measure_character,
    raw_energy,
    room_motif,
    scale_of_day,
    tempo_target,
)
from entropy_fm.day_analysis import analyze_day
from entropy_fm.palette import DayTraits
from entropy_fm.plan import (
    BUILD,
    CLIMAX,
    INTRO,
    MIN_FORM_CELLS,
    OUTRO,
    PICK_TOLERANCE,
    RELEASE,
    PieceOptions,
    Plan,
    build_plan,
)
from entropy_fm.quality import STANDARD
from entropy_fm.references import rescale
from entropy_fm.scape import analysis_of
from entropy_fm.styles import STYLES
from entropy_fm.synth import SCALES, Scale
from entropy_fm.timeline import EntityInfo, Series, VoiceKind, build_series

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import house_fixture

HOUR = 3600.0
DAY = 86400.0
HOME = Scale(2, "dorian")
TUESDAY = date(2026, 11, 17)
SATURDAY = date(2026, 11, 14)
FIFTHS_IN_AN_OCTAVE = 12
MAX_FIFTHS_BETWEEN_DAYS = 2
BARS = (1.8, 2.4, 4.0)
REAL_HOUSE = house_fixture.FIXTURE_DIR.is_dir() and bool(house_fixture.fixture_days())
needs_fixture = pytest.mark.skipif(not REAL_HOUSE, reason="no real house fixture in .data/fixtures")


def _co2(peak: float, hour: float = 20.0, width: float = 1.5) -> Series:
    """Return the CO2 of a room that rises to `peak` ppm around an hour of the day, for about `width` hours."""
    hours = np.arange(0.0, 24.0, 0.25)
    values = 420.0 + (peak - 420.0) * np.exp(-(((hours - hour) / width) ** 2))
    info = EntityInfo("sensor.co2", "CO2", "room", VoiceKind.AIR, device_class="carbon_dioxide")
    return build_series(info, (hours * HOUR).tolist(), values.tolist())


def _lamp(*periods: tuple[float, float]) -> Series:
    """Return a lamp that is on during the periods (hours of the day)."""
    times, values = [0.0], [0.0]
    for start, end in periods:
        times += [start * HOUR, end * HOUR]
        values += [1.0, 0.0]
    return build_series(EntityInfo("light.lamp", "Lamp", "room", VoiceKind.CHORD), times, values)


def _day(peak: float, *periods: tuple[float, float], hour: float = 20.0) -> list[Series]:
    return [_co2(peak, hour), _lamp(*periods)]


def _character(
    series: list[Series],
    day: date = TUESDAY,
    *,
    week: list[list[Series]] | None = None,
    workday: bool | None = None,
) -> DayCharacter:
    """Measure a day against the week before it, as the provider does."""
    baselines = build_baselines(week or [_day(1600.0, (18.0, 23.0)) for _ in range(5)])
    measured, _ = rescale(series, baselines)
    return measure_character(analyze_day(measured), DayTraits(), day, measured, workday=workday)


# ---- what is measured


def test_a_quiet_day_and_a_busy_day_are_clearly_apart() -> None:
    """The same room on a quiet day and on a busy one lands in different thirds, far apart in energy."""
    quiet = _character(_day(700.0, (19.0, 20.0)))
    busy = _character(_day(3200.0, (6.0, 23.5)))
    assert busy.energy - quiet.energy >= 0.4
    assert quiet.level == LOW
    assert busy.level == HIGH


def test_the_energy_uses_the_whole_range() -> None:
    """A dead day is near zero and a day well over the usual is at the top."""
    dead = _character([_co2(430.0, 3.0), _lamp()])
    top = _character([_co2(3200.0, 14.0, 7.0), _lamp((6.0, 23.0))])
    assert dead.energy < 0.1
    assert top.energy > 0.9


def _week_of(*peaks: float) -> list[float]:
    """Return the raw energies of earlier days that peak at the given CO2 levels, on the scale of `_character`."""
    baselines = build_baselines([_day(1600.0, (18.0, 23.0)) for _ in range(5)])
    return [raw_energy(rescale(_day(peak, (18.0, 23.0)), baselines)[0], DAY) for peak in peaks]


def test_the_raw_energy_follows_how_lively_the_day_was() -> None:
    """A busy day reads higher than a quiet one, and a dead one lowest."""
    dead = raw_energy([_co2(430.0, 3.0), _lamp()], DAY)
    quiet = raw_energy(_day(700.0, (19.0, 20.0)), DAY)
    assert 0.0 <= dead < quiet < raw_energy(_day(2200.0, (7.0, 23.0)), DAY) <= 1.0


def test_the_energy_is_the_place_between_the_quietest_and_busiest_days_before() -> None:
    """The quietest day of the week is near 0 and the busiest near 1, whatever their absolute level."""
    week = _week_of(500.0, 900.0, 1300.0, 1700.0)
    series = _day(1000.0, (18.0, 23.0))
    baselines = build_baselines([_day(1600.0, (18.0, 23.0)) for _ in range(5)])

    def energy(items: list[Series]) -> float:
        measured, _ = rescale(items, baselines)
        return measure_character(
            analyze_day(measured), DayTraits(), TUESDAY, measured, week=week
        ).energy

    assert energy(_day(450.0, (18.0, 23.0))) < 0.2
    assert energy(_day(3000.0, (18.0, 23.0))) > 0.8
    assert energy(_day(450.0, (18.0, 23.0))) < energy(series) < energy(_day(3000.0, (18.0, 23.0)))


def test_the_energy_stays_absolute_with_fewer_than_three_days_before() -> None:
    """Two earlier days are too few to rank against: the energy is what it is without them."""
    series = _day(1000.0, (18.0, 23.0))
    baselines = build_baselines([_day(1600.0, (18.0, 23.0)) for _ in range(5)])
    measured, _ = rescale(series, baselines)
    analysis = analyze_day(measured)
    alone = measure_character(analysis, DayTraits(), TUESDAY, measured)
    short = measure_character(analysis, DayTraits(), TUESDAY, measured, week=_week_of(500.0, 900.0))
    assert short.energy == alone.energy
    ranked = measure_character(
        analysis, DayTraits(), TUESDAY, measured, week=_week_of(500.0, 900.0, 1300.0)
    )
    assert ranked.energy != alone.energy


def test_events_do_not_move_the_energy() -> None:
    """An anomaly is unusual, not busy: it changes the events and leaves the energy alone."""
    series = _day(1200.0, (18.0, 22.0))
    baselines = build_baselines([_day(1600.0, (18.0, 23.0)) for _ in range(5)])
    measured, _ = rescale(series, baselines)
    plain = measure_character(analysis_of(measured, DAY, []), DayTraits(), TUESDAY, measured)
    odd = measure_character(
        analysis_of(measured, DAY, [(3 * HOUR, 1.0, "light.lamp"), (13 * HOUR, 1.0, "sensor.co2")]),
        DayTraits(),
        TUESDAY,
        measured,
    )
    assert odd.events > plain.events
    assert odd.energy == plain.energy


def test_the_levels_are_thirds() -> None:
    """The level follows the thirds of the energy."""
    assert [level_of(0.0), level_of(0.5), level_of(1.0)] == [LOW, MEDIUM, HIGH]


@pytest.mark.parametrize(
    ("hour", "lamps", "shape"),
    [
        (7.5, ((6.0, 9.0),), MORNING),
        (20.0, ((18.0, 23.0),), EVENING),
    ],
)
def test_the_shape_follows_where_the_activity_sits(
    hour: float, lamps: tuple[tuple[float, float], ...], shape: str
) -> None:
    """Activity in the morning is a morning day, in the evening an evening day."""
    assert _character(_day(2000.0, *lamps, hour=hour)).shape == shape


def test_a_day_with_nothing_is_flat_and_the_peak_is_in_the_day() -> None:
    """Nothing happening is a flat day, and the position of the peak is a share of the day."""
    flat = _character([_co2(420.0, 3.0), _lamp()])
    assert flat.shape == FLAT
    busy = _character(_day(2400.0, (18.0, 23.0)))
    assert 0.6 < busy.peak < 0.95


def test_stretches_are_counted_and_their_regularity_is_a_share() -> None:
    """Three lamps spread over the day are three stretches; evenly spaced they are regular."""
    even = _character([_co2(420.0, 3.0), _lamp((7.0, 9.0), (13.0, 15.0), (19.0, 21.0))])
    assert even.stretches == 3
    uneven = _character([_co2(420.0, 3.0), _lamp((7.0, 8.0), (8.5, 9.5), (21.0, 23.0))])
    assert 0.0 <= uneven.regularity < even.regularity <= 1.0


def test_a_day_off_is_a_weekend_or_what_the_workday_sensor_says() -> None:
    """The weekend is a day off, a workday argument wins, and so does a workday sensor in the series."""
    series = _day(1200.0, (18.0, 22.0))
    assert not _character(series, TUESDAY).day_off
    assert _character(series, SATURDAY).day_off
    assert _character(series, TUESDAY, workday=False).day_off
    assert not _character(series, SATURDAY, workday=True).day_off
    sensor = build_series(
        EntityInfo("binary_sensor.workday_sensor", "Workday", None, VoiceKind.BLIP), [0.0], [0.0]
    )
    assert _character([*series, sensor], TUESDAY).day_off


def test_a_late_night_is_real_activity_after_half_past_eleven() -> None:
    """A lamp that comes on at midnight is a late night; one that goes off then is not."""
    late = _character([_co2(1200.0), _lamp((18.0, 22.0), (23.75, 23.9))])
    early = _character([_co2(1200.0), _lamp((18.0, 23.0))])
    assert late.late_night
    assert not early.late_night
    assert late.events > early.events


def test_the_warmth_follows_the_weather_and_the_season() -> None:
    """A sunny day in summer is warm, a snowy day in winter is cold."""
    series = _day(1200.0, (18.0, 22.0))
    baselines = build_baselines([_day(1600.0, (18.0, 23.0)) for _ in range(5)])
    measured, _ = rescale(series, baselines)
    analysis = analyze_day(measured)
    summer = measure_character(analysis, DayTraits(weather="sunny"), date(2026, 7, 15), measured)
    winter = measure_character(analysis, DayTraits(weather="snowy"), date(2026, 1, 15), measured)
    assert summer.warmth > 0.5 > -0.5 > winter.warmth


def test_the_label_is_one_short_line() -> None:
    """The label names the level, the shape, the warmth and whether it was a day off."""
    label = _character(_day(2000.0, (18.0, 23.0))).label()
    assert "\n" not in label
    assert label.startswith(("low", "medium", "high"))
    assert "workday" in label


# ---- what the character picks


def _character_with(**changes: object) -> DayCharacter:
    return replace(NEUTRAL, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("style", list(STYLES.values()), ids=list(STYLES))
def test_the_key_stays_within_two_fifths_from_day_to_day_and_the_mode_is_the_styles(style) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Over a year of days of every kind the key moves at most two fifths between days, in a mode of the style."""
    rng = np.random.default_rng(3)
    previous: int | None = None
    for offset in range(400):
        day = date(2026, 1, 1) + timedelta(days=offset)
        character = _character_with(
            warmth=float(rng.uniform(-1, 1)),
            energy=float(rng.uniform(0, 1)),
            day_off=bool(rng.integers(0, 2)),
            late_night=bool(rng.integers(0, 2)),
        )
        scale = scale_of_day(style, HOME, day, character)
        assert scale.mode in style.modes
        assert scale.mode in SCALES
        fifths = ((scale.root - HOME.root) * 7) % FIFTHS_IN_AN_OCTAVE
        if previous is not None:
            step = abs(fifths - previous)
            assert min(step, FIFTHS_IN_AN_OCTAVE - step) <= MAX_FIFTHS_BETWEEN_DAYS
        previous = fifths


def test_the_mode_is_bright_for_a_warm_busy_day_and_dark_for_a_cold_quiet_one() -> None:
    """Warm and busy picks a bright mode of the style, cold, quiet and late a dark one."""
    style = STYLES["ambient"]
    warm = scale_of_day(style, HOME, TUESDAY, _character_with(warmth=0.9, energy=0.9))
    cold = scale_of_day(style, HOME, TUESDAY, _character_with(warmth=-0.9, energy=0.1))
    late = scale_of_day(
        style, HOME, TUESDAY, _character_with(warmth=-0.2, energy=0.3, late_night=True)
    )
    assert warm.mode in {"lydian", "major", "mixolydian"}
    assert cold.mode in {"aeolian", "dorian", "phrygian"}
    assert late.mode in {"aeolian", "dorian", "phrygian"}


def test_tempo_and_chords_follow_the_energy() -> None:
    """A busy day is faster and changes chord sooner than a quiet one."""
    style = STYLES["lofi"]
    quiet, busy = _character_with(energy=0.1, level=LOW), _character_with(energy=0.9, level=HIGH)
    assert 0.0 <= tempo_target(style, quiet) < tempo_target(style, busy) <= 1.0
    assert min(chord_bars(quiet)) > max(chord_bars(busy))
    assert chord_bars(_character_with(level=MEDIUM)) == (2, 4)


# ---- the dealer


def _plan(day: date | None = TUESDAY, character: DayCharacter | None = None) -> Plan:
    series = _day(1800.0, (18.0, 22.0))
    return build_plan(
        series,
        analyze_day(series),
        "lofi",
        "seed",
        HOME,
        STANDARD,
        options=PieceOptions(day=day, area_key="kitchen", character=character),
    )


def test_pick_gives_the_best_fit_outright() -> None:
    """An option clearly better than the others wins on every day."""
    for offset in range(20):
        plan = _plan(TUESDAY + timedelta(days=offset))
        assert plan.pick("tempo", ["a", "b", "c"], [0.2, 0.9, 0.5]) == "b"
        assert plan.choices["tempo"] == "b"


def test_pick_rotates_between_options_that_fit_equally() -> None:
    """Options within the tolerance of the best take turns across days, and nothing else is ever picked."""
    near = [0.90, 0.90 - PICK_TOLERANCE / 2, 0.40]
    seen = [
        _plan(TUESDAY + timedelta(days=offset)).pick("groove", "xyz", near) for offset in range(12)
    ]
    assert set(seen) == {"x", "y"}
    assert all(first != second for first, second in pairwise(seen))


def test_pick_checks_its_arguments() -> None:
    """One fit for every option is required."""
    with pytest.raises(ValueError, match="one fit"):
        _plan().pick("x", ["a", "b"], [1.0])


def test_the_plan_knows_the_character_and_the_room_motif() -> None:
    """A plan without a character is neutral; the motif is that of its room."""
    plan = _plan()
    assert plan.character == NEUTRAL
    assert plan.motif() == room_motif("kitchen")
    assert _plan(character=_character_with(energy=0.9)).character.energy == 0.9


# ---- the form


@pytest.mark.parametrize("shape", [MORNING, EVENING, ALL_DAY, PEAK, FLAT])
@pytest.mark.parametrize("bar", BARS)
def test_the_form_is_on_phrase_boundaries_and_every_build_resolves(shape: str, bar: float) -> None:
    """Sections tile the piece, edges lie on 8 bars (4 when short), and a build is followed by a peak or release."""
    plan = _plan(character=_character_with(shape=shape, peak=0.7, energy=0.6))
    found = plan.sections(bar)
    grid = 8 * bar if plan.length / (8 * bar) >= MIN_FORM_CELLS else 4 * bar
    assert found[0][0] == 0.0
    assert found[-1][1] == plan.length
    assert found[0][2] == INTRO
    assert found[-1][2] == OUTRO
    for (_, end, _, _), (start, _, _, _) in pairwise(found):
        assert end == start
        assert start / grid == pytest.approx(round(start / grid))
    for index, (start, end, role, intensity) in enumerate(found):
        assert end > start
        assert 0.0 <= intensity <= 1.0
        if role == BUILD:
            assert found[index + 1][2] in {CLIMAX, RELEASE}


def test_the_form_follows_the_shape_of_the_day() -> None:
    """A morning day peaks early, an evening day late, a peak day where its peak was, a flat day never climbs."""

    def climax(shape: str, peak: float = 0.5) -> float:
        plan = _plan(character=_character_with(shape=shape, peak=peak))
        start, end, _, _ = next(item for item in plan.sections(2.0) if item[2] == CLIMAX)
        return (start + end) / 2 / plan.length

    assert climax(MORNING) < climax(EVENING)
    assert climax(MORNING) < 0.5 < climax(EVENING)
    assert climax(PEAK, 0.55) < climax(PEAK, 0.7)
    flat = _plan(character=_character_with(shape=FLAT)).sections(2.0)
    assert not {CLIMAX, BUILD} & {role for _, _, role, _ in flat}


# ---- the motif of a room


def test_a_room_motif_is_short_stepwise_and_resolves() -> None:
    """For many rooms: 3 to 5 notes within an octave, one leap of a fifth at most, ending on 0, 2 or 4, 4 or 8 beats."""
    keys = [f"area_{index}" for index in range(300)] + ["_house", "kitchen", "living_room"]
    for key in keys:
        motif = room_motif(key)
        degrees = [degree for degree, _ in motif]
        beats = sum(length for _, length in motif)
        leaps = [abs(b - a) for a, b in pairwise(degrees) if abs(b - a) > 1]
        assert 3 <= len(motif) <= 5
        assert max(degrees) - min(degrees) <= OCTAVE_DEGREES
        assert len(leaps) <= 1
        assert all(leap <= LARGEST_LEAP for leap in leaps)
        assert degrees[-1] in {0, 2, 4}
        assert beats in {4, 8}
        assert all(length > 0 for _, length in motif)


def test_motifs_are_stable_and_the_rooms_differ() -> None:
    """The same room always has the same motif, and a house of rooms has mostly different ones."""
    rooms = ["_house", "kitchen", "living_room", "bedroom", "office", "bathroom"]
    assert room_motif("kitchen") == room_motif("kitchen")
    assert len({room_motif(room) for room in rooms}) >= len(rooms) - 1


# ---- the loader of the real house


def _write_fixture(folder: Path, days: int = 6) -> None:
    """Write a small fixture of the documented shape: a CO2 monitor, a lamp, a workday sensor and the sun."""
    zone = ZoneInfo("Europe/Vienna")
    for offset in range(days):
        day = date(2026, 11, 10) + timedelta(days=offset)
        start = datetime(day.year, day.month, day.day, tzinfo=zone).timestamp()
        hours = np.arange(0.0, 24.0, 0.5)
        co2 = 450.0 + (300.0 + 200.0 * offset) * np.exp(-(((hours - 20.0) / 2.0) ** 2))

        def entity(
            entity_id: str, area: str, points: list[list[Any]], cls: str = "", unit: str = ""
        ) -> dict[str, object]:
            return {
                "entity_id": entity_id,
                "area": area,
                "domain": entity_id.partition(".")[0],
                "device_class": cls,
                "unit": unit,
                "source": "history",
                "points": points,
            }

        payload = {
            "day": day.isoformat(),
            "timezone": "Europe/Vienna",
            "entities": [
                entity(
                    "sensor.living_room_co2",
                    "living_room",
                    [[start + h * HOUR, float(v)] for h, v in zip(hours, co2, strict=True)],
                    "carbon_dioxide",
                    "ppm",
                ),
                entity(
                    "light.living_room",
                    "living_room",
                    [[start, 0.0], [start + 18 * HOUR, 1.0], [start + 23 * HOUR, 0.0]],
                ),
                entity("binary_sensor.workday", "house", [[start, float(day.weekday() < 5)]]),
                entity("sun.sun", "house", [[start, 0.0], [start + 7 * HOUR, 1.0]]),
                entity("weather.home", "house", [[start, "sunny"]]),
            ],
        }
        (folder / f"{day.isoformat()}.json").write_text(json.dumps(payload))


def test_the_loader_builds_series_and_baselines_like_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixture of the documented shape becomes classified, rescaled series with baselines from the earlier days."""
    _write_fixture(tmp_path)
    monkeypatch.setattr(house_fixture, "FIXTURE_DIR", tmp_path)
    house_fixture.forget_days()
    days = house_fixture.fixture_days()
    assert len(days) == 6
    assert house_fixture.areas() == ["living_room"]
    series, baselines, covered = house_fixture.load_day("living_room", days[-1])
    kinds = {item.entity_id: item.kind for item in series}
    assert kinds["sensor.living_room_co2"] == VoiceKind.AIR
    assert kinds["light.living_room"] == VoiceKind.CHORD
    assert any(kind == VoiceKind.SUN for kind in kinds.values())
    assert "binary_sensor.workday" not in kinds
    assert covered == DAY
    assert baselines["sensor.living_room_co2"].days == len(days) - 1
    character, analysis = house_fixture.measure("living_room", days[-1])
    assert analysis.measured
    assert 0.0 <= character.energy <= 1.0
    assert character.warmth > -1.0
    assert house_fixture.workday_of(days[-1]) is (days[-1].weekday() < 5)
    assert house_fixture.weather_of(days[-1]) == "sunny"
    whole, _, _ = house_fixture.load_day(None, days[-1])
    assert len(whole) >= len(series)
    with pytest.raises(FileNotFoundError):
        house_fixture.load_day(None, date(2020, 1, 1))
    house_fixture.forget_days()


# ---- the real house (skipped where the fixture is not)


@needs_fixture
@pytest.mark.parametrize(("room", "quiet", "busy", "_why"), house_fixture.PAIRS)
def test_the_days_of_the_real_house_are_clearly_apart(
    room: str, quiet: date, busy: date, _why: str
) -> None:
    """Each pair of days that the handover calls quiet and busy differs clearly in energy and level."""
    if quiet not in house_fixture.fixture_days() or busy not in house_fixture.fixture_days():
        pytest.skip("the fixture does not have these days")
    low, _ = house_fixture.measure(room, quiet)
    high, _ = house_fixture.measure(room, busy)
    assert high.energy - low.energy >= house_fixture.CLEARLY_APART
    assert high.level != low.level or (low.day_off and not high.day_off)
