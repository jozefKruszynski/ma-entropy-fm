"""
Load the days of a real house that `pull_house_fixture.py` saved, as the series the provider builds, and print their character.

    uv run python scripts/house_fixture.py            a table of the character of every room on every day
    uv run python scripts/house_fixture.py --pairs    only the pairs of contrasting days of the handover

The fixture is local, real and gitignored: one file per local day in `.data/fixtures/real_house/YYYY-MM-DD.json`,
`{"day", "timezone", "entities": [{"entity_id", "area", "domain", "device_class", "unit", "source", "points": [[unix
seconds, value], ...]}]}`. An entity is classified by `roles.classify`, rescaled against the days before it as the
provider does, and the baselines are built from the earlier days of the fixture. `load_day` is what the tests and the
tools use; without the folder everything says so and gives nothing.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY))

from entropy_fm.baseline import EntityBaseline, build_baselines, has_data  # noqa: E402
from entropy_fm.character import DayCharacter, measure_character, raw_energy  # noqa: E402
from entropy_fm.constants import BASELINE_DAYS  # noqa: E402
from entropy_fm.palette import measure_traits  # noqa: E402
from entropy_fm.references import rescale  # noqa: E402
from entropy_fm.roles import classify  # noqa: E402
from entropy_fm.scape import analysis_of  # noqa: E402
from entropy_fm.surprise import Surprise  # noqa: E402
from entropy_fm.timeline import (  # noqa: E402
    SUN_ELEVATION_ENTITY,
    SUN_ENTITY,
    EntityInfo,
    Series,
    VoiceKind,
    build_series,
)

if TYPE_CHECKING:
    from entropy_fm.day_analysis import DayAnalysis

FIXTURE_DIR = REPOSITORY / ".data" / "fixtures" / "real_house"
HOUSE_AREA = "house"
WORKDAY_ENTITY = "binary_sensor.workday"
SUN_GENERIC = frozenset({"sun.sun", "sensor.sun_elevation"})
WEATHER_DOMAIN = "weather"
DAY_SECONDS = 86400.0
ANOMALY_SCORE = 0.5
BINARY_DOMAINS = frozenset({"binary_sensor", "switch", "light", "input_boolean", "fan"})
# the pairs of contrasting days: (room, quiet day, busy day, what makes the difference)
PAIRS = (
    ("living_room", date(2026, 11, 14), date(2026, 11, 16), "CO2 max 760 against 2600 ppm"),
    ("bedroom", date(2026, 11, 15), date(2026, 11, 16), "CO2 mean 520 against the busiest night"),
    ("bathroom", date(2026, 11, 15), date(2026, 11, 19), "no shower against an 84 % spike"),
    # Thursday 19 was a light workday (the desk on half the day), so the busy workday of the office is Wednesday 18
    ("office", date(2026, 11, 15), date(2026, 11, 18), "day off against a workday"),
)
# two days of a pair are clearly apart when their energy differs by at least this much
CLEARLY_APART = 0.35
# a feature that spreads over less than this of its scale across the days of a room barely varies
BARELY_VARIES = 0.2


@dataclass
class Loaded:
    """One day of the house as the provider reads it: the series, what they usually do, how much of the day happened."""

    series: list[Series]
    baselines: dict[str, EntityBaseline]
    covered: float
    day: date
    area: str | None


def fixture_days() -> list[date]:
    """Return the days that the fixture has, oldest first; empty when the folder is missing."""
    if not FIXTURE_DIR.is_dir():
        return []
    found = []
    for path in FIXTURE_DIR.glob("*.json"):
        try:
            found.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return sorted(found)


@functools.cache
def _payload(day: date) -> dict[str, Any] | None:
    """Return the file of a day, or None when there is none."""
    path = FIXTURE_DIR / f"{day.isoformat()}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def forget_days() -> None:
    """Forget the files that were read, for a test that moves `FIXTURE_DIR`."""
    _payload.cache_clear()


def _start(payload: dict[str, Any], day: date) -> float:
    """Return the epoch seconds of the start of the local day."""
    return datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(payload["timezone"])).timestamp()


def areas() -> list[str]:
    """Return the rooms of the fixture (every area but the house), in the order they first appear."""
    found: list[str] = []
    for day in fixture_days():
        payload = _payload(day)
        for entity in (payload or {}).get("entities", []):
            area = str(entity["area"])
            if area != HOUSE_AREA and area not in found:
                found.append(area)
    return found


def _text_points(entity: dict[str, Any]) -> list[tuple[float, str]]:
    """Return the points of a text entity (the weather) as (time, state)."""
    return [(float(stamp), str(value)) for stamp, value in entity["points"]]


def weather_of(day: date) -> str | None:
    """Return the weather state that lasted longest on a day, or None without a weather entity."""
    payload = _payload(day)
    if payload is None:
        return None
    for entity in payload["entities"]:
        if entity["domain"] != WEATHER_DOMAIN or not entity["points"]:
            continue
        points = _text_points(entity)
        start = _start(payload, day)
        ends = [t for t, _ in points[1:]] + [start + DAY_SECONDS]
        lasted: dict[str, float] = {}
        for (begin, state), end in zip(points, ends, strict=True):
            lasted[state] = lasted.get(state, 0.0) + max(end - max(begin, start), 0.0)
        return max(lasted, key=lambda state: lasted[state])
    return None


def workday_of(day: date) -> bool | None:
    """Return whether the workday sensor says the day is a workday, or None when the fixture has none."""
    payload = _payload(day)
    if payload is None:
        return None
    for entity in payload["entities"]:
        if entity["entity_id"] == WORKDAY_ENTITY and entity["points"]:
            start = _start(payload, day)
            noon = start + DAY_SECONDS / 2.0
            before = [float(v) for t, v in entity["points"] if float(t) <= noon]
            return bool((before[-1] if before else float(entity["points"][0][1])) > 0.5)
    return None


def _info(entity: dict[str, Any]) -> EntityInfo | None:
    """Classify an entity of the fixture as the provider does; None for what makes no sound."""
    entity_id = str(entity["entity_id"])
    domain = str(entity["domain"])
    if domain == WEATHER_DOMAIN or entity_id == WORKDAY_ENTITY:
        return None
    area = str(entity["area"])
    if entity_id in SUN_GENERIC:
        return EntityInfo(
            SUN_ELEVATION_ENTITY if entity_id == "sensor.sun_elevation" else SUN_ENTITY,
            entity_id,
            None,
            VoiceKind.SUN,
        )
    state = "on" if domain in BINARY_DOMAINS else "1.0"
    device_class = str(entity.get("device_class") or "") or None
    found = classify(
        entity_id, entity_id, device_class, str(entity.get("unit") or "") or None, state
    )
    if found is None:
        return None
    return EntityInfo(
        entity_id,
        entity_id,
        area,
        found.kind,
        found.cumulative,
        found.description,
        (device_class or "").lower(),
    )


def _day_series(day: date, area: str | None) -> list[Series] | None:
    """Return the series of a room (or of the house) on a day, without the sun, or None when the day is missing."""
    payload = _payload(day)
    if payload is None:
        return None
    start = _start(payload, day)
    found: list[Series] = []
    for entity in payload["entities"]:
        info = _info(entity)
        if info is None or info.kind == VoiceKind.SUN or not entity["points"]:
            continue
        if area is not None and entity["area"] != area:
            continue
        times = [float(stamp) - start for stamp, _ in entity["points"]]
        values = [float(value) for _, value in entity["points"]]
        found.append(build_series(info, times, values))
    return found


def _sun(day: date) -> list[Series]:
    """Return the sun of a day: the elevation when the fixture has it, else the sun entity."""
    payload = _payload(day)
    if payload is None:
        return []
    start = _start(payload, day)
    chosen: dict[str, Series] = {}
    for entity in payload["entities"]:
        info = _info(entity)
        if info is None or info.kind != VoiceKind.SUN or not entity["points"]:
            continue
        times = [float(stamp) - start for stamp, _ in entity["points"]]
        values = [float(value) for _, value in entity["points"]]
        chosen[info.entity_id] = build_series(info, times, values)
    for key in (SUN_ELEVATION_ENTITY, SUN_ENTITY):
        if key in chosen:
            return [chosen[key]]
    return []


def load_day(area: str | None, day: date) -> tuple[list[Series], dict[str, EntityBaseline], float]:
    """
    Return one day of a room, or of the whole house, as the provider reads it.

    The series are rescaled against the baselines, which are built from the days of the fixture before the day (at
    most a week), so that a day with at least three days before it is measured against its own days.

    :param area: The room, such as `living_room`; None for the whole house.
    :param day: The day; the fixture has to have it.
    :return: The series (with the sun), the baselines by entity id, and the seconds of the day that happened.
    :raises FileNotFoundError: When the fixture has no file for the day.
    """
    today = _day_series(day, area)
    if today is None:
        raise FileNotFoundError(f"no fixture for {day}: {FIXTURE_DIR}")
    before = [
        found
        for offset in range(1, BASELINE_DAYS + 1)
        if (found := _day_series(day - timedelta(days=offset), area)) is not None
    ]
    baselines = build_baselines(before)
    series, _ = rescale([*today, *_sun(day)], baselines)
    return series, baselines, DAY_SECONDS


def load_week(area: str | None, day: date, baselines: dict[str, EntityBaseline]) -> list[float]:
    """
    Return how busy the days before a day were, as the provider's `HistoryStore.week_energies` measures them.

    :param area: The room, or None for the whole house.
    :param day: The day; the fixture days before it (at most a week) count.
    :param baselines: The baselines of the day, which every earlier day is rescaled with.
    """
    week = []
    for offset in range(1, BASELINE_DAYS + 1):
        found = _day_series(day - timedelta(days=offset), area)
        if found is not None and has_data(found):
            week.append(raw_energy(rescale(found, baselines)[0], DAY_SECONDS))
    return week


def measure(
    area: str | None, day: date, *, with_traits: bool = True
) -> tuple[DayCharacter, DayAnalysis]:
    """
    Return the character and the analysis of a day as the provider works them out.

    :param area: The room, or None for the whole house.
    :param day: The day.
    :param with_traits: Whether the weather of the day goes into the warmth.
    """
    series, baselines, covered = load_day(area, day)
    traits = measure_traits(series, baselines, covered, weather_of(day)) if with_traits else None
    featured = Surprise(series, baselines).featured(0.0, 1.0, covered)
    anomalies = [(t, score, entity) for t, score, entity in featured if score >= ANOMALY_SCORE]
    analysis = analysis_of(series, covered, anomalies)
    character = measure_character(
        analysis, traits, day, series, workday=workday_of(day), week=load_week(area, day, baselines)
    )
    return character, analysis


def _row(area: str, day: date, character: DayCharacter, *, measured: bool) -> str:
    """Return one line of the table."""
    return (
        f"{day:%a %d %b}  {area:12} {'yes' if measured else 'no ':3}  "
        f"{character.energy:5.2f} {character.level:6}  {character.shape:8}  "
        f"{character.peak * 24:5.1f}h  {character.warmth:+5.2f}  {character.stretches:2}  "
        f"{character.regularity:4.2f}  {character.events:4.2f}  "
        f"{'late' if character.late_night else '    '}  {'off' if character.day_off else 'work'}"
    )


HEADER = "day           room         meas  energy level   shape     peak   warmth  st  reg   evts  late  day"
FEATURES = ("energy", "peak", "warmth", "stretches", "regularity", "events")
# the scale a feature is judged on, to say that it barely varies
FEATURE_SCALE = {"stretches": 6.0}


def _spread_report(rows: list[tuple[str, date, DayCharacter]]) -> list[str]:
    """Return lines about how much each feature varies over the days of each room."""
    lines = [
        "",
        "spread of each feature over the days of a room (range / scale; * = barely varies)",
    ]
    rooms = sorted({area for area, _, _ in rows})
    lines.append(f"{'feature':12}" + "".join(f"{room[:11]:>12}" for room in rooms) + "     overall")
    for feature in FEATURES:
        scale = FEATURE_SCALE.get(feature, 1.0)
        cells = []
        for room in [*rooms, None]:
            values = [
                float(getattr(character, feature))
                for area, _, character in rows
                if room in (None, area)
            ]
            span = (max(values) - min(values)) / scale if values else 0.0
            cells.append(f"{span:10.2f}{'*' if span < BARELY_VARIES else ' '} ")
        lines.append(f"{feature:12}" + "".join(f"{cell:>12}" for cell in cells))
    shapes = {room: sorted({c.shape for a, _, c in rows if a == room}) for room in rooms}
    lines.extend(f"shapes {room:12} {', '.join(found)}" for room, found in shapes.items())
    return lines


def _pair_report(table: dict[tuple[str, date], DayCharacter]) -> list[str]:
    """Return lines that compare the contrasting pairs of days of the handover."""
    lines = ["", "the pairs of contrasting days"]
    for room, quiet, busy, why in PAIRS:
        low, high = table.get((room, quiet)), table.get((room, busy))
        if low is None or high is None:
            lines.append(f"{room:12} {quiet:%d %b} / {busy:%d %b}: missing")
            continue
        gap = high.energy - low.energy
        verdict = "apart" if gap >= CLEARLY_APART and low.level != high.level else "TOO CLOSE"
        lines.append(
            f"{room:12} {quiet:%a %d} {low.energy:4.2f} {low.level:6}  vs  {busy:%a %d} {high.energy:4.2f} "
            f"{high.level:6}  gap {gap:+.2f}  {verdict}  ({why}; events {low.events:.2f} vs {high.events:.2f}, "
            f"{'day off' if low.day_off else 'work'} vs {'day off' if high.day_off else 'work'})"
        )
    return lines


def main() -> None:
    """Print the character of every room on every day of the fixture."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--pairs", action="store_true", help="only the pairs of contrasting days")
    parser.add_argument(
        "--no-weather", action="store_true", help="leave the weather out of the warmth"
    )
    args = parser.parse_args()
    days = fixture_days()
    if not days:
        print(f"No fixture in {FIXTURE_DIR}: run scripts/pull_house_fixture.py first.")
        return
    rooms = [*areas(), None]
    rows: list[tuple[str, date, DayCharacter]] = []
    table: dict[tuple[str, date], DayCharacter] = {}
    lines = [HEADER]
    for day in days:
        for room in rooms:
            character, analysis = measure(room, day, with_traits=not args.no_weather)
            name = room or "whole_house"
            lines.append(_row(name, day, character, measured=analysis.measured))
            if room is not None:
                rows.append((name, day, character))
            table[name, day] = character
        lines.append("")
    if not args.pairs:
        print("\n".join(lines))
        print("\n".join(_spread_report(rows)))
    print("\n".join(_pair_report(table)))
    print(f"\n{len(days)} fixture days, {len(rooms) - 1} rooms")


if __name__ == "__main__":
    main()
