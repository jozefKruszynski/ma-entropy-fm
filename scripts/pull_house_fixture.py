"""
Save days of a real Home Assistant history as a local fixture, under generic entity ids.

    HA_URL=http://homeassistant.local:8123 HA_TOKEN=... \
        uv run python scripts/pull_house_fixture.py .data/fixtures/mapping.json 2026-11-11 2026-11-19

The mapping is a JSON file `{"timezone": "Europe/Vienna", "candidates": {"<generic id>": "<real id>"}}`. One file per
local day is written to `.data/fixtures/real_house/YYYY-MM-DD.json`. The folder is gitignored: real data never goes
into the repository. Only numbers and on/off states are kept; the real ids stay in the mapping file.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).parent.parent / ".data" / "fixtures" / "real_house"
TIMEOUT = 120
ON_STATES = frozenset({"on", "open", "above_horizon", "home", "true"})
DROPPED = frozenset({"unavailable", "unknown", ""})
# generic id: (area, device class, unit)
META = {
    "sensor.living_room_co2": ("living_room", "carbon_dioxide", "ppm"),
    "sensor.bedroom_co2": ("bedroom", "carbon_dioxide", "ppm"),
    "sensor.office_co2": ("office", "carbon_dioxide", "ppm"),
    "sensor.bathroom_humidity": ("bathroom", "humidity", "%"),
    "switch.tv_plug": ("living_room", "outlet", ""),
    "sensor.tv_plug_power": ("living_room", "power", "W"),
    "sensor.desk_plug_power": ("office", "power", "W"),
    "sensor.bedroom_desk_plug_power": ("bedroom", "power", "W"),
    "sensor.total_power": ("house", "power", "W"),
    "light.kitchen": ("kitchen", "", ""),
    "light.office": ("office", "", ""),
    "light.living_room": ("living_room", "", ""),
    "binary_sensor.dishwasher_door": ("kitchen", "door", ""),
    "binary_sensor.workday": ("house", "", ""),
    "sun.sun": ("house", "", ""),
    "sensor.sun_elevation": ("house", "", "°"),
    "weather.home": ("house", "", ""),
}


def _value(domain: str, state: str | None) -> float | str | None:
    """Return a state as a number, 1.0/0.0 for on/off, the text for a weather entity, None to drop it."""
    if state is None or state in DROPPED:
        return None
    if domain == "weather":
        return state
    try:
        return float(state)
    except ValueError:
        return 1.0 if state in ON_STATES else 0.0


def _history(
    base: str, token: str, real_ids: list[str], start: datetime, end: datetime
) -> list[list[dict[str, str | None]]]:
    query = urllib.parse.urlencode(
        {
            "filter_entity_id": ",".join(real_ids),
            "end_time": end.isoformat(),
            "minimal_response": "",
            "no_attributes": "",
            "significant_changes_only": "0",
        }
    )
    url = f"{base}/api/history/period/{urllib.parse.quote(start.isoformat())}?{query}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
        rows: list[list[dict[str, str | None]]] = json.loads(response.read())
        return rows


def main() -> None:
    """Pull the days and write one fixture file per day."""
    base, token = os.environ["HA_URL"].rstrip("/"), os.environ["HA_TOKEN"]
    mapping = json.loads(Path(sys.argv[1]).read_text())
    tz = ZoneInfo(mapping["timezone"])
    to_generic = {real: generic for generic, real in mapping["candidates"].items()}
    first, last = date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
    OUT.mkdir(parents=True, exist_ok=True)
    day = first
    while day <= last:
        start = datetime(day.year, day.month, day.day, tzinfo=tz)
        rows = _history(base, token, list(to_generic), start, start + timedelta(days=1))
        entities = []
        for states in rows:
            if not states:
                continue
            generic = to_generic[states[0]["entity_id"]]
            domain = generic.partition(".")[0]
            area, device_class, unit = META.get(generic, ("house", "", ""))
            points = []
            for row in states:
                value = _value(domain, row["state"])
                changed = row["last_changed"]
                if value is not None and changed is not None:
                    stamp = datetime.fromisoformat(changed).timestamp()
                    points.append([max(stamp, start.timestamp()), value])
            entities.append(
                {
                    "entity_id": generic,
                    "area": area,
                    "domain": domain,
                    "device_class": device_class,
                    "unit": unit,
                    "source": "history",
                    "points": points,
                }
            )
            numbers = [v for _, v in points if isinstance(v, float)]
            span = f"{min(numbers):.0f} to {max(numbers):.0f}" if numbers else "text"
            print(f"{day} {generic:32} {len(points):6} points  {span}")
        payload = {"day": day.isoformat(), "timezone": mapping["timezone"], "entities": entities}
        (OUT / f"{day.isoformat()}.json").write_text(json.dumps(payload))
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
