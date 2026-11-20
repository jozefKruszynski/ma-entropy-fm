"""
The invented sparse house of the demos: a house with no motion or occupancy sensors at all.

It is modelled on the statistics of a real house with air quality monitors, plugs and a few lights
(synthetic numbers, no personal data): CO2 in the living room, the bedroom and the office, a bathroom
humidity sensor with a shower almost every day, a TV plug in the evening, the feed of a desk on workdays, the
total power with the kettle and the cooking, a fridge that cycles, the lights of a kitchen that come on for a few
minutes at a time, an office backlight, the door of a dishwasher that opens about seven times a day, a washer
that runs three times in a week, and the things that give nothing: blinds that never move, a dead plant sensor and
a temperature sensor that is stuck. Entities are classified with `roles.classify` of the engine that is
imported, so the same house can be played by an older engine to compare.

    from sparse_house import build_day
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import demo_render
import numpy as np
from demo_render import DAY, hours, intervals_to_events

if TYPE_CHECKING:
    from entropy_fm.timeline import Series

FIRST_DAY = date(2026, 11, 4)
DAYS = 14
# the day without a shower, and the days without TV and the days the washer runs, counted from FIRST_DAY
NO_SHOWER = 5
NO_TV = (2, 9)
WASH_DAYS = (1, 4, 11)
DOOR_OPENS = (
    ("07:55", "08:20"),
    ("12:40", "13:00"),
    ("18:10", "18:40"),
    ("19:45", "20:15"),
    ("21:10", "21:40"),
)

Points = list[tuple[str, float]]


def _smooth_curve(
    points: Points, rng: np.random.Generator, noise: float, step: float = 300.0
) -> tuple[list[float], list[float]]:
    """Return a sensor sampled every `step` seconds through the points (linear), with noise."""
    xs = [hours(clock) for clock, _ in points]
    ys = [value for _, value in points]
    times = np.arange(0.0, DAY, step)
    values = np.interp(times, xs, ys) + rng.normal(0.0, noise, len(times))
    return times.tolist(), values.tolist()


def _co2(
    points: Points, rng: np.random.Generator, noise: float = 12.0
) -> tuple[list[float], list[float]]:
    """Return the CO2 of a room in ppm, never under the outdoor level."""
    times, values = _smooth_curve(points, rng, noise)
    return times, [max(value, 395.0) for value in values]


def _living_co2(rng: np.random.Generator, *, weekend: bool) -> tuple[list[float], list[float]]:
    peak = rng.uniform(1400.0, 2600.0)
    day = 750.0 if weekend else 520.0
    return _co2(
        [
            ("00:00", rng.uniform(700.0, 1100.0)),
            ("03:00", 460.0),
            ("06:30", 430.0),
            ("07:30", 650.0),
            ("09:00", day),
            ("14:00", day + (250.0 if weekend else 0.0)),
            ("17:00", 700.0),
            ("19:00", peak * 0.7),
            ("21:30", peak),
            ("23:30", peak * 0.45),
            ("23:59", peak * 0.4),
        ],
        rng,
    )


def _bedroom_co2(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    peak = rng.uniform(2400.0, 3700.0)
    return _co2(
        [
            ("00:00", peak * 0.55),
            ("03:00", peak * 0.85),
            ("06:45", peak),
            ("07:20", peak * 0.5),
            ("09:00", 560.0),
            ("12:00", 450.0),
            ("22:00", 520.0),
            ("23:00", 850.0),
            ("23:59", 1500.0),
        ],
        rng,
        20.0,
    )


def _office_co2(rng: np.random.Generator, *, workday: bool) -> tuple[list[float], list[float]]:
    if not workday:
        return _co2([("00:00", 430.0), ("11:00", 450.0), ("15:00", 520.0), ("23:59", 420.0)], rng)
    peak = rng.uniform(950.0, 1270.0)
    return _co2(
        [
            ("00:00", 420.0),
            ("08:45", 410.0),
            ("10:00", peak * 0.8),
            ("12:30", peak),
            ("13:15", peak * 0.6),
            ("15:00", peak * 0.95),
            ("17:30", peak * 0.9),
            ("18:30", 520.0),
            ("23:59", 410.0),
        ],
        rng,
    )


def _bathroom_humidity(
    rng: np.random.Generator, index: int, *, weekend: bool
) -> tuple[list[float], list[float]]:
    """Return a baseline of 35 to 45 percent with a shower of 65 to 84 percent on most days."""
    times = np.arange(0.0, DAY, 120.0)
    values = 40.0 + 3.0 * np.sin(2 * np.pi * (times / 3600.0 - 14.0) / 24.0)
    values = values + rng.normal(0.0, 0.4, len(times))
    if index != NO_SHOWER:
        start = hours("09:10" if weekend else "06:50") + rng.uniform(0.0, 1800.0)
        peak = rng.uniform(65.0, 84.0)
        since = times - start
        rise = np.clip(since / 420.0, 0.0, 1.0)
        fall = np.exp(-np.clip(since - 420.0, 0.0, None) / 1500.0)
        values = values + np.where(since > 0.0, (peak - 40.0) * rise * fall, 0.0)
    return times.tolist(), values.tolist()


def _tv(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Return the power of a TV plug: standby, then 90 to 270 W that moves with the picture."""
    start = hours("19:00") + rng.uniform(0.0, 5400.0)
    end = start + rng.uniform(2.0, 3.6) * 3600.0
    times, values = [0.0], [0.4]
    for moment in np.arange(start, end, 150.0):
        times.append(float(moment))
        values.append(float(rng.uniform(90.0, 270.0)))
    times.append(float(end))
    values.append(0.4)
    return times, values


def _desk(rng: np.random.Generator, *, workday: bool) -> tuple[list[float], list[float]]:
    """Return the feed of a desk: 8 W idle, 75 to 135 W in use on workdays, with peaks to 460 W."""
    times, values = [0.0], [8.0]
    if workday:
        mean = rng.uniform(75.0, 135.0)
        for moment in np.arange(hours("08:45"), hours("17:40"), 60.0):
            spike = rng.uniform(250.0, 460.0) if rng.random() < 0.04 else 0.0
            times.append(float(moment))
            values.append(float(max(mean + rng.normal(0.0, 18.0), 20.0) + spike))
        times.append(hours("17:45"))
        values.append(8.0)
    else:
        start = hours("14:00") + rng.uniform(0.0, 7200.0)
        for moment in np.arange(start, start + 3000.0, 60.0):
            times.append(float(moment))
            values.append(float(max(80.0 + rng.normal(0.0, 12.0), 20.0)))
        times.append(start + 3000.0)
        values.append(8.0)
    return times, values


def _total_power(rng: np.random.Generator, late: float) -> tuple[list[float], list[float]]:
    """Return the measured total: a few hundred watts, with the kettle, the hob and the oven."""
    grid = np.arange(0.0, DAY, 60.0)
    values = 180.0 + 40.0 * rng.random(len(grid)) + 90.0 * (grid > hours("06:30"))
    peaks = [
        (hours("07:10") + late, 180.0, rng.uniform(2000.0, 2300.0)),
        (hours("12:20"), 420.0, rng.uniform(2000.0, 3200.0)),
        (hours("17:50"), 3300.0, rng.uniform(2800.0, 4500.0)),
        (hours("21:10"), 150.0, rng.uniform(1800.0, 2300.0)),
    ]
    for start, length, watts in peaks:
        inside = (grid >= start) & (grid < start + length)
        values = values + np.where(inside, watts * rng.uniform(0.6, 1.0, len(grid)), 0.0)
    return grid.tolist(), values.tolist()


def _fridge(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Return a compressor of 60 W that runs a third of the time: about 20 W on average."""
    times, values = [], []
    now = 0.0
    while now < DAY:
        on = rng.uniform(11.0, 15.0) * 60.0
        times += [now, now + on]
        values += [float(rng.normal(60.0, 3.0)), 1.0]
        now += on + rng.uniform(24.0, 30.0) * 60.0
    return times, values


def _kitchen_lights(rng: np.random.Generator, late: float) -> list[tuple[float, float]]:
    """Return two to four short periods, about an hour in all."""
    options = [
        (hours("07:00") + late, rng.uniform(600.0, 1500.0)),
        (hours("12:15"), rng.uniform(600.0, 1800.0)),
        (hours("18:20"), rng.uniform(900.0, 2400.0)),
        (hours("21:05"), rng.uniform(300.0, 900.0)),
    ]
    count = int(rng.integers(2, 5))
    chosen = sorted(rng.choice(len(options), size=count, replace=False))
    return [(options[i][0], options[i][0] + options[i][1]) for i in chosen]


def _jitter(rng: np.random.Generator, span: float = 900.0) -> float:
    return float(rng.uniform(-span, span))


def build_day(index: int) -> dict[str, list[Series]]:
    """
    Return the series of the five rooms of the sparse house and the sun on one day of the fortnight.

    :param index: The day, counted from the 4th of November.
    """
    from entropy_fm import roles  # noqa: PLC0415
    from entropy_fm.timeline import EntityInfo, VoiceKind, build_series  # noqa: PLC0415

    rng = np.random.default_rng(7000 + index)
    weekday = (FIRST_DAY + timedelta(days=index)).weekday()
    weekend = weekday >= 5
    workday = not weekend
    late = 5400.0 if weekend else 0.0
    areas: dict[str, list[Series]] = {
        "kitchen": [],
        "living": [],
        "bedroom": [],
        "office": [],
        "bathroom": [],
    }

    def add(  # noqa: PLR0913, PLR0917
        area: str,
        entity_id: str,
        name: str,
        device_class: str | None,
        unit: str | None,
        data: tuple[list[float], list[float]],
        state: str = "1.0",
    ) -> None:
        role = roles.classify(entity_id, name, device_class, unit, state)
        if role is None:
            return
        info = EntityInfo(
            entity_id,
            name,
            area,
            role.kind,
            role.cumulative,
            role.description,
            (device_class or "").lower(),
        )
        areas[area].append(build_series(info, data[0], data[1]))

    def lights(area: str, entity_id: str, name: str, periods: list[tuple[float, float]]) -> None:
        add(area, entity_id, name, None, None, intervals_to_events(periods), "on")

    # the living room
    add(
        "living",
        "sensor.living_room_air_quality_monitor_carbon_dioxide",
        "CO2",
        "carbon_dioxide",
        "ppm",
        _living_co2(rng, weekend=weekend),
    )
    tv = ([0.0], [0.4]) if index in NO_TV else _tv(rng)
    add("living", "sensor.tv_plug_power", "TV plug power", "power", "W", tv)
    lights(
        "living",
        "light.white_and_colour",
        "Lamps",
        [(hours("17:20") + _jitter(rng), hours("23:00") + _jitter(rng))],
    )
    lights(
        "living",
        "light.wled_soundreactive",
        "WLED",
        []
        if index % 3 == 0
        else [(hours("19:30") + _jitter(rng), hours("22:40") + _jitter(rng, 1200))],
    )
    add(
        "living",
        "sensor.living_room_temperature",
        "Living room temperature",
        "temperature",
        "°C",
        _smooth_curve(
            [("00:00", 21.0), ("06:00", 20.4), ("15:00", 22.0), ("21:00", 22.6), ("23:59", 21.4)],
            rng,
            0.05,
        ),
    )
    add("living", "cover.living_room_blinds", "Blinds", None, None, ([0.0, 41000.0], [0.0, 1.0]))
    add("living", "sensor.plant_moisture", "Plant", "moisture", "%", ([], []))

    # the bedroom
    add(
        "bedroom",
        "sensor.bedroom_air_quality_monitor_carbon_dioxide",
        "CO2",
        "carbon_dioxide",
        "ppm",
        _bedroom_co2(rng),
    )
    add(
        "bedroom",
        "sensor.bedroom_temperature",
        "Bedroom temperature",
        "temperature",
        "°C",
        ([0.0], [19.5]),
    )

    # the office
    add(
        "office",
        "sensor.office_air_quality_monitor_carbon_dioxide",
        "CO2",
        "carbon_dioxide",
        "ppm",
        _office_co2(rng, workday=workday),
    )
    add(
        "office",
        "sensor.desk_feed_meter_power",
        "Desk feed meter",
        "power",
        "W",
        _desk(rng, workday=workday),
    )
    lights(
        "office",
        "light.office_backlight_left_light",
        "Office backlight",
        [(hours("13:20") + _jitter(rng), hours("16:50") + _jitter(rng))]
        if workday
        else ([(hours("15:00"), hours("15:50"))] if index % 2 else []),
    )

    # the bathroom
    add(
        "bathroom",
        "sensor.bathroom_sensor_humidity",
        "Bathroom humidity",
        "humidity",
        "%",
        _bathroom_humidity(rng, index, weekend=weekend),
    )

    # the kitchen
    add(
        "kitchen",
        "sensor.total_measured_power",
        "Total measured power",
        "power",
        "W",
        _total_power(rng, late),
    )
    add("kitchen", "sensor.kitchen_fridge_plug_power", "Fridge", "power", "W", _fridge(rng))
    lights(
        "kitchen", "light.kitchen_coffee_light_light", "Coffee light", _kitchen_lights(rng, late)
    )
    door = [(hours(a) + _jitter(rng, 600.0), hours(b) + _jitter(rng, 600.0)) for a, b in DOOR_OPENS]
    keep = [item for i, item in enumerate(door) if (i != 0 or workday) and rng.random() < 0.92]
    # a few short extra openings, so that there are about seven a day
    keep += [
        (hours("19:10") + k * 600.0 + _jitter(rng, 120.0), hours("19:10") + k * 600.0 + 45.0)
        for k in range(2)
    ]
    add(
        "kitchen",
        "binary_sensor.siemens_dishwasher_door",
        "Dishwasher door",
        "door",
        None,
        intervals_to_events(
            sorted((a, max(b, a + 20.0)) for a, b in keep if hours("00:30") < a < hours("23:30"))
        ),
        "off",
    )
    wash = demo_render.washing_machine(rng) if index in WASH_DAYS else ([0.0], [1.0])
    if index in WASH_DAYS:
        wash = demo_render.shifted(wash, rng.uniform(-3600.0, 7200.0))
    add("kitchen", "sensor.washing_machine_power", "Washing machine", "power", "W", wash)

    sun = EntityInfo("sensor.sun_solar_elevation", "Sun", None, VoiceKind.SUN)
    sun_data = demo_render.sun()
    return {**areas, "sun": [build_series(sun, sun_data[0], sun_data[1])]}


__all__ = ["DAYS", "FIRST_DAY", "build_day"]
