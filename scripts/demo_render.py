"""
The invented house of the demos of Entropy FM, and the helpers that render and write its soundscapes.

Each day of an invented fortnight has a Kitchen and a Living Room: a fridge that cycles, a kettle, a door, an oven
at dinner, a washing machine, motion at meals, lights in the evening, a temperature that drifts from 20 to 23
degrees, humidity and the sun. The fortnight 4 to 17 November 2026 has weekends (a later start of the day), a
heatwave on the 12th, a washing machine that runs at about 10:00 on most days, and on the last day, a Tuesday, a
door at 03:00 and a wash at 21:30. `Demo` reads a day the way the provider does (baselines of the week before, the
character of the day, the anomalies) and makes the score of its soundscape in a style.

    uv run python scripts/soundscapes.py [output folder]    the deliverable files of the soundscapes
"""

import subprocess
import sys
import wave
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY))
DAY = 86400.0


def hours(clock: str) -> float:
    """Return seconds since midnight of a clock time such as 07:30."""
    hour, minute = clock.split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0


def intervals_to_events(intervals: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Turn on-intervals into times and values of a binary entity that starts off."""
    times, values = [0.0], [0.0]
    for start, end in sorted(intervals):
        times += [start, end]
        values += [1.0, 0.0]
    return times, values


def bursts(
    rng: np.random.Generator,
    start: float,
    end: float,
    on: tuple[float, float],
    gap: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return motion bursts between two times."""
    found = []
    now = start
    while now < end:
        length = rng.uniform(*on)
        found.append((now, min(now + length, end)))
        now += length + rng.uniform(*gap)
    return found


def curve(
    points: list[tuple[str, float]], rng: np.random.Generator, noise: float, step: float
) -> tuple[list[float], list[float]]:
    """Return a slowly drifting sensor sampled every `step` seconds through the given points."""
    xs = [hours(clock) for clock, _ in points]
    ys = [value for _, value in points]
    times = np.arange(0.0, DAY, step)
    values = np.interp(times, xs, ys) + rng.normal(0, noise, len(times))
    return times.tolist(), values.tolist()


def fridge(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Return the power of a compressor that runs about 14 minutes in every 40."""
    times, values = [], []
    now = 0.0
    while now < DAY:
        for offset in range(0, 14 * 60, 30):
            spike = 150.0 if offset < 60 else 0.0
            times.append(now + offset)
            values.append(78 + rng.normal(0, 3) + spike)
        times.append(now + 14 * 60)
        values.append(2.5)
        now += rng.uniform(38, 42) * 60
    return times, values


def heating(
    rng: np.random.Generator, start: float, end: float, watts: float, duty: tuple[float, float]
) -> tuple[list[float], list[float]]:
    """Return the power of an element that a thermostat switches on and off."""
    times, values = [0.0], [1.5]
    now = start
    while now < end:
        on = rng.uniform(*duty)
        times += [now, now + on]
        values += [watts + rng.normal(0, 20), 1.5]
        now += on + rng.uniform(*duty) * 0.6
    return times, values


def washing_machine(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Return the power of a wash from 10:00 to 11:30: fill, heat, wash, rinse and spin."""
    times, values = [0.0], [1.0]

    def add(start: float, end: float, base: float, wobble: float, step: float = 15.0) -> None:
        for moment in np.arange(start, end, step):
            times.append(float(moment))
            values.append(max(base + rng.normal(0, wobble), 0.0))

    add(hours("10:00"), hours("10:03"), 8, 2)
    # the heater, switched by its thermostat
    now = hours("10:03")
    while now < hours("10:26"):
        add(now, now + 150, 2050, 15)
        add(now + 150, now + 210, 120, 20)
        now += 210
    # the drum turns one way, then the other
    now = hours("10:26")
    while now < hours("10:56"):
        add(now, now + 20, 190, 15, 5)
        add(now + 20, now + 28, 40, 8, 4)
        now += 28
    add(hours("10:56"), hours("11:10"), 130, 40)
    for index, moment in enumerate(np.arange(hours("11:10"), hours("11:24"), 20.0)):
        times.append(float(moment))
        values.append(min(180 + index * 12 + rng.normal(0, 15), 430))
    times.append(hours("11:24"))
    values.append(60)
    times.append(hours("11:30"))
    values.append(1.2)
    return times, values


def dishwasher(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Return the power of a dishwasher run from 20:30 to 22:30."""
    times, values = [0.0], [0.8]

    def add(start: str, end: str, base: float, wobble: float, step: float = 20.0) -> None:
        for moment in np.arange(hours(start), hours(end), step):
            times.append(float(moment))
            values.append(max(base + rng.normal(0, wobble), 0.0))

    add("20:30", "20:32", 6, 1)  # filling
    add("20:32", "20:47", 2010, 25)  # heating the wash water
    add("20:47", "21:00", 95, 12)  # the circulation pump
    add("21:00", "21:01", 340, 20, 10.0)  # drain pump
    add("21:01", "21:20", 85, 12)  # pump again, a cooler rinse
    add("21:20", "21:35", 1950, 30)  # second heat, the hot rinse
    add("21:35", "21:50", 110, 12)
    add("21:50", "21:51", 360, 20, 10.0)  # drain
    add("21:51", "22:25", 55, 8)  # drying with a fan
    times.append(hours("22:25"))
    values.append(0.8)
    return times, values


def sun() -> tuple[list[float], list[float]]:
    """Return the elevation of the sun in degrees through a day of November."""
    times = np.arange(0.0, DAY, 300.0)
    elevation = -14 + 36 * np.cos(2 * np.pi * (times / 3600 - 11.75) / 24)
    return times.tolist(), elevation.tolist()


FIRST_DAY = date(2026, 11, 4)
DAYS = 14
HEATWAVE = 8
ANOMALY_DAY = DAYS - 1
ROUTINE_TUESDAY = 6


def build_day(timeline: ModuleType, index: int = ANOMALY_DAY) -> dict[str, list[object]]:
    """
    Return the series of the Kitchen, the Living Room and the sun on one day of the fortnight.

    :param timeline: The imported timeline module.
    :param index: The day, counted from the 4th of November; the last one is the anomaly day.
    """
    rng = np.random.default_rng(2026 + index)
    kind = timeline.VoiceKind
    weekend = (FIRST_DAY + timedelta(days=index)).weekday() >= 5
    late = 5400.0 if weekend else 0.0
    heat = 9.0 if index == HEATWAVE else 0.0
    anomaly = index == ANOMALY_DAY

    def later(start: str, end: str) -> tuple[float, float]:
        return hours(start) + late, hours(end) + late

    def series(
        entity_id: str,
        name: str,
        voice: object,
        data: tuple[list[float], list[float]],
        device_class: str = "",
    ) -> object:
        info = timeline.EntityInfo(entity_id, name, None, voice, device_class=device_class)
        return timeline.build_series(info, data[0], data[1])

    def warm(points: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return [(clock, value + heat) for clock, value in points]

    door_times = [
        (hours("07:10") + late + rng.uniform(0, 600), 25.0),
        (hours("12:15") + rng.uniform(0, 600), 20.0),
        (hours("17:20") + rng.uniform(0, 900), 30.0),
    ]
    if anomaly:
        door_times.append((hours("03:00"), 40.0))
    wash = washing_machine(rng)
    if anomaly:
        wash = shifted(wash, hours("21:30") - hours("10:00"))
    elif index % 7 == 3:
        wash = ([0.0], [1.0])
    else:
        wash = shifted(wash, rng.uniform(600.0, 2400.0))

    kitchen = [
        series("sensor.kitchen_fridge_power", "Fridge", kind.DRONE, fridge(rng)),
        series(
            "sensor.kitchen_kettle_power",
            "Kettle",
            kind.DRONE,
            (
                [0.0, hours("07:12") + late, hours("07:15") + late, hours("12:20"), hours("12:23")],
                [0.0, 2150.0, 0.0, 2150.0, 0.0],
            ),
        ),
        series(
            "binary_sensor.kitchen_door",
            "Kitchen door",
            kind.PLUCK,
            intervals_to_events([(start, start + length) for start, length in door_times]),
            "door",
        ),
        series(
            "sensor.kitchen_oven_power",
            "Oven",
            kind.DRONE,
            heating(rng, hours("17:55"), hours("18:50"), 2300, (240, 420)),
        ),
        series("sensor.washing_machine_power", "Washing machine", kind.DRUM, wash),
        series("sensor.dishwasher_power", "Dishwasher", kind.DISHWASHER, dishwasher(rng)),
        series(
            "binary_sensor.kitchen_motion",
            "Kitchen motion",
            kind.PLUCK,
            intervals_to_events(
                bursts(rng, hours("06:50") + late, hours("07:40") + late, (15, 50), (40, 150))
                + bursts(rng, hours("12:05"), hours("12:45"), (15, 50), (40, 150))
                + bursts(rng, hours("17:25"), hours("19:30"), (15, 60), (50, 200))
            ),
            "motion",
        ),
        series(
            "light.kitchen_ceiling",
            "Kitchen ceiling light",
            kind.CHORD,
            intervals_to_events([later("06:45", "08:10"), (hours("17:20"), hours("22:30"))]),
        ),
        series(
            "light.kitchen_counter",
            "Kitchen counter light",
            kind.CHORD,
            intervals_to_events([later("06:55", "07:42"), (hours("17:40"), hours("19:35"))]),
        ),
        series(
            "sensor.kitchen_temperature",
            "Kitchen temperature",
            kind.TEMPERATURE,
            curve(
                warm(
                    [
                        ("00:00", 20.0),
                        ("05:00", 19.6),
                        ("08:00", 20.4),
                        ("11:00", 21.6),
                        ("15:00", 22.6),
                        ("18:00", 23.0),
                        ("21:00", 21.0),
                        ("23:59", 19.0),
                    ]
                ),
                rng,
                0.04,
                600,
            ),
        ),
        series(
            "sensor.kitchen_humidity",
            "Kitchen humidity",
            kind.HUMIDITY,
            curve(
                [
                    ("00:00", 48),
                    ("07:00", 50),
                    ("07:30", 58),
                    ("10:30", 62),
                    ("12:30", 55),
                    ("18:30", 66),
                    ("23:59", 50),
                ],
                rng,
                0.5,
                600,
            ),
        ),
    ]
    living = [
        series(
            "sensor.living_room_tv_power",
            "TV",
            kind.DRONE,
            ([0.0, hours("19:45"), hours("22:40")], [0.5, 96.0, 0.5]),
        ),
        series(
            "binary_sensor.living_room_motion",
            "Living room motion",
            kind.PLUCK,
            intervals_to_events(
                bursts(rng, hours("06:30") + late, hours("08:00") + late, (20, 80), (90, 300))
                + bursts(rng, hours("16:50"), hours("22:40"), (20, 90), (90, 400))
            ),
            "motion",
        ),
        series(
            "binary_sensor.living_room_window",
            "Living room window",
            kind.PLUCK,
            intervals_to_events(
                [(hours("08:15"), hours("08:31")), (hours("13:05"), hours("13:22"))]
            ),
            "window",
        ),
        series(
            "light.living_room_lamp",
            "Living room lamp",
            kind.CHORD,
            intervals_to_events([later("06:35", "07:50"), (hours("17:30"), hours("23:05"))]),
        ),
        series(
            "switch.living_room_floor_light",
            "Floor light",
            kind.CHORD,
            intervals_to_events([(hours("18:05"), hours("22:10"))]),
        ),
        series(
            "sensor.living_room_temperature",
            "Living room temperature",
            kind.TEMPERATURE,
            curve(
                warm(
                    [
                        ("00:00", 20.5),
                        ("06:00", 19.8),
                        ("10:00", 21.2),
                        ("14:00", 22.4),
                        ("20:00", 23.4),
                        ("23:59", 20.0),
                    ]
                ),
                rng,
                0.04,
                600,
            ),
        ),
        series(
            "sensor.living_room_humidity",
            "Living room humidity",
            kind.HUMIDITY,
            curve([("00:00", 44), ("12:00", 41), ("20:00", 47), ("23:59", 45)], rng, 0.4, 600),
        ),
    ]
    sun_series = [series("sensor.sun_solar_elevation", "Sun elevation", kind.SUN, sun())]
    return {"kitchen": kitchen, "living": living, "sun": sun_series}


def shifted(
    data: tuple[list[float], list[float]], seconds: float
) -> tuple[list[float], list[float]]:
    """Return a series moved later in the day, with its first value kept from midnight."""
    times, values = data
    return [0.0, *[time + seconds for time in times[1:]]], values


def write_wav(path: Path, audio: np.ndarray) -> None:
    """Write float stereo audio as a 16 bit wav."""
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(pcm.tobytes())


def encode(wav: Path, mp3: Path) -> None:
    """Encode a wav as a 192 kbit/s mp3 with ffmpeg."""
    subprocess.run(  # noqa: S603
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(mp3)],  # noqa: S607
        check=True,
    )


def render_to(out: Path, name: str, audio: np.ndarray) -> Path:
    """Write float stereo audio as `<name>.mp3` in a folder, through a temporary wav."""
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{name}.wav"
    write_wav(wav, audio)
    mp3 = out / f"{name}.mp3"
    encode(wav, mp3)
    wav.unlink()
    return mp3


class Demo:
    """The fortnight, and what the provider would work out from it for one of its days."""

    def __init__(self, house: str = "motion") -> None:
        """
        Build the fortnight.

        :param house: `motion` for the invented house with motion sensors, `sparse` for the one without
            (`sparse_house`).
        """
        from entropy_fm import (  # noqa: PLC0415
            baseline,
            palette,
            quality,
            scape,
            styles,
            surprise,
            synth,
            timeline,
        )

        self.baseline, self.palette, self.quality = baseline, palette, quality
        self.scape, self.styles, self.surprise, self.synth, self.timeline = (
            scape,
            styles,
            surprise,
            synth,
            timeline,
        )
        self.house = house
        if house == "sparse":
            import sparse_house  # noqa: PLC0415

            self.days = [sparse_house.build_day(timeline, index) for index in range(DAYS)]
        else:
            self.days = [build_day(timeline, index) for index in range(DAYS)]

    def day_of(self, index: int) -> date:
        """Return the date of a day of the fortnight."""
        return FIRST_DAY + timedelta(days=index)

    def baselines(self, index: int, areas: tuple[str, ...]) -> dict[str, object]:
        """Return the baseline of the entities of some areas from the week before a day."""
        found: dict[str, object] = {}
        for area in areas:
            before = [self.days[i][area] for i in range(max(index - 7, 0), index)]
            found.update(self.baseline.build_baselines(before))
        return found

    def make(  # noqa: PLR0913, PLR0917
        self,
        series: list[object],
        day: date,
        style: str,
        area_key: str,
        baselines: dict[str, object] | None = None,
        tier: object = None,
        length_mode: str = "full",
    ) -> tuple[object, object]:
        """Return the score and the plan of the soundscape of some series on a day, as the provider makes them."""
        chosen = self.styles.get_style(style)
        home = self.synth.Scale(2, "dorian")
        found = baselines or {}
        try:
            from entropy_fm.references import rescale  # noqa: PLC0415
        except ImportError:  # an older engine measures every day by itself
            rescale_series: Any = None
        else:
            rescale_series = rescale
        if rescale_series is not None:
            series, _ = rescale_series(series, found)
        traits = self.palette.measure_traits(series, found, DAY)
        palette = self.palette.derive_palette(chosen, day, home, traits)
        anomalies = [
            (a.start, a.score, a.entity_id)
            for a in self.surprise.Surprise(series, found).anomalies
            if a.score >= 0.5
        ]
        analysis = self.scape.analysis_of(series, DAY, anomalies)
        return self.scape.scape_score(
            series,
            analysis,
            style,
            f"{area_key}|{day.isoformat()}|_scape|{style}",
            home,
            day=day,
            palette=palette,
            quality=tier or self.quality.STANDARD,
            area_key=area_key,
            length_mode=length_mode,
        )

    def score(
        self,
        index: int,
        areas: tuple[str, ...],
        style: str,
        tier: object = None,
        length_mode: str = "full",
    ) -> tuple[object, object]:
        """Return the score and the plan of a day of the fortnight, for some areas, in a style."""
        series = [item for area in areas for item in self.days[index][area]] + self.days[index][
            "sun"
        ]
        return self.make(
            series,
            self.day_of(index),
            style,
            "+".join(areas),
            self.baselines(index, areas),
            tier,
            length_mode,
        )
