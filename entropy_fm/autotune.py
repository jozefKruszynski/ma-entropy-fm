"""
The Auto setting of the render quality: a short benchmark that picks the richest tier that fits.

At start-up the provider renders a few seconds of a small invented house in each tier and measures
the CPU time that one second of audio takes on this machine, on one thread. Auto picks the richest
tier that would use less than a share of one core in real time (see `quality.choose_tier`). The
benchmark plays the dearest style that is offered, so a tier that Auto picks fits every style that
the library lists, not only the cheap ones. The result is kept in the cache of the server, so the
benchmark runs once per machine, engine version and heaviest style.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .day_analysis import analyze_day
from .quality import MS_PER_SECOND, ORDER, Quality
from .render import render_blocks
from .scape import scape_score
from .synth import DEFAULT_MODE, DEFAULT_ROOT, Scale
from .timeline import DAY_SECONDS, EntityInfo, Series, VoiceKind, build_series

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

BENCH_SECONDS = 6
BENCH_SEED = "bench"
# the benchmark begins this far into the piece, which is its busy part
BENCH_START_SHARE = 0.4
# the invented house: its random seed, the fridge's cycle and run (seconds) and level, its noise, and the kettle
HOUSE_SEED = 7
FRIDGE_CYCLE_SECONDS = 2400.0
FRIDGE_RUN_SECONDS = 840.0
FRIDGE_RUNNING_WATTS = 90.0
FRIDGE_IDLE_WATTS = 2.0
FRIDGE_NOISE_WATTS = 3.0
# the motion sensor flips every this many seconds, between these two times of the day
MOTION_STEP_SECONDS = 400.0
MOTION_FROM_SECONDS = 25000
MOTION_UNTIL_SECONDS = 30000
CLIMATE_STEP_SECONDS = 600.0
# the temperature swings around its mean, by this much, with this period over 2 pi in seconds; the humidity climbs by this much a second
TEMPERATURE_MEAN = 20.0
TEMPERATURE_SWING = 2.0
TEMPERATURE_PERIOD = 9000.0
HUMIDITY_START = 45.0
HUMIDITY_RISE = 0.002
# a piece makes notes as it goes, which the measure of a short stretch with the notes made leaves out
NOTE_FACTOR = 1.5
# the styles from the dearest to the cheapest, by the CPU that scripts/bench.py measures at every tier
COST_ORDER = ("glitch", "berlin_school", "ritual", "lofi", "ambient")


def heaviest_style(offered: Iterable[str]) -> str:
    """
    Return the dearest of the offered styles to render, which is the one Auto measures.

    :param offered: The keys of the styles the library offers.
    """
    wanted = set(offered)
    return next((key for key in COST_ORDER if key in wanted), COST_ORDER[0])


# the kettle boils for 200 seconds at 2100 W, the door is open for a minute, the light is on for five thousand seconds
KETTLE_TIMES = (0.0, 26000.0, 26200.0)
KETTLE_WATTS = (0.0, 2100.0, 0.0)
DOOR_TIMES = (0.0, 25500.0, 25560.0)
DOOR_OPEN = (0.0, 1.0, 0.0)
LIGHT_TIMES = (0.0, 25000.0, 30000.0)
LIGHT_ON = (0.0, 1.0, 0.0)


def mini_house() -> list[Series]:
    """Return a small invented house: a fridge, a kettle, motion, a door, lights, the climate."""
    rng = np.random.default_rng(HOUSE_SEED)

    def make(
        entity_id: str, kind: VoiceKind, times: list[float], values: list[float], cls: str = ""
    ) -> Series:
        info = EntityInfo(entity_id, entity_id, None, kind, device_class=cls)
        return build_series(info, times, values)

    fridge_t, fridge_v = [], []
    for start in np.arange(0.0, DAY_SECONDS, FRIDGE_CYCLE_SECONDS):
        fridge_t += [float(start), float(start) + FRIDGE_RUN_SECONDS]
        fridge_v += [
            FRIDGE_RUNNING_WATTS + float(rng.normal(0, FRIDGE_NOISE_WATTS)),
            FRIDGE_IDLE_WATTS,
        ]
    moments = np.arange(0.0, DAY_SECONDS, MOTION_STEP_SECONDS)
    flips = [
        float(m)
        for m in moments[(moments > MOTION_FROM_SECONDS) & (moments < MOTION_UNTIL_SECONDS)]
    ]
    motion_t = [0.0, *flips]
    motion_v = [float(i % 2) for i in range(len(motion_t))]
    temp_t = [float(t) for t in np.arange(0.0, DAY_SECONDS, CLIMATE_STEP_SECONDS)]
    return [
        make("sensor.fridge_power", VoiceKind.DRONE, fridge_t, fridge_v),
        make(
            "sensor.kettle_power",
            VoiceKind.DRONE,
            list(KETTLE_TIMES),
            list(KETTLE_WATTS),
        ),
        make(
            "binary_sensor.door",
            VoiceKind.PLUCK,
            list(DOOR_TIMES),
            list(DOOR_OPEN),
            "door",
        ),
        make("binary_sensor.motion", VoiceKind.PLUCK, motion_t, motion_v, "motion"),
        make(
            "light.ceiling",
            VoiceKind.CHORD,
            list(LIGHT_TIMES),
            list(LIGHT_ON),
        ),
        make(
            "sensor.temperature",
            VoiceKind.TEMPERATURE,
            temp_t,
            [
                TEMPERATURE_MEAN + TEMPERATURE_SWING * float(np.sin(t / TEMPERATURE_PERIOD))
                for t in temp_t
            ],
        ),
        make(
            "sensor.humidity",
            VoiceKind.HUMIDITY,
            temp_t,
            [HUMIDITY_START + HUMIDITY_RISE * t for t in temp_t],
        ),
    ]


def measure(tier: Quality, seconds: int = BENCH_SECONDS, style: str = COST_ORDER[0]) -> float:
    """
    Return the CPU milliseconds that one second of audio takes in a tier, on this thread.

    The soundscape of the invented house is rendered from its busy part, twice: the first pass makes the notes and
    the filters and is thrown away, the second is what a piece costs once its notes are made. The block that comes
    first pays the setup and the seconds that a seek renders before the start, so it is not counted. The median of
    the rest (a second that makes a long note is dearer than most, and a piece makes those as it goes) is scaled by
    `NOTE_FACTOR`, which stands for the notes that a real piece still has to make.

    :param tier: The tier.
    :param seconds: How many seconds of the piece to render and count (one more is rendered and not counted).
    :param style: The style to render, the dearest that is offered by default.
    """
    series = mini_house()
    score, plan = scape_score(
        series,
        analyze_day(series),
        style,
        BENCH_SEED,
        Scale(DEFAULT_ROOT, DEFAULT_MODE),
        quality=tier,
    )
    first = int(plan.length * BENCH_START_SHARE)
    spent: list[float] = []
    for _ in range(2):
        spent = []
        last = time.thread_time()
        for _block in render_blocks(score, tier, BENCH_SEED, first, end_s=first + seconds + 1):
            now = time.thread_time()
            spent.append(now - last)
            last = now
    counted = sorted(spent[1:] or spent)
    return counted[len(counted) // 2] * MS_PER_SECOND * NOTE_FACTOR


def benchmark(
    tiers: Sequence[Quality] = ORDER, style: str = COST_ORDER[0], seconds: int = BENCH_SECONDS
) -> dict[str, float]:
    """
    Measure every tier with the dearest style that is offered.

    :param tiers: The tiers.
    :param style: The style to render.
    :param seconds: How many seconds to count in each tier.
    """
    return {tier.key: measure(tier, seconds, style) for tier in tiers}
