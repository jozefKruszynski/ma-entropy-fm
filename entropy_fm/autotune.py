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
from .quality import ORDER, Quality
from .render import render_blocks
from .scape import scape_score
from .synth import Scale
from .timeline import EntityInfo, Series, VoiceKind, build_series

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DAY = 86400.0
BENCH_SECONDS = 6
# a piece makes notes as it goes, which the measure of a short stretch with the notes made leaves out
NOTE_FACTOR = 1.5
# the styles from the dearest to the cheapest, by the CPU that scripts/bench.py measures at every tier
COST_ORDER = ("glitch", "berlin_school", "ritual", "lofi", "ambient", "minimal")


def heaviest_style(offered: Iterable[str]) -> str:
    """
    Return the dearest of the offered styles to render, which is the one Auto measures.

    :param offered: The keys of the styles the library offers.
    """
    wanted = set(offered)
    return next((key for key in COST_ORDER if key in wanted), COST_ORDER[0])


def mini_house() -> list[Series]:
    """Return a small invented house: a fridge, a kettle, motion, a door, lights, the climate."""
    rng = np.random.default_rng(7)

    def make(
        entity_id: str, kind: VoiceKind, times: list[float], values: list[float], cls: str = ""
    ) -> Series:
        info = EntityInfo(entity_id, entity_id, None, kind, device_class=cls)
        return build_series(info, times, values)

    fridge_t, fridge_v = [], []
    for start in np.arange(0.0, DAY, 2400.0):
        fridge_t += [float(start), float(start) + 840.0]
        fridge_v += [90.0 + float(rng.normal(0, 3)), 2.0]
    moments = np.arange(0.0, DAY, 400.0)
    flips = [float(m) for m in moments[(moments > 25000) & (moments < 30000)]]
    motion_t = [0.0, *flips]
    motion_v = [float(i % 2) for i in range(len(motion_t))]
    temp_t = [float(t) for t in np.arange(0.0, DAY, 600.0)]
    return [
        make("sensor.fridge_power", VoiceKind.DRONE, fridge_t, fridge_v),
        make("sensor.kettle_power", VoiceKind.DRONE, [0.0, 26000.0, 26200.0], [0.0, 2100.0, 0.0]),
        make(
            "binary_sensor.door", VoiceKind.PLUCK, [0.0, 25500.0, 25560.0], [0.0, 1.0, 0.0], "door"
        ),
        make("binary_sensor.motion", VoiceKind.PLUCK, motion_t, motion_v, "motion"),
        make("light.ceiling", VoiceKind.CHORD, [0.0, 25000.0, 30000.0], [0.0, 1.0, 0.0]),
        make(
            "sensor.temperature",
            VoiceKind.TEMPERATURE,
            temp_t,
            [20.0 + 2.0 * float(np.sin(t / 9000.0)) for t in temp_t],
        ),
        make("sensor.humidity", VoiceKind.HUMIDITY, temp_t, [45.0 + 0.002 * t for t in temp_t]),
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
        series, analyze_day(series), style, "bench", Scale(2, "dorian"), quality=tier
    )
    first = int(plan.length * 0.4)
    spent: list[float] = []
    for _ in range(2):
        spent = []
        last = time.thread_time()
        for _block in render_blocks(score, tier, "bench", first, end_s=first + seconds + 1):
            now = time.thread_time()
            spent.append(now - last)
            last = now
    counted = sorted(spent[1:] or spent)
    return counted[len(counted) // 2] * 1000.0 * NOTE_FACTOR


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
