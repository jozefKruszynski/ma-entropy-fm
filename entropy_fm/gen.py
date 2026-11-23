"""
The small generative tools that the four styles share.

Weighted choices, Markov chains, random walks and the curves of anomaly events.

The generator is seeded by the plan from the room, the day and the style, so a piece is the same every time it
is made.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .score import CURVE_RATE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .plan import Plan

# an anomaly lifts the strength that its curves are drawn with by this much, up to 1
EVENT_STRENGTH_LIFT = 0.3
# the rise of a dip, in seconds
DIP_RISE_SECONDS = 0.6
# the walk of a block is made at this step, in seconds, and drawn out to every value
BLOCK_WALK_STEP_SECONDS = 2.0


def weighted[T](rng: np.random.Generator, items: Sequence[T], weights: Sequence[float]) -> T:
    """Pick an item with probabilities in proportion to the weights."""
    w = np.asarray(weights, dtype=np.float64)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def markov[T](rng: np.random.Generator, state: T, table: dict[T, Sequence[tuple[T, float]]]) -> T:
    """Move from a state to the next one, by the weights of a table (a state without a row stays)."""
    row = table.get(state)
    if not row:
        return state
    return weighted(rng, [item for item, _ in row], [w for _, w in row])


def swung(index: int, step: float, swing: float) -> float:
    """Return the time of a sixteenth note of a bar: the odd ones are late by `swing` of a step, the even ones are on the grid."""
    return (index - index % 2) * step + (index % 2) * step * (1.0 + swing)


def walk(rng: np.random.Generator, count: int, tau: float, amplitude: float) -> np.ndarray:
    """
    Return a slow random walk with a typical size of `amplitude` and a memory of `tau` values.

    :param rng: The generator.
    :param count: The number of values.
    :param tau: The memory in values; the walk changes over about this many values.
    :param amplitude: The standard deviation of the walk.
    """
    pull = 1.0 / max(tau, 1.0)
    out = np.empty(count)
    x = 0.0
    kick = rng.standard_normal(count) * amplitude * np.sqrt(2.0 * pull)
    for i in range(count):
        x += -pull * x + kick[i]
        out[i] = x
    return out


def block_walk(
    rng: np.random.Generator, seconds: int, tau_s: float, amplitude: float
) -> np.ndarray:
    """Return a slow random walk with CURVE_RATE values a second."""
    coarse = walk(
        rng,
        max(int(seconds / BLOCK_WALK_STEP_SECONDS) + 2, 2),
        tau_s / BLOCK_WALK_STEP_SECONDS,
        amplitude,
    )
    return np.interp(
        np.arange(seconds * CURVE_RATE) / (BLOCK_WALK_STEP_SECONDS * CURVE_RATE),
        np.arange(len(coarse)),
        coarse,
    )


def ramp(count: int, start: int, end: int, up: int, down: int) -> np.ndarray:
    """Return a bump: 0 before `start`, rising over `up` values, 1 until `end`, falling over `down` values."""
    x = np.arange(count)
    rise = np.clip((x - start) / max(up, 1), 0.0, 1.0)
    fall = np.clip((end + down - x) / max(down, 1), 0.0, 1.0)
    smooth = np.minimum(rise, fall)
    return smooth * smooth * (3.0 - 2.0 * smooth)


def dips(plan: Plan, depth: float = 0.85, hold: float = 0.7, back: float = 3.5) -> np.ndarray:
    """
    Return a multiplier of a cutoff that closes like a tape slowing down and opens again, at each anomaly.

    It is a filter, never a pitch: 1 when nothing happens, down to `1 - depth` at the dip.

    :param plan: The plan, whose events are the anomalies.
    :param depth: How far the cutoff closes at the deepest dip, a share of it.
    :param hold: Seconds that the dip stays closed.
    :param back: Seconds that it takes to open again.
    """
    out = np.ones(plan.count)
    for second, strength in plan.events:
        start = int(second * CURVE_RATE)
        shape = ramp(
            plan.count,
            start,
            start + int(hold * CURVE_RATE),
            int(DIP_RISE_SECONDS * CURVE_RATE),
            int(back * CURVE_RATE),
        )
        out = out * (1.0 - depth * min(strength + EVENT_STRENGTH_LIFT, 1.0) * shape)
    return out


def swells(plan: Plan, rise: float = 5.0, fall: float = 10.0, hold: float = 2.0) -> np.ndarray:
    """Return a curve that swells to 1 at each anomaly and falls back to 0."""
    out = np.zeros(plan.count)
    for second, strength in plan.events:
        start = int((second - rise / 2.0) * CURVE_RATE)
        shape = ramp(
            plan.count,
            start,
            start + int(hold * CURVE_RATE),
            int(rise * CURVE_RATE),
            int(fall * CURVE_RATE),
        )
        out = np.maximum(out, shape * min(strength + EVENT_STRENGTH_LIFT, 1.0))
    return out
