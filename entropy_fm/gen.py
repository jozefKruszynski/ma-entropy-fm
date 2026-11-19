"""
The small generative tools that the four styles share.

Weighted choices, Markov chains, Euclidean rhythms, random walks and the curves of anomaly events. Everything
takes a numpy generator.

Everything takes a numpy generator that the plan seeds from the room, the day and the style, so a piece is the same
every time it is made.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from .score import CR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .plan import Plan

T = TypeVar("T")


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


def euclid(hits: int, steps: int, rotate: int = 0) -> list[bool]:
    """Return the Euclidean rhythm of `hits` onsets in `steps` steps, turned by `rotate` steps."""
    pattern = [((i * hits) % steps) < hits for i in range(steps)]
    return pattern[rotate % steps :] + pattern[: rotate % steps]


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
    """Return a slow random walk with CR values a second."""
    coarse = walk(rng, max(int(seconds / 2.0) + 2, 2), tau_s / 2.0, amplitude)
    return np.interp(np.arange(seconds * CR) / (2.0 * CR), np.arange(len(coarse)), coarse)


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
    """
    out = np.ones(plan.count)
    for second, strength in plan.events:
        start = int(second * CR)
        shape = ramp(plan.count, start, start + int(hold * CR), int(0.6 * CR), int(back * CR))
        out = out * (1.0 - depth * min(strength + 0.3, 1.0) * shape)
    return out


def swells(plan: Plan, rise: float = 5.0, fall: float = 10.0, hold: float = 2.0) -> np.ndarray:
    """Return a curve that swells to 1 at each anomaly and falls back to 0."""
    out = np.zeros(plan.count)
    for second, strength in plan.events:
        start = int((second - rise / 2.0) * CR)
        shape = ramp(plan.count, start, start + int(hold * CR), int(rise * CR), int(fall * CR))
        out = np.maximum(out, shape * min(strength + 0.3, 1.0))
    return out


def event_kinds(plan: Plan, kinds: Sequence[str], weights: Sequence[float]) -> list[str]:
    """Choose what each anomaly of the piece does, from a style's list of textural events."""
    rng = plan.rng("event kinds")
    return [weighted(rng, kinds, weights) for _ in plan.events]


def at_times(plan: Plan, curve: np.ndarray, times: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Return the values of a curve at several seconds of the piece."""
    return np.interp(times * CR, np.arange(len(curve)), curve)


def vel_scale(value: float, low: float, high: float) -> float:
    """Return a velocity between `low` and `high` for a value from 0 to 1."""
    return float(low + (high - low) * min(max(value, 0.0), 1.0))
