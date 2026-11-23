"""Helpers of the soundscape tests: the demo house, its scores, and short renders."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from entropy_fm.plan import Plan
from entropy_fm.quality import LIGHT, STANDARD, Quality
from entropy_fm.render import render_all
from entropy_fm.score import Score

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from typing import TYPE_CHECKING

import demo_render

if TYPE_CHECKING:
    import numpy as np


@lru_cache(maxsize=1)
def the_demo() -> demo_render.Demo:
    """Return the invented fortnight."""
    return demo_render.Demo()


@lru_cache(maxsize=1)
def the_sparse() -> demo_render.Demo:
    """Return the invented fortnight of the sparse house, the one without motion sensors."""
    return demo_render.Demo("sparse")


def layers_playing(score: Score) -> list[int]:
    """Return, for each minute of a piece without its first and last, how many layers have a note sounding in it."""
    length = int(score.length)
    minutes = max(length // 60, 1)
    found: list[set[str]] = [set() for _ in range(minutes)]
    for note in score.notes:
        layer = score.layers.get(note.layer)
        first = int(note.t // 60)
        last = int(min(note.t + note.dur, length - 1) // 60)
        for minute in range(first, min(last, minutes - 1) + 1):
            if layer is not None and layer.presence is not None:
                at = int(min(max((minute * 60 + 30) * 10, 0), len(layer.presence) - 1))
                if layer.presence[at] < 0.12:
                    continue
            found[minute].add(note.layer)
    counts = [len(item) for item in found]
    return counts[1:-1] if len(counts) > 4 else counts


@lru_cache(maxsize=64)
def demo(
    style: str,
    index: int = demo_render.ANOMALY_DAY,
    areas: tuple[str, ...] = ("kitchen",),
    tier: Quality = STANDARD,
) -> tuple[Score, Plan]:
    """Return the score and the plan of a day of the fortnight in a style."""
    return the_demo().score(index, areas, style, tier)


def window(
    score: Score,
    plan: Plan,
    seconds: int = 40,
    tier: Quality = STANDARD,
    stats: dict[str, float] | None = None,
) -> np.ndarray:
    """Render `seconds` of the active daytime of a piece."""
    from soundscapes import pick_windows  # noqa: PLC0415

    start = pick_windows(plan, seconds)[1][1]
    return render_all(score, tier, "test", stats, start_s=start, end_s=start + seconds)


__all__ = ["LIGHT", "STANDARD", "demo", "layers_playing", "the_demo", "the_sparse", "window"]
