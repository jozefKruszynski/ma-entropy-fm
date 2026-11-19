"""Helpers of the soundscape tests: the demo house, its scores, and short renders."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from entropy_fm.quality import LIGHT, STANDARD, Quality
from entropy_fm.render import render_all

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from typing import TYPE_CHECKING

import demo_render

if TYPE_CHECKING:
    import numpy as np


@lru_cache(maxsize=1)
def the_demo() -> demo_render.Demo:
    """Return the invented fortnight."""
    return demo_render.Demo()


@lru_cache(maxsize=64)
def demo(
    style: str,
    index: int = demo_render.ANOMALY_DAY,
    areas: tuple[str, ...] = ("kitchen",),
    tier: Quality = STANDARD,
    length_mode: str = "full",
) -> tuple:
    """Return the score and the plan of a day of the fortnight in a style, full or condensed."""
    return the_demo().score(index, areas, style, tier, length_mode)


def window(
    score: object,
    plan: object,
    seconds: int = 40,
    tier: Quality = STANDARD,
    stats: dict[str, float] | None = None,
) -> np.ndarray:
    """Render `seconds` of the active daytime of a piece."""
    from soundscapes import pick_windows  # noqa: PLC0415

    start = pick_windows(plan, seconds)[1][1]
    return render_all(score, tier, "test", stats, start_s=start, end_s=start + seconds)  # type: ignore[arg-type]


__all__ = ["LIGHT", "STANDARD", "demo", "the_demo", "window"]
