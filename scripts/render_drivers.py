"""
Render three minutes of each style on a room of a demo house: the night, a busy stretch of the day and the evening.

    uv run python scripts/render_drivers.py OUT_FOLDER NAME [--engine PATH] [--house sparse|motion] [--room living]
        [--hours hours.json | --write-hours hours.json] [style ...]

It writes `<OUT_FOLDER>/<style>_<NAME>.mp3`. With `--engine` the engine of another checkout of the repository (a git
worktree of an older commit) plays the same house, so that two engines can be compared on the same data.
`--write-hours` saves where the three excerpts of each style are in the day (the hours of the clock), and `--hours`
reads them, so that both engines play the same moments of the day and not each its own best stretch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import demo_render

if "--engine" in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index("--engine") + 1]).resolve()))

import soundscapes


def option(name: str, default: str | None = None) -> str | None:
    """Return the value that follows an option, or the default."""
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    """Render the excerpts."""
    out, name = Path(sys.argv[1]), sys.argv[2]
    house = option("--house", "sparse") or "sparse"
    room = option("--room", "living") or "living"
    read, write = option("--hours"), option("--write-hours")
    taken = {
        sys.argv[i + 1]
        for i, a in enumerate(sys.argv)
        if a in {"--engine", "--house", "--room", "--hours", "--write-hours"}
    }
    styles = [a for a in sys.argv[3:] if not a.startswith("--") and a not in taken]
    demo = demo_render.Demo(house)
    from entropy_fm.quality import STANDARD  # noqa: PLC0415
    from entropy_fm.render import render_all  # noqa: PLC0415

    saved: dict[str, list[float]] = json.loads(Path(read).read_text()) if read else {}
    for style in styles or list(demo.styles.STYLES):
        score: Any
        plan: Any
        score, plan = demo.score(demo_render.DAYS - 1, (room,), style, STANDARD)
        if style in saved:
            starts = [
                int(
                    min(
                        max(plan.piece_second_of_hour(hour) - soundscapes.EXCERPT_SECONDS / 2, 2),
                        plan.length - 70,
                    )
                )
                for hour in saved[style]
            ]
        else:
            starts = [start for _, start in soundscapes.pick_windows(plan)]
        saved[style] = [
            round(plan.second_of_day(start + soundscapes.EXCERPT_SECONDS / 2) / 3600.0, 2)
            for start in starts
        ]
        pieces = [
            render_all(
                score,
                STANDARD,
                "drivers",
                None,
                start_s=start,
                end_s=start + soundscapes.EXCERPT_SECONDS,
            )
            for start in starts
        ]
        audio = soundscapes.crossfade(pieces)[: 180 * soundscapes.RATE]
        path = demo_render.render_to(out, f"{style}_{name}", audio)
        print(path, saved[style], flush=True)
    if write:
        Path(write).write_text(json.dumps(saved))


if __name__ == "__main__":
    main()
