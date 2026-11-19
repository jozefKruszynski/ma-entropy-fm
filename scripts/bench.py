"""
Measure what the engine costs: CPU milliseconds per second of audio, on one core.

    uv run python scripts/bench.py [--seconds N] [--house] [style ...]

For each style and each quality tier it renders the soundscape of the anomaly day of the demo house (the kitchen, or with
`--house` the whole house: kitchen and living room) from its start, with the caches of notes empty, and prints the CPU
time per second of audio. That is the cost of a track that is played from the start to the end: it pays for making every
note once, which is most of what a soundscape costs, since a pad is made once and looped. `--seconds` renders only
the first seconds of the piece, which is quicker but leaves out the busy part of the day. The math libraries are held to one thread, as on a small
machine. The budgets of the tiers are in `BUDGETS`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).parent))

import demo_render as demo_module  # noqa: E402

# milliseconds of CPU per second of audio, for the whole house, on the machine of the reference runs
BUDGETS = {"light": 20.0, "standard": 60.0, "rich": 120.0}


def measure(
    demo: demo_module.Demo, style: str, areas: tuple[str, ...], tier: object, seconds: int | None
) -> float:
    """Return the CPU milliseconds per second of audio of one style, area set and tier, cold."""
    from entropy_fm.glitch_voices import STACKS  # noqa: PLC0415
    from entropy_fm.render import render_blocks  # noqa: PLC0415
    from entropy_fm.voices import NOTES  # noqa: PLC0415

    NOTES.clear()
    STACKS.clear()
    score, _ = demo.score(demo_module.ANOMALY_DAY, areas, style, tier)
    stop = score.length if seconds is None else min(seconds, score.length)
    start = time.process_time()
    for _ in render_blocks(score, tier, "bench", 0, None, stop):  # type: ignore[arg-type]
        pass
    return (time.process_time() - start) / stop * 1000.0


def main() -> None:
    """Print the table."""
    arguments = sys.argv[1:]
    seconds = int(arguments[arguments.index("--seconds") + 1]) if "--seconds" in arguments else None
    house = "--house" in arguments
    demo = demo_module.Demo()
    wanted = [a for a in arguments if a in demo.styles.STYLES] or list(demo.styles.STYLES)
    areas = ("kitchen", "living") if house else ("kitchen",)
    tiers = [demo.quality.LIGHT, demo.quality.STANDARD, demo.quality.RICH]
    print(f"CPU ms per second of audio, {'whole house' if house else 'kitchen'}, cold, one core")
    print("| style | " + " | ".join(tier.title for tier in tiers) + " |")
    print("|---|" + "---|" * len(tiers))
    worst = {tier.key: 0.0 for tier in tiers}
    for style in wanted:
        costs = []
        for tier in tiers:
            cost = measure(demo, style, areas, tier, seconds)
            worst[tier.key] = max(worst[tier.key], cost)
            costs.append(cost)
        print(f"| {style} | " + " | ".join(f"{c:.1f}" for c in costs) + " |", flush=True)
    print("| worst | " + " | ".join(f"{worst[t.key]:.1f}" for t in tiers) + " |")
    print("| budget | " + " | ".join(f"{BUDGETS[t.key]:.0f}" for t in tiers) + " |")


if __name__ == "__main__":
    main()
