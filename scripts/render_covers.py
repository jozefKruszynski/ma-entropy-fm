"""
Render sample procedural covers of the invented fortnight: every style on two days.

    uv run python scripts/render_covers.py [output folder]

Writes each cover as the 1024 pixel JPEG that the provider serves, and prints what a cover costs.
"""

import sys
import time
from datetime import timedelta
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import demo_render as demo_module


def main() -> None:
    """Write the covers."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/out/covers")
    out.mkdir(parents=True, exist_ok=True)
    demo = demo_module.Demo()
    covers = import_module("entropy_fm.covers")
    palette = demo.palette
    times: list[float] = []
    for style in demo.styles.STYLES:
        for index in (6, demo_module.ANOMALY_DAY):
            series = demo.days[index]["kitchen"]
            profile = covers.profile_of(series)
            day = demo_module.FIRST_DAY + timedelta(days=index)
            traits = palette.DayTraits(energy=profile.intensity)
            chosen = demo.styles.get_style(style)
            derived = palette.derive_palette(chosen, day, demo.synth.Scale(2, "dorian"), traits)
            started = time.process_time()
            data = covers.procedural_cover(
                profile,
                style,
                derived.root,
                derived.mode,
                f"kitchen|{day}|{style}",
                f"Kitchen, {day.strftime('%a %-d %b')}",
            )
            times.append(time.process_time() - started)
            stem = f"cover_{style}_{day.isoformat()}"
            (out / f"{stem}.jpg").write_bytes(data)
            print(stem, len(data), "bytes")
    print(
        f"CPU time per cover: mean {1000 * sum(times) / len(times):.0f} ms, worst {1000 * max(times):.0f} ms"
    )


if __name__ == "__main__":
    main()
