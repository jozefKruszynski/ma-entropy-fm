"""
Render sample procedural covers of the invented fortnight: every style at three energies.

    uv run python scripts/render_covers.py [output folder]

Writes each cover as the 1024 pixel JPEG that the provider serves, a contact sheet (`sheet.png`, a row
for every style and a column for every energy), and prints what a cover costs. The same day is drawn
at every energy, so that what the energy changes is all that differs.
"""

import sys
import time
from datetime import timedelta
from importlib import import_module
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import demo_render as demo_module

ENERGIES = (0.1, 0.5, 0.95)
DAY_INDEX = 6
TILE = 256


def main() -> None:
    """Write the covers and the contact sheet."""
    from PIL import Image  # noqa: PLC0415

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/out/covers")
    out.mkdir(parents=True, exist_ok=True)
    demo = demo_module.Demo()
    covers = import_module("entropy_fm.covers")
    palette = demo.palette
    times: list[float] = []
    styles = list(demo.styles.STYLES)
    sheet = Image.new("RGB", (TILE * len(ENERGIES), TILE * len(styles)))
    day = demo_module.FIRST_DAY + timedelta(days=DAY_INDEX)
    profile = covers.profile_of(demo.days[DAY_INDEX]["kitchen"])
    for row, style in enumerate(styles):
        for column, energy in enumerate(ENERGIES):
            mood = covers.CoverMood(energy)
            # the traits that the provider would measure, here set by hand
            traits = palette.DayTraits(energy=energy)
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
                mood,
            )
            times.append(time.process_time() - started)
            stem = f"cover_{style}_{day.isoformat()}_e{round(100 * energy):02d}"
            (out / f"{stem}.jpg").write_bytes(data)
            tile = (
                Image.open(BytesIO(data))
                .convert("RGB")
                .resize((TILE, TILE), Image.Resampling.LANCZOS)
            )
            sheet.paste(tile, (column * TILE, row * TILE))
            print(stem, len(data), "bytes")
    sheet.save(out / "sheet.png")
    print(
        f"CPU time per cover: mean {1000 * sum(times) / len(times):.0f} ms, worst {1000 * max(times):.0f} ms"
    )


if __name__ == "__main__":
    main()
