"""
Run the glide tracker of the tests over audio files and report the continuous pitch glides it finds.

    uv run python scripts/glide_report.py file.mp3 [...]

A glide is a track of spectral peaks that moves smoothly in one direction by more than three semitones in a second and
a half, between 150 Hz and 3 kHz. Entropy FM has none: pitches are constant for the length of a note, a vibrato is at
most 15 cents and the kick's fall is below 150 Hz. The file is decoded with ffmpeg; the longer the file the longer the tracker takes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

from pitch_tools import find_glides


def decode(path: str) -> np.ndarray:
    """Return a file as mono float samples at 44.1 kHz."""
    raw = subprocess.run(  # noqa: S603
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "44100", "-"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def count_glides(mono: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return the falls and the rises that the tracker finds."""
    return find_glides(mono, 3.0, 1.5, -1), find_glides(mono, 3.0, 1.5, 1)


def main() -> None:
    """Print one line per file."""
    print("| file | falls | rises | where |")
    print("|---|---|---|---|")
    for path in sys.argv[1:]:
        falls, rises = count_glides(decode(path))
        where = ", ".join(f"{t:.0f}s" for t, _ in (falls + rises)[:6])
        print(f"| {Path(path).name} | {len(falls)} | {len(rises)} | {where} |")


if __name__ == "__main__":
    main()
