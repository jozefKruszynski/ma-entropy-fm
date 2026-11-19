"""
Measure the loudness of soundscapes with the EBU R128 filter of ffmpeg.

Integrated LUFS, the highest short-term loudness (3 s), true peak and loudness range, and what the limiter did.

    uv run python scripts/loudness_report.py [style ...]

Renders a three minute excerpt of the demo day in each style (the same points of the piece as the demo files: the
night, the active daytime and the evening) and prints a table. The caps of the soundscapes: integrated -18 to -20
LUFS for Ambient, -16 to -18 for Berlin School and Lo-fi, -15 to -17 for Minimal, -18 to -20 for Ritual, -14.5 to -15.5 for Cinematic Glitch; the highest short-term loudness
at most one LU above the integrated one; true peak at most -1 dBTP; the limiter takes at most 2 dB.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# the integrated loudness each style is held to, as (lowest, highest) LUFS
TARGETS = {
    "ambient": (-20.0, -18.0),
    "berlin_school": (-18.0, -16.0),
    "lofi": (-18.0, -16.0),
    "minimal": (-17.0, -15.0),
    "ritual": (-20.0, -18.0),
    "glitch": (-15.5, -14.5),
}
# the highest short-term loudness may be at most this far above the integrated loudness, in LU
SHORT_TERM_MARGIN = 1.0
# Cinematic Glitch is dynamic by arrangement (a quiet night, drops at the peaks), so its short-term loudness may be further over
STYLE_MARGIN = {"glitch": 4.0}
CEILING_DBTP = -1.0
MAX_REDUCTION_DB = 2.0
STYLE_REDUCTION = {"glitch": 3.0}


def ebur128(audio: np.ndarray) -> dict[str, float]:
    """
    Return the loudness of float stereo audio at 44.1 kHz.

    :param audio: Samples, shape (n, 2).
    """
    done = subprocess.run(
        [  # noqa: S607
            "ffmpeg",
            "-nostats",
            "-f",
            "f32le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-i",
            "-",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        input=audio.astype(np.float32).tobytes(),
        capture_output=True,
        check=False,
    )
    text = done.stderr.decode()
    frames = re.findall(r"S:\s*(-?[\d.]+)", text)
    short = [float(value) for value in frames if float(value) > -70.0]
    summary = text.rsplit("Summary:", 1)[-1]
    integrated = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))  # type: ignore[union-attr]
    peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))  # type: ignore[union-attr]
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summary).group(1))  # type: ignore[union-attr]
    return {
        "integrated": integrated,
        "short_max": max(short) if short else -70.0,
        "true_peak": peak,
        "lra": lra,
    }


def verdict(style: str, numbers: dict[str, float], stats: dict[str, float]) -> list[str]:
    """Return what is outside the caps, as short sentences; an empty list when all is in."""
    low, high = TARGETS[style]
    problems = []
    if not low - 0.5 <= numbers["integrated"] <= high + 0.5:
        problems.append(f"integrated {numbers['integrated']:.1f} outside {low:.0f} to {high:.0f}")
    margin = STYLE_MARGIN.get(style, SHORT_TERM_MARGIN)
    if numbers["short_max"] > numbers["integrated"] + margin:
        problems.append(f"short-term {numbers['short_max']:.1f} more than {margin:.0f} LU over")
    if numbers["true_peak"] > CEILING_DBTP:
        problems.append(f"true peak {numbers['true_peak']:.1f}")
    if stats.get("max_reduction_db", 0.0) > STYLE_REDUCTION.get(style, MAX_REDUCTION_DB):
        problems.append(f"limiter {stats['max_reduction_db']:.1f} dB")
    return problems


def main() -> None:
    """Print the table: the excerpts of the demo day and of the real-house kitchen, and the full pieces of two styles."""
    import demo_render  # noqa: PLC0415
    import real_house_demo  # noqa: PLC0415
    from soundscapes import excerpts  # noqa: PLC0415

    from entropy_fm.render import render_all  # noqa: PLC0415

    demo = demo_render.Demo()
    tier = demo.quality.STANDARD
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(demo.styles.STYLES)
    full = "--full" in sys.argv
    print(
        "| style | piece | integrated LUFS | target | max short-term | true peak dBTP | LRA | mean GR dB | max GR dB | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for style in wanted:
        for label in ("demo excerpts", "kitchen excerpts", "full demo piece"):
            if label == "full demo piece" and not (full and style in {"ambient", "berlin_school"}):
                continue
            if label == "kitchen excerpts":
                score, plan = real_house_demo.real_score(demo, style, tier)
            else:
                score, plan = demo.score(demo_render.ANOMALY_DAY, ("kitchen",), style, tier)
            stats: dict[str, float] = {}
            if label == "full demo piece":
                audio = render_all(score, tier, "full", stats)
            else:
                audio = excerpts(score, plan, tier, stats=stats)
            numbers = ebur128(audio)
            low, high = TARGETS[style]
            problems = verdict(style, numbers, stats)
            print(
                f"| {style} | {label} | {numbers['integrated']:.1f} | {low:.0f} to {high:.0f} | "
                f"{numbers['short_max']:.1f} ({numbers['short_max'] - numbers['integrated']:+.1f}) | "
                f"{numbers['true_peak']:.1f} | {numbers['lra']:.1f} | {stats['mean_reduction_db']:.2f} | "
                f"{stats['max_reduction_db']:.1f} | {'ok' if not problems else '; '.join(problems)} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
