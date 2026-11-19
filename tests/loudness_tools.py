"""Short-term loudness of a rendered song, measured with ffmpeg's EBU R128 filter."""

import re
import shutil
import subprocess

import numpy as np

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def short_term(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return the times, the short-term loudness (3 s) in LUFS and the integrated loudness.

    :param audio: Stereo float samples at 44.1 kHz.
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
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        input=audio.astype(np.float32).tobytes(),
        capture_output=True,
        check=False,
    )
    rows = re.findall(
        r"t:\s+([\d.]+)\s+TARGET:.*?M:\s*([-\d.]+)\s+S:\s*([-\d.]+)\s+I:\s*([-\d.]+)",
        done.stderr.decode(),
    )
    times = np.array([float(row[0]) for row in rows])
    levels = np.array([float(row[2]) for row in rows])
    return times, levels, float(rows[-1][3])
