"""
Print level, spectrum and silence statistics of an audio file, as a stand-in for listening.

    uv run python scripts/analyze_audio.py file.mp3 [more files]
    uv run python scripts/analyze_audio.py --table file.mp3 [more files]

Needs ffmpeg on the path. It decodes the file to 16 bit stereo at 44.1 kHz and reports the peak
and RMS levels, the DC offset, the longest quiet stretch, the share of energy per band and the
loudness over time.
"""

import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

BANDS = [(20, 120), (120, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 22050)]


def decode(path: str) -> np.ndarray:
    """Return the file as float stereo samples at 44.1 kHz."""
    raw = subprocess.run(  # noqa: S603
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "2", "-ar", "44100", "-"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0


def db(value: float) -> float:
    """Return a linear level in dB, floored at -120."""
    return float(20 * np.log10(max(value, 1e-6)))


def measure(audio: np.ndarray) -> dict[str, float]:
    """
    Return the headline numbers of stereo audio at 44.1 kHz.

    :param audio: Float samples, shape (n, 2).
    """
    mono = audio.mean(axis=1)
    rate = 44100
    window = rate // 2
    frames = len(mono) // window
    rms = np.array(
        [np.sqrt((mono[i * window : (i + 1) * window] ** 2).mean()) for i in range(frames)]
    )
    longest = run = 0
    for flag in rms < 10 ** (-60 / 20):
        run = run + 1 if flag else 0
        longest = max(longest, run)
    spectrum = np.zeros(window // 2 + 1)
    for index in range(frames):
        chunk = mono[index * window : (index + 1) * window] * np.hanning(window)
        spectrum += np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(window, 1 / rate)
    total = spectrum.sum()
    return {
        "seconds": len(audio) / rate,
        "peak_db": db(np.abs(audio).max()),
        "rms_db": db(np.sqrt((audio**2).mean())),
        "dc": float(max(abs(audio[:, 0].mean()), abs(audio[:, 1].mean()))),
        "above_6k_pct": float(100 * spectrum[freqs >= 6000].sum() / total),
        "floor_6k_db": noise_floor_db(mono),
        "centroid_hz": float((spectrum * freqs).sum() / total),
        "silence_s": longest * 0.5,
        "full_scale": float((np.abs(audio) > 0.999).sum()),
    }


def noise_floor_db(mono: np.ndarray, frame: int = 2205, share: float = 0.2) -> float:
    """
    Return the level in dBFS of what sits above 6 kHz in the quietest 20 percent of 50 ms frames.

    Steady hiss shows there; hats and cymbals only sound in the loud frames and do not count.

    :param mono: Samples at 44.1 kHz.
    :param frame: The length of a frame in samples, 50 ms.
    :param share: The share of the quietest frames that is looked at.
    """
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / 44100)
    levels = []
    for start in range(0, len(mono) - frame, frame):
        piece = mono[start : start + frame]
        spectrum = np.abs(np.fft.rfft(piece * window)) ** 2
        high = spectrum[freqs >= 6000.0].sum() * 2.0 / (frame**2 * np.mean(window**2))
        levels.append((float(np.mean(piece**2)), float(high)))
    quiet = sorted(levels)[: max(1, int(len(levels) * share))]
    return float(10.0 * np.log10(max(float(np.mean([high for _, high in quiet])), 1e-14)))


def analyse(path: str) -> None:
    """Print the statistics of one file."""
    audio = decode(path)
    mono = audio.mean(axis=1)
    rate = 44100
    print(f"== {path}: {len(audio) / rate:.1f} s")
    print(f"peak {db(np.abs(audio).max()):.2f} dBFS, RMS {db(np.sqrt((audio**2).mean())):.2f} dBFS")
    print(f"DC offset L {audio[:, 0].mean():+.6f} R {audio[:, 1].mean():+.6f}")
    clipped = int((np.abs(audio) > 0.999).sum())
    print(f"samples at full scale: {clipped}")
    side = audio[:, 0] - audio[:, 1]
    print(f"stereo width (side/mid RMS): {db(side.std() / max(mono.std(), 1e-9)):.1f} dB")
    window = rate // 2
    frames = len(mono) // window
    rms = np.array(
        [np.sqrt((mono[i * window : (i + 1) * window] ** 2).mean()) for i in range(frames)]
    )
    quiet = rms < 10 ** (-60 / 20)
    longest = run = 0
    for flag in quiet:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    print(f"longest stretch below -60 dB: {longest * 0.5:.1f} s")
    loud = db(rms.max()) - db(np.median(rms))
    print(f"loudest half second vs median: {loud:+.1f} dB")
    spectrum = np.zeros(window // 2 + 1)
    for index in range(frames):
        chunk = mono[index * window : (index + 1) * window] * np.hanning(window)
        spectrum += np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(window, 1 / rate)
    total = spectrum.sum()
    parts = []
    for low, high in BANDS:
        share = spectrum[(freqs >= low) & (freqs < high)].sum() / total
        parts.append(f"{low}-{high} Hz {100 * share:.1f}%")
    print("energy: " + ", ".join(parts))
    flat = float(np.exp(np.mean(np.log(spectrum[1:] + 1e-12))) / np.mean(spectrum[1:] + 1e-12))
    high = spectrum[freqs >= 6000].sum() / total
    print(
        f"spectral flatness {10 * np.log10(flat):.1f} dB (0 is white noise), above 6 kHz {100 * high:.3f}% of energy ({db(np.sqrt(high)):.1f} dB re total)"
    )
    centroid = (spectrum * freqs).sum() / total
    print(f"spectral centroid {centroid:.0f} Hz")
    print(
        f"hiss floor: above 6 kHz in the quietest 20% of 50 ms frames {noise_floor_db(mono):.1f} dBFS"
    )
    step = max(frames // 30, 1)
    curve = [f"{db(rms[i : i + step].mean()):.0f}" for i in range(0, frames, step)]
    print("loudness over time (dB, equal steps): " + " ".join(curve))


def difference(first_path: str, second_path: str) -> dict[str, float]:
    """
    Measure how much two renders of the same window differ: the effect of the data on the music.

    Each render is cut into frames of one second, and each frame into 16 bands on a log scale from
    60 Hz to 12 kHz. The distance of a frame is the mean difference of the band levels in dB. The
    result is the mean distance, the share of frames more than 3 dB apart and the difference of
    the RMS in dB.

    :param first_path: The render with the events.
    :param second_path: The render with the data muted.
    """
    a, b = decode(first_path).mean(axis=1), decode(second_path).mean(axis=1)
    length = min(len(a), len(b)) // 44100 * 44100
    edges = np.geomspace(60.0, 12000.0, 17)
    freqs = np.fft.rfftfreq(44100, 1 / 44100)

    def bands(mono: np.ndarray) -> np.ndarray:
        frames = mono[:length].reshape(-1, 44100) * np.hanning(44100)
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        return np.stack(
            [
                10 * np.log10(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) + 1e-12)
                for lo, hi in pairwise(edges)
            ],
            axis=1,
        )

    distance = np.abs(bands(a) - bands(b)).mean(axis=1)
    return {
        "mean distance dB": float(distance.mean()),
        "frames over 3 dB %": float((distance > 3.0).mean() * 100.0),
        "rms difference dB": abs(
            db(float(np.sqrt((a[:length] ** 2).mean())))
            - db(float(np.sqrt((b[:length] ** 2).mean())))
        ),
    }


def table(paths: list[str]) -> None:
    """Print one row of headline numbers per file."""
    print(
        "| file | s | peak dBFS | RMS dBFS | DC | >6 kHz % | hiss floor dBFS | centroid Hz | longest silence s |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        row = measure(decode(path))
        name = path.rsplit("/", 1)[-1]
        print(
            f"| {name} | {row['seconds']:.0f} | {row['peak_db']:.2f} | {row['rms_db']:.2f} | "
            f"{row['dc']:.5f} | {row['above_6k_pct']:.3f} | {row['floor_6k_db']:.1f} | {row['centroid_hz']:.0f} | "
            f"{row['silence_s']:.1f} |"
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--compare"]:
        for first, second in zip(sys.argv[2::2], sys.argv[3::2], strict=False):
            print(Path(first).name, {k: round(v, 2) for k, v in difference(first, second).items()})
    elif sys.argv[1:2] == ["--table"]:
        table(sys.argv[2:])
    else:
        for name in sys.argv[1:]:
            analyse(name)
