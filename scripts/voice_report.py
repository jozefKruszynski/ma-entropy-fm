"""
Measure the voices of the library.

The fingerprint of each (centroid, attack, harmonic profile, modulation), the share of energy that is not on a
harmonic (aliasing and noise), and the time it takes to make a note.

    uv run python scripts/voice_report.py [voice ...]

The inharmonic share is the energy outside windows of 6 percent of the pitch round each harmonic, above
100 Hz, as a share of the total; a tonal voice should stay under 2 percent. A growl that is waveshaped
before its filter makes harmonics only, so it has to stay under it too.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy_fm.quality import STANDARD
from entropy_fm.voices import VOICES, render_note

RATE = 44100


@dataclass(frozen=True)
class Fingerprint:
    """The numbers that tell a voice from another."""

    centroid: float
    attack_ms: float
    # the share of the energy in the first harmonic, the 2nd to 4th, the 5th to 10th and above the 10th
    profile: tuple[float, float, float, float]
    # the depth of the slow movement of the level in the sustained part: std over mean of the 20 ms RMS
    modulation: float
    inharmonic: float
    # the same with windows of 1.6 percent of each harmonic, for voices that are detuned or chorused on purpose
    inharmonic_wide: float = 0.0


def fingerprint(wave: np.ndarray, f0: float) -> Fingerprint:
    """
    Measure a note.

    :param wave: Stereo or mono samples of one note, held for at least two seconds.
    :param f0: Its pitch in Hz.
    """
    mono = wave.mean(axis=1) if wave.ndim == 2 else wave
    mono = mono.astype(np.float64)
    hop = int(0.005 * RATE)
    env = np.sqrt(np.convolve(mono**2, np.ones(hop) / hop, mode="same"))
    peak = float(env[: int(1.5 * RATE)].max()) or 1.0
    attack = float(np.argmax(env >= 0.9 * peak)) / RATE * 1000.0
    start = int(0.35 * RATE)
    piece = mono[start : start + 32768]
    if len(piece) < 32768:
        piece = np.pad(piece, (0, 32768 - len(piece)))
    spectrum = np.abs(np.fft.rfft(piece * np.hanning(len(piece)))) ** 2
    freqs = np.fft.rfftfreq(len(piece), 1.0 / RATE)
    total = float(spectrum.sum()) or 1.0
    centroid = float((freqs * spectrum).sum() / total)
    harmonic = np.zeros_like(spectrum, dtype=bool)
    wide = np.zeros_like(spectrum, dtype=bool)
    shares = np.zeros(4)
    for k in range(1, int(20000 / f0) + 1):
        near = np.abs(freqs - k * f0) < f0 * 0.06
        harmonic |= near
        wide |= np.abs(freqs - k * f0) < max(f0 * 0.06, 0.016 * k * f0)
        index = 0 if k == 1 else 1 if k <= 4 else 2 if k <= 10 else 3
        shares[index] += spectrum[near].sum()
    inharmonic = float(spectrum[~harmonic & (freqs > 100.0)].sum() / total)
    inharmonic_wide = float(spectrum[~wide & (freqs > 100.0)].sum() / total)
    frame = int(0.02 * RATE)
    steady = mono[int(0.5 * RATE) : int(2.0 * RATE)]
    levels = np.sqrt(
        np.mean(steady[: len(steady) // frame * frame].reshape(-1, frame) ** 2, axis=1)
    )
    trend = np.convolve(levels, np.ones(15) / 15.0, mode="same")[7:-7]
    ripple = levels[7:-7] - trend
    modulation = float(ripple.std() / max(levels.mean(), 1e-9))
    return Fingerprint(
        centroid,
        attack,
        (
            float(shares[0] / total),
            float(shares[1] / total),
            float(shares[2] / total),
            float(shares[3] / total),
        ),
        modulation,
        inharmonic,
        inharmonic_wide,
    )


def main() -> None:
    """Print the table."""
    wanted = sys.argv[1:] or list(VOICES)
    print(
        "| voice | note | centroid Hz | attack ms | H1 / H2-4 / H5-10 / >10 | modulation | inharmonic % | wide % | ms to make |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key in wanted:
        voice = VOICES[key]
        for midi in (voice.ref_midi, voice.ref_midi + 24):
            began = time.perf_counter()
            wave = render_note(key, midi, 2.6, 0.7, STANDARD)
            cost = (time.perf_counter() - began) * 1000.0
            f0 = 440.0 * 2.0 ** ((midi - 69.0) / 12.0) * voice.fundamental
            fp = fingerprint(wave, f0)
            profile = " / ".join(f"{100 * share:.0f}" for share in fp.profile)
            print(
                f"| {key} | {midi:.0f} | {fp.centroid:.0f} | {fp.attack_ms:.0f} | {profile} | "
                f"{fp.modulation:.3f} | {100 * fp.inharmonic:.2f} | {100 * fp.inharmonic_wide:.2f} | {cost:.0f} |"
            )


if __name__ == "__main__":
    main()
