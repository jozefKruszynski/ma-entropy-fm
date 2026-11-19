"""A small pitch tracker for the tests: it finds continuous glides, and ignores stepped notes."""

from __future__ import annotations

import numpy as np

RATE = 44100
FRAME = 4096
HOP = 1024
LOW_HZ = 150.0
HIGH_HZ = 3000.0
# a track follows a peak that moves by less than this between two frames (a glide is smooth)
MAX_STEP_SEMITONES = 0.55
PEAKS_PER_FRAME = 6
PROMINENCE = 4.0


def _peaks(frame: np.ndarray, window: np.ndarray, freqs: np.ndarray) -> list[tuple[float, float]]:
    """Return the strongest spectral peaks of a frame as (Hz, magnitude), interpolated."""
    spectrum = np.abs(np.fft.rfft(frame * window))
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    floor = float(np.median(spectrum[band])) + 1e-9
    inside = np.flatnonzero(band)
    found: list[tuple[float, float]] = []
    for index in inside[1:-1]:
        here = spectrum[index]
        if here > spectrum[index - 1] and here >= spectrum[index + 1] and here > PROMINENCE * floor:
            before, after = np.log(spectrum[index - 1] + 1e-12), np.log(spectrum[index + 1] + 1e-12)
            centre = np.log(here + 1e-12)
            shift = 0.5 * (before - after) / (before - 2 * centre + after)
            found.append((float((index + shift) * RATE / FRAME), float(here)))
    found.sort(key=lambda item: -item[1])
    return found[:PEAKS_PER_FRAME]


def find_glides(  # noqa: C901, PLR0912
    mono: np.ndarray, minimum: float = 3.0, within: float = 1.5, direction: int = -1
) -> list[tuple[float, float]]:
    """
    Return the continuous glides in a mono signal: (start second, semitones moved).

    A glide is a track of peaks that moves smoothly from frame to frame, always the same way, and
    covers more than `minimum` semitones inside `within` seconds. Notes that jump from one pitch
    to the next, as an arpeggio does, are tracks of one frame or of a flat pitch and never count.

    :param mono: Samples at 44.1 kHz.
    :param minimum: The distance in semitones that counts.
    :param within: The time it has to happen in, in seconds.
    :param direction: -1 for falls, 1 for rises.
    """
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / RATE)
    frames = [
        _peaks(mono[start : start + FRAME], window, freqs)
        for start in range(0, len(mono) - FRAME, HOP)
    ]
    tracks: list[list[tuple[int, float]]] = []
    live: list[list[tuple[int, float]]] = []
    for number, peaks in enumerate(frames):
        taken: set[int] = set()
        carried: list[list[tuple[int, float]]] = []
        for track in live:
            last = track[-1][1]
            best, best_index = MAX_STEP_SEMITONES, -1
            for index, (freq, _) in enumerate(peaks):
                if index in taken:
                    continue
                step = abs(12.0 * np.log2(freq / last))
                if step < best:
                    best, best_index = step, index
            if best_index >= 0:
                taken.add(best_index)
                track.append((number, peaks[best_index][0]))
                carried.append(track)
            else:
                tracks.append(track)
        for index, (freq, _) in enumerate(peaks):
            if index not in taken:
                carried.append([(number, freq)])
        live = carried
    tracks.extend(live)
    span = int(within * RATE / HOP)
    found: list[tuple[float, float]] = []
    for track in tracks:
        if len(track) < 6:
            continue
        pitch = 12.0 * np.log2(np.array([f for _, f in track]) / 440.0)
        # smooth the track a little, then look for a monotonic move inside the window
        smooth = np.convolve(pitch, np.ones(3) / 3.0, mode="valid")
        for first in range(len(smooth)):
            last = min(first + span, len(smooth) - 1)
            piece = smooth[first : last + 1]
            moves = np.diff(piece)
            if len(moves) < 5:
                continue
            # a fall (or rise): most steps go the same way and the total is large
            if direction < 0:
                total = float(piece[0] - piece.min())
                steady = float(np.mean(moves <= 0.05))
            else:
                total = float(piece.max() - piece[0])
                steady = float(np.mean(moves >= -0.05))
            if total > minimum and steady > 0.8:
                found.append((track[first][0] * HOP / RATE, total))
                break
    return found


def noise_floor_db(mono: np.ndarray, frame: int = 2205, share: float = 0.2) -> float:
    """
    Return the level in dBFS of what sits above 6 kHz in the quietest 20 percent of 50 ms frames.

    Steady hiss shows there; hats and cymbals only sound in the loud frames and do not count.

    :param mono: Samples at 44.1 kHz.
    :param frame: The length of a frame in samples, 50 ms.
    :param share: The share of the quietest frames that is looked at.
    """
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / RATE)
    levels = []
    for start in range(0, len(mono) - frame, frame):
        piece = mono[start : start + frame]
        spectrum = np.abs(np.fft.rfft(piece * window)) ** 2
        high = spectrum[freqs >= 6000.0].sum() * 2.0 / (frame**2 * np.mean(window**2))
        levels.append((float(np.mean(piece**2)), float(high)))
    quiet = sorted(levels)[: max(1, int(len(levels) * share))]
    return float(10.0 * np.log10(max(float(np.mean([high for _, high in quiet])), 1e-14)))
