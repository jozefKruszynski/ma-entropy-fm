"""
Pitch and level helpers of Entropy FM: scales, frequencies, panning.

The sound itself is made by `voices` and `drums`; this module only holds what the music theory and
the mixing need and imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timeline import as_array

SAMPLE_RATE = 44100
TWO_PI = 2.0 * np.pi

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SCALES: dict[str, tuple[int, ...]] = {
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
}


@dataclass(frozen=True)
class Scale:
    """A key and a mode; every pitch of the music is a degree of it."""

    root: int = 2
    mode: str = "dorian"

    @property
    def intervals(self) -> tuple[int, ...]:
        """Return the semitones of the mode above its root."""
        return SCALES.get(self.mode, SCALES["dorian"])

    @property
    def size(self) -> int:
        """Return the number of notes in the mode."""
        return len(self.intervals)

    def midi(self, degree: int, octave: int) -> int:
        """
        Return the MIDI note of a scale degree; degrees outside 0..n-1 wrap into other octaves.

        :param degree: The scale degree, 0 is the root.
        :param octave: The octave of the root, 4 holds middle C.
        """
        steps = self.intervals
        octave_shift, index = divmod(degree, len(steps))
        return 12 * (octave + 1 + octave_shift) + self.root + steps[index]


def midi_to_hz(note: float) -> float:
    """
    Return the frequency of a MIDI note.

    :param note: The note number, 69 is A at 440 Hz.
    """
    return float(440.0 * 2.0 ** ((note - 69.0) / 12.0))


def pan_gains(pan: float) -> tuple[float, float]:
    """
    Return left and right gain for a pan between -1 and 1, with equal power.

    :param pan: -1 is left, 1 is right.
    """
    angle = (pan + 1.0) * np.pi / 4.0
    return float(np.cos(angle)), float(np.sin(angle))


def smoothstep(x: np.ndarray) -> np.ndarray:
    """
    Return the smooth 0 to 1 ramp of x, clipped to that range.

    :param x: Positions, 0 and below give 0, 1 and above give 1.
    """
    x = np.clip(x, 0.0, 1.0)
    return as_array(x * x * (3.0 - 2.0 * x))
