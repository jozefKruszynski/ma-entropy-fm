"""
Pitch and level helpers of Entropy FM: scales and the smooth ramp.

The sound itself is made by `voices` and `drums`; this module only holds what the music theory and
the mixing need and imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timeline import as_array

SAMPLE_RATE = 44100
TWO_PI = 2.0 * np.pi
SEMITONES_PER_OCTAVE = 12
# a fifth up is seven semitones
FIFTH_SEMITONES = 7
# the pitch of the tuning note: A4 is MIDI note 69, at 440 Hz
A4_MIDI = 69.0
A4_HZ = 440.0
DEFAULT_MODE = "dorian"
DEFAULT_ROOT = 2

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SCALES: dict[str, tuple[int, ...]] = {
    DEFAULT_MODE: (0, 2, 3, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
}


@dataclass(frozen=True)
class Scale:
    """A key and a mode; every pitch of the music is a degree of it."""

    root: int = DEFAULT_ROOT
    mode: str = DEFAULT_MODE

    @property
    def intervals(self) -> tuple[int, ...]:
        """Return the semitones of the mode above its root."""
        return SCALES.get(self.mode, SCALES[DEFAULT_MODE])

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
        return SEMITONES_PER_OCTAVE * (octave + 1 + octave_shift) + self.root + steps[index]


def smoothstep(x: np.ndarray) -> np.ndarray:
    """
    Return the smooth 0 to 1 ramp of x, clipped to that range.

    :param x: Positions, 0 and below give 0, 1 and above give 1.
    """
    x = np.clip(x, 0.0, 1.0)
    return as_array(x * x * (3.0 - 2.0 * x))
