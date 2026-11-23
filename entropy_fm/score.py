"""
The score of a soundscape: notes made by generative processes, and the layers that carry them.

A `Note` says what to play (a voice of `voices`, or a drum hit named `drum:<hit>`), when, how long and how
hard. A `Layer` is a bus of notes with a gain that follows a curve, optionally a filter whose cutoff follows
another, and how much of it goes to the reverb and the echo. Curves have a value every tenth of a second.
Nothing here is audio: the score is made at once from the data of the day, which is what lets a track start
at any second and be cached.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .lead import Leads

# values per second of a curve
CURVE_RATE = 10
# the keys of the layers and the voices that more than one module names
LAYER_LEAD = "lead"
LAYER_BASS = "bass"
LAYER_DRONE = "drone"
LAYER_PIANO = "piano"
# the strings of Ambient, Berlin School and Lo-fi
LAYER_STRINGS = "strings"
# the strings of Cinematic Glitch
LAYER_STR = "str"
LAYER_KEYS = "keys"
# the plucked layer of Lo-fi
LAYER_PLUCK = "pluck"
# the plucked layer of Ritual
LAYER_PLUCKS = "plucks"
LAYER_SEQ_A = "seqA"
LAYER_WASH = "wash"
LAYER_GROWL = "growl"
# the drums of the bus of Cinematic Glitch, which `Glitch` names
LAYER_KICK = "kick"
LAYER_SNARE = "snare"
LAYER_HATS = "hats"
VOICE_TAPE_CHOIR = "tape_choir"
VOICE_TAPE_STRINGS = "tape_strings"
# the defaults of a bus: its low-pass resonance, and how much goes to the reverb
DEFAULT_RESONANCE = 0.15
DEFAULT_REVERB = 0.2
# the defaults of the space of a piece: the echo in seconds with its feedback, and the loudness in dB RMS
DEFAULT_ECHO_SECONDS = 0.375
DEFAULT_ECHO_FEEDBACK = 0.4
DEFAULT_LEVEL_DB = -20.0


@dataclass(frozen=True)
class Note:
    """One note or hit: a start in seconds, a length, a pitch (MIDI), a velocity and the voice that plays it."""

    t: float
    dur: float
    midi: float
    vel: float
    voice: str
    layer: str
    patch: int = 0
    # the depth in cents of a vibrato that fades in as the note is held (0 for none)
    vib: float = 0.0


@dataclass
class Layer:
    """A bus of the mix."""

    key: str
    # the constant gain of the bus, a design value of the style
    gain: float = 1.0
    # how present the layer is, 0 to 1, a value every tenth of a second (None means always 1)
    presence: np.ndarray | None = None
    # the cutoff of a low-pass in Hz that the house moves, a value every tenth of a second (None for none)
    cutoff: np.ndarray | None = None
    resonance: float = DEFAULT_RESONANCE
    # how much goes to the reverb and to the echo
    reverb: float = DEFAULT_REVERB
    echo: float = 0.0
    # how much goes to the long hall that only the layers that ask for it are sent to (0 for none); a tier without the
    # hall sends it to the shared reverb instead
    hall: float = 0.0
    # the echo send as a curve (a value every tenth of a second) that scales `echo`, or None for a constant
    echo_curve: np.ndarray | None = None
    # the human name, for the notes about a piece
    title: str = ""
    # a high-pass in Hz (0 for none), which keeps the low end of the mix for the layers that own it
    highpass: float = 0.0


@dataclass
class Texture:
    """A layer that is made from noise, not from notes: the room rumble of Ambient and the vinyl of Lo-fi."""

    key: str
    kind: str
    gain: float
    # an optional level curve, a value every tenth of a second
    level: np.ndarray | None = None


@dataclass
class Edit:
    """A splice of a bus of Cinematic Glitch: a stutter, a tape stop, a brake, a freeze, a crush, a gate chop, a reversal, a pitch repeat, a bit drop or a roll."""

    # where it starts, in seconds of the piece, on the grid
    t: float
    # one of `glitch.EDIT_KINDS`
    kind: str
    # the layers it is made on
    layers: tuple[str, ...]
    # how many seconds from its start it changes, and how many before its start it reads
    dur: float
    back: float
    # what the kind needs: lengths, gains, a seed
    params: dict[str, Any] = field(default_factory=dict)
    # what brought it: an event of the house, the groove of the bars, or the structure of the piece
    why: str = ""


@dataclass
class Glitch:
    """What the render of Cinematic Glitch needs besides the notes: the splices, the kicks that duck, the tempo."""

    tempo: float
    edits: list[Edit] = field(default_factory=list)
    # the kicks as (second, strength), sorted: the bass, the pads, the strings and the stabs duck from them
    kicks: list[tuple[float, float]] = field(default_factory=list)
    # the layers that duck: (depth, time constant in seconds)
    duck: dict[str, tuple[float, float]] = field(default_factory=dict)
    # the layers that make up the drum bus
    drum_layers: tuple[str, str, str] = (LAYER_KICK, LAYER_SNARE, LAYER_HATS)
    # the set of vowels of the growl that the day chose
    vowels: int = 0


@dataclass
class Score:
    """Everything a piece is made of."""

    length: int
    notes: list[Note]
    layers: dict[str, Layer]
    textures: list[Texture] = field(default_factory=list)
    # the reverb: its size 0 to 1, the share of wet at each moment (a curve), and the echo in seconds with its feedback
    space: np.ndarray | None = None
    echo_seconds: float = DEFAULT_ECHO_SECONDS
    echo_feedback: float = DEFAULT_ECHO_FEEDBACK
    # the tempo, for the notes about a piece, and 0 where there is none
    tempo: float = 0.0
    # what the day chose, what the generative systems did, how the house shaped it (for soundscapes.md)
    choices: dict[str, str] = field(default_factory=dict)
    notes_about: list[str] = field(default_factory=list)
    # the swell and dip events of anomalies as (second, strength), already in the curves
    events: list[tuple[float, float]] = field(default_factory=list)
    # the share of the reverb send that is also fed an octave up, in the richest tier (a shimmer), 0 for none
    shimmer: float = 0.0
    # the threshold in dBFS of the compressor before the limiter, which tames the peaks of drums (None for none)
    compress_db: float | None = None
    # a wobble of the tape over the whole mix, in cents (Lo-fi)
    wobble: float = 0.0
    # the loudness the leveler holds the style at, in dB RMS
    level_db: float = DEFAULT_LEVEL_DB
    # the gain in dB that the loudness holds with, estimated from the notes (`levels.feed_forward`), a value every tenth of a second
    gain_db: np.ndarray | None = None
    # the splices, kicks and tempo of Cinematic Glitch, whose render is its own (`glitch_render`)
    glitch: Glitch | None = None
    # who leads each phrase of the form, where the style has leads (`lead`)
    leads: Leads | None = None

    def voices_used(self) -> dict[str, int]:
        """Return how many notes each voice plays."""
        count: dict[str, int] = {}
        for note in self.notes:
            count[note.voice] = count.get(note.voice, 0) + 1
        return count


def curve_at(curve: np.ndarray, t: float) -> float:
    """Return the value of a curve (CURVE_RATE per second) at a second."""
    position = min(max(t * CURVE_RATE, 0.0), len(curve) - 1.0)
    low = int(position)
    high = min(low + 1, len(curve) - 1)
    return float(curve[low] + (curve[high] - curve[low]) * (position - low))
