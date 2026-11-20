"""
The five styles of Entropy FM: Ambient, Berlin School, Lo-fi, Ritual and Cinematic Glitch.

A style is a name, the modes it allows, a tempo range where it has a pulse, and how it is listed in the
library. What it sounds like is decided by its composer (`scape_ambient`, `scape_berlin`, `scape_lofi`,
`scape_ritual`, `scape_glitch`), a generative system with a vocabulary of its own.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .synth import Scale

DEFAULT_STYLE = "ambient"


@dataclass(frozen=True)
class Style:
    """A way for the house to sound."""

    key: str
    title: str
    # the version of a track of this style: the name of the style, which the library shows in brackets
    version: str
    modes: tuple[str, ...]
    # the tempo range in beats per minute, (0, 0) for a style without a pulse
    tempo: tuple[float, float]
    description: str = ""
    genres: tuple[str, ...] = ("Generative",)

    def scale_for(self, wanted: Scale) -> Scale:
        """
        Return the scale of a track: the one the user chose, unless the style does not allow it.

        :param wanted: The key and mode from the settings.
        """
        if wanted.mode in self.modes:
            return wanted
        return Scale(wanted.root, self.modes[0])


AMBIENT = Style(
    key="ambient",
    title="Ambient",
    version="Ambient",
    modes=("dorian", "aeolian", "mixolydian", "lydian", "major"),
    tempo=(0.0, 0.0),
    description="Tape loops of unrelated lengths that phase against each other, a sparse felt piano, drones. No beat.",
    genres=("Ambient", "Generative"),
)
BERLIN_SCHOOL = Style(
    key="berlin_school",
    title="Berlin School",
    version="Berlin School",
    modes=("aeolian", "dorian", "mixolydian", "phrygian"),
    tempo=(102.0, 116.0),
    description="Hypnotic Moog-style sequences that mutate by rule, tape strings and choir, an organ bed, a slow lead.",
    genres=("Electronic", "Berlin School", "Generative"),
)
LOFI = Style(
    key="lofi",
    title="Lo-fi",
    version="Lo-fi",
    modes=("dorian", "mixolydian", "aeolian", "major", "lydian"),
    tempo=(70.0, 88.0),
    description="A dusty swung boom-bap loop, electric piano chords on jazzy progressions, a round bass, vinyl.",
    genres=("Lo-fi", "Chillhop", "Generative"),
)
RITUAL = Style(
    key="ritual",
    title="Ritual",
    version="Ritual",
    modes=("aeolian", "dorian"),
    tempo=(0.0, 0.0),
    description="Plucked loops in a Japanese scale that phase over a breathing choir drone, stretched strings, a slow chant leader, a soft heartbeat and taiko.",
    genres=("Ambient", "Ritual", "Generative"),
)
GLITCH = Style(
    key="glitch",
    title="Cinematic Glitch",
    version="Cinematic Glitch",
    modes=("aeolian", "phrygian", "dorian"),
    tempo=(54.0, 72.0),
    description="Film-score swells, brass-like stabs and a growling bass over hard half-time drums, cut up by the doors, switches and appliances of the house.",
    genres=("Electronic", "Cinematic", "Generative"),
)
STYLES: dict[str, Style] = {
    style.key: style for style in (AMBIENT, BERLIN_SCHOOL, LOFI, RITUAL, GLITCH)
}


def get_style(key: str | None) -> Style:
    """
    Return a style by key; an unknown key gives the default style.

    :param key: The key of a style.
    """
    return STYLES.get(key or "", STYLES[DEFAULT_STYLE])
