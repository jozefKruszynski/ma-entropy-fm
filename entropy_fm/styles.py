"""
The five styles of Entropy FM: Ambient, Berlin School, Lo-fi, Ritual and Cinematic Glitch.

A style is a name, the modes it allows, a tempo range where it has a pulse, and how it is listed in the
library. What it sounds like is decided by its composer (`scape_ambient`, `scape_berlin`, `scape_lofi`,
`scape_ritual`, `scape_glitch`), a generative system with a vocabulary of its own.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass

STYLE_AMBIENT = "ambient"
STYLE_BERLIN_SCHOOL = "berlin_school"
STYLE_LOFI = "lofi"
STYLE_RITUAL = "ritual"
STYLE_GLITCH = "glitch"
DEFAULT_STYLE = STYLE_AMBIENT
# the tempo of a style without a pulse
NO_TEMPO = (0.0, 0.0)
GENERATIVE = "Generative"


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
    genres: tuple[str, ...] = (GENERATIVE,)


AMBIENT = Style(
    key=STYLE_AMBIENT,
    title="Ambient",
    version="Ambient",
    modes=("dorian", "aeolian", "mixolydian", "lydian", "major"),
    tempo=NO_TEMPO,
    description="Tape loops of unrelated lengths that phase against each other, a sparse felt piano, drones. No beat.",
    genres=("Ambient", GENERATIVE),
)
BERLIN_SCHOOL = Style(
    key=STYLE_BERLIN_SCHOOL,
    title="Berlin School",
    version="Berlin School",
    modes=("aeolian", "dorian", "mixolydian", "phrygian"),
    tempo=(102.0, 116.0),
    description="Hypnotic Moog-style sequences that mutate by rule, tape strings and choir, an organ bed, a slow lead.",
    genres=("Electronic", "Berlin School", GENERATIVE),
)
LOFI = Style(
    key=STYLE_LOFI,
    title="Lo-fi",
    version="Lo-fi",
    modes=("dorian", "mixolydian", "aeolian", "major", "lydian"),
    tempo=(70.0, 88.0),
    description="A dusty swung boom-bap loop, felt piano or organ chords on jazzy progressions, a round bass, vinyl.",
    genres=("Lo-fi", "Chillhop", GENERATIVE),
)
RITUAL = Style(
    key=STYLE_RITUAL,
    title="Ritual",
    version="Ritual",
    modes=("aeolian", "dorian"),
    tempo=NO_TEMPO,
    description="Plucked loops in a Japanese scale that phase over a breathing choir drone, stretched strings, a slow chant leader, a soft heartbeat and taiko.",
    genres=("Ambient", "Ritual", GENERATIVE),
)
GLITCH = Style(
    key=STYLE_GLITCH,
    title="Cinematic Glitch",
    version="Cinematic Glitch",
    modes=("aeolian", "phrygian", "dorian"),
    tempo=(54.0, 72.0),
    description="Film-score swells, brass-like stabs and a growling bass over hard half-time drums, cut up by the doors, switches and appliances of the house.",
    genres=("Electronic", "Cinematic", GENERATIVE),
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
