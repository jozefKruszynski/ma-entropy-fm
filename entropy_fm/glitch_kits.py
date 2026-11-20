"""
The drum families of Cinematic Glitch: half-time, trap, broken, ghosted break, cinematic, industrial, dub and IDM kits.

A family is a handful of lanes drawn on the grid of the kit, sixteen steps to a half bar (an eighth of a beat of the
tempo each) or thirty-two for a family of two half bars. A lane is a string: `.` is a rest, `1` to `9` a hit at a tenth
of full velocity, `X` a full hit, spaces are ignored. A family has three or four variants (they change the kick and
the ghosts), and the level of the day picks how many lanes sound: the kick and the snare alone, then the hats, the
ghosts, the rim and the ticks, then the open hats and every kick. Each family also has fills of one half bar.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import Any, NamedTuple

Pattern = dict[str, Any]

STEPS = 16
KINDS = ("sparse", "light", "mid", "full", "big")
# the lanes that sound from each level of the kit on; the kick and the snare are always there
LANE_FROM = {
    "kick": 0,
    "snare": 0,
    "hats": 1,
    "toms": 1,
    "ghost": 2,
    "rim": 2,
    "ticks": 2,
    "open": 3,
}
# the softest kick that sounds at each level: the first level keeps the downbeat alone, the last every kick
KICK_FLOOR = {"sparse": 0.9, "light": 0.75, "mid": 0.6, "full": 0.5, "big": 0.0}
# the level at which the hats are every other step: the softer hats of the lane wait for the next level
HAT_FLOOR = {"sparse": 0.0, "light": 0.35, "mid": 0.0, "full": 0.0, "big": 0.0}


class Family(NamedTuple):
    """A family of patterns: its energy (0 to 1), the modes it suits, its patterns by level and its fills."""

    name: str
    energy: float
    modes: tuple[str, ...]
    patterns: dict[str, tuple[Pattern, ...]]
    fills: tuple[Pattern, ...]


def _lane(text: str) -> list[tuple[int, float]]:
    """Return the hits of a lane as (step, velocity)."""
    steps = text.replace(" ", "")
    return [
        (index, 1.0 if char == "X" else int(char) / 10.0)
        for index, char in enumerate(steps)
        if char != "."
    ]


def _build(lanes: dict[str, str], kind: str) -> Pattern:
    """Return the pattern of the lanes at a level: the lanes that sound there, the kick thinned."""
    rank = KINDS.index(kind)

    def lane(name: str) -> list[tuple[int, float]]:
        return _lane(lanes.get(name, "")) if rank >= LANE_FROM[name] else []

    hats = lane("hats")
    steps = max(len(text.replace(" ", "")) for text in lanes.values())
    return {
        "kick": [(s, v) for s, v in lane("kick") if v >= KICK_FLOOR[kind]],
        "snare": lane("snare"),
        "ghost": lane("ghost"),
        "hats": tuple(
            next((v for s, v in hats if s == step and v >= HAT_FLOOR[kind]), 0.0)
            for step in range(steps)
        ),
        "rim": lane("rim"),
        "ticks": lane("ticks"),
        "open": lane("open"),
        "toms": lane("toms"),
        "span": steps // STEPS,
    }


def _family(  # noqa: PLR0913, PLR0917
    name: str,
    energy: float,
    modes: tuple[str, ...],
    shared: dict[str, str],
    variants: tuple[dict[str, str], ...],
    fills: tuple[dict[str, str], ...],
) -> Family:
    lanes = [{**shared, **variant} for variant in variants]
    return Family(
        name,
        energy,
        modes,
        {kind: tuple(_build(each, kind) for each in lanes) for kind in KINDS},
        tuple(_build(fill, "big") for fill in fills),
    )


HEAVY, HALF_TIME, DRIVING = "heavy", "half-time", "driving"
# fmt: off
FAMILIES: tuple[Family, ...] = (
    _family(
        "dubstep half-time", 0.7, (HALF_TIME, DRIVING),
        {"snare": "....  ....  X...  ....", "hats": "5242 5242 5242 5243"},
        (
            {"kick": "X... .... ..8. ...5", "ghost": "...2 .... ...2 ...2", "rim": "..5. .... .... ....",
             "ticks": "..4. ..3. .... ..4.", "open": ".... ..4. .... ...."},
            {"kick": "X... .... ..9. 5...", "ghost": "...2 .... .2.. ....", "rim": ".... .... .... .5..",
             "ticks": "..4. ..3. .... 4...", "open": ".... ..4. .... ..5."},
            {"kick": "X..5 ..8. .... ....", "ghost": ".... 2... ...2 ...2", "rim": ".5.. .... .... ....",
             "ticks": "..4. ..4. ..4. ....", "open": ".... ..4. .... ...."},
            {"kick": "X... ..5. ..9. ....", "ghost": "..2. .... .2.. .2.2", "rim": "..5. .... ..5. ....",
             "ticks": "..3. 4... ..4. ..3.", "open": ".... .... .... ..5."},
        ),
        (
            {"snare": "........3456789X", "hats": "5.5.5.5............."},
            {"snare": "....4.5.6.7.89X.", "kick": "X..............."},
        ),
    ),
    _family(
        "trap half-time", 0.8, (HALF_TIME, DRIVING),
        {"snare": "....  ....  X...  ....", "rim": ".... .... 5... ...."},
        (
            {"kick": "X... .... .... 8.5.", "hats": "5.4. 5.4. 5.4. 5.77", "open": ".... .... .... ..5.",
             "ticks": ".... .... .... ...."},
            {"kick": "X... .... ..8. ..5.", "hats": "5.4. 5.4. 5.55 6789", "open": ".... .... .... ....",
             "ticks": "..3. ..3. ..3. ..3."},
            {"kick": "X... .... .8.. ...5", "hats": "5.45 5.45 5.45 6.78", "open": ".... ..4. .... ....",
             "ticks": ".... ..3. .... ..3."},
            {"kick": "X... .5.. ..8. ....", "hats": "5.4. 5.5. 5.4. 5567", "open": ".... .... ..4. ....",
             "ticks": "..3. .... ..3. ...."},
        ),
        (
            {"hats": "....5566 7788 99XX"},
            {"snare": "........4.5.6.8X", "hats": "5.5.5.5.5.5.5.5."},
        ),
    ),
    _family(
        "broken half-time", 0.5, (HALF_TIME, DRIVING),
        {"hats": "5.4. 5.4. 5.4. 5.4.", "ticks": "..3. ..3. ..3. ..3.", "open": ".... .... .... ..4."},
        (
            {"snare": "....  ....  ..X.  ....", "kick": "X... .... .... 8.5.", "ghost": "...2 ..2. .... .2.2",
             "rim": ".... .... .... ..4."},
            {"snare": "....  ....  ...X  ....", "kick": "X... .... .... 8..5", "ghost": "..2. ...2 ..2. ....",
             "rim": ".... .4.. .... ...."},
            {"snare": "....  ....  ..X.  ...6", "kick": "X... .... ...8 ..5.", "ghost": ".2.. ..2. ...2 .2..",
             "rim": "..4. .... .... ...."},
            {"snare": "....  ...5  ....  X...", "kick": "X... ..8. .... ...5", "ghost": "..2. .... .2.2 ..2.",
             "rim": ".... .... ..4. ...."},
        ),
        (
            {"snare": "........2.3.4.7X", "hats": "5.5.5.5............."},
            {"snare": "....2.3.4.5.6.8X"},
        ),
    ),
    _family(
        "ghosted breakbeat half-time", 0.55, (HALF_TIME, DRIVING),
        {"snare": "....  ....  X...  ....", "hats": "5.4. 5.4. 5.4. 5.4.", "open": ".... ..5. .... ..5.",
         "ticks": "..3. ..3. ..3. ..3."},
        (
            {"kick": "X... .... ..9. ..5.", "ghost": "...3 .... .3.3 ...3", "rim": ".... .... .... ...."},
            {"kick": "X.5. .... ..9. ....", "ghost": "...3 ..3. ...3 .3..", "rim": ".... ..4. .... ...."},
            {"kick": "X... .... ..9. 5...", "ghost": "..3. .3.. ...3 ..3.", "rim": ".... .... .... ..4."},
        ),
        (
            {"snare": "........3.4.5.7X", "hats": "5.5.5.5............."},
            {"snare": "....3.3.4.5.6.8X"},
        ),
    ),
    _family(
        "IDM stutter half-time", 0.9, (DRIVING,),
        {"snare": "....  ....  X...  ....", "hats": "6.45 6.45 6.45 6.45", "open": ".... ..4. .... ....",
         "rim": ".... .... .... ..5."},
        (
            {"kick": "X.5. .... ..9. ....", "ghost": "...2 ..2. ...2 ..22", "ticks": "..4. .4.4 ..4. 4.4."},
            {"kick": "X... .5.. ..9. ....", "ghost": "..2. ...2 .2.. ...2", "ticks": ".4.. 4... .4.4 ..4."},
            {"kick": "X6.. .... ..9. ....", "ghost": "...2 .2.. ..22 ....", "ticks": "..4. ..4. 4.4. ..4."},
            {"kick": "X..6 .... ..9. ....", "ghost": ".2.. ...2 ...2 2.2.", "ticks": "4.4. ..4. ..4. 4..."},
        ),
        (
            {"snare": "........3456789X", "hats": "6.6.6.6............."},
            {"snare": "....4444.6.7.8.X", "kick": "X..............."},
        ),
    ),
    _family(
        "cinematic low hits", 0.15, (HEAVY, HALF_TIME),
        {"snare": "....  ....  ....  ....  X...  ....  ....  ....",
         "hats": "....  ....  4...  ....  ....  ....  4...  ....",
         "ticks": "....  3...  ....  ....  ....  3...  ....  ...."},
        (
            {"kick": "X...  ....  ....  ....  ....  ....  ....  ....", "toms": "....  ....  ....  6.8.  ....  ....  ....  ...."},
            {"kick": "X...  ....  ....  ....  ....  ..9.  ....  ....", "toms": "....  ....  ....  ....  ....  ....  7.9.  ...."},
            {"kick": "X...  ....  ....  ....  ....  ....  9...  ....", "toms": "....  ....  ..5.  ....  ....  ....  ....  6.8."},
        ),
        (
            {"toms": "........5.6.7.8X", "hats": "........4.5.6.7."},
            {"toms": "....5...6...7.9X", "snare": "........4.5.6.7X"},
        ),
    ),
    _family(
        "sparse industrial", 0.3, (HEAVY, HALF_TIME),
        {"snare": "....  ....  ....  ....  ....  ....  7...  ....",
         "hats": "4...  ....  4...  ....  4...  ....  4...  ....",
         "rim": "....  5...  ....  5...  ....  5...  ....  5...",
         "open": "....  ....  ....  ....  ....  ....  ....  ..4."},
        (
            {"kick": "X...  ....  ....  ....  ....  9...  ....  ....", "ticks": "..3.  ..3.  ..3.  ..3.  ..3.  ..3.  ..3.  ..3.",
             "ghost": "....  ....  ....  ....  ....  ....  ....  ..2."},
            {"kick": "X...  ....  ....  ..6.  ....  ....  9...  ....", "ticks": ".3..  .3..  .3..  .3..  .3..  .3..  .3..  .3.."},
            {"kick": "X...  ....  ..6.  ....  ....  ....  ..9.  ....", "ticks": "3...  ..3.  3...  ..3.  3...  ..3.  3...  ..3.",
             "ghost": "....  ....  ....  ....  ..2.  ....  ....  ...."},
        ),
        (
            {"snare": "........4.5.6.7X", "hats": "....3.4.5.6.7.8."},
            {"snare": "....4...5...6.8X", "hats": "4.4.4.4.4.4.4.4."},
        ),
    ),
    _family(
        "one-drop dub", 0.4, (HEAVY, HALF_TIME),
        {"snare": "....  ....  ....  ....  X...  ....  ....  ....",
         "hats": "....  4...  ....  4...  ....  4...  ....  4...",
         "rim": "....  ....  ....  ....  5...  ....  ....  ....",
         "ticks": "..3.  ....  ..3.  ....  ..3.  ....  ..3.  ...."},
        (
            {"kick": "....  ....  ....  ....  X...  ....  ....  ..6.", "ghost": "....  ....  ....  ....  ....  ....  ....  ....",
             "open": "....  ....  ....  ....  ....  ....  ....  ..4."},
            {"kick": "9...  ....  ....  ....  X...  ..5.  ....  ....", "ghost": "....  ....  ....  ....  ....  ....  ..2.  ....",
             "open": "....  ....  ....  ..4.  ....  ....  ....  ...."},
            {"kick": "....  ....  ....  ....  X..6  ....  ....  ...5", "ghost": "....  ..2.  ....  ....  ....  ....  ....  ....",
             "open": "....  ....  ....  ....  ....  ....  ..4.  ...."},
        ),
        (
            {"snare": "........3.4.5.7X", "hats": "....4.4.4.4.4.4."},
            {"snare": "....2...4...6.9X", "hats": "4...4...4...4..."},
        ),
    ),
)
# fmt: on
DRUM_ENERGY = {family.name: family.energy for family in FAMILIES}
