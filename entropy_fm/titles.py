"""
Names and words for the tracks and albums of Entropy FM, made from what a day was like.

A track gets a title of its own, built from a few facts of the day (the busiest hour, an appliance that ran,
a door that opened at a notable time, the weather inside) and a word bank per style: ceremonial for Ritual,
warm for Lo-fi, and so on. The same room, day and style always give the same
title. The facts hold kinds of things ("dishwasher", "door") and times, never an entity id, a name or a person.

Besides the titles this module holds the short text of each style (its history and what inspired it, how the
house drives it) and the line about the day, in English and Dutch.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from .character import FIRST_WEEKEND_DAY
from .day_analysis import QUIET_KINDS, analyze_day
from .styles import DEFAULT_STYLE
from .timeline import (
    BINARY_KINDS,
    DAY_SECONDS,
    HOURS_PER_DAY,
    MINUTES_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    VoiceKind,
    stable_hash,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .timeline import Series

MAX_TITLE = 40
# a room name is part of a title only when it is short
MAX_ROOM = 14
LANGUAGES = ("en", "nl")
DEFAULT_LANGUAGE = "en"
# the hours where a door or a light is out of the ordinary
NIGHT_START = 23
NIGHT_END = 5
# a temperature above this is taken to be Fahrenheit
FAHRENHEIT_ABOVE = 45.0
FAHRENHEIT_OFFSET = 32.0
FAHRENHEIT_PER_CELSIUS = 1.8
WARM_C = 24.0
COOL_C = 17.0
BUSY_ABOVE = 0.67
QUIET_BELOW = 0.34
# the kinds of thing that a day can be about
NOUN_DISHWASHER = "dishwasher"
NOUN_LAUNDRY = "laundry"
NOUN_HOB = "hob"
NOUN_HUM = "hum"
NOUN_LIGHT = "light"
NOUN_MEDIA = "media"
NOUN_BLINDS = "blinds"
NOUN_CYCLE = "cycle"
NOUN_SIGNAL = "signal"
NOUN_WINDOW = "window"
NOUN_DOOR = "door"
NOUN_MOTION = "motion"
CLIMATE_WARM = "warm"
CLIMATE_COOL = "cool"
# the weight of one event of a kind in telling what the day was about
WEIGHTS = {NOUN_DISHWASHER: 15.0, NOUN_LAUNDRY: 15.0, NOUN_HOB: 6.0}
DEFAULT_WEIGHT = 1.0
APPLIANCES = frozenset({NOUN_DISHWASHER, NOUN_LAUNDRY})
# what counts as an event of a kind: an appliance level that it crosses, or how much a series changes
RUN_LEVEL = 0.2
CYCLE_CHANGE = 0.05
OTHER_CHANGE = 0.1
CHANGING_NOUNS = frozenset({NOUN_HOB, NOUN_CYCLE, NOUN_MEDIA, NOUN_BLINDS})
OPENING_NOUNS = frozenset({NOUN_DOOR, NOUN_WINDOW, NOUN_MOTION, NOUN_SIGNAL})
# how notable an event is as the event of a day: at night an opening or a light, an appliance, a hob or an opening
ODD_AT_NIGHT = frozenset({NOUN_DOOR, NOUN_WINDOW, NOUN_MOTION, NOUN_LIGHT})
NOTABLE_NOUNS = frozenset({NOUN_HOB, NOUN_DOOR, NOUN_WINDOW})
PRIORITY_ODD = 3
PRIORITY_APPLIANCE = 2
PRIORITY_NOTABLE = 1
PRIORITY_NONE = 0
DEVICE_CLASS_WINDOW = "window"
WINDOW_WORDS = ("window", "fenster", "raam")
MOMENT_PEAK = "peak"
DEFAULT_BUSIEST_HOUR = 12
KEY_DECIMALS = 2
DAYS_PER_WEEK = 7
MINUTES_PER_HOUR = 60
# the parts of the day by the hour they end, and the hour before which it is night
NIGHT_UNTIL = 4
PART_LIMITS = (
    (8, "dawn"),
    (12, "morning"),
    (14, "noon"),
    (17, "afternoon"),
    (20, "dusk"),
)
PART_NIGHT = "night"
PART_EVENING = "evening"
# how busy a day was: quiet, lived-in or busy
MOOD_QUIET = 0
MOOD_LIVED_IN = 1
MOOD_BUSY = 2
# titles of the inspiring records and artists' songs that must never come out of a word bank
FORBIDDEN_TITLES = frozenset(
    {
        "music for airports",
        "thursday afternoon",
        "discreet music",
        "an ending",
        "small craft on a milk sea",
        "phaedra",
        "rubycon",
        "stratosfear",
        "tangram",
        "sorcerer",
        "force majeure",
        "cyclone",
        "hyperborea",
        "moondawn",
        "timewind",
        "mirage",
        "body love",
        "cyborg",
        "irrlicht",
        "blackdance",
        "piano phase",
        "violin phase",
        "music for 18 musicians",
        "drumming",
        "clapping music",
        "different trains",
        "pendulum music",
        "come out",
        "kaneda",
        "mutation",
        "requiem",
        "illusion",
        "exodus",
        "dolls' polyphony",
        "doll's polyphony",
        "dysphoria",
        "funk blaster",
        "aviation",
        "sensorial",
        "akira",
    }
)


@dataclass(frozen=True)
class DayFacts:
    """What a day was like, in kinds and times only."""

    intensity: float = 0.0
    # the busiest hour of the day, 0 to 23
    busiest_hour: int = DEFAULT_BUSIEST_HOUR
    # the kind of thing the day was about, such as "dishwasher" or "door"
    thing: str = ""
    # how many times that kind of thing did something
    count: int = 0
    # a notable event: its kind and the minute of the day
    event: str = ""
    event_minute: int = 0
    # warm, cool or empty
    climate: str = ""
    weekday: int = 0
    # the day is not over yet
    partial: bool = False

    @property
    def has_data(self) -> bool:
        """Return whether the day has anything to tell."""
        return bool(self.thing)

    @property
    def weekend(self) -> bool:
        """Return whether the day is a Saturday or a Sunday."""
        return self.weekday >= FIRST_WEEKEND_DAY

    @property
    def key(self) -> tuple[object, ...]:
        """Return the facts as a value that tells whether two readings of a day are the same."""
        return (
            round(self.intensity, KEY_DECIMALS),
            self.busiest_hour,
            self.thing,
            self.count,
            self.event,
            self.event_minute,
            self.climate,
            self.weekday,
        )


def facts_of(series: Sequence[Series], day: date, covered: float = DAY_SECONDS) -> DayFacts:
    """
    Read a day: how busy it was and when, what it was about and what stood out.

    :param series: Every series of the room or the house on the day.
    :param day: The day.
    :param covered: How many seconds of the day have happened.
    """
    weekday = day.weekday()
    partial = covered < DAY_SECONDS
    played = [item for item in series if item.kind not in QUIET_KINDS and len(item.times) > 1]
    if not played:
        return DayFacts(weekday=weekday, partial=partial, climate=_climate(series))
    analysis = analyze_day(played, covered)
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    candidates: list[tuple[int, float, str]] = []
    for item in played:
        noun = _noun_of(item)
        if not noun:
            continue
        times = _event_times(item, noun)
        counts[noun] = counts.get(noun, 0) + len(times)
        scores[noun] = scores.get(noun, 0.0) + len(times) * WEIGHTS.get(noun, DEFAULT_WEIGHT)
        for second in times:
            priority = _priority(noun, int(second // SECONDS_PER_HOUR) % HOURS_PER_DAY)
            if not priority:
                continue
            candidates.append((priority, float(second), noun))
    thing = ""
    if scores:
        thing = max(sorted(scores), key=lambda noun: (scores[noun], noun))
        if scores[thing] <= 0.0:
            thing = ""
    event, minute = "", 0
    if candidates:
        top = max(priority for priority, _, _ in candidates)
        best = sorted(c for c in candidates if c[0] == top)
        _, second, event = best[len(best) // 2]
        minute = int(second // SECONDS_PER_MINUTE) % MINUTES_PER_DAY
    peak = analysis.moments_of(MOMENT_PEAK)
    hour = int(peak[0].time // SECONDS_PER_HOUR) % HOURS_PER_DAY if peak else DEFAULT_BUSIEST_HOUR
    return DayFacts(
        analysis.busyness,
        hour,
        thing,
        counts.get(thing, 0),
        event,
        minute,
        _climate(series),
        weekday,
        partial,
    )


def mood_of(facts: DayFacts) -> int:
    """Return how busy the day was: 0 quiet, 1 lived-in or 2 busy."""
    if facts.intensity < QUIET_BELOW:
        return MOOD_QUIET
    return MOOD_LIVED_IN if facts.intensity < BUSY_ABOVE else MOOD_BUSY


def part_of(hour: int) -> str:
    """Return the part of the day an hour is in: night, dawn, morning, noon, afternoon, dusk or evening."""
    if hour >= NIGHT_START or hour < NIGHT_UNTIL:
        return PART_NIGHT
    for limit, name in PART_LIMITS:
        if hour < limit:
            return name
    return PART_EVENING


# -- the words ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleWords:
    """The words of the titles of one style."""

    adjectives: tuple[str, ...]
    nouns: tuple[str, ...]
    templates: tuple[str, ...]
    # titles for a day that has nothing to tell
    plain: tuple[str, ...]
    # what inspired the style, and how the house drives it
    history: str
    drive: str


@dataclass(frozen=True)
class Words:
    """The word banks of one language."""

    # part of the day to its name, and to the name with a preposition
    parts: dict[str, tuple[tuple[str, str], ...]]
    # a kind of thing to its names, each with the article that goes before it
    things: dict[str, tuple[tuple[str, str], ...]]
    moods: tuple[str, str, str]
    weekdays: tuple[str, ...]
    climates: dict[str, str]
    styles: dict[str, StyleWords]
    # the day line and the review: the part of the day, how busy it was, what happened, and the frames of the sentences
    day_parts: dict[str, str]
    day_moods: tuple[str, str, str]
    events: dict[str, str]
    # the weekdays as they are written inside a sentence
    weekday_names: tuple[str, ...]
    no_day: str
    start_whole: str
    start_partial: str
    day_frame: str
    weekend_day_frame: str
    event_frame: str
    review_kind: str
    review_frame: str


EN = Words(
    parts={
        "night": (("Midnight", "at Midnight"), ("Small Hours", "in the Small Hours")),
        "dawn": (("Dawn", "at Dawn"), ("First Light", "at First Light")),
        "morning": (("Morning", "in the Morning"),),
        "noon": (("Noon", "at Noon"), ("Midday", "at Midday")),
        "afternoon": (("Afternoon", "in the Afternoon"),),
        "dusk": (("Dusk", "at Dusk"), ("Twilight", "at Twilight")),
        "evening": (("Evening", "in the Evening"), ("Late Light", "in Late Light")),
    },
    things={
        "dishwasher": (("Dishwasher", "the"), ("Rinse Cycle", "the")),
        "laundry": (("Laundry", "the"), ("Spin Cycle", "the")),
        "hob": (("Hob", "the"), ("Simmer", "the")),
        "door": (("Door", "the"), ("Latch", "the"), ("Hinge", "the")),
        "window": (("Window", "the"), ("Draught", "the")),
        "motion": (("Footsteps", "the"), ("Passage", "the")),
        "light": (("Lamp", "the"), ("Switch", "the")),
        "media": (("Speakers", "the"), ("Stereo", "the")),
        "blinds": (("Blinds", "the"), ("Shutters", "the")),
        "cycle": (("Program", "the"), ("Cycle", "the")),
        "hum": (("Hum", "the"), ("Standby", "the")),
        "signal": (("Signal", "the"), ("Sensor", "the")),
    },
    moods=("Quiet", "Lived-in", "Busy"),
    weekdays=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    climates={"warm": "Warm", "cool": "Cool"},
    styles={
        "ambient": StyleWords(
            adjectives=(
                "Slow",
                "Hushed",
                "Drifting",
                "Pale",
                "Soft",
                "Distant",
                "Unhurried",
                "Quiet",
            ),
            nouns=("Loops", "Drift", "Haze", "Tide", "Air", "Glow"),
            templates=(
                "{thing} {at_part}",
                "{adj} {thing}",
                "{adj} {part}",
                "{part} {noun}",
                "{thing} {noun}",
                "{climate} {part}",
                "{adj} {room}",
                "{weekday} {noun}",
            ),
            plain=("Room Tone", "Slow Light", "Still Air", "Quiet House", "Held Breath"),
            history="Ambient music as an idea goes back to Brian Eno, who made Music for Airports in 1978 from tape loops "
            "of unequal length: each loop returns at its own pace, so the same few sounds meet in a new "
            "arrangement every time and the piece never repeats itself.",
            drive="In this house the loops follow the day: busy hours thicken the notes and brighten the pads, quiet "
            "ones leave a few notes and a lot of air.",
        ),
        "berlin_school": StyleWords(
            adjectives=(
                "Analogue",
                "Pulsing",
                "Sequenced",
                "Hypnotic",
                "Cyclic",
                "Looping",
                "Rolling",
            ),
            nouns=("Sequence", "Arpeggio", "Oscillator", "Pattern"),
            templates=(
                "Sequencer for a {mood} {room}",
                "{adj} {thing} {at_part}",
                "{thing} Sequence",
                "Arpeggio of {the_thing}",
                "{weekday} Sequence",
                "Cycles of {the_thing}",
                "Oscillator {at_part}",
                "Tape Strings {at_part}",
            ),
            plain=(
                "Analogue Sequence",
                "Slow Oscillator",
                "Sequencer for a Quiet House",
                "Tape Strings",
            ),
            history="In the 1970s electronic musicians in West Berlin, Tangerine Dream and Klaus Schulze among them, "
            "built long pieces from analogue sequencers: patterns of repeating notes that mutate slowly under "
            "pads and a lone lead. The drums often wait until well into the piece, as on Schulze's Moondawn, "
            "which makes their entrance feel earned.",
            drive="Here the rooms steer the sequencer: the more happens, the faster the pattern mutates, and "
            "an appliance that runs can bring in the drums.",
        ),
        "lofi": StyleWords(
            adjectives=("Cosy", "Sleepy", "Warm", "Dusty", "Mellow", "Lazy"),
            nouns=("Tape", "Beats", "Blanket", "Static", "Cassette"),
            templates=(
                "{adj} {part}",
                "{thing} Tape",
                "{thing} Beats",
                "Tea with {the_thing}",
                "{weekday} {thing} Beats",
                "{climate} {noun}",
                "{noun} {at_part}",
                "{adj} {room}",
                "{adj} {room} {part}",
            ),
            plain=("Cosy Corner", "Dusty Tape", "Warm Blanket", "Sleepy Beats", "Mellow Hours"),
            history="Lo-fi hip hop grew out of beat makers who liked flaws: drums swung loosely behind the beat, dusty "
            "keys, a round bass and the crackle of old vinyl. It became music for studying, cooking and "
            "winding down.",
            drive="Here the house sets the dust and the swing: lamps and kettles warm the keys, "
            "and a busy hour loosens the drums.",
        ),
        "ritual": StyleWords(
            adjectives=("Lantern", "Silent", "Veiled", "Slow", "Candlelit"),
            nouns=("Procession", "Vigil", "Rite", "Offering", "Gathering", "Ceremony"),
            templates=(
                "Ceremony of {the_thing}",
                "{part} {noun}",
                "Vigil for {the_thing}",
                "Rite of {the_thing}",
                "Offering {at_part}",
                "{adj} {noun}",
                "Bell at {time}",
                "Gathering {at_part}",
            ),
            plain=("Lantern Procession", "Silent Vigil", "Veiled Rite", "The Slow Offering"),
            history="Geinoh Yamashirogumi, the collective behind the Akira soundtrack, mixed folk festival with temple "
            "sounds, and Doll's Polyphony is the mood this style chases: pentatonic plucks, breathing drones and "
            "the weight of taiko drums. It borrows the mood, not the notes.",
            drive="The house breathes into the drone: movement lifts the plucks, and the taiko "
            "answers the busiest hours.",
        ),
        "glitch": StyleWords(
            adjectives=("Fractured", "Stuttering", "Severed", "Corrupt", "Sliced", "Overclocked"),
            nouns=("Overflow", "Reboot", "Static", "Buffer"),
            templates=(
                "{event_thing} Fault {time}",
                "{event_thing} Error {time}",
                "Packet Loss at {time}",
                "{adj} {thing}",
                "{adj} {part}",
                "{thing} {noun}",
                "{thing} Signal Lost",
                "{noun} {at_part}",
            ),
            plain=("Signal Lost", "Buffer Underrun", "Dropped Frames", "Fractured House"),
            history="Cinematic glitch comes from the meeting of film scores and bass music: the half-time drums and "
            "growling basses of neurofunk, as played by acts like KOAN Sound, with swells that sound like "
            "trailers and edits so precise that every stutter reads as a choice.",
            drive="The doors, switches and appliances of the house are the edits: each one cuts the "
            "drums or the bass, and a quiet room leaves the swells alone.",
        ),
    },
    day_parts={
        "night": "night",
        "dawn": "early morning",
        "morning": "morning",
        "noon": "midday",
        "afternoon": "afternoon",
        "dusk": "early evening",
        "evening": "evening",
    },
    day_moods=("quiet", "steady", "busy"),
    events={
        "dishwasher": "a dishwasher run",
        "laundry": "a laundry run",
        "hob": "the hob",
        "door": "a door",
        "window": "a window",
        "motion": "movement",
        "light": "a light",
        "media": "the music",
        "blinds": "the blinds",
        "cycle": "a program change",
        "hum": "a hum",
        "signal": "a signal",
    },
    weekday_names=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    no_day="There is not much to hear of this day yet.",
    start_whole="A",
    start_partial="So far, a",
    day_frame="{start} {mood} {part}{event}.",
    weekend_day_frame="{start} {mood} weekend {part}{event}.",
    event_frame="; {event} at {clock}",
    review_kind=" A {mood} day, liveliest in the {part}.",
    review_frame=(
        "{room} on {weekday} {date}, the day as generative music.{kind} "
        "Every piece is steered by what the house did."
    ),
)

NL = Words(
    parts={
        "night": (("Middernacht", "om Middernacht"), ("Nacht", "in de Nacht")),
        "dawn": (("Dageraad", "bij Dageraad"),),
        "morning": (("Ochtend", "in de Ochtend"),),
        "noon": (("Middag", "rond de Middag"),),
        "afternoon": (("Namiddag", "in de Namiddag"),),
        "dusk": (("Schemering", "bij Schemering"),),
        "evening": (("Avond", "in de Avond"),),
    },
    things={
        "dishwasher": (("Vaatwasser", "de"),),
        "laundry": (("Wasmachine", "de"), ("Centrifuge", "de")),
        "hob": (("Kookplaat", "de"), ("Sudderen", "het")),
        "door": (("Deur", "de"), ("Klink", "de")),
        "window": (("Raam", "het"), ("Tocht", "de")),
        "motion": (("Voetstappen", "de"), ("Gang", "de")),
        "light": (("Lamp", "de"), ("Schakelaar", "de")),
        "media": (("Luidsprekers", "de"), ("Stereo", "de")),
        "blinds": (("Jaloezieën", "de"), ("Luiken", "de")),
        "cycle": (("Programma", "het"), ("Cyclus", "de")),
        "hum": (("Gezoem", "het"), ("Standby", "de")),
        "signal": (("Signaal", "het"), ("Sensor", "de")),
    },
    moods=("Stille", "Bewoonde", "Drukke"),
    weekdays=("Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"),
    climates={"warm": "Warme", "cool": "Koele"},
    styles={
        "ambient": StyleWords(
            adjectives=("Trage", "Gedempte", "Zwevende", "Bleke", "Zachte", "Verre", "Rustige"),
            nouns=("Lussen", "Drift", "Waas", "Getij", "Lucht", "Gloed"),
            templates=(
                "{thing} {at_part}",
                "{adj} {part}",
                "{part}waas",
                "{thing}drift",
                "{weekday}waas",
                "{room} in Gloed",
            ),
            plain=("Kamertoon", "Traag Licht", "Stille Lucht", "Stil Huis", "Ingehouden Adem"),
            history="Ambient als idee gaat terug op Brian Eno, die in 1978 Music for Airports maakte van tapelussen van "
            "ongelijke lengte: elke lus keert in zijn eigen tempo terug, zodat dezelfde paar klanken telkens in "
            "een nieuwe samenstelling samenkomen en het stuk zichzelf nooit herhaalt.",
            drive="In dit huis volgen de lussen de dag: drukke uren maken de noten dichter en de pads helderder, "
            "stille uren laten een paar noten en veel lucht over.",
        ),
        "berlin_school": StyleWords(
            adjectives=("Analoge", "Pulserende", "Hypnotische", "Cyclische", "Rollende"),
            nouns=("Sequentie", "Arpeggio", "Oscillator", "Patroon"),
            templates=(
                "{room} Sequentie",
                "{thing} {at_part}",
                "{thing}sequentie",
                "Arpeggio van {the_thing}",
                "{weekday}sequentie",
                "Cycli van {the_thing}",
                "Oscillator {at_part}",
                "Bandstrijkers {at_part}",
            ),
            plain=("Analoge Sequentie", "Trage Oscillator", "Stille Sequencer", "Bandstrijkers"),
            history="In de jaren zeventig bouwden elektronische muzikanten in West-Berlijn, onder wie Tangerine Dream en "
            "Klaus Schulze, lange stukken op analoge sequencers: patronen van herhaalde noten die langzaam "
            "muteren onder pads en één solo-lijn. De drums komen vaak pas ver in het stuk binnen, zoals op "
            "Schulzes Moondawn, waardoor hun intrede verdiend voelt.",
            drive="Hier sturen de kamers de sequencer: hoe meer er gebeurt, hoe sneller het patroon muteert, "
            "en een apparaat dat draait kan de drums laten binnenkomen.",
        ),
        "lofi": StyleWords(
            adjectives=("Gezellige", "Slaperige", "Warme", "Stoffige", "Zachte"),
            nouns=("Tape", "Beats", "Deken", "Ruis", "Cassette"),
            templates=(
                "{adj} {part}",
                "{thing}tape",
                "{thing}beats",
                "Thee bij {the_thing}",
                "{weekday}beats",
                "{climate} {part}",
                "{noun} {at_part}",
                "{room} Hoekje",
            ),
            plain=(
                "Gezellig Hoekje",
                "Stoffige Tape",
                "Warme Deken",
                "Slaperige Beats",
                "Zachte Uren",
            ),
            history="Lo-fi hiphop groeide uit beatmakers die van imperfectie hielden: losjes achter de tel zwingende "
            "drums, stoffige toetsen, een ronde bas en het geknetter van oud vinyl. Het werd muziek om bij te "
            "studeren, koken en tot rust te komen.",
            drive="Hier bepaalt het huis het stof en de swing: lampen en waterkokers verwarmen de toetsen, "
            "en een druk uur maakt de drums losser.",
        ),
        "ritual": StyleWords(
            adjectives=("Stille", "Gesluierde", "Trage", "Plechtige"),
            nouns=("Processie", "Wake", "Ritus", "Offerande", "Samenkomst", "Ceremonie"),
            templates=(
                "Ceremonie van {the_thing}",
                "{part}processie",
                "Wake bij {the_thing}",
                "Ritus van {the_thing}",
                "Offerande {at_part}",
                "{adj} {noun}",
                "Bel om {time}",
                "Samenkomst {at_part}",
            ),
            plain=("Lantaarnprocessie", "Stille Wake", "Gesluierde Ritus", "De Trage Offerande"),
            history="Geinoh Yamashirogumi, het collectief achter de Akira-soundtrack, mengde volksfeest met tempelklanken, "
            "en Doll's Polyphony is de sfeer die deze stijl zoekt: pentatonische tokkels, ademende drones en het "
            "gewicht van taiko-trommen. Het leent de sfeer, niet de noten.",
            drive="Het huis ademt in de drone: beweging tilt de tokkels op en de taiko "
            "antwoordt op de drukste uren.",
        ),
        "glitch": StyleWords(
            adjectives=("Gebroken", "Stotterende", "Afgesneden", "Corrupte", "Gehakte"),
            nouns=("Overloop", "Herstart", "Ruis", "Buffer"),
            templates=(
                "{event_thing}storing {time}",
                "Glitch bij {event_thing} {time}",
                "Pakketverlies om {time}",
                "{thing}ruis",
                "{adj} {part}",
                "{noun} {at_part}",
                "{thing}: Signaal Weg",
            ),
            plain=("Signaal Weg", "Buffer Leeg", "Gedropte Frames", "Gebroken Huis"),
            history="Cinematic glitch komt voort uit de ontmoeting van filmmuziek en basmuziek: de halftime drums en "
            "grommende bassen van neurofunk, gespeeld door acts als KOAN Sound, met zwellen die klinken als "
            "trailers en edits zo precies dat elke stotter een keuze lijkt.",
            drive="De deuren, schakelaars en apparaten van het huis zijn de edits: elk snijdt in de "
            "drums of de bas, en een stille kamer laat de zwellen met rust.",
        ),
    },
    day_parts={
        "night": "nacht",
        "dawn": "vroege ochtend",
        "morning": "ochtend",
        "noon": "middag",
        "afternoon": "namiddag",
        "dusk": "vroege avond",
        "evening": "avond",
    },
    day_moods=("rustige", "gewone", "drukke"),
    events={
        "dishwasher": "een vaatwasbeurt",
        "laundry": "een wasbeurt",
        "hob": "de kookplaat",
        "door": "een deur",
        "window": "een raam",
        "motion": "beweging",
        "light": "een lamp",
        "media": "de muziek",
        "blinds": "de jaloezieën",
        "cycle": "een programmawissel",
        "hum": "een gezoem",
        "signal": "een signaal",
    },
    weekday_names=("maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"),
    no_day="Van deze dag valt nog weinig te horen.",
    start_whole="Een",
    start_partial="Tot nu toe een",
    day_frame="{start} {mood} {part}{event}.",
    weekend_day_frame="{start} {mood} weekend{part}{event}.",
    event_frame="; {event} om {clock}",
    review_kind=" Een {mood} dag, het levendigst in de {part}.",
    review_frame=(
        "{room} op {weekday} {date}, de dag als generatieve muziek.{kind} "
        "Elk stuk wordt gestuurd door wat het huis deed."
    ),
)

BANKS = {"en": EN, "nl": NL}


def make_title(
    facts: DayFacts | None,
    style_key: str,
    seed: str,
    room: str = "",
    lang: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Return the title of a track, without the style: the same facts, seed and style give the same title.

    :param facts: What the day was like; None, or a day without events, gives a title of the style that does not
        depend on the day.
    :param style_key: The style.
    :param seed: Text that is the same for the same room and day, such as `kitchen|2026-11-17`.
    :param room: The name of the room; it is used when it is short.
    :param lang: `en` or `nl`; any other language gives English.
    """
    words = BANKS.get(lang, EN)
    style = words.styles.get(style_key, words.styles["ambient"])
    rng = _rng(f"{seed}|{style_key}", "title")
    if facts is None or not facts.has_data:
        return rng.choice(style.plain)
    slots = _slots(words, facts, rng, room, style)
    order = list(style.templates)
    rng.shuffle(order)
    for template in order:
        if not _fields(template) <= slots.keys():
            continue
        title = " ".join(template.format(**slots).split())
        if len(title) <= MAX_TITLE and title.lower() not in FORBIDDEN_TITLES:
            return title
    return rng.choice(style.plain)


# -- the text of the styles -----------------------------------------------------------------------


def day_line(facts: DayFacts | None, lang: str = DEFAULT_LANGUAGE) -> str:
    """
    Return one line about the day, such as: A busy afternoon; a door at 03:12.

    :param facts: What the day was like.
    :param lang: `en` or `nl`.
    """
    words = BANKS.get(lang, EN)
    if facts is None or not facts.has_data:
        return words.no_day
    event = ""
    if facts.event in words.events:
        event = words.event_frame.format(
            event=words.events[facts.event], clock=_clock_of(facts.event_minute)
        )
    frame = words.weekend_day_frame if facts.weekend else words.day_frame
    return frame.format(
        start=words.start_partial if facts.partial else words.start_whole,
        mood=words.day_moods[mood_of(facts)],
        part=words.day_parts[part_of(facts.busiest_hour)],
        event=event,
    )


def track_description(style_key: str, facts: DayFacts | None, lang: str = DEFAULT_LANGUAGE) -> str:
    """
    Return the description of a track: what inspired its style, how the house drives it, and the day.

    :param style_key: The style.
    :param facts: What the day was like.
    :param lang: `en` or `nl`.
    """
    words = BANKS.get(lang, EN)
    style = words.styles.get(style_key, words.styles[DEFAULT_STYLE])
    return f"{style.history} {style.drive} {day_line(facts, lang)}"


def album_review(room: str, day: date, facts: DayFacts | None, lang: str = DEFAULT_LANGUAGE) -> str:
    """
    Return the short review of an album: the room, the day and what kind of day it was.

    :param room: The name of the room, or of the whole house.
    :param day: The day.
    :param facts: What the day was like.
    :param lang: `en` or `nl`.
    """
    words = BANKS.get(lang, EN)
    kind = ""
    if facts is not None and facts.has_data:
        kind = words.review_kind.format(
            mood=words.day_moods[mood_of(facts)], part=words.day_parts[part_of(facts.busiest_hour)]
        )
    return words.review_frame.format(
        room=room, weekday=words.weekday_names[day.weekday()], date=day.isoformat(), kind=kind
    )


def _noun_of(item: Series) -> str:
    """Return the kind of thing a series is, in words that name no one and nothing in particular."""
    kind = item.kind
    simple = {
        VoiceKind.DISHWASHER: NOUN_DISHWASHER,
        VoiceKind.DRUM: NOUN_LAUNDRY,
        VoiceKind.HOB: NOUN_HOB,
        VoiceKind.DRONE: NOUN_HUM,
        VoiceKind.CHORD: NOUN_LIGHT,
        VoiceKind.MEDIA: NOUN_MEDIA,
        VoiceKind.COVER: NOUN_BLINDS,
        VoiceKind.PHASE: NOUN_CYCLE,
        VoiceKind.BLIP: NOUN_SIGNAL,
    }
    if kind in simple:
        return simple[kind]
    if kind == VoiceKind.PLUCK:
        text = f"{item.entity_id} {item.name}".lower()
        if item.device_class == DEVICE_CLASS_WINDOW or any(w in text for w in WINDOW_WORDS):
            return NOUN_WINDOW
        if item.is_opening:
            return NOUN_DOOR
        return NOUN_MOTION if item.is_motion else NOUN_SIGNAL
    return ""


def _event_times(item: Series, noun: str) -> np.ndarray:
    """Return when something of a kind happened: appliance runs, hob and cycle changes, openings."""
    if noun in APPLIANCES:
        return item.crossings(0.0, DAY_SECONDS, RUN_LEVEL, rising=True)
    if noun in CHANGING_NOUNS:
        return item.changes(0.0, DAY_SECONDS, CYCLE_CHANGE)
    if noun == NOUN_HUM:
        return np.empty(0)
    if item.kind in BINARY_KINDS or noun in OPENING_NOUNS:
        return item.onsets(0.0, DAY_SECONDS)
    return item.changes(0.0, DAY_SECONDS, OTHER_CHANGE)


def _climate(series: Sequence[Series]) -> str:
    """Return warm or cool when the temperature of the day says so."""
    values = [float(item.values.mean()) for item in series if item.kind == VoiceKind.TEMPERATURE]
    if not values:
        return ""
    mean = sum(values) / len(values)
    if mean > FAHRENHEIT_ABOVE:
        mean = (mean - FAHRENHEIT_OFFSET) / FAHRENHEIT_PER_CELSIUS
    if mean >= WARM_C:
        return CLIMATE_WARM
    return CLIMATE_COOL if mean <= COOL_C else ""


def _priority(noun: str, hour: int) -> int:
    """Return how notable an event is as the event of a day: 3 odd, 2 an appliance, 1 a hob or a door, 0 not."""
    if (hour >= NIGHT_START or hour < NIGHT_END) and noun in ODD_AT_NIGHT:
        return PRIORITY_ODD
    if noun in APPLIANCES:
        return PRIORITY_APPLIANCE
    return PRIORITY_NOTABLE if noun in NOTABLE_NOUNS else PRIORITY_NONE


def _fields(template: str) -> set[str]:
    """Return the names of the slots of a template."""
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def _slots(
    words: Words,
    facts: DayFacts,
    rng: random.Random,
    room: str,
    style: StyleWords,
) -> dict[str, str]:
    """Return the words that a title of a day can use; a slot with nothing to say is missing."""
    slots: dict[str, str] = {
        "adj": rng.choice(style.adjectives),
        "noun": rng.choice(style.nouns),
        "weekday": words.weekdays[facts.weekday % DAYS_PER_WEEK],
    }
    bare, at = rng.choice(words.parts[part_of(facts.busiest_hour)])
    slots["part"], slots["at_part"] = bare, at
    slots["mood"] = words.moods[mood_of(facts)]
    if room and len(room) <= MAX_ROOM:
        slots["room"] = room
    if facts.thing in words.things:
        noun, article = rng.choice(words.things[facts.thing])
        slots["thing"] = noun
        slots["the_thing"] = f"{article} {noun}"
    if facts.event in words.things:
        slots["event_thing"] = words.things[facts.event][0][0]
        slots["time"] = _clock_of(facts.event_minute)
    if facts.climate in words.climates:
        slots["climate"] = words.climates[facts.climate]
    return slots


def _clock_of(minute: int) -> str:
    """Return a minute of the day as a clock time, such as 03:12."""
    return f"{minute // MINUTES_PER_HOUR:02d}:{minute % MINUTES_PER_HOUR:02d}"


def _rng(seed: str, salt: str) -> random.Random:
    """Return a generator that is the same for the same text."""
    return random.Random(stable_hash(salt, seed))  # noqa: S311 - not for security
