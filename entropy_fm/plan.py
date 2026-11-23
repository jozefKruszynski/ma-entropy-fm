"""
The plan of a soundscape: how long it is, how the day maps onto it, and the curves that the house steers.

A soundscape is one slowly evolving piece of 5 to 7 minutes for one room on one day. The day (night, morning, daytime,
evening, night) is spread over the piece in order, so the piece follows the day: sparse and dark when the
house sleeps, warmer and denser when it is busy. The house does not write notes. It gives curves, a value
every tenth of a second of the piece:

| Curve | Comes from | Steers |
|---|---|---|
| activity | how lively the house is (`day_analysis`) | how many layers play, and how dense they are |
| power | the appliances that draw power | bass and drone presence, the cutoff and accent of the sequencer |
| bright | temperature, the sun and the time | brightness (the cutoffs) and warmth |
| humid | humidity | reverb and space |
| lights | lights and the evening | the harmonic colour (which chords and sets are drawn) |
| runs | a dishwasher, a washer, the hob | a layer that fades in and out while the appliance runs |
| events | anomalies | a distinct texture event: a filter dip, a swell, a harmonic shift |

The day also deals the material: which motifs, progression and voices a day uses, drawn from a
library, so that two days in a row never get the same.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from .character import FLAT, MOMENT_ANOMALY, MOMENT_SHOWER, NEUTRAL, PEAK, DayCharacter, room_motif
from .day_analysis import STEP, DayAnalysis
from .score import CURVE_RATE, curve_at
from .styles import STYLE_AMBIENT, STYLE_BERLIN_SCHOOL, STYLE_GLITCH, STYLE_LOFI, STYLE_RITUAL
from .synth import Scale, smoothstep
from .timeline import (
    BINARY_KINDS,
    DAY_SECONDS,
    HOURS_PER_DAY,
    POWER_KINDS,
    SECONDS_PER_HOUR,
    SEED_MODULUS,
    Series,
    VoiceKind,
    stable_hash,
)
from .variation import variant_of

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from .palette import DayPalette
    from .quality import Quality

T = TypeVar("T")
# a piece is as long as the style needs to say something: slow styles take 7 minutes, fast ones 5
CONDENSED_SECONDS = {
    STYLE_AMBIENT: 420,
    STYLE_BERLIN_SCHOOL: 420,
    STYLE_RITUAL: 420,
    STYLE_LOFI: 300,
    STYLE_GLITCH: 300,
}
CONDENSED_DEFAULT_SECONDS = 300
# the activity and the lights of a piece change over at least this many seconds, so that a build is a build
MIN_SMOOTH_SECONDS = 12.0
# unusual moments keep this many seconds apart, and the little things (a switch, a door) are thinned to one
# every few seconds, the rarer kinds first
CONDENSED_EVENT_GAP = 25.0
CONDENSED_TRIGGER_GAP = 6.0
# the things that happen in the house, as the kinds of a trigger
TRIGGER_DOOR = "door"
TRIGGER_MOTION = "motion"
TRIGGER_LIGHT = "light"
TRIGGER_APPLIANCE = "appliance"
TRIGGER_PHASE = "phase"
TRIGGER_HOB = "hob"
TRIGGER_COVER = "cover"
TRIGGER_MEDIA = "media"
TRIGGER_BLIP = "blip"
TRIGGER_PRIORITY = {
    TRIGGER_APPLIANCE: 0,
    TRIGGER_HOB: 0,
    TRIGGER_PHASE: 1,
    TRIGGER_DOOR: 1,
    TRIGGER_COVER: 2,
    TRIGGER_MEDIA: 2,
    TRIGGER_LIGHT: 3,
    TRIGGER_MOTION: 3,
    TRIGGER_BLIP: 4,
}
# a trigger of a kind that is not in the table is thinned last
UNLISTED_TRIGGER_PRIORITY = 5
# the strength of a trigger, which has none of its own
TRIGGER_STRENGTH = 0.5
# a trigger sits inside the piece, this far from its start and its end in seconds
TRIGGER_MARGIN_START = 3.0
TRIGGER_MARGIN_END = 5.0
# the level at which an appliance counts as started or stopped, and the change of a series that counts as an event
APPLIANCE_LEVEL = 0.2
BINARY_CHANGE = 0.5
ANALOG_CHANGE = 0.02
# values are ranked at this many decimals, so that rounding noise between platforms never reorders them
RANK_DECIMALS = 9
# the arc: where the five stretches of the day sit in the piece, as shares of its length
# how much of the activity of a day that is measured against its own days is the curve itself, and how much its rank
MEASURED_CURVE = 0.5
# showers: a couple at most in a piece; with the anomalies at most four texture moments
MAX_SHOWERS = 2
MAX_EVENTS = 4
MAX_ANOMALY_EVENTS = 3
# a texture moment sits inside the piece, this far from its start and its end in seconds
EVENT_MARGIN_START = 4.0
EVENT_MARGIN_END = 6.0
# a shower is a texture moment a little softer than an anomaly, and neither is ever inaudible or all of it
SHOWER_SOFTENING = 0.8
SHOWER_MIN_STRENGTH = 0.25
SHOWER_MAX_STRENGTH = 0.85
ANOMALY_MIN_STRENGTH = 0.2
# a curve is smoothed over this many of its values at least, and a piece over a fortieth of its length at least
MIN_SMOOTH_VALUES = 3
SMOOTH_LENGTH_DIVISOR = 40.0
# the activity: the curve of the day to the power, and the rank, weighted; the day is read for at least an hour
MEASURED_EXPONENT = 0.7
UNMEASURED_EXPONENT = 0.6
UNMEASURED_CURVE = 0.5
UNMEASURED_RANK = 0.5
MIN_COVERED_SECONDS = SECONDS_PER_HOUR
# the windows in seconds that the series are averaged over for each curve
POWER_WINDOW = 900.0
SUN_WINDOW = 900.0
CLIMATE_WINDOW = 1800.0
LIGHT_WINDOW = 600.0
RUN_WINDOW = 240.0
# power: from this level of the loudest appliance, over this span, is full presence
POWER_FLOOR = 0.5
POWER_SPAN = 0.4
# the daylight: the hours of sunrise and sunset, and how long each takes; the sun sensor shares the say
SUNRISE_HOUR = 5.5
SUNRISE_HOURS = 3.0
SUNSET_HOUR = 17.0
SUNSET_HOURS = 3.5
DAYLIGHT_CLOCK_SHARE = 0.5
DAYLIGHT_SUN_SHARE = 0.5
# the brightness: shares of the daylight, the temperature and the activity; a room without a sensor reads mid
BRIGHT_DAYLIGHT_SHARE = 0.5
BRIGHT_TEMPERATURE_SHARE = 0.3
BRIGHT_ACTIVITY_SHARE = 0.2
DEFAULT_TEMPERATURE = 0.5
DEFAULT_HUMIDITY = 0.4
# the evening rises from this hour over this many hours, and is gone by the next hour over a short fall
EVENING_RISE_HOUR = 16.5
EVENING_RISE_HOURS = 3.0
EVENING_FALL_HOUR = 23.0
EVENING_FALL_HOURS = 1.0
# the lights: shares of what is switched on and of the evening; the night is what the day and the evening leave
LIGHTS_SWITCHED_SHARE = 0.6
LIGHTS_EVENING_SHARE = 0.4
NIGHT_EVENING_SHARE = 0.6
# a run is on above this level of its appliances (the hob lower), eased in over this range of level
RUN_THRESHOLD = 0.12
COOK_THRESHOLD = 0.05
RUN_RAMP = 0.18
# the dealer only breaks ties between options whose fit is within this of the best
PICK_TOLERANCE = 0.15
# the tempo of a piece is picked tighter: a few beats per minute of swing is plenty
TEMPO_TOLERANCE = 0.1
# the most layers of a tier that is light on the machine
LIGHT_MAX_LAYERS = 4
# the form of a piece: sections snap to this many bars, or half of it when the piece has too few of them
FORM_BARS = 8
SHORT_FORM_BARS = 4
MIN_FORM_CELLS = 10
# the smallest width of a gate, and the fewest seconds of a day that a second of the piece is measured against
MIN_GATE_WIDTH = 1e-6
MIN_COVERED_SPAN = 1.0
# the climax of a day with one peak sits in this share of the piece
PEAK_AT = (0.45, 0.72)
PEAK_WIDTH = 0.15
INTRO, BUILD, CLIMAX, RELEASE, PLATEAU, OUTRO = (
    "intro",
    "build",
    "peak",
    "release",
    "plateau",
    "outro",
)
# how loud a role of the form is at its design level; the energy of the day scales it
ROLE_INTENSITY = {
    INTRO: 0.25,
    BUILD: 0.55,
    CLIMAX: 1.0,
    RELEASE: 0.5,
    PLATEAU: 0.65,
    OUTRO: 0.2,
}
# the energy of a day sets this share of the intensity of a section, and the rest is the role
FORM_ENERGY_SHARE = 0.4
# the form of each shape of day, as (role, share of the piece); the peak is placed from the day itself
FORMS = {
    "morning": (
        (INTRO, 0.08),
        (BUILD, 0.17),
        (CLIMAX, 0.2),
        (RELEASE, 0.35),
        (OUTRO, 0.2),
    ),
    "evening": (
        (INTRO, 0.28),
        (BUILD, 0.32),
        (CLIMAX, 0.22),
        (RELEASE, 0.1),
        (OUTRO, 0.08),
    ),
    "all_day": (
        (INTRO, 0.08),
        (BUILD, 0.12),
        (CLIMAX, 0.22),
        (PLATEAU, 0.33),
        (RELEASE, 0.15),
        (OUTRO, 0.1),
    ),
    "flat": ((INTRO, 0.1), (PLATEAU, 0.8), (OUTRO, 0.1)),
}
PEAK_FORM_BUILD = 0.25
PEAK_FORM_OUTRO = 0.1
ARC_NAMES = ("night", "morning", "daytime", "evening", "late night")
ARC_EDGES = (0.0, 6.0, 11.0, 17.0, 22.0, 24.0)
RUN_DISH = "dish"
RUN_WASH = "wash"
RUN_COOK = "cook"
RUN_KINDS = {
    RUN_DISH: frozenset({VoiceKind.DISHWASHER}),
    RUN_WASH: frozenset({VoiceKind.DRUM}),
    RUN_COOK: frozenset({VoiceKind.HOB, VoiceKind.PHASE}),
}


def condensed_seconds(style: str) -> int:
    """
    Return the length in seconds of the piece of a style, which does not depend on the day.

    :param style: The style key.
    """
    return CONDENSED_SECONDS.get(style, CONDENSED_DEFAULT_SECONDS)


@dataclass
class Plan:
    """The curves and the deals of one soundscape."""

    style: str
    seed: str
    length: int
    covered: float
    scale: Scale
    quality: Quality
    analysis: DayAnalysis
    day: date | None
    # every curve has CURVE_RATE values per second
    hour: np.ndarray
    activity: np.ndarray
    power: np.ndarray
    bright: np.ndarray
    humid: np.ndarray
    lights: np.ndarray
    evening: np.ndarray
    daylight: np.ndarray
    night: np.ndarray
    runs: dict[str, np.ndarray]
    # unusual moments as (second of the piece, strength 0 to 1)
    events: list[tuple[float, float]]
    palette: DayPalette | None = None
    area_key: str = ""
    character: DayCharacter = NEUTRAL
    choices: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # things that happened in the house as (second of the piece, strength 0 to 1, kind), the kinds being door, motion,
    # light, appliance, phase, hob, cover, media and blip: a switch, a door, an appliance starting or stopping
    triggers: list[tuple[float, float, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Return the number of values of each curve."""
        return len(self.hour)

    def rng(self, *parts: object) -> np.random.Generator:
        """Return a random generator that is the same every time for this piece and these parts."""
        return np.random.default_rng(stable_hash(self.seed, *parts) % SEED_MODULUS)

    def pick(
        self,
        component: str,
        options: Sequence[T],
        fit: Sequence[float],
        tolerance: float = PICK_TOLERANCE,
    ) -> T:
        """
        Pick the option that fits the day best; the dealer only breaks ties among those within tolerance of the best.

        :param component: What is being chosen, such as the tempo of the piece.
        :param options: The family to choose from.
        :param fit: How well each option fits the character of the day, higher is better; one per option.
        :param tolerance: How close to the best fit an option has to be to count as equally good.
        """
        if len(options) != len(fit) or not options:
            raise ValueError("pick needs one fit for every option, and at least one option")
        best = max(fit)
        tied = [index for index, value in enumerate(fit) if value >= best - tolerance]
        return self._record(component, options[tied[self._turn(component, len(tied))]])

    def motif(self) -> tuple[tuple[int, float], ...]:
        """Return the motif of the room as (scale degree, beats), which makes the room recognisable across days."""
        return room_motif(self.area_key)

    def at(self, curve: np.ndarray, second: float) -> float:
        """Return the value of a curve at a second of the piece."""
        return curve_at(curve, second)

    def lift(self, curve: np.ndarray, floor: float) -> np.ndarray:
        """
        Return a 0 to 1 curve that never falls below a floor: the data shapes what is above it.

        This is what keeps an arrangement full: a layer that is part of the base plays at its floor on a calm day,
        and the house only makes it louder, denser or brighter.

        :param curve: A curve between 0 and 1.
        :param floor: The least value, 0 to 1.
        """
        return np.asarray(floor + (1.0 - floor) * np.clip(curve, 0.0, 1.0))

    def gate(self, curve: np.ndarray, low: float, high: float) -> np.ndarray:
        """Return a curve as a smooth 0 to 1 gate: 0 below `low`, 1 above `high`."""
        return np.asarray(smoothstep((curve - low) / max(high - low, MIN_GATE_WIDTH)))

    def second_of_day(self, second: float) -> float:
        """Return the second of the day that a second of the piece stands for."""
        return self.covered * second / self.length

    def piece_second_of_hour(self, hour: float) -> float:
        """Return the second of the piece at which the clock of the day says `hour`."""
        return float(
            np.clip(
                hour * SECONDS_PER_HOUR / max(self.covered, MIN_COVERED_SPAN) * self.length,
                0.0,
                self.length,
            )
        )

    def sections(self, bar: float) -> list[tuple[float, float, str, float]]:
        """
        Return the form of the piece as (start second, end second, role, intensity 0 to 1).

        The roles are intro, build, peak, release, plateau and outro. The edges lie on multiples of 8 bars (4 when the
        piece is short), a build is always followed by a peak or a release, and the shape of the day sets the form: a
        morning day builds early and fades, an evening day builds late, a day with one peak has its climax where the
        day had it, and a flat day is a plateau.

        :param bar: The length of a bar in seconds.
        """
        grid = self.form_grid(bar)
        cells = max(round(self.length / grid), 1)
        items: list[tuple[str, int]] = []
        done = 0.0
        taken = 0
        for role, share in self._form():
            done += share
            edge = round(done * cells)
            if edge > taken:
                items.append((role, edge - taken))
                taken = edge
        items = _resolve_builds(items)
        scale = (1.0 - FORM_ENERGY_SHARE) + 2.0 * FORM_ENERGY_SHARE * self.character.energy
        found: list[tuple[float, float, str, float]] = []
        first = 0
        for index, (role, count) in enumerate(items):
            last = first + count
            end = float(self.length) if index == len(items) - 1 else last * grid
            found.append(
                (first * grid, end, role, float(np.clip(ROLE_INTENSITY[role] * scale, 0.0, 1.0)))
            )
            first = last
        return found

    def form_grid(self, bar: float) -> float:
        """
        Return the length in seconds of a phrase of the form: 8 bars, 4 when the piece is short.

        :param bar: The length of a bar in seconds.
        """
        grid = FORM_BARS * bar
        return SHORT_FORM_BARS * bar if self.length / grid < MIN_FORM_CELLS else grid

    def arc(self) -> list[tuple[str, float, float]]:
        """Return the five stretches of the day as (name, start second, end second) of the piece."""
        marks = [self.piece_second_of_hour(h) for h in ARC_EDGES]
        return [
            (ARC_NAMES[i], marks[i], marks[i + 1])
            for i in range(len(ARC_NAMES))
            if marks[i + 1] > marks[i]
        ]

    def _turn(self, component: str, count: int) -> int:
        """Return which of `count` members of a family plays: dealt in rounds over the days, seeded by the room."""
        if self.day is None:
            return int(self.rng("deal", component).integers(0, count))
        return variant_of(f"{self.style}|{self.area_key}|{component}", self.day, count)

    def _record(self, component: str, chosen: T) -> T:
        """Note what was chosen for a component, so that the log and the descriptions can say it."""
        label = (
            chosen[0]
            if isinstance(chosen, tuple) and chosen and isinstance(chosen[0], str)
            else chosen
        )
        self.choices[component] = str(getattr(label, "key", label))
        return chosen

    def _form(self) -> tuple[tuple[str, float], ...]:
        """Return the roles of the piece in order with their share of it, from the shape of the day."""
        shape = self.character.shape
        if shape != PEAK:
            return FORMS.get(shape, FORMS[FLAT])
        at = float(
            np.clip(
                self.piece_second_of_hour(self.character.peak * HOURS_PER_DAY) / self.length,
                *PEAK_AT,
            )
        )
        start, end = at - PEAK_WIDTH / 2.0, at + PEAK_WIDTH / 2.0
        return (
            (INTRO, start - PEAK_FORM_BUILD),
            (BUILD, PEAK_FORM_BUILD),
            (CLIMAX, PEAK_WIDTH),
            (RELEASE, 1.0 - PEAK_FORM_OUTRO - end),
            (OUTRO, PEAK_FORM_OUTRO),
        )


@dataclass(frozen=True)
class PieceOptions:
    """
    The choices that shape a piece beyond the data of its day.

    :param day: The day, which deals the material.
    :param palette: What the day derives (key, mode, density).
    :param area_key: The room, which seeds the deals.
    :param character: What kind of day it was (`character.measure_character`); without it the day is neutral.
    """

    day: date | None = None
    palette: DayPalette | None = None
    area_key: str = ""
    character: DayCharacter | None = None


DEFAULT_OPTIONS = PieceOptions()


def build_plan(
    series: Sequence[Series],
    analysis: DayAnalysis,
    style: str,
    seed: str,
    scale: Scale,
    quality: Quality,
    *,
    options: PieceOptions = DEFAULT_OPTIONS,
) -> Plan:
    """
    Make the plan of a soundscape from the data of a day.

    :param series: Every series of the room or the house and the sun, on the day.
    :param analysis: The analysis of the day (`day_analysis.analyze_day`).
    :param style: The style key.
    :param seed: Text that is the same for the same room, day and style.
    :param scale: The key and mode of the piece.
    :param quality: The tier of the engine.
    :param options: The day, palette, room and character of the piece.
    """
    length = condensed_seconds(style)
    t = (np.arange(length * CURVE_RATE) + 0.5) / CURVE_RATE
    covered = float(min(max(analysis.covered, MIN_COVERED_SECONDS), DAY_SECONDS))
    house = covered * t / length
    hour = house / SECONDS_PER_HOUR
    smooth = max(int(max(length / SMOOTH_LENGTH_DIVISOR, MIN_SMOOTH_SECONDS) * CURVE_RATE / 2), 1)

    activity = _activity_curve(analysis, house, smooth)
    power, daylight, bright, humid = _environment_curves(series, house, hour, activity, smooth)
    lights, evening = _light_curves(series, house, hour, smooth)
    runs = _run_curves(series, house, smooth)
    events, triggers = _events_and_triggers(series, analysis, covered, length)

    return Plan(
        style=style,
        seed=seed,
        length=length,
        covered=covered,
        scale=scale,
        quality=quality,
        analysis=analysis,
        day=options.day,
        hour=hour,
        activity=activity,
        power=power,
        bright=bright,
        humid=humid,
        lights=lights,
        evening=evening,
        daylight=daylight,
        night=1.0 - np.maximum(daylight, evening * NIGHT_EVENING_SHARE),
        runs=runs,
        events=events,
        palette=options.palette,
        area_key=options.area_key,
        character=options.character or NEUTRAL,
        triggers=triggers,
    )


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Smooth a curve with a moving average of `width` values, keeping its ends."""
    if width <= 1 or len(values) < MIN_SMOOTH_VALUES:
        return values
    pad = np.concatenate((np.full(width, values[0]), values, np.full(width, values[-1])))
    kernel = np.hanning(width * 2 + 1)
    return np.asarray(np.convolve(pad, kernel / kernel.sum(), mode="same")[width:-width])


def _resolve_builds(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Make every build end in a peak or a release: a build that does not is taken into the section before it."""
    found = list(items)
    index = 0
    while index < len(found):
        role, count = found[index]
        follower = found[index + 1][0] if index + 1 < len(found) else ""
        if role == BUILD and follower not in {CLIMAX, RELEASE}:
            target = index - 1 if index > 0 else index + 1
            if target >= len(found):
                break
            found[target] = (found[target][0], found[target][1] + count)
            del found[index]
            index = max(index - 1, 0)
            continue
        index += 1
    merged: list[tuple[str, int]] = []
    for role, count in found:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + count)
        else:
            merged.append((role, count))
    return merged


def _daylight(hour: np.ndarray) -> np.ndarray:
    """Return how light it is at each hour of the day, from sunrise to sunset."""
    rise = smoothstep((hour - SUNRISE_HOUR) / SUNRISE_HOURS)
    fall = 1.0 - smoothstep((hour - SUNSET_HOUR) / SUNSET_HOURS)
    return np.asarray(rise * fall)


def _trigger_kind(item: Series) -> str:
    """Name what kind of happening a series stands for."""
    if item.is_opening:
        return TRIGGER_DOOR
    if item.is_motion:
        return TRIGGER_MOTION
    return {
        VoiceKind.CHORD: TRIGGER_LIGHT,
        VoiceKind.DRUM: TRIGGER_APPLIANCE,
        VoiceKind.DISHWASHER: TRIGGER_APPLIANCE,
        VoiceKind.PHASE: TRIGGER_PHASE,
        VoiceKind.HOB: TRIGGER_HOB,
        VoiceKind.COVER: TRIGGER_COVER,
        VoiceKind.MEDIA: TRIGGER_MEDIA,
        VoiceKind.BLIP: TRIGGER_BLIP,
    }.get(item.kind, "")


def _triggers(
    series: Sequence[Series], covered: float, length: int
) -> list[tuple[float, float, str]]:
    """Return the switches, doors and appliance starts of the day as (second of the piece, strength, kind), in order."""
    found: list[tuple[float, float, str]] = []
    for item in series:
        kind = _trigger_kind(item)
        if not kind:
            continue
        if item.kind in {VoiceKind.DRUM, VoiceKind.DISHWASHER}:
            times = np.concatenate(
                (
                    item.crossings(0.0, covered, APPLIANCE_LEVEL, rising=True),
                    item.crossings(0.0, covered, APPLIANCE_LEVEL, rising=False),
                )
            )
        else:
            times = item.changes(
                0.0, covered, BINARY_CHANGE if item.kind in BINARY_KINDS else ANALOG_CHANGE
            )
        found.extend(
            (float(t / covered * length), TRIGGER_STRENGTH, kind)
            for t in times
            if TRIGGER_MARGIN_START <= t / covered * length < length - TRIGGER_MARGIN_END
        )
    return sorted(found)


def _thin_events(events: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Keep the strongest unusual moments that are at least `gap` seconds apart, at most three, in order."""
    kept: list[tuple[float, float]] = []
    for second, strength in sorted(events, key=lambda item: (-item[1], item[0])):
        if all(abs(second - other) >= gap for other, _ in kept):
            kept.append((second, strength))
    return sorted(kept)[:MAX_ANOMALY_EVENTS]


def _thin_triggers(
    triggers: list[tuple[float, float, str]], gap: float
) -> list[tuple[float, float, str]]:
    """Keep at most one switch, door or appliance every `gap` seconds: appliances and doors before lights and motion."""
    kept: list[tuple[float, float, str]] = []
    for item in sorted(
        triggers, key=lambda t: (TRIGGER_PRIORITY.get(t[2], UNLISTED_TRIGGER_PRIORITY), t[0])
    ):
        if all(abs(item[0] - other[0]) >= gap for other in kept):
            kept.append(item)
    return sorted(kept)


def _inside_event_margins(second: float, length: int) -> bool:
    """Return True when a second is far enough from the start and the end of the piece for a texture moment."""
    return EVENT_MARGIN_START <= second < length - EVENT_MARGIN_END


def _with_showers(
    events: list[tuple[float, float]],
    analysis: DayAnalysis,
    covered: float,
    length: int,
) -> list[tuple[float, float]]:
    """
    Add the showers (a spike of humidity well over what a room usually reads) to the unusual moments.

    A shower is a texture moment of its own, a little softer than an anomaly: at most `MAX_SHOWERS` of them, apart
    from each other and from the anomalies, and `MAX_EVENTS` moments in all, the strongest kept.

    :param events: The unusual moments as (second of the piece, strength).
    :param analysis: The analysis of the day, which found the showers.
    :param covered: The seconds of the day that happened.
    :param length: The length of the piece in seconds.
    """
    found: list[tuple[float, float]] = []
    for moment in sorted(analysis.moments_of(MOMENT_SHOWER), key=lambda m: (-m.strength, m.time)):
        audio = moment.time / covered * length
        strength = float(
            np.clip(SHOWER_SOFTENING * moment.strength, SHOWER_MIN_STRENGTH, SHOWER_MAX_STRENGTH)
        )
        if _inside_event_margins(audio, length) and all(
            abs(audio - other) >= CONDENSED_EVENT_GAP for other, _ in [*events, *found]
        ):
            found.append((float(audio), strength))
        if len(found) == MAX_SHOWERS:
            break
    merged = sorted([*events, *found], key=lambda item: (-item[1], item[0]))[:MAX_EVENTS]
    return sorted(merged)


def _events_and_triggers(
    series: Sequence[Series],
    analysis: DayAnalysis,
    covered: float,
    length: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, str]]]:
    """Return the unusual moments and the things that happened, as seconds of the piece."""
    events = []
    for moment in analysis.moments_of(MOMENT_ANOMALY):
        audio = moment.time / covered * length
        if _inside_event_margins(audio, length):
            events.append(
                (float(audio), float(np.clip(moment.strength, ANOMALY_MIN_STRENGTH, 1.0)))
            )
    events = _with_showers(_thin_events(events, CONDENSED_EVENT_GAP), analysis, covered, length)
    triggers = _thin_triggers(_triggers(series, covered, length), CONDENSED_TRIGGER_GAP)
    return events, triggers


def _activity_curve(analysis: DayAnalysis, house: np.ndarray, smooth: int) -> np.ndarray:
    """Return the activity of the piece: the curve of the day mixed with its rank, so a day has a quiet and a lively half."""
    centres = (np.arange(len(analysis.curve)) + 0.5) * STEP
    ranks = _ranks(analysis.curve)
    if analysis.measured:
        # measured against its own days the curve already says how lively this day was: the rank is a smaller part
        mixed = MEASURED_CURVE * analysis.curve**MEASURED_EXPONENT + (1.0 - MEASURED_CURVE) * ranks
    else:
        mixed = UNMEASURED_CURVE * analysis.curve**UNMEASURED_EXPONENT + UNMEASURED_RANK * ranks
    activity = np.interp(house, centres, mixed)
    return np.clip(_smooth(activity, smooth), 0.0, 1.0)


def _combine(
    series: Sequence[Series], house: np.ndarray, kinds: frozenset[VoiceKind], window: float
) -> np.ndarray | None:
    """Return the highest of the series of some kinds at each second of the day, or None when there are none."""
    found = [item for item in series if item.kind in kinds and not item.is_baseline_load]
    if not found:
        return None
    return np.max([item.at(house, window) for item in found], axis=0)


def _environment_curves(
    series: Sequence[Series],
    house: np.ndarray,
    hour: np.ndarray,
    activity: np.ndarray,
    smooth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the power, the daylight, the brightness and the humidity of the piece."""
    count = len(house)
    power_raw = _combine(series, house, POWER_KINDS, POWER_WINDOW)
    power = (
        np.zeros(count)
        if power_raw is None
        else np.clip((power_raw - POWER_FLOOR) / POWER_SPAN, 0.0, 1.0)
    )
    power = _smooth(power, smooth)

    daylight = _daylight(hour)
    sun = _combine(series, house, frozenset({VoiceKind.SUN}), SUN_WINDOW)
    if sun is not None:
        daylight = DAYLIGHT_CLOCK_SHARE * daylight + DAYLIGHT_SUN_SHARE * np.clip(sun, 0.0, 1.0)
    temp = _combine(series, house, frozenset({VoiceKind.TEMPERATURE}), CLIMATE_WINDOW)
    temperature = np.full(count, DEFAULT_TEMPERATURE) if temp is None else np.clip(temp, 0.0, 1.0)
    bright = np.clip(
        BRIGHT_DAYLIGHT_SHARE * daylight
        + BRIGHT_TEMPERATURE_SHARE * temperature
        + BRIGHT_ACTIVITY_SHARE * activity,
        0.0,
        1.0,
    )
    bright = _smooth(bright, smooth)
    hum = _combine(series, house, frozenset({VoiceKind.HUMIDITY}), CLIMATE_WINDOW)
    humid = (
        np.full(count, DEFAULT_HUMIDITY) if hum is None else _smooth(np.clip(hum, 0.0, 1.0), smooth)
    )
    return power, daylight, bright, humid


def _light_curves(
    series: Sequence[Series], house: np.ndarray, hour: np.ndarray, smooth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lights and the evening of the piece."""
    evening = np.asarray(
        smoothstep((hour - EVENING_RISE_HOUR) / EVENING_RISE_HOURS)
        * (1.0 - smoothstep((hour - EVENING_FALL_HOUR) / EVENING_FALL_HOURS))
    )
    light_series = [item for item in series if item.kind == VoiceKind.CHORD]
    if light_series:
        lit = np.mean([item.at(house, LIGHT_WINDOW) for item in light_series], axis=0)
        lights = np.clip(LIGHTS_SWITCHED_SHARE * lit + LIGHTS_EVENING_SHARE * evening, 0.0, 1.0)
    else:
        lights = evening
    return _smooth(lights, smooth), evening


def _run_curves(series: Sequence[Series], house: np.ndarray, smooth: int) -> dict[str, np.ndarray]:
    """Return a curve for each appliance that runs for a while: 1 while it works, with rounded edges."""
    runs: dict[str, np.ndarray] = {}
    for name, kinds in RUN_KINDS.items():
        found = [item for item in series if item.kind in kinds]
        if not found:
            runs[name] = np.zeros(len(house))
            continue
        level = np.max([item.at(house, RUN_WINDOW) for item in found], axis=0)
        threshold = RUN_THRESHOLD if name != RUN_COOK else COOK_THRESHOLD
        # a run is on while the appliance works; the edges are rounded, a bar or two long
        runs[name] = _smooth(
            np.asarray(smoothstep((level - threshold) / RUN_RAMP)), max(smooth // 2, 1)
        )
    return runs


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return the rank of each value from 0 to 1, equal values sharing the mean of their ranks."""
    values = np.round(values, RANK_DECIMALS)
    ordered = np.sort(values)
    low = np.searchsorted(ordered, values, side="left")
    high = np.searchsorted(ordered, values, side="right") - 1
    return (low + high) / 2.0 / max(len(values) - 1, 1)
