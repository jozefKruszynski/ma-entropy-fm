"""
The plan of a soundscape: how long it is, how the day maps onto it, and the curves that the house steers.

A soundscape is one long, slowly evolving piece for one room on one day. The day (night, morning, daytime,
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

from .const import NOMINAL_SCAPE_SECONDS
from .day_analysis import DAY_SECONDS, DayAnalysis
from .score import CR
from .synth import Scale, smoothstep
from .timeline import BINARY_KINDS, Series, VoiceKind, stable_hash
from .variation import variant_of

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from .palette import DayPalette
    from .quality import Quality

T = TypeVar("T")
MIN_SECONDS = 600
MAX_SECONDS = 1320
# the length of a piece: the whole of the day's busyness, or a condensed one of a fixed length per style
FULL = "full"
CONDENSED = "condensed"
LENGTHS = (FULL, CONDENSED)
DEFAULT_LENGTH = FULL
# a condensed piece is as long as the style needs to say something: slow styles take 7 minutes, fast ones 5
CONDENSED_SECONDS = {
    "ambient": 420,
    "berlin_school": 420,
    "ritual": 420,
    "lofi": 300,
    "minimal": 300,
    "glitch": 300,
}
CONDENSED_DEFAULT_SECONDS = 300
# the activity and the lights of a condensed piece change over at least this many seconds, so that a build is a build
MIN_SMOOTH_SECONDS = 12.0
# in a condensed piece the house happens two to three times as fast: unusual moments keep this far apart,
# and the little things (a switch, a door) are thinned to one every few seconds, the rarer kinds first
CONDENSED_EVENT_GAP = 25.0
CONDENSED_TRIGGER_GAP = 6.0
TRIGGER_PRIORITY = {
    "appliance": 0,
    "hob": 0,
    "phase": 1,
    "door": 1,
    "cover": 2,
    "media": 2,
    "light": 3,
    "motion": 3,
    "blip": 4,
}
# the arc: where the five stretches of the day sit in the piece, as shares of its length
# how much of the activity of a day that is measured against its own days is the curve itself, and how much its rank
MEASURED_CURVE = 0.5
# showers: a couple at most in a piece, 60 seconds apart; with the anomalies at most four texture moments
MAX_SHOWERS = 2
MAX_EVENTS = 4
SHOWER_GAP = 60.0
ARC_NAMES = ("night", "morning", "daytime", "evening", "late night")
POWER_KINDS = frozenset({VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER})
RUN_KINDS = {
    "dish": frozenset({VoiceKind.DISHWASHER}),
    "wash": frozenset({VoiceKind.DRUM}),
    "cook": frozenset({VoiceKind.HOB, VoiceKind.PHASE}),
}


def condensed_seconds(style: str) -> int:
    """
    Return the length in seconds of the condensed piece of a style, which does not depend on the day.

    :param style: The style key.
    """
    return CONDENSED_SECONDS.get(style, CONDENSED_DEFAULT_SECONDS)


def nominal_seconds(style: str, length_mode: str) -> int:
    """
    Return the length a track is listed with before its day has been read.

    The full length is only known from the day (10 to 22 minutes), so it is listed with a typical one; a condensed
    piece is always the same length, so it is listed with that.

    :param style: The style key.
    :param length_mode: `full` or `condensed`.
    """
    return condensed_seconds(style) if length_mode == CONDENSED else NOMINAL_SCAPE_SECONDS


def length_of(analysis: DayAnalysis, style: str = "", length_mode: str = FULL) -> int:
    """
    Return the length of the piece in seconds.

    A full piece is about 12 minutes on a quiet day to 20 on a busy one; a condensed one is 5 minutes in the fast
    styles and 7 in the slow ones, whatever the day.

    :param analysis: The analysis of the day; busyness and the number of anomalies set the length.
    :param style: The style key, which sets the condensed length.
    :param length_mode: `full` or `condensed`.
    """
    if length_mode == CONDENSED:
        return condensed_seconds(style)
    unusual = sum(1 for moment in analysis.moments if moment.kind == "anomaly")
    seconds = 720.0 + 480.0 * analysis.busyness + 30.0 * min(unusual, 2)
    return int(min(max(round(seconds), MIN_SECONDS), MAX_SECONDS))


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Smooth a curve with a moving average of `width` values, keeping its ends."""
    if width <= 1 or len(values) < 3:
        return values
    pad = np.concatenate((np.full(width, values[0]), values, np.full(width, values[-1])))
    kernel = np.hanning(width * 2 + 1)
    return np.asarray(np.convolve(pad, kernel / kernel.sum(), mode="same")[width:-width])


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
    # every curve has CR values per second
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
    choices: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # things that happened in the house as (second of the piece, strength 0 to 1, kind), the kinds being door, motion,
    # light, appliance, phase, hob, cover, media and blip: a switch, a door, an appliance starting or stopping
    triggers: list[tuple[float, float, str]] = field(default_factory=list)
    # a condensed piece: the same day in a fraction of the time, so the composers keep it from getting crowded
    condensed: bool = False

    @property
    def count(self) -> int:
        """Return the number of values of each curve."""
        return len(self.hour)

    def rng(self, *parts: object) -> np.random.Generator:
        """Return a random generator that is the same every time for this piece and these parts."""
        return np.random.default_rng(stable_hash(self.seed, *parts) % (2**63))

    def deal(self, component: str, options: Sequence[T]) -> T:
        """
        Pick one of the options for a component of the day, so that two days in a row never get the same one.

        The days are dealt out in shuffled rounds of the whole family (see `variation.variant_of`), seeded
        from the style, the room and the component.

        :param component: What is being chosen, such as the motif of the sequence.
        :param options: The family to draw from; with a single option there is no choice.
        """
        if self.day is None:
            index = int(self.rng("deal", component).integers(0, len(options)))
        else:
            index = variant_of(f"{self.style}|{self.area_key}|{component}", self.day, len(options))
        chosen = options[index]
        label = (
            chosen[0]
            if isinstance(chosen, tuple) and chosen and isinstance(chosen[0], str)
            else chosen
        )
        self.choices[component] = str(getattr(label, "key", label))
        return chosen

    def at(self, curve: np.ndarray, second: float) -> float:
        """Return the value of a curve at a second of the piece."""
        position = min(max(second * CR, 0.0), len(curve) - 1.0)
        low = int(position)
        high = min(low + 1, len(curve) - 1)
        return float(curve[low] + (curve[high] - curve[low]) * (position - low))

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
        return np.asarray(smoothstep((curve - low) / max(high - low, 1e-6)))

    def second_of_day(self, second: float) -> float:
        """Return the second of the day that a second of the piece stands for."""
        return self.covered * second / self.length

    def piece_second_of_hour(self, hour: float) -> float:
        """Return the second of the piece at which the clock of the day says `hour`."""
        return float(
            np.clip(hour * 3600.0 / max(self.covered, 1.0) * self.length, 0.0, self.length)
        )

    def arc(self) -> list[tuple[str, float, float]]:
        """Return the five stretches of the day as (name, start second, end second) of the piece."""
        edges = [0.0, 6.0, 11.0, 17.0, 22.0, 24.0]
        marks = [self.piece_second_of_hour(h) for h in edges]
        return [(ARC_NAMES[i], marks[i], marks[i + 1]) for i in range(5) if marks[i + 1] > marks[i]]


def _daylight(hour: np.ndarray) -> np.ndarray:
    rise = smoothstep((hour - 5.5) / 3.0)
    fall = 1.0 - smoothstep((hour - 17.0) / 3.5)
    return np.asarray(rise * fall)


def _trigger_kind(item: Series) -> str:
    """Name what kind of happening a series stands for."""
    if item.is_opening:
        return "door"
    if item.is_motion:
        return "motion"
    return {
        VoiceKind.CHORD: "light",
        VoiceKind.DRUM: "appliance",
        VoiceKind.DISHWASHER: "appliance",
        VoiceKind.PHASE: "phase",
        VoiceKind.HOB: "hob",
        VoiceKind.COVER: "cover",
        VoiceKind.MEDIA: "media",
        VoiceKind.BLIP: "blip",
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
                    item.crossings(0.0, covered, 0.2, rising=True),
                    item.crossings(0.0, covered, 0.2, rising=False),
                )
            )
        else:
            times = item.changes(0.0, covered, 0.5 if item.kind in BINARY_KINDS else 0.02)
        found.extend(
            (float(t / covered * length), 0.5, kind)
            for t in times
            if 3.0 <= t / covered * length < length - 5.0
        )
    return sorted(found)


def _thin_events(events: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Keep the strongest unusual moments that are at least `gap` seconds apart, at most three, in order."""
    kept: list[tuple[float, float]] = []
    for second, strength in sorted(events, key=lambda item: (-item[1], item[0])):
        if all(abs(second - other) >= gap for other, _ in kept):
            kept.append((second, strength))
    return sorted(kept)[:3]


def _thin_triggers(
    triggers: list[tuple[float, float, str]], gap: float
) -> list[tuple[float, float, str]]:
    """Keep at most one switch, door or appliance every `gap` seconds: appliances and doors before lights and motion."""
    kept: list[tuple[float, float, str]] = []
    for item in sorted(triggers, key=lambda t: (TRIGGER_PRIORITY.get(t[2], 5), t[0])):
        if all(abs(item[0] - other[0]) >= gap for other in kept):
            kept.append(item)
    return sorted(kept)


def _with_showers(
    events: list[tuple[float, float]],
    analysis: DayAnalysis,
    covered: float,
    length: int,
    *,
    condensed: bool,
) -> list[tuple[float, float]]:
    """
    Add the showers (a spike of humidity well over what a room usually reads) to the unusual moments.

    A shower is a texture moment of its own, a little softer than an anomaly: at most `MAX_SHOWERS` of them, apart
    from each other and from the anomalies, and `MAX_EVENTS` moments in all, the strongest kept.

    :param events: The unusual moments as (second of the piece, strength).
    :param analysis: The analysis of the day, which found the showers.
    :param covered: The seconds of the day that happened.
    :param length: The length of the piece in seconds.
    :param condensed: Whether the piece is condensed.
    """
    gap = CONDENSED_EVENT_GAP if condensed else SHOWER_GAP
    found: list[tuple[float, float]] = []
    for moment in sorted(analysis.moments_of("shower"), key=lambda m: (-m.strength, m.time)):
        audio = moment.time / covered * length
        strength = float(np.clip(0.8 * moment.strength, 0.25, 0.85))
        if 4.0 <= audio < length - 6.0 and all(
            abs(audio - other) >= gap for other, _ in [*events, *found]
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
    *,
    condensed: bool,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, str]]]:
    """Return the unusual moments and the things that happened, as seconds of the piece; thinned when condensed."""
    events = []
    for moment in analysis.moments_of("anomaly"):
        audio = moment.time / covered * length
        if 4.0 <= audio < length - 6.0:
            events.append((float(audio), float(np.clip(moment.strength, 0.2, 1.0))))
    events = _thin_events(events, CONDENSED_EVENT_GAP) if condensed else sorted(events)[:3]
    events = _with_showers(events, analysis, covered, length, condensed=condensed)
    triggers = _triggers(series, covered, length)
    if condensed:
        triggers = _thin_triggers(triggers, CONDENSED_TRIGGER_GAP)
    return events, triggers


def build_plan(  # noqa: PLR0913, PLR0917
    series: Sequence[Series],
    analysis: DayAnalysis,
    style: str,
    seed: str,
    scale: Scale,
    quality: Quality,
    *,
    day: date | None = None,
    palette: DayPalette | None = None,
    energy_offset: float = 0.0,
    area_key: str = "",
    length_mode: str = FULL,
) -> Plan:
    """
    Make the plan of a soundscape from the data of a day.

    :param series: Every series of the room or the house and the sun, on the day.
    :param analysis: The analysis of the day (`day_analysis.analyze_day`).
    :param style: The style key.
    :param seed: Text that is the same for the same room, day and style.
    :param scale: The key and mode of the piece.
    :param quality: The tier of the engine.
    :param day: The day, which deals the material.
    :param palette: What the day derives (key, mode, density).
    :param energy_offset: The setting that moves the activity, -0.5 to 0.5.
    :param area_key: The room, which seeds the deals.
    :param length_mode: `full` (as long as the day was busy) or `condensed` (5 to 7 minutes, the same day faster).
    """
    condensed = length_mode == CONDENSED
    length = length_of(analysis, style, length_mode)
    count = length * CR
    t = (np.arange(count) + 0.5) / CR
    covered = float(min(max(analysis.covered, 3600.0), DAY_SECONDS))
    house = covered * t / length
    hour = house / 3600.0
    smooth = max(int(max(length / 40.0, MIN_SMOOTH_SECONDS) * CR / 2), 1)

    centres = (np.arange(len(analysis.curve)) + 0.5) * 900.0
    # the curve is stretched by its rank, so that a day with one big spike still has a quiet and a lively half
    ranks = np.argsort(np.argsort(analysis.curve)) / max(len(analysis.curve) - 1, 1)
    if analysis.measured:
        # measured against its own days the curve already says how lively this day was: the rank is a smaller part
        mixed = MEASURED_CURVE * analysis.curve**0.7 + (1.0 - MEASURED_CURVE) * ranks
    else:
        mixed = 0.5 * analysis.curve**0.6 + 0.5 * ranks
    activity = np.interp(house, centres, mixed)
    activity = np.clip(_smooth(activity, smooth) + energy_offset, 0.0, 1.0)

    def combine(kinds: frozenset[VoiceKind], window: float) -> np.ndarray | None:
        found = [item for item in series if item.kind in kinds and not item.is_baseline_load]
        if not found:
            return None
        return np.max([item.at(house, window) for item in found], axis=0)

    power_raw = combine(POWER_KINDS, 900.0)
    power = np.zeros(count) if power_raw is None else np.clip((power_raw - 0.5) / 0.4, 0.0, 1.0)
    power = _smooth(power, smooth)

    daylight = _daylight(hour)
    sun = combine(frozenset({VoiceKind.SUN}), 900.0)
    if sun is not None:
        daylight = 0.5 * daylight + 0.5 * np.clip(sun, 0.0, 1.0)
    temp = combine(frozenset({VoiceKind.TEMPERATURE}), 1800.0)
    temperature = np.full(count, 0.5) if temp is None else np.clip(temp, 0.0, 1.0)
    bright = np.clip(0.5 * daylight + 0.3 * temperature + 0.2 * activity, 0.0, 1.0)
    bright = _smooth(bright, smooth)
    hum = combine(frozenset({VoiceKind.HUMIDITY}), 1800.0)
    humid = np.full(count, 0.4) if hum is None else _smooth(np.clip(hum, 0.0, 1.0), smooth)

    evening = np.asarray(smoothstep((hour - 16.5) / 3.0) * (1.0 - smoothstep((hour - 23.0) / 1.0)))
    light_series = [item for item in series if item.kind == VoiceKind.CHORD]
    if light_series:
        lit = np.mean([item.at(house, 600.0) for item in light_series], axis=0)
        lights = np.clip(0.6 * lit + 0.4 * evening, 0.0, 1.0)
    else:
        lights = evening
    lights = _smooth(lights, smooth)

    runs: dict[str, np.ndarray] = {}
    for name, kinds in RUN_KINDS.items():
        found = [item for item in series if item.kind in kinds]
        if not found:
            runs[name] = np.zeros(count)
            continue
        level = np.max([item.at(house, 240.0) for item in found], axis=0)
        threshold = 0.12 if name != "cook" else 0.05
        # a run is on while the appliance works; the edges are rounded, a bar or two long
        runs[name] = _smooth(
            np.asarray(smoothstep((level - threshold) / 0.18)), max(smooth // 2, 1)
        )

    events, triggers = _events_and_triggers(series, analysis, covered, length, condensed=condensed)

    return Plan(
        style=style,
        seed=seed,
        length=length,
        covered=covered,
        scale=scale,
        quality=quality,
        analysis=analysis,
        day=day,
        hour=hour,
        activity=activity,
        power=power,
        bright=bright,
        humid=humid,
        lights=lights,
        evening=evening,
        daylight=daylight,
        night=1.0 - np.maximum(daylight, evening * 0.6),
        runs=runs,
        events=events,
        palette=palette,
        area_key=area_key,
        triggers=triggers,
        condensed=condensed,
    )
