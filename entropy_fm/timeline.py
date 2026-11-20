"""
Turn Home Assistant history into control signals.

This module is pure: it imports nothing from Music Assistant, so it can be tested on its own and
used from a standalone script. A day of history for one entity becomes a `Series`, a step
function with a normalised value between 0 and 1 that the synth reads at any moment.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def as_array(value: Any) -> npt.NDArray[np.float64]:
    """
    Return a value as an array of floats; this keeps the type checker sure of what it is.

    :param value: Anything numpy can turn into an array.
    """
    return np.asarray(value, dtype=np.float64)


class VoiceKind(StrEnum):
    """The musical role an entity plays."""

    DRONE = "drone"
    DRUM = "drum"
    DISHWASHER = "dishwasher"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PLUCK = "pluck"
    CHORD = "chord"
    SUN = "sun"
    PAD_PARTIAL = "pad_partial"
    BLIP = "blip"
    # an enum sensor or a select: every change of state is an event (operation state, program phase)
    PHASE = "phase"
    # the progress of a program, 0 to 100 percent: a slow build parameter
    PROGRESS = "progress"
    # the power level of one zone of a hob: a note on a degree of the scale
    HOB = "hob"
    # a media player: playing or not
    MEDIA = "media"
    # a cover, a blind or a window shutter: open or closed
    COVER = "cover"
    # an air quality monitor (CO2 in ppm): how many people are in the room, and when someone comes in
    AIR = "air"


# kinds whose value is 0 or 1
BINARY_KINDS = frozenset(
    {VoiceKind.PLUCK, VoiceKind.CHORD, VoiceKind.BLIP, VoiceKind.MEDIA, VoiceKind.COVER}
)
# kinds whose every change is a musical event, whatever the direction
EVENT_KINDS = frozenset({VoiceKind.PHASE, VoiceKind.HOB, VoiceKind.COVER, VoiceKind.MEDIA})
HOB_LEVELS = 9.0
PHASE_CODES = 8
# what a sensor says when it is doing nothing
IDLE_STATES = frozenset(
    {
        "inactive",
        "off",
        "idle",
        "ready",
        "standby",
        "finished",
        "none",
        "power_level_off",
        "stopped",
    }
)
SKIPPED_STATES = frozenset({"unknown", "unavailable", ""})
POWER_LEVEL = re.compile(r"power_level_?(\d+)")
# sensors and binary sensors that say nothing about the music: how the device is, not what it does
DIAGNOSTIC_WORDS = (
    "connectivity",
    "signal_strength",
    "rssi",
    "linkquality",
    "firmware",
    "update",
    "version",
    "wifi",
    "ip_address",
    "uptime",
    "esphome",
    "voltage",
    "battery",
)
# domains that never make a voice, whatever the settings
ALWAYS_EXCLUDED_DOMAINS = frozenset({"lock", "alarm_control_panel", "camera"})
# domains that say who is home; excluded unless the user includes presence
PRESENCE_DOMAINS = frozenset({"person", "device_tracker"})
SECRET_WORDS = re.compile(r"password|code")
# sensors that say nothing musical
IGNORED_DEVICE_CLASSES = frozenset(
    {
        "timestamp",
        "date",
        "enum",
        "battery",
        "signal_strength",
        "duration",
        "data_size",
        "data_rate",
        "update",
        "connectivity",
        "problem",
        "plug",
        "running",
        "tamper",
        "firmware",
        "voltage",
        "current",
        "frequency",
        "power_factor",
    }
)
# the binary sensors that are about the device, not the house
IGNORED_BINARY_CLASSES = frozenset(
    {"connectivity", "problem", "update", "tamper", "battery", "battery_charging", "firmware"}
)
PLUCK_DEVICE_CLASSES = frozenset(
    {"motion", "occupancy", "door", "window", "opening", "garage_door", "vibration", "moving"}
)
OPENING_DEVICE_CLASSES = frozenset({"door", "window", "opening", "garage_door"})
MOTION_DEVICE_CLASSES = frozenset({"motion", "occupancy", "presence", "moving"})
# last resort for a sensor without a device class, in a few languages
OPENING_WORDS = ("door", "window", "gate", "tür", "tuer", "fenster", "deur", "raam")
# roles of power sensors that cycle all day: what the house always draws, never something that happens
BASELINE_ROLES = frozenset({"fridge"})
ROLE_TEXT = re.compile(r"role: ([a-z_]+)")
SUN_ELEVATION_ENTITY = "sensor.sun_solar_elevation"
# the sun entity of Home Assistant only reports above or below the horizon in its history
SUN_ENTITY = "sun.sun"
MIN_POWER_PEAK = 1.0
MIN_TEMPERATURE_SPAN = 3.0
ON_STATES = frozenset(
    {"on", "home", "open", "opening", "above_horizon", "true", "playing", "detected", "buffering"}
)
OFF_STATES = frozenset(
    {
        "off",
        "not_home",
        "closed",
        "closing",
        "below_horizon",
        "false",
        "clear",
        "idle",
        "paused",
        "standby",
        "stopped",
    }
)


def stable_hash(*parts: object) -> int:
    """
    Return a 64 bit hash of the parts that is the same in every process.

    :param parts: The values to hash; their text is joined.
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Reference:
    """
    What an entity read on its own previous days: the scale that its values are measured against.

    It is built from the baseline (`baseline.EntityBaseline.reference`) and is what makes every room its own
    measure: a CO2 monitor that reads 400 to 900 is as lively on a busy day as one that reads 400 to 3000.
    """

    # how many days it is built from
    days: int
    # the 5th, 50th and 95th percentile of the values (weighted by how long each lasted)
    low: float
    mid: float
    high: float
    # the highest 98th percentile of a single day: the level of the loudest run of an appliance
    peak: float
    # for a binary entity: the hours it was on and the number of times it came on, in a day
    on_hours: float
    on_count: float
    # nothing ever changed in all those days
    flat: bool


@dataclass(frozen=True)
class EntityInfo:
    """An entity that can play a part in the music."""

    entity_id: str
    name: str
    area_id: str | None
    kind: VoiceKind
    # a meter that only counts up: its rate is what plays
    cumulative: bool = False
    # why the entity has its voice, such as "role: dishwasher (label)"
    why: str = ""
    # the device class from the attributes, such as "door" or "motion"
    device_class: str = ""
    # the device of the entity, which tells what belongs together
    device_id: str = ""

    @property
    def domain(self) -> str:
        """Return the Home Assistant domain of the entity."""
        return self.entity_id.split(".", 1)[0]


def is_excluded(
    entity_id: str, extra_globs: Iterable[str] = (), *, include_presence: bool = False
) -> bool:
    """
    Return True when an entity must never be turned into sound.

    :param entity_id: The entity id.
    :param extra_globs: Patterns of the user, matched against the entity id.
    :param include_presence: Allow persons and device trackers.
    """
    lowered = entity_id.lower()
    domain = lowered.split(".", 1)[0]
    if domain in ALWAYS_EXCLUDED_DOMAINS:
        return True
    if domain in PRESENCE_DOMAINS and not include_presence:
        return True
    if SECRET_WORDS.search(lowered):
        return True
    return any(
        fnmatch.fnmatch(lowered, glob.strip().lower()) for glob in extra_globs if glob.strip()
    )


def is_number(value: str | None) -> bool:
    """
    Return True when the text is a finite number.

    :param value: The text.
    """
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except ValueError:
        return False


def state_to_value(state: str) -> float | None:
    """
    Turn a Home Assistant state into a number, or None when it says nothing.

    :param state: The state text, such as "on", "21.5" or "unavailable".
    """
    lowered = state.strip().lower()
    if lowered in ON_STATES:
        return 1.0
    if lowered in OFF_STATES:
        return 0.0
    try:
        value = float(lowered)
    except ValueError:
        return category_value(lowered)
    return value if np.isfinite(value) else None


def category_value(state: str) -> float | None:
    """
    Turn the text state of an enum sensor into a number that changes when the state does.

    A hob says `power_level_5`, which is 5. The states that mean nothing is going on are 0, the
    others are 1 to 7, the same for the same text every time.

    :param state: The lower case state.
    """
    if state in SKIPPED_STATES:
        return None
    if (match := POWER_LEVEL.fullmatch(state)) is not None:
        return float(match.group(1))
    if state in IDLE_STATES:
        return 0.0
    return float(1 + stable_hash("category", state) % (PHASE_CODES - 1))


def thin_numeric(
    times: np.ndarray, values: np.ndarray, min_dt: float = 20.0, rel: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop points of a numeric series that add nothing audible.

    A point stays when it is at least `min_dt` seconds after the last kept one, or when it
    differs from it by more than `rel` of the range of the series.

    :param times: Ascending times in seconds.
    :param values: The values.
    :param min_dt: The shortest gap between kept points, in seconds.
    :param rel: A change, as a fraction of the range, that always stays.
    """
    if len(times) < 3:
        return times, values
    span = float(np.ptp(values)) or 1.0
    keep = [0]
    for index in range(1, len(times) - 1):
        last = keep[-1]
        if times[index] - times[last] >= min_dt or abs(values[index] - values[last]) > rel * span:
            keep.append(index)
    keep.append(len(times) - 1)
    return times[keep], values[keep]


def normalise(  # noqa: PLR0911 - one scaling per kind
    kind: VoiceKind, values: np.ndarray
) -> np.ndarray:
    """
    Scale raw values to between 0 and 1 in the way that suits the kind of voice.

    :param kind: The voice kind.
    :param values: The raw values of one day.
    """
    if kind in BINARY_KINDS or kind == VoiceKind.SUN:
        return np.clip(values, 0.0, 1.0)
    if kind in {VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER}:
        peak = float(np.percentile(values, 98))
        if peak < MIN_POWER_PEAK:
            return np.zeros_like(values)
        return np.clip(values / peak, 0.0, 1.0)
    if kind in {VoiceKind.HUMIDITY, VoiceKind.PROGRESS}:
        return np.clip(values / 100.0, 0.0, 1.0)
    if kind == VoiceKind.PHASE:
        return np.clip(values / PHASE_CODES, 0.0, 1.0)
    if kind == VoiceKind.HOB:
        return np.clip(values / HOB_LEVELS, 0.0, 1.0)
    if kind == VoiceKind.TEMPERATURE:
        low, high = float(values.min()), float(values.max())
        span = max(high - low, MIN_TEMPERATURE_SPAN)
        return np.clip((values - (low + high) / 2) / span + 0.5, 0.0, 1.0)
    low, high = (float(np.percentile(values, 2)), float(np.percentile(values, 98)))
    if high - low < 1e-9:
        return np.full_like(values, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


@dataclass
class Series:
    """One entity over a stretch of time, as a step function."""

    entity_id: str
    name: str
    kind: VoiceKind
    # seconds of house time since the start of the day, ascending, the first one is 0
    times: np.ndarray
    # the raw values at those times
    values: np.ndarray
    # the same values normalised between 0 and 1
    norm: np.ndarray
    device_class: str = ""
    device: str = ""
    # event times that rate limiting and merging have taken away, see `condense`
    suppressed: frozenset[float] = frozenset()
    # the role of the entity ("fridge", "washer", ...), which `build_series` reads from why the entity plays
    role: str = ""
    # what the entity read on the days before, when they are known (see `references.rescale`)
    ref: Reference | None = None

    def __post_init__(self) -> None:
        """Prepare the running integral that makes smoothing exact."""
        steps = np.diff(self.times)
        self._integral = np.concatenate(([0.0], np.cumsum(self.norm[:-1] * steps)))

    @property
    def is_opening(self) -> bool:
        """Return True for a door, a window or a gate: what styles turn into gongs and fills."""
        if self.device_class in OPENING_DEVICE_CLASSES:
            return True
        if self.kind != VoiceKind.PLUCK or self.device_class in MOTION_DEVICE_CLASSES:
            return False
        text = f"{self.entity_id} {self.name}".lower()
        return any(word in text for word in OPENING_WORDS)

    @property
    def is_baseline_load(self) -> bool:
        """Return True for a load that runs all day by itself (a fridge): it is the floor, never activity."""
        return self.role in BASELINE_ROLES

    @property
    def is_motion(self) -> bool:
        """Return True for a sensor that says someone is moving about the house."""
        if self.kind != VoiceKind.PLUCK or self.is_opening:
            return False
        return self.device_class in MOTION_DEVICE_CLASSES or not self.device_class

    @property
    def activity(self) -> float:
        """Return how much happens in the series, to rank voices."""
        if self.kind in BINARY_KINDS:
            return float(np.count_nonzero(np.diff(self.norm) > 0.5))
        return float(np.ptp(self.norm)) * (1 + min(len(self.times), 400) / 400)

    def onsets(self, start: float, end: float) -> np.ndarray:
        """
        Return the times in [start, end) where the value goes from off to on.

        :param start: First second of house time.
        :param end: End of the stretch, exclusive.
        """
        rising = np.flatnonzero((self.norm[1:] > 0.5) & (self.norm[:-1] <= 0.5)) + 1
        return self._within(self.times[rising], start, end)

    def _within(self, found: np.ndarray, start: float, end: float) -> np.ndarray:
        """Return the times of a stretch that rate limiting has not taken away."""
        found = found[(found >= start) & (found < end)]
        if self.suppressed:
            found = found[[float(t) not in self.suppressed for t in found]]
        return found

    def changes(self, start: float, end: float, minimum: float = 0.02) -> np.ndarray:
        """
        Return the times in [start, end) where the value changes by more than `minimum`.

        :param start: First second of house time.
        :param end: End of the stretch, exclusive.
        :param minimum: The smallest change of the normalised value that counts.
        """
        moved = np.flatnonzero(np.abs(np.diff(self.norm)) > minimum) + 1
        return self._within(self.times[moved], start, end)

    def crossings(self, start: float, end: float, threshold: float, *, rising: bool) -> np.ndarray:
        """
        Return the times in [start, end) where the normalised value crosses a threshold.

        :param start: First second of house time.
        :param end: End of the stretch, exclusive.
        :param threshold: The level, between 0 and 1.
        :param rising: True for crossings upwards, False for downwards.
        """
        above = self.norm > threshold
        flips = np.flatnonzero(above[1:] != above[:-1]) + 1
        flips = flips[above[flips]] if rising else flips[~above[flips]]
        return self._within(self.times[flips], start, end)

    def _integral_at(self, time: np.ndarray) -> np.ndarray:
        """Return the integral of the step function from 0 to each time."""
        index = np.clip(np.searchsorted(self.times, time, side="right") - 1, 0, len(self.times) - 1)
        result = self._integral[index] + self.norm[index] * (time - self.times[index])
        # before the start the first value is assumed to hold
        return np.where(time < 0, self.norm[0] * time, result)

    def at(self, time: np.ndarray, smooth: float = 0.0) -> np.ndarray:
        """
        Return the normalised value at the given times of house time.

        :param time: Seconds since the start of the day.
        :param smooth: Width in seconds of a moving average that rounds off the steps.
        """
        time = np.asarray(time, dtype=np.float64)
        if smooth <= 0:
            index = np.clip(np.searchsorted(self.times, time, side="right") - 1, 0, None)
            return self.norm[index]
        return as_array((self._integral_at(time) - self._integral_at(time - smooth)) / smooth)


def build_series(info: EntityInfo, times: Sequence[float], values: Sequence[float]) -> Series:
    """
    Make a series from the raw history of an entity.

    :param info: The entity.
    :param times: Seconds since the start of the day of each change; may start after 0.
    :param values: The value after each change.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if info.cumulative and len(t) > 1:
        # a meter counts up; what plays is how fast. A reset gives a negative step: zero.
        dt = np.maximum(np.diff(t), 1.0)
        rate = np.maximum(np.diff(v), 0.0) / dt
        t, v = t[1:], rate
    if len(t) == 0:
        t, v = np.array([0.0]), np.array([0.0])
    if t[0] > 0:
        before = 0.0 if info.kind in BINARY_KINDS and v[0] > 0.5 else v[0]
        t, v = np.concatenate(([0.0], t)), np.concatenate(([before], v))
    else:
        t = t.copy()
        t[0] = 0.0
    if info.kind not in BINARY_KINDS and info.kind != VoiceKind.SUN:
        t, v = thin_numeric(t, v)
    if info.kind in BINARY_KINDS:
        v = (v > 0.5).astype(np.float64)
    elif info.entity_id == SUN_ELEVATION_ENTITY:
        # an elevation in degrees becomes daylight
        v = np.clip((v + 6.0) / 18.0, 0.0, 1.0)
    found = ROLE_TEXT.match(info.why)
    return Series(
        info.entity_id,
        info.name,
        info.kind,
        t,
        v,
        normalise(info.kind, v),
        info.device_class,
        info.device_id,
        role=found.group(1) if found else "",
    )


def raw_to_json(
    info: EntityInfo, times: Sequence[float], values: Sequence[float]
) -> dict[str, Any]:
    """
    Return the raw history of an entity in the form the cache stores, thinned.

    :param info: The entity.
    :param times: Seconds since the start of the day.
    :param values: The values.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(t) > 2:
        # the rate of a meter needs its points, so a meter is thinned by time only
        meter = info.cumulative
        t, v = thin_numeric(
            t, v, min_dt=60.0 if meter else 10.0, rel=float("inf") if meter else 0.02
        )
    return {"t": np.round(t, 2).tolist(), "v": np.round(v, 4).tolist()}


def series_from_json(info: EntityInfo, data: dict[str, Any]) -> Series:
    """
    Rebuild a series from what `raw_to_json` stored.

    :param info: The entity.
    :param data: The stored data.
    """
    return build_series(info, data["t"], data["v"])


DEFAULT_LIMITS: dict[VoiceKind, int] = {
    VoiceKind.DRONE: 6,
    VoiceKind.DRUM: 3,
    VoiceKind.DISHWASHER: 2,
    VoiceKind.TEMPERATURE: 3,
    VoiceKind.HUMIDITY: 3,
    VoiceKind.PLUCK: 10,
    VoiceKind.CHORD: 12,
    VoiceKind.SUN: 1,
    VoiceKind.PAD_PARTIAL: 4,
    VoiceKind.BLIP: 6,
    VoiceKind.PHASE: 6,
    VoiceKind.PROGRESS: 2,
    VoiceKind.HOB: 9,
    VoiceKind.MEDIA: 2,
    VoiceKind.COVER: 3,
    VoiceKind.AIR: 4,
}


def select_voices(
    series: Iterable[Series], limits: dict[VoiceKind, int] | None = None
) -> list[Series]:
    """
    Keep the most active voices of each kind, so a whole house stays clear and cheap to render.

    :param series: All candidate series.
    :param limits: The most voices per kind.
    """
    limits = limits or DEFAULT_LIMITS
    chosen: list[Series] = []
    for kind in VoiceKind:
        of_kind = sorted(
            (item for item in series if item.kind == kind),
            key=lambda item: (-item.activity, item.entity_id),
        )
        chosen.extend(of_kind[: limits.get(kind, 4)])
    return chosen


def parse_history(
    raw: dict[str, Any], day_start_ts: float, day_length: float
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Read the result of Home Assistant's `history/history_during_period` command.

    With `minimal_response` an entry is `{"s": state, "lu": epoch seconds}`, the first one of an
    entity carries the state at the start of the period. Unknown states are skipped.

    :param raw: The command result, entity id to a list of entries.
    :param day_start_ts: The epoch seconds of the start of the day.
    :param day_length: The length of the day in seconds, the last time that is kept.
    """
    result: dict[str, tuple[list[float], list[float]]] = {}
    for entity_id, entries in raw.items():
        times: list[float] = []
        values: list[float] = []
        for entry in entries:
            state = entry.get("s", entry.get("state"))
            stamp = entry.get("lu", entry.get("lc"))
            if state is None or stamp is None:
                continue
            value = state_to_value(str(state))
            if value is None:
                continue
            seconds = min(max(float(stamp) - day_start_ts, 0.0), day_length)
            times.append(seconds)
            values.append(value)
        result[entity_id] = (times, values)
    return result


# the fewest seconds of house time between two events of one entity: at most six a minute
MIN_EVENT_GAP = 10.0
# changes of one device, or of lights and switches together, that are this close are one gesture
MERGE_WINDOW = 8.0
# kinds whose events are rate limited and merged
LIMITED_KINDS = frozenset(
    {
        VoiceKind.PLUCK,
        VoiceKind.BLIP,
        VoiceKind.CHORD,
        VoiceKind.PHASE,
        VoiceKind.COVER,
        VoiceKind.MEDIA,
    }
)


def _event_times(item: Series) -> np.ndarray:
    """Return when a series has something happen: a rise for a binary one, any change else."""
    if item.kind in {VoiceKind.PLUCK, VoiceKind.BLIP}:
        return item.onsets(-1.0, float("inf"))
    return item.changes(-1.0, float("inf"))


def condense(
    series: Sequence[Series], min_gap: float = MIN_EVENT_GAP, window: float = MERGE_WINDOW
) -> list[Series]:
    """
    Rate limit the events of the series and merge the ones that happen together.

    An entity gets at most one event every `min_gap` seconds. Events of the entities of one
    device, and of all the lights and switches that have no device of their own, that are within
    `window` seconds of each other are one gesture: only the first of them stays. A scene that
    toggles twenty lights at once is one event. The series are returned as copies that remember
    which of their times are suppressed; their values are not touched.

    :param series: The candidates.
    :param min_gap: The shortest time between two events of one entity.
    :param window: The width of a merge.
    """
    taken: dict[str, float] = {}
    events: list[tuple[float, str, int]] = []
    for index, item in enumerate(series):
        if item.kind not in LIMITED_KINDS:
            continue
        group = item.device or ("lights" if item.kind == VoiceKind.CHORD else item.entity_id)
        events.extend((float(t), group, index) for t in _event_times(item))
    suppressed: dict[int, set[float]] = {}
    last_of_entity: dict[int, float] = {}
    for time, group, index in sorted(events):
        entity_last = last_of_entity.get(index, -1e9)
        merged = time - taken.get(group, -1e9) < window and group != series[index].entity_id
        if time - entity_last < min_gap or merged:
            suppressed.setdefault(index, set()).add(time)
            continue
        taken[group] = time
        last_of_entity[index] = time
    return [
        replace(item, suppressed=frozenset(suppressed.get(index, ())))
        if index in suppressed
        else item
        for index, item in enumerate(series)
    ]


def dedupe(
    series: Sequence[Series], devices: dict[str, str] | None = None
) -> tuple[list[Series], list[tuple[str, str]]]:
    """
    Drop the sensors that say the same thing twice.

    Two power sensors of one device, or two series with the same values at the same times (a plug
    and the current power of the same machine), play once: the one with more activity stays.

    :param series: The candidates.
    :param devices: Entity id to device id, for the series that have no device of their own.
    :return: The kept series and the pairs `(dropped, kept)`.
    """
    devices = devices or {}
    kept: list[Series] = []
    dropped: list[tuple[str, str]] = []
    seen: dict[tuple[object, ...], Series] = {}
    for item in sorted(series, key=lambda item: (-item.activity, item.entity_id)):
        if item.kind in {VoiceKind.SUN, VoiceKind.PHASE, VoiceKind.HOB}:
            kept.append(item)
            continue
        device = item.device or devices.get(item.entity_id, "")
        shape = (
            round(len(item.times), 0),
            np.round(item.times, 0).tobytes(),
            np.round(item.norm, 2).tobytes(),
        )
        keys: list[tuple[object, ...]] = [("same", item.kind, *shape)]
        if device and item.kind in {VoiceKind.DRONE, VoiceKind.DRUM, VoiceKind.DISHWASHER}:
            keys.append(("device", item.kind, device))
        twin = next((seen[key] for key in keys if key in seen), None)
        if twin is not None and item.activity > 0:
            dropped.append((item.entity_id, twin.entity_id))
            continue
        for key in keys:
            seen[key] = item
        kept.append(item)
    order = {item.entity_id: index for index, item in enumerate(series)}
    return sorted(kept, key=lambda item: order[item.entity_id]), dropped
