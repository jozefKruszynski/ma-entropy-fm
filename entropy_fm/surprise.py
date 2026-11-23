"""
Surprise instead of raw values: what the house always does fades, what it does not is featured.

The ear habituates to a fridge. So an entity whose episodes are usual at this hour of the day (see
`baseline`) is turned down towards the texture, and an episode that is not usual (a door at 03:00,
a washing machine on a day it never runs, a heatwave) comes forward: it is louder, it gets its own
leitmotif and a stranger beat. Everything is a pure function of the series and the baseline, so a
day always sounds the same.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .baseline import MIN_EVIDENCE_DAYS, episodes
from .timeline import DAY_SECONDS, SECONDS_PER_HOUR, VoiceKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from .baseline import EntityBaseline
    from .timeline import Series

# how far a fully familiar voice fades: its level is multiplied by 1 - ROUTINE_FADE
ROUTINE_FADE = 0.5
# how much louder an episode gets, at a surprise of 1
EPISODE_BOOST = 0.45
# an episode this unusual is featured: a leitmotif and a stranger beat
FEATURE_THRESHOLD = 0.55
MAX_FEATURED = 4
# the featured episodes are at least this far apart, in seconds of house time
MIN_SPACING = 1800.0
# a day mean this many standard deviations above the baseline is a heatwave (or a cold snap)
CLIMATE_Z = 1.8
MIN_SPREAD = 0.6
CURVE_STEP = 60.0
EDGE_SECONDS = 300.0
# the familiarity is read once every this many curve steps and held in between
FAMILIARITY_STEPS = 10
# the kinds of an anomaly
KIND_EPISODE = "episode"
KIND_CLIMATE = "climate"
# an entity that is active on few days is unusual on this share of every day it is active
DAY_SCORE_SHARE = 0.8
# a climate anomaly scores from this many standard deviations, over this span of them, and lasts this many hours
CLIMATE_SCORE_FROM = 1.5
CLIMATE_SCORE_SPAN = 2.0
CLIMATE_HOURS = 3
# an anomaly is featured when it starts this many seconds before the end of the track at the latest
FEATURE_END_MARGIN = 2.0


@dataclass(frozen=True)
class Anomaly:
    """An episode that is not usual."""

    entity_id: str
    # "episode" for a start that is unusual, "climate" for a day mean far from the baseline
    kind: str
    # seconds from midnight, in house time
    start: float
    end: float
    # how unusual, 0 to 1
    score: float


def episode_scores(series: Series, baseline: EntityBaseline) -> list[Anomaly]:
    """
    Score the episodes of an entity against its baseline.

    The score is how unusual the hour of the start is; an entity that is active on few days is
    unusual on every day it is active.

    :param series: The entity on the day.
    :param baseline: What it usually does.
    """
    found: list[Anomaly] = []
    last_start = -MIN_SPACING
    for start, end in episodes(series):
        # the movements of one burst count as one episode
        if start - last_start < MIN_SPACING / 2:
            continue
        last_start = start
        hour_score = 1.0 - baseline.familiarity(start / SECONDS_PER_HOUR)
        day_score = DAY_SCORE_SHARE * (1.0 - baseline.active_days)
        score = max(hour_score, day_score) * baseline.evidence
        found.append(
            Anomaly(series.entity_id, KIND_EPISODE, start, end, float(np.clip(score, 0, 1)))
        )
    return found


def climate_anomaly(series: Series, baseline: EntityBaseline) -> Anomaly | None:
    """
    Return an anomaly when the mean of a temperature sensor is far from its usual one.

    :param series: The sensor on the day.
    :param baseline: What it usually reads.
    """
    if series.kind != VoiceKind.TEMPERATURE or baseline.days < MIN_EVIDENCE_DAYS - 1:
        return None
    z = (float(np.mean(series.values)) - baseline.mean) / max(baseline.spread, MIN_SPREAD)
    if abs(z) < CLIMATE_Z:
        return None
    peak = float(series.times[int(np.argmax(np.abs(series.values - baseline.mean)))])
    score = float(np.clip((abs(z) - CLIMATE_SCORE_FROM) / CLIMATE_SCORE_SPAN, 0.0, 1.0))
    end = min(peak + CLIMATE_HOURS * SECONDS_PER_HOUR, DAY_SECONDS)
    return Anomaly(series.entity_id, KIND_CLIMATE, peak, end, score)


def find_anomalies(
    series: Sequence[Series], baselines: Mapping[str, EntityBaseline]
) -> list[Anomaly]:
    """
    Return the episodes of a day that are unusual enough to feature, the strongest first.

    :param series: The voices of the track.
    :param baselines: The baseline per entity id; an entity without one has no surprises.
    """
    found: list[Anomaly] = []
    for item in series:
        baseline = baselines.get(item.entity_id)
        if baseline is None:
            continue
        found.extend(a for a in episode_scores(item, baseline) if a.score >= FEATURE_THRESHOLD)
        if (climate := climate_anomaly(item, baseline)) is not None:
            found.append(climate)
    found.sort(key=lambda a: (-a.score, a.start, a.entity_id))
    chosen: list[Anomaly] = []
    for anomaly in found:
        if all(abs(anomaly.start - other.start) >= MIN_SPACING for other in chosen):
            chosen.append(anomaly)
        if len(chosen) == MAX_FEATURED:
            break
    return sorted(chosen, key=lambda a: a.start)


class Surprise:
    """Turns routine down and anomalies up, for the entities of one track."""

    def __init__(self, series: Sequence[Series], baselines: Mapping[str, EntityBaseline]) -> None:
        """
        Compare the series with the baseline.

        :param series: The voices of the track, on the day.
        :param baselines: What each entity usually does.
        """
        self.baselines = dict(baselines)
        self.anomalies = find_anomalies(series, baselines)
        minutes = np.arange(int(DAY_SECONDS / CURVE_STEP)) * CURVE_STEP / SECONDS_PER_HOUR
        self._curves: dict[str, np.ndarray] = {}
        for entity_id, baseline in self.baselines.items():
            familiar = np.array(
                [baseline.familiarity(hour) for hour in minutes[::FAMILIARITY_STEPS]]
            )
            fade = (
                1.0
                - ROUTINE_FADE
                * baseline.evidence
                * np.repeat(familiar, FAMILIARITY_STEPS)[: len(minutes)]
            )
            self._curves[entity_id] = fade
        for anomaly in self.anomalies:
            curve = self._curves.get(anomaly.entity_id)
            if curve is None:
                continue
            first = max(int((anomaly.start - EDGE_SECONDS) / CURVE_STEP), 0)
            last = int((anomaly.end + EDGE_SECONDS) / CURVE_STEP) + 1
            curve[first:last] = np.maximum(curve[first:last], 1.0 + EPISODE_BOOST * anomaly.score)

    def gain(self, entity_id: str, house: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Return the level multiplier of an entity at some moments of house time.

        :param entity_id: The entity.
        :param house: Seconds from midnight of the day, in house time.
        """
        times = np.asarray(house, dtype=np.float64)
        curve = self._curves.get(entity_id)
        if curve is None:
            return np.ones(times.shape)
        index = np.clip((times / CURVE_STEP).astype(int), 0, len(curve) - 1)
        return np.asarray(curve[index])

    def event_gain(self, entity_id: str, house: float) -> float:
        """
        Return the velocity multiplier of one event.

        :param entity_id: The entity.
        :param house: Seconds from midnight of the day, in house time.
        """
        return float(self.gain(entity_id, np.array([house]))[0])

    def featured(
        self, origin: float, compression: float, duration_s: float
    ) -> list[tuple[float, float, str]]:
        """
        Return the anomalies that fall inside a track, as audio time, score and entity.

        :param origin: Seconds of house time at the start of the track.
        :param compression: Seconds of house time per second of audio.
        :param duration_s: The length of the track.
        """
        inside = []
        for anomaly in self.anomalies:
            audio = (anomaly.start - origin) / compression
            if 0.0 <= audio < duration_s - FEATURE_END_MARGIN:
                inside.append((audio, anomaly.score, anomaly.entity_id))
        return inside
