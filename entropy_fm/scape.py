"""
Make the soundscape of a day: analyse the day, plan the piece, let the style's generative system write its score.

This is what the provider and the demos call; the pieces are in `day_analysis` (the day), `plan` (the curves
and the deals) and the composers of the styles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import scape_ambient, scape_berlin, scape_glitch, scape_lofi, scape_minimal, scape_ritual
from .day_analysis import DAY_SECONDS, DayAnalysis, analyze_day
from .levels import feed_forward, trim_db
from .plan import FULL, Plan, build_plan, length_of
from .quality import STANDARD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from .palette import DayPalette
    from .quality import Quality
    from .score import Score
    from .synth import Scale
    from .timeline import Series

SCAPE_ID = "_scape"
LIGHT_MAX_LAYERS = 4
COMPOSERS: dict[str, Callable[[Plan], Score]] = {
    "ambient": scape_ambient.compose,
    "berlin_school": scape_berlin.compose,
    "lofi": scape_lofi.compose,
    "minimal": scape_minimal.compose,
    "ritual": scape_ritual.compose,
    "glitch": scape_glitch.compose,
}


def analysis_of(
    series: Sequence[Series],
    covered: float = DAY_SECONDS,
    anomalies: Sequence[tuple[float, float, str]] = (),
) -> DayAnalysis:
    """
    Analyse a day for its soundscapes; the analysis does not depend on the style.

    :param series: Every series of the day (the whole area or house, and the sun).
    :param covered: How many seconds of the day have happened.
    :param anomalies: Unusual events as (second of the day, score, entity).
    """
    return analyze_day(series, covered, anomalies)


def duration_of(analysis: DayAnalysis, style_key: str = "", length_mode: str = FULL) -> int:
    """
    Return the length of the soundscape of a day in seconds: 10 to 22 minutes, or 5 to 7 when condensed.

    :param analysis: The analysis of the day.
    :param style_key: The style, which sets the condensed length.
    :param length_mode: `full` or `condensed`.
    """
    return length_of(analysis, style_key, length_mode)


def scape_score(  # noqa: PLR0913
    series: Sequence[Series],
    analysis: DayAnalysis,
    style_key: str,
    seed: str,
    scale: Scale,
    *,
    day: date | None = None,
    palette: DayPalette | None = None,
    quality: Quality = STANDARD,
    energy_offset: float = 0.0,
    area_key: str = "",
    length_mode: str = FULL,
) -> tuple[Score, Plan]:
    """
    Return the score of the soundscape of a day in a style, and the plan it was made from.

    :param series: Every series of the day.
    :param analysis: The analysis of the day.
    :param style_key: The style.
    :param seed: Text that is the same for the same room, day and style; seeds everything that is chosen.
    :param scale: The key and mode the user chose (the palette of the day moves away from it).
    :param day: The day, which deals the material so that two days in a row differ.
    :param palette: What the day derives: key, mode, tempo shift.
    :param quality: The tier.
    :param energy_offset: The setting that moves the activity.
    :param area_key: The room (or the whole house), which seeds the deals.
    :param length_mode: `full` or `condensed`.
    """
    plan = build_plan(
        series,
        analysis,
        style_key,
        seed,
        scale,
        quality,
        day=day,
        palette=palette,
        energy_offset=energy_offset,
        area_key=area_key,
        length_mode=length_mode,
    )
    score = COMPOSERS[style_key](plan)
    for key, layer in score.layers.items():
        layer.gain *= 10.0 ** (trim_db(style_key, key) / 20.0)
    _spare_filters(score, quality)
    if score.glitch is None:
        score.gain_db = feed_forward(score, score.level_db)
    return score, plan


# the most buses whose low-pass the cheapest tier runs: a ladder filter costs about three milliseconds of a second
LIGHT_FILTERS = 2


def _spare_filters(score: Score, quality: Quality) -> None:
    """In the cheapest tier keep the filter of the busiest buses only; the others play unfiltered."""
    if quality.max_layers > LIGHT_MAX_LAYERS:
        return
    count: dict[str, int] = {}
    for note in score.notes:
        count[note.layer] = count.get(note.layer, 0) + 1
    filtered = [key for key, layer in score.layers.items() if layer.cutoff is not None]
    keep = set(sorted(filtered, key=lambda key: -count.get(key, 0))[:LIGHT_FILTERS])
    for key in filtered:
        if key not in keep:
            score.layers[key].cutoff = None
