"""
Make the soundscape of a day: analyse the day, plan the piece, let the style's generative system write its score.

This is what the provider and the demos call; the pieces are in `day_analysis` (the day), `plan` (the curves
and the deals) and the composers of the styles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import scape_ambient, scape_berlin, scape_glitch, scape_lofi, scape_ritual
from .character import scale_of_day
from .day_analysis import DayAnalysis, analyze_day
from .levels import feed_forward, trim_db
from .plan import DEFAULT_OPTIONS, LIGHT_MAX_LAYERS, PieceOptions, Plan, build_plan
from .quality import STANDARD
from .signature import gain_of_db
from .styles import (
    STYLE_AMBIENT,
    STYLE_BERLIN_SCHOOL,
    STYLE_GLITCH,
    STYLE_LOFI,
    STYLE_RITUAL,
    get_style,
)
from .timeline import DAY_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .quality import Quality
    from .score import Score
    from .synth import Scale
    from .timeline import Series

SCAPE_ID = "_scape"
# the most buses whose low-pass the cheapest tier runs: a ladder filter costs about three milliseconds of a second
LIGHT_FILTERS = 2
COMPOSERS: dict[str, Callable[[Plan], Score]] = {
    STYLE_AMBIENT: scape_ambient.compose,
    STYLE_BERLIN_SCHOOL: scape_berlin.compose,
    STYLE_LOFI: scape_lofi.compose,
    STYLE_RITUAL: scape_ritual.compose,
    STYLE_GLITCH: scape_glitch.compose,
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


def scape_score(
    series: Sequence[Series],
    analysis: DayAnalysis,
    style_key: str,
    seed: str,
    scale: Scale,
    *,
    quality: Quality = STANDARD,
    options: PieceOptions = DEFAULT_OPTIONS,
) -> tuple[Score, Plan]:
    """
    Return the score of the soundscape of a day in a style, and the plan it was made from.

    :param series: Every series of the day.
    :param analysis: The analysis of the day.
    :param style_key: The style.
    :param seed: Text that is the same for the same room, day and style; seeds everything that is chosen.
    :param scale: The key and mode the user chose (the palette of the day moves away from it).
    :param quality: The tier.
    :param options: The day, palette, room, length mode and character of the piece; the character
        picks the key and mode of the day and the identity of the piece.
    """
    if options.character is not None and options.day is not None:
        scale = scale_of_day(get_style(style_key), scale, options.day, options.character)
    plan = build_plan(series, analysis, style_key, seed, scale, quality, options=options)
    score = COMPOSERS[style_key](plan)
    for key, layer in score.layers.items():
        layer.gain *= gain_of_db(trim_db(style_key, key))
    _spare_filters(score, quality)
    if score.glitch is None:
        score.gain_db = feed_forward(score, score.level_db)
    return score, plan


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
