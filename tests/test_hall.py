"""Tests of the long hall that a layer can ask for: only those layers are sent to it, and the cheapest tier skips it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from entropy_fm.quality import LIGHT, RICH, STANDARD, Quality
from entropy_fm.render import make_renderer, render_all
from entropy_fm.score import CURVE_RATE, Layer, Note, Score

RATE = 44100
LENGTH = 16
NOTE_AT = 1.0
# seconds after the note, further out than the shared room of the tier lasts
TAIL_FROM = 5.0
TAIL_TO = 9.0
HALL_SEND = 2.0


def _score(hall: float, *, on_layer: str = "a", other: str = "b") -> Score:
    """Two layers, a note on `on_layer`; only the layer `a` can ask for the hall."""
    layers = {
        "a": Layer("a", 1.0, None, None, 0.0, reverb=0.1, hall=hall, title="a"),
        other: Layer(other, 1.0, None, None, 0.0, reverb=0.1, title=other),
    }
    notes = [Note(NOTE_AT, 0.5, 60.0, 0.8, "felt_piano", on_layer)]
    return Score(LENGTH, notes, layers, space=np.full(LENGTH * CURVE_RATE + 1, 1.0))


def _tail_energy(score: Score, tier: Quality) -> float:
    audio = render_all(score, tier, "hall")
    tail = audio[int((NOTE_AT + TAIL_FROM) * RATE) : int((NOTE_AT + TAIL_TO) * RATE)]
    return float(np.mean(tail.astype(np.float64) ** 2))


@pytest.mark.parametrize("tier", [STANDARD, RICH])
def test_the_hall_is_only_for_the_layer_that_asks(tier: Quality) -> None:
    """A layer with a send to the hall rings on long after its note; the same note on a layer without one does not."""
    asking = _tail_energy(_score(HALL_SEND), tier)
    plain = _tail_energy(_score(0.0), tier)
    other = _tail_energy(_score(HALL_SEND, on_layer="b"), tier)
    assert asking > 10.0 * plain
    assert other == pytest.approx(plain, rel=1e-3)


def test_the_hall_is_only_made_where_a_layer_that_plays_asks_for_it() -> None:
    """No hall is run for a score whose layers do not ask for it, or whose asking layer has no notes."""
    assert make_renderer(_score(HALL_SEND), STANDARD).hall is not None
    assert make_renderer(_score(0.0), STANDARD).hall is None
    assert make_renderer(_score(HALL_SEND, on_layer="b"), STANDARD).hall is None


def test_the_lightest_tier_skips_the_hall_and_sends_to_the_shared_reverb() -> None:
    """On the Light tier there is no hall, and the send of the layer goes to the shared reverb instead."""
    assert make_renderer(_score(HALL_SEND), LIGHT).hall is None
    near = int((NOTE_AT + 1.0) * RATE)
    asking = render_all(_score(HALL_SEND), LIGHT, "hall")[near : near + RATE]
    plain = render_all(_score(0.0), LIGHT, "hall")[near : near + RATE]
    assert np.isfinite(asking).all()
    assert float(np.mean(asking**2)) > float(np.mean(plain**2))


def test_the_hall_is_not_run_while_nothing_is_sent_to_it_and_its_tail_is_gone() -> None:
    """The hall runs from the first send until its tail has died away, and not in the seconds before or after."""
    renderer = make_renderer(_score(HALL_SEND), STANDARD)
    assert renderer.hall is not None
    real = renderer.hall
    runs: list[int] = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        runs.append(1)
        return real(*args, **kwargs)

    renderer.hall = counted  # type: ignore[assignment]
    for block in range(LENGTH):
        renderer.render_block(block)
    assert 0 < len(runs) < LENGTH - 1
